"""The radar's core: the list of objects, kept over time by model-based filtering.

One cycle per scan (a sensor's message, `process`):

1. every object is brought to the scan's time: the hypotheses interact (models.mix) and each predicts forward
   (models.predict); what the car's own movement in that interval is not sure about is added to where the
   object is believed to be;
2. the scan's detections are compared with what each object predicts for this sensor at the car's pose of that
   moment (sensors.py): a gate, then one global assignment of detections to objects;
3. a detection updates each hypothesis of its object, as far as the pairing is believed when all objects and
   detections of the scan are weighed together (`marginals`); how well each hypothesis had predicted it sets
   how much that hypothesis is believed from now on (the IMM's model probabilities: "it stands" against "it
   drives"), and how well the object as a whole had predicted it feeds the object's existence and its record;
4. an object the scan should have seen and did not loses existence; a detection nobody claims may start an
   object, tentative until later scans agree with what it predicts.

Checked against later frames, explicitly: besides the innovation of every update, each object writes down
where it expects to be 0.5 s and 1 s ahead; when that time has passed, the expectation is compared with where
the object turned out to be (`consistency`, about 1 when the predictions hold within their own stated
uncertainty; `pred_err_m`). An object that keeps being mispredicted has its uncertainty opened.

What a new object may be doing is three stories, not one blur: it stands; it drives with the traffic (about
the car's own velocity); it comes the other way; a vehicle whose box shows it side-on crosses (`process`).
Beside the hypotheses every object keeps a free estimate (`Obj.xf`): a steady-velocity filter fed the same
detections and never told that things stand or drive with the traffic. It is what the sightings alone have
shown. "It stands" hands it to the moving hypotheses when they take over, and the verdict "moving" waits
until it shows a speed.

Objects have a memory: a trajectory on the ground (`history`), the class vote, hits and age, the sensors that
have seen them and when. Their life: tentative -> confirmed -> coast (inside a sensor's field, not reported for
the moment) -> gone when the sensors that should see it keep not seeing it. An object that has LEFT every
sensor's field is not gone: nothing says it stopped existing. It is kept as `memory`, carried on by its own
motion hypotheses and the car's movement, for as long as where it is remains worth knowing (`memory_*`: a
standing car stays put for a long time, a moving one's place blurs within seconds), and it is the same object,
with the same id, when a sensor sees it again (another camera, a radar looking sideways or back).
"""
import math
from collections import deque

import numpy as np

from ..core import classvote
from . import models as M
from .models import CA, CV, N, PX, PY, STAND, VX, VY
from .sensors import kind_code

_NR = M.NR                       # the native core: the tracker's arithmetic (chris_native.c, the radar part)

CFG = dict(
    gate={1: 9.0, 2: 13.8, 3: 16.3, 4: 18.5},          # chi-square gates (99.7 %) by measurement dimension: the first sieve
    new_rate_gain=0.1,           # how fast the count of unexplained detections per scan is followed
    # where detections that no object explained have been turning up lately (a map of them, in the sensor's
    # measurement space): there a detection is more probably one more of those than a known object's
    map_s=5.0, map_bw=(math.radians(0.4), math.radians(1.5), 0.4),     # memory; kernel widths: angle, bearing, log height
    # the box as a feature: an object's box does not jump from where the object predicts it by more than a
    # part of its own width; another object's box is anywhere near
    feat_sigma=0.25, feat_same=0.9, feat_span=3.0, feat_max_s=1.0,
    with_rel=0.3, with_std=3.0,  # "drives with the traffic": the car's own speed within this share of it + this (m/s)
    new_exist=0.2,               # existence of an object started by one detection
    size_from_class=True,        # the height prior follows the class vote (`_size`); False: the first box's label's
    # the detector's own confidence in a box: how much likelier a box of that score is a real thing than a
    # false one, learnt from the boxes themselves (the scores of boxes that went into confirmed objects against
    # those of objects that never were confirmed); bins over the score, what is assumed before anything is
    # learnt (mean, spread of a true and of a false box's score, weight in boxes), and how much one box says
    score_bins=10, score_prior=((0.62, 0.14), (0.50, 0.14), 30.0), score_gain=0.6, score_max=1.2, score_keep=3000.0,
    confirm_exist=0.95, confirm_hits=3, delete_exist=0.04,
    max_objects=64,
    split_rel=0.12,              # a hypothesis whose bearing is unsure by this much (rad) is not linearised in one
                                 # piece (docs/RADAR.md, section 5); 0: always in one piece
    # the hypotheses (chris_native.c `Hyp`; docs/RADAR.md). Each is a knob of cost against quality, the default
    # the full setting: how many scans back a doubtful pairing stays open (0: decided at once, the best
    # assignment of each scan), how many global hypotheses a cluster of competing objects keeps, how many leaves
    # an object, how much less probable than the best a hypothesis may be (log), the largest cluster whose
    # assignments are enumerated (a larger one gets the best assignment per hypothesis only)
    hyp_n_scan=5, hyp_k_best=32, hyp_max_leaves=12, hyp_prune_ln=math.log(1e4), hyp_max_cluster=16,
    hyp_keep_steps=40, hyp_keep_s=2.0,   # each story's history kept this far back (the smoother's reach)
    rows=1024,                   # of the state array at the start (it grows as needed)
    # the past, smoothed (docs/RADAR.md, section 6): a backward pass over each story after every sighting, as far
    # back as smooth_s (the whole trajectory once per smooth_hist_s; 0: at every sighting); the velocity that is
    # put out as `vs` and judged is the one smooth_lag_s ago
    smooth=True, smooth_s=2.0, smooth_lag_s=0.5, smooth_hist_s=0.5,
    # structure (docs/RADAR.md, section 7): rows of standing things, queues, platoons. The free room between two
    # members (standing / driving), how alike two drivers' velocities are, how straight the line is (m + share
    # of its length), how near the car's own line a queue's passes, members needed (a queue: two)
    groups=True, group_gap_stand_m=7.5, group_gap_move_m=30.0, group_dv=2.5, group_line_m=1.2, group_line_rel=0.03,
    group_lane_m=1.6, group_min=3, group_move_mps=3.0, group_still_m2=4.0,
    # what a row or a queue says about a member, once per group_s: it stands on the others' line to group_sigma_m,
    # and "it stands" is group_gain times likelier
    group_support=True, group_s=0.5, group_sigma_m=0.8, group_gain=2.0,
    # the road's axis as a standing vehicle's heading prior (`road_prior`, docs/RADAR.md, section 4): once per
    # road_prior_s, for a vehicle within road_prior_off_m of the road model's path (and no further than
    # road_prior_beyond_m past its end), to road_prior_std, where its heading is within road_prior_snap of the axis
    # (road_prior_mirror: the heading's mirror about the line of sight is taken when it lies this near the axis)
    road_prior=True, road_prior_s=1.0, road_prior_std=0.1, road_prior_snap=0.7, road_prior_mirror=0.25,
    road_prior_off_m=12.0, road_prior_beyond_m=10.0,
    # what a sensor can see: an object behind another one, or cut by the edge of the field, is not expected to be
    # reported (docs/RADAR.md, section 3). occlusion False: every object in the field counts as in the open. The
    # chance of a report falls from `occl_from` of the silhouette hidden to its lowest at `occl_full`
    occlusion=True, occl_from=0.25, occl_full=0.65, occl_floor=0.03, occl_floor_radar=0.3,
    hidden_s=12.0, hidden_std_m=30.0,   # hidden behind another object: kept this long, while its place is known this well
    oosm_max_s=0.35,             # a scan older than an object's time by more than this does not touch it
    history_s=6.0, history_n=120,
    check_horizons=(0.5, 1.0),   # the object writes down where it will be this far ahead
    check_window_s=0.45,         # ... and is checked by the first update within this time after it
    inconsistent=3.5, inflate=1.4,
    # verdict "stationary" / "moving"
    stand_p=0.85, move_p=0.10, release_p=0.5, verdict_hits=3, verdict_s=0.0,
    move_m2=9.0,                 # "moving" is said once the sightings by themselves show a speed (shown_moving)
    side_m=2.5, p_any_side=0.5,  # a new object this far beside the car's own line: as likely doing "anything
                                 # else" (oncoming, crossing) as driving with the traffic
    verdict_hits_velocity=2, verdict_s_velocity=0.2,    # with a sensor that measures speed
    merge_m=2.5, merge_rel=0.12, merge_s=0.6, merge_dv=2.0,
    merge_d2=9.2,                # ... and only within what both know of their places (squared, 2 dof: 99 %)
    # life
    view_gap_s=0.7,              # no scan has covered the object's place for this long: it is out of view
    seen_max_s=5.0,              # inside a field: dropped after this long unseen whatever its existence says
    memory_s=45.0,               # out of view: kept at most this long
    memory_std_m=10.0,           # ... and only while its place is known this well (1 sigma, larger axis)
    memory_range_m=150.0,        # ... and it is this near
    memory_tentative_s=1.0,      # an object never confirmed is not worth remembering
    mu_hold_hits=2,              # the hypotheses are weighed from this many sightings on (velocity initialised)
    mu_floor=0.01,               # no hypothesis is ever ruled out for good
    # a moving hypothesis does not learn the object's height (lh) and angle offset (dp) from a camera's boxes: they
    # are consider parameters there (their uncertainty in the covariance, their values held; docs/RADAR.md, section
    # 2), from the sighting on at which the hypotheses are weighed (mu_hold_hits), unless a sensor that measures range
    # has seen the object within consider_range_s. consider_stand: the standing hypothesis too (measured, not taken:
    # docs/RADAR.md, section 2)
    consider_moving=True, consider_range_s=2.0, consider_stand=False,
    common_min=3, common_pair=0.8,   # the scan's common pitch: from at least this many settled objects, each paired this surely
    bias_ripe=0.5,               # an object speaks about the sensor's bias once its own angle offset is known
                                 # this much better (share of the variance it was born with)
    bias_seen_s=1.0,
)


def _npsum(a):
    """sum(a) added up as numpy adds a contiguous array (pairwise, eight at a time): the same float."""
    n = len(a)
    if n < 8:
        r = 0.0
        for v in a:
            r += v
        return r
    if n <= 128:
        r = list(a[:8])
        i = 8
        while i < n - n % 8:
            for k in range(8):
                r[k] += a[i + k]
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for v in a[i:]:
            res += v
        return 0.0 + res
    n2 = n // 2
    n2 -= n2 % 8
    return _npsum(a[:n2]) + _npsum(a[n2:])


def marginals(ratio):
    """Probability of each pairing (object, detection) when all objects and detections of the scan are
    considered together: every detection is at most one object's, every object has at most one detection.
    ratio[i, j]: odds of "detection j is object i's" against "i was missed and j is something else", for the
    pair alone. Two objects that both fit one detection cannot both have it; the pairing's probability says
    so, where the pair's own odds do not. (Loopy belief propagation on the assignment graph: Williams & Lau
    2014; converges for this problem, a few sweeps; 30 sweeps at most, until the messages move less than 1e-4.)
    The tracker's scans do this inside the native association (radar_scan)."""
    L = np.ascontiguousarray(ratio, float)
    out = np.empty(L.shape)
    _NR.radar_marginals(L, out)
    return out


# An object's numbers live in rows of the tracker's state array (`State`): one row per node of its tree of
# hypotheses (chris_native.c, "the radar's hypotheses"). A row holds the motion hypotheses (xs, Ps), their
# probabilities (mu), the free estimate (xf, Pf), the combination (x, P), the extent (e: length, width, heading;
# Pe), the step that led into the node (st) and the node's own record (m: `M_*`). The object's attributes are
# views of its PRIMARY row: its leaf in the best hypothesis of its cluster, which the native scan names.
_LAY = (("xs", (3, N)), ("Ps", (3, N, N)), ("mu", (3,)), ("xf", (N,)), ("Pf", (N, N)), ("x", (N,)), ("P", (N, N)),
        ("e", (3,)), ("Pe", (3, 3)), ("st", (13,)), ("m", (20,)))
_OFF = []
_o = 0
for _k, _sh in _LAY:
    _OFF.append(("_" + _k, _o, _o + int(np.prod(_sh)), _sh))
    _o += int(np.prod(_sh))
ROW = _o                         # numbers per row
OFF = {k[1:]: (a, b) for k, a, b, _ in _OFF}
# a node's record (chris_native.c `M_*`): its time, the log odds that the object exists in this story, sightings,
# the last sighting's time, the last time a detection said how far it is, its last box (angular width, time,
# sensor number), the last update's normalised innovation and existence gain, the detection it took at its
# scan (-1: none), the scan's number, how much of it the sensor could see, its probability of being reported,
# the scan's sensor, what hides it (0: nothing, 1: another object, 2: the edge of the field, 3: outside the
# field), the sensor's coverage at its place, the id of what hides it, 1 once its motion has shown which end is
# the front
(M_T, M_L, M_HITS, M_THIT, M_RANGE_T, M_BOXW, M_BOXT, M_BOXS, M_NIS, M_DL, M_DET, M_LEVEL, M_VIS, M_PD, M_SRC, M_OCC,
 M_COV, M_OCCBY, M_SENSE, M_HOLD) = range(20)


class State:
    """The rows of the objects' numbers: an array (capacity, ROW). The native hypotheses (`Hyp`) say which rows are
    in use; a State by itself (tests, tools) hands out rows in order."""

    def __init__(self, cap=96):
        self.a = np.zeros((cap, ROW))
        self.free = list(range(cap - 1, -1, -1))

    def grow(self, cap):
        a = np.zeros((cap, ROW))
        a[:len(self.a)] = self.a
        self.a = a

    def take(self, live=()):
        return self.free.pop()

    def give(self, slot):
        self.free.append(slot)


def _view(name):
    def get(self):
        return getattr(self, "_" + name)

    def put(self, v):
        getattr(self, "_" + name)[...] = v
    return property(get, put)


def _meta(k, kind=float):
    def get(self):
        return kind(self._m[k])

    def put(self, v):
        self._m[k] = v
    return property(get, put)


class Obj:
    __slots__ = ("id", "h", "slot", "_xs", "_Ps", "_mu", "_xf", "_Pf", "_x", "_P", "_e", "_Pe", "_st", "_m", "t0", "state",
                 "sources", "last_seen", "vote", "history", "checks", "consistency", "pred_err", "n_checks", "nis",
                 "verdict", "coast_s", "velocity_seen", "merge_t", "flags", "last_cover", "dp0", "scores", "cls", "kin",
                 "store", "pe", "pp", "nleaf", "in_map", "label_code", "hidden_t", "vs", "fs", "ps_lag", "grp", "grp_kind", "hist_t",
                 "lhp")
    xs, Ps, mu, xf, Pf, x, P, e, Pe, st, m = (_view(k) for k, _ in _LAY)
    t = _meta(M_T)                   # the time its numbers are of
    logodds = _meta(M_L)             # that it exists, in the primary's story (`exist`: over all stories)
    hits = _meta(M_HITS, int)
    range_t = _meta(M_RANGE_T)       # the last time a detection said how far it is (not a bearing alone)

    def __init__(self, oid, h, slot, t, src, coast_s, store):
        self.id = oid
        self.h = h                   # its handle in the native hypotheses
        self.slot = slot
        self.bind(store)
        self.t0 = t
        self.state = "tentative"
        self.sources = {src}
        self.last_seen = {src: t}
        self.vote = {}
        self.history = deque(maxlen=CFG["history_n"])
        self.checks = []
        self.consistency = None
        self.pred_err = None
        self.n_checks = 0
        self.nis = {}
        self.verdict = "unknown"
        self.coast_s = coast_s
        self.velocity_seen = False
        self.merge_t = {}
        self.flags = set()
        self.last_cover = t          # the last time a scan's field covered where the object is
        self.dp0 = float(self._P[M.DP, M.DP])   # how unknown its angle offset was at birth
        self.scores = []             # (sensor, score) of its boxes while it is not confirmed
        self.cls = None              # classvote.result of `vote` without its age (box.py), until the next vote
        self.kin = M._KIN4[None]     # what its kind can do (models._KIN4 of `label`), set with every vote
        self.pe = self.pp = 0.0      # probability that it exists; that the primary's story is the true one
        self.nleaf = 1               # its hypotheses
        self.in_map = True           # it is in the best hypothesis of its cluster
        self.label_code = 0.0
        self.hidden_t = -1e18        # the last time a scan could not see it for another object or the field's edge
        self.vs = None               # its velocity a lag ago, smoothed: (vx, vy, 1 sigma x, y, how long ago), ground frame
        self.fs = None               # the free estimate's smoothed velocity and its covariance (vx, vy, 00, 01, 11)
        self.ps_lag = None           # the smoothed probability of "it stands" at the lag
        self.hist_t = -1e18          # when its trajectory was last smoothed as a whole
        self.grp = None              # the group it is of (id), and the group's kind: row | queue | platoon
        self.grp_kind = None
        self.lhp = None              # the prior of its height (lh) its numbers hold: (mean, variance) (Tracker._size)

    def bind(self, store, slot=None):
        """Point the object's arrays at its (primary) row of the state array."""
        self.store = store
        if slot is not None:
            self.slot = slot
        row = store.a[self.slot]
        for k, a, b, sh in _OFF:
            setattr(self, k, row[a:b].reshape(sh))

    @property
    def exist(self):
        return self.pe

    @property
    def label(self):
        return self.vote.get("shown") if self.vote else None

    def combine(self):
        """x, P from the hypotheses and their probabilities (after a change to them)."""
        _NR.radar_combine_rows(self.store.a, [self.slot])

    @property
    def speed(self):
        return math.hypot(self.x[VX], self.x[VY])


class Tracker:
    def __init__(self, ego, cfg=None, model_cfg=None):
        self.ego = ego
        self.cfg = dict(CFG, **(cfg or {}))
        self.mcfg = dict(M.CFG, **(model_cfg or {}))
        self.next_id = 1
        self.reset()

    def reset(self):
        c = self.cfg
        self.objects = []
        self.store = State(c["rows"])    # the objects' numbers: a row per node of their trees
        self.hyp = _NR.Hyp(c["rows"], c["hyp_n_scan"], c["hyp_k_best"], c["hyp_max_leaves"], c["hyp_prune_ln"],
                           c["hyp_keep_steps"], c["hyp_keep_s"], c["hyp_max_cluster"])
        self.by_h = {}               # handle -> object
        self.groups = {}             # group id -> {"kind", "members" (object ids, in order along the line), "line", "rms"}
        self.group_t = None
        self.next_group = 1
        self.src_code = {}           # sensor id -> its number in the rows' records
        self.stats = {}              # sensor id -> counters
        self.sensors = {}            # sensor id -> (last model, time)
        self.bias = {}               # sensor id -> [bias, variance]: what all its objects share (a camera's pitch)
        self.new_rate = {}           # sensor id -> detections per scan no object explained (followed over time)
        self.unexplained = {}        # sensor id -> deque of (t, kind, z): the map's memory
        self.scan_times = {}         # sensor id -> deque of scan times within the map's memory
        self.real_ids = set()        # objects that were confirmed: only their first detections make the map
        self.score_hist = {}         # sensor id -> [true counts, false counts] over the score bins
        self.score_odds = {}         # sensor id -> the log odds of each bin, as far as asked since the last change
        self.bias_t = None
        self.t = None
        self.road_t = None

    def peek(self, o, t):
        """Where the object is believed to be at time t, without changing it: (x, P, mu) (for tools)."""
        x, P, mu = np.empty((1, N)), np.empty((1, N, N)), np.empty((1, 3))
        _NR.radar_peek(self.store.a, [o.slot], [float(t) - o.t], [o.kin], M._native_par(self.mcfg), x, P, mu)
        return x[0], P[0], mu[0]

    @staticmethod
    def shown_moving(o):
        """How far the object's sightings by themselves show that it moves: the distance of "no speed" from the
        free estimate of its velocity, in that estimate's own uncertainty (squared; 9 is one chance in a
        hundred for a thing that stands). The free estimate a lag ago, smoothed, where there is one (what the
        later sightings say of the velocity then is far firmer than the newest sighting's own)."""
        if o.fs is not None:
            vx, vy, s00, s01, s11 = o.fs
        else:
            v, S = o.xf[VX:VY + 1], o.Pf[VX:VY + 1, VX:VY + 1]
            vx, vy, s00, s01, s11 = v[0], v[1], S[0, 0], S[0, 1], S[1, 1]
        det = s00 * s11 - s01 * s01
        if det <= 0.0:
            return 0.0
        return float((s11 * vx * vx - 2.0 * s01 * vx * vy + s00 * vy * vy) / det)

    def _pose(self, t):
        p = self.ego.pose_at(t)
        if p is None:
            return None
        v = self.ego.velocity_at(t)
        return (p[0], p[1], p[2], v[0], v[1], v[2])

    # ---- the detector's confidence ------------------------------------------------------------------------
    @staticmethod
    def _score(d):
        at = d.get("attrs") or {}
        v = at.get("class_conf", at.get("prob"))
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0 else None

    def _hist(self, src):
        h = self.score_hist.get(src)
        if h is None:
            c = self.cfg
            n = c["score_bins"]
            mid = (np.arange(n) + 0.5) / n
            (mt, st), (mf, sf), wgt = c["score_prior"]
            ht, hf = np.exp(-0.5 * ((mid - mt) / st) ** 2), np.exp(-0.5 * ((mid - mf) / sf) ** 2)
            h = self.score_hist[src] = [(wgt * ht / ht.sum()).tolist(), (wgt * hf / hf.sum()).tolist()]
        return h

    def _score_odds(self, src, sc):
        """Log odds a box of this score adds for "a real thing" (0 without a score). Per bin, kept until the
        sensor's histograms change (`_score_learn`)."""
        if sc is None:
            return 0.0
        c = self.cfg
        b = min(int(sc * c["score_bins"]), c["score_bins"] - 1)
        tab = self.score_odds.get(src)
        if tab is None:
            tab = self.score_odds[src] = [None] * c["score_bins"]
        v = tab[b]
        if v is None:
            ht, hf = self._hist(src)
            r = math.log(((ht[b] + 0.5) / (_npsum(ht) + 0.5 * len(ht))) / ((hf[b] + 0.5) / (_npsum(hf) + 0.5 * len(hf))))
            v = tab[b] = c["score_gain"] * max(-c["score_max"], min(c["score_max"], r))
        return v

    def _score_learn(self, scores, real):
        c = self.cfg
        for src, sc in scores:
            self.score_odds.pop(src, None)
            h = self._hist(src)[0 if real else 1]
            h[min(int(sc * c["score_bins"]), c["score_bins"] - 1)] += 1.0
            if _npsum(h) > c["score_keep"]:
                h[:] = [v * 0.5 for v in h]

    # ---- one scan -----------------------------------------------------------------------------------------
    def process(self, src, t, sensor, dets):
        """dets: [{"z": array, "R": variances (1-D), "attrs": {...}}]."""
        c = self.cfg
        st = self.stats.setdefault(src, {"scans": 0, "dets": 0, "assigned": 0, "started": 0, "late": 0})
        st["scans"] += 1
        if not getattr(sensor, "extended", None):
            st["dets"] += len(dets)
        self.sensors[src] = (sensor, t)
        b = None
        if getattr(sensor, "has_bias", False):
            # the camera's pitch error: its slow part (value, variance) and its fast part (value, variance, time)
            b = self.bias.setdefault(src, [0.0, sensor.bias_var, 0.0, 0.0, -1e18])
            sensor.bias, sensor.bias_var = b[0], b[1]
        pose = self._pose(t)
        if pose is None:
            return
        objs = []
        for o in self.objects:
            if t < o.t - c["oosm_max_s"]:
                st["late"] += 1
                continue
            objs.append(o)
        self.t = t if self.t is None else max(self.t, t)
        if getattr(sensor, "extraneous", None) is not None:
            sensor.extraneous = max(sensor.extraneous, self.new_rate.get(src, 0.0))
        code = self.src_code.setdefault(src, len(self.src_code) + 1)
        if b is not None:                                # how much the camera's pitch moves: by how fast the car drives
            cc = sensor.cfg
            f = min(math.hypot(pose[3], pose[4]) / cc["pitch_gm_full_mps"], 1.0)
            sensor.gm = (math.radians(cc["pitch_gm_stand_deg"] + f * (cc["pitch_gm_deg"] - cc["pitch_gm_stand_deg"])) ** 2,
                         cc["pitch_gm_tau_s"])
        if getattr(sensor, "extended", None):
            dets = self._cells(sensor, pose, t, objs, dets)
            st["dets"] += len(dets)
        D, KB = self._det_rows(src, dets)
        # what the car's own movement over the interval is not sure about goes into where each object is
        noise, info = {}, []
        for o in objs:
            ot = o.t
            nz = noise.get(ot)
            if nz is None:
                nz = noise[ot] = self.ego.motion_noise(ot, t)
            k = o.kin
            info.append((float(t) - ot, nz[0], nz[1], k[0], k[1], k[2], k[3], o.label_code, o.id,
                         1.0 if o.state in ("confirmed", "coast") else 0.0, self._consider(o, t)))
        mem = self.unexplained.get(src)
        rows = None
        if mem:
            real = self.real_ids
            rows = [(len(zz), zz[0], zz[1], zz[2] if len(zz) > 2 else 0.0, oid) for (_, _, zz, oid) in mem if oid in real] or None
        g, bw = c["gate"], c["map_bw"]
        acfg = (g.get(1, 16.0), g.get(2, 16.0), g.get(3, 16.0), g.get(4, 16.0), c["feat_sigma"], c["feat_same"],
                c["feat_span"], bw[0], bw[1], bw[2], c["feat_max_s"], c["common_min"], c["common_pair"],
                1.0 if c["occlusion"] else 0.0, c["occl_from"], c["occl_full"], c["occl_floor"], c["occl_floor_radar"],
                c["split_rel"])
        bcfg = (c["new_exist"], c["p_any_side"], c["side_m"], c["with_rel"], c["with_std"], self.mcfg["p_ca0"],
                c["mu_floor"], max(0, c["max_objects"] - len(self.objects)))
        args = ([o.h for o in objs], info, float(t), pose, sensor.native(), code, D, KB, rows,
                float(len(self.scan_times.get(src) or ())), acfg, M._native_par(self.mcfg),
                (c["mu_hold_hits"], c["mu_floor"]), bcfg, b)
        res = self.hyp.scan(self.store.a, *args)
        if isinstance(res, int):                         # the state array is too small for this scan: more rows
            self.store.grow(max(2 * len(self.store.a), len(self.store.a) + res + 64))
            for o in self.objects:
                o.bind(self.store)
            res = self.hyp.scan(self.store.a, *args)
        out, births, dead, bias = res
        if bias is not None:                             # the sensor's pitch error after the scan
            b[:] = bias
            sensor.bias, sensor.bias_var = bias[0], bias[1]
        store = self.store
        hit = []
        for o, (alive, prim, pd, cover, pe, pp, nleaf, in_map) in zip(objs, out):
            if not alive:
                continue
            o.bind(store, prim)
            o.pe, o.pp, o.nleaf, o.in_map = pe, pp, nleaf, bool(in_map)
            if cover > 0.0:
                o.last_cover = t
            if o._m[M_OCC] in (1.0, 2.0):                # hidden by an object or cut by the edge in this scan
                o.hidden_t = t
            j = int(o._m[M_DET])
            if j >= 0:
                o.last_cover = t
                self._after_update(o, dets[j], sensor, src, t)
                hit.append(o)
                st["assigned"] += 1
        if dead:
            gone = set(dead)
            for o in self.objects:
                if o.h in gone and self.by_h.get(o.h) is o:
                    del self.by_h[o.h]
                    if o.state == "tentative" and o.scores:
                        self._score_learn(o.scores, False)   # never confirmed and gone: its boxes were false ones
            self.objects = [o for o in self.objects if o.h not in gone]
            hit = [o for o in hit if o.h not in gone]
        if hit:                                          # written down now, checked when the time has come
            hz = c["check_horizons"]
            ahead = np.empty((len(hit) * len(hz), 6))    # where, and how well: (x, y, P00, P01, P10, P11)
            _NR.radar_peek_pos(store.a, [o.slot for o in hit for _ in hz], [t + h - o.t for o in hit for h in hz],
                               [o.kin for o in hit for _ in hz], M._native_par(self.mcfg), ahead)
            for k, r in enumerate(ahead.tolist()):
                hit[k // len(hz)].checks.append((t + hz[k % len(hz)], r[0], r[1], r[2:]))
        if c["smooth"]:
            for o in hit:
                self._smooth(o)
        memq = self.unexplained.setdefault(src, deque())
        times = self.scan_times.setdefault(src, deque())
        times.append(t)
        while times and times[0] < t - c["map_s"]:
            times.popleft()
        while memq and memq[0][0] < t - c["map_s"]:
            memq.popleft()
        for h, j, prim, pe, pp, in_map in births:
            d = dets[j]
            o = Obj(self.next_id, h, prim, t, src, sensor.coast_s, store)
            self.next_id += 1
            o.pe, o.pp, o.in_map = pe, pp, bool(in_map)
            sc = self._score(d)
            if sc is not None:
                o.scores.append((src, sc))
            o.velocity_seen = sensor.velocity
            kb = M.kinb(classvote.norm_label((d.get("attrs") or {}).get("class")))
            o.lhp = (math.log(kb[7]), kb[8] * kb[8])     # the height prior it was born with (chris_native.c r_birth)
            self._vote(o, d, src, t)
            o.history.append((t, float(o._x[PX]), float(o._x[PY]), 0.0, 0.0))
            kb = (d.get("kind") or "").rstrip("s")
            if kb in ("rb", "rbh"):
                # the map lives in the space of the ground angle, the bearing and the log height: a box with a
                # scale-rate row enters without it, a box cut at the bottom (its rows are another space) not at all
                memq.append((t, kb, np.asarray(d["z"], float)[:len(kb)], o.id))
            self.objects.append(o)
            self.by_h[h] = o
            st["started"] += 1
        r0 = self.new_rate.get(src, 0.0)
        self.new_rate[src] = r0 + c["new_rate_gain"] * (len(births) - r0)

    def _cells(self, sensor, pose, t, objs, returns):
        """The returns of a sensor that gives several per object, as detections: one per cell (radar_cells), the
        returns that lie on a known object being that object's cell."""
        eps_m, eps_v, margin = sensor.extended
        sr, sa, sv = sensor.sigma
        known = [o for o in objs if o.state != "tentative" or o.hits >= 2]
        cells = _NR.radar_cells(returns, eps_m, eps_v, self.store.a, [o.slot for o in known], [t - o.t for o in known],
                                sensor.native(), pose, margin) if returns else ()
        return [{"z": np.array([r, az, rr]), "R": np.array([sr * sr / n, sa * sa / n, sv * sv / n]), "attrs": {},
                 "cell": (float(n), sxx, sxy, syy)} for r, az, rr, n, sxx, sxy, syy in cells]

    def _smooth(self, o):
        """The object's past as its present story tells it, smoothed (Hyp.smooth: a backward pass over the story's
        nodes): its trajectory (`history`) and its velocity a lag ago (`vs`), from the detections that came after
        as well. The filtered state stays what is put out for now."""
        c = self.cfg
        # back to the lag after every sighting (the velocity that is judged and put out); the whole trajectory, as
        # far as smooth_s, once per smooth_hist_s (it costs four times the short pass with a 17 Hz sensor)
        whole = o.t - o.hist_t >= c["smooth_hist_s"]
        r = self.hyp.smooth(self.store.a, o.h, M._native_par(self.mcfg), c["hyp_keep_steps"],
                            c["smooth_s"] if whole else 1.5 * c["smooth_lag_s"], c["smooth_lag_s"])
        if r is None:
            return
        hist, o.vs, o.fs, o.ps_lag = r
        if not whole:
            return
        o.hist_t = o.t
        t_old = hist[-1][0]
        keep = [h for h in o.history if h[0] < t_old - 1e-6]     # older than the smoother reaches: as it was
        o.history.clear()
        o.history.extend(keep)
        o.history.extend(h[:5] for h in reversed(hist))

    def _det_rows(self, src, dets):
        """The scan's detections as the native core reads them (chris_native.c `RDET`: kind, scale-rate baseline,
        group (kind and baseline: what is predicted alike), vehicle (1) / person (-1) / not known (0), box width, z,
        the variances (R's diagonal), what the detector's score adds to the log odds of a real thing, the whole
        box's aspect) and what each detection's kind can do, for an object started by it (`RKB`)."""
        rows, kb = [], []
        kinds = {}
        pad = (0.0, 0.0, 0.0, 0.0)
        for j, d in enumerate(dets):
            k = d.get("kind")
            at = d.get("attrs") or {}
            lab = classvote.norm_label(at.get("class"))
            z, R = d["z"], d["R"]
            z = z.tolist() if isinstance(z, np.ndarray) else list(z)
            R = R.tolist() if isinstance(R, np.ndarray) else list(R)
            rows.append([kind_code(k), float(d.get("sdt") or 0.0), kinds.setdefault((k, d.get("sdt")), len(kinds)),
                         0 if lab is None else (1 if lab in M.VEHICLES else -1), float(d.get("w") or 0.0)]
                        + z + list(pad[len(z):]) + R + list(pad[len(R):])
                        + [self._score_odds(src, self._score(d)), float(at.get("aspect") or 0.0)]
                        + list(d.get("cell") or (1.0, 0.0, 0.0, 0.0)) + [1.0 if d.get("wok") else 0.0])
            kb.append(M.kinb(lab))
        return np.array(rows, float).reshape(len(dets), 20), np.array(kb, float).reshape(len(dets), 12)

    def _vote(self, o, d, src, t):
        at = d.get("attrs") or {}
        if at.get("class") is not None:
            classvote.vote(o.vote, at.get("class"), at.get("class_conf", at.get("prob")), src, t, at.get("weight", 1.0))
            o.cls = None
            o.kin = M._KIN4.get(o.label) or M._KIN4[None]
            la = o.label
            o.label_code = 0.0 if la is None else (1.0 if la in M.VEHICLES else -1.0)
            self._size(o)

    def _size(self, o):
        """The prior of the object's height follows what the class detector says of it (models.size_prior of the
        vote's tallies). An object is born with the prior of its first box's label; when the boxes that follow
        say another kind (a van is a `truck` in two boxes of three and a `car` in the third), the prior its numbers
        hold is replaced by the new one in every story of it: the exact Gaussian exchange (the old prior's
        information out, the new one's in) on lh, which moves the place through what lh and the place share (the
        box height's range). Without it a van born from a `car` box was read at the range a car of its box height
        would have: 0.6 of the truth, the box height being the strongest range beyond 40 m (docs/RADAR.md, section 4)."""
        if o.lhp is None or not o.vote or not o.vote.get("w") or not self.cfg["size_from_class"]:
            return
        m1, v1 = M.size_prior(o.vote["w"])
        m0, v0 = o.lhp
        if abs(m1 - m0) < 0.02 and abs(math.log(v1 / v0)) < 0.05:
            return
        a, b = 1.0 / v1 - 1.0 / v0, m1 / v1 - m0 / v0
        rows = self.hyp.rows(o.h)
        _NR.radar_lh_prior(self.store.a, rows, a, b)
        o.lhp = (m1, v1)

    def _after_update(self, o, d, sensor, src, t):
        """What a detection does to an object besides its numbers (those, its existence and its sightings are the
        native scan's): its record, the detector's score, its sensors, class vote, the checks of its predictions,
        its trajectory."""
        c = self.cfg
        n = float(o._m[M_NIS])
        if n == n:
            o.nis[src] = n if o.nis.get(src) is None else o.nis[src] + 0.2 * (n - o.nis[src])
        sc = self._score(d)
        if sc is not None:
            if o.state == "tentative":
                o.scores.append((src, sc))
            else:
                self._score_learn([(src, sc)], True)
        o.sources.add(src)
        o.last_seen[src] = t
        o.coast_s = max(o.coast_s, sensor.coast_s) if len(o.sources) == 1 else min(o.coast_s, sensor.coast_s)
        o.velocity_seen = o.velocity_seen or sensor.velocity
        self._vote(o, d, src, t)
        self._check(o, t)
        o.history.append((t, float(o.x[PX]), float(o.x[PY]), float(o.x[VX]), float(o.x[VY])))
        while o.history and o.history[0][0] < t - c["history_s"]:
            o.history.popleft()

    def _check(self, o, t):
        """Expectations whose time has passed against where the object is now believed to have been then."""
        c = self.cfg
        keep = []
        for (tf, px, py, Pp) in o.checks:
            if tf > t:
                keep.append((tf, px, py, Pp))
                continue
            if t - tf > c["check_window_s"]:
                continue                                 # no update near that time: nothing to check it with
            back = t - tf                                # the present estimate, taken back to that time
            # half the squared distance in both uncertainties (+ 0.05 m2 each way) and the distance; None: singular
            r = _NR.radar_check(self.store.a, o.slot, back, px, py, Pp)
            if r is None:
                continue
            m2, e = r
            o.n_checks += 1
            o.consistency = m2 if o.consistency is None else o.consistency + 0.25 * (m2 - o.consistency)
            o.pred_err = e if o.pred_err is None else o.pred_err + 0.25 * (e - o.pred_err)
        o.checks = keep[-8:]
        o.flags.discard("unpredictable")
        if o.n_checks >= 4 and o.consistency is not None and o.consistency > c["inconsistent"]:
            o.flags.add("unpredictable")                 # it keeps being somewhere else than predicted: believe less
            rows = self.hyp.rows(o.h)                   # every story of it
            a = self.store.a
            a[rows, OFF["Ps"][0]:OFF["Ps"][1]] *= c["inflate"]
            a[rows, OFF["Pf"][0]:OFF["Pf"][1]] *= c["inflate"]
            _NR.radar_combine_rows(a, rows)
            o.consistency = 0.5 * (o.consistency + 1.0)

    # ---- life ---------------------------------------------------------------------------------------------
    def manage(self, T):
        c = self.cfg
        keep = []
        pose = self.ego.pose_at(T)
        # "out of view" is what scans say that came and did not cover the object's place; while no scan comes
        # at all (frames dropped, the detector shed by the load governor) nothing has left any field
        scan = T if self.t is None else min(T, self.t)
        life = self._life(T, pose, scan)
        for o, (p_st, finite, _, sd, rng) in zip(self.objects, life):
            last = max(o.last_seen.values())
            if o.exist < c["delete_exist"] or not finite:
                continue
            if scan - o.last_cover <= c["view_gap_s"]:
                # a sensor is looking at where it should be: it lives by being seen
                # (the time it was hidden behind another object does not count: nothing was to be seen)
                hid = o.hidden_t > last
                if T - max(last, o.hidden_t) > max(o.coast_s, c["seen_max_s"]):
                    continue
                # (a member of a queue hidden by the queue itself is kept as long as a remembered object)
                if hid and (T - last > (c["memory_s"] if o.grp_kind == "queue" else c["hidden_s"])
                            or (sd is not None and sd > c["hidden_std_m"])):
                    continue                             # hidden for too long: where it is is no longer known
                if o.state == "tentative":
                    if o.exist >= c["confirm_exist"] and o.hits >= c["confirm_hits"] and o.in_map:
                        o.state = "confirmed"
                        self.real_ids.add(o.id)
                        self._score_learn(o.scores, True)
                        o.scores = []
                else:
                    o.state = "coast" if T - last > 0.3 else "confirmed"
            else:
                # out of every field: remembered while where it is remains worth knowing
                if o.state == "tentative":
                    if scan - o.last_cover > c["memory_tentative_s"]:
                        continue
                else:
                    if T - o.last_cover > c["memory_s"] or sd > c["memory_std_m"] or rng > c["memory_range_m"]:
                        continue
                    o.state = "memory"
            self._judge(o, T, p_st)
            keep.append(o)
        kept = {o.id for o in keep}
        for o in self.objects:                           # never confirmed and gone: its boxes were false ones
            if o.id not in kept and o.state == "tentative" and o.scores:
                self._score_learn(o.scores, False)
        gone = [o for o in self.objects if o.id not in kept]
        self.objects = keep
        for o in gone:
            self._drop(o)
        if len(self.real_ids) > 4096:
            self.real_ids = {o.id for o in keep if o.state != "tentative"}
        self._merge(T)
        self._register(T)
        self._groups(T, pose)

    def _groups(self, T, pose):
        """Structure among the objects (radar_groups): standing things in a line beside the road are a row, in the
        car's own lane a queue, things driving one behind the other at one speed a platoon. A group keeps its id
        while it keeps two of its members. A row or a queue supports its members (radar_group_support, once per
        `group_s`): each stands on the line the OTHERS make, and standing is likelier for it than for a thing by
        itself; a member whose own sightings show a speed is not of the group."""
        c = self.cfg
        objs = [o for o in self.objects if o.state != "tentative"]
        prev = {o.id: o.grp for o in objs if o.grp is not None}
        for o in self.objects:
            o.grp = o.grp_kind = None
        if not c["groups"] or len(objs) < 2 or pose is None:
            self.groups = {}
            return
        info = []
        for o in objs:
            sm = self.shown_moving(o)
            stands = o.verdict == "stationary" or (o.verdict != "moving" and sm < c["group_still_m2"] and o.mu[STAND] > 0.3)
            drives = o.verdict == "moving" and o.speed > c["group_move_mps"]
            info.append((1.0 if stands else 2.0 if drives else 0.0, 1.0 if o.state in ("confirmed", "coast", "memory") else 0.0))
        found = _NR.radar_groups(self.store.a, [o.slot for o in objs], info, pose,
                                 (c["group_gap_stand_m"], c["group_gap_move_m"], c["group_dv"], c["group_line_m"],
                                  c["group_line_rel"], c["group_lane_m"], c["group_min"]))
        support = self.group_t is None or T - self.group_t >= c["group_s"]
        groups, used = {}, set()
        for kind, mem, line, rms in found:
            votes = {}
            for k in mem:
                g = prev.get(objs[k].id)
                if g is not None and g not in used:
                    votes[g] = votes.get(g, 0) + 1
            gid = max(votes, key=lambda g: (votes[g], -g)) if votes and max(votes.values()) >= 2 else None
            if gid is None:
                gid = self.next_group
                self.next_group += 1
            used.add(gid)
            name = ("row", "queue", "platoon")[kind - 1]
            groups[gid] = {"kind": name, "members": [objs[k].id for k in mem], "line": line, "rms": rms}
            for k in mem:
                objs[k].grp, objs[k].grp_kind = gid, name
            if support and kind != 3 and len(mem) >= 3 and c["group_support"]:
                a = self.store.a
                for k in mem:
                    o = objs[k]
                    if o.verdict == "moving" or self.shown_moving(o) >= c["group_still_m2"]:
                        continue
                    # the line the others make (a member does not support itself), and how well their mean place
                    # across it is known (their own uncertainties across the line, as independent)
                    oth = [objs[q] for q in mem if q != k]
                    mx, my = sum(float(q.x[PX]) for q in oth) / len(oth), sum(float(q.x[PY]) for q in oth) / len(oth)
                    nx, ny = -line[3], line[2]
                    vl = sum(nx * nx * q.P[PX, PX] + 2.0 * nx * ny * q.P[PX, PY] + ny * ny * q.P[PY, PY] for q in oth) / len(oth) ** 2
                    _NR.radar_group_support(a, self.hyp.rows(o.h), (mx, my, line[2], line[3]),
                                            math.sqrt(c["group_sigma_m"] ** 2 + rms * rms + float(vl)), c["group_gain"], c["mu_floor"])
        if support:
            self.group_t = T
        self.groups = groups

    def road_prior(self, T, path):
        """The road's axis at each standing vehicle's place as a prior of its heading (radar_heading_prior), once
        per `road_prior_s`. path: the road model's path as a polyline on the ground (xs, ys), or None. A camera's
        box width cannot tell on which side of the line of sight a vehicle points, and a standing one gets no
        heading from its motion: born along the car's own heading, a car parked on a bend kept it, or drifted to
        the mirror heading, 0.2-0.55 rad off (curve_parked; docs/RADAR.md, section 4)."""
        c = self.cfg
        if not c["road_prior"] or path is None or (self.road_t is not None and T - self.road_t < c["road_prior_s"]):
            return
        self.road_t = T
        px, py = np.asarray(path[0], float), np.asarray(path[1], float)
        if len(px) < 2:
            return
        objs = [o for o in self.objects if (o.label is None or o.label in M.VEHICLES) and o.verdict != "moving"
                and float(o.mu[STAND]) >= 0.5]
        if not objs:
            return
        ax, ay = px[:-1], py[:-1]
        ex, ey = px[1:] - ax, py[1:] - ay
        L2 = np.maximum(ex * ex + ey * ey, 1e-12)
        ox = np.array([float(o.x[PX]) for o in objs])[:, None]
        oy = np.array([float(o.x[PY]) for o in objs])[:, None]
        f = ((ox - ax) * ex + (oy - ay) * ey) / L2
        lo = np.zeros_like(f)
        hi = np.ones_like(f)
        lo[:, 0] = -np.inf                               # the end segments reach on beyond the polyline
        hi[:, -1] = np.inf
        fc = np.clip(f, lo, hi)
        d2 = (ox - (ax + fc * ex)) ** 2 + (oy - (ay + fc * ey)) ** 2
        k = np.argmin(d2, axis=1)
        idx = np.arange(len(objs))
        beyond = (fc[idx, k] - 1.0) * np.sqrt(L2[k])        # past the end of the last segment (m)
        for o, kk, dd, bb in zip(objs, k.tolist(), d2[idx, k].tolist(), beyond.tolist()):
            if dd > c["road_prior_off_m"] ** 2 or (kk == len(ex) - 1 and bb > c["road_prior_beyond_m"]):
                continue
            pose = self.ego.pose_at(o.t)
            if pose is None:
                continue
            los = math.atan2(float(o.x[PY]) - pose[1], float(o.x[PX]) - pose[0])
            _NR.radar_heading_prior(self.store.a, self.hyp.rows(o.h), math.atan2(ey[kk], ex[kk]), los,
                                    c["road_prior_std"], c["road_prior_snap"], c["road_prior_mirror"])

    def _drop(self, o):
        """The object taken out of the hypotheses (it is no longer in `objects`); what ends with it ends too."""
        self.by_h.pop(o.h, None)
        dead = set(self.hyp.drop(self.store.a, o.h)) - {o.h}
        if dead:
            for q in self.objects:
                if q.h in dead:
                    self.by_h.pop(q.h, None)
            self.objects = [q for q in self.objects if q.h not in dead]
        for q in self.objects:                           # their primaries may have changed
            pass

    def _life(self, T, pose, scan):
        """Per object: (p_still, its combined state finite, and for one out of every field (not tentative) where
        it will be at T: the probability that it stands, 1 sigma of its place (larger axis), its range; else
        None x 3)."""
        c = self.cfg
        objs = self.objects
        kin = [o.kin for o in objs]
        return _NR.radar_life(self.store.a, [o.slot for o in objs], [k[3] for k in kin],
                              [1.0 if o.state != "tentative" and (scan - o.last_cover > c["view_gap_s"] or o.hidden_t > max(o.last_seen.values()))
                               else 0.0 for o in objs],
                              [T - o.t for o in objs], kin, M._native_par(self.mcfg), pose[:2] if pose else None,
                              self.mcfg["stand_drift"] ** 2)

    def _register(self, T):
        """Sensor registration: what the objects of one sensor agree on in their angle offsets is not theirs but
        the sensor's (a camera's pitch error). The ripe objects (the ones whose own history has pinned their
        offset: passed by, crossing, sized by their box, ranged by another sensor) say what it is; it moves into
        the sensor's bias and out of every object's offset, so each object's sum stays what its own data said,
        and an object born from now on starts with the pitch already known."""
        c = self.cfg
        dt = 0.0 if self.bias_t is None else max(0.0, min(T - self.bias_t, 1.0))
        self.bias_t = T
        for src, b in self.bias.items():
            sen = self.sensors[src][0]
            cc = sen.cfg
            b[1] = min(b[1] + math.radians(cc["bias_walk_deg"]) ** 2 * dt, math.radians(cc["bias_std0_deg"]) ** 2)
            if dt <= 0.0:
                continue
            ripe = [o for o in self.objects if o.state in ("confirmed", "coast") and src in o.last_seen
                    and T - o.last_seen[src] <= c["bias_seen_s"] and o.P[M.DP, M.DP] < c["bias_ripe"] * o.dp0]
            if not ripe:
                continue
            a = 1.0 - math.exp(-dt / cc["bias_tau_s"])
            # what the ripe objects agree on (weights 1 / (P[dp, dp] + the object's own spread); from three on the
            # middle one by weight: one wrong object does not steer it), a share of it out of every object's offset
            m, ws = _NR.radar_register(self.store.a, [o.slot for o in ripe], sen.dp_obj_var,
                                       [r for o in self.objects if src in o.sources for r in self.hyp.rows(o.h)], a)
            b[0] += a * m
            b[1] = max((1.0 - a) * b[1] + a / ws, math.radians(cc["bias_floor_deg"]) ** 2)

    def _consider(self, o, t):
        """1 (2: and its standing hypothesis) if the object's moving hypotheses take a camera's boxes with its height
        and angle offset held (consider parameters), 0 if not. The box height and the bottom edge's angle tell the
        range only through those two, and while the object moves (its range a free path between the boxes) the two
        cannot be told apart from it: an extended Kalman filter that learns them there drifts (a car driving with the
        traffic at 25 m was put 4-10 m too far on boxes that by themselves were right to 2 %; docs/RADAR.md, section
        2). Held from the sighting on at which the hypotheses are weighed (`mu_hold_hits`). A sensor that measures
        the range (a microwave radar, a position) pins them: then they are learnt."""
        c = self.cfg
        if not c["consider_moving"] or o.hits < c["mu_hold_hits"]:
            return 0.0
        for src, ts in o.last_seen.items():
            sen = self.sensors.get(src)
            if sen is not None and sen[0].kind in ("microwave_radar", "position") and t - ts <= c["consider_range_s"]:
                return 0.0
        return 2.0 if c["consider_stand"] else 1.0

    def _judge(self, o, T, p):
        """"stationary" | "moving" | "unknown" from how much the standing hypothesis is believed, with hysteresis;
        no verdict before the object has been seen often and long enough for the hypotheses to differ."""
        c = self.cfg
        if o.velocity_seen:
            ripe = o.hits >= c["verdict_hits_velocity"] and T - o.t0 >= c["verdict_s_velocity"]
        else:
            ripe = o.hits >= c["verdict_hits"] and T - o.t0 >= c["verdict_s"]
        v = o.verdict
        if v == "stationary" and p < c["release_p"]:
            v = "unknown"
        elif v == "moving" and p > 1.0 - c["release_p"]:
            v = "unknown"
        if v == "unknown" and ripe:
            if p >= c["stand_p"]:
                v = "stationary"
            elif p <= c["move_p"] and (o.velocity_seen or self.shown_moving(o) >= c["move_m2"]):
                v = "moving"
        o.verdict = v

    def _merge(self, T):
        """Two objects that are one (a detection that started a second object beside an existing one): the same
        place within both uncertainties for a while. The older one lives on."""
        c = self.cfg
        dead = set()
        obs = self.objects
        n = len(obs)
        if n < 2:
            for o in obs:
                o.merge_t.clear()
            return
        # the pairs worth a look: nearer than the limit can be at any range in reach, with room for what a
        # difference of the two objects' times may have moved one of them; and whether each is at one place and
        # moves alike
        nearj, nearset = {}, set()
        e = self.ego.pose_at(T)
        here, poses = [], {}
        for o in obs:
            if o.t not in poses:
                poses[o.t] = self.ego.pose_at(o.t)[:2]
            here.append(poses[o.t])
        for i, j, same in _NR.radar_merge(self.store.a, [o.slot for o in obs], [o.t for o in obs], [o.kin for o in obs],
                                          M._native_par(self.mcfg), here, e[:2], (c["merge_m"], c["merge_rel"], c["merge_dv"], c["merge_d2"])):
            nearj.setdefault(i, []).append((j, same))
            nearset.add((i, j))
        index = {o.id: k for k, o in enumerate(obs)}
        for i in range(n):
            mt = obs[i].merge_t
            if mt:                                       # a pair that is no longer near stops counting
                for key in list(mt):
                    j = index.get(key)
                    if j is None:                        # the other object is gone: nothing left to count
                        mt.pop(key, None)
                    elif j > i and (i, j) not in nearset and key not in dead and obs[i].id not in dead:
                        mt.pop(key, None)
            for j, same in nearj.get(i, ()):
                a, b = obs[i], obs[j]
                if a.id in dead or b.id in dead:
                    continue
                la, lb = a.label, b.label
                same_kind = la is None or lb is None or (la in M.VEHICLES) == (lb in M.VEHICLES)
                key = b.id
                if not same or not same_kind:
                    a.merge_t.pop(key, None)
                    continue
                t0 = a.merge_t.setdefault(key, T)
                if T - t0 < c["merge_s"]:
                    continue
                rank = lambda o: (o.state != "tentative", o.hits, -o.t0)
                old, new = (a, b) if rank(a) >= rank(b) else (b, a)
                rows = self.hyp.rows(old.h)             # every story of the one that lives on
                am = self.store.a[:, OFF["m"][0]:OFF["m"][1]]
                am[rows, M_HITS] += new.hits
                am[rows, M_L] = np.maximum(am[rows, M_L], new.logodds)
                old.pe = max(old.pe, new.pe)
                old.sources |= new.sources
                for s, t in new.last_seen.items():
                    old.last_seen[s] = max(old.last_seen.get(s, -1e18), t)
                if old.state == "tentative" and new.state != "tentative":
                    old.state = new.state
                dead.add(new.id)
        if dead:
            self.objects = [o for o in obs if o.id not in dead]
            for o in obs:
                if o.id in dead:
                    self._drop(o)
            for o in self.objects:
                for key in [k for k in o.merge_t if k in dead]:
                    del o.merge_t[key]
