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

from .. import native as _native
from ..core import classvote
from ..core.objects import assign as _hungarian
from . import models as M
from .models import CA, CV, N, PX, PY, STAND, VX, VY
from .sensors import kind_code

# the native core's radar part (chris_native.c), or None: the switch for the functions of this module that have a
# native twin (the advance, peek and stillness of the objects, one scan's association and update)
_NR = _native.radar

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
    # the detector's own confidence in a box: how much likelier a box of that score is a real thing than a
    # false one, learnt from the boxes themselves (the scores of boxes that went into confirmed objects against
    # those of objects that never were confirmed); bins over the score, what is assumed before anything is
    # learnt (mean, spread of a true and of a false box's score, weight in boxes), and how much one box says
    score_bins=10, score_prior=((0.62, 0.14), (0.50, 0.14), 30.0), score_gain=0.6, score_max=1.2, score_keep=3000.0,
    confirm_exist=0.95, confirm_hits=3, delete_exist=0.04,
    max_objects=64,
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
    # life
    view_gap_s=0.7,              # no scan has covered the object's place for this long: it is out of view
    seen_max_s=5.0,              # inside a field: dropped after this long unseen whatever its existence says
    memory_s=45.0,               # out of view: kept at most this long
    memory_std_m=10.0,           # ... and only while its place is known this well (1 sigma, larger axis)
    memory_range_m=150.0,        # ... and it is this near
    memory_tentative_s=1.0,      # an object never confirmed is not worth remembering
    mu_hold_hits=2,              # the hypotheses are weighed from this many sightings on (velocity initialised)
    mu_floor=0.01,               # no hypothesis is ever ruled out for good
    common_min=3, common_pair=0.8,   # the scan's common pitch: from at least this many settled objects, each paired this surely
    bias_ripe=0.5,               # an object speaks about the sensor's bias once its own angle offset is known
                                 # this much better (share of the variance it was born with)
    bias_seen_s=1.0,
)


def assign(gain):
    """The pairs (object, detection) of the largest total gain; a pair is optional (gain -inf: not allowed).
    Settled per group of objects and detections that compete with one another."""
    nt, nd = gain.shape
    ok = np.isfinite(gain)
    if not ok.any():
        return []
    out = []
    seen_t = set()
    for i0 in np.nonzero(ok.any(axis=1))[0]:
        if i0 in seen_t:
            continue
        ts, ds, stack = {int(i0)}, set(), [("t", int(i0))]
        while stack:                                     # the connected group
            kind, k = stack.pop()
            if kind == "t":
                for j in np.nonzero(ok[k])[0]:
                    if int(j) not in ds:
                        ds.add(int(j))
                        stack.append(("d", int(j)))
            else:
                for i in np.nonzero(ok[:, k])[0]:
                    if int(i) not in ts:
                        ts.add(int(i))
                        stack.append(("t", int(i)))
        seen_t |= ts
        ts, ds = sorted(ts), sorted(ds)
        if len(ts) == 1:
            j = max(ds, key=lambda j: gain[ts[0], j])
            out.append((ts[0], j))
            continue
        if len(ds) == 1:
            i = max(ts, key=lambda i: gain[i, ds[0]])
            out.append((i, ds[0]))
            continue
        # every object may also stay without a detection (its own column, cost 0)
        cst = np.full((len(ts), len(ds) + len(ts)), np.inf)
        for a, i in enumerate(ts):
            for b, j in enumerate(ds):
                if ok[i, j]:
                    cst[a, b] = -gain[i, j]
            cst[a, len(ds) + a] = 0.0
        for a, b in _hungarian(cst):
            if b < len(ds) and np.isfinite(cst[a, b]):
                out.append((ts[a], ds[b]))
    return out


def marginals(ratio, iters=30, tol=1e-4):
    """Probability of each pairing (object, detection) when all objects and detections of the scan are
    considered together: every detection is at most one object's, every object has at most one detection.
    ratio[i, j]: odds of "detection j is object i's" against "i was missed and j is something else", for the
    pair alone. Two objects that both fit one detection cannot both have it; the pairing's probability says
    so, where the pair's own odds do not. (Loopy belief propagation on the assignment graph: Williams & Lau
    2014; converges for this problem, a few sweeps.)"""
    L = np.asarray(ratio, float)
    if L.size == 0 or not L.any():
        return np.zeros_like(L)
    nu = np.ones_like(L)
    for _ in range(iters):
        val = L * nu
        mu = L / (1.0 + val.sum(axis=1, keepdims=True) - val)
        new = 1.0 / (1.0 + mu.sum(axis=0, keepdims=True) - mu)
        done = float(np.abs(new - nu).max()) < tol
        nu = new
        if done:
            break
    val = L * nu
    return val / (1.0 + val.sum(axis=1, keepdims=True))


# An object's numbers live in one row of the tracker's state array (`State`): the hypotheses (xs, Ps), their
# probabilities (mu), the free estimate (xf, Pf) and the combination (x, P). The object's attributes are views
# of its row; the native calls read and write the rows of a scan's objects directly.
_LAY = (("xs", (3, N)), ("Ps", (3, N, N)), ("mu", (3,)), ("xf", (N,)), ("Pf", (N, N)), ("x", (N,)), ("P", (N, N)))
_OFF = {}
_o = 0
for _k, _sh in _LAY:
    _OFF[_k] = (_o, _sh)
    _o += int(np.prod(_sh))
ROW = _o                         # numbers per object


class State:
    """The rows of the objects' numbers (an array (capacity, ROW)) and which rows are free."""

    def __init__(self, cap=96):
        self.a = np.zeros((cap, ROW))
        self.free = list(range(cap - 1, -1, -1))

    def take(self, live):
        if not self.free:                                # full: twice the rows, every live object re-pointed
            cap = len(self.a)
            a = np.zeros((2 * cap, ROW))
            a[:cap] = self.a
            self.a = a
            self.free = list(range(2 * cap - 1, cap - 1, -1))
            for o in live:
                o.bind(self)
        return self.free.pop()

    def give(self, slot):
        self.free.append(slot)


def _view(name):
    def get(self):
        return getattr(self, "_" + name)

    def put(self, v):
        getattr(self, "_" + name)[...] = v
    return property(get, put)


class Obj:
    __slots__ = ("id", "slot", "_xs", "_Ps", "_mu", "_xf", "_Pf", "_x", "_P", "t", "t0", "logodds", "hits", "state",
                 "sources", "last_seen", "vote", "history", "checks", "consistency", "pred_err", "n_checks", "nis",
                 "verdict", "coast_s", "velocity_seen", "merge_t", "flags", "last_cover", "dp0", "gain", "last_box",
                 "scores", "range_t")
    xs, Ps, mu, xf, Pf, x, P = (_view(k) for k, _ in _LAY)

    def __init__(self, oid, x0, P0, mu, t, src, exist, coast_s, store=None, live=()):
        self.id = oid
        store = store or State(1)
        self.slot = store.take(live)
        self.bind(store)
        self.xs = x0[None]                       # (3, N): one state per hypothesis
        self.Ps = P0[None]                       # (3, N, N)
        self._xs[STAND][VX:VY + 3] = 0.0         # velocity and acceleration of the standing hypothesis
        # the free estimate: a steady-velocity filter of its own, fed the same detections, never mixed with the
        # hypotheses. What it knows of the velocity is what the sightings have shown, no more.
        self.xf, self.Pf = x0, P0
        self.mu = mu
        self.t = self.t0 = t
        self.logodds = math.log(exist / (1.0 - exist))
        self.hits = 1
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
        self.dp0 = float(P0[M.DP, M.DP])   # how unknown its angle offset was at birth
        self.gain = None             # of its last pairing: log odds for "this detection is mine"
        self.last_box = None         # (time, sensor, angular width) of its last box
        self.scores = []             # (sensor, score) of its boxes while it is not confirmed
        self.range_t = t             # the last time a detection said how far it is (not a bearing alone)
        self.x, self.P = M.combine(self.xs, self.Ps, self.mu)

    def bind(self, store):
        """Point the object's arrays at its row of the state array."""
        row = store.a[self.slot]
        for k, (o, sh) in _OFF.items():
            setattr(self, "_" + k, row[o:o + int(np.prod(sh))].reshape(sh))

    @property
    def exist(self):
        return 1.0 / (1.0 + math.exp(-self.logodds))

    @property
    def label(self):
        return self.vote.get("shown") if self.vote else None

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
        self.objects = []
        self.store = State()         # the objects' numbers, one row each
        self.stats = {}              # sensor id -> counters
        self.sensors = {}            # sensor id -> (last model, time)
        self.bias = {}               # sensor id -> [bias, variance]: what all its objects share (a camera's pitch)
        self.new_rate = {}           # sensor id -> detections per scan no object explained (followed over time)
        self.unexplained = {}        # sensor id -> deque of (t, kind, z): the map's memory
        self.scan_times = {}         # sensor id -> deque of scan times within the map's memory
        self.real_ids = set()        # objects that were confirmed: only their first detections make the map
        self.score_hist = {}         # sensor id -> [true counts, false counts] over the score bins
        self.bias_t = None
        self.t = None

    # ---- time ---------------------------------------------------------------------------------------------
    def _evolve(self, objs, times, combine=False, free=False):
        """The hypotheses of each object as they will be at its target time (interaction, then each forward by
        its own motion), nothing committed: (xs, Ps, mu, dt), with `combine` + (x, P) their combination, with
        `free` + (xf, Pf) the free estimate's prediction (models.evolve_b). A time behind an object's own (a late
        scan) takes it back by the hypotheses' own motion, without switching."""
        dt = [float(t) - o.t for t, o in zip(times, objs)]
        xs, Ps, mu, x, P, xf, Pf = M.evolve_b([o.xs for o in objs], [o.Ps for o in objs], [o.mu for o in objs],
                                              [o.xf for o in objs], [o.Pf for o in objs], dt,
                                              [o.label for o in objs], self.mcfg, combine, free)
        return (xs, Ps, mu, dt) + ((x, P) if combine else ()) + ((xf, Pf) if free else ())

    def _advance_all(self, objs, t):
        """Bring the objects to time t (committed)."""
        objs = [o for o in objs if abs(t - o.t) >= 1e-6]
        if not objs:
            return
        if _NR is not None:
            noise, nz = {}, []
            for o in objs:
                if o.t not in noise:
                    noise[o.t] = self.ego.motion_noise(o.t, t)
                nz.append(noise[o.t])
            pose = self.ego.pose_at(t) if any(a > 0.0 or b > 0.0 for a, b in nz) else (0.0, 0.0, 0.0)
            _NR.radar_advance(self.store.a, [o.slot for o in objs], [float(t) - o.t for o in objs],
                              [M._KIN4.get(o.label) or M._KIN4[None] for o in objs], M._native_par(self.mcfg), nz, pose)
            for o in objs:
                o.t = t
            return
        xs, Ps, mu, dt, xf, Pf = self._evolve(objs, [t] * len(objs), free=True)
        pose = None
        noise = {}
        for k, o in enumerate(objs):
            if dt[k] > 0.0:
                o.mu = mu[k]
            if o.t not in noise:
                noise[o.t] = self.ego.motion_noise(o.t, t)
            sa, sp = noise[o.t]
            if sa > 0.0 or sp > 0.0:
                if pose is None:
                    pose = self.ego.pose_at(t)
                    ch, sh = math.cos(pose[2]), math.sin(pose[2])
                dx, dy = o.x[PX] - pose[0], o.x[PY] - pose[1]
                # the car's travel error lies along its heading; its heading error swings the object about the car
                Qe = sa * sa * np.array([[ch * ch, ch * sh], [ch * sh, sh * sh]]) + sp * sp * np.array([[dy * dy, -dx * dy], [-dx * dy, dx * dx]])
                Ps[k, :, :2, :2] += Qe
                Pf[k, :2, :2] += Qe
            o.xs, o.Ps = xs[k], Ps[k]
            o.xf, o.Pf = xf[k], Pf[k]
            o.t = t
        x, P = M.combine_many(xs, Ps, [o.mu for o in objs])
        for k, o in enumerate(objs):
            o.x, o.P = x[k], P[k]

    def _advance(self, o, t):
        self._advance_all([o], t)

    def peek_many(self, objs, times):
        """Where each object is believed to be at its time, without changing it: [(x, P, mu)]."""
        if not objs:
            return []
        if _NR is not None:
            B = len(objs)
            x, P, mu = np.empty((B, N)), np.empty((B, N, N)), np.empty((B, 3))
            _NR.radar_peek(self.store.a, [o.slot for o in objs], [float(tm) - o.t for tm, o in zip(times, objs)],
                           [M._KIN4.get(o.label) or M._KIN4[None] for o in objs], M._native_par(self.mcfg), x, P, mu)
            return [(x[k], P[k], mu[k]) for k in range(B)]
        xs, Ps, mu, dt, x, P = self._evolve(objs, times, combine=True)
        return [(o.x, o.P, o.mu) if abs(dt[k]) < 1e-6 else (x[k], P[k], mu[k]) for k, o in enumerate(objs)]

    def peek(self, o, t):
        """Where the object is believed to be at time t, without changing it: (x, P, mu)."""
        if abs(t - o.t) < 1e-6:
            return o.x, o.P, o.mu
        return self.peek_many([o], [t])[0]

    def _local_density(self, src, d):
        """Unexplained detections per scan and unit of measurement space around this detection, from the map:
        (total, {object id: the part that is that object's own first detection})."""
        mem = self.unexplained.get(src)
        k = (d.get("kind") or "").rstrip("s")
        if not mem or k not in ("rb", "rbh"):
            return 0.0, {}
        n = len(k)                                        # "rb": 2 rows, "rbh": 3
        bw = self.cfg["map_bw"][:n]
        z = d["z"]
        norm = float(max(len(self.scan_times.get(src) or ()), 1))
        for a in range(n):
            norm *= bw[a] * 2.5066
        tot, own = 0.0, {}
        for (_, kk, zz, oid) in mem:
            if len(zz) < n or oid not in self.real_ids:   # a detection that never became an object is no place
                continue                                  # where objects appear
            q = 0.0
            for a in range(n):
                q += ((z[a] - zz[a]) / bw[a]) ** 2
            if q < 25.0:
                v = math.exp(-0.5 * q) / norm
                tot += v
                own[oid] = own.get(oid, 0.0) + v
        return tot, own

    @staticmethod
    def _still(o):
        """How still the object's two moving hypotheses are (models.still)."""
        q = M.still_b(o.xs[1:], o.Ps[1:], M.kin(o.label)["v_still"])
        return float(q[0]), float(q[1])

    def p_still(self, o):
        """Probability that the object does not move: "it stands", and the moving hypotheses as far as they
        have found no speed."""
        q = self._still(o)
        return float(min(1.0, o.mu[STAND] + o.mu[CV] * q[0] + o.mu[CA] * q[1]))

    @staticmethod
    def shown_moving(o):
        """How far the object's sightings by themselves show that it moves: the distance of "no speed" from the
        free estimate of its velocity, in that estimate's own uncertainty (squared; 9 is one chance in a
        hundred for a thing that stands)."""
        v, S = o.xf[VX:VY + 1], o.Pf[VX:VY + 1, VX:VY + 1]
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if det <= 0.0:
            return 0.0
        return float((S[1, 1] * v[0] * v[0] - 2.0 * S[0, 1] * v[0] * v[1] + S[0, 0] * v[1] * v[1]) / det)

    def p_still_many(self, objs):
        if not objs:
            return []
        if _NR is not None:
            return _NR.radar_still(self.store.a, [o.slot for o in objs], [(M._KIN4.get(o.label) or M._KIN4[None])[3] for o in objs])
        return M.p_still_many([o.xs for o in objs], [o.Ps for o in objs], [o.mu for o in objs], [o.label for o in objs])

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
            h = self.score_hist[src] = [wgt * ht / ht.sum(), wgt * hf / hf.sum()]
        return h

    def _score_odds(self, src, sc):
        """Log odds a box of this score adds for "a real thing" (0 without a score)."""
        if sc is None:
            return 0.0
        c = self.cfg
        ht, hf = self._hist(src)
        b = min(int(sc * c["score_bins"]), c["score_bins"] - 1)
        r = math.log(((ht[b] + 0.5) / (ht.sum() + 0.5 * len(ht))) / ((hf[b] + 0.5) / (hf.sum() + 0.5 * len(hf))))
        return c["score_gain"] * max(-c["score_max"], min(c["score_max"], r))

    def _score_learn(self, scores, real):
        c = self.cfg
        for src, sc in scores:
            h = self._hist(src)[0 if real else 1]
            h[min(int(sc * c["score_bins"]), c["score_bins"] - 1)] += 1.0
            if h.sum() > c["score_keep"]:
                h *= 0.5

    # ---- one scan -----------------------------------------------------------------------------------------
    def process(self, src, t, sensor, dets):
        """dets: [{"z": array, "R": variances (1-D), "attrs": {...}}]."""
        c = self.cfg
        st = self.stats.setdefault(src, {"scans": 0, "dets": 0, "assigned": 0, "started": 0, "late": 0})
        st["scans"] += 1
        st["dets"] += len(dets)
        self.sensors[src] = (sensor, t)
        if getattr(sensor, "has_bias", False):
            b = self.bias.setdefault(src, [0.0, sensor.bias_var])
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
        self._advance_all(objs, t)
        self.t = t if self.t is None else max(self.t, t)
        cph, sph = math.cos(pose[2]), math.sin(pose[2])
        if getattr(sensor, "extraneous", None) is not None:
            sensor.extraneous = max(sensor.extraneous, self.new_rate.get(src, 0.0))
        Rs = [np.diag(np.asarray(d["R"], float)) for d in dets]
        native = _NR is not None and hasattr(sensor, "native")
        if native:
            D = self._det_rows(dets)
            pairs, pds, cover = self._associate_native(src, t, sensor, pose, objs, D)
        else:
            pairs, pds, cover = self._associate(src, t, sensor, pose, objs, dets)
        hit, used = set(), set()
        if self._common(sensor, src, pose, objs, dets, Rs, pairs) and native:
            D = self._det_rows(dets, Rs)
        if native and pairs:
            res = _NR.radar_update(self.store.a, [objs[i].slot for i, *_ in pairs], sensor.native(), pose,
                                   D[[j for _, j, *_ in pairs]], [(pds[i], lam, pr, objs[i].hits) for i, _, _, pr, lam in pairs],
                                   (c["mu_hold_hits"], c["mu_floor"]))
        for k, (i, j, g, pr, lam) in enumerate(pairs):
            objs[i].gain = g
            if native:
                self._after_update(objs[i], dets[j], sensor, src, t, *res[k])
            else:
                self._update(objs[i], dets[j], Rs[j], sensor, pose, src, t, pds[i], lam, pr)
            if dets[j].get("w"):
                objs[i].last_box = (t, src, dets[j]["w"])
            hit.add(i)
            used.add(j)
            st["assigned"] += 1
        if hit:                                          # written down now, checked when the time has come
            who = [objs[i] for i in sorted(hit)]
            hz = c["check_horizons"]
            ahead = self.peek_many([o for o in who for _ in hz], [t + h for _ in who for h in hz])
            for k, (xp, Pp, _) in enumerate(ahead):
                who[k // len(hz)].checks.append((t + hz[k % len(hz)], float(xp[PX]), float(xp[PY]), Pp[:2, :2].copy()))
        for i, o in enumerate(objs):
            pd = pds[i]
            if cover[i] > 0.0:
                o.last_cover = t
            if i in hit:
                o.last_cover = t
                continue
            if pd > 0.0:
                o.logodds += math.log(max(1.0 - pd, 1e-3))
        free = 0
        mem = self.unexplained.setdefault(src, deque())
        times = self.scan_times.setdefault(src, deque())
        times.append(t)
        while times and times[0] < t - c["map_s"]:
            times.popleft()
        while mem and mem[0][0] < t - c["map_s"]:
            mem.popleft()
        for j, d in enumerate(dets):
            if j in used or len(self.objects) >= c["max_objects"]:
                continue
            label = classvote.norm_label((d.get("attrs") or {}).get("class"))
            s0 = sensor.start(d, pose, label)
            if s0 is None:
                continue
            free += 1
            x0, P0 = s0
            k = M.kin(label)
            if sensor.velocity:                          # the speed along the line of sight is measured
                moving = math.hypot(x0[VX], x0[VY]) > 1.0
                mu = [0.05, 0.8, 0.15] if moving else [0.7, 0.25, 0.05]
            else:
                ps = k["p_stand"]
                dxo, dyo = x0[PX] - pose[0], x0[PY] - pose[1]
                pa = c["p_any_side"] if abs(-sph * dxo + cph * dyo) > c["side_m"] else self.mcfg["p_ca0"]
                mu = [ps, (1.0 - ps) * (1.0 - pa), (1.0 - ps) * pa]
            o = Obj(self.next_id, x0, P0, mu, t, src, c["new_exist"], sensor.coast_s, self.store, self.objects)
            self.next_id += 1
            sc = self._score(d)
            o.logodds += self._score_odds(src, sc)
            if sc is not None:
                o.scores.append((src, sc))
            if not sensor.velocity:
                # What a new object may be doing, before anything is known, is not one blur around "no speed":
                # it stands (its own hypothesis), or it drives with the traffic (the steady hypothesis: about
                # the car's own velocity, along the road the car is on), or it comes the other way (the
                # manoeuvring hypothesis: about the car's velocity reversed, and wider across). The faster
                # the car itself drives, the more that says about the traffic's speed; standing, nothing.
                # A vehicle whose box shows it side-on does neither: it crosses.
                U = np.array([[cph, -sph], [sph, cph]])
                lat = min(k["v0_lat"], k["v0_std"]) ** 2
                big = k["v0_std"] ** 2
                at = d.get("attrs") or {}
                ps = M.side_on(label, at.get("aspect"), float(d["z"][sensor.rows(d)[0]])) if at.get("aspect") else None
                ve = min(math.hypot(pose[3], pose[4]), k["v_max"]) if "wl" in k else 0.0
                if ps is None:
                    ps = 0.0 if "wl" in k else None
                if ps is None:                           # a kind without a direction of its own (a person)
                    Pv, mv = U @ np.diag([big, lat]) @ U.T, np.zeros(2)
                else:
                    w_ = max(c["with_rel"] * ve + c["with_std"], k["v0_std"] - ve) ** 2
                    Pv = U @ np.diag([(1.0 - ps) * w_ + ps * lat + ps * (1.0 - ps) * ve * ve,
                                      (1.0 - ps) * lat + ps * big]) @ U.T
                    mv = (1.0 - ps) * ve * np.array([cph, sph])
                o.xs[CV][VX:VY + 1] = mv
                if ps is not None:
                    lat2 = min(2.0 * k["v0_lat"], k["v0_std"]) ** 2
                    o.xs[CA][VX:VY + 1] = -mv
                    o.Ps[CA][VX:VY + 1, VX:VY + 1] = U @ np.diag([(1.0 - ps) * w_ + ps * lat2 + ps * (1.0 - ps) * ve * ve,
                                                                   (1.0 - ps) * lat2 + ps * big]) @ U.T
                o.Ps[CV][VX:VY + 1, VX:VY + 1] = Pv
                o.Pf[VX:VY + 1, VX:VY + 1] = U @ np.diag([big, lat if not ps else (1.0 - ps) * lat + ps * big]) @ U.T
                o.x, o.P = M.combine(o.xs, o.Ps, o.mu)
            o.velocity_seen = sensor.velocity
            self._vote(o, d, src, t)
            o.history.append((t, float(x0[PX]), float(x0[PY]), 0.0, 0.0))
            if d.get("w"):
                o.last_box = (t, src, d["w"])
            if (d.get("kind") or "").endswith("s"):      # its first box already says how fast it comes nearer
                self._first_rate(o, d, sensor, pose)
            mem.append((t, (d.get("kind") or "").rstrip("s"), np.asarray(d["z"], float), o.id))
            self.objects.append(o)
            st["started"] += 1
        r0 = self.new_rate.get(src, 0.0)
        self.new_rate[src] = r0 + c["new_rate_gain"] * (free - r0)

    def _associate(self, src, t, sensor, pose, objs, dets):
        """Which detection is whose: [(object, detection, gain, the pairing's probability, the other story's
        density over the box feature)], each object's probability of being reported, the sensor's coverage at its
        place."""
        c = self.cfg
        nt, nd = len(objs), len(dets)
        cph, sph = math.cos(pose[2]), math.sin(pose[2])
        car = []                                         # each object's place in the car frame at the scan
        for o in objs:
            dx, dy = o.x[PX] - pose[0], o.x[PY] - pose[1]
            car.append((cph * dx + sph * dy, -sph * dx + cph * dy))
        # Which detection is whose. A detection's fit to an object is the likelihood the object's hypotheses give
        # it, each with its own prediction and weight (one blurred Gaussian over "it stands here" and "it may be
        # driving at any speed" would let a standing car claim its neighbour's box). It is weighed against the
        # other explanation: the object was missed this time and the detection is something not known yet. Only
        # a pair that is the likelier story may be made, and the scan is settled as a whole for the largest
        # total gain.
        gain = np.full((nt, nd), -np.inf)
        ratio = np.zeros((nt, nd))
        feat = {}
        pds = [sensor.p_detect(car[i][0], car[i][1], o.x) for i, o in enumerate(objs)]
        if nd and nt:
            lam0 = np.array([sensor.density(d) for d in dets])
            loc = [self._local_density(src, d) for d in dets]
            loc_tot = np.array([l[0] for l in loc])
            labs = [classvote.norm_label((d.get("attrs") or {}).get("class")) for d in dets]
            veh = np.array([0 if l is None else (1 if l in M.VEHICLES else -1) for l in labs])
            wid = np.array([float(d.get("w") or 0.0) for d in dets])
            kinds = {}                                   # a sensor's detections may be of several kinds
            for jj, d in enumerate(dets):                # (a scale rate's baseline is part of what is predicted)
                kinds.setdefault((d.get("kind"), d.get("sdt")), []).append(jj)
            groups = []
            for k, J in kinds.items():
                J = np.array(J)
                Zk = np.array([np.asarray(dets[jj]["z"], float) for jj in J])
                Rk = np.array([np.asarray(dets[jj]["R"], float) for jj in J])
                groups.append((dets[J[0]], J, Zk, Rk, list(sensor.rows(dets[J[0]])), Zk.shape[1]))
            eye = {m: np.eye(m) for m in {g[5] for g in groups}}
            for i, o in enumerate(objs):
                pd = pds[i]
                if pd <= 0.0:
                    continue
                la = o.label
                ov = 0 if la is None else (1 if la in M.VEHICLES else -1)
                lb = o.last_box
                for d0, J, Zk, Rk, rows, m in groups:
                    if not sensor.takes(d0, *car[i]):
                        continue
                    ok = (veh[J] * ov) >= 0          # a pedestrian's box does not continue a vehicle's object
                    if not ok.any():
                        continue
                    zp, H = sensor.h(o.x, pose, d0)
                    nu = Zk - zp
                    for r_ in rows:
                        nu[:, r_] = (nu[:, r_] + math.pi) % (2.0 * math.pi) - math.pi
                    S = (H @ o.P @ H.T)[None] + Rk[:, :, None] * eye[m][None]
                    try:
                        d2 = np.einsum("jm,jm->j", nu, np.linalg.solve(S, nu[:, :, None])[:, :, 0])
                    except np.linalg.LinAlgError:
                        continue
                    ok &= d2 <= c["gate"].get(m, 16.0)
                    if not ok.any():
                        continue
                    sel = np.nonzero(ok)[0]
                    Js, nus = J[sel], nu[sel]
                    like = np.zeros(len(sel))
                    for mm in range(len(M.MODELS)):
                        if o.mu[mm] < 1e-4:
                            continue
                        zm, Hm = sensor.h(o.xs[mm], pose, d0)
                        num = Zk[sel] - zm
                        for r_ in rows:
                            num[:, r_] = (num[:, r_] + math.pi) % (2.0 * math.pi) - math.pi
                        Sm = (Hm @ o.Ps[mm] @ Hm.T)[None] + Rk[sel][:, :, None] * eye[m][None]
                        try:
                            q = np.einsum("jm,jm->j", num, np.linalg.solve(Sm, num[:, :, None])[:, :, 0])
                        except np.linalg.LinAlgError:
                            continue
                        det_s = np.linalg.det(Sm)
                        good = (q < 60.0) & (det_s > 0.0)
                        like += np.where(good, o.mu[mm] * np.exp(-0.5 * np.minimum(q, 60.0))
                                         / np.sqrt((2.0 * math.pi) ** m * np.where(det_s > 0.0, det_s, 1.0)), 0.0)
                    F = np.ones(len(sel))
                    if lb is not None and o.hits >= 2 and lb[1] == src and t - lb[0] <= c["feat_max_s"]:
                        w_ = wid[Js]
                        has = w_ > 0.0
                        e = np.abs(nus[:, rows[0]]) / np.maximum(0.5 * (w_ + lb[2]), 1e-4)
                        sg = c["feat_sigma"]
                        F = np.where(has, c["feat_span"] * (c["feat_same"] * 0.7979 / sg * np.exp(-0.5 * np.minimum(e / sg, 12.0) ** 2)
                                                           + (1.0 - c["feat_same"]) / c["feat_span"]), 1.0)
                    for a_, jj in enumerate(Js):
                        if like[a_] <= 0.0:
                            continue
                        # the other story's density here: not counting this object's own first detection
                        lam = max(lam0[jj], loc_tot[jj] - loc[jj][1].get(o.id, 0.0))
                        ratio[i, jj] = pd * like[a_] * F[a_] / ((1.0 - pd) * lam)
                        g = math.log(ratio[i, jj])
                        if g > 0.0:
                            gain[i, jj] = g
                            feat[(i, jj)] = lam / F[a_]
        pair = marginals(ratio)
        cover = [sensor.coverage(*car[i]) for i in range(nt)]
        return [(i, j, float(gain[i, j]), float(pair[i, j]), feat[(i, j)]) for i, j in assign(gain)], pds, cover

    def _det_rows(self, dets, Rs=None):
        """The scan's detections as the native core reads them (chris_native.c `RDET`): kind, scale-rate baseline,
        group (kind and baseline: what is predicted alike), vehicle (1) / person (-1) / not known (0), box width,
        z, the variances (R's diagonal)."""
        D = np.zeros((len(dets), 13))
        kinds = {}
        for j, d in enumerate(dets):
            k = d.get("kind")
            lab = classvote.norm_label((d.get("attrs") or {}).get("class"))
            z = d["z"]
            m = len(z)
            D[j, :5] = (kind_code(k), float(d.get("sdt") or 0.0), kinds.setdefault((k, d.get("sdt")), len(kinds)),
                        0 if lab is None else (1 if lab in M.VEHICLES else -1), float(d.get("w") or 0.0))
            D[j, 5:5 + m] = z
            D[j, 9:9 + m] = d["R"] if Rs is None else np.diag(Rs[j])
        return D

    def _associate_native(self, src, t, sensor, pose, objs, D):
        """_associate in one native call (chris_native.c `radar_scan`)."""
        c = self.cfg
        oi = []
        for o in objs:
            la, lb = o.label, o.last_box
            oi.append((0 if la is None else (1 if la in M.VEHICLES else -1),
                       lb[2] if lb is not None and o.hits >= 2 and lb[1] == src and t - lb[0] <= c["feat_max_s"] else math.nan,
                       o.id))
        mem = self.unexplained.get(src)
        rows = None
        if mem:
            real = self.real_ids
            rows = [(len(zz), zz[0], zz[1], zz[2] if len(zz) > 2 else 0.0, oid) for (_, _, zz, oid) in mem if oid in real] or None
        g, bw = c["gate"], c["map_bw"]
        nt = len(objs)
        pds, cover = np.empty(nt), np.empty(nt)
        pairs = _NR.radar_scan(self.store.a, [o.slot for o in objs], sensor.native(), pose, oi, D, rows,
                               float(len(self.scan_times.get(src) or ())),
                               (g.get(1, 16.0), g.get(2, 16.0), g.get(3, 16.0), g.get(4, 16.0), c["feat_sigma"],
                                c["feat_same"], c["feat_span"], bw[0], bw[1], bw[2]), pds, cover, None, None, None)
        return pairs, pds.tolist(), cover.tolist()

    def _common(self, sensor, src, pose, objs, dets, Rs, pairs):
        """What all objects of one scan share: the camera body's pitch at that moment. Every box of the scan
        then stands higher or lower by the same angle, which no object's own history predicts and which, left
        in, moves every range at once. The settled objects of the scan say what it was (the middle one of their
        angle innovations, by weight: one wrong pairing does not steer it); the part of it that the sensor's
        slow pitch error explains goes into that error (`bias`), the rest is this scan's own and is taken
        out of the scan's angles."""
        c = self.cfg
        if not getattr(sensor, "has_bias", False) or len(pairs) < c["common_min"]:
            return False
        v, w = [], []
        for i, j, _, pr, _ in pairs:
            o, d = objs[i], dets[j]
            if (d.get("kind") or "").rstrip("s") not in ("rb", "rbh") or o.state not in ("confirmed", "coast") or o.hits < 4 \
                    or pr < c["common_pair"]:
                continue
            zp, H = sensor.h(o.x, pose, d)
            s00 = float(H[0] @ o.P @ H[0]) + float(Rs[j][0, 0])
            v.append(float(d["z"][0]) - float(zp[0]))
            w.append(1.0 / s00)
        if len(v) < c["common_min"]:
            return False
        v, w = np.array(v), np.array(w)
        k = np.argsort(v)
        cw = np.cumsum(w[k])
        m = float(v[k][int(np.searchsorted(cw, 0.5 * cw[-1]))])
        var_m = 1.57 / float(w.sum())                    # of a weighted median
        jit = math.radians(sensor.cfg["pitch_jitter_deg"]) ** 2
        b = self.bias[src]
        tot = b[1] + jit + var_m
        slow, fast = m * b[1] / tot, m * jit / tot
        b[0] += slow
        b[1] = max(b[1] * (1.0 - b[1] / tot), math.radians(sensor.cfg["bias_floor_deg"]) ** 2)
        sensor.bias, sensor.bias_var = b[0], b[1]
        left = jit * (1.0 - jit / tot)                   # what is still not known of this scan's own pitch
        for j, d in enumerate(dets):
            if (d.get("kind") or "").rstrip("s") in ("rb", "rbh"):
                d["z"] = np.array(d["z"], float)
                d["z"][0] -= fast
                Rs[j][0, 0] = max(Rs[j][0, 0] - jit + left, 0.25 * Rs[j][0, 0])
        return True

    def _first_rate(self, o, d, sensor, pose):
        """A new object's first box came with a scale rate: how fast its image grows. That is its speed along
        the line of sight over its distance, measured, before a second box exists. Every hypothesis takes it
        (for "it stands" it is a range: the car's own speed over the distance), and how well each had expected
        it weighs them at once: a thing that grows as the car's speed says stands; one that does not grow
        drives with the traffic."""
        c = self.cfg
        z = float(d["z"][-1])
        R = float(d["R"][-1])
        sd = {"kind": "s", "sdt": d.get("sdt")}
        like = np.zeros(len(M.MODELS))
        for j in range(len(M.MODELS) + 1):
            x, P = (o.xf, o.Pf) if j == len(M.MODELS) else (o.xs[j], o.Ps[j])
            zp, H = sensor.h(x, pose, sd)
            S = float(H[0] @ P @ H[0]) + R
            nu = z - float(zp[0])
            K = P @ H[0] / S
            xn = x + K * nu
            Pn = P - np.outer(K, K) * S
            Pn = 0.5 * (Pn + Pn.T)
            if j == len(M.MODELS):
                o.xf, o.Pf = xn, Pn
                break
            like[j] = math.exp(-0.5 * min(nu * nu / S, 400.0)) / math.sqrt(2.0 * math.pi * S)
            o.xs[j], o.Ps[j] = xn, Pn
            if j == STAND:
                o.xs[j][VX:VY + 3] = 0.0
        w = o.mu * like
        if float(w.sum()) > 0.0:
            o.mu = np.clip(w / float(w.sum()), c["mu_floor"], None)
            o.mu /= o.mu.sum()
        o.x, o.P = M.combine(o.xs, o.Ps, o.mu)

    @staticmethod
    def _compatible(o, det):
        """A pedestrian's box does not continue a vehicle's object and the reverse (when both are known)."""
        a, b = o.label, classvote.norm_label((det.get("attrs") or {}).get("class"))
        if a is None or b is None:
            return True
        return (a in M.VEHICLES) == (b in M.VEHICLES)

    def _vote(self, o, d, src, t):
        at = d.get("attrs") or {}
        if at.get("class") is not None:
            classvote.vote(o.vote, at.get("class"), at.get("class_conf", at.get("prob")), src, t, at.get("weight", 1.0))

    def _update(self, o, d, R, sensor, pose, src, t, pd=None, lam=None, pair=None):
        """One detection into one object. Each hypothesis takes it only as far as it can be this object's at
        all under that hypothesis: against the other story (the object was missed, the detection is something
        else: probability 1 - beta). A hypothesis the detection does not fit is neither dragged to it nor
        ruled out by it; it is believed less by exactly the odds of the pairing (IMM with probabilistic data
        association)."""
        c = self.cfg
        z = np.asarray(d["z"], float)
        pd = sensor.p_d if pd is None else pd
        lam = sensor.density(d) if lam is None else lam
        other = max((1.0 - pd) * lam, 1e-300)
        like = np.zeros(len(M.MODELS))
        work = []
        for j in range(len(M.MODELS)):
            zp, H = sensor.h(o.xs[j], pose, d)
            nu = sensor.residual(z, zp, d)
            S = H @ o.Ps[j] @ H.T + R
            Si = np.linalg.inv(S)
            d2 = float(nu @ Si @ nu)
            like[j] = math.exp(-0.5 * min(d2, 400.0)) / math.sqrt(max((2.0 * math.pi) ** len(nu) * np.linalg.det(S), 1e-30))
            work.append((H, nu, S, Si))
        # Whose the detection is, is one question for the whole object: if the object as it is believed owns
        # it, every hypothesis learns from it, also one that is still too unsure of itself to claim anything
        # (a moving hypothesis without a velocity yet: left out, it would never get one and lose to "it
        # stands" for that alone). A hypothesis that fits the detection on its own learns from it even while
        # the object as a whole doubts (the standing car that pulls away).
        # With other objects competing for the same detections, the pairing's probability in the scan as a
        # whole (`pair`) is what counts, and each hypothesis' own claim shrinks by as much.
        lo = float((o.mu * like).sum())
        beta_o = pd * lo / (pd * lo + other)
        comp = 1.0 if pair is None or beta_o <= 0.0 else min(1.0, pair / beta_o)
        beta_o *= comp
        for j in range(len(M.MODELS)):
            H, nu, S, Si = work[j]
            P = o.Ps[j]
            beta = max(comp * pd * like[j] / (pd * like[j] + other), beta_o)
            K = P @ H.T @ Si
            dx = K @ nu
            o.xs[j] = o.xs[j] + beta * dx
            Pn = P - beta * (K @ S @ K.T) + beta * (1.0 - beta) * np.outer(dx, dx)
            o.Ps[j] = 0.5 * (Pn + Pn.T)
            if j == STAND:                               # the rigid hypothesis has no velocity to learn
                o.xs[j][VX:VY + 3] = 0.0
            o.xs[j][M.DP] = min(0.05, max(-0.05, o.xs[j][M.DP]))
        # the free estimate takes the detection the same way
        zp, H = sensor.h(o.xf, pose, d)
        nu = sensor.residual(z, zp, d)
        S = H @ o.Pf @ H.T + R
        Si = np.linalg.inv(S)
        lf = math.exp(-0.5 * min(float(nu @ Si @ nu), 400.0)) / math.sqrt(max((2.0 * math.pi) ** len(nu) * np.linalg.det(S), 1e-30))
        beta = max(comp * pd * lf / (pd * lf + other), beta_o)
        K = o.Pf @ H.T @ Si
        dx = K @ nu
        o.xf = o.xf + beta * dx
        Pn = o.Pf - beta * (K @ S @ K.T) + beta * (1.0 - beta) * np.outer(dx, dx)
        o.Pf = 0.5 * (Pn + Pn.T)
        o.xf[M.DP] = min(0.05, max(-0.05, o.xf[M.DP]))
        # how the object as a whole had predicted the detection (before using it)
        zp, H = sensor.h(o.x, pose, d)
        nu = sensor.residual(z, zp, d)
        S = H @ o.P @ H.T + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        n = d2 / len(nu)
        tot = float((o.mu * like).sum())
        w = o.mu * (pd * like + other)                   # each hypothesis: "mine and it fits" or "not mine"
        # The hypotheses are weighed by how well each PREDICTED the detection. A moving hypothesis cannot predict
        # before it has a velocity: its first detections are explained by whatever velocity fits them, and until
        # then it would lose to "it stands" merely for not having been told (an object dead ahead at the car's
        # own speed keeps its bearing exactly like a standing one). So the weighing starts once the object has
        # been seen `mu_hold_hits` times (two-point initialisation of the velocity), unless the sensor measures
        # speed itself. No hypothesis is ever ruled out for good (`mu_floor`): the next seconds may bring it back.
        if tot > 0.0 and (sensor.velocity or o.hits >= c["mu_hold_hits"] or (d.get("kind") or "").endswith("s")):
            o.mu = np.clip(w / float(w.sum()), c["mu_floor"], None)
            o.mu /= o.mu.sum()
        elif tot > 0.0 and float(w[1:].sum()) > 0.0:
            # between the moving hypotheses the comparison is fair from the first sighting on: neither has a
            # velocity, they differ in what they expect of one (steady along the road; anything: crossing,
            # turning). "It stands" keeps its share meanwhile.
            mv = float(o.mu[1:].sum())
            o.mu[1:] = np.clip(mv * w[1:] / float(w[1:].sum()), c["mu_floor"], None)
            o.mu /= o.mu.sum()
        like_all = max(tot, 1e-30)
        dl = min(max(math.log(max(sensor.p_d * like_all / sensor.density(d), 1e-9)), math.log(1.5)), math.log(20.0))
        o.x, o.P = M.combine(o.xs, o.Ps, o.mu)
        self._after_update(o, d, sensor, src, t, n, dl)

    def _after_update(self, o, d, sensor, src, t, n, dl):
        """What a detection does to an object besides its numbers: its record (NIS n), its existence (dl: what the
        fit adds to its log odds), score, sightings, class vote, the checks of its predictions, its trajectory."""
        c = self.cfg
        o.nis[src] = n if o.nis.get(src) is None else o.nis[src] + 0.2 * (n - o.nis[src])
        o.logodds += dl
        sc = self._score(d)
        o.logodds += self._score_odds(src, sc)
        if sc is not None:
            if o.state == "tentative":
                o.scores.append((src, sc))
            else:
                self._score_learn([(src, sc)], True)
        o.logodds = max(-12.0, min(12.0, o.logodds))
        o.hits += 1
        if (d.get("kind") or "rb").rstrip("s") not in ("b", "bt"):
            o.range_t = t            # a box cut at the bottom gives a direction, and a range only by the top edge
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
            ex, ey = o.x[PX] - o.x[VX] * back - px, o.x[PY] - o.x[VY] * back - py
            S = Pp + o.P[:2, :2] + np.eye(2) * 0.05
            try:
                m2 = float(np.array([ex, ey]) @ np.linalg.solve(S, np.array([ex, ey]))) / 2.0
            except np.linalg.LinAlgError:
                continue
            e = math.hypot(ex, ey)
            o.n_checks += 1
            o.consistency = m2 if o.consistency is None else o.consistency + 0.25 * (m2 - o.consistency)
            o.pred_err = e if o.pred_err is None else o.pred_err + 0.25 * (e - o.pred_err)
        o.checks = keep[-8:]
        o.flags.discard("unpredictable")
        if o.n_checks >= 4 and o.consistency is not None and o.consistency > c["inconsistent"]:
            o.flags.add("unpredictable")                 # it keeps being somewhere else than predicted: believe less
            for j in range(len(M.MODELS)):
                o.Ps[j] = o.Ps[j] * c["inflate"]
            o.Pf = o.Pf * c["inflate"]
            o.x, o.P = M.combine(o.xs, o.Ps, o.mu)
            o.consistency = 0.5 * (o.consistency + 1.0)

    # ---- life ---------------------------------------------------------------------------------------------
    def manage(self, T):
        c = self.cfg
        keep = []
        pose = self.ego.pose_at(T)
        still = dict(zip((o.id for o in self.objects), self.p_still_many(self.objects)))
        # "out of view" is what scans say that came and did not cover the object's place; while no scan comes
        # at all (frames dropped, the detector shed by the load governor) nothing has left any field
        scan = T if self.t is None else min(T, self.t)
        away = [o for o in self.objects if o.state != "tentative" and scan - o.last_cover > c["view_gap_s"]
                and np.all(np.isfinite(o.x))]
        there = dict(zip((o.id for o in away), self.peek_many(away, [T] * len(away))))
        for o in self.objects:
            last = max(o.last_seen.values())
            if o.exist < c["delete_exist"] or not np.all(np.isfinite(o.x)):
                continue
            if scan - o.last_cover <= c["view_gap_s"]:
                # a sensor is looking at where it should be: it lives by being seen
                if T - last > max(o.coast_s, c["seen_max_s"]):
                    continue
                if o.state == "tentative":
                    if o.exist >= c["confirm_exist"] and o.hits >= c["confirm_hits"]:
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
                    x, P, mu = there[o.id]
                    if mu[STAND] >= 0.5:
                        # believed standing: where it stands is known as well as ever (that it MAY have driven
                        # off since is in its probabilities, not a blur of its place)
                        Ps = o.Ps[STAND][:2, :2] + np.eye(2) * (self.mcfg["stand_drift"] ** 2) * max(T - o.t, 0.0)
                        sd = math.sqrt(max(float(np.linalg.eigvalsh(Ps)[-1]), 0.0))
                    else:
                        sd = math.sqrt(max(float(np.linalg.eigvalsh(P[:2, :2])[-1]), 0.0))
                    rng = math.hypot(x[PX] - pose[0], x[PY] - pose[1]) if pose else 0.0
                    if T - o.last_cover > c["memory_s"] or sd > c["memory_std_m"] or rng > c["memory_range_m"]:
                        continue
                    o.state = "memory"
            self._judge(o, T, still[o.id])
            keep.append(o)
        kept = {o.id for o in keep}
        for o in self.objects:                           # never confirmed and gone: its boxes were false ones
            if o.id not in kept and o.state == "tentative" and o.scores:
                self._score_learn(o.scores, False)
        for o in self.objects:
            if o.id not in kept:
                self.store.give(o.slot)
        self.objects = keep
        if len(self.real_ids) > 4096:
            self.real_ids = {o.id for o in keep if o.state != "tentative"}
        self._merge(T)
        self._register(T)

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
            w = np.array([1.0 / (o.P[M.DP, M.DP] + sen.dp_obj_var) for o in ripe])
            v = np.array([o.x[M.DP] for o in ripe])
            if len(ripe) >= 3:                           # the middle one by weight: one wrong object does not steer it
                k = np.argsort(v)
                cw = np.cumsum(w[k])
                m = float(v[k][int(np.searchsorted(cw, 0.5 * cw[-1]))])
            else:
                m = float((w * v).sum() / w.sum())
            a = 1.0 - math.exp(-dt / cc["bias_tau_s"])
            shift = a * m
            b[0] += shift
            b[1] = max((1.0 - a) * b[1] + a / float(w.sum()), math.radians(cc["bias_floor_deg"]) ** 2)
            for o in self.objects:
                if src in o.sources:
                    for j in range(len(M.MODELS)):
                        o.xs[j][M.DP] -= shift
                    o.x[M.DP] -= shift
                    o.xf[M.DP] -= shift

    def _judge(self, o, T, p=None):
        """"stationary" | "moving" | "unknown" from how much the standing hypothesis is believed, with hysteresis;
        no verdict before the object has been seen often and long enough for the hypotheses to differ."""
        c = self.cfg
        p = self.p_still(o) if p is None else p
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
            return
        # the pairs worth a look: nearer than the limit can be at any range in reach, with room for what a
        # difference of the two objects' times may have moved one of them
        xy = np.array([o.x[:2] for o in obs])
        tt = np.array([o.t for o in obs])
        sp = np.array([math.hypot(o.x[VX], o.x[VY]) for o in obs])
        gap = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        e = self.ego.pose_at(T)
        far = np.hypot(xy[:, 0] - e[0], xy[:, 1] - e[1]) + 10.0     # the car has not moved more than this since
        room = c["merge_m"] + c["merge_rel"] * far[:, None] + np.abs(tt[:, None] - tt[None, :]) * (sp[None, :] + 30.0) * 2.0
        near = np.triu(gap <= room, 1)
        for i in range(n):
            if obs[i].merge_t:                           # a pair that is no longer near stops counting
                for j in range(i + 1, n):
                    if not near[i, j] and obs[j].id not in dead and obs[i].id not in dead:
                        obs[i].merge_t.pop(obs[j].id, None)
            for j in np.nonzero(near[i])[0]:
                a, b = obs[i], obs[j]
                if a.id in dead or b.id in dead:
                    continue
                ta, tb = a.t, b.t
                xb = b.x if abs(ta - tb) < 1e-6 else self.peek(b, ta)[0]
                d = xb[:2] - a.x[:2]
                r = math.hypot(*(a.x[:2] - np.array(self.ego.pose_at(ta)[:2])))
                lim = c["merge_m"] + c["merge_rel"] * r
                la, lb = a.label, b.label
                same_kind = la is None or lb is None or (la in M.VEHICLES) == (lb in M.VEHICLES)
                key = b.id
                dv = float(np.hypot(*(xb[VX:VY + 1] - a.x[VX:VY + 1])))
                sv = math.sqrt(max(float(np.trace(a.P[VX:VY + 1, VX:VY + 1]) + np.trace(b.P[VX:VY + 1, VX:VY + 1])), 0.0))
                if math.hypot(d[0], d[1]) > lim or not same_kind or dv > c["merge_dv"] + 2.0 * sv:
                    a.merge_t.pop(key, None)
                    continue
                t0 = a.merge_t.setdefault(key, T)
                if T - t0 < c["merge_s"]:
                    continue
                rank = lambda o: (o.state != "tentative", o.hits, -o.t0)
                old, new = (a, b) if rank(a) >= rank(b) else (b, a)
                old.hits += new.hits
                old.sources |= new.sources
                for s, t in new.last_seen.items():
                    old.last_seen[s] = max(old.last_seen.get(s, -1e18), t)
                old.logodds = max(old.logodds, new.logodds)
                if old.state == "tentative" and new.state != "tentative":
                    old.state = new.state
                dead.add(new.id)
        if dead:
            for o in obs:
                if o.id in dead:
                    self.store.give(o.slot)
            self.objects = [o for o in obs if o.id not in dead]
