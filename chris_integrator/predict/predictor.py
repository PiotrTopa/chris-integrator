"""The predictor's face for in-process consumers: `Predictor.update(radar_msg, context)` -> `Prediction`.

Python here is glue: it reads the radar message's objects into one array, keeps each object's filter state between
frames (a row per object id), turns the road context into the arrays the native core reads, and wraps what the core
wrote. Every number of the prediction is computed in `native/src/predict.c` (`predict_frame`).
"""
import math

import numpy as np

from .. import native as _native
from ..radar import models as _M
from .selfcheck import SelfCheck

_N = _native.predict
round_lists = _N.round_lists
NAMES, LANES, SLOTS, _W = _N.predict_names()
NO, NS_, NF, NR, NB, NA, NE, NK, KMAX, EMODES = _W
NATIVE_CFG = _N.predict_cfg()                 # {name: default} of the native constants, in the array's order
_NAT_KEYS = tuple(NATIVE_CFG)

# object row (predict.c PO_*), state row, road row, band row, arm row
(O_X, O_Y, O_VX, O_VY, O_AX, O_AY, O_SX, O_SY, O_SVX, O_SVY, O_KIND, O_PSTAT, O_LEN, O_WID, O_HDG, O_HDGSD, O_EXIST,
 O_SEEN, O_IND, O_BRAKE, O_PARKED, O_ID, O_VIS, O_MOTION, O_WITH) = range(25)
R_W, R_Q, R_KERBL, R_KERBR, R_VLIM = range(5)
L_EGO, L_LEFT, L_RIGHT, L_ONC, L_CROSS, L_PARK, L_OFF = range(7)
F_S, F_D, F_SD, F_DC, F_DIR, F_LANE, F_LEAD, F_GAP, F_E, F_R, F_A, F_PSI, F_STOP, F_JDIST, F_KERB, F_OM = range(16)

KINDS = ("car", "van", "truck", "bus", "tram", "motorbike", "bike", "human", None)
_KIND = {k: i for i, k in enumerate(KINDS)}
# width and length of a kind where the radar's models do not say (m): class priors, used while the radar message
# has no `wid` / `len` of the object itself
_WL = {"bike": (0.6, 1.8), "human": (0.6, 0.5), None: (1.8, 4.5)}
CONTROL = {None: 0.0, "none": 0.0, "uncontrolled": 0.0, "give_way": 0.5, "signals": 1.0, "traffic_signals": 1.0, "stop": 2.0}

DEFAULT_CFG = dict(
    enabled=True,
    horizon_s=8.0,               # how far ahead
    step_s=0.25, fine_to_s=3.0,  # samples every step_s up to fine_to_s ...
    step_coarse_s=0.5,           # ... and every step_coarse_s beyond
    objects_max=48,              # objects predicted per frame at most (the nearest; the native core takes 63)
    tentative=False,             # predict tentative objects too
    lane_width_m=3.5,            # where the context does not say
    oncoming_prior=0.6,          # a road is two-way with this probability where nothing says
    park_prior=0.5,              # a strip where vehicles stand beside the outer lane, where nothing says
    park_width_m=2.5,
    learn=True,                  # learn "oncoming traffic exists" / "cars stand on this side" from the objects seen
    decimals=2,                  # of the message's positions and spreads
    selfcheck=dict(enabled=True, every_s=0.5),
    native={},                   # overrides of the native constants (predict_cfg(): modes_max, interact, joint, ...)
)


def _kin_table():
    t = np.zeros((9, NK))
    for i, k in enumerate(KINDS):
        m = _M.KIN.get(k, _M.KIN[None])
        wl = m.get("wl") or _WL.get(k) or _WL[None]
        t[i] = (m["a_lon"], m["a_lat"], m["v_max"], m["v_still"], wl[0], wl[1])
    return t


def sample_times(cfg):
    """The futures' sample times (s from now): every step_s up to fine_to_s, then every step_coarse_s to horizon_s."""
    out, t = [], 0.0
    h, hc, fine, hor = cfg["step_s"], cfg["step_coarse_s"], cfg["fine_to_s"], cfg["horizon_s"]
    while t < min(fine, hor) - 1e-9:
        t += h
        out.append(round(t, 6))
    while t < hor - 1e-9:
        t += hc
        out.append(round(t, 6))
    return np.array(out[:64])


_STD = (2.0, 1.0, 2.0, 1.0)
_IDLE = {"path": "idle"}
_NAN = math.nan


def _D(v, d):
    return d if v is None else v


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _num(v, d=math.nan):
    return float(v) if _fin(v) else d


def _ctr(c):
    """The radar's `ctr` (the object's centre, car axes) as two numbers, NaN where it is not given: the object's place
    on the road (lanes, the path); x_m, y_m (the near face) stay where its futures start."""
    if c.__class__ in (list, tuple) and len(c) == 2 and _fin(c[0]) and _fin(c[1]):
        return float(c[0]), float(c[1])
    return _NAN, _NAN


def _dist(v, n):
    """P(k) for k = 0..n-1 from an int, a probability of "at least one", or a list of probabilities."""
    if isinstance(v, (list, tuple)):
        p = [max(0.0, float(x)) for x in v[:n]] + [0.0] * (n - len(v))
    elif isinstance(v, bool) or v is None:
        p = [0.0, 1.0, 0.0] if v else [1.0, 0.0, 0.0]
    elif isinstance(v, int):
        p = [1.0 if k == min(v, n - 1) else 0.0 for k in range(n)]
    else:
        p = [1.0 - float(v), float(v)] + [0.0] * (n - 2)
    s = sum(p) or 1.0
    return [x / s for x in p[:n]]


def _lane_key(ctx, learned):
    g = ctx.get
    kl, kr = g("kerb_left_m"), g("kerb_right_m")
    nl, nr = g("n_left"), g("n_right")
    return (g("lane_width_m"), tuple(nl) if isinstance(nl, list) else nl, tuple(nr) if isinstance(nr, list) else nr,
            g("oncoming"), g("n_oncoming"), g("park_left"), g("park_right"), kl, kr, g("quality"), g("speed_limit_mps"),
            learned.get("oncoming"), learned.get("park_left"), learned.get("park_right"))


def lane_arrays(ctx, cfg, right_hand=True, learned=None):
    """The lanes of the road context as the native core reads them: (road (NR), bands (nb, NB)).

    The lanes beside the car's own are hypotheses: how many lanes run the car's way on its left and right, whether
    oncoming lanes follow, whether vehicles stand beside the outer lane. Each hypothesis puts bands at offsets from
    the centre line; a band carries the probability of the hypotheses that put it there. A measured road edge
    weighs the hypotheses by where each would have the edge."""
    learned = learned or {}
    w = _num(ctx.get("lane_width_m"), cfg["lane_width_m"])
    if not 2.2 <= w <= 5.5:
        w = cfg["lane_width_m"]
    has_path = ctx.get("centre") is not None and len(ctx["centre"]) >= 2
    q = _num(ctx.get("quality"), 0.6 if has_path else 0.15)
    pw = cfg["park_width_m"]
    # the side of the oncoming traffic: left where traffic keeps right
    sides = {}
    for name, sg in (("left", 1.0), ("right", -1.0)):
        onc_side = (sg > 0) == bool(right_hand)
        same = _dist(ctx.get("n_" + name, learned.get("n_" + name, 0.3)), 3)
        po = 0.0
        if onc_side:
            po = ctx.get("oncoming")
            po = float(po) if _fin(po) or isinstance(po, bool) else learned.get("oncoming", cfg["oncoming_prior"])
        n_onc = int(ctx.get("n_oncoming") or 1)
        pk = ctx.get("park_" + name)
        pk = float(pk) if _fin(pk) or isinstance(pk, bool) else learned.get("park_" + name, cfg["park_prior"])
        hyp = []                                   # (probability, same lanes, oncoming lanes, kerb offset)
        for k, pk_same in enumerate(same):
            for onc, p_o in ((n_onc, po), (0, 1.0 - po)):
                if pk_same * p_o > 1e-4:
                    hyp.append([pk_same * p_o, k, onc, 0.5 * w + (k + onc) * w])
        kerb = ctx.get("kerb_%s_m" % name)
        if _fin(kerb):
            kerb = abs(float(kerb))
            for h in hyp:
                h[0] *= math.exp(-0.5 * ((h[3] - kerb) / 1.2) ** 2) + 0.02
            s = sum(h[0] for h in hyp)
            for h in hyp:
                h[0] /= s
        sides[name] = (sg, hyp, pk, kerb if _fin(kerb) else sum(h[0] * h[3] for h in hyp))
    bands = [(-0.5 * w, 0.5 * w, L_EGO, 1.0, 1.0, 0.0)]
    for name, (sg, hyp, pk, kerb) in sides.items():
        acc = {}
        for p, k, onc, edge in hyp:
            for j in range(k):
                key = (0.5 * w + j * w, 0.5 * w + (j + 1) * w, L_LEFT if sg > 0 else L_RIGHT, 1.0)
                acc[key] = acc.get(key, 0.0) + p
            for j in range(onc):
                key = (0.5 * w + (k + j) * w, 0.5 * w + (k + j + 1) * w, L_ONC, -1.0)
                acc[key] = acc.get(key, 0.0) + p
            key = (edge, edge + pw, L_PARK, 0.0)
            acc[key] = acc.get(key, 0.0) + p * pk
        for (lo, hi, ty, d), p in acc.items():
            if p > 1e-3:
                bands.append((sg * lo, sg * hi, ty, p, d, 0.0) if sg > 0 else (sg * hi, sg * lo, ty, p, d, 0.0))
    road = np.zeros(NR)
    road[R_W], road[R_Q] = w, min(1.0, max(0.0, q))
    road[R_KERBL], road[R_KERBR] = sides["left"][3], -sides["right"][3]
    road[R_VLIM] = _num(ctx.get("speed_limit_mps"))
    return road, np.array(bands, dtype=float).reshape(-1, NB)


_NONE2, _NONEA = np.zeros((0, 2)), np.zeros((0, NA))


def path_arrays(ctx):
    """The geometry of the road context as the native core reads it: (centre (m, 2), arm_meta (na, NA), arm_pts
    (np, 2), xing (nx, 2), has_path)."""
    centre = ctx.get("centre")
    has_path = centre is not None and len(centre) >= 2
    centre = np.ascontiguousarray(centre, dtype=float).reshape(-1, 2) if has_path else _NONE2
    if len(centre) > 200:
        centre = np.ascontiguousarray(centre[:: (len(centre) + 199) // 200])
    arms, xs = ctx.get("arms"), ctx.get("crossings")
    if not arms and not xs:
        return centre, _NONEA, _NONE2, _NONE2, has_path
    meta, pts = [], []
    for a in (arms or ())[:32]:
        p = a.get("pts") if isinstance(a, dict) else None
        if p is None or len(p) < 2:
            continue
        side = a.get("side")
        meta.append((len(pts), len(p), _num(a.get("s_m")), 1.0 if side == "left" or side == 1 else -1.0,
                     _num(a.get("priority"), 0.0), 1.0 if a.get("entry", True) else 0.0,
                     1.0 if a.get("exit", True) else 0.0, CONTROL.get(a.get("control"), 0.0)))
        pts.extend(p.tolist() if isinstance(p, np.ndarray) else ((float(x), float(y)) for x, y in p))
    xing = [(_num(x.get("s_m")), _num(x.get("half_m"), 2.0)) for x in (xs or ())
            if isinstance(x, dict) and _fin(x.get("s_m"))]
    return (centre, np.array(meta, dtype=float).reshape(-1, NA), np.array(pts, dtype=float).reshape(-1, 2),
            np.array(xing, dtype=float).reshape(-1, 2), has_path)


def road_arrays(ctx, cfg, right_hand=True, learned=None):
    """The road context (the dict documented in chris_integrator/predict/__init__.py) as the native core reads it:
    (centre (m, 2), road (NR), bands (nb, NB), arm_meta (na, NA), arm_pts (np, 2), xing (nx, 2), has_path)."""
    ctx = ctx or {}
    road, bands = lane_arrays(ctx, cfg, right_hand, learned)
    centre, meta, pts, xing, has_path = path_arrays(ctx)
    return centre, road, bands, meta, pts, xing, has_path


class EgoTrack:
    """The car's pose over the last seconds, integrated from the radar message's own speed and yaw rate: what turns
    a place of an earlier frame into this frame's (the context's messages, the self-check's stored predictions)."""

    def __init__(self, keep_s=20.0):
        self.keep = keep_s
        self.reset()

    def reset(self):
        self.t, self.p = [], []            # times, (X, Y, psi, v, w)

    def add(self, t, v, w):
        if self.t and t <= self.t[-1]:
            if t < self.t[-1] - 2.0:
                self.reset()               # the clock went back: a new ride
            else:
                return
        if not self.t or t - self.t[-1] > 2.0:
            self.t, self.p = [t], [(0.0, 0.0, 0.0, v, w)]
            return
        X, Y, psi, v0, w0 = self.p[-1]
        dt = t - self.t[-1]
        vm, wm = 0.5 * (v + v0), 0.5 * (w + w0)
        pm = psi + 0.5 * wm * dt
        self.t.append(t)
        self.p.append((X + vm * dt * math.cos(pm), Y + vm * dt * math.sin(pm), psi + wm * dt, v, w))
        while self.t[-1] - self.t[0] > self.keep:
            del self.t[0], self.p[0]

    def pose(self, t):
        """(X, Y, psi) at t in the track's own ground frame (between samples: interpolated; beyond: carried on)."""
        ts = self.t
        if not ts:
            return 0.0, 0.0, 0.0
        if t >= ts[-1] or len(ts) == 1 or t <= ts[0]:
            X, Y, psi, v, w = self.p[-1] if t >= ts[-1] else self.p[0]
            dt = t - (ts[-1] if t >= ts[-1] else ts[0])
            dt = max(-1.0, min(1.0, dt))
            pm = psi + 0.5 * w * dt
            return X + v * dt * math.cos(pm), Y + v * dt * math.sin(pm), psi + w * dt
        lo, hi = 0, len(ts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        u = (t - ts[lo]) / (ts[hi] - ts[lo])
        a, b = self.p[lo], self.p[hi]
        return a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2])

    def to_ground(self, xy, t):
        """Points (n, 2) of the car frame of t, in the track's ground frame."""
        X, Y, p = self.pose(t)
        c, s = math.cos(p), math.sin(p)
        out = np.empty_like(xy)
        out[:, 0] = X + c * xy[:, 0] - s * xy[:, 1]
        out[:, 1] = Y + s * xy[:, 0] + c * xy[:, 1]
        return out

    def from_ground(self, g, t):
        """Ground points (..., 2) in the car frame of t."""
        X, Y, p = self.pose(t)
        c, s = math.cos(p), math.sin(p)
        out = np.empty_like(g)
        gx, gy = g[..., 0] - X, g[..., 1] - Y
        out[..., 0] = c * gx + s * gy
        out[..., 1] = -s * gx + c * gy
        return out

    def move(self, xy, t_from, t_to):
        """Points (n, 2) given in the car frame of t_from, in the car frame of t_to."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if t_from == t_to or not self.t:
            return xy
        X0, Y0, p0 = self.pose(t_from)
        X1, Y1, p1 = self.pose(t_to)
        c0, s0, c1, s1 = math.cos(p0), math.sin(p0), math.cos(p1), math.sin(p1)
        gx = X0 + c0 * xy[:, 0] - s0 * xy[:, 1] - X1
        gy = Y0 + s0 * xy[:, 0] + c0 * xy[:, 1] - Y1
        return np.stack([c1 * gx + s1 * gy, -s1 * gx + c1 * gy], axis=1)


class Prediction:
    """One frame's prediction, as arrays (the message form: `message()`).

        t          (N,)              sample times, s from now, shared by all futures
        ids        [n]               the radar's object ids, in the arrays' order
        n_modes    (n,) int          futures of each object (<= K = 8)
        p, p0      (n, K)            their probabilities: p with the interaction among all road users AND the car's
                                     own futures taken into account, p0 without the car (nobody is taken to give
                                     way to it); each row sums to 1 over its n_modes
        m          (n, K) int        manoeuvre name code (`names[m]`), slot (n, K) int the intent filter's slot
        traj       (n, K, N, 6)      x, y (m, the car frame of this message's time, fixed on the ground; x forward,
                                     y left), sx, sy (1 sigma, m), rxy (their correlation), v (speed, m/s)
        lane       (n, 7)            probabilities of `lanes` = (ego, left, right, onc, cross, park, off)
        intent     (n, 15)           the intent filter's probabilities per slot (`slots`), before interaction
        feat       (n, 16)           what was measured of each object in the road's frame (F_* in predictor.py)
        ego_n, ego_p (3,), ego_m (3,) int, ego_traj (3, N, 3): the car's own futures: x, y, v
        stats      (8,)              pair tables built, interacting groups, largest group, search nodes, groups
                                     solved by belief propagation, objects, futures, the car's leader (index, -1)
    """
    names, lanes, slots = NAMES, LANES, SLOTS

    def __init__(self, ts, t, ids, modes, traj, lane, intent, feat, ego_n, ego_out, ego_traj, stats, dec, source):
        self.ts_mono_ns, self.t, self.ids = ts, t, ids
        self._modes = modes
        self.n_modes = modes[:, 0, 0].astype(int)
        self.slot = modes[:, 1].astype(int)
        self.m = modes[:, 2].astype(int)
        self.p, self.p0 = modes[:, 3], modes[:, 4]
        self.traj, self.lane, self.intent, self.feat = traj, lane, intent, feat
        self.ego_n, self.ego_p, self.ego_m, self.ego_traj = ego_n, ego_out[:, 0], ego_out[:, 1].astype(int), ego_traj
        self.stats, self._dec, self.source = stats, dec, source
        self._index = self._f = self._lane = self._intent = None

    def index(self, oid):
        if self._index is None:
            self._index = {i: k for k, i in enumerate(self.ids)}
        return self._index.get(oid)

    def intent_names(self, i):
        """{manoeuvre name: probability} of object i: the intent filter's slots under the names this object's
        futures are published with (a slot without a future keeps its slot name)."""
        if self._intent is None:
            self._intent = (np.round(self.intent, 3).tolist(), self.slot.tolist(), self.m.tolist(), self.n_modes.tolist())
        rows, slot, mm, nm = self._intent
        named = {slot[i][k]: NAMES[mm[i][k]] for k in range(nm[i])}
        out = {}
        for s, v in enumerate(rows[i]):
            if v > 0.0005:
                k = named.get(s) or SLOTS[s]
                out[k] = round(out[k] + v, 3) if k in out else v
        return out

    def fut(self, i):
        """The message form of object i's futures (rounded lists)."""
        return self._futs()[i]

    def _futs(self):
        if self._f is None:
            self._f = _N.predict_futs(self.traj, self._modes, NAMES, self._dec)
        return self._f

    def ego_fut(self):
        n = self.ego_n
        if not n:
            return {"modes": []}
        tr = self.ego_traj[:n]
        flat = round_lists(np.ascontiguousarray(tr.transpose(0, 2, 1)).reshape(-1, tr.shape[1]), self._dec, False)
        p = np.round(self.ego_p[:n], 3).tolist()
        return {"modes": [{"p": p[k], "m": NAMES[self.ego_m[k]], "x": flat[3 * k], "y": flat[3 * k + 1], "v": flat[3 * k + 2]}
                          for k in range(n)]}

    def lane_dict(self, i):
        if self._lane is None:
            self._lane = np.round(self.lane, 3).tolist()
        return dict(zip(LANES, self._lane[i]))

    def message(self, radar_msg):
        """`radar_fut`: the radar message of the same stamp with the prediction added. Top level `fut` {"t"},
        `ego_fut` {"modes"}, `fut_ctx` (what the road context was made of); per object `fut` [modes], `lane`,
        `intent`. Objects the predictor did not take (beyond `objects_max`, tentative) are passed on as they are."""
        futs = self._futs()
        objs = []
        for o in radar_msg.get("objects") or ():
            i = self.index(o.get("id"))
            if i is not None:
                o = dict(o, fut=futs[i], lane=self.lane_dict(i), intent=self.intent_names(i))
            objs.append(o)
        out = dict(radar_msg, type="radar_fut", objects=objs, fut={"t": self.t.tolist()}, ego_fut=self.ego_fut(),
                   fut_ctx=self.source)
        return out


class Predictor:
    """Futures of every radar object and of the car, frame by frame.

        p = Predictor(cfg)                       # cfg: keys of DEFAULT_CFG; cfg["native"]: the native constants
        pred = p.update(radar_msg, context)      # a Prediction (arrays), or None for a message without a time
        pred.message(radar_msg)                  # the `radar_fut` message
        p.health()                               # counts and the self-check's running scores

    `context`: the road as far as it is known, in the car frame of the radar message (see the package's docstring
    for the dict; `ContextBuilder` makes it from the stack's messages), or None: then the road is the arc of the
    car's own speed and yaw rate and the free (kinematic) futures carry more weight."""

    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        self.cfg = dict(DEFAULT_CFG, **{k: v for k, v in cfg.items() if k in DEFAULT_CFG and k not in ("native", "selfcheck")})
        nat = dict(NATIVE_CFG)
        for k, v in (cfg.get("native") or {}).items():
            if k not in nat:
                raise KeyError("predict: no native constant %r" % k)
            nat[k] = float(v)
        nat["step_s"] = float(self.cfg["step_s"])
        self.native_cfg = nat
        self._cfg = np.array([nat[k] for k in _NAT_KEYS])
        self.right_hand = nat["right_hand"] >= 0.0
        self.times = sample_times(self.cfg)
        self._times = self.times.tolist()
        self._kin = _kin_table()
        self.ego = EgoTrack()
        self.check = SelfCheck(self.times, dict(DEFAULT_CFG["selfcheck"], **(cfg.get("selfcheck") or {})))
        self.reset()

    def reset(self, reason=None):
        self._rows = {}                       # object id -> row of the state array
        self._S = np.zeros((64, NS_))
        self._free = list(range(63, -1, -1))
        self.ego.reset()
        self.check.reset()
        self.frames = self.idle_frames = 0
        self.last = None
        self._learn = {"oncoming": None, "park_left": None, "park_right": None}
        self._lane_key = self._lanes = self._lane_nopath = None
        self._ground = None

    # ---- the radar message's objects as the native core reads them ------------------------------------------
    def _objects(self, msg):
        c = self.cfg
        objs = [o for o in msg.get("objects") or () if isinstance(o, dict) and _fin(o.get("x_m")) and _fin(o.get("y_m"))
                and (c["tentative"] or o.get("state") != "tentative")]
        if len(objs) > c["objects_max"]:
            objs = sorted(objs, key=lambda o: o["x_m"] * o["x_m"] + o["y_m"] * o["y_m"])[:c["objects_max"]]
        n = len(objs)
        a = np.full((n, NO), math.nan)
        ids = []
        kind = _KIND
        for i, o in enumerate(objs):
            g = o.get
            std = g("std") or _STD
            cls = g("class")
            # the estimation core's fields when the radar sends them: extent, body direction, visibility
            hd, hs = g("hdg"), g("hdg_std")
            if hd.__class__ is list:                           # [value, std]
                hd, hs = hd[0], hd[1] if len(hd) > 1 else None
            ind, br, vis = g("indicator"), g("brake_lights"), g("vis")
            fl, mo = g("flags") or (), g("motion")              # the radar's own verdicts
            a[i] = (o["x_m"], o["y_m"], g("vx_mps") or 0.0, g("vy_mps") or 0.0, g("ax_mps2") or 0.0, g("ay_mps2") or 0.0,
                    std[0], std[1], std[2], std[3], kind.get(cls.get("label"), 8) if cls else 8,
                    _D(g("p_stationary"), 0.5), _D(g("len"), _NAN), _D(g("wid"), _NAN), _D(hd, _NAN), _D(hs, _NAN),
                    _D(g("exist"), 1.0), 0.0 if g("state") == "memory" else 1.0,
                    _NAN if ind is None else 1.0 if ind == "left" else -1.0 if ind == "right" else 0.0,
                    _NAN if br is None else 1.0 if br else 0.0, 1.0 if "parked" in fl else 0.0, i,
                    _NAN if vis is None else 0.0 if vis in ("occluded", "hidden", False) else 1.0,
                    1.0 if mo == "stationary" else -1.0 if mo == "moving" else 0.0, 1.0 if "with_traffic" in fl or "oncoming" in fl else 0.0,
                    *_ctr(g("ctr")))
            ids.append(g("id"))
        if n:
            num = a[:, :12]
            if not np.isfinite(num).all():                     # a number that is none: the object's row is made harmless
                bad = ~np.isfinite(num).all(axis=1)
                a[bad, 2:6], a[bad, 6:10], a[bad, 11] = 0.0, (2.0, 1.0, 2.0, 1.0), 0.5
            np.maximum(a[:, 6:10], 0.05, out=a[:, 6:10])
        return a, ids, objs

    def _state(self, ids):
        rows, S = self._rows, self._S
        live = set(ids)
        for k in [k for k in rows if k not in live]:
            S[rows[k]] = 0.0
            self._free.append(rows.pop(k))
        idx = []
        for k in ids:
            r = rows.get(k)
            if r is None:
                if not self._free:
                    S = self._S = np.concatenate([S, np.zeros_like(S)])
                    self._free = list(range(len(S) - 1, len(S) // 2 - 1, -1))
                r = rows[k] = self._free.pop()
            idx.append(r)
        return idx

    def _learned(self, a, lane, feat):
        """What the objects themselves have shown of the road: oncoming traffic on the left, vehicles standing on a
        side. Slow counters, used only where the context does not say."""
        L = self._learn
        if not self.cfg["learn"] or not len(a) or self.frames % 10:
            return
        seen = a[:, O_SEEN] > 0.5
        sp = np.hypot(a[:, O_VX], a[:, O_VY])
        onc = seen & (sp > 3.0) & (np.abs(np.abs(feat[:, F_PSI]) - math.pi) < 0.5) & (np.abs(feat[:, F_D]) < 12.0)
        L["oncoming"] = min(0.95, (L["oncoming"] or self.cfg["oncoming_prior"]) + 0.2) if onc.any() else L["oncoming"]
        for name, sg in (("park_left", 1.0), ("park_right", -1.0)):
            st = seen & (a[:, O_PSTAT] > 0.7) & (a[:, O_KIND] < 6) & (sg * feat[:, F_D] > 2.0) & (sg * feat[:, F_D] < 9.0)
            if st.any():
                L[name] = min(0.9, (L[name] or self.cfg["park_prior"]) + 0.1)

    def learned(self):
        return {k: v for k, v in self._learn.items() if v is not None}

    def update(self, radar_msg, context=None):
        ts = radar_msg.get("ts_mono_ns")
        eg = radar_msg.get("ego") or {}
        if not isinstance(ts, int) or not self.cfg["enabled"]:
            return None
        T = ts / 1e9
        gf = eg.get("ground_frame")
        if gf != self._ground:                   # the radar started a new ground frame: its objects are new ones
            if self._ground is not None:
                self.reset()
            self._ground = gf
        v, w, acc = _num(eg.get("speed_mps"), 0.0), _num(eg.get("yaw_rate_rps"), 0.0), _num(eg.get("accel_mps2"), 0.0)
        self.ego.add(T, v, w)
        a, ids, objs = self._objects(radar_msg)
        n = len(ids)
        idx = self._state(ids)
        st = np.ascontiguousarray(self._S[idx]) if n else np.zeros((0, NS_))
        context = context or {}
        learned = self.learned()
        key = _lane_key(context, learned)                 # the lanes change seldom: their arrays are kept
        if key != self._lane_key or (context.get("centre") is None) != self._lane_nopath:
            self._lane_key, self._lane_nopath = key, context.get("centre") is None
            self._lanes = lane_arrays(context, self.cfg, self.right_hand, learned)
        road, bands = self._lanes
        centre, ameta, apts, xing, has_path = path_arrays(context)
        sig = context.get("turn_signal")
        ego = np.array([v, w, acc, 1.0 if sig == "left" else -1.0 if sig == "right" else 0.0, 1.0 if has_path else 0.0,
                        0.0, 0.0, 0.0])
        nt = len(self.times)
        lane, intent, feat = np.zeros((n, 7)), np.zeros((n, len(SLOTS))), np.zeros((n, NF))
        modes, traj = np.zeros((n, 5, KMAX)), np.full((n, KMAX, nt, 6), math.nan)
        ego_out, ego_traj, stats = np.zeros((EMODES, 2)), np.zeros((EMODES, nt, 3)), np.zeros(8)
        ego_n = _N.predict_frame(self._cfg, self.times, ego, road, bands, ameta, apts, xing, centre, self._kin, a, st, T,
                                 lane, intent, feat, modes, traj, ego_out, ego_traj, stats)
        if n:
            self._S[idx] = st
        self._learned(a, lane, feat)
        src = {"path": context.get("source") or ("given" if has_path else "kinematic"),
               "quality": round(float(road[R_Q]), 2), "arms": len(ameta), "lane_width_m": round(float(road[R_W]), 2)}
        pred = Prediction(ts, self.times, ids, modes, traj, lane, intent, feat, ego_n, ego_out, ego_traj, stats,
                          self.cfg["decimals"], src)
        pred.kind = a[:, O_KIND].astype(int) if n else np.zeros(0, int)
        pred.seen = a[:, O_SEEN] > 0.5 if n else np.zeros(0, bool)
        pred.xy = a[:, :2]
        self.check.update(T, pred, self.ego)
        self.frames += 1
        self.last = pred
        return pred

    def idle(self, radar_msg):
        """A radar message without an object to predict: the car's pose is kept up, the objects' states are let go,
        predictions kept for the self-check come due. Returns the `radar_fut` message of such a frame: no futures,
        `ego_fut` without modes, `fut_ctx.path: "idle"`. (`update` gives the car's futures also then; this is the
        cheap way, for a plug-in whose every call costs.)"""
        ts = radar_msg.get("ts_mono_ns")
        if not isinstance(ts, int) or not self.cfg["enabled"]:
            return None
        eg = radar_msg.get("ego") or {}
        T = ts / 1e9
        gf = eg.get("ground_frame")
        if gf != self._ground:
            if self._ground is not None:
                self.reset()
            self._ground = gf
        self.ego.add(T, _num(eg.get("speed_mps"), 0.0), _num(eg.get("yaw_rate_rps"), 0.0))
        if self._rows:
            self._state(())
        self.check.update(T, None, self.ego)
        self.frames += 1
        self.idle_frames += 1
        return dict(radar_msg, type="radar_fut", fut={"t": self._times}, ego_fut={"modes": []}, fut_ctx=_IDLE)

    def health(self):
        h = {"frames": self.frames, "idle_frames": self.idle_frames, "objects": len(self._rows), "learned": {k: round(v, 2) for k, v in self.learned().items()}}
        if self.last is not None:
            s = self.last.stats
            h.update(futures=int(s[6]), groups=int(s[1]), largest_group=int(s[2]), context=self.last.source)
        h["selfcheck"] = self.check.stats()
        return h
