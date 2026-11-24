"""The futures the decision layer judges: the objects' predicted modes and the car's own, on one time grid.

Contract with the prediction layer (chris_integrator/predict, fixed 2026-11-23). Frame: the car frame of the
message's output time, held fixed on the ground (x forward, y left, m). In the `radar` message, or in a
companion `radar_fut` message of the same `ts_mono_ns`:

    top level  "fut": {"t": [t1..tN]}                      seconds from now, shared by all modes
               "ego_fut": {"modes": [{"p", "m", "x":[N], "y":[N], "v":[N]}]}
    per object "fut": [{"p": conditioned on interaction, "p0": unconditioned, "m": manoeuvre name,
                        "x":[N], "y":[N], "sx":[N], "sy":[N], "rxy":[N] (correlation), "v":[N]}, ...]
               "lane": {"ego", "left", "right", "onc", "cross", "park", "off"}, "intent"
    (`radar_fut`: the same top level, and "objects": [{"id", "fut", "lane", "intent"}])
    in process: chris_integrator.predict.Predictor.update(radar_msg, context) -> the same, arrays for lists

An object's x, y are the radar's reference point of it (its near face: where the line of sight from the car's
origin to its centre enters its footprint, as `x_m`, `y_m`); its extent comes from `len`, `wid`, `hdg` (with their uncertainties) where the radar gives them,
else from its class's prior size (CLASS_SIZE). The footprint's centre is the radar's `ctr` where given (its offset
from the reference point, `centre_offset`); else half the footprint's depth beyond the reference point along the
line of sight.

Degraded mode: an object without `fut` gets constant-velocity futures built from its `radar` fields (`cv_futures`:
by default the radar's own two hypotheses, it stands with p_stationary and it drives on at its velocity with the
rest; `cv_modes: "single"` one mode), and without `ego_fut` the car gets its own modes from its present motion and the road model's
plan (`ego_futures`). Both are reported (`FrameFutures.degraded`) and in the decision layer's health."""
import math

import numpy as np

from .. import native as _native
from ..core import classvote

LANES = _native.predict.predict_names()[1]

# prior footprint by class: length, width, 1 sigma of each (m)
CLASS_SIZE = {
    "car": (4.5, 1.8, 0.4, 0.1), "van": (5.2, 2.0, 0.5, 0.15), "truck": (9.0, 2.5, 3.0, 0.15),
    "bus": (12.0, 2.55, 1.5, 0.1), "tram": (25.0, 2.65, 5.0, 0.1), "human": (0.6, 0.6, 0.15, 0.15),
    "bike": (1.8, 0.7, 0.2, 0.15), "motorbike": (2.1, 0.85, 0.2, 0.15), None: (4.2, 1.8, 1.0, 0.3),
}
VEHICLES = ("car", "van", "truck", "bus", "tram")
VRU = ("human", "bike", "motorbike")
KIND_SET = {"vehicle": 0, "vru": 1, "unknown": 2}     # threshold sets of the native layer (threat.py THRESHOLDS)

DEFAULT_CFG = dict(
    grid_dt_s=0.25, horizon_s=5.0,        # the degraded futures' grid (a contract grid replaces it)
    cv_speed_cap_mps=60.0,
    cv_v_std_cap_mps=6.0,                 # the radar's velocity spread is capped at this in the degraded future
    cv_still_v_std_mps=0.15,               # an object the radar calls stationary: velocity 0 with this spread
    cv_along_path=True,                   # an object that drives along the path (or against it) follows its bends
    # "radar": the radar's own hypotheses as two constant-velocity modes (it stands: p_stationary, v = 0; it
    # drives: the rest, its velocity); "single": one mode, the radar's velocity (a stationary verdict: v = 0)
    cv_modes="radar", cv_stand_min=0.02,
    along_heading_deg=25.0, along_min_mps=2.0,
    ego_path_p=0.85,                      # the car follows the road model's plan with this probability ...
    ego_accel_clip=(-8.0, 3.0),           # ... at its present acceleration, clipped
    plan_max_age_s=1.0,
)


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def kind_of(label):
    label = classvote.norm_label(label)
    if label in VEHICLES:
        return "vehicle"
    if label in VRU:
        return "vru"
    return "unknown"


def _vs(o, k):
    """A value with its 1 sigma from the radar's object: a number + `<k>_std`, a [value, std] pair, or a
    {"value"/"v", "std"} dict. (value, std) with None where absent."""
    v = o.get(k)
    if isinstance(v, (list, tuple)) and len(v) == 2 and _fin(v[0]):
        return float(v[0]), (float(v[1]) if _fin(v[1]) else None)
    if isinstance(v, dict):
        val = v.get("value", v.get("v"))
        sd = v.get("std")
        return (float(val) if _fin(val) else None), (float(sd) if _fin(sd) else None)
    if _fin(v):
        sd = o.get(k + "_std", o.get("std_" + k))
        return float(v), (float(sd) if _fin(sd) else None)
    return None, None


def extent(o):
    """(len, wid, sd_len, sd_wid, hdg or nan, sd_hdg or nan, label, source) of a radar object."""
    cls = o.get("class")
    label = classvote.norm_label(cls.get("label")) if isinstance(cls, dict) else None
    L0, W0, sL0, sW0 = CLASS_SIZE.get(label, CLASS_SIZE[None])
    L, sL = _vs(o, "len")
    W, sW = _vs(o, "wid")
    h, sh = _vs(o, "hdg")
    es = o.get("ext_std")                   # the radar's 1 sigma of len, wid, hdg (docs/TOPICS.md): the heading's
    if sh is None and h is not None and isinstance(es, (list, tuple)) and len(es) >= 3 and _fin(es[2]):
        sh = float(es[2])                   # (len, wid keep the class's spread: the box's length is a view's)
    src = "radar" if (L is not None and W is not None) else "class"
    L = L if L is not None and 0.1 <= L <= 40.0 else L0
    W = W if W is not None and 0.1 <= W <= 5.0 else W0
    sL = sL if sL is not None and sL >= 0.0 else sL0
    sW = sW if sW is not None and sW >= 0.0 else sW0
    if h is None and _fin(o.get("heading_rad")):
        # the radar's direction of travel, only where the speed stands clear of its spread (a slow, noisy object's
        # heading is noise, and a footprint turned by it is wider than the car)
        sd = o.get("std") or ()
        sv = max(float(sd[2]), float(sd[3])) if len(sd) >= 4 and _fin(sd[2]) and _fin(sd[3]) else math.inf
        sp = o.get("speed_mps") if _fin(o.get("speed_mps")) else math.hypot(float(o.get("vx_mps") or 0.0),
                                                                            float(o.get("vy_mps") or 0.0))
        if sp >= max(2.0, 2.0 * sv):
            h = float(o["heading_rad"])
    return L, W, sL, sW, (h if h is not None else math.nan), (sh if sh is not None else math.nan), label, src


def centre_offset(o):
    """The radar's footprint centre (`ctr`) less its reference point (`x_m`, `y_m`, the footprint's near face),
    car axes, or None where the radar does not give it. The futures carry the reference point; the footprint is
    drawn this far from it (turning with the object), not reconstructed from the line of sight."""
    c = o.get("ctr")
    if not (isinstance(c, (list, tuple)) and len(c) == 2 and _fin(c[0]) and _fin(c[1])
            and _fin(o.get("x_m")) and _fin(o.get("y_m"))):
        return None
    dx, dy = float(c[0]) - float(o["x_m"]), float(c[1]) - float(o["y_m"])
    return (dx, dy) if math.hypot(dx, dy) <= 25.0 else None


def stands(o):
    """The radar says the object stands (its verdict, or p_stationary at least one half and not called moving)."""
    if o.get("motion") == "moving":
        return False
    ps = o.get("p_stationary")
    return o.get("motion") == "stationary" or "parked" in (o.get("flags") or ()) or (_fin(ps) and ps >= 0.5)


def heading_from_motion(o):
    """The radar's heading of the object comes from its own travel (a direction of travel, at a speed clear of its
    spread)."""
    if not _fin(o.get("heading_rad")):
        return False
    sd = o.get("std") or ()
    sv = max(float(sd[2]), float(sd[3])) if len(sd) >= 4 and _fin(sd[2]) and _fin(sd[3]) else math.inf
    sp = o.get("speed_mps") if _fin(o.get("speed_mps")) else math.hypot(float(o.get("vx_mps") or 0.0),
                                                                        float(o.get("vy_mps") or 0.0))
    return sp >= max(2.0, 2.0 * sv)


def travel_heading(o):
    """(heading, 1 sigma) of the radar's direction of travel (where heading_from_motion): the velocity's spread
    across it over the speed."""
    sd = o.get("std") or ()
    sv = max(float(sd[2]), float(sd[3])) if len(sd) >= 4 and _fin(sd[2]) and _fin(sd[3]) else 0.0
    sp = o.get("speed_mps") if _fin(o.get("speed_mps")) else math.hypot(float(o.get("vx_mps") or 0.0),
                                                                        float(o.get("vy_mps") or 0.0))
    return float(o["heading_rad"]), math.atan2(sv, max(sp, 1e-6))


def heading_resolved(sh, resolved):
    """The radar's heading of the box is a measurement of the object's heading: its stated 1 sigma at most
    `resolved` (or not stated). A box fitted to a vehicle seen whole states its heading within about 0.1 rad; one
    fitted to a partial view (side-on beside the car, cut by the field's edge) states 0.2-0.8 rad and is often a
    quarter turn off (a car parked along the road drawn across it)."""
    return not (math.isfinite(sh) and sh > resolved)


def road_axis(h, sh, road, snap, sd):
    """(heading, sd) of a standing object: the road's axis at its place (either direction, whichever is nearer the
    stated heading) when the stated heading is within `snap` of it or not known; else as stated."""
    if not math.isfinite(h):
        return road, sd
    d = (h - road + math.pi / 2.0) % math.pi - math.pi / 2.0          # to the axis, modulo a half turn
    if abs(d) > snap:
        return h, sh
    return h - d, (min(sh, sd) if math.isfinite(sh) else sd)


def in_path_weight(o, lane, floor):
    """The weight of an object's standing modes from the prediction's lane probabilities: P(ego) + P(cross), at
    least `floor`; 1 without lanes, or where the radar's corridor test puts it surely in the path."""
    fl = o.get("flags") or ()
    if "in_path" in fl or "threat" in fl or not isinstance(lane, dict):
        return 1.0
    vals = {k: v for k, v in lane.items() if _fin(v)}
    if sum(vals.values()) < 0.5:
        return 1.0
    return min(1.0, max(floor, float(vals.get("ego", 0.0)) + float(vals.get("cross", 0.0))))


def along_lane_p(lane):
    """The prediction's probability that the object is in a lane along the road (the car's, beside it, oncoming); 0
    without lanes."""
    if not isinstance(lane, dict):
        return 0.0
    return float(sum(lane.get(k, 0.0) for k in ("ego", "left", "right", "onc") if _fin(lane.get(k, 0.0))))


def mode_stands(x, y, move_m):
    """A mode whose place moves less than move_m over the horizon."""
    return float(np.max(np.hypot(np.asarray(x) - x[0], np.asarray(y) - y[0]))) < move_m


class FrameFutures:
    """One frame's futures on one grid t (t[0] = 0): the car's modes (E x [p, x, y, v]) and per object a list of
    modes (p0, p, x, y, sx, sy, rxy). `degraded`: which parts came from the adapters."""

    def __init__(self, t):
        self.t = np.asarray(t, float)
        self.ego = []                     # (name, p, x, y, v)
        self.objects = []                 # (radar object, [(name, p0, p, x, y, sx, sy, rxy)], source)
        self.degraded = {"ego": False, "objects": 0, "invalid": 0}
        self.lane = {}                    # object id -> the prediction's lane probabilities
        self.intent = {}
        self.road_hdg = {}                # object id -> the path's direction at its place (an unknown heading)
        self.along = set()                # ids the adapter judged to drive along the path or against it (cv_path)


def _grid(top):
    t = (top or {}).get("t") if isinstance(top, dict) else None
    if t is None:
        return None
    try:
        t = np.asarray(t, float).ravel()
    except (TypeError, ValueError):
        return None
    if len(t) < 1 or not np.all(np.isfinite(t)) or (len(t) > 1 and not np.all(np.diff(t) > 0)) or t[0] < 0.0:
        return None
    return t


def _arr(m, k, n):
    v = m.get(k)
    if v is None:
        return None
    try:
        a = np.asarray(v, float).ravel()
    except (TypeError, ValueError):
        return None
    return a if len(a) == n and np.all(np.isfinite(a)) else None


class Futures:
    """The contract's futures of one frame as arrays: what the prediction layer hands the decision layer in process
    (`Prediction.handoff()`, no message made) and what `from_message` reads out of a `radar` / `radar_fut` message.
    The numbers are the message's (rounded as it rounds them), so a decision is the same either way.

        t          (N,) the grid, s from now; None: no grid (then no futures count)
        ego        (E, 1 + 3 N): p (NaN: not given), x, y, v of the car's futures; None: none usable
        ego_names  [E]; ego_given: the message had an `ego_fut`
        modes      (K, 2 + 5 N): p0, p (NaN: not given), x, y, sx, sy, rxy (a row of NaN: not given)
        names      [K]
        rows       (R, 3): first mode, modes, state (0 no futures, 1 as the contract says, 2 not: the adapter
                   stands in and it is counted)
        lanes      None | [R] of the message's lane dicts (or None) | (R, 7) probabilities of LANES
        sel        {object id: row} | [row or -1 per object of the radar message]"""
    __slots__ = ("t", "ego", "ego_names", "ego_given", "modes", "names", "rows", "lanes", "sel")

    def __init__(self, t, ego, ego_names, ego_given, modes, names, rows, lanes, sel):
        self.t, self.ego, self.ego_names, self.ego_given = t, ego, ego_names, ego_given
        self.modes, self.names, self.rows, self.lanes, self.sel = modes, names, rows, lanes, sel


def ego_speed(ego, t):
    """Rows of the car's futures (p, x, y, v) with the speed derived from the places where it is not given."""
    n = len(t)
    ego = np.array(ego)
    for row in ego:
        if not np.all(np.isfinite(row[1 + 2 * n:])):
            row[1 + 2 * n:] = np.hypot(np.gradient(row[1:1 + n], t), np.gradient(row[1 + n:1 + 2 * n], t))
    return ego


def _raw(v):
    return float(v) if _fin(v) else math.nan


def _mode_rows(fut, n):
    """The modes of one object's `fut` as rows (p0, p, x, y, sx, sy, rxy), or None where it is not as the contract
    says (not a list of dicts with x and y of the grid's length, a number that is none)."""
    if not isinstance(fut, (list, tuple)) or not fut:
        return None
    rows = np.empty((len(fut), 2 + 5 * n))
    nan = (math.nan,) * n
    for i, m in enumerate(fut):
        if not isinstance(m, dict) or m.get("x") is None or m.get("y") is None:
            return None
        sx, sy, r = m.get("sx"), m.get("sy"), m.get("rxy")
        try:
            a = np.array((m["x"], m["y"], sx if sx is not None else nan, sy if sy is not None else nan,
                          r if r is not None else nan), float)
        except (TypeError, ValueError):
            return None
        if a.shape != (5, n):
            return None
        for q, given in enumerate((True, True, sx is not None, sy is not None, r is not None)):
            if given and not np.isfinite(a[q]).all():
                return None
        rows[i, 0], rows[i, 1] = _raw(m.get("p0")), _raw(m.get("p"))
        rows[i, 2:] = a.ravel()
    return rows


def from_message(radar_msg, companion=None, cv=False):
    """The contract's fields of a `radar` message (and its companion `radar_fut`) as `Futures`: the message's adapter
    (a recording, a prediction layer in another process). None where the messages carry none of it. cv: the
    futures are not taken (cfg `futures: "cv"`), only what the radar message's objects say of their lanes."""
    comp = companion if isinstance(companion, dict) and not cv else {}
    grid = None if cv else _grid(radar_msg.get("fut") or comp.get("fut"))
    efut = None if cv else (radar_msg.get("ego_fut") or comp.get("ego_fut"))
    n = len(grid) if grid is not None else 0
    ego, ego_names = None, []
    if grid is not None and isinstance(efut, dict) and isinstance(efut.get("modes"), (list, tuple)) and efut["modes"]:
        ego = np.full((len(efut["modes"]), 1 + 3 * n), math.nan)
        for i, m in enumerate(efut["modes"]):
            x, y, v = (_arr(m, k, n) if isinstance(m, dict) else None for k in ("x", "y", "v"))
            if x is None or y is None:
                ego = None
                break
            ego[i, 0], ego[i, 1:1 + n], ego[i, 1 + n:1 + 2 * n] = _raw(m.get("p")), x, y
            if v is not None:
                ego[i, 1 + 2 * n:] = v
            ego_names.append(str(m.get("m") or "mode%d" % i))
        if ego is not None and not np.isfinite(ego[:, 1 + 2 * n:]).all():
            ego = ego_speed(ego, grid)
    by_id = {}
    for co in comp.get("objects") or ():
        if isinstance(co, dict) and "id" in co:
            by_id[co["id"]] = co
    sel, rows, lanes, modes, names = [], [], [], [], []
    k = 0
    some = grid is not None or isinstance(efut, dict)
    for o in radar_msg.get("objects") or ():
        if not isinstance(o, dict) or o.get("state") not in ("confirmed", "coast"):
            sel.append(-1)
            continue
        c = by_id.get(o.get("id")) or {}
        fut = None if cv else o.get("fut", c.get("fut"))
        lane = o.get("lane", c.get("lane"))
        mr = _mode_rows(fut, n) if grid is not None and fut is not None else None
        sel.append(len(rows))
        rows.append((k, 0 if mr is None else len(mr), 0 if fut is None else 1 if mr is not None else 2))
        lanes.append(lane if isinstance(lane, dict) else None)
        some = some or fut is not None or isinstance(lane, dict)
        if mr is not None:
            modes.append(mr)
            names += [str(m.get("m") or "mode%d" % i) for i, m in enumerate(fut)]
            k += len(mr)
    if not some:
        return None
    return Futures(grid, ego, ego_names, isinstance(efut, dict),
                   np.concatenate(modes) if modes else np.zeros((0, 2 + 5 * n)), names,
                   np.array(rows, float).reshape(-1, 3), lanes, sel)


def _prob(v, default):
    return min(1.0, max(0.0, float(v))) if math.isfinite(v) else default


def contract_modes(o, H, row, prepend):
    """The object's modes of the contract on the grid (now prepended when the grid starts later), or None when
    their probabilities do not sum."""
    n = len(H.t)
    x0, y0 = float(o.get("x_m", 0.0)), float(o.get("y_m", 0.0))
    sd = o.get("std") or (1.0, 1.0)
    sx0, sy0 = (float(sd[0]), float(sd[1])) if len(sd) >= 2 and _fin(sd[0]) and _fin(sd[1]) else (1.0, 1.0)
    out = []
    head = np.array([[x0], [y0], [sx0], [sy0], [0.0]]) if prepend else None
    first, cnt = int(H.rows[row, 0]), int(H.rows[row, 1])
    for i in range(cnt):
        r = H.modes[first + i]
        a = r[2:].reshape(5, n).copy()
        if math.isnan(a[2, 0]):
            a[2] = sx0
        if math.isnan(a[3, 0]):
            a[3] = sy0
        if math.isnan(a[4, 0]):
            a[4] = 0.0
        a[2:4] = np.abs(a[2:4])
        np.clip(a[4], -0.99, 0.99, out=a[4])
        if prepend:
            a = np.concatenate([head, a], axis=1)
        p = _prob(r[1], 1.0 / cnt)
        p0 = _prob(r[0], p)
        out.append((H.names[first + i], p0, p, a[0], a[1], a[2], a[3], a[4]))
    s0, sc = sum(q[1] for q in out), sum(q[2] for q in out)
    if not (s0 > 0.0 and sc > 0.0):
        return None
    return [(nm, p0 / s0, p / sc, x, y, sx, sy, r) for nm, p0, p, x, y, sx, sy, r in out]


def _polyline_s(px, py):
    d = np.hypot(np.diff(px), np.diff(py))
    return np.concatenate([[0.0], np.cumsum(d)])


def _along(px, py, ps, s):
    """Points of a polyline at arc lengths s (extrapolated along its end segments), and the headings there."""
    s = np.asarray(s, float)
    x, y = np.interp(s, ps, px), np.interp(s, ps, py)
    hd = np.arctan2(np.diff(py), np.diff(px))
    h = np.interp(s, 0.5 * (ps[1:] + ps[:-1]), hd) if len(hd) > 1 else np.full(len(s), hd[0] if len(hd) else 0.0)
    lo, hi = s < ps[0], s > ps[-1]
    if lo.any():
        x[lo] = px[0] + (s[lo] - ps[0]) * math.cos(hd[0])
        y[lo] = py[0] + (s[lo] - ps[0]) * math.sin(hd[0])
    if hi.any():
        x[hi] = px[-1] + (s[hi] - ps[-1]) * math.cos(hd[-1])
        y[hi] = py[-1] + (s[hi] - ps[-1]) * math.sin(hd[-1])
    return x, y, h


def _project(px, py, ps, x, y):
    """(arc length, signed offset left) of a point on a polyline: the nearest segment."""
    best = None
    for i in range(len(px) - 1):
        ax, ay, bx, by = px[i], py[i], px[i + 1], py[i + 1]
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey
        if L2 <= 1e-12:
            continue
        f = ((x - ax) * ex + (y - ay) * ey) / L2
        lo = -math.inf if i == 0 else 0.0                 # the end segments reach on beyond the polyline
        hi = math.inf if i == len(px) - 2 else 1.0
        fc = min(hi, max(lo, f))
        qx, qy = ax + fc * ex, ay + fc * ey
        d2 = (x - qx) ** 2 + (y - qy) ** 2
        if best is None or d2 < best[0]:
            L = math.sqrt(L2)
            off = ((x - ax) * (-ey) + (y - ay) * ex) / L
            best = (d2, ps[i] + fc * L, off)
    return (best[1], best[2]) if best else (0.0, 0.0)


def cv_futures(o, t, cfg, path=None):
    """The degraded futures of a radar object (cfg cv_modes): [(name, p0, p, x, y, sx, sy, rxy)]."""
    if cfg["cv_modes"] != "radar":
        return [cv_future(o, t, cfg, path)]
    ps = o.get("p_stationary")
    flags = o.get("flags") or ()
    if o.get("motion") == "stationary" or "parked" in flags:
        ps = 1.0
    elif o.get("motion") == "moving":
        ps = 0.0
    elif not _fin(ps):
        return [cv_future(o, t, cfg, path)]
    ps = min(1.0, max(0.0, float(ps)))
    if ps >= 1.0 - cfg["cv_stand_min"]:
        return [cv_future(dict(o, motion="stationary"), t, cfg, path)]
    if ps <= cfg["cv_stand_min"]:
        return [cv_future(dict(o, motion="moving"), t, cfg, path)]
    st = cv_future(dict(o, motion="stationary"), t, cfg, path)
    mv = cv_future(dict(o, motion="moving"), t, cfg, path)
    return [(st[0], ps, ps) + st[3:], (mv[0], 1.0 - ps, 1.0 - ps) + mv[3:]]


def cv_future(o, t, cfg, path=None):
    """One constant-velocity mode from a radar object's fields: (name, p0, p, x, y, sx, sy, rxy). Over the ground in
    the car frame of now (the frame held fixed); an object the radar calls stationary stands. With the path known,
    an object that drives along it (or against it) keeps its offset beside it and its speed along it, so a car
    ahead in a bend is not predicted off the road."""
    x0, y0 = float(o.get("x_m", 0.0)), float(o.get("y_m", 0.0))
    sd = o.get("std") or (1.0, 1.0, 1.0, 1.0)
    sd = [float(v) if _fin(v) else 1.0 for v in (list(sd) + [1.0] * 4)[:4]]
    cap = cfg["cv_v_std_cap_mps"]
    if o.get("motion") == "stationary":
        vx = vy = 0.0
        svx = svy = cfg["cv_still_v_std_mps"]
        name = "stand"
    else:
        vx = float(o.get("vx_mps", 0.0)) if _fin(o.get("vx_mps")) else 0.0
        vy = float(o.get("vy_mps", 0.0)) if _fin(o.get("vy_mps")) else 0.0
        sp = math.hypot(vx, vy)
        if sp > cfg["cv_speed_cap_mps"]:
            vx, vy = vx * cfg["cv_speed_cap_mps"] / sp, vy * cfg["cv_speed_cap_mps"] / sp
        svx, svy = min(sd[2], cap), min(sd[3], cap)
        name = "cv"
    sx = np.sqrt(sd[0] ** 2 + (svx * t) ** 2)
    sy = np.sqrt(sd[1] ** 2 + (svy * t) ** 2)
    x, y = x0 + vx * t, y0 + vy * t
    if path is not None and cfg["cv_along_path"] and math.hypot(vx, vy) >= cfg["along_min_mps"]:
        px, py, ps = path
        s0, off = _project(px, py, ps, x0, y0)
        _, _, h = _along(px, py, ps, [s0])
        dv = math.atan2(vy, vx) - h[0]
        dv = (dv + math.pi) % (2.0 * math.pi) - math.pi
        lim = math.radians(cfg["along_heading_deg"])
        if abs(dv) <= lim or abs(dv) >= math.pi - lim:
            va = math.hypot(vx, vy) * (1.0 if abs(dv) <= lim else -1.0)
            cx, cy, ch = _along(px, py, ps, s0 + va * t)
            x, y = cx - off * np.sin(ch), cy + off * np.cos(ch)
            name += "_path"
    return (name, 1.0, 1.0, x, y, sx, sy, np.zeros(len(t)))


def plan_path(road_msg, ts_ns, max_age_s=1.0):
    """The road model's plan as a polyline (x, y left, arc length) in the car frame, or None."""
    if not isinstance(road_msg, dict):
        return None
    ts = road_msg.get("ts_mono_ns")
    if not isinstance(ts, int) or abs(ts_ns - ts) > max_age_s * 1e9:
        return None
    plan = road_msg.get("plan")
    if not isinstance(plan, dict):
        return None
    try:
        px, py = np.asarray(plan.get("x"), float), -np.asarray(plan.get("y"), float)
    except (TypeError, ValueError):
        return None
    if px.ndim != 1 or len(px) < 2 or len(px) != len(py) or not (np.all(np.isfinite(px)) and np.all(np.isfinite(py))):
        return None
    if not np.all(np.diff(px) >= 0.0) or px[-1] < 5.0:
        return None
    if px[0] > 0.0:                                   # from the car
        px, py = np.concatenate([[0.0], px]), np.concatenate([[py[0]], py])
    return px, py, _polyline_s(px, py)


def ego_futures(ego, t, cfg, path=None):
    """The car's own modes from its motion now (speed, yaw rate, acceleration) and the plan: [(name, p, x, y, v)].
    With a plan: along it (p = ego_path_p) and on its present turn (the rest); without one the turn alone."""
    v0 = max(0.0, float(ego.get("speed_mps") or 0.0))
    w0 = float(ego.get("yaw_rate_rps") or 0.0)
    lo, hi = cfg["ego_accel_clip"]
    a0 = min(hi, max(lo, float(ego.get("accel_mps2") or 0.0)))
    # speed: v0 + a0 t, never below 0
    v = np.maximum(0.0, v0 + a0 * t)
    t_stop = v0 / -a0 if a0 < 0.0 else math.inf
    s = np.where(t <= t_stop, v0 * t + 0.5 * a0 * t * t, v0 * t_stop + 0.5 * a0 * t_stop * t_stop) if a0 < 0.0 \
        else v0 * t + 0.5 * a0 * t * t
    # on its turn: the heading integrates the yaw rate over the distance (the yaw rate scaled by the speed)
    k = w0 / v0 if v0 > 0.5 else 0.0
    psi = k * s
    if abs(k) > 1e-6:
        xt, yt = np.sin(psi) / k, (1.0 - np.cos(psi)) / k
    else:
        xt, yt = s.copy(), np.zeros(len(t))
    modes = []
    if path is not None:
        px, py, ps = path
        s0, _ = _project(px, py, ps, 0.0, 0.0)
        cx, cy, _ = _along(px, py, ps, s0 + s)
        # the plan's line from the car's own place: its offset at the car taken out over the first 2 s
        _, off0 = _project(px, py, ps, 0.0, 0.0)
        fade = np.clip(1.0 - t / 2.0, 0.0, 1.0)
        _, _, ch = _along(px, py, ps, s0 + s)
        cx = cx - (-off0) * fade * np.sin(ch)
        cy = cy + (-off0) * fade * np.cos(ch)
        modes.append(("path", cfg["ego_path_p"], cx, cy, v))
        modes.append(("turn", 1.0 - cfg["ego_path_p"], xt, yt, v))
    else:
        modes.append(("turn", 1.0, xt, yt, v))
    return modes


def build(radar_msg, cfg, path=None, H=None):
    """FrameFutures for one `radar` message and the contract's futures of its frame (`Futures`, or None: the degraded
    adapters). Objects listed: confirmed and coast (a remembered object is never judged: nothing sees it)."""
    grid = H.t if H is not None else None
    if grid is None:
        t = np.arange(0.0, cfg["horizon_s"] + 1e-9, cfg["grid_dt_s"])
        prepend = False
    else:
        prepend = grid[0] > 1e-9
        t = np.concatenate([[0.0], grid]) if prepend else grid
    F = FrameFutures(t)
    ego = radar_msg.get("ego") or {}
    if not any(isinstance(o, dict) and o.get("state") in ("confirmed", "coast") for o in radar_msg.get("objects") or ()):
        F.ego = [("none", 1.0, None, None, None)]          # nothing to judge: the car's futures are not made
        F.degraded["ego"] = not (H is not None and H.ego_given)
        return F
    em = None
    if grid is not None and H.ego is not None:
        em = []
        n = len(grid)
        v0 = float(ego.get("speed_mps") or 0.0)
        for i, row in enumerate(H.ego):
            x, y, v = row[1:1 + n], row[1 + n:1 + 2 * n], np.maximum(row[1 + 2 * n:], 0.0)
            if prepend:
                x, y, v = np.concatenate([[0.0], x]), np.concatenate([[0.0], y]), np.concatenate([[v0], v])
            em.append((H.ego_names[i], _prob(row[0], 1.0 / len(H.ego)), x, y, v))
        if em:
            sp = sum(q[1] for q in em)
            em = [(nm, p / sp, x, y, v) for nm, p, x, y, v in em] if sp > 0 else None
    if em is None:
        em = ego_futures(ego, t, cfg, path)
        F.degraded["ego"] = True
    F.ego = em
    by_id = H is not None and isinstance(H.sel, dict)
    for j, o in enumerate(radar_msg.get("objects") or ()):
        if not isinstance(o, dict) or o.get("state") not in ("confirmed", "coast"):
            continue
        row = -1 if H is None else H.sel.get(o.get("id"), -1) if by_id else H.sel[j]
        state = int(H.rows[row, 2]) if row >= 0 else 0
        modes = contract_modes(o, H, row, prepend) if grid is not None and state == 1 else None
        if modes is None:
            if state != 0:
                F.degraded["invalid"] += 1
            modes = cv_futures(o, t, cfg, path)
            F.degraded["objects"] += 1
            src = "cv"
            if any(m[0].endswith("_path") for m in modes):
                F.along.add(o.get("id"))
        else:
            src = "contract"
        if path is not None:
            # where no heading is known, a road user stands or drives parallel to the road at its place
            px, py, ps = path
            sp, _ = _project(px, py, ps, float(o.get("x_m", 0.0)), float(o.get("y_m", 0.0)))
            F.road_hdg[o.get("id")] = float(_along(px, py, ps, [sp])[2][0])
        if row >= 0 and H.lanes is not None:
            lane = H.lanes[row] if isinstance(H.lanes, list) else dict(zip(LANES, H.lanes[row].tolist()))
            if lane is not None:
                F.lane[o.get("id")] = lane
        F.objects.append((o, modes, src))
    return F
