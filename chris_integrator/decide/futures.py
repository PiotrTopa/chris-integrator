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

Degraded mode: an object without `fut` gets constant-velocity futures built from its `radar` fields (by default the
radar's own two hypotheses, it stands with p_stationary and it drives on at its velocity with the rest; `cv_modes:
"single"` one mode; an object that drives along the path or against it follows its bends), and without `ego_fut`
the car gets its own modes from its present motion and the road model's plan. Both are reported (the `decide`
message's `futures`) and in the decision layer's health.

Here: the contract's futures as arrays (`Futures`: what the predictor hands over in process, and what
`from_message` reads out of a message), the constants, the plan as a path. What is made of them (the grid, the
degraded futures, each object's footprint, the threat, the records) is the native core's `decide_frame`
(native/src/decide.c), called through `threat.ThreatModel.frame`."""
import math

import numpy as np

from .. import native as _native

# prior footprint by class: length, width, 1 sigma of each (m); decide.c DF_SIZE holds the same
CLASS_SIZE = {
    "car": (4.5, 1.8, 0.4, 0.1), "van": (5.2, 2.0, 0.5, 0.15), "truck": (9.0, 2.5, 3.0, 0.15),
    "bus": (12.0, 2.55, 1.5, 0.1), "tram": (25.0, 2.65, 5.0, 0.1), "human": (0.6, 0.6, 0.15, 0.15),
    "bike": (1.8, 0.7, 0.2, 0.15), "motorbike": (2.1, 0.85, 0.2, 0.15), None: (4.2, 1.8, 1.0, 0.3),
}
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


def cfg_array(cfg):
    """The futures' constants as the native core reads them (decide.c FC_*)."""
    lo, hi = cfg["ego_accel_clip"]
    return np.array([cfg["cv_speed_cap_mps"], cfg["cv_v_std_cap_mps"], cfg["cv_still_v_std_mps"],
                     1.0 if cfg["cv_along_path"] else 0.0, 1.0 if cfg["cv_modes"] == "radar" else 0.0, cfg["cv_stand_min"],
                     cfg["along_heading_deg"], cfg["along_min_mps"], cfg["ego_path_p"], lo, hi], float)


def own_grid(cfg):
    """The degraded futures' grid (a contract grid replaces it)."""
    return np.arange(0.0, cfg["horizon_s"] + 1e-9, cfg["grid_dt_s"])


_GRID = [None, None, None, 1.0]    # the last grid read: the list it came as, a copy of it, the array made of it,
                                   # the sign of its first time (the one member that may be a zero: 0.0 == -0.0)


def _grid(top):
    t = (top or {}).get("t") if isinstance(top, dict) else None
    if t is None:
        return None
    # a producer in this process sends the same list with every message (the predictor's sample times): the array
    # made of it is kept while the list is that one and still holds what it held
    kept = _GRID
    if t is kept[0] and t == kept[1] and math.copysign(1.0, t[0]) == kept[3]:
        return kept[2]
    src = t
    try:
        t = np.asarray(t, float).ravel()
    except (TypeError, ValueError):
        return None
    if len(t) < 1 or not np.all(np.isfinite(t)) or (len(t) > 1 and not np.all(np.diff(t) > 0)) or t[0] < 0.0:
        return None
    if src.__class__ is list and all(v.__class__ is float for v in src):
        t.flags.writeable = False              # shared from now on: nobody may write into it
        kept[0], kept[1], kept[2], kept[3] = src, list(src), t, math.copysign(1.0, src[0])
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
        sel        {object id: row} | [row or -1 per object of the radar message]

    The arrays are float64 arrays or their bytes (the predictor hands bytes over: the native core reads either);
    `arrays()` gives them as arrays."""
    __slots__ = ("t", "ego", "ego_names", "ego_given", "modes", "names", "rows", "lanes", "sel", "fields")

    def __init__(self, t, ego, ego_names, ego_given, modes, names, rows, lanes, sel):
        self.t, self.ego, self.ego_names, self.ego_given = t, ego, ego_names, ego_given
        self.modes, self.names, self.rows, self.lanes, self.sel = modes, names, rows, lanes, sel
        self.fields = (t, ego, ego_names, ego_given, modes, names, rows, lanes, sel)      # as decide_frame takes them

    def arrays(self):
        """(ego (E, 1 + 3 N) or None, modes (K, 2 + 5 N), rows (R, 3), lanes (R, 7) | list | None) as numpy arrays."""
        n = len(self.t) if self.t is not None else 0

        def arr(a, w):
            return a if a is None or isinstance(a, list) else np.frombuffer(a).reshape(-1, w) if isinstance(a, bytes) else a
        return arr(self.ego, 1 + 3 * n), arr(self.modes, 2 + 5 * n), arr(self.rows, 3), arr(self.lanes, 7)


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


def plan_path(road_msg, ts_ns, max_age_s=1.0):
    """The road model's plan as a polyline (x, y left, arc length: each the bytes of float64, native `decide_path`)
    in the car frame, or None."""
    if not isinstance(road_msg, dict):
        return None
    ts = road_msg.get("ts_mono_ns")
    if not isinstance(ts, int) or abs(ts_ns - ts) > max_age_s * 1e9:
        return None
    plan = road_msg.get("plan")
    if not isinstance(plan, dict):
        return None
    x, y = plan.get("x"), plan.get("y")
    return _native.decide.decide_path(x.tolist() if isinstance(x, np.ndarray) else x,
                                      y.tolist() if isinstance(y, np.ndarray) else y)
