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

An object's x, y are the radar's reference point of it (the nearest point of its footprint as the camera sees it,
as `x_m`, `y_m`); its extent comes from `len`, `wid`, `hdg` (with their uncertainties) where the radar gives them,
else from its class's prior size (CLASS_SIZE).

Degraded mode: an object without `fut` gets constant-velocity futures built from its `radar` fields (`cv_futures`:
by default the radar's own two hypotheses, it stands with p_stationary and it drives on at its velocity with the
rest; `cv_modes: "single"` one mode), and without `ego_fut` the car gets its own modes from its present motion and the road model's
plan (`ego_futures`). Both are reported (`FrameFutures.degraded`) and in the decision layer's health."""
import math

import numpy as np

from ..core import classvote

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


def _prob(v, default):
    return min(1.0, max(0.0, float(v))) if _fin(v) else default


def contract_modes(o, fut, t_in, prepend):
    """The object's modes of the contract on the grid (now prepended when t_in[0] > 0), or None when invalid."""
    if not isinstance(fut, (list, tuple)) or not fut:
        return None
    n = len(t_in)
    x0, y0 = float(o.get("x_m", 0.0)), float(o.get("y_m", 0.0))
    sd = o.get("std") or (1.0, 1.0)
    sx0, sy0 = (float(sd[0]), float(sd[1])) if len(sd) >= 2 and _fin(sd[0]) and _fin(sd[1]) else (1.0, 1.0)
    out = []
    for i, m in enumerate(fut):
        if not isinstance(m, dict):
            return None
        x, y = _arr(m, "x", n), _arr(m, "y", n)
        if x is None or y is None:
            return None
        sx, sy, r = _arr(m, "sx", n), _arr(m, "sy", n), _arr(m, "rxy", n)
        sx = np.full(n, sx0) if sx is None else np.abs(sx)
        sy = np.full(n, sy0) if sy is None else np.abs(sy)
        r = np.zeros(n) if r is None else np.clip(r, -0.99, 0.99)
        p = _prob(m.get("p"), 1.0 / len(fut))
        p0 = _prob(m.get("p0"), p)
        if prepend:
            x, y = np.concatenate([[x0], x]), np.concatenate([[y0], y])
            sx, sy, r = np.concatenate([[sx0], sx]), np.concatenate([[sy0], sy]), np.concatenate([[0.0], r])
        out.append((str(m.get("m") or "mode%d" % i), p0, p, x, y, sx, sy, r))
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


def build(radar_msg, cfg, path=None, companion=None):
    """FrameFutures for one `radar` message (and its companion `radar_fut`, if any). Objects listed: confirmed and
    coast (a remembered object is never judged: nothing sees it)."""
    comp = companion if isinstance(companion, dict) else {}
    top = radar_msg.get("fut") or comp.get("fut")
    grid = _grid(top)
    if grid is None:
        t = np.arange(0.0, cfg["horizon_s"] + 1e-9, cfg["grid_dt_s"])
        prepend = False
    else:
        prepend = grid[0] > 1e-9
        t = np.concatenate([[0.0], grid]) if prepend else grid
    F = FrameFutures(t)
    t_in = grid if grid is not None else t
    ego = radar_msg.get("ego") or {}
    efut = radar_msg.get("ego_fut") or comp.get("ego_fut")
    em = None
    if grid is not None and isinstance(efut, dict) and isinstance(efut.get("modes"), (list, tuple)) and efut["modes"]:
        em = []
        n = len(t_in)
        v0 = float(ego.get("speed_mps") or 0.0)
        for i, m in enumerate(efut["modes"]):
            x, y, v = (_arr(m, k, n) if isinstance(m, dict) else None for k in ("x", "y", "v"))
            if x is None or y is None:
                em = None
                break
            v = np.hypot(np.gradient(x, t_in), np.gradient(y, t_in)) if v is None else np.maximum(v, 0.0)
            if prepend:
                x, y, v = np.concatenate([[0.0], x]), np.concatenate([[0.0], y]), np.concatenate([[v0], v])
            em.append((str(m.get("m") or "mode%d" % i), _prob(m.get("p"), 1.0 / len(efut["modes"])), x, y, v))
        if em:
            sp = sum(q[1] for q in em)
            em = [(nm, p / sp, x, y, v) for nm, p, x, y, v in em] if sp > 0 else None
    if em is None:
        em = ego_futures(ego, t, cfg, path)
        F.degraded["ego"] = True
    F.ego = em
    by_id = {}
    for co in comp.get("objects") or ():
        if isinstance(co, dict) and "id" in co:
            by_id[co["id"]] = co
    for o in radar_msg.get("objects") or ():
        if not isinstance(o, dict) or o.get("state") not in ("confirmed", "coast"):
            continue
        c = by_id.get(o.get("id")) or {}
        fut = o.get("fut", c.get("fut"))
        modes = contract_modes(o, fut, t_in, prepend) if grid is not None and fut is not None else None
        if modes is None:
            if fut is not None:
                F.degraded["invalid"] += 1
            modes = cv_futures(o, t, cfg, path)
            F.degraded["objects"] += 1
            src = "cv"
        else:
            src = "contract"
        if path is not None:
            # where no heading is known, a road user stands or drives parallel to the road at its place
            px, py, ps = path
            sp, _ = _project(px, py, ps, float(o.get("x_m", 0.0)), float(o.get("y_m", 0.0)))
            F.road_hdg[o.get("id")] = float(_along(px, py, ps, [sp])[2][0])
        lane, intent = o.get("lane", c.get("lane")), o.get("intent", c.get("intent"))
        if isinstance(lane, dict):
            F.lane[o.get("id")] = lane
        if intent is not None:
            F.intent[o.get("id")] = intent
        F.objects.append((o, modes, src))
    return F
