"""Shared pieces of the real-drive truth set for the radar (docs/RADAR_TRUTH.md): replaying a recorded session
through the radar, the detector's boxes and camera frames of the same session, and matching radar objects to
labelled things by time and bearing.

Everything here reads the radar through its public output (the `radar` message of chris_integrator/radar/box.py),
so labels and scores survive changes inside the tracker. Times are the session's monotonic clock in seconds
(`ts_mono_ns` / 1e9); bearings are degrees, left positive (atan2(y_m, x_m) of the car frame).
"""
import bisect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NS = 1e9
CALIB_OVERRIDE = (-0.0140, -0.0624)      # pitch, yaw (rad) of the 2026-11-22 recordings (VISUAL_RADAR_EVAL.md)
SEEN = ("confirmed", "coast")            # states in which a sensor still sees the object (memory: carried on)
REAL = ("parked", "standing_in_lane", "standing", "moving_with", "oncoming", "crossing", "pedestrian", "cyclist")
STANDING = ("parked", "standing_in_lane", "standing")
MOVING = ("moving_with", "oncoming", "crossing")


def bearing(x, y):
    return math.degrees(math.atan2(y, x))


def session_name(path):
    return os.path.basename(os.path.normpath(path))


def compact(o):
    """The part of a radar object the truth tools use."""
    cls = o.get("class")
    return {"id": o["id"], "state": o["state"], "x": o["x_m"], "y": o["y_m"], "vx": o["vx_mps"], "vy": o["vy_mps"],
            "speed": o["speed_mps"], "v_rel": o.get("v_rel_mps"), "motion": o["motion"], "flags": list(o["flags"]),
            "ttc": o.get("ttc_s"), "p_threat": o.get("p_threat"), "off": o.get("path_offset_m"),
            "label": cls.get("label") if isinstance(cls, dict) else None,
            "b": bearing(o["x_m"], o["y_m"]) if o["x_m"] > 0.3 else None}


def replay(session, calib_override=CALIB_OVERRIDE, cfg=None):
    """[(T, ego_speed, [compact objects])] of every radar frame of the session (chris_integrator.radar.replay)."""
    from chris_integrator.radar.replay import run
    out = []
    for msg in run(session, calib_override=calib_override, cfg=cfg):
        out.append((msg["ts_mono_ns"] / NS, (msg.get("ego") or {}).get("speed_mps"), [compact(o) for o in msg["objects"]]))
    return out


def inputs(session, calib_override=CALIB_OVERRIDE):
    """The detector's boxes as the radar got them (road_users messages), the camera frames on disk with their
    times, and the calibration in use: {"rus": [(T, view, [obj])], "images": [(T, path)], "calib": dict}."""
    from chris_integrator.core.roadusers import PROJ_CFG, from_lights
    from chris_integrator.replay import read_lines
    pcfg = dict(PROJ_CFG, sensor="cam")
    calib, rus, cams = None, [], {}
    for m in read_lines(session):
        t = m.get("type")
        if t == "calib":
            calib = dict(m)
            if calib_override is not None:
                calib["pitch"], calib["yaw"] = calib_override
        elif t == "lights" and calib is not None:
            ru = from_lights(m, calib, "cam", pcfg)
            if ru is not None and isinstance(ru.get("ts_mono_ns"), int) and not ru.get("shed"):
                objs = [{"b": o.get("bearing_deg"), "r": o.get("range_m"), "box": o.get("box"), "cls": o.get("class"),
                         "score": o.get("class_conf"), "clip": o.get("clip") or ""} for o in ru["objects"]
                        if isinstance(o.get("bearing_deg"), (int, float))]
                rus.append((ru["ts_mono_ns"] / NS, ru.get("view"), objs))
        elif t == "camera" and isinstance(m.get("seq"), int) and isinstance(m.get("ts_mono_ns"), int):
            cams[m["seq"]] = m["ts_mono_ns"] / NS
    fdir = os.path.join(session, "frames")
    images = []
    if os.path.isdir(fdir):
        for fn in os.listdir(fdir):
            stem = fn.split(".")[0]
            if fn.endswith(".jpg") and stem.isdigit() and int(stem) in cams:
                images.append((cams[int(stem)], os.path.join(fdir, fn)))
    images.sort()
    rus.sort(key=lambda r: r[0])
    return {"rus": rus, "images": images, "calib": calib}


def tracks(frames):
    """Radar object id -> [(T, compact object)] over the session."""
    tr = {}
    for T, _, objs in frames:
        for o in objs:
            tr.setdefault(o["id"], []).append((T, o))
    return tr


def episodes(track, flag, gap_s=0.5):
    """[(t0, t1, [(T, o)])] of the runs in which the object carries `flag` (gaps up to gap_s merged)."""
    out, cur = [], None
    for T, o in track:
        if flag in o["flags"]:
            if cur is not None and T - cur[1] <= gap_s:
                cur[1] = T
                cur[2].append((T, o))
            else:
                cur = [T, T, [(T, o)]]
                out.append(cur)
    return [tuple(c) for c in out]


def frame_dt(frames):
    """The time each radar frame stands for (the step to the next one, capped)."""
    ts = [f[0] for f in frames]
    return [min(max(ts[i + 1] - ts[i], 0.0), 0.25) if i + 1 < len(ts) else 0.08 for i in range(len(ts))]


# ---- anchors: a labelled thing as time + bearing (and a rough range), independent of the radar's ids ----------
def box_near(rus, T, b, tol_deg=4.0, win_s=0.2):
    """The detector box nearest in bearing to b within win_s of T: (T_box, obj) or None."""
    ts = [r[0] for r in rus] if not hasattr(rus, "_ts") else rus._ts
    i, j = bisect.bisect_left(ts, T - win_s), bisect.bisect_right(ts, T + win_s)
    best = None
    for k in range(i, j):
        for o in rus[k][2]:
            d = abs(o["b"] - b)
            if d <= tol_deg and (best is None or d < best[0]):
                best = (d, rus[k][0], o)
    return None if best is None else (best[1], best[2])


class Rus(list):
    """road_users messages with their times indexed."""
    def __init__(self, rus):
        super().__init__(rus)
        self._ts = [r[0] for r in rus]


def anchor_of(points, rus):
    """Anchor points [{"t", "b", "r", "box", "src"}] for a list of (T, compact object): the detector box that the
    radar's bearing points at where there is one (src "box"), else the radar's own place (src "radar")."""
    out, used = [], set()
    for T, o in points:
        if o["b"] is None:
            continue
        hit = box_near(rus, T, o["b"])
        if hit is not None:
            tb, bo = hit
            if (tb, tuple(bo["box"] or ())) in used:
                continue
            used.add((tb, tuple(bo["box"] or ())))
            out.append({"t": round(tb, 3), "b": round(bo["b"], 2), "r": None if bo["r"] is None else round(bo["r"], 1),
                        "box": [round(v) for v in bo["box"]] if bo["box"] else None, "src": "box"})
    if len(out) < 2:                    # never boxed near the radar's bearing: the radar's own places, thinned
        step = max(1, len(points) // 12)
        out = [{"t": round(T, 3), "b": round(o["b"], 2), "r": round(math.hypot(o["x"], o["y"]), 1), "box": None, "src": "radar"}
               for T, o in points[::step] if o["b"] is not None]
    out.sort(key=lambda a: a["t"])
    return out


def anchor_bearing(anchor, T, reach_s=0.6):
    """The anchor's bearing at T (linear between its points; held reach_s beyond its ends), or None."""
    if not anchor:
        return None
    ts = [a["t"] for a in anchor]
    if T < ts[0] - reach_s or T > ts[-1] + reach_s:
        return None
    k = bisect.bisect_left(ts, T)
    if k == 0:
        return anchor[0]["b"]
    if k >= len(ts):
        return anchor[-1]["b"]
    a, b = anchor[k - 1], anchor[k]
    if b["t"] - a["t"] > 2.5:           # a hole in the anchor: only near either end
        if T - a["t"] <= reach_s:
            return a["b"]
        if b["t"] - T <= reach_s:
            return b["b"]
        return None
    w = (T - a["t"]) / max(b["t"] - a["t"], 1e-6)
    return a["b"] + w * (b["b"] - a["b"])


def bearing_tol(b_deg, rng=None):
    """How far a radar object's bearing may sit from the anchor's: 3 deg, more for something near and wide."""
    tol = 3.0
    if rng is not None and rng < 12.0:
        tol += (12.0 - rng) * 0.5
    return tol


def match(track, label, min_share=0.5, min_frames=2):
    """Does this radar track belong to the labelled thing? The share of its frames in the label's span in which a
    sensor saw it (SEEN) at the anchor's bearing. Returns (matched, share, n)."""
    t0, t1 = label["t0"] - 0.3, label["t1"] + 0.3
    n = ok = 0
    for T, o in track:
        if not (t0 <= T <= t1) or o["state"] not in SEEN or o["b"] is None:
            continue
        ab = anchor_bearing(label["anchor"], T)
        if ab is None:
            continue
        n += 1
        if abs(o["b"] - ab) <= bearing_tol(ab, math.hypot(o["x"], o["y"])):
            ok += 1
    share = ok / n if n else 0.0
    return (n >= min_frames and share >= min_share), share, n
