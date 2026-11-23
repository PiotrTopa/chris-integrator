"""The radar component's face: messages in, the `radar` message out.

In:
    on_ego_state(msg)     the integrator's `ego_state` (speed, yaw rate, acceleration with uncertainty)
    on_ego(msg)           or a plain vehicle feed (`ego`: speed_mps, yaw_rate_rps, accel_long_mps2)
    on_road_users(msg)    a camera's class detector (`road_users`, core/roadusers.py): bearing, ground range
    on_objects(msg)       a range sensor's detections (`objects` of kind radar / lidar, vehicle input API)
    on_road(msg)          the road model's frame: its plan is the path the objects are judged against

Out, `frame(ts_mono_ns)` -> {"type": "radar", ...}: every object in the car frame of that time (x forward,
y LEFT, origin the reference camera):

    id, state (tentative | confirmed | coast | memory: out of every sensor's field, carried on by its motion
        model and the car's movement), exist, age_s, hits, sources, seen {sensor: s since},
    x_m, y_m, vx_mps, vy_mps, ax_mps2, ay_mps2     position, velocity and acceleration OVER THE GROUND, car axes
    speed_mps, heading_rad                          over the ground; heading (car frame, left positive) null below 1 m/s
    v_rel_mps                                       along the car's axis, relative to the car
    std [x, y, vx, vy]                              1 sigma
    motion ("stationary" | "moving" | "unknown"), p_stationary, models {stand, cv, ca}
    class {"label", "confidence", "source", "votes"} | null
    flags: on_path (inside the path's corridor), in_path (stands on the path: an obstacle), parked (a vehicle
           that stands beside the path), with_traffic (drives with the car: see predict_path), unpredictable
           (keeps being elsewhere than predicted)
    path_offset_m, ttc_s (time to reach it at the present closing speed, on the path only)
    p_threat                                        probability that the car reaches it within `threat_ttc_s` at the
                                                    present speeds, over everything the object may be doing (it
                                                    stands, it crawls, it drives with the traffic), for an object
                                                    surely inside the path's corridor; flag `threat` when that is
                                                    nearly certain and has held. It does not wait for the verdict
                                                    "stationary": a thing in the way that the car closes on fast
                                                    is a threat whether it stands or creeps
    consistency, pred_err_m                         how its earlier predictions held (tracker.py)
    angle_offset_deg                                this object's own offset of the camera's ground angle (sensors.py)
    predict [{"t_s", "x_m", "y_m", "std_m"}]        where it will be, in the car frame of NOW (the car's own
                                                    further movement not applied): by its own motion, straight on
    predict_path [{"t_s", "x_m", "y_m"}] | null     for an object that drives with the car (same heading, similar
                                                    speed; flag `with_traffic`): where it will be if it keeps
                                                    following the road as the car does: along the car's own
                                                    path, at its present offset beside it, at its present speed

Scans are processed in time order once the car's own motion is known up to their time; a scan that arrives
late is applied to the objects taken back to its time (tracker.py).
"""
import math

import numpy as np

from ..core import classvote
from ..core.roadusers import Camera
from .. import native as _native
from ..native import round                  # builtins.round, native where built
from . import models as M
from .ego import EgoMotion
from .models import AX, AY, DP, PX, PY, VX, VY
from .sensors import CAMERA_CFG, CameraBoxes, MicrowaveRadar
from .tracker import Tracker

NS = 1e9
_NR = _native.radar             # the native core's radar part, or None (the output's numbers: radar_frame)

CFG = dict(
    corridor_m=1.3, corridor_rel=0.012,      # on the path: |offset| <= corridor_m + corridor_rel x distance
    parked_offset_m=1.9,                     # a standing vehicle this far beside the path is parked
    path_max_m=150.0,
    ttc_min_closing=0.5,
    in_path_sigma=1.0, in_path_slack_m=0.4, in_path_hold_s=0.4,
    threat_ttc_s=2.7, threat_p=0.7, threat_hold_s=0.2,
    threat_range_s=1.0,
    threat_min_mps=5.5,                      # 20 km/h: below, a time to reach says little (walking pace, manoeuvring)
    predict_s=(0.5, 1.0, 2.0),
    # "drives with the car": heading within this of the path's direction where it is, speed within this ratio
    # of the car's own, and both really driving
    with_heading_deg=20.0, with_speed=(0.5, 1.5), with_min_mps=3.0,
    queue_max=200,
    tentative=False,                         # list tentative objects too
)


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class Radar:
    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        self.cfg = dict(CFG, **{k: v for k, v in cfg.items() if k in CFG})
        self.camera_cfg = dict(CAMERA_CFG, **(cfg.get("camera") or {}))
        self.ego = EgoMotion()
        self.tracker = Tracker(self.ego, cfg.get("tracker"), cfg.get("models"))
        self.radars = {}                 # microwave radar sensor id -> MicrowaveRadar
        self.focal_px = float(cfg.get("focal_px", 975.0))    # of the frame the boxes are given in (on_calib sets it)
        self.cam_height = float(self.camera_cfg["cam_height_m"])
        self.h_min = {}                  # view (its nearest ground point) -> the smallest box height seen (px)
        self.cam = None
        self.frame_scale = 1.0
        self.reset()

    def reset(self, reason=None):
        self.ego.reset()
        self.tracker.reset()
        self.queue = []                  # (t, src, sensor, dets)
        self._plan = None                # the road model's plan (x, y right) in the car frame of path_t
        self._path = None                # ... as arrays, y left (`path`)
        self.path_t = None
        self.seq = 0
        self.dropped = 0
        self._in_since = {}              # object id -> since when it has surely stood on the path
        self._threat_since = {}

    # ---- inputs -------------------------------------------------------------------------------------------
    def on_ego_state(self, msg):
        ts = msg.get("ts_mono_ns")
        sp = msg.get("speed") or {}
        if not isinstance(ts, int) or not sp.get("valid") or not _fin(sp.get("value")):
            return
        yr, ac, st = msg.get("yaw_rate") or {}, msg.get("accel") or {}, msg.get("standstill") or {}
        stands = bool(st.get("value"))
        self.ego.add(ts / NS, sp["value"], yr.get("value") if yr.get("valid") and _fin(yr.get("value")) else 0.0,
                     ac.get("value") if ac.get("valid") and _fin(ac.get("value")) else 0.0,
                     sp.get("std") if _fin(sp.get("std")) else 0.1, yr.get("std") if _fin(yr.get("std")) else 0.003, stands)

    def on_ego(self, msg):
        ts = msg.get("ts_mono_ns")
        if not isinstance(ts, int) or not _fin(msg.get("speed_mps")):
            return
        stands = bool(msg.get("standstill"))
        v = abs(msg["speed_mps"]) if not msg.get("reverse") else -abs(msg["speed_mps"])
        self.ego.add(ts / NS, v, msg.get("yaw_rate_rps") if _fin(msg.get("yaw_rate_rps")) else 0.0,
                     msg.get("accel_long_mps2") if _fin(msg.get("accel_long_mps2")) else 0.0, 0.15, 0.004, stands)

    def on_road_users(self, msg):
        ts, objs = msg.get("ts_mono_ns"), msg.get("objects")
        if not isinstance(ts, int) or not isinstance(objs, list) or msg.get("shed"):
            return                       # shed by the load governor: not "nothing there"
        c = self.camera_cfg
        cov, cover = msg.get("coverage"), None
        if isinstance(cov, dict) and all(_fin(cov.get(k)) for k in ("bearing_min_deg", "bearing_max_deg", "ground_min_m")):
            cover = {"bearing_min": math.radians(cov["bearing_min_deg"]) - 0.02,
                     "bearing_max": math.radians(cov["bearing_max_deg"]) + 0.02, "ground_min_m": float(cov["ground_min_m"])}
        dets = []
        f, hc = self.focal_px, self.cam_height
        jit = math.radians(c["pitch_jitter_deg"])
        for o in objs:
            if not isinstance(o, dict) or not isinstance(o.get("class"), str) or not _fin(o.get("bearing_deg")):
                continue
            b = math.radians(o["bearing_deg"])
            at = {"class": o["class"], "class_conf": o.get("class_conf"), "weight": o.get("weight", 1.0), "box": o.get("box")}
            clip = o.get("clip") or ""
            box = o.get("box")
            box = [float(v) for v in box] if isinstance(box, (list, tuple)) and len(box) == 4 and all(_fin(v) for v in box) else None
            hpx = box[3] - box[1] if box else None
            wa = (box[2] - box[0]) / f if box and box[2] > box[0] else None      # the box's width as an angle
            if box and not clip and hpx and hpx >= 6.0 and wa:
                at["aspect"] = (box[2] - box[0]) / hpx   # of a whole box: which way a vehicle points
            # the box's centre: as good as its edges, unless a side is cut (then the centre is not the object's)
            if box and "l" not in clip and "r" not in clip:
                sb = math.hypot(c["bearing_rel"] * (box[2] - box[0]) / f, math.radians(c["bearing_floor_deg"]))
            else:
                sb = math.radians(o["bearing_std_deg"]) if _fin(o.get("bearing_std_deg")) else math.radians(1.0)
            if _fin(o.get("range_m")) and o["range_m"] > 0.5:
                # the ground point as the camera saw it: the angle below the horizon of the bottom edge
                r = float(o["range_m"])
                ang = math.atan2(hc, r)
                if hpx is not None and hpx > 0.0:
                    sa = math.hypot((c["bottom_px"] + c["bottom_rel"] * hpx) / f, jit)
                else:                                    # no box given: what the stated range error is as an angle
                    sr = float(o["range_std_m"]) if _fin(o.get("range_std_m")) else 0.2 * r
                    sa = max(sr * hc / (r * r + hc * hc), jit)
                if hpx is not None and hpx >= 6.0 and "t" not in clip and "b" not in clip:
                    sh = (c["height_std_px"] + c["height_std_rel"] * hpx) / hpx          # of the logarithm
                    dets.append({"kind": "rbh", "z": np.array([ang, b, math.log(hpx * self.frame_scale)]),
                                 "R": np.array([sa * sa, sb * sb, sh * sh]), "attrs": at, "w": wa})
                else:
                    dets.append({"kind": "rb", "z": np.array([ang, b]), "R": np.array([sa * sa, sb * sb]), "attrs": at, "w": wa})
                self._scale_rate(dets[-1], o)
            elif "b" in clip:
                top = None
                if box and "t" not in clip and self.cam is not None and hpx:
                    ry = self.cam.ray(0.5 * (box[0] + box[2]), box[1])
                    top = math.atan2(ry[2], math.hypot(ry[0], ry[1]))          # below the horizon: positive
                if top is not None:
                    st = math.hypot((c["bottom_px"] + c["bottom_rel"] * hpx) / f, jit)
                    dets.append({"kind": "bt", "z": np.array([b, top]), "R": np.array([sb * sb, st * st]), "attrs": at, "w": wa})
                else:
                    dets.append({"kind": "b", "z": np.array([b]), "R": np.array([sb * sb]), "attrs": at, "w": wa})
                self._scale_rate(dets[-1], o)
        # the detector's size limit in this view, as its boxes show it: the smallest seen, forgotten slowly
        key = round(cover["ground_min_m"]) if cover else None
        hs = [math.exp(d["z"][2]) for d in dets if d["kind"] in ("rbh", "rbhs")]
        hm = self.h_min.get(key)
        if hs:
            hm = min(hs) if hm is None else min(min(hs), hm * 1.002)
            self.h_min[key] = hm
        elif hm is None and self.h_min:
            hm = float("inf")                            # a box feed, but this view has not shown a box yet
        self._push((ts / NS, msg.get("sensor") or "cam", CameraBoxes(cover, c, f, hc, hm), dets))

    def _scale_rate(self, det, o):
        """The detector's own measurement of how fast the object's image grows (`scale_rate`, 1/s, with its
        1 sigma and baseline) becomes one more row of the detection (kind + "s")."""
        r, sd = o.get("scale_rate"), o.get("scale_rate_std")
        if not (_fin(r) and _fin(sd) and sd > 0.0) or not self.camera_cfg["use_scale_rate"]:
            return
        sd = max(float(sd), self.camera_cfg["scale_rate_floor"])
        det["kind"] += "s"
        det["z"] = np.append(det["z"], float(r))
        det["R"] = np.append(det["R"], sd * sd)
        det["sdt"] = round(float(o.get("scale_dt_s") or 0.0), 3)

    def on_calib(self, msg):
        """The camera's calibration message: the focal length that turns a box height into a size at a distance."""
        if _fin(msg.get("focal_px")) and msg["focal_px"] > 100.0:
            self.focal_px = float(msg["focal_px"])
        if _fin(msg.get("height_m")) and 0.3 < msg["height_m"] < 4.0:
            self.cam_height = float(msg["height_m"])
        try:
            self.cam = Camera(msg)                        # rows of the picture as angles (a box's top edge)
        except (TypeError, ValueError):
            self.cam = None

    def on_objects(self, msg):
        """A range sensor's scan (vehicle input API `objects`, kind radar / lidar): range_m, azimuth_deg,
        range_rate_mps per detection, in the sensor's frame; `mount` {x_m, y_m, yaw_deg} in the car frame."""
        ts, objs = msg.get("ts_mono_ns"), msg.get("objects")
        if not isinstance(ts, int) or not isinstance(objs, list) or msg.get("kind") not in ("radar", "lidar", None):
            return
        sid = msg.get("sensor") or "radar"
        mt = msg.get("mount") or {}
        mount = (float(mt.get("x_m", 0.0)), float(mt.get("y_m", 0.0)), math.radians(float(mt.get("yaw_deg", 0.0))))
        sen = self.radars.get(sid)
        if sen is None or (sen.mx, sen.my, sen.yaw) != mount:
            sen = self.radars[sid] = MicrowaveRadar(mount)
        dets = []
        sr, sa, sv = sen.sigma
        for o in objs:
            if not isinstance(o, dict) or not (_fin(o.get("range_m")) and _fin(o.get("azimuth_deg")) and _fin(o.get("range_rate_mps"))):
                continue
            r, az = float(o["range_m"]), math.radians(o["azimuth_deg"])
            if r < 0.5 or r > sen.max_range or abs(az) > sen.fov:
                continue
            dets.append({"z": np.array([r, az, float(o["range_rate_mps"])]), "R": np.array([sr * sr, sa * sa, sv * sv]),
                         "attrs": {k: o[k] for k in ("class", "prob") if k in o}})
        self._push((ts / NS, sid, sen, dets))

    def on_road(self, msg):
        plan, ts = msg.get("plan"), msg.get("ts_mono_ns")
        if isinstance(plan, dict) and isinstance(plan.get("x"), list) and isinstance(plan.get("y"), list) \
                and len(plan["x"]) >= 2 and len(plan["x"]) == len(plan["y"]) and isinstance(ts, int):
            xs = plan["x"]
            if all(b - a >= 0.0 for a, b in zip(xs, xs[1:])) and xs[-1] > 5.0:
                self._plan, self._path, self.path_t = (xs, plan["y"]), None, ts / NS
                return
        if isinstance(ts, int) and self.path_t is not None and ts / NS - self.path_t > 1.0:
            self._plan = self._path = None

    @property
    def path(self):
        """The path: (xs, ys left) in the car frame of path_t, or None; made from the plan when first asked."""
        if self._path is None and self._plan is not None:
            self._path = (np.asarray(self._plan[0], float), -np.asarray(self._plan[1], float))   # the road model's y is right
        return self._path

    def _push(self, scan):
        self.queue.append(scan)
        if len(self.queue) > self.cfg["queue_max"]:
            self.queue.sort(key=lambda s: s[0])
            del self.queue[0]
            self.dropped += 1

    # ---- one output ---------------------------------------------------------------------------------------
    def _drain(self, T):
        self.queue.sort(key=lambda s: s[0])
        k = 0
        while k < len(self.queue) and self.queue[k][0] <= T + 1e-6:
            t, src, sensor, dets = self.queue[k]
            k += 1
            self.tracker.process(src, t, sensor, dets)
        del self.queue[:k]

    def _offset(self, x, y, v, w):
        """Lateral offset of a car-frame point from the path (positive left) and what the path was."""
        if self.path is not None and x <= self.path[0][-1] + 5.0:
            return y - float(np.interp(x, self.path[0], self.path[1])), "plan"
        k = w / v if v > 2.0 else 0.0
        return y - 0.5 * k * x * x, "arc"

    def _path_point(self, s, v, w):
        """The path at distance s ahead (car frame): (x, y, direction). The road model's plan as far as it
        reaches, beyond it (and without one) the arc the car is driving."""
        if self.path is not None and s <= self.path[0][-1]:
            xs, ys = self.path
            y = float(np.interp(s, xs, ys))
            d = 1.0
            return s, y, math.atan2(float(np.interp(s + d, xs, ys)) - float(np.interp(s - d, xs, ys)), 2.0 * d)
        k = w / v if v > 2.0 else 0.0
        if abs(k) < 1e-6:
            return s, 0.0, 0.0
        return math.sin(k * s) / k, (1.0 - math.cos(k * s)) / k, k * s

    @staticmethod
    def _class(o, T):
        """classvote.result(o.vote, T): it changes with a vote only (kept on the object until the next one) and
        lapses with its age."""
        cls = o.cls
        if cls is None:
            cls = o.cls = (classvote.result(o.vote),)
        cls = cls[0]
        if cls is not None and o.vote.get("t") is not None and max(0.0, T - o.vote["t"]) > classvote.EXPIRE_S:
            return None
        return cls

    def _ncfg(self):
        c = self.cfg
        return (c["corridor_m"], c["corridor_rel"], c["path_max_m"], c["ttc_min_closing"], c["in_path_sigma"],
                c["in_path_slack_m"], c["threat_ttc_s"], c["threat_min_mps"], c["with_heading_deg"], c["with_speed"][0],
                c["with_speed"][1], c["with_min_mps"])

    def _numbers(self, T, shown, X, Y, psi, v_ego, w_ego):
        """Every listed object's numbers in the car frame of time T, as the message carries them (rounded), with
        the path offset and the threat probability also as they are: [(x_m, y_m, vx_mps, vy_mps, ax_mps2, ay_mps2,
        speed_mps, heading_rad, v_rel_mps, std, models, p_stationary, angle_offset_deg, predict, offset,
        path_offset_m, on, sure_in, p_threat, ttc_s, along)]. The native core's `radar_frame` is the same."""
        c = self.cfg
        tr = self.tracker
        ch, sh = math.cos(psi), math.sin(psi)
        Rm = np.array([[ch, sh], [-sh, ch]])              # ground -> car axes
        hz = (0.0,) + tuple(c["predict_s"])
        at = tr.peek_many([o for o in shown for _ in hz], [T + h for _ in shown for h in hz])     # now and ahead, in one go
        still = tr.p_still_many(shown)
        res = []
        for n, o in enumerate(shown):
            x, P, mu = at[n * len(hz)]
            pc = Rm @ np.array([x[PX] - X, x[PY] - Y])
            vc = Rm @ np.array([x[VX], x[VY]])
            ac = Rm @ np.array([x[AX], x[AY]])
            Pp, Pv = Rm @ P[:2, :2] @ Rm.T, Rm @ P[2:4, 2:4] @ Rm.T
            speed = math.hypot(vc[0], vc[1])
            off, by = self._offset(pc[0], pc[1], v_ego, w_ego)
            on = 0.3 < pc[0] <= c["path_max_m"] and abs(off) <= c["corridor_m"] + c["corridor_rel"] * pc[0]
            sy = math.sqrt(max(float(Pp[1, 1]), 0.0))
            sure_in = on and abs(off) + c["in_path_sigma"] * sy <= c["corridor_m"] + c["corridor_rel"] * pc[0] + c["in_path_slack_m"]
            # reached within threat_ttc_s? Each hypothesis says how likely its speed along the car's axis is
            # below the one that would just make it
            p_thr = 0.0
            if sure_in and v_ego >= c["threat_min_mps"] and o.state != "tentative":
                # reached within T at the present speeds: g = distance - T x closing speed < 0, with what is not
                # known of BOTH the distance and the object's speed (a near object whose box is cut has a poor
                # range: its closeness must not be taken for a fact)
                Tt = c["threat_ttc_s"]
                u = np.zeros(M.N)
                u[PX], u[PY], u[VX], u[VY] = ch, sh, Tt * ch, Tt * sh
                for m_ in range(len(M.MODELS)):
                    xm = o.xs[m_]
                    g = ch * (xm[PX] - X) + sh * (xm[PY] - Y) + Tt * (ch * xm[VX] + sh * xm[VY]) - Tt * v_ego
                    sg = math.sqrt(max(float(u @ o.Ps[m_] @ u), 0.0) + 0.09 * Tt * Tt)
                    p_thr += float(o.mu[m_]) * 0.5 * (1.0 + math.erf(-g / (sg * 1.4142)))
            v_rel = vc[0] - v_ego
            ttc = pc[0] / -v_rel if on and v_rel < -c["ttc_min_closing"] else None
            pred = []
            for m, h in enumerate(c["predict_s"]):
                xp, Pq, _ = at[n * len(hz) + 1 + m]
                q = Rm @ np.array([xp[PX] - X, xp[PY] - Y])
                pred.append({"t_s": h, "x_m": round(float(q[0]), 2), "y_m": round(float(q[1]), 2),
                             "std_m": round(math.sqrt(max(float(np.trace(Pq[:2, :2])), 0.0)), 2)})
            # the same object if it keeps following the road as the car does
            along = None
            if speed >= c["with_min_mps"] and v_ego >= c["with_min_mps"] and pc[0] > 0.0 \
                    and c["with_speed"][0] <= speed / v_ego <= c["with_speed"][1]:
                _, _, th = self._path_point(float(pc[0]), v_ego, w_ego)
                dh = (math.atan2(vc[1], vc[0]) - th + math.pi) % (2.0 * math.pi) - math.pi
                if abs(dh) <= math.radians(c["with_heading_deg"]):
                    along = []
                    for h in c["predict_s"]:
                        px_, py_, th = self._path_point(float(pc[0]) + speed * h, v_ego, w_ego)
                        along.append({"t_s": h, "x_m": round(px_ - off * math.sin(th), 2), "y_m": round(py_ + off * math.cos(th), 2)})
            res.append((round(float(pc[0]), 2), round(float(pc[1]), 2), round(float(vc[0]), 2), round(float(vc[1]), 2),
                        round(float(ac[0]), 2), round(float(ac[1]), 2), round(speed, 2),
                        round(math.atan2(vc[1], vc[0]), 3) if speed >= 1.0 else None, round(float(v_rel), 2),
                        [round(math.sqrt(max(float(Pp[0, 0]), 0.0)), 2), round(math.sqrt(max(float(Pp[1, 1]), 0.0)), 2),
                         round(math.sqrt(max(float(Pv[0, 0]), 0.0)), 2), round(math.sqrt(max(float(Pv[1, 1]), 0.0)), 2)],
                        {n_: round(float(m), 3) for n_, m in zip(M.MODELS, mu)}, round(still[n], 3),
                        round(math.degrees(float(x[DP])), 3), pred, off, round(off, 2), on, sure_in, p_thr,
                        None if ttc is None else round(float(ttc), 2), along))
        return res

    def frame(self, ts):
        """The `radar` message for time ts (ns), or None while the car's own motion is unknown."""
        T = ts / NS
        if self.ego.newest is None or T - self.ego.newest > 1.0:
            return None
        c = self.cfg
        self._drain(T)
        tr = self.tracker
        tr.manage(T)
        X, Y, psi = self.ego.pose_at(T)
        es = self.ego.state_at(T)
        v_ego, w_ego = es[3], es[4]
        out = []
        shown = [o for o in tr.objects if o.state != "tentative" or c["tentative"]]
        if _NR is not None:
            # the objects of the message in one native call: numbers, flags with their hysteresis, the dicts
            hz = (0.0,) + tuple(c["predict_s"])
            kin = [M._KIN4.get(o.label) or M._KIN4[None] for o in shown]
            out = _NR.radar_frame(tr.store.a, [o.slot for o in shown], [[T + h - o.t for h in hz] for o in shown], kin,
                                  M._native_par(tr.mcfg), (X, Y, psi), (v_ego, w_ego), self._ncfg(),
                                  tuple(c["predict_s"]), self.path[0] if self.path is not None else None,
                                  self.path[1] if self.path is not None else None,
                                  [0.0 if o.state == "tentative" else 1.0 for o in shown], [k[3] for k in kin],
                                  [(o.id, o.state, o.logodds, o.t0, o.hits, o.sources, o.last_seen, o.verdict, o.flags,
                                    o.label, o.range_t, o.consistency, o.pred_err, self._class(o, T)) for o in shown],
                                  self._in_since, self._threat_since, T,
                                  (c["in_path_hold_s"], c["threat_p"], c["threat_range_s"], c["threat_hold_s"],
                                   c["parked_offset_m"]))
            shown = ()
        else:
            nums = self._numbers(T, shown, X, Y, psi, v_ego, w_ego)
        for n, o in enumerate(shown):
            (x_m, y_m, vx, vy, ax, ay, speed, heading, v_rel, std, models, p_st, angle, pred, off, off_r, on, sure_in,
             p_thr, ttc, along) = nums[n]
            label = o.label
            flags = set(o.flags)
            if on:
                flags.add("on_path")
            # "it stands in the way" is the strongest thing the radar says: it needs the place to be inside the
            # corridor beyond what the place's own uncertainty could explain, and to have stayed there
            hold = self._in_since.get(o.id)
            if o.verdict == "stationary" and sure_in:
                if hold is None:
                    hold = self._in_since[o.id] = T
                if T - hold >= c["in_path_hold_s"]:
                    flags.add("in_path")
            else:
                self._in_since.pop(o.id, None)
            th0 = self._threat_since.get(o.id)
            # a threat is not STARTED on an object whose distance has lately come from cut boxes only (a
            # vehicle close beside the car: its range is the poorest there is); one raised before stays
            if p_thr >= c["threat_p"] and (th0 is not None or T - o.range_t <= c["threat_range_s"]):
                if th0 is None:
                    th0 = self._threat_since[o.id] = T
                if T - th0 >= c["threat_hold_s"]:
                    flags.add("threat")
            else:
                self._threat_since.pop(o.id, None)
            if o.verdict == "stationary" and not on and abs(off) > c["parked_offset_m"] \
                    and label not in ("human", "bike", "motorbike"):
                flags.add("parked")
            if along is not None:
                flags.add("with_traffic")
            cls = self._class(o, T)
            out.append({
                "id": o.id, "state": o.state, "exist": round(o.exist, 3), "age_s": round(T - o.t0, 2), "hits": o.hits,
                "sources": sorted(o.sources), "seen": {s: round(max(0.0, T - t), 3) for s, t in o.last_seen.items()},
                "x_m": x_m, "y_m": y_m, "vx_mps": vx, "vy_mps": vy, "ax_mps2": ax, "ay_mps2": ay,
                "speed_mps": speed, "heading_rad": heading, "v_rel_mps": v_rel, "std": std,
                "motion": o.verdict, "p_stationary": p_st, "models": models,
                "class": None if cls is None else {"label": cls["label"], "confidence": cls["confidence"],
                                                   "source": cls["source"], "votes": cls["votes"]},
                "flags": sorted(flags), "path_offset_m": off_r, "ttc_s": ttc,
                "p_threat": round(p_thr, 3),
                "consistency": None if o.consistency is None else round(o.consistency, 2),
                "pred_err_m": None if o.pred_err is None else round(o.pred_err, 2),
                "angle_offset_deg": angle, "predict": pred,
                "predict_path": along})
        live = {o.id for o in tr.objects}
        for k in [k for k in self._in_since if k not in live]:
            del self._in_since[k]
        for k in [k for k in self._threat_since if k not in live]:
            del self._threat_since[k]
        out.sort(key=lambda r: r["x_m"])
        self.seq += 1
        sensors = {}
        for src, (sen, t) in tr.sensors.items():
            sensors[src] = dict(tr.stats.get(src, {}), kind=sen.name, age_s=round(max(0.0, T - t), 3))
            if src in tr.bias:                           # the camera's pitch error as the objects have shown it
                sensors[src].update(pitch_bias_deg=round(math.degrees(tr.bias[src][0]), 3),
                                    pitch_bias_std_deg=round(math.degrees(math.sqrt(tr.bias[src][1])), 3))
        return {"type": "radar", "api": 1, "seq": self.seq, "ts_mono_ns": ts, "frame": "car", "objects": out,
                "ego": {"speed_mps": round(v_ego, 2), "yaw_rate_rps": round(w_ego, 4), "accel_mps2": round(es[5], 2),
                        "stands": self.ego.stands_at(T), "ground_frame": self.ego.frame},
                "path": None if self.path is None else "plan", "sensors": sensors,
                "counts": {"objects": len(tr.objects), "listed": len(out), "queued": len(self.queue), "dropped": self.dropped}}

    def health(self):
        return {"objects": len(self.tracker.objects), "queued": len(self.queue), "dropped": self.dropped,
                "ego_samples": len(self.ego)}
