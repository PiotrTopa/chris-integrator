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

    ctr [x, y]                                      the object's centre; x_m, y_m (and predict) are its NEAR FACE
    len, wid, hdg, ext_std [len, wid, hdg]          its rectangle: length, width (m), where its front points (rad,
                                                    car frame, left positive; held when it stands) and their 1 sigma
    vs [vx, vy], vs_std, vs_age_s                   its velocity over the ground vs_age_s ago, smoothed (from the
                                                    sightings since as well), car axes of now; null before a second sighting
    grp, grp_kind                                   the group it is of (id; null) and its kind: "row" (standing in a
                                                    line beside the road) | "queue" (standing in the car's lane) |
                                                    "platoon" (driving one behind the other); the message's `groups`
                                                    lists them with their members in order
    hyp                                             how many hypotheses the tracker holds of it (1: no doubt)
    vis, occ, occ_by                                how much of it its last scan's sensor could see (0..1); what
                                                    hides it ("object" | "edge" | "fov" | null) and the hiding
                                                    object's id (docs/RADAR.md)

Scans are processed in time order once the car's own motion is known up to their time; a scan that arrives
late is applied to the objects taken back to its time (tracker.py).
"""
import math
import operator

import numpy as np

from ..core import classvote
from ..core.roadusers import Camera
from ..native import round                  # builtins.round, native where built
from . import models as M
from .ego import EgoMotion
from .sensors import CAMERA_CFG, CameraBoxes, MicrowaveRadar
from .tracker import Tracker

NS = 1e9
_NR = M.NR                      # the native core (the message's objects: radar_frame)

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
    # microwave radars that give several returns per object (a message may say so itself: "extended": true):
    # True (all), or the sensor ids; how near in place and in range rate two returns of one cell are
    radar_extended=False, cell_m=1.6, cell_mps=1.5, cell_margin_m=0.6,
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
            wok = bool(box) and "l" not in clip and "r" not in clip             # ... and it is the object's own
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
                                 "R": np.array([sa * sa, sb * sb, sh * sh]), "attrs": at, "w": wa, "wok": wok})
                else:
                    dets.append({"kind": "rb", "z": np.array([ang, b]), "R": np.array([sa * sa, sb * sb]), "attrs": at,
                                 "w": wa, "wok": wok})
                self._scale_rate(dets[-1], o)
            elif "b" in clip:
                top = None
                if box and "t" not in clip and self.cam is not None and hpx:
                    ry = self.cam.ray(0.5 * (box[0] + box[2]), box[1])
                    top = math.atan2(ry[2], math.hypot(ry[0], ry[1]))          # below the horizon: positive
                if top is not None:
                    st = math.hypot((c["bottom_px"] + c["bottom_rel"] * hpx) / f, jit)
                    dets.append({"kind": "bt", "z": np.array([b, top]), "R": np.array([sb * sb, st * st]), "attrs": at,
                                 "w": wa, "wok": wok})
                else:
                    dets.append({"kind": "b", "z": np.array([b]), "R": np.array([sb * sb]), "attrs": at, "w": wa, "wok": wok})
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
        ext = self.cfg["radar_extended"]
        if msg.get("extended") is True or ext is True or (isinstance(ext, (list, tuple)) and sid in ext):
            # a sensor that gives several returns per object: the returns of one scan are grouped into cells (the
            # returns on a known object; the rest by nearness in place and range rate), and a cell is one detection:
            # the mean of its returns, how many they are, and how they are spread (the tracker reads the spread as
            # the object's sides: docs/RADAR.md)
            # the tracker groups them (it knows where the objects are): Tracker._cells
            sen.extended = (self.cfg["cell_m"], self.cfg["cell_mps"], self.cfg["cell_margin_m"])
            dets = [(float(o["range_m"]), math.radians(o["azimuth_deg"]), float(o["range_rate_mps"])) for o in objs
                    if isinstance(o, dict) and _fin(o.get("range_m")) and _fin(o.get("azimuth_deg")) and _fin(o.get("range_rate_mps"))]
            dets = [z for z in dets if 0.5 <= z[0] <= sen.max_range and abs(z[1]) <= sen.fov]
            self._push((ts / NS, sid, sen, dets))
            return
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
            if all(map((0.0).__le__, map(operator.sub, xs[1:], xs))) and xs[-1] > 5.0:
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

    def frame(self, ts):
        """The `radar` message for time ts (ns), or None while the car's own motion is unknown."""
        T = ts / NS
        if self.ego.newest is None or T - self.ego.newest > 1.0:
            return None
        c = self.cfg
        self._drain(T)
        tr = self.tracker
        tr.manage(T)
        path = self.path
        if path is not None and tr.cfg["road_prior"]:   # the road model's path on the ground, for the heading prior
            pp = self.ego.pose_at(self.path_t)
            if pp is not None:
                cp, sp = math.cos(pp[2]), math.sin(pp[2])
                tr.road_prior(T, (pp[0] + cp * path[0] - sp * path[1], pp[1] + sp * path[0] + cp * path[1]))
        X, Y, psi = self.ego.pose_at(T)
        es = self.ego.state_at(T)
        v_ego, w_ego = es[3], es[4]
        shown = [o for o in tr.objects if o.state != "tentative" or c["tentative"]]
        # the objects of the message in one native call (chris_native.c `radar_frame`): each one taken to now and
        # to the horizons, in the car frame; on the path (inside its corridor) or not; "it stands in the way" (the
        # strongest thing the radar says: the place inside the corridor beyond what its own uncertainty could
        # explain, and stayed there); reached within threat_ttc_s at the present speeds, over every hypothesis and
        # with what is not known of both the distance and the object's speed (not started on an object whose
        # distance has lately come from cut boxes only); parked; driving with the car and where it will be if it
        # keeps following the road as the car does; and the message's dict of it
        hz = (0.0,) + tuple(c["predict_s"])
        kin = [o.kin for o in shown]
        out = _NR.radar_frame(tr.store.a, [o.slot for o in shown], [[T + h - o.t for h in hz] for o in shown], kin,
                              M._native_par(tr.mcfg), (X, Y, psi), (v_ego, w_ego), self._ncfg(), tuple(c["predict_s"]),
                              path[0] if path is not None else None, path[1] if path is not None else None,
                              [0.0 if o.state == "tentative" else 1.0 for o in shown], [k[3] for k in kin],
                              [(o.id, o.state, math.log(max(o.pe, 1e-6) / max(1.0 - o.pe, 1e-6)), o.t0, o.hits, o.sources, o.last_seen, o.verdict, o.flags,
                                o.label, o.range_t, o.consistency, o.pred_err, self._class(o, T)) for o in shown],
                              self._in_since, self._threat_since, T,
                              (c["in_path_hold_s"], c["threat_p"], c["threat_range_s"], c["threat_hold_s"],
                               c["parked_offset_m"]))
        ch, sh = math.cos(psi), math.sin(psi)
        for o, d in zip(shown, out):                     # the smoothed velocity (car axes) and the group
            vs = o.vs
            if vs is None:
                d["vs"] = d["vs_std"] = d["vs_age_s"] = None
            else:
                d["vs"] = [round(ch * vs[0] + sh * vs[1], 2), round(-sh * vs[0] + ch * vs[1], 2)]
                d["vs_std"] = round(math.hypot(vs[2], vs[3]), 2)
                d["vs_age_s"] = round(vs[4] + T - o.t, 2)
            d["grp"], d["grp_kind"] = o.grp, o.grp_kind
            d["hyp"] = o.nleaf
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
                "groups": [{"id": g, "kind": v["kind"], "members": v["members"]} for g, v in sorted(tr.groups.items())],
                "counts": {"objects": len(tr.objects), "listed": len(out), "queued": len(self.queue), "dropped": self.dropped}}

    def health(self):
        return {"objects": len(self.tracker.objects), "queued": len(self.queue), "dropped": self.dropped,
                "ego_samples": len(self.ego)}
