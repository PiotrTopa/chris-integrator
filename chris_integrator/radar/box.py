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

    std_rho                                         the correlation of the x and y place errors (-1..1; with `std`
                                                    the place's whole covariance), with `std_rho: true`
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
from .tracker import (M_HITS, M_RANGE_T, M_T, OT_CONS, OT_HASVS, OT_NCHECKS, OT_NLEAF, OT_PE, OT_PERR, OT_UNPRED, OT_VS,
                      Tracker)

NS = 1e9
# the appearance vector's kinds (roadsense docs/ARCHITECTURE.md): the native core's code and the vector's length
_EMB_KINDS = {"yuv3x3": (1, 27), "reid32+yuv3x3": (2, 59)}
_LAMP_C = {"red": 1.0, "white": 2.0}
_SCALE_SRC = {"lamps": 1.0, "both": 2.0}
_NR = M.NR                      # the native core (the message's objects: radar_frame)
_UNPREDICTABLE = ("unpredictable",)

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
    # std_rho: every object carries `std_rho`, the correlation of its place's x and y errors (car axes; with `std`
    # [x, y] the place's whole covariance: a camera's place is a needle along the line of sight). path_normal: `path_offset_m`, `on_path`, `in_path` are judged ACROSS the
    # path where the object is (the offset times the cosine of the path's direction there; "surely in" by the
    # place's spread across the path, from the whole covariance), not across the car's axis
    std_rho=False, path_normal=False,
)


def _fin(v):
    if type(v) is float:                 # (the common case, without the two isinstance calls: the same answer)
        return math.isfinite(v)
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
        self._pose = False               # the pose head's fields are read in this message: (ground point, centre, heading)
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
        yv, av, sv, yd = yr.get("value"), ac.get("value"), sp.get("std"), yr.get("std")     # (each looked up once)
        self.ego.add(ts / NS, sp["value"], yv if yr.get("valid") and _fin(yv) else 0.0,
                     av if ac.get("valid") and _fin(av) else 0.0,
                     sv if _fin(sv) else 0.1, yd if _fin(yd) else 0.003, stands)

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
        # the detector's further inputs (docs/RADAR.md, section 8), as far as the tracker is set to take them
        tc = self.tracker.cfg
        xin = tc["_in_any"]
        ek = _EMB_KINDS.get(msg.get("emb_kind"), (0, 0)) if tc["in_app"] else (0, 0)
        # the pose head's fields (contract pose1; a kind this code does not know is not read)
        self._pose = msg.get("pose_kind") == "pose1" and self.cam is not None and \
            (tc["in_pose_gn"] or tc["in_pose_gc"] or tc["in_pose_yaw"]) and \
            (tc["in_pose_gn"], tc["in_pose_gc"], tc["in_pose_yaw"])
        for o in objs:
            self._box_det(dets, o, c, xin, ek, False)
        n_hi = len(dets)
        low = msg.get("objects_low")
        if tc["in_low"] and isinstance(low, list):
            for o in low:
                self._box_det(dets, o, c, xin, ek, True)
        # the detector's size limit in this view, as its boxes show it: the smallest seen, forgotten slowly
        key = round(cover["ground_min_m"]) if cover else None
        hs = [math.exp(d["z"][2]) for d in dets[:n_hi] if d["kind"] in ("rbh", "rbhs")]
        hm = self.h_min.get(key)
        if hs:
            hm = min(hs) if hm is None else min(min(hs), hm * 1.002)
            self.h_min[key] = hm
        elif hm is None and self.h_min:
            hm = float("inf")                            # a box feed, but this view has not shown a box yet
        sen = CameraBoxes(cover, c, self.focal_px, self.cam_height, hm)
        sen.emb_kind = ek[0]
        self._push((ts / NS, msg.get("sensor") or "cam", sen, dets))

    def _box_det(self, dets, o, c, xin, ek, low):
        """One box of a `road_users` message as a detection of the tracker (appended to dets), or nothing. xin: the
        tracker takes the detector's further inputs (flags, lamps, appearance, the scale rate's source: `x`, `emb`);
        low: a box of the low-score list (sloppier edges, a class vote of less weight)."""
        f, hc = self.focal_px, self.cam_height
        jit = math.radians(c["pitch_jitter_deg"])
        if not isinstance(o, dict) or not isinstance(o.get("class"), str) or not _fin(o.get("bearing_deg")):
            return
        tc = self.tracker.cfg
        b = math.radians(o["bearing_deg"])
        at = {"class": o["class"], "class_conf": o.get("class_conf"), "weight": o.get("weight", 1.0), "box": o.get("box")}
        clip = o.get("clip") or ""
        occ, flags = 0.0, xin and tc["in_flags"]
        if flags:
            # the hidden edges of the box: cut by the view (clip), by the frame (trunc), under a nearer box
            # (occ_edges). No ground angle from a hidden bottom, no height from a hidden top, no width or centre from
            # a hidden side
            for k in ("trunc", "occ_edges"):
                v = o.get(k)
                if isinstance(v, str):
                    clip += v
            if _fin(o.get("occ")):
                occ = min(max(float(o["occ"]), 0.0), 1.0)
        bpx, brel, hpxs, hrel, crel = c["bottom_px"], c["bottom_rel"], c["height_std_px"], c["height_std_rel"], c["bearing_rel"]
        if tc["sil_bearing"]:                            # the box's middle against the silhouette's: as good as its edges
            crel = tc["sil_bearing_rel"]
        if low:
            bpx, brel = bpx * tc["low_px"], tc["low_rel"]
            hpxs, hrel = hpxs * tc["low_px"], 1.25 * tc["low_rel"]
            crel *= tc["low_rel"] / 0.04
            at["weight"] = tc["low_weight"] * (at["weight"] if _fin(at["weight"]) else 1.0)
            at["low"] = True
        box = o.get("box")
        box = [float(v) for v in box] if isinstance(box, (list, tuple)) and len(box) == 4 and all(_fin(v) for v in box) else None
        # the pose head (docs/RADAR.md, section 10): the footprint centre's column for the bearing, the near-face
        # ground point for the ground angle, the heading
        po = o.get("pose") if self._pose and box and not low else None
        gn = gcb = yaw = None
        if isinstance(po, dict):
            kd = po.get("kind")
            if not (isinstance(kd, list) and kd and _fin(kd[0]) and kd[0] >= tc["pose_kind_min"]) \
                    or classvote.norm_label(o["class"]) not in M.VEHICLES:
                po = None
        if isinstance(po, dict):
            use_gn, use_gc, use_yaw = self._pose
            v = po.get("gn")
            if use_gn and isinstance(v, list) and len(v) == 4 and all(_fin(q) for q in v) and v[2] > 0.0:
                gn = v
            v = po.get("gc")
            if use_gc and isinstance(v, list) and len(v) == 5 and all(_fin(q) for q in v) and v[2] > 0.0:
                gcb = (self.cam.bearing(v[0]), math.hypot(v[2] / f, math.radians(c["bearing_floor_deg"])))
                b = gcb[0]
            if use_yaw and all(_fin(po.get(k)) for k in ("yaw", "yaw_std", "p", "flip")):
                yaw = (-math.radians(po["yaw"]), math.radians(po["yaw_std"]) ** 2, float(po["p"]), float(po["flip"]),
                       1.0 if po.get("vp") == "front" and _fin(po.get("vp_conf")) and po["vp_conf"] >= 0.8 else 0.0)
        hpx = box[3] - box[1] if box else None
        wa = (box[2] - box[0]) / f if box and box[2] > box[0] else None      # the box's width as an angle
        wok = bool(box) and "l" not in clip and "r" not in clip             # ... and it is the object's own
        if box and not clip and hpx and hpx >= 6.0 and wa:
            at["aspect"] = (box[2] - box[0]) / hpx   # of a whole box: which way a vehicle points
        # the box's centre: as good as its edges, unless a side is cut (then the centre is not the object's)
        if gcb is not None:                              # the pose head's centre: its own column, its own sigma
            sb = gcb[1]
        elif box and "l" not in clip and "r" not in clip:
            sb = math.hypot(crel * (box[2] - box[0]) / f, math.radians(c["bearing_floor_deg"]))
        else:
            sb = math.radians(o["bearing_std_deg"]) if _fin(o.get("bearing_std_deg")) else math.radians(1.0)
        pgn = None                                       # the pose head's ground point as an angle: (angle, sigma)
        hidb = False
        if gn is not None:
            g = self.cam.ground(gn[0], gn[1])
            rg = math.hypot(g[0], g[1]) if g is not None else 0.0
            if 0.5 < rg < 200.0:
                pgn = (math.atan2(hc, rg), math.hypot(gn[2] / f, jit))
                oe, tr = o.get("occ_edges"), o.get("trunc")
                under = isinstance(oe, str) and "b" in oe
                hidb = "b" in (o.get("clip") or "") or under or (isinstance(tr, str) and "b" in tr)
                if hidb:
                    # the bottom of the box is not the object's: the head's point is the row this box lacked. The
                    # contact lies no higher in the picture than the visible bottom: the Gaussian cut there
                    sg = pgn[1] * (tc["pose_gn_occ"] if under else 1.0)
                    gb = self.cam.ground(0.5 * (box[0] + box[2]), box[3])
                    rb = math.hypot(gb[0], gb[1]) if gb is not None else 0.0
                    if rb > 0.5:
                        m, var = _NR.radar_trunc_low(pgn[0], sg, math.atan2(hc, rb))
                        pgn = (m, math.sqrt(var))
                    else:
                        pgn = (pgn[0], sg)
                elif gn[3] < tc["pose_min_seen"]:
                    pgn = None
        if pgn is not None and hidb:
            dets.append({"kind": "rb", "z": np.array([pgn[0], b]), "R": np.array([pgn[1] * pgn[1], sb * sb]), "attrs": at,
                         "w": wa, "wok": wok})
        elif _fin(o.get("range_m")) and o["range_m"] > 0.5 and not (flags and "b" in clip):
            # the ground point as the camera saw it: the angle below the horizon of the bottom edge
            r = float(o["range_m"])
            ang = math.atan2(hc, r)
            if hpx is not None and hpx > 0.0:
                sa = math.hypot((bpx + brel * hpx) / f, jit)
            else:                                    # no box given: what the stated range error is as an angle
                sr = float(o["range_std_m"]) if _fin(o.get("range_std_m")) else 0.2 * r
                sa = max(sr * hc / (r * r + hc * hc), jit)
            if pgn is not None and pgn[1] < sa:          # a whole box: ONE row, whichever is the surer (the same pixels)
                ang, sa = pgn
            else:
                pgn = None
            if hpx is not None and hpx >= 6.0 and "t" not in clip and "b" not in clip:
                sh = (hpxs + hrel * hpx) / hpx          # of the logarithm
                if occ > 0.0:                            # partly covered with no edge hidden: its size is less sure
                    sh *= 1.0 + occ
                dets.append({"kind": "rbh", "z": np.array([ang, b, math.log(hpx * self.frame_scale)]),
                             "R": np.array([sa * sa, sb * sb, sh * sh]), "attrs": at, "w": wa, "wok": wok})
            else:
                dets.append({"kind": "rb", "z": np.array([ang, b]), "R": np.array([sa * sa, sb * sb]), "attrs": at,
                             "w": wa, "wok": wok})
        elif "b" in clip:
            top = None
            if box and "t" not in clip and self.cam is not None and hpx:
                ry = self.cam.ray(0.5 * (box[0] + box[2]), box[1])
                top = math.atan2(ry[2], math.hypot(ry[0], ry[1]))          # below the horizon: positive
            if top is not None:
                st = math.hypot((bpx + brel * hpx) / f, jit)
                dets.append({"kind": "bt", "z": np.array([b, top]), "R": np.array([sb * sb, st * st]), "attrs": at,
                             "w": wa, "wok": wok})
            else:
                dets.append({"kind": "b", "z": np.array([b]), "R": np.array([sb * sb]), "attrs": at, "w": wa, "wok": wok})
        else:
            return
        d = dets[-1]
        self._scale_rate(d, o)
        if not xin:
            return
        if pgn is not None or gcb is not None or yaw is not None:
            d["p"] = (0.0, 1.0 if pgn is not None else 0.0, 1.0 if gcb is not None else 0.0, 1.0 if yaw is not None else 0.0) \
                + (yaw or (0.0, 0.0, 0.0, 0.0, 0.0))
        if occ >= 0.1:
            d["wok"] = False                             # a width (and a length) is not learnt from a covered box
        hid = ("l" in clip) + 2 * ("t" in clip) + 4 * ("r" in clip) + 8 * ("b" in clip)
        lc = ls = lv = 0.0
        lm = o.get("lamps")
        if (tc["in_lamp_colour"] or tc["in_lamp_size"]) and isinstance(lm, dict) and not low:
            sp, sd = lm.get("sep"), lm.get("sep_std")
            if _fin(sp) and _fin(sd) and sd > 0.0 and sp >= tc["lamp_min_px"]:
                lc = _LAMP_C.get(lm.get("c"), 3.0)
                ls, lv = math.log(sp / f), (sd / sp) ** 2
        src = _SCALE_SRC.get(o.get("scale_src"), 0.0) if tc["in_scale"] or tc["in_lamp_size"] else 0.0
        top = bot = math.nan
        tr = o.get("trunc")                              # (the bottom cut by the view or the frame, not by a box)
        cut = 1.0 if "b" in (o.get("clip") or "") or (isinstance(tr, str) and "b" in tr) else 0.0
        if tc["in_box_occl"] and box and self.cam is not None:
            # the box in the picture as angles below the horizon (what it hides of the things behind it); its
            # bottom cut by the view or the frame is lower than it shows
            u, ray = 0.5 * (box[0] + box[2]), self.cam.ray
            ry = ray(u, box[1])
            top = math.atan2(ry[2], math.hypot(ry[0], ry[1]))
            ry = ray(u, box[3])
            bot = math.atan2(ry[2], math.hypot(ry[0], ry[1]))
        d["x"] = (float(hid), occ, 1.0 if low else 0.0, lc, ls, lv, src, top, bot, cut)
        e = o.get("emb")
        if ek[0] and isinstance(e, str) and len(e) == 2 * ek[1]:
            try:
                d["emb"] = bytes.fromhex(e)
            except ValueError:
                pass

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
        rv, rmax, fov = (sr * sr, sa * sa, sv * sv), sen.max_range, sen.fov      # (the same for every detection)
        for o in objs:
            if not isinstance(o, dict):
                continue
            r, az, rr = o.get("range_m"), o.get("azimuth_deg"), o.get("range_rate_mps")
            if not (_fin(r) and _fin(az) and _fin(rr)):
                continue
            r, az = float(r), math.radians(az)
            if r < 0.5 or r > rmax or abs(az) > fov:
                continue
            dets.append({"z": np.array([r, az, float(rr)]), "R": np.array(rv),
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
        if not self.queue:
            return
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
                c["with_speed"][1], c["with_min_mps"], 1.0 if c["std_rho"] else 0.0, 1.0 if c["path_normal"] else 0.0)

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
        ctx = tr.cfg["ctx_prior"]
        if ctx:
            tr.road_path = None
        # the road model's path on the ground: for the heading prior (not made within road_prior_s of the prior's
        # last turn: Tracker.road_prior does nothing then), and for where a new object is beside the road
        due = tr.cfg["road_prior"] and (tr.road_t is None or not T - tr.road_t < tr.cfg["road_prior_s"])
        if path is not None and (due or ctx):
            pp = self.ego.pose_at(self.path_t)
            if pp is not None:
                cp, sp = math.cos(pp[2]), math.sin(pp[2])
                gp = (pp[0] + cp * path[0] - sp * path[1], pp[1] + sp * path[0] + cp * path[1])
                if ctx:
                    tr.road_path = gp
                if due:
                    tr.road_prior(T, gp)
        X, Y, psi = self.ego.pose_at(T)
        es = self.ego.state_at(T)
        v_ego, w_ego = es[3], es[4]
        slots, tb, rec = tr.looked()                     # the objects' numbers in the tracker's arrays
        shown = tr.objects
        if not c["tentative"]:
            keep = [o.state != "tentative" for o in shown]
            if not all(keep):
                slots, tb, rec = ([v for v, k in zip(q, keep) if k] for q in (slots, tb, rec))
                shown = [o for o, k in zip(shown, keep) if k]
        # the objects of the message in one native call (chris_native.c `radar_frame`): each one taken to now and
        # to the horizons, in the car frame; on the path (inside its corridor) or not; "it stands in the way" (the
        # strongest thing the radar says: the place inside the corridor beyond what its own uncertainty could
        # explain, and stayed there); reached within threat_ttc_s at the present speeds, over every hypothesis and
        # with what is not known of both the distance and the object's speed (not started on an object whose
        # distance has lately come from cut boxes only); parked; driving with the car and where it will be if it
        # keeps following the road as the car does; and the message's dict of it
        hz = (0.0,) + tuple(c["predict_s"])
        kin = [o.kin for o in shown]
        out = _NR.radar_frame(tr.store.a, slots, [[T + h - m[M_T] for h in hz] for m in rec], kin,
                              M._native_par(tr.mcfg), (X, Y, psi), (v_ego, w_ego), self._ncfg(), tuple(c["predict_s"]),
                              path[0] if path is not None else None, path[1] if path is not None else None,
                              [0.0 if o.state == "tentative" else 1.0 for o in shown], [k[3] for k in kin],
                              [(o.id, o.state, math.log(max(r[OT_PE], 1e-6) / max(1.0 - r[OT_PE], 1e-6)), o.t0, int(m[M_HITS]),
                                o.sources, o.last_seen, o.verdict, _UNPREDICTABLE if r[OT_UNPRED] else (), o.label,
                                m[M_RANGE_T], r[OT_CONS] if r[OT_NCHECKS] else None, r[OT_PERR] if r[OT_NCHECKS] else None,
                                self._class(o, T)) for o, r, m in zip(shown, tb, rec)],
                              self._in_since, self._threat_since, T,
                              (c["in_path_hold_s"], c["threat_p"], c["threat_range_s"], c["threat_hold_s"],
                               c["parked_offset_m"]))
        ch, sh = math.cos(psi), math.sin(psi)
        for o, d, r, m in zip(shown, out, tb, rec):      # the smoothed velocity (car axes) and the group
            if not r[OT_HASVS]:
                d["vs"] = d["vs_std"] = d["vs_age_s"] = None
            else:
                vx, vy, sx, sy, age = r[OT_VS:OT_VS + 5]
                d["vs"] = [round(ch * vx + sh * vy, 2), round(-sh * vx + ch * vy, 2)]
                d["vs_std"] = round(math.hypot(sx, sy), 2)
                d["vs_age_s"] = round(age + T - m[M_T], 2)
            d["grp"], d["grp_kind"] = o.grp, o.grp_kind
            d["hyp"] = int(r[OT_NLEAF])
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
                "path": None if path is None else "plan", "sensors": sensors,
                "groups": [{"id": g, "kind": v["kind"], "members": v["members"]} for g, v in sorted(tr.groups.items())],
                "counts": {"objects": len(tr.objects), "listed": len(out), "queued": len(self.queue), "dropped": self.dropped}}

    def health(self):
        return {"objects": len(self.tracker.objects), "queued": len(self.queue), "dropped": self.dropped,
                "ego_samples": len(self.ego)}
