"""A simulator of the camera's class detector (roadsense-signs `signd --lights-road-users`) and of the traffic it
sees, with ground truth: the judge's side of the "visual radar" evaluation (docs/VISUAL_RADAR_EVAL.md).

    w = scenario("parked_street", seed=3, night=True)        # a ScenarioWorld (a sim.World with actors)
    det = DetectorSim(seed=3, night=True)
    for t, msg in w.messages(default_sensors(...) + [det]):  # `lights` messages among the stack's other inputs
        ...
    w.actors_at(t)                                           # truth per actor in the car frame
    det.log                                                  # what was emitted, per message, with the actor ids

Nothing here imports the tracker; nothing here is tuned to make a tracker look good. The numbers come from
core/roadusers.py's documented uncertainties and from the two recorded night drives of 2026-11-22 (1240 light
messages in 6 min, 3.4 Hz, 0.37 road-user boxes per message, box heights >= 22 px in the wide view and >= 13 px
in the zoom view, scores p10/50/90 0.44/0.61/0.81, 40 % of the boxes cut at the bottom of the region).

Frames and signs: World's car frame is x forward, y LEFT, z ignored; the camera frame used for projection is
x forward, y RIGHT, z DOWN with the camera at the origin and the road `height_m` below it (core/roadusers.Camera).
"""
import json
import math
import os
import random

import numpy as np

from .sensors import _GM
from .world import CAM_TO_AXLE, NS, World

CALIB = {"type": "calib", "pitch": 0.02, "yaw": -0.01, "roll": 0.005, "height_m": 1.22, "focal_px": 975, "cx": 640,
         "cy": 360, "width": 1280, "height": 720, "source": "config"}

# ---- actors ---------------------------------------------------------------------------------------------------
DIMS = {"car": (4.5, 1.8, 1.5), "truck": (8.0, 2.5, 3.4), "human": (0.5, 0.6, 1.75), "post": (0.25, 0.25, 1.6),
        "bike": (1.8, 0.6, 1.7)}
VEHICLE_KINDS = ("car", "truck")
LANE_W = 3.5


class Actor:
    """One road user. Placement on the road: `s0` the road arc length (m, absolute, same axis as World.S) of its
    footprint centre at the start, `speed` [(t, m/s along the road; negative = against the ego's direction)],
    `lateral` [(t, m left of the ORIGINAL ego lane's centre)], `yaw_rel` its heading relative to the road while it
    does not move (rad), `t_on`/`t_off` (s from the start) when it exists. `role` names it in the reports
    (parked, oncoming, crossing, adjacent, pedestrian, crossing_ped, stopped_in_lane, slow_lead, lead, overtaken)."""

    def __init__(self, aid, kind, role, s0, speed=((0, 0.0),), lateral=((0, 0.0),), yaw_rel=0.0, t_on=0.0, t_off=1e9,
                 dims=None, group=None, group_kind=None, group_t=(-1e9, 1e9)):
        self.id, self.kind, self.role, self.s0 = aid, kind, role, float(s0)
        self.speed, self.lateral, self.yaw_rel = list(speed), list(lateral), float(yaw_rel)
        self.t_on, self.t_off = float(t_on), float(t_off)
        self.L, self.W, self.H = dims or DIMS[kind]
        # truth group membership (a parked row, a queue, a platoon): its id and kind while (s from the start)
        # group_t[0] <= t < group_t[1]
        self.group, self.group_kind, self.group_t = group, group_kind, (float(group_t[0]), float(group_t[1]))


# per-actor size spread (world option dims_spread): (L, W, H) relative log-normal sigmas around the nominal (or the
# scenario's explicit) size. A: European fleet, small cars to SUVs (about 1.7-1.95 m wide, 1.4-1.75 m tall); vans and
# trucks half the relative spread; pedestrians the height only.
DIMS_SPREAD = {"car": (0.08, 0.04, 0.06), "truck": (0.04, 0.02, 0.03), "human": (0.0, 0.0, 0.05),
               "bike": (0.0, 0.0, 0.05), "post": (0.0, 0.0, 0.0)}


def spread_dims(actors, seed):
    """Scale each actor's (L, W, H) by exp(N(0, sigma)) per DIMS_SPREAD, from a generator of its own per actor
    (seeded from the scenario seed and the actor id; no other stream is touched)."""
    for a in actors:
        sg = DIMS_SPREAD.get(a.kind, (0.0, 0.0, 0.0))
        r = random.Random(int(seed) * 1_000_003 + int(a.id) * 7_919 + 104_729)
        f = [math.exp(r.gauss(0.0, s)) for s in sg]
        a.L, a.W, a.H = a.L * f[0], a.W * f[1], a.H * f[2]


def _plan(plan, t):
    return np.interp(t, [p[0] for p in plan], [p[1] for p in plan])


class ScenarioWorld(World):
    """A World whose other road users are `Actor`s (vehicles, pedestrians, crossing traffic), with optional ego
    lane change (an S of the ego's path; the road model's plan follows the old lane until the ego starts to
    move over), a lane-based camera lead (World.lead_at uses the target's road offset; here the lane the ego is
    in), and per-actor truth in the car frame.

    spec: dict(T, speed [(t, v)], curv [(arc m, 1/m)], accel_lim, grade_amp, lane_change (s_start, length,
    shift) or None, actors: fn(world) -> [Actor] (called after the ego's motion is known), lead_range_m)."""

    def __init__(self, spec, seed=1, name="custom", night=False, world=None):
        self.name, self.night, self.spec = name, night, spec
        # world options (all off by default; the world is bit-identical then): the environment's CHRIS_SIM_DET key
        # "world" (or a preset's), then `world` (explicit wins). dims_spread: per-actor size spread (spread_dims)
        self.world_opts = {}
        env = os.environ.get("CHRIS_SIM_DET")
        if env:
            self.world_opts.update(det_params_arg(env).get("world") or {})
        self.world_opts.update(world or {})
        curv = list(spec.get("curv") or [(0, 0.0)])
        lc = spec.get("lane_change")
        if lc:
            s1, ln, sh = lc
            k = 4.0 * sh / (ln * ln)
            curv = [(0, 0.0), (s1, 0.0), (s1, k), (s1 + ln / 2, k), (s1 + ln / 2, -k), (s1 + ln, -k), (s1 + ln, 0.0)]
        super().__init__(seed=seed, scenario=dict(T=spec["T"], speed=spec["speed"], curv=curv), targets=[],
                         statics=False, grade_amp=spec.get("grade_amp", 0.03), accel_lim=spec.get("accel_lim", 2.5))
        self.lead_range = float(spec.get("lead_range_m", 100.0))
        # the camera road model's lead (read by sim.VisionSim through lead_at): reported with probability 1 up to
        # 60 % of lead_range, falling linearly to 0 at lead_range; a slowly varying range-scale error (the road
        # model's own stated x_std is 0.5 m + 6 % of the distance; VisionSim alone adds only 1.2 %) and a white
        # speed error. lead_noise=False: VisionSim's own noise only.
        self.lead_noise = spec.get("lead_noise", True)
        self._lead_rng = random.Random(seed * 13 + 5)
        self._lead_gm = _GM(self._lead_rng, 0.08 if night else 0.05, 3.0)
        self._lead_vsig = (0.45, 0.022) if night else (0.3, 0.015)
        # lateral displacement of the ego path from the original lane (only a straight road may change lane)
        h0 = float(self.heading[0])
        if lc:
            self.Lofs = -math.sin(h0) * self.px + math.cos(h0) * self.py
            self.Lofs = self.Lofs - self.Lofs[0]
            self.lc_s0 = lc[0]
        else:
            self.Lofs = np.zeros(len(self.S))
            self.lc_s0 = None
        self.actors = list(spec["actors"](self)) if spec.get("actors") else []
        if self.world_opts.get("dims_spread"):
            spread_dims(self.actors, seed)
        self._actor_arrays()

    # ---- geometry helpers ----------------------------------------------------------------------------------
    def lane_shift(self, s):
        return np.interp(s, self.S, self.Lofs)

    def _road_xy_v(self, s, d):
        hd = np.interp(s, self.S, self.heading)
        return np.interp(s, self.S, self.px) - d * np.sin(hd), np.interp(s, self.S, self.py) + d * np.cos(hd), hd

    def _actor_arrays(self):
        tt = self.t - self.t0
        dt = self.dt
        self._A = {}
        for a in self.actors:
            v = _plan(a.speed, tt)
            s = a.s0 + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
            q = _plan(a.lateral, tt)                       # left of the original lane
            d = q - self.lane_shift(s)                     # left of the ego's path
            X, Y, hd = self._road_xy_v(s, d)
            vx, vy = np.gradient(X, dt), np.gradient(Y, dt)
            sp = np.hypot(vx, vy)
            head = np.where(sp > 0.3, np.arctan2(vy, vx), hd + a.yaw_rel)
            # one that has moved and stands now points the way it came (a car waiting at a junction's line), not
            # along the road
            mv = sp > 0.3
            if mv.any() and not mv.all():
                last = np.maximum.accumulate(np.where(mv, np.arange(len(sp)), -1))
                held = (~mv) & (last >= 0)
                head = np.where(held, np.arctan2(vy, vx)[np.maximum(last, 0)], head)
            self._A[a.id] = (s, q, X, Y, vx, vy, sp, head)

    def ego_lane(self, t):
        tr = self.truth_at(t)
        return int(round(float(self.lane_shift(tr["s"] + CAM_TO_AXLE)) / LANE_W))

    def actors_at(self, t, max_range=150.0):
        """Truth per existing actor ahead of or near the car: [{id, kind, role, x, y (centre, left), heading
        (car frame, left positive), L, W, H, ref_x, ref_y (nearest footprint point), range, vx, vy (over the ground,
        car axes), speed, moving (bool), stationary_label ("in_path" | "parked" | None), lane_offset (m from the
        ego's lane centre), dz (ground height above the ego's road plane, m), len, wid, yaw (the footprint: = L, W,
        heading), group, group_kind (truth group membership at t, or None)}]."""
        tr = self.truth_at(t)
        i = self._i(t)
        rel = t - self.t0
        c, sn = math.cos(tr["psi"]), math.sin(tr["psi"])
        k_e = self.ego_lane(t)
        s_e = tr["s"] + CAM_TO_AXLE
        g_e = float(np.interp(s_e, self.S, self.grade_S))
        alt_e = float(np.interp(s_e, self.S, self.alt_S))
        out = []
        for a in self.actors:
            if not (a.t_on <= rel <= a.t_off):
                continue
            s, q, X, Y, VX, VY, SP, HD = self._A[a.id]
            if abs(float(s[i]) - s_e) > max_range + 40.0 and abs(float(q[i])) < 40.0:
                continue                                   # far along the road (cheap pre-filter)
            dx, dy = float(X[i]) - tr["pe"], float(Y[i]) - tr["pn"]
            x, y = c * dx + sn * dy - CAM_TO_AXLE, -sn * dx + c * dy
            if x < -30.0 or math.hypot(x, y) > max_range:
                continue
            vx, vy = c * float(VX[i]) + sn * float(VY[i]), -sn * float(VX[i]) + c * float(VY[i])
            h = float(HD[i]) - tr["psi"]
            ch, sh = math.cos(h), math.sin(h)
            # nearest footprint point to the camera (the box's bottom edge / the lead's distance refer to it)
            px, py = -x * ch - y * sh, x * sh - y * ch                        # camera in the actor's frame
            px, py = max(-a.L / 2, min(a.L / 2, px)), max(-a.W / 2, min(a.W / 2, py))
            rx, ry = x + px * ch - py * sh, y + px * sh + py * ch
            lane_off = float(q[i]) - k_e * LANE_W
            moving = float(SP[i]) > 0.5
            lab = None
            if not moving:
                inner = abs(lane_off) - a.W / 2
                lab = "in_path" if inner < 1.1 and x > 0.0 else "parked"
            ds = float(s[i]) - s_e
            dz = (float(np.interp(s[i], self.S, self.alt_S)) - alt_e) - math.sin(g_e) * ds
            out.append({"id": a.id, "kind": a.kind, "role": a.role, "x": x, "y": y, "heading": h, "L": a.L, "W": a.W,
                        "H": a.H, "ref_x": rx, "ref_y": ry, "range": math.hypot(rx, ry), "vx": vx, "vy": vy,
                        "speed": float(SP[i]), "moving": moving, "stationary_label": lab, "lane_offset": lane_off,
                        "dz": dz, "len": a.L, "wid": a.W, "yaw": h,
                        "group": a.group if a.group is not None and a.group_t[0] <= rel < a.group_t[1] else None,
                        "group_kind": a.group_kind if a.group is not None and a.group_t[0] <= rel < a.group_t[1] else None})
        return out

    # ---- what the other simulated sensors read ----------------------------------------------------------------
    def lead_at(self, t, half_lane=1.5):
        """The camera road model's lead: the nearest vehicle whose centre is in the lane the ego is in, measured to
        its nearest point, within `lead_range_m` (World.lead_at: by the target's own road offset)."""
        best = None
        for o in self.actors_at(t, max_range=self.lead_range + 10.0):
            if o["kind"] not in VEHICLE_KINDS or o["ref_x"] <= 0.5 or o["ref_x"] > self.lead_range:
                continue
            if abs(o["lane_offset"]) <= half_lane and (best is None or o["ref_x"] < best["x"]):
                best = {"id": o["id"], "x": o["ref_x"], "y": o["y"], "vx": o["vx"], "vy": o["vy"], "kind": o["kind"],
                        "d": o["lane_offset"]}
        if best is None or not self.lead_noise:
            return best
        r = self.lead_range
        if self._lead_rng.random() > min(1.0, max(0.0, (r - best["x"]) / (0.4 * r))):
            return None
        e = self._lead_gm.at(t)
        best["x"] *= 1.0 + e
        best["vx"] += self._lead_rng.gauss(0.0, self._lead_vsig[0] + self._lead_vsig[1] * best["x"])
        return best

    def path_ahead(self, t, xs):
        """The road model's plan: the centre of the lane the ego is in, until the ego starts its lane change."""
        tr = self.truth_at(t)
        s_e = tr["s"] + CAM_TO_AXLE
        if self.lc_s0 is None or s_e >= self.lc_s0:
            return World.path_ahead(self, t, xs)
        out = []
        for x in xs:
            s = s_e + x
            px, py, _ = self._road_xy(s, -float(self.lane_shift(s)))
            out.append(self._to_car(tr, px, py)[1])
        return out

    def targets_at(self, t):
        return [{"id": o["id"], "x": o["x"], "y": o["y"], "vx": o["vx"], "vy": o["vy"], "kind": o["kind"],
                 "d": o["lane_offset"], "s": 0.0} for o in self.actors_at(t)]


    def road_context(self, t):
        """The road as a map and a lane tracker would give it, for the predictor (chris_integrator/predict; the dict
        its docstring lists), in the car frame of time t: the centre line of the lane the ego is in, the lanes
        beside it, junction arms and marked crossings. From the scenario's `layout` (n_left, n_right: lanes of the
        ego's direction beside the ORIGINAL ego lane; oncoming, n_oncoming, park_left, park_right, kerb_left_m,
        kerb_right_m), `junctions` [{"s", "arms": [(side, priority)], "control"}] and `crossings` [arc length].
        None for a scenario without a layout."""
        lay = self.spec.get("layout")
        if lay is None:
            return None
        tr = self.truth_at(t)
        s_e = tr["s"] + CAM_TO_AXLE
        k = self.ego_lane(t)
        pts = []
        for x in np.arange(0.0, 160.0, 5.0):
            s = s_e + float(x)
            px, py, _ = self._road_xy(s, k * LANE_W - float(self.lane_shift(s)))
            pts.append(list(self._to_car(tr, px, py)))
        out = {"centre": pts, "lane_width_m": LANE_W, "n_left": max(0, lay.get("n_left", 0) - k),
               "n_right": max(0, lay.get("n_right", 0) + k), "oncoming": 1.0 if lay.get("oncoming") else 0.0,
               "n_oncoming": lay.get("n_oncoming", 1), "park_left": lay.get("park_left", 0.0),
               "park_right": lay.get("park_right", 0.0), "quality": 0.9, "source": "sim"}
        for key in ("kerb_left_m", "kerb_right_m"):
            if lay.get(key) is not None:
                out[key] = lay[key] + (-k if key == "kerb_left_m" else k) * LANE_W
        arms = []
        for j in self.spec.get("junctions") or ():
            if not -40.0 <= j["s"] - s_e <= 160.0:
                continue
            for side, prio in j["arms"]:
                sg = 1.0 if side == "left" else -1.0
                ap = []
                for dist in np.arange(0.0, 65.0, 5.0):
                    px, py, _ = self._road_xy(j["s"], sg * float(dist) - float(self.lane_shift(j["s"])))
                    ap.append(list(self._to_car(tr, px, py)))
                arms.append({"pts": ap, "side": side, "s_m": j["s"] - s_e, "priority": prio, "control": j.get("control")})
        if arms:
            out["arms"] = arms
        xs = [{"s_m": sx - s_e, "half_m": 2.0} for sx in self.spec.get("crossings") or () if -30.0 <= sx - s_e <= 160.0]
        if xs:
            out["crossings"] = xs
        return out


# ---- the camera ------------------------------------------------------------------------------------------------
def rotation(pitch, yaw, roll):
    """device <- car (x forward, y right, z down): Rz(yaw) Ry(pitch) Rx(roll), as core/roadusers.Camera."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def project(R, f, cx, cy, P):
    """Points P (N, 3) in the camera-centred car frame (x fwd, y right, z down) -> (u, v, depth) arrays."""
    D = P @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return cx + f * D[:, 1] / D[:, 0], cy + f * D[:, 2] / D[:, 0], D[:, 0]


def ground_pixel(calib, x, y_left):
    """Pixel of a ground point (car frame, y left) through a `calib` message's camera, or None behind it."""
    R = rotation(float(calib.get("pitch") or 0.0), float(calib.get("yaw") or 0.0), float(calib.get("roll") or 0.0))
    u, v, d = project(R, float(calib.get("focal_px") or 910.0), float(calib.get("cx") or 640.0),
                      float(calib.get("cy") or 360.0), np.array([[x, -y_left, float(calib.get("height_m") or 1.22)]]))
    return None if not d[0] > 0.3 else (float(u[0]), float(v[0]))


def box_corners(o, height):
    """The actor's 3D box corners (8, 3) in the camera-centred frame (x fwd, y right, z down)."""
    ch, sh = math.cos(o["heading"]), math.sin(o["heading"])
    pts = []
    zg = height - o["dz"]
    for lx in (-o["L"] / 2, o["L"] / 2):
        for ly in (-o["W"] / 2, o["W"] / 2):
            x = o["x"] + lx * ch - ly * sh
            y = o["y"] + lx * sh + ly * ch
            pts.append((x, -y, zg))
            pts.append((x, -y, zg - o["H"]))
    return np.array(pts)


# ---- the detector ----------------------------------------------------------------------------------------------
VIEWS = (  # (view, roi) as the unit sends them; the wide view of the recordings is [0, 0, 1280, 446]
    (0, (0, 0, 1280, 446)),
    (1, (160, 90, 1120, 410)),
)

DET_DAY = dict(
    rate_hz=5.0,             # detector slots; a slot is skipped with p_skip (recorded: 3.4 Hz from 5 Hz slots)
    p_skip=0.32,
    latency_s=(0.125, 0.165),  # arrival after the frame's time (recorded lights_ms ~100 + transport; rate.latency_ms 125-160)
    # size: a box is detectable from h_min px (full-frame px) and reliable from h_full px, per view
    h_min=(16.0, 10.0), h_full=(32.0, 20.0),   # assumed (no day recording): below the night minima
    p_miss=0.15, miss_run=2.0,     # steady miss share of a detectable object, mean run length (messages)
    vru_factor=0.9,                # detection probability factor for pedestrians
    min_frac_w=0.5, min_frac_h=0.4,  # share of the box that must be inside the region
    edge_px=2.0, edge_rel=0.03,    # left/right/top edge noise (1 sigma): px + share of the box size
    bottom_px=3.0, bottom_rel=0.04,  # bottom edge noise: core/roadusers PROJ_CFG's documented numbers
    bottom_bias_rel=0.02,          # per-actor persistent bottom-edge offset (1 sigma, share of the box height)
    offaxis_deg=12.0, offaxis_factor=1.0,  # detection factor for objects more than offaxis_deg off the axis
    p_gross=0.02,                  # share of boxes placed beside the object by 0.3-1.0 box widths
    p_car_as_truck=0.03, p_truck_as_car=0.30, p_person_as_rider=0.02,
    score=(0.62, 0.12),            # mean, sigma of a true box's score (scaled by the size ramp below)
    fp_transient_per_min=1.0,      # single-message boxes at random places
    fp_fixed_per_min=0.1,          # image-fixed boxes lasting fp_dur_s (reflection, sticker, street furniture far away)
    fp_post_per_min=0.3,           # roadside posts called "person": world-fixed, lasting fp_dur_s
    fp_dur_s=(0.5, 2.0), fp_p_det=0.8,
    occl_cover=0.6,                # a box covered by nearer (reported-size) boxes for >= this share is not reported;
                                   # None: no occlusion at all (the brief's "occlusion ignored")
    pitch_err_deg=0.4,             # true pitch - pitch of the calib message (sign random per seed unless given)
    yaw_err_deg=0.2,               # true yaw - calib yaw (sign random per seed)
    pitch_per_mps2_deg=0.15,       # body pitch: nose down 0.15 deg per m/s^2 of braking (1.5 deg/g)
    pitch_bump_deg=0.08, pitch_bump_tau_s=0.4,  # road-roughness pitch (Gauss-Markov)
    # ---- detection inputs (lane detin; docs/VISUAL_RADAR_EVAL.md "Detection inputs"): every option OFF here; with
    # all of them off the messages are bit-identical to the simulator without them. M = measured (roadsense
    # docs/lanes/detin-detection-inputs.md, roadsense-signs docs/DETECTION_INPUTS.md, detin-emb-distances.json),
    # A = assumed. Everything new is drawn from a second generator (seed * 31 + 7), never from the base stream.
    visible_boxes=False,     # an actor's box = bounding box of the part of its projected box that nearer actors (and
                             # `roadside` occluders) do not cover and that lies inside the 1280 x 720 frame
    vis_grid=24,             # raster cells per box side for that (A: resolution only)
    occl_pd_from=0.15, occl_pd_at=0.5,  # detection probability x1 at cover <= occl_pd_from, falling linearly to
                             # x occl_pd_at at cover = occl_cover (dropped beyond, as before) (A)
    roadside=None,           # world occluders that are no road users (with visible_boxes; vectors: their own latent):
                             # e.g. ROADSIDE (posts, trees with crowns, low kerbside railings / hedges; density A)
    flags=False,             # trunc / occ / occ_by / occ_edges from the EMITTED boxes, signd/detin.cc's rule (ported)
    low=None,                # low-score list floor (e.g. 0.10): `road_users_low`; None = off
    low_thr=(("person", 0.40), ("rider", 0.35), ("car", 0.40), ("truck", 0.40), ("bus", 0.40), ("train", 0.50),
             ("motor", 0.35), ("bike", 0.35)),   # class thresholds of the first list (signd lights.h ru_thr): a low
                             # box scores uniformly in (low, threshold]
    p_low_run=0.68,          # a missed actor inside a miss run (miss chain state) is in the low list: M (gap passes with
                             # a low box on the object at floor 0.10: warsaw-city 68 %, night rides 75-100 %)
    p_low_draw=0.4,          # ... missed by the size ramp / off-axis / partial-cover draw: M (BDD100K: 41 % of the misses)
    p_low_small=0.5, low_h_from=(0.5, 0.8),  # below the first list's h_min: p_low_small x a ramp from low_h_from[0] to
                             # [1] x h_min (A; aims at M: boxes under 30 px found 15 % -> 46 % with the low list)
    p_low_occl=0.3, low_occl_max=0.85,  # an actor dropped for cover >= occl_cover but < low_occl_max (A: its visible
                             # part still scores)
    low_edge_x=1.5, low_edge_rel=0.10,  # low box edge noise: 1.5 x the px terms, relative term 0.10 (M, BDD100K)
    low_cls_wrong=0.05,      # class wrong (M: right 95 % against 98 %)
    low_clutter=0.65,        # low boxes on nothing per pass (M x A: ride rate of low boxes on no first-list object,
                             # day 1.22 x 0.93, x 0.57 = BDD100K's share of those on no label at all)
    low_clutter_fixed=0.3, low_clutter_dur_s=(1.0, 2.0), low_clutter_p_det=0.6,  # share of clutter at one image place
                             # for 1-2 s, re-boxed with p per pass (A: as the camera's false boxes recur)
    p_low_beside=0.15,       # per first-list actor box: a low box on part of / beside it, IoU <= the NMS limit (M range:
                             # rides 0.04-0.45 per first-list box, day 0.18; A pick)
    low_max=20, low_nms=0.5,  # signd: at most 20 low boxes a pass; dropped when a first-list or better low box
                             # overlaps by IoU > 0.5
    emb=None,                # None | "yuv3x3" | "reid32+yuv3x3": `emb` per box (both lists) and `emb_kind`
    emb_level=0.0,           # common level of the nine Y cells ((Y - 110) / 50) of the scene (A: day 0, night -1.5)
    emb_max=16,              # reid32+yuv3x3: vectors on at most this many boxes a pass (signd --lights-ru-emb-max)
    lamps=False,             # lamp pairs (night only; never by day: the detector's frame-darkness gate)
    lamp_p=0.65,             # found on a LIT vehicle facing within lamp_cone_deg: M (61-73 % of night vehicle boxes)
    lamp_p_unlit=0.05,       # ... on a parked (unlit) one: reflections of the ego's headlamps, colour red / unknown (A)
    lamp_cone_deg=40.0,      # rear or front within this of the line of sight (A)
    lamp_S=(0.67, 0.12),     # lamp separation per actor: 0.67 x width x exp(N(0, 0.12)) (A: ~1.2 m for a car)
    lamp_h_rel=0.6,          # lamp height as a share of the body height (A)
    lamp_wrong=0.1, lamp_wrong_x=(0.6, 0.9),  # a pair on the wrong blobs: separation x 0.6-0.9 (M: by-eye sheet, ~1 in 10..15)
    lamp_unknown=0.13, lamp_colour_wrong=0.03,  # colour unknown / wrong (M: 19 of 142 unknown; sheet: 28 of 30 right)
    lamp_noise=(0.35, 0.02), lamp_small_px=10.0, lamp_small_noise=0.8,  # separation noise hypot(0.35 px, 2 %) (M);
                             # under 10 px 0.8 px (M: the pair flips between lamps and reflections); stated sep_std
                             # = the detector's hypot(0.25 px per centroid, 2 %) whatever the size
    scale_corr=False,        # with `scale_rate`: per actor and pass a noisy log size, the rate = WLS slope over the
                             # window's samples (successive values share errors); lamp-pair rates with `lamps` at night
    scale_occ_max=0.3,       # no patch value from a box covered more than this (A)
    scale_inc_T=0.6,         # the per-pass alignment error's variance per second: a value over 0.6 s has the base
                             # model's std (A: the typical baseline of the recorded values)
    lamp_scale_min_n=3, lamp_scale_min_px=10.0, lamp_scale_step=0.25, lamp_scale_x=1.5,  # signd scale.cc's lamp
                             # line: >= 3 passes, separations >= 10 px, no step above 0.25 in ln, std x 1.5 (M)
    lamp_scale_tail=(0.1, 3.0),  # a lamp sample is off by tail[1] x its noise with probability tail[0] (A: heavier tail)
)
DET_NIGHT = dict(
    DET_DAY,
    h_min=(22.0, 13.0), h_full=(50.0, 32.0),   # recorded night minimum box heights: 22.4 px wide, 12.8 px zoom
    p_miss=0.45, miss_run=3.0,
    vru_factor=0.6,
    offaxis_factor=0.5,            # outside the low beam, unlit: rows of parked cars are mostly missed (recordings)
    p_gross=0.05,
    edge_px=3.0, edge_rel=0.04,
    bottom_px=4.5, bottom_rel=0.06, bottom_bias_rel=0.03,   # dark car bodies: 1.5x the documented edge noise
    score=(0.58, 0.13),
    fp_transient_per_min=3.0, fp_fixed_per_min=0.5, fp_post_per_min=1.0,
    p_low_run=0.75,          # M: a1-night-fog 75 %, krakow-night-tunnel 6 of 7, gdansk-night-snow 4 of 4
    low_clutter=0.35,        # M x A: night rides 0.55-1.04 low boxes a pass, 87-99 % on no first-list object, x 0.57
    p_low_beside=0.2,        # M range (night rides 0.17-0.45 per first-list box); A pick
    emb_level=-1.5,          # A: a dark scene (Y ~ 35)
)

# Roadside occluders that are no road users (posts, trees, low kerbside railings / hedges), per km and side of the road,
# |lateral| in `lat` m from the original lane centre (the kerb line beyond parked cars): A. Used with visible_boxes.
ROADSIDE = dict(post_per_km=25.0, tree_per_km=15.0, hedge_per_km=6.0, lat=(4.0, 5.2))

# The preset: everything on at the recommended values (the detector's documented switch-on set: floor 0.10, flags,
# model-free vector, lamps). scale_corr acts only together with a `scale_rate` dict (tools/visual_radar_eval.py
# SCALE_DAY / SCALE_NIGHT). DETIN_REID: the same with the learnt vector kind.
DETIN = dict(visible_boxes=True, roadside=ROADSIDE, flags=True, low=0.10, emb="yuv3x3", lamps=True, scale_corr=True,
             world={"dims_spread": True})   # "world": read by scenario() / ScenarioWorld (DetectorSim ignores it)
DETIN_REID = dict(DETIN, emb="reid32+yuv3x3")
DETIN_KEYS = ("visible_boxes", "roadside", "flags", "low", "emb", "lamps", "scale_corr")


def det_params_arg(s):
    """A DetectorSim parameter dict from a command line / environment string: "detin" (DETIN), "detin_reid"
    (DETIN_REID), or a JSON object ({"detin": true, ...} starts from DETIN and overrides)."""
    s = s.strip()
    if s in ("detin", "detin_reid"):
        return dict(DETIN if s == "detin" else DETIN_REID)
    d = json.loads(s)
    if not isinstance(d, dict):
        raise ValueError("detector parameters: a JSON object, 'detin' or 'detin_reid'")
    base = {}
    for k, pre in (("detin", DETIN), ("detin_reid", DETIN_REID)):
        if d.pop(k, None):
            base = dict(pre)
    base.update(d)
    return base

FLAGS_CFG = dict(edge_px=2.0, tie_px=2.0, tie_rel=0.04, edge_hidden=0.5)   # signd RuFlagsConfig defaults
FRAME = (1280, 720)
EDGE_BITS = {"l": 1, "t": 2, "r": 4, "b": 8}


def _bits(s):
    return sum(EDGE_BITS[c] for c in s)


def _letters(bits):
    return "".join(c for c, v in (("l", 1), ("t", 2), ("r", 4), ("b", 8)) if bits & v)


def ru_nearer(a, b, cfg=FLAGS_CFG):
    """signd detin.cc ru_nearer: a, b = (box, cut bits = clip | trunc). +1 a nearer, -1 b nearer, 0 not ordered."""
    ca, cb = a[1] & 8, b[1] & 8
    if ca and cb:
        return 0
    if ca:
        return 1
    if cb:
        return -1
    tie = max(cfg["tie_px"], cfg["tie_rel"] * max(a[0][3] - a[0][1], b[0][3] - b[0][1]))
    d = a[0][3] - b[0][3]
    return 1 if d > tie else (-1 if d < -tie else 0)


def flag_one(u, self_i, occluders, cfg=FLAGS_CFG):
    """signd detin.cc flag_one: u = (box, cut bits); occluders: [(box, cut bits)] -> (occ, occ_by, occ_edges bits);
    occ 0 / occ_by -1 when nothing covers it."""
    x0, y0, x1, y1 = u[0]
    w, h = x1 - x0, y1 - y0
    if not (w > 0) or not (h > 0):
        return 0.0, -1, 0
    G = 16
    cov = [[0] * G for _ in range(G)]
    best, occ_by, anyc = 0.0, -1, False
    for j, o in enumerate(occluders):
        if j == self_i:
            continue
        ob = o[0]
        ix0, iy0, ix1, iy1 = max(x0, ob[0]), max(y0, ob[1]), min(x1, ob[2]), min(y1, ob[3])
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        if ru_nearer(o, u, cfg) < 0:
            continue
        anyc = True
        a = (ix1 - ix0) * (iy1 - iy0)
        if a > best:
            best, occ_by = a, j
        c0 = max(0, int(math.ceil((ix0 - x0) / w * G - 0.5)))
        c1 = min(G - 1, int(math.floor((ix1 - x0) / w * G - 0.5)))
        r0 = max(0, int(math.ceil((iy0 - y0) / h * G - 0.5)))
        r1 = min(G - 1, int(math.floor((iy1 - y0) / h * G - 0.5)))
        for r in range(r0, r1 + 1):
            row = cov[r]
            for c in range(c0, c1 + 1):
                row[c] = 1
    if not anyc:
        return 0.0, -1, 0
    n = sum(sum(r) for r in cov)
    el = sum(cov[i][0] for i in range(G))
    er = sum(cov[i][G - 1] for i in range(G))
    et, eb = sum(cov[0]), sum(cov[G - 1])
    need = int(math.ceil(cfg["edge_hidden"] * G))
    edges = (1 if el >= need else 0) | (2 if et >= need else 0) | (4 if er >= need else 0) | (8 if eb >= need else 0)
    return n / float(G * G), (occ_by if n else -1), edges


def trunc_of(box, fw=FRAME[0], fh=FRAME[1], e=FLAGS_CFG["edge_px"]):
    """signd detin.cc trunc_of: frame edges within e px of the box, as bits."""
    return (1 if box[0] <= e else 0) | (2 if box[1] <= e else 0) | (4 if box[2] >= fw - e else 0) | (8 if box[3] >= fh - e else 0)


def ru_flags(users, low=None):
    """Fill `trunc` (always) and `occ`, `occ_by`, `occ_edges` (only when covered) into road-user dicts in place, as
    signd ru_flags: low boxes are flagged against the first list and are never occluders."""
    def cut(d):
        tb = trunc_of(d["box"])
        d["trunc"] = _letters(tb)
        return (d["box"], _bits(d.get("clip") or "") | tb)
    occl = [cut(d) for d in users]
    lows = [cut(d) for d in (low or ())]
    for lst, items, self_idx in ((users, occl, True), (low or (), lows, False)):
        for i, (d, u) in enumerate(zip(lst, items)):
            occ, by, edges = flag_one(u, i if self_idx else -1, occl)
            if occ > 0 and by >= 0:
                d["occ"], d["occ_by"], d["occ_edges"] = round(occ, 2), by, _letters(edges)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


# ---- appearance vectors: a latent per object (per-object log-normal scale), an Ornstein-Uhlenbeck drift (the longer
# since the last observation, the further it has moved) and white per-observation noise, both scaled by a per-object
# log-normal volatility and the white part also by a per-observation one; a few dominant directions (exponential
# spectra, a fixed rotation). Fitted to roadsense docs/lanes/detin-emb-distances.json (warsaw-city by day,
# krakow-night-tunnel at night; the fit and its tables: docs/VISUAL_RADAR_EVAL.md "Detection inputs").
# yuv: (sigma latent, tau latent, kappa latent spectrum, sigma drift, T drift s, sigma white, kappa noise spectrum,
# tau volatility, tau per-observation[, share of frozen objects, their volatility]); learnt (32 numbers about a common direction of length 1, then L2-normalised):
# (sigma latent, kappa, sigma drift, sigma white), sharing T, the taus and the scale draws with the yuv part.
EMB_MODEL = {
    "day": {"yuv": [0.99870, 0.55409, 14.40613, 1.51218, 24.79792, 0.46091, 0.99950, 0.05418, 0.42828, 0.08500, 0.05479],
            "learnt": [1.00161, 1.62142, 0.18533, 0.45248]},
    "night": {"yuv": [0.99107, 0.89105, 0.04000, 0.89842, 1.88598, 0.07750, 4.06507, 0.29006, 1.83907],
              "learnt": [0.24408, 5.30138, 0.39046, 0.08944]},
}


def _emb_basis(d, seed):
    q, _ = np.linalg.qr(np.random.RandomState(seed).randn(d, d))
    return q


def _emb_lam(d, kappa):
    v = np.exp(-np.arange(d) / kappa)
    return v / np.sqrt(np.sum(v * v))


class EmbGen:
    """Appearance vectors as the detector publishes them (int8 lists) for keys (actor ids, false boxes, roadside
    items). observe(key, t) steps the key's drift by the time since its last observation and adds white noise;
    `mix` (share, other key) blends towards another key's look (a partly covered box). Draws from `rng` only."""

    def __init__(self, kind, night, rng, level=0.0):
        self.kind, self.rng, self.level = kind, rng, float(level)
        m = EMB_MODEL["night" if night else "day"]
        self.y, self.l = m["yuv"], m["learnt"]
        self.QY = _emb_basis(27, 27)
        self.LM, self.LN = _emb_lam(27, self.y[2]), _emb_lam(27, self.y[6])
        if kind == "reid32+yuv3x3":
            self.QL, self.LL = _emb_basis(32, 32), _emb_lam(32, self.l[1])
        self.st = {}

    def _g(self, n):
        g = self.rng.gauss
        return np.array([g(0.0, 1.0) for _ in range(n)])

    def _new(self, key):
        sm, tm, _, sa, _, _, _, tv, _ = self.y[:9]
        so = math.exp(tm * self.rng.gauss(0.0, 1.0))
        sv = math.exp(tv * self.rng.gauss(0.0, 1.0))
        if len(self.y) > 9 and self.rng.random() < self.y[9]:
            sv = self.y[10]              # a "frozen" object: next to no change between its crops (standing scenes)
        s = {"sv": sv, "m": sm * so * self._g(27) * self.LM, "a": sa * sv * self._g(27) * self.LN, "t": None}
        if self.kind == "reid32+yuv3x3":
            sml, _, sal, _ = self.l
            s["ml"], s["al"] = sml * so * self._g(32) * self.LL, sal * sv * self._g(32) * self.LL
        self.st[key] = s
        return s

    def _step(self, s, t):
        if s["t"] is not None and t > s["t"]:
            rho = math.exp(-(t - s["t"]) / self.y[4])
            c = math.sqrt(max(0.0, 1.0 - rho * rho))
            s["a"] = rho * s["a"] + c * self.y[3] * s["sv"] * self._g(27) * self.LN
            if "al" in s:
                s["al"] = rho * s["al"] + c * self.l[2] * s["sv"] * self._g(32) * self.LL
        s["t"] = t if s["t"] is None else max(s["t"], t)

    def _raw(self, s, noise=True):
        xy, xl = s["m"] + s["a"], (s["ml"] + s["al"]) if "ml" in s else None
        if noise:
            go = s["sv"] * math.exp(self.y[8] * self.rng.gauss(0.0, 1.0))
            xy = xy + self.y[5] * go * self._g(27) * self.LN
            if xl is not None:
                xl = xl + self.l[3] * go * self._g(32) * self.LL
        return xy, xl

    def observe(self, key, t, mix=0.0, other=None):
        s = self.st.get(key) or self._new(key)
        self._step(s, t)
        xy, xl = self._raw(s)
        if mix > 0.0:
            so = self.st.get(other) if other is not None else None
            if so is None:
                so = self._new(other if other is not None else ("anon", len(self.st)))
            self._step(so, t)
            oy, ol = self._raw(so, noise=False)
            xy = (1.0 - mix) * xy + mix * oy
            if xl is not None:
                xl = (1.0 - mix) * xl + mix * ol
        y = self.QY @ xy
        y[:9] += self.level
        out = [int(v) for v in np.clip(np.round(y * 32.0), -127, 127)]
        if xl is not None:
            v = self.QL @ xl
            v[0] += 1.0
            v = v / max(1e-9, float(np.linalg.norm(v)))
            out = [int(q) for q in np.clip(np.round(v * 127.0), -127, 127)] + out
        return out


def emb_hex(v):
    return "".join("%02x" % (q & 0xFF) for q in v)


def emb_unhex(s):
    b = bytes.fromhex(s)
    return [q - 256 if q > 127 else q for q in b]


def emb_distance(a, b, kind):
    """The contract's distance between two published vectors (int8 lists or hex)."""
    a = emb_unhex(a) if isinstance(a, str) else a
    b = emb_unhex(b) if isinstance(b, str) else b
    if kind == "yuv3x3":
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / 32.0
    dl = math.sqrt(sum((x - y) ** 2 for x, y in zip(a[:32], b[:32]))) / 127.0
    dy = math.sqrt(sum((x - y) ** 2 for x, y in zip(a[32:], b[32:]))) / 32.0
    return math.sqrt(dl * dl + (0.5 * dy) ** 2)


def _hull(P):
    """Convex hull (CCW) of image points (n, 2)."""
    pts = sorted(set(map(tuple, np.round(P, 4).tolist())))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])


def _in_polys(X, Y, polys):
    """Grid points inside any of the convex CCW polygons."""
    m = np.zeros(X.shape, bool)
    for ax, ay, ex, ey in polys:
        m |= np.all(ex * (Y[None] - ay) - ey * (X[None] - ax) >= 0, axis=0)
    return m


def _poly_edges(P):
    """A convex CCW polygon as broadcastable edge arrays for _in_polys (None when degenerate)."""
    if len(P) < 3:
        return None
    B = np.roll(P, -1, axis=0)
    return (P[:, 0, None, None], P[:, 1, None, None], (B[:, 0] - P[:, 0])[:, None, None], (B[:, 1] - P[:, 1])[:, None, None])


def _wls_slope(samples):
    """[(t, y, sigma)] -> (slope, its 1-sigma) of the weighted least-squares line."""
    sw = sum(1.0 / (s * s) for _, _, s in samples)
    tm = sum(t / (s * s) for t, _, s in samples) / sw
    ym = sum(y / (s * s) for _, y, s in samples) / sw
    stt = sum((t - tm) ** 2 / (s * s) for t, _, s in samples)
    if stt <= 0:
        return None, None
    sl = sum((t - tm) * (y - ym) / (s * s) for t, y, s in samples) / stt
    return sl, 1.0 / math.sqrt(stt)

FP_CLASSES = (("car", 0.55), ("person", 0.45))


class DetectorSim:
    """`lights` messages with `road_users` as the phone's unit sends them. Deterministic per seed.

    params: DET_DAY / DET_NIGHT overrides (see the dicts: every number is documented there and in
    docs/VISUAL_RADAR_EVAL.md). calib: the calibration the unit publishes (the stack is given this one); the true
    camera differs from it by pitch_err_deg / yaw_err_deg plus the body pitch of braking and road roughness, and
    the road ahead is not flat (World's grade profile: the actors' ground height relative to the ego's plane)."""

    def __init__(self, seed=51, night=False, params=None, calib=None, pitch_sign=None, yaw_sign=None, views=VIEWS):
        # parameters: the day / night dict, then the environment's CHRIS_SIM_DET (a JSON object, or "detin" /
        # "detin_reid" for the presets: lets whole-stack tools that build DetectorSim(seed, night) themselves switch
        # the detection-input options on), then `params` (explicit parameters win)
        self.p = dict(DET_NIGHT if night else DET_DAY)
        env = os.environ.get("CHRIS_SIM_DET")
        if env:
            self.p.update(det_params_arg(env))
        self.p.update(params or {})
        self.seed, self.night, self.calib, self.views = seed, night, dict(calib or CALIB), views
        rng = random.Random(seed * 7 + 3)
        self.pitch_err = math.radians(self.p["pitch_err_deg"]) * (pitch_sign if pitch_sign else rng.choice((-1, 1)))
        self.yaw_err = math.radians(self.p["yaw_err_deg"]) * (yaw_sign if yaw_sign else rng.choice((-1, 1)))
        self.log = []            # [(t, view, [(actor id | "fp:...", cls, box, clip)])]
        self._seen = {}          # actor -> [(time, true box height)] of its whole boxes within the baseline window
        # detection inputs (only with an option on): the low list as the first list's log, and per message the truth
        # of every emitted box of both lists (see _generate_detin)
        self.log_low = []        # [(t, view, [(actor id | "fp:clutter" | "fp:clutter_fixed" | "fp:beside:<id>", cls, box, clip)])]
        self.log_x = []          # [{"t", "view", "first": [truth per box], "low": [truth per box]}]
        self._detin = any(self.p.get(k) for k in DETIN_KEYS) or self.p.get("low") is not None

    def calib_msg(self, ts_ns):
        return dict(self.calib, ts_mono_ns=ts_ns, ts_real_ns=ts_ns + 1_700_000_000 * NS)

    def camera(self, w, t, bump=0.0):
        c = self.calib
        a = w.truth_at(t)["a"]
        pitch = float(c["pitch"]) + self.pitch_err + math.radians(self.p["pitch_per_mps2_deg"]) * (-a) + bump
        return rotation(pitch, float(c["yaw"]) + self.yaw_err, float(c.get("roll") or 0.0))

    def _box(self, R, o):
        c = self.calib
        u, v, d = project(R, float(c["focal_px"]), float(c["cx"]), float(c["cy"]), box_corners(o, float(c["height_m"])))
        if not np.all(d > 0.5):
            return None
        return [float(u.min()), float(v.min()), float(u.max()), float(v.max())]

    def _ramp(self, h, view):
        lo, hi = self.p["h_min"][view], self.p["h_full"][view]
        return max(0.0, min(1.0, (h - lo) / (hi - lo)))

    @staticmethod
    def _frac(b, roi):
        w, h = b[2] - b[0], b[3] - b[1]
        if w <= 0 or h <= 0:
            return 0.0, 0.0
        ow = max(0.0, min(b[2], roi[2]) - max(b[0], roi[0]))
        oh = max(0.0, min(b[3], roi[3]) - max(b[1], roi[1]))
        return ow / w, oh / h

    def visibility(self, w, t, actors=None, detail=False):
        """Noise-free detectability per actor at t: {id: (detectable in some view, ground point possible)}.
        Detectable: box inside a view's region by min_frac and at least h_min px tall; ground point: in some view
        where it is detectable its bottom is inside the region (no "b"). Occlusion by nearer actors is not
        applied here; detail=True returns occlusion()'s dicts instead (occluded / outside the field / cut)."""
        if detail:
            return self.occlusion(w, t, actors)
        R = self.camera(w, t)
        out = {}
        for o in actors if actors is not None else w.actors_at(t):
            b = None if o["ref_x"] < 1.0 else self._box(R, o)
            det = gnd = False
            if b is not None:
                h = b[3] - b[1]
                for view, roi in self.views:
                    fw, fh = self._frac(b, roi)
                    if fw >= self.p["min_frac_w"] and fh >= self.p["min_frac_h"] and h >= self.p["h_min"][view]:
                        det = True
                        gnd = gnd or b[3] < roi[3] - 1.0
            out[o["id"]] = (det, gnd)
        return out

    def occlusion(self, w, t, actors=None):
        """Noise-free truth of why an actor is or is not seen at t, per actor id: {occ, occ_by, cover, edge,
        in_fov, small, detectable, ground}.

        occ: "fov" (no view holds its box by min_frac_w / min_frac_h, or the box is behind / too near the camera),
        "object" (its box is covered by the boxes of nearer actors for at least occl_cover of its area, the rule
        by which the detector drops it; occl_cover None: 0.6), "edge" (seen, but its box is cut at the left or
        right edge of the widest view), or None. occ_by: the nearer actor covering the largest share of it (when
        cover > 0). cover: the covered share (0..1, capped). edge: "l", "r", "lr" or "". small: in a view's field
        but below h_min in every such view. detectable, ground: as visibility()."""
        R = self.camera(w, t)
        acts = actors if actors is not None else w.actors_at(t)
        thr = self.p["occl_cover"] or 0.6
        x_lo = min(r[0] for _, r in self.views)
        x_hi = max(r[2] for _, r in self.views)
        cand = []
        out = {}
        for o in acts:
            b = None if o["ref_x"] < 1.0 else self._box(R, o)
            if b is not None:
                cand.append((o["range"], o, b))
            else:
                out[o["id"]] = {"occ": "fov", "occ_by": None, "cover": 0.0, "edge": "", "in_fov": False,
                                "small": False, "detectable": False, "ground": False}
        cand.sort(key=lambda c: c[0])
        hidden = []
        for j, (_, o, b) in enumerate(cand):
            a = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
            cov, best = 0.0, (0.0, None)
            for k in range(j):
                if hidden[k]:
                    continue
                c = cand[k][2]
                ov = max(0.0, min(b[2], c[2]) - max(b[0], c[0])) * max(0.0, min(b[3], c[3]) - max(b[1], c[1]))
                cov += ov
                if ov > best[0]:
                    best = (ov, cand[k][1]["id"])
            cov = min(1.0, cov / a)
            hidden.append(cov >= thr)
            h = b[3] - b[1]
            in_fov = det = gnd = False
            for view, roi in self.views:
                fw, fh = self._frac(b, roi)
                if fw >= self.p["min_frac_w"] and fh >= self.p["min_frac_h"]:
                    in_fov = True
                    if h >= self.p["h_min"][view]:
                        det = True
                        gnd = gnd or b[3] < roi[3] - 1.0
            edge = ("l" if b[0] <= x_lo + 1.0 else "") + ("r" if b[2] >= x_hi - 1.0 else "")
            occ = "fov" if not in_fov else ("object" if hidden[-1] else ("edge" if edge else None))
            out[o["id"]] = {"occ": occ, "occ_by": best[1] if cov > 0.0 else None, "cover": round(cov, 3),
                            "edge": edge, "in_fov": in_fov, "small": in_fov and not det, "detectable": det,
                            "ground": gnd}
        return out

    def generate(self, w):
        if self._detin:
            yield from self._generate_detin(w)
            return
        p = self.p
        rng = random.Random(self.seed)
        miss = {}
        bias = {}
        bump, bump_t = 0.0, None
        fps = []                                         # active persistent false positives
        t = w.t0 + 0.031
        seq = 0
        k_view = 0
        f = float(self.calib["focal_px"])
        H = float(self.calib["height_m"])
        pm, run = p["p_miss"], p["miss_run"]
        p_h2m = pm / (run * (1.0 - pm)) if pm < 1.0 else 1.0
        p_m2h = 1.0 / run
        while t < w.t0 + w.T - 0.5:
            dt_slot = 1.0 / p["rate_hz"]
            if rng.random() < p["p_skip"]:
                t += dt_slot
                continue
            view, roi = self.views[k_view % len(self.views)]
            k_view += 1
            # body pitch from road roughness (Gauss-Markov)
            if bump_t is not None:
                rho = math.exp(-(t - bump_t) / p["pitch_bump_tau_s"])
                bump = rho * bump + math.sqrt(1 - rho * rho) * rng.gauss(0.0, math.radians(p["pitch_bump_deg"]))
            bump_t = t
            R = self.camera(w, t, bump)
            dets, logd = [], []
            cand = []
            for o in w.actors_at(t):
                aid = o["id"]
                # miss runs: a two-state Markov chain per actor, stepped every message
                st = miss.get(aid)
                if st is None:
                    st = rng.random() < pm
                else:
                    st = (rng.random() >= p_m2h) if st else (rng.random() < p_h2m)
                miss[aid] = st
                if o["ref_x"] < 1.0:
                    continue
                b = self._box(R, o)
                if b is not None:
                    cand.append((o["range"], o, b, st))
            cand.sort(key=lambda c: c[0])
            occl = self._occluded([c[2] for c in cand]) if p["occl_cover"] else [False] * len(cand)
            for (_, o, b, st), hidden in zip(cand, occl):
                aid = o["id"]
                if hidden:
                    continue
                h, wd = b[3] - b[1], b[2] - b[0]
                fw, fh = self._frac(b, roi)
                if fw < p["min_frac_w"] or fh < p["min_frac_h"]:
                    continue
                pd = self._ramp(h, view) * (p["vru_factor"] if o["kind"] == "human" else 1.0)
                if abs(math.degrees(math.atan2(o["ref_y"], o["ref_x"]))) > p["offaxis_deg"]:
                    pd *= p["offaxis_factor"]
                if st or rng.random() >= pd:
                    continue
                if aid not in bias:
                    bias[aid] = rng.gauss(0.0, p["bottom_bias_rel"])
                x0 = b[0] + rng.gauss(0.0, p["edge_px"] + p["edge_rel"] * wd)
                x1 = b[2] + rng.gauss(0.0, p["edge_px"] + p["edge_rel"] * wd)
                y0 = b[1] + rng.gauss(0.0, p["edge_px"] + p["edge_rel"] * h)
                y1 = b[3] + bias[aid] * h + rng.gauss(0.0, p["bottom_px"] + p["bottom_rel"] * h)
                if rng.random() < p["p_gross"]:        # a gross box: beside the object by a share of its width
                    sh = rng.choice((-1, 1)) * rng.uniform(0.3, 1.0) * (x1 - x0)
                    x0, x1 = x0 + sh, x1 + sh
                if x1 - x0 < 3 or y1 - y0 < 3:
                    continue
                cls = {"car": "car", "truck": "truck", "human": "person", "bike": "rider"}[o["kind"]]
                r = rng.random()
                if cls == "car" and r < p["p_car_as_truck"]:
                    cls = "truck"
                elif cls == "truck" and r < p["p_truck_as_car"]:
                    cls = "car"
                elif cls == "person" and r < p["p_person_as_rider"]:
                    cls = "rider"
                score = max(0.3, min(0.95, rng.gauss(p["score"][0] - 0.12 * (1.0 - self._ramp(h, view)), p["score"][1])))
                box, clip = self._clip([x0, y0, x1, y1], roi, rng)
                if box is None:
                    continue
                dets.append({"cls": cls, "score": round(score, 3), "box": [round(v, 1) for v in box], "clip": clip,
                             "rider": False})
                sp = p.get("scale_rate")
                if sp:
                    # the detector's scale-rate measurement: this object's image against its image in an
                    # earlier pass of the same view (patch alignment: a fraction of a pixel on the size)
                    # (the detector aligns patches of the full frame, so a box of either view continues the
                    # chain; the oldest patch within its baseline window is the reference; a fit whose stated
                    # spread is too large gives no value)
                    hist = [q for q in self._seen.get(aid, ()) if t - q[0] <= sp["max_dt"]]
                    old = [q for q in hist if t - q[0] >= sp["min_dt"]]
                    if old and not clip and h >= 8.0 and rng.random() < sp["p"]:
                        prev = old[0]
                        dts = t - prev[0]
                        sl = math.hypot(sp["px"] / h, sp["px"] / prev[1], sp["rel"])
                        if sl <= sp["max_std"]:
                            dets[-1].update(scale_rate=round(math.log(h / prev[1]) / dts + rng.gauss(0.0, sl / dts), 5),
                                            scale_rate_std=round(sl / dts, 5), scale_dt_s=round(dts, 3))
                    if not clip:
                        self._seen[aid] = hist + [(t, h)]
                logd.append((aid, cls, box, clip))
            # false positives
            rate = 1.0 / 60.0 / (p["rate_hz"] * (1.0 - p["p_skip"]))      # per message, from per minute
            if rng.random() < p["fp_transient_per_min"] * rate:
                bx = self._fp_box(rng, roi)
                if bx:
                    dets.append(dict(bx[0], score=round(rng.uniform(0.32, 0.55), 3)))
                    logd.append(("fp:transient", bx[0]["cls"], bx[0]["box"], bx[0]["clip"]))
            if rng.random() < p["fp_fixed_per_min"] * rate:
                bx = self._fp_box(rng, (0, 0, 1280, 446))
                if bx:
                    fps.append(("fixed", t + rng.uniform(*p["fp_dur_s"]), bx[1], bx[0]["cls"]))
            if rng.random() < p["fp_post_per_min"] * rate:
                fps.append(("post", t + rng.uniform(*p["fp_dur_s"]), self._post(w, t, rng), "person"))
            keep = []
            for kind, t_end, geo, cls in fps:
                if t > t_end:
                    continue
                keep.append((kind, t_end, geo, cls))
                if rng.random() >= p["fp_p_det"]:
                    continue
                if kind == "fixed":
                    b = [v + rng.gauss(0.0, 1.0) for v in geo]
                else:
                    o = self._post_actor(w, t, geo)
                    b = None if o is None or o["ref_x"] < 2.0 else self._box(R, o)
                    if b is not None:
                        b = [b[0] + rng.gauss(0, 1.5), b[1] + rng.gauss(0, 2), b[2] + rng.gauss(0, 1.5),
                             b[3] + rng.gauss(0, p["bottom_px"] + p["bottom_rel"] * (b[3] - b[1]))]
                if b is None:
                    continue
                fw, fh = self._frac(b, roi)
                if fw < 0.5 or fh < 0.5:
                    continue
                box, clip = self._clip(b, roi, rng)
                if box is None:
                    continue
                dets.append({"cls": cls, "score": round(rng.uniform(0.4, 0.7), 3), "box": [round(v, 1) for v in box],
                             "clip": clip, "rider": False})
                logd.append(("fp:" + kind, cls, box, clip))
            fps = keep
            ts = int(round(t * NS))
            msg = {"type": "lights", "seq": seq, "ts_mono_ns": ts, "ts_real_ns": ts + 1_700_000_000 * NS,
                   "lights_ms": 100.0, "frame": [1280, 720], "roi": list(roi), "view": view, "lights": [],
                   "road_users": dets}
            self.log.append((t, view, logd))
            yield t + rng.uniform(*p["latency_s"]), msg
            seq += 1
            t += dt_slot

    # ---- detection inputs (lane detin) ---------------------------------------------------------------------------
    # _generate_detin runs the base detector draw for draw on the base generator `rng` (the same calls in the same
    # order as generate(), on the full truth boxes and the old occlusion rule), records those draws, and decides what
    # is emitted on top of them: wherever the new behaviour leaves an actor's outcome alone, its box, class and score
    # are the base ones; what the base did not draw (an actor the base dropped but the new rules report) and
    # everything new comes from the second generator `r2`. So the base stream (miss chains, false positives, the
    # base boxes) never shifts with the options.
    def _roadside_items(self, w):
        cfg = self.p["roadside"]
        r = random.Random(self.seed * 31 + 9)
        items = []
        s_lo = float(w.s[0]) + CAM_TO_AXLE
        s_hi = float(w.s[-1]) + CAM_TO_AXLE + 160.0
        for side in (-1, 1):
            for kind in ("post", "tree", "hedge"):
                dens = float(cfg.get(kind + "_per_km") or 0.0)
                if dens <= 0.0:
                    continue
                s = s_lo + r.expovariate(dens / 1000.0)
                while s < s_hi:
                    d = side * r.uniform(*cfg["lat"])
                    if kind == "post":
                        dims = (0.25, 0.25, r.uniform(4.0, 7.0))           # lamp / sign posts
                    elif kind == "tree":
                        tw = r.uniform(0.3, 0.55)
                        dims = (tw, tw, 2.6)                                # the trunk (the crown is above the traffic)
                    else:
                        dims = (r.uniform(4.0, 12.0), 0.6, r.uniform(0.9, 1.4))  # railing / hedge along the kerb
                    items.append({"key": "rs:%s:%d" % (kind, len(items)), "s": s, "d": d, "dims": dims})
                    s += r.expovariate(dens / 1000.0)
        items.sort(key=lambda it: it["s"])
        return items

    def _roadside_boxes(self, w, t, R, items):
        """[(range, key, box)] of the roadside items in front of the camera at t."""
        tr = w.truth_at(t)
        s_e = tr["s"] + CAM_TO_AXLE
        g_e = float(np.interp(s_e, w.S, w.grade_S))
        alt_e = float(np.interp(s_e, w.S, w.alt_S))
        out = []
        for it in items:
            ds = it["s"] - s_e
            if ds < -2.0 or ds > 150.0:
                continue
            X, Y, hd = w._road_xy(it["s"], it["d"])
            x, y = w._to_car(tr, X, Y)
            L, W_, H_ = it["dims"]
            if x - L / 2 < 1.0:
                continue
            dz = float(np.interp(it["s"], w.S, w.alt_S) - alt_e) - math.sin(g_e) * ds
            o = {"x": x, "y": y, "heading": hd - tr["psi"], "L": L, "W": W_, "H": H_, "dz": dz}
            b = self._box(R, o)
            if b is None or b[2] < 0 or b[0] > FRAME[0] or b[3] < 0 or b[1] > FRAME[1]:
                continue
            out.append((max(0.5, math.hypot(x, y) - 0.5 * max(L, W_)), it["key"], b))
        return out

    # silhouettes for the visible-part raster: a vehicle is a body cuboid (lower body_h of its height) and a cabin
    # (the rest, inset to cabin_l x cabin_w of its length / width; cars only); a van / truck one cuboid, a person a
    # cuboid of 0.8 x its width (A: a box's corners are not the object)
    SIL = {"car": (0.6, 0.55, 0.85), "truck": (1.0, 1.0, 1.0), "bike": (1.0, 1.0, 1.0), "human": (1.0, 1.0, 0.8)}

    def _silhouette(self, R, o):
        """Convex image polygons (CCW, (k, 2) arrays) of an actor's silhouette, or None."""
        c = self.calib
        Hc = float(c["height_m"])
        bh, cl, cw = self.SIL.get(o["kind"], (1.0, 1.0, 1.0))
        parts = [(o["L"], o["W"] * (cw if bh >= 1.0 else 1.0), 0.0, o["H"] * bh)]
        if bh < 1.0:
            parts.append((o["L"] * cl, o["W"] * cw, o["H"] * bh, o["H"]))
        ch, sh = math.cos(o["heading"]), math.sin(o["heading"])
        zg = Hc - o["dz"]
        polys = []
        for L, W_, z0, z1 in parts:
            pts = []
            for lx in (-L / 2, L / 2):
                for ly in (-W_ / 2, W_ / 2):
                    x = o["x"] + lx * ch - ly * sh
                    y = o["y"] + lx * sh + ly * ch
                    pts.append((x, -y, zg - z0))
                    pts.append((x, -y, zg - z1))
            u, v, d = project(R, float(c["focal_px"]), float(c["cx"]), float(c["cy"]), np.array(pts))
            if not np.all(d > 0.5):
                return None
            e = _poly_edges(_hull(np.stack([u, v], axis=1)))
            if e is not None:
                polys.append(e)
        return polys

    def _vis(self, cand, occs, R, roi):
        """Per candidate (range, o, b, st): the visible part of its silhouette against nearer occluders [(range, key,
        box, polygons | None)] on a grid over b, inside the frame: {box (bounding box of the visible cells, frame-
        clipped; None: nothing visible), cover (covered share of the in-frame silhouette), cover_box (covered share
        of the silhouette cells within `box`), hidden (truth edges hidden by occluders), out (truth edges beyond the
        frame), by (key covering most), keys {key: covered cells}}."""
        G = int(self.p["vis_grid"])
        FW, FH = FRAME
        idx = (np.arange(G) + 0.5) / G
        sil = {}
        res = []
        hmin = 0.5 * min(self.p["h_min"])
        for rg, o, b, _ in cand:
            bw, bh = b[2] - b[0], b[3] - b[1]
            if bh < hmin or b[2] < roi[0] or b[0] > roi[2] or b[3] < roi[1] or b[1] > roi[3]:
                res.append({"box": None, "cover": 0.0, "cover_box": 0.0, "hidden": "", "out": "", "by": None, "keys": {}})
                continue                 # can be in neither list in this view: not rasterised
            xc, yc = b[0] + idx * bw, b[1] + idx * bh
            X, Y = np.meshgrid(xc, yc)
            inx, iny = (xc >= 0) & (xc <= FW), (yc >= 0) & (yc <= FH)
            if o["id"] not in sil:
                sil[o["id"]] = self._silhouette(R, o)
            own = _in_polys(X, Y, sil[o["id"]]) if sil[o["id"]] else np.ones((G, G), bool)
            inf = np.outer(iny, inx) & own
            nin0 = int(np.count_nonzero(inf))
            cov = np.zeros((G, G), bool)
            keys = {}
            for rk, key, ob, polys in occs:
                if rk >= rg or key == o["id"] or ob[2] <= b[0] or ob[0] >= b[2] or ob[3] <= b[1] or ob[1] >= b[3]:
                    continue
                if polys == "actor":
                    if key not in sil:
                        sil[key] = self._silhouette(R, next(c[1] for c in cand if c[1]["id"] == key))
                    polys = sil[key]
                mx, my = (xc >= ob[0]) & (xc <= ob[2]), (yc >= ob[1]) & (yc <= ob[3])
                m = np.outer(my, mx) & inf
                if polys and m.any():
                    ri, ci = np.where(my)[0], np.where(mx)[0]
                    sl_ = (slice(ri[0], ri[-1] + 1), slice(ci[0], ci[-1] + 1))
                    sub = np.zeros((G, G), bool)
                    sub[sl_] = _in_polys(X[sl_], Y[sl_], polys)
                    m &= sub
                n = int(np.count_nonzero(m))
                if n:
                    keys[key] = n
                    cov |= m
                    if n == nin0 or np.count_nonzero(cov) == nin0:
                        break                    # all of it covered: nearer things change nothing more
            out = ("l" if b[0] < 0 else "") + ("t" if b[1] < 0 else "") + ("r" if b[2] > FW else "") + ("b" if b[3] > FH else "")
            nin = int(np.count_nonzero(inf))
            vis = inf & ~cov
            if nin == 0 or not vis.any():
                res.append({"box": None, "cover": 1.0 if nin else 0.0, "cover_box": 1.0, "hidden": "", "out": out,
                            "by": max(keys, key=keys.get) if keys else None, "keys": keys})
                continue
            rows, cols = np.where(vis.any(axis=1))[0], np.where(vis.any(axis=0))[0]
            r0, r1, c0, c1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
            # a side whose visible extent reaches the silhouette's own (in-frame) extent keeps the exact truth edge
            rows_s, cols_s = np.where(inf.any(axis=1))[0], np.where(inf.any(axis=0))[0]
            vb = [max(0.0, b[0] if c0 == cols_s[0] else b[0] + c0 * bw / G),
                  max(0.0, b[1] if r0 == rows_s[0] else b[1] + r0 * bh / G),
                  min(float(FW), b[2] if c1 == cols_s[-1] else b[0] + (c1 + 1) * bw / G),
                  min(float(FH), b[3] if r1 == rows_s[-1] else b[1] + (r1 + 1) * bh / G)]
            cover = float(np.count_nonzero(cov)) / nin
            cover_box = float(np.count_nonzero(cov[r0:r1 + 1, c0:c1 + 1])) / max(1, int(np.count_nonzero(inf[r0:r1 + 1, c0:c1 + 1])))
            hid = ""
            if keys:
                # an edge of the (in-frame) silhouette is hidden when at least half of its outermost cells are covered
                rows_in, cols_in = np.where(inf.any(axis=1))[0], np.where(inf.any(axis=0))[0]
                for letter, line, mask, mine in (
                        ("l", cov[:, cols_in[0]], inf[:, cols_in[0]], inx[0] or cols_in[0] > 0),
                        ("t", cov[rows_in[0], :], inf[rows_in[0], :], iny[0] or rows_in[0] > 0),
                        ("r", cov[:, cols_in[-1]], inf[:, cols_in[-1]], inx[-1] or cols_in[-1] < G - 1),
                        ("b", cov[rows_in[-1], :], inf[rows_in[-1], :], iny[-1] or rows_in[-1] < G - 1)):
                    if mine and np.count_nonzero(line) >= 0.5 * max(1, np.count_nonzero(mask)):
                        hid += letter
            res.append({"box": vb, "cover": cover, "cover_box": cover_box, "hidden": hid, "out": out,
                        "by": max(keys, key=keys.get) if keys else None, "keys": keys})
        return res

    @staticmethod
    def _clip_v(b, roi, ub):
        """_clip with the bottom overshoot from ub() (called only when the bottom is cut, as _clip draws it)."""
        x0, y0, x1, y1 = b
        clip = ""
        if x0 <= roi[0] + 1.0:
            x0, clip = float(roi[0]), clip + "l"
        if y0 <= roi[1] + 1.0:
            y0, clip = float(roi[1]), clip + "t"
        if x1 >= roi[2] - 1.0:
            x1, clip = float(roi[2]), clip + "r"
        if y1 >= roi[3] - 1.0:
            y1, clip = roi[3] + ub(), clip + "b"
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None, ""
        return [x0, y0, x1, y1], clip

    @staticmethod
    def _lit(o):
        """A vehicle with its lamps on: moving or standing in traffic; parked cars (roles naming a parked car or
        a parked row) are unlit."""
        role = o.get("role") or ""
        return not ("park" in role or role in ("occ_row", "pair_d"))

    def _cls_wrong(self, cls, r2):
        alt = {"car": ("truck",), "truck": ("car", "bus"), "bus": ("truck",), "person": ("rider",),
               "rider": ("person", "bike"), "bike": ("rider", "motor"), "motor": ("bike",), "train": ("bus",)}
        return r2.choice(alt.get(cls, ("car",)))

    def _generate_detin(self, w):
        p = self.p
        rng = random.Random(self.seed)
        r2 = random.Random(self.seed * 31 + 7)
        miss, bias, bias2, biasL = {}, {}, {}, {}
        bump, bump_t = 0.0, None
        fps = []
        t = w.t0 + 0.031
        seq = 0
        k_view = 0
        pm, run = p["p_miss"], p["miss_run"]
        p_h2m = pm / (run * (1.0 - pm)) if pm < 1.0 else 1.0
        p_m2h = 1.0 / run
        f = float(self.calib["focal_px"])
        Hc = float(self.calib["height_m"])
        sp = p.get("scale_rate")
        low_on = p.get("low") is not None
        lamps_on = bool(p.get("lamps")) and self.night
        vis_on = bool(p.get("visible_boxes"))
        thr_cls = dict(p["low_thr"])
        items = self._roadside_items(w) if p.get("roadside") else []
        emb = EmbGen(p["emb"], self.night, r2, p["emb_level"]) if p.get("emb") else None
        lamp_S = {}
        chains = {}          # actor -> [(t, ln size + error, sigma)]   (scale_corr: the pass's own log-size error)
        lchains = {}         # actor -> [(t, ln sep, sigma, colour)]    (lamp-pair separations)
        clutter = []         # persistent low clutter: [t_end, box, cls, key]
        msgs_s = p["rate_hz"] * (1.0 - p["p_skip"])
        rate = 1.0 / 60.0 / msgs_s
        nfp = 0
        while t < w.t0 + w.T - 0.5:
            dt_slot = 1.0 / p["rate_hz"]
            if rng.random() < p["p_skip"]:
                t += dt_slot
                continue
            view, roi = self.views[k_view % len(self.views)]
            k_view += 1
            if bump_t is not None:
                rho = math.exp(-(t - bump_t) / p["pitch_bump_tau_s"])
                bump = rho * bump + math.sqrt(1 - rho * rho) * rng.gauss(0.0, math.radians(p["pitch_bump_deg"]))
            bump_t = t
            R = self.camera(w, t, bump)
            dets, logd = [], []
            ents = []                 # per first-list box: {"aid", "o", "V", "key", "h_true"}
            missed = []               # low-list candidates: (o, visible box, why, V, truth box)
            elig = []                 # actors seen in this view's region (visible part inside it by min_frac)
            cand = []
            for o in w.actors_at(t):
                aid = o["id"]
                st = miss.get(aid)
                if st is None:
                    st = rng.random() < pm
                else:
                    st = (rng.random() >= p_m2h) if st else (rng.random() < p_h2m)
                miss[aid] = st
                if o["ref_x"] < 1.0:
                    continue
                b = self._box(R, o)
                if b is not None:
                    cand.append((o["range"], o, b, st))
            cand.sort(key=lambda c: c[0])
            occl = self._occluded([c[2] for c in cand]) if p["occl_cover"] else [False] * len(cand)
            occs = [(c[0], c[1]["id"], c[2], "actor") for c in cand]
            if items:
                occs += [(rk, key, bx, None) for rk, key, bx in self._roadside_boxes(w, t, R, items)]
            V_all = self._vis(cand, occs, R, roi)
            for (_, o, b, st), hidden, V in zip(cand, occl, V_all):
                aid = o["id"]
                # -- the base path, draw for draw as generate() --
                rec = {}
                h, wd = b[3] - b[1], b[2] - b[0]
                fw, fh = self._frac(b, roi)
                if not hidden and fw >= p["min_frac_w"] and fh >= p["min_frac_h"]:
                    pd = self._ramp(h, view) * (p["vru_factor"] if o["kind"] == "human" else 1.0)
                    if abs(math.degrees(math.atan2(o["ref_y"], o["ref_x"]))) > p["offaxis_deg"]:
                        pd *= p["offaxis_factor"]
                    if not st:
                        rec["u"] = rng.random()
                        if rec["u"] < pd:
                            if aid not in bias:
                                bias[aid] = rng.gauss(0.0, p["bottom_bias_rel"])
                            sg = (p["edge_px"] + p["edge_rel"] * wd, p["edge_px"] + p["edge_rel"] * wd,
                                  p["edge_px"] + p["edge_rel"] * h, p["bottom_px"] + p["bottom_rel"] * h)
                            e = (rng.gauss(0.0, sg[0]), rng.gauss(0.0, sg[1]), rng.gauss(0.0, sg[2]), rng.gauss(0.0, sg[3]))
                            rec["e"], rec["sg"] = e, sg
                            x0, x1, y0, y1 = b[0] + e[0], b[2] + e[1], b[1] + e[2], b[3] + bias[aid] * h + e[3]
                            rec["gross"] = None
                            if rng.random() < p["p_gross"]:
                                rec["gross"] = (rng.choice((-1, 1)), rng.uniform(0.3, 1.0))
                                sh = rec["gross"][0] * rec["gross"][1] * (x1 - x0)
                                x0, x1 = x0 + sh, x1 + sh
                            if not (x1 - x0 < 3 or y1 - y0 < 3):
                                rec["r"] = rng.random()
                                mu = p["score"][0] - 0.12 * (1.0 - self._ramp(h, view))
                                rec["sv"], rec["mu"] = rng.gauss(mu, p["score"][1]), mu

                                def ub0(rec=rec):
                                    rec["ub"] = rng.uniform(-0.6, 2.2)
                                    return rec["ub"]
                                box0, clip0 = self._clip_v([x0, y0, x1, y1], roi, ub0)
                                if box0 is not None and sp:
                                    hist = [q for q in self._seen.get(aid, ()) if t - q[0] <= sp["max_dt"]]
                                    old = [q for q in hist if t - q[0] >= sp["min_dt"]]
                                    if old and not clip0 and h >= 8.0 and rng.random() < sp["p"]:
                                        prev = old[0]
                                        dts = t - prev[0]
                                        sl = math.hypot(sp["px"] / h, sp["px"] / prev[1], sp["rel"])
                                        if sl <= sp["max_std"]:
                                            rec["scale"] = dict(scale_rate=round(math.log(h / prev[1]) / dts + rng.gauss(0.0, sl / dts), 5),
                                                                scale_rate_std=round(sl / dts, 5), scale_dt_s=round(dts, 3))
                                            rec["scale_true"] = math.log(h / prev[1]) / dts
                                    if not clip0:
                                        self._seen[aid] = hist + [(t, h)]
                # -- the new decision on top --
                vb = V["box"] if vis_on else b
                if vb is None:
                    continue
                hv, wv = vb[3] - vb[1], vb[2] - vb[0]
                fw, fh = self._frac(vb, roi)
                if fw < p["min_frac_w"] or fh < p["min_frac_h"]:
                    continue
                elig.append(aid)
                cover = V["cover"]
                if vis_on and p["occl_cover"]:
                    if cover >= p["occl_cover"]:
                        if cover < p["low_occl_max"]:
                            missed.append((o, vb, "occl", V, b))
                        continue
                elif hidden:
                    continue
                pd = self._ramp(hv, view) * (p["vru_factor"] if o["kind"] == "human" else 1.0)
                if abs(math.degrees(math.atan2(o["ref_y"], o["ref_x"]))) > p["offaxis_deg"]:
                    pd *= p["offaxis_factor"]
                if vis_on and p["occl_cover"] and cover > p["occl_pd_from"]:
                    u_ = min(1.0, (cover - p["occl_pd_from"]) / max(1e-6, p["occl_cover"] - p["occl_pd_from"]))
                    pd *= 1.0 - u_ * (1.0 - p["occl_pd_at"])
                if st:
                    missed.append((o, vb, "run", V, b))
                    continue
                u = rec["u"] if "u" in rec else r2.random()
                if u >= pd:
                    missed.append((o, vb, "small" if self._ramp(hv, view) <= 0.0 else "draw", V, b))
                    continue
                # the persistent bottom offset: the base's draw whenever the base has drawn it (an actor that the new
                # rules report before the base ever did uses a second-stream draw until then)
                if aid in bias:
                    bv = bias[aid]
                elif aid in bias2:
                    bv = bias2[aid]
                else:
                    bv = bias2[aid] = r2.gauss(0.0, p["bottom_bias_rel"])
                sg = (p["edge_px"] + p["edge_rel"] * wv, p["edge_px"] + p["edge_rel"] * wv,
                      p["edge_px"] + p["edge_rel"] * hv, p["bottom_px"] + p["bottom_rel"] * hv)
                if "e" in rec:
                    e = tuple(ev if s1 == s0 else ev * (s1 / s0) for ev, s0, s1 in zip(rec["e"], rec["sg"], sg))
                else:
                    e = tuple(r2.gauss(0.0, s) for s in sg)
                x0, x1, y0, y1 = vb[0] + e[0], vb[2] + e[1], vb[1] + e[2], vb[3] + bv * hv + e[3]
                gross = rec["gross"] if "gross" in rec else (
                    (r2.choice((-1, 1)), r2.uniform(0.3, 1.0)) if r2.random() < p["p_gross"] else None)
                if gross:
                    sh = gross[0] * gross[1] * (x1 - x0)
                    x0, x1 = x0 + sh, x1 + sh
                if x1 - x0 < 3 or y1 - y0 < 3:
                    continue
                cls = {"car": "car", "truck": "truck", "human": "person", "bike": "rider"}[o["kind"]]
                r = rec["r"] if "r" in rec else r2.random()
                if cls == "car" and r < p["p_car_as_truck"]:
                    cls = "truck"
                elif cls == "truck" and r < p["p_truck_as_car"]:
                    cls = "car"
                elif cls == "person" and r < p["p_person_as_rider"]:
                    cls = "rider"
                mu = p["score"][0] - 0.12 * (1.0 - self._ramp(hv, view))
                if "sv" in rec:
                    sv = rec["sv"] if mu == rec["mu"] else rec["sv"] - rec["mu"] + mu
                else:
                    sv = r2.gauss(mu, p["score"][1])
                score = max(0.3, min(0.95, sv))

                def ub(rec=rec):
                    return rec["ub"] if "ub" in rec else r2.uniform(-0.6, 2.2)
                box, clip = self._clip_v([x0, y0, x1, y1], roi, ub)
                if box is None:
                    continue
                d = {"cls": cls, "score": round(score, 3), "box": [round(v, 1) for v in box], "clip": clip, "rider": False}
                en = {"aid": aid, "o": o, "V": V, "b": b, "key": aid}
                if "scale" in rec and not p.get("scale_corr") and not clip and (not vis_on or V["cover_box"] <= p["scale_occ_max"]):
                    d.update(rec["scale"])
                    en["scale_true"] = rec["scale_true"]
                dets.append(d)
                logd.append((aid, cls, box, clip))
                ents.append(en)
            # false positives: the base code, draw for draw
            if rng.random() < p["fp_transient_per_min"] * rate:
                bx = self._fp_box(rng, roi)
                if bx:
                    dets.append(dict(bx[0], score=round(rng.uniform(0.32, 0.55), 3)))
                    logd.append(("fp:transient", bx[0]["cls"], bx[0]["box"], bx[0]["clip"]))
                    nfp += 1
                    ents.append({"aid": "fp:transient", "o": None, "V": None, "key": ("fp", nfp)})
            if rng.random() < p["fp_fixed_per_min"] * rate:
                bx = self._fp_box(rng, (0, 0, 1280, 446))
                if bx:
                    fps.append(("fixed", t + rng.uniform(*p["fp_dur_s"]), bx[1], bx[0]["cls"]))
            if rng.random() < p["fp_post_per_min"] * rate:
                fps.append(("post", t + rng.uniform(*p["fp_dur_s"]), self._post(w, t, rng), "person"))
            keep = []
            for kind, t_end, geo, cls in fps:
                if t > t_end:
                    continue
                keep.append((kind, t_end, geo, cls))
                if rng.random() >= p["fp_p_det"]:
                    continue
                if kind == "fixed":
                    bb = [v + rng.gauss(0.0, 1.0) for v in geo]
                else:
                    o = self._post_actor(w, t, geo)
                    bb = None if o is None or o["ref_x"] < 2.0 else self._box(R, o)
                    if bb is not None:
                        bb = [bb[0] + rng.gauss(0, 1.5), bb[1] + rng.gauss(0, 2), bb[2] + rng.gauss(0, 1.5),
                              bb[3] + rng.gauss(0, p["bottom_px"] + p["bottom_rel"] * (bb[3] - bb[1]))]
                if bb is None:
                    continue
                fw, fh = self._frac(bb, roi)
                if fw < 0.5 or fh < 0.5:
                    continue
                box, clip = self._clip(bb, roi, rng)
                if box is None:
                    continue
                dets.append({"cls": cls, "score": round(rng.uniform(0.4, 0.7), 3), "box": [round(v, 1) for v in box],
                             "clip": clip, "rider": False})
                logd.append(("fp:" + kind, cls, box, clip))
                ents.append({"aid": "fp:" + kind, "o": None, "V": None, "key": ("fp", kind, t_end)})
            fps = keep
            # ---- the low list ----
            low, lents = [], []
            if low_on:
                floor = float(p["low"])
                cands = []
                for o, vb, why, V, b in missed:
                    hv, wv = vb[3] - vb[1], vb[2] - vb[0]
                    if why == "run":
                        pl = p["p_low_run"]
                    elif why == "draw":
                        pl = p["p_low_draw"]
                    elif why == "occl":
                        pl = p["p_low_occl"]
                    else:
                        hm = p["h_min"][view]
                        a0, a1 = p["low_h_from"]
                        pl = p["p_low_small"] * max(0.0, min(1.0, (hv - a0 * hm) / max(1e-6, (a1 - a0) * hm)))
                    if pl <= 0.0 or r2.random() >= pl:
                        continue
                    aid = o["id"]
                    bv = bias[aid] if aid in bias else bias2.get(aid)
                    if bv is None:
                        bv = biasL[aid] if aid in biasL else biasL.setdefault(aid, r2.gauss(0.0, p["bottom_bias_rel"]))
                    kx, kr = p["low_edge_x"], p["low_edge_rel"]
                    sx, sy, sb = kx * p["edge_px"] + kr * wv, kx * p["edge_px"] + kr * hv, kx * p["bottom_px"] + kr * hv
                    bb = [vb[0] + r2.gauss(0, sx), vb[1] + r2.gauss(0, sy), vb[2] + r2.gauss(0, sx),
                          vb[3] + bv * hv + r2.gauss(0, sb)]
                    if bb[2] - bb[0] < 3 or bb[3] - bb[1] < 3:
                        continue
                    cls = {"car": "car", "truck": "truck", "human": "person", "bike": "rider"}[o["kind"]]
                    if r2.random() < p["low_cls_wrong"]:
                        cls = self._cls_wrong(cls, r2)
                    cands.append((bb, cls, {"aid": aid, "o": o, "V": V, "b": b, "key": aid, "why": why}))
                # clutter on nothing: transient, and recurring at one image place
                lam = float(p["low_clutter"])
                fx = float(p["low_clutter_fixed"])
                per_fixed = max(1.0, p["low_clutter_p_det"] * 0.5 * sum(p["low_clutter_dur_s"]) * msgs_s)
                for _ in range(4):
                    if r2.random() < lam * (1.0 - fx) / 4.0:
                        bx = self._low_fp_box(r2, roi)
                        if bx:
                            nfp += 1
                            cands.append((bx[0], bx[1], {"aid": "fp:clutter", "o": None, "V": None, "key": ("lowfp", nfp),
                                                         "why": "clutter"}))
                    if r2.random() < lam * fx / per_fixed / 4.0:
                        bx = self._low_fp_box(r2, (0, 0, 1280, 446))
                        if bx:
                            nfp += 1
                            clutter.append([t + r2.uniform(*p["low_clutter_dur_s"]), bx[0], bx[1], ("lowfp", nfp), True])
                kept = []
                for c in clutter:
                    if t > c[0]:
                        continue
                    kept.append(c)
                    first = c[4]
                    c[4] = False
                    if first or r2.random() < p["low_clutter_p_det"]:
                        bb = [v + r2.gauss(0.0, 1.5) for v in c[1]]
                        if self._frac(bb, roi)[0] >= 0.5 and self._frac(bb, roi)[1] >= 0.5:
                            cands.append((bb, c[2], {"aid": "fp:clutter_fixed", "o": None, "V": None, "key": c[3],
                                                     "why": "clutter_fixed"}))
                clutter = kept
                # beside a first-list actor box: part of it, or beside it
                for d, en in zip(dets, ents):
                    if en["o"] is None or r2.random() >= p["p_low_beside"]:
                        continue
                    x0, y0, x1, y1 = d["box"]
                    bw_, bh_ = x1 - x0, y1 - y0
                    if r2.random() < 0.5:
                        sh = r2.choice((-1, 1)) * r2.uniform(0.4, 0.8) * bw_
                        bb = [x0 + sh, y0 + r2.gauss(0, 0.05 * bh_), x1 + sh, y1 + r2.gauss(0, 0.05 * bh_)]
                    else:
                        k = r2.uniform(0.3, 0.45)
                        bb = [x0, y0, x1, y0 + k * bh_] if r2.random() < 0.5 else (
                            [x0, y0, x0 + k * bw_, y1] if r2.random() < 0.5 else [x1 - k * bw_, y0, x1, y1])
                    cands.append((bb, d["cls"], {"aid": "fp:beside:%s" % en["aid"], "o": en["o"], "V": None,
                                                 "key": en["key"], "why": "beside", "beside": True}))
                # scores, the region, NMS against the first list and better low boxes, the cap
                scored = []
                for bb, cls, en in cands:
                    box, clip = self._clip_v(bb, roi, lambda: r2.uniform(-0.6, 2.2))
                    if box is None:
                        continue
                    sc = r2.uniform(floor + 0.0005, thr_cls.get(cls, 0.40))     # scores above the floor
                    scored.append((sc, box, clip, cls, en))
                scored.sort(key=lambda s: -s[0])
                first_boxes = [d["box"] for d in dets]
                for sc, box, clip, cls, en in scored:
                    if len(low) >= p["low_max"]:
                        break
                    rb = [round(v, 1) for v in box]
                    if any(_iou(rb, fb) > p["low_nms"] for fb in first_boxes) or any(_iou(rb, l["box"]) > p["low_nms"] for l in low):
                        continue
                    low.append({"cls": cls, "score": round(sc, 3), "box": rb, "clip": clip, "rider": False})
                    lents.append(dict(en, box=box, clip=clip, cls=cls))
            # ---- flags ----
            if p.get("flags"):
                ru_flags(dets, low if low_on else None)
            # ---- lamps (night), scale, appearance ----
            for lst, el in ((dets, ents), (low, lents)):
                for d, en in zip(lst, el):
                    o = en["o"]
                    if o is None or en.get("beside"):
                        continue
                    if lamps_on and d["cls"] in ("car", "truck", "bus", "train") and o["kind"] in VEHICLE_KINDS:
                        lp = self._lamps(o, d, R, r2, lamp_S)
                        if lp is not None:
                            d["lamps"], en["sep_true"] = lp
                    if p.get("scale_corr") and sp:
                        self._scale_x(o, d, en, t, r2, sp, chains, lchains, lst is low)
            if emb is not None:
                n_emb = 0
                order = sorted(range(len(dets)), key=lambda i: -(dets[i]["box"][2] - dets[i]["box"][0]) * (dets[i]["box"][3] - dets[i]["box"][1]))
                todo = [(dets[i], ents[i]) for i in order] + list(zip(low, lents))
                for d, en in todo:
                    bx = d["box"]
                    if bx[2] - bx[0] < 8.0 or bx[3] - bx[1] < 8.0:
                        continue
                    if p["emb"] == "reid32+yuv3x3" and n_emb >= p["emb_max"]:
                        continue
                    n_emb += 1
                    V = en.get("V")
                    mix, other = 0.0, None
                    if en.get("beside"):
                        mix, other = 0.5, ("bg", r2.random())
                    elif V is not None:
                        mix = V["cover_box"] if vis_on else V["cover"]
                        other = V["by"]
                        if mix <= 0.0:
                            mix = 0.0
                    d["emb"] = emb_hex(emb.observe(en["key"], t, mix, other))
            ts = int(round(t * NS))
            msg = {"type": "lights", "seq": seq, "ts_mono_ns": ts, "ts_real_ns": ts + 1_700_000_000 * NS,
                   "lights_ms": 100.0, "frame": [1280, 720], "roi": list(roi), "view": view, "lights": [],
                   "road_users": dets}
            if low_on:
                msg["road_users_low"] = low
            if emb is not None:
                msg["emb_kind"] = p["emb"]
            self.log.append((t, view, logd))
            if low_on:
                self.log_low.append((t, view, [(en["aid"], en["cls"], en["box"], en["clip"]) for en in lents]))
            self.log_x.append({"t": t, "view": view, "seen": elig, "first": [self._truth(en) for en in ents],
                               "low": [dict(self._truth(en), why=en["why"]) for en in lents]})
            yield t + rng.uniform(*p["latency_s"]), msg
            seq += 1
            t += dt_slot

    @staticmethod
    def _truth(en):
        V = en.get("V")
        out = {"id": en["aid"], "cover": None}
        if V is not None:
            out = {"id": en["aid"], "cover": round(V["cover"], 3), "cover_box": round(V["cover_box"], 3),
                   "hidden": V["hidden"], "out": V["out"], "by": V["by"]}
        for k in ("scale_true", "sep_true"):
            if k in en:
                out[k] = en[k]
        if en.get("b") is not None:
            out["h"] = en["b"][3] - en["b"][1]
        return out

    @staticmethod
    def _low_fp_box(r2, roi):
        cls = "car" if r2.random() < FP_CLASSES[0][1] else "person"
        h = r2.uniform(12.0, 50.0)
        wd = h * (r2.uniform(1.0, 2.2) if cls == "car" else r2.uniform(0.3, 0.5))
        if roi[2] - roi[0] < wd + 2:
            return None
        cx = r2.uniform(roi[0] + wd / 2, roi[2] - wd / 2)
        y1 = r2.uniform(roi[1] + 0.5 * (roi[3] - roi[1]), roi[3] + 0.2 * h)
        return [cx - wd / 2, y1 - h, cx + wd / 2, y1], cls

    def _lamps(self, o, d, R, r2, lamp_S):
        """The lamp pair of a vehicle box at night, as the detector reports it, or None."""
        p = self.p
        h = o["heading"]
        ch, sh = math.cos(h), math.sin(h)
        cos_a = (ch * o["x"] + sh * o["y"]) / max(1e-6, math.hypot(o["x"], o["y"]))   # > 0: driving away, rear seen
        if abs(cos_a) < math.cos(math.radians(p["lamp_cone_deg"])):
            return None
        lit = self._lit(o)
        if r2.random() >= (p["lamp_p"] if lit else p["lamp_p_unlit"]):
            return None
        aid = o["id"]
        if aid not in lamp_S:
            lamp_S[aid] = p["lamp_S"][0] * o["W"] * math.exp(r2.gauss(0.0, p["lamp_S"][1]))
        S = lamp_S[aid]
        sg = 1.0 if cos_a > 0 else -1.0
        ex, ey = o["x"] - sg * o["L"] / 2 * ch, o["y"] - sg * o["L"] / 2 * sh       # the end facing the camera
        zl = float(self.calib["height_m"]) - o["dz"] - p["lamp_h_rel"] * o["H"]
        P = np.array([[ex - S / 2 * sh, -(ey + S / 2 * ch), zl], [ex + S / 2 * sh, -(ey - S / 2 * ch), zl]])
        c = self.calib
        u, v, dd = project(R, float(c["focal_px"]), float(c["cx"]), float(c["cy"]), P)
        if not np.all(dd > 0.5):
            return None
        if u[0] > u[1]:
            u, v = u[::-1], v[::-1]
        cx_, cy_ = 0.5 * (u[0] + u[1]), 0.5 * (v[0] + v[1])
        sep_t = sep_true = float(math.hypot(u[1] - u[0], v[1] - v[0]))
        if r2.random() < p["lamp_wrong"]:
            sep_t *= r2.uniform(*p["lamp_wrong_x"])
        noise = p["lamp_small_noise"] if sep_t < p["lamp_small_px"] else math.hypot(p["lamp_noise"][0], p["lamp_noise"][1] * sep_t)
        k = noise / math.sqrt(2.0)
        xl, xr = cx_ - sep_t / 2 + r2.gauss(0, k), cx_ + sep_t / 2 + r2.gauss(0, k)
        yl, yr = cy_ + r2.gauss(0, 0.3 * k), cy_ + r2.gauss(0, 0.3 * k)
        sep = math.hypot(xr - xl, yr - yl)
        bx = d["box"]
        bw = bx[2] - bx[0]
        if not (bx[0] <= xl <= bx[2] and bx[0] <= xr <= bx[2] and bx[1] <= yl <= bx[3] and bx[1] <= yr <= bx[3]):
            return None
        if sep < 0.3 * bw or sep > 0.98 * bw or sep < 4.0:
            return None
        cut = _bits(d.get("clip") or "") | _bits(d.get("trunc") or "") | trunc_of(bx)
        if not (cut & 5) and abs(0.5 * (xl + xr) - 0.5 * (bx[0] + bx[2])) > 0.25 * bw:
            return None
        if lit:
            col = "red" if cos_a > 0 else "white"
            r = r2.random()
            if r < p["lamp_unknown"]:
                col = "unknown"
            elif r < p["lamp_unknown"] + p["lamp_colour_wrong"]:
                col = "white" if col == "red" else "red"
        else:
            col = "red" if r2.random() < 0.5 else "unknown"
        sep_std = math.hypot(math.hypot(0.25, 0.25), p["lamp_noise"][1] * sep)
        return ({"c": col, "l": [round(xl, 1), round(yl, 1)], "r": [round(xr, 1), round(yr, 1)], "sep": round(sep, 2),
                 "sep_std": round(sep_std, 2)}, sep_true)

    def _scale_x(self, o, d, en, t, r2, sp, chains, lchains, is_low):
        """scale_corr: per actor and pass a noisy log size (a chain of increments, each with the pass's own error);
        the rate is this pass against the oldest sample of the window, as the patch alignment does; at night with
        lamps also the WLS line through ln(separation) of the window (signd scale.cc), combined as signd does."""
        p = self.p
        aid = o["id"]
        b = en.get("b")
        if b is None:                                   # a low box: its truth box from the visible-part record
            b = en["V"]["box"] if en.get("V") and en["V"].get("box") else d["box"]
        h = b[3] - b[1]
        V = en.get("V")
        clip = d.get("clip") or ""
        patch = None
        ok_patch = (not clip and h >= 8.0 and (V is None or not p.get("visible_boxes") or V["cover_box"] <= p["scale_occ_max"]))
        if ok_patch:
            # the patch chain: this pass's log size = the previous pass's + the true change + this pass's own alignment
            # error (variance in proportion to the step: a value over scale_inc_T s has the base model's std
            # hypot(px / h, px / h_ref, rel)); the value = this pass against the oldest sample of the window, its
            # stated std the accumulated one. Successive values share the increments of their overlap.
            ch = chains.get(aid) or []
            lnh = math.log(h)
            if ch and t - ch[-1][0] <= sp["max_dt"]:
                tl, yl, vl, ytl, hl = ch[-1]
                var = math.hypot(sp["px"] / h, sp["px"] / hl, sp["rel"]) ** 2 * (t - tl) / p["scale_inc_T"]
                ch = [q for q in ch if t - q[0] <= sp["max_dt"]] or [ch[-1]]
                ch.append((t, yl + (lnh - ytl) + r2.gauss(0.0, math.sqrt(var)), vl + var, lnh, h))
            else:
                ch = [(t, lnh, 0.0, lnh, h)]
            chains[aid] = ch
            old = [q for q in ch[:-1] if sp["min_dt"] <= t - q[0] <= sp["max_dt"]]
            if old and r2.random() < sp["p"]:
                ref, now = old[0], ch[-1]
                dts = t - ref[0]
                sl = math.sqrt(max(1e-12, now[2] - ref[2]))
                if sl <= sp["max_std"]:
                    patch = ((now[1] - ref[1]) / dts, sl / dts, dts, (now[3] - ref[3]) / dts)
        lamp = None
        lp = d.get("lamps")
        if lp is not None and lp["sep"] >= p["lamp_scale_min_px"]:
            sig = lp["sep_std"] / lp["sep"]
            y = math.log(lp["sep"])
            if r2.random() < p["lamp_scale_tail"][0]:
                y += r2.choice((-1, 1)) * p["lamp_scale_tail"][1] * sig
            lc = [q for q in lchains.get(aid, ()) if t - q[0] <= sp["max_dt"] and q[3] == lp["c"]]
            lc.append((t, y, sig, lp["c"], math.log(en["sep_true"])))
            lchains[aid] = lc
            if (len(lc) >= p["lamp_scale_min_n"] and t - lc[0][0] >= sp["min_dt"]
                    and all(abs(lc[i + 1][1] - lc[i][1]) <= p["lamp_scale_step"] for i in range(len(lc) - 1))):
                sl, sd = _wls_slope([(q[0], q[1], q[2]) for q in lc])
                dts = t - lc[0][0]
                if sl is not None and sd * dts <= sp["max_std"]:
                    lamp = (sl, sd, dts, _wls_slope([(q[0], q[4], q[2]) for q in lc])[0])
        if patch and lamp:
            wp, wl = 1.0 / patch[1] ** 2, 1.0 / lamp[1] ** 2
            v = (wp * patch[0] + wl * lamp[0]) / (wp + wl)
            sd = max(1.0 / math.sqrt(wp + wl), 0.7 * min(patch[1], lamp[1]))
            if abs(patch[0] - lamp[0]) > 3.0 * math.hypot(patch[1], lamp[1]):
                sd += 0.5 * abs(patch[0] - lamp[0])
            d.update(scale_rate=round(v, 5), scale_rate_std=round(sd, 5), scale_dt_s=round(max(patch[2], lamp[2]), 3),
                     scale_src="both")
            en["scale_true"] = (wp * patch[3] + wl * lamp[3]) / (wp + wl)
        elif patch:
            d.update(scale_rate=round(patch[0], 5), scale_rate_std=round(patch[1], 5), scale_dt_s=round(patch[2], 3))
            en["scale_true"] = patch[3]
        elif lamp:
            d.update(scale_rate=round(lamp[0], 5), scale_rate_std=round(lamp[1] * p["lamp_scale_x"], 5),
                     scale_dt_s=round(lamp[2], 3), scale_src="lamps")
            en["scale_true"] = lamp[3]

    def _occluded(self, boxes):
        """Boxes sorted near to far -> hidden flags: a box covered by nearer boxes for at least `occl_cover` of its
        area (sum of the overlaps, capped at 1) is not reported (what a parked row looks like after NMS)."""
        out = []
        for j, b in enumerate(boxes):
            a = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
            cov = 0.0
            for k in range(j):
                c = boxes[k]
                if out[k]:
                    continue
                cov += max(0.0, min(b[2], c[2]) - max(b[0], c[0])) * max(0.0, min(b[3], c[3]) - max(b[1], c[1]))
            out.append(cov / a >= self.p["occl_cover"])
        return out

    @staticmethod
    def _clip(b, roi, rng):
        """A box cut to the detector's region; `clip` letters l t r b for the edges it touches (the unit's order)."""
        x0, y0, x1, y1 = b
        clip = ""
        if x0 <= roi[0] + 1.0:
            x0, clip = float(roi[0]), clip + "l"
        if y0 <= roi[1] + 1.0:
            y0, clip = float(roi[1]), clip + "t"
        if x1 >= roi[2] - 1.0:
            x1, clip = float(roi[2]), clip + "r"
        if y1 >= roi[3] - 1.0:
            y1, clip = roi[3] + rng.uniform(-0.6, 2.2), clip + "b"     # recorded: bottoms 0-2 px past the region
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None, ""
        return [x0, y0, x1, y1], clip

    @staticmethod
    def _fp_box(rng, roi):
        r = rng.random()
        cls = "car" if r < FP_CLASSES[0][1] else "person"
        h = rng.uniform(20.0, 80.0)
        w = h * (rng.uniform(1.0, 2.2) if cls == "car" else rng.uniform(0.3, 0.5))
        cx = rng.uniform(roi[0] + w / 2, roi[2] - w / 2)
        y1 = rng.uniform(roi[1] + 0.55 * (roi[3] - roi[1]), roi[3] + 0.3 * h)
        b, clip = DetectorSim._clip([cx - w / 2, y1 - h, cx + w / 2, y1], roi, rng)
        if b is None:
            return None
        return {"cls": cls, "box": [round(v, 1) for v in b], "clip": clip, "rider": False}, b

    @staticmethod
    def _post(w, t, rng):
        tr = w.truth_at(t)
        s = tr["s"] + CAM_TO_AXLE + rng.uniform(14.0, 40.0)
        d = rng.choice((-1, 1)) * rng.uniform(2.6, 6.0)
        x, y, _ = w._road_xy(s, d)
        return (x, y, s)

    @staticmethod
    def _post_actor(w, t, geo):
        tr = w.truth_at(t)
        x, y = w._to_car(tr, geo[0], geo[1])
        s_e = tr["s"] + CAM_TO_AXLE
        g_e = float(np.interp(s_e, w.S, w.grade_S))
        dz = float(np.interp(geo[2], w.S, w.alt_S) - np.interp(s_e, w.S, w.alt_S)) - math.sin(g_e) * (geo[2] - s_e)
        L, W, Hh = DIMS["post"]
        return {"x": x, "y": y, "heading": 0.0, "L": L, "W": W, "H": Hh, "dz": dz, "ref_x": x}


# ---- scenarios -------------------------------------------------------------------------------------------------
def _parked_row(rng, aid, s_from, s_to, side, lat=(2.6, 3.6), p_gap=0.2, p_truck=0.08):
    out = []
    s = s_from + rng.uniform(0, 4)
    while s < s_to:
        if rng.random() >= p_gap:
            kind = "truck" if rng.random() < p_truck else "car"
            dims = (6.0, 2.1, 2.6) if kind == "truck" else None         # a van, as parked vans are
            out.append(Actor(aid + len(out), kind, "parked", s, lateral=[(0, side * rng.uniform(*lat))],
                             yaw_rel=rng.gauss(0.0, 0.03), dims=dims))
            s += (dims[0] if dims else 4.5) + rng.uniform(0.8, 3.0)
        else:
            s += rng.uniform(5.0, 12.0)
    return out


def _ego_s(w, t):
    return float(np.interp(t, w.t - w.t0, w.s)) + CAM_TO_AXLE


def sc_parked_street(seed, night):
    """A one-way town street at 11 m/s: parked rows on both sides (2.6-3.6 m from the lane centre), pedestrians on
    the pavements (5-6 m), one pedestrian crossing in front of the car."""
    rng = random.Random(seed * 101 + 1)

    def actors(w):
        a = _parked_row(rng, 100, 25.0, 640.0, -1) + _parked_row(rng, 300, 25.0, 640.0, +1)
        for k in range(8):
            side = rng.choice((-1, 1))
            v = rng.choice((-1, 1)) * rng.uniform(1.0, 1.6)
            a.append(Actor(500 + k, "human", "pedestrian", rng.uniform(40, 560), speed=[(0, v)],
                           lateral=[(0, side * rng.uniform(5.0, 6.0))]))
        # crossing: starts when the car is ~85 m away, crosses 10 m in ~8 s
        tc = 22.0
        sc = _ego_s(w, tc) + 85.0
        a.append(Actor(600, "human", "crossing_ped", sc, lateral=[(0, -5.5), (tc, -5.5), (tc + 8.0, 5.5)],
                       speed=[(0, 0.0)]))
        return a
    return dict(T=50.0, speed=[(0, 11.0)], curv=[(0, 0.0), (200, 0.0), (200, 1 / 400.0), (380, 1 / 400.0), (380, 0.0)],
                actors=actors, lead_range_m=70.0 if night else 100.0)


def sc_oncoming(seed, night):
    """A two-way road at 14 m/s: oncoming cars in the opposite lane (3.5 m left) every 4-9 s at 11-17 m/s, a
    parked row on the right (2.6-3.6 m), a lead far ahead at 15 m/s."""
    rng = random.Random(seed * 101 + 2)

    def actors(w):
        a = _parked_row(rng, 100, 30.0, 950.0, -1, p_gap=0.35)
        t, k = 0.0, 0
        while t < 60.0:
            v = rng.uniform(11.0, 17.0)
            # meets the ego (x = 0) at t_meet; starts far ahead
            t_meet = t + 6.0
            s_meet = _ego_s(w, t_meet)
            s0 = s_meet + v * t_meet
            a.append(Actor(400 + k, "truck" if rng.random() < 0.1 else "car", "oncoming", s0, speed=[(0, -v)],
                           lateral=[(0, LANE_W + rng.gauss(0, 0.2))]))
            k += 1
            t += rng.uniform(4.0, 9.0)
        a.append(Actor(700, "car", "lead", 90.0, speed=[(0, 15.0)], lateral=[(0, 0.0)]))
        return a
    return dict(T=60.0, speed=[(0, 14.0)], curv=[(0, 0.0), (300, 0.0), (300, -1 / 500.0), (600, -1 / 500.0), (600, 0.0)],
                actors=actors, lead_range_m=70.0 if night else 100.0)


def sc_junction(seed, night):
    """A junction: the ego approaches at 12 m/s, stops 10 m before the crossing road, waits 14 s, goes on at 8 m/s.
    Crossing cars from both sides at 9-13 m/s on the crossing road's two lanes; a parked car on each corner."""
    rng = random.Random(seed * 101 + 3)
    spec = dict(T=50.0, speed=[(0, 12.0), (12, 12.0), (16, 0.0), (30, 0.0), (33, 8.0)], curv=[(0, 0.0)],
                lead_range_m=70.0 if night else 100.0, accel_lim=2.5)

    def actors(w):
        i_stop = int(np.argmax((w.t - w.t0 > 15.0) & (w.v < 0.05)))
        s_cross = float(w.s[i_stop]) + CAM_TO_AXLE + 10.0
        a = []
        t, k = 2.0, 0
        while t < 46.0:
            side = rng.choice((-1, 1))                      # coming from the right (side -1) or the left
            v = rng.uniform(9.0, 13.0)
            off = 2.0 if side < 0 else -2.0                 # its lane on the crossing road (right-hand traffic)
            a.append(Actor(400 + k, "car", "crossing", s_cross + off, speed=[(0, 0.0)],
                           lateral=[(0, side * 45.0), (t, side * 45.0), (t + 90.0 / v, -side * 45.0)],
                           t_on=t, t_off=t + 90.0 / v))
            k += 1
            t += rng.uniform(2.5, 6.0)
        for j, (ds, q) in enumerate(((-8.0, -3.0), (-15.0, 3.0), (12.0, -3.2), (18.0, 3.1))):
            a.append(Actor(100 + j, "car", "parked", s_cross + ds, lateral=[(0, q)]))
        a.append(Actor(500, "human", "pedestrian", s_cross - 30.0, speed=[(0, 1.2)], lateral=[(0, -5.5)]))
        return a
    spec["actors"] = actors
    return spec


def sc_adjacent(seed, night):
    """A dual carriageway at 25 m/s: a car in the left lane at about the ego's speed, 10-30 m ahead, for the
    whole run; a truck in the right lane at 21 m/s being overtaken; a lead 60 m ahead at 26 m/s."""
    rng = random.Random(seed * 101 + 4)

    def actors(w):
        return [Actor(400, "car", "adjacent", 20.0 + rng.uniform(-5, 5),
                      speed=[(0, 25.0), (20, 25.6), (40, 24.6), (60, 25.2)], lateral=[(0, LANE_W)]),
                Actor(401, "truck", "overtaken", 160.0, speed=[(0, 21.0)], lateral=[(0, -LANE_W)]),
                Actor(402, "car", "lead", 65.0, speed=[(0, 26.0), (30, 25.0), (60, 26.0)], lateral=[(0, 0.0)])]
    return dict(T=60.0, speed=[(0, 25.0)],
                curv=[(0, 0.0), (400, 0.0), (400, 1 / 900.0), (900, 1 / 900.0), (900, 0.0)],
                actors=actors, lead_range_m=70.0 if night else 100.0)


def sc_slow_lead(seed, night):
    """A country road: the ego at 20 m/s closes on a lead at 7 m/s (180 m ahead at the start), follows it from
    about 18 s; a few parked cars on the verge."""
    rng = random.Random(seed * 101 + 5)

    def actors(w):
        a = [Actor(700, "car", "slow_lead", 182.0, speed=[(0, 7.0)], lateral=[(0, 0.0)])]
        a += _parked_row(rng, 100, 200.0, 420.0, -1, lat=(3.0, 3.8), p_gap=0.6)
        return a
    return dict(T=60.0, speed=[(0, 20.0), (8, 20.0), (16, 7.0)], curv=[(0, 0.0)], actors=actors,
                lead_range_m=70.0 if night else 100.0)


def sc_curve_parked(seed, night):
    """The classic trap: a town road with a left then a right curve (R 110 m) at 13 m/s, cars parked on the
    OUTSIDE of each curve (right of the left curve, left of the right curve) and some on the inside; nothing
    stands in the ego's path."""
    rng = random.Random(seed * 101 + 6)

    def actors(w):
        a = _parked_row(rng, 100, 60.0, 330.0, -1, lat=(2.6, 3.2), p_gap=0.1)     # outside of the left curve
        a += _parked_row(rng, 200, 400.0, 680.0, +1, lat=(2.6, 3.2), p_gap=0.1)   # outside of the right curve
        a += _parked_row(rng, 300, 100.0, 650.0, +1, lat=(5.5, 6.5), p_gap=0.6)   # beyond the oncoming lane
        return a
    return dict(T=50.0, speed=[(0, 13.0)],
                curv=[(0, 0.0), (80, 0.0), (80, 1 / 110.0), (300, 1 / 110.0), (300, 0.0), (400, 0.0),
                      (400, -1 / 110.0), (620, -1 / 110.0), (620, 0.0)],
                actors=actors, lead_range_m=70.0 if night else 100.0)


def sc_stopped(speed, mode):
    """A stationary car in the ego lane on a straight road. mode "brake": the ego cruises at `speed` and brakes
    (up to 4.9 m/s^2, jerk 3 m/s^3) to stop 6 m behind it; "lane": the ego keeps its speed and changes to the
    left lane (3.5 m over 2.5 s) starting at 3 s to collision, passing the car."""
    def build(seed, night):
        rng = random.Random(seed * 101 + 7 + int(speed))
        far = max(200.0, 12.0 * speed) + rng.uniform(0, 20)
        L = DIMS["car"][0]
        if mode == "brake":
            spec = dict(curv=[(0, 0.0)], accel_lim=3.5)
            # brake time so that the stop lands 6 m behind the car: solve on a ego-only world
            t_cr = far / speed
            lo, hi = 0.5, t_cr
            for _ in range(30):
                tb = 0.5 * (lo + hi)
                T = tb + 2.0 + speed / 3.0 + 12.0
                w0 = World(seed=1, scenario=dict(T=T, speed=[(0, speed), (tb, speed), (tb + 0.01, 0.0)], curv=[(0, 0.0)]),
                           targets=[], statics=False, accel_lim=3.5)
                gap = (far + CAM_TO_AXLE) - (float(w0.s[-1]) + CAM_TO_AXLE)
                if gap > 6.0:
                    lo = tb
                else:
                    hi = tb
            spec.update(T=T, speed=[(0, speed), (tb, speed), (tb + 0.01, 0.0)], brake_t=tb)
        else:
            t_lc = (far - 3.0 * speed) / speed
            T = far / speed + 6.0
            spec = dict(T=T, speed=[(0, speed)], curv=[(0, 0.0)],
                        lane_change=(t_lc * speed + CAM_TO_AXLE, 2.5 * speed, LANE_W), lc_t=t_lc)
        s_car = far + CAM_TO_AXLE + L / 2

        def actors(w):
            return [Actor(800, "car", "stopped_in_lane", s_car, lateral=[(0, rng.gauss(0, 0.15))],
                          yaw_rel=rng.gauss(0, 0.02))]
        spec.update(actors=actors, lead_range_m=70.0 if night else 100.0, approach=speed, mode=mode)
        return spec
    return build


# ---- tracker-core scenarios (association, occlusion, rows, queues): docs/VISUAL_RADAR_EVAL.md -----------------
# Each spec carries `events` [(t s from the start, what)]: the designed moments (the measured ones, e.g. the
# truth-occluded intervals, come from DetectorSim.occlusion); and `groups` {group id: kind}.
def _travel(plan, t, dt=0.01):
    """Distance an actor with speed plan `plan` covers from 0 to t (as ScenarioWorld integrates it)."""
    if t <= 0.0:
        return 0.0
    tt = np.arange(0.0, t + 0.5 * dt, dt)
    v = _plan(plan, tt)
    return float(np.sum(0.5 * (v[1:] + v[:-1]) * dt))


def _place(w, t, x, plan):
    """s0 of an actor with speed plan `plan` whose footprint centre is x metres (along the road) ahead of the
    camera at time t (s from the start)."""
    return _ego_s(w, t) + x - _travel(plan, t)


def sc_crossing_pair(seed, night):
    """Association ambiguity, three acts on a straight road (ego 13 m/s, slowing to 6 m/s from 22 s):
    (a) 0-14 s: pair_b drives ahead in the ego lane at 13 m/s (~40 m); pair_a overtakes it in the left lane at
        16.5 m/s from 22 m: equal range side by side (3.5 m apart) at ~5 s, then it cuts into the ego lane in front
        of pair_b (9-12 s): the two share a bearing at different ranges, pair_a hidden, until pair_b moves to the
        right lane (15-17 s) and reveals it;
    (b) 11-18 s: pair_c, an oncoming car (3.5 m left, 12 m/s), passes in front of pair_d, a car parked on the
        far verge (6.6 m left): the two at the same bearing at ~16 s (ranges ~23 and ~44 m);
    (c) 27-37 s: ped_a and ped_b cross the road ~26 m ahead in opposite directions (1.4 and 1.25 m/s, 0.8 m apart
        along the road) and pass each other in the middle at ~32 s."""
    rng = random.Random(seed * 101 + 11)
    events = []

    def actors(w):
        a = []
        pb = [(0, 13.0)]
        a.append(Actor(901, "car", "pair_b", _place(w, 0.0, 40.0 + rng.uniform(-2, 2), pb), speed=pb,
                       lateral=[(0, rng.gauss(0, 0.1)), (15.0, 0.0), (17.0, -LANE_W)]))
        va = 16.5 + rng.uniform(-0.3, 0.3)
        pa = [(0, va)]
        a.append(Actor(900, "car", "pair_a", _place(w, 0.0, 22.0, pa), speed=pa,
                       lateral=[(0, LANE_W), (9.0, LANE_W), (12.0, 0.0)]))
        events.extend([(5.1, "pair_a / pair_b side by side at equal range (~40 m, 3.5 m apart)"),
                       (9.0, "pair_a starts to cut in in front of pair_b"),
                       (12.0, "pair_a in the ego lane beyond pair_b: same bearing, ranges ~40 / ~60 m (hidden)"),
                       (15.0, "pair_b changes to the right lane: pair_a revealed")])
        tc = 16.0
        xd = 44.0 + rng.uniform(-2, 2)
        qd = 6.6
        a.append(Actor(903, "car", "pair_d", _ego_s(w, tc) + xd, lateral=[(0, qd)], yaw_rel=rng.gauss(0, 0.03)))
        pc = [(0, -12.0 + rng.uniform(-0.5, 0.5))]
        a.append(Actor(902, "car", "pair_c", _place(w, tc, xd * LANE_W / qd, pc), speed=pc, lateral=[(0, LANE_W)]))
        events.append((tc, "oncoming pair_c in front of parked pair_d at the same bearing"))
        tm = 32.0
        sx = _ego_s(w, tm) + 26.0
        va, vb = 1.4, 1.25
        ha, hb = 6.0 / va, 6.0 / vb
        a.append(Actor(904, "human", "ped_a", sx, lateral=[(0, -6.0), (tm - ha, -6.0), (tm + ha, 6.0)]))
        a.append(Actor(905, "human", "ped_b", sx + 0.8, lateral=[(0, 6.0), (tm - hb, 6.0), (tm + hb, -6.0)]))
        events.append((tm, "ped_a / ped_b pass each other in the middle of the road"))
        events.sort()
        return a
    return dict(T=40.0, speed=[(0, 13.0), (22, 13.0), (26, 6.0)], curv=[(0, 0.0)], actors=actors, events=events,
                groups={}, lead_range_m=70.0 if night else 100.0)


def sc_occlusion(seed, night):
    """Objects hidden behind others and re-appearing (ego 15 m/s, 9 m/s from 17 s), three acts:
    (a) 0-11 s: occ_lead drives ahead in the ego lane (15 m/s, ~36 m); occ_truck (8 x 2.5 x 3.4 m) overtakes on
        the left at 18.5 m/s from beside the car, cuts in between the ego and the lead (2.5-4.5 s), drives there
        at 15 m/s and pulls back out to the left (7-9 s) and away at 22 m/s: the lead hidden for ~4 s;
    (b) 15-30 s: occ_ped walks away from the car on the right pavement (5.6 m right, 1.3 m/s), beyond a short row
        of a parked car and a van (occ_row, 3.0 m right) ~20 m nearer: it goes behind the row at ~22.6 s and comes
        out in front of it at ~27.9 s (the row slides past its line of sight);
    (c) 24-45 s: edge_car drives in the left lane beside and behind the car (out of the image), moves ahead into
        the image (cut at its left edge first) from ~33 s, drops back out of it (~37-38 s) and overtakes again
        (back in the image ~42 s)."""
    rng = random.Random(seed * 101 + 12)
    events = []
    ego = [(0, 15.0), (14, 15.0), (17, 9.0)]

    def actors(w):
        a = []
        pl = [(0, 15.0)]
        a.append(Actor(910, "car", "occ_lead", _place(w, 0.0, 36.0 + rng.uniform(-1.5, 1.5), pl), speed=pl,
                       lateral=[(0, rng.gauss(0, 0.1))]))
        pt = [(0, 18.5), (3.5, 18.5), (5.0, 15.0), (7.5, 15.0), (9.5, 22.0)]
        a.append(Actor(911, "truck", "occ_truck", _place(w, 0.0, 4.0, pt), speed=pt,
                       lateral=[(0, LANE_W), (2.5, LANE_W), (4.5, 0.0), (7.0, 0.0), (9.0, LANE_W)]))
        events.extend([(2.5, "occ_truck starts to cut in between the ego and occ_lead"),
                       (7.0, "occ_truck pulls back out to the left: occ_lead revealed")])
        # (b) the row and the pedestrian behind it
        t_ref = 22.0
        x0 = 50.0 + rng.uniform(-0.5, 0.5)
        kinds = ["car", "van"]
        x = x0
        for k, kd in enumerate(kinds):
            dims = (6.0, 2.1, 2.6) if kd == "van" else None
            ln = dims[0] if dims else DIMS["car"][0]
            a.append(Actor(920 + k, "truck" if dims else "car", "occ_row", _ego_s(w, t_ref) + x + ln / 2,
                           lateral=[(0, -3.0 + rng.gauss(0, 0.1))], yaw_rel=rng.gauss(0, 0.03), dims=dims,
                           group="occ_row", group_kind="row"))
            x += ln + rng.uniform(1.0, 1.4)
        pp = [(0, 1.3)]
        a.append(Actor(912, "human", "occ_ped", _place(w, t_ref, 70.0, pp), speed=pp, lateral=[(0, -5.6)]))
        events.extend([(22.6, "occ_ped (walking away, beyond occ_row) goes behind occ_row"),
                       (27.9, "occ_ped re-emerges in front of occ_row")])
        # (c) the car beside: the ego's own speed plan up to 30 s, about 6 m behind the camera
        pe = [(0, 9.0), (30.0, 9.0), (31.0, 13.0), (35.5, 13.0), (36.5, 6.0), (40.5, 6.0), (41.5, 13.0)]
        a.append(Actor(913, "car", "edge_car", _place(w, 30.0, -6.0, pe), speed=pe, lateral=[(0, LANE_W)], t_on=24.0))
        events.extend([(30.0, "edge_car beside and behind the car (outside the image) moves ahead"),
                       (37.0, "edge_car drops back out of the image (left edge)"), (41.5, "edge_car overtakes again")])
        events.sort()
        return a
    return dict(T=45.0, speed=ego, curv=[(0, 0.0)], actors=actors, events=events, groups={"occ_row": "row"},
                lead_range_m=70.0 if night else 100.0)


def sc_parked_row(seed, night):
    """Rows of parked cars on a one-way street at 10 m/s, a gentle right curve (R 250 m) from 230 to 380 m:
    row_r1: 10 slots parallel on the right (2.9 m right, 5.5-6.5 m pitch), one slot empty, at 55-115 m;
    row_l1: 14 bays perpendicular on the left (4.4 m left, 2.6 m pitch, yaw_rel +-90 deg), one bay empty, 70-106 m;
    row_r2: 8 cars parallel on the right in the curve at 255-305 m; its 6th car (role pull_out) pulls out into the
    ego lane at 24 s, ~44 m ahead of the car, and drives off at 11 m/s (it leaves the group when it moves)."""
    rng = random.Random(seed * 101 + 13)
    events = []

    def actors(w):
        a = []
        s_e0 = _ego_s(w, 0.0)
        s = s_e0 + 55.0
        gap = rng.randrange(2, 8)
        for k in range(10):
            pitch = rng.uniform(5.5, 6.5)
            if k != gap:
                a.append(Actor(1000 + k, "car", "parked_par", s + 2.25, lateral=[(0, -2.9 + rng.gauss(0, 0.1))],
                               yaw_rel=rng.gauss(0, 0.03), group="row_r1", group_kind="row"))
            s += pitch
        s = s_e0 + 70.0
        gap = rng.randrange(3, 11)
        for k in range(14):
            if k != gap:
                a.append(Actor(1100 + k, "car", "parked_bay", s, lateral=[(0, 4.4 + rng.gauss(0, 0.15))],
                               yaw_rel=rng.choice((-1, 1)) * math.pi / 2 + rng.gauss(0, 0.04),
                               group="row_l1", group_kind="row"))
            s += 2.6
        s = s_e0 + 255.0
        t_pull = 24.0
        for k in range(8):
            q = -2.9 + rng.gauss(0, 0.1)
            if k == 5:
                sp = [(0, 0.0), (t_pull, 0.0), (t_pull + 4.0, 8.0), (t_pull + 6.0, 11.0)]
                a.append(Actor(1205, "car", "pull_out", s + 2.25, speed=sp,
                               lateral=[(0, q), (t_pull + 0.3, q), (t_pull + 3.8, 0.0)], yaw_rel=rng.gauss(0, 0.03),
                               group="row_r2", group_kind="row", group_t=(-1e9, t_pull)))
            else:
                a.append(Actor(1200 + k, "car", "parked_par", s + 2.25, lateral=[(0, q)], yaw_rel=rng.gauss(0, 0.03),
                               group="row_r2", group_kind="row"))
            s += rng.uniform(5.5, 6.5)
        events.extend([(5.0, "row_r1 / row_l1 begin ~50 m ahead"), (12.0, "the car passes row_r1 / row_l1"),
                       (21.0, "the curve: row_r2 ahead on the right"),
                       (t_pull, "pull_out (row_r2's 6th car) pulls out ~44 m ahead")])
        return a
    return dict(T=40.0, speed=[(0, 10.0)], curv=[(0, 0.0), (230, 0.0), (230, -1 / 250.0), (380, -1 / 250.0), (380, 0.0)],
                actors=actors, events=events, groups={"row_r1": "row", "row_l1": "row", "row_r2": "row"},
                lead_range_m=70.0 if night else 100.0)


def sc_queue(seed, night):
    """A queue at a light in a gentle left bend (R 150 m, starting 4 m ahead of where the ego stops): the ego
    (12 m/s) brakes from 6 s and stops 6 m behind the last of 5 standing vehicles (q1 the front ... q5 the last,
    6.2-7.2 m pitch, q2 a van that shows above the cars; on the straight approach the farther ones are hidden
    behind the nearer ones, in the bend they come out to the left one by one), waits 8 s; the queue pulls away front to back from 0.5 s after the ego stands (q1 first,
    each next 1.5 s later, 0 -> 8 m/s in ~5.3 s: the accordion; q5 6.5 s after the stop) and the ego follows
    (8 m/s). Group "queue1" (kind queue) for the whole run."""
    rng = random.Random(seed * 101 + 14)
    events = []
    approach = [(0, 12.0), (6, 12.0), (10, 0.0)]
    w0 = World(seed=1, scenario=dict(T=40.0, speed=approach, curv=[(0, 0.0)]), targets=[], statics=False,
               accel_lim=2.5)
    i0 = int(np.argmax((w0.t - w0.t0 > 6.0) & (w0.v < 0.05)))
    t_stop = float(w0.t[i0] - w0.t0)
    s_bend = float(w0.s[i0]) + CAM_TO_AXLE + 4.0        # the light stands in a gentle left bend (R 150 m)
    spec = dict(T=40.0, speed=approach + [(t_stop + 8.0, 0.0), (t_stop + 13.0, 8.0)],
                curv=[(0, 0.0), (s_bend, 0.0), (s_bend + 15.0, 1 / 150.0), (s_bend + 250.0, 1 / 150.0)],
                accel_lim=2.5, events=events, groups={"queue1": "queue"}, lead_range_m=70.0 if night else 100.0)

    def actors(w):
        i_stop = int(np.argmax((w.t - w.t0 > 6.0) & (w.v < 0.05)))
        s_stop = float(w.s[i_stop]) + CAM_TO_AXLE
        a = []
        s = s_stop + 6.0 + 2.0 + DIMS["car"][0] / 2         # 6 m from the car's front (2 m ahead of the camera)
        n = 5
        for k in range(n):                                   # k = 0: the last vehicle (q5), nearest
            qn = n - k
            van = qn == 2
            dims = (6.0, 2.1, 2.6) if van else None
            ln = dims[0] if dims else DIMS["car"][0]
            t_go = t_stop + 0.5 + 1.5 * (qn - 1)
            a.append(Actor(1300 + qn - 1, "truck" if van else "car", "q%d" % qn, s + (ln - DIMS["car"][0]) / 2,
                           speed=[(0, 0.0), (t_go, 0.0), (t_go + 5.3 + rng.uniform(-0.5, 0.5), 8.0)],
                           lateral=[(0, rng.gauss(0, 0.3))], yaw_rel=rng.gauss(0, 0.02), dims=dims, group="queue1",
                           group_kind="queue"))
            s += ln + rng.uniform(1.7, 2.7)
        events.extend([(round(t_stop, 2), "the ego stands 6 m behind q5"), (round(t_stop + 0.5, 2), "q1 (front) pulls away"),
                       (round(t_stop + 6.5, 2), "q5 (nearest) pulls away"), (round(t_stop + 8.0, 2), "the ego follows")])
        return a
    spec["actors"] = actors
    return spec


# ---- scenarios that exercise the predictor's manoeuvres (chris_integrator/predict, tools/predict_eval.py) -------
# Each carries a `layout` (the lanes of the road: ScenarioWorld.road_context) and `labels`: (actor id, from, to (s
# of the run), names of the manoeuvre as the predictor publishes it), the truth of what the actor does when.
def _lc(t0, d0, d1, dur=4.0):
    """A lane change as a lateral plan: minimum jerk from d0 to d1 over dur, sampled."""
    out = []
    for i in range(11):
        u = i / 10.0
        out.append((t0 + u * dur, d0 + (d1 - d0) * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)))
    return out


def _brake_to(t_end, d_end, v, dec, sg, v_end=0.0):
    """A lateral plan that arrives at d_end at t_end with speed v_end, braking from speed v with deceleration dec
    (approaching from the side sg, +1 = from the left): [(t, d)] from the start of the braking."""
    tb = (v - v_end) / dec
    out = []
    for i in range(9):
        left = tb * (1.0 - i / 8.0)                      # time still to go
        out.append((t_end - left, d_end + sg * (v_end * left + 0.5 * dec * left * left)))
    return out


def sc_cut_in(seed, night):
    """Two lanes one way at 22 m/s (the ego in the right one): a car in the left lane a little ahead cuts in in
    front of the ego at 9 s (4 s), brakes gently, and cuts out again at 24 s; a second car stays in the left lane;
    a lead far ahead."""
    rng = random.Random(seed * 101 + 11)
    t1, t2 = 9.0 + rng.uniform(-1, 1), 24.0 + rng.uniform(-1, 1)

    def actors(w):
        a = [Actor(400, "car", "cut_in", 14.0 + rng.uniform(0, 4), speed=[(0, 23.0), (t1, 23.0), (t1 + 4, 21.3), (t2, 21.8), (t2 + 3, 23.5)],
                   lateral=[(0, LANE_W)] + _lc(t1, LANE_W, 0.0) + _lc(t2, 0.0, LANE_W)),
             Actor(401, "car", "adjacent", 55.0, speed=[(0, 23.0), (20, 23.6), (40, 22.6)], lateral=[(0, LANE_W)]),
             Actor(402, "car", "lead", 95.0, speed=[(0, 22.0), (20, 22.5), (40, 22.0)], lateral=[(0, 0.0)])]
        w.spec["labels"] = [(400, t1, t1 + 4.0, ("cut_in",)), (400, t2, t2 + 4.0, ("cut_out",))]
        return a
    return dict(T=36.0, speed=[(0, 22.0)], curv=[(0, 0.0), (300, 0.0), (300, 1 / 800.0), (700, 1 / 800.0), (700, 0.0)],
                actors=actors, lead_range_m=70.0 if night else 100.0, layout=dict(n_left=1, n_right=0, oncoming=False))


def sc_junction_priority(seed, night):
    """A junction on a priority road, driven through at 11 m/s: cars on the side road that stop at the line and
    wait (give way), cars that cross before the ego arrives, one that turns right out of the side road into the
    ego's lane ahead of it, an oncoming car that turns left across the ego's lane into the side road, oncoming
    traffic that drives straight on; a parked car on each corner."""
    rng = random.Random(seed * 101 + 12)

    def actors(w):
        s_x = 190.0 + rng.uniform(0, 20)
        t_x = float(np.interp(s_x, w.s + CAM_TO_AXLE, w.t - w.t0))          # the ego is at the junction then
        a, lab = [], []
        # stop and wait at the line: from the right (side -1), 5.5 m short of the ego lane's centre
        for k, (side, t_arr) in enumerate(((-1, t_x - 7.0), (1, t_x - 5.0))):
            v = rng.uniform(8.0, 11.0)
            line = side * (LANE_W * (0.5 if side < 0 else 1.5) + 3.5)
            br = _brake_to(t_arr, line, v, 2.0, side)           # brakes at 2 m/s2 to the line
            t0 = br[0][0] - (45.0 - abs(br[0][1])) / v
            a.append(Actor(410 + k, "car", "yielding", s_x + (2.0 if side < 0 else -2.0), speed=[(0, 0.0)],
                           lateral=[(0, side * 45.0), (t0, side * 45.0)] + br + [(t_x + 3.0, line), (t_x + 9.0, -side * 30.0)],
                           t_on=max(0.0, t0)))
            lab.append((410 + k, br[0][0], t_x + 2.0, ("stop", "stopped", "move_off")))
        # crossing well before the ego
        for k in range(2):
            side = rng.choice((-1, 1))
            v = rng.uniform(9.0, 12.0)
            t0 = t_x - 16.0 + 4.0 * k
            a.append(Actor(420 + k, "car", "crossing", s_x + (2.0 if side < 0 else -2.0), speed=[(0, 0.0)],
                           lateral=[(0, side * 45.0), (t0, side * 45.0), (t0 + 90.0 / v, -side * 45.0)], t_on=max(0.0, t0),
                           t_off=t0 + 90.0 / v))
            lab.append((420 + k, t0, t0 + 90.0 / v, ("cross",)))
        # out of the right side road into the ego's lane, ahead of the ego: a right turn
        tt = t_x - 9.0
        br = _brake_to(tt + 1.5, -8.0, 9.0, 1.8, -1, v_end=3.2)       # brakes to 3.2 m/s for the corner
        t0 = br[0][0] - (45.0 + br[0][1]) / 9.0
        a.append(Actor(430, "car", "turn_in", s_x + 2.0, speed=[(0, 0.0), (tt + 1.5, 0.0), (tt + 4.0, 5.0), (tt + 9.0, 11.5)],
                       lateral=[(0, -45.0), (t0, -45.0)] + br + [(tt + 2.8, -3.6), (tt + 4.5, -0.6), (tt + 6.0, 0.0)],
                       t_on=max(0.0, t0)))
        lab.append((430, br[0][0], tt + 4.5, ("turn_right",)))
        # an oncoming car turns left into the right side road, across the ego's lane
        tl = t_x - 12.0
        s0 = s_x + 11.0 * (tl + 3.0) + 4.0
        a.append(Actor(440, "car", "onc_turn", s0, speed=[(0, -11.0), (tl, -11.0), (tl + 3.0, -4.0), (tl + 5.0, 0.0)],
                       lateral=[(0, LANE_W), (tl + 2.5, LANE_W), (tl + 3.5, LANE_W - 1.2), (tl + 4.5, -1.5), (tl + 5.5, -5.5),
                                (tl + 10.0, -40.0)], t_off=tl + 10.0))
        lab.append((440, tl + 0.5, tl + 5.0, ("turn_left",)))
        t = 1.0
        k = 0
        while t < w.T - 8.0:
            v = rng.uniform(10.0, 13.0)
            t_meet = t + 7.0
            if abs(t_meet - (tl + 4.0)) > 3.0:
                a.append(Actor(450 + k, "car", "oncoming", _ego_s(w, t_meet) + v * t_meet, speed=[(0, -v)],
                               lateral=[(0, LANE_W + rng.gauss(0, 0.15))]))
                k += 1
            t += rng.uniform(5.0, 9.0)
        for j, (ds, q) in enumerate(((-22.0, -3.0), (16.0, -3.1))):
            a.append(Actor(100 + j, "car", "parked", s_x + ds, lateral=[(0, q)]))
        w.spec["junctions"] = [{"s": s_x, "arms": [("left", -1), ("right", -1)], "control": "give_way"}]
        w.spec["labels"] = lab
        return a
    return dict(T=38.0, speed=[(0, 11.0)], curv=[(0, 0.0)], actors=actors, lead_range_m=70.0 if night else 100.0,
                layout=dict(n_left=0, n_right=0, oncoming=True, park_right=0.6, park_left=0.0))


def sc_ped_wait_cross(seed, night):
    """A two-way town street at 10 m/s with a marked crossing: a pedestrian walks along the pavement to the
    crossing, waits at the kerb for 4 s and crosses while the ego is still approaching (the ego brakes and lets it
    pass); a second one stands at the kerb further on and never crosses; a third walks along; one crosses away
    from any crossing."""
    rng = random.Random(seed * 101 + 13)
    spec = dict(T=42.0, curv=[(0, 0.0)], lead_range_m=70.0 if night else 100.0, accel_lim=2.5,
                layout=dict(n_left=0, n_right=0, oncoming=True, park_right=0.0, park_left=0.0, kerb_right_m=2.1, kerb_left_m=5.6))
    t_wait = 11.0 + rng.uniform(-1, 1)
    t_go = t_wait + 4.0
    spec["speed"] = [(0, 10.0), (t_go - 0.5, 10.0), (t_go + 2.5, 0.0), (t_go + 6.5, 0.0), (t_go + 10.0, 9.0)]

    def actors(w):
        s_x = _ego_s(w, t_go + 3.0) + 9.0                # the ego stops 9 m short of the crossing
        a = [Actor(600, "human", "wait_cross", s_x - 1.3 * t_wait, speed=[(0, 1.3), (t_wait - 0.5, 1.3), (t_wait, 0.0)],
                   lateral=[(0, -3.6), (t_wait - 2.0, -3.6), (t_wait, -2.6), (t_go, -2.6), (t_go + 7.0, 6.6), (t_go + 8.0, 6.8)]),
             Actor(601, "human", "waiting", s_x + 45.0, speed=[(0, 0.0)], lateral=[(0, -2.7)]),
             Actor(602, "human", "pedestrian", s_x + 70.0, speed=[(0, -1.3)], lateral=[(0, -3.8)]),
             Actor(603, "human", "pedestrian", s_x - 30.0, speed=[(0, 1.4)], lateral=[(0, 6.9)])]
        tc = t_go + 14.0
        sc2 = _ego_s(w, tc) + 60.0
        a.append(Actor(604, "human", "crossing_ped", sc2, speed=[(0, 0.0)], lateral=[(0, 7.0), (tc, 7.0), (tc + 7.0, -3.5)]))
        w.spec["crossings"] = [s_x]
        w.spec["labels"] = [(600, t_wait, t_go, ("wait", "start_cross")), (600, t_go - 1.5, t_go, ("start_cross",)),
                            (600, t_go, t_go + 6.5, ("crossing", "start_cross")), (601, 0.0, 99.0, ("wait", "stand")),
                            (604, tc, tc + 7.0, ("crossing", "start_cross"))]
        return a
    spec["actors"] = actors
    return spec


def sc_platoon(seed, night):
    """A platoon: 4 vehicles (p1 the nearest ... p4; p3 a van) ahead in the ego lane at 20 m/s +- 0.25 m/s with
    12-20 m gaps, the ego following p1 at ~25 m, on a gently curved road (left R 350 m, then right R 400 m): on
    the straight the farther ones hide behind the nearer ones, the curves open them. Group "platoon1"."""
    rng = random.Random(seed * 101 + 15)
    events = [(7.5, "left curve begins"), (22.5, "right curve begins")]

    def actors(w):
        a = []
        x = 25.0 + rng.uniform(-2, 2)
        for k in range(4):
            van = k == 2
            dims = (6.0, 2.1, 2.6) if van else None
            ln = dims[0] if dims else DIMS["car"][0]
            d = rng.uniform(-0.25, 0.25)
            sp = [(0, 20.0 + d), (12, 20.0 - d), (24, 20.0 + d), (35, 20.0)]
            a.append(Actor(1400 + k, "truck" if van else "car", "p%d" % (k + 1), _place(w, 0.0, x + ln / 2, sp),
                           speed=sp, lateral=[(0, rng.gauss(0, 0.15))], dims=dims, group="platoon1",
                           group_kind="platoon"))
            x += ln + rng.uniform(13.0, 18.0)
        return a
    return dict(T=35.0, speed=[(0, 20.0)],
                curv=[(0, 0.0), (150, 0.0), (150, 1 / 350.0), (420, 1 / 350.0), (450, -1 / 400.0), (650, -1 / 400.0),
                      (680, 0.0)], actors=actors, events=events, groups={"platoon1": "platoon"},
                lead_range_m=70.0 if night else 100.0)


def sc_curve_parked_oncoming(seed, night):
    """The trap with traffic: a two-way town road with a left then a right curve (R 110 m) at 13 m/s, cars parked on
    the right all along (outside of the left curve, inside of the right one), oncoming cars every 4-8 s, one of
    which drifts 1.3 m towards the centre line for a few seconds; nothing stands in the ego's path."""
    rng = random.Random(seed * 101 + 14)

    def actors(w):
        a = _parked_row(rng, 100, 60.0, 680.0, -1, lat=(2.7, 3.2), p_gap=0.25)
        t, k, lab = 0.0, 0, []
        drift_k = 2
        while t < 44.0:
            v = rng.uniform(10.0, 14.0)
            t_meet = t + 6.5
            lat = [(0, LANE_W + rng.gauss(0, 0.15))]
            if k == drift_k:
                lat = [(0, LANE_W), (t_meet - 5.0, LANE_W), (t_meet - 2.5, LANE_W - 1.3), (t_meet - 1.0, LANE_W - 1.3), (t_meet + 1.0, LANE_W)]
                lab.append((400 + k, t_meet - 5.0, t_meet - 2.0, ("onc_drift", "onc_overtake")))
            a.append(Actor(400 + k, "car", "oncoming", _ego_s(w, t_meet) + v * t_meet, speed=[(0, -v)], lateral=lat))
            k += 1
            t += rng.uniform(4.0, 8.0)
        w.spec["labels"] = lab
        return a
    return dict(T=50.0, speed=[(0, 13.0)],
                curv=[(0, 0.0), (80, 0.0), (80, 1 / 110.0), (300, 1 / 110.0), (300, 0.0), (400, 0.0),
                      (400, -1 / 110.0), (620, -1 / 110.0), (620, 0.0)],
                actors=actors, lead_range_m=70.0 if night else 100.0,
                layout=dict(n_left=0, n_right=0, oncoming=True, park_right=0.9, park_left=0.0))


def sc_pull_out(seed, night):
    """A town street at 11 m/s with a parked row on the right: one of the parked cars pulls out into the ego's lane
    45 m ahead of the ego (3.5 s to the lane's centre, 1.5 m/s2 up to 9 m/s); the ego eases off behind it."""
    rng = random.Random(seed * 101 + 15)
    t_p = 12.0 + rng.uniform(-1, 1)
    spec = dict(T=34.0, speed=[(0, 11.0), (t_p + 0.5, 11.0), (t_p + 3.0, 7.0), (t_p + 8.0, 9.0)], curv=[(0, 0.0)],
                lead_range_m=70.0 if night else 100.0,
                layout=dict(n_left=0, n_right=0, oncoming=True, park_right=0.9, park_left=0.0))

    def actors(w):
        s_p = _ego_s(w, t_p) + 45.0
        a = [x for x in _parked_row(rng, 100, 30.0, 420.0, -1, lat=(2.8, 3.1), p_gap=0.25) if abs(x.s0 - s_p) > 7.0]
        a.append(Actor(700, "car", "pull_out", s_p, speed=[(0, 0.0), (t_p, 0.0), (t_p + 6.0, 9.0)],
                       lateral=[(0, -2.9), (t_p, -2.9)] + _lc(t_p + 0.3, -2.9, 0.0, 3.5)))
        w.spec["labels"] = [(700, t_p, t_p + 4.0, ("pull_out",))]
        return a
    spec["actors"] = actors
    return spec


def sc_queue_stop(seed, night):
    """A queue: two cars stand in the ego's lane; a lead 40 m ahead of the ego brakes from 13 m/s and stops behind
    them; the ego stops behind the lead; after 8 s the queue moves off."""
    rng = random.Random(seed * 101 + 16)
    t_b = 9.0 + rng.uniform(-1, 1)
    spec = dict(T=36.0, speed=[(0, 13.0), (t_b + 1.0, 13.0), (t_b + 7.5, 0.0), (t_b + 15.0, 0.0), (t_b + 20.0, 8.0)],
                curv=[(0, 0.0)], lead_range_m=70.0 if night else 100.0, accel_lim=2.5,
                layout=dict(n_left=0, n_right=0, oncoming=True, park_right=0.0, park_left=0.0))

    def actors(w):
        s_stop = float(w.s[int(np.argmax((w.t - w.t0 > t_b + 8.0) & (w.v < 0.05)))]) + CAM_TO_AXLE
        s_l = s_stop + 4.5 / 2 + 7.0                      # the lead's centre when it stands
        d_brake = 13.0 * 6.0 / 2.0
        s_l0 = s_l - d_brake - 13.0 * t_b
        go = t_b + 14.0
        a = [Actor(700, "car", "lead", s_l0, speed=[(0, 13.0), (t_b, 13.0), (t_b + 6.0, 0.0), (go + 1.0, 0.0), (go + 6.0, 9.0)]),
             Actor(701, "car", "stopped_in_lane", s_l + 7.0, speed=[(0, 0.0), (go, 0.0), (go + 5.0, 9.0)]),
             Actor(702, "car", "stopped_in_lane", s_l + 14.0, speed=[(0, 0.0), (go - 1.0, 0.0), (go + 4.0, 9.0)])]
        w.spec["labels"] = [(700, t_b, t_b + 6.0, ("stop", "follow"))]
        return a
    spec["actors"] = actors
    return spec


def sc_cyclist(seed, night):
    """A cyclist rides along the right edge of the ego's lane at 5 m/s; a car is parked ahead of it half in its
    way; it swerves 1.3 m into the lane to pass it and goes back. The ego follows 35 m behind at the cyclist's
    speed. A second cyclist comes the other way on the far side."""
    rng = random.Random(seed * 101 + 17)
    spec = dict(T=40.0, speed=[(0, 9.0), (4.0, 5.2)], curv=[(0, 0.0)], lead_range_m=70.0 if night else 100.0,
                layout=dict(n_left=0, n_right=0, oncoming=True, park_right=0.5, park_left=0.0))

    def actors(w):
        t_s = 16.0 + rng.uniform(-1, 1)
        s_c0 = 55.0
        s_park = s_c0 + 5.0 * (t_s + 3.0)
        a = [Actor(800, "bike", "cyclist", s_c0, speed=[(0, 5.0)],
                   lateral=[(0, -1.4), (t_s, -1.4)] + _lc(t_s, -1.4, -0.1, 2.5) + [(t_s + 4.5, -0.1)] + _lc(t_s + 4.5, -0.1, -1.4, 2.5)),
             Actor(100, "car", "parked", s_park, lateral=[(0, -2.5)]),
             Actor(801, "bike", "cyclist", 320.0, speed=[(0, -5.5)], lateral=[(0, LANE_W + 1.3)])]
        w.spec["labels"] = [(800, t_s, t_s + 2.5, ("swerve",))]
        return a
    spec["actors"] = actors
    return spec


PREDICT_SCENARIOS = {
    "cut_in": sc_cut_in,
    "junction_priority": sc_junction_priority,
    "ped_wait_cross": sc_ped_wait_cross,
    "curve_parked_oncoming": sc_curve_parked_oncoming,
    "pull_out": sc_pull_out,
    "queue_stop": sc_queue_stop,
    "cyclist": sc_cyclist,
}


SCENARIOS = {
    "parked_street": sc_parked_street,
    "oncoming": sc_oncoming,
    "junction": sc_junction,
    "adjacent": sc_adjacent,
    "slow_lead": sc_slow_lead,
    "curve_parked": sc_curve_parked,
    "crossing_pair": sc_crossing_pair,
    "occlusion": sc_occlusion,
    "parked_row": sc_parked_row,
    "queue": sc_queue,
    "platoon": sc_platoon,
}
CORE_SCENARIOS = ("crossing_pair", "occlusion", "parked_row", "queue", "platoon")
ALERT_SCENARIOS = {}
for _v in (14, 25, 33):
    for _m in ("brake", "lane"):
        ALERT_SCENARIOS["stopped_%s_%d" % (_m, _v)] = sc_stopped(float(_v), _m)


def sc_park_out_curve(seed, night):
    """Ego-motion hygiene (docs/EGO_MOTION.md): the car leaves a parking place, 18 s of creeping at 0.15 m/s
    through a tight turn (0.03-0.04 rad/s: under a speed sensor's floor the feed calls that a standstill), then
    the town road of `curve_parked` (a left then a right curve, R 110 m, at 13 m/s) with cars parked along it."""
    rng = random.Random(seed * 101 + 31)

    def actors(w):
        a = _parked_row(rng, 100, 30.0, 340.0, -1, lat=(2.6, 3.2), p_gap=0.1)
        a += _parked_row(rng, 200, 410.0, 690.0, +1, lat=(2.6, 3.2), p_gap=0.1)
        a += _parked_row(rng, 300, 60.0, 660.0, +1, lat=(5.5, 6.5), p_gap=0.6)
        return a
    return dict(T=75.0, speed=[(0, 0.15), (18, 0.15), (25, 13.0)],
                curv=[(0, 0.9), (4.0, 0.9), (4.5, 0.0), (90, 0.0), (90, 1 / 110.0), (310, 1 / 110.0), (310, 0.0), (410, 0.0),
                      (410, -1 / 110.0), (630, -1 / 110.0), (630, 0.0)],
                actors=actors, lead_range_m=70.0 if night else 100.0)


EGO_SCENARIOS = {"park_out_curve": sc_park_out_curve}


def scenario(name, seed=1, night=False, world=None):
    """A ScenarioWorld by name (SCENARIOS, PREDICT_SCENARIOS, EGO_SCENARIOS or ALERT_SCENARIOS). world: world
    options (e.g. {"dims_spread": True}); also read from CHRIS_SIM_DET's "world" key (explicit `world` wins)."""
    fn = SCENARIOS.get(name) or PREDICT_SCENARIOS.get(name) or EGO_SCENARIOS.get(name) or ALERT_SCENARIOS[name]
    spec = fn(seed, night)
    return ScenarioWorld(spec, seed=seed, name=name, night=night, world=world)


# ---- lane DECIDE: the driver does not react (forward-collision-warning tests) -------------------------------------
def sc_stopped_cruise(speed):
    """A stationary car in the ego lane on a straight road, 200 m+ ahead; the ego keeps its speed and never reacts.
    The run ends 0.3 s before the car's front would touch it (`collision_t`): an alert's lead time is measured to
    that time (tools/decide_replay.py)."""
    def build(seed, night):
        rng = random.Random(seed * 101 + 9 + int(speed))
        far = max(200.0, 12.0 * speed) + rng.uniform(0, 20)
        L = DIMS["car"][0]
        s_car = far + CAM_TO_AXLE + L / 2
        t_col = (far - 2.1) / speed                  # the ego's front (2.1 m ahead of the camera) at the car's rear

        def actors(w):
            return [Actor(800, "car", "stopped_in_lane", s_car, lateral=[(0, rng.gauss(0, 0.15))],
                          yaw_rel=rng.gauss(0, 0.02))]
        return dict(T=t_col - 0.3, speed=[(0, speed)], curv=[(0, 0.0)], actors=actors,
                    lead_range_m=70.0 if night else 100.0, approach=speed, mode="cruise", collision_t=t_col)
    return build


for _v in (14, 25):
    ALERT_SCENARIOS["stopped_cruise_%d" % _v] = sc_stopped_cruise(float(_v))


# ---- lane DECIDE: true conflicts in and around bends (the ego never reacts; collision_t as sc_stopped_cruise) ------
# The ego's camera starts at arc CAM_TO_AXLE and moves at the constant `v`, so its front (2.1 m ahead of the camera)
# reaches arc s at (s - 2.1 - CAM_TO_AXLE) / v.
_EGO_FRONT = 2.1
_CURVE_PARKED_CURV = [(0, 0.0), (80, 0.0), (80, 1 / 110.0), (300, 1 / 110.0), (300, 0.0), (400, 0.0),
                      (400, -1 / 110.0), (620, -1 / 110.0), (620, 0.0)]


def _t_front_at(s, v):
    return (s - _EGO_FRONT - CAM_TO_AXLE) / v


def _alert_spec(v, curv, t_col, actors, night):
    return dict(T=t_col - 0.3, speed=[(0, v)], curv=curv, actors=actors, lead_range_m=70.0 if night else 100.0,
                approach=v, mode="cruise", collision_t=t_col)


def _bend_stopped(v, curv, s_rear, row_side, k, seed, night):
    """A stationary car in the ego lane, rear at arc s_rear; a parked row on side row_side (None: none)."""
    rng = random.Random(seed * 101 + k)
    s_rear = s_rear + rng.uniform(-3.0, 3.0)
    L = DIMS["car"][0]
    s_car = s_rear + L / 2
    t_col = _t_front_at(s_rear, v)
    lat = rng.gauss(0, 0.15)
    yaw = rng.gauss(0, 0.02)

    def actors(w):
        a = [Actor(800, "car", "stopped_in_lane", s_car, lateral=[(0, lat)], yaw_rel=yaw)]
        if row_side is not None:
            a += _parked_row(rng, 100, 60.0, 330.0, row_side, lat=(2.6, 3.2), p_gap=0.1)
        return a
    return _alert_spec(v, curv, t_col, actors, night)


def sc_bend_stopped_13(seed, night):
    """Town road of `curve_parked` (straight 80 m, a LEFT curve R 110 m from arc 80 to 300, then a right one), ego
    13 m/s, never reacting. A stationary car stands IN the ego lane (lateral N(0, 0.15) m) ~75 m into the left curve:
    its rear at arc 155 +- 3 m, so it comes into view around the bend. Cars are parked on the OUTSIDE (right,
    2.6-3.2 m) from arc 60 to 330, beside and beyond it. collision_t: the ego's front at its rear (~11.6 s)."""
    return _bend_stopped(13.0, _CURVE_PARKED_CURV, 80.0 + 75.0, -1, 41, seed, night)


def sc_bend_stopped_clear_13(seed, night):
    """As bend_stopped_13 (left curve R 110 m, the stopped car's rear at arc 155 +- 3 m, in the ego lane) without any
    parked car: the stopped car alone. collision_t: the ego's front at its rear (~11.6 s)."""
    return _bend_stopped(13.0, _CURVE_PARKED_CURV, 80.0 + 75.0, None, 42, seed, night)


def sc_bend_exit_stopped_13(seed, night):
    """Straight 80 m, a LEFT curve R 110 m from arc 80 to 230, then straight; ego 13 m/s, never reacting. A
    stationary car stands in the ego lane on the straight ~15 m after the curve's END (rear at arc 245 +- 3 m), so for
    most of the approach it is seen across the inside of the bend. Parked row on the outside (right, 2.6-3.2 m) from
    arc 60 to 330. collision_t: the ego's front at its rear (~18.5 s)."""
    curv = [(0, 0.0), (80, 0.0), (80, 1 / 110.0), (230, 1 / 110.0), (230, 0.0)]
    return _bend_stopped(13.0, curv, 245.0, -1, 43, seed, night)


def sc_right_bend_stopped_13(seed, night):
    """Mirror of bend_stopped_13: straight 80 m, a RIGHT curve R 110 m from arc 80 to 300; a stationary car in the
    ego lane with its rear at arc 155 +- 3 m (~75 m into the curve); parked row on the outside = LEFT (+2.6..+3.2 m)
    from arc 60 to 330. collision_t: the ego's front at its rear (~11.6 s)."""
    curv = [(0, 0.0), (80, 0.0), (80, -1 / 110.0), (300, -1 / 110.0), (300, 0.0)]
    return _bend_stopped(13.0, curv, 80.0 + 75.0, +1, 44, seed, night)


def _half_in_lane(v, curv, s_rear, row_to, k, seed, night):
    rng = random.Random(seed * 101 + k)
    s_rear = s_rear + rng.uniform(-3.0, 3.0)
    L = DIMS["car"][0]
    s_car = s_rear + L / 2
    t_col = _t_front_at(s_rear, v)
    yaw = rng.gauss(0, 0.02)

    def actors(w):
        row = _parked_row(rng, 100, 60.0, row_to, -1, lat=(2.6, 3.2), p_gap=0.1)
        row = [a for a in row if abs(a.s0 - s_car) >= 7.0]
        # body covers lateral -2.35..-0.55: 0.33 m into the ego's swept band (-0.88..0.88)
        return row + [Actor(810, "car", "half_in_lane", s_car, lateral=[(0, -1.45)], yaw_rel=yaw)]
    return _alert_spec(v, curv, t_col, actors, night)


def sc_half_in_lane_13(seed, night):
    """The `curve_parked` road (LEFT curve R 110 m from arc 80), ego 13 m/s, never reacting; parked row on the
    outside (right, 2.6-3.2 m) from arc 60 to 330 and ONE car of that side standing half in the ego lane: centre
    lateral -1.45 m (body -2.35..-0.55, 0.33 m into the ego's swept band), rear at arc 140 +- 3 m (~60 m into the
    curve); row cars within 7 m of arc of it removed. collision_t: the ego's front at its rear (~10.4 s)."""
    return _half_in_lane(13.0, _CURVE_PARKED_CURV, 80.0 + 60.0, 330.0, 45, seed, night)


def sc_half_in_lane_straight_13(seed, night):
    """A straight road, ego 13 m/s, never reacting; parked row on the right (2.6-3.2 m) from arc 60 to 330 and one
    car half in the ego lane (centre lateral -1.45 m, body -2.35..-0.55) with its rear ~180 m ahead of the camera at
    the start (arc 181.5 +- 3 m); row cars within 7 m of arc of it removed. collision_t: the ego's front at its rear
    (~13.7 s)."""
    return _half_in_lane(13.0, [(0, 0.0)], CAM_TO_AXLE + 180.0, 330.0, 46, seed, night)


def sc_ped_between_parked_13(seed, night):
    """The `curve_parked` road (LEFT curve R 110 m from arc 80), ego 13 m/s, never reacting; parked row on the outside
    (right, 2.6-3.2 m) from arc 60 to 330 with a gap at arc 135 +- 3 m (~55 m into the curve; row cars whose footprint
    comes within 3 m of it removed). A pedestrian stands in the gap at lateral -2.9 m and steps out at 1.4 m/s,
    starting 2.6/1.4 = 1.86 s before the ego's front arrives, so it is at lateral -0.3 m (inside the ego's swept band)
    exactly when the ego's front reaches its arc (collision_t, ~10.1 s); it walks on to the left after."""
    rng = random.Random(seed * 101 + 47)
    v = 13.0
    s_p = 80.0 + 55.0 + rng.uniform(-3.0, 3.0)
    t_col = _t_front_at(s_p, v)
    vp = 1.4
    t_go = t_col - (2.9 - 0.3) / vp

    def actors(w):
        row = _parked_row(rng, 100, 60.0, 330.0, -1, lat=(2.6, 3.2), p_gap=0.1)
        row = [a for a in row if abs(a.s0 - s_p) >= a.L / 2 + 3.0]
        ped = Actor(610, "human", "crossing_ped", s_p, speed=[(0, 0.0)],
                    lateral=[(0, -2.9), (t_go, -2.9), (t_col, -0.3), (t_col + 6.0, -0.3 + 6.0 * vp)])
        return row + [ped]
    return _alert_spec(v, _CURVE_PARKED_CURV, t_col, actors, night)


ALERT_SCENARIOS.update({
    "bend_stopped_13": sc_bend_stopped_13,
    "bend_stopped_clear_13": sc_bend_stopped_clear_13,
    "bend_exit_stopped_13": sc_bend_exit_stopped_13,
    "right_bend_stopped_13": sc_right_bend_stopped_13,
    "half_in_lane_13": sc_half_in_lane_13,
    "half_in_lane_straight_13": sc_half_in_lane_straight_13,
    "ped_between_parked_13": sc_ped_between_parked_13,
})
TRUE_BEND_SCENARIOS = ("bend_stopped_13", "bend_stopped_clear_13", "bend_exit_stopped_13", "right_bend_stopped_13",
                       "half_in_lane_13", "half_in_lane_straight_13", "ped_between_parked_13")


# ---- lane SPREAD: a pedestrian at a parked row (the ego never reacts; collision_t as above) --------------------------
_CHILD = (0.35, 0.45, 1.15)
_VAN = (6.0, 2.1, 2.6)


def _ped_row(v, curv, s_ped, lat0, v_ped, k, seed, night, clear_m=3.0, vans=False, dims=None, steps=True, t_after=4.0):
    """A parked row on the right (2.6-3.2 m) from arc 60 to 330 with a pedestrian in a gap of it at arc s_ped +- 3 m,
    standing at lateral lat0 (m left of the lane's centre: negative). steps: it walks out at v_ped so that it is at
    lateral -0.3 m (inside the strip the car sweeps) exactly when the car's front reaches its arc (collision_t), and
    on to the left after; not steps: it stands there throughout (role `kerb_ped`, no collision; the run ends t_after
    seconds after the car's front has passed it). clear_m: row cars whose footprint comes within this of the
    pedestrian's arc are removed. vans: a van (6.0 x 2.1 x 2.6 m, lateral -3.0) stands on either side of the gap,
    0.9 m of arc from the pedestrian's: what stands behind their road-side faces is hidden from the car."""
    rng = random.Random(seed * 101 + k)
    s_p = s_ped + rng.uniform(-3.0, 3.0)
    t_at = _t_front_at(s_p, v)
    t_go = t_at - (abs(lat0) - 0.3) / v_ped

    def actors(w):
        row = _parked_row(rng, 100, 60.0, 330.0, -1, lat=(2.6, 3.2), p_gap=0.1)
        gap = (0.9 + _VAN[0]) if vans else 0.0
        row = [a for a in row if abs(a.s0 - s_p) >= a.L / 2 + max(clear_m, gap + (0.8 if vans else 0.0))]
        if vans:
            row += [Actor(190 + i, "truck", "parked", s_p + sg * (0.9 + _VAN[0] / 2), lateral=[(0, -3.0)],
                          yaw_rel=rng.gauss(0.0, 0.02), dims=_VAN) for i, sg in enumerate((-1, 1))]
        if steps:
            ped = Actor(610, "human", "crossing_ped", s_p, speed=[(0, 0.0)], dims=dims,
                        lateral=[(0, lat0), (t_go, lat0), (t_at, -0.3), (t_at + 6.0, -0.3 + 6.0 * v_ped)])
        else:
            ped = Actor(610, "human", "kerb_ped", s_p, speed=[(0, 0.0)], lateral=[(0, lat0)], dims=dims)
        return row + [ped]
    if steps:
        return _alert_spec(v, curv, t_at, actors, night)
    return dict(T=t_at + t_after, speed=[(0, v)], curv=curv, actors=actors, lead_range_m=70.0 if night else 100.0)


def sc_ped_early_13(seed, night):
    """A straight road, ego 13 m/s, never reacting; parked row on the right with a WIDE gap (row cars within 9 m of
    arc removed) ~135 m ahead; a pedestrian stands in it at the lane's edge (lateral -2.4 m), in view from far off,
    and steps out at 1.4 m/s 1.5 s before the car's front arrives (collision_t, ~10 s)."""
    return _ped_row(13.0, [(0, 0.0)], 135.0, -2.4, 1.4, 51, seed, night, clear_m=9.0)


def sc_ped_hidden_13(seed, night):
    """The `curve_parked` road (LEFT curve R 110 m from arc 80), ego 13 m/s, never reacting; in the parked row on the
    outside two VANS stand 1.8 m apart at arc ~135; a pedestrian stands between them at lateral -3.6 m, behind their
    road-side faces (-1.95 m), and walks out at 1.4 m/s from 2.36 s before the car's front arrives: it passes the
    vans' faces, into view, about 1.2 s before (collision_t, ~10 s)."""
    return _ped_row(13.0, _CURVE_PARKED_CURV, 135.0, -3.6, 1.4, 52, seed, night, vans=True)


def sc_child_between_parked_13(seed, night):
    """As ped_between_parked_13 (left curve, a gap in the parked row at arc ~135) with a CHILD (0.35 x 0.45 x 1.15 m)
    standing in the gap at lateral -2.9 m, which runs out at 2.2 m/s 1.18 s before the car's front arrives."""
    return _ped_row(13.0, _CURVE_PARKED_CURV, 135.0, -2.9, 2.2, 53, seed, night, dims=_CHILD)


def sc_ped_kerb_stands_13(seed, night):
    """The false-warning control of ped_between_parked_13: the same road, row and gap, the pedestrian standing in
    the gap at lateral -2.9 m and NOT stepping out; the car passes it (no collision)."""
    return _ped_row(13.0, _CURVE_PARKED_CURV, 135.0, -2.9, 1.4, 47, seed, night, steps=False)


def sc_ped_kerb_stands_early_13(seed, night):
    """The false-warning control of ped_early_13: the straight road and the wide gap, the pedestrian standing at the
    lane's edge (lateral -2.4 m) and NOT stepping out."""
    return _ped_row(13.0, [(0, 0.0)], 135.0, -2.4, 1.4, 51, seed, night, clear_m=9.0, steps=False)


ALERT_SCENARIOS.update({
    "ped_early_13": sc_ped_early_13,
    "ped_hidden_13": sc_ped_hidden_13,
    "child_between_parked_13": sc_child_between_parked_13,
    "ped_kerb_stands_13": sc_ped_kerb_stands_13,
    "ped_kerb_stands_early_13": sc_ped_kerb_stands_early_13,
})
PED_SCENARIOS = ("ped_early_13", "ped_hidden_13", "child_between_parked_13")
PED_CONTROL_SCENARIOS = ("ped_kerb_stands_13", "ped_kerb_stands_early_13")
