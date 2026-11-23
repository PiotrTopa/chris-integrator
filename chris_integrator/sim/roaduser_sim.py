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
import math
import random

import numpy as np

from .sensors import _GM
from .world import CAM_TO_AXLE, NS, World

CALIB = {"type": "calib", "pitch": 0.02, "yaw": -0.01, "roll": 0.005, "height_m": 1.22, "focal_px": 975, "cx": 640,
         "cy": 360, "width": 1280, "height": 720, "source": "config"}

# ---- actors ---------------------------------------------------------------------------------------------------
DIMS = {"car": (4.5, 1.8, 1.5), "truck": (8.0, 2.5, 3.4), "human": (0.5, 0.6, 1.75), "post": (0.25, 0.25, 1.6)}
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


def _plan(plan, t):
    return np.interp(t, [p[0] for p in plan], [p[1] for p in plan])


class ScenarioWorld(World):
    """A World whose other road users are `Actor`s (vehicles, pedestrians, crossing traffic), with optional ego
    lane change (an S of the ego's path; the road model's plan follows the old lane until the ego starts to
    move over), a lane-based camera lead (World.lead_at uses the target's road offset; here the lane the ego is
    in), and per-actor truth in the car frame.

    spec: dict(T, speed [(t, v)], curv [(arc m, 1/m)], accel_lim, grade_amp, lane_change (s_start, length,
    shift) or None, actors: fn(world) -> [Actor] (called after the ego's motion is known), lead_range_m)."""

    def __init__(self, spec, seed=1, name="custom", night=False):
        self.name, self.night, self.spec = name, night, spec
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
)

FP_CLASSES = (("car", 0.55), ("person", 0.45))


class DetectorSim:
    """`lights` messages with `road_users` as the phone's unit sends them. Deterministic per seed.

    params: DET_DAY / DET_NIGHT overrides (see the dicts: every number is documented there and in
    docs/VISUAL_RADAR_EVAL.md). calib: the calibration the unit publishes (the stack is given this one); the true
    camera differs from it by pitch_err_deg / yaw_err_deg plus the body pitch of braking and road roughness, and
    the road ahead is not flat (World's grade profile: the actors' ground height relative to the ego's plane)."""

    def __init__(self, seed=51, night=False, params=None, calib=None, pitch_sign=None, yaw_sign=None, views=VIEWS):
        self.p = dict(DET_NIGHT if night else DET_DAY, **(params or {}))
        self.seed, self.night, self.calib, self.views = seed, night, dict(calib or CALIB), views
        rng = random.Random(seed * 7 + 3)
        self.pitch_err = math.radians(self.p["pitch_err_deg"]) * (pitch_sign if pitch_sign else rng.choice((-1, 1)))
        self.yaw_err = math.radians(self.p["yaw_err_deg"]) * (yaw_sign if yaw_sign else rng.choice((-1, 1)))
        self.log = []            # [(t, view, [(actor id | "fp:...", cls, box, clip)])]
        self._seen = {}          # actor -> [(time, true box height)] of its whole boxes within the baseline window

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
                cls = {"car": "car", "truck": "truck", "human": "person"}[o["kind"]]
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


def scenario(name, seed=1, night=False):
    """A ScenarioWorld by name (SCENARIOS or ALERT_SCENARIOS)."""
    fn = SCENARIOS.get(name) or ALERT_SCENARIOS[name]
    spec = fn(seed, night)
    return ScenarioWorld(spec, seed=seed, name=name, night=night)
