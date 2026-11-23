"""Sensors as models: what each kind of object sensor measures of an object, as a function of the object's
state in the ground frame and of where the car was at that moment.

A sensor model supplies, for one scan (one message of that sensor):

    h(x, pose)        predicted measurement and its Jacobian (m x 7) for an object state x
    residual(z, zp)   measurement minus prediction (angles wrapped)
    coverage(xc, yc)  probability that an object at this place of the CAR frame is reported in this scan
                      (0 outside the field of view / range of this scan)
    clutter           false detections per unit of measurement volume
    start(det, pose, label)   (x0, P0) of an object first seen in this detection, or None if a detection of this
                      kind cannot start one
    velocity          True if the sensor measures speed (a microwave radar's range rate)

`pose`: (X, Y, psi, vx, vy, w) of the car frame's origin on the ground at the scan's time.

Built here:

* `CameraBoxes`: a camera's class detector (core/roadusers.py turns its boxes into a bearing and, where the box
  stands on the road inside the view, a ground point). The bearing is good. The ground point is used as what
  it is, the angle below the horizon of the box's bottom edge: its error is the camera's pitch error, shared
  by all objects and learnt as the sensor's bias, plus an offset of each object's own (state `dp`), which the
  object's history resolves when its bearings move (a standing object passed by, a crossing one) and which a
  range sensor resolves at once. The box's height is a second, independent range: the object's own size seen
  from its distance (state `lh`). A box cut at the bottom of the view has neither: a bearing only.
* `MicrowaveRadar`: range, azimuth, range rate from a mount anywhere on the car. The range rate measures the
  object's velocity over the ground along the line of sight (the sensor's own motion taken out).
* `Position`: a sensor that reports a place in the car frame (another camera's object list, a lidar's).
"""
import math

import numpy as np

from .. import native as _native
from .models import DP, LH, N, PX, PY, VX, VY, kin

# The measurement functions in the native core (chris_native.c `r_h`), when its radar part is in use: the sensor
# hands its constants over as 24 numbers (`native()`), a detection its kind as a code (`kind_code`)
_NR = _native.radar
_KCODE = {"b": 0, "rb": 1, "rbh": 2, "bt": 3, "": 4}


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Sensor:
    name = "sensor"
    kind = "sensor"
    p_d = 0.9
    clutter = 1e-3
    velocity = False
    coast_s = 1.0                # an object only this sensor sees is kept this long without a detection
    angle_rows = ()              # measurement rows that are angles

    def h(self, x, pose, det=None):
        raise NotImplementedError

    def rows(self, det):
        """Which measurement rows of a detection are angles."""
        return self.angle_rows

    def residual(self, z, zp, det=None):
        d = np.asarray(z, float) - zp
        for i in self.rows(det):
            d[i] = wrap(d[i])
        return d

    def takes(self, det, xc, yc):
        """May this detection belong to an object at this place of the car frame (beyond the statistical gate)?"""
        return True

    def density(self, det):
        """Detections per unit of this detection's measurement space that do NOT come from a known object:
        false ones and objects seen for the first time. What a detection's fit to an object is weighed against."""
        return self.clutter

    def coverage(self, xc, yc):
        return self.p_d

    def p_detect(self, xc, yc, x=None):
        """Probability that an object (state x, at this place of the car frame) is reported in this scan:
        `coverage`, less where the sensor has a reason to miss this very object."""
        return self.coverage(xc, yc)

    def start(self, det, pose, label):
        return None


def kind_code(kind):
    """A detection kind as the native core takes it: the base kind (b 0, rb 1, rbh 2, bt 3, none 4; anything
    else is measured as rbh, like `_h` does) + 8 for the scale-rate row."""
    if not kind:
        return 2
    if kind.endswith("s"):
        return _KCODE.get(kind[:-1], 2) + 8
    return _KCODE.get(kind, 2) if kind else 2


def base(kind):
    """A detection kind without the mark of an extra scale-rate row ("rbhs" -> "rbh")."""
    return kind[:-1] if kind and kind.endswith("s") else kind


def _rel(x, pose):
    dx, dy = x[PX] - pose[0], x[PY] - pose[1]
    r = max(math.hypot(dx, dy), 0.5)
    return dx, dy, r


def _prior(label, cfg):
    """Velocity / acceleration / scale part of a new object's covariance: what its kind may be doing."""
    k = kin(label)
    P = np.zeros((N, N))
    P[VX, VX] = P[VY, VY] = k["v0_std"] ** 2
    P[4, 4] = P[5, 5] = (0.6 * k["a_lon"]) ** 2
    P[DP, DP] = math.radians(cfg.get("bias_std0_deg", 0.6)) ** 2 + math.radians(cfg.get("dp_obj_std_deg", 0.2)) ** 2
    P[LH, LH] = k["h_rel"] ** 2
    return P


class CameraBoxes(Sensor):
    """A camera's boxes, one scan per message. A detection is one of three kinds (`det["kind"]`):

        "rbh"   z = [angle below the horizon, bearing, log box height]   the box stands on the road inside the view
                                                                          and is whole from top to bottom
        "rb"    z = [angle below the horizon, bearing]                    ... but cut at the top
        "b"     z = [bearing]                                             cut at the bottom (nearer than the view's
                                                                          first ground point): a direction only
        "bt"    z = [bearing, angle below the horizon of the TOP edge]    cut at the bottom, the top inside the view:
                                                                          a thing of known height seen from near;
                                                                          how high its top stands says how far it is

    Where the box stands on the road is measured as what the camera sees: the angle below the horizon of the
    box's bottom edge, atan(camera height / distance). Its errors live in that angle: the bottom edge's noise
    from frame to frame (in `R`), the camera's pitch error, the same for every object (the sensor's `bias`, learnt
    by the tracker from the objects themselves: tracker.py), and each object's own offset (state `dp`: the road's
    slope under it, where the edge sits on this vehicle). As a range all of that is an error that grows with
    the square of the distance, and that changes as the car comes nearer: a standing car read through a wrong
    pitch looks like one that drives, unless the angle is what is modelled.

    The box height is the object's own size seen from its distance: log h_px = log f + lh - log depth, depth
    along the camera's axis. The object's size `lh` is a state: unknown within what its kind allows at first
    (a car's height varies by an eighth, a person's less), then learnt, and from then on the box says how far
    the object is and how fast it comes nearer, independent of the road's slope and the camera's pitch."""
    name = "camera"
    kind = "camera_boxes"
    coast_s = 2.0                # the detector misses in runs
    has_bias = True

    def __init__(self, cover=None, cfg=None, focal_px=975.0, height_m=None, h_min_px=None):
        """cover: {"bearing_min", "bearing_max" (rad, left positive), "ground_min_m"} of this scan's view, or
        None (unknown field: everything ahead counts as coverable)."""
        c = self.cfg = dict(CAMERA_CFG, **(cfg or {}))
        self.cover = cover
        self.logf = math.log(focal_px)
        self.hc = float(height_m or c["cam_height_m"])
        # the smallest box this view's detector reports, as its boxes have shown. A view that has not shown a
        # box yet has shown nothing about what it can box: a missing box then says next to nothing
        # (h_min_px = inf, set by box.py). None: this feed gives no boxes at all, sizes play no part
        self.h_min = None if h_min_px is None else float(h_min_px)
        self.p_d = c["p_d"]
        self.bias = 0.0                                   # the camera's pitch error (rad) as known now, and how
        self.bias_var = math.radians(c["bias_std0_deg"]) ** 2    # well (set by the tracker before every scan)
        self.dp_obj_var = math.radians(c["dp_obj_std_deg"]) ** 2
        span = max((cover["bearing_max"] - cover["bearing_min"]) if cover else 1.0, 0.1)
        gmin = cover["ground_min_m"] if cover else 12.0
        # the measurement space of one scan: bearings x ground angles x (log) box heights
        self.vol = {"b": span, "rb": span * max(math.atan2(self.hc, gmin) - math.atan2(self.hc, c["max_range_m"]), 0.01)}
        self.vol["rbh"] = self.vol["rb"] * c["logh_span"]
        self.vol["bt"] = span * 0.5
        self.extraneous = c["false_per_scan"] + c["new_per_scan"]     # per scan; the tracker raises it in a busy scene
        self.clutter = self.extraneous / self.vol["rb"]

    def density(self, det):
        k = base(det["kind"])
        d = self.extraneous / self.vol[k]
        if k != det["kind"]:
            d /= self.cfg["scale_rate_span"]             # one more row: scale rates of unknown things, anywhere
        if k != "rbh":
            return d
        # objects are mostly born where they become detectable: as small boxes. A part appears anywhere (from
        # behind another, from the side): uniform over the heights
        c = self.cfg
        u = float(det["z"][2]) - (math.log(self.h_min) if self.h_min and math.isfinite(self.h_min) else float(det["z"][2]))
        born = math.exp(-max(u, 0.0) / c["born_small_scale"]) / c["born_small_scale"]
        return d * c["logh_span"] * ((1.0 - c["born_small"]) / c["logh_span"] + c["born_small"] * born)

    def p_detect(self, xc, yc, x=None):
        p = self.coverage(xc, yc)
        if p > 0.0 and self.cover is not None and math.hypot(xc, yc) < self.cover["ground_min_m"]:
            p *= self.cfg["p_cut"]                        # nearer than the view's first ground point: its box
                                                          # is cut there, and a cut box is often not reported
        if p <= 0.0 or x is None or self.h_min is None:
            return p
        # a detector does not report boxes below a size, and is unsure just above it
        h = math.exp(self.logf + x[LH]) / max(xc, 0.5)
        ramp = (h / self.h_min - 1.0) / (self.cfg["h_full_rel"] - 1.0)
        return p * min(1.0, max(self.cfg["p_small"], ramp))

    def rows(self, det):
        k = base(det["kind"]) if det else "rb"
        return (0,) if k in ("b", "bt") else () if k == "" else (1,)

    def _scale_row(self, x, pose, det):
        """How fast the object's image grows: the detector's measurement on the pixels of two frames
        `sdt` apart, (ln size now - ln size then) / sdt. A thing of fixed size seen from depth d along the
        camera's axis has size ~ 1 / d, so this is ln(d_then / d_now) / sdt, with d_then = d_now - d' sdt for the
        relative motion now: no size, no road, no pitch in it. For a standing thing it is the car's own speed
        over the distance: a range. Returns (value, Jacobian row)."""
        dx, dy = x[PX] - pose[0], x[PY] - pose[1]
        c, s_ = math.cos(pose[2]), math.sin(pose[2])
        depth = max(c * dx + s_ * dy, 0.5)
        w = pose[5] if len(pose) > 5 else 0.0
        rate = c * (x[VX] - pose[3]) + s_ * (x[VY] - pose[4]) + w * (-s_ * dx + c * dy)     # d(depth)/dt
        dt = max(float(det.get("sdt") or 0.0), 1e-3)
        u = max(1.0 - rate * dt / depth, 0.05)
        row = np.zeros(N)
        row[VX], row[VY] = -c / (u * depth), -s_ / (u * depth)
        row[PX] = (w * s_ / depth + rate * c / (depth * depth)) / u
        row[PY] = (-w * c / depth + rate * s_ / (depth * depth)) / u
        return math.log(u) / dt, row

    def native(self):
        """The constants the native core's measurement, coverage and density functions read (24 numbers)."""
        c, cv = self.cfg, self.cover
        v = self.vol
        return (0.0, self.hc, self.logf, self.bias, 1.0 if cv else 0.0, cv["bearing_min"] if cv else 0.0,
                cv["bearing_max"] if cv else 0.0, cv["ground_min_m"] if cv else 0.0, c["max_range_m"], self.p_d,
                c["p_cut"], math.nan if self.h_min is None else self.h_min, c["h_full_rel"], c["p_small"],
                c["near_margin"], self.extraneous, v["b"], v["rb"], v["rbh"], v["bt"], c["scale_rate_span"],
                c["logh_span"], c["born_small"], c["born_small_scale"])

    def h(self, x, pose, det=None):
        if _NR is not None:
            z, H = np.empty(4), np.empty((4, N))
            m = _NR.radar_h(self.native(), kind_code(det["kind"] if det else "rb"),
                            float((det.get("sdt") if det else 0.0) or 0.0), x, pose, z, H)
            return z[:m], H[:m]
        return self.h_py(x, pose, det)

    def h_py(self, x, pose, det=None):
        """h in Python: the reference."""
        k = det["kind"] if det else "rb"
        if k.endswith("s"):                               # the base kind's rows, then the scale rate
            zs, rs = self._scale_row(x, pose, det)
            if k == "s":
                return np.array([zs]), rs.reshape(1, N)
            z0, H0 = self._h(x, pose, k[:-1])
            return np.append(z0, zs), np.vstack([H0, rs])
        return self._h(x, pose, k)

    def _h(self, x, pose, k):
        dx, dy, r = _rel(x, pose)
        brow = np.zeros(N)
        brow[PX], brow[PY] = -dy / (r * r), dx / (r * r)
        b = wrap(math.atan2(dy, dx) - pose[2])
        if k == "b":
            return np.array([b]), brow.reshape(1, N)
        hc = self.hc
        if k == "bt":
            ho = math.exp(x[LH])
            dh = hc - ho                                  # the top below (+) or above (-) the camera
            q = r * r + dh * dh
            trow = np.zeros(N)
            trow[PX], trow[PY], trow[LH] = -dh / q * dx / r, -dh / q * dy / r, -ho * r / q
            return np.array([b, math.atan2(dh, r) + self.bias]), np.array([brow, trow])
        g = -hc / (r * r + hc * hc)                       # d(angle) / d(distance)
        arow = np.zeros(N)
        arow[PX], arow[PY], arow[DP] = g * dx / r, g * dy / r, 1.0
        ang = math.atan2(hc, r) + self.bias + x[DP]
        if k == "rb":
            return np.array([ang, b]), np.array([arow, brow])
        c, s_ = math.cos(pose[2]), math.sin(pose[2])
        depth = max(c * dx + s_ * dy, 0.5)                # along the camera's axis
        hrow = np.zeros(N)
        hrow[PX], hrow[PY], hrow[LH] = -c / depth, -s_ / depth, 1.0
        return np.array([ang, b, self.logf + x[LH] - math.log(depth)]), np.array([arow, brow, hrow])

    def in_field(self, xc, yc):
        if xc <= 0.5:
            return False
        if self.cover is None:
            return True
        return self.cover["bearing_min"] <= math.atan2(yc, xc) <= self.cover["bearing_max"]

    def coverage(self, xc, yc):
        if not self.in_field(xc, yc) or math.hypot(xc, yc) > self.cfg["max_range_m"]:
            return 0.0
        return self.p_d

    def takes(self, det, xc, yc):
        if base(det["kind"]) not in ("b", "bt") or self.cover is None:
            return True
        # a bearing alone fits everything on its ray: it holds only an object near enough to be cut at the bottom
        return math.hypot(xc, yc) <= self.cfg["near_margin"] * self.cover["ground_min_m"]

    def _start_top(self, det, pose, label):
        """An object first seen in a box cut at the bottom: its feet are out of the picture, its top is not.
        For a kind whose height is known well and differs enough from the camera's own (a person, a lorry) the
        top edge says how far it is; for a car, about as tall as the camera sits, it says nothing and the box
        starts no object."""
        k = kin(label)
        c = self.cfg
        ho = k["h"]
        dh = self.hc - ho
        if label is None or abs(dh) < c["top_start_rel"] * k["h_rel"] * ho:
            return None
        a = float(det["z"][1]) - self.bias
        if a * dh <= 0.0 or abs(math.tan(a)) < 1e-4:
            return None
        r = dh / math.tan(a)
        if not 1.0 < r <= c["max_range_m"]:
            return None
        b = float(det["z"][0])
        dr_dh = -ho / math.tan(a)                         # range per (log) height
        dr_da = -dh / (math.sin(a) ** 2)                  # range per angle
        var_r = dr_dh ** 2 * k["h_rel"] ** 2 + dr_da ** 2 * (det["R"][1] + self.bias_var)
        if var_r > (c["top_start_rel_range"] * r) ** 2:
            return None
        x0 = np.zeros(N)
        x0[LH] = math.log(ho)
        P0 = _prior(label, c)
        P0[DP, DP] = self.bias_var + self.dp_obj_var
        ang = b + pose[2]
        ca, sa = math.cos(ang), math.sin(ang)
        x0[PX], x0[PY] = pose[0] + r * ca, pose[1] + r * sa
        U = np.array([[ca, -sa], [sa, ca]])
        P0[:2, :2] = U @ np.diag([var_r, det["R"][0] * r * r + 0.05]) @ U.T
        c_lh = dr_dh * k["h_rel"] ** 2
        P0[PX, LH] = P0[LH, PX] = ca * c_lh
        P0[PY, LH] = P0[LH, PY] = sa * c_lh
        return x0, P0

    def start(self, det, pose, label):
        kd = base(det["kind"])
        if kd == "bt":
            return self._start_top(det, pose, label)
        if kd in ("b", ""):
            return None
        k = kin(label)
        hc = self.hc
        b = float(det["z"][1])
        x0 = np.zeros(N)
        x0[LH] = math.log(k["h"])
        P0 = _prior(label, self.cfg)
        p_dp = self.bias_var + self.dp_obj_var            # what is not known of this object's angle offset
        P0[DP, DP] = p_dp
        # the road's range: the angle, the known part of its error taken out; never beyond the sensor's reach
        ang = max(float(det["z"][0]) - self.bias, math.atan2(hc, self.cfg["max_range_m"]))
        rg = hc / math.tan(ang)
        J = (rg * rg + hc * hc) / hc                      # distance per angle
        var_g = J * J * (det["R"][0] + p_dp)
        r, var_r, wg, rh = rg, var_g, 1.0, 0.0
        if kd == "rbh":
            # a second range, independent of the first: the object's own size
            rh = math.exp(self.logf + x0[LH] - float(det["z"][2])) / max(math.cos(b), 0.3)
            var_h = rh * rh * (k["h_rel"] ** 2 + det["R"][2])
            wg = var_h / (var_g + var_h)
            r, var_r = wg * rg + (1.0 - wg) * rh, var_g * var_h / (var_g + var_h)
        a = b + pose[2]
        ca, sa = math.cos(a), math.sin(a)
        x0[PX], x0[PY] = pose[0] + r * ca, pose[1] + r * sa
        U = np.array([[ca, -sa], [sa, ca]])
        P0[:2, :2] = U @ np.diag([var_r, det["R"][1] * r * r + 0.05]) @ U.T
        # the place came from the angle and from the size: its error along the line of sight is tied to what is
        # not known of either (without these terms the object would not know that learning one fixes the other)
        c_dp = wg * J * p_dp
        c_lh = (1.0 - wg) * rh * k["h_rel"] ** 2
        P0[PX, DP] = P0[DP, PX] = ca * c_dp
        P0[PY, DP] = P0[DP, PY] = sa * c_dp
        P0[PX, LH] = P0[LH, PX] = ca * c_lh
        P0[PY, LH] = P0[LH, PY] = sa * c_lh
        return x0, P0


CAMERA_CFG = dict(
    p_d=0.55,                    # share of frames a visible road user is boxed (night drives: about half)
    false_per_scan=0.05,         # false boxes per scan over the field
    new_per_scan=0.15,           # boxes of objects not seen before, per scan, in a quiet scene
    logh_span=3.0,               # box heights span this much (log): 8 px to 160 px
    born_small=0.7, born_small_scale=0.5,     # share of new objects that appear at the smallest box size (log scale)
    h_full_rel=2.0,              # a box this many times the smallest is detected as well as any
    p_small=0.1,                 # ... and one at the limit this much of that
    top_start_rel=2.5, top_start_rel_range=0.45,   # a cut box starts an object when its kind's height differs from
                                 # the camera's by this many of its own spreads, and the range comes out this well
    p_cut=0.4,
    use_scale_rate=True, scale_rate_floor=0.01,   # (1/s) the detector's stated spread is not believed below this
    scale_rate_span=1.5,         # (1/s) scale rates of things not known yet lie within about this much
    max_range_m=80.0,
    near_margin=1.25,            # the bearing-only zone reaches this far beyond the view's first ground point
    cam_height_m=1.22,           # until the calibration message says
    # the bottom edge's angle below the horizon
    bias_std0_deg=0.6,           # the camera's pitch error before anything is learnt (the online calibration's spread)
    bias_tau_s=2.0,              # what the ripe objects agree on moves into the sensor's bias this fast
    bias_walk_deg=0.03,          # per sqrt(s): the pitch drifts (load, braking, the calibration settling)
    bias_floor_deg=0.05,
    dp_obj_std_deg=0.2,          # each object's own offset on top (slope under it, where the edge sits)
    bottom_px=3.0, bottom_rel=0.04,   # the bottom edge from frame to frame: px + share of the box height
    pitch_jitter_deg=0.1,        # the body's pitch from frame to frame
    # the box's centre and height
    bearing_rel=0.11, bearing_floor_deg=0.15,   # centre: share of the box width + floor
    height_std_px=3.6, height_std_rel=0.05,     # height: px + share of it (top and bottom edge together)
)


class MicrowaveRadar(Sensor):
    """z = [range, azimuth, range rate] in the sensor's frame; mount (x, y) in the car frame and yaw."""
    name = "radar"
    kind = "microwave_radar"
    angle_rows = (1,)
    velocity = True
    p_d = 0.9
    coast_s = 1.0

    def __init__(self, mount=(0.0, 0.0, 0.0), fov=math.radians(50.0), max_range=160.0, sigma=(0.25, math.radians(0.5), 0.15),
                 false_per_scan=2.0):
        self.mx, self.my, self.yaw = mount
        self.fov, self.max_range, self.sigma = fov, max_range, sigma
        self.clutter = false_per_scan / (max_range * 2.0 * fov * 60.0)

    def native(self):
        return (1.0, self.mx, self.my, self.yaw, self.fov, self.max_range, self.p_d, self.clutter) + (0.0,) * 16

    def _sensor(self, pose):
        X, Y, psi, vx, vy, w = pose
        c, s = math.cos(psi), math.sin(psi)
        ox, oy = c * self.mx - s * self.my, s * self.mx + c * self.my          # the mount's lever on the ground
        return X + ox, Y + oy, vx - w * oy, vy + w * ox

    def h(self, x, pose, det=None):
        if _NR is not None:
            z, H = np.empty(4), np.empty((4, N))
            _NR.radar_h(self.native(), 0, 0.0, x, pose, z, H)
            return z[:3], H[:3]
        return self.h_py(x, pose, det)

    def h_py(self, x, pose, det=None):
        """h in Python: the reference."""
        sx, sy, svx, svy = self._sensor(pose)
        dx, dy = x[PX] - sx, x[PY] - sy
        r = max(math.hypot(dx, dy), 0.5)
        rvx, rvy = x[VX] - svx, x[VY] - svy
        rr = (dx * rvx + dy * rvy) / r
        H = np.zeros((3, N))
        H[0, PX], H[0, PY] = dx / r, dy / r
        H[1, PX], H[1, PY] = -dy / (r * r), dx / (r * r)
        H[2, PX], H[2, PY] = rvx / r - rr * dx / (r * r), rvy / r - rr * dy / (r * r)
        H[2, VX], H[2, VY] = dx / r, dy / r
        return np.array([r, wrap(math.atan2(dy, dx) - pose[2] - self.yaw), rr]), H

    def coverage(self, xc, yc):
        dx, dy = xc - self.mx, yc - self.my
        r = math.hypot(dx, dy)
        if r < 0.5 or r > self.max_range or abs(wrap(math.atan2(dy, dx) - self.yaw)) > self.fov:
            return 0.0
        return self.p_d

    def start(self, det, pose, label):
        r, az, rr = (float(v) for v in det["z"])
        sx, sy, svx, svy = self._sensor(pose)
        a = az + self.yaw + pose[2]
        ca, sa = math.cos(a), math.sin(a)
        x0 = np.zeros(N)
        x0[PX], x0[PY] = sx + r * ca, sy + r * sa
        vr = rr + ca * svx + sa * svy                    # the object's own speed along the line of sight
        x0[VX], x0[VY] = vr * ca, vr * sa
        P0 = _prior(label, {})
        U = np.array([[ca, -sa], [sa, ca]])
        P0[:2, :2] = U @ np.diag([det["R"][0] + 0.1, det["R"][1] * r * r + 0.1]) @ U.T
        # along the line of sight the speed is measured; across it nothing is known
        k = kin(label)
        P0[2:4, 2:4] = U @ np.diag([4.0 * det["R"][2] + 0.5, k["v0_std"] ** 2]) @ U.T
        x0[LH] = math.log(k["h"])
        return x0, P0


class Position(Sensor):
    """z = [x, y] in the car frame (an object list of another camera or of a lidar)."""
    name = "position"
    kind = "position"
    p_d = 0.85

    def __init__(self, fov=math.radians(60.0), max_range=70.0, false_per_scan=0.3):
        self.fov, self.max_range = fov, max_range
        self.clutter = false_per_scan / (max_range * max_range)

    def h(self, x, pose, det=None):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        dx, dy = x[PX] - pose[0], x[PY] - pose[1]
        H = np.zeros((2, N))
        H[0, PX], H[0, PY] = c, s
        H[1, PX], H[1, PY] = -s, c
        return np.array([c * dx + s * dy, -s * dx + c * dy]), H

    def coverage(self, xc, yc):
        r = math.hypot(xc, yc)
        return self.p_d if 1.0 < r < self.max_range and abs(math.atan2(yc, xc)) < self.fov else 0.0

    def start(self, det, pose, label):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        zx, zy = float(det["z"][0]), float(det["z"][1])
        x0 = np.zeros(N)
        x0[PX], x0[PY] = pose[0] + c * zx - s * zy, pose[1] + s * zx + c * zy
        P0 = _prior(label, {})
        x0[LH] = math.log(kin(label)["h"])
        Rm = np.array([[c, -s], [s, c]])
        R = np.diag(det["R"]) if np.ndim(det["R"]) == 1 else np.asarray(det["R"], float)
        P0[:2, :2] = Rm @ R @ Rm.T
        return x0, P0
