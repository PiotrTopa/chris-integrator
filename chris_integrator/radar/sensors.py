"""Sensors as models: what each kind of object sensor measures of an object, as a function of the object's
state in the ground frame and of where the car was at that moment.

A sensor model supplies, for one scan (one message of that sensor):

    native()          its constants as the native core reads them (24 numbers): there (chris_native.c `r_h`,
                      `r_coverage`, `r_p_detect`, `r_density`) are its measurement function, its coverage (the
                      probability that an object at a place of the CAR frame is reported in this scan, 0 outside the
                      field of view / range), the probability of reporting a given object and the density of
                      detections that are not a known object's; the native core is their only implementation
    h(x, pose, det)   predicted measurement and its Jacobian (m x N) for an object state x (the native function)
    residual(z, zp)   measurement minus prediction (angles wrapped)
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

from .models import DP, LH, NR as _NR, N, PX, PY, VX, VY, kin

# a detection's kind as the native core takes it (`kind_code`)
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

    def native(self):
        raise NotImplementedError

    def h(self, x, pose, det=None):
        z, H = np.empty(4), np.empty((4, N))
        m = _NR.radar_h(self.native(), kind_code((det or {}).get("kind")), float((det or {}).get("sdt") or 0.0),
                        np.ascontiguousarray(x, float), [float(v) for v in pose] + [0.0] * (6 - len(pose)), z, H)
        return z[:m], H[:m]

    def rows(self, det):
        """Which measurement rows of a detection are angles."""
        return self.angle_rows

    def residual(self, z, zp, det=None):
        d = np.asarray(z, float) - zp
        for i in self.rows(det):
            d[i] = wrap(d[i])
        return d



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

    def rows(self, det):
        k = base(det["kind"]) if det else "rb"
        return (0,) if k in ("b", "bt") else () if k == "" else (1,)

    def native(self):
        """The constants the native core's measurement, coverage and density functions read (24 numbers)."""
        c, cv = self.cfg, self.cover
        v = self.vol
        return (0.0, self.hc, self.logf, self.bias, 1.0 if cv else 0.0, cv["bearing_min"] if cv else 0.0,
                cv["bearing_max"] if cv else 0.0, cv["ground_min_m"] if cv else 0.0, c["max_range_m"], self.p_d,
                c["p_cut"], math.nan if self.h_min is None else self.h_min, c["h_full_rel"], c["p_small"],
                c["near_margin"], self.extraneous, v["b"], v["rb"], v["rbh"], v["bt"], c["scale_rate_span"],
                c["logh_span"], c["born_small"], c["born_small_scale"], self.bias_var, self.dp_obj_var,
                math.radians(c["pitch_jitter_deg"]) ** 2, math.radians(c["bias_floor_deg"]) ** 2, c["top_start_rel"],
                c["top_start_rel_range"])

    def h(self, x, pose, det=None):
        return Sensor.h(self, x, pose, det or {"kind": "rb"})


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
        return (1.0, self.mx, self.my, self.yaw, self.fov, self.max_range, self.p_d, self.clutter) + (0.0,) * 22


class Position(Sensor):
    """z = [x, y] in the car frame (an object list of another camera or of a lidar)."""
    name = "position"
    kind = "position"
    p_d = 0.85

    def __init__(self, fov=math.radians(60.0), max_range=70.0, false_per_scan=0.3):
        self.fov, self.max_range = fov, max_range
        self.clutter = false_per_scan / (max_range * max_range)

    def native(self):
        return (2.0, self.fov, self.max_range, self.p_d, self.clutter) + (0.0,) * 25
