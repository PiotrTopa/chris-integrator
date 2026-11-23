"""Kinematics and dynamics of the objects: the state, the motion hypotheses, and what each kind of road user
may do.

State of an object, in the ground frame (ego.py), one vector for every hypothesis:

    [px, py, vx, vy, ax, ay, dp, lh] position (m), velocity (m/s), acceleration (m/s2) over the ground; dp:
                                    this object's own offset (rad) of the angle below the horizon at which a
                                    camera sees where it stands on the road: the road's slope under it and
                                    where the box's bottom edge sits on this vehicle. What all objects share
                                    of that angle error (the camera's pitch) is the sensor's bias, learnt by
                                    the tracker (tracker.py, sensors.py); lh: the logarithm of the
                                    object's own height in metres. A thing does not change its size: how tall
                                    its box is in the picture then says how far it is, and how fast the box
                                    grows says how fast it comes nearer, without the road's help.

Hypotheses (an interacting multiple model set; tracker.py mixes and weighs them):

    stand   the object stands: velocity and acceleration are zero, the position stays. The rigid one: seen from
            a moving car its bearings must keep pointing at one place.
    cv      steady velocity: disturbed by an acceleration noise as large as this kind of road user drives.
    ca      manoeuvring: an acceleration that persists for `tau` seconds (braking, pulling away, turning: a turn
            is a lateral acceleration), driven by a jerk noise.

Noise is shaped by the object's direction of travel when it has one: along it what the kind accelerates and
brakes, across it what it corners. The kind (`KIN`) also bounds the speed, sets how likely a new object of
that kind stands, and how fast it may change between standing and moving.
"""
import math

import numpy as np

from .. import native as _native

# The radar's arithmetic is the native core's (chris_native.c, the `radar` part): its only implementation (the
# package does not import without the module: chris_integrator/native)
NR = _native.radar

N = 8
PX, PY, VX, VY, AX, AY, DP, LH = range(N)
MODELS = ("stand", "cv", "ca")
STAND, CV, CA = range(3)

# kind of road user -> what it can do. a_lon / a_lat: accelerations it uses along / across its travel (m/s2),
# v_max: top speed over the ground (m/s), p_stand: prior that a newly seen one stands, v0_std: prior spread of
# its speed (m/s) when nothing is known yet; h / h_rel: how tall one is (m) and how much that varies (1 sigma, share);
# v_still: below this speed over the ground one of this kind counts as not moving; v0_lat: prior spread of its
# speed ACROSS the car's own direction of travel: road users mostly drive along the road the car is on, so a
# steady mover's unknown speed is mostly along it (the manoeuvring hypothesis keeps the full spread: crossing,
# turning). wl: width and length (m) of a vehicle: how wide its box is for its height says whether it is seen
# end-on or side-on, and a vehicle moves the way it points.
KIN = {
    # a detector's "car" is anything from a small hatchback to a minibus or a van it did not call a truck
    "car":       dict(a_lon=2.5, a_lat=2.0, v_max=60.0, p_stand=0.40, v0_std=15.0, v0_lat=1.5, h=1.60, h_rel=0.20, v_still=0.7, wl=(1.8, 4.5)),
    "van":       dict(a_lon=2.2, a_lat=1.8, v_max=50.0, p_stand=0.40, v0_std=15.0, v0_lat=1.5, h=2.10, h_rel=0.18, v_still=0.7, wl=(2.0, 5.5)),
    "truck":     dict(a_lon=1.5, a_lat=1.2, v_max=35.0, p_stand=0.35, v0_std=12.0, v0_lat=1.5, h=3.20, h_rel=0.25, v_still=0.7, wl=(2.5, 10.0)),
    "bus":       dict(a_lon=1.5, a_lat=1.2, v_max=35.0, p_stand=0.30, v0_std=12.0, v0_lat=1.5, h=3.10, h_rel=0.15, v_still=0.7, wl=(2.5, 12.0)),
    "tram":      dict(a_lon=1.2, a_lat=0.6, v_max=25.0, p_stand=0.30, v0_std=8.0, v0_lat=1.0, h=3.40, h_rel=0.10, v_still=0.7, wl=(2.5, 25.0)),
    "motorbike": dict(a_lon=4.0, a_lat=3.0, v_max=60.0, p_stand=0.20, v0_std=15.0, v0_lat=1.5, h=1.55, h_rel=0.15, v_still=0.7, wl=(0.8, 2.1)),
    "bike":      dict(a_lon=1.5, a_lat=1.5, v_max=14.0, p_stand=0.20, v0_std=5.0, v0_lat=1.5, h=1.65, h_rel=0.12, v_still=0.5),
    "human":     dict(a_lon=1.2, a_lat=1.2, v_max=7.0, p_stand=0.40, v0_std=1.8, v0_lat=1.8, h=1.70, h_rel=0.08, v_still=0.4),
    None:        dict(a_lon=2.5, a_lat=2.0, v_max=60.0, p_stand=0.40, v0_std=15.0, v0_lat=2.0, h=1.70, h_rel=0.35, v_still=0.7),   # unknown: as a car
}
VEHICLES = ("car", "van", "truck", "bus", "tram")

CFG = dict(
    ca_tau_s=2.0,                # a manoeuvre's acceleration persists this long
    cv_noise=0.35,               # cv: share of the kind's acceleration that disturbs a "steady" velocity
    ca_jerk=1.0,                 # ca: jerk noise = this x the kind's acceleration per second
    stand_drift=0.05,            # stand: (m/s) what a standing object's position may wander (the car's motion error)
    stand_v_std=0.05,            # stand: spread left on velocity / acceleration so the covariance stays regular
    dp_walk=math.radians(0.05),  # per sqrt(s): an object's angle offset changes slowly (the slope under it)
    # switching between hypotheses, per second
    r_stand_to_move=0.05,        # a standing object pulls away
    r_move_to_stand=2.0,         # a moving hypothesis whose speed is zero within what it knows: it stands
    r_move_to_stand_fast=0.002,  # one that has a speed does not stand the next moment
    r_cv_to_ca=0.25, r_ca_to_cv=0.5,
    p_ca0=0.15,                  # a new moving object manoeuvres with this share
)


def kin(label):
    return KIN.get(label, KIN[None])


# ---- what the native core reads of the model ----------------------------------------------------------------
_KIN4 = {l: (k["a_lon"], k["a_lat"], k["v_max"], k["v_still"]) for l, k in KIN.items()}
_PAR = {}


def _native_par(cfg):
    """The model's constants as the native core reads them (chris_native.c `RPar`); per cfg dict, which the
    tracker does not change after it is made."""
    e = _PAR.get(id(cfg))
    if e is None or e[0] is not cfg:
        par = (cfg["dp_walk"] ** 2, cfg["stand_drift"] ** 2, cfg["stand_v_std"] ** 2, cfg["cv_noise"], cfg["ca_jerk"],
               cfg["ca_tau_s"], 0.05 ** 2, cfg["r_stand_to_move"], cfg["r_move_to_stand"], cfg["r_move_to_stand_fast"],
               cfg["r_cv_to_ca"], cfg["r_ca_to_cv"])
        if len(_PAR) > 64:
            _PAR.clear()
        e = _PAR[id(cfg)] = (cfg, par)
    return e[1]


def size_prior(shares):
    """The prior of an object's height (lh: log metres) from what the class detector has said of it: {label: share}
    (the class vote's tallies, any scale). Each label's kind gives its height and spread; a detector that calls the
    same van `car` in one frame and `truck` in the next has not said which, so the prior is the mixture of the kinds
    by their shares, as one Gaussian (its mean and variance; the spread between the kinds included). Labels that
    are not a kind count as not known (`KIN[None]`). Returns (mean, variance)."""
    tot = sum(v for v in shares.values() if v > 0.0)
    if tot <= 0.0:
        k = KIN[None]
        return math.log(k["h"]), k["h_rel"] ** 2
    m = v2 = 0.0
    parts = []
    for lab, w in shares.items():
        if w <= 0.0:
            continue
        k = KIN.get(lab, KIN[None])
        lm = math.log(k["h"])
        parts.append((w / tot, lm, k["h_rel"] ** 2))
        m += w / tot * lm
    for w, lm, vv in parts:
        v2 += w * (vv + (lm - m) ** 2)
    return m, v2


_KINB = {}


def kinb(label):
    """What a kind can do, for an object started by a detection of it, as the native core reads it (chris_native.c
    `RKB`): a_lon, a_lat, v_max, v_still, p_stand, v0_std, v0_lat, h, h_rel, width, length (0: no direction of its
    own), 1 if the kind is known."""
    r = _KINB.get(label)
    if r is None:
        k = KIN.get(label, KIN[None])
        wl = k.get("wl", (0.0, 0.0))
        r = _KINB[label] = (k["a_lon"], k["a_lat"], k["v_max"], k["v_still"], k["p_stand"], k["v0_std"], k["v0_lat"],
                            k["h"], k["h_rel"], wl[0], wl[1], 0.0 if label is None or label not in KIN else 1.0)
    return r


# ---- one hypothesis, for tests and tools (the tracker works on its objects' rows: tracker.py) -----------------
def still(x, P, v0):
    """How much a hypothesis says "it does not move": the weight of its velocity (mean and uncertainty) inside
    the window exp(-|v|^2 / v0^2). 1 for a velocity known to be zero, near 0 for a known speed and for a
    velocity nothing is known about."""
    return NR.radar_still1(np.asarray(x, float), np.asarray(P, float), float(v0))


def transition(dt, q=(0.0, 0.0), cfg=CFG):
    """Model switching probabilities over dt (rows: from, columns: to), 3 x 3. q: how still the cv and the ca
    hypothesis are (`still`): a moving hypothesis that has found no speed hands over to "it stands", which then
    has to hold as the rigid statement it is."""
    return np.array(NR.radar_transition1(float(dt), float(q[0]), float(q[1]), _native_par(cfg)))


def predict(model, x, P, dt, label, cfg=CFG):
    """One hypothesis forward by dt (which may be slightly negative: a late measurement), with what its kind
    cannot do (faster than its top speed: brought back to the bound, the velocity's uncertainty opened).
    Returns (x, P)."""
    xo, Po = np.empty(N), np.empty((N, N))
    NR.radar_predict1(model, np.ascontiguousarray(x, float), np.ascontiguousarray(P, float), float(dt),
                      _KIN4.get(label) or _KIN4[None], _native_par(cfg), xo, Po)
    return xo, Po


def side_on(label, aspect, bearing):
    """Probability that a vehicle whose whole box is `aspect` times as wide as tall, seen at `bearing` (rad) off
    the car's own direction, points ACROSS that direction and not along it (None: the kind or the box does
    not say). A car seen from behind is about as wide as tall, one seen from the side three times as wide;
    seen from 45 degrees the two cannot be told apart, and the answer is one half."""
    k = KIN.get(label)
    if k is None or "wl" not in k or aspect is None or aspect <= 0.0:
        return None
    w, l = k["wl"]
    cb, sb = abs(math.cos(bearing)), abs(math.sin(bearing))
    along, across = (w * cb + l * sb) / k["h"], (w * sb + l * cb) / k["h"]
    la, lc = (math.log(aspect / along) / 0.25) ** 2, (math.log(aspect / across) / 0.25) ** 2
    m = min(la, lc)
    pa, pc = math.exp(-0.5 * (la - m)), math.exp(-0.5 * (lc - m))
    return pc / (pa + pc)
