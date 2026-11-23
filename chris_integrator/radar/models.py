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


# ---- the kernel -----------------------------------------------------------------------------------------------
# Everything below works on a batch: B objects (or B copies of objects taken to different times), each with its
# three hypotheses. xs: (B, 3, N), Ps: (B, 3, N, N), mu: (B, 3). The functions without `_b` are the same thing for
# one object or one hypothesis (tests, tools).
_I = np.arange(N)
_MOVE = (VX, VY, AX, AY)


def kin_arrays(labels):
    """(a_lon, a_lat, v_max, v_still) of each label, as arrays."""
    ks = [KIN.get(l, KIN[None]) for l in labels]
    return (np.array([k["a_lon"] for k in ks]), np.array([k["a_lat"] for k in ks]),
            np.array([k["v_max"] for k in ks]), np.array([k["v_still"] for k in ks]))


def still_b(x, P, v0):
    """How much a hypothesis says "it does not move": the weight of its velocity (mean and uncertainty) inside
    the window exp(-|v|^2 / v0^2). 1 for a velocity known to be zero, near 0 for a known speed and for a
    velocity nothing is known about. x: (..., N), P: (..., N, N), v0: broadcastable to the leading shape."""
    v2 = v0 * v0
    a00, a11, a01 = 2.0 * P[..., VX, VX] + v2, 2.0 * P[..., VY, VY] + v2, 2.0 * P[..., VX, VY]
    det = a00 * a11 - a01 * (2.0 * P[..., VY, VX])
    ok = det > 0.0
    dd = np.where(ok, det, 1.0)
    vx, vy = x[..., VX], x[..., VY]
    q = (a11 * vx * vx - 2.0 * a01 * vx * vy + a00 * vy * vy) / dd
    return np.where(ok, np.minimum(1.0, v2 / np.sqrt(dd) * np.exp(-np.minimum(q, 60.0))), 0.0)


def still(x, P, v0):
    return float(still_b(x, P, v0))


def transition_b(dt, q, cfg=CFG):
    """Model switching probabilities over dt (rows: from, columns: to), (B, 3, 3). q: (B, 2), how still the cv
    and the ca hypothesis are (`still`): a moving hypothesis that has found no speed hands over to "it stands",
    which then has to hold as the rigid statement it is."""
    dt = np.maximum(np.asarray(dt, float), 0.0)
    B = len(dt)
    f, r = cfg["r_move_to_stand_fast"], cfg["r_move_to_stand"]
    R = np.zeros((B, 3, 3))
    R[:, STAND, CV], R[:, STAND, CA] = cfg["r_stand_to_move"] * 0.7, cfg["r_stand_to_move"] * 0.3
    R[:, CV, STAND], R[:, CV, CA] = f + r * q[:, 0], cfg["r_cv_to_ca"]
    R[:, CA, STAND], R[:, CA, CV] = f + r * q[:, 1], cfg["r_ca_to_cv"]
    P = 1.0 - np.exp(-R * dt[:, None, None])
    off = P.sum(axis=2)
    P *= np.where(off > 0.9, 0.9 / np.where(off > 0.9, off, 1.0), 1.0)[:, :, None]
    P[:, (0, 1, 2), (0, 1, 2)] = 1.0 - P.sum(axis=2)
    return P


def transition(dt, q=(0.0, 0.0), cfg=CFG):
    return transition_b(np.array([dt], float), np.array([q], float), cfg)[0]


def _shape_b(q_lon, q_lat, x):
    """(B, 2, 2) intensity with q_lon along each object's travel and q_lat across it (isotropic mean for one
    that has no direction worth the name: slower than 1 m/s)."""
    vx, vy = x[:, VX], x[:, VY]
    sp = np.hypot(vx, vy)
    has = sp >= 1.0
    sps = np.where(has, sp, 1.0)
    c, s = vx / sps, vy / sps
    out = np.empty((len(sp), 2, 2))
    iso = 0.5 * (q_lon + q_lat)
    out[:, 0, 0] = np.where(has, c * c * q_lon + s * s * q_lat, iso)
    out[:, 1, 1] = np.where(has, s * s * q_lon + c * c * q_lat, iso)
    out[:, 0, 1] = out[:, 1, 0] = np.where(has, c * s * (q_lon - q_lat), 0.0)
    return out


def predict_b(xs, Ps, dt, a_lon, a_lat, v_max, cfg=CFG):
    """Every hypothesis of every object forward by its dt (which may be slightly negative: a late measurement).
    Returns (xs, Ps)."""
    dt = np.asarray(dt, float)
    B = len(dt)
    adt = np.abs(dt)
    F = np.zeros((B, 3, N, N))
    Q = np.zeros((B, 3, N, N))
    F[:, :, _I, _I] = 1.0
    Q[:, :, DP, DP] = (cfg["dp_walk"] ** 2 * adt)[:, None]
    # stand
    F[:, STAND, _MOVE, _MOVE] = 0.0
    Q[:, STAND, PX, PX] = Q[:, STAND, PY, PY] = (cfg["stand_drift"] ** 2) * adt
    Q[:, STAND, _MOVE, _MOVE] = cfg["stand_v_std"] ** 2
    # cv: an acceleration noise
    dt2, dt3 = adt * adt, adt ** 3
    F[:, CV, PX, VX] = F[:, CV, PY, VY] = dt
    F[:, CV, AX, AX] = F[:, CV, AY, AY] = 0.0
    qa = _shape_b((cfg["cv_noise"] * a_lon) ** 2, (cfg["cv_noise"] * a_lat) ** 2, xs[:, CV])
    Q[:, CV, 0:2, 0:2] = qa * (dt3 / 3.0)[:, None, None]
    Q[:, CV, 0:2, 2:4] = Q[:, CV, 2:4, 0:2] = qa * (dt2 / 2.0)[:, None, None]
    Q[:, CV, 2:4, 2:4] = qa * adt[:, None, None]
    Q[:, CV, AX, AX] = Q[:, CV, AY, AY] = 0.05 ** 2
    # ca: an acceleration that persists, driven by a jerk noise
    rho = np.exp(-adt / cfg["ca_tau_s"])
    F[:, CA, PX, VX] = F[:, CA, PY, VY] = dt
    F[:, CA, PX, AX] = F[:, CA, PY, AY] = 0.5 * dt * adt
    F[:, CA, VX, AX] = F[:, CA, VY, AY] = dt
    F[:, CA, AX, AX] = F[:, CA, AY, AY] = rho
    qj = _shape_b((cfg["ca_jerk"] * a_lon) ** 2, (cfg["ca_jerk"] * a_lat) ** 2, xs[:, CA])
    dt4, dt5 = adt ** 4, adt ** 5
    Q[:, CA, 0:2, 0:2] = qj * (dt5 / 20.0)[:, None, None]
    Q[:, CA, 0:2, 2:4] = Q[:, CA, 2:4, 0:2] = qj * (dt4 / 8.0)[:, None, None]
    Q[:, CA, 0:2, 4:6] = Q[:, CA, 4:6, 0:2] = qj * (dt3 / 6.0)[:, None, None]
    Q[:, CA, 2:4, 2:4] = qj * (dt3 / 3.0)[:, None, None]
    Q[:, CA, 2:4, 4:6] = Q[:, CA, 4:6, 2:4] = qj * (dt2 / 2.0)[:, None, None]
    Q[:, CA, 4:6, 4:6] = qj * adt[:, None, None]
    x1 = np.matmul(F, xs[..., None])[..., 0]
    P1 = np.matmul(np.matmul(F, Ps), F.swapaxes(-1, -2)) + Q
    # what this kind cannot do: faster than its top speed. The state is brought back to the bound and the
    # velocity's uncertainty opened, so the next measurements decide (a wrong class must not pin a wrong speed)
    sp = np.hypot(x1[:, 1:, VX], x1[:, 1:, VY])
    over = sp > v_max[:, None]
    if over.any():
        f = np.where(over, v_max[:, None] / np.where(over, sp, 1.0), 1.0)
        x1[:, 1:, VX] *= f
        x1[:, 1:, VY] *= f
        ex = np.where(over, (sp - v_max[:, None]) ** 2, 0.0)
        P1[:, 1:, VX, VX] += ex
        P1[:, 1:, VY, VY] += ex
    return x1, 0.5 * (P1 + P1.swapaxes(-1, -2))


def predict(model, x, P, dt, label, cfg=CFG):
    """One hypothesis forward by dt. Returns (x, P)."""
    xs, Ps = np.repeat(x[None, None], 3, axis=1), np.repeat(P[None, None], 3, axis=1)
    al, at, vm, _ = kin_arrays([label])
    x1, P1 = predict_b(xs, Ps, np.array([dt], float), al, at, vm, cfg)
    return x1[0, model], P1[0, model]


def combine_b(xs, Ps, mu):
    """Moment-matched single estimate of the hypotheses: (B, N), (B, N, N)."""
    x = np.einsum("bi,bin->bn", mu, xs)
    d = xs - x[:, None, :]
    P = np.einsum("bi,bimn->bmn", mu, Ps) + np.einsum("bi,bim,bin->bmn", mu, d, d)
    return x, P


def combine(xs, Ps, mu):
    if _NR is not None:                              # (the native core's combination, when in use: combine_many)
        x, P = np.empty(N), np.empty((N, N))
        try:
            _NR.radar_combine((xs,), (Ps,), (mu,), x, P)
            return x, P
        except _NATIVE_SKIP:
            pass
    x, P = combine_b(np.asarray(xs)[None], np.asarray(Ps)[None], np.asarray(mu, float)[None])
    return x[0], P[0]


def mix_b(xs, Ps, mu, PI, xf=None, Pf=None):
    """The interaction step: each hypothesis starts the interval from a blend of all of them, weighted by how
    likely the object was in each and switched. Returns (mixed xs, mixed Ps, predicted model probabilities).

    xf, Pf (B, N), (B, N, N): the object's free estimate (tracker.py). When given, it is what "it stands" hands
    to a moving hypothesis in place of its own state: "it stands" knows its velocity to be zero, but if the
    object moves after all, the best knowledge of how is what its sightings have shown without the assumption
    that it stands. (Handing over the standing state itself would pin the moving hypotheses to zero speed:
    they could never learn the speed of a slow vehicle first taken for a standing one.)"""
    c = np.einsum("bij,bi->bj", PI, mu)             # predicted probability of each model
    low = c < 1e-12
    w = PI * mu[:, :, None] / np.where(low, 1.0, c)[:, None, :]      # [b, from i, to j]
    if low.any():                                   # a hypothesis nothing flows into keeps what it has
        w = np.where(low[:, None, :], np.eye(3)[None], w)
    if xf is not None:
        SX = np.repeat(xs[:, :, None, :], 3, axis=2)             # [b, from i, to j, n]: what i hands to j
        SP = np.repeat(Ps[:, :, None, :, :], 3, axis=2)
        SX[:, STAND, CV], SX[:, STAND, CA] = xf, xf
        SP[:, STAND, CV], SP[:, STAND, CA] = Pf, Pf
        xo = np.einsum("bij,bijn->bjn", w, SX)
        d = SX - xo[:, None, :, :]
        Po = np.einsum("bij,bijmn->bjmn", w, SP) + np.einsum("bij,bijm,bijn->bjmn", w, d, d)
        return xo, Po, c
    xo = np.einsum("bij,bin->bjn", w, xs)
    d = xs[:, :, None, :] - xo[:, None, :, :]       # [b, i, j, n]
    Po = np.einsum("bij,bimn->bjmn", w, Ps) + np.einsum("bij,bijm,bijn->bjmn", w, d, d)
    return xo, Po, c


def mix(xs, Ps, mu, PI):
    xo, Po, c = mix_b(np.asarray(xs)[None], np.asarray(Ps)[None], np.asarray(mu, float)[None], np.asarray(PI)[None])
    return xo[0], Po[0], c[0]


# ---- the kernel in one call per batch ------------------------------------------------------------------------
# What the tracker asks of the model, batched: the interaction and prediction of every hypothesis of B objects
# (with their combination, or the free estimate's prediction), the combination alone, how still each object is.
# The functions above are the reference; where the native core's `radar` part is in use (chris_integrator.native)
# the same steps run in C (results to rounding: docs/NATIVE.md), else the reference runs, unchanged. The arrays
# of an object (xs, Ps, mu, xf, Pf) may come as (B, ...) arrays or as lists of B per-object arrays.
_NR = _native.radar
_NATIVE_SKIP = (_native.Fallback, TypeError, ValueError, BufferError)
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


def evolve_b(xs, Ps, mu, xf, Pf, dt, labels, cfg=CFG, combine=False, free=False):
    """The hypotheses of B objects after dt (each its own; <= 0: no switching): stillness of the moving ones
    (`still_b`), switching (`transition_b`), interaction with the free estimate's hand-over (`mix_b`), prediction
    (`predict_b`). Returns (xs, Ps, mu, x, P, xf, Pf): with `combine` (x, P) the moment-matched estimate of the
    result (`combine_b` with the predicted probabilities), with `free` (xf, Pf) the free estimate predicted by
    the steady hypothesis; None where not asked."""
    if _NR is not None:
        B = len(dt)
        out = (np.empty((B, 3, N)), np.empty((B, 3, N, N)), np.empty((B, 3)),
               np.empty((B, N)) if combine else None, np.empty((B, N, N)) if combine else None,
               np.empty((B, N)) if free else None, np.empty((B, N, N)) if free else None)
        try:
            _NR.radar_evolve(xs, Ps, mu, xf, Pf, dt, [_KIN4.get(l) or _KIN4[None] for l in labels],
                             _native_par(cfg), *out)
            return out
        except _NATIVE_SKIP:
            pass
    return _evolve_py(xs, Ps, mu, xf, Pf, dt, labels, cfg, combine, free)


def _evolve_py(xs, Ps, mu, xf, Pf, dt, labels, cfg=CFG, combine=False, free=False):
    """evolve_b in numpy: the reference."""
    xs, Ps, mu, xf, Pf = np.asarray(xs, float), np.asarray(Ps, float), np.asarray(mu, float), np.asarray(xf, float), np.asarray(Pf, float)
    dt = np.asarray(dt, float)
    a_lon, a_lat, v_max, v0 = kin_arrays(labels)
    q = still_b(xs[:, 1:], Ps[:, 1:], v0[:, None])
    PI = transition_b(dt, q, cfg)                    # dt <= 0: no switching (the identity)
    xs1, Ps1, mu1 = mix_b(xs, Ps, mu, PI, xf=xf, Pf=Pf)
    xs1, Ps1 = predict_b(xs1, Ps1, dt, a_lon, a_lat, v_max, cfg)
    x = P = xf1 = Pf1 = None
    if combine:
        x, P = combine_b(xs1, Ps1, mu1)
    if free:
        xf1, Pf1 = predict_b(np.repeat(xf[:, None], 3, axis=1), np.repeat(Pf[:, None], 3, axis=1), dt, a_lon, a_lat, v_max, cfg)
        xf1, Pf1 = xf1[:, CV], Pf1[:, CV]
    return xs1, Ps1, mu1, x, P, xf1, Pf1


def combine_many(xs, Ps, mu):
    """combine_b of B objects: (B, N), (B, N, N)."""
    if _NR is not None:
        B = len(mu)
        x, P = np.empty((B, N)), np.empty((B, N, N))
        try:
            _NR.radar_combine(xs, Ps, mu, x, P)
            return x, P
        except _NATIVE_SKIP:
            pass
    return combine_b(np.asarray(xs, float), np.asarray(Ps, float), np.asarray(mu, float))


def p_still_many(xs, Ps, mu, labels):
    """Probability that each object does not move: "it stands", and the moving hypotheses as far as they have
    found no speed (`still_b` with the kind's v_still). A list of floats."""
    if _NR is not None:
        try:
            return _NR.radar_p_still(xs, Ps, mu, [(_KIN4.get(l) or _KIN4[None])[3] for l in labels])
        except _NATIVE_SKIP:
            pass
    xs, Ps, mu = np.asarray(xs, float), np.asarray(Ps, float), np.asarray(mu, float)
    q = still_b(xs[:, 1:], Ps[:, 1:], kin_arrays(labels)[3][:, None])
    return np.minimum(1.0, mu[:, STAND] + mu[:, CV] * q[:, 0] + mu[:, CA] * q[:, 1]).tolist()


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
