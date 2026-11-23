"""The ego state and its process model (docs/DESIGN.md section 2).

Core (kinematic) states, in this order:
    PE, PN   position east / north of the local origin, m
    PSI      heading, rad counter-clockwise from east
    V, A     speed along the heading (signed) and its rate
    W, AL    yaw rate (positive left) and its rate
    Z, TH    altitude and road grade (rad)
    SV       scale of the speed state against the ground: V is the speed as the car's own speed signal scales it
             (what odometry-minded consumers want, known to centimetres per second); the distance over ground is
             V / (1 + SV). SV is observable only against absolute sources (positions over distance, a radar's
             stationary world); without one it keeps its prior.
Sources append their own calibration states (`AuxState`): random walk or first-order Gauss-Markov.
"""
import math

import numpy as np

from ..native import core as _C

_empty = np.empty

PE, PN, PSI, V, A, W, AL, Z, TH, SV = range(10)
NCORE = 10
CORE_NAMES = ("pe", "pn", "psi", "v", "a", "w", "al", "z", "th", "sv")
G = 9.80665

DEFAULT_PROCESS = dict(
    jerk_psd=2.0,                # (m/s3)^2 s: how fast the longitudinal acceleration changes
    yaw_jerk_psd=1.0,            # (rad/s3)^2 s: how fast the yaw acceleration changes
    pos_psd=0.01,                # m^2 / s: position noise not explained by speed and heading (slip)
    alt_psd=0.01,                # m^2 / s
    grade_psd=4e-5,              # rad^2 / s: grade changes by about 0.6 % in 1 s (1 sigma)
    scale_psd=1e-9,              # 1 / s: the speed scale is a tyre property
    scale_sigma0=0.02,           # prior of the speed scale
)
# before any measurement: nothing is known but that it is a car
X0 = np.zeros(NCORE)
P0_DIAG = np.array([1e-4, 1e-4, 1e-6, 1e4, 9.0, 1.0, 1.0, 1e4, 0.01, 0.02 ** 2])


_CAPS = ((V, 60.0 ** 2), (A, 5.0 ** 2), (W, 0.6 ** 2), (AL, 2.0 ** 2), (PSI, math.pi ** 2))


class AuxState:
    """A calibration state a source asks the estimator to carry.

    name    unique within the source ("tau", "s_v", ...); the state's full name is "<source id>.<name>"
    x0, sigma0   prior
    psd     random walk: variance added per second          (tau_s None)
    tau_s   Gauss-Markov: correlation time; `sigma0` is then also its stationary sigma
    lo, hi  optional clamp of the estimate
    """
    __slots__ = ("name", "x0", "sigma0", "psd", "tau_s", "lo", "hi", "unit", "doc")

    def __init__(self, name, x0=0.0, sigma0=1.0, psd=0.0, tau_s=None, lo=None, hi=None, unit="", doc=""):
        self.name, self.x0, self.sigma0, self.psd, self.tau_s = name, x0, sigma0, psd, tau_s
        self.lo, self.hi, self.unit, self.doc = lo, hi, unit, doc


class EgoModel:
    """Process model over [core | aux...]."""

    def __init__(self, process=None):
        self.c = dict(DEFAULT_PROCESS, **(process or {}))
        self.aux = []                      # (full name, AuxState)
        self.index = {n: i for i, n in enumerate(CORE_NAMES)}
        self.n = NCORE
        self._psd = np.zeros(0)
        self._gm = []                      # (index, tau_s, sigma)
        self._clamp = []
        self.heading_free = True           # False: the heading does not move the position yet (not initialised)
        self._par = None                   # the native core's parameter block (made on first use)

    def add_aux(self, owner, aux):
        name = "%s.%s" % (owner, aux.name)
        if name in self.index:
            return self.index[name]
        i = self.n
        self.index[name] = i
        self.aux.append((name, aux))
        self.n += 1
        self._psd = np.append(self._psd, 0.0 if aux.tau_s else aux.psd)
        self._par = None
        if aux.tau_s:
            self._gm.append((i, aux.tau_s, aux.sigma0))
        if aux.lo is not None or aux.hi is not None:
            self._clamp.append((i, aux.lo, aux.hi))
        return i

    def set_clamp(self, i, lo=None, hi=None):
        """The bounds of one calibration state, changed at run time (a source whose prior was replaced: another
        receiver has another delay, and the bounds of the old one would clip it)."""
        self._par = None
        self._clamp = [c for c in self._clamp if c[0] != i]
        if lo is not None or hi is not None:
            self._clamp.append((i, lo, hi))

    def initial(self):
        x = np.concatenate([X0, [a.x0 for _, a in self.aux]])
        p0 = P0_DIAG.copy()
        p0[SV] = self.c["scale_sigma0"] ** 2
        P = np.diag(np.concatenate([p0, [a.sigma0 ** 2 for _, a in self.aux]]))
        return x, P

    def rate(self, x):
        """dx/dt of the core states (for time-delay Jacobians)."""
        c, s = math.cos(x[PSI]), math.sin(x[PSI])
        vg = x[V] / (1.0 + x[SV])
        return (vg * c, vg * s, x[W], x[A], 0.0, x[AL], 0.0, vg * math.sin(x[TH]), 0.0, 0.0)

    def _native_par(self):
        """The model's constants as the native core reads them (chris_native.c `ekf_predict`)."""
        c = self.c
        nan = float("nan")
        par = [c["jerk_psd"], c["yaw_jerk_psd"], c["pos_psd"], c["alt_psd"], c["grade_psd"], c["scale_psd"]]
        par += [lim for _, lim in _CAPS]
        par += [float(len(self._clamp)), float(len(self._gm))]
        par += [float(v) for v in self._psd]
        for i, lo, hi in self._clamp:
            par += [float(i), nan if lo is None else float(lo), nan if hi is None else float(hi)]
        for i, tau, sig in self._gm:
            par += [float(i), float(tau), float(sig)]
        self._par = np.array(par, dtype=float)
        return self._par

    def predict(self, x, P, dt):
        """(x, P, F) after dt (the native core: chris_native.c `ekf_predict`): the kinematic states moved over dt
        (speed by the acceleration, heading by the yaw rate and its rate, position along the mean heading and
        altitude by the grade, both with the speed scaled to the ground), a random walk or a Gauss-Markov decay
        for each calibration state, the process noise (white jerk and yaw jerk integrated over dt, position noise
        along the heading), after a gap of more than 0.5 s the caps of the bounded quantities' variance, and the
        clamps of the calibration states. F = (Fc, rho): the kinematic block of the transition and the
        Gauss-Markov states' decay over dt (None without such states)."""
        n = self.n
        if x.shape[0] != n:
            raise ValueError("predict: a state of %d, the model has %d" % (x.shape[0], n))
        rho = _empty(n) if self._gm else None
        x1, P1, Fc = _empty(n), _empty((n, n)), _empty((NCORE, NCORE))
        par = self._par if self._par is not None else self._native_par()
        try:
            _C.ekf_predict(x, P, dt, par, rho, x1, P1, Fc)
        except (TypeError, ValueError, BufferError):      # (a view, or another dtype: made a contiguous array)
            _C.ekf_predict(np.ascontiguousarray(x, dtype=float), np.ascontiguousarray(P, dtype=float), dt, par,
                           rho, x1, P1, Fc)
        return x1, P1, (Fc, rho)

    def rts(self, Pp, F, Pa, xa, xp, xs, Ps, wrap_idx):
        """One Rauch-Tung-Striebel step back over a prediction (the native core: chris_native.c `ekf_rts`):
        C = solve(Pp, F Pa)^T, (xa + C (xs - xp), Pa + C (Ps - Pp) C^T), the differences of the angle states in
        `wrap_idx` wrapped; False when Pp is singular."""
        n = xa.shape[0]
        xo, Po = np.empty(n), np.empty((n, n))
        args = (Pp, F[0], F[1], Pa, xa, xp, xs, Ps)
        try:
            ok = _C.ekf_rts(*args, wrap_idx, xo, Po)
        except (TypeError, ValueError, BufferError):
            ok = _C.ekf_rts(*(None if a is None else np.ascontiguousarray(a, dtype=float) for a in args), wrap_idx, xo, Po)
        return (xo, Po) if ok else False


# ---- geodesy (local tangent plane; fine over a drive) -------------------------------------------------
R_EARTH = 6378137.0


def to_local(origin, lat, lon):
    lat0, lon0 = origin
    return (math.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0)), math.radians(lat - lat0) * R_EARTH)


def to_geo(origin, e, n):
    lat0, lon0 = origin
    return (lat0 + math.degrees(n / R_EARTH), lon0 + math.degrees(e / (R_EARTH * math.cos(math.radians(lat0)))))


def course_to_psi(course_deg):
    return math.radians(90.0 - course_deg)


def psi_to_course(psi):
    return (90.0 - math.degrees(psi)) % 360.0
