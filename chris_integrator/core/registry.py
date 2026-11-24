"""What a source is (docs/DESIGN.md section 7): an input description, a measurement model, and its mount
and calibration states. The core knows nothing else about any sensor."""
import math

import numpy as np

NS = 1e9


def fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


_NUM = frozenset((int, float))


def all_fin(seq):
    """all(fin(v) for v in seq), without a Python-level call per element for lists of plain ints and floats
    (what a parsed JSON array holds); anything else is judged element by element as `fin` does."""
    if set(map(type, seq)) <= _NUM:
        return all(map(math.isfinite, seq))
    return all(fin(v) for v in seq)


class Mount:
    """Where a sensor sits, in the car frame (x forward, y left, origin the reference camera), and where its
    boresight points (yaw, rad, positive left)."""
    __slots__ = ("x", "y", "z", "yaw")

    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.x, self.y, self.z, self.yaw = float(x), float(y), float(z), float(yaw)

    def as_dict(self):
        return {"x_m": self.x, "y_m": self.y, "z_m": self.z, "yaw_deg": round(math.degrees(self.yaw), 3)}


class Meas:
    """One scalar ego measurement, as a measurement model returns it.

    q       quantity tag. Rows of one tag share anchoring and NIS accounting; "speed" and "yaw_rate" (and any tag
            with `mon`) are also compared raw against the other sources' by the pair monitors.
    t       time of validity, s on the device's monotonic clock (the source subtracts its estimated delay)
    z, sigma    value and 1-sigma noise
    h       the row's model (role "fuse"): native, a tuple (kind, indices, parameters) made by `lin`, `scaled`,
            ... below (the filter pass evaluates it: nothing of a late row runs in Python when the pass is run
            again), or a callable h(x) -> (prediction, indices, coefficients) for a model the core does not have
    role    "fuse" | "monitor" (not fused; monitors only) | "self" (z = stated / characterised sigma) |
            "flag" (q "standstill", z bool) | "signal" (q "signal:<name>", pass-through)
    robust  heavy-tailed source: Student-t weighting and a hard gate while an exact source anchors `q`
    wrap    the innovation is an angle
    mon     (value, characterised sigma) the monitors judge the source by; None: z and sigma
    cb      cb(nu, s0, t): called once with the row's first innovation (adaptive noise)
    local   the row corrects the kinematic state and its own source's calibration states only; other sources'
            calibration states are "consider" states for it (their uncertainty counts, they are not corrected).
            For a source without an absolute scale of its own (a monocular camera's speed must not teach the
            wheel-speed scale).
    consider  indices of further states this row must not correct (their uncertainty still counts): a state
            that this row alone cannot tell from another one (a camera's scale without a speed reference)
    level   a standstill flag's standing (role "flag", z true): None or 2 = the car is at rest (the exact
            standstill rows); 1 = the source's signals read zero and nothing has confirmed the rest yet (they read
            zero below a floor: the speed is zero to that floor, the turn is not known to be zero)
    """
    __slots__ = ("q", "t", "z", "sigma", "h", "src", "role", "robust", "wrap", "mon", "cb", "local", "consider", "level")

    def __init__(self, q, t, z, sigma=None, h=None, src=None, role="fuse", robust=False, wrap=False, mon=None,
                 cb=None, local=False, consider=None, level=None):
        self.q, self.t, self.z, self.sigma, self.h, self.src, self.role = q, t, z, sigma, h, src, role
        self.robust, self.wrap, self.mon, self.cb, self.local, self.consider = robust, wrap, mon, cb, local, consider
        self.level = level


# The native models of a row (chris_native.c / egomotion.c `eh_eval`, where each is written out): (kind, idx, par).
EH_LIN, EH_SCALE, EH_ACCEL, EH_PRODUCT, EH_VSPEED, EH_HEADING = range(6)
_NOPAR = np.zeros(0)


def _ix(idx):
    return np.array(idx, dtype=np.intp)


def lin(idx, coef, const=0.0):
    """A linear row: prediction = const + sum(coef * x[idx])."""
    return EH_LIN, _ix(idx), np.array([const] + list(coef), dtype=float)


def scaled(i, i_scale, var, floor, i_bias=None, delay=0.0, i_rate=None):
    """(1 + x[i_scale]) x[i] (+ x[i_bias]): a quantity as a sensor with a scale (and a bias) reads it. The scale's
    regressor is x[i] shrunk by its own uncertainty `var` (the note in sources/vehicle.py, dead zone `floor`).
    `delay` s (not 0): the sensor's value is that old at the time the row is inserted, (1 + x[i_scale])
    (x[i] - delay x[i_rate]): the quantity as it was then, by its rate state (docs/EGO_MOTION.md)."""
    idx = [i, i_scale] if i_bias is None else [i, i_scale, i_bias]
    if not delay:
        return EH_SCALE, _ix(idx), np.array([var, floor], dtype=float)
    return EH_SCALE, _ix(idx + [i_rate]), np.array([var, floor, delay, 0.0 if i_bias is None else 1.0], dtype=float)


def accel_grade(i_a, i_th, i_bias, g):
    """x[i_a] + g sin(x[i_th]) + x[i_bias]: a longitudinal accelerometer on a grade."""
    return EH_ACCEL, _ix([i_a, i_th, i_bias]), np.array([g], dtype=float)


def product_bias(i, j, i_bias):
    """x[i] x[j] + x[i_bias]: a lateral accelerometer (speed x yaw rate)."""
    return EH_PRODUCT, _ix([i, j, i_bias]), _NOPAR


def delayed_speed(i_v, i_scale, i_m, i_mf, i_tau, tau0, var, floor, cs=1.0, cf=1.0):
    """(1 + x[i_scale]) (x[i_v] - a (x[i_tau] - tau0)) + cs x[i_m] + cf x[i_mf]: a speed with a scale, a delay
    (the row was inserted at the delay tau0) and two coloured errors; a is the acceleration state."""
    return EH_VSPEED, _ix([i_v, i_scale, i_m, i_mf, i_tau]), np.array([tau0, var, floor, cs, cf], dtype=float)


def delayed_heading(i_psi, i_tau, tau0):
    """x[i_psi] - w (x[i_tau] - tau0): a heading with a delay; w is the yaw-rate state."""
    return EH_HEADING, _ix([i_psi, i_tau]), np.array([tau0], dtype=float)


class Source:
    """Base class of a source. Subclass, set the description, override `measure` and / or `scan`."""
    id = None                # unique name; the owner of its calibration states
    group = None             # health / isolation unit (several sources may form one group)
    rank = 0                 # isolation by rank: the higher rank is believed when no vote decides
    family = None            # groups of one family share physical sensors: one vote between them
    types = ()               # message types it reads
    stale_s = 1.0
    witness = False          # never fused, never starts a fault
    counts = True            # False: a side channel (its messages do not show the group alive)
    info = None
    aux = ()                 # AuxState list: calibration states the estimator carries for it
    mount = None
    lags = ()                # time offsets of its rows to be estimated outside the filter (docs/EGO_MOTION.md):
                             # (name, reference signal, prior s, lo, hi, least rate of the reference, its noise)

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        if self.group is None:
            self.group = self.id
        self.ix = {}             # aux name -> state index (set at registration)

    def measure(self, msg, ctx):
        """Ego measurement rows of a message (list of Meas)."""
        return []

    def scan(self, msg, ctx):
        """Object detections of a message (objects.Scan) or None."""
        return None

    def report(self, ctx):
        """Extra fields for the group's health entry."""
        return {}

    def calib_state(self, ctx):
        """Extra fields for `calibration.<id>` (e.g. a state word)."""
        return {}

    def reset(self):
        """A new epoch: forget what is tied to the old time axis."""


class Ctx:
    """What a measurement model may read: the current estimate (never raw data of other sources)."""

    def __init__(self, integ, src):
        self.integ, self.src = integ, src
        self.mem = {}

    @property
    def x(self):
        return self.integ.x_now

    @property
    def P(self):
        return self.integ.P_now

    @property
    def now(self):
        return self.integ.now

    def ix(self, name):
        """Index of a core state ("v") or of one of this source's calibration states ("tau")."""
        i = self.src.ix.get(name)
        return self.integ.model.index[name] if i is None else i

    def val(self, name):
        return float(self.integ.x_now[self.ix(name)])

    def std(self, name):
        i = self.ix(name)
        return math.sqrt(max(float(self.integ.P_now[i, i]), 0.0))

    def rate(self):
        return self.integ.model.rate(self.integ.x_now)

    def faulted(self, group=None):
        g = self.integ.groups.get(group or self.src.group)
        return g is not None and g.fault is not None

    def lag(self, name):
        """(the estimator of this source's time offset `name`, the offset to insert its rows with now), or
        (None, None) where it is not estimated (docs/EGO_MOTION.md)."""
        return self.integ.lag_of(self.src.id + "." + name)

    def speed_known(self):
        """(speed, std) of the integrator's own estimate, or None while it has no speed measurement."""
        return self.integ.speed_now()

    def ego_at(self, t):
        return self.integ.ego_at(t)

    @property
    def cfg(self):
        return self.integ.cfg
