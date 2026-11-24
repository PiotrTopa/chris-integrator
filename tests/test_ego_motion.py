"""Ego-motion hygiene (docs/EGO_MOTION.md): the standstill confirmed by several sources, the gyro's bias, the time
offsets estimated outside the filter. The numerics are the native core's (`Stand`, `Lag`); the numbers below are
frozen from it."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from chris_integrator import Integrator  # noqa: E402
from chris_integrator.core import ego as E  # noqa: E402
from chris_integrator.native import core  # noqa: E402
from chris_integrator.sim import World  # noqa: E402
from chris_integrator.sim.sensors import GnssSim, MapSim, VehicleSim, VisionSim  # noqa: E402

NAN = float("nan")
STAND = (0.5, 2.0, 1.0, 0.0095, 0.03, 0.25, 3.0, 0.15, 0.005, 0.6, 0.02, 0.6, 1.0, 0.006, 10.0)


def stand_run(gyro, flag=True, accel=(0.0, 0.0), cam=None, n=30, bias=(0.0, 0.0003), lsb=0.0044):
    """States of a Stand fed n bus samples at 10 Hz (gyro: a function of the time) and camera poses at 12 Hz."""
    s = core.Stand(*STAND)
    out = []
    tc = 0.0
    for k in range(n):
        t = 100.0 + 0.1 * k
        while cam is not None and tc <= 0.1 * k - 0.1:           # (poses arrive a tenth of a second late)
            s.cam(100.0 + tc, *cam(tc))
            tc += 1.0 / 12.0
        g = gyro(0.1 * k)
        g = g if g != g else lsb * round(g / lsb)
        out.append(s.bus(t, flag if isinstance(flag, bool) else flag(0.1 * k), g, accel[0], accel[1], bias[0], bias[1]))
    return s, out


class StandTest(unittest.TestCase):
    def test_inertial_confirms(self):
        # a gyro toggling one quantum about zero, quiet accelerometers: confirmed once the window is full
        s, st = stand_run(lambda t: 0.0044 * (int(t * 10) % 2))
        self.assertEqual(st[:7], [1] * 7)
        self.assertEqual(st[7:], [2] * 23)
        self.assertEqual(s.evidence & 1, 1)
        self.assertEqual(s.counts, (30, 7, 23, 1))
        self.assertAlmostEqual(s.gyro_mean, 0.0024, places=9)

    def test_turning_under_the_floor(self):
        # the bus says standstill, the gyro reads a turn (the reference car: up to 0.136 rad/s under that flag)
        s, st = stand_run(lambda t: 0.05)
        self.assertEqual(st, [1] * 30)
        self.assertEqual(s.evidence & 2, 2)
        # ... a turn that changes: the span speaks
        s, st = stand_run(lambda t: 0.004 * t * 10)
        self.assertEqual(st, [1] * 30)

    def test_slow_steady_turn_or_changed_bias(self):
        # quiet, but not where the bias is believed to be (a slow steady turn under the floor, or a bias that has
        # changed): only after the gyro has been that quiet for 10 s
        s, st = stand_run(lambda t: 0.0132, n=120)
        self.assertEqual(st[:107], [1] * 107)
        self.assertEqual(st[107:], [2] * 13)
        self.assertEqual(s.evidence & 64, 64)
        # ... the camera's quiet does not shorten that (it read no turn under the floor while the gyro did)
        s, st = stand_run(lambda t: 0.0132, cam=lambda t: (0.02, 0.001))
        self.assertEqual(st, [1] * 30)
        # ... and never while the camera sees the turn
        s, st = stand_run(lambda t: 0.0132, cam=lambda t: (0.1, 0.03), n=120)
        self.assertEqual(st, [1] * 120)
        self.assertEqual(s.evidence & 8, 8)
        # where the bias is believed to be (within 3 sigma of an uncertain one): at once
        s, st = stand_run(lambda t: 0.0132, bias=(0.0, 0.01))
        self.assertEqual(st[7:], [2] * 23)
        # a gyro neither quiet nor clearly moving (the car comes to rest): not yet
        s, st = stand_run(lambda t: 0.0132 if int(t * 10) % 4 else 0.0)
        self.assertEqual(st, [1] * 30)
        # a turn that fades (0.028 rad/s down to 0.009 over 20 s) is quiet at every moment and never about one mean
        s, st = stand_run(lambda t: 0.028 - 0.00095 * t, n=200)
        self.assertEqual(st, [1] * 200)

    def test_camera_overrules_a_quiet_gyro_only_by_turning(self):
        # a camera at night reports speeds of a standing car: no word against
        s, st = stand_run(lambda t: 0.0, cam=lambda t: (1.5, 0.001))
        self.assertEqual(st[7:], [2] * 23)
        s, st = stand_run(lambda t: 0.0, cam=lambda t: (0.0, 0.05))
        self.assertEqual(st, [1] * 30)

    def test_no_witness(self):
        # a feed without inertial signals, no camera: the flag alone, after 2 s
        s, st = stand_run(lambda t: NAN, accel=(NAN, NAN))
        self.assertEqual(st[:20], [1] * 20)
        self.assertEqual(st[20:], [2] * 10)
        self.assertEqual(s.evidence & 16, 16)
        # no gyro, quiet accelerometers: their quiet is what there is
        s, st = stand_run(lambda t: NAN)
        self.assertEqual(st[7:], [2] * 23)
        self.assertEqual(s.evidence & 1, 1)

    def test_flag_ends(self):
        s, st = stand_run(lambda t: 0.0, flag=lambda t: t < 1.45 or t > 2.0)
        self.assertEqual(st[:15], [1] * 7 + [2] * 8)
        self.assertEqual(st[15:21], [0] * 6)
        self.assertEqual(st[21:26], [1] * 5)              # the count starts anew
        self.assertEqual(st[26:], [2] * 4)


def lag_run(true_offset, applied=0.0, prior=0.0, n=3000, noise=0.004, seed=3, amp=0.25, period=7.0, row_noise=0.006):
    """A Lag fed a reference yaw rate (10 Hz, noisy) and rows (12 Hz) that are `true_offset` s late and were
    inserted with `applied`: the innovation of such a row is the signal at (t - true + applied) less the
    reference at t. Returns the Lag."""
    import random
    rng = random.Random(seed)
    lag = core.Lag(prior, 0.05, -0.3, 0.4, 300.0, 0.25, 0.05, 3.0, 2.5, 4.0, 0.004)

    def w(t):                                  # turns: a slow sine with straights in between
        ph = (t % (3 * period)) / period
        return amp * math.sin(2 * math.pi * ph) if ph < 1.0 else 0.0
    ev = sorted([(100.0 + 0.1 * k + 0.004, 0, k) for k in range(n)] + [(100.0 + k / 12.0 + 0.11, 1, k) for k in range(int(n * 1.2))])
    for _, kind, k in ev:
        if kind == 0:
            t = 100.0 + 0.1 * k
            lag.ref(t, w(t) + rng.gauss(0, noise))
        else:
            ts = 100.0 + k / 12.0                 # the row's stamp; valid at ts - true_offset, inserted at ts - applied
            z = w(ts - true_offset) + rng.gauss(0, row_noise)
            lag.offer(ts - applied, z - w(ts - applied), row_noise ** 2 + 1e-5, applied)
    return lag


class LagTest(unittest.TestCase):
    def test_recovers_an_offset(self):
        for true, applied, prior in ((0.08, 0.0, 0.0), (-0.06, 0.0, 0.0), (0.125, 0.125, 0.125), (0.17, 0.125, 0.125)):
            lag = lag_run(true, applied, prior)
            self.assertLess(abs(lag.value - true), 0.008, (true, lag.value))
            self.assertLess(lag.std, 0.004)
            self.assertGreater(lag.info, 0.99)
            self.assertGreater(lag.n, 300)

    def test_frozen(self):
        lag = lag_run(0.08)
        self.assertAlmostEqual(lag.value, FROZEN["lag_value"], places=9)
        self.assertAlmostEqual(lag.std, FROZEN["lag_std"], places=9)
        self.assertEqual(lag.counts, FROZEN["lag_counts"])

    def test_no_excitation_keeps_the_configured_value(self):
        lag = lag_run(0.08, amp=0.0, prior=0.02)
        self.assertEqual(lag.n, 0)
        self.assertEqual(lag.value, 0.02)
        self.assertAlmostEqual(lag.std, 0.05, places=12)
        self.assertEqual(lag.info, 0.0)
        self.assertGreater(lag.counts[1], 3000)           # gated: no excitation

    def test_bounds_and_seed(self):
        lag = lag_run(0.6)                                # beyond the bound: clamped
        self.assertEqual(lag.value, 0.4)
        lag = core.Lag(0.0, 0.05, -0.3, 0.3, 300.0, 0.25, 0.05, 3.0, 2.5, 4.0, 0.004)
        lag.seed(0.07, 0.01)                              # what was learnt before is this run's prior
        self.assertEqual(lag.value, 0.07)
        self.assertAlmostEqual(lag.std, 0.01, places=12)
        with self.assertRaises(ValueError):
            core.Lag(0.0, 0.0, -0.3, 0.3, 300.0, 0.25, 0.05, 3.0, 2.5, 4.0, 0.004)

    def test_outliers_and_a_gap_of_the_reference(self):
        import random
        rng = random.Random(1)
        lag = core.Lag(0.0, 0.05, -0.3, 0.3, 300.0, 0.25, 0.05, 3.0, 2.5, 4.0, 0.004)
        for k in range(40):                               # the reference with a hole: rows inside it are dropped
            if 10 <= k < 20:
                continue
            lag.ref(100.0 + 0.1 * k, 0.3 * 0.1 * k)
            lag.offer(100.0 + 0.1 * k - 0.3, 5.0 * rng.choice((-1, 1)), 1e-4, 0.0)
        n, gated, dropped, waiting = lag.counts
        self.assertGreater(dropped, 3)
        self.assertLess(abs(lag.value), 0.02)             # wild rows move it by a bounded amount


class RowModelTest(unittest.TestCase):
    """The rows' native measurement functions (egomotion.c `eh_eval`) against their expressions, to the last bit."""

    def setUp(self):
        import random
        rng = random.Random(4)
        self.xs = [np.array([rng.uniform(-3, 3) for _ in range(20)]) for _ in range(50)]

    def check(self, model, expr):
        kind, idx, par = model
        for x in self.xs:
            pred, coef = core.ekf_h(kind, idx, par, x)
            ep, ec = expr(x)
            self.assertEqual(pred, ep)
            self.assertEqual(coef, [float(c) for c in ec])

    @staticmethod
    def reg(v, var, floor):
        a, lim = v * v, max(9.0 * var, floor * floor)
        return 0.0 if a <= lim else v * (1.0 - lim / a)

    def test_kinds(self):
        from chris_integrator.core import registry as R
        V, A, W, AL, TH, PSI = E.V, E.A, E.W, E.AL, E.TH, E.PSI
        reg = self.reg
        self.check(R.lin([11, 3, 14], [0.5, -2.0, 3.0], 0.25),
                   lambda x: (0.25 + (((0.0 + 0.5 * x[11]) + -2.0 * x[3]) + 3.0 * x[14]), [0.5, -2.0, 3.0]))
        self.check(R.lin([V], [1.0]), lambda x: (x[V], [1.0]))
        self.check(R.scaled(W, 12, 0.04, 0.02, 13),
                   lambda x: ((1.0 + x[12]) * x[W] + x[13], [1.0 + x[12], reg(x[W], 0.04, 0.02), 1.0]))
        self.check(R.scaled(W, 12, 0.5, 0.02), lambda x: ((1.0 + x[12]) * x[W], [1.0 + x[12], reg(x[W], 0.5, 0.02)]))
        self.check(R.scaled(W, 12, 0.04, 0.02, 13, 0.06, AL),
                   lambda x: ((1.0 + x[12]) * (x[W] - 0.06 * x[AL]) + x[13],
                              [1.0 + x[12], reg(x[W], 0.04, 0.02), 1.0, -(1.0 + x[12]) * 0.06]))
        self.check(R.scaled(W, 12, 0.04, 0.02, None, -0.03, AL),
                   lambda x: ((1.0 + x[12]) * (x[W] - -0.03 * x[AL]), [1.0 + x[12], reg(x[W], 0.04, 0.02), -(1.0 + x[12]) * -0.03]))
        self.check(R.accel_grade(A, TH, 15, E.G),
                   lambda x: (x[A] + E.G * math.sin(x[TH]) + x[15], [1.0, E.G * math.cos(x[TH]), 1.0]))
        self.check(R.product_bias(V, W, 16), lambda x: (x[V] * x[W] + x[16], [x[W], x[V], 1.0]))
        self.check(R.delayed_speed(V, 10, 11, 12, 13, 0.125, 0.3, 1.0),
                   lambda x: ((1.0 + x[10]) * (x[V] - x[A] * (x[13] - 0.125)) + 1.0 * x[11] + 1.0 * x[12],
                              [1.0 + x[10], reg(x[V], 0.3, 1.0), 1.0, 1.0, -(1.0 + x[10]) * x[A]]))
        self.check(R.delayed_heading(PSI, 17, 2.0), lambda x: (x[PSI] - x[W] * (x[17] - 2.0), [1.0, -x[W]]))

    def test_refused(self):
        from chris_integrator.core import registry as R
        x = self.xs[0]
        with self.assertRaises(ValueError):                  # a model that does not fit its kind
            core.ekf_h(R.EH_ACCEL, np.array([1, 2], dtype=np.intp), np.array([9.8]), x)
        with self.assertRaises(ValueError):
            core.ekf_h(9, np.array([1], dtype=np.intp), np.array([0.0, 1.0]), x)
        with self.assertRaises(IndexError):
            core.ekf_h(R.EH_LIN, np.array([40], dtype=np.intp), np.array([0.0, 1.0]), x)

    def test_pass_takes_both(self):
        # a row with a native model and one with a callable give the same pass
        from chris_integrator.core.ekf import Filter, Row
        from chris_integrator.core.registry import lin

        def run(h):
            m = E.EgoModel()
            x0, P0 = m.initial()
            x0[E.V] = 5.0
            f = Filter(m, x0, P0)
            for k in range(5):
                f.insert(10.0 + 0.1 * k, Row("speed", 5.0 + 0.1 * k, 0.01, h, "s", "g"))
            return f.latest()
        idx, one = np.array([E.V], dtype=np.intp), np.array([1.0])
        a, b = run(lin([E.V], [1.0])), run(lambda x: (x[E.V], idx, one))
        self.assertTrue(np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]))


# ---- the estimator on simulated drives ----------------------------------------------------------------------------
def manoeuvre_world(tight=True):
    """A drive with a parking manoeuvre: 8 m/s, then 18 s of creeping at 0.15 m/s through a tight turn (R 5 m:
    0.03 rad/s, under the speed sensor's floor; `tight` False: R 17 m, 0.009 rad/s: a turn the gyro's quantum
    barely shows), a real standstill of 13 s, and away."""
    speed = [(0, 0.0), (4, 8.0), (20, 8.0), (26, 0.15), (44, 0.15), (45, 0.0), (60, 0.0), (64, 8.0), (110, 8.0)]
    w0 = World(seed=5, scenario=dict(T=110.0, speed=speed, curv=[(0, 0.0), (5000, 0.0)]), targets=None, statics=False)
    s0, s1 = float(np.interp(27.0, w0.t - w0.t0, w0.s)), float(np.interp(46.0, w0.t - w0.t0, w0.s))
    if tight:        # (the world smooths its curvature over 15 m: the tight turn begins before the creeping does)
        s0 = float(np.interp(25.3, w0.t - w0.t0, w0.s))
        curv = [(0, 0.0), (s0 - 0.5, 0.0), (s0, 0.45), (s1 + 1.0, 0.45), (s1 + 1.5, 0.0), (5000, 0.0)]
    else:
        curv = [(0, 0.0), (s0 - 0.6, 0.0), (s0, 0.2), (s1 + 0.3, 0.2), (s1 + 0.9, 0.0), (5000, 0.0)]
    w = World(seed=5, scenario=dict(T=110.0, speed=speed, curv=curv), targets=None, statics=False)
    rest = (w.t - w.t0 > 46.5) & (w.t - w.t0 < 59.5)       # (the world's speed only approaches zero: make it rest)
    w.v[rest] = 0.0
    w.w[rest] = 0.0
    w.a[rest] = 0.0
    return w


def drive(w, cfg=None, vehicle=None, vision=None, sensors=None, probe=None, restore=None):
    """A world through the estimator. probe(I, t rel, msg) after every message. restore: calibration carried over
    from an earlier drive ({state: [value, variance]}). Returns the Integrator."""
    veh = dict(seed=11, speed_floor=0.21, gyro_raw=True, gyro_lsb=0.0044, accel_sigma_rest=0.03)
    veh.update(vehicle or {})
    I = Integrator(dict({"gnss_absolute": True}, **(cfg or {})))
    if restore:
        I.calibration_restore(restore)
    sens = sensors if sensors is not None else [VehicleSim(**veh), VisionSim(leads=False, **(vision or {})), GnssSim(), MapSim()]
    for arr, m in w.messages(sens):
        if m["type"] == "road":
            I.frame(m)
        else:
            I.on_message(m)
        if probe is not None:
            probe(I, m["ts_mono_ns"] / 1e9 - w.t0, m)
    return I


class StandDriveTest(unittest.TestCase):
    """The creeping turn under the speed sensor's floor: with the bus's flag alone the turn is taught to the
    gyro's bias and the heading misses it; confirmed by the inertial sensors it is not."""

    @classmethod
    def setUpClass(cls):
        cls.w = manoeuvre_world()
        cls.res = {}
        for name, on in (("off", False), ("on", True)):
            log = []

            def probe(I, t, m, log=log):
                if m["type"] == "ego":
                    ib = I.model.index["vehicle.b_g"]
                    log.append((t, float(I.x_now[ib]), math.sqrt(float(I.P_now[ib, ib])), float(I.x_now[E.W]), float(I.x_now[E.PSI]),
                                bool(m["standstill"])))
            I = drive(cls.w, {"ego_motion": {"stand_confirm": on}}, vehicle=dict(gyro_bias=0.004), probe=probe)
            cls.res[name] = (I, np.array(log))

    def at(self, name, t):
        L = self.res[name][1]
        return L[np.searchsorted(L[:, 0], t)]

    def test_the_flag_alone_teaches_the_turn_to_the_bias(self):
        r = self.at("off", 44.0)                          # the end of the creeping turn
        self.assertGreater(r[1] - 0.004, 0.010)           # the bias is wrong by more than 0.01 rad/s ...
        self.assertLess(r[2], 0.001)                      # ... and stated to better than 0.001
        self.assertLess(abs(r[3]), 1e-3)                  # the yaw rate was held at zero through the turn
        self.assertAlmostEqual(r[1], FROZEN["off_bias_44"], places=6)

    def test_confirmed_standstill_keeps_the_bias(self):
        r = self.at("on", 44.0)
        self.assertLess(abs(r[1] - 0.004), 0.0015)
        self.assertGreater(r[3], 0.02)                    # the turn is in the yaw rate
        self.assertAlmostEqual(r[1], FROZEN["on_bias_44"], places=6)
        self.assertAlmostEqual(r[3], FROZEN["on_w_44"], places=6)
        r = self.at("on", 59.0)                           # after the real standstill: the bias as the gyro reads it at rest
        self.assertLess(abs(r[1] - 0.004), 0.001)
        self.assertAlmostEqual(r[1], FROZEN["on_bias_59"], places=6)

    def test_heading_through_the_turn(self):
        # the heading the car turned by between 26 s and 45 s, against the truth
        tr = self.w.truth_at(self.w.t0 + 45.0)["psi"] - self.w.truth_at(self.w.t0 + 26.0)["psi"]
        off = self.at("off", 45.0)[4] - self.at("off", 26.0)[4]
        on = self.at("on", 45.0)[4] - self.at("on", 26.0)[4]
        self.assertGreater(tr, 1.0)
        self.assertGreater(abs(off - tr), 0.7)            # most of the turn is missing
        self.assertLess(abs(on - tr), 0.08)
        self.assertAlmostEqual(on - tr, FROZEN["on_heading_err"], places=5)
        self.assertAlmostEqual(off - tr, FROZEN["off_heading_err"], places=5)

    def test_report(self):
        I = self.res["on"][0]
        c = I.state(int((self.w.t0 + 110.0) * 1e9))["calibration"]["vehicle"]
        self.assertEqual(c["stand"], FROZEN["report_stand"])
        self.assertGreater(c["stand"]["claimed_only_share"], 0.5)        # the creeping turn was never confirmed
        self.assertEqual(c["gyro_bias"], FROZEN["report_gyro_bias"])
        self.assertTrue(c["gyro_bias"]["straight"])
        self.assertGreater(c["gyro_bias"]["since_standstill_s"], 45.0)
        self.assertNotIn("stand", self.res["off"][0].state(int((self.w.t0 + 110.0) * 1e9))["calibration"]["vehicle"])
        # the camera was not "handled" (its pose moved under a standstill the bus only claimed)
        self.assertEqual(I.state(int((self.w.t0 + 110.0) * 1e9))["calibration"]["vision"]["handled"], 0)
        self.assertEqual(self.res["off"][0].state(int((self.w.t0 + 110.0) * 1e9))["calibration"]["vision"]["handled"], 1)


class BiasStepTest(unittest.TestCase):
    """A bias that has changed since it was learnt (turn-on, temperature): the car arrives at a standstill with a
    bias state of 0 +- 0.0005 rad/s and a gyro that reads 0.02 at rest. Quiet, but not where the bias is believed
    to be: after 10 s of that quiet the standstill is confirmed, the state's uncertainty is opened and the
    standstill places it."""

    @classmethod
    def setUpClass(cls):
        speed = [(0, 8.0), (10, 8.0), (14, 0.0), (40, 0.0), (44, 8.0), (60, 8.0)]
        w = World(seed=9, scenario=dict(T=60.0, speed=speed, curv=[(0, 0.0), (5000, 0.0)]), targets=None, statics=False)
        rest = (w.t - w.t0 > 16.0) & (w.t - w.t0 < 39.5)
        w.v[rest] = 0.0
        w.w[rest] = 0.0
        w.a[rest] = 0.0
        cls.log = []

        def probe(I, t, m):
            if m["type"] == "ego":
                ib = I.model.index["vehicle.b_g"]
                cls.log.append((t, float(I.x_now[ib]), math.sqrt(float(I.P_now[ib, ib])), I.stand.state, I.stand.evidence))
        cls.I = drive(w, vehicle=dict(gyro_bias=0.02, gyro_sigma=0.0015), probe=probe, restore={"vehicle.b_g": [0.0, 0.0005 ** 2]})
        cls.log = np.array(cls.log)

    def at(self, t):
        return self.log[np.searchsorted(self.log[:, 0], t)]

    def test_waits_then_relearns(self):
        r = self.at(25.0)                                 # nine seconds into the rest: claimed, the bias far off still
        self.assertEqual(int(r[3]), 1)                    # (it creeps: the camera sees no turn, so the gyro's rows teach it)
        self.assertEqual(int(r[4]) & 64, 64)
        self.assertGreater(abs(r[1] - 0.02), 0.012)
        r = self.at(27.5)                                 # confirmed after 10 s of quiet: the bias is the gyro's reading
        self.assertEqual(int(r[3]), 2)
        self.assertLess(abs(r[1] - 0.02), 0.0015)
        self.assertAlmostEqual(r[1], FROZEN["step_bias_27"], places=6)
        ev = [e for e in self.I.fdi.events if e["kind"] == "gyro_bias"]
        self.assertEqual(len(ev), 1)
        self.assertIn("re-estimating", ev[0]["text"])
        c = self.I.state(int((self.I.now) * 1e9))["calibration"]["vehicle"]
        self.assertEqual(c["stand"]["bias_steps"], 1)


def turns_world(T=240.0):
    """A town drive with a turn every 20 s (alternating, R 25-40 m at 9 m/s: yaw accelerations of 0.2-0.5 rad/s2
    at their ends) and speed changes: excitation for the offsets."""
    speed = [(0, 0.0), (4, 9.0), (50, 9.0), (54, 12.0), (100, 12.0), (104, 8.0), (150, 8.0), (154, 11.0), (T, 11.0)]
    curv, s = [(0, 0.0)], 60.0
    k = 0
    while s < 9.0 * T * 1.5:
        r = (1.0 / (25.0 + 5.0 * (k % 4))) * (1 if k % 2 == 0 else -1)
        curv += [(s, 0.0), (s + 4.0, r), (s + 34.0, r), (s + 38.0, 0.0)]
        s += 170.0
        k += 1
    return World(seed=7, scenario=dict(T=T, speed=speed, curv=curv), targets=None, statics=False)


class OffsetDriveTest(unittest.TestCase):
    """A gyro whose value is 60 ms old at its stamp (injected): the camera yaw rate's lag behind the gyro reads
    65 ms instead of the image pair's 125 ms, and with `gyro_to_camera` the gyro's rows are put where they
    belong."""

    @classmethod
    def setUpClass(cls):
        cls.w = turns_world()
        cls.res = {}
        for name, lagged, cfg in (("clean", 0.0, {}), ("lagged", 0.06, {}),
                                  ("applied", 0.06, {"offsets_apply": True, "gyro_to_camera": True})):
            err = []

            def probe(I, t, m, err=err, w=cls.w):
                if m["type"] == "road" and t > 60.0:
                    sm = I.ekf.at(m["ts_mono_ns"] / 1e9)
                    if sm is not None:
                        tr = w.truth_at(m["ts_mono_ns"] / 1e9)
                        err.append((tr["w"], float(sm[0][E.W]) - tr["w"]))
            I = drive(cls.w, {"ego_motion": cfg}, vehicle=dict(gyro_lag=lagged, speed_floor=0.0, gyro_lsb=0.0), probe=probe)
            cls.res[name] = (I, np.array(err))

    def off(self, name, key):
        return self.res[name][0].lags[key][0]

    def test_clean_reads_the_pair_lag(self):
        lag = self.off("clean", "vision.yaw")
        self.assertLess(abs(lag.value - 0.125), 0.012)
        self.assertGreater(lag.info, 0.9)
        self.assertLess(abs(self.off("clean", "wheels.yaw").value), 0.015)
        self.assertLess(abs(self.off("clean", "vision.speed").value - 0.125), 0.03)

    def test_injected_gyro_lag_is_recovered(self):
        lag = self.off("lagged", "vision.yaw")
        self.assertLess(abs(lag.value - 0.065), 0.012)     # the camera is 125 ms late, the gyro 60: 65 between them
        self.assertLess(abs(self.off("lagged", "wheels.yaw").value + 0.06), 0.015)     # the wheels lead the gyro by it
        self.assertLess(lag.std, 0.015)
        self.assertAlmostEqual(lag.value, FROZEN["lagged_vision_yaw"], places=6)
        self.assertAlmostEqual(self.off("lagged", "wheels.yaw").value, FROZEN["lagged_wheels_yaw"], places=6)
        c = self.res["lagged"][0].state(int((self.w.t0 + 240.0) * 1e9))["calibration"]["time_offsets"]
        self.assertEqual(c["vision.yaw"]["applied"], 0.125)               # estimated and published, not applied
        self.assertFalse(c["gyro_shift"]["applied"])

    def test_applied_puts_the_gyro_on_the_camera_clock(self):
        I = self.res["applied"][0]
        self.assertLess(abs(I.gyro_shift() - 0.06), 0.015)        # (four minutes of turns; the prior still pulls)
        self.assertAlmostEqual(I.gyro_shift(), FROZEN["applied_gyro_shift"], places=6)
        # the yaw rate at the camera frames' times against the truth
        rms = {}
        for name in ("clean", "lagged", "applied"):
            e = self.res[name][1]
            rms[name] = float(np.sqrt(np.mean(e[:, 1] ** 2)))
        self.assertGreater(rms["lagged"], 2.0 * rms["clean"])
        self.assertLess(rms["applied"], 1.35 * rms["clean"])
        for name in rms:
            self.assertAlmostEqual(rms[name], FROZEN["w_rms_" + name], places=6)

    def test_persisted(self):
        # what a drive learnt is the next one's prior (the learnt store's record), bounded
        from chris_integrator.learners import TimeOffsets
        from chris_integrator.learnt import LearntStore

        class Est:
            on_new = []
        a, b = Est(), Est()
        a.integ = self.res["lagged"][0]
        b.integ = Integrator({})
        store = LearntStore(None)
        TimeOffsets(a).export(store, {})
        self.assertIn("estimator.offset.vision.yaw", store.records)
        ln = TimeOffsets(b)
        ln.seed(store, {})
        self.assertEqual(ln.n_prior, 3)
        lag = b.integ.lags["vision.yaw"][0]
        self.assertEqual(lag.value, round(self.off("lagged", "vision.yaw").value, 5))      # (the record's precision)
        self.assertEqual(b.integ.state(10 ** 12)["calibration"]["time_offsets"]["vision.yaw"]["value"], round(lag.value, 4))
        self.assertGreater(lag.std, 0.005)                # a restart may change it: never as sure as it was
        self.assertLess(lag.std, 0.05)
        # a record out of bounds is not a prior
        store.records["estimator.offset.vision.yaw"]["value"] = 3.0
        c = Est()
        c.integ = Integrator({})
        TimeOffsets(c).seed(store, {})
        self.assertEqual(c.integ.lags["vision.yaw"][0].value, 0.125)


# frozen from the native core (2026-11-24)
FROZEN = {'lag_value': 0.08213271448475129,
 'lag_std': 0.0027960638738384344,
 'lag_counts': (1044, 2550, 2, 4),
 'off_bias_44': 0.030207,
 'on_bias_44': 0.0040289,
 'on_w_44': 0.0309707,
 'on_bias_59': 0.0040703,
 'on_heading_err': 0.062013,
 'off_heading_err': -0.774036,
 'report_stand': {'state': 'moving',
                  'evidence': [],
                  'confirmations': 3,
                  'claimed_only_share': 0.657,
                  'bias_steps': 0},
 'report_gyro_bias': {'observed_by': 'observed',
                      'straight': True,
                      'since_standstill_s': 49.7,
                      'std_at_standstill': 0.000358,
                      'std_trend_per_min': -1.94e-05},
 'lagged_vision_yaw': 0.0705543,
 'lagged_wheels_yaw': -0.0534399,
 'applied_gyro_shift': 0.0489621,
 'w_rms_clean': 0.0024418,
 'w_rms_lagged': 0.0058552,
 'w_rms_applied': 0.003188,
 'step_bias_27': 0.0195007}
