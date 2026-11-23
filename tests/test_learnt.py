"""The store of learnt values (chris_integrator/learnt.py) and its learners (chris_integrator/learners.py): priors
with the uncertainty inflated by age, sanity bounds and conditions, atomic writes that survive a cut at any
moment, damaged files, rate limits, reset, and the learners of the stack handed their priors on the next start."""
import json
import math
import os
import tempfile
import unittest

from chris_integrator import learnt as LT
from chris_integrator.learners import Learnt, calib_signature

DAY_NS = 86400 * 10 ** 9


class Clock:
    def __init__(self, t=1_800_000_000 * 10 ** 9):
        self.t = t

    def __call__(self):
        return self.t


class Store(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "learnt.json")
        self.clock = Clock()

    def spec(self, **kw):
        return LT.Spec(**dict(dict(kind="gauss", lo=-1.0, hi=1.0, var_cap=0.04, q_restart=1e-4, q_per_day=1e-4), **kw))

    def test_a_value_comes_back_as_a_prior_with_its_uncertainty_inflated_by_age(self):
        st = LT.LearntStore(self.path, clock=self.clock)
        self.assertTrue(st.put("a", self.spec(), 0.12, 1e-4, 50))
        self.assertTrue(st.save(force=True))
        self.clock.t += 3 * DAY_NS
        st2 = LT.LearntStore(self.path, clock=self.clock)
        p = st2.prior("a", self.spec())
        self.assertEqual(p.value, 0.12)
        self.assertAlmostEqual(p.var, 1e-4 + 1e-4 + 3 * 1e-4, places=12)
        self.assertAlmostEqual(p.age_s, 3 * 86400.0)
        self.clock.t += 1000 * DAY_NS                       # never surer than the start spread
        self.assertIsNone(LT.LearntStore(self.path, clock=self.clock).prior("a", self.spec()))     # too old (365 d)
        p = LT.LearntStore(self.path, clock=self.clock).prior("a", self.spec(max_age_days=5000))
        self.assertEqual(p.var, 0.04)

    def test_bounds_and_conditions(self):
        st = LT.LearntStore(self.path, clock=self.clock)
        self.assertFalse(st.put("a", self.spec(), 3.0, 1e-4))            # outside the sanity bounds: not kept
        self.assertFalse(st.put("a", self.spec(), float("nan"), 1e-4))
        sp = self.spec(conditions=("calib",))
        st.put("b", sp, 0.1, 1e-4, conditions={"calib": "X"})
        self.assertIsNotNone(st.prior("b", sp, {"calib": "X"}))
        self.assertIsNone(st.prior("b", sp, {"calib": "Y"}))
        self.assertIn("learnt under calib", st.rejected["b"])
        self.assertIsNone(st.prior("b", sp, {}))
        self.assertIsNone(st.prior("b", self.spec(lo=0.5, hi=1.0)))      # bounds as the learner states them now

    def test_counts_are_discounted_and_a_limit_relaxes(self):
        st = LT.LearntStore(self.path, clock=self.clock)
        st.put("h", LT.Spec("counts", lo=0, hi=1e6), [[10.0, 20.0], [4.0, 0.0]], None, 34)
        st.put("m", LT.Spec("min", lo=2, hi=400), {"14": 20.0}, None, 1)
        self.clock.t += 30 * DAY_NS
        h = st.prior("h", LT.Spec("counts", lo=0, hi=1e6, half_life_days=30))
        self.assertEqual(h.value, [[5.0, 10.0], [2.0, 0.0]])
        m = st.prior("m", LT.Spec("min", lo=2, hi=400, relax_per_day=0.01))
        self.assertAlmostEqual(m.value["14"], 26.0)

    def test_a_cut_at_any_moment_of_a_write_leaves_a_readable_store(self):
        st = LT.LearntStore(self.path, clock=self.clock)
        st.put("a", self.spec(), 0.1, 1e-4)
        st.save(force=True)
        st.put("a", self.spec(), 0.2, 1e-4)
        st.save(force=True)
        good = open(self.path, "rb").read()
        prev = open(self.path + ".prev", "rb").read()
        # every prefix of the new file as a torn temp, with the old file in place: the old one is read
        for cut in range(0, len(good), max(1, len(good) // 40)):
            with open(self.path + ".tmp", "wb") as f:
                f.write(good[:cut])
            self.assertEqual(LT.LearntStore(self.path, clock=self.clock).records["a"]["value"], 0.2)
        os.remove(self.path + ".tmp")
        # the cut between the two renames: no file, the new one complete as temp, the old one as previous
        os.replace(self.path, self.path + ".tmp")
        self.assertEqual(LT.LearntStore(self.path, clock=self.clock).records["a"]["value"], 0.2)
        os.remove(self.path + ".tmp")
        self.assertEqual(LT.LearntStore(self.path, clock=self.clock).records["a"]["value"], 0.1)    # previous
        with open(self.path + ".prev", "wb") as f:
            f.write(prev[:len(prev) // 2])
        st = LT.LearntStore(self.path, clock=self.clock)
        self.assertEqual(st.records, {})
        self.assertIsNone(st.loaded_from)
        self.assertEqual(st.health()["state"], "degraded")

    def test_a_damaged_file_is_moved_aside_and_the_previous_one_used(self):
        st = LT.LearntStore(self.path, clock=self.clock)
        st.put("a", self.spec(), 0.1, 1e-4)
        st.save(force=True)
        st.put("a", self.spec(), 0.3, 1e-4)
        st.save(force=True)
        d = json.load(open(self.path))
        d["values"]["a"]["value"] = 0.9                    # a flipped bit: still JSON, the checksum no longer holds
        json.dump(d, open(self.path, "w"))
        st = LT.LearntStore(self.path, clock=self.clock)
        self.assertEqual(st.loaded_from, "previous")
        self.assertEqual(st.records["a"]["value"], 0.1)
        self.assertTrue(os.path.exists(self.path + ".bad"))
        with open(self.path, "w") as f:
            f.write('{"format": "chris-learnt", "version": 99, "values": {}}')
        self.assertIn("version", LT.LearntStore(self.path, clock=self.clock).load_problems[0])

    def test_writes_are_rate_limited(self):
        st = LT.LearntStore(self.path, {"min_interval_s": 600, "max_writes_per_hour": 3}, clock=self.clock)
        st.put("a", self.spec(), 0.1, 1e-4)
        self.assertTrue(st.due(1000.0))
        st.save(now=1000.0)
        st.put("a", self.spec(), 0.2, 1e-4)
        self.assertFalse(st.due(1100.0))
        self.assertTrue(st.due(1100.0, "stop"))             # a stop after driving: sooner
        st.save(now=1100.0)
        st.put("a", self.spec(), 0.25, 1e-4)
        st.save(now=1700.0)
        st.put("a", self.spec(), 0.3, 1e-4)
        self.assertFalse(st.due(2500.0))                    # three this hour
        self.assertTrue(st.due(4700.0))
        self.assertFalse(LT.LearntStore(self.path, clock=self.clock).due(5000.0))     # nothing changed

    def test_reset(self):
        st = LT.LearntStore(self.path, clock=self.clock)
        st.put("radar.a", self.spec(), 0.1, 1e-4)
        st.put("estimator.b", self.spec(), 0.1, 1e-4)
        self.assertEqual(st.reset(["radar."]), ["radar.a"])
        self.assertEqual(sorted(LT.LearntStore(self.path, clock=self.clock).records), ["estimator.b"])
        self.assertEqual(LT.main(["reset", "--path", self.path]), 0)
        self.assertEqual(LT.LearntStore(self.path, clock=self.clock).records, {})


CALIB = {"type": "calib", "pitch": 0.02, "yaw": -0.01, "roll": 0.005, "height_m": 1.22, "focal_px": 975, "cx": 640,
         "cy": 360, "width": 1280, "height": 720, "source": "config"}


class Learners(unittest.TestCase):
    """A drive, then a second start: the learners get their priors back."""

    def drive(self, datadir, T=25.0, seed=1):
        from chris_integrator.mixer import MessageClock
        from chris_integrator.sim import default_sensors
        from chris_integrator.sim import roaduser_sim as rs
        from chris_integrator.stack import build
        w = rs.scenario("parked_street", seed=seed)
        w.T = T
        det = rs.DetectorSim(seed=1001)
        sensors = default_sensors(vehicle={"seed": 108}, vision={"seed": 118, "leads": True}, gnss={"seed": 128},
                                  map={"seed": 138}) + [det]
        st = build(cfg={"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}}, mapper=None,
                   clock=MessageClock(), datadir=datadir)
        sent = False
        for t, m in w.messages(sensors):
            if t - w.t0 > T:
                break
            if not sent:
                st.deliver(det.calib_msg(m["ts_mono_ns"]))
                sent = True
            st.deliver(m)
        return st

    def test_the_stacks_learners_are_kept_and_given_back_as_priors(self):
        d = tempfile.mkdtemp()
        st = self.drive(d)
        keeper = st.mixer.plugin("learnt")
        keeper.close()                                     # shutdown: written
        rec = LT.LearntStore(os.path.join(d, "learnt.json")).records
        self.assertIn("radar.score_hist.cam", rec)
        self.assertIn("radar.h_min", rec)
        self.assertIn("estimator.vision.s_w", rec)
        self.assertIn("estimator.vehicle.b_g", rec)
        self.assertNotIn("estimator.vision.m", rec)        # a Gauss-Markov state is not the car's
        self.assertIn("radar.pitch_bias.cam", rec)
        # the next start: priors handed back
        from chris_integrator.mixer import MessageClock
        from chris_integrator.stack import build
        st2 = build(mapper=None, clock=MessageClock(), datadir=d)
        k2 = st2.mixer.plugin("learnt")
        I = st2.mixer.plugin("estimator").integ
        i = I.model.index["vehicle.b_g"]
        self.assertAlmostEqual(I.ekf.x0[i], rec["estimator.vehicle.b_g"]["value"])
        self.assertGreater(I.ekf.P0[i, i], rec["estimator.vehicle.b_g"]["var"])          # inflated, not the truth
        tr = st2.mixer.plugin("radar").box.tracker
        for a, b in zip(tr.score_hist["cam"][0], rec["radar.score_hist.cam"]["value"][0]):   # discounted by its age
            self.assertAlmostEqual(a, b, delta=1e-6 * max(1.0, b))
        k2.on_input("calib", CALIB, None)
        hm = st2.mixer.plugin("radar").box.h_min
        self.assertEqual(sorted(hm), sorted(int(k) for k in rec["radar.h_min"]["value"]))
        h = k2.health()
        self.assertEqual(h["loaded_from"], "file")
        self.assertIn("estimator.vehicle.b_g", h["priors"])

    def test_without_a_state_directory_nothing_is_written(self):
        st = self.drive(None, T=3.0)
        self.assertEqual(st.mixer.plugin("learnt").health()["state"], "idle")

    def test_calibration_signature(self):
        self.assertEqual(calib_signature(CALIB), {"calib": "p0.020 y-0.010 r0.005 h1.22", "focal": "f975 w1280"})
        self.assertEqual(calib_signature(None), {})
        self.assertTrue(math.isfinite(1.0))


if __name__ == "__main__":
    unittest.main()
