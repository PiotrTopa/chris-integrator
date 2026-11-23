"""The decision layer's native numerics (native/src/decide.c): closed forms, an independent reference for the
exact Gaussian integral, Monte Carlo against it, and the golden data (tests/data/decide_golden.npz, made by
`python3 tests/data/make_decide_golden.py`)."""
import math
import os
import unittest

import numpy as np

from chris_integrator.decide import futures as FU, threat as TH
from chris_integrator.native import decide as ND

HERE = os.path.dirname(os.path.abspath(__file__))
EXACT = {"pc_method": "mc", "k_sigma": 0.0, "margin_long_m": 0.0, "margin_lat_m": 0.0, "brake_buildup_s": 0.0,
         "steer_buildup_s": 0.0, "ego_accel_sigma": 0.0, "ego_lat_sigma": 0.0, "others": False, "obj_ref": "centre"}


def one(x0, y0=0.0, v_ego=14.0, vx=0.0, vy=0.0, label="car", std=(0.01, 0.01, 0.01, 0.01), cfg=None, motion=None):
    o = {"id": 1, "state": "confirmed", "x_m": x0, "y_m": y0, "vx_mps": vx, "vy_mps": vy, "std": list(std),
         "motion": motion or ("stationary" if vx == 0 and vy == 0 else "moving"), "class": {"label": label}}
    m = {"objects": [o], "ego": {"speed_mps": v_ego, "yaw_rate_rps": 0.0, "accel_mps2": 0.0}}
    F = FU.build(m, dict(FU.DEFAULT_CFG, cv_modes="single", horizon_s=8.0))
    tm = TH.ThreatModel(dict(EXACT, **(cfg or {})))
    res, info = tm.evaluate(F, m["ego"])
    o, ms, src, orow, prow, ext = res[0]
    return TH.object_record(o, ms, src, orow, prow, ext, F), info, prow


def phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


class ClosedForms(unittest.TestCase):
    def test_braking_required_and_latest_onset_for_a_standing_car_ahead(self):
        for x0 in (30.0, 45.0, 60.0):
            r, info, _ = one(x0)
            g = x0 - 2.25 - 2.1                 # the object's rear (centre - half length) to the car's front
            self.assertAlmostEqual(r["a_req"]["brake"], 14.0 ** 2 / (2 * g), delta=0.08)
            A = info["a_avail_mps2"][0]
            # the latest onset of full braking: the gap minus the braking distance, at the speed (one sim step)
            self.assertAlmostEqual(r["ttb_s"], (g - 14.0 ** 2 / (2 * A)) / 14.0, delta=0.06)
            self.assertAlmostEqual(r["btn"], r["a_req"]["brake"] / A, delta=0.02)

    def test_a_slower_lead_needs_the_speed_difference_squared_over_twice_the_gap(self):
        r, _, _ = one(40.0, vx=8.0, v_ego=20.0)
        g = 40.0 - 2.25 - 2.1
        self.assertAlmostEqual(r["a_req"]["brake"], (20.0 - 8.0) ** 2 / (2 * g), delta=0.08)

    def test_steering_needs_the_lateral_overlap_over_the_time_to_reach(self):
        # a standing car offset 1.0 m right: the car must move 0.88 + 0.9 - (-1.0) ... to the left by 0.78 m, or right
        r, _, _ = one(40.0, y0=-1.0)
        tau = (40.0 - 2.25 - 2.1) / 14.0
        need_left = 0.88 + 0.9 - 1.0
        a = 2.0 * need_left / tau ** 2
        self.assertAlmostEqual(r["a_req"]["left"], a, delta=0.15 * a + 0.06)
        self.assertGreater(r["a_req"]["right"], r["a_req"]["left"])

    def test_no_collision_course_requires_nothing(self):
        r, _, _ = one(40.0, y0=-4.0)
        self.assertEqual(r["level"], 0)
        self.assertIsNone(r["a_req"])
        self.assertEqual(r["p_coll"], 0.0)

    def test_a_standing_car_certainly_in_the_path_is_hit_with_probability_one(self):
        for meth in ("mc", "analytic"):
            r, _, _ = one(40.0, cfg={"pc_method": meth})
            self.assertGreater(r["p_coll"], 0.999)
            self.assertLess(abs(r["why"]["t_contact_s"] - (40.0 - 2.25 - 2.1) / 14.0), 0.06)


class GaussianIntegral(unittest.TestCase):
    def test_axis_aligned_overlap_is_a_product_of_normal_distributions(self):
        rng = np.random.default_rng(3)
        for _ in range(30):
            e = (0.0, 0.0, 0.0, rng.uniform(0.5, 3), rng.uniform(0.5, 1.5))
            o = (0.0, 0.0, 0.0, rng.uniform(0.2, 3), rng.uniform(0.2, 1.5))
            mu = (rng.uniform(-6, 6), rng.uniform(-4, 4))
            sx, sy = rng.uniform(0.1, 3), rng.uniform(0.1, 2)
            hx, hy = e[3] + o[3], e[4] + o[4]
            want = (phi((hx - mu[0]) / sx) - phi((-hx - mu[0]) / sx)) * (phi((hy - mu[1]) / sy) - phi((-hy - mu[1]) / sy))
            got = ND.decide_inst_prob(e, o, mu, (sx * sx, 0.0, sy * sy))
            self.assertAlmostEqual(got, want, delta=1e-9)        # the adaptive quadrature of Owen's T per edge

    def test_turned_footprints_and_correlated_spread_against_sampling(self):
        rng = np.random.default_rng(5)
        z = rng.standard_normal((400000, 2))
        for _ in range(6):
            e = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1, 2.5), rng.uniform(0.6, 1))
            o = (0.0, 0.0, rng.uniform(-1.5, 1.5), rng.uniform(0.3, 2.5), rng.uniform(0.3, 1.2))
            mu = np.array([rng.uniform(-4, 4), rng.uniform(-3, 3)])
            A = rng.standard_normal((2, 2))
            S = A @ A.T + 0.2 * np.eye(2)
            pts = mu + z @ np.linalg.cholesky(S).T
            got = ND.decide_inst_prob(e, o, tuple(mu), (S[0, 0], S[0, 1], S[1, 1]))
            want = _overlap_share(e, o, pts)
            self.assertAlmostEqual(got, want, delta=0.004)

    def test_the_monte_carlo_draws_are_standard_normal(self):
        d = np.array(ND.decide_draws(4096))
        self.assertLess(abs(d.mean()), 0.01)
        self.assertLess(abs(d.std() - 1.0), 0.01)
        self.assertEqual(d.tolist(), np.array(ND.decide_draws(4096)).tolist())      # fixed


def _overlap_share(e, o, pts):
    """The share of centre points at which rectangle o (turned by o[2]) overlaps rectangle e (SAT, numpy)."""
    def axes(r):
        c, s = math.cos(r[2]), math.sin(r[2])
        return np.array([[c, s], [-s, c]])
    ea, oa = axes(e), axes(o)
    ok = np.ones(len(pts), bool)
    for ax in np.vstack([ea, oa]):
        d = np.abs((pts - np.array(e[:2])) @ ax)
        pe = e[3] * abs(ea[0] @ ax) + e[4] * abs(ea[1] @ ax)
        po = o[3] * abs(oa[0] @ ax) + o[4] * abs(oa[1] @ ax)
        ok &= d <= pe + po
    return ok.mean()


class Golden(unittest.TestCase):
    def test_the_recorded_results(self):
        """Every case to rounding; where a bisection or a Monte Carlo count decides, within its tolerance (a libm
        that differs in the last bit can tip a boundary: the cross-built module on musl)."""
        g = np.load(os.path.join(HERE, "data", "decide_golden.npz"))
        names = list(g["names"])
        self.assertGreater(len(names), 100)
        tol = np.full(TH.PF, 1e-9)
        for f in (TH.DP_AB, TH.DP_AL, TH.DP_AR, TH.DP_AC):
            tol[f] = 0.06
        for f in (TH.DP_BTN, TH.DP_STN, TH.DP_TN):
            tol[f] = 0.03
        for f in (TH.DP_TTB, TH.DP_TTS, TH.DP_TM, TH.DP_TCON):
            tol[f] = 0.06
        tol[TH.DP_PC] = 0.01
        tol[TH.DP_DIR] = 0.4
        bad = 0
        for nm in names:
            q = nm.split(":")[0]
            a = [g["%s_%s" % (q, k)] for k in ("t", "ego", "ohdr", "modes", "cfg", "thr")]
            pair = np.empty_like(g[q + "_pair"])
            obj = np.empty_like(g[q + "_obj"])
            ND.decide_eval(*a, pair, obj)
            want = g[q + "_pair"]
            same = np.isclose(pair, want, rtol=1e-9, atol=tol[None, :], equal_nan=True) | \
                (np.isinf(pair) & np.isinf(want))
            if not same.all():
                bad += 1
            levels_same = (obj[:, TH.DO_LE] == g[q + "_obj"][:, TH.DO_LE]).all()
            bad += 0 if levels_same else 1
        self.assertEqual(bad, 0)


if __name__ == "__main__":
    unittest.main()
