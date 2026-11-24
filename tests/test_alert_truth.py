"""tools/alert_truth.py: a said alert against the simulator's truth (true / arguable / false), its cause, the lead
time on a scripted conflict's obstacle, and the paired comparison of two result sets."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import alert_truth as AT  # noqa: E402
import alert_truth_lib as L  # noqa: E402


def truth(role="parked", kind="car", moving=False, speed=0.0, clear=1.0, x=30.0, d=-2.8):
    return {"role": role, "kind": kind, "moving": moving, "speed": speed, "clear_m": clear, "ref": [x, -1.0],
            "d_centre": d, "W": 1.8}


def row(tr, v=13.0, **kw):
    r = {"truth": tr, "ego_v": v}
    r.update(kw)
    return r


class Classify(unittest.TestCase):
    def test_parked_beside_the_strip_is_false(self):
        self.assertEqual(L.classify(row(truth(clear=0.8))), "false")

    def test_no_truth_actor_is_false(self):
        self.assertEqual(L.classify({"truth": None, "ego_v": 13.0}), "false")
        self.assertEqual(L.classify({"ego_v": 13.0}), "false")

    def test_stopped_in_the_strip_is_true_when_it_needs_braking_or_is_near(self):
        # 13 m/s, 30 m ahead: 169 / (2 x 27.9) = 3.0 m/s^2
        self.assertEqual(L.classify(row(truth("stopped_in_lane", clear=-1.6, x=30.0))), "true")
        # 13 m/s, 90 m ahead: 0.96 m/s^2 and 6.8 s: in the strip and closing, but mild
        self.assertEqual(L.classify(row(truth("stopped_in_lane", clear=-1.6, x=90.0))), "arguable")

    def test_half_in_the_lane_counts_by_its_body(self):
        self.assertEqual(L.classify(row(truth("half_in_lane", clear=-0.33, x=25.0, d=-1.45))), "true")

    def test_a_slow_lead_the_car_closes_on(self):
        # 14 m/s behind 9 m/s, 12 m ahead: 25 / 19.8 = 1.26 m/s^2, 1.98 s: true by the time
        self.assertEqual(L.classify(row(truth("slow_lead", moving=True, speed=9.0, clear=-1.7, x=12.0), v=14.0)), "true")
        # 40 m ahead: 0.33 m/s^2, 7.6 s
        self.assertEqual(L.classify(row(truth("slow_lead", moving=True, speed=9.0, clear=-1.7, x=40.0), v=14.0)), "arguable")
        # a lead that pulls away
        self.assertEqual(L.classify(row(truth("lead", moving=True, speed=16.0, clear=-1.7, x=12.0), v=14.0)), "false")

    def test_oncoming_in_the_strip_closes_with_both_speeds(self):
        self.assertEqual(L.classify(row(truth("oncoming", moving=True, speed=14.0, clear=-0.2, x=60.0), v=14.0)), "true")
        self.assertEqual(L.classify(row(truth("oncoming", moving=True, speed=14.0, clear=1.2, x=60.0), v=14.0)), "false")

    def test_a_pedestrian_walking_near_the_strip_is_arguable_a_standing_one_false(self):
        self.assertEqual(L.classify(row(truth("crossing_ped", "human", True, 1.4, clear=0.6, x=20.0))), "arguable")
        self.assertEqual(L.classify(row(truth("crossing_ped", "human", True, 1.4, clear=-0.2, x=20.0))), "true")
        self.assertEqual(L.classify(row(truth("kerb_ped", "human", False, 0.0, clear=0.6, x=20.0))), "false")

    def test_behind_the_car_or_standing_still_ourselves(self):
        self.assertEqual(L.classify(row(truth("stopped_in_lane", clear=-1.6, x=-3.0))), "false")
        self.assertEqual(L.classify(row(truth("stopped_in_lane", clear=-1.6, x=20.0), v=0.2)), "false")


def said(tr, d_track=-2.8, rng_err=0.5, std=(4.0, 0.5), hist=(), modes=None, ego=None, kind="vehicle"):
    modes = modes if modes is not None else [
        {"name": "stopped", "p0n": 0.4, "p0w": 0.4, "move_m": 0.0, "pairs": [{"ego": "path", "le": 3, "pc": 0.9}]},
        {"name": "parked", "p0n": 0.6, "p0w": 0.03, "move_m": 0.0, "pairs": [{"ego": "path", "le": 0, "pc": 0.0}]}]
    return {"truth": tr, "ego_v": 13.0, "rec": {}, "reason": {"kind": kind}, "track": {"std": list(std)},
            "err": {"d_track": d_track, "range": rng_err, "hdg_err": 0.0}, "hist": list(hist), "modes": modes,
            "ego_modes": ego or [{"name": "path", "p": 0.95}, {"name": "ctra", "p": 0.05}]}


class Cause(unittest.TestCase):
    def test_real(self):
        self.assertEqual(L.cause(said(truth("stopped_in_lane", clear=-1.6, d=0.1))), "real")
        self.assertEqual(L.cause(said(truth("slow_lead", moving=True, speed=9.0, clear=-1.7, d=0.2))), "real")

    def test_unmatched(self):
        self.assertEqual(L.cause({"truth": None}), "unmatched")

    def test_spread_against_mean_in_the_lane(self):
        self.assertEqual(L.cause(said(truth(), d_track=-2.8)), "spread")
        self.assertEqual(L.cause(said(truth(), d_track=-0.9, rng_err=3.0)), "in_lane_mean:honest")
        self.assertEqual(L.cause(said(truth(), d_track=-0.9, rng_err=-9.0)), "in_lane_mean:bad_range")

    def test_a_jump_of_the_track_within_the_last_second(self):
        # [t, level, exit, p, mass, x, y, vx, vy, ps]: the car closes 13 m/s; the place hops 5 m between two frames
        h = [[-0.4, 0, 0, 0.0, 0.0, 35.2, -2.0, 0, 0, 0.9], [-0.3, 0, 0, 0.0, 0.0, 33.9, -2.0, 0, 0, 0.9],
             [-0.2, 2, 2, 0.5, 0.5, 27.6, -1.0, 0, 0, 0.9], [-0.1, 2, 2, 0.5, 0.5, 26.3, -1.0, 0, 0, 0.9]]
        self.assertEqual(L.cause(said(truth(), hist=h)), "jump")
        self.assertTrue(L.jumped({"hist": h, "ego_v": 13.0}))
        self.assertFalse(L.jumped({"hist": h[:2], "ego_v": 13.0}))

    def test_moving_futures_of_a_standing_car(self):
        m = [{"name": "keep", "p0n": 0.5, "p0w": 0.5, "move_m": 30.0, "pairs": [{"ego": "path", "le": 3, "pc": 0.8}]},
             {"name": "parked", "p0n": 0.5, "p0w": 0.03, "move_m": 0.0, "pairs": [{"ego": "path", "le": 0, "pc": 0.0}]}]
        self.assertEqual(L.cause(said(truth(), modes=m)), "moving_modes")

    def test_the_cars_own_off_road_future(self):
        m = [{"name": "stopped", "p0n": 1.0, "p0w": 1.0, "move_m": 0.0,
              "pairs": [{"ego": "path", "le": 0, "pc": 0.0}, {"ego": "ctra", "le": 3, "pc": 1.0}]}]
        e = [{"name": "path", "p": 0.7}, {"name": "ctra", "p": 0.3}]
        self.assertEqual(L.cause(said(truth(), modes=m, ego=e)), "ego_straight")

    def test_masses(self):
        m_all, m_ego, m_st, m_both = L.masses(said(truth()))
        self.assertAlmostEqual(m_all, 0.4 * 0.95 * 0.9)
        self.assertAlmostEqual(m_ego, 0.4 * 0.9)


def run(scen, seed, night, n_false=0, lead=None, conflict=False):
    rows = [row(truth(clear=0.8)) for _ in range(n_false)]
    return {"scenario": scen, "seed": seed, "night": night, "fut": "auto", "collision_t": 11.0 if conflict else None,
            "first_true": lead, "said": rows}


class Tables(unittest.TestCase):
    def test_table_counts_the_streets_only(self):
        runs = [run("curve_parked", 1, False, 2), run("curve_parked", 2, False, 1), run("bend_stopped_13", 1, False, 1, 1.8, True)]
        runs[0]["said"].append(row(truth("slow_lead", moving=True, speed=9.0, clear=-1.7, x=12.0), v=14.0))
        tab, cz, tot = L.table(runs)
        self.assertEqual(tab, {"curve_parked": {"true": 1, "arguable": 0, "false": 3}})
        self.assertEqual(tot["false"], 3)
        tab2, _, tot2 = L.table(runs, conflicts=True)
        self.assertEqual(tot2["false"], 1)

    def test_leads_and_changes(self):
        base = [run("bend_stopped_13", s, n, lead=ld, conflict=True) for s, n, ld in ((1, False, 2.0), (2, False, 1.5), (3, False, None), (1, True, 0.8))]
        new = [run("bend_stopped_13", s, n, lead=ld, conflict=True) for s, n, ld in ((1, False, 2.02), (2, False, 1.1), (3, False, 0.9), (1, True, None))]
        ld = L.leads(base)
        self.assertEqual(ld[("bend_stopped_13", "D")], {"n": 3, "lead": [1.5, 2.0]})
        self.assertEqual(ld[("bend_stopped_13", "N")], {"n": 1, "lead": [0.8]})
        ch = L.lead_changes(base, new)
        self.assertEqual([(k[1], k[2], a, b) for k, a, b in ch], [(1, True, 0.8, None), (2, False, 1.5, 1.1), (3, False, None, 0.9)])
        self.assertEqual(L.paired(base, new, L.warned)["diff"], 0)

    def test_paired_difference_and_its_spread(self):
        base = [run(sc, s, n, 2) for sc in ("curve_parked", "parked_street") for s in range(1, 7) for n in (False, True)]
        same = L.paired(base, base, L.n_false)
        self.assertEqual((same["n"], same["units"], same["base"], same["diff"], same["sd"]), (24, 12, 48, 0, 0.0))
        new = [run(sc, s, n, 1 if s <= 3 else 2) for sc in ("curve_parked", "parked_street") for s in range(1, 7) for n in (False, True)]
        r = L.paired(base, new, L.n_false)
        self.assertEqual((r["base"], r["new"], r["diff"]), (48, 36, -12))
        self.assertTrue(1.5 < r["sd"] < 5.0 and r["lo"] <= -12 <= r["hi"] and r["hi"] < 0, r)
        # runs only one set has are left out
        self.assertEqual(L.paired(base, new[:10], L.n_false)["n"], 10)

    def test_other_episodes_of_a_conflict_run(self):
        r = run("bend_stopped_13", 1, False, 2, 1.8, True)
        r["said"].append(row(truth("stopped_in_lane", clear=-1.6, x=30.0)))
        self.assertEqual(L.n_other(r), 2)
        self.assertEqual(L.n_false(r), 0)
        self.assertEqual(L.warned(r), 1)


class Stack(unittest.TestCase):
    """One scripted conflict through the whole stack: the alert names the stopped car, and that episode is true."""

    def test_stopped_car_is_warned_and_true(self):
        r = AT.run_one(("stopped_cruise_14", 1, False, "auto", {"trace": True}))
        self.assertEqual((r["scenario"], r["seed"], r["night"]), ("stopped_cruise_14", 1, False))
        self.assertIsNotNone(r["first_true"])
        self.assertTrue(1.0 < r["first_true"] < 6.0, r["first_true"])
        self.assertTrue(r["said"] and all(L.classify(x) == "true" for x in r["said"]))
        self.assertEqual(L.cause(r["said"][0]), "real")
        self.assertEqual(r["said"][0]["truth"]["role"], "stopped_in_lane")
        # the trace follows the obstacle: its truth every frame, the track beside it once there is one
        self.assertTrue(len(r["trace"]) > 50 and all(x["role"] == "stopped_in_lane" for x in r["trace"]))
        last = r["trace"][-1]
        self.assertTrue(abs(last["track"]["d_ctr_err"]) < 1.0 and last["truth"]["clear_m"] < 0.0)

    def test_scenario_lists(self):
        self.assertIn("ped_between_parked_13", AT.true_scenarios())
        self.assertEqual(AT.seeds("1-3"), [1, 2, 3])
        self.assertEqual(AT.seeds("2,5"), [2, 5])


if __name__ == "__main__":
    unittest.main()
