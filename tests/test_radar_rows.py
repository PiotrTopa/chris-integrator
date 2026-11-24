"""One car, one object in a row; the detector's miss runs; the pose head (docs/RADAR.md, section 10): the knobs
`row_ctx`, `det_chain`, `pd_learn`, `exist_shared`, `low_runs_only`, `low_exist_only`, `in_pose_gn`, `in_pose_gc`,
`in_pose_yaw` of the tracker. Each off: the radar is what it was (tests/test_native.py's goldens and digests, the
detector-input and parked-car goldens: none of them was recorded anew). Each on: what it is for, on the simulator; and
the recorded scans of tests/data/make_radar_golden.py --rows."""
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chris_integrator.native import radar as C
from chris_integrator.radar import tracker as TK
from chris_integrator.radar.box import Radar
from chris_integrator.sim import radar_score
from chris_integrator.sim import roaduser_sim as rs

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NEW = ("row_ctx", "det_chain", "pd_learn", "exist_shared", "low_runs_only", "low_exist_only", "in_pose_gn", "in_pose_gc",
       "in_pose_yaw")
ROWS = dict(ctx_prior=True, row_ctx=True, det_chain=True, pd_learn=True, exist_shared=True)
POSE = dict(ROWS, in_app=True, in_pose_gn=True, in_pose_gc=True, in_pose_yaw=True)


def hops(name, seed, night, tracker, det=None):
    """(updates, updates with another actor's box) of a simulated drive: the simulator knows whose every box is."""
    sys.path.insert(0, os.path.join(os.path.dirname(DATA), "..", "tools"))
    import radar_trace as RT
    w, d, box, R = RT.run(name, seed, night, {"tracker": tracker} if tracker else None, rs.det_params_arg(det) if det else None)
    return len(R["upd"]), sum(1 for u in R["upd"] if u["hop"])


class Knobs(unittest.TestCase):
    def test_the_scan_constants_end_where_they_did_with_the_knobs_off(self):
        for k in NEW:
            self.assertIn(k, TK.CFG)
        rd = Radar({"tracker": {"sil_bearing": True}})
        tr = rd.tracker
        from chris_integrator.radar.sensors import CameraBoxes
        n0 = len(tr._xcfg("cam", CameraBoxes()))
        for k in ("row_ctx", "det_chain", "pd_learn", "exist_shared", "in_pose_yaw"):
            t2 = Radar({"tracker": {k: True}}).tracker
            self.assertGreater(len(t2._xcfg("cam", CameraBoxes())), n0, k)
            self.assertTrue(t2.cfg["_in_any"])
        self.assertFalse(Radar().tracker.cfg["_in_any"])

    def test_a_gaussian_cut_below_a_bound(self):
        # far below: nothing is cut; at the mean: the half normal; far above: just above the bound
        m, v = C.radar_trunc_low(1.0, 0.5, -10.0)
        self.assertEqual((m, v), (1.0, 0.25))
        m, v = C.radar_trunc_low(0.0, 1.0, 0.0)
        self.assertAlmostEqual(m, math.sqrt(2.0 / math.pi), places=12)
        self.assertAlmostEqual(v, 1.0 - 2.0 / math.pi, places=12)
        m, v = C.radar_trunc_low(0.0, 1.0, 3.0)
        self.assertTrue(3.0 < m < 3.4 and 0.0 < v < 0.12)
        m, v = C.radar_trunc_low(0.0, 1.0, 9.0)
        self.assertTrue(9.0 < m < 9.2 and v > 0.0)


class Rows(unittest.TestCase):
    def test_fewer_updates_with_another_cars_box_in_a_parked_street(self):
        # three drives together (one drive by itself does not show it: parked_street, seed 1 by day has 86 of 765
        # without and 88 of 776 with; over twelve drives 12.0 % against 9.2 %, docs/RADAR.md section 10)
        runs = (("parked_street", 1, False), ("parked_street", 1, True), ("curve_parked", 2, False))

        def share(tracker):
            got = [hops(n, s, ni, tracker) for n, s, ni in runs]
            return sum(h for _, h in got) / sum(n for n, _ in got)
        base = share(None)
        self.assertLess(share(ROWS), 0.9 * base)
        self.assertLess(share(dict(ROWS, sil_bearing=True)), 0.9 * base)

    def test_with_the_appearance_and_the_pose_head_fewer_still(self):
        n0, h0 = hops("parked_street", 1, False, None, "detin_pose")
        n1, h1 = hops("parked_street", 1, False, POSE, "detin_pose")
        self.assertLess(h1 / n1, 0.5 * h0 / n0)

    def test_moving_traffic_is_followed_as_before(self):
        for name in ("platoon", "oncoming"):
            s0 = radar_score.score(radar_score.run(name, seed=1))
            s1 = radar_score.score(radar_score.run(name, seed=1, cfg={"tracker": ROWS}))
            held = lambda s: sum(a["held_share"] * a["det_s"] for a in s["actors"].values() if a.get("held_share") is not None)
            self.assertGreater(held(s1), 0.95 * held(s0), name)


class Night(unittest.TestCase):
    def confirmed_at(self, name, seed, tracker):
        """The range at which an object on the stopped car was first put out as confirmed (None: never)."""
        r = radar_score.run(name, seed=seed, night=True, cfg={"tracker": tracker} if tracker else None)
        prev = {}
        for f in r["frames"]:
            outs = [dict(o, oid=o["id"], x=o["x_m"], y=o["y_m"]) for o in f["objects"]]
            m = radar_score.match(outs, f["actors"], prev)
            prev = {outs[i]["oid"]: a["id"] for i, (a, _, _) in m.items()}
            for i, (a, _, _) in m.items():
                if a["role"] == "stopped_in_lane" and outs[i]["state"] in ("confirmed", "coast"):
                    return a["range"]
        return None

    def test_a_stopped_car_at_night_is_confirmed_where_it_was_not(self):
        # four boxes in a row, each of which a new object took in the best story (the view's size ramp said a tenth)
        k = dict(det_chain=True, pd_learn=True, exist_shared=True)
        got = [(self.confirmed_at(n, s, None), self.confirmed_at(n, s, k))
               for n, s in (("right_bend_stopped_13", 2), ("bend_stopped_13", 2))]
        self.assertTrue(all(b is not None for _, b in got), got)
        self.assertTrue(all(a is None or b >= a - 1.0 for a, b in got), got)
        self.assertTrue(any(a is None or b > a + 2.0 for a, b in got), got)


class Recorded(unittest.TestCase):
    def test_recorded_calls_give_what_they_gave(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import radar_golden_lib as G
        cases, codec = G.load(os.path.join(DATA, "radar_golden_rows.npz"))
        seen = {}
        for n, c in enumerate(cases):
            seen[c["fn"]] = seen.get(c["fn"], 0) + 1
            ret, args, hyp = G.replay(C, c, codec)
            where = "case %d %s" % (n, c["fn"])
            self.assertIsNone(G.same(ret, codec.dec(c["ret"]), where + " result"))
            for i, a in c["post"].items():
                if isinstance(a, dict) and "rows" in a:
                    rows = codec.arrays[a["rows"] + "i"]
                    self.assertIsNone(G.same(args[int(i)][rows], codec.arrays[a["rows"]], where + " rows"))
                else:
                    self.assertIsNone(G.same(args[int(i)], codec.dec(a), where + " argument %s" % i))
            if c["hyp_after"] is not None:
                self.assertIsNone(G.same(hyp, codec.dec(c["hyp_after"]), where + " hypotheses"))
        self.assertGreaterEqual(seen.get("Hyp.cycle", 0), 8)
        self.assertTrue(any("y" in (codec.dec(c["hyp_after"]) or {}) for c in cases if c["hyp_after"] is not None))

    @unittest.skipIf(os.uname().machine != "x86_64", "the digests are of the development machine's arithmetic")
    def test_the_messages_on_simulated_drives_are_what_they_were(self):
        sys.path.insert(0, DATA)
        import make_radar_golden as MG
        with open(os.path.join(DATA, "radar_digests_rows.json")) as f:
            want = json.load(f)
        self.assertEqual(MG.rows_digests(), want)


if __name__ == "__main__":
    unittest.main()
