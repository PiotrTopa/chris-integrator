"""The simulator's side of docs/RADAR.md, section 12: a queue that creeps and starts with the lead's box cut by the
bottom of the view, the pose head's ground point of such a box, a car beside that falls back."""
import os
import statistics
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chris_integrator.sim import roaduser_sim as rs  # noqa: E402

REACH = (0.23, 0.035)


def lead_boxes(reach, seed=1, night=False, lead=900, name="queue_creep"):
    """[(t, view, the lead's rear from the camera (m), clip, the pose head's ground row, box)] of a run."""
    w = rs.scenario(name, seed=seed, night=night)
    p = dict(rs.det_params_arg("detin_pose_scale"))
    if reach:
        p["pose"] = dict(p["pose"], gn_cut_reach=reach)
    det = rs.DetectorSim(seed=1000 + seed, night=night, params=p)
    msgs = [m for _, m in w.messages([det]) if m.get("type") == "lights"]
    logs = {(round(t, 3), view): {tuple(round(v, 1) for v in bx): aid for aid, cls, bx, clip in logd} for t, view, logd in det.log}
    out = []
    for m in msgs:
        t = m["ts_mono_ns"] / 1e9
        lk = logs.get((round(t, 3), m["view"]), {})
        a = {x["id"]: x for x in w.actors_at(t)}.get(lead)
        for o in m.get("road_users") or []:
            if lk.get(tuple(round(float(v), 1) for v in o["box"])) == lead and a is not None:
                out.append((t - w.t0, m["view"], a["x"] - a.get("L", 4.5) / 2, o.get("clip") or "",
                            (o.get("pose") or {}).get("gn", [None, None])[1], o["box"]))
    return w, det, out


class Scenarios(unittest.TestCase):
    def test_the_presets_do_not_model_the_reach(self):
        self.assertIsNone(rs.POSE["gn_cut_reach"])
        self.assertIsNone(rs.det_params_arg("detin_pose_scale")["pose"]["gn_cut_reach"])

    def test_the_lead_of_the_creeping_queue_is_cut_by_the_view(self):
        w, det, out = lead_boxes(None)
        self.assertGreater(len(out), 60)
        near = [x[2] for x in out]
        self.assertLess(min(near), 5.6)                                    # standing: its rear 4.3-5.3 m from the camera
        creep = [x for x in out if x[2] < 10.0]
        self.assertGreater(len(creep), 40)
        self.assertEqual([x for x in creep if "b" not in x[3]], [])         # every box of it within 10 m is cut below
        self.assertTrue(all("l" not in x[3] and "r" not in x[3] for x in creep))   # ... and whole from side to side
        # the ego stands, creeps, stands, pulls away: under 3.2 m/s for the first 20 s and 12 m/s at the end
        import numpy as np
        v = np.asarray(w.v)
        t = np.asarray(w.t) - w.t0
        self.assertLess(float(v[t < 20.0].max()), 3.3)
        self.assertGreater(float(v[t > w.T - 3.0].min()), 9.0)
        self.assertLess(float(v[t < 3.0].max()), 0.05)

    def test_the_ground_point_of_a_cut_box_reaches_only_so_far(self):
        cal = rs.DetectorSim(seed=1).calib
        f, hc, cy = cal["focal_px"], cal["height_m"], cal["cy"] + cal["pitch"] * cal["focal_px"]
        far = {}
        for reach in (None, REACH):
            w, det, out = lead_boxes(reach)
            for view, (y0, y1) in ((0, (0, 446)), (1, (90, 410))):
                rows = [x for x in out if x[1] == view and x[2] < 6.5 and x[4] is not None]
                self.assertGreater(len(rows), 8)
                far[(reach is not None, view)] = statistics.median(hc * f / max(x[4] - cy, 1.0) - x[2] for x in rows)
                if reach:
                    lim = y1 + (REACH[0] + 3.0 * REACH[1]) * (y1 - y0)
                    self.assertLessEqual(max(x[4] for x in rows), lim + 0.6)
        # in the narrow view the point of a lead 5 m ahead then reads 4 m and more too far (measured on the drives:
        # 6 m), in the wide one 1.5-3 m (measured: 2.5 m); as the presets have it, about the same in both
        self.assertGreater(far[(True, 1)], 4.0)
        self.assertGreater(far[(True, 1)], far[(False, 1)] + 2.0)
        self.assertTrue(1.0 < far[(True, 0)] < 3.5, far)
        self.assertLess(abs(far[(False, 1)] - far[(False, 0)]), 1.0)

    def test_the_car_beside_falls_back_and_nothing_is_left_ahead_in_its_lane(self):
        w = rs.scenario("left_behind", seed=1)
        ev = dict((txt, t) for t, txt in w.spec["events"])
        t_b = ev["the car beside falls back"]
        a0 = {x["id"]: x for x in w.actors_at(w.t0 + t_b - 1.0)}
        a1 = {x["id"]: x for x in w.actors_at(w.t0 + t_b + 3.0)}
        self.assertTrue(5.0 < a0[921]["x"] - 2.25 < 9.5 and -4.0 < a0[921]["y"] < -2.6)
        self.assertLess(a1[921]["x"], 0.0)                                 # behind the camera three seconds later
        self.assertTrue(10.0 < a1[920]["x"] - 2.75 < 15.0 and abs(a1[920]["y"]) < 0.6)   # the van ahead, as before


if __name__ == "__main__":
    unittest.main()
