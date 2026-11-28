"""The road model's lead as a sensor of the tracker (docs/RADAR.md, section 13): the range and the range rate of the
vehicle ahead in the car's own lane, no evidence that it is there. Fixtures at the radar box (a level camera's boxes
cut at the bottom of the view, the road message's `leads`); the simulator's lead as measured on recorded drives."""
import hashlib
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chris_integrator.radar import Radar  # noqa: E402
from chris_integrator.radar import models as M  # noqa: E402
from chris_integrator.radar import tracker as TK  # noqa: E402
from chris_integrator.radar.sensors import RoadLead  # noqa: E402
from tests.test_radar_stale import CAL, NS, boxes  # noqa: E402

LEAD = {"in_lead": True}
LS = {"in_lead": True, "sil_near": True}         # ... with the box width read from where the camera is


def road(t, x, v_rel, v_ego, prob=0.99, sx=None, sv=None, y=0.0):
    """A `road` message with the lead x m ahead of the camera, its distance changing at v_rel."""
    sx = 0.09 * x if sx is None else sx
    sv = (0.04 if abs(v_rel + v_ego) < 0.2 else 0.3) if sv is None else sv
    return {"type": "road", "ts_mono_ns": int(t * NS), "valid": True, "pose": {"vx": v_ego},
            "leads": [{"prob": prob, "x": [x], "y": [y], "v": [v_ego + v_rel], "x_std": [sx], "y_std": [0.2], "v_std": [sv]}]}


def run(cfg, T, ego_v=None, lead_v=None, x0=20.0, others=None, feed=True, box_until=1e9, prob=0.99, stop=4.0):
    """The radar box over T s at 20 Hz behind a car in its lane, x0 m ahead of the camera at the start (its near
    face). ego_v(s, x), lead_v(s): the two speeds over the ground (default: the car drives up to `stop` m behind
    the lead and stands; the lead stands). The lead's box every 0.2 s, cut at the bottom of the view where it
    reaches below it (nearer than 13.8 m: a direction and a top edge, no range); the road model's lead of it every
    frame (feed). others(s, X): further things. Returns [(s, the true distance, the true v_rel, the message's
    objects)] and the box."""
    rd = Radar({"tracker": cfg})
    rd.on_calib(CAL)
    out, X, x = [], 0.0, x0
    ego_v = ego_v or (lambda s, x: min(5.0, max(0.0, 1.2 * (x - stop))) if x - stop > 0.02 else 0.0)
    lead_v = lead_v or (lambda s: 0.0)
    for i in range(int(T * 20)):
        s, t = i * 0.05, 30.0 + i * 0.05
        ve, vl = ego_v(s, x), lead_v(s)
        rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": ve, "yaw_rate_rps": 0.0, "standstill": ve < 0.01})
        if feed:
            rd.on_road(road(t, x, vl - ve, ve, prob=prob))
        if i % 4 == 0:
            th = dict(others(s, X, x)) if others else {}
            if s < box_until:
                th["lead"] = (x, 0.0, 1.8, 1.5, "car")
            rd.on_road_users(boxes(t, th, cut=True))
        out.append((s, x, vl - ve, rd.frame(int(t * NS))["objects"]))
        X += ve * 0.05
        x += (vl - ve) * 0.05
    return out, rd


def nearest(objs, x, tol=6.0):
    c = [o for o in objs if abs(o["y_m"]) < 1.5 and abs(o["x_m"] - x) < tol]
    return min(c, key=lambda o: abs(o["x_m"] - x)) if c else None


T0 = 9.0                         # the queue starts: until then the car has driven up to the standing lead and stands


def pulls_away(s):
    return min(5.0, max(0.0, 1.2 * (s - T0)))


def stays(s, x):                 # the car drives up to 4 m behind the lead and does not follow it when it starts
    return (min(5.0, max(0.0, 1.2 * (x - 4.0))) if x > 4.02 else 0.0) if s < T0 else 0.0


class Off(unittest.TestCase):
    def test_the_knobs_are_off(self):
        self.assertIs(TK.CFG["in_lead"], False)
        self.assertIs(TK.CFG["sil_near"], False)
        self.assertEqual(TK.CFG["pose_gn_cut_deg"], 0.0)

    def test_without_the_knob_the_lead_is_not_read(self):
        a, _ = run({}, 14.0, stays, pulls_away, feed=True)
        b, _ = run({}, 14.0, stays, pulls_away, feed=False)
        self.assertEqual(a, b)
        self.assertTrue(any(objs for _, _, _, objs in a))

    def test_with_the_knob_and_no_lead_the_tracker_is_what_it_was(self):
        a, ra = run(LEAD, 14.0, stays, pulls_away, feed=False)
        b, rb = run({}, 14.0, stays, pulls_away, feed=False)
        self.assertEqual(a, b)
        self.assertEqual(list(ra.tracker.sensors), ["cam"])

    def test_an_improbable_lead_is_not_read(self):
        a, _ = run(LEAD, 14.0, stays, pulls_away, prob=0.6)
        b, _ = run({}, 14.0, stays, pulls_away, feed=False)
        self.assertEqual(a, b)


class Measurement(unittest.TestCase):
    def test_what_it_measures_and_its_jacobian(self):
        sen = RoadLead()
        pose = (1.0, -2.0, 0.3, 4.0 * math.cos(0.3), 4.0 * math.sin(0.3), 0.0)
        x = np.zeros(M.N)
        x[M.PX], x[M.PY], x[M.VX], x[M.VY] = 13.0, 2.5, 6.0, 2.0
        z, H = sen.h(x, pose)
        dx, dy = 12.0, 4.5
        r = math.hypot(dx, dy)
        self.assertAlmostEqual(z[0], r, places=9)                      # (no extent given: the centre)
        self.assertAlmostEqual(z[1], math.atan2(dy, dx) - 0.3, places=9)
        self.assertAlmostEqual(z[2], (dx * (6.0 - pose[3]) + dy * (2.0 - pose[4])) / r, places=9)
        for k in (M.PX, M.PY, M.VX, M.VY):
            d = np.zeros(M.N)
            d[k] = 1e-6
            num = (sen.h(x + d, pose)[0] - sen.h(x - d, pose)[0]) / 2e-6
            np.testing.assert_allclose(H[:, k], num, atol=1e-5)

    def test_its_range_is_the_near_face(self):
        out, _ = run(LS, 12.0)
        s, x, _, objs = out[-1]
        o = nearest(objs, x)
        self.assertIsNotNone(o)
        self.assertLess(abs(o["x_m"] - x), 0.3)
        self.assertGreater(o["ctr"][0] - o["x_m"], 1.5)             # (the centre is half a car beyond)


class Range(unittest.TestCase):
    """A lead whose boxes are cut at the bottom of the view from 13.8 m in: its range is then what the kind's height
    and the top edge give; with the lead it is where the lead is."""

    def test_a_cut_box_object_takes_the_leads_range(self):
        out, _ = run(LS, 12.0)
        base, _ = run({"sil_near": True}, 12.0, feed=False)
        err = [abs(o["x_m"] - x) for s, x, _, objs in out if s > 6.0 for o in [nearest(objs, x)] if o is not None]
        berr = [abs(o["x_m"] - x) for s, x, _, objs in base if s > 6.0 for o in [nearest(objs, x, tol=10.0)] if o is not None]
        self.assertGreater(len(err), 100)
        self.assertLess(max(err), 0.3)
        o, b = nearest(out[-1][3], out[-1][1]), nearest(base[-1][3], base[-1][1], tol=10.0)
        self.assertIn("lead", o["sources"])
        self.assertLess(np.median(err), np.median(berr) + 1e-9)
        # ... and what it states of its range is the lead's, where the cut boxes alone left it open
        self.assertLess(o["std"][0], 0.5 * b["std"][0])

    def test_the_rear_of_a_near_car_is_a_car_along_the_lane(self):
        # `sil_near`: the box of a car 4 m ahead is as wide as its rear seen from 4 m, not as its silhouette seen
        # from its centre's 6.3 m; read the old way the rectangle turns and grows to explain the width
        for cfg, feed in ((LS, True), ({"sil_near": True}, False)):
            out, _ = run(cfg, 12.0, feed=feed)
            o = nearest(out[-1][3], out[-1][1])
            self.assertLess(abs(o["hdg"]), 0.05)
            self.assertLess(o["len"], 5.3)
            self.assertLess(abs(o["wid"] - 1.8), 0.15)
        out, _ = run({}, 12.0, feed=False)
        o = nearest(out[-1][3], out[-1][1], tol=10.0)
        self.assertTrue(abs(o["hdg"]) > 0.15 or o["len"] > 6.0)

    def test_its_sigma_is_no_better_than_the_leads(self):
        out, _ = run(LS, 16.0, stop=10.0)
        s, x, _, objs = out[-1]
        o = nearest(objs, x)
        # the lead states 0.9 m at 10 m (read as 1.4 x that); a hundred returns do not make it 0.2 m
        self.assertGreater(o["std"][0], 0.3)
        self.assertLess(abs(o["x_m"] - x), 2.0 * o["std"][0])

    def test_a_far_object_does_not_take_the_lead_of_a_near_vehicle(self):
        # the lead speaks of a vehicle 6 m ahead that pulls away and has no box; a car stands 30 m ahead in the lane
        def others(s, X, x):
            return {"far": (30.0, 0.0, 1.8, 1.5, "car")}
        out, _ = run(LEAD, 8.0, ego_v=lambda s, x: 0.0, lead_v=lambda s: 0.5, x0=6.0, others=others, box_until=0.0)
        o = nearest(out[-1][3], 30.0)
        self.assertIsNotNone(o)
        self.assertNotIn("lead", o["sources"])
        self.assertEqual(o["motion"], "stationary")
        self.assertLess(abs(o["x_m"] - 30.0), 2.5)


class QueueStart(unittest.TestCase):
    def test_a_standing_lead_that_starts_is_followed(self):
        out, _ = run(LEAD, 15.0, stays, pulls_away)
        before = nearest(out[int((T0 - 0.1) * 20)][3], 4.0)
        self.assertIsNotNone(before)
        self.assertEqual(before["motion"], "stationary")
        oid = before["id"]
        # "it stands" is let go within 1 s of the start, and the same object is with the lead from then on
        t_go = next(s for s, x, _, objs in out if s > T0 and all(o["motion"] != "stationary" for o in objs if o["id"] == oid))
        self.assertLess(t_go, T0 + 1.0)
        for s, x, vr, objs in out:
            if s < T0 + 1.5:
                continue
            o = next((o for o in objs if o["id"] == oid), None)
            self.assertIsNotNone(o, "the lead's object is gone at %.2f s" % s)
            self.assertLess(abs(o["x_m"] - x), 1.5, "at %.2f s" % s)
            self.assertLess(abs(o["v_rel_mps"] - vr), 1.0, "at %.2f s" % s)

    def test_without_the_lead_it_stands_on(self):
        # (what the knob is for: by the same boxes alone the object still "stands" 1.5 s after the start, a metre
        # and more behind the lead, with half its speed)
        out, _ = run({}, 12.5, stays, pulls_away, feed=False)
        s, x, vr, objs = out[int((T0 + 1.5) * 20)]
        o = nearest(objs, x, tol=30.0)
        self.assertEqual(o["motion"], "stationary")
        s, x, vr, objs = out[int((T0 + 2.5) * 20)]
        o = nearest(objs, x, tol=30.0)
        self.assertTrue(o is None or x - o["x_m"] > 1.0)
        self.assertTrue(o is None or vr - o["v_rel_mps"] > 1.0)

    def test_a_lead_that_brakes_to_a_stop_stands(self):
        def lead_v(s):                                   # away at 3 m/s from 3 s, to rest between 5 and 6.5 s
            return 0.0 if s < 3.0 else 3.0 if s < 5.0 else max(0.0, 3.0 - 2.0 * (s - 5.0))
        out, _ = run(LEAD, 15.0, ego_v=lambda s, x: 0.0, lead_v=lead_v, x0=15.0)
        s, x, _, objs = out[-1]
        o = nearest(objs, x)
        self.assertIsNotNone(o)
        self.assertEqual(o["motion"], "stationary")
        self.assertLess(abs(o["x_m"] - x), 2.5)
        moving = [o["motion"] for s, x, _, objs in out if 4.0 < s < 5.0 for o in [nearest(objs, x)] if o]
        self.assertTrue(moving and all(m != "stationary" for m in moving))

    def test_a_lead_creeping_with_the_car_is_not_called_standing(self):
        # the car and the lead creep at 1.8 m/s, 5 m apart, from the first box on
        out, _ = run(LEAD, 10.0, ego_v=lambda s, x: 1.8, lead_v=lambda s: 1.8, x0=15.0, stop=15.0)
        seen = [(s, o) for s, x, _, objs in out for o in [nearest(objs, x)] if o is not None]
        self.assertTrue(seen)
        self.assertTrue(all(o["motion"] != "stationary" for s, o in seen if s > seen[0][0] + 1.0))
        self.assertLess(abs(seen[-1][1]["v_rel_mps"]), 0.6)


class NoExistence(unittest.TestCase):
    def test_a_lead_without_an_object_starts_none(self):
        out, rd = run(LEAD, 6.0, box_until=0.0)
        self.assertTrue(all(not objs for _, _, _, objs in out))
        self.assertEqual(len(rd.tracker.objects), 0)
        self.assertEqual(rd.tracker.stats["lead"]["started"], 0)

    def test_a_lead_does_not_keep_an_object_the_boxes_have_left(self):
        # boxed for 5 s, then never again while the lead goes on naming a vehicle there: the object goes as it would
        with_lead, _ = run(LEAD, 16.0, box_until=5.0, stop=10.0)
        without, _ = run({}, 16.0, box_until=5.0, feed=False, stop=10.0)

        def last(out):
            return max((s for s, x, _, objs in out if nearest(objs, x) is not None), default=None)
        self.assertIsNotNone(last(with_lead))
        self.assertLess(last(with_lead), 15.0)
        self.assertLess(last(with_lead), (last(without) or 5.0) + 3.0)

    def test_a_return_counts_as_no_sighting(self):
        out, rd = run(LEAD, 6.0)
        o = max(rd.tracker.objects, key=lambda o: o.hits)
        self.assertLessEqual(o.hits, 30)                 # (its boxes: one frame in four of 120)
        self.assertIn("lead", o.sources)


class Simulator(unittest.TestCase):
    def test_off_the_world_is_what_it_was(self):
        from chris_integrator.sim import roaduser_sim as rs
        from chris_integrator.sim.sensors import VisionSim
        w = rs.scenario("queue_creep", seed=1)
        self.assertFalse(w.world_opts.get("lead_real"))
        msgs = [m for _, m in zip(range(200), (m for _, m in VisionSim(seed=118, leads=True).generate(w)))]
        h = hashlib.sha256(repr([m["leads"] for m in msgs]).encode()).hexdigest()[:16]
        self.assertEqual(h, GOLD_OFF)

    def test_the_lead_as_measured(self):
        from chris_integrator.sim import roaduser_sim as rs
        from chris_integrator.sim.sensors import VisionSim
        self.assertTrue(rs.det_params_arg("detin_pose_scale+lead")["world"]["lead_real"])
        w = rs.scenario("queue_creep", seed=1, world={"lead_real": True})
        ex, ev, n, have = [], [], 0, 0
        for _, m in VisionSim(seed=118, leads=True).generate(w):
            t = m["ts_mono_ns"] / 1e9
            tr = min((o for o in w.actors_at(t, max_range=110.0) if o["kind"] in rs.VEHICLE_KINDS and o["ref_x"] > 0.5
                      and abs(o["lane_offset"]) <= 1.5), key=lambda o: o["ref_x"], default=None)
            if tr is None:
                continue
            n += 1
            ld = m["leads"][0]
            if ld["prob"] < 0.9:
                continue
            have += 1
            ex.append((ld["x"][0] - tr["ref_x"]) / ld["x_std"][0])
            ev.append(ld["v"][0] - m["pose"]["vx"] - (tr["vx"] - w.truth_at(t)["v"]))
        self.assertGreater(have / n, 0.9)
        ex, ev = np.array(ex), np.array(ev)
        # the distance: within its stated sigma in about two frames of three (measured 0.56-0.70), no gross bias
        self.assertTrue(0.5 < np.mean(np.abs(ex) < 1.0) < 0.85, np.mean(np.abs(ex) < 1.0))
        self.assertLess(abs(np.median(ex)), 0.6)
        self.assertLess(np.sqrt(np.mean(ev ** 2)), 0.7)
        # ... and its errors live: neighbouring frames share most of them
        self.assertGreater(np.corrcoef(ex[:-1], ex[1:])[0, 1], 0.5)


GOLD_OFF = "d9a41c5c0a8b9321"


if __name__ == "__main__":
    unittest.main()
