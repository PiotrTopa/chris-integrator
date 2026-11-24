"""The radar component (chris_integrator/radar): the car's own motion, the motion hypotheses, the sensors and
the tracker, on small worlds with known truth. The realistic judgement (detector simulator, traffic scenarios,
real drives) is tools/visual_radar_eval.py; these tests pin what each part must do."""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from chris_integrator.radar import Radar  # noqa: E402
from chris_integrator.radar import models as M  # noqa: E402
from chris_integrator.radar.ego import EgoMotion  # noqa: E402

NS = 10 ** 9


class Ego(unittest.TestCase):
    def test_circle_is_exact_and_closes(self):
        e = EgoMotion()
        n = 251
        v, w = 10.0, 2 * math.pi / (n * 0.05)
        for i in range(n + 1):
            e.add(100.0 + i * 0.05, v, w)
        x0, y0, p0 = e.pose_at(100.0)
        x1, y1, p1 = e.pose_at(100.0 + n * 0.05)
        self.assertLess(math.hypot(x1 - x0, y1 - y0), 0.05)                # back where it started
        self.assertAlmostEqual(p1 - p0, 2 * math.pi, places=6)             # one full turn

    def test_ground_and_car_frames_round_trip_and_a_standing_point_stays(self):
        e = EgoMotion()
        for i in range(200):
            e.add(i * 0.05, 8.0 + 0.02 * i, 0.2 * math.sin(i * 0.05))
        g = e.to_ground(30.0, 4.0, 2.0)
        self.assertAlmostEqual(e.to_car(g[0], g[1], 2.0)[0], 30.0, places=9)
        # the same ground point seen later: nearer, and still one point
        x, y = e.to_car(g[0], g[1], 4.0)
        self.assertLess(x, 30.0 - 14.0)
        self.assertAlmostEqual(e.to_ground(x, y, 4.0)[0], g[0], places=9)

    def test_the_reference_point_moves_sideways_in_a_turn(self):
        e = EgoMotion()
        e.add(0.0, 0.0, 0.5)
        e.add(0.1, 0.0, 0.5)
        vx, vy, w = e.velocity_at(0.05)                                    # turning on the spot: the camera swings
        self.assertAlmostEqual(math.hypot(vx, vy), 0.5 * 1.5, places=6)


class Models(unittest.TestCase):
    def test_hypotheses_predict_what_they_say(self):
        x = np.zeros(M.N)
        x[M.PX], x[M.VX], x[M.AX] = 10.0, 5.0, 1.0
        P = np.eye(M.N)
        self.assertAlmostEqual(M.predict(M.STAND, x, P, 1.0, "car")[0][M.PX], 10.0)
        self.assertAlmostEqual(M.predict(M.CV, x, P, 1.0, "car")[0][M.PX], 15.0)
        self.assertGreater(M.predict(M.CA, x, P, 1.0, "car")[0][M.PX], 15.3)
        self.assertEqual(M.predict(M.STAND, x, P, 1.0, "car")[0][M.VX], 0.0)

    def test_a_kind_bounds_its_speed_and_shapes_its_noise(self):
        x = np.zeros(M.N)
        x[M.VX] = 20.0
        xp, Pp = M.predict(M.CV, x, np.eye(M.N) * 0.01, 0.2, "human")
        self.assertLessEqual(math.hypot(xp[M.VX], xp[M.VY]), M.KIN["human"]["v_max"] + 1e-9)
        self.assertGreater(Pp[M.VX, M.VX], 100.0)                          # and the doubt about it is opened
        x[M.VX] = 10.0
        _, Pc = M.predict(M.CV, x, np.zeros((M.N, M.N)), 1.0, "car")
        _, Pt = M.predict(M.CV, x, np.zeros((M.N, M.N)), 1.0, "truck")
        self.assertGreater(Pc[M.VX, M.VX], Pt[M.VX, M.VX])                 # a car accelerates harder than a lorry
        self.assertGreater(Pc[M.VX, M.VX], Pc[M.VY, M.VY])                 # along its travel more than across

    def test_switching_rows_are_probabilities_and_only_a_hypothesis_without_speed_hands_over_to_standing(self):
        x, P = np.zeros(M.N), np.eye(M.N) * 0.04
        still = M.still(x, P, 0.7)                                        # no speed, known to 0.2 m/s
        x[M.VX] = 20.0
        fast = M.still(x, P, 0.7)
        unknown = M.still(np.zeros(M.N), np.eye(M.N) * 225.0, 0.7)        # nothing known: that is not "still"
        self.assertGreater(still, 0.8)
        self.assertLess(fast, 1e-6)
        self.assertLess(unknown, 0.01)
        for q in (still, fast):
            PI = M.transition(0.2, (q, q))
            self.assertTrue(np.allclose(PI.sum(axis=1), 1.0) and (PI >= 0).all())
        self.assertLess(M.transition(0.2, (fast, fast))[M.CV, M.STAND], 0.01 * 0.2 * 2)
        self.assertGreater(M.transition(0.2, (still, still))[M.CV, M.STAND], 0.2)


def world(v_ego, actors, T, seed=1, miss=0.3, pitch_err_deg=0.0, w_ego=0.0, radar=False, hz=5.0, on_out=None):
    """The car drives at v_ego (yaw rate w_ego); actors: name -> (x0, y0, vx, vy) on the ground (the car's
    start frame). The camera reports bearing and a ground point seen through a pitch error (as a range), alternating a wide
    and a zoom view; optionally a microwave radar reports range, azimuth, range rate. Returns the radar
    component and its last message."""
    rng = random.Random(seed)
    rd = Radar()
    ego = EgoMotion()
    last = None
    k = 0
    step = int(round(20.0 / hz))
    for i in range(int(T / 0.05) + 1):
        t = 50.0 + i * 0.05
        ego.add(t, v_ego, w_ego)
        rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": v_ego, "yaw_rate_rps": w_ego, "accel_long_mps2": 0.0, "standstill": v_ego == 0.0})
        if i % step == 0:
            k += 1
            gmin, half = (14.0, math.radians(33)) if k % 2 == 0 else (24.0, math.radians(26))
            objs, dets = [], []
            for name, (x0, y0, vx, vy) in actors.items():
                gx, gy = x0 + vx * (t - 50.0) - 1.5 * 0 , y0 + vy * (t - 50.0)
                x, y = ego.to_car(gx, gy, t)
                r, b = math.hypot(x, y), math.atan2(y, x)
                if radar and x > 1 and abs(b) < math.radians(45) and r < 150:
                    evx, evy, _ = ego.velocity_at(t)
                    psi = ego.pose_at(t)[2]
                    los = (math.cos(b + psi), math.sin(b + psi))
                    rr = (vx - evx) * los[0] + (vy - evy) * los[1]
                    dets.append({"range_m": r + rng.gauss(0, 0.25), "azimuth_deg": math.degrees(b + rng.gauss(0, math.radians(0.5))),
                                 "range_rate_mps": rr + rng.gauss(0, 0.15)})
                if x < 2 or abs(b) > half or r > 70 or rng.random() < miss:
                    continue
                o = {"class": "car", "class_conf": 0.7, "bearing_deg": math.degrees(b + rng.gauss(0, math.radians(0.3))), "bearing_std_deg": 0.5}
                if r >= gmin:
                    ang = math.atan2(1.22, r) + math.radians(pitch_err_deg) + rng.gauss(0, math.radians(0.2))
                    rm = 1.22 / math.tan(ang) if ang > 0.0 else 1e9
                    if rm > 80.0:
                        continue                                           # at the horizon: no ground point, no box used
                    o["range_m"], o["range_std_m"] = rm, math.radians(0.65) * rm * rm / 1.22
                else:
                    o["clip"] = "b"
                objs.append(o)
            rd.on_road_users({"type": "road_users", "sensor": "cam", "ts_mono_ns": int(t * NS), "objects": objs,
                              "coverage": {"bearing_min_deg": -math.degrees(half), "bearing_max_deg": math.degrees(half), "ground_min_m": gmin}})
            if radar:
                rd.on_objects({"type": "objects", "kind": "radar", "sensor": "front", "ts_mono_ns": int(t * NS), "objects": dets})
        last = rd.frame(int(t * NS))
        if on_out is not None:
            on_out(t - 50.0, last, ego)
    return rd, last


def nearest(msg, x, y, tol=8.0):
    best = None
    for o in msg["objects"]:
        d = math.hypot(o["x_m"] - x, o["y_m"] - y)
        if d < tol and (best is None or d < best[0]):
            best = (d, o)
    return best[1] if best else None


class Tracking(unittest.TestCase):
    def test_a_parked_car_passed_by_is_stationary_and_parked_despite_a_biased_range(self):
        for bias in (-0.5, 0.0, 0.5):                                      # the camera's pitch error, degrees
            seen, last = [], [None]
            world(10.0, {"p": (70.0, -4.0, 0.0, 0.0)}, 6.5, seed=3, pitch_err_deg=bias,
                  on_out=lambda t, m, e: (seen.extend(o for o in m["objects"]), last.__setitem__(0, m)))
            self.assertTrue(seen, bias)
            verdicts = [o["motion"] for o in seen]
            self.assertNotIn("moving", verdicts, bias)                     # never called driving
            # no box here (no size), an unknown pitch: its bearings tell once it is near; from then on for good
            self.assertTrue(all(v == "stationary" for v in verdicts[-len(verdicts) // 4:]), bias)
            st = [o for o in seen if o["motion"] == "stationary"]
            self.assertTrue(all("parked" in o["flags"] for o in st[len(st) // 2:]), bias)
            self.assertLess(abs(st[-1]["speed_mps"]), 1.5, bias)
            if bias:                                                       # its own bearings showed the pitch error
                self.assertGreater(last[0]["sensors"]["cam"]["pitch_bias_deg"] * bias, 0.0, bias)

    def test_a_car_ahead_at_our_speed_is_moving_and_on_the_path(self):
        rd, last = world(14.0, {"lead": (35.0, 0.0, 14.0, 0.0)}, 8.0, seed=2)
        o = nearest(last, 35.0, 0.0, 12.0)
        self.assertIsNotNone(o)
        self.assertEqual(o["motion"], "moving")
        self.assertIn("on_path", o["flags"])
        self.assertNotIn("in_path", o["flags"])
        self.assertLess(abs(o["vx_mps"] - 14.0), 4.0)
        self.assertEqual(o["class"]["label"], "car")

    def test_a_car_driving_with_us_is_also_predicted_along_our_path(self):
        rd, m = world(15.0, {"lead": (35.0, 0.3, 15.0, 0.0)}, 8.0)
        o = nearest(m, 35.0, 0.3)
        self.assertIn("with_traffic", o["flags"])
        far = o["predict_path"][-1]                                        # straight road: the path is straight on
        self.assertAlmostEqual(far["x_m"], o["x_m"] + o["speed_mps"] * far["t_s"], delta=0.3)
        self.assertAlmostEqual(far["y_m"], o["y_m"], delta=0.3)
        # the car turns: the path bends, and so does where the one ahead will be if it keeps to the road (one object
        # 35 m ahead on the car's arc, heading along it: its prediction along the path, 65 m on, is well to the left)
        from chris_integrator.radar.tracker import State
        st = State(1)
        x = np.zeros(M.N)
        th = 0.15 / 15.0 * 35.0
        x[M.PX], x[M.PY], x[M.VX], x[M.VY] = math.sin(th) / 0.01, (1.0 - math.cos(th)) / 0.01, 15.0 * math.cos(th), 15.0 * math.sin(th)
        st.a[0, 0:24], st.a[0, 216:219], st.a[0, 219:227] = np.tile(x, 3), (0.0, 1.0, 0.0), x
        st.a[0, 24:216] = np.tile(np.eye(M.N).ravel() * 0.01, 3)
        st.a[0, 227:291] = np.eye(M.N).ravel() * 0.01
        M.NR.radar_combine_rows(st.a, [0])
        num = M.NR.radar_frame(st.a, [0], [[0.0, 0.5, 1.0, 2.0]], [M._KIN4["car"]], M._native_par(M.CFG), (0.0, 0.0, 0.0),
                               (15.0, 0.15), rd._ncfg(), (0.5, 1.0, 2.0), None, None, [1.0], [0.7])[0]
        far = num[20][-1]
        self.assertGreater(far["y_m"], 7.0)
        self.assertGreater(math.atan2(far["y_m"] - num[1], far["x_m"] - num[0]), 0.3)
        # a standing car and an oncoming one are not "with the traffic"
        rd, m = world(15.0, {"stands": (70.0, 0.0, 0.0, 0.0)}, 3.0)
        for q in m["objects"]:
            self.assertIsNone(q["predict_path"])

    def test_pairings_are_judged_together_and_a_box_shape_says_which_way_a_car_points(self):
        from chris_integrator.radar.tracker import marginals
        self.assertAlmostEqual(float(marginals(np.array([[9.0]]))[0, 0]), 0.9, places=3)       # alone: its own odds
        two = marginals(np.array([[9.0], [9.0]]))                          # two objects, one detection: not both
        self.assertLess(float(two.sum()), 1.0)
        self.assertAlmostEqual(float(two[0, 0]), float(two[1, 0]), places=6)
        own = marginals(np.array([[50.0, 1.0], [1.0, 50.0]]))              # each has its own: no doubt
        self.assertGreater(float(own[0, 0]), 0.95)
        self.assertLess(M.side_on("car", 1.2, 0.0), 0.05)                  # seen from behind
        self.assertGreater(M.side_on("car", 3.0, 0.0), 0.95)               # seen from the side
        self.assertAlmostEqual(M.side_on("car", 2.9, math.radians(45)), 0.5, places=2)   # from 45 degrees: cannot tell
        self.assertIsNone(M.side_on("human", 0.4, 0.0))

    def test_a_car_standing_in_our_lane_is_an_obstacle_with_a_time_to_reach_it(self):
        got = []
        world(14.0, {"obst": (110.0, 0.2, 0.0, 0.0)}, 6.5, seed=4,
              on_out=lambda t, m, e: got.extend((t, o) for o in m["objects"] if "in_path" in o["flags"]))
        self.assertTrue(got)
        t, o = got[0]
        self.assertGreater(o["x_m"], 20.0)                                 # said while it is still ahead
        self.assertIsNotNone(o["ttc_s"])
        self.assertLess(abs(o["ttc_s"] - o["x_m"] / 14.0), 0.6 * o["x_m"] / 14.0)

    def test_crossing_and_oncoming_traffic_is_never_stationary(self):
        for actors in ({"cross": (40.0, -25.0, 0.0, 7.0)}, {"onc": (120.0, 3.5, -14.0, 0.0)}):
            seen = []
            world(10.0, actors, 5.0, seed=5, miss=0.2, on_out=lambda t, m, e: seen.extend(o["motion"] for o in m["objects"]))
            self.assertNotIn("stationary", seen, actors)

    def test_standing_car_while_we_stand_and_then_it_drives_off(self):
        seen = []
        actors = {"a": (30.0, 0.0, 0.0, 0.0)}
        world(0.0, actors, 4.0, seed=6, miss=0.2, on_out=lambda t, m, e: seen.extend(o["motion"] for o in m["objects"]))
        self.assertNotIn("moving", seen)
        self.assertIn("stationary", seen)

    def test_a_turn_of_the_car_is_not_a_movement_of_the_object(self):
        seen = []
        world(8.0, {"p": (45.0, -6.0, 0.0, 0.0)}, 4.0, seed=7, w_ego=0.12, miss=0.2,
              on_out=lambda t, m, e: seen.extend(o for o in m["objects"]))
        self.assertTrue(seen)
        self.assertNotIn("moving", [o["motion"] for o in seen])
        self.assertLess(seen[-1]["speed_mps"], 2.0)

    def test_predictions_are_checked_and_hold_for_a_steady_object(self):
        rd, last = world(14.0, {"lead": (35.0, 0.0, 12.0, 0.0)}, 8.0, seed=8, miss=0.1)
        o = nearest(last, 35.0 - 2.0 * 8.0, 0.0, 12.0)
        self.assertIsNotNone(o)
        self.assertIsNotNone(o["consistency"])
        self.assertLess(o["consistency"], 3.0)
        self.assertEqual([p["t_s"] for p in o["predict"]], [0.5, 1.0, 2.0])
        # where it will be on the ground, in the car frame of now: 2 s further along at its own speed
        self.assertLess(abs(o["predict"][-1]["x_m"] - (o["x_m"] + 2.0 * o["vx_mps"])), 3.0)

    def test_a_microwave_radar_gives_speed_at_once_and_sharpens_the_camera_object(self):
        cam = world(14.0, {"lead": (60.0, 0.0, 9.0, 0.0)}, 3.0, seed=9)[1]
        both = world(14.0, {"lead": (60.0, 0.0, 9.0, 0.0)}, 3.0, seed=9, radar=True)[1]
        oc, ob = nearest(cam, 45.0, 0.0, 15.0), nearest(both, 45.0, 0.0, 6.0)
        self.assertIsNotNone(ob)
        self.assertEqual(ob["motion"], "moving")
        self.assertLess(abs(ob["vx_mps"] - 9.0), 0.8)
        self.assertLess(abs(ob["x_m"] - 45.0), 1.5)
        self.assertEqual(sorted(ob["sources"]), ["cam", "front"])          # one object, two sensors
        self.assertEqual(len([o for o in both["objects"] if abs(o["y_m"]) < 3]), 1)
        if oc is not None:
            self.assertLess(ob["std"][0], oc["std"][0])

    def test_an_object_that_left_the_field_is_remembered_with_its_id_where_it_stands(self):
        seen = {}
        def note(t, m, e):
            for o in m["objects"]:
                seen.setdefault(o["id"], []).append((t, o["state"], o["x_m"], o["y_m"], o["motion"]))
        world(10.0, {"p": (60.0, -4.0, 0.0, 0.0)}, 14.0, seed=3, on_out=note)
        rows = max(seen.values(), key=len)
        mem = [r for r in rows if r[1] == "memory"]
        self.assertTrue(mem)                                               # passed at about 6 s; still known at 14 s
        self.assertGreater(mem[-1][0], 13.5)
        self.assertLess(abs(mem[-1][2] - (60.0 - 10.0 * mem[-1][0])), 4.0)   # behind the car now, where it stands
        self.assertLess(abs(mem[-1][3] + 4.0), 2.0)
        self.assertEqual(mem[-1][4], "stationary")

    def test_a_moving_object_out_of_view_is_forgotten_when_its_place_blurs(self):
        seen = []
        world(10.0, {"cross": (40.0, -25.0, 0.0, 7.0)}, 30.0, seed=5, miss=0.2,
              on_out=lambda t, m, e: seen.extend((t, o["state"]) for o in m["objects"]))
        self.assertTrue(seen)
        self.assertLess(max(t for t, s in seen), 25.0)

    def test_the_display_draws_the_radars_objects_marks_parked_ones_and_raises_what_stands_in_the_way(self):
        from chris_integrator.trackers.roadusers_view import RadarDisplay

        def obj(i, x, y, flags, motion="stationary", state="confirmed", vx=0.0, label="car"):
            return {"id": i, "state": state, "x_m": x, "y_m": y, "vx_mps": vx, "vy_mps": 0.0, "v_rel_mps": vx - 15.0,
                    "std": [2.0, 0.3, 1.0, 0.3], "age_s": 3.0, "seen": {"cam": 0.1}, "motion": motion, "flags": flags,
                    "ttc_s": 2.4 if "threat" in flags else None,
                    "class": {"label": label, "confidence": 0.8, "source": "cam", "votes": 5}}

        rd = RadarDisplay()
        msg = {"type": "radar", "ts_mono_ns": 10 * NS, "ego": {"speed_mps": 15.0}, "objects": [
            obj(1, 20.0, -3.5, ["parked"]), obj(2, 40.0, 0.2, ["in_path", "on_path"]),
            obj(3, 30.0, 3.0, ["with_traffic"], motion="moving", vx=14.0),
            obj(4, 25.0, -4.0, [], label="bike"),                           # a two-wheeler standing beside the road
            obj(5, 60.0, 0.0, ["threat", "on_path"], motion="unknown"),
            obj(6, 35.0, 0.5, ["in_path"], state="memory"),                 # not seen any more: not shown
            obj(7, 50.0, -0.1, ["on_path"], motion="moving", vx=15.0)]}     # ... the lead
        rd.update(msg)
        objs, alert = rd.objects(10 * NS + NS // 10, lead={"x": 50.0, "y": 0.0})
        by = {o["id"]: o for o in objs}
        self.assertEqual(sorted(by), ["ru:1", "ru:2", "ru:3", "ru:5"])      # bike, memory and the lead's own: not drawn
        self.assertTrue(by["ru:1"]["attrs"]["parked"])
        self.assertFalse(by["ru:3"]["attrs"]["parked"])
        self.assertTrue(by["ru:2"]["attrs"]["obstacle"])
        self.assertAlmostEqual(by["ru:1"]["y"], 3.5)                        # the view's y is right
        self.assertAlmostEqual(by["ru:1"]["x"], 20.0 - 15.0 * 0.1, places=2)   # carried to the frame's time
        self.assertEqual(alert["level"], "threat")                         # the threat outranks the nearer obstacle
        self.assertEqual(alert["id"], "ru:5")
        # it leaves the radar: fades out where it was, then is gone
        rd.update(dict(msg, ts_mono_ns=10 * NS + NS // 5, objects=[]))
        objs, alert = rd.objects(10 * NS + 3 * NS // 10, lead=None)
        self.assertTrue(objs and all(o["phase"] == "leave" for o in objs))
        self.assertIsNone(alert)
        self.assertEqual(rd.objects(12 * NS, lead=None), ([], None))

    def test_around_lists_every_object_all_round_marks_the_remembered_ones_and_keeps_objects_as_they_were(self):
        import copy
        import json
        from chris_integrator.trackers.roadusers_view import RadarDisplay

        def obj(i, x, y, flags, motion="stationary", state="confirmed", vx=0.0, vy=0.0, label="car", sd=(2.0, 0.3),
                seen=0.1, conf=0.8):
            return {"id": i, "state": state, "x_m": x, "y_m": y, "vx_mps": vx, "vy_mps": vy, "v_rel_mps": vx - 15.0,
                    "std": [sd[0], sd[1], 1.0, 0.3], "age_s": 3.0, "seen": {"cam": seen}, "motion": motion,
                    "flags": flags, "ttc_s": 2.4 if ("threat" in flags or "in_path" in flags) else None,
                    "class": {"label": label, "confidence": conf, "source": "cam", "votes": 5}}

        objects = [
            obj(1, 20.0, -3.5, ["parked"]), obj(2, 40.0, 0.2, ["in_path", "on_path"]),
            obj(3, 30.0, 3.0, ["with_traffic"], motion="moving", vx=14.0),
            obj(4, 25.0, -4.0, [], label="bike"),                                    # riderless, beside the road
            obj(5, 60.0, 0.0, ["threat", "on_path"], motion="unknown"),
            obj(6, -12.0, 3.4, ["parked", "in_path", "threat"], state="memory", sd=(3.5, 1.0), seen=6.0),   # passed
            obj(7, 50.0, -0.1, ["on_path"], motion="moving", vx=15.0),             # the lead
            obj(8, 8.0, 30.0, [], motion="moving", vy=-9.0, state="coast", label="van", conf=0.3),   # crossing
            obj(9, -120.0, 0.0, ["parked"], state="memory")]                       # beyond the largest range
        msg = {"type": "radar", "ts_mono_ns": 10 * NS, "ego": {"speed_mps": 15.0}, "objects": objects}
        lead = {"id": "lead:3", "kind": "vehicle", "role": "lead", "x": 50.0, "y": 0.0, "vx": 0.0, "sx": 0.6,
                "phase": "steady", "alpha": 1.0, "attrs": {"dist_std_m": 0.6, "ttc_s": None, "standing": False},
                "cls": {"label": "truck", "confidence": 0.9}}
        T = 10 * NS + NS // 10
        plain = RadarDisplay({"around": False})
        plain.update(copy.deepcopy(msg))
        want = plain.objects(T, lead=lead)
        rd = RadarDisplay()
        rd.update(msg)
        self.assertEqual(rd.objects(T, lead=lead), want)        # the road view's objects do not change
        ar = rd.around(T, lead)
        by = {e["id"]: e for e in ar}
        self.assertEqual(sorted(by), ["lead:3", "ru:1", "ru:2", "ru:3", "ru:5", "ru:6", "ru:8"])
        self.assertEqual(by["lead:3"]["fl"], "L")               # the lead stands for the radar's object 7
        self.assertEqual((by["lead:3"]["k"], by["lead:3"]["mv"], by["lead:3"]["gx"]), ("truck", "m", 15.0))
        m = by["ru:6"]                                          # passed, remembered: behind, never a threat
        self.assertEqual((m["st"], m["fl"], m["sd"]), ("memory", "p", 3.5))
        self.assertNotIn("ttc", m)
        self.assertAlmostEqual(m["x"], -12.0 - 1.5)              # carried to the frame's time
        self.assertAlmostEqual(m["ago"], 6.1)
        self.assertEqual((by["ru:5"]["fl"], by["ru:5"]["ttc"]), ("t", 2.4))
        self.assertEqual((by["ru:2"]["fl"], by["ru:2"]["ttc"]), ("o", 2.4))
        self.assertEqual(by["ru:3"]["fl"], "w")
        self.assertEqual(by["ru:3"]["gx"], 14.0)                # over the ground
        self.assertNotIn("gx", by["ru:1"])                       # a standing object sends no velocity
        c = by["ru:8"]
        self.assertEqual((c["st"], c["y"], c["gy"]), ("coast", -29.1, 9.0))     # view frame: y right
        self.assertNotIn("k", c)                                 # the class below its confidence is not sent
        self.assertEqual([e["id"] for e in ar][0], "ru:6")       # nearest first (all round)
        self.assertTrue(all(e.get("a") == 0.0 for e in ar if e["id"] != "lead:3"))   # they fade in
        size = len(json.dumps(ar, separators=(",", ":")))
        self.assertLess(size / len(ar), 100)                     # bytes per object
        # the cap keeps threats, obstacles and the lead first, then the nearest
        rd2 = RadarDisplay({"around_max": 3})
        rd2.update(msg)
        self.assertEqual(sorted(e["id"] for e in rd2.around(T, lead)), ["lead:3", "ru:2", "ru:5"])
        # it leaves the radar: fades out where it was
        rd.update(dict(msg, ts_mono_ns=10 * NS + NS // 5, objects=[]))
        ar = rd.around(10 * NS + 3 * NS // 10, None)
        self.assertTrue(ar and all(0.0 <= e["a"] < 1.0 for e in ar))
        self.assertEqual(rd.around(12 * NS, None), [])

    def test_the_map_of_unexplained_boxes_holds_ground_boxes_only_and_merge_counts_do_not_outlive_objects(self):
        rd = Radar()
        box = lambda u, v0, h: [u - 0.6 * h, v0 - h, u + 0.6 * h, v0]
        for i in range(120):
            t = 10.0 + i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 8.0, "yaw_rate_rps": 0.0})
            if i % 4 == 0:
                objs = []
                for k, (rng, b) in enumerate(((40.0 - 8.0 * (t - 10.0), 3.0), (41.5 - 8.0 * (t - 10.0), 3.4), (9.0, -20.0))):
                    if rng < 3.0:
                        continue
                    o = {"class": "car", "class_conf": 0.7, "bearing_deg": b, "bearing_std_deg": 0.4,
                         "box": box(640.0 - 975.0 * math.tan(math.radians(b)), 400.0, 975.0 * 1.5 / rng),
                         "scale_rate": 8.0 / rng, "scale_rate_std": 0.05, "scale_dt_s": 0.2}
                    if k == 2:
                        o["clip"] = "b"                                      # cut at the bottom: a direction only
                    else:
                        o["range_m"], o["range_std_m"] = rng, 0.1 * rng
                    objs.append(o)
                rd.on_road_users({"type": "road_users", "sensor": "cam", "ts_mono_ns": int(t * NS), "objects": objs})
            rd.frame(int(t * NS))
            tr = rd.tracker
            for mem in tr.unexplained.values():
                self.assertTrue(mem)
                for _, kind, z, _ in mem:
                    self.assertIn(kind, ("rb", "rbh"))
                    self.assertEqual(len(z), len(kind))                    # the scale-rate row is not in the map
            live = {o.id for o in tr.objects}
            for o in tr.objects:
                self.assertLessEqual(set(o.merge_t), live)                 # no count kept for an object that is gone

    def test_a_doubtful_pairing_is_decided_by_the_scans_that_follow(self):
        """Two standing things 1.4 m apart, a sensor that places them to 0.4 m across. One scan brings a single
        detection between them, a hair nearer A. It was B's (B has moved towards A; the next scans show B there and A
        where it was). With hypotheses kept over scans the first reading (A's) is revised: A's history says it was not
        reported in that scan, and A is not dragged. Decided scan by scan (hyp_n_scan 0) the first reading stands."""
        from chris_integrator.radar.sensors import Position
        from chris_integrator.radar.tracker import M_DET, M_T, OFF, Tracker

        def run(cfg):
            ego = EgoMotion()
            tr = Tracker(ego, cfg)
            sen = Position()
            R = np.array([0.01, 0.16])

            def scan(k, pts):
                t = 10 + 0.1 * k
                ego.add(t, 0.0, 0.0, stands=True)
                tr.process("pos", t, sen, [{"z": np.array(p, float), "R": R, "attrs": {"class": "car", "class_conf": 0.8}} for p in pts])
                tr.manage(t)
                return t
            for k in range(12):
                scan(k, [(20, 0.7), (20, -0.7)])
            A = min(tr.objects, key=lambda o: abs(o.x[1] - 0.7))
            B = min(tr.objects, key=lambda o: abs(o.x[1] + 0.7))
            t_amb = scan(12, [(20, 0.05)])
            first = (int(A.m[M_DET]), int(B.m[M_DET]), A.nleaf, round(A.pp, 2))
            for k in range(13, 17):
                scan(k, [(20, 0.7), (20, 0.0)])

            def took(o):                                 # what the object's story now says about that scan
                S, m0 = tr.store.a, OFF["m"][0]
                return [int(S[r, m0 + M_DET]) for r in tr.hyp.chain(o.h) if abs(S[r, m0 + M_T] - t_amb) < 1e-6][0]
            return first, took(A), took(B), float(A.x[1]), sorted(o.id for o in tr.objects), (A.id, B.id), tr

        first, a, b, ya, ids, ab, tr = run({})
        self.assertEqual(first[:2], (0, -1))                              # at the scan itself: A's, and in doubt
        self.assertGreaterEqual(first[2], 2)
        self.assertLess(first[3], 0.9)
        self.assertEqual((a, b), (-1, 0))                                 # four scans on: it was B's
        self.assertAlmostEqual(ya, 0.7, delta=0.03)                       # A was not dragged
        self.assertEqual(ids, sorted(ab))                                 # the same two objects, the same ids
        info = tr.hyp.info()
        self.assertEqual((info["objects"], info["n_scan"]), (2, 5))
        self.assertGreater(info["assignments"], info["scans"])           # more than one reading was weighed
        first, a, b, ya, ids, ab, tr = run({"hyp_n_scan": 0, "hyp_k_best": 1})
        self.assertEqual((first[:3], a, b), ((0, -1, 1), 0, -1))           # decided at once, for good
        self.assertLess(ya, 0.68)

    def test_two_riders_swapping_sides_keep_their_ids_more_often_with_hypotheses_over_scans(self):
        """Two riders 45 m ahead swap sides (2.4 m apart, passing through each other's bearing) in a microwave radar
        whose azimuth noise is 0.8 m there, with misses and clutter. Counted: how often each still has, two seconds
        after the swap, the id it had before it."""
        def run(seed, cfg):
            rng = random.Random(seed)
            rd = Radar({"tracker": cfg})
            v, first, last = 12.0, {}, {}
            for i in range(181):
                t = 50 + i * 0.05
                s = t - 50
                rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": v, "yaw_rate_rps": 0.0})
                u = min(1.0, max(0.0, (s - 3.0) / 3.0))                    # they swap sides between 3 s and 6 s
                yy = 1.2 * math.cos(math.pi * u)
                tr = {"a": (45.0, yy), "b": (46.0, -yy)}
                if i % 2 == 0:
                    dets = [{"range_m": math.hypot(x - 2.0, y) + rng.gauss(0, 0.25), "range_rate_mps": rng.gauss(0, 0.15),
                             "azimuth_deg": math.degrees(math.atan2(y, x - 2.0)) + rng.gauss(0, 1.0), "class": "bike", "prob": 0.9}
                            for x, y in tr.values() if rng.random() >= 0.15]
                    dets += [{"range_m": rng.uniform(38, 52), "azimuth_deg": rng.uniform(-6, 6), "range_rate_mps": rng.gauss(0, 3)}
                             for _ in range(rng.choice([0, 1, 1, 2]))]     # and clutter
                    rd.on_objects({"type": "objects", "kind": "radar", "sensor": "front", "ts_mono_ns": int(t * NS),
                                   "mount": {"x_m": 2.0, "y_m": 0.0, "yaw_deg": 0.0}, "objects": dets})
                m = rd.frame(int(t * NS))
                if 2.0 <= s <= 2.9 or s >= 8.0:
                    for n, (x, y) in tr.items():
                        near = [(math.hypot(o["x_m"] - x, o["y_m"] - y), o["id"]) for o in m["objects"] if o["state"] != "memory"]
                        near = [q for q in near if q[0] < 1.0]
                        if near:
                            (first if s < 3.0 else last)[n] = min(near)[1]
            return sum(1 for n in "ab" if n in first and last.get(n) == first[n])
        seeds = range(60)
        multi = sum(run(seed, {}) for seed in seeds)
        single = sum(run(seed, {"hyp_n_scan": 0, "hyp_k_best": 1}) for seed in seeds)
        self.assertGreater(multi, single + 3, (multi, single))            # (100 seeds: 182 against 163 of 200)
        self.assertGreater(multi, 0.85 * 2 * len(seeds))

    def test_the_hypotheses_can_be_written_down_and_restored(self):
        rd, _ = world(10.0, {"p": (50.0, -3.0, 0.0, 0.0), "q": (52.0, -3.3, 0.0, 0.0), "lead": (30.0, 0.2, 9.0, 0.0)}, 3.0, seed=3)
        H = rd.tracker.hyp
        d = H.dump()
        H2 = M.NR.Hyp(1, 1, 1, 1, 1.0, 1, 1.0, 1)
        H2.load(d)
        self.assertEqual(H2.dump(), d)
        for o in rd.tracker.objects:
            self.assertEqual(H2.chain(o.h), H.chain(o.h))
            self.assertEqual(H.chain(o.h)[0], o.slot)                     # the object's numbers are its primary's
            self.assertEqual(H.rows(o.h)[0], o.slot)
            self.assertLessEqual(sum(p for _, p in H.leaves(o.h)), 1.0 + 1e-9)
            self.assertEqual(len(H.leaves(o.h)), o.nleaf)
        info = H.info()
        self.assertEqual(info["objects"], len(rd.tracker.objects))
        self.assertEqual(info["rows"] - info["rows_free"], sum(len(H.rows(o.h)) for o in rd.tracker.objects))

    def test_one_box_is_not_an_object(self):
        rd = Radar()
        for i in range(40):
            t = 10.0 + i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 10.0, "yaw_rate_rps": 0.0})
            if i == 4:
                rd.on_road_users({"type": "road_users", "sensor": "cam", "ts_mono_ns": int(t * NS), "objects": [
                    {"class": "person", "class_conf": 0.5, "bearing_deg": 5.0, "bearing_std_deg": 0.5, "range_m": 30.0, "range_std_m": 5.0}]})
            elif i % 4 == 0:
                rd.on_road_users({"type": "road_users", "sensor": "cam", "ts_mono_ns": int(t * NS), "objects": []})
            self.assertEqual(rd.frame(int(t * NS))["objects"], [])
        self.assertEqual(rd.health()["objects"], 0)                        # and it is forgotten


F_PX, CX, CY, HC = 975.0, 640.0, 360.0, 1.22


def cam_boxes(t, things, hide=()):
    """A `road_users` message of a level camera at the car frame's origin: things {name: (x of the near face, y left,
    width, height, class)} as boxes standing on the road."""
    objs = []
    for name, (x, y, w, h, cls) in things.items():
        if name in hide:
            continue
        u, hw = CX - F_PX * y / x, 0.5 * F_PX * w / x
        objs.append({"class": cls, "class_conf": 0.8, "bearing_deg": math.degrees(math.atan2(y, x)), "bearing_std_deg": 0.3,
                     "range_m": x, "range_std_m": 0.05 * x, "box": [u - hw, CY + F_PX * (HC - h) / x, u + hw, CY + F_PX * HC / x]})
    return {"type": "road_users", "sensor": "cam", "ts_mono_ns": int(t * NS), "objects": objs,
            "coverage": {"bearing_min_deg": -33, "bearing_max_deg": 33, "ground_min_m": 8.0}}


class Seeing(unittest.TestCase):
    """What a sensor can see of an object (occlusion, the field's edge) and the object's extent."""

    def hidden_car(self, cfg):
        """A van 14 m ahead and a car 32 m ahead, both at the car's speed; from 3 s to 7 s the van drives in front
        of the car (no box of the car in that time)."""
        rd = Radar({"tracker": cfg})
        out = []
        for i in range(190):
            t, s = 20 + i * 0.05, i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 12.0, "yaw_rate_rps": 0.0})
            yv = 3.0 if s < 1.5 else max(0.0, 3.0 - 2.0 * (s - 1.5)) if s < 3.0 else 0.0 if s < 7.0 else min(3.0, 2.0 * (s - 7.0))
            things = {"van": (14.0, yv, 2.0, 2.4, "truck"), "car": (32.0, 0.2, 1.8, 1.5, "car"),
                      "far": (75.0, -9.0, 1.8, 1.5, "car")}            # (the detector boxes things far smaller than the car)
            if i % 4 == 0:
                rd.on_road_users(cam_boxes(t, things, hide=("car",) if 3.0 <= s <= 7.0 else ()))
            m = rd.frame(int(t * NS))
            car = [o for o in m["objects"] if abs(o["x_m"] - 32.0) < 6 and abs(o["y_m"] - 0.2) < 2.0]
            van = [o for o in m["objects"] if abs(o["x_m"] - 14.0) < 3]
            out.append((s, car[0] if car else None, van[0]["id"] if van else None))
        return out

    def test_an_object_hidden_behind_another_is_kept_and_found_again_as_itself(self):
        out = self.hidden_car({})
        before = [c for s, c, v in out if 2.0 < s < 2.9]
        hidden = [(s, c, v) for s, c, v in out if 3.6 < s < 6.9]
        after = [c for s, c, v in out if s > 8.0]
        self.assertTrue(all(c is not None for c in before) and all(c is not None for s, c, v in hidden) and all(c is not None for c in after))
        cid = before[0]["id"]
        self.assertEqual({c["id"] for s, c, v in hidden} | {c["id"] for c in after}, {cid})   # one object throughout
        for s, c, v in hidden:
            self.assertEqual((c["occ"], c["occ_by"]), ("object", v))        # hidden, and by what
            self.assertLess(c["vis"], 0.1)
            self.assertGreater(c["exist"], 0.95)                           # not being seen there says nothing
            self.assertEqual(c["state"], "coast")
        sd = [c["std"][0] for s, c, v in hidden]
        self.assertGreater(sd[-1], 1.5 * sd[0])                            # where it is, is known less and less
        self.assertTrue(all(c["state"] == "confirmed" and c["occ"] is None and c["vis"] > 0.9 for c in after[4:]))
        self.assertLess(after[-1]["std"][0], sd[-1] / 1.5)

    def test_without_the_visibility_model_the_hidden_object_is_lost(self):
        out = self.hidden_car({"occlusion": False})
        before = [c for s, c, v in out if 2.0 < s < 2.9]
        after = [c for s, c, v in out if s > 8.5]
        self.assertTrue(all(c is not None and c["occ"] is None for c in before))
        self.assertTrue(any(c is None for s, c, v in out if 6.0 < s < 7.0))      # dropped while hidden: "not seen"
        self.assertTrue(all(c is None or c["id"] != before[0]["id"] for c in after))

    def test_an_object_cut_by_the_edge_of_the_field_is_not_held_against(self):
        """A parked car passed by: as it leaves the picture its box is cut and then gone; its existence does not fall
        while part of it is outside the field."""
        rd = Radar()
        seen = []
        for i in range(160):
            t, s = 20 + i * 0.05, i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 8.0, "yaw_rate_rps": 0.0})
            x = 40.0 - 8.0 * s                                             # its near face; 6 m to the right
            edge = math.degrees(math.atan2(6.0 + 0.9, x)) if x > 0.5 else 90.0     # its far side's bearing
            if i % 4 == 0:                                                 # (its box: as wide as it looks from there)
                b = math.atan2(6.0, max(x, 0.5))
                rd.on_road_users(cam_boxes(t, {"p": (max(x, 0.5), -6.0, 1.8 * math.cos(b) + 4.5 * math.sin(b), 1.5, "car")},
                                           hide=("p",) if edge > 33.0 or x < 8.0 else ()))
            m = rd.frame(int(t * NS))
            seen.extend((s, o) for o in m["objects"])
        cut = [(s, o) for s, o in seen if o["occ"] == "edge"]
        self.assertTrue(cut)
        self.assertTrue(all(o["exist"] > 0.9 and o["vis"] < 0.8 for s, o in cut))
        self.assertEqual(len({o["id"] for s, o in seen}), 1)
        self.assertEqual(seen[-1][1]["state"], "memory")                   # passed: remembered where it stands
        self.assertEqual(seen[-1][1]["occ"], "fov")

    def rectangle(self, heading_deg, speed, T=6.0, start=(22.0, -6.0), seed=4):
        """A car 4.6 x 1.9 m in an extended microwave radar's view (several returns per scan on the sides that face
        the sensor), driving at `speed` along `heading_deg` (ground). Returns the last messages' object and truth."""
        from chris_integrator.sim.radar import RadarSim
        sim = RadarSim(extended=True)
        rng = random.Random(seed)
        rd = Radar()
        h = math.radians(heading_deg)
        out = None
        for i in range(int(T / 0.05) + 1):
            t, s = 30 + i * 0.05, i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 0.0, "yaw_rate_rps": 0.0, "standstill": True})
            x, y = start[0] + speed * s * math.cos(h), start[1] + speed * s * math.sin(h)
            pts = sim._returns({"x": x, "y": y, "heading": h, "L": 4.6, "W": 1.9, "kind": "car"}, rng, 2.0, 0.0)
            dets = []
            for px, py in pts:
                dx, dy = px - 2.0, py
                r = math.hypot(dx, dy)
                rr = speed * (math.cos(h) * dx + math.sin(h) * dy) / r
                dets.append({"range_m": r + rng.gauss(0, 0.15), "azimuth_deg": math.degrees(math.atan2(dy, dx) + rng.gauss(0, 0.004)),
                             "range_rate_mps": rr + rng.gauss(0, 0.1)})
            rd.on_objects({"type": "objects", "kind": "radar", "sensor": "front", "extended": True, "ts_mono_ns": int(t * NS),
                           "mount": {"x_m": 2.0, "y_m": 0.0, "yaw_deg": 0.0}, "objects": dets})
            m = rd.frame(int(t * NS))
            if s > T - 1.0:
                out = (m, x, y)
        return out

    def test_an_extended_radars_returns_give_one_object_with_its_size_heading_and_centre(self):
        for heading, speed, start in ((90.0, 4.0, (22.0, -9.0)), (0.0, 6.0, (12.0, 3.0)), (35.0, 5.0, (14.0, -8.0))):
            m, x, y = self.rectangle(heading, speed, start=start)
            objs = [o for o in m["objects"] if o["state"] == "confirmed"]
            self.assertEqual(len(objs), 1, (heading, [(o["id"], o["x_m"], o["y_m"]) for o in objs]))   # not one per side
            o = objs[0]
            self.assertLess(math.hypot(o["ctr"][0] - x, o["ctr"][1] - y), 1.0, heading)      # the centre, not the near side
            self.assertLess(abs(o["len"] - 4.6), 2.0 if heading == 0.0 else 1.2, (heading, o["len"]))   # (from behind: hardly seen)
            self.assertLess(abs(o["wid"] - 1.9), 0.9, (heading, o["wid"]))                   # (side-on the width hardly shows)
            dh = (o["hdg"] - math.radians(heading) + math.pi) % (2 * math.pi) - math.pi
            self.assertLess(abs(dh), 0.2, heading)                                           # the way it drives
            self.assertLess(o["ext_std"][2], 0.2)
            self.assertLess(math.hypot(o["x_m"] - x, o["y_m"] - y), 3.0)                     # the near face is on it

    def test_the_heading_follows_the_motion_and_is_held_when_the_object_stops(self):
        rd = Radar()
        hdg = []
        for i in range(240):
            t, s = 20 + i * 0.05, i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 0.0, "yaw_rate_rps": 0.0, "standstill": True})
            v = 6.0 if s < 5.0 else max(0.0, 6.0 - 3.0 * (s - 5.0))       # crossing from the right, then it stops
            y = -14.0 + (6.0 * min(s, 5.0) + (6.0 * (min(s, 7.0) - 5.0) - 1.5 * (min(s, 7.0) - 5.0) ** 2 if s > 5.0 else 0.0))
            if i % 2 == 0:
                az = math.atan2(y, 28.0)
                rd.on_objects({"type": "objects", "kind": "radar", "sensor": "front", "ts_mono_ns": int(t * NS),
                               "mount": {"x_m": 2.0, "y_m": 0.0, "yaw_deg": 0.0},
                               "objects": [{"range_m": math.hypot(28.0, y), "azimuth_deg": math.degrees(az),
                                            "range_rate_mps": v * math.sin(az), "class": "car", "prob": 0.9}]})
            m = rd.frame(int(t * NS))
            hdg.extend((s, o["hdg"], o["heading_rad"], o["motion"], o["ext_std"][2]) for o in m["objects"])
        moving = [h for h in hdg if 3.0 < h[0] < 5.0]
        self.assertTrue(moving and all(abs(h[1] - math.pi / 2) < 0.15 for h in moving))      # across, to the left
        stopped = [h for h in hdg if h[0] > 10.0]
        self.assertTrue(stopped and all(h[2] is None for h in stopped[-20:]))               # no direction of travel now
        self.assertTrue(all(abs(h[1] - math.pi / 2) < 0.25 for h in stopped))               # ... but it still points there
        self.assertLess(stopped[-1][4], 0.3)

    def test_a_camera_box_seen_side_on_is_a_long_object_and_its_place_is_its_centre(self):
        rd = Radar()
        last = None
        for i in range(120):
            t = 20 + i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 0.0, "yaw_rate_rps": 0.0, "standstill": True})
            if i % 4 == 0:      # a car standing across the road 20 m ahead (4.5 m of it across the picture), one end-on beside it
                rd.on_road_users(cam_boxes(t, {"side": (20.0, 0.0, 4.5, 1.5, "car"), "end": (20.0, 6.0, 1.8, 1.5, "car")}))
            last = rd.frame(int(t * NS))
        side, end = nearest(last, 20.0, 0.0, 4.0), nearest(last, 20.0, 6.0, 4.0)
        self.assertLess(abs(abs(side["hdg"]) - math.pi / 2), 0.5)          # it points across
        self.assertLess(abs(end["hdg"]), 0.5)                              # that one along
        self.assertAlmostEqual(side["ctr"][0] - side["x_m"], 0.5 * side["wid"], delta=0.35)   # half its depth behind its face
        self.assertAlmostEqual(end["ctr"][0] - end["x_m"], 0.5 * end["len"], delta=0.35)
        self.assertGreater(side["len"], 3.5)
        self.assertLess(side["wid"], 2.4)
        self.assertLess(abs(end["wid"] - 1.8), 0.4)                        # its width is what the box shows


class Sizes(unittest.TestCase):
    def test_a_van_called_car_then_truck_is_ranged_by_the_mixture_of_the_two_heights(self):
        """A van (2.1 x 2.6 m) standing 55 m ahead, its first box called `car`, then `truck` in two boxes of three
        (the detector's confusion): the object is born with a car's height and would be read at the range a car of
        that box height has (0.6 of the truth, the box height being the strongest range there: 37.6 m, 6.5 m
        stated, 2.7 sigma off). The height prior follows the vote (tracker._size): within its stated uncertainty."""
        rd = Radar()
        cls = ["car", "truck", "truck", "car", "truck", "truck", "truck", "car", "truck", "truck"] * 3
        k, m = 0, None
        for i in range(120):
            t = 20 + i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 0.0, "yaw_rate_rps": 0.0})
            if i % 4 == 0:
                rd.on_road_users(cam_boxes(t, {"van": (55.0, 1.0, 2.1, 2.6, cls[k])}))
                k += 1
            m = rd.frame(int(t * NS))
        (o,) = m["objects"]
        self.assertEqual(o["class"]["label"], "truck")
        self.assertLess(abs(o["x_m"] - 55.0), 1.5 * o["std"][0])
        self.assertGreater(o["x_m"], 42.0)


class RoadAxis(unittest.TestCase):
    def test_the_road_decides_which_side_of_the_line_of_sight_a_standing_car_points(self):
        """radar_heading_prior on one row: a car parked along a bend (the road's axis 0.34 rad), seen at 0.07 rad,
        held by its box widths at the mirror heading -0.21 (2 x 0.07 - 0.35: the same |sin a|) is turned to the
        mirror nearer the axis and pulled to the axis to 0.1 rad at most; a car parked across the road (1.3 rad) is
        left as it is; one whose motion has shown its front is not touched."""
        from chris_integrator.radar import tracker as TK
        (e0, e1), (p0, p1) = TK.OFF["e"], TK.OFF["Pe"]
        m0 = TK.OFF["m"][0]
        a = np.zeros((3, TK.ROW))
        for k, h in enumerate((-0.21, 1.3, -0.21)):
            a[k, e0:e1] = (4.5, 1.8, h)
            a[k, p0:p1] = np.diag([0.5, 0.05, 0.002]).ravel()
            a[k, p0 + 2] = a[k, p0 + 6] = 0.01                  # length and heading correlated
        a[2, m0 + TK.M_SENSE] = 1.0
        n = M.NR.radar_heading_prior(a, [0, 1, 2], 0.34, 0.07, 0.1, 0.7, 0.25)
        self.assertEqual(n, 1)
        self.assertLess(abs(a[0, e0 + 2] - 0.34), 0.03)
        self.assertLess(a[0, p0 + 8], 0.0101)
        self.assertLess(a[0, p0 + 2], 0.0)                      # the reflection turned the correlation's sign
        self.assertEqual(a[1, e0 + 2], 1.3)
        self.assertEqual(a[2, e0 + 2], -0.21)

    def test_a_car_parked_on_a_bend_takes_the_roads_axis(self):
        """The car drives round a bend (R 110 m) past a car parked along it, the road model's path given: the
        parked car's heading is the road's there, not the car's own when it was first seen."""
        rd = Radar()
        R, v = 110.0, 13.0
        th = 70.0 / R                                            # the parked car 70 m along the bend, 3 m outside
        px, py = (R + 3.0) * math.sin(th), R - (R + 3.0) * math.cos(th)
        best = None
        for i in range(60):
            t = 20 + i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": v, "yaw_rate_rps": v / R})
            s0 = v * (t - 20)
            a0 = s0 / R
            ex, ey, eh = R * math.sin(a0), R - R * math.cos(a0), a0
            xs = [float(k) for k in range(0, 100, 5)]
            # the path in the car frame: the circle ahead of the car (the road model's y is right)
            pl = {"x": [R * math.sin(k / R) for k in xs], "y": [-(R - R * math.cos(k / R)) for k in xs]}
            rd.on_road({"ts_mono_ns": int(t * NS), "plan": pl})
            c, s = math.cos(eh), math.sin(eh)
            dx, dy = px - ex, py - ey
            x, y = c * dx + s * dy, -s * dx + c * dy
            if i % 4 == 0 and x > 8:
                rd.on_road_users(cam_boxes(t, {"p": (x - 2.0, y, 2.4, 1.5, "car")}))
            m = rd.frame(int(t * NS))
            if m and m["objects"]:
                best = (m["objects"][0], eh)
        o, eh = best
        self.assertLess(abs(((o["hdg"] - (th - eh)) + math.pi / 2) % math.pi - math.pi / 2), 0.15)


class PastAndStructure(unittest.TestCase):
    def test_the_past_is_smoothed_and_the_present_is_not_delayed(self):
        """A car ahead brakes from 14 to 6 m/s (microwave radar, noisy range rate). The smoothed velocity `vs` is of
        half a second ago and nearer the truth of then than the filtered velocity was at that time; the filtered
        velocity is of now."""
        rng = random.Random(3)
        rd = Radar()
        x, filt, smooth, hist = 40.0, {}, [], None
        for i in range(200):
            t, s = 20 + i * 0.05, i * 0.05
            v = 14.0 if s < 3.0 else max(6.0, 14.0 - 4.0 * (s - 3.0))
            x += (v - 14.0) * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 14.0, "yaw_rate_rps": 0.0})
            if i % 2 == 0:
                rd.on_objects({"type": "objects", "kind": "radar", "sensor": "front", "ts_mono_ns": int(t * NS),
                               "mount": {"x_m": 2.0, "y_m": 0.0, "yaw_deg": 0.0},
                               "objects": [{"range_m": x - 2.0 + rng.gauss(0, 0.4), "azimuth_deg": rng.gauss(0, 0.3),
                                            "range_rate_mps": v - 14.0 + rng.gauss(0, 0.8), "class": "bike", "prob": 0.9}]})
            m = rd.frame(int(t * NS))
            for o in m["objects"]:
                filt[round(s, 2)] = (o["vx_mps"], v)
                if o["vs"] is not None and s > 2.0:
                    then = round(s - o["vs_age_s"], 2)
                    v_then = 14.0 if then < 3.0 else max(6.0, 14.0 - 4.0 * (then - 3.0))
                    f_then = min(filt.items(), key=lambda kv: abs(kv[0] - then))[1][0]
                    smooth.append((abs(o["vs"][0] - v_then), abs(f_then - v_then), o["vs_age_s"], abs(o["vx_mps"] - v)))
            hist = rd.tracker.objects[0].history if rd.tracker.objects else hist
        self.assertGreater(len(smooth), 100)
        es, ef = sum(a for a, _, _, _ in smooth) / len(smooth), sum(b for _, b, _, _ in smooth) / len(smooth)
        self.assertLess(es, 0.75 * ef, (es, ef))                           # the smoothed past beats the filtered past
        self.assertTrue(all(0.3 <= a <= 0.75 for _, _, a, _ in smooth))    # ... and is half a second old
        self.assertLess(sum(d for _, _, _, d in smooth) / len(smooth), 1.2)   # the filtered present follows the braking
        ts = [h[0] for h in hist]
        self.assertEqual(ts, sorted(ts))                                   # the trajectory: in order, smoothed
        self.assertGreater(len(ts), 20)

    @staticmethod
    def scene(things, v_ego=0.0, T=6.0, cfg=None, hide=lambda s: ()):
        rd = Radar({"tracker": cfg or {}})
        m = None
        for i in range(int(T / 0.05)):
            t, s = 20 + i * 0.05, i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": v_ego, "yaw_rate_rps": 0.0, "standstill": v_ego == 0.0})
            if i % 4 == 0:
                rd.on_road_users(cam_boxes(t, {k: (x - v_ego * s + (vx * s), y, w, h, c) for k, (x, y, w, h, c, vx) in things.items()},
                                           hide=hide(s)))
            m = rd.frame(int(t * NS))
        return rd, m

    def test_a_row_of_parked_cars_a_queue_and_a_platoon_are_groups(self):
        row = {"p%d" % k: (14.0 + 6.5 * k, -4.0 - 0.6 * k, 1.8, 1.5, "car", 0.0) for k in range(4)}    # beside the road
        rd, m = self.scene(dict(row, lone=(30.0, 7.0, 1.8, 1.5, "car", 0.0)), v_ego=0.0)
        by = {round(o["y_m"]): o for o in m["objects"]}
        self.assertEqual(len(m["groups"]), 1)
        g = m["groups"][0]
        self.assertEqual((g["kind"], len(g["members"])), ("row", 4))
        members = [o for o in m["objects"] if o["grp"] == g["id"]]
        self.assertEqual(sorted(o["id"] for o in members), sorted(g["members"]))
        self.assertTrue(all(o["grp_kind"] == "row" and o["motion"] == "stationary" for o in members))
        self.assertIsNone(by[7]["grp"])                                    # the one across the road is by itself
        # two cars standing ahead in the car's own lane: a queue
        rd, m = self.scene({"q1": (12.0, 0.1, 1.8, 1.5, "car", 0.0), "q2": (30.0, 4.2, 1.8, 1.5, "car", 0.0),
                            "q3": (19.0, 0.2, 2.0, 2.4, "truck", 0.0)}, v_ego=0.0)
        kinds = {o["grp_kind"] for o in m["objects"] if abs(o["y_m"]) < 1.0}
        self.assertEqual(kinds, {"queue"})
        # three cars driving ahead at the car's speed: a platoon
        rd, m = self.scene({"a": (15.0, 0.0, 1.8, 1.5, "car", 10.0), "b": (33.0, 3.5, 1.8, 1.5, "car", 10.0),
                            "c": (52.0, 7.5, 1.8, 1.5, "car", 10.0)}, v_ego=10.0, T=9.0)
        self.assertEqual({o["grp_kind"] for o in m["objects"]}, {"platoon"})
        self.assertEqual(len(m["groups"][0]["members"]), 3)
        rd, m = self.scene(row, cfg={"groups": False})
        self.assertEqual((m["groups"], {o["grp"] for o in m["objects"]}), ([], {None}))

    def test_a_row_holds_its_members_in_line_and_standing_but_not_one_that_pulls_out(self):
        row = {"p%d" % k: (14.0 + 6.5 * k, -4.0, 1.8, 1.5, "car", 0.0) for k in range(5)}
        res = {}
        for name, cfg in (("with", {}), ("without", {"group_support": False})):
            # (6 s: the fifth car is 65 % hidden behind the fourth, its existence grows at the occlusion floor; its
            # object is grouped from 4.5-5.5 s on, as the scene's chance has it)
            rd, m = self.scene(row, v_ego=0.0, T=6.0, cfg=cfg)
            objs = [o for o in rd.tracker.objects if o.grp is not None]
            self.assertEqual(len(objs), 5, name)
            res[name] = (sum(float(o.mu[0]) for o in objs) / 5, max(abs(float(o.xs[0][1]) + 4.0 - 0.0) for o in objs),
                         max(math.sqrt(float(o.Ps[0][1, 1])) for o in objs))
        self.assertGreater(res["with"][0], res["without"][0])              # standing is believed more
        self.assertLess(res["with"][2], res["without"][2])                 # their places across the line are known better
        # the third car pulls out of the row at 3 s and drives off across the road: found moving, and out of the group
        rd = Radar()
        seen = []
        for i in range(200):
            t, s = 20 + i * 0.05, i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 0.0, "yaw_rate_rps": 0.0, "standstill": True})
            if i % 4 == 0:
                things = {k: v[:5] for k, v in row.items()}
                things["p2"] = (27.0, -4.0 + 2.5 * max(0.0, s - 3.0), 1.8, 1.5, "car")
                rd.on_road_users(cam_boxes(t, things))
            m = rd.frame(int(t * NS))
            seen.extend((s, o["motion"], o["grp"], o["y_m"]) for o in m["objects"] if abs(o["x_m"] - 27.0) < 3.0)
        early = [q for q in seen if 2.0 < q[0] < 3.0]
        late = [q for q in seen if q[0] > 6.5]
        self.assertTrue(early and all(q[2] is not None and q[1] == "stationary" for q in early))
        self.assertTrue(late and all(q[2] is None and q[1] == "moving" for q in late), late[:3])
        self.assertGreater(late[-1][3], 3.0)                               # it has left for the other side


class ParkedCars(unittest.TestCase):
    """The defect of the real drives (docs/RADAR.md, section 5): parked cars called moving."""

    @staticmethod
    def drive(seed, cam_cfg, sigma_deg=0.6, tau=1.0):
        """The car passes six parked cars at 9 m/s while its camera's pitch moves as real ones do (a Gauss-Markov
        process, 0.6 degrees, 1 s: every box of a frame moves up or down together, and a ground range read through
        it moves by metres). Seconds of the verdicts moving / stationary over the parked cars."""
        rng = random.Random(seed)
        rd = Radar({"camera": cam_cfg})
        p = mv = st = 0.0
        cars = [(45.0 + 18.0 * k, -3.2 if k % 2 else 3.4) for k in range(6)]
        for i in range(300):
            t, s = 20 + i * 0.05, i * 0.05
            rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 9.0, "yaw_rate_rps": 0.0})
            rho = math.exp(-0.05 / tau)
            p = rho * p + math.sqrt(1 - rho * rho) * rng.gauss(0, math.radians(sigma_deg))
            if i % 4 == 0:
                things = {k: (gx - 9.0 * s, gy, 1.8, 1.5, "car") for k, (gx, gy) in enumerate(cars)
                          if 9.0 < gx - 9.0 * s < 70.0 and rng.random() > 0.3}
                m = cam_boxes(t, things)
                for o in m["objects"]:
                    o["box"][1] += F_PX * p
                    o["box"][3] += F_PX * p
                    o["range_m"] = HC * F_PX / max(o["box"][3] - CY, 1.0)      # the ground point as projected
                rd.on_road_users(m)
            for o in rd.frame(int(t * NS))["objects"]:
                if o["state"] in ("confirmed", "coast"):
                    mv += 0.05 * (o["motion"] == "moving")
                    st += 0.05 * (o["motion"] == "stationary")
        return mv, st

    def test_a_pitching_camera_does_not_make_parked_cars_drive(self):
        seeds = range(6)
        with_model = [self.drive(s, {}) for s in seeds]
        white_only = [self.drive(s, {"pitch_gm_deg": 0.0, "pitch_gm_stand_deg": 0.0}) for s in seeds]
        self.assertLess(sum(a for a, _ in with_model), 1.0)                # (12 seeds: 0.0 s of 406 s)
        self.assertGreater(sum(b for _, b in with_model), 150.0)
        self.assertGreater(sum(a for a, _ in white_only), 5.0)             # the defect, reproduced (12 seeds: 30 s)


class PrimaryAfterDrop(unittest.TestCase):
    def test_an_objects_numbers_are_its_primarys_after_another_object_is_taken_out(self):
        """An object that is dropped (or merged into another) settles its cluster anew, and a cluster-mate's primary
        may then be another of its leaves. The object's numbers must be that leaf's from then on (Hyp.sync in
        Tracker._drop), not the former primary's until its next scan: on this drive 16 object-messages read a row
        that was no longer the primary's."""
        from chris_integrator.radar import box
        from chris_integrator.sim import radar_score
        frame, seen = box.Radar.frame, {"objects": 0, "stale": 0, "drops": 0}

        def checked(rd, ts):
            m = frame(rd, ts)
            tr = rd.tracker
            for o in tr.objects:
                seen["objects"] += 1
                seen["stale"] += tr.hyp.rows(o.h)[0] != o.slot
            return m
        box.Radar.frame = checked
        try:
            radar_score.run("parked_street", seed=1)
        finally:
            box.Radar.frame = frame
        self.assertGreater(seen["objects"], 10000)
        self.assertEqual(seen["stale"], 0)


if __name__ == "__main__":
    unittest.main()
