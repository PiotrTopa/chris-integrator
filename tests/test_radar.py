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


if __name__ == "__main__":
    unittest.main()
