"""The display channel's road users: where they are drawn and which way they face (drive of 2026-11-26/27).

What the owner saw on the car's screen: a vehicle in the lane to the left, and oncoming traffic, drawn partly in
the car's own lane; oncoming and crossing traffic drawn from behind. The radar's place of an object is its NEAR
point (where the line of sight to its centre enters its rectangle): for a vehicle beside the car's lane that is
its corner towards the car, and a screen that draws a vehicle's width centred on it moves the vehicle 0.3-0.9 m
into the car's lane. And `view` carried no heading at all. Here: the simulator shows each case (a car in the next
lane, oncoming cars, crossing cars) and the view's `objects` are held to the truth: the body's middle, and the
heading with how it is known.

The detector and the tracker are the phone's (the simulator's preset `detin_pose_scale`, the tracker knobs of the
phone's radar.json).
"""
import math
import os
import statistics
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chris_integrator.mixer import MessageClock  # noqa: E402
from chris_integrator.sim import default_sensors  # noqa: E402
from chris_integrator.sim import roaduser_sim as rs  # noqa: E402
from chris_integrator.stack import build  # noqa: E402
from chris_integrator.trackers.roadusers_view import RadarDisplay, body_place  # noqa: E402

NS = 1_000_000_000
PHONE = {"in_app": True, "in_width": True, "in_lamp_size": True, "in_scale": True, "in_pose_gn": True,
         "in_pose_gc": True, "in_pose_yaw": True, "scale_iter": True, "scale_exact": True}
LANE_W = 3.5
_RUNS = {}


def run(world, place, seed=1):
    """A scenario (name, or a spec) through the whole stack: (world, the `view` messages)."""
    key = (world if isinstance(world, str) else id(world), place, seed)
    if key in _RUNS:
        return _RUNS[key]
    w = rs.scenario(world, seed=seed) if isinstance(world, str) else rs.ScenarioWorld(world, seed=seed, name="custom")
    det = rs.DetectorSim(seed=1000 + seed, night=False, params=rs.det_params_arg("detin_pose_scale"))
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
    sensors.append(det)
    st = build(cfg={"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}, "radar": {"tracker": PHONE},
                    "view": {"radar": {"place": place}}}, mapper=None, clock=MessageClock())
    views = []
    st.mixer.add_output("test", lambda e: views.append(e.msg), streams=("view",))
    sent = False
    for _, m in w.messages(sensors):
        if not sent:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            sent = True
        st.deliver(m)
    _RUNS[key] = (w, views)
    return w, views


def matched(w, views, role, x_max=45.0):
    """(truth actor, view object) per frame: the view's road user at the actor's bearing and about its range (by
    the radar's near point, which both placements carry or are)."""
    out = []
    for m in views:
        t = m["ts_mono_ns"] / 1e9
        for a in w.actors_at(t):
            if not a["role"].startswith(role) or not 4.0 < a["x"] < x_max:
                continue
            best = None
            for o in m["objects"]:
                if o.get("kind") != "road_user":
                    continue
                n = o.get("near") or (o["x"], o["y"])
                db = abs(math.atan2(-n[1], n[0]) - math.atan2(a["ref_y"], a["ref_x"]))
                if db < math.radians(3.0) and abs(n[0] - a["ref_x"]) < 0.4 * a["ref_x"] + 2.0 and (best is None or db < best[0]):
                    best = (db, o)
            if best is not None:
                out.append((a, best[1]))
    return out


def towards_own_lane(rows):
    """Per row: how far the drawn middle is from the true middle, positive AWAY from the car's own lane."""
    return [((-o["y"]) - a["y"]) * (1.0 if a["y"] > 0 else -1.0) for a, o in rows]


def crossing_spec(dist=35.0):
    """The car stands at a junction; cars cross `dist` m ahead, from the left and from the right, at 7-10 m/s."""
    spec = dict(T=40.0, speed=[(0, 8.0), (4, 8.0), (7, 0.0)], curv=[(0, 0.0)], lead_range_m=100.0, accel_lim=2.5)

    def actors(w):
        i_stop = int(np.argmax((w.t - w.t0 > 8.0) & (w.v < 0.05)))
        s_cross = float(w.s[i_stop]) + rs.CAM_TO_AXLE + dist
        a = []
        for k, (t, side, v) in enumerate(((9.0, 1, 8.0), (15.0, -1, 9.0), (21.0, 1, 7.0), (27.0, -1, 10.0), (32.0, 1, 8.5))):
            a.append(rs.Actor(400 + k, "car", "crossing", s_cross + (2.0 if side < 0 else -2.0), speed=[(0, 0.0)],
                              lateral=[(0, side * 40.0), (t, side * 40.0), (t + 80.0 / v, -side * 40.0)],
                              t_on=t, t_off=t + 80.0 / v))
        return a
    spec["actors"] = actors
    return spec


CROSSING = crossing_spec()


class BodyPlace(unittest.TestCase):
    def test_the_middle_of_a_car_in_the_next_lane_not_its_corner(self):
        # a car 4.4 x 1.8 m in the lane to the left, its centre 10 m ahead; the radar's near point (r_near_offset:
        # where the line of sight to the centre enters the rectangle) is its rear right corner region
        ctr, ln, wd = (10.0, 3.5), 4.4, 1.8
        rc = math.hypot(*ctr)
        t = min(0.5 * ln / (ctr[0] / rc), 0.5 * wd / (ctr[1] / rc))
        near = (ctr[0] - t * ctr[0] / rc, ctr[1] - t * ctr[1] / rc)
        self.assertLess(near[1], 3.5 - 0.7)                         # the near point: 0.8 m towards the car's lane
        o = {"x_m": near[0], "y_m": near[1], "ctr": list(ctr), "len": ln, "wid": wd, "hdg": 0.0}
        x, y, h = body_place(o, near[0], -near[1])
        self.assertAlmostEqual(y, -3.5)                             # view frame: y right
        self.assertAlmostEqual(x, 10.0 - 2.2)                       # its rear face
        self.assertAlmostEqual(h, 0.0)
        # carried on to the frame's time, the middle goes with the near point
        x2, y2, _ = body_place(o, near[0] - 1.0, -near[1] + 0.2)
        self.assertAlmostEqual((x2, y2), (x - 1.0, y + 0.2))
        # an oncoming one: its front faces the car; the view's heading is positive to the RIGHT
        o = dict(o, hdg=math.pi - 0.1)
        self.assertAlmostEqual(abs(body_place(o, near[0], -near[1])[2]), math.pi - 0.1)
        self.assertLess(body_place(o, near[0], -near[1])[2], 0.0)
        # it drives to the LEFT, seen from the side (its near point then lies on that side)
        o = dict(o, hdg=math.pi / 2, x_m=9.1, y_m=3.185)
        x, y, h = body_place(o, 9.1, -3.185)
        self.assertAlmostEqual(h, -math.pi / 2)
        self.assertAlmostEqual(x, 10.0 - 0.9)                       # its side is the nearest edge
        # no rectangle in the message: the near point, no heading
        self.assertEqual(body_place({"x_m": 5.0, "y_m": 1.0}, 5.0, -1.0), (5.0, -1.0, None))

    def _obj(self, **kw):
        o = {"id": 1, "state": "confirmed", "x_m": 8.6, "y_m": 2.6, "vx_mps": 10.0, "vy_mps": 0.0, "v_rel_mps": 0.0,
             "std": [1.0, 0.3, 1.0, 0.3], "age_s": 3.0, "seen": {"cam": 0.1}, "motion": "unknown", "speed_mps": 10.0,
             "flags": [], "ttc_s": None, "class": {"label": "car", "confidence": 0.8, "source": "cam", "votes": 5},
             "ctr": [10.8, 3.5], "len": 4.4, "wid": 1.8, "hdg": 0.0, "ext_std": [0.5, 0.1, 0.1]}
        o.update(kw)
        return o

    def test_the_view_gives_the_body_the_heading_and_how_it_is_known(self):
        msg = {"type": "radar", "ts_mono_ns": 10 * NS, "ego": {"speed_mps": 10.0}, "objects": [self._obj()]}
        rd = RadarDisplay()
        rd.update(msg)
        o = rd.objects(10 * NS, None)[0][0]
        self.assertEqual((o["x"], o["y"]), (8.6, -3.5))             # the rear face, the middle of the body
        self.assertEqual(o["near"], [8.6, -2.6])                    # the radar's near point is still there
        self.assertEqual((o["hdg"], o["hdg_std"], o["len"], o["wid"]), (0.0, 0.1, 4.4, 1.8))
        self.assertEqual(o["hdg_src"], "guess")                     # not seen driving: the front is a guess
        rd.update(dict(msg, ts_mono_ns=10 * NS + NS // 10, objects=[self._obj(motion="moving")]))
        self.assertEqual(rd.objects(10 * NS + NS // 10, None)[0][0]["hdg_src"], "motion")
        rd.update(dict(msg, ts_mono_ns=10 * NS + NS // 5, objects=[self._obj(motion="stationary", speed_mps=0.0)]))
        self.assertEqual(rd.objects(10 * NS + NS // 5, None)[0][0]["hdg_src"], "motion")     # it stopped: still known
        # the old placement, by configuration: the near point, no further keys
        old = RadarDisplay({"place": "near"})
        old.update(msg)
        o = old.objects(10 * NS, None)[0][0]
        self.assertEqual((o["x"], o["y"]), (8.6, -2.6))
        self.assertNotIn("hdg", o)
        self.assertEqual(RadarDisplay().cfg["place"], "body")


class Simulated(unittest.TestCase):
    def test_a_car_in_the_next_lane_is_drawn_in_its_lane(self):
        rows_n = matched(*run("adjacent", "near"), "adjacent")
        rows_b = matched(*run("adjacent", "body"), "adjacent")
        self.assertGreater(len(rows_b), 400)
        e_n, e_b = towards_own_lane(rows_n), towards_own_lane(rows_b)
        # the near point: a third of a metre and more towards the car's own lane (the problem)
        self.assertLess(statistics.median(e_n), -0.3)
        # the body: about its middle
        self.assertLess(abs(statistics.median(e_b)), 0.25)
        self.assertGreater(sorted(e_b)[len(e_b) // 10], -0.45)
        # a 1.8 m wide vehicle drawn centred there: how often it comes within 0.45 m of the car's lane line
        # (the car in the middle of the next lane leaves 0.85 m)
        def close(rows):
            return sum(1 for a, o in rows if (-o["y"]) - 0.9 < 0.5 * LANE_W + 0.45) / len(rows)
        self.assertGreater(close(rows_n), 0.2)
        self.assertLess(close(rows_b), 0.03)
        # it drives with the car: its rear, known from its motion
        self.assertGreater(sum(1 for a, o in rows_b if abs(o["hdg"]) < math.radians(50) and o["hdg_src"] == "motion")
                           / len(rows_b), 0.95)

    def test_oncoming_cars_face_the_car_and_stay_in_their_lane(self):
        rows_n = matched(*run("oncoming", "near"), "oncoming")
        rows_b = matched(*run("oncoming", "body"), "oncoming")
        self.assertGreater(len(rows_b), 80)
        self.assertLess(statistics.median(towards_own_lane(rows_n)), -0.25)
        self.assertLess(abs(statistics.median(towards_own_lane(rows_b))), 0.2)
        front = sum(1 for a, o in rows_b if abs(o["hdg"]) > math.radians(130)) / len(rows_b)
        known = sum(1 for a, o in rows_b if abs(o["hdg"]) > math.radians(130) and o["hdg_src"] == "motion") / len(rows_b)
        self.assertGreater(front, 0.9)                              # the front points at the car
        self.assertGreater(known, 0.85)                             # ... and its motion has shown it
        self.assertFalse(any("hdg" in o for a, o in rows_n))        # (the old view said nothing: drawn from behind)

    def test_crossing_cars_show_their_side_and_which_way_they_go(self):
        w, views = run(CROSSING, "body")
        n = ok = 0
        for m in views:
            t = m["ts_mono_ns"] / 1e9
            for a in w.actors_at(t):
                if a["role"] != "crossing" or abs(a["y"]) > 15.0:
                    continue
                near = [o for o in m["objects"] if o.get("kind") == "road_user" and "hdg" in o
                        and math.hypot(o["x"] - a["x"], -o["y"] - a["y"]) < 8.0]
                if not near:
                    continue
                o = min(near, key=lambda q: math.hypot(q["x"] - a["x"], -q["y"] - a["y"]))
                n += 1
                # the view's heading is positive to the right, the truth's to the left
                err = abs(math.atan2(math.sin(-o["hdg"] - a["heading"]), math.cos(-o["hdg"] - a["heading"])))
                side = math.radians(50) < abs(o["hdg"]) < math.radians(130)
                ok += side and err < math.radians(40) and o["hdg_src"] == "motion"
        self.assertGreater(n, 100)
        self.assertGreater(ok / n, 0.9)


if __name__ == "__main__":
    unittest.main()
