"""Unit tests of the ego-light selection (chris_integrator/trackers/lights.py).

    cd <repo> && python3 -m unittest discover -s chris-integrator/tests -v
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chris_integrator.trackers import lights  # noqa: E402

F, CX, CY, W, H = 690.0, 640.0, 360.0, 1280, 720
X = [192.0 * (i / 32) ** 2 for i in range(33)]


def road(curve=0.0):
    return {"type": "road", "plan": {"x": X, "y": [curve * x * x for x in X], "z": [0.0] * 33}}


def light(tid, cx, top, h, state="red", score=0.7, hits=5):
    return {"track_id": tid, "state": state, "score": score, "hits": hits, "flashing": False,
            "box": [cx - 0.4 * h, top, cx + 0.4 * h, top + h]}


def msg(*ls):
    return {"type": "lights", "frame": [W, H], "lights": list(ls)}


class EgoTest(unittest.TestCase):
    def ann(self, m, r=None):
        return lights.annotate(m, r if r is not None else road(), {"pitch": 0, "yaw": 0, "roll": 0, "height_m": 1.2},
                               (F, CX, CY, W, H))

    def test_distance_and_lateral(self):
        m = self.ann(msg(light(1, CX + 69, 200, 31)))      # 0.9 m housing at 31 px -> 20 m; 69 px right -> 2 m
        li = m["lights"][0]
        self.assertAlmostEqual(li["dist_m"], 20.0, delta=0.2)
        self.assertAlmostEqual(li["lateral_m"], 2.0, delta=0.2)
        self.assertTrue(li["ego"])
        self.assertEqual(m["ego"]["state"], "red")

    def test_side_street_light_not_ego(self):
        # a green light 15 m to the left (cross street) vs a red one over the lane
        m = self.ann(msg(light(1, CX - 500, 250, 31, "green"), light(2, CX + 30, 220, 31, "red")))
        self.assertEqual(m["ego"]["track_id"], 2)
        self.assertEqual(m["ego"]["state"], "red")
        self.assertEqual(m["lights"][0]["relevance"], 0.0)

    def test_back_view_ignored(self):
        m = self.ann(msg(light(1, CX + 20, 220, 31, "off")))
        self.assertLess(m["lights"][0]["relevance"], lights.MIN_RELEVANCE)
        self.assertIsNone(m["ego"])

    def test_group_vote(self):
        # two heads of one signal group (pole right, overhead) agree on green; a weak far one says red
        m = self.ann(msg(light(1, CX + 120, 230, 40, "green"), light(2, CX + 10, 150, 36, "green"),
                         light(3, CX + 60, 300, 9, "red", score=0.3, hits=2)))
        self.assertEqual(m["ego"]["state"], "green")
        self.assertEqual(m["ego"]["agree"], 2)

    def test_follows_curved_path(self):
        # path bends right: a light right of the image centre at the bend is ahead of the car
        r = road(curve=0.004)
        m = self.ann(msg(light(1, CX + 150, 230, 31, "red"), light(2, CX - 150, 230, 31, "green")), r)
        self.assertEqual(m["ego"]["track_id"], 1)

    def test_no_road(self):
        m = lights.annotate(msg(light(1, CX + 30, 220, 31, "yellow")), None, None, (F, CX, CY, W, H))
        self.assertEqual(m["ego"]["source"], "image")
        self.assertEqual(m["ego"]["state"], "yellow")


class EgoOverTimeTest(unittest.TestCase):
    """EgoLightSelector: the ego light does not blink when a frame misses it or its relevance dips."""
    CAL = {"pitch": 0, "yaw": 0, "roll": 0, "height_m": 1.2}

    def step(self, sel, t, *ls):
        m = msg(*ls)
        m["ts_mono_ns"] = int(t * 1e9)
        return sel.annotate(m, road(), self.CAL, (F, CX, CY, W, H))["ego"]

    def test_hold_bridges_missed_frames_and_ends(self):
        sel = lights.EgoLightSelector()
        e = self.step(sel, 10.0, light(1, CX + 30, 220, 31, "red"))
        self.assertEqual((e["track_id"], e["state"], e["held"]), (1, "red", False))
        e = self.step(sel, 10.33)                                   # the detector missed it
        self.assertEqual((e["track_id"], e["state"], e["held"]), (1, "red", True))
        self.assertAlmostEqual(e["age_s"], 0.33, places=2)
        e = self.step(sel, 10.67, light(1, CX + 31, 220, 32, "green"))
        self.assertEqual((e["state"], e["held"]), ("green", False))
        self.assertTrue(self.step(sel, 11.67)["held"])              # 1.0 s after the last sight: still held
        self.assertIsNone(self.step(sel, 12.0))                     # 1.33 s: gone
        self.assertIsNone(self.step(sel, 12.33))                    # and it does not come back by itself

    def test_hysteresis_keeps_the_current_ego_light(self):
        sel = lights.EgoLightSelector(hold_s=0.0)
        weak = dict(light(1, CX + 30, 220, 31, "red", score=0.7, hits=5))
        self.assertIsNotNone(self.step(sel, 10.0, weak))
        # the same light, far to the side for a frame (the path swings in a turn): under MIN_RELEVANCE, above KEEP
        side = light(1, CX + 290, 220, 31, "red")
        m = lights.annotate(msg(dict(side)), road(), self.CAL, (F, CX, CY, W, H))
        self.assertIsNone(m["ego"])
        self.assertGreaterEqual(m["lights"][0]["relevance"], lights.KEEP_RELEVANCE)
        e = self.step(sel, 10.33, dict(side))
        self.assertEqual((e["track_id"], e["held"]), (1, False))
        # another light with the same low relevance never becomes the ego light that way
        sel2 = lights.EgoLightSelector(hold_s=0.0)
        self.assertIsNone(self.step(sel2, 10.0, dict(side)))

    def test_coasting_light_counts_less_and_restart_resets(self):
        a = lights.annotate(msg(light(1, CX + 30, 220, 31)), road(), self.CAL, (F, CX, CY, W, H))["lights"][0]["relevance"]
        c = dict(light(1, CX + 30, 220, 31), coast=2)
        b = lights.annotate(msg(c), road(), self.CAL, (F, CX, CY, W, H))["lights"][0]["relevance"]
        self.assertAlmostEqual(b, a * lights.COAST_FACTOR ** 2, places=2)
        sel = lights.EgoLightSelector()
        self.assertIsNotNone(self.step(sel, 100.0, light(1, CX + 30, 220, 31)))
        self.assertIsNone(self.step(sel, 5.0))                      # a new ride: the clock restarted, no hold

    def test_map_signal_ahead_is_a_soft_prior(self):
        def map_msg(t, dist=80.0, **kw):
            m = {"type": "map", "ts_mono_ns": int(t * 1e9), "available": True, "match": {"confidence": 0.9},
                 "ahead": {"features": [{"kind": "crossing", "dist_m": 20.0, "confidence": 0.9},
                                        {"kind": "traffic_signals", "dist_m": dist, "confidence": 0.8}]}}
            m.update(kw)
            return m
        # a far, weak light: relevance between MAP_MIN_RELEVANCE and MIN_RELEVANCE
        weak = light(1, CX + 40, 300, 9, "red", score=0.3, hits=1)
        rel = lights.annotate(msg(dict(weak)), road(), self.CAL, (F, CX, CY, W, H))["lights"][0]["relevance"]
        self.assertTrue(lights.MAP_MIN_RELEVANCE <= rel < lights.MIN_RELEVANCE, rel)
        sel = lights.EgoLightSelector()
        self.assertIsNone(self.step(sel, 10.0, dict(weak)))                 # no map: not an ego light
        sel.on_map(map_msg(10.2))
        m = msg(dict(weak))
        m["ts_mono_ns"] = int(10.33e9)
        sel.annotate(m, road(), self.CAL, (F, CX, CY, W, H))
        self.assertEqual(m["map_signal"], {"kind": "traffic_signals", "dist_m": 80.0, "confidence": 0.8})
        self.assertEqual(m["ego"]["track_id"], 1)                           # the map expects a light: accepted
        self.assertTrue(self.step(sel, 12.2)["held"])                       # and held 2 s (map refreshed below)
        sel.on_map(map_msg(12.3))
        self.assertIsNone(self.step(sel, 12.5))
        # an unusable map changes nothing
        for bad in (map_msg(20.0, available=False), map_msg(20.0, match={"confidence": 0.3}), map_msg(20.0, dist=400.0),
                    map_msg(10.0)):                                         # the last one: stale at t = 20.3
            sel = lights.EgoLightSelector()
            sel.on_map(bad)
            m = msg(dict(weak))
            m["ts_mono_ns"] = int(20.3e9)
            sel.annotate(m, road(), self.CAL, (F, CX, CY, W, H))
            self.assertIsNone(m["map_signal"])
            self.assertIsNone(m["ego"])


if __name__ == "__main__":
    unittest.main()


L = lights


class LaneAssignment(unittest.TestCase):
    """Lane L16: which of two signal groups applies to the ego lane."""
    F = 975.0
    INTR = (975.0, 640.0, 360.0, 1280, 720)

    def light(self, tid, state, lat, d=25.0, hits=6):
        h = self.F * L.HOUSING_M / d
        u = 640.0 + lat * self.F / d
        return {"track_id": tid, "state": state, "score": 0.7, "hits": hits,
                "box": [u - h / 6, 200.0, u + h / 6, 200.0 + h]}

    def lanes(self, yl=-1.7, yr=1.7, far_left=None, far_right=None, edge_l=None, edge_r=None, ts=0):
        xs = [0.0, 10.0, 30.0, 60.0]

        def ln(y):
            return {"present": y is not None, "y": [y] * 4 if y is not None else None}
        return {"type": "lanes", "ts_mono_ns": ts, "x": xs, "lane_width_m": (yr - yl) if None not in (yl, yr) else None,
                "lines": [ln(far_left), ln(yl), ln(yr), ln(far_right)], "edges": [ln(edge_l), ln(edge_r)]}

    def run_(self, lights, lanes=None, **kw):
        msg = {"type": "lights", "frame": [1280, 720], "lights": lights}
        return L.annotate(msg, None, None, self.INTR, lanes=lanes, **kw)["ego"]

    def test_left_turn_red_beside_straight_green(self):
        # ego in the right one of two lanes: green over it, red over the left (turn) lane
        ego = self.run_([self.light(1, "red", -3.4), self.light(2, "green", 0.2)],
                        self.lanes(far_left=-5.1, edge_l=-5.4, edge_r=2.0))
        self.assertEqual(ego["state"], "green")
        self.assertTrue(ego["assign"]["conflict"])
        self.assertGreaterEqual(ego["assign"]["confidence"], L.AMB_CONFIDENCE)
        self.assertEqual(ego["assign"]["alt"]["state"], "red")
        self.assertEqual(ego["assign"]["alt"]["side"], "left")
        self.assertFalse(ego["amb"])
        self.assertEqual(ego["agree"], 1)                 # the red head does not vote

    def test_ego_in_left_turn_lane(self):
        # the same junction from the left lane: the lane lines move, the red head is now over the ego lane
        ego = self.run_([self.light(1, "red", 0.0), self.light(2, "green", 3.6)],
                        self.lanes(far_right=5.1, edge_l=-2.0, edge_r=5.4))
        self.assertEqual(ego["state"], "red")
        self.assertEqual(ego["assign"]["alt"]["side"], "right")

    def test_between_two_heads_is_ambiguous(self):
        ego = self.run_([self.light(1, "red", -1.6), self.light(2, "green", 1.6)], self.lanes())
        self.assertTrue(ego["assign"]["conflict"])
        self.assertLess(ego["assign"]["confidence"], L.AMB_CONFIDENCE)
        self.assertTrue(ego["amb"])

    def test_right_post_counts_for_the_rightmost_lane(self):
        # ego in the rightmost lane: the post at the roadside (green) against a head two lanes to the left (red)
        ego = self.run_([self.light(1, "red", -6.5), self.light(2, "green", 4.5)],
                        self.lanes(far_left=-5.1, edge_l=-8.6, edge_r=2.0))
        self.assertEqual(ego["state"], "green")
        self.assertGreater(ego["assign"]["confidence"], 0.8)

    def test_no_lane_lines_uses_the_path(self):
        ego = self.run_([self.light(1, "red", -5.0), self.light(2, "green", 0.5)])
        self.assertEqual(ego["state"], "green")
        self.assertEqual(ego["assign"]["basis"], "path")

    def test_agreeing_heads_are_no_conflict(self):
        ego = self.run_([self.light(1, "green", -3.4), self.light(2, "green", 0.2)], self.lanes())
        self.assertFalse(ego["assign"]["conflict"])
        self.assertIsNone(ego["assign"]["alt"])
        self.assertEqual(ego["agree"], 2)

    def test_lone_light_over_another_lane_is_served_and_marked(self):
        # (with the map expecting a signal the window lets a light this far to the left through)
        ego = self.run_([self.light(1, "red", -4.5, d=20.0)],
                        self.lanes(yl=-1.4, yr=1.4, far_left=-4.2, edge_l=-7.2),
                        min_relevance=L.MAP_MIN_RELEVANCE)
        self.assertEqual(ego["state"], "red")             # never removed
        self.assertTrue(ego["assign"].get("other_lane"))
        self.assertTrue(ego["amb"])

    def test_switched_off_is_the_old_selection(self):
        ego = self.run_([self.light(1, "red", -3.4), self.light(2, "green", 0.2)], self.lanes(), lane_assign=False)
        self.assertNotIn("assign", ego)
        self.assertNotIn("slot", ego)

    def test_map_lane_count_caps_the_right_side(self):
        fr = L.lane_frame(self.lanes(far_right=5.1, edge_r=8.6), None, 20.0, lanes_forward=1)
        self.assertEqual(fr["n_right"], 0)
        self.assertEqual(L.lane_frame(self.lanes(far_right=5.1, edge_r=8.6), None, 20.0)["n_right"], 2)

    def test_selector_reports_ambiguity_once_it_lasts(self):
        sel = L.EgoLightSelector()
        out = []
        for k in range(3):
            sel.on_lanes(self.lanes(ts=int(k * 0.33e9)))
            msg = {"type": "lights", "ts_mono_ns": int(k * 0.33e9), "frame": [1280, 720],
                   "lights": [self.light(1, "red", -1.6), self.light(2, "green", 1.6)]}
            out.append(sel.annotate(msg, None, None, self.INTR)["ego"]["amb"])
        self.assertEqual(out, [False, True, True])


class SelectorMemory(unittest.TestCase):
    """What the selector remembers between messages (lane "drive-quality"): the cases of the drives of 26 / 27
    November 2026, rebuilt from their geometry (a camera at 1.2 m, f 975 px, the provider's two views). Every case
    is also run with `memory = False`, the selection per message, which must show the fault the case is about."""
    F, CX, CY = 975.0, 640.0, 360.0
    INTR = (975.0, 640.0, 360.0, 1280, 720)
    CAL = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "height_m": 1.2}
    WIDE, ZOOM = [0, 0, 1280, 446], [160, 90, 1120, 410]

    def head(self, tid, state, y, d, height=5.2, hits=20, score=0.7, roi=None, **kw):
        """A 0.9 m housing `y` m right of the car's axis, `d` m ahead, its centre `height` m above the road, as the
        view `roi` shows it: cut at the view's edge (None when nothing of it is inside)."""
        h = self.F * L.HOUSING_M / d
        u, v = self.CX + self.F * y / d, self.CY - self.F * (height - 1.2) / d
        box = [u - h / 6, v - h / 2, u + h / 6, v + h / 2]
        roi = roi or self.WIDE
        if box[3] <= roi[1] + 4 or box[1] >= roi[3] or box[2] <= roi[0] or box[0] >= roi[2]:
            return None
        box = [max(box[0], roi[0]), max(box[1], roi[1]), min(box[2], roi[2]), min(box[3], roi[3])]
        return dict({"track_id": tid, "state": state, "frame_state": state, "score": score, "hits": hits, "coast": 0,
                     "box": box}, **kw)

    def lanes(self, ts, **kw):
        xs = [0.0, 10.0, 30.0, 60.0]
        y = dict(far_left=None, yl=-1.7, yr=1.7, far_right=None, edge_l=None, edge_r=None)
        y.update(kw)

        def ln(v):
            return {"present": v is not None, "y": [v] * 4 if v is not None else None}
        return {"type": "lanes", "ts_mono_ns": int(ts * 1e9), "x": xs, "lane_width_m": 3.4,
                "lines": [ln(y["far_left"]), ln(y["yl"]), ln(y["yr"]), ln(y["far_right"])],
                "edges": [ln(y["edge_l"]), ln(y["edge_r"])]}

    def ego_state(self, east, odo, v, psi=0.0, north=0.0):
        return {"type": "ego_state", "speed": {"value": v}, "odometer_m": odo,
                "position": {"local": {"frame": 0, "e_m": east, "n_m": north, "heading_rad": psi}}}

    def road(self, v, curve=0.0):
        """A plan of 8 s of driving at v (a point at rest), on a road that bends by `curve` (y = curve x^2)."""
        xs = [v * 8.0 * (i / 32.0) ** 1.5 for i in range(33)]
        return {"type": "road", "plan": {"x": xs, "y": [curve * x * x for x in xs], "z": [0.0] * 33}}

    def step(self, sel, t, heads, view=0, road=None, lanes=None, es=None):
        if lanes is not None:
            sel.on_lanes(lanes)
        if es is not None:
            sel.on_ego(es)
        m = {"type": "lights", "ts_mono_ns": int(t * 1e9), "frame": [1280, 720], "view": view,
             "roi": self.ZOOM if view else self.WIDE, "lights": [h for h in heads if h is not None]}
        return sel.annotate(m, road, self.CAL, self.INTR)

    def both(self, run):
        """The outcome of `run(selector)` with the memory and without."""
        out = []
        for memory in (True, False):
            sel = L.EgoLightSelector()
            sel.memory = memory
            out.append(run(sel))
        return out

    def test_two_gantry_heads_at_the_stop_line(self):
        """Three lanes, the car in the middle one, standing at the line: the head over its lane goes green, the one
        over the left-turn lane stays red. The lane lines of the junction's mouth say there is no lane on the left,
        so the red head counted as the ego lane's post, and with the zoomed view cutting it the choice alternated."""
        def run(sel):
            states = []
            for k in range(40):
                t, view = 100.0 + 0.2 * k, k % 2
                roi = self.ZOOM if view else self.WIDE
                # (the detector's score of a box differs between the views by a few hundredths, as on the drive)
                mine = self.head(1, "red" if k < 10 else "green", 1.1, 15.0, height=4.8, roi=roi,
                                 score=0.75 if view else 0.62)
                left = self.head(2, "red", -2.5, 15.0, height=5.6, roi=roi, score=0.75)
                m = self.step(sel, t, [mine, left], view, self.road(0.0), self.lanes(t, edge_l=-2.0),
                              self.ego_state(0.0, 0.0, 0.0))
                states.append((m["ego"] or {}).get("state"))
            return states
        new, old = self.both(run)
        self.assertEqual(new[:10], ["red"] * 10)
        self.assertEqual(new[10:], ["green"] * 30)
        self.assertIn("red", old[12:])                    # per message: the left-turn head takes over

    def test_the_ego_head_leaves_through_the_top(self):
        """The car stops at the line under its own head; the head over the lane to the right stays in the picture
        and goes green. It was served as the ego light for the whole red phase."""
        def run(sel, post=False):
            seen = []
            d, odo, t = 40.0, 0.0, 200.0
            for k in range(120):
                v = 8.0 if d > 12.5 else 0.0
                roi = self.WIDE
                mine = self.head(1, "red", 0.0, d, height=5.6, roi=roi)
                if mine is not None and mine["box"][1] <= 0.0:
                    mine.update(state="off", frame_state="off")     # only its lower lamps are in the picture
                right = self.head(2, "red" if k < 70 else "green", 3.5, d + 1.0, height=4.9, roi=roi)
                side = self.head(3, "red", -2.6, d, height=2.8, roi=roi) if post else None
                lanes = self.lanes(t, far_right=5.1) if d > 20.0 else self.lanes(t, yl=None, yr=None)
                m = self.step(sel, t, [mine if k < 50 else None, right, side], 0, self.road(v), lanes,
                              self.ego_state(odo, odo, v))
                seen.append(((m["ego"] or {}).get("state"), (m["ego"] or {}).get("track_id"), m.get("out_of_view")))
                t += 0.2
                if v:
                    d, odo = max(12.5, d - 1.6), odo + 1.6
            return seen
        new, old = self.both(run)
        self.assertTrue(all(s == "red" and tid == 1 for s, tid, _ in new[:18]))
        self.assertNotIn("green", [s for s, _, _ in new])           # never the other lane's green
        self.assertEqual(new[-1][0], None)
        self.assertEqual((new[-1][2]["track_id"], new[-1][2]["state"]), (1, "red"))
        self.assertGreater(new[-1][2]["age_s"], 5.0)
        self.assertIn("green", [s for s, _, _ in old])              # per message: it was
        # with a post of the ego lane's own signal at the left kerb, that post is the ego light
        new, old = self.both(lambda sel: run(sel, post=True))
        self.assertEqual({(s, tid) for s, tid, _ in new[60:]}, {("red", 3)})
        self.assertIn("green", [s for s, _, _ in old[75:]])

    def test_the_other_lane_head_is_served_again_after_the_junction(self):
        """`out_of_view` ends with the place: 15 m on, a light is a light again."""
        sel = L.EgoLightSelector()
        t = 300.0
        for k in range(12):                                         # both heads, mine over the lane
            d = 30.0 - k
            self.step(sel, t, [self.head(1, "red", 0.0, d, height=5.6), self.head(2, "red", 3.5, d + 1.0, height=4.9)],
                      0, self.road(5.0), self.lanes(t, far_right=5.1), self.ego_state(float(k), float(k), 5.0))
            t += 0.2
        cut = self.head(1, "red", 0.0, 12.5, height=5.6)
        self.assertLessEqual(cut["box"][1], 0.0)
        self.step(sel, t, [cut, self.head(2, "red", 3.5, 13.5, height=4.9)], 0, self.road(0.0), None,
                  self.ego_state(12.0, 12.0, 0.0))
        m = self.step(sel, t + 0.2, [self.head(2, "green", 3.5, 13.5, height=4.9)], 0, self.road(0.0), None,
                      self.ego_state(12.0, 12.0, 0.0))
        self.assertEqual((m["ego"]["track_id"], m["ego"]["state"], m["ego"]["held"]), (1, "red", True))   # the hold
        m = self.step(sel, t + 2.0, [self.head(2, "green", 3.5, 13.5, height=4.9)], 0, self.road(0.0), None,
                      self.ego_state(12.0, 12.0, 0.0))
        self.assertIsNone(m["ego"])
        self.assertEqual(m["out_of_view"]["track_id"], 1)
        m = self.step(sel, t + 4.0, [self.head(2, "green", 3.5, 13.5, height=4.9)], 0, self.road(5.0), None,
                      self.ego_state(30.0, 30.0, 5.0))
        self.assertEqual(m["ego"]["state"], "green")
        self.assertNotIn("out_of_view", m)

    def test_a_cut_housing_keeps_its_colour_and_its_size(self):
        """The zoomed view shows the lower half of a near head: detected as "off" with half the box. The provider's
        state and the distance alternated with the views."""
        def run(sel):
            out = []
            for k in range(20):
                t, view = 400.0 + 0.2 * k, k % 2
                h = self.head(1, "red", 0.5, 14.0, height=5.4, roi=self.ZOOM if view else self.WIDE)
                if view:
                    self.assertEqual(h["box"][1], 90)                 # cut by the view
                    h.update(state="off", frame_state="off")
                m = self.step(sel, t, [h], view, self.road(0.0), None, self.ego_state(0.0, 0.0, 0.0))
                out.append(((m["ego"] or {}).get("state"), m["lights"][0]["dist_m"], m["lights"][0].get("state_in")))
            return out
        new, old = self.both(run)
        self.assertEqual({s for s, _, _ in new}, {"red"})
        self.assertLess(max(d for _, d, _ in new) - min(d for _, d, _ in new), 1.0)
        self.assertEqual({si for _, _, si in new[1::2]}, {"off"})       # what the provider said is kept beside it
        self.assertGreater(max(d for _, d, _ in old) - min(d for _, d, _ in old), 5.0)
        self.assertNotEqual({s for s, _, _ in old}, {"red"})

    def test_a_red_lamp_does_not_go_dark(self):
        """An LED lamp between its pulses: the provider reads red, off, off, off, red. Green and yellow lamps do
        go dark (a conditional arrow, a flashing yellow): they are not held."""
        def run(sel, colour="red"):
            out = []
            for k in range(30):
                st = colour if k % 6 == 0 else "off"
                m = self.step(sel, 500.0 + 0.2 * k, [self.head(1, st, 0.5, 20.0)], 0, self.road(0.0), None,
                              self.ego_state(0.0, 0.0, 0.0))
                out.append((m["ego"] or {}).get("state"))
            return out
        new, old = self.both(run)
        self.assertEqual(set(new), {"red"})
        self.assertIn("off", old)
        self.assertIn("off", run(L.EgoLightSelector(), "green"))

    def test_two_heads_of_one_colour_do_not_alternate(self):
        """Two red heads 60 m ahead, 10 px boxes that are a pixel taller or shorter with the view: the ego light
        changed with every message."""
        def run(sel):
            ids = []
            for k in range(30):
                t, d = 600.0 + 0.2 * k, 60.0 - 0.5 * k
                a = self.head(1, "red", -0.8, d, height=5.5)
                b = self.head(2, "red", 3.2, d, height=3.0)
                for h, sign in ((a, 1), (b, -1)):                   # the measured length of the housings alternates
                    h["box"][3] += 1.5 * sign * (1 if k % 2 else -1)
                m = self.step(sel, t, [a, b], k % 2, self.road(10.0), self.lanes(t), self.ego_state(5.0 * k, 5.0 * k, 10.0))
                ids.append(m["ego"]["track_id"])
            return sum(1 for x, y in zip(ids, ids[1:]) if x != y)
        new, old = self.both(run)
        self.assertLessEqual(new, 1)
        self.assertGreater(old, 5)

    def test_the_path_is_remembered_at_rest(self):
        """A road that bends left, a queue before the lights: at rest the plan is a point, and the lights 40 m on,
        over the car's lane, were measured from the camera's axis: metres to the left, no ego light."""
        curve = -0.004
        def run(sel):
            out = []
            for k in range(40):
                t = 700.0 + 0.2 * k
                v = 8.0 if k < 15 else 0.0
                x = min(1.6 * k, 1.6 * 14)                          # the car's place along the bend (small angles)
                d = 62.4 - x
                # the light hangs over the lane where the bend is 62.4 m from the start: 15.6 m left of the first axis;
                # seen from the car (on the bend at x, heading 2 curve x): its offset from the car's axis
                y_w, y_car, psi = curve * 62.4 ** 2, curve * x * x, -2.0 * curve * x
                lat_axis = (y_w - y_car) - math.tan(-psi) * d
                es = self.ego_state(x, x, v, psi=psi, north=-y_car)
                road = self.road(v, curve) if v else self.road(0.0)
                m = self.step(sel, t, [self.head(1, "red", lat_axis, d, height=5.5)], 0, road, None, es)
                out.append(((m["ego"] or {}).get("state"), m["lights"][0]["lateral_m"], (m["ego"] or {}).get("source")))
            return out
        new, old = self.both(run)
        self.assertEqual({s for s, _, _ in new}, {"red"})
        self.assertLess(max(abs(lat) for _, lat, _ in new), 1.5)
        self.assertEqual(new[-1][2], "path_mem")
        self.assertGreater(abs(old[-1][1]), 5.0)
        self.assertIsNone(old[-1][0])

    def test_the_ego_lane_estimate_counts_the_lanes(self):
        sel = L.EgoLightSelector()
        self.assertIsNone(sel._hint(1.0))
        sel.on_egolane({"probs": [0.1, 0.85, 0.05], "probs_from": "right"}, int(1e9))
        self.assertEqual(sel._hint(1.0), (None, 1))
        sel.on_egolane({"probs": [0.9, 0.08, 0.02], "probs_from": "right"}, int(1e9))
        self.assertEqual(sel._hint(1.0), (None, 0))
        sel.on_egolane({"probs": [0.5, 0.5], "probs_from": "right"}, int(1e9))
        self.assertIsNone(sel._hint(1.0))
        sel.on_egolane({"probs": [0.05, 0.9, 0.05], "probs_from": "left", "forward": 3}, int(1e9))
        self.assertEqual(sel._hint(1.0), (1, 1))
        self.assertIsNone(sel._hint(10.0))                           # stale
        # a head 1.2 lanes to the right: the roadside post of a single lane, another lane's head beside two
        fr = L.lane_frame(self.lanes(0.0), None, 20.0, hint=(None, 0))
        post = L.lane_weight(4.1, 20.0, fr)[1]
        fr = L.lane_frame(self.lanes(0.0), None, 20.0, hint=(None, 1))
        self.assertLess(L.lane_weight(4.1, 20.0, fr)[1], 0.6 * post)

    def test_a_plan_at_rest_is_no_path(self):
        self.assertIsNone(L.Path.of_road({"plan": {"x": [0.0, 0.01, -0.02, 0.0], "y": [0.0] * 4, "z": [0.0] * 4}}, L.PATH_MIN_M))
        p = L.Path.of_road({"plan": {"x": [0.0, 10.0, 20.0, 19.0], "y": [0.0, 0.5, 2.0, 2.0], "z": [0.0] * 4}})
        self.assertEqual(p.reach, 20.0)                              # the rising part
        self.assertAlmostEqual(p.at(15.0)[0], 1.25)
        self.assertAlmostEqual(p.at(30.0)[0], 2.0 + 0.15 * 10.0)     # continued along its last heading
        self.assertIsNone(p.at(40.0))                                # too far for that
