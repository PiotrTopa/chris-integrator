"""An object that is no longer there (docs/RADAR.md, section 12): boxes seen through it, the car's own body, the
place that is only a guess, the view's cut zone, the gate per hypothesis. Fixtures at the radar box (a level camera's
boxes); the simulator's scenarios are in test_radar_stale_sim.py."""
import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chris_integrator.radar import Radar  # noqa: E402
from chris_integrator.radar import tracker as TK  # noqa: E402

NS = 10 ** 9
F_PX, CX, CY, HC = 975.0, 640.0, 360.0, 1.22
CAL = {"type": "calib", "focal_px": F_PX, "width": 1280, "height": 720, "height_m": HC}
VIEW_BOTTOM = 446.0              # the wide view of the recordings: rows 0 .. 446 of 720


def boxes(t, things, ground_min=8.0, cut=False):
    """A `road_users` message of a level camera: things {name: (x of the near face, y left, width, height, class)}.
    cut: the view ends at VIEW_BOTTOM (a box that reaches below it is cut there and has no range of its own)."""
    objs = []
    for name, (x, y, w, h, cls) in things.items():
        u, hw = CX - F_PX * y / x, 0.5 * F_PX * w / x
        top, bot = CY + F_PX * (HC - h) / x, CY + F_PX * HC / x
        o = {"class": cls, "class_conf": 0.8, "bearing_deg": math.degrees(math.atan2(y, x)), "bearing_std_deg": 0.3}
        if cut and bot > VIEW_BOTTOM:
            if top > VIEW_BOTTOM - 8.0:
                continue
            o.update(box=[u - hw, top, u + hw, VIEW_BOTTOM], clip="b", range_m=None)
        else:
            o.update(box=[u - hw, top, u + hw, bot], range_m=x, range_std_m=0.05 * x)
        objs.append(o)
    cov = {"bearing_min_deg": -33, "bearing_max_deg": 33,
           "ground_min_m": round(F_PX * HC / (VIEW_BOTTOM - CY), 2) if cut else ground_min}
    return {"type": "road_users", "sensor": "cam", "ts_mono_ns": int(t * NS), "objects": objs, "coverage": cov}


def drive(cfg, scene, T, v=8.0, cut=False):
    """The radar box over T seconds at 20 Hz with a scan every 0.2 s. scene(s, X) -> the things of the scan at time
    s with the car's travel X (their places in the car frame); v: the car's speed, a number or v(s). Returns
    [(s, X, the message's objects)]."""
    rd = Radar({"tracker": cfg})
    rd.on_calib(CAL)
    out, X = [], 0.0
    for i in range(int(T * 20)):
        s, t = i * 0.05, 30.0 + i * 0.05
        vv = v(s) if callable(v) else v
        rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": vv, "yaw_rate_rps": 0.0, "standstill": vv < 0.01})
        if i % 4 == 0:
            rd.on_road_users(boxes(t, scene(s, X), cut=cut))
        out.append((s, X, rd.frame(int(t * NS))["objects"]))
        X += vv * 0.05
    return out, rd


def ghost_scene(lead):
    """A car standing in the lane 40 m ahead of the start, boxed until the car is 16 m from it and never again (it
    is not there any more); lead: a van 45 m ahead of the car, at the car's speed, in the same lane."""
    def scene(s, X):
        th = {}
        if 40.0 - X > 16.0:
            th["ghost"] = (40.0 - X, 0.2, 1.8, 1.5, "car")
        if lead:
            th["lead"] = (45.0, 0.0, 2.0, 2.4, "truck")
        return th
    return scene


def ghost(objs, X):
    """The object at the ghost's place (within 3 m of where it stood), or None."""
    for o in objs:
        if abs(o["x_m"] - (40.0 - X)) < 3.0 and abs(o["y_m"] - 0.2) < 1.5 and o["motion"] != "moving":
            return o
    return None


class Layout(unittest.TestCase):
    def test_the_scan_constants_end_where_the_native_core_reads_the_gate(self):
        src = open(os.path.join(os.path.dirname(TK.__file__), "..", "native", "src", "chris_native.c")).read()
        i = src.index("enum { XC_FLAGS")
        body = re.sub(r"/\*.*?\*/", "", src[i:src.index("};", i)], flags=re.S)
        val, env = -1, {"RSPREV": int(re.search(r"#define RSPREV (\d+)", src).group(1))}
        for n in [x.strip() for x in body[body.index("{") + 1:].split(",") if x.strip()]:
            if "=" in n:
                k, e = [q.strip() for q in n.split("=")]
                val = eval(e, {}, env)
            else:
                k, val = n, val + 1
            env[k] = val
        self.assertEqual(env["XC_GHYP"], TK._XC_END)
        self.assertEqual(env["RXCFG"], TK._XC_END + 1)

    def test_the_knobs_are_off(self):
        for k in ("see_through", "pd_cut", "swept"):
            self.assertIs(TK.CFG[k], False)
        self.assertEqual((TK.CFG["coast_std_m"], TK.CFG["gate_hyp"]), (0.0, 0.0))


class SeenThrough(unittest.TestCase):
    """A standing object that is no longer boxed, and the lead's box inside its silhouette."""

    def run_it(self, cfg, lead=True):
        out, rd = drive(cfg, ghost_scene(lead), 6.5)
        # when the ghost was last put out, as the distance left to it then (None: it was never an object)
        seen = [(s, 40.0 - X) for s, X, objs in out if ghost(objs, X) is not None]
        return out, rd, seen

    def test_as_it_was_the_ghost_is_kept_until_the_car_is_in_it(self):
        out, rd, seen = self.run_it({})
        self.assertTrue(seen and seen[0][1] > 25.0)                         # an object while it was boxed
        self.assertLess(seen[-1][1], 2.5)                                  # ... and until the car stands in its place

    def test_a_box_seen_through_it_ends_it(self):
        out, rd, seen = self.run_it({"see_through": True})
        self.assertTrue(seen and seen[0][1] > 25.0)
        self.assertGreater(seen[-1][1], 8.0)                               # gone with 8 m (a second) to go
        self.assertGreater(rd.tracker.ev_stats["seen_through"], 2)
        # the lead it was seen through is one object throughout
        lead = [[o["id"] for o in objs if o["x_m"] > 30.0 and abs(o["y_m"]) < 1.5] for s, X, objs in out if s > 2.5]
        self.assertTrue(all(len(x) == 1 for x in lead))
        self.assertEqual(len({x[0] for x in lead}), 1)

    def test_with_nothing_behind_it_nothing_is_seen_through(self):
        out, rd, seen = self.run_it({"see_through": True}, lead=False)
        self.assertLess(seen[-1][1], 2.5)
        self.assertEqual(rd.tracker.ev_stats["seen_through"], 0)

    def test_an_object_that_is_boxed_is_not_touched(self):
        """A car that stands and is boxed all the way, the lead beyond it boxed too (taller: seen over it)."""
        def scene(s, X):
            return {"car": (40.0 - X, 0.2, 1.8, 1.5, "car"), "lead": (45.0, 0.0, 2.0, 2.4, "truck")}
        a, _ = drive({}, scene, 3.5)
        b, rd = drive({"see_through": True}, scene, 3.5)
        self.assertEqual(rd.tracker.ev_stats["seen_through"], 0)
        self.assertEqual([[(o["id"], o["exist"], o["x_m"]) for o in objs] for s, X, objs in a],
                         [[(o["id"], o["exist"], o["x_m"]) for o in objs] for s, X, objs in b])

    def test_a_box_beside_it_is_not_seen_through_it(self):
        """The lead in the next lane (3.5 m to the side): while the ghost is 10 m away its silhouette does not hold
        the lead's box (from 5 m it does: a car 5 m ahead covers 7 degrees either side)."""
        def scene(s, X):
            th = {"lead": (45.0, -3.5, 2.0, 2.4, "truck")}
            if 40.0 - X > 16.0:
                th["ghost"] = (40.0 - X, 0.2, 1.8, 1.5, "car")
            return th
        out, rd = drive({"see_through": True}, scene, 5.5)
        at10 = [ghost(objs, X) for s, X, objs in out if 9.5 < 40.0 - X < 11.0]
        self.assertTrue(at10 and all(o is not None and o["exist"] > 0.999 for o in at10))


class Swept(unittest.TestCase):
    def run_it(self, cfg):
        out, rd = drive(cfg, ghost_scene(False), 7.5)
        return out, rd

    def test_as_it_was_the_ghost_passes_through_the_car(self):
        out, rd = self.run_it({})
        inside = [s for s, X, objs in out if any(-2.0 < o["x_m"] < 1.0 and abs(o["y_m"]) < 0.8 for o in objs)]
        self.assertGreater(len(inside), 4)

    def test_the_cars_own_body_is_empty(self):
        out, rd = self.run_it({"swept": True})
        self.assertGreater(rd.tracker.ev_stats["swept"], 0)
        # its existence falls as the bumper reaches it; with the car in it, it is gone; nothing is left behind the car
        g = [(40.0 - X, ghost(objs, X)) for s, X, objs in out]
        self.assertTrue(all(o is not None and o["exist"] > 0.99 for d, o in g if 6.0 < d < 12.0))
        self.assertTrue(any(o is not None and o["exist"] < 0.999 for d, o in g if -1.5 < d < 3.0))
        self.assertTrue(all(o is None for d, o in g if d < -1.5))
        self.assertEqual([o["id"] for s, X, objs in out if 40.0 - X < -1.5 for o in objs], [])

    def test_a_car_passed_at_the_kerb_is_not_swept(self):
        """A parked car 2.6 m beside the path, boxed until it leaves the view: it is passed, not driven through."""
        def scene(s, X):
            x = 40.0 - X
            return {"parked": (x, -2.6, 1.8, 1.5, "car")} if x > 5.0 else {}
        a, _ = drive({}, scene, 6.0)
        b, rd = drive({"swept": True}, scene, 6.0)
        self.assertEqual([[o["id"] for o in objs] for s, X, objs in a], [[o["id"] for o in objs] for s, X, objs in b])
        # (what is not known of its place across the path reaches into the car's side by a few hundredths)
        self.assertGreater(min(o["exist"] for s, X, objs in b for o in objs if o["state"] != "tentative"), 0.9)


class CoastStd(unittest.TestCase):
    """A car in the next lane, 8 m ahead at the car's own speed, boxed for 3 s and then not."""

    def run_it(self, cfg):
        def scene(s, X):
            return {"beside": (8.0, -3.3, 1.8, 1.5, "car")} if s < 3.0 else {}
        return drive(cfg, scene, 8.5, v=12.0)

    def test_as_it_was_it_coasts_for_seconds_on_a_place_that_is_a_guess(self):
        out, rd = self.run_it({})
        last = max(s for s, X, objs in out if objs)
        self.assertGreater(last, 6.5)
        std = [max(o["std"][0], o["std"][1]) for s, X, objs in out if 5.5 < s < 6.5 for o in objs]
        self.assertGreater(min(std), 3.0)

    def test_a_place_that_is_a_guess_is_not_kept(self):
        out, rd = self.run_it({"coast_std_m": 3.0})
        last = max(s for s, X, objs in out if objs)
        self.assertGreater(last, 3.6)                                      # the first misses do not end it
        self.assertLess(last, 5.6)
        self.assertEqual(rd.tracker.ev_stats["coast_std"], 1)
        # (the bound is the larger of coast_std_m and coast_std_rel of the range: 5.4 m at 10.8 m)
        self.assertLess(max(max(o["std"][0], o["std"][1]) for s, X, objs in out for o in objs if s > 3.0), 6.0)

    def test_a_standing_object_is_not_dropped_by_it(self):
        """Where a standing thing stands is known as well as ever: the rule is for what moves."""
        out, rd = drive({"coast_std_m": 3.0}, ghost_scene(False), 4.6)
        self.assertEqual(rd.tracker.ev_stats["coast_std"], 0)
        self.assertIsNotNone(ghost(out[-1][2], out[-1][1]))


class CutZone(unittest.TestCase):
    """The view's cut zone: a lead 6 m ahead whose box is cut at the bottom of the view."""

    def run_it(self, cfg, t_gone):
        def scene(s, X):                                 # from 20 m (a whole box) to 6 m in 4 s, then held
            return {"lead": (max(6.0, 20.0 - 3.5 * s), 0.0, 1.8, 1.5, "car")} if s < t_gone else {}
        return drive(cfg, scene, t_gone + 7.0, v=10.0, cut=True)

    def test_the_share_of_boxes_in_the_cut_zone_is_followed(self):
        out, rd = self.run_it({"pd_cut": True}, 8.0)
        (key, (h, n)), = rd.tracker.cut_stat.items()
        self.assertEqual(key[0], "cam")
        self.assertGreater(n, 20)
        self.assertGreater(h / n, 0.9)                                     # (the misses after it left: a few)

    def test_an_object_of_the_cut_zone_that_is_no_longer_boxed_ends_sooner(self):
        a, _ = self.run_it({}, 8.0)
        b, _ = self.run_it({"pd_cut": True}, 8.0)
        la = max(s for s, X, objs in a if any(o["state"] != "memory" for o in objs))
        lb = max(s for s, X, objs in b if any(o["state"] != "memory" for o in objs))
        self.assertGreater(la, 12.5)                                       # as it was: 5 s (seen_max_s)
        self.assertLess(lb, 10.8)
        self.assertGreater(lb, 8.7)                                        # (two scans without a box do not end it)


if __name__ == "__main__":
    unittest.main()
