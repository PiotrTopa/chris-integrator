"""The radar's use of the detector's further inputs (docs/RADAR.md, section 8: flags, the low-score list, the
appearance vector, the lamp pair, the rulers, the road's vertical curvature, the scale rate's shared baseline), on
small worlds with known truth. Every one is a knob of `radar.tracker`, off by default; the first test holds that
with the knobs off the fields change nothing."""
import json
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from chris_integrator.core.roadusers import from_lights  # noqa: E402
from chris_integrator.native import _chris_native as C  # noqa: E402
from chris_integrator.radar import Radar  # noqa: E402
from chris_integrator.radar import models as M  # noqa: E402
from chris_integrator.radar import tracker as T  # noqa: E402

NS = 10 ** 9
F_PX, CX, CY, HC = 975.0, 640.0, 360.0, 1.22
CAL = {"type": "calib", "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "height_m": HC, "focal_px": F_PX, "cx": CX, "cy": CY,
       "width": 1280, "height": 720}
ALL = {"in_flags": True, "in_box_occl": True, "in_low": True, "in_app": True, "in_reid": True, "in_lamp_colour": True,
       "in_lamp_size": True, "in_width": True, "in_scale": True, "grade_k": 1 / 3000.0}


def box(x, y, w, h, cls="car", **kw):
    """A thing whose near face stands x ahead, y left, w wide, h tall, as a `road_users` object of a level camera."""
    u, hw = CX - F_PX * y / x, 0.5 * F_PX * w / x
    o = {"class": cls, "class_conf": 0.8, "bearing_deg": math.degrees(math.atan2(y, x)), "bearing_std_deg": 0.3,
         "range_m": x, "range_std_m": 0.05 * x, "box": [u - hw, CY + F_PX * (HC - h) / x, u + hw, CY + F_PX * HC / x]}
    o.update(kw)
    return o


def msg(t, objs, low=None, kind=None):
    m = {"type": "road_users", "sensor": "cam", "ts_mono_ns": int(t * NS), "objects": objs,
         "coverage": {"bearing_min_deg": -33, "bearing_max_deg": 33, "ground_min_m": 8.0}}
    if low is not None:
        m["objects_low"] = low
    if kind:
        m["emb_kind"] = kind
    return m


def emb(seed):
    r = random.Random(seed)
    return bytes([max(-127, min(127, int(round(32 * r.gauss(0, 1.2))))) & 255 for _ in range(27)]).hex()


def lamps(x, S=1.2, c="red"):
    sep = F_PX * S / x
    return {"c": c, "l": [0.0, 0.0], "r": [0.0, 0.0], "sep": sep, "sep_std": math.hypot(0.35, 0.02 * sep)}


def run(cfg, things, T_s=6.0, v=12.0, every=4, kind=None, low=None):
    """things(s, k) -> the first list's objects of scan k at s seconds; low(s, k) -> the low list's. The radar with
    `cfg` as its tracker's configuration, and its message at every 0.05 s."""
    rd = Radar({"tracker": cfg})
    rd.on_calib(CAL)
    out = []
    for i in range(int(round(T_s / 0.05))):
        t = 20 + i * 0.05
        rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": v, "yaw_rate_rps": 0.0, "standstill": v == 0.0})
        if i % every == 0:
            rd.on_road_users(msg(t, things(i * 0.05, i // every), low=low(i * 0.05, i // every) if low else None, kind=kind))
        out.append((i * 0.05, rd.frame(int(t * NS))))
    return rd, out


def dets_of(cfg, obj, low=False):
    rd = Radar({"tracker": cfg})
    rd.on_calib(CAL)
    rd.on_road_users(msg(20.0, [] if low else [obj], low=[obj] if low else None))
    return rd.queue[-1][3]


class KnobsOff(unittest.TestCase):
    def test_with_the_knobs_off_the_new_fields_change_nothing(self):
        rng = random.Random(4)

        def things(rich):
            def f(s, k):
                o = [box(40.0 - 2.0 * s, 0.3, 1.8, 1.5), box(25.0, 3.0, 1.8, 1.5), box(18.0, -3.2, 2.2, 2.6, cls="truck")]
                if rich:
                    o[0].update(trunc="", occ=0.3, occ_by=1, occ_edges="b", emb=emb(1), lamps=lamps(40.0), sym=0.8,
                                scale_src="lamps")
                    o[1].update(trunc="l", emb=emb(2))
                return [q for q in o if rng.random() > 0.2]
            return f
        outs = []
        for rich in (False, True):
            rng = random.Random(4)
            _, out = run({}, things(rich), T_s=5.0, kind="yuv3x3" if rich else None,
                         low=(lambda s, k: [box(60.0, 1.0, 1.8, 1.5)]) if rich else None)
            outs.append(json.dumps([m for _, m in out], sort_keys=True))
        self.assertEqual(outs[0], outs[1])

    def test_the_fields_are_passed_on_from_the_detector_as_sent_and_absent_ones_stay_absent(self):
        ru = {"cls": "car", "score": 0.8, "box": [600, 380, 660, 420], "clip": "", "rider": False}
        m = {"type": "lights", "ts_mono_ns": 5, "frame": [1280, 720], "roi": [0, 0, 1280, 446], "view": 0, "lights": [],
             "road_users": [dict(ru)]}
        plain = from_lights(m, CAL)
        self.assertNotIn("objects_low", plain)
        self.assertNotIn("emb_kind", plain)
        self.assertFalse({"trunc", "occ", "occ_by", "occ_edges", "sym", "lamps", "emb", "scale_src"} & set(plain["objects"][0]))
        rich = dict(ru, trunc="", occ=0.35, occ_by=0, occ_edges="rb", sym=0.8, lamps=lamps(30.0), emb="0a" * 27, scale_src="both")
        m2 = dict(m, road_users=[rich], road_users_low=[dict(ru, score=0.2)], emb_kind="yuv3x3")
        out = from_lights(m2, CAL)
        o = out["objects"][0]
        self.assertEqual((o["trunc"], o["occ"], o["occ_by"], o["occ_edges"], o["emb"], o["scale_src"]), ("", 0.35, 0, "rb", "0a" * 27, "both"))
        self.assertEqual(out["emb_kind"], "yuv3x3")
        self.assertEqual(len(out["objects_low"]), 1)
        self.assertEqual({k: v for k, v in o.items() if k in plain["objects"][0]}, plain["objects"][0])
        # shed by the governor: no low list either
        self.assertNotIn("objects_low", from_lights(dict(m2, road_users=None), CAL))


class Flags(unittest.TestCase):
    def test_the_hidden_edges_decide_which_rows_a_box_has(self):
        kind = lambda **kw: dets_of({"in_flags": True}, box(30.0, 2.0, 1.8, 1.5, **kw))[0]
        self.assertEqual(kind()["kind"], "rbh")
        self.assertEqual(kind(occ=0.3, occ_edges="t")["kind"], "rb")          # no height from a hidden top
        self.assertEqual(kind(occ=0.3, occ_edges="b")["kind"], "bt")          # no ground angle from a hidden bottom
        self.assertEqual(kind(trunc="b")["kind"], "bt")
        self.assertEqual(kind(occ=0.5, occ_edges="tb")["kind"], "b")
        self.assertFalse(kind(occ=0.3, occ_edges="r")["wok"])                 # no extent from a hidden side
        self.assertFalse(kind(occ=0.15)["wok"])                               # ... nor from a covered box
        self.assertTrue(kind(occ=0.05)["wok"])
        # a covered box whose edges are all its own: its size is less sure
        self.assertAlmostEqual(kind(occ=0.5)["R"][2] / kind()["R"][2], 1.5 ** 2, places=9)
        # without the knob the flags are not read
        self.assertEqual(dets_of({}, box(30.0, 2.0, 1.8, 1.5, occ=0.3, occ_edges="b"))[0]["kind"], "rbh")

    def test_a_box_with_a_hidden_side_is_read_as_the_visible_part_of_the_object(self):
        """A car 20 m ahead whose right half is under a nearer box: the box is the left half. Read as the object's
        centre it pulls the object left by a quarter of its width; with the flags the object stays where it is."""
        def things(s, k):
            full = box(20.0, 0.0, 1.8, 1.5)
            if s < 2.0:
                return [full]
            half = box(20.0, 0.45, 0.9, 1.5, occ=0.5, occ_edges="r")
            return [half]
        res = {}
        for name, cfg in (("off", {}), ("on", {"in_flags": True})):
            _, out = run(cfg, things, T_s=5.0, v=0.0)
            res[name] = out[-1][1]["objects"][0]["y_m"]
        self.assertGreater(res["off"], 0.3)
        self.assertLess(abs(res["on"]), 0.2)

    def test_a_box_of_the_scan_hides_what_stands_behind_it_before_it_is_an_object(self):
        def things(s, k):
            return [box(40.0, 0.2, 1.8, 1.5)] if s < 2.0 else [box(15.0, 0.2, 2.2, 2.6, cls="truck")]
        first = {}
        for name, cfg in (("off", {}), ("on", {"in_box_occl": True})):
            _, out = run(cfg, things, T_s=3.6)
            car = [(s, o) for s, m in out for o in m["objects"] if o["x_m"] > 30.0 and s > 2.0]
            first[name] = min(s for s, o in car if o["occ"] == "object")
            self.assertTrue(all(o["exist"] > 0.9 for _, o in car))
        self.assertLess(first["on"], 2.15)                                    # hidden from the van's first box on
        self.assertGreater(first["off"], 2.4)                                 # ... not only once the van is confirmed

    def test_an_objects_own_box_does_not_hide_it(self):
        ex = {}
        for name, cfg in (("off", {}), ("on", {"in_box_occl": True})):
            _, out = run(cfg, lambda s, k: [box(40.0, 0.2, 1.8, 1.5)] if k % 2 == 0 else [], T_s=4.0)
            ex[name] = [m["objects"][0]["vis"] for s, m in out[20:] if m["objects"]]
        self.assertEqual(ex["on"], ex["off"])


class LowList(unittest.TestCase):
    def low_box(self, x=30.0):
        b = box(x, 2.0, 1.8, 1.5)
        b["class_conf"] = 0.2
        return b

    def test_a_low_box_starts_nothing(self):
        for cfg in ({}, {"in_low": True}):
            rd, out = run(cfg, lambda s, k: [], low=lambda s, k: [self.low_box()], T_s=5.0, v=0.0)
            self.assertEqual(len(rd.tracker.objects), 0)

    def test_a_low_box_continues_an_object_and_confirms_nothing(self):
        """A car boxed for 1 s (not yet confirmed by its sightings alone), then only in the low list."""
        def things(s, k):
            return [box(30.0, 2.0, 1.8, 1.5)] if s < 3.0 else []
        hits = {}
        for name, cfg in (("off", {}), ("on", {"in_low": True})):
            rd, out = run(cfg, things, low=lambda s, k: [] if s < 3.0 else [self.low_box()], T_s=10.0, v=0.0)
            o = [m["objects"][0] if m["objects"] else None for s, m in out]
            hits[name] = (o[58]["hits"], o[-1]["hits"] if o[-1] else None, o[-1]["state"] if o[-1] else None,
                          o[70]["state"], o[110]["state"] if o[110] else None)
        self.assertEqual(hits["off"][1:3], (None, None))                      # unseen for 5 s: gone
        self.assertEqual(hits["on"][0], hits["on"][1])                        # low boxes: no sighting counted
        self.assertEqual(hits["on"][2], "coast")                              # seen, weakly
        self.assertEqual(hits["on"][3], "confirmed")                          # (the first low boxes: still confirmed)

    def test_a_low_box_is_sloppier_and_its_class_weighs_less(self):
        hi = dets_of({"in_low": True}, box(30.0, 2.0, 1.8, 1.5))[0]
        lo = dets_of({"in_low": True}, box(30.0, 2.0, 1.8, 1.5), low=True)[0]
        self.assertGreater(lo["R"][0], 1.5 * hi["R"][0])
        self.assertGreater(lo["R"][2], 2.0 * hi["R"][2])
        self.assertAlmostEqual(lo["attrs"]["weight"], 0.3)
        self.assertEqual(lo["x"][2], 1.0)
        self.assertEqual(dets_of({}, box(30.0, 2.0, 1.8, 1.5), low=True), [])  # without the knob: not read

    def test_a_missing_object_goes_faster_when_the_low_list_would_have_shown_it(self):
        """A false object (three boxes, then nothing in either list): with the low list read, "not even a low box"
        tells against it more than a plain miss."""
        ex = {}
        for name, cfg in (("off", {}), ("on", {"in_low": True})):
            rd = Radar({"tracker": cfg, "tentative": True})
            rd.on_calib(CAL)
            e = None
            for i in range(60):
                t = 20 + i * 0.05
                rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 6.0, "yaw_rate_rps": 0.0})
                if i % 4 == 0:
                    rd.on_road_users(msg(t, [box(30.0 - 6.0 * i * 0.05, 2.0, 1.8, 1.5)] if i < 12 else [], low=[]))
                m = rd.frame(int(t * NS))
                if i == 28:
                    e = m["objects"][0]["exist"] if m["objects"] else 0.0
            ex[name] = e
        self.assertLess(ex["on"], ex["off"])


class Appearance(unittest.TestCase):
    def xcfg(self, **kw):
        class Sen:
            kind = "camera_boxes"
            emb_kind = 1
        return T.Tracker(None, dict({"in_app": True}, **kw))._xcfg("cam", Sen())

    def lr(self, d, gap=1.0, dark=0.0, **kw):
        n = 71
        a, b = [0.0] * n, [0.0] * n
        a[0] = b[0] = 1.0
        a[2] = b[2] = 1.0
        b[12] = d
        return C.radar_app_lr(self.xcfg(**kw), dark, a, b, gap)

    def test_the_likelihood_ratio_is_bounded_monotone_and_weaker_at_night(self):
        ds = [0.02, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 40.0]
        for dark, B in ((0.0, 8.0), (1.0, 3.0)):
            for gap in (0.2, 1.0, 2.0, 4.0, 9.0):
                v = [self.lr(d, gap, dark) for d in ds]
                self.assertTrue(all(1.0 / B - 1e-9 <= x <= B + 1e-9 for x in v))
                self.assertTrue(all(a >= b - 1e-9 for a, b in zip(v, v[1:])), (dark, gap, v))
                self.assertAlmostEqual(v[0], B, places=6)
                self.assertLess(v[-1], 0.4)
        self.assertGreater(self.lr(1.2, gap=4.0), self.lr(1.2, gap=1.0))       # a look changes with time
        self.assertAlmostEqual(self.lr(0.3, dark=0.5), math.sqrt(8.0 * 3.0), places=6)
        self.assertEqual(self.lr(0.3, app_bound=(1.0, 1.0)), 1.0)

    def test_a_vector_of_an_unknown_kind_or_none_at_all_says_nothing(self):
        E = emb(1)
        base = None
        for kind, e in ((None, None), ("yuv3x3", None), ("something-new", E), ("yuv3x3", E[:20])):
            _, out = run({"in_app": True}, lambda s, k: [box(30.0, 2.0, 1.8, 1.5, **({"emb": e} if e else {}))], T_s=3.0, v=0.0,
                         kind=kind)
            j = json.dumps([m for _, m in out], sort_keys=True)
            base = base or j
            self.assertEqual(j, base)

    def test_the_memory_is_the_objects_whole_first_list_boxes(self):
        E1, E2 = emb(1), emb(2)
        rd, _ = run({"in_app": True, "in_flags": True},
                    lambda s, k: [box(30.0, 2.0, 1.8, 1.5, emb=E1 if k < 6 else E2, **({} if k < 6 else {"occ": 0.4, "occ_edges": "r"}))],
                    T_s=4.0, v=0.0, kind="yuv3x3")
        o = rd.tracker.objects[0]
        xr = rd.tracker.hyp.xrow(o.slot)
        want = [((b + 128) % 256 - 128) / 32.0 for b in bytes.fromhex(E1)]
        self.assertEqual(xr[0], 6.0)                                          # the covered boxes did not teach it
        self.assertTrue(np.allclose(xr[12:12 + 27], want, atol=1e-9))
        self.assertEqual(xr[2], 1.0)

    def test_how_dark_the_scene_is_is_read_off_the_vectors(self):
        def e(level):
            return bytes([int(32 * level) & 255] * 9 + [0] * 18).hex()
        for level, dark in ((0.2, 0.0), (-1.6, 1.0), (-0.85, 0.5)):
            rd, _ = run({"in_app": True}, lambda s, k: [box(30.0, 2.0, 1.8, 1.5, emb=e(level))], T_s=2.0, v=0.0, kind="yuv3x3")
            self.assertAlmostEqual(rd.tracker.hyp.xstate()[2], dark, places=1)
        rd, _ = run({"in_app": True, "app_dark": 1.0}, lambda s, k: [box(30.0, 2.0, 1.8, 1.5, emb=e(0.2))], T_s=1.0, v=0.0,
                    kind="yuv3x3")
        self.assertEqual(rd.tracker.hyp.xstate()[2], 1.0)

    def test_a_dropped_object_that_comes_back_looking_the_same_is_itself_again(self):
        E1, E2 = emb(1), emb(2)

        def things(back):
            return lambda s, k: [box(30.0, 3.0, 1.8, 1.5, emb=E1)] if s < 2.5 else [] if s < 9.0 else [box(30.0, 3.0, 1.8, 1.5, emb=back)]
        ids = {}
        for name, cfg, back in (("off", {}, E1), ("same", {"in_app": True, "in_reid": True}, E1),
                                ("other", {"in_app": True, "in_reid": True}, E2)):
            _, out = run(cfg, things(back), T_s=12.0, v=0.0, kind="yuv3x3")
            ids[name] = ({o["id"] for s, m in out for o in m["objects"] if s < 8.0}, {o["id"] for s, m in out for o in m["objects"] if s > 9.5})
            self.assertTrue(all(not m["objects"] for s, m in out if 8.2 < s < 9.0))     # it WAS dropped
        self.assertNotEqual(ids["off"][0], ids["off"][1])
        self.assertEqual(ids["same"][0], ids["same"][1])
        self.assertNotEqual(ids["other"][0], ids["other"][1])

    def test_two_cars_of_a_row_seen_in_turn_are_told_apart_sooner(self):
        """A parked car boxed six times; then, while it is not boxed, the car one place further down the row is,
        for a second: the story "the same car" holds on for as long as the look of the box is not asked."""
        EA, EB = emb(11), emb(12)

        def things(s, k):
            if k < 6:
                return [box(30.0, 3.0, 1.8, 1.5, emb=EA)]
            return [box(35.5, 3.0, 1.8, 1.5, emb=EB)]
        t_two = {}
        for name, cfg in (("off", {}), ("on", {"in_app": True})):
            rd = Radar({"tracker": cfg, "tentative": True})
            rd.on_calib(CAL)
            t_two[name] = None
            for i in range(120):
                t = 20 + i * 0.05
                rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": 0.0, "yaw_rate_rps": 0.0, "standstill": True})
                if i % 4 == 0:
                    rd.on_road_users(msg(t, things(i * 0.05, i // 4), kind="yuv3x3"))
                m = rd.frame(int(t * NS))
                xs = sorted(o["x_m"] for o in m["objects"])
                if t_two[name] is None and len(xs) >= 2 and xs[0] < 32.5 and xs[-1] > 34.5:
                    t_two[name] = i * 0.05
        self.assertIsNotNone(t_two["on"])
        self.assertTrue(t_two["off"] is None or t_two["on"] < t_two["off"], t_two)


class Lamps(unittest.TestCase):
    def test_white_lamps_say_the_front_faces_us_and_a_new_object_comes_the_other_way(self):
        got = {}
        for col in ("red", "white"):
            rd, _ = run({"in_lamp_colour": True}, lambda s, k: [box(40.0, 3.0, 1.8, 1.5, lamps=lamps(40.0, c=col))], T_s=0.1)
            o = rd.tracker.objects[0]
            got[col] = (float(o.xs[M.CV][M.VX]), float(o.xs[M.CA][M.VX]), float(o.e[2]), float(o.mu[M.STAND]))
        self.assertGreater(got["red"][0], 5.0)                                # with the traffic, or
        self.assertLess(got["red"][1], -5.0)                                  # ... the other way
        self.assertLess(abs(got["red"][2]), 0.3)
        self.assertLess(got["white"][0], -5.0)                                # both moving stories: oncoming
        self.assertLess(got["white"][1], -5.0)
        self.assertGreater(abs(got["white"][2]), math.pi - 0.3)               # its front points at us
        self.assertEqual(got["white"][3], got["red"][3])                      # "it stands" keeps its share
        rd, _ = run({}, lambda s, k: [box(40.0, 3.0, 1.8, 1.5, lamps=lamps(40.0, c="white"))], T_s=0.1)
        self.assertGreater(float(rd.tracker.objects[0].xs[M.CV][M.VX]), 5.0)  # without the knob: as before

    def test_an_oncoming_cars_speed_is_known_sooner_with_its_lamps(self):
        v = {}
        for name, cfg in (("off", {}), ("on", {"in_lamp_colour": True})):
            _, out = run(cfg, lambda s, k: [box(70.0 - 24.0 * s, 3.2, 1.8, 1.5, lamps=lamps(70.0 - 24.0 * s, c="white"))], T_s=1.0)
            v[name] = out[12][1]["objects"][0]["vx_mps"]
        self.assertLess(v["on"], v["off"] - 3.0)                              # truth: -12 m/s over the ground

    def test_one_wrong_colour_does_not_turn_a_car_round(self):
        def things(s, k):
            return [box(30.0, 0.3, 1.8, 1.5, lamps=lamps(30.0, c="white" if k == 5 else "red"))]
        rd, _ = run({"in_lamp_colour": True}, things, T_s=3.0, v=0.0)
        self.assertLess(abs(float(rd.tracker.objects[0].e[2])), 0.4)

    def test_the_lamp_pair_is_a_ruler_whose_prior_is_used_once(self):
        """A car 50 m ahead at our speed in the dark, its box height jittering by a fifth: the pair's separation
        says how far it is to what a car's lamp separation is known (once), and from then on how the distance
        changes."""
        res = {}
        for name, cfg in (("off", {}), ("on", {"in_lamp_size": True})):
            rng = random.Random(3)
            rd, out = run(cfg, lambda s, k: [box(50.0, 0.3, 1.8, 1.5 * (1 + rng.gauss(0, 0.2)), lamps=lamps(50.0))], T_s=6.0)
            xs = [m["objects"][0]["x_m"] for s, m in out[20:] if m["objects"]]
            res[name] = (max(xs) - min(xs), out[20][1]["objects"][0]["std"][0], rd.tracker.hyp.xrow(rd.tracker.objects[0].slot))
        self.assertLess(res["on"][0], 0.75 * res["off"][0])                   # the range wanders less
        self.assertLess(res["on"][1], 0.9 * res["off"][1])                    # and is better known from the start
        self.assertLessEqual(res["on"][2][4], T.CFG["ruler_uses"])            # the prior: so many shares, no more
        self.assertGreater(res["on"][2][10], 0.0)                             # the last pair is remembered

    def test_a_pair_on_the_wrong_blobs_is_not_taken(self):
        """One pair in ten is found on other blobs (a separation of 0.6-0.9 of the lamps'): the object's range does
        not follow it."""
        res = {}
        for name, bad in (("clean", False), ("wrong", True)):
            rd, out = run({"in_lamp_size": True},
                          lambda s, k: [box(40.0, 0.3, 1.8, 1.5, lamps=lamps(40.0, S=1.2 * (0.65 if bad and k % 5 == 3 else 1.0)))], T_s=6.0)
            res[name] = [m["objects"][0]["x_m"] for s, m in out[20:] if m["objects"]]
        self.assertLess(max(abs(a - b) for a, b in zip(res["clean"], res["wrong"])), 2.5)


class Rulers(unittest.TestCase):
    def test_the_width_of_a_car_seen_from_behind_says_how_far_it_is(self):
        """A car straight ahead at our speed: its box width against a car's width (1.8 m to a tenth) is a second
        range beside the box height (1.6 m to a fifth)."""
        res = {}
        for name, cfg in (("off", {}), ("on", {"in_width": True})):
            rd, out = run(cfg, lambda s, k: [box(45.0, 0.2, 1.8, 1.7)], T_s=3.0)           # a tall car: the height reads near
            o = out[-1][1]["objects"][0]
            res[name] = (o["x_m"], o["std"][0], rd.tracker.hyp.xrow(rd.tracker.objects[0].slot))
        self.assertLess(abs(res["on"][0] - 45.0), abs(res["off"][0] - 45.0))
        self.assertLessEqual(res["on"][2][5], T.CFG["ruler_uses"])
        self.assertGreater(res["on"][2][5], 0.0)

    def test_a_side_view_is_no_ruler(self):
        rd, out = run({"in_width": True}, lambda s, k: [box(30.0, 6.0 - 3.0 * s, 4.5, 1.5)], T_s=2.0, v=0.0)
        o = rd.tracker.objects[0]
        self.assertEqual(rd.tracker.hyp.xrow(o.slot)[5], 0.0)

    def test_the_roads_vertical_curvature_is_in_a_far_objects_own_angle_offset(self):
        sd = {}
        for k in (0.0, 1 / 3000.0):
            for x in (20.0, 60.0):
                rd, _ = run({"grade_k": k}, lambda s, kk: [box(x, 0.3, 1.8, 1.5)], T_s=0.1)
                sd[(k > 0, x)] = math.degrees(math.sqrt(rd.tracker.objects[0].P[M.DP, M.DP]))
        self.assertAlmostEqual(sd[(False, 20.0)], sd[(False, 60.0)], places=9)
        self.assertAlmostEqual(math.sqrt(sd[(True, 60.0)] ** 2 - sd[(False, 60.0)] ** 2), math.degrees(60.0 / 6000.0), places=2)
        self.assertLess(sd[(True, 20.0)], sd[(False, 20.0)] + 0.05)


class ScaleRate(unittest.TestCase):
    def test_values_that_share_their_baseline_are_not_counted_as_independent(self):
        """A car 40 m ahead, closing at 4 m/s; scale rates over 0.8 s every 0.2 s, each with the same error (they are
        the same data): taken as independent the velocity is believed four times too well."""
        def things(s, k):
            x = 40.0 - 4.0 * s
            return [box(x, 0.2, 1.8, 1.5, scale_rate=4.0 / x + 0.02, scale_rate_std=0.02, scale_dt_s=0.8)]
        sd = {}
        for name, cfg in (("off", {}), ("on", {"in_scale": True})):
            _, out = run(cfg, things, T_s=3.0)
            sd[name] = out[-1][1]["objects"][0]["std"][2]
        self.assertGreater(sd["on"], 1.1 * sd["off"])

    def test_a_lamp_only_value_far_off_is_taken_for_little(self):
        """A scale rate from the lamp pair alone, four of its stated sigma off (the blobs changed): inside the gate,
        so it is taken; with Huber's weight for half of what it would move the velocity by."""
        def things(s, k):
            return [box(40.0, 0.2, 1.8, 1.5, scale_rate=0.12 if k == 8 else 0.0, scale_rate_std=0.03, scale_dt_s=0.2,
                        scale_src="lamps")]
        v = {}
        for name, cfg in (("off", {}), ("on", {"in_scale": True})):
            _, out = run(cfg, things, T_s=1.75)
            v[name] = abs(out[-1][1]["objects"][0]["vx_mps"] - 12.0)
        self.assertGreater(v["off"], 0.5)
        self.assertLess(v["on"], 0.6 * v["off"])


class Hypotheses(unittest.TestCase):
    def test_the_rows_memory_is_written_down_and_restored_with_the_hypotheses(self):
        rd, _ = run(dict(ALL), lambda s, k: [box(30.0, 2.0, 1.8, 1.5, emb=emb(1), lamps=lamps(30.0)),
                                             box(45.0, -1.0, 1.8, 1.5, emb=emb(2))], T_s=2.0, kind="yuv3x3")
        d = rd.tracker.hyp.dump()
        self.assertIn("x", d)
        H = C.Hyp(1, 1, 1, 1, 1.0, 1, 1.0, 1)
        H.load(d)
        self.assertEqual(H.dump(), d)
        for o in rd.tracker.objects:
            self.assertEqual(H.xrow(o.slot), rd.tracker.hyp.xrow(o.slot))
        # without the knobs the hypotheses hold no such memory, and their record is what it was
        rd, _ = run({}, lambda s, k: [box(30.0, 2.0, 1.8, 1.5)], T_s=1.0)
        self.assertNotIn("x", rd.tracker.hyp.dump())
        self.assertIsNone(rd.tracker.hyp.xrow(0))

    def test_everything_on_runs_a_busy_scene(self):
        rng = random.Random(9)

        def things(s, k):
            o = []
            for n in range(8):
                x = 15.0 + 7.0 * n - (0.0 if n % 2 else 3.0 * s)
                if x > 9.0 and rng.random() > 0.3:
                    o.append(box(x, 3.0 if n % 2 else -0.2, 1.8, 1.5, emb=emb(n), trunc="", lamps=lamps(x, c=rng.choice(("red", "white", "unknown"))),
                                 **({"occ": 0.3, "occ_by": 0, "occ_edges": rng.choice(("r", "b", "l", ""))} if rng.random() < 0.3 else {})))
            return o
        rd, out = run(dict(ALL), things, T_s=8.0, kind="yuv3x3", low=lambda s, k: [box(20.0 + 9.0 * rng.random(), 3.0, 1.8, 1.5, emb=emb(50))])
        self.assertGreater(len(out[-1][1]["objects"]), 2)
        for _, m in out:
            for o in m["objects"]:
                self.assertTrue(all(math.isfinite(v) for v in (o["x_m"], o["y_m"], o["vx_mps"], o["exist"], *o["std"])))


if __name__ == "__main__":
    unittest.main()
