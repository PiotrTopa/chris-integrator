"""The range and the extent of a standing thing (docs/RADAR.md, section 11): the tracker's knobs `scale_iter`,
`scale_seed`, `scale_exact`, `near_axis`, `ext_size`. Each off: the radar is what it was (tests/test_native.py's goldens
and digests and the detector-input, parked-car and row goldens: none of them was recorded anew). Each on: what it is
for, on small worlds with known truth and on the simulator; the recorded scans of tests/data/make_radar_golden.py
--range; and the simulator's options the measurements use (the detector's scale rate as a preset, a fleet of sizes)."""
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chris_integrator.native import radar as C  # noqa: E402
from chris_integrator.radar import Radar  # noqa: E402
from chris_integrator.radar import models as M  # noqa: E402
from chris_integrator.radar import tracker as TK  # noqa: E402
from chris_integrator.sim import roaduser_sim as rs  # noqa: E402

import test_radar_inputs as TI  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NS = 10 ** 9
NEW = ("scale_iter", "scale_seed", "scale_exact", "near_axis", "ext_size")
RANGE = dict(scale_iter=True, scale_seed=True, scale_exact=True, near_axis=True, ext_size=True)


def drive(cfg, thing, T_s=5.0, v=lambda s: 12.0, w=lambda s: 0.0, every=4, base=0.4):
    """The radar with `cfg` as its tracker's configuration on a car driving v(s), turning w(s) (its feed every 0.05 s,
    a scan every `every` of them). thing(pose, pose a baseline ago or None, s, k) -> the scan's object or None.
    Returns [(s, the radar's message, the car's true (X, Y, heading))]."""
    rd = Radar({"tracker": cfg})
    rd.on_calib(TI.CAL)
    X = Y = psi = 0.0
    poses, out = [], []
    nb = int(round(base / 0.05))
    for i in range(int(round(T_s / 0.05))):
        s = i * 0.05
        t = 20 + s
        rd.on_ego({"ts_mono_ns": int(t * NS), "speed_mps": v(s), "yaw_rate_rps": w(s), "standstill": False})
        poses.append((X, Y, psi))
        if i % every == 0:
            k = i // every
            o = thing((X, Y, psi), poses[i - nb] if i >= nb and k >= 1 else None, s, k)
            rd.on_road_users(TI.msg(t, [o] if o else []))
        out.append((s, rd.frame(int(t * NS)), (X, Y, psi)))
        vv, ww = v(s), w(s)
        X += vv * 0.05 * math.cos(psi + 0.5 * ww * 0.05)
        Y += vv * 0.05 * math.sin(psi + 0.5 * ww * 0.05)
        psi += ww * 0.05
    return out


def rel(p, gx, gy):
    dx, dy = gx - p[0], gy - p[1]
    return math.cos(p[2]) * dx + math.sin(p[2]) * dy, -math.sin(p[2]) * dx + math.cos(p[2]) * dy


def standing(gx, gy, wid, h, cls="car", base=0.4, sd=0.01):
    """A standing thing whose near face is at the ground point (gx, gy), boxed while it is 9 m or more ahead, with the
    scale rate the detector would give: the logarithm of its depth a baseline ago over its depth now, per second."""
    def thing(pose, prev, s, k):
        x, y = rel(pose, gx, gy)
        if x < 9.0:
            return None
        kw = {}
        if prev is not None:
            kw = dict(scale_rate=math.log(rel(prev, gx, gy)[0] / x) / base, scale_rate_std=sd, scale_dt_s=base)
        return TI.box(x, y, wid, h, cls=cls(k) if callable(cls) else cls, **kw)
    return thing


def oldest(f):
    return max(f["objects"], key=lambda q: q["age_s"]) if f["objects"] else None


def at(out, s):
    return next((f, p) for q, f, p in out if abs(q - s) < 1e-6)


class KnobsOff(unittest.TestCase):
    def test_the_knobs_are_off(self):
        for k in NEW:
            self.assertIs(TK.CFG[k], False, k)

    def test_with_the_knobs_off_a_scan_with_the_constants_sent_is_what_it_was(self):
        """The knobs' constants go to the native scan only with a knob on; a scan given them all as "off" (an earlier
        knob on, none of these) is the scan of before: the messages of a drive are the same."""
        th = standing(70.0, 3.0, 1.8, 1.5)
        a = drive({"in_scale": True}, th, T_s=3.0)
        b = drive({"in_scale": True, "scale_iter_n": 3, "scale_seed_sigma": 0.3, "scale_seed_k": 9.0, "ext_size_step": 0.5}, th, T_s=3.0)
        self.assertEqual(json.dumps([f for _, f, _ in a], sort_keys=True), json.dumps([f for _, f, _ in b], sort_keys=True))


class ScaleRateIsRange(unittest.TestCase):
    def test_a_standing_van_taken_for_a_car_is_ranged_by_its_first_scale_rate(self):
        """A van 2.6 m tall, boxed as `car` (1.6 m to a fifth): its first box reads it at 0.7 of its range. Its scale
        rate is the car's speed over its depth, whatever its size. One linear step of that hyperbola leaves "it
        stands" short of the range with the spread it would have there; the next rate refutes it. Iterated: at the
        range the rate says, standing, from the first rate on."""
        th = standing(70.0, 3.0, 2.1, 2.6)
        off, on = drive({}, th, T_s=3.0), drive({"scale_iter": True}, th, T_s=3.0)
        for s in (0.8, 1.2):
            (fo, p), (fn, _) = at(off, s), at(on, s)
            true = rel(p, 70.0, 3.0)[0]
            o, n = oldest(fo), oldest(fn)
            self.assertLess(o["x_m"], true - 10.0, (s, o["x_m"], true))       # what it was: 13-18 m short, not standing
            self.assertLess(o["p_stationary"], 0.5)
            self.assertLess(abs(n["x_m"] - true), 2.0 * n["std"][0] + 0.5, (s, n["x_m"], true, n["std"]))
            self.assertLess(abs(n["x_m"] - true), 4.0)
            self.assertEqual(n["motion"], "stationary")
        # and the stated spread is the one the rate has at the range found (it stated less from a nearer range)
        (fn, p) = at(on, 2.0)
        n = oldest(fn)
        self.assertLess(abs(n["x_m"] - rel(p, 70.0, 3.0)[0]), 2.0 * n["std"][0])

    def test_the_iteration_changes_nothing_where_one_step_was_enough(self):
        """A car of the prior's height, ranged right by its first box: the iterated update is the linear one's to a
        few centimetres."""
        th = standing(60.0, 0.2, 1.8, 1.6)
        off, on = drive({}, th, T_s=3.0), drive({"scale_iter": True}, th, T_s=3.0)
        for s in (1.0, 2.0, 2.9):
            o, n = oldest(at(off, s)[0]), oldest(at(on, s)[0])
            self.assertLess(abs(o["x_m"] - n["x_m"]), 0.25, (s, o["x_m"], n["x_m"]))
            self.assertEqual(o["motion"], n["motion"])

    def test_a_scale_rate_is_the_depth_then_over_the_depth_now_while_the_car_brakes(self):
        """The car brakes at 3 m/s^2 towards a standing car; the rates' baseline is 0.8 s, over which the car was
        1.2 m/s faster on average than it is at its end. Read as the depth's rate of this moment the standing car is
        3 m near at 20-40 m with a stated spread of 0.5-1.3 m; with the car's pose at the baseline's far end it is where
        it is."""
        v = lambda s: max(16.0 - 3.0 * max(s - 1.0, 0.0), 2.0)   # noqa: E731
        th = standing(75.0, 0.3, 1.8, 1.5, base=0.8)
        off = drive({}, th, T_s=5.0, v=v, base=0.8)
        on = drive({"scale_exact": True}, th, T_s=5.0, v=v, base=0.8)
        for s in (2.4, 3.0, 3.6, 4.2):
            (fo, p), (fn, _) = at(off, s), at(on, s)
            true = rel(p, 75.0, 0.3)[0]
            o, n = oldest(fo), oldest(fn)
            self.assertLess(o["x_m"], true - 2.5, (s, o["x_m"], true))
            self.assertGreater(abs(o["x_m"] - true), 2.0 * o["std"][0])
            self.assertLess(abs(n["x_m"] - true), 0.4, (s, n["x_m"], true))
            self.assertEqual(n["motion"], "stationary")

    def test_a_scale_rate_on_a_bend_is_read_with_the_cameras_axis_as_it_pointed_then(self):
        """A bend of 110 m radius at 13 m/s: the camera's axis swings 4 degrees over a baseline of 0.6 s. A car standing
        beside the bend reads 0.4-0.6 m far at 25-45 m; with the pose of the baseline's far end, to 0.2 m."""
        w = lambda s: 13.0 / 110.0                              # noqa: E731
        gx, gy = 110.0 * math.sin(60.0 / 110.0) + 2.0, 110.0 * (1 - math.cos(60.0 / 110.0)) - 3.0
        th = standing(gx, gy, 1.8, 1.5, base=0.6)
        off = drive({"scale_iter": True}, th, T_s=3.2, v=lambda s: 13.0, w=w, base=0.6)
        on = drive({"scale_iter": True, "scale_exact": True}, th, T_s=3.2, v=lambda s: 13.0, w=w, base=0.6)
        eo, en = [], []
        for s in (1.2, 1.6, 2.0, 2.4, 2.8):
            (fo, p), (fn, _) = at(off, s), at(on, s)
            true = math.hypot(*rel(p, gx, gy))
            o, n = oldest(fo), oldest(fn)
            eo.append(math.hypot(o["x_m"], o["y_m"]) - true)
            en.append(math.hypot(n["x_m"], n["y_m"]) - true)
        self.assertGreater(min(eo), 0.3, eo)
        self.assertLess(max(abs(e) for e in en), 0.2, en)

    def test_the_knobs_together_leave_a_moving_car_alone(self):
        """A car driving ahead at the car's own speed less 3 m/s: its scale rates say nothing of its range, and the
        knobs do not make it stand."""
        def th(pose, prev, s, k):
            x = 40.0 - 3.0 * s
            kw = dict(scale_rate=math.log((x + 3.0 * 0.4) / x) / 0.4, scale_rate_std=0.01, scale_dt_s=0.4) if k >= 2 else {}
            return TI.box(x, 0.2, 1.8, 1.6, **kw)
        off, on = drive({}, th, T_s=4.0), drive(RANGE, th, T_s=4.0)
        o, n = oldest(off[-1][1]), oldest(on[-1][1])
        self.assertEqual(n["motion"], "moving")
        self.assertLess(abs(n["x_m"] - o["x_m"]), 1.0)
        self.assertLess(abs(math.hypot(n["vx_mps"], n["vy_mps"]) - 9.0), 1.5)


class OnTheSimulator(unittest.TestCase):
    def frames(self, cfg):
        sys.path.insert(0, os.path.join(os.path.dirname(DATA), "..", "tools"))
        import radar_trace as RT
        return RT.run("curve_parked", 5, False, {"tracker": cfg} if cfg else None, rs.det_params_arg("scale"))[3]["frames"]

    def test_a_parked_van_boxed_as_a_car_stands_at_its_range(self):
        """curve_parked, seed 5, with the detector's scale rate: the parked van 109 (2.6 m, boxed `car` first) was a
        car driving at 3-5 m/s at 0.6-0.8 of its range all the way down ("it stands" at the floor from its second
        scale rate on, given the place of the hypotheses that drive at every box). With the update iterated:
        standing, at its range."""
        sel = lambda fr: [f for f in fr if f.get("owner") == 109 and f["state"] != "tentative" and 15.0 <= f.get("range", 0) <= 45.0]  # noqa: E731
        off, on = sel(self.frames(None)), sel(self.frames(dict(scale_iter=True)))
        self.assertGreater(len(off), 12)
        self.assertGreater(len(on), 12)
        med = lambda v: sorted(v)[len(v) // 2]                                  # noqa: E731
        self.assertLess(med([f["mu"][0] for f in off]), 0.2)
        self.assertLess(med([f["along_c"] for f in off]), -3.0)
        self.assertGreater(med([f["mu"][0] for f in on]), 0.9)
        self.assertLess(abs(med([f["along_c"] for f in on])), 2.0)


class Extent(unittest.TestCase):
    def test_the_size_of_a_vehicle_follows_its_height(self):
        L, vL, W, vW = M.extent_of_height(1.5)
        self.assertEqual((L, round(W, 3)), (4.5, 1.8))
        self.assertAlmostEqual(math.sqrt(vL) / L, 0.12, places=6)
        self.assertAlmostEqual(M.extent_of_height(2.6)[0], 6.0, places=6)
        self.assertAlmostEqual(M.extent_of_height(3.4)[0], 9.0, places=6)
        self.assertGreater(M.extent_of_height(1.5, 0.2)[3], vW)                 # an unknown height: a wider width
        self.assertGreater(M.extent_of_height(2.2, 0.2)[1], M.extent_of_height(2.2)[1])
        hs = [1.0 + 0.05 * k for k in range(70)]
        ls = [M.extent_of_height(h)[0] for h in hs]
        self.assertTrue(all(b >= a for a, b in zip(ls, ls[1:])))                # no vehicle is shorter for being taller

    def test_a_prior_is_exchanged_and_the_face_the_camera_saw_stays(self):
        """radar_ext_prior on one row: a length prior of 10 +- 2.2 m exchanged for 4.5 +- 0.54 m: the length is the new
        prior's, and the centre of every hypothesis comes nearer the camera by half the difference."""
        rd = Radar({"tracker": {}})
        rd.on_calib(TI.CAL)
        rd.on_ego({"ts_mono_ns": 20 * NS, "speed_mps": 0.0, "yaw_rate_rps": 0.0, "standstill": True})
        rd.on_road_users(TI.msg(20.0, [TI.box(40.0, 0.0, 2.5, 3.2, cls="truck")]))
        rd.frame(20 * NS)
        tr = rd.tracker
        o = tr.objects[0]
        self.assertAlmostEqual(float(o.e[0]), 10.0)
        x0 = float(o.x[M.PX])
        v0, v1 = 2.2 ** 2, 0.54 ** 2
        C.radar_ext_prior(tr.store.a, tr.hyp.rows(o.h), 0, 1.0 / v1 - 1.0 / v0, 4.5 / v1 - 10.0 / v0, 0.0, 0.0)
        self.assertAlmostEqual(float(o.e[0]), 4.5, places=6)
        self.assertAlmostEqual(math.sqrt(float(o.Pe[0, 0])), 0.54, places=6)
        self.assertAlmostEqual(float(o.x[M.PX]) - x0, -0.5 * (10.0 - 4.5), places=3)
        self.assertAlmostEqual(float(o.xs[0, M.PX]) - x0, -0.5 * (10.0 - 4.5), places=2)

    def test_a_standing_van_called_truck_is_a_van_long(self):
        """A van (2.6 m tall, 6 m long) the detector calls `truck`: born 10 m long, its centre 5 m behind its near
        face. Ranged by its scale rates its height is its own, and its length a van's: the centre within a metre."""
        th = standing(70.0, 3.0, 2.1, 2.6, cls="truck")
        off = drive({"scale_iter": True}, th, T_s=4.0)
        on = drive({"scale_iter": True, "ext_size": True}, th, T_s=4.0)
        (fo, p), (fn, _) = at(off, 3.6), at(on, 3.6)
        true = rel(p, 70.0, 3.0)[0]
        o, n = oldest(fo), oldest(fn)
        self.assertGreater(o["len"], 8.5)
        self.assertGreater(o["ctr"][0] - (true + 3.0), 1.2)
        self.assertLess(abs(n["len"] - 6.0), 1.2, n["len"])
        self.assertLess(abs(n["ctr"][0] - (true + 3.0)), 0.8)
        self.assertLess(abs(n["x_m"] - true), 0.6)                               # the near face: where it was


class Simulator(unittest.TestCase):
    def test_the_scale_rate_presets(self):
        self.assertEqual(rs.det_params_arg("scale"), {"scale_rate": "auto"})
        self.assertEqual(rs.det_params_arg("detin_scale"), dict(rs.DETIN, scale_rate="auto"))
        self.assertEqual(rs.det_params_arg("detin_pose_scale"), dict(rs.DETIN_POSE, scale_rate="auto"))
        self.assertEqual(rs.DetectorSim(seed=3, night=False, params={"scale_rate": "auto"}).p["scale_rate"], rs.SCALE_DAY)
        self.assertEqual(rs.DetectorSim(seed=3, night=True, params={"scale_rate": True}).p["scale_rate"], rs.SCALE_NIGHT)
        self.assertIsNone(rs.DetectorSim(seed=3, night=True).p.get("scale_rate"))
        sys.path.insert(0, os.path.join(os.path.dirname(DATA), "..", "tools"))
        import visual_radar_eval as VE
        self.assertEqual((VE.SCALE_DAY, VE.SCALE_NIGHT), (rs.SCALE_DAY, rs.SCALE_NIGHT))

    def test_a_fleet_of_sizes_behind_an_option(self):
        a = rs.scenario("parked_street", seed=2)
        self.assertTrue(all((q.L, q.W, q.H) == rs.DIMS["car"] for q in a.actors if q.kind == "car"))
        b, b2 = (rs.scenario("parked_street", seed=2, world={"dims_fleet": True}) for _ in range(2))
        self.assertEqual([(q.L, q.W, q.H) for q in b.actors], [(q.L, q.W, q.H) for q in b2.actors])
        cars = [q for q in b.actors if q.kind == "car"]
        hs = [q.H for q in cars]
        self.assertGreater(len(cars), 80)
        self.assertTrue(all(1.42 <= h <= 1.90 for h in hs))
        self.assertTrue(1.50 < sum(hs) / len(hs) < 1.64, sum(hs) / len(hs))
        self.assertGreater(sum(1 for h in hs if h > 1.58) / len(hs), 0.25)       # crossovers and taller
        self.assertTrue(all(1.63 <= q.W <= 2.0 and 3.6 <= q.L <= 4.8 for q in cars))
        # where it was parked and how it drives is what it was; a van's explicit size is the scenario's own
        self.assertEqual([(q.id, q.s0, q.lateral) for q in a.actors], [(q.id, q.s0, q.lateral) for q in b.actors])
        vans = [(x, y) for x, y in zip(a.actors, b.actors) if x.kind == "truck"]
        self.assertTrue(vans and all(abs(y.H / x.H - 1.0) < 0.12 for x, y in vans))


class Recorded(unittest.TestCase):
    def test_recorded_calls_give_what_they_gave(self):
        import radar_golden_lib as G
        cases, codec = G.load(os.path.join(DATA, "radar_golden_range.npz"))
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
        self.assertGreaterEqual(seen.get("radar_ext_prior", 0), 2)

    @unittest.skipIf(os.uname().machine != "x86_64", "the digests are of the development machine's arithmetic")
    def test_the_messages_on_simulated_drives_are_what_they_were(self):
        sys.path.insert(0, DATA)
        import make_radar_golden as MG
        with open(os.path.join(DATA, "radar_digests_range.json")) as f:
            want = json.load(f)
        self.assertEqual(MG.range_digests(), want)


if __name__ == "__main__":
    unittest.main()
