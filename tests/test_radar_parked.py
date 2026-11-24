"""Parked cars (docs/RADAR.md, section 9): the knobs `sil_bearing`, `ctx_prior`, `size_learn` of the tracker and `cov`,
`path_normal` of the radar box. Each off: the radar is what it was (tests/test_native.py's goldens and digests, the
detector-input goldens). Each on: what it is for, on the simulator; and the recorded scans of
tests/data/make_radar_golden.py --parked."""
import json
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chris_integrator.native import radar as C
from chris_integrator.radar import models as M
from chris_integrator.radar import tracker as TK
from chris_integrator.sim import radar_score

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def standing(r, lo=8.0, hi=60.0):
    """(output object, actor, along error) of every confirmed output matched to a standing actor."""
    out, prev = [], {}
    for f in r["frames"]:
        outs = [dict(o, oid=o["id"], x=o["x_m"], y=o["y_m"]) for o in f["objects"]]
        m = radar_score.match(outs, f["actors"], prev)
        prev = {outs[i]["oid"]: a["id"] for i, (a, _, _) in m.items()}
        for i, (a, al, _) in m.items():
            if not a["moving"] and lo <= a["range"] < hi:
                out.append((outs[i], a, al))
    return out


class Knobs(unittest.TestCase):
    def test_all_are_off_by_default_or_said_so(self):
        # (a default that changes is a deliberate act: the goldens are recorded anew with it)
        from chris_integrator.radar import box
        for k in ("sil_bearing", "ctx_prior", "size_learn"):
            self.assertIn(k, TK.CFG)
        for k in ("std_rho", "path_normal"):
            self.assertIn(k, box.CFG)

    def test_the_place_correlation_is_in_the_message_when_asked(self):
        r0 = radar_score.run("parked_street", seed=1)
        self.assertTrue(all("std_rho" not in o for f in r0["frames"] for o in f["objects"]))
        radar_score.KEEP += ("std_rho",)
        try:
            r1 = radar_score.run("parked_street", seed=1, cfg={"std_rho": True})
        finally:
            radar_score.KEEP = radar_score.KEEP[:-1]
        objs = [o for f in r1["frames"] for o in f["objects"]]
        self.assertTrue(objs and all(-1.0 <= o["std_rho"] <= 1.0 for o in objs))
        # a camera's place is unsure along the line of sight: beside the axis x and y err together
        rho = [o["std_rho"] * (1.0 if o["y_m"] > 0 else -1.0) for o in objs
               if o["state"] == "confirmed" and abs(o["y_m"]) > 2.0 and o["x_m"] > 15.0]
        self.assertGreater(np.median(rho), 0.5)

    def test_beside_the_path_is_across_the_path(self):
        # on a bend the offset from the path is smaller than the lateral difference (its cosine), never larger
        radar_score.KEEP += ("path_offset_m",)
        try:
            a = radar_score.run("curve_parked", seed=1)
            b = radar_score.run("curve_parked", seed=1, cfg={"path_normal": True})
        finally:
            radar_score.KEEP = radar_score.KEEP[:-1]
        pa = [o["path_offset_m"] for f in a["frames"] for o in f["objects"]]
        pb = [o["path_offset_m"] for f in b["frames"] for o in f["objects"]]
        self.assertEqual(len(pa), len(pb))
        self.assertTrue(all(abs(y) <= abs(x) + 0.006 for x, y in zip(pa, pb) if x is not None and y is not None))
        self.assertTrue(any(abs(y) < abs(x) - 0.05 for x, y in zip(pa, pb) if x is not None and y is not None))


class StandPrior(unittest.TestCase):
    def test_it_only_raises(self):
        a = np.zeros((2, TK.ROW))
        mu = TK.OFF["mu"][0]
        a[0, mu:mu + 3] = (0.4, 0.5, 0.1)
        a[1, mu:mu + 3] = (0.95, 0.04, 0.01)
        C.radar_stand_prior(a, [0, 1], 0.8)
        np.testing.assert_allclose(a[0, mu:mu + 3], (0.8, 0.2 * 5 / 6, 0.2 / 6))
        np.testing.assert_allclose(a[1, mu:mu + 3], (0.95, 0.04, 0.01))

    def test_a_parked_street_tells_its_next_car(self):
        raised = []
        orig = TK.Tracker._context

        def ctx(self, o, pose):
            p0 = float(o.mu[M.STAND])
            orig(self, o, pose)
            raised.append(float(o.mu[M.STAND]) - p0)
        TK.Tracker._context = ctx
        try:
            radar_score.run("parked_street", seed=1, cfg={"tracker": {"ctx_prior": True}})
            street = list(raised)
            del raised[:]
            radar_score.run("adjacent", seed=1, cfg={"tracker": {"ctx_prior": True}})
            traffic = list(raised)
        finally:
            TK.Tracker._context = orig
        self.assertGreater(sum(1 for d in street if d > 0.2), 0.3 * len(street))      # beside a row of parked cars
        self.assertTrue(all(d >= 0.0 for d in street + traffic))                     # never lowered
        # among vehicles that drive (and the roadside's few parked ones) hardly a new one is told that it stands
        self.assertLess(sum(1 for d in traffic if d > 0.2), 0.15 * max(len(traffic), 1))

    def test_parked_cars_are_called_standing_sooner_and_moving_traffic_as_before(self):
        def share(name, cfg):
            v = standing(radar_score.run(name, seed=2, cfg=cfg))
            return sum(1 for o, _, _ in v if o["motion"] == "stationary") / len(v), sum(1 for o, _, _ in v if o["motion"] == "moving") / len(v)
        s0, m0 = share("curve_parked", None)
        s1, m1 = share("curve_parked", {"tracker": {"ctx_prior": True}})
        self.assertGreater(s1, s0 + 0.05)
        self.assertLessEqual(m1, m0 + 0.01)


class SizeLearn(unittest.TestCase):
    def test_the_fit(self):
        m0, v0 = math.log(1.6), 0.04
        self.assertEqual(C.radar_size_fit([], m0, v0, 3.0, 0.1, 0.25), (m0, v0))
        rng = np.random.default_rng(1)
        s = [(math.log(1.5) + 0.05 * rng.standard_normal(), 0.05 ** 2) for _ in range(40)]
        mu, var = C.radar_size_fit(s, m0, v0, 3.0, 0.1, 0.25)
        self.assertAlmostEqual(mu, math.log(1.5), delta=0.02)
        self.assertAlmostEqual(var, 0.01, delta=0.004)              # the floor: the samples scatter no more than they say
        # three objects say little against the built-in prior; one wild track does not steer it
        mu3, var3 = C.radar_size_fit(s[:3], m0, v0, 3.0, 0.1, 0.25)
        self.assertGreater(mu3, mu + 0.01)
        self.assertGreater(var3, 0.015)
        muw, _ = C.radar_size_fit(s + [(math.log(3.5), 0.05 ** 2)], m0, v0, 3.0, 0.1, 0.25)
        self.assertAlmostEqual(muw, mu, delta=0.01)
        mus, _ = C.radar_size_fit([(math.log(1.0), 1e-4)] * 50, m0, v0, 3.0, 0.1, 0.25)
        self.assertAlmostEqual(mus, m0 - 0.25, delta=1e-9)          # no further than the bound

    def _fit(self, name, seed=1):
        got = {}
        orig = TK.Tracker.manage

        def manage(self, T):
            orig(self, T)
            got["fit"], got["n"] = dict(self.size_fit), {k: len(v) for k, v in self.size_obs.items()}
        TK.Tracker.manage = manage
        try:
            r = radar_score.run(name, seed=seed, cfg={"tracker": {"size_learn": True}})
        finally:
            TK.Tracker.manage = orig
        return got, r

    def test_passed_cars_teach_the_kind_its_height(self):
        # the simulator's cars are 1.5 m, the built-in prior says 1.6 m +- 20 %
        got, r = self._fit("parked_street")
        self.assertGreaterEqual(got["n"]["car"], 15)
        mu, var = got["fit"]["car"]
        self.assertAlmostEqual(mu, math.log(1.5), delta=0.035)
        self.assertLess(math.sqrt(var), 0.13)
        # ... and the range of the cars that follow is read with it
        r0 = radar_score.run("parked_street", seed=1)
        e0 = [al for _, a, al in standing(r0, 15.0, 50.0)]
        e1 = [al for _, a, al in standing(r, 15.0, 50.0)]
        self.assertLess(abs(np.mean(e1)), abs(np.mean(e0)) - 0.2)

    def test_what_was_not_passed_teaches_nothing(self):
        # a queue ahead, on the car's own line: no parallax, the "range" would be the ground angle's
        got, _ = self._fit("queue")
        self.assertFalse(got["fit"])


class Silhouette(unittest.TestCase):
    def test_the_row_is_read_no_worse_and_its_cars_are_told_apart(self):
        res = {}
        for tag, cfg in (("off", None), ("on", {"tracker": {"sil_bearing": True}})):
            err = n = 0
            for seed in (1, 2):
                v = standing(radar_score.run("curve_parked", seed=seed, cfg=cfg), 8.0, 40.0)
                err += sum(abs(al) for _, _, al in v)
                n += len(v)
            res[tag] = err / n
        self.assertLess(res["on"], res["off"] * 1.05)


class Golden(unittest.TestCase):
    """The radar with the parked-car knobs on, held to what was recorded (make_radar_golden.py --parked)."""

    def test_recorded_calls_give_what_they_gave(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import radar_golden_lib as G
        cases, codec = G.load(os.path.join(DATA, "radar_golden_parked.npz"))
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
        for fn in ("Hyp.cycle", "radar_frame", "radar_size_fit"):
            self.assertGreaterEqual(seen.get(fn, 0), 4, fn)
        cyc = [codec.dec(c["args"]) for c in cases if c["fn"] == "Hyp.cycle"]
        self.assertTrue(all(len(a) == 20 and a[19][-1] == 1.0 for a in cyc))          # an xcfg with the silhouette on

    @unittest.skipIf(os.uname().machine != "x86_64", "the digests are of the development machine's arithmetic")
    def test_the_messages_on_simulated_drives_are_what_they_were(self):
        sys.path.insert(0, DATA)
        import make_radar_golden as MG
        with open(os.path.join(DATA, "radar_digests_parked.json")) as f:
            want = json.load(f)
        self.assertEqual(MG.parked_digests(), want)


if __name__ == "__main__":
    unittest.main()
