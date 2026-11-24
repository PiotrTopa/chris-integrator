"""The decision layer (chris_integrator/decide, plug-in `decide`): futures, threat levels on multi-modal fixture
scenarios, the alert logic, the view's flags and alert from it, and the simulator's scenarios through the stack."""
import math
import unittest

import numpy as np

from chris_integrator.decide import alerts as AL, fixtures as FX, futures as FU, threat as TH
from chris_integrator.mixer.testing import Bench
from chris_integrator.plugins.decide import Decide
from chris_integrator.trackers.roadusers_view import RadarDisplay

NS = 1_000_000_000


def run_fixture(name, cfg=None):
    b = Bench(Decide(dict({"rate_hz": 1000.0}, **(cfg or {}))))
    stages, events, msgs = [], [], []
    for msg, tr in FX.frames(name):
        for st, m in b.feed("radar", msg):
            a = next(x for x in m["alerts"] if x["id"] == "obstacle_ahead")
            stages.append((tr["T"], a["stage"], a["suppressed"]))
            events += [e for e in m["events"] if e["alert"] == "obstacle_ahead"]
            msgs.append(m)
    return stages, events, msgs, b


class Fixtures(unittest.TestCase):
    def test_collision_courses_warn_with_lead_time_and_the_rest_stays_silent(self):
        for name in FX.NAMES:
            stages, _, _, _ = run_fixture(name)
            expect = FX.scenario(name)[3]
            first_warn = next((t for t, s, sup in stages if s >= 3 and sup is None), None)
            if expect == "alert":
                tc = FX.collision_time(name)
                self.assertIsNotNone(first_warn, name)
                self.assertGreaterEqual(tc - first_warn, 1.2, name)          # a warning at least 1.2 s ahead
            else:
                self.assertEqual(max(s for _, s, _ in stages), 0, name)

    def test_the_unconditioned_mode_decides_and_the_conditioned_figure_is_reported(self):
        # the crossing car may yield (conditioned 0.5) but the threat takes the unconditioned 0.8 of "go"
        _, _, msgs, _ = run_fixture("crossing")
        o = next(r for m in msgs[10:] for r in m["objects"] if r["level"] >= 2)
        self.assertGreater(o["p_coll"], o["p_coll_cond"])
        self.assertEqual(o["why"]["mode"], "go")
        self.assertIn(o["why"]["measure"], ("threat_number", "time_margin"))

    def test_every_decision_says_why(self):
        _, events, _, _ = run_fixture("lead_braking")
        raise_ = next(e for e in events if e["event"] in ("raise", "escalate") and e["stage"] >= 3)
        for k in ("measure", "mode", "ego_mode", "p_coll", "tn", "tm_s", "ttc_s"):
            self.assertIn(k, raise_["reason"])
        self.assertEqual(raise_["reason"]["mode"], "brake")
        self.assertEqual(raise_["mode"], "shadow")
        self.assertFalse(raise_["active"])                   # shadow: logged, never active
        self.assertTrue(raise_["would_show"])


def plan(xs, ys_left):
    """A road model's plan (y right, as it gives it) as the decision layer's path."""
    return FU.plan_path({"ts_mono_ns": 0, "plan": {"x": [float(x) for x in xs], "y": [-float(y) for y in ys_left]}}, 0)


STRAIGHT = plan([0.0, 100.0], [0.0, 0.0])           # the road along x: its direction 0 everywhere


def judged(o, ms, t, v=11.0, lane=None, path=STRAIGHT):
    """One object with the given futures [(name, p0, p, x, y, sx, sy, rxy)] on the grid t against the car driving
    straight on at v, on a straight road: what the native core made of it (threat.Explained)."""
    fut = [{"m": nm, "p0": p0, "p": pc, "x": np.asarray(x).tolist(), "y": np.asarray(y).tolist(), "sx": np.asarray(sx).tolist(),
            "sy": np.asarray(sy).tolist(), "rxy": np.asarray(r).tolist()} for nm, p0, pc, x, y, sx, sy, r in ms]
    msg = {"objects": [dict(o, fut=fut, **({"lane": lane} if lane else {}))], "ego": {"speed_mps": v},
           "fut": {"t": t.tolist()}, "ego_fut": {"modes": [{"p": 1.0, "m": "path", "x": (v * t).tolist(), "y": (0.0 * t).tolist(),
                                                           "v": (v + 0.0 * t).tolist()}]}}
    return TH.ThreatModel().explain(msg, path=path)


def footprint(o, path=STRAIGHT, lane=None):
    """(heading, its 1 sigma, weight of its standing mode) of a radar object without futures, as it is judged."""
    ex = TH.ThreatModel().explain({"objects": [dict(o, **({"lane": lane} if lane else {}))], "ego": {"speed_mps": 11.0}}, path=path)
    return ex.ohdr[0, 6], ex.ohdr[0, 7], ex.modes[0, 0], ex


class Futures(unittest.TestCase):
    def test_contract_futures_are_taken_and_now_prepended(self):
        msg, _ = FX.frames("cut_in")[5]
        ex = TH.ThreatModel().explain(msg)
        self.assertEqual(ex.t[0], 0.0)
        self.assertEqual(len(ex.t), len(FX.T_GRID) + 1)
        self.assertFalse(ex.ego_own)
        self.assertEqual(ex.src, ["contract"])
        self.assertEqual(ex.mode_names(0), ["cut_in", "keep"])
        self.assertAlmostEqual(ex.modes[0, 2], msg["objects"][0]["x_m"])

    def test_without_futures_the_adapter_stands_in_and_says_so(self):
        msg, _ = FX.frames("cut_in")[5]
        bare = dict(msg, objects=[{k: v for k, v in o.items() if k != "fut"} for o in msg["objects"]])
        bare.pop("fut")
        bare.pop("ego_fut")
        b = Bench(Decide())
        out = b.feed("radar", bare)
        m = out[0][1]
        self.assertEqual(m["futures"]["objects"], "cv")
        self.assertEqual(m["futures"]["ego"], "own")
        h = b.plugin.health()
        self.assertEqual(h["state"], "degraded")
        self.assertIn("constant-velocity adapter", h["reason"])

    def test_a_future_not_as_the_contract_says_is_replaced_and_counted(self):
        msg, _ = FX.frames("cut_in")[5]
        bad = dict(msg, objects=[dict(msg["objects"][0], fut=[{"p": 1.0, "x": [1.0, 2.0]}])])
        ex = TH.ThreatModel().explain(bad)
        self.assertEqual(ex.src[0], "cv")
        self.assertEqual(ex.invalid, 1)

    def test_a_companion_message_carries_the_futures(self):
        msg, _ = FX.frames("lead_braking")[25]
        comp = {"type": "radar_fut", "ts_mono_ns": msg["ts_mono_ns"], "fut": msg["fut"], "ego_fut": msg["ego_fut"],
                "objects": [{"id": o["id"], "fut": o["fut"]} for o in msg["objects"]]}
        bare = dict(msg, objects=[{k: v for k, v in o.items() if k != "fut"} for o in msg["objects"]])
        bare.pop("fut")
        bare.pop("ego_fut")
        b = Bench(Decide())
        self.assertEqual(b.feed("radar_fut", comp), [])          # nothing yet: no radar message
        out = b.feed("radar", bare)                              # the companion of its time is on the bus
        self.assertEqual(out[0][1]["futures"]["objects"], "contract")

    def test_the_degraded_future_follows_the_path_for_traffic_along_it_and_uses_the_radars_stand_hypothesis(self):
        xs = np.linspace(0, 100, 51)
        path = plan(xs, 0.002 * xs ** 2)
        o = {"id": 1, "state": "confirmed", "x_m": 20.0, "y_m": 0.8, "vx_mps": 10.0, "vy_mps": 0.8, "std": [1, 0.3, 1, 0.3],
             "motion": "moving"}
        ego = {"speed_mps": 10.0}
        ex = TH.ThreatModel().explain({"objects": [o], "ego": ego}, path=path)
        n = len(ex.t)
        self.assertEqual(ex.t.tolist(), np.arange(0, 5.01, 0.25).tolist())
        self.assertEqual(ex.names, ["cv_path"])
        x, y = ex.modes[0, 2:2 + n], ex.modes[0, 2 + n:2 + 2 * n]
        self.assertAlmostEqual(y[-1] - 0.002 * x[-1] ** 2, 0.0, delta=0.6)    # keeps its place beside the path
        ex = TH.ThreatModel().explain({"objects": [dict(o, motion="unknown", p_stationary=0.7)], "ego": ego})
        self.assertEqual(ex.names, ["stand", "cv"])
        self.assertAlmostEqual(ex.p[0, 0], 0.7)

    def test_a_parked_car_beside_the_path_is_drawn_where_the_radar_says_and_along_the_road(self):
        """The footprint's centre is the radar's `ctr` (not reconstructed along the line of sight); a standing car's
        heading is the road's axis when near it; a standing mode counts as far as the prediction puts it in the
        car's lane; a mode starting beside the radar's point does not turn the car across the path."""
        o = {"id": 7, "state": "confirmed", "x_m": 9.0, "y_m": -1.8, "ctr": [11.25, -2.7], "len": 4.5, "wid": 1.8,
             "hdg": -0.3, "motion": "stationary", "flags": ["parked"], "std": [0.5, 0.2, 0.1, 0.1]}
        road = plan([0.0, 100.0], [0.0, 100.0 * math.tan(0.05)])         # a road in the direction 0.05 rad
        h, sh, _, ex = footprint(o, road)
        np.testing.assert_allclose(ex.ohdr[0, 9:12], (1.0, 2.25, -0.9))  # the centre: the radar's, from its point
        self.assertAlmostEqual(h, 0.05)                                  # it stands, its heading is not its travel's:
        self.assertAlmostEqual(sh, 0.1)                                  # ... the road's axis, within 0.7 rad of it
        self.assertAlmostEqual(footprint(dict(o, hdg=-3.0), road)[0], -3.0 + (0.05 - (-3.0 + math.pi)))   # either direction
        self.assertEqual(footprint(dict(o, hdg=1.5, hdg_std=0.1))[:2], (1.5, 0.1))    # a bay-parked car: as stated
        # the weight of its standing mode: P(lane ego) + P(lane cross), unless the radar says it is in the path
        self.assertAlmostEqual(footprint(o, lane={"ego": 0.1, "park": 0.85, "cross": 0.05})[2], 0.15)
        self.assertEqual(footprint(dict(o, flags=["in_path"]), lane={"park": 1.0})[2], 1.0)
        self.assertEqual(footprint(o)[2], 1.0)
        # through the native core: the car stands beside a straight path, 1.0 m clear of the car's side
        t = np.arange(0.0, 4.01, 0.25)
        x0, y0 = np.full(len(t), 9.0), np.full(len(t), -1.8)
        jump = y0.copy()
        jump[1:] -= 0.4                                   # a mode 0.4 m beside the radar's point from its first step
        ex = judged(o, [("stopped", 0.6, 0.6, x0, y0, 0.1 + 0 * t, 0.1 + 0 * t, 0 * t),
                        ("move_off", 0.4, 0.4, x0, jump, 0.1 + 0 * t, 0.1 + 0 * t, 0 * t)], t)
        self.assertEqual(ex.src, ["contract"])
        self.assertLess(ex.obj[0, TH.DO_PC], 0.05)

    def test_a_box_heading_the_radar_did_not_resolve_and_a_margin_breach_without_contact_do_not_alarm(self):
        """A standing car beside the path whose box the radar turned a quarter turn with a wide heading spread is
        drawn along the road; an oncoming car the prediction puts in its lane is drawn along it whatever its box
        says; a pair whose mean futures miss (only the margin breached) has no time margin."""
        t = np.arange(0.0, 4.01, 0.25)
        # parked 2.8 m right of the path (centre), its box turned to 1.6 rad with a heading spread of 0.33 rad
        o = {"id": 3, "state": "confirmed", "x_m": 20.0, "y_m": -1.9, "ctr": [21.5, -2.8], "len": 3.2, "wid": 1.8,
             "hdg": 1.6, "ext_std": [0.4, 0.18, 0.33], "motion": "stationary", "std": [1.0, 0.3, 0.1, 0.1]}
        # a heading is resolved at a 1 sigma of 0.15 rad at most, or where none is stated
        self.assertEqual(footprint(dict(o, ext_std=[0.4, 0.18, 0.08]), None)[0], 1.6)
        self.assertEqual(footprint({k: v for k, v in o.items() if k != "ext_std"}, None)[0], 1.6)
        self.assertLess(abs(math.sin(footprint(o)[0])), 1e-9)            # 0.33 rad: not resolved, the road's axis
        stand = [("stopped", 1.0, 1.0, np.full(len(t), 20.0), np.full(len(t), -1.9), 0.3 + 0 * t, 0.3 + 0 * t, 0 * t)]
        ex = judged(o, stand, t)
        self.assertLess(ex.obj[0, TH.DO_PC], 0.05)
        self.assertEqual(int(ex.obj[0, TH.DO_LE]), 0)
        ex = judged(dict(o, ext_std=[0.4, 0.18, 0.05]), stand, t)           # resolved: a bay-parked car, as stated
        self.assertGreater(ex.obj[0, TH.DO_PC], 0.2)                     # (its length across: 0.3 m off the side)
        # an oncoming truck in its lane (centre 3.6 m left), its 9.6 m box turned 0.35 rad toward the car
        tr = {"id": 4, "state": "confirmed", "x_m": 30.0, "y_m": 2.7, "ctr": [34.8, 3.6], "len": 9.6, "wid": 2.5,
              "hdg": 0.35, "ext_std": [1.9, 0.18, 0.06], "motion": "moving", "class": {"label": "truck"},
              "std": [2.0, 0.4, 2.0, 0.5]}
        onc = [("onc_keep", 1.0, 1.0, 30.0 - 12.0 * t, np.full(len(t), 2.7), 0.5 + 0.3 * t, 0.4 + 0 * t, 0 * t)]
        ex = judged(tr, onc, t)
        self.assertEqual(ex.ohdr[0, 6], 0.35)
        self.assertGreater(ex.obj[0, TH.DO_PC], 0.3)                     # its box's corner in the car's lane
        ex = judged(tr, onc, t, lane={"onc": 0.94, "ego": 0.02})         # P(a lane along the road) 0.96: along it
        self.assertLess(abs(math.sin(ex.ohdr[0, 6])), 1e-9)
        self.assertLess(ex.obj[0, TH.DO_PC], 0.05)
        self.assertEqual(judged(tr, onc, t, lane={"onc": 0.4, "park": 0.6}).ohdr[0, 6], 0.35)      # below one half: as stated
        # a car beside the path, 0.15 m clear of the car's side on the mean futures, a wide spread: the margin is
        # breached, there is no collision to time
        b = {"id": 5, "state": "confirmed", "x_m": 15.0, "y_m": -1.9, "ctr": [17.25, -2.8], "len": 4.5, "wid": 1.8,
             "hdg": 0.0, "ext_std": [0.4, 0.18, 0.05], "motion": "stationary", "std": [1.0, 0.8, 0.1, 0.1]}
        ms = [("stopped", 1.0, 1.0, np.full(len(t), 15.0), np.full(len(t), -1.9 + 0.95), 0.8 + 0 * t,
               0.8 + 0 * t, 0 * t)]
        p = judged(b, ms, t).pairs(0)[0, 0]
        self.assertTrue(math.isnan(p[TH.DP_TTC]) and p[TH.DP_COURSE] > 0.5)    # mean miss, the margin breached
        self.assertTrue(math.isnan(p[TH.DP_TM]))
        self.assertLessEqual(int(p[TH.DP_LE]), 1)

    def test_class_priors_and_labels_as_the_native_core_reads_them(self):
        """The footprint of a class where the radar gives no extent, and the threshold set of its kind: the native
        core's table against futures.CLASS_SIZE; labels as the class vote normalises them."""
        from chris_integrator.core import classvote
        o = {"id": 1, "state": "confirmed", "x_m": 30.0, "y_m": 0.0, "std": [1, 1, 1, 1], "motion": "stationary"}
        kinds = {"car": "vehicle", "van": "vehicle", "truck": "vehicle", "bus": "vehicle", "tram": "vehicle",
                 "human": "vru", "bike": "vru", "motorbike": "vru"}
        labels = list(FU.CLASS_SIZE) + list(classvote.ALIASES) + ["  Pedestrian ", "TRUCK", "animal", "", 7]
        for lab in labels:
            norm = classvote.norm_label(lab)
            ex = TH.ThreatModel().explain({"objects": [dict(o, **{"class": {"label": lab}})], "ego": {"speed_mps": 10.0}})
            self.assertEqual(ex.ohdr[0, 2:6].tolist(), list(FU.CLASS_SIZE.get(norm, FU.CLASS_SIZE[None])), lab)
            self.assertEqual(ex.records[0]["kind"], kinds.get(norm, "unknown"), lab)
            self.assertEqual(int(ex.ohdr[0, 8]), FU.KIND_SET[kinds.get(norm, "unknown")], lab)
            self.assertEqual(ex.records[0]["size_src"], "class")


def rec(oid, le, lx=None, tn=0.5, tm=0.5, kind="vehicle", ab=3.0, al=2.0, ar=2.0, mass=None):
    return {"id": oid, "level": le, "level_exit": le if lx is None else lx, "kind": kind, "tn": tn, "tm_s": tm,
            "p_coll": 0.8, "mass": mass or [0.8] * 4, "a_req": {"brake": ab, "left": al, "right": ar},
            "why": {"measure": "time_margin", "mode": "stand", "ego_mode": "path"}, "fut": "contract"}


def robj(oid, **kw):
    return dict({"id": oid, "state": "confirmed", "exist": 0.99, "seen": {"cam": 0.05}, "motion": "moving",
                 "std": [1, 1, 0.5, 0.5], "flags": []}, **kw)


class AlertLogic(unittest.TestCase):
    CTX = {"v": 15.0, "a_long": 0.0, "a_lat": 0.0, "sensor_age_s": 0.05}

    def feed(self, a, seq, dt=0.05, objs=None, ctx=None):
        out = []
        for k, recs in enumerate(seq):
            o, ev = a.step(k * dt, recs, objs or {r["id"]: robj(r["id"]) for r in recs}, ctx or self.CTX)
            out.append((o, ev))
        return out

    def test_entry_hold_minimum_hold_and_exit_hysteresis(self):
        a = AL.Alert("obstacle_ahead")
        out = self.feed(a, [[rec(1, 3)]] * 4 + [[rec(1, 3, lx=3)]] * 10 + [[rec(1, 2, lx=2)]] * 40)
        st = [o["stage"] for o, _ in out]
        self.assertEqual(st[0], 0)                      # warning needs 0.1 s of holding
        self.assertEqual(st[2], 3)
        # below the stage: 0.8 s of exit hold after the stage was shown 1.5 s
        t_down = next(k for k, s in enumerate(st) if k > 2 and s < 3) * 0.05
        self.assertGreaterEqual(t_down, 1.5)

    def test_imminent_latches_until_the_threat_is_gone(self):
        a = AL.Alert("obstacle_ahead")
        seq = [[rec(1, 4)]] * 3 + [[rec(1, 2, lx=2)]] * 80 + [[rec(1, 1, lx=1)]] * 40
        st = [o["stage"] for o, _ in self.feed(a, seq)]
        self.assertTrue(all(s == 4 for s in st[1:80]))  # latched while the exit level stays above information
        self.assertLess(st[-1], 4)

    def test_a_stable_primary(self):
        a = AL.Alert("obstacle_ahead")
        seq = []
        for k in range(30):                              # two objects at the same level trade places every frame
            m1, m2 = (0.6, 0.62) if k % 2 else (0.62, 0.6)
            seq.append([rec(1, 3, mass=[m1] * 4), rec(2, 3, mass=[m2] * 4)])
        prim = [o["object"] for o, _ in self.feed(a, seq) if o["stage"]]
        self.assertEqual(len(set(prim[2:])), 1)          # no hopping
        out = self.feed(AL.Alert("obstacle_ahead"), [[rec(1, 2), rec(2, 4)]] * 5)
        self.assertEqual(out[-1][0]["object"], 2)        # a higher level takes over at once

    def test_suppression_rules(self):
        cases = [({"a_long": -4.0}, "driver_braking"), ({"a_lat": 2.5}, "driver_steering_away"),
                 ({"v": 1.0}, "low_speed"), ({"sensor_age_s": 2.0}, "sensor_degraded")]
        for ctx, why in cases:
            a = AL.Alert("obstacle_ahead")
            out = self.feed(a, [[rec(1, 3, ab=3.0, al=2.0)]] * 5, ctx=dict(self.CTX, **ctx))
            o = out[-1][0]
            self.assertEqual(o["suppressed"], why)
            self.assertGreater(o["stage"], 0)            # the stage is kept, nothing is said
            self.assertFalse(o["active"])

    def test_ineligible_objects_do_not_raise(self):
        for ob in (robj(1, exist=0.3), robj(1, state="memory"), robj(1, state="tentative"), robj(1, seen={"cam": 3.0})):
            out = self.feed(AL.Alert("obstacle_ahead"), [[rec(1, 4)]] * 5, objs={1: ob})
            self.assertEqual(out[-1][0]["stage"], 0)

    def test_modes(self):
        on = AL.Alert("obstacle_ahead", {"mode": "on"})
        self.assertTrue(self.feed(on, [[rec(1, 3)]] * 5)[-1][0]["active"])
        sh = AL.Alert("obstacle_ahead")
        o, _ = self.feed(sh, [[rec(1, 3)]] * 5)[-1]
        self.assertEqual(o["mode"], "shadow")
        self.assertFalse(o["active"])
        off = AL.Alert("obstacle_ahead", {"mode": "off"})
        out = self.feed(off, [[rec(1, 4)]] * 5)
        self.assertEqual(out[-1][0]["stage"], 0)
        self.assertEqual(sum(len(ev) for _, ev in out), 0)       # nothing is evaluated or logged

    def test_headway(self):
        a = AL.Alert("headway")
        r = dict(rec(1, 0), thw_s=0.7)
        out = self.feed(a, [[r]] * 60, ctx=dict(self.CTX, v=20.0))
        self.assertEqual(out[-1][0]["stage"], 2)          # caution below 0.9 s held 2 s
        self.assertEqual(out[30][0]["stage"], 0)


class ViewFromDecide(unittest.TestCase):
    def test_flags_and_alert_come_from_decide_and_the_radar_flags_stay_the_fallback(self):
        rd = RadarDisplay()
        ts = 10 * NS
        objs = [{"id": 1, "state": "confirmed", "x_m": 30.0, "y_m": 0.0, "vx_mps": 0.0, "vy_mps": 0.0, "v_rel_mps": -10.0,
                 "flags": ["in_path", "on_path"], "motion": "stationary", "class": {"label": "car"}, "std": [1, 0.3, 0.5, 0.3],
                 "seen": {"cam": 0.05}, "age_s": 3.0, "ttc_s": 3.0},
                {"id": 2, "state": "confirmed", "x_m": 40.0, "y_m": -3.0, "vx_mps": 0.0, "vy_mps": 0.0, "v_rel_mps": -10.0,
                 "flags": ["threat"], "motion": "stationary", "class": {"label": "car"}, "std": [1, 0.3, 0.5, 0.3],
                 "seen": {"cam": 0.05}, "age_s": 3.0, "ttc_s": 4.0}]
        rd.update({"ts_mono_ns": ts, "objects": objs, "ego": {"speed_mps": 10.0}})
        _, alert = rd.objects(ts)
        self.assertEqual(alert["level"], "threat")                 # no decide: the radar's flags
        self.assertEqual(alert["id"], "ru:2")
        dec = {"ts_mono_ns": ts, "objects": [{"id": 1, "level": 3, "shown": 3, "ttc_s": 2.4}, {"id": 2, "level": 0, "shown": 0}],
               "alerts": [{"id": "obstacle_ahead", "mode": "shadow", "stage": 3, "stage_name": "warning", "object": 1,
                           "suppressed": None, "reason": {"tn": 0.4, "p_coll": 0.9, "tm_s": 0.3}}]}
        rd.on_decide(dec)
        out, alert = rd.objects(ts)
        self.assertEqual((alert["level"], alert["id"], alert["stage"], alert["mode"]), ("threat", "ru:1", 3, "shadow"))
        self.assertEqual(alert["ttc_s"], 2.4)
        by = {o["id"]: o for o in out}
        self.assertTrue(by["ru:1"]["attrs"]["threat"])
        self.assertFalse(by["ru:2"]["attrs"]["threat"])            # the measure says it is no threat
        ar = {e["id"]: e for e in rd.around(ts)}
        self.assertIn("t", ar["ru:1"]["fl"])
        self.assertNotIn("t", ar["ru:2"].get("fl", ""))
        dec["alerts"][0]["suppressed"] = "driver_braking"
        self.assertIsNone(rd.objects(ts)[1])                       # not said while suppressed


def sim_run(name, seed=1):
    from chris_integrator.mixer import MessageClock
    from chris_integrator.sim import default_sensors
    from chris_integrator.sim import roaduser_sim as rs
    from chris_integrator.stack import build
    w = rs.scenario(name, seed=seed, night=False)
    det = rs.DetectorSim(seed=1000 + seed, night=False)
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
    sensors.append(det)
    st = build(cfg={"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}, "radar": {"rate_hz": 1000.0}},
               mapper=None, clock=MessageClock())
    t0 = int(round(w.t0 * NS))
    shown, last = {"threat": 0.0, "any": 0.0}, [None]
    first = {}

    def on(e):
        ts = e.msg["ts_mono_ns"]
        a = next(x for x in e.msg["alerts"] if x["id"] == "obstacle_ahead")
        if last[0] is not None and a["stage"] >= 2 and not a["suppressed"]:
            shown["any"] += (ts - last[0]) / NS
            if a["stage"] >= 3:
                shown["threat"] += (ts - last[0]) / NS
            first.setdefault(a["stage"], (ts - t0) / NS)
        last[0] = ts
    st.mixer.add_output("t", on, streams=("decide",))
    sent = False
    for _, m in w.messages(sensors):
        if not sent:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            sent = True
        st.deliver(m)
    return w, shown, first


class Simulated(unittest.TestCase):
    """The simulator's scenarios through the whole stack, the degraded futures (no prediction layer)."""

    def test_a_standing_car_ahead_of_a_driver_who_does_not_react(self):
        w, shown, first = sim_run("stopped_cruise_14")
        self.assertIn(3, first)
        self.assertGreaterEqual(w.spec["collision_t"] - first[3], 2.0)       # warning 2 s before the impact
        w, shown, first = sim_run("stopped_cruise_25")
        self.assertGreaterEqual(w.spec["collision_t"] - min(first.get(3, 99), first.get(4, 99)), 1.0)

    def test_traffic_beside_and_against_the_car_does_not_warn(self):
        for name in ("adjacent", "oncoming", "junction"):
            _, shown, _ = sim_run(name)
            self.assertEqual(shown["threat"], 0.0, name)
            self.assertLess(shown["any"], 2.0, name)


if __name__ == "__main__":
    unittest.main()
