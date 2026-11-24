"""The stack's pace (plugins/common.py `Pace`): which radar messages `predict` and `decide` look at.

The default (`full`) is every message; `fixed` thins to `idle_hz`; `adaptive` thins only while nothing could
matter. What is frozen here: the rule itself on made-up streams, the bound it promises (two looks are never further
apart than the low rate's period or one radar period), that the two plug-ins leave out the same messages, and the
warnings' times of the simulator's collision scenarios under each mode."""
import random
import unittest

from chris_integrator.plugins.common import Pace

NS = 1_000_000_000


def radar(t, objects=(), ground=0):
    return {"type": "radar", "ts_mono_ns": int(round(t * NS)), "objects": list(objects), "ego": {"ground_frame": ground}}


def obj(i, **kw):
    return dict({"id": i, "state": "confirmed", "flags": [], "ttc_s": None, "p_threat": 0.0}, **kw)


def looks(p, msgs):
    return [m["ts_mono_ns"] for m in msgs if p.due(m)]


class Rule(unittest.TestCase):
    def test_full_is_every_message_and_the_default_is_adaptive(self):
        # the default since 2026-11-24 (was `full`): adaptive, with the careful signs (a collision probability from
        # 0.01, a hold of 0.5 s). docs/TOPICS.md has what was measured: no warning's time in the straight collision
        # scenarios, one of 63 by one frame in the bends, 2 of 95 false episodes fewer; the looser signs (levels only,
        # hold 0.3 s) save more and lost one warning of 63 in the bends
        self.assertEqual((Pace.DEFAULT_CFG["mode"], Pace.DEFAULT_CFG["hot_p"], Pace.DEFAULT_CFG["hold_s"]), ("adaptive", 0.01, 0.5))
        p = Pace({"mode": "full"})
        self.assertEqual(p.cfg["mode"], "full")
        msgs = [radar(0.05 * k) for k in range(40)]
        self.assertEqual(len(looks(p, msgs)), 40)
        self.assertEqual(Pace({"mode": "nonsense"}).cfg["mode"], "full")              # not understood: nothing left out

    def test_one_answer_per_message(self):
        p = Pace({"mode": "fixed"})
        for k in range(20):
            m = radar(0.05 * k)
            self.assertEqual(p.due(m), p.due(m))
        self.assertEqual(p.stats["messages"], 20)

    def test_fixed_is_every_second_message_at_twenty_a_second(self):
        p = Pace({"mode": "fixed"})
        got = looks(p, [radar(0.05 * k) for k in range(41)])
        self.assertEqual(got, [int(round(0.05 * k * NS)) for k in range(0, 41, 2)])

    def test_a_radar_at_the_low_rate_or_below_is_not_thinned(self):
        for period in (0.085, 0.1, 0.15):
            p = Pace({"mode": "fixed"})
            self.assertEqual(len(looks(p, [radar(period * k) for k in range(30)])), 30, period)

    def test_a_gap_in_the_radar_is_looked_at_at_once(self):
        p = Pace({"mode": "fixed"})
        ts = [0.0, 0.05, 0.10, 0.15, 0.25, 0.30, 0.35, 0.40]         # one message lost after 0.15
        got = looks(p, [radar(t) for t in ts])
        # 0.15 was left out for the message expected at 0.20; the one that came instead is looked at
        self.assertEqual(got, [int(round(t * NS)) for t in (0.0, 0.10, 0.25, 0.35)])

    def test_the_clock_going_back_starts_anew(self):
        p = Pace({"mode": "fixed"})
        looks(p, [radar(10.0 + 0.05 * k) for k in range(10)])
        self.assertTrue(p.due(radar(1.0)))

    def test_adaptive_quiet_is_the_low_rate(self):
        p = Pace({"mode": "adaptive"})
        got = looks(p, [radar(0.05 * k, [obj(1)]) for k in range(41)])
        self.assertEqual(len(got), 21)
        self.assertEqual(p.stats["hot"], 0)

    def test_adaptive_keeps_the_full_rate_while_the_decision_layer_says_so_and_a_hold_on(self):
        p = Pace({"mode": "adaptive", "hold_s": 0.5})
        seen = []
        for k in range(60):
            m = radar(0.05 * k, [obj(1)])
            if p.due(m):
                seen.append(k)
                p.report(m["ts_mono_ns"], 10 <= k <= 20)             # hot from 0.5 s to 1.0 s
        # every message from the first look that found it to the hold's end (1.0 s + 0.5 s), every second otherwise
        first = min(k for k in seen if k >= 10)
        self.assertLessEqual(first, 11)
        self.assertEqual([k for k in seen if first <= k <= 30], list(range(first, 31)))
        self.assertEqual(len([k for k in seen if k > 31]), 14)

    def test_adaptive_cues_of_the_radar_message_itself(self):
        base = [radar(0.05 * k, [obj(1)]) for k in range(7)]         # looked at: 0, 0.10, 0.20, 0.30
        for cue in ({"flags": ["in_path"]}, {"flags": ["on_path", "threat"]}, {"ttc_s": 6.0}, {"p_threat": 0.05}):
            p = Pace({"mode": "adaptive"})
            self.assertEqual(looks(p, base)[-1], base[6]["ts_mono_ns"])
            self.assertTrue(p.due(radar(0.35, [obj(1, **cue)])), cue)   # 0.05 s after a look: left out without the cue
            self.assertEqual(p.stats["cue"], 1)
            # ... and the hold keeps the full rate after it
            self.assertTrue(all(p.due(radar(0.35 + 0.05 * k, [obj(1)])) for k in range(1, 11)), cue)
            self.assertFalse(p.due(radar(0.90, [obj(1)])), cue)        # the hold (0.5 s) is over: the low rate again
        # no cue: nothing, a far time to reach, a tentative object
        for quiet in ({}, {"ttc_s": 9.0}, {"state": "tentative", "flags": ["in_path"]}):
            p = Pace({"mode": "adaptive"})
            looks(p, base)
            self.assertFalse(p.due(radar(0.35, [obj(1, **quiet)])), quiet)

    def test_adaptive_a_new_object_is_looked_at_once(self):
        base = [radar(0.05 * k, [obj(1)]) for k in range(7)]
        p = Pace({"mode": "adaptive"})
        looks(p, base)
        self.assertTrue(p.due(radar(0.35, [obj(1), obj(2)])))
        self.assertEqual(p.stats["new"], 1)
        self.assertFalse(p.due(radar(0.40, [obj(1), obj(2)])))      # no hold: the low rate again
        self.assertTrue(p.due(radar(0.45, [obj(1), obj(2)])))
        p = Pace({"mode": "adaptive"})
        looks(p, base)
        self.assertTrue(p.due(radar(0.35, [obj(1)], ground=1)))      # a new ground frame

    def test_the_bound_two_looks_are_never_further_apart_than_the_low_rate_but_for_a_late_message(self):
        """A message is left out only when the next one, expected one interval on, still comes within the low rate's
        period (+ 0.01 s) of the last look: two looks are at most that far apart, plus however much later than
        expected the message after a left-out one came (a lost radar message: its period)."""
        rng = random.Random(7)
        for mode in ("fixed", "adaptive"):
            for trial in range(200):
                hz = rng.choice((5.0, 10.0))
                p = Pace({"mode": mode, "idle_hz": hz, "hold_s": rng.choice((0.2, 0.5))})
                period = rng.choice((0.04, 0.05, 0.066, 0.085, 0.1))
                t, last, hist, skipped = 100.0, None, [], False
                for k in range(300):
                    t += period * (1.0 + rng.uniform(-0.1, 0.1)) * (2 if rng.random() < 0.08 else 1)
                    m = radar(t, [obj(1, flags=["in_path"] if rng.random() < 0.02 else [])])
                    hist.append(m["ts_mono_ns"])
                    if p.due(m):
                        if last is not None and skipped:
                            d_next, d_prev = hist[-1] - hist[-2], hist[-2] - hist[-3]
                            self.assertLessEqual(hist[-1] - last, (1.0 / hz + 0.01) * NS + max(0, d_next - d_prev),
                                                 (mode, trial, k))
                        last, skipped = hist[-1], False
                        p.report(m["ts_mono_ns"], rng.random() < 0.05)
                    else:
                        skipped = True
                self.assertGreater(p.stats["messages"] - p.stats["looked"], 0 if period < 0.06 else -1, (mode, period))


def stack_run(name, pace, seed=1, cfg=None, hz=20.0):
    """A simulator scenario through the stack with the road model at `hz` frames a second (the simulator's own 13
    are the older road model's; the jitter of its frames makes 20 a mean of 17.5, so the low rate is set to 8 here:
    every second message, as 10 are of a phone's even 20)."""
    from chris_integrator.mixer import MessageClock
    from chris_integrator.sim import default_sensors
    from chris_integrator.sim import roaduser_sim as rs
    from chris_integrator.stack import build
    w = rs.scenario(name, seed=seed, night=False)
    det = rs.DetectorSim(seed=1000 + seed, night=False)
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True, "hz": hz},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
    sensors.append(det)
    c = dict({"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}, "radar": {"rate_hz": 1000.0},
              "decide": {"rate_hz": 1000.0}}, **(cfg or {}))
    if pace is not None:
        c["pace"] = pace
    st = build(cfg=c, mapper=None, clock=MessageClock())
    seen = {"radar": [], "radar_fut": [], "decide": []}
    said = []

    def on(e):
        seen[e.stream].append(e.ts)
        if e.stream == "decide":
            a = next(x for x in e.msg["alerts"] if x["id"] == "obstacle_ahead")
            if a["stage"] >= 2 and not a["suppressed"]:
                said.append(e.msg["ts_mono_ns"])
    st.mixer.add_output("t", on, streams=tuple(seen))
    sent = False
    for _, m in w.messages(sensors):
        if not sent:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            sent = True
        st.deliver(m)
    first = round(w.spec["collision_t"] - (said[0] / NS - w.t0), 2) if said and w.spec.get("collision_t") else None
    return st, seen, first


class InTheStack(unittest.TestCase):
    def test_the_stack_has_the_service_and_its_configuration(self):
        from chris_integrator.stack import build
        st = build(mapper=None)
        self.assertEqual(st.mixer.services["pace"].cfg["mode"], Pace.DEFAULT_CFG["mode"])
        st = build(cfg={"pace": {"mode": "adaptive", "idle_hz": 5.0}}, mapper=None)
        self.assertEqual((st.mixer.services["pace"].cfg["mode"], st.mixer.services["pace"].cfg["idle_hz"]), ("adaptive", 5.0))

    def test_full_decides_and_predicts_every_radar_message(self):
        _, seen, first = stack_run("stopped_cruise_14", {"mode": "full"})
        self.assertEqual(seen["decide"], seen["radar"])
        self.assertEqual(seen["radar_fut"], seen["radar"])
        self.assertEqual(first, 2.58)                                # first said, s before the collision

    def test_the_simulators_own_road_model_rate_is_not_thinned(self):
        # 13 frames a second (the older road model): at or below the low rate's reach, every message is looked at
        for mode in ("fixed", "adaptive"):
            _, seen, _ = stack_run("stopped_cruise_25", {"mode": mode}, hz=13.0)
            self.assertEqual(seen["decide"], seen["radar"], mode)

    def test_the_two_layers_leave_out_the_same_messages(self):
        for mode in ("fixed", "adaptive"):
            st, seen, _ = stack_run("stopped_cruise_14", {"mode": mode, "idle_hz": 8.0})
            self.assertEqual(seen["decide"], seen["radar_fut"], mode)
            self.assertLess(len(seen["decide"]), 0.7 * len(seen["radar"]), mode)
            self.assertEqual(set(seen["decide"]) - set(seen["radar"]), set(), mode)
            h = st.mixer.services["pace"].health()
            self.assertEqual(h["looked"], len(seen["decide"]), mode)
            # no decision was made without its futures (the degraded adapter standing in for a left-out prediction)
            self.assertEqual(st.mixer.plugin("decide").stats["cv_objects"], 0, mode)

    def test_adaptive_says_the_collision_scenarios_warnings_when_full_does(self):
        # every message: 2.58 s and 1.73 s before the collision; adaptive: the same frames
        self.assertEqual(stack_run("stopped_cruise_14", {"mode": "adaptive", "idle_hz": 8.0})[2], 2.58)
        self.assertEqual(stack_run("stopped_cruise_25", {"mode": "full"})[2], 1.73)
        self.assertEqual(stack_run("stopped_cruise_25", {"mode": "adaptive", "idle_hz": 8.0})[2], 1.73)

    def test_a_fixed_low_rate_pays_in_the_warning_time(self):
        # the price of halving whatever happens, frozen: the warnings come one left-out message later
        # (2.58 s -> 2.51 s, 1.73 s -> 1.67 s before the collision)
        self.assertEqual(stack_run("stopped_cruise_14", {"mode": "fixed", "idle_hz": 8.0})[2], 2.51)
        self.assertEqual(stack_run("stopped_cruise_25", {"mode": "fixed", "idle_hz": 8.0})[2], 1.67)


if __name__ == "__main__":
    unittest.main()
