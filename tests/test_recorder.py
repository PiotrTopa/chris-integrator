"""Drive recorder (chris_integrator/trackers/recorder.py).

    python3 -m unittest api.tests.test_recorder -v     (from the repository root)
"""
import gzip
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chris_integrator.trackers.lanes import RoadGeometryTracker      # noqa: E402
from chris_integrator.io.recorder import Recorder, load_config, slim_road      # noqa: E402

X = [192.0 * (i / 32.0) ** 2 for i in range(33)]


def road(seq):
    line = {"x": X, "y": [1.23456789 + 0.001 * i for i in range(33)], "z": [0.0] * 33, "y_std": [0.3] * 33}
    return {"type": "road", "seq": seq, "ts_mono_ns": seq * 50_000_000, "ts_real_ns": 1, "model_ms": 55.1, "v_ego": 12.345678,
            "pose": {"yaw_rate": 0.0123456, "vy": 0.1}, "lane_lines": [dict(line, prob=0.912345) for _ in range(4)],
            "road_edges": [dict(line) for _ in range(2)],
            "leads": [{"prob": 0.9, "x": [30.123456] * 6, "y": [0.1] * 6, "v": [10.0] * 6, "x_std": [1.0] * 6, "v_std": [1.0] * 6}] * 3,
            "plan": {"x": X, "y": [0.0] * 33, "z": [0.0] * 33, "vx": [1.0] * 33, "t": [0.0] * 33},
            "meta": {"big": list(range(200))}, "desire_state": [0.0] * 8}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.preview = os.path.join(self.tmp.name, "preview.jpg")
        with open(self.preview, "wb") as f:
            f.write(b"\xff\xd8jpeg\xff\xd9")
        self.clock = Clock()

    def rec(self, **cfg):
        cfg.setdefault("dir", os.path.join(self.tmp.name, "rec"))
        return Recorder(cfg, self.preview, clock=self.clock, wall=lambda: 1794860000.0 + self.clock.t)

    def lines(self, r):
        r.close()
        with gzip.open(os.path.join(r.dir, "messages.jsonl.gz"), "rt") as f:
            return [json.loads(ln) for ln in f]

    def test_records_topics_slims_road_and_copies_frames(self):
        r = self.rec()
        for k in range(40):                                   # 2 s at 20 Hz
            self.clock.t += 0.05
            r.on_message("camera", json.dumps({"type": "camera", "seq": k + 1}))
            r.on_message("road", json.dumps(road(k + 1)))
            r.on_message("lanes", "{\"type\":\"lanes\"}")      # derived: not written
            if k % 7 == 0:
                r.on_message("lights", json.dumps({"type": "lights", "seq": k + 1, "lights": []}))
        out = self.lines(r)
        kinds = [m["type"] for m in out]
        self.assertEqual(kinds.count("road"), 40)
        self.assertEqual(kinds.count("lights"), 6)
        self.assertEqual(kinds.count("camera"), 2)            # one a second
        self.assertNotIn("lanes", kinds)
        rd = next(m for m in out if m["type"] == "road")
        self.assertNotIn("meta", rd)
        self.assertEqual(len(rd["leads"]), 1)
        self.assertEqual(rd["lane_lines"][0]["y"][0], 1.235)
        self.assertLess(len(json.dumps(rd)), 0.6 * len(json.dumps(road(1))))
        self.assertEqual(sorted(os.listdir(os.path.join(r.dir, "frames"))), ["1.jpg", "21.jpg"])
        self.assertEqual(r.status()["frames"], 2)

    def test_recorded_road_replays_through_the_tracker(self):
        trk = RoadGeometryTracker()
        o = None
        for k in range(30):
            o = trk.on_road(slim_road(road(k + 1)))
        self.assertTrue(o["lines"][1]["present"] and o["edges"][0]["present"])

    def test_rate_caps(self):
        r = self.rec(road_hz=5.0, ego_hz=2.0, frame_every_s=0)
        for k in range(40):
            self.clock.t += 0.05
            r.on_message("road", json.dumps(road(k + 1)))
            r.on_message("ego", json.dumps({"type": "ego", "seq": k}))
        kinds = [m["type"] for m in self.lines(r)]
        self.assertEqual((kinds.count("road"), kinds.count("ego")), (10, 4))
        self.assertEqual(os.listdir(os.path.join(r.dir, "frames")), [])

    def _feed_road(self, r, stamps, dt_clock, envelope):
        from chris_integrator.mixer.bus import Envelope
        for k, ts in enumerate(stamps):
            self.clock.t += dt_clock
            m = dict(road(k + 1), ts_mono_ns=ts)
            if envelope:
                r.on_envelope(Envelope("road", m, self.clock.t, "roadsense", k))
            else:
                r.on_message("road", json.dumps(m))
        return [m["seq"] for m in self.lines(r) if m["type"] == "road"]

    def test_default_caps_road_on_the_clock_only(self):
        """No `stamp_hz`: every road message whose arrival keeps the clock cap is written, whatever its stamp
        (stamps 10 ms apart, arrivals 50 ms apart: all 40), and one that arrives too soon is not."""
        for envelope in (False, True):
            stamps = [10 ** 12 + k * 10_000_000 for k in range(40)]
            d = os.path.join(self.tmp.name, "d%d" % envelope)
            self.assertEqual(self._feed_road(self.rec(frame_every_s=0, dir=d + "a"), stamps, 0.05, envelope),
                             list(range(1, 41)))
            self.assertEqual(self._feed_road(self.rec(frame_every_s=0, dir=d + "b"),
                                             stamps[:10], 0.03, envelope), [1, 3, 5, 7, 9])

    def test_stamp_hz_thins_road_by_its_own_stamps(self):
        """`stamp_hz: {"road": 10}`: a message is written when its ts_mono_ns is >= 95 ms after the last written
        one's (or before it: a new clock), whatever the arrival times (here all at one clock time); messages
        without a stamp fall back to the clock cap; other topics keep their clock caps."""
        for envelope in (False, True):
            self.clock.t = 1000.0 + (100.0 if envelope else 0.0)
            base = 10 ** 12
            stamps = [base + k * 50_000_000 for k in range(20)]                  # 20 Hz: every second frame
            stamps += [base + d for d in (994_000_000, 995_000_000, 1089_000_000, 1090_000_000)]     # 94 ms: no
            stamps += [base, base + 20_000_000, base + 96_000_000]                # back in time: a new clock
            got = self._feed_road(self.rec(frame_every_s=0, stamp_hz={"road": 10},
                                           dir=os.path.join(self.tmp.name, "s%d" % envelope)), stamps, 0.0, envelope)
            self.assertEqual(got, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 22, 24, 25, 27])
        r = self.rec(frame_every_s=0, stamp_hz={"road": 10}, ego_hz=2.0, dir=os.path.join(self.tmp.name, "x"))
        for k in range(40):
            self.clock.t += 0.05
            m = road(k + 1)
            del m["ts_mono_ns"]
            r.on_message("road", json.dumps(m))
            r.on_message("ego", json.dumps({"type": "ego", "seq": k}))
        kinds = [m["type"] for m in self.lines(r)]
        self.assertEqual((kinds.count("road"), kinds.count("ego")), (40, 4))

    def test_old_sessions_are_pruned_and_the_current_one_kept(self):
        base = os.path.join(self.tmp.name, "rec")
        for name in ("20261101-080000", "20261102-080000"):
            os.makedirs(os.path.join(base, name))
            with open(os.path.join(base, name, "messages.jsonl.gz"), "wb") as f:
                f.write(b"x" * 600_000)
        r = self.rec(max_mb=1)
        self.clock.t += 1
        r.on_message("env", "{\"type\":\"env\"}")
        self.assertEqual(sorted(os.listdir(base)), ["20261102-080000", os.path.basename(r.dir)])
        r.close()

    def test_a_failing_disk_stops_the_recording_not_the_caller(self):
        blocker = os.path.join(self.tmp.name, "file")
        with open(blocker, "w") as f:
            f.write("x")
        r = self.rec(dir=os.path.join(blocker, "rec"))        # cannot be created
        r.on_message("env", "{\"type\":\"env\"}")
        self.assertTrue(r.failed)
        r.on_message("env", "{\"type\":\"env\"}")             # and stays quiet

    def test_config_file_switches_it_on(self):
        self.assertIsNone(load_config(self.tmp.name))
        with open(os.path.join(self.tmp.name, "recorder.json"), "w") as f:
            json.dump({"dir": "/x", "max_mb": 10, "nonsense": 1}, f)
        self.assertEqual(load_config(self.tmp.name), {"dir": "/x", "max_mb": 10})
        with open(os.path.join(self.tmp.name, "recorder.json"), "w") as f:
            json.dump({"dir": "/x", "stamp_hz": {"road": 10}}, f)
        self.assertEqual(load_config(self.tmp.name), {"dir": "/x", "stamp_hz": {"road": 10}})
        with open(os.path.join(self.tmp.name, "recorder.json"), "w") as f:
            json.dump({"max_mb": 10}, f)
        self.assertIsNone(load_config(self.tmp.name))

    def test_vehicle_objects_and_view_are_recorded(self):
        r = self.rec()
        for k in range(20):                                   # 2 s at 10 Hz
            self.clock.t += 0.1
            r.on_message("vehicle", json.dumps({"type": "vehicle", "seq": k}))
            r.on_message("objects", json.dumps({"type": "objects", "seq": k, "objects": []}))
            r.on_message("view", json.dumps({"type": "view", "seq": k}))
        n = {}
        for m in self.lines(r):
            n[m["type"]] = n.get(m["type"], 0) + 1
        self.assertEqual(n["vehicle"], 20)                    # what the host said about the car: every message
        self.assertIn(n["objects"], (19, 20))                 # at most objects_hz (10)
        self.assertIn(n["view"], (4, 5))                  # at most view_hz (2)

    def _feed_decisions(self, r, envelope):
        """4 s: `radar` and `radar_fut` 12.5 a second, `decide` 10 a second with two alert transitions between the
        cap's slots, `governor` twice a second."""
        from chris_integrator.mixer.bus import Envelope
        seq = 0

        def put(mtype, m):
            nonlocal seq
            seq += 1
            if envelope:
                r.on_envelope(Envelope(mtype, m, self.clock.t, "stack", seq))
            else:
                r.on_message(mtype, json.dumps(m, separators=(",", ":")))
        for k in range(100):
            self.clock.t += 0.04
            if k % 2 == 0:
                obj = {"id": 7, "state": "confirmed", "x_m": 30.0 - 0.1 * k, "y_m": 0.2, "vx_mps": 9.0}
                put("radar", {"type": "radar", "seq": k, "objects": [obj]})
                put("radar_fut", {"type": "radar_fut", "seq": k, "objects": [dict(obj, fut=[{"x": [1.0] * 22}])]})
            if k % 5 == 0:
                ev = [{"alert": "obstacle_ahead", "event": "raise", "stage": 2, "object": 7}] if k in (35, 40) else []
                put("decide", {"type": "decide", "seq": k, "alerts": [{"id": "obstacle_ahead", "stage": 2 if k >= 35 else 0}],
                               "events": ev})
            if k % 25 == 0:
                put("governor", {"type": "governor", "level": k // 25, "shed": [], "headroom_k": 9.5})
        out = {}
        for m in self.lines(r):
            out.setdefault(m["type"], []).append(m)
        return out

    def test_the_trackers_objects_and_the_alert_transitions_are_recorded(self):
        """What a replay cannot give back (module docstring): the tracker's objects and the decision layer of the
        drive itself. Until lane "drive-quality" neither was in a session."""
        for envelope in (True, False):
            out = self._feed_decisions(self.rec(frame_every_s=0), envelope)
            self.assertIn(len(out["radar"]), (16, 17, 18))                # radar_hz 5 of 12.5: every third message
            self.assertEqual(out["radar"][0]["objects"][0]["id"], 7)      # as it went out
            self.assertNotIn("radar_fut", out)                            # three times the size: not by default
            dec = out["decide"]
            self.assertIn(len(dec), (8, 9, 10, 11))                       # decide_hz 2 of 10, and the transitions
            with_events = [m["seq"] for m in dec if m["events"]]
            self.assertEqual(with_events, [35, 40])                       # every one of them, whatever the cap
            self.assertEqual(len(out["governor"]), 4)
        self.assertTrue({"radar", "decide", "governor"} <= set(Recorder({"dir": "x"}).topics))

    def test_recorded_decision_topics_are_configurable(self):
        out = self._feed_decisions(self.rec(frame_every_s=0, radar_hz=1.0, decide_hz=0, more_topics=["radar_fut"]), True)
        self.assertIn(len(out["radar"]), (4, 5))
        self.assertEqual(len(out["decide"]), 20)                          # no cap: every message
        self.assertIn(len(out["radar_fut"]), (7, 8, 9))                   # radar_fut_hz 2
        out = self._feed_decisions(self.rec(frame_every_s=0, topics=["road", "decide"]), True)
        self.assertEqual(set(out), {"decide"})

    def test_session_started_on_a_stale_clock_is_renamed_when_the_clock_steps(self):
        wall = [1794300000.0]                                 # a boot without a clock source: days in the past
        r = Recorder({"dir": os.path.join(self.tmp.name, "rec")}, self.preview, clock=self.clock,
                     wall=lambda: wall[0] + self.clock.t)
        r.on_message("env", json.dumps({"type": "env", "seq": 1}))
        old = r.dir
        wall[0] += 439427.0                                   # the clock is set
        self.clock.t += 6.0
        r.on_message("env", json.dumps({"type": "env", "seq": 2}))
        self.assertNotEqual(r.dir, old)
        self.assertFalse(os.path.exists(old))
        import time as _t
        self.assertEqual(os.path.basename(r.dir), _t.strftime("%Y%m%d-%H%M%S", _t.localtime(wall[0] + self.clock.t - 6.0)))
        self.clock.t += 6.0
        r.on_message("env", json.dumps({"type": "env", "seq": 3}))      # no second rename
        self.assertEqual([m["seq"] for m in self.lines(r)], [1, 2, 3])


class TestRoadLine(unittest.TestCase):
    """The recorder's `road` line made from number tables is json.dumps of the reduced message."""

    def test_line_is_dumps_of_slim(self):
        import random
        from chris_integrator.io.recorder import slim_road_line
        rnd = random.Random(9)
        for k in range(200):
            m = road(k)
            for ln in m["lane_lines"] + m.get("road_edges", []):
                ln["x"] = [0, 3, 12] + [rnd.uniform(0, 250) for _ in range(30)]
                ln["y"] = [rnd.choice((rnd.uniform(-40, 40), -0.0004, 0.0625, 0)) for _ in range(33)]
                ln["y_std"] = [rnd.uniform(0, 3) for _ in range(33)]
            m["plan"] = {"x": [-0] + [rnd.uniform(0, 400) for _ in range(32)], "y": [rnd.gauss(0, 1e-3) for _ in range(33)],
                         "z": [rnd.uniform(-2, 2) for _ in range(33)], "t": [0.1] * 33, "held": True, "age_ms": 12.5}
            m["leads"] = [{"prob": 0.5, "t_offset": 0, "t": [0, 2, 4, 6, 8, 10], "x": [rnd.uniform(0, 200) for _ in range(6)]}]
            if k % 50 == 7:
                m["road_model"] = "\x01"                    # a string equal to the marker: the plain path
            if k % 50 == 9:
                m["lane_lines"][0]["y"][3] = None
            self.assertEqual(slim_road_line(m), json.dumps(slim_road(m), separators=(",", ":")))


if __name__ == "__main__":
    unittest.main()
