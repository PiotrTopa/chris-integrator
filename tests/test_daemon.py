"""The daemon: sources over Unix sockets in, `ego_state` / `tracks` / `lead` lines out; the car's direct feed."""
import asyncio
import json
import os
import tempfile
import unittest

from . import simrun  # noqa: F401  (path)
from chris_integrator import fastjson
from chris_integrator.daemon import Daemon
from chris_integrator.feed import Feed
from chris_integrator.sim import World, default_sensors


class DaemonTest(unittest.TestCase):
    def test_sockets_end_to_end(self):
        w = World(seed=1)
        msgs = [(t, m) for t, m in w.messages(default_sensors()) if t - w.t0 < 20.0]

        async def go():
            tmp = tempfile.mkdtemp(prefix="chris-test-")
            rs, run = os.path.join(tmp, "rs"), os.path.join(tmp, "chris")
            os.makedirs(rs)
            clients = []

            async def modeld(reader, writer):
                clients.append(writer)
                await reader.read()
            srv = await asyncio.start_unix_server(modeld, path=os.path.join(rs, "modeld.sock"))
            # the simulation runs on its own clock (from 1000 s): the `vehicle` topic is stamped with it; the messages
            # are written faster than they are read, so the age check (against the newest written) is off
            sim_now = [int(msgs[0][1]["ts_mono_ns"])]
            d = Daemon(run, [(rs, ["modeld.sock"])], {}, tmp, feed_cfg={"max_input_age_s": None}, mono_ns=lambda: sim_now[0])
            task = asyncio.create_task(d.run())
            for _ in range(100):
                await asyncio.sleep(0.02)
                if clients and os.path.exists(os.path.join(run, "feed.sock")):
                    break
            fr, fw = await asyncio.open_unix_connection(os.path.join(run, "feed.sock"))
            fw.write(b'{"type":"hello","api":1,"name":"sim"}\n')
            hello = json.loads(await fr.readline())
            out_r, out_w = await asyncio.open_unix_connection(os.path.join(run, "integrator.sock"), limit=1 << 24)
            for t, m in msgs:
                sim_now[0] = max(sim_now[0], int(m.get("ts_mono_ns") or 0))
                if m["type"] == "road":
                    clients[0].write(json.dumps(m, separators=(",", ":")).encode() + b"\n")
                else:
                    fw.write(json.dumps(dict(m, type="motion") if m["type"] == "ego" else m).encode() + b"\n")
                await asyncio.sleep(0)
            await asyncio.sleep(0.3)
            got = {}
            try:
                while True:
                    raw = await asyncio.wait_for(out_r.readline(), 0.2)
                    if not raw:
                        break
                    m = json.loads(raw)
                    got.setdefault(m["type"], []).append(m)
            except asyncio.TimeoutError:
                pass
            task.cancel()
            srv.close()
            return hello, got, d
        hello, got, d = asyncio.run(go())
        self.assertIn("motion", hello["accepts"])
        frames = {m["ts_mono_ns"] for _, m in msgs if m["type"] == "road"}
        for t in ("ego_state", "tracks", "lead"):               # every road frame is answered by all three
            self.assertTrue(frames <= {m["ts_mono_ns"] for m in got[t]}, t)
        self.assertGreater(len(got["ego_state"]), len(frames))      # and the car's samples before the first frame too
        es, ld = got["ego_state"][-1], got["lead"][-1]
        self.assertEqual(es["sources"]["vehicle"]["state"], "ok")            # the car's feed arrived over feed.sock
        self.assertTrue(es["speed"]["valid"])
        self.assertEqual(ld["ts_mono_ns"], es["ts_mono_ns"])
        self.assertTrue(ld["present"])
        self.assertTrue(os.path.exists(d.state_file))                        # calibration and faults carried on disk

    def test_feed_and_partial_parse(self):
        f = Feed()
        t0 = 100 * 10 ** 9
        ego = f.convert({"type": "motion", "ts_mono_ns": t0, "speed_mps": 3.0, "gear": "R", "yaw_rate_rps": 0.1}, now_ns=t0)[0]
        self.assertEqual((ego["type"], ego["speed_mps"], ego["reverse"], ego["yaw_source"]), ("ego", -3.0, True, "yaw_rate"))
        v = f.convert({"type": "controls", "brake": True}, now_ns=t0 + 10 ** 9)[0]
        self.assertEqual((v["type"], v["controls"]["brake"], v["ts_mono_ns"]), ("vehicle", True, t0 + 10 ** 9))
        g = f.convert({"type": "gnss", "age_ms": 500, "lat": 1.0, "lon": 2.0}, now_ns=t0 + 2 * 10 ** 9)[0]
        self.assertEqual(g["ts_mono_ns"], t0 + 15 * 10 ** 8)
        self.assertEqual(f.convert({"type": "gnss", "age_ms": 500, "lat": 1.0}, now_ns=t0 + 3 * 10 ** 9), [])
        self.assertEqual(f.last_error, "missing field lon")                  # rejected with mapd's reason
        road = {"type": "road", "seq": 1, "pose": {"vx": 1.0}, "lane_lines": [{"x": [1, 2], "y": [0.5, 1], "prob": 0.3}],
                "road_edges": [{"y": [1.0]}], "leads": [{"prob": 0.9, "x": [10.0]}], "plan": {"x": [0, 1], "y": [0, 0]},
                "meta": {"a": [1, 2], "b": {"c": 3}}}
        line = json.dumps(road, separators=(",", ":")).encode()
        got = fastjson.loads_road(line)
        self.assertEqual({k: road[k] for k in ("type", "seq", "pose", "leads", "plan")}, got)
        odd = json.dumps(dict(road, lane_lines=[{"note": 'a " quote'}]), separators=(",", ":")).encode()
        self.assertEqual(fastjson.loads_road(odd)["pose"], road["pose"])     # anything unexpected: the whole line is parsed


class StopTest(unittest.TestCase):
    """The stop (SIGTERM from the service manager): the carried state first, the clients within limits, the process
    never longer than its own limit. On 27 November 2026 the unit was killed 90 s after its stop signal."""

    def daemon(self, tmp, record=True, **kw):
        rs, run = os.path.join(tmp, "rs"), os.path.join(tmp, "chris")
        os.makedirs(rs, exist_ok=True)
        rec = {"dir": os.path.join(tmp, "rec"), "frame_every_s": 0} if record else None
        return Daemon(run, [(rs, ["modeld.sock"])], {}, tmp, record=rec, feed_cfg={"max_input_age_s": None}, **kw)

    def feed(self, d, seconds=5.0):
        w = World(seed=1)
        n = 0
        for t, m in w.messages(default_sensors()):
            if t - w.t0 > seconds:
                break
            d.on_message(dict(m))
            n += 1
        return n

    def test_stop_saves_the_state_closes_the_recording_and_is_quick(self):
        import gzip
        import time

        async def go():
            tmp = tempfile.mkdtemp(prefix="chris-test-")
            d = self.daemon(tmp)
            task = asyncio.create_task(d.run())
            await asyncio.sleep(0.1)
            self.feed(d)
            t0 = time.monotonic()
            d.stop()
            await asyncio.wait_for(task, 3.0)
            return d, time.monotonic() - t0
        d, took = asyncio.run(go())
        self.assertLess(took, 1.0)
        self.assertTrue(d.stopped["state_saved"])
        self.assertEqual(d.stopped["steps"]["recorder"], "ok")
        saved = json.load(open(d.state_file))
        self.assertIn("calibration", saved)
        with gzip.open(os.path.join(d.recorder.dir, "messages.jsonl.gz"), "rt") as f:   # a whole gzip stream: closed
            kinds = {json.loads(ln)["type"] for ln in f}
        # the stack's wiring: the tracker's objects and the decision layer reach the recorder
        self.assertTrue({"road", "ego_state", "radar", "decide"} <= kinds, kinds)

    def test_the_state_is_also_saved_while_running(self):
        async def go():
            tmp = tempfile.mkdtemp(prefix="chris-test-")
            d = self.daemon(tmp, record=False)
            d.state_save_s = 0.05
            task = asyncio.create_task(d.run())
            await asyncio.sleep(0.1)
            self.feed(d, 2.0)
            await asyncio.sleep(0.15)
            saved = os.path.exists(d.state_file)                     # a host that loses power is not stopped first
            d.stop()
            await asyncio.wait_for(task, 3.0)
            return saved
        self.assertTrue(asyncio.run(go()))

    def test_a_client_that_does_not_let_go_does_not_hold_the_stop(self):
        """The HTTP side's cleanup waits for its clients (a WebSocket that never answers the close, a stream without
        an end; some versions of the library wait a minute): the stop gives it its limit and goes on."""
        import time

        class Runner:
            async def cleanup(self):
                await asyncio.sleep(3600)

        async def go():
            tmp = tempfile.mkdtemp(prefix="chris-test-")
            d = self.daemon(tmp)
            d.stop_step_s = 0.2
            self.feed(d, 2.0)
            t0 = time.monotonic()
            await d._shutdown([], [], Runner())
            return d, time.monotonic() - t0
        d, took = asyncio.run(go())
        self.assertLess(took, 2.0)
        self.assertEqual(d.stopped["steps"]["http"], "timeout")
        self.assertTrue(d.stopped["state_saved"])
        self.assertTrue(os.path.exists(d.state_file))

    def test_a_stop_that_cannot_finish_ends_the_process_with_the_state_saved(self):
        """Something blocks the loop itself (no limit of a step can interrupt that): after `stop_hard_s` the process
        ends itself, with exit code 0 because the state is on the disk by then."""
        import time
        calls = []

        class Runner:
            async def cleanup(self):
                time.sleep(0.6)                                      # blocks the thread, not an await

        async def go():
            tmp = tempfile.mkdtemp(prefix="chris-test-")
            d = self.daemon(tmp, record=False)
            d.hard_stop, d.stop_hard_s = True, 0.2
            d.hard_exit = lambda code: calls.append((code, os.path.exists(d.state_file)))
            self.feed(d, 2.0)
            await d._shutdown([], [], Runner())
        asyncio.run(go())
        self.assertEqual(calls, [(0, True)])

    def test_sigterm_stops_the_process(self):
        """The daemon as a process, as the unit runs it (no HTTP side: that part is in tests/test_api.py)."""
        import signal
        import subprocess
        import sys
        import time
        tmp = tempfile.mkdtemp(prefix="chris-test-")
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p = subprocess.Popen([sys.executable, "-m", "chris_integrator", "--rundir", os.path.join(tmp, "chris"),
                              "--datadir", tmp, "--config-dir", os.path.join(tmp, "etc"), "--roadsense",
                              os.path.join(tmp, "rs"), "--http-port", "0", "--no-mapper"],
                             env=dict(os.environ, PYTHONPATH=here), stderr=subprocess.PIPE, text=True)
        try:
            end = time.monotonic() + 20.0
            while not os.path.exists(os.path.join(tmp, "chris", "feed.sock")) and time.monotonic() < end:
                time.sleep(0.05)
            self.assertTrue(os.path.exists(os.path.join(tmp, "chris", "feed.sock")))
            time.sleep(0.3)
            t0 = time.monotonic()
            p.send_signal(signal.SIGTERM)
            rc = p.wait(10.0)
            took = time.monotonic() - t0
        finally:
            if p.poll() is None:
                p.kill()
        err = p.stderr.read()
        self.assertEqual(rc, 0, err[-2000:])
        self.assertLess(took, 3.0)
        self.assertIn("state saved", err)
        self.assertIn("chris-integrator: stopped", err)
        self.assertTrue(os.path.exists(os.path.join(tmp, "integrator.json")))


if __name__ == "__main__":
    unittest.main()
