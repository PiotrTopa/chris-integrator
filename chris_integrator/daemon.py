"""chris-integrator as a daemon: the mixer with its channels.

    input channels
      each chris-roadsense instance's run directory: camerad.sock (`camera`), modeld.sock (`road`, `calib`),
        signd.sock (`signs`), lightd.sock (`lights`), envd.sock (`env`), laned.sock (`display_lanes`),
        api.sock (`governor`)
      <run dir>/feed.sock   the car backend (vehicle input API 1.1, docs/VEHICLE_INPUT.md: `gnss`, `motion`, `reset`,
                            `controls`, `body`, `vehicle`, `objects`; error replies as mapd gave them); also under
                            the name position.sock. Store and validation: io/vehicle_input.py
                            (<config dir>/vehicle_input.json); the `vehicle` topic twice a second and on changes
      POST /v2/vehicle      the same over HTTP
      --legacy-mapd         also a roadsense mapd's relays (ego.sock, vehicle.sock) while a car still feeds it
    plug-ins                the stack (`chris_integrator.stack`): estimator, map box, trackers, decisions, view
    output channels
      HTTP + WebSocket      the consolidated API (/v2/..., /ws), --http-port (0: off)
      <run dir>/integrator.sock   JSON lines of the public streams (a first line {"subscribe": [...]} narrows;
                            default: `ego_state`, `tracks`, `lead`)
      the recorder          <config dir>/recorder.json

JSON lines over Unix stream sockets (the stack's convention). A slow client is cut at 1 MiB of backlog.

Stopping (SIGTERM from the service manager, Ctrl-C). The signal sets an event in the loop; it never interrupts a
message half-way through the mixer. What the stop owes the next start is done first and does not wait for
anybody: the carried state is saved (`Stack.save`; it is also saved every STATE_SAVE_S while running, for the host
that loses power instead of being stopped) and the recorder's file is closed. Only then the clients are let go,
each step with its own limit (STOP_STEP_S), and a process that is still there STOP_HARD_S after the signal ends
itself: a client that never answers the close of its WebSocket, a `/preview` stream that the camera's debug face
keeps feeding, or a version of the HTTP library that waits a minute for them, cannot hold the stop (on
27 November 2026 one did: the unit was killed after 90 s).
"""
import argparse
import asyncio
import gc
import json
import logging
import os
import signal
import threading
import time

from . import native
from .io.vehicle_input import VehicleInput
from .io.sockets import LINE_LIMIT, LineServer, Reader, parse_line_trimmed
from .stack import build

log = logging.getLogger("chris_integrator")
OUT = ("ego_state", "tracks", "lead")
ROADSENSE_SOCKETS = ("camerad", "modeld", "signd", "lightd", "envd", "laned", "api")
LEGACY_SOCKETS = ("ego", "vehicle")
STATE_SAVE_S = 120.0            # the carried state is written this often while running
STOP_STEP_S = 2.0               # limit of each step of letting the clients go
STOP_HARD_S = 6.0               # a process still there this long after the stop signal ends itself


def load_config(path):
    try:
        with open(path) as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


class Daemon:
    def __init__(self, rundir, sources, cfg=None, datadir=None, config_dir=None, http=None, roadsense_http=None,
                 mapper="auto", out_streams=OUT, record="config", feed_cfg=None, mono_ns=None):
        """sources: list of (run directory of a roadsense instance, [socket names]); the first is the default
        instance, a further one's streams are suffixed with its directory's name (`road@rear`).
        cfg: the estimator's configuration (integrator.json). http: (host, port) or None.
        feed_cfg: overrides of the vehicle input store's configuration (<config dir>/vehicle_input.json).
        mono_ns: the clock the feed's stamps are checked against and the `vehicle` topic is stamped with (default
        CLOCK_MONOTONIC; a simulation on its own clock passes its own)."""
        self.rundir, self.sources, self.datadir, self.http, self.roadsense_http = rundir, sources, datadir, http, roadsense_http
        self.stack = build(cfg={"estimator": cfg or {}}, config_dir=config_dir, datadir=datadir, mapper=mapper)
        self.mixer = self.stack.mixer
        self.integ = self.mixer.plugin("estimator").integ
        vcfg = dict(load_config(os.path.join(config_dir, "vehicle_input.json")) if config_dir else {}, **(feed_cfg or {}))
        self.vin = VehicleInput(vcfg, calibration=self._calibration)
        self.mono_ns = mono_ns or time.monotonic_ns
        self.readers = {}
        self.out = LineServer(os.path.join(rundir, "integrator.sock"), streams=None)
        self.out_default = set(out_streams)
        self.mixer.add_output("socket", self._to_socket, info="integrator.sock: JSON lines for other processes")
        self.recorder = None
        if record:
            from .io import recorder as recorder_mod
            rec_cfg = recorder_mod.load_config(config_dir) if record == "config" else record
            if rec_cfg:
                preview = os.path.join(sources[0][0], "preview.jpg") if sources else None
                self.recorder = recorder_mod.Recorder(rec_cfg, preview)
                self.mixer.add_output("recorder", self._record, info="drive recorder")
        self.stats = {"road": 0, "in": 0, "cpu_s": 0.0, "t0": time.monotonic()}
        st = self.mixer.services["store"]
        self.state_file = st.path("estimator") if st.dir else None
        self.stop_event = None          # set by the stop signal (run())
        self.stopped = None             # how the stop went: {"state_saved", "steps": {name: "ok" | "timeout" | error}, "s"}
        self.state_save_s, self.stop_step_s, self.stop_hard_s = STATE_SAVE_S, STOP_STEP_S, STOP_HARD_S
        self.hard_stop = False          # True (main()): the process ends itself `stop_hard_s` after the stop signal
        self.hard_exit = os._exit       # (a test puts its own here)

    # ---- output channels ---------------------------------------------------------------------------------
    def _to_socket(self, env):
        out = self.out
        stream = env.stream
        default = stream in self.out_default
        if default:
            out.last[stream] = env         # serialised when a client connects (LineServer._client), not per message
        if not out.clients:
            return
        data = None
        for w, want in list(out.clients.items()):
            if (not default) if want is None else (stream not in want):
                continue
            if data is None:
                data = env.line.encode() + b"\n"
            if w.transport.get_write_buffer_size() > (1 << 20):
                del out.clients[w]
                w.close()
                continue
            w.write(data)
            out.out_bytes += len(data)

    def _record(self, env):
        rec = self.recorder
        if env.stream == "governor":
            rec.frames_enabled = (env.msg.get("caps") or {}).get("record_frames", -1) != 0
        rec.on_envelope(env)

    # ---- input channels ----------------------------------------------------------------------------------
    def on_message(self, msg, line=None, instance=None):
        c = time.process_time()
        self.stats["in"] += 1
        if msg.get("type") == "road":
            self.stats["road"] += 1
        try:
            self.stack.deliver(msg, line, instance)
        except Exception:
            log.exception("the mixer failed on a %s message", msg.get("type"))
        self.stats["cpu_s"] += time.process_time() - c

    def _calibration(self):
        """The estimator's calibration for the `vehicle` topic (its newest `ego_state`)."""
        e = self.mixer.bus.latest.get("ego_state")
        return (e.msg or {}).get("calibration") if e is not None else None

    def feed_message(self, m, key=None):
        """One feeder message: None, or the reason it was rejected (the error reply)."""
        outs, err = self.vin.on_message(m, self.mono_ns(), time.time_ns(), key)
        for im in outs:
            self.on_message(im)
        return err

    def feed_messages(self, msgs):
        """Feeder messages (vehicle input API) from HTTP: {"accepted", "rejected": [{"index", "reason"}]}."""
        rejected = []
        for k, m in enumerate(msgs):
            err = self.feed_message(m)
            if err is not None:
                rejected.append({"index": k, "reason": err})
        return {"accepted": len(msgs) - len(rejected), "rejected": rejected}

    async def feed_client(self, reader, writer):
        """One feeder connection, line for line as mapd's position.sock: a hello reply to the feeder's hello, an
        error line `{"type":"error","line":n,"reason":...}` for every rejected line (n counts the lines of this
        connection from 1), nothing for an accepted one; the connection stays open."""
        vin = self.vin

        def reply(obj):
            writer.write(json.dumps(obj, separators=(",", ":")).encode() + b"\n")
        vin.connect(writer, self.mono_ns())
        n = 0
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                n += 1
                line = raw.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except ValueError:
                    vin.counts["errors"] += 1
                    vin.last_error = "bad JSON"
                    reply({"type": "error", "line": n, "reason": "bad JSON"})
                    continue
                if isinstance(m, dict) and m.get("type") == "hello":
                    reply(vin.hello(writer, m, self.mono_ns()))
                    continue
                err = self.feed_message(m, writer)
                if err is not None:
                    reply({"type": "error", "line": n, "reason": err})
        except (OSError, ValueError, asyncio.LimitOverrunError) as e:
            log.warning("feeder: %s", e)
        finally:
            vin.disconnect(writer)
            writer.close()

    async def vehicle_ticks(self, period=0.05):
        """The `vehicle` topic also without feeder messages (twice a second; `present: false` without a feeder), and a
        change that the 20-a-second limit held back goes out within `period`."""
        while True:
            m = self.vin.tick(self.mono_ns(), time.time_ns())
            if m is not None:
                self.on_message(m)
            await asyncio.sleep(period)

    def stop(self):
        """Ask the daemon to stop (from the loop's thread: a signal handler, a test)."""
        if self.stop_event is not None:
            self.stop_event.set()

    def save_state(self):
        """The carried state to the data directory; False when it could not be written."""
        try:
            self.stack.save()
            return True
        except Exception:
            log.exception("the carried state was not saved")
            return False

    async def state_saver(self):
        """The carried state, every `state_save_s`: a host that loses power is not stopped first."""
        while True:
            await asyncio.sleep(self.state_save_s)
            self.save_state()

    async def _shutdown(self, tasks, feed_srvs, runner):
        """The stop, in the order of what it owes (module docstring). Never raises, never waits without a limit."""
        t0 = time.monotonic()
        how = self.stopped = {"state_saved": False, "steps": {}, "s": None}

        def die():                                   # the limit of the whole stop, from a thread of its own
            log.warning("chris-integrator: still stopping after %.0f s (%s); ending the process",
                        self.stop_hard_s, ", ".join("%s: %s" % kv for kv in how["steps"].items()) or "no step done")
            self.hard_exit(0 if how["state_saved"] else 1)
        if self.hard_stop:                           # never cancelled: it also covers what asyncio.run() does after run()
            timer = threading.Timer(self.stop_hard_s, die)
            timer.daemon = True
            timer.start()
        how["state_saved"] = self.save_state()       # first, and without waiting for anybody
        if self.recorder is not None:
            try:
                how["steps"]["recorder"] = "ok" if self.recorder.close(timeout=self.stop_step_s) else "timeout"
            except Exception as e:
                how["steps"]["recorder"] = repr(e)
        for t in tasks:
            t.cancel()
        for name, fn in (("mixer", self.mixer.close), ("socket", self.out.close)):
            try:
                fn()
                how["steps"][name] = "ok"
            except Exception as e:
                how["steps"][name] = repr(e)
        for srv in feed_srvs:
            srv.close()
        if runner is not None:
            try:
                await asyncio.wait_for(runner.cleanup(), self.stop_step_s + 1.0)
                how["steps"]["http"] = "ok"
            except asyncio.TimeoutError:
                how["steps"]["http"] = "timeout"
            except Exception as e:
                how["steps"]["http"] = repr(e)
        if tasks:
            await asyncio.wait(tasks, timeout=self.stop_step_s)
        how["s"] = round(time.monotonic() - t0, 2)
        log.info("chris-integrator: stop took %.2f s (state %s; %s)", how["s"], "saved" if how["state_saved"] else "NOT saved",
                 ", ".join("%s: %s" % kv for kv in how["steps"].items()))

    async def run(self):
        os.makedirs(self.rundir, exist_ok=True)
        loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:                                     # in the loop: the handler runs between two messages
                loop.add_signal_handler(sig, self.stop_event.set)
            except (NotImplementedError, RuntimeError, ValueError):
                pass                                 # not the main thread (a test), or a platform without it
        await self.out.start()
        feed_srvs = []
        for name in ("feed.sock", "position.sock"):     # position.sock: the name a car backend written for mapd knows
            p = os.path.join(self.rundir, name)
            try:
                os.unlink(p)
            except OSError:
                pass
            feed_srvs.append(await asyncio.start_unix_server(self.feed_client, path=p, limit=LINE_LIMIT))
            os.chmod(p, 0o666)
        tasks = [asyncio.create_task(self.vehicle_ticks()), asyncio.create_task(self.state_saver())]
        for i, (d, socks) in enumerate(self.sources):
            inst = None if i == 0 else os.path.basename(d.rstrip("/"))
            if inst is not None:
                from .stack import ROADSENSE
                self.stack.add_channel("roadsense@" + inst, ROADSENSE, instance=inst, info="a further chris-roadsense instance")
            for s in socks:
                name = s[:-5] if s.endswith(".sock") else s
                r = Reader(os.path.join(d, name + ".sock"), lambda m, ln, inst=inst: self.on_message(m, ln, inst), name,
                           parse=parse_line_trimmed if self.stack.trim_road else None)
                if i == 0:
                    self.readers[name] = r
                tasks.append(asyncio.create_task(r.run()))
        runner = None
        if self.http:
            from aiohttp import web
            from .io.http import make_app
            app = make_app(self.stack, self.readers, self.datadir, self.feed_messages, self.recorder, self.roadsense_http)
            app["stop_step_s"] = self.stop_step_s
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            await web.TCPSite(runner, self.http[0], self.http[1]).start()
        for slot in self.mixer.slots.values():          # out-of-process plug-ins in async mode: publish as they answer
            pl = slot.plugin
            if getattr(pl, "remote", False) and not pl.sync and pl.fileno() is not None:
                loop.add_reader(pl.fileno(), pl.pump)
        log.info("chris-integrator: %d source sockets, %d plug-ins (%s), state size %d, http %s", len(tasks) - 2,
                 len(self.mixer.order), ", ".join(s.name for s in self.mixer.order), self.integ.model.n, self.http)
        # what start-up built lives as long as the process: out of the collector's sight from here on. A collection
        # of the older generations otherwise walks all of it, and stalls a frame by tens of milliseconds on a
        # phone's little core (tools/cost_bench.py --gc freeze); what is made from now on is collected as before
        gc.collect()
        gc.freeze()
        nat = native.status()
        log.info("chris-integrator: native core version %d (%s; %s)", nat["version"], nat["module"], ", ".join(nat["parts"]))
        stop = asyncio.create_task(self.stop_event.wait())
        try:
            # until the stop signal; a task of the daemon's own that ends (it should not) ends the daemon with it
            done, _ = await asyncio.wait([stop] + tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t is not stop and not t.cancelled() and t.exception() is not None:
                    log.error("chris-integrator: a task of the daemon failed: %r", t.exception())
        finally:
            stop.cancel()
            await asyncio.shield(self._shutdown(tasks, feed_srvs, runner))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    ap.add_argument("--rundir", default=env("CHRIS_RUNDIR", "/run/chris"))
    ap.add_argument("--datadir", default=env("CHRIS_DATADIR", "/var/lib/chris"))
    ap.add_argument("--config-dir", default=env("CHRIS_CONFIG_DIR", "/etc/chris"))
    ap.add_argument("--config", default=None, help="the estimator's configuration file (default <config dir>/integrator.json)")
    ap.add_argument("--roadsense", action="append", default=None,
                    help="run directory of a chris-roadsense instance (repeat for more cameras; default /run/roadsense)")
    ap.add_argument("--roadsense-http", default=env("CHRIS_ROADSENSE_HTTP"),
                    help="base URL of the first instance's debug face; /preview, /v2/frame.jpg, /v2/governor are forwarded to it")
    ap.add_argument("--legacy-mapd", action="store_true",
                    help="also read a roadsense mapd's relays (ego.sock, vehicle.sock) while a car still feeds its position.sock")
    ap.add_argument("--no-mapd", action="store_true", help="(default now; kept for older unit files)")
    ap.add_argument("--no-mapper", action="store_true", help="run without the map box")
    # built-in defaults for a stand-alone start by hand: 8080 on all interfaces. An installed unit sets
    # CHRIS_API_PORT=8090 and CHRIS_API_HOST=127.0.0.1 (scripts/install.sh, docs/INSTALL.md)
    ap.add_argument("--host", default=env("CHRIS_API_HOST", "0.0.0.0"), help="bind address (default $CHRIS_API_HOST, else all interfaces)")
    ap.add_argument("--http-port", type=int, default=int(env("CHRIS_API_PORT", "8080")),
                    help="consolidated API port (default $CHRIS_API_PORT, else 8080; 0: none)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(a.config) if a.config else None
    socks = list(ROADSENSE_SOCKETS) + (list(LEGACY_SOCKETS) if a.legacy_mapd else [])
    sources = [(d, socks if i == 0 else list(ROADSENSE_SOCKETS)) for i, d in enumerate(a.roadsense or ["/run/roadsense"])]
    d = Daemon(a.rundir, sources, cfg, a.datadir, a.config_dir, (a.host, a.http_port) if a.http_port else None,
               a.roadsense_http, mapper=None if a.no_mapper else "auto")
    d.hard_stop = True
    # systemd stops a unit with SIGTERM: run() takes it (and Ctrl-C) in its loop, saves the carried state and lets
    # the clients go within limits (module docstring, "Stopping")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(d.run())
    except KeyboardInterrupt:
        pass
    log.info("chris-integrator: stopped")
    # run() has saved the state and let the clients go as far as they went within its limits. What is left (the tasks of
    # clients that did not let go, the loop's and the interpreter's own teardown, which wait for them without a limit)
    # is not worth the service manager's time: the process ends here.
    logging.shutdown()
    os._exit(0 if d.stopped is None or d.stopped["state_saved"] else 1)


if __name__ == "__main__":
    main()
