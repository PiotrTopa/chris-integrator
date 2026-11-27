"""Drive recorder: what the stack saw and said, kept on disk for the replay tools.

Off unless `<config dir>/recorder.json` exists (or `make_app(record=...)`): `{"dir": "/var/lib/chris/rec"}`
switches it on. Every start of the integrator (the host starts the stack when the car is switched on) opens a session
directory `<dir>/<YYYYmmdd-HHMMSS>/` with

    messages.jsonl.gz   the topics as they went out on /ws, one JSON object per line, in order: `road` reduced to
                        what the trackers and the replay tools read (lane lines, road edges, leads[0], plan x / y,
                        speed, pose) with the numbers shortened, at most `road_hz` per second; `lights`, `signs`,
                        `env`, `speed_limit`, `rules`, `fused`, `map`, `ego` (at most `ego_hz`), `vehicle` (what
                        the host says about the car), `objects` (range-sensor scans, at most `objects_hz`),
                        `view` (the display channel as consumers got it, with the road picture `view.road`, at most `view_hz`),
                        `calib` and one `camera` message a second, as they are; `radar` (the object tracker's objects as
                        the stack had them on the drive, at most `radar_hz`), `decide` (the decision layer: at most
                        `decide_hz`, and EVERY message that carries an alert transition in `events`, whatever the cap),
                        `governor` (the camera instance's load governor: level, what is shed, the headroom; at most
                        `governor_hz`). `lanes` and `lead` are not written: they are
                        derived from `road` and `ego`, and chris-roadsense `tools/lanes_replay.py` / chris-roadsense `tools/lead_replay.py`
                        recompute them from this file. `radar_fut` (the objects with their predicted futures, three times
                        the size of `radar`) is not written by default: name it in `topics` (cap `radar_fut_hz`).

What a review of a drive needs and a replay cannot give back: a replay of `road` / `lights` / `ego` through today's
code tells what today's tracker and decision layer would do, not what the car's did (another configuration, another
version, the load governor's shedding, the arrival order of the messages). Until lane "drive-quality" the tracker's
objects and the alerts of a drive were only in the journal's log lines.
    frames/<seq>.jpg    the camera's preview JPEG every `frame_every_s` seconds (the file camerad already
                        writes for its /preview; copied, not re-encoded; only when the recorder is given that path): the pictures a detector replay
                        (roadsense-signs `lights_replay`, `signd_replay`) needs to say what was there to see.

The file is flushed and synced to the disk every 5 s (a host that loses power keeps all but the last
seconds). Compressing and writing happen on a thread of the recorder's own, in the order the lines came: the
compressor works through its buffer in pieces of about 128 kB (tens of milliseconds on a phone's little core),
and the per-message path does not wait for that. A session opened while the wall clock was wrong (a boot without a clock source) is renamed to its
true start time when the clock is stepped.

Cost: about 40 kB/s of compressed messages (`radar`, `decide` and `governor` are 1-3 kB/s of that: 1.1 kB/s in a
replay of the 9-minute night drive of 27 November 2026, more in dense traffic) and one 40-110 kB JPEG a second, a few milliseconds of one little
core per second: 4000 MB hold about ten hours of driving. Old sessions are deleted, oldest first, when the directory exceeds `max_mb`. Nothing leaves
the device. The recorder never raises into the fan-out: a full disk stops the recording, not the api.

Config keys (all optional but `dir`): `max_mb` (4000), `road_hz` (20), `ego_hz` (25), `ego_state_hz` (2), `objects_hz` (10), `view_hz` (2),
`radar_hz` (5), `decide_hz` (2), `governor_hz` (1), `radar_fut_hz` (2), `frame_every_s` (1.0; 0 = no frames),
`topics` (list; default below), `more_topics` (list: topics on top of `topics`, e.g. `["radar_fut"]`), `stamp_hz` ({}; see below).

The `<topic>_hz` caps are counted on the recorder's clock (the host's monotonic clock: when a message reaches the
recorder). `stamp_hz`, a map topic -> rate (e.g. `{"road": 10}`), thins the topics it names by the messages' own
`ts_mono_ns` instead: a message is written when its stamp is at least 1/rate - 5 ms after the stamp of the last one
written (or earlier than it: a new clock); so the choice does not depend on when a message arrived, and a replay
of the same messages writes the same lines. A message of such a topic without an integer `ts_mono_ns` falls back
to the clock cap. Not set (the default): every topic by its clock cap, as before. `{"road": 10}` halves the
recorded `road` lines of a 20 Hz camera (every second frame) and with them most of the recorder's per-frame cost.
"""
import gzip
import json
import logging
import os
import queue
import shutil
import threading
import time

from ..native import core as _C

log = logging.getLogger("chris_integrator.recorder")

DEFAULT_TOPICS = ("road", "lights", "signs", "env", "speed_limit", "rules", "fused", "map", "ego", "calib", "camera",
                  "vehicle", "objects", "view", "gnss", "ego_state", "tracks", "radar", "decide", "governor")
DEFAULTS = dict(dir=None, max_mb=4000, road_hz=20.0, ego_hz=25.0, objects_hz=10.0, view_hz=2.0, frame_every_s=1.0,
                ego_state_hz=2.0, radar_hz=5.0, decide_hz=2.0, governor_hz=1.0, radar_fut_hz=2.0,
                topics=list(DEFAULT_TOPICS), more_topics=[], stamp_hz={})
CLOCK_STEP_S = 120.0            # the wall clock moved this much against the monotonic one: re-date the session


def load_config(config_dir=None):
    """`<config dir>/recorder.json` as a dict with a `dir`, or None (no file, no dir: the recorder stays off)."""
    path = os.path.join(config_dir or os.environ.get("CHRIS_CONFIG_DIR", "/etc/chris"), "recorder.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict) or not cfg.get("dir"):
        return None
    return {k: v for k, v in cfg.items() if k in DEFAULTS}


def slim_road_line(road):
    """The part of a `road` message the trackers and the replay tools read, numbers shortened, as one JSON line
    (the native core: chris_native.c `slim_road_line`; json.dumps writes what it leaves to json):

        type, seq, ts_mono_ns, ts_real_ns, model_ms, v_ego, valid, frames_skipped, pair_dt_ms, road_model,
        road_model_fallback, policy_hz, frame_seq_prev, camera     as they are (those present)
        pose            its floats to 4 decimals
        lane_lines, road_edges    of each dict: x (2 decimals), y, y_std (3), prob (a number, 4)
        leads           the first, every list in it to 3 decimals
        plan            x, y, z (2 decimals; those present), held, age_ms

    `road_model` selects the trackers' profile in a replay (it was dropped until lane L37, so replays ran the
    supercombo profile on vision-model sessions), `camera` names the camera. A list is written with every number
    rounded (round(v, n); None stays null), anything else as it is."""
    return _C.slim_road_line(road)


def slim_road(road):
    """`slim_road_line` as a dict."""
    return json.loads(slim_road_line(road))


class Recorder:
    def __init__(self, cfg, preview_path=None, clock=time.monotonic, wall=time.time):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self.preview_path = preview_path
        self.clock, self.wall = clock, wall
        self.topics = set(self.cfg["topics"]) | set(self.cfg["more_topics"] or ())
        c = self.cfg
        self._caps = {"road": c["road_hz"], "ego": c["ego_hz"], "objects": c["objects_hz"], "view": c["view_hz"],
                      "camera": 1.0, "ego_state": c["ego_state_hz"], "tracks": c["ego_state_hz"],
                      "radar": c["radar_hz"], "decide": c["decide_hz"], "governor": c["governor_hz"],
                      "radar_fut": c["radar_fut_hz"]}
        # topic -> least spacing of the written messages' ts_mono_ns, in ns (`stamp_hz`; the 5 ms of the clock cap)
        self._stamp = {k: 1e9 / hz - 5e6 for k, hz in (c["stamp_hz"] or {}).items()
                       if isinstance(hz, (int, float)) and not isinstance(hz, bool) and hz > 0}
        self._last_ts = {}          # topic -> ts_mono_ns of the last written message (stamp_hz)
        self.dir = None
        self.fh = None
        self.failed = False
        self.frames_enabled = True      # the load governor switches the pictures off under heat (governor.py)
        self.written = {"messages": 0, "frames": 0, "bytes_in": 0}
        self._last = {}             # topic -> clock of the last written message (rate caps)
        self._flush_t = 0.0
        self._frame_t = -1e9
        self._prune_t = 0.0
        self._q = None              # lines (and flush marks) for the writer thread
        self._thread = None

    # ---- session ---------------------------------------------------------
    def _open(self):
        base = self.cfg["dir"]
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.wall()))
        self.dir = os.path.join(base, name)
        os.makedirs(os.path.join(self.dir, "frames"), exist_ok=True)
        self.fh = gzip.open(os.path.join(self.dir, "messages.jsonl.gz"), "at", compresslevel=3)
        self._opened = (self.clock(), self.wall())
        self._q = queue.Queue(maxsize=4096)        # (a writer that falls far behind slows the caller, never loses lines)
        self._thread = threading.Thread(target=self._writer, args=(self.fh, self._q), name="recorder", daemon=True)
        self._thread.start()
        self.prune()
        log.info("recording to %s (cap %s MB)", self.dir, self.cfg["max_mb"])

    _FLUSH = object()               # a mark in the writer's queue: flush the file and sync it to the disk
    _STOP = object()

    def _writer(self, fh, q):
        """The writer thread: every line into the compressed file, in order; flush + fsync at each mark."""
        while True:
            item = q.get()
            if item is self._STOP:
                return
            if self.failed:
                continue
            try:
                if item is self._FLUSH:
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except (OSError, ValueError, AttributeError):
                        pass
                else:
                    fh.write(item)
            except (OSError, ValueError) as e:
                if not self.failed:
                    log.warning("recorder stopped (write): %s", e)
                self.failed = True

    def close(self, timeout=None):
        """Write what is queued and close the file. `timeout` (s): how long to wait for the writer thread (a disk
        that does not answer); False when it did not finish in time (the file is then closed under it: the lines
        still queued are lost, what was written stays), True otherwise."""
        ok = True
        if self._thread is not None:
            try:
                self._q.put(self._STOP, timeout=timeout)
            except queue.Full:
                ok = False
            self._thread.join(timeout)
            ok = ok and not self._thread.is_alive()
            if not ok:
                self.failed = True              # the writer drops what it still takes from the queue
                log.warning("recorder: the writer did not finish within %s s", timeout)
            self._thread = self._q = None
        if self.fh is not None:
            try:
                self.fh.close()
            except (OSError, ValueError):
                pass
            self.fh = None
        return ok

    def _fail(self, what, exc):
        if not self.failed:
            log.warning("recorder stopped (%s): %s", what, exc)
        self.failed = True
        self.close()

    # ---- fan-out hook ----------------------------------------------------
    def on_envelope(self, env):
        """The recorder as an output channel of the mixer: the message is serialised only when it is written
        (after the rate cap), and `road` is reduced from the message the bus already holds."""
        mtype = env.stream
        if self.failed or mtype not in self.topics:
            return
        now = self.clock()
        if mtype == "decide" and isinstance(env.msg, dict) and env.msg.get("events"):
            self._last[mtype] = now             # an alert transition: written whatever the cap
        elif self._stamp and mtype in self._stamp and env.ts is not None:
            if not self._due_ts(mtype, env.ts, now):
                return
        elif not self._due(mtype, now):
            return
        if mtype == "road" and isinstance(env.msg, dict):
            self._write(mtype, slim_road_line(env.msg), now)
        else:
            self._write(mtype, env.line, now)

    def on_message(self, mtype, line):
        """Called with every outgoing message (topic, serialised line)."""
        if self.failed or mtype not in self.topics:
            return
        now = self.clock()
        msg = None
        if mtype == "decide" and '"events":[]' not in line and '"events": []' not in line and '"events"' in line:
            self._last[mtype] = now             # an alert transition: written whatever the cap
        elif self._stamp and mtype in self._stamp:
            msg = json.loads(line)
            ts = msg.get("ts_mono_ns") if isinstance(msg, dict) else None
            if isinstance(ts, int):
                if not self._due_ts(mtype, ts, now):
                    return
            elif not self._due(mtype, now):
                return
        elif not self._due(mtype, now):
            return
        if mtype == "road":
            line = slim_road_line(json.loads(line) if msg is None else msg)
        self._write(mtype, line, now)

    def _due(self, mtype, now):
        """The rate cap of a topic: False when its last written message is too recent."""
        cap = self._caps.get(mtype)
        if cap and now - self._last.get(mtype, -1e9) < 1.0 / cap - 0.005:
            return False
        self._last[mtype] = now
        return True

    def _due_ts(self, mtype, ts, now):
        """The `stamp_hz` cap of a topic: False when the last written message's stamp is too close before `ts`."""
        last = self._last_ts.get(mtype)
        if last is not None and last <= ts and ts - last < self._stamp[mtype]:
            return False
        self._last_ts[mtype] = ts
        self._last[mtype] = now
        return True

    def _write(self, mtype, line, now):
        try:
            if self.fh is None:
                self._open()
            self._q.put(line + "\n")
            self.written["messages"] += 1
            self.written["bytes_in"] += len(line) + 1
            if now - self._flush_t > 5.0:
                self._flush_t = now
                self._q.put(self._FLUSH)
                self._redate(now)
            if mtype == "camera":
                self._frame(now, line)
            if now - self._prune_t > 300.0:
                self.prune()
        except (OSError, ValueError) as e:
            self._fail("write", e)

    def _redate(self, now):
        """The wall clock was stepped since the session was opened (it started on a stale clock): name the
        directory after the true start time. The open file follows the rename."""
        c0, w0 = self._opened
        start = self.wall() - (now - c0)
        if abs(start - w0) < CLOCK_STEP_S:
            return
        base = self.cfg["dir"]
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(start))
        new = os.path.join(base, name)
        n = 1
        while os.path.exists(new):
            new = os.path.join(base, "%s-%d" % (name, n))
            n += 1
        try:
            os.rename(self.dir, new)
        except OSError as e:
            log.warning("recorder: session not re-dated: %s", e)
            self._opened = (c0, start)
            return
        log.info("recorder: clock stepped by %.0f s, session %s is now %s", start - w0, os.path.basename(self.dir), new)
        self.dir = new
        self._opened = (c0, start)

    def _frame(self, now, camera_line):
        every = self.cfg["frame_every_s"]
        if not every or not self.frames_enabled or not self.preview_path or now - self._frame_t < every - 0.05:
            return
        self._frame_t = now
        try:
            seq = json.loads(camera_line).get("seq")
        except ValueError:
            seq = None
        dst = os.path.join(self.dir, "frames", "%s.jpg" % (seq if isinstance(seq, int) else int(now * 1000)))
        try:
            shutil.copyfile(self.preview_path, dst)
            self.written["frames"] += 1
        except FileNotFoundError:
            pass                    # no preview yet (camerad --no-preview, or still starting)

    # ---- retention -------------------------------------------------------
    def prune(self):
        """Delete the oldest sessions while the recordings exceed `max_mb` (never the current one)."""
        self._prune_t = self.clock()
        base = self.cfg["dir"]
        try:
            sessions = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        except OSError:
            return
        sizes = {}
        for d in sessions:
            total = 0
            for root, _, files in os.walk(os.path.join(base, d)):
                for fn in files:
                    try:
                        total += os.path.getsize(os.path.join(root, fn))
                    except OSError:
                        pass
            sizes[d] = total
        limit = float(self.cfg["max_mb"]) * 1e6
        cur = os.path.basename(self.dir) if self.dir else None
        for d in sessions:
            if sum(sizes.values()) <= limit:
                break
            if d == cur:
                continue
            shutil.rmtree(os.path.join(base, d), ignore_errors=True)
            log.info("recorder: removed old session %s (%.0f MB)", d, sizes[d] / 1e6)
            sizes[d] = 0

    def status(self):
        return {"dir": self.dir, "failed": self.failed, "frames_enabled": self.frames_enabled, **self.written}
