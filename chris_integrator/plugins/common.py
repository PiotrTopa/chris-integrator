"""Helpers shared by the stack's plug-ins."""
import collections

from ..native import round                  # builtins.round (its results; the native core)

DEFAULT_INTRINSICS = (1500.0, 960.0, 540.0, 1920, 1080)


def intrinsics(calib):
    """(focal_px, cx, cy, width, height) of a camera from its `calib` message (roadsense publishes the camera's
    intrinsics there); nominal values before the first one."""
    c = calib or {}
    w = c.get("width") or DEFAULT_INTRINSICS[3]
    h = c.get("height") or DEFAULT_INTRINSICS[4]
    f = c.get("focal_px") or DEFAULT_INTRINSICS[0]
    return float(f), float(c.get("cx") or w / 2.0), float(c.get("cy") or h / 2.0), int(w), int(h)


class Rates:
    """The marker a measurement topic carries (`rate`): measured rate and latency, and whether the consumer is
    getting less than the configured floor, with the reasons: `shed` (a roadsense instance's load governor has
    capped the stage), `limited` (the platform limiter is cutting clocks), `slow`, `late`. The governor's state
    is an input (stream `governor`); the floors are this API's promise and are configured here.

    A provider that thins itself out on purpose says so: a `lights` message with `idle: {"hz": H, ...}` comes from a
    light detector that tracks nothing and runs a sparser pattern of H passes a second until something shows up.
    While that holds (and for one rate window after the last such message, while the window still holds the idle
    arrivals), the topic's floor is IDLE_TOLERANCE x H instead of the configured one, and the rate is measured
    over the idle arrivals without the bias of a short window (first to last arrival; an arrival overdue by more
    than two announced periods counts up to now). So a stream that keeps the pattern it announces is not `slow`,
    and one that falls behind that pattern is."""
    IDLE_SOURCE = {"lights": "lights_raw"}    # topic -> the provider's stream whose messages may announce idling
    IDLE_TOLERANCE = 0.8                      # share of the announced idle rate a stream must keep
    RATE_WINDOW_S = 2.0
    FLOORS = {"road": {"hz": 10.0, "latency_ms": 250.0}, "lead": {"hz": 10.0, "latency_ms": 250.0},
              "lanes": {"hz": 10.0, "latency_ms": 250.0}, "lights": {"hz": 3.0, "latency_ms": 400.0},
              "signs": {"hz": 3.0, "latency_ms": 400.0}}
    TOPIC_CAPS = {"road": ("road_hz",), "lead": ("road_hz",), "lanes": ("road_hz",), "lights": ("lights_hz",),
                  "signs": ("signs_hz",), "env": ("env_hz",), "view": ("view_hz",)}

    def __init__(self, mixer, floors=None):
        self.mixer = mixer
        self.floors = dict(self.FLOORS, **(floors or {}))
        self._idle = {}                       # topic -> (bus time of the last idle announcement, its hz)

    def governor(self):
        return self.mixer.bus.get("governor") or {}

    def caps(self):
        return self.governor().get("caps") or {}

    def status(self, topic, msg=None, latency_ms=None, hz=None):
        if latency_ms is None and msg is not None:
            ts = msg.get("ts_real_ns")
            latency_ms = (self.mixer.bus.clock.real_ns() - ts) / 1e6 if isinstance(ts, int) and ts > 0 else None
        idle_src = topic in self.IDLE_SOURCE
        idle_hz, floor_hz = self.idle(topic, msg) if idle_src else None, None      # (idle(): None for the others)
        if idle_hz is not None:
            floor_hz = self.IDLE_TOLERANCE * idle_hz
            if hz is None:
                hz = self.idle_rate(topic, idle_hz)
        if hz is None:
            hz = self.mixer.fps(topic) or None
        g = self.mixer.bus.get("governor") or {}
        caps = g.get("caps") or {}
        why = []
        if caps:
            for k in self.TOPIC_CAPS.get(topic, ()):
                if caps.get(k, -1) >= 0:
                    why.append("shed")
                    break
        if g.get("limited"):
            why.append("limited")
        fl = self.floors.get(topic) or {}
        fl_hz = fl.get("hz")
        if fl_hz and floor_hz is not None:
            floor_hz = min(fl["hz"], floor_hz)
        elif fl_hz:
            floor_hz = fl["hz"]
        if hz is not None and floor_hz and hz < floor_hz:
            why.append("slow")
        if latency_ms is not None and fl.get("latency_ms") and latency_ms > fl["latency_ms"]:
            why.append("late")
        out = {"hz": None if hz is None else round(hz, 1),
               "latency_ms": None if latency_ms is None else round(latency_ms, 1),
               # (only "limited" is not a degradation; it is the second entry when there are two or more)
               "degraded": bool(why) and (why[0] != "limited" or len(why) > 1), "why": why,
               "level": g.get("level", 0)}
        if idle_src:
            out["idle_hz"] = None if idle_hz is None else round(idle_hz, 2)
        return out

    def idle(self, topic, msg=None):
        """The idle rate a topic's provider announces (in this message, the provider's newest one, or one within
        the last rate window), else None."""
        src = self.IDLE_SOURCE.get(topic)
        if src is None:
            return None
        m = msg if msg is not None else self.mixer.bus.get(src)
        now = self.mixer.bus.clock.now()
        idle = (m or {}).get("idle")
        hz = idle.get("hz") if isinstance(idle, dict) else None
        if isinstance(hz, (int, float)) and not isinstance(hz, bool) and hz > 0:
            self._idle[topic] = (now, float(hz))
            return float(hz)
        seen = self._idle.get(topic)
        if seen is not None and now - seen[0] <= self.RATE_WINDOW_S:
            return seen[1]                    # the window still holds idle arrivals
        return None

    def idle_rate(self, topic, idle_hz):
        """Messages a second of the provider's stream over max(2 s, 4 announced periods): (n - 1) / (last - first),
        or up to now when the newest arrival is more than two announced periods old (a stall is not hidden)."""
        q = self.mixer.arrivals.get(self.IDLE_SOURCE.get(topic, topic)) or ()
        now = self.mixer.bus.clock.now()
        window = max(self.RATE_WINDOW_S, 4.0 / idle_hz)
        ts = [t for t in q if now - t <= window]
        if len(ts) < 2:
            return 0.0
        end = now if now - ts[-1] > 2.0 / idle_hz else ts[-1]
        span = end - ts[0]
        return (len(ts) - 1) / span if span > 0 else 0.0


class Latest(collections.abc.Mapping):
    """A read-only dict view of declared streams for logic written against a `latest` dict."""

    def __init__(self, ctx, names, unwrap=None):
        self.ctx, self.names, self.unwrap = ctx, tuple(names), unwrap or {}

    def __getitem__(self, k):
        if k not in self.names:
            raise KeyError(k)
        m = self.ctx.get(k)
        if m is None:
            raise KeyError(k)
        return m.get(self.unwrap[k]) if k in self.unwrap else m

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


class Pace:
    """Which radar messages the expensive layers look at (`predict`, `decide`): a service of the mixer both ask
    (`ctx.service("pace")`), so the two skip the same messages. Configuration `pace.json`:

        mode        "full"      every radar message (20 a second with the road model)
                    "fixed"     `idle_hz` a second, whatever happens
                    "adaptive"  every message while something could matter, `idle_hz` a second otherwise
        idle_hz     10.0        the low rate
        hold_s      0.5         full rate is kept this long after the last sign that something could matter

    The low rate: a message is left out only when the one after it (expected one radar period later) still comes
    within `1 / idle_hz` of the last look. With the radar at 20 a second that is every second message; a radar
    that runs at 12 a second is not thinned at all. Adaptive: on top of that a message is looked at at once

    * while the decision layer says something could matter (`report`: an object at level `hot_level` or above
      (entry or exit level), with a collision probability from `hot_p`, a headway under `thw_s`; an alert at a
      stage or on its way to one), and `hold_s` on;
    * when the radar message itself shows a cue: an object the last looked-at message did not have (once), an
      object in the path or a threat by the radar's own corridor test (flags `in_path`, `threat`), a time to reach
      of at most `ttc_s`, a `p_threat` from `p_threat` (then `hold_s` on); a new ground frame.

    What that guarantees: a message is left out only when the look before it found nothing that could matter,
    the message shows no cue, and the next message is due within `1 / idle_hz` (+ 0.01 s) of that look. So whatever
    arises is first judged at most `1 / idle_hz` minus the radar's period later than at full rate (0.05 s at 20
    messages a second and 10 looks; a radar message that comes late or is lost adds its delay, as it does at full
    rate), and from that judgement on every message is judged: the alerts' holds run in time, so a stage is
    entered at most that much later. (The prediction's filters see fewer frames while nothing matters;
    their evidence is weighed in time. What it costs in warnings' times and false warnings: docs/TOPICS.md.)"""
    DEFAULT_CFG = dict(mode="adaptive", idle_hz=10.0, hold_s=0.5, hot_level=1, hot_p=0.01, thw_s=2.0, ttc_s=8.0,
                       p_threat=0.02)

    def __init__(self, cfg=None):
        self.cfg = dict(self.DEFAULT_CFG, **(cfg or {}))
        if self.cfg["mode"] not in ("full", "fixed", "adaptive"):
            self.cfg["mode"] = "full"
        self.reset()

    def reset(self):
        self._ts = None                  # the message last asked about, and the answer
        self._due = True
        self._last = None                # the message last looked at (ts_mono_ns)
        self._prev = None                # the message before this one, looked at or not
        self._hot_until = None           # ns
        self._ids = frozenset()
        self._ground = None
        self.stats = {"messages": 0, "looked": 0, "interval": 0, "hot": 0, "cue": 0, "new": 0}

    def due(self, msg):
        """Is this radar message looked at? The same answer for everyone who asks about one message."""
        ts = msg.get("ts_mono_ns")
        if ts == self._ts:
            return self._due
        c, st = self.cfg, self.stats
        st["messages"] += 1
        self._ts = ts
        mode = c["mode"]
        why = None
        prev, self._prev = self._prev, ts
        if mode == "full" or self._last is None or not isinstance(ts, int) or not isinstance(prev, int):
            why = "interval"
        else:
            # left out only when the next message (expected one radar period on) still comes within the low rate's
            # period of the last look: two looks are never further apart than that period (or one radar period)
            gap = ts - self._last
            if not (0 <= gap and prev <= ts and gap + (ts - prev) <= (1.0 / c["idle_hz"] + 0.01) * 1e9):
                why = "interval"
            elif mode == "adaptive":
                if self._hot_until is not None and ts <= self._hot_until:
                    why = "hot"
                else:
                    why = self._cue(msg, ts)
        self._due = why is not None
        if why is not None:
            st["looked"] += 1
            st[why] += 1
            self._last = ts
            if mode == "adaptive":           # the objects of the message looked at: a later one's new ones are a cue
                self._ids = frozenset([o.get("id") for o in msg.get("objects") or ()])
                self._ground = (msg.get("ego") or {}).get("ground_frame")
        return self._due

    def _cue(self, msg, ts):
        c = self.cfg
        if (msg.get("ego") or {}).get("ground_frame") != self._ground:
            return "new"
        ids, ttc_max, p_min = self._ids, c["ttc_s"], c["p_threat"]
        new = False
        for o in msg.get("objects") or ():
            if o.get("state") == "tentative":
                continue
            fl = o.get("flags")
            ttc, pt = o.get("ttc_s"), o.get("p_threat")
            if (fl and ("in_path" in fl or "threat" in fl)) or (ttc is not None and ttc <= ttc_max) \
                    or (pt is not None and pt >= p_min):
                self._hot_until = ts + int(c["hold_s"] * 1e9)
                return "cue"
            if o.get("id") not in ids:
                new = True
        return "new" if new else None

    def report(self, ts, hot):
        """The decision layer's word on the message it has just judged: could something matter?"""
        if hot and isinstance(ts, int):
            until = ts + int(self.cfg["hold_s"] * 1e9)
            if self._hot_until is None or until > self._hot_until or ts < self._hot_until - 10 * 1e9:
                self._hot_until = until

    def health(self):
        st = self.stats
        return dict(st, mode=self.cfg["mode"], share=round(st["looked"] / st["messages"], 3) if st["messages"] else None)
