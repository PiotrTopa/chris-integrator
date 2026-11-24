"""Fault detection as a general capability (docs/DESIGN.md section 5a): detectors that apply to every stream of
every source without knowing what the stream is, and the findings they publish.

Three families, each seeing what the others cannot:

* **signal integrity** of one raw stream by itself (`Stream`): a value that does not change (stuck), samples that
  repeat (sample-and-hold, a stale relay), noise far above the characterised level (burst), isolated wild
  samples (outliers), values that are not numbers. No second source is needed, so the culprit is known: these
  are *self-evident* and isolate a source whatever its rank.
* **timing integrity** of one message stream (`Clock`): its stamps against the other sources' stamps at arrival
  (a stamp offset, a latency jump), its rate, stamps that repeat or go back.
* **consistency over time** of the filter's innovations (`Trend`): a signed innovation that keeps one sign for
  long (bias, slow drift, a calibration that moved), where the windowed chi-square monitors see nothing yet.

What needs a second source (bias, scale, sign, unit, delay of the content) is the pair monitors' business
(`monitors.py`); `describe_pair` names the kind of disagreement from the pair's own samples.

A finding: {"source", "quantity", "kind", "level": "info" | "warn" | "fault", "since_s", "text"}.
"""
import collections
import math
import statistics

from ..native import core as _C
from ..native import round                  # builtins.round (its results; the native core)

DEFAULT_CFG = dict(
    stuck_s=1.5, stuck_n=10,          # identical non-zero values for this long and this many samples: stuck
    stuck_fault_s=2.5,                # ... and a fault after this long
    learn_n=60, learn_repeat=0.2,     # ... for a stream that has shown, over this many samples, that it does not repeat
    repeat_window=40, repeat_share=0.45,   # share of samples equal to their predecessor (while the value is not zero)
    noise_window=40, noise_ratio=4.0, noise_fault_ratio=25.0,   # robust sigma of the second difference / characterised sigma
    outlier_z=8.0, outlier_share=0.04,
    clear_s=4.0,                      # a finding ends after this long without its condition
    clock_base_s=20.0, clock_window=15, clock_jump_s=0.2,
    # (fire level and the correlation allowance: quiet on the reference car's healthy drives, where innovations
    # of one class stay correlated for seconds; set on that thin data)
    trend_tau_s=20.0, trend_fire=6.0, trend_clear=3.0, trend_min_n=60, trend_corr=10.0,
)

UNITS = ((3.6, "km/h read as m/s"), (1.0 / 3.6, "m/s read as km/h"), (2.237, "mph read as m/s"), (57.2958, "deg read as rad"),
         (1.0 / 57.2958, "rad read as deg"), (1.609, "mph read as km/h"), (0.6214, "km/h read as mph"))


class Finding:
    __slots__ = ("source", "quantity", "kind", "level", "since", "last", "text")

    def __init__(self, source, quantity, kind, level, t, text):
        self.source, self.quantity, self.kind, self.level, self.since, self.last, self.text = source, quantity, kind, level, t, t, text

    def as_dict(self, now):
        return {"source": self.source, "quantity": self.quantity, "kind": self.kind, "level": self.level,
                "since_s": round(max(0.0, now - self.since), 1), "text": self.text}


class Stream(_C.StreamState):
    """Signal integrity of one raw stream (group, quantity): a value frozen while it should vary, samples that
    repeat their predecessor, noise beyond the characterised level, wild samples, values that are not numbers. The
    arithmetic is the native core's (chris_native.c `StreamState`); the findings' texts are made here."""

    def __init__(self, cfg):
        super().__init__(cfg["stuck_s"], cfg["stuck_n"], cfg["stuck_fault_s"], cfg["learn_n"], cfg["learn_repeat"],
                         cfg["repeat_window"], cfg["repeat_share"], cfg["noise_window"], cfg["noise_ratio"],
                         cfg["outlier_z"])
        self.c = cfg
        self.held = []

    def offer(self, t, v, s):
        """-> [(kind, level, text)] conditions that hold now."""
        if not (isinstance(v, (int, float)) and math.isfinite(v)):
            return [("invalid", "fault", "value is not a number")]
        r = super().offer(t, v, s)
        if r is None:
            return []
        if r == 1:
            return list(self.held)
        due, stuck, rep, noise = r
        c, out = self.c, []
        if stuck is not None:
            out.append(("stuck", "fault" if stuck[0] else "warn",
                        "value frozen at %.4g for %.1f s (%d samples)" % (v, stuck[1], stuck[2])))
        if not due:
            return out + self.held
        self.held = []
        out0 = len(out)
        if rep is not None and not out:
            out.append(("repeat", "warn", "%d %% of samples repeat their predecessor" % round(100.0 * rep[0] / rep[1])))
        if noise is not None:
            ratio, wild, nd = noise
            if ratio >= c["noise_ratio"]:
                out.append(("noise", "fault" if ratio >= c["noise_fault_ratio"] else "warn",
                            "noise %.1fx the characterised level" % ratio))
            elif wild / nd >= c["outlier_share"] and wild >= 3:
                out.append(("outliers", "warn", "%d wild samples in the last %d" % (wild, nd)))
        self.held = out[out0:]
        return out


def _median(data):
    """statistics.median of a non-empty collection (its arithmetic, without its generic overhead)."""
    data = sorted(data)
    n = len(data)
    if n % 2 == 1:
        return data[n // 2]
    i = n // 2
    return (data[i - 1] + data[i]) / 2


class Clock:
    """Timing integrity of one message stream: how far its stamp is ahead of the newest stamp of any *other*
    stream when it arrives. Normally a constant of the stream (its latency); a step in it is a stamp offset or a
    latency jump of this stream (when the others' offsets stay) or of the device clock (when all move)."""

    def __init__(self, cfg):
        self.c = cfg
        self.base = None
        self.t0 = None
        self.early = []
        self.win = collections.deque(maxlen=cfg["clock_window"])
        self.shift = 0.0

    def offer(self, t, others_newest):
        if others_newest is None:
            return None
        d = t - others_newest
        self.win.append(d)
        if self.t0 is None:
            self.t0 = t
        if self.base is None:
            self.early.append(d)
            if t - self.t0 >= self.c["clock_base_s"] and len(self.early) >= self.c["clock_window"]:
                self.base = statistics.median(self.early)
            return None
        if len(self.win) < self.win.maxlen:
            return None
        med = _median(self.win)
        self.shift = med - self.base
        if abs(self.shift) < 0.5 * self.c["clock_jump_s"]:
            self.base += 0.02 * (med - self.base)           # the latency may creep; a step may not
        return self.shift

    def clear(self):
        self.win.clear()
        self.base = self.t0 = None
        self.early = []
        self.shift = 0.0


def Trend(cfg):
    """A signed, normalised innovation averaged over time (exponential window): |mean| x sqrt(effective n) is
    about N(0, 1) for a consistent row class, and grows without bound for a bias the filter's states do not
    explain (a slow drift, a calibration that moved). The native core's (chris_native.c `Trend`); the filter
    pass offers every row's first innovation to it directly."""
    return _C.Trend(cfg["trend_tau_s"], cfg["trend_fire"], cfg["trend_clear"], cfg["trend_min_n"], cfg["trend_corr"])



def describe_pair(samples):
    """What kind of disagreement a pair shows, from its (tested, reference) samples: (kind, text)."""
    pts = [(v, r) for v, r in samples if abs(r) > 1e-6]
    if len(pts) < 5:
        return None, None
    ratio = statistics.median(v / r for v, r in pts)
    off = statistics.median(v - r for v, r in pts)
    span = max(r for _, r in pts) - min(r for _, r in pts)
    mid = statistics.median(abs(r) for _, r in pts)
    if ratio < -0.5:
        return "sign", "opposite sign (ratio %.2f)" % ratio
    for k, name in UNITS:
        if abs(ratio / k - 1.0) < 0.04:
            return "unit", "ratio %.3g: %s?" % (ratio, name)
    # a scale error keeps the ratio over the range of values, a bias keeps the difference
    rs = [v / r for v, r in pts]
    ds = [v - r for v, r in pts]
    spread_r = _mad(rs) * mid
    spread_d = _mad(ds)
    if span > 0.3 * mid and spread_d < spread_r:
        return "bias", "offset %+.3g" % off
    if abs(ratio - 1.0) > 0.03:
        return "scale", "ratio %.2f" % ratio
    return "bias", "offset %+.3g" % off


def _mad(x):
    m = statistics.median(x)
    return statistics.median(abs(v - m) for v in x)


class Diagnostics:
    """All detectors of an integrator, and the list of findings."""

    def __init__(self, cfg=None):
        self.c = dict(DEFAULT_CFG, **(cfg or {}))
        self.streams, self.clocks, self.trends = {}, {}, {}
        self.findings = {}                 # (source, quantity, kind) -> Finding
        self.newest = {}                   # stream key -> newest stamp
        self.now = 0.0
        self.events = collections.deque(maxlen=16)      # one-off reports (a mount step, an epoch)
        self.qfaults = None                # the integrator's quantity-level isolation, for the report
        self.reported = {}                 # verdicts of detectors in `report` mode: (source, quantity, detector) -> {"since", "text"}

    # ---- inputs ----------------------------------------------------------------------------------------
    def raw(self, group, quantity, t, v, s):
        """A raw measurement of a source (before any fusion). True when a fault-level finding began or ended."""
        st = self.streams.get((group, quantity))
        if st is None:
            st = self.streams[(group, quantity)] = Stream(self.c)
        self.now = max(self.now, t)
        changed = False
        seen = set()
        for kind, level, text in st.offer(t, v, s):
            seen.add(kind)
            changed |= self._set(group, quantity, kind, level, t, text)
        for key in [k for k in self.findings if k[0] == group and k[1] == quantity and k[2] in ("stuck", "repeat", "noise", "outliers", "invalid")]:
            f = self.findings[key]
            if key[2] not in seen and t - f.last > self.c["clear_s"]:
                changed |= f.level == "fault"
                del self.findings[key]
        return changed

    def stamp(self, key, group, t):
        """A message of stream `key` stamped t has arrived."""
        others = [v for k, v in self.newest.items() if k != key]
        ck = self.clocks.get(key)
        if ck is None:
            ck = self.clocks[key] = Clock(self.c)
        shift = ck.offer(t, max(others) if others else None)
        self.newest[key] = t
        self.now = max(self.now, t)
        fk = (group, "time", "stamp")
        if shift is not None and abs(shift) >= self.c["clock_jump_s"]:
            # a step of every stream at once is the others' (or the clock's); blame the stream only when most of
            # the other streams did not move the opposite way
            moved = [c.shift for k, c in self.clocks.items() if k != key and c.base is not None]
            if not moved or statistics.median(moved) * shift > -0.25 * shift * shift:
                self._set(group, "time", "stamp", "warn", t,
                          "stamps %.2f s %s than before against the other sources (stamp offset or latency jump)" % (abs(shift), "later" if shift > 0 else "earlier"))
        elif fk in self.findings and t - self.findings[fk].last > self.c["clear_s"]:
            del self.findings[fk]

    def trend_new(self, key):
        """The trend of a row class (group, quantity), made and entered (the filter pass asks for it)."""
        tr = self.trends[key] = Trend(self.c)
        return tr

    def trend_note(self, key, tr, t):
        """A row class's trend holds (its finding is set or renewed), or has just ended (its finding goes)."""
        group, quantity = key
        if tr.high:
            self._set(group, quantity, "trend", "warn", t,
                      "innovations keep one sign (%+.1f sigma of the mean): a bias or drift the model does not explain" % tr.score)
        else:
            self.findings.pop((group, quantity, "trend"), None)

    def event(self, source, kind, t, text, **more):
        self.events.append(dict(source=source, kind=kind, t=t, text=text, **more))

    def _set(self, source, quantity, kind, level, t, text):
        key = (source, quantity, kind)
        f = self.findings.get(key)
        if f is None:
            self.findings[key] = Finding(source, quantity, kind, level, t, text)
            return level == "fault"
        ch = f.level != level and "fault" in (f.level, level)
        f.level, f.last, f.text = level, t, text
        return ch

    # ---- queries ---------------------------------------------------------------------------------------
    def self_evident(self, group):
        """Fault-level findings a source shows by itself: [(quantity, kind, text)]."""
        return [(f.quantity, f.kind, "%s: %s" % (f.kind, f.text)) for f in self.findings.values() if f.source == group and f.level == "fault"]

    def noise_scale(self, group, quantity):
        """Factor on a stream's stated sigma while its measured noise is above the characterised level: the
        stream stays in the estimate with the weight its present noise deserves (adaptive measurement noise)."""
        if (group, quantity, "noise") not in self.findings:
            return 1.0
        return max(1.0, min(self.streams[(group, quantity)].noise_ratio, 30.0))

    def suspected(self, group):
        """True while a source shows any signal-integrity condition of its own (also below fault level)."""
        return any(f.source == group and f.kind in ("stuck", "repeat", "noise", "outliers", "invalid") for f in self.findings.values())

    def report(self, now=None):
        now = self.now if now is None else now
        out = [f.as_dict(now) for f in self.findings.values()]
        out += [{"source": g, "quantity": q, "kind": "isolated", "level": "fault", "since_s": round(max(0.0, now - f["since"]), 1),
                 "text": "%s of %s set aside (%s): %s" % (q, g, f["rule"], f["reason"])} for (g, q), f in (self.qfaults or {}).items()]
        out += [{"source": g, "quantity": q, "kind": "reported", "detector": det, "level": "warn", "mode": "report",
                 "since_s": round(max(0.0, now - r["since"]), 1),
                 "text": "%s (detector %s runs in report mode: nothing is set aside)" % (r["text"], det)}
                for (g, q, det), r in self.reported.items()]
        for e in self.events:
            if e["t"] is None:
                e["t"] = now                              # reported for the first time
            out.append({"source": e["source"], "quantity": e.get("quantity", "mount"), "kind": e["kind"], "level": "info",
                        "since_s": round(max(0.0, now - e["t"]), 1), "text": e["text"]})
        return out

    def reset(self):
        """A new epoch: forget the windows (they are tied to the old time axis); findings end."""
        for d in (self.streams, self.clocks, self.trends):
            for x in d.values():
                x.clear()
        self.findings.clear()
        self.newest.clear()
