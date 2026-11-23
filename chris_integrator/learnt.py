"""One store for everything the daemon learns while driving, kept across drives (lane DECIDE, 2026-11-23).

A value learnt on one drive is a prior for the next, never the truth: on start it is handed back with its
uncertainty inflated by its age (and by what a restart can change), within sanity bounds, and only while the
conditions it was learnt under still hold (the camera's calibration in use, its focal length). The learners
(chris_integrator/learners.py) put their values here and take their priors from here through `Learner`.

File (default `<data dir>/learnt.json`, the daemon's state directory; `learnt.json` in `<config dir>/learnt.json`
can set `path`):

    {"format": "chris-learnt", "version": 1, "saved_real_ns": ..., "writes": 17, "sha256": "...",
     "values": {"radar.pitch_bias.cam": {"kind": "gauss", "value": -0.0021, "var": 1.1e-6, "n": 412,
                                         "t_real_ns": ..., "valid": {"calib": "p0.020 y-0.010 h1.22 f975"}}, ...}}

Safe against power loss (the reference phone has no battery: the power goes with the car, at any moment):

* written as write-temp, fsync, rename (the previous file kept as `.prev`), fsync of the directory: a torn write
  leaves the old file or the new one, never a mix;
* read in the order file, temp, previous, each checked (JSON, format, version, the checksum over its values,
  every record's bounds); a file that fails is moved aside as `.bad` once and the next one is used; if none
  holds, the daemon starts with no priors and says so in health;
* written seldom, to spare the flash: when something changed and `min_interval_s` (10 min) has passed, when the
  car has come to a standstill after driving (a stop before the power goes), at shutdown; never more than
  `max_writes_per_hour`.

Reset: `python -m chris_integrator.learnt reset [--path P] [--key PREFIX ...]` (with the daemon stopped), or
`POST /v2/learnt/reset {"keys": [...]}` while it runs (the store forgets them; the learners keep what they hold
now until the next drive starts them anew). `GET /v2/learnt`: the store's health and records."""
import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time

log = logging.getLogger("chris_integrator.learnt")
FORMAT, VERSION = "chris-learnt", 1
DAY = 86400.0

DEFAULT_CFG = dict(
    path=None,                   # default: <data dir>/learnt.json
    min_interval_s=600.0,        # between writes while something changed
    stop_write_s=10.0,           # a standstill this long after driving `stop_after_drive_s` writes (power may go)
    stop_after_drive_s=60.0,
    max_writes_per_hour=12,
    collect_s=10.0,              # how often the learners' values are taken into the store (memory only)
)


class Spec:
    """How one learnt value is kept and given back.

    kind       "gauss" (value, var), "counts" (lists of non-negative counts), "min" (a lower limit that relaxes)
    lo, hi     sanity bounds of the value (each element of a list); a record outside is not used
    var_cap    a prior is never more certain than ... never less certain than this variance (gauss): the
               default the learner starts with
    q_restart  variance added at every start (what a restart, a new drive, can change at once)
    q_per_day  variance added per day of age (random walk); q_per_s the same per second
    half_life_days   counts: halved per this many days of age
    relax_per_day    min: the limit relaxes (grows) by this share per day
    max_age_days     a record older than this is not used
    conditions the names of the conditions the value is valid under (they must match the present ones)"""

    def __init__(self, kind="gauss", lo=-math.inf, hi=math.inf, var_cap=math.inf, q_restart=0.0, q_per_day=0.0,
                 q_per_s=0.0, half_life_days=30.0, relax_per_day=0.05, max_age_days=365.0, conditions=(),
                 length=None, n_cap=None):
        self.kind, self.lo, self.hi, self.var_cap = kind, lo, hi, var_cap
        self.q_restart, self.q_per_day, self.q_per_s = q_restart, q_per_day, q_per_s
        self.half_life_days, self.relax_per_day, self.max_age_days = half_life_days, relax_per_day, max_age_days
        self.conditions, self.length, self.n_cap = tuple(conditions), length, n_cap


class Prior:
    __slots__ = ("value", "var", "n", "age_s", "raw")

    def __init__(self, value, var, n, age_s, raw):
        self.value, self.var, self.n, self.age_s, self.raw = value, var, n, age_s, raw

    def as_dict(self):
        return {"value": self.value, "var": self.var, "n": self.n, "age_s": round(self.age_s, 1)}


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _numbers(v):
    """The value as a flat list of numbers (None kept), or None when it is not numbers."""
    if _fin(v):
        return [float(v)]
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, list):
                sub = _numbers(x)
                if sub is None:
                    return None
                out += sub
            elif x is None or _fin(x):
                out.append(x)
            else:
                return None
        return out
    return None


def checksum(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class LearntStore:
    def __init__(self, path, cfg=None, clock=time.time_ns):
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.path = path
        self.clock = clock                       # real time, ns (the records' ages are on the wall clock)
        self.records = {}
        self.writes = 0
        self.write_times = []                    # monotonic times of this run's writes
        self.dirty = set()
        self.last_write = None                   # monotonic
        self.last_error = None
        self.loaded_from = None
        self.load_problems = []
        self.applied = {}                        # key -> what was given back as a prior (health)
        self.rejected = {}                       # key -> why a record was not used
        self.load()

    # ---- reading --------------------------------------------------------------------------------------------
    def _candidates(self):
        return [(self.path, "file"), (self.path + ".tmp", "temp"), (self.path + ".prev", "previous")]

    def load(self):
        self.records, self.loaded_from, self.load_problems = {}, None, []
        if not self.path:
            return
        for p, what in self._candidates():
            if not os.path.exists(p):
                continue
            try:
                with open(p, "rb") as f:
                    d = json.loads(f.read().decode("utf-8"))
                if not isinstance(d, dict) or d.get("format") != FORMAT:
                    raise ValueError("not a %s file" % FORMAT)
                if d.get("version") != VERSION:
                    raise ValueError("version %r, this daemon reads %d" % (d.get("version"), VERSION))
                vals = d.get("values")
                if not isinstance(vals, dict) or d.get("sha256") != checksum(vals):
                    raise ValueError("checksum does not match (a torn or damaged file)")
            except (OSError, ValueError, UnicodeDecodeError) as e:
                self.load_problems.append("%s: %s" % (what, e))
                if what == "file":
                    try:
                        os.replace(p, p + ".bad")          # kept for a look, out of the way of the next start
                    except OSError:
                        pass
                continue
            for k, r in vals.items():
                v = r.get("value") if isinstance(r, dict) else None
                if isinstance(k, str) and isinstance(r, dict) and (isinstance(v, dict) or _numbers(v) is not None):
                    self.records[k] = r
            self.writes = int(d.get("writes") or 0)
            self.loaded_from = what
            break
        if self.load_problems:
            log.warning("learnt values: %s; %s", "; ".join(self.load_problems),
                        "using the %s file" % self.loaded_from if self.loaded_from else "starting without priors")

    # ---- priors ---------------------------------------------------------------------------------------------
    def prior(self, key, spec, conditions=None):
        """The record as a prior now: Prior or None (why in `rejected`)."""
        r = self.records.get(key)
        if r is None:
            return None
        why = self._check(r, spec, conditions or {})
        if why:
            self.rejected[key] = why
            return None
        age = max(0.0, (self.clock() - int(r.get("t_real_ns") or 0)) / 1e9)
        if age > spec.max_age_days * DAY:
            self.rejected[key] = "older than %g days" % spec.max_age_days
            return None
        v, var, n = r["value"], r.get("var"), r.get("n") or 0
        if spec.kind == "gauss":
            var = (var if _fin(var) and var > 0 else spec.var_cap) + spec.q_restart + spec.q_per_s * age + \
                spec.q_per_day * age / DAY
            var = min(var, spec.var_cap)
        elif spec.kind == "counts":
            f = 0.5 ** (age / (spec.half_life_days * DAY))
            v = [[x * f for x in row] for row in v] if v and isinstance(v[0], list) else [x * f for x in v]
            n = n * f
        elif spec.kind == "min":
            g = 1.0 + spec.relax_per_day * age / DAY
            v = {k: min(spec.hi, x * g) for k, x in v.items()} if isinstance(v, dict) else min(spec.hi, v * g)
        p = Prior(v, var, n, age, r)
        self.applied[key] = p.as_dict()
        self.rejected.pop(key, None)
        return p

    @staticmethod
    def _check(r, spec, conditions):
        if r.get("kind", "gauss") != spec.kind:
            return "kind %s, expected %s" % (r.get("kind"), spec.kind)
        v = r.get("value")
        nums = _numbers(list(v.values()) if isinstance(v, dict) else v)
        if nums is None or not any(x is not None for x in nums):
            return "no value"
        if spec.length is not None and isinstance(v, list) and len(v) != spec.length:
            return "length %d, expected %d" % (len(v), spec.length)
        if any(x is not None and not (spec.lo <= x <= spec.hi) for x in nums):
            return "outside the sanity bounds [%g, %g]" % (spec.lo, spec.hi)
        if spec.kind == "gauss" and r.get("var") is not None and not (_fin(r.get("var")) and r["var"] >= 0):
            return "variance not a number"
        valid = r.get("valid") or {}
        for c in spec.conditions:
            if c in conditions and valid.get(c) != conditions[c]:
                return "learnt under %s %r, now %r" % (c, valid.get(c), conditions[c])
            if c not in conditions:
                return "condition %s not known yet" % c
        return None

    # ---- updates ----------------------------------------------------------------------------------------------
    def put(self, key, spec, value, var=None, n=None, conditions=None):
        """A learner's present value. Out of bounds or not finite: not kept (the old record stays)."""
        nums = _numbers(list(value.values()) if isinstance(value, dict) else value)
        if nums is None or any(x is not None and not (spec.lo <= x <= spec.hi) for x in nums):
            return False
        if spec.kind == "gauss" and var is not None and not (_fin(var) and var >= 0):
            return False
        old = self.records.get(key)
        rec = {"kind": spec.kind, "value": value, "var": var, "n": n, "t_real_ns": self.clock(),
               "valid": {c: (conditions or {}).get(c) for c in spec.conditions}}
        if old is not None and {k: old.get(k) for k in ("value", "var", "n", "valid")} == \
                {k: rec[k] for k in ("value", "var", "n", "valid")}:
            return False
        self.records[key] = rec
        self.dirty.add(key)
        return True

    def reset(self, keys=None):
        """Forget the records (all, or those whose key starts with one of `keys`); written at once."""
        gone = [k for k in self.records if not keys or any(k.startswith(p) for p in keys)]
        for k in gone:
            del self.records[k]
            self.applied.pop(k, None)
        self.dirty.add("__reset__")
        self.save(force=True)
        return gone

    # ---- writing ---------------------------------------------------------------------------------------------
    def due(self, now, why=None):
        """Whether a write may happen now (monotonic s): something changed, and the rate allows it."""
        c = self.cfg
        if not self.dirty or not self.path:
            return False
        self.write_times = [t for t in self.write_times if now - t < 3600.0]
        if len(self.write_times) >= c["max_writes_per_hour"]:
            return False
        if why in ("stop", "shutdown"):
            return self.last_write is None or now - self.last_write >= 60.0 or why == "shutdown"
        return self.last_write is None or now - self.last_write >= c["min_interval_s"]

    def save(self, force=False, now=None):
        if not self.path or (not force and not self.dirty):
            return False
        now = time.monotonic() if now is None else now
        vals = dict(self.records)
        d = {"format": FORMAT, "version": VERSION, "saved_real_ns": self.clock(), "writes": self.writes + 1,
             "sha256": checksum(vals), "values": vals}
        data = json.dumps(d, separators=(",", ":"), sort_keys=True).encode()
        tmp, prev = self.path + ".tmp", self.path + ".prev"
        try:
            dname = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(dname, exist_ok=True)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
            if os.path.exists(self.path):
                os.replace(self.path, prev)
            os.replace(tmp, self.path)
            try:
                dfd = os.open(dname, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
        except OSError as e:
            self.last_error = str(e)
            log.warning("learnt values not written: %s", e)
            return False
        self.writes += 1
        self.write_times.append(now)
        self.last_write = now
        self.dirty = set()
        self.last_error = None
        return True

    def health(self):
        bad = [p for p in self.load_problems]
        return {"state": "degraded" if (bad and not self.loaded_from) or self.last_error else "ok",
                "reason": ("; ".join(bad) if bad else None) or self.last_error,
                "path": self.path, "loaded_from": self.loaded_from, "records": len(self.records),
                "dirty": sorted(self.dirty), "writes": self.writes, "writes_this_run": len(self.write_times),
                "last_write_age_s": None if self.last_write is None else round(time.monotonic() - self.last_write, 1),
                "priors": dict(self.applied), "rejected": dict(self.rejected)}

    def describe(self):
        now = self.clock()
        return {k: {"value": r.get("value") if not isinstance(r.get("value"), list) or len(r["value"]) <= 4 else
                    "[%d values]" % len(r["value"]), "var": r.get("var"), "n": r.get("n"),
                    "age_s": round((now - int(r.get("t_real_ns") or 0)) / 1e9, 1), "valid": r.get("valid")}
                for k, r in sorted(self.records.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("show", "reset"))
    ap.add_argument("--path", default=os.path.join(os.environ.get("CHRIS_DATADIR", "/var/lib/chris"), "learnt.json"))
    ap.add_argument("--key", action="append", default=None, help="a key prefix (reset: only these)")
    a = ap.parse_args(argv)
    st = LearntStore(a.path)
    if a.cmd == "show":
        print(json.dumps({"health": st.health(), "records": st.describe()}, indent=1, default=str))
        return 0
    gone = st.reset(a.key)
    print("forgot %d learnt values in %s" % (len(gone), a.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
