"""The radar's golden data: calls of the native radar functions recorded on simulated drives, with what they
gave (tests/data/radar_golden.npz, written by tests/data/make_radar_golden.py, replayed by
tests/test_native.py::test_radar_golden).

A case: {"fn": name, "hyp": Hyp.dump() before the call (a method of the hypotheses) or None, "args": [...],
"ret": the result, "post": {argument index: the array after the call} for arrays the call wrote into,
"hyp_after": Hyp.dump() after}. The state array travels as its rows that are not all zero; a function among the
arguments (Hyp.cycle asks the tracker for the car's motion noise, a kind's code, a label's numbers, a score bin's
odds) as what it was asked and what it said (`Calls`)."""
import json

import numpy as np

from chris_integrator.radar import tracker as T

ROW = T.ROW


class Calls:
    """A function given to a native call, with what it was asked and what it answered."""

    def __init__(self, fn, codec):
        self.fn, self.codec, self.log = fn, codec, []

    def __call__(self, *a):
        r = self.fn(*a)
        self.log.append([self.codec.enc(list(a)), self.codec.enc(r)])
        return r


class Codec:
    def __init__(self, arrays=None):
        self.arrays = {} if arrays is None else arrays

    def enc(self, v):
        if isinstance(v, Calls):
            return {"calls": v.log}                      # (filled while the call runs)
        if isinstance(v, np.ndarray):
            key = "a%d" % len(self.arrays)
            if v.ndim == 2 and v.shape[1] == ROW and v.dtype == np.float64:
                used = np.flatnonzero(np.any(v != 0.0, axis=1))
                self.arrays[key] = v[used].copy()
                self.arrays[key + "i"] = used.astype(np.int64)
                return {"state": key, "cap": int(v.shape[0])}
            self.arrays[key] = v.copy()
            return {"array": key}
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, tuple):
            return {"tuple": [self.enc(x) for x in v]}
        if isinstance(v, list):
            return [self.enc(x) for x in v]
        if isinstance(v, (set, frozenset)):
            return {"set": sorted(self.enc(x) for x in v)}
        if isinstance(v, dict):
            return {"dict": [[self.enc(k), self.enc(x)] for k, x in v.items()]}
        if isinstance(v, float) and v != v:
            return {"nan": 1}
        if isinstance(v, float) and v in (float("inf"), float("-inf")):
            return {"inf": 1 if v > 0 else -1}
        return v

    def dec(self, v):
        if isinstance(v, list):
            return [self.dec(x) for x in v]
        if isinstance(v, dict):
            if "array" in v:
                return np.array(self.arrays[v["array"]])
            if "state" in v:
                a = np.zeros((v["cap"], ROW))
                a[self.arrays[v["state"] + "i"]] = self.arrays[v["state"]]
                return a
            if "tuple" in v:
                return tuple(self.dec(x) for x in v["tuple"])
            if "set" in v:
                return set(self.dec(x) for x in v["set"])
            if "dict" in v:
                return {self.dec(k): self.dec(x) for k, x in v["dict"]}
            if "calls" in v:
                table = [(self.dec(a), self.dec(r)) for a, r in v["calls"]]

                def answer(*a):
                    for q, r in table:
                        if q == list(a):
                            return r
                    raise KeyError("a recorded function asked something new: %r" % (a,))
                return answer
            if "nan" in v:
                return float("nan")
            if "inf" in v:
                return float("inf") * v["inf"]
        return v


def same(a, b, path="", tol=1e-7):
    """None where a (what the native core gives now) is b (what was recorded): the same structure, texts, ints and
    truth values; floats within tol of their size (and 1e-9 absolutely: what rounding leaves of a number that
    cancels). Else a text that says where they differ."""
    if isinstance(b, np.ndarray) or isinstance(a, np.ndarray):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape:
            return "%s: shape %s, recorded %s" % (path, a.shape, b.shape)
        bad = ~((np.abs(a - b) <= 1e-9 + tol * np.abs(b)) | ((a != a) & (b != b)) | (a == b))
        if bad.any():
            k = np.argwhere(bad)[0]
            return "%s%s: %r, recorded %r (%d of %d differ)" % (path, list(k), a[tuple(k)], b[tuple(k)], bad.sum(), bad.size)
        return None
    if isinstance(b, bool) or b is None or isinstance(b, (str, int)) and not isinstance(a, float):
        return None if (a == b and type(a) is type(b)) or (isinstance(a, int) and isinstance(b, int) and a == b) \
            else "%s: %r, recorded %r" % (path, a, b)
    if isinstance(b, (float, int)):
        if not isinstance(a, (float, int)) or isinstance(a, bool):
            return "%s: %r, recorded %r" % (path, a, b)
        ok = a == b or (a != a and b != b) or abs(a - b) <= 1e-9 + tol * abs(b)
        return None if ok else "%s: %r, recorded %r" % (path, a, b)
    if isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or len(a) != len(b):
            return "%s: %r, recorded %r" % (path, a, b)
        for i, (x, y) in enumerate(zip(a, b)):
            d = same(x, y, "%s[%d]" % (path, i), tol)
            if d:
                return d
        return None
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(map(repr, a)) != set(map(repr, b)):
            return "%s: keys %r, recorded %r" % (path, sorted(map(repr, a)), sorted(map(repr, b)))
        for k in b:
            d = same(a[k], b[k], "%s.%s" % (path, k), tol)
            if d:
                return d
        return None
    if isinstance(b, (set, frozenset)):
        return None if a == b else "%s: %r, recorded %r" % (path, a, b)
    return None if a == b else "%s: %r, recorded %r" % (path, a, b)


def save(path, cases, arrays):
    np.savez_compressed(path, index=np.frombuffer(json.dumps(cases).encode(), np.uint8), **arrays)


def load(path):
    z = np.load(path)
    return json.loads(bytes(z["index"]).decode()), Codec({k: z[k] for k in z.files if k != "index"})


def replay(C, case, codec):
    """The recorded call made again on the native module C: (result, the arguments after it, the hypotheses after
    it or None)."""
    args = codec.dec(case["args"])
    if case.get("hyp") is not None:
        H = C.Hyp(1, 1, 1, 1, 1.0, 1, 1.0, 1)
        H.load(codec.dec(case["hyp"]))
        ret = getattr(H, case["fn"].split(".")[1])(*args)
        return ret, args, H.dump()
    return getattr(C, case["fn"])(*args), args, None
