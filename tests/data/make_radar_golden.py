#!/usr/bin/env python3
"""Records the radar's frozen results anew, after a deliberate change of its behaviour:

    python3 tests/data/make_radar_golden.py

writes
  tests/data/radar_golden.npz      calls of every native radar function (and of the hypotheses: Hyp.cycle,
                                   Hyp.after, Hyp.drop) sampled on simulated drives, with their results
                                   (tests/test_native.py::test_radar_golden replays them, to rounding: it also
                                   runs where the phone's build is tested)
  tests/data/radar_digests.json    sha256 of the radar's messages on the simulated drives of
                                   tests/test_native.py::RADAR_DIGEST_RUNS (held byte for byte on x86-64)
  chris_integrator/native/_radar_selftest.py   a few results for the self-test on a target
Run it on the development machine (x86-64, glibc) with the native core built (`sh native/build.sh`), and say in
the commit why the results changed."""
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

PER_FN = {"Hyp.cycle": 16, "Hyp.after": 10, "Hyp.drop": 6, "Hyp.history": 6, "radar_frame": 8}
DEFAULT_N = 6
MAX_ROWS = 40                   # a case's state array has at most this many rows in use


class Trace:
    """The native radar module with every radar function (and the hypotheses' scan and drop) recorded."""

    def __init__(self, C, lib):
        self.C, self.lib = C, lib
        self.seen = {}
        self.kept = {}
        self.rng = random.Random(5)

    def __getattr__(self, name):
        v = getattr(self.C, name)
        if name == "Hyp":
            return lambda *a: HypTrace(self, v(*a))
        if name.startswith("radar_") and callable(v):
            return lambda *a: self.call(name, v, None, a)
        return v

    def call(self, name, fn, hyp, args):
        n = self.seen[name] = self.seen.get(name, 0) + 1
        cap = PER_FN.get(name, DEFAULT_N)
        slot = None
        big = any(isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == self.lib.ROW
                  and int(np.any(a != 0.0, axis=1).sum()) > MAX_ROWS for a in args)
        if not big:                                      # a reservoir: every call of the run has the same chance
            keep = self.kept.setdefault(name, [])
            if len(keep) < cap:
                slot = len(keep)
                keep.append(None)
            else:
                k = self.rng.randrange(n)
                slot = k if k < cap else None
        if slot is None:
            return fn(*args)
        codec = self.lib.Codec()

        def wrap(v):                                     # a function among the arguments: what it is asked is recorded
            if isinstance(v, tuple):
                return tuple(wrap(x) for x in v)
            return self.lib.Calls(v, codec) if callable(v) else v
        args = tuple(wrap(a) for a in args)
        case = {"fn": name, "hyp": codec.enc(hyp.dump()) if hyp is not None else None, "args": codec.enc(list(args))}
        before = [a.copy() if isinstance(a, np.ndarray) else None for a in args]
        dicts = [json.dumps(codec.enc(a), sort_keys=True) if isinstance(a, dict) else None for a in args]
        ret = fn(*args)
        case["ret"] = codec.enc(ret)
        case["post"] = {}
        for i, a in enumerate(args):
            if before[i] is not None and not np.array_equal(a, before[i], equal_nan=True):
                if a.ndim == 2 and a.shape[1] == self.lib.ROW:      # the state array: the rows that changed
                    ch = np.flatnonzero(np.any((a != before[i]) & ~((a != a) & (before[i] != before[i])), axis=1))
                    key = "d%d" % len(codec.arrays)
                    codec.arrays[key] = a[ch].copy()
                    codec.arrays[key + "i"] = ch.astype(np.int64)
                    case["post"][str(i)] = {"rows": key}
                else:
                    case["post"][str(i)] = codec.enc(a)
            if dicts[i] is not None and json.dumps(codec.enc(a), sort_keys=True) != dicts[i]:
                case["post"][str(i)] = codec.enc(a)
        case["hyp_after"] = codec.enc(hyp.dump()) if hyp is not None else None
        self.kept[name][slot] = (case, codec.arrays)
        return ret


class HypTrace:
    def __init__(self, tr, H):
        self._tr, self._H = tr, H

    def __getattr__(self, name):
        v = getattr(self._H, name)
        if name in ("drop", "cycle", "after", "history"):
            return lambda *a: self._tr.call("Hyp." + name, v, self._H, a)
        return v


def drives(trace):
    """Simulated drives through the radar with the traced module."""
    import test_radar as TR
    from chris_integrator.radar import box, models, sensors, tracker
    from chris_integrator.sim import radar_score
    saved = (models.NR, tracker._NR, box._NR, sensors._NR, dict(tracker.CFG))
    models.NR = tracker._NR = box._NR = sensors._NR = trace
    tracker.CFG.update(hyp_keep_steps=8)                 # short histories: small cases
    try:
        TR.world(10.0, {"p": (70.0, -4.0, 0.0, 0.0), "q": (64.0, -4.2, 0.0, 0.0), "lead": (30.0, 0.2, 9.0, 0.0)}, 7.0, seed=3,
                 pitch_err_deg=0.4)
        TR.world(14.0, {"lead": (60.0, 0.0, 9.0, 0.0), "onc": (120.0, 3.5, -14.0, 0.0)}, 5.0, seed=9, radar=True)
        TR.world(8.0, {"a": (45.0, -6.0, 0.0, 0.0), "b": (45.0, -3.5, 0.0, 0.0), "c": (40.0, -25.0, 0.0, 7.0)}, 6.0, seed=7,
                 w_ego=0.12, miss=0.2)
        for name in ("crossing_pair", "parked_row"):
            radar_score.run(name, seed=1, cfg={"tracker": {"hyp_keep_steps": 8}})
        radar_score.run("occlusion", seed=1, radar="point", cfg={"tracker": {"hyp_keep_steps": 8}})
        radar_score.run("platoon", seed=1, radar="extended", cfg={"tracker": {"hyp_keep_steps": 8}})
        frame = box.Radar.frame

        def asked(self, ts):                             # the trajectories are asked for now and then (Hyp.history)
            m = frame(self, ts)
            if m is not None and m["seq"] % 9 == 0:
                for o in self.tracker.objects:
                    o.history
            return m
        box.Radar.frame = asked
        try:
            radar_score.run("slow_lead", seed=1, cfg={"tracker": {"hyp_keep_steps": 8}})
        finally:
            box.Radar.frame = frame
    finally:
        models.NR, tracker._NR, box._NR, sensors._NR = saved[:4]
        tracker.CFG.clear()
        tracker.CFG.update(saved[4])


def extra(trace):
    """Functions the drives do not call (tools and tests do): on random inputs."""
    from chris_integrator.radar import tracker as T
    g = np.random.default_rng(3)
    cam = [0.0, 1.3, 6.88, 0.004, 1.0, -0.6, 0.55, 8.0, 80.0, 0.55, 0.4, 14.0, 2.0, 0.1, 1.25, 0.2, 1.15, 0.1666, 0.4997,
           0.575, 1.5, 3.0, 0.7, 0.5, 1e-4, 1e-5, 3e-6, 1e-6, 2.5, 0.45, 7e-5, 1.0]
    mw = [1.0, 2.0, 0.3, 0.05, 0.9, 160.0, 0.9, 1e-4] + [0.0] * 24
    for sen, kind in ((cam, 0), (cam, 1), (cam, 2), (cam, 3), (cam, 10), (mw, 2)):
        x = g.standard_normal(8) * [5, 3, 8, 3, 1, 1, 0.01, 0.1] + [25, 0, 0, 0, 0, 0, 0, 0.4]
        trace.radar_h(sen, kind, 0.4, x, [1.0, -0.5, 0.05, 9.0, 0.4, 0.02], np.zeros(4), np.zeros((4, 8)))
    for it in range(6):
        n, m = int(g.integers(1, 5)), int(g.integers(1, 6))
        L = g.uniform(0, 30, (n, m)) * (g.uniform(size=(n, m)) < 0.6)
        trace.radar_marginals(L, np.zeros((n, m)))
        trace.radar_assign(np.where(L > 1.0, np.log(np.maximum(L, 1e-9)), -np.inf))
        c = g.uniform(0, 10, (n, n + m))
        c[g.uniform(size=c.shape) < 0.2] = 1e17
        trace.radar_kbest(c, 6)
        S = np.zeros((2, T.ROW))
        for r in range(2):
            a = g.standard_normal((3, 8, 8))
            S[r, 0:24] = g.standard_normal(24)
            S[r, 24:216] = np.concatenate([(q @ q.T).ravel() for q in a])
            S[r, 216:219] = (0.2, 0.5, 0.3)
        trace.radar_combine_rows(S, [0, 1])
        S[:, 219:227] = S[:, 0:8]
        S[:, 227:291] = S[:, 24:88]
        trace.radar_group_support(S, [0, 1], (0.3, -0.2, math.cos(0.1 * it), math.sin(0.1 * it)), 0.8, 2.0, 0.01)


def selftest(C):
    """A few results of the motion model, the measurement functions and the marginals, for the self-test."""
    from chris_integrator.radar import models as R, tracker as T
    g = np.random.default_rng(11)
    ev, hh = [], []
    for it in range(3):
        row = np.zeros(T.ROW)
        a = g.standard_normal((4, 8, 8))
        x = g.standard_normal((4, 8)) * [30, 30, 10, 10, 2, 2, 0.01, 0.3]
        for j in range(3):
            row[j * 8:(j + 1) * 8] = x[j]
            row[24 + j * 64:24 + (j + 1) * 64] = (a[j] @ a[j].T + np.eye(8) * 0.1).ravel()
        row[216:219] = (0.2, 0.5, 0.3)
        row[219:227], row[227:291] = x[3], (a[3] @ a[3].T + np.eye(8) * 0.1).ravel()
        S = row[None].copy()
        C.radar_combine_rows(S, [0])
        dt, label = (0.2, 0.05, 1.0)[it], ("car", "human", None)[it]
        x, P, mu = np.empty((1, 8)), np.empty((1, 8, 8)), np.empty((1, 3))
        C.radar_peek(S, [0], [dt], [R._KIN4.get(label) or R._KIN4[None]], R._native_par(R.CFG), x, P, mu)
        ev.append((S[0, :363].tolist(), dt, label, [float(v) for v in (x[0, 0], x[0, 1], P[0, 0, 0], P[0, 0, 1], P[0, 1, 0], P[0, 1, 1])]))
    sens = ([0.0, 1.3, 6.882437470997847, 0.004, 1.0, -0.6, 0.55, 8.0, 80.0, 0.55, 0.4, 14.0, 2.0, 0.1, 1.25, 0.2, 1.15,
             0.1665698348469775, 0.49970950454093244, 0.575, 1.5, 3.0, 0.7, 0.5] + [0.0] * 8,
            [1.0, 2.0, 0.3, 0.05, 0.9, 160.0, 0.9, 1e-4] + [0.0] * 24, [2.0, 1.0, 70.0, 0.85, 1e-4] + [0.0] * 27)
    for sen, kind in ((sens[0], 10), (sens[0], 3), (sens[0], 1), (sens[1], 2), (sens[2], 2)):
        x = (g.standard_normal(8) * [5, 3, 8, 3, 1, 1, 0.01, 0.1] + [25, 0, 0, 0, 0, 0, 0, 0.4]).tolist()
        pose = [1.0, -0.5, 0.05, 9.0, 0.4, 0.02]
        z, H = np.empty(4), np.empty((4, 8))
        m = C.radar_h(sen, kind, 0.4, np.array(x), pose, z, H)
        hh.append((sen, kind, 0.4, x, pose, z[:m].tolist(), H[:m].tolist()))
    L = np.array([[9.0, 0.0, 2.0], [9.0, 30.0, 0.0], [0.0, 0.5, 4.0]])
    out = np.empty(L.shape)
    C.radar_marginals(L, out)
    path = os.path.join(HERE, "chris_integrator", "native", "_radar_selftest.py")
    with open(path, "w") as f:
        f.write('"""A few results of the native core\'s radar part on the development machine (x86-64): the self-test on a\n'
                'target (python -m chris_integrator.native) holds them to rounding. Written by\n'
                'tests/data/make_radar_golden.py; tests/data/radar_golden.npz holds many more, for the tests."""\n'
                "# (the motion part of a state row: xs, Ps, mu, xf, Pf, x, P; dt; label; the place after dt: x, y and its covariance)\n"
                "RADAR_EVOLVE = %r\n# (sensor constants, kind code, scale-rate baseline, x, pose, z, H)\nRADAR_H = %r\n"
                "# (ratio, marginals)\nRADAR_MARGINALS = %r\n" % (ev, hh, (L.tolist(), out.tolist())))
    print("wrote", path)


def main():
    import radar_golden_lib as lib
    from chris_integrator.native import radar as C
    trace = Trace(C, lib)
    drives(trace)
    extra(trace)
    cases, arrays = [], {}
    for name in sorted(trace.kept):
        for item in trace.kept[name]:
            case, arr = item
            pre = "c%d_" % len(cases)

            def ren(v):
                if isinstance(v, list):
                    return [ren(x) for x in v]
                if isinstance(v, dict):
                    return {k: (pre + x if k in ("array", "state", "rows") else ren(x)) for k, x in v.items()}
                return v
            cases.append(ren(case))
            arrays.update({pre + k: a for k, a in arr.items()})
    path = os.path.join(HERE, "tests", "data", "radar_golden.npz")
    lib.save(path, cases, arrays)
    print("wrote %s: %d cases, %.0f kB" % (path, len(cases), os.path.getsize(path) / 1e3))
    print("  " + ", ".join("%s %d of %d" % (k, len(v), trace.seen[k]) for k, v in sorted(trace.kept.items())))
    selftest(C)
    import test_native as TN
    got = TN._radar_digests()
    path = os.path.join(HERE, "tests", "data", "radar_digests.json")
    with open(path, "w") as f:
        json.dump(got, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
