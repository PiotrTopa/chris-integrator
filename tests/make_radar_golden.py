"""Freeze what the radar's Python reference gives, at the boundaries of the native functions, into
tests/data/radar_golden.npz (read by tests/test_native.py `test_radar_golden`).

Simulated drives run with the native core; at chosen calls of each native radar function the call's inputs are
kept (the objects' rows it reads, compacted) together with what the Python reference computes for the same
state: Tracker._associate for radar_scan, Tracker._update_numbers for radar_update, Radar._numbers for
radar_frame, and so on. Plus random and edge cases for the measurement functions, marginals and assignment.

This needs the Python reference, which exists up to the commit that adds this file; it is kept with the data
it made and goes with the reference.

    python3 tests/make_radar_golden.py            (writes tests/data/radar_golden.npz)
"""
import copy
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from chris_integrator import native as N  # noqa: E402
from chris_integrator.radar import box as Bx, models as M, sensors as Sn, tracker as T  # noqa: E402

C = N.radar
assert C is not None, "the native radar part must be in use to freeze the reference against its calls"
QUOTA = {"radar_scan": 8, "radar_update": 6, "radar_advance": 4, "radar_peek_pos": 4, "radar_life": 4,
         "radar_merge": 5, "radar_frame": 5, "radar_check": 8, "radar_register": 3, "radar_common": 4}
MAXROWS = {"radar_scan": 6, "radar_update": 3, "radar_advance": 3, "radar_peek_pos": 4, "radar_life": 4,
           "radar_merge": 5, "radar_frame": 3, "radar_register": 5, "radar_common": 5, "radar_check": 1}
cases, arrays = [], {}
taken = {k: 0 for k in QUOTA}


def reference(fn):
    saved = (T._NR, Sn._NR, Bx._NR)
    T._NR = Sn._NR = Bx._NR = None
    try:
        return fn()
    finally:
        T._NR, Sn._NR, Bx._NR = saved


def arr(a):
    """An array of the file: its name."""
    name = "a%d" % len(arrays)
    arrays[name] = np.array(a, float, copy=True, order="C")          # a copy: a call's arrays change afterwards
    return {"array": name}


def enc(v):
    if isinstance(v, np.ndarray):
        return arr(v)
    if isinstance(v, (list, tuple)):
        if v and all(isinstance(x, np.ndarray) for x in v) and len({x.shape for x in v}) == 1:
            return arr(np.array(v))
        return [enc(x) for x in v]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def compact(S, rows):
    """The rows a call reads, as a state array of their own, and the call's rows renumbered."""
    order = sorted(set(rows))
    new = {r: k for k, r in enumerate(order)}
    return S[order].copy(), [new[r] for r in rows]


def clone_objs(objs):
    """Copies of objects on a state array of their own (the reference changes them in place)."""
    st = T.State(max(len(objs), 1))
    st.free = []
    out = []
    for k, o in enumerate(objs):
        q = T.Obj.__new__(T.Obj)
        for a in T.Obj.__slots__:
            if a.startswith("_"):
                continue
            v = getattr(o, a)
            setattr(q, a, copy.deepcopy(v) if isinstance(v, (dict, set, list)) or hasattr(v, "maxlen") else v)
        q.slot = k
        q.bind(st)
        out.append(q)
    for q, o in zip(out, objs):
        for name in ("xs", "Ps", "mu", "xf", "Pf", "x", "P"):
            getattr(q, name)[...] = getattr(o, name)
    return st, out


def keep(fn, args, ref, S=None, rows=None, extra=None):
    case = {"fn": fn, "args": enc(list(args)), "ref": enc(ref)}
    if S is not None:
        case["rows"] = arr(S)
    if extra:
        case.update(extra)
    cases.append(case)
    taken[fn] += 1


class Recorder:
    """Stands for the native module in tracker.py and box.py: every call goes to the native function; chosen
    ones are kept with the reference's answer for the same state."""

    def __init__(self, tr, box):
        self.tr, self.box = tr, box
        self.cap = {}

    def __getattr__(self, name):
        return getattr(C, name)

    def want(self, fn, rows):
        return taken[fn] < QUOTA[fn] and 0 < len(set(rows)) <= MAXROWS[fn] and np.random.random() < 0.25

    def objs(self, rows):
        by = {o.slot: o for o in self.tr.objects}
        return [by[r] for r in rows]

    def radar_scan(self, S, rows, sens, pose, oi, D, mem, nscan, cfg, pd, cov, r1, r2, r3):
        if not self.want("radar_scan", rows) or "dets" not in self.cap:
            return C.radar_scan(S, rows, sens, pose, oi, D, mem, nscan, cfg, pd, cov, r1, r2, r3)
        cp = self.cap
        objs = self.objs(rows)
        ref = reference(lambda: self.tr._associate(cp["src"], cp["t"], cp["sensor"], pose, objs, copy.deepcopy(cp["dets"])))
        S2, rows2 = compact(S, rows)
        keep("radar_scan", (None, rows2, sens, pose, oi, D, mem, nscan, cfg), ([list(p) for p in ref[0]], list(ref[1]), list(ref[2])), S2)
        return C.radar_scan(S, rows, sens, pose, oi, D, mem, nscan, cfg, pd, cov, r1, r2, r3)

    def radar_update(self, S, rows, sens, pose, Dsel, PP, cfg):
        cp = self.cap
        if self.want("radar_update", rows) and "pairs" in cp:
            objs = self.objs(rows)
            st, cl = clone_objs(objs)
            res = []
            for q, ((i, j, g, pr, lam), o) in enumerate(zip(cp["pairs"], cl)):
                d = cp["live"][j]                      # as _common left it
                m = len(d["z"])
                R = np.diag(np.asarray(Dsel[q, 9:9 + m], float))
                res.append(reference(lambda: self.tr._update_numbers(o, d, R, cp["sensor"], pose, PP[q][0], PP[q][1], PP[q][2])))
            S2, rows2 = compact(S, rows)
            keep("radar_update", (None, rows2, sens, pose, Dsel, PP, cfg), {"out": res, "after": arr(st.a[[o.slot for o in cl]])}, S2)
        return C.radar_update(S, rows, sens, pose, Dsel, PP, cfg)

    def radar_advance(self, S, rows, dt, kin, par, noise, pose):
        if self.want("radar_advance", rows):
            objs = self.objs(rows)
            st, cl = clone_objs(objs)
            t = objs[0].t + dt[0]
            reference(lambda: self.tr._advance_all(cl, t))
            S2, rows2 = compact(S, rows)
            keep("radar_advance", (None, rows2, dt, kin, par, noise, pose), {"after": arr(st.a[[o.slot for o in cl]])}, S2)
        return C.radar_advance(S, rows, dt, kin, par, noise, pose)

    def radar_peek_pos(self, S, rows, dt, kin, par, out):
        if self.want("radar_peek_pos", rows):
            objs = self.objs(rows)
            ref = reference(lambda: self.tr.peek_many(objs, [o.t + d for o, d in zip(objs, dt)]))
            S2, rows2 = compact(S, rows)
            keep("radar_peek_pos", (None, rows2, dt, kin, par),
                 [[float(x[0]), float(x[1]), float(P[0, 0]), float(P[0, 1]), float(P[1, 0]), float(P[1, 1])] for x, P, _ in ref], S2)
        return C.radar_peek_pos(S, rows, dt, kin, par, out)

    def radar_life(self, S, rows, v0, away, dt, kin, par, pose, sd2):
        if self.want("radar_life", rows) and len(rows) == len(self.tr.objects):
            tr = self.tr
            T_ = tr.objects[0].t + dt[0]
            scan = T_ if tr.t is None else min(T_, tr.t)
            ref = reference(lambda: tr._life(T_, None if pose is None else tuple(pose) + (0.0,), scan))
            S2, rows2 = compact(S, rows)
            keep("radar_life", (None, rows2, v0, away, dt, kin, par, pose, sd2), [list(r) for r in ref], S2)
        return C.radar_life(S, rows, v0, away, dt, kin, par, pose, sd2)

    def radar_merge(self, S, rows, times, kin, par, here, ego, cfg):
        if self.want("radar_merge", rows) and len(rows) == len(self.tr.objects):
            ref = reference(lambda: self.tr._merge_geometry(self.tr.objects, tuple(ego) + (0.0,)))
            if ref:
                S2, rows2 = compact(S, rows)
                keep("radar_merge", (None, rows2, times, kin, par, here, ego, cfg), [list(r) for r in ref], S2)
        return C.radar_merge(S, rows, times, kin, par, here, ego, cfg)

    def radar_frame(self, S, rows, dt, kin, par, pose, ego, cfg, hz, px, py, lst, v0, *rest):
        if self.want("radar_frame", rows) and rest:
            objs = self.objs(rows)
            T_ = rest[3]
            ref = reference(lambda: self.box._numbers(T_, objs, pose[0], pose[1], pose[2], ego[0], ego[1]))
            S2, rows2 = compact(S, rows)
            keep("radar_frame", (None, rows2, dt, kin, par, pose, ego, cfg, hz, px, py, lst, v0), [list(r) for r in ref], S2)
        return C.radar_frame(S, rows, dt, kin, par, pose, ego, cfg, hz, px, py, lst, v0, *rest)

    def radar_check(self, S, slot, back, px, py, Pp):
        if self.want("radar_check", [slot]):
            o = self.objs([slot])[0]
            ref = reference(lambda: T.Tracker._check_numbers(o, back, px, py, Pp))
            S2, rows2 = compact(S, [slot])
            keep("radar_check", (None, rows2[0], back, px, py, list(Pp)), None if ref is None else list(ref), S2)
        return C.radar_check(S, slot, back, px, py, Pp)

    def radar_register(self, S, ripe, dpv, rows, a):
        if self.want("radar_register", list(ripe) + list(rows)):
            allr = sorted(set(ripe) | set(rows))
            objs = self.objs(allr)
            st, cl = clone_objs(objs)
            by = dict(zip(allr, cl))
            ref = reference(lambda: T.Tracker._register_numbers([by[r] for r in ripe], dpv, [by[r] for r in rows], a))
            S2, idx = compact(S, list(ripe) + list(rows))
            keep("radar_register", (None, idx[:len(ripe)], dpv, idx[len(ripe):], a),
                 {"out": list(ref), "after": arr(st.a[[by[r].slot for r in allr]])}, S2)
        return C.radar_register(S, ripe, dpv, rows, a)

    def radar_common(self, S, rows, sens, pose, items):
        if self.want("radar_common", rows) and "sensor" in self.cap:
            objs = self.objs(rows)
            names = {0: "b", 1: "rb", 2: "rbh", 3: "bt", 4: ""}
            sel = []
            for o, it in zip(objs, items):
                k = names[int(it[0]) & 7] + ("s" if int(it[0]) & 8 else "")
                sel.append((o, {"kind": k, "sdt": it[1], "z": [it[2]]}, [it[3]]))
            sen = self.cap["sensor"]
            ref = reference(lambda: T.Tracker._common_numbers(sen, pose, sel))
            S2, rows2 = compact(S, rows)
            keep("radar_common", (None, rows2, sens, pose, items), list(ref), S2)
        return C.radar_common(S, rows, sens, pose, items)


def drive(name, seed, night, scale):
    from chris_integrator.mixer import MessageClock
    from chris_integrator.sim import default_sensors
    from chris_integrator.sim import roaduser_sim as rs
    from chris_integrator.stack import build
    SC = {"px": 0.08, "rel": 0.002, "p": 0.78, "min_dt": 0.2, "max_dt": 1.0, "max_std": 0.06}
    w = rs.scenario(name, seed=seed, night=night)
    det = rs.DetectorSim(seed=1000 + seed, night=night, params={"scale_rate": SC} if scale else None)
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
    sensors.append(det)
    st = build(cfg={"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}}, mapper=None, clock=MessageClock())
    box = st.mixer.slots["radar"].plugin.box
    hook(box)
    sent = False
    for _, m in w.messages(sensors):
        if not sent:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            sent = True
        st.deliver(m)


def hook(box):
    tr = box.tracker
    rec = Recorder(tr, box)
    process, assoc = tr.process, tr._associate_native

    def proc(src, t, sensor, dets):
        rec.cap = {"src": src, "t": t, "sensor": sensor, "dets": copy.deepcopy(dets), "live": dets}
        try:
            return process(src, t, sensor, dets)
        finally:
            rec.cap = {}

    def asc(src, t, sensor, pose, objs, D):
        r = assoc(src, t, sensor, pose, objs, D)
        rec.cap["pairs"] = r[0]
        return r

    tr.process, tr._associate_native = proc, asc
    T._NR = Bx._NR = rec


def microwave():
    from chris_integrator.radar import Radar
    g = np.random.default_rng(14)
    rd = Radar()
    hook(rd)
    NS = 1_000_000_000
    targets = [(30.0, 3.5, 0.0, 0.0), (55.0, -0.2, 18.0, 0.0), (12.0, -3.0, 0.0, 0.0)]
    for k in range(200):
        ts = 10 * NS + k * NS // 20
        rd.on_ego({"ts_mono_ns": ts, "speed_mps": 15.0, "yaw_rate_rps": 0.01, "accel_long_mps2": 0.0})
        if k % 2:
            continue
        t = (ts - 10 * NS) / NS
        objs = []
        for (x0, y0, vx, vy) in targets:
            x, y = x0 + (vx - 15.0) * t, y0 + vy * t
            r = math.hypot(x, y)
            if g.random() < 0.85:
                objs.append({"range_m": r + g.normal() * 0.2, "azimuth_deg": math.degrees(math.atan2(y, x)) + g.normal() * 0.3,
                             "range_rate_mps": (x * (vx - 15.0) + y * vy) / r + g.normal() * 0.1, "class": "car"})
        rd.on_objects({"ts_mono_ns": ts, "sensor": "front_radar", "kind": "radar", "objects": objs,
                       "mount": {"x_m": 2.0, "y_m": 0.0, "yaw_deg": 0.0}})
        rd.frame(ts)


def random_cases():
    """The measurement functions (every kind), marginals and assignment on random and edge inputs."""
    from tests.test_native import _cameras
    from chris_integrator.radar.sensors import MicrowaveRadar, kind_code
    g = np.random.default_rng(21)
    kinds = ["b", "rb", "rbh", "bt", "bs", "rbs", "rbhs", "bts", "s"]
    for it in range(63):
        s = _cameras(g)
        x = g.standard_normal(8) * [30, 30, 10, 10, 2, 2, 0.01, 0.3]
        if it % 9 == 0:
            x[:2] *= 1e-3
        pose = [float(v) for v in g.standard_normal(6) * [10, 10, 3, 10, 10, 0.3]]
        k = kinds[it % len(kinds)]
        sdt = float(g.choice([0.0, 1e-4, 0.2, 0.8]))
        z, H = s.h_py(x, pose, {"kind": k, "sdt": sdt})
        cases.append({"fn": "radar_h", "args": enc([list(s.native()), kind_code(k), sdt, x, pose]), "ref": enc([z, H])})
        r = MicrowaveRadar((float(g.normal()), float(g.normal()), float(g.normal() * 0.3)))
        z, H = r.h_py(x, pose)
        cases.append({"fn": "radar_h", "args": enc([list(r.native()), 0, 0.0, x, pose]), "ref": enc([z, H])})
    for it in range(60):
        nt, nd = int(g.integers(0, 9)), int(g.integers(0, 9))
        L = g.random((nt, nd)) * 10.0 ** g.uniform(-3, 3, (nt, nd))
        L[g.random((nt, nd)) < 0.6] = 0.0
        G = np.where(L > 1.0, np.log(np.where(L > 1.0, L, 1.0)), -np.inf)
        if it % 4 == 0:
            G = np.where(np.isfinite(G), np.round(G), -np.inf)
            G[G == 0.0] = -np.inf
        cases.append({"fn": "radar_marginals", "args": enc([L]), "ref": enc(T.marginals(L)), "shape": [nt, nd]})
        cases.append({"fn": "radar_assign", "args": enc([G]), "ref": [[int(i), int(j)] for i, j in T.assign(G)], "shape": [nt, nd]})


def main():
    np.random.seed(5)
    for spec in (("junction", 1, False, True), ("parked_street", 1, True, False), ("oncoming", 1, True, True),
                 ("adjacent", 1, False, False), ("curve_parked", 1, False, False)):
        drive(*spec)
    microwave()
    random_cases()
    buf = io.StringIO()
    json.dump(cases, buf)
    arrays["index"] = np.frombuffer(buf.getvalue().encode(), dtype=np.uint8)
    out = os.path.join(HERE, "tests", "data", "radar_golden.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **arrays)
    print("cases", {k: v for k, v in taken.items()}, "+ random", len(cases) - sum(taken.values()),
          "->", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
