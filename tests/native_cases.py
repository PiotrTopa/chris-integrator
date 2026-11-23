"""The cases the native core is held to (tests/test_native.py), and the canonical digest of a result.

Each `cases_*` function yields the inputs of one native function (or of the package function that calls it),
made from `random.Random` alone, so the same inputs come out on every machine and Python version. What the
pure-Python reference gave for each of them, before it was removed, is in tests/data/native_golden.json.gz:
the digest of every exact result (`digest`: types and the bits of every float), and the full value where the
native result agrees with the reference to rounding only (sums of products: `ekf`). The file also holds the
digest of the inputs, so a generator that changed is a test failure, not a silent pass.

    tests/data/native_golden.json.gz    {name: {"inputs": digest, "out": [digest per case] | "values": [...]}}
"""
import hashlib
import json
import os
import random
import struct

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(HERE, "tests", "data", "native_golden.json.gz")
ROAD_RAW = os.path.join(HERE, "tests", "data", "road_raw_vision.jsonl")


# ---- the digest -------------------------------------------------------------------------------------------
def _canon(o, out):
    if o is None:
        out.append(b"n")
    elif o is True or o is False:
        out.append(b"b1" if o else b"b0")
    elif type(o) is float:
        out.append(b"f" + struct.pack("<d", o))
    elif type(o) is int:
        out.append(b"i%d;" % o)
    elif type(o) is str:
        b = o.encode("utf-8", "surrogatepass")
        out.append(b"s%d:" % len(b) + b)
    elif type(o) is bytes:
        out.append(b"y%d:" % len(o) + o)
    elif isinstance(o, np.ndarray):
        a = np.ascontiguousarray(o)
        out.append(b"A" + a.dtype.str.encode() + repr(a.shape).encode() + a.tobytes())
    elif isinstance(o, np.generic):
        out.append(b"S" + type(o).__name__.encode() + o.tobytes())
    elif isinstance(o, (list, tuple)):
        out.append((b"L%d:" if type(o) is list else b"T%d:") % len(o))
        for v in o:
            _canon(v, out)
    elif isinstance(o, dict):
        out.append(b"D%d:" % len(o))
        for k, v in o.items():
            _canon(k, out)
            _canon(v, out)
    elif isinstance(o, type) and issubclass(o, BaseException):
        out.append(b"E" + o.__name__.encode())
    else:
        raise TypeError("digest: %r" % type(o))


def digest(o):
    """16 hex digits of sha256 over the value's canonical bytes: equal digests = equal types and bits."""
    out = []
    _canon(o, out)
    return hashlib.sha256(b"".join(out)).hexdigest()[:16]


def outcome(fn, *args):
    """fn(*args), or the type of the exception it raised (a refusal is a result too)."""
    try:
        return fn(*args)
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        return type(e)


def load_golden():
    import gzip
    with gzip.open(GOLDEN, "rt") as f:
        return json.load(f)


# The inputs are made with additions and multiplications of random() only: no libm function (random.gauss uses
# log and cos, whose last bit differs between C libraries) and no BLAS product (whose order of addition differs
# between machines), so every machine makes the same inputs.
def normal(rng, scale=1.0):
    """About N(0, scale^2): the sum of twelve uniforms, less six."""
    return scale * (sum(rng.random() for _ in range(12)) - 6.0)


def _normal_array(rng, shape, scale=1.0):
    n = int(np.prod(shape))
    return np.array([normal(rng, scale) for _ in range(n)]).reshape(shape)


def _cov(rng, n):
    """A symmetric positive definite matrix (diagonally dominant), with correlations of all signs."""
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            a[i, j] = a[j, i] = rng.uniform(-1.0, 1.0) * rng.random()
    for i in range(n):
        a[i, i] = float(np.sum(np.abs(a[i]))) + rng.uniform(0.01, 1.0)
    return a


# ---- view: interpolation, a drawn line's points and smoothing ----------------------------------------------
def cases_interp():
    rng = random.Random(9)
    for i in range(500):
        n = rng.randrange(0, 20)
        if i % 2:
            xs = [rng.uniform(0, 100) for _ in range(n)]
            if rng.random() < 0.8:
                xs.sort()
            if n > 2 and rng.random() < 0.2:
                xs[rng.randrange(1, n)] = xs[rng.randrange(0, n - 1)]
            ys = [rng.choice([None, float("nan")]) if rng.random() < 0.05 else rng.uniform(-5, 5)
                  for _ in range(rng.randrange(max(n - 2, 0), n + 3))]
            qs = [rng.choice([float("nan"), -1.0, 200.0]) if rng.random() < 0.05 else rng.uniform(-5, 105)
                  for _ in range(rng.randrange(0, 30))]
            if rng.random() < 0.7:
                qs.sort(key=lambda v: (v != v, v))
        else:
            def num():
                return rng.choice([rng.randrange(-5, 120), float(rng.randrange(-5, 120)), rng.uniform(-5, 120)])
            xs = sorted(num() for _ in range(max(n, 1)))
            ys = [None if rng.random() < 0.05 else num() for _ in range(len(xs))]
            qs = sorted(num() for _ in range(rng.randrange(0, 12)))
            if rng.random() < 0.5:
                xs, ys, qs = tuple(xs), tuple(ys), tuple(qs)
        yield xs, ys, qs


VIEW_XS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0)


def cases_view_prep():
    """(gx, y, y_std, xs, reach_std, b) of trackers.view `_prep`."""
    rng = random.Random(17)
    for _ in range(2000):
        xs = VIEW_XS if rng.random() < 0.7 else tuple(sorted(rng.choice([rng.uniform(-2, 130), float(rng.randrange(0, 130))])
                                                             for _ in range(rng.randrange(2, 16))))
        n = rng.randrange(0, 40)
        gx = sorted(rng.uniform(-5, 150) for _ in range(n))
        if n > 2 and rng.random() < 0.1:
            gx[rng.randrange(1, n)] = gx[rng.randrange(0, n - 1)]

        def num():
            return rng.choice([None, float("nan")]) if rng.random() < 0.03 else rng.uniform(-6, 6)
        y = [num() for _ in range(rng.randrange(max(0, n - 3), n + 2))]
        k = rng.randrange(3)
        ys = [] if k == 0 else [rng.uniform(0, 0.6) if rng.random() < 0.97 else None for _ in range(n)] if k == 1 \
            else [rng.uniform(0, 2.0) for _ in range(max(0, n - 5))]
        b = None if rng.random() < 0.3 else [rng.uniform(-6, 6) for _ in range(len(xs) - rng.choice([0, 0, 0, 1]))]
        yield gx, y, ys, xs, rng.choice([0.3, 0.5, 1.0]), b
    yield [0, 50, 100], [1, 2, 3], [], (0, 5, 20, 60), 0.5, None            # ints: a new line keeps them


def cases_view_blend():
    """(a, b, dy, ys, ks, dt, shifting) of trackers.view `_blend`."""
    rng = random.Random(18)
    for _ in range(1500):
        nx = rng.randrange(2, 16)
        xs = sorted(rng.uniform(0, 120) for _ in range(nx))

        def row(cut=True):
            return [rng.uniform(-6, 6) for _ in range(nx - (rng.choice([0, 0, 1]) if cut else 0))]
        ks = rng.choice([[0.05 / (0.25 + 0.01 * x) for x in xs], [rng.random()] * nx, [1.0] * nx])
        yield row(), row(), row(), row(False), ks, rng.choice([0.0, -0.05, 0.05, rng.uniform(0.001, 0.2)]), rng.random() < 0.2
    yield [1, 2], [1, 2], [0.0, 0.0], [2, 3], [0.5, 0.5], 0.05, False


# ---- the road line: trim and the recorder's reduced form ---------------------------------------------------
def _road_bases():
    with open(ROAD_RAW, "rb") as f:
        base = [ln.strip() for ln in f if ln.strip()]
    base.append(json.dumps({"type": "road", "plan": {"vx": [1], "x": [0, 1], "vy": [2], "vx2": [], "prob": None},
                            "lane_lines": [{"z": [1], "x": [], "z ": [2]}, {"x": [1], "z": [3, [4]]}],
                            "road_edges": [{"y": [], "z": []}]}, separators=(",", ":")).encode())
    return base


def cases_trim():
    rng = random.Random(29)
    base = _road_bases()
    yield b""
    for b in base:
        yield b
    for _ in range(6000):
        b = bytearray(rng.choice(base))
        for _ in range(rng.randrange(1, 4)):
            if not b:
                break
            i, k = rng.randrange(len(b)), rng.randrange(4)
            if k == 0:
                del b[i:]
            elif k == 1:
                b[i] = rng.choice(b'[]{},:"z0')
            elif k == 2:
                del b[i:i + rng.randrange(1, 40)]
            else:
                j = rng.randrange(len(b))
                b[i:i] = b[j:j + rng.randrange(1, 60)]
        yield bytes(b)


def cases_slim():
    """road messages (dicts) for recorder.slim_road_line."""
    rng = random.Random(31)

    def num():
        return rng.choice([rng.uniform(-60, 60), rng.randrange(-5, 99), round(rng.uniform(-3, 3), rng.randrange(1, 6)),
                           rng.randrange(-999, 999) / 1000 + 0.0005, rng.uniform(-1, 1) * 1e-5, -0.0, 0.0, 1e300,
                           float("nan"), float("inf")] + ([None, True] if rng.random() < 0.05 else []))

    def arr():
        return [num() for _ in range(rng.randrange(0, 34))]

    with open(ROAD_RAW, "rb") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)
    for _ in range(2000):
        m = {"type": "road", "seq": rng.randrange(10 ** 6), "ts_mono_ns": rng.randrange(10 ** 15), "v_ego": num(),
             "valid": rng.choice([True, False]), "road_model": rng.choice(["supercombo", "vision", None]), "x": 5,
             "camera": rng.choice(["road", "wide", "é\n"]), "frame_seq_prev": rng.choice([None, 4, [1, 2]]),
             "pose": {"trans": arr(), "speed": num(), "w": rng.uniform(-1, 1), "s": "x", "n": None},
             "lane_lines": [{"x": arr(), "y": arr(), "y_std": arr(), "z": arr(),
                             "prob": rng.choice([rng.random(), True, 1, None, "p"])} for _ in range(4)] + ["odd"],
             "road_edges": [{"x": arr(), "y": rng.choice([arr(), None, 3]), "y_std": arr()} for _ in range(2)],
             "leads": [{"x": arr(), "prob": rng.random(), "t": rng.choice([arr(), 1.23456, "s"])}, {"x": [1]}],
             "plan": {"x": arr(), "y": arr(), "z": arr(), "vx": arr(), "held": rng.choice([True, None]), "age_ms": 30}}
        for k in list(m):
            if k != "type" and rng.random() < 0.08:
                del m[k]
        if rng.random() < 0.05:
            m["leads"] = rng.choice([[], None, ["x"]])
        if rng.random() < 0.05:
            m["plan"] = rng.choice([None, [1], {"held": False}])
        if rng.random() < 0.02:
            m["pose"] = {1: 2.5, "k": 1.25}            # a key json converts
        yield m


# ---- the road geometry's centre line ----------------------------------------------------------------------
def cases_geom():
    """(x, y, sd or conf, reach or None) for RoadGeometryFusion `_centre_vision` / `_centre`."""
    rng = random.Random(37)
    for i in range(2000):
        n = rng.randrange(1, 40)
        x = np.array(sorted(rng.uniform(-5, 150) for _ in range(n)))
        y = np.array([rng.uniform(-5, 5) for _ in range(n)])
        if i % 2:
            sd = np.array([float("nan") if rng.random() < 0.05 else rng.uniform(0, 3) for _ in range(n)])
            yield x, y, sd, rng.uniform(-10, 160)
        else:
            yield x, y, np.array([rng.random() for _ in range(n)]), None


def cases_reach():
    """(x, sd, reach_std, reach_max, reach_min) for RoadGeometryFusion `_reach`."""
    rng = random.Random(38)
    for _ in range(3000):
        n = rng.randrange(1, 40)
        x = np.array(sorted(rng.uniform(-5, 150) for _ in range(n)))
        sd = np.array([float("nan") if rng.random() < 0.05 else rng.uniform(0, 2) for _ in range(n)])
        yield x, sd, rng.choice([0.9, 1, 0.5]), rng.choice([100.0, 100, 50, 1e9]), rng.choice([15.0, 15, 200])


# ---- the lane tracker ---------------------------------------------------------------------------------------
def road_lines(rng, n=33, odd=0.0):
    """The lines of a road message (the vision model's layout), with odd members at the rate `odd`."""
    x = [0.0] + [round(0.1875 * (i ** 2) * 192.0 / 1024.0, 4) for i in range(1, n)]

    def line(edge):
        d = {"x": list(x), "y": [normal(rng, 2.0) for _ in range(n)], "y_std": [abs(0.2 + normal(rng, 0.2)) for _ in range(n)]}
        if not edge:
            d["prob"] = rng.choice([rng.random(), 0, 1, True, None, 1.5, -0.2, float("nan")])
        if rng.random() < odd:
            k = rng.randrange(9)
            if k == 0:
                d["y"][rng.randrange(n)] = float("nan")
            elif k == 1:
                d["y_std"][rng.randrange(n)] = float("inf")
            elif k == 2:
                del d["y_std"]
            elif k == 3:
                d["y"] = d["y"][:-1]
            elif k == 4:
                d["y"][rng.randrange(n)] = 3
            elif k == 5:
                d["y_std"][rng.randrange(n)] = 0.0
            elif k == 6:
                d["y"][rng.randrange(n)] = "1.5"
            elif k == 7:
                d["y_std"] = None
            else:
                d["prob"] = "x"
        return d
    return {"lane_lines": [line(False) for _ in range(rng.choice([4, 4, 4, 3]))],
            "road_edges": [line(True) for _ in range(rng.choice([2, 2, 2, 1]))]}


def cases_lanes_measure():
    rng = random.Random(41)
    for i in range(3000):
        road = road_lines(rng, odd=0.3 if i % 2 else 0.0)
        if i % 97 == 0:
            road = rng.choice([{}, {"lane_lines": []}, {"lane_lines": [None, 3]}, {"road_edges": [{"x": [1, 2]}]},
                               {"lane_lines": [{"x": [0, 1, 2, 3, 4], "y": [0, 0, 0, 0, 0]}]}])
        yield road


def cases_lanes_arrays():
    """(cfg changes, x, zy, zs, prob, y, p, upd) for the tracker's `_fit`, `_update`, `_round2`, `lanes_predict`."""
    rng = random.Random(43)
    for i in range(600):
        n = rng.choice([33, 33, 17, 5])
        x = np.array(sorted(rng.uniform(0, 190) for _ in range(n)))
        zy, zs = _normal_array(rng, (6, n), 2.0), np.abs(_normal_array(rng, (6, n), 0.3)) + 1e-3
        prob = np.array([rng.random() for _ in range(6)])
        y, p = _normal_array(rng, (6, n), 2.0), np.abs(_normal_array(rng, (6, n))) + 1e-3
        c = {}
        if i % 5 == 0:
            c["gate_x"] = (500.0, 600.0)
        elif i % 7 == 0:
            c["gate_x"] = (rng.uniform(0, 50), rng.uniform(50, 200))
        upd = sorted(rng.sample(range(6), rng.randrange(0, 7)))
        yield c, x, zy, zs, prob, y, p, upd


def lane_frames():
    """1500 frames (road, ego_state) for the whole lane tracker."""
    rng = random.Random(42)
    frames = []
    ts = 10 ** 12
    base = road_lines(rng)
    for i in range(1500):
        ts += int(rng.choice([0.05, 0.066, 0.07, 0.2, 1.4]) * 1e9) if rng.random() < 0.97 else 0
        if rng.random() < 0.05:
            base = road_lines(rng, odd=0.3)
        road = {"type": "road", "seq": i, "ts_mono_ns": ts, "pose": {"vy": rng.uniform(-0.2, 0.2)}}
        for key in ("lane_lines", "road_edges"):
            road[key] = [dict(d, y=[v + normal(rng, 0.05 if rng.random() < 0.9 else 2.0) if isinstance(v, float) else v
                                    for v in d["y"]],
                              **({"prob": min(1.0, max(0.0, 0.6 + normal(rng, 0.3)))} if "prob" in d else {}))
                         for d in base[key]]
        es = {"speed": {"value": rng.uniform(0, 30), "valid": True, "sources": ["vehicle.speed"]},
              "yaw_rate": {"value": rng.uniform(-0.1, 0.1), "valid": True}, "sources": {"vision": {"state": "ok"}}}
        frames.append((road, es))
    return frames


# ---- the estimator's filter -----------------------------------------------------------------------------------
def ego_model(rng, naux):
    from chris_integrator.core import ego as E
    m = E.EgoModel()
    for i in range(naux):
        k = i % 3
        if k == 0:
            m.add_aux("s%d" % i, E.AuxState("a", 0.0, 0.1, psd=1e-6))
        elif k == 1:
            m.add_aux("s%d" % i, E.AuxState("a", 0.0, 0.05, tau_s=30.0 + i))
        else:
            m.add_aux("s%d" % i, E.AuxState("a", 0.0, 0.2, psd=1e-5, lo=-0.5, hi=0.5))
    return m


def ego_state(rng, m):
    x = _normal_array(rng, (m.n,))
    x[3] = rng.uniform(-5, 40)
    x[9] = rng.uniform(-0.05, 0.05)
    return x, _cov(rng, m.n)


def cases_predict():
    """(naux, x, P, dt) for EgoModel.predict."""
    rng = random.Random(10)
    for i in range(90):
        naux = (0, 1, 7)[i % 3]
        m = ego_model(rng, naux)
        x, P = ego_state(rng, m)
        if i % 7 == 0:
            x[10:] = [rng.uniform(-2, 2) for _ in range(m.n - 10)]                   # beyond the clamps
        if i % 11 == 0:
            P[3, 3] += 1e5                                                              # the caps of a long gap
        yield naux, x, P, rng.choice([0.01, 0.02, 0.05, 0.1, 0.3, 0.7, 2.5])


def cases_rows():
    """(P, x, idx, coef, nu, var, freeze) of one scalar row (core/ekf.py `_project`, `_apply`, `_sym`). The first
    1500 rows read one state (the native result has the reference's bits), the rest several (to rounding)."""
    rng = random.Random(20)
    n = 27
    for i in range(2000):
        P = _cov(rng, n)
        x = _normal_array(rng, (n,))
        if i < 1500:
            idx = np.array([rng.randrange(n)], dtype=np.intp)
            coef = np.array([rng.choice([1.0, -1.0, rng.uniform(-3, 3)])])
        else:
            m = rng.randrange(2, 7)
            idx = np.array(rng.sample(range(n), m), dtype=np.intp)
            coef = _normal_array(rng, (m,))
        freeze = None if i % 3 else np.array(sorted(set(rng.randrange(n) for _ in range(6))), dtype=np.intp)
        yield P, x, idx, coef, normal(rng), rng.uniform(1e-4, 1.0), freeze


def cases_rts():
    """(x, P, dt, xs noise, wrap) for one smoother step (EgoModel.rts) after EgoModel.predict."""
    rng = random.Random(30)
    for i in range(40):
        m = ego_model(rng, 17)
        x, P = ego_state(rng, m)
        yield x, P, rng.choice([0.01, 0.05, 0.1, 0.4]), _normal_array(rng, (m.n,), 0.01), (2,) if i % 4 == 0 else ()
