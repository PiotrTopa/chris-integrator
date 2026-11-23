"""`python -m chris_integrator.native`: the native core in use, and a self-test of it on this machine (a few
seconds; exit status 0 = it gives the results it must, 1 = built but wrong: do not use it). JSON and rounding
are held to CPython's own json and round; the other parts to the results the development build gives on
fixed inputs. The install script runs this before it keeps a binary; a package without a loadable module of
the right version does not import at all (and says how to build it)."""
import hashlib
import json
import math
import random
import struct
import sys

import numpy as np

from . import PARTS, _n, status


def selftest(n=20000, seed=1):
    """Problems found (an empty list: the native functions gave the reference's results)."""
    bad = []
    rng = random.Random(seed)

    def rnd_float():
        k = rng.randrange(4)
        if k == 0:
            v = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
            return v if math.isfinite(v) else 0.5
        if k == 1:
            return round(rng.uniform(-300.0, 300.0), rng.randrange(0, 6))
        if k == 2:
            return (rng.randrange(-10 ** 6, 10 ** 6) + 0.5) / 10.0 ** rng.randrange(0, 5)
        return rng.uniform(-1e3, 1e3)

    for _ in range(n):
        v = rnd_float()
        if _n.dumps(v) != repr(v):
            bad.append("dumps(%r) = %s" % (v, _n.dumps(v)))
        d = rng.randrange(0, 6)
        if repr(_n.round(v, d)) != repr(round(v, d)):
            bad.append("round(%r, %d) = %r" % (v, d, _n.round(v, d)))
        t = repr(v)
        if repr(_n.loads(t)) != repr(json.loads(t)):
            bad.append("loads(%s) = %r" % (t, _n.loads(t)))
        if len(bad) > 5:
            return bad
    msg = {"type": "x", "seq": 7, "ts_real_ns": 1795084321890056390, "a": [rnd_float() for _ in range(50)],
           "s": "a\"b\\c\né\U0001F600", "n": None, "t": True, "d": {"k": [1, 2.5, {"z": -0.0}]}}
    text = json.dumps(msg, separators=(",", ":"))
    if _n.dumps(msg) != text:
        bad.append("dumps of a message differs from json.dumps")
    if json.dumps(_n.loads(text), separators=(",", ":")) != text or json.dumps(_n.loads(text.encode()), separators=(",", ":")) != text:
        bad.append("loads of a message differs from json.loads")
    # the other parts: their results on fixed inputs, held to what the development build gave (exact parts:
    # the digest of every result; the filter's covariance steps, which sum products: to rounding). That build is
    # held to the results the removed pure-Python references gave by tests/test_native.py.
    for name, fn in _CHECKS:
        got = fn()
        want = EXPECT[name]
        if isinstance(want, str):
            ok = _digest(got) == want
        else:
            ok = len(got) == len(want) and all(abs(u - v) <= 1e-7 * max(1.0, abs(v)) for u, v in zip(got, want))
        if not ok:
            bad.append("%s differs from the recorded results" % name)
    g = np.random.default_rng(seed)
    # the radar's motion model against its numpy reference (models._evolve_py): to rounding
    from ..radar import models as R
    labels = list(R.KIN)
    for it in range(40):
        B = 1 + it % 7
        xs = g.standard_normal((B, 3, 8)) * np.array([30.0, 30.0, 10.0 + 20.0 * (it % 3 == 0), 10.0, 2.0, 2.0, 0.01, 0.3])
        a = g.standard_normal((B, 3, 8, 8))
        Ps = np.einsum("bhij,bhkj->bhik", a, a) + np.eye(8)
        mu = g.random((B, 3))
        mu[0, it % 3] = 0.0
        mu /= mu.sum(axis=1, keepdims=True)
        xf, a = g.standard_normal((B, 8)), g.standard_normal((B, 8, 8))
        Pf = np.einsum("bij,bkj->bik", a, a) + np.eye(8)
        dt = g.choice([-0.05, 0.0, 1e-14, 0.05, 0.5, 2.0], B)
        lab = [labels[i] for i in g.integers(len(labels), size=B)]
        ref = R._evolve_py(xs, Ps, mu, xf, Pf, dt, lab, R.CFG, True, True)
        got = tuple(np.empty_like(r) for r in ref)
        _n.radar_evolve(list(xs), list(Ps), list(mu), list(xf), list(Pf), list(dt),
                        [R._KIN4[l] for l in lab], R._native_par(R.CFG), *got)
        if any(float(np.max(np.abs(r - o))) > 1e-12 * max(1.0, float(np.max(np.abs(r)))) for r, o in zip(ref, got)):
            bad.append("radar_evolve differs from the numpy reference")
            break
        q = R.still_b(xs[:, 1:], Ps[:, 1:], R.kin_arrays(lab)[3][:, None])
        ps = np.minimum(1.0, mu[:, 0] + mu[:, 1] * q[:, 0] + mu[:, 2] * q[:, 1])
        if float(np.max(np.abs(ps - _n.radar_p_still(xs, Ps, mu, list(R.kin_arrays(lab)[3]))))) > 1e-12:
            bad.append("radar_p_still differs from the numpy reference")
            break
    return bad


def _digest(o):
    """16 hex digits of sha256 over the value's types and bits (as tests/native_cases.py `digest`)."""
    out = []

    def walk(o):
        if o is None:
            out.append(b"n")
        elif o is True or o is False:
            out.append(b"b1" if o else b"b0")
        elif type(o) is float:
            out.append(b"fNaN" if o != o else b"f" + struct.pack("<d", o))
        elif type(o) is int:
            out.append(b"i%d;" % o)
        elif type(o) is str:
            out.append(b"s%d:" % len(o.encode()) + o.encode())
        elif type(o) is bytes:
            out.append(b"y%d:" % len(o) + o)
        elif isinstance(o, np.ndarray):
            a = np.ascontiguousarray(o)
            if a.dtype.kind == "f" and np.isnan(a).any():
                a = np.where(np.isnan(a), np.nan, a)
            out.append(b"A" + a.dtype.str.encode() + repr(a.shape).encode() + a.tobytes())
        elif isinstance(o, (list, tuple)):
            out.append((b"L%d:" if type(o) is list else b"T%d:") % len(o))
            for v in o:
                walk(v)
        elif isinstance(o, dict):
            out.append(b"D%d:" % len(o))
            for k, v in o.items():
                walk(k)
                walk(v)
        else:
            raise TypeError(type(o))
    walk(o)
    return hashlib.sha256(b"".join(out)).hexdigest()[:16]


# (inputs from additions and multiplications of random() only: the same on every machine, whatever its C library
# and BLAS)
def _normal(rng, scale=1.0):
    return scale * (sum(rng.random() for _ in range(12)) - 6.0)


def _rng_cov(rng, n):
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            a[i, j] = a[j, i] = rng.uniform(-1.0, 1.0) * rng.random()
    for i in range(n):
        a[i, i] = float(np.sum(np.abs(a[i]))) + rng.uniform(0.01, 1.0)
    return a


def _check_view():
    rng = random.Random(3)
    xs = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0)
    out = []
    for _ in range(200):
        gx = sorted(rng.uniform(0.0, 150.0) for _ in range(rng.randrange(2, 34)))
        y = [None if rng.random() < 0.03 else rng.uniform(-5.0, 5.0) for _ in gx]
        sd = [rng.uniform(0.0, 1.0) for _ in gx]
        b = [rng.uniform(-5.0, 5.0) for _ in xs]
        out.append(_n.interp_all(gx, y, xs))
        try:
            r = _n.view_prep(gx, y, sd, xs, 0.5, b)
        except ZeroDivisionError:
            r = "zero"
        out.append(r)
        if isinstance(r, tuple):
            ks = [rng.random() for _ in xs]
            out.append(_n.view_blend(list(b), list(b), [0.0] * len(xs), r[0], ks, 0.05, rng.random() < 0.2))
    return out


def _road(rng, n=33):
    x = [round(192.0 * (i / (n - 1.0)) ** 2, 4) for i in range(n)]

    def arr(k=n, s=2.0):
        return [round(_normal(rng, s), rng.randrange(1, 8)) for _ in range(k)]
    return {"type": "road", "seq": rng.randrange(10 ** 6), "ts_mono_ns": rng.randrange(10 ** 15), "v_ego": rng.uniform(0, 40),
            "pose": {"vx": rng.uniform(0, 40), "vy": _normal(rng, 0.1), "yaw_rate": _normal(rng, 0.05)},
            "plan": {"x": arr(), "y": arr(), "z": arr(), "vx": arr(), "prob": arr()},
            "lane_lines": [{"x": list(x), "y": arr(), "z": arr(), "y_std": [abs(v) for v in arr(s=0.3)],
                            "prob": rng.random()} for _ in range(4)],
            "road_edges": [{"x": list(x), "y": arr(), "z": arr(), "y_std": [abs(v) for v in arr(s=0.6)]} for _ in range(2)],
            "leads": [{"x": arr(6), "prob": rng.random()}]}


def _check_road_line():
    from ..fastjson import trim_road
    rng = random.Random(4)
    out = []
    for _ in range(50):
        r = _road(rng)
        line = json.dumps(r, separators=(",", ":")).encode()
        out.append(trim_road(line))
        out.append(trim_road(line[:rng.randrange(len(line))]))
        out.append(_n.slim_road_line(r))
    return out


def _check_geometry():
    rng = random.Random(5)
    out = []
    for _ in range(200):
        n = rng.randrange(1, 34)
        x = np.array(sorted(rng.uniform(0.0, 150.0) for _ in range(n)))
        y, sd = np.array([_normal(rng, 2.0) for _ in range(n)]), np.array([rng.uniform(0, 2) for _ in range(n)])
        try:
            out.append(_n.geom_centre(x, y, sd, rng.uniform(-5.0, 160.0)))
        except IndexError:
            out.append("empty")
        out.append(_n.geom_reach(x, sd, 0.9, 100.0, 15))
    return out


def _check_lanes():
    from ..trackers import lanes as L
    rng = random.Random(6)
    tr = L.RoadGeometryTracker()
    c = L.DEFAULT_CFG
    out = []
    for i in range(60):
        r = _road(rng)
        if i % 3 == 0:
            r["lane_lines"][rng.randrange(4)]["y"][rng.randrange(33)] = float("nan")
            del r["road_edges"][rng.randrange(2)]["y_std"]
        x, zy, zs, prob, ok = m = tr._measure(r)
        out.append(m)
        y = np.array([[_normal(rng, 2.0) for _ in range(33)] for _ in range(6)])
        p = np.array([[rng.uniform(0.001, 1.0) for _ in range(33)] for _ in range(6)])
        fit = L._fit(c, x, zs, zy, prob, y, p)
        out.append(fit)
        L._update(y, p, fit[0], fit[2], [0, 2, 5])
        out.extend([y, p, L._round2(y), L._round2(p, True)])
        _n.lanes_predict(p, x, c["q0"], c["q1"], 0.05, c["max_std"] ** 2)
        out.append(p)
    return out


def _model():
    from ..core import ego as E
    m = E.EgoModel()
    for i in range(17):
        m.add_aux("s%d" % i, E.AuxState("a", 0.0, 0.1, psd=1e-6) if i % 3 == 0 else
                  E.AuxState("a", 0.0, 0.05, tau_s=30.0 + i) if i % 3 == 1 else E.AuxState("a", 0.0, 0.2, psd=1e-5, lo=-0.5, hi=0.5))
    return m


def _check_rows():
    from ..core import ekf as K
    rng = random.Random(7)
    out = []
    for i in range(100):
        P = _rng_cov(rng, 27)
        x = np.array([_normal(rng) for _ in range(27)])
        ph, s0 = K._project(P, np.array([rng.randrange(27)], dtype=np.intp), np.array([rng.uniform(-3, 3)]))
        freeze = None if i % 3 else np.array(sorted(set(rng.randrange(27) for _ in range(5))), dtype=np.intp)
        P1 = K._apply(x, P, ph, s0 + 0.1, _normal(rng), freeze, False)
        out.extend([ph, s0, x, P1, K._sym(P1 + 1e-3 * np.triu(P1, 1), True)])
    return out


def _check_filter():
    """The filter's prediction (state, transition, covariance) and one smoother step: sums and traces, to rounding
    (sin, cos and exp are the C library's; the covariance steps sum products)."""
    rng = random.Random(9)
    m = _model()
    vals = []
    for dt in (0.01, 0.05, 0.3, 2.5):
        x = np.array([_normal(rng) for _ in range(m.n)])
        x[3] = 17.0
        P = _rng_cov(rng, m.n)
        xp, Pp, F = m.predict(x, P, dt)
        xs, Ps = m.rts(Pp, F, P, x, xp, xp + 0.01, Pp * 0.9, (2,))
        vals += [float(xp.sum()), float(F[0].sum()), float(F[1].sum()), float(Pp.sum()), float(np.trace(Pp)),
                 float(xs.sum()), float(Ps.sum()), float(np.trace(Ps))]
    return vals


_CHECKS = (("view", _check_view), ("road line", _check_road_line), ("road geometry", _check_geometry),
           ("lanes", _check_lanes), ("filter rows", _check_rows), ("filter", _check_filter))
# what the development build gives (x86-64, glibc; held to the removed references by tests/test_native.py)
EXPECT = {"view": "1d8746b3a5684186", "road line": "6ec3625f028f58ce", "road geometry": "4997564407c2f49f",
          "lanes": "6ca8cd4c07318197", "filter rows": "87ce1e5c039ae00b",
          "filter": [7.723780535727707, 10.435565184078538, 26.99841329569997, 193.8061211825356, 189.01412870797915,
                     8.782660472555797, 168.72298218978383, 169.76519932373355, 23.100822279577056, 10.480539787057944,
                     26.9920707506736, 207.80726029609707, 193.519507522703, 24.874612526046533, 180.82839053291355,
                     173.4712050972677, 20.708935982170157, 10.472420533251132, 26.952584286671037, 460.22065652485065,
                     349.9137133339737, 14.261576202244669, 170.36402725669132, 173.6538253283887, -27.67484334942202,
                     103.45638385057444, 26.61632833321396, 10112.430097201888, 13253.838829430555, 36.24608045588081,
                     -2125.9564503241677, -2336.4921069275197]}


def main():
    st = status()
    print("chris-integrator native core: %s" % json.dumps(st))
    try:
        bad = selftest()
    except Exception as e:                 # a compiled module that raises where it must not
        bad = ["%s: %s" % (type(e).__name__, e)]
    if bad:
        print("SELF-TEST FAILED: the native core does not give its results on this machine:")
        for b in bad:
            print("  " + b)
        return 1
    print("self-test ok: %s" % ", ".join(PARTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
