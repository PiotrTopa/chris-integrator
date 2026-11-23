"""The native core (chris_integrator/native): every part held to known results.

JSON and rounding against CPython's own json and round (they stay the reference: the native writer and
reader hand to them what they do not take). The other parts against what their pure-Python references gave
before those were removed (tests/native_cases.py, tests/data/native_golden.json.gz): exact results bit for
bit, the filter's sums of products to rounding. The radar's motion model against its numpy reference.

    python -m pytest tests/test_native.py
"""
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from chris_integrator import native as N  # noqa: E402
from chris_integrator.native import _chris_native as C  # noqa: E402


def _always(f):
    return f


needs_native = needs_json = needs_ekf = _always          # (the native core is part of the package)
SEP = (",", ":")


def _ref_dumps(o):
    return json.dumps(o, separators=SEP)


def _native_dumps(o):
    """The native writer, or None where it hands the object to json."""
    try:
        return C.dumps(o)
    except C.Fallback:
        return None


def _rand_double(rng):
    k = rng.randrange(8)
    if k == 0:
        return struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]       # any bit pattern
    if k == 1:
        return round(rng.uniform(-200.0, 200.0), rng.randrange(0, 7))               # what the stack writes most
    if k == 2:
        return rng.uniform(-1.0, 1.0) * 10.0 ** rng.randrange(-30, 30)
    if k == 3:
        return float(rng.randrange(-10 ** 6, 10 ** 6))
    if k == 4:
        return rng.randrange(-10 ** 17, 10 ** 17) / 10.0 ** rng.randrange(0, 23)
    if k == 5:
        return rng.choice([0.0, -0.0, 1e-4, 9.999e-5, 1e15, 1e16, 1e-5, 5e-324, 1.7976931348623157e308, 0.1, 0.3,
                           2.675, 1e22, 1e23, 123456789012345.0, 999999999999999.9, 0.30000000000000004])
    if k == 6:
        return math.ldexp(rng.randrange(1, 1 << 53), rng.randrange(-80, 30))
    return rng.uniform(-1e4, 1e4)


def _rand_str(rng):
    alphabet = ["a", "Z", "0", " ", '"', "\\", "/", "\n", "\r", "\t", "\b", "\f", "\x00", "\x1f", "\x7f", "é", "ś",
                " ", "￿", "\U0001F600", "\U0010ffff", "{", "]", ","]
    return "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 12)))


def _rand_obj(rng, depth=0):
    k = rng.randrange(14 if depth < 5 else 9)
    if k == 0:
        return None
    if k == 1:
        return rng.choice([True, False])
    if k == 2:
        return rng.randrange(-10 ** 6, 10 ** 6)
    if k == 3:
        return rng.choice([0, -1, 2 ** 63 - 1, -2 ** 63, 2 ** 63, -2 ** 63 - 1, 10 ** 30, 1795084321890056390])
    if k in (4, 5, 6):
        return _rand_double(rng)
    if k == 7:
        return rng.choice([float("nan"), float("inf"), float("-inf")])
    if k == 8:
        return _rand_str(rng)
    if k in (9, 10):
        return [_rand_obj(rng, depth + 1) for _ in range(rng.randrange(0, 6))]
    if k == 11:
        return tuple(_rand_obj(rng, depth + 1) for _ in range(rng.randrange(0, 4)))
    return {_rand_str(rng): _rand_obj(rng, depth + 1) for _ in range(rng.randrange(0, 6))}


# ---- the module ---------------------------------------------------------------------------------------------
def test_status():
    st = N.status()
    assert st["active"] and st["implementation"] == "native" and st["version"] == N.VERSION == C.VERSION
    assert st["parts"] == list(N.PARTS)
    assert N.dumps({"a": [1, 2.5, None]}) == '{"a":[1,2.5,null]}'
    assert N.loads('{"a":[1,2.5,null]}') == {"a": [1, 2.5, None]}
    assert N.round(2.675, 2) == round(2.675, 2)


def test_a_package_without_the_module_says_how_to_build_it():
    """Without the compiled module, or with one of another version, the package does not import, and the error
    says what to do."""
    tmp = tempfile.mkdtemp()
    try:
        pkg = os.path.join(tmp, "chris_integrator")
        shutil.copytree(os.path.join(HERE, "chris_integrator"), pkg, ignore=shutil.ignore_patterns("*.so", "__pycache__"))
        code = "import chris_integrator.stack"
        r = subprocess.run([sys.executable, "-c", code], cwd=tmp, capture_output=True, text=True)
        assert r.returncode != 0 and "ImportError" in r.stderr and "is not built" in r.stderr and "native/build.sh" in r.stderr
        with open(os.path.join(pkg, "native", "_chris_native.py"), "w") as f:
            f.write("VERSION = 1\nFallback = ValueError\n")
        r = subprocess.run([sys.executable, "-c", code], cwd=tmp, capture_output=True, text=True)
        assert r.returncode != 0 and "version 1, this package needs version %d" % N.VERSION in r.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_self_test_passes_here():
    r = subprocess.run([sys.executable, "-m", "chris_integrator.native"], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0 and "self-test ok" in r.stdout, r.stdout + r.stderr


# ---- JSON text ---------------------------------------------------------------------------------------------
@needs_native
def test_dumps_floats_as_repr():
    rng = random.Random(1)
    taken = 0
    for _ in range(400000):
        v = _rand_double(rng)
        if not math.isfinite(v):
            continue
        assert C.dumps(v) == repr(v), v
        taken += 1
    assert taken > 300000
    # every number of two and three decimals in the range the stack writes
    for d in (2, 3):
        div = 10.0 ** d
        vals = [k / div for k in range(-40000, 40000)]
        assert C.dumps(vals) == _ref_dumps(vals)


@needs_native
def test_dumps_objects_as_json():
    rng = random.Random(2)
    native = 0
    for _ in range(20000):
        o = _rand_obj(rng)
        got = _native_dumps(o)
        if got is not None:
            native += 1
            assert got == _ref_dumps(o)
        assert N.dumps(o) == _ref_dumps(o)             # the public function: native or the reference, same text
    assert native > 10000


@needs_native
def test_dumps_leaves_to_json_what_it_does_not_know():
    class Sub(float):
        pass
    import enum

    class E(enum.IntEnum):
        A = 3
    for o in ({1: 2}, {None: 1}, {1.5: 1}, {True: 1}, 10 ** 30, "\ud800", [np.float64(1.5)], [Sub(2.5)], [E.A],
              [np.int64(3)], {"a": object()}, [set()], b"x"):
        try:
            ref = _ref_dumps(o)
        except (TypeError, ValueError) as e:
            with pytest.raises(type(e)):
                N.dumps(o)
            continue
        assert N.dumps(o) == ref
    import collections
    od = collections.OrderedDict([("a", 1), ("b", 2), ("c", 3)])
    od.move_to_end("a")                                 # its own order, not the underlying dict's
    dd = collections.defaultdict(list, {"k": [1.5]})
    for o in (od, {"x": od}, dd):
        assert N.dumps(o) == _ref_dumps(o)
    deep = cur = []
    for _ in range(500):
        cur.append([])
        cur = cur[0]
    assert N.dumps(deep) == _ref_dumps(deep)
    loop = []
    loop.append(loop)
    with pytest.raises(ValueError):
        N.dumps(loop)


# ---- JSON parsing ------------------------------------------------------------------------------------------
def _same(a, b):
    """Equal as json.loads results: types, values, the sign of zero, NaN, key order."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (a != a and b != b) or (a == b and math.copysign(1.0, a) == math.copysign(1.0, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _both_loads(raw):
    """The native reader and json.loads give the same object or both refuse."""
    try:
        ref = json.loads(raw)
    except (ValueError, RecursionError, TypeError) as e:
        with pytest.raises((ValueError, RecursionError) if not isinstance(e, TypeError) else TypeError):
            N.loads(raw)
        return None, type(e)
    got = N.loads(raw)
    assert _same(got, ref), raw[:200]
    return got, None


@needs_native
def test_loads_numbers():
    rng = random.Random(5)
    for _ in range(200000):
        k = rng.randrange(6)
        if k == 0:
            t = repr(_rand_double(rng))
        elif k == 1:
            t = "%s%d.%s" % (rng.choice(["", "-"]), rng.randrange(0, 10 ** rng.randrange(1, 20)),
                             "".join(rng.choice("0123456789") for _ in range(rng.randrange(1, 26))))
        elif k == 2:
            t = "%.*e" % (rng.randrange(0, 20), _rand_double(rng))
        elif k == 3:
            t = str(rng.randrange(-10 ** rng.randrange(1, 25), 10 ** rng.randrange(1, 25)))
        elif k == 4:
            t = "%.*f" % (rng.randrange(0, 20), rng.uniform(-1e6, 1e6))
        else:
            t = rng.choice(["-0", "-0.0", "0.0", "0", "1E5", "1e+5", "1e-5", "01", "1.", ".5", "-", "+1", "1e", "1e+",
                            "0x10", "NaN", "Infinity", "-Infinity", "1.0e400", "-1e-400", "9007199254740993",
                            "9007199254740993.0", "0.1e1", "1_0", "9223372036854775807", "9223372036854775808",
                            "-9223372036854775808", "-9223372036854775809", "1.7976931348623157e308", "4.9e-324"])
        if t in ("nan", "inf", "-inf"):
            continue
        _both_loads(t)
        _both_loads("[%s,%s]" % (t, t))


@needs_native
def test_loads_objects_and_text():
    rng = random.Random(6)
    for _ in range(20000):
        o = _rand_obj(rng)
        try:
            text = json.dumps(o, separators=rng.choice([SEP, (", ", ": ")]), ensure_ascii=rng.random() < 0.7)
        except (TypeError, ValueError):
            continue
        _both_loads(text)
        try:
            _both_loads(text.encode("utf-8"))
        except UnicodeEncodeError:
            pass
    for text in ('{"a":1,"a":2}', ' \t\n{"a" : [ 1 , 2 ] }\r\n', "[]", "{}", '""', '"\\u0041\\ud83d\\ude00\\/"', '"\\ud83d"',
                 '"\\udc00x"', '"a\\qb"', '"a\nb"', '"\\u12"', '{"a":1,}', "[1,]", "[1 2]", '{"a" 1}', "{1:2}", "", " ",
                 "nul", "true false", "[1]x", '"abc', "[[[[", '{"a":{"b":{"c":[1,{"d":null}]}}}', "﻿[1]"):
        _both_loads(text)
        _both_loads(text.encode("utf-8", "surrogatepass"))
    for raw in (b"\xff", b'"\xff"', b'"\xc3"', b'"\xed\xa0\x80"', b"\xef\xbb\xbf[1]", "[1]".encode("utf-16"),
                bytearray(b"[1,2]"), memoryview(b'{"a":1}')):
        _both_loads(raw)
    deep = "[" * 200 + "]" * 200
    assert _same(N.loads(deep), json.loads(deep))


@needs_native
def test_loads_survives_damaged_input():
    """Lines cut short and with bytes changed: the same result as json.loads or an error, never anything else
    (the reader's bounds checks; build with -fsanitize=address to make this test a memory check)."""
    rng = random.Random(7)
    base = []
    for _ in range(300):
        try:
            base.append(json.dumps(_rand_obj(rng), separators=SEP).encode())
        except (TypeError, ValueError):
            pass
    road = json.dumps({"type": "road", "seq": 5, "ts_mono_ns": 3226220852221, "plan": {"x": [0.5 * i for i in range(33)]},
                       "lane_lines": [{"x": [1.25 * i for i in range(33)], "prob": 0.9871}] * 4, "s": "é\n"}).encode()
    base.append(road)
    for _ in range(40000):
        b = bytearray(rng.choice(base))
        if not b:
            continue
        k = rng.randrange(4)
        if k == 0:
            del b[rng.randrange(len(b)):]
        elif k == 1:
            b[rng.randrange(len(b))] = rng.randrange(256)
        elif k == 2:
            i = rng.randrange(len(b))
            b[i:i] = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 4)))
        else:
            i = rng.randrange(len(b))
            del b[i:i + rng.randrange(1, 5)]
        try:
            _both_loads(bytes(b))
        except UnicodeDecodeError:
            # json.loads raises UnicodeDecodeError (a ValueError) for bytes that are not text: so does the native path
            pass


# ---- held to what the references gave (tests/native_cases.py, tests/data/native_golden.json.gz) ---------------
sys.path.insert(0, os.path.join(HERE, "tests"))
import native_cases as K  # noqa: E402

GOLD = K.load_golden()


def _held(name, cases, fn):
    """Every case's result digest equal to the reference's; the inputs the same as when it was recorded."""
    cases = list(cases)
    g = GOLD[name]
    assert K.digest(cases) == g["inputs"], "the cases of %s are not the recorded ones (tests/native_cases.py)" % name
    bad = [i for i, c in enumerate(cases) if K.digest(K.outcome(fn, *c)) != g["out"][i]]
    assert not bad, "%s: %d of %d cases differ from the reference, first %d: %r" % (name, len(bad), len(cases), bad[0],
                                                                                    cases[bad[0]])


@needs_native
def test_golden_view():
    from chris_integrator.trackers import view as V
    _held("interp", K.cases_interp(), V._interp_all)
    _held("view_prep", K.cases_view_prep(), V._prep)
    _held("view_blend", K.cases_view_blend(), V._blend)


@needs_json
def test_golden_road_line():
    from chris_integrator import fastjson as F
    from chris_integrator.io import recorder as R
    _held("trim_road", ((b,) for b in K.cases_trim()), F.trim_road)
    _held("slim_road_line", ((m,) for m in K.cases_slim()), R.slim_road_line)


@needs_native
def test_golden_road_geometry():
    from chris_integrator.trackers.roadgeom import RoadGeometryFusion as G
    _held("geom_centre", K.cases_geom(),
          lambda x, y, c, reach: G._centre(x, y, c, "vb", 3.0, 7.5) if reach is None else G._centre_vision(x, y, c, reach))
    g = G()

    def reach(x, sd, rstd, rmax, rmin):
        g.cfg.update(reach_std_m=rstd, reach_max_m=rmax, reach_min_m=rmin)
        return g._reach(x, sd)
    _held("geom_reach", K.cases_reach(), reach)


@needs_native
def test_golden_lane_tracker():
    from chris_integrator.trackers import lanes as L
    tr = L.RoadGeometryTracker()
    _held("lanes_measure", ((r,) for r in K.cases_lanes_measure()), tr._measure)

    def lane_arrays(c, x, zy, zs, prob, y, p, upd):
        cfg = dict(L.DEFAULT_CFG, **c)
        fit = L._fit(cfg, x, zs, zy, prob, y, p)
        y1, p1 = y.copy(), p.copy()
        if upd:
            L._update(y1, p1, fit[0], fit[2], upd)
        p2 = p1.copy()
        p2[0, 0] = -1.0
        rounded = [L._round2(y1), L._round2(p2, True), L._round2(x)]
        pred = p1.copy()
        C.lanes_predict(pred, x, cfg["q0"], cfg["q1"], 0.07, cfg["max_std"] ** 2)
        return fit, (y1, p1), rounded, pred
    _held("lanes_arrays", K.cases_lanes_arrays(), lane_arrays)
    frames = K.lane_frames()
    assert K.digest(frames) == GOLD["lane_messages"]["inputs"]
    t = L.RoadGeometryTracker()
    for i, (road, es) in enumerate(frames):
        try:
            got = json.dumps(t.on_road(road, es))
        except (TypeError, ValueError) as e:
            got = "raised %s" % type(e).__name__
        assert K.digest(got) == GOLD["lane_messages"]["out"][i], (i, got[:200])


@needs_native
def test_golden_filter():
    from chris_integrator.core import ekf as KF
    pc = list(K.cases_predict())
    g = GOLD["predict"]
    assert K.digest(pc) == g["inputs"]
    for i, (naux, x, P, dt) in enumerate(pc):
        m = K.ego_model(None, naux)
        xn, Pn, (Fn, rn) = m.predict(x, P, dt)
        xr, Fr, rr = g["state"][i]                     # the state and transition: elementwise, with the C library's
        np.testing.assert_allclose(xn, xr, rtol=1e-14, atol=1e-14)               # sin, cos, exp (the same bits
        np.testing.assert_allclose(Fn, Fr, rtol=1e-14, atol=1e-14)               # with the same library)
        assert (rr is None) == (rn is None) and (rr is None or np.allclose(rn, rr, rtol=1e-14, atol=0.0))
        Pr = np.array(g["values"][i])
        scale = np.sqrt(np.outer(np.diag(Pr), np.diag(Pr)))
        assert float(np.max(np.abs(Pr - Pn) / scale)) < 1e-12, i              # the covariance: to rounding
    rc = list(K.cases_rows())
    g = GOLD["rows"]
    assert K.digest(rc) == g["inputs"]
    for i, (P, x, idx, coef, nu, var, freeze) in enumerate(rc):
        ph, s0 = KF._project(P, idx, coef, np.empty(P.shape[0]))
        if g["values"][i] is not None:                 # a row over several states: the projection to rounding
            ph_r, s0_r = np.array(g["values"][i][0]), g["values"][i][1]
            np.testing.assert_allclose(ph, ph_r, rtol=1e-12, atol=1e-12)
            assert abs(s0 - s0_r) <= 1e-12 * max(abs(s0_r), 1.0)
            ph, s0 = ph_r, s0_r
        x1 = x.copy()
        P1 = KF._apply(x1, P, ph.copy(), s0 + var, nu, freeze, False)
        assert P1 is not P
        S = KF._sym(P1 + 1e-3 * np.triu(P1, 1), True)
        want = (ph, s0, x1, P1, S) if g["values"][i] is None else (x1, P1, S)
        assert K.digest(want) == g["out"][i], i
    tc = list(K.cases_rts())
    g = GOLD["rts"]
    assert K.digest(tc) == g["inputs"]
    for i, (x, P, dt, noise, wrap) in enumerate(tc):
        m = K.ego_model(None, 17)
        xp, Pp, Fm = m.predict(x, P, dt)
        xs = xp + noise
        if wrap:
            xs[2] += 2 * np.pi
        Ps = Pp - 0.1 * np.diag(np.diag(Pp))
        xn, Pn = m.rts(Pp, Fm, P, x, xp, xs, Ps, wrap)
        np.testing.assert_allclose(xn, g["values"][i][0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(Pn, g["values"][i][1], rtol=1e-8, atol=1e-9)


# ---- round, interpolation ----------------------------------------------------------------------------------
@needs_native
def test_round_is_builtin_round():
    rng = random.Random(8)
    for _ in range(400000):
        n = rng.randrange(0, 8)
        k = rng.randrange(5)
        if k == 0:
            x = rng.uniform(-500.0, 500.0)
        elif k == 1:
            x = (rng.randrange(-10 ** 6, 10 ** 6) + 0.5) / 10.0 ** n                   # decimal ties
        elif k == 2:
            x = _rand_double(rng)
        elif k == 3:
            x = math.nextafter((rng.randrange(-10 ** 5, 10 ** 5) + 0.5) / 10.0 ** n, rng.choice([-math.inf, math.inf]))
        else:
            x = rng.randrange(-10 ** 4, 10 ** 4) / 10.0 ** rng.randrange(0, 5)
        if not math.isfinite(x):
            continue
        a, b = C.round(x, n), round(x, n)
        assert type(a) is type(b) and a == b and math.copysign(1.0, a) == math.copysign(1.0, b), (x, n)
    for args in ((5, 2), (5,), (2.5,), (3.5, None), (2.675, -1), (1234.5, -2), (True, 1), (np.float64(2.675), 2),
                 (1e300, 3), (float("inf"), 2), (1.5, 30)):
        a, b = C.round(*args), round(*args)
        assert type(a) is type(b) and (a == b)
    assert math.isnan(C.round(float("nan"), 2))
    for _ in range(2000):
        n = rng.randrange(0, 6)
        vals = [rng.choice([None, 5, True, rng.uniform(-300, 300), (rng.randrange(-10 ** 5, 10 ** 5) + 0.5) / 10.0 ** n,
                            -1e-9, 0.0, -0.0, 1e300]) for _ in range(rng.randrange(0, 30))]
        a, b = C.round_list(vals, n), [v if v is None else round(v, n) for v in vals]
        assert len(a) == len(b) and all(type(u) is type(v) and repr(u) == repr(v) for u, v in zip(a, b)), (vals, n)
    for bad in ((), ("x", 2), (1.5, "a")):
        with pytest.raises(TypeError):
            C.round(*bad)


# ---- the estimator's filter: what the row and smoother steps are given ------------------------------------------
def test_filter_steps_take_lists_and_views():
    """A row model may return lists, and an array may be a view: the steps make contiguous arrays of them and give
    what they give for those; an index beyond the state is an error."""
    from chris_integrator.core import ekf as KF
    rng = random.Random(21)
    n = 27
    P = K._cov(rng, n)
    ph, s0 = KF._project(P, [3], [1.0])
    assert np.array_equal(ph, P[:, 3]) and s0 == P[3, 3]
    V = P[::-1]
    ph, s0 = KF._project(V, np.array([3], dtype=np.intp), np.array([1.0]))
    assert np.array_equal(ph, V[:, 3])
    with pytest.raises(IndexError):
        KF._project(P, np.array([99], dtype=np.intp), np.array([1.0]))
    x = np.arange(n, dtype=float)
    P1 = KF._apply(x, V, ph, s0 + 1.0, 0.5, [1, 2], False)
    x2 = np.arange(n, dtype=float)
    P2 = KF._apply(x2, np.ascontiguousarray(V), ph, s0 + 1.0, 0.5, np.array([1, 2], dtype=np.intp), False)
    assert np.array_equal(x, x2) and np.array_equal(P1, P2)
    assert np.array_equal(KF._sym(V, False), 0.5 * (V + V.T))


def test_smoother_step_singular_and_wrapped():
    rng = random.Random(30)
    m = K.ego_model(None, 17)
    n = m.n
    x, P = K.ego_state(rng, m)
    xp, Pp, F = m.predict(x, P, 0.05)
    xs = xp + 0.01
    Ps = Pp - 0.1 * np.diag(np.diag(Pp))
    xn, _ = m.rts(Pp, F, P, x, xp, xs, Ps, ())
    assert m.rts(np.zeros((n, n)), F, P, x, xp, xs, Ps, ()) is False                   # singular: the caller keeps the filtered state
    xw, _ = m.rts(Pp, F, P, x, xp, xs + np.eye(n)[2] * 2 * math.pi, Ps, (2,))          # a wrapped angle difference
    np.testing.assert_allclose(xw, xn, rtol=1e-9, atol=1e-9)
    with pytest.raises(ValueError):
        m.predict(x[:-1], P, 0.05)




# ---- the lane tracker's array steps ------------------------------------------------------------------------


# ---- the radar's motion model ------------------------------------------------------------------------------
needs_radar = pytest.mark.skipif(N.radar is None, reason="the native radar part is not in use in this process")


def _radar_batch(g, B):
    """B objects' hypotheses with the corners in them: speeds above every kind's top speed and below 1 m/s,
    probabilities with zeros and certainties, dt behind, at and ahead of the objects' time (and a dt so small
    that nothing flows into a hypothesis), kinds known and not."""
    from chris_integrator.radar import models as M

    def cov():
        a = g.standard_normal((8, 8)) * 10.0 ** g.uniform(-2, 1, 8)
        P = a @ a.T + np.diag(10.0 ** g.uniform(-4, 1, 8))
        return 0.5 * (P + P.T)
    xs = g.standard_normal((B, 3, 8)) * np.array([30.0, 30.0, 10.0, 10.0, 2.0, 2.0, 0.01, 0.3])
    r = g.random()
    if r < 0.3:
        xs[:, :, 2:4] *= 10.0
    elif r < 0.6:
        xs[:, :, 2:4] *= 0.05
    Ps = np.array([[cov() for _ in range(3)] for _ in range(B)])
    mu = g.random((B, 3))
    for b in range(B):
        r = g.random()
        if r < 0.2:
            mu[b, g.integers(3)] = 0.0
        elif r < 0.35:
            mu[b] = np.eye(3)[g.integers(3)]
    mu /= mu.sum(axis=1, keepdims=True)
    xf, Pf = g.standard_normal((B, 8)) * 10.0, np.array([cov() for _ in range(B)])
    dt = g.choice([-0.3, -0.05, 0.0, 1e-14, 0.033, 0.1, 0.5, 1.0, 3.0], B) * (1.0 + 0.1 * g.random(B))
    dt[g.random(B) < 0.15] = 0.0
    kinds = list(M.KIN) + ["zeppelin"]
    labels = [kinds[i] for i in g.integers(len(kinds), size=B)]
    return xs, Ps, mu, xf, Pf, dt, labels


def _to_rounding(ref, got, block):
    """The largest difference of two batched arrays, relative to the largest magnitude of each object's (or
    hypothesis') block (at least 1)."""
    ref, got = np.asarray(ref), np.asarray(got)
    lead = ref.shape[:block]
    scale = np.maximum(np.abs(ref).reshape(lead + (-1,)).max(axis=-1), 1.0)
    return float((np.abs(ref - got).reshape(lead + (-1,)).max(axis=-1) / scale).max())


@needs_native
def test_radar_evolve_agrees_with_numpy():
    """Interaction, prediction (speed bound included), combination and the free estimate's prediction of the
    native core against the numpy reference (models._evolve_py), on stacked arrays and on lists of per-object
    arrays: to rounding (1e-12 of each block's size)."""
    from chris_integrator.radar import models as M
    g = np.random.default_rng(5)
    par = M._native_par(M.CFG)
    worst = 0.0
    seen = {"over": 0, "low": 0, "back": 0, "still": 0}
    for it in range(1500):
        B = int(g.integers(1, 9))
        xs, Ps, mu, xf, Pf, dt, labels = _radar_batch(g, B)
        ref = M._evolve_py(xs, Ps, mu, xf, Pf, dt, labels, M.CFG, True, True)
        got = (np.empty((B, 3, 8)), np.empty((B, 3, 8, 8)), np.empty((B, 3)), np.empty((B, 8)), np.empty((B, 8, 8)),
               np.empty((B, 8)), np.empty((B, 8, 8)))
        kin = [M._KIN4.get(l) or M._KIN4[None] for l in labels]
        if it % 2:
            C.radar_evolve(xs, Ps, mu, xf, Pf, dt, np.array(kin), np.array(par), *got)
        else:
            C.radar_evolve(list(xs), list(Ps), list(mu), list(xf), list(Pf), list(dt), kin, par, *got)
        for a, b, blk in zip(ref, got, (2, 2, 1, 1, 1, 1, 1)):
            worst = max(worst, _to_rounding(a, b, blk))
        vmax = np.array([k[2] for k in kin])
        seen["over"] += int((np.hypot(ref[0][:, 1:, 2], ref[0][:, 1:, 3]) >= vmax[:, None] * (1 - 1e-12)).any())
        seen["low"] += int(((dt > 0) & (dt < 1e-12) & (mu.min(axis=1) == 0.0)).any())
        seen["back"] += int((dt < 0).any())
        seen["still"] += int((np.hypot(xs[:, 1:, 2], xs[:, 1:, 3]) < 1.0).any())
    assert worst < 1e-12, worst
    assert min(seen.values()) > 20, seen
    # without the optional outputs: the same hypotheses
    xs, Ps, mu, xf, Pf, dt, labels = _radar_batch(g, 4)
    a = M.evolve_b(xs, Ps, mu, xf, Pf, dt, labels)
    assert a[3] is None and a[5] is None
    b = M._evolve_py(xs, Ps, mu, xf, Pf, dt, labels)
    assert _to_rounding(b[1], a[1], 2) < 1e-12 and _to_rounding(b[0], a[0], 2) < 1e-12


@needs_native
def test_radar_p_still_and_combine_agree_with_numpy():
    from chris_integrator.radar import models as M
    g = np.random.default_rng(6)
    for _ in range(500):
        B = int(g.integers(1, 9))
        xs, Ps, mu, xf, Pf, dt, labels = _radar_batch(g, B)
        v0 = M.kin_arrays(labels)[3]
        q = M.still_b(xs[:, 1:], Ps[:, 1:], v0[:, None])
        ref = np.minimum(1.0, mu[:, 0] + mu[:, 1] * q[:, 0] + mu[:, 2] * q[:, 1])
        got = C.radar_p_still(list(xs), list(Ps), list(mu), list(v0))
        assert isinstance(got, list) and len(got) == B
        assert float(np.max(np.abs(ref - np.array(got)))) < 1e-12
        x0, P0 = M.combine_b(xs, Ps, mu)
        x1, P1 = np.empty((B, 8)), np.empty((B, 8, 8))
        C.radar_combine(xs, Ps, mu, x1, P1)
        assert _to_rounding(x0, x1, 1) < 1e-12 and _to_rounding(P0, P1, 1) < 1e-12
        x2, P2 = M.combine(xs[0], Ps[0], mu[0])
        assert _to_rounding(x0[:1], x2[None], 1) < 1e-12 and _to_rounding(P0[:1], P2[None], 1) < 1e-12


@needs_native
def test_radar_leaves_what_is_not_finite_to_the_reference():
    """A state that is not finite (an object gone wrong) is the reference's: the native call says so and the
    switch point runs numpy, whose NaN and inf are what the tracker checks for."""
    from chris_integrator.radar import models as M
    g = np.random.default_rng(7)
    xs, Ps, mu, xf, Pf, dt, labels = _radar_batch(g, 3)
    Ps[1, 2, 0, 0] = np.inf
    xs[2, 1, 3] = np.nan
    out = (np.empty((3, 3, 8)), np.empty((3, 3, 8, 8)), np.empty((3, 3)), None, None, None, None)
    with pytest.raises(C.Fallback):
        C.radar_evolve(xs, Ps, mu, xf, Pf, dt, [M._KIN4["car"]] * 3, M._native_par(M.CFG), *out)
    with pytest.raises(C.Fallback):
        C.radar_p_still(xs, Ps, mu, [0.7] * 3)
    with np.errstate(invalid="ignore", over="ignore"):
        ref = M._evolve_py(xs, Ps, mu, xf, Pf, dt, labels, M.CFG, True, True)
        got = M.evolve_b(xs, Ps, mu, xf, Pf, dt, labels, M.CFG, True, True)
    for a, b in zip(ref, got):
        assert np.array_equal(a, b, equal_nan=True)
    # arrays it does not take (not float64, not contiguous): the reference, through the switch point
    with np.errstate(invalid="ignore", over="ignore"):
        got = M.evolve_b(xs.astype(np.float32), Ps, mu, xf, Pf, dt, labels)
        ref = M._evolve_py(xs.astype(np.float32), Ps, mu, xf, Pf, dt, labels)
    assert np.array_equal(got[0], ref[0], equal_nan=True)


RADAR = r"""
import json, sys
sys.path.insert(0, %r)
import os
from chris_integrator.mixer import MessageClock
if os.environ.get("RADAR_REF") == "1":                   # the radar's numpy reference
    from chris_integrator.radar import models as _M
    _M._NR = None
from chris_integrator.sim import default_sensors
from chris_integrator.sim import roaduser_sim as rs
from chris_integrator.stack import build
out = {"radar_native": os.environ.get("RADAR_REF") != "1", "runs": {}}
for name, seed, night in (("parked_street", 1, True), ("adjacent", 1, False), ("stopped_brake_25", 1, False)):
    w = rs.scenario(name, seed=seed, night=night)
    det = rs.DetectorSim(seed=1000 + seed, night=night)
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
    sensors.append(det)
    st = build(cfg={"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}}, mapper=None, clock=MessageClock())
    box = st.mixer.slots["radar"].plugin.box
    msgs, frame = [], box.frame
    def grab(ts, frame=frame, msgs=msgs):
        m = frame(ts)
        if m is not None:
            msgs.append(m)
        return m
    box.frame = grab
    sent = False
    for _, m in w.messages(sensors):
        if not sent:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            sent = True
        st.deliver(m)
    out["runs"][name] = msgs
print(json.dumps(out))
"""


def _msg_diffs(u, v, path, out):
    if isinstance(u, dict) and isinstance(v, dict):
        if list(u) != list(v):
            out.append((path, "keys"))
            return
        for k in u:
            _msg_diffs(u[k], v[k], path + "." + k, out)
    elif isinstance(u, list) and isinstance(v, list):
        if len(u) != len(v):
            out.append((path, "length"))
            return
        for i, (a, b) in enumerate(zip(u, v)):
            _msg_diffs(a, b, "%s[%d]" % (path, i), out)
    elif type(u) is not type(v) or repr(u) != repr(v):
        out.append((path, u, v))


def _last_digit(*vs):
    """One unit of the last printed digit of the numbers (the finer of them): 0.01 for 7.47, 0.001 for 0.125."""
    return min(10.0 ** -len(repr(v).split(".")[1]) if "." in repr(v) and "e" not in repr(v) else 1.0 for v in vs)


@needs_radar
def test_radar_messages_native_against_reference():
    """Simulated drives through the whole stack with the radar part native and not (every other part the same):
    the radar's messages agree except, at most, in a last printed digit or the sign of a zero, in a handful of
    messages (none on the development machine)."""
    res = {}
    for ref in ("1", "0"):
        out = subprocess.run([sys.executable, "-c", RADAR % HERE], cwd=HERE, env=dict(os.environ, RADAR_REF=ref),
                             capture_output=True, text=True, check=True).stdout
        res[ref == "0"] = json.loads(out)
    assert not res[False]["radar_native"] and res[True]["radar_native"]
    for name, a in res[False]["runs"].items():
        b = res[True]["runs"][name]
        assert len(a) == len(b) and len(a) > 150, name
        bad = 0
        for u, v in zip(a, b):
            d = []
            _msg_diffs(u, v, "", d)
            for item in d:
                assert len(item) == 3 and isinstance(item[1], float) and isinstance(item[2], float), (name, item)
                assert abs(item[1] - item[2]) <= 1.01 * _last_digit(item[1], item[2]), (name, item)
            bad += bool(d)
        assert bad <= max(3, len(a) // 30), (name, bad, len(a))


# ---- the radar, phase 2: sensors, association, update, output, upkeep ---------------------------------------
def _cameras(g):
    from chris_integrator.radar.sensors import CameraBoxes
    cov = None if g.random() < 0.3 else {"bearing_min": -0.6, "bearing_max": 0.55, "ground_min_m": float(g.uniform(4, 12))}
    s = CameraBoxes(cov, None, 975.0, float(g.uniform(1.0, 1.6)), None if g.random() < 0.3 else float(g.choice([9.0, 14.0, math.inf])))
    s.bias = float(g.normal() * 0.01)
    s.extraneous = float(g.uniform(0.1, 3.0))
    return s


@needs_radar
def test_radar_measurement_functions_are_the_python_bit_for_bit():
    """sensors.py `h` (every camera kind with and without the scale-rate row, the microwave radar), coverage,
    p_detect and density in C against their Python: the same bits (math.hypot as CPython computes it)."""
    from chris_integrator.radar.sensors import MicrowaveRadar, kind_code
    g = np.random.default_rng(11)
    kinds = ["b", "rb", "rbh", "bt", "bs", "rbs", "rbhs", "bts", "s"]
    for it in range(30000):
        s = _cameras(g)
        x = g.standard_normal(8) * [30, 30, 10, 10, 2, 2, 0.01, 0.3]
        if it % 7 == 0:
            x[:2] *= 1e-3                                # at the car: the 0.5 m floor of the range
        pose = tuple(g.standard_normal(6) * [10, 10, 3, 10, 10, 0.3])
        k = kinds[it % len(kinds)]
        d = {"kind": k, "sdt": float(g.choice([0.0, 1e-4, 0.2, 0.8]))}
        z1, H1 = s.h(x, pose, d)
        z2, H2 = s.h_py(x, pose, d)
        assert z1.shape == z2.shape and H1.shape == H2.shape and np.array_equal(z1, z2) and np.array_equal(H1, H2), (k, z1, z2)
        r = MicrowaveRadar((float(g.normal()), float(g.normal()), float(g.normal() * 0.3)))
        z1, H1 = r.h(x, pose)
        z2, H2 = r.h_py(x, pose)
        assert np.array_equal(z1, z2) and np.array_equal(H1, H2)
        assert kind_code(k) == ({"b": 0, "rb": 1, "rbh": 2, "bt": 3, "": 4}[k.rstrip("s") if k != "s" else ""] + 8 * k.endswith("s"))
    # math.hypot (CPython's own algorithm, not libm's) through the camera's bearing row at awkward magnitudes
    s = _cameras(g)
    for _ in range(20000):
        x = np.zeros(8)
        x[:2] = g.standard_normal(2) * 10.0 ** g.uniform(-1, 4, 2)
        z1, H1 = s.h(x, (0.0,) * 6, {"kind": "b"})
        z2, H2 = s.h_py(x, (0.0,) * 6, {"kind": "b"})
        assert np.array_equal(H1, H2)


@needs_radar
def test_radar_marginals_and_assignment_are_the_python():
    """tracker.marginals (loopy belief propagation, numpy's pairwise sums) and tracker.assign (groups, the
    Hungarian of core.objects): the same bits, the same pairs in the same order."""
    from chris_integrator.radar import tracker as T
    g = np.random.default_rng(12)
    for it in range(4000):
        nt, nd = int(g.integers(0, 14)), int(g.integers(0, 14))
        L = g.random((nt, nd)) * 10.0 ** g.uniform(-3, 3, (nt, nd))
        L[g.random((nt, nd)) < 0.6] = 0.0
        if it % 9 == 0:
            L[:] = 0.0
        out = np.empty((nt, nd))
        C.radar_marginals(np.ascontiguousarray(L), out)
        assert np.array_equal(out, T.marginals(L)), (nt, nd)
        G = np.where(L > 1.0, np.log(np.where(L > 1.0, L, 1.0)), -np.inf)
        if it % 5 == 0:                                  # ties
            G = np.where(np.isfinite(G), np.round(G), -np.inf)
            G[G == 0.0] = -np.inf
        assert C.radar_assign(np.ascontiguousarray(G)) == [(int(i), int(j)) for i, j in T.assign(G)], (nt, nd)


def _clone_tracker(tr):
    """An independent copy of a tracker: its state array copied, every object re-made on it."""
    import copy
    from chris_integrator.radar import tracker as T
    new = copy.copy(tr)
    new.store = T.State(len(tr.store.a))
    new.store.a[:] = tr.store.a
    new.store.free = list(tr.store.free)
    objs = []
    for o in tr.objects:
        q = T.Obj.__new__(T.Obj)
        for k in T.Obj.__slots__:
            if k.startswith("_"):
                continue
            v = getattr(o, k)
            setattr(q, k, copy.deepcopy(v) if isinstance(v, (dict, set, list)) or hasattr(v, "maxlen") else v)
        q.bind(new.store)
        objs.append(q)
    new.objects = objs
    for k in ("stats", "sensors", "bias", "new_rate", "unexplained", "scan_times", "real_ids", "score_hist", "score_odds"):
        setattr(new, k, copy.deepcopy(getattr(tr, k)))
    return new


def _rows_close(a, b, tol=1e-7):
    """Two trackers' objects: the same ids and discrete record, numbers to rounding (relative to each block's
    largest magnitude: a state, a covariance, the probabilities). Rounding, not 1e-16: one update can lose digits
    (a covariance update that takes most of a variance away subtracts nearly equal numbers; small systems solved
    by LAPACK in the reference and by elimination here), seen up to 2e-9 on the simulated drives."""
    assert [o.id for o in a.objects] == [o.id for o in b.objects]
    worst, where = 0.0, None
    for o, q in zip(a.objects, b.objects):
        assert (o.state, o.hits, o.verdict, sorted(o.sources), o.n_checks, sorted(o.flags)) == \
               (q.state, q.hits, q.verdict, sorted(q.sources), q.n_checks, sorted(q.flags)), o.id
        for k in ("xs", "Ps", "mu", "xf", "Pf", "x", "P"):
            u, v = getattr(o, k), getattr(q, k)
            blocks = (u.reshape(3, -1), v.reshape(3, -1)) if k in ("xs", "Ps") else (u.reshape(1, -1), v.reshape(1, -1))
            scale = np.maximum(np.abs(blocks[1]).max(axis=1, keepdims=True), 1.0)
            e = float((np.abs(blocks[0] - blocks[1]) / scale).max())
            if e > worst:
                worst, where = e, (o.id, k)
        e = abs(o.logodds - q.logodds) / max(1.0, abs(q.logodds))
        if e > worst:
            worst, where = e, (o.id, "logodds")
    assert worst < tol, (worst, where)
    return worst


@needs_radar
def test_radar_scan_by_scan_native_against_reference(monkeypatch):
    """On the states of simulated drives (camera boxes with and without the scale-rate row, a microwave radar),
    every scan's processing, every message's upkeep and every message's numbers: native against the Python
    reference on a copy of the same tracker, before the two can drift apart. The same pairs, objects, states
    and verdicts; numbers to rounding."""
    from chris_integrator.mixer import MessageClock
    from chris_integrator.radar import box as Bx, sensors as Sn, tracker as T
    from chris_integrator.sim import default_sensors
    from chris_integrator.sim import roaduser_sim as rs
    from chris_integrator.stack import build
    seen = {"scans": 0, "pairs": 0, "frames": 0, "objects": 0}

    def reference(fn):                                   # run fn with the Python reference of the radar
        saved = (T._NR, Sn._NR, Bx._NR)
        T._NR = Sn._NR = Bx._NR = None
        try:
            return fn()
        finally:
            T._NR, Sn._NR, Bx._NR = saved

    SC = {"px": 0.08, "rel": 0.002, "p": 0.78, "min_dt": 0.2, "max_dt": 1.0, "max_std": 0.06}
    errors = []

    def guarded(fn):                                     # the mixer logs a plug-in's exception and goes on
        def call(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:
                errors.append(repr(e))
                raise
        return call

    for name, seed, night, scale in (("curve_parked", 1, False, False), ("parked_street", 1, True, True),
                                     ("junction", 1, False, True), ("adjacent", 1, False, False)):
        w = rs.scenario(name, seed=seed, night=night)
        det = rs.DetectorSim(seed=1000 + seed, night=night, params={"scale_rate": SC} if scale else None)
        sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True},
                                  gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
        sensors.append(det)
        st = build(cfg={"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}}, mapper=None, clock=MessageClock())
        box = st.mixer.slots["radar"].plugin.box
        tr = box.tracker
        process, manage, numbers = tr.process, tr.manage, box._numbers

        def proc(src, t, sensor, dets, process=process, tr=tr):
            import copy
            twin = _clone_tracker(tr)
            dets2 = copy.deepcopy(dets)
            sen2 = copy.copy(sensor)
            before = tr.stats.get(src, {}).get("assigned", 0)
            r = process(src, t, sensor, dets)
            reference(lambda: T.Tracker.process(twin, src, t, sen2, dets2))
            _rows_close(tr, twin)
            seen["scans"] += 1
            seen["pairs"] += tr.stats[src]["assigned"] - before
            return r

        def mng(T_, manage=manage, tr=tr):
            twin = _clone_tracker(tr)
            manage(T_)
            reference(lambda: T.Tracker.manage(twin, T_))
            _rows_close(tr, twin)
            for o, q in zip(tr.objects, twin.objects):
                assert o.merge_t.keys() == q.merge_t.keys()

        tr.process, tr.manage = guarded(proc), guarded(mng)
        frame = box.frame

        def frm(ts, frame=frame, box=box):
            out = frame(ts)
            if out is not None and out["objects"]:
                # the numbers of the objects as the reference computes them, against the message's
                T_ = ts / 1e9
                X, Y, psi = box.ego.pose_at(T_)
                es = box.ego.state_at(T_)
                shown = [o for o in box.tracker.objects if o.state != "tentative"]
                ref = reference(lambda: numbers(T_, shown, X, Y, psi, es[3], es[4]))
                byid = {r["id"]: r for r in out["objects"]}
                for o, r in zip(shown, ref):
                    m = byid[o.id]
                    got = (m["x_m"], m["y_m"], m["vx_mps"], m["vy_mps"], m["ax_mps2"], m["ay_mps2"], m["speed_mps"],
                           m["heading_rad"], m["v_rel_mps"], m["std"], m["models"], m["p_stationary"], m["angle_offset_deg"],
                           m["predict"], None, m["path_offset_m"], None, None, None, m["ttc_s"], m["predict_path"])
                    for k, (u, v) in enumerate(zip(got, r)):
                        if k in (14, 16, 17, 18):
                            continue
                        d = []
                        _msg_diffs(v, u, "", d)
                        for item in d:
                            assert len(item) == 3 and abs(item[1] - item[2]) <= 1.01 * _last_digit(item[1], item[2]), (k, item)
                    assert m["p_threat"] == round(r[18], 3) or abs(m["p_threat"] - round(r[18], 3)) <= 0.0011
                    seen["objects"] += 1
                seen["frames"] += 1
            return out

        box.frame = guarded(frm)
        sent = False
        for _, m in w.messages(sensors):
            if not sent:
                st.deliver(det.calib_msg(m["ts_mono_ns"]))
                sent = True
            st.deliver(m)
    assert not errors, errors[:5]
    assert seen["scans"] > 500 and seen["pairs"] > 1500 and seen["frames"] > 500 and seen["objects"] > 3000, seen


@needs_radar
def test_radar_microwave_scans_native_against_reference(monkeypatch):
    """A microwave radar's scans (range, azimuth, range rate) through the tracker: native against reference."""
    import copy
    from chris_integrator.radar import Radar, sensors as Sn, tracker as T
    g = np.random.default_rng(14)
    rd = Radar()
    NS = 1_000_000_000
    targets = [(30.0, 3.5, 0.0, 0.0), (55.0, -0.2, 18.0, 0.0), (12.0, -3.0, 0.0, 0.0), (80.0, 0.0, -20.0, 0.0)]
    n = 0
    for k in range(240):
        ts = 10 * NS + k * NS // 20
        rd.on_ego({"ts_mono_ns": ts, "speed_mps": 15.0, "yaw_rate_rps": 0.01, "accel_long_mps2": 0.0})
        if k % 2:
            continue
        t = (ts - 10 * NS) / NS
        objs = []
        for (x0, y0, vx, vy) in targets:
            x, y = x0 + (vx - 15.0) * t, y0 + vy * t
            r = math.hypot(x, y)
            if g.random() < 0.85 and r < 150:
                objs.append({"range_m": r + g.normal() * 0.2, "azimuth_deg": math.degrees(math.atan2(y, x)) + g.normal() * 0.3,
                             "range_rate_mps": (x * (vx - 15.0) + y * vy) / r + g.normal() * 0.1, "class": "car"})
        if g.random() < 0.3:
            objs.append({"range_m": float(g.uniform(5, 100)), "azimuth_deg": float(g.uniform(-40, 40)), "range_rate_mps": float(g.normal() * 5)})
        rd.on_objects({"ts_mono_ns": ts, "sensor": "front_radar", "kind": "radar", "objects": objs,
                       "mount": {"x_m": 2.0, "y_m": 0.0, "yaw_deg": 0.0}})
        tr = rd.tracker
        twin = _clone_tracker(tr)
        scans = list(rd.queue)
        rd.frame(ts)
        saved = (T._NR, Sn._NR)
        T._NR = Sn._NR = None
        try:
            for (t_, src, sen, dets) in scans:
                if t_ <= ts / NS + 1e-6:
                    T.Tracker.process(twin, src, t_, copy.copy(sen), copy.deepcopy(dets))
            T.Tracker.manage(twin, ts / NS)
        finally:
            T._NR, Sn._NR = saved
        _rows_close(tr, twin)
        n += len(tr.objects)
    assert n > 300


@needs_radar
def test_radar_prediction_peeks_are_the_full_evolution():
    """r_peek_part (radar_peek_pos, and the message's present and horizons) against the full evolution and
    combination (radar_peek): the same bits for the entries it computes, over states with the corners in them."""
    from chris_integrator.radar import models as M, tracker as T
    g = np.random.default_rng(15)
    st = T.State(64)
    par = M._native_par(M.CFG)
    for it in range(300):
        B = int(g.integers(1, 9))
        xs, Ps, mu, xf, Pf, dt, labels = _radar_batch(g, B)
        rows = list(range(B))
        for b in rows:
            r = st.a[b]
            r[0:24], r[24:216], r[216:219], r[219:227], r[227:291] = xs[b].ravel(), Ps[b].ravel(), mu[b], xf[b], Pf[b].ravel()
        kin = [M._KIN4.get(l) or M._KIN4[None] for l in labels]
        x, P, m = np.empty((B, 8)), np.empty((B, 8, 8)), np.empty((B, 3))
        dts = [d if abs(d) >= 1e-6 else 0.3 for d in dt]
        C.radar_peek(st.a, rows, dts, kin, par, x, P, m)
        out = np.empty((B, 6))
        C.radar_peek_pos(st.a, rows, dts, kin, par, out)
        assert np.array_equal(out[:, :2], x[:, :2]) and np.array_equal(out[:, 2:], P[:, :2, :2].reshape(B, 4)), it


def _golden():
    z = np.load(os.path.join(HERE, "tests", "data", "radar_golden.npz"))
    cases = json.loads(bytes(z["index"]).decode())

    def dec(v):
        if isinstance(v, dict) and "array" in v:
            return z[v["array"]].copy()
        if isinstance(v, list):
            return [dec(x) for x in v]
        if isinstance(v, dict):
            return {k: dec(x) for k, x in v.items()}
        return v
    return [dict(c, args=dec(c["args"]), ref=dec(c["ref"]), rows=dec(c.get("rows"))) for c in cases]


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0), initial=0.0))


def _rows_rel(a, b):
    """Rows of objects: the largest difference relative to each block's largest magnitude."""
    from chris_integrator.radar import tracker as T
    worst = 0.0
    for name, lo, hi, sh in T._OFF:
        u, v = a[:, lo:hi], b[:, lo:hi]
        if name in ("_xs", "_Ps"):
            u, v = u.reshape(len(a) * 3, -1), v.reshape(len(b) * 3, -1)
        scale = np.maximum(np.abs(v).max(axis=1, keepdims=True), 1.0)
        worst = max(worst, float((np.abs(u - v) / scale).max(initial=0.0)))
    return worst


@needs_native
def test_radar_golden():
    """The native radar functions against what the Python reference gave for the same inputs (frozen in
    tests/data/radar_golden.npz by tests/make_radar_golden.py while the reference existed): measurement functions,
    marginals and assignment the same bits; association, update, advance, peeks, upkeep, merge geometry, checks,
    registration, the common pitch and the message's numbers: the same discrete results, numbers to rounding."""
    seen = {}
    for c in _golden():
        fn, a, ref = c["fn"], list(c["args"]), c["ref"]
        seen[fn] = seen.get(fn, 0) + 1
        S = c["rows"]
        if S is not None:
            a[0] = S
        if fn == "radar_h":
            z, H = np.empty(4), np.empty((4, 8))
            m = C.radar_h(a[0], a[1], a[2], a[3], a[4], z, H)
            assert np.array_equal(z[:m], ref[0]) and np.array_equal(H[:m], np.reshape(ref[1], (m, 8)))
        elif fn == "radar_marginals":
            L = np.reshape(a[0], c["shape"])
            out = np.empty(L.shape)
            C.radar_marginals(np.ascontiguousarray(L), out)
            assert np.array_equal(out, np.reshape(ref, L.shape))
        elif fn == "radar_assign":
            assert C.radar_assign(np.ascontiguousarray(np.reshape(a[0], c["shape"]))) == [tuple(p) for p in ref]
        elif fn == "radar_scan":
            nt = len(a[1])
            pd, cov = np.empty(nt), np.empty(nt)
            D = np.reshape(a[5], (-1, 13)) if len(a[5]) else np.zeros((0, 13))
            pairs = C.radar_scan(S, a[1], a[2], a[3], a[4], D, a[6], a[7], a[8], pd, cov, None, None, None)
            assert [(p[0], p[1]) for p in pairs] == [(int(p[0]), int(p[1])) for p in ref[0]]
            for p, q in zip(pairs, ref[0]):
                assert _rel(p[2:], q[2:]) < 1e-9, (p, q)
            assert _rel(pd, ref[1]) < 1e-12 and _rel(cov, ref[2]) < 1e-12
        elif fn == "radar_update":
            out = C.radar_update(S, a[1], a[2], a[3], np.reshape(a[4], (-1, 13)), a[5], a[6])
            assert _rel(out, ref["out"]) < 1e-9
            assert _rows_rel(S[a[1]], np.reshape(ref["after"], (len(a[1]), -1))) < 1e-7
        elif fn == "radar_advance":
            C.radar_advance(S, a[1], a[2], a[3], a[4], a[5], a[6])
            assert _rows_rel(S[a[1]], np.reshape(ref["after"], (len(a[1]), -1))) < 1e-9
        elif fn == "radar_peek_pos":
            out = np.empty((len(a[1]), 6))
            C.radar_peek_pos(S, a[1], a[2], a[3], a[4], out)
            assert _rel(out, ref) < 1e-9
        elif fn == "radar_life":
            out = C.radar_life(S, *a[1:])
            for u, v in zip(out, ref):
                assert u[1] == v[1] and (u[2] is None) == (v[2] is None)
                assert _rel([x for x in u if x is not None and not isinstance(x, bool)],
                            [x for x in v if x is not None and not isinstance(x, bool)]) < 1e-9
        elif fn == "radar_merge":
            assert [tuple(r) for r in C.radar_merge(S, *a[1:])] == [tuple(r) for r in ref]
        elif fn == "radar_frame":
            out = C.radar_frame(S, *a[1:])
            for u, v in zip(out, ref):
                for k, (x, y) in enumerate(zip(u, v)):
                    if k in (14, 18):                     # the offset and the threat probability as they are
                        assert _rel(x, y) < 1e-9
                        continue
                    d = []
                    _msg_diffs(y, x, "", d)
                    for item in d:
                        assert len(item) == 3 and abs(item[1] - item[2]) <= 1.01 * _last_digit(item[1], item[2]), (k, item)
        elif fn == "radar_check":
            out = C.radar_check(S, *a[1:])
            assert (out is None) == (ref is None) and (out is None or _rel(out, ref) < 1e-9)
        elif fn == "radar_register":
            out = C.radar_register(S, *a[1:])
            assert _rel(out, ref["out"]) < 1e-9
            rows = sorted(set(a[1]) | set(a[3]))
            assert _rows_rel(S[rows], np.reshape(ref["after"], (len(rows), -1))) < 1e-9
        elif fn == "radar_common":
            assert _rel(C.radar_common(S, *a[1:]), ref) < 1e-9
        else:
            raise AssertionError("a golden case of an unknown function: " + fn)
    assert all(seen.get(k, 0) >= 3 for k in ("radar_h", "radar_marginals", "radar_assign", "radar_scan", "radar_update",
                                             "radar_advance", "radar_peek_pos", "radar_life", "radar_merge",
                                             "radar_frame", "radar_check", "radar_register", "radar_common")), seen


RADAR_DIGEST_RUNS = (("parked_street", 1, True, False), ("adjacent", 1, False, False), ("junction", 1, False, True))
# sha256 of the radar messages of RADAR_DIGEST_RUNS (json.dumps of the list) on the development machine (x86-64);
# made when the radar's Python reference still existed, from the native core then (whose messages agree with the
# reference's to a last printed digit: test_radar_messages_native_against_reference had both). They change when
# anything upstream of the radar changes its output on purpose: then write them anew (RADAR_DIGESTS=print shows them).
RADAR_DIGESTS = {"parked_street:1:1:0": "951cac781ef48e369501918b4603cd6720e7bef0f0036d485e423305cdfd68b2",
                 "adjacent:1:0:0": "c9103ca895e5bf8dffd9feb4a130e9ce79622fcb3800d9e9a6dae05002f625a1",
                 "junction:1:0:1": "f6eb08c0c2ef1810df74413a83a58d3e7ecf40c75c7aab28ac5229723513d5dc"}


def _radar_digests():
    import hashlib
    from chris_integrator.mixer import MessageClock
    from chris_integrator.sim import default_sensors
    from chris_integrator.sim import roaduser_sim as rs
    from chris_integrator.stack import build
    SC = {"px": 0.08, "rel": 0.002, "p": 0.78, "min_dt": 0.2, "max_dt": 1.0, "max_std": 0.06}
    out = {}
    for name, seed, night, scale in RADAR_DIGEST_RUNS:
        w = rs.scenario(name, seed=seed, night=night)
        det = rs.DetectorSim(seed=1000 + seed, night=night, params={"scale_rate": SC} if scale else None)
        sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True},
                                  gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
        sensors.append(det)
        st = build(cfg={"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}}, mapper=None, clock=MessageClock())
        box = st.mixer.slots["radar"].plugin.box
        msgs, frame = [], box.frame

        def grab(ts, frame=frame, msgs=msgs):
            m = frame(ts)
            if m is not None:
                msgs.append(m)
            return m
        box.frame = grab
        sent = False
        for _, m in w.messages(sensors):
            if not sent:
                st.deliver(det.calib_msg(m["ts_mono_ns"]))
                sent = True
            st.deliver(m)
        out["%s:%d:%d:%d" % (name, seed, night, scale)] = hashlib.sha256(json.dumps(msgs).encode()).hexdigest()
    return out


@needs_radar
@pytest.mark.skipif(os.uname().machine != "x86_64", reason="the digests are of the development machine's arithmetic")
def test_radar_messages_digests():
    """The radar's messages on simulated drives, byte for byte as they were when the Python reference existed."""
    got = _radar_digests()
    if os.environ.get("RADAR_DIGESTS") == "print":
        print(json.dumps(got, indent=1))
    assert got == RADAR_DIGESTS
