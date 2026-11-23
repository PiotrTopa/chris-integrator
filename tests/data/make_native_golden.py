"""Writes tests/data/native_golden.json.gz: what the pure-Python references gave for the cases of
tests/native_cases.py. Ran once, with CHRIS_NATIVE=0, on the tree that still held the references (lane
cost-rest, commit 43fc680 with this script and tests/native_cases.py as they are now); kept as the record of how
the file was made. It does not run on a later tree: the references are gone.

    CHRIS_NATIVE=0 python3 tests/data/make_native_golden.py
"""
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

import native_cases as K  # noqa: E402
from chris_integrator import native as N  # noqa: E402

assert not N.ACTIVE, "run with CHRIS_NATIVE=0: the references are what is recorded"

from chris_integrator import fastjson as F  # noqa: E402
from chris_integrator.core import ekf as KF  # noqa: E402
from chris_integrator.io import recorder as R  # noqa: E402
from chris_integrator.trackers import lanes as L  # noqa: E402
from chris_integrator.trackers import view as V  # noqa: E402
from chris_integrator.trackers.roadgeom import RoadGeometryFusion as G  # noqa: E402

gold = {}


def put(name, cases, fn):
    cases = list(cases)
    gold[name] = {"inputs": K.digest(cases), "out": [K.digest(K.outcome(fn, *c)) for c in cases]}
    print(name, len(cases))


put("interp", K.cases_interp(), V._interp_all_py)
put("view_prep", K.cases_view_prep(), V._prep_py)
put("view_blend", K.cases_view_blend(), V._blend_py)
put("trim_road", ((b,) for b in K.cases_trim()), F._trim_road_py)
put("slim_road_line", ((m,) for m in K.cases_slim()), lambda m: json.dumps(R.slim_road(m), separators=(",", ":")))
put("geom_centre", K.cases_geom(),
    lambda x, y, c, reach: G._centre_py(x, y, c, "vb", 3.0, 7.5) if reach is None else G._centre_vision_py(x, y, c, reach))
g = G()


def reach(x, sd, rstd, rmax, rmin):
    g.cfg.update(reach_std_m=rstd, reach_max_m=rmax, reach_min_m=rmin)
    return g._reach_py(x, sd)


put("geom_reach", K.cases_reach(), reach)
tr = L.RoadGeometryTracker()
put("lanes_measure", ((r,) for r in K.cases_lanes_measure()), tr._measure)


def lane_arrays(c, x, zy, zs, prob, y, p, upd):
    cfg = dict(L.DEFAULT_CFG, **c)
    fit = L._fit_py(cfg, x, zs, zy, prob, y, p)
    r, nu = fit[0], fit[2]
    y1, p1 = y.copy(), p.copy()
    if upd:
        L._update_py(y1, p1, r, nu, upd)
    p2 = p1.copy()
    p2[0, 0] = -1.0                                    # a negative variance: NaN from the root
    rounded = [L._round2_py(y1), L._round2_py(p2, True), L._round2_py(x)]
    pred = np.minimum(p1 + (cfg["q0"] + cfg["q1"] * x) ** 2 * 0.07, cfg["max_std"] ** 2)
    return fit, (y1, p1), rounded, pred


put("lanes_arrays", K.cases_lanes_arrays(), lane_arrays)


def lane_messages():
    t = L.RoadGeometryTracker()
    out = []
    for road, es in K.lane_frames():
        try:
            out.append(json.dumps(t.on_road(road, es)))
        except (TypeError, ValueError) as e:
            out.append("raised %s" % type(e).__name__)
    return out


msgs = lane_messages()
gold["lane_messages"] = {"inputs": K.digest(K.lane_frames()), "out": [K.digest(s) for s in msgs]}
print("lane_messages", len(msgs))

# the filter: exact parts as digests, sums of products as values
pc = list(K.cases_predict())
vals, state = [], []
for naux, x, P, dt in pc:                              # (sin, cos, exp of the C library: to the last bit or so)
    m = K.ego_model(None, naux)
    xr, Pr, (Fr, rr) = m._predict_py(x.copy(), P.copy(), dt)
    state.append([xr.tolist(), Fr.tolist(), None if rr is None else rr.tolist()])
    vals.append(Pr.tolist())
gold["predict"] = {"inputs": K.digest(pc), "state": state, "values": vals}
print("predict", len(pc))

rc = list(K.cases_rows())
outs, vals = [], []
for P, x, idx, coef, nu, var, freeze in rc:
    ph, s0 = KF._project_py(P, idx, coef)
    x1 = x.copy()
    P1 = KF._apply_py(x1, P, ph, s0 + var, nu, freeze, False)
    S = KF._sym_py(P1 + 1e-3 * np.triu(P1, 1), False)
    if len(idx) == 1:
        outs.append(K.digest((ph, s0, x1, P1, S)))
        vals.append(None)
    else:                                              # the projection to rounding; the rest from its value, exact
        outs.append(K.digest((x1, P1, S)))
        vals.append([ph.tolist(), s0])
gold["rows"] = {"inputs": K.digest(rc), "out": outs, "values": vals}
print("rows", len(rc))

tc = list(K.cases_rts())
vals = []
for x, P, dt, noise, wrap in tc:
    m = K.ego_model(None, 17)
    xp, Pp, Fm = m._predict_py(x, P, dt)
    xs = xp + noise
    if wrap:
        xs[2] += 2 * np.pi
    Ps = Pp - 0.1 * np.diag(np.diag(Pp))
    C_ = np.linalg.solve(Pp, m.apply_F(Fm, P)).T
    d = xs - xp
    for i in wrap:
        d[i] = KF.wrap_pi(d[i])
    vals.append([(x + C_ @ d).tolist(), (P + C_ @ (Ps - Pp) @ C_.T).tolist()])
gold["rts"] = {"inputs": K.digest(tc), "values": vals}
print("rts", len(tc))

with gzip.open(K.GOLDEN, "wt", compresslevel=9) as f:
    json.dump(gold, f, separators=(",", ":"))
print(os.path.getsize(K.GOLDEN), "bytes")
