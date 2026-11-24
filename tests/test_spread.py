"""Lane SPREAD (2026-11-24): the radar's own needle (`std_rho`), the hold of a standing vehicle of two places while its
range can still resolve (`place_hold`), a pedestrian at parked vehicles (`ped_gap`, `ped_onset`, `ped_cov`), what is
hidden between parked vehicles (`gaps`, the alert `hidden_gap`), a pedestrian's entry holds (`entry_hold_vru_s`)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import predict_cases as K  # noqa: E402
import test_curves as C  # noqa: E402
from chris_integrator.decide import alerts as AL, threat as TH  # noqa: E402
from chris_integrator.native import decide as ND  # noqa: E402
from chris_integrator.plugins.decide import Decide  # noqa: E402
from chris_integrator.predict import Predictor  # noqa: E402

STRAIGHT = {"centre": [[float(x), 0.0] for x in range(0, 150, 5)], "n_left": 0, "n_right": 0, "oncoming": 1.0,
            "park_right": 0.9, "park_left": 0.0, "quality": 0.9}


# ---- the radar's needle ---------------------------------------------------------------------------------------------
RHO = dict(C.NEW, std_rho=1.0)                # the knob on, whatever its default


def rho_of(x, y, sr, sc=0.1):
    """(sx, sy, rho) of a place known to sr along the line of sight and sc across it."""
    r = math.hypot(x, y)
    c, s = x / r, y / r
    sx, sy = math.sqrt(sr * sr * c * c + sc * sc * s * s), math.sqrt(sr * sr * s * s + sc * sc * c * c)
    return sx, sy, (sr * sr - sc * sc) * c * s / (sx * sy)


def bend_car(i, s, d, sr, scale=1.0, rho=True):
    x, y = C.centre(s, d)
    x, y = x * scale, y * scale
    sx, sy, r = rho_of(x, y, sr)
    o = K.obj(i, x, y, 0.0, 0.0, ps=0.95, std=(sx, sy, 0.5, 0.3))
    if rho:
        o["std_rho"] = r
    return o


def test_a_needle_along_the_line_of_sight_reads_as_the_reconstruction_did():
    _, a = C.run([bend_car(1, 55.0, -2.9, 9.0, scale=0.9, rho=False)], native=RHO)
    _, b = C.run([bend_car(1, 55.0, -2.9, 9.0, scale=0.9, rho=True)], native=RHO)
    pa, pb = float(a._modes[0, 0, 1]), float(b._modes[0, 0, 1])
    assert 0.05 < pa < 0.75 and abs(pa - pb) < 0.01, (pa, pb)
    assert np.allclose(a.lane[0], b.lane[0], atol=0.01)
    fa, fb = dict((n, t) for n, _, t in C.futs(a, 1)), dict((n, t) for n, _, t in C.futs(b, 1))
    for n in ("parked", "stopped"):
        assert np.allclose(fa[n][0, :2], fb[n][0, :2], atol=0.15), (n, fa[n][0], fb[n][0])


def test_with_the_knob_off_the_field_is_as_if_not_given():
    _, a = C.run([bend_car(1, 55.0, -2.9, 9.0, scale=0.9, rho=False), bend_car(2, 30.0, 0.0, 3.0, rho=False)], native=RHO)
    _, b = C.run([bend_car(1, 55.0, -2.9, 9.0, scale=0.9), bend_car(2, 30.0, 0.0, 3.0)], native=dict(C.NEW, std_rho=0.0))
    assert np.array_equal(a.traj, b.traj, equal_nan=True) and np.array_equal(a.lane, b.lane) and np.array_equal(a.p0, b.p0)


def test_a_needle_that_does_not_lie_along_the_line_of_sight_is_read_as_it_is():
    # a standing car 40 m ahead, 2.9 m right of the lane's middle on a straight road; its place long along an axis 20
    # degrees off the car's own, towards the lane with the range (rho > 0) or away from it (rho < 0). `std` alone is
    # the same for both: only the correlation says which way the needle crosses the lane.
    th = math.radians(20.0)
    sx, sy = math.hypot(6.0 * math.cos(th), 0.2 * math.sin(th)), math.hypot(6.0 * math.sin(th), 0.2 * math.cos(th))
    rho = (36.0 - 0.04) * math.cos(th) * math.sin(th) / (sx * sy)

    def one(r):
        o = K.obj(1, 40.0, -2.9, 0.0, 0.0, ps=0.95, std=(sx, sy, 0.5, 0.3))
        if r is not None:
            o["std_rho"] = r
        P = Predictor(dict(C.NOCHECK, native=RHO, row_odds_max=0.0))
        for f in range(21):
            pred = P.update(K.msg(f * 0.05, [o], v=13.0), STRAIGHT)
        return pred
    up, down = one(rho), one(-rho)
    for pred, sign in ((up, 1.0), (down, -1.0)):
        f = dict((n, (p, t)) for n, p, t in C.futs(pred, 1))
        assert "stopped" in f and "parked" in f, list(f)
        xs, ys = f["stopped"][1][0, 0], f["stopped"][1][0, 1]
        assert abs(ys) < 1.3                                    # the place in the car's way is in the lane ...
        assert sign * (xs - 40.0) > 2.0, (sign, xs)             # ... beyond the track's place (rho > 0) or short of it
        assert sign * f["parked"][1][0, 4] > 0.5                # the futures' own correlation is the needle's
    assert abs(float(up._modes[0, 0, 1]) - float(down._modes[0, 0, 1])) < 0.05     # mirror images: the same probability


def test_a_standing_cars_one_place_starts_with_the_radars_covariance():
    # in the lane on a straight road, 1 m right of its middle: one place; its future's spread is the needle
    sx, sy, r = rho_of(30.0, -1.0, 1.5)
    assert r < -0.4
    for given in (True, False):
        o = K.obj(1, 30.0, -1.0, 0.0, 0.0, ps=0.95, std=(sx, sy, 0.5, 0.3))
        if given:
            o["std_rho"] = r
        P = Predictor(dict(C.NOCHECK, native=RHO, row_odds_max=0.0))
        for f in range(21):
            pred = P.update(K.msg(f * 0.05, [o], v=13.0), STRAIGHT)
        f = dict((n, t) for n, _, t in C.futs(pred, 1))
        assert float(pred._modes[0, 0, 1]) == 0.0 and "stopped" in f, (float(pred._modes[0, 0, 1]), list(f))
        assert abs(f["stopped"][0, 4] - (r if given else 0.0)) < 0.02, (given, f["stopped"][0])


def test_a_pedestrians_futures_keep_the_spread_they_had():
    # (a pedestrian has no two places: its one standing future as a needle across a bend was said in the controls)
    x, y = C.centre(55.0, -2.9)
    sx, sy, r = rho_of(x, y, 4.0)
    res = []
    for given in (False, True):
        o = K.obj(1, x, y, 0.0, 0.0, "human", ps=0.9, std=(sx, sy, 0.4, 0.3))
        if given:
            o["std_rho"] = r
        res.append(C.run([o], native=RHO)[1])
    # sx, sy, rxy of every future (the one that walks along: to what the offset's spread, read along the needle, differs)
    assert np.allclose(res[0].traj[:, :, :, 2:5], res[1].traj[:, :, :, 2:5], atol=1e-3, equal_nan=True)
    for pred in res:
        assert C.futs(pred, 1)[0][0] == "wait" and C.futs(pred, 1)[0][2][0, 4] == 0.0


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
