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


# ---- the hold of a standing vehicle of two places ---------------------------------------------------------------------
def two_places(x_way, x_beside, p_way, v=13.0):
    n = 12
    t = [0.5 * (k + 1) for k in range(n)]
    o = K.obj(1, x_way, 0.0, 0.0, 0.0, ps=0.95, std=(6.0, 0.2, 0.3, 0.2))

    def fut(name, p, x, y):
        return {"p": p, "p0": p, "m": name, "x": [x] * n, "y": [y] * n, "sx": [1.0] * n, "sy": [0.2] * n, "rxy": [0.0] * n}
    o["fut"] = [fut("stopped", p_way, x_way, 0.0), fut("parked", 1.0 - p_way, x_beside, -2.9)]
    o["lane"] = {"ego": p_way, "left": 0.0, "right": 0.0, "onc": 0.0, "cross": 0.0, "park": 1.0 - p_way, "off": 0.0}
    o["lane_res"] = p_way
    m = K.msg(0.0, [o], v=v)
    m["fut"] = {"t": t}
    return m


def test_the_hold_waits_while_comfortable_braking_can_and_no_longer():
    # at 8 m/s caution (time margin 1.5 s) begins 3.6 s before the place in the way, braking at 3 m/s^2 after the
    # reaction time is due 2.9 s before: 0.7 s in which the range may still say which place it is. (At 13 m/s that
    # window is 0.2 s, from 15 m/s on there is none: 1.5 s - v (1 / 2 a_comfortable - 1 / 2 a_full).)
    off, on = TH.ThreatModel(), TH.ThreatModel({"place_hold": True})
    far = two_places(28.5, 40.0, 0.5, v=8.0)                   # 3.3 s away
    a, b = off.explain(far).records[0], on.explain(far).records[0]
    assert a["level"] == 2 and "hold" not in a, a
    assert b["level"] == 1 and b["level_exit"] == a["level_exit"] >= 2 and b["hold"]["held"] is True, b
    assert b["hold"]["res_s"] == 0.0 and 0.0 < b["hold"]["act_s"] < 0.6 and b["mass"] == a["mass"]
    near = two_places(23.5, 35.0, 0.5, v=8.0)                  # 2.7 s away: comfortable braking is due: the masses' word
    c = on.explain(near).records[0]
    assert c["level"] >= 2 and c["hold"]["held"] is False and c["hold"]["act_s"] < 0.0, c
    sure = two_places(28.5, 40.0, 0.95, v=8.0)                 # in the car's way with 0.95: no second place to wait for
    assert "hold" not in on.explain(sure).records[0] and on.explain(sure).records[0]["level"] >= 2
    close = two_places(28.5, 29.5, 0.5, v=8.0)                 # 1 m apart: at 9 % of the range not told apart before 5.6 m
    d = on.explain(close).records[0]
    assert d["level"] >= 2 and d["hold"]["held"] is False and d["hold"]["res_s"] > d["hold"]["act_s"], d
    fast = two_places(52.0, 64.0, 0.5, v=13.0)                 # 13 m/s: by the time caution begins, braking is nearly due
    assert off.explain(fast).records[0]["level"] == 1 and off.explain(two_places(50.0, 62.0, 0.5)).records[0]["level"] == 2
    assert on.explain(two_places(50.0, 62.0, 0.5)).records[0]["hold"]["held"] is False


def test_the_hold_is_off_by_default_and_changes_nothing_then():
    assert TH.DEFAULT_CFG["place_hold"] is False
    assert len(TH.ThreatModel().native_cfg()) == len(TH.ThreatModel({"place_hold": True}).native_cfg())


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
