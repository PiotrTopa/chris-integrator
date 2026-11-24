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


# ---- a pedestrian at parked vehicles ----------------------------------------------------------------------------------
PED_STD = (2.8, 0.15, 0.4, 0.3)
PED_OFF = {"ped_gap": 1.0, "ped_onset": 0.0, "ped_cov": 0.0, "ped_carry": 0.0}       # each knob named, whatever its default


def ped(x, y, vy=0.0, ps=0.9):
    return K.obj(5, x, y, 0.0, vy, "human", ps=ps, std=PED_STD)


def parked(i, x, y=-2.9):
    return K.obj(i, x, y, 0.0, 0.0, ps=0.97, std=(0.8, 0.2, 0.3, 0.2), len=4.5, wid=1.8, ctr=[x + 2.25, y])


def ped_run(scene, native, T=1.0, dt=0.05):
    """scene(t) -> objects; the prediction at T, with the knobs of `native` on and the other pedestrian knobs off."""
    P = Predictor(dict(C.NOCHECK, native=dict(C.NEW, **dict(PED_OFF, **native)), row_odds_max=0.0))
    pred = None
    for f in range(int(round(T / dt)) + 1):
        pred = P.update(K.msg(f * dt, scene(f * dt), v=13.0), STRAIGHT)
    return pred


def weight(pred, oid, names):
    return sum(p for n, p, _ in C.futs(pred, oid) if n in names)


def test_a_pedestrian_between_parked_cars_is_likelier_about_to_step_out():
    between = lambda t: [ped(40.0 - 13.0 * t, -2.9), parked(1, 31.0 - 13.0 * t), parked(2, 42.5 - 13.0 * t)]   # noqa: E731
    alone = lambda t: [ped(40.0 - 13.0 * t, -2.9)]                                                              # noqa: E731
    behind = lambda t: [ped(40.0 - 13.0 * t, -5.2), parked(1, 31.0 - 13.0 * t), parked(2, 42.5 - 13.0 * t)]   # noqa: E731
    a0, b0 = weight(ped_run(alone, {}), 5, ("start_cross",)), weight(ped_run(between, {}), 5, ("start_cross",))
    assert 0.02 < a0 < 0.15 and abs(a0 - b0) < 0.01, (a0, b0)   # as it was: where it stands makes no difference
    a1, b1 = (weight(ped_run(s, {"ped_gap": 6.0}), 5, ("start_cross",)) for s in (alone, between))
    assert abs(a1 - a0) < 1e-9 and b1 > 3.0 * b0 and 0.2 < b1 < 0.6, (a1, b1)
    c0, c1 = (weight(ped_run(behind, n), 5, ("start_cross",)) for n in ({}, {"ped_gap": 6.0}))
    assert abs(c1 - c0) < 1e-9                                  # on the pavement behind the cars: not in their line


def test_a_first_step_is_read_from_the_place_before_the_velocity_shows_it():
    # it stands for a second, then its place comes towards the road at 1.4 m/s; the tracker still says "it stands,
    # speed 0" (as it does for a box or two)
    def scene(t):
        return [ped(60.0 - 13.0 * t, -2.9 + 1.4 * max(0.0, t - 1.0))]
    cross = ("start_cross", "crossing")
    on = {"ped_onset": 1.0}
    assert weight(ped_run(scene, {}, T=1.8), 5, cross) < 0.15
    w6, w8 = (weight(ped_run(scene, on, T=T), 5, cross) for T in (1.6, 1.8))
    assert w6 > 0.4 and w8 > 0.8, (w6, w8)                     # 0.6 s after the step it is likelier than not, 0.8 s: said
    assert weight(ped_run(scene, on, T=1.3), 5, cross) < 0.15   # 0.4 m: within what the place it stood at follows
    # ... and its future starts walking, not a second later
    f = dict((n, t) for n, _, t in C.futs(ped_run(scene, on, T=1.8), 5))
    assert f["start_cross"][3, 1] - f["start_cross"][0, 1] > 0.5
    # a pedestrian who stays where it is: the knob changes nothing at all
    still = lambda t: [ped(60.0 - 13.0 * t, -2.9)]                                     # noqa: E731
    a, b = ped_run(still, {}, T=1.5), ped_run(still, on, T=1.5)
    assert np.array_equal(a.traj, b.traj, equal_nan=True) and np.array_equal(a.p0, b.p0)
    # not a step: walking away from the road; the track's place settling at 0.3 m/s as the car closes; the track's
    # place corrected by half a metre between two frames
    away = lambda t: [ped(60.0 - 13.0 * t, -2.9 - 1.4 * max(0.0, t - 1.0))]            # noqa: E731
    settle = lambda t: [ped(60.0 - 13.0 * t, -2.9 + 0.3 * max(0.0, t - 1.0))]          # noqa: E731
    hop = lambda t: [ped(60.0 - 13.0 * t, -2.9 + (0.5 if t > 1.0 else 0.0))]           # noqa: E731
    for sc in (away, settle, hop):
        assert max(weight(ped_run(sc, on, T=T), 5, cross) for T in (1.2, 1.5, 1.8, 2.4, 3.0)) < 0.15
    # boxes every 0.4 s (the place carried between them): still a step, two boxes on

    def sparse(t):
        return [ped(60.0 - 13.0 * t, -2.9 + 1.4 * max(0.0, 0.4 * int(t / 0.4 + 1e-9) - 1.0))]
    assert weight(ped_run(sparse, on, T=1.8), 5, cross) > 0.6


def test_a_step_out_goes_on_as_the_crossing_it_is_when_the_pedestrian_enters_the_lane():
    # the tracker sees it walk out from 0.3 s after the step; 1.1 s after the step it is over the lane's edge
    def scene(t):
        mv = t > 1.3
        return [ped(60.0 - 13.0 * t, -2.9 + 1.4 * max(0.0, t - 1.0), vy=1.4 if mv else 0.0, ps=0.05 if mv else 0.9)]
    cross = ("start_cross", "crossing")
    times = (1.9, 2.0, 2.1, 2.2, 2.4)
    was = [weight(ped_run(scene, {"ped_carry": 0.0}, T=T), 5, cross) for T in times]
    now = [weight(ped_run(scene, {"ped_carry": 1.0}, T=T), 5, cross) for T in times]
    assert was[1] > 0.6 and was[2] < 0.3, was                  # as it was: "walks along" at the moment it is in the lane
    assert all(b >= a - 0.01 for a, b in zip(now, now[1:])) and now[2] > 0.7, now
    assert abs(now[0] - was[0]) < 1e-9                         # the same until then


def test_a_crossing_future_starts_with_the_places_own_spread():
    scene = lambda t: [ped(30.0, -2.9)]                                                # noqa: E731
    f0 = dict((n, t) for n, _, t in C.futs(ped_run(scene, {}), 5))
    f1 = dict((n, t) for n, _, t in C.futs(ped_run(scene, {"ped_cov": 1.0}), 5))
    assert f0["start_cross"][0, 3] > 1.5                        # as it was: round, 2 m across the road too
    assert f1["start_cross"][0, 3] < 0.5 and f1["start_cross"][0, 2] > 2.0, f1["start_cross"][0]


# ---- what is hidden -----------------------------------------------------------------------------------------------------
GCFG = np.array([2.1, 0.88, 1.5, 1.0, 6.0, 2.0, 0.6, 1.5, 1.0, 0.1])


def gcar(i, x, y, **kw):
    return dict({"id": i, "state": "confirmed", "x_m": x, "y_m": y, "ctr": [x + 2.25, y], "len": 4.5, "wid": 1.8,
                 "motion": "stationary", "class": {"label": "car"}, "p_stationary": 0.95}, **kw)


def test_gaps_between_standing_vehicles_and_the_speed_the_car_could_stop_from():
    row = [gcar(1, 10.0, -2.9), gcar(2, 16.0, -2.9), gcar(3, 30.0, -2.9)]
    g = ND.decide_gaps(row, 13.0, None, GCFG)
    # the faces 2.0 m from the lane's middle: 1.12 m from the car's side; a pedestrian at 1.5 m/s is there in 0.75 s,
    # before a driver reacting in 1 s has touched the brake: no speed is one to stop from
    assert g["level"] == 2 and g["gaps"] == 3 and g["side"] == "right" and g["clear_m"] == 1.12 and g["v_stop_mps"] == 0.0
    assert g["s_m"] == 14.5 and abs(g["t_reach_s"] - 12.4 / 13.0) < 0.01
    assert ND.decide_gaps(row, 2.0, None, GCFG) is None         # at walking pace the next gap is beyond twice the stopping distance
    wide = [gcar(1, 20.0, -4.2), gcar(2, 40.0, -4.2)]           # 2.42 m clear: beyond clear_max
    assert ND.decide_gaps(wide, 13.0, None, GCFG) is None
    c2 = GCFG.copy()
    c2[5] = 3.0                                                 # ... counted: 2.42 / 1.5 = 1.61 s: stop from 2 x 6 x 0.61 = 7.4 m/s
    g = ND.decide_gaps(wide, 13.0, None, c2)
    assert g["level"] == 2 and g["clear_m"] == 2.42 and abs(g["v_stop_mps"] - 7.4) < 0.06 and g["s_m"] == 24.5
    # (a gap the car is past before a pedestrian could be in its way is none: 12.4 m at 13 m/s, 1.61 s to walk)
    assert ND.decide_gaps([gcar(1, 10.0, -4.2)], 13.0, None, c2) is None
    assert ND.decide_gaps(wide, 7.0, None, c2) is None          # below it: nothing to say
    assert ND.decide_gaps([gcar(1, 10.0, -2.9), gcar(2, 14.9, -2.9)], 13.0, None, GCFG)["gaps"] == 1     # bumper to bumper: one end
    assert ND.decide_gaps([], 13.0, None, GCFG) is None
    assert ND.decide_gaps([gcar(1, 10.0, -2.9, motion="moving", p_stationary=0.05)], 13.0, None, GCFG) is None   # it drives: no screen
    assert ND.decide_gaps([gcar(1, 10.0, 0.0)], 13.0, None, GCFG) is None                               # in the car's way: the threat's


def test_a_pedestrian_already_seen_at_a_gap_makes_it_no_hidden_one():
    row = [gcar(1, 10.0, -2.9), gcar(2, 30.0, -2.9)]
    assert ND.decide_gaps(row, 13.0, None, GCFG)["gaps"] == 2
    seen = row + [{"id": 9, "state": "confirmed", "x_m": 15.5, "y_m": -2.9, "class": {"label": "human"}}]
    assert ND.decide_gaps(seen, 13.0, None, GCFG)["gaps"] == 1
    other_side = row + [{"id": 9, "state": "confirmed", "x_m": 15.5, "y_m": 2.9, "class": {"label": "human"}}]
    assert ND.decide_gaps(other_side, 13.0, None, GCFG)["gaps"] == 2


def test_gaps_along_a_bend_are_judged_along_the_plan():
    # the same row along a left bend of 110 m radius, in the car frame; the plan is the bend
    xs = [float(s) for s in range(0, 100, 5)]
    path = ND.decide_path([C.centre(s)[0] for s in xs], [-C.centre(s)[1] for s in xs])
    cars = []
    for i, s in enumerate((10.0, 16.0, 30.0)):
        x, y = C.centre(s, -2.9)
        cx, cy = C.centre(s + 2.25, -2.9)
        cars.append(gcar(i + 1, x, y, ctr=[cx, cy]))
    g = ND.decide_gaps(cars, 13.0, path, GCFG)
    assert g["level"] == 2 and g["gaps"] == 3 and abs(g["clear_m"] - 1.12) < 0.05 and abs(g["s_m"] - 14.5) < 0.3
    assert ND.decide_gaps(cars, 13.0, None, GCFG)["clear_m"] != g["clear_m"]       # straight ahead is another street


def test_the_gap_alert_is_a_caution_with_holds_and_never_more():
    a = AL.Alert("hidden_gap", {})
    assert a.cfg["source"] == "gap"
    g = {"level": 2, "level_exit": 2, "side": "right", "s_m": 14.5, "clear_m": 1.12, "v_stop_mps": 0.0, "t_reach_s": 0.95}
    stages = [a.step(k * 0.05, [], {}, {"v": 13.0, "gap": dict(g, level=4, level_exit=4)})[0]["stage"] for k in range(30)]
    assert stages[5] == 0 and stages[10] == 2 and max(stages) == 2          # entered after 0.5 s, and no higher
    out, _ = a.step(1.5, [], {}, {"v": 13.0, "gap": g})
    assert out["object"] is None and out["reason"]["clear_m"] == 1.12 and out["reason"]["v_mps"] == 13.0
    assert out["text"] == "CAUTION GAP RIGHT 0 km/h" and out["active"] is False and out["mode"] == "shadow"
    left = [a.step(1.5 + k * 0.05, [], {}, {"v": 13.0, "gap": None})[0]["stage"] for k in range(1, 60)]
    assert left[10] == 2 and left[-1] == 0                                   # held a second after the last gap, then gone
    assert a.step(9.0, [], {}, {"v": 1.0, "gap": g})[0]["stage"] == 0


def test_the_plug_in_has_the_alert_only_with_the_knob_and_it_does_not_hold_the_pace():
    off, on = Decide({}), Decide({"gaps": True})
    assert "hidden_gap" not in off.alerts.alerts and off._gcfg is None
    assert "hidden_gap" in on.alerts.alerts and list(on._gcfg) == list(GCFG)
    pc = {"hot_p": 0.01, "thw_s": 2.0, "hot_level": 1}
    on.alerts.alerts["hidden_gap"].stage = 2
    on.alerts.alerts["hidden_gap"].up_since = {1: 0.0, 2: 0.0}
    assert on._hot([], pc) is False                             # a caution of a gap: the low rate judges it
    on.alerts.alerts["obstacle_ahead"].stage = 1
    assert on._hot([], pc) is True


# ---- a pedestrian's entry holds --------------------------------------------------------------------------------------------
def test_a_pedestrians_entry_hold_is_its_own_where_configured():
    def entered(cfg, kind):
        a = AL.Alert("obstacle_ahead", cfg)
        rec = {"id": 1, "kind": kind, "level": 3, "level_exit": 3, "mass": [0.9] * 4, "p_coll": 0.9, "tm_s": 0.5,
               "a_req": {"brake": 3.0, "left": 2.0, "right": 2.0}, "why": {}}
        ob = {1: {"id": 1, "state": "confirmed", "exist": 1.0, "motion": "moving", "p_stationary": 0.05, "flags": []}}
        for k in range(20):
            if a.step(k * 0.05, [rec], ob, {"v": 13.0})[0]["stage"] >= 3:
                return round(k * 0.05, 2)
    assert entered({}, "vru") == 0.1 and entered({}, "vehicle") == 0.1
    quick = {"entry_hold_vru_s": (0.1, 0.1, 0.0, 0.0)}
    assert entered(quick, "vru") == 0.0 and entered(quick, "vehicle") == 0.1


# ---- the evidence rule only while waiting costs nothing ---------------------------------------------------------------
def test_a_place_in_doubt_enters_by_time_once_the_time_margin_is_gone():
    def entered(cfg, tm, mass=0.32):
        a = AL.Alert("obstacle_ahead", cfg)
        rec = {"id": 1, "kind": "vehicle", "level": 2, "level_exit": 2, "mass": [mass] * 4, "p_coll": mass, "tm_s": tm,
               "p_place": 0.4, "a_req": {"brake": 3.0, "left": 2.0, "right": 2.0}, "why": {}}
        ob = {1: {"id": 1, "state": "confirmed", "exist": 1.0, "motion": "stationary", "p_stationary": 0.95, "flags": []}}
        for k in range(61):
            if a.step(k * 0.05, [rec], ob, {"v": 13.0})[0]["stage"] >= 2:
                return round(k * 0.05, 2)
    t = entered({}, 1.0)                                        # as it was: 0.07 over the minimum: five times as long
    assert t is not None and 0.9 < t < 1.1 and entered({}, -0.3) == t
    cfg = {"entry_evidence_tm_s": 0.0}
    assert entered(cfg, 1.0) == t                               # time left: the evidence has to add up
    assert entered(cfg, -0.3) == 0.2 and entered(cfg, 0.0) == 0.2      # none left: by time
    assert entered(cfg, None) == t                              # no time margin (no contact on the mean futures): the rule


# ---- a stage never said whose object is gone ---------------------------------------------------------------------------
def test_a_stage_suppressed_all_along_is_not_said_once_its_object_is_gone():
    rec = {"id": 1, "kind": "vehicle", "level": 3, "level_exit": 3, "mass": [0.9] * 4, "p_coll": 0.9, "tm_s": 0.5,
           "a_req": {"brake": 3.0, "left": 1.0, "right": 9.0}, "why": {}}
    ob = {1: {"id": 1, "state": "confirmed", "exist": 1.0, "motion": "stationary", "p_stationary": 0.95, "flags": []}}

    def run(cfg, steer_until):
        """The object for 0.5 s, the driver steering away (2 m/s^2 to the left) until steer_until; then no object."""
        a = AL.Alert("obstacle_ahead", cfg)
        out = []
        for k in range(50):
            T = k * 0.05
            o, _ = a.step(T, [rec] if T < 0.5 else [], ob, {"v": 13.0, "a_lat": 2.0 if T < steer_until else 0.0})
            out.append((o["stage"], o["suppressed"], o["object"]))
        return out
    was = run({"unsaid_needs_object": False}, 0.7)
    assert (3, None, None) in was                               # as it was: the held stage said without an object
    now = run({"unsaid_needs_object": True}, 0.7)
    assert all(s[1] is not None for s in now if s[0] > 0) and (3, "object_gone", None) in now
    assert [s[0] for s in now] == [s[0] for s in was]           # the stages and their holds as they were
    # a stage that WAS said keeps its holds as before, object or not
    said = run({"unsaid_needs_object": True}, 0.3)
    assert (3, None, 1) in said and (3, None, None) in said and not any(s[1] == "object_gone" for s in said)
    assert said == run({"unsaid_needs_object": False}, 0.3)
    # information shown first (below show_from: nothing said), then a warning while the driver steers away: not said
    a = AL.Alert("obstacle_ahead", {"unsaid_needs_object": True})
    out = []
    for k in range(60):
        T = k * 0.05
        r = dict(rec, level=1 if T < 0.5 else 3, level_exit=1 if T < 0.5 else 3)
        o, _ = a.step(T, [r] if T < 1.0 else [], ob, {"v": 13.0, "a_lat": 2.0 if 0.5 <= T < 1.2 else 0.0})
        out.append((o["stage"], o["suppressed"]))
    assert (1, None) in out and (3, "driver_steering_away") in out and (3, "object_gone") in out and (3, None) not in out


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
