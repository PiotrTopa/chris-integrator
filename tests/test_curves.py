"""Roadside cars on a bend (lane CURVES, 2026-11-24): the place of an object a camera ranges poorly is a needle along
the line of sight, on a bend that needle crosses the lanes, and the prediction says so: two places with their
probabilities (native predict.c lane_step / object_modes: lane_los, lane_mem, lane_hyp; predictor.py row_odds_max),
the decision layer takes the futures' weights as they are (state 3 of the futures' rows, `p_place`). (The alert on an
object whose place is in doubt: tests/test_alert_evidence.py.)"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import predict_cases as K  # noqa: E402
from chris_integrator.decide import futures as FU, threat as TH  # noqa: E402
from chris_integrator.predict import Predictor  # noqa: E402

NOCHECK = {"selfcheck": {"enabled": False}}
NEW = {"lane_los": 1.0, "lane_mem": 0.0, "lane_hyp": 1.0, "motion_scale": 0.0}
OLD = {"lane_los": 0.0, "lane_mem": 1.0, "lane_hyp": 0.0, "motion_scale": 0.0}
R = 110.0


def centre(s, d=0.0):
    """A left bend from the car on: the point at arc s, d to the left of the lane's centre."""
    return ((R - d) * math.sin(s / R), R - (R - d) * math.cos(s / R))


CTX = {"centre": [list(centre(s)) for s in range(0, 150, 5)], "n_left": 0, "n_right": 0, "oncoming": 1.0,
       "park_right": 0.9, "park_left": 0.0, "quality": 0.9}


def needle(x, y, sr, sc=0.1):
    """`std` (sx, sy) of a place known to sr along the line of sight and sc across it."""
    r = math.hypot(x, y)
    c, s = x / r, y / r
    return (round(math.sqrt(sr * sr * c * c + sc * sc * s * s), 2), round(math.sqrt(sr * sr * s * s + sc * sc * c * c), 2))


def car(i, s, d, sr, scale=1.0, ps=0.95):
    """A standing car at arc s, offset d, its range read `scale` times the true one, 1 sigma sr."""
    x, y = centre(s, d)
    x, y = x * scale, y * scale
    sx, sy = needle(x, y, sr)
    return K.obj(i, x, y, 0.0, 0.0, ps=ps, std=(sx, sy, 0.5, 0.3))


def run(objs, native=NEW, T=1.0, cfg=None):
    P = Predictor(dict(NOCHECK, native=native, **dict({"row_odds_max": 0.0}, **(cfg or {}))))
    pred = None
    for f in range(int(round(T / 0.05)) + 1):
        pred = P.update(K.msg(f * 0.05, objs, v=13.0), CTX)
    return P, pred


def futs(pred, oid):
    i = pred.index(oid)
    return [(pred.names[int(pred.m[i, k])], float(pred.p0[i, k]), pred.traj[i, k]) for k in range(pred.n_modes[i])]


def offset(x, y):
    """Metres to the left of the bend's centre line."""
    return R - math.hypot(x, y - R)


def test_a_poorly_ranged_car_on_a_bend_has_two_places():
    # parked on the outside of the bend 55 m on, its range read 10 % short (so its mean is in the lane), known to 9 m
    _, pred = run([car(1, 55.0, -2.9, 9.0, scale=0.9)])
    f = futs(pred, 1)
    names = [n for n, _, _ in f]
    assert "parked" in names and "stopped" in names, names
    assert abs(sum(p for _, p, _ in f) - 1.0) < 1e-6
    i = pred.index(1)
    p_way = float(pred._modes[i, 0, 1])                       # the flag: P(in the car's way)
    in_way = sum(p for n, p, t in f if abs(offset(t[0, 0], t[0, 1])) < 1.75)
    assert 0.05 < p_way < 0.75 and abs(in_way - p_way) < 0.02, (p_way, in_way)
    for n, p, t in f:
        if n == "parked":                                     # beside the road, and it stays clear of the lane
            assert offset(t[0, 0], t[0, 1]) < -2.0 and t[0, 2] < 5.0
        if n == "stopped":
            assert abs(offset(t[0, 0], t[0, 1])) < 1.2
        if n in ("parked", "stopped"):
            assert t[0, 4] > 0.95                              # a needle along the line of sight, not an upright box
    # the same futures as the decision layer takes them: state 3 + P(in the way)
    h = pred.handoff()
    st = float(h.arrays()[2][pred.index(1)][2])
    assert int(st) == 3 and abs(st - 3.0 - p_way) < 2e-3
    msg = pred.message(K.msg(1.0, [car(1, 55.0, -2.9, 9.0, scale=0.9)], v=13.0))
    assert abs(msg["objects"][0]["lane_res"] - p_way) < 2e-3
    back = FU.from_message(msg)
    assert abs(float(back.rows[0][2]) - st) < 2e-3


def test_well_ranged_cars_keep_one_place():
    # no doubt about the range: the futures and their weights are as they always were (the decision weighs the
    # standing ones by the lanes), whether beside the road, in the lane or half in it
    _, pred = run([car(1, 55.0, -2.9, 0.6), car(2, 55.0, 0.0, 0.6), car(3, 30.0, -1.45, 0.5)])
    for oid in (1, 2, 3):
        assert float(pred._modes[pred.index(oid), 0, 1]) == 0.0
    assert pred.lane[pred.index(1)][5] > 0.6 and [n for n, _, _ in futs(pred, 1)][0] == "parked"
    assert pred.lane[pred.index(2)][0] > 0.6 and "parked" not in [n for n, _, _ in futs(pred, 2)]
    h = pred.handoff()
    assert all(float(r[2]) == 1.0 for r in h.arrays()[2])


def test_a_stopped_car_in_the_lane_keeps_most_of_its_weight_at_the_same_doubt():
    _, pred = run([car(1, 55.0, -2.9, 9.0), car(2, 55.0, 0.0, 9.0)])
    p_side, p_lane = float(pred._modes[pred.index(1), 0, 1]), float(pred._modes[pred.index(2), 0, 1])
    assert p_lane > 0.5 > p_side and p_lane > p_side + 0.2, (p_side, p_lane)


def test_a_moving_vehicle_keeps_its_lanes_futures():
    x, y = centre(55.0, 0.0)
    sx, sy = needle(x, y, 9.0)
    _, pred = run([K.obj(1, x, y, 12.0, 6.0, ps=0.02, std=(sx, sy, 0.8, 0.4))])
    assert float(pred._modes[0, 0, 1]) == 0.0 and "parked" not in [n for n, _, _ in futs(pred, 1)]


def test_the_place_is_not_counted_again_frame_after_frame():
    objs = [car(1, 55.0, -2.9, 9.0, scale=0.9)]
    _, a = run(objs, T=0.3)
    _, b = run(objs, T=3.0)
    assert abs(a.lane[0][0] - b.lane[0][0]) < 0.02             # the same place says the same after three seconds
    _, c = run(objs, native=OLD, T=0.3)
    _, d = run(objs, native=OLD, T=3.0)
    assert abs(d.lane[0][0] - c.lane[0][0]) > 0.1              # as it was: the same place, multiplied in again and again


def test_cars_seen_standing_beside_the_road_weigh_on_the_next_one():
    far = car(9, 60.0, -2.9, 9.0, scale=0.9)
    row = [car(1 + k, 10.0 + 6.0 * k, -2.9, 0.3) for k in range(4)]
    _, alone = run([far], cfg={"row_odds_max": 12.0})
    P, with_row = run(row + [far], cfg={"row_odds_max": 12.0})
    assert P._row_odds[1] == 5.0 and P._row_odds[0] == 1.0     # four on the right, none in the lane: (4 + 1) / (0 + 1)
    pa, pb = float(alone._modes[alone.index(9), 0, 1]), float(with_row._modes[with_row.index(9), 0, 1])
    assert pb < 0.65 * pa, (pa, pb)
    _, off = run(row + [far])
    assert abs(float(off._modes[off.index(9), 0, 1]) - pa) < 0.05


def test_a_camera_velocity_within_the_ranges_scale_is_not_it_moves():
    def standing(pred, oid):
        return sum(p for n, p, _ in futs(pred, oid) if n in ("parked", "stopped"))
    x, y = centre(20.0, -2.9)
    lx, ly = centre(30.0, 0.0)
    ox, oy = centre(40.0, 3.5)
    objs = [K.obj(1, x, y, 3.0, 0.0, ps=0.1, std=(3.0, 0.5, 2.5, 0.4)),          # a parked car "driving along" at 0.23 v
            K.obj(2, lx, ly, 12.0, 3.0, ps=0.02, std=(3.0, 0.5, 2.5, 0.4)),      # a lead at the car's speed
            K.obj(3, ox, oy, -12.0, -3.0, ps=0.02, std=(3.0, 0.5, 2.5, 0.4))]    # an oncoming car
    for o in objs:
        o["motion"] = "unknown" if o["id"] == 1 else "moving"
    _, off = run(objs, native=dict(NEW, motion_scale=0.0))
    _, on = run(objs, native=dict(NEW, motion_scale=1.0))
    assert standing(off, 1) < 0.35 and standing(on, 1) > 0.5, (standing(off, 1), standing(on, 1))
    for i in (2, 3):
        assert standing(on, i) < 0.1 and abs(standing(on, i) - standing(off, i)) < 0.02


def _frame(lane_res):
    """One object 30 m ahead in the lane with a standing future and lanes that call it parked."""
    n = 5
    t = [0.5 * (k + 1) for k in range(n)]
    o = K.obj(1, 30.0, 0.0, 0.0, 0.0, ps=0.95, std=(1.0, 0.2, 0.3, 0.2))
    o["fut"] = [{"p": 1.0, "p0": 1.0, "m": "stopped", "x": [30.0] * n, "y": [0.0] * n, "sx": [1.0] * n, "sy": [0.2] * n,
                 "rxy": [0.0] * n}]
    o["lane"] = {"ego": 0.1, "left": 0.0, "right": 0.0, "onc": 0.0, "cross": 0.0, "park": 0.9, "off": 0.0}
    if lane_res is not None:
        o["lane_res"] = lane_res
    m = K.msg(0.0, [o], v=13.0)
    m["fut"] = {"t": t}
    return m


def test_the_decision_takes_two_place_futures_at_their_weight():
    tm = TH.ThreatModel()
    plain, res = tm.explain(_frame(None)), tm.explain(_frame(0.4))
    assert abs(plain.modes[0, 0] - 0.1) < 1e-9                 # as before: a standing future weighs P(ego) + P(cross)
    assert abs(res.modes[0, 0] - 1.0) < 1e-9                   # its weight is the prediction's: not weighed again
    assert "p_place" not in plain.records[0] and res.records[0]["p_place"] == 0.4
    assert res.records[0]["p_coll"] > 0.9 > 0.2 > plain.records[0]["p_coll"]
    m = _frame(0.4)                                            # of the two places, the one beside the way keeps the floor
    m["objects"][0]["fut"][0]["m"] = "parked"
    beside = tm.explain(m)
    assert abs(beside.modes[0, 0] - tm.cfg["lane_weight_floor"]) < 1e-9 and beside.records[0]["p_coll"] < 0.1


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
