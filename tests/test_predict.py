"""The predictor (chris_integrator/predict, native/src/predict.c): what each part must do, on small made-up scenes
with clean inputs, and the golden data of the fixed scene (tests/predict_cases.py). The judgement on traffic
scenarios with the truth and on recorded drives is tools/predict_eval.py."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import predict_cases as K  # noqa: E402
from chris_integrator import native  # noqa: E402
from chris_integrator.predict import ContextBuilder, Predictor  # noqa: E402
from chris_integrator.predict import predictor as PM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NOCHECK = {"selfcheck": {"enabled": False}}


def drive(scene, T, ctx, cfg=None, dt=0.05, v=14.0, w=0.0, each=None):
    """scene(t) -> objects; returns the predictor and its last prediction."""
    P = Predictor(dict(NOCHECK, **(cfg or {})))
    pred = None
    for f in range(int(round(T / dt)) + 1):
        t = f * dt
        m = K.msg(t, scene(t), v=v, w=w)
        pred = P.update(m, ctx(t) if callable(ctx) else ctx)
        if each is not None:
            each(t, m, pred)
    return P, pred


def futures(pred, oid):
    i = pred.index(oid)
    return {pred.names[int(pred.m[i, k])]: (float(pred.p[i, k]), float(pred.p0[i, k]), pred.traj[i, k]) for k in range(pred.n_modes[i])}


def lane(pred, oid):
    i = pred.index(oid)
    return pred.lanes[int(np.argmax(pred.lane[i]))]


def test_native_part_and_constants():
    assert "predict" in native.PARTS and native.status()["version"] == native.VERSION
    cfg = native.predict.predict_cfg()
    assert list(cfg) == list(PM.NATIVE_CFG) and cfg["modes_max"] == 6.0 and cfg["interact"] == 1.0
    names, lanes, slots, widths = native.predict.predict_names()
    assert lanes == ("ego", "left", "right", "onc", "cross", "park", "off") and len(slots) == 15
    assert {"cut_in", "cut_out", "turn_left", "turn_right", "stop", "pull_out", "parked", "u_turn", "onc_keep",
            "onc_drift", "start_cross", "crossing", "wait", "walk", "ride", "swerve", "follow", "keep", "free"} <= set(names)
    try:
        Predictor({"native": {"no_such_constant": 1.0}})
        raise AssertionError("an unknown native constant must be refused")
    except KeyError:
        pass


def test_times_and_message_form():
    t = PM.sample_times(PM.DEFAULT_CFG)
    assert len(t) == 22 and t[0] == 0.25 and t[11] == 3.0 and t[12] == 3.5 and t[-1] == 8.0
    P = Predictor(NOCHECK)
    m, pred = K.run(P, T=2.0)
    out = pred.message(m)
    assert out["type"] == "radar_fut" and out["ts_mono_ns"] == m["ts_mono_ns"] and m["type"] == "radar"
    assert "fut" not in m["objects"][0]                       # the radar message itself is not touched
    assert out["fut"]["t"] == t.tolist()
    eg = out["ego_fut"]["modes"]
    assert eg and abs(sum(e["p"] for e in eg) - 1.0) < 2e-3 and all(len(e[k]) == 22 for e in eg for k in ("x", "y", "v"))
    for o in out["objects"]:
        assert set(o["lane"]) == {"ego", "left", "right", "onc", "cross", "park", "off"}
        assert abs(sum(o["lane"].values()) - 1.0) < 5e-3 and abs(sum(o["intent"].values()) - 1.0) < 2e-2
        assert 1 <= len(o["fut"]) <= 6
        assert abs(sum(f["p"] for f in o["fut"]) - 1.0) < 3e-3 and abs(sum(f["p0"] for f in o["fut"]) - 1.0) < 3e-3
        for f in o["fut"]:
            assert set(f) == {"p", "p0", "m", "x", "y", "sx", "sy", "rxy", "v"}
            assert all(len(f[k]) == 22 for k in ("x", "y", "sx", "sy", "rxy", "v"))
            assert min(f["sx"]) > 0.0 and min(f["sy"]) > 0.0 and max(abs(r) for r in f["rxy"]) < 1.0
    json.dumps(out)                                           # plain data
    # the parts of the message one by one (the same native builder): what the message holds
    assert pred.ego_fut() == out["ego_fut"] and len(out["objects"]) == len(m["objects"])
    for o in out["objects"]:
        i = pred.index(o["id"])
        assert (pred.fut(i), pred.lane_dict(i), pred.intent_names(i)) == (o["fut"], o["lane"], o["intent"])
        f = max(o["fut"], key=lambda f: f["p"])               # rounded: positions to 0.01, probabilities to 0.001
        k = [pred.names[c] for c in pred.m[i, :pred.n_modes[i]]].index(f["m"])
        assert f["x"] == np.round(pred.traj[i, k, :, 0], 2).tolist() and f["p"] == round(float(pred.p[i, k]), 3)
        assert o["lane"] == dict(zip(pred.lanes, np.round(pred.lane[i], 3).tolist()))
        named = {int(pred.slot[i, q]): pred.names[int(pred.m[i, q])] for q in range(pred.n_modes[i])}
        want = {}
        for sl, v in enumerate(np.round(pred.intent[i], 3).tolist()):     # the slots under their futures' names
            if v > 0.0005:
                name = named.get(sl) or pred.slots[sl]
                want[name] = round(want.get(name, 0.0) + v, 3) if name in want else v
        assert o["intent"] == want
    # an object the predictor did not take is passed on as it is
    more = dict(m, objects=m["objects"] + [{"id": 99, "state": "tentative", "x_m": 5.0, "y_m": 0.0}])
    assert pred.message(more)["objects"][-1] is more["objects"][-1]
    # arrays: the same content
    i = pred.index(1)
    assert pred.traj.shape[1:] == (8, 22, 6) and np.isfinite(pred.traj[i, :pred.n_modes[i]]).all()
    assert abs(pred.p[i, :pred.n_modes[i]].sum() - 1.0) < 1e-9 and abs(pred.p0[i, :pred.n_modes[i]].sum() - 1.0) < 1e-9


def test_the_radars_objects_as_the_native_core_reads_them():
    """predict.c predict_pack: which objects are taken, each one's row (PO_*), a row with a number that is none."""
    nan = math.nan

    def o(**kw):
        return dict({"id": 1, "state": "confirmed", "x_m": 10.0, "y_m": 1.0}, **kw)
    objs = [o(id=1, vx_mps=3.0, std=[0.5, 0.2, 0.01, 2.0], p_stationary=0.2, len=9.0, wid=2.5, hdg=[0.1, 0.05], exist=0.9,
              indicator="left", brake_lights=True, flags=["parked", "oncoming"], vis="occluded", motion="moving",
              ctr=[14.0, 1.2], **{"class": {"label": "truck"}}),
            o(id=2, state="memory", x_m=4, indicator="right", brake_lights=False, vis=True, motion="stationary", hdg=0.3,
              hdg_std=0.2, ctr=[1.0], **{"class": {"label": "lorry"}}),
            o(id=3, state="tentative", x_m=2.0), o(id=4, x_m=None), "junk", o(id=6, y_m=nan),
            o(id=5, x_m=30.0, vx_mps=nan, ay_mps2=1.0, std=[1.0, 1.0, 1.0, 1.0], p_stationary=0.9, vis=False, indicator="hazard")]
    rows, ids, taken, sel = native.predict.predict_pack(objs, False, 48)
    a = np.frombuffer(rows).reshape(-1, PM.NO)
    assert ids == [1, 2, 5] and sel == {1: 0, 2: 1, 5: 2} and all(u is v for u, v in zip(taken, (objs[0], objs[1], objs[6])))
    want = [[10.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.05, 2.0, 2.0, 0.2, 9.0, 2.5, 0.1, 0.05, 0.9, 1.0, 1.0, 1.0, 1.0, 0.0,
             0.0, -1.0, 1.0, 14.0, 1.2],
            # no std: (2, 1, 2, 1); a label that is not a kind: 8; remembered: not seen; a centre of one number: none
            [4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 1.0, 8.0, 0.5, nan, nan, 0.3, 0.2, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0,
             1.0, 1.0, 0.0, nan, nan],
            # a velocity that is no number: no motion, wide spreads, p_stationary one half
            [30.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 1.0, 8.0, 0.5, nan, nan, nan, nan, 1.0, 1.0, 0.0, nan, 0.0, 2.0,
             0.0, 0.0, 0.0, nan, nan]]
    assert np.array_equal(a, np.array(want), equal_nan=True)
    assert native.predict.predict_pack(objs, True, 48)[1] == [1, 2, 3, 5]             # tentative ones too
    rows, ids, _, sel = native.predict.predict_pack(objs, True, 2)                   # the nearest two, nearest first
    assert ids == [3, 2] and sel == {3: 0, 2: 1} and np.frombuffer(rows).reshape(-1, PM.NO)[:, PM.O_ID].tolist() == [0.0, 1.0]
    assert native.predict.predict_pack([], False, 48)[1:] == ([], [], {}) and native.predict.predict_pack(None, False, 48)[1] == []
    kinds = [o(id=k, **{"class": {"label": lab}}) for k, lab in enumerate(PM.KINDS)] + [o(id=9, **{"class": {"label": "Car"}})]
    a = np.frombuffer(native.predict.predict_pack(kinds, False, 48)[0]).reshape(-1, PM.NO)
    assert a[:, PM.O_KIND].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]      # the row's kind: PM.KINDS' order


def test_spread_grows_and_starts_at_the_trackers():
    _, pred = drive(lambda t: [K.obj(1, 40.0 - t, 0.0, 13.0, 0.0, std=(1.5, 0.4, 0.8, 0.3))], 2.0, K.context(arm=False))
    f = futures(pred, 1)["keep"][2]
    assert 1.4 < f[0, 2] < 1.9 and f[-1, 2] > 4.0 * f[0, 2]             # along the road: from the tracker's, growing
    assert np.all(np.diff(f[:, 2]) > -1e-9)
    assert f[-1, 3] < 1.0                                               # across: a lane keeper stays in its lane
    # the free future of a road user off the road grows both ways
    _, pred = drive(lambda t: [K.obj(1, 30.0, 25.0 + 2.0 * t, 0.0, 2.0, label="car")], 2.0, K.context(arm=False))
    f = futures(pred, 1)
    assert "free" in f and f["free"][2][-1, 3] > 3.0 and f["free"][2][-1, 2] > 3.0


def test_lane_on_a_curve_is_the_roads_not_the_cars_axis():
    R = 110.0                                                 # a left bend; x, y of the lane centre at arc s
    def centre(s, d=0.0):
        return ((R - d) * math.sin(s / R), R - (R - d) * math.cos(s / R))
    ctx = {"centre": [list(centre(s)) for s in range(0, 150, 5)], "n_left": 0, "n_right": 0, "oncoming": 1.0,
           "park_right": 0.9, "park_left": 0.0, "quality": 0.9}
    def scene(t):
        px, py = centre(55.0, -2.9)                           # parked on the outside of the bend: straight ahead of the car
        lx, ly = centre(70.0, 0.0)                            # a car in the car's own lane, well to the left of its axis
        ox, oy = centre(80.0 - 12.0 * t, 3.5)
        h = (80.0 - 12.0 * t) / R + math.pi
        return [K.obj(1, px, py, 0.0, 0.0, ps=0.95, std=(2.5, 0.4, 0.5, 0.3)),
                K.obj(2, lx, ly, 12.0 * math.cos(70.0 / R), 12.0 * math.sin(70.0 / R), std=(2.5, 0.4, 0.8, 0.4)),
                K.obj(3, ox, oy, 12.0 * math.cos(h), 12.0 * math.sin(h), std=(2.5, 0.4, 0.8, 0.4))]
    _, pred = drive(scene, 1.5, ctx, v=12.0, w=0.0)
    assert abs(pred.xy[pred.index(1)][1]) < 12.0 and pred.xy[pred.index(2)][1] > 15.0
    assert lane(pred, 1) == "park" and pred.lane[pred.index(1)][0] < 0.1          # not "in my lane"
    assert lane(pred, 2) == "ego" and lane(pred, 3) == "onc"
    f = futures(pred, 2)["keep"][2]                           # and it follows the bend, not its velocity's line
    ex, ey = centre(70.0 + 12.0 * 5.0)
    assert math.hypot(f[15, 0] - ex, f[15, 1] - ey) < 2.5


def test_cut_in_is_recognised_and_conditioned_on_the_car():
    seen = {}
    def scene(t):
        lat = min(3.5, 0.9 * max(0.0, t - 1.0))
        return [K.obj(1, 60.0, 0.0, 14.0, 0.0), K.obj(2, 14.0, -3.5 + lat, 14.0, 0.9 if 1.0 < t < 4.8 else 0.0)]
    def each(t, m, pred):
        f = futures(pred, 2)
        if "cut_in" in f and f["cut_in"][1] > 0.5:
            seen.setdefault("t", t)
            seen.setdefault("p", []).append(f["cut_in"][:2])
    _, pred = drive(scene, 6.0, K.context(arm=False), each=each)
    assert 1.0 < seen["t"] < 2.6                              # within 1.6 s of its start
    assert all(p <= p0 + 1e-9 for p, p0 in seen["p"]) and any(p < p0 - 0.01 for p, p0 in seen["p"])
    assert lane(pred, 2) == "ego" and max(futures(pred, 2), key=lambda k: futures(pred, 2)[k][0]) in ("keep", "follow")


def test_follower_brakes_behind_its_leader_and_the_joint_weights_say_so():
    def scene(t):
        return [K.obj(1, 60.0 - 8.0 * t, 0.0, 6.0, 0.0), K.obj(2, 30.0, 0.0, 14.0, 0.0)]
    _, pred = drive(scene, 1.0, K.context(arm=False))
    f = futures(pred, 2)
    assert "follow" in f and "keep" in f
    assert f["follow"][2][-1, 5] < 8.0 < f["keep"][2][-1, 5]              # the follower's speed comes down to the leader's
    i = pred.index(2)
    prior = dict(zip(pred.slots, pred.intent[i]))
    assert f["follow"][0] > prior["follow"] + 0.05                          # driving on would end in the leader
    lead = futures(pred, 1)["keep"][2]
    assert np.all(lead[:, 0] - f["follow"][2][:, 0] > 4.0)                # and never reaches it
    # without interaction: no follow future at all, weights are the filter's
    _, pred = drive(scene, 1.0, K.context(arm=False), cfg={"native": {"interact": 0.0}})
    f = futures(pred, 2)
    assert "follow" not in f and abs(f["keep"][0] - f["keep"][1]) < 1e-12


def test_pedestrian_waits_then_crosses():
    hist = []
    def scene(t):
        if t < 2.0:
            return [K.obj(1, 40.0, -5.9, 0.0, 0.0, "human", ps=0.9, std=(0.6, 0.3, 0.3, 0.3))]
        return [K.obj(1, 40.0, -5.9 + 1.4 * (t - 2.0), 0.0, 1.4, "human", std=(0.6, 0.3, 0.3, 0.3))]
    def ctx(t):
        c = K.context(arm=False)
        c["crossings"] = [{"s_m": 40.0, "half_m": 2.0}]
        return c
    _, pred = drive(scene, 4.0, ctx, v=0.0, each=lambda t, m, p: hist.append((t, futures(p, 1))))
    at = dict((round(t, 2), f) for t, f in hist)
    assert at[1.5]["wait"][0] > 0.5 and at[1.5]["start_cross"][0] > 0.15    # at a marked crossing it may well go
    assert at[1.5]["start_cross"][2][-1, 1] > 1.0                           # ... and that future ends beyond the road
    f = at[4.0]
    assert f["crossing"][0] > 0.8
    assert 4.0 < f["crossing"][2][-1, 1] < 9.5                              # to the far kerb, not on for ever
    # away from a crossing the same standing pedestrian is less likely to step out
    _, pred = drive(lambda t: [K.obj(1, 40.0, -5.9, 0.0, 0.0, "human", ps=0.9, std=(0.6, 0.3, 0.3, 0.3))], 1.5,
                    K.context(arm=False), v=0.0)
    assert futures(pred, 1)["start_cross"][0] < at[1.5]["start_cross"][0]


def test_side_road_vehicle_braking_for_the_junction_is_expected_to_stop():
    def scene(t):
        v = max(0.0, 9.0 - 2.0 * t)
        y = -36.0 + 9.0 * t - t * t if t < 4.5 else -36.0 + 20.25
        o = K.obj(1, 61.5, y, 0.0, v, ps=0.02 if v > 0.3 else 0.95)
        o["ax_mps2"], o["ay_mps2"] = 0.0, -2.0 if v > 0.0 else 0.0
        return [o]
    hist = []
    _, pred = drive(scene, 3.0, K.context(), v=0.0, each=lambda t, m, p: hist.append((t, futures(p, 1), lane(p, 1))))
    assert hist[-1][2] == "cross" and hist[-1][1]["stop"][0] > 0.5
    early = hist[10][1]
    assert {"turn_left", "turn_right", "cross", "stop"} <= set(early)        # every way through the junction is a future
    assert early["stop"][2][-1, 1] < -4.0 < early["cross"][2][-1, 1]         # the stop ends before the road, the other crosses it
    assert early["turn_right"][2][-1, 0] > 67.0 and early["turn_left"][2][-1, 0] < 56.0   # into the road, either way
    assert abs(early["turn_right"][2][-1, 1]) < 2.5 and 1.0 < early["turn_left"][2][-1, 1] < 6.0


def test_parked_stays_and_may_pull_out_and_a_standing_queue_moves_off():
    def scene(t):
        return [K.obj(1, 40.0, -6.3, 0.0, 0.0, ps=0.97), K.obj(2, 30.0, 0.0, 0.0, 0.0, ps=0.97),
                K.obj(3, 55.0, -6.4, 0.0, 0.0, ps=0.97, indicator="left")]
    _, pred = drive(scene, 1.0, K.context(arm=False), v=0.0)
    f = futures(pred, 1)
    assert lane(pred, 1) == "park" and f["parked"][0] > 0.97
    i = pred.index(1)
    assert 0.001 < dict(zip(pred.slots, pred.intent[i]))["pull_out"] < 0.02  # rare, never impossible
    g = futures(pred, 3)                                                    # its indicator set towards the road
    assert g["pull_out"][0] > 5.0 * dict(zip(pred.slots, pred.intent[i]))["pull_out"]
    assert g["pull_out"][2][-1, 1] > -4.5 and g["pull_out"][2][-1, 0] > 60.0
    q = futures(pred, 2)                                                    # standing in the car's lane: stopped, will move off
    assert lane(pred, 2) == "ego" and {"stopped", "move_off"} <= set(q) and q["move_off"][2][-1, 0] > 40.0
    assert q["move_off"][2][-1, 2] > 5.0                                    # when is not known: a wide spread along the lane


def test_the_place_on_the_road_is_the_centre_and_the_futures_start_at_the_near_face():
    """A car standing in the right lane, its near face (x_m, y_m: what the radar sees nearest) 0.25 m inside the
    car's lane, its centre (`ctr`) 0.65 m inside the right lane: placed by the near face it is in the car's lane,
    by its centre in its own; either way its futures carry the reference point the radar gave (the decision
    layer draws the footprint from it)."""
    def run(ctr):
        def scene(t):
            return [K.obj(1, 40.0, -1.5, 0.0, 0.0, ps=0.97, std=(1.0, 0.3, 0.3, 0.3), **({"ctr": ctr} if ctr else {}))]
        return drive(scene, 1.0, K.context(arm=False), v=0.0)[1]
    near, centre = run(None), run([40.0, -2.4])
    assert lane(near, 1) == "ego" and lane(centre, 1) == "right"
    for pred in (near, centre):
        f = futures(pred, 1)
        assert np.allclose(f["stopped"][2][0, :2], (40.0, -1.5)) and abs(f["move_off"][2][-1, 1] + 1.5) < 0.05


def test_oncoming_drift_and_cyclist_swerve():
    def scene(t):
        lat = min(1.4, 0.45 * max(0.0, t - 0.5))
        return [K.obj(1, 110.0 - 24.0 * t, 3.5 - lat, -12.0, -0.45 if 0.5 < t < 3.6 else 0.0),
                K.obj(2, 20.0 - 9.0 * t, -1.4, 5.0, 0.0, "bike", std=(0.8, 0.25, 0.5, 0.25)),
                K.obj(3, 38.0 - 14.0 * t, -2.0, 0.0, 0.0, ps=0.97)]
    _, pred = drive(scene, 2.5, K.context(arm=False))
    f = futures(pred, 1)
    assert lane(pred, 1) == "onc" and f["onc_drift"][0] + f.get("onc_overtake", (0.0,))[0] > 0.15
    assert f["onc_drift"][2][-1, 1] < 1.0                                  # towards the car's own lane
    _, pred = drive(scene, 1.5, K.context(arm=False))
    b = futures(pred, 2)                                                    # a parked car in the cyclist's way
    assert "ride" in b and "swerve" in b and b["swerve"][1] > 0.05
    assert b["swerve"][0] <= b["swerve"][1]                                 # into the car's lane: less so with the car coming
    assert b["swerve"][2][-1, 1] > b["ride"][2][-1, 1] + 0.8


def test_without_context_it_still_predicts_and_trusts_the_road_less():
    P = Predictor(NOCHECK)
    m, pred = K.run(P, T=2.0, ctx=None)
    assert pred.source["path"] == "kinematic" and pred.ego_n >= 1 and pred.names[int(pred.ego_m[0])] in ("ctra", "follow")
    i = pred.index(1)
    free_none = dict(zip(pred.slots, pred.intent[i]))["free"]
    m, pred2 = K.run(Predictor(NOCHECK), T=2.0)
    assert free_none > 3.0 * dict(zip(pred2.slots, pred2.intent[pred2.index(1)]))["free"]
    # beyond the known centre line the road may bend: the spread across grows there
    k = futures(pred, 1)["keep"][2] if "keep" in futures(pred, 1) else None
    k2 = futures(pred2, 1)["keep"][2]
    assert k is None or k[-1, 3] > k2[-1, 3]


def test_the_radars_motion_verdict_takes_hold_quickly_and_ignorance_stays_open():
    ctx = dict(K.context(arm=False), n_right=0.3)                # a lane on the right is possible, not known
    hist = []

    def scene(t):
        # tracked as driving with the traffic, then (from 1 s) the radar finds that it stands beside the path
        if t < 1.0:
            return [K.obj(1, 40.0, -3.0, 13.0, 0.0, ps=0.01, std=(8.0, 0.6, 4.0, 0.6), motion="moving", flags=["with_traffic"])]
        return [K.obj(1, 40.0 - 14.0 * (t - 1.0), -3.0, 0.7, 0.0, ps=0.88, std=(3.5, 0.4, 2.1, 0.25), motion="stationary", flags=["parked"])]
    drive(scene, 2.0, ctx, each=lambda t, m, p: hist.append((round(t, 2), futures(p, 1), lane(p, 1))))
    at = {t: (f, ln) for t, f, ln in hist}
    assert at[0.95][0]["keep"][0] > 0.9 and at[0.95][1] == "right"
    f, ln = at[1.4]                                              # 0.4 s after the radar's verdict
    assert sum(v[0] for k, v in f.items() if k in ("parked", "stopped", "move_off")) > 0.85 and f.get("keep", (0.0,))[0] < 0.1
    f, ln = at[1.5]
    assert ln == "park" and f["parked"][0] > 0.85

    def newborn(t):
        # a lead that has just appeared far ahead: speed known to 11 m/s, p_stationary the radar's prior; then it
        # is found to drive
        if t < 1.0:
            return [K.obj(1, 62.0, -0.4, 5.5, 0.0, ps=0.6, std=(14.0, 0.3, 11.0, 0.5), motion="unknown")]
        return [K.obj(1, 62.0, -0.4, 18.0, 0.0, ps=0.12, std=(13.0, 0.3, 9.0, 0.6), motion="unknown", flags=["with_traffic"])]
    hist.clear()
    drive(newborn, 1.5, ctx, each=lambda t, m, p: hist.append((round(t, 2), futures(p, 1), lane(p, 1))))
    at = {t: f for t, f, ln in hist}
    stand = lambda f: sum(v[0] for k, v in f.items() if k in ("parked", "stopped", "move_off"))
    assert 0.4 < stand(at[0.9]) < 0.6                            # not known: neither hardens
    assert stand(at[1.4]) < 0.12 and at[1.4]["keep"][0] > 0.85   # 0.4 s after the radar says it drives


def test_knobs():
    m, pred = K.run(Predictor(dict(NOCHECK, native={"modes_max": 1.0})), T=1.5)
    assert set(pred.n_modes.tolist()) == {1} and np.allclose(pred.p[:, 0], 1.0)
    m, pred = K.run(Predictor(dict(NOCHECK, horizon_s=4.0, step_coarse_s=1.0)), T=1.5)
    assert pred.t[-1] == 4.0 and len(pred.t) == 13 and pred.traj.shape[2] == 13
    m, a = K.run(Predictor(dict(NOCHECK, native={"clique_max": 1.0})), T=3.0)         # groups by belief propagation
    m, b = K.run(Predictor(NOCHECK), T=3.0)                                           # ... and by the exact search
    assert b.stats[4] == 0 and a.stats[4] >= 1 and b.stats[3] > 0
    for oid in b.ids:
        i = b.index(oid)
        assert np.allclose(a.p[i], b.p[i], atol=0.08), (oid, a.p[i], b.p[i])
    m, pred = K.run(Predictor(dict(NOCHECK, enabled=False)), T=0.5)
    assert pred is None


def test_objects_max_and_state_rows_are_recycled():
    P = Predictor(dict(NOCHECK, objects_max=3))
    for f in range(40):
        t = f * 0.05
        objs = [K.obj(100 + f // 5 * 10 + k, 20.0 + 9.0 * k, 0.0 if k % 2 else -8.0, 10.0, 0.0) for k in range(6)]
        pred = P.update(K.msg(t, objs), K.context(arm=False))
        assert len(pred.ids) == 3 and all(abs(o) < 45 for o in pred.xy[:, 0])
    assert len(P._rows) == 3 and P.health()["objects"] == 3


def test_selfcheck_scores_what_happened():
    P = Predictor()
    for f in range(240):                                      # 12 s: a lead that drives on as predicted
        t = f * 0.05
        P.update(K.msg(t, [K.obj(1, 40.0 - t, 0.0, 13.0, 0.0), K.obj(2, 30.0, -3.5, 14.0, 0.0, ps=0.02, label=None)]),
                 K.context(arm=False))
    s = P.health()["selfcheck"]
    assert s["scored"] >= 10 and s["horizons_s"] == [1.0, 2.0, 3.0, 5.0, 8.0]
    v = s["vehicle"]
    assert v["n"][0] >= 10 and v["min_fde"][2] < 1.0 and v["min_ade"][4] < 1.0
    assert v["maha2"][2] < 2.0 and v["nll"][0] < 3.0
    # the native score of one prediction: a future that is exactly right, another 3 m beside it
    nt = 5
    tr = np.zeros((2, nt, 6))
    tr[:, :, 0] = np.arange(nt)
    tr[1, :, 1] = 3.0
    tr[:, :, 2:4] = 0.5
    act = np.stack([np.arange(nt, dtype=float), np.zeros(nt)], axis=1)
    out = np.zeros((nt + 2, 6))
    win, pw = native.predict.predict_score(tr, np.array([0.3, 0.7]), act, np.array([1.0, 1, 1, 0, 1]), out)
    assert (win, pw) == (0, 0.3) and np.isnan(out[3]).all()
    assert np.allclose(out[[0, 1, 2, 4], 0], 0.0) and np.allclose(out[[0, 4], 3], 3.0) and np.allclose(out[4, 5], 0.0)
    assert abs(out[0, 4] - (-math.log(0.3 / (2 * math.pi * 0.25) + 0.7 * math.exp(-18.0) / (2 * math.pi * 0.25)))) < 1e-9
    assert np.allclose(out[nt].tolist()[:2], [0.0, 3.0])
    # ... into the statistics: by horizon (here the samples 0, 3, 4; 3 was not seen) for the vehicles and for all,
    # the calibration (the future given 0.3 was right, the one given 0.7 was not), the Brier sum
    from chris_integrator.predict.selfcheck import SelfCheck
    sc = SelfCheck(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    sc.hk = [0, 3, 4]
    sc.reset()
    sc.add(0, out, np.array([1.0, 1, 1, 0, 1]), np.array([0.3, 0.7]), win)
    sc.add(7, out, np.array([1.0, 1, 1, 0, 1]), np.array([0.3, 0.7]), win)           # a pedestrian's
    assert sc.scored == 2 and sc.n.tolist() == [[1.0, 0.0, 1.0], [0.0] * 3, [1.0, 0.0, 1.0], [2.0, 0.0, 2.0]]
    assert np.allclose(sc.sums[0, :, 2], [3.0, 0.0, 3.0]) and np.allclose(sc.sums[3, :, 2], [6.0, 0.0, 6.0])
    assert np.allclose(sc.sums[0, 0], out[0, [0, 1, 3, 4, 5]]) and np.allclose(sc.sums[2], sc.sums[0])
    assert sc.cal[3].tolist() == [2.0, 2.0] and sc.cal[7].tolist() == [2.0, 0.0] and sc.cal.sum() == 6.0
    assert np.allclose(sc.brier, [2 * (0.7 ** 2 + 0.7 ** 2), 4.0])


def test_context_builder_reads_the_stacks_messages():
    cb = ContextBuilder()
    ego = PM.EgoTrack()
    for k in range(21):
        ego.add(100.0 + 0.05 * k, 10.0, 0.0)
    xs = [float(x) for x in range(0, 100, 3)][:33]
    line = lambda y: {"present": True, "y": [y] * 33, "y_std": [0.1] * 33}
    cb.on_road({"ts_mono_ns": 100_500_000_000, "plan": {"x": [0.0, 20.0, 60.0], "y": [0.0, 0.0, 0.0]}})
    c = cb.build(101.0, ego)
    assert c["source"] == "plan" and abs(c["centre"][0][0] + 5.0) < 1e-6      # moved back by the 5 m driven since
    cb.on_lanes({"ts_mono_ns": 101_000_000_000, "x": xs, "valid": True, "lane_width_m": 3.4, "lane_to_left": True,
                 "lane_to_right": False, "lines": [line(-5.1), line(-1.7), line(1.7), line(5.1)],
                 "edges": [{"present": True, "y": [-5.6] * 33}, {"present": True, "y": [2.2] * 33}],
                 "prior": {"total": 2, "forward": 1, "backward": 1}})
    c = cb.build(101.0, ego)
    assert c["source"] == "lanes" and c["lane_width_m"] == 3.4 and c["oncoming"] > 0.9
    assert abs(c["kerb_left_m"] - 5.6) < 0.13 and abs(c["kerb_right_m"] + 2.2) < 0.13
    # between two messages of the sources only the points move: the same context, 2 m further on
    c2 = cb.build(101.2, ego)
    assert c2["source"] == "lanes" and abs((c["centre"][0][0] - c2["centre"][0][0]) - 2.0) < 1e-9
    cb.on_roadgeom({"ts_mono_ns": 101_000_000_000, "geom": {
        "lane_width_m": 3.3, "centre": {"x": [0.0, 30.0, 60.0, 90.0], "y": [0.0, 0.0, 1.0, 3.0], "conf": [1, 1, 0.9, 0.2], "src": "vvmm"},
        "branches": [{"dist_m": 50.0, "pts": [[50.0, 4.0], [50.0, 30.0]], "side": "right", "minor": True, "exit": True, "entry": True}],
        "junction": {"dist_m": 50.0, "control": "give_way"}}})
    c = cb.build(101.0, ego)
    assert c["source"].startswith("roadgeom") and len(c["centre"]) == 3 and c["centre"][-1][1] == -1.0     # y left; the doubtful end cut
    assert c["arms"][0]["priority"] == -1 and c["arms"][0]["pts"][1][1] == -30.0 and c["arms"][0]["control"] == "give_way"
    arr = PM.road_arrays(c, PM.DEFAULT_CFG)
    assert arr[1][PM.R_KERBL] > 4.0 and arr[1][PM.R_KERBR] < -1.5 and len(arr[3]) == 1
    kinds = sorted(set(arr[2][:, 2].astype(int).tolist()))
    assert PM.L_EGO in kinds and PM.L_ONC in kinds


def test_plugin_in_the_stack():
    from chris_integrator.mixer import MessageClock
    from chris_integrator.stack import build
    for cfg, want in (({}, True), ({"predict": {"enabled": False}}, False), ({"mixer": {"disable": ["predict"]}}, False)):
        st = build(cfg=cfg, mapper=None, clock=MessageClock())
        got = {"radar": [], "radar_fut": []}
        st.mixer.add_output("t", lambda e: got[e.stream].append(e.msg), streams=("radar", "radar_fut"))
        st.deliver({"type": "road", "seq": 1, "ts_mono_ns": 100_400_000_000, "valid": True, "plan": {"x": [0.0, 30.0, 80.0], "y": [0.0, 0.0, 0.0]}})
        box = st.plugin("radar").box
        for k in range(30):
            ts = 100_000_000_000 + k * 50_000_000
            box.on_ego({"ts_mono_ns": ts, "speed_mps": 10.0, "yaw_rate_rps": 0.0, "accel_long_mps2": 0.0})
            m = box.frame(ts)
            st.mixer.publish("radar", m, "radar")
        assert len(got["radar"]) == 30
        assert (len(got["radar_fut"]) == 30) == want and (want or not got["radar_fut"])
        if want:
            f = got["radar_fut"][-1]
            assert f["ts_mono_ns"] == got["radar"][-1]["ts_mono_ns"] and f["seq"] == got["radar"][-1]["seq"]
            assert f["fut_ctx"]["path"] == "idle" and f["ego_fut"]["modes"] == [] and f["fut"]["t"][-1] == 8.0     # nothing to predict
            m = dict(got["radar"][-1], objects=[K.obj(1, 30.0, 0.0, 9.0, 0.0)], ts_mono_ns=got["radar"][-1]["ts_mono_ns"] + 50_000_000)
            st.mixer.publish("radar", m, "radar")
            f = got["radar_fut"][-1]
            assert f["fut_ctx"]["path"] == "plan" and f["ego_fut"]["modes"][0]["m"] == "path" and f["objects"][0]["fut"]
            assert "fut" not in got["radar"][-1]["objects"][0] and st.plugin("predict").health()["frames"] == 31


def test_golden():
    """The fixed scene's prediction as recorded (tests/data/make_predict_golden.py). The arithmetic is the C
    library's sin / cos / exp / erfc in places: equal to 2e-5, not to the bit; futures are compared by their slot."""
    sys.path.insert(0, os.path.join(HERE, "data"))
    import make_predict_golden as G
    with open(os.path.join(HERE, "data", "predict_golden.json")) as f:
        want = json.load(f)
    got = json.loads(json.dumps(G.make()))
    assert got["frames"] == want["frames"]
    for a, b in zip(got["snapshots"], want["snapshots"]):
        assert set(a["objects"]) == set(b["objects"]) and set(a["ego"]) == set(b["ego"])
        for k in a["ego"]:
            assert np.allclose(a["ego"][k], b["ego"][k], rtol=2e-5, atol=2e-5), k
        for oid, o in a["objects"].items():
            w = b["objects"][oid]
            assert np.allclose(o["lane"], w["lane"], atol=2e-5) and np.allclose(o["intent"], w["intent"], atol=2e-5), oid
            assert set(o["futures"]) == set(w["futures"]), (oid, o["futures"].keys(), w["futures"].keys())
            for s, fu in o["futures"].items():
                assert fu[0] == w["futures"][s][0] and np.allclose(fu[1:], w["futures"][s][1:], rtol=2e-5, atol=2e-5), (oid, s)


def test_simulated_cut_in_with_an_ideal_tracker():
    """One of the simulator's scenarios end to end (tools/predict_eval.py): the cut-in is named within 2 s."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
    import predict_eval as E
    w, frames, P, cost = E.run_scenario("cut_in", 1, False, "sim", objects="truth")
    sc = E.Score(P.times)
    E.score_run(w, frames, sc, ideal=True)
    s = sc.stats()
    ev = s["manoeuvres"]["cut_in"]
    assert ev["events"] == 1 and ev["first_above_half_s"][0] is not None and ev["first_above_half_s"][0] < 2.0
    assert s["lane"]["ego"]["accuracy"] > 0.95 and s["lane"]["left"]["accuracy"] > 0.95
    r = s["rel"]["vehicle"]
    assert r["min_fde"][2] < 0.75 * s["cv_fde_rel"]["vehicle"][2]           # better than carrying the velocity on, at 3 s
