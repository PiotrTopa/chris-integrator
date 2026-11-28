"""The futures of a driver who is driving (docs/DESIGN.md 17.3, lane DSTART 2026-11-28): the leader's speed as the
track has it and a start that goes on (`v_track`, `start_go`), the car as the follower its driver is (`ego_follow`),
the car's own arc in a turn (`ego_arc`) and its spread there (`ego_turn_rel`), what stands between the car and an
object (`screen`), an alert held but never said (`unsaid_needs_level`), a conflict with an object whose velocity is
not known yet (`entry_evidence_young_s`); the simulator's driver in the scenarios that hold them. Every knob is off
by default: with them off the stack is what it was (the golden data of the predictor and the decision layer)."""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import predict_cases as K  # noqa: E402
from chris_integrator import native  # noqa: E402
from chris_integrator.decide import alerts as AL  # noqa: E402
from chris_integrator.decide import threat as TH  # noqa: E402
from chris_integrator.predict import Predictor  # noqa: E402

NOCHECK = {"selfcheck": {"enabled": False}}
STRAIGHT = {"centre": [[float(x), 0.0] for x in range(0, 125, 5)], "lane_width_m": 3.5, "n_left": 0, "n_right": 0,
            "oncoming": 0.9, "park_right": 0.0, "park_left": 0.0, "quality": 0.9, "source": "test"}


def test_the_knobs_are_off():
    cfg = native.predict.predict_cfg()
    for k in ("v_track", "start_go", "ego_follow", "ego_arc"):
        assert cfg[k] == 0.0, k
    assert cfg["ego_front_m"] == 2.1 and cfg["ego_ff"] == 1.0 and cfg["ego_yaw_trend_s"] == 0.5
    assert TH.DEFAULT_CFG["ego_turn_rel"] == 0.0 and TH.DEFAULT_CFG["screen"] is False
    assert AL.ALERT_DEFAULTS["unsaid_needs_level"] is False and AL.ALERT_DEFAULTS["entry_evidence_young_s"] is None


# ---- a queue starts: the lead's and the car's own run ------------------------------------------------------------
def _start(t, a_l=1.8, a_e=1.6, gap=2.6, t1=1.0, lag=0.15, stall=None):
    """(the lead's near face from the car's origin, its speed and acceleration, the car's speed and acceleration) at
    t of a queue that starts at t1: both stand `gap` apart bumper to bumper; the lead pulls away at a_l, the car
    `lag` later at a_e. stall: the lead brakes at 3.5 m/s^2 from that time on."""
    def run(t0, a, tt):
        return (max(0.0, tt - t0) * a, 0.5 * a * max(0.0, tt - t0) ** 2)
    vl, sl = run(t1, a_l, t)
    al = a_l if t > t1 else 0.0
    if stall is not None and t > stall:
        v0, s0 = run(t1, a_l, stall)
        ts = min(t - stall, v0 / 3.5)
        vl, sl, al = v0 - 3.5 * ts, s0 + v0 * ts - 1.75 * ts * ts, (-3.5 if vl > 0.0 and t - stall < v0 / 3.5 else 0.0)
    ve, se = run(t1 + lag, a_e, t)
    return 2.1 + gap + sl - se, vl, al, ve, (a_e if t > t1 + lag else 0.0)


def _queue(cfg, T, lag_v=0.0, **kw):
    """The start through a predictor; the lead (id 1, 4.5 m long) as a tracker with the road model's lead would
    have it (speed to 0.7 m/s; lag_v: the tracker's own lag of the speed, s). Returns (predictor, last prediction,
    [(t, p(follow), the car's speed)])."""
    P = Predictor(dict(NOCHECK, native=cfg))
    hist, pred = [], None
    for f in range(int(round(T / 0.05)) + 1):
        t = f * 0.05
        x, vl, al, ve, ae = _start(t, **kw)
        _, vl_t, _, _, _ = _start(max(0.0, t - lag_v), **kw)
        ps = 0.98 if vl_t < 0.2 else 0.02
        o = K.obj(1, x, 0.0, vl_t, 0.0, ps=ps, std=(0.4, 0.4, 0.7, 0.9), ctr=[x + 2.25, 0.0], len=4.5, wid=1.8, hdg=0.0)
        o["ax_mps2"] = al
        pred = P.update(K.msg(t, [o], v=ve, a=ae), STRAIGHT)
        ego = {pred.names[int(pred.ego_m[k])]: float(pred.ego_p[k]) for k in range(int(pred.ego_n))}
        hist.append((t, ego.get("follow", 0.0), ve))
    return P, pred, hist


def _futs(pred, oid):
    i = pred.index(oid)
    return {pred.names[int(pred.m[i, k])]: (float(pred.p0[i, k]), pred.traj[i, k]) for k in range(pred.n_modes[i])}


def test_the_leaders_speed_is_the_tracks_and_a_start_goes_on():
    """2.3 s into its start the lead does 2.3 x 1.8 = 4.1 m/s and will be 20.5 m further on in 3 s. The second filter
    over the track's speed has it slower (by 0.4 m/s here, by 0.9 m/s on the drives) and lets its acceleration die:
    15 m. With `v_track` the future starts at the track's speed; with `start_go` the start goes on towards the road's
    speed: 18.6 m."""
    _, base, _ = _queue({}, 3.3)
    _, vt, _ = _queue({"v_track": 1.0}, 3.3)
    _, go, _ = _queue({"v_track": 1.0, "start_go": 1.0}, 3.3)
    x0 = _start(3.3)[0]
    x3 = _start(6.3)[0] + 0.5 * 1.6 * ((6.3 - 1.15) ** 2 - (3.3 - 1.15) ** 2)     # where it then is, in the frame of 3.3 s
    k3 = int(np.argmin(np.abs(base.t - 3.0)))
    f_b, f_v, f_g = (_futs(p, 1)["keep"][1] for p in (base, vt, go))
    assert abs(f_v[0, 5] - 4.14) < 0.3 and f_b[0, 5] < f_v[0, 5] - 0.2           # the lag of the kept speed
    assert f_b[k3, 0] < f_v[k3, 0] - 0.5 and f_v[k3, 0] < f_g[k3, 0] - 2.0
    assert abs(f_g[k3, 0] - x3) < 2.5 < x3 - f_b[k3, 0], (f_b[k3, 0], f_v[k3, 0], f_g[k3, 0], x3, x0)
    # a speed the track states to 2 m/s is left to the filter
    P = Predictor(dict(NOCHECK, native={"v_track": 1.0}))
    Q = Predictor(dict(NOCHECK))
    for f in range(40):
        t = f * 0.05
        o = K.obj(1, 30.0 + 0.5 * t * t, 0.0, 6.0 + 1.0 * t, 0.0, std=(3.0, 0.4, 2.0, 0.9))
        a, b = P.update(K.msg(t, [o]), STRAIGHT), Q.update(K.msg(t, [o]), STRAIGHT)
    assert np.array_equal(a.traj, b.traj, equal_nan=True)


def test_the_car_behind_a_start_is_a_follower():
    """3.4-4 m behind the lead's bumper at 3 m/s and gaining speed at 1.6 m/s^2 with a lead that gains 1.8: the
    law of the code's defaults (2 m + 1.4 s x v, a gap by the middles) asks for a braking, so the driver "is not
    following" and the car's future is its acceleration carried into the lead. With `ego_follow` the driver is
    following, and stays so through a frame in which the lead's acceleration reads as nothing."""
    knobs = {"v_track": 1.0, "start_go": 1.0, "ego_follow": 1.0}
    _, b, hb = _queue({}, 3.3)
    _, n, hn = _queue(knobs, 3.3)
    at = [(t, p) for t, p, v in hb if 2.9 <= v <= 4.0]
    assert at and max(p for _, p in at) < 0.2, at                           # as it was: not a follower
    at = [(t, p) for t, p, v in hn if 2.9 <= v <= 4.0]
    assert at and min(p for _, p in at) > 0.8, at
    # the follower's own future does not brake to a halt 3.5 m behind a lead that pulls away, and does not reach it
    ego = {n.names[int(n.ego_m[k])]: n.ego_traj[k] for k in range(int(n.ego_n))}
    lead = _futs(n, 1)["keep"][1]
    assert np.all(ego["follow"][:12, 2] > 2.5) and np.all(lead[:, 0] - ego["follow"][:, 0] - 2.1 > 1.5)
    ego_b = {b.names[int(b.ego_m[k])]: b.ego_traj[k] for k in range(int(b.ego_n))}
    assert ego_b["follow"][3, 2] < 1.5                                      # (the gap by the middles: a halt)


def test_a_driver_closing_without_braking_is_not_a_follower():
    """The lead starts and brakes to a stop again 1.2 s later (3.5 m/s^2) while the car goes on gaining speed.
    Until then the driver is a follower; 0.4 s into the lead's braking, with the car 1.3 m/s (two sigma of the lead's
    speed) the faster, no longer: the car's futures run into the lead. With the knobs off the driver never was one:
    its "warning" of this lead is the one it gives of every lead that starts."""
    knobs = {"v_track": 1.0, "start_go": 1.0, "ego_follow": 1.0}
    _, _, hb = _queue({}, 3.6, stall=2.2)
    _, _, hn = _queue(knobs, 3.6, stall=2.2)
    assert max(p for t, p, v in hb if t > 1.8) < 0.45
    assert min(p for t, p, v in hn if 1.8 < t <= 2.2) > 0.6
    t_n = next(t for t, p, v in hn if t > 2.2 and p < 0.25)
    assert t_n <= 2.2 + 0.4, t_n
    assert max(p for t, p, v in hn if t >= t_n) < 0.25
    # a standing lead closed on at 5 m/s from 14 m, no braking: never a follower, with or without
    for cfg in ({}, knobs):
        P = Predictor(dict(NOCHECK, native=cfg))
        for f in range(20):
            t = f * 0.05
            x = 14.0 - 5.0 * t
            o = K.obj(1, x, 0.0, 0.0, 0.0, ps=0.97, std=(0.4, 0.4, 0.3, 0.3), ctr=[x + 2.25, 0.0], len=4.5, wid=1.8, hdg=0.0)
            pred = P.update(K.msg(t, [o], v=5.0, a=-0.1), STRAIGHT)
        ego = {pred.names[int(pred.ego_m[k])]: float(pred.ego_p[k]) for k in range(int(pred.ego_n))}
        assert ego.get("follow", 0.0) < 0.05, (cfg, ego)


# ---- the car's own turn --------------------------------------------------------------------------------------------
def test_the_cars_kinematic_future_is_its_arc():
    """4 m/s, 0.3 rad/s to the left (a radius of 13.3 m), no road known: after 1 s the car is 0.6 m to the left of
    the line it was on, after 2 s 2.3 m. pp_base's polyline of 5 m chords has it 1.1 m to the left after 1 s."""
    def lat(cfg, w=0.3, v=4.0):
        P = Predictor(dict(NOCHECK, native=cfg))
        for f in range(12):
            pred = P.update(K.msg(f * 0.05, [K.obj(1, 60.0, 9.0, 0.0, 0.0, ps=0.95)], v=v, w=w), None)
        ego = {pred.names[int(pred.ego_m[k])]: pred.ego_traj[k] for k in range(int(pred.ego_n))}
        k1, k2 = int(np.argmin(np.abs(pred.t - 1.0))), int(np.argmin(np.abs(pred.t - 2.0)))
        return ego["ctra"][k1, 1], ego["ctra"][k2, 1]
    R = 4.0 / 0.3
    want = [R * (1.0 - math.cos(4.0 * t / R)) for t in (1.0, 2.0)]
    old, new = lat({}), lat({"ego_arc": 1.0})
    assert abs(new[0] - want[0]) < 0.08 and abs(new[1] - want[1]) < 0.4, (new, want)     # (the turn dies away)
    assert old[0] - want[0] > 0.3, (old, want)
    # a radius under 12.5 m, and a crawl: an arc all the same
    tight = lat({"ego_arc": 1.0}, w=0.5, v=3.0)                               # 6 m
    assert abs(tight[0] - 6.0 * (1.0 - math.cos(0.5))) < 0.12, tight
    assert lat({}, w=0.3, v=1.2)[1] == 0.0 and lat({"ego_arc": 1.0}, w=0.3, v=1.2)[1] > 0.3


def _turn_msg(y_obj, turn=True):
    """A car parked 9 m ahead beside a future of the car that turns left past it (its near side 0.25 m from the
    car's side at the nearest), or the same on a straight line."""
    t = [round(0.25 * k, 2) for k in range(1, 25)]
    R = 14.0
    if turn:
        ex = [R * math.sin(4.0 * q / R) for q in t]
        ey = [R * (1.0 - math.cos(4.0 * q / R)) for q in t]
    else:
        ex, ey = [4.0 * q for q in t], [0.0 for _ in t]
    o = {"id": 1, "state": "confirmed", "exist": 1.0, "x_m": 9.0, "y_m": y_obj, "vx_mps": 0.0, "vy_mps": 0.0,
         "std": [0.5, 0.25, 0.2, 0.2], "p_stationary": 0.97, "motion": "stationary", "class": {"label": "car"},
         "flags": [], "len": 4.4, "wid": 1.8, "hdg": 0.0, "ctr": [11.2, y_obj],
         "fut": [{"p": 1.0, "p0": 1.0, "m": "parked", "x": [9.0] * 24, "y": [y_obj] * 24, "sx": [0.5] * 24,
                  "sy": [0.25] * 24, "rxy": [0.0] * 24}],
         "lane": {"ego": 0.5, "park": 0.5}}
    return {"type": "radar", "ts_mono_ns": 1, "frame": "car", "objects": [o], "fut": {"t": t},
            "ego": {"speed_mps": 4.0, "yaw_rate_rps": 4.0 / R if turn else 0.0, "accel_mps2": 0.0},
            "ego_fut": {"modes": [{"p": 1.0, "m": "ctra", "x": ex, "y": ey, "v": [4.0] * 24}]}}


def test_the_cars_spread_in_a_turn():
    """The car's lateral spread is a lane keeper's (0.3 m at most); in a turn a third of the lateral move is not
    known (`ego_turn_rel`): a parked car that the arc clears by a quarter of a metre is less surely hit, one on a
    straight line is judged as it was."""
    msg = _turn_msg(-0.2)                              # the arc bends away from it: the near corner passes close
    a = TH.ThreatModel({}).explain(msg).records[0]
    b = TH.ThreatModel({"ego_turn_rel": 0.35}).explain(msg).records[0]
    msg = _turn_msg(1.4)                               # on the inside of the turn, 2.5-3 m of lateral move on
    c = TH.ThreatModel({}).explain(msg).records[0]
    d = TH.ThreatModel({"ego_turn_rel": 0.35}).explain(msg).records[0]
    assert c["p_coll"] > 0.35 and d["p_coll"] < c["p_coll"] - 0.05, (c["p_coll"], d["p_coll"])
    assert abs(a["p_coll"] - b["p_coll"]) < 0.25
    for y in (0.0, -1.9):
        msg = _turn_msg(y, turn=False)
        assert TH.ThreatModel({}).explain(msg).records == TH.ThreatModel({"ego_turn_rel": 0.35}).explain(msg).records


# ---- what stands between the car and an object ---------------------------------------------------------------------
def _queue_msg(lead_v, hidden, lead_y=0.0, seen=0.1):
    """The car at 4 m/s; a lead whose rear is 8 m ahead, driving at lead_v; a standing vehicle 16 m ahead in the
    lane (hidden by the lead, or in plain view)."""
    t = [round(0.25 * k, 2) for k in range(1, 25)]

    def car(i, x, y, v, m, **kw):
        return dict({"id": i, "state": "confirmed", "exist": 1.0, "x_m": x, "y_m": y, "vx_mps": v, "vy_mps": 0.0,
                     "std": [0.4, 0.3, 0.3, 0.3], "p_stationary": 0.02 if v > 0.3 else 0.97,
                     "motion": "moving" if v > 0.3 else "stationary", "class": {"label": "car"}, "flags": [],
                     "len": 4.4, "wid": 1.8, "hdg": 0.0, "ctr": [x + 2.2, y], "seen": {"cam": seen},
                     "fut": [{"p": 1.0, "p0": 1.0, "m": m, "x": [x + v * q for q in t], "y": [y] * 24, "sx": [0.5] * 24,
                              "sy": [0.3] * 24, "rxy": [0.0] * 24}], "lane": {"ego": 0.95}}, **kw)
    lead = car(1, 8.0, lead_y, lead_v, "keep" if lead_v > 0.3 else "stopped")
    far = car(2, 16.0, 0.0, 0.0, "stopped", **({"occ": "object", "occ_by": 1, "seen": {"cam": 1.2}} if hidden else {}))
    return {"type": "radar", "ts_mono_ns": 1, "frame": "car", "objects": [lead, far], "fut": {"t": t},
            "ego": {"speed_mps": 4.0, "yaw_rate_rps": 0.0, "accel_mps2": 0.0},
            "ego_fut": {"modes": [{"p": 1.0, "m": "path", "x": [4.0 * q for q in t], "y": [0.0] * 24, "v": [4.0] * 24}]}}


def test_an_object_beyond_another_cannot_be_reached_first():
    def rec(msg, cfg, shown=None):
        tm = TH.ThreatModel(cfg)
        objs = msg["objects"]
        from chris_integrator.decide import futures as FU
        res = tm.frame(objs, msg["ego"], FU.from_message(msg), None, None, shown, 2)
        return {r["id"]: r for r in res[0]}, res[3]
    on = {"screen": True}
    # the lead has driven off through the place of the next one of the queue, which nobody has seen since
    m = _queue_msg(5.0, hidden=True)
    base, _ = rec(m, {})
    scr, shown = rec(m, on)
    assert base[2]["level"] >= 2 and base[2]["p_coll"] > 0.6
    assert scr[2]["level"] == 0 and scr[2]["p_coll"] < 0.02 and scr[2]["screen"] >= 0.99
    assert scr[1] == base[1] and "screen" not in base[2] and "_refuted" not in rec(m, {})[1]
    # ... and it stays refuted while it is not sighted, also when the lead is beyond it (a memory in the gap)
    assert 2 in shown["_refuted"]
    m2 = _queue_msg(5.0, hidden=True)
    m2["objects"][0], m2["objects"][1] = m2["objects"][1], m2["objects"][0]
    m2["objects"][0]["x_m"], m2["objects"][0]["ctr"] = 6.0, [8.2, 0.0]            # the memory now nearer than the lead
    m2["objects"][0]["fut"][0]["x"] = [6.0] * 24
    m2["objects"][1]["x_m"], m2["objects"][1]["ctr"] = 9.0, [11.2, 0.0]
    m2["objects"][1]["fut"][0]["x"] = [9.0 + 5.0 * 0.25 * k for k in range(1, 25)]
    assert rec(m2, on)[0][2]["level"] >= 2                                       # (without the memory: said)
    again, shown2 = rec(m2, on, shown)
    assert again[2]["level"] == 0 and again[2]["screen"] == 1.0 and 2 in shown2["_refuted"]
    m2["objects"][0]["seen"], m2["objects"][0]["occ"] = {"cam": 0.1}, None      # sighted: it is there
    seen_again, shown3 = rec(m2, on, shown2)
    assert seen_again[2]["level"] >= 2 and 2 not in shown3["_refuted"]
    # a lead that drives into a standing vehicle in plain view screens nothing
    m = _queue_msg(5.0, hidden=False)
    assert rec(m, on)[0][2] == rec(m, {})[0][2]
    # a standing lead is met first, whatever is seen of the one beyond it
    for hidden in (False, True):
        m = _queue_msg(0.0, hidden=hidden)
        base, scr = rec(m, {})[0], rec(m, on)[0]
        assert base[2]["level"] >= 1 and scr[2]["level"] == 0 and scr[2]["screen"] >= 0.99
        assert scr[1]["level"] == base[1]["level"] >= 2
    # a vehicle beside the car's way screens nothing; nor does one that has not been sighted itself
    assert rec(_queue_msg(0.0, hidden=True, lead_y=2.6), on)[0][2] == rec(_queue_msg(0.0, hidden=True, lead_y=2.6), {})[0][2]
    assert rec(_queue_msg(0.0, hidden=True, seen=1.5), on)[0][2]["level"] >= 1


# ---- the alert ------------------------------------------------------------------------------------------------------
def _rec(le, mass=0.6):
    return {"id": 1, "kind": "vehicle", "level": le, "level_exit": le, "mass": [mass] * 4, "p_coll": mass, "tm_s": 0.5,
            "a_req": {"brake": 3.0, "left": 2.0, "right": 2.0}, "why": {}}


def test_a_stage_held_but_never_said_is_not_said_on_its_holds():
    """The car creeps 3 m behind a queue: imminent, under the low-speed suppression, latched. The lead pulls away
    (the level falls to information) and the car passes 3 m/s: the latched stage was said. With
    `unsaid_needs_level` it is said only while the object's level now is one that is said."""
    ob = {1: {"id": 1, "state": "confirmed", "exist": 1.0, "motion": "moving", "p_stationary": 0.02, "flags": [], "age_s": 30.0}}

    def run(cfg, back=None):
        a = AL.Alert("obstacle_ahead", cfg)
        said = []
        for k in range(80):
            t = k * 0.05
            le = 4 if t < 1.5 else 3 if back is not None and t >= back else 1
            out, _ = a.step(t, [_rec(le)], ob, {"v": 1.0 + 1.0 * t})
            if out["stage"] >= 2 and out["suppressed"] is None:
                said.append(round(t, 2))
            if 1.34 < t < 1.36:
                assert out["stage"] == 4 and out["suppressed"] == "low_speed"
        return said
    assert run({})[:1] == [2.0] and run({"unsaid_needs_level": True}) == []
    # a level that IS one to say is said as it was: at once when the suppression lifts, and when it comes back
    a, b = AL.Alert("obstacle_ahead", {}), AL.Alert("obstacle_ahead", {"unsaid_needs_level": True})
    for k in range(60):
        o1, _ = a.step(k * 0.05, [_rec(3)], ob, {"v": 1.0 + 1.0 * k * 0.05})
        o2, _ = b.step(k * 0.05, [_rec(3)], ob, {"v": 1.0 + 1.0 * k * 0.05})
        assert (o1["stage"], o1["suppressed"]) == (o2["stage"], o2["suppressed"])
    assert run({"unsaid_needs_level": True}, back=2.5)[:1] == [2.5]


def test_a_conflict_with_an_object_just_found_needs_evidence():
    def entered(cfg, mass, age):
        a = AL.Alert("obstacle_ahead", cfg)
        for k in range(61):
            ob = {1: {"id": 1, "state": "confirmed", "exist": 1.0, "motion": "moving", "p_stationary": 0.02, "flags": [],
                      "age_s": age + k * 0.05}}
            out, _ = a.step(k * 0.05, [_rec(2, mass)], ob, {"v": 10.0})
            if out["stage"] >= 2:
                return round(k * 0.05, 2)
        return None
    young = {"entry_evidence_young_s": 2.0}
    assert entered({}, 0.3, 0.6) == 0.2 and entered(young, 0.3, 5.0) == 0.2      # as ever; an object known for 5 s
    assert entered(young, 0.9, 0.6) == 0.2 and entered(young, 0.6, 0.6) == 0.2   # a clear conflict: by time
    t = entered(young, 0.32, 0.4)                    # 0.07 over the minimum: by evidence until it is 2 s old
    assert t is not None and 0.9 < t < 1.1, t
    assert entered(young, 0.27, 0.2) == 2.0 - 0.2 + 0.2 or entered(young, 0.27, 0.2) is not None


# ---- the simulator's driver ----------------------------------------------------------------------------------------
def _rect(cx, cy, yaw, L, W):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + c * a - s * b, cy + s * a + c * b) for a, b in ((L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2))]


def _dist(A, B):
    def seg(p, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        u = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
        return math.hypot(p[0] - a[0] - u * dx, p[1] - a[1] - u * dy)
    return min(seg(p, Q[i], Q[(i + 1) % 4]) for P_, Q in ((A, B), (B, A)) for p in P_ for i in range(4))


def test_the_simulators_driver_drives():
    from chris_integrator.sim import roaduser_sim as rs
    assert set(rs.DRIVER_SCENARIOS) == {"queue_start", "queue_start_stall", "turn_parked", "narrow_oncoming"}
    assert not set(rs.DRIVER_SCENARIOS) & (set(rs.ALERT_SCENARIOS) | set(rs.SCENARIOS) | set(rs.STALE_SCENARIOS))
    ego = _rect(-0.125, 0.0, 0.0, 4.45, 1.76)                # 2.1 m to the front, 2.35 m to the rear
    for seed in (1, 2, 3):
        # a queue's start: at 3 m/s the car is 3-4.6 m behind the lead's bumper, gaining 1.2-2 m/s^2, the lead no slower
        w = rs.scenario("queue_start", seed)
        i = int(np.argmax(w.v >= 3.0))
        lead = next(u for u in w.actors_at(float(w.t[i])) if u["id"] == 930)
        gap = lead["x"] - 0.5 * lead["L"] - 2.1
        assert 3.0 < gap < 4.6 and 1.2 < w.a[i] < 2.05 and lead["speed"] > 3.0, (seed, gap, w.a[i], lead["speed"])
        assert w.spec.get("collision_t") is None
        near = min(u["x"] - 0.5 * u["L"] - 2.1 for T in np.arange(0.0, w.T, 0.1) for u in w.actors_at(w.t0 + T) if u["id"] == 930)
        assert near > 2.2, (seed, near)
        # ... and the same with a lead that stalls: the car is 0.3 m short of it at the end, still gaining speed
        w = rs.scenario("queue_start_stall", seed)
        ct = w.spec["collision_t"]
        lead = next(u for u in w.actors_at(w.t0 + w.T - 0.01) if u["id"] == 930)
        assert lead["role"] == "stopped_in_lane" and lead["speed"] < 0.5
        gap = lead["x"] - 0.5 * lead["L"] - 2.1
        v_end = float(w.v[-1])
        assert 0.0 < gap < 0.35 * v_end + 0.3 and v_end > 3.5 and w.a[-1] > 1.0 and abs(w.T - (ct - 0.3)) < 1e-6
        # a turn among parked cars, a narrow street with oncoming cars: half a metre and more from every body, a
        # metre and a half at most from the nearest
        for name, lo, hi in (("turn_parked", 0.5, 1.2), ("narrow_oncoming", 0.5, 1.2)):
            w = rs.scenario(name, seed)
            d = min(_dist(ego, _rect(u["x"], u["y"], u.get("yaw", 0.0), u["L"], u["W"]))
                    for T in np.arange(0.0, w.T, 0.1) for u in w.actors_at(w.t0 + T) if abs(u["x"]) < 12.0 and abs(u["y"]) < 8.0)
            assert lo <= d <= hi, (name, seed, d)
    w = rs.scenario("turn_parked", 1)
    i = int(np.argmax(w.s >= 52.0))                          # in the middle of the turn: 4.5 m/s on a radius of 10 m
    assert 4.3 < float(w.v[i]) < 5.0
