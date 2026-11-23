#!/usr/bin/env python3
"""Evaluation of the predictor (chris_integrator/predict) off line: simulator scenarios with the truth, and
recorded captures against the tracker's own later word.

    tools/predict_eval.py sim [--scenario NAME ...] [--seeds 2] [--night] [--context sim|stack|none]
                              [--objects radar|truth] [--native KEY=VALUE ...] [--json OUT.json] [--trace ACTOR]
    tools/predict_eval.py capture CAPTURE.cap.gz [--limit N] [--pack PACK] [--json OUT.json]

`sim`: each scenario (chris_integrator/sim/roaduser_sim.py; default: the predictor's own, PREDICT_SCENARIOS) runs
through the whole stack with the detector simulator; every radar message goes through a Predictor (`--objects
truth`: a message made from the truth with a small honest error instead: the predictor apart from the tracker)
with the chosen road context:

    sim     the lanes, junction arms and crossings as the scenario defines them (what a map and a lane tracker would
            give; the simulated road model itself sends a plan only)
    stack   what the stack's own messages give (here: the plan as the centre line, nothing else)
    none    no context: the arc of the car's own motion

Scored against the TRUTH (the actor each radar object is matched to, its true places over the next 8 s, in the
frame of the prediction's time), every 0.5 s per object, for objects the radar holds as seen:

    minADE / minFDE     the best future's mean / final distance, at 1, 2, 3, 5, 8 s
    topFDE              the most probable future's
    cvFDE               the baseline: the object's own velocity carried on (what the radar's `predict` does)
    rel                 the same after taking out the tracker's error at the prediction's time (the object's
                        estimated place against the actor's true reference point): the error of the predicted
                        MOTION alone
    nll, maha2          negative log-likelihood under the futures' mixture; squared Mahalanobis distance to the
                        best future (2 = the stated spread is right, above: too narrow)
    calibration         futures by stated probability against how often they were the best
    lane                share of frames in which the most probable lane kind is the true one, mean P(true lane)
    manoeuvres          for each labelled manoeuvre of the scenario: mean probability of its name while it lasts,
                        when it was first held above 0.5 against its start, and how often the name was above 0.5
                        on actors that were not doing it (false alarms)

`capture`: a recorded drive (tools/cost_capture.py) through the daemon's stack with the predictor's plug-in: the
self-check's running scores (the object's later estimate as what it did), the share of manoeuvres, cost.
Real-drive captures never go into the repository.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

NS = 1_000_000_000
HOR = (1.0, 2.0, 3.0, 5.0, 8.0)
TRACK_SOURCES = [{"kind": "road_users", "id": "*"}]


# ---- one scenario through the stack ---------------------------------------------------------------------------
class IdealTracker:
    """Radar messages made from the truth: every actor near the car as an object with a small, honest error (a
    slowly varying offset and white noise, both stated in `std`), its true class and motion state. What the
    predictor does with a good tracker, apart from what today's tracker gives it."""

    def __init__(self, w, seed):
        import random
        self.w, self.rng, self.bias = w, random.Random(seed * 7 + 5), {}
        self.prev = {}

    def message(self, T, ego):
        rng, objs = self.rng, []
        for a in self.w.actors_at(T, max_range=120.0):
            if a["x"] < -25.0:
                continue
            r = math.hypot(a["x"], a["y"])
            b = self.bias.setdefault(a["id"], (rng.gauss(0, 0.012), rng.gauss(0, 0.1)))
            sr, sl = 0.25 + 0.012 * r, 0.1 + 0.003 * r
            c, s = a["x"] / max(r, 1.0), a["y"] / max(r, 1.0)
            er, el = b[0] * r + rng.gauss(0, 0.5 * sr), b[1] + rng.gauss(0, 0.5 * sl)
            x, y = a["x"] + c * er - s * el, a["y"] + s * er + c * el
            vx, vy = a["vx"] + rng.gauss(0, 0.25), a["vy"] + rng.gauss(0, 0.12)
            pv = self.prev.get(a["id"])
            ax = ay = 0.0
            if pv is not None and T - pv[0] > 1e-3:
                # the true acceleration over the ground (in the car's axes of now), with noise
                dt = T - pv[0]
                ax, ay = (a["vx"] - pv[1]) / dt + rng.gauss(0, 0.2), (a["vy"] - pv[2]) / dt + rng.gauss(0, 0.2)
                ax, ay = 0.8 * pv[3] + 0.2 * ax, 0.8 * pv[4] + 0.2 * ay
            self.prev[a["id"]] = (T, a["vx"], a["vy"], ax, ay)
            sx = math.hypot(c * sr, s * sl)
            sy = math.hypot(s * sr, c * sl)
            lab = {"human": "human", "bike": "bike", "truck": "truck"}.get(a["kind"], "car")
            objs.append({"id": a["id"], "state": "confirmed", "exist": 1.0, "x_m": round(x, 2), "y_m": round(y, 2),
                         "vx_mps": round(vx, 2), "vy_mps": round(vy, 2), "ax_mps2": round(ax, 2), "ay_mps2": round(ay, 2),
                         "speed_mps": round(math.hypot(vx, vy), 2), "std": [round(sx, 2), round(sy, 2), 0.3, 0.15],
                         "p_stationary": 0.02 if a["moving"] else 0.97, "class": {"label": lab},
                         "flags": ["parked"] if a["stationary_label"] == "parked" else [],
                         "len": a["L"], "wid": a["W"], "hdg": [round(a["heading"], 3), 0.05]})
        return {"type": "radar", "ts_mono_ns": int(round(T * NS)), "frame": "car", "objects": objs, "ego": dict(ego)}


def run_scenario(name, seed, night, context="sim", cfg=None, trace=None, objects="radar"):
    from chris_integrator.mixer import MessageClock
    from chris_integrator.predict import ContextBuilder, Predictor
    from chris_integrator.sim import default_sensors
    from chris_integrator.sim import roaduser_sim as rs
    from chris_integrator.stack import build
    w = rs.scenario(name, seed=seed, night=night)
    det = rs.DetectorSim(seed=1000 + seed, night=night)
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed}) + [det]
    st = build(cfg={"estimator": {"sources": TRACK_SOURCES}, "mixer": {"disable": ["predict"]}}, mapper=None,
               clock=MessageClock())
    P = Predictor(cfg)
    cb = ContextBuilder()
    ideal = IdealTracker(w, seed) if objects == "truth" else None
    frames = []
    cost = [0.0, 0]

    def rec(e):
        m = e.msg
        if e.stream == "road":
            cb.on_road(m)
            return
        ts = m["ts_mono_ns"]
        T = ts / NS
        if ideal is not None:
            tr = w.truth_at(T)
            m = ideal.message(T, {"speed_mps": tr["v"], "yaw_rate_rps": tr["w"], "accel_mps2": tr["a"], "ground_frame": 0})
        if context == "sim":
            ctx = w.road_context(T) if hasattr(w, "road_context") else None
        elif context == "stack":
            P.ego.add(T, float(m["ego"]["speed_mps"]), float(m["ego"]["yaw_rate_rps"]))
            ctx = cb.build(T, P.ego)
        else:
            ctx = None
        t0 = time.perf_counter()
        pred = P.update(m, ctx)
        if pred is not None:
            pred.message(m)
        cost[0] += time.perf_counter() - t0
        cost[1] += 1
        if pred is not None:
            frames.append((T, m, pred))

    st.mixer.add_output("eval", rec, streams=("radar", "road"))
    first = True
    for _, m in w.messages(sensors):
        if first:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            first = False
        st.deliver(m)
    return w, frames, P, cost


def _gate(a, x, y):
    """Is the output at (x, y) this actor: along / across its own axis, against its size and the range."""
    dx, dy = x - a["ref_x"], y - a["ref_y"]
    r = math.hypot(a["ref_x"], a["ref_y"])
    b = math.atan2(a["ref_y"], a["ref_x"])
    al, ac = dx * math.cos(b) + dy * math.sin(b), -dx * math.sin(b) + dy * math.cos(b)
    return math.hypot(al / (4.0 + 0.15 * r), ac / (1.5 + 0.03 * r))


def match(objs, actors, prev, ideal=False):
    if ideal:
        ids = {a["id"] for a in actors}
        return {o["id"]: o["id"] for o in objs if o["id"] in ids}
    out, used = {}, set()
    cand = []
    for o in objs:
        if o.get("state") == "memory":
            continue
        for a in actors:
            c = _gate(a, o["x_m"], o["y_m"])
            if c <= 1.0:
                cand.append((c - (0.3 if prev.get(o["id"]) == a["id"] else 0.0), o["id"], a["id"]))
    for c, oid, aid in sorted(cand):
        if oid not in out and aid not in used:
            out[oid] = aid
            used.add(aid)
    return out


def truth_lane(w, a):
    """The lane kind an actor is truly in (by the scenario's layout), or None where that is not defined."""
    lay = w.spec.get("layout") or {}
    lw = 3.5
    off = a["lane_offset"]
    if a["kind"] == "human" or a["role"] in ("pull_out", "onc_turn"):
        return None
    if a["role"] in ("crossing", "yielding", "turn_in"):
        return "cross" if abs(off) > lw + 1.5 else None
    if abs(off) <= lw / 2:
        return "ego"
    if not a["moving"] and a["role"] == "parked":
        return "park"
    k = int(math.floor((abs(off) - lw / 2) / lw)) + 1
    if off > 0:
        nl = lay.get("n_left", 0)
        return "left" if k <= nl else "onc" if lay.get("oncoming") and k <= nl + lay.get("n_oncoming", 1) else "off"
    return "right" if k <= lay.get("n_right", 0) else "off"


def actor_future(w, aid, T0, times):
    """The actor's true reference point at T0 + times, in the car frame of T0 ((n, 2); NaN where it does not
    exist): its centre's path, shifted by where its nearest point lay from its centre at T0."""
    from chris_integrator.sim.world import CAM_TO_AXLE
    tr = w.truth_at(T0)
    c, s = math.cos(tr["psi"]), math.sin(tr["psi"])
    S_, Q_, X, Y, VX, VY, SP, HD = w._A[aid]
    act = next(a for a in w.actors if a.id == aid)
    out = np.full((len(times), 2), math.nan)
    i0 = w._i(T0)
    for k, t in enumerate(times):
        rel = T0 + t - w.t0
        if not (act.t_on <= rel <= act.t_off) or T0 + t > w.t0 + w.T - 0.02:
            continue
        i = w._i(T0 + t)
        dx, dy = float(X[i]) - tr["pe"], float(Y[i]) - tr["pn"]
        out[k] = (c * dx + s * dy - CAM_TO_AXLE, -s * dx + c * dy)
    dx, dy = float(X[i0]) - tr["pe"], float(Y[i0]) - tr["pn"]
    return out, (c * dx + s * dy - CAM_TO_AXLE, -s * dx + c * dy)


class Score:
    def __init__(self, times):
        from chris_integrator.predict.selfcheck import SelfCheck
        self.times = times
        self.abs_ = SelfCheck(times)
        self.rel = SelfCheck(times)
        self.hk = self.abs_.hk
        self.cv = np.zeros((4, len(self.hk), 2))          # n, sum of the baseline's FDE (relative)
        self.lane = {}                                    # truth lane -> [n, right, sum P(true)]
        self.man = {}                                     # manoeuvre -> dict
        self.modes = {}

    def stats(self):
        out = {"abs": self.abs_.stats(), "rel": self.rel.stats(), "cv_fde_rel": {}, "lane": {}, "manoeuvres": self.man,
               "futures": self.modes}
        from chris_integrator.predict.selfcheck import GROUPS
        for g, name in enumerate(GROUPS):
            if self.cv[g, :, 0].any():
                out["cv_fde_rel"][name] = [None if not n else round(s / n, 3) for n, s in self.cv[g]]
        for k, (n, ok, sp) in sorted(self.lane.items()):
            out["lane"][k] = {"n": n, "accuracy": round(ok / n, 3), "p_true": round(sp / n, 3)}
        return out


def score_run(w, frames, sc, every_s=0.5, trace=None, ideal=False):
    from chris_integrator import native
    N = native.predict
    times = sc.times
    out = np.zeros((len(times) + 2, 6))
    prev, nxt = {}, None
    labels = w.spec.get("labels") or []                   # (actor id, t0, t1, manoeuvre names)
    events = {}
    for T, m, pred in frames:
        actors = {a["id"]: a for a in w.actors_at(T)}
        mt = match(m["objects"], actors.values(), prev, ideal)
        prev = mt
        rel_t = T - w.t0
        # lanes and manoeuvres: every frame
        for oid, aid in mt.items():
            i = pred.index(oid)
            if i is None:
                continue
            a = actors[aid]
            tl = truth_lane(w, a)
            if tl is not None:
                e = sc.lane.setdefault(tl, [0, 0, 0.0])
                e[0] += 1
                e[1] += pred.lanes[int(np.argmax(pred.lane[i]))] == tl
                e[2] += float(pred.lane[i][pred.lanes.index(tl)])
            intent = pred.intent_names(i)
            if trace is not None and aid == trace:
                print("%6.2f  x %6.1f y %5.1f  lane %-5s %.2f  d %5.2f  %s" % (
                    rel_t, m["objects"][[o["id"] for o in m["objects"]].index(oid)]["x_m"], a["y"],
                    pred.lanes[int(np.argmax(pred.lane[i]))], float(pred.lane[i].max()), pred.feat[i, 1],
                    " ".join("%s %.2f" % kv for kv in sorted(intent.items(), key=lambda kv: -kv[1])[:5])))
            active = set()
            for li, (lid, t0, t1, names) in enumerate(labels):
                if lid != aid:
                    continue
                p = sum(intent.get(nm, 0.0) for nm in names)
                ev = events.setdefault(li, {"names": names, "t0": t0, "first": None, "sum": 0.0, "n": 0, "pre": 0.0, "npre": 0})
                if t0 <= rel_t <= t1:
                    ev["sum"] += p
                    ev["n"] += 1
                    active.update(names)
                elif t0 - 2.0 <= rel_t < t0:
                    ev["pre"] += p
                    ev["npre"] += 1
                    active.update(names)
                elif t1 < rel_t <= t1 + 1.5:
                    active.update(names)                       # the tail of it is not a false alarm
                if ev["first"] is None and p > 0.5 and t0 - 3.0 <= rel_t <= t1:
                    ev["first"] = rel_t - t0
            for nm, p in intent.items():
                fa = sc.man.setdefault(nm, {"frames_other": 0, "false_alarm": 0})
                if nm not in active:
                    fa["frames_other"] += 1
                    fa["false_alarm"] += p > 0.5
        if nxt is not None and T < nxt - 1e-6:
            continue
        nxt = T + every_s
        for oid, aid in mt.items():
            i = pred.index(oid)
            if i is None or not pred.seen[i]:
                continue
            fut, c0 = actor_future(w, aid, T, times)
            a = actors[aid]
            ref = (a["x"], a["y"]) if ideal else (a["ref_x"], a["ref_y"])     # what the object's place refers to
            fut = fut + np.array([ref[0] - c0[0], ref[1] - c0[1]])
            val = np.isfinite(fut[:, 0]).astype(float)
            if val.sum() < 3:
                continue
            nm = int(pred.n_modes[i])
            p = np.ascontiguousarray(pred.p[i, :nm])
            tr = np.ascontiguousarray(pred.traj[i, :nm])
            act = np.ascontiguousarray(np.nan_to_num(fut))
            win, _ = N.predict_score(tr, p, act, val, out)
            sc.abs_.add(int(pred.kind[i]), out, val, p, win)
            for k in range(nm):
                key = pred.names[int(pred.m[i, k])]
                e = sc.modes.setdefault(key, [0, 0, 0.0])
                e[0] += 1
                e[1] += k == win
                e[2] += float(p[k])
            # the motion alone: both starting from the same point
            est0 = pred.xy[i]
            act_rel = np.ascontiguousarray(act - np.array(ref) + est0)
            win, _ = N.predict_score(tr, p, act_rel, val, out)
            sc.rel.add(int(pred.kind[i]), out, val, p, win)
            o = next(x for x in m["objects"] if x["id"] == oid)
            g = 2 if pred.kind[i] == 7 else 1 if pred.kind[i] == 6 else 0
            for h, k in enumerate(sc.hk):
                if val[k] > 0.5:
                    t = times[k]
                    e = math.hypot(est0[0] + o["vx_mps"] * t - act_rel[k, 0], est0[1] + o["vy_mps"] * t - act_rel[k, 1])
                    for gg in (g, 3):
                        sc.cv[gg, h] += (1, e)
    for li, ev in events.items():
        for nm in ("+".join(ev["names"]),):
            e = sc.man.setdefault(nm, {"frames_other": 0, "false_alarm": 0})
            e.setdefault("events", 0)
            e["events"] += 1
            e.setdefault("mean_p", []).append(round(ev["sum"] / ev["n"], 3) if ev["n"] else None)
            e.setdefault("mean_p_2s_before", []).append(round(ev["pre"] / ev["npre"], 3) if ev["npre"] else None)
            e.setdefault("first_above_half_s", []).append(None if ev["first"] is None else round(ev["first"], 2))


def _row(name, d, key):
    v = d.get(key)
    return "  %-11s %-8s " % (name, key) + " ".join("%7s" % ("-" if x is None else ("%.2f" % x)) for x in v) if v else None


def print_stats(title, s):
    print("== %s" % title)
    for part in ("abs", "rel"):
        st = s[part]
        print(" %s  (horizons %s s; scored %d)" % ("against the truth" if part == "abs" else "motion alone (tracker error at t0 taken out)",
                                                    st.get("horizons_s"), st.get("scored", 0)))
        for g in ("vehicle", "cyclist", "pedestrian", "all"):
            if g not in st:
                continue
            print("  %-11s %-8s " % (g, "n") + " ".join("%7d" % x for x in st[g]["n"]))
            for key in ("min_ade", "min_fde", "top_fde", "nll", "maha2"):
                r = _row(g, st[g], key)
                if r:
                    print(r)
            if part == "rel" and g in s["cv_fde_rel"]:
                print("  %-11s %-8s " % (g, "cv_fde") + " ".join("%7s" % ("-" if x is None else "%.2f" % x) for x in s["cv_fde_rel"][g]))
        if "calibration" in st:
            c = st["calibration"]
            print("  calibration (stated p bin: futures / best): " + "  ".join(
                "%.1f: %d/%d=%.2f" % (0.1 * k + 0.05, b, a, b / a) for k, (a, b) in enumerate(c["bins"]) if a) + "   brier %.3f" % c["brier"])
    if s["lane"]:
        print(" lane: " + "  ".join("%s n %d acc %.2f p %.2f" % (k, v["n"], v["accuracy"], v["p_true"]) for k, v in s["lane"].items()))
    for k, v in sorted(s["manoeuvres"].items()):
        if "events" in v:
            print(" manoeuvre %-22s events %d  mean p %s  2 s before %s  first > 0.5 at (s from start) %s" % (
                k, v["events"], v["mean_p"], v["mean_p_2s_before"], v["first_above_half_s"]))
    fa = ["%s %d/%d" % (k, v["false_alarm"], v["frames_other"]) for k, v in sorted(s["manoeuvres"].items())
          if v["frames_other"] and "events" not in v and v["false_alarm"]]
    if fa:
        print(" held above 0.5 where no such manoeuvre is labelled (frames / object frames): " + ", ".join(fa))
    if s["futures"]:
        print(" futures (count, was best, mean p): " + ", ".join("%s %d %.2f %.2f" % (k, n, b / n, sp / n) for k, (n, b, sp) in sorted(s["futures"].items())))


def cmd_sim(a):
    from chris_integrator.predict.predictor import DEFAULT_CFG, sample_times
    from chris_integrator.sim import roaduser_sim as rs
    names = a.scenario or list(getattr(rs, "PREDICT_SCENARIOS", {})) or list(rs.SCENARIOS)
    if names == ["all"]:
        names = list(getattr(rs, "PREDICT_SCENARIOS", {})) + list(rs.SCENARIOS)
    cfg = {"native": dict(kv.split("=") for kv in a.native or ())}
    cfg["native"] = {k: float(v) for k, v in cfg["native"].items()}
    times = sample_times(DEFAULT_CFG)
    res, total = {}, Score(times)
    tc = [0.0, 0]
    for name in names:
        sc = Score(times)
        for seed in range(1, a.seeds + 1):
            w, frames, P, cost = run_scenario(name, seed, a.night, a.context, cfg, objects=a.objects)
            for s in (sc, total):
                score_run(w, frames, s, trace=a.trace if s is sc else None, ideal=a.objects == "truth")
            tc[0] += cost[0]
            tc[1] += cost[1]
        res[name] = sc.stats()
        print_stats("%s (%d seed%s, %s, context %s, objects %s)" % (name, a.seeds, "" if a.seeds == 1 else "s", "night" if a.night else "day", a.context, a.objects), res[name])
    if len(names) > 1:
        res["ALL"] = total.stats()
        print_stats("ALL", res["ALL"])
    print("cost on this machine: %.3f ms per radar message (update + message), %d messages" % (1e3 * tc[0] / max(1, tc[1]), tc[1]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


def cmd_capture(a):
    import tempfile
    import shutil
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import cost_bench as cbm
    lines = cbm.read_input(a.input, a.limit)
    tmp = tempfile.mkdtemp(prefix="predicteval-")
    try:
        ns = argparse.Namespace(tree=None, no_recorder=True, pack=a.pack)
        tree, d, api, clock = cbm.make(ns, tmp)
        parse = cbm.parser(tree, d.stack)
        counts, nobj, sizes = {}, [0, 0], [0, 0]
        from chris_integrator import native

        def tap(e):
            m = e.msg
            nobj[0] += 1
            for o in m["objects"]:
                if "fut" in o:
                    nobj[1] += 1
                    top = max(o["fut"], key=lambda f: f["p"])
                    counts[top["m"]] = counts.get(top["m"], 0) + 1
            if nobj[0] % 50 == 0:
                sizes[0] += len(native.dumps(m))
                sizes[1] += 1
        d.mixer.add_output("eval", tap, streams=("radar_fut",))
        for name, raw in lines:
            msg, line = parse(raw)
            if msg is None:
                continue
            if name == "feed":
                d.on_message(msg)
            else:
                d.on_message(msg, line)
        slot = d.mixer.slots["predict"]
        h = slot.plugin.health()
        res = {"input": os.path.basename(a.input), "radar_fut_messages": nobj[0], "objects_per_message": round(nobj[1] / max(1, nobj[0]), 2),
               "bytes_per_message": round(sizes[0] / max(1, sizes[1])), "top_future": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
               "ms_per_call_here": round(1e3 * slot.cpu / max(1, slot.calls), 4), "health": h}
        sc = h["selfcheck"]
        print(json.dumps({k: v for k, v in res.items() if k != "health"}, indent=1))
        print_stats("self-check (against the tracker's later estimate)", {"abs": sc, "rel": {}, "cv_fde_rel": {}, "lane": {}, "manoeuvres": {}, "futures": {}})
        print("context:", h.get("context"), "learned:", h.get("learned"))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--scenario", nargs="*")
    s.add_argument("--seeds", type=int, default=2)
    s.add_argument("--night", action="store_true")
    s.add_argument("--context", choices=("sim", "stack", "none"), default="sim")
    s.add_argument("--objects", choices=("radar", "truth"), default="radar",
                   help="radar: the stack's radar; truth: an ideal tracker made from the truth (the predictor alone)")
    s.add_argument("--native", nargs="*", help="native constants, KEY=VALUE (predict_cfg)")
    s.add_argument("--json")
    s.add_argument("--trace", type=int, help="print this actor's lane and intent frame by frame")
    s.set_defaults(fn=cmd_sim)
    c = sub.add_parser("capture")
    c.add_argument("input")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--pack")
    c.add_argument("--json")
    c.set_defaults(fn=cmd_capture)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
