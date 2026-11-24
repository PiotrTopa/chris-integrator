#!/usr/bin/env python3
"""The alerts against the simulator's truth: which said episodes of `obstacle_ahead` were true, arguable or false,
why the false ones were said, and how long before a scripted conflict the alert named the obstacle.

    tools/alert_truth.py run OUT.json [--false A,B] [--true A,B] [--seeds 1-6] [--tseeds 1-3] [--nights 0,1]
                             [--det detin | JSON] [--radar JSON] [--radarbox JSON] [--predict JSON] [--decide JSON]
                             [--fut auto,cv] [--pool N] [--trace]
    tools/alert_truth.py sum A.json [B.json ...] [--fut auto] [-v]
    tools/alert_truth.py cmp BASE.json NEW.json [--fut auto]

run: every scenario of --false (default: the six streets without a conflict and PED_CONTROL_SCENARIOS, a pedestrian
who does not step out) x --seeds x --nights and of --true (default: the scripted conflicts: `stopped_cruise_*`,
TRUE_BEND_SCENARIOS, PED_SCENARIOS) x --tseeds x --nights
through the whole stack, alerts in shadow. --det detin: the detector's realistic stream (roaduser_sim.DETIN).
--radar: the tracker's knobs ({"ctx_prior": true}); --radarbox: the radar box's ({"std_rho": true}); --predict,
--decide: the plug-ins' configuration. --fut cv: the degraded futures. --trace: per frame of a conflict run, the
obstacle's truth beside the track nearest it and what prediction and decision made of it (item by item, for
diagnosis). One row per said episode with the frame's arrays (tools/alert_truth_lib.py has the classes and causes).

sum: per file, the said episodes of the streets by scenario (true / arguable / false, the false ones by cause), the
conflict runs' other episodes, and per conflict scenario by day and night: runs warned on the obstacle / runs, the
median and the least lead. cmp: the paired difference NEW - BASE over the same runs (false episodes, said episodes,
conflicts warned), bootstrapped over seeds, and every conflict run whose warning was lost, gained or moved.

Wrap long runs in a memory cap (systemd-run --user --wait --collect --pipe -q -p MemoryMax=6G ...). The older
count (every said episode of a scenario without a scripted collision is false) calls a slow lead the car closes on,
or a pedestrian walking into the lane, false; use this one."""
import argparse
import json
import math
import os
import sys
from collections import deque

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import alert_truth_lib as L  # noqa: E402

NS = 1e9
FALSE_SCENARIOS = ("parked_street", "curve_parked", "slow_lead", "adjacent", "oncoming", "junction")
STOPPED_SCENARIOS = ("stopped_cruise_14", "stopped_cruise_25", "stopped_brake_14")


def true_scenarios():
    from chris_integrator.sim import roaduser_sim as rs
    return STOPPED_SCENARIOS + tuple(rs.TRUE_BEND_SCENARIOS) + tuple(getattr(rs, "PED_SCENARIOS", ()))


def false_scenarios():
    """The streets without a conflict, and the pedestrian scenarios' controls (a pedestrian who does not step out)."""
    from chris_integrator.sim import roaduser_sim as rs
    return FALSE_SCENARIOS + tuple(getattr(rs, "PED_CONTROL_SCENARIOS", ()))


def frenet(w, X, Y):
    """World point -> (s, d left) of the world's road line (the car's path)."""
    import numpy as np
    i = int(np.argmin((w.px - X) ** 2 + (w.py - Y) ** 2))
    hd = float(w.heading[i])
    dx, dy = X - float(w.px[i]), Y - float(w.py[i])
    return float(w.S[i]) + dx * math.cos(hd) + dy * math.sin(hd), -dx * math.sin(hd) + dy * math.cos(hd)


def car_to_world(tr, x, y):
    from chris_integrator.sim.world import CAM_TO_AXLE
    c, s = math.cos(tr["psi"]), math.sin(tr["psi"])
    xx = x + CAM_TO_AXLE
    return tr["pe"] + c * xx - s * y, tr["pn"] + s * xx + c * y


def match_score(u, o):
    """How far a truth actor is from a radar object: mostly across the line of sight (a camera's range may be far
    off), a little along it. Under about 6: the same thing."""
    d = math.hypot(u["ref_x"] - o["x_m"], u["ref_y"] - o["y_m"])
    b = abs(math.atan2(u["ref_y"], u["ref_x"]) - math.atan2(o["y_m"], o["x_m"])) * max(5.0, u["range"])
    return 3.0 * b + 0.3 * d


def run_one(job):
    name, seed, night, fut, A = job
    import numpy as np
    from chris_integrator.sim import roaduser_sim as rs
    if A.get("det"):
        det_cfg = rs.det_params_arg(A["det"])
        os.environ["CHRIS_SIM_DET"] = json.dumps(det_cfg)
    else:
        os.environ.pop("CHRIS_SIM_DET", None)
    from chris_integrator.sim import default_sensors
    from chris_integrator.mixer import MessageClock
    from chris_integrator.stack import build
    from chris_integrator.decide import threat as TH
    import decide_replay as DR
    cfg = json.loads(A["decide"]) if A.get("decide") else {}
    if fut == "cv":
        cfg["futures"] = "cv"
    w = rs.scenario(name, seed=seed, night=night)
    det = rs.DetectorSim(seed=1000 + seed, night=night)
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
    sensors.append(det)
    c = {"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}, "radar": {"rate_hz": 1000.0},
         "decide": dict(cfg, rate_hz=1000.0)}
    if A.get("radarbox"):
        c["radar"].update(json.loads(A["radarbox"]))
    if A.get("radar"):
        c["radar"]["tracker"] = json.loads(A["radar"])
    if A.get("predict"):
        c["predict"] = json.loads(A["predict"])
    st = build(cfg=c, mapper=None, clock=MessageClock())
    col = DR.Collector()
    col.show_from = DR.show_from(st)
    st.mixer.add_output("decide_replay", col.on, streams=("decide", "radar"))
    dec = st.mixer.plugin("decide")
    try:
        prd = st.mixer.plugin("predict")
    except Exception:
        prd = None
    model = dec.model
    last = {}
    frame0 = model.frame

    def frame(objects, ego, futures=None, path=None, road=None, shown=None, opt=0):
        res = frame0(objects, ego, futures, path, road, shown, opt | 2)
        last["ex"] = TH.Explained(res, model)
        last["objs"] = {o.get("id"): o for o in objects if isinstance(o, dict)}
        return res
    model.frame = frame
    trace = []
    hist = deque(maxlen=120)
    rows = []
    state = {"said": False, "first_true": None, "first_any": None, "other": {}, "T": None, "t0": None}
    show = col.show_from.get("obstacle_ahead", 2)
    ct = w.spec.get("collision_t")
    curv = np.gradient(w.heading, w.S)

    def truth_match(T, o):
        best, bd = None, 1e9
        for u in w.actors_at(T):
            sc = match_score(u, o)
            if sc < bd:
                best, bd = u, sc
        return best, bd

    def place(tr, x, y):
        return frenet(w, *car_to_world(tr, x, y))

    def summary():
        ex = last.get("ex")
        out = {}
        if ex is None or ex.obj is None:
            return out
        for r in ex.records:
            o = last["objs"].get(r["id"]) or {}
            out[r["id"]] = (r.get("level"), r.get("level_exit"), r.get("p_coll"), (r.get("mass") or [0, 0, 0, 0])[1],
                            o.get("x_m"), o.get("y_m"), o.get("vx_mps"), o.get("vy_mps"), o.get("p_stationary"))
        return out

    def trace_frame(T, m, al):
        """The obstacle's truth beside the track nearest it, and what was made of it."""
        tr = w.truth_at(T)
        objs = last.get("objs") or {}
        recs = {r["id"]: r for r in (last["ex"].records if last.get("ex") is not None else ())}
        for u in w.actors_at(T):
            if u["role"] not in L.OBSTACLE_ROLES and u["role"] != "kerb_ped":
                continue
            row = {"t_to": round(ct - (T - w.t0), 2) if ct else round(T - w.t0, 2), "role": u["role"],
                   "truth": {"ref": [round(u["ref_x"], 2), round(u["ref_y"], 2)], "ctr": [round(u["x"], 2), round(u["y"], 2)],
                             "moving": u["moving"], "speed": round(u["speed"], 2), "W": u["W"], "L": u["L"]}}
            s_c, d_c = place(tr, u["x"], u["y"])
            row["truth"]["d_centre"] = round(d_c, 2)
            row["truth"]["clear_m"] = round(abs(d_c) - 0.5 * u["W"] - L.EGO_HALF_W, 2)
            best, bd = None, 6.0
            for o in objs.values():
                if o.get("x_m") is None or o["x_m"] < 1.0:
                    continue
                sc = match_score(u, o)
                if sc < bd:
                    best, bd = o, sc
            if best is not None:
                o = best
                s_o, d_o = place(tr, o["x_m"], o["y_m"])
                cx, cy = o.get("ctr") or (o["x_m"], o["y_m"])
                d_oc = place(tr, cx, cy)[1]
                d_t = place(tr, u["ref_x"], u["ref_y"])[1]
                wid = o.get("wid") or 0.0
                row["track"] = {"id": o.get("id"), "state": o.get("state"), "age_s": o.get("age_s"), "exist": o.get("exist"),
                                "x": round(o["x_m"], 2), "y": round(o["y_m"], 2), "std": o.get("std"), "std_rho": o.get("std_rho"),
                                "v": round(math.hypot(o.get("vx_mps") or 0.0, o.get("vy_mps") or 0.0), 2),
                                "p_stat": o.get("p_stationary"), "motion": o.get("motion"), "class": (o.get("class") or {}).get("label"),
                                "wid": o.get("wid"), "len": o.get("len"), "hdg": o.get("hdg"), "ext_std": o.get("ext_std"),
                                "flags": o.get("flags"), "vis": o.get("vis"), "path_offset_m": o.get("path_offset_m"),
                                "range_err": round(math.hypot(o["x_m"], o["y_m"]) - u["range"], 2),
                                "d_ref": round(d_o, 2), "d_ref_err": round(d_o - d_t, 2),
                                "d_ctr": round(d_oc, 2), "d_ctr_err": round(d_oc - d_c, 2),
                                "wid_err": round(wid - u["W"], 2) if wid else None,
                                "clear_m": round(abs(d_oc) - 0.5 * (wid or u["W"]) - L.EGO_HALF_W, 2)}
                r = recs.get(o.get("id"))
                if r is not None:
                    why = r.get("why") or {}
                    row["rec"] = {"level": r.get("level"), "mass": r.get("mass"), "p_coll": r.get("p_coll"),
                                  "p_place": r.get("p_place"), "lane": r.get("lane"), "mode": why.get("mode"),
                                  "pc": why.get("pc"), "tm_s": r.get("tm_s"), "ttc_s": r.get("ttc_s"), "tn": r.get("tn"),
                                  "hold": r.get("hold"), "in_path": r.get("in_path")}
                    ex = last["ex"]
                    idx = next((i for i, q in enumerate(ex.records) if q["id"] == o.get("id")), None)
                    if idx is not None and ex.obj is not None:
                        k0, nk = int(ex.ohdr[idx, 0]), int(ex.ohdr[idx, 1])
                        row["modes"] = [[ex.names[k], round(float(ex.p[k, 0]), 3), round(float(ex.modes[k][0]), 3),
                                         round(float(np.nanmax(ex.pair[k, :, TH.DP_PC])), 3)] for k in range(k0, k0 + nk)]
            if al is not None:
                row["alert"] = [al["stage"], al.get("object")]
            trace.append(row)

    def on_decide(e):
        if e.stream != "decide":
            return
        m = e.msg
        T = m["ts_mono_ns"] / NS
        hist.append((T, summary()))
        al = next((x for x in m.get("alerts") or () if x["id"] == "obstacle_ahead"), None)
        if A.get("trace") and ct and last.get("ex") is not None:
            trace_frame(T, m, al)
        dT = 0.0 if state["T"] is None else min(0.5, max(0.0, T - state["T"]))
        state["T"] = T
        if state["t0"] is None:
            state["t0"] = T
        for x in m.get("alerts") or ():                  # the other alerts (a low stage of their own: `hidden_gap`)
            if x["id"] in ("obstacle_ahead", "headway"):
                continue
            e = state["other"].setdefault(x["id"], {"first": None, "first_said": None, "said_s": 0.0})
            if x["stage"] >= 1 and e["first"] is None:
                e["first"] = round(T - w.t0, 2)
            if x["stage"] >= show and not x.get("suppressed"):
                e["said_s"] += dT
                if e["first_said"] is None:
                    e["first_said"] = round(T - w.t0, 2)
        if al is None:
            return
        if al["stage"] == 0:
            state["said"] = False
            return
        would = al["stage"] >= show and not al.get("suppressed")
        if would and state["first_true"] is None and ct:
            o1 = last["objs"].get(al.get("object")) or {}
            if o1.get("x_m") is not None:
                u1, _ = truth_match(T, o1)
                # (a pedestrian is warned of by a pedestrian's track: a parked van's track on the same line of sight,
                # said for its own reasons, is not a warning of the person behind it)
                if u1 is not None and u1["role"] in L.OBSTACLE_ROLES and (
                        u1["kind"] != "human" or (al.get("reason") or {}).get("kind") == "vru"):
                    state["first_true"] = (T - w.t0, al["stage"])
        if not would or state["said"]:
            if state["said"] and rows:
                rows[-1]["max_stage"] = max(rows[-1]["max_stage"], al["stage"])
                rows[-1]["end_t"] = round(T - w.t0, 2)
            return
        state["said"] = True
        if state["first_any"] is None:
            state["first_any"] = T - w.t0
        oid = al.get("object")
        ex = last["ex"]
        o = last["objs"].get(oid) or {}
        row = {"scenario": name, "seed": seed, "night": night, "fut": fut, "t": round(T - w.t0, 2), "object": oid,
               "stage": al["stage"], "max_stage": al["stage"], "reason": al.get("reason"), "ego_v": m["ego"]["speed_mps"]}
        idx = next((i for i, r in enumerate(ex.records) if r["id"] == oid), None)
        tr = w.truth_at(T)
        row["track"] = {k: o.get(k) for k in ("x_m", "y_m", "vx_mps", "vy_mps", "std", "std_rho", "p_stationary", "motion",
                                              "flags", "hdg", "len", "wid", "ext_std", "class", "state", "exist", "age_s",
                                              "group", "lane", "seen")}
        if o.get("x_m") is not None:
            u, sc = truth_match(T, o)
            if u is not None:
                rng_t, rng_o = math.hypot(u["ref_x"], u["ref_y"]), math.hypot(o["x_m"], o["y_m"])
                brg_t, brg_o = math.atan2(u["ref_y"], u["ref_x"]), math.atan2(o["y_m"], o["x_m"])
                s_o, d_o = place(tr, o["x_m"], o["y_m"])
                s_t, d_t = place(tr, u["ref_x"], u["ref_y"])
                s_c, d_c = place(tr, u["x"], u["y"])
                row["truth"] = {"id": u["id"], "role": u["role"], "kind": u["kind"], "moving": u["moving"],
                                "speed": round(u["speed"], 2), "lane_offset": round(u["lane_offset"], 2),
                                "W": round(u["W"], 2), "L": round(u["L"], 2), "x": round(u["x"], 2), "y": round(u["y"], 2),
                                "ref": [round(u["ref_x"], 2), round(u["ref_y"], 2)], "heading": round(u["heading"], 3),
                                "label": u["stationary_label"], "match_score": round(sc, 2),
                                "clear_m": round(abs(d_c) - 0.5 * u["W"] - L.EGO_HALF_W, 2),
                                "d_centre": round(d_c, 2), "d_ref": round(d_t, 2)}
                row["err"] = {"range": round(rng_o - rng_t, 2), "range_rel": round((rng_o - rng_t) / max(rng_t, 1.0), 3),
                              "bearing_m": round((brg_o - brg_t) * rng_t, 2), "d_track": round(d_o, 2),
                              "d_err": round(d_o - d_t, 2), "s_err": round(s_o - s_t, 2),
                              "v_err": round(math.hypot((o.get("vx_mps") or 0.0) - u["vx"], (o.get("vy_mps") or 0.0) - u["vy"]), 2),
                              "hdg_err": None if o.get("hdg") is None else round(
                                  (o["hdg"] - u["heading"] + math.pi / 2) % math.pi - math.pi / 2, 3)}
                row["curv_here"] = round(float(np.interp(tr["s"], w.S, curv)), 5)
                row["curv_obj"] = round(float(np.interp(s_t, w.S, curv)), 5)
        if idx is not None and ex.obj is not None:
            rec = ex.records[idx]
            row["rec"] = {k: rec.get(k) for k in ("level", "level_exit", "p_coll", "p_coll_cond", "mass", "tn", "btn", "stn",
                                                  "tm_s", "ttb_s", "ttc_s", "why", "lane", "fut", "in_path", "sep_m", "size",
                                                  "a_req", "p_place", "hold")}
            n = len(ex.t)
            k0, nk = int(ex.ohdr[idx, 0]), int(ex.ohdr[idx, 1])
            row["ego_modes"] = [{"name": en, "p": round(float(ex.ego[j, 0]), 3)} for j, en in enumerate(ex.ego_names)]
            row["modes"] = []
            f = lambda v: None if not math.isfinite(v) else round(float(v), 3)  # noqa: E731
            for k in range(k0, k0 + nk):
                md = ex.modes[k]
                X, Y, SX, SY, RXY = (md[2 + q * n:2 + (q + 1) * n] for q in range(5))
                mv = float(np.nanmax(np.hypot(X[1:] - X[1], Y[1:] - Y[1])))
                mm = {"name": ex.names[k], "p0n": round(float(ex.p[k, 0]), 3), "p0w": round(float(md[0]), 3), "move_m": round(mv, 1),
                      "xy0": [round(float(X[1]), 1), round(float(Y[1]), 1)], "sx0": round(float(SX[1]), 2),
                      "sy0": round(float(SY[1]), 2), "rxy0": round(float(RXY[1]), 2),
                      "d0": round(place(tr, float(X[1]), float(Y[1]))[1], 2), "pairs": []}
                for j in range(len(ex.ego_names)):
                    p = ex.pair[k, j]
                    mm["pairs"].append({"ego": ex.ego_names[j], "w0": f(p[TH.DP_W0]), "pc": f(p[TH.DP_PC]),
                                        "course": f(p[TH.DP_COURSE]), "le": f(p[TH.DP_LE]), "tn": f(p[TH.DP_TN]),
                                        "btn": f(p[TH.DP_BTN]), "tm": f(p[TH.DP_TM]), "ttb": f(p[TH.DP_TTB]),
                                        "ttc": f(p[TH.DP_TTC]), "sep": f(p[TH.DP_SEP]), "tcon": f(p[TH.DP_TCON])})
                row["modes"].append(mm)
        if prd is not None and fut != "cv" and prd.box.last is not None:
            pl = prd.box.last
            i = pl.sel.get(oid) if isinstance(pl.sel, dict) else None
            if i is not None and i >= 0:
                row["pred"] = {"feat": [round(float(v), 3) for v in pl.feat[i]], "lane": [round(float(v), 3) for v in pl.lane[i]],
                               "src": pl.source}
        h = []
        for (Th, sm) in hist:
            v = sm.get(oid)
            if v is not None:
                h.append([round(Th - T, 2)] + [None if q is None else round(float(q), 3) for q in v])
        row["hist"] = h[-60:]
        rows.append(row)

    st.mixer.add_output("alert_truth", on_decide, streams=("decide",))
    sent = False
    for _, m in w.messages(sensors):
        if not sent:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            sent = True
        st.deliver(m)
    ft, fa = state["first_true"], state["first_any"]
    return {"scenario": name, "seed": seed, "night": night, "fut": fut, "collision_t": ct,
            "first_true": None if ft is None or not ct else round(ct - ft[0], 2),
            "first_true_stage": None if ft is None else ft[1],
            "first_said": None if fa is None or not ct else round(ct - fa, 2),
            "duration_s": None if state["T"] is None else round(state["T"] - state["t0"], 2),
            # the other alerts: when first at a stage, when first said (s before the collision where there is one),
            # and for how long said
            "other_alerts": {k: {"first": None if v["first"] is None else round(ct - v["first"], 2) if ct else v["first"],
                                 "first_said": None if v["first_said"] is None else round(ct - v["first_said"], 2) if ct else v["first_said"],
                                 "said_s": round(v["said_s"], 2)} for k, v in state["other"].items()},
            "said": rows, "trace": trace}


def seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


def cmd_run(A):
    from multiprocessing import Pool
    false = [x for x in (A.false.split(",") if A.false is not None else false_scenarios()) if x]
    true = [x for x in (A.true.split(",") if A.true is not None else true_scenarios()) if x]
    nights = [bool(int(x)) for x in A.nights.split(",")]
    opt = {k: getattr(A, k) for k in ("det", "radar", "radarbox", "predict", "decide", "trace")}
    jobs = [(n, s, ni, f, opt) for n in false for s in seeds(A.seeds) for ni in nights for f in A.fut.split(",")]
    jobs += [(n, s, ni, f, opt) for n in true for s in seeds(A.tseeds) for ni in nights for f in A.fut.split(",")]
    if A.pool > 1:
        with Pool(A.pool) as p:
            res = p.map(run_one, jobs, chunksize=1)
    else:
        res = [run_one(j) for j in jobs]
    with open(A.out, "w") as fh:
        json.dump(res, fh)
    summarise(A.out, res, A.fut.split(",")[0], False)


def load(path, fut):
    with open(path) as fh:
        return [r for r in json.load(fh) if r.get("fut", "auto") == fut]


def summarise(path, runs, fut, verbose):
    runs = [r for r in runs if r.get("fut", "auto") == fut]
    tab, cz, tot = L.table(runs)
    print("== %s (%s futures)" % (os.path.basename(path), "predicted" if fut == "auto" else "constant-velocity"))
    print("   said episodes of the streets              true arguable false")
    for sc in sorted(tab):
        print("   %-40s %5d %8d %5d" % (sc, tab[sc]["true"], tab[sc]["arguable"], tab[sc]["false"]))
    print("   %-40s %5d %8d %5d" % ("TOTAL", tot["true"], tot["arguable"], tot["false"]))
    if cz:
        print("   the false ones by cause: " + ", ".join("%s %d" % (c, sum(v.values())) for c, v in sorted(cz.items(), key=lambda kv: -sum(kv[1].values()))))
    tab2, cz2, tot2 = L.table(runs, conflicts=True)
    ld = L.leads(runs)
    if ld:
        print("   conflicts (runs warned on the obstacle / runs, median and least lead in s; false episodes on other things)")
        allv, n = [], 0
        for k in sorted(ld):
            e = ld[k]
            v = e["lead"]
            n += e["n"]
            allv += v
            print("   %-32s %s %d/%d  med %5s  min %5s" % (k[0], k[1], len(v), e["n"], ("%.2f" % v[len(v) // 2]) if v else "-",
                                                          ("%.2f" % v[0]) if v else "-"))
        print("   TOTAL %d/%d warned, mean lead %.2f s; false episodes on other things %d%s" % (
            len(allv), n, sum(allv) / max(1, len(allv)), tot2["false"],
            (" (" + ", ".join("%s %d" % (c, sum(v.values())) for c, v in sorted(cz2.items(), key=lambda kv: -sum(kv[1].values()))) + ")") if cz2 else ""))
    oth = {}
    for r in runs:
        for k, v in (r.get("other_alerts") or {}).items():
            e = oth.setdefault((k, r["scenario"]), {"n": 0, "said": 0, "said_s": 0.0, "T": 0.0, "lead": []})
            e["n"] += 1
            e["T"] += r.get("duration_s") or 0.0
            e["said_s"] += v["said_s"]
            if v["first_said"] is not None:
                e["said"] += 1
                if r.get("collision_t"):
                    e["lead"].append(v["first_said"])
    for (k, sc), e in sorted(oth.items()):
        ld = sorted(e["lead"])
        print("   alert %-12s %-28s said in %d of %d runs, %.0f %% of the time%s" % (
            k, sc, e["said"], e["n"], 100.0 * e["said_s"] / max(e["T"], 1e-9),
            (", first said %.1f s before the collision (median; least %.1f)" % (ld[len(ld) // 2], ld[-1] if False else ld[0])) if ld else ""))
    if verbose:
        for r in runs:
            for row in r["said"]:
                tr, er = row.get("truth") or {}, row.get("err") or {}
                print("   %-22s s%d %s t %6.1f obj %-4s %-8s %-22s role %-14s clear %5s range %5.1f err %6s d %5s p_place %s" % (
                    r["scenario"], r["seed"], "N" if r["night"] else "D", row["t"], row["object"], L.classify(row), L.cause(row),
                    tr.get("role"), tr.get("clear_m"), math.hypot(*(tr.get("ref") or [0, 0])), er.get("range"), er.get("d_track"),
                    (row.get("rec") or {}).get("p_place")))


def cmd_sum(A):
    for p in A.files:
        summarise(p, load(p, A.fut), A.fut, A.verbose)


def cmd_cmp(A):
    b, n = load(A.base, A.fut), load(A.new, A.fut)
    print("== %s -> %s (paired over the same runs; +- the bootstrap's sd over seeds, [2.5 %%, 97.5 %%])" % (
        os.path.basename(A.base), os.path.basename(A.new)))
    for name, f in (("false episodes (streets)", L.n_false), ("said episodes (streets)", L.n_said),
                    ("false episodes on other things (conflicts)", L.n_other), ("conflicts warned", L.warned)):
        r = L.paired(b, n, f)
        print("   %-44s %4d -> %4d   %+d +- %.1f  [%+d, %+d]  (%d runs)" % (name, r["base"], r["new"], r["diff"], r["sd"],
                                                                          r["lo"], r["hi"], r["n"]))
    ch = L.lead_changes(b, n)
    both = [(a, c) for _, a, c in ch if a is not None and c is not None]
    print("   conflict warnings: lost %d, gained %d, later by > 0.05 s %d, earlier %d" % (
        sum(1 for _, a, c in ch if c is None), sum(1 for _, a, c in ch if a is None),
        sum(1 for a, c in both if c < a), sum(1 for a, c in both if c > a)))
    for k, a, c in ch:
        print("      %-28s s%d %s  %s -> %s" % (k[0], k[1], "N" if k[2] else "D", a, c))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--false", default=None)
    r.add_argument("--true", default=None)
    r.add_argument("--seeds", default="1-6")
    r.add_argument("--tseeds", default="1-3")
    r.add_argument("--nights", default="0,1")
    r.add_argument("--det", default=None)
    r.add_argument("--radar", default=None)
    r.add_argument("--radarbox", default=None)
    r.add_argument("--predict", default=None)
    r.add_argument("--decide", default=None)
    r.add_argument("--fut", default="auto")
    r.add_argument("--pool", type=int, default=6)
    r.add_argument("--trace", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("sum")
    s.add_argument("files", nargs="+")
    s.add_argument("--fut", default="auto")
    s.add_argument("-v", dest="verbose", action="store_true")
    s.set_defaults(fn=cmd_sum)
    c = sub.add_parser("cmp")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--fut", default="auto")
    c.set_defaults(fn=cmd_cmp)
    A = ap.parse_args(argv)
    A.fn(A)


if __name__ == "__main__":
    main()
