#!/usr/bin/env python3
"""Trace the radar component against the simulator's truth: what goes in, what it concludes, and where the two part.

The detector simulator knows which actor every box came from (DetectorSim.log); this tool carries that through
the radar and reports, for one scenario run:

  MEAS    the measurements against the truth: ground range, bearing, box-height range, with the errors the
          radar is TOLD (R) beside the errors that are there
  EGO     the car's own movement as the radar integrates it against the true movement
  ASSOC   which actor's box updated which object: hops, duplicates, objects started on false boxes
  STATE   per truth kind (standing / moving): speed error, p(stand), verdict; objects that only ever took boxes
          of one actor ("pure") apart from the ones that hopped
  NIS     innovation statistics per detection kind (1 = the filter's stated uncertainty is right)

`--actor ID` / `--object ID` print a timeline. `--dump FILE` writes all rows as JSON lines.

    python3 tools/radar_trace.py --scenario parked_street --seed 1 [--night]
"""
import argparse
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

NS = 1_000_000_000


def run(name, seed, night, radar_cfg=None, det_params=None):
    from chris_integrator.mixer import MessageClock
    from chris_integrator.radar import models as M
    from chris_integrator.radar import tracker as TK
    from chris_integrator.sim import default_sensors
    from chris_integrator.sim import roaduser_sim as rs
    from chris_integrator.stack import build

    w = rs.scenario(name, seed=seed, night=night)
    det = rs.DetectorSim(seed=1000 + seed, night=night, params=det_params)
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
    sensors.append(det)
    cfg = {"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}}
    if radar_cfg:
        cfg["radar"] = radar_cfg
    st = build(cfg=cfg, mapper=None, clock=MessageClock())
    box = st.mixer.slots["radar"].plugin.box
    tr = box.tracker
    R = {"meas": [], "upd": [], "start": [], "frames": [], "ego": [], "scans": []}
    owner = {}                   # object id -> actor of its last box
    actors_of = collections.defaultdict(list)     # object id -> [actor per update]
    ctx = {}

    def truth(t):
        return {a["id"]: a for a in w.actors_at(t)}

    index = {}                   # scan time (ns) -> {box: actor}; the simulator's log runs ahead of the delivery

    def box_actor(t):
        while ctx.get("n", 0) < len(det.log):
            tl, view, logd = det.log[ctx.get("n", 0)]
            m = index.setdefault(int(round(tl * NS)), {})
            for (aid, cls, bx, clip) in logd:
                m[tuple(round(v, 1) for v in bx)] = aid
            ctx["n"] = ctx.get("n", 0) + 1
        return index.get(int(round(t * NS)), {})

    orig_process, orig_after = tr.process, tr._after_update

    def process(src, t, sensor, dets):
        ctx["dets"] = dets
        lk = box_actor(t)
        T = truth(t)
        pose = tr._pose(t)
        for d in dets:
            bx = (d.get("attrs") or {}).get("box")
            d["actor"] = lk.get(tuple(round(float(v), 1) for v in bx)) if bx else None
            a = T.get(d["actor"])
            if a is None:
                continue
            row = {"t": t, "actor": d["actor"], "kind": d["kind"], "moving": a["moving"], "role": a["role"],
                   "range": a["range"], "x": a["x"], "y": a["y"], "clip": (d["attrs"] or {}).get("clip", "")}
            bt_ref, bt_c = math.atan2(a["ref_y"], a["ref_x"]), math.atan2(a["y"], a["x"])
            if d["kind"] in ("b", "bt"):
                row.update(b=float(d["z"][0]), sb=math.sqrt(d["R"][0]))
            else:
                rm = sensor.hc / math.tan(max(float(d["z"][0]), 1e-4))
                row.update(r=rm, sr=math.sqrt(d["R"][0]) * (rm * rm + sensor.hc ** 2) / sensor.hc, b=float(d["z"][1]),
                           sb=math.sqrt(d["R"][1]), ang=float(d["z"][0]), sang=math.sqrt(d["R"][0]))
                if d["kind"] == "rbh":
                    row.update(rh=math.exp(sensor.logf + math.log(a["H"]) - float(d["z"][2])), sh=math.sqrt(d["R"][2]),
                               depth=a["ref_x"])
            row.update(b_ref=bt_ref, b_ctr=bt_c)
            R["meas"].append(row)
        before = {o.id for o in tr.objects}
        ctx.update(t=t, truth=T, pose=pose, sensor=sensor, present={d.get("actor") for d in dets},
                   has_obj={owner.get(o.id) for o in tr.objects})
        n_obj = len(tr.objects)
        orig_process(src, t, sensor, dets)
        new = [o for o in tr.objects if o.id not in before]
        for o in new:
            d = dets[int(o.m[TK.M_DET])]                 # the detection that started it
            owner[o.id] = d.get("actor")
            actors_of[o.id].append(d.get("actor"))
            # is there already an object on this actor?
            dup = [q.id for q in tr.objects if q.id != o.id and owner.get(q.id) == d.get("actor") and d.get("actor") is not None
                   and not str(d.get("actor")).startswith("fp")]
            R["start"].append({"t": t, "obj": o.id, "actor": d.get("actor"), "kind": d["kind"], "dup_of": dup})
        R["scans"].append({"t": t, "dets": len(dets), "objs": n_obj, "view": getattr(sensor, "cover", None) and
                           round(sensor.cover["ground_min_m"], 1)})

    def upd_row(o, d, Rm, sensor, pose, src, t):
        """The row of an update: the object as its story had it before the detection (the node before the primary,
        stepped to the scan's time: every motion hypothesis by itself, and their combination)."""
        T = ctx["truth"]
        prev = owner.get(o.id)
        act = d.get("actor")
        chain = tr.hyp.chain(o.h)
        S0, m0 = tr.store.a, TK.OFF["m"][0]
        par = M._native_par(tr.mcfg)
        xs, Ps, mu0 = np.empty((3, M.N)), np.empty((3, M.N, M.N)), np.empty(3)
        x0, P0, _mu = np.empty((1, M.N)), np.empty((1, M.N, M.N)), np.empty((1, 3))
        dt0 = t - float(S0[chain[1], m0 + TK.M_T])
        M.NR.radar_peek_models(S0, chain[1], dt0, o.kin, par, xs, Ps, mu0)
        M.NR.radar_peek(S0, [chain[1]], [dt0], [o.kin], par, x0, P0, _mu)
        hits0 = int(S0[chain[1], m0 + TK.M_HITS])
        # the object's own prediction of this detection, before it is used
        zp, H = sensor.h(x0[0], pose, d)
        nu = sensor.residual(np.asarray(d["z"], float), zp, d)
        S = H @ P0[0] @ H.T + Rm
        d2 = float(nu @ np.linalg.solve(S, nu))
        a = T.get(prev) if prev is not None else None
        row = {"t": t, "obj": o.id, "actor": act, "prev": prev, "hop": prev is not None and act != prev, "kind": d["kind"],
               "hits": hits0, "gain": float(o.m[TK.M_DL]), "leaves": o.nleaf, "p_story": o.pp, "nis": d2 / len(nu),
               "dim": len(nu), "mu0": [float(m) for m in mu0],
               "nu": [float(v) for v in nu], "sd": [float(math.sqrt(max(S[i, i], 0.0))) for i in range(len(nu))]}
        like = []
        for j in range(3):
            zj, Hj = sensor.h(xs[j], pose, d)
            nj = sensor.residual(np.asarray(d["z"], float), zj, d)
            Sj = Hj @ Ps[j] @ Hj.T + Rm
            like.append(float(nj @ np.linalg.solve(Sj, nj)) / len(nj))
            row.setdefault("sd_m", []).append([float(math.sqrt(max(Sj[i, i], 0.0))) for i in range(len(nj))])
            row.setdefault("ldet_m", []).append(float(0.5 * math.log(max(np.linalg.det(Sj), 1e-300))))
        row["nis_m"] = like
        row["R"] = [float(math.sqrt(Rm[i, i])) for i in range(len(nu))]
        an, ap = T.get(act), T.get(prev)
        bx = (d.get("attrs") or {}).get("box")
        lb = ctx.setdefault("lastbox", {}).get(o.id)
        if bx is not None and lb is not None:
            psi0, psi1 = tr.ego.pose_at(lb[0])[2], tr.ego.pose_at(t)[2]
            f = math.exp(sensor.logf)
            du = 0.5 * (bx[0] + bx[2]) - 0.5 * (lb[1][0] + lb[1][2]) - f * (psi1 - psi0)
            wm = 0.5 * ((bx[2] - bx[0]) + (lb[1][2] - lb[1][0]))
            row.update(f_du=du / max(wm, 1.0), f_dh=math.log(max(bx[3] - bx[1], 1.0) / max(lb[1][3] - lb[1][1], 1.0)), f_dt=t - lb[0],
                       f_dv=(bx[3] - lb[1][3]) / max(0.5 * ((bx[3] - bx[1]) + (lb[1][3] - lb[1][1])), 1.0))
        if bx is not None:
            ctx["lastbox"][o.id] = (t, [float(v) for v in bx])
        if an is not None:
            row.update(range=an["range"], moving=an["moving"])
        if row["hop"]:
            row.update(prev_present=prev in ctx["present"], new_had_obj=act in ctx["has_obj"], state=o.state,
                       sep=math.hypot(an["x"] - ap["x"], an["y"] - ap["y"]) if an is not None and ap is not None else None,
                       sep_b=abs(math.degrees(math.atan2(an["y"], an["x"]) - math.atan2(ap["y"], ap["x"]))) if an is not None and ap is not None else None,
                       seen_ago=t - float(S0[chain[1], m0 + TK.M_THIT]),
                       sd_pos=float(math.sqrt(max(np.linalg.eigvalsh(P0[0][:2, :2])[-1], 0.0))))
        return row

    def after(o, d, sensor, src, t):
        if len(tr.hyp.chain(o.h)) < 2:
            return orig_after(o, d, sensor, src, t)
        row = upd_row(o, d, np.diag(np.asarray(d["R"], float)), sensor, ctx["pose"], src, t)
        orig_after(o, d, sensor, src, t)
        row["mu1"] = [float(m) for m in o.mu]
        owner[o.id] = d.get("actor")
        actors_of[o.id].append(d.get("actor"))
        R["upd"].append(row)

    tr.process, tr._after_update = process, after

    orig_frame = box.frame

    def frame(ts):
        out = orig_frame(ts)
        if out is None:
            return out
        T = ts / NS
        tt = w.truth_at(T)
        p = box.ego.pose_at(T)
        R["ego"].append((T, p[0], p[1], p[2], tt["pe"] + rs.CAM_TO_AXLE * math.cos(tt["psi"]), tt["pn"] + rs.CAM_TO_AXLE * math.sin(tt["psi"]), tt["psi"], box.ego.state_at(T)[3], tt["v"],
                         box.ego.state_at(T)[4], tt["w"]))
        bb = tr.bias.get("cam") or next(iter(tr.bias.values()), None)
        if bb is not None:
            R.setdefault("bias", []).append((T, math.degrees(bb[0]), math.degrees(math.sqrt(bb[1])),
                                             sum(1 for o in tr.objects if o.state != "tentative")))
        if len(R["ego"]) % 2:
            return out
        Tr = truth(T)
        X, Y, psi = p
        ch, sh = math.cos(psi), math.sin(psi)
        for o in tr.objects:
            x, P, mu = tr.peek(o, T)
            dx, dy = x[M.PX] - X, x[M.PY] - Y
            xc, yc = ch * dx + sh * dy, -sh * dx + ch * dy
            vx, vy = ch * x[M.VX] + sh * x[M.VY], -sh * x[M.VX] + ch * x[M.VY]
            ow = owner.get(o.id)
            a = Tr.get(ow)
            acts = [q for q in actors_of[o.id]]
            pure = len(set(acts)) == 1
            row = {"t": T, "obj": o.id, "owner": ow, "state": o.state, "hits": o.hits, "age": T - o.t0, "x": xc, "y": yc,
                   "vx": vx, "vy": vy, "mu": [float(m) for m in mu], "verdict": o.verdict, "pure": pure,
                   "sc": math.degrees(float(x[M.DP])), "lh": float(x[M.LH]), "label": o.label,
                   "sd": [float(math.sqrt(max(P[i, i], 0.0))) for i in range(4)],
                   "seen": T - max(o.last_seen.values())}
            vf = np.array([ch * o.xf[M.VX] + sh * o.xf[M.VY], -sh * o.xf[M.VX] + ch * o.xf[M.VY]])
            Rg = np.array([[ch, sh], [-sh, ch]])
            Pv = Rg @ o.Pf[2:4, 2:4] @ Rg.T
            row["vf"], row["sdf"] = [float(vf[0]), float(vf[1])], [float(math.sqrt(max(Pv[0, 0], 0.0))), float(math.sqrt(max(Pv[1, 1], 0.0)))]
            row["m2"] = float(vf @ np.linalg.solve(Pv + np.eye(2) * 1e-9, vf))
            row["res"] = 0.0
            rr = max(math.hypot(dx, dy), 0.5)
            ul = np.array([dx / rr, dy / rr])
            ua = np.array([-dy / rr, dx / rr])
            Pc = Rg @ P[:2, :2] @ Rg.T
            row["Pc"] = [float(Pc[0, 0]), float(Pc[0, 1]), float(Pc[1, 1])]     # the centre's covariance, car axes
            row["sd_al"], row["sd_ac"] = float(math.sqrt(max(ul @ P[:2, :2] @ ul, 0.0))), float(math.sqrt(max(ua @ P[:2, :2] @ ua, 0.0)))
            for q in out["objects"]:
                if q["id"] == o.id:
                    row["flags"], row["off"] = q["flags"], q["path_offset_m"]
                    row["len"], row["wid"] = q.get("len"), q.get("wid")
            if a is not None:
                r = max(a["range"], 0.5)
                ux, uy = a["ref_x"] / r, a["ref_y"] / r
                ex, ey = xc - a["ref_x"], yc - a["ref_y"]
                # the state is the object's centre (native version 8): its errors against the truth's centre,
                # along and across the line of sight to it
                rc = max(math.hypot(a["x"], a["y"]), 0.5)
                cx, cy = a["x"] / rc, a["y"] / rc
                fx, fy = xc - a["x"], yc - a["y"]
                row.update(moving=a["moving"], role=a["role"], tx=a["ref_x"], ty=a["ref_y"], tvx=a["vx"], tvy=a["vy"],
                           along=ex * ux + ey * uy, across=-ex * uy + ey * ux, range=r, tlab=a["stationary_label"],
                           tH=a["H"], tL=a.get("len"), lane_off=a["lane_offset"], tcx=a["x"], tcy=a["y"],
                           along_c=fx * cx + fy * cy, across_c=-fx * cy + fy * cx)
            R["frames"].append(row)
        return out
    box.frame = frame

    calib_sent = False
    for t_arr, m in w.messages(sensors):
        if not calib_sent:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            calib_sent = True
        st.deliver(m)
    R["pitch_err"] = math.degrees(det.pitch_err)
    return w, det, box, R


# ---- reports ------------------------------------------------------------------------------------------------------
def _ms(v):
    v = np.asarray(v, float)
    return (float(np.mean(v)), float(np.std(v))) if len(v) else (float("nan"), float("nan"))


def report_meas(R):
    print("\nMEAS  measurement against truth (truth point: the nearest point of the actor's footprint)")
    print("  %-9s %5s | %-22s %-8s | %-22s %-6s | %-18s %-6s" % ("range", "n", "ground r/true-1 mean sd", "told sd",
                                                                  "bearing err deg (ref)", "told", "height r/depth-1", "told"))
    bins = [(0, 10), (10, 20), (20, 35), (35, 55), (55, 200)]
    for mv in (False, True):
        rows_mv = [m for m in R["meas"] if m["moving"] == mv]
        print("  -- %s actors" % ("moving" if mv else "standing"))
        for lo, hi in bins:
            rows = [m for m in rows_mv if lo <= m["range"] < hi]
            if not rows:
                continue
            rr = [m["r"] / m["range"] - 1.0 for m in rows if "r" in m]
            ts = [m["sr"] / m["r"] for m in rows if "r" in m]
            be = [math.degrees(m["b"] - m["b_ref"]) for m in rows]
            bc = [math.degrees(m["b"] - m["b_ctr"]) for m in rows]
            tb = [math.degrees(m["sb"]) for m in rows]
            hh = [m["rh"] / max(m["depth"], 0.5) - 1.0 for m in rows if "rh" in m]
            th = [m["sh"] for m in rows if "rh" in m]
            kinds = collections.Counter(m["kind"] for m in rows)
            print("  %3d-%-3d m %5d | %+6.3f %6.3f          %6.3f   | ref %+5.2f %5.2f ctr %+5.2f %5.2f %5.2f | %+6.3f %6.3f     %5.3f  %s"
                  % (lo, hi, len(rows), *_ms(rr), np.mean(ts) if ts else float("nan"), *_ms(be), *_ms(bc), np.mean(tb),
                     *_ms(hh), np.mean(th) if th else float("nan"), dict(kinds)))
    # per-actor persistent part of the ground-range error against the scatter around it
    per = collections.defaultdict(list)
    for m in R["meas"]:
        if "r" in m and 10 <= m["range"] < 55:
            per[m["actor"]].append(m["r"] / m["range"] - 1.0)
    means = [np.mean(v) for v in per.values() if len(v) >= 5]
    within = [np.std(v) for v in per.values() if len(v) >= 5]
    if means:
        print("  ground-range scale error per actor (10-55 m, >=5 boxes, %d actors): mean of means %+.3f, sd between actors %.3f, "
              "sd within an actor %.3f   (the filter's prior for `s`: 0 +- 0.18)" % (len(means), np.mean(means), np.std(means), np.mean(within)))
    perh = collections.defaultdict(list)
    for m in R["meas"]:
        if "rh" in m and 10 <= m["range"] < 55:
            perh[m["actor"]].append(math.log(m["rh"] / max(m["depth"], 0.5)))
    means = [np.mean(v) for v in perh.values() if len(v) >= 5]
    within = [np.std(v) for v in perh.values() if len(v) >= 5]
    if means:
        print("  box-height range (TRUE object height used) per actor: mean of log ratio %+.3f, sd between %.3f, sd within %.3f"
              % (np.mean(means), np.std(means), np.mean(within)))
    # the same in the angle domain: the bottom edge's angle below the horizon, measured minus true
    HC = 1.22
    pa = collections.defaultdict(list)
    for m in R["meas"]:
        if "r" in m:
            pa[m["actor"]].append((m["range"], math.degrees(math.atan2(HC, m["r"]) - math.atan2(HC, m["range"])), m["t"]))
    allv = [v[1] for vs in pa.values() for v in vs]
    told = [math.degrees(m["sang"]) for m in R["meas"] if "sang" in m]
    means = [np.mean([v[1] for v in vs]) for vs in pa.values() if len(vs) >= 5]
    within = [np.std([v[1] for v in vs]) for vs in pa.values() if len(vs) >= 5]
    if told:
        print("  bottom-edge angle noise the radar is told per box: mean %.3f deg (to compare with the sd within an actor below)" % np.mean(told))
    sl = [np.polyfit([v[0] for v in vs], [v[1] for v in vs], 1)[0] * 10.0 for vs in pa.values() if len(vs) >= 8 and np.ptp([v[0] for v in vs]) > 8.0]
    if means:
        print("  bottom-edge ANGLE error (deg): all boxes mean %+.3f sd %.3f; per actor: mean of means %+.3f, sd between actors %.3f, sd within %.3f; "
              "change per 10 m of approach %+.3f +- %.3f" % (np.mean(allv), np.std(allv), np.mean(means), np.std(means), np.mean(within),
                                                              np.mean(sl) if sl else float("nan"), np.std(sl) if sl else float("nan")))
        for lo, hi in ((10, 20), (20, 35), (35, 55), (55, 200)):
            v = [x[1] for vs in pa.values() for x in vs if lo <= x[0] < hi]
            if v:
                print("    %3d-%-3d m: angle error mean %+.3f sd %.3f deg (n %d)" % (lo, hi, np.mean(v), np.std(v), len(v)))
    # how a range ratio changes along one actor's approach
    for mv in (False, True):
        sl = []
        for aid, _ in per.items():
            rows = [m for m in R["meas"] if m["actor"] == aid and "r" in m and m["moving"] == mv]
            if len(rows) >= 8:
                x = np.array([m["range"] for m in rows])
                y = np.array([m["r"] / m["range"] - 1.0 for m in rows])
                if np.ptp(x) > 8.0:
                    sl.append(np.polyfit(x, y, 1)[0] * 10.0)
        if sl:
            print("  %s: change of the ground-range scale error per 10 m of approach: mean %+.3f sd %.3f (n %d)   (the filter: a slow walk, 0.01/sqrt(s))"
                  % ("moving" if mv else "standing", np.mean(sl), np.std(sl), len(sl)))


def report_ego(R):
    E = np.array(R["ego"])
    if len(E) < 50:
        return
    print("\nEGO   the car's movement as integrated by the radar against the truth")
    t = E[:, 0]
    for win in (1.0, 5.0):
        k = int(round(win / np.median(np.diff(t))))
        if k >= len(E):
            continue
        out = []
        for i in range(0, len(E) - k, max(1, k // 4)):
            def rel(X0, Y0, p0, X1, Y1, p1):
                c, s = math.cos(p0), math.sin(p0)
                return c * (X1 - X0) + s * (Y1 - Y0), -s * (X1 - X0) + c * (Y1 - Y0), p1 - p0
            a = rel(E[i, 1], E[i, 2], E[i, 3], E[i + k, 1], E[i + k, 2], E[i + k, 3])
            b = rel(E[i, 4], E[i, 5], E[i, 6], E[i + k, 4], E[i + k, 5], E[i + k, 6])
            out.append((a[0] - b[0], a[1] - b[1], math.degrees(a[2] - b[2])))
        o = np.array(out)
        print("  over %.0f s: along %+.3f +- %.3f m, across %+.3f +- %.3f m, heading %+.3f +- %.3f deg"
              % (win, o[:, 0].mean(), o[:, 0].std(), o[:, 1].mean(), o[:, 1].std(), o[:, 2].mean(), o[:, 2].std()))
    B = R.get("bias") or []
    if B:
        step = max(1, len(B) // 12)
        print("  the camera's pitch bias as learnt (deg +- its stated sd): " + "  ".join("%.0fs %+.2f+-%.2f" % (b[0] - B[0][0], b[1], b[2]) for b in B[::step]))
        print("    (the true one is the mean bottom-edge ANGLE error in MEAS; simulator pitch error %+.2f deg)" % R.get("pitch_err", float("nan")))
    print("  speed error %+.3f +- %.3f m/s, yaw rate error %+.4f +- %.4f rad/s"
          % ((E[:, 7] - E[:, 8]).mean(), (E[:, 7] - E[:, 8]).std(), (E[:, 9] - E[:, 10]).mean(), (E[:, 9] - E[:, 10]).std()))


def report_assoc(R):
    U, S = R["upd"], R["start"]
    print("\nASSOC which actor's box went to which object")
    n = len(U)
    hop = [u for u in U if u["hop"]]
    fp_u = [u for u in U if str(u["actor"]).startswith("fp")]
    print("  updates %d, of them onto an object last fed by ANOTHER actor: %d (%.1f %%); false boxes used as updates: %d"
          % (n, len(hop), 100.0 * len(hop) / max(n, 1), len(fp_u)))
    dets = sum(s["dets"] for s in R["scans"])
    print("  boxes %d in %d scans (%.2f per scan); objects alive at a scan: mean %.1f, max %d"
          % (dets, len(R["scans"]), dets / max(len(R["scans"]), 1), np.mean([s["objs"] for s in R["scans"]]), max(s["objs"] for s in R["scans"])))
    dup = [s for s in S if s["dup_of"]]
    fps = [s for s in S if str(s["actor"]).startswith("fp") or s["actor"] is None]
    print("  objects started %d: on an actor that already HAS an object %d (%.0f %%), on a false box %d"
          % (len(S), len(dup), 100.0 * len(dup) / max(len(S), 1), len(fps)))
    # hops: by kind and by how the update had fitted
    if hop:
        k = collections.Counter(u["kind"] for u in hop)
        print("  hops by detection kind: %s; their NIS median %.2f (all updates: %.2f)"
              % (dict(k), np.median([u["nis"] for u in hop]), np.median([u["nis"] for u in U])))
    if hop:
        hh = [u for u in hop if u.get("sep") is not None]
        for lo, hi in ((0, 30), (30, 200)):
            gh = [u["gain"] for u in hop if u.get("gain") is not None and lo <= u.get("range", -1) < hi]
            gs = [u["gain"] for u in U if not u["hop"] and u.get("gain") is not None and lo <= u.get("range", -1) < hi]
            if gh and gs:
                print("    %d-%d m: gain (log odds for the pairing) of hops: quartiles %.1f %.1f %.1f; of right pairings: %.1f %.1f %.1f"
                      % (lo, hi, *np.percentile(gh, [25, 50, 75]), *np.percentile(gs, [25, 50, 75])))
        for lo, hi in ((0, 15), (15, 30), (30, 50), (50, 200)):
            fh = [u for u in hop if "f_du" in u and lo <= u.get("range", -1) < hi and u["f_dt"] <= 1.0]
            fs = [u for u in U if not u["hop"] and "f_du" in u and lo <= u.get("range", -1) < hi and u["f_dt"] <= 1.0]
            if len(fh) >= 3 and len(fs) >= 3:
                q = lambda v: "%.2f %.2f %.2f" % tuple(np.percentile(v, [50, 80, 95]))
                print("    %d-%d m: the box against the object's last box (<= 1 s ago), shift of the centre / box width (50/80/95 %%): right %s | hops %s (5/20/50 %%: %s);"
                      " |log height ratio|: right %s | hops %s" % (lo, hi, q([abs(u["f_du"]) for u in fs]), q([abs(u["f_du"]) for u in fh]),
                      "%.2f %.2f %.2f" % tuple(np.percentile([abs(u["f_du"]) for u in fh], [5, 20, 50])),
                      q([abs(u["f_dh"]) for u in fs]), q([abs(u["f_dh"]) for u in fh])))
        print("  hops: the old actor's box was in the same scan %.0f %%; the new actor already had an object %.0f %%; object state %s"
              % (100.0 * np.mean([u["prev_present"] for u in hop]), 100.0 * np.mean([u["new_had_obj"] for u in hop]),
                 dict(collections.Counter(u["state"] for u in hop))))
        for lo, hi in ((0, 15), (15, 30), (30, 50), (50, 200)):
            v = [u for u in hh if lo <= u.get("range", -1) < hi]
            allu = [u for u in U if lo <= u.get("range", -1) < hi]
            if v:
                print("    %3d-%-3d m: %3d hops of %4d updates (%.0f %%); actors apart med %.1f m / %.2f deg; object's place known to %.1f m; unseen for %.2f s; "
                      "old box present %.0f %%, new had object %.0f %%"
                      % (lo, hi, len(v), len(allu), 100.0 * len(v) / max(len(allu), 1), np.median([u["sep"] for u in v]), np.median([u["sep_b"] for u in v]),
                         np.median([u["sd_pos"] for u in v]), np.median([u["seen_ago"] for u in v]),
                         100.0 * np.mean([u["prev_present"] for u in v]), 100.0 * np.mean([u["new_had_obj"] for u in v])))
    per = collections.defaultdict(set)
    for u in U:
        per[u["obj"]].add(u["actor"])
    c = collections.Counter(min(len(v), 4) for v in per.values())
    print("  objects by number of distinct actors that fed them: %s" % ", ".join("%s%d: %d" % (">=" if k == 4 else "", k, c[k]) for k in sorted(c)))


def report_state(R):
    F = [f for f in R["frames"] if "moving" in f and f["state"] in ("confirmed", "coast")]
    print("\nSTATE the objects against the truth of the actor that last fed them (confirmed and coasting)")
    print("  %-22s %7s | %-17s %-17s | %-8s | %-22s | %s" % ("", "obj-s", "speed err m/s med p90", "along m  med p90", "p(stand)",
                                                              "verdict stat/mov/unk %", "across m p90"))
    for mv in (False, True):
        for pure in (True, False):
            for lo, hi in ((0, 6), (6, 12), (12, 25), (25, 10 ** 6)):
                rows = [f for f in F if f["moving"] == mv and f["pure"] == pure and lo <= f["hits"] < hi]
                if len(rows) < 10:
                    continue
                ve = [math.hypot(f["vx"] - f["tvx"], f["vy"] - f["tvy"]) for f in rows]
                al = [abs(f["along"]) for f in rows]
                ac = [abs(f["across"]) for f in rows]
                ps = [f["mu"][0] for f in rows]
                vd = collections.Counter(f["verdict"] for f in rows)
                nn = len(rows)
                print("  %-8s %-5s hits %2d-%-3s %6.0f | %6.2f %6.2f     %6.2f %6.2f     | %5.2f    | %4.0f %4.0f %4.0f         | %.2f"
                      % ("moving" if mv else "standing", "pure" if pure else "mixed", lo, hi if hi < 1000 else "", nn * 0.1,
                         np.median(ve), np.percentile(ve, 90), np.median(al), np.percentile(al, 90), np.mean(ps),
                         100.0 * vd["stationary"] / nn, 100.0 * vd["moving"] / nn, 100.0 * vd["unknown"] / nn, np.percentile(ac, 90)))
    # range dependence for pure standing ones
    print("  standing, pure, by true range:")
    for lo, hi in ((0, 10), (10, 20), (20, 35), (35, 80)):
        rows = [f for f in F if not f["moving"] and f["pure"] and lo <= f["range"] < hi and f["hits"] >= 6]
        if len(rows) < 10:
            continue
        ve = [math.hypot(f["vx"], f["vy"]) for f in rows]
        vd = collections.Counter(f["verdict"] for f in rows)
        nn = len(rows)
        print("    %2d-%2d m: n %5d, vx mean %+.2f, speed med %.2f p90 %.2f, p(stand) %.2f, stat/mov/unk %3.0f/%3.0f/%3.0f %%, scale state mean %+.3f, along med %+.2f"
              % (lo, hi, nn, np.mean([f["vx"] for f in rows]), np.median(ve), np.percentile(ve, 90), np.mean([f["mu"][0] for f in rows]),
                 100.0 * vd["stationary"] / nn, 100.0 * vd["moving"] / nn, 100.0 * vd["unknown"] / nn,
                 np.mean([f["sc"] for f in rows]), np.median([f["along"] for f in rows])))
    print("  is the stated uncertainty of the centre right? hits >= 4, against the truth's centre: |error| / stated sd (rms; 1 = right),"
          " share within 1 / 2 sd across")
    for pure in (True, False):
        for lo, hi in ((0, 15), (15, 30), (30, 50), (50, 100)):
            rows = [f for f in F if f["pure"] == pure and lo <= f["range"] < hi and f["hits"] >= 4 and f["state"] == "confirmed"
                    and "across_c" in f]
            if len(rows) < 10:
                continue
            print("    %s %2d-%-3d m: n %4d, along: stated sd med %.2f m, ratio %.2f; across: |err| med %.2f m, stated sd med %.2f m (%.2f deg), "
                  "ratio %.2f, within 1/2 sd %.2f/%.2f"
                  % ("pure " if pure else "mixed", lo, hi, len(rows), np.median([f["sd_al"] for f in rows]),
                     math.sqrt(np.mean([(f["along_c"] / max(f["sd_al"], 1e-6)) ** 2 for f in rows])),
                     np.median([abs(f["across_c"]) for f in rows]),
                     np.median([f["sd_ac"] for f in rows]), np.median([math.degrees(f["sd_ac"] / f["range"]) for f in rows]),
                     math.sqrt(np.mean([(f["across_c"] / max(f["sd_ac"], 1e-6)) ** 2 for f in rows])),
                     np.mean([abs(f["across_c"]) <= f["sd_ac"] for f in rows]),
                     np.mean([abs(f["across_c"]) <= 2.0 * f["sd_ac"] for f in rows])))
    # false "stands on the path"
    bad = [f for f in F if not f["moving"] and f.get("tlab") == "parked" and "on_path" in f.get("flags", ()) and f["verdict"] == "stationary"]
    if bad:
        print("  parked actors reported standing ON the path: %.1f object-s; their true lane offset med %.2f m, the radar's path offset med %.2f m, "
              "across error med %+.2f m, along error med %+.2f m, range med %.0f m"
              % (len(bad) * 0.1, np.median([abs(f["lane_off"]) for f in bad]), np.median([abs(f["off"]) for f in bad]),
                 np.median([f["across"] for f in bad]), np.median([f["along"] for f in bad]), np.median([f["range"] for f in bad])))


def report_nis(R):
    print("\nNIS   how the objects had predicted their boxes (per dimension; 1 = as uncertain as stated)")
    for kind in ("rbh", "rb", "b", "bt"):
        for hop in (False, True):
            rows = [u for u in R["upd"] if u["kind"] == kind and u["hop"] == hop]
            if len(rows) < 5:
                continue
            nis = np.array([u["nis"] for u in rows])
            nu = np.array([[abs(a) / max(b, 1e-9) for a, b in zip(u["nu"], u["sd"])] for u in rows])
            print("  %-3s %-9s n %5d: NIS mean %.2f median %.2f; |residual|/stated sd per row (rms): %s"
                  % (kind, "hop" if hop else "same", len(rows), nis.mean(), np.median(nis),
                     " ".join("%.2f" % v for v in np.sqrt((nu ** 2).mean(axis=0)))))
    # per hypothesis, for objects that are pure
    for mv, nm in ((False, "standing"), (True, "moving")):
        pass


def timeline(R, actor=None, obj=None):
    if actor is not None:
        objs = sorted({u["obj"] for u in R["upd"] if u["actor"] == actor} | {s["obj"] for s in R["start"] if s["actor"] == actor})
        print("\nactor %s: objects %s" % (actor, objs))
        for s in R["start"]:
            if s["actor"] == actor:
                print("  %.2f start obj %d (%s) dup_of %s" % (s["t"], s["obj"], s["kind"], s["dup_of"]))
        for u in R["upd"]:
            if u["actor"] == actor or (u["obj"] in objs):
                print("  %.2f obj %3d <- actor %-6s %s%-3s hits %2d nis %5.2f per-model %s mu %s -> %s"
                      % (u["t"], u["obj"], u["actor"], "HOP " if u["hop"] else "    ", u["kind"], u["hits"], u["nis"],
                         "/".join("%.1f" % v for v in u["nis_m"]), "/".join("%.2f" % v for v in u["mu0"]),
                         "/".join("%.2f" % v for v in u["mu1"])))
    if obj is not None:
        print("\nobject %d" % obj)
        ev = [(u["t"], "U", u) for u in R["upd"] if u["obj"] == obj] + [(f["t"], "F", f) for f in R["frames"] if f["obj"] == obj]
        ev.sort(key=lambda e: e[0])
        lastf = -1.0
        for t, k, e in ev:
            if k == "U":
                print("  %.2f  box of %-6s %s%-3s nis %5.2f per-model %s  resid/sd %s  mu -> %s"
                      % (t, e["actor"], "HOP " if e["hop"] else "    ", e["kind"], e["nis"], "/".join("%.1f" % v for v in e["nis_m"]),
                         " ".join("%+.1f" % (a / max(b, 1e-9)) for a, b in zip(e["nu"], e["sd"])), "/".join("%.2f" % v for v in e["mu1"])))
            elif t - lastf >= 0.35:
                lastf = t
                tru = ("truth (%.1f, %.1f) v (%.1f, %.1f) along %+.1f across %+.1f" % (e["tx"], e["ty"], e["tvx"], e["tvy"], e["along"], e["across"])) if "tx" in e else "no truth"
                print("  %.2f    %-9s at (%.1f, %.1f) v (%.1f, %.1f) sd %s  p(stand) %.2f %-10s s %+.2f h %.2f  %s  %s"
                      % (t, e["state"], e["x"], e["y"], e["vx"], e["vy"], "/".join("%.1f" % v for v in e["sd"]), e["mu"][0], e["verdict"],
                         e["sc"], math.exp(e["lh"]), tru, ",".join(e.get("flags", ()))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="parked_street")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--night", action="store_true")
    ap.add_argument("--actor", default=None)
    ap.add_argument("--object", type=int, default=None)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--cfg", default=None, help="radar configuration (JSON)")
    ap.add_argument("--det", default=None, help="detector simulator parameter overrides (JSON)")
    a = ap.parse_args(argv)
    w, det, box, R = run(a.scenario, a.seed, a.night, json.loads(a.cfg) if a.cfg else None, json.loads(a.det) if a.det else None)
    print("%s seed %d %s: %d scans, %d updates, %d objects started" % (a.scenario, a.seed, "night" if a.night else "day",
                                                                      len(R["scans"]), len(R["upd"]), len(R["start"])))
    if a.actor is not None or a.object is not None:
        act = a.actor
        if act is not None and act.lstrip("-").isdigit():
            act = int(act)
        timeline(R, act, a.object)
    else:
        report_meas(R)
        report_ego(R)
        report_assoc(R)
        report_state(R)
        report_nis(R)
    if a.dump:
        with open(a.dump, "w") as f:
            for k in ("meas", "upd", "start", "frames"):
                for r in R[k]:
                    f.write(json.dumps(dict(r, _k=k), default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
