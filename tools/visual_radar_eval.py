#!/usr/bin/env python3
"""Evaluation harness of the "visual radar" (road-user detections as a tracking sensor of the object layer):
scenarios with ground truth through the whole stack, metrics, a table and a JSON file. docs/VISUAL_RADAR_EVAL.md.

    python3 tools/visual_radar_eval.py                         # all scenarios, 3 seeds, day; red alert 10 seeds
    python3 tools/visual_radar_eval.py --both --json out.json  # day and night
    python3 tools/visual_radar_eval.py --baseline              # score today's display objects (view, kind road_user)
    python3 tools/visual_radar_eval.py --scenario curve_parked --seeds 5 --night

Scored outputs. Baseline runs build the estimator WITHOUT the road_users source (as before road users were
tracked: tracks and lead do not see the boxes; the view draws them). Tracks mode (default): every track of the `tracks` message (camera lead and road users alike);
state confirmed / coast / tentative; `motion` ("moving" | "stationary" | "unknown"; absent -> "stationary" when
`flags` has stationary or parked, else "unknown"); `on_path` from `flags`. Baseline mode (--baseline): the
`tracks` message's tracks plus the view's `road_user` objects (y sign turned to left; ground velocity = relative
velocity + ego speed - w y, relative + w x), both with a motion DERIVED from the ground speed: stationary below
1.0 m/s, moving above 2.0 m/s, unknown in between (fields the tree may already send are ignored there).
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

NS = 1_000_000_000
STAT_BELOW, MOVE_ABOVE = 1.0, 2.0          # baseline's derived motion thresholds (m/s, ground speed)
TRACK_SOURCES = [{"kind": "road_users", "id": "*"}]   # + the estimator's default sources (camera lead, ...); no radar


def _imports(retries=12, wait_s=20.0):
    """The stack's modules; the tree may be mid-edit by someone else: wait and retry, never patch."""
    last = None
    for _ in range(retries):
        try:
            from chris_integrator.mixer import MessageClock
            from chris_integrator.sim import default_sensors
            from chris_integrator.sim import roaduser_sim as rs
            from chris_integrator.stack import build
            return MessageClock, default_sensors, rs, build
        except Exception as e:                      # noqa: BLE001  (a syntax error of a file being edited, ...)
            last = e
            print("import failed (%s: %s); retrying in %.0f s" % (type(e).__name__, e, wait_s), file=sys.stderr)
            time.sleep(wait_s)
    raise last


# ---- one run through the stack ----------------------------------------------------------------------------------
def run_stack(name, seed, night, road_users=True, det_params=None, track_source=True, camera_lead=True,
              radar_plugin=None):
    """One scenario run. road_users: deliver the detector's `lights` messages; track_source: register the
    `road_users` source kind with the estimator (False = the baseline: the boxes reach the road_users box and the
    view's display objects only, so tracks and lead are what they were before road users existed)."""
    MessageClock, default_sensors, rs, build = _imports()
    w = rs.scenario(name, seed=seed, night=night)
    det = rs.DetectorSim(seed=1000 + seed, night=night, params=det_params)
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": camera_lead},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
    if road_users:
        sensors.append(det)
    srcs = TRACK_SOURCES if (road_users and track_source) else []
    extra = []
    if radar_plugin and track_source:
        mod, _, cls = radar_plugin.partition(":")
        import importlib
        extra.append(getattr(importlib.import_module(mod), cls)(None))
    st = build(cfg={"estimator": {"sources": srcs}}, mapper=None, clock=MessageClock(), plugins=extra or None)
    frames = {}
    order = []

    def rec(e):
        m = e.msg
        ts = m.get("ts_mono_ns")
        if not isinstance(ts, int):
            return
        f = frames.get(ts)
        if f is None:
            f = frames[ts] = {"ts": ts, "tracks": None, "radar": None, "lead": None, "view": None, "ego_v": None, "ego_w": None}
            order.append(ts)
        s = e.stream
        if s == "tracks":
            f["tracks"] = [{k: tr.get(k) for k in ("id", "state", "x_m", "y_m", "vx_mps", "vy_mps", "flags", "class",
                                                    "sources", "motion", "ttc_s")} for tr in m.get("tracks") or ()]
        elif s == "radar":
            f["radar"] = [{k: o.get(k) for k in ("id", "state", "x_m", "y_m", "vx_mps", "vy_mps", "flags", "class",
                                                  "sources", "motion", "ttc_s", "p_stationary")}
                          for o in m.get("objects") or ()]
        elif s == "lead":
            f["lead"] = {k: m.get(k) for k in ("present", "dist_m", "lateral_m", "v_lead_mps", "v_rel_mps", "track_id",
                                               "ttc_s", "on_path")}
            eg = m.get("ego") or {}
            if f["ego_v"] is None and isinstance(eg.get("speed_mps"), (int, float)):
                f["ego_v"] = float(eg["speed_mps"])
        elif s == "ego_state":
            sp, yr = m.get("speed") or {}, m.get("yaw_rate") or {}
            if isinstance(sp.get("value"), (int, float)):
                f["ego_v"] = float(sp["value"])
            if isinstance(yr.get("value"), (int, float)):
                f["ego_w"] = float(yr["value"])
        elif s == "view":
            f["view"] = [{k: o.get(k) for k in ("id", "x", "y", "vx", "vy", "phase", "cls")}
                         for o in m.get("objects") or () if o.get("kind") == "road_user"]

    st.mixer.add_output("eval", rec, streams=("tracks", "radar", "lead", "ego_state", "view"))
    cost = {"road_frames": 0, "deliver_s": 0.0, "lights": 0}
    calib_sent = False
    for t_arr, m in w.messages(sensors):
        if not calib_sent:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            calib_sent = True
        t0 = time.perf_counter()
        st.deliver(m)
        cost["deliver_s"] += time.perf_counter() - t0
        if m["type"] == "road":
            cost["road_frames"] += 1
        elif m["type"] == "lights":
            cost["lights"] += 1
    return w, det, [frames[ts] for ts in sorted(order)], cost


# ---- outputs as scored objects ----------------------------------------------------------------------------------
def _verdict_track(tr):
    m = tr.get("motion")
    if m in ("moving", "stationary", "unknown"):
        return m
    fl = tr.get("flags") or ()
    return "stationary" if ("stationary" in fl or "parked" in fl) else "unknown"


def _derived(speed):
    return "stationary" if speed < STAT_BELOW else ("moving" if speed > MOVE_ABOVE else "unknown")


def resolve_source(frames, want):
    """auto: `radar` if the stack emitted it, else `tracks`. view: the baseline."""
    if want in ("radar", "tracks", "view"):
        return want
    return "radar" if any(f.get("radar") is not None for f in frames) else "tracks"


def outputs_of(f, baseline, source="tracks"):
    """[{oid, state, x, y (left), vx, vy (ground), motion, on_path, parked_flag, label, ttc}]. source: "radar"
    (the radar component's objects), "tracks" (the estimator's), "view" (= baseline)."""
    out = []
    if source == "radar" and not baseline:
        for o in f.get("radar") or ():
            if not all(isinstance(o.get(k), (int, float)) for k in ("x_m", "y_m")):
                continue
            vx, vy = o.get("vx_mps"), o.get("vy_mps")
            vx = float(vx) if isinstance(vx, (int, float)) else float("nan")
            vy = float(vy) if isinstance(vy, (int, float)) else float("nan")
            fl = o.get("flags") or []
            cls = o.get("class")
            out.append({"oid": "r:%s" % o.get("id"), "state": o.get("state") or "confirmed", "x": float(o["x_m"]),
                        "y": float(o["y_m"]), "vx": vx, "vy": vy, "motion": _verdict_track(o),
                        "on_path": "on_path" in fl or "in_path" in fl, "in_path": "in_path" in fl, "threat": "threat" in fl,
                        "parked_flag": "parked" in fl or "stationary" in fl,
                        "label": cls.get("label") if isinstance(cls, dict) else None,
                        "ttc": o.get("ttc_s") if isinstance(o.get("ttc_s"), (int, float)) else None})
        return out
    for tr in f["tracks"] or ():
        if baseline and any(str(x).replace("_", "") == "roadusers" for x in tr.get("sources") or ()):
            continue                     # baseline: road users are display objects only (the view's, below)
        if not all(isinstance(tr.get(k), (int, float)) for k in ("x_m", "y_m")):
            continue
        vx, vy = tr.get("vx_mps"), tr.get("vy_mps")
        vx = float(vx) if isinstance(vx, (int, float)) else float("nan")
        vy = float(vy) if isinstance(vy, (int, float)) else float("nan")
        fl = tr.get("flags") or []
        mot = _derived(math.hypot(vx, vy)) if baseline and vx == vx else _verdict_track(tr)
        if baseline and vx != vx:
            mot = "unknown"
        cls = tr.get("class")
        out.append({"oid": "t:%s" % tr.get("id"), "state": tr.get("state") or "confirmed", "x": float(tr["x_m"]),
                    "y": float(tr["y_m"]), "vx": vx, "vy": vy, "motion": mot, "on_path": "on_path" in fl,
                    "parked_flag": "parked" in fl or "stationary" in fl,
                    "label": cls.get("label") if isinstance(cls, dict) else None,
                    "ttc": tr.get("ttc_s") if isinstance(tr.get("ttc_s"), (int, float)) else None})
    if baseline:
        v = f["ego_v"] or 0.0
        wz = f["ego_w"] or 0.0
        for o in f["view"] or ():
            ph = o.get("phase")
            if ph == "leave" or not isinstance(o.get("x"), (int, float)):
                continue
            x, y = float(o["x"]), -float(o["y"])
            vxr, vyr = float(o.get("vx") or 0.0), -float(o.get("vy") or 0.0)
            vx, vy = vxr + v - wz * y, vyr + wz * x
            cls = o.get("cls")
            out.append({"oid": "v:%s" % o["id"], "state": "coast" if ph == "coast" else "confirmed", "x": x, "y": y,
                        "vx": vx, "vy": vy, "motion": _derived(math.hypot(vx, vy)), "on_path": False,
                        "parked_flag": False, "label": cls.get("label") if isinstance(cls, dict) else None, "ttc": None})
    return out


def _gate(a, ox, oy):
    """(cost, along error, across error) of an output point against an actor (LOS of its nearest footprint point)."""
    r = max(a["range"], 0.5)
    ux, uy = a["ref_x"] / r, a["ref_y"] / r
    ex, ey = ox - a["ref_x"], oy - a["ref_y"]
    al = ex * ux + ey * uy
    ac = -ex * uy + ey * ux
    ext = 0.5 * max(a["L"], a["W"])
    ga = max(3.0, 0.35 * r) + ext
    gc = max(1.5, 0.08 * r) + 0.5 * a["W"]
    return math.hypot(al / ga, ac / gc), al, ac


def match(outs, actors, prev=None):
    """CLEAR-MOT style assignment of confirmed/coast outputs to truth actors: a correspondence of the previous
    frame (prev: {output id: actor id}) is kept while it is inside the gate; the rest by optimal assignment on the
    gated cost. Returns {output index: (actor, along error, across error)}."""
    idx = [i for i, o in enumerate(outs) if o["state"] in ("confirmed", "coast")]
    if not idx or not actors:
        return {}
    out = {}
    used_a = set()
    if prev:
        amap = {a["id"]: a for a in actors}
        for i in idx:
            a = amap.get(prev.get(outs[i]["oid"]))
            if a is None or a["id"] in used_a:
                continue
            cost, al, ac = _gate(a, outs[i]["x"], outs[i]["y"])
            if cost <= 1.0:
                out[i] = (a, al, ac)
                used_a.add(a["id"])
    idx = [i for i in idx if i not in out]
    acts = [a for a in actors if a["id"] not in used_a]
    if not idx or not acts:
        return out
    C = np.full((len(idx), len(acts)), 1e6)
    E = {}
    for r, i in enumerate(idx):
        o = outs[i]
        for c, a in enumerate(acts):
            cost, al, ac = _gate(a, o["x"], o["y"])
            if cost <= 1.0:
                C[r, c] = cost
                E[(r, c)] = (al, ac)
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(C)
        pairs = list(zip(rows, cols))
    except ImportError:                                   # greedy fallback
        pairs, ur, uc = [], set(), set()
        for r, c in sorted(np.ndindex(C.shape), key=lambda rc: C[rc]):
            if r not in ur and c not in uc and C[r, c] < 1e6:
                pairs.append((r, c))
                ur.add(r)
                uc.add(c)
    for r, c in pairs:
        if C[r, c] <= 1.0:
            out[idx[r]] = (acts[c],) + E[(r, c)]
    return out


# ---- scoring one run --------------------------------------------------------------------------------------------
def score_run(w, det, frames, baseline, source="tracks"):
    first_det = {}
    nbox = nfp = 0
    for t, view, logd in det.log:
        for d in logd:
            nbox += 1
            if str(d[0]).startswith("fp"):
                nfp += 1
            else:
                first_det.setdefault(d[0], t)
    A = {}                     # per actor accumulators
    trk = {}                   # per output id: confirmed time, unmatched confirmed time
    fa = {"track_s": 0.0, "track_episodes": set(), "lead_s": 0.0, "lead_episodes": 0, "late_lc_s": 0.0,
          "strict_s": 0.0, "strict_episodes": set(), "threat_s": 0.0, "threat_episodes": set()}
    alert = {}
    spec = w.spec
    lead_fa_prev = False
    prev = {}
    T0 = w.t0
    for k, f in enumerate(frames):
        t = f["ts"] / NS
        dt = min(0.2, (frames[k + 1]["ts"] - f["ts"]) / NS) if k + 1 < len(frames) else 0.0
        actors = [a for a in w.actors_at(t) if a["ref_x"] > -10.0]
        vis = det.visibility(w, t, actors)
        outs = outputs_of(f, baseline, source)
        M = match(outs, actors, prev)
        prev = {outs[i]["oid"]: a["id"] for i, (a, _, _) in M.items()}
        by_actor = {a["id"]: (i, al, ac) for i, (a, al, ac) in M.items()}
        ego_v = w.truth_at(t)["v"]
        for a in actors:
            aid = a["id"]
            S = A.get(aid)
            if S is None:
                S = A[aid] = {"role": a["role"], "kind": a["kind"], "moving": a["moving"], "det_s": 0.0, "cov_s": 0.0,
                              "held_s": 0.0, "first_vis": None, "first_conf": None, "ids": [], "last_id": None,
                              "switches": 0, "segments": 0, "last_match_t": None, "errs": [], "conf": {},
                              "verdicts": [], "parked_s": 0.0, "stat_s": 0.0, "matched_s": 0.0, "labels": {},
                              "label_truth": a["stationary_label"]}
            d_ok = vis.get(aid, (False, False))[0]
            if d_ok:
                S["det_s"] += dt
                if S["first_vis"] is None:
                    S["first_vis"] = t
            m = by_actor.get(aid)
            if m is None:
                continue
            i, al, ac = m
            o = outs[i]
            if S["last_match_t"] is None or t - S["last_match_t"] > 1.0:
                S["segments"] += 1
            S["last_match_t"] = t
            if S["last_id"] is not None and o["oid"] != S["last_id"]:
                S["switches"] += 1
            S["last_id"] = o["oid"]
            if o["oid"] not in S["ids"]:
                S["ids"].append(o["oid"])
            if d_ok:
                S["held_s"] += dt
            if o["state"] != "confirmed":
                continue
            if d_ok:
                S["cov_s"] += dt
            if S["first_conf"] is None:
                S["first_conf"] = t
            S["matched_s"] += dt
            tm = "moving" if a["moving"] else "stationary"
            key = (tm, o["motion"])
            S["conf"][key] = S["conf"].get(key, 0.0) + dt
            S["verdicts"].append((t, o["motion"] == tm))
            if o["motion"] == "stationary" or o["parked_flag"]:
                S["stat_s"] += dt
            S["labels"][o["label"]] = S["labels"].get(o["label"], 0.0) + dt
            if o["vx"] == o["vx"]:
                S["errs"].append((a["range"], al, ac, o["vx"] - a["vx"], o["vy"] - a["vy"]))
            else:
                S["errs"].append((a["range"], al, ac, None, None))
        # false tracks
        for i, o in enumerate(outs):
            if o["state"] != "confirmed":
                continue
            R = trk.setdefault(o["oid"], {"conf_s": 0.0, "free_s": 0.0, "t0": t, "t1": t})
            R["conf_s"] += dt
            R["t1"] = t
            if i not in M:
                R["free_s"] += dt
        # stationary-in-path verdicts
        sip = [a for a in actors if a["stationary_label"] == "in_path"]
        lc_done = spec.get("lc_t") is not None and t - T0 > spec["lc_t"] + 3.0
        in_lc = spec.get("lc_t") is not None and spec["lc_t"] <= t - T0 <= spec["lc_t"] + 3.0   # grace: neither
        if in_lc:                    # during the move the stopped car is neither a hit nor a false alarm
            sip = [a for a in sip if a["role"] != "stopped_in_lane"]
        for i, o in enumerate(outs):                 # the threat flag: false when the thing is not reached within 4 s
            if not o.get("threat") or ego_v <= 3.0:
                continue
            a = M.get(i, (None,))[0]
            if in_lc and a is not None and a["role"] == "stopped_in_lane":
                continue
            closing = ego_v - a["vx"] if a is not None else 0.0
            real = a is not None and closing > 0.5 and a["ref_x"] / closing <= 4.0 and abs(a.get("lane_offset", 0.0)) < 1.8
            if not real:
                fa["threat_s"] += dt
                fa["threat_episodes"].add(o["oid"])
        for i, o in enumerate(outs):
            if o["state"] != "confirmed" or not o["on_path"] or o["motion"] != "stationary":
                continue
            a = M.get(i, (None,))[0]
            if in_lc and a is not None and a["role"] == "stopped_in_lane":
                continue
            true_sip = a is not None and a["stationary_label"] == "in_path"
            if a is not None and a["role"] == "stopped_in_lane" and lc_done:
                fa["late_lc_s"] += dt
                continue
            if not true_sip and ego_v > 3.0:
                fa["track_s"] += dt
                fa["track_episodes"].add(o["oid"])
                if o.get("in_path"):                     # the radar's own obstacle flag: what an alert is raised on
                    fa["strict_s"] += dt
                    fa["strict_episodes"].add(o["oid"])
        ld = f["lead"] or {}
        lead_stat = False
        lead_actor = None
        if ld.get("present") and isinstance(ld.get("dist_m"), (int, float)) and isinstance(ld.get("v_lead_mps"), (int, float)):
            ve = f["ego_v"] if f["ego_v"] is not None else ego_v
            lead_stat = ve > 3.0 and abs(ld["v_lead_mps"]) <= max(1.5, 0.15 * ve)
            lx, ly = float(ld["dist_m"]), -float(ld.get("lateral_m") or 0.0)
            for o in outs:
                if o["oid"] == "t:%s" % ld.get("track_id"):
                    lx, ly = o["x"], o["y"]
            best = None
            for a in actors:
                c, _, _ = _gate(a, lx, ly)
                if c <= 1.0 and (best is None or c < best[0]):
                    best = (c, a)
            lead_actor = best[1] if best else None
        lead_true = lead_actor is not None and lead_actor["stationary_label"] == "in_path"
        lfa = lead_stat and not lead_true and ego_v > 3.0
        if in_lc and lead_actor is not None and lead_actor["role"] == "stopped_in_lane":
            lfa = False
        if lead_stat and lead_actor is not None and lead_actor["role"] == "stopped_in_lane" and lc_done:
            fa["late_lc_s"] += dt
            lfa = False
        if lfa:
            fa["lead_s"] += dt
            if not lead_fa_prev:
                fa["lead_episodes"] += 1
        lead_fa_prev = lfa
        # red alert on the stopped car
        for a in sip:
            if a["role"] != "stopped_in_lane" or lc_done:
                continue
            ttc_true = a["ref_x"] / ego_v if ego_v > 0.5 else None
            m = by_actor.get(a["id"])
            if "track" not in alert and m is not None:
                o = outs[m[0]]
                if o["state"] == "confirmed" and o["on_path"] and o["motion"] == "stationary":
                    alert["track"] = {"t": t - T0, "dist_m": a["ref_x"], "ego_v": ego_v, "ttc_s": ttc_true,
                                      "stack_ttc_s": o["ttc"]}
            if "threat" not in alert and m is not None and outs[m[0]].get("threat"):
                alert["threat"] = {"t": t - T0, "dist_m": a["ref_x"], "ego_v": ego_v, "ttc_s": ttc_true,
                                   "stack_ttc_s": outs[m[0]]["ttc"]}
            if "obstacle" not in alert and m is not None and outs[m[0]].get("in_path"):
                alert["obstacle"] = {"t": t - T0, "dist_m": a["ref_x"], "ego_v": ego_v, "ttc_s": ttc_true,
                                     "stack_ttc_s": outs[m[0]]["ttc"]}
            if "lead" not in alert and lead_stat and lead_actor is not None and lead_actor["id"] == a["id"]:
                alert["lead"] = {"t": t - T0, "dist_m": a["ref_x"], "ego_v": ego_v, "ttc_s": ttc_true,
                                 "stack_ttc_s": ld.get("ttc_s")}
    if spec.get("approach"):
        a800 = A.get(800, {})
        fd = first_det.get(800)
        def at(tq):
            tr = [a for a in w.actors_at(tq) if a["id"] == 800]
            v = w.truth_at(tq)["v"]
            return {"t": tq - T0, "dist_m": tr[0]["ref_x"] if tr else None,
                    "ttc_s": tr[0]["ref_x"] / v if tr and v > 0.5 else None}
        if fd is not None:
            alert["first_detection"] = at(fd)
        if a800.get("first_conf"):
            alert["first_confirmed"] = at(a800["first_conf"])
        ta = spec.get("brake_t") if spec["mode"] == "brake" else spec.get("lc_t")
        if ta is not None:
            alert["driver_action"] = at(T0 + ta)
        alert.update(approach=spec["approach"], mode=spec["mode"], brake_t=spec.get("brake_t"), lc_t=spec.get("lc_t"))
    dur = (frames[-1]["ts"] - frames[0]["ts"]) / NS if frames else 0.0
    return {"source": "view" if baseline else source, "actors": A, "first_det": first_det, "tracks": trk, "fa": fa, "alert": alert, "duration_s": dur,
            "boxes": nbox, "fp_boxes": nfp, "messages": len(det.log)}


# ---- aggregation ------------------------------------------------------------------------------------------------
def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def _r(x, n=2):
    return None if x is None else round(x, n)


def stable_time(verdicts, t_first, hold=2.0):
    """Time from the first detection to a correct verdict that holds for `hold` s of matched frames."""
    for j, (t, ok) in enumerate(verdicts):
        if not ok:
            continue
        good, end = True, t
        for t2, ok2 in verdicts[j:]:
            if t2 - t > hold:
                break
            end = t2
            if not ok2:
                good = False
                break
        if good and end - t >= min(hold, verdicts[-1][0] - t) and end - t >= 0.6:
            return t - t_first
    return None


def err_stats(E):
    """[(range, along, across, dvx, dvy)] -> medians / p90 of the errors."""
    al = [e[1] for e in E]
    ac = [e[2] for e in E]
    rel = [abs(e[1]) / max(e[0], 1.0) for e in E]
    dv = [math.hypot(e[3], e[4]) for e in E if e[3] is not None]
    return {"n": len(E), "along_bias_m": _r(statistics.fmean(al)) if al else None,
            "along_abs_p50_m": _r(_pct([abs(x) for x in al], 0.5)),
            "along_abs_p90_m": _r(_pct([abs(x) for x in al], 0.9)),
            "along_rel_p50": _r(_pct(rel, 0.5), 3),
            "across_abs_p50_m": _r(_pct([abs(x) for x in ac], 0.5)),
            "across_abs_p90_m": _r(_pct([abs(x) for x in ac], 0.9)),
            "vel_err_p50_mps": _r(_pct(dv, 0.5)), "vel_err_p90_mps": _r(_pct(dv, 0.9))}


def aggregate(scored):
    """scored: list of score_run outputs of one scenario x light -> summary dict."""
    det_s = cov_s = held_s = 0.0
    ttfc, sw, seg, nact = [], 0, [], 0
    errs = {}
    conf = {}
    conf_kind = {}
    stab = {}
    parked = [0.0, 0.0]
    moving = [0.0, 0.0]
    labels = {}
    false_n = 0
    false_life = []
    free_s = 0.0
    dur = 0.0
    fa = {"track_s": 0.0, "track_episodes": 0, "lead_s": 0.0, "lead_episodes": 0, "late_lc_s": 0.0, "strict_s": 0.0,
          "strict_episodes": 0, "threat_s": 0.0, "threat_episodes": 0}
    boxes = msgs = fpb = 0
    for sc in scored:
        dur += sc["duration_s"]
        boxes += sc["boxes"]
        msgs += sc["messages"]
        fpb += sc["fp_boxes"]
        for k in ("track_s", "lead_s", "late_lc_s", "lead_episodes", "strict_s", "threat_s"):
            fa[k] += sc["fa"][k]
        fa["track_episodes"] += len(sc["fa"]["track_episodes"])
        fa["strict_episodes"] += len(sc["fa"]["strict_episodes"])
        fa["threat_episodes"] += len(sc["fa"]["threat_episodes"])
        for aid, S in sc["actors"].items():
            if S["det_s"] <= 0.0 and S["matched_s"] <= 0.0:
                continue
            nact += 1
            det_s += S["det_s"]
            cov_s += S["cov_s"]
            held_s += S["held_s"]
            fd = sc["first_det"].get(aid)
            if fd is not None and S["first_conf"] is not None and S["first_conf"] >= fd - 1.0:
                ttfc.append(S["first_conf"] - fd)
            sw += S["switches"]
            if S["segments"]:
                seg.append(S["segments"])
            role = S["role"]
            errs.setdefault(role, []).extend(S["errs"])
            for key, v in S["conf"].items():
                conf[key] = conf.get(key, 0.0) + v
                ck = conf_kind.setdefault(role, {})
                ck[key] = ck.get(key, 0.0) + v
            if fd is not None and S["verdicts"]:
                stab.setdefault(role, []).append(stable_time(S["verdicts"], fd))
            if role == "parked":
                parked[0] += S["stat_s"]
                parked[1] += S["matched_s"]
            if S["moving"]:
                moving[0] += S["stat_s"]
                moving[1] += S["matched_s"]
            truth_lab = {"car": "car", "truck": "truck", "human": "human"}[S["kind"]]
            for lab, v in S["labels"].items():
                kk = "none" if lab is None else ("right" if lab == truth_lab or (truth_lab == "truck" and lab in ("van", "bus")) else "wrong")
                labels[kk] = labels.get(kk, 0.0) + v
        for oid, R in sc["tracks"].items():
            free_s += R["free_s"]
            if R["conf_s"] > 0.0 and R["free_s"] >= 0.5 * R["conf_s"] and R["free_s"] >= 0.3:
                false_n += 1
                false_life.append(R["conf_s"])
    mins = dur / 60.0 if dur else 1.0
    err_out = {role: err_stats(E) for role, E in errs.items()}

    def cm(d):
        return {"%s->%s" % k: round(v, 1) for k, v in sorted(d.items())}
    return {
        "runs": len(scored), "duration_s": round(dur, 1), "actors_scored": nact,
        "detector": {"messages": msgs, "boxes_per_msg": _r(boxes / msgs if msgs else None), "fp_boxes": fpb},
        "coverage": _r(cov_s / det_s if det_s else None, 3), "held_coverage": _r(held_s / det_s if det_s else None, 3),
        "detectable_s": round(det_s, 1),
        "time_to_first_confirmed_s": {"p50": _r(_pct(ttfc, 0.5)), "p90": _r(_pct(ttfc, 0.9)), "n": len(ttfc)},
        "id_switches_per_actor": _r(sw / nact if nact else None, 3),
        "segments_per_tracked_actor": _r(statistics.fmean(seg) if seg else None),
        "false_tracks_per_min": _r(false_n / mins), "false_track_life_s": {"p50": _r(_pct(false_life, 0.5)),
                                                                          "p90": _r(_pct(false_life, 0.9)),
                                                                          "max": _r(max(false_life) if false_life else None)},
        "unmatched_confirmed_track_s_per_min": _r(free_s / mins),
        "errors": err_out,
        "motion_confusion_s": cm(conf),
        "motion_confusion_by_role_s": {r: cm(d) for r, d in conf_kind.items()},
        "stable_verdict_s": {r: {"p50": _r(_pct([x for x in v if x is not None], 0.5)),
                                 "never": sum(1 for x in v if x is None), "n": len(v)} for r, v in stab.items()},
        "parked_marked_stationary": _r(parked[0] / parked[1] if parked[1] else None, 3), "parked_matched_s": round(parked[1], 1),
        "moving_marked_stationary": _r(moving[0] / moving[1] if moving[1] else None, 4), "moving_matched_s": round(moving[1], 1),
        "class_share": {k: _r(v / sum(labels.values()), 3) for k, v in labels.items()} if labels else {},
        "false_alarms": {"stationary_in_path_track_s": round(fa["track_s"], 2), "track_episodes": fa["track_episodes"],
                         "obstacle_flag_s": round(fa["strict_s"], 2), "obstacle_flag_episodes": fa["strict_episodes"],
                         "threat_flag_s": round(fa["threat_s"], 2), "threat_flag_episodes": fa["threat_episodes"],
                         "lead_stationary_s": round(fa["lead_s"], 2), "lead_episodes": fa["lead_episodes"],
                         "after_lane_change_s": round(fa["late_lc_s"], 2), "per_min_track": _r(fa["track_episodes"] / mins),
                         "per_min_lead": _r(fa["lead_episodes"] / mins)},
    }


def alert_summary(scored):
    out = {}
    for key in ("track", "obstacle", "threat", "lead", "first_detection", "first_confirmed", "driver_action"):
        ds = [sc["alert"][key]["dist_m"] for sc in scored if key in sc["alert"] and sc["alert"][key].get("dist_m") is not None]
        tt = [sc["alert"][key]["ttc_s"] for sc in scored if key in sc["alert"] and sc["alert"][key].get("ttc_s") is not None]
        out[key] = {"hit": len(ds), "of": len(scored), "dist_p50_m": _r(_pct(ds, 0.5)), "dist_min_m": _r(min(ds) if ds else None),
                    "dist_max_m": _r(max(ds) if ds else None), "ttc_p50_s": _r(_pct(tt, 0.5)), "ttc_min_s": _r(min(tt) if tt else None)}
        if key in ("track", "obstacle", "threat", "lead"):
            pre = 0
            for sc in scored:
                al = sc["alert"]
                ev = al.get(key)
                tb = al.get("brake_t") if al.get("mode") == "brake" else al.get("lc_t")
                if ev is not None and tb is not None and ev["t"] <= tb:
                    pre += 1
            out[key]["before_driver_action"] = pre
            errs = [abs(sc["alert"][key]["stack_ttc_s"] - sc["alert"][key]["ttc_s"]) / sc["alert"][key]["ttc_s"]
                    for sc in scored if key in sc["alert"] and isinstance(sc["alert"][key].get("stack_ttc_s"), (int, float))
                    and sc["alert"][key].get("ttc_s")]
            out[key]["stack_ttc_rel_err_p50"] = _r(_pct(errs, 0.5), 3)
            out[key]["stack_ttc_given"] = len(errs)
    return out


# ---- report -----------------------------------------------------------------------------------------------------
def _f(x, fmt="%.2f"):
    return "-" if x is None else fmt % x


def print_table(res):
    print("\n== visual radar evaluation (scored: %s; camera lead %s) ==" % (
        "BASELINE view display objects" if res["meta"]["baseline"] else ", ".join(
            "%s x%d" % kv for kv in (res["meta"].get("sources_scored") or {}).items()),
        "on" if res["meta"].get("camera_lead", True) else "OFF"))
    hdr = ("scenario", "lt", "cov", "held", "ttfc50", "sw/act", "seg", "falseT/min", "fl50", "moving->stat",
           "parked->stat", "FA trk/lead", "box/msg")
    print("%-15s %-2s %5s %5s %6s %6s %4s %10s %5s %12s %12s %11s %7s" % hdr)
    for key, s in res["scenarios"].items():
        name, lt = key.rsplit("/", 1)
        fa = s["false_alarms"]
        print("%-15s %-2s %5s %5s %6s %6s %4s %10s %5s %12s %12s %11s %7s" % (
            name, lt[0].upper(), _f(s["coverage"]), _f(s["held_coverage"]), _f(s["time_to_first_confirmed_s"]["p50"], "%.1f"),
            _f(s["id_switches_per_actor"]), _f(s["segments_per_tracked_actor"], "%.1f"), _f(s["false_tracks_per_min"]),
            _f(s["false_track_life_s"]["p50"], "%.1f"), _f(s["moving_marked_stationary"], "%.3f"),
            _f(s["parked_marked_stationary"]), "%d/%d" % (fa["track_episodes"], fa["lead_episodes"]),
            _f(s["detector"]["boxes_per_msg"])))
    print("\nmotion confusion (s, truth->output), all roles:")
    for key, s in res["scenarios"].items():
        print("  %-20s %s" % (key, s["motion_confusion_s"]))
    print("\nposition / velocity error per role (pooled over scenarios; m, m/s):")
    print("  %-22s %6s %7s %7s %7s %7s %7s %7s" % ("role", "n", "al_bias", "|al|50", "|al|90", "|ac|50", "|dv|50", "|dv|90"))
    for role, e in res["errors_by_role"].items():
        print("  %-22s %6d %7s %7s %7s %7s %7s %7s" % (role, e["n"], _f(e["along_bias_m"]), _f(e["along_abs_p50_m"]),
                                                       _f(e["along_abs_p90_m"]), _f(e["across_abs_p50_m"]),
                                                       _f(e["vel_err_p50_mps"]), _f(e["vel_err_p90_mps"])))
    if res.get("alert"):
        print("\nred alert: stationary car in the ego lane (dist m / TTC s at the first verdict; p50 [min..max]; n hit/of; pre = before brake/lane change)")
        print("  %-22s %-26s %-26s %-18s %-18s %-12s" % ("run", "track(on_path+stationary)", "lead(present, v~0)", "first det",
                                                        "first confirmed", "driver acts"))
        for key, a in res["alert"].items():
            def cell(c, pre=True):
                if not c["hit"]:
                    return "none 0/%d" % c["of"]
                s = "%s/%s m %s s %d/%d" % (_f(c["dist_p50_m"], "%.0f"), _f(c["dist_min_m"], "%.0f"), _f(c["ttc_p50_s"], "%.1f"),
                                            c["hit"], c["of"])
                return s + (" pre%d" % c["before_driver_action"] if pre else "")
            da = a.get("driver_action") or {}
            print("  %-22s %-26s %-26s %-18s %-18s %-12s" % (key, cell(a["track"]), cell(a["lead"]), cell(a["first_detection"], False),
                                                             cell(a["first_confirmed"], False),
                                                             "%s m %s s" % (_f(da.get("dist_p50_m"), "%.0f"), _f(da.get("ttc_p50_s"), "%.1f"))))
    if res.get("cost"):
        c = res["cost"]
        print("\ncost (%s): %.3f ms per road frame with road users, %.3f ms without (+%.0f %%)" % (
            c["scenario"], c["with_ms"], c["without_ms"], 100.0 * (c["with_ms"] / c["without_ms"] - 1.0) if c["without_ms"] else 0))


# ---- main -------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="all", help="a scenario name, 'traffic', 'alert' or 'all'")
    ap.add_argument("--seeds", type=int, default=3, help="seeds per traffic scenario")
    ap.add_argument("--alert-seeds", type=int, default=10, help="seeds per red-alert run (>= 10 for the report)")
    ap.add_argument("--night", action="store_true", help="night parameter set only")
    ap.add_argument("--both", action="store_true", help="day and night")
    ap.add_argument("--baseline", action="store_true", help="= --source view: today's display objects (view road_user)")
    ap.add_argument("--source", default="auto", choices=("auto", "radar", "tracks", "view"),
                    help="what to score: the radar component's `radar` stream, the estimator's `tracks`, or the "
                         "view's road_user display objects (baseline); auto: radar if emitted, else tracks")
    ap.add_argument("--radar-plugin", default=None,
                    help="module:Class of a radar plug-in to add to the stack when build() does not include it")
    ap.add_argument("--json", default=None, help="write the results here")
    ap.add_argument("--no-cost", action="store_true", help="skip the cost measurement")
    ap.add_argument("--lead-off", action="store_true",
                    help="the road model reports no lead (road.leads absent): what the road-user tracks do alone")
    a = ap.parse_args(argv)
    if a.source == "view":
        a.baseline = True
    _, _, rs, _ = _imports()
    if a.scenario == "all":
        names = list(rs.SCENARIOS) + list(rs.ALERT_SCENARIOS)
    elif a.scenario == "traffic":
        names = list(rs.SCENARIOS)
    elif a.scenario == "alert":
        names = list(rs.ALERT_SCENARIOS)
    else:
        names = [a.scenario]
    lights = ["day", "night"] if a.both else (["night"] if a.night else ["day"])
    res = {"meta": {"baseline": a.baseline, "camera_lead": not a.lead_off, "seeds": a.seeds, "alert_seeds": a.alert_seeds, "lights": lights,
                    "thresholds": {"baseline_stationary_below_mps": STAT_BELOW, "baseline_moving_above_mps": MOVE_ABOVE},
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")},
           "scenarios": {}, "alert": {}, "errors": [], "runs": []}
    pooled = {}
    t_all = time.time()
    for name in names:
        is_alert = name in rs.ALERT_SCENARIOS
        n = a.alert_seeds if is_alert else a.seeds
        for lt in lights:
            scored = []
            for seed in range(1, n + 1):
                t0 = time.time()
                try:
                    w, det, frames, cost = run_stack(name, seed, lt == "night", track_source=not a.baseline,
                                                     camera_lead=not a.lead_off, radar_plugin=a.radar_plugin)
                    src = "view" if a.baseline else resolve_source(frames, a.source)
                    sc = score_run(w, det, frames, a.baseline, src)
                    res["meta"].setdefault("sources_scored", {}).setdefault(src, 0)
                    res["meta"]["sources_scored"][src] += 1
                except Exception as e:             # noqa: BLE001  (the tree may be mid-edit; the run is reported)
                    import traceback
                    res["errors"].append({"scenario": name, "seed": seed, "light": lt, "error": repr(e),
                                          "trace": traceback.format_exc()[-1500:]})
                    print("  %s/%s seed %d FAILED: %r" % (name, lt, seed, e), file=sys.stderr)
                    continue
                scored.append(sc)
                res["runs"].append({"scenario": name, "light": lt, "seed": seed, "alert": sc["alert"],
                                    "wall_s": round(time.time() - t0, 2), "frames": len(frames)})
                print("  %s/%s seed %d: %d frames, %.1f s" % (name, lt, seed, len(frames), time.time() - t0), file=sys.stderr)
            if not scored:
                continue
            key = "%s/%s" % (name, lt)
            res["scenarios"][key] = aggregate(scored)
            for sc in scored:
                for S in sc["actors"].values():
                    pooled.setdefault("%s/%s" % (S["role"], lt), []).extend(S["errs"])
            if is_alert:
                res["alert"][key] = alert_summary(scored)
    res["errors_by_role"] = {r: err_stats(v) for r, v in sorted(pooled.items()) if v}
    if not a.no_cost:
        try:
            cost = {}
            for ru in (True, False):
                _, _, frames, c = run_stack("parked_street", 1, True, road_users=ru)
                cost[ru] = 1000.0 * c["deliver_s"] / max(1, c["road_frames"])
            res["cost"] = {"scenario": "parked_street/night seed 1", "with_ms": round(cost[True], 3),
                           "without_ms": round(cost[False], 3),
                           "note": "all stack.deliver() wall time (every input message) divided by road frames; one machine, rough"}
        except Exception as e:                     # noqa: BLE001
            res["errors"].append({"cost": repr(e)})
    res["meta"]["wall_s"] = round(time.time() - t_all, 1)
    print_table(res)
    if res["errors"]:
        print("\n%d run(s) failed; see the JSON" % len(res["errors"]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print("\nJSON: %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
