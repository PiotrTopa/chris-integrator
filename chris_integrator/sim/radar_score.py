"""The radar component (chris_integrator/radar) ALONE on a simulated scenario, scored against the truth: the
tracker-core judge (association, occlusion, rows/queues/platoons, extent). The whole-stack judge is
tools/visual_radar_eval.py; this one feeds the `Radar` box directly, as radar/replay.py feeds it a recording.

    r = run("occlusion", seed=1, night=False, radar="extended")   # frames with the truth attached
    s = score(r)                                                  # the metrics (see score())
    print_table(s)

    python3 -m chris_integrator.sim.radar_score NAME [--seed N] [--night] [--radar point|extended] [--hz 20]
                                                     [--json OUT] [--ego feed|estimator] [--vehicle JSON]
                                                     [--estimator JSON] [--phantom]

`--ego estimator`: the car's own motion goes through the estimator (core.Integrator, its configuration `--estimator`)
and the radar gets its `ego_state` per road frame, as in the stack, instead of the feed's raw `ego`. `--vehicle`:
parameters of the simulated vehicle feed (sim.VehicleSim: gyro_bias, gyro_lag, stamp_lag, speed_floor, ...).
`--phantom`: the velocity the radar gives to standing actors (`phantom()`): what an error of the car's own motion
shows as (docs/EGO_MOTION.md).

Inputs fed: the detector's calibration (`on_calib`), the car's own motion (sim.VehicleSim `ego` -> `on_ego`),
the camera class detector (roaduser_sim.DetectorSim `lights` -> core.roadusers.from_lights -> `on_road_users`),
the microwave radar if asked (sim.radar.RadarSim, point or extended model -> `on_objects`), the road model's
plan (sim.VisionSim `road` -> `on_road`). `frame(ts)` is called at hz_out on the simulation's arrival clock
(every input that has arrived by ts is in).

Output fields read when present (absent ones are tolerated and give None): id, state, exist, x_m, y_m, vx_mps,
vy_mps, motion, flags, and the later ones len, wid, hdg (m, m, rad, car frame, left positive), ext_std, vs
[vx, vy], vis, occ (null | "object" | "edge" | "fov"), occ_by, grp, grp_kind ("row" | "queue" | "platoon").
"""
import argparse
import json
import math
import statistics
import sys
import time

import numpy as np

from ..core.roadusers import PROJ_CFG, from_lights
from ..radar.box import Radar
from . import roaduser_sim as rs
from .radar import attach
from .sensors import VehicleSim, VisionSim

NS = 1_000_000_000
KEEP = ("id", "state", "exist", "x_m", "y_m", "vx_mps", "vy_mps", "motion", "flags", "len", "wid", "hdg", "ext_std",
        "vs", "vis", "occ", "occ_by", "grp", "grp_kind")
LATER_FIELDS = ("len", "wid", "hdg", "ext_std", "vs", "vis", "occ", "occ_by", "grp", "grp_kind")
REACQ_S = 1.0                    # the same id matched again within this after re-appearance
PRE_S = 0.5                      # an interval counts as "tracked before" if the actor was matched this recently


# ---- one run ------------------------------------------------------------------------------------------------
def run(name, seed=1, night=False, radar=None, cfg=None, hz_out=20.0, det_params=None, radar_kw=None, ego="feed",
        vehicle_kw=None, integ_cfg=None):
    """A scenario through the Radar component alone. radar: None (camera boxes only), "point" or "extended".
    ego: "feed" (the feed's raw `ego` into the radar) or "estimator" (through core.Integrator, configuration
    `integ_cfg`: the radar gets `ego_state` per road frame, as in the stack). vehicle_kw: sim.VehicleSim parameters.
    Returns {name, seed, night, radar, hz_out, events, groups, frames: [{t (s from start), dt, objects, actors,
    occ, radar_fov, ego_w, ego_al (the car's true yaw rate and yaw acceleration)}], first_det, wall_s, duration_s}."""
    t_wall = time.time()
    w = rs.scenario(name, seed=seed, night=night)
    det = rs.DetectorSim(seed=1000 + seed, night=night, params=det_params)
    sensors = [VehicleSim(seed=11 + 97 * seed, **(vehicle_kw or {})), VisionSim(seed=21 + 97 * seed, leads=False), det]
    integ = None
    if ego == "estimator":
        from ..core.integrator import Integrator
        integ = Integrator(dict(integ_cfg or {}), objects=False)
    rsim = None
    if radar:
        rsim = attach(w, radar, **(radar_kw or {}))
        sensors.append(rsim)
    rd = Radar(cfg)
    pcfg = dict(PROJ_CFG, sensor="cam")
    calib = None
    frames = []
    step = 1.0 / float(hz_out)
    t_next = w.t0 + step

    def emit(T):
        out = rd.frame(int(round(T * NS)))
        if out is None:
            return
        acts = [a for a in w.actors_at(T) if a["ref_x"] > -10.0]
        occ = det.occlusion(w, T, acts)
        fov = {}
        if rsim is not None:
            mx, my, myaw = rsim.mount
            for a in acts:
                r = math.hypot(a["x"] - mx, a["y"] - my)
                az = (math.atan2(a["y"] - my, a["x"] - mx) - myaw + math.pi) % (2 * math.pi) - math.pi
                fov[a["id"]] = 1.0 < r <= rsim.max_range and abs(az) <= math.radians(rsim.fov_deg)
        objs = [{k: o[k] for k in KEEP if k in o} for o in out.get("objects") or ()]
        w0, w1 = w.truth_at(T - 0.1)["w"], w.truth_at(T + 0.1)["w"]
        frames.append({"t": T - w.t0, "objects": objs, "actors": acts, "occ": occ, "radar_fov": fov,
                       "ego_w": w.truth_at(T)["w"], "ego_al": (w1 - w0) / 0.2})

    for t_arr, m in w.messages(sensors):
        while t_next <= t_arr:
            emit(t_next)
            t_next += step
        typ = m.get("type")
        if calib is None:
            calib = det.calib_msg(m["ts_mono_ns"])
            rd.on_calib(calib)
        if typ == "ego":
            if integ is None:
                rd.on_ego(m)
            else:
                integ.on_message(m)
        elif typ == "lights":
            ru = from_lights(m, calib, "cam", pcfg)
            if ru is not None:
                rd.on_road_users(ru)
        elif typ == "objects":
            rd.on_objects(m)
        elif typ == "road":
            if integ is not None:
                es = integ.on_road(m)
                if es is not None:
                    rd.on_ego_state(es)
            rd.on_road(m)
    while t_next <= w.t0 + w.T:
        emit(t_next)
        t_next += step
    for k, f in enumerate(frames):
        f["dt"] = min(0.2, frames[k + 1]["t"] - f["t"]) if k + 1 < len(frames) else step
    first_det = {}
    for t, _, logd in det.log:
        for d in logd:
            if not str(d[0]).startswith("fp"):
                first_det.setdefault(d[0], t - w.t0)
    return {"name": name, "seed": seed, "night": night, "radar": radar, "hz_out": hz_out,
            "events": list(w.spec.get("events") or ()), "groups": dict(w.spec.get("groups") or {}),
            "frames": frames, "first_det": first_det, "duration_s": w.T, "wall_s": round(time.time() - t_wall, 2),
            "boxes": sum(len(lg) for _, _, lg in det.log), "messages": len(det.log)}


# ---- matching (tools/visual_radar_eval.py `_gate` / `match`, copied: tools/ is not a package) ---------------
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


# ---- the velocity given to what stands ------------------------------------------------------------------------
def phantom(r, min_turn=0.05, min_al=0.05):
    """The velocity over ground the radar gives to standing actors (confirmed objects matched to them; the truth
    is zero): what an error of the car's own motion shows as. Per condition {n, lat_rms, lat_p90, lon_rms, lon_p90}
    (m/s; lateral and along the car's axis): `all`, `turning` (the car's yaw rate beyond min_turn rad/s),
    `changing` (its yaw acceleration beyond min_al rad/s2: turning in or out, where a time offset of the yaw rate
    shows), `straight`."""
    prev, rows = None, []
    for f in r["frames"]:
        outs = _outs(f)
        m = match(outs, f["actors"], prev)
        prev = {outs[i]["oid"]: a["id"] for i, (a, _, _) in m.items()}
        for i, (a, _, _) in m.items():
            o = outs[i]
            if a["moving"] or o["state"] != "confirmed" or o["vx"] is None or o["vy"] is None:
                continue
            rows.append((abs(f["ego_w"]), abs(f["ego_al"]), o["vx"], o["vy"]))

    def stat(sel):
        v = [x for x in rows if sel(x)]
        if not v:
            return {"n": 0, "lat_rms": None, "lat_p90": None, "lon_rms": None, "lon_p90": None}
        lat, lon = sorted(abs(x[3]) for x in v), sorted(abs(x[2]) for x in v)
        return {"n": len(v), "lat_rms": round(math.sqrt(sum(x * x for x in lat) / len(v)), 3), "lat_p90": round(lat[int(0.9 * len(v))], 3),
                "lon_rms": round(math.sqrt(sum(x * x for x in lon) / len(v)), 3), "lon_p90": round(lon[int(0.9 * len(v))], 3)}
    return {"all": stat(lambda x: True), "turning": stat(lambda x: x[0] > min_turn), "changing": stat(lambda x: x[1] > min_al),
            "straight": stat(lambda x: x[0] <= 0.01)}


# ---- scoring ------------------------------------------------------------------------------------------------
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _outs(f):
    out = []
    for o in f["objects"]:
        if not (_num(o.get("x_m")) and _num(o.get("y_m"))):
            continue
        d = dict(o)
        d.update(oid=o.get("id"), state=o.get("state") or "confirmed", x=float(o["x_m"]), y=float(o["y_m"]),
                 vx=float(o["vx_mps"]) if _num(o.get("vx_mps")) else None,
                 vy=float(o["vy_mps"]) if _num(o.get("vy_mps")) else None,
                 motion=o.get("motion") if o.get("motion") in ("moving", "stationary", "unknown") else "unknown")
        out.append(d)
    return out


def _wrap(a, period):
    return (a + period / 2) % period - period / 2


def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _share(a, b):
    return _r(a / b) if b > 0 else None


OCC_KINDS = ("object", "fov")


def _new_actor(a):
    return {"role": a["role"], "kind": a["kind"], "group": None, "present_s": 0.0, "det_s": 0.0, "conf_s": 0.0,
            "held_s": 0.0, "first_vis": None, "first_conf": None, "last_id": None, "last_match_t": None,
            "switches": 0, "segments": 0, "ids": [], "errs": [], "motion": {}, "stat_truth_s": 0.0,
            "stat_called_s": 0.0, "mov_truth_s": 0.0, "mov_called_stat_s": 0.0, "ext": [], "occ_open": {},
            "intervals": [], "occ_flag": [0.0, 0.0], "last_err": None}


def score(r):
    """Metrics of one run() (see the module docstring for the outputs read). Returns
    {scenario, seed, night, radar, duration_s, wall_s, frames, fields_present, events, actors {id: stats + role,
    kind, group, ids}, roles {role: stats}, all: stats, groups {gid: {kind, member_s, matched_s, membership,
    kind_ok}}, group_membership, group_false, false_tracks, false_tracks_per_min, false_tracks_dup_per_min (a
    false track sitting on a real object: a duplicate / a rival of its matched one), false_tracks_phantom_per_min,
    unmatched_confirmed_s_per_min, detector}.
    False track (as tools/visual_radar_eval.py): a confirmed output unmatched for >= half its confirmed time and
    >= 0.3 s. Matched shares are over the actor's det_s: camera-detectable (size and region) and not hidden by a
    nearer object, or inside the radar's field when a radar is fed.
    stats: n_actors, det_s, conf_share, held_share, id_switches, id_switches_per_actor, fragments, ttfc_s,
    pos_err {along_bias, along_abs_p50, along_abs_p90, across_abs_p50, across_abs_p90, n}, vel_err {p50, p90},
    motion_s {truth: {output: s}}, stationary_called_stationary, moving_called_stationary,
    occlusion {object|fov: {intervals, tracked_before, occ_s, held_share, alive_share, exist_min, err_growth_p50,
    err_growth_max, reacq_ended, reacq_same_id, reacq_any}}, extent {n, len_err_p50, len_err_p90, wid_err_p50,
    wid_err_p90, hdg_err_pi_p50, hdg_err_pi_p90, hdg_err_2pi_p50 (moving actors)}, occ_flag_agree."""
    frames = r["frames"]
    A = {}
    trk = {}
    present = {k: False for k in LATER_FIELDS}
    gstat = {}                                   # truth group -> [member matched s, member grouped-with-member s, kind ok s]
    gfalse = [0.0, 0.0]                          # non-member matched s, of which with a grp
    prev = {}
    for f in frames:
        t, dt = f["t"], f["dt"]
        actors = f["actors"]
        outs = _outs(f)
        for o in outs:
            for k in LATER_FIELDS:
                if o.get(k) is not None:
                    present[k] = True
        M = match(outs, actors, prev)
        prev = {outs[i]["oid"]: a["id"] for i, (a, _, _) in M.items()}
        by_actor = {a["id"]: (i, al, ac) for i, (a, al, ac) in M.items()}
        by_oid = {o["oid"]: o for o in outs}
        occ = f["occ"]
        # groups: which output group each matched member carries
        gmem = {}
        for a in actors:
            m = by_actor.get(a["id"])
            g = a.get("group")
            if g is not None:
                gmem.setdefault(g, []).append((a, outs[m[0]] if m else None))
            elif m is not None:
                gfalse[0] += dt
                if outs[m[0]].get("grp") is not None:
                    gfalse[1] += dt
        for g, mem in gmem.items():
            st = gstat.setdefault(g, {"kind": mem[0][0].get("group_kind"), "member_s": 0.0, "matched_s": 0.0,
                                      "grouped_s": 0.0, "kind_ok_s": 0.0})
            for a, o in mem:
                st["member_s"] += dt
                if o is None:
                    continue
                st["matched_s"] += dt
                gid = o.get("grp")
                if gid is None:
                    continue
                if any(o2 is not None and o2 is not o and o2.get("grp") == gid for _, o2 in mem):
                    st["grouped_s"] += dt
                    if o.get("grp_kind") == st["kind"]:
                        st["kind_ok_s"] += dt
        for a in actors:
            aid = a["id"]
            S = A.get(aid)
            if S is None:
                S = A[aid] = _new_actor(a)
            if a.get("group") is not None:
                S["group"] = a["group"]
            S["present_s"] += dt
            oc = occ.get(aid) or {}
            cam_ok = bool(oc.get("detectable")) and oc.get("occ") != "object"
            ok = cam_ok or bool(f["radar_fov"].get(aid))
            if ok:
                S["det_s"] += dt
                if S["first_vis"] is None:
                    S["first_vis"] = t
            m = by_actor.get(aid)
            o = outs[m[0]] if m else None
            # occlusion intervals (truth: the camera's) -----------------------------------------------------
            kind_now = oc.get("occ") if oc.get("occ") in OCC_KINDS else None
            for kd in OCC_KINDS:
                iv = S["occ_open"].get(kd)
                if kind_now == kd:
                    if iv is not None and iv["end"] is not None and not iv["reacq"]:
                        iv["end"] = None                         # hidden again before it was re-acquired: the same one
                    elif iv is None or iv["end"] is not None:
                        if iv is not None:                       # re-acquired, then hidden again: a new interval
                            iv["reacq"], iv["reacq_any"] = True, True
                            S["intervals"].append(iv)
                        pre = S["last_id"] if S["last_match_t"] is not None and t - S["last_match_t"] <= PRE_S else None
                        iv = S["occ_open"][kd] = {"kind": kd, "t0": t, "end": None, "pre_id": pre, "occ_s": 0.0,
                                                  "held_s": 0.0, "alive_s": 0.0, "exist_min": None,
                                                  "err0": S["last_err"], "err_max": None, "reacq": None,
                                                  "reacq_any": None}
                    iv["occ_s"] += dt
                    pid = iv["pre_id"]
                    if pid is not None:
                        if o is not None and o["oid"] == pid:
                            iv["held_s"] += dt
                        po = by_oid.get(pid)
                        if po is not None:
                            iv["alive_s"] += dt
                            if _num(po.get("exist")):
                                iv["exist_min"] = po["exist"] if iv["exist_min"] is None else min(iv["exist_min"], po["exist"])
                            _, al, ac = _gate(a, po["x"], po["y"])
                            e = math.hypot(al, ac)
                            iv["err_max"] = e if iv["err_max"] is None else max(iv["err_max"], e)
                elif iv is not None:
                    if iv["end"] is None:
                        iv["end"] = t
                    if iv["reacq"] is None and o is not None and t - iv["end"] <= REACQ_S:
                        iv["reacq_any"] = True
                        if iv["pre_id"] is not None and o["oid"] == iv["pre_id"]:
                            iv["reacq"] = True
                    if t - iv["end"] > REACQ_S:
                        iv["reacq"] = bool(iv["reacq"])
                        iv["reacq_any"] = bool(iv["reacq_any"])
                        S["intervals"].append(iv)
                        del S["occ_open"][kd]
            # the occ flag against the truth's reason
            if o is not None and "occ" in o:
                S["occ_flag"][1] += dt
                if o.get("occ") == oc.get("occ"):
                    S["occ_flag"][0] += dt
            if m is None:
                continue
            i, al, ac = m
            S["last_err"] = math.hypot(al, ac)
            if S["last_match_t"] is None or t - S["last_match_t"] > 1.0:
                S["segments"] += 1
            S["last_match_t"] = t
            if S["last_id"] is not None and o["oid"] != S["last_id"]:
                S["switches"] += 1
            S["last_id"] = o["oid"]
            if o["oid"] not in S["ids"]:
                S["ids"].append(o["oid"])
            if ok:
                S["held_s"] += dt
            if o["state"] != "confirmed":
                continue
            if ok:
                S["conf_s"] += dt
            if S["first_conf"] is None:
                S["first_conf"] = t
            tm = "moving" if a["moving"] else "stationary"
            S["motion"].setdefault(tm, {}).setdefault(o["motion"], 0.0)
            S["motion"][tm][o["motion"]] += dt
            if a["moving"]:
                S["mov_truth_s"] += dt
                if o["motion"] == "stationary":
                    S["mov_called_stat_s"] += dt
            else:
                S["stat_truth_s"] += dt
                if o["motion"] == "stationary":
                    S["stat_called_s"] += dt
            dv = None
            vx, vy = o["vx"], o["vy"]
            if vx is not None and vy is not None:
                dv = math.hypot(vx - a["vx"], vy - a["vy"])
            S["errs"].append((a["range"], al, ac, dv))
            ln, wd, hd = o.get("len"), o.get("wid"), o.get("hdg")
            if _num(ln) or _num(wd) or _num(hd):
                S["ext"].append((abs(ln - a["len"]) if _num(ln) else None, abs(wd - a["wid"]) if _num(wd) else None,
                                 abs(_wrap(hd - a["yaw"], math.pi)) if _num(hd) else None,
                                 abs(_wrap(hd - a["yaw"], 2 * math.pi)) if _num(hd) and a["moving"] else None))
        # false tracks
        for i, o in enumerate(outs):
            if o["state"] != "confirmed":
                continue
            R = trk.setdefault(o["oid"], {"conf_s": 0.0, "free_s": 0.0, "near_s": 0.0})
            R["conf_s"] += dt
            if i not in M:
                R["free_s"] += dt
                if any(_gate(a, o["x"], o["y"])[0] <= 1.0 for a in actors):
                    R["near_s"] += dt          # a second object on something already matched (or a rival)
    for S in A.values():                         # intervals still open at the end
        for iv in S["occ_open"].values():
            if iv["end"] is not None:
                iv["reacq"], iv["reacq_any"] = bool(iv["reacq"]), bool(iv["reacq_any"])
            S["intervals"].append(iv)
        S["occ_open"] = {}
    dur = (frames[-1]["t"] - frames[0]["t"] + frames[-1]["dt"]) if frames else 0.0
    mins = dur / 60.0 if dur else 1.0
    false = [R for R in trk.values() if R["conf_s"] > 0 and R["free_s"] >= 0.5 * R["conf_s"] and R["free_s"] >= 0.3]
    false_n = len(false)
    false_dup = sum(1 for R in false if R["near_s"] >= 0.5 * R["free_s"])
    free_s = sum(R["free_s"] for R in trk.values())
    actors_out = {aid: _stats([S], r["first_det"]) for aid, S in A.items() if S["det_s"] > 0 or S["held_s"] > 0}
    for aid, st in actors_out.items():
        st.update(role=A[aid]["role"], kind=A[aid]["kind"], group=A[aid]["group"], ids=len(A[aid]["ids"]))
    roles = {}
    for S in A.values():
        if S["det_s"] > 0 or S["held_s"] > 0:
            roles.setdefault(S["role"], []).append(S)
    groups = {g: {"kind": st["kind"], "member_s": _r(st["member_s"], 1), "matched_s": _r(st["matched_s"], 1),
                  "membership": _share(st["grouped_s"], st["matched_s"]) if present["grp"] else None,
                  "kind_ok": _share(st["kind_ok_s"], st["grouped_s"]) if present["grp_kind"] else None}
              for g, st in gstat.items()}
    gm = [st for st in gstat.values()]
    return {"scenario": r["name"], "seed": r["seed"], "night": r["night"], "radar": r["radar"],
            "duration_s": _r(dur, 1), "wall_s": r.get("wall_s"), "frames": len(frames),
            "fields_present": present, "events": r.get("events"),
            "actors": actors_out,
            "roles": {role: _stats(Ss, r["first_det"]) for role, Ss in sorted(roles.items())},
            "all": _stats([S for S in A.values() if S["det_s"] > 0 or S["held_s"] > 0], r["first_det"]),
            "groups": groups,
            "group_membership": _share(sum(s["grouped_s"] for s in gm), sum(s["matched_s"] for s in gm))
            if present["grp"] and gm else None,
            "group_false": _share(gfalse[1], gfalse[0]) if present["grp"] else None,
            "false_tracks": false_n, "false_tracks_per_min": _r(false_n / mins, 2),
            "false_tracks_dup_per_min": _r(false_dup / mins, 2),
            "false_tracks_phantom_per_min": _r((false_n - false_dup) / mins, 2),
            "unmatched_confirmed_s_per_min": _r(free_s / mins, 2),
            "detector": {"messages": r.get("messages"), "boxes": r.get("boxes")}}


def _stats(Ss, first_det):
    """Pooled metrics of actor accumulators Ss (one actor or a role)."""
    det_s = sum(S["det_s"] for S in Ss)
    errs = [e for S in Ss for e in S["errs"]]
    ttfc = []
    for S in Ss:
        t0 = S["first_vis"]
        if t0 is not None and S["first_conf"] is not None and S["first_conf"] >= t0 - 1.0:
            ttfc.append(S["first_conf"] - t0)
    mot = {}
    for S in Ss:
        for tm, d in S["motion"].items():
            for om, v in d.items():
                mot.setdefault(tm, {}).setdefault(om, 0.0)
                mot[tm][om] += v
    mot = {tm: {om: round(v, 2) for om, v in d.items()} for tm, d in mot.items()}
    occ = {}
    for kd in OCC_KINDS:
        ivs = [iv for S in Ss for iv in S["intervals"] if iv["kind"] == kd]
        if kd == "fov":                      # only the ones it came back from (a passed parked car is not one)
            ivs = [iv for iv in ivs if iv["end"] is not None]
        tb = [iv for iv in ivs if iv["pre_id"] is not None]
        ended = [iv for iv in tb if iv["end"] is not None and iv["reacq"] is not None]
        occ_s = sum(iv["occ_s"] for iv in tb)
        growth = [iv["err_max"] - iv["err0"] for iv in tb if iv["err_max"] is not None and iv["err0"] is not None]
        ex = [iv["exist_min"] for iv in tb if iv["exist_min"] is not None]
        occ[kd] = {"intervals": len(ivs), "tracked_before": len(tb), "occ_s": _r(occ_s, 2),
                   "held_share": _share(sum(iv["held_s"] for iv in tb), occ_s),
                   "alive_share": _share(sum(iv["alive_s"] for iv in tb), occ_s),
                   "exist_min": _r(min(ex)) if ex else None,
                   "err_growth_p50": _r(_pct(growth, 0.5), 2), "err_growth_max": _r(max(growth), 2) if growth else None,
                   "reacq_ended": len(ended),
                   "reacq_same_id": _share(sum(1 for iv in ended if iv["reacq"]), len(ended)),
                   "reacq_any": _share(sum(1 for iv in ended if iv["reacq_any"]), len(ended))}
    ext = [e for S in Ss for e in S["ext"]]

    def col(k, q):
        return _r(_pct([e[k] for e in ext], q), 3) if ext else None
    stat_t = sum(S["stat_truth_s"] for S in Ss)
    mov_t = sum(S["mov_truth_s"] for S in Ss)
    of = [sum(S["occ_flag"][0] for S in Ss), sum(S["occ_flag"][1] for S in Ss)]
    sw = sum(S["switches"] for S in Ss)
    al = [e[1] for e in errs]
    ac = [abs(e[2]) for e in errs]
    dv = [e[3] for e in errs if e[3] is not None]
    return {
        "n_actors": len(Ss), "det_s": _r(det_s, 1),
        "conf_share": _share(sum(S["conf_s"] for S in Ss), det_s),
        "held_share": _share(sum(S["held_s"] for S in Ss), det_s),
        "id_switches": sw, "id_switches_per_actor": _r(sw / len(Ss)) if Ss else None,
        "fragments": sum(S["segments"] for S in Ss),
        "ttfc_s": _r(_pct(ttfc, 0.5), 2), "ttfc_p90_s": _r(_pct(ttfc, 0.9), 2), "ttfc_n": len(ttfc),
        "never_confirmed": sum(1 for S in Ss if S["first_conf"] is None),
        "pos_err": {"n": len(errs), "along_bias": _r(statistics.fmean(al), 2) if al else None,
                    "along_abs_p50": _r(_pct([abs(x) for x in al], 0.5), 2),
                    "along_abs_p90": _r(_pct([abs(x) for x in al], 0.9), 2),
                    "across_abs_p50": _r(_pct(ac, 0.5), 2), "across_abs_p90": _r(_pct(ac, 0.9), 2)},
        "vel_err": {"p50": _r(_pct(dv, 0.5), 2), "p90": _r(_pct(dv, 0.9), 2)},
        "motion_s": mot,
        "stationary_called_stationary": _share(sum(S["stat_called_s"] for S in Ss), stat_t),
        "moving_called_stationary": _share(sum(S["mov_called_stat_s"] for S in Ss), mov_t),
        "occlusion": occ,
        "extent": {"n": len(ext), "len_err_p50": col(0, 0.5), "len_err_p90": col(0, 0.9), "wid_err_p50": col(1, 0.5),
                   "wid_err_p90": col(1, 0.9), "hdg_err_pi_p50": col(2, 0.5), "hdg_err_pi_p90": col(2, 0.9),
                   "hdg_err_2pi_p50": col(3, 0.5)},
        "occ_flag_agree": _share(of[0], of[1]),
    }


# ---- report -------------------------------------------------------------------------------------------------
def _f(x, fmt="%.2f"):
    return "-" if x is None else fmt % x


def print_table(s, file=None):
    file = file or sys.stdout
    p = lambda *a: print(*a, file=file)                                   # noqa: E731
    p("== %s seed %d %s radar=%s: %s s, %d frames, wall %s s; false tracks/min %s (on a real object %s, phantom "
      "%s), unmatched conf s/min %s ==" % (
          s["scenario"], s["seed"], "night" if s["night"] else "day", s["radar"], s["duration_s"], s["frames"],
          s["wall_s"], _f(s["false_tracks_per_min"]), _f(s["false_tracks_dup_per_min"]),
          _f(s["false_tracks_phantom_per_min"]), _f(s["unmatched_confirmed_s_per_min"])))
    hdr = ("role", "n", "det_s", "conf", "held", "sw", "frag", "ttfc", "|al|50", "|ac|50", "|dv|50", "stat->S",
           "mov->S", "occ_iv", "o_held", "o_alive", "reacq", "fov_iv", "f_reacq", "len50", "wid50", "hdg50")
    fmt = "%-12s %3s %6s %5s %5s %3s %4s %5s %6s %6s %6s %7s %6s %6s %6s %7s %5s %6s %7s %5s %5s %5s"
    p(fmt % hdr)
    rows = list(s["roles"].items()) + [("ALL", s["all"])]
    for role, st in rows:
        o, fv, ex = st["occlusion"]["object"], st["occlusion"]["fov"], st["extent"]
        p(fmt % (role[:12], st["n_actors"], _f(st["det_s"], "%.1f"), _f(st["conf_share"]), _f(st["held_share"]),
                 st["id_switches"], st["fragments"], _f(st["ttfc_s"], "%.1f"), _f(st["pos_err"]["along_abs_p50"]),
                 _f(st["pos_err"]["across_abs_p50"]), _f(st["vel_err"]["p50"]), _f(st["stationary_called_stationary"]),
                 _f(st["moving_called_stationary"]), "%d/%d" % (o["tracked_before"], o["intervals"]),
                 _f(o["held_share"]), _f(o["alive_share"]), _f(o["reacq_same_id"]),
                 "%d/%d" % (fv["tracked_before"], fv["intervals"]), _f(fv["reacq_same_id"]), _f(ex["len_err_p50"]),
                 _f(ex["wid_err_p50"]), _f(ex["hdg_err_pi_p50"])))
    if s["groups"]:
        p("groups: " + "; ".join("%s (%s): matched %ss, membership %s, kind %s" % (
            g, d["kind"], d["matched_s"], _f(d["membership"]), _f(d["kind_ok"])) for g, d in s["groups"].items())
          + "; false grouping %s" % _f(s["group_false"]))
    missing = [k for k, v in s["fields_present"].items() if not v]
    if missing:
        p("output fields absent (metrics None): " + ", ".join(missing))


def main(argv=None):
    ap = argparse.ArgumentParser(description="The radar component alone on a simulated scenario, scored.")
    ap.add_argument("name", help="a scenario of roaduser_sim.SCENARIOS (or ALERT_SCENARIOS)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--night", action="store_true")
    ap.add_argument("--radar", default=None, choices=("point", "extended"), help="add a microwave radar")
    ap.add_argument("--hz", type=float, default=20.0, help="output frames per second")
    ap.add_argument("--json", default=None, help="write the score here")
    ap.add_argument("--ego", default="feed", choices=("feed", "estimator"), help="the car's own motion: the feed's raw `ego`, "
                    "or through the estimator (`ego_state`, as in the stack)")
    ap.add_argument("--vehicle", default=None, help="sim.VehicleSim parameters as JSON (gyro_bias, gyro_lag, stamp_lag, ...)")
    ap.add_argument("--estimator", default=None, help="the estimator's configuration as JSON (with --ego estimator)")
    ap.add_argument("--phantom", action="store_true", help="the velocity given to standing actors")
    a = ap.parse_args(argv)
    r = run(a.name, seed=a.seed, night=a.night, radar=a.radar, hz_out=a.hz, ego=a.ego,
            vehicle_kw=json.loads(a.vehicle) if a.vehicle else None, integ_cfg=json.loads(a.estimator) if a.estimator else None)
    s = score(r)
    print_table(s)
    if a.phantom:
        ph = phantom(r)
        s["phantom"] = ph
        for k, v in ph.items():
            print("phantom velocity of standing actors, %-9s n %5d  lateral rms %s p90 %s  along rms %s p90 %s m/s" % (
                k, v["n"], _f(v["lat_rms"]), _f(v["lat_p90"]), _f(v["lon_rms"]), _f(v["lon_p90"])))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(s, fh, indent=1, default=str)
        print("JSON: %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
