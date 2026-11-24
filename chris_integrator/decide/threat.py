"""The threat measure: for the car against every object, from the braking and steering actually required.

Per object, over its modes (unconditioned probabilities `p0`: the other party does not yield; the conditioned
figure `p` alongside) and the car's own modes, the native core (native/src/decide.c `decide_eval`) gives per pair:

    pc          collision probability over the horizon: the two position distributions and both extents
    a_req       the smallest constant demand that avoids it: braking, steering left / right, both on the friction
                circle (with the build-up of braking and steering)
    btn, stn, tn  threat numbers: required / available (brake, steer, the best of all directions)
    ttb, tts    the last onset of full braking / steering that still avoids it; tm = max(ttb, tts) - the driver's
                reaction time (the configured quantile of a log-normal): the time the driver has left to start
    ttc, thw    time to collision on the mean futures, time headway; sep the least separation
    level       per pair from per-class thresholds: 1 information, 2 caution, 3 warning, 4 imminent (an entry and
                an exit set: the alert logic's hysteresis)

and per object the levels whose probability mass (sum of weight x pc over the pairs at that level or above)
reaches the level's minimum, with the reasoning: which pair (object mode, car mode) and which measure decided.

Available grip: mu x g of the road condition (configured, or from the `env` topic when it says wet / snow),
times the share a driver gets of it (ABS braking, evasive steering), falling slightly with speed; at low speed
the steering is bounded by the car's turning circle (v^2 / R_min)."""
import math

import numpy as np

from ..native import decide as _ND, round
from . import futures as FU

G = 9.81
LEVELS = ("none", "information", "caution", "warning", "imminent")
MEASURES = {1: "threat_number", 2: "time_margin", 3: "collision_course"}
ROAD_MU = {"dry": 0.9, "wet": 0.6, "snow": 0.3, "ice": 0.12}

# entry thresholds per level (information, caution, warning, imminent): a pair is at a level when its threat
# number reaches tn or its time margin falls to tm (s); an object when the probability mass at the level reaches
# mass. Exit: the same scaled (tn x exit_tn, tm + exit_tm, mass x exit_mass): hysteresis.
THRESHOLDS = {
    "vehicle": {"tn": (0.15, 0.30, 0.50, 0.85), "tm": (3.0, 1.5, 0.6, 0.0), "mass": (0.10, 0.25, 0.40, 0.50)},
    "vru": {"tn": (0.12, 0.25, 0.42, 0.80), "tm": (3.5, 2.0, 1.0, 0.2), "mass": (0.08, 0.20, 0.30, 0.40)},
    "unknown": {"tn": (0.15, 0.30, 0.55, 0.90), "tm": (3.0, 1.4, 0.5, 0.0), "mass": (0.15, 0.30, 0.50, 0.60)},
}

DEFAULT_CFG = dict(
    # the car (reference camera = origin): to the front and rear bumper, the body's width
    front_m=2.1, rear_m=2.35, width_m=1.76,
    road="dry", road_from_env=True,
    brake_share=0.85, steer_share=0.7,        # of mu g a driver gets: ABS braking, evasive steering
    speed_fade=0.002,                         # the available grip falls by this share per m/s
    turn_radius_m=5.2,                        # bounds the lateral acceleration at low speed (v^2 / R)
    reaction_median_s=1.0, reaction_sigma_log=0.3, reaction_quantile=0.9,
    brake_buildup_s=0.25, steer_buildup_s=0.2,
    max_offset_m=4.0,                         # an evasion moves the car at most this far beside its path
    k_sigma=1.0,                              # the threat numbers keep clear of the mean +- k sigma ...
    k_sigma_cap_m=(1.0, 0.3),                 # ... at most this much (along, across the object)
    margin_long_m=0.3, margin_lat_m=0.15,
    # a time margin needs a collision to have a time: only where the mean futures make contact (the footprint as it
    # is) is the time left to act measured (against the margin, as the threat numbers); where they miss and only the
    # margin is breached, the pair's level is its threat number's. Against the margin alone a car passed 0.2 m wider
    # than its spread "had" no time left (breached at once, the last full steer before the driver could react)
    tm_needs_contact=True,
    pc_method="mc",                           # "mc": quasi-Monte Carlo over the horizon | "analytic": per time
    samples=256,                              # draws of the Monte Carlo (full quality; cost scales with it)
    sim_dt_s=0.05,                            # the horizon's step (collision tests, evasions)
    prune_sigma=5.0,
    ego_accel_sigma=0.3, ego_lat_sigma=0.15,  # the car's own execution (m/s^2): its place spreads with t^2 / 2 (pc only)
    ego_long_cap_m=3.0, ego_lat_cap_m=0.3,    # ... up to this (1 sigma): a driver keeps the lane
    combo_dirs=3,                             # braking-and-steering directions per side tried on the friction circle
    search_factor=1.6,                        # required demands are searched up to this x the available
    tol_accel=0.05, tol_time=0.02,
    obj_ref="nearest",                        # the futures' x, y: the footprint's nearest point ("nearest") or centre
    others=True,                              # an evasion must keep clear of the other objects too
    p_mode_min=0.02,                          # pairs below this weight are not judged
    horizon_s=6.0,                            # collision probability and course: within this (and the futures')
    horizon_ext_s=4.0,                        # an evasion is followed this much further (objects on their last motion)
    exit_tn=0.8, exit_tm=0.5, exit_mass=0.7,
    thresholds=None,                          # {kind: {"tn", "tm", "mass"}} overrides
    # a standing road user's heading: the camera's box width does not say on which side of the line of sight it
    # points (the radar's rectangle turned the wrong way by 0.2-0.55 rad on bends), the road does: the stated heading
    # snapped to the road's axis at its place when within road_axis_snap_rad of it (else as stated: a bay-parked car)
    road_axis_standing=True, road_axis_snap_rad=0.7, road_axis_std_rad=0.1,
    # ... and the stated heading is evidence only where the radar resolved it (its 1 sigma, `ext_std`, at most
    # hdg_resolved_rad: the boxes of whole vehicles state 0.04-0.11 rad; partial side-on views 0.2-0.8 rad, often a
    # quarter turn off, which turned cars parked along the road across it, into the car's footprint). An unresolved
    # box heading gives way to the direction of travel (resolved speed) or the road's axis at the object's place
    # (either direction, nearer the stated one), whatever the turn; standing or not
    hdg_resolved_rad=0.15,
    # a vehicle the prediction puts in a lane along the road (P(ego) + P(left) + P(right) + P(onc) at least
    # lane_axis_p: more likely than not) points along it: its footprint's heading now is the road's axis at its
    # place (the direction nearer the stated one). A box seen head-on states a heading it cannot see (an oncoming
    # truck's 9.6 m box turned 0.35 rad across the centre line, sigma 0.06, its corner in the car's lane). Without
    # the prediction's lanes, the degraded adapter's judgement stands in: it drives along the path or against it
    # (futures.cv_future, "cv_path")
    lane_axis_p=0.5,
    # the prediction's lane probabilities (contract futures): a mode that stands (moves less than stand_move_m over
    # the horizon) can only be hit where the object is in the car's lane or across it: its weight is scaled by
    # P(lane ego) + P(lane cross), at least lane_weight_floor; not where the radar's own corridor test says it surely
    # stands in the path (flags in_path, threat). Moving modes (pulling out, crossing) keep their weight.
    lane_weight=True, lane_weight_floor=0.05, stand_move_m=1.0,
)


def reaction_time(c):
    z = {0.5: 0.0, 0.8: 0.8416, 0.9: 1.2816, 0.95: 1.6449, 0.99: 2.3263}.get(round(c["reaction_quantile"], 2), 1.2816)
    return c["reaction_median_s"] * math.exp(z * c["reaction_sigma_log"])


def road_from_env(env):
    """The road condition an `env` message says (wet when it rains, snow when it snows), or None."""
    if not isinstance(env, dict):
        return None
    w = env.get("weather")
    lab = None
    if isinstance(w, dict):
        lab = w.get("label", w.get("value"))
        conf = w.get("confidence", 1.0)
        if isinstance(conf, (int, float)) and conf < 0.5:
            lab = None
    elif isinstance(w, str):
        lab = w
    if not isinstance(lab, str):
        return None
    lab = lab.lower()
    if "snow" in lab or "ice" in lab or "frost" in lab:
        return "snow"
    if "rain" in lab or "wet" in lab or "drizzle" in lab:
        return "wet"
    return None


def available(v, c, road=None):
    """(braking, lateral) acceleration a driver can get at speed v on the road condition."""
    mu = ROAD_MU.get(road or c["road"], ROAD_MU["dry"])
    fade = max(0.5, 1.0 - c["speed_fade"] * max(0.0, v))
    along = mu * G * c["brake_share"] * fade
    lat = mu * G * c["steer_share"] * fade
    lat = min(lat, max(v, 0.5) ** 2 / c["turn_radius_m"])
    return along, max(lat, 0.05)


def threshold_table(c):
    th = {k: dict(v) for k, v in THRESHOLDS.items()}
    for k, v in (c.get("thresholds") or {}).items():
        if k in th and isinstance(v, dict):
            th[k].update({q: tuple(v[q]) for q in ("tn", "tm", "mass") if q in v})
    out = np.zeros((len(FU.KIND_SET), 2, 4, 3))
    for kind, s in FU.KIND_SET.items():
        t = th[kind]
        for lv in range(4):
            out[s, 0, lv] = (t["tn"][lv], t["tm"][lv], t["mass"][lv])
            out[s, 1, lv] = (t["tn"][lv] * c["exit_tn"], t["tm"][lv] + c["exit_tm"], t["mass"][lv] * c["exit_mass"])
    return out


# fields of the native results (decide.c DP_*, DO_*)
PF, OF = 24, 18
(DP_PC, DP_TCON, DP_TTC, DP_TTCI, DP_AB, DP_AL, DP_AR, DP_AC, DP_DIR, DP_AVL, DP_AVT, DP_BTN, DP_STN, DP_TN, DP_TTB,
 DP_TTS, DP_TM, DP_LE, DP_LX, DP_SEP, DP_THW, DP_W0, DP_WC, DP_COURSE) = range(PF)
DO_PC, DO_PCC, DO_LE, DO_LX, DO_ME, DO_MX, DO_BEST, DO_WHY, DO_LC, DO_TTC, DO_THW, DO_SEP = 0, 1, 2, 3, 4, 8, 12, 13, 14, 15, 16, 17


class ThreatModel:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.thr = threshold_table(self.cfg).ravel()
        self.t_react = reaction_time(self.cfg)
        self.last_n = 0

    def native_cfg(self, v0, a0, along, lat):
        c = self.cfg
        return np.array([c["front_m"], c["rear_m"], 0.5 * c["width_m"], v0, a0, along, lat, self.t_react,
                         c["brake_buildup_s"], c["steer_buildup_s"], c["sim_dt_s"], c["max_offset_m"], c["k_sigma"],
                         c["margin_long_m"], c["margin_lat_m"], 1.0 if c["pc_method"] == "analytic" else 0.0,
                         float(c["samples"]), c["prune_sigma"], c["ego_accel_sigma"], c["ego_lat_sigma"],
                         float(c["combo_dirs"]), c["search_factor"], c["tol_accel"], c["tol_time"],
                         1.0 if c["obj_ref"] == "nearest" else 0.0, 1.0 if c["others"] else 0.0, c["p_mode_min"],
                         c["horizon_s"], c["horizon_ext_s"], c["ego_long_cap_m"], c["ego_lat_cap_m"],
                         c["k_sigma_cap_m"][0], c["k_sigma_cap_m"][1], 1.0 if c["tm_needs_contact"] else 0.0])

    def evaluate(self, F, ego, road=None):
        """Per object of FrameFutures F: (object, modes, source, native object row, native pair rows (K_obj x E)).
        ego: the radar message's `ego`. Returns (results, frame info)."""
        t = F.t
        N = len(t)
        E = len(F.ego)
        ego_rows = np.empty((E, 1 + 3 * N))
        for j, (_, p, x, y, v) in enumerate(F.ego):
            ego_rows[j, 0] = p
            ego_rows[j, 1:1 + N] = x
            ego_rows[j, 1 + N:1 + 2 * N] = y
            ego_rows[j, 1 + 2 * N:] = v
        M = len(F.objects)
        K = sum(len(m) for _, m, _ in F.objects)
        ohdr = np.zeros((M, 12))
        modes = np.empty((K, 2 + 5 * N))
        k = 0
        ext = []
        for i, (o, ms, _) in enumerate(F.objects):
            L, W, sL, sW, h, sh, label, esrc = FU.extent(o)
            rh = F.road_hdg.get(o.get("id"))
            if math.isfinite(h) and not FU.heading_resolved(sh, self.cfg["hdg_resolved_rad"]):
                if FU.heading_from_motion(o):
                    h, sh = FU.travel_heading(o)
                elif rh is not None:
                    h, sh = FU.road_axis(h, math.nan, rh, math.pi, self.cfg["road_axis_std_rad"])
            elif not math.isfinite(h) and rh is not None:
                h = F.road_hdg[o.get("id")]
            elif rh is not None and FU.kind_of(label) == "vehicle" and (
                    FU.along_lane_p(F.lane.get(o.get("id"))) >= self.cfg["lane_axis_p"] or o.get("id") in F.along):
                h, sh = FU.road_axis(h, sh, rh, math.pi, self.cfg["road_axis_std_rad"])
            elif rh is not None and self.cfg["road_axis_standing"] and FU.stands(o) and not FU.heading_from_motion(o):
                h, sh = FU.road_axis(h, sh, rh, self.cfg["road_axis_snap_rad"], self.cfg["road_axis_std_rad"])
            ext.append((L, W, label, esrc))
            w_stand = FU.in_path_weight(o, F.lane.get(o.get("id")), self.cfg["lane_weight_floor"]) \
                if self.cfg["lane_weight"] else 1.0
            ohdr[i, :9] = (k, len(ms), L, W, sL, sW, h, sh, FU.KIND_SET[FU.kind_of(label)])
            off = FU.centre_offset(o)
            if off is not None:          # the radar says where the footprint's centre is (decide.c: has_ctr)
                ohdr[i, 9:] = (1.0, off[0], off[1])
            for (_, p0, p, x, y, sx, sy, r) in ms:
                row = modes[k]
                f = w_stand if w_stand < 1.0 and FU.mode_stands(x, y, self.cfg["stand_move_m"]) else 1.0
                row[0], row[1] = p0 * f, p * f
                row[2:2 + N] = x
                row[2 + N:2 + 2 * N] = y
                row[2 + 2 * N:2 + 3 * N] = sx
                row[2 + 3 * N:2 + 4 * N] = sy
                row[2 + 4 * N:] = r
                k += 1
        v0 = max(0.0, float(ego.get("speed_mps") or 0.0))
        a0 = float(ego.get("accel_mps2") or 0.0)
        along, lat = available(v0, self.cfg, road)
        pair = np.empty((K * E, PF))
        obj = np.empty((M, OF))
        if M:
            n, _ = _ND.decide_eval(np.ascontiguousarray(t), ego_rows, ohdr, modes,
                                   self.native_cfg(v0, a0, along, lat), self.thr, pair, obj)
            self.last_n = n
        res = []
        k = 0
        for i, (o, ms, src) in enumerate(F.objects):
            nk = len(ms)
            res.append((o, ms, src, obj[i], pair[k * E:(k + nk) * E].reshape(nk, E, PF), ext[i]))
            k += nk
        return res, {"a_avail_mps2": (along, lat), "t_react_s": self.t_react, "road": road or self.cfg["road"]}


def _r(v, n=2):
    return None if v is None or not math.isfinite(v) else round(float(v), n)


def object_record(o, ms, src, orow, prow, ext, F):
    """The message's dict of one judged object (decide.objects[])."""
    E = prow.shape[1]
    lv = int(orow[DO_LE])
    rec = {"id": o.get("id"), "level": lv, "level_name": LEVELS[lv], "level_exit": int(orow[DO_LX]),
           "level_cond": int(orow[DO_LC]), "p_coll": _r(orow[DO_PC], 3), "p_coll_cond": _r(orow[DO_PCC], 3),
           "ttc_s": _r(orow[DO_TTC]), "thw_s": _r(orow[DO_THW]), "sep_m": _r(orow[DO_SEP]),
           "kind": FU.kind_of(ext[2]), "size": [_r(ext[0], 1), _r(ext[1], 1)], "size_src": ext[3], "fut": src,
           "mass": [_r(v, 3) for v in orow[DO_ME:DO_ME + 4]]}
    b = int(orow[DO_BEST])
    if b >= 0:
        km, j = divmod(b, E)
        p = prow[km, j]
        rec.update({"btn": _r(p[DP_BTN]), "stn": _r(p[DP_STN]), "tn": _r(p[DP_TN]),
                    "a_req": {"brake": _r(p[DP_AB]), "left": _r(p[DP_AL]), "right": _r(p[DP_AR]),
                              "comb": _r(p[DP_AC]), "comb_dir_deg": _r(math.degrees(p[DP_DIR]), 0)},
                    "ttb_s": _r(p[DP_TTB]), "tts_s": _r(p[DP_TTS]), "tm_s": _r(p[DP_TM]),
                    "why": {"mode": ms[km][0], "p0": _r(ms[km][1], 3), "p": _r(ms[km][2], 3),
                            "ego_mode": F.ego[j][0], "measure": MEASURES.get(int(orow[DO_WHY]), "collision_course"),
                            "pc": _r(p[DP_PC], 3), "t_contact_s": _r(p[DP_TCON])}})
    else:
        rec.update({"btn": 0.0, "stn": 0.0, "tn": 0.0, "a_req": None, "ttb_s": None, "tts_s": None, "tm_s": None,
                    "why": None})
    if o.get("id") in F.lane:
        rec["lane"] = F.lane[o.get("id")]
    return rec


def pairs_record(ms, prow, F):
    """Every judged pair (for the explain view and the tools): [{mode, ego_mode, p0, p, pc, tn, tm, ttc, level}]."""
    out = []
    for km, m in enumerate(ms):
        for j in range(prow.shape[1]):
            p = prow[km, j]
            if not (p[DP_W0] > 0.0 or p[DP_WC] > 0.0):
                continue
            out.append({"mode": m[0], "ego_mode": F.ego[j][0], "w0": _r(p[DP_W0], 3), "w": _r(p[DP_WC], 3),
                        "pc": _r(p[DP_PC], 3), "course": bool(p[DP_COURSE] > 0.5), "tn": _r(p[DP_TN]),
                        "btn": _r(p[DP_BTN]), "stn": _r(p[DP_STN]), "tm_s": _r(p[DP_TM]), "ttc_s": _r(p[DP_TTC]),
                        "level": int(p[DP_LE]) if math.isfinite(p[DP_LE]) else 0})
    return out
