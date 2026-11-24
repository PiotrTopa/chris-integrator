"""The threat measure: for the car against every object, from the braking and steering actually required.

Per object, over its modes (unconditioned probabilities `p0`: the other party does not yield; the conditioned
figure `p` alongside) and the car's own modes, the native core (native/src/decide.c: `decide_frame` makes the
futures and the footprints of a radar message and writes the records, `decide_eval` inside it judges) gives per
pair:

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
    samples=128,                              # draws of the Monte Carlo (256: half the error, a tenth more of the
                                              # layer's cost; docs/TOPICS.md)
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
    # A standing vehicle the prediction gave two places along its needle (`lane_res`), the one in the car's way
    # with a probability below place_hold_p: it is at ONE of them and does not move, and the car learns which as
    # it closes (the range's spread shrinks with the range, to place_hold_range_rel of it: the tracker states
    # 8-10 % at 15-40 m for a clean track). The two places, d apart on the ground, are told apart
    # (place_hold_sigma spreads between them) once the car is within d / (sigma x rel) of their middle: in t_res.
    # The driver must act, at the latest, when braking at place_hold_decel after the reaction time still stops
    # short of the place in the way: in t_act. While t_res <= t_act the object's entry level is held at
    # place_hold_level (information); once acting cannot wait, the level is what the masses say. Off: measured
    # 2026-11-24 it holds none of the false episodes with comfortable braking (docs/DESIGN.md 17.2)
    place_hold=False, place_hold_decel=3.0, place_hold_range_rel=0.09, place_hold_sigma=2.0, place_hold_p=0.9,
    place_hold_level=1,
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
    """The threat measure of one frame, by the native core (native/src/decide.c `decide_frame`).

        tm = ThreatModel(cfg, future_cfg)               # keys of DEFAULT_CFG, of futures.DEFAULT_CFG
        tm.frame(objects, ego, futures, path, road, shown, opt)       # the plug-in's call: the native tuple
        tm.explain(radar_msg, futures, path)            # the same frame with its arrays (tests, tools)"""

    def __init__(self, cfg=None, future_cfg=None, list_min_p=0.01):
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.fcfg = dict(FU.DEFAULT_CFG, **(future_cfg or {}))
        self.thr = threshold_table(self.cfg).ravel()
        self.t_react = reaction_time(self.cfg)
        self.last_n = 0
        self._fc = FU.cfg_array(self.fcfg)
        self._t_own = FU.own_grid(self.fcfg)
        self._tc = self.native_cfg(list_min_p)

    def native_cfg(self, list_min_p=0.01):
        """The model's constants as the native core reads them (decide.c DC_*, DX_*). The car's speed and
        acceleration and the accelerations available are the frame's: the native core sets them."""
        c = self.cfg
        return np.array([c["front_m"], c["rear_m"], 0.5 * c["width_m"], 0.0, 0.0, 0.0, 0.0, self.t_react,
                         c["brake_buildup_s"], c["steer_buildup_s"], c["sim_dt_s"], c["max_offset_m"], c["k_sigma"],
                         c["margin_long_m"], c["margin_lat_m"], 1.0 if c["pc_method"] == "analytic" else 0.0,
                         float(c["samples"]), c["prune_sigma"], c["ego_accel_sigma"], c["ego_lat_sigma"],
                         float(c["combo_dirs"]), c["search_factor"], c["tol_accel"], c["tol_time"],
                         1.0 if c["obj_ref"] == "nearest" else 0.0, 1.0 if c["others"] else 0.0, c["p_mode_min"],
                         c["horizon_s"], c["horizon_ext_s"], c["ego_long_cap_m"], c["ego_lat_cap_m"],
                         c["k_sigma_cap_m"][0], c["k_sigma_cap_m"][1], 1.0 if c["tm_needs_contact"] else 0.0,
                         # beyond the evaluation's (DX_*): the grip, the footprint's heading, the lanes' weight
                         c["brake_share"], c["steer_share"], c["speed_fade"], c["turn_radius_m"], c["hdg_resolved_rad"],
                         1.0 if c["road_axis_standing"] else 0.0, c["road_axis_snap_rad"], c["road_axis_std_rad"],
                         c["lane_axis_p"], 1.0 if c["lane_weight"] else 0.0, c["lane_weight_floor"], c["stand_move_m"],
                         list_min_p,
                         # (DX_HOLD..): a standing vehicle of two places: the level held while the range can still resolve
                         1.0 if c["place_hold"] else 0.0, c["place_hold_decel"], c["place_hold_range_rel"],
                         c["place_hold_sigma"], c["place_hold_p"], float(c["place_hold_level"])], float)

    def frame(self, objects, ego, futures=None, path=None, road=None, shown=None, opt=0):
        """One radar message's decision (decide.c `decide_frame`): objects, ego: the message's; futures: a
        `futures.Futures` or None (the degraded adapters); path: `futures.plan_path`; road: the road condition
        (None: the configured); shown: {id: level shown} of the frame before; opt: 1 every pair in the records,
        2 a record of every judged object and the arrays. Returns the native tuple:
        (records, listed, objects by id, shown, judged, cv, invalid, ego_own, steps, a_long, a_lat, ego_modes,
        horizon, detail)."""
        res = _ND.decide_frame(objects, ego, None if futures is None else futures.fields, path, self._t_own, self._fc,
                               self._tc, self.thr, ROAD_MU.get(road or self.cfg["road"], ROAD_MU["dry"]),
                               {} if shown is None else shown, opt)
        if res[8] >= 0:
            self.last_n = res[8]
        return res

    def explain(self, radar_msg, futures=None, path=None, road=None):
        """A frame with everything the native core made of it (tests, tools): `Explained`. futures: None reads
        the contract's fields out of the message itself."""
        if futures is None:
            futures = FU.from_message(radar_msg)
        objs = radar_msg.get("objects") or []
        return Explained(self.frame(objs if isinstance(objs, list) else list(objs), radar_msg.get("ego"), futures, path,
                                    road, None, 2), self)


class Explained:
    """One frame as the native core judged it:

        records     the record (decide.objects[]) of every judged object, in the radar message's order
        t           (N,) the grid (now first); ego (E, 1 + 3 N): p, x, y, v; ego_names [E]
        modes       (K, 2 + 5 N): p0, p (weighted by the lanes where the mode stands), x, y, sx, sy, rxy; names [K];
                    p (K, 2): p0, p as the futures gave them (normalised)
        ohdr        (M, 12): first mode, modes, len, wid, their 1 sigma, heading, its 1 sigma, threshold set,
                    has_ctr, ctr_dx, ctr_dy (decide.c: the footprint as it is judged)
        pair        (K, E, PF) the pairs' measures (DP_*); obj (M, OF) the objects' (DO_*); src [M]: "cv" | "contract"
        cv, invalid, ego_own, a_avail, t_react, steps"""

    def __init__(self, res, model):
        (self.records, self.listed, self.objects, self.shown, self.judged, self.cv, self.invalid, ego_own, self.steps,
         al, at, self.ego_modes, self.horizon, d) = res
        self.ego_own, self.a_avail, self.t_react = bool(ego_own), (al, at), model.t_react
        if d is None:                          # nothing judged
            self.t = self.ego = self.modes = self.p = self.ohdr = self.pair = self.obj = None
            self.names, self.ego_names, self.src = [], [], []
            return
        self.names, self.ego_names, self.src = d["names"], d["ego_names"], d["src"]
        n, E = len(d["t"]) // 8, len(self.ego_names)
        self.t = np.frombuffer(d["t"])
        self.ego = np.frombuffer(d["ego"]).reshape(E, 1 + 3 * n)
        self.modes = np.frombuffer(d["modes"]).reshape(-1, 2 + 5 * n)
        self.p = np.frombuffer(d["p"]).reshape(-1, 2)
        self.ohdr = np.frombuffer(d["ohdr"]).reshape(-1, 12)
        self.pair = np.frombuffer(d["pair"]).reshape(-1, E, PF)
        self.obj = np.frombuffer(d["obj"]).reshape(-1, OF)

    def pairs(self, i):
        """The pair rows (modes x car's modes x PF) of judged object i."""
        k, nk = int(self.ohdr[i, 0]), int(self.ohdr[i, 1])
        return self.pair[k:k + nk]

    def mode_names(self, i):
        k, nk = int(self.ohdr[i, 0]), int(self.ohdr[i, 1])
        return self.names[k:k + nk]
