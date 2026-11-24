"""What a said alert was, against the simulator's truth: the pure part of tools/alert_truth.py (no stack, no
simulator: rows in, classes and tables out; tests/test_alert_truth.py).

A ROW is one said episode of `obstacle_ahead` (stage >= show_from, not suppressed) as tools/alert_truth.py records
it: the alert's object matched to the truth actor nearest its line of sight (`truth`: role, kind, moving, speed,
`clear_m` = the lateral gap between its body and the strip the car's body sweeps along its lane, negative: inside;
`ref`: its near face in the car frame), the car's speed (`ego_v`), the track's errors (`err`), the frame's pairs
(`modes`, `ego_modes`) and the object's history before it was said (`hist`). A RUN is one scenario run: its rows
(`said`), `collision_t` (the scripted conflict's time, None where the scenario has none) and `first_true`: how long
before the collision the alert was first said with the truth OBSTACLE as its primary (None: never).

    classify(row)    "true" | "arguable" | "false"
    cause(row)       why it was said, by counterfactual on the frame's own pairs (see CAUSES)
    table(runs)      per scenario: true / arguable / false; causes of the false ones
    leads(runs)      per scenario and day / night: runs warned on the obstacle / runs, the leads
    paired(a, b, f)  the paired difference of a per-run figure between two result sets, with a bootstrap interval

The older counter (tools/decide_replay.py `false_candidate`, and counting every said episode of a scenario without a
scripted collision) calls a warning of a slow lead the car closes on, or of a pedestrian walking into the lane,
false. This one asks the truth.
"""
import math
import random
from collections import defaultdict

OBSTACLE_ROLES = ("stopped_in_lane", "half_in_lane", "crossing_ped")
SAME_WAY = ("lead", "slow_lead", "adjacent", "overtaken", "stopped_in_lane", "parked", "half_in_lane")
EGO_HALF_W = 0.88
EGO_FRONT = 2.1
# classify: a conflict is "true" from this much braking (m/s^2, v_rel^2 / 2 gap) or this near in time (s)
TRUE_DECEL, TRUE_TTC_S = 1.5, 3.0
PED_NEAR_M = 1.0

CAUSES = """causes (the alert's caution needs a mass of 0.25 for a vehicle, 0.20 for a person, 0.30 else, at level >= 2):
  real            the truth is in the car's swept strip, or is a lead / a pedestrian walking: said with a reason
  jump            the track's place jumped by more than 3 m beyond the car's motion within the second before
  ego_straight    without the car's own off-road futures (only `path`, `follow`) the mass is under the minimum
  moving_modes    the truth stands; with the object's standing futures alone (renormalised) under the minimum
  ego+moving      only both together take it under
  moving_object   the truth moves (role given)
  heading         the track's heading is off by more than 0.35 rad and its place is clear of the corridor
  in_lane_mean    the standing futures against the path carry it and the track's reference point is in the corridor
                  (:honest the range error under 2 stated sigma, :bad_range beyond)
  spread          ... the track's reference point is clear of the corridor: the stated spread alone carries it
  unmatched       no truth actor near the object, or no record of it"""

MASS_MIN = {"vehicle": 0.25, "vru": 0.20, "unknown": 0.30}
EGO_ROAD = ("path", "follow")


def classify(row):
    """A said episode against the truth at the moment it was said: "true" | "arguable" | "false".

    true      the truth actor's body is in the strip the car's body sweeps along its lane (clear_m < 0), ahead, the
              car closes on it, and avoiding it from there needs at least 1.5 m/s^2 of braking (v_rel^2 / 2 gap) or
              contact is within 3 s
    arguable  in the strip and closing but milder than that; or a pedestrian walking, less than 1 m from the strip
    false     the rest: clear of the strip, not closing, or no truth actor near the alert's object"""
    tr = row.get("truth")
    if tr is None:
        return "false"
    v = row.get("ego_v") or 0.0
    role = tr.get("role")
    same = role in SAME_WAY
    sp = tr.get("speed") or 0.0
    vrel = v - sp if (same and tr.get("moving")) else v + sp if role == "oncoming" else v
    x = (tr.get("ref") or (0.0, 0.0))[0]
    gap = max(0.1, x - EGO_FRONT)
    if tr.get("clear_m", 1.0) < 0.0 and vrel > 0.5 and x > 0.0:
        return "true" if vrel * vrel / (2.0 * gap) >= TRUE_DECEL or gap / vrel <= TRUE_TTC_S else "arguable"
    if tr.get("kind") == "human" and tr.get("moving") and tr.get("clear_m", 9.0) < PED_NEAR_M:
        return "arguable"
    return "false"


def masses(row):
    """The mass at caution or above of the said object's pairs: (all, without the car's off-road futures, the
    object's standing futures only, both)."""
    ems = {e["name"]: e["p"] for e in row.get("ego_modes", [])}
    p_road = sum(p for n, p in ems.items() if n in EGO_ROAD) or 1e-9
    modes = row.get("modes", [])
    st_sum = sum(m["p0n"] for m in modes if m["move_m"] < 1.0) or 1e-9
    m_all = m_ego = m_st = m_both = 0.0
    for m in modes:
        stand = m["move_m"] < 1.0
        f = (m["p0w"] / m["p0n"]) if m["p0n"] > 0 else 1.0
        for pr in m["pairs"]:
            if (pr.get("le") or 0) < 2 or not pr.get("pc"):
                continue
            pe = ems.get(pr["ego"], 0.0)
            m_all += m["p0w"] * pe * pr["pc"]
            if pr["ego"] in EGO_ROAD:
                m_ego += m["p0w"] * pe / p_road * pr["pc"]
            if stand:
                m_st += m["p0n"] / st_sum * f * pe * pr["pc"]
                if pr["ego"] in EGO_ROAD:
                    m_both += m["p0n"] / st_sum * f * pe / p_road * pr["pc"]
    return m_all, m_ego, m_st, m_both


def jumped(row):
    """The track's place jumped (more than 3 m between two frames beyond the car's motion) within the second before."""
    h = row.get("hist") or []
    v = row.get("ego_v") or 0.0
    for a, b in zip(h, h[1:]):
        if a[0] >= -1.0 and a[5] is not None and b[5] is not None and a[6] is not None and b[6] is not None:
            if math.hypot((b[5] - a[5]) + v * (b[0] - a[0]), b[6] - a[6]) > 3.0:
                return True
    return False


def cause(row):
    """Why the episode was said (CAUSES)."""
    tr, er = row.get("truth"), row.get("err")
    if tr is None or "rec" not in row or er is None:
        return "unmatched"
    kind = (row.get("reason") or {}).get("kind", "vehicle")
    thr = MASS_MIN.get(kind, 0.25) * 0.999
    real = tr["clear_m"] < 0.1 or tr["role"] == "stopped_in_lane"
    if tr["role"] in ("slow_lead", "lead") and abs(tr["d_centre"]) < 1.5:
        real = True
    if tr["role"] == "crossing_ped" and tr.get("moving"):
        real = True
    if real:
        return "real"
    if jumped(row):
        return "jump"
    m_all, m_ego, m_st, m_both = masses(row)
    if m_ego < thr:
        return "ego_straight"
    if not tr["moving"] and m_st < thr:
        return "moving_modes"
    if not tr["moving"] and m_both < thr:
        return "ego+moving"
    if tr["moving"]:
        return "moving_object:" + str(tr["role"])
    inlane = abs(er["d_track"]) - 0.3 < EGO_HALF_W + 0.15
    sd = (row.get("track") or {}).get("std") or [1.0, 1.0]
    z = abs(er["range"]) / (math.hypot(sd[0], sd[1]) or 1.0)
    if abs(er.get("hdg_err") or 0.0) > 0.35 and not inlane:
        return "heading"
    if inlane:
        return "in_lane_mean:bad_range" if z >= 2.0 else "in_lane_mean:honest"
    return "spread"


def key(run):
    return (run["scenario"], run["seed"], bool(run["night"]), run.get("fut", "auto"))


def counts(run):
    """{"true": n, "arguable": n, "false": n} of a run's said episodes."""
    out = {"true": 0, "arguable": 0, "false": 0}
    for row in run["said"]:
        out[classify(row)] += 1
    return out


def table(runs, conflicts=False):
    """({scenario: {"true", "arguable", "false"}}, {cause: {scenario: n}} of the false ones, total) over the runs
    without a scripted collision (conflicts=True: over those with one, their obstacle's own episodes left out)."""
    tab = defaultdict(lambda: {"true": 0, "arguable": 0, "false": 0})
    cz = defaultdict(lambda: defaultdict(int))
    for r in runs:
        if bool(r.get("collision_t")) != conflicts:
            continue
        tab[r["scenario"]]
        for row in r["said"]:
            if conflicts and (row.get("truth") or {}).get("role") in OBSTACLE_ROLES:
                continue
            c = classify(row)
            tab[r["scenario"]][c] += 1
            if c == "false":
                cz[cause(row)][r["scenario"]] += 1
    tot = {k: sum(v[k] for v in tab.values()) for k in ("true", "arguable", "false")}
    return dict(tab), {k: dict(v) for k, v in cz.items()}, tot


def leads(runs):
    """{(scenario, "D" | "N"): {"n": runs, "lead": [s before the collision, the runs warned on the obstacle]}}."""
    out = {}
    for r in runs:
        if not r.get("collision_t"):
            continue
        e = out.setdefault((r["scenario"], "N" if r["night"] else "D"), {"n": 0, "lead": []})
        e["n"] += 1
        if r.get("first_true") is not None:
            e["lead"].append(r["first_true"])
    for e in out.values():
        e["lead"].sort()
    return out


def lead_changes(base, new, tol=0.05):
    """Per run of both sets with a scripted collision: [(key, base lead, new lead)] where the warning was lost
    (new None), gained (base None) or moved by more than tol seconds."""
    B = {key(r): r for r in base if r.get("collision_t")}
    out = []
    for r in new:
        k = key(r)
        if k not in B or not r.get("collision_t"):
            continue
        a, b = B[k].get("first_true"), r.get("first_true")
        if (a is None) != (b is None) or (a is not None and abs(a - b) > tol):
            out.append((k, a, b))
    return sorted(out)


def paired(base, new, f, n_boot=4000, seed=7, unit=lambda k: (k[1], k[2])):
    """The difference new - base of sum(f(run)) over the runs both sets have, and its bootstrap spread: runs are
    resampled in whole units (default: a seed by day or night, all its scenarios together: the runs of one seed share
    the simulator's draws). Returns {"n", "base", "new", "diff", "sd", "lo", "hi"} (lo, hi: the 2.5 and 97.5 %
    points of the resampled difference)."""
    B = {key(r): r for r in base}
    N = {key(r): r for r in new}
    ks = sorted(set(B) & set(N))
    groups = defaultdict(list)
    for k in ks:
        groups[unit(k)].append(f(N[k]) - f(B[k]))
    d = [sum(v) for _, v in sorted(groups.items())]
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)) if n else [0.0]
    mean = sum(boots) / len(boots)
    sd = math.sqrt(sum((b - mean) ** 2 for b in boots) / len(boots))
    return {"n": len(ks), "units": n, "base": sum(f(B[k]) for k in ks), "new": sum(f(N[k]) for k in ks),
            "diff": sum(d), "sd": sd, "lo": boots[int(0.025 * (len(boots) - 1))], "hi": boots[int(0.975 * (len(boots) - 1))]}


def n_false(run):
    return 0 if run.get("collision_t") else counts(run)["false"]


def n_said(run):
    return 0 if run.get("collision_t") else len(run["said"])


def n_other(run):
    """Said episodes of a conflict run whose object is not the obstacle (and is not a true conflict of its own)."""
    if not run.get("collision_t"):
        return 0
    return sum(1 for row in run["said"] if (row.get("truth") or {}).get("role") not in OBSTACLE_ROLES and classify(row) == "false")


def warned(run):
    return 1 if run.get("collision_t") and run.get("first_true") is not None else 0
