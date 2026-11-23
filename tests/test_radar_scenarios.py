"""The tracker-core scenarios (roaduser_sim: crossing_pair, occlusion, parked_row, queue, platoon) through the radar
component alone (sim/radar_score.py). What must hold for ANY sane tracker is asserted outright; the stronger
properties are thresholds in EXPECT, applied only where the metric exists (output fields present) and the
threshold is not None (set them as the tracker core lands; the baselines are in the lane report)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chris_integrator.sim import radar_score as RS  # noqa: E402
from chris_integrator.sim import roaduser_sim as rs  # noqa: E402
from chris_integrator.sim.radar import RadarSim, attach  # noqa: E402

ROLES = {
    "crossing_pair": {"pair_a", "pair_b", "pair_c", "pair_d", "ped_a", "ped_b"},
    "occlusion": {"occ_lead", "occ_truck", "occ_ped", "occ_row", "edge_car"},
    "parked_row": {"parked_par", "parked_bay", "pull_out"},
    "queue": {"q1", "q2", "q3", "q4", "q5"},
    "platoon": {"p1", "p2", "p3", "p4"},
}
# runs: (scenario, light, radar); the key in EXPECT is "scenario/light/radar" (radar "camera" = boxes only)
RUNS = [(n, "day", None) for n in rs.CORE_SCENARIOS] + [("occlusion", "day", "extended"), ("parked_row", "day", "extended")]


def _key(name, light, radar):
    return "%s/%s/%s" % (name, light, radar or "camera")


# Stronger properties: {run key: {metric path: (op, threshold)}}; path into score()'s dict, dot separated
# ("roles.<role>...." / "all...." / "groups.<gid>...." / top-level). None = not enforced yet.
_COMMON = {
    "all.id_switches_per_actor": ("<=", None),
    "all.held_share": (">=", None),
    "all.moving_called_stationary": ("<=", None),
    "all.occlusion.object.held_share": (">=", None),
    "all.occlusion.object.reacq_same_id": (">=", None),
    "all.occlusion.fov.reacq_same_id": (">=", None),
    "all.extent.len_err_p50": ("<=", None),
    "all.extent.wid_err_p50": ("<=", None),
    "all.extent.hdg_err_pi_p50": ("<=", None),
    "all.extent.hdg_err_2pi_p50": ("<=", None),
    "false_tracks_per_min": ("<=", None),
}
EXPECT = {
    "crossing_pair/day/camera": dict(_COMMON, **{
        "roles.pair_a.id_switches": ("<=", None),
        "roles.pair_b.id_switches": ("<=", None),
        "roles.ped_a.id_switches": ("<=", None),
        "roles.ped_b.id_switches": ("<=", None),
        "roles.pair_a.occlusion.object.reacq_same_id": (">=", None),
    }),
    "occlusion/day/camera": dict(_COMMON, **{
        "roles.occ_lead.occlusion.object.held_share": (">=", None),
        "roles.occ_lead.occlusion.object.reacq_same_id": (">=", None),
        "roles.occ_ped.occlusion.object.held_share": (">=", None),
        "roles.occ_ped.occlusion.object.reacq_same_id": (">=", None),
        "roles.edge_car.occlusion.fov.reacq_same_id": (">=", None),
        "all.occ_flag_agree": (">=", None),
    }),
    "occlusion/day/extended": dict(_COMMON, **{
        "roles.occ_lead.occlusion.object.held_share": (">=", None),
        "false_tracks_dup_per_min": ("<=", None),
    }),
    "parked_row/day/camera": dict(_COMMON, **{
        "roles.parked_par.stationary_called_stationary": (">=", None),
        "roles.parked_bay.stationary_called_stationary": (">=", None),
        "roles.pull_out.moving_called_stationary": ("<=", None),
        "group_membership": (">=", None),
        "group_false": ("<=", None),
        "roles.parked_bay.extent.hdg_err_pi_p50": ("<=", None),
    }),
    "parked_row/day/extended": dict(_COMMON, **{
        "roles.parked_par.stationary_called_stationary": (">=", None),
        "group_membership": (">=", None),
        "false_tracks_dup_per_min": ("<=", None),
    }),
    "queue/day/camera": dict(_COMMON, **{
        "groups.queue1.membership": (">=", None),
        "group_false": ("<=", None),
    }),
    "platoon/day/camera": dict(_COMMON, **{
        "groups.platoon1.membership": (">=", None),
        "groups.platoon1.kind_ok": (">=", None),
    }),
}

_CACHE = {}


def _scored(name, light, radar):
    k = _key(name, light, radar)
    if k not in _CACHE:
        r = RS.run(name, seed=1, night=light == "night", radar=radar, hz_out=20.0)
        _CACHE[k] = (r, RS.score(r))
    return _CACHE[k]


def _get(d, path):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


TOP_KEYS = {"scenario", "seed", "night", "radar", "duration_s", "frames", "fields_present", "events", "actors", "roles",
            "all", "groups", "group_membership", "group_false", "false_tracks", "false_tracks_per_min",
            "false_tracks_dup_per_min", "false_tracks_phantom_per_min", "unmatched_confirmed_s_per_min"}
STAT_KEYS = {"n_actors", "det_s", "conf_share", "held_share", "id_switches", "id_switches_per_actor", "fragments",
             "ttfc_s", "pos_err", "vel_err", "motion_s", "stationary_called_stationary", "moving_called_stationary",
             "occlusion", "extent", "occ_flag_agree"}
OCC_KEYS = {"intervals", "tracked_before", "occ_s", "held_share", "alive_share", "exist_min", "err_growth_p50",
            "err_growth_max", "reacq_ended", "reacq_same_id", "reacq_any"}
EXT_KEYS = {"n", "len_err_p50", "wid_err_p50", "hdg_err_pi_p50", "hdg_err_2pi_p50"}


@pytest.mark.parametrize("name", rs.CORE_SCENARIOS)
def test_scenario_builds_with_truth(name):
    w = rs.scenario(name, seed=1)
    assert 25.0 <= w.T <= 45.0
    assert w.spec.get("events"), "designed moments listed"
    seen, groups = set(), {}
    for t in (w.t0 + 0.2 * w.T, w.t0 + 0.5 * w.T, w.t0 + 0.8 * w.T):
        for a in w.actors_at(t):
            seen.add(a["role"])
            for k in ("len", "wid", "yaw", "group", "group_kind"):
                assert k in a
            assert a["len"] == a["L"] and a["wid"] == a["W"] and a["yaw"] == a["heading"]
            if a["group"] is not None:
                groups[a["group"]] = a["group_kind"]
    assert ROLES[name] <= {a.role for a in w.actors}
    assert seen, "actors in front of the car"
    for g, kd in groups.items():
        assert w.spec["groups"].get(g) == kd
    det = rs.DetectorSim(seed=1001)
    oc = det.occlusion(w, w.t0 + 0.5 * w.T)
    for v in oc.values():
        assert v["occ"] in (None, "object", "edge", "fov") and 0.0 <= v["cover"] <= 1.0
    assert set(det.visibility(w, w.t0 + 0.5 * w.T, detail=True)) == set(oc)


def test_occlusion_truth_moments():
    """The designed occlusions are in the truth: the lead behind the truck, the pedestrian behind the row, the
    edge car out of the field and back; in crossing_pair pair_a behind pair_b."""
    w = rs.scenario("occlusion", seed=1)
    det = rs.DetectorSim(seed=1001)
    ids = {a.role: a.id for a in w.actors}
    hist = {}
    t = w.t0
    while t < w.t0 + w.T:
        for aid, v in det.occlusion(w, t).items():
            hist.setdefault(aid, []).append((t - w.t0, v["occ"], v["occ_by"], v["edge"]))
        t += 0.1
    lead = [h for h in hist[ids["occ_lead"]] if h[1] == "object"]
    assert lead and all(h[2] == ids["occ_truck"] for h in lead) and 2.0 <= len(lead) * 0.1 <= 5.5
    assert 2.5 <= min(h[0] for h in lead) and max(h[0] for h in lead) <= 10.0
    ped = [h for h in hist[ids["occ_ped"]] if h[1] == "object"]
    assert 2.0 <= len(ped) * 0.1 <= 7.0
    assert any(h[1] is None for h in hist[ids["occ_ped"]] if h[0] > max(p[0] for p in ped))   # re-emerges
    ec = [h[1] for h in hist[ids["edge_car"]]]
    assert "fov" in ec and ("edge" in ec or None in ec)
    k_vis = [i for i, h in enumerate(ec) if h != "fov"]
    assert any(ec[i] == "fov" for i in range(k_vis[0], k_vis[-1])), "leaves the field and comes back"
    w2 = rs.scenario("crossing_pair", seed=1)
    ids2 = {a.role: a.id for a in w2.actors}
    occ = [det.occlusion(w2, w2.t0 + 13.0)[ids2["pair_a"]], det.occlusion(w2, w2.t0 + 20.0)[ids2["pair_a"]]]
    assert occ[0]["occ"] == "object" and occ[0]["occ_by"] == ids2["pair_b"] and occ[1]["occ"] is None


def test_extended_radar_returns():
    w = rs.scenario("crossing_pair", seed=1)
    r = attach(w, "extended")
    msgs = [m for _, m in r.generate(w)]
    assert len(msgs) == len(r.truth_log) > 100
    n_det = 0
    per_car = {}
    for m, (_, ids) in zip(msgs, r.truth_log):
        assert m["type"] == "objects" and m["kind"] == "radar" and len(m["objects"]) == len(ids)
        for o in m["objects"]:
            assert set(o) == {"range_m", "azimuth_deg", "range_rate_mps"}       # no object id in the message
        n_det += len(ids)
        for i in ids:
            if i >= 0:
                per_car[i] = per_car.get(i, 0) + 1
    assert max(per_car.values()) > 1.5 * len(msgs)       # several returns per scan from a near vehicle
    # the point model is as it was: one detection per vehicle, each with its truth id
    p = attach(w, "point")
    assert isinstance(p, RadarSim) and not p.extended
    m0 = next(m for _, m in p.generate(w) if m["objects"])
    assert all("truth" in o for o in m0["objects"])


@pytest.mark.parametrize("name,light,radar", RUNS)
def test_radar_alone_on_scenario(name, light, radar):
    r, s = _scored(name, light, radar)
    assert r["boxes"] > 20, "the detector emits boxes"
    assert r["frames"] and all("actors" in f and "occ" in f for f in r["frames"])
    assert TOP_KEYS <= set(s)
    for st in list(s["roles"].values()) + [s["all"]]:
        assert STAT_KEYS <= set(st)
        for kd in ("object", "fov"):
            assert OCC_KEYS <= set(st["occlusion"][kd])
        assert EXT_KEYS <= set(st["extent"])
    assert ROLES[name] & set(s["roles"]), "truth roles scored"
    confirmed = [a for a in s["actors"].values() if a["ttfc_n"] or (a["conf_share"] or 0) > 0]
    assert len(confirmed) >= 2, "the radar confirms some actors"
    assert s["all"]["held_share"] is not None and s["all"]["held_share"] > 0.05
    for k in RS.LATER_FIELDS:                                 # absent fields: the metrics that need them are None
        if not s["fields_present"][k] and k in ("len", "wid", "hdg"):
            assert s["all"]["extent"]["n"] == 0
        if not s["fields_present"]["grp"]:
            assert s["group_membership"] is None and s["group_false"] is None
    bad = []
    for path, (op, thr) in EXPECT.get(_key(name, light, radar), {}).items():
        v = _get(s, path)
        if thr is None or v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if not (v <= thr if op == "<=" else v >= thr):
            bad.append("%s = %s, want %s %s" % (path, v, op, thr))
    assert not bad, "; ".join(bad)


def test_scorer_tolerates_and_reads_later_fields():
    """The later output fields (len, wid, hdg, grp, grp_kind, occ) are read when present: a run whose outputs are
    given the truth's own values scores them as perfect."""
    r, _ = _scored("platoon", "day", None)
    import copy
    r2 = copy.deepcopy(r)
    for f in r2["frames"]:
        M = RS.match(RS._outs(f), f["actors"])
        outs = [o for o in f["objects"] if RS._num(o.get("x_m")) and RS._num(o.get("y_m"))]
        for i, (a, _, _) in M.items():
            outs[i].update(len=a["len"], wid=a["wid"], hdg=a["yaw"], grp=a["group"], grp_kind=a["group_kind"],
                           occ=f["occ"][a["id"]]["occ"])
    s = RS.score(r2)
    assert s["fields_present"]["len"] and s["fields_present"]["grp"]
    ex = s["all"]["extent"]
    assert ex["n"] > 0 and ex["len_err_p50"] == 0.0 and ex["wid_err_p50"] == 0.0 and ex["hdg_err_pi_p50"] == 0.0
    g = s["groups"]["platoon1"]
    assert g["membership"] is not None and g["membership"] > 0.5 and g["kind_ok"] == 1.0
    assert s["all"]["occ_flag_agree"] > 0.9            # (the scorer keeps last frame's pairs: a few differ)
