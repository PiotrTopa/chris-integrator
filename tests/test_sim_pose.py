"""The detector simulator's pose head (option `pose`, preset DETIN_POSE; contract "pose1", roadsense
docs/ARCHITECTURE.md "Pose of a road user", numbers: docs/lanes/detin-consumption-design.md section 9.6 and the
chris-pose model card): off = bit-identical streams; on = the contract's layout, the head's cap and classes, its
own random stream, and the error statistics against the simulator's truth within loose bounds."""
import hashlib
import json
import math

import numpy as np
import pytest

from chris_integrator.core import roadusers
from chris_integrator.sim import roaduser_sim as rs

SC_DAY = {"px": 0.08, "rel": 0.002, "p": 0.78, "min_dt": 0.2, "max_dt": 1.0, "max_std": 0.06}
SC_NIGHT = {"px": 0.45, "rel": 0.004, "p": 0.6, "min_dt": 0.2, "max_dt": 1.0, "max_std": 0.06}
VEH = ("car", "truck", "bus", "train")

# sha256 of json.dumps([arrival, message], sort_keys=True) over all yielded messages, recorded with the simulator
# BEFORE the pose option existed (da29a9a); DetectorSim(seed=1000 + seed), DETIN runs with scale_rate and the world's
# dims_spread
OFF_DIGESTS = {
    ("parked_street", 1, False, "plain"): "dd47a65fea53763b809d95122c508177aeaa4be645acd72df21eb211bc7c4b9e",
    ("oncoming", 2, True, "plain"): "fafb17bf3ff0e58ce0f9465bed114867e1578ea939e10606305744f092c3ee3a",
    ("parked_street", 2, True, "detin"): "5d64d30785f6faa71923f2f9d72466cca3ba0395e7a4136934b1472b2c807b81",
    ("platoon", 1, False, "detin"): "4636762e1c211be1450fe82b9dab87f6e744407ed542c9a4200b3a55d379fe7f",
}


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("CHRIS_SIM_DET", raising=False)


def _run(name, seed, night, detin=True, **kw):
    par = dict(rs.DETIN, scale_rate=SC_NIGHT if night else SC_DAY, **kw) if detin else (kw or None)
    w = rs.scenario(name, seed=seed, night=night, world={"dims_spread": True} if detin else None)
    det = rs.DetectorSim(seed=1000 + seed, night=night, params=par)
    return det, list(det.generate(w))


def _digest(msgs):
    h = hashlib.sha256()
    for arr, msg in msgs:
        h.update(json.dumps([arr, msg], sort_keys=True).encode())
    return h.hexdigest()


_CACHE = {}


def _pose_run(name, seed, is_night, **kw):
    name, seed, night = name, seed, is_night
    key = (name, seed, night, json.dumps(kw, sort_keys=True))
    if key not in _CACHE:
        _CACHE[key] = _run(name, seed, night, pose=dict(kw) if kw else True)
    return _CACHE[key]


# ---- off ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(OFF_DIGESTS))
def test_off_bit_identical(key):
    name, seed, night, mode = key
    _, msgs = _run(name, seed, night, detin=mode == "detin")
    assert _digest(msgs) == OFF_DIGESTS[key]
    for _, m in msgs:
        assert "pose_kind" not in m
        assert all("pose" not in d for d in m["road_users"] + (m.get("road_users_low") or []))


@pytest.mark.parametrize("off", [None, False])
def test_off_values(off):
    _, a = _run("platoon", 1, False)
    _, b = _run("platoon", 1, False, pose=off)
    assert _digest(a) == _digest(b)


def test_on_changes_nothing_else():
    """The pose head draws from a stream of its own: strip `pose` / `pose_kind` and the stream is DETIN's."""
    for name, night in (("parked_street", False), ("oncoming", True)):
        _, a = _run(name, 1, night)
        b = json.loads(json.dumps(_pose_run(name, 1, night)[1]))     # a copy: the runs are cached
        for _, m in b:
            assert m.pop("pose_kind") == "pose1"
            for d in m["road_users"] + (m.get("road_users_low") or []):
                d.pop("pose", None)
        assert _digest(a) == _digest(b)


def test_params_arg():
    d = rs.det_params_arg("detin_pose")
    assert d["pose"] == rs.POSE and d["flags"] and d["low"] == 0.10
    d = rs.det_params_arg('{"detin": true, "pose": true}')
    assert d["pose"] is True and rs.pose_params(d["pose"]) == rs.POSE
    d = rs.det_params_arg('{"detin_pose": true, "pose": {"max_n": 2, "night": "city"}}')
    pp = rs.pose_params(d["pose"])
    assert pp["max_n"] == 2 and pp["night"] == "city" and pp["gn_med"] == rs.POSE["gn_med"]
    assert rs.pose_params(None) is None and rs.pose_params(False) is None
    assert "pose" in rs.DETIN_KEYS


# ---- the contract ---------------------------------------------------------------------------------------------------
def _check_layout(p):
    assert set(p) == {"yaw", "yaw_std", "p", "flip", "vp", "vp_conf", "gc", "gn", "gl", "gr", "dim", "dim_std", "occ",
                      "trunc", "kind"}
    assert -180.0 <= p["yaw"] <= 180.0 and p["yaw_std"] >= 0.0
    assert 0.0 <= p["p"] <= 1.0 and 0.0 <= p["flip"] <= 1.0 and p["p"] + p["flip"] <= 1.0 + 0.011
    assert p["vp"] in ("rear", "front", "side", "threequarter") and 0.25 <= p["vp_conf"] <= 1.0
    assert p["vp"] == rs._view_class(p["yaw"])
    assert len(p["gc"]) == 5 and len(p["gn"]) == 4 and len(p["gl"]) == 5 and len(p["gr"]) == 5
    assert p["gn"][0] == p["gc"][0]                       # the near-face point is in the centre's column
    for g in (p["gc"], p["gl"], p["gr"]):
        assert g[2] >= 0 and g[3] >= 0 and 0.0 <= g[4] <= 1.0
    assert p["gn"][2] >= 0 and 0.0 <= p["gn"][3] <= 1.0
    assert len(p["dim"]) == 3 and len(p["dim_std"]) == 3 and all(v > 0 for v in p["dim"])
    assert 0.0 <= p["occ"] <= 1.0 and 0.0 <= p["trunc"] <= 1.0
    assert len(p["kind"]) == 3 and abs(sum(p["kind"]) - 1.0) < 0.02
    for v in [p["yaw"], p["yaw_std"]] + p["gc"][:4] + p["gn"][:3] + p["gl"][:4] + p["gr"][:4]:
        assert round(v, 1) == v                           # signd's decimals
    for v in [p["p"], p["flip"], p["vp_conf"], p["occ"], p["trunc"], p["gc"][4], p["gn"][3]] + p["dim"] + p["kind"]:
        assert round(v, 2) == v


def test_contract_layout_cap_and_classes():
    det, msgs = _pose_run("parked_street", 1, False)
    rr_box = 0
    for _, m in msgs:
        assert m["pose_kind"] == "pose1"
        boxes = m["road_users"] + (m.get("road_users_low") or [])
        posed = [d for d in boxes if "pose" in d]
        assert len(posed) <= rs.POSE["max_n"]
        big = [d for d in boxes if d["cls"] in VEH and d["box"][2] - d["box"][0] >= 8 and d["box"][3] - d["box"][1] >= 8]
        assert len(posed) == min(len(big), rs.POSE["max_n"])  # the cap is filled while boxes wait
        rr_box += len(big) > rs.POSE["max_n"]
        for d in posed:
            assert d["cls"] in VEH
            assert d["box"][2] - d["box"][0] >= 8 and d["box"][3] - d["box"][1] >= 8
            _check_layout(d["pose"])
    assert rr_box > 10                                    # the busy passes take turns (this scenario has many)
    # through the stack's own converter
    m = next(m for _, m in msgs if any("pose" in d for d in m["road_users"]))
    out = roadusers.from_lights(m, rs.CALIB)
    assert out["pose_kind"] == "pose1"
    assert [q.get("pose") for q in out["objects"]] == [d.get("pose") for d in m["road_users"]]
    assert [q.get("pose") for q in out.get("objects_low", [])] == [d.get("pose") for d in m.get("road_users_low", [])]
    _, plain = _run("platoon", 1, False)
    assert "pose_kind" not in roadusers.from_lights(plain[3][1], rs.CALIB)


def test_turns_take_every_box():
    """More vehicle boxes than the cap: within a few passes every box of a steady scene gets a pose."""
    det, msgs = _pose_run("parked_street", 1, False)
    got, seen = set(), {}
    for k, (_, m) in enumerate(msgs):
        for lst, tl in ((m["road_users"], det.log_x[k]["first"]),):
            for d, tr in zip(lst, tl):
                if d["cls"] in VEH and isinstance(tr["id"], int) and d["box"][3] - d["box"][1] >= 8 and d["box"][2] - d["box"][0] >= 8:
                    seen[tr["id"]] = seen.get(tr["id"], 0) + 1
                    if "pose" in d:
                        got.add(tr["id"])
    often = [a for a, n in seen.items() if n >= 5]
    assert len(often) > 30 and sum(a in got for a in often) >= 0.95 * len(often)


# ---- statistics against the truth -----------------------------------------------------------------------------------
def _rows(runs):
    rows = []
    for name, seed, night in runs:
        det, msgs = _pose_run(name, seed, night)
        for k, (_, m) in enumerate(msgs):
            lx = det.log_x[k]
            for lst, tl in ((m["road_users"], lx["first"]), (m.get("road_users_low") or [], lx["low"])):
                for d, tr in zip(lst, tl):
                    pt = tr.get("pose_true")
                    if "pose" in d and pt and not pt.get("junk"):
                        rows.append(dict(run=(name, seed, night), t=lx["t"], box=d["box"], pose=d["pose"], tr=pt))
    return rows


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _lagcorr(series, lo, hi):
    x, y = [], []
    for s in series.values():
        s.sort()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                dt = s[j][0] - s[i][0]
                if dt > hi:
                    break
                if dt >= lo:
                    x.append(s[i][1])
                    y.append(s[j][1])
    return float(np.corrcoef(x, y)[0, 1])


@pytest.fixture(scope="module")
def rows():
    return _rows([("parked_street", 1, False), ("curve_parked", 1, False), ("oncoming", 1, False), ("platoon", 2, True),
                  ("adjacent", 2, True)])


def test_heading_statistics(rows):
    err = np.array([abs(_wrap(r["pose"]["yaw"] - r["tr"]["yaw"])) for r in rows])
    rear = np.array([e for e, r in zip(err, rows) if r["tr"]["view"] == "rear"])
    assert 1.0 < np.median(rear) < 3.0                    # M 1.7 deg
    assert 0.02 < (rear > 90).mean() < 0.09               # M 4.9 %
    assert 0.04 < (err > 30).mean() < 0.14                # M 8.8 % (view mix differs)
    lobe = err < 52.5
    z = err[lobe] / np.maximum(0.05, [r["pose"]["yaw_std"] for r, l in zip(rows, lobe) if l])
    assert 0.60 < (z < 1).mean() < 0.78                   # M 68.6 %
    assert 0.86 < (z < 2).mean() < 0.97                   # M 91.7 %
    assert 0.008 < (z > 3).mean() < 0.06                  # M 2.8 %
    # honest flip / p: realised against stated
    fl = np.array([r["pose"]["flip"] for r in rows])
    isf = err > 127.5
    assert abs(isf.mean() - fl.mean()) < 0.02
    hi = fl >= 0.1
    assert hi.sum() > 50 and abs(isf[hi].mean() - fl[hi].mean()) < 0.08
    pm = np.array([r["pose"]["p"] for r in rows])
    assert abs(lobe.mean() - pm.mean()) < 0.02
    # sign convention: the car-frame heading = the bearing of the centre - yaw (from the emitted gc and the true camera)
    f, cx, cy = rs.CALIB["focal_px"], rs.CALIB["cx"], rs.CALIB["cy"]
    d_ok, d_bad = [], []
    for r, l in zip(rows, lobe):
        if not l:
            continue
        P = np.array(r["tr"]["R"]).reshape(3, 3).T @ np.array([1.0, (r["pose"]["gc"][0] - cx) / f, (r["pose"]["gc"][1] - cy) / f])
        b = math.degrees(math.atan2(-P[1], P[0]))
        h = math.degrees(r["tr"]["heading"])
        d_ok.append(_wrap(b - r["pose"]["yaw"] - h))
        d_bad.append(_wrap(b + r["pose"]["yaw"] - h))
    assert abs(np.median(d_ok)) < 0.5 and abs(np.mean(d_ok)) < 1.5
    assert np.median(np.abs(d_bad)) > 5.0


def test_time_correlation(rows):
    sh, sg = {}, {}
    for r in rows:
        st = r["tr"]["state"]
        key = (r["run"], r["tr"]["aid"])
        if st != 2:
            e = _wrap(r["pose"]["yaw"] - r["tr"]["yaw"] - (180.0 if st == 1 else 0.0))
            sh.setdefault(key, []).append((r["t"], e / max(0.05, r["pose"]["yaw_std"])))
        if r["tr"]["bottom"] != "whole":
            sg.setdefault(key, []).append((r["t"], (r["pose"]["gn"][1] - r["tr"]["vn"]) / r["tr"]["unit"]))
    c2, c1 = _lagcorr(sh, 0.15, 0.25), _lagcorr(sh, 0.85, 1.15)
    assert 0.4 < c2 < 0.85 and 0.05 < c1 < c2             # M 0.65 / 0.33
    g2, g1 = _lagcorr(sg, 0.15, 0.25), _lagcorr(sg, 0.85, 1.15)
    assert 0.4 < g2 < 0.9 and g1 < g2                     # M 0.68 / 0.26 (hidden bottoms)
    # flips come and go
    seq = {}
    for r in rows:
        seq.setdefault((r["run"], r["tr"]["aid"]), []).append((r["t"], abs(_wrap(r["pose"]["yaw"] - r["tr"]["yaw"])) > 90))
    ch = []
    for v in seq.values():
        v.sort()
        ch += [a[1] != b[1] for a, b in zip(v, v[1:]) if 0.15 <= b[0] - a[0] <= 0.25]
    assert 0.01 < np.mean(ch) < 0.09                      # rides: 1.4-8 % between boxes 0.2 s apart


def test_ground_point_statistics(rows):
    W0 = [r for r in rows if r["tr"]["bottom"] == "whole" and r["tr"]["cover"] <= 0.1 and r["tr"]["side"] <= 0.1]
    e = np.array([(r["pose"]["gn"][1] - r["tr"]["vn"]) / r["tr"]["unit"] for r in W0])
    assert len(e) > 300 and 0.018 < np.median(np.abs(e)) < 0.042        # M 0.028 box heights
    W = [r for r in rows if r["tr"]["bottom"] == "whole"]
    eg = np.array([(r["pose"]["gn"][1] - r["tr"]["vn"]) / r["tr"]["unit"] for r in W])
    eb = np.array([r["tr"]["e_b"] for r in W])
    rho = np.corrcoef(np.argsort(np.argsort(eg)), np.argsort(np.argsort(eb)))[0, 1]
    assert 0.5 < rho < 0.9                                # M 0.72: the same pixels as the box's bottom edge
    z = np.abs(eg) * np.array([r["tr"]["unit"] for r in W]) / np.maximum(0.05, [r["pose"]["gn"][2] for r in W])
    assert 0.6 < (z < 1).mean() < 0.8                     # M 70.2 %
    C = [r for r in rows if r["tr"]["bottom"] == "cut"]
    ec = np.array([abs(r["pose"]["gn"][1] - r["tr"]["vn"]) / r["tr"]["Ht"] for r in C])
    sc = np.array([r["pose"]["gn"][2] / r["tr"]["Ht"] for r in C])
    assert len(C) > 100 and 0.035 < np.median(ec) < 0.14  # M 0.04-0.08 of the true box height (design: sigma ~0.10)
    assert 0.75 < (ec < sc).mean() < 0.97                 # M 80-92 % within one stated sigma
    U = [r for r in rows if r["tr"]["bottom"] == "under"]
    eu = np.array([abs(r["pose"]["gn"][1] - r["tr"]["vn"]) / r["tr"]["Ht"] for r in U])
    su = np.array([r["pose"]["gn"][2] / r["tr"]["Ht"] for r in U])
    assert len(U) > 100 and 0.03 < np.median(eu) < 0.08 and 0.14 < np.quantile(eu, 0.9) < 0.35   # M 0.051 / 0.236
    assert 0.45 < (eu < su).mean() < 0.68                 # M 61.1 % (the stated too small)
    # the centre's column
    G = [r for r in rows if r["tr"]["cover"] <= 0.1 and r["tr"]["side"] <= 0.1]
    eu_ = np.array([abs(r["pose"]["gc"][0] - r["tr"]["uc"]) / r["tr"]["we"] for r in G])
    assert 0.012 < np.median(eu_) < 0.032                 # M 0.021 box widths
    # the near-face range from the emitted gn row, the true camera and ground: unbiased
    f, cx, cy, Hc = rs.CALIB["focal_px"], rs.CALIB["cx"], rs.CALIB["cy"], rs.CALIB["height_m"]
    dr = []
    for r in rows:
        P = np.array(r["tr"]["R"]).reshape(3, 3).T @ np.array([1.0, (r["pose"]["gn"][0] - cx) / f, (r["pose"]["gn"][1] - cy) / f])
        if P[2] > 1e-4:
            k = (Hc - r["tr"]["dz"]) / P[2]
            dr.append(math.hypot(P[0] * k, P[1] * k) / r["tr"]["range_near"] - 1.0)
    assert abs(np.median(dr)) < 0.02


def test_other_fields(rows):
    assert np.mean([r["pose"]["kind"][0] >= 0.5 for r in rows]) > 0.98                       # M 99.6 %
    assert 0.82 < np.mean([(r["pose"]["occ"] >= 0.5) == r["tr"]["occ_true"] for r in rows]) < 0.93   # M 87.7 %
    cars = [r for r in rows if r["tr"]["kind"] == "car"]
    le = np.array([math.log(r["pose"]["dim"][0] / r["tr"]["L"]) for r in cars])
    assert 0.04 < np.median(np.abs(le)) < 0.11            # M 0.073 (no better than the prior)
    trucks = [r for r in rows if r["tr"]["kind"] == "truck"]
    if len(trucks) > 20:
        assert np.median([math.log(r["pose"]["dim"][2] / r["tr"]["H"]) for r in trucks]) < -0.2   # lorries read low


def test_city_night_more_flips_same_stated():
    _, a = _pose_run("adjacent", 2, True)
    _, b = _pose_run("adjacent", 2, True, night="city")
    def stats(msgs):
        p = [d["pose"] for _, m in msgs for d in m["road_users"] + (m.get("road_users_low") or []) if "pose" in d]
        return np.mean([q["flip"] for q in p]), len(p)
    fa, na = stats(a)
    fb, nb = stats(b)
    assert na == nb and abs(fa - fb) < 0.02               # stated as by day
    rows_a = _rows([("adjacent", 2, True)])
    det, msgs = _pose_run("adjacent", 2, True, night="city")
    flips = []
    for k, (_, m) in enumerate(msgs):
        for d, tr in zip(m["road_users"] + (m.get("road_users_low") or []), det.log_x[k]["first"] + det.log_x[k]["low"]):
            pt = tr.get("pose_true")
            if "pose" in d and pt and not pt.get("junk"):
                flips.append(abs(_wrap(d["pose"]["yaw"] - pt["yaw"])) > 127.5)
    base = np.mean([abs(_wrap(r["pose"]["yaw"] - r["tr"]["yaw"])) > 127.5 for r in rows_a])
    assert np.mean(flips) > 1.4 * base
