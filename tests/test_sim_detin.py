"""The detector simulator's detection-input options (lane detin; docs/VISUAL_RADAR_EVAL.md "Detection inputs"):
off = bit-identical to the simulator before them; on = the contract's keys and encodings, the detector's flag rule,
the measured rates and distance tables within tolerance."""
import hashlib
import json
import math
import random
import re
import statistics

import pytest

from chris_integrator.sim import roaduser_sim as rs

SC_DAY = {"px": 0.08, "rel": 0.002, "p": 0.78, "min_dt": 0.2, "max_dt": 1.0, "max_std": 0.06}
SC_NIGHT = {"px": 0.45, "rel": 0.004, "p": 0.6, "min_dt": 0.2, "max_dt": 1.0, "max_std": 0.06}

# sha256 of json.dumps([arrival, message], sort_keys=True) over all yielded messages, and of det.log, recorded with
# the simulator BEFORE the options existed (commit 76e66fd)
OFF_DIGESTS = {
    ("parked_street", 1, False, False): ("dd47a65fea53763b809d95122c508177aeaa4be645acd72df21eb211bc7c4b9e",
                                         "939c5cc935435b900108b388a392194afb502599a96fce14b7a15a6eb151bc0b"),
    ("parked_street", 2, True, True): ("7d303848219a357c136f01b6f7442bd0ba471f4a27f11bc5b21ecd52357babd9",
                                       "17507ad9003724d78174446f6de01d674688167f521f6f31a4786f173ee95866"),
    ("occlusion", 1, True, False): ("cf1cf3a41c1729f6ec3c9d983f9e501e70ae5fa182342af3a62e0ed7a80b496c",
                                    "1b1c71f940a08372a63870bf9989ca0be5b43091129ebae2708cae30ee507b46"),
    ("queue", 3, False, True): ("131cf12d3fd15704fed49b3ea6affcb4f8acc7d2ad0d02e3973b3bf8fa7f8989",
                                "c3316040da393d99cebc44e6223cac4e022afc8dc15f9ed6907b70e7c1752b89"),
    ("platoon", 2, True, False): ("b60d808c58c28d0ab40a9984a49ca47017123cc1b8a20edfad7b9e72ff961fe2",
                                  "d768b6161481bdb23ce34b3d30b76be691768415622aac4132de8b3e75d92d32"),
    ("oncoming", 1, True, True): ("3266152b1b890b74f7d4e3fd65ce49dae74e6f545a0bf75cc3ca715b4780971d",
                                  "d7ef6b9ef30b9b5dbe41b6fbad53e3d0d7fdf080202425526933c4858508969d"),
    ("curve_parked", 3, False, False): ("435a4fbcbc282bd56f4a1fc6051a0f080b334c4e7ec4765610febed50bc48de2",
                                        "d9854962febbfe5a1cee42736a71768114eb35de6c6cb29db1dc7a62254bbed3"),
}


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("CHRIS_SIM_DET", raising=False)


def _run(name, seed, night, params=None):
    w = rs.scenario(name, seed=seed, night=night)
    det = rs.DetectorSim(seed=1000 + seed, night=night, params=params)
    return det, list(det.generate(w))


_CACHE = {}


def _preset(name, seed, night, **kw):
    key = (name, seed, night, json.dumps(kw, sort_keys=True))
    if key not in _CACHE:
        par = dict(rs.DETIN, scale_rate=SC_NIGHT if night else SC_DAY, **kw)
        _CACHE[key] = _run(name, seed, night, par)
    return _CACHE[key]


# ---- options off --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("run", sorted(OFF_DIGESTS))
def test_options_off_bit_identical(run):
    name, seed, night, scale = run
    w = rs.scenario(name, seed=seed, night=night)
    for params in ({"scale_rate": SC_DAY} if scale else None,
                   dict({"scale_rate": SC_DAY} if scale else {}, visible_boxes=False, flags=False, low=None, emb=None,
                        lamps=False, scale_corr=False, roadside=None)):
        det = rs.DetectorSim(seed=1000 + seed, night=night, params=params)
        h = hashlib.sha256()
        for arr, msg in det.generate(w):
            h.update(json.dumps([arr, msg], sort_keys=True).encode())
        hl = hashlib.sha256(json.dumps(det.log, sort_keys=True, default=str).encode()).hexdigest()
        assert (h.hexdigest(), hl) == OFF_DIGESTS[run]


def test_env_and_params_precedence(monkeypatch):
    monkeypatch.setenv("CHRIS_SIM_DET", '{"low": 0.2, "flags": true}')
    d = rs.DetectorSim(seed=1, params={"low": 0.15})
    assert d.p["low"] == 0.15 and d.p["flags"] is True and d._detin
    monkeypatch.setenv("CHRIS_SIM_DET", "detin")
    d = rs.DetectorSim(seed=1, night=True, params={"emb": None})
    assert d.p["visible_boxes"] and d.p["lamps"] and d.p["emb"] is None and d.p["emb_level"] == -1.5
    assert rs.det_params_arg('{"detin_reid": true, "low": 0.2}')["emb"] == "reid32+yuv3x3"
    monkeypatch.delenv("CHRIS_SIM_DET")
    assert not rs.DetectorSim(seed=1).__dict__["_detin"]


# ---- keys and encodings ---------------------------------------------------------------------------------------------
EDGES = re.compile(r"^l?t?r?b?$")
KEYS = {"cls", "score", "box", "clip", "rider", "scale_rate", "scale_rate_std", "scale_dt_s", "scale_src", "trunc",
        "occ", "occ_by", "occ_edges", "lamps", "emb"}


@pytest.mark.parametrize("kind", ["yuv3x3", "reid32+yuv3x3"])
def test_keys_and_encodings(kind):
    det, msgs = _preset("occlusion", 1, True, emb=kind)
    n_hex = 2 * (27 if kind == "yuv3x3" else 59)
    seen = set()
    for _, m in msgs:
        assert m["emb_kind"] == kind and isinstance(m["road_users_low"], list)
        assert len(m["road_users_low"]) <= 20
        for lst in ("road_users", "road_users_low"):
            for d in m[lst]:
                assert set(d) <= KEYS, set(d) - KEYS
                seen |= set(d)
                assert EDGES.match(d["trunc"])
                occ = [k in d for k in ("occ", "occ_by", "occ_edges")]
                assert all(occ) or not any(occ)
                if "occ" in d:
                    assert 0 < d["occ"] <= 1 and EDGES.match(d["occ_edges"])
                    assert 0 <= d["occ_by"] < len(m["road_users"])
                if "emb" in d:
                    assert re.match("^[0-9a-f]{%d}$" % n_hex, d["emb"])
                    assert min(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) >= 8.0
                if "lamps" in d:
                    lp = d["lamps"]
                    assert set(lp) == {"c", "l", "r", "sep", "sep_std"} and lp["c"] in ("red", "white", "unknown")
                    assert lp["l"][0] < lp["r"][0] and abs(math.hypot(lp["r"][0] - lp["l"][0], lp["r"][1] - lp["l"][1]) - lp["sep"]) < 0.2
                    assert abs(lp["sep_std"] - math.hypot(math.hypot(0.25, 0.25), 0.02 * lp["sep"])) < 0.011
                    b = d["box"]
                    assert b[0] <= lp["l"][0] <= b[2] and b[0] <= lp["r"][0] <= b[2] and b[1] <= lp["l"][1] <= b[3]
                if "scale_src" in d:
                    assert d["scale_src"] in ("lamps", "both") and "scale_rate" in d
                if lst == "road_users_low":
                    assert d["score"] <= dict(rs.DET_DAY["low_thr"])[d["cls"]] and d["score"] > 0.10
    assert {"occ", "lamps", "emb", "scale_rate", "scale_src"} <= seen
    if kind == "reid32+yuv3x3":
        assert all(sum("emb" in d for d in m["road_users"] + m["road_users_low"]) <= 16 for _, m in msgs)


def test_no_lamps_by_day_and_emb_levels():
    _, day = _preset("occlusion", 1, False)
    assert not any("lamps" in d for _, m in day for d in m["road_users"] + m["road_users_low"])
    _, night = _preset("occlusion", 1, True)
    y = lambda msgs: statistics.mean(v / 32.0 for _, m in msgs for d in m["road_users"] if "emb" in d
                                     for v in rs.emb_unhex(d["emb"])[:9])
    assert abs(y(day)) < 0.6 and abs(y(night) + 1.5) < 0.6


# ---- flags ------------------------------------------------------------------------------------------------------------
def test_flag_rule_port():
    a = ([100, 100, 200, 200], 0)
    b = ([150, 120, 260, 230], 0)                  # bottom lower: b nearer
    assert rs.ru_nearer(b, a) == 1 and rs.ru_nearer(a, b) == -1
    c = ([150, 120, 260, 203], 0)                  # within the tie (4 % of 110 = 4.4 px): not ordered, counts both ways
    assert rs.ru_nearer(a, c) == 0
    assert rs.flag_one(a, -1, [c])[0] > 0 and rs.flag_one(c, -1, [a])[0] > 0
    occ, by, edges = rs.flag_one(a, -1, [b])
    assert by == 0 and abs(occ - 0.5 * 0.8) < 0.05 and edges == 4 | 8     # right and bottom half hidden
    assert rs.flag_one(b, -1, [a]) == (0.0, -1, 0)
    cut = ([100, 100, 200, 200], 8)                # a cut bottom is nearer than any seen bottom
    assert rs.ru_nearer(cut, b) == 1 and rs.ru_nearer(cut, cut) == 0
    assert rs.trunc_of([0, 5, 1279, 400]) == 1 | 4


def test_flags_consistent_with_emitted_boxes():
    for name, night in (("platoon", False), ("queue", True), ("parked_street", False)):
        _, msgs = _preset(name, 2, night)
        n_occ = 0
        for _, m in msgs:
            first = [dict((k, v) for k, v in d.items() if k not in ("trunc", "occ", "occ_by", "occ_edges")) for d in m["road_users"]]
            low = [dict((k, v) for k, v in d.items() if k not in ("trunc", "occ", "occ_by", "occ_edges")) for d in m["road_users_low"]]
            rs.ru_flags(first, low)
            assert first == m["road_users"] and low == m["road_users_low"]
            n_occ += sum("occ" in d for d in m["road_users"])
        assert n_occ > 0


def test_visible_boxes_truth():
    """Partly covered actors get their visible part's box (smaller than the truth box), and a covered share in the
    truth log; covered boxes without a flag exist (occluders not boxed in the pass, roadside occluders)."""
    det, msgs = _preset("parked_street", 1, False)
    cov = noflag = 0
    for (_, m), lx in zip(msgs, det.log_x):
        for d, tr in zip(m["road_users"], lx["first"]):
            if isinstance(tr["id"], int):
                assert 0.0 <= tr["cover"] < rs.DET_DAY["occl_cover"]
                if tr["cover"] > 0:
                    cov += 1
                    noflag += "occ" not in d
    assert cov > 100 and 0.2 < noflag / cov < 0.9
    assert any(str(tr["by"]).startswith("rs:") for lx in det.log_x for tr in lx["first"] if tr.get("by"))


# ---- the low list ---------------------------------------------------------------------------------------------------
def _strip(d):
    return {k: v for k, v in d.items() if k not in ("trunc", "occ", "occ_by", "occ_edges", "emb", "lamps")}


def test_first_list_unchanged_by_low_and_cues():
    for name, night in (("parked_street", True), ("occlusion", False)):
        base_det, base = _run(name, 1, night, {"scale_rate": SC_DAY})
        _, low = _run(name, 1, night, {"scale_rate": SC_DAY, "low": 0.10})
        assert [a for a, _ in base] == [a for a, _ in low]
        assert [m["road_users"] for _, m in base] == [m["road_users"] for _, m in low]
        assert any(m["road_users_low"] for _, m in low)
        _, cues = _run(name, 1, night, {"scale_rate": SC_DAY, "low": 0.10, "flags": True, "emb": "yuv3x3", "lamps": True})
        assert [[_strip(d) for d in m["road_users"]] for _, m in base] == [[_strip(d) for d in m["road_users"]] for _, m in cues]


def test_visible_boxes_keep_base_stream():
    """Paired: with visible_boxes the base draws are untouched; an actor not covered and inside the frame keeps its
    box exactly, and the false positives are the same."""
    _, base = _run("parked_street", 2, False)
    det, vis = _run("parked_street", 2, False, {"visible_boxes": True, "roadside": rs.ROADSIDE})
    assert len(base) == len(vis)
    det0 = rs.DetectorSim(seed=1002, params=None)
    list(det0.generate(rs.scenario("parked_street", seed=2)))
    same = diff = 0
    for (t0, v0, l0), (t1, v1, l1), lx in zip(det0.log, det.log, det.log_x):
        assert t0 == t1
        b0 = {e[0]: e[2] for e in l0}
        for e, tr in zip(l1, lx["first"]):
            if isinstance(e[0], str):
                assert e[0] in b0 or e[0] == "fp:transient"
                continue
            if tr["cover"] == 0 and not tr["out"] and e[0] in b0:
                same += b0[e[0]] == e[2]
                diff += b0[e[0]] != e[2]
    assert same > 80 and diff == 0


def _gaps(det):
    """(gap passes, of them with a low box of the actor): passes in which the first list misses an actor it boxes
    before and after within 1.5 s while the actor is seen in the pass's region (the rides' definition)."""
    seen = {}
    for k, (t, view, logd) in enumerate(det.log):
        for e in logd:
            if isinstance(e[0], int):
                seen.setdefault(e[0], []).append(k)
    lowids = [set(e[0] for e in ll[2]) for ll in det.log_low]
    n = n_low = 0
    for aid, ks in seen.items():
        for a, b in zip(ks, ks[1:]):
            if b - a > 1 and det.log[b][0] - det.log[a][0] <= 1.5:
                for k in range(a + 1, b):
                    if aid in det.log_x[k]["seen"]:
                        n += 1
                        n_low += aid in lowids[k]
    return n, n_low


def test_low_list_rates():
    for night in (False, True):
        n_msg = n_clutter = n_gap = n_gap_low = n_first = n_beside = 0
        for name in ("parked_street", "occlusion", "platoon"):
            det, msgs = _preset(name, 1, night)
            n_msg += len(msgs)
            for (_, m), lx in zip(msgs, det.log_x):
                n_first += sum(isinstance(tr["id"], int) for tr in lx["first"])
                n_clutter += sum(tr["why"] in ("clutter", "clutter_fixed") for tr in lx["low"])
                n_beside += sum(tr["why"] == "beside" for tr in lx["low"])
            g, gl = _gaps(det)
            n_gap += g
            n_gap_low += gl
        p = rs.DET_NIGHT if night else rs.DET_DAY
        assert 0.6 * p["low_clutter"] < n_clutter / n_msg < 1.5 * p["low_clutter"], (night, n_clutter / n_msg)
        assert 0.4 * p["p_low_beside"] < n_beside / n_first < 1.1 * p["p_low_beside"], (night, n_beside / n_first)
        # M: 68-100 % of the gap passes have a low box on the object (floor 0.10); here the gaps also hold misses by
        # the size / off-axis draw (0.4) and by cover beyond occl_cover (0.3)
        assert n_gap > 50 and 0.38 < n_gap_low / n_gap < 0.85, (night, n_gap_low, n_gap)


def test_low_class_and_nms():
    det, msgs = _preset("parked_street", 3, False)
    kinds = {a.id: a.kind for a in rs.scenario("parked_street", seed=3).actors}
    right = {"car": ("car",), "truck": ("truck",), "human": ("person",), "bike": ("rider",)}
    wrong = n = 0
    for (_, m), lx, ll in zip(msgs, det.log_x, det.log_low):
        for e, tr in zip(ll[2], lx["low"]):
            if tr["why"] in ("run", "draw", "small", "occl"):
                n += 1
                wrong += e[1] not in right[kinds[e[0]]]
    assert n > 100 and 0.01 < wrong / n < 0.12, wrong / n         # M: class wrong 5 %
    for _, m in msgs:                                              # never a duplicate of a first-list box (NMS 0.5)
        for lb in m["road_users_low"]:
            assert all(rs._iou(lb["box"], fb["box"]) <= 0.5 for fb in m["road_users"])


# ---- appearance -----------------------------------------------------------------------------------------------------
TABLE = {  # roadsense docs/lanes/detin-emb-distances.json, quantiles 5 / 25 / 50 / 75 / 95 % (other: mean over the gaps)
    ("day", "yuv3x3"): {1: [0.07, 0.331, 0.542, 0.87, 1.65], 2: [0.083, 0.407, 0.664, 1.021, 1.834],
                        4: [0.083, 0.5, 0.798, 1.204, 2.002], "other": [1.177, 1.848, 2.556, 3.486, 4.804]},
    ("day", "reid32+yuv3x3"): {1: [0.056, 0.315, 0.466, 0.661, 1.132], 2: [0.064, 0.388, 0.557, 0.77, 1.259],
                               4: [0.063, 0.453, 0.657, 0.89, 1.398], "other": [1.03, 1.453, 1.813, 2.228, 2.819]},
    ("night", "yuv3x3"): {1: [0.406, 0.595, 0.819, 1.335, 2.574], 2: [0.477, 0.682, 0.985, 1.528, 3.517],
                          4: [0.577, 0.884, 1.139, 1.719, 3.369], "other": [0.777, 1.294, 1.933, 3.379, 6.797]},
    ("night", "reid32+yuv3x3"): {1: [0.344, 0.456, 0.611, 0.929, 1.563], 2: [0.368, 0.512, 0.698, 1.032, 2.063],
                                 4: [0.428, 0.601, 0.787, 1.117, 2.196], "other": [0.678, 0.931, 1.252, 1.998, 3.603]},
}


def emb_quantiles(tod, kind, n=1500, seed=5):
    """Simulated distance quantiles through EmbGen (the simulator's own generator): same object at 1 / 2 / 4 s and
    other objects."""
    out = {}
    for gap in (1, 2, 4, "other"):
        g = rs.EmbGen(kind, tod == "night", random.Random(seed), -1.5 if tod == "night" else 0.0)
        ds = []
        for i in range(n):
            if gap == "other":
                a, b = g.observe(("a", i), 0.0), g.observe(("b", i), 0.0)
            else:
                a, b = g.observe(i, 0.0), g.observe(i, float(gap))
            ds.append(rs.emb_distance(a, b, kind))
        ds.sort()
        out[gap] = [ds[int(q * (n - 1))] for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    return out


@pytest.mark.parametrize("tod,kind", sorted(TABLE))
def test_emb_distance_quantiles(tod, kind):
    sim = emb_quantiles(tod, kind)
    for gap, tab in TABLE[(tod, kind)].items():
        for i, (s, t) in enumerate(zip(sim[gap], tab)):
            tol = 0.35 if i in (1, 2, 3) else 0.75          # in ln: the central quantiles closer than the tails
            assert abs(math.log(s / t)) < tol, (tod, kind, gap, i, s, t)


def test_emb_cover_corrupts():
    g = rs.EmbGen("yuv3x3", False, random.Random(3))
    d0 = [rs.emb_distance(g.observe(1, 0.0), g.observe(1, 0.2), "yuv3x3") for _ in range(1)]
    ds_clean, ds_cov = [], []
    for i in range(300):
        a = g.observe(("x", i), 0.0)
        ds_clean.append(rs.emb_distance(a, g.observe(("x", i), 0.2), "yuv3x3"))
        ds_cov.append(rs.emb_distance(a, g.observe(("x", i), 0.4, 0.5, ("occ", i)), "yuv3x3"))
    assert statistics.median(ds_cov) > 1.5 * statistics.median(ds_clean) and d0


# ---- lamps ----------------------------------------------------------------------------------------------------------
def test_lamp_statistics():
    found = facing = unknown = red = white = unlit = unlit_found = 0
    for name in ("oncoming", "platoon", "queue"):
        det, msgs = _preset(name, 1, True)
        w = rs.scenario(name, seed=1, night=True)
        for (t, m), (tt, view, logd) in zip(msgs, det.log):
            acts = {o["id"]: o for o in w.actors_at(tt)}
            for d, e in zip(m["road_users"], logd):
                o = acts.get(e[0])
                if o is None or o["kind"] not in rs.VEHICLE_KINDS or d["cls"] not in ("car", "truck"):
                    continue
                ca = (math.cos(o["heading"]) * o["x"] + math.sin(o["heading"]) * o["y"]) / math.hypot(o["x"], o["y"])
                if abs(ca) < math.cos(math.radians(40)) or d.get("occ"):
                    continue
                lit = rs.DetectorSim._lit(o)
                if not lit:
                    unlit += 1
                    unlit_found += "lamps" in d
                    continue
                facing += 1
                if "lamps" in d:
                    found += 1
                    c = d["lamps"]["c"]
                    unknown += c == "unknown"
                    red += c == "red" and ca > 0
                    white += c == "white" and ca < 0
    assert facing > 100
    assert 0.45 < found / facing < 0.72, found / facing     # 0.65 less the geometric refusals (pair outside the box)
    assert 0.06 < unknown / found < 0.22
    assert (red + white) / (found - unknown) > 0.9 and white > 10
    assert unlit == 0 or unlit_found / max(1, unlit) < 0.15


# ---- scale rate -----------------------------------------------------------------------------------------------------
def _scale_err_corr(params, names=("platoon", "slow_lead", "adjacent", "queue")):
    pairs = []
    for name in names:
        for seed in (1, 2):
            det, msgs = _run(name, seed, False, params)
            last = {}
            for (_, m), lx in zip(msgs, det.log_x):
                for d, tr in zip(m["road_users"], lx["first"]):
                    if "scale_rate" not in d or "scale_true" not in tr:
                        continue
                    err = (d["scale_rate"] - tr["scale_true"]) / d["scale_rate_std"]
                    aid = tr["id"]
                    if aid in last and 0 < lx["t"] - last[aid][0] <= 0.45:
                        pairs.append((last[aid][1], err))
                    last[aid] = (lx["t"], err)
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    ma, mb = statistics.mean(a), statistics.mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in pairs) / len(pairs)
    return cov / (statistics.pstdev(a) * statistics.pstdev(b)), len(pairs), statistics.pstdev(a)


def test_scale_corr_shares_errors():
    r_off, n_off, s_off = _scale_err_corr({"scale_rate": SC_DAY, "flags": True})
    r_on, n_on, s_on = _scale_err_corr({"scale_rate": SC_DAY, "flags": True, "scale_corr": True})
    assert n_off > 100 and n_on > 100
    assert abs(r_off) < 0.15, r_off
    assert r_on > 0.3, r_on
    assert 0.7 < s_on < 1.3 and 0.7 < s_off < 1.3          # the stated std is the value's 1-sigma in both


def test_scale_lamps_at_night():
    src, low = [], 0
    for name in ("occlusion", "platoon", "oncoming"):
        det, msgs = _preset(name, 1, True)
        src += [d.get("scale_src") for _, m in msgs for d in m["road_users"] if "scale_rate" in d]
        low += sum("scale_rate" in d for _, m in msgs for d in m["road_users_low"])
    assert src.count("both") + src.count("lamps") >= 5 and low > 0


# ---- world option: per-actor size spread ----------------------------------------------------------------------------
def test_dims_spread(monkeypatch):
    base = rs.scenario("parked_street", seed=4)
    off = rs.scenario("parked_street", seed=4, world={"dims_spread": False})
    assert [(a.L, a.W, a.H) for a in base.actors] == [(a.L, a.W, a.H) for a in off.actors]
    on = rs.scenario("parked_street", seed=4, world={"dims_spread": True})
    monkeypatch.setenv("CHRIS_SIM_DET", "detin")
    env = rs.scenario("parked_street", seed=4)
    assert [(a.L, a.W, a.H) for a in on.actors] == [(a.L, a.W, a.H) for a in env.actors]
    assert [a.s0 for a in on.actors] == [a.s0 for a in base.actors]            # placement untouched
    for kind, nom, sg in (("car", rs.DIMS["car"], rs.DIMS_SPREAD["car"]), ("human", rs.DIMS["human"], rs.DIMS_SPREAD["human"])):
        acts = [a for a in on.actors if a.kind == kind and a.role != "crossing_ped"]
        for k, (attr, n, s) in enumerate(zip(("L", "W", "H"), nom, sg)):
            lr = [math.log(getattr(a, attr) / n) for a in acts]
            if s == 0.0:
                assert max(abs(x) for x in lr) < 1e-12
            else:
                assert abs(statistics.mean(lr)) < 3 * s / math.sqrt(len(lr)) + 0.01
                assert 0.7 * s < statistics.pstdev(lr) < 1.3 * s, (kind, attr, statistics.pstdev(lr))
    vans = [a for a in on.actors if a.kind == "truck"]
    assert all(abs(math.log(a.W / 2.1)) < 5 * rs.DIMS_SPREAD["truck"][1] for a in vans)   # explicit dims spread too
