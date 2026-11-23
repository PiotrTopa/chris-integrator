#!/usr/bin/env python3
"""Score the current radar against the reviewed truth of recorded drives (data/radar_truth/, docs/RADAR_TRUTH.md).

    python3 tools/radar_truth_eval.py                          # every session that has labels and is on disk
    python3 tools/radar_truth_eval.py --json /tmp/truth.json   # also the full result
    python3 tools/radar_truth_eval.py --sessions-root /home/piotr/rs-rec-20261122 --no-gate

Re-runs chris_integrator.radar over each session, matches its objects to the labelled things by time and bearing
(radar_truth_lib.match: the radar object's bearing within 3 deg of the label's anchor in at least half of the
frames in which a sensor saw it inside the label's span; nothing about radar ids), and reports:

  verdict seconds  per truth class, the seconds the matched objects were called stationary / moving / unknown;
                   "wrong" = moving for a standing thing (parked / standing / standing_in_lane), stationary for a
                   moving one (moving_with / oncoming / crossing); pedestrians and cannot_tell are shown, not judged
  flags            threat / in_path seconds and episodes against the reviewed episodes: justified, false,
                   cannot_tell, and UNREVIEWED (the radar raises a flag where nobody has looked: re-run the review)
  false tracks     things labelled not_an_object that the radar confirmed
  missed           things labelled real (confidence sure or likely) that no radar object followed

Exit status 1 when a threshold of GATE is broken (unless --no-gate), 2 when no session could be scored.
"""
import argparse
import collections
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radar_truth_lib as L  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(REPO, "data", "radar_truth")
SESSIONS_ROOT = os.environ.get("RADAR_TRUTH_SESSIONS", "/home/piotr/rs-rec-20261122")

# The regression gate: (metric, limit, why). Set from the baseline of 2026-11-23 (radar at 94e54dd) with a little
# room; tighten a line when the radar improves on it, never loosen one without a reviewed reason.
GATE = (
    ("false_threat_s", 0.8, "seconds of `threat` on things reviewed as not justified (baseline 0.62)"),
    ("false_threat_episodes", 3, "episodes of `threat` reviewed as not justified (baseline 3)"),
    ("unreviewed_threat_s", 0.3, "seconds of `threat` where no review has looked: re-run tools/radar_review.py"),
    ("false_in_path_s", 1.6, "seconds of `in_path` reviewed as not justified (baseline 1.23)"),
    ("unreviewed_in_path_s", 0.5, "seconds of `in_path` where no review has looked"),
    ("wrong_verdict_share", 0.22, "share of judged object-seconds with the wrong motion verdict (baseline 0.166)"),
    ("not_an_object_confirmed", 0, "things reviewed as not_an_object that the radar confirmed (baseline 0)"),
    ("missed_real", 13, "real things (sure / likely) no radar object followed (baseline 12)"),
    ("replay_s_per_session", 60.0, "wall time of one session's replay"),
)


def load_labels(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def flag_seconds(frames, dts, tr, labels, flag):
    """Each radar frame in which an object carries `flag`, judged against the reviewed episodes of that flag."""
    eps = [lb for lb in labels if lb["kind"] == flag]
    secs = collections.Counter()
    episodes = collections.Counter()
    covered = set()
    idx = {f[0]: k for k, f in enumerate(frames)}
    for oid, track in tr.items():
        for t0, t1, pts in L.episodes(track, flag):
            votes = collections.Counter()
            for T, o in pts:
                verdict = "unreviewed"
                for n, lb in enumerate(eps):
                    if not (lb["t0"] - 1.0 <= T <= lb["t1"] + 1.0) or o["b"] is None:
                        continue
                    ab = L.anchor_bearing(lb["anchor"], T, reach_s=1.0)
                    if ab is not None and abs(o["b"] - ab) <= L.bearing_tol(ab, math.hypot(o["x"], o["y"])) + 2.0:
                        j = lb["justified"]
                        verdict = "justified" if j is True else ("false" if j is False else "cannot_tell")
                        covered.add(n)
                        break
                dt = dts[idx[T]]
                secs[verdict] += dt
                votes[verdict] += dt
            episodes[votes.most_common(1)[0][0]] += 1
    missed = [lb for n, lb in enumerate(eps) if lb["justified"] is True and n not in covered]
    return ({k: round(v, 2) for k, v in secs.items()}, dict(episodes), len(missed))


def score_session(session, labels, co=L.CALIB_OVERRIDE):
    t_start = time.perf_counter()
    frames = L.replay(session, co)
    wall = time.perf_counter() - t_start
    dts = L.frame_dt(frames)
    tr = L.tracks(frames)
    fidx = {f[0]: k for k, f in enumerate(frames)}
    objs = [lb for lb in labels if lb["kind"] == "object"]
    per_class = collections.defaultdict(lambda: collections.Counter())
    judged = wrong = 0.0
    fa_confirmed, missed, rows = [], [], []
    for lb in objs:
        ms = []
        for oid, track in tr.items():
            ok, share, n = L.match(track, lb)
            if ok:
                ms.append((share, oid))
        ms.sort(reverse=True)
        matched = [oid for _, oid in ms]
        secs = collections.Counter()
        if matched:
            by_t = collections.defaultdict(dict)
            for oid in matched:
                for T, o in tr[oid]:
                    if lb["t0"] <= T <= lb["t1"]:
                        by_t[T][oid] = o
            for T, d in by_t.items():
                o = d[next(oid for oid in matched if oid in d)]         # the best-matching object listed then
                secs[o["motion"]] += dts[fidx[T]]
        truth = lb["truth"]
        for k, v in secs.items():
            per_class[truth][k] += v
        w = 0.0
        if truth in L.STANDING:
            w = secs["moving"]
        elif truth in L.MOVING:
            w = secs["stationary"]
        if truth in L.STANDING or truth in L.MOVING:
            judged += sum(secs.values())
            wrong += w
        if truth == "not_an_object" and matched:
            fa_confirmed.append(lb["review"]["event"])
        if truth in L.REAL and lb["confidence"] in ("sure", "likely") and not matched:
            missed.append(lb["review"]["event"])
        rows.append({"label": lb["review"]["event"], "truth": truth, "confidence": lb["confidence"],
                     "t0": lb["t0"], "t1": lb["t1"], "matched": matched,
                     "seconds": {k: round(v, 2) for k, v in secs.items()}, "wrong_s": round(w, 2)})
    th_s, th_e, th_missed = flag_seconds(frames, dts, tr, labels, "threat")
    ip_s, ip_e, ip_missed = flag_seconds(frames, dts, tr, labels, "in_path")
    res = {
        "session": L.session_name(session), "frames": len(frames), "replay_s": round(wall, 2),
        "objects_listed": len(tr), "labels": len(labels),
        "verdict_seconds": {c: {k: round(v, 2) for k, v in d.items()} for c, d in sorted(per_class.items())},
        "judged_s": round(judged, 2), "wrong_verdict_s": round(wrong, 2),
        "threat_seconds": th_s, "threat_episodes": th_e, "missed_justified_threat": th_missed,
        "in_path_seconds": ip_s, "in_path_episodes": ip_e, "missed_justified_in_path": ip_missed,
        "not_an_object_confirmed": fa_confirmed, "missed_real": missed, "objects": rows,
    }
    return res


def totals(results):
    t = collections.Counter()
    for r in results:
        t["false_threat_s"] += r["threat_seconds"].get("false", 0.0)
        t["false_threat_episodes"] += r["threat_episodes"].get("false", 0)
        t["unreviewed_threat_s"] += r["threat_seconds"].get("unreviewed", 0.0)
        t["false_in_path_s"] += r["in_path_seconds"].get("false", 0.0)
        t["unreviewed_in_path_s"] += r["in_path_seconds"].get("unreviewed", 0.0)
        t["judged_s"] += r["judged_s"]
        t["wrong_verdict_s"] += r["wrong_verdict_s"]
        t["not_an_object_confirmed"] += len(r["not_an_object_confirmed"])
        t["missed_real"] += len(r["missed_real"])
        t["missed_justified_flags"] += r["missed_justified_threat"] + r["missed_justified_in_path"]
    out = {k: round(v, 3) for k, v in t.items()}
    out["wrong_verdict_share"] = round(t["wrong_verdict_s"] / t["judged_s"], 3) if t["judged_s"] else 0.0
    out["replay_s_per_session"] = max((r["replay_s"] for r in results), default=0.0)
    return out


def gate(tot):
    return [(m, tot.get(m, 0), lim, why) for m, lim, why in GATE if tot.get(m, 0) > lim]


def find_sessions(root, labels_dir=LABELS):
    out = []
    if not os.path.isdir(labels_dir):
        return out
    for fn in sorted(os.listdir(labels_dir)):
        if fn.endswith(".jsonl"):
            name = fn[:-len(".jsonl")]
            out.append((name, os.path.join(root, name), os.path.join(labels_dir, fn)))
    return out


def summary(results, tot, broken):
    lines = []
    for r in results:
        lines.append("== %s: %d frames, replay %.1f s, %d radar objects, %d labels" % (
            r["session"], r["frames"], r["replay_s"], r["objects_listed"], r["labels"]))
        for c, d in r["verdict_seconds"].items():
            lines.append("   %-14s stationary %5.1f s  moving %5.1f s  unknown %5.1f s" % (
                c, d.get("stationary", 0), d.get("moving", 0), d.get("unknown", 0)))
        lines.append("   threat s %s episodes %s | in_path s %s episodes %s" % (
            r["threat_seconds"], r["threat_episodes"], r["in_path_seconds"], r["in_path_episodes"]))
        lines.append("   not_an_object confirmed %d | real things never followed %d" % (
            len(r["not_an_object_confirmed"]), len(r["missed_real"])))
    lines.append("== total: wrong verdict %.1f of %.1f judged s (%.0f %%); false threat %.2f s in %d episodes, unreviewed %.2f s; "
                 "false in_path %.2f s, unreviewed %.2f s; not_an_object confirmed %d; real missed %d" % (
                     tot["wrong_verdict_s"], tot["judged_s"], 100 * tot["wrong_verdict_share"], tot["false_threat_s"],
                     tot["false_threat_episodes"], tot["unreviewed_threat_s"], tot["false_in_path_s"],
                     tot["unreviewed_in_path_s"], tot["not_an_object_confirmed"], tot["missed_real"]))
    for m, v, lim, why in broken:
        lines.append("GATE BROKEN: %s = %s > %s (%s)" % (m, v, lim, why))
    if not broken:
        lines.append("gate: all %d thresholds hold" % len(GATE))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions-root", default=SESSIONS_ROOT, help="directory holding the session directories")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--json", default=None, help="write the full result here")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args(argv)
    results = []
    for name, ses, lab in find_sessions(a.sessions_root, a.labels):
        if not os.path.isfile(os.path.join(ses, "messages.jsonl.gz")):
            print("%s: not on disk (%s), skipped" % (name, ses))
            continue
        results.append(score_session(ses, load_labels(lab)))
    if not results:
        print("no session scored")
        return 2
    tot = totals(results)
    broken = [] if a.no_gate else gate(tot)
    print(summary(results, tot, broken))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"totals": tot, "gate": [list(g) for g in GATE], "broken": broken, "sessions": results}, f, indent=1)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
