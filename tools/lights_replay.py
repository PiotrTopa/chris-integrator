#!/usr/bin/env python3
"""The ego-light selection of a recorded session, offline: every recorded `lights` message through the selector
again (chris_integrator/trackers/lights.py) with the lanes, the ego-lane estimate and the road of the replay
pipeline, and what the choice did over time.

    python3 tools/lights_replay.py SESSION [SESSION ...] [--rows] [--from S --to S] [--legacy]

Per session: the episodes (stretches with a facing light or an ego light) with, for the selection as recorded
and as replayed: messages with an ego light / held / none, changes of the ego track, A-B-A flaps of the track,
colour flips that revert within 4 s (stop -> go -> stop: a real signal does not do that), messages marked
ambiguous. `--rows` prints every message of the window (time, speed, the ego light as recorded and as replayed,
the lights with their slots). `--legacy` replays with the selection as it was before lane "drive-quality"
(no memory: per message). The recorded messages are the selector's own output: every field it writes is
written again, so a replay of a session recorded by an older selector shows what the present one would say.
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chris_integrator import contract                                   # noqa: E402
from chris_integrator.plugins.common import intrinsics                  # noqa: E402
from chris_integrator.replay import Pipeline, read_session              # noqa: E402
from chris_integrator.trackers import lights as lights_mod              # noqa: E402

FACING = ("red", "green", "yellow", "red_yellow")
STOP = ("red", "red_yellow", "yellow")


def replay(session, legacy=False, selector=None):
    """Yields (t_s from the first message, recorded lights message, replayed one, speed m/s or None)."""
    p = Pipeline()
    sel = selector or lights_mod.EgoLightSelector()
    if legacy:
        sel.memory = False
    t0 = None
    es = road = calib = None
    for m in read_session(session):
        ts, typ = m.get("ts_mono_ns"), m.get("type")
        if not isinstance(ts, int):
            continue
        if t0 is None and typ == "camera":
            t0 = ts
        if typ == "lights":
            if t0 is None:
                continue
            out = copy.deepcopy(m)
            lat = p.latest
            es = lat.get("ego_state")
            if lat.get("lanes") is not None:
                sel.on_lanes(lat["lanes"])
            if hasattr(sel, "on_egolane") and lat.get("egolane") is not None:
                sel.on_egolane(lat["egolane"], (lat.get("lanes") or {}).get("ts_mono_ns"))
            if hasattr(sel, "on_ego"):
                sel.on_ego(es)
            r = road if contract.usable(es, "vision") else None
            sel.annotate(out, r, calib, intrinsics(calib))
            v = ((es or {}).get("speed") or {}).get("value")
            yield (ts - t0) / 1e9, m, out, v
            continue
        if typ == "map":
            sel.on_map(m)
        elif typ == "calib":
            calib = m
        elif typ == "road":
            road = m
        p.feed(m)


def _stats(rows):
    """rows: [(t, ego dict or None)] of one episode."""
    n = len(rows)
    live = [(t, e) for t, e in rows if e and not e.get("held")]
    held = sum(1 for _, e in rows if e and e.get("held"))
    tracks = []
    for _, e in live:
        if not tracks or tracks[-1] != e.get("track_id"):
            tracks.append(e.get("track_id"))
    runs = []
    for t, e in rows:
        if not e:
            continue
        c = "stop" if e.get("state") in STOP else "go" if e.get("state") == "green" else None
        if c is None:
            continue
        if not runs or runs[-1][0] != c:
            runs.append([c, t, t])
        else:
            runs[-1][2] = t
    return {"n": n, "ego": len(live), "held": held, "none": n - len(live) - held,
            "switches": max(0, len(tracks) - 1), "aba": sum(1 for i in range(2, len(tracks)) if tracks[i] == tracks[i - 2]),
            "flips": sum(1 for i in range(1, len(runs) - 1) if runs[i][2] - runs[i][1] < 4.0),
            "amb": sum(1 for _, e in rows if e and e.get("amb")),
            "go": sum(1 for _, e in rows if e and e.get("state") == "green"),
            "stop": sum(1 for _, e in rows if e and e.get("state") in STOP)}


def episodes(rows, gap_s=3.0):
    """rows: [(t, recorded, replayed, v)] -> [[rows of an episode]]: a facing light or an ego light in either."""
    out, cur = [], None
    for row in rows:
        t, a, b, _ = row
        fac = any(li.get("state") in FACING and (li.get("hits") or 0) >= 3 for li in a.get("lights") or [])
        if not (fac or a.get("ego") or b.get("ego")):
            continue
        if cur is None or t - cur[-1][0] > gap_s:
            cur = []
            out.append(cur)
        cur.append(row)
    return [e for e in out if len(e) >= 4]


def _ego(e):
    if not e:
        return "-"
    a = e.get("assign") or {}
    return "%s %s%s%s%s%s" % (e.get("track_id"), e.get("state"), " held" if e.get("held") else "", " amb" if e.get("amb") else "",
                              " c%.2f" % a["confidence"] if a.get("conflict") else "", " other" if a.get("other_lane") else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--rows", action="store_true")
    ap.add_argument("--from", dest="t_from", type=float, default=-1e9)
    ap.add_argument("--to", dest="t_to", type=float, default=1e9)
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    keys = ("n", "ego", "held", "none", "switches", "aba", "flips", "amb", "go", "stop")
    grand = {"rec": dict.fromkeys(keys, 0), "new": dict.fromkeys(keys, 0)}
    for s in a.sessions:
        rows = [r for r in replay(s, a.legacy) if a.t_from <= r[0] <= a.t_to]
        if a.rows:
            for t, old, new, v in rows:
                if not (old.get("lights") or old.get("ego") or new.get("ego")):
                    continue
                ls = " ".join("[%s%s %s u%d d%.0f lat%+.1f s%+.2f%s%s]" % (
                    "*" if li.get("ego") else "", li.get("track_id"), li.get("state"), (li["box"][0] + li["box"][2]) / 2,
                    li.get("dist_m") or 0, li.get("lateral_m") or 0, li.get("slot") or 0,
                    " m%+.2f" % li["slot_mem"] if "slot_mem" in li else "", " carried" if li.get("carried") else "")
                    for li in new.get("lights") or [] if li.get("state") != "off" or li.get("ego"))
                print("%7.1f v%5.1f view%s | rec %-22s | new %-26s | %s" % (t, v if v is not None else -1, old.get("view"), _ego(old.get("ego")), _ego(new.get("ego")), ls))
        print("== %s" % os.path.basename(s.rstrip("/")))
        for ep in episodes(rows):
            so, sn = _stats([(t, o.get("ego")) for t, o, _, _ in ep]), _stats([(t, n.get("ego")) for t, _, n, _ in ep])
            for k in keys:
                grand["rec"][k] += so[k]
                grand["new"][k] += sn[k]
            print("  %7.1f-%7.1f n=%3d | rec ego/held/none %3d/%2d/%3d sw %2d aba %2d flips %2d amb %3d go/stop %3d/%3d"
                  " | new %3d/%2d/%3d sw %2d aba %2d flips %2d amb %3d go/stop %3d/%3d" % (
                      ep[0][0], ep[-1][0], so["n"], so["ego"], so["held"], so["none"], so["switches"], so["aba"], so["flips"], so["amb"], so["go"], so["stop"],
                      sn["ego"], sn["held"], sn["none"], sn["switches"], sn["aba"], sn["flips"], sn["amb"], sn["go"], sn["stop"]))
    print("TOTAL recorded %s" % grand["rec"])
    print("TOTAL replayed %s" % grand["new"])


if __name__ == "__main__":
    main()
