#!/usr/bin/env python3
"""What each alert of the decision layer would have done: a recorded drive or simulated scenarios replayed through
the stack, every alert evaluated in shadow (or as configured), and per alert a report of its episodes. This is
the evidence for switching an alert `on` after drives (docs/TOPICS.md, `decide`).

    tools/decide_replay.py CAPTURE [CAPTURE ...] [--json OUT] [--cfg JSON] [--events]
    tools/decide_replay.py --scenario NAME[,NAME...] [--seeds N] [--night] [--json OUT] [--cfg JSON]

CAPTURE: a capture of tools/cost_capture.py (`*.cap.gz`) or a recorder session directory (tools/cost_bench.py
reads both). --cfg: overrides of the `decide` plug-in's configuration as JSON (for instance
'{"alerts": {"obstacle_ahead": {"show_from": 3}}}'). Wrap long runs in a memory cap:
systemd-run --user --wait --collect --pipe -q -p MemoryMax=6G --working-directory=$PWD python3 tools/decide_replay.py ...

Per alert: episodes (from the stage leaving none to its return there), the highest stage of each, the counts of
episodes that would have been said (stage >= show_from, not suppressed), and per said episode:
    lead_s        time from the first moment it would have been said to the closest approach of its object
    closest_m     the object's least distance (its point to the car's origin) and where it was (x, y) then
    false_candidate and why: passed beside (|y| at the closest approach beyond `beside_m`), never came near
                  (closest beyond `near_m`), the radar called it parked, the driver did not brake although the
                  object came near (on a real drive: either a false alert or a driver who reacted late)
Simulated scenarios add the truth: the scripted collision time (stopped_cruise_*) and the time of the first stage.
Real drives never leave the machine: the JSON report holds only numbers, ids and times."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

NS = 1_000_000_000
BESIDE_M, NEAR_M, BRAKE_MPS2, AFTER_S = 2.2, 6.0, -1.5, 8.0


class Collector:
    """Episodes of every alert from the `decide` messages, the objects' places from `radar`."""

    def __init__(self):
        self.t0 = None
        self.track = {}                  # object id -> [(t, x, y, flags)]
        self.ep = {}                     # alert -> current episode
        self.done = {}                   # alert -> [episodes]
        self.accel = []                  # (t, a_long)
        self.events = []
        self.frames = 0

    def on(self, e):
        m = e.msg
        ts = m.get("ts_mono_ns")
        if not isinstance(ts, int):
            return
        if self.t0 is None:
            self.t0 = ts
        t = (ts - self.t0) / NS
        if e.stream == "radar":
            for o in m.get("objects") or ():
                if o.get("state") in ("confirmed", "coast"):
                    self.track.setdefault(o["id"], []).append((t, o["x_m"], o["y_m"], tuple(o.get("flags") or ())))
            a = (m.get("ego") or {}).get("accel_mps2")
            if isinstance(a, (int, float)):
                self.accel.append((t, a))
            return
        self.frames += 1
        for ev in m.get("events") or ():
            self.events.append(dict(ev, t_rel=round(t, 3)))
        for a in m.get("alerts") or ():
            name = a["id"]
            cur = self.ep.get(name)
            if a["stage"] > 0:
                if cur is None:
                    cur = self.ep[name] = {"alert": name, "mode": a["mode"], "start": t, "max_stage": 0, "objects": [],
                                           "said_at": None, "said_object": None, "suppressed": set(), "reasons": []}
                cur["max_stage"] = max(cur["max_stage"], a["stage"])
                if a.get("object") is not None and a["object"] not in cur["objects"]:
                    cur["objects"].append(a["object"])
                if a.get("suppressed"):
                    cur["suppressed"].add(a["suppressed"])
                would = a["stage"] >= self.show_from.get(name, 2) and not a.get("suppressed")
                if would and cur["said_at"] is None:
                    cur["said_at"], cur["said_object"] = t, a.get("object")
                    cur["reasons"].append(a.get("reason"))
                cur["end"] = t
            elif cur is not None:
                cur["end"] = t
                self.done.setdefault(name, []).append(cur)
                del self.ep[name]

    def finish(self):
        for name, cur in list(self.ep.items()):
            self.done.setdefault(name, []).append(cur)
        self.ep = {}

    def report(self):
        self.finish()
        out = {}
        for name, eps in self.done.items():
            rows = []
            for ep in eps:
                r = {"start_s": round(ep["start"], 2), "end_s": round(ep.get("end", ep["start"]), 2),
                     "max_stage": ep["max_stage"], "objects": ep["objects"], "suppressed": sorted(ep["suppressed"]),
                     "said": ep["said_at"] is not None}
                if ep["said_at"] is not None:
                    r.update(self._judge(ep))
                rows.append(r)
            said = [r for r in rows if r["said"]]
            out[name] = {"episodes": len(rows), "said": len(said),
                         "by_stage": {s: sum(1 for r in rows if r["max_stage"] == s) for s in (1, 2, 3, 4)},
                         "false_candidates": sum(1 for r in said if r.get("false_candidate")),
                         "lead_s": [r.get("lead_s") for r in said], "rows": rows}
        return out

    def _judge(self, ep):
        oid, t1 = ep["said_object"], ep["said_at"]
        pts = [p for p in self.track.get(oid, ()) if t1 - 0.01 <= p[0] <= t1 + AFTER_S]
        r = {"said_s": round(t1, 2), "object": oid}
        if not pts:
            r.update(lead_s=None, closest_m=None, false_candidate=True, why=["object not tracked after"])
            return r
        best = min(pts, key=lambda p: (p[1] ** 2 + p[2] ** 2))
        d = (best[1] ** 2 + best[2] ** 2) ** 0.5
        r.update(lead_s=round(best[0] - t1, 2), closest_m=round(d, 2), at=[round(best[1], 1), round(best[2], 1)])
        why = []
        if abs(best[2]) > BESIDE_M and best[1] < 8.0:
            why.append("passed beside")
        if d > NEAR_M and best[1] > 8.0:
            why.append("never came near")
        if any("parked" in p[3] for p in pts[:3]):
            why.append("the radar called it parked")
        brake = min((a for t, a in self.accel if t1 <= t <= best[0] + 0.5), default=0.0)
        if brake > BRAKE_MPS2 and d <= NEAR_M:
            why.append("no braking although it came near")
        r["min_accel_mps2"] = round(brake, 2)
        r["false_candidate"] = bool(why) and not (d <= NEAR_M and abs(best[2]) <= BESIDE_M and brake <= BRAKE_MPS2)
        r["why"] = why
        return r


def build_stack(cfg):
    from chris_integrator.mixer import MessageClock
    from chris_integrator.stack import build
    c = {"radar": {"rate_hz": 1000.0}, "decide": dict(cfg or {}, rate_hz=1000.0)}
    return build(cfg=c, mapper=None, clock=MessageClock())


def show_from(stack):
    dec = stack.mixer.plugin("decide")
    return {n: a.cfg["show_from"] for n, a in dec.alerts.alerts.items()}


def run_capture(path, cfg):
    import cost_bench as CB
    from chris_integrator.io import sockets
    lines = CB.read_input(path)
    st = build_stack(cfg)
    col = Collector()
    col.show_from = show_from(st)
    st.mixer.add_output("decide_replay", col.on, streams=("decide", "radar"))
    parse = sockets.parse_line_trimmed if st.trim_road else sockets.parse_line
    n = 0
    for name, raw in lines:
        try:
            msg, line = parse(raw)
        except ValueError:
            continue
        if msg is None:
            continue
        n += 1
        st.deliver(msg, None if name == "feed" else line)
    rep = col.report()
    h = next(p for p in st.mixer.describe()["plugins"] if p["name"] == "decide")
    hh = h.get("health") or {}
    return {"input": os.path.basename(path), "messages": n, "decide_frames": col.frames, "alerts": rep,
            "health": {"state": hh.get("state"), "reason": hh.get("reason"), "futures": hh.get("futures"),
                       "ms_per_call": h.get("ms_per_call"), "max_ms": h.get("max_ms")}}, col.events


def run_scenario(name, seed, night, cfg):
    from chris_integrator.mixer import MessageClock  # noqa: F401
    from chris_integrator.sim import default_sensors
    from chris_integrator.sim import roaduser_sim as rs
    w = rs.scenario(name, seed=seed, night=night)
    det = rs.DetectorSim(seed=1000 + seed, night=night)
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
    sensors.append(det)
    from chris_integrator.stack import build
    c = {"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}, "radar": {"rate_hz": 1000.0},
         "decide": dict(cfg or {}, rate_hz=1000.0)}
    st = build(cfg=c, mapper=None, clock=MessageClock())
    col = Collector()
    col.show_from = show_from(st)
    st.mixer.add_output("decide_replay", col.on, streams=("decide", "radar"))
    sent = False
    for _, m in w.messages(sensors):
        if not sent:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            sent = True
        st.deliver(m)
    rep = col.report()
    off = (col.t0 - int(round(w.t0 * NS))) / NS if col.t0 is not None else 0.0
    out = {"scenario": name, "seed": seed, "night": night, "alerts": rep, "collision_t": w.spec.get("collision_t")}
    oa = rep.get("obstacle_ahead") or {"rows": []}
    said = [r for r in oa["rows"] if r["said"]]
    if said and out["collision_t"]:
        out["first_said_before_collision_s"] = round(out["collision_t"] - (said[0]["said_s"] + off), 2)
    return out, col.events


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--night", action="store_true")
    ap.add_argument("--cfg", default=None, help="the decide plug-in's configuration overrides (JSON)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--events", action="store_true", help="print the shadow log")
    a = ap.parse_args(argv)
    cfg = json.loads(a.cfg) if a.cfg else {}
    results = []
    if a.scenario:
        for name in a.scenario.split(","):
            for s in range(1, a.seeds + 1):
                r, ev = run_scenario(name, s, a.night, cfg)
                results.append(r)
                _print(r, "%s seed %d%s" % (name, s, " night" if a.night else ""))
                if a.events:
                    _events(ev)
    for p in a.inputs:
        r, ev = run_capture(p, cfg)
        results.append(r)
        _print(r, r["input"])
        if a.events:
            _events(ev)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1, default=list)
    return 0


def _print(r, title):
    print("== %s" % title + ("  (collision at %.1f s; first said %.2f s before)" % (
        r["collision_t"], r["first_said_before_collision_s"]) if r.get("first_said_before_collision_s") is not None else ""))
    if r.get("health"):
        print("   decide: %s frames, %s ms per call (max %s), %s" % (r.get("decide_frames"), r["health"].get("ms_per_call"),
                                                              r["health"].get("max_ms"), r["health"].get("reason")))
    for name, s in sorted(r["alerts"].items()):
        leads = [x for x in s["lead_s"] if x is not None]
        print("   %-15s episodes %3d (stage 1/2/3/4: %s) said %3d, false candidates %d, lead before closest approach %s" % (
            name, s["episodes"], "/".join(str(s["by_stage"][k]) for k in (1, 2, 3, 4)), s["said"], s["false_candidates"],
            ("median %.1f s, min %.1f s" % (sorted(leads)[len(leads) // 2], min(leads))) if leads else "-"))
        for row in s["rows"]:
            if row["said"]:
                print("      %7.1f s stage %d object %s lead %s closest %s at %s%s" % (
                    row["said_s"], row["max_stage"], row.get("object"), row.get("lead_s"), row.get("closest_m"),
                    row.get("at"), ("  FALSE? " + ", ".join(row["why"])) if row.get("false_candidate") else ""))


def _events(ev):
    for e in ev:
        print("      %8.2f %-15s %-9s %-10s object %s%s" % (e["t_rel"], e["alert"], e["event"], e["stage_name"], e["object"],
                                                          (" suppressed " + e["suppressed"]) if e.get("suppressed") else ""))


if __name__ == "__main__":
    sys.exit(main())
