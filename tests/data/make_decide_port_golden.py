#!/usr/bin/env python3
"""Golden data of the prediction and decision layers as plug-ins: what `predict` and `decide` put out (`radar_fut`,
`decide`) for recorded inputs, frozen before their Python glue moved into the native core (2026-11-24) and held
since by tests/test_decide_port.py.

    python3 tests/data/make_decide_port_golden.py           # -> tests/data/decide_port_golden.json.gz

The inputs are the messages of every stream the two plug-ins read (`radar`, `road`, `roadgeom`, `lanes`, `egolane`,
`map`, `ego`, `ego_state`, `vehicle`, `env`), taken from simulator scenarios run through the whole stack; they are
stored (of `road` only what the two read: its stamp and plan), so the golden data do not move when the layers
before them change. The outputs are recorded by replaying the stored inputs through the two plug-ins alone:
per case with the prediction (`pred`) and with the degraded futures (`cv`: decide `{"futures": "cv"}`), the
`decide` messages in full and of `radar_fut` every KEEP-th message in full and a digest of each.

Record anew only for a deliberate change of what the layers put out, and say why in the commit."""
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

NS = 1_000_000_000
OUT = os.path.join(HERE, "decide_port_golden.json.gz")
# (scenario, seed, night, from s, to s): dense traffic braking to a stop, parked rows on a bend, a junction at
# night, a true warning, a street of parked cars, a cut-in from the first frame
CASES = (("platoon", 1, False, 22.0, 30.0), ("curve_parked", 2, False, 34.0, 44.0), ("junction", 1, True, 12.0, 18.0),
         ("stopped_cruise_14", 1, False, 9.5, 15.5), ("parked_street", 3, False, 12.0, 20.0), ("cut_in", 1, False, 0.0, 5.0))
INPUTS = ("radar", "road", "roadgeom", "lanes", "egolane", "map", "ego", "ego_state", "vehicle", "env")
KEEP = 10


def slim(stream, msg):
    if stream == "road":                       # predict's context and decide's path read the plan, nothing else
        return {k: msg[k] for k in ("type", "ts_mono_ns", "ts_real_ns", "plan") if k in msg}
    return msg


def record_inputs(name, seed, night, t_from, t_to):
    """The two plug-ins' input messages of a scenario through the whole stack: [[stream, message], ...]."""
    from chris_integrator.mixer import MessageClock
    from chris_integrator.sim import default_sensors
    from chris_integrator.sim import roaduser_sim as rs
    from chris_integrator.stack import build
    w = rs.scenario(name, seed=seed, night=night)
    det = rs.DetectorSim(seed=1000 + seed, night=night)
    sensors = default_sensors(vehicle={"seed": 11 + 97 * seed}, vision={"seed": 21 + 97 * seed, "leads": True},
                              gnss={"seed": 31 + 97 * seed}, map={"seed": 41 + 97 * seed})
    sensors.append(det)
    st = build(cfg={"estimator": {"sources": [{"kind": "road_users", "id": "*"}]}, "radar": {"rate_hz": 1000.0}},
               mapper=None, clock=MessageClock())
    rows, t0 = [], [None]

    def tap(e):
        ts = e.ts
        if ts is None:
            return
        if t0[0] is None:
            t0[0] = ts
        if t_from <= (ts - t0[0]) / NS <= t_to:
            rows.append([e.stream, slim(e.stream, e.msg)])
    st.mixer.add_output("golden", tap, streams=[s for s in INPUTS if s in st.mixer.bus.streams])
    sent = False
    for _, m in w.messages(sensors):
        if not sent:
            st.deliver(det.calib_msg(m["ts_mono_ns"]))
            sent = True
        st.deliver(m)
    return json.loads(json.dumps(rows))        # as they are stored


def replay(rows, decide_cfg=None, lazy=False):
    """The stored inputs through `predict` and `decide` alone: ([radar_fut messages], [decide messages]).
    lazy: `radar_fut` has no consumer but `decide` (its message is then never built on the way)."""
    from chris_integrator.mixer import MessageClock, Mixer, StreamSpec
    from chris_integrator.plugins.decide import Decide
    from chris_integrator.plugins.predict import Predict
    m = Mixer(MessageClock())
    ch = m.add_input("golden", [StreamSpec(s, "raw", "sample") for s in INPUTS])
    m.add(Predict())
    m.add(Decide(dict(decide_cfg or {}, rate_hz=1000.0)))
    fut, dec = [], []
    m.add_output("out", lambda e: (fut if e.stream == "radar_fut" else dec).append(e.msg),
                 streams=("decide",) if lazy else ("radar_fut", "decide"))
    for stream, msg in rows:
        ch.deliver(stream, msg)
    return fut, dec


def digest(msg):
    return hashlib.sha256(json.dumps(msg, separators=(",", ":"), sort_keys=True).encode()).hexdigest()[:16]


def plain(o):
    return json.loads(json.dumps(o))


def make():
    out = {"keep": KEEP, "cases": []}
    for name, seed, night, a, b in CASES:
        rows = record_inputs(name, seed, night, a, b)
        fut, dec = replay(rows)
        _, dec_cv = replay(rows, {"futures": "cv"})
        out["cases"].append({"scenario": name, "seed": seed, "night": night, "inputs": rows,
                             "radar_fut_digest": [digest(f) for f in fut], "radar_fut": plain(fut[::KEEP]),
                             "decide": plain(dec), "decide_cv": plain(dec_cv)})
        print("%-18s inputs %5d, radar_fut %4d, decide %4d (levels up to %d), cv %4d" % (
            name, len(rows), len(fut), len(dec), max([o["level"] for d in dec for o in d["objects"]] or [0]), len(dec_cv)))
    return out


if __name__ == "__main__":
    g = make()
    with gzip.GzipFile(OUT, "wb", compresslevel=9, mtime=0) as f:
        f.write(json.dumps(g, separators=(",", ":")).encode())
    print("%s: %.0f kB" % (os.path.basename(OUT), os.path.getsize(OUT) / 1e3))
