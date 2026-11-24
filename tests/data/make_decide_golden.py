#!/usr/bin/env python3
"""Golden data of the decision layer's native numerics (native/src/decide.c `decide_eval`): the inputs of every
frame of the fixture scenarios (chris_integrator/decide/fixtures.py) and of degraded futures, and the results the
native core gave when they were recorded. tests/test_decide_native.py replays them.

    python3 tests/data/make_decide_golden.py          # -> tests/data/decide_golden.npz (records new results)

Record anew only for a deliberate change of the numerics, and say why in the commit."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from chris_integrator.decide import fixtures as FX, threat as TH  # noqa: E402
from chris_integrator.native import decide as ND  # noqa: E402


def cases():
    out = []
    for name in FX.NAMES:
        fr = FX.frames(name, hz=1.0)
        for msg, _ in fr:
            out.append((name, msg, {}))
            # the same frame through the degraded adapter (no futures), and with the exact per-time integral
            out.append((name + ":cv", {k: v for k, v in msg.items() if k not in ("fut", "ego_fut")} |
                        {"objects": [{k: v for k, v in o.items() if k != "fut"} for o in msg["objects"]]}, {}))
        out.append((name + ":analytic", fr[len(fr) // 2][0], {"pc_method": "analytic"}))
    return out


def inputs(msg, tcfg):
    """The evaluation's arrays of one frame, as the native core's frame makes them (threat.Explained)."""
    tm = TH.ThreatModel(tcfg)
    ex = tm.explain(msg)
    cfg = tm.native_cfg()[:34]
    cfg[3], cfg[5], cfg[6] = float(msg["ego"]["speed_mps"]), ex.a_avail[0], ex.a_avail[1]
    return ex.t, ex.ego, ex.ohdr, ex.modes, cfg, tm.thr


def run(inp):
    t, ego, ohdr, modes, cfg, thr = inp
    t, ego, ohdr, modes = (np.ascontiguousarray(a) for a in (t, ego, ohdr, modes))
    pair = np.empty((len(modes) * len(ego), TH.PF))
    obj = np.empty((len(ohdr), TH.OF))
    ND.decide_eval(t, ego, ohdr, modes, cfg, thr, pair, obj)
    return pair, obj


def main():
    arrays = {}
    names = []
    for q, (name, msg, tcfg) in enumerate(cases()):
        if not msg["objects"]:
            continue
        inp = inputs(msg, tcfg)
        pair, obj = run(inp)
        for key, a in zip(("t", "ego", "ohdr", "modes", "cfg", "thr", "pair", "obj"), inp + (pair, obj)):
            arrays["%d_%s" % (q, key)] = a
        names.append("%d:%s" % (q, name))
    arrays["names"] = np.array(names)
    np.savez_compressed(os.path.join(HERE, "decide_golden.npz"), **arrays)
    print("decide_golden.npz: %d cases" % len(names))


if __name__ == "__main__":
    main()
