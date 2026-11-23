#!/usr/bin/env python3
"""Records tests/data/predict_golden.json: what the predictor (native/src/predict.c) gives on the fixed scene of
tests/predict_cases.py, for tests/test_predict.py::test_golden and the target's self-test
(chris_integrator/native/_predict_selftest.py). Run after a deliberate change of the predictor's arithmetic:

    python3 tests/data/make_predict_golden.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

import predict_cases as K  # noqa: E402
from chris_integrator.predict import Predictor  # noqa: E402

FRAMES = (10, 30, 50, 60)


def snapshot(pred):
    """A prediction as plain numbers, keyed by object id and intent slot (the order of futures of equal
    probability is not part of the result)."""
    out = {"ego": {pred.names[int(pred.ego_m[k])]: [round(float(pred.ego_p[k]), 6)] + np.round(pred.ego_traj[k, 3::6], 4).ravel().tolist()
                   for k in range(pred.ego_n)}, "objects": {}}
    for i, oid in enumerate(pred.ids):
        o = {"lane": np.round(pred.lane[i], 6).tolist(), "intent": np.round(pred.intent[i], 6).tolist(), "futures": {}}
        for k in range(int(pred.n_modes[i])):
            o["futures"][str(int(pred.slot[i, k]))] = [pred.names[int(pred.m[i, k])], round(float(pred.p[i, k]), 6),
                                                       round(float(pred.p0[i, k]), 6)] + np.round(pred.traj[i, k, 3::6], 4).ravel().tolist()
        out["objects"][str(oid)] = o
    return out


def make():
    res = K.run(Predictor({"selfcheck": {"enabled": False}}), keep=FRAMES)
    return {"frames": list(FRAMES), "snapshots": [snapshot(p) for _, p in res]}


if __name__ == "__main__":
    g = make()
    with open(os.path.join(HERE, "predict_golden.json"), "w") as f:
        json.dump(g, f, separators=(",", ":"))
    print("recorded %d frames, %d objects" % (len(g["frames"]), len(g["snapshots"][-1]["objects"])))
