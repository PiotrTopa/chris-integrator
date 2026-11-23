"""Adds to tests/data/native_golden.json.gz what the pure-Python filter pass and innovation monitors gave for the
cases of tests/native_cases.py (`cases_nis`, `cases_trend`, `cases_filter_pass`). Ran once, with CHRIS_EKF_PY=1,
on the tree that still held them (lane estimator-native, the commit that added this script); kept as the record
of how the entries were made. It does not run on a later tree: the Python versions are gone.

    CHRIS_EKF_PY=1 python3 tests/data/make_native_golden_estimator.py
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

import native_cases as K  # noqa: E402
from chris_integrator.core import ekf as KF  # noqa: E402

assert KF.PY_PASS, "run with CHRIS_EKF_PY=1: the Python pass and monitors are what is recorded"

from chris_integrator.core import fdi as FD  # noqa: E402
from chris_integrator.core import monitors as MO  # noqa: E402

gold = K.load_golden()

cases = list(K.cases_nis())
gold["nis"] = {"inputs": K.digest(cases), "out": [K.digest(K.nis_run(MO.NisMonitor, *c)) for c in cases]}
print("nis", len(cases))

cases = list(K.cases_trend())
out, vals = [], []
for c in cases:
    highs, values = K.trend_run(FD.Trend, *c)
    out.append(K.digest(highs))
    vals.append(values)
gold["trend"] = {"inputs": K.digest(cases), "out": out, "values": vals}
print("trend", len(cases))

cases = list(K.cases_filter_pass())
out, vals = [], []
for ego, ops in cases:
    res = K.filter_pass(ego, ops, KF.Filter, KF.Row, MO.NisMonitor, FD.Trend, native=False)
    exact, values = K.filter_pass_split(ego, res)
    out.append(K.digest(exact))
    vals.append(values)
gold["filter_pass"] = {"inputs": K.digest(cases), "out": out, "values": vals}
print("filter_pass", len(cases))

with gzip.open(K.GOLDEN, "wt", compresslevel=9) as f:
    json.dump(gold, f, separators=(",", ":"))
print(os.path.getsize(K.GOLDEN), "bytes")
