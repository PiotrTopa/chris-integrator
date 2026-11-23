"""Adds to tests/data/native_golden.json.gz what the pure-Python SelfCheck and fdi.Stream gave for the cases of
tests/native_cases.py (`cases_selfcheck`, `cases_stream`). Ran once, with CHRIS_MON_PY=1, on the tree that still
held them (lane estimator-native, the commit that added this script); kept as the record of how the entries were
made. It does not run on a later tree: the Python versions are gone.

    CHRIS_MON_PY=1 python3 tests/data/make_native_golden_monitors.py
"""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

import native_cases as K  # noqa: E402
from chris_integrator.core import fdi as FD  # noqa: E402
from chris_integrator.core import monitors as MO  # noqa: E402

assert MO.MON_PY, "run with CHRIS_MON_PY=1: the Python monitors are what is recorded"

gold = K.load_golden()
for name, cases, fn in (("selfcheck", K.cases_selfcheck(), lambda *c: K.selfcheck_run(MO.SelfCheck, *c)),
                        ("stream", K.cases_stream(), lambda *c: K.stream_run(FD.Stream, *c))):
    cases = list(cases)
    gold[name] = {"inputs": K.digest(cases), "out": [K.digest(K.outcome(fn, *c)) for c in cases]}
    print(name, len(cases))
with gzip.open(K.GOLDEN, "wt", compresslevel=9) as f:
    json.dump(gold, f, separators=(",", ":"))
print(os.path.getsize(K.GOLDEN), "bytes")
