"""The native core: the hot path's arithmetic and text work, compiled (one C file, `src/chris_native.c`; build:
docs/NATIVE.md). It is part of the package: where it is missing or of another version, importing the package
fails here and says how to build it. Each function exists once, in C; what is not ours (`json`, `round` of
CPython) is called where the C code leaves an input to it.

    dumps(obj)            json.dumps(obj, separators=(",", ":")): the native writer; json.dumps for what it
                          leaves to json (a type it does not know, a big int, a key json converts)
    loads(raw)            json.loads: the native reader; json.loads for what it leaves to json
    round(x, n)           builtins.round: the same results (builtins.round itself next to a decimal tie)
    round_list(list, n)   [v if v is None else round(v, n) for v in list]
    core                  the module: the estimator's filter (core/ekf.py, core/ego.py), the lane tracker
                          (trackers/lanes.py), the road geometry (trackers/roadgeom.py), the view
                          (trackers/view.py), the road line (fastjson.py, io/recorder.py)
    radar                 the radar (chris_integrator/radar: motion model, measurement functions, association, update, upkeep, the message's objects)
"""
import functools
import json
import os

VERSION = 9                    # of the compiled module's functions: the package needs this one
PARTS = ("json", "round", "interp", "ekf", "lanes", "radar", "predict", "decide")
_HOW = ("build it: `sh native/build.sh` in the source tree (this machine: a C compiler and the Python headers), "
        "`sh native/cross-build.sh` for another platform (docs/NATIVE.md), or `pip install .`")

try:
    from . import _chris_native as _n
except ImportError as e:
    _there = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if f.startswith("_chris_native")]
    raise ImportError("chris-integrator: the native core %s (%s). It is required: %s" % (
        "does not load here" if _there else "is not built", str(e).splitlines()[0][:200], _HOW)) from e
if getattr(_n, "VERSION", 0) != VERSION:
    raise ImportError("chris-integrator: the native core is version %s, this package needs version %d: rebuild it. %s"
                      % (getattr(_n, "VERSION", "?"), VERSION, _HOW))

core = radar = predict = decide = _n
Fallback = _n.Fallback                 # the C code leaves an input to json (dumps, loads)
_dumps_py, _loads_py = json.dumps, json.loads
_SEP = (",", ":")
_nd, _nl = _n.dumps, _n.loads
_n.set_json_dumps(functools.partial(_dumps_py, separators=_SEP))


def dumps(obj):
    try:
        return _nd(obj)
    except Fallback:
        return _dumps_py(obj, separators=_SEP)


def loads(raw):
    try:
        return _nl(raw)
    except Fallback:
        return _loads_py(raw)


round = _n.round
round_list = _n.round_list


def status():
    """For `/v2/health`."""
    return {"active": True, "implementation": "native", "parts": list(PARTS), "version": _n.VERSION,
            "module": os.path.basename(getattr(_n, "__file__", "") or "")}
