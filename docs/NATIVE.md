# The native core

On a slow core (the reference deployment runs chris-integrator on a phone's little core) the price of Python
is the budget, so the arithmetic and the text work of the per-message hot path are written in C: one translation
unit, `chris_integrator/native/src/chris_native.c` (it includes `decide.c`, the decision layer's part), compiled to one module, `chris_integrator/native/_chris_native.abi3.so`.
The module is part of the package: it is required, and every function in it exists once. There is no second,
pure-Python implementation of what is native, and no switch between two.

    python -m chris_integrator.native      the module in use, and its self-test on this machine
    GET /v2/health  ->  "native": {"active": true, "implementation": "native", "parts": [...], "version": 8, "module": "..."}
    the daemon's first log lines: "native core version 8 (_chris_native.abi3.so; json, round, interp, ekf, lanes, radar, decide)"

A package without the module (or with one of another version) does not import: the error says that the native
core is not built, does not load, or is of the wrong version, and how to build it (below).

## The rules

* **One implementation.** What is native is written in C only; the Python around it prepares the arrays and
  takes the results. A part that is moved into C loses its Python code in the same change.
* **What is not ours stays CPython's.** Where the C code cannot be certain to give CPython's result for an input
  (a type the JSON writer does not know, a number beyond the reader's exact path, a float next to a decimal tie
  in `round`), it hands that input to CPython's own function: `json.dumps`, `json.loads`, `builtins.round`. That
  is the standard library deciding, not a second implementation of ours.
* **Exact where it can be.** Elementwise steps are written in the order numpy (or Python) computes them, with
  plain IEEE arithmetic (`-ffp-contract=off`: no fused multiply-add), so they give the same bits. Sums of
  products are added in index order: correct to the last bit or two, not the bits a BLAS gives ("to rounding").
* **Held to known results.** `tests/test_native.py` holds every function: JSON and rounding against CPython's
  json and round; the rest against what the pure-Python references gave before they were removed, frozen as
  golden data (`tests/native_cases.py` makes the inputs, `tests/data/native_golden.json.gz` holds the results:
  the digest of every exact result, the values where a result agrees to rounding). A change to a native function
  that changes a result fails those tests; a deliberate change records new golden data and says why.

## What each part does, and what "the same" means

| part | native function | what it gives | held to |
|---|---|---|---|
| `json` | `dumps(obj)` | `json.dumps(obj, separators=(",", ":"))`, byte for byte; `json.dumps` itself for what the writer leaves to it | `json.dumps` |
| `json` | `loads(bytes or str)` | `json.loads`: the same objects (types, values, sign of zero, key order); `json.loads` itself for what the reader leaves to it, so errors are CPython's | `json.loads` |
| `json` | `trim_road(line, ...)` (fastjson) | the `road` line without the members the default stack does not read, cut from the bytes | golden: the bytes |
| `json` | `slim_road_line(road)` (io/recorder) | the recorder's reduced `road`, written in one pass: lists rounded (`round(v, n)`, None stays null), keys as json converts them | golden: the text |
| `round` | `round(x, n)`, `round_list(list, n)` | `builtins.round`: the same value and type | `builtins.round` |
| `interp` | `interp_all(xs, ys, qs)` (trackers/view) | the view's table walk | golden: values and types |
| `interp` | `view_prep`, `view_blend` (trackers/view) | a drawn line's points, reach and jump (CPython's compensated `sum`); its two smoothing stages | golden |
| `lanes` | `lanes_measure`, `lanes_predict`, `lanes_fit`, `lanes_apply`, `round_lists` (trackers/lanes) | the lane tracker's array steps | golden: the bits |
| `lanes` | `geom_centre`, `geom_reach` (trackers/roadgeom) | the road geometry's centre line from the camera and its reach (a configured number stays its own object) | golden: the bits |
| `ekf` | `ekf_project`, `ekf_update`, `ekf_sym` (core/ekf) | one scalar row of the filter | golden: the bits for rows of one state index (most rows); the projection to rounding for rows over several states |
| `ekf` | `ekf_predict` (core/ego) | the process model | golden: state and transition to the last bit (sin, cos, exp of the C library), covariance to rounding |
| `ekf` | `ekf_rts` (core/ego) | one smoother step | golden: to rounding |
| `ekf` | `ekf_run` (core/ekf `Filter._run`) | the filter's pass over its queued entries (re-runs after late rows included): prediction, every row's projection, gate, Student-t weight and correction, the entry's symmetrisation; each row's first innovation to its class's window and trend; Python only for the rows' models h(x), a row's callback, the integrator's hooks when a verdict changes or holds | golden: every bit over an arithmetic model; over the estimator's model the decisions exactly, the states to rounding (sin, cos, exp) |
| `ekf` | `NisWindow` (core/monitors `NisMonitor` subclasses it), `Trend` (core/fdi) | a row class's innovation window (median from an ordered copy; statistics.median while a NaN is in it) and its signed trend | golden: the window every bit; the trend's verdicts exactly, its sums to rounding (exp) |
| `ekf` | `SelfWindow` (core/monitors `SelfCheck` subclasses it), `StreamState` (core/fdi `Stream` subclasses it) | a source's stated over characterised uncertainty on evidence time (two medians from ordered copies); a raw stream's run, repeat share, noise and wild samples (the texts and the held findings stay in Python) | golden: every bit |

What the estimator's native parts do with inputs the stack does not produce (where the removed Python differed):
a row whose innovation variance is exactly zero gives an infinite or NaN statistic instead of Python's
ZeroDivisionError; a row's prediction, value and variance are read as floats (`float()`), so a numpy scalar
becomes a float of the same value: the monitors hold floats where the Python held numpy scalars (the same bits;
only `round()` of a value next to a decimal tie could differ, numpy rounding by its own rule); a state that is
not float64 is made a float64 copy; a stream's value beyond 2^53 is read as the nearest double.
| `radar` | `radar_scan` (one scan's association: p_detect, coverage, the map's density, gate, likelihoods of the hypotheses, the box feature, the pairings' probabilities by loopy belief propagation, the assignment), `radar_update` (every pair of a scan: hypotheses, free estimate, NIS, model probabilities, existence increment, combination), `radar_common`, `radar_check`, `radar_register` (radar/tracker) | the tracker's arithmetic for a scan | golden: pairs and assignment the same; numbers to rounding (below) |
| `radar` | `radar_advance`, `radar_peek`, `radar_peek_pos`, `radar_life`, `radar_merge`, `radar_combine_rows` (radar/tracker, on the rows of the tracker's state array) | the motion model and the objects' upkeep | golden: to rounding |
| `radar` | `radar_frame` (radar/box) | the message's objects: their numbers, flags with their hysteresis, the dicts | golden: to rounding (printed digits the same but for a last digit now and then); digests of three drives' messages (x86-64) |
| `radar` | `radar_h` (radar/sensors: camera kinds b, rb, rbh, bt with and without the scale-rate row, microwave radar, position), `radar_marginals`, `radar_assign`, `radar_predict1`, `radar_still1`, `radar_transition1` (radar/models: one hypothesis) | measurement functions, pairings' probabilities, assignment, one hypothesis' step | golden: marginals and assignment the bits, measurement functions to the last bit of libm |

The lane tracker's steps are exact because its arithmetic is elementwise, and its two means over the gate points
are added in sequence, as numpy adds a masked selection. The road geometry's arc length is a sequential sum of
`hypot`, as `np.cumsum` takes it.

"To rounding": the C code adds sums of products in index order with plain IEEE arithmetic; numpy's BLAS (which
gave the frozen results) adds them in an order of its own, with fused multiply-adds on ARM. Both are correct to
the last bit or two of a double. After the monitors and trackers, on recorded drives, that shows as a handful of
messages per drive in which a number that rounds to zero carries the other sign (`-0.0` / `0.0`) and, rarely, a
last printed digit; lane report L50 has the counts.

| `decide` | `decide_eval` (decide/threat.py), `decide_inst_prob`, `decide_draws` (`native/src/decide.c`, included by chris_native.c) | per object, mode and car mode: collision probability over the horizon (quasi-Monte Carlo on fixed Halton draws, or the exact Gaussian over the footprints' Minkowski sum per time), required braking / steering / combined demand, threat numbers, time-to-brake / -steer, time to collision, headway, levels | closed forms (required deceleration v^2 / 2d, latest onset), the exact integral against products of normal distributions and against sampling; golden `tests/data/decide_golden.npz` (`python3 tests/data/make_decide_golden.py`): to rounding, bisected demands and times within their tolerances (0.06), Monte Carlo shares within 0.01 |

The decision layer was written native from the start (no Python reference was removed): what holds it are the
closed forms and the independent references of `tests/test_decide_native.py`, and its golden data freeze it.

### The radar: native only

Since native version 6 the radar (`chris_integrator/radar`) has no Python implementation of its arithmetic either:
the motion model, the measurement functions, a scan's association and update, the objects' upkeep and the
message's objects are C only. Python keeps the objects' life (states,
votes, records, merging), the message's assembly around the objects, and the glue. The objects' numbers live in one
float64 array of the tracker (`tracker.State`, a row per object), which the native calls read and write in place.

How the native radar was held to its Python reference before that was removed (lane report native-radar2): the
measurement functions, marginals and assignment gave the reference's bits (on one machine: the frozen measurement
functions hold to the last bit of libm elsewhere); everything else the reference's
formulas with the products summed in index order and small systems solved by elimination with partial pivoting
(the reference: BLAS, LAPACK): results to rounding. Scan by scan on simulated drives, native and reference on
copies of the same tracker agreed to 2e-9 of each block's size (a covariance update that takes most of a variance
away subtracts nearly equal numbers); whole drives gave the same messages but for a last printed digit in a
handful of values (12 drives, 1.38 M values: 9; one object whose filter amplifies rounding-level differences).
What the reference gave is frozen: `tests/data/radar_golden.npz` (the inputs of chosen calls of every native radar
function on simulated drives and the reference's answers, plus random and edge cases), replayed by
`test_radar_golden` with those tolerances; `test_radar_messages_digests` holds three simulated drives' radar
messages byte for byte on the development machine; `python -m chris_integrator.native` checks a few of the
reference's answers on a target (`native/_radar_selftest.py`). A state that is not finite is computed like any
other (no hand-over to a reference any more): NaN and inf go where the arithmetic takes them, and the upkeep
drops such an object as before.

### Inputs the C code takes, and what it does with others

* Arrays: float64 (index arrays: the platform's 64-bit integer), C-contiguous. The Python around a call makes
  lists and views into such arrays first (`np.ascontiguousarray`).
* Numbers in lists (the view, the road geometry's settings): floats (and float subclasses, read as their value)
  and ints up to 2^53, bools as ints. Anything else, and an int beyond 2^53, is a `TypeError`. (The removed
  Python computed with big ints exactly and returned numpy scalars as numpy scalars; neither reaches these
  functions in the stack.)
* A lane line's `y` / `y_std` (`lanes_measure`): a list or a tuple of n items that `float()` converts (ints,
  bools, numeric text), all finite; else the line is not taken (no `y`) or its deviations default (no
  `y_std`); an int beyond the float range raises `OverflowError`, as numpy does. Another sequence type (a numpy
  array) is not read as one: the line is not taken.
* `trim_road` takes `bytes` (anything else is made `bytes` first) and returns the line itself when there is
  nothing to cut.

### Why the exact parts of the JSON work are exact

* **Writing a float.** `repr(x)` is the shortest decimal string that reads back as `x`. If `x` equals
  `k / 10^n` for an integer `k` below 10^15 (tested by one exact division), the string of `k` with `n` decimals
  is that shortest string: decimals of 15 significant digits or fewer and doubles correspond one to one. Every
  number the stack rounds before it writes is of this kind. Everything else (17 digits, exponent forms) goes
  through CPython's own `PyOS_double_to_string`.
* **Writing a rounded float** (`slim_road_line`): `rint(x * 10^n)` is the integer `k` of that string, so the text
  is `k` with `n` decimals and the trailing zeros dropped, written without a second test; next to a decimal tie,
  or below 1e-4 (where `repr` writes an exponent), `round` and the float writer decide.
* **Reading a float.** A decimal with a mantissa up to 2^53 and at most 22 decimals is one exact division of
  two exact doubles, correctly rounded: what `strtod` returns. Everything else goes through
  `PyOS_string_to_double`.
* **`round(x, n)`.** `round` rounds correctly in decimal. `rint(x * 10^n) / 10^n` gives the same double except
  next to a decimal tie; within 1e-6 of a tie (and beyond 1e9, and for anything that is not a plain float)
  `builtins.round` itself is called.

### Memory safety of the parser

`loads` reads external input (the daemons' lines, the vehicle feed). It is a recursive-descent reader over a
(pointer, end) pair: every read is checked against `end`, nesting is limited to 64 levels (deeper input goes
to `json.loads`), strings are validated as UTF-8 by CPython's decoder, numbers longer than 63 characters or
19 digits go to `json.loads`. `tests/test_native.py::test_loads_survives_damaged_input` feeds it 40000 cut
and corrupted lines and requires `json.loads`' result or an error; build with `SANITIZE=1` to run the suite
under the address and undefined-behaviour sanitizers:

    SANITIZE=1 OUT=/tmp/asan sh native/build.sh
    cp /tmp/asan/_chris_native.abi3.so chris_integrator/native/
    LD_PRELOAD=$(cc -print-file-name=libasan.so) ASAN_OPTIONS=detect_leaks=0 python -m pytest tests/test_native.py

## Building

The module uses the CPython limited API (`abi3`): one binary loads in every CPython from 3.11 on, for the
machine type and C library it was built for. It needs a C compiler and the Python headers to build, nothing
to run but libm. No numpy headers: arrays pass through the buffer protocol.

    sh native/build.sh                        this machine -> chris_integrator/native/_chris_native.abi3.so
    pip install .                             builds it (a compiler and the Python headers are needed)
    sh native/cross-build.sh                  another platform, in a container (default: linux/arm64, Alpine: musl)

Flags that matter: `-ffp-contract=off` (no fused multiply-add: it would change bits that are otherwise
numpy's), no `-ffast-math`, no `-march`. The code is plain C99. `VERSION` in the module and in
`chris_integrator/native/__init__.py` must agree: a change of a function's contract raises both.

### A target that must not be touched (the reference phone)

The phone's operating system is pinned and has no Python headers. Nothing is compiled on it:

    sh native/cross-build.sh
    # -> native/dist/linux-aarch64-musl/_chris_native.abi3.so  and BUILD.txt (compiler, image, checksums, test result)

builds in an `alpine:edge` arm64 container on the development machine (docker with qemu-user binfmt), runs
the self-test and `tests/test_native.py` there against the container's numpy, and leaves the artefact in
`native/dist/`, which travels with the source tree that is copied to the target. `scripts/install.sh` then:

1. takes `native/dist/linux-<machine>-<libc>/_chris_native.abi3.so` for the target (or `NATIVE=/path` names a
   file, or the tree holds a module built in place); without one it stops before it changes anything;
2. puts a copy of the package with that module into a staging directory and runs `python3 -m
   chris_integrator.native` and the daemon's imports there with the target's interpreter;
3. replaces the installed package with the staged one only when that passes; otherwise it stops, and the
   installed package stays as it was. The log line is
   `native core: linux-aarch64-musl, sha256 ..., from ...; self-test ok`.

The script never compiles.

### Wheels

`pip wheel .` gives `chris_integrator-<v>-cp311-abi3-<platform>.whl` with the module. For a release:
cibuildwheel with `CIBW_BUILD=cp311-*` (abi3 covers later versions), manylinux and musllinux, x86_64 and
aarch64; plus the sdist, from which `pip install` builds.

## Tests

    python -m pytest tests                          the whole suite (builds first: sh native/build.sh)
    python -m pytest tests/test_native.py           every native function against its golden results
    tools/cost_bench.py run REC --dump A.gz         every message of every stream of a recorded drive;
    tools/cost_bench.py diff A.gz B.gz              two trees (before / after a change) compared

## A native plug-in

The mixer's plug-in contract (`mixer/plugin.py`, DESIGN.md 14.3) is about streams, not about languages:
a box declares `name`, `inputs`, `outputs`, `DEFAULT_CFG` and answers `on_input(stream, msg, ctx)`. Three
ways to put compiled code behind it, in order of cost:

1. **A Python box with a native kernel** (what the estimator and the display box do): the box stays the
   declared, replaceable unit in Python; its arithmetic calls the native core. The contract and the wiring
   do not change; a replay is deterministic.
2. **A box written as an extension type** (C, C++, Rust with PyO3, Cython): any importable class with the
   attributes above and `on_input` is a plug-in; `mixer.json` names it (`{"plugins": [{"class":
   "package.module:Class"}]}`, see `stack.py`). Messages are the bus' dicts, so such a box pays the
   conversion at its edges and nothing else.
3. **A box in another process**, in any language: `mixer.remote.RemotePlugin` speaks the same declarations
   as JSON lines over a pipe or a Unix socket (`{"type":"plugin","contract":1,...}` handshake). The hop costs
   a serialisation each way (`tools/hop_cost.py`); for boxes that are not tied to a camera frame
   (`sync=False`).

A part moved into C follows the rules above: its Python goes, its results are frozen as golden data first.

## Licences

The native core is original code under the package's MIT licence. Nothing is vendored; it links libm and
the CPython stable ABI only.
