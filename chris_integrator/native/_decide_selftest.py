"""The self-test of the native core's `decide` frame on a target (python -m chris_integrator.native): two small
fixed frames through `decide_frame` (the degraded futures along a bending path; a fixture's contract futures read
out of its message), held to what the development build gave: the futures, footprints and weights to 1e-9 (plain
arithmetic and the C library's sin, cos, atan2, hypot, pow, fmod), the collision probabilities to 0.01, the levels
as they were. tests/test_decide.py and tests/test_decide_port.py hold much more; this says "the binary on this
machine decides".

Recorded with:  python3 -m chris_integrator.native._decide_selftest record"""
import sys


def _frames():
    from ..decide import fixtures as FX, futures as FU, threat as TH
    xs = [2.0 * k for k in range(51)]
    path = FU.plan_path({"ts_mono_ns": 0, "plan": {"x": xs, "y": [-0.0015 * x * x for x in xs]}}, 0)

    def o(i, x, y, vx, vy, **kw):
        return dict({"id": i, "state": "confirmed", "x_m": x, "y_m": y, "vx_mps": vx, "vy_mps": vy, "std": [0.8, 0.3, 0.7, 0.3],
                     "class": {"label": "car"}, "flags": []}, **kw)
    objs = [o(1, 42.0, 2.4, 0.0, 0.0, motion="stationary", p_stationary=0.97, ctr=[44.2, 2.9], len=4.4, wid=1.8, hdg=0.5,
              ext_std=[0.4, 0.1, 0.4]),
            o(2, 25.0, 1.1, 9.0, 0.7, motion="moving", p_stationary=0.02, heading_rad=0.078, speed_mps=9.03),
            o(3, 18.0, -2.9, 0.4, 0.0, p_stationary=0.6, flags=["parked"], lane={"ego": 0.08, "park": 0.9, "cross": 0.02},
              **{"class": {"label": "Pedestrian"}}),
            o(4, 60.0, 4.0, -12.0, -0.5, p_stationary=0.3, hdg=[3.0, 0.05], len=[9.0, 1.0], wid={"value": 2.5, "std": 0.1},
              **{"class": {"label": "lorry"}})]
    tm = TH.ThreatModel()
    a = tm.explain({"objects": objs, "ego": {"speed_mps": 13.0, "yaw_rate_rps": 0.04, "accel_mps2": -0.5}}, path=path)
    b = tm.explain(FX.frames("cut_in", hz=2.0)[5][0])
    return a, b


def result():
    """(numbers held to 1e-9, collision probabilities, levels)"""
    import numpy as np
    exact, pc, levels = [], [], []
    for ex in _frames():
        n = len(ex.t)
        exact += np.nan_to_num(ex.ohdr, nan=-9.0).ravel().tolist() + ex.modes[:, :2].ravel().tolist()
        for q in range(5):                                       # x, y, sx, sy, rxy of every mode: three of the times
            exact += ex.modes[:, [2 + q * n + 1, 2 + q * n + n // 2, 2 + q * n + n - 1]].ravel().tolist()
        exact += ex.ego[:, [0, 1 + n - 1, 1 + 2 * n - 1, 1 + 3 * n - 1]].ravel().tolist() + list(ex.a_avail)
        pc += ex.obj[:, 0].tolist()
        levels += [r["level"] for r in ex.records] + [r["fut"] for r in ex.records] + ex.names
    return exact, pc, levels


EXACT = [0.0, 1.0, 4.4, 1.8, 0.4, 0.1, 0.12524930011386926, 0.1, 0.0, 1.0, 2.200000000000003, 0.5, 1.0, 1.0, 4.5, 1.8, 0.4, 0.1, 0.07489564381841511, 0.1, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.6, 0.6, 0.15, 0.15, 0.05343151037047744, -9.0, 1.0, 0.0, 0.0, 0.0, 3.0, 2.0, 9.0, 2.5, 1.0, 0.1, 3.318984723026542, 0.05, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.3, 0.3, 0.7, 0.7, 42.0, 42.0, 42.0, 27.248808204504375, 47.42106207654489, 69.6345984403124, 18.0, 18.0, 18.0, 60.0, 60.0, 60.0, 57.03344526434625, 30.13547812665968, 0.0304223281008441, 2.4, 2.4, 2.4, 1.2762283246796369, 3.5365067878388587, 7.438314271006906, -2.9, -2.9, -2.9, 4.0, 4.0, 4.0, 3.482137292675484, -0.021915240549006176, -1.378486833073901, 0.8008784239820674, 0.8835298523536147, 1.0965856099730655, 0.8189169677079601, 1.9241881404893857, 3.5902646142032486, 0.8008784239820674, 0.8835298523536147, 1.0965856099730655, 0.8008784239820674, 0.8835298523536147, 1.0965856099730655, 0.8189169677079601, 1.9241881404893857, 3.5902646142032486, 0.3023346655611956, 0.4802343178074637, 0.8077747210701756, 0.30923292192132457, 0.8077747210701756, 1.5297058540778354, 0.3023346655611956, 0.4802343178074637, 0.8077747210701756, 0.3023346655611956, 0.4802343178074637, 0.8077747210701756, 0.30923292192132457, 0.8077747210701756, 1.5297058540778354, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 58.451711406065655, 5.1259529188736215, 10.5, 0.15000000000000002, 58.43055504795217, 5.295651833153456, 10.5, 7.309529100000001, 6.0196122, 0.0, 2.0, 4.5, 1.8, 0.4, 0.1, -0.10419540885454678, -9.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.48, 0.19999999999999996, 0.52, 7.5, 62.5, 122.5, 7.5, 62.5, 122.5, 1.1631970541057737, 0.0, 0.0, 1.209220259843842, 1.209220259843842, 1.209220259843842, 0.47500000000000003, 1.2999999999999998, 2.1999999999999997, 0.47500000000000003, 1.2999999999999998, 2.1999999999999997, 0.33749999999999997, 0.75, 1.2, 0.33749999999999997, 0.75, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 150.0, 0.0, 25.0, 7.129417500000001, 5.871284999999999]
PC = [0.998828, 0.015625, 0.0, 0.557734, 0.660156]
LEVELS = [2, 0, 0, 3, 'cv', 'cv', 'cv', 'cv', 'stand', 'cv_path', 'stand', 'stand', 'cv_path', 4, 'contract', 'cut_in', 'keep']


def check():
    """Problems found (an empty list: this build decides what the development build decided)."""
    exact, pc, levels = result()
    if len(exact) != len(EXACT) or len(pc) != len(PC):
        return ["decide_frame: %d numbers, %d recorded" % (len(exact), len(EXACT))]
    bad = [i for i, (u, v) in enumerate(zip(exact, EXACT)) if not abs(u - v) <= 1e-9 * max(1.0, abs(v))]
    out = []
    if bad:
        out.append("decide_frame differs from the recorded futures and footprints (%d of %d numbers, first: %r against %r)"
                   % (len(bad), len(exact), exact[bad[0]], EXACT[bad[0]]))
    if any(abs(u - v) > 0.01 for u, v in zip(pc, PC)):
        out.append("decide_frame differs from the recorded collision probabilities: %r against %r" % (pc, PC))
    if levels != LEVELS:
        out.append("decide_frame differs from the recorded levels and futures: %r against %r" % (levels, LEVELS))
    return out


if __name__ == "__main__" and sys.argv[1:] == ["record"]:
    import os
    import re
    exact, pc, levels = result()
    here = os.path.abspath(__file__)
    with open(here) as f:
        src = f.read()
    for name, vals in (("EXACT", exact), ("PC", [round(v, 6) for v in pc]), ("LEVELS", levels)):
        src = re.sub(r"^%s = \[.*\]$" % name, lambda m: "%s = %r" % (name, vals), src, count=1, flags=re.M)
    with open(here, "w") as f:
        f.write(src)
    print("recorded %d numbers, %d objects" % (len(exact), len(pc)))
