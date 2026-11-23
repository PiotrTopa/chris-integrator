"""The self-test of the native core's `predict` part on a target (python -m chris_integrator.native): a small fixed
scene through the predictor, held to what the development build gave (to 2e-5: sin, cos, exp and erfc are the C
library's). tests/test_predict.py holds much more; this says "the binary on this machine predicts".

Recorded with:  python3 -m chris_integrator.native._predict_selftest record"""
import sys

NS = 1_000_000_000


def _scene(t):
    def o(i, x, y, vx, vy, label="car", ps=0.02):
        return {"id": i, "state": "confirmed", "exist": 1.0, "x_m": x, "y_m": y, "vx_mps": vx, "vy_mps": vy, "ax_mps2": 0.0,
                "ay_mps2": 0.0, "std": [1.0, 0.3, 0.6, 0.3], "p_stationary": ps, "class": {"label": label}, "flags": []}
    lat = 0.9 * max(0.0, t - 0.5)
    return [o(1, 60.0 - 8.0 * t, 0.0, 6.0, 0.0), o(2, 30.0, 0.0, 14.0, 0.0), o(3, 15.0, -3.5 + min(lat, 3.5), 14.0, 0.9),
            o(4, 110.0 - 26.0 * t, 3.5, -12.0, 0.0), o(5, 45.0 - 14.0 * t, -6.2, 0.0, 0.0, "human", 0.9),
            o(6, 40.0 - 14.0 * t, -8.5, 0.0, 0.0, "car", 0.95)]


def result():
    from ..predict import Predictor
    P = Predictor({"selfcheck": {"enabled": False}})
    ctx = {"centre": [[float(x), 0.0] for x in range(0, 125, 5)], "n_left": 0, "n_right": 1, "oncoming": 0.9,
           "park_right": 0.5, "quality": 0.9}
    pred = None
    for f in range(31):
        t = f * 0.05
        pred = P.update({"type": "radar", "ts_mono_ns": int(round((100.0 + t) * NS)), "objects": _scene(t),
                         "ego": {"speed_mps": 14.0, "yaw_rate_rps": 0.0, "accel_mps2": 0.0, "ground_frame": 0}}, ctx)
    out = []
    for i in range(len(pred.ids)):
        k = int(pred.p0[i].argmax())
        out += [float(pred.lane[i].max()), float(pred.p0[i, k]), float(pred.p[i, k])] + pred.traj[i, k, 11].tolist() + pred.traj[i, k, 21].tolist()
    return out + pred.ego_traj[0, 21].tolist()


EXPECT = [0.942390385, 1.0, 1.0, 65.64, 0.0, 2.494128455, 0.390922441, 0.0, 5.88, 95.04, 0.0, 10.44676006, 0.392280446, 0.0, 5.88, 0.942391108, 0.639331275, 0.745425259, 71.16, 0.0, 2.494128455, 0.390922441, 0.0, 13.72, 139.76, 0.0, 10.44676006, 0.704920213, 0.0, 13.72, 0.942112266, 0.964669194, 0.433275983, 38.78125539, -0.08391372, 2.493382816, 0.400057716, 0.0, 5.4954313, 65.34055682, 0.0, 10.445137649, 0.392280446, 0.0, 5.295234207, 0.942095404, 1.0, 1.0, 35.72, 3.5, 2.494128455, 0.390922441, 0.0, 11.76, -23.08, 3.5, 10.44676006, 0.890130468, 0.0, 11.76, 0.938312973, 0.930232558, 0.930232558, 24.0, -6.2, 1.011187421, 0.335410197, 0.0, 0.0, 24.0, -6.2, 1.077032961, 0.5, 0.0, 0.0, 0.884504698, 1.0, 1.0, 19.0, -8.5, 1.011187421, 0.335410197, 0.0, 0.0, 19.0, -8.5, 1.077032961, 0.5, 0.0, 0.0, 112.0, 0.0, 14.0]


def check():
    """Problems found (an empty list: this build predicts what the development build predicted)."""
    got = result()
    if len(got) != len(EXPECT):
        return ["predict: %d numbers, %d recorded" % (len(got), len(EXPECT))]
    bad = [i for i, (u, v) in enumerate(zip(got, EXPECT)) if not abs(u - v) <= 2e-5 * max(1.0, abs(v))]
    return ["predict differs from the recorded results (%d of %d numbers, first: %r against %r)"
            % (len(bad), len(got), got[bad[0]], EXPECT[bad[0]])] if bad else []


if __name__ == "__main__" and sys.argv[1:] == ["record"]:
    import os
    import re
    vals = "EXPECT = [" + ", ".join(repr(round(v, 9)) for v in result()) + "]"
    here = os.path.abspath(__file__)
    with open(here) as f:
        src = f.read()
    with open(here, "w") as f:
        f.write(re.sub(r"^EXPECT = \[.*\]$", lambda m: vals, src, count=1, flags=re.M))
    print("recorded %d numbers" % len(result()))
