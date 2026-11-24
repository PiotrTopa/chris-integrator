"""The self-test of the native core's `predict` part on a target (python -m chris_integrator.native): a small fixed
scene through the predictor, held to what the development build gave (to 2e-5: sin, cos, exp and erfc are the C
library's). tests/test_predict.py holds much more; this says "the binary on this machine predicts".

Recorded with:  python3 -m chris_integrator.native._predict_selftest record"""
import sys

NS = 1_000_000_000


def _scene(t):
    def o(i, x, y, vx, vy, label="car", ps=0.02):
        return {"id": i, "state": "confirmed", "exist": 1.0, "x_m": x, "y_m": y, "vx_mps": vx, "vy_mps": vy, "ax_mps2": 0.0,
                "ay_mps2": 0.0, "std": [1.0, 0.3, 0.6, 0.3], "p_stationary": ps, "motion": "moving" if ps < 0.5 else "stationary",
                "class": {"label": label}, "flags": []}
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


EXPECT = [0.942394544, 1.0, 1.0, 65.82, 0.0, 2.494128455, 0.390922441, 0.0, 5.94, 95.52, 0.0, 10.44676006, 0.392280446, 0.0, 5.94, 0.942395314, 0.638002298, 0.736122145, 71.58, 0.0, 2.494128455, 0.390922441, 0.0, 13.86, 140.88, 0.0, 10.44676006, 0.762594075, 0.0, 13.86, 0.942102284, 0.965322732, 0.43460112, 38.940816721, -0.08376078, 2.493382816, 0.400037913, 0.0, 5.511114202, 65.609915786, 0.0, 10.445137649, 0.392280459, 0.0, 5.318413794, 0.942106433, 1.0, 1.0, 35.36, 3.5, 2.494128455, 0.390922573, 0.0, 11.88, -24.04, 3.5, 10.44676006, 0.951508825, 0.0, 11.88, 0.738490942, 0.885169449, 0.885169449, 24.0, -6.2, 1.011187421, 0.335410197, 0.0, 0.0, 24.0, -6.2, 1.077032961, 0.5, 0.0, 0.0, 0.662644734, 0.976889325, 0.976889325, 19.0, -8.5, 1.011187421, 0.335410197, 0.0, 0.0, 19.0, -8.5, 1.077032961, 0.5, 0.0, 0.0, 112.0, 0.0, 14.0]


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
