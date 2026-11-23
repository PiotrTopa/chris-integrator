"""Fixed inputs of the predictor's tests (tests/test_predict.py) and of its golden data
(tests/data/make_predict_golden.py -> tests/data/predict_golden.json): a small made-up traffic scene as radar
messages, from additions and multiplications only (the same numbers on every machine)."""
NS = 1_000_000_000


def obj(i, x, y, vx, vy, label="car", ps=0.02, std=(1.0, 0.3, 0.6, 0.3), **kw):
    return dict({"id": i, "state": "confirmed", "exist": 1.0, "x_m": round(x, 3), "y_m": round(y, 3), "vx_mps": vx,
                 "vy_mps": vy, "ax_mps2": 0.0, "ay_mps2": 0.0, "std": list(std), "p_stationary": ps,
                 "motion": "moving" if ps < 0.3 else "stationary" if ps > 0.7 else "unknown",    # as the radar says it
                 "class": {"label": label} if label else None, "flags": []}, **kw)


def msg(t, objs, v=14.0, w=0.0, a=0.0):
    return {"type": "radar", "api": 1, "ts_mono_ns": int(round((100.0 + t) * NS)), "frame": "car", "objects": objs,
            "ego": {"speed_mps": v, "yaw_rate_rps": w, "accel_mps2": a, "stands": False, "ground_frame": 0}}


def context(arm=True):
    """A straight two-way road with a lane of the car's direction on its right, a side road on the right 60 m on."""
    c = {"centre": [[float(x), 0.0] for x in range(0, 125, 5)], "lane_width_m": 3.5, "n_left": 0, "n_right": 1,
         "oncoming": 0.9, "park_right": 0.5, "park_left": 0.0, "quality": 0.9, "source": "test"}
    if arm:
        c["arms"] = [{"pts": [[60.0, -7.0 - 5.0 * k] for k in range(9)], "side": "right", "priority": -1}]
        c["crossings"] = [{"s_m": 50.0, "half_m": 2.0}]
    return c


def scene(t):
    """The scene at time t (s): the car at 14 m/s (the frame moves with it; objects' places are given in it).
    1 a lead; 2 a car in the right lane that moves over into the car's lane from t = 1 s; 3 an oncoming car; 4 a
    parked car; 5 a pedestrian standing at the kerb by the crossing; 6 a car on the side road driving to the
    junction; 7 a car behind the lead, closing on it; 8 a cyclist at the lane's edge; 9 a pedestrian crossing."""
    lat = 0.9 * max(0.0, t - 1.0)
    return [obj(1, 45.0 - 1.0 * t, 0.0, 13.0, 0.0),
            obj(2, 15.0 + 0.0 * t, -3.5 + min(lat, 3.5), 14.0, 0.9 if 1.0 < t < 1.0 + 3.5 / 0.9 else 0.0),
            obj(3, 120.0 - 26.0 * t, 3.5, -12.0, 0.0),
            obj(4, 40.0 - 14.0 * t, -8.3, 0.0, 0.0, ps=0.95),
            obj(5, 50.0 - 14.0 * t, -5.9, 0.0, 0.0, "human", ps=0.9, std=(0.8, 0.3, 0.4, 0.3)),
            obj(6, 61.5 - 14.0 * t, -40.0 + 8.0 * t, 0.0, 8.0),
            obj(7, 22.0 + 2.0 * t, 0.1, 16.0, 0.0),
            obj(8, 30.0 - 9.0 * t, -1.4, 5.0, 0.0, "bike", std=(0.8, 0.25, 0.5, 0.25)),
            obj(9, 70.0 - 14.0 * t, -2.0 + 1.4 * t, 0.0, 1.4, "human", std=(0.8, 0.3, 0.4, 0.3))]


def run(predictor, T=3.0, dt=0.05, ctx=True, keep=None):
    """The scene through a predictor; returns the last (message, prediction), or those of the frames in `keep`."""
    out, last = [], None
    c = context() if ctx is True else ctx
    n = int(round(T / dt))
    for f in range(n + 1):
        t = f * dt
        m = msg(t, scene(t))
        last = (m, predictor.update(m, c))
        if keep is not None and f in keep:
            out.append(last)
    return out if keep is not None else last
