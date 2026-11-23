"""Test fixtures: `radar` messages carrying the prediction contract's futures (futures.py) for scenarios with known
truth, so the decision layer is exercised on multi-modal futures before the prediction layer exists.

    for msg, truth in frames("cut_in", hz=10):        # msg: a `radar` message with fut / ego_fut; truth: per actor
        ...

A scenario is a straight road (x along it, y left), the car and actors given as functions of time (position,
velocity over the ground). At each output time T the message is in the car frame of T; each actor's futures are
its modes' motions from its state at T (constant velocity, braking to a stop, a lateral move to another offset,
stepping out), each with a probability `p0` (unconditioned: the actor does not yield) and `p` (conditioned on the
interaction), a position spread growing with the horizon, in the frame of T held fixed on the ground."""
import math

import numpy as np

T_GRID = tuple(round(0.25 * k, 2) for k in range(1, 25))       # 0.25 .. 6 s


class Actor:
    def __init__(self, aid, label, pos, vel, modes, t_on=0.0, t_off=1e9, std=(0.4, 0.3, 0.3, 0.3)):
        """pos(t), vel(t): truth over the ground (world frame); modes(t, x, y, vx, vy) -> [(name, p0, p, fn(tau) ->
        (x, y))] the futures from the state at t."""
        self.id, self.label, self.pos, self.vel, self.modes = aid, label, pos, vel, modes
        self.t_on, self.t_off, self.std = t_on, t_off, std


def _cv(x, y, vx, vy):
    return lambda tau: (x + vx * tau, y + vy * tau)


def _brake(x, y, vx, vy, a):
    v = math.hypot(vx, vy)
    ux, uy = (vx / v, vy / v) if v > 1e-6 else (1.0, 0.0)
    ts = v / a if a > 0 else 1e9

    def f(tau):
        tt = min(tau, ts)
        d = v * tt - 0.5 * a * tt * tt
        return (x + ux * d, y + uy * d)
    return f


def _lateral(x, y, vx, vy, y_to, dur):
    """Along at vx, sideways to y_to over dur seconds (a smooth S: half a cosine)."""
    y0 = y

    def f(tau):
        q = min(1.0, tau / dur)
        return (x + vx * tau, y0 + (y_to - y0) * 0.5 * (1.0 - math.cos(math.pi * q)))
    return f


def scenario(name):
    """(ego (pos(t), speed(t)), [Actor], T_end, expect) for a named scenario. expect: "alert" (a collision course
    the driver must be warned of) or "silent"."""
    if name == "stationary_in_path":
        v = 20.0
        ego = (lambda t: (v * t, 0.0), lambda t: v)
        a = Actor(1, "car", lambda t: (150.0, 0.0), lambda t: (0.0, 0.0),
                  lambda t, x, y, vx, vy: [("stand", 1.0, 1.0, _cv(x, y, 0.0, 0.0))])
        return ego, [a], 7.4, "alert"
    if name == "lead_braking":
        v = 20.0
        ego = (lambda t: (v * t, 0.0), lambda t: v)

        def lp(t):                    # 30 m ahead at our speed, brakes at 6 m/s^2 from t = 2 s
            if t < 2.0:
                return (32.0 + v * t, 0.0)
            tt = min(t - 2.0, v / 6.0)
            return (32.0 + v * 2.0 + v * tt - 3.0 * tt * tt, 0.0)

        def lv(t):
            return (v if t < 2.0 else max(0.0, v - 6.0 * (t - 2.0)), 0.0)

        def modes(t, x, y, vx, vy):
            if t < 2.0:
                return [("keep", 0.85, 0.85, _cv(x, y, vx, vy)), ("brake", 0.15, 0.15, _brake(x, y, vx, vy, 6.0))]
            return [("brake", 0.9, 0.9, _brake(x, y, vx, vy, 6.0)), ("keep", 0.1, 0.1, _cv(x, y, vx, vy))]
        return ego, [Actor(1, "car", lp, lv, modes)], 5.0, "alert"
    if name == "cut_in":
        v = 25.0
        ego = (lambda t: (v * t, 0.0), lambda t: v)
        # a car in the left lane, 15 m ahead and 5 m/s slower, moves into our lane from t = 1 s over 2.5 s

        def cp(t):
            x = 15.0 + 20.0 * t
            q = min(1.0, max(0.0, (t - 1.0) / 2.5))
            return (x, 3.5 * (1.0 - 0.5 * (1.0 - math.cos(math.pi * q))))

        def cv_(t):
            q = min(1.0, max(0.0, (t - 1.0) / 2.5))
            vy = -3.5 * 0.5 * math.pi * math.sin(math.pi * q) / 2.5 if 0.0 < q < 1.0 else 0.0
            return (20.0, vy)

        def modes(t, x, y, vx, vy):
            p_cut = 0.3 if t < 1.0 else 0.8
            return [("cut_in", p_cut, p_cut * 0.6, _lateral(x, y, vx, vy, 0.0, max(0.5, 3.5 - (t - 1.0)))),
                    ("keep", 1.0 - p_cut, 1.0 - p_cut * 0.6, _cv(x, y, vx, 0.0))]
        return ego, [Actor(1, "car", cp, cv_, modes)], 5.0, "alert"
    if name == "crossing":
        v = 14.0
        ego = (lambda t: (v * t, 0.0), lambda t: v)
        # a car from the right crosses at 10 m/s: at the ego's path (x = 70) at t = 5 s
        a = Actor(1, "car", lambda t: (72.0, -50.0 + 10.0 * t), lambda t: (0.0, 10.0),
                  lambda t, x, y, vx, vy: [("go", 0.8, 0.5, _cv(x, y, vx, vy)), ("yield", 0.2, 0.5, _brake(x, y, vx, vy, 4.0))])
        return ego, [a], 5.4, "alert"
    if name == "oncoming_our_side":
        v = 15.0
        ego = (lambda t: (v * t, 0.0), lambda t: v)
        # an oncoming car drifts from its lane (3.5 m left) into ours from t = 1 s
        def op(t):
            q = min(1.0, max(0.0, (t - 1.0) / 2.0))
            return (160.0 - 15.0 * t, 3.5 - 3.3 * 0.5 * (1.0 - math.cos(math.pi * q)))
        a = Actor(1, "car", op, lambda t: (-15.0, 0.0),
                  lambda t, x, y, vx, vy: [("drift", 0.7 if t > 1.2 else 0.1, 0.7 if t > 1.2 else 0.1, _lateral(x, y, -15.0, 0.0, 0.2, 2.0)),
                                           ("keep", 0.3 if t > 1.2 else 0.9, 0.3 if t > 1.2 else 0.9, _lateral(x, y, -15.0, 0.0, 3.5, 1.0))])
        return ego, [a], 5.0, "alert"
    if name == "ped_step_out":
        v = 11.0
        ego = (lambda t: (v * t, 0.0), lambda t: v)
        # a pedestrian on the kerb (3.5 m right) at x = 55 steps out at 1.5 m/s at t = 2.5 s
        def pp(t):
            return (55.0, -3.5 + 1.5 * max(0.0, t - 2.5))
        a = Actor(1, "human", pp, lambda t: (0.0, 1.5 if t > 2.5 else 0.0),
                  lambda t, x, y, vx, vy: ([("walk", 0.9, 0.6, _cv(x, y, 0.0, 1.5)), ("stop", 0.1, 0.4, _cv(x, y, 0.0, 0.0))]
                                           if t > 2.5 else
                                           [("wait", 0.8, 0.9, _cv(x, y, 0.0, 0.0)), ("step_out", 0.2, 0.1, _cv(x, y, 0.0, 1.5))]),
                  std=(0.6, 0.3, 0.3, 0.3))
        return ego, [a], 5.0, "alert"
    if name == "parked_row":
        v = 11.0
        ego = (lambda t: (v * t, 0.0), lambda t: v)
        acts = [Actor(10 + k, "car", (lambda x0: lambda t: (x0, -3.0))(20.0 + 7.0 * k), lambda t: (0.0, 0.0),
                      lambda t, x, y, vx, vy: [("stand", 1.0, 1.0, _cv(x, y, 0.0, 0.0))]) for k in range(12)]
        return ego, acts, 8.0, "silent"
    if name == "adjacent":
        v = 25.0
        ego = (lambda t: (v * t, 0.0), lambda t: v)
        a = Actor(1, "car", lambda t: (12.0 + 24.5 * t, 3.5), lambda t: (24.5, 0.0),
                  lambda t, x, y, vx, vy: [("keep", 0.95, 0.95, _cv(x, y, vx, 0.0)), ("cut_in", 0.05, 0.03, _lateral(x, y, vx, vy, 0.0, 3.0))])
        return ego, [a], 8.0, "silent"
    if name == "oncoming_passing":
        v = 15.0
        ego = (lambda t: (v * t, 0.0), lambda t: v)
        a = Actor(1, "car", lambda t: (150.0 - 15.0 * t, 3.4), lambda t: (-15.0, 0.0),
                  lambda t, x, y, vx, vy: [("keep", 0.97, 0.97, _cv(x, y, vx, 0.0)), ("drift", 0.03, 0.03, _lateral(x, y, vx, 0.0, 0.0, 2.0))])
        return ego, [a], 7.0, "silent"
    raise KeyError(name)


NAMES = ("stationary_in_path", "lead_braking", "cut_in", "crossing", "oncoming_our_side", "ped_step_out",
         "parked_row", "adjacent", "oncoming_passing")


def frames(name, hz=10.0, t_grid=T_GRID, spread=(0.3, 0.15)):
    """[(radar message, truth)] of a scenario at hz. spread: the futures' 1 sigma grows by these per second (along,
    across); a standing actor's by a share of it (its speed / 5 m/s, at least a tenth)."""
    (epos, espd), actors, T_end, _ = scenario(name)
    out = []
    tg = np.asarray(t_grid, float)
    n = int(T_end * hz) + 1
    for q in range(n):
        T = q / hz
        ex, ey = epos(T)
        v = espd(T)
        objs, truth = [], []
        for a in actors:
            if not (a.t_on <= T <= a.t_off):
                continue
            x, y = a.pos(T)
            vx, vy = a.vel(T)
            cx, cy = x - ex, y - ey
            o = {"id": a.id, "state": "confirmed", "exist": 0.99, "x_m": round(cx, 3), "y_m": round(cy, 3),
                 "vx_mps": vx, "vy_mps": vy, "std": list(a.std), "class": {"label": a.label, "confidence": 0.9},
                 "motion": "stationary" if math.hypot(vx, vy) < 0.3 else "moving", "flags": [],
                 "heading_rad": math.atan2(vy, vx) if math.hypot(vx, vy) > 1.0 else None}
            fut = []
            g = max(0.1, min(1.0, math.hypot(vx, vy) / 5.0))      # a standing actor's future spreads little
            for nm, p0, p, fn in a.modes(T, x, y, vx, vy):
                xs, ys = zip(*(fn(tau) for tau in tg))
                gm = 1.0 if nm not in ("stand", "wait") else g
                sx = [a.std[0] + gm * spread[0] * tau for tau in tg]
                sy = [a.std[1] + gm * spread[1] * tau for tau in tg]
                fut.append({"p": p, "p0": p0, "m": nm, "x": [xx - ex for xx in xs], "y": [yy - ey for yy in ys],
                            "sx": sx, "sy": sy, "rxy": [0.0] * len(tg), "v": [math.hypot(vx, vy)] * len(tg)})
            o["fut"] = fut
            objs.append(o)
            truth.append({"id": a.id, "x": cx, "y": cy})
        ego_fut = {"modes": [{"p": 1.0, "m": "keep", "x": [espd(T) * tau for tau in tg], "y": [0.0] * len(tg),
                              "v": [v] * len(tg)}]}
        msg = {"type": "radar", "api": 1, "seq": q + 1, "ts_mono_ns": int(round((100.0 + T) * 1e9)), "frame": "car",
               "objects": objs, "ego": {"speed_mps": v, "yaw_rate_rps": 0.0, "accel_mps2": 0.0, "stands": v < 0.1},
               "fut": {"t": list(tg)}, "ego_fut": ego_fut, "sensors": {"cam": {"kind": "camera", "age_s": 0.05}}}
        out.append((msg, {"T": T, "actors": truth, "ego_x": ex}))
    return out


def collision_time(name, step=0.01):
    """The first time the actors' truth footprints touch the car's at constant speeds as scripted (or None)."""
    (epos, espd), actors, T_end, _ = scenario(name)
    from .futures import CLASS_SIZE
    t = 0.0
    while t <= T_end + 3.0:
        ex, ey = epos(t)
        for a in actors:
            x, y = a.pos(t)
            L, W = CLASS_SIZE.get(a.label, CLASS_SIZE[None])[:2]
            # the car: 2.1 m ahead of the camera to 2.35 m behind, 1.76 wide; actor's point = its nearest point
            if (ex - 2.35 <= x + L and x <= ex + 2.1) and abs(y - ey) <= 0.88 + W / 2.0:
                return t
        t += step
    return None
