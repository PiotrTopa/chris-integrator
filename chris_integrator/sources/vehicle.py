"""The car's own motion (mapd's `ego` relay of the feeder's `motion`, CAN) and its signals (`vehicle`).

Noise measured on the reference car (chris-roadsense docs/lanes/L35-ego-fusion-audit.md section 4, L37 section 3.5):
speed 0x0B4 robust sigma 0.012-0.020 m/s; gyro zeroed at standstill; rear wheel difference sigma 0.010-0.019
rad/s; longitudinal accelerometer sigma 0.20-0.25 m/s2 about a slow offset.
"""
import math

import numpy as np

from ..core import ego as E
from ..core.ego import AuxState
from ..core.registry import NS, Meas, Source, accel_grade, fin, lin, product_bias, scaled

FEED_GAP_S, FEED_WARMUP_S = 1.0, 0.35  # a feed silent this long is not believed for this long after it resumes
YAW_SIGMA = {"yaw_rate": (0.004, 0.03), "wheel_speeds": (0.02, 0.05), "steering": (0.03, 0.10)}
TRACK_M = 1.48
# Wheel-speed sensors have a floor: on the reference car a wheel reads exactly 0 below about 0.35 m/s, and the four
# come alive one at a time (126-320 samples per drive of a moving car with a wheel at zero, up to a speed signal
# of 1.1 m/s; one wheel at 0.4 m/s beside one at 0 is a "yaw rate" of 0.3-0.6 rad/s). Below the floor the wheels
# are no measurement of anything: neither their mean nor their difference is offered.
WHEEL_FLOOR = 0.6                # m/s: every wheel at least this fast, or the sample is not used
# The rear wheels' difference carries a slow offset against the gyro and the camera (which agree with each other
# to 0.001-0.003 rad/s on straights): window medians of -0.022 to +0.003 rad/s over two minutes, not proportional
# to the speed, on four drives. Its error is therefore not white; the sigma says so (rms of those medians).
WHEEL_YAW_SLOW = 0.012           # rad/s
_H_V, _H_W = lin([E.V], [1.0]), lin([E.W], [1.0])


SCALE_CLEAR = 3.0                # a scale factor is learned where its regressor stands this many sigma clear of its noise
SCALE_MIN_YAW = 0.02             # ... and at least this far from zero (rad/s): a straight road with its wiggles teaches nothing
SCALE_MIN_SPEED = 1.0            # m/s


# The regressor of a scale state, as it enters a row's Jacobian (core/registry.py `scaled`, `delayed_speed`; the
# arithmetic is the native core's, egomotion.c `eh_reg`).
#
# The Jacobian of a scale state is the quantity it multiplies. Where that quantity is mostly its own estimation
# noise (a yaw rate on a straight road), a noisy regressor teaches the scale nonsense: the errors-in-variables
# bias of an EKF (the scale walks steadily, on a straight road, by rows that should say nothing about it).
# The regressor is therefore shrunk towards zero by its own uncertainty: zero inside the dead zone
# max(SCALE_CLEAR sigma, floor), the value itself far outside (value x (1 - limit^2 / value^2)).
#
# Before L41 the rule was a fixed floor (0.05 rad/s, 3 m/s) below which the Jacobian was dropped. That also
# dropped the scale's *uncertainty* from every gentle bend (an unlearned gyro scale of 8 % at 0.04 rad/s is
# 3 degrees of heading in a long motorway bend that the covariance did not contain), although the yaw rate is
# known there to 0.003 rad/s and the scale is perfectly learnable.


def scale_reg(value, floor):
    """The fixed-floor rule (kept for regressors that are either far from zero or absent, like a radar's own
    speed while moving): the value, or 0 below the floor."""
    return value if abs(value) >= floor else 0.0


def warming(t, mem):
    """After a silence of the feed its first samples repeat what it held before the silence (measured: 19-25
    silences of about 7 s per drive, the first 2-3 samples after each, up to 0.28 s, stale by up to 7 m/s):
    they are not measurements. True while the feed warms up."""
    last = mem.get("last_t")
    mem["last_t"] = t
    if last is not None and t - last > FEED_GAP_S:
        mem["warm_until"] = t + FEED_WARMUP_S
        mem["warmups"] = mem.get("warmups", 0) + 1
    return t < mem.get("warm_until", -1e18)


class VehicleSource(Source):
    id = "vehicle"
    group = "vehicle"
    rank = 3
    types = ("ego",)
    stale_s = 0.6
    info = "vehicle motion from the host's feeder (CAN), relayed by mapd"
    aux = (
        AuxState("b_g", 0.0, 0.01, 1e-9, unit="rad/s", doc="gyro bias"),
        AuxState("s_g", 0.0, 0.08, 1e-9, unit="1", doc="gyro = (1 + s_g) x yaw rate + b_g"),
        AuxState("s_wy", 0.0, 0.08, 1e-9, unit="1", doc="rear wheel difference / track = (1 + s_wy) x yaw rate"),
        AuxState("b_ax", 0.0, 0.5, 1e-5, unit="m/s2", doc="longitudinal accelerometer offset (with grade when no altitude source)"),
        AuxState("b_ay", 0.0, 1.0, 1e-2, unit="m/s2", doc="lateral accelerometer offset (bank and bias)"),
    )
    track_m = TRACK_M
    fuse_accel_lat = True

    def __init__(self, ego_motion=None, **kw):
        Source.__init__(self, **kw)
        psd = (ego_motion or {}).get("gyro_bias_psd")
        if psd is not None:            # the gyro bias's random walk (temperature, turn-on drift): docs/EGO_MOTION.md
            self.aux = tuple(AuxState("b_g", 0.0, 0.01, float(psd), unit="rad/s", doc="gyro bias") if a.name == "b_g" else a
                             for a in self.aux)

    def measure(self, msg, ctx):
        integ = ctx.integ
        t = msg["ts_mono_ns"] / NS - integ.offset_cfg("vehicle")     # (configured: the bus's stamps are this late)
        mem = ctx.mem
        out = []
        warm = warming(t, mem)
        mem["tag"] = msg.get("tag")
        if warm:
            return out
        sig = msg.get("turn_signal")
        if sig is not None:
            out.append(Meas("signal:turn_signal", t, sig, role="signal"))
        v = msg.get("speed_mps")
        if not fin(v):
            return out
        v = float(v)
        still = bool(msg.get("standstill")) and v == 0.0
        level = None
        stand = ctx.integ.stand
        if stand is not None:          # the flag confirmed by the inertial sensors and the camera (docs/EGO_MOTION.md)
            nan = float("nan")
            ib = self.ix["b_g"]
            sd = math.sqrt(max(float(ctx.P[ib, ib]), 0.0))
            g_, ax_, ay_ = msg.get("yaw_rate_rps"), msg.get("accel_long_mps2"), msg.get("accel_lat_mps2")
            level = stand.bus(t, still, float(g_) if fin(g_) and msg.get("yaw_source") == "yaw_rate" else nan,
                              float(ax_) if fin(ax_) else nan, float(ay_) if fin(ay_) else nan, float(ctx.x[ib]), sd)
            if level == 2:
                gm = stand.gyro_mean
                d = None if gm is None else gm - float(ctx.x[ib])
                if d is not None and abs(d) > max(4.0 * sd, ctx.cfg["ego_motion"]["stand_step_tol"]) and \
                        t - mem.get("bias_step_t", -1e18) > 5.0:
                    # the car is at rest and the gyro does not read what its bias is believed to be: the bias has
                    # changed (turn-on, temperature), or was taught wrong. Its uncertainty is opened to the
                    # difference: the standstill's rows then place it within a second instead of a minute.
                    mem["bias_step_t"] = t
                    mem["bias_steps"] = mem.get("bias_steps", 0) + 1
                    integ.reset_aux(ib, float(ctx.x[ib]), max(sd, abs(d)))
                    integ.fdi.event("vehicle", "gyro_bias", None, "the gyro reads %+.4f rad/s at rest, its bias was believed to be "
                                    "%+.4f +- %.4f: re-estimating" % (gm, float(ctx.x[ib]), sd), quantity="yaw_rate")
                    sd = max(sd, abs(d))
                mem["rest_t"], mem["rest_sd"] = t, sd      # the bias as the last confirmed standstill left it
            hist = mem.setdefault("bias_sd", [])           # (time, std) every 2 s over the last 20 s: is it being observed?
            if not hist or t - hist[-1][0] >= 2.0 or t < hist[-1][0]:
                hist.append((t, sd))
                del hist[:-11]
        out.append(Meas("standstill", t, still, src="vehicle.standstill", role="flag", level=level if still else None))
        ibg, isg, iswy, ibax, ibay = (self.ix[k] for k in ("b_g", "s_g", "s_wy", "b_ax", "b_ay"))
        vw = float(ctx.P[E.W, E.W])        # how well the yaw rate is known: the scales' regressor is judged by it
        if not still:
            # the speed state is in this signal's scale by definition (core/ego.py, SV)
            sg = 0.03 + 0.002 * abs(v)
            out.append(Meas("speed", t, v, sg, _H_V, "vehicle.speed", mon=(v, sg)))
            integ.lag_ref("speed", t, v)
            a = msg.get("accel_long_mps2")
            if fin(a):          # a + g sin(grade) + offset
                out.append(Meas("accel", t, float(a), 0.25, accel_grade(E.A, E.TH, ibax, E.G), "vehicle.accel",
                                mon=(float(a), 0.25)))
            ay = msg.get("accel_lat_mps2")
            if fin(ay) and self.fuse_accel_lat:      # v w + offset
                out.append(Meas("accel_lat", t, float(ay), 0.5, product_bias(E.V, E.W, ibay), "vehicle.accel_lat"))
        else:
            out.append(Meas("speed", t, 0.0, 0.03, src="vehicle.speed", role="monitor"))
        r = msg.get("yaw_rate_rps")
        ysrc = msg.get("yaw_source")
        if fin(r):
            r = float(r)
            s0, s1 = YAW_SIGMA.get(ysrc, YAW_SIGMA["steering"])
            sg = s0 + s1 * abs(r)
            if ysrc == "wheel_speeds":               # the feed's yaw rate is the rear wheels' difference, relayed
                ws = msg.get("wheel_speeds_mps")
                if isinstance(ws, (list, tuple)) and ws and all(fin(w) for w in ws) and min(ws) < WHEEL_FLOOR:
                    return out                       # under the wheel sensors' floor: no yaw measurement
                sg = math.hypot(sg, WHEEL_YAW_SLOW)
            # (1 + s_g) w + b_g, the scale's regressor judged by how well the yaw rate is known; another yaw source: w.
            # A gyro whose value is old at its stamp (configured, or estimated with `gyro_to_camera`:
            # docs/EGO_MOTION.md) reads the yaw rate of then: w - delay x yaw acceleration
            h_g = scaled(E.W, isg, vw, SCALE_MIN_YAW, ibg, integ.gyro_delay(), E.AL) if ysrc == "yaw_rate" else _H_W
            # the gyro's own row never corrects the gyro's scale: nothing in a yaw-rate sensor says what its scale is
            # (every yaw sensor of the car has one). The scale is a consider state here, so the heading carries its
            # uncertainty through every bend, and it is corrected by what does know: heading references (a
            # receiver's track over turns, the map's bearing) through that very correlation. The other yaw
            # sensors' scales are then learned relative to the gyro's.
            if ysrc == "yaw_rate":
                integ.lag_ref("yaw_rate", t, r)          # the reference of the other yaw sensors' offsets
            out.append(Meas("yaw_rate", t, r, sg, h_g, "vehicle.yaw", mon=(r, sg),
                            consider=(isg,) if ysrc == "yaw_rate" else None))
        return out

    def report(self, ctx):
        return {"tag": ctx.mem.get("tag"), "warmups": ctx.mem.get("warmups", 0)}

    EVIDENCE = ((1, "inertial"), (2, "inertial_against"), (64, "inertial_offset"), (4, "camera"), (8, "camera_turning"), (16, "bus_only"))

    def calib_state(self, ctx):
        """`calibration.vehicle.stand`: the standstill as confirmed (docs/EGO_MOTION.md); `calibration.vehicle.gyro_bias`:
        how the gyro's bias is observed now. At a confirmed standstill the gyro reads its bias (`standstill`). While
        driving nothing measures it directly: the other yaw sensors (the camera always as a consider state, the rear
        wheels by `gyro_bias_wheels`) hold the yaw rate, and the bias is `held` with its random walk, or `observed`
        where the state's uncertainty is still falling (`std_trend_per_min`: the change of its std over the last
        20 s, per minute); `straight` says whether the car drives straight now (no scale error can pass for a bias)."""
        st = ctx.integ.stand
        if st is None:
            return {}
        mem = ctx.mem
        n_bus, n_claimed, n_conf, n_ev = st.counts
        ev = st.evidence
        out = {"stand": {"state": ("moving", "claimed", "confirmed")[st.state],
                         "evidence": [name for bit, name in self.EVIDENCE if ev & bit],
                         "confirmations": n_ev, "claimed_only_share": round(n_claimed / (n_claimed + n_conf), 3) if n_claimed + n_conf else None,
                         "bias_steps": mem.get("bias_steps", 0)}}
        hist, now = mem.get("bias_sd") or [], ctx.now
        ib = self.ix["b_g"]
        sd = ctx.std("b_g")
        trend = None
        if len(hist) >= 2 and hist[-1][0] > hist[0][0]:
            trend = 60.0 * (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
        rest_t = mem.get("rest_t")
        v, w = abs(float(ctx.x[E.V])), abs(float(ctx.x[E.W]))
        if st.state == 2:
            how = "standstill"
        elif trend is not None and trend < -0.02 * sd:
            how = "observed"
        else:
            how = "held"
        out["gyro_bias"] = {"observed_by": how, "straight": bool(v > 3.0 and w < 0.02),
                            "since_standstill_s": None if rest_t is None or now is None else round(max(0.0, now - rest_t), 1),
                            "std_at_standstill": None if rest_t is None else round(mem["rest_sd"], 6),
                            "std_trend_per_min": None if trend is None else round(trend, 7)}
        return out


class VehicleSignals(Source):
    id = "vehicle.signals"
    group = "vehicle"
    rank = 3
    types = ("vehicle",)
    stale_s = 0.6
    counts = False

    def measure(self, msg, ctx):
        t = msg["ts_mono_ns"] / NS
        b = (msg.get("controls") or {}).get("brake")
        return [Meas("signal:brake", t, b, role="signal")] if isinstance(b, bool) else []


class WheelsSource(Source):
    """Mean of the wheel speeds: a second reading of the speed with its own decoding. Family `vehicle`: it
    checks the speed signal's decoding, it is no independent vote (same sensors), and it is not fused."""
    id = "wheels"
    group = "wheels"
    rank = 2
    family = "vehicle"
    types = ("ego",)
    stale_s = 0.6
    track_m = TRACK_M
    independent = ("yaw_rate",)  # its yaw rate comes from the wheels, not from the gyro: an opinion of its own
    info = "mean of the wheel speeds: a second reading of the speed; rear wheel difference: a second yaw rate"
    lags = (("yaw", "yaw_rate", 0.0, -0.3, 0.3, 0.05, 0.004),)       # the wheels' yaw rate behind the gyro's

    def measure(self, msg, ctx):
        t = msg["ts_mono_ns"] / NS - ctx.integ.offset_cfg("vehicle")
        if warming(t, ctx.mem):
            return []
        ws = msg.get("wheel_speeds_mps")
        if not isinstance(ws, (list, tuple)) or len(ws) < 2 or not all(fin(w) for w in ws):
            return []
        m = sum(ws) / len(ws)
        if msg.get("reverse"):
            m = -m
        still = bool(msg.get("standstill")) and msg.get("speed_mps") == 0.0
        alive = min(ws) >= WHEEL_FLOOR
        if not alive and not (still and max(ws) == 0.0):
            # under the sensors' floor (creeping, pulling away): no measurement; the source is alive all the same
            return [Meas("signal:wheels_below_floor", t, True, role="signal")]
        # the scale of wheel speeds against the speed signal is a car's own (0.98 here): the tolerance covers it
        out = [Meas("speed", t, m, 0.10 + 0.04 * abs(m), src="wheels.speed", role="monitor")]
        # the rear wheels' difference is a yaw rate of its own (another sensor than the gyro): fused, with its
        # scale against the gyro's as a state (declared by the vehicle source, which owns the feed's calibration)
        iswy = ctx.integ.model.index.get("vehicle.s_wy")
        # (not when the feed's own yaw rate is this very difference, relayed: that would count it twice. A feed
        # without a gyro has the wheels as its yaw sensor: before L43 such a car had none at all, and its position
        # left the road at the first bend.)
        if iswy is not None and len(ws) == 4 and not still and alive and msg.get("yaw_source") in ("yaw_rate", None):
            rw = (ws[3] - ws[2]) / self.track_m
            if msg.get("reverse"):
                rw = -rw
            vw = float(ctx.P[E.W, E.W])
            sg = math.hypot(0.02 + 0.05 * abs(rw), WHEEL_YAW_SLOW)
            # (the wheels' difference carries a slow offset of its own: whether it may teach the gyro's bias is a knob)
            ibg = None if ctx.cfg["ego_motion"]["gyro_bias_wheels"] else ctx.integ.model.index.get("vehicle.b_g")
            # its time offset (docs/EGO_MOTION.md): `off` behind the gyro (configured, or estimated and applied),
            # the gyro itself `gyro_delay` old: the wheels read the yaw rate of off + gyro_delay ago. What a row
            # tells the estimator of its offset is how far behind the gyro its model has put it.
            lag, off = ctx.lag("yaw")
            if lag is None:
                off, cb = ctx.integ.offset_cfg("wheels.yaw"), None
            else:
                def cb(nu, s0, tt, lag=lag, var=sg * sg, rel=off):
                    lag.offer(tt, nu, s0 + var, rel)
            h_wy = scaled(E.W, iswy, vw, SCALE_MIN_YAW, None, off + ctx.integ.gyro_delay(), E.AL)       # (1 + s_wy) w
            out.append(Meas("yaw_rate", t, rw, sg, h_wy, "wheels.yaw", robust=True, mon=(rw, sg), cb=cb,
                            consider=None if ibg is None else (ibg,)))
        return out
