"""The road model's pose (`road.pose`): speed and yaw rate of the 200 ms image pair, the direction of travel
in the device frame (camera mounting), its own stated uncertainty, a standstill detector for cars without a
vehicle feed.

Characterised on the reference car: speed sigma 0.25 + 0.035 v with heavy tails and an error correlated over
seconds; valid 0.10-0.15 s before the frame stamp; direction of travel robust spread 0.003-0.005 per frame at
speed on straights (docs/DESIGN.md section 1).
"""
import math

import numpy as np

from ..native import round                  # builtins.round (its results; the native core)
from ..core import ego as E
from ..core.ego import AuxState
from ..core.registry import NS, Meas, Source, delayed_speed, fin, lin, scaled
from .vehicle import SCALE_MIN_SPEED, SCALE_MIN_YAW

PAIR_LAG_S = 0.125               # prior of the delay: the pose describes the image pair
STOP_SPEED, GO_SPEED, STOP_CONFIRM_S = 0.15, 0.8, 1.0
MOUNT_MIN_SPEED, MOUNT_MAX_YAW = 5.0, 0.10
PITCH_CUSUM_K, PITCH_CUSUM_H = 1.5, 14.0   # in sigma of the pitch row; a step of 1 degree at 10 m/s fires in about half a second
# the pose's motion at a standstill another source knows (m/s, rad/s, frames within 2 s). Deliberately coarse: on
# the reference car the road model reports small rotations while the car stands in traffic, and a threshold low
# enough to catch a gentle re-aiming by hand (0.01 rad/s: one such event recorded) fired 3-4 times per healthy
# drive. A gentle re-aiming at standstill is therefore NOT detected here; it shows at the next drive (pitch step,
# model out of range, sign rows).
HANDLED_SPEED, HANDLED_RATE, HANDLED_FRAMES = 0.5, 0.04, 6
PITCH_STEP_MIN = 0.006           # rad: smaller "steps" are the estimator's ordinary business
# The speed error's structure, fitted to its autocorrelation on the reference car's two healthy drives (12 000
# frames; correlation 0.80 at one frame, 0.54 at 0.5 s, 0.42 at 1 s, 0.21 at 3 s, 0.13 at 5 s): hardly any of it
# is white. Shares of the characterised variance: a slow part 0.52 (3.5 s), a fast part 0.37 (0.35 s), white
# 0.11. Both coloured parts are Gauss-Markov states in m/s with the sigma they have at town speeds; what the
# characterised sigma has beyond them at higher speed is white. (States scaled by the speed-dependent sigma were
# tried: such a state is a second scale factor, and the filter then split the camera's scale between the two
# and got it wrong by 8 %.)
SLOW_SIGMA, SLOW_TAU_S = 0.45, 3.5
FAST_SIGMA, FAST_TAU_S = 0.38, 0.35
WHITE_SHARE = 0.33               # sqrt(0.11): at least this share of the sigma is white per frame
# the error is heavy-tailed (rms 1.3-1.5 x its robust sigma, p95 at 2.5 robust sigmas): the monitors judge by
# the robust sigma (`speed_sigma`), the estimate's uncertainty has to cover 95 % of the errors within 2 sigma
TAIL = 1.35


# In turns the pose's speed is worse: robust sigma against the vehicle 0.36-0.50 m/s below 0.2 rad/s, 0.5-0.8 at
# 0.2-0.3, 0.8-0.9 at 0.3-0.6 rad/s (two healthy drives, 12 000 frames), and 5-14 % of the frames above 0.2 rad/s
# are beyond 4 of the straight-road sigmas (0.1-0.7 % below). A tight turn is not a camera fault: the term below
# is the characterised error there (2 m/s per rad/s, in quadrature).
TURN_SIGMA = 2.0


def speed_sigma(v, w=0.0):
    """Characterised 1-sigma of the road model's speed against the vehicle's (two healthy drives); w: the pose's
    own yaw rate."""
    return math.hypot(0.25 + 0.035 * abs(v), TURN_SIGMA * w)


class VisionSource(Source):
    id = "vision"
    group = "vision"
    rank = 1
    types = ("road", "calib")
    stale_s = 0.6
    info = "the road model's pose (200 ms image pair)"
    aux = (
        AuxState("tau", PAIR_LAG_S, 0.04, 1e-9, lo=0.0, hi=0.4, unit="s", doc="the pose is this old at the frame stamp"),
        AuxState("s", 0.0, 0.05, 1e-8, unit="1", doc="speed = (1 + s) x true speed + m (a monocular scale)"),
        AuxState("m", 0.0, SLOW_SIGMA, tau_s=SLOW_TAU_S, unit="m/s", doc="speed error correlated over seconds"),
        AuxState("mf", 0.0, FAST_SIGMA, tau_s=FAST_TAU_S, unit="m/s", doc="speed error correlated over a few frames"),
        AuxState("s_w", 0.0, 0.10, 1e-8, unit="1", doc="yaw rate = (1 + s_w) x true yaw rate (measured 0.91-0.96 in turns)"),
        AuxState("dyaw", 0.0, 0.15, 1e-9, unit="rad", doc="direction of travel in the device frame, yaw (vy / vx): the mount against the calibration in use"),
        AuxState("dpitch", 0.0, 0.15, 1e-9, unit="rad", doc="... pitch (-vz / vx)"),
    )
    lever_m = 1.5                # camera ahead of the rear axle
    # time offsets estimated outside the filter (docs/EGO_MOTION.md): the pose's yaw rate behind the gyro (the
    # filter's delay state is learnt from the speed rows only), and a witness of that state: the pose's speed
    # behind the bus speed
    lags = (("yaw", "yaw_rate", PAIR_LAG_S, 0.0, 0.4, 0.05, 0.004), ("speed", "speed", PAIR_LAG_S, 0.0, 0.4, 0.3, 0.02))

    def reset(self):
        pass

    def measure(self, msg, ctx):
        mem = ctx.mem
        if msg.get("type") == "calib":
            key = (msg.get("pitch"), msg.get("yaw"))
            old = mem.get("calib")
            if old is not None and key != old:
                # another calibration is in use. The mount itself has not moved: the residual against the new
                # calibration is the old residual plus what the calibration changed by (a little less certain:
                # the model's response to its warp is not exactly linear). Without numbers: back to the prior.
                for name, k in (("dpitch", 0), ("dyaw", 1)):
                    i = self.ix[name]
                    a = next(a for a in self.aux if a.name == name)
                    if fin(old[k]) and fin(key[k]):
                        ctx.integ.reset_aux(i, ctx.val(name) + old[k] - key[k], math.hypot(ctx.std(name), 0.005))
                    else:
                        ctx.integ.reset_aux(i, a.x0, a.sigma0)
                mem["cusum"] = None
            mem["calib"] = key
            return []
        out = []
        ts = msg["ts_mono_ns"] / NS
        pose = msg.get("pose") or {}
        v = pose.get("vx")
        if not fin(v):
            v = msg.get("v_ego")
        mem["road_model"] = msg.get("road_model")
        if msg.get("valid") is False or not fin(v):
            return out
        v = float(v)
        ix = self.ix
        itau, isc, im, idy, idp, isw = ix["tau"], ix["s"], ix["m"], ix["dyaw"], ix["dpitch"], ix["s_w"]
        x = ctx.x
        P = ctx.P                                        # (read once: nothing below changes the estimate before the
        tau = float(x[itau])                             # mount rows, where it is last read)
        t = ts - tau
        # (the turn is taken from the estimate, which a gyro holds on a car that has one: a model out of its range
        # reports any yaw rate; the pose's own is used only while nothing else knows the yaw rate)
        r_own = pose.get("yaw_rate")
        w_est = float(x[E.W]) if float(P[E.W, E.W]) < 0.01 ** 2 else (float(r_own) if fin(r_own) else 0.0)
        char = speed_sigma(v, w_est)
        vs = pose.get("vx_std")
        white = WHITE_SHARE * char
        if fin(vs) and float(vs) > 2.0 * char:
            white = float(vs)                            # the model itself says it is lost: believed
        imf = ix["mf"]
        fz = TAIL * char                                 # the sigma that covers the error's tails (as fused)
        white = max(white, WHITE_SHARE * fz, math.sqrt(max(fz * fz - SLOW_SIGMA ** 2 - FAST_SIGMA ** 2, 0.0)))
        vv, vw = float(P[E.V, E.V]), float(P[E.W, E.W])
        # (1 + s) (v - a (tau - tau0)) + m + mf: the scale's regressor judged by how well the speed is known
        h_speed = delayed_speed(E.V, isc, im, imf, itau, tau, vv, SCALE_MIN_SPEED)
        # without a speed reference the scale cannot be told from the speed: it then stays at its prior and its
        # uncertainty is the speed's (a "consider" state), instead of drifting on its own linearisation
        sp0 = ctx.speed_known()
        # (a reference is another source holding the speed of a moving car: the camera's own inferred standstill
        # makes the speed certain too, and taught the scale nonsense at every pull-away before L41)
        ref = sp0 is not None and sp0[1] < 0.1 and abs(sp0[0]) > 1.0
        lag_v, _ = ctx.lag("speed")
        cb_v = None
        if lag_v is not None and ref:

            def cb_v(nu, s0, tt, lag=lag_v, var=white * white, rel=tau):
                lag.offer(tt, nu, s0 + var, rel)
        out.append(Meas("speed", t, v, white, h_speed, "vision.pose", robust=True, mon=(v, char), local=True,
                        consider=None if ref else (isc,), cb=cb_v))
        if fin(vs) and abs(v) >= 1.0:
            # (against the straight-road level, as before L43: a model that has left its range reports any yaw rate,
            # and the turn term would then excuse its exploded uncertainty; measured on the incident drive, 0.6 s)
            out.append(Meas("self:speed", ts, float(vs) / speed_sigma(v), role="self"))
        r = pose.get("yaw_rate")
        if ctx.integ.stand is not None and fin(r):       # a witness of the standstill (docs/EGO_MOTION.md)
            ctx.integ.stand.cam(t, v, float(r))
        if fin(r):
            r = -float(r)                                # the device frame is z down: positive right
            rs = pose.get("yaw_rate_std")
            sig = 0.007 + 0.25 * abs(r)
            sig_f = 0.009 + 0.08 * abs(r)                # as fused: the gain error is a state, not noise
            # (the delay is estimated from the speed rows only: the yaw acceleration the delay's Jacobian would
            # need here is itself too poorly known, and a noisy regressor biases what it teaches)
            ref_w = P[E.W, E.W] < 0.01 ** 2              # a gyro (or wheels) holds the yaw rate
            sg_w = max(sig_f, float(rs)) if fin(rs) else sig_f
            # The yaw rate's time offset (docs/EGO_MOTION.md). The row is inserted with the speed row, at the
            # delay state's time; where the yaw rate's own lag behind the gyro is configured, or estimated and
            # applied, its model carries the rest: the yaw rate of (lag + the gyro's own delay - tau) before that.
            # What a row tells the estimator of that lag is how far behind the gyro its model has put it.
            integ = ctx.integ
            lag_w, off = ctx.lag("yaw")
            gd, d_w, cb_w = integ.gyro_delay(), 0.0, None
            if ctx.cfg["ego_motion"]["offsets_apply"] if lag_w is not None else False:
                d_w = off + gd - tau
            elif ctx.cfg["ego_motion"]["time_offsets"].get(self.id + ".yaw") is not None:
                d_w = integ.offset_cfg(self.id + ".yaw") + gd - tau
            if lag_w is not None:        # (without a gyro's samples about the row's time the estimator drops it)

                def cb_w(nu, s0, tt, lag=lag_w, var=sg_w * sg_w, rel=tau + d_w - gd):
                    lag.offer(tt, nu, s0 + var, rel)
            h_w = scaled(E.W, isw, vw, SCALE_MIN_YAW, None, d_w, E.AL)       # (1 + s_w) w
            out.append(Meas("yaw_rate", t, r, sg_w, h_w, "vision.pose",
                            robust=True, mon=(r, sig), local=True, consider=None if ref_w else (isw,), cb=cb_w))
        # camera mounting: the direction of travel in the device frame, judged at the integrator's speed
        sp = sp0                                         # (the same query as above: nothing between changed the estimate)
        vy, vz = pose.get("vy"), pose.get("vz")
        if sp is not None and sp[0] >= MOUNT_MIN_SPEED and abs(float(x[E.W])) < MOUNT_MAX_YAW and fin(vy) and fin(vz) \
                and not ctx.faulted() and v > 0.5 * sp[0]:
            L = self.lever_m
            # The rows are about the *direction* of the pose: its lateral and vertical velocity against its own
            # forward velocity (the same pose, the same monocular scale). The regressor is therefore the measured
            # vx, not the speed state: with the state there, these rows were a speed measurement in disguise
            # (vy = v x yaw offset), which let a camera without any reference "learn" its own scale.
            h_y, h_z = lin([idy, E.W], [v, -L]), lin([idp], [-v])        # v dyaw - w L;  -v dpitch
            sy, sz = pose.get("vy_std"), pose.get("vz_std")
            fl = 0.02 + 0.006 * sp[0]
            sgz = max(fl, float(sz)) if fin(sz) else fl
            # the mount has one degree of freedom (pitch); yaw is fixed: two row classes, judged separately
            out.append(Meas("mount_yaw", t, float(vy), max(fl, float(sy)) if fin(sy) else fl, h_y, "vision.mount", robust=True, local=True))
            out.append(Meas("mount_pitch", t, float(vz), sgz, h_z, "vision.mount", robust=True, local=True))
            mem["mount_n"] = mem.get("mount_n", 0) + 1
            self._pitch_step(ctx, mem, ts, float(vz), sgz, v, idp)
        if ctx.faulted():
            # the road model far outside its range (measured: a camera tilted by 16 degrees) returns a collapsed
            # speed with an exploded stated std, a pose direction that reads "aligned", a phantom near lead. The
            # direction is then no measurement of the mount: the mount is unknown, and a large pitch change is
            # one of the things this looks like
            vsr = float(vs) / char if fin(vs) else None
            if sp is not None and sp[0] >= MOUNT_MIN_SPEED and (v < 0.5 * sp[0] or (vsr is not None and vsr > 4.0)):
                mem["out_of_range_t"] = ts
                ctx.integ.fdi._set("vision", "mount", "model_out_of_range", "warn", ts,
                                   "road model out of its range (speed %.2f of the estimate%s): a large change of the camera's "
                                   "pitch or a blocked view; the pose direction is not a mount measurement now, the mount is unknown"
                                   % (v / sp[0], ", stated std %.0fx normal" % vsr if vsr is not None else ""))
        elif ts - mem.get("out_of_range_t", -1e18) > 5.0:
            ctx.integ.fdi.findings.pop(("vision", "mount", "model_out_of_range"), None)
        # the camera moves while the car stands (another source knows the standstill exactly): somebody is handling
        # it. Whatever the mount was, it is not known any more; its pitch is free again until the next drive.
        # (a standstill the bus only claims is not one: under its speed floor the car creeps and turns, and the pose
        # moves with it; before the standstill was confirmed that reset the mount's pitch 9 times on one drive)
        st = ctx.integ.still.get("vehicle")
        if st and st[-1][1] is not None and ts - st[-1][0] < 0.5 and ts - st[-1][1] > 2.0 and \
                (ctx.integ.stand is None or ctx.integ.stand.state == 2):
            pr = pose.get("pitch_rate")
            moving = abs(v) > HANDLED_SPEED or (fin(r) and abs(r) > HANDLED_RATE) or (fin(pr) and abs(pr) > HANDLED_RATE)
            hq = mem.setdefault("handled", [])
            hq.append((ts, moving))
            while hq and hq[0][0] < ts - 2.0:
                hq.pop(0)
            if sum(1 for _, b in hq if b) >= HANDLED_FRAMES and ts - mem.get("handled_t", -1e18) > 10.0:
                mem["handled_t"] = ts
                mem["handled_n"] = mem.get("handled_n", 0) + 1
                a = next(a for a in self.aux if a.name == "dpitch")
                ctx.integ.reset_aux(idp, ctx.val("dpitch"), a.sigma0)
                mem["mount_n"] = 0
                ctx.integ.fdi.event("vision", "camera_handled", None, "the camera moved while the car stood (handled or re-aimed): its "
                                    "pitch is unknown until the next drive", quantity="mount")
        # standstill for cars without a vehicle feed: the model's speed under the threshold for the confirmation
        # time. Leaky, so a single-frame blip (a wiper pass) does not end it; a real start does within 0.3 s.
        last = mem.get("t")
        dt = min(max(ts - last, 0.0), 1.0) if last is not None else 0.0
        mem["t"] = ts
        acc = mem.get("stop_s", 0.0)
        if v >= GO_SPEED:
            acc = 0.0
        elif v < STOP_SPEED:
            acc = min(acc + dt, STOP_CONFIRM_S + 0.3)
        else:
            acc = max(0.0, acc - 3.0 * dt)
        mem["stop_s"] = acc
        # soft: the camera infers the standstill (speed within 0.05 m/s, yaw rate within 0.005 rad/s)
        # (only while no better source holds the speed: a creeping car's wheels know it moves)
        alone = sp is None or sp[1] > 0.04 or abs(sp[0]) < 0.02
        out.append(Meas("standstill", ts, acc >= STOP_CONFIRM_S, sigma=0.05, src="vision.standstill" if alone else None,
                        role="flag"))
        return out

    def _pitch_step(self, ctx, mem, ts, vz, sig, v, idp):
        """The pitch is the one thing about this mount that changes, and it changes in steps (it is re-aimed by
        hand). A two-sided CUSUM of the pitch row's own residual finds a step within a few frames; the state is
        then put at the step's estimate with an uncertainty that says so, and converges again within seconds.
        The step is reported with its size (docs/DESIGN.md section 2)."""
        dp = ctx.val("dpitch")
        r = vz + v * dp                                   # residual of the row; in angle: -r / v
        z = max(-6.0, min(6.0, r / sig))
        cs = mem.get("cusum") or {"p": 0.0, "n": 0.0, "d": []}
        mem["cusum"] = cs
        cs["p"] = max(0.0, cs["p"] + z - PITCH_CUSUM_K)
        cs["n"] = max(0.0, cs["n"] - z - PITCH_CUSUM_K)
        if cs["p"] == 0.0 and cs["n"] == 0.0:
            cs["d"] = []
        else:
            cs["d"].append(-r / v)
            del cs["d"][:-40]
        pend = mem.get("pitch_step")
        if max(cs["p"], cs["n"]) >= PITCH_CUSUM_H and len(cs["d"]) >= 6 and ctx.std("dpitch") < 0.01:
            d = sorted(cs["d"])
            step = d[len(d) // 2]
            if abs(step) >= PITCH_STEP_MIN:
                ctx.integ.reset_aux(idp, dp + step, max(0.01, 0.5 * abs(step)))
                mem["pitch_step"] = {"t": ts, "from": (pend or {}).get("from", dp), "n": mem.get("mount_n", 0)}
                mem["steps"] = mem.get("steps", 0) + 1
                ctx.integ.fdi._set("vision", "mount", "pitch_step", "warn", ts,
                                   "camera pitch changed by about %+.1f deg; re-estimating" % math.degrees(step))
            mem["cusum"] = None
        elif pend is not None and ctx.std("dpitch") < 0.003 and mem.get("mount_n", 0) - pend["n"] >= 40:
            step = dp - pend["from"]
            mem["pitch_step"] = None
            ctx.integ.fdi.findings.pop(("vision", "mount", "pitch_step"), None)
            mem["last_step"] = {"rad": round(step, 4), "deg": round(math.degrees(step), 2)}
            ctx.integ.fdi.event("vision", "mount_step", None, "camera pitch changed by %+.2f deg (%+.4f rad, +- %.2f deg); residual "
                                "against the calibration in use now %+.4f rad" % (math.degrees(step), step, math.degrees(2 * ctx.std("dpitch")), dp))

    def report(self, ctx):
        return {"road_model": ctx.mem.get("road_model")}

    def calib_state(self, ctx):
        """The mount as the estimator sees it: residual angles against the calibration in use, their sum with it
        (the calibration this camera should run with), and a state word."""
        cal = ctx.mem.get("calib")
        dy, dp = ctx.val("dyaw"), ctx.val("dpitch")
        sy, sp = ctx.std("dyaw"), ctx.std("dpitch")
        nm = [ctx.integ.nis.get((self.group, q)) for q in ("mount_yaw", "mount_pitch")]
        if ctx.faulted():
            state = "unknown"
        elif ctx.mem.get("pitch_step") is not None:
            state = "pitch_changed"
        elif any(m is not None and m.high for m in nm):
            state = "moved"
        elif max(sy, sp) > 0.005 or ctx.mem.get("mount_n", 0) < 100:
            state = "converging"
        else:
            state = "ok" if max(abs(dy), abs(dp)) < 0.02 else "offset"
        out = {"state": state, "samples": ctx.mem.get("mount_n", 0), "pitch_steps": ctx.mem.get("steps", 0),
               "handled": ctx.mem.get("handled_n", 0),
               "freedom": "pitch (yaw fixed)"}
        if ctx.mem.get("last_step"):
            out["last_pitch_step"] = ctx.mem["last_step"]
        if cal is not None and fin(cal[0]) and fin(cal[1]):
            out["in_use"] = {"pitch": cal[0], "yaw": cal[1]}
            out["suggested"] = {"pitch": round(cal[0] + dp, 4), "yaw": round(cal[1] + dy, 4)}
        return out
