"""The car's own movement: its pose on the ground over time.

The ground frame is local: origin and axes are where the reference point of the car (the car frame's origin:
the reference camera, `axle_m` ahead of the rear axle) was at the first sample, x along the car's heading
then, y to its LEFT. It drifts like any dead reckoning; that does not matter here, because the objects live
in the same frame and only the car's movement between two moments enters what is concluded about them. How
uncertain that movement is (speed and yaw rate errors) is reported per interval (`motion_noise`) and added
to the objects' position uncertainty.

Samples: (t, speed, yaw rate, longitudinal acceleration) from `ego_state` (the integrator's fused estimate
from the wheels, the gyro, the accelerometer and the camera) or straight from a vehicle feed. Between samples
the car turns at the mean yaw rate and its speed changes linearly; beyond the newest sample it carries on with
the last speed, yaw rate and acceleration for a short while.
"""
import bisect
import math

AXLE_M = 1.5                     # the car frame's origin is this far ahead of the rear axle
KEEP_S = 30.0                    # pose history kept
EXTRAPOLATE_S = 0.5              # the newest sample is trusted this far ahead


class EgoMotion:
    def __init__(self, axle_m=AXLE_M):
        self.axle = axle_m
        self.reset()

    def reset(self):
        self._at = {}            # state_at of the times asked since the last sample (one message asks a few)
        self.t = []              # sample times
        self.s = []              # (xr, yr, psi, v, w, a, sig_v, sig_w, stands) at each time: rear axle pose
        self.frame = 0           # counts resets: objects of an older frame are meaningless

    def __len__(self):
        return len(self.t)

    @property
    def newest(self):
        return self.t[-1] if self.t else None

    def add(self, t, v, w, a=0.0, sig_v=0.1, sig_w=0.003, stands=False):
        """One sample of the car's motion. Returns False for a sample that is not newer than the newest."""
        v = 0.0 if stands else float(v)
        w = 0.0 if stands else float(w)
        if self._at:
            self._at = {}
        if not self.t:
            self.t.append(t)
            self.s.append((-self.axle, 0.0, 0.0, v, w, float(a), sig_v, sig_w, bool(stands)))
            return True
        dt = t - self.t[-1]
        if dt <= 0.0:
            return False
        if dt > 2.0:             # a gap: what happened in between is not known; a new ground frame
            self.reset_keep_frame()
            self.t.append(t)
            self.s.append((-self.axle, 0.0, 0.0, v, w, float(a), sig_v, sig_w, bool(stands)))
            return True
        xr, yr, psi, v0, w0 = self.s[-1][:5]
        x1, y1, p1 = _advance(xr, yr, psi, 0.5 * (v0 + v), 0.5 * (w0 + w), dt)
        self.t.append(t)
        self.s.append((x1, y1, p1, v, w, float(a), sig_v, sig_w, bool(stands)))
        if self.t[0] < t - KEEP_S:
            k = bisect.bisect_left(self.t, t - KEEP_S)
            del self.t[:k], self.s[:k]
        return True

    def reset_keep_frame(self):
        f = self.frame + 1
        self.reset()
        self.frame = f

    def state_at(self, t):
        """(xr, yr, psi, v, w, a) of the rear axle at time t, or None before any sample."""
        s = self._at.get(t)
        if s is None:
            s = self._at[t] = self._state_at(t)
            if len(self._at) > 64:
                self._at = {t: s}
        return s

    def _state_at(self, t):
        if not self.t:
            return None
        i = bisect.bisect_right(self.t, t)
        if i == 0:
            xr, yr, psi, v, w, a = self.s[0][:6]
            x1, y1, p1 = _advance(xr, yr, psi, v, w, t - self.t[0])           # backwards from the first sample
            return x1, y1, p1, v, w, a
        t0 = self.t[i - 1]
        xr, yr, psi, v, w, a = self.s[i - 1][:6]
        dt = t - t0
        if i < len(self.t):
            t1 = self.t[i]
            v1, w1 = self.s[i][3], self.s[i][4]
            f = dt / (t1 - t0) if t1 > t0 else 0.0
            vm, wm = v + 0.5 * f * (v1 - v), w + 0.5 * f * (w1 - w)          # mean over [t0, t]
            x1, y1, p1 = _advance(xr, yr, psi, vm, wm, dt)
            return x1, y1, p1, v + f * (v1 - v), w + f * (w1 - w), a
        dt = min(dt, EXTRAPOLATE_S)
        stands = self.s[i - 1][8]
        v_end = v if stands else max(0.0, v + a * dt)
        x1, y1, p1 = _advance(xr, yr, psi, 0.5 * (v + v_end), w, dt)
        return x1, y1, p1, v_end, w, a

    def pose_at(self, t):
        """(X, Y, psi) of the car frame's origin on the ground at time t."""
        s = self.state_at(t)
        if s is None:
            return None
        xr, yr, psi = s[:3]
        return xr + self.axle * math.cos(psi), yr + self.axle * math.sin(psi), psi

    def velocity_at(self, t):
        """Ground velocity (vx, vy) of the car frame's origin and the yaw rate at time t."""
        s = self.state_at(t)
        if s is None:
            return None
        psi, v, w = s[2], s[3], s[4]
        c, sn = math.cos(psi), math.sin(psi)
        # the origin sits `axle` ahead of the rear axle: it also moves sideways when the car turns
        return v * c - w * self.axle * sn, v * sn + w * self.axle * c, w

    def to_ground(self, x, y, t):
        X, Y, psi = self.pose_at(t)
        c, s = math.cos(psi), math.sin(psi)
        return X + c * x - s * y, Y + s * x + c * y

    def to_car(self, gx, gy, t):
        X, Y, psi = self.pose_at(t)
        c, s = math.cos(psi), math.sin(psi)
        dx, dy = gx - X, gy - Y
        return c * dx + s * dy, -s * dx + c * dy

    def motion_noise(self, t0, t1):
        """(sigma along the heading, sigma of the heading) of the car's movement between two times: what the
        speed's and the yaw rate's uncertainty amount to over that interval."""
        if not self.t:
            return 0.0, 0.0
        i = max(0, min(len(self.t) - 1, bisect.bisect_right(self.t, t1) - 1))
        sv, sw, stands = self.s[i][6], self.s[i][7], self.s[i][8]
        dt = abs(t1 - t0)
        return (0.0 if stands else sv * dt), (0.0 if stands else sw * dt)

    def stands_at(self, t):
        if not self.t:
            return False
        i = max(0, min(len(self.t) - 1, bisect.bisect_right(self.t, t) - 1))
        return self.s[i][8]


def _advance(x, y, psi, v, w, dt):
    """A circular arc at speed v and yaw rate w for dt (exact for constant v, w; dt may be negative)."""
    if abs(w) < 1e-6:
        return x + v * dt * math.cos(psi), y + v * dt * math.sin(psi), psi + w * dt
    p1 = psi + w * dt
    r = v / w
    return x + r * (math.sin(p1) - math.sin(psi)), y - r * (math.cos(p1) - math.cos(psi)), p1
