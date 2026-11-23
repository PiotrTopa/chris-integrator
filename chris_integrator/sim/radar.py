"""A simulated radar: per scan the detections of every vehicle and roadside object in its field of view
(range, azimuth, range rate, with noise and missed detections), clutter, and double-bounce ghosts of the
nearest vehicle ahead; emitted as `objects` messages of the vehicle input API, polar in the sensor's frame.
Mount anywhere on the car (x, y in the car frame, yaw of the boresight); `misalign` is a true yaw error the
integrator has to find."""
import math
import random

from .world import CAM_TO_AXLE, NS


# The extended model (RadarSim(extended=True)): returns per vehicle per scan ~ Poisson(rate), rate = n20 x
# (20 / r) ** range_exp, capped at n_max (car ~4 at 20 m, ~1.3 at 80 m); each return a point on the footprint edges
# that face the sensor (the L of a car seen from a corner), with p_corner of them at the corner nearest to it;
# a return whose line of sight passes through a nearer actor's footprint is dropped with p_shadow; static clutter
# (points that stand on the ground: guard rails, posts, kerbs) Poisson per scan; the point model's random clutter
# and the double-bounce ghost are kept.
EXT = dict(n20={"car": 4.0, "truck": 8.0, "human": 0.7}, range_exp=0.8, n_max=10.0, p_corner=0.35, p_shadow=0.85,
           static_clutter=3.0, static_clutter_range=(4.0, 90.0), static_clutter_lat=(4.0, 12.0))


class RadarSim:
    """Point model (default): one detection per vehicle (its footprint centre) per scan, as it always was, each
    with a `truth` key (the target id; -1 ghost, -2 clutter). extended=True (see EXT; `ext` overrides): several
    returns per vehicle over the footprint edges facing the sensor, range rate of that point's own ground velocity,
    shadowing, static clutter; the detections carry NO truth key: what each return came from is kept on the sim
    (`truth_log`: [(t_scan, [actor id | -1 ghost | -2 clutter | -3 static clutter, ...] in message order)])."""

    def __init__(self, sensor="radar_front", mount=(2.2, 0.0, 0.0), seed=51, hz=17.0, latency=0.05, fov_deg=50.0,
                 max_range=160.0, sigma=(0.2, 0.4, 0.12), p_d=0.92, p_d_static=0.7, clutter=2.0, ghost=0.3,
                 misalign_deg=0.0, blackout=(), with_mount=True, extended=False, ext=None):
        self.__dict__.update(locals())
        self.ext = dict(EXT, **(ext or {}))
        self.truth_log = []

    def _det(self, o, tr, rng, mx, my, yaw):
        dx, dy = o["x"] - mx, o["y"] - my
        r = math.hypot(dx, dy)
        az = math.atan2(dy, dx) - yaw
        az = (az + math.pi) % (2.0 * math.pi) - math.pi
        if r < 1.0 or r > self.max_range or abs(az) > math.radians(self.fov_deg):
            return None
        vsx, vsy = tr["v"] - tr["w"] * my, tr["w"] * (mx + CAM_TO_AXLE)
        rr = (dx * (o["vx"] - vsx) + dy * (o["vy"] - vsy)) / r
        return r, az, rr

    def generate(self, w):
        if self.extended:
            yield from self._generate_ext(w)
            return
        rng = random.Random(self.seed)
        mx, my, myaw = self.mount
        yaw = myaw + math.radians(self.misalign_deg)          # where the sensor really points
        t = w.t0 + 0.07
        sr, sa, sv = self.sigma[0], math.radians(self.sigma[1]), self.sigma[2]
        while t < w.t0 + w.T:
            rel = t - w.t0
            if not any(a <= rel < b for a, b in self.blackout):
                tr = w.truth_at(t)
                objs = []
                near = None
                for o in w.targets_at(t) + w.statics_at(t):
                    d = self._det(o, tr, rng, mx, my, yaw)
                    if d is None or rng.random() > (self.p_d_static if o["kind"] == "static" else self.p_d):
                        continue
                    r, az, rr = d
                    objs.append((r + rng.gauss(0, sr), az + rng.gauss(0, sa), rr + rng.gauss(0, sv), o["id"]))
                    if o["kind"] != "static" and abs(o["y"]) < 1.5 and o["x"] > 0 and (near is None or r < near[0]):
                        near = d
                if near is not None and 2.0 * near[0] < self.max_range and rng.random() < self.ghost:
                    objs.append((2.0 * near[0] + rng.gauss(0, 2 * sr), near[1] + rng.gauss(0, sa), 2.0 * near[2] + rng.gauss(0, 2 * sv), -1))
                for _ in range(rng.choice([0, 1, 2, 3, 4]) if self.clutter else 0):
                    if rng.random() < self.clutter / 2.0:
                        objs.append((rng.uniform(3.0, self.max_range), rng.uniform(-1, 1) * math.radians(self.fov_deg),
                                     rng.uniform(-35.0, 10.0), -2))
                msg = {"type": "objects", "sensor": self.sensor, "kind": "radar", "tracked": False, "ts_mono_ns": int(round(t * NS)),
                       "objects": [{"range_m": round(r, 2), "azimuth_deg": round(math.degrees(az), 3),
                                    "range_rate_mps": round(rr, 3), "truth": i} for r, az, rr, i in objs]}
                if self.with_mount:
                    msg["mount"] = {"x_m": mx, "y_m": my, "yaw_deg": math.degrees(myaw)}
                yield t + self.latency + rng.uniform(0, 0.01), msg
            t += 1.0 / self.hz + rng.uniform(-0.003, 0.003)

    # ---- the extended model ---------------------------------------------------------------------------------
    @staticmethod
    def _footprint(o):
        """Corners (4, 2) of an actor's footprint in the car frame, counter-clockwise."""
        h = o.get("heading", 0.0)
        c, s = math.cos(h), math.sin(h)
        L, W = o.get("L", 4.5), o.get("W", 1.8)
        out = []
        for lx, ly in ((L / 2, W / 2), (-L / 2, W / 2), (-L / 2, -W / 2), (L / 2, -W / 2)):
            out.append((o["x"] + lx * c - ly * s, o["y"] + lx * s + ly * c))
        return out

    @staticmethod
    def _seg_hits(p0, p1, q0, q1):
        """Does segment p0-p1 cross segment q0-q1."""
        def cr(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        d1, d2 = cr(q0, q1, p0), cr(q0, q1, p1)
        d3, d4 = cr(p0, p1, q0), cr(p0, p1, q1)
        return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)

    def _returns(self, o, rng, mx, my):
        """Points (x, y) on the edges of o's footprint that face the sensor at (mx, my)."""
        e = self.ext
        P = self._footprint(o)
        r = math.hypot(o["x"] - mx, o["y"] - my)
        lam = min(e["n_max"], e["n20"].get(o.get("kind"), 0.0) * (20.0 / max(r, 3.0)) ** e["range_exp"])
        n = _poisson(rng, lam)
        if n == 0:
            return []
        edges = []
        for k in range(4):
            a, b = P[k], P[(k + 1) % 4]
            mxe, mye = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            nx, ny = b[1] - a[1], -(b[0] - a[0])                 # outward normal of a counter-clockwise polygon
            if nx * (mx - mxe) + ny * (my - mye) > 0.0:
                edges.append((a, b, math.hypot(b[0] - a[0], b[1] - a[1])))
        if not edges:
            return []
        corner = min(P, key=lambda q: (q[0] - mx) ** 2 + (q[1] - my) ** 2)
        tot = sum(ed[2] for ed in edges)
        pts = []
        for _ in range(n):
            if rng.random() < e["p_corner"]:
                pts.append(corner)
                continue
            u = rng.random() * tot
            for a, b, ln in edges:
                if u <= ln:
                    f = u / ln if ln > 0 else 0.0
                    pts.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
                    break
                u -= ln
            else:
                a, b, _ = edges[-1]
                pts.append(b)
        return pts

    def _generate_ext(self, w):
        rng = random.Random(self.seed * 7919 + 17)
        e = self.ext
        mx, my, myaw = self.mount
        yaw = myaw + math.radians(self.misalign_deg)
        fov = math.radians(self.fov_deg)
        t = w.t0 + 0.07
        sr, sa, sv = self.sigma[0], math.radians(self.sigma[1]), self.sigma[2]
        actors_at = getattr(w, "actors_at", None)
        while t < w.t0 + w.T:
            rel = t - w.t0
            if not any(a <= rel < b for a, b in self.blackout):
                tr = w.truth_at(t)
                vsx, vsy = tr["v"] - tr["w"] * my, tr["w"] * (mx + CAM_TO_AXLE)      # the sensor over the ground
                if actors_at is not None:
                    objs_t = [o for o in actors_at(t, max_range=self.max_range + 10.0)]
                else:                                            # a plain World: its targets as cars
                    objs_t = [dict(o, heading=math.atan2(o["vy"], o["vx"]) if math.hypot(o["vx"], o["vy"]) > 0.3
                                   else 0.0, L=4.5, W=1.8) for o in w.targets_at(t)]
                vis = []
                for o in objs_t:
                    r = math.hypot(o["x"] - mx, o["y"] - my)
                    if r < 1.0 or r > self.max_range + 5.0:
                        continue
                    vis.append((r, o, self._footprint(o)))
                vis.sort(key=lambda v: v[0])
                dets = []
                near = None
                for j, (r_o, o, P) in enumerate(vis):
                    if rng.random() > self.p_d:
                        continue
                    for px, py in self._returns(o, rng, mx, my):
                        dx, dy = px - mx, py - my
                        r = math.hypot(dx, dy)
                        az = (math.atan2(dy, dx) - yaw + math.pi) % (2.0 * math.pi) - math.pi
                        if r < 1.0 or r > self.max_range or abs(az) > fov:
                            continue
                        hid = False
                        for r2, o2, P2 in vis[:j]:
                            if r2 < r and any(self._seg_hits((mx, my), (px, py), P2[k], P2[(k + 1) % 4]) for k in range(4)):
                                hid = True
                                break
                        if hid and rng.random() < e["p_shadow"]:
                            continue
                        rr = (dx * (o["vx"] - vsx) + dy * (o["vy"] - vsy)) / r       # the point's ground velocity
                        dets.append((r + rng.gauss(0, sr), az + rng.gauss(0, sa), rr + rng.gauss(0, sv), o["id"]))
                        if o.get("kind") != "human" and abs(o["y"]) < 1.5 and o["x"] > 0 and (near is None or r < near[0]):
                            near = (r, az, rr)
                if near is not None and 2.0 * near[0] < self.max_range and rng.random() < self.ghost:
                    dets.append((2.0 * near[0] + rng.gauss(0, 2 * sr), near[1] + rng.gauss(0, sa),
                                 2.0 * near[2] + rng.gauss(0, 2 * sv), -1))
                for _ in range(rng.choice([0, 1, 2, 3, 4]) if self.clutter else 0):
                    if rng.random() < self.clutter / 2.0:
                        dets.append((rng.uniform(3.0, self.max_range), rng.uniform(-1, 1) * fov,
                                     rng.uniform(-35.0, 10.0), -2))
                for _ in range(_poisson(rng, e["static_clutter"])):
                    x = rng.uniform(*e["static_clutter_range"])
                    y = rng.choice((-1, 1)) * rng.uniform(*e["static_clutter_lat"])
                    dx, dy = x - mx, y - my
                    r = math.hypot(dx, dy)
                    az = (math.atan2(dy, dx) - yaw + math.pi) % (2.0 * math.pi) - math.pi
                    if r > self.max_range or abs(az) > fov:
                        continue
                    rr = (dx * -vsx + dy * -vsy) / r
                    dets.append((r + rng.gauss(0, sr), az + rng.gauss(0, sa), rr + rng.gauss(0, sv), -3))
                rng.shuffle(dets)
                msg = {"type": "objects", "sensor": self.sensor, "kind": "radar", "tracked": False, "extended": True,
                       "ts_mono_ns": int(round(t * NS)),
                       "objects": [{"range_m": round(r, 2), "azimuth_deg": round(math.degrees(az), 3),
                                    "range_rate_mps": round(rr, 3)} for r, az, rr, _ in dets]}
                self.truth_log.append((t, [d[3] for d in dets]))
                if self.with_mount:
                    msg["mount"] = {"x_m": mx, "y_m": my, "yaw_deg": math.degrees(myaw)}
                yield t + self.latency + rng.uniform(0, 0.01), msg
            t += 1.0 / self.hz + rng.uniform(-0.003, 0.003)


def _poisson(rng, lam):
    """A Poisson draw (Knuth; lam is small here)."""
    if lam <= 0.0:
        return 0
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def attach(world, radar="point", seed=None, **kw):
    """A RadarSim for a scenario run: radar "point" (today's model) or "extended"; seed defaults to one derived
    from the world's; other keywords go to RadarSim. Add it to the run's sensors (world.messages(sensors))."""
    if radar not in ("point", "extended"):
        raise ValueError("radar must be 'point' or 'extended'")
    if seed is None:
        seed = 51 + 97 * int(getattr(world, "seed", 1) or 1)
    return RadarSim(seed=seed, extended=radar == "extended", **kw)
