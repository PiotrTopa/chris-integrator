"""The road context of a frame, from the stack's own messages.

`ContextBuilder` keeps the newest of each source and, when asked, puts them into the car frame of the radar
message's time (they are stamped with their camera frame's time; the car has moved since) as the dict the
predictor reads (the package's docstring lists its keys):

    roadgeom (view.road's geometry)   the ego lane's centre line (camera near the car, the map beyond), the lane
                                      width, the side roads of the junctions ahead (arms) with the junction's control
    lanes                             the neighbouring lanes seen (both lines present), the road edges (kerbs), the
                                      map's lane counts when the tracker carries them (`prior`); the centre line
                                      from the two inner lines when there is no geometry message
    egolane                           how many lanes of the car's direction lie to its right and left
    road                              the road model's plan: the path where no line is tracked
    map                               one-way or two-way (`road.oneway`, `road.lanes.backward`)

Everything is optional: with nothing at all the dict holds no centre line and the predictor falls back to the arc
of the car's own motion.
"""
import math

import numpy as np

NS = 1e9
DEFAULT_CFG = dict(
    max_age_s=1.5,               # a source older than this is not used
    centre_step_m=4.0,           # the centre line is thinned to about this spacing
    centre_conf=0.3,             # a geometry point below this confidence ends the centre line
    kerb_x_m=(8.0, 30.0),        # the road edges are read over this stretch ahead
)


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class ContextBuilder:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.reset()

    def reset(self, reason=None):
        self.geom = self.lanes = self.plan = self.egolane = self.map = None
        self.turn_signal = None

    # ---- inputs (each keeps the newest message and its time) ------------------------------------------------
    def on_roadgeom(self, msg):
        g = msg.get("geom") if isinstance(msg, dict) else None
        if isinstance(g, dict) and isinstance(msg.get("ts_mono_ns"), int):
            self.geom = (msg["ts_mono_ns"] / NS, g)

    def on_lanes(self, msg):
        if isinstance(msg.get("ts_mono_ns"), int):
            self.lanes = (msg["ts_mono_ns"] / NS, msg)

    def on_road(self, msg):
        plan, ts = msg.get("plan"), msg.get("ts_mono_ns")
        if isinstance(plan, dict) and isinstance(plan.get("x"), list) and isinstance(plan.get("y"), list) \
                and len(plan["x"]) >= 2 and len(plan["x"]) == len(plan["y"]) and isinstance(ts, int):
            self.plan = (ts / NS, plan)

    def on_egolane(self, msg):
        if isinstance(msg.get("est"), dict) and isinstance(msg.get("ts_mono_ns"), int):
            self.egolane = (msg["ts_mono_ns"] / NS, msg["est"])

    def on_map(self, msg):
        if isinstance(msg.get("ts_mono_ns"), int):
            self.map = (msg["ts_mono_ns"] / NS, msg)

    def on_ego(self, msg):
        s = msg.get("turn_signal")
        self.turn_signal = s if s in ("left", "right") else None

    # ---- the context of time T --------------------------------------------------------------------------------
    def _fresh(self, e, T):
        return e is not None and -0.5 <= T - e[0] <= self.cfg["max_age_s"]

    def build(self, T, ego):
        """The context dict for time T (s); ego: the predictor's EgoTrack (moves a source's frame to T's)."""
        c = self.cfg
        out = {"source": "kinematic", "turn_signal": self.turn_signal}
        centre = None
        if self._fresh(self.geom, T):
            t, g = self.geom
            ce = g.get("centre")
            if isinstance(ce, dict) and isinstance(ce.get("x"), list) and len(ce["x"]) >= 2:
                x, y = np.asarray(ce["x"], float), -np.asarray(ce["y"], float)        # the geometry's y is right
                conf = ce.get("conf")
                n = len(x)
                if isinstance(conf, list) and len(conf) == n:
                    low = np.flatnonzero(np.asarray([v if _fin(v) else 0.0 for v in conf]) < c["centre_conf"])
                    if len(low) and low[0] >= 2:
                        n = int(low[0])
                centre = ego.move(np.stack([x[:n], y[:n]], axis=1), t, T)
                out["source"] = "roadgeom" if "m" not in (ce.get("src") or "")[:n] else "roadgeom+map"
                out["quality"] = 0.9
            if _fin(g.get("lane_width_m")):
                out["lane_width_m"] = g["lane_width_m"]
            arms = []
            ctrl = (g.get("junction") or {}).get("control")
            jd = (g.get("junction") or {}).get("dist_m")
            for b in g.get("branches") or ():
                pts = b.get("pts")
                if not isinstance(pts, list) or len(pts) < 2 or b.get("ring"):
                    continue
                p = np.asarray(pts, float)
                p[:, 1] = -p[:, 1]
                near = _fin(jd) and _fin(b.get("dist_m")) and abs(b["dist_m"] - jd) < 25.0
                arms.append({"pts": ego.move(p, t, T).tolist(), "side": b.get("side"),
                             "priority": -1 if b.get("minor") else 0, "entry": bool(b.get("entry", True)),
                             "exit": bool(b.get("exit", True)), "control": ctrl if near else None})
            if arms:
                out["arms"] = arms
        if self._fresh(self.lanes, T):
            t, ln = self.lanes
            xs = ln.get("x")
            lines = ln.get("lines") or ()
            if centre is None and isinstance(xs, list) and len(lines) == 4 and ln.get("valid", True):
                l, r = lines[1], lines[2]
                if l.get("present") and r.get("present") and isinstance(l.get("y"), list) and isinstance(r.get("y"), list):
                    x = np.asarray(xs, float)
                    y = -0.5 * (np.asarray(l["y"], float) + np.asarray(r["y"], float))
                    sd = np.asarray(l.get("y_std") or [0.0] * len(x), float)
                    n = int(np.argmax(sd > 0.9)) if (sd > 0.9).any() else len(x)
                    keep = (x[:n] <= 100.0)
                    if keep.sum() >= 2:
                        centre = ego.move(np.stack([x[:n][keep], y[:n][keep]], axis=1), t, T)
                        out["source"], out["quality"] = "lanes", 0.8
            if _fin(ln.get("lane_width_m")) and "lane_width_m" not in out:
                out["lane_width_m"] = ln["lane_width_m"]
            # a neighbouring lane the camera sees says "a lane is there"; one it does not see says little
            seen_l, seen_r = bool(ln.get("lane_to_left")), bool(ln.get("lane_to_right"))
            out["_seen"] = (seen_l, seen_r)
            pr = ln.get("prior")
            if isinstance(pr, dict) and _fin(pr.get("forward")) and _fin(pr.get("backward")):
                out["_count"] = (int(pr["forward"]), int(pr["backward"]))
            edges = ln.get("edges") or ()
            if isinstance(xs, list) and len(edges) == 2:
                x = np.asarray(xs, float)
                m = (x >= c["kerb_x_m"][0]) & (x <= c["kerb_x_m"][1])
                cy = 0.0
                if centre is not None and len(centre) >= 2:
                    cy = float(np.interp(0.5 * sum(c["kerb_x_m"]), centre[:, 0], centre[:, 1]))
                for e, name in zip(edges, ("kerb_left_m", "kerb_right_m")):
                    if e.get("present") and isinstance(e.get("y"), list) and m.any():
                        out[name] = float(-np.mean(np.asarray(e["y"], float)[m])) - cy
        if centre is None and self._fresh(self.plan, T):
            t, plan = self.plan
            x, y = np.asarray(plan["x"], float), -np.asarray(plan["y"], float)
            if np.all(np.diff(x) >= 0.0) and x[-1] > 5.0:
                centre = ego.move(np.stack([x, y], axis=1), t, T)
                out["source"], out["quality"] = "plan", 0.5
        if centre is not None and len(centre) >= 2:
            # thinned; the stretch behind the car is the predictor's to extend
            d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(centre[:, 0]), np.diff(centre[:, 1])))])
            keep = [0]
            for i in range(1, len(d)):
                if d[i] - d[keep[-1]] >= c["centre_step_m"] or i == len(d) - 1:
                    keep.append(i)
            out["centre"] = centre[keep]
        # how many lanes run beside the car's own
        fwd = bwd = None
        if "_count" in out:
            fwd, bwd = out.pop("_count")
        if self._fresh(self.map, T + 3.0):
            rd = (self.map[1].get("road") or {})
            ln = rd.get("lanes") or {}
            if fwd is None and _fin(ln.get("forward")) and _fin(ln.get("backward")):
                fwd, bwd = int(ln["forward"]), int(ln["backward"])
            if rd.get("oneway") is True:
                out["oncoming"] = 0.03
        seen = out.pop("_seen", None)
        right = None
        if self._fresh(self.egolane, T + 1.0):
            est = self.egolane[1]
            pr = est.get("probs")
            if isinstance(pr, list) and pr:
                right = list(pr) if est.get("probs_from") != "left" else list(reversed(pr))   # P(k lanes to the right)
        if fwd is not None:
            if right is not None and len(right) == fwd:
                out["n_right"] = right[:3]
                out["n_left"] = list(reversed(right))[:3]
            else:
                out["n_right"] = out["n_left"] = [1.0 / fwd] * min(fwd, 3) if fwd <= 3 else [1.0 / 3] * 3
            out["oncoming"] = 0.97 if bwd else 0.03
            out["n_oncoming"] = max(1, bwd or 1)
        elif seen is not None:
            # the camera alone: a lane seen on a side is a lane (of the car's direction or the oncoming one, which
            # the two-way prior splits); none seen leaves the prior
            if right is not None and len(right) >= 2:
                out["n_right"] = right[:3]
            else:
                out["n_right"] = 0.85 if seen[1] else 0.15
            out["_left_seen"] = seen[0]
        return out
