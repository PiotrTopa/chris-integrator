"""The decision layer's plug-in face: `radar` (and the prediction's futures) in, `decide` out.

In: `radar` (trigger: one decision per radar message), `radar_fut` (the prediction layer's companion message,
when it publishes one; trigger), `road` (the plan: the car's own path while no `ego_fut` is given), `ego_state`
(the car's acceleration and yaw rate: is the driver braking or steering away), `vehicle` (brake pedal), `env`
(the road condition: wet, snow). Out: `decide` (docs/TOPICS.md): per judged object its threat measures and level
with the reasoning, and every alert's stage, mode and reasons.

Futures (cfg `futures`): "auto" the contract's fields in `radar` or `radar_fut`, else the degraded adapter per
object; "predictor" the prediction layer's in-process API (`chris_integrator.predict.Predictor.update`); "cv"
the adapter always. The adapter in use is reported in health (`degraded`)."""
import math

from ..decide import alerts as AL, futures as FU, threat as TH
from ..mixer import In, Out, Plugin
from ..native import round

NS = 1e9


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class Decide(Plugin):
    name = "decide"
    inputs = (In("radar", trigger=True), In("radar_fut", trigger=True, optional=True),
              In("road", optional=True), In("ego_state", optional=True), In("vehicle", optional=True),
              In("env", optional=True))
    outputs = (Out("decide", "decision", "frame",
                   doc="threat per object (collision probability, braking / steering required, time to brake / steer, "
                       "levels with reasons) and the alerts' stages (off / shadow / on)"),)
    DEFAULT_CFG = dict(
        futures="auto",
        rate_hz=20.0,                 # decisions a second at most (one per radar message at the radar's rate)
        list_min_p=0.01,              # an object is listed with p_coll from this, a level, or a headway
        pairs=False,                  # list every judged pair (explain view; larger messages)
        companion_wait_s=0.3,         # a `radar` message waits this long for its `radar_fut` once one was seen
        future=None, threat=None, alerts=None,
    )

    def __init__(self, cfg=None, name=None):
        super().__init__(cfg, name)
        c = self.cfg
        self.fcfg = dict(FU.DEFAULT_CFG, **(c.get("future") or {}))
        self.model = TH.ThreatModel(c.get("threat"))
        self.alerts = AL.AlertSet(c.get("alerts"))
        self.predictor = None
        if c["futures"] == "predictor":
            from ..predict import Predictor          # the prediction layer, in process
            self.predictor = Predictor()
        self.reset()

    def reset(self, reason=None):
        self.alerts.reset()
        self.seq = 0
        self._ts = None
        self._comp_t = None               # when a `radar_fut` was last seen (s, bus)
        self._pending = None
        self.shown = {}                   # object id -> the level shown (hysteresis between entry and exit)
        self.stats = {"frames": 0, "cv_objects": 0, "contract_objects": 0, "invalid_futures": 0, "ego_own": 0}
        self.last = None

    # ---- inputs ----------------------------------------------------------------------------------------------
    def on_input(self, stream, msg, ctx):
        if stream == "radar_fut":
            self._comp_t = ctx.now
            if self._pending is not None and self._pending.get("ts_mono_ns") == msg.get("ts_mono_ns"):
                radar, self._pending = self._pending, None
                self._decide(radar, msg, ctx)
            return
        if self.cfg["futures"] == "auto" and "fut" not in msg and self._comp_t is not None \
                and ctx.now - self._comp_t <= self.cfg["companion_wait_s"] + 1.0:
            comp = ctx.get("radar_fut")
            if comp is not None and comp.get("ts_mono_ns") == msg.get("ts_mono_ns"):
                self._decide(msg, comp, ctx)
            else:
                self._pending = msg                   # its futures follow
            return
        self._decide(msg, None, ctx)

    # ---- one decision ----------------------------------------------------------------------------------------
    def _decide(self, radar, comp, ctx):
        c = self.cfg
        ts = radar.get("ts_mono_ns")
        if not isinstance(ts, int):
            return
        if self._ts is not None and 0 <= ts - self._ts < (1.0 / c["rate_hz"] - 0.01) * NS:
            return
        self._ts = ts
        T = ts / NS
        path = FU.plan_path(ctx.get("road"), ts, self.fcfg["plan_max_age_s"])
        if c["futures"] == "cv":
            radar = {k: v for k, v in radar.items() if k not in ("fut", "ego_fut")}
            radar["objects"] = [{k: v for k, v in o.items() if k != "fut"} for o in radar.get("objects") or ()]
            comp = None
        elif self.predictor is not None:
            comp = self.predictor.update(radar, {"road": ctx.get("road"), "ego_state": ctx.get("ego_state")})
        F = FU.build(radar, self.fcfg, path, comp)
        road = TH.road_from_env(ctx.get("env")) if self.model.cfg["road_from_env"] else None
        res, info = self.model.evaluate(F, radar.get("ego") or {}, road)
        recs, out_objs = [], []
        for o, ms, src, orow, prow, ext in res:
            r = TH.object_record(o, ms, src, orow, prow, ext, F)
            # the level shown for this object: up at the entry level, down below the exit level (hysteresis)
            sh = self.shown.get(r["id"], 0)
            sh = r["level"] if r["level"] > sh else (min(sh, r["level_exit"]) if r["level_exit"] < sh else sh)
            self.shown[r["id"]] = sh
            r["shown"] = sh
            r["in_path"] = "in_path" in (o.get("flags") or ())
            recs.append(r)
            if r["level"] or sh or (r["p_coll"] or 0.0) >= c["list_min_p"] or r["thw_s"] is not None:
                if c["pairs"]:
                    r["pairs"] = TH.pairs_record(ms, prow, F)
                out_objs.append(r)
        live = {r["id"] for r in recs}
        for k in [k for k in self.shown if k not in live]:
            del self.shown[k]
        objs = {o.get("id"): o for o in radar.get("objects") or () if isinstance(o, dict)}
        actx = self._context(radar, ctx, T)
        alerts, events = self.alerts.step(T, recs, objs, actx)
        st = self.stats
        st["frames"] += 1
        n_cv = sum(1 for _, _, s, *_ in res if s == "cv")
        st["cv_objects"] += n_cv
        st["contract_objects"] += len(res) - n_cv
        st["invalid_futures"] += F.degraded["invalid"]
        st["ego_own"] += 1 if F.degraded["ego"] else 0
        self.seq += 1
        out_objs.sort(key=lambda r: (-r["level"], -(r["p_coll"] or 0.0)))
        al, at = info["a_avail_mps2"]
        msg = {"type": "decide", "api": 1, "seq": self.seq, "ts_mono_ns": ts, "ts_real_ns": radar.get("ts_real_ns"),
               "radar_seq": radar.get("seq"), "frame": "car",
               "ego": {"speed_mps": round(float((radar.get("ego") or {}).get("speed_mps") or 0.0), 2),
                       "a_avail_mps2": [round(al, 2), round(at, 2)], "road": info["road"],
                       "t_react_s": round(info["t_react_s"], 2), "driver": {k: (round(v, 2) if _fin(v) else v) for k, v in
                                                                           actx.items() if k in ("a_long", "a_lat", "brake")}},
               "futures": {"objects": "cv" if n_cv == len(res) and res else ("contract" if not n_cv else "mixed"),
                           "cv": n_cv, "invalid": F.degraded["invalid"], "ego": "own" if F.degraded["ego"] else "contract",
                           "ego_modes": [[m[0], round(m[1], 3)] for m in F.ego], "horizon_s": round(float(F.t[-1]), 2)},
               "objects": out_objs, "alerts": alerts,
               "primary": next((a for a in alerts if a["id"] == "obstacle_ahead" and a["stage"] > 0), None),
               "events": events,
               "counts": {"judged": len(res), "listed": len(out_objs), "steps": self.model.last_n}}
        self.last = msg
        ctx.emit("decide", msg)

    def _context(self, radar, ctx, T):
        """What the driver does and how fresh the sensors are: for the suppression rules."""
        eg = radar.get("ego") or {}
        v = float(eg.get("speed_mps") or 0.0)
        a_long = eg.get("accel_mps2") if _fin(eg.get("accel_mps2")) else None
        w = eg.get("yaw_rate_rps") if _fin(eg.get("yaw_rate_rps")) else None
        es = ctx.get("ego_state")
        if isinstance(es, dict):
            ac = es.get("accel") or {}
            if ac.get("valid") and _fin(ac.get("value")):
                a_long = float(ac["value"])
        a_lat = v * w if w is not None else None
        brake = None
        vh = ctx.get("vehicle")
        if isinstance(vh, dict):
            sig = vh.get("signals") or {}
            ctl = vh.get("controls") or {}
            if isinstance(ctl.get("brake"), bool):
                brake = ctl["brake"]
            elif isinstance(sig.get("brake"), dict) and isinstance(sig["brake"].get("value"), bool):
                brake = sig["brake"]["value"]
        ages = [s.get("age_s") for s in (radar.get("sensors") or {}).values() if isinstance(s, dict) and _fin(s.get("age_s"))]
        return {"v": v, "a_long": a_long, "a_lat": a_lat, "brake": brake, "sensor_age_s": min(ages) if ages else None}

    def health(self):
        st = self.stats
        deg = []
        if st["cv_objects"]:
            deg.append("futures from the constant-velocity adapter (%d object-frames; no prediction layer)" % st["cv_objects"])
        if st["ego_own"]:
            deg.append("the car's own futures from its motion and the plan (%d frames; no ego_fut)" % st["ego_own"])
        if st["invalid_futures"]:
            deg.append("%d object futures not as the contract says (the adapter stood in)" % st["invalid_futures"])
        return {"state": "degraded" if deg else ("ok" if st["frames"] else "idle"), "reason": "; ".join(deg) or None,
                "frames": st["frames"], "futures": dict(st),
                "alerts": {n: {"mode": a.cfg["mode"], "stage": a.stage, "counts": self.alerts.counts[n]}
                           for n, a in self.alerts.alerts.items()},
                "shadow_log": list(self.alerts.events)[-20:]}

    def set_mode(self, name, mode):
        return self.alerts.set_mode(name, mode)
