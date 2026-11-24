"""The decision layer's plug-in face: `radar` (and the prediction's futures) in, `decide` out.

In: `radar` (trigger: one decision per radar message; the stack's pace, `pace.json`, leaves out messages while
nothing could matter: plugins/common.py `Pace`), `radar_fut` (the prediction layer's companion message,
when it publishes one; trigger), `road` (the plan: the car's own path while no `ego_fut` is given), `ego_state`
(the car's acceleration and yaw rate: is the driver braking or steering away), `vehicle` (brake pedal), `env`
(the road condition: wet, snow). `radar_fut` is taken as the predictor's own object (arrays) when it runs in this
process, as the message otherwise (a recording, another process): the same numbers either way. Out: `decide` (docs/TOPICS.md): per judged object its threat measures and level
with the reasoning, and every alert's stage, mode and reasons.

Futures (cfg `futures`): "auto" the contract's fields in `radar` or `radar_fut`, else the degraded adapter per
object; "cv"
the adapter always. The adapter in use is reported in health (`degraded`)."""
import math

from ..decide import alerts as AL, futures as FU, threat as TH
from ..mixer import In, Out, Plugin
from ..native import round

NS = 1e9


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _stamp(m):
    """`ts_mono_ns` of a `radar_fut`: the message, or the predictor's own object (in process)."""
    return m.get("ts_mono_ns") if m.__class__ is dict else m.ts_mono_ns


class Decide(Plugin):
    name = "decide"
    inputs = (In("radar", trigger=True), In("radar_fut", trigger=True, optional=True, native=True),
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
        # What is hidden (off): gaps between vehicles standing beside the car's way from which someone could step
        # out. The alert `hidden_gap` (at most caution; shadow unless configured) and the message's `gap` say where
        # the nearest such gap is, how far its vehicles' faces are from the car's swept strip, and the speed from
        # which the car could still stop for a pedestrian stepping out there at the worst moment (walking at
        # ped_mps, the driver reacting in react_s and braking at decel_mps2): v_stop = 2 a (clear / v_ped - t_react).
        # A gap counts while the car could neither stop for it nor be past before the pedestrian is in its way.
        gaps=False,
        gap=dict(ped_mps=1.5, react_s=1.0, decel_mps2=6.0, clear_max_m=2.0, clear_min_m=0.1, gap_min_m=0.6, seen_m=1.5,
                 margin_mps=1.0),
    )

    def __init__(self, cfg=None, name=None):
        super().__init__(cfg, name)
        c = self.cfg
        self.fcfg = dict(FU.DEFAULT_CFG, **(c.get("future") or {}))
        self.model = TH.ThreatModel(c.get("threat"), self.fcfg, c["list_min_p"])
        acfg = dict(c.get("alerts") or {})
        self._gcfg = None
        if c["gaps"] and hasattr(TH._ND, "decide_gaps"):         # (a native core without it: the knob does nothing)
            import numpy as np
            g = dict(self.DEFAULT_CFG["gap"], **(c.get("gap") or {}))
            self._gcfg = np.array([self.model.cfg["front_m"], 0.5 * self.model.cfg["width_m"], g["ped_mps"], g["react_s"],
                                   g["decel_mps2"], g["clear_max_m"], g["gap_min_m"], g["seen_m"], g["margin_mps"],
                                   g["clear_min_m"]], float)
            acfg.setdefault("hidden_gap", {})
        self.alerts = AL.AlertSet(acfg)
        self.reset()

    def reset(self, reason=None):
        self.alerts.reset()
        self.seq = 0
        self._ts = None
        self._comp_t = None               # when a `radar_fut` was last seen (s, bus)
        self._pending = None
        self.shown = {}                   # object id -> the level shown (hysteresis between entry and exit; the
                                          # objects with a record: decide.c)
        self.stats = {"frames": 0, "cv_objects": 0, "contract_objects": 0, "invalid_futures": 0, "ego_own": 0}
        self.last = None

    # ---- inputs ----------------------------------------------------------------------------------------------
    def on_input(self, stream, msg, ctx):
        if stream == "radar_fut":
            self._comp_t = ctx.now
            if self._pending is not None and self._pending.get("ts_mono_ns") == _stamp(msg):
                radar, self._pending = self._pending, None
                self._decide(radar, msg, ctx)
            return
        if self._pending is not None:                 # its futures never came: decided without them
            radar, self._pending = self._pending, None
            self._decide(radar, None, ctx)
        pace = ctx.service("pace")
        if pace is not None and not pace.due(msg):    # a message the stack's pace leaves out (plugins/common.py Pace)
            return
        if self.cfg["futures"] == "auto" and "fut" not in msg and self._comp_t is not None \
                and ctx.now - self._comp_t <= self.cfg["companion_wait_s"] + 1.0:
            comp = ctx.get("radar_fut")
            if comp is not None and _stamp(comp) == msg.get("ts_mono_ns"):
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
        objs_in = radar.get("objects") or ()
        if objs_in.__class__ is not list:
            objs_in = list(objs_in)
        path = FU.plan_path(ctx.get("road"), ts, self.fcfg["plan_max_age_s"]) if objs_in else None
        if c["futures"] == "cv":
            H = FU.from_message(radar, None, cv=True)
        elif comp is not None and comp.__class__ is not dict:
            H = comp.handoff()                        # the predictor in this process: its arrays, no message
        else:
            H = FU.from_message(radar, comp)
        model = self.model
        road = TH.road_from_env(ctx.get("env")) if model.cfg["road_from_env"] else None
        # the native core: the futures on one grid, each footprint, the threat, the records of the objects that
        # matter (`recs`: to an alert or to the list; `out_objs`: the list, in its order) and the level shown
        (recs, out_objs, objs, self.shown, judged, n_cv, invalid, ego_own, _, al, at, ego_modes, horizon, _) = model.frame(
            objs_in, radar.get("ego"), H, path, road, self.shown, 1 if c["pairs"] else 0)
        actx = self._context(radar, ctx, T)
        gap = None
        if self._gcfg is not None:
            if path is None and objs_in:
                path = FU.plan_path(ctx.get("road"), ts, self.fcfg["plan_max_age_s"])
            gap = TH._ND.decide_gaps(objs_in, actx["v"], path, self._gcfg)
            actx["gap"] = gap
        alerts, events = self.alerts.step(T, recs, objs, actx)
        st = self.stats
        st["frames"] += 1
        st["cv_objects"] += n_cv
        st["contract_objects"] += judged - n_cv
        st["invalid_futures"] += invalid
        st["ego_own"] += ego_own
        self.seq += 1
        msg = {"type": "decide", "api": 1, "seq": self.seq, "ts_mono_ns": ts, "ts_real_ns": radar.get("ts_real_ns"),
               "radar_seq": radar.get("seq"), "frame": "car",
               "ego": {"speed_mps": round(float((radar.get("ego") or {}).get("speed_mps") or 0.0), 2),
                       "a_avail_mps2": [round(al, 2), round(at, 2)], "road": road or model.cfg["road"],
                       "t_react_s": round(model.t_react, 2), "driver": {k: (round(v, 2) if _fin(v) else v) for k, v in
                                                                        actx.items() if k in ("a_long", "a_lat", "brake")}},
               "futures": {"objects": "cv" if n_cv == judged and judged else ("contract" if not n_cv else "mixed"),
                           "cv": n_cv, "invalid": invalid, "ego": "own" if ego_own else "contract",
                           "ego_modes": ego_modes, "horizon_s": round(horizon, 2)},
               "objects": out_objs, "alerts": alerts,
               "primary": next((a for a in alerts if a["id"] == "obstacle_ahead" and a["stage"] > 0), None),
               "events": events,
               "counts": {"judged": judged, "listed": len(out_objs), "steps": model.last_n}}
        if self._gcfg is not None:
            msg["gap"] = gap
        self.last = msg
        pace = ctx.service("pace")
        if pace is not None:
            pace.report(ts, self._hot(recs, pace.cfg))
        ctx.emit("decide", msg)

    def _hot(self, recs, pc):
        """Could something matter soon (the stack's pace keeps the full rate then)? An object at level `hot_level` or
        above (entry or exit), with a collision probability from `hot_p` or a headway under `thw_s`; an alert at a
        stage, or with a level held on its way to one."""
        for a in self.alerts.alerts.values():
            # (not what is hidden between parked cars: that caution is slow by nature, the low rate judges it, and
            # along a parked row it would hold the full rate for the whole street)
            if a.cfg["source"] != "gap" and (a.stage or a.up_since or a.challenger):
                return True
        hot_p, thw_s, lv = pc["hot_p"], pc["thw_s"], pc["hot_level"]
        for r in recs:
            if r.get("level", 0) >= lv or r.get("level_exit", 0) >= lv or (r.get("p_coll") or 0.0) >= hot_p:
                return True
            thw = r.get("thw_s")
            if thw is not None and thw < thw_s:
                return True
        return False

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
