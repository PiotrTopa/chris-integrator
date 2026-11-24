"""Alert logic: staged alerts over the threat levels, with hysteresis, holds, latching, suppression and modes.

Stages: 0 none, 1 information, 2 caution, 3 warning, 4 imminent. Per alert (`ALERTS`, configurable):

* candidates: the judged objects of the alert's kinds that are eligible (state confirmed or coast, existence at
  least `exist_min`, seen within `seen_max_s`); each with an entry level and an exit level (threat.py: the exit
  thresholds are the entry ones relaxed: hysteresis on the measure itself);
* one primary threat: the candidate with the highest entry level (then the larger probability mass, then the
  shorter time margin); a challenger replaces the primary only when it ranks higher for `switch_hold_s`, or
  when the primary is gone: the choice does not hop between two similar objects;
* the stage rises to the primary's entry level once that level has held for `entry_hold_s[stage]`; it falls
  to the primary's exit level once that has held below the stage for `exit_hold_s[stage]` and the stage has been
  shown for `min_hold_s[stage]`; `imminent` latches until the exit level is at most information (held) or the
  car stands;
* suppression (the stage is kept, nothing is said, the reason is given): the driver already brakes at least
  `brake_share` of what is required (and at least `brake_min_mps2`); the driver steers away (lateral acceleration
  `steer_min_mps2` towards a side whose required demand it already covers); speed below the kind's
  `min_speed_mps`; the sensors degraded (the radar's newest sensor sample older than `sensor_max_age_s`, or the
  radar message itself stale);
* modes per alert: `off` (not evaluated), `shadow` (evaluated and logged exactly as it would have been said;
  `active` false), `on` (`active` true from stage `show_from`). Default `shadow` for every alert.

Every decision carries its reasons (`reason`, `suppressed`, the primary's measures), and every change of what
would be said is an event (`events`, the shadow log).
"""
import collections
import logging
import math

from ..native import round

log = logging.getLogger("chris_integrator.decide")
STAGES = ("none", "information", "caution", "warning", "imminent")
MODES = ("off", "shadow", "on")

ALERT_DEFAULTS = dict(
    mode="shadow",
    show_from=2,                                   # mode on: active from this stage (caution)
    kinds=("vehicle", "vru", "unknown"),
    entry_hold_s=(0.3, 0.2, 0.1, 0.0),             # per stage 1..4: the level must hold this long to enter
    exit_hold_s=(0.5, 0.6, 0.8, 1.0),              # ... and be below the stage this long to leave it
    min_hold_s=(0.5, 1.0, 1.5, 2.0),               # a stage is shown at least this long
    # Entry by evidence (0: by time alone, as before). A level is entered once the conflict's probability mass at it
    # (decide.objects[].mass), as far as it exceeds the level's minimum, has added up to `entry_hold_s` of a clear
    # conflict: per second (mass - minimum) / (entry_evidence - minimum), at most 1. A conflict of mass
    # `entry_evidence` or more enters after entry_hold_s as before; one just over the minimum (a roadside car whose
    # range, for a second, leaves a third of it in the lane) has to last in proportion. Below the level the sum runs
    # down again second for second.
    entry_evidence=0.6,
    entry_evidence_of="mass",                       # what adds up: the object's whole collision probability | "mass" at the level
    mass_min=None,                                  # {kind: (4 minima)}; None: threat.THRESHOLDS
    latch_imminent=True,
    switch_hold_s=0.3,
    exist_min=0.6, seen_max_s=2.5,
    # an object whose velocity is not observed yet (the radar's verdict unknown, its velocity's 1 sigma above
    # this, whichever futures are judged): its "it stands" is near the radar's prior, not evidence; it raises at
    # most `immature_max_stage` until p_stationary has moved to `immature_stand_p` (a lead appearing far ahead
    # falls off the prior within half a second, a standing car rises above it)
    immature_v_std_mps=2.5, immature_max_stage=1, immature_stand_p=0.65,
    # degraded futures: an object the radar has judged to stand beside the path (flag `parked`: its corridor
    # test with its own uncertainty and holds) raises at most this stage (with the prediction layer its lane
    # probabilities say it)
    parked_max_stage=1,
    # ... and an object that most probably stands rises above this stage only when the radar's corridor test puts
    # it surely in the path (flags `in_path` or `threat`): without the prediction layer's lanes that test is the
    # lane evidence there is
    corridor_max_stage=2,
    min_speed_mps={"vehicle": 3.0, "vru": 2.0, "unknown": 3.0},
    brake_share=0.9, brake_min_mps2=2.0,
    steer_min_mps2=1.5,
    sensor_max_age_s=1.0,
    source="threat",                               # "threat": the threat levels | "headway": time headway |
                                                   # "gap": what is hidden between parked vehicles (ctx["gap"])
    headway_s=(1.5, 0.9, 0.6),                     # source headway: information, caution, warning below these
    headway_exit_s=0.2,
    headway_min_mps=13.9,
)
ALERTS = {
    "obstacle_ahead": {},                           # the MFD's id: anything on a collision course ahead
    "headway": {"source": "headway", "kinds": ("vehicle",), "entry_hold_s": (2.0, 2.0, 1.0, 0.0),
                "exit_hold_s": (1.0, 1.0, 1.0, 1.0), "min_hold_s": (1.0, 2.0, 2.0, 2.0), "latch_imminent": False},
}
# What is hidden (decide cfg `gaps`; not an alert unless that is on): gaps between vehicles standing beside the car's
# way, from which someone could step out, at a speed above the one the car could still stop from (decide.c
# decide_gaps). At most caution: it says "slower here", not "brake". Its record is the frame's `gap`.
GAP_ALERT = {"source": "gap", "kinds": (), "entry_hold_s": (0.5, 0.5, 0.5, 0.5), "exit_hold_s": (1.0, 1.0, 1.0, 1.0),
             "min_hold_s": (1.0, 2.0, 2.0, 2.0), "latch_imminent": False, "min_speed_mps": 3.0}


def alert_cfg(name, cfg=None):
    c = dict(ALERT_DEFAULTS, **(GAP_ALERT if name == "hidden_gap" else ALERTS.get(name, {})))
    for k, v in (cfg or {}).items():
        c[k] = dict(c[k], **v) if isinstance(c.get(k), dict) and isinstance(v, dict) else v
    if c["mode"] not in MODES:
        c["mode"] = "shadow"
    return c


def _r(v, n=2):
    return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else round(float(v), n)


class Alert:
    """One alert's state machine."""

    def __init__(self, name, cfg=None):
        self.name = name
        self.cfg = alert_cfg(name, cfg)
        self.reset()

    def reset(self):
        self.stage = 0
        self.primary = None             # object id
        self.since = None               # when the stage was entered
        self.up_since = {}              # level -> since when the primary's entry level has been at least it
        self.ev = {}                    # level -> the evidence added up (s of a clear conflict; cfg entry_evidence)
        self._T = None                  # the last step's time
        self.down_since = None          # since when the primary's exit level has been below the stage
        self.challenger = None          # (id, since)
        self.latched = False
        self.suppressed = None
        self.said = (0, None, None)     # (stage, object, suppressed) last logged

    # ---- candidates ------------------------------------------------------------------------------------------
    def _levels(self, rec):
        """(entry level, exit level, rank key) of a judged object for this alert."""
        c = self.cfg
        if c["source"] == "headway":
            thw = rec.get("thw_s")
            if thw is None:
                return 0, 0, (0, 0.0, 0.0)
            hs = c["headway_s"]
            le = sum(1 for h in hs if thw < h)
            lx = sum(1 for h in hs if thw < h + c["headway_exit_s"])
            return le, lx, (le, -thw, 0.0)
        tm = rec.get("tm_s")
        mass = rec.get("mass") or [0.0] * 4
        le = rec.get("level", 0)
        return le, rec.get("level_exit", 0), (le, mass[le - 1] if le > 0 else (rec.get("p_coll") or 0.0),
                                               -(tm if tm is not None else 99.0))

    def _immature(self, obj):
        if obj is None or obj.get("motion") in ("stationary", "moving") or "parked" in (obj.get("flags") or ()):
            return False
        sd = obj.get("std") or ()
        ps = obj.get("p_stationary")
        if isinstance(ps, (int, float)) and ps >= self.cfg["immature_stand_p"]:
            return False
        return len(sd) >= 4 and isinstance(sd[2], (int, float)) and max(sd[2], sd[3]) > self.cfg["immature_v_std_mps"]

    def _eligible(self, rec, obj, T):
        c = self.cfg
        if rec.get("kind") not in c["kinds"]:
            return False
        if obj is None or obj.get("state") not in ("confirmed", "coast"):
            return False
        ex = obj.get("exist")
        if isinstance(ex, (int, float)) and ex < c["exist_min"]:
            return False
        seen = obj.get("seen")
        if isinstance(seen, dict) and seen:
            ages = [v for v in seen.values() if isinstance(v, (int, float))]
            if ages and min(ages) > c["seen_max_s"]:
                return False
        return True

    # ---- one frame --------------------------------------------------------------------------------------------
    def step(self, T, recs, objs, ctx):
        """T (s), recs: the judged objects' records (decide.objects[]: native decide_frame), objs: radar objects by id, ctx: {"v",
        "a_long", "a_lat", "sensor_age_s", "radar_age_s"}. Returns (output dict, [events])."""
        c = self.cfg
        if c["mode"] == "off":
            self.reset()
            return {"id": self.name, "mode": "off", "stage": 0, "stage_name": STAGES[0], "active": False}, []
        cand = {}
        if c["source"] == "gap":                    # one candidate: the frame's gap record (ctx["gap"])
            g = ctx.get("gap")
            recs = ()
            if g is not None and (g.get("level") or g.get("level_exit")):
                le, lx = min(2, g.get("level", 0)), min(2, g.get("level_exit", 0))
                cand["gap"] = (le, lx, (le, -(g.get("v_stop_mps") or 0.0), 0.0), g)
        for r in recs:
            ob = objs.get(r["id"])
            if self._eligible(r, ob, T):
                le, lx, key = self._levels(r)
                cap = 4
                if self._immature(ob):              # the radar's input itself is unripe, whoever made the futures
                    cap = c["immature_max_stage"]
                fl = ob.get("flags") or ()
                if r.get("fut") == "cv" and "parked" in fl and "in_path" not in fl:
                    cap = min(cap, c["parked_max_stage"])
                ps = ob.get("p_stationary")
                stands = ob.get("motion") == "stationary" or (ob.get("motion") != "moving" and isinstance(ps, (int, float)) and ps >= 0.5)
                if r.get("fut") == "cv" and stands and "in_path" not in fl and "threat" not in fl:
                    cap = min(cap, c["corridor_max_stage"])
                if cap < 4:
                    le, lx, key = min(le, cap), min(lx, cap), (min(key[0], cap),) + key[1:]
                if le > 0 or lx > 0:
                    cand[r["id"]] = (le, lx, key, r)
        # the primary: stable
        best = max(cand.items(), key=lambda kv: kv[1][2])[0] if cand else None
        if self.primary not in cand:
            self.primary = best
            self.challenger = None
        elif best is not None and best != self.primary and cand[best][2] > cand[self.primary][2]:
            if cand[best][0] > cand[self.primary][0] or (self.challenger and self.challenger[0] == best
                                                          and T - self.challenger[1] >= c["switch_hold_s"]):
                self.primary = best
                self.challenger = None
                self.up_since = {}
                self.ev = {}
            elif not self.challenger or self.challenger[0] != best:
                self.challenger = (best, T)
        else:
            self.challenger = None
        le, lx, _, rec = cand[self.primary] if self.primary in cand else (0, 0, None, None)
        # rise: per level, since when the entry level has reached it
        for L in range(1, 5):
            if le >= L:
                self.up_since.setdefault(L, T)
            else:
                self.up_since.pop(L, None)
        ref = c["entry_evidence"]
        if ref and c["source"] == "threat":
            dt = 0.0 if self._T is None else min(0.5, max(0.0, T - self._T))
            mass = (rec.get("mass") if rec is not None else None) or (0.0,) * 4
            if c["mass_min"] is None:
                from .threat import THRESHOLDS
                c["mass_min"] = {k: v["mass"] for k, v in THRESHOLDS.items()}
            thr = c["mass_min"].get((rec or {}).get("kind"), c["mass_min"]["unknown"])
            # only where the object's place is what is in doubt (decide.objects[].p_place: the prediction gave it two
            # places, P(in the car's way) below entry_evidence); a pedestrian about to step out, a car surely in
            # the lane enter by time as ever
            pp = (rec or {}).get("p_place")
            doubt = isinstance(pp, (int, float)) and pp < ref
            for L in range(1, 5):
                if le >= L:
                    # how sure the conflict is: its whole collision probability (the level is held by the mass at it;
                    # a car half in the lane kept P(collision) at 0.3-0.7 while the mass at the level sat at its minimum)
                    m = mass[L - 1] if isinstance(mass[L - 1], (int, float)) else 1.0
                    pc = (rec or {}).get("p_coll") if c["entry_evidence_of"] == "p_coll" else None
                    if isinstance(pc, (int, float)) and pc > m:
                        m = pc
                    rate = 1.0 if ref <= thr[L - 1] or not doubt else min(1.0, max(0.0, (m - thr[L - 1]) / (ref - thr[L - 1])))
                    self.ev[L] = self.ev.get(L, 0.0) + rate * dt
                else:
                    self.ev[L] = max(0.0, self.ev.get(L, 0.0) - dt)
        else:
            doubt = False
        self._T = T
        new = self.stage
        for L in range(4, self.stage, -1):
            if L in self.up_since and (self.ev.get(L, 0.0) if doubt else T - self.up_since[L]) >= c["entry_hold_s"][L - 1] - 1e-9:
                new = L
                break
        if new > self.stage:
            self._enter(new, T)
            self.down_since = None
        elif self.stage > 0:
            # fall: the exit level below the stage, held; the stage shown long enough; imminent latched
            if lx < self.stage:
                self.down_since = T if self.down_since is None else self.down_since
            else:
                self.down_since = None
            held = self.down_since is not None and T - self.down_since >= c["exit_hold_s"][self.stage - 1] - 1e-9
            shown = T - self.since >= c["min_hold_s"][self.stage - 1] - 1e-9
            if self.latched and (lx > 1 or not held) and not (ctx.get("v", 0.0) < 0.1):
                pass
            elif held and shown:
                self.latched = False
                self._enter(max(lx, 0), T)
                self.down_since = None
        # suppression
        sup = None
        if self.stage > 0:
            sup = self._suppress(rec, ctx)
        self.suppressed = sup
        out = {"id": self.name, "mode": c["mode"], "stage": self.stage, "stage_name": STAGES[self.stage],
               "active": c["mode"] == "on" and self.stage >= c["show_from"] and sup is None,
               "object": self.primary if self.stage > 0 else None,
               "since_s": _r(T - self.since) if self.stage > 0 else None, "suppressed": sup,
               "latched": self.latched}
        if self.stage > 0 and rec is not None and c["source"] == "gap":
            out["object"] = None
            out["reason"] = dict(rec, v_mps=_r(ctx.get("v", 0.0), 1))
            out["text"] = "%s GAP %s %.0f km/h" % (STAGES[self.stage].upper(), (rec.get("side") or "").upper(),
                                                   3.6 * (rec.get("v_stop_mps") or 0.0))
        elif self.stage > 0 and rec is not None:
            out["reason"] = self._reason(rec, le, lx)
            out["text"] = self._text(rec)
        ev = []
        key = (self.stage, out["object"], sup)
        if key != self.said:
            ev.append(self._event(T, out, self.said))
            self.said = key
        return out, ev

    def _enter(self, stage, T):
        self.stage = stage
        self.since = T
        if stage == 4 and self.cfg["latch_imminent"]:
            self.latched = True
        if stage == 0:
            self.primary = None if self.primary is None else self.primary

    def _suppress(self, rec, ctx):
        c = self.cfg
        v = ctx.get("v", 0.0)
        kind = (rec or {}).get("kind", "unknown")
        ms = c["min_speed_mps"]
        if v < (ms.get(kind, 3.0) if isinstance(ms, dict) else ms):
            return "low_speed"
        if c["source"] == "headway" and v < c["headway_min_mps"]:
            return "low_speed"
        sa, ra = ctx.get("sensor_age_s"), ctx.get("radar_age_s")
        if (sa is not None and sa > c["sensor_max_age_s"]) or (ra is not None and ra > c["sensor_max_age_s"]):
            return "sensor_degraded"
        if rec is None or c["source"] != "threat":
            return None
        req = rec.get("a_req") or {}
        ab = req.get("brake")
        a_long = ctx.get("a_long")
        if ab is not None and a_long is not None and -a_long >= max(c["brake_min_mps2"], c["brake_share"] * ab):
            return "driver_braking"
        a_lat = ctx.get("a_lat")
        if a_lat is not None and abs(a_lat) >= c["steer_min_mps2"]:
            side = req.get("left") if a_lat > 0 else req.get("right")
            if side is not None and abs(a_lat) >= side:
                return "driver_steering_away"
        return None

    @staticmethod
    def _reason(rec, le, lx):
        why = rec.get("why") or {}
        return {"level": le, "level_exit": lx, "kind": rec.get("kind"), "measure": why.get("measure"),
                "mode": why.get("mode"), "ego_mode": why.get("ego_mode"), "p0": why.get("p0"),
                "p_coll": rec.get("p_coll"), "p_coll_cond": rec.get("p_coll_cond"), "tn": rec.get("tn"),
                "btn": rec.get("btn"), "stn": rec.get("stn"), "tm_s": rec.get("tm_s"), "ttb_s": rec.get("ttb_s"),
                "tts_s": rec.get("tts_s"), "ttc_s": rec.get("ttc_s"), "thw_s": rec.get("thw_s"), "fut": rec.get("fut")}

    def _text(self, rec):
        what = {"vru": "PERSON", "vehicle": "VEHICLE"}.get(rec.get("kind"), "OBJECT")
        if self.cfg["source"] == "headway":
            return "HEADWAY %.1fs" % (rec.get("thw_s") or 0.0)
        ttc = rec.get("ttc_s")
        return "%s %s%s" % (STAGES[self.stage].upper(), what, "" if ttc is None else " %.1fs" % ttc)

    def _event(self, T, out, before):
        st0 = before[0]
        st = out["stage"]
        if st == 0:
            kind = "clear"
        elif st0 == 0:
            kind = "raise"
        elif st > st0:
            kind = "escalate"
        elif st < st0:
            kind = "lower"
        elif out["object"] != before[1]:
            kind = "switch"
        else:
            kind = "suppress" if out["suppressed"] else "unsuppress"
        e = {"t": _r(T, 3), "alert": self.name, "mode": out["mode"], "event": kind, "stage": st,
             "stage_name": out["stage_name"], "object": out["object"], "suppressed": out["suppressed"],
             "would_show": out["mode"] != "off" and st >= self.cfg["show_from"] and out["suppressed"] is None,
             "active": out["active"]}
        if "reason" in out:
            e["reason"] = out["reason"]
            e["text"] = out.get("text")
        return e


class AlertSet:
    """Every configured alert, the shadow log (a ring buffer of the events, and the log)."""

    def __init__(self, cfg=None, log_max=500):
        cfg = dict(cfg or {})
        names = list(ALERTS) + [k for k in cfg if k not in ALERTS]              # ("hidden_gap": the decide plug-in, cfg `gaps`)
        self.alerts = {n: Alert(n, cfg.get(n)) for n in names}
        self.events = collections.deque(maxlen=log_max)
        self.counts = {n: {"raise": 0, "would_show": 0} for n in names}

    def reset(self):
        for a in self.alerts.values():
            a.reset()

    def set_mode(self, name, mode):
        if name in self.alerts and mode in MODES:
            self.alerts[name].cfg["mode"] = mode
            return True
        return False

    def step(self, T, recs, objs, ctx):
        outs, evs = [], []
        for a in self.alerts.values():
            o, ev = a.step(T, recs, objs, ctx)
            outs.append(o)
            for e in ev:
                if e["event"] == "raise":
                    self.counts[a.name]["raise"] += 1
                if e["would_show"]:
                    self.counts[a.name]["would_show"] += 1
                self.events.append(e)
                if e["mode"] != "off":
                    log.info("alert %s (%s): %s stage %s object %s%s%s", e["alert"], e["mode"], e["event"],
                             e["stage_name"], e["object"], " suppressed: %s" % e["suppressed"] if e["suppressed"] else "",
                             " [%s]" % e["text"] if e.get("text") else "")
            evs += ev
        return outs, evs
