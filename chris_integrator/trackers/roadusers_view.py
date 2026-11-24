"""Display objects for the road users the object layer cannot place (lane L42; display channel only).

On a car without range sensors the object layer holds one object: the lead (the road model's vehicle ahead).
Pedestrians, cyclists and the vehicles in other lanes are seen only by the camera's class detector
(core/roadusers.py: a box, a class, a bearing, and a ground position where the box's bottom edge is in the
detector's view). This builder turns those into `view.objects` a screen can draw:

* only boxes with a ground position are placed (the box's bottom inside the detector's region and below the
  horizon: from about 14 m outwards in the wide view of the phone's unit, 24 m in the second view); a box cut at
  the bottom has a bearing only and is not drawn (its distance would be a guess);
* a riderless bike or motorbike is not drawn (parked ones line every street);
* the detection the lead points at is not drawn again (the lead is drawn from the `lead` topic);
* identity by nearest neighbour inside 3 sigma (position) among compatible classes (any vehicle with any
  vehicle, a person with a cyclist); an alpha-beta filter per object for position and relative velocity, the
  drawn position predicted to the frame's time;
* nothing pops: an object appears after `confirm_hits` placed detections within `confirm_s` and fades in, it
  is kept `keep_s` after its last detection (marked `coast`), then fades out; its class is a vote over its life
  (core/classvote.py), so its shape does not hop from frame to frame.

Output objects: {"id": "ru:<n>", "kind": "road_user", "role": "other", "x", "y" (car frame of `view`: x forward,
y RIGHT), "vx", "vy", "sx", "sy" (1 sigma), "age_s", "meas_age_s", "phase", "alpha", "attrs": {"source",
"approximate": true}, "cls", "box"}. Their positions are the camera's monocular estimate: about 15 % (1 sigma)
of the distance at 20 m, 30 % at 40 m, plus the calibration's share; they are informational, never a
measurement topic.
"""
import math
import operator

from ..native import round                  # builtins.round (its results; the native core)
from ..core import classvote

NS = 1e9
VEHICLES = {"car", "van", "truck", "bus", "tram"}
VRU = {"human", "bike", "motorbike"}

DEFAULT_CFG = dict(
    confirm_hits=2, confirm_s=1.5,
    keep_s=1.2, fade_s=0.3, coast_after_s=0.45,
    gate_sigma=3.0, gate_floor_m=1.5,
    alpha=0.55, beta=0.15,       # alpha-beta filter on x, y
    max_range_m=60.0, max_lateral_m=12.0, max_objects=12,
    max_speed_mps=40.0,          # relative speed clamp of the filter
    lead_halfwidth_m=1.2,
)


def _compatible(a, b):
    if a is None or b is None:
        return True
    return (a in VEHICLES) == (b in VEHICLES)


class _Obj:
    __slots__ = ("id", "x", "y", "vx", "vy", "sx", "sy", "t", "t0", "hits", "first", "vote", "conf_t", "box", "fade0")

    def __init__(self, oid, x, y, sx, sy, t, box):
        self.id, self.x, self.y, self.vx, self.vy, self.sx, self.sy = oid, x, y, 0.0, 0.0, sx, sy
        self.t = self.t0 = self.first = t
        self.hits = 1
        self.vote = {}
        self.conf_t = None           # time the object was confirmed (appears)
        self.box = box
        self.fade0 = None            # time the fade-out started


class RoadUserDisplay:
    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.reset()

    def reset(self):
        self.objs = []
        self.next_id = 1
        self.last_t = None

    def update(self, msg, lead=None):
        """A `road_users` message in. lead: the view's lead object (x, y right) or None."""
        c = self.cfg
        ts = msg.get("ts_mono_ns")
        if not isinstance(ts, int):
            return
        t = ts / NS
        if self.last_t is not None and t < self.last_t - 2.0:
            self.reset()                 # a new time axis
        self.last_t = t if self.last_t is None else max(self.last_t, t)
        dets = []
        for o in msg.get("objects") or ():
            if o.get("x_m") is None or o.get("range_m") is None:
                continue
            label = classvote.norm_label(o.get("class"))
            if label in ("bike", "motorbike") and not o.get("rider") and o.get("det_class") != "rider":
                continue
            x, y = float(o["x_m"]), -float(o["y_m"])          # view frame: y right
            if not (0.5 < x <= c["max_range_m"] and abs(y) <= c["max_lateral_m"]):
                continue
            if lead is not None and self._is_lead(o, lead):
                continue
            rng, sr = float(o["range_m"]), float(o.get("range_std_m") or 0.2 * float(o["range_m"]))
            sb = math.radians(float(o.get("bearing_std_deg") or 1.0))
            # range along the line of sight, bearing across it, approximated on the axes
            sx = math.hypot(sr * x / rng, sb * rng * abs(y) / rng)
            sy = math.hypot(sr * abs(y) / rng, sb * rng * x / rng)
            dets.append((x, y, sx, sy, label, o.get("class_conf"), o.get("weight", 1.0), o.get("box")))
        # association: nearest compatible object within the gate, greedy by normalised distance
        pairs = []
        for j, (x, y, sx, sy, label, _, _, _) in enumerate(dets):
            for ob in self.objs:
                if ob.fade0 is not None:
                    continue
                dt = max(0.0, t - ob.t)
                px, py = ob.x + ob.vx * dt, ob.y + ob.vy * dt
                gx = max(c["gate_floor_m"], c["gate_sigma"] * math.hypot(sx, ob.sx))
                gy = max(c["gate_floor_m"], c["gate_sigma"] * math.hypot(sy, ob.sy))
                d = math.hypot((x - px) / gx, (y - py) / gy)
                if d <= 1.0 and _compatible(label, (classvote.result(ob.vote) or {}).get("label")):
                    pairs.append((d, j, ob))
        pairs.sort(key=lambda p: p[0])
        used_d, used_o = set(), set()
        for d, j, ob in pairs:
            if j in used_d or ob.id in used_o:
                continue
            used_d.add(j)
            used_o.add(ob.id)
            self._correct(ob, dets[j], t)
        for j, det in enumerate(dets):
            if j in used_d or len(self.objs) >= 4 * c["max_objects"]:
                continue
            x, y, sx, sy, label, conf, w, box = det
            ob = _Obj(self.next_id, x, y, sx, sy, t, box)
            self.next_id += 1
            classvote.vote(ob.vote, label, conf, "roadusers", t, w)
            self.objs.append(ob)

    def _is_lead(self, o, lead):
        lx, ly = float(lead["x"]), float(lead["y"])          # y right
        if lx <= 0.5:
            return False
        lb = -math.atan2(ly, lx)                              # left positive, as the detector's bearing
        e = abs(math.radians(float(o["bearing_deg"])) - lb)
        gate = max(math.radians(1.5), 3.0 * math.radians(float(o.get("bearing_std_deg") or 1.0)),
                   math.atan2(self.cfg["lead_halfwidth_m"], math.hypot(lx, ly)))
        if e > gate:
            return False
        if o.get("range_m") is None:
            return True
        sr = float(o.get("range_std_m") or 0.2 * float(o["range_m"]))
        return abs(float(o["range_m"]) - math.hypot(lx, ly)) <= 3.0 * max(sr, 1.0)

    def _correct(self, ob, det, t):
        c = self.cfg
        x, y, sx, sy, label, conf, w, box = det
        dt = max(1e-3, t - ob.t)
        px, py = ob.x + ob.vx * dt, ob.y + ob.vy * dt
        rx, ry = x - px, y - py
        a, b = c["alpha"], c["beta"]
        ob.x, ob.y = px + a * rx, py + a * ry
        vm = c["max_speed_mps"]
        ob.vx = max(-vm, min(vm, ob.vx + b * rx / dt))
        ob.vy = max(-vm, min(vm, ob.vy + b * ry / dt))
        ob.sx, ob.sy = sx * math.sqrt(a), sy * math.sqrt(a)
        ob.t = t
        ob.hits += 1
        ob.box = box
        classvote.vote(ob.vote, label, conf, "roadusers", t, w)
        if ob.conf_t is None and ob.hits >= c["confirm_hits"] and t - ob.first <= c["confirm_s"] * (ob.hits - 1):
            ob.conf_t = t

    def objects(self, ts_ns):
        """The display objects at a frame time (ns)."""
        c = self.cfg
        T = ts_ns / NS
        out, keep = [], []
        for ob in self.objs:
            age = T - ob.t
            if ob.conf_t is None:
                if T - ob.first > c["confirm_s"] * max(1, c["confirm_hits"]):
                    continue                                  # never confirmed: dropped silently
                keep.append(ob)
                continue
            if ob.fade0 is None and age > c["keep_s"]:
                ob.fade0 = T
            if ob.fade0 is not None:
                f = 1.0 - (T - ob.fade0) / c["fade_s"]
                if f <= 0.0:
                    continue                                  # gone
                phase, alpha = "leave", f
            else:
                fin = (T - ob.conf_t) / c["fade_s"]
                phase = "appear" if fin < 1.0 else ("coast" if age > c["coast_after_s"] else "steady")
                alpha = min(1.0, max(0.0, fin))
            keep.append(ob)
            dt = max(0.0, min(age, c["keep_s"]))
            x, y = ob.x + ob.vx * dt, ob.y + ob.vy * dt
            if not 0.5 < x <= c["max_range_m"] + 10.0:
                continue
            cls = classvote.result(ob.vote, T)
            out.append({"id": "ru:%d" % ob.id, "kind": "road_user", "role": "other",
                        "x": round(x, 2), "y": round(y, 2), "vx": round(ob.vx, 2), "vy": round(ob.vy, 2),
                        "sx": round(ob.sx + 0.5 * abs(ob.vx) * dt, 2), "sy": round(ob.sy + 0.5 * abs(ob.vy) * dt, 2),
                        "age_s": round(T - ob.t0, 2), "meas_age_s": round(max(0.0, age), 2), "phase": phase,
                        "alpha": round(alpha, 2), "attrs": {"source": "roadusers", "approximate": True},
                        "cls": None if cls is None else {"label": cls["label"], "confidence": cls["confidence"],
                                                         "source": cls["source"]},
                        "box": ob.box})
        self.objs = keep
        out.sort(key=lambda o: o["x"])
        return out[:c["max_objects"]]


RADAR_CFG = dict(
    fade_s=0.3,                  # appear / leave
    max_range_m=70.0, max_lateral_m=12.0, max_objects=12,
    stale_s=1.0,                 # a radar message older than this draws nothing (the old builder takes over)
    lead_halfwidth_m=1.2, lead_range_rel=0.25, lead_range_m=4.0,
    alert_states=("confirmed", "coast"),
    # `view.around`: every object all round the car for a plan display (radar scope), the lead included
    around=True,
    around_range_m=85.0,         # in every direction (the MFD radar view's largest range is 80 m)
    around_max=20,
    around_states=("confirmed", "coast", "memory"),
    # the threat flags (`attrs.threat`, `attrs.obstacle`, `fl` t / o) and `alert` from the decision layer's `decide`
    # while it is fresh (its per-object level as shown, with hysteresis, and its `obstacle_ahead` alert); "radar":
    # the radar's own flags as before
    flags_from="decide", decide_stale_s=0.5,
)

# a class label is passed in `around` only from this confidence on (the MFD's own rule for `objects`; a screen
# draws the generic vehicle below it)
AROUND_MIN_CONF = 0.5
AROUND_MIN_CONF_VRU = 0.35
_VRU_LABELS = ("human", "bike", "motorbike")
_MOTION = {"stationary": "s", "moving": "m"}     # `view.around` "mv"
_K0, _K1 = operator.itemgetter(0), operator.itemgetter(1)


def _r1(v):
    v = round(float(v), 1)
    return 0.0 if v == 0 else v                 # no "-0.0"


class RadarDisplay:
    """Display objects from the radar's objects (chris_integrator/radar): the same output as `RoadUserDisplay`,
    with what the radar knows on top: `attrs.motion`, `attrs.parked` (a vehicle that stands beside the path: a
    screen draws it dimmed), `attrs.obstacle` (it stands ON the path), `attrs.threat` (the car reaches it within
    seconds at the present speeds), `attrs.ttc_s`, `attrs.with_traffic`.

    `alert(...)`: the one thing a screen must shout about, if any: {"level": "threat" | "obstacle", "id", "x",
    "y" (right), "ttc_s", "lead": bool (it is the object the lead is drawn from)}. It is taken from all radar
    objects, also the one not drawn again because the lead shows it.

    Nothing pops: an object fades in when the radar first lists it and fades out where it was last (carried on by
    its velocity) when the radar drops it. A riderless two-wheeler that stands beside the path is not drawn
    (parked ones line every street)."""

    def __init__(self, cfg=None):
        self.cfg = dict(RADAR_CFG, **(cfg or {}))
        self.reset()

    def reset(self):
        self.msg = None
        self.seen = {}               # radar id -> {"t0", "t", "o"}: first and last time listed, last object
        self.ar = {}                 # the same for `around` (remembered objects too)
        self.dec = None              # the decision layer's newest `decide` (on_decide)
        self._by_id = None

    def on_decide(self, msg):
        """The decision layer's `decide` message (or None): while it is fresh against the radar's, the threat flags
        and the alert are its (`_flags`, `_alert`)."""
        self.dec = msg if isinstance(msg, dict) else None

    def _dec(self):
        d, m = self.dec, self.msg
        if self.cfg["flags_from"] != "decide" or d is None or m is None or not isinstance(d.get("ts_mono_ns"), int):
            return None
        if abs(d["ts_mono_ns"] - m["ts_mono_ns"]) / NS > self.cfg["decide_stale_s"]:
            return None
        return d

    def _flags(self, o):
        """(threat, obstacle, ttc_s) of a radar object. From `decide`: threat = its level as shown is warning or
        above; obstacle = caution, or information on an object the radar says stands on the path (`in_path`);
        ttc_s the decision layer's time to collision. Without a fresh `decide`: the radar's flags."""
        d = self._dec()
        return self._flags_of(o, d, None if d is None else self._dec_by_id(d))

    @staticmethod
    def _flags_of(o, d, by_id):
        """`_flags` with the frame's `_dec()` and its objects by id (`_dec_by_id`) found once by the caller."""
        fl = o.get("flags") or ()
        if d is None:
            return "threat" in fl, "in_path" in fl, o.get("ttc_s")
        r = by_id.get(o.get("id"))
        if r is None:
            return False, False, o.get("ttc_s")
        sh = r.get("shown", r.get("level", 0)) or 0
        return sh >= 3, sh == 2 or (sh == 1 and "in_path" in fl), r.get("ttc_s") if r.get("ttc_s") is not None else o.get("ttc_s")

    def _dec_by_id(self, d):
        if self._by_id is None or self._by_id[0] is not d:
            self._by_id = (d, {r.get("id"): r for r in d.get("objects") or ()})
        return self._by_id[1]

    def update(self, msg):
        ts = msg.get("ts_mono_ns")
        if not isinstance(ts, int):
            return
        t = ts / NS
        if self.msg is not None and t < self.msg["ts_mono_ns"] / NS - 2.0:
            self.reset()             # a new time axis
        self.msg = msg
        # one pass over the objects for both tables (each table sees the same entries in the same order as in a
        # pass of its own)
        around = self.cfg["around"]
        states = self.cfg["around_states"]
        ar, seen = self.ar, self.seen
        listed_ar, listed = set(), set()
        for o in msg.get("objects") or ():
            st = o.get("state")
            if around and st in states:
                s = ar.get(o["id"])
                if s is None:
                    s = ar[o["id"]] = {"t0": t}
                s["t"], s["o"] = t, o
                s.pop("lost", None)
                listed_ar.add(o["id"])
            if st in ("confirmed", "coast"):
                s = seen.get(o["id"])
                if s is None:
                    s = seen[o["id"]] = {"t0": t}
                s["t"], s["o"] = t, o
                s.pop("lost", None)
                listed.add(o["id"])
        if around:
            for oid, s in ar.items():
                if oid not in listed_ar:
                    s.setdefault("lost", t)
        for oid, s in seen.items():                      # not in this message any more: it starts to leave
            if oid not in listed:
                s.setdefault("lost", t)

    def fresh(self, ts_ns):
        return self.msg is not None and abs(ts_ns - self.msg["ts_mono_ns"]) / NS <= self.cfg["stale_s"]

    def _same_as_lead(self, x, y, lead, sx=0.0):
        """The radar's place of an object against the lead's: within the lead's width, and within what the
        radar itself says it knows of the distance (a camera-only object: a quarter of it, more when new)."""
        if lead is None:
            return False
        c = self.cfg
        lx, ly = float(lead["x"]), float(lead["y"])
        return abs(y - ly) <= c["lead_halfwidth_m"] and abs(x - lx) <= max(c["lead_range_m"], c["lead_range_rel"] * lx, 2.0 * sx)

    def objects(self, ts_ns, lead=None):
        """(display objects, alert or None) at a frame time (ns). lead: the view's lead object (x, y right)."""
        c = self.cfg
        T = ts_ns / NS
        out, alert = [], None
        v_ego = float(((self.msg or {}).get("ego") or {}).get("speed_mps") or 0.0)
        # with the decision layer: the one alert is its `obstacle_ahead` while that would be said (caution or above,
        # not suppressed, mode shadow or on: a consumer applies its own mode on top)
        dec = self._dec()
        dec_al = None
        if dec is not None:
            dec_al = next((a for a in dec.get("alerts") or () if a.get("id") == "obstacle_ahead"), None)
            if dec_al is None or dec_al.get("mode") == "off" or (dec_al.get("stage") or 0) < 2 or dec_al.get("suppressed"):
                dec_al = None
        by_id = None if dec is None else self._dec_by_id(dec)
        flags_of = self._flags_of
        place = {}
        for oid in list(self.seen):
            s = self.seen[oid]
            o = s["o"]
            gone = T - s["lost"] if "lost" in s else None        # since the radar dropped it
            if (gone is not None and gone > c["fade_s"]) or T - s["t"] > c["stale_s"]:
                del self.seen[oid]
                continue
            dt = max(0.0, min(T - s["t"], 1.0))
            place[oid] = (float(o["x_m"]) + (float(o["vx_mps"]) - v_ego) * dt,      # the car frame of now: relative motion
                          -(float(o["y_m"]) + float(o["vy_mps"]) * dt), gone)        # view frame: y right
        # the one radar object the lead is drawn from: the nearest to it in range among those in its place
        near = []
        if lead is not None:                             # (_same_as_lead written out, the lead's place read once)
            lx, ly = float(lead["x"]), float(lead["y"])
            hw, rm, rr = c["lead_halfwidth_m"], c["lead_range_m"], c["lead_range_rel"]
            seen = self.seen
            for oid, (x, y, _) in place.items():
                sx = float((seen[oid]["o"].get("std") or [0.0])[0])
                if abs(y - ly) <= hw and abs(x - lx) <= max(rm, rr * lx, 2.0 * sx):
                    near.append((abs(x - lx), oid))
        lead_ids = set()
        if near:
            # ... and what the radar holds twice at that place (a second object on the same vehicle, until the
            # radar has merged them): not a second vehicle to draw
            first = min(near)[1]
            lead_ids.add(first)
            for _, oid in near:
                sx = float((self.seen[oid]["o"].get("std") or [0.0])[0])
                if abs(place[oid][0] - place[first][0]) <= max(3.0, 2.0 * sx):
                    lead_ids.add(oid)
        for oid, (x, y, gone) in place.items():
            s = self.seen[oid]
            o = s["o"]
            fl = o.get("flags") or ()
            is_lead = oid in lead_ids
            th, ob, ttc = flags_of(o, dec, by_id)
            if dec is None and gone is None and o.get("state") in c["alert_states"] and (th or ob):
                level = "threat" if th else "obstacle"
                if alert is None or (level == "threat") > (alert["level"] == "threat") or \
                        (level == alert["level"] and x < alert["x"]):
                    alert = {"level": level, "id": "ru:%d" % oid, "x": round(x, 2), "y": round(y, 2),
                             "ttc_s": ttc, "lead": is_lead}
            if dec_al is not None and dec_al.get("object") == oid and gone is None:
                st = dec_al["stage"]
                rs = dec_al.get("reason") or {}
                alert = {"level": "threat" if st >= 3 else "obstacle", "id": "ru:%d" % oid, "x": round(x, 2),
                         "y": round(y, 2), "ttc_s": ttc, "lead": is_lead, "stage": st, "stage_name": dec_al.get("stage_name"),
                         "mode": dec_al.get("mode"), "source": "decide", "tn": rs.get("tn"), "p_coll": rs.get("p_coll"),
                         "tm_s": rs.get("tm_s")}
            if is_lead or not (0.5 < x <= c["max_range_m"] and abs(y) <= c["max_lateral_m"]):
                s.pop("d0", None)                        # not drawn now: it fades in again when it is
                continue
            cls = o.get("class") or None
            label = cls.get("label") if isinstance(cls, dict) else None
            parked = "parked" in fl
            if label in ("bike", "motorbike") and o.get("motion") != "moving" and "on_path" not in fl:
                continue
            if gone is not None:
                phase, alpha = "leave", max(0.0, 1.0 - max(gone, 0.0) / c["fade_s"])
            else:
                fin = (T - s.setdefault("d0", T)) / c["fade_s"]      # from the frame it is first DRAWN in
                phase = "appear" if fin < 1.0 else ("coast" if o.get("state") == "coast" else "steady")
                alpha = min(1.0, max(0.0, fin))
            sd = o.get("std") or [0.0, 0.0, 0.0, 0.0]
            out.append({"id": "ru:%d" % oid, "kind": "road_user", "role": "other",
                        "x": round(x, 2), "y": round(y, 2),
                        "vx": round(float(o.get("v_rel_mps") or 0.0), 2), "vy": round(-float(o["vy_mps"]), 2),
                        "sx": sd[0], "sy": sd[1], "age_s": o.get("age_s"),
                        "meas_age_s": round(min((o.get("seen") or {"": 0.0}).values()) + max(T - s["t"], 0.0), 2),
                        "phase": phase, "alpha": round(alpha, 2),
                        "attrs": {"source": "radar", "approximate": True, "motion": o.get("motion"), "parked": parked,
                                  "obstacle": ob, "threat": th,
                                  "with_traffic": "with_traffic" in fl, "ttc_s": ttc},
                        "cls": None if not isinstance(cls, dict) else {"label": cls.get("label"), "confidence": cls.get("confidence"),
                                                                       "source": cls.get("source")},
                        "box": None})
        out.sort(key=lambda q: q["x"])
        return out[:c["max_objects"]], alert

    @staticmethod
    def _label(cls):
        if not isinstance(cls, dict) or not isinstance(cls.get("label"), str):
            return None
        conf = cls.get("confidence")
        need = AROUND_MIN_CONF_VRU if cls["label"] in _VRU_LABELS else AROUND_MIN_CONF
        if isinstance(conf, (int, float)) and conf < need:
            return None
        return cls["label"]

    def around(self, ts_ns, lead=None):
        """`view.around` at a frame time (ns): every object the radar lists all round the car, out to
        `around_range_m` in every direction, remembered ones (state `memory`: out of every sensor's field, carried
        on by their motion) included and marked, and the lead in place of the radar's own objects on it. Compact
        (it goes to a small screen at 10 Hz): see docs/TOPICS.md, `view.around`. lead: the view's lead object."""
        c = self.cfg
        T = ts_ns / NS
        v_ego = float(((self.msg or {}).get("ego") or {}).get("speed_mps") or 0.0)
        R = c["around_range_m"]
        fade_s = c["fade_s"]
        dec = self._dec()                                # (the same for every object of the frame: _flags)
        by_id = None if dec is None else self._dec_by_id(dec)
        flags_of = self._flags_of
        place = {}
        for oid in list(self.ar):
            s = self.ar[oid]
            o = s["o"]
            gone = T - s["lost"] if "lost" in s else None
            if (gone is not None and gone > c["fade_s"]) or T - s["t"] > c["stale_s"]:
                del self.ar[oid]
                continue
            dt = max(0.0, min(T - s["t"], 1.0))
            place[oid] = (float(o["x_m"]) + (float(o["vx_mps"]) - v_ego) * dt,
                          -(float(o["y_m"]) + float(o["vy_mps"]) * dt), gone)
        # the radar's objects on the lead (the nearest in range among those in its place, and what it holds twice
        # there): the lead stands for them
        lead_ids, lead_obj = set(), None
        if lead is not None:
            near = []                                    # (_same_as_lead written out, the lead's place read once)
            lx, ly = float(lead["x"]), float(lead["y"])
            hw, rm, rr = c["lead_halfwidth_m"], c["lead_range_m"], c["lead_range_rel"]
            ar = self.ar
            for oid, (x, y, gone) in place.items():
                if gone is None and ar[oid]["o"].get("state") != "memory":
                    sx = float((ar[oid]["o"].get("std") or [0.0])[0])
                    if abs(y - ly) <= hw and abs(x - lx) <= max(rm, rr * lx, 2.0 * sx):
                        near.append((abs(x - lx), oid))
            if near:
                first = min(near)[1]
                lead_obj = self.ar[first]["o"]
                for _, oid in near:
                    sx = float((self.ar[oid]["o"].get("std") or [0.0])[0])
                    if abs(place[oid][0] - place[first][0]) <= max(3.0, 2.0 * sx):
                        lead_ids.add(oid)
        rows = []
        for oid, (x, y, gone) in place.items():
            if oid in lead_ids or math.hypot(x, y) > R:
                continue
            s = self.ar[oid]
            o = s["o"]
            fl = o.get("flags") or ()
            st = o.get("state")
            label = self._label(o.get("class"))
            raw = (o.get("class") or {}).get("label") if isinstance(o.get("class"), dict) else None
            if raw in ("bike", "motorbike") and o.get("motion") != "moving" and "on_path" not in fl:
                continue                                 # a riderless two-wheeler beside the road
            # (_r1 written out for the floats here: round to 0.1, no "-0.0")
            rx, ry = round(x, 1), round(y, 1)
            e = {"id": "ru:%d" % oid, "x": 0.0 if rx == 0 else rx, "y": 0.0 if ry == 0 else ry}
            gx, gy = float(o.get("vx_mps") or 0.0), -float(o.get("vy_mps") or 0.0)
            if math.hypot(gx, gy) >= 0.5:
                rx, ry = round(gx, 1), round(gy, 1)
                e["gx"], e["gy"] = 0.0 if rx == 0 else rx, 0.0 if ry == 0 else ry
            sd = o.get("std") or (0.0, 0.0)
            rx = round(max(float(sd[0]), float(sd[1])), 1)
            e["sd"] = 0.0 if rx == 0 else rx
            if st != "confirmed":
                e["st"] = st
                e["ago"] = _r1(min((o.get("seen") or {"": 0.0}).values()) + max(T - s["t"], 0.0))
            if label:
                e["k"] = label
            mv = _MOTION.get(o.get("motion"))
            if mv:
                e["mv"] = mv
            # a remembered object is never an obstacle or a threat: nothing sees it
            th, ob, ttc = flags_of(o, dec, by_id)
            mem = st == "memory"
            to = (th and not mem, ob and not mem)
            f = ("t" if to[0] else "") + ("o" if to[1] else "") + ("p" if "parked" in fl else "") + \
                ("w" if "with_traffic" in fl else "")
            if f:
                e["fl"] = f
            if (to[0] or to[1]) and isinstance(ttc, (int, float)):
                e["ttc"] = _r1(ttc)
            if gone is not None:
                a = max(0.0, 1.0 - max(gone, 0.0) / fade_s)
            else:
                a = min(1.0, max(0.0, (T - s.setdefault("d0", T)) / fade_s))
            if a < 1.0:
                e["a"] = round(a, 2)
            rows.append(e)
        if lead is not None:
            at = lead.get("attrs") or {}
            e = {"id": lead.get("id") or "lead", "x": _r1(lead["x"]), "y": _r1(lead["y"])}
            gx = float(lead.get("vx") or 0.0) + v_ego
            if abs(gx) >= 0.5:
                e["gx"], e["gy"] = _r1(gx), 0.0
            sd = at.get("dist_std_m") if isinstance(at.get("dist_std_m"), (int, float)) else lead.get("sx")
            if isinstance(sd, (int, float)):
                e["sd"] = _r1(sd)
            if lead.get("phase") == "coast":
                e["st"] = "coast"
                e["ago"] = _r1(lead.get("meas_age_s") or 0.0)
            label = self._label(lead.get("cls"))
            if label:
                e["k"] = label
            lfl = (lead_obj or {}).get("flags") or ()
            if lead_obj is not None:
                lth, lob, _ = self._flags(lead_obj)
                lfl = tuple(k for k in lfl if k not in ("threat", "in_path")) + (("threat",) if lth else ()) + \
                    (("in_path",) if lob else ())
            mv = _MOTION.get((lead_obj or {}).get("motion"))
            if mv is None and isinstance(at.get("standing"), bool):
                mv = "s" if at["standing"] else "m"
            if mv:
                e["mv"] = mv
            e["fl"] = "L" + "".join(ch for ch, key in (("t", "threat"), ("o", "in_path"), ("w", "with_traffic"))
                                    if key in lfl)
            ttc = at.get("ttc_s") if isinstance(at.get("ttc_s"), (int, float)) else (lead_obj or {}).get("ttc_s")
            if ("t" in e["fl"] or "o" in e["fl"]) and isinstance(ttc, (int, float)):
                e["ttc"] = _r1(ttc)
            a = lead.get("alpha")
            if isinstance(a, (int, float)) and a < 1.0:
                e["a"] = round(a, 2)
            rows.append(e)
        # flagged first, then by distance; the first around_max of those, by distance. (The keys are made once per
        # row: the same tuples and distances the two sorts compare, the sorts stable as before.)
        keyed = []
        for e in rows:
            fs = e.get("fl", "")
            d = math.hypot(e["x"], e["y"])
            keyed.append(((0 if ("t" in fs or "o" in fs or "L" in fs) else 1, d), d, e))
        keyed.sort(key=_K0)
        keyed = keyed[:c["around_max"]]
        keyed.sort(key=_K1)
        return [k[2] for k in keyed]
