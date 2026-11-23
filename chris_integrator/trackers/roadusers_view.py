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

from ..native import round                  # builtins.round, native where built
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
)


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

    def update(self, msg):
        ts = msg.get("ts_mono_ns")
        if not isinstance(ts, int):
            return
        t = ts / NS
        if self.msg is not None and t < self.msg["ts_mono_ns"] / NS - 2.0:
            self.reset()             # a new time axis
        self.msg = msg
        listed = set()
        for o in msg.get("objects") or ():
            if o.get("state") not in ("confirmed", "coast"):
                continue
            s = self.seen.get(o["id"])
            if s is None:
                s = self.seen[o["id"]] = {"t0": t}
            s["t"], s["o"] = t, o
            s.pop("lost", None)
            listed.add(o["id"])
        for oid, s in self.seen.items():                 # not in this message any more: it starts to leave
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
        near = [(abs(x - float(lead["x"])), oid) for oid, (x, y, _) in place.items()
                if self._same_as_lead(x, y, lead, float((self.seen[oid]["o"].get("std") or [0.0])[0]))]
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
            if gone is None and o.get("state") in c["alert_states"] and ("threat" in fl or "in_path" in fl):
                level = "threat" if "threat" in fl else "obstacle"
                if alert is None or (level == "threat") > (alert["level"] == "threat") or \
                        (level == alert["level"] and x < alert["x"]):
                    alert = {"level": level, "id": "ru:%d" % oid, "x": round(x, 2), "y": round(y, 2),
                             "ttc_s": o.get("ttc_s"), "lead": is_lead}
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
                                  "obstacle": "in_path" in fl, "threat": "threat" in fl,
                                  "with_traffic": "with_traffic" in fl, "ttc_s": o.get("ttc_s")},
                        "cls": None if not isinstance(cls, dict) else {"label": cls.get("label"), "confidence": cls.get("confidence"),
                                                                       "source": cls.get("source")},
                        "box": None})
        out.sort(key=lambda q: q["x"])
        return out[:c["max_objects"]], alert
