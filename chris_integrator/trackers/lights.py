"""Traffic lights: which light applies to the ego lane (adds `ego` and per-light geometry to `lights` messages).

A light provider (roadsense-signs `signd --lights`, docs/ARCHITECTURE.md "lights") publishes tracked light boxes
with a smoothed state. The provider does not know the road; the api does (modeld's `road` plan, the calibration
and the intrinsics), so it adds per light:

  dist_m     distance from the box's long side, assuming a 0.9 m housing (three 200-300 mm lamps):
             f * 0.9 / long_side_px (a 30 px housing at f 690 px = 21 m). Rough (+-30 %), only used for ranking.
  lateral_m  offset of the light from the planned path at that distance, metres (+ = right of the path), from
             the path point projected into the image: (u_light - u_path(dist)) * dist / f.
  relevance  0..1: facing state (red/yellow/red_yellow/green; off/unknown are mostly side or back views) x
             lateral window (lights for the ego lane hang over it or stand at its right edge, a little to the
             left on a median: centre +1.5 m, scale 4.5 m, cut outside -7..+10 m) x distance (full up to 60 m,
             zero at 120 m) x track maturity (hits / 3) x sqrt(detector score)
  ego        true for the selected light

and a message-level `ego`: {track_id, state, flashing, dist_m, lateral_m, relevance, agree, n_relevant, source}
for the most relevant light, whose state is the relevance-weighted vote of the lights within 30 % of the best
(two heads of the same signal group agree), or null when nothing is relevant. Without a road message the
path is the image column of the principal point (source "image").

A light the provider reports while coasting (`coast` > 0: a confirmed track it did not detect in this frame,
roadsense-signs >= 2026-11-16) counts with COAST_FACTOR per missed frame.

Over time (`EgoLightSelector`, used by the api): the choice above is made per message, so it used to drop out
whenever the ego light's relevance dipped under the threshold or its track missed a frame, and consumers saw
the light blink. The selector keeps the light that was the ego light while its relevance stays above
KEEP_RELEVANCE (hysteresis), and when no light qualifies it serves the last ego light for up to HOLD_S more
with `held: true` and its `age_s` (the state shown is then that old).

Map prior (optional, soft): when mapd has matched the road and its look-ahead lists a signal-controlled
junction or crossing within MAP_NEAR_M on the most likely path (`ahead.features` kind `traffic_signals` /
`crossing_signals`), a light is expected: the selector accepts an ego light from MAP_MIN_RELEVANCE instead
of MIN_RELEVANCE, holds it MAP_HOLD_S, and the message carries `map_signal: {kind, dist_m, confidence}` so a
consumer can tell "the map expects a light here and the camera has none" from "no light". Without a usable
map message (no position source, stale, a doubtful match) nothing changes and `map_signal` is null.

Lane assignment ("which of several lights is mine", lane L16): at a junction with separate signal groups (a
left-turn signal beside the one for straight on) the heads that face the car show different states at the
same moment. Then, and only then, the lateral window above is not what decides: every facing head gets a
`slot` (its offset from the centre of the ego lane in lane widths at its distance, from the tracked lane
lines of the `lanes` topic; 0 = over the ego lane, -1 = over the lane to the left) and a lane weight (a
Gaussian in the slot whose width grows with the distance, because the lateral offset of a far light is rough;
a head beyond the outermost lane, i.e. on a post at the roadside, counts as a head of the outermost lane it stands
beside, a little weaker). The heads are split into "stop" (red, red_yellow, yellow) and "go" (green); the group with the best
head wins and the ego light is chosen inside it, `assign.confidence` is that group's share of the two best
scores, and `amb` is set when the share stays under AMB_CONFIDENCE (the selector reports it from the second
message on). The other group is reported as `assign.alt` {state, side}. Without lane lines the slot is
measured from the planned path with a nominal lane width (`basis: "path"`). When all facing heads agree there
is no conflict and the choice is the one above; a lone light that is clearly over another lane (lane weight
under LANE_GATE with lane lines present, nearer than LANE_GATE_D) is still served, since its state is all
there is, with `assign.other_lane` and `amb` set. The assignment never removes a light: it decides between
groups and says how sure it is. A tagged lane count
of the matched road (map, optional) caps the number of lanes assumed beside the ego lane. Arrow lenses are
not told apart: the provider has no arrow state and a lens is 4-9 px wide at the distances that matter.

What the selector remembers (lane "drive-quality", from the drives of 26 / 27 November 2026; `memory = False` is
the selection without any of it). Per message the choice above was right about as often as its inputs were, and
at a junction they are worst exactly where the car stands:

* The path. The plan is a few seconds of driving: at walking pace it ends a few metres ahead and at rest it is a
  point, so a light's offset "from the path" was measured from the camera's axis (and, before, from the picture's
  centre column, which is not where a yawed camera's car is heading). The selector keeps the last plan that
  reached PATH_MEM_REACH_M while the car moved, in the estimator's local frame (`ego_state.position.local`), and
  uses it, carried to the car's present pose, while the live plan is shorter than that (`ego.source`
  "path_mem"; for PATH_MEM_M of driving and PATH_MEM_S at most). A plan that does not reach a light is continued
  along its last heading when it covers PATH_REACH of the distance; otherwise the light is measured from the
  car's own axis (through the calibration).
* The size of a housing, and with it the distance and the offset, follows the whole sightings of the track with
  SIZE_GAIN per message instead of being read from one 10 px box.
* What a view saw of a light. The provider alternates two views; a housing that a view (or the picture) cuts is
  detected as its lower lamps only, "off", with a shorter box, and the provider's smoothed state then alternates
  with the views. A box that touches the edge of the view it was detected in is no evidence against a colour seen
  CLIP_STATE_S ago at most, and its size is not smaller than the last whole one (SIZE_MEM_S); a track the provider
  only predicts because the frame's view does not cover it is not counted as coasting. A red lamp does not go
  dark before it changes (an LED lamp caught between its pulses, a lamp behind a wiper): a housing that showed red
  RED_HOLD_S ago at most and reads `off` is red. Where the selector serves
  another state than the provider's, the light carries `state_in` (the provider's).
* The choice. The ego light stays the ego light while no other is better by SWITCH_RATIO: between two heads of
  one colour (relevance), and between two signal groups (lane score), where it also carries the track through
  its own change of colour.
* The slot of a light. Lane lines end at the junction's mouth, and the plan there may already turn: the slot a
  track had while there were lines (within SLOT_MEM_D, SLOT_GAIN per message) is the slot it keeps where there
  are none.
* Heads on one gantry. Housings at the same height above the road (`rise`, the height over the camera in housing
  lengths, needs no distance) belong to the lanes under them: with a head of the same gantry nearer the ego lane, a
  head is never taken for the roadside post of the outermost lane. Such a head, when it is all that is left to
  see, is `other_lane` also without lane lines.
* The lanes beside the car. The ego-lane estimate (`egolane`: lanes to the right, and to the left where the map
  counts them) says how many lanes there are on a side when the lane lines at the junction do not.
* The ego light that left the picture. A head over the ego lane is the first to leave through the top of the
  picture when the car stops at the line. A head of another lane (`other_lane`) is then not served in its place
  while the car stands there (LOST_M of driving, LOST_S): the message says `ego: null` and
  `out_of_view: {track_id, state, age_s}` (the state is the last one seen and that old).
"""
import math

import numpy as np

from ..native import round                  # builtins.round (its results; the native core)
from .projection import Projector

HOUSING_M = 0.9
FACING = {"red": 1.0, "yellow": 1.0, "red_yellow": 1.0, "green": 1.0, "off": 0.15, "unknown": 0.15}
LAT_CENTRE, LAT_SCALE, LAT_MIN, LAT_MAX = 1.5, 4.5, -7.0, 10.0
D_FULL, D_ZERO = 60.0, 120.0
MIN_RELEVANCE = 0.15
GROUP_FRAC = 0.3
COAST_FACTOR = 0.85          # relevance factor per frame a reported track has been coasting
KEEP_RELEVANCE = 0.07        # the current ego light stays the ego light down to this relevance
HOLD_S = 1.2                 # the last ego light is served this long after no light qualifies
MAP_NEAR_M = 150.0           # a mapped signal this near on the path = a light is expected
MAP_MIN_RELEVANCE = 0.10     # ... then a light qualifies from this relevance
MAP_HOLD_S = 2.0             # ... and the hold is this long
MAP_MAX_AGE_S = 3.0          # a map message older than this is not used
MAP_MIN_CONFIDENCE = 0.5     # match confidence, and confidence of the look-ahead feature
SIGNAL_KINDS = ("traffic_signals", "crossing_signals")
STOP_STATES, GO_STATES = ("red", "red_yellow", "yellow"), ("green",)
LANE_EVAL_M = (6.0, 30.0)    # the lane lines are read at the light's distance, clamped to this range
LANE_WIDTH_M = 3.3           # nominal lane width when it cannot be measured
LANE_WIDTH_RANGE = (2.5, 4.5)
SLOT_SIGMA = 0.7             # lane weight = exp(-(slot_eff / (SLOT_SIGMA * (1 + d / SLOT_SIGMA_D))) ** 2)
SLOT_SIGMA_D = 60.0
POST_EXCESS = 0.5            # a head beyond the outermost lane (a post): its distance past that lane's centre counts this much
LANE_MIN = 0.12              # in a conflict: lane score (relevance without the lateral window x lane weight) to qualify
LANE_KEEP = 0.06             # ... for the current ego track
LANE_GATE = 0.08             # no conflict: a lone head with a lane weight under this is marked `other_lane` (ambiguous)
LANE_GATE_D = 45.0           # ... when it is nearer than this (the offset of a far light is too rough to say)
LAT_FACING = 14.0            # heads farther than this from the path take no part in the lane assignment
AMB_CONFIDENCE = 0.7         # conflict won with a smaller share than this = ambiguous
AMB_MSGS = 2                 # ... reported (`amb`) from this many messages in a row
# --- what the selector remembers (module docstring, "What the selector remembers")
PATH_MIN_M = 8.0             # a plan shorter than this is no path at all
PATH_REACH = 0.6             # a plan is continued along its last heading to a light when it covers this share of the distance
PATH_MEM_REACH_M = 25.0      # a plan this long, driven at PATH_MEM_SPEED or more, is remembered; a shorter live one yields to it
PATH_MEM_SPEED = 2.0
PATH_MEM_M = 80.0            # ... for this much driving
PATH_MEM_S = 180.0           # ... and this long (a red phase and a queue)
VIEW_EDGE_PX = 3.0           # a box this near the edge of the view (or of the picture) it was detected in is cut
CLIP_STATE_S = 2.0           # a cut or unseen housing keeps the colour it showed this long
RED_HOLD_S = 3.0             # a red lamp does not go dark: a red housing read as off stays red this long
CLIP_RECENT_S = 1.0          # ... and the provider's smoothed state is doubted this long after a cut detection
SIZE_MEM_S = 1.5             # a cut box is not smaller than the last whole one seen within this
SIZE_GAIN = 0.4              # the housing's length follows the whole sightings with this gain per message
SLOT_MEM_D = 70.0            # a slot measured on lane lines within this distance is remembered for the track ...
SLOT_GAIN = 0.3              # ... with this gain per message
SLOT_MEM_N = 3               # ... and used where there are no lane lines once it rests on this many
SWITCH_RATIO = 1.3           # another light takes over when it is better by this factor
RISE_OVERHEAD = 3.0          # housing centre this many housing lengths above the camera: a gantry head (a post: about 2)
RISE_SAME = 0.35             # two heads within this relative difference of rise ...
DIST_SAME = 1.6              # ... and this ratio of distance hang on one gantry
SIBLING_NEARER = 0.3         # ... and one of them is this much nearer the ego lane (lane widths): the other is no post
OTHER_LANE_W = 0.3           # a gantry head with a nearer sibling and a lane weight under this is another lane's
EGOLANE_P = 0.6              # the ego-lane estimate counts the lanes on a side from this probability
LOST_M = 15.0                # the ego light that left the picture is remembered for this much driving
LOST_S = 150.0               # ... and this long
LOST_EDGE_PX = 40.0          # it left through the top when its last box began this near the top edge


class Path:
    """A planned path as points in the device frame at a message's time (x forward, y right, z down from the
    camera), and where it is at a distance."""

    def __init__(self, x, y, z, source="path"):
        self.x, self.y, self.z, self.source = x, y, z, source
        self.reach = float(x[-1]) if len(x) else 0.0

    @classmethod
    def of_road(cls, road, min_reach=0.0):
        """The plan of a `road` message, or None (no plan, or one that ends before `min_reach`). The plan of a
        car at rest folds back on itself: the rising part from the car is what counts."""
        plan = (road or {}).get("plan") or {}
        xs = plan.get("x")
        if not xs:
            return None
        x = np.asarray(xs, float)
        y = np.asarray(plan.get("y") or np.zeros_like(x), float)
        z = np.asarray(plan.get("z") or np.zeros_like(x), float)
        n = len(x)
        if len(y) != n or len(z) != n or not np.all(np.isfinite(x)):
            return None
        back = np.nonzero(np.diff(x) <= 0.0)[0]
        if len(back):
            n = int(back[0]) + 1
        if n < 2 or x[n - 1] < min_reach:
            return None
        return cls(x[:n], y[:n], z[:n])

    def at(self, d):
        """(y, z) of the path at distance d, or None when it does not reach that far (PATH_REACH)."""
        x, y, z = self.x, self.y, self.z
        end = x[-1]
        if d <= end:
            d = max(d, float(x[0]))
            return float(np.interp(d, x, y)), float(np.interp(d, x, z))
        if end < PATH_MIN_M or end < PATH_REACH * d:
            return None
        k = int(np.searchsorted(x, 0.8 * end))
        k = min(max(k, 0), len(x) - 2)
        dx = end - x[k]
        if dx <= 0.0:
            return float(y[-1]), float(z[-1])
        return (float(y[-1] + (y[-1] - y[k]) / dx * (d - end)), float(z[-1] + (z[-1] - z[k]) / dx * (d - end)))

    def world(self, pose):
        """The points in the local frame of `pose` = (east, north, heading counter-clockwise from east)."""
        e, n, psi = pose
        c, s = math.cos(psi), math.sin(psi)
        return e + self.x * c + self.y * s, n + self.x * s - self.y * c

    @classmethod
    def of_world(cls, we, wn, z, pose, source="path_mem"):
        """World points seen from `pose`: the part ahead of the car, or None when little of it is."""
        e, n, psi = pose
        c, s = math.cos(psi), math.sin(psi)
        de, dn = we - e, wn - n
        x, y = de * c + dn * s, de * s - dn * c
        ahead = np.nonzero(x > 0.0)[0]
        if not len(ahead):
            return None
        k = int(ahead[0])
        x, y, z = x[k:], y[k:], z[k:]
        back = np.nonzero(np.diff(x) <= 0.0)[0]
        if len(back):
            m = int(back[0]) + 1
            x, y, z = x[:m], y[:m], z[:m]
        if len(x) < 2:
            return None
        if k > 0 or x[0] > 0.5:                    # from the car: the path passes where the car is now
            x, y, z = np.concatenate(([0.0], x)), np.concatenate(([0.0], y)), np.concatenate(([z[0]], z))
        return cls(x, y, z, source)


def _plan_u(road, proj, d):
    """Image column of the plan at distance d, the plan's end standing for everything beyond it; None without a
    plan or with one that ends at the camera (the selection per message, as before lane "drive-quality")."""
    plan = (road or {}).get("plan") or {}
    xs = plan.get("x")
    if not xs:
        return None
    x = np.asarray(xs, float)
    y = np.asarray(plan.get("y") or np.zeros_like(x), float)
    z = np.asarray(plan.get("z") or np.zeros_like(x), float)
    if d <= x[0] or d > x[-1]:
        d = min(max(d, x[0]), x[-1])
    yi, zi = float(np.interp(d, x, y)), float(np.interp(d, x, z)) + proj.height
    pts = proj.project([d], [yi], [zi])
    return pts[0][0] if pts else None


def _path_u(path, proj, d):
    """Image column of the path at distance d, or of the car's own axis where the path does not reach."""
    at = path.at(d) if path is not None else None
    yi, zi = at if at is not None else (0.0, 0.0)
    pts = proj.project([d], [yi], [zi + proj.height])
    return pts[0][0] if pts else proj.cx


def map_signal(map_msg, now_s=None):
    """The nearest signal on the path from a mapd `map` message: {"kind", "dist_m", "confidence"}, or None
    when the map says nothing usable (no message, not available, stale, a doubtful match, no signal near)."""
    if not isinstance(map_msg, dict) or not map_msg.get("available"):
        return None
    ts = map_msg.get("ts_mono_ns")
    if now_s is not None and isinstance(ts, (int, float)) and now_s - ts / 1e9 > MAP_MAX_AGE_S:
        return None
    conf = (map_msg.get("match") or {}).get("confidence")
    if not isinstance(conf, (int, float)) or conf < MAP_MIN_CONFIDENCE:
        return None
    best = None
    for ft in (map_msg.get("ahead") or {}).get("features") or []:
        if not isinstance(ft, dict) or ft.get("kind") not in SIGNAL_KINDS:
            continue
        d, fc = ft.get("dist_m"), ft.get("confidence")
        if not isinstance(d, (int, float)) or d < 0 or d > MAP_NEAR_M:
            continue
        if isinstance(fc, (int, float)) and fc < MAP_MIN_CONFIDENCE:
            continue
        if best is None or d < best["dist_m"]:
            best = {"kind": ft["kind"], "dist_m": round(float(d), 1),
                    "confidence": round(float(fc), 2) if isinstance(fc, (int, float)) else None}
    return best


def _at(xs, ys, d):
    """Linear interpolation of a line (x increasing) at d, or None when a value is missing."""
    try:
        v = float(np.interp(d, np.asarray(xs, float), np.asarray(ys, float)))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def lane_frame(lanes, road, d, lanes_forward=None, path=None, hint=None):
    """The ego lane at a light's distance: {"c": lane centre minus path (m, + right), "w": lane width,
    "n_left" / "n_right": lanes between the ego lane and the road edge on that side (None = unknown),
    "basis": "lanes" | "path"}. `lanes` is the api's `lanes` message (tracked lines); without usable inner
    lines the frame is the planned path with a nominal width. `lanes_forward`: a tagged lane count of the
    matched road in the car's direction (map), which caps the lanes beside the ego lane. `path`: the path the
    lights are measured from (a Path; None: the plan of `road` as it is). `hint`: (lanes to the left, lanes to the right)
    of the ego-lane estimate, each None where it does not know; it replaces what the lines at the junction say."""
    out = {"c": 0.0, "w": LANE_WIDTH_M, "n_left": None, "n_right": None, "basis": "path"}
    if isinstance(lanes, dict) and isinstance(lanes.get("x"), list):
        dq = min(max(d, LANE_EVAL_M[0]), LANE_EVAL_M[1])
        xs = lanes["x"]
        if path is not None:
            at = path.at(dq)
            yp = at[0] if at is not None else 0.0
        else:
            plan = (road or {}).get("plan") or {}
            yp = _at(plan.get("x"), plan.get("y"), dq) if plan.get("x") else 0.0
            if yp is None:
                yp = 0.0

        def y_of(ln):
            if not isinstance(ln, dict) or ln.get("present") is not True or not ln.get("y"):
                return None
            return _at(xs, ln["y"], dq)
        lines = [y_of(ln) for ln in (lanes.get("lines") or [])[:4]] + [None] * 4
        edges = [y_of(ln) for ln in (lanes.get("edges") or [])[:2]] + [None] * 2
        yl, yr = lines[1], lines[2]
        w0 = lanes.get("lane_width_m")
        w0 = float(w0) if isinstance(w0, (int, float)) and LANE_WIDTH_RANGE[0] <= w0 <= LANE_WIDTH_RANGE[1] \
            else LANE_WIDTH_M
        if yl is not None and yr is not None and yr > yl:
            w = min(max(yr - yl, LANE_WIDTH_RANGE[0]), LANE_WIDTH_RANGE[1])
            out.update(c=0.5 * (yl + yr) - yp, w=w, basis="lanes")
        elif yl is not None:
            out.update(c=yl + 0.5 * w0 - yp, w=w0, basis="lanes")
            yr = yl + w0
        elif yr is not None:
            out.update(c=yr - 0.5 * w0 - yp, w=w0, basis="lanes")
            yl = yr - w0
        if out["basis"] == "lanes":
            w = out["w"]
            if edges[0] is not None:
                out["n_left"] = int(min(4, max(0, round((yl - edges[0] - 0.3) / w))))
            if lines[0] is not None:
                out["n_left"] = max(out["n_left"] or 0, 1)
            if edges[1] is not None:
                out["n_right"] = int(min(4, max(0, round((edges[1] - yr - 0.3) / w))))
            if lines[3] is not None:
                out["n_right"] = max(out["n_right"] or 0, 1)
    if hint is not None:
        if hint[0] is not None:
            out["n_left"] = hint[0]
        if hint[1] is not None:
            out["n_right"] = hint[1]
    if isinstance(lanes_forward, int) and lanes_forward >= 1:
        cap = lanes_forward - 1                  # lanes of the own direction beside the ego lane
        if out["n_right"] is None or out["n_right"] > cap:
            out["n_right"] = cap
    return out


def lane_weight(lat, d, fr, post=True, slot=None):
    """(slot, weight 0..1): the head's offset from the ego-lane centre in lane widths, and how much it counts
    as a light for the ego lane (module docstring, "Lane assignment"). post=False: a head that hangs over a
    lane of its own (a gantry head with a sibling nearer the ego lane) is never the outermost lane's post. `slot`:
    the slot to weigh instead of the one `lat` gives (the selector's memory of the track)."""
    s = (lat - fr["c"]) / fr["w"] if slot is None else slot
    n = fr["n_right"] if s > 0 else fr["n_left"]
    a = abs(s)
    if not post:
        eff = a
    elif n is None:                              # unknown: a post is expected on the right, not on the left
        eff = a if a <= 0.5 or s < 0 else 0.5 + POST_EXCESS * (a - 0.5)
    elif a > n + 0.5:                            # beyond the outermost lane: a post at the roadside
        eff = n + POST_EXCESS * (a - n)
    else:
        eff = a
    sig = SLOT_SIGMA * (1.0 + d / SLOT_SIGMA_D)
    return s, math.exp(-(eff / sig) ** 2)


def _colour(state):
    return "stop" if state in STOP_STATES else "go" if state in GO_STATES else None


def _horizon_v(proj):
    """Image row of the horizon straight ahead (a far point at the camera's height)."""
    pts = proj.project([1000.0], [0.0], [0.0])
    return pts[0][1] if pts else proj.cy


def annotate(msg, road=None, calib=None, intrinsics=(690.0, 640.0, 360.0, 1280, 720), keep_track=None,
             min_relevance=MIN_RELEVANCE, lanes=None, lanes_forward=None, lane_assign=True, path=None, hint=None,
             switch_ratio=None):
    """Adds dist_m / lateral_m / relevance / ego to every light of `msg` (in place) and msg['ego']; returns msg.
    `keep_track`: the track that was the ego light so far; it qualifies down to KEEP_RELEVANCE.
    `lanes`: the api's `lanes` message for the lane assignment; `lane_assign=False` is the selection as it
    was before lane L16 (lateral window only).
    What the selector adds from its memory (module docstring): `path`, the Path the lights are measured from
    (its `source` goes into `ego.source`; None: the car's own axis); `hint`, the ego-lane estimate's lanes
    (left, right); `switch_ratio`, the factor by which another light must be better than `keep_track` to take
    over (None: the choice per message as before the memory, with the plan of `road` as the path and no gantry rule). A light may carry `_long_px` (its housing length when the
    box is cut), `_coast` (the coasting frames that count) and `_nopost` (a gantry head that had a sibling nearer
    the ego lane); they are removed. With `switch_ratio` the message gets `_gantry`, the tracks found to be such
    heads in this message."""
    f, cx, cy, w, h = intrinsics
    fw, fh = (msg.get("frame") or [w, h])[:2]
    proj = Projector((f * fw / w, cx * fw / w, cy * fw / w, fw, fh), calib, (fw, fh))   # frame pixels
    fpx = proj.f
    best, cands = None, []
    mem = switch_ratio is not None               # the selector's memory is at work (else: per message, as before)
    if mem:
        source = path.source if path is not None else "image"
    else:
        path = None
        source = "path" if (road or {}).get("plan", {}).get("x") else "image"
    vh = _horizon_v(proj) if lane_assign else None
    for li in msg.get("lights") or []:
        box = li.get("box")
        long_px, coast, nopost = li.pop("_long_px", None), li.pop("_coast", None), li.pop("_nopost", False)
        slot_mem = li.pop("_slot", None)
        if not box or len(box) != 4:
            continue
        bw, bh = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
        size = long_px if long_px else max(bw, bh)
        d = fpx * HOUSING_M / size
        u = 0.5 * (box[0] + box[2])
        if mem:
            up = _path_u(path, proj, d)
        else:
            up = _plan_u(road, proj, d) if source == "path" else None
            if up is None:
                up = proj.cx
        lat = (u - up) * d / fpx
        state = li.get("state") or "unknown"
        coast = max(0, int(li.get("coast") or 0)) if coast is None else coast
        base = FACING.get(state, 0.15) * max(0.0, min(1.0, (D_ZERO - d) / (D_ZERO - D_FULL)))
        base *= min(1.0, (li.get("hits") or 1) / 3.0) * math.sqrt(max(0.0, float(li.get("score") or 0.0)))
        base *= COAST_FACTOR ** coast
        rel = base * math.exp(-((lat - LAT_CENTRE) / LAT_SCALE) ** 2) if LAT_MIN <= lat <= LAT_MAX else 0.0
        li.update(dist_m=round(d, 1), lateral_m=round(lat, 1), relevance=round(rel, 3), ego=False)
        li.pop("slot", None)
        if lane_assign:
            fr = lane_frame(lanes, road, d, lanes_forward, path, hint)
            # without lane lines the slot the track had while they were there says more than the path at a junction
            slot_in = slot_mem if fr["basis"] != "lanes" else None
            slot, wl = lane_weight(lat, d, fr, post=not nopost, slot=slot_in)
            li["slot"] = round(slot, 2)
            # the housing's centre above the camera, in housing lengths (the long side may lie: a horizontal head)
            rise = (vh - 0.5 * (box[1] + box[3])) / size
            li["_lane"] = [fr, wl, base * wl if abs(lat) <= LAT_FACING else 0.0, base, lat, d, rise, bool(nopost), slot_in]
        cands.append(li)
        if rel >= min_relevance and (best is None or rel > best["relevance"]):
            best = li
    if lane_assign and mem:
        _siblings(cands)
        # for the selector's memory: the gantry heads with a nearer sibling, the slots measured on lane lines
        msg["_gantry"] = [li.get("track_id") for li in cands if li["_lane"][7]]
        msg["_slots"] = {li.get("track_id"): (li["_lane"][4] - li["_lane"][0]["c"]) / li["_lane"][0]["w"]
                         for li in cands if li["_lane"][0]["basis"] == "lanes" and li["_lane"][5] <= SLOT_MEM_D}
        # a head of another lane is not the choice while a head that may be mine qualifies
        if best is not None and _other_lane(best):
            mine = [li for li in cands if li["relevance"] >= min_relevance and not _other_lane(li)]
            if mine:
                best = max(mine, key=lambda li: li["relevance"])
    if switch_ratio and best is not None and keep_track is not None and best.get("track_id") != keep_track:
        # two heads that both qualify: the one that was the ego light stays until the other is clearly better
        kept = [li for li in cands if li.get("track_id") == keep_track
                and li["relevance"] >= max(KEEP_RELEVANCE, best["relevance"] / switch_ratio)]
        if kept:
            best = kept[0]
    assign = None
    if lane_assign:
        best, assign = _assign(cands, best, keep_track, switch_ratio)
    if best is None and keep_track is not None and not (assign and assign["conflict"]):
        # hysteresis: the light that was the ego light so far
        kept = [li for li in cands if li.get("track_id") == keep_track and li["relevance"] >= KEEP_RELEVANCE]
        best = kept[0] if kept else None
        if best is not None and lane_assign:
            assign = _assign_single(best)
    for li in cands:
        li.pop("_lane", None) if li is not best else None
    if best is None:
        msg["ego"] = None
        return msg
    best["ego"] = True
    best.pop("_lane", None)
    group = [li for li in cands if li["relevance"] >= GROUP_FRAC * best["relevance"]]
    if assign and assign["conflict"]:                # two signal groups: the vote stays inside the chosen one
        col = _colour(best.get("state"))
        group = [li for li in group if _colour(li.get("state")) == col] or [best]
    votes = {}
    for li in group:
        votes[li.get("state")] = votes.get(li.get("state"), 0.0) + li["relevance"]
    state = max(votes, key=votes.get)
    msg["ego"] = {"track_id": best.get("track_id"), "state": state,
                  "flashing": bool(best.get("flashing")) and state == "yellow",
                  "dist_m": best["dist_m"], "lateral_m": best["lateral_m"], "relevance": best["relevance"],
                  "agree": sum(1 for li in group if li.get("state") == state), "n_relevant": len(group),
                  "source": source}
    if assign is not None:
        msg["ego"]["assign"] = assign
        msg["ego"]["amb"] = bool(assign["conflict"] and assign["confidence"] < AMB_CONFIDENCE
                                 or assign.get("other_lane"))
    return msg


def _siblings(cands):
    """Heads of one gantry (module docstring): a gantry head with another one of the same height and distance
    nearer the ego lane hangs over a lane of its own; its lane weight is taken without the post rule."""
    over = [li for li in cands if li["_lane"][6] >= RISE_OVERHEAD]
    for li in over:
        la = li["_lane"]
        for o in over:
            lo = o["_lane"]
            if o is li or abs(lo[6] - la[6]) > RISE_SAME * max(lo[6], la[6]):
                continue
            if max(lo[5], la[5]) > DIST_SAME * min(lo[5], la[5]):
                continue
            if abs(o["slot"]) + SIBLING_NEARER <= abs(li["slot"]):
                _, wl = lane_weight(la[4], la[5], la[0], post=False, slot=la[8])
                la[1], la[2], la[7] = wl, (la[3] * wl if abs(la[4]) <= LAT_FACING else 0.0), True
                break


def _other_lane(li):
    """A facing light that hangs over another lane: by the lane lines, or a gantry head with a sibling nearer
    the ego lane."""
    fr, wl = li["_lane"][0], li["_lane"][1]
    if li["dist_m"] > LANE_GATE_D:
        return False
    return fr["basis"] == "lanes" and wl < LANE_GATE or li["_lane"][7] and wl < OTHER_LANE_W


def _assign_single(best):
    fr, wl = best["_lane"][0], best["_lane"][1]
    out = {"conflict": False, "confidence": round(min(1.0, wl / 0.5), 2), "basis": fr["basis"],
           "slot": best["slot"], "n_left": fr["n_left"], "n_right": fr["n_right"], "alt": None}
    if _other_lane(best):
        out["other_lane"] = True                 # the only facing light, and it hangs over another lane
    return out


def _assign(cands, best, keep_track, switch_ratio=None):
    """Lane assignment over the annotated candidates: (the ego light or None, the `assign` object or None)."""
    groups = {"stop": [], "go": []}
    for li in cands:
        col = _colour(li.get("state"))
        if col and li["_lane"][2] > 0.0 and (li.get("hits") or 1) >= 2:
            groups[col].append(li)
    if not (groups["stop"] and groups["go"]):
        if best is None:
            return None, None
        return best, _assign_single(best)
    top = {c: max(g, key=lambda li: li["_lane"][2]) for c, g in groups.items()}
    sc = {c: top[c]["_lane"][2] for c in top}
    win = "stop" if sc["stop"] >= sc["go"] else "go"
    forced = None
    if switch_ratio:
        # the head that was the ego light keeps its group (also through its own change of colour) until the other
        # group's best head is better placed by the ratio
        for c in ("stop", "go"):
            o = "go" if c == "stop" else "stop"
            for li in groups[c]:
                if li.get("track_id") == keep_track and li["_lane"][2] >= LANE_KEEP \
                        and li["_lane"][2] * switch_ratio >= sc[o]:
                    win, forced = c, li
    lose = "go" if win == "stop" else "stop"
    share = sc[win] / (sc[win] + sc[lose])
    kept = [li for li in groups[win] if li.get("track_id") == keep_track and li["_lane"][2] >= LANE_KEEP]
    pick = forced if forced is not None else kept[0] if kept and kept[0]["_lane"][2] >= 0.6 * sc[win] else top[win]
    alt = top[lose]
    fr = pick["_lane"][0]
    assign = {"conflict": True, "confidence": round(share, 2), "basis": fr["basis"], "slot": pick["slot"],
              "n_left": fr["n_left"], "n_right": fr["n_right"],
              "alt": {"track_id": alt.get("track_id"), "state": alt.get("state"),
                      "side": "left" if alt["lateral_m"] < pick["lateral_m"] else "right",
                      "slot": alt["slot"]}}
    if pick["_lane"][2] < (LANE_KEEP if kept and pick is kept[0] else LANE_MIN):
        return None, assign
    return pick, assign


def _cut(box, roi):
    """Does the box touch an edge of the view it was detected in?"""
    return (box[0] <= roi[0] + VIEW_EDGE_PX or box[1] <= roi[1] + VIEW_EDGE_PX
            or box[2] >= roi[2] - VIEW_EDGE_PX or box[3] >= roi[3] - VIEW_EDGE_PX)


class EgoLightSelector:
    """The ego light over time: `annotate` with hysteresis on the current ego track and a short hold when no
    light qualifies, and with what the selector remembers of the path, of each light and of the ego light that
    left the picture (module docstring). One instance per camera stream; `reset` on a ride change."""

    def __init__(self, hold_s=HOLD_S):
        self.hold_s = hold_s
        self.map = None
        self.lanes = None
        self.egolane = None
        self.lane_assign = True
        self.memory = True          # False: the choice per message, as before lane "drive-quality"
        self.reset()

    def reset(self):
        self.track = None           # track_id of the current ego light
        self.last = None            # the last ego dict that came from a frame
        self.last_ts = None
        self.msg_ts = None          # stamp of the newest message
        self.amb_run = 0
        self.pose = None            # (local frame id, east, north, heading), odometer, speed of the newest ego_state
        self.path_mem = None        # (frame id, odometer, ts, world east, world north, z) of the last long plan
        self.tracks = {}            # track_id -> what the selector remembers of a light
        self.lost = None            # the ego light that left the picture: {track_id, state, ts, odo}

    def on_lanes(self, msg):
        """The api's `lanes` message (tracked lane lines): the frame of the lane assignment."""
        self.lanes = msg if isinstance(msg, dict) else None

    def on_egolane(self, est, ts_mono_ns=None):
        """The ego-lane estimate (`egolane.est`) and its stamp: how many lanes there are beside the car."""
        self.egolane = (est, ts_mono_ns) if isinstance(est, dict) else None

    def on_ego(self, es):
        """The estimator's `ego_state`: where the car is in its local frame (the path memory), and its odometer."""
        if not isinstance(es, dict):
            self.pose = None
            return
        loc = (es.get("position") or {}).get("local") or {}
        odo, v = es.get("odometer_m"), (es.get("speed") or {}).get("value")
        try:
            pose = (loc.get("frame"), float(loc["e_m"]), float(loc["n_m"]), float(loc["heading_rad"]))
        except (KeyError, TypeError, ValueError):
            pose = None
        self.pose = (pose, float(odo) if isinstance(odo, (int, float)) else None,
                     float(v) if isinstance(v, (int, float)) else None)

    def _lanes_forward(self, ts):
        """A tagged lane count in the car's direction from a usable `map` message, else None."""
        mp = self.map
        if not isinstance(mp, dict) or not mp.get("available"):
            return None
        t = mp.get("ts_mono_ns")
        if ts is not None and isinstance(t, (int, float)) and ts - t / 1e9 > MAP_MAX_AGE_S:
            return None
        conf = (mp.get("match") or {}).get("confidence")
        ln = (mp.get("road") or {}).get("lanes") or {}
        if not isinstance(conf, (int, float)) or conf < MAP_MIN_CONFIDENCE or ln.get("source") != "tagged":
            return None
        fwd = ln.get("forward")
        return int(fwd) if isinstance(fwd, (int, float)) and fwd >= 1 else None

    def on_map(self, msg):
        """mapd's `map` message: the signals on the path ahead (the soft prior of the module docstring)."""
        self.map = msg if isinstance(msg, dict) else None

    # ---- what the selector remembers -----------------------------------------------------------------
    def _path(self, road, ts):
        """The path the lights of this message are measured from: the live plan, or the remembered one."""
        live = Path.of_road(road, PATH_MIN_M)
        pose, odo, v = self.pose if self.pose is not None else (None, None, None)
        if pose is None or pose[0] is None or odo is None or ts is None:
            return live
        if live is not None and live.reach >= PATH_MEM_REACH_M:
            if v is not None and v >= PATH_MEM_SPEED:
                we, wn = live.world(pose[1:])
                self.path_mem = (pose[0], odo, ts, we, wn, live.z)
            return live
        mem = self.path_mem
        if mem is None:
            return live
        if mem[0] != pose[0] or not 0.0 <= odo - mem[1] <= PATH_MEM_M or not 0.0 <= ts - mem[2] <= PATH_MEM_S:
            self.path_mem = None
            return live
        old = Path.of_world(mem[3], mem[4], mem[5], pose[1:])
        if old is None or (live is not None and live.reach >= old.reach):
            return live
        return old

    def _hint(self, ts):
        """(lanes to the left, lanes to the right) of the ego-lane estimate; None where it does not know."""
        if self.egolane is None:
            return None
        est, t = self.egolane
        if ts is not None and isinstance(t, (int, float)) and abs(ts - t / 1e9) > MAP_MAX_AGE_S:
            return None
        p = est.get("probs")
        if not isinstance(p, list) or not p or not all(isinstance(v, (int, float)) for v in p):
            return None
        n = len(p)
        left = est.get("probs_from") == "left"       # the map counts the own lanes: both sides are known
        pr = p[::-1] if left else p                  # by lanes to the right
        k = max(range(n), key=lambda i: pr[i])
        nr = nl = None
        if pr[0] >= EGOLANE_P:
            nr = 0
        elif 1.0 - pr[0] >= EGOLANE_P:
            nr = max(1, k)
        if left:
            if p[0] >= EGOLANE_P:
                nl = 0
            elif 1.0 - p[0] >= EGOLANE_P:
                nl = max(1, n - 1 - k)
        return (nl, nr) if (nl is not None or nr is not None) else None

    def _observe(self, msg, ts):
        """What each view saw of each light (module docstring): the state and the size of a cut housing, the
        coasting that counts."""
        fw, fh = (msg.get("frame") or [1280, 720])[:2]
        roi = msg.get("roi")
        roi = roi if isinstance(roi, list) and len(roi) == 4 else [0, 0, fw, fh]
        seen_now = set()
        for li in msg.get("lights") or []:
            box, tid = li.get("box"), li.get("track_id")
            if not box or len(box) != 4 or tid is None:
                continue
            seen_now.add(tid)
            r = self.tracks.get(tid)
            if r is None:
                r = self.tracks[tid] = {"state": None, "state_ts": None, "long": None, "long_ts": None, "cut_ts": None,
                                        "box": None, "ts": None, "gantry": False, "lit": False,
                                        "big": None, "big_ts": None, "slot": None, "slot_n": 0}
            seen = not li.get("coast")
            cu, cv = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            in_view = roi[0] <= cu <= roi[2] and roi[1] <= cv <= roi[3]
            cut = seen and _cut(box, roi)
            st, fs = li.get("state"), li.get("frame_state", li.get("state"))
            if cut:
                r["cut_ts"] = ts
            if seen and (st in STOP_STATES or st in GO_STATES or fs in STOP_STATES or fs in GO_STATES):
                # a lit lamp was seen in this frame: the provider's smoothed colour, else the frame's
                r["state"], r["state_ts"] = (st if st in STOP_STATES or st in GO_STATES else fs), ts
            doubt = cut or not seen or not in_view or (r["cut_ts"] is not None and ts - r["cut_ts"] <= CLIP_RECENT_S)
            if st not in STOP_STATES and st not in GO_STATES and r["state"] is not None \
                    and (doubt and ts - r["state_ts"] <= CLIP_STATE_S
                         or r["state"] == "red" and ts - r["state_ts"] <= RED_HOLD_S):
                li["state_in"], li["state"] = st, r["state"]
            long_px = max(1.0, box[2] - box[0], box[3] - box[1])
            if seen and not cut:
                # the housing's length over the last few whole sightings: a 10 px box is a pixel wrong either way
                if r["long"] is None or ts - r["long_ts"] > SIZE_MEM_S:
                    r["long"] = long_px
                else:
                    r["long"] += SIZE_GAIN * (long_px - r["long"])
                r["long_ts"] = ts
                li["_long_px"] = r["long"]
            else:
                # cut by this view (or not seen): not smaller than the last whole sighting, nor than what another
                # view showed of it
                if seen:
                    if r["big"] is None or ts - r["big_ts"] > SIZE_MEM_S or long_px >= r["big"]:
                        r["big"], r["big_ts"] = long_px, ts
                size = [long_px] if seen else []
                if r["long"] is not None and ts - r["long_ts"] <= SIZE_MEM_S:
                    size.append(r["long"])
                if r["big"] is not None and ts - r["big_ts"] <= SIZE_MEM_S:
                    size.append(r["big"])
                if size:
                    li["_long_px"] = max(size)
            if r["slot_n"] >= SLOT_MEM_N:
                li["_slot"] = r["slot"]
            if not seen and not in_view:
                li["_coast"] = 0                 # the frame's view does not cover it: not a miss
            if r["gantry"]:
                li["_nopost"] = True
            r["lit"] = li.get("state") in STOP_STATES or li.get("state") in GO_STATES
            if seen:
                r["box"] = box
            r["ts"] = ts
        for tid in [t for t, r in self.tracks.items() if t not in seen_now and ts - r["ts"] > 5.0]:
            del self.tracks[tid]

    def _left_the_picture(self, ts):
        """The ego light is not in this message (or is no longer lit to see): did it leave through the top?"""
        r = self.tracks.get(self.track)
        if r is None or r["box"] is None or self.last is None:
            return
        if r["ts"] == ts and r["lit"]:
            return                               # it is there and lit: the choice moved on for another reason
        if r["box"][1] <= LOST_EDGE_PX and (self.last.get("dist_m") or 1e9) <= LANE_GATE_D:
            odo = self.pose[1] if self.pose is not None else None
            self.lost = {"track_id": self.track, "state": self.last.get("state"), "ts": self.last_ts, "odo": odo}

    def _lost(self, ts):
        """The ego light that left the picture, while the car still stands where it did; else None."""
        lo = self.lost
        if lo is None:
            return None
        odo = self.pose[1] if self.pose is not None else None
        if ts - lo["ts"] > LOST_S or (odo is not None and lo["odo"] is not None and not 0.0 <= odo - lo["odo"] <= LOST_M):
            self.lost = None
            return None
        return lo

    def annotate(self, msg, road=None, calib=None, intrinsics=(690.0, 640.0, 360.0, 1280, 720)):
        ts = msg.get("ts_mono_ns")
        ts = ts / 1e9 if isinstance(ts, (int, float)) else None
        if ts is not None and self.msg_ts is not None and (ts < self.msg_ts or ts - self.msg_ts > 10.0):
            self.reset()            # the stream restarted
        elif ts is not None and self.last_ts is not None and ts - self.last_ts > 10.0:
            self.last = self.last_ts = None
        if ts is not None:
            self.msg_ts = ts
        sig = map_signal(self.map, ts)
        lanes = self.lanes
        lt = lanes.get("ts_mono_ns") if isinstance(lanes, dict) else None
        if ts is not None and isinstance(lt, (int, float)) and abs(ts - lt / 1e9) > 1.0:
            lanes = None            # stale lane lines: the planned path is the frame
        mem = self.memory and ts is not None
        path = hint = None
        if mem:
            self._observe(msg, ts)
            path = self._path(road, ts)
            hint = self._hint(ts)
        annotate(msg, road, calib, intrinsics, keep_track=self.track,
                 min_relevance=MAP_MIN_RELEVANCE if sig else MIN_RELEVANCE,
                 lanes=lanes, lanes_forward=self._lanes_forward(ts), lane_assign=self.lane_assign,
                 path=path, hint=hint, switch_ratio=SWITCH_RATIO if mem else None)
        msg["map_signal"] = sig
        ego = msg.get("ego")
        if mem:
            for tid in msg.pop("_gantry", None) or ():
                if tid in self.tracks:
                    self.tracks[tid]["gantry"] = True
            for tid, slot in (msg.pop("_slots", None) or {}).items():
                r = self.tracks.get(tid)
                if r is not None:
                    r["slot"] = slot if r["slot"] is None else r["slot"] + SLOT_GAIN * (slot - r["slot"])
                    r["slot_n"] += 1
            if self.track is not None and (ego is None or ego.get("track_id") != self.track):
                self._left_the_picture(ts)       # the choice moved on: where did the last ego light go?
            lost = self._lost(ts)
            if lost is not None and ego is not None:
                if ego.get("track_id") == lost["track_id"]:
                    self.lost = lost = None      # it is back in the picture
                elif (ego.get("assign") or {}).get("other_lane"):
                    ego = msg["ego"] = None      # another lane's head, and mine is above the picture
                    for li in msg.get("lights") or []:
                        li["ego"] = False
                elif (ego.get("dist_m") or 0.0) > LANE_GATE_D:
                    self.lost = lost = None      # a light farther on: the next junction
            if lost is not None and ego is None:
                msg["out_of_view"] = {"track_id": lost["track_id"], "state": lost["state"],
                                      "age_s": round(ts - lost["ts"], 2)}
        if ego is not None and "amb" in ego:        # ambiguity is reported once it lasts
            self.amb_run = self.amb_run + 1 if ego["amb"] else 0
            ego["amb"] = self.amb_run >= AMB_MSGS
        elif ego is None:
            self.amb_run = 0
        hold_s = max(self.hold_s, MAP_HOLD_S) if sig and self.hold_s > 0 else self.hold_s
        if ego is not None:
            ego["held"] = False
            self.track, self.last, self.last_ts = ego.get("track_id"), dict(ego), ts
            return msg
        self.track = None
        if self.last is not None and ts is not None and self.last_ts is not None \
                and ts - self.last_ts <= hold_s:
            msg["ego"] = dict(self.last, held=True, age_s=round(ts - self.last_ts, 2))
        else:
            self.last = None
        return msg
