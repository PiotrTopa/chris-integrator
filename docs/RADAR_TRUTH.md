# Radar: truth from recorded drives

Until now only the simulator (docs/VISUAL_RADAR_EVAL.md) had truth. Any claim about the radar on real roads was
unchecked. This is the real-drive counterpart: events taken from recorded drives, labels from looking at the
camera frames, and a scorer that re-runs the current radar against those labels and can act as a regression
gate.

| file | what |
|---|---|
| `tools/radar_review.py` | replays a session through the radar, extracts the events worth judging, writes a contact sheet and a JSON record per event |
| `data/radar_truth/<session>.jsonl` | the reviewed labels (in the repository; frames and sheets are NOT) |
| `tools/radar_truth_eval.py` | re-runs the radar, matches its objects to the labels by time and bearing, prints the scores, exits 1 when a gate threshold breaks |
| `tools/radar_truth_lib.py` | shared pieces: replay, detector boxes, camera frame times, anchors, matching |
| `tests/test_radar_truth.py` | label-file checks, plus the gate (skipped where the recordings are absent) |

## How to run

    # the review: events + contact sheets (keep the output OUTSIDE the repository: frames show people and plates)
    python3 tools/radar_review.py /home/piotr/rs-rec-20261122/20261122-202233 /home/piotr/rs-rec-20261122/20261122-203305 \
        --out /home/piotr/.claude/jobs/fc03ae36/tmp/review
    # the score and the gate (about 5 s per session)
    python3 tools/radar_truth_eval.py [--json /tmp/truth.json] [--no-gate] [--sessions-root DIR]
    RADAR_TRUTH_SESSIONS=/path/to/sessions python3 -m pytest -q tests/test_radar_truth.py

On the laptop, wrap the review and the eval as
`systemd-run --user --wait --collect --pipe -q -p MemoryMax=4G --working-directory=$PWD python3 ...`.
Both use calibration override pitch/yaw = -0.0140/-0.0624 rad (`--calib-override` on the review). That is the
value the 2026-11-22 recordings need.

## Event kinds (radar_review.py)

| kind | what | evidence on the sheet |
|---|---|---|
| `threat` | every episode of flag `threat` (gaps up to 0.5 s merged) | the object from 2 s before to 2 s after |
| `in_path` | every episode of flag `in_path` | same |
| `flip` | every object called both `moving` and `stationary` in its life | frames around the changes |
| `life` | every confirmed object's whole life: first/last time, seen span, verdict history, positions | up to 9 frames across it |
| `short` | confirmed objects seen (confirmed/coast) for under 1 s: candidate false tracks | frames around it |
| `unconfirmed` | a run of detector boxes (at least 3 boxes over at least 0.8 s at one bearing) that no confirmed radar object followed: candidate missed objects | frames of the run, a white bar at its bearing |

Each sheet shows the camera frames of the event, brightened (gamma 0.55). It also draws every detector box within
0.35 s (view 0 white, view 1 cyan), every radar object as a ring at its ground point, and the event's object as
a thick white ring with its claims in the caption: id, state, range, bearing, speed over the ground, verdict,
flags, ttc. A box drawn white with a margin is the box the event is anchored to. A 2x crop of that box sits
top right. `events.jsonl` holds the session, event id, kind, object id, time span, the radar's claims (verdict
and flag frame counts, ttc min, p_threat max, positions) and the anchor.

Camera frames are recorded at about 1 Hz (one jpg per 20 video frames, named by the `camera` message's `seq`,
which gives each frame its monotonic time). Boxes come at about 3.4 Hz, radar frames at about 12 Hz. An event
shorter than a second gets 1-2 frames of its own. The sheet widens to the 5 nearest frames.

## Label schema (`data/radar_truth/<session>.jsonl`, one JSON object per line)

    {"session": "20261122-203305", "kind": "object" | "threat" | "in_path",
     "t0": 1424.6, "t1": 1429.4,                 # session monotonic clock, s (ts_mono_ns / 1e9)
     "anchor": [{"t": 1424.62, "b": -2.5,        # bearing, deg, left positive, through the calibration above
                 "r": 15.2,                      # rough ground range of the box (or the radar's, src "radar"), m
                 "box": [x0, y0, x1, y1],        # detector box, 1280x720 pixels
                 "src": "box" | "radar"}, ...],
     "truth": "parked",                          # kind object: parked | standing (in traffic, not parked) |
                                                 #   standing_in_lane | moving_with | oncoming | crossing |
                                                 #   pedestrian | cyclist | not_an_object | cannot_tell
     "justified": false,                         # kind threat / in_path: true | false | "cannot_tell"
     "confidence": "sure" | "likely" | "guess",
     "reason": "one line",
     "review": {"event": "...", "radar_id": 24}} # where the label came from: informational, never used to match

A labelled thing is its anchor: the detector boxes the radar's bearing pointed at (or the radar's own places
when no box was there, `src: "radar"`: those labels are weak and mostly `cannot_tell`). It is not a radar id,
so labels survive tracker changes. Where one radar track wandered from one thing to another, the label is split
at that time (in 20261122-203305, track 17 becomes a car facing the ego and then a parked white car). `t0..t1` is
the span of the anchor, so verdicts are scored only while there was box evidence.

Matching in the eval: a radar object belongs to a label when, in at least 2 of the frames in `t0-0.3..t1+0.3`
where a sensor saw it (state confirmed/coast), its bearing lay within 3 deg of the anchor (interpolated; more
for objects nearer than 12 m) in at least half of them. A flag frame belongs to a reviewed episode when it falls
in `t0-1..t1+1` and within the bearing tolerance + 2 deg of the episode's anchor. A flag raised where no
episode was reviewed counts as `unreviewed`. That gate line tells you to re-run the review after a radar change.

Strictness: night frames come once a second. A motion label needs the thing visibly standing against the
scenery (or the ego standing). Otherwise it is `cannot_tell` or `guess`. Only `sure`/`likely` labels count for
"real thing missed". Verdict seconds count for every confidence but `cannot_tell`.

## Scores (radar_truth_eval.py)

* verdict seconds per truth class (stationary / moving / unknown) of the best-matching radar object, inside the
  label's span. Wrong = `moving` for parked/standing, `stationary` for moving_with/oncoming/crossing. Pedestrians
  are shown but not judged.
* `threat` / `in_path` seconds and episodes: justified, false, cannot_tell, unreviewed. Also justified episodes
  the radar no longer flags.
* `not_an_object` labels that the radar confirmed.
* real things (`sure`/`likely`) that no radar object followed.

The gate is the table `GATE` at the top of the tool, set from the baseline below with a little room. Tighten a
line when the radar improves. Loosen one only with a reviewed reason.

## Baseline (2026-11-23, radar at 94e54dd, 66 labels)

| metric | 20261122-202233 | 20261122-203305 | total | gate |
|---|---|---|---|---|
| judged object-seconds with the wrong verdict | 5.5 of 58.4 s | 9.5 of 31.6 s | 15.0 of 90.0 s (17 %) | <= 22 % |
| parked: stationary / moving / unknown s | 2.2 / 4.3 / 10.4 | 8.1 / 9.5 / 14.0 | 10.3 / 13.8 / 24.4 | |
| moving_with: stationary / moving / unknown s | 0 / 36.5 / 0.2 | – | 0 / 36.5 / 0.2 | |
| false `threat` | 0.31 s, 1 episode | 0.31 s, 2 episodes | 0.62 s, 3 episodes | <= 0.8 s, <= 3 |
| `in_path` | – | 1.23 s false (1 ep.), 1.34 s cannot_tell (1 ep.) | 1.23 s false | <= 1.6 s |
| justified `threat` / `in_path` episodes | 0 | 0 | 0 | |
| not_an_object confirmed | 0 of 2 | 0 of 1 | 0 | 0 |
| real things never followed | 5 | 7 | 12 | <= 13 |

What these numbers say:

* Not one `threat` or `in_path` episode in the two drives was justified. All three threats were on parked or
  adjacent-lane cars. One was a car in the right lane whose range came out 2.9 m when it was about 10 m ahead.
  The other two were parked cars at the right kerb and on the outside of a right bend (the simulator's
  `curve_parked` trap, here for real).
* Parked cars are called `moving` for 14 of the 48 parked seconds with a box. The cases: a blue car parked at a
  T-junction ahead followed as `with_traffic` at 3-7 m/s, a silver car parked on the left while the ego stopped,
  and parked cars on the right that one track hopped between. Cars driving with the ego (two lead cars, two in
  the next lane) were right: 36.5 s `moving`, nothing `stationary`.
* Missed: a pedestrian crossing close in front of the car (202233, t 1045.5) never became a radar object. The
  same goes for a crossing car while the ego stood, and for five cars parked around the ego's final parking
  manoeuvre (ego at walking pace or standing).
* Two reflective bollards/barriers the detector boxed as `human`, and a lit window boxed as `car`, were not
  confirmed (right).

### Labels breakdown

66 labels: 61 objects, 5 flag episodes. Objects: parked 31, cannot_tell 17 (28 % of objects), moving_with 4,
crossing 3, not_an_object 3, pedestrian 2, standing 1. Confidence: sure 8, likely 29, guess 29. Episodes:
threat 3 (all false), in_path 2 (1 false, 1 cannot_tell).

## What the two sessions cover, and what they do not

| session | length | distance | speed | content |
|---|---|---|---|---|
| 20261122-202233 | 394 s (radar active 331 s) | 1.45 km | max 15.7 m/s, stopped 179 s | leaving a parking lot, residential streets with parked rows, a multi-lane city road with traffic lights, a lead car for about 30 s, a pedestrian at a crossing, wet road, glare |
| 20261122-203305 | 270 s | 0.20 km | max 6.5 m/s, stopped 192 s | a junction, narrow residential streets with cars parked both sides, a right bend with a parked car on its outside, parking the car |

Both are at night, in a city, wet, with the camera frames at 1 Hz and detector boxes on only 0.2-0.4 of the
messages. They do not cover:

* daylight (the detector sees far more by day; the simulator's day profile is unchecked);
* any speed above 16 m/s: no motorway, no expressway, no rural road;
* a real obstacle in the ego lane: a stopped car ahead that the ego brakes for or steers around. This is the one
  event `threat`/`in_path` exist for, and the truth set has no justified episode;
* oncoming traffic on a two-way road at speed, overtaking, being overtaken, cut-ins;
* queues at a light with the ego stopping behind a lead (only the approach is recorded);
* pedestrians or cyclists on the road ahead in good light;
* rain at speed, fog, low sun, a dirty windscreen;
* a second camera or a microwave radar (both sessions are one camera only).

## Recordings still missing, in order of priority

Record each with the camera frames at a higher rate if storage allows (at least 5 Hz around events: 1 Hz cannot
show a 1 s threat). Keep the calibration valid and note the time and place in the session's README.

1. **Day, city, stopped vehicles ahead**: 20 min of urban driving with queues at lights (the ego stops behind
   a lead at least 10 times), delivery vans and buses stopped in the lane, double-parked cars the ego goes
   around. This gives the justified `in_path`/`threat` episodes. Today there are none.
2. **Day, motorway / expressway at 25-36 m/s**: 30 min including slowing into a jam (closing on a slow lead),
   lane changes, overtaking and being overtaken, a truck in the next lane, at least one stopped or very slow
   vehicle on the hard shoulder. Covers `with_traffic` at speed and the stopped-ahead case at the speeds where
   it matters.
3. **Day, rural two-way road at 14-25 m/s**: 20 min with oncoming traffic every few seconds, curves with cars
   parked or stopped on the outside, tractors or cyclists ahead. Covers `oncoming`, `curve_parked` by day, and
   slow road users in the lane.
4. **Day, residential streets at walking to 10 m/s**: 15 min with pedestrians crossing, cyclists, cars pulling
   out of parking spaces and reversing. Covers `crossing`, `pedestrian` and parked-to-moving changes.
5. **Night, motorway and rural**: 20 min each at speed, for the night detector at range. Today's night data is
   city only.
6. **Rain / wet at speed, and low sun**: 15 min each on any road above 15 m/s.
7. **The same night city loop as 2026-11-22 with frames at 5 Hz**: 10 min. Lets today's `cannot_tell` labels
   (28 % of objects) be judged, and checks whether the pedestrian miss repeats.
