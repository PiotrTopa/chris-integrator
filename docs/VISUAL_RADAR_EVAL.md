# Visual radar: evaluation harness

The judge for the "visual radar" (the camera class detector's road users as tracked objects). It is
independent of the feature. It has a detector simulator with ground truth, traffic scenarios, metrics over
the whole stack, and a replay tool for real recordings.

| file | what |
|---|---|
| `chris_integrator/sim/roaduser_sim.py` | `ScenarioWorld` (a `sim.World` with actors), `DetectorSim` (`lights` messages with `road_users`), the scenarios |
| `tools/visual_radar_eval.py` | runs the scenarios through `stack.build(...)`, matches outputs to truth, prints the table and writes the JSON |
| `tools/visual_radar_replay.py` | replays a recorder session through the stack, prints per-session stats and writes contact sheets |

## How to run

    python3 tools/visual_radar_eval.py --both --json /tmp/vr.json        # everything, day + night (~10 min)
    python3 tools/visual_radar_eval.py --both --baseline                 # today's display objects
    python3 tools/visual_radar_eval.py --scenario alert --both --lead-off  # the road-user path alone, no camera lead
    python3 tools/visual_radar_eval.py --scenario curve_parked --seeds 5 --night
    python3 tools/visual_radar_replay.py SESSION [SESSION ...] --out DIR --calib-override -0.0140,-0.0624

On the laptop, wrap long runs as `systemd-run --user --wait --collect --pipe -p MemoryMax=6G -- python3 ...`.

What gets scored (`--source`, default `auto`):

1. `radar`: the radar component's stream, if the stack emits one. `--radar-plugin module:Class` adds the
   plug-in to the stack when `build()` does not include it.
2. `tracks`: the estimator's tracks, every track, camera lead included.
3. `view` (= `--baseline`): the view's `road_user` display objects plus the estimator's tracks. In this mode
   the estimator is built WITHOUT the `road_users` source, so tracks and the lead are what they were before
   road users existed.

Absent fields count as unknown. A track without `motion` counts as `stationary` when its `flags` contain
`stationary` or `parked`; otherwise it is `unknown`. `on_path` (or `in_path`) is read from `flags`.

Fair treatment of the baseline: its display objects carry relative velocity in a view frame with y to the
right. The harness converts it to ground velocity: `vx = vx_rel + v_ego - w*y`, `vy = vy_rel + w*x` (y left;
`v_ego` and `w` come from `ego_state`). Its motion is derived from ground speed: stationary below 1.0 m/s,
moving above 2.0 m/s, unknown in between.

## Scenarios (30-60 s each; seeded)

All of them run on a straight or gently curved road with the World's grade profile (±3 % over about 1.1 km).
Parked cars are 2.6-3.6 m from the lane centre. That keeps a car passable; the brief's range was 2-4 m.

| name | content | truth labels |
|---|---|---|
| `parked_street` | one-way street at 11 m/s; parked rows on both sides; 8 pedestrians on the pavements (5-6 m, 1.0-1.6 m/s); 1 pedestrian crossing 10 m in 8 s | parked / pedestrian / crossing_ped |
| `oncoming` | two-way road at 14 m/s; oncoming cars 3.5 m to the left every 4-9 s at 11-17 m/s (10 % trucks); parked row on the right; lead 90 m ahead at 15 m/s | oncoming / parked / lead |
| `junction` | approach at 12 m/s, stop 10 m before the crossing road, wait 14 s, go on at 8 m/s; crossing cars from both sides at 9-13 m/s every 2.5-6 s; 4 parked cars on the corners; 1 pedestrian | crossing / parked |
| `adjacent` | dual carriageway at 25 m/s; a car in the left lane at 24.6-25.6 m/s 15-25 m ahead the whole run; a truck in the right lane at 21 m/s being overtaken; lead at 25-26 m/s | adjacent / overtaken / lead |
| `slow_lead` | ego at 20 m/s closes on a 7 m/s lead (180 m ahead at the start) and follows it; a few parked cars on the verge | slow_lead / parked |
| `curve_parked` | the trap: R 110 m left curve, then right curve, at 13 m/s; parked rows on the OUTSIDE of each curve (2.6-3.2 m) plus a row beyond the oncoming lane; nothing in the path | parked |
| `stopped_brake_{14,25,33}` | stationary car in the ego lane, 200 m+ ahead at the start; the ego cruises, then brakes (up to 4.9 m/s², jerk 3 m/s³) to stop 6 m behind it | stopped_in_lane (in_path) |
| `stopped_lane_{14,25,33}` | same car; the ego keeps its speed and moves to the left lane (3.5 m over 2.5 s), starting at TTC 3 s; the road model's plan follows the old lane until the move starts | in_path until the lane change, parked after |

Truth for each actor at each frame, in the car frame: footprint centre, heading, nearest footprint point
(errors and gates use this point, which is what a box's bottom edge and the lead's distance refer to),
ground velocity (finite differences of world positions, so lateral motion counts), moving (more than
0.5 m/s), and for a standing actor `in_path` or `parked`. `in_path` means the actor's inner edge is
within 1.1 m of the centre of the lane the ego is in.

Camera lead: `sim.VisionSim` gets its lead from `ScenarioWorld.lead_at`. That is the nearest vehicle whose
centre is in the ego's lane, within `lead_range_m` (100 m by day, 70 m at night), reported with probability
1 up to 60 % of that range and falling linearly to 0 at it. Noise on top of VisionSim's own 1.2 %:

* a Gauss-Markov range-scale error, 5 % by day and 8 % at night, τ 3 s (the road model's own stated x_std is
  0.5 m + 6 %);
* white speed noise of 0.3 + 0.015 d m/s (×1.5 at night).

## Detector model (`DET_DAY` / `DET_NIGHT` in roaduser_sim.py)

Measured from the two recorded night drives of 2026-11-22:

* 3.4 Hz, as 5 Hz slots with 32 % skipped;
* the views alternate per message: view 0 roi [0, 0, 1280, 446], view 1 roi [160, 90, 1120, 410]. The brief
  said [0, 90, ...] for view 0; the recordings say 0;
* 0.37 boxes per message in session 1 and 0.23 in session 2;
* 38-58 % of the boxes cut at the bottom ("b"), with bottoms 0-2 px past the roi;
* box heights at least 22.4 px in the wide view and 12.8 px in the zoom view;
* scores p10/50/90 0.44/0.61/0.81;
* 400 car, 1 truck and 59 person boxes in session 1.

| parameter | day | night | source |
|---|---|---|---|
| detectable from / reliable from (box height px, wide / zoom) | 16 / 10 → 32 / 20 | 22 / 13 → 50 / 32 | night = recorded minima; day ASSUMED |
| steady miss share, mean miss run (messages) | 0.15, 2 | 0.45, 3 | brief |
| pedestrian detection factor | 0.9 | 0.6 | assumed (recordings: walking pedestrians missed) |
| off-axis (>12°) detection factor | 1.0 | 0.5 | eyeballed in the recordings: rows of unlit parked cars mostly missed |
| occlusion: box covered ≥ 60 % by nearer boxes is not reported | on | on | NMS / parked rows; `occl_cover=None` disables it (the brief said "occlusion ignored") |
| bottom edge noise | 3 px + 4 % h | 4.5 px + 6 % h | core/roadusers PROJ_CFG; night ×1.5 assumed |
| per-actor persistent bottom offset | 2 % h | 3 % h | assumed |
| other edges | 2 px + 3 % | 3 px + 4 % | assumed |
| gross box (beside the object by 0.3-1.0 widths) | 2 % | 5 % | recordings show boxes beside cars |
| car→truck, truck→car, person→rider | 3 %, 30 %, 2 % | same | recordings: trucks are nearly never called truck |
| false positives: transient / image-fixed 0.5-2 s / roadside post "person" 0.5-2 s (per min) | 1 / 0.1 / 0.3 | 3 / 0.5 / 1 | brief ("a few per minute"; the post called person) |
| calibration error: true pitch - calib pitch | ±0.4° | ±0.4° | brief (sign per seed) |
| yaw error | ±0.2° | ±0.2° | half the documented 0.3° (sign per seed) |
| body pitch | 0.15°/(m/s²) braking + 0.08° road roughness (τ 0.4 s) | same | assumed |
| road not flat | actor ground height from World's grade profile | same | geometry |

Boxes are the projection of each actor's 3D box (car 4.5×1.8×1.5 m, truck 8×2.5×3.4, van 6×2.1×2.6,
person 0.5×0.6×1.75, post 0.25×0.25×1.6) through the true camera. A crossing car's box is its side view.
The detector rejects a box if less than half its width or less than 40 % of its height is inside the roi.
Edges touching the roi get the clip letters l, t, r, b.

Known realism gap: the dense parked-row scenarios produce 3 to 7 boxes per message at night. The real city
drives averaged 0.37, so the simulator still sees far more parked cars than the real night detector.

## Metrics (per scenario × day/night; JSON has everything, the table a summary)

Matching is CLEAR-MOT style. A correspondence from the previous frame is kept while it stays inside the
gate; everything else goes through optimal assignment. The gate is measured on the line of sight to the
actor's nearest point: along max(3 m, 0.35 r) + half the actor's length, across max(1.5 m, 0.08 r) + half
its width. Only confirmed and coast outputs are matched; tentative ones are ignored.

**Coverage**

* `cov`: share of detectable truth time with a matched confirmed output. Detectable means some view could
  report the noise-free box (size and roi).
* `held`: the same with confirmed or coast.
* `ttfc50`: median time from the first emitted box to the first confirmed match.
* `sw/act`: id switches per actor.
* `seg`: matched segments per actor (gaps over 1 s).

**False tracks**

* `falseT/min`: outputs unmatched for at least half of their confirmed life (and at least 0.3 s), per
  minute.
* `fl50`: median confirmed life of a false track.
* Also unmatched confirmed track-seconds per minute.

**Errors per role**

* Along / across line of sight: bias, |p50|, |p90|, relative along error.
* Ground-velocity error magnitude: p50 / p90.

**Motion**

* Confusion matrix in seconds, truth {moving, stationary} × output {moving, stationary, unknown}, per
  scenario and per role.
* Time from the first box to a correct verdict that then holds for 2 s.

**Parked question**

* Share of parked-car matched time marked stationary (motion, or a parked/stationary flag).
* Share of MOVING actors' matched time marked stationary or parked. This must be near zero.

**Red alert** (stopped_*). The first frame at which either:

* a confirmed output matched to the stopped car carries on_path and motion stationary, or
* the lead is present on the car with |v_lead| ≤ max(1.5, 0.15 v_ego) and v_ego > 3 m/s.

Reported: true distance, true TTC (distance / ego speed), how many seeds hit, how many came before the
driver acted (brake start or lane-change start), the stack's own `ttc_s` against the truth, the first box,
the first confirmed match, and where the driver acted. `--lead-off` repeats this without the camera lead.

**False alarms**

* Seconds and episodes of confirmed on-path stationary outputs not matched to an in-path standing actor,
  with ego > 3 m/s.
* Seconds and episodes of a lead stationary by the rule above that is not on an in-path standing actor.
* In-path verdicts on the stopped car after the lane change is over, listed separately.

**Cost**

* Wall time of all `stack.deliver()` calls divided by road frames, with and without road users
  (parked_street, night, seed 1).

## Replay tool

Inputs are fed like the daemon's stack. The estimator uses its default sources, which are radar and
road_users. There is no map box; the recorded `map` goes through the external-map channel. Recorded
outputs (tracks, ego_state, view, lead, fused, rules, speed_limit, radar) are not fed. They are only
compared against.

`--calib-override pitch,yaw` replaces every `calib` message's pitch and yaw. Session 20261122-202233 had a
wrong calibration (+0.0561) until 20:28; use -0.0140,-0.0624.

`--mode radar-alone` runs `chris_integrator.radar.replay.run(session, calib_override)` when that module
exists, and says so when it does not.

Each tile is one recorded frame (one per second; a frame's mtime is its `ts_real_ns`) with at least one
detection or object. Detector boxes are drawn thin. Objects are drawn as a ring at their ground point
through the calibration in use, labelled `#id class motion x y v`. Colours: green moving, blue stationary
off the path, red stationary on/in the path, yellow unknown. A sheet holds 15 tiles of 640×360 at JPEG
quality 80.

Printed per session:

* confirmed objects, their lives and the motion verdicts (object-frames), for the new code and for the
  recorded view's road_user objects;
* lead presence share.
