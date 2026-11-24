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

## Detection inputs (lane detin): the detector's optional outputs

The detector (roadsense-signs `signd`, branch `detin`) can publish more per box: frame-edge and occlusion
flags, a low-score list, an appearance vector, the lamp pair at night, and lamp-based scale rates. The contract
is in roadsense `docs/ARCHITECTURE.md` (the `lights` message), the consumer's design in roadsense
`docs/lanes/detin-consumption-design.md`. `DetectorSim` emits all of them. Each one is a parameter of `DET_DAY`
/ `DET_NIGHT` and is off by default. **With every option off the messages and `det.log` are bit-identical to the
simulator without them** (`tests/test_sim_detin.py` holds seven runs to digests recorded before the change).

How to switch them on:

* `DetectorSim(seed, night, params=dict(rs.DETIN))` turns everything on at the recommended values (the
  detector's documented set: floor 0.10, flags, `yuv3x3`, lamps). `rs.DETIN_REID` is the same with
  `reid32+yuv3x3`. `scale_corr` only acts together with a `scale_rate` dict (`SCALE_DAY` / `SCALE_NIGHT` in
  `tools/visual_radar_eval.py`).
* The environment variable `CHRIS_SIM_DET` holds a JSON object, or `detin` / `detin_reid`; `{"detin": true, ...}`
  starts from the preset and then overrides. It is merged into the parameters after the day/night dict and
  before `params`, so explicit parameters win. Whole-stack tools that build `DetectorSim(seed, night)`
  themselves pick it up without code changes.
* `python3 -m chris_integrator.sim.radar_score NAME --det detin` (or `--det '{...}'`).

**Paired runs.** The base detector runs draw for draw on its own generator (`random.Random(seed)`), on the full
truth boxes and the old occlusion rule, whatever the options. Every new thing comes from a second generator
(`seed * 31 + 7`; roadside occluders `seed * 31 + 9`). That covers the low list, flags, vectors, lamps, scale
chains, and the draws for an actor that the base dropped but the new rules report. So the miss chains, the false
positives and the boxes of actors the new rules leave alone are the same with the options on. With `low`,
`flags`, `emb` or `lamps` alone, the first list is the base one box for box (only keys are added).

Outcomes do differ where `visible_boxes` has to change them:

* a covered actor gets a smaller box, a lower score mean and a lower detection probability;
* cover is the union of everything nearer, hidden actors included. The old rule summed the overlaps of
  reported-size boxes, so a different set of actors passes `occl_cover`;
* an actor the base dropped (old rule, or the region test on the full box) but whose visible part passes gets
  second-stream draws;
* such an actor's persistent bottom offset is a second-stream draw until the base draws its own;
* the base scale value is dropped from boxes covered more than `scale_occ_max`.

With `scale_corr` every scale value is the new model's; the base draws are still consumed.

### What each option models (M = measured in the documents above, A = assumed)

| option | what is emitted | numbers |
|---|---|---|
| `visible_boxes` (+ `roadside`) | The box of the visible part of the actor's silhouette. The silhouette is a body cuboid plus an inset cabin for cars (A) and one cuboid for vans and trucks; it is rasterised on a 24×24 grid over the truth box and cut by the silhouettes of everything nearer (true range) and by the 1280×720 frame. Covered ≥ `occl_cover` (0.6): not reported, as before. Between 0.15 and 0.6 the detection probability falls linearly to ×0.5 (A). Edge noise, `clip` and the region test then apply to the visible box. `ROADSIDE`: posts 25, tree trunks 15, kerbside railings/hedges 6 per km and side, 4.0-5.2 m from the lane centre (A). | A; "no flag ≠ whole" comes from occluders not boxed in the pass, from roadside occluders and from the silhouettes |
| `flags` | `trunc` on every box; `occ`, `occ_by`, `occ_edges` only when covered. Computed from the EMITTED boxes by signd `detin.cc`'s rule, ported 1:1 (`ru_nearer`, `flag_one` on a 16×16 grid, `trunc_of`; ties count both ways). Low boxes are flagged against the first list. | M (rule) |
| `low` (floor) | `road_users_low`. A visible actor missed by the first list is in it with p 0.68 (day) / 0.75 (night) inside a miss run (M), 0.4 when missed by the size/off-axis/cover draw (M, BDD100K), 0.3 when covered 0.6-0.85 (A), and 0.5 × a ramp between 0.5 and 0.8 × h_min below the first list's size limit (A). Score uniform in (floor, class threshold] (signd thresholds 0.35-0.50). Edges 1.5 × the px terms + 10 % (M). Class wrong 5 % (M). Clutter on nothing: 0.65 a pass by day, 0.35 at night; 30 % of it recurs at one image place for 1-2 s (rates M × the BDD share of 0.57 of low boxes on no label; recurrence A). Beside or on part of a first-list actor box: 0.15 / 0.2 per box (A within the rides' 0.04-0.45). Then signd's NMS 0.5 against the first list and better low boxes, and the cap of 20. | as marked |
| `emb` | `emb` (hex of int8) on boxes ≥ 8 px a side in both lists, and `emb_kind`. Per key (actor, persistent false box, roadside item, clutter) there is a latent and an Ornstein-Uhlenbeck drift that grows with the time since the key's last observation, plus white noise per observation. A per-object volatility and a per-observation scale are both log-normal; by day a share of "frozen" objects barely changes (the table's near-zero same-object pairs). The nine Y cells carry the scene level, 0 by day and -1.5 at night (A); it does not change distances. A covered box is blended towards its main occluder's look in proportion to its covered share (A). `reid32+yuv3x3`: the 32 learnt numbers around a common direction, L2-normalised, sharing the scale draws; at most 16 vectors a pass. | fitted to `detin-emb-distances.json` (M), table below |
| `lamps` | Night only, vehicles whose rear or front is within 40° of the line of sight (A). `lit` (moving or standing in traffic) p 0.65 (M 0.61-0.73); parked (unlit) p 0.05, colour red/unknown (A). The two lamp points are projected: S = 0.67 W e^N(0,0.12) apart (A), at 0.6 H (A). Noise hypot(0.35 px, 2 %), 0.8 px under 10 px (M). Wrong blobs 10 %, sep × 0.6-0.9 (M). Colour unknown 13 %, wrong 3 % (M). `sep_std` is the detector's hypot(0.35, 2 %) whatever the size. The detector's own checks apply: centroids inside the box, 0.3-0.98 of its width apart, about its middle unless a side is cut. | as marked |
| `scale_corr` | Per actor and pass a noisy log size: a chain of increments, each with that pass's own alignment error, its variance in proportion to the step so that a 0.6 s baseline has the base model's std (A). The value is this pass against the oldest sample of the window (the patch alignment), so successive values share the increments of their overlap. The stated std is the value's own 1-sigma. At night with `lamps`: the WLS line through ln sep over ≥ 3 passes in the window (≥ 10 px, one colour, no step above 0.25), std × 1.5 (M), 10 % of the samples off by 3 sigma (A, the heavier tail). Combined with the patch value as signd does: `scale_src` "both" / "lamps". With `low`, low boxes continue the chains and carry values. | as marked |

Truth for scoring, per message (only with an option on):

* `det.log_low`: (t, view, [(actor id | `fp:clutter` | `fp:clutter_fixed` | `fp:beside:<id>`, cls, box, clip)]).
* `det.log_x`: {t, view, seen (actors whose visible part is in the view's region), first / low: per box {id,
  cover, cover_box, hidden (truth edges covered), out (truth edges beyond the frame), by (main occluder: actor id
  or `rs:...`), h (truth box height), scale_true (the noise-free value of the emitted scale rate), sep_true (the
  true lamp separation), why (low: run / draw / small / occl / clutter / clutter_fixed / beside)}}.

**World option `dims_spread`** (`scenario(name, seed, night, world={"dims_spread": True})`, or the `"world"` key of
`CHRIS_SIM_DET`; part of `DETIN`; `radar_score.run` passes `det_params["world"]` to `scenario()`). Each actor's (L,
W, H) is multiplied by exp(N(0, sigma)), drawn from a generator of its own seeded from the scenario seed and the
actor id. Cars use sigma 0.08 / 0.04 / 0.06, vans and trucks half that, pedestrians and riders 0.05 on the height
(A: the European fleet, small cars to SUVs). Explicit `dims=` are spread the same way. Off: the worlds are
unchanged. The lamp separation follows the spread width.

**World option `dims_fleet`** (`world={"dims_fleet": True}`, the same ways): every car of the nominal size takes a
size of `DIMS_FLEET` (small hatchbacks, compacts, crossovers, car-derived vans, large SUVs: 1.42-1.90 m tall, mean
1.57 m, 1.63-2.0 m wide, at most 4.8 m long; a generator of its own per actor); a scenario's explicit size stays;
everything else as `dims_spread`. **The detector's scale rate as a preset**: `scale_rate: "auto"` (or true) is
`SCALE_DAY` or `SCALE_NIGHT` by the run; `det_params_arg("scale")` is today's stream with it, `"detin_scale"` and
`"detin_pose_scale"` the presets with it (the presets themselves emit no scale rate; the car's unit does:
docs/RADAR.md, section 11).

### Appearance fit (`EMB_MODEL`)

Quantiles 5 / 25 / 50 / 75 / 95 % of the contract's distance, simulated with `EmbGen` against the table
(`detin-emb-distances.json`; "other" = the mean of its three other-object rows):

| case | simulated | table |
|---|---|---|
| day yuv3x3, same 1 s | 0.06 0.30 0.56 0.92 1.61 | 0.07 0.33 0.54 0.87 1.65 |
| day yuv3x3, same 2 s | 0.07 0.33 0.63 1.02 1.77 | 0.08 0.41 0.66 1.02 1.83 |
| day yuv3x3, same 4 s | 0.08 0.39 0.74 1.19 2.04 | 0.08 0.50 0.80 1.20 2.00 |
| day yuv3x3, other | 1.13 1.76 2.42 3.31 4.87 | 1.18 1.85 2.56 3.49 4.80 |
| day reid32+yuv3x3, same 1 s | 0.05 0.33 0.49 0.71 1.12 | 0.06 0.32 0.47 0.66 1.13 |
| day reid32+yuv3x3, same 2 s | 0.05 0.36 0.53 0.75 1.16 | 0.06 0.39 0.56 0.77 1.26 |
| day reid32+yuv3x3, same 4 s | 0.05 0.40 0.59 0.81 1.23 | 0.06 0.45 0.66 0.89 1.40 |
| day reid32+yuv3x3, other | 0.87 1.27 1.58 1.94 2.76 | 1.03 1.45 1.81 2.23 2.82 |
| night yuv3x3, same 1 s | 0.36 0.63 0.88 1.26 2.78 | 0.41 0.59 0.82 1.33 2.57 |
| night yuv3x3, same 2 s | 0.45 0.76 1.05 1.50 2.90 | 0.48 0.68 0.98 1.53 3.52 |
| night yuv3x3, same 4 s | 0.51 0.88 1.20 1.70 3.22 | 0.58 0.88 1.14 1.72 3.37 |
| night yuv3x3, other | 0.77 1.31 1.90 3.03 6.92 | 0.78 1.29 1.93 3.38 6.80 |
| night reid32+yuv3x3, same 1 s | 0.30 0.46 0.61 0.90 2.17 | 0.34 0.46 0.61 0.93 1.56 |
| night reid32+yuv3x3, same 2 s | 0.37 0.56 0.73 1.02 2.23 | 0.37 0.51 0.70 1.03 2.06 |
| night reid32+yuv3x3, same 4 s | 0.43 0.63 0.82 1.14 2.28 | 0.43 0.60 0.79 1.12 2.20 |
| night reid32+yuv3x3, other | 0.63 0.95 1.27 1.83 3.79 | 0.68 0.93 1.25 2.00 3.60 |

The worst spots: by day the learnt kind's other-object distances are 10-15 % short, and at night the learnt
kind's same-object 95 % point at 1 s is 2.2 against 1.6. The fit was made by Nelder-Mead on log-quantile errors
with a vectorised copy of the generator (scratch, not in the repo); `tests/test_sim_detin.py` checks the
simulator's own generator against the table. Caveat from the source: the tables come from full-frame crops at
10 fps; the unit's own views show larger distances.

### Emitted streams with `DETIN` on

`SCALE_DAY` / `SCALE_NIGHT` scale rates, seeds 1-3 per row; shares are of first-list boxes. Columns:

* truly covered: truth cover > 0;
* covered, no flag: of the truly covered, those without `occ`. M: 26-28 % on BDD100K, where every vehicle in
  front is a box and the camera sees the whole frame;
* no flag, covered: of the unflagged, those truly covered. M: 40 %;
* gap passes: as the rides define them (boxed before and after within 1.5 s and seen in the pass's region).
  M: 68-100 %.

| scenario | | first / msg | low / msg | occ > 0 | with occ_edges | truly covered | covered, no flag | no flag, covered | truth edge hidden or cut | lamps on vehicle boxes | scale coverage (lamps / both) | gap passes with a low box |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| parked_street | day | 5.08 | 4.98 | 0.33 | 0.32 | 0.62 | 0.57 | 0.53 | 0.76 | - | 0.39 (0 / 0) | 381/771 |
| parked_street | night | 2.39 | 6.10 | 0.17 | 0.17 | 0.65 | 0.78 | 0.61 | 0.73 | 0.01 | 0.32 (0 / 0) | 356/612 |
| curve_parked | day | 5.57 | 4.98 | 0.38 | 0.37 | 0.72 | 0.53 | 0.62 | 0.76 | - | 0.50 (0 / 0) | 445/1006 |
| curve_parked | night | 2.20 | 6.72 | 0.22 | 0.21 | 0.71 | 0.73 | 0.67 | 0.75 | 0.01 | 0.41 (0 / 0) | 321/513 |
| platoon | day | 1.79 | 1.62 | 0.30 | 0.29 | 0.47 | 0.38 | 0.26 | 0.47 | - | 0.72 (0 / 0) | 106/192 |
| platoon | night | 0.97 | 1.82 | 0.15 | 0.14 | 0.41 | 0.64 | 0.30 | 0.41 | 0.52 | 0.60 (16 / 15) | 143/218 |
| occlusion | day | 1.47 | 1.23 | 0.17 | 0.16 | 0.20 | 0.44 | 0.11 | 0.24 | - | 0.49 (0 / 0) | 77/160 |
| occlusion | night | 0.66 | 1.32 | 0.09 | 0.08 | 0.25 | 0.78 | 0.22 | 0.27 | 0.43 | 0.42 (10 / 15) | 87/147 |
| queue | day | 2.02 | 1.73 | 0.15 | 0.15 | 0.32 | 0.62 | 0.23 | 0.27 | - | 0.54 (0 / 0) | 110/194 |
| queue | night | 0.88 | 2.08 | 0.11 | 0.11 | 0.23 | 0.63 | 0.17 | 0.21 | 0.55 | 0.40 (26 / 9) | 126/190 |
| oncoming | day | 2.66 | 2.54 | 0.26 | 0.26 | 0.48 | 0.54 | 0.34 | 0.57 | - | 0.45 (0 / 0) | 205/429 |
| oncoming | night | 1.22 | 3.04 | 0.13 | 0.13 | 0.49 | 0.76 | 0.43 | 0.56 | 0.10 | 0.31 (1 / 0) | 151/264 |

Reading it:

* The low list is far busier than on the rides (0.4-1.2 a pass). This is the simulator's known excess of
  visible actors: 2-6 first-list boxes a pass against 0.2-0.5. Clutter alone is at the documented rate.
* Covered boxes go unflagged more often than on BDD100K, by night most of all: the occluder is often missed in
  that pass (night p_miss 0.45, off-axis 0.5).
* The gap share with a low box is 0.43-0.66, below the rides' 0.68-1.0. Here a gap also counts passes in which
  the actor is lost to the size / off-axis draw (low p 0.4) or covered beyond 0.6 (low p 0.3).
* Lamps on parked vehicles are rare by design: unlit, p 0.05.

### Ground: the scenarios' vertical curvature

All scenarios use World's grade profile `grade_amp * sin(s / 180 m)` with `grade_amp` 0.03 (no scenario sets
it). Its vertical curvature is `grade_amp / 180 * cos(s / 180)`. Over the driven arcs plus 150 m ahead, |k_v|
has p50 1.2e-4 /m, p90 1.6e-4 /m and max 1.67e-4 /m (R 6000 m); no point exceeds the consumer's assumed
1/3000 m. Stronger crests and sags: the spec key `grade_amp` is honoured by `ScenarioWorld`.

    spec = rs.SCENARIOS["platoon"](seed, night)
    spec["grade_amp"] = 0.06
    w = rs.ScenarioWorld(spec, seed=seed, name="platoon", night=night)

That gives k_v up to 1/3000 m, but with grades of ±6 % (0.09: 1/2000 m at ±9 %). The 1.13 km wavelength is fixed
in `world.py`, so realistic grades with tighter crests would need a shorter wavelength there.

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
