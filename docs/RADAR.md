# The radar's estimation core

What `chris_integrator/radar` does with a scan, and why it is built this way. The component's design (the car's
own motion, sensors as models, objects with a memory) is in `chris_integrator/radar/__init__.py`; the message is
in `docs/TOPICS.md`; how the arithmetic is held to recorded results is in `docs/NATIVE.md`. All arithmetic named
here is in the native core (`chris_integrator/native/src/chris_native.c`); the Python is the objects' life and
the glue.

## 1. Which detection is whose: hypotheses over several scans

### The problem
A scan's detections have to be given to objects. Where one object is near one detection there is nothing to
decide. Where two objects could have given a detection, where a detection could be a known object or a new one,
or where a box beside a parked car could be that car or its neighbour, one scan cannot tell. The tracker before
this one decided at once: a soft blend (each object took the detection as far as the pairing was probable) and
one assignment per scan. A blend drags an object towards a detection that was not its own, and never takes that
back.

### What is built: track-oriented multiple-hypothesis tracking in its Bernoulli form
* Every object keeps a **tree of hypotheses**. A node is the object's numbers after one scan under one story
  about that scan: it took detection j, or it was not reported. The leaves are the object's present hypotheses.
* Objects that compete for detections (now, or within the scans still open) form a **cluster**. A cluster holds
  **global hypotheses**: one leaf of every object of the cluster (or none: in that story the object does not
  exist, its first detection was another object's), no detection used twice, with a probability.
* **A scan** (`Hyp.cycle`): every leaf is stepped to the scan's time; each leaf is compared with each detection
  (gate, then the likelihood ratio against "something else"); objects that share a gated detection are joined
  into one cluster; for every global hypothesis of a cluster the K best assignments are enumerated in order
  (Murty's ranked assignment; a hypothesis is given a share of K by its weight) and the best of all kept; a
  detection that a kept hypothesis leaves over starts an object in that hypothesis.
* **N-scan pruning**: what the best global hypothesis says about each object N scans back is taken as settled;
  the branches that disagree go, with the global hypotheses built on them. So a doubtful pairing stays open for
  N scans and is decided by what those scans show.
* **What is put out**: of each object the leaf in its cluster's best hypothesis (`primary`): the filtered state
  of one consistent story, not a mixture. The probability that the object exists is summed over all global
  hypotheses of the cluster. The object's id is its tree: when the best hypothesis changes its mind about a
  pairing two scans ago, the object's numbers change (to those of the other branch), its id does not.
* An object no hypothesis doubts (one leaf, in every hypothesis) leaves its cluster for one of its own; clusters
  are joined again when their objects compete. In an unambiguous scene every cluster is one object with one
  hypothesis, and the scan costs what a single assignment costs.

### The weights
For a leaf with existence probability r (the logistic of its log odds L), probability of being reported pd
(coverage, size, later: occlusion) and likelihood l(z) of detection z (over its motion hypotheses, with the box
feature F), against the density lam(z) of "something else" at z (the sensor's clutter and new objects, and the
map of where unexplained detections that became objects have been turning up):

    it took z:        weight r pd l(z) F / lam(z)        L += the fit's evidence, bounded; + the detector's score
    not reported:     weight 1 - r pd                    L += ln(1 - pd)
    z is left over:   weight 1 (lam / lam)               a new object, existence `new_exist` (+ the score)

A global hypothesis' weight is the product over its objects and detections. These are the Bernoulli update of a
Poisson multi-Bernoulli mixture (PMBM) filter written per track; with L as the track score they are also the
classical track-oriented MHT (the score of a track is its likelihood ratio against "all its detections were
false"). Two things differ from the textbook, on purpose:

1. Within a branch that took a detection, existence does not jump to 1. The evidence of one detection is bounded
   (as before: between ln 1.5 and ln 20, plus the learnt score term), because the camera's false boxes are not
   Poisson clutter: they recur at one place for a second or two. The doubt "this box was false" is therefore
   carried twice while a sibling branch lives (inside the branch, and as the sibling); that errs towards later
   confirmation, never earlier.
2. Survival is not a model term. An object dies by its misses (L falls) and by the upkeep's rules (unseen for too
   long inside a field of view, remembered outside one while its place is worth knowing).

### Why this formulation
The candidates were (a) classical track-oriented MHT with N-scan pruning and Murty, (b) a PMBM filter, (c) a
labelled multi-Bernoulli (delta-GLMB / LMB) filter.

* **Identity.** The output needs stable ids. GLMB carries labels by construction; PMBM is unlabelled, but its
  Bernoulli components are born from one detection and followed over time, which is a track in everything but
  name. Track trees give the id for free.
* **Existence.** MHT's track score and the Bernoulli's existence probability are the same quantity under the
  weights above; the PMBM view adds the clean treatment of "a detection is a new object or clutter" (the
  Poisson part: `lam`) and of an object that exists in some global hypotheses only.
* **Deciding late.** All three keep several global hypotheses; what differs is what is cut. GLMB truncates by
  weight every scan (and its LMB approximation collapses to marginals every scan: the blend this work is meant
  to get rid of). N-scan pruning is the explicit statement "this pairing is decided by the next N scans", with
  N a knob.
* **Sensors of different kinds, several detections per object.** The per-leaf likelihood is the only place a
  sensor enters: camera boxes (bearing good, range poor), a microwave radar (range, azimuth, range rate), a
  position sensor. A sensor that returns several detections per object enters as cells (a group of returns as
  one detection with its count and scatter: section 4), which keeps "one detection per object per scan" true
  for the assignment.
* **Cost on a phone core.** Clusters keep the enumeration local: the number of global hypotheses is bounded per
  cluster, not for the scene. LMB would be cheaper per scan and worse exactly where it matters.

So: the Bernoulli weights of PMBM, the track-oriented structure and N-scan pruning of MHT, K-best enumeration
by Murty per cluster.

### The knobs (`tracker.CFG`, under `radar.tracker` in the configuration)

| knob | default | what it buys, what it costs |
|---|---|---|
| `hyp_n_scan` | 5 | scans a pairing stays open. 0: decided at once (the best assignment of each scan) |
| `hyp_k_best` | 32 | global hypotheses kept per cluster (and assignments enumerated per scan) |
| `hyp_max_leaves` | 12 | hypotheses per object |
| `hyp_prune_ln` | ln 1e4 | a hypothesis this much less probable than the best is dropped |
| `hyp_max_cluster` | 16 | a larger cluster gets the best assignment per hypothesis only |
| `hyp_keep_steps`, `hyp_keep_s` | 40, 2.0 s | how far back each story's nodes are kept (the smoother's reach) |
| `merge_d2` | 9.2 | two objects are merged as duplicates only within what both know of their places |

`hyp_n_scan=0, hyp_k_best=1` is a single-hypothesis tracker with hard assignment (not the blend of before:
that is gone).

### What it shows
* `tests/test_radar.py::test_a_doubtful_pairing_is_decided_by_the_scans_that_follow`: one detection between two
  objects, read as A's at its scan, revised to B's four scans on; A is not dragged. With `hyp_n_scan=0` the
  first reading stands and A is.
* `...::test_two_riders_swapping_sides_keep_their_ids_more_often_with_hypotheses_over_scans`: two riders swapping
  sides in a microwave radar with 0.8 m of azimuth noise, misses and clutter: of 200 identities, 182 kept with
  the defaults against 163 decided scan by scan (100 seeds).
* The duplicate test: two cars side by side in a radar's view were merged into one object by the old rule (nearer
  than 2.5 m + 12 % of the range); with `merge_d2` they are two.

### Not built
* Multi-scan (Lagrangian / S-D) assignment over the window: the enumeration is per scan, the window is in the
  trees. For clusters of the size met here Murty per scan with N-scan pruning is the standard, and exact up to
  K.
* Hypothesis merging (joining leaves of one object whose recent histories agree) instead of pruning them. The
  bounded evidence above is what keeps existence honest without it.
* Out-of-sequence scans beyond `oosm_max_s`: a late scan is applied to the leaves taken back to its time (a
  negative step), as before; an object too far ahead of the scan does not take part.

## 2. The state: an object's centre, and what each sensor sees of it

The state of every motion hypothesis is `[px, py, vx, vy, ax, ay, dp, lh]` in the ground frame
(`radar/models.py`); since native version 8 **the place is the object's centre**. Before, it was "where the box
stands on the road", the point of the object nearest the camera: not a point of the object at all (it slides
from the rear bumper to the side as a parked car is passed, two metres of apparent motion of a thing that
stands), and not the point a radar's returns average to. Each sensor's model now says what it sees of a
rectangle (section 4) at that centre:

* a camera's box stands on the road at the **near face**: the ground angle, the box height, the top edge and the
  scale rate are predicted for the range of the centre less half the object's depth along the line of sight
  (`r_h`, `r_half_depth`); what is not known of that depth (the extent's own covariance) is added to those rows'
  variances (`r_reff`). The bearing is the centre's.
* a microwave radar's returns lie on the sides that face it; their mean is nearer than the centre by what the
  rectangle says (`r_scatter`, `r_cell_centre`), also for a single return.
* a position sensor gives the centre.

The message keeps `x_m`, `y_m` (and `predict`) as the near face, as they were (what is in the way is the face,
not the centre: `on_path`, `in_path`, `ttc_s` use it); the centre goes out as `ctr`. The near face is where the
line of sight from the car's origin to the centre enters the rectangle (`r_near_offset`): on the rectangle,
continuous, for a thing ahead its rear face at about the centre's lateral place. Until 2026-11-23 it was the centre
brought nearer along the line of sight by half the depth (`r_half_depth`, the camera model's range): a point off the
rectangle, towards the car, by up to the half width when the object is beside. (The rectangle's nearest point, the
simulator's `ref_x`, `ref_y`, was tried: it puts a car in the next lane ahead at its inner corner, 0.9 m nearer the
car's lane than its centre, and the prediction, which reads `x_m`, `y_m` as the object's place in its lane, called
the adjacent car a cut-in with 0.97.) A car parked 2.7 m to
the side read 0.4 to 0.8 m nearer the path than its side as it was passed, and a long one beside the car was put at
the origin itself (the half depth clamped to the range): the decision layer's `obstacle_ahead` warned on parked
cars at "contact" (parked_street: near-face lateral error under 8 m 0.35 m median, inward). The decision layer
draws the footprint from `ctr` less `x_m`, `y_m` (native/src/decide.c `df_parse`: the centre's offset), not from the line of sight.

**A floored hypothesis is re-seeded.** The motion hypotheses are weighed by how well each predicted a detection
and none is ever ruled out (`mu_floor`). A hypothesis held at the floor used to keep its own old story: "it
stands" of a car that has driven on for four seconds stood fifty metres back, and its share of the combination
pulled the object's place and blew up its covariance (found on the hidden-car test). What the floor keeps open is
that the object does so *from where it is now*: the share the floor adds is given the other hypotheses' state (for
"it stands": their place, no speed) (`r_update_full`).

**A moving story does not learn the object's height and angle offset from a camera (2026-11-24).** A camera
gives a moving object's range only through two unknowns of the object's own: its height (`lh`, read through the box
height) and its ground-angle offset (`dp`, read through the bottom edge). Their priors are the only absolute
information: the boxes relate the two to the range, and while the object drives its range is a free path between the
boxes (the motion hypotheses' process noise), so the boxes cannot tell them apart from it. An extended Kalman filter
that learns them there anyway is a biased estimator of them (the gain's own dependence on the parameters is
ignored: Ljung 1979), and the bias went one way: a car driving with the traffic at 20-30 m was put 4-10 m too far
with a stated spread of 2.5-3 m. Shown on the simulator with every error source of its own switched off (no pitch
error, no road grade, no bottom-edge bias, cars of exactly the prior's 1.6 m, the pitch bias held at zero): the
boxes by themselves were right (ground range +1 to +3 %, box-height range 0 to +2 %, residuals at the truth
-0.16..+0.1 sigma), yet the tracker's centre was +3.1 m off on average (adjacent and platoon, 10-40 m), and a plain
single-model EKF over the same detections, started AT the truth, drifted the same way (height 1.60 -> 1.83 m, offset
0 -> +0.3 deg, range +2.6 m). The drift grows with the velocity process noise (cv_noise 0.05 / 0.35 / 2: -1.7 / +5.7
/ +9.6 m) and is not a matter of linearisation (iterated EKF: -8 m, the other way; unscented: +6 m). With height and
offset as consider parameters (Schmidt: their variance in the covariance and in every innovation, their values
held) the same EKF was right to +0.03..+0.9 m with 1 / 2 sigma coverage 0.85-0.98 / 0.93-1.0. The drifting offsets
also taught the camera's pitch bias the wrong sign (learnt +0.3..+0.4 deg against a true -0.33 deg) through the
objects that "ripened" on it.

Change: `Tracker._consider` (per object, `info` column 10 of the scan): the moving hypotheses take a camera's
detection with the gain on `lh` and `dp` held at zero (Joseph form: the covariance stays the filter's exact one for
that gain; native `r_take_parts`, `cons`) from the sighting on at which the hypotheses are weighed
(`mu_hold_hits`), unless a sensor that measures range (a microwave radar, a position) has seen the object within
`consider_range_s` (2 s): then the range pins them and they are learnt. The standing hypothesis learns them as
before (its rigid motion lets the car's own passing measure the range: parallax), and so does the free estimate.

Measured (tools/radar_trace.py, seeds 1-3 day + 1 night, confirmed, moving cars of 10-40 m, the centre against the
truth's centre; before -> after): along error mean / p50 / p90, share within 1 / 2 sigma:

| | before | after |
|---|---|---|
| platoon | +6.7 / 6.4 / 13.5 m, 0.33 / 0.65 | +2.0 / 2.7 / 6.4 m, 0.86 / 1.00 |
| adjacent | +3.6 / 3.4 / 9.4 m, 0.35 / 0.61 | +2.4 / 2.9 / 6.0 m, 0.58 / 0.95 |
| crossing_pair | +6.7 / 7.2 / 18.1 m, 0.44 / 0.78 | +4.0 / 3.9 / 12.0 m, 0.84 / 0.99 |
| queue | +0.6 / 1.7 / 6.5 m, 0.57 / 0.86 | -0.2 / 1.5 / 4.3 m, 0.74 / 0.94 |

What remains of the cars' bias is honest: the car prior's 1.6 m against the simulator's 1.5 m (+7 %: KITTI's cars
are 1.53 +- 0.14 m, today's fleet with its SUVs and the vans a detector calls car is taller; the prior was left) and
the camera's pitch where nothing passed teaches it (seeds 2-3: +0.4 deg not learnt, read far), both inside the stated
spread. What it costs: a vehicle whose range changes a lot no longer refines its height from its own boxes: the
overtaken truck of `adjacent` (3.4 m, prior from the class vote 2.6 m) at 40-80 m along p50 11 -> 16 m (within
2 sigma 0.58 -> 0.80), at 80-150 m 28 -> 32 m (0.47 -> 0.66). The variant with the standing hypothesis held too
(`consider_stand`) cures `slow_lead` (a slow car ahead read as a standing thing 1.5 times as far and as tall: the same
scale ambiguity; moving 10-40 m mean +4.4 -> +1.4 m, p90 11.3 -> 4.7 m) but takes the parallax-pinned learning from
parked cars (three behaviour tests fail: a parked car passed with a biased range, the row, the pitching camera); not
taken. A moving vehicle whose range changes by more than a factor (1.5, 2.5) learning again was measured and
dropped (the drift comes back first).

## 3. What a sensor can see: occlusion, the edge of the field

An object that is not reported loses existence by `ln(1 - pd)` per scan. `pd` is no longer a constant inside
the field: it is the sensor's own probability (range, box size, a box cut at the bottom) times **how much of
the object the sensor can see** (`r_visible`, record `M_VIS`):

* **Behind another object.** The settled objects' primaries are occluders: an interval of bearings (the
  rectangle's silhouette from the sensor), a range, and for a camera the rows of the picture its box covers (from
  its height state). A leaf's own silhouette is sampled at five bearings; a sample is hidden as far as a nearer
  occluder's interval holds it (softly: neither bearing is known exactly, but the softness is capped at a third
  of the occluder's half width, because how far a hidden thing may have strayed while nobody saw it does not open
  the view) and, for a camera, as far as the occluder's box reaches up the leaf's box. The chance of a report
  falls linearly from `occl_from` (a quarter hidden) to its lowest, `occl_floor`, at `occl_full` (0.65 hidden;
  the detector drops a box covered by 0.6). A microwave radar sees under and past vehicles: its floor is
  `occl_floor_radar` 0.3.
* **Cut by the edge of the field.** The part of the silhouette inside the field: all of it, no effect; under 0.4,
  the floor (the detector rejects a box with less than half its width in the picture).
* **Outside the field**: `pd = 0`, as before.

What follows from it, with no further rule: a hidden object's existence does not fall; it is carried on by its
motion hypotheses, its covariance grows, and its gate grows with it, so the detection that comes when it
reappears is its own and it keeps its id. The upkeep counts the time an object was hidden as time in which
nothing could be seen (state `coast`), and drops a hidden object only when its place is no longer worth knowing
(`hidden_s` 12 s, `hidden_std_m` 30 m). The message says `vis` (0..1), `occ` (`"object"`, `"edge"`, `"fov"` or
null) and `occ_by` (the occluder's id).

`occlusion: false` (tracker configuration) switches the model off. Shown by `tests/test_radar.py::Seeing`: a car
hidden behind a van for four seconds is one object throughout, with growing range uncertainty, confirmed again
half a second after it reappears; without the model it is dropped while hidden and comes back as a new object.

Not modelled: occluders that are not tracked objects (buildings, the road's crest), and the conditioning of a
hidden object's place on the fact that it is hidden (its place density is not cut to the shadow).

## 4. Extent and orientation

Every leaf carries `e = [length, width, heading]` with a 3 x 3 covariance (`RO_E`, `RO_PE`), a Gaussian of its own
beside the motion state (the two are taken as independent, as the random-matrix trackers factorise them). The
shape is a **rectangle**: vehicles are rectangles, a camera box and a radar's returns both see a rectangle's
sides, and a rectangle's parameters are what the consumers need (`len`, `wid`, `hdg`). A random-matrix ellipse
was the alternative; it has a cheaper update, but no notion of "the side that faces the sensor", which is the
whole of what both sensors measure.

* **Birth** (`r_extent_birth`): the kind's size (a car 4.5 x 1.8 m, 22 % and 10 %; a person 0.6 m; not known:
  as a car, wide open); the heading along the car's own for a vehicle, across it when the box shows it side-on,
  as sure as the box's aspect says which; no direction for a person.
* **Height** (state `lh`, the box height's range): born with the height of the first box's label (`KIN`: a car
  1.6 m +- 20 %, a truck 3.2 m +- 25 %, not known 1.7 m +- 35 %); from then on the prior follows the class vote
  (`Tracker._size`, `models.size_prior`): the mixture of the kinds by their tallies, as one Gaussian with the spread
  between the kinds, exchanged for the old one in every story of the object (`radar_lh_prior`: the old prior's
  information out, the new one's in). The detector calls a van `truck` in two boxes of three and `car` in the
  third; born from a `car` box it kept a car's height and, the box height being the strongest range beyond 40 m,
  was read at the range a car of its box height would have: 0.6 of the truth, with a stated spread of a fifth of
  the error (curve_parked: a parked van at 44 m for 75 m; parked_street 44 m for 68 m; adjacent: the overtaken
  truck at 35-50 m for 120 m, then "moving on the path"). A car whose boxes are all `car` keeps its prior.
* **Heading and motion** (`r_extent_step`): a leaf that moves (speed over 2.5 m/s, "it stands" under one half)
  takes its direction of travel as a measurement of its heading, as good as the velocity's direction is known;
  the first time, that also says which end is the front. A leaf that stands gets no such measurement and next to
  no process noise: its heading is held.
* **The road's axis** (`Tracker.road_prior`, native `radar_heading_prior`, once per `road_prior_s`): a standing
  vehicle within `road_prior_off_m` of the road model's path, whose motion has not shown its front and whose
  rectangle no extended sensor measures, takes the path's direction at its place as a prior of its heading. The
  box width measures |sin a|, not on which side of the line of sight the vehicle points: the heading and its mirror
  about the line of sight explain the boxes alike, and the one within `road_prior_mirror` of the axis is taken.
  Then the axis as a measurement, to `road_prior_std` and no better however often it is asked (the heading's
  variance brought down to that at most), where the heading is within `road_prior_snap` of it (a car parked across
  the road keeps its own). Born along the car's own heading, a car parked on a bend kept it or drifted to the mirror
  (curve_parked, seed 1: heading error median 0.42 rad; with the prior 0.10). The decision layer's own snap to the
  axis (`road_axis_standing`) is still needed: without it the cv futures' false alerts rose from 55 to 64 over 18
  runs.
* **A camera box** (`r_extent_update`): a box with both sides in the picture measures the silhouette
  `(L |sin a| + W |cos a|) / range` (a: heading against the line of sight), with the range's own uncertainty in
  its variance.
* **A microwave radar's returns** (the extended-object model): the returns of one scan are grouped into cells
  (`radar_cells`: first the returns on a known object's rectangle, widened by a margin and by what is not known
  of its place, with a fitting range rate; the rest by single linkage in place and range rate). A cell is one
  detection: the mean, the number of returns, the scatter of their places. The model of a rectangle's returns
  (`r_scatter`): they lie on the two sides that face the sensor, each side in proportion to its length (less at
  a grazing angle, never nothing), three in ten at the corner between them. From it follow the mean's offset from
  the centre (the motion update takes the cell as the centre: section 2) and the expected scatter matrix, whose
  three entries are the measurement of (L, W, heading), with the sampling variance of a covariance of n points.
  Held against the simulator's returns: scatter and offset agree to a few centimetres except exactly end-on.
* **One object, not one per side.** Two objects whose places are both known and of which one's centre lies in the
  other's rectangle are one (`radar_merge`); so are two that an extended sensor sees driving as one, end to end.

The message: `len`, `wid` (m), `hdg` (rad, car frame, left positive: where the front points), `ext_std` [m, m, rad],
`ctr` [x, y]. `heading_rad` stays what it was: the direction of travel, null below 1 m/s.

Measured on the simulator (seed 1, day; median errors): camera alone, length 0.14 to 0.50 m, width 0.01 to 0.17 m,
heading 0.03 to 0.12 rad; with the extended radar the place along the line of sight to 0.09 to 0.17 m (camera
alone: 1.4 to 5.5 m), length 1.0 to 4 m (a truck is born as a car and its length is seen only where its side is),
id switches per run 2 to 24 where the tracker before this lane had 40 to 150.

Not built: a yaw rate state (the heading follows the velocity's direction; the range rate of a return off the
centre does not use the body's turn); alternative partitions of the returns kept as hypotheses (one partition per
scan: the objects' rectangles, then distance); the returns of the far sides and of the underbody. With the
simulator's clutter (three static returns per scan beside the road) 30 to 110 false tracks a minute are still
confirmed with the extended radar (before: 125 to 460); the camera-only rate is unchanged.

## 5. Standing or moving: why parked cars were called moving, and what was changed

Found on the two recorded night drives of 2026-11-22 (docs/RADAR_TRUTH.md; the per-update traces of the four
labelled cases are in the lane's report). Parked cars were called `moving` for 13.9 of 41 labelled parked
seconds. Two mechanisms, both in the measurement model, neither in the hypotheses' vote:

1. **The camera's pitch moves, and the model did not say so (9.8 s of the 13.9 s).** While the car drives, the
   camera's pitch against the road moves by 0.33 to 0.40 degrees within 0.4 s and 0.5 to 0.63 degrees within
   0.8 s (the road model's own pitch on these drives; standing: 0.05 to 0.14): bumps, braking, the road's waves.
   The model had a white 0.1 degrees per frame and a slow walk. At 25 to 40 m, 20 mrad of ground angle is ten
   metres of range: "it stands" lost by three to four sigma and the moving hypotheses explained the error as
   speed. The guard "moving only once the sightings by themselves show a speed" did not hold, because the free
   estimate takes the same angles with the same too-small variance. A fit of the same boxes without the ground
   angle (bearing and box height only) says "stands" with chi-square per degree of freedom 0.14 to 0.53.
   **Change:** the pitch's fast part is a Gauss-Markov process (`pitch_gm_deg` 0.5 while driving, 0.1 standing,
   `pitch_gm_tau_s` 1 s) kept per camera beside the slow bias: what the last scan's common angle (`h_common`)
   found of it is carried to the next scan with its memory, taken out of the scan's ground angles, and what it no
   longer says is their variance (`Hyp.cycle`; `Tracker.bias[src]` is now `[slow, variance, fast, variance, time]`).
   A scan with enough settled objects still measures its own common angle and takes it out, as before.
2. **A bearing linearised over a place that is not known (4.1 s).** A new object's moving hypotheses carry a
   velocity prior of 12 to 15 m/s along the car's heading; a second later that is tens of metres of place, and
   linearised at the mean a bearing "spread" of 22 to 45 degrees: a box on the other side of the street passed
   the gate and paired with probability 0.63 to 0.65, though no car drives fifteen metres sideways in a second.
   **Change:** a hypothesis whose bearing is unsure by more than `split_rel` (0.12 rad) is cut along the direction
   in which its place is least known into five narrower Gaussians (same mean and covariance together), each
   linearised where it is; its likelihood is their sum, its update their moment-matched posterior (`r_split`,
   `r_take_parts`). With the hypotheses over scans (section 1) a wrong pairing of this kind is also no longer
   blended into the object.

Not the cause, measured: the detector's scale rate (absent from these recordings; switching it off changes
nothing); a time offset between the car's feed and the camera (the yaw rates of the two agree to within 40 ms;
shifting the feed by up to 200 ms moves the result by 2 s at most); the yaw rate's scale (the camera's and the
car's differ by 5.5 %, which side is wrong is not known; scaling by 5 % changes nothing); the hypotheses' priors
and switching rates (every setting that helps parked cars costs the cars that drive).

Real drives, the scorer of docs/RADAR_TRUTH.md (seconds stationary / moving / unknown):

| | parked | driving with the car | wrong verdicts |
|---|---|---|---|
| before the lane (a36d860) | 11.0 / 13.9 / 16.0 | 0 / 35.8 / 0.9 | 18 % |
| hypotheses over scans, extent (neither change) | 22.2 / 9.5 / 20.2 | 0 / 36.3 / 0 | 12 % |
| + the split only | 26.0 / 8.7 / 17.3 | 0 / 36.8 / 0 | 11 % |
| + the pitch process only | 30.6 / 2.8 / 19.8 | 0 / 36.8 / 0 | 4 % |
| both | 29.3 / 2.0 / 21.2 | 0 / 36.6 / 0 | 3 % |

Simulator: `tests/test_radar.py::ParkedCars` drives past six parked cars with a camera pitch of 0.6 degrees and
1 s: 30 s of `moving` in 406 s without the pitch process, none with it. The traffic scenarios' moving objects are
as before (tools/visual_radar_eval.py). What remains in the simulator's dense parked streets (12 s per run at
night against 16 s before) is objects fed by the boxes of two neighbouring cars in turn.

**Still open: one track fed by neighbouring cars of a row.** Traced on the simulator (`curve_parked`, seed 1, day;
`tools/radar_trace.py --object 1`): a car parked on the outside of a curve gets two boxes at 44 m, is then not
boxed for 0.8 s, and the next box is of the car two places further down the row, where a car driving with the
traffic would be by then. "The same car, driving with us" explains that box with a normalised innovation of 0.1
per row, "it stands" with 3.2; the hypotheses give "the same object" 0.992 against "it was missed and this is
another car" 0.006, the standing hypothesis falls to 0.01, and the object is `moving` at 13 to 17 m/s with a stated
velocity spread of 4 to 6 m/s until, 0.8 s later, the box comes nearer at the car's speed and the story is taken
back (`unknown` at 20 m, the next car of the row then `stationary` from 52 m on). The free estimate's velocity is
that sure because an object's own unknowns (its size, its angle offset) are constant, so that two boxes of ONE
object tell a displacement far better than a place: two boxes of two alike cars defeat exactly that. From one
camera and with no row known yet, the two stories differ only in their priors: how often a new car appears at
that place (the model: 0.15 new objects per scan over the whole field; in a parked street at 13 m/s: one every
half second at the detector's horizon) against how likely a thing beside the road drives with the traffic. Not
solved here. What would: the density of new objects following the structure (section 7: along the extension of a
known row and at the horizon of the detector), an appearance feature that tells two cars apart, or a sensor that
measures range rate (a microwave radar's first return settles it).

**What the place's error across the road is made of (2026-11-24).** Against the centre of the actor whose box an
object last took (tools/radar_trace.py, "is the stated uncertainty of the centre right?"), the error ACROSS the line
of sight is as stated (within 1 / 2 sigma 0.68-0.79 / 0.94-0.98 in the parked scenarios): the bearing is good and
the model knows it. What reads as a lateral error of 1-2.6 m at 25-60 m is the error ALONG the line of sight seen
through the bearing (a car-frame y error of the range error times the sine of the bearing; on a bend the lane's
place also moves with the range). The large ones came from vans and trucks read with a car's height (section 4,
"Height": the false obstacle_ahead objects of curve_parked 299 and parked_street 248 were single-actor tracks of
parked vans), not from rows; the cars that remained (curve_parked 233, 343) are single-actor tracks with the range
1.5-3 m long (the car prior's 1.6 m against the simulator's 1.5 m cars, within the stated 20 %) passing 1.4-1.8 m
beside the car. Hops between the cars of a row are real (7 % of the updates in curve_parked, 14 % beyond 50 m) but
were not behind those alerts. The simulator's detector boxes every car on its own (a box covered by nearer ones for
0.6 or more is dropped, never merged), so a box covering two cars does not occur there.

**A car and a van that swap boxes (platoon, seed 1, day; 2026-11-24).** radar_score's length error of 3.05 m
(0.24 m without the class-vote height prior, `size_from_class: false`) is a swap: at 20.83 s the object of the car
p1 (26 m, its height learnt 1.48 m) takes a box of the van p3 (67 m) with a normalised innovation of 1.0-1.3 per
row, and 0.6 s later the van's object takes p1's box. The swap happens at the same scans with the same innovations
without the class-vote prior: the prior does not let one track take the other's box. What lets it: at 67 m the van's
bottom edge says 33.5 m (the road's grade there and the 0.4 degree pitch error; MEAS, 55-200 m: ground range -35 %),
and a car of 1.48 m reads the van's 38 px box as a car at 36 m; both rows agree on "the car, 6-8 m further", in a
scan in which p1 was not boxed, and that box was labelled `car` (a class term in the pairing would not have stopped
it). The length moved because the van's object, born from a `truck` box with the truck kind's length (10 m, never
seen from the side), then stands where p1 is and the scorer pairs it with p1; without the prior the van's object was
born otherwise and the metric happened to stay small. After the previous paragraph's change: 1.62 m (other seeds and
nights 0.11-0.24 m). Not solved: what would is a far ground angle that knows its error (the grade's share grows with
the square of the range) or a class / appearance term over several scans.

## 6. The past, smoothed

The filtered state is the estimate from the detections up to now; it is what the message carries as the present.
The past can be known better: the chain of nodes from an object's primary back is one story's history, each node
holding the motion hypotheses after its scan and the step that led into it (switching matrix, dt, what the car's
own movement added). The smoother (in `Hyp.after`) runs a backward pass over it after every sighting back to the lag; an object's trajectory (`Obj.history`, read by tools and tests, not by the message) is smoothed when it is asked for (`Hyp.history`), and was until then once per `smooth_hist_s`
(0.5 s; 0: every sighting) as far back as `smooth_s` (2 s) for the trajectory: per motion hypothesis a Rauch-Tung-Striebel step (the gain from a Cholesky factor of the predicted
covariance), with the hypotheses mixed backwards (the smoother of the interacting multiple model filter: each
hypothesis of a node receives the child's smoothed hypotheses in proportion to what it handed them), and a plain
RTS pass for the free estimate. Nothing is written back into the nodes: the filter's rows stay the filter's.

What uses it:
* `vs` in the message: the velocity `smooth_lag_s` (0.5 s) ago from the detections since as well, with `vs_std`
  and `vs_age_s`;
* the verdict: "moving" needs the sightings by themselves to show a speed; that test now reads the free
  estimate's SMOOTHED velocity at the lag (`shown_moving`);
* the trajectory (`Obj.history`): the smoothed places and velocities as far back as the smoother reaches.

Two knobs lower its cost (both off; `radar.json` `{"tracker": {...}}`): `smooth_every_s` (0: after every
sighting): an object's past is smoothed at most this often, a sighting that comes sooner keeps the `vs` it has with
`vs_age_s` grown; `smooth_for: "read"` (default `"all"`): only objects whose verdict is not "moving" are smoothed
(the verdict and the groups read the smoothed free estimate of those only), a moving object has `vs: null`.
Measured 2026-11-24 (`sim.radar_score`, 26 scenarios x day and night, plain and `DETIN`): `"read"` changes no
score in any of the 104 runs; `smooth_every_s` 0.25 changes 8 of 104 runs, each by a few thousandths (id switches
351 -> 345 in all). In a dense scene with a detector at frame rate the backward pass is 5.9 % of the stack's
instructions: `"read"` saves 2.0 %, 0.25 s saves 3.2 %; on a recorded night drive (few objects, a detector at
6-7 a second) neither saves anything measurable.

Because the history is the primary's story, a pairing revised by the hypotheses (section 1) is revised in the
trajectory too. Measured (`tests/test_radar.py::PastAndStructure`): a car braking in a radar with 0.8 m/s range
rate noise, the smoothed velocity of half a second ago is under three quarters of the filtered velocity's error
at that time, while the filtered present follows the braking.

Approximations: the backward mixing ignores that a standing hypothesis hands the FREE estimate (not itself) to
the moving ones; a hypothesis that received nothing stays as filtered; the extent is not smoothed. `smooth: false`
switches it off (`vs` null, the verdict on the filtered free estimate, the trajectory filtered).

## 7. Structure: rows, queues, platoons

`radar_groups` (after every upkeep): two objects are linked when both stand, or both drive at like velocity
(`group_dv`), and the free room between their rectangles along the line that joins them is under the gap (7.5 m
standing, 30 m driving). A linked set whose centres lie on one line (the farthest one off taken out while they do
not: `group_line_m` + 3 % of the length) is a group of at least three; standing with its nearest member ahead in
the car's own lane: a **queue** (two are enough); standing beside: a **row**; driving: a **platoon**. A group keeps
its id while it keeps two members. The message has `grp` and `grp_kind` per object and `groups` (id, kind, the
members in their order along the line).

What a group does for its members:
* **A row or a queue holds its members standing and in line** (`radar_group_support`, every `group_s`): each
  member's standing hypothesis takes "I stand on the line the OTHERS make" as a measurement across the line
  (`group_sigma_m`, the line's own scatter and how well the others' mean place across it is known), and standing is
  `group_gain` times likelier for it than for a thing by itself. The row says that to its sigma and no better
  however often it is asked: the member's variance across the line is brought down to sigma^2 at most, and one
  that knows its place better by its own detections takes nothing from it. (Until 2026-11-24 it was an independent
  measurement at every `group_s`: after five seconds a row member's place was held to a third of `group_sigma_m`
  from the same few neighbours, and the stated place was overconfident: curve_parked, seed 1, standing objects at
  25-60 m, y error within 2 sigma 0.94 with the support, 0.99 without.) A member whose own sightings show a speed is not of the group (the test of section 6), and its own
  detections weigh far more than the gain: a car that pulls out of the row is `moving` and out of the group
  (`PastAndStructure`).
* **A hidden member of a queue persists**: hidden by the queue's own vehicles (section 3) it is kept as long as a
  remembered object (`memory_s`), not `hidden_s`.

Knobs: `groups`, `group_support`, the gaps and tolerances (`tracker.CFG`).

Measured on the simulator (seed 1, day; share of the truth group's member time with a group id shared with
another member): with the extended radar a row 0.90 to 0.98 (the bay-parked row at 2.6 m pitch 0.08: its cars are
not resolved), the queue 0.80, the platoon 0.94; with the camera alone 0.16 to 0.25, because the night detector
follows one or two members of a group at a time. Roadside clutter in a line is a row to it (false grouping 0.49
with the simulator's static radar clutter).

Not built: a group as an estimated object of its own (a shared velocity for a platoon, a pitch between the slots
of a row from which a hidden slot could be predicted, births along a row: section 5's open case); groups do not
constrain moving members.

## 8. The detector's further inputs

The class detector (roadsense-signs `signd`, roadsense `docs/ARCHITECTURE.md`, "Detection inputs for a tracker") can
send more per box than its edges, class and score: which edges are hidden, a second list of boxes below the class
threshold, an appearance vector, a vehicle's lamp pair in the dark, where a scale rate came from. All of it is
optional there and off by default. This section is the tracker's half: what each input does here. Every one is a
knob of `radar.tracker` (`tracker.CFG`, the `in_*` knobs and `grade_k`), **off by default**; with all of them off a
scan is, bit for bit, what it was (the detector's fields are passed on by `core/roadusers.from_lights` and not
read). The arithmetic is in the native core (`chris_native.c`, "The detector's further inputs"); `Hyp.cycle` takes
the scan's constants for it as an optional last argument (`xcfg`), and the hypotheses keep a small memory beside
every row (`Hyp.X`: how the object looks, its last lamp pair, how often a ruler's prior was used, when it last took
a scale rate), copied where a row is copied, so that every story has its own.

The proposal these follow is roadsense `docs/lanes/detin-consumption-design.md`. Where the tracker as it is asked for
something else than the proposal, it is said below ("against the proposal").

### Flags (`in_flags`, `in_box_occl`)
The hidden edges of a box are `E = clip + trunc + occ_edges` (cut by the view, by the frame, under a nearer box).
* **Rows** (`radar/box.py`): a hidden bottom gives no ground angle (kind `b` / `bt`), a hidden top no box height
  (`rb`), a hidden side no width for the extent and no aspect; a box covered by a tenth or more teaches no extent
  either (a pedestrian merged with a neighbour, a car half behind another: the boxes that taught a person 3 m of
  length). Covered with no edge hidden: the height row's spread times `1 + occ`.
* **A box with one side hidden is the visible part of the object** (`x_det_leaf`): its centre is not the object's.
  The hidden part's width is what the leaf's silhouette at its range has beyond the box; the box centre is moved by
  half of that towards the hidden side, and that half is its added spread. (Against the proposal, which only widens
  the bearing's variance: the side is known, so the error has a sign.)
* **The scan's own boxes hide** (`in_box_occl`, `x_box_cover`): section 3 knows settled objects as occluders. A box
  of this scan, tracked or not, first list or low, hides what stands behind it: nearer as the detector orders two
  boxes (the bottom edge lower; a bottom cut by the view or the frame lower than it shows), as far as the leaf's own
  bottom edge is known (`box_occl_margin`). In the story "the leaf took box j" box j is the leaf's own and hides
  nothing of it; in the story "not reported" every box hides. So an object behind a van that was boxed a moment ago
  for the first time is not held to its missing box.
* **A far box with its bottom under another box** is a direction only (kind `b` / `bt`). Until now such kinds came
  from boxes cut by the view's bottom, which only an object nearer than the view's first ground point can have
  given; a box whose bottom a nearer box hides can be of an object at any range. `flags_far_bearing` lets a far
  object take it; the default leaves it to the near ones, because measured on the simulator a direction alone does
  not tell the cars of a row apart: taken, id switches were 264 against 207 (132 runs) for 3 % more tracked time.
  What would use such a box properly is its one-sided range (the true bottom lies below the visible one: the object
  is no farther than the visible bottom says): not built.
* Not built: the detector's `occ_by` (an index into the message) is not carried into the message's `occ_by`; `sym`
  is not read.

### The low-score list (`in_low`)
`objects_low`: the boxes between the detector's floor and the class threshold. They enter the same scan as a second
kind of detection (no second pass: the hypotheses weigh all stories at once). For a leaf with existence r:

    in the first list    pd                      weight  r pd l(z) F / lam(z)
    in the low list      pdlo = k (vis - pd)     weight  r pdlo l(z) F / lam_low(z)
    in neither           1 - pd - pdlo           weight  1 - r (pd + pdlo)

`k` is `low_p` (0.4: of what the first list misses on BDD100K, 41 % is in the low list at a floor of 0.10), and
`low_p_run` (0.7) for an object the first list had within `low_run_s` (the detector misses in runs, and on rides a
low box lay on the object in 68-100 % of such passes). `lam_low`: `low_false_per_scan` low boxes on nothing per scan
at least, followed upwards from the scans themselves. A low box starts nothing (no birth, not in the map of
unexplained boxes), confirms nothing (no sighting counted, existence gains the least a detection gains and no
score), does not weigh "it stands" against "it drives" (the motion hypotheses keep their probabilities: its edges
are too sloppy for that), does not teach the extent, the camera's size limit or the scan's common pitch; its edges are sloppier
(`low_px`, `low_rel`), its class votes with `low_weight`. An object whose last `low_coast` sightings are all low is
put out as `coast`. A scan in which a visible object is in neither list tells against it more than a plain miss did.

### The appearance vector (`in_app`, `in_reid`)
A leaf remembers the mean of its object's vectors (`app_gain`; only whole, uncovered first-list boxes teach it; a
vector of another kind starts it anew). A pairing of a box with a leaf is weighed by

    F_app = clamp( p_same(d | gap) / p_other(d),  1 / B,  B )

d the distance between the box's vector and the memory (the kind's own distance), gap the time since the memory was
refreshed, `p_same` / `p_other` the measured log-normal fits (`app_tables`: roadsense
`docs/lanes/detin-emb-distances.json`; the fit at the gap, the 1 s fit below 1 s). The ratio of two log-normals is a
parabola in ln d that turns round in the fits' tails; beyond its turning point it is held (monotone). B (`app_bound`)
is 8 by day and 3 at night; day and night are mixed in the logarithm by how dark the scene is, which the tracker
reads off the vectors' own luma cells (`app_y_day`, `app_y_night`; or told: `app_dark`). No memory, no vector, a
kind without a table, two kinds: the factor is 1.
* **Re-identification** (`in_reid`): an object dropped by the upkeep goes into a gallery (its id, place, velocity,
  memory) for `reid_s`. An object born where a dropped one could be by now (its last velocity carried on, unsure by
  that velocity; `reid_gate`) takes the dropped one's id, age and class vote when its first vector says so `reid_lr`
  times likelier than not and twice likelier than the next candidate. It is tentative again until its own boxes
  confirm it.

### The lamp pair (`in_lamp_colour`, `in_lamp_size`)
* **Colour**: red, the rear faces the camera; white, the front. For the extent: which end is the front, where the
  object's motion has not shown it (turned when two more boxes say the other end than say this one: a colour is
  wrong now and then), and the heading along the line of sight to `lamp_colour_deg` and no better however often
  (like the road's axis). For a birth with white lamps: both moving stories start from the car's speed reversed
  (an oncoming vehicle; "it stands" keeps its share).
* **Separation: a ruler** (`in_lamp_size`, `x_aux_rows`): see "Rulers" below.

### Rulers: the box width of an axial view (`in_width`), the lamp pair's separation (`in_lamp_size`)
A ruler is a size of the object that the detection does not know, seen from the object's distance: a vehicle's
width (known to a tenth), its lamp separation (to a fifth). The proposal takes each as a range row with a constant
of the object's own, "like the height". **Against the proposal**, neither is a row on the range here, and neither
gets a state:

* **Two sizes seen from one distance measure their ratio, whatever the distance.** The box height against the box
  width, or against the lamp separation, is a measurement of the object's height state `lh` itself, free of the
  range:

      ln(h_px / w_px) = lh - ln(W(lh) s)        s = (L |sin a| + W |cos a|) / W, a: the heading against the line of sight
      ln(h_px / sep)  = lh - ln(lamp_sep_ratio W(lh) |cos a|)

  with `W(h) = 1.8 + 0.35 (h - 1.5)` m the width of a vehicle of that height (cars 1.8 x 1.5, vans 2.0-2.1 x 2.5,
  lorries and buses 2.5 x 3.4: widths vary far less than heights across everything the detector calls a vehicle,
  which makes the width a ruler without knowing the class). What the height then says of the range it says in every
  later box, through the box-height row and the filter's own knowledge of how the two hang together: the ruler's
  information stays with the object as its height does. (The first build had the rulers as rows on the range; their
  information was gone with the next seconds' process noise: a car 1.7 m tall, taken as 1.6 m, read 6 % near again
  two seconds after the ruler's last use. As a measurement of the height it reads right for as long as it is
  followed: `tests/test_radar_inputs.py::Rulers`.) Every motion hypothesis takes the row, also the moving ones that
  hold the height against the box's own rows (section 2): there is no range in it, so the drift that made the height
  a consider parameter there cannot arise from it.
* **The constant is not counted once per box.** A width's own deviation from `W(h)` is the same in every box. Each
  of `ruler_uses` uses takes that variance (`width_rel`^2, `lamp_sep_rel`^2) times `ruler_uses`, with the box's own
  noise beside it (which does average): all uses together hold the prior's information once.
* **Where it applies.** The width: a vehicle whose both sides, top and bottom are in the picture and uncovered,
  seen along its axis: the length's share of the silhouette under 0.4 of the width's (within 8 degrees for a car:
  what the detector calls a car is 3.5 to 6 m long, and at a larger angle that is what the box width would
  measure), the silhouette at the heading believed known to a third (its spread over what is not known of the
  heading by the folded normal: at the axis the silhouette has a kink). **Against the proposal** ("a car 1.8 m,
  10 %: half the height prior's 20 %", for any view "within 20 deg of the line of sight"): at 20 degrees a car's
  silhouette is 3.2 m, not 1.8 m, and what is known of it is what is known of the heading and the length. The lamp
  pair: within about 50 degrees of the axis, a box whose top and bottom are its own.
* **How the range changes needs no constant**: the lamp separation against the leaf's last pair (the heading against
  the line of sight taken out) is a scale rate over the time between them, a row like the detector's own, to the
  lamp centroids' fraction of a pixel; not taken where the box's scale rate already comes from the lamp pair
  (`scale_src`), nor across more than 1.2 s.
* All these rows are mixtures: with `ruler_eps` the number is not the object's (one lamp pair in ten is found on
  other blobs; a box edge on a neighbour) and says nothing; taken only where it is likelier the object's than not.
  They are scalar rows processed after the box's other rows: in the update per motion hypothesis; in the pairing on
  the leaf as a whole, as the likelihood of the extra rows given the others (a van's box on a car's track: the
  wrong aspect).

### The road's vertical curvature (`grade_k`)
With a vertical curvature k the ground at distance x lies k x^2 / 2 off the plane under the car, and the ground
angle is off by k x / 2. **Against the proposal** (a variance term on the row): that error is not noise from box
to box, it is the object's own offset and the same in the next box, so as row noise it would be averaged away over
the object's boxes. It goes where the tracker already has such an offset: the state `dp` is born unknown by
`grade_k` x / 2 on top (`r_birth`), and walks by `grade_k` dr / 2 when the range changes by dr (`h_step`).

### The scale rate (`in_scale`)
* Successive values of one object share most of their baseline (a value is the mean rate over `scale_dt_s`, a new
  one comes every 0.2-0.4 s): the variance times the baseline over the time since this leaf's previous scale row,
  where that is shorter (`x_det_leaf`).
* `scale_src` "lamps": Huber's weight at `scale_lamps_k` sigma on the leaf's own innovation (the variance raised
  until the innovation is that many sigma).

### What was measured (2026-11-24)
On the detector simulator emitting what a camera detector would (`sim/roaduser_sim.py` `DETIN`: visible-part boxes,
roadside occluders, flags, the low list, vectors fitted to the measured distance tables, lamp pairs, scale rates
with shared errors, vehicles of spread sizes; with the scale rate on: docs/VISUAL_RADAR_EVAL.md), the tracker alone
(`sim.radar_score`), all 11 scenarios x seeds 1-6 x day and night = 132 runs per row. Confirmed outputs against the
truth: error along the line of sight by range band (m), id switches, share of detectable time held and confirmed,
seconds a standing actor was called `moving`. Then the whole stack's false `obstacle_ahead` episodes (shadow mode;
6 scenarios x seeds 1-6 x day and night, with the prediction / with constant-velocity futures). The real drives'
recordings carry none of the fields: there, with the knobs off, the output is identical (both captures of
tools/cost_bench.py, every stream), and nothing more can be said from them.

| knobs on | 0-25 p50 / p90 | 25-60 p50 / p90 | 60+ p50 / p90 | id switches | held | confirmed | standing called moving (s) | false episodes pred / cv |
|---|---|---|---|---|---|---|---|---|
| none (today's stream on today's simulator) | | | | | | | | 91 / 79 |
| none (the realistic stream: the "before") | 0.76 / 3.77 | 2.40 / 8.21 | 6.06 / 19.5 | 262 | 0.262 | 0.091 | 74 | 110 / 93 |
| `in_flags` | 0.74 / 3.70 | 2.52 / 8.12 | 7.67 / 21.5 | 207 | 0.250 | 0.086 | 69 | 117 / 108 |
| `in_box_occl` | 0.75 / 3.71 | 2.36 / 8.12 | 6.34 / 19.5 | 270 | 0.261 | 0.090 | 72 | not run |
| `in_low` | 0.82 / 3.93 | 2.66 / 8.90 | 7.81 / 22.0 | 282 | 0.270 | 0.115 | 111 | (with the set below: 126 / 128) |
| `in_app` | 0.75 / 3.77 | 2.48 / 8.09 | 6.97 / 19.8 | 271 | 0.267 | 0.092 | 64 | 101 / 92 |
| `in_lamp_size` | 0.77 / 3.66 | 2.39 / 8.06 | 6.34 / 21.3 | 252 | 0.264 | 0.091 | 72 | 108 / 96 |
| `in_width` | 0.76 / 3.67 | 2.34 / 7.71 | 7.66 / 20.2 | 277 | 0.267 | 0.091 | 74 | 104 / 99 |
| `in_scale` | 0.75 / 3.71 | 2.39 / 8.00 | 7.03 / 19.3 | 254 | 0.263 | 0.091 | 70 | 109 / 99 |
| `grade_k` 1/3000 | 0.77 / 3.91 | 2.53 / 8.69 | 7.30 / 20.3 | 281 | 0.258 | 0.089 | 71 | not run |
| `in_app` + `in_width` | | | | | | | | 104 / 91 |
| **`in_app` + `in_width` + `in_lamp_size` + `in_scale`** | 0.74 / 3.48 | 2.09 / 7.59 | 7.46 / 18.4 | 255 | 0.273 | 0.093 | 64 | 101 / 103 |
| ... + `in_flags` | 0.76 / 3.49 | 2.20 / 7.38 | 6.68 / 17.0 | 212 | 0.260 | 0.089 | 58 | 116 / 107 |
| ... + `in_flags` + `in_low` | 0.77 / 3.63 | 2.27 / 7.55 | 6.82 / 19.2 | 208 | 0.268 | 0.112 | 87 | 126 / 128 |

On the (actor, frame) pairs confirmed both before and with the four knobs of the bold row: at 25-60 m the mean error
along the line of sight 3.22 -> 2.88 m by day and 3.77 -> 3.06 m at night (p90 8.8 -> 6.7 m at night); at 0-25 m
1.77 -> 1.56 m at night; beyond 60 m the mean 8.2 -> 7.8 m by day with the median worse (5.7 -> 6.3 m).
`in_lamp_colour` (the oncoming scenario at night, 12 seeds) and `in_reid` (it fires two or three times a run) change
no score measurably.

What it says:
* **Range at 25-60 m**: the width ruler is the one input that moves it (leave it out of everything-on: 25-60 m
  p50 2.31 -> 2.61, beyond 60 m 7.5 -> 8.9), the lamp pair adds at night. Neither needs a new state.
* **Phantom motion of standing things** falls with the appearance ratio (74 -> 64 s) and is the one thing that also
  lowers the false episodes with the prediction (110 -> 101). The id switches do NOT fall with it alone.
* **The flags** cut id switches by a fifth and cost 3-5 % of the tracked time, and **the false episodes rise** with
  them (110 / 93 -> 117 / 108; not the side shift: 115 / 112 without it). Dropping the rows of hidden edges takes
  away wrong information and, in a row of parked cars where most boxes have a hidden edge, most of the information.
* **The low list** buys a quarter more confirmed time and pays with standing things called moving (74 -> 111 s) and
  more false episodes. The simulator's low list is three to five times as busy as the rides' (its excess of visible
  actors), so this is a pessimistic reading; it is not evidence for switching it on.
* **`grade_k`** does nothing for this simulator, whose roads' vertical curvature (at most 1 / 6000 m) is half the
  1 / 3000 m assumed; **`in_box_occl`** is neutral.
* **No set of knobs lowers the false `obstacle_ahead` episodes beyond what one run's noise could be** (101-104
  against 110 with the prediction; with constant-velocity futures none is better than 91-92 against 93).

Recommended: all knobs stay off in the code (the detector's options are off on the car, and the appearance tables
are from other cameras). When the detector's options are switched on, with its scale rate: `in_app`, `in_width`,
`in_lamp_size`, `in_scale`. Not `in_width` on today's stream without a scale rate: there (today's simulator with
spread sizes, 132 runs) it moves 25-60 m from 3.43 to 3.22 m and standing things called moving from 192 to 239 s
(each use shifts the height, and with no scale rate the range's shift reads as motion).

Cost (this laptop, one scan of the tracker with the detections' making, the realistic stream; a phone's small core
is 10 to 16 times slower): 0.30 ms with the knobs off; each knob but one adds 0.01 to 0.03 ms; `in_low` adds 0.14 ms
(more detections); the four recommended together 0.30 ms; everything on 0.46 ms. With the knobs off the whole stack
on the recorded captures costs what it did (tools/cost_bench.py: 1.461 -> 1.462 ms per road frame on the real
capture, 6.42 -> 6.44 ms on the dense simulated one; outputs identical).

## 9. Parked cars: what the tracker's own arrays said, and five knobs (2026-11-24)

All five are **off by default** (with them off every scan and every message is what it was: the goldens and digests
of tests/test_native.py and tests/test_radar_inputs.py; the calls with an `xcfg` or a frame configuration of the
earlier length are taken as before). Recorded with them on: `tests/data/make_radar_golden.py --parked`
(tests/test_radar_parked.py).

### What was found (plain and realistic stream, `curve_parked` + `parked_street`, seeds 1-3, day and night)
Every 0.1 s every confirmed object against the actor whose box it last took (the lane's scratch tool on
tools/radar_trace.py: 8881 / 6535 standing track-frames).

* **The range's stated spread is what the boxes hold, not a filter loss.** At its sightings the standing hypothesis
  of a track fed by one car states 8-10 % of the range at 15-40 m, and a batch fit of all its boxes under "it
  stands" with the same variances gives 7-10 %: the filter is within a tenth of the information. The 17-20 % "at
  every distance" of all track-frames is three things: (1) the simulator's detector boxes a car 14 times in a pass
  but only 3 times under 25 m and 2 under 15 m, so a near car is mostly carried on from its last far box, with a
  spread of 1.5-2.5 m that stays while the range shrinks; (2) tracks are young: 43 % (plain) of the standing
  track-frames are of objects fed by more than one car; (3) the bearing of a near box was taken as noise of a tenth
  of its width.
* **"Parked cars are called moving" is the row, not the scale.** Objects that only ever took one car's boxes:
  `p_stationary` under 0.5 in 9 % of their frames (most of it in their first 2.5 s, the birth prior 0.4), called
  `moving` in 1 %. Objects fed by two or more cars: 46 % and 32 %. 12 % (plain) / 8.6 % (realistic) of all updates
  put a box of ANOTHER car into an object; at such an update "it stands" was believed with a median of 0.39 (the
  birth prior: the object had not settled), and fitted the box with 1.5 per row against 0.5 for its own car's.
* **Jumps**: of the 78 / 35 jumps of a confirmed object's ground place by more than 3 m and 2 sigma between two
  samples (under 60 m; 10 minutes each), 47 / 16 are at an update with another car's box; the rest are the
  hypotheses changing their mind about an earlier one and far boxes.
* **The bias** (+7 to +11 % long at 8-50 m, plain) is the car prior's 1.6 m against the simulator's 1.5 m, as section
  2 said; on the realistic stream (sizes spread) +2 to +7 %.
* **Night, true conflicts** (the obstacle of the stopped-car and half-in-lane scenarios): born at 24-45 m at night
  against 48-65 m by day, confirmed at 11-28 m against 30-41 m, in 9 of 24 night runs never confirmed. Its speed at
  confirmation is small (0.2-2.9 m/s) and its range right to 2 m: the limit is the first box and the confirmation,
  not the range or a phantom speed. `confirm_hits` 2 changes nothing, `confirm_exist` 0.9 next to nothing: what is
  slow is the existence itself (the misses between a night detector's boxes each take ln(1 - pd) off it). Not
  solved here.

### `ctx_prior`: the street's own count (tracker)
A new vehicle's "it stands" is raised by the vehicles already judged at its distance beside the road's path (kernel
`ctx_band_m` 0.8 m; the road model's path on the ground, else the car's own line): n_s standing, n_m driving ->
(n_s + `ctx_n` p0) / (n_s + n_m + `ctx_n`), at most `ctx_max` 0.9, never lowered (a stopped car among driving ones
keeps its kind's 0.4). A vehicle raised to a half or more takes the road's axis as its heading at once (section 4's
prior, not a second later). `Tracker._context`, native `radar_stand_prior`.

### `size_learn`: the kind's height from the objects the car has passed (tracker)
A standing vehicle whose line of sight swung by `size_learn_swing_deg` (5) over its sightings was ranged by the
car's own travel; its standing hypothesis' height with the prior it was given taken out is what its boxes said.
`radar_size_fit` makes the kind's mean and spread of these (Huber's weights, held to the built-in prior with the
weight of three objects, the spread no smaller than 0.10, the mean no further than 0.25); `_size` gives them to
every object of the kind from its next vote on. It is every scale error of the box-height range at once (the cars'
real size, the detector's boxes, the focal length). On the simulator: 30 cars -> 1.51-1.55 m +- 10 % (truth 1.5 m,
built-in 1.6 m +- 20 %); on the two recorded drives 5 cars -> 1.53 m +- 14 %. A queue ahead (no parallax) teaches
nothing.

### `sil_bearing`: the box's middle is the silhouette's (tracker)
The bearing row predicts the middle of the rectangle's picture (`r_sil_shift`: the corners' picture coordinates),
not the centre's bearing; what is not known of the rectangle goes into the row's variance (`r_sil_var`, in
`r_reff`); the box's middle is then taken with `sil_bearing_rel` 0.05 of the box width (the camera's 0.11 had to
cover the difference: 0.9 degrees for a car passed at 10 m).

### `std_rho`, `path_normal` (the radar box)
`std_rho`: the correlation of the place's x and y errors beside `std` (docs/TOPICS.md). `path_normal`: beside the
path is across the path where the object is: `path_offset_m` is the lateral difference times the cosine of the
path's direction there, and "surely in" (`in_path`, `p_threat`) takes the place's spread across the path from the
whole covariance (it took the car's lateral axis: a kerbside car on a bend, unsure along the line of sight by 5 m,
was "surely in" by its 0.4 m of y).

### Measured
Standing actors, both scenarios, seeds 1-3, day + night (12 runs, 10 minutes of driving per row); range bands
8-15 / 15-30 / 30-50 m.

| plain stream | spread / range | bias % | within 1 / 2 sigma (15-30 m) | `p_stationary` < 0.5 | called moving | called stationary | updates with another car's box | birth -> `stationary` med / p90 (never) |
|---|---|---|---|---|---|---|---|---|
| base | 0.19 / 0.17 / 0.19 | +9.4 / +10.6 / +7.4 | 0.67 / 0.93 | 0.248 | 0.146 | 0.53 | 12.0 % | 1.72 / 3.05 s (31) |
| `ctx_prior` | 0.18 / 0.16 / 0.17 | +7.7 / +8.1 / +4.9 | 0.68 / 0.94 | 0.168 | 0.108 | 0.71 | 10.9 % | 1.36 / 2.66 s (17) |
| `size_learn` | 0.18 / 0.15 / 0.15 | +8.8 / +8.8 / +4.8 | 0.66 / 0.91 | 0.258 | 0.161 | 0.53 | 12.2 % | 1.75 / 2.92 s (28) |
| `sil_bearing` | 0.18 / 0.16 / 0.18 | +8.7 / +8.3 / +4.7 | 0.68 / 0.93 | 0.214 | 0.117 | 0.57 | 10.7 % | 1.71 / 3.00 s (29) |
| all three | 0.16 / 0.13 / 0.13 | +7.1 / +5.8 / +1.9 | 0.69 / 0.93 | 0.152 | 0.100 | 0.75 | 9.0 % | 1.31 / 2.66 s (12) |

Realistic stream (`DETIN`), base -> all three: spread / range 0.20 / 0.18 / 0.18 -> 0.17 / 0.15 / 0.14,
`p_stationary` < 0.5 0.220 -> 0.161, called moving 0.115 -> 0.106, updates with another car's box 8.6 -> 6.7 %; but
the place ACROSS the line of sight within 2 sigma 0.85 -> 0.73 and the range beyond 50 m within 2 sigma 0.91 -> 0.82:
a half-hidden car's box is not its silhouette, and `sil_bearing` believes its middle too well there (it needs the
detector's flags, section 8, to know). Moving actors of the six traffic scenarios (36 runs per stream): called
stationary 0.047 -> 0.048 (plain), 0.054 -> 0.046 (realistic), birth -> `moving` 1.94 -> 1.84 s and 1.60 -> 1.53 s.

The whole stack's false `obstacle_ahead` said episodes (shadow mode, predicted futures, 6 scenarios x seeds 1-6 x
day and night; paired bootstrap over seeds against the base; d2x counts slow leads as false):

| | plain: total, paired difference | realistic: total, paired difference | true conflicts warned (of 54), mean lead |
|---|---|---|---|
| base | 73 | 91 | 37, 1.54 s / 36, 1.77 s |
| `ctx_prior` | 76, +3 +- 5.1 | 90, -1 +- 3.0 | 35, 1.60 s / 37, 1.76 s |
| `size_learn` | 91, +18 +- 5.3 | 92, +1 +- 4.1 | 36, 1.55 s / 36, 1.80 s |
| `ctx_prior` + `size_learn` | 86, +13 +- 4.3 | 94, +3 +- 4.6 | |
| all three (before the heading at birth) | 96 | 99 | 36, 1.69 s / 39, 1.71 s |

So: `ctx_prior` makes the tracker say the truth about parked cars sooner (called stationary 0.53 -> 0.71 of the
time, fewer hops) at no cost to moving traffic, and the alerts do not move: the causes change places (place jumps
31 -> 25, moving futures of a parked car 4 -> 5, the stated spread alone 4 -> 8: a car now known to stand, unsure
along the line of sight on a bend, is warned of by its spread). `size_learn` reads the range better (the bias and
the spread fall) and, by itself on the plain stream, raises the alerts significantly (jumps 31 -> 30, spread 4 ->
11, moving futures 4 -> 10): with the size known to a tenth, the next car's box is a surer displacement, and an
object that has not settled is the sooner called moving. `sil_bearing` costs alerts on moving traffic (`adjacent`
6 -> 11 / 8 -> 18 with all three). Real drives (docs/RADAR_TRUTH.md; base: wrong verdict 3.1 of 94.0 judged s,
real things never followed 8): `ctx_prior` identical; `size_learn` 1.1 of 91.6 s, never followed 9; `sil_bearing`
1.9 of 91.1 s, 9; all with the two box knobs 1.9 of 89.0 s, 10.

Recommended: `ctx_prior` on (with `std_rho` for the prediction when it reads it, and `path_normal`); `size_learn`
and `sil_bearing` not yet: the first once the decision layer takes a standing object's spread as it now is, the
second with the detector's flags. Not built: births along a known row's extension, appearance in the row (the
`in_app` knob of section 8 is the place), a confirmation that knows a night detector's miss runs.

`sim.radar_score`, all 11 scenarios x seeds 1-3 x day and night (66 runs per stream), base -> `ctx_prior`: seconds a
moving actor was called moving / stationary 470 / 16.1 -> 473 / 16.2 (plain), 475 / 20.9 -> 471 / 20.4 (realistic);
seconds a standing actor was called stationary / moving 649 / 89 -> 819 / 80 and 501 / 54 -> 617 / 38; id switches
370 -> 348 and 235 -> 235; time held and confirmed, the place and the velocity errors as before (along p50 3.10 ->
3.04 m in traffic). Cost (tools/cost_bench.py, radar per road frame, three interleaved runs on a shared machine):
dense simulated capture 2.41-2.52 ms at the base, 2.11-2.52 ms with the knobs off, 2.40-2.57 ms with all five on;
recorded capture 0.24 / 0.22-0.25 / 0.15-0.27 ms: nothing beyond the runs' own scatter. With the knobs off every
stream of both captures is identical to the base's (`cost_bench.py diff`).

## 10. One car, one object in a row; the detector's miss runs; the pose head (2026-11-24)

Nine knobs of `radar.tracker`, all **off by default** in the code of this section's commit (with them off every scan
and every message is what it was: the goldens and digests of tests/test_native.py, tests/test_radar_inputs.py and
tests/test_radar_parked.py stand as recorded; a detection's row grew by eight numbers and the hypotheses got a second
row memory, `Hyp.Y`, and a table, `Hyp.pdl`, both made only when a scan asks). Recorded with them on:
`tests/data/make_radar_golden.py --rows` (tests/test_radar_rows.py). `Hyp.debug(1)` keeps a scan's pairing numbers
for tools (the log ratio of every leaf and detection, ln r, ln(1 - r pd), the densities).

### What the arrays said of the wrong-car updates
Every update of `curve_parked` + `parked_street`, seeds 1-3, day and night (5941 updates on the plain stream, 4288 on
the realistic one), with the scan's own pairing numbers and the hypothesis tree after it:

* **The right story was nearly always "missed, and this box is a car that has no object yet"** (plain: 665 of 714
  wrong-car updates; an object of the box's own car that was more than a tentative alternative existed in 49). In a
  quarter of them the object's own car was boxed in the same scan as well.
* **The pairing did not refuse them, and could not at that range.** The log odds of "took it" against "missed + something
  else" were 3.4 in the median for a wrong-car box against 5.0 for a right one (quartiles 2.0-5.1 against 3.2-6.6):
  no bar between the two separates them. Under "it stands" a neighbour's box fits with 1.5 per row (a right one 0.5):
  the neighbour stands 6.5 m further along the line of sight, which is 1.5 sigma of the standing hypothesis' own range
  (0.17 of the range), about 1 degree of bearing at 30 m against a box centre taken to 0.11 of its width (0.5
  degree), and a fifth of box height against 0.1. The gate is in measurement space with the object's own covariance
  already; what is missing is information, not a metric.
* **Who hops is who has not settled.** The share of updates that are wrong-car by the object's "it stands" before
  the update: under 0.3: 35 %; 0.3-0.5 (the birth prior): 14 %; 0.5-0.8: 10 %; 0.8-0.95: 7 %; above: 2 %. Two
  thirds of all wrong-car updates happen below 0.5. An object that was called moving by one such update goes on
  eating the row's boxes at a third of its updates.
* **The alternative was there and lost.** After the scan the story "missed" still held a median 0.10 of the weight
  (present in 93 %), a birth from the box in 75 %; five scans on, one wrong-car reading in nine had been taken back.
  The two stories become alike: in one the object jumped to the neighbour and a new object sits on its car, in the
  other the reverse; from then on both explain every box equally. What tells them apart is only the price of the jump
  when it was made. Deeper trees do not help (`hyp_n_scan` 8, `hyp_k_best` 128, `hyp_max_leaves` 24: wrong-car updates
  9.9 % against 9.1 % at the defaults, with the knobs below on).
* **A new object could not hold its own car's second box.** It exists with 0.2 and its three stories are wide; and
  where its box was near the view's smallest, the camera's model gave it a tenth of the usual chance of a box (below).

### `row_ctx`: the place of every box as a row (tracker `_row_ctx`, native `XC_ROWB`, `r_birth`)
Before a scan pairs anything, every vehicle box with a ground point is given q = n_s / (n_s + n_m + `ctx_n`): the
vehicles already judged at its distance beside the road's path (the count of `ctx_prior`, of the box's own place).
* `row_birth` (2.5): "something else" is exp(`row_birth` q) times denser at the box. The density of new objects was
  one number over the whole field; beside a row the next box along IS another car.
* `ctx_exist` (0.6): an object born there exists with `new_exist` + (`ctx_exist` - `new_exist`) q.
* `ctx_vel` (2 m/s): its moving stories start from (1 - q) x the traffic's speed, and as narrow as `ctx_vel` where q
  is 1: if a car in a row moves at all it starts from rest. "The same car, driving with us" was what explained the
  next car of the row (section 5).
The density alone does nothing (it lowers right and wrong pairings alike: 11.6 % against 11.2 % on one drive); with
`ctx_prior` (which makes the pairing read the box mostly through "it stands") the three together take the wrong-car
updates from 10.9 % to 9.6 %.

### `det_chain`, `pd_learn`, `exist_shared`: a detector that misses in runs, and what a view can see
Traced on the night runs of the true-conflict scenarios (a stopped car in the lane; 30 runs): the first box at
39 m, the second at 33 m, the third at 26 m (median), confirmed at 20 m, in 16 runs never; several of those had four
to six boxes of the car in a row. What happened there:
* the camera's two views alternate; the object is born in one, the next scan is the other's. The model of what a
  view can box is a ramp from "the smallest box this view has shown" (a tenth of the usual chance) to twice that
  (all of it). Most of a view's boxes ARE near its smallest (wide view at night: smallest 35 px, the boxes 35-45
  px), so the model gave the object's next box a chance of 0.06, and the best story was "a new object" at every
  box: three objects on one car, each existing in a part of the stories, none confirmed;
* the misses themselves were not the limit: the object's misses between the boxes cost 0.05-0.2 each (the same
  model said it could hardly be seen).

`pd_learn`: the chance of a box is learnt per view and box height (three bins to the octave) from the settled
objects' own boxes and misses there, held to the sensor's model with the weight of `pd_learn_n0` scans (native
`Hyp.pdl`, `r_p_detect`). `det_chain`: per leaf and view a two-state chain, "the detector has it" (a box with
probability `det_chain_d`) or "has lost it", whose long-run share is the sensor's model and which forgets within
`det_chain_tau_s`; the chance of a box, the price of a miss and the pairing read it (native `x_chain`): an object
boxed in this view a moment ago is expected again, a run of misses costs its first miss and little more.
`exist_shared`: an object's existence counts the stories in which ANOTHER object took its last box (the same
detection of the same scan) with that object's existence (native `h_settle`): the stories of a young object disagree
on whose its boxes were, every one of them has a thing there. (A rival that shares the box is then put out with the
same `exist`; it is still tentative: `confirm_hits`.)

`low_runs_only` (the low-score list only for an object the first list had within `low_run_s`) and `low_exist_only`
(a low box takes the miss back and leaves the object's numbers to its motion) are built and measured: on the
realistic stream at night they confirm nothing sooner (never confirmed 15 of 30 against 11 without the low list) and
keep objects alive on the low boxes of other cars. Not recommended. Lamp pairs as existence evidence: not built.

### The pose head (`in_pose_gn`, `in_pose_gc`, `in_pose_yaw`; contract pose1)
The tracker's half of roadsense `docs/lanes/detin-consumption-design.md` section 9. `core/roadusers.from_lights`
passes `pose` and `pose_kind` on; a pose is read where `pose_kind` is `pose1`, the box is of a vehicle class and
`kind[0]` >= `pose_kind_min` (radar/box.py).
* **`gn`, the near-face ground point**, is the ground-angle row. On a whole box ONE row: `gn` where its sigma is
  smaller than the bottom edge's and it is seen (`pose_min_seen`), else the bottom edge (the same pixels). On a box
  whose bottom is hidden (cut by the view or the frame, under a nearer box) it is the row that box lacked: the
  detection is "ground angle + bearing" where it was a bearing only, the size rows stay off, the sigma times
  `pose_gn_occ` (1.5) under a nearer box, and the Gaussian is cut at the visible bottom (the contact lies no higher:
  `radar_trunc_low`, the moments of the one-sided normal). Successive rows of one leaf share their error for about a
  second: the variance times `pose_corr_s` over the time since the leaf's previous one (`x_det_leaf`, `Hyp.Y`).
* **`gc`, the footprint centre's column**, is the bearing row, with its own sigma, also where a side of the box is
  hidden (the flags' side shift is then not applied). Not to be combined with `sil_bearing` (that predicts the
  silhouette's middle for every box of the scan).
* **The heading** (`x_pose_yaw`): the end by votes (ln(p / flip) bounded by `pose_vote_max`, half within
  `pose_vote_half_s` of the previous vote; the front is turned at `pose_vote_turn` against it unless the motion has
  shown it), the axis as a row on the extent's heading folded into a half turn (sigma max(yaw_std,
  `pose_yaw_floor_deg`), the same memory of errors, Huber's weight at `pose_yaw_huber` sigma), not for a leaf that
  drives. A new object's heading is the head's; seen from the front, its "drives with the traffic" story comes the
  other way. In the dark (the tracker's own reading of the vectors' luma) a lamp pair's colour on the box votes
  with the full bound, the head with `pose_vote_dark`, and its axis is taken only with p >= `pose_axis_dark_p`.
* Not built: `gl` / `gr`, `dim` as a class vote, `occ` as a flag, the head's `kind` as a class hint, the width
  ruler's test read from `yaw` (it reads the extent's heading, which the head now sets).

The simulator emits the head's fields with its measured error statistics behind `pose` (preset `detin_pose`:
`sim/roaduser_sim.py` `POSE`, `PoseGen`; tests/test_sim_pose.py).

### Measured (2026-11-24; paired: the same seeds and runs for every row)
Knob sets: **rows** = `ctx_prior` + `row_ctx` + `det_chain` + `pd_learn`; **+shared** = rows + `exist_shared`;
**+sil** = +shared + `sil_bearing`; on the realistic stream (`detin`) each with `in_app`; **+pose** = +shared +
`in_app` + the three pose knobs on the stream with the pose head (`detin_pose`).

The tracker's own arrays (the lane's scratch tools on tools/radar_trace.py; `curve_parked` + `parked_street`, seeds
1-3, day and night: 10 minutes of driving per row; standing actors):

| | updates with another car's box | of settled objects | standing track-frames of objects fed by several cars | jumps over 3 m (at such an update) | called moving / stationary | `p_stationary` < 0.5 | range spread / range, bias at 15-30 m, within 1 / 2 sigma | birth -> `stationary` med / p90 (never) |
|---|---|---|---|---|---|---|---|---|
| plain, base | 12.0 % | 9.7 % | 0.43 | 78 (47) | 0.146 / 0.53 | 0.248 | 0.172, +10.6 %, 0.67 / 0.93 | 1.72 / 3.05 s (31) |
| plain, rows | 9.4 % | 7.5 % | 0.33 | 50 (26) | 0.085 / 0.75 | 0.128 | 0.166, +9.1 %, 0.70 / 0.95 | 1.30 / 2.50 s (24) |
| plain, +shared | 9.2 % | 8.2 % | 0.32 | 62 (35) | 0.078 / 0.75 | 0.123 | 0.164, +8.8 %, 0.71 / 0.95 | 1.27 / 2.47 s (24) |
| plain, +sil (the first build of `exist_shared`) | 7.2 % | 5.7 % | 0.26 | 58 (31) | 0.061 / 0.78 | 0.103 | 0.157, +6.9 %, 0.71 / 0.95 | 1.26 / 2.50 s (24) |
| detin, base | 8.6 % | 6.7 % | 0.29 | 35 (16) | 0.115 / 0.54 | 0.220 | 0.176, +4.9 %, 0.69 / 0.92 | 1.92 / 3.06 s (39) |
| detin, +shared (first build) | 4.9 % | 2.9 % | 0.17 | 37 (15) | 0.060 / 0.72 | 0.118 | 0.172, +2.2 %, 0.72 / 0.95 | 1.32 / 2.85 s (28) |
| detin_pose, +pose (first build) | 4.0 % | 2.6 % | 0.15 | 37 (10) | 0.053 / 0.74 | 0.103 | 0.172, +5.6 %, 0.73 / 0.96 | 1.22 / 2.69 s (26) |

("First build": `exist_shared` before it was held to the best story's object and away from settled rivals; the
tracker's calibration differs little between the builds, the id switches do: below.) Moving actors of the six traffic
scenarios (36 runs per stream), plain base -> rows: called stationary 0.047 -> 0.051, called moving 0.886 -> 0.869,
birth -> `moving` 1.94 -> 1.93 s, range within 2 sigma at 15-30 m 0.95 -> 0.95; realistic base -> +shared -> +pose:
called stationary 0.054 -> 0.048 -> 0.051. The place across the line of sight within 2 sigma (standing, realistic
stream): 0.85 -> 0.84 -> 0.91 with the pose head's centre.

`sim.radar_score`, all 18 scenarios x seeds 1-3 x day and night (108 runs per row); the paired difference of the id
switches against the base with its 1 sigma over the runs:

| | id switches (paired) | in moving traffic | held / confirmed share | moving actors called moving / stationary (s) | standing called stationary / moving (s) | along p50 moving / standing (m) | heading p50 (rad) |
|---|---|---|---|---|---|---|---|
| plain, base | 503 | 205 | 0.267 / 0.095 | 712 / 33 | 819 / 129 | 3.48 / 2.78 | 0.084 |
| plain, rows | 440 (-63 +- 28) | 189 (-16 +- 19) | 0.269 / 0.096 | 727 / 31 | 994 / 91 | 3.37 / 2.59 | 0.079 |
| plain, +shared | 518 | 228 | | | | | |
| plain, +shared, first build | 560 (+57 +- 33) | 248 (+43 +- 22) | 0.279 / 0.100 | 732 / 32 | 1059 / 83 | 3.43 / 2.59 | 0.082 |
| plain, +sil, first build | 485 (-18 +- 39) | 206 | 0.276 / 0.098 | 716 / 35 | 1081 / 71 | 3.57 / 2.53 | 0.074 |
| detin, base | 306 | 138 | 0.246 / 0.083 | 700 / 41 | 622 / 72 | 3.63 / 2.60 | 0.081 |
| detin, rows | 296 | 135 | | | | | |
| detin, +shared | 350 | 151 | | | | | |
| detin_pose, +pose, first build | 281 (-25 +- 29) | 122 (-16 +- 16) | 0.275 / 0.093 | 810 / 37 | 844 / 21 | 3.31 / 2.34 | 0.054 |

Night, the obstacle of the ten true-conflict scenarios (seeds 1-3: 30 runs by day, 30 at night; the range at which an
object on it is first seen confirmed by the next scan, mean with "never" as 0; confirmed objects whose boxes were
never an actor's):

| plain | day: confirmed, never, mean | night: confirmed, never, mean | clutter confirmed |
|---|---|---|---|
| base | 34.5 m, 2, 33.9 m | 20.0 m, 16, 10.3 m | 1 of 354 |
| rows | 35.4 m, 3, 33.6 m | 20.9 m, 15, 11.8 m | 1 of 354 |
| +shared | 35.2 m, 2, 34.9 m | 22.1 m, 11, 14.9 m | 1 of 377 |

(The boxes themselves: by day the first three at 54 / 47 / 43 m, at night at 39 / 33 / 26 m; in 10 night runs the car
is boxed three times or fewer before the collision.) `confirm_hits` 2 on top changes nothing.

Through the stack (shadow mode; the lane's scratch tools `diag.py`, `honest.py`: every said `obstacle_ahead` episode
against the truth; the six false-episode scenarios and `stopped_brake_14`, seeds 1-6, day and night, predicted
futures: 78 runs; then the ten true-conflict scenarios, seeds 1-3, day and night: 27 + 27 runs):

| | false episodes (day / night), paired difference | arguable / true | causes: jump / mean in lane / spread / moving futures | true conflicts warned by day, mean lead | at night, mean lead |
|---|---|---|---|---|---|
| plain, base | 54 (21 / 33) | 7 / 9 | 26 / 7 / 1 / 5 | 23 of 27, 1.79 s | 12 of 27, 1.17 s |
| plain, rows | 64 (30 / 34), +10 +- 8 | 6 / 8 | | 24, 1.85 s | 11, 1.27 s |
| plain, +shared | 66 (32 / 34), +12 +- 9 | 6 / 7 | 22 / 14 / 5 / 5 (first build) | 24, 1.87 s | 14, 1.27 s |
| plain, +sil, first build | 75 (39 / 36), +21 +- 10 | 6 / 6 | 22 / 19 / 8 / 6 | 24, 2.04 s | 13, 1.53 s |
| plain, +shared + `size_learn`, first build | 71, +17 +- 9 | | | | |
| detin, base | 83 (37 / 46) | 4 / 7 | 25 / 14 / 9 / 14 | 23, 1.98 s | 14, 1.45 s |
| detin, rows | 77 (37 / 40), -6 +- 8.5 | 3 / 7 | | 21, 2.11 s | 13, 1.37 s |
| detin, +shared | 83 (39 / 44), 0 +- 9 | 5 / 6 | 31 / 18 / 5 / 8 (first build) | 23, 2.15 s | 14, 1.38 s |
| detin_pose, +pose, first build | 69 (34 / 35), -14 +- 7 | 4 / 9 | 21 / 20 / 5 / 4 | 23, 2.20 s | 15, 1.44 s |

Real drives (docs/RADAR_TRUTH.md, `tools/radar_truth_eval.py --no-gate`; the recordings carry no pose, flags or
vectors): base: wrong verdict 3.1 of 94.0 judged s, real things never followed 8, `in_path` unreviewed 4.73 s; rows:
3.5 of 93.0 s, 9, 4.73 s (parked cars called stationary 32.8 s against 34.5 s, moving 1.6 s against 1.2 s);
+shared: 3.5 of 93.3 s, 9, 4.73 s; +shared (first build): 2.7 of 93.3 s, 9, 4.94 s; with `sil_bearing`: 1.1 of
92.2 s, 9, 4.90 s. The two night drives do not show the simulator's gain: nothing there is better with the rows set. One knob at a time on
`ctx_prior`: `row_ctx` as the base; `det_chain` 3.5 of 93.7 s, 8; `pd_learn` 1.9 of 90.9 s, 9, 4.42 s;
`exist_shared` 3.1 of 92.4 s, 10.

What it says:
* **The tracker says more of the truth about a parked row**: a quarter fewer wrong-car updates on today's stream, a
  third fewer jumps, parked cars called moving in 8.5 % of their frames against 14.6 %, known to stand in 75 %
  against 53 %, half a second sooner, with fewer id switches and no loss on moving traffic. With the appearance
  ratio on the realistic stream the wrong-car updates halve (8.6 -> 4.9 %), with the pose head they fall to 4.0 %
  (of settled objects: 6.7 -> 2.6 %), and the moving traffic gains (810 s called moving against 700 s, heading
  0.054 against 0.081 rad).
* **What separates the cars of a row is the bearing, not the range**: `sil_bearing` on the plain stream (12.0 ->
  7.2 % with the rest) and the pose head's centre on the realistic one do what no prior does. The range stays what
  the boxes hold (0.16-0.17 of the range).
* **The false warnings do not follow on today's stream** (+10 +- 8 on plain; -6 +- 8.5 on the realistic one; with the
  pose head -14 +- 7). The jumps fall (26 -> 22); what rises is a car now known to stand, sooner and further out, on
  the outside of a bend, whose honest range error puts its mean in the lane (`curve_parked` 22 -> 30-35: "mean in
  lane" 7 -> 14, "spread" 1 -> 5). That is the decision layer reading a needle along the line of sight as a place in
  the lane; `std_rho` and `path_normal` change nothing there today (the decision layer does not read them: identical
  episodes). `sil_bearing` and `size_learn` make it worse on the plain stream (+21 +- 10, +17 +- 9).
* **Night**: `exist_shared` is what confirms (never confirmed 16 -> 11 of 30 night runs, warned 12 -> 14 of 27) and
  it costs identities in moving traffic (id switches 189 -> 228 against the set without it). The detector's own
  boxes are the larger part of the night gap: by day the third box comes at 43 m, at night at 26 m.

Proposed as defaults: **rows** (`ctx_prior`, `row_ctx`, `det_chain`, `pd_learn`); `exist_shared` where night
confirmation matters more than identities in traffic; with the detector's options on `in_app`, and with the pose head
`in_pose_gn`, `in_pose_gc`, `in_pose_yaw`. Not `sil_bearing`, `size_learn` (the alerts), not the low list.

**The defaults were not switched.** With the four knobs on by default, 14 tests of the suite fail that pass with
them off, and they are not all recordings: `tests/test_decide.py::Simulated::
test_traffic_beside_and_against_the_car_does_not_warn` (`row_ctx`: an oncoming car's warning mass 0.07 where it must
be 0), `tests/test_radar.py::PastAndStructure::test_a_row_holds_its_members_in_line_and_standing_but_not_one_that_
pulls_out` (`row_ctx`, and `det_chain` + `pd_learn`), `...::Seeing::test_without_the_visibility_model_the_hidden_
object_is_lost` and `tests/test_road_users.py::RoadUsersInTheStack::test_the_pedestrian_is_a_display_object_not_a_
track` (`det_chain` + `pd_learn`), `...::ParkedCars::test_a_pitching_camera_does_not_make_parked_cars_drive` (all
four together). `ctx_prior` alone fails three, all of them tests that compare "default" with "knob on" or pin a
default's digest (`test_radar_parked.py::StandPrior`, `test_radar_smooth_knobs.py`). Each of the behaviour tests is
a finding to be looked at before a default changes; none was loosened.

Cost (`tools/cost_bench.py`, the radar per road frame, three interleaved runs on a shared machine): dense simulated
capture: base 2.18-2.40 ms, this code with the knobs off 2.38-2.40 ms, with the four "rows" knobs on 2.36-2.43 ms;
recorded capture 0.15-0.22 / 0.10-0.21 / 0.13-0.22 ms: nothing beyond the runs' own scatter (the row context is one
vectorised pass over the scan's boxes and the judged objects; the chain and the learnt table are a few numbers per
leaf). With the knobs off every stream of both captures is identical to the base's (`cost_bench.py diff`). The pose
knobs and `in_app` were not costed through the stack: the captures carry neither. Deeper trees were measured and not
taken (no gain), so nothing was added to the hypotheses' own cost.

## 11. The range and the extent of a standing thing (2026-11-24)

Five knobs of `radar.tracker`, all **off by default** (with them off every scan and every message is what it was: the
goldens and digests of tests/test_native.py, tests/test_radar_inputs.py, tests/test_radar_parked.py and
tests/test_radar_rows.py stand as recorded; both recorded captures and both simulated ones of tools/cost_bench.py:
every stream identical). Recorded with them on: `tests/data/make_radar_golden.py --range` (tests/test_radar_range.py).
The simulator got two options, both off by default and bit-identical then: the detector's scale rate as a preset
(`scale`, `detin_scale`, `detin_pose_scale` of `roaduser_sim.det_params_arg`; `scale_rate: "auto"`: `SCALE_DAY` /
`SCALE_NIGHT` by the run), and a fleet of vehicle sizes (world option `dims_fleet`: `DIMS_FLEET`, cars 1.42-1.90 m
tall, mean 1.57 m, 1.63-2.0 m wide).

### What was found

Tools: the lane's scratch tools on tools/radar_trace.py (every 0.1 s every confirmed object with four sightings or
more against the actor whose box it last took; `curve_parked` + `parked_street`, seeds 1-3, day and night: 10 minutes
of driving per row) and tools/alert_truth.py.

* **The bias is the height prior, per car, and half of it something else.** Objects fed by one car, 10-40 m, today's
  stream: range +8.4 % long, the height state +7.0 % (the prior's 1.6 m against 1.5 m, hardly learnt). With the
  simulator's cars made 1.6 m: +4.1 % and +3.2 %: the prior's mean is half of it; the other half was not traced (it
  is gone with a scale rate, below). On a fleet of sizes the error is each car's own: cars under 1.55 m +10.5 to
  +10.8 %, 1.55-1.70 m +1.6 %, 1.70-2.2 m -2.3 %. No constant is right; the height has to be the object's.
* **None of the measured streams carried the detector's scale rate.** `--lights-ru-scale` is on in the car; the plain
  stream and the `DETIN` preset emit no `scale_rate` unless given a `scale_rate` dictionary (section 8's rows had
  one; the alert measurements of sections 9, 10 and DESIGN.md 17.2 had none). A thing that stands comes nearer at
  the car's own speed: its image grows at (the car's speed) / (its depth), so the rate measures the depth outright,
  to 2-4 % from one value at 30 m by day, whatever the thing's size, the camera's pitch and the road's slope; and the
  box height at that range is the object's own height from then on. With the stream's scale rate and no change to
  the tracker (`use_scale_rate` is on): the standing objects' median error along the line of sight 1.91 / 2.69 /
  3.51 m at 10-20 / 20-30 / 30-40 m -> 0.44 / 0.64 / 0.89 m, the stated spread 0.17 of the range -> 0.05, the bias
  +9 to +12 % -> -1 to +2 %, on the fleet +0.8 / +2.9 / +0.4 / -2.9 % by the four height classes, standing things
  called moving 14.3 % of their frames -> 5.0 %, called stationary 63 -> 84 %, jumps 101 -> 47. And the stated spread
  no longer held: within 1 / 2 sigma 0.65-0.76 / 0.85-0.92 where it was 0.59-0.79 / 0.87-0.97. (The other half of
  section 9's finding stands: the simulator's detector boxes a near car 2-3 times under 25 m; with the range pinned
  earlier that no longer matters to the range.)
* **The false warnings did not follow** (false `obstacle_ahead` episodes of the streets: 58 without and 75 with the
  scale rate on today's stream, 89 and 86 on the realistic one; today's stream draws the rates from the boxes' own
  generator, so its two rows are other noise realisations and a six-seed total has a spread of 11-16): they are said
  of the objects the tracker has NOT settled. Of
  the 58-75 false episodes 35-56 are on a track believed standing with less than one half, 2.5 s old (median), 25 m
  away, its range 3.2 m off; the well-ranged settled tracks, nearly all track-frames, raise none. What is left is
  association (section 10), and three defects in how the scale rate was read, which the range of exactly those
  objects hangs on:
* **One linear step on a hyperbola.** The rate is v / r. A Kalman update linearised at a range r0 lands at r0 (2 -
  r0 / r) for a true range r (0.84 r from 0.6 r) and states the spread the rate has at r0: (r0 / r)^2 of the right
  one. Traced (`curve_parked`, seed 5, the parked van 109, 2.6 m tall, boxed `car`): first box read at 0.6 of its
  range; after its first scale rate (0.219 /s at 13 m/s: 59.3 m; truth 59.4 m) "it stands" stood at 34.7 m +- 0.4 m;
  the next rate refuted it by tens of sigma; it went to the floor, was given the place of the hypotheses that DRIVE
  at every box after (where no standing thing gives that rate) and was refuted again each time: the object was a car
  driving at 5 -> 2.5 m/s at 0.6-0.8 of its range down to 6 m. (A standing thing at r and a thing at k r closing k
  times slower give the same picture at every moment; nothing in the object's own boxes takes that story back.)
* **The rate was read as the depth's rate of this moment.** It is the logarithm of the depth a baseline ago (0.2-1 s,
  along the camera's axis as it pointed then) over the depth now. On a bend of 110 m radius at 13 m/s the axis swings
  4 degrees in 0.6 s: a standing thing reads +0.3 to +0.8 m far at 5-60 m (noise-free boxes and rates: `curve_parked`
  +0.29 / +0.38 / +0.56 / +0.76 m at 5-10 / 10-20 / 20-40 / 40-60 m). Braking at 3 m/s^2 the car was 1.2 m/s faster
  on average over 0.8 s than at its end: a standing car reads 3 m near at 20-40 m with a stated spread of 0.5-1.3 m.
* **The picture's size goes with the depth along the camera's axis**, the model had the range along the line of
  sight less the rectangle's half depth along it: a car parked 3 m beside reads 0.05-0.1 m far. Small.
* **"Two cars' boxes in one object" was one car born as a truck.** Of the standing cars' track-frames at 5-60 m,
  4.2 % (337 of 8082) are wider than 2.3 m or longer than 6.5 m; 114 of them are of objects that only ever took one
  car's boxes. They are objects started by a box the detector called `truck`
  (3 % of a car's boxes; a van's two in three): born 10 x 2.5 m, and nothing takes that back: the class vote moves
  the HEIGHT prior (section 4), not the extent, and a camera sees of the extent a sum of length and width at a
  heading it does not know either. (The simulator's detector never merges two cars into one box.) Objects fed by
  several cars are wider by a little (width error p90 +0.28 to +0.47 m against +0.10 m). With the range known to a
  few tenths of a metre the centre's error IS the extent's: a van read 10 m long for its 6 m has its centre 2 m
  behind where it is with a stated spread of 0.3-0.7 m (of the 552 standing track-frames at 5-20 m whose centre is
  off by more than two stated sigma, 214 are vans; the cars among the rest include those with a truck's extent and
  those whose length the silhouette had shrunk to 2-3 m). The near face (`x_m`, `y_m`: what the boxes measure) was
  right within its spread throughout (within 1 / 2 sigma 0.67-0.84 / 0.92-0.98).

### The knobs

* `scale_iter`: the standing hypothesis takes a box with a scale rate by iteration (the iterated extended Kalman
  update: Gauss-Newton on the quadratic form the linear step minimises once; at most `scale_iter_n` 8 steps, each
  held to a factor of two in range; the covariance from the linearisation at the result; native `r_take_iter`). The
  hypothesis' weight is what the linear step's innovation gave it, as before.
* `scale_exact`: a scale rate is the depth then over the depth now: per scan the car's pose at the far end of each
  baseline, from the record of its movement (`Tracker.process`: at most eight baselines a scan, the scan's constants'
  tail; native `r_scale_row` with `sp`, `RSens.sprev`); the object's place then is its place now less its velocity
  times the baseline. A baseline whose pose is not given is read as before.
* `near_axis`: the box height's and the scale rate's depth is the centre's depth less the rectangle's half depth
  along the camera's axis (`r_h`).
* `ext_size`: a vehicle's length and width priors follow its height (`Tracker._ext_size`,
  `models.extent_of_height`: 4.5 x 1.8 m to 1.75 m, 6 m at 2.6 m, 9 m at 3.4 m; spread 12 % to 30 %): the class
  vote's height until the object's numbers know its own to `ext_size_learnt` (a quarter) of that prior's variance,
  then its own (a height not learnt moves with every box's noise and the hypotheses' weights); where
  that says another size by `ext_size_step` the priors are exchanged in every story (native `radar_ext_prior`: the
  old one's information out, the new one's in, as `_size` does with the height), and the centres move along the
  line of sight by the change of the half depth: the face the camera measured stays.
* `scale_seed` (built, measured, **not recommended**): "it stands", brought back from the floor, is opened along the
  common factor of range and size (`scale_seed_sigma` 0.7; native `r_reopen`) and takes the box's scale rate from
  there, iterated: it stands at the range THIS box's rate says. With the update iterated the hypothesis no longer
  falls to the floor by its own error; what the seed then does is make an object that took a neighbour's box stand
  again at the neighbour: +8 +- 3.0 false episodes against `scale_iter` alone on today's stream with the scale rate
  (jumps 18 -> 23), nothing in the conflict runs.
* The simulator's options above. Scale rates whose errors are shared (the realistic stream's) want `in_scale`
  (section 8) beside these.

### Measured (paired: the same seeds and runs for every row)

Knob sets: **I** = `scale_iter`; **IX** = I + `scale_exact`; **C** = IX + `near_axis` + `ext_size` (with `in_scale`
on the realistic stream); **C+rows** = C + `ctx_prior` + `row_ctx` + `det_chain` + `pd_learn` (with `in_app` on the
realistic stream).

Standing actors of `curve_parked` + `parked_street`, seeds 1-3, day + night (12 runs, 10 minutes of driving per row),
every confirmed object with four sightings or more against the centre of the actor whose box it last took; bands
10-20 / 20-30 / 30-40 m. "fleet": the world option `dims_fleet`.

| stream, knobs | median error along the line of sight (m) | bias % | spread / range | centre within 1 / 2 sigma | called moving / stationary | updates with another car's box | jumps over 3 m and 2 sigma (at such an update) |
|---|---|---|---|---|---|---|---|
| today's, no scale rate | 1.91 / 2.69 / 3.51 | +8.9 / +11.9 / +10.2 | 0.17 / 0.17 / 0.18 | 0.61-0.76 / 0.89-0.97 | 0.143 / 0.63 | 12.0 % | 101 (53) |
| + scale rate, base | 0.44 / 0.64 / 0.89 | +2.3 / +1.7 / -0.6 | 0.050 / 0.053 / 0.059 | 0.69-0.76 / 0.89-0.92 | 0.050 / 0.84 | 9.8 % | 47 (22) |
| I | 0.42 / 0.61 / 0.81 | +2.5 / +1.3 / -0.1 | 0.049 / 0.051 / 0.054 | 0.69-0.78 / 0.90-0.95 | 0.034 / 0.88 | 9.5 % | 45 (30) |
| IX | 0.34 / 0.51 / 0.74 | +2.0 / +0.8 / -0.8 | 0.052 / 0.053 / 0.056 | 0.74-0.82 / 0.91-0.95 | 0.032 / 0.89 | 9.5 % | 41 (26) |
| C | 0.29 / 0.46 / 0.67 | +1.4 / +0.3 / -1.2 | 0.053 / 0.053 / 0.057 | 0.82-0.84 / 0.95-0.96 | 0.045 / 0.87 | 10.0 % | 54 (30) |
| the rows set alone | 0.48 / 0.66 / 0.86 | +2.1 / +0.6 / -1.2 | 0.049 / 0.049 / 0.053 | 0.67-0.76 / 0.89-0.93 | 0.035 / 0.91 | 8.0 % | 30 (13) |
| C+rows | 0.29 / 0.41 / 0.59 | +0.9 / -0.2 / -1.9 | 0.050 / 0.049 / 0.052 | 0.82-0.85 / 0.95-0.97 | 0.024 / 0.93 | 8.3 % | 31 (20) |
| realistic, no scale rate | 1.85 / 2.75 / 3.25 | +6.1 / +4.8 / +3.4 | 0.18 / 0.18 / 0.18 | 0.65-0.81 / 0.90-0.96 | 0.122 / 0.63 | 8.6 % | 56 (26) |
| + scale rate, base | 0.50 / 0.89 / 1.32 | +1.7 / -0.1 / -3.1 | 0.061 / 0.062 / 0.065 | 0.71-0.74 / 0.90-0.92 | 0.073 / 0.80 | 8.3 % | 22 (11) |
| I | 0.47 / 0.85 / 1.22 | +2.0 / +0.2 / -1.9 | 0.059 / 0.061 / 0.061 | 0.72-0.76 / 0.92-0.93 | 0.060 / 0.83 | 8.4 % | 24 (14) |
| C | 0.40 / 0.63 / 1.07 | -0.1 / -1.7 / -3.7 | 0.067 / 0.068 / 0.066 | 0.81-0.83 / 0.94-0.96 | 0.067 / 0.83 | 8.0 % | 29 (15) |
| C+rows | 0.38 / 0.57 / 0.99 | +0.6 / -0.9 / -2.8 | 0.063 / 0.065 / 0.063 | 0.82-0.85 / 0.96-0.98 | 0.021 / 0.89 | 4.8 % | 15 (7) |
| today's + scale rate, fleet, base | 0.55 / 0.71 / 1.06 | +2.4 / +0.9 / -1.7 | 0.050 / 0.053 / 0.058 | 0.62-0.73 / 0.86-0.90 | 0.089 / 0.81 | 10.0 % | 43 (20) |
| C | 0.41 / 0.53 / 0.78 | +1.6 / +0.2 / -1.2 | 0.051 / 0.052 / 0.056 | 0.76-0.82 / 0.94-0.95 | 0.055 / 0.85 | 9.9 % | 56 (28) |
| realistic + scale rate, fleet, base | 0.59 / 0.84 / 1.33 | +3.2 / +0.8 / -0.7 | 0.059 / 0.057 / 0.061 | 0.63-0.74 / 0.89-0.93 | 0.075 / 0.80 | 8.5 % | 51 (27) |
| C | 0.43 / 0.64 / 1.02 | +2.2 / +0.6 / -1.6 | 0.067 / 0.061 / 0.062 | 0.78-0.85 / 0.95-0.96 | 0.058 / 0.84 | 8.0 % | 40 (23) |

(The jumps are counted against the stated spread, which the knobs make smaller and `ext_size` moves centres on
purpose; they are not fewer for the range's sake. Fewer come with the rows set.) By day, today's stream with the scale
rate, C against the base: 0.19 / 0.29 / 0.48 m against 0.30 / 0.44 / 0.72 m, within 1 / 2 sigma 0.87-0.91 / 0.96-0.97
against 0.75-0.80 / 0.91-0.93; at night 0.84 / 1.33 / 2.39 m against 0.94 / 1.65 / 2.48 m, 0.67-0.71 / 0.92-0.95
against 0.59-0.64 / 0.85-0.91 (the night's rates are five times coarser, with half as many track-frames).

On the fleet, objects fed by one car, 10-40 m, by the car's true height (under 1.45 / 1.45-1.55 / 1.55-1.70 /
1.70-2.2 m), the range's bias: no scale rate +10.5 / +10.8 / +1.6 / -2.3 %; with it +0.8 / +2.9 / +0.4 / -2.9 %
(within 1 sigma 0.69-0.85); C +0.8 / +1.9 / +0.1 / -1.3 % (within 1 / 2 sigma 0.81-0.95 / 0.98-1.00). The size is
the object's own once a rate has ranged it; no kind's constant enters.

Extent (the same frames, cars): wider than 2.3 m or longer than 6.5 m 2.2 % of the frames with the scale rate, base
-> 0.0-0.4 % with C; the length's error at its 90 % point +0.31 to +0.40 m -> +0.11 to +0.18 m. `sim.radar_score`
(the 11 scenarios x seeds 1-3 x day and night, today's stream with the scale rate; base -> C): length error p50 / p90
0.54 / 2.05 -> 0.22 / 0.95 m in traffic, 0.26 / 2.19 -> 0.09 / 0.61 m in the parked streets, width p90 0.30 ->
0.21 m there; moving actors called moving / stationary 519 / 13.9 s -> 534 / 15.3 s, standing ones called
stationary / moving 897 / 48 s -> 944 / 40 s, id switches 163 -> 165 (+2 +- 20), time held and confirmed as before;
on the realistic stream 529 / 11.6 -> 529 / 12.1 s, 639 / 42 -> 672 / 35 s, id switches 128 -> 120 (-8 +- 12),
length p90 2.09 -> 1.08 m in traffic. What is not fixed: the range of an object NOT believed to stand (it is the
height prior's under "it drives": at 30-60 m on the realistic stream those tracks read 15-16 % short, vans and
trucks 13 %, while cars believed standing read -0.2 to -1.7 %); and with the scale rate the realistic stream's
standing objects at night beyond 30 m (few frames) read 11-14 % short with or without the knobs.

The obstacle of the vehicle conflicts (`alert_truth.py run --trace`, eight scenarios x seeds 1-6, by day; its track
by the time to contact; base -> IX): today's stream with the scale rate, the range's median error 0.93 -> 0.54 m at
4-3 s, 0.89 -> 0.34 m at 3-2.5 s, 0.54 -> 0.23 m at 2-1.5 s, 0.32 -> 0.17 m in the last second, its mean +0.46 to
+0.57 m long -> +0.16 to +0.23 m, within 1 sigma 0.74-0.91 -> 0.91-1.00; on the realistic stream 1.17 -> 1.17 m,
0.85 -> 0.66 m, 0.64 -> 0.37 m, 0.29 -> 0.18 m, within 1 sigma 0.81-0.94 -> 0.94-0.98. At night nothing changes
(it is tracked in 5-55 % of the frames: the first box and the confirmation, section 10).

Through the stack (shadow mode; tools/alert_truth.py: the six streets and the two pedestrian controls x seeds 1-6
x day and night, 96 runs; the conflicts x seeds 1-3 x day and night, of which 48 runs with a vehicle; paired, +- the
bootstrap's spread over seeds). "imminent": false episodes that went on to imminent; "not standing": false episodes
said of a track believed standing with less than one half; "lost / gained / later / earlier": the vehicle conflicts'
warnings against the base of the same stream.

| stream, knobs | false episodes (day / night), paired difference | false on other things in the conflict runs | vehicle conflicts warned, mean lead | half in the lane warned, lead | lost / gained / later / earlier | imminent | not standing | causes: jump / mean in lane / spread / moving futures |
|---|---|---|---|---|---|---|---|---|
| today's + scale rate, base | 75 (31 / 44) | 20 | 33 of 48, 2.09 s | 7 of 12, 1.90 s | | 40 | 56 | 22 / 13 / 7 / 19 |
| I | 59 (23 / 36), -16 +- 2.6 | 17, -3 +- 2.1 | 33, 2.01 s | 7, 1.82 s | 0 / 0 / 7 / 1 | 25 | 41 | 18 / 7 / 4 / 20 |
| IX | 62 (23 / 39), -13 +- 4.8 | 18, -2 +- 2.8 | 33, 2.05 s | 7, 1.82 s | 0 / 0 / 6 / 2 | 32 | 46 | 15 / 9 / 6 / 22 |
| C | 71 (27 / 44), -4 +- 6.3 | 17, -3 +- 2.5 | 33, 2.06 s | 7, 1.82 s | 0 / 0 / 6 / 3 | 29 | 48 | 18 / 8 / 7 / 22 |
| the rows set alone | 55 (29 / 26), -20 +- 7.3 | 18, -2 +- 2.8 | 34, 1.95 s | 8, 1.82 s | 1 / 2 / 6 / 1 | 31 | 38 | 14 / 12 / 5 / 12 |
| C+rows | 55 (20 / 35), -20 +- 5.9 | 14, -6 +- 3.6 | 32, 2.05 s | 7, 1.96 s | 1 / 0 / 7 / 3 | 29 | 35 | 18 / 6 / 7 / 12 |
| realistic + scale rate, base | 86 (42 / 44) | 28 | 37 of 48, 1.97 s | 8 of 12, 2.15 s | | 58 | 52 | 21 / 21 / 10 / 22 |
| I | 85 (38 / 47), -1 +- 6.1 | 22, -6 +- 4.4 | 38, 1.89 s | 8, 1.93 s | 0 / 1 / 7 / 2 | 59 | 43 | 25 / 22 / 8 / 18 |
| IX | 79 (39 / 40), -7 +- 4.6 | 23, -5 +- 5.1 | 38, 1.91 s | 8, 1.98 s | 0 / 1 / 8 / 6 | 55 | 36 | 23 / 20 / 9 / 12 |
| IX + `in_scale` | 76 (37 / 39), -10 +- 3.4 | 23, -5 +- 3.9 | 38, 1.94 s | 8, 2.09 s | 0 / 1 / 7 / 9 | 54 | 37 | 20 / 23 / 7 / 13 |
| C | 76 (39 / 37), -10 +- 4.0 | 23, -5 +- 4.6 | 38, 1.91 s | 8, 1.96 s | 0 / 1 / 8 / 8 | 54 | 37 | 17 / 22 / 10 / 15 |
| the rows set alone (`in_app`, `in_scale`) | 80 (41 / 39), -6 +- 5.9 | 25, -3 +- 5.9 | 38, 1.94 s | 8, 2.04 s | 0 / 1 / 5 / 7 | 47 | 36 | 19 / 25 / 10 / 18 |
| C+rows | 78 (38 / 40), -8 +- 4.8 | 19, -9 +- 3.4 | 40, 1.85 s | 8, 2.01 s | 0 / 3 / 9 / 8 | 54 | 34 | 18 / 19 / 13 / 15 |
| today's + scale rate, fleet, base | 91 (41 / 50) | 32 | 36 of 48, 2.04 s | 7 of 12, 1.93 s | | 53 | 66 | 16 / 20 / 6 / 31 |
| I | 75 (30 / 45), -16 +- 4.9 | 25, -7 +- 2.6 | 36, 2.06 s | 7, 2.09 s | 0 / 0 / 2 / 2 | 50 | 53 | 15 / 13 / 7 / 22 |
| IX | 81 (31 / 50), -10 +- 5.3 | 31, -1 +- 4.4 | 37, 2.10 s | 8, 1.94 s | 0 / 1 / 2 / 5 | 52 | 59 | 21 / 15 / 7 / 23 |
| C | 82 (32 / 50), -9 +- 5.5 | 29, -3 +- 3.2 | 37, 2.07 s | 8, 1.98 s | 0 / 1 / 2 / 6 | 52 | 59 | 21 / 13 / 6 / 27 |
| C+rows | 62 (29 / 33), -29 +- 10.0 | 24, -8 +- 2.6 | 35, 2.20 s | 7, 2.09 s | 1 / 0 / 4 / 9 | 36 | 38 | 9 / 17 / 8 / 16 |
| realistic + scale rate, fleet, base | 81 (36 / 45) | 29 | 37 of 48, 2.02 s | 8 of 12, 1.94 s | | 55 | 45 | 19 / 19 / 9 / 18 |
| I | 73 (33 / 40), -8 +- 4.8 | 30, +1 +- 1.7 | 37, 2.09 s | 8, 2.07 s | 0 / 0 / 4 / 4 | 44 | 39 | 16 / 14 / 11 / 20 |
| IX | 75 (34 / 41), -6 +- 3.8 | 27, -2 +- 1.3 | 37, 2.11 s | 8, 2.06 s | 0 / 0 / 3 / 5 | 44 | 37 | 12 / 15 / 16 / 20 |
| C (`in_scale`) | 76 (37 / 39), -5 +- 5.4 | 27, -2 +- 1.3 | 36, 2.10 s | 7, 2.27 s | 1 / 0 / 5 / 6 | 43 | 33 | 20 / 14 / 15 / 14 |
| C+rows | 85 (40 / 45), +4 +- 6.3 | 28, -1 +- 2.2 | 36, 2.22 s | 7, 2.29 s | 1 / 0 / 5 / 8 | 55 | 35 | 14 / 27 / 18 / 15 |
| today's, no scale rate, base | 58 (21 / 37) | 21 | 32 of 48, 1.65 s | 7 of 12, 1.29 s | | 31 | 35 | 30 / 7 / 1 / 6 |
| IX | 58, every run's episodes as they were | | | | | | | |
| `ext_size` | 63 (26 / 37), +5 +- 7.9 | 20, -1 +- 2.6 | 32, 1.68 s | 7, 1.39 s | 0 / 0 / 1 / 5 | 37 | 41 | 29 / 9 / 3 / 8 |
| `ext_size` + `near_axis` | 65 (27 / 38), +7 +- 8.1 | 20, -1 +- 2.2 | 32, 1.68 s | 7, 1.38 s | 0 / 0 / 2 / 4 | 40 | 43 | 28 / 8 / 3 / 10 |
| realistic, no scale rate, base | 89 (40 / 49) | 20 | 36 of 48, 1.76 s | 8 of 12, 1.61 s | | 60 | 42 | 30 / 18 / 9 / 14 |
| IX | 89, every run's episodes as they were | | | | | | | |
| `ext_size` | 84 (34 / 50), -5 +- 3.7 | 22, +2 +- 1.9 | 38, 1.72 s | 8, 1.48 s | 0 / 2 / 5 / 5 | 59 | 44 | 33 / 13 / 8 / 13 |
| `ext_size` + `near_axis` | 89 (37 / 52), +0 +- 4.0 | 21, +1 +- 2.2 | 38, 1.71 s | 8, 1.49 s | 0 / 2 / 7 / 5 | 64 | 48 | 37 / 13 / 9 / 12 |

Over the four streams with a scale rate together (333 false episodes at the base; the bootstrap over seeds across
all four): I -41 +- 10.6, IX -36 +- 11.2, C -28 +- 11.4, C+rows -53 +- 18.1. The vehicle conflicts with seeds 1-6 (96
runs a stream), IX against the base: today's stream with the scale rate 70 -> 71 warned, mean lead 2.04 -> 1.98 s,
2 lost (`bend_stopped_13` at night, seeds 5 and 6, 0.91 and 1.35 s before) and 3 gained, 9 later and 9 earlier,
false episodes on other things 30 -> 22 (-8 +- 5.1); realistic (with `in_scale`) 73 -> 75, 1.99 -> 1.97 s, none lost
and 2 gained, 9 later and 14 earlier, 36 -> 28 (-8 +- 3.6).

What it says:
* **Read right, the scale rate gives a standing vehicle's range to 0.2-0.5 m at 10-40 m by day** (0.8-2.4 m at
  night) where the box height and the ground angle gave 1.5-3.5 m; the object's size is its own, whatever its kind
  (the fleet's four height classes within 2 %); the stated spread holds again (within 1 / 2 sigma 0.82-0.85 /
  0.95-0.97 with C, the near face throughout); the obstacle ahead is known to 0.2-0.3 m in its last two seconds.
* **The false warnings fall by an eighth with the iterated update alone** (-41 +- 10.6 of 333 over four streams;
  none of the 4 x 48 vehicle conflict warnings lost, mean lead -0.08 to +0.07 s), on today's stream clearly (-16
  +- 2.6, -16 +- 4.9), on the realistic one within the noise (-1 +- 6.1, -8 +- 4.8). Those that go on to imminent
  fall on today's stream (40 -> 25) and on the realistic fleet (55 -> 44), not on the other two.
* **`scale_exact` changes no count** (+5 +- 8 against I over the four streams) and takes the later warnings back
  (later / earlier 7 / 1 -> 6 / 2, 7 / 2 -> 8 / 6, 2 / 2 -> 2 / 5, 4 / 4 -> 3 / 5): the obstacles of the bends were
  read far.
* **`ext_size` and `near_axis` do not lower the warnings** (C against IX: +9 +- 6.0, -3 +- 4.7, +1 +- 3.3, +1 +-
  4.2; without a scale rate +5 +- 7.9 and -5 +- 3.7), and C loses one half-in-the-lane warning of the realistic
  fleet (0.54 s before, at night). They are for the extent and the centre (table above), not for the alerts.
* **The rows set of section 10 now helps on today's stream with the scale rate** (alone -20 +- 7.3; with C -20 +-
  5.9 and -29 +- 10.0 on the fleet; one vehicle warning lost in each, none gained, the false episodes on a track
  not believed standing 56 -> 35) **and not on the realistic one** (-6 +- 5.9 alone, -8 +- 4.8 and +4 +- 6.3 with C).
  Without a scale rate it was +10 +- 8 and -6 +- 8.5.
* **What is left is association.** With every set, 33-59 of the 55-85 false episodes are on a track believed
  standing with less than one half: an object fed by two cars of a row, or a van whose "it stands" needs a height
  its kind's prior calls unlikely, read at the range a car driving with the traffic would have. A range that
  resolves at the object's second box does not reach them; and a decision that waited for the range (DESIGN.md
  17.2, `place_hold`: "the spread 9 % of the range") would now wait for a spread of 3-5 % that comes with the second
  or third box, not with the distance: not tried here.

Real drives (docs/RADAR_TRUTH.md, `tools/radar_truth_eval.py --no-gate`; the recordings carry no scale rate): the
three scale knobs: identical (wrong verdict 3.1 of 94.0 judged s, real things never followed 8, `in_path` unreviewed
4.73 s); `near_axis`: the same counts; `ext_size`: 2.3 of 90.2 s, never followed 9, 4.42 s (parked cars called moving
0.4 s against 1.2 s); C: 2.3 of 90.5 s, 9, 4.42 s; C+rows: 3.5 of 93.0 s, 9, 4.94 s. What the knobs do with real scale
rates is not known: no recording has them.

Cost (this laptop; a phone's small core is 10 to 16 times slower). One scan of the tracker (`Tracker.process`,
`parked_street`, three interleaved runs): today's stream with the scale rate 0.80-0.84 ms with the knobs off, 0.84-
0.85 ms with IX, 0.92-0.94 ms with C (`ext_size` and `near_axis` cost nothing measurable alone, 0.83-0.85 ms; together
with the scale knobs 0.1 ms: not traced); realistic 0.58-0.59 / 0.59-0.60 / 0.65-0.67 ms. Through the stack
(tools/cost_bench.py, the radar per road frame, three interleaved runs; the captures carry no scale rate): dense
simulated capture 0.983 ms at the base, 0.991-0.999 ms with the knobs off, 0.975-1.016 ms with C; recorded capture
0.093 / 0.092-0.096 / 0.095-0.097 ms. With the knobs off every stream of the three captures is identical to the
base's (`cost_bench.py diff`).

Proposed for the car (where the detector's scale rate is on): **`scale_iter` + `scale_exact`, with `in_scale`**
(the detector's successive rates share their baseline). `ext_size` where the centre and the extent matter more than
0.1 ms a scan; `near_axis` with it. Not `scale_seed`. The rows set of section 10 stays what it was: a gain on
today's stream, none shown on the realistic one, one conflict warning lost.

**The defaults were not switched.** What blocks `scale_iter` (and `scale_exact` with it): (1) on the realistic stream
with the scale rate the paired difference is not beyond its noise (I -1 +- 6.1 and -8 +- 4.8 on the fleet; IX -7 +-
4.6 and -6 +- 3.8; with `in_scale`, a second default, -10 +- 3.4): it lowers the warnings clearly on one of the two
streams only; (2) with seeds 1-6 IX loses two night warnings of `bend_stopped_13` on today's stream and gains three
(none lost with seeds 1-3 on any stream); (3) on the streams without a scale rate it changes no said episode of any
run (174 of 174 on both), yet with `scale_iter` on by default six tests fail beyond the known one: three recordings
(tests/test_native.py `test_radar_messages_digests`, tests/test_radar_inputs.py `Golden` and `Hypotheses::test_the_
rows_memory_is_written_down_and_restored_with_the_hypotheses`: a scan given the further inputs' constants is not the
scan without them to the last bit), `tests/test_radar_rows.py::Knobs::test_the_scan_constants_end_where_they_did_
with_the_knobs_off`, and two of this section's own that compare the knob with the default. No behaviour test among
them. `ext_size`: no gain in the warnings, one real thing more never followed on the recorded drives, one half-in-
the-lane warning lost on one stream. Nothing was loosened.

Not built: the rows' order and occupancy as a constraint; a track split where its residuals turn bimodal; a robust
update or a new object instead of a jump of a settled standing object (with the scale rate a settled standing
object's neighbour is three of its spreads away, and the jumps that remain are of objects that have not settled);
the scale rate's own link (a rate says that this box and one of the detector's boxes a baseline ago are one thing;
which box that was the message does not say: with it the cars of a row would be told apart by the detector's own
chain); a height prior with a van's tail (a van boxed `car` needs "it stands" to believe 2.6 m against
1.6 m +- 20 %); the size learnt per kind from the scale-ranged objects (`size_learn` of section 9 with the scale
rate and the rows set: 60 false episodes against 55 without it, nothing either way); the residual +0.1 to +0.4 m far
of a parked car on a straight road with noise-free boxes and rates (not traced).

## 12. An object that is no longer there (2026-11-27/28)

The first real drives with alerts in shadow (27 November, night, session `20261127-202650`; the day and dusk sessions
of 26 November replayed) raised `obstacle_ahead` on objects the car could not be about to hit. Traced on the
recordings (the stack re-run on them with the phone's knobs; `/home/piotr/scratch/stale/tools/`):

* **20:31:44-49, said imminent (the car's object 516).** A queue creeping at 2-3 m/s. The lead's boxes (wide view,
  cut at the bottom, the pose head's ground point 7.2 m +- 2.7 degrees) start an object that "stands" (0.98 after six
  boxes: at 1.8 m/s the car's own travel does not yet tell a lead that creeps with it from one that stands, and the
  scale rates that would went to other, one-box objects). From then on no box is its own: its predicted box grows as
  the car "approaches" it, the lead's does not, and the box feature and the gate give every box of the lead to a new
  object, which the standing object then hides (`r_visible`: `occ` by it, `pd` 0.01), so that the next box starts
  the next one. A miss costs the standing object ln(1 - 0.22) (below). It coasts 2.5 s into the car.
* **20:32:14-17, said caution (547).** A car in the lane to the right, its rear 8 m ahead, boxed in every wide scan
  (bearing -18 to -20 degrees, cut at the bottom). Its object's velocity has a lateral part (1-1.8 m/s) that is not
  the car's; the object leaves its boxes, its place opens to 4.4 m (1 sigma) within 1.2 s and 9.7 m within 2.4 s
  while it is put out 4 m ahead in the car's own lane, between the car and the van it follows; there it hides the
  van's object (one scan: `occ` by it).
* **20:30:53 and 20:33:07 (153, 843), imminent / caution under the low-speed suppression.** A queue starts. The
  standing lead's object (0.97 "it stands", existence at its cap) takes none of the lead's boxes from the first one
  that shows the start: the scale rate (-0.061 +- 0.007 1/s with the car at rest) and the box sliding put them
  outside the gate of the object as a whole, each starts a tentative object, and the lead is not an object at all
  for the next seconds. The car pulls away towards the standing one.

What the inputs are, measured on the six driving sessions against the road model's lead (probability >= 0.95/0.97):

| | wide view (rows 0-446) | narrow view (90-410) |
|---|---|---|
| first ground point of the view | 15.6 m | 30 m |
| share of scans with a box on a lead within 10 m, day / night | 0.87-0.91 / 0.88-0.94 | 0.75-0.80 / 0.39-0.45 |
| the same, 10-20 m | 0.90 / 0.83 | 0.61 / 0.30 |
| pose head's ground point of a cut box, read as a range, minus the lead's: lead < 4 m / 4-6 / 6-9 / 9-14 m | +2.7 / +2.5 / +0.4 / -1.1 m | +6.3 / +6.3 / +4.4 / +2.2 m |
| the box width as a range (1.8 m wide) minus the lead's, all distances | +0.7 to +0.9 m (quartiles 0.6 m apart) | +0.8 to +1.4 m |

Every vehicle within 15 m has a box cut at the bottom in both views. The sensor's model gives such an object the
chance `p_d` x `p_cut` = 0.22 of a box where the wide view boxes it in nine scans of ten: a miss there costs a
quarter of a nat, an object at the existence cap (12) outlives 5 s of misses (`seen_max_s` ends it), and a box is
worth little against "a new object". The ground point of a cut box is an extrapolation below the picture that
reaches 85-125 px below the wide view's cut and 50-95 px below the narrow one's (0.23 of the view's height), where
the wheels of a lead 4-5 m ahead stand 150-250 px below it: the two views state ranges 4-8 m apart for one vehicle,
both too far, with stated sigmas of 1-3 degrees. The tracked lead's near face is 2.9 m (median) beyond the road
model's lead at night and 1.3-2.6 m by day, and wanders +-3 m about it within seconds while it creeps.

### What is built (each a knob of `tracker.CFG`, off; with all off the stack's output is the base's, bit for bit)

* **`see_through`** (native `Hyp.evidence`, called after a camera's `Hyp.cycle`). The visibility model says of a
  thing behind a settled object that it is hidden; the scan's boxes are that statement read the other way. For a
  settled object A that the best story left without a box, and every first-list box D of the scan that is not A's:
  how much of D's box lies inside A's silhouette (across: five bearings of the box against A's rectangle as seen, as
  soft as A's bearing is unsure, here not capped; up: the rows of the box that A's box holds), and how sure the
  thing is the farther one (q: A's near face against the thing's as angles below the horizon, by more than
  `see_margin` sigma of what is not known of the two; the thing's from the object that took D, else from D's own
  ground row; a box without a range is not read). A box so far covered has the chance pv(cover) of the model's ramp
  (`occl_from` .. `occl_full`, no lower than `see_floor` 0.1: the detector does box what it sees a part of), so A's
  log odds take ln(1 - q (1 - pv)), every story alike (`h_exist_shift`), no more than `see_min` per scan. A is not
  touched in a scan that boxes it. Geometry counts as it is: a car 5 m ahead covers 7 degrees either side, also a
  van in the next lane 45 m on.
* **`swept`** (native `radar_swept`, `Hyp.shift`; from `Tracker.manage`). The car drives where nothing is. For a
  settled object without a box for `swept_unseen_s`: the probability p that its rectangle lies where the car's body
  is or has been since that box (front and rear bumper from the reference camera as the decision layer has them,
  half the width, drawn out behind by the car's travel since; `swept_margin_m` off; the place's own spread along and
  across the car). Its existence takes ln(1 - p) once, as p grows; from `swept_drop` (0.9) it is dropped, because
  what is left of it is not at the place its numbers state (they are not cut to the room left). How soon is the
  place's own spread: an object placed to 0.7 m goes as the bumper reaches its near face. Not for a tentative
  object: its place is three stories (stands, drives with the traffic, comes the other way) whose blur reaches into
  the car from 15 m away; with those included one night warning of the simulator's `bend_exit_stopped_13` was lost.
* **`coast_std_m`** (`Tracker.manage`, `radar_life`). Inside a sensor's field and in the open, an object that is
  believed to move, has had no box for `coast_min_s` (0.6 s) and whose place is known no better than `coast_std_m`
  and `coast_std_rel` (0.5) of its range (1 sigma, larger axis) is dropped. Out of view the rule has been
  `memory_std_m`, behind another object `hidden_std_m`; in view there was none, and the object's motion model was
  put out for 5 s. A standing object is not touched (its place is known as well as ever). With `coast_std_rel` 0.35
  the lead was without an object in 29 % of the night session's lead frames against 21 %; with 0.5 in 24 %.
* **`pd_cut`** (`Tracker.process`, `Hyp.evidence`). The chance of a box in a view's cut zone followed per view from
  the settled objects themselves: of those in the open there with a box within `pd_cut_fresh_s`, the share the scan
  boxes (held to the model with `pd_cut_n0` scans, at most `pd_cut_max`); `CameraBoxes.p_cut` carries it into the
  scan. Built and measured, not proposed (below).
* **`gate_hyp`** (native `ras_row`, the scan's constant `XC_GHYP`). A box outside the gate of the object as a whole
  but within `gate_hyp` gates, and lying on the object's last box, is inside where it is inside the gate of one of
  the object's hypotheses. Built and measured, not proposed (below).

The simulator (`sim/roaduser_sim.py`): `queue_creep` (stop and go behind a lead whose box is cut by both views the
whole time, a queue in the next lane), `left_behind` (a car beside that falls back out of the view while the car
follows a van) in `STALE_SCENARIOS`; `POSE["gn_cut_reach"]` (None in the presets) makes the ground point of a
view-cut box reach only so far below the cut: with (0.23, 0.035) the point of a lead 5 m ahead reads 1.8 m far in
the wide view and 5.6 m in the narrow one, as measured. Tests: `tests/test_radar_stale.py` (fixtures at the radar
box: a standing object that is no longer boxed with a lead beyond it, driven through, a car beside that is no longer
boxed, the cut zone), `tests/test_radar_stale_sim.py` (the scenarios, the reach).

### Measured

**The recordings** (six driving sessions re-run through the whole stack with the phone's knobs, `cost_bench.py run
--dump`; said = `obstacle_ahead` at caution or higher, not suppressed; a re-run is not the car's run: object ids
differ, and on `202650` the re-run says two cautions at 255.6 and 258.7 s that the car did not). The 47 said
episodes of the base judged one by one on the frames and the tracks, by cause:

| cause | said in the base | still said with `see_through` + `swept` + `coast_std_m` 3 |
|---|---|---|
| an object that is not there: left behind by its boxes, coasting beside or into the lane, driven through (`202650` 297 and 324 s, the car's 516 and 547; `152055` 367, 373, 378; `153449` 267, 592, 600, 1043) | 9 | 3 (`153449` 267, 592, 1043) |
| a vehicle in the next lane (a standing queue beside, a coasting neighbour predicted to cut in or turn) | 3 | 0 |
| a queue that starts or creeps: the lead's object stands on, or lags the lead that pulls away (`152055` 250, 567; `153449` 581, 947, 1458, 1460, 1778, 1870; on the car 20:30:53 and 20:33:07, suppressed) | 8 | 8 |
| a young object whose velocity is not known yet (far, at the roadside, crossing, cutting in) | 7 | 5 |
| the car turns or manoeuvres: parked and standing vehicles on the turn's exit, parked ones called moving (the car's 1118 among them) | 8 | 7 |
| an oncoming or a standing vehicle on a narrow street | 3 | 3 |
| the re-run's own two cautions on the coasting lead of `202650` | 2 | 2 |
| true or arguable: a slow or stopped lead the car closes on | 7 | 7 |

Per session, base -> with the three knobs: `113849` 2 -> 2, `114700` 7 -> 7, `152055` 12 -> 9, `153449` 21 -> 20
(three of them new: a young object in a car park, a standing object beside the lane, a parked car that turns
"moving" as the car passes), `195700` 0 -> 0, `202650` 5 -> 3. In total 47 -> 41; twelve of the base's are no longer
said, none of them a true one. The car's cases: 516 is no longer said (the standing object's existence falls with
the lead's boxes seen through it; the decision layer lets go of it at 296.5 s, 0.8 s before the low-speed
suppression lifts, and it is gone at 297.3 s, its near face 1 m ahead of the camera; it still reaches "imminent"
under the suppression: it is born standing, see below); 547 is gone (dropped at 322.65 s, 1.7 s before the alert
was raised, when its place was a guess); 153 and 843 are as they were. The road model's lead (within 20 m, probability >= 0.97) has no
object in 21 -> 24 % of its frames on `202650`, 19 -> 18 % on `152055`, 20 -> 25 % on `153449`: a lead that the
detector loses is coasted 2-3 s, no longer 5.

**The simulator** (`tools/alert_truth.py`, the phone's knobs as the base, paired over the same runs: six streets and
the pedestrian controls x seeds 1-6, the scripted conflicts x seeds 1-3, day and night; +- the bootstrap's 1 sigma):

| stream | knobs added | false episodes | conflicts warned | lost / gained / later / earlier |
|---|---|---|---|---|
| `detin_pose_scale` | `see_through` + `swept` + `coast_std_m` 3 | 87 -> 85 (-2 +- 2.4) | 42 -> 42 of 72 | 0 / 0 / 0 / 0 |
| `scale` | the same | 70 -> 67 (-3 +- 2.6) | 37 -> 37 | 0 / 0 / 0 / 1 (2.12 -> 2.30 s) |
| `detin_pose_scale` | `see_through` | 87 -> 86 (-1 +- 1.0) | 42 -> 42 | 0 / 0 / 0 / 0 |
| `detin_pose_scale` | `swept` | 87 -> 86 (-1 +- 1.0) | 42 -> 42 | 0 / 0 / 0 / 0 |
| `detin_pose_scale` | `coast_std_m` 3 | 87 -> 87 (0 +- 2.0) | 42 -> 42 | 0 / 0 / 0 / 0 |
| `detin_pose_scale` | the three + `pd_cut` | 87 -> 85 (-2 +- 2.4) | 42 -> 42 | 0 / 0 / 1 (1.78 -> 1.52 s) / 0 |
| `detin_pose_scale` | the three + `gate_hyp` 6 | 87 -> 86 (-1 +- 2.3) | 42 -> 41 | 1 (`bend_exit_stopped_13` s1 night, 0.88 s) / 0 / 0 / 0 |
| `queue_creep` + `left_behind`, the reach modelled, seeds 1-6 | the three | 25 -> 24 (-1 +- 1.7) | 6 -> 6 of 6 | 0 / 0 / 0 / 0 |

The simulator shows the queue's start as the car had it (`queue_creep`: two false said episodes a run, one at each
start, 24 in 12 runs; the standing lead's object stays 6-8 m behind the lead that pulls away, the car drives at it,
said as the speed passes 3 m/s) and none of the knobs changes it (22 with the three). The objects that the knobs end on the recordings (left by
their boxes, coasting beside) do not arise in the simulator's streets: there the three knobs are within the noise.

On the recordings `pd_cut` with the three (against the three alone) says eight episodes more on five sessions
(`202650` 243.7 and 250.1 s, `114700` four, `152055` one, and three of the ended ones come back): with the real
chance of a box the lead's object holds its boxes through the start, and what it then says of the lead is wrong by
the range and speed errors above. `gate_hyp` changes no said episode on five sessions.

Identities (the radar alone on the simulator, `sim/radar_score.py`, `detin_pose_scale`, eleven scenarios x seeds
1-3 x day and night, the three knobs against the phone's): id switches 131 -> 137 (paired +0.09 +- 0.06 a run),
held share 0.379 -> 0.373. The difference is `slow_lead` (3 -> 9 switches, held 0.56 -> 0.50; `adjacent` 11 -> 13,
`curve_parked` 35 -> 33, the other eight unchanged): a lead that the detector loses for more than 2-3 s comes back
as a new object where it used to keep its id through 5 s (`coast_std_m`).

Cost (`tools/cost_bench.py run`, the whole path over a session, three runs each, on the P52): the radar plug-in
0.190 -> 0.203 ms per road frame on `202650` and 0.447 -> 0.487 ms on `152055` (the whole path 2.03 -> 2.04 and
2.94 -> 3.01 ms): `Hyp.evidence` once per camera scan, `radar_swept` per message for the objects without a box. Not
measured on the phone. With the knobs off the three captures of `/home/piotr/scratch/cost-rest` (`cost_bench.py
diff`: IDENTICAL) and the six sessions (the dumps' sha256) are the base's.

### Proposed for the car, and what is not solved

In the phone's `radar.json`: **`"see_through": true, "swept": true, "coast_std_m": 3.0`**.

Not `pd_cut`: more said episodes on the recordings (above); it wants the range of a cut box first. Not `gate_hyp`:
no gain on the recordings, one night warning lost in the simulator.

Not solved, and why:

* **The queue's start.** With clean boxes the tracker lets "it stands" go 0.6 s after the lead starts (fixture: a
  scale rate at the car's rest) and 1.4 s without a scale rate. On the car and in `queue_creep` it takes 3-3.5 s,
  because for the first seconds there is nothing to read: the detector's miss run at the start (the simulator), or
  boxes that the standing object cannot take and whose ranges are wrong by metres (the car). The witness that knows
  is the road model's lead (its range and speed, `road.leads`): it said "opening" in every case. It is not an input
  of the tracker. As existence evidence against the standing object it does not work on the car's cases: the
  standing object stands 3 m BEYOND the road model's lead (the ground point's reach), not inside its range. It
  belongs in as a sensor: a range and a range rate of the vehicle in the car's own lane, like a microwave radar's
  return (`MicrowaveRadar`), which would also give a cut box's object its range.
* **The range of a box cut at the bottom.** The box width is the cue the data has (table above). A row on the
  range from it was built and taken out again: the width the object holds is learnt from these very boxes at the
  range believed (a lead 4.5 m ahead is held 7 m ahead and 3 m wide, or turned by 25 degrees: the pose head's
  heading of a rear view is +-16 degrees), so the row has to bring the kind's width and a heading of its own, and
  that is the lead as a sensor again.
* **A standing object born of a creeping lead** (the car's 516 before it is ended): at walking pace a few boxes with
  3-degree ground points do not tell "creeps with the car" from "stands", and the kind's prior says 0.4 "it stands".
* The other causes of the table (young objects, the car's own turns, narrow streets) are not the tracker's memory
  of things that are gone; they are named there and not touched here.

## 13. The vehicle ahead: the road model's lead as a sensor (2026-11-28)

Section 12 ended at a witness the tracker did not read. The road model's lead (`road.leads[0]`: the vehicle ahead in
the car's own path as an end-to-end network reads it off the same picture, with a distance, a speed, their stated
spreads and a probability) knew at every queue start that the lead was pulling away, and it knows the distance of a
vehicle whose box is cut at the bottom of the view, where the boxes' own range is an extrapolation. It is now a sensor
of the tracker: a range and a range rate of an object the boxes have found.

### What the lead is, measured

On the six driving sessions of 26 / 27 November (1 h 17 min; `/home/piotr/scratch/qstart/tools/`: `leadtab.py`,
`an_stand.py`, `an_appr.py`, `an_dyn.py`). The references: (a) the car's own odometer on approaches to a lead that
stands (the lead's world place is the stop distance + the odometer: 510 frames of 14 approaches); (b) the lead's own
speed against its own distance over 2 s; (c) the detector's ground range of whole boxes beyond 18 m; (d) the two cars
at rest (23 stands of 3-100 s).

| | measured |
|---|---|
| rate, latency | 13-19 messages a second; its speed is of an image pair 200 ms apart |
| it is there | probability >= 0.9 in 0.56 of all frames, 0.1-0.9 in 0.33; of the tracker's settled vehicle on the path it reports (>= 0.9) 0.99 within 8 m, 0.98 to 25 m, 0.61 at 25-40 m, 0.17 at 40-60 m, none beyond; a vehicle that STANDS and is closed on faster than 3 m/s: 0.73 at 8-15 m, 0.1 beyond 25 m (few cases) |
| short losses inside a steady lead within 20 m | one per 170 s, 0.1 s (median) to 0.7 s |
| it names another vehicle (a jump of the distance) | 108 times an hour |
| the distance, against (a) | 0-5 m: +0.01 m (p10 -0.46, p90 +0.50); 5-8 m: +0.20 (-0.65, +0.67); 8-12 m: +0.24 (-0.74, +0.99); 12-18 m: +0.30 (-1.16, +1.16). No scale error (+1 % by day; +8 % on the one night approach) |
| ... its stated sigma (0.09 of the distance; 0.27-0.6 m within 6 m) | the error is within it in 0.70 / 0.56 / 0.59 / 0.43 of the frames of those four bands, within twice it in 0.93 / 0.89 / 0.86 / 0.90: honest times 1.2-1.4 |
| ... where its zero is | not settled: a box's width read as a range (a car 1.8 m wide) is 0.3-1.0 m beyond it at 2.5-9 m and 2 % beyond at 20-30 m; a number plate (0.52 m) puts a lead read at 3.6 m at 4.1 m. A constant of some tenths of a metre between the network's zero and the camera's (`lead_off_m`, 0) |
| ... both cars at rest | sd 0.10-0.29 m about its mean (0.5-0.7 m in two short stands of one session; half the stated sigma on the whole), no drift; it forgets within a second (autocorrelation 0.45 at 0.1 s, 0.26 at 0.5 s, 0.15 at 1 s) |
| ... against (c) beyond 18 m | the detector's ground range is 6-11 % nearer than the lead at 18-32 m, the box width agrees to 0.5-4 %: no finding against the lead |
| the speed (relative: the lead's less the model's own of the same pair), of a lead that stands, both at rest | +0.01 m/s, sd 0.006-0.05 (stated 0.03-0.06) |
| ... of a lead that stands while the car closes | +0.07 m/s (the car at 0.5 m/s) to +0.28 (5.7 m/s): about +0.05 x the car's speed; sd 0.11 |
| ... of a lead that moves, against (b) over 2 s | it under-reads the opening by 0.06 m/s (lead at 0.3-2 m/s), 0.21 (2-5), 0.33 (above 5), 0.30 while the lead accelerates; spread 0.3-0.5 m/s, which is what it states (0.30-0.52): the error is within the stated sigma in 0.58-0.68 of the 2 s windows. Its error lives for seconds, not from frame to frame |
| ... lag | none found (against the distance's own rate the speed is 0.1-0.25 s EARLY, at a correlation of 0.5) |
| its lateral place | on a straight road 0.05 m +- 0.42, whatever the vehicle's place in the lane (correlation with the box's lateral place 0.21; stated sigma 0.19, the difference 0.56 m); on bends it follows the path (slope 1.01 against the path's lateral offset at its distance, correlation 0.89). It says "in the car's lane, here along the road" and no more |
| no lead | probability under 0.3, a distance anywhere (median 27 m), stated sigma 8 m |

So: a range to about +-0.4 m at 4 m and +-1 m at 15 m with no bias the odometer can see, a speed to +-0.1 m/s of a
lead that stands and +-0.4 m/s of one that moves, both with errors that last a second and more; nothing lateral
beyond the lane; nothing of a standing vehicle that the car closes on from far. Not measured: how often it names a
vehicle that is not there with a probability above 0.9; night (two short sessions, one approach).

### What is built (each a knob of `tracker.CFG`, off; with all off the stack's output is the base's, bit for bit)

* **`in_lead`** (`radar/box.py` `_on_lead`, `sensors.RoadLead`, native `SEN_LEAD`, `r_take_lead`; `Tracker.process`).
  `leads[0]` with a probability of at least `lead_prob` (0.9), at most every `lead_every_s` (0.2 s), is a scan of one
  return: z = [the range of the vehicle's NEAR FACE from the reference camera, its direction, the range's rate]. The
  range is the centre's less half the rectangle's depth along the line of sight, as a camera's rows read it (what is
  not known of that depth goes into the row's variance); the rate is the lead's speed less the model's own of the
  same image pair; the direction is its lateral place with a sigma of `lead_y_std` (0.7 m) at least.
  * **Which object it is of** is the scan's own machinery (`Hyp.cycle`): the objects about its place take part
    (`_lead_objs`: generously; the others are not touched, not even stepped), each leaf is gated and weighed with the
    sigmas as stated, "none of them: a vehicle the tracker has no object for" has the density `lead_other`, and the
    pairing stays open over the scans like any other. A return outside the gate of the object as a whole is inside
    where it is inside the gate of one of its hypotheses (a believed-standing object and a return that says "it has
    started").
  * **No existence.** A return adds nothing to the log odds of the leaf that takes it, counts as no sighting, leaves
    the time of the last box as it was; a leaf that takes none loses nothing; a return no object takes starts none.
    The object's life (seen, coast, `seen_max_s`, `swept`, `coast_std_m`, the field's cover, hidden) is its boxes'
    as before: the record of what the last scan could see of it is not overwritten, and `Tracker.manage` reads the
    last sighting of a sensor that looks at objects. What it does give: `sources` and `seen` carry `lead`, the
    distance counts as measured (`range_t`: the consider parameters of section 2 are learnt while it is fresh, the
    threat's `threat_range_s`), and the verdict takes it as a sensor that measures speed.
  * **Errors that last.** A row is taken with the smaller of two variances (`r_take_lead`): R S / (S - R), which
    leaves the hypothesis with the row's own stated variance and no better however often it comes (what the motion
    model has opened since the last return is closed at once: a change is followed without lag); and R x (the
    error's life `lead_tau_x` 1 s, `lead_tau_v` 2 s over the time since the previous return), by which a hypothesis
    that states its range better than the network does is still moved, at the rate at which the network's errors are
    new ones. The stated sigmas are read as `lead_x_k` (1.4) x the range's, no less than 0.25 m, and `lead_v_k`
    (1.25) x the speed's, no less than 0.1 m/s; the speed of a lead that drives is raised by `lead_v_bias` (0.045) x
    that speed, at most 0.35 m/s (the under-reading of the table).
  * **"It stands" expects a sharp speed.** For the standing hypothesis the rate's sigma is `lead_v_stand` (0.12 m/s)
    + `lead_v_stand_rel` (0.06) x the car's speed, whatever the network states of this return (it states 0.3-0.5 as
    soon as it sees motion): that is the measured error of a lead that stands, and it is what lets "it stands" lose
    to the first returns of a start.
  * **A range the object does not know as well as it states** (`lead_open`, 1 sigma). A hypothesis whose near face
    is further from the return's than 1 sigma of the two together is opened along the line of sight until the
    return is that far, and the return then taken (covariance matching on the innovation). The range of a standing
    thing from boxes cut at the bottom is stated to 0.2-0.6 m and wrong by metres (section 12).
* **`sil_near`** (native `r_sil_angle`, `r_extent_update`). A box's width is the angle between the outermost corners
  of the object's rectangle as the camera sees them. It was the rectangle's silhouette over the range of its CENTRE:
  right far away, but the rear of a car 4 m ahead (centre 6.2 m) fills 0.44 rad where that says 0.29, and the
  rectangle explained the rest by its length and heading (the fixture of `tests/test_radar_lead.py`: a car standing
  4 m ahead becomes 6.8 m long and turns 0.27 rad in 6 s; with the knob 4.9 m and 0.00). The tracked lead within 6 m
  on the recordings was 6.3 m long (median; 8.4 m at the ninth decile) and turned 0.24 rad off the lane (0.42): the
  "3 m wide, or turned by 25 degrees" of section 12. Measured, not proposed (below).
* **`pose_gn_cut_deg`** (`radar/box.py`). Degrees added, in quadrature, to the sigma of the pose head's ground point of
  a box cut by the bottom of the VIEW (the extrapolation that reads 2.5-6.3 m far with a stated 1-3 degrees).
  Measured, not proposed (below).

The simulator: the world option `lead_real` (`ScenarioWorld._lead_real`, `LEAD_REAL`; the detector presets take
`+lead`: `detin_pose_scale+lead`) gives the road model's lead the structure of the table: whether it has the vehicle
(by distance; a standing one closed on, far less; held or lost for a second at a time; short losses), an offset per
ride (0.3 m), a slow error of 8.5 % of the distance (4 % while both stand) that lives 1.2 s plus a little white, the
stated 9 %; a speed of the image pair 0.1 s ago with the bias of a standing lead closed on, the under-reading of a
moving one, a slow error of what it states that lives 2 s; a lateral place that carries a fifth of the vehicle's. Off
(the default) the world and `VisionSim`'s stream are what they were (`tests/test_radar_lead.py::Simulator`: the hash of
the first 200 leads of `queue_creep`). `sim/radar_score.py` feeds the lead to the radar where the world has it.

Tests: `tests/test_radar_lead.py` (fixtures at the radar box: off and unread; the measurement and its Jacobian; a
cut-box lead takes the lead's range and states no better than it; a far standing car does not take the return of a
near vehicle; a standing lead that starts is let go within 1 s and followed with its id; a lead that brakes to a stop
stands; a lead creeping with the car is not called standing; no object started, kept or counted by returns; the rear
of a near car is a car along the lane; the simulator's lead). `lead_open` and `pose_gn_cut_deg` have no fixture of
their own: what holds them is the recordings' numbers below.

### Measured

**The recordings** (the six driving sessions re-run through the whole stack with the phone's knobs, which now include
`see_through`, `swept`, `coast_std_m` 3: 41 said `obstacle_ahead` episodes, section 12's count; the base against the
base + `in_lead`; `/home/piotr/scratch/qstart/tools/`: `replay.sh`, `honest.py`, `eps.py`, `epclass.py`, `fate.py`). A
re-run is one sample: knob sets that differ in nothing a case depends on still differ by three to five said episodes
(the lead with eight variations of its own knobs: 38 to 48), so the totals say little and the cases are read one by
one.

The tracked lead (the object in the lane nearest the lead's place) against two references: the stop distance + the
car's odometer on the approaches to a lead that stands (independent of the lead's own numbers but for the stop
distance), and the road model's lead itself, smoothed both ways in time (not independent: it is the input).

| | base | `in_lead` |
|---|---|---|
| the lead (probability >= 0.97, within 20 m) has an object: day, within 6 / 6-10 / 10-20 m | 80 / 80 / 77 % | 85 / 82 / 83 % |
| ... night | 72 / 46 / 92 % | 69 / 94 / 99 % |
| near face minus stop distance + odometer, 0-6 m: median, rms; stated sigma; within 1 / 2 sigma | +1.78 m, 2.88; 0.63; 0.16 / 0.33 | +0.58 m, 0.78; 0.40; 0.30 / 0.76 |
| ... 6-10 m | +1.78, 2.32; 0.80; 0.31 / 0.47 | +0.62, 0.90; 0.61; 0.45 / 0.84 |
| near face minus the lead's, lead within 6 m: median, p10..p90, rms | +2.90 m, +0.93..+6.76, 4.02 | +0.34 m, +0.01..+0.75, 0.64 |
| ... 6-10 m | +1.94, -0.21..+4.61, 2.76 | +0.15, -0.72..+0.81, 0.58 |
| ... 10-20 m | +0.91, -2.04..+4.88, 2.81 | -0.08, -1.01..+0.96, 0.82 |
| ... a lead creeping at 0.3-3 m/s within 10 m | +2.41, +0.42..+6.29, 3.47 | +0.29, -0.40..+0.89, 0.77 |
| ... night, all | +3.04, -1.69..+6.93, 4.00 | +0.41, -0.63..+0.93, 0.93 |
| ... within 1 / 2 stated sigma: day, night | 0.15 / 0.27, 0.29 / 0.47 | 0.42 / 0.83, 0.55 / 0.85 |
| relative speed minus the rate of the lead's distance over 2 s, creeping lead: rms; within 1 / 2 sigma | 1.00 m/s; 0.57 / 0.74 | 0.66 m/s; 0.65 / 0.88 |
| ... on the approaches (the lead stands: against the car's own speed), 0-6 m | 0.53; 0.84 / 0.99 | 0.32; 0.78 / 0.96 |
| the lead's rectangle within 6 m (cars): length median / ninth decile; heading off the lane | 6.3 / 8.4 m; 0.24 / 0.42 rad | 4.7 / 5.8 m; 0.11 / 0.34 rad |

The range bias of section 12 (+2 to +3 m, wandering by 3 m) is +0.3 to +0.6 m with a spread of 0.6-0.9 m, about the
lead's own. The stated sigma covers the approach reference in 0.76-0.84 of the frames at 2 sigma (0.33-0.47 before);
at 1 sigma it is still short (0.30-0.45 where 0.68 would be honest): a standing object within 6 m states 0.2-0.4 m
and sits 0.3 m beyond the lead, where its boxes' rows pull it (`lead_open` 1 is what holds it within 2 sigma: with
2 the coverage was 0.48). The heading that remains is the pose head's (+-16 degrees on a rear view). The share of
the night lead within 6 m with an object (69 %) is two standstills of one session; the variants put it at 69-87 %.

The said episodes by what the alert's object is against the lead at the said time (`epclass.py`; the lead present
with a probability of 0.9; "the lead's object": within 1.5 m or 20 % of its distance, in the lane):

| | base | `in_lead` | `in_lead` + `sil_near` |
|---|---|---|---|
| A: the lead, followed right (speed within 1 m/s), closing with less than 4 s to the bumper: true or arguable | 2 | 6 | 11 |
| B: the lead, followed right, not closing | 0 | 4 | 5 |
| C: the lead's object with its speed off by more than 1 m/s | 3 | 0 | 2 |
| D: a standing object beyond the lead in the lane | 1 | 2 | 2 |
| E: another object | 35 | 26 | 25 |
| said | 41 | 38 | 45 |

* **The queue starts** (section 12's eight; nine by the said times of this re-run). With `in_lead`: `153449` 583,
  1778 and 1871 s are no longer said. `152055` 250 and 567 s and `153449` 1458 s are still said, on a lead that is
  now followed right (B: near face within 0.4-0.6 m, speed within 0.5 m/s of the lead's): the car pulls away
  2.7-4.3 m behind the lead's bumper and the alert is said as the low-speed suppression lifts at 3 m/s. `153449` 582
  and 948 s are still said on a standing object 7-8 m beyond the lead (the next vehicle of the queue, hidden by the
  lead), 1462 s on an object 3.8 m beside the lane. The car's own two of 27 November
  (20:30:53, 20:33:07, both under the suppression): at 20:30:53 the lead is an object now (near face 0.4 m from the
  lead's) and the alert is said at 20:30:54, as the car passes 3 m/s 4.4 m behind the lead's bumper accelerating at
  1.9 m/s^2 (B); 20:33:07 stays under the suppression (in this re-run on a standing object 6 m beyond the lead; in
  another of the variants on the lead's own object, 0.2 m from it).
* **Why B is said.** The tracker's picture is right there. The decision layer's future has the car keep its
  acceleration and the lead its speed; at a queue's start both accelerate. What would settle it is the lead's
  acceleration: the network's own is next to nothing (0.02-0.04 m/s^2 while the lead pulls away at 1 m/s^2), and the
  tracker's follows from a speed it knows to +-0.4 m/s, with a stated spread (0.6-1.2 m/s at such a start) that the
  decision layer reads as risk. Not the tracker's to settle, and not settled: the queue start is no longer a lead
  that is not tracked, it is a lead the decision layer warns of.
* **The true ones.** The base's closings on a slow or stopped lead that the road model confirms (`153449` 389, 403,
  680 s; `114700` 378 s) are all still said. Two more are said that the base did not say: `202650` 202 s (the lead
  10.7 m ahead, closing at 6.5 m/s) and `114700` 46 s (9.2 m, 6.0 m/s): 1.2-1.3 s to its bumper, on a lead the base
  did not have.
* **Others.** Of the base's 35 on other objects 9 are no longer said and 5 new ones are (young objects, turns, narrow
  streets: section 12's classes; the lead does not touch them, the re-run's noise does).

**The simulator** (`tools/alert_truth.py`, the phone's knobs as the base, the road model's lead as measured
(`+lead`), paired over the same runs: six streets and the pedestrian controls x seeds 1-6, the scripted conflicts x
seeds 1-3, day and night; +- the bootstrap's 1 sigma):

| stream | knobs added | false episodes | conflicts warned | lost / gained / later / earlier |
|---|---|---|---|---|
| `queue_creep` + `left_behind`, the reach modelled, seeds 1-6 | `in_lead` | 21 -> 8 (-13 +- 3.0) | 4 -> 4 of 6 | 0 / 0 / 0 / 0 |
| `detin_pose_scale+lead` | `in_lead` | 84 -> 82 (-2 +- 3.4) | 42 -> 42 of 72 | 0 / 0 / 0 / 3 (0.82 -> 0.91, 0.92 -> 1.29, 2.97 -> 3.21 s) |
| `scale+lead` | `in_lead` | 66 -> 71 (+5 +- 2.2) | 37 -> 40 | 0 / 3 (night: `bend_exit_stopped_13` s2, `bend_stopped_13` s1, `half_in_lane_straight_13` s1) / 2 (1.45 -> 1.29, 2.97 -> 2.81 s) / 1 |
| `detin_pose_scale+lead` | `in_lead` + `sil_near` | 84 -> 75 (-9 +- 4.9) | 42 -> 41 | 1 (`half_in_lane_straight_13` s2 night, 0.96 s) / 0 / 0 / 3 |
| `scale+lead` | `in_lead` + `sil_near` | 66 -> 74 (+8 +- 3.3) | 37 -> 38 | 0 / 1 / 4 (by 0.1-0.5 s) / 2 |
| `detin_pose_scale+lead` | `sil_near` alone | 84 -> 79 (-5 +- 3.0) | 42 -> 41 | 1 (the same) / 0 / 0 / 0 |

The +5 of the plain stream is `slow_lead` (1 -> 6: the car follows a lead at 7 m/s, and the lead's speed error of
0.5 m/s that lives for seconds reads as closing). How the lead's speed is weighed decides that scenario against
`queue_creep`, on the stream with the reach modelled (12 runs each): the stated sigma as it is and no bias taken out
8 / 22 false (queue / slow lead; base 21 / 3), the bias taken out 7 / 10, times 1.25 (the default) 8 / 7, times 1.5
13 / 4, times 2 15 / 2. No true warning is lost with `in_lead` on either stream; two of the plain stream's come
0.16 s later, three are gained.

Identities (the radar alone, `sim/radar_score.py`, `detin_pose_scale+lead`, eleven scenarios x seeds 1-3 x day and
night, `in_lead` against the phone's): id switches 138 -> 128 (paired -0.15 +- 0.12 a run), held share 0.373 ->
0.381. `slow_lead` 9 -> 1 (held 0.50 -> 0.62), `queue` 14 -> 10, `crossing_pair` 8 -> 5; `platoon` 3 -> 9 (held
0.505 -> 0.485; not looked into); the other seven within one switch.

Cost (`tools/cost_bench.py run`, the whole path over a session, three runs each, on the P52; `in_lead` + `sil_near`
against the phone's knobs): the radar plug-in 0.205 -> 0.246 ms per road frame on `202650` and 0.489 -> 0.577 ms on
`152055` (the whole path 2.06 -> 2.08 and 3.02 -> 3.15 ms): five scans a second of one return, in which only the
objects about the lead's place are stepped. `in_lead` by itself was not timed apart. Not measured on the
phone. With the knobs off the three captures of `/home/piotr/scratch/cost-rest` (`cost_bench.py
diff`: IDENTICAL) and the six sessions (the dumps' sha256) are the base's.

**`pd_cut`, `gate_hyp`, `pose_gn_cut_deg`, measured again** with the lead in (on `in_lead` + `sil_near`):

| knob added | recordings: said, by class A / B / C / D / E | simulator `detin_pose_scale+lead`: false episodes; conflicts |
|---|---|---|
| (none) | 45: 11 / 5 / 2 / 2 / 25 | - |
| `pd_cut` | 44: 9 / 5 / 1 / 0 / 29 | 75 -> 75 (+0 +- 2.5); 41 -> 41 warned, one earlier |
| `gate_hyp` 6 | 47: 10 / 5 / 0 / 4 / 28 | 75 -> 76 (+1 +- 1.7); 41 -> 41 |
| `pose_gn_cut_deg` 6 | 44: 8 / 5 / 1 / 2 / 28 | 75 -> 78 (+3 +- 2.9); 41 -> 41, one later (0.75 -> 0.51 s), two earlier |

### Proposed for the car, and what is not solved

In the phone's `radar.json`: **`"in_lead": true`**.

Not `sil_near`: one night warning of the simulator is lost with it (`half_in_lane_straight_13`, 0.96 s), four of the
plain stream's come later, and on the recordings the lead's rectangle then comes out short (3.7 m, 1.7 m wide:
the lead's zero is some tenths of a metre nearer than the box's width puts the vehicle) with the heading no better
(0.17 rad). The model is the right one; it wants the lead's zero settled (`lead_off_m`) and the pose head's heading
of a rear view weighed as what it is.

Not `pd_cut`, not `gate_hyp`, not `pose_gn_cut_deg`: none shows a gain with the lead in.
`pd_cut`: section 12's objection (a lead that keeps its boxes is said with its range and speed errors) has fallen for
the lead itself, whose range is now the road model's; what is left is no gain in the simulator (0 +- 2.5) and four
more said episodes on other objects on the recordings (25 -> 29: inside a re-run's noise, and not a gain).
`gate_hyp`: +1 +- 1.7, two more said. `pose_gn_cut_deg` 6: the lead's range bias within 6 m goes from +0.4 to +0.2 m
(rms 0.6 -> 0.4 m against the lead), but +3 +- 2.9 false episodes and one night warning 0.24 s later in the
simulator.

Not solved, and why:

* **The queue's start as an alert** (B above): the lead is followed, and the decision layer says it. Its future needs
  the lead's acceleration, or to know that a queue starts.
* **The next vehicle of the queue, hidden by the lead**, kept as a standing object 7-8 m beyond it while the queue
  drives off (`153449` 582, 948 s): the lead names one vehicle and says nothing of the next; `coast_std_m` does not
  touch a standing object.
* **The stated sigma of a standing lead's range** (0.2-0.4 m against an error of 0.6-0.8 m at 1 sigma): the boxes'
  rows of a cut box are stated too well; the lead holds the object within 2 sigma, not within 1.
* **The lead's zero** (some tenths of a metre between the network's distance and the camera's), **its false
  positives** (a lead named with a probability above 0.9 where no vehicle is: not measured; the tracker then places
  whatever object is there, and adds no existence), **night** (two short sessions).
* **Following a lead at speed**: where the boxes are whole the lead's speed is the worse witness, and the
  simulator's `slow_lead` on the plain stream says 5 false episodes more for it in 12 runs. The recordings show no
  such episode in 43 minutes of following (the four "B" are queue starts).
