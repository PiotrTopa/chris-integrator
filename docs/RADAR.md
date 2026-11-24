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
