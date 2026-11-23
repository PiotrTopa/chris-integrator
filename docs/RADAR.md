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
* **A scan** (`Hyp.scan`): every leaf is stepped to the scan's time; each leaf is compared with each detection
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
not the centre: `on_path`, `in_path`, `ttc_s` use it); the centre goes out as `ctr`.

**A floored hypothesis is re-seeded.** The motion hypotheses are weighed by how well each predicted a detection
and none is ever ruled out (`mu_floor`). A hypothesis held at the floor used to keep its own old story: "it
stands" of a car that has driven on for four seconds stood fifty metres back, and its share of the combination
pulled the object's place and blew up its covariance (found on the hidden-car test). What the floor keeps open is
that the object does so *from where it is now*: the share the floor adds is given the other hypotheses' state (for
"it stands": their place, no speed) (`r_update_full`).

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
* **Heading and motion** (`r_extent_step`): a leaf that moves (speed over 2.5 m/s, "it stands" under one half)
  takes its direction of travel as a measurement of its heading, as good as the velocity's direction is known;
  the first time, that also says which end is the front. A leaf that stands gets no such measurement and next to
  no process noise: its heading is held.
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
