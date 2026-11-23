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
* `...::test_two_cars_swapping_sides_keep_their_ids_more_often_with_hypotheses_over_scans`: two cars swapping
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
