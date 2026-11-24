"""The radar's core: the list of objects, kept over time by model-based filtering.

One cycle per scan (a sensor's message, `process`):

1. every object is brought to the scan's time: the hypotheses interact (models.mix) and each predicts forward
   (models.predict); what the car's own movement in that interval is not sure about is added to where the
   object is believed to be;
2. the scan's detections are compared with what each object predicts for this sensor at the car's pose of that
   moment (sensors.py): a gate, then one global assignment of detections to objects;
3. a detection updates each hypothesis of its object, as far as the pairing is believed when all objects and
   detections of the scan are weighed together (`marginals`); how well each hypothesis had predicted it sets
   how much that hypothesis is believed from now on (the IMM's model probabilities: "it stands" against "it
   drives"), and how well the object as a whole had predicted it feeds the object's existence and its record;
4. an object the scan should have seen and did not loses existence; a detection nobody claims may start an
   object, tentative until later scans agree with what it predicts.

Checked against later frames, explicitly: besides the innovation of every update, each object writes down
where it expects to be 0.5 s and 1 s ahead; when that time has passed, the expectation is compared with where
the object turned out to be (`consistency`, about 1 when the predictions hold within their own stated
uncertainty; `pred_err_m`). An object that keeps being mispredicted has its uncertainty opened.

What a new object may be doing is three stories, not one blur: it stands; it drives with the traffic (about
the car's own velocity); it comes the other way; a vehicle whose box shows it side-on crosses (`process`).
Beside the hypotheses every object keeps a free estimate (`Obj.xf`): a steady-velocity filter fed the same
detections and never told that things stand or drive with the traffic. It is what the sightings alone have
shown. "It stands" hands it to the moving hypotheses when they take over, and the verdict "moving" waits
until it shows a speed.

Objects have a memory: a trajectory on the ground (`history`), the class vote, hits and age, the sensors that
have seen them and when. Their life: tentative -> confirmed -> coast (inside a sensor's field, not reported for
the moment) -> gone when the sensors that should see it keep not seeing it. An object that has LEFT every
sensor's field is not gone: nothing says it stopped existing. It is kept as `memory`, carried on by its own
motion hypotheses and the car's movement, for as long as where it is remains worth knowing (`memory_*`: a
standing car stays put for a long time, a moving one's place blurs within seconds), and it is the same object,
with the same id, when a sensor sees it again (another camera, a radar looking sideways or back).
"""
import math
from collections import deque

import numpy as np

from ..core import classvote
from . import models as M
from .models import CA, CV, N, PX, PY, STAND, VX, VY
from .sensors import kind_code

_NR = M.NR                       # the native core: the tracker's arithmetic (chris_native.c, the radar part)

CFG = dict(
    gate={1: 9.0, 2: 13.8, 3: 16.3, 4: 18.5},          # chi-square gates (99.7 %) by measurement dimension: the first sieve
    new_rate_gain=0.1,           # how fast the count of unexplained detections per scan is followed
    # where detections that no object explained have been turning up lately (a map of them, in the sensor's
    # measurement space): there a detection is more probably one more of those than a known object's
    map_s=5.0, map_bw=(math.radians(0.4), math.radians(1.5), 0.4),     # memory; kernel widths: angle, bearing, log height
    # the box as a feature: an object's box does not jump from where the object predicts it by more than a
    # part of its own width; another object's box is anywhere near
    feat_sigma=0.25, feat_same=0.9, feat_span=3.0, feat_max_s=1.0,
    with_rel=0.3, with_std=3.0,  # "drives with the traffic": the car's own speed within this share of it + this (m/s)
    new_exist=0.2,               # existence of an object started by one detection
    size_from_class=True,        # the height prior follows the class vote (`_size`); False: the first box's label's
    # the detector's own confidence in a box: how much likelier a box of that score is a real thing than a
    # false one, learnt from the boxes themselves (the scores of boxes that went into confirmed objects against
    # those of objects that never were confirmed); bins over the score, what is assumed before anything is
    # learnt (mean, spread of a true and of a false box's score, weight in boxes), and how much one box says
    score_bins=10, score_prior=((0.62, 0.14), (0.50, 0.14), 30.0), score_gain=0.6, score_max=1.2, score_keep=3000.0,
    confirm_exist=0.95, confirm_hits=3, delete_exist=0.04,
    max_objects=64,
    split_rel=0.12,              # a hypothesis whose bearing is unsure by this much (rad) is not linearised in one
                                 # piece (docs/RADAR.md, section 5); 0: always in one piece
    # the hypotheses (chris_native.c `Hyp`; docs/RADAR.md). Each is a knob of cost against quality, the default
    # the full setting: how many scans back a doubtful pairing stays open (0: decided at once, the best
    # assignment of each scan), how many global hypotheses a cluster of competing objects keeps, how many leaves
    # an object, how much less probable than the best a hypothesis may be (log), the largest cluster whose
    # assignments are enumerated (a larger one gets the best assignment per hypothesis only)
    hyp_n_scan=5, hyp_k_best=32, hyp_max_leaves=12, hyp_prune_ln=math.log(1e4), hyp_max_cluster=16,
    hyp_keep_steps=40, hyp_keep_s=2.0,   # each story's history kept this far back (the smoother's reach)
    rows=1024,                   # of the state array at the start (it grows as needed)
    # the past, smoothed (docs/RADAR.md, section 6): a backward pass over each story after every sighting, back to
    # the lag: the velocity that is put out as `vs` and judged is the one smooth_lag_s ago (the first node that far
    # back within 1.5 lags; once per smooth_hist_s the pass looks as far as smooth_s for it). An object's trajectory
    # (`Obj.history`) is smoothed as far back as smooth_s when it is asked for
    smooth=True, smooth_s=2.0, smooth_lag_s=0.5, smooth_hist_s=0.5,
    # smooth_every_s: an object's past is smoothed at most this often (0: after every sighting). A sighting that
    # comes sooner keeps the smoothed velocity the object has (`vs`, and what the verdict "it moves" reads), its age
    # (`vs_age_s`) grown by the time since; the backward pass is a third of the tracker's arithmetic in a dense scene
    # at a detector's frame rate. What it changes: docs/RADAR.md, section 6
    smooth_every_s=0.0,
    # smooth_for: whose past is smoothed. "all": every object's. "read": only where the tracker itself reads the
    # smoothed estimate: an object whose verdict is not "moving" (the verdict "it moves" and the groups read what its
    # sightings by themselves show of its speed). A moving object then has no `vs` (null, as before a second
    # sighting); should its verdict be released, the filtered estimate stands in until its next sighting
    smooth_for="all",
    # structure (docs/RADAR.md, section 7): rows of standing things, queues, platoons. The free room between two
    # members (standing / driving), how alike two drivers' velocities are, how straight the line is (m + share
    # of its length), how near the car's own line a queue's passes, members needed (a queue: two)
    groups=True, group_gap_stand_m=7.5, group_gap_move_m=30.0, group_dv=2.5, group_line_m=1.2, group_line_rel=0.03,
    group_lane_m=1.6, group_min=3, group_move_mps=3.0, group_still_m2=4.0,
    # what a row or a queue says about a member, once per group_s: it stands on the others' line to group_sigma_m,
    # and "it stands" is group_gain times likelier
    group_support=True, group_s=0.5, group_sigma_m=0.8, group_gain=2.0,
    # the road's axis as a standing vehicle's heading prior (`road_prior`, docs/RADAR.md, section 4): once per
    # road_prior_s, for a vehicle within road_prior_off_m of the road model's path (and no further than
    # road_prior_beyond_m past its end), to road_prior_std, where its heading is within road_prior_snap of the axis
    # (road_prior_mirror: the heading's mirror about the line of sight is taken when it lies this near the axis)
    road_prior=True, road_prior_s=1.0, road_prior_std=0.1, road_prior_snap=0.7, road_prior_mirror=0.25,
    road_prior_off_m=12.0, road_prior_beyond_m=10.0,
    # what a sensor can see: an object behind another one, or cut by the edge of the field, is not expected to be
    # reported (docs/RADAR.md, section 3). occlusion False: every object in the field counts as in the open. The
    # chance of a report falls from `occl_from` of the silhouette hidden to its lowest at `occl_full`
    occlusion=True, occl_from=0.25, occl_full=0.65, occl_floor=0.03, occl_floor_radar=0.3,
    hidden_s=12.0, hidden_std_m=30.0,   # hidden behind another object: kept this long, while its place is known this well
    oosm_max_s=0.35,             # a scan older than an object's time by more than this does not touch it
    history_s=6.0, history_n=120,
    check_horizons=(0.5, 1.0),   # the object writes down where it will be this far ahead
    check_window_s=0.45,         # ... and is checked by the first update within this time after it
    inconsistent=3.5, inflate=1.4,
    # verdict "stationary" / "moving"
    stand_p=0.85, move_p=0.10, release_p=0.5, verdict_hits=3, verdict_s=0.0,
    move_m2=9.0,                 # "moving" is said once the sightings by themselves show a speed (shown_moving)
    side_m=2.5, p_any_side=0.5,  # a new object this far beside the car's own line: as likely doing "anything
                                 # else" (oncoming, crossing) as driving with the traffic
    verdict_hits_velocity=2, verdict_s_velocity=0.2,    # with a sensor that measures speed
    merge_m=2.5, merge_rel=0.12, merge_s=0.6, merge_dv=2.0,
    merge_d2=9.2,                # ... and only within what both know of their places (squared, 2 dof: 99 %)
    # life
    view_gap_s=0.7,              # no scan has covered the object's place for this long: it is out of view
    seen_max_s=5.0,              # inside a field: dropped after this long unseen whatever its existence says
    memory_s=45.0,               # out of view: kept at most this long
    memory_std_m=10.0,           # ... and only while its place is known this well (1 sigma, larger axis)
    memory_range_m=150.0,        # ... and it is this near
    memory_tentative_s=1.0,      # an object never confirmed is not worth remembering
    mu_hold_hits=2,              # the hypotheses are weighed from this many sightings on (velocity initialised)
    mu_floor=0.01,               # no hypothesis is ever ruled out for good
    # a moving hypothesis does not learn the object's height (lh) and angle offset (dp) from a camera's boxes: they
    # are consider parameters there (their uncertainty in the covariance, their values held; docs/RADAR.md, section
    # 2), from the sighting on at which the hypotheses are weighed (mu_hold_hits), unless a sensor that measures range
    # has seen the object within consider_range_s. consider_stand: the standing hypothesis too (measured, not taken:
    # docs/RADAR.md, section 2)
    consider_moving=True, consider_range_s=2.0, consider_stand=False,
    common_min=3, common_pair=0.8,   # the scan's common pitch: from at least this many settled objects, each paired this surely
    bias_ripe=0.5,               # an object speaks about the sensor's bias once its own angle offset is known
                                 # this much better (share of the variance it was born with)
    bias_seen_s=1.0,
    # ---- the detector's further inputs (docs/RADAR.md, section 8; roadsense docs/ARCHITECTURE.md, "Detection inputs
    # for a tracker"). Each is a knob, off until its evidence says otherwise; with all of them off a scan is what it
    # was, bit for bit. They act only on a camera's boxes, and only where the detector sends the fields.
    # flags: the hidden edges of a box (`clip` + `trunc` + `occ_edges`) decide which rows it has; the centre of a
    # box with a hidden side is moved to the object's by what the object's silhouette has beyond the box; a covered
    # box teaches no width
    in_flags=False,
    # flags_far_bearing: a box whose bottom is under a nearer box (a direction only) may be taken by an object beyond
    # the view's first ground point. False: such a box is left to the near objects (measured: taken, it costs a
    # fifth more id switches in rows for 3 % more tracked time; docs/RADAR.md, section 8)
    flags_far_bearing=False,
    # flags_side: the centre of a box with one side hidden is moved to the object's (by half of what the object's
    # silhouette has beyond the box). False: such a box keeps its centre and its wider bearing spread
    flags_side=True,
    # the scan's own boxes hide what stands behind them (the miss probability): a box counts as nearer where its
    # bottom lies lower than the object's by more than box_occl_margin of what is not known of the difference
    in_box_occl=False, box_occl_margin=1.5,
    # the low-score list (`objects_low`): a second kind of detection. Of what the first list leaves of a visible
    # object the low list holds low_p (0.41 measured on BDD100K at a floor of 0.10), low_p_run inside a run of
    # misses of an object the first list had within low_run_s (0.68-1.0 measured on rides); low_false_per_scan low
    # boxes a scan on no known object at least (followed upwards from the scans: 0.35-1.1 measured); its edges:
    # low_px times the px term and low_rel relative (measured 10-13 % of the box height against 4 %); its class
    # vote weighs low_weight; an object whose last low_coast sightings are all low is put out as `coast`
    in_low=False, low_p=0.4, low_p_run=0.7, low_run_s=1.0, low_false_per_scan=0.6, low_px=1.5, low_rel=0.10,
    low_weight=0.3, low_coast=4,
    # the appearance vector (`emb`): a likelihood ratio in the pairing, bounded to app_bound (day, night), from the
    # measured distance fits (app_tables: per kind, day and night, the same object 1 / 2 / 4 s apart and another
    # object: mu, sigma of the logarithm; roadsense docs/lanes/detin-emb-distances.json); the memory follows the
    # object's whole first-list boxes with app_gain. How dark the scene is: read off the vectors' luma cells
    # (app_y_day and above: day, app_y_night and below: night), or told (app_dark: 0..1; None: read).
    # reid: an object that was dropped and is born again within reid_s where it could be by then takes its id back
    # when its vector says so reid_lr times likelier than not (and twice likelier than the next candidate)
    in_app=False, app_bound=(8.0, 3.0), app_gain=0.3, app_dark=None, app_y_day=-0.5, app_y_night=-1.2,
    app_tables={1: ((-0.729, 0.903, -0.552, 0.884, -0.408, 0.914, 0.901, 0.462),
                    (-0.087, 0.603, 0.067, 0.604, 0.225, 0.539, 0.686, 0.616)),
                2: ((-0.925, 0.837, -0.771, 0.835, -0.641, 0.871, 0.558, 0.344),
                    (-0.409, 0.497, -0.285, 0.519, -0.166, 0.473, 0.266, 0.491))},
    in_reid=False, reid_s=6.0, reid_lr=4.0, reid_gate=16.0,
    # the lamp pair (`lamps`). Its colour: which end faces the camera (the heading, to lamp_colour_deg and no
    # better; a new object with white lamps comes the other way if it drives). Its separation: a ruler on the object
    # (lamp centres lamp_sep_ratio of the vehicle's width apart, unknown to lamp_sep_rel in the logarithm), read
    # against the box height as a measurement of the object's height, and its change from pair to pair as a scale
    # rate; a pair under lamp_min_px apart is not used (the detector picks other blobs there)
    in_lamp_colour=False, lamp_colour_deg=40.0, in_lamp_size=False, lamp_sep_ratio=0.67, lamp_sep_rel=0.2, lamp_min_px=10.0,
    # the box width of a vehicle seen along its axis as a ruler: read against the box height it measures the
    # object's height (a vehicle's width follows from its height to width_rel)
    in_width=False, width_rel=0.10,
    # the rulers: with ruler_eps their number is not the object's; "anywhere" spans ruler_span (logarithm); each
    # ruler is used ruler_uses times, each use with a share 1 / ruler_uses of what its prior knows
    ruler_eps=0.1, ruler_span=3.0, ruler_uses=8,
    # the road's vertical curvature, 1 sigma (1/m; crest and sag radii of built roads are 2-10 km: 1 / 3000), in each
    # object's own ground-angle offset. 0: the road ahead is the plane under the car
    grade_k=0.0,
    # the scale rate: successive values of one object share their baseline (the variance times the baseline over the
    # time since the object's previous value); a value from the lamp pair alone is taken with Huber's weight at
    # scale_lamps_k sigma
    in_scale=False, scale_lamps_k=2.0,
    # ---- parked cars (docs/RADAR.md, section 9) ----
    # sil_bearing: a camera box's middle is the middle of the object's SILHOUETTE (its rectangle's picture), not the
    # bearing of its centre: the difference (a tenth of the box width for a car passed at 10 m) is predicted, what
    # is not known of the rectangle goes into the row's variance, and the box's middle is then as good as its edges
    # (sil_bearing_rel of the box width, where the camera's bearing_rel 0.11 had to cover the difference). It is what
    # lets the car's own travel range a standing thing (parallax) and tell the cars of a row apart
    sil_bearing=False, sil_bearing_rel=0.05,
    # ctx_prior: where a new vehicle is first seen says whether it stands before its own boxes can: the vehicles
    # already judged, at its distance beside the road's path (kernel ctx_band_m), by their verdicts ("the street's
    # own count"): standing n_s, driving n_m -> (n_s + ctx_n p0) / (n_s + n_m + ctx_n), p0 the kind's own prior. It
    # only ever raises "it stands" (never above ctx_max): a stopped car among driving ones keeps the kind's prior
    ctx_prior=False, ctx_band_m=0.8, ctx_n=1.0, ctx_max=0.9,
    # size_learn: how tall the things of a kind are is learnt from the tracker's own objects (`_size_learn`): a
    # standing vehicle that the car has passed is ranged by the car's own travel, and its box height at that range
    # is its size, whatever the kind's prior said. What such objects agree on is the kind's (this camera's cars, its
    # detector's boxes, its focal length: every scale error of the range from a box height), and becomes the prior
    # of every object of the kind: the mean, and the spread between objects (no smaller than size_learn_floor).
    # From objects seen size_learn_hits times, held to the built-in prior with the weight of size_learn_n0 objects,
    # the mean moved by size_learn_shift (log) at most; the last size_learn_keep objects count
    # and only objects whose line of sight has swung by size_learn_swing_deg over their sightings (the car passed
    # them: without that the "range" is the ground angle's, and the camera's pitch error would be learnt as a size)
    size_learn_swing_deg=5.0,
    size_learn=False, size_learn_hits=6, size_learn_n0=3.0, size_learn_floor=0.10, size_learn_shift=0.25, size_learn_keep=200,
    # ---- one car, one object in a row; the detector's miss runs; the pose head (docs/RADAR.md, section 10) ----
    # row_ctx: every box of a scan is given "how much the place it was seen at is a row of standing vehicles" before
    # the scan pairs anything: the vehicles already judged at its distance beside the road's path (the count of
    # `ctx_prior`, of the box's own ground point), q = n_s / (n_s + n_m + ctx_n). What follows from q:
    #   row_birth   "something else" is exp(row_birth q) times denser at such a box: beside a row the next box along
    #               IS another car (the density of new objects was the same over the whole field)
    #   ctx_exist   an object born there exists with new_exist + (ctx_exist - new_exist) q: a box in a row of parked
    #               cars is a car (with new_exist 0.2 a car's own second box went to the settled neighbour's object)
    #   ctx_vel     ... and if it moves at all it starts from rest: its moving stories start from (1 - q) x the
    #               traffic's speed, as narrow as ctx_vel (m/s) where q is 1 ("the same car, driving with us" was what
    #               explained the next car of the row); 0: as anywhere
    row_ctx=False, row_birth=2.5, ctx_exist=0.6, ctx_vel=2.0,
    # det_chain: a detector misses in runs. Per object and view of the camera a two-state chain: the detector has the
    # object (a box with probability det_chain_d) or has lost it; its long-run share is the sensor's own model (range,
    # box size at the view's limit), it forgets within det_chain_tau_s. The chance of a box, the price of a miss and
    # the pairing all read it: an object boxed a moment ago is expected again (and holds its next box against a new
    # object), a run of misses costs its first miss and little more
    det_chain=False, det_chain_tau_s=0.8, det_chain_d=0.9,
    # pd_learn: the chance of a box is learnt per view of the camera and box height from the settled objects' own
    # boxes and misses there (held to the sensor's model with the weight of pd_learn_n0 scans; a bin remembers
    # pd_learn_keep scans). The model's ramp from "the smallest box this view has shown" to twice that is a guess:
    # where most of a view's boxes are near its smallest it said a tenth, and a young object's next box went to a new
    # object. exist_shared: an object's existence counts the stories in which ANOTHER object took its last box (the
    # stories of a young object disagree on whose its boxes were; every one of them has a thing there)
    pd_learn=False, pd_learn_n0=6.0, pd_learn_keep=400.0, exist_shared=False,
    # the low-score list inside runs of misses only (low_runs_only: an object the first list had within low_run_s;
    # nowhere else), and as existence only (low_exist_only: a low box takes the miss back and leaves the object's
    # numbers to its motion: its edges made standing things look moving)
    low_runs_only=False, low_exist_only=False,
    # the pose head (`pose`, contract pose1; roadsense docs/lanes/detin-consumption-design.md section 9):
    #   in_pose_gn   the near-face ground point as the ground-angle row: on a whole box whichever of it and the box's
    #                bottom has the smaller sigma (one row: the same pixels); on a box whose bottom is hidden the row
    #                that box lacked, cut at the visible bottom (the contact lies no higher: a one-sided bound), its
    #                sigma times pose_gn_occ under a nearer box; successive rows of one object share their error for
    #                pose_corr_s (the variance times pose_corr_s over the time since the object's previous one)
    #   in_pose_gc   the footprint centre's column as the bearing row (the centre's own bearing, also where a side of
    #                the box is hidden)
    #   in_pose_yaw  the heading: votes on which end is the front (ln(p / flip), at most pose_vote_max; the front is
    #                turned at pose_vote_turn against it; votes nearer than pose_vote_half_s count half) and the axis
    #                as a row on the extent's heading (sigma max(yaw_std, pose_yaw_floor_deg), Huber at
    #                pose_yaw_huber sigma, the same memory of errors); a new object's heading; in the dark a lamp
    #                pair's colour votes with the full weight, the head with pose_vote_dark, and its axis is taken
    #                only with p >= pose_axis_dark_p
    # a pose is read only where its `kind` says vehicle with pose_kind_min and the box is of a vehicle class; a point
    # is taken where its `seen` is at least pose_min_seen or its box edge is hidden
    in_pose_gn=False, in_pose_gc=False, in_pose_yaw=False, pose_corr_s=1.0, pose_gn_occ=1.5, pose_yaw_floor_deg=3.0,
    pose_yaw_huber=2.0, pose_vote_max=math.log(8.0), pose_vote_turn=math.log(4.0), pose_vote_dark=math.log(2.0),
    pose_axis_dark_p=0.9, pose_vote_half_s=0.5, pose_kind_min=0.5, pose_min_seen=0.5,
    # ---- the range of a standing thing (docs/RADAR.md, section 11) ----
    # scale_iter: the standing hypothesis takes a box with a scale rate by iteration (at most scale_iter_n steps).
    # The image of a thing that stands grows at (the car's speed) / (its depth): a hyperbola in the range, of which
    # one linear step from 0.6 of the range reaches 0.84 of it and states a spread as if it had arrived
    # scale_seed: "it stands", brought back from the floor, is the standing thing THIS box's scale rate describes (at
    # the range the rate says, as tall as its box is from there): the hypothesis is opened along the common factor of
    # range and size (scale_seed_sigma, of its logarithm) and takes the rate from there, where the rate says the range
    # scale_seed_k times better than the others' place did. Before, it was given the place of the hypotheses that
    # DRIVE, where no standing thing gives that rate, and was refuted again at the next box. Measured (docs/RADAR.md,
    # section 11): with the update iterated the hypothesis no longer falls to the floor by its own error, and the
    # seed then lets an object that took a neighbour's box stand again at the neighbour: not recommended
    scale_iter=False, scale_iter_n=8, scale_seed=False, scale_seed_sigma=0.7, scale_seed_k=3.0,
    # scale_exact: a scale rate is read as what it is, the object's depth at the far end of its baseline (`scale_dt_s`
    # ago, along the camera's axis as it pointed then) over its depth now: with the car's own pose then, from its
    # movement's record. It was read as the depth's rate of this moment: right while the car drives straight at a
    # steady speed, 1.3 % far 40 m ahead on a bend of 110 m radius, 9 % near while braking at 3 m/s^2
    scale_exact=False,
    # near_axis: a box's height and its scale rate go with the depth of the object's near face along the camera's
    # axis (the picture's size is the size over that depth), not with its range along the line of sight: a car parked
    # 3 m beside, 15 m ahead, read 0.1-0.2 m far
    near_axis=False,
    # ext_size: a vehicle's length and width priors follow its height (`_ext_size`, models.extent_of_height): the
    # class vote's until its numbers know its own ext_size_learnt times better (in variance), then its own; exchanged
    # in every story of it when the height says another size by ext_size_step (share of the size, or four times that
    # of the prior's variance in the logarithm). An object keeps the extent of its first box's label otherwise
    # (a car born from a box called `truck`: 10 x 2.5 m; a van called `truck`: 10 m for its 6): its centre then lies
    # half that length behind a near face known to a few tenths of a metre
    ext_size=False, ext_size_step=0.08, ext_size_learnt=0.25,
)
_IN_KNOBS = ("in_flags", "in_box_occl", "in_low", "in_app", "in_reid", "in_lamp_colour", "in_lamp_size", "in_width", "in_scale",
             "sil_bearing", "row_ctx", "det_chain", "in_pose_gn", "in_pose_gc", "in_pose_yaw", "pd_learn", "exist_shared",
             "scale_iter", "scale_seed", "scale_exact", "near_axis")
_RANGE_KNOBS = ("scale_iter", "scale_seed", "scale_exact", "near_axis")
_SPREV = 8                       # scale-rate baselines of one scan whose far end's pose is given (chris_native.c RSPREV)
_POSE_KNOBS = ("in_pose_gn", "in_pose_gc", "in_pose_yaw")


def _npsum(a):
    """sum(a) added up as numpy adds a contiguous array (pairwise, eight at a time): the same float."""
    n = len(a)
    if n < 8:
        r = 0.0
        for v in a:
            r += v
        return r
    if n <= 128:
        r = list(a[:8])
        i = 8
        while i < n - n % 8:
            for k in range(8):
                r[k] += a[i + k]
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for v in a[i:]:
            res += v
        return 0.0 + res
    n2 = n // 2
    n2 -= n2 % 8
    return _npsum(a[:n2]) + _npsum(a[n2:])


def marginals(ratio):
    """Probability of each pairing (object, detection) when all objects and detections of the scan are
    considered together: every detection is at most one object's, every object has at most one detection.
    ratio[i, j]: odds of "detection j is object i's" against "i was missed and j is something else", for the
    pair alone. Two objects that both fit one detection cannot both have it; the pairing's probability says
    so, where the pair's own odds do not. (Loopy belief propagation on the assignment graph: Williams & Lau
    2014; converges for this problem, a few sweeps; 30 sweeps at most, until the messages move less than 1e-4.)
    The tracker's scans do this inside the native association (radar_scan)."""
    L = np.ascontiguousarray(ratio, float)
    out = np.empty(L.shape)
    _NR.radar_marginals(L, out)
    return out


# An object's numbers live in rows of the tracker's state array (`State`): one row per node of its tree of
# hypotheses (chris_native.c, "the radar's hypotheses"). A row holds the motion hypotheses (xs, Ps), their
# probabilities (mu), the free estimate (xf, Pf), the combination (x, P), the extent (e: length, width, heading;
# Pe), the step that led into the node (st) and the node's own record (m: `M_*`). The object's attributes are
# views of its PRIMARY row: its leaf in the best hypothesis of its cluster, which the native scan names.
_LAY = (("xs", (3, N)), ("Ps", (3, N, N)), ("mu", (3,)), ("xf", (N,)), ("Pf", (N, N)), ("x", (N,)), ("P", (N, N)),
        ("e", (3,)), ("Pe", (3, 3)), ("st", (13,)), ("m", (20,)))
_OFF = []
_o = 0
for _k, _sh in _LAY:
    _OFF.append(("_" + _k, _o, _o + int(np.prod(_sh)), _sh))
    _o += int(np.prod(_sh))
ROW = _o                         # numbers per row
OFF = {k[1:]: (a, b) for k, a, b, _ in _OFF}
# a node's record (chris_native.c `M_*`): its time, the log odds that the object exists in this story, sightings,
# the last sighting's time, the last time a detection said how far it is, its last box (angular width, time,
# sensor number), the last update's normalised innovation and existence gain, the detection it took at its
# scan (-1: none), the scan's number, how much of it the sensor could see, its probability of being reported,
# the scan's sensor, what hides it (0: nothing, 1: another object, 2: the edge of the field, 3: outside the
# field), the sensor's coverage at its place, the id of what hides it, 1 once its motion has shown which end is
# the front
(M_T, M_L, M_HITS, M_THIT, M_RANGE_T, M_BOXW, M_BOXT, M_BOXS, M_NIS, M_DL, M_DET, M_LEVEL, M_VIS, M_PD, M_SRC, M_OCC,
 M_COV, M_OCCBY, M_SENSE, M_HOLD) = range(20)


# Beside the rows the tracker keeps a table of its objects, one row per handle (chris_native.c `ROT_*`): what a scan
# needs to know of an object and what it says of it, so that a scan is one native call over all objects
# (`Hyp.cycle`) and what follows its sightings another (`Hyp.after`). The primary row; the probabilities that it
# exists and that the primary's story is the true one; its leaves; 1 if it is in the best story of its cluster; when
# a scan's field last covered its place; when it was last hidden; how unknown its angle offset was at birth; its
# id; its state (`_STATES`); what its kind can do (4); vehicle 1 / person -1 / not known 0; the last time a sensor
# that measures range saw it; how its predictions held (consistency, error, the number of checks); 1 if it keeps
# being elsewhere than predicted; when its trajectory was last smoothed as a whole; 1 once it has a smoothed
# velocity, that velocity (vx, vy, 1 sigma x, y, how long ago), the free estimate's (vx, vy, 00, 01, 11) and the
# smoothed probability of "it stands" at the lag; the predictions it has written down: their number and
# (time, x, y, P00, P01, P10, P11) each.
OT_MAXCHK = 12
(OT_SLOT, OT_PE, OT_PP, OT_NLEAF, OT_INMAP, OT_COVER, OT_HIDDEN, OT_DP0, OT_ID, OT_STATE, OT_KIN) = range(11)
(OT_LABEL, OT_RANGE, OT_CONS, OT_PERR, OT_NCHECKS, OT_UNPRED, OT_HIST_T, OT_HASVS, OT_VS) = range(OT_KIN + 4, OT_KIN + 13)
OT_FS = OT_VS + 5
OT_PSLAG = OT_FS + 5
OT_NCHK = OT_PSLAG + 1
OT_CHK = OT_NCHK + 1
OTW = OT_CHK + 7 * OT_MAXCHK     # numbers per object
_STATES = {"tentative": 0.0, "confirmed": 1.0, "coast": 2.0, "memory": 3.0}
_MO = OFF["m"][0]
_XLH, _PLH = OFF["x"][0] + M.LH, OFF["P"][0] + M.LH * N + M.LH      # of a row: the height (lh) and its variance


# of a primary row: the free estimate's velocity and its covariance, "it stands", the velocity (Tracker._groups)
_GCOLS = [OFF["xf"][0] + VX, OFF["xf"][0] + VY, OFF["Pf"][0] + VX * N + VX, OFF["Pf"][0] + VX * N + VY,
          OFF["Pf"][0] + VY * N + VY, OFF["mu"][0] + STAND, OFF["x"][0] + VX, OFF["x"][0] + VY]


class State:
    """The tracker's numbers: the rows (capacity, ROW), the objects' table (handles, OTW) and their trajectories
    (handles, 1 + 5 x history_n: the count, then (t, x, y, vx, vy) per point, the oldest first). The native
    hypotheses (`Hyp`) say which rows are in use; a State by itself (tests, tools) hands out rows in order."""

    def __init__(self, cap=96, objects=64):
        self.a = np.zeros((cap, ROW))
        self.free = list(range(cap - 1, -1, -1))
        self.tab = np.zeros((objects, OTW))
        self.tracker = None              # the tracker whose numbers these are (an object's trajectory asks its smoother)
        self.hist = np.zeros((objects, 1 + 5 * CFG["history_n"]))

    def grow(self, cap):
        a = np.zeros((cap, ROW))
        a[:len(self.a)] = self.a
        self.a = a

    def grow_objects(self, n):
        for k in ("tab", "hist"):
            old = getattr(self, k)
            new = np.zeros((n, old.shape[1]))
            new[:len(old)] = old
            setattr(self, k, new)

    def take(self, live=()):
        return self.free.pop()

    def give(self, slot):
        self.free.append(slot)


def _view(a, b, sh):
    """A part of the object's primary row, as an array that is the row's own memory."""
    if len(sh) == 1:
        def get(self):
            s = self.store
            return s.a[int(s.tab[self.h, OT_SLOT]), a:b]
    else:
        def get(self):
            s = self.store
            return s.a[int(s.tab[self.h, OT_SLOT]), a:b].reshape(sh)

    def put(self, v):
        get(self)[...] = v
    return property(get, put)


def _meta(k, kind=float):
    k += _MO

    def get(self):
        s = self.store
        return kind(s.a[int(s.tab[self.h, OT_SLOT]), k])

    def put(self, v):
        s = self.store
        s.a[int(s.tab[self.h, OT_SLOT]), k] = v
    return property(get, put)


def _col(k, kind=float):
    """A number of the object's row of the table."""
    def get(self):
        return kind(self.store.tab[self.h, k])

    def put(self, v):
        self.store.tab[self.h, k] = v
    return property(get, put)


class Obj:
    """An object: its handle in the native hypotheses, what the tracker remembers of it in Python (sensors, class
    vote, verdict, group) and, as attributes, its numbers in the tracker's arrays (`State`): the parts of its
    primary row and its row of the table."""
    __slots__ = ("id", "h", "store", "t0", "_state", "sources", "last_seen", "vote", "nis", "verdict", "coast_s",
                 "velocity_seen", "merge_t", "scores", "cls", "_kin", "grp", "grp_kind", "lhp", "low_run", "sized", "los", "extp")
    xs, Ps, mu, xf, Pf, x, P, e, Pe, st, m = (_view(a, b, sh) for _, a, b, sh in _OFF)
    t = _meta(M_T)                   # the time its numbers are of
    logodds = _meta(M_L)             # that it exists, in the primary's story (`exist`: over all stories)
    hits = _meta(M_HITS, int)
    range_t = _meta(M_RANGE_T)       # the last time a detection said how far it is (not a bearing alone)
    slot = _col(OT_SLOT, int)        # its primary row: its leaf in the best hypothesis of its cluster
    pe = _col(OT_PE)                 # probability that it exists
    pp = _col(OT_PP)                 # ... that the primary's story is the true one
    nleaf = _col(OT_NLEAF, int)      # its hypotheses
    in_map = _col(OT_INMAP, bool)    # it is in the best hypothesis of its cluster
    last_cover = _col(OT_COVER)      # the last time a scan's field covered where the object is
    hidden_t = _col(OT_HIDDEN)       # the last time a scan could not see it for another object or the field's edge
    dp0 = _col(OT_DP0)               # how unknown its angle offset was at birth
    label_code = _col(OT_LABEL)
    n_checks = _col(OT_NCHECKS, int)
    hist_t = _col(OT_HIST_T)         # when its trajectory was last smoothed as a whole

    def __init__(self, oid, h, t, src, coast_s, store):
        self.id = oid
        self.h = h                   # its handle in the native hypotheses: its row of the table (written at its
        self.store = store           # birth by Hyp.cycle)
        store.tab[h, OT_ID] = oid
        self.t0 = t
        self.state = "tentative"
        self.sources = {src}
        self.last_seen = {src: t}
        self.vote = {}
        self.nis = {}
        self.verdict = "unknown"
        self.coast_s = coast_s
        self.velocity_seen = False
        self.merge_t = {}
        self.scores = []             # (sensor, score) of its boxes while it is not confirmed
        self.cls = None              # classvote.result of `vote` without its age (box.py), until the next vote
        self.kin = M._KIN4[None]     # what its kind can do (models._KIN4 of `label`), set with every vote
        self.label_code = 0.0
        self.grp = None              # the group it is of (id), and the group's kind: row | queue | platoon
        self.grp_kind = None
        self.lhp = None              # the prior of its height (lh) its numbers hold: (mean, variance) (Tracker._size)
        self.low_run = 0             # its last sightings in a row that were boxes of the low-score list
        self.sized = False           # what its boxes said of its height has gone into its kind's prior (`_size_learn`)
        self.extp = None             # the priors of its length and width its numbers hold: (L, variance, W, variance)
        self.los = None              # the line of sight to it at its first sighting, and how far it has swung since
                                     # (ground frame: [first, least, most]; `size_learn`)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, v):
        if getattr(self, "_state", None) != v:
            self._state = v
            self.store.tab[self.h, OT_STATE] = _STATES[v]

    @property
    def kin(self):
        return self._kin

    @kin.setter
    def kin(self, v):
        self._kin = v
        self.store.tab[self.h, OT_KIN:OT_KIN + 4] = v

    @property
    def consistency(self):
        """How its earlier predictions held: about 1 when within their own stated uncertainty (None: none checked)."""
        r = self.store.tab[self.h]
        return float(r[OT_CONS]) if r[OT_NCHECKS] else None

    @property
    def pred_err(self):
        r = self.store.tab[self.h]
        return float(r[OT_PERR]) if r[OT_NCHECKS] else None

    @property
    def flags(self):
        return {"unpredictable"} if self.store.tab[self.h, OT_UNPRED] else set()

    @property
    def checks(self):
        """Where it expects to be, written down and not yet checked: [(time, x, y, [P00, P01, P10, P11])]."""
        r = self.store.tab[self.h]
        return [(c[0], c[1], c[2], c[3:]) for c in r[OT_CHK:OT_CHK + 7 * int(r[OT_NCHK])].reshape(-1, 7).tolist()]

    @property
    def history(self):
        """Its trajectory on the ground: [(t, x, y, vx, vy)], the oldest first, smoothed as far as the smoother
        reaches (docs/RADAR.md, section 6). The smoothing is done when this is asked (Hyp.history: a backward pass
        over the primary's story as far as smooth_s); nothing in the stack reads it per message."""
        s = self.store
        tr = s.tracker
        if tr is None:
            r = s.hist[self.h]
            return [tuple(p) for p in r[1:1 + 5 * int(r[0])].reshape(-1, 5).tolist()]
        return tr.hyp.history(s.a, s.hist, self.h, M._native_par(tr.mcfg), tr.cfg["hyp_keep_steps"], tr.cfg["smooth_s"])

    @property
    def vs(self):
        """Its velocity a lag ago, smoothed: (vx, vy, 1 sigma x, y, how long ago), ground frame; None before."""
        r = self.store.tab[self.h]
        return tuple(r[OT_VS:OT_VS + 5].tolist()) if r[OT_HASVS] else None

    @property
    def fs(self):
        """The free estimate's smoothed velocity and its covariance (vx, vy, 00, 01, 11); None before."""
        r = self.store.tab[self.h]
        return tuple(r[OT_FS:OT_FS + 5].tolist()) if r[OT_HASVS] else None

    @property
    def ps_lag(self):
        """The smoothed probability of "it stands" at the lag; None before."""
        r = self.store.tab[self.h]
        return float(r[OT_PSLAG]) if r[OT_HASVS] else None

    @property
    def exist(self):
        return self.pe

    @property
    def label(self):
        return self.vote.get("shown") if self.vote else None

    def combine(self):
        """x, P from the hypotheses and their probabilities (after a change to them)."""
        _NR.radar_combine_rows(self.store.a, [self.slot])

    @property
    def speed(self):
        return math.hypot(self.x[VX], self.x[VY])


class Tracker:
    def __init__(self, ego, cfg=None, model_cfg=None):
        self.ego = ego
        self.cfg = dict(CFG, **(cfg or {}))
        # (any of the detector's further inputs is taken: box.py makes the detections' extra fields)
        self.cfg["_in_any"] = any(self.cfg[k] for k in _IN_KNOBS) or self.cfg["grade_k"] > 0.0
        self.mcfg = dict(M.CFG, **(model_cfg or {}))
        self.next_id = 1
        self.reset()

    def reset(self):
        c = self.cfg
        self.objects = []
        self.store = State(c["rows"])    # the objects' numbers: a row per node of their trees
        self.store.tracker = self
        self.hyp = _NR.Hyp(c["rows"], c["hyp_n_scan"], c["hyp_k_best"], c["hyp_max_leaves"], c["hyp_prune_ln"],
                           c["hyp_keep_steps"], c["hyp_keep_s"], c["hyp_max_cluster"])
        self.by_h = {}               # handle -> object
        self.groups = {}             # group id -> {"kind", "members" (object ids, in order along the line), "line", "rms"}
        self.group_t = None
        self.next_group = 1
        self.src_code = {}           # sensor id -> its number in the rows' records
        self.stats = {}              # sensor id -> counters
        self.sensors = {}            # sensor id -> (last model, time)
        self.bias = {}               # sensor id -> [bias, variance]: what all its objects share (a camera's pitch)
        self.new_rate = {}           # sensor id -> detections per scan no object explained (followed over time)
        self.unexplained = {}        # sensor id -> deque of (t, kind, z): the map's memory
        self.scan_times = {}         # sensor id -> deque of scan times within the map's memory
        self.real_ids = set()        # objects that were confirmed: only their first detections make the map
        self.score_hist = {}         # sensor id -> [true counts, false counts] over the score bins
        self.score_odds = {}         # sensor id -> the log odds of each bin, as far as asked since the last change
        self.bias_t = None
        self.t = None
        self.road_t = None
        self._kinds = {}             # a detection's kind -> its code (sensors.kind_code), as far as asked
        self._labs = {}              # a detection's class as given -> (vehicle / person / not known, models.kinb)
        self._ranges = {}            # sensor id -> it measures range (a microwave radar, a position)
        self._snap = None            # (the list of objects, `look` of them): see `looked`
        self._memrows = {}           # sensor id -> the map's entries of confirmed objects as the native core reads them
        self.low_rate = {}           # sensor id -> low-list boxes per scan that no object took (followed over time)
        self.gallery = deque()       # dropped objects that may come back: (id, time, x, y, vx, vy, var, memory, t0, vote)
        self._xstatic = None         # the part of a scan's xcfg that the configuration alone gives
        self.road_path = None        # the road model's path on the ground (xs, ys), as the box last gave it (`_context`)
        self.size_obs = {}           # kind -> deque of (mean, variance): what its objects' own boxes said of their height
        self.size_fit = {}           # kind -> (mean, variance) of its heights as learnt (`_size_learn`)

    def _xcfg(self, src, sensor):
        """The scan's constants for the detector's further inputs as the native core reads them (chris_native.c
        `XC_*`), or None where nothing of them is taken (the scan is then what it was)."""
        c = self.cfg
        if not c["_in_any"] or sensor.kind != "camera_boxes":
            return None
        st = self._xstatic
        if st is None:
            dark = c["app_dark"]
            st = self._xstatic = (
                ((2.0 if c["flags_far_bearing"] else 1.0) if c["in_flags"] else 0.0, 1.0 if c["in_box_occl"] else 0.0,
                 c["box_occl_margin"],
                 1.0 if c["in_low"] else 0.0, 0.0 if c["low_runs_only"] else c["low_p"], c["low_p_run"], c["low_run_s"]),
                (-1.0 if dark is None else float(dark), c["app_bound"][0], c["app_bound"][1], c["app_gain"], c["app_y_day"],
                 c["app_y_night"]),
                (1.0 if c["in_lamp_colour"] else 0.0, math.radians(c["lamp_colour_deg"]), 1.0 if c["in_lamp_size"] else 0.0,
                 c["lamp_sep_ratio"], c["lamp_sep_rel"], c["ruler_eps"], c["ruler_span"], float(c["ruler_uses"]),
                 1.0 if c["in_width"] else 0.0, c["width_rel"], c["grade_k"], 1.0 if c["in_scale"] else 0.0,
                 c["scale_lamps_k"] if c["in_scale"] else 0.0, 1.0 if c["flags_side"] else 0.0,
                 1.0 if c["sil_bearing"] else 0.0),
                # rows, the detect / miss chain, the pose head (chris_native.c `XC_ROWB` on); nothing of it set: the
                # scan's constants end where they did
                ((c["row_birth"], c["ctx_exist"], c["ctx_vel"]) if c["row_ctx"] else (0.0, 0.0, 0.0))
                + (1.0 if c["det_chain"] else 0.0, c["det_chain_tau_s"], c["det_chain_d"],
                   1.0 if c["low_exist_only"] and c["in_low"] else 0.0,
                   1.0 if c["in_pose_gn"] else 0.0, c["pose_corr_s"], 1.0 if c["in_pose_yaw"] else 0.0,
                   math.radians(c["pose_yaw_floor_deg"]), c["pose_yaw_huber"], c["pose_vote_max"], c["pose_vote_turn"],
                   c["pose_vote_dark"], c["pose_axis_dark_p"], c["pose_vote_half_s"],
                   1.0 if c["pd_learn"] else 0.0, c["pd_learn_n0"], c["pd_learn_keep"], 1.0 if c["exist_shared"] else 0.0)
                if (c["row_ctx"] or c["det_chain"] or (c["low_exist_only"] and c["in_low"]) or c["pd_learn"]
                    or c["exist_shared"] or any(c[k] for k in _POSE_KNOBS) or any(c[k] for k in _RANGE_KNOBS)) else (),
                # the range of a standing thing (chris_native.c `XC_SITER` on)
                (float(c["scale_iter_n"]) if c["scale_iter"] else 0.0, c["scale_seed_sigma"] if c["scale_seed"] else 0.0,
                 c["scale_seed_k"], 1.0 if c["near_axis"] else 0.0) if any(c[k] for k in _RANGE_KNOBS) else ())
        ek = getattr(sensor, "emb_kind", 0) if c["in_app"] else 0
        tab = c["app_tables"].get(ek) or c["app_tables"].get(str(ek))
        if tab is None:
            ek, tab = 0, ((0.0, 1.0) * 4,) * 2           # a kind of vector there is no table for: not used
        return st[0] + (max(c["low_false_per_scan"], self.low_rate.get(src, 0.0)), 1.0 if ek else 0.0, float(ek)) + st[1] \
            + tuple(tab[0]) + tuple(tab[1]) + st[2] + st[3] + st[4]

    def look(self, objs):
        """Several objects' numbers at once, as lists (the upkeep and the message read every object once per
        message): their primary rows, their rows of the table up to the predictions written down, and the start of
        each primary row's record (`M_T` .. `M_RANGE_T`)."""
        if not objs:
            return [], [], []
        s = self.store
        if len(objs) < 8:                                # (a few objects: one by one costs less than numpy's indexing)
            tab, a = s.tab, s.a
            tb = [tab[o.h, :OT_NCHK].tolist() for o in objs]
            slots = [int(r[OT_SLOT]) for r in tb]
            return slots, tb, [a[k, _MO:_MO + M_RANGE_T + 1].tolist() for k in slots]
        tb = s.tab[[o.h for o in objs], :OT_NCHK]
        slots = tb[:, OT_SLOT].astype(np.intp)
        return slots.tolist(), tb.tolist(), s.a[slots, _MO:_MO + M_RANGE_T + 1].tolist()

    def looked(self):
        """`look` of all objects, in their order: what the upkeep read at its start where nothing has changed the
        list of objects since (objects dropped by the upkeep's own rules are out of both), else read now."""
        sn = self._snap
        if sn is None or sn[0] is not self.objects or len(sn[1]) != len(self.objects):
            sn = self._snap = (self.objects,) + self.look(self.objects)
        return sn[1:]

    def peek(self, o, t):
        """Where the object is believed to be at time t, without changing it: (x, P, mu) (for tools)."""
        x, P, mu = np.empty((1, N)), np.empty((1, N, N)), np.empty((1, 3))
        _NR.radar_peek(self.store.a, [o.slot], [float(t) - o.t], [o.kin], M._native_par(self.mcfg), x, P, mu)
        return x[0], P[0], mu[0]

    @staticmethod
    def _shown(vx, vy, s00, s01, s11):
        det = s00 * s11 - s01 * s01
        if det <= 0.0:
            return 0.0
        return (s11 * vx * vx - 2.0 * s01 * vx * vy + s00 * vy * vy) / det

    @staticmethod
    def shown_moving(o):
        """How far the object's sightings by themselves show that it moves: the distance of "no speed" from the
        free estimate of its velocity, in that estimate's own uncertainty (squared; 9 is one chance in a
        hundred for a thing that stands). The free estimate a lag ago, smoothed, where there is one (what the
        later sightings say of the velocity then is far firmer than the newest sighting's own)."""
        if o.fs is not None:
            vx, vy, s00, s01, s11 = o.fs
        else:
            v, S = o.xf[VX:VY + 1], o.Pf[VX:VY + 1, VX:VY + 1]
            vx, vy, s00, s01, s11 = v[0], v[1], S[0, 0], S[0, 1], S[1, 1]
        det = s00 * s11 - s01 * s01
        if det <= 0.0:
            return 0.0
        return float((s11 * vx * vx - 2.0 * s01 * vx * vy + s00 * vy * vy) / det)

    def _pose(self, t):
        p = self.ego.pose_at(t)
        if p is None:
            return None
        v = self.ego.velocity_at(t)
        return (p[0], p[1], p[2], v[0], v[1], v[2])

    # ---- the detector's confidence ------------------------------------------------------------------------
    @staticmethod
    def _score(d):
        at = d.get("attrs") or {}
        v = at.get("class_conf", at.get("prob"))
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0 else None

    def _hist(self, src):
        h = self.score_hist.get(src)
        if h is None:
            c = self.cfg
            n = c["score_bins"]
            mid = (np.arange(n) + 0.5) / n
            (mt, st), (mf, sf), wgt = c["score_prior"]
            ht, hf = np.exp(-0.5 * ((mid - mt) / st) ** 2), np.exp(-0.5 * ((mid - mf) / sf) ** 2)
            h = self.score_hist[src] = [(wgt * ht / ht.sum()).tolist(), (wgt * hf / hf.sum()).tolist()]
        return h

    def _bin_odds(self, src, b):
        """Log odds a box whose score falls into bin b adds for "a real thing". Per bin, kept until the sensor's
        histograms change (`_score_learn`). (A box without a score adds 0: Hyp.cycle.)"""
        c = self.cfg
        tab = self.score_odds.get(src)
        if tab is None:
            tab = self.score_odds[src] = [None] * c["score_bins"]
        v = tab[b]
        if v is None:
            ht, hf = self._hist(src)
            r = math.log(((ht[b] + 0.5) / (_npsum(ht) + 0.5 * len(ht))) / ((hf[b] + 0.5) / (_npsum(hf) + 0.5 * len(hf))))
            v = tab[b] = c["score_gain"] * max(-c["score_max"], min(c["score_max"], r))
        return v

    def _kind_code(self, kind):
        v = self._kinds[kind] = kind_code(kind)
        return v

    def _lab_row(self, raw):
        """What a detection's class says, as the native core reads it: (vehicle 1 / person -1 / not known 0) and
        what the kind can do for an object started by it (models.kinb)."""
        lab = classvote.norm_label(raw)
        r = (0.0 if lab is None else (1.0 if lab in M.VEHICLES else -1.0),) + tuple(M.kinb(lab))
        try:
            if len(self._labs) < 256:
                self._labs[raw] = r
        except TypeError:                                # not a thing a dict takes as a key
            pass
        return r

    def _range_seen(self, o):
        """The last time a sensor that measures range saw the object, into its row of the table (`Hyp.cycle`: the
        consider parameters)."""
        rg = self._ranges
        self.store.tab[o.h, OT_RANGE] = max((ts for s, ts in o.last_seen.items() if rg.get(s)), default=-1e18)

    def _score_learn(self, scores, real):
        c = self.cfg
        for src, sc in scores:
            self.score_odds.pop(src, None)
            h = self._hist(src)[0 if real else 1]
            h[min(int(sc * c["score_bins"]), c["score_bins"] - 1)] += 1.0
            if _npsum(h) > c["score_keep"]:
                h[:] = [v * 0.5 for v in h]

    # ---- one scan -----------------------------------------------------------------------------------------
    def process(self, src, t, sensor, dets):
        """dets: [{"z": array, "R": variances (1-D), "attrs": {...}}]."""
        c = self.cfg
        st = self.stats.setdefault(src, {"scans": 0, "dets": 0, "assigned": 0, "started": 0, "late": 0})
        st["scans"] += 1
        if not getattr(sensor, "extended", None):
            st["dets"] += len(dets)
        self.sensors[src] = (sensor, t)
        b = None
        if getattr(sensor, "has_bias", False):
            # the camera's pitch error: its slow part (value, variance) and its fast part (value, variance, time)
            b = self.bias.setdefault(src, [0.0, sensor.bias_var, 0.0, 0.0, -1e18])
            sensor.bias, sensor.bias_var = b[0], b[1]
        rg = sensor.kind in ("microwave_radar", "position")
        if self._ranges.get(src) is not rg:
            self._ranges[src] = rg
            for o in self.objects:
                self._range_seen(o)
        pose = self._pose(t)
        if pose is None:
            return
        self._snap = None
        self.t = t if self.t is None else max(self.t, t)
        if getattr(sensor, "extraneous", None) is not None:
            sensor.extraneous = max(sensor.extraneous, self.new_rate.get(src, 0.0))
        code = self.src_code.setdefault(src, len(self.src_code) + 1)
        if b is not None:                                # how much the camera's pitch moves: by how fast the car drives
            cc = sensor.cfg
            f = min(math.hypot(pose[3], pose[4]) / cc["pitch_gm_full_mps"], 1.0)
            sensor.gm = (math.radians(cc["pitch_gm_stand_deg"] + f * (cc["pitch_gm_deg"] - cc["pitch_gm_stand_deg"])) ** 2,
                         cc["pitch_gm_tau_s"])
        if getattr(sensor, "extended", None):
            dets = self._cells(sensor, pose, t, [o for o in self.objects if not t < o.t - c["oosm_max_s"]], dets)
            st["dets"] += len(dets)
        elif type(dets) is not list:
            dets = list(dets)
        rows = self._memrows.get(src, False)             # the map's entries as rows, made anew when the map changed
        if rows is False:
            mem, real = self.unexplained.get(src), self.real_ids
            rows = self._memrows[src] = [(len(zz), zz[0], zz[1], zz[2] if len(zz) > 2 else 0.0, oid)
                                         for (_, _, zz, oid) in mem or () if oid in real] or None
        g, bw = c["gate"], c["map_bw"]
        acfg = (g.get(1, 16.0), g.get(2, 16.0), g.get(3, 16.0), g.get(4, 16.0), c["feat_sigma"], c["feat_same"],
                c["feat_span"], bw[0], bw[1], bw[2], c["feat_max_s"], c["common_min"], c["common_pair"],
                1.0 if c["occlusion"] else 0.0, c["occl_from"], c["occl_full"], c["occl_floor"], c["occl_floor_radar"],
                c["split_rel"])
        bcfg = (c["new_exist"], c["p_any_side"], c["side_m"], c["with_rel"], c["with_std"], self.mcfg["p_ca0"],
                c["mu_floor"], max(0, c["max_objects"] - len(self.objects)))
        # the scan in one native call (chris_native.c `Hyp.cycle`): the objects that are not too far ahead of its
        # time take part (what the car's own movement over the interval is not sure about goes into where each is;
        # whether its moving hypotheses hold its height and angle offset: the consider parameters, see `CFG`); the
        # detections as the native core reads them; what the scan did to each object goes into its row of the table
        store, par = self.store, M._native_par(self.mcfg)
        objs = self.objects
        args = ([o.h for o in objs], float(t), self.ego.motion_noise, pose, sensor.native(), code, dets,
                (self._kinds, self._kind_code, self._labs, self._lab_row, lambda k: self._bin_odds(src, k), c["score_bins"]),
                rows, float(len(self.scan_times.get(src) or ())), acfg, par, (c["mu_hold_hits"], c["mu_floor"]), bcfg, b,
                (c["oosm_max_s"], 1.0 if c["consider_moving"] else 0.0, c["consider_range_s"],
                 1.0 if c["consider_stand"] else 0.0))
        xcfg = self._xcfg(src, sensor) if c["_in_any"] else None
        if xcfg is not None and c["scale_exact"]:
            # the car's pose at the far end of each scale rate's baseline (the scan's boxes share a few baselines:
            # the detector's earlier passes)
            sp = []
            for d in dets:
                dt = d.get("sdt")
                if dt and dt > 0.0 and dt not in sp and len(sp) < _SPREV:
                    sp.append(dt)
            tail = []
            for dt in sp:
                p = self.ego.pose_at(t - dt)
                if p is not None:
                    tail += [float(dt), float(p[0]), float(p[1]), float(p[2])]
            if tail:
                xcfg = xcfg + (float(len(tail) // 4),) + tuple(tail)
        if xcfg is not None:                             # the detector's further inputs (docs/RADAR.md, section 8)
            args += (xcfg,)
            if c["row_ctx"] and dets and objs:
                self._row_ctx(dets, sensor, pose)
        while True:
            res = self.hyp.cycle(store.a, store.tab, store.hist, *args)
            if not isinstance(res, int):
                break
            if res < 0:                                  # the table is too small for the objects this scan may start
                store.grow_objects(max(2 * len(store.tab), 16 - res))
            else:                                        # the state array is too small for this scan: more rows
                store.grow(max(2 * len(store.a), len(store.a) + res + 64))
        late, hits, births, dead, bias = res
        st["late"] += late
        if bias is not None:                             # the sensor's pitch error after the scan
            b[:] = bias
            sensor.bias, sensor.bias_var = bias[0], bias[1]
        hit = []
        for i, j in hits:                                # (object, detection): the object took it in its primary's story
            o = objs[i]
            self._after_update(o, dets[j], sensor, src, t)
            hit.append(o)
        st["assigned"] += len(hit)
        kept = hit
        if xcfg is not None and c["in_low"]:
            # the low-score boxes no object took, per scan, followed over time: the low list's own "something else"
            nlow = sum(1 for d in dets if (d.get("attrs") or {}).get("low"))
            took = sum(1 for _, j in hits if (dets[j].get("attrs") or {}).get("low"))
            r0 = self.low_rate.get(src, 0.0)
            self.low_rate[src] = r0 + c["new_rate_gain"] * (nlow - took - r0)
        if dead:
            gone = set(dead)
            for o in self.objects:
                if o.h in gone and self.by_h.get(o.h) is o:
                    del self.by_h[o.h]
                    if o.state == "tentative" and o.scores:
                        self._score_learn(o.scores, False)   # never confirmed and gone: its boxes were false ones
            self.objects = [o for o in self.objects if o.h not in gone]
            kept = [o for o in hit if o.h not in gone]
        memq = self.unexplained.setdefault(src, deque())
        times = self.scan_times.setdefault(src, deque())
        times.append(t)
        while times and times[0] < t - c["map_s"]:
            times.popleft()
        while memq and memq[0][0] < t - c["map_s"]:
            memq.popleft()
            self._memrows.pop(src, None)
        for h, j in births:
            d = dets[j]
            back = self._reid(h, t, xcfg) if xcfg is not None and c["in_reid"] and self.gallery else None
            if back is None:
                o = Obj(self.next_id, h, t, src, sensor.coast_s, store)
                self.next_id += 1
            else:                                        # a dropped object, back: its id, its age, its class vote
                o = Obj(back[0], h, t, src, sensor.coast_s, store)
                o.t0, o.vote = back[8], back[9]
                st["reid"] = st.get("reid", 0) + 1
            if rg:
                self._range_seen(o)
            sc = self._score(d)
            if sc is not None:
                o.scores.append((src, sc))
            o.velocity_seen = sensor.velocity
            lab = classvote.norm_label((d.get("attrs") or {}).get("class"))
            kb = M.kinb(lab)
            o.lhp = (math.log(kb[7]), kb[8] * kb[8])     # the height prior it was born with (chris_native.c r_birth)
            if kb[10] > 0.0 and lab in M.VEHICLES:       # ... and a vehicle's extent (r_extent_birth)
                o.extp = (kb[10], (0.22 * kb[10]) ** 2, kb[9], (0.10 * kb[9]) ** 2)
            self._vote(o, d, src, t)
            kb = (d.get("kind") or "").rstrip("s")
            if kb in ("rb", "rbh"):
                # the map lives in the space of the ground angle, the bearing and the log height: a box with a
                # scale-rate row enters without it, a box cut at the bottom (its rows are another space) not at all
                memq.append((t, kb, np.asarray(d["z"], float)[:len(kb)], o.id))
                self._memrows.pop(src, None)
            self.objects.append(o)
            self.by_h[h] = o
            st["started"] += 1
            if c["ctx_prior"] and not rg and lab in M.VEHICLES:
                self._context(o, pose)
        if hit or births:
            # what follows the sightings, in one native call (chris_native.c `Hyp.after`): each sighted object's
            # earlier predictions checked against where it turned out to be (one that keeps being elsewhere has
            # its uncertainty opened) and its trajectory carried on; where it expects to be at the horizons ahead
            # written down; its past smoothed back to the lag (docs/RADAR.md, section 6: the velocity a lag ago);
            # a new object's first point
            hz = c["check_horizons"]
            self.hyp.after(store.a, store.tab, store.hist, [o.h for o in hit], [o.h for o in kept], [h for h, _ in births],
                           float(t), par,
                           (c["check_window_s"], c["inconsistent"], c["inflate"], c["history_s"], len(hz)) + tuple(hz)
                           + (0.0,) * (4 - len(hz))
                           + (1.0 if c["smooth"] else 0.0, c["hyp_keep_steps"], c["smooth_s"], 1.5 * c["smooth_lag_s"],
                              c["smooth_lag_s"], c["smooth_hist_s"], c["smooth_every_s"]),
                           None if c["smooth_for"] != "read" else [o.h for o in kept if o.verdict != "moving"])
        r0 = self.new_rate.get(src, 0.0)
        self.new_rate[src] = r0 + c["new_rate_gain"] * (len(births) - r0)

    def _cells(self, sensor, pose, t, objs, returns):
        """The returns of a sensor that gives several per object, as detections: one per cell (radar_cells), the
        returns that lie on a known object being that object's cell."""
        eps_m, eps_v, margin = sensor.extended
        sr, sa, sv = sensor.sigma
        known = [o for o in objs if o.state != "tentative" or o.hits >= 2]
        cells = _NR.radar_cells(returns, eps_m, eps_v, self.store.a, [o.slot for o in known], [t - o.t for o in known],
                                sensor.native(), pose, margin) if returns else ()
        return [{"z": np.array([r, az, rr]), "R": np.array([sr * sr / n, sa * sa / n, sv * sv / n]), "attrs": {},
                 "cell": (float(n), sxx, sxy, syy)} for r, az, rr, n, sxx, sxy, syy in cells]

    def _vote(self, o, d, src, t):
        at = d.get("attrs") or {}
        if at.get("class") is not None:
            classvote.vote(o.vote, at.get("class"), at.get("class_conf", at.get("prob")), src, t, at.get("weight", 1.0))
            o.cls = None
            o.kin = M._KIN4.get(o.label) or M._KIN4[None]
            la = o.label
            o.label_code = 0.0 if la is None else (1.0 if la in M.VEHICLES else -1.0)
            self._size(o)
            if self.cfg["ext_size"] and o.extp is not None:
                self._ext_size(o)

    def _ext_size(self, o):
        """A vehicle's length and width priors follow its height (`ext_size`): the height its class vote's prior
        gives it (`_size`) until its own boxes have said more (its numbers' lh known ext_size_learnt times better
        than that prior: a standing thing a scale rate has ranged), then its own. That height gives the size of a
        vehicle that tall (models.extent_of_height), and where that differs from the priors its extent holds they
        are exchanged in every story of it (the old one's information out, the new one's in: radar_ext_prior), as
        `_size` does with the height. (A height not yet learnt moves with every box's noise and with the hypotheses'
        weights: the size does not follow that.)"""
        if o.label_code <= 0.0:
            return
        c = self.cfg
        row = self.store.a[o.slot]
        lh, vh = float(row[_XLH]), max(float(row[_PLH]), 0.0)
        if o.lhp is not None and not vh < c["ext_size_learnt"] * o.lhp[1]:
            lh, vh = o.lhp
        L1, vL1, W1, vW1 = M.extent_of_height(math.exp(lh), math.sqrt(vh))
        L0, vL0, W0, vW0 = o.extp
        st = c["ext_size_step"]
        rows = None
        for k, m0, v0, m1, v1 in ((0, L0, vL0, L1, vL1), (1, W0, vW0, W1, vW1)):
            if abs(m1 - m0) < st * m0 and abs(math.log(v1 / v0)) < 4.0 * st:
                continue
            if rows is None:
                rows = self.hyp.rows(o.h)
                p = self.ego.pose_at(o.t)                # the face the camera saw stays where it is (radar_ext_prior)
                only = all(getattr(self.sensors.get(q, (None,))[0], "kind", "") == "camera_boxes" for q in o.sources)
                cam = (float(p[0]), float(p[1])) if p is not None and only else (math.nan, math.nan)
            _NR.radar_ext_prior(self.store.a, rows, k, 1.0 / v1 - 1.0 / v0, m1 / v1 - m0 / v0, cam[0], cam[1])
            if k == 0:
                L0, vL0 = m1, v1
            else:
                W0, vW0 = m1, v1
        if rows is not None:
            o.extp = (L0, vL0, W0, vW0)

    def _size(self, o):
        """The prior of the object's height follows what the class detector says of it (models.size_prior of the
        vote's tallies). An object is born with the prior of its first box's label; when the boxes that follow
        say another kind (a van is a `truck` in two boxes of three and a `car` in the third), the prior its numbers
        hold is replaced by the new one in every story of it: the exact Gaussian exchange (the old prior's
        information out, the new one's in) on lh, which moves the place through what lh and the place share (the
        box height's range). Without it a van born from a `car` box was read at the range a car of its box height
        would have: 0.6 of the truth, the box height being the strongest range beyond 40 m (docs/RADAR.md, section 4)."""
        if o.lhp is None or not o.vote or not o.vote.get("w") or not self.cfg["size_from_class"]:
            return
        m1, v1 = M.size_prior(o.vote["w"], self.size_fit) if self.size_fit else M.size_prior(o.vote["w"])
        m0, v0 = o.lhp
        if abs(m1 - m0) < 0.02 and abs(math.log(v1 / v0)) < 0.05:
            return
        a, b = 1.0 / v1 - 1.0 / v0, m1 / v1 - m0 / v0
        rows = self.hyp.rows(o.h)
        _NR.radar_lh_prior(self.store.a, rows, a, b)
        o.lhp = (m1, v1)

    def _offsets(self, xs, ys, pose):
        """The places' distances beside the road's path (left positive) and the path's direction there: from the road
        model's path on the ground where the box has given one (the direction None otherwise: beside the car's own
        line straight on)."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        rp = self.road_path
        if rp is None or len(rp[0]) < 2:
            return -math.sin(pose[2]) * (xs - pose[0]) + math.cos(pose[2]) * (ys - pose[1]), None
        px, py = rp
        ax, ay = px[:-1], py[:-1]
        ex, ey = px[1:] - ax, py[1:] - ay
        L2 = np.maximum(ex * ex + ey * ey, 1e-12)
        f = ((xs[:, None] - ax) * ex + (ys[:, None] - ay) * ey) / L2
        lo, hi = np.zeros_like(f), np.ones_like(f)
        lo[:, 0], hi[:, -1] = -np.inf, np.inf            # the end segments reach on beyond the polyline
        fc = np.clip(f, lo, hi)
        qx, qy = xs[:, None] - (ax + fc * ex), ys[:, None] - (ay + fc * ey)
        k = np.argmin(qx * qx + qy * qy, axis=1)
        i = np.arange(len(xs))
        return (ex[k] * qy[i, k] - ey[k] * qx[i, k]) / np.sqrt(L2[k]), np.arctan2(ey[k], ex[k])

    def _context(self, o, pose):
        """A new vehicle's "it stands" from where it was first seen (`ctx_prior`): the vehicles already judged at its
        distance beside the road's path, by their verdicts. A parked street says of the next car that appears at the
        kerb that it is parked, before that car's own boxes can (at 60 m a box in three says next to nothing about
        the range's rate): what the hypotheses' prior left open there, "the same car, driving with us" explained
        the next car of the row, and the object hopped along the row at the car's speed (docs/RADAR.md, sections 5
        and 9). Raised only: among driving vehicles a new one keeps its kind's prior (a stopped car in the lane)."""
        c = self.cfg
        don = [q for q in self.objects if q is not o and q.label_code > 0.0 and q.state != "tentative"
               and q.verdict != "unknown"]
        if not don:
            return
        d, axis = self._offsets([float(o.x[PX])] + [float(q.x[PX]) for q in don],
                                [float(o.x[PY])] + [float(q.x[PY]) for q in don], pose)
        w = np.exp(-0.5 * ((d[1:] - d[0]) / c["ctx_band_m"]) ** 2)
        st = np.array([q.verdict == "stationary" for q in don])
        ns, nm = float(w[st].sum()), float(w[~st].sum())
        p0 = float(o.mu[STAND])
        p = min((ns + c["ctx_n"] * p0) / (ns + nm + c["ctx_n"]), c["ctx_max"])
        if p > p0 + 1e-3:
            rows = self.hyp.rows(o.h)
            _NR.radar_stand_prior(self.store.a, rows, p)
            if axis is not None and p >= 0.5 and c["road_prior"] and abs(float(d[0])) <= c["road_prior_off_m"]:
                # ... and a vehicle that stands beside the road stands along it: the road's axis there as its
                # heading's prior at once (`road_prior` would give it within road_prior_s; until then it had the
                # car's own heading, on a bend half a radian off, and its rectangle lay across the lane)
                los = math.atan2(float(o.x[PY]) - pose[1], float(o.x[PX]) - pose[0])
                _NR.radar_heading_prior(self.store.a, rows, float(axis[0]), los, c["road_prior_std"], c["road_prior_snap"],
                                        c["road_prior_mirror"])

    def _row_ctx(self, dets, sensor, pose):
        """How much the place each box of the scan was seen at is a row of standing vehicles (`row_ctx`): the box's
        own ground point (its ground angle with the camera's known pitch error taken out, its bearing) beside the
        road's path, against the vehicles already judged there, by their verdicts: q = n_s / (n_s + n_m + ctx_n),
        the count of `_context`. Written into the detection (`p`[0]) for the native scan: the density of new cars
        there, a new object's existence and what it may be doing (chris_native.c `XC_ROWB`, r_birth)."""
        c = self.cfg
        don = [q for q in self.objects if q.label_code > 0.0 and q.state != "tentative" and q.verdict != "unknown"]
        if not don:
            return
        hc, bias, cp, sp = sensor.hc, sensor.bias, math.cos(pose[2]), math.sin(pose[2])
        js, xs, ys = [], [], []
        for j, d in enumerate(dets):
            k = d["kind"]
            if k[:2] != "rb" or (d.get("attrs") or {}).get("low"):
                continue
            lab = self._labs.get(d["attrs"].get("class"))
            if lab is None:
                lab = self._lab_row(d["attrs"].get("class"))
            if lab[0] <= 0.0:                            # (vehicles: a person in a row of cars is not one more car)
                continue
            z = d["z"]
            r = hc / math.tan(max(float(z[0]) - bias, 0.003))
            b = float(z[1])
            js.append(j)
            xs.append(pose[0] + r * (cp * math.cos(b) - sp * math.sin(b)))
            ys.append(pose[1] + r * (sp * math.cos(b) + cp * math.sin(b)))
        if not js:
            return
        n = len(js)
        off, _ = self._offsets(xs + [float(q.x[PX]) for q in don], ys + [float(q.x[PY]) for q in don], pose)
        w = np.exp(-0.5 * ((off[n:][None, :] - off[:n][:, None]) / c["ctx_band_m"]) ** 2)
        st = np.array([q.verdict == "stationary" for q in don])
        ns, nm = w[:, st].sum(axis=1), w[:, ~st].sum(axis=1)
        q = ns / (ns + nm + c["ctx_n"])
        for i, j in enumerate(js):
            if q[i] > 1e-3:
                d = dets[j]
                d["p"] = (float(q[i]),) + (d["p"][1:] if "p" in d else (0.0,) * 8)

    def _size_learn(self, o):
        """What a standing vehicle's own boxes have said of its height goes into its kind's prior (`size_learn`). The
        object's standing hypothesis holds its height as prior x likelihood (both Gaussian in the logarithm): the
        prior it was given taken out leaves what its boxes said (a thing the car passed is ranged by the car's own
        travel: its box height at that range is its size). radar_size_fit makes the kind's mean and spread of the
        objects' sayings; `_size` gives them to every object of the kind from its next vote on."""
        c = self.cfg
        o.sized = True
        lab = o.label
        if (o.lhp is None or lab not in M.VEHICLES or o.verdict != "stationary" or o.hits < c["size_learn_hits"]
                or o.state == "tentative" or o.los is None or o.los[2] - o.los[1] < math.radians(c["size_learn_swing_deg"])):
            return
        k = M.KIN[lab]
        m0, v0 = self.size_fit.get(lab) or (math.log(k["h"]), k["h_rel"] ** 2)
        mp, vp = o.lhp
        if abs(mp - m0) > 0.03:                          # its prior is a mixture of kinds (a van): not this kind's
            return
        lh, v = float(o.xs[STAND, M.LH]), float(o.Ps[STAND, M.LH, M.LH])
        if not (0.0 < v < 0.95 * vp):
            return
        pr = 1.0 / v - 1.0 / vp
        q = self.size_obs.setdefault(lab, deque(maxlen=c["size_learn_keep"]))
        q.append(((lh / v - mp / vp) / pr, 1.0 / pr))
        self.size_fit[lab] = _NR.radar_size_fit(list(q), math.log(k["h"]), k["h_rel"] ** 2, c["size_learn_n0"],
                                               c["size_learn_floor"], c["size_learn_shift"])

    def _after_update(self, o, d, sensor, src, t):
        """What a detection does to an object besides its numbers (those, its existence and its sightings are the
        native scan's; the checks of its predictions and its trajectory follow in `Hyp.after`): its record, the
        detector's score, its sensors, class vote."""
        n = float(self.store.a[o.slot, _MO + M_NIS])
        if n == n:
            o.nis[src] = n if o.nis.get(src) is None else o.nis[src] + 0.2 * (n - o.nis[src])
        low = (d.get("attrs") or {}).get("low")          # a box of the low-score list: seen, weakly
        o.low_run = o.low_run + 1 if low else 0
        sc = None if low else self._score(d)
        if sc is not None:
            if o.state == "tentative":
                o.scores.append((src, sc))
            else:
                self._score_learn([(src, sc)], True)
        o.sources.add(src)
        o.last_seen[src] = t
        if self._ranges[src]:
            self._range_seen(o)
        o.coast_s = max(o.coast_s, sensor.coast_s) if len(o.sources) == 1 else min(o.coast_s, sensor.coast_s)
        o.velocity_seen = o.velocity_seen or sensor.velocity
        if self.cfg["size_learn"]:                       # how far the line of sight to it has swung (`_size_learn`)
            p = self.ego.pose_at(t)
            if p is not None:
                x = o.x
                a = math.atan2(float(x[PY]) - p[1], float(x[PX]) - p[0])
                if o.los is None:
                    o.los = [a, 0.0, 0.0]
                else:
                    da = (a - o.los[0] + math.pi) % (2.0 * math.pi) - math.pi
                    o.los[1], o.los[2] = min(o.los[1], da), max(o.los[2], da)
        self._vote(o, d, src, t)

    # ---- life ---------------------------------------------------------------------------------------------
    def manage(self, T):
        c = self.cfg
        keep = []
        pose = self.ego.pose_at(T)
        # "out of view" is what scans say that came and did not cover the object's place; while no scan comes
        # at all (frames dropped, the detector shed by the load governor) nothing has left any field
        scan = T if self.t is None else min(T, self.t)
        self._snap = None
        slots, tb, rec = self.looked()
        lasts = [max(o.last_seen.values()) for o in self.objects]
        life = self._life(T, pose, scan, slots, tb, rec, lasts)
        sn = ([], [], [])                                # ... of the objects that stay
        delete_exist, view_gap, seen_max = c["delete_exist"], c["view_gap_s"], c["seen_max_s"]   # (out of the loop)
        for o, (p_st, finite, _, sd, rng), s, r, m, last in zip(self.objects, life, slots, tb, rec, lasts):
            exist, cover, hidden, hits = r[OT_PE], r[OT_COVER], r[OT_HIDDEN], int(m[M_HITS])
            if exist < delete_exist or not finite:
                continue
            if scan - cover <= view_gap:
                # a sensor is looking at where it should be: it lives by being seen
                # (the time it was hidden behind another object does not count: nothing was to be seen)
                hid = hidden > last
                if T - max(last, hidden) > max(o.coast_s, seen_max):
                    continue
                # (a member of a queue hidden by the queue itself is kept as long as a remembered object)
                if hid and (T - last > (c["memory_s"] if o.grp_kind == "queue" else c["hidden_s"])
                            or (sd is not None and sd > c["hidden_std_m"])):
                    continue                             # hidden for too long: where it is is no longer known
                if o.state == "tentative":
                    if exist >= c["confirm_exist"] and hits >= c["confirm_hits"] and r[OT_INMAP]:
                        o.state = "confirmed"
                        self.real_ids.add(o.id)
                        self._memrows.clear()
                        self._score_learn(o.scores, True)
                        o.scores = []
                else:
                    o.state = "coast" if T - last > 0.3 or o.low_run >= c["low_coast"] else "confirmed"
            else:
                # out of every field: remembered while where it is remains worth knowing
                if o.state == "tentative":
                    if scan - cover > c["memory_tentative_s"]:
                        continue
                else:
                    if T - cover > c["memory_s"] or sd > c["memory_std_m"] or rng > c["memory_range_m"]:
                        continue
                    o.state = "memory"
            self._judge(o, T, p_st, hits)
            keep.append(o)
            sn[0].append(s)
            sn[1].append(r)
            sn[2].append(m)
        kept = {o.id for o in keep}
        for o in self.objects:                           # never confirmed and gone: its boxes were false ones
            if o.id not in kept and o.state == "tentative" and o.scores:
                self._score_learn(o.scores, False)
        gone = [o for o in self.objects if o.id not in kept]
        self.objects = keep
        self._snap = (keep,) + sn
        if c["size_learn"]:
            for o in gone + keep:                        # an object that has left the view, or is gone: its size is said
                if not o.sized and (o.state == "memory" or o.id not in kept):
                    self._size_learn(o)
        for o in gone:
            self._remember(o, T)
            self._drop(o)
        if len(self.real_ids) > 4096:
            self.real_ids = {o.id for o in keep if o.state != "tentative"}
            self._memrows.clear()
        self._merge(T)
        self._register(T)
        self._groups(T, pose)

    def _groups(self, T, pose):
        """Structure among the objects (radar_groups): standing things in a line beside the road are a row, in the
        car's own lane a queue, things driving one behind the other at one speed a platoon. A group keeps its id
        while it keeps two of its members. A row or a queue supports its members (radar_group_support, once per
        `group_s`): each stands on the line the OTHERS make, and standing is likelier for it than for a thing by
        itself; a member whose own sightings show a speed is not of the group."""
        c = self.cfg
        nt = [o.state != "tentative" for o in self.objects]
        objs = [o for o, k in zip(self.objects, nt) if k]
        prev = {o.id: o.grp for o in objs if o.grp is not None}
        for o in self.objects:
            o.grp = o.grp_kind = None
        if not c["groups"] or len(objs) < 2 or pose is None:
            self.groups = {}
            return
        info = []
        slots, tb, _ = self.looked()
        slots, tb = ([v for v, k in zip(q, nt) if k] for q in (slots, tb))
        a = self.store.a
        for o, s, r in zip(objs, slots, tb):
            v = o.verdict
            if v == "stationary":
                kind = 1.0
            elif v == "moving":                          # it drives: its speed
                vx, vy = a[s, _GCOLS[6]:_GCOLS[7] + 1].tolist()
                kind = 2.0 if math.hypot(vx, vy) > c["group_move_mps"] else 0.0
            else:
                # it stands as far as is known: its sightings by themselves show no speed (`shown_moving`: the free
                # estimate a lag ago, smoothed, where there is one), and "it stands" is believed
                sm = self._shown(*(r[OT_FS:OT_FS + 5] if r[OT_HASVS] else a[s, _GCOLS[:5]].tolist()))
                kind = 1.0 if sm < c["group_still_m2"] and a[s, _GCOLS[5]] > 0.3 else 0.0
            info.append((kind, 1.0 if o.state in ("confirmed", "coast", "memory") else 0.0))
        found = _NR.radar_groups(self.store.a, slots, info, pose,
                                 (c["group_gap_stand_m"], c["group_gap_move_m"], c["group_dv"], c["group_line_m"],
                                  c["group_line_rel"], c["group_lane_m"], c["group_min"]))
        support = self.group_t is None or T - self.group_t >= c["group_s"]
        groups, used = {}, set()
        for kind, mem, line, rms in found:
            votes = {}
            for k in mem:
                g = prev.get(objs[k].id)
                if g is not None and g not in used:
                    votes[g] = votes.get(g, 0) + 1
            gid = max(votes, key=lambda g: (votes[g], -g)) if votes and max(votes.values()) >= 2 else None
            if gid is None:
                gid = self.next_group
                self.next_group += 1
            used.add(gid)
            name = ("row", "queue", "platoon")[kind - 1]
            groups[gid] = {"kind": name, "members": [objs[k].id for k in mem], "line": line, "rms": rms}
            for k in mem:
                objs[k].grp, objs[k].grp_kind = gid, name
            if support and kind != 3 and len(mem) >= 3 and c["group_support"]:
                a = self.store.a
                for k in mem:
                    o = objs[k]
                    if o.verdict == "moving" or self.shown_moving(o) >= c["group_still_m2"]:
                        continue
                    # the line the others make (a member does not support itself), and how well their mean place
                    # across it is known (their own uncertainties across the line, as independent)
                    oth = [objs[q] for q in mem if q != k]
                    mx, my = sum(float(q.x[PX]) for q in oth) / len(oth), sum(float(q.x[PY]) for q in oth) / len(oth)
                    nx, ny = -line[3], line[2]
                    vl = sum(nx * nx * q.P[PX, PX] + 2.0 * nx * ny * q.P[PX, PY] + ny * ny * q.P[PY, PY] for q in oth) / len(oth) ** 2
                    _NR.radar_group_support(a, self.hyp.rows(o.h), (mx, my, line[2], line[3]),
                                            math.sqrt(c["group_sigma_m"] ** 2 + rms * rms + float(vl)), c["group_gain"], c["mu_floor"])
        if support:
            self.group_t = T
        self.groups = groups

    def road_prior(self, T, path):
        """The road's axis at each standing vehicle's place as a prior of its heading (radar_heading_prior), once
        per `road_prior_s`. path: the road model's path as a polyline on the ground (xs, ys), or None. A camera's
        box width cannot tell on which side of the line of sight a vehicle points, and a standing one gets no
        heading from its motion: born along the car's own heading, a car parked on a bend kept it, or drifted to
        the mirror heading, 0.2-0.55 rad off (curve_parked; docs/RADAR.md, section 4)."""
        c = self.cfg
        if not c["road_prior"] or path is None or (self.road_t is not None and T - self.road_t < c["road_prior_s"]):
            return
        self.road_t = T
        px, py = np.asarray(path[0], float), np.asarray(path[1], float)
        if len(px) < 2:
            return
        objs = [o for o in self.objects if (o.label is None or o.label in M.VEHICLES) and o.verdict != "moving"
                and float(o.mu[STAND]) >= 0.5]
        if not objs:
            return
        ax, ay = px[:-1], py[:-1]
        ex, ey = px[1:] - ax, py[1:] - ay
        L2 = np.maximum(ex * ex + ey * ey, 1e-12)
        ox = np.array([float(o.x[PX]) for o in objs])[:, None]
        oy = np.array([float(o.x[PY]) for o in objs])[:, None]
        f = ((ox - ax) * ex + (oy - ay) * ey) / L2
        lo = np.zeros_like(f)
        hi = np.ones_like(f)
        lo[:, 0] = -np.inf                               # the end segments reach on beyond the polyline
        hi[:, -1] = np.inf
        fc = np.clip(f, lo, hi)
        d2 = (ox - (ax + fc * ex)) ** 2 + (oy - (ay + fc * ey)) ** 2
        k = np.argmin(d2, axis=1)
        idx = np.arange(len(objs))
        beyond = (fc[idx, k] - 1.0) * np.sqrt(L2[k])        # past the end of the last segment (m)
        for o, kk, dd, bb in zip(objs, k.tolist(), d2[idx, k].tolist(), beyond.tolist()):
            if dd > c["road_prior_off_m"] ** 2 or (kk == len(ex) - 1 and bb > c["road_prior_beyond_m"]):
                continue
            pose = self.ego.pose_at(o.t)
            if pose is None:
                continue
            los = math.atan2(float(o.x[PY]) - pose[1], float(o.x[PX]) - pose[0])
            _NR.radar_heading_prior(self.store.a, self.hyp.rows(o.h), math.atan2(ey[kk], ex[kk]), los,
                                    c["road_prior_std"], c["road_prior_snap"], c["road_prior_mirror"])

    def _remember(self, o, T):
        """A dropped object that may come back (`in_reid`): its id, where it was and how it moved, how it looked."""
        c = self.cfg
        if not c["in_reid"] or o.state == "tentative":
            return
        try:
            xr = self.hyp.xrow(o.slot)
        except (IndexError, ValueError):
            return
        if not xr or not xr[0] > 0.0:                    # no memory of its appearance
            return
        x, P = o.x, o.P
        self.gallery.append((o.id, T, float(x[PX]), float(x[PY]), float(x[VX]), float(x[VY]),
                             float(max(P[PX, PX], P[PY, PY])), xr, o.t0, o.vote))
        while self.gallery and (len(self.gallery) > 32 or T - self.gallery[0][1] > c["reid_s"]):
            self.gallery.popleft()

    def _reid(self, h, t, xcfg):
        """The dropped object that the new object h is, or None: among those dropped within reid_s whose place by
        now (carried on at their last velocity, unsure by that velocity) the new one's is within reid_gate, the one
        whose remembered appearance the new one's box matches: reid_lr times likelier the same than another, and
        twice likelier than the next candidate. It leaves the gallery."""
        c = self.cfg
        row = int(self.store.tab[h, OT_SLOT])
        xn = self.hyp.xrow(row)
        if not xn or not xn[0] > 0.0:
            return None
        a = self.store.a[row]
        ox, oP = OFF["x"][0], OFF["P"][0]
        px, py = float(a[ox + PX]), float(a[ox + PY])
        pv = float(max(a[oP + PX * N + PX], a[oP + PY * N + PY]))
        dark = self.hyp.xstate()[2]
        best = second = None
        for g in self.gallery:
            gap = t - g[1]
            if gap <= 0.0 or gap > c["reid_s"]:
                continue
            sv = max(2.0, 0.5 * math.hypot(g[4], g[5]))
            d2 = ((px - g[2] - g[4] * gap) ** 2 + (py - g[3] - g[5] * gap) ** 2) / (pv + g[6] + (sv * gap) ** 2)
            if d2 > c["reid_gate"]:
                continue
            lr = _NR.radar_app_lr(xcfg, dark, g[7], xn, gap)
            if best is None or lr > best[0]:
                best, second = (lr, g), best
            elif second is None or lr > second[0]:
                second = (lr, g)
        if best is None or best[0] < c["reid_lr"] or (second is not None and best[0] < 2.0 * second[0]):
            return None
        if any(o.id == best[1][0] for o in self.objects):
            return None
        self.gallery.remove(best[1])
        return best[1]

    def _drop(self, o):
        """The object taken out of the hypotheses (it is no longer in `objects`); what ends with it ends too."""
        self.by_h.pop(o.h, None)
        dead = set(self.hyp.drop(self.store.a, o.h)) - {o.h}
        if dead:
            for q in self.objects:
                if q.h in dead:
                    self.by_h.pop(q.h, None)
            self.objects = [q for q in self.objects if q.h not in dead]
        # the cluster it was of is settled anew: another leaf of a cluster-mate may be its primary now, and what the
        # hypotheses say of its existence has changed; the objects' rows of the table follow
        self.hyp.sync(self.store.tab, [q.h for q in self.objects])
        self._snap = None

    def _life(self, T, pose, scan, slots, tb, rec, lasts):
        """Per object: (p_still, its combined state finite, and for one out of every field (not tentative) where
        it will be at T: the probability that it stands, 1 sigma of its place (larger axis), its range; else
        None x 3)."""
        c = self.cfg
        objs = self.objects
        if not objs:                                     # (radar_life of no objects: an empty list)
            return []
        kin = [o.kin for o in objs]
        return _NR.radar_life(self.store.a, slots, [k[3] for k in kin],
                              [1.0 if o.state != "tentative" and (scan - r[OT_COVER] > c["view_gap_s"] or r[OT_HIDDEN] > last)
                               else 0.0 for o, r, last in zip(objs, tb, lasts)],
                              [T - m[M_T] for m in rec], kin, M._native_par(self.mcfg), pose[:2] if pose else None,
                              self.mcfg["stand_drift"] ** 2)

    def _register(self, T):
        """Sensor registration: what the objects of one sensor agree on in their angle offsets is not theirs but
        the sensor's (a camera's pitch error). The ripe objects (the ones whose own history has pinned their
        offset: passed by, crossing, sized by their box, ranged by another sensor) say what it is; it moves into
        the sensor's bias and out of every object's offset, so each object's sum stays what its own data said,
        and an object born from now on starts with the pitch already known."""
        c = self.cfg
        dt = 0.0 if self.bias_t is None else max(0.0, min(T - self.bias_t, 1.0))
        self.bias_t = T
        for src, b in self.bias.items():
            sen = self.sensors[src][0]
            cc = sen.cfg
            b[1] = min(b[1] + math.radians(cc["bias_walk_deg"]) ** 2 * dt, math.radians(cc["bias_std0_deg"]) ** 2)
            if dt <= 0.0:
                continue
            ripe = [o for o in self.objects if o.state in ("confirmed", "coast") and src in o.last_seen
                    and T - o.last_seen[src] <= c["bias_seen_s"] and o.P[M.DP, M.DP] < c["bias_ripe"] * o.dp0]
            if not ripe:
                continue
            a = 1.0 - math.exp(-dt / cc["bias_tau_s"])
            # what the ripe objects agree on (weights 1 / (P[dp, dp] + the object's own spread); from three on the
            # middle one by weight: one wrong object does not steer it), a share of it out of every object's offset
            m, ws = _NR.radar_register(self.store.a, [o.slot for o in ripe], sen.dp_obj_var,
                                       [r for o in self.objects if src in o.sources for r in self.hyp.rows(o.h)], a)
            b[0] += a * m
            b[1] = max((1.0 - a) * b[1] + a / ws, math.radians(cc["bias_floor_deg"]) ** 2)

    def _consider(self, o, t):
        """1 (2: and its standing hypothesis) if the object's moving hypotheses take a camera's boxes with its height
        and angle offset held (consider parameters), 0 if not. The box height and the bottom edge's angle tell the
        range only through those two, and while the object moves (its range a free path between the boxes) the two
        cannot be told apart from it: an extended Kalman filter that learns them there drifts (a car driving with the
        traffic at 25 m was put 4-10 m too far on boxes that by themselves were right to 2 %; docs/RADAR.md, section
        2). Held from the sighting on at which the hypotheses are weighed (`mu_hold_hits`). A sensor that measures
        the range (a microwave radar, a position) pins them: then they are learnt."""
        c = self.cfg
        if not c["consider_moving"] or o.hits < c["mu_hold_hits"]:
            return 0.0
        for src, ts in o.last_seen.items():
            sen = self.sensors.get(src)
            if sen is not None and sen[0].kind in ("microwave_radar", "position") and t - ts <= c["consider_range_s"]:
                return 0.0
        return 2.0 if c["consider_stand"] else 1.0

    def _judge(self, o, T, p, hits=None):
        """"stationary" | "moving" | "unknown" from how much the standing hypothesis is believed, with hysteresis;
        no verdict before the object has been seen often and long enough for the hypotheses to differ."""
        c = self.cfg
        if hits is None:
            hits = o.hits
        if o.velocity_seen:
            ripe = hits >= c["verdict_hits_velocity"] and T - o.t0 >= c["verdict_s_velocity"]
        else:
            ripe = hits >= c["verdict_hits"] and T - o.t0 >= c["verdict_s"]
        v = o.verdict
        if v == "stationary" and p < c["release_p"]:
            v = "unknown"
        elif v == "moving" and p > 1.0 - c["release_p"]:
            v = "unknown"
        if v == "unknown" and ripe:
            if p >= c["stand_p"]:
                v = "stationary"
            elif p <= c["move_p"] and (o.velocity_seen or self.shown_moving(o) >= c["move_m2"]):
                v = "moving"
        o.verdict = v

    def _merge(self, T):
        """Two objects that are one (a detection that started a second object beside an existing one): the same
        place within both uncertainties for a while. The older one lives on."""
        c = self.cfg
        dead = set()
        obs = self.objects
        n = len(obs)
        if n < 2:
            for o in obs:
                o.merge_t.clear()
            return
        # the pairs worth a look: nearer than the limit can be at any range in reach, with room for what a
        # difference of the two objects' times may have moved one of them; and whether each is at one place and
        # moves alike
        nearj, nearset = {}, set()
        e = self.ego.pose_at(T)
        here, poses = [], {}
        slots, _, rec = self.looked()
        ts = [m[M_T] for m in rec]
        for ot in ts:
            if ot not in poses:
                poses[ot] = self.ego.pose_at(ot)[:2]
            here.append(poses[ot])
        for i, j, same in _NR.radar_merge(self.store.a, slots, ts, [o.kin for o in obs],
                                          M._native_par(self.mcfg), here, e[:2], (c["merge_m"], c["merge_rel"], c["merge_dv"], c["merge_d2"])):
            nearj.setdefault(i, []).append((j, same))
            nearset.add((i, j))
        index = {o.id: k for k, o in enumerate(obs)}
        for i in range(n):
            mt = obs[i].merge_t
            if mt:                                       # a pair that is no longer near stops counting
                for key in list(mt):
                    j = index.get(key)
                    if j is None:                        # the other object is gone: nothing left to count
                        mt.pop(key, None)
                    elif j > i and (i, j) not in nearset and key not in dead and obs[i].id not in dead:
                        mt.pop(key, None)
            for j, same in nearj.get(i, ()):
                a, b = obs[i], obs[j]
                if a.id in dead or b.id in dead:
                    continue
                la, lb = a.label, b.label
                same_kind = la is None or lb is None or (la in M.VEHICLES) == (lb in M.VEHICLES)
                key = b.id
                if not same or not same_kind:
                    a.merge_t.pop(key, None)
                    continue
                t0 = a.merge_t.setdefault(key, T)
                if T - t0 < c["merge_s"]:
                    continue
                rank = lambda o: (o.state != "tentative", o.hits, -o.t0)
                old, new = (a, b) if rank(a) >= rank(b) else (b, a)
                rows = self.hyp.rows(old.h)             # every story of the one that lives on
                am = self.store.a[:, OFF["m"][0]:OFF["m"][1]]
                am[rows, M_HITS] += new.hits
                am[rows, M_L] = np.maximum(am[rows, M_L], new.logodds)
                old.pe = max(old.pe, new.pe)
                old.sources |= new.sources
                for s, t in new.last_seen.items():
                    old.last_seen[s] = max(old.last_seen.get(s, -1e18), t)
                self._range_seen(old)
                self._snap = None
                if old.state == "tentative" and new.state != "tentative":
                    old.state = new.state
                dead.add(new.id)
        if dead:
            self.objects = [o for o in obs if o.id not in dead]
            for o in obs:
                if o.id in dead:
                    self._drop(o)
            for o in self.objects:
                for key in [k for k in o.merge_t if k in dead]:
                    del o.merge_t[key]
