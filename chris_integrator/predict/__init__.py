"""Where every object around the car will be: several futures each, with probabilities.

In: the `radar` message (chris_integrator/radar: every object's place, velocity, acceleration, class, standing or
moving, uncertainty) and what is known of the road (lane centre line, neighbouring lanes, edges, junction arms).
Out, per object: a set of futures over 8 s, each a manoeuvre with a probability and a trajectory whose spread
grows honestly; the probability of each kind of lane it may be in; the manoeuvre probabilities; and the car's own
futures. No learned network: models of how road users drive, a recursive Bayesian filter over them, and a joint
treatment of road users whose futures cannot all happen.

How it is built (each step is a function of `native/src/predict.c`, the only implementation of the arithmetic):

1. **The road's frame** (`pp_base`, `lane_step`). The ego lane's centre line as a polyline, extended behind the
   car along the arc it drove and ahead with the end's curvature dying away; without a line, the arc of the car's
   speed and yaw rate. Every object is projected onto it: arc length s, offset d to the left (a Frenet frame). An
   object's position uncertainty is long and thin (range poor, bearing good); on a bend its long axis crosses
   lanes, so three points along that axis are projected, not one.
2. **Lane assignment** (`lane_step`). The lanes beside the car's own are hypotheses with probabilities (bands of
   offset: ego, left, right, oncoming, parking strip; made by `predictor.road_arrays`). What a lane kind says
   about an object: how much of the object's offset distribution lies in its bands, and whether the object moves
   as that kind requires (with the road, against it, across it, not at all). A discrete Bayes filter over
   (ego, left, right, onc, cross, park, off) per object, with hysteresis on the published best.
3. **What is observed of an object** (`feat_step`): a Kalman filter on its offset and offset rate in the road's
   frame and one on its speed and acceleration (the tracker's frames share errors: they count as one observation
   per `evid_tau_s`); its yaw rate against the lane's; its distance into the carriageway from the kerb.
4. **The situation** (`context_step`): the one ahead of it in its lane (and the gap, the car-following
   acceleration that gap asks for), the junction ahead with the arms on its left and right, the place where it
   would stop (a junction it has no priority at, the tail of a queue, a marked crossing with someone at it).
5. **Manoeuvres and their recognition** (`intent_step`). Fifteen slots: keep, follow, lane change left / right,
   turn left / right, stop, pull out, stand, U-turn, drift (oncoming), cross, free, start to cross, swerve; which
   are possible depends on the kind of road user and where it is. Each has a prior weight from the situation and
   a likelihood of what is observed (offset and rate against the lane, acceleration against what the manoeuvre
   needs, yaw rate, standing or moving, indicators and brake lights where the message has them), every
   observation with its own uncertainty. The filter: a manoeuvre lasts (its own exit rate); when it ends the next
   is drawn from what is possible here; the frame's evidence is applied tempered. A dynamic Bayesian network over
   manoeuvres, not a per-frame classifier. Names are published per situation: `cut_in`, `cut_out`, `onc_keep`,
   `onc_drift`, `onc_overtake`, `parked`, `stopped`, `wait`, `walk`, `ride`, `crossing`, ...
6. **Futures** (`mode_build`, `roll_out`, `roll_free`). Each manoeuvre above `mode_p_min` (up to `modes_max`) is a
   trajectory generator: a path (the lane at an offset; a corner onto a side road; a side road over the junction;
   a quarter or half circle; a straight line across the road) with a lateral profile (minimum jerk to the target
   offset) and a speed profile integrated step by step (the observed acceleration dying away; not faster into a
   bend than it can be driven; behind the one ahead by the intelligent-driver model; down to a stop where it
   stops; from standstill after an onset time). The covariance is propagated with the mean: (s, v, a) along the
   path with an acceleration that wanders, (d, d rate) across it under the lateral control of a path follower
   (bounded) or none (growing), turned into the frame at each sample, plus what is not known of the execution
   (when a lane change or a start really begins, the line of a turn, where a stop ends). The free future is the
   tracker's own motion carried on; it always keeps a prior and wins when nothing structured explains the object.
7. **Interaction** (`pair_table`, `joint_solve`, `interact`). Futures are generated leaders first, so a follower's
   car following runs against its leader's future. Then, for every pair of road users (the car among them) and
   every pair of their futures: the probability that they meet (Gaussian places against the bodies' rectangle,
   lengthened by the headway a follower keeps), who has priority (the manoeuvres' ranks, the one in front, the
   right-hand rule), and a factor psi = (1 - gamma x conflict) x (a penalty when the one with priority is the
   one who gives way). Road users that touch form groups; a group's joint distribution prod prior x prod psi is
   summed exactly by a depth-first search with pruning (bounded group size and node count), beyond that by
   belief propagation. Marginals: `p` with the car's own futures in the group, `p0` without the car (nobody is
   taken to give way to it: what a threat measure must use).
8. **The car's own futures** (`ego_modes`): along the road's path at the present speed and acceleration, the same
   behind the one ahead, and by steady yaw rate and acceleration.
9. **Self-check** (`selfcheck.py`, `predict_score`): every half second the futures are kept and later scored
   against what the tracker says the object did: minADE / minFDE, the most probable future's error, negative
   log-likelihood, the calibration of the probabilities and of the spreads; in `health()`. Offline over
   simulator scenarios (with the truth) and captures: `tools/predict_eval.py`.

The context dict (`Predictor.update(radar_msg, context)`; `context.ContextBuilder` makes it from the stack):

    centre          [[x, y], ...] the ego lane's centre line, car frame of the radar message (x forward, y LEFT)
    lane_width_m    3.5
    n_left, n_right lanes of the car's direction on that side: an int, P(at least one), or [P(0), P(1), P(2)]
    oncoming        probability that oncoming lanes follow on the far side; n_oncoming (1)
    park_left, park_right    probability of a strip where vehicles stand beside the outer lane
    kerb_left_m, kerb_right_m   measured road edges, offsets from the centre line (positive numbers)
    arms            [{"pts": [[x, y], ...] outwards from the junction, "side": "left" | "right", "s_m": the
                      junction's distance along the centre line (else: projected), "priority": 1 the arm has
                      priority over this road / 0 equal (right-hand rule) / -1 it gives way, "entry", "exit",
                      "control": None | "give_way" | "signals" | "stop"}]
    crossings       [{"s_m": distance along the centre line, "half_m": 2.0}] marked pedestrian crossings
    quality         0..1: how far the road is known (the free futures' prior falls with it)
    turn_signal     the car's own: "left" | "right" | None
    source          a name for `fut_ctx.path`

Cost and knobs: docs/TOPICS.md (`radar_fut`). The plug-in: `chris_integrator/plugins/predict.py`.
"""
from .context import ContextBuilder  # noqa: F401
from .predictor import DEFAULT_CFG, NAMES, LANES, SLOTS, Prediction, Predictor  # noqa: F401
