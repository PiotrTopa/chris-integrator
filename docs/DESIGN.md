# chris-integrator: design

The component every input of the car passes through before any logic reads it, and the component consumers
talk to. Its shape is an **audio mixer**: input channels, a bus, plug-in transformers that take named streams
from the bus and put named streams back, output channels. Sections 1-8 and 10-13 describe the estimator box
(one ego estimate with a full state, the object layer, sources that plug in without touching the core);
section 9 is the cut of the whole stack; **section 14 is the mixer**: channels, the bus, the plug-in contract,
and the boxes of the stack; section 15 the map box; section 16 object classes. Sections 3a (the receiver),
4a (gaps of all input) and 15a (the loop estimator - map box) are lane L47's. Sections marked "(as built)"
were corrected after the code and the replays.

```
 INPUT CHANNELS                      BUS (named streams; envelopes: time of validity,                OUTPUT CHANNELS
 (any process, rate, latency)        arrival time, producer, one serialised line)
                                                                                                 
 chris-roadsense, one per camera ─┐    road calib camera env lights_raw signs_raw governor       ┌─> HTTP /v2 + WebSocket /ws
   road, calib, camera, env,      │    ego vehicle gnss objects                                  │     (the consolidated API)
   lights, signs, governor        ├──>   │        ▲                                          ────┼─> integrator.sock (JSON lines
 the car (feed.sock, POST):       │      ▼        │                                              │     for other processes)
   motion, gnss, controls, body,  │   PLUG-IN TRANSFORMERS, in an order derived from             └─> recorder
   objects (radars, cameras)     ─┘   what each declares it reads and writes:
                                        estimator   motion, fixes, pose, map match, scans -> ego_state, tracks, lead
                                        mapper      ego_state -> map                  (chris-mapper; feeds back)
                                        lanes       road, ego_state, map -> lanes
                                        egolane, roadgeom   lanes, map -> (wiring for view / fused)
                                        ego_light   lights_raw, road, lanes, map -> lights
                                        sign_context  signs_raw, lead, lights -> signs
                                        view        all of the above -> view      (the display channel)
                                        rules       signs, env, road, ego_state -> speed_limit, rules
                                        fused       map + camera decisions -> fused
```

## 0. Requirements this design answers

1. **One box, on top**: its own component (`chris-integrator/`, imports nothing from roadsense), an input contract
   and an output contract, a daemon and a library; since the owner's ruling during the lane it is the top
   component of the car's perception stack and roadsense is one of its sources (section 9).
2. **One coherent ego estimate, full state**: position, heading, speed, acceleration, yaw rate, grade, sensor
   biases and scale factors, per-sensor time delays, camera mounting angles (sections 2-4).
3. **Object layer inside the box**: one multi-target track list, per-sensor measurement models and
   association, life cycle, existence probability, cross-sensor consistency, the lead picked by the path
   (section 6).
4. **Top-tier machinery**, chosen against the literature (section 11).
5. **Extensibility proven**: front radar, second radar, second camera added as source files only, driven by a
   simulator with truth and by injection into real recordings (sections 7, 10).
6. **Metadata on everything that leaves** (section 8). 7. **Consumers stay connected** (section 9).

"Not needed on today's data" is not a reason used anywhere below. Where the car has no sensor, the part is
built and proven on the simulator and on synthetic sensors injected into the real recordings.

## 1. What the data says

From the audit and L37 (recordings of 2026-11-19), plus two things settled in this lane before the design
(roadsense [`tools/integrator_eval.py`](https://github.com/PiotrTopa/chris-roadsense/blob/main/tools/integrator_eval.py)` gnss`, `calib`; numbers in the report):

* **Vehicle speed (CAN 0x0B4)**: 10 Hz, robust sigma 0.012-0.020 m/s against its own neighbours; wheel speeds
  at 0.979-0.981 of it. The feed falls silent for about 7 s 19-25 times a drive and its first samples after a
  silence are stale (up to 0.28 s, up to 7 m/s wrong).
* **Gyro** zeroed at standstill; the rear wheel difference and the camera's yaw rate both read 0.91-0.96 of it
  in turns: three sources, one scale question, which an estimator with scale states can hold.
* **Accelerometers**: longitudinal = dv/dt + slow offset (grade, calibration), sigma 0.2-0.25 m/s2.
* **Road model pose**: speed sigma 0.25 + 0.035 v, heavy tails, error correlated over seconds, valid 0.10-0.15 s
  before the frame stamp (the 200 ms image pair). Arrives 85-130 ms after the frame.
* **The direction of the pose is a precise mount-angle measurement.** On the healthy drives (calibration pitch
  0, yaw 0 in use) `pose.vy / pose.vx` = -0.062 and `pose.vz / pose.vx` = -0.056 with a robust spread of
  0.003-0.005 per frame at speed on straights; the lane lines' direction says the same (-0.065). The camera
  sat 3.5 degrees in yaw and 3.2 degrees in pitch off the calibration in use, and the model worked. On the two
  incident drives the calibration in use was pitch 0.0561, yaw -0.0621 (those very offsets), the direction
  reads 0.00 / +0.01, and the model's speed is 0.2-0.4 of the true one. So in the incident the *direction* cue
  says "aligned" while the *scale* is lost: the fault is visible in the speed against the vehicle, not in the
  direction. (What exactly broke on the car is not established by the recordings; the report states it.)
* **GNSS (head unit), the bias question settled.** Positions: no repeats while moving (the 37-50 % repeated
  positions are standstill), distance over chords >= 100 m is 1.02-1.05 times the CAN odometer on the three
  usable drives, displacement speed between fixes is 0.97-1.03 of the CAN speed at a lag of 1.75-2.5 s with
  sigma 0.9-1.6 m/s. The **speed field** is a different, more delayed and smoothed output: its best lag per
  60 s window ranges 1.5-8.75 s within one drive and it reads 0.82-0.97 of the CAN speed in the median with
  windows near 0. L35 (4-10 % low at the best lag per
  drive) and L37 (half, at a fixed 3 s lag) both looked at this field. Conclusion: the receiver is not biased
  in scale; its positions are usable with their real noise and a lag of about 2 s, its speed field is not a
  measurement of the current speed and is kept as a reported witness only. 113201 is a poor-reception drive
  (4 chords, ratio 0.7-0.8): the filter must survive it by weighting, not by trusting.

## 2. The ego estimator: state

One extended Kalman filter on one time axis. The kinematic state is planar with altitude (a car without a
6-axis IMU does not observe roll, and observes pitch only as road grade through the accelerometer):

| core state | meaning | process |
|---|---|---|
| `pe`, `pn` | position, local east / north (m) about an origin set by the first absolute fix | integrates speed along heading |
| `psi` | heading, rad counter-clockwise from east; innovations wrapped (the filter is an error-state EKF on SE(2): the heading error is the small quantity) | integrates yaw rate |
| `v` | speed along the heading, signed (m/s) | integrates `a` |
| `a` | longitudinal acceleration (m/s2) | random walk, jerk PSD |
| `w` | yaw rate, positive left (rad/s) | integrates `al` |
| `al` | yaw acceleration (rad/s2) | random walk |
| `z` | altitude (m) | integrates `v sin(th)` |
| `th` | road grade (rad) | random walk |
| `sv` | scale of the speed state against the ground (as built): `v` is the speed as the car's own speed signal scales it, known to cm/s, which is what consumers and the object layer want; distance over ground is `v / (1 + sv)` | random walk, tiny |

Everything else in the state belongs to a source and is **declared by the source** (`AuxState`: name, initial
value and sigma, process: random walk with a PSD or first-order Gauss-Markov with a time constant). The core
only appends them to the state vector and propagates them. This stack's sources declare:

| source | calibration states (each estimated, each published with its std) |
|---|---|
| vehicle | gyro bias `b_g` and scale `s_g`; wheel-difference yaw scale `s_wy`; accelerometer offsets `b_ax` (longitudinal, with grade), `b_ay` (lateral: bank and bias) |
| vision | time delay `tau` (the image pair, prior 0.125 s); speed scale `s`; yaw-rate scale `s_w`; two Gauss-Markov errors of the speed, `m` (0.45 m/s, 3.5 s) and `mf` (0.38 m/s, 0.35 s), fitted to the autocorrelation of the car's residuals (0.80 at one frame, 0.42 at 1 s, 0.21 at 3 s: hardly any of the error is white); mount yaw and pitch against the calibration in use (`dyaw`, `dpitch`) |
| gnss | time delay `tau` (prior 2 s, sigma 0.6 s, measured: section 3a); the slowly varying position error, east / north, Gauss-Markov over 20 s, in units of the receiver's present error level (`ue`, `un`: the level may change from fix to fix) |
| map match | the car's offset from the matched road's centre line (`off`, Gauss-Markov, 120 s): a thousand matches must not pretend to know the position across the road to centimetres; which of the candidate roads the car is on (`amb`, unit variance, in units of the spread of their lines; never corrected by the map's own rows: section 15a) |
| radar (each) | mount yaw misalignment `dyaw`; time delay `tau` |

Observability, stated rather than assumed: `sv` is observable only against an absolute source (GNSS
positions over distance, a radar's stationary targets); with neither it keeps its prior and says so through
its std. `b_ax` and `th` are separable only with an altitude source: without one their sum is observable and
the published grade carries the bias prior as uncertainty. The absolute yaw scale is split between `s_g`,
`s_wy` and the camera by their priors until a heading reference (GNSS track over turns, map bearing) decides.
Time delays are observable only under excitation (acceleration for a speed delay, speed for a position
delay); their process noise is tiny and the std says how far they are known.

**What the build taught about calibration states (as built).** (1) A scale state's Jacobian is the quantity
it multiplies; near zero that quantity is its own estimation noise and teaches the scale nonsense (the
errors-in-variables bias of an EKF: the gyro scale walked to -0.4 on a straight road in simulation). The
regressor is therefore shrunk towards zero by its own uncertainty (zero inside 3 sigma or 0.02 rad/s, the
value itself far outside). A fixed floor (0.05 rad/s, the first build) also removed the scale's *uncertainty*
from every gentle bend: an unlearned gyro scale of 8 % at 0.04 rad/s is 3 degrees of heading in a long
motorway bend that the covariance did not contain. (1a) Nothing in a yaw-rate sensor says what its scale is:
the gyro's scale is a consider state of the gyro's own row, corrected only through heading references; the
other yaw sensors' scales are learned relative to it. Without that gauge all yaw scales walked together
(to -0.25, stated +- 0.03) whenever no absolute source was allowed to teach them. (1b) A standstill says what
the motion is and nothing about a calibration: standstill rows do not correct calibration states, or the
correlations built at speed are cashed in at v = 0, where they mean nothing. (1c) A coloured-noise state
scaled by a speed-dependent sigma is a second scale factor; the filter split the camera's scale between the
two and got it wrong by 8 %: coloured-noise states have a constant sigma. (2) The same holds for the delay of the camera's yaw rate, whose Jacobian is the poorly known yaw
acceleration: the camera's delay is learned from the speed rows only. (3) States a row cannot tell apart are
**consider states** for it (Schmidt-Kalman: their uncertainty counts, they are not corrected): the camera's
rows never correct another source's calibration; the camera's own scale is frozen while no other source holds
the speed; the receiver's and the map's rows do not teach the speed and yaw scales unless the receiver is
declared good (`gnss_absolute`; the reference car's is not).

**Unobservable directions have to be in the covariance.** A position placed on a fix is that fix carried
forward by speed x delay along the heading: at constant speed the receiver's delay and the along-track
position are one quantity. The anchor therefore gives the position the delay's uncertainty along the track
and the heading's across it, correlated with those states; the filter learns the delay when the speed
changes and the stated position uncertainty is honest until it does. (The first build anchored with an
uncorrelated 7 m and became overconfident by 15-35 m along the track.) Since L47 the anchor also carries the
receiver's slowly varying error, fully anti-correlated with the position placed on it (the fix is the sum of
the two), and the delay is measured by a row of its own (section 3a).

**Time delays as states.** A measurement stamped `t` by a source with delay `tau` is `h(x(t - tau))`. It is
inserted into the buffer at `t - tau_hat` (the current estimate) and its Jacobian carries
`dh/dtau = -H f(x)` (the state's rate), so the filter corrects `tau` whenever the state moves.

**Camera mounting (revised in L41).** The mount model follows the hardware: on the reference car the mount has
one degree of freedom, pitch (it is re-aimed by hand); yaw and roll are fixed. So `dyaw` is a constant to be
found once, and `dpitch` is a constant *between steps*.

* The rows are about the direction of the pose: `pose.vy`, `pose.vz` against the pose's *own* `vx` (plus the
  yaw-rate lever arm). The regressor is the measured `vx`, not the speed state: with the state there the rows
  were a speed measurement in disguise, and a camera without a reference "learned" its own scale from them.
  They are used only at speed (judged by the integrator's speed, not the camera's) and never while the camera
  is faulted.
* Two row classes, `mount_yaw` and `mount_pitch`, judged separately. A two-sided CUSUM on the pitch row's
  residual finds a step in a few frames; the state is then put at the step's estimate with an uncertainty
  that says so, converges again within seconds, and the step is reported with its size ("camera pitch
  changed by +1.8 deg"). The yaw row has no such mechanism: a persistent yaw residual is reported as a trend
  of the camera, not followed as a movement of the mount.
* A calibration change does not move the camera: the residual against the new calibration is the old
  residual plus what the calibration changed by.
* The pose moving while another source knows the car stands is somebody handling the camera: the pitch is
  free again (prior uncertainty) until the next drive.
* Far outside its range (measured: a camera tilted by about 16 degrees) the road model returns a collapsed
  speed, an exploded stated std, a phantom near lead, and a pose direction that reads "aligned". That direction
  is not a mount measurement: the camera is faulted, the mount is `unknown`, and the finding says what this
  looks like. The size of such a change cannot be estimated from the model's outputs.
* `sources/signrows.py`: the image rows of sign detections at equal box size, against a reference learned
  while the camera was believed. Rows / focal length is a pitch change that does not pass through the road
  model; coarse (about 2 degrees), needs ten sign tracks per size class, keeps working when the model is lost.

Published as `calibration.vision` (angles, std, `state`: `converging`, `ok`, `offset`, `pitch_changed`, `moved`,
`unknown`; `pitch_steps`, `last_pitch_step`, `handled`) and `calibration.signrows`.

**Standstill** (vehicle: speed and every wheel exactly 0) is a set of exact pseudo-measurements (`v`, `a`,
`w`, `al` = 0): the gyro then measures its bias, the accelerometer its offset plus grade. Since the ego-motion
lane the feed's flag is only a claim (its signals read zero below a floor, 0.21 m/s on the reference car, while
the car rolls and turns): the exact rows wait for the inertial sensors' quiet or the camera to confirm it
([EGO_MOTION.md](EGO_MOTION.md) section 1, which also has the time offsets estimated outside the filter).

## 3. The ego estimator: measurements

Every measurement is a scalar row: value, sigma, time of validity, and a function `h(x) -> (prediction,
sparse Jacobian)` supplied by the source's model. Rows carry a noise class:

* `exact`: applied as is (vehicle speed, standstill).
* `robust`: heavy-tailed source. The update is a Student-t / Huber re-weighting: the row's variance is
  inflated by `max(1, (|nu| / (k sqrt(S)))^2)`-type factor, so a wild sample moves the state by a bounded
  amount instead of being either believed or thrown away; beyond a hard gate (chi-square 0.9999) it is not
  applied and counted. A robust row is only gated while an exact row anchors the same quantity (L37's lesson:
  a lone camera must be followed, not locked out by its own filter). As built: the Student-t weight (4 degrees
  of freedom) scales the whole innovation variance, so the state moves by weight x the Kalman step however
  small the row's own noise is.
* `monitor`: not fused (a witness); it feeds the monitors only.

| source | rows (noise) |
|---|---|
| vehicle | speed `v` (0.03 + 0.002 v, exact); gyro `(1+s_g) w + b_g` (0.004 + 0.03 abs(w)); rear wheel difference `(1+s_wy) w` (0.02 + 0.05 abs(w), robust); accelerometer long `a + g sin(th) + b_ax` (0.25); lateral `v w + b_ay` (0.5); standstill rows; mean wheel speed (monitor); warm-up after a feed silence (L37) |
| vision | speed `(1+s) v + m` at `t - tau` (max of characterised and stated sigma for the white part, robust); yaw rate `w` (0.007 + 0.25 abs(w), robust); `vy`, `vz` (mount rows, robust, at speed) |
| gnss | position east, north at `t - tau`: 1.5 m white about the slowly varying error states x the receiver's level (robust; a step the receiver made alone is subtracted while it is held back); the delay as measured from the receiver's own motion (robust); course while moving straight and the receiver steady (robust, 15 degrees); altitude (robust); speed field (monitor). Section 3a |
| map match | cross-track distance to the share-weighted line of the matched road and of the candidates parallel to it = 0 (1 m about the lane offset state and the hypothesis state, robust); heading = road bearing while moving straight (8 degrees and the road's change of direction around the match, robust; not while candidates of another direction hold 15 % or the heading is known worse than 20 degrees). Section 15a |
| radar | ego speed and yaw rate from the range rates of stationary detections (robust fit per scan; section 6) |

**Adaptive noise.** A source whose stated accuracy is not to be believed (the head unit's `hacc`) gets a scale
on its variance from its own windowed normalised innovations (innovation-based adaptive estimation), bounded
(1-8) and published.

**Correlated sources.** Two measurements derived from the same physical sensor (mean wheel speed and the CAN
speed) are never both fused: one is fused, the other monitors (family rule, L37). Errors correlated in time
are modelled by Gauss-Markov states (camera speed). Where the object layer fuses two tracks of unknown
cross-correlation (track-to-track across sensors that share the camera's ego motion) it uses covariance
intersection (section 6).

## 3a. The receiver: what a fix is, and what its own motion says (L47)

A fix is `position(t - delay) + slowly varying error + white noise`. On the reference car (four drives of one
day; roadsense [`tools/integrator_eval.py`](https://github.com/PiotrTopa/chris-roadsense/blob/main/tools/integrator_eval.py)` gnss`, roadsense [`tools/mapper_eval.py`](https://github.com/PiotrTopa/chris-roadsense/blob/main/tools/mapper_eval.py)` real`, lane report L47):

* From one fix to the next the receiver moves as the odometer says to 0.75-1.55 m (robust sigma): the white
  part is small. Raw fixes lie 7-10 m (median) beside the road they are on, the same way for 5-20 s: the
  error is slowly varying. Fused as white noise of `hacc x 1..8` (before L47), twenty fixes average it to a
  quarter: the stated position uncertainty was 4-5 m where held-out fixes lay 10-19 m away (normalised 3-19
  where 2 is expected). The slowly varying error is now what `hacc` is taken to describe: two Gauss-Markov
  states (20 s) in units of `sigma_b = level x hacc x scale`, entering the rows multiplied by `sigma_b`.
  `level` is measured, raw against raw, where there is a map: the distance of raw fixes from the line of a
  confidently matched single road on straights, robustly, over the last 60 such fixes (prior 2 for a
  receiver nobody has measured). `scale` (1-2.5) is the realised second moment of those states: where the
  filter finds the error larger than its level says, the level rises. On the car `sigma_b` comes out at
  5-9 m (median per drive) with 5-10 satellites and at 14-20 m with 3-5.
* The receiver's own motion against the car's, raw against raw: the displacement from one fix to the next
  against the distance the car drove (its speed sensors) and the direction of the receiver's own earlier
  track turned by the yaw the car made since (its yaw sensors). The estimator's heading and position are not
  asked: they may be the wrong party. Three things come out of it.
  - *The delay.* The lag at which the receiver's step lengths fit the distances driven, over the last
    minute of steps, is a measurement of the delay state that involves neither position nor map (a row,
    with a sigma from how sharply the steps determine it; nothing while the speed hardly varied). On the
    car: 1.75-2.5 s on three drives, 7.25 s on the fourth. The delay is a property of how the receiver is
    tracking at the time, not a constant of the device: with 3-5 satellites it reports a smoothed track.
    When the satellite count falls to 5 or fewer the delay state's uncertainty is opened to 1.5 s until
    the measurement narrows it.
  - *Unsteady.* Two steps in three off by more than 2.5 m/s (healthy: 0.5-1 m/s; with few satellites
    4-7 m/s for 10-20 s at a time, e.g. 13 s of 4-7 m/s after the wheels stood still): the receiver is
    not moving as the car does. What it moves beyond the car's own motion is held back from its fixes
    (decaying over 45 s) and counted into `sigma_b`; such fixes do not teach the delay.
  - *A jump.* One step beyond 8 m + half the car's own step: the receiver left the truth, or returned to it.
    The step is held back the same way, and the position's own covariance takes the step in (if the
    receiver returned to the truth, the position that had followed it is the one that is wrong). Held-back
    steps beyond 250 m in all (mirrored coordinates, a position somewhere else): the receiver is not
    used until its fixes come back to the estimate, or until dead reckoning has run out and the receiver
    has moved as the car does ten fixes running.
* The heading at an anchoring is the direction of the receiver's own track (two pieces in a row that agree,
  driven forwards) carried to now by the yaw the car made since; the course field only when the yaw sensors
  were alive since the time it describes and the car has not turned since. Never the chord between two
  fixes (it points backwards while the car reverses: a three-point turn anchored the heading 180 degrees
  wrong). A heading more than 57 degrees off the direction a steady receiver moves in, three times running
  on a straight, is placed anew: position rows do not mend that through a linearisation.
* While the car stands and the receiver has come to rest, a position that is not placed is put on the fix
  with the heading unknown (said so: `heading.valid` false); the heading follows when the car moves.

What is still wrong with this model: the level is measured across the road only, and on the two drives with
good reception the stated uncertainty is right in the median and still 1.1-1.4 times too small in sigma in the
mean; the measured delay is noisy (0.9-4.4 s on one drive); with a trailing receiver at constant speed nothing
shows how far it trails.

## 4. Time

* **Clocks.** Every input is stamped on the phone's CLOCK_MONOTONIC, which restarts at boot. A source whose
  stamp steps back by more than 2 s starts a new **epoch**: the buffer, the monitors' windows and the tracks
  are dropped, the calibration states (biases, scales, delays, mount) and their covariance are kept, faults
  are carried, the position keeps its value with its uncertainty inflated by the unknown time parked.
  Wall-clock time is never used for ordering.
* **Buffer and out-of-sequence measurements.** Measurements are kept in a time-ordered buffer (horizon 10 s:
  the receiver's fixes are seconds old). A late one is inserted at its time of validity and the filter is
  re-run from there with every later measurement re-applied: the exact solution of the out-of-sequence
  problem (the same result as if it had arrived in order), which the approximate one-step algorithms
  (Bar-Shalom's A1/B1) only approach. Cost: one re-run of at most the lag times the input rate, per late
  measurement; measured in the report (and in [EGO_MOTION.md](EGO_MOTION.md) section 3: 5-7 % of the estimator;
  the rows' measurement functions are native, so a re-run does not enter Python for them). Older than the
  horizon: dropped and counted.
* **Output time.** The state for a frame is the **fixed-lag smoothed** state at the frame's time of validity
  given everything that has arrived (a Rauch-Tung-Striebel pass over the entries newer than the frame, at
  most 0.6 s): by the time a road message arrives, vehicle samples on both sides of its frame time exist.
  Nothing is delayed for it and nothing is predicted to "now": measurement topics refer to the frame.
* **Linearisation.** The process model is mildly non-linear (heading into position); the re-run re-linearises
  at the current estimate on every pass (an iterated fixed-lag filter in effect).

## 4a. Gaps of all input (L47)

On the reference car the whole input stops for 5-13 s at a time (camera frames, vehicle samples and fixes
together: 20, 9, 5 and 0 times on the four drives), and the car turns inside some of those gaps. Without any
measurement of speed and yaw rate for more than 1.5 s the heading is not known afterwards. What the filter
did with that (before L47): its white-jerk model stated hundreds of metres after 6 s, the first row after the
gap was applied through a linearisation about the old heading, and a road's bearing (a line) locked the
heading the other way round with 8 degrees stated. Now:

* No position or heading row whose time lies in such a gap is used.
* When the car's motion is measured again and it moves, position and heading are placed anew: the position is
  stated to `0.5 T^2 + 0.5 v T` (at least 40 m; what a car does in T s, and it may have turned), put on the
  receiver's next fix give or take what the car drives in the fix's age, and the heading comes from the
  receiver's own track once two pieces of it were driven after the gap (section 3a). Until then the heading
  is published invalid and the position wide; neither is guessed.
* `Integrator.yaw_between(t0, t1)` and `dist_between(t0, t1)` give the heading change and the distance the
  car's own sensors measured between two times (None across a gap): what the receiver's motion is compared
  with, whatever the heading and position themselves are believed to be.

## 5. Fault detection and isolation

Three layers, because each catches what the others cannot:

1. **Raw pair monitors** (L37, kept): two sources' raw measurements of one quantity at a common time,
   normalised by their *characterised* sigmas, over 5 s of evidence. Never against the fused state (the
   dominant source would test itself). A disagreeing pair faults the lower-ranked source unless independent
   sources side with it against the higher one: with three or more independent sources of a quantity (vehicle,
   camera, a radar's ego speed) the isolation is a **vote**, and a wrong *vehicle* speed is isolated too.
   Groups that share sensors form a family and cast one vote.
2. **Self monitors** (L37, kept): a source's stated uncertainty against its characterised one.
3. **Innovation consistency (NIS)** of every fused row class per source: the windowed mean of
   `nu^2 / S` against its chi-square bounds. It adapts noise where the source allows it (GNSS), marks a
   source `suspect`, and for sources with no raw pair (position, heading, mount angles) it is the fault
   detector: a persistent excess faults the source if a higher-ranked one anchors the quantity.

As built: innovation monitors use the scaled median over 8 s. A row class whose innovations are far beyond the
stated uncertainty (`pos`, `heading`, `map_cross`, `map_heading`, `mount`, `alt`) is **set aside**: its rows
are still evaluated, so it returns by itself, but they are not applied. Absolute positions set aside for 8 s
mean the position is lost: the local frame is anchored again on the next usable fix (`reanchors`). Blame by
rank waits while a vote is forming (an independent third source agrees with the lower-ranked one and drifts
away from the higher), and a disagreement held only by its recovery hysteresis blames nobody anew.

A faulted source leaves the estimate at once (the buffer is re-run without it), its calibration states freeze,
its detections leave the object layer, and everything made from it goes out `valid: false` with the reason.
Recovery needs evidence of agreement (hysteresis as in L37). Faults survive epochs and restarts (carried).

## 5a. Fault detection as a general capability (L41)

Section 5 is isolation between sources. Below it sits detection that knows nothing about any particular
source or incident (`core/fdi.py`), and above it the campaign that scores all of it (`sim/faults.py`,
[`tools/fault_campaign.py`](https://github.com/PiotrTopa/chris-roadsense/blob/main/tools/fault_campaign.py) in chris-roadsense, `tests/test_faults.py`).

**Taxonomy** (the campaign injects each into every source and quantity it applies to): bias, scale error, slow
drift, stuck value, repeated samples, drop-out, noise burst, outliers, delay and delay jump of the content,
time-stamp error, clock epoch change, stale-after-silence, sign / unit / frame error, mount change of a camera
or a radar, ghosts and mis-association in the object layer, a source lying about its uncertainty, two sources
failing together, slowly developing faults.

**Detectors.**

| family | sees | needs |
|---|---|---|
| signal integrity of one raw stream (`Stream`) | stuck, repeated samples, noise burst (robust sigma of the second difference against the characterised one), outliers, not-a-number | nothing: *self-evident*, the culprit is known whatever its rank |
| timing integrity of one message stream (`Clock`) | a step of its stamps against the other streams' stamps at arrival (stamp offset, latency jump); staleness and epochs as before | the other streams as a clock |
| pair monitors (`monitors.Pair`) | bias, scale, sign, unit between two raw sources: a share-of-bad-samples test for gross faults, a CUSUM for what never puts one sample beyond the gate, a CUSUM in units of the pair's own learned spread for slow noisy sources; the kind of the disagreement is named from the samples | a second source |
| innovation consistency (`NisMonitor`, `Trend`) | a source's stated uncertainty against what its innovations show; a signed innovation that keeps its sign (bias, drift, a calibration that moved) | the filter |
| source-specific (`sources/`) | camera pitch step, handled camera, road model out of range, sign rows | that source's physics |

**Responses**, by what is known: a self-evident fault takes the quantity out at once; a noise burst below the
fault level stays in with the weight its measured noise deserves (adaptive measurement noise); a
disagreement of two is decided by vote when an independent third exists, and while the third has not yet
spoken the quantity is published `mode: disputed` with a std that covers both readings; by rank only when no
third exists. Faults are per (source, quantity) unless the quantity is the source's main signal: a gyro
fault sets the gyro's rows aside and keeps the vehicle's speed.

**Independent opinions on today's car.** Speed: the vehicle's speed signal (wheels: same family), the camera,
and the receiver's *track* (`gnss_track`: path length over three fixes; slow, noisy, no scale of its own).
Yaw rate: gyro, rear wheel difference, camera. Position, heading: the receiver against dead reckoning and the
map's geometry.

**What cannot be detected with the present sensors** (the campaign shows each as such):

| fault | why | what would make it detectable |
|---|---|---|
| which of two is wrong, when only two measure the quantity | no third opinion | a third source (speed: radar ego-speed, a second receiver) |
| two sources wrong the same way (vehicle and camera speed both x 0.7) | they outvote the truth | a second absolute source of the speed (radar on stationary objects) |
| a persistent offset or slow drift of the receiver's positions | the only absolute source; the map gives the cross-track part only | a second receiver, or map matching of landmarks |
| mirrored / swapped receiver coordinates | distances stay right; only turns disagree | as above |
| the camera's lead range or a phantom lead | one sensor reports the lead | a radar (the object layer then cross-checks) |
| a gyro sign error on a straight road | nothing turns | time: it shows at the first bend |
| a delay of the whole vehicle feed with fresh stamps | camera and vehicle disagree in time, not which one is late | honest time stamps from the feeder (`age_ms`) |
| an accelerometer bias | it is the same thing as road grade | an altitude source |

## 5b. Detector modes: gate or report (L43)

A detector that can take something out of the estimate has a mode, per detector, in the configuration
(`integrator.json`: `{"detectors": {"speed.cusum": "gate"}}`; defaults in `core/integrator.py` `DETECTORS`):

* `gate`: its verdict takes effect. The source's rows (or one quantity's rows, or a row class) leave the
  estimate; what depends on that source alone is then published as unknown (a camera's lead, a speed with no
  other source).
* `report`: it runs all the same and publishes its verdict with the evidence: an entry of kind `reported` in
  `ego_state.diagnostics` (`detector`, `mode: "report"`, the text names who would have been set aside, by which
  rule, and the numbers), the detector's name under `reported` in the source's health entry. Nothing is set
  aside, nothing goes invalid, a verdict held this way has no vote in another pair's isolation, and nobody
  waits for it.

`ego_state.detectors` lists the modes in use. Every monitor also keeps the largest value its statistic has
reached as a fraction of its firing level (`checks[].peak`): how near a healthy source came to an alarm.

| detector | what it sees | may set aside |
|---|---|---|
| `self_check` | a source's stated uncertainty far above its characterised one (median over 5 s above 4 x) | the source |
| `signal.invalid`, `signal.noise`, `signal.stuck` | one stream by itself: not a number; noise 25 x the characterised level; a value frozen for 2.5 s | the source (or that quantity) |
| `speed.share`, `yaw_rate.share` | two sources of one quantity: half of 5 s of evidence beyond 4 sigma | by vote, else (speed) by rank |
| `speed.cusum`, `yaw_rate.cusum` | the same pairs: a small difference that adds up (bias, 10 % scale, drift) | as above |
| `speed.track` | a receiver's track as a third opinion on the speed: its own verdicts and its votes | as above |
| `innovation` | a row class whose innovations are far beyond the stated uncertainty (position, heading, altitude, map rows, the mount's yaw row) | that row class, while it lasts |
| `position.lost` | absolute positions set aside for 8 s | re-anchors the position |
| `map.conflict` | the lines the matcher offers against the receiver's recent fixes, raw against raw (median of 8 fixes; beyond 12 m and beyond even odds that the receiver, with its present error, is the one that is right: section 15a) | the map's rows |

Never gating by construction: stale / absent sources (a fact, not a verdict), the feed's warm-up after a
silence, the timing and trend findings, repeats and outliers, the mount's findings, the sign rows.

The rule for the defaults: a detector gates only if it was quiet, with a stated margin, on the reference car's
healthy drives; the others report. `docs/FAULTS.md` ("Which detectors gate") has the numbers per detector and
what each `report` costs in the fault table. A CUSUM verdict that the share test confirms becomes the share
test's verdict (and gates with it); a verdict held by a CUSUM alone lapses after 30 s without a sample.

Position after a loss (`reanchor`, default `soft`): the position stays published with its uncertainty opened
by 40 m and the heading's by 20 degrees, under a new local frame id, and the next fixes pull it in; nothing is
invalid in between and nothing is confident. `hard` (the behaviour before L43) declares it unknown until the
next usable fix. A host's `reset` (vehicle input API) and a change of the receiver (`gnss.source`) forget the
position, the heading, the local frame and the receiver's states, and keep the motion and every calibration
state of the car's sensors (`Integrator.new_position_source`). A receiver that names itself may have its own
delay prior (`gnss_delay`: `{tag: [delay_s, sigma_s]}`; `synthetic`, the vehicle input API's tag of a simulated
drive, is stamped at its time of validity).

## 6. The object layer

**Track state** in the car frame (x forward, y left, origin the reference camera; z ignored):
`[x, y, vx, vy, ax]`, the target's position relative to the car and its velocity and longitudinal acceleration
*over the ground*, expressed in the car's axes. Process: the car's own motion from the ego estimator at that
time is the input,

    x' = vx - v_ego + w y      y' = vy - w x      vx' = ax + w vy      vy' = -w vx      ax' = -ax / tau + noise

so "we both stand" and "the gap closes" are different hypotheses (the lead filter's insight, kept), a turn
of the car does not look like lateral motion of the target, and the ego speed's uncertainty enters the
position's. A track in the standing regime (car and target stand) switches its process noise and is watched
by a CUSUM on its speed (the lead filter's measured behaviour: the `departed` cue).

**Sensors are models, not code paths.** An object source supplies per message a scan: time of validity
(minus its estimated delay), detections, and a sensor model with
`h(track) -> z, H`, `R(det)`, `initiate(det) -> track`, `p_detect(track)` (field of view, range, occlusion),
clutter density, the association gate, and its life-cycle policy. The core does association and bookkeeping
only. Built in this lane:

| object source | measurement | notes |
|---|---|---|
| camera lead (`road.leads`) | x, y, relative speed along x | reports at most the model's own lead hypotheses; noise and evidence rules are the lead filter's measured ones (L14, L27, L29, L32) |
| radar (front; any mount) | range, azimuth, range rate (non-linear `h`, EKF) | many detections per scan, clutter, multipath ghosts; stationary detections also give the ego speed |
| second camera (any mount) | bearing-dominant x, y in its own frame | detections converted through its mount |

**Association.** Per scan: gates (chi-square on the innovation with the source's own extra rules), then a
global nearest neighbour assignment (optimal assignment on the gated cost matrix; the scans here have at most
tens of detections). A detection assigned to no track may start a tentative track (if its sensor may
initiate). JPDA and MHT were considered and rejected for this sensor set (section 11).

**Existence.** Each track carries an existence probability updated by Bayes per scan from the sensor's
detection probability at the track's place and its clutter density (integrated probabilistic data
association's existence recursion, with hard assignment): a hit multiplies the odds by
`P_D L / lambda_c`, a miss inside the sensor's coverage by `1 - P_D`, a miss outside changes nothing.
`tentative -> confirmed -> coast -> deleted` are thresholds on it together with the source's policy (the
camera's measured acquire / weak-evidence / coast rules are that source's policy).

**Cross-sensor consistency.** Tracks started by different sensors that are statistically the same object
(position and velocity within a gate for 0.3 s) are merged by covariance intersection and keep the older id.
Per sensor pair the layer keeps the residual statistics of jointly seen tracks: camera range against radar
range (the camera's range scale, applied to the camera's measurements while a radar taught it: L14's
`vision_scale`), radar azimuth bias (the radar's mount yaw state). Ghost rules: a radar track at about twice
the range and twice the range rate of a nearer one on the same bearing is marked `ghost` (double bounce); a
track seen only by one radar, never moving, is `stationary` and not a lead candidate on its own.

**The lead** is the nearest confirmed, non-ghost track on the path. The path is the road model's plan while
the camera is usable, else the arc of the ego state (`v`, `w`). A track designated by the camera's lead
output is on the path by the model's own judgement; a radar-only track is on the path when its lateral offset
from the path is inside the corridor and it moves or has moved. The `lead` output keeps every field the api
served (`present`, `mode`, `standing`, `departed`, events, raw `vision`, `radar`, `vision_scale`) plus
validity and sources.

**Out-of-sequence scans.** The layer runs on scan time. A scan older than the layer's time by less than
0.3 s is applied with the track retrodicted to the scan's time by its own velocity (one-step retrodiction);
older scans are dropped and counted.

## 7. Extensibility: what "adding a sensor" is

A file in `chris_integrator/sources/` (or outside the package) that defines:

1. an **input description**: message types, group, rank, family, staleness;
2. a **measurement model**: message -> ego measurement rows and / or an object scan with its sensor model;
3. **mount and calibration**: a `Mount` (position, yaw in the car frame) and the `AuxState`s it wants estimated;

and one `register()` call (or a line in `integrator.json`). No change to `core/` or to any consumer: tracks
carry `sources`, the health table grows a row. Proven in the commit history of this lane: the front radar,
the second radar and the second camera were each added by commits that touch `sources/`, `sim/`, tests and
tools only.

## 8. Output contract of the estimator box (`contract.py`)

Three messages per road frame (and per vehicle sample while no camera frames arrive), all stamped with the
frame's `ts_mono_ns` = time of validity:

* **`ego_state`** (api 3; a superset of L37's): `speed`, `accel`, `yaw_rate`, `position` (lat, lon, local
  east / north, covariance, `hacc_m`), `heading`, `grade`, `odometer_m`, `standstill`; each quantity
  `{value, std, valid, reason, mode: fused | coast | none, age_s, sources}`. `calibration`: per source every
  estimated bias / scale / delay / mount angle with std and state. `sources`: per group `state` (`absent |
  stale | unchecked | ok | suspect | fault`), `usable`, reason, isolation rule, age, rate, NIS. `checks`: every
  monitor's verdict. `signals`: pass-through values with age. Counters (late, dropped, rejected).
* **`tracks`**: `frame: "car"`, every track with id, `state` (`tentative | confirmed | coast`), `exist`,
  position / velocity / acceleration with std, `age_s`, `meas_age_s` per source, `sources`, `flags`
  (`standing`, `ghost`, `stationary`, `on_path`), `class` (section 16: what kind of road user, with confidence
  and source, or null; non-critical), and per-sensor health of the object sources.
* **`lead`**: section 6.

Rules: `valid: false` always carries a `reason`; a value that is not a fresh fusion is marked (`mode: coast`,
`meas_age_s`); nothing is held. Display smoothing stays in `view` (downstream).

## 9. Architecture: the integrator is the top component (owner's ruling, 2026-11-19, during the lane)

"roadsense being just what analyses the video; its output is one of the sources to the integrator; the
integrator is what talks with the MFD and presents the API of consolidated data. With a new camera a new
model will just add more inputs to the integrator, and the integrator is where the decisions are taken."

```
 roadsense instance (per camera)        car backend (CAN, head unit)      radars, further roadsense instances
   camerad -> modeld, signd, lightd,      motion, controls, body, gnss       objects / road / signs / lights
   envd, laned; thin debug api                    |                                   |
        road, calib, signs, lights, env           | feed.sock (vehicle input API)     |
                     \                            v                                   v
                      +------------------>  INTEGRATOR  <------------------------------+
                         ego estimator · object layer · map side (matcher) · trackers and decisions
                         (lanes, ego lane, road geometry, rules / speed limit, ego light, sign context,
                         fused, view) · recorder · consolidated API (/v1, /v2, /ws) -> MFD
```

**The rule for the line.** A module stays in roadsense when it needs one camera's pixels or one camera's
model output and nothing else: one frame of reference, no ego motion from elsewhere, no other source, no
map, no memory across sources. A module belongs to the integrator when it combines sources, needs the ego
motion or the position, carries state across the gaps of a source, takes a decision a consumer acts on, or
is the consumer's interface.

| today | goes to | why |
|---|---|---|
| `camerad*`, `modeld`, `signd` / lights provider (roadsense-signs), `envd`, `laned` | roadsense | pixels of one camera |
| modeld's online calibrator | roadsense keeps the warp; the **mount estimate** is the integrator's (section 2), fed back as a `calib` suggestion | it needs the car's speed |
| `api/overlay.py` (`/preview`), the status page of raw topics, frame ring readers | roadsense (thin debug api) | draws one camera's raw output on its own picture |
| `api/governor.py` | roadsense | it sheds this phone's perception load by its heat; its status is an input (`governor`) the consolidated API shows |
| `api/lead.py` | integrator: object layer (done in this lane) | fusion across sensors, ego motion |
| `api/lanes.py` (lane-line tracker) | integrator | ego speed and yaw rate, the map's lane count prior, later a second camera's lines |
| `api/egolane.py`, `api/roadgeom.py`, `api/roadview.py`, `api/view.py` | integrator | map + camera + ego motion; `view` is the consumer's picture |
| `api/speedlimit.py`, `api/rules.py`, `api/rules/` | integrator | signs (camera) + map + odometer: a decision |
| `api/lights.py` (ego light selection) | integrator | plan + lanes + map signals; the per-light detection stays in the provider |
| `api/signctx.py` | integrator | explains a sign by the lead and the lights: cross-source |
| `api/fusion.py` (`fused`) | integrator | by definition |
| `api/recorder.py` | integrator | it records the consolidated inputs and outputs; the video DVR stays with the camera |
| `api/state.py`, `api/server.py`, `api/sockets.py` (HTTP / WebSocket / topics) | integrator: the consolidated API | the consumer's interface |
| `mapd/posfilter.py`, `mapd/vehicle.py` calibration | absorbed by the ego estimator (position, heading, scales, GNSS delay are its states) | position is estimated once |
| `mapd/engine.py` input handling (`position.sock`, vehicle input API 1.1) | integrator: the feed input channel (the car backend feeds the integrator directly) | not video analysis |
| `mapd/matcher.py`, `rsmap.py`, `context.py`, `limits/` (the map) | **chris-mapper**, its own box (section 15): it reads the integrator's pose, its match is a measurement | needs the position, estimates none |

What remains of roadsense's api: the sockets it already publishes (`road`, `calib`, `signs`, `lights`, `env`,
`camera`, `display_lanes`), the frame ring, and a small HTTP face for stand-alone use and debugging
(`/preview`, raw topic dump, health of its own daemons, the governor). It serves no tracked or fused topic.

**A second camera** is a second roadsense instance with its own run directory; its `road` / `signs` /
`lights` are further sources (`sources/` descriptions with a mount), nothing else changes.

**Inputs reach the integrator directly.** The car backend connects to the integrator's `feed.sock` and speaks
the vehicle input API as it does to mapd today (`gnss`, `motion`, `controls`, `body`, `vehicle`, `objects`,
`reset`; roadsense [docs/VEHICLE_API.md](https://github.com/PiotrTopa/chris-roadsense/blob/main/docs/VEHICLE_API.md)). The sources accept both the feeder's `motion` / `gnss` and mapd's relayed
`ego` / `gnss`, so the switch is a socket path on the car.

**Process and the cost of crossing it.** One daemon (`python3 -m chris_integrator`): it reads each
roadsense instance's sockets and the feed and runs the mixer (section 14) with the API as an output channel.
The integrator is the consumer that parses `road` for the trackers; a message that is only passed on (a raw
`road` line to a WebSocket client) keeps the line it arrived as and is never serialised again. Library form
for replay and tests: the same mixer and plug-ins on a clock that follows the messages.

**The position is estimated once.** The map box (section 15) reads the ego state's position and returns
`match` as a constraint measurement. What returns is the *map's geometry* (cross-track
to a road, the road's bearing), not the position that was sent; what keeps the loop from confirming itself
(which road it is stays a hypothesis with a share; the receiver's delay is measured outside the loop) is
section 15a.

**Consumers through the transition.** The consolidated API keeps today's message shapes, paths and topic
names (additive fields; `ego_state` api 3, new topic `tracks`). The MFD repoints an endpoint and the car
backend a feed socket; the report lists both.

**Its own repository.** The directory imports nothing from roadsense and speaks to it over sockets only, so
the split is a move of `chris-integrator/` plus packaging (`pyproject`, a unit file, its tests and tools). Licence:
roadsense is MIT; the integrator can be MIT as well (it contains no code of roadsense-signs, which is AGPL
and stays a separate provider process whose *messages* it reads).

**As built.** Everything in the table is where the table says: the trackers and decision logic are
`chris_integrator/trackers/` behind plug-in faces (`plugins/`), the consolidated HTTP / WebSocket API and the
recorder are output channels (`io/`), the car's feed and each roadsense instance's sockets are input channels,
the map is the box chris-mapper (section 15) and the position is estimated once. roadsense's api process is
the debug face of one camera: raw topics, `/preview`, the frame ring, the load governor (whose state it
publishes as the `governor` message the integrator marks measurement topics with). chris-integrator imports
nothing from roadsense and reads only its sockets. What a consumer sees is unchanged: the paths, topic names
and message shapes of the API are those roadsense's api served (api 2), with additions.

## 10. Simulator and test plan

`chris_integrator/sim/`: a world with known truth (ego trajectory with speed, curvature, grade profiles;
vehicles with lane offsets and speed profiles, cut-ins, a standing queue; roadside stationary objects) and
sensor models that emit the stack's real message types with noise, bias, scale, latency, jitter, drop-outs
and faults (vehicle feed silences with stale resumes, a knocked camera, a wrong vehicle speed scale, GNSS
lag / outage / jumps, radar clutter and double-bounce ghosts, a radar blackout). `sim/inject.py` adds
synthetic radars and a second camera to a *real* recording, derived consistently from it (the car's own
motion from the CAN speed and gyro, targets from the non-causally smoothed lead of the recording).

Shown (commands in the report): ego accuracy and NEES / coverage against simulated truth and against
held-out real sources (position against fixes not yet used; speed with the vehicle held out); delays, scales
and mount angles converging to simulated truth and to plausible values on the car; the incident detected and
its products unknown; isolation by vote with three sources (simulation, and injected radar on real drives:
camera fault and vehicle-speed fault); lead and lanes on the good drives against L37 frame by frame; tracks
against simulated truth (continuity, id switches, false tracks, ghosts, position error) and with injected
radar on real drives; behaviour under delays and drop-outs; CPU and memory.

## 11. Machinery chosen, and what it was compared with

* **EKF with error-state heading, sequential scalar updates** against a **UKF**: the only non-linearities are
  heading into position and the radar's polar measurement, both mild at these update rates; the UKF's sigma
  points cost 2n+1 propagations of a ~25-state vector per step for no accuracy gain here, and scalar
  sequential updates make per-row robust weighting, gating and NIS accounting natural. An invariant EKF
  (SE(2)) would improve consistency under large heading uncertainty at start; the heading starts unknown here
  and is handled by a wide prior and wrapped innovations; noted as a possible refinement.
* **Buffered re-run + fixed-lag RTS** against a **factor graph / sliding-window smoother** (GTSAM / iSAM2,
  what a survey-grade fusion would use) and against **one-step OOSM updates**: the re-run is exact for
  out-of-sequence data and re-linearises; a factor graph gives the same estimate for a linear-Gaussian window
  and better ones under strong non-linearity, at the price of a sparse solver dependency and tens of ms per
  solve in Python; openpilot's `locationd` (error-state EKF, buffered rewind for late measurements) is the
  production precedent for the choice made here.
* **Robust (Student-t / Huber) updates and chi-square gates** against plain gating: a gate alone discards
  information and locks out a lone source; a heavy-tailed likelihood is the principled model of the camera's
  measured tails.
* **Coloured-noise states (Gauss-Markov)** against inflating R or flooring the std (L37's patch).
* **IMM** (interacting multiple models): for the ego state the manoeuvre is measured directly at 10 Hz
  (speed, gyro), so a second ego model has nothing to add. For targets the standing / moving switch with a
  CUSUM change detector (quickest detection) is used; a full IMM over constant-velocity / constant-acceleration
  target models is the alternative, not adopted because the measured lead filter already beats its own
  earlier variants on the recordings and its two regimes are that switch.
* **GNN association with existence probabilities (IPDA-style)** against **JPDA** (soft assignment: better in
  dense clutter with unresolved targets, blurs well-separated road targets) and **MHT** (delayed decisions:
  best for crossing targets, an order more code and CPU). The sensor set here gives sparse, well-separated
  targets with strong range-rate discrimination.
* **Covariance intersection** for track-to-track fusion of unknown correlation, against naive fusion (double
  counting) and against keeping separate track lists (the consumer would have to associate).
* **Consistency**: NEES on simulation and NIS on real data are part of the test suite, not an afterthought;
  **FDI** uses parity between raw sources (pairs, vote) plus innovation monitoring.

## 12. From L37: kept, replaced

Kept: the source registry idea (input description + measurement model), families and ranks, the raw pair and
self monitors (`monitors.py`, unchanged in logic), isolation by rank or vote, the insert-and-re-run buffer
and fixed-lag smoothing (generalised from two linear channels to one EKF), the vehicle feed's warm-up rule,
the vision standstill detector, the `ego_state` quantity format and the consumers' helper functions, the
recorder and replay fixes, carried faults.

Replaced: two decoupled linear channels -> one EKF with position, heading, grade and calibration states;
GNSS as a non-fused witness -> positions fused with estimated delay and adaptive noise (the speed field stays
a witness, for the measured reason in section 1); the std floor -> a coloured-noise state; no object layer ->
section 6, with the lead tracker moved into it; a package inside the api -> a component with a daemon;
position only in mapd -> position in the integrator, the map as a measurement.

## 13. Limits known at design time

* No 6-axis IMU: roll, bank and pitch dynamics are not states; bank hides in `b_ay`, grade and accelerometer
  bias are separable only with altitude.
* Sideslip is not modelled (no-slip kinematics): fine for town driving, wrong on ice.
* The camera reports its lead hypotheses only (the recorder keeps `leads[0]`): multi-target evidence from the
  camera on real recordings is limited to that; the multi-target machinery is exercised by the simulator and
  the injected radars.
* Thresholds of the monitors rest on the drives of one day (L37 section 6); the simulator covers what the car
  cannot show, and no threshold is tuned to the drives without the report saying so.
* One absolute source. A slow drift of the receiver along the road is followed (across the road the map holds);
  a step is held back for the best part of a minute and then followed. With the receiver gone or distrusted the
  position is dead reckoning with the map across the road and at turns, and its stated uncertainty grows along
  the road until a turn or the receiver ends that (docs/FAULTS.md, "The receiver distrusted").
* The numbers of sections 3a, 4a and 15a (noise floor of the receiver's steps, the level, the satellite count
  at which the delay is opened) come from four drives of one day with one receiver, two of them with 3-5
  satellites; they are class attributes of the receiver's source.
* Python + numpy: the cost on the phone is an estimate from laptop measurements until measured there.

## 14. The mixer: channels, bus, plug-ins

The owner's picture: "the integrator is like an audio mixer, balancing and getting things together; it can
take many inputs, many outputs, and have plug-in input-output transformers." This is code structure
(`chris_integrator/mixer/`, 600 lines that know no sensor, tracker or consumer), not a metaphor.

### 14.1 The bus

Named **streams** of messages. A message travels in an **envelope**: the stream, the message (a JSON object),
`ts` (its `ts_mono_ns`: the time it is valid at, on the stack's one time axis), `t_arr` (when it reached the
bus), `source` (the channel or plug-in that put it there), a sequence number, and its serialised line, made at
most once and shared by every output channel.

A stream is declared by its producer (`StreamSpec`): `kind` (`raw`: a source's message as it came;
`measurement`; `estimate`: fused state; `decision`: what a consumer acts on; `context`: the map; `display`:
informational, may be smoothed; `status`), `time` (`frame`: one message per camera frame, stamped with the
frame's time of validity; `sample`: the source's own samples; `periodic`; `event`: on change), `public`
(served to consumers under its name, or wiring between plug-ins), a description. One producer per stream: a
second one is refused when it is added.

**Clock.** Plug-ins never read the wall clock. "Now" is the bus clock: the host's clocks in a running stack
(`LiveClock`), the messages' own time in a replay (`MessageClock`). A recording replayed through the mixer is
deterministic and runs the same code as the car.

### 14.2 Input channels

A channel declares the streams it provides and delivers messages (`channel.deliver(stream, msg, line)`). It
moves messages and holds no logic. In the stack:

| channel | transport | streams |
|---|---|---|
| `roadsense` (one per camera instance; a further instance's streams are suffixed, `road@rear`) | the instance's Unix sockets, JSON lines | `road`, `calib`, `camera`, `env`, `lights_raw`, `signs_raw`, `display_lanes`, `governor` |
| `vehicle` | `feed.sock` (the vehicle input API; also under the name `position.sock`), `POST /v2/vehicle` | `ego`, `vehicle`, `gnss`, `objects` (radars, further cameras, class detectors) |
| `map_in` (only without the map box) | a socket or a recording | `map` of an external matcher |
| a recorder session (`replay.py`) | a file | everything it recorded |

No channel is required: a plug-in whose input has no producer reports `idle` with the reason and the rest
runs.

### 14.3 The plug-in contract

A plug-in is a box that takes named streams and puts named streams back (`mixer/plugin.py`). It **declares**

* `name`;
* `inputs`: `In(stream, trigger, optional, feedback)`. A *trigger* input calls the plug-in for each of its
  messages; the others are read inside the call (`ctx.get(stream)`: the newest message; with `frame=True`
  only if it belongs to the same camera frame as the message being handled, so a box never pairs this frame's
  road with the last frame's lanes);
* `outputs`: `Out(stream, kind, time, public, doc)`;
* `DEFAULT_CFG`: its configuration with defaults; `<config dir>/<name>.json` overrides per key.

and **implements** `on_input(stream, msg, ctx)`, publishing with `ctx.emit(stream, msg)`. A stream it did not
declare cannot be read or written (the mixer raises): the declarations are the wiring, and
`GET /v2/mixer` shows them as they are.

**Time semantics.** An output for a camera frame carries the frame's `ts_mono_ns`. Periodic outputs are paced
on the bus clock (`ctx.now`). Nothing is delayed or smoothed on a measurement stream; display smoothing lives
in the `view` box only.

**Order.** Dispatch is synchronous and depth-first: a message goes to the output channels, then to each
plug-in triggered by it, in order, and what a plug-in emits is dispatched before the next plug-in runs. The
order is derived from the declarations (a producer before its consumers), stable in the order added; a loop
is an error unless one of its inputs is declared `feedback`. A feedback input is delivered when it is
published and read as "the newest there is" (the map match that returns to the estimator: a time-stamped
measurement, inserted where it belongs by the estimator's out-of-sequence machinery). A plug-in is never
re-entered: what its own output causes to come back waits until it has finished.

**Health.** `health()` returns `{"state": "ok" | "idle" | "degraded" | "fault", "reason", ...}`. The mixer adds
what it measures: calls, errors and the last error, cost per call, age of the last output, unsatisfied
inputs. An exception inside a plug-in is caught, counted and reported as that plug-in's; the other plug-ins
and the fan-out go on.

**Life cycle.** `reset(reason)` (a new drive, a camera restart), `close()`, `state_dump()` / `state_load()`
(what survives a restart: faults, calibration; one file per plug-in in the data directory).

**Adding, ordering, enabling, replacing.** At run time: `mixer.add(plugin)`, `remove(name)`,
`replace(name, plugin)`, `enable(name, on)`; the order is recomputed. From the configuration (`mixer.json`):
`{"disable": [...], "plugins": [{"class": "pkg.mod:Class", "cfg": {...}} | {"remote": {...}}],
"replace": {"lanes": {"class": ...}}}`. Removing a producer removes its streams; its consumers say
`idle: no producer for <stream>`.

**Alone.** `mixer.testing.Bench(plugin)` provides every declared input: `bench.feed(stream, msg)` returns what
the plug-in emitted. The logic under each plug-in is also a plain class with no bus in it
(`chris_integrator/trackers/`).

**Out of process, same contract** (`mixer/remote.py`). JSON lines over a child process' pipe or a Unix socket:
the plug-in's first line carries exactly the declarations above; the mixer sends
`{"in": stream, "msg": ..., "n": k}`, the plug-in answers `{"out": stream, "msg": ...}` lines and
`{"done": k, "health": ...}`. Two modes behind the same `Plugin` face: *sync* (the mixer waits for `done`, at
most a deadline: the outputs are on the bus before the next plug-in runs, a replay stays deterministic; the
hop is paid on every trigger) and *async* (outputs are published when they arrive; for boxes whose output is
not tied to a frame). A plug-in that does not answer is reported `fault`, its late answers are dropped, and
it is restarted. A plug-in written for the in-process contract runs out of process unchanged
(`remote.serve`); one in another language only has to speak the lines.

### 14.4 The boxes of the stack (`stack.py`: a configuration of the mixer)

| plug-in | triggers | reads | puts back | kind |
|---|---|---|---|---|
| `ride_watch` | `camera` | | (resets every plug-in on a new ride) | |
| `estimator` | `road`, and every type a registered source reads: `ego`, `vehicle`, `gnss`, `objects`, `calib`, `map` (feedback) | `governor` | `ego_state`, `tracks`, `lead` | estimate, estimate, decision |
| `mapper` | `ego_state` | | `map` | context |
| `lanes` | `road`, `map` | `ego_state` | `lanes` | measurement |
| `egolane` | `lanes`, `map` | `road`, `ego_state` | `egolane` (wiring) | estimate |
| `roadgeom` | `lanes`, `map` | `road`, `ego_state` | `roadgeom` (wiring) | estimate |
| `ego_light` | `lights_raw`, `lanes`, `map` | `road`, `calib`, `ego_state` | `lights` | decision |
| `sign_context` | `signs_raw` | `lead`, `lights`, `calib` | `signs` | measurement |
| `view` | `road`, `map`, `lights`, `signs`, `display_lanes` | `ego_state`, `lead`, `lanes`, `roadgeom`, `egolane`, `tracks`, `calib`, `governor` | `view` | display |
| `rules` | `road`, `signs`, `env`, `lights` | `ego_state`, `calib` | `speed_limit`, `rules` | decision |
| `fused` | `road`, `map`, `env`, `signs` | `speed_limit`, `ego_state`, `lanes`, `egolane`, `lead` | `fused` | decision |

The estimator box is the `Integrator` of sections 2-8, unchanged: its source registry is how a *sensor* is
added (a source file and a configuration line, section 7), the mixer is how a *box* is added. A source added
to the configuration adds an input to the estimator plug-in and nothing else.

Services shared through the mixer (not streams): `rates` (the `rate` marker of measurement topics: measured
rate and latency against the configured floors, with the roadsense governor's state as an input; a provider
that thins itself out on purpose announces it, and the floor follows: a `lights` message with `idle: {"hz": H}`
comes from a light detector that tracks nothing and runs H passes a second, and while that holds the floor is
0.8 x H, measured over the idle arrivals without the bias of a short window, so the stream is `slow` only when it
falls behind the pattern it announces; `env` has no floor), `store` (carried state), `reset`.

### 14.5 Output channels

An output channel is a function called with every envelope of the public streams (or of the streams it
names): the consolidated HTTP / WebSocket API (`io/http.py`; `GET /v2/<stream>` is the newest envelope, `/ws`
the stream of lines), `integrator.sock` (JSON lines for other processes; a first line
`{"subscribe": [...]}` narrows), the recorder (`io/recorder.py`). Any number may be attached; each sees the
same lines.

### 14.6 What the mixer costs

Measured on a laptop core, a 15 minute drive replayed: the mixer with its ten boxes costs the same as the
hand-wired pipeline it replaced (3.34 against 3.33 ms per camera frame, the same messages byte for byte);
the dispatch is not visible in the total. A box moved out of process in sync mode costs its hop on every
trigger: 0.4 ms per message for the map box, whose input is a 5 kB `ego_state` (serialising it, two pipe
crossings, parsing the answer); its outputs are the same message for message.

### 14.7 What the daemon costs on a phone core

Measured on the reference phone (a little core of a Snapdragon 845 at 1.77 GHz, Python 3.14, numpy 2.5; one
camera at 15 road frames a second, sign and light providers at 5 a second, a recorder, one WebSocket consumer
that takes `road`, `lanes`, `view`, `lead` and the decisions): about **17 ms of CPU per road frame** for the
whole per-message path on a quiet core and 25 ms with the camera pipeline running beside it (the kernel's
part of 15 kB lines in and out, the event loop, shared caches): 39 % of one little core. The same path costs
about 3 ms on a laptop core: the ratio is 6.5, and what dominates is not arithmetic but the unit prices of
Python on such a core: 1 us to read a JSON number and 1.5 us to write one, 2 us for a `round()`, 5-18 us for a
numpy call on a small array. By stage (ms per road frame): reading the inputs 1.4, the estimator 5.2, lanes
2.1, view 1.7, the other boxes 1.9, the recorder 2.7, serialising for the WebSocket consumer 1.3, dispatch 0.6.

What keeps it there, and what an extension has to respect:

* **A message is serialised only when something takes its line** (`Envelope.line`): an output channel must
  decide whether it wants a message (subscription, rate cap) before it touches `line`. The WebSocket channel
  serialises a stream only for a subscriber of it (`Api.subscribe(cb, topics)`), the socket channel makes its
  greeting lines when a client connects, the recorder asks after its rate cap.
* **Lists of numbers are rounded and written in bulk** by the native core (docs/NATIVE.md): `round_list`,
  `round_lists` give exactly `round()`'s (`np.round`'s) results, and the recorder's `road` line is written in
  one pass from the message (`slim_road_line`), byte for byte what `json.dumps` of the rounded message gives.
* **`road` is parsed without what the default boxes do not read** (`fastjson.trim_road`: most of the plan's
  arrays, the lines' heights; 40 % of the line). The line itself goes to every consumer whole, and
  `GET /v2/<topic>` serves the line. A stack with boxes of its own gets the whole message (`Stack.trim_road`,
  `"road_whole": true` in `mixer.json` forces it).
* On arrays of a few dozen values a numpy call costs more than the arithmetic: the trackers convert and test
  a frame's lists in one array, not one by one, and avoid `np.median` and friends on tiny inputs.

* **The arithmetic and the text are native** (`chris_integrator/native`, docs/NATIVE.md): the filter's
  prediction, scalar row and smoother step, the lane tracker's arrays, the road geometry's centre line, the
  view's drawn lines, `json.dumps` / `json.loads`, the `road` line's trim and the recorder's reduced form and
  `round` are in one C file, which the package requires. Each exists once; `tests/test_native.py` holds them to
  CPython's json and round and to the results of the pure-Python references they replaced (frozen golden data:
  bit for bit where the steps are elementwise, the filter's sums of products to rounding). On the reference
  phone's little core the native core of lane L50 took the per-message path from 15.3 to 7.9 ms per road frame
  on the bench; lane cost-rest took a further part of the rest (its report has the numbers). Most of what remains
  is Python byte code in the estimator's bookkeeping, the trackers' decisions and the mixer.

`tools/cost_capture.py` records what a running daemon is fed, `tools/cost_bench.py` replays it through the
daemon's whole per-message path (cost by stage, and a dump of every output for comparing two trees line by
line), `tools/cost_live.py` measures a running unit.

## 15. The map box (chris-mapper)

Its own component (top-level `chris-mapper/`, package `chris_mapper`, its own tests, README and licence;
nothing of chris-integrator is imported by it and nothing of it by the mixer core). Why its own directory
rather than a plug-in package inside the integrator: it has its own data and tooling (the pack format, the
pack builder, OpenStreetMap's licence on the data), its own heavy state (a tile cache: memory and I/O that a
host may want in another process, on other cores), its own test world (a network, not a drive), and it is
useful without the integrator (any pose source can drive it). The integrator keeps only the box's plug-in
face (`plugins/mapper.py`, 50 lines).

* **In**: `ego_state`. The pose with its uncertainty: `position` (lat, lon, east / north covariance, `mode`:
  `fused` while an absolute fix is fresh, `coast` in dead reckoning; the local frame id), `heading` with std,
  `speed`, `odometer_m`, the time of validity.
* **Out**: `map`, every 0.45 s of input time. `match`: the matched point and the road's bearing, confidence,
  ambiguity: *geometry of the map*, which the estimator's map source takes as a measurement (cross-track to
  the road = 0 about a lane-offset state, heading = the road's bearing; robust rows, set aside while far off:
  section 5). `position`: the pose that was used and its state. `road`, `speed_limit`, `ahead` (features,
  junctions, curves, limit changes, turn lanes, `junction_view`): the map context, read by `lanes`, `egolane`,
  `roadgeom`, `ego_light`, `view`, `fused`.
* **It estimates no position.** No filter, no sensor input. A pose that is 30 m off the road comes out 30 m
  off the road with a match that says so. When the estimator anchors its position anew the box starts the
  match anew.

The loop estimator -> map box -> estimator is declared (`map` is a feedback input of the estimator); section
15a is how it is kept from confirming itself.

In process by default; `mapper.json` `{"remote": {"cmd": ["python3", "-m", "chris_mapper", "--plugin"]}}` runs
it as its own process behind the same contract (the outputs are the same message for message; test and lane
report).

## 15a. The loop estimator - map box (L47)

The match is made from the estimate it then corrects. Measured before L47 on the drive with the fewest
satellites: through the box the delay state walked to 4.6 s, the box matched nearer its road (2 m against
12 m) and with higher confidence than the legacy matcher, and held-out fixes lay 63 m from the estimate. What
was wrong, and what is done about each:

| what | why it is wrong | now |
|---|---|---|
| The delay and the along-track position are one quantity at constant speed, and the map says nothing along the road. With the map's rows stiff across the road and the receiver's error modelled as white, every along-track residual of a receiver that trails went into the delay state. | Nothing in the loop measures the delay. | The delay is measured outside the loop, from the receiver's step lengths against the odometer (section 3a); the receiver's slowly varying error is a pair of states of honest size; position rows do not teach the delay while the receiver is unsteady. |
| A match with share >= 0.7 was fused with 1 m; the pose then sat on that line, which made the line fit better. The matcher's confidence was the pose's own distance returned. | The choice of road was counted as certain, and each match as independent of the last. | The matcher reports the candidates with their shares (`match.weight`, `match.alternatives`: from the fit alone, with a mild preference for continuity instead of the matcher's own strong one). The row's target is the share-weighted line of the winner and the candidates parallel to it; the spread of those lines multiplies a state `amb` of unit variance that the map's rows carry and never correct (a consider state). The position across the road stays as uncertain as the choice between the roads, however many matches agree. |
| An ambiguous match (share < 0.7) gave no row at all. | "One of these two roads 8 m apart" is information. | The same row, with the spread. Candidates that are not parallel to the winner (a crossing road, the piece after a corner) take weight from the row (their share is the probability that this line does not constrain the car), and below half the row is not made. |
| The heading row took the bearing at the matched point. | In a bend that bearing is only as good as the position along the road; a road is a line, and which way along it the car drives has to be known already. | Its sigma grows with the road's change of direction 15 m either side of the match; no heading row while candidates of another direction hold 15 %, or while the heading is known worse than 20 degrees. |
| `map.conflict` set the map aside when the receiver's fixes lay 12 m beside the matched road. | With a receiver whose slowly varying error is 10-17 m that is 1 sigma: the map was set aside for 16-21 % of healthy driving, and in simulation a correct map was taken away from a receiver 30 m off. Below 12 m a parallel road was not seen at all. | Odds: how likely the offset is for the receiver's present error (its measured level) against how likely a wrong match is at all (3 %, anywhere within 60 m either side). Above even odds the map's rows are set aside; below, "the receiver is right and this road is wrong" is one more candidate line in the spread. What a receiver can tell apart is about 3 of its own sigmas (12 m for a 3 m receiver, 35-40 m for the car's with few satellites); below that the answer is the stated uncertainty, and the matcher's own candidates. |

Where the map does tell the position along the road: at a turn. The new piece's line constrains what was
along-track before the turn, and the turn is where the gyro says it is. That needs nothing special, only a
covariance that is honest along the track before the turn (then the matcher's search reaches the new piece,
and the row pulls). Measured on the hand-made network: the east error's stated sigma falls from 3-8 m before a
90 degree turn to 2.2-2.5 m after it, the error itself to under 1 m.

Alternatives looked at, and why not: (1) a second position filter without the map's rows, to give the matcher
a pose the map has not touched: exact, but the position would be estimated twice, against the ruling this
component is built on. (2) Covariance intersection of the map row with the state: it assumes nothing about
the correlation and therefore throws away the one thing that is known about it (it is the choice of road,
which is constant between junctions); the consider state says exactly that. (3) A matcher that carries
several road hypotheses through time (an HMM over raw fixes and odometry): the right matcher for a pose
source that cannot be trusted, and the next step if the candidates' shares prove too crude at junctions; it
needs the fixes inside the map box, which today reads the ego estimate only. (4) Bounding the delay state
(to 3.5 s) and nothing else: built first; in simulation with a receiver that trails by 8 s it left the position
39 m behind with 9 times the stated variance.

## 16. Object classes: non-critical metadata

Road users are to be distinguished (human, bike, bus, truck, car) and drawn accordingly. In the contract this
is an attribute of a track, never a measurement:

* `tracks.tracks[].class`, `lead.class`: `{"label", "confidence", "source", "age_s", "votes"}` or `null`.
  `label` is open vocabulary, lower case, with common aliases folded (`person` -> `human`, `bicycle` -> `bike`,
  ...). `confidence` is the decayed share of confident votes for the label; `source` the sensor that said so.
* `view.objects[].cls` (vehicles): `{"label", "confidence", "source"}` or `null`. A consumer draws by it and
  must draw something sensible without it.

Two ways in, both an input only:

* a detector that measures positions well enough to track by is a `camera_objects` source; the `class` (and
  `class_conf` or `prob`) of its objects feeds the attribute of the track the object is associated with;
* a detector that only says *what* is where it points is an `object_classes` source: `objects` messages of
  `kind: "classes"` with `class`, `class_conf` and either `x_m`, `y_m` or `bearing_deg` (`range_m` optional).
  Each object is attached to the track it points at; nothing is started, moved, confirmed or kept alive by
  it, the lead is not chosen by it, and it does not count as an object source.

By construction nothing reads the attribute: gating, association, existence, the lead selection and the ego
estimate are the same, number for number, with and without a class detector (a test asserts exactly that).

**Lane L42: the first class detector, and the vote** (report: roadsense [`docs/lanes/L42-road-users.md`](https://github.com/PiotrTopa/chris-roadsense/blob/main/docs/lanes/L42-road-users.md)).

* The detector is the traffic-light detector of roadsense-signs (BDD100K YOLO11n, eight road classes besides
  lights and signs): `signd --lights-road-users` decodes its road-user boxes from the inference that already runs
  for the lights and writes them into every `lights` message as `road_users` (`cls`, `score`, `box`, `clip`:
  edges of the view's region the box touches, `rider`: a rider paired with the bike / motorbike under it, `p`:
  the eight class scores); `"road_users": null` = shed by the load governor (first step, `road_users_hz` 0).
* The `road_users` box (`plugins/roadusers.py`, `core/roadusers.py`) places each box with the camera's `calib`:
  bearing from the box's centre column (1 sigma: a quarter of the box width + 0.3 deg), ground distance from
  the bottom edge on a flat road when the box is not cut at the bottom of the detector's region (1 sigma from
  0.6 deg of pitch, 3 px + 4 % of the box height, 5 cm of camera height: about 15 % at 20 m, 30 % at 40 m), no
  distance from the box size (a car's width depends on the view). Stream `road_users` (public).
* The `road_users` source kind (in the default estimator configuration) attaches each road user to at most one
  track and each track to at most one road user per message (bearing gate 3 sigma, at least the half width of a
  vehicle at the track's distance; distance within 3 sigma where both have one) and folds the class into the
  track's vote. Annotation only, like `object_classes`.
* The vote (`core/classvote.py`, used by every class input): per label a tally of score x quality (a box cut by
  the region weighs 0.7), half-life 30 s, the sum held at 8 (a share over the recent life of the track); the
  first label is shown once two observations are in and it leads by 1.5x, another label replaces it only at
  2x the shown one's tally; `confidence` = share x (1 - exp(-total)); below 0.35 the class is `null`.
  Aliases: `person` -> `human`, `rider` / `bicycle` -> `bike`, `motor` / `motorcycle` -> `motorbike`,
  `train` -> `tram`; `van` stays `van` (the detector has no van class).
* On a car without range sensors the object layer holds only the lead, so other road users have no track to
  carry their class. The view draws them itself (`trackers/roadusers_view.py`): `view.objects` of kind
  `road_user` (role `other`, `cls`, `sx` / `sy`, phases and fades) for boxes with a ground distance only,
  riderless bikes and motorbikes left out, the lead's own detection not drawn twice; display only, never a
  measurement topic. The equality test for this path: `tests/test_road_users.py` (tracks, lead, ego state
  equal; the view equal apart from the class and the `road_user` objects).


## 17. The decision layer (lane DECIDE, 2026-11-23)

`chris_integrator/decide` (plug-in `decide`, topic `decide`, docs/TOPICS.md). It replaces the radar's fixed
time-to-reach (`threat_ttc_s`) as the source of the view's threat flags and alert with a measure of what the
driver would have to do: for the car against every object, over the object's predicted modes and the car's own,

* the **collision probability** over the horizon: both position distributions (the object's covariance per time
  from the prediction, the car's own execution spread, bounded) and both extents (the object's length, width and
  heading with their uncertainty; the car's footprint from the reference camera: 2.1 m to the front bumper, 2.35 m
  to the rear, 1.76 m wide), by quasi-Monte Carlo with draws coherent in time (a draw is a whole future: the
  prediction error is mostly a bias of the initial state and the manoeuvre, not white noise per step);
* the **demand required to avoid** (Brännström et al., threat assessment for arbitrary collisions): the least
  constant braking, steering left or right, or braking and steering together on the friction circle, applied from
  now with the build-up of brakes and steering, that keeps the car clear of the object's footprint inflated by k
  sigma (bounded: a poorly ranged camera object's spread is the collision probability's matter, not a wall);
  steering must also keep clear of the other objects; against the **grip available** (mu g of the road
  condition, the share a driver gets, falling with speed, the turning circle at low speed): the brake, steer and
  combined threat numbers;
* **time-to-brake and time-to-steer**: the last onset of full braking or steering that still avoids, and the
  **time margin** tm = max(ttb, tts) - the driver's reaction time (0.9 quantile of a log-normal, median 1 s),
  only where the mean futures make contact with the footprint as it is: a time margin needs a collision to have a
  time. Against the margin alone a car passed 0.2 m wider than its spread "had" none left (the margin breached at
  once; a quarter of the false warnings of the degraded futures in the simulated parked streets, 2026-11-24). The
  margin's inflation is scaled down to the gap the car has now (no demand keeps a margin wider than that);
* time to collision, time headway and the least separation on the mean futures, secondary;
* a **level** per pair from per-class thresholds on the threat number and the time margin, and per object the
  highest level whose probability mass (weight x pc over the pairs at or above it) reaches its minimum, with
  entry and exit threshold sets (hysteresis) and the reasoning (which pair, which measure).

Weights are the **unconditioned** mode probabilities (the other party does not yield); the conditioned figure is
given alongside. The **alert logic** (alerts.py) stages information, caution, warning, imminent with entry and
exit holds, minimum holds, a latched imminent, suppression (driver braking or steering away enough, low speed,
sensors degraded), visibility and existence gates, a stable primary, and modes off / shadow / on per alert.

Degraded mode (no prediction layer yet): the radar's own two hypotheses as constant-velocity modes (it stands
with p_stationary; it drives on at its velocity with the rest; along the path's bends for traffic along it), the
car's own modes from its motion and the plan; footprints of standing objects parallel to the road at their place.
Three rules then stand in for the prediction's lanes, all from the radar's own evidence: an object whose velocity
is not observed yet raises at most information until its p_stationary has moved off the prior to 0.65 (a lead
appearing far ahead falls off within half a second, a standing car rises); one the radar calls parked raises at
most information; one that most probably stands rises above caution only when the radar's corridor test puts it
surely in the path. Measured on the simulator (seed 1, by day; `tools/decide_replay.py --scenario ...`):

| scenario | today's radar flags (`threat` / `obstacle` s) | decide (`threat` / `obstacle` s, stages >= caution said) |
|---|---|---|
| stopped_cruise_14 (no reaction; impact at 15.5 s) | obstacle at 12.8 s, threat at 13.7 s | caution 12.0 s, warning 12.9 s (2.6 s before impact) |
| stopped_cruise_25 (impact at 12.5 s) | threat at 11.0 s | warning 11.0 s (1.5 s before; the camera confirms the car at ~50 m) |
| curve_parked | 0.6 / 1.9 | 3.0 / 4.4 (on parked cars the radar itself places in its corridor) |
| slow_lead | 1.1 / 0.0 | 2.8 / 3.5 |
| parked_street, adjacent, oncoming, junction | 0 / 0 | 0 / <= 0.1 |

On fixtures with contract futures (decide/fixtures.py: stationary in path, lead braking, cut-in, crossing,
oncoming on our side, a pedestrian stepping out; parked row, adjacent lane, oncoming passing) the collision
courses warn 1.5 to 3 s before contact and the rest stays at stage 0 (tests/test_decide.py).


### 17.1 A standing vehicle's place under a camera's range (lane CURVES, 2026-11-24)

Why the alert said parked cars on bends, measured on the simulator (docs/TOPICS.md has the constants):

* The tracker's range is known to 17-20 % of itself at every age and distance (its error within that, 10-17 % long
  below 30 m); the bearing to a tenth of a degree. The place is a needle along the line of sight, and on a bend
  of 110 m radius that needle crosses the road at 0.2 m per metre of range at 45 m: a car parked on the outside
  of the bend and a car stopped in the lane share a line of sight, 6 m of range apart. At 45 m the honest
  probability that a roadside car is in the lane is 0.2-0.4 whenever its range reads half a sigma short; with 50
  parked cars a run, that happens for a second or so several times a run.
* What was wrong on our side: the lane filter multiplied the same range error in every 0.4 s (certainty out of a
  one-second error); the futures stood at the track's mean with an upright spread and were weighed by the lanes,
  so the doubt was counted twice for the car that IS in the lane and not at all for the futures that "moved" a
  metre; the car's straight-on future held a fifth of every frame.
* What is done: the place is read along the line of sight; a thing never seen to move takes its lane from the
  track's place now; a standing vehicle whose range leaves both open gets futures of two places (in the car's way,
  beside it), the first with the probability the ranges give it, the second with the floor a standing thing beside
  the lane always had; where cars have been seen standing beside the road, one standing in the lane is that many
  times less likely; a camera velocity that the range's scale explains does not make a roadside car move; and an
  alert on an object whose place is in doubt needs the conflict to last in proportion to how little it exceeds
  the level's minimum (a sequential test in place of a fixed hold).
* What was tried and withdrawn, each for a measured reason: the lane from the place alone for everything
  (cut-ins and crossing traffic were no longer recognised: how a thing moves does add up); two places for moving
  vehicles (a car in the next lane at speed got a standing future in the car's way); "in the way" as the weight
  of a vehicle whose place is open only across the line of sight (a parked car on a real drive whose track stood
  a metre nearer the plan than the car was passed was said); the velocity's scale bound for vehicles in the
  car's way (a lead read at 17 instead of 26 m/s became a stopped car) and with a range share above 0.3 (a car
  keeping the car's own speed 14 m ahead became a standing one on a real drive); the evidence rule on the whole
  collision probability instead of the mass at the level (no gain on the car half in the lane, one false threat
  on an oncoming car).
* What it cannot do: tell the two cars apart before the range does. That is the tracker's (parallax ranging of
  standing objects, the range scale in its velocity covariance). The reconstruction of the needle from `std`
  assumes a spread along the line of sight; a correlation in the radar message (`std_rho`, the correlation of
  the x and y errors, or `std_los`: 1 sigma along and across the line of sight) would replace it.

### 17.2 The needle as the radar states it, waiting for the range, pedestrians at parked cars (lane SPREAD, 2026-11-24)

Numbers: `tools/alert_truth.py` (the said episodes against the simulator's truth; six streets and two pedestrian
controls x seeds 1-6 x day and night, thirteen scripted conflicts x seeds 1-3 x day and night; paired over the same
runs, +- the bootstrap's spread over seeds), the plain and the realistic (`DETIN`) stream; the tracker's knobs
`ctx_prior` + `std_rho` + `path_normal` ("C"), `size_learn` ("S"), both ("CS").

**The radar's own needle (`std_rho`).** With the radar's `std_rho` the place's covariance is complete, and the
predictor reads the needle along its own long axis instead of rebuilding one along the line of sight from `std`
(native `std_rho`, 1; the futures of a place start with that covariance too: a standing car's one place was an
upright box; a pedestrian's and a cyclist's futures keep the spread they had). The rebuilt needle was a faithful
one. With the field read (and the two pedestrian defaults below) the false episodes of the six streets go 58 ->
55 (-3 +- 2.5) and 71 -> 74 (+3 +- 2.6) with C on the plain and the realistic stream, 67 -> 68 (+1 +- 2.6) and
77 -> 79 (+2 +- 1.9) with CS; of the two controls 6 -> 7, 9 -> 10, 5 -> 7, 8 -> 10 (+- 1.0-1.3: a parked car,
the standing pedestrian's lane read along its needle); the conflicts with a vehicle: none lost, one gained and four
earlier on the realistic stream (a car half in the lane 1.08 -> 2.44 s before), one later in each of three sets
(3.42 -> 3.25 s, 2.82 -> 2.74 s, 2.64 -> 2.32 s). Nothing either way. Things behind the car take three places of
their needle, not nine, and no second place (the rebuilt needle left most of them out by its fit): the predictor
alone costs 5 % fewer instructions on dense streets with the field read. With the knob 0 and the field present
every run is what it was; without the field every run of the standard sets is what it was (54 and 81 false, the
vehicle conflicts 32 and 36 of 48 at the same frames).

**Can the decision wait for the range?** A standing car of two places (17.1) is at ONE of them, and the car learns
which as it closes: the range's spread shrinks with the range. `place_hold` (off) puts that as a rule: the two
places, d apart on the ground, are told apart (two spreads between them, the spread 9 % of the range: what the
tracker states of a clean track at 15-40 m) once the car is within d / 0.18 of their middle: in t_res. The driver
must act when braking at 3 m/s^2 after the reaction time still stops short of the place in the way: in t_act.
While t_res <= t_act the object's entry level is held at information. Measured: **it holds nothing.** With
comfortable braking not one run of 1056 (four tracker settings, two streams) changes, and that is arithmetic, not
chance: caution begins when full braking has a time margin of 1.5 s, comfortable braking is due when
1.5 s - v (1 / (2 x 3) - 1 / (2 x 7.3)) of that is gone: 0.2 s at 13 m/s, nothing from 15 m/s, 0.7 s at 8 m/s. The
false episodes of roadside cars are first said 0.8-3.5 s before the place in the way would be reached (median
1.9 s): by then acting cannot wait for anything. With emergency braking as the bound (`place_hold_decel` 7) the
hold takes 4-5 false episodes of 54-83 (-5 +- 2.2, -4 +- 2.2, -4 +- 1.7, 0) and says two true warnings later in
each realistic set (a car half in the lane 2.10 -> 1.08 s before, a stopped car in a bend 2.74 -> 2.15 s): not a
trade to make. On the recorded drive (8 m/s) it held one of the two episodes for 1.1 s; it was then said one stage
higher.

What the arrays say instead (the two-place episodes of the base runs): the two places are 6-20 m apart, two to
four of the 9 % spreads: by the tracker's own figure for a clean track they are told apart already, and the
episode is said because the track is NOT that track: its range is off by one of its stated sigmas, which is a
fifth of the range (a car 25 m away read 5-9 m short), its extent is another car's (a parked car 1.8 m wide
tracked 2.4 m wide and 9.6 m long: two cars' boxes; its road-side face 0.05 m from the car's side instead of
0.8 m), or its place has just hopped. Of the 54 / 81 false episodes of the six streets (plain / realistic, no
tracker knob) 39 / 60 are on a parked car and 22 / 33 on one given two places; when first said the place in the
way is 2.1 / 1.6 s ahead (median; a tenth under 0.6 s, a tenth over 3.1 s), the two places 7.9 / 6.9 m apart, the
track's range off by 0.6 / 0.8 of its stated spread; by cause: place jump 26 / 25, the track's mean in the lane
7 / 14, moving futures of a parked car 5 / 14, the stated spread alone 1 / 9. 27 / 55 go on to imminent: the range
does not resolve as the car closes (the simulator's detector boxes a near car 2-3 times under 25 m). Nothing
in prediction or decision tells these from a car that is where its track says: that is the tracker's range and
association.

**An alert said without an object.** Of the said episodes 2-7 a set had no object when they were first said: a
stage suppressed since its entry (on a bend the driver's own steering "steers away"; braking; a crawl), or only
ever at information, whose object then left the list: without a record the suppression rules have nothing to
judge, and the held stage was said until its minimum hold ran out. `unsaid_needs_object` (alerts, on): an alert
not said since it left none is not said without its object (`suppressed: "object_gone"`). Paired, the eight sets:
false episodes 54 -> 53, 81 -> 79 (no tracker knob), 55 -> 54, 74 -> 74 (C), 69 -> 66, 77 -> 76 (S), 68 -> 64,
79 -> 79 (CS): -12 of 557, none up; every conflict warning at the frame it had; the recorded drives as they were.

**The car half in the lane** (`half_in_lane_13`; the obstacle's truth beside its track, six seeds, `alert_truth.py
run --trace`). By day on the plain stream, from 3 to 2 s before contact the track's centre stands 1.08 m further
from the lane than the car (its body clear of the car's swept strip in 88 % of the frames; from 2 to 1 s 0.54 m and
61 %), its width is right to 0.03 m, and its range is 4.3 m long (0.65 of the stated spread): on the bend a range
read long IS a place further out. On the straight (`half_in_lane_straight_13`) the centre is right to 0.02-0.14 m
and the body is in the strip in 80-90 % of the frames. So the decision uses the footprint and its width as they
are (`lane_step`: "in the way" is the body reaching the strip, half the car's width plus half the object's); what
it is given is a needle of which a fifth to a third lies in the way (`p_place` 0.30-0.37), honest for a range
0.65 sigma long, and indistinguishable from a roadside car read short. The bias (+2.5 to +4.3 m at 20-40 m: the
height prior's 1.6 m against 1.5 m) is the tracker's: with `ctx_prior` + `size_learn` the lead of that run is back
at 2.64 s on the plain stream (the half-in-lane runs' mean lead 1.29 -> 1.82 s), while on the realistic stream
`ctx_prior` loses three of eight half-in-lane warnings (the car's "it stands beside the road" comes sooner, and
on the straight its body, 0.3 m further out in the track than in truth, is at the strip's very edge: P(in the way)
0.4-0.5, and the alert's evidence rule waits). Entering by time once the time margin is gone
(`entry_evidence_tm_s` 0, off) brings two half-in-lane warnings back, 0.46 and 0.66 s before contact, and 15 +- 3.8
false episodes with them (59 -> 74; 22 -> 34 in the conflict runs): the evidence rule earns its keep.

**A pedestrian at parked cars.** `ped_between_parked_13` and three more (`PED_SCENARIOS`: in plain view on a
straight road, hidden between two vans until 1.2 s before, a child running), with two controls (the same pedestrian
not stepping out). Counted honestly (the alert's primary is the pedestrian's own track, not a parked van's on the
same line of sight) the pedestrian was warned of in 6 of 48 runs on the plain stream (0.4-1.1 s before) and 3 of 48
on the realistic one, never in plain view by day, never the child, never from between the vans on the realistic
stream. Why, from the arrays (six seeds):

| | first confirmed (median, s before contact) | never tracked | the step is taken | track's place behind the pedestrian at -1.0 / -0.5 s | tracker speed > 0.7 m/s |
|---|---|---|---|---|---|
| between parked cars, day (plain / realistic) | 2.1 / 2.7 | 0 / 0 of 6 | 1.86 s before | 0.7 / 1.3 m, 1.1 / 0.5 m | in 4 of 6, 0.8-0.9 s after the step |
| ... night | 2.1 / 1.5 | 1 / 3 | | 0.3 / 0.3 m, 0.5 m | 4 / 3 of 6 |
| in plain view, day | 1.9 / 2.8 | 0 / 0 | 1.5 s before | 0.5 / 1.0 m, 0.7 / 1.2 m | 3 / 0 of 6 |
| ... night | 1.3 / 3.1 | 5 / 5 | | | 0 / 1 |
| hidden between vans | 0.9 / - | 5 / 6 (day), 6 / 6 (night) | 2.36 s before | | |
| a child, day | 2.7 / 1.7 | 1 / 1 | 1.18 s before | | 0 / 0 |

A pedestrian in plain view from 100 m is confirmed 27-37 m away; after the step the track's place stays where it
stood (the standing hypothesis holds it: 0.5-1.3 m behind within a second, its stated spread 0.2 m) and no box
comes in the last second; at night it is mostly never tracked. A warning that needs the pedestrian to have moved
can lead by 1.5-1.9 s at the very most. That is the tracker's (confirmation of pedestrians, the place following a
step, night). What prediction and decision do with what they get:

* `ped_carry` (native, 1): **a defect.** "In another lane the change that got it there is over" is a vehicle's
  rule; applied to a pedestrian it turned four fifths of "about to step out" into "walks along" at the moment the
  pedestrian entered the car's lane, and half of the rest into the priors: the crossing futures of a pedestrian
  the tracker saw walking out fell from 0.72 to 0.19 as it crossed the lane's edge and took 0.3 s to come back
  (the arrays of the base runs: `walk` 0.66-0.89 with the pedestrian 0.5-0.75 s from contact). Now the step out
  goes on as the crossing it is.
* `ped_onset` (native, 1): the first step read from the track's place (0.3 m from where it stood, against the
  spread a CHANGE of its offset has) before the tracker's velocity shows it; not a step: the track's place
  settling as the car closes (a standing pedestrian's track came 1.6 m towards the road over 4 s at night: the
  place it stood at follows 0.3 m/s), a correction faster than anyone runs, anything away from the road. On a
  place that follows the pedestrian the crossing futures hold 0.5 0.6 s after the step and 0.9 after 0.8 s where
  they held 0.07.
* `ped_cov` (native, 0): the crossing futures start with the place's own spread, not a round one of the same size
  (a pedestrian 2.9 m beside the lane's middle, range known to 2.8 m, stood "in the lane" with a fifth of it).
  One more warning and two earlier ones on the plain stream; the futures' likelihood at 1 s a little worse
  (`predict_eval`: 2.74 -> 2.92): not on by default.
* `ped_gap` (native, 1 = off; tried at 6): a pedestrian standing in the line of the parked vehicles next to one of
  them (not behind them on the pavement) is that many times likelier about to step out: `start_cross` 0.06 ->
  0.2-0.5 while it stands. That reaches information, not caution: no said episode of any run changes.
* `entry_hold_vru_s` (alerts, off): a pedestrian's own entry holds (0.1, 0.1, 0, 0): 0.09 s earlier, two more of
  48 warned on the plain stream, one more false episode in the 24 control runs of each stream: not taken.

With `ped_carry` and `ped_onset` (the defaults since): `tools/predict_eval.py`: a crossing is recognised 1.28 /
1.82 / 1.55 s after its start where it was 1.69 / 1.91 / 1.63 s (by night 1.55 -> 1.31 s), its mean probability
0.74 / 0.70 / 0.77 -> 0.83 / 0.72 / 0.80, a pedestrian who waits or stands as before, every vehicle manoeuvre's
line as it was. The pedestrian scenarios (48 runs a stream): plain 6 -> 7 warned (two gained, 0.47 and 0.71 s
before; one earlier, 0.39 -> 0.73 s; one lost: 0.66 s before, its mass 0.21-0.26 against the minimum 0.20 for four
frames, now 0.20-0.25), realistic 3 -> 3 with two earlier (0.35 -> 0.86 s, 0.69 -> 0.95 s); the streets' and the
controls' episodes unchanged in every run; the recorded drives as they were. With `ped_cov` and `ped_gap` as well:
plain 6 -> 9, none lost. The warnings stay under a second: what is missing is the track.


**What is hidden** (`decide.json` `gaps`, off; alert `hidden_gap`, at most caution; native `decide_gaps`). Behind
the far end of a vehicle standing beside the car's way someone is not seen until past its road-side face, g metres
from the strip the car sweeps: g / v_ped seconds before being in the way. The car can still stop for that only
below v_stop = 2 a (g / v_ped - t_react); a gap is in reach while the car could neither stop before it nor be past
it in time. With the parked rows of the simulator (faces 0.8-1.4 m from the car's side) and a pedestrian at
1.5 m/s, a driver reacting in 1 s, braking at 6 m/s^2: v_stop is 0: at such a distance from parked cars no speed is
one to stop from, and the alert says so for 27-54 % of the pedestrian scenarios' time (first 6.5 s before their
collisions, median; before the pedestrian hidden between vans 6.9 s, never under 2.7 s; in the controls as much)
and once on the two recorded drives; on the streets: `parked_street` 67-74 % of the time, `curve_parked` 59-64 %,
`oncoming` 42-50 %, a junction 4-5 %, nothing where nothing is parked. As a caution it is a statement about the
street, not about a moment; it does not hold the stack's pace.

### 17.3 The futures of a driver who is driving (lane DSTART, 2026-11-28)

The tracker's lead is right at close range since `in_lead` (docs/RADAR.md 13); what the alert still said on the six
driving sessions of 26 / 27 November (re-run through the whole stack with the phone's tracker knobs and `in_lead`,
the decision layer on its defaults: 38 said `obstacle_ahead` episodes; scripts and per-episode tables in
`/home/piotr/scratch/dstart/tools`: `eps.py`, `classes.py`, `trace.py`, `lead_fut_err.py`, `ego_fut_err.py`,
`follow_data.py`) was read frame by frame against what the car and the objects then did. A re-run of one tree is
exact, and with the tracker untouched every knob below leaves the radar's stream as it was: the comparisons on the
recordings are of the same objects at the same times.

| class | base | cause in prediction / decision |
|---|---|---|
| 1 a queue starts, the lead followed right | 4 | the lead's future starts 0.9 m/s slow and its start dies away; the car's "follow" future is one of a law the driver does not keep and of a gap measured between the wrong points; a stage latched under the low-speed suppression is said when that lifts |
| 2 the car's own turn or manoeuvre among standing vehicles | 8 | the car's kinematic future runs 0.5 m inside its arc after 1 s; its lateral spread is a lane keeper's 0.3 m |
| 3 young, far, roadside, crossing, cutting in | 9 | futures of a velocity three to five sightings old; mass 0.3-0.55 entered by time |
| 4 an oncoming or standing vehicle passed on a narrow street | 5 | (not touched) |
| 5 the next vehicle of the queue, hidden beyond the lead | 2 | a standing memory behind a lead that drives off through its place is judged as a standing car |
| true or arguable (closing on a slow or stopped vehicle ahead) | 7 | |
| other (a long vehicle turning beside; a parked car the track puts in the car's line; the lead's object off) | 3 | |

**What the futures were, measured.** The lead's top future against where the same object then was (objects in the
car's lane within 25 m, moving, 3 s old at least; 3872 frames), and the car's futures against its own track (speed
and yaw rate integrated; frames above 1 m/s):

| | published (base) | with `v_track` 1.0 and `start_go` (the lead), `ego_arc` (the car) |
|---|---|---|
| lead, all: rms at 1 / 2 / 3 s; error over stated spread (rms) | 1.56 / 3.10 / 5.16 m; 1.36 / 1.78 / 2.09 | 1.31 / 2.55 / 4.36 m; 0.89 / 1.18 / 1.58 |
| lead that starts (below 6 m/s, gaining 0.5 m/s^2): its future's first speed less the track's; bias at 1 / 2 / 3 s; rms | -0.89 m/s; -1.9 / -3.9 / -6.5 m; 2.2 / 4.6 / 7.8 m | +0.27 m/s; -0.6 / -0.9 / -1.2 m; 1.2 / 2.2 / 4.0 m |
| lead that brakes (losing 0.5 m/s^2): the same | +1.08 m/s; +1.5 / +3.4 / +5.6 m; 2.2 / 4.7 / 7.6 m | +0.19 m/s; +0.8 / +2.1 / +3.7 m; 1.7 / 3.6 / 6.0 m |
| ... with `start_go` alone (the speed still the filter's): a lead that starts; one that brakes | | -0.80 m/s; -1.6 / -2.8 / -3.9 m; 2.1 / 3.7 / 5.5 m. +1.06 m/s; as the base |
| the car's `ctra` future in a turn (0.12-0.3 rad/s, below 6 m/s): lateral error, median at 1 / 2 / 3 s; mean towards the inside at 1 s | 0.49 / 0.86 / 1.65 m; +0.49 m | 0.06 / 0.32 / 0.98 m; -0.05 m |
| ... tighter than 0.3 rad/s | 0.56 / 0.72 / 1.37 m; +0.53 m | 0.13 / 0.61 / 1.37 m; -0.16 m |
| ... its `path` future (the road model's plan), the same turns | 0.15 / 0.39 / 0.88 m | (the same) |
| the exact arc of the speed and yaw rate, off line: turns below 6 m/s, median (ninth decile) at 1 / 2 / 3 s | 0.06 (0.23) / 0.38 (1.36) / 1.10 (3.49) m | |
| ... straight road (below 0.03 rad/s) | 0.00 (0.02) / 0.02 (0.13) / 0.04 (0.43) m | |

The predictor keeps a filter of its own over the track's speed (`feat_step`: one observation per `evid_tau_s`, the
track's variance times 5.7 per frame); it lags a start by 0.9 m/s and a braking by 1.1 m/s, in both directions
against the car. The car's kinematic path (`pp_base` without a centre line) is a polyline of 5 m chords each turned
by its WHOLE step from its start, at a curvature of 0.08 / m at most and straight below 1.5 m/s: 0.3 m inside the
arc after 1 m, 0.5-0.8 m after 4 m at a radius of 12 m; in a turn that future has up to 0.95 of the weight. An arc
of the car's own motion is off by a third of its lateral move (median; a tenth in tight turns), a metre after 3 s:
the decision layer's cap of 0.3 m on the car's lateral spread is a lane keeper's.

The driver behind a lead (28 380 frames with the road model's lead, 13 568 moving): the default following law (2 m
+ 1.4 s x v, 1.5 / 2 m/s^2) is 1.0-2.1 m/s^2 below what the driver does in every speed band (the driver's time gap
to the bumper: 1.1 s median, 0.7-0.9 s at the first decile). Where the car gains 0.7 m/s^2 or more below 4.5 m/s
(1132 frames: queue starts) the gap is 2.8 / 5.4 / 8.6 m (first decile / median / ninth), the lead 0.3 / 1.0 /
2.0 m/s the faster and itself gaining 0.2 / 1.0 / 1.9 m/s^2, the car 0.8 / 1.2 / 1.8 m/s^2: the law asks for
-2.3 m/s^2 at the first decile. And the "follow" future's gap was the leader's reference point (its near face)
less the car's ORIGIN less half of both lengths: 2.4 m short for a 4.5 m lead, 3.3 m for a 6.2 m one (that future
braked to a halt 3.4 m behind a lead that pulls away).

**What is built** (every knob off in the code's defaults; with them off the golden data and the three captures of
`/home/piotr/scratch/cost-rest` are as they were, `cost_bench.py diff`: IDENTICAL):

* `v_track` (predict native, m/s; `feat_step`). A future starts at the track's own speed where the track states it
  to this or better, in proportion up to 1.5 times it; the acceleration stays the filter's (the track's own
  smoothed, and what the track's speed gained beyond it since the last frame). The track is the sum of its
  sightings.
* `start_go` (predict native; `mode_build`). A vehicle in a lane that gains speed below 0.8 of the road's speed
  (the limit, else 11 m/s) goes on gaining it by the following law's free term a (1 - (v / v0)^4) scaled to its
  acceleration now, where that acceleration is two of its sigmas or 0.5 m/s^2 at least; one that stood there and
  has moved off within 3 s, at the law's own acceleration at least.
* `ego_follow` (predict native; `ego_modes`, `roll_out`; the belief is kept by `Predictor`). The car behind a
  vehicle in its lane as the follower its driver is. The gap is between the bumpers (`ego_front_m`). The gap the
  driver keeps now is one the driver keeps: the law's gap at rest and time gap are its own or the present ones,
  whichever are shorter. The leader's acceleration (its top futures' first step) is fed forward (`ego_ff`), the
  driver's own acceleration is the law's maximum up to `ego_a_max`. The car's "follow" future is its own
  acceleration carried on, no further than that law allows behind the leader's run. "The driver is following" is
  the likelihood of the acceleration measured under that law against a driver's who drives on its own (as
  before), times that of the speeds (a follower's is its leader's to `ego_dv_fol` and the leader's spread, a
  driver's own is off it by `ego_dv_free`), and is kept from frame to frame (one observation per `evid_tau_s`,
  forgotten at `ego_fol_hz`): a frame in which the leader's acceleration reads as nothing does not end it. Against
  following a frame counts at once and in full; and the belief of the frame before counts only while the car is no
  faster than its leader by more than one sigma of the leader's speed, not at all from two: a closing speed is
  judged by the present alone.
* `ego_arc` (predict native; `pp_kin`). The car's kinematic future, and the road's frame where no road is known,
  is the arc of its speed and yaw rate (half-metre steps, a radius down to `ego_arc_r_min`, also at a crawl), the
  yaw rate changing on for `ego_yaw_trend_s` as it has over the last half second (steering under way: off line
  0.07 / 0.40 / 1.15 m become 0.04 / 0.26 / 0.72 m in turns).
* `ego_turn_rel` (decide threat; `dk_ego_cov`). Of what a future moves the car aside of the line it is on now, this
  share is not known: 1 sigma across the car, on top of the lane keeper's cap. It enters the collision probability
  (not the time margin, which is of the mean futures).
* `screen` (decide threat; `decide_frame`). A vehicle (or a thing of no known kind) ahead on the car's most
  likely future is not reached before another object on that future nearer the car (existence 0.9, sighted within
  0.5 s). A future of the nearer one screens where it stays in the strip the car sweeps until the car would be at
  the further object and (a) is then still the nearer one (the car meets it first, and its own record says what
  that takes), or (b) has by then driven through the place of an object that is NOT SEEN now (the radar's `occ`
  "object", or no sighting for 0.5 s). The screened object's futures weigh 1 - P(screened) (`screen` in its
  record); refuted by (b) with 0.9, it stays refuted until it is sighted again (the ids ride in the dict of the
  levels shown, `_refuted`): the lead is then beyond it and what is left is a standing thing in the gap that
  nobody has seen there. A lead that drives into a standing vehicle in plain view screens nothing: the cut-out
  that reveals a stopped car is warned of as it was from the moment the car is seen.
* `unsaid_needs_level` (alerts). An alert not said since it left none is said only while its object's entry level
  NOW is one that is said; below that it is `suppressed: "held_unsaid"` until the level is there again or the
  holds run out. (15:25:06: imminent under the low-speed suppression 2.3 s before, latched; the lead pulled away,
  the level fell to information, the car passed 3 m/s, and the latch was said.) Holds and latches keep something
  SHOWN from flickering; this takes none of them away from an alert that was said.
* `entry_evidence_young_s` (alerts). The evidence rule of 17.1 also for an object tracked for less than this.

The simulator's driver (`sim/roaduser_sim.py` `DRIVER_SCENARIOS`, not in the lists the other tools walk):
`queue_start` (the car 2.3-3.1 m behind a lead that pulls away at 1.5-2 m/s^2, starting a tenth of a second after
it at 0.85-1 of its acceleration: at 3 m/s 3.0-4.6 m behind its bumper and gaining 1.3-2 m/s^2; the next vehicle
of the queue ahead of the lead, hidden), `queue_start_stall` (the same, and the lead brakes to a stop again 1.6-
2.2 s after it began while the car goes on: a collision), `turn_parked` (4.5 m/s round a 10 m corner between rows
of parked cars, 0.5 m and more from every body), `narrow_oncoming` (7 m/s, parked cars on the right and oncoming
cars 0.6-1 m from the car's side). `queue_creep` (section 12) is the driver who starts a second after the lead.
Tests: `tests/test_driver_futures.py` (every knob on fixtures; the scenarios' driver).

**Measured: the recordings** (base against all the knobs but `v_track`, and against the set proposed below;
matched by the alert's object within 4 s):

| | base | all but `v_track` (no evidence rule) | proposed |
|---|---|---|---|
| 1 queue start | 4 (15:25:06, 15:30:23, 15:59:08; 27 Nov 20:30:54) | 1: 20:30:54 is said 2.4 s later, on the lead's own "follow" future | 0 |
| 2 own turn | 8 | 8 | 7 (15:30:33 gone) |
| 3 young / far / roadside | 9 | 8 (15:27:13 gone) | 7 (15:27:13 gone; by the evidence rule 15:25:11 gone and 15:39:19 said 0.57 s later) |
| 4 narrow street | 5 | 5 | 5 |
| 5 hidden beyond the lead | 2 (15:44:31, 15:50:37) | 0 on the hidden object (15:44:31 is said 1.6 s later on the car merging from the right) | 0 (1.4 s later, the same) |
| true or arguable | 7 | 7 | 7 |
| other | 3 | 3 | 3 |
| new | | 4: two closings on a lead that brakes to a stop (15:42:51: 7.9 m, 3.6 m/s, the driver braking 1.9 m/s^2; 16:00:50: 10 m, 3.9 m/s, 1.2 m/s^2), two false (15:30:29 a car 3.6 m beside; 20:33:26 a standing vehicle in a turn) | 6: four such closings (15:42:51, 15:50:09, 16:00:07, 16:00:50: 7.8-11.9 m, 2.8-5.2 m/s, the driver braking 0.5-2.6 m/s^2), the same two false |
| said | 38 | 37 | 36 |

Of the base's 31 that are not true or arguable, 23 are still said with the proposed set and two new ones are; four
are added of the kind the last lane counted as true or arguable (closing on the lead with less than 4 s to its
bumper). By knob, alone on the recordings: `start_go` + `ego_follow` take 15:25:06 and 15:30:23 (15:59:08 comes
0.6 s later); `unsaid_needs_level` 15:25:06; `screen` 15:44:31 (15:50:37 needs the lead's future to drive off as
well); `ego_arc` + `ego_turn_rel` none (15:30:33 comes 0.5 s later, and two episodes are new).

The seven true or arguable ones, the time first said against the base's (the base's time to contact and time
margin at that moment), without / with `v_track`: 26 Nov 11:47:46 (1.0 s, -1.2 s) +0.00 / +0.00 s; 11:53:17
(2.3 s, 0.3 s) -2.31 / -2.31 s (earlier: the car 5 m/s faster than a lead that slows is no follower, whatever the
braking); 15:30:51 (3.05 s, 1.4 s) +0.00 / -0.21 s; 15:41:18 (1.85 s, -0.2 s) +0.00 / +0.00 s; 15:41:33 (2.05 s,
-0.1 s) +0.00 / +0.00 s; 15:46:08 (2.45 s, 0.2 s) +0.00 / -0.10 s; 27 Nov 20:30:12 (3.25 s, 0.9 s) +0.00 /
-0.82 s. None is later.

What is left of class 2, read frame by frame with the knobs on (the car's arc now agrees with its track to 0.1 m
at 1 s): the car parks (11:39:15: it turns towards a parked car 9 m ahead and stops within 3 s at -0.3 m/s^2, which
no future knows); a slow vehicle ahead in a junction closed on at 3 m/s (11:41:16: arguable); near objects the
track puts in the car's line while they stand in the next lane or beside it (15:24:17 a van 5 m ahead at y +0.35
m; 11:53:50 a track 1.1 s old). On the recordings the turn's knobs make the car's future right and take no
episode by themselves. Class 4 is as it was.

**Measured: the simulator** (`tools/alert_truth.py`, the phone's tracker knobs with `in_lead` as the base, the road
model's lead as measured (`+lead`); six streets and the pedestrian controls x seeds 1-6, the scripted conflicts x
seeds 1-3, day and night; paired over the same runs, +- the bootstrap's 1 sigma):

| knobs added | `detin_pose_scale+lead`: false episodes; conflicts lost / gained / later / earlier | `scale+lead` |
|---|---|---|
| `start_go` | 82 -> 82; 0 / 0 / 0 / 0 | 71 -> 70 (-1 +- 1.0); 0 / 0 / 0 / 0 |
| `ego_follow` | 82 -> 81 (-1 +- 1.0); 0 / 0 / 0 / 0 | 71 -> 71; 0 / 0 / 1 (1.55 -> 1.46 s) / 0 |
| `ego_arc` | 82 -> 84 (+2 +- 1.3); 0 / 0 / 0 / 3 (1.48 -> 3.25, 1.55 -> 2.42, 2.24 -> 2.40 s) | 71 -> 73 (+2 +- 1.3); 0 / 0 / 0 / 1 (1.71 -> 2.51 s) |
| `ego_turn_rel` 0.35 | 82 -> 79 (-3 +- 2.5); 0 / 0 / 1 (3.21 -> 3.14 s) / 0 | 71 -> 67 (-4 +- 1.6); 0 / 0 / 1 (3.48 -> 3.40 s) / 0 |
| `screen` | 82 -> 82; 0 / 0 / 0 / 0 | 71 -> 71; 0 / 0 / 0 / 0 |
| `unsaid_needs_level` | 82 -> 82; 0 / 0 / 0 / 0 | 71 -> 68 (-3 +- 2.1); 0 / 0 / 0 / 0 |
| `entry_evidence_young_s` 2 | 82 -> 81 (-1 +- 1.0); 0 / 0 / 0 / 5 (by 0.08-0.24 s) | 71 -> 70 (-1 +- 1.0); 0 / 0 / 0 / 4 (by 0.08-1.09 s) |
| `v_track` 1.0 | 82 -> 93 (+11 +- 3.7): `slow_lead` 7 -> 16; 0 / 0 / 0 / 0 | 71 -> 76 (+5 +- 2.2): `slow_lead` 6 -> 11; 0 / 0 / 0 / 1 |
| all but `v_track` and the evidence rule | 82 -> 78 (-4 +- 3.6); 42 -> 42 warned; 0 / 0 / 1 (3.21 -> 3.14 s) / 3 | 71 -> 65 (-6 +- 3.3); 40 -> 40; 0 / 0 / 2 (3.48 -> 3.40, 1.55 -> 1.46 s) / 1 |
| ... with `v_track` 1.0 (the proposed set less the evidence rule) | 82 -> 86 (+4 +- 4.6); 42 -> 42; 0 / 0 / 1 / 3 | 71 -> 67 (-4 +- 4.1); 40 -> 40; 0 / 0 / 3 (4.12 -> 3.41, 3.48 -> 3.40, 1.55 -> 1.46 s) / 2 |

`queue_creep` + `left_behind` (the reach modelled, seeds 1-6): 8 -> 5 false (-3 +- 2.5), `stopped_cruise_14` 4 of 6
warned at the same frames, with or without `v_track`. The driver scenarios, the reach modelled, seeds 1-6, day
and night (48 runs; all but `v_track`):

| | base | knobs |
|---|---|---|
| `queue_start` false episodes (12 runs) | 12 | 2 |
| `turn_parked` (12 runs) | 5 | 3 |
| `narrow_oncoming` (12 runs) | 13 | 14 |
| all three | 30 | 19 (-11 +- 1.0) |
| `queue_start_stall`: warned; first said before contact, median (least) | 12 of 12; 1.19 s (0.56 s) | 12 of 12; 0.56 s (0.43 s); with `v_track` 0.60-0.71 s (0.49 s) |

Every start was said by the base (12 of 12): its warning of the lead that stalls is the one it gives of every lead
that starts, said as the car passes 3 m/s: in six of the twelve runs within a quarter of a second of the lead's
first braking (which is 1.41-1.54 s before contact), once 0.07 s after it. With the knobs the stall is said once the
lead's braking is in its track and the car is the faster: 0.43-0.87 s before contact (once 1.29 s), 0.55-1.1 s into
the braking. That is the price of class 1, stated: a lead that stalls within 2 s of a start is warned of 0.6 s later
(median). On the plain stream (`scale+lead`) `queue_start`
is never said and the stall never warned of, with or without the knobs (not looked into); `turn_parked` 9 -> 9,
`narrow_oncoming` 19 -> 18.

`v_track`'s false episodes are `slow_lead`: the car follows a lead at 7 m/s 4 m behind it (the simulator's driver
there keeps 0.55 s), and the road model's speed of the lead is off by up to 1 m/s for seconds with a stated spread
of a third of that (docs/RADAR.md 13: the same episodes came with `in_lead` itself). The predictor's second filter
hid half of them; the honest speed shows them.

**Cost** (`tools/cost_bench.py run`, the three captures of `/home/piotr/scratch/cost-rest`, three runs each on the
P52, medians; base against the proposed knobs; with `v_track` the same within 0.01 ms): `bench` 1.226 -> 1.212 ms per
road frame (predict 0.048 -> 0.048, decide 0.042 -> 0.041); `sim1` 3.71 -> 3.58 (predict 0.514 -> 0.510, decide
0.337 -> 0.314); `sim2` 3.65 -> 3.61 (predict 0.509 -> 0.522, decide 0.311 -> 0.280); the 99th centile of a road
frame 4.64 -> 4.34 and 4.43 -> 3.97 ms. Nothing added that a run-to-run difference of 0.05 ms shows: the predictor
does a few multiplications more per frame and one arc of 100 points where a future of the car is kinematic, the
decision layer projects every object once on the car's most likely future and judges a screened object's futures
at no weight (the dense scenes' decide cost falls by 7-10 %). Not measured on the phone.

**Proposed for the car** (the alert stays in shadow). `/etc/chris/predict.json`: `{"native": {"v_track": 1.0,
"start_go": 1, "ego_follow": 1, "ego_arc": 1}}`; `/etc/chris/decide.json`: `{"threat": {"ego_turn_rel": 0.35,
"screen": true}, "alerts": {"obstacle_ahead": {"unsaid_needs_level": true, "entry_evidence_young_s": 2.0}}}`.

`v_track` is the one with a measured cost, and the choice is stated rather than hidden. For it: only with it is
the lead's future right (the table above: a braking lead's future starts 1.1 m/s fast without it, against the
car); it takes the last queue start; three of the seven true warnings come 0.1-0.8 s earlier; the four episodes
it adds on the recordings are closings on a lead that brakes to a stop, which the lagging speed hid. Against it:
the simulator's `slow_lead` (+11 +- 3.7 and +5 +- 2.2 false episodes), a lead followed 4 m behind at 7 m/s whose
speed the tracker has wrong by 1 m/s for seconds with a stated spread of a third of that. The drives have no
such episode in 43 minutes of following (the driver's time gap is 1.1 s, 0.7-0.9 s at the first decile, where
the simulator's is 0.55 s), and the error is the lead sensor's (docs/RADAR.md 13: `lead_v_k`, `lead_tau_v`). If
the shadow log shows cautions in steady following, `v_track` is the knob to take out: without it the set is
-4 +- 3.6 and -6 +- 3.3 false episodes in the simulator and one queue start more on the recordings.

Not solved, and why:

* **Class 4**, the oncoming or standing vehicle passed on a narrow street (5 of 5; `narrow_oncoming` 13 of 12
  runs): the driver passes 0.6-1 m from a body whose place the track knows to 0.3-0.7 m. What is right is the same
  interaction model across the car as `ego_follow` is along it (a driver who passes is steering by the thing it
  passes: the car's lateral place is not independent of the object's); not built.
* **Class 2** beyond the arc: a stop the driver is about to make (parking), and the track's place of near objects
  in a turn.
* **Class 3**: the futures of a track three to five sightings old are built on its velocity as stated (2.5-3 m/s
  for a mean of 5-14 m/s); the evidence rule takes the marginal ones (mass under 0.6), a turn or a crossing given
  0.5 of the weight on the first second's lateral velocity is said as before.
* **The other followers.** The law of the car's own following is still the default one for every OBJECT that
  follows: a lead 6.7 m behind its own leader at 7 m/s gets a "follow" future that brakes at 4 m/s^2 in its first
  quarter second (27 Nov 20:30:56, the one queue start left), and the car's leader profile is the least far of the
  lead's weighty futures.
* **A stall within a start** is said half a second later (above).
* The driver's following law is one driver's on six sessions; nothing is learnt per drive. `ego_turn_rel` and the
  arc's numbers are of the same sessions (1 h 17 min, 1100 frames in turns).

## 18. Learnt values across drives (lane DECIDE, 2026-11-23)

`chris_integrator/learnt.py` (the store) and `chris_integrator/learners.py` (the learners' adapters and the plug-in
`learnt`; table of what is kept and why the rest is not). One versioned file, `<data dir>/learnt.json` (on the
phone `/var/lib/chris/learnt.json`, the unit's `StateDirectory=chris`; `learnt.json` in the configuration
directory may set `path`), per value: value, variance (or counts), samples, last update (wall clock), the
conditions it was learnt under. On start a value is a prior, not the truth: its variance grows by what a restart
can change and by its own walk over its age, never beyond the spread the learner starts with; counts are halved
per 30 days; a limit relaxes; a record outside its sanity bounds, too old, or learnt under another calibration is
not used (health says why). Vision calibration states are given under the last drive's calibration and undone
(`reset_aux`) if the camera's calibration message says another one is in use.

Power may go at any moment (the phone has no battery): writes are write-temp, fsync, rename (the previous file
kept), fsync of the directory; reads try the file, the temp, the previous one, each checked by format, version
and a checksum over the values; a damaged file is moved aside (`.bad`). Writes are rare: when something changed
and 10 minutes passed, at a standstill after driving (10 s still after 60 s of driving: the power may go next),
at shutdown; at most 12 an hour; about 4 kB each. The old `integrator.json` (faults and calibration) is still
written and read as before; where both hold a calibration state the learnt store's prior wins (it is applied
after).
