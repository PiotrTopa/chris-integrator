# Ego-motion hygiene

What every other road user's motion is measured against is the car's own. A small error of it shows as motion
of what stands: parked cars sliding sideways in a turn, a closing speed that is not there. This note is about
three places where the estimator ([DESIGN.md](DESIGN.md) sections 2-4) was wrong or costly, what the recordings
say, what was built, and the knobs. All knobs are keys of `ego_motion` in the estimator's configuration
(`integrator.json`); the numerics are the native core's (`native/src/egomotion.c`: `Stand`, `Lag`, the rows'
measurement functions).

Tools: `tools/ego_hygiene.py lags | bias SESSION...` (the numbers of sections 1 and 2 on a recorded drive),
`python3 -m chris_integrator.sim.radar_score NAME --ego estimator --phantom` (the velocity the tracker gives to
standing actors), `tools/cost_bench.py run ... --estimator JSON` (cost and output with a configuration).
Tests: `tests/test_ego_motion.py`.

## 1. The standstill, and the gyro's bias

**What the estimator did.** The gyro's bias is a state (`vehicle.b_g`, prior 0.01 rad/s, random walk 1e-9
(rad/s)^2/s; its value is kept across drives in the learnt store, with 0.001 rad/s added at a restart). It is
taught at a standstill: the feed's flag (speed and every wheel read zero) became four exact rows, speed,
acceleration, yaw rate and yaw acceleration = 0, and the gyro's own row then reads its bias. The accelerometers'
offsets (`b_ax`, `b_ay`) are states the same way. So bias, zero-velocity update and random walk were there. What
was wrong is what the flag means.

**What the recordings say** (`tools/ego_hygiene.py lags`; the two drives of 2026-11-22, the first recorded with
the gyro as read; the drives of 2026-11-19 came through a relay that sent 0 at standstill):

* The speed signal has a floor: its lowest value while moving is 0.206 m/s on every drive. Below it the speed and
  the wheels read zero, the feed says "standstill", and the car still rolls and turns.
* Under that flag the gyro read up to 0.081 and 0.136 rad/s (p90 0.05 / 0.02 in the first half second, 0.03 /
  0.09 from 0.5 to 2 s in), and the camera's yaw rate up to 0.125.
* The exact rows held the yaw rate at 0.0000 through those turns and taught them to the bias: replayed through
  the estimator (`tools/ego_hygiene.py bias`), the bias state went from -0.005 to +0.014 rad/s within one drive
  while its stated sigma was 0.0003. The truth is what the gyro reads at a quiet standstill (five of them, -0.0005
  to +0.0020 rad/s).
* The same flag made the camera source find the camera "handled" (its pose moves while the car "stands") and
  open the mount's pitch: 1 and 8 times on the two drives.
* The gyro steps by 0.0022 rad/s (two channels of 0.0044 averaged); the accelerometers step by 0.04-0.10 m/s2 at
  rest and 0.2-0.5 moving. A camera at night reports speeds of a standing car (median 0.14 m/s, p90 0.7 on one
  drive), and its yaw rate reads no turn under the floor while the gyro reads 0.02 rad/s.

**What was built.** `Stand` (native): the bus's flag is *claimed* until a witness confirms it.

| witness | says |
|---|---|
| inertial, over the last 1 s of bus samples | *against*: the gyro's span (max - min) beyond 3 quiet spans (0.0095 rad/s), or its mean beyond 0.03 rad/s. *for*: gyro and accelerometers within their quiet spans and the gyro's mean where its bias is believed to be (3 sigma of the state, at least 0.006). *offset*: quiet, but the mean is elsewhere: a slow steady turn, or a bias that has changed. *unsettled*: the gyro neither quiet nor clearly moving. The accelerometers never speak against. |
| camera, the last poses within 0.6 s | *for*: medians of speed and yaw rate under 0.15 m/s and 0.005 rad/s. *turning*: yaw rate over 0.02. Its speed never speaks against. |

Confirmed once the flag has held 0.5 s and: inertial for and the camera not turning; inertial offset after the gyro
has been quiet about one mean for 10 s (a turn that fades is not that; then the bias has changed: where the gyro's
mean at a confirmed standstill is 0.006 rad/s and 4 sigma from the state, the state's uncertainty is opened to the
difference, the standstill places it within a second, and an event says so); inertial silent and the camera for;
no witness speaking either way after 2 s (a feed without inertial signals and no camera).

* Confirmed: the four exact rows as before.
* Claimed only: one row, speed = 0 to the sensor's floor (sigma 0.12 m/s). Nothing is said about the turn or the
  accelerations, so nothing is taught to a bias, and the yaw rate follows the gyro.
* The camera is "handled" only at a confirmed standstill.
* Published: `calibration.vehicle.stand` (`state`, `evidence`, `confirmations`, `claimed_only_share`,
  `bias_steps`) and `calibration.vehicle.gyro_bias`: `observed_by` (`standstill`: the gyro reads its bias now;
  `observed`: its uncertainty is still falling through the other yaw sensors; `held`: carried with its random
  walk), `straight`, `since_standstill_s`, `std_at_standstill`, `std_trend_per_min`. While the car drives straight
  nothing measures the bias directly: the camera's yaw rows never correct it (a consider state of theirs), the
  rear wheels' rows may (`gyro_bias_wheels`), and the stated sigma with its trend is what there is to report.

**Evidence, recordings** (two drives, five quiet standstills; `tools/ego_hygiene.py bias`):

| | flag alone (`stand_confirm` false) | confirmed (default) |
|---|---|---|
| bias state on arriving at a quiet standstill, rms / max | 0.0039 / 0.0083 rad/s | 0.0009 / 0.0018 |
| ... in sigma of its stated uncertainty, rms | 5.5 | 1.0 |
| bias state while moving, against the standstills' truth interpolated, rms / max | 0.0023 / 0.0143 rad/s | 0.0009 / 0.0046 |
| yaw-rate state under the flag, max (the gyro there: 0.081, 0.136) | 0.0000, 0.0002 rad/s | 0.076, 0.140 |
| camera "handled" | 1, 8 | 0, 6 |
| flag samples confirmed / claimed only | all / - | 82 % / 18 %, 80 % / 20 % |

(The remaining six "handled" are a night camera reporting motion at confirmed standstills: not this note's.)

**Evidence, simulation** (`sim.VehicleSim(speed_floor=0.21, gyro_raw=True, gyro_lsb=0.0044, gyro_bias=0.004)`; a
parking manoeuvre: 18 s of creeping at 0.15 m/s through a turn of 0.03-0.04 rad/s, then a real standstill;
`tests/test_ego_motion.py StandDriveTest`):

| | flag alone | confirmed |
|---|---|---|
| bias state at the end of the creeping turn (injected 0.0040) | 0.0302 rad/s, stated +- 0.0004 | 0.0040 |
| heading through the turn (truth 1.054 rad) | 0.774 rad missing | 0.062 rad off |
| bias state after the standstill that follows | 0.0164 | 0.0041 |

A bias that has changed (`BiasStepTest`: a state of 0 +- 0.0005 rad/s carried over, a gyro that reads 0.020 at
rest): claimed for the first 10 s of the rest, then confirmed, and 0.0195 half a second later.

On the tracker (`radar_score park_out_curve --ego estimator --phantom`: the car leaves a parking place through
such a turn, then a town road with two curves and parked cars; the velocity over ground given to standing actors,
lateral rms / p90 and along the car's axis rms / p90, m/s; seed 1, camera boxes only):

| | lateral | along |
|---|---|---|
| flag alone | 2.09 / 3.91 | 4.68 / 8.10 |
| confirmed | 0.47 / 0.46 | 2.91 / 4.02 |
| a feed without the floor and the bias (what is left is the tracker's own) | 0.43 / 0.37 | 2.61 / 3.62 |

**Knobs.**

| key | default | recommended | |
|---|---|---|---|
| `stand_confirm` | true | on | the standstill confirmed; false: the feed's flag alone, as before |
| `stand_dwell_s`, `stand_alone_dwell_s`, `stand_offset_dwell_s` | 0.5, 2, 10 | | how long the flag (the gyro's quiet) must have held |
| `stand_window_s`, `stand_gyro_span`, `stand_gyro_max`, `stand_mean_tol`, `stand_accel_span`, `stand_move_k` | 1, 0.0095, 0.03, 0.006, 0.25, 3 | | the inertial witness; the spans are the reference car's |
| `stand_camera_speed`, `stand_camera_yaw`, `stand_camera_move_yaw`, `stand_camera_fresh_s` | 0.15, 0.005, 0.02, 0.6 | | the camera witness |
| `stand_floor_sigma` | 0.12 m/s | | the speed under a claimed flag |
| `stand_step_tol` | 0.006 rad/s | leave | a bias that has changed. 0.003 fired six times on the two recorded drives and made the worst error inside a standstill 0.0043 rad/s (0.0018) |
| `gyro_bias_psd` | none (1e-9) | leave | the bias's random walk. 1e-8 and 1e-7 made the recorded errors larger (arriving rms 0.0014, 0.0017): a freer bias follows the other yaw sensors' errors |
| `gyro_bias_wheels` | true | leave | may the rear wheels' rows correct the bias. False made no difference on the recordings (arriving rms 0.0008, moving 0.0009) |

Limits. A steady turn under the floor slower than the gyro can tell from its bias (0.006 rad/s) is confirmed as a
standstill; one between 0.006 and 0.03 rad/s that stays within 0.0095 rad/s for 10 s is too. The evidence is two
drives of one evening and five standstills. Without a standstill the bias is not measured; a turn-on change is
covered by the learnt store's 0.001 rad/s and, beyond that, by the "offset" case at the first standstill.

## 2. Time offsets

**What the stamps mean** (from the code and the recordings):

| source | stamp | offset against the bus speed's stamps |
|---|---|---|
| bus (`ego`: speed, wheels, gyro, accelerometers) | the speed frame's arrival at the car backend. The message is built when sent: wheels, gyro and accelerometers are the newest frames within 0.3 s of the speed frame, under its stamp | the filter's time axis, by definition (the feed's messages reach the integrator 2 ms after their stamp, p95 26 ms) |
| camera pose (`road`) | the frame; the pose describes the image pair (206 ms apart) | a filter state, `vision.tau` (prior 0.125 s), learnt from the speed rows; the pose's yaw rows share it |
| receiver (`gnss`) | arrival (5 ms after the stamp) | a filter state, `gnss.tau` (prior 2 s), plus a row of its own from the receiver's motion |
| radar (`objects`) | the scan | a filter state per radar |

So the camera's and the receiver's delays were already estimated, as states of the filter, and kept across drives.
What had no offset at all: the gyro against the speed, the wheels' yaw rate against the gyro, the camera's yaw rate
against the gyro (it rides on the speed's delay).

Measured, raw against raw (`tools/ego_hygiene.py lags`; per drive the median over 60 s windows with excitation,
and the fit over the whole drive):

| | 11-22 a | 11-22 b | 11-19 a | 11-19 b |
|---|---|---|---|---|
| camera yaw rate behind the gyro | 0.142 / 0.104 s | 0.114 / 0.117 | 0.087 / 0.087 | 0.103 / 0.107 |
| camera speed behind the bus speed | 0.188 / 0.279 | -0.034 / 0.054 | 0.146 / 0.229 | 0.038 / - |
| rear wheels' yaw behind the gyro | 0.036 / -0.045 | 0.048 / -0.049 | 0.048 / 0.049 | 0.040 / 0.052 |
| steering's yaw behind the gyro (the car's response, not a stamp) | -0.079 | -0.077 | -0.063 | -0.078 |
| accelerometer behind d(speed)/dt | 0.061 / 0.022 | - / 0.022 | 0.031 / 0.023 | 0.021 / 0.002 |

The camera's yaw rate lags the gyro by 0.09-0.14 s, about the image pair's 0.125; the speed pair is too noisy
this way to say more than 0.04-0.19 s; the wheels' and the accelerometer's are within 0.05 s of zero with either
sign. Nothing on this car is resolved as an offset that needs correcting.

**Augment the filter, or estimate outside it?** Outside, for the offsets added here:

* The filter's own delay states are learnt through the Jacobian `-H x rate of the state`. For a yaw rate that rate
  is the yaw acceleration, which the filter knows poorly at the row's time; DESIGN.md records that a noisy
  regressor biases what it teaches, which is why the camera's delay is learnt from its speed rows only. Outside
  the filter the regressor can be taken from the reference sensor's own samples on *both* sides of the row.
* An offset is a property of how a source stamps, changing over days if at all. It needs excitation gating, robust
  weights, bounds, a fallback and a place in the learnt store: all simple for a scalar regression, none natural
  for a filter state that every row touches. And each state is paid for in every row of every pass.
* The source that holds a quantity cannot see its own offset (the state follows it). What is observable is one
  source against another, which is exactly what a regression of one source's innovations on the other's rate
  measures.

`vision.tau` and `gnss.tau` stay what they are.

**What was built.** `Lag` (native), one per offset a source declares (`Source.lags`): a row stamped t whose
quantity is `o` s late, inserted as if `applied` s late, has the innovation `-(o - applied) x rate + noise`. The
rate is the slope of the reference signal's samples within +-0.25 s of the row (a row waits for samples on both
sides). A row counts where that slope stands 3 sigma clear of its own noise and above a floor (0.05 rad/s2 for a
yaw rate, 0.3 m/s2 for a speed): turning in or out, accelerating. The estimate is the posterior of a Gaussian
prior (the configured value, sigma 0.05 s) and the weighted rows (Huber's weight from 2.5 sigma, falling off as
1 / u^2 beyond 7.5; the variance x 4 for rows correlated in time); the rows' information halves over 300 s of excitation, so without
excitation the estimate stays, and with none ever it is the configured value. Clamped (+-0.3 s; 0-0.4 for the
camera). Kept in the learnt store (`estimator.offset.<name>`: 0.005 s added at a restart, 0.01 s a day; a record
out of bounds is not used), published in `calibration.time_offsets`.

| offset | of | against | prior |
|---|---|---|---|
| `vision.yaw` | the camera's yaw rate | the gyro | 0.125 s |
| `wheels.yaw` | the rear wheels' yaw rate | the gyro | 0 |
| `vision.speed` | the camera's speed | the bus speed | 0.125 s: a witness of `vision.tau`, never applied |

Configured (`time_offsets`): `vehicle` (every row of the bus this late: the rows' time), `vehicle.yaw` (the gyro's
value this old at its stamp), `wheels.yaw`, `vision.yaw`.

Applying (`offsets_apply`) does not move a row in time. A source whose value is `d` s old where its row sits has
the model `(1 + scale) (w - d x yaw acceleration)`: the yaw rate as it was then, by its rate state. No entry is
added to the filter's buffer and no row arrives "late" because of it. With `gyro_to_camera` the gyro goes onto the
camera's clock: its delay is the image pair's lag as configured less `vision.yaw`, and the camera's and the
wheels' yaw rows follow.

Not observable, and said so: the bus as a whole against the camera's clock. `vision.tau` is the camera's delay less
the bus's lateness; only their difference shows. On the recordings `vision.tau` settles at 0.146-0.160 s against a
pair lag of 0.125-0.14: the bus is not late by more than about 0.03 s.

**Evidence, recordings** (the two drives of 11-22, each from the configured values):

| | drive a | drive b |
|---|---|---|
| `vision.yaw` | 0.132 +- 0.010 s (996 rows, 96 s of excitation) | 0.117 +- 0.013 |
| `wheels.yaw` | +0.003 +- 0.015 | +0.011 +- 0.023 |
| `vision.speed` (the filter's `vision.tau`: 0.160, 0.146 +- 0.03) | 0.142 +- 0.024 | 0.129 +- 0.032 |

The gyro on the camera's clock: 0.125 - 0.132 = -0.007 and +0.008 s, zero within 0.013.

**Evidence, simulation** (`sim.VehicleSim(gyro_lag=0.06)`: the gyro's value is 60 ms old at its stamp; a town drive
of 240 s with a turn every 20 s; `tests/test_ego_motion.py OffsetDriveTest`):

| | clean | gyro 60 ms late | ... applied (`offsets_apply`, `gyro_to_camera`) |
|---|---|---|---|
| `vision.yaw` (truth 0.125, 0.065) | 0.1228 | 0.0706 | |
| `wheels.yaw` (truth 0, -0.060) | 0.0004 | -0.0534 | |
| the gyro's delay as applied (truth 0.060) | | | 0.0490 (four minutes; the prior still pulls) |
| yaw rate at the camera frames against the truth, rms | 0.0024 rad/s | 0.0059 | 0.0032 |
| ... where the yaw rate changes | 0.0024 | 0.0118 | 0.0041 |

On the tracker (`radar_score curve_parked --ego estimator --phantom`, seed 1; lateral velocity given to standing
actors while the car's yaw rate changes, rms / p90, about 210 samples): clean 0.37 / 0.48 m/s; gyro 60 ms late
0.50 / 0.85; applied, learning within the 50 s run, 0.36 / 0.40; with the offset configured 0.40 / 0.53. Over the
whole run the numbers do not differ (0.59-0.68 rms): a 60 ms offset of the gyro is a small effect, at the ends
of turns only.

**Knobs.**

| key | default | recommended | |
|---|---|---|---|
| `time_offsets` | `{}` | leave | configured offsets |
| `offsets_estimate` | true | on | the estimators run and are published; cost within the measurement's noise |
| `offsets_apply` | false | off on this car | nothing here is resolved as away from its configured value, and the applied model adds the yaw acceleration's noise to the yaw rate on straights (0.0026 -> 0.0029 rad/s rms in simulation). On once an estimate stands 3 sigma away from its configured value |
| `gyro_to_camera` | false | with `offsets_apply`, where the image pair's lag is trusted | |
| `offset_sigma_s`, `offset_half_life_s`, `offset_span_s`, `offset_clear`, `offset_huber`, `offset_corr` | 0.05, 300, 0.25, 3, 2.5, 4 | | the estimator |

## 3. Late data

**What the filter did.** A late row is inserted at its time and the pass is run again from there to the newest
entry, once per message (not from the start, and not once per row): a fixed-lag buffer of 10 s with
re-propagation from the late stamp was already what there is. `ekf_reruns` counts the entries passed again.

**Where the 10,214 come from** (the dense simulated capture, 1431 road frames): the receiver's rows, 2 s old
(5091: 50 entries per fix); the camera's, 0.19 s old (4764: 3.3 per frame); the radar's ego rows (359). On a
recorded drive (4771 frames): 14,403, nearly all the camera's.

**What it costs** (this laptop; the estimator plug-in's own time per road frame, timers around its parts):

| | dense simulation | recorded drive |
|---|---|---|
| estimator | 8.0-8.6 ms | 2.6-2.7 ms |
| the object layer (`core/objects.py`) | 4.8-5.1 ms, 59-60 % | 0.28 ms, 10 % |
| inputs: sources' measurement models, monitors, isolation, and the filter's passes | 2.3-2.5 ms, 29 % | 1.1 ms, 42 % |
| the `ego_state` message (smoother, tables) | 0.8-0.9 ms, 10 % | 1.1 ms, 42 % |
| of the inputs: all filter passes | 0.61-0.64 ms, 7.5 % | 0.36 ms, 13 % |
| ... of which passed again for late rows | 0.41-0.44 ms, 5 % | 0.20 ms, 7.5 % |

Of the phone's 37 ms per frame in the dense scene, in these proportions: about 22 ms the object layer, 2 ms the
filter passing entries again, 13 ms the rest. An entry passed again cost 40-60 us here, most of it not arithmetic:
each row's measurement function was a Python closure called from the pass (4.9 of 11 us per row), the rest
attribute traffic between the pass and the row objects.

**What was built.** The rows' measurement functions in the native core. A row's model is `(kind, indices,
parameters)` (`core/registry.py`: `lin`, `scaled`, `accel_grade`, `product_bias`, `delayed_speed`,
`delayed_heading`), evaluated inside the pass: prediction and Jacobian row from the pass's own state, so a pass
run again for a late row does not enter Python for them. The Python closures of the vehicle, vision, receiver and
map sources are gone (one implementation); a callable is still taken for a model the core does not have (the
radar's ego rows, tests).

**Equality.** `cost_bench run ... --dump` of this tree with the new behaviour of sections 1 and 2 switched off,
against the base (63cdb71): every message of every stream identical, byte for byte (tolerance 0), on two dense
simulated captures (13 streams, 1433 `ego_state` each) and on two recorded drives (5448 and 4024 `tracks`).
`tests/test_ego_motion.py RowModelTest` holds each kind against its expression to the last bit.

With the defaults, against the base: on the simulated captures (no standstill under a floor) only `ego_state`
differs, by the entries added to its calibration table, no number; on the recorded drives the car's motion under
a claimed flag differs and what follows from it (587 of 5448 and 45 of 4024 `tracks` messages, the view, a speed
of 0.2-0.7 km/h where it was 0.0).

**Cost after** (same harness):

| | dense simulation | recorded drive |
|---|---|---|
| all filter passes | 0.64 -> 0.47 ms | 0.36 -> 0.26 ms |
| ... passed again for late rows | 0.44 -> 0.32 ms | 0.20 -> 0.14 ms |
| `ekf_reruns` | 10,214 -> 10,214 | 14,403 -> 14,403 |
| all filter passes, share of the estimator (stable from run to run; the absolute times were not, on a loaded machine) | 7.4 % -> 5.4-5.7 % | 13.5 % -> 10.0-10.2 % |
| ... passed again for late rows, share | 5.0 % -> 3.6-3.8 % | 7.5 % -> 5.5-5.7 % |
| estimator | 8.6 -> 8.3 ms (new behaviour off), 8.7 (defaults); run-to-run +-0.3 | 2.67 -> 2.60, 2.63 ms |

The standstill detector, the offset estimators and their rows' callbacks cost nothing that shows: with the defaults
the shares are those of the row above, and the estimator's time is inside the run-to-run spread.

What this is and is not: the passes are 27 % cheaper, the count of entries passed again is what it was, and the
estimator as a whole moves by 1-3 %, inside the run-to-run spread in the dense scene. An update of an old row
without the pass (the one-step out-of-sequence algorithms) was not built: it would not be equal to the present
result (rows are re-linearised and robustly re-weighted in the pass), and the item it removes is 5-7 % of the
estimator. The estimator's large items are the object layer in a dense scene and the `ego_state` message on an
ordinary drive.
