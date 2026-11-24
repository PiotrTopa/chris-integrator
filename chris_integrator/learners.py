"""The daemon's learnt values wired to the store (learnt.py): one small adapter per learner, and the plug-in
`learnt` that keeps them (takes their values in, writes the store seldom, hands priors back on start and after a
new drive restarts a learner).

What is learnt while driving, and what of it is kept (lane DECIDE, 2026-11-23):

| key | learner | kept as | valid under | prior on start |
|---|---|---|---|---|
| `estimator.<source>.<state>` | the estimator's calibration states that walk (core/ego.py AuxState without `tau_s`: vision delay `tau`, scale `s`, yaw scale `s_w`, mount `dyaw`, `dpitch`; vehicle gyro bias `b_g`, scale `s_g`, wheel-yaw scale `s_wy`, accelerometer offsets `b_ax`, `b_ay`; the receiver's and the radars' and camera objects' delays `tau`; radar mount `dyaw`) and the odometry speed scale `sv` | value, variance | `vision.*`: the camera's calibration in use | the variance + (0.1 sigma0)^2 + the state's own walk over the age, at most sigma0^2 (never surer than the learner would start) |
| `radar.pitch_bias.<sensor>` | the radar's camera pitch error (radar/tracker.py `bias`) | value, variance | the calibration in use | the variance + (0.15 deg)^2 + (0.05 deg)^2 per day, at most the sensor's start spread (0.6 deg) |
| `radar.score_hist.<sensor>` | the detector's score histograms of real and false boxes (tracker `score_hist`, `_score_learn`) | counts | - | counts halved per 30 days |
| `radar.h_min` | the detector's smallest box per view (radar/box.py `h_min`): its size limit | px per view | the focal length | relaxed upwards 5 % per day |
| `view.lines_scale` | the external lane source's metres against the road model's (trackers/view.py `ext_scale`) | value | - | as kept (it is low-passed again at once) |
| `signrows.reference` | the sign rows' reference period (sources/signrows.py `ref_med`): a pitch reference | rows per size class | the calibration in use, the focal length | as kept: the reference is fixed |

Not kept, and why: the camera lead's range-sensor scale (`vision_scale`) expires after 60 s by design; the
estimator's Gauss-Markov states (`vision.m`, `mf`, the receiver's slowly varying error, the map offset) describe
minutes of one drive; the radar's count of unexplained detections per scan, the fault detectors' learnt noise
and windows, the monitors' spreads describe the present stream, not the car.

The radar learners live in files of the radar (its estimation core); they are reached here through their
attributes (`Tracker.bias`, `Tracker.score_hist`, `Tracker.score_odds`, `Radar.h_min`, `Radar.focal_px`), not by
code there: a prior is placed where the learner will find it (a histogram before its first box) or replaces the
learner's start value while it has learnt nothing yet (the pitch while its spread is still its start spread)."""
import math
import time

from .learnt import DEFAULT_CFG as STORE_CFG, LearntStore, Spec
from .mixer import In, Plugin

DEG = math.pi / 180.0


def calib_signature(msg):
    """The conditions a `calib` message sets: {"calib": "...", "focal": "..."} or {}."""
    if not isinstance(msg, dict):
        return {}

    def f(k, fmt):
        v = msg.get(k)
        return fmt % v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else "-"
    out = {"calib": "p%s y%s r%s h%s" % (f("pitch", "%.3f"), f("yaw", "%.3f"), f("roll", "%.3f"), f("height_m", "%.2f"))}
    if isinstance(msg.get("focal_px"), (int, float)):
        out["focal"] = "f%.0f w%s" % (msg["focal_px"], msg.get("width") or "-")
    return out


class Learner:
    """A learner's face to the store: `export(store, cond)` puts its present values in, `seed(store, cond)` takes
    its priors (called on start, after the learner was started anew, and when the conditions become known)."""
    name = "?"

    def export(self, store, cond):
        pass

    def seed(self, store, cond):
        pass

    def status(self):
        return {}


class EstimatorCalibration(Learner):
    name = "estimator"

    def __init__(self, est):
        self.est = est
        self.seeded = None               # the integrator object the priors went into
        self.assumed = None              # the calibration the vision priors were taken under, before it was known
        self.n_prior = 0
        if hasattr(est, "on_new"):
            est.on_new.append(lambda e: self._on_new())

    def _specs(self):
        I = self.est.integ
        m = I.model
        out = {"sv": (m.index["sv"], Spec("gauss", lo=-0.2, hi=0.2, var_cap=m.c["scale_sigma0"] ** 2,
                                          q_restart=(0.1 * m.c["scale_sigma0"]) ** 2,
                                          q_per_day=(0.1 * m.c["scale_sigma0"]) ** 2))}
        for name, a in m.aux:
            if a.tau_s:                  # Gauss-Markov: minutes of one drive, not the car
                continue
            lo = a.lo if a.lo is not None else a.x0 - 6.0 * a.sigma0
            hi = a.hi if a.hi is not None else a.x0 + 6.0 * a.sigma0
            out[name] = (m.index[name], Spec("gauss", lo=lo, hi=hi, var_cap=a.sigma0 ** 2, q_restart=(0.1 * a.sigma0) ** 2,
                                             q_per_s=a.psd, conditions=("calib",) if name.startswith("vision.") else ()))
        return out

    def export(self, store, cond):
        I = self.est.integ
        snap = I.calibration_snapshot()
        if not snap:
            return
        for name, (i, spec) in self._specs().items():
            if name in snap:
                v, var = snap[name]
                store.put("estimator." + name, spec, float(v), float(var), None, cond)

    def _on_new(self):
        self.seeded = None
        if getattr(self, "store", None) is not None:
            self.seed(self.store, self.cond)

    def seed(self, store, cond):
        self.store, self.cond = store, dict(cond)
        I = self.est.integ
        if self.seeded is I or I.ekf.base is not None:
            return
        snap = {}
        for name, (i, spec) in self._specs().items():
            p = store.prior("estimator." + name, spec, cond)
            if p is not None:
                snap[name] = [p.value, p.var]
        if snap:
            I.calibration_restore(snap)
            if "calib" not in cond.get("_known", ()):
                self.assumed = cond.get("calib")
        self.n_prior = len(snap)
        self.seeded = I

    def calib_known(self, cond):
        """The calibration in use has arrived: vision priors taken under another one are undone."""
        if self.assumed is not None and self.assumed != cond.get("calib"):
            I = self.est.integ
            for name, a in I.model.aux:
                if name.startswith("vision.") and not a.tau_s:
                    I.reset_aux(I.model.index[name], a.x0, a.sigma0)
        self.assumed = None

    def status(self):
        return {"priors": self.n_prior}


class TimeOffsets(Learner):
    """The estimator's time offsets that are estimated outside its filter (docs/EGO_MOTION.md): a property of how
    the car's sources stamp their data, kept across drives. A restart may change it a little (another boot of the
    feeder), a day more; the prior never claims more than the configured sigma allows."""
    name = "time_offsets"
    LO, HI = -0.5, 0.5

    def __init__(self, est):
        self.est = est
        self.seeded = None
        self.n_prior = 0
        if hasattr(est, "on_new"):
            est.on_new.append(lambda e: self._on_new())

    def _spec(self):
        sig = self.est.integ.cfg["ego_motion"]["offset_sigma_s"]
        return Spec("gauss", lo=self.LO, hi=self.HI, var_cap=sig ** 2, q_restart=0.005 ** 2, q_per_day=0.01 ** 2)

    def export(self, store, cond):
        spec = self._spec()
        for name, (v, var, n) in self.est.integ.offsets_snapshot().items():
            if n > 0:                    # (what no row has spoken about is the configuration, not something learnt)
                store.put("estimator.offset." + name, spec, round(float(v), 5), float(var), int(n), cond)

    def _on_new(self):
        self.seeded = None
        if getattr(self, "store", None) is not None:
            self.seed(self.store, self.cond)

    def seed(self, store, cond):
        self.store, self.cond = store, dict(cond)
        I = self.est.integ
        if self.seeded is I:
            return
        spec, snap = self._spec(), {}
        for name in I.lags:
            p = store.prior("estimator.offset." + name, spec, cond)
            if p is not None:
                snap[name] = [p.value, p.var]
        I.offsets_restore(snap)
        self.n_prior = len(snap)
        self.seeded = I

    def status(self):
        return {"priors": self.n_prior}


class RadarLearners(Learner):
    name = "radar"
    BIAS = "radar.pitch_bias."
    HIST = "radar.score_hist."
    HMIN = "radar.h_min"

    def __init__(self, radar_plugin):
        self.box = radar_plugin.box
        self.seeded_bias = {}            # sensor -> the list object the prior went into
        self.seeded_hist = {}
        self.hmin_done = False
        self.ok = all(hasattr(self.box, k) for k in ("tracker", "h_min", "focal_px", "camera_cfg")) and \
            all(hasattr(self.box.tracker, k) for k in ("bias", "score_hist", "score_odds", "cfg"))

    def _bias_spec(self):
        c = self.box.camera_cfg
        v0 = (c["bias_std0_deg"] * DEG) ** 2
        return Spec("gauss", lo=-5 * DEG, hi=5 * DEG, var_cap=v0, q_restart=(0.15 * DEG) ** 2,
                    q_per_day=(0.05 * DEG) ** 2, conditions=("calib",))

    def _hist_spec(self):
        return Spec("counts", lo=0.0, hi=1e7, half_life_days=30.0, length=2)

    @staticmethod
    def _hmin_spec():
        return Spec("min", lo=2.0, hi=400.0, relax_per_day=0.05, conditions=("focal",))

    def export(self, store, cond):
        if not self.ok:
            return
        tr = self.box.tracker
        bs = self._bias_spec()
        for src, b in tr.bias.items():
            if "calib" in cond.get("_known", ()):
                store.put(self.BIAS + str(src), bs, float(b[0]), float(b[1]), None, cond)
        hs = self._hist_spec()
        for src, h in tr.score_hist.items():
            store.put(self.HIST + str(src), hs, [[round(float(x), 3) for x in h[0]], [round(float(x), 3) for x in h[1]]],
                      None, round(float(sum(h[0]) + sum(h[1])), 1))
        hm = {str(k): float(v) for k, v in self.box.h_min.items() if k is not None and math.isfinite(v)}
        if hm and "focal" in cond.get("_known", ()):
            store.put(self.HMIN, self._hmin_spec(), hm, None, len(hm), cond)

    def seed(self, store, cond):
        if not self.ok:
            return
        tr = self.box.tracker
        known = cond.get("_known", ())
        # the histograms: placed before the sensor's first box (the tracker makes its own only when none is there)
        n_bins = tr.cfg["score_bins"]
        for key in [k for k in store.records if k.startswith(self.HIST)]:
            src = key[len(self.HIST):]
            cur = tr.score_hist.get(src)
            if cur is not None and self.seeded_hist.get(src) is cur:
                continue
            if cur is not None:
                continue                 # the tracker has started its own: it keeps it (a sensor's boxes came first)
            p = store.prior(key, self._hist_spec(), cond)
            if p is None or len(p.value[0]) != n_bins or len(p.value[1]) != n_bins:
                continue
            h = tr.score_hist[src] = [list(p.value[0]), list(p.value[1])]
            tr.score_odds.pop(src, None)
            self.seeded_hist[src] = h
        # the pitch: replaces the start value of a sensor that has learnt nothing yet
        if "calib" in known:
            bs = self._bias_spec()
            for src, b in tr.bias.items():
                if self.seeded_bias.get(src) is b or b[1] < 0.5 * bs.var_cap:
                    continue
                p = store.prior(self.BIAS + str(src), bs, cond)
                self.seeded_bias[src] = b
                if p is not None:
                    b[0], b[1] = float(p.value), float(p.var)
        # the detector's size limit per view
        if not self.hmin_done and "focal" in known:
            p = store.prior(self.HMIN, self._hmin_spec(), cond)
            self.hmin_done = True
            if p is not None:
                for k, v in p.value.items():
                    try:
                        key = int(k)
                    except ValueError:
                        continue
                    cur = self.box.h_min.get(key)
                    self.box.h_min[key] = v if cur is None or not math.isfinite(cur) else min(cur, v)

    def reset_seen(self):
        """The tracker was started anew (a new drive): its histograms and pitch start again from the priors."""
        self.seeded_bias, self.seeded_hist = {}, {}

    def status(self):
        return {"available": self.ok, "histograms": sorted(self.seeded_hist), "pitch": sorted(self.seeded_bias)}


class ViewScale(Learner):
    name = "view"
    KEY = "view.lines_scale"
    SPEC = Spec("gauss", lo=0.5, hi=2.0)

    def __init__(self, view_plugin):
        self.b = view_plugin.builder

    def export(self, store, cond):
        v = getattr(self.b, "ext_scale", None)
        if isinstance(v, float) and math.isfinite(v):
            store.put(self.KEY, self.SPEC, round(v, 5), None, None, cond)

    def seed(self, store, cond):
        if getattr(self.b, "ext_scale", 0.0) is None:
            p = store.prior(self.KEY, self.SPEC, cond)
            if p is not None:
                self.b.ext_scale = float(p.value)


class SignRowsReference(Learner):
    name = "signrows"
    KEY = "signrows.reference"
    SPEC = Spec("gauss", lo=-5000.0, hi=10000.0, conditions=("calib", "focal"))

    def __init__(self, est):
        self.est = est
        self.seeded = None

    def _src(self):
        return next((s for s in getattr(self.est.integ, "specs", ()) if getattr(s, "id", None) == "signrows"), None)

    def export(self, store, cond):
        s = self._src()
        if s is not None and s.ref_med is not None and {"calib", "focal"} <= set(cond.get("_known", ())):
            store.put(self.KEY, self.SPEC, [None if v is None else round(float(v), 2) for v in s.ref_med], None,
                      sum(len(r) for r in s.ref), cond)

    def seed(self, store, cond):
        s = self._src()
        if s is None or s is self.seeded or s.ref_med is not None or not {"calib", "focal"} <= set(cond.get("_known", ())):
            return
        self.seeded = s
        p = store.prior(self.KEY, self.SPEC, cond)
        if p is not None and isinstance(p.value, list) and len(p.value) == len(s.ref):
            s.ref_med = list(p.value)


class Learnt(Plugin):
    """The keeper: every learner's values into the store every `collect_s`, the store written seldom (learnt.py),
    priors handed back on start, after a learner restarts, and once the conditions (the calibration) are known."""
    name = "learnt"
    inputs = (In("ego_state", trigger=True, optional=True), In("calib", trigger=True, optional=True),
              In("radar", trigger=True, optional=True))
    outputs = ()

    def __init__(self, cfg=None, name=None, path=None):
        super().__init__(None, name)
        self.cfg = dict(STORE_CFG, **(cfg or {}))
        self.store = LearntStore(self.cfg.get("path") or path, self.cfg)
        self.learners = []
        self.cond = {"_known": ()}
        self._assumed = self.store.records.get("_conditions") or {}
        if isinstance(self._assumed.get("value"), dict):        # the last drive's conditions, until the present
            self.cond.update({k: v for k, v in self._assumed["value"].items() if isinstance(v, str)})   # arrive
        self._t_collect = None
        self._moving_since = None
        self._still_since = None
        self._stop_written = False

    def attach(self, stack):
        """Wire the learners of a built stack (stack.py calls it) and hand their priors out."""
        m = stack.mixer
        est = m.plugin("estimator") if "estimator" in m.slots else None
        rad = m.plugin("radar") if "radar" in m.slots else None
        view = m.plugin("view") if "view" in m.slots else None
        if est is not None and hasattr(est, "integ"):
            self.learners += [EstimatorCalibration(est), TimeOffsets(est), SignRowsReference(est)]
        if rad is not None and hasattr(rad, "box"):
            self.learners.append(RadarLearners(rad))
        if view is not None and hasattr(view, "builder"):
            self.learners.append(ViewScale(view))
        self._seed()

    def _seed(self):
        for ln in self.learners:
            try:
                ln.seed(self.store, self.cond)
            except Exception as e:           # a learner's file changed under us: it loses its priors, not the stack
                self.store.rejected[ln.name] = "seed failed: %s" % e

    def _collect(self):
        for ln in self.learners:
            try:
                ln.export(self.store, self.cond)
            except Exception as e:
                self.store.rejected[ln.name] = "export failed: %s" % e
        keep = {k: v for k, v in self.cond.items() if k != "_known"}
        if keep and set(keep) <= set(self.cond.get("_known", ())):
            # the conditions of this drive: assumed on the next start until its own calibration arrives
            old = self.store.records.get("_conditions")
            if old is None or old.get("value") != keep:
                self.store.records["_conditions"] = {"kind": "conditions", "value": keep, "t_real_ns": self.store.clock()}
                self.store.dirty.add("_conditions")

    def on_input(self, stream, msg, ctx):
        if stream == "calib":
            sig = calib_signature(msg)
            if sig:
                known = tuple(sorted(set(self.cond.get("_known", ())) | set(sig)))
                changed = any(self.cond.get(k) != v for k, v in sig.items())
                self.cond.update(sig)
                first = set(known) != set(self.cond.get("_known", ()))
                self.cond["_known"] = known
                if first or changed:
                    for ln in self.learners:
                        if isinstance(ln, EstimatorCalibration):
                            ln.calib_known(self.cond)
                    self._seed()
            return
        if stream == "radar":
            for ln in self.learners:
                if isinstance(ln, RadarLearners):
                    try:
                        ln.seed(self.store, self.cond)
                    except Exception as e:
                        self.store.rejected[ln.name] = "seed failed: %s" % e
            return
        now = ctx.now
        c = self.cfg
        if self._t_collect is None or now - self._t_collect >= c["collect_s"]:
            self._t_collect = now
            self._collect()
            mono = time.monotonic()
            if self.store.due(mono):
                self.store.save(now=mono)
        # a stop after driving: the power may go with the car
        sp = (msg.get("speed") or {}).get("value")
        stands = bool((msg.get("standstill") or {}).get("value")) or (isinstance(sp, (int, float)) and abs(sp) < 0.1)
        if not stands:
            self._still_since = None
            self._moving_since = now if self._moving_since is None else self._moving_since
            self._stop_written = False
        else:
            self._still_since = now if self._still_since is None else self._still_since
            if (not self._stop_written and self._moving_since is not None and now - self._still_since >= c["stop_write_s"]
                    and self._still_since - self._moving_since >= c["stop_after_drive_s"]):
                self._collect()
                mono = time.monotonic()
                if self.store.due(mono, "stop"):
                    self.store.save(now=mono)
                self._stop_written = True
                self._moving_since = None

    def reset(self, reason=None):
        for ln in self.learners:
            if isinstance(ln, RadarLearners):
                ln.reset_seen()
        self._seed()

    def close(self):
        self._collect()
        if self.store.due(time.monotonic(), "shutdown"):
            self.store.save()

    def reset_values(self, keys=None):
        return self.store.reset(keys)

    def health(self):
        h = self.store.health()
        h["learners"] = {ln.name: ln.status() for ln in self.learners}
        h["conditions"] = {k: v for k, v in self.cond.items() if k != "_known"}
        h["conditions_known"] = list(self.cond.get("_known", ()))
        if not self.store.path:
            h.update(state="idle", reason="no state directory: nothing is kept across drives")
        return h
