"""Every prediction is scored later against what the object did.

Every `every_s` the predictor's futures of all seen objects are kept (a batch: the frame's time and pose, the
futures). In the frames that follow, each kept batch takes the place of its objects at the sample times as they
come (the radar's own later estimate of the object, turned into the batch's frame by the car's movement: what the
object "actually did" is the tracker's word, with the tracker's error in it). When a batch's horizon has passed
it is scored by the native core (`predict_score`) and folded into running statistics:

    per horizon (1, 2, 3, 5, 8 s) and kind of road user (vehicle, cyclist, pedestrian, all)
        n          scored predictions that were seen at that horizon
        min_ade    the best future's mean distance up to the horizon
        min_fde    the best future's distance at the horizon
        top_fde    the most probable future's distance at the horizon
        nll        negative log-likelihood of the place under the futures' mixture (probabilities and spreads)
        maha2      squared Mahalanobis distance to the future that fitted best (2 when the spreads are honest)
    calibration: over the futures of predictions with a choice, in ten bins of the stated probability: how many
        futures, and how many of them were right: the one that fitted best, or as good as it (mean error within
        0.3 m + 10 %: two futures that are the same path are both right); `brier`

The cost is a few array operations per kept batch in the frames in which a sample time comes round, and one
native call per object when a batch ends.
"""
import math

import numpy as np

from .. import native as _native

_N = _native.predict
HORIZONS = (1.0, 2.0, 3.0, 5.0, 8.0)
GROUPS = ("vehicle", "cyclist", "pedestrian", "all")
_COLS = [0, 1, 3, 4, 5]                   # of predict_score's row: min_ade, min_fde, top_fde, nll, maha2


class SelfCheck:
    def __init__(self, times, cfg=None):
        self.cfg = dict(enabled=True, every_s=0.5, min_seen=3)
        self.cfg.update(cfg or {})
        self.times = np.asarray(times, dtype=float)
        self.hk = [int(np.argmin(np.abs(self.times - h))) for h in HORIZONS if h <= self.times[-1] + 1e-6]
        self._grid = {int(round(t * 20.0)): k for k, t in enumerate(self.times)}
        self.reset()

    def reset(self):
        self.batches = []
        self._next = None
        nh = len(self.hk)
        self.n = np.zeros((4, nh))
        self.sums = np.zeros((4, nh, 5))          # min_ade, min_fde, top_fde, nll, maha2
        self.cal = np.zeros((10, 2))              # futures in the bin, of them the best-fitting one
        self.brier = [0.0, 0]
        self.scored = 0

    def update(self, T, pred, ego):
        """A frame: pred its prediction, or None for a frame without objects (kept batches still end)."""
        if not self.cfg["enabled"] or (pred is None and not self.batches):
            return
        if self.batches and (T < self.batches[-1]["T"] or T - self.batches[-1]["T"] > 30.0):
            self.batches = []
            self._next = None
        times = self.times
        index = ground = None
        keep = []
        grid = self._grid
        for b in self.batches:
            dt = T - b["T"]
            k = grid.get(int(dt * 20.0 + 0.5), -1) if dt < times[-1] + 0.1 else -1     # the sample this frame is (20 a second)
            if k >= 0 and pred is not None and abs(times[k] - dt) <= 0.03 and not b["done"][k]:
                if index is None:
                    index = {i: j for j, i in enumerate(pred.ids) if pred.seen[j]}
                    ground = ego.to_ground(pred.xy, T)          # where the objects are now, on the ground
                rows = [(r, index[i]) for r, i in enumerate(b["ids"]) if i in index]
                if rows:
                    r, j = zip(*rows)
                    b["actual"][r, k] = ground[j, :]
                    b["valid"][r, k] = 1.0
                b["done"][k] = True
            if dt > times[-1] + 0.2:
                b["actual"] = ego.from_ground(b["actual"], b["T"])     # into the frame the futures were given in
                self._score(b)
            else:
                keep.append(b)
        self.batches = keep
        if pred is not None and (self._next is None or T >= self._next - 1e-6) and len(pred.ids):
            self._next = T + self.cfg["every_s"]
            sel = np.flatnonzero(pred.seen)
            if len(sel):
                nt = len(times)
                self.batches.append({"T": T, "ids": [pred.ids[i] for i in sel], "kind": pred.kind[sel],
                                     "nm": pred.n_modes[sel], "p": pred.p[sel].copy(), "traj": pred.traj[sel].copy(),
                                     "actual": np.zeros((len(sel), nt, 2)), "valid": np.zeros((len(sel), nt)),
                                     "done": [False] * nt})

    def _score(self, b):
        nt = len(self.times)
        out = np.zeros((nt + 2, 6))
        for r in range(len(b["ids"])):
            val = b["valid"][r]
            if val.sum() < self.cfg["min_seen"]:
                continue
            nm = int(b["nm"][r])
            p = np.ascontiguousarray(b["p"][r, :nm])
            win, pw = _N.predict_score(np.ascontiguousarray(b["traj"][r, :nm]), p, np.ascontiguousarray(b["actual"][r]),
                                       np.ascontiguousarray(val), out)
            self.add(int(b["kind"][r]), out, val, p, win)

    def add(self, kind, out, val, p, win):
        """One scored prediction into the statistics (also the offline tool's entry, with the truth as `actual`)."""
        g = 2 if kind == 7 else 1 if kind == 6 else 0
        self.scored += 1
        hk = self.hk
        ok = val[hk] > 0.5
        if ok.any():
            rows = np.where(ok[:, None], out[hk][:, _COLS], 0.0)
            self.n[g] += ok
            self.n[3] += ok
            self.sums[g] += rows
            self.sums[3] += rows
        if len(p) >= 2 and win >= 0:
            ade = out[-2:].ravel()[:len(p)]
            best = ade[win]
            for m, pm in enumerate(p):
                hit = m == win or ade[m] <= 1.1 * best + 0.3
                bin_ = min(9, int(pm * 10.0))
                self.cal[bin_, 0] += 1
                self.cal[bin_, 1] += hit
                self.brier[0] += (pm - hit) ** 2
                self.brier[1] += 1

    def stats(self):
        out = {"scored": self.scored, "pending": len(self.batches), "horizons_s": [float(self.times[k]) for k in self.hk]}
        for g, name in enumerate(GROUPS):
            if not self.n[g].any():
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                m = self.sums[g] / self.n[g][:, None]
            out[name] = {"n": self.n[g].astype(int).tolist()}
            for j, key in enumerate(("min_ade", "min_fde", "top_fde", "nll", "maha2")):
                out[name][key] = [None if not self.n[g][h] else round(float(m[h, j]), 3) for h in range(len(self.hk))]
        if self.brier[1]:
            out["calibration"] = {"bins": [[int(a), int(b)] for a, b in self.cal],
                                  "brier": round(self.brier[0] / self.brier[1], 4)}
        return out
