#!/usr/bin/env python3
"""Ego-motion hygiene on recorded drives (docs/EGO_MOTION.md): what the sources' time stamps and the gyro's bias are,
raw against raw, and how the estimator does against the truth a standstill gives.

    tools/ego_hygiene.py lags SESSION...                   time offsets between the ego sources, raw against raw
    tools/ego_hygiene.py bias [--cfg JSON] SESSION...      the session through the estimator: the gyro bias state
                                                           against the gyro at rest, the standstill detector's
                                                           counts, the estimated time offsets

SESSION is a recorder session directory (`messages.jsonl.gz`). Recordings stay where they are; nothing is written.

`lags`: for two signals of one quantity (the gyro and the camera's yaw rate; the bus speed and the camera's speed;
the gyro and the rear wheels' difference, the steering's yaw rate, ...) the lag L and gain k that minimise
sum (a(t) - k b(t + L))^2: b's stamps are L later than a's for the same event. Per 60 s window with excitation
(median and quartiles) and over the whole drive. Also the gyro under the bus's standstill flag (what it reads while
the bus says the car stands), and the speed signal's floor.

`bias`: the truth of the gyro's bias is what the gyro reads at a quiet standstill (at least 5 s; its samples from
2 s in to 0.5 s before the end, their spread under 0.002 rad/s). Reported: the bias state's error on arriving at
each such standstill (and in sigma of its stated uncertainty), the worst inside, on leaving; while moving, the
state against those truths interpolated in time. `--cfg` is the estimator's configuration, e.g.
'{"ego_motion": {"stand_confirm": false}}' for the behaviour before the standstill was confirmed.
"""
import argparse
import gzip
import json
import math
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAN = float("nan")


def _f(v):
    return NAN if v is None or isinstance(v, bool) else float(v)


def messages(path, types):
    """The session's messages of the given types, in order."""
    try:
        with gzip.open(os.path.join(path, "messages.jsonl.gz"), "rb") as fh:
            for ln in fh:
                i = ln.find(b'"type":"')
                if i < 0:
                    continue
                t = ln[i + 8:ln.find(b'"', i + 8)].decode()
                if t not in types:
                    continue
                try:
                    yield t, json.loads(ln)
                except ValueError:
                    continue
    except (EOFError, zlib.error):         # a session cut by the power going off
        return


def load(path):
    """{ego: [t, speed, gyro, ax, ay, steering deg, standstill, 4 wheels], road: [t, vx, yaw rate (left +)],
    gnss: [t, arrival, lat, lon, speed]}"""
    ego, road, gnss = [], [], []
    for t, m in messages(path, ("ego", "road", "gnss")):
        ts = m.get("ts_mono_ns")
        if not isinstance(ts, int):
            continue
        if t == "ego":
            ws = m.get("wheel_speeds_mps") or [NAN] * 4
            ego.append([ts / 1e9, _f(m.get("speed_mps")), _f(m.get("yaw_rate_rps")) if m.get("yaw_source") == "yaw_rate" else NAN,
                        _f(m.get("accel_long_mps2")), _f(m.get("accel_lat_mps2")), _f(m.get("steering_wheel_deg")),
                        1.0 if m.get("standstill") else 0.0] + [_f(w) for w in ws[:4]])
        elif t == "road":
            p = m.get("pose") or {}
            if m.get("valid") is not False and p.get("vx") is not None and p.get("yaw_rate") is not None:
                road.append([ts / 1e9, _f(p.get("vx")), -_f(p.get("yaw_rate"))])
        else:
            gnss.append([ts / 1e9, _f(m.get("arrival_mono_ns")) / 1e9, _f(m.get("lat")), _f(m.get("lon")), _f(m.get("speed_mps"))])
    return {"ego": np.array(ego).reshape(-1, 11), "road": np.array(road).reshape(-1, 3), "gnss": np.array(gnss).reshape(-1, 5)}


# ---- lags ---------------------------------------------------------------------------------------------------------
def lag_fit(ta, a, tb, b, lo=-0.6, hi=0.6, step=0.005, mask=None):
    """(L, k, residual / rms a, n): L and k minimising sum (a(t) - k b(t + L))^2; None without 30 samples."""
    best, res = None, []
    for L in np.arange(lo, hi + 1e-9, step):
        bi = np.interp(ta + L, tb, b, left=np.nan, right=np.nan)
        ok = np.isfinite(bi) & np.isfinite(a)
        if mask is not None:
            ok &= mask
        if ok.sum() < 30:
            res.append(np.inf)
            continue
        x, y = bi[ok], a[ok]
        k = (x @ y) / max(x @ x, 1e-12)
        r = math.sqrt(np.mean((y - k * x) ** 2)) / max(math.sqrt(np.mean(y ** 2)), 1e-12)
        res.append(r)
        if best is None or r < best[2]:
            best = (float(L), float(k), r, int(ok.sum()))
    if best is None:
        return None
    res = np.array(res)
    i = int(np.argmin(res))
    L = best[0]
    if 0 < i < len(res) - 1 and np.isfinite(res[i - 1]) and np.isfinite(res[i + 1]):
        d = res[i - 1] - 2 * res[i] + res[i + 1]
        if d > 0:                              # the minimum of the parabola through the three points about it
            L += step * 0.5 * (res[i - 1] - res[i + 1]) / d
    return L, best[1], best[2], best[3]


def pair(name, ta, a, tb, b, excited, lo=-0.6, hi=0.6, mask=None):
    wins, t0 = [], ta[0] if len(ta) else 0.0
    while len(ta) and t0 < ta[-1]:
        m = (ta >= t0) & (ta < t0 + 60.0)
        if m.sum() > 50 and excited(a[m]):
            r = lag_fit(ta[m], a[m], tb, b, lo, hi)
            if r is not None and r[2] < 0.6:
                wins.append(r)
        t0 += 60.0
    whole = lag_fit(ta, a, tb, b, lo, hi, mask=mask)
    txt = "   %-34s" % name
    if wins:
        L = np.array([r[0] for r in wins])
        txt += " windows %2d: lag median %+.3f s (quartiles %+.3f %+.3f), gain %.3f;" % (
            len(L), np.median(L), np.percentile(L, 25), np.percentile(L, 75), np.median([r[1] for r in wins]))
    else:
        txt += " no window with excitation;"
    if whole is not None:
        txt += " whole drive: lag %+.3f s, gain %.3f, residual %.2f" % whole[:3]
    print(txt)


def cmd_lags(a):
    for s in a.sessions:
        d = load(s)
        e, r = d["ego"], d["road"]
        print(os.path.basename(s.rstrip("/")))
        if len(e) < 500 or np.sum(np.abs(e[:, 1]) > 1) < 300 or len(r) < 500:
            print("   too little driving")
            continue
        te, v, w, ax, ay, sw, still = (e[:, i] for i in range(7))
        ws = e[:, 7:11]
        tr, vx, wr = r[:, 0], r[:, 1], r[:, 2]
        dte = np.diff(te)
        dw = np.round(np.abs(np.diff(w[np.isfinite(w)])), 4)
        dw = dw[dw > 0]
        vals, cnt = np.unique(dw, return_counts=True) if len(dw) else (np.array([NAN]), np.array([0]))
        mv = still == 0
        print("   bus: period p50 %.3f s; gyro's most frequent step %.4f rad/s (its quantum); lowest speed while moving %.3f m/s "
              "(the sensor's floor)" % (np.median(dte), vals[int(np.argmax(cnt))], np.nanmin(np.abs(v[mv])) if mv.any() else NAN))
        turn, acc = (lambda x: np.nanstd(x) > 0.03), (lambda x: np.nanstd(x) > 1.0)
        pair("camera yaw rate behind the gyro", te, w, tr, wr, turn, mask=np.abs(v) > 2)
        pair("camera speed behind the bus speed", te, v, tr, vx, acc)
        ww = (ws[:, 3] - ws[:, 2]) / 1.48
        ok = np.isfinite(ww) & (np.nanmin(ws, axis=1) > 0.6)
        if ok.sum() > 200:
            pair("rear wheels' yaw behind the gyro", te[ok], w[ok], te[ok], ww[ok], turn, -0.3, 0.3)
        st = v * np.tan(np.radians(sw / 19.1)) / 2.7
        ok = np.isfinite(st) & (np.abs(v) > 3)
        if ok.sum() > 200:
            pair("steering's yaw behind the gyro", te[ok], w[ok], te[ok], st[ok], turn, -0.3, 0.3)
        ok, oka = (dte < 0.25), np.isfinite(ax)
        if oka.sum() > 200:
            pair("accelerometer behind d(speed)/dt", (0.5 * (te[1:] + te[:-1]))[ok], (np.diff(v) / np.maximum(dte, 1e-3))[ok], te[oka],
                 (ax - np.nanmedian(ax))[oka], lambda x: np.nanstd(x) > 0.4, -0.4, 0.4)
        g = d["gnss"]
        if len(g) > 5:
            ok = np.isfinite(g[:, 2])
            print("   receiver: %d fixes, arrival - stamp p50 %.3f s" % (ok.sum(), np.nanmedian(g[:, 1] - g[:, 0])))
        # what the gyro and the camera read while the bus says standstill
        fl = (still > 0) & np.isfinite(w)
        if fl.sum() > 20:
            since, t0 = np.zeros(len(te)), None
            for i in range(len(te)):
                if still[i] > 0:
                    if t0 is None or te[i] - te[i - 1] > 1.0:
                        t0 = te[i]
                    since[i] = te[i] - t0
                else:
                    t0 = None
            cw = np.abs(np.interp(te, tr - 0.125, wr))
            for lo, hi in ((0.0, 0.5), (0.5, 2.0), (2.0, 1e9)):
                m = fl & (since >= lo) & (since < hi)
                if m.sum() >= 5:
                    print("   under the standstill flag, %3.1f-%s s in: gyro |.| p50 %.4f p90 %.4f max %.4f; camera yaw rate p50 %.4f p90 %.4f max %.4f rad/s (n %d)" % (
                        lo, "%.1f" % hi if hi < 100 else "..", *np.percentile(np.abs(w[m]), [50, 90, 100]), *np.percentile(cw[m], [50, 90, 100]), m.sum()))


# ---- the gyro bias state against the gyro at rest ------------------------------------------------------------------
def replay(path, cfg):
    sys.path.insert(0, HERE)
    from chris_integrator.core.integrator import Integrator
    I = Integrator(dict(cfg))
    ib, log = I.model.index["vehicle.b_g"], []
    for t, m in messages(path, set(I.by_type)):
        if t == "road":
            I.on_road(m)
        else:
            I.on_message(m)
        if t == "ego" and isinstance(m.get("yaw_rate_rps"), (int, float)) and m.get("yaw_source") == "yaw_rate":
            log.append((m["ts_mono_ns"] / 1e9, 1.0 if m.get("standstill") else 0.0, float(m["yaw_rate_rps"]), float(I.x_now[ib]),
                        math.sqrt(max(float(I.P_now[ib, ib]), 0.0)), float(m.get("speed_mps") or 0.0), float(I.x_now[5])))
    return I, np.array(log).reshape(-1, 7)


def cmd_bias(a):
    cfg = json.loads(a.cfg) if a.cfg else {}
    rest, moving = [], []
    for s in a.sessions:
        I, L = replay(s, cfg)
        print(os.path.basename(s.rstrip("/")))
        if len(L) < 100:
            print("   no gyro as read in this session")
            continue
        t, still, gy, bg, sd, v, w = L.T
        eps, cur = [], None
        for i in range(len(t)):
            if still[i] > 0 and (cur is None or t[i] - t[cur[-1]] < 1.0):
                cur = (cur or []) + [i]
            else:
                if cur and t[cur[-1]] - t[cur[0]] >= 5.0:
                    eps.append(cur)
                cur = [i] if still[i] > 0 else None
        if cur and t[cur[-1]] - t[cur[0]] >= 5.0:
            eps.append(cur)
        rows = []
        for c in eps:
            c = np.array(c)
            q = c[(t[c] - t[c[0]] >= 2.0) & (t[c[-1]] - t[c] >= 0.5)]
            if len(q) < 10 or np.std(gy[q]) > 0.002:       # the car was not at rest: no truth here
                continue
            truth, i0 = float(np.median(gy[q])), max(c[0] - 1, 0)
            worst = int(c[np.argmax(np.abs(bg[c] - truth))])
            rows.append((t[c[0]], t[c[-1]] - t[c[0]], truth, bg[i0] - truth, sd[i0], bg[worst] - truth, sd[worst], bg[c[-1]] - truth))
            print("   at %5.0f s for %5.1f s: gyro at rest %+.4f | bias state on arriving %+.4f (%.1f sigma), worst inside %+.4f (%.1f sigma), "
                  "on leaving %+.4f rad/s" % (rows[-1][0] - t[0], rows[-1][1], truth, rows[-1][3], abs(rows[-1][3]) / max(rows[-1][4], 1e-9),
                                              rows[-1][5], abs(rows[-1][5]) / max(rows[-1][6], 1e-9), rows[-1][7]))
        fl = still > 0
        print("   the yaw-rate state under the bus's standstill flag: |.| p99 %.4f max %.4f rad/s (the gyro there: p99 %.4f max %.4f)" % (
            *(np.percentile(np.abs(w[fl]), [99, 100]) if fl.any() else (0, 0)), *(np.percentile(np.abs(gy[fl]), [99, 100]) if fl.any() else (0, 0))))
        if rows:
            tt, tv = np.array([r[0] + 0.5 * r[1] for r in rows]), np.array([r[2] for r in rows])
            mv = np.abs(v) > 2.0
            if mv.sum() > 50:
                err = bg[mv] - np.interp(t[mv], tt, tv)
                print("   while moving, the bias state against those truths interpolated: rms %.4f p95 %.4f max %.4f rad/s (%.1f sigma rms)" % (
                    math.sqrt(np.mean(err ** 2)), np.percentile(np.abs(err), 95), np.max(np.abs(err)), math.sqrt(np.mean((err / sd[mv]) ** 2))))
                moving.append(np.stack([err, sd[mv]]))
        rest += rows
        cal = I.state(int(t[-1] * 1e9))["calibration"]
        st = getattr(I, "stand", None)
        if st is not None:
            n_bus, n_claimed, n_conf, n_ev = st.counts
            print("   standstill: %d bus samples, %d under the flag confirmed, %d claimed only (%.0f %%), %d confirmations; gyro bias steps %s" % (
                n_bus, n_conf, n_claimed, 100.0 * n_claimed / max(1, n_claimed + n_conf), n_ev, cal["vehicle"].get("stand", {}).get("bias_steps")))
        for k, o in (cal.get("time_offsets") or {}).items():
            if "std" in o:
                print("   time offset %-13s %+.4f +- %.4f s (configured %+.4f; %d rows over %.0f s of excitation, their share %.2f)" % (
                    k, o["value"], o["std"], o["configured"], o["rows"], o["excited_s"], o["info"]))
        print("   filter states: vision.tau %.4f +- %.4f s; camera `handled` %s, mount %s" % (
            cal["vision"]["tau"]["value"], cal["vision"]["tau"]["std"], cal["vision"].get("handled"), cal["vision"].get("state")))
    if rest:
        r = np.array(rest)
        print("ALL quiet standstills (%d): bias state on arriving rms %.4f max %.4f rad/s, %.1f sigma rms; worst inside rms %.4f max %.4f, "
              "%.1f sigma rms; on leaving rms %.4f" % (len(r), math.sqrt(np.mean(r[:, 3] ** 2)), np.max(np.abs(r[:, 3])),
                                                       math.sqrt(np.mean((r[:, 3] / np.maximum(r[:, 4], 1e-9)) ** 2)),
                                                       math.sqrt(np.mean(r[:, 5] ** 2)), np.max(np.abs(r[:, 5])),
                                                       math.sqrt(np.mean((r[:, 5] / np.maximum(r[:, 6], 1e-9)) ** 2)), math.sqrt(np.mean(r[:, 7] ** 2))))
    if moving:
        m = np.concatenate(moving, axis=1)
        print("ALL moving: bias state error rms %.4f p95 %.4f max %.4f rad/s, %.1f sigma rms" % (
            math.sqrt(np.mean(m[0] ** 2)), np.percentile(np.abs(m[0]), 95), np.max(np.abs(m[0])), math.sqrt(np.mean((m[0] / m[1]) ** 2))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lags")
    p.add_argument("sessions", nargs="+")
    p = sub.add_parser("bias")
    p.add_argument("--cfg")
    p.add_argument("sessions", nargs="+")
    a = ap.parse_args()
    with np.errstate(all="ignore"):
        (cmd_lags if a.cmd == "lags" else cmd_bias)(a)


if __name__ == "__main__":
    main()
