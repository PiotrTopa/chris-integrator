#!/usr/bin/env python3
"""Replay a recorded session through the radar and pull out the events a human should judge from the camera
frames, each with a contact sheet and a JSON record (docs/RADAR_TRUTH.md).

    python3 tools/radar_review.py /home/piotr/rs-rec-20261122/20261122-202233 --out /tmp/review
    python3 tools/radar_review.py SESSION [SESSION ...] --out DIR [--no-sheets] [--calib-override P,Y]

Event kinds:
    threat       every episode of flag `threat` (the object, from 2 s before to 2 s after)
    in_path      every episode of flag `in_path`
    flip         every object called both `moving` and `stationary` in its life (frames around the changes)
    life         every confirmed object's life as a whole: first / last time, verdict history, positions
    short        confirmed objects seen for under 1 s (candidate false tracks)
    unconfirmed  a run of detector boxes (>= 3 boxes over >= 0.8 s at one bearing) that no confirmed radar
                 object followed (candidate missed objects)

Writes DIR/<session>/events.jsonl and DIR/<session>/<event id>.jpg. Sheets show the camera frames of the event
(brightened; the recordings are at night) with every detector box (thin: view 0 white, view 1 cyan), every
radar object as a ring at its ground point (thin) and the event's object thick in white with its claims
(id, range, speed over the ground, verdict, flags, ttc), plus a zoomed crop of it. The recordings show people
and number plates: keep the output directory OUT of the repository.
"""
import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radar_truth_lib as L  # noqa: E402

COL = {"moving": (60, 230, 60), "stationary": (80, 140, 255), "unknown": (255, 220, 0)}
VIEW_COL = {0: (235, 235, 235), 1: (0, 220, 220)}
MAX_TILES = 9


def runs(track):
    """Verdict history: [[t0, t1, motion]]."""
    out = []
    for T, o in track:
        if out and out[-1][2] == o["motion"]:
            out[-1][1] = T
        else:
            out.append([T, T, o["motion"]])
    return [[round(a, 2), round(b, 2), m] for a, b, m in out]


def claims(points):
    """What the radar said over these frames."""
    mot, fl, st = collections.Counter(), collections.Counter(), collections.Counter()
    for _, o in points:
        mot[o["motion"]] += 1
        st[o["state"]] += 1
        for f in o["flags"]:
            fl[f] += 1
    ttc = [o["ttc"] for _, o in points if o["ttc"] is not None]
    pth = [o["p_threat"] for _, o in points if o["p_threat"] is not None]
    step = max(1, len(points) // 10)
    pos = [{"t": round(T, 2), "x": o["x"], "y": o["y"], "speed": o["speed"], "motion": o["motion"], "state": o["state"]}
           for T, o in points[::step]]
    labels = collections.Counter(o["label"] for _, o in points if o["label"])
    return {"frames": len(points), "motion_frames": dict(mot), "state_frames": dict(st), "flag_frames": dict(fl),
            "ttc_min_s": min(ttc) if ttc else None, "p_threat_max": max(pth) if pth else None,
            "class": labels.most_common(1)[0][0] if labels else None, "positions": pos}


def box_chains(rus, min_boxes=3, min_span=0.8):
    """Runs of detector boxes at one bearing: [[(T, obj)]]."""
    chains, open_ = [], []
    for T, _, objs in rus:
        open_ = [c for c in open_ if T - c[-1][0] <= 0.8]
        for o in objs:
            best = None
            for c in open_:
                d = abs(c[-1][1]["b"] - o["b"])
                if d <= 3.0 + 4.0 * (T - c[-1][0]) and c[-1][0] < T and (best is None or d < best[0]):
                    best = (d, c)
            if best is None:
                c = [(T, o)]
                chains.append(c)
                open_.append(c)
            else:
                best[1].append((T, o))
    return [c for c in chains if len(c) >= min_boxes and c[-1][0] - c[0][0] >= min_span]


def find_events(name, frames, ins):
    rus = L.Rus(ins["rus"])
    tr = L.tracks(frames)
    ev = []

    def add(kind, oid, t0, t1, pts, focus, extra=None):
        eid = "%s-%s-%s-%.1f" % (name, kind, oid if oid is not None else "x", t0)
        rec = {"session": name, "event": eid, "kind": kind, "object": oid, "t0": round(t0, 2), "t1": round(t1, 2),
               "duration_s": round(t1 - t0, 2), "claims": claims(pts) if pts else None,
               "anchor": L.anchor_of(pts, rus) if pts else None, "focus": [round(f, 2) for f in focus]}
        rec.update(extra or {})
        ev.append(rec)

    for oid, track in sorted(tr.items()):
        seen = [(T, o) for T, o in track if o["state"] in L.SEEN]
        if not seen:
            continue
        for flag in ("threat", "in_path"):
            for t0, t1, pts in L.episodes(track, flag):
                ctx = [(T, o) for T, o in track if t0 - 2.0 <= T <= t1 + 2.0]
                add(flag, oid, t0, t1, ctx, [t0, t1])
        mot = collections.Counter(o["motion"] for _, o in track)
        hist = runs(track)
        if mot["moving"] and mot["stationary"]:
            ch = [hist[k][0] for k in range(1, len(hist)) if {hist[k][2], hist[k - 1][2]} == {"moving", "stationary"}
                  or hist[k][2] in ("moving", "stationary")]
            add("flip", oid, track[0][0], track[-1][0], track, ch[:6], {"verdicts": hist})
        add("life", oid, track[0][0], track[-1][0], track, [], {"seen_t0": round(seen[0][0], 2),
                                                              "seen_t1": round(seen[-1][0], 2), "verdicts": hist})
        if seen[-1][0] - seen[0][0] < 1.0:
            add("short", oid, seen[0][0], seen[-1][0], seen, [seen[0][0]], {"verdicts": hist})
    # detector box runs no confirmed object followed
    for c in box_chains(rus):
        anchor = [{"t": round(T, 3), "b": round(o["b"], 2), "r": None if o["r"] is None else round(o["r"], 1),
                   "box": [round(v) for v in o["box"]] if o["box"] else None, "src": "box"} for T, o in c]
        lab = {"t0": c[0][0], "t1": c[-1][0], "anchor": anchor}
        if any(L.match(t, lab, min_share=0.3, min_frames=2)[0] for t in tr.values()):
            continue
        cls = collections.Counter(o["cls"] for _, o in c).most_common(1)[0][0]
        eid = "%s-unconfirmed-b%+.0f-%.1f" % (name, c[0][1]["b"], c[0][0])
        ev.append({"session": name, "event": eid, "kind": "unconfirmed", "object": None, "t0": round(c[0][0], 2),
                   "t1": round(c[-1][0], 2), "duration_s": round(c[-1][0] - c[0][0], 2), "claims": None,
                   "boxes": len(c), "det_class": cls, "anchor": anchor, "focus": [round(c[0][0], 2)]})
    ev.sort(key=lambda e: (e["t0"], e["kind"]))
    return ev, tr


# ---- contact sheets ---------------------------------------------------------------------------------------------
def pick_images(images, t0, t1, focus, pad=2.0, n=MAX_TILES):
    ts = [t for t, _ in images]
    cand = [k for k, t in enumerate(ts) if t0 - pad <= t <= t1 + pad]
    if len(cand) < 3:                                    # images come ~1 Hz: widen around a short event
        mid = 0.5 * (t0 + t1)
        cand = sorted(sorted(range(len(ts)), key=lambda k: abs(ts[k] - mid))[:min(5, len(ts))])
    if len(cand) <= n:
        return cand
    keep = set()
    for f in focus:                                      # the frames nearest the moments that matter first
        k = min(cand, key=lambda k: abs(ts[k] - f))
        keep.add(k)
        if len(keep) >= n // 2:
            break
    rest = [k for k in cand if k not in keep]
    need = n - len(keep)
    keep.update(rest[int(i * len(rest) / need)] for i in range(need))
    return sorted(keep)


def nearest_frame(frames, ts, T, win=0.15):
    import bisect
    i = bisect.bisect_left(ts, T)
    best = min((j for j in (i - 1, i) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - T), default=None)
    return frames[best] if best is not None and abs(ts[best] - T) <= win else None


def draw_sheet(path, ev, frames, fts, ins):
    from PIL import Image, ImageDraw
    from chris_integrator.sim.roaduser_sim import ground_pixel
    import bisect
    calib = ins["calib"]
    imgs = ins["images"]
    rus = ins["rus"]
    rts = [r[0] for r in rus]
    ks = pick_images(imgs, ev["t0"], ev["t1"], ev["focus"])
    W, H, CAP = 640, 360, 34
    rows = max(1, (len(ks) + 2) // 3)
    head = 36
    sheet = Image.new("RGB", (3 * W, head + rows * (H + CAP)), "black")
    dd = ImageDraw.Draw(sheet)
    c = ev["claims"] or {}
    dd.text((6, 4), "%s  kind %s  object %s  %.1f..%.1f s (%.1f s)  class %s" % (
        ev["event"], ev["kind"], ev["object"], ev["t0"], ev["t1"], ev["duration_s"], c.get("class") or ev.get("det_class")),
        fill=(255, 255, 255))
    dd.text((6, 19), "motion frames %s  flags %s  ttc min %s  p_threat max %s" % (
        c.get("motion_frames"), c.get("flag_frames"), c.get("ttc_min_s"), c.get("p_threat_max")), fill=(200, 200, 200))
    for n, k in enumerate(ks):
        T, fp = imgs[k]
        im = Image.open(fp).convert("RGB")
        im = im.point(lambda v: min(255, int(255 * (v / 255.0) ** 0.55)))      # lift the night
        sx, sy = im.size[0] / 1280.0, im.size[1] / 720.0
        d = ImageDraw.Draw(im)
        a, b = bisect.bisect_left(rts, T - 0.35), bisect.bisect_right(rts, T + 0.35)
        for tb, view, objs in rus[a:b]:
            for o in objs:
                bx = o["box"]
                if bx:
                    col = VIEW_COL.get(view, (200, 200, 200))
                    d.rectangle([bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy], outline=col, width=2)
                    d.text((bx[0] * sx + 2, bx[3] * sy + 2), "%s %.2f %s b%.1f" % (o["cls"], o["score"] or 0, o["clip"], o["b"]), fill=col)
        fr = nearest_frame(frames, fts, T)
        me, crop_at = None, None
        for o in (fr[2] if fr else ()):
            is_me = o["id"] == ev["object"]
            p = ground_pixel(calib, o["x"], o["y"]) if o["x"] > 0.5 else None
            if is_me:
                me = o
            if p is None or not (-50 < p[0] < 1330 and 0 < p[1] < 770):
                continue
            u, v = p[0] * sx, p[1] * sy
            col = (255, 255, 255) if is_me else COL.get(o["motion"], (200, 200, 200))
            rr = 12 if is_me else 7
            d.ellipse([u - rr, v - rr, u + rr, v + rr], outline=col, width=4 if is_me else 2)
            d.line([u - 2 * rr, v, u + 2 * rr, v], fill=col, width=2)
            if is_me:
                crop_at = (u, v)
            else:
                d.text((u + rr + 2, v - 14), "#%s %s" % (o["id"], o["motion"][:4]), fill=col)
        # the anchor's point (an unconfirmed box run has no radar object)
        if me is None and ev.get("anchor"):
            ab = L.anchor_bearing(ev["anchor"], T)
            if ab is not None:
                p = ground_pixel(calib, 20.0 * math.cos(math.radians(ab)), 20.0 * math.sin(math.radians(ab)))
                u = (p[0] if p else 640.0) * sx
                d.line([u, 380 * sy, u, 470 * sy], fill=(255, 255, 255), width=3)
                crop_at = (u, 420 * sy)
        near = [a for a in ev.get("anchor") or () if a["box"] and abs(a["t"] - T) <= 0.6]
        if near:                                         # the box the event is anchored to: crop on it, mark it
            a = min(near, key=lambda a: abs(a["t"] - T))
            bx = a["box"]
            d.rectangle([bx[0] * sx - 3, bx[1] * sy - 3, bx[2] * sx + 3, bx[3] * sy + 3], outline=(255, 255, 255), width=1)
            crop_at = (0.5 * (bx[0] + bx[2]) * sx, (bx[3] - 0.2 * (bx[3] - bx[1])) * sy)
        if crop_at is not None:
            u, v = crop_at
            cw, chh = 150, 100
            box = (int(max(0, min(im.size[0] - 2 * cw, u - cw))), int(max(0, min(im.size[1] - 2 * chh, v - 1.2 * chh))))
            crop = im.crop((box[0], box[1], box[0] + 2 * cw, box[1] + 2 * chh)).resize((300, 200))
        im = im.resize((W, H))
        if crop_at is not None:
            im.paste(crop, (W - 302, 2))
            ImageDraw.Draw(im).rectangle([W - 303, 1, W - 1, 203], outline=(255, 255, 255))
        x0, y0 = (n % 3) * W, head + (n // 3) * (H + CAP)
        sheet.paste(im, (x0, y0))
        ego = fr[1] if fr else None
        cap = "t %+.1f s (T %.1f)  ego %s m/s  %s" % (T - ev["t0"], T, "-" if ego is None else "%.1f" % ego, os.path.basename(fp))
        dd.text((x0 + 4, y0 + H + 2), cap, fill=(255, 255, 255))
        if me is not None:
            dd.text((x0 + 4, y0 + H + 17), "#%s %s r%.1f b%+.1f v%.1f %s %s ttc %s" % (
                me["id"], me["state"], math.hypot(me["x"], me["y"]), me["b"] or 0.0, me["speed"], me["motion"],
                ",".join(me["flags"]) or "-", me["ttc"]), fill=(255, 255, 160))
        elif ev["object"] is not None:
            dd.text((x0 + 4, y0 + H + 17), "#%s not listed at this time" % ev["object"], fill=(160, 160, 160))
    sheet.save(path, quality=82)


def review(session, out_dir, co=L.CALIB_OVERRIDE, sheets=True):
    name = L.session_name(session)
    frames = L.replay(session, co)
    ins = L.inputs(session, co)
    ev, tr = find_events(name, frames, ins)
    od = os.path.join(out_dir, name)
    os.makedirs(od, exist_ok=True)
    fts = [f[0] for f in frames]
    with open(os.path.join(od, "events.jsonl"), "w") as f:
        for e in ev:
            if sheets:
                e["sheet"] = os.path.join(od, e["event"] + ".jpg")
                draw_sheet(e["sheet"], e, frames, fts, ins)
            f.write(json.dumps(e) + "\n")
    return ev, frames, ins


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", nargs="+")
    ap.add_argument("--out", required=True, help="output directory, outside the repository")
    ap.add_argument("--calib-override", default="%s,%s" % L.CALIB_OVERRIDE, help="pitch,yaw (rad); 'none' for the recorded")
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args(argv)
    co = None if a.calib_override == "none" else tuple(float(v) for v in a.calib_override.split(","))
    for s in a.session:
        ev, frames, ins = review(s, a.out, co, not a.no_sheets)
        k = collections.Counter(e["kind"] for e in ev)
        print("%s: %d radar frames over %.0f s, %d camera images, %d box messages; events %s -> %s" % (
            L.session_name(s), len(frames), frames[-1][0] - frames[0][0] if frames else 0, len(ins["images"]),
            len(ins["rus"]), dict(k), os.path.join(a.out, L.session_name(s))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
