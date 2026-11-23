#!/usr/bin/env python3
"""Replay a recorder session through the stack and make contact sheets of what the visual radar says, for
eyeballing against the camera frames (docs/VISUAL_RADAR_EVAL.md).

    python3 tools/visual_radar_replay.py /home/piotr/rs-rec-20261122/20261122-202233 --out /tmp/sheets \\
        --calib-override -0.0140,-0.0624
    python3 tools/visual_radar_replay.py SESSION --out DIR --mode radar-alone   # chris_integrator.radar.replay.run

Inputs fed to the stack (built like the daemon: default estimator sources, i.e. radar + road_users; no map box,
the recorded `map` through the external-map channel): road, calib, camera, env, lights (-> lights_raw),
road_users, signs, display_lanes, governor, ego, vehicle, gnss, objects, map. NOT fed (outputs of the code that
ran in the car; compared against): tracks, ego_state, view, lead, fused, rules, speed_limit, radar.

Scored objects, in order of preference: the `radar` stream (if the stack or --mode radar-alone emits it), the
`tracks` whose sources include road users, else the view's road_user display objects. Colours: moving green,
stationary off the path (parked) blue, stationary on/in the path red, unknown yellow. Detector boxes thin
(view 0 white, view 1 cyan, view 2 magenta). Marker = the object's ground point through the calibration in use.
"""
import argparse
import bisect
import collections
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NS = 1_000_000_000
INPUTS = ("road", "calib", "camera", "env", "lights", "road_users", "signs", "display_lanes", "governor", "ego",
          "vehicle", "gnss", "objects", "map")
RECORDED_OUT = ("tracks", "ego_state", "view", "lead", "fused", "rules", "speed_limit", "radar")
COL = {"moving": (60, 230, 60), "parked": (80, 140, 255), "in_path": (255, 50, 50), "unknown": (255, 220, 0)}
VIEW_COL = {0: (235, 235, 235), 1: (0, 220, 220), 2: (230, 0, 230)}


def _imports(retries=12, wait_s=20.0):
    last = None
    for _ in range(retries):
        try:
            from chris_integrator.mixer import MessageClock
            from chris_integrator.replay import read_lines
            from chris_integrator.sim.roaduser_sim import ground_pixel
            from chris_integrator.stack import build
            return MessageClock, read_lines, ground_pixel, build
        except Exception as e:                      # noqa: BLE001  (the tree may be mid-edit)
            last = e
            print("import failed (%r); retrying in %.0f s" % (e, wait_s), file=sys.stderr)
            time.sleep(wait_s)
    raise last


def override(msg, co):
    if co is not None and msg.get("type") == "calib":
        msg = dict(msg, pitch=co[0], yaw=co[1])
    return msg


def load(session, co):
    """The session's messages: inputs (calib overridden), plus the recorded outputs kept aside."""
    _, read_lines, _, _ = _imports()
    inputs, rec = [], collections.defaultdict(list)
    for m in read_lines(session):
        t = m.get("type")
        if t in RECORDED_OUT:
            rec[t].append(m)
        elif t in INPUTS:
            inputs.append(override(m, co))
    return inputs, rec


def run_stack(inputs, radar_plugin=None):
    MessageClock, _, _, build = _imports()
    extra = []
    if radar_plugin:
        import importlib
        mod, _, cls = radar_plugin.partition(":")
        extra.append(getattr(importlib.import_module(mod), cls)(None))
    st = build(cfg={"estimator": {}}, mapper=None, external_map=True, clock=MessageClock(), plugins=extra or None)
    out = collections.defaultdict(list)
    st.mixer.add_output("replay", lambda e: out[e.stream].append(e.msg),
                        streams=("tracks", "radar", "lead", "view", "road_users"))
    t0 = time.perf_counter()
    for m in inputs:
        if m.get("type") == "road" and not isinstance(m.get("ts_mono_ns"), int):
            continue
        st.deliver(m)
    return out, time.perf_counter() - t0


def run_radar_alone(session, co):
    try:
        from chris_integrator.radar.replay import run
    except Exception as e:                          # noqa: BLE001
        return None, "chris_integrator.radar.replay.run not importable (%r)" % (e,)
    return {"radar": list(run(session, calib_override=co))}, None


# ---- the objects to draw ----------------------------------------------------------------------------------------
def objects_of(source, msg):
    """[{id, state, x, y (left), v, motion, path, label}] of one output message."""
    out = []
    if source == "radar":
        for o in msg.get("objects") or ():
            fl = o.get("flags") or []
            out.append(_obj(o, o.get("x_m"), o.get("y_m"), o.get("vx_mps"), o.get("vy_mps"), o.get("motion"),
                            "on_path" in fl or "in_path" in fl, fl))
    elif source == "tracks":
        for o in msg.get("tracks") or ():
            src = [str(s).replace("_", "") for s in o.get("sources") or ()]
            if "roadusers" not in src:
                continue
            fl = o.get("flags") or []
            out.append(_obj(o, o.get("x_m"), o.get("y_m"), o.get("vx_mps"), o.get("vy_mps"), o.get("motion"),
                            "on_path" in fl, fl))
    else:                                           # view display objects: y right, relative velocity
        v = (msg.get("ego") or {}).get("speed_mps") or 0.0
        for o in msg.get("objects") or ():
            if o.get("kind") != "road_user" or o.get("x") is None:
                continue
            vx, vy = (o.get("vx") or 0.0) + v, -(o.get("vy") or 0.0)
            sp = (vx * vx + vy * vy) ** 0.5
            mot = "stationary" if sp < 1.0 else ("moving" if sp > 2.0 else "unknown")     # derived, as the eval
            d = {"id": o.get("id"), "state": o.get("phase"), "x_m": o["x"], "y_m": -o["y"], "class": o.get("cls")}
            out.append(_obj(d, o["x"], -o["y"], vx, vy, mot, False, []))
    return [o for o in out if o is not None]


def _obj(o, x, y, vx, vy, motion, path, flags):
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    v = (vx * vx + vy * vy) ** 0.5 if isinstance(vx, (int, float)) and isinstance(vy, (int, float)) else None
    cls = o.get("class")
    mot = motion if motion in ("moving", "stationary", "unknown") else (
        "stationary" if ("stationary" in flags or "parked" in flags) else "unknown")
    return {"id": o.get("id"), "state": o.get("state"), "x": float(x), "y": float(y), "v": v, "motion": mot,
            "path": path, "label": cls.get("label") if isinstance(cls, dict) else None}


def colour(o):
    if o["motion"] == "moving":
        return COL["moving"]
    if o["motion"] == "stationary":
        return COL["in_path"] if o["path"] else COL["parked"]
    return COL["unknown"]


# ---- statistics -------------------------------------------------------------------------------------------------
def life_stats(msgs, source):
    first, last, conf = {}, {}, set()
    verdicts = collections.Counter()
    for m in msgs:
        t = m.get("ts_mono_ns")
        if not isinstance(t, int):
            continue
        for o in objects_of(source, m):
            first.setdefault(o["id"], t)
            last[o["id"]] = t
            if o["state"] in ("confirmed", "steady", "appear", "coast") or source == "view":
                if o["state"] != "coast":
                    conf.add(o["id"])
                verdicts[o["motion"]] += 1
    lives = sorted((last[i] - first[i]) / NS for i in conf)

    def q(p):
        return round(lives[min(len(lives) - 1, int(p * (len(lives) - 1) + 0.5))], 2) if lives else None
    return {"confirmed": len(conf), "life_p50_s": q(0.5), "life_p90_s": q(0.9), "life_max_s": q(1.0),
            "life_min_s": q(0.0), "motion_object_frames": dict(verdicts)}


def lead_share(leads):
    n = [bool(m.get("present")) for m in leads]
    return round(sum(n) / len(n), 3) if n else None


# ---- contact sheets ---------------------------------------------------------------------------------------------
def sheets(session, out_dir, source, out_msgs, lights, calibs, per_sheet=15, every=1, max_sheets=None, tz="Europe/Warsaw"):
    from PIL import Image, ImageDraw
    _, _, ground_pixel, _ = _imports()
    from datetime import datetime
    from zoneinfo import ZoneInfo
    zone = ZoneInfo(tz)
    fdir = os.path.join(session, "frames")
    files = sorted((f for f in os.listdir(fdir) if f.endswith(".jpg")), key=lambda x: int(x.split(".")[0]))
    om = [m for m in out_msgs if isinstance(m.get("_real"), int)]
    ot = [m["_real"] for m in om]
    lt = [m["ts_real_ns"] for m in lights]
    ct = [c["ts_real_ns"] for c in calibs]
    tiles = []
    for fn in files[::every]:
        tr = int(os.path.getmtime(os.path.join(fdir, fn)) * NS)
        i = bisect.bisect_left(ot, tr)
        cand = [j for j in (i - 1, i) if 0 <= j < len(ot) and abs(ot[j] - tr) <= 0.3 * NS]
        msg = om[min(cand, key=lambda j: abs(ot[j] - tr))] if cand else None
        objs = objects_of(source, msg) if msg else []
        a, b = bisect.bisect_left(lt, tr - int(0.35 * NS)), bisect.bisect_right(lt, tr + int(0.35 * NS))
        boxes = [(lm.get("view"), r) for lm in lights[a:b] for r in (lm.get("road_users") or ())]
        if not objs and not boxes:
            continue
        k = bisect.bisect_right(ct, tr) - 1
        calib = calibs[k] if k >= 0 else (calibs[0] if calibs else None)
        tiles.append((fn, tr, objs, boxes, calib))
    os.makedirs(out_dir, exist_ok=True)
    W, H, CAP = 640, 360, 30
    paths = []
    for s0 in range(0, len(tiles), per_sheet):
        if max_sheets is not None and len(paths) >= max_sheets:
            break
        chunk = tiles[s0:s0 + per_sheet]
        rows = (len(chunk) + 2) // 3
        sheet = Image.new("RGB", (3 * W, rows * (H + CAP)), "black")
        for n, (fn, tr, objs, boxes, calib) in enumerate(chunk):
            im = Image.open(os.path.join(fdir, fn)).convert("RGB")
            sx, sy = im.size[0] / 1280.0, im.size[1] / 720.0
            d = ImageDraw.Draw(im)
            for view, r in boxes:
                bx = r.get("box")
                if isinstance(bx, list) and len(bx) == 4:
                    d.rectangle([bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy], outline=VIEW_COL.get(view, (200, 200, 200)), width=1)
                    d.text((bx[0] * sx + 2, bx[3] * sy + 1), "%s %.2f %s" % (r.get("cls"), r.get("score") or 0, r.get("clip") or ""),
                           fill=VIEW_COL.get(view, (200, 200, 200)))
            off = []
            for o in objs:
                c = colour(o)
                lab = "#%s %s %s x%.0f y%.1f v%s" % (o["id"], o["label"] or "?", o["motion"][:4], o["x"], o["y"],
                                                     "-" if o["v"] is None else "%.1f" % o["v"])
                if o["state"] == "tentative":
                    lab = "(" + lab + ")"
                p = ground_pixel(calib, o["x"], o["y"]) if calib and o["x"] > 0.5 else None
                if p is None or not (-50 < p[0] < 1330 and 0 < p[1] < 770):
                    off.append((lab, c))
                    continue
                u, v = p[0] * sx, p[1] * sy
                rr = 9 if o["state"] == "confirmed" else 6
                d.ellipse([u - rr, v - rr, u + rr, v + rr], outline=c, width=4 if o["state"] == "confirmed" else 2)
                d.line([u - 2 * rr, v, u + 2 * rr, v], fill=c, width=2)
                d.text((u + rr + 3, v - 18), lab, fill=c)
            for j, (lab, c) in enumerate(off[:6]):
                d.text((8, 8 + 16 * j), "off-image " + lab, fill=c)
            im = im.resize((W, H))
            x0, y0 = (n % 3) * W, (n // 3) * (H + CAP)
            sheet.paste(im, (x0, y0))
            dd = ImageDraw.Draw(sheet)
            when = datetime.fromtimestamp(tr / NS, zone).strftime("%H:%M:%S")
            dd.text((x0 + 4, y0 + H + 2), "%s %s  boxes %d  objects %d (%s)  pitch %.4f" % (
                fn, when, len(boxes), len(objs), source, (calib or {}).get("pitch") or 0.0), fill=(255, 255, 255))
            dd.text((x0 + 4, y0 + H + 15), "green moving  blue stationary/parked  red stationary in path  yellow unknown",
                    fill=(150, 150, 150))
        p = os.path.join(out_dir, "sheet_%03d.jpg" % len(paths))
        sheet.save(p, quality=80)
        paths.append(p)
    return paths, len(tiles)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", nargs="+", help="recorder session directories")
    ap.add_argument("--out", required=True, help="output directory (a subdirectory per session)")
    ap.add_argument("--calib-override", default=None, help="pitch,yaw (rad) for every calib message, e.g. -0.0140,-0.0624")
    ap.add_argument("--mode", default="stack", choices=("stack", "radar-alone"))
    ap.add_argument("--source", default="auto", choices=("auto", "radar", "tracks", "view"))
    ap.add_argument("--radar-plugin", default=None, help="module:Class of a radar plug-in to add to the stack")
    ap.add_argument("--per-sheet", type=int, default=15)
    ap.add_argument("--every", type=int, default=1, help="use every n-th recorded frame")
    ap.add_argument("--max-sheets", type=int, default=None)
    a = ap.parse_args(argv)
    co = tuple(float(v) for v in a.calib_override.split(",")) if a.calib_override else None
    summary = {}
    for ses in a.session:
        name = os.path.basename(os.path.normpath(ses))
        inputs, rec = load(ses, co)
        lights = sorted((m for m in inputs if m.get("type") == "lights" and isinstance(m.get("ts_real_ns"), int)),
                        key=lambda m: m["ts_real_ns"])
        calibs = sorted((m for m in inputs if m.get("type") == "calib" and isinstance(m.get("ts_real_ns"), int)),
                        key=lambda m: m["ts_real_ns"])
        roads = [m for m in inputs if m.get("type") == "road" and isinstance(m.get("ts_real_ns"), int)
                 and isinstance(m.get("ts_mono_ns"), int)]
        offs = sorted(m["ts_real_ns"] - m["ts_mono_ns"] for m in roads)
        off = offs[len(offs) // 2] if offs else 0
        note = None
        wall = None
        if a.mode == "radar-alone":
            out, note = run_radar_alone(ses, co)
            if out is None:
                print("%s: radar-alone mode skipped: %s" % (name, note))
                summary[name] = {"skipped": note}
                continue
        else:
            out, wall = run_stack(inputs, a.radar_plugin)
        source = a.source
        if source == "auto":
            source = "radar" if out.get("radar") else ("tracks" if any(
                "roadusers" in [str(s).replace("_", "") for s in tr.get("sources") or ()]
                for m in out.get("tracks", []) for tr in m.get("tracks") or ()) else "view")
        msgs = out.get(source if source != "view" else "view", [])
        for m in msgs:
            m["_real"] = m["ts_real_ns"] if isinstance(m.get("ts_real_ns"), int) else (
                m["ts_mono_ns"] + off if isinstance(m.get("ts_mono_ns"), int) else None)
        msgs.sort(key=lambda m: m["_real"] or 0)
        paths, ntiles = sheets(ses, os.path.join(a.out, name), source, msgs, lights, calibs, a.per_sheet, a.every,
                               a.max_sheets)
        st = {"source": source, "calib_override": co, "replay_wall_s": None if wall is None else round(wall, 1),
              "new": life_stats(msgs, source), "lead_present_share_new": lead_share(out.get("lead", [])),
              "recorded_view_road_users": life_stats(rec.get("view", []), "view"),
              "recorded_lead_present_share": lead_share(rec.get("lead", [])),
              "lights_messages": len(lights), "road_user_boxes": sum(len(m.get("road_users") or ()) for m in lights),
              "tiles": ntiles, "sheets": paths}
        summary[name] = st
        print("\n== %s (scored: %s; calib override %s) ==" % (name, source, co))
        n = st["new"]
        print("  new:      confirmed objects %d, life p50 %s s p90 %s s max %s s; motion verdict object-frames %s; lead present %s" % (
            n["confirmed"], n["life_p50_s"], n["life_p90_s"], n["life_max_s"], n["motion_object_frames"], st["lead_present_share_new"]))
        r = st["recorded_view_road_users"]
        print("  recorded: view road_user objects %d, life p50 %s s p90 %s s max %s s; lead present %s" % (
            r["confirmed"], r["life_p50_s"], r["life_p90_s"], r["life_max_s"], st["recorded_lead_present_share"]))
        print("  lights %d, boxes %d; %d tiles -> %d sheets in %s" % (len(lights), st["road_user_boxes"], ntiles, len(paths),
                                                                      os.path.join(a.out, name)))
    import json
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
