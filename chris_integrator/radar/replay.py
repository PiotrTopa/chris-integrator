"""A recorded session through the radar component alone.

    for msg in run("/path/to/session"):          # `radar` messages, one per recorded road frame
        ...

Inputs taken from the recording: the vehicle feed (`ego`: speed, yaw rate, acceleration from the car's bus) as
the car's own motion, the light detector's messages (`lights` with `road_users` boxes) placed with the recorded
calibration (`calib`; `calib_override=(pitch, yaw)` replaces its angles for the whole session), and the road
model's frames (`road`: the plan as the path, and the output times). Recorded outputs of the stack (`tracks`,
`view`, `lead`, `ego_state`) are not read.
"""
from ..core.roadusers import PROJ_CFG, from_lights
from ..replay import read_lines
from .box import Radar


def run(session, calib_override=None, cfg=None, radar=None, with_inputs=False):
    """Yields the `radar` message of every road frame (with_inputs: (radar message, road_users message of the
    frame's interval or None))."""
    rd = radar or Radar(cfg)
    pcfg = dict(PROJ_CFG, sensor="cam")
    calib = None
    last_ru = None
    for m in read_lines(session):
        t = m.get("type")
        if t == "calib":
            calib = dict(m)
            if calib_override is not None:
                calib["pitch"], calib["yaw"] = calib_override
            rd.on_calib(calib)
        elif t == "ego":
            rd.on_ego(m)
        elif t == "lights":
            if calib is None and calib_override is None:
                continue
            ru = from_lights(m, calib, "cam", pcfg)
            if ru is not None:
                rd.on_road_users(ru)
                last_ru = ru
        elif t == "objects":
            rd.on_objects(m)
        elif t == "road" and isinstance(m.get("ts_mono_ns"), int):
            rd.on_road(m)
            out = rd.frame(m["ts_mono_ns"])
            if out is not None:
                out["ts_real_ns"] = m.get("ts_real_ns")
                yield (out, last_ru) if with_inputs else out
