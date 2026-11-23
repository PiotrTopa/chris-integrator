"""The radar's plug-in face: the component `chris_integrator/radar` wired into the mixer, like the map box.

In: the estimator's `ego_state` (the car's own motion; its samples also set the output times), a camera's
`road_users`, range sensors' `objects`, the road model's `road` (its plan is the path). Out: `radar`, one
message per `ego_state` (the objects around the car in the car frame of that moment; radar/box.py).

Nothing here feeds the estimator: the radar reads the car's motion and the sensors, and consumers read the
radar. A reset of the stack (a new ride, a camera restart) starts it empty."""
from ..mixer import In, Out, Plugin
from ..radar import Radar as _Radar


class Radar(Plugin):
    name = "radar"
    inputs = (In("ego_state", trigger=True), In("road_users", trigger=True, optional=True),
              In("objects", trigger=True, optional=True), In("road", trigger=True, optional=True),
              In("calib", trigger=True, optional=True))
    outputs = (Out("radar", "measurement", "frame",
                   doc="the objects around the car: position, velocity, class, standing or moving, prediction"),)

    def __init__(self, cfg=None, name=None):
        super().__init__(None, name)
        self.box = _Radar(cfg)

    def on_input(self, stream, msg, ctx):
        b = self.box
        if stream == "road_users":
            b.on_road_users(msg)
        elif stream == "objects":
            b.on_objects(msg)
        elif stream == "road":
            b.on_road(msg)
        elif stream == "calib":
            b.on_calib(msg)
        else:
            b.on_ego_state(msg)
            ts = msg.get("ts_mono_ns")
            if isinstance(ts, int):
                out = b.frame(ts)
                if out is not None:
                    out["ts_real_ns"] = msg.get("ts_real_ns")
                    ctx.emit("radar", out)

    def reset(self, reason=None):
        self.box.reset(reason)

    def health(self):
        return self.box.health()
