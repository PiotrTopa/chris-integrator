"""The predictor's plug-in face: the component `chris_integrator/predict` wired into the mixer.

In: `radar` (every message is predicted; `rate_hz` lowers that), and what the stack knows of the road, read as it
stands when the radar speaks: `roadgeom`, `lanes`, `egolane`, `road` (its plan), `map`, the car's own `ego` (turn
signal). Out: `radar_fut`, at the radar message's stamps: that message with the futures added (docs/TOPICS.md).

`{"enabled": false}` in `predict.json` (or `"disable": ["predict"]` in `mixer.json`) leaves it out: the `radar`
message is the same either way, this box only reads it. A reset of the stack starts it empty."""
from ..mixer import In, Out, Plugin
from ..predict import ContextBuilder, Predictor


class Predict(Plugin):
    name = "predict"
    inputs = (In("radar", trigger=True), In("roadgeom", optional=True), In("lanes", optional=True),
              In("egolane", optional=True), In("road", optional=True), In("map", optional=True), In("ego", optional=True))
    outputs = (Out("radar_fut", "estimate", "frame",
                   doc="the radar's objects with their futures (manoeuvres, probabilities, trajectories), lanes and intent"),)

    def __init__(self, cfg=None, name=None):
        super().__init__(None, name)
        cfg = dict(cfg or {})
        self.rate_hz = float(cfg.pop("rate_hz", 20.0))
        self.context = ContextBuilder(cfg.pop("context", None))
        self.box = Predictor(cfg or None)
        self._ts = None
        self._seen = {}

    def on_input(self, stream, msg, ctx):
        b = self.box
        if not b.cfg["enabled"]:
            return
        ts = msg.get("ts_mono_ns")
        if not isinstance(ts, int):
            return
        if self._ts is not None and 0 <= ts - self._ts < (1.0 / self.rate_hz - 0.01) * 1e9:
            return
        self._ts = ts
        cb = self.context
        for name, fn in (("roadgeom", cb.on_roadgeom), ("lanes", cb.on_lanes), ("egolane", cb.on_egolane),
                         ("road", cb.on_road), ("map", cb.on_map), ("ego", cb.on_ego)):
            m = ctx.get(name)
            if m is not None and self._seen.get(name) is not m:
                self._seen[name] = m
                fn(m)
        # the car's pose of this frame first: the context's messages are moved into this frame by it
        eg = msg.get("ego") or {}
        b.ego.add(ts / 1e9, float(eg.get("speed_mps") or 0.0), float(eg.get("yaw_rate_rps") or 0.0))
        pred = b.update(msg, cb.build(ts / 1e9, b.ego))
        if pred is not None:
            ctx.emit("radar_fut", pred.message(msg))

    def reset(self, reason=None):
        self._ts = None
        self._seen = {}
        self.context.reset()
        self.box.reset(reason)

    def health(self):
        return self.box.health()
