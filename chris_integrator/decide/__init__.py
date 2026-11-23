"""The decision layer: how threatening each object is, and what the driver should be told (lane DECIDE,
2026-11-23).

    futures.py   the futures judged: the prediction layer's modes per object and the car's own (contract), or the
                 degraded adapters (one constant-velocity mode per object; the car's motion and the plan)
    threat.py    the threat measure: collision probability over the horizon from both position distributions and
                 both extents; the braking, steering and combined demand required to avoid (threat numbers against
                 the grip available at this speed on this road); time-to-brake / time-to-steer against the driver's
                 reaction time; time to collision, time headway; a level per object with its reasons. The
                 arithmetic is native (native/src/decide.c), Python packs and unpacks
    alerts.py    the alert logic: staged alerts with hysteresis, holds, latching, suppression, a stable primary
                 threat, modes off / shadow / on, the shadow log
    fixtures.py  scenarios as `radar` messages with contract futures (tests, tools)

The plug-in is chris_integrator/plugins/decide.py (topic `decide`); `tools/decide_replay.py` replays captures and
simulated scenarios and reports what each alert would have done.
"""
