"""The decision layer: how threatening each object is, and what the driver should be told (lane DECIDE,
2026-11-23).

    futures.py   the contract with the prediction layer: its futures as arrays (`Futures`: handed over in process, or
                 read out of a `radar` / `radar_fut` message), the constants of the degraded adapters, the plan as
                 a path
    threat.py    the threat measure's constants and its face: collision probability over the horizon from both
                 position distributions and both extents; the braking, steering and combined demand required to
                 avoid (threat numbers against the grip available at this speed on this road); time-to-brake /
                 time-to-steer against the driver's reaction time; time to collision, time headway; a level per
                 object with its reasons
    (native/src/decide.c: one call per radar message, `decide_frame`: the futures on one grid, the degraded
                 adapters, each object's footprint, the measures, the message's records. Python configures it,
                 keeps the alerts' state and builds the rest of the message)
    alerts.py    the alert logic: staged alerts with hysteresis, holds, latching, suppression, a stable primary
                 threat, modes off / shadow / on, the shadow log
    fixtures.py  scenarios as `radar` messages with contract futures (tests, tools)

The plug-in is chris_integrator/plugins/decide.py (topic `decide`); `tools/decide_replay.py` replays captures and
simulated scenarios and reports what each alert would have done.
"""
