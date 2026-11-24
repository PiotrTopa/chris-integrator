"""Alert entry by evidence (decide/alerts.py `entry_evidence`, lane CURVES 2026-11-24): an alert on an object whose
PLACE is in doubt (decide.objects[].p_place below entry_evidence: the prediction gave it two places, in the car's way
and not) needs the conflict to last in proportion to how little it exceeds the level's minimum; every other object
enters by time as before."""
from chris_integrator.decide import alerts as AL


def _rec(mass, p_place=None, le=2):
    r = {"id": 1, "kind": "vehicle", "level": le, "level_exit": le, "mass": [mass] * 4, "p_coll": mass, "tm_s": 1.0,
         "a_req": {"brake": 3.0, "left": 2.0, "right": 2.0}, "why": {}}
    if p_place is not None:
        r["p_place"] = p_place
    return r


def _entered(cfg, rec, T=4.0):
    a = AL.Alert("obstacle_ahead", cfg)
    ob = {1: {"id": 1, "state": "confirmed", "exist": 1.0, "motion": "stationary", "p_stationary": 0.95, "flags": []}}
    for k in range(int(T / 0.05) + 1):
        out, _ = a.step(k * 0.05, [rec], ob, {"v": 13.0})
        if out["stage"] >= 2:
            return round(k * 0.05, 2)
    return None


def test_an_alert_on_a_place_in_doubt_needs_the_conflict_to_last():
    ev = {"entry_evidence": 0.6}
    assert _entered({"entry_evidence": 0.0}, _rec(0.3, p_place=0.3)) == 0.2    # by time, as ever
    assert _entered(ev, _rec(0.3)) == 0.2                      # no doubt about the place: by time
    assert _entered(ev, _rec(0.3, p_place=0.95)) == 0.2
    assert _entered(ev, _rec(0.9, p_place=0.5)) == 0.2         # a place in doubt but a clear conflict
    t = _entered(ev, _rec(0.32, p_place=0.35))                 # 0.07 over the minimum of 0.25: five times as long
    assert t is not None and 0.9 < t < 1.1, t
    assert _entered(ev, _rec(0.26, p_place=0.3), T=3.0) is None
