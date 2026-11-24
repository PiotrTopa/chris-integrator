"""Truth checks of the bend / half-in-lane / pedestrian conflict scenarios (TRUE_BEND_SCENARIOS): the world builds,
the obstacle is where collision_t says at the end of the run, and nothing else is in the ego's swept band."""
import numpy as np
import pytest

from chris_integrator.sim.roaduser_sim import TRUE_BEND_SCENARIOS, ALERT_SCENARIOS, scenario

HALF = 0.88          # ego half width
FRONT = 2.1          # ego front ahead of the camera
OBST = ("stopped_in_lane", "half_in_lane", "crossing_ped")


@pytest.mark.parametrize("seed", (1, 2, 3))
@pytest.mark.parametrize("name", TRUE_BEND_SCENARIOS)
def test_truth(name, seed):
    assert name in ALERT_SCENARIOS
    w = scenario(name, seed)
    sp = w.spec
    t_col = sp["collision_t"]
    assert sp["T"] == pytest.approx(t_col - 0.3)
    assert sp["mode"] == "cruise" and sp["approach"] == 13.0
    v = sp["approach"]

    obs = [o for o in w.actors_at(w.t0 + t_col - 0.3) if o["role"] in OBST]
    assert len(obs) == 1, obs
    o = obs[0]
    exp = FRONT + 0.3 * v
    assert exp - 1.5 <= o["ref_x"] <= exp + 1.5, (o["ref_x"], exp)
    if o["kind"] == "human":
        assert abs(o["lane_offset"]) < HALF + 0.3, o["lane_offset"]
    else:
        assert abs(o["lane_offset"]) - o["W"] / 2 < HALF, o["lane_offset"]

    for t in np.arange(0.0, sp["T"], 0.1):
        for a in w.actors_at(w.t0 + t):
            if a["id"] == o["id"]:
                continue
            assert abs(a["lane_offset"]) - a["W"] / 2 >= HALF + 0.3, (t, a["role"], a["id"], a["lane_offset"])


# ---- lane SPREAD: the pedestrian scenarios (PED_SCENARIOS) and their controls (PED_CONTROL_SCENARIOS) ----------------
from chris_integrator.sim.roaduser_sim import PED_SCENARIOS, PED_CONTROL_SCENARIOS  # noqa: E402


@pytest.mark.parametrize("seed", (1, 2, 3))
@pytest.mark.parametrize("name", PED_SCENARIOS)
def test_ped_truth(name, seed):
    """The pedestrian is in the car's swept strip when the car's front reaches it, stands before it steps out, and
    nothing else is in the strip."""
    w = scenario(name, seed)
    sp = w.spec
    t_col = sp["collision_t"]
    assert sp["T"] == pytest.approx(t_col - 0.3) and sp["approach"] == 13.0
    obs = [o for o in w.actors_at(w.t0 + t_col - 0.3) if o["role"] == "crossing_ped"]
    assert len(obs) == 1 and obs[0]["kind"] == "human"
    o = obs[0]
    assert FRONT + 0.3 * 13.0 - 1.5 <= o["ref_x"] <= FRONT + 0.3 * 13.0 + 1.5
    assert abs(o["lane_offset"]) < HALF + 0.9 and o["moving"]            # 0.3 s before: still a step from the strip's middle
    early = [a for a in w.actors_at(w.t0 + t_col - 4.0) if a["role"] == "crossing_ped"][0]
    assert not early["moving"] and early["lane_offset"] < -(HALF + 1.0)
    for t in np.arange(0.0, sp["T"], 0.1):
        for a in w.actors_at(w.t0 + t):
            if a["id"] != o["id"]:
                assert abs(a["lane_offset"]) - a["W"] / 2 >= HALF + 0.15, (t, a["role"], a["id"], a["lane_offset"])


def test_ped_variants():
    """What tells the variants apart: the child's size and pace, the vans that hide the pedestrian, the wide gap."""
    w = scenario("child_between_parked_13", 1)
    c = [a for a in w.actors_at(w.t0 + w.spec["collision_t"] - 0.5) if a["role"] == "crossing_ped"][0]
    assert c["H"] == pytest.approx(1.15) and c["speed"] == pytest.approx(2.2, abs=0.1)
    w = scenario("ped_hidden_13", 1)
    t = w.t0 + w.spec["collision_t"] - 4.0
    p = [a for a in w.actors_at(t) if a["role"] == "crossing_ped"][0]
    vans = [a for a in w.actors_at(t) if a["kind"] == "truck" and abs(a["x"] - p["x"]) < 6.0]
    assert len(vans) == 2 and all(v["H"] == pytest.approx(2.6) for v in vans)
    assert p["lane_offset"] < min(v["lane_offset"] + v["W"] / 2 for v in vans) - 1.0     # behind their road-side faces
    w = scenario("ped_early_13", 1)
    t = w.t0 + w.spec["collision_t"] - 4.0
    p = [a for a in w.actors_at(t) if a["role"] == "crossing_ped"][0]
    assert all(abs(a["x"] - p["x"]) > 8.0 for a in w.actors_at(t) if a["role"] == "parked")


@pytest.mark.parametrize("seed", (1, 2, 3))
@pytest.mark.parametrize("name", PED_CONTROL_SCENARIOS)
def test_ped_control_truth(name, seed):
    """The control's pedestrian never moves and is never in the strip; the car passes it."""
    w = scenario(name, seed)
    sp = w.spec
    assert not sp.get("collision_t")
    seen = 0
    for t in np.arange(0.0, sp["T"], 0.1):
        for a in w.actors_at(w.t0 + t):
            assert abs(a["lane_offset"]) - a["W"] / 2 >= HALF + 0.15, (t, a["role"], a["lane_offset"])
            if a["role"] == "kerb_ped":
                seen += 1
                assert not a["moving"]
    assert seen > 50
    assert not [a for a in w.actors_at(w.t0 + sp["T"] - 0.2) if a["role"] == "kerb_ped" and a["x"] > 0.0]   # passed
