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
