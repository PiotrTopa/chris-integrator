"""The smoother's cost knobs (radar/tracker.py `smooth_every_s`, `smooth_for`; chris_native.c `Hyp.after`).

Defaults keep the old behaviour (every sighting of every object is smoothed: the goldens of tests/test_native.py and
tests/test_radar_inputs.py hold it bit for bit). Frozen here on the simulator's `queue` scenario, the radar alone:
what each knob changes and what it leaves."""
import unittest

from chris_integrator.radar.tracker import CFG
from chris_integrator.sim import radar_score as RS


def run(tracker=None):
    keep = RS.KEEP
    RS.KEEP = keep + ("vs_age_s",)
    try:
        return RS.run("queue", seed=1, cfg={"tracker": tracker} if tracker else None)["frames"]
    finally:
        RS.KEEP = keep


def strip(objs):
    return [{k: v for k, v in o.items() if k not in ("vs", "vs_age_s")} for o in objs]


def refreshes(frames):
    """(object-frames with a smoothed velocity, how often one changed, the oldest one put out (s))."""
    n, changes, oldest, last = 0, 0, 0.0, {}
    for f in frames:
        for o in f["objects"]:
            if o["vs"] is None:
                continue
            n += 1
            oldest = max(oldest, o["vs_age_s"])
            if last.get(o["id"]) != o["vs"]:
                changes += 1
                last[o["id"]] = o["vs"]
    return n, changes, oldest


class SmoothKnobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = run()

    def test_defaults_are_the_old_behaviour(self):
        self.assertEqual((CFG["smooth_every_s"], CFG["smooth_for"]), (0.0, "all"))
        same = run({"smooth_every_s": 0.0, "smooth_for": "all"})
        self.assertEqual([f["objects"] for f in same], [f["objects"] for f in self.base])
        self.assertEqual(refreshes(self.base), (1917, 1032, 2.02))

    def test_smooth_every_keeps_the_smoothed_velocity_and_says_its_age(self):
        r = run({"smooth_every_s": 0.6})
        n, changes, oldest = refreshes(r)
        self.assertEqual(n, 1917)                                    # every object-frame still has one
        self.assertEqual(changes, 992)                               # ... refreshed less often (1032 without)
        self.assertEqual(oldest, 2.32)                               # ... and at most the knob's time older (2.02)
        self.assertLessEqual(oldest, 2.02 + 0.6)
        # the tracker itself moves little: frames whose objects are the same but for `vs`
        same = sum(1 for a, b in zip(self.base, r) if strip(a["objects"]) == strip(b["objects"]))
        self.assertEqual((same, len(r)), (745, 801))

    def test_smooth_for_read_leaves_the_moving_without_and_the_tracker_as_it_was(self):
        r = run({"smooth_for": "read"})
        self.assertEqual(refreshes(r)[0], 837)                       # 1917 with every object smoothed
        for a, b in zip(self.base, r):
            self.assertEqual(strip(a["objects"]), strip(b["objects"]))       # everything but `vs`: identical
            for o, q in zip(a["objects"], b["objects"]):
                if o["motion"] != "moving":
                    self.assertEqual(q["vs"], o["vs"])               # who is not moving keeps its smoothed velocity
        moving = [o["vs"] is None for f in r for o in f["objects"] if o["motion"] == "moving"]
        self.assertEqual((sum(moving), len(moving)), (1080, 1100))   # (20: the frames before their next sighting)


if __name__ == "__main__":
    unittest.main()
