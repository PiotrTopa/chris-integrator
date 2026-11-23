"""The radar against the reviewed truth of recorded drives (tools/radar_truth_eval.py, docs/RADAR_TRUTH.md).

The label files are in the repository; the recordings are not. Skipped where the session directories are absent
(set RADAR_TRUTH_SESSIONS to the directory that holds them)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import radar_truth_eval as E  # noqa: E402
import radar_truth_lib as L  # noqa: E402

SESSIONS = [(n, s, lab) for n, s, lab in E.find_sessions(E.SESSIONS_ROOT)
            if os.path.isfile(os.path.join(s, "messages.jsonl.gz"))]


class Labels(unittest.TestCase):
    def test_label_files_are_well_formed(self):
        files = E.find_sessions(E.SESSIONS_ROOT)
        self.assertTrue(files, "no label files in data/radar_truth")
        for name, _, lab in files:
            for lb in E.load_labels(lab):
                self.assertEqual(lb["session"], name)
                self.assertIn(lb["kind"], ("object", "threat", "in_path"))
                self.assertIn(lb["confidence"], ("sure", "likely", "guess"))
                self.assertTrue(lb["anchor"] and lb["t0"] <= lb["t1"])
                if lb["kind"] == "object":
                    self.assertIn(lb["truth"], L.REAL + ("not_an_object", "cannot_tell"))
                else:
                    self.assertIn(lb["justified"], (True, False, "cannot_tell"))


@unittest.skipUnless(SESSIONS, "the recorded sessions are not on this machine (RADAR_TRUTH_SESSIONS)")
class RealDrives(unittest.TestCase):
    def test_gate_holds(self):
        results = [E.score_session(s, E.load_labels(lab)) for _, s, lab in SESSIONS]
        tot = E.totals(results)
        broken = E.gate(tot)
        self.assertFalse(broken, E.summary(results, tot, broken))


if __name__ == "__main__":
    unittest.main()
