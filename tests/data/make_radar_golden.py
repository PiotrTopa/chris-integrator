#!/usr/bin/env python3
"""Records the radar's frozen results anew, after a deliberate change of its behaviour:

    python3 tests/data/make_radar_golden.py

writes `tests/data/radar_digests.json`: the sha256 of the radar's messages on the simulated drives of
`tests/test_native.py::RADAR_DIGEST_RUNS` (held byte for byte on x86-64 by `test_radar_messages_digests`).
Run it on the development machine (x86-64, glibc) with the native core built (`sh native/build.sh`), and say in
the commit why the results changed."""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))


def main():
    import test_native as T
    got = T._radar_digests()
    path = os.path.join(HERE, "tests", "data", "radar_digests.json")
    with open(path, "w") as f:
        json.dump(got, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote", path)
    for k, v in sorted(got.items()):
        print(" ", k, v)


if __name__ == "__main__":
    main()
