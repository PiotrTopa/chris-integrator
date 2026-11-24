"""The prediction and decision layers as plug-ins against golden data frozen before their Python glue moved into
the native core (tests/data/decide_port_golden.json.gz, made by tests/data/make_decide_port_golden.py): recorded
inputs through `predict` and `decide` alone must give the recorded `radar_fut` and `decide` messages.

Numbers are compared to 1e-9 relative (the messages' numbers are rounded, so in practice they are the same text;
the tolerance is for a C library whose last bit differs); everything else must be equal."""
import gzip
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "data"))

import make_decide_port_golden as G  # noqa: E402

RTOL = 1e-9


def differences(a, b, path="", out=None, limit=5):
    """Where two messages differ: [(path, a, b)] (numbers: beyond RTOL relative)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            out.append((path + " keys", list(a), list(b)))
            return out
        for k in a:
            differences(a[k], b[k], "%s.%s" % (path, k), out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path + " len", len(a), len(b)))
            return out
        for i, (u, v) in enumerate(zip(a, b)):
            differences(u, v, "%s[%d]" % (path, i), out, limit)
    elif isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
            and not isinstance(b, bool) and (a == b or abs(a - b) <= RTOL * max(abs(a), abs(b)) or (a != a and b != b))
        if not ok:
            out.append((path, a, b))
    elif a != b or type(a) is not type(b):
        out.append((path, a, b))
    return out


def load():
    with gzip.open(G.OUT, "rb") as f:
        return json.loads(f.read())


class Golden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = load()

    def same(self, got, want, what):
        self.assertEqual(len(got), len(want), what)
        for k, (a, b) in enumerate(zip(got, want)):
            d = differences(G.plain(a), b)
            self.assertEqual(d, [], "%s, message %d" % (what, k))

    def test_predict_and_decide_reproduce_the_recorded_messages(self):
        exact = total = 0
        for c in self.g["cases"]:
            what = "%s seed %d" % (c["scenario"], c["seed"])
            fut, dec = G.replay(c["inputs"])
            self.same(fut[::self.g["keep"]], c["radar_fut"], what + ": radar_fut")
            self.same(dec, c["decide"], what + ": decide")
            self.assertEqual(len(fut), len(c["radar_fut_digest"]), what)
            exact += sum(1 for f, d in zip(fut, c["radar_fut_digest"]) if G.digest(f) == d)
            total += len(fut)
        # every radar_fut message, as text: the same on the machine that recorded them (a C library that differs in
        # a last bit may tip a rounded digit somewhere: then the sampled messages above still hold to RTOL)
        self.assertGreaterEqual(exact, 0.98 * total)
        if sys.platform == "linux" and os.uname().machine == "x86_64":
            self.assertEqual(exact, total)

    def test_the_degraded_futures_reproduce_the_recorded_decisions(self):
        for c in self.g["cases"]:
            _, dec = G.replay(c["inputs"], {"futures": "cv"})
            self.same(dec, c["decide_cv"], "%s seed %d: decide, futures cv" % (c["scenario"], c["seed"]))

    def test_decide_is_the_same_when_nobody_takes_radar_fut(self):
        """`radar_fut` goes to `decide` as arrays; its message is built only for a consumer that reads it."""
        for c in self.g["cases"][:3]:
            fut, dec = G.replay(c["inputs"], lazy=True)
            self.assertEqual(fut, [])
            self.same(dec, c["decide"], "%s: decide without a radar_fut consumer" % c["scenario"])

    def test_decide_from_recorded_radar_fut_messages_alone(self):
        """The `radar_fut` messages (as a recording holds them: JSON) in place of `radar`, no prediction layer:
        the same decisions, through the message's adapter."""
        from chris_integrator.mixer import MessageClock, Mixer, StreamSpec
        from chris_integrator.plugins.decide import Decide
        for c in self.g["cases"]:
            for companion in (False, True):
                m = Mixer(MessageClock())
                names = [s for s in G.INPUTS] + ["radar_fut"]
                ch = m.add_input("golden", [StreamSpec(s, "raw", "sample") for s in names])
                m.add(Decide({"rate_hz": 1000.0}))
                dec = []
                m.add_output("out", lambda e: dec.append(e.msg), streams=("decide",))
                fut = iter(json.loads(json.dumps(G.replay(c["inputs"])[0])))
                for stream, msg in c["inputs"]:
                    if stream == "radar" and companion:
                        ch.deliver("radar_fut", next(fut))        # the companion first, as the predictor emits it
                        ch.deliver("radar", msg)
                    elif stream == "radar":
                        ch.deliver("radar", dict(next(fut), type="radar"))
                    else:
                        ch.deliver(stream, msg)
                self.same(dec, c["decide"], "%s: decide from radar_fut %s" % (
                    c["scenario"], "as the companion" if companion else "in place of radar"))


if __name__ == "__main__":
    unittest.main()
