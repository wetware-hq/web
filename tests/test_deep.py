import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import deep  # noqa: E402
import web  # noqa: E402

PINS = json.loads(Path(ROOT, "web.json").read_text())


def path(p):
    return Path(ROOT, p)


def refuses(test, code, fn, *args):
    with test.assertRaises(web.Refusal) as e:
        fn(*args)
    test.assertEqual(e.exception.code, code)


def negate():
    """A hand-set 1 -> 1 stage: level l -> 255 - l, the same construction as `deep.luma`."""
    e = deep.LUMA_EPS
    return web.pack((deep.LEVELS,), 1, 1, 1, [np.full((1, 1), e), np.zeros(1), np.ones((1, 1)), np.zeros(1), np.full((1, 1), -1 / e), np.zeros(1)])


class Samples(unittest.TestCase):
    def test_pinned_compositions(self):  # C1, C4
        for d in PINS["composition"]["deep"]:
            text = path(d["path"]).read_text("ascii")
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), d["sha256"])
            blobs = [path(s["path"]).read_bytes() for s in d["stages"]]
            self.assertEqual([web.root(b) for b in blobs], [s["root"] for s in d["stages"]])
            self.assertEqual(deep.record(blobs), text)
            self.assertEqual(deep.verify(text, blobs), d["head"])

    def test_luma_stage_is_hand_set(self):
        luma = [s for s in PINS["composition"]["deep"][0]["stages"] if s["path"] == "samples/luma.web"][0]
        self.assertEqual(deep.luma(), path(luma["path"]).read_bytes())
        self.assertEqual(len(deep.luma()), luma["bytes"])

    def test_pinned_fidelity(self):
        d = PINS["composition"]["deep"][0]
        out = deep.compose([path(s["path"]).read_bytes() for s in d["stages"]])[-1]
        src = web.read_png(path("samples/astronaut-64.png")).astype(np.float64)
        want = np.rint(src @ np.array(deep.LUMA)).astype(np.uint8)[..., None]
        self.assertAlmostEqual(web.psnr(out, want), d["fidelity"]["psnr"], delta=0.01)


class Compose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.web = path("samples/astronaut-64.web").read_bytes()
        cls.luma = deep.luma()

    def test_depth_one_is_the_web(self):  # C1
        self.assertTrue(np.array_equal(deep.compose([self.web])[0], web.decode(self.web)))

    def test_step_reads_the_next_decode(self):  # C2
        small = web.pack((4, 4), 1, 1, 1, [np.zeros((2, 1)) + 0.01, np.zeros(1), np.ones((1, 1)), np.zeros(1), np.full((1, 1), 50.0), np.zeros(1)])
        prev = web.decode(small)
        table = web.decode(negate())[:, 0]
        self.assertTrue(np.array_equal(deep.compose([small, negate()])[1][..., 0], table[prev[..., 0]]))

    def test_induction(self):  # C2
        three = deep.compose([self.web, self.luma, negate()])
        for a, b in zip(three, deep.compose([self.web, self.luma])):
            self.assertTrue(np.array_equal(a, b))
        self.assertLessEqual(int(np.abs(three[2].astype(int) + three[1] - 255).max()), 1)

    def test_interface_must_fit(self):  # C3
        refuses(self, "bad_stage", deep.compose, [self.web, negate()])
        refuses(self, "bad_stage", deep.compose, [self.web, self.web])
        refuses(self, "bad_stage", deep.compose, [])

    def test_chain_binds_every_interface(self):  # C4
        text = deep.record([self.web, self.luma, negate()])
        ls = deep.parse(text)[1]
        h = hashlib.sha256(b"deep1").digest()
        for w, _, o in ls:
            h = hashlib.sha256(h + bytes.fromhex(w) + bytes.fromhex(o)).digest()
        self.assertEqual(deep.head(text), h.hex())
        self.assertEqual(ls[1][2], deep.out_root(deep.compose([self.web, self.luma])[1]))

    def test_tamper_is_refused(self):  # C5
        text = deep.record([self.web, self.luma])
        o = deep.parse(text)[1][1][2]
        refuses(self, "bad_head", deep.verify, text.replace(o, "0" * 64), [self.web, self.luma])
        refuses(self, "bad_head", deep.head, text.replace(o, "z" * 64))
        refuses(self, "root_mismatch", deep.verify, text, [self.web, negate()])
        refuses(self, "bad_stage", deep.verify, text, [self.web])
        refuses(self, "bad_deep", deep.verify, text.replace("%stage 1", "%stage 2"), [self.web, self.luma])
        refuses(self, "bad_deep", deep.head, text[:-1])
        refuses(self, "bad_deep", deep.head, "%deep1 stages=x\n%head 0\n")
        lines = text.split("\n")
        lines[0] = "%deep1 stages=2 grid=64x63"
        refuses(self, "mismatch", deep.verify, "\n".join(lines), [self.web, self.luma])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            w, lu, rec, png = (os.path.join(d, n) for n in ("a.web", "l.web", "x.deep", "x.png"))
            Path(w).write_bytes(self.web)
            self.assertEqual(deep.main(["luma", lu]), 0)
            self.assertEqual(deep.main(["luma", lu]), 1)
            self.assertEqual(deep.main(["run", rec, w, lu]), 0)
            self.assertEqual(deep.main(["verify", rec, w, lu]), 0)
            self.assertEqual(deep.main(["verify", rec, lu, w]), 1)
            self.assertEqual(deep.main(["head", rec]), 0)
            self.assertEqual(deep.main(["decode", png, w, lu]), 0)
            self.assertTrue(np.array_equal(web.read_png(png), deep.compose([self.web, self.luma])[1]))
            self.assertEqual(deep.main(["run", os.path.join(d, "y.deep"), w, w]), 1)


if __name__ == "__main__":
    unittest.main()
