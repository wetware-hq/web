import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import web  # noqa: E402

PINS = json.loads(Path(ROOT, "web.json").read_text())


def path(p):
    return os.path.join(ROOT, p)


class Samples(unittest.TestCase):
    def test_pinned_webs(self):
        for w in PINS["webs"]:
            blob = Path(path(w["path"])).read_bytes()
            media = web.read_png(path(w["media"]["path"]))
            self.assertEqual(web.root(blob), w["root"])  # W3
            self.assertEqual(len(blob), w["bytes"])
            self.assertEqual(hashlib.sha256(Path(path(w["media"]["path"])).read_bytes()).hexdigest(), w["media"]["sha256"])  # W5
            self.assertLess(abs(web.psnr(web.decode(blob), media) - w["psnr"]), 0.01)  # W4
            self.assertTrue(np.array_equal(web.decode(blob), web.decode(blob)))  # W2

    def test_header_matches_pin(self):
        for w in PINS["webs"]:
            u = web.unpack(Path(path(w["path"])).read_bytes())
            h = w["header"]
            self.assertEqual((list(u["grid"]), u["d_out"], u["hidden"], u["omega0"]), (h["grid"], h["channels"], h["hidden"], h["omega0"]))


class Format(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.params = [rng.normal(0, 0.3, s) for s in web.shapes(2, 3, 8)]
        self.blob = web.pack((5, 7), 3, 8, 30, self.params)

    def refused(self, blob, code):
        with self.assertRaises(web.Refusal) as ctx:
            web.unpack(blob)
        self.assertEqual(ctx.exception.code, code)

    def test_canonical(self):  # W1
        u = web.unpack(self.blob)
        self.assertEqual(web.pack(u["grid"], u["d_out"], u["hidden"], u["omega0"], u["params"]), self.blob)
        self.assertEqual(len(self.blob), 12 + 2 * 2 + 2 * (2 * 8 + 8 + 64 + 8 + 24 + 3))
        self.assertEqual(web.decode(self.blob).shape, (5, 7, 3))

    def test_refusals(self):
        self.refused(b"nope" + self.blob[4:], "bad_magic")
        self.refused(self.blob[:-1], "truncated")
        self.refused(self.blob + b"\x00", "trailing_bytes")
        self.refused(self.blob[:4] + b"\x02" + self.blob[5:], "bad_header")
        self.refused(self.blob[:-2] + struct.pack("<e", float("nan")), "bad_weight")
        self.refused(self.blob[:-2] + b"\x00\x80", "bad_weight")  # negative zero
        self.refused(b"\x00" * (web.MAX_BYTES + 1), "too_large")

    def test_negative_zero_dropped(self):
        p = [np.zeros_like(x) - 0.0 for x in self.params]
        web.unpack(web.pack((5, 7), 3, 8, 30, p))

    def test_overflow_refused(self):
        p = [x.copy() for x in self.params]
        p[0][0, 0] = 1e6
        with self.assertRaises(web.Refusal) as ctx:
            web.pack((5, 7), 3, 8, 30, p)
        self.assertEqual(ctx.exception.code, "bad_weight")

    def test_pack_refuses_unreadable_header(self):
        for grid, omega0 in (((1, 7), 30), ((5, 7), 256), ((70000, 2), 30)):
            p = [np.zeros(s) for s in web.shapes(2, 3, 8)]
            with self.assertRaises(web.Refusal) as ctx:
                web.pack(grid, 3, 8, omega0, p)
            self.assertEqual(ctx.exception.code, "bad_header")

    def test_decode_chunks_match(self):
        whole = web.decode(self.blob)
        old, web.DECODE_ROWS = web.DECODE_ROWS, 4
        try:
            self.assertTrue(np.array_equal(web.decode(self.blob), whole))
        finally:
            web.DECODE_ROWS = old


class Media(unittest.TestCase):
    def test_png_round_trip(self):
        img = np.random.default_rng(2).integers(0, 256, (6, 9, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "x.png")
            web.write_png(f, img)
            self.assertTrue(np.array_equal(web.read_png(f), img))
            data = Path(f).read_bytes()
            for cut in (12, len(data) // 2, len(data) - 20):
                Path(f).write_bytes(data[:-cut])
                with self.assertRaises(web.Refusal) as ctx:
                    web.read_png(f)
                self.assertEqual(ctx.exception.code, "bad_media")

    def test_shrink_refuses_bad_factor(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        for factor in (0, -2, 3):
            with self.assertRaises(web.Refusal) as ctx:
                web.shrink(img, factor)
            self.assertEqual(ctx.exception.code, "bad_media")

    def test_encode_small(self):
        img = np.random.default_rng(3).integers(0, 256, (8, 8, 1), dtype=np.uint8)
        blob = web.encode(img, hidden=16, steps=200)
        self.assertEqual(web.decode(blob).shape, img.shape)
        self.assertGreater(web.psnr(web.decode(blob), img), 10.0)


class Cli(unittest.TestCase):
    def test_check_root_mismatch(self):
        w = PINS["webs"][0]
        rc = web.main(["check", path(w["path"]), path(w["media"]["path"]), "--root", "0" * 64])
        self.assertEqual(rc, 1)
        rc = web.main(["check", path(w["path"]), path(w["media"]["path"]), "--root", w["root"], "--min-psnr", "30"])
        self.assertEqual(rc, 0)

    def test_encode_refuses_existing_output(self):
        w = PINS["webs"][0]
        before = Path(path(w["path"])).read_bytes()
        rc = web.main(["encode", path(w["media"]["path"]), path(w["path"]), "--steps", "1"])
        self.assertEqual(rc, 1)
        self.assertEqual(Path(path(w["path"])).read_bytes(), before)


class Index(unittest.TestCase):
    def test_og_checksum(self):  # W8
        text = Path(path("ooc/web.og")).read_text(encoding="utf-8")
        lines = text.split("\n")
        claimed = [l for l in lines if l.startswith("checksum: sha256:")][0].split("sha256:")[1]
        body = "\n".join(l for l in lines if not l.startswith("checksum:"))
        self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), claimed)


if __name__ == "__main__":
    unittest.main()
