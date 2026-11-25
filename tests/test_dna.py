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
import dna  # noqa: E402
import web  # noqa: E402

PINS = json.loads(Path(ROOT, "web.json").read_text())


def path(p):
    return os.path.join(ROOT, p)


class Samples(unittest.TestCase):
    def test_pinned_sequences(self):
        for s in PINS["dna"]:
            blob = Path(path(s["web"]["path"])).read_bytes()
            seq = dna.read_fasta(path(s["path"]))
            self.assertEqual(dna.encipher(blob), seq)  # D1
            self.assertEqual(dna.decipher(seq), blob)  # D2
            self.assertEqual(web.root(blob), s["web"]["root"])
            self.assertEqual(len(seq), s["nt"])
            self.assertEqual(dna.seq_root(seq), s["sequence"])  # D6
            self.assertEqual(dna.longest_run(seq), 1)  # D3
            self.assertEqual(round(dna.gc(seq), 4), s["gc"])

    def test_pinned_cgr(self):
        for s in PINS["dna"]:
            c = s["cgr"]
            seq = dna.read_fasta(path(s["path"]))
            png = Path(path(c["path"])).read_bytes()
            self.assertEqual(hashlib.sha256(png).hexdigest(), c["sha256"])
            img = dna.fcgr(seq, c["k"])
            self.assertTrue(np.array_equal(web.read_png(path(c["path"])), img))
            self.assertTrue(np.array_equal(img > 0, dna.allowed(c["k"]) > 0))  # D5


class Cipher(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.blob = web.pack((5, 7), 3, 8, 30, [rng.normal(0, 0.3, s) for s in web.shapes(2, 3, 8)])
        self.seq = dna.encipher(self.blob)

    def refused(self, seq, code):
        with self.assertRaises(web.Refusal) as ctx:
            dna.decipher(seq)
        self.assertEqual(ctx.exception.code, code)

    @staticmethod
    def word(v, prev):
        out, p = [], dna.BASES.index(prev)
        for t in np.base_repr(v, 3).zfill(dna.WORD_NT):
            p = (p + 1 + int(t)) % 4
            out.append(dna.BASES[p])
        return "".join(out)

    def test_round_trip(self):  # D1, D2
        self.assertEqual(len(self.seq), dna.WORD_NT * len(self.blob) // 2)
        self.assertEqual(dna.decipher(self.seq), self.blob)
        self.assertEqual(dna.longest_run(self.seq), 1)
        self.assertNotEqual(self.seq[0], dna.START)

    def test_word_extremes(self):
        for v in (0, 0xFFFF):
            out, q = 0, dna.BASES.index(dna.START)
            for b in self.word(v, dna.START):
                c = dna.BASES.index(b)
                out, q = 3 * out + (c - q - 1) % 4, c
            self.assertEqual(out, v)

    def test_only_webs_are_enciphered(self):
        with self.assertRaises(web.Refusal) as ctx:
            dna.encipher(b"nope" + self.blob[4:])
        self.assertEqual(ctx.exception.code, "bad_magic")

    def test_refusals(self):  # D4
        self.refused("", "bad_length")
        self.refused(self.seq[:-1], "bad_length")
        self.refused(self.seq[:10] + "N" + self.seq[11:], "bad_base")
        self.refused(self.seq[:10] + self.seq[9] + self.seq[11:], "homopolymer")
        self.refused(dna.START + self.seq[1:], "homopolymer")
        self.refused(self.seq[: -dna.WORD_NT], "truncated")
        self.refused(self.seq + self.word(0, self.seq[-1]), "trailing_bytes")
        self.refused("A" * dna.WORD_NT * (web.MAX_BYTES // 2 + 1), "too_large")
        self.refused(self.word(3**dna.WORD_NT - 1, dna.START) + self.seq[dna.WORD_NT :], "bad_word")

    def test_substitution_is_local(self):
        i = 5 * dna.WORD_NT + 3
        b = self.seq[i]
        for c in dna.BASES:
            if c not in (b, self.seq[i - 1], self.seq[i + 1]):
                break
        try:
            out = dna.decipher(self.seq[:i] + c + self.seq[i + 1 :])
        except web.Refusal:
            return
        diff = np.flatnonzero(np.frombuffer(out, np.uint8) != np.frombuffer(self.blob, np.uint8))
        self.assertTrue(set(diff // 2) <= {5, 6})

    def test_fasta_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.fa")
            dna.write_fasta(p, self.seq, "x")
            self.assertEqual(dna.read_fasta(p), self.seq)
            with self.assertRaises(FileExistsError):
                dna.write_fasta(p, self.seq, "x")

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            w, fa, back = (os.path.join(d, n) for n in ("x.web", "x.fa", "y.web"))
            Path(w).write_bytes(self.blob)
            self.assertEqual(dna.main(["encipher", w, fa]), 0)
            self.assertEqual(dna.main(["encipher", w, fa]), 1)
            self.assertEqual(dna.main(["decipher", fa, back, "--root", "0" * 64]), 1)
            self.assertEqual(dna.main(["decipher", fa, back, "--root", web.root(self.blob)]), 0)
            self.assertEqual(Path(back).read_bytes(), self.blob)
            self.assertEqual(dna.main(["stats", fa, "--salt", "00" * 32]), 0)
            self.assertEqual(dna.main(["cgr", fa, os.path.join(d, "c.png"), "--k", "11"]), 1)
            self.assertEqual(dna.main(["cgr", fa, w]), 1)
            self.assertEqual(Path(w).read_bytes(), self.blob)


class Cgr(unittest.TestCase):
    def test_rare_kmer_is_present(self):
        img = dna.fcgr("AC" * 600 + "G", 1)
        self.assertEqual(int((img > 0).sum()), 3)
        self.assertEqual(int(img.max()), 255)


class Bases(unittest.TestCase):
    def test_salted_root(self):
        salt = bytes(range(32))
        self.assertEqual(dna.seq_root("ACGT", salt), hashlib.sha256(salt + b"ACGT").hexdigest())


if __name__ == "__main__":
    unittest.main()
