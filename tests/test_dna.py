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

    def test_fasta_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.fa")
            for body, code in ((b">x\n", "bad_length"), (b">x\nACGNT\n", "bad_base"), (b">x\nAC\xc3\xa9\n", "bad_base")):
                Path(p).write_bytes(body)
                with self.assertRaises(web.Refusal) as ctx:
                    dna.read_fasta(p)
                self.assertEqual(ctx.exception.code, code)
                self.assertEqual(dna.main(["stats", p]), 1)

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
            self.assertEqual(dna.main(["stats", fa, "--salt", "00" * 31]), 1)
            self.assertEqual(dna.main(["stats", fa, "--salt", "zz"]), 1)
            self.assertEqual(Path(w).read_bytes(), self.blob)


class Pool(unittest.TestCase):
    def setUp(self):
        self.seq = dna.read_fasta(path("samples/astronaut-64.fa"))
        self.oligos = dna.pool(self.seq)

    def unpooled(self, reads):
        return dna.unpool(reads)[0]

    def refused(self, reads, code):
        with self.assertRaises(web.Refusal) as ctx:
            dna.unpool(reads)
        self.assertEqual(ctx.exception.code, code)

    def test_pinned_pool(self):  # D7
        for s in PINS["dna"]:
            p = s["pool"]
            self.assertEqual(hashlib.sha256(Path(path(p["path"])).read_bytes()).hexdigest(), p["sha256"])
            reads = dna.read_pool(path(p["path"]))
            self.assertEqual(reads, dna.pool(dna.read_fasta(path(s["path"])), p["file"]))
            self.assertEqual(len(reads), p["oligos"])

    def test_oligo_shape(self):  # D8
        for o in self.oligos:
            self.assertEqual(len(o), dna.OLIGO_NT)
            self.assertEqual(dna.longest_run(o), 1)
            self.assertIn(o[0], "AT")
            self.assertIn(o[-1], "CG")
        self.assertEqual(len(self.oligos), (len(self.seq) + dna.LEN_TRITS) // dna.STEP_NT - 3)

    def test_round_trip(self):  # D9
        self.assertEqual(self.unpooled(self.oligos), self.seq)
        rng = np.random.default_rng(0)
        reads = [dna.revcomp(o) if rng.random() < 0.5 else o for o in self.oligos]
        reads = [reads[i] for i in rng.permutation(len(reads))] + ["ACGT" * 30, "N" * dna.OLIGO_NT]
        seq, used, dropped = dna.unpool(reads)
        self.assertEqual((seq, used, dropped), (self.seq, len(self.oligos), 2))
        self.assertEqual(web.root(dna.decipher(seq)), PINS["webs"][0]["root"])

    def test_short_sequence(self):
        rng = np.random.default_rng(3)
        seq = dna.rotate(rng.integers(0, 3, 7).tolist())
        self.assertEqual(len(dna.pool(seq)), 1)
        self.assertEqual(self.unpooled(dna.pool(seq)), seq)

    def test_loss_of_interior_oligos(self):  # D9
        keep = [o for i, o in enumerate(self.oligos) if i % 4 != 1 or i in (1, len(self.oligos) - 2, len(self.oligos) - 1)]
        self.assertLess(len(keep), len(self.oligos))
        self.assertEqual(self.unpooled(keep), self.seq)

    def test_substitutions_are_outvoted(self):  # D10
        reads = list(self.oligos)
        for i in range(3, len(reads) - 3, 7):
            o = reads[i]
            j = 1 + (i * 13) % dna.SEG_NT
            c = next(b for b in dna.BASES if b not in (o[j - 1], o[j], o[j + 1]))
            reads[i] = o[:j] + c + o[j + 1 :]
        self.assertNotEqual(reads, self.oligos)
        self.assertEqual(self.unpooled(reads), self.seq)

    def test_bad_index_is_dropped(self):
        o = self.oligos[5]
        tail = dna.unrotate(o[1 + dna.SEG_NT : -1], o[dna.SEG_NT])
        tail[-1] = (tail[-1] + 1) % 3
        bad = o[: 1 + dna.SEG_NT] + dna.rotate(tail, o[dna.SEG_NT]) + o[-1]
        self.assertIsNone(dna.read_oligo(bad))

    def test_file_ids_share_a_pool(self):
        rng = np.random.default_rng(4)
        other = dna.rotate(rng.integers(0, 3, 300).tolist())
        reads = self.oligos + dna.pool(other, 8)
        self.assertEqual(dna.unpool(reads, 8)[0], other)
        self.assertEqual(self.unpooled(reads), self.seq)
        for f in (dna.pool, dna.unpool):
            with self.assertRaises(web.Refusal) as ctx:
                f(other if f is dna.pool else reads, 9)
            self.assertEqual(ctx.exception.code, "bad_file")

    def test_refusals(self):  # D11
        self.refused([], "gap")
        self.refused(self.oligos[1:], "gap")
        self.refused(self.oligos[:-1], "bad_pool")
        o = self.oligos[0]
        j = 30
        c = next(b for b in dna.BASES if b not in (o[j - 1], o[j], o[j + 1]))
        self.refused([o[:j] + c + o[j + 1 :]] + self.oligos[1:], "tie")

    def test_far_index_is_a_gap(self):
        o = self.oligos[0]
        f = o[1 : 1 + dna.SEG_NT]
        ix = dna.trits(0, dna.FILE_TRITS) + dna.trits(3**dna.INDEX_TRITS - 1, dna.INDEX_TRITS)
        core = f + dna.rotate(ix + [dna.parity(ix)], f[-1])
        far = ("T" if core[0] == "A" else "A") + core + ("G" if core[-1] == "C" else "C")
        self.assertEqual(dna.read_oligo(far)[1], 3**dna.INDEX_TRITS - 1)
        self.refused(self.oligos + [far], "gap")

    def test_plurality_is_not_majority(self):
        p = 60
        true = self.seq[p]
        reads = [self.oligos[0]] + list(self.oligos)

        def sub(i, b):
            o = self.oligos[i]
            j = 1 + (p - i * dna.STEP_NT if i % 2 == 0 else dna.SEG_NT - 1 - (p - i * dna.STEP_NT))
            c = b if i % 2 == 0 else dna.revcomp(b)
            return None if c in (o[j - 1], o[j + 1]) else o[:j] + c + o[j + 1 :]

        for b1 in dna.BASES:
            for b2 in dna.BASES:
                if len({true, b1, b2}) == 3 and sub(1, b1) and sub(2, b2):
                    reads[2], reads[3] = sub(1, b1), sub(2, b2)
                    self.refused(reads, "tie")
                    return
        self.fail("no substitution pair")

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            p, back = os.path.join(d, "p.fa"), os.path.join(d, "s.fa")
            self.assertEqual(dna.main(["pool", path("samples/astronaut-64.fa"), p]), 0)
            self.assertEqual(dna.main(["pool", path("samples/astronaut-64.fa"), p]), 1)
            self.assertEqual(dna.main(["unpool", p, back, "--file", "1"]), 1)
            self.assertEqual(dna.main(["unpool", p, back]), 0)
            self.assertEqual(dna.read_fasta(back), self.seq)
            raw = Path(p).read_bytes().split(b"\n")
            raw[0] += " é".encode()
            raw[3] = raw[3][:5] + "é".encode() + raw[3][7:]
            Path(p).write_bytes(b"\n".join(raw))
            reads = dna.read_pool(p)
            self.assertEqual(dna.unpool(reads)[1:], (len(self.oligos) - 1, 1))


class Bases(unittest.TestCase):
    def test_salted_root(self):
        salt = bytes(range(32))
        self.assertEqual(dna.seq_root("ACGT", salt), hashlib.sha256(salt + b"ACGT").hexdigest())


if __name__ == "__main__":
    unittest.main()
