import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import dna  # noqa: E402
import loop  # noqa: E402
import web  # noqa: E402

PINS = json.loads(Path(ROOT, "web.json").read_text())


def path(p):
    return Path(ROOT, p)


def rehead(text):
    blocks = loop.split(text)[0]
    return "".join(blocks) + "%head " + loop.chain(blocks) + "\n"


def refuses(test, code, fn, *args):
    with test.assertRaises(web.Refusal) as e:
        fn(*args)
    test.assertEqual(e.exception.code, code)


class Samples(unittest.TestCase):
    def test_pinned_loops(self):
        for s in PINS["dna"]:
            pin, seq = s["loop"], dna.read_fasta(path(s["path"]))
            text = path(pin["path"]).read_text()
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), pin["sha256"])
            plan = json.loads(path(pin["plan"]).read_text())
            self.assertEqual(loop.run(seq, plan), text)  # D14
            verdicts = loop.verify(text, seq)
            self.assertEqual({str(i): v for i, v in verdicts}, pin["verdicts"])  # D16
            self.assertEqual(loop.head(text), pin["head"])  # D15


class Loop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seq = dna.read_fasta(path("samples/astronaut-64.fa"))
        cls.text = path("samples/astronaut-64.loop.txt").read_text()

    def test_head_is_chain(self):  # D15
        blocks = self.text.split("%head ")[0]
        h = hashlib.sha256(b"loop1").digest()
        for b in [blk for blk in blocks.replace("\n%frame ", "\n\0%frame ").split("\0")]:
            h = hashlib.sha256(h + b.encode()).digest()
        self.assertEqual(loop.head(self.text), h.hex())

    def test_tamper_is_refused(self):
        bad = self.text.replace("verdict=refutes", "verdict=supports")
        refuses(self, "bad_head", loop.verify, bad, self.seq)
        refuses(self, "mismatch", loop.verify, rehead(bad), self.seq)
        lines = self.text.split("\n")
        i = next(n for n, line in enumerate(lines) if line.startswith("SWS |"))
        lines[i] = lines[i].replace("#", "@", 1)
        refuses(self, "mismatch", loop.verify, rehead("\n".join(lines)), self.seq)
        refuses(self, "bad_head", loop.verify, self.text.replace("%head a", "%head b"), self.seq)
        refuses(self, "bad_loop", loop.verify, rehead(self.text.replace("%frame 2 ", "%frame 3 ")), self.seq)

    def test_verdict_is_computed(self):  # D16
        plan = [{"hypothesis": {"expect": "refused:tie", "k": 1}, "act": {"op": "lose", "file": 0, "oligos": "0"}}]
        text = loop.run(self.seq, plan)
        self.assertIn("outcome=refused:gap verdict=refutes", text)
        self.assertEqual(loop.verify(text, self.seq), [(2, "refutes")])

    def test_damage_is_seeded(self):
        oligos = dna.pool(self.seq, 0)
        a, b = loop.damage(oligos, 7, 50, 1, 500), loop.damage(oligos, 7, 50, 1, 500)
        self.assertEqual(a, b)
        self.assertNotEqual(a, loop.damage(oligos, 8, 50, 1, 500))
        self.assertEqual(loop.damage(oligos, 0, 0, 0, 0), oligos)

    def test_reads_pinned_by_hash(self):
        reads = dna.pool(self.seq, 0)[::-1]
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / "reads.fa").write_text("".join(f">r{i}\n{r}\n" for i, r in enumerate(reads)))
            h = hashlib.sha256((base / "reads.fa").read_bytes()).hexdigest()
            plan = [{"hypothesis": {"expect": "intact", "k": 2}, "act": {"op": "reads", "file": 0, "sha256": h, "path": "reads.fa"}}]
            text = loop.run(self.seq, plan, base)
            self.assertEqual(loop.verify(text, self.seq, base), [(2, "supports")])
            (base / "reads.fa").write_text(">r0\nACGT\n")
            refuses(self, "hash_drift", loop.verify, text, self.seq, base)
            plan[0]["act"]["path"] = "../reads.fa"
            refuses(self, "bad_loop", loop.run, self.seq, plan, base)

    def test_bad_plans(self):
        for step in (
            {"think": {"view": "cgr", "k": 7}},
            {"think": {"view": "web", "size": 0}},
            {"hypothesis": {"expect": "maybe", "k": 1}, "act": {"op": "lose", "file": 0, "oligos": "0"}},
            {"hypothesis": {"expect": "intact", "k": 1}, "act": {"op": "lose", "file": 0, "oligos": "3,1"}},
            {"hypothesis": {"expect": "intact", "k": 1}, "act": {"op": "lose", "file": 9, "oligos": "1"}},
            {"hypothesis": {"expect": "intact", "k": 1}, "act": {"op": "burn", "file": 0}},
            {"hypothesis": {"expect": "intact", "k": 1}, "act": {"op": "lose", "file": 0, "oligos": "-1"}},
            {"hypothesis": {"expect": "refused:gapp", "k": 1}, "act": {"op": "lose", "file": 0, "oligos": "0"}},
            {"hypothesis": {"expect": "intact", "k": 1}, "act": {"op": "reads", "file": 0, "sha256": "0" * 64, "path": "a b.fa"}},
        ):
            with self.subTest(step=step), self.assertRaises(web.Refusal):
                loop.run(self.seq, [step])

    def test_view_is_text(self):
        v = loop.view(path("samples/astronaut-64.web").read_bytes(), 8)
        rows = [line for line in v if line.startswith("|")]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(len(r) == 10 and set(r[1:-1]) <= set(loop.VIEW_RAMP) for r in rows))


if __name__ == "__main__":
    unittest.main()
