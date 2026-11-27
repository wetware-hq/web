import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import dna  # noqa: E402

PINS = json.loads(Path(ROOT, "web.json").read_text())
BASES = re.compile(r"[ACGT]{12,}")


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode:
        raise unittest.SkipTest("not a git checkout")
    return [Path(ROOT, p) for p in out.stdout.split()]


def hexes(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v):
                yield k, v
            yield from hexes(v)
    elif isinstance(node, list):
        for v in node:
            yield from hexes(v)


class Stack(unittest.TestCase):
    def test_s1_every_sequence_is_the_primitive(self):
        for entry in PINS["dna"]:
            seq = dna.read_fasta(Path(ROOT, entry["path"]))
            self.assertRegex(seq, r"^[ACGT]+$")
            for oligo in dna.read_pool(Path(ROOT, entry["pool"]["path"])):
                self.assertRegex(oligo, r"^[ACGT]+$")

    def test_s2_one_root_rule(self):
        for entry in PINS["dna"]:
            seq = dna.read_fasta(Path(ROOT, entry["path"]))
            self.assertEqual(entry["sequence"], hashlib.sha256(seq.encode("ascii")).hexdigest())
            self.assertEqual(entry["sequence"], dna.seq_root(seq))

    def test_s3_no_value_under_two_names(self):
        names = {}
        for key, value in hexes(PINS):
            names.setdefault(value, set()).add(key)
        crossed = {v: k for v, k in names.items() if len(k) > 1}
        self.assertEqual(crossed, {})

    def test_s4_bases_stay_in_sequence_files(self):
        for p in tracked():
            if p.suffix in {".fa", ".dna", ".png", ".web", ".tmap"}:
                continue
            self.assertIsNone(BASES.search(p.read_text(errors="ignore")), p.name)


if __name__ == "__main__":
    unittest.main()
