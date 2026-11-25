"""Reference `dna1` cipher: a `.web` file as one synthetic-DNA sequence, and back.

Needs only Python 3.10+, numpy and `ref/web.py`. Both directions are normative and exact
(docs/follow-agent-contract.md §9): decipher(encipher(web)) == web for every accepted web.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import web  # noqa: E402

BASES = "ACGT"
START = "A"  # the base before the first base; the sequence never starts with it
WORD_NT = 11  # one binary16 weight (2 bytes) is 11 trits, since 3**11 = 177,147 >= 2**16
TAG = b"dna1"
LINE = 80  # FASTA line width; layout only, never part of the sequence
CGR_CORNER = {"A": (0, 0), "C": (0, 1), "G": (1, 1), "T": (1, 0)}  # (x, y), Jeffrey 1990


def keystream(n):
    """Public whitening bytes: sha256(TAG || counter) blocks. Not a secret, not encryption."""
    out = b"".join(hashlib.sha256(TAG + i.to_bytes(8, "big")).digest() for i in range((n + 31) // 32))
    return np.frombuffer(out[:n], np.uint8)


def whiten(blob):
    return (np.frombuffer(blob, np.uint8) ^ keystream(len(blob))).tobytes()


def encipher(blob):
    """Web bytes -> sequence over ACGT with no base repeated (Goldman et al. 2013 rotating code)."""
    web.unpack(blob)
    words = np.frombuffer(whiten(blob), "<u2")
    out, prev = [], BASES.index(START)
    for v in words.tolist():
        trits = []
        for _ in range(WORD_NT):
            v, t = divmod(v, 3)
            trits.append(t)
        for t in reversed(trits):  # most significant trit first
            prev = (prev + 1 + t) % 4
            out.append(BASES[prev])
    return "".join(out)


def decipher(seq):
    """Sequence -> web bytes, refusing anything encipher could not have written."""
    if len(seq) > WORD_NT * web.MAX_BYTES // 2:
        raise web.Refusal("too_large", f"{len(seq)} nt")
    if not seq or len(seq) % WORD_NT:
        raise web.Refusal("bad_length", f"{len(seq)} nt is not a positive multiple of {WORD_NT}")
    words, prev = [], BASES.index(START)
    for i in range(0, len(seq), WORD_NT):
        v = 0
        for j, b in enumerate(seq[i : i + WORD_NT]):
            cur = BASES.find(b)
            if cur < 0:
                raise web.Refusal("bad_base", f"{b!r} at {i + j}")
            if cur == prev:
                raise web.Refusal("homopolymer", f"{b * 2} at {i + j - 1}")
            v = 3 * v + (cur - prev - 1) % 4
            prev = cur
        if v > 0xFFFF:
            raise web.Refusal("bad_word", f"word {i // WORD_NT} is {v}")
        words.append(v)
    blob = whiten(np.array(words, "<u2").tobytes())
    web.unpack(blob)
    return blob


def seq_root(seq, salt=b""):
    """sha256(salt || ASCII bases): with a 32-byte salt this is the `Bases` token root."""
    return hashlib.sha256(salt + seq.encode("ascii")).hexdigest()


def gc(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)


def longest_run(seq):
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def fcgr(seq, k):
    """Frequency chaos-game image: k-mer counts on a 2^k x 2^k grid, row 0 at the top.

    The newest base of a k-mer picks the quadrant, as in the CGR, where each point moves halfway
    to its base's corner. Returns uint8 gray levels, 255 at the most frequent k-mer, 0 if absent.
    """
    n = 1 << k
    counts = np.zeros((n, n), np.int64)
    for i in range(len(seq) - k + 1):
        x = y = 0
        for bit, b in enumerate(seq[i : i + k]):
            cx, cy = CGR_CORNER[b]
            x |= cx << bit
            y |= cy << bit
        counts[n - 1 - y, x] += 1
    return np.rint(counts * 255.0 / counts.max()).astype(np.uint8)[:, :, None]


def allowed(k):
    """Gray image of the k-mers a dna1 sequence may contain (no base repeated): 255, else 0."""
    n = 1 << k
    img = np.zeros((n, n, 1), np.uint8)
    for code in range(4**k):
        kmer = "".join(BASES[(code >> (2 * i)) & 3] for i in range(k))
        if all(a != b for a, b in zip(kmer, kmer[1:])):
            img[:, :, 0] |= np.where(fcgr(kmer, k)[:, :, 0] > 0, 255, 0).astype(np.uint8)
    return img


def read_fasta(path):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    if not lines or not lines[0].startswith(">"):
        raise web.Refusal("bad_base", "not a FASTA record")
    if any(ln.startswith(">") for ln in lines[1:]):
        raise web.Refusal("bad_base", "more than one FASTA record")
    return "".join(ln.strip() for ln in lines[1:])


def write_fasta(path, seq, header):
    body = "\n".join(seq[i : i + LINE] for i in range(0, len(seq), LINE))
    with open(path, "x", encoding="ascii") as f:
        f.write(f">{header}\n{body}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dna", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encipher", help="write a web as a dna1 FASTA record")
    e.add_argument("web")
    e.add_argument("out")
    d = sub.add_parser("decipher", help="rebuild the web bytes from a dna1 FASTA record")
    d.add_argument("fasta")
    d.add_argument("out")
    d.add_argument("--root", help="refuse unless the rebuilt web hashes to this root")
    s = sub.add_parser("stats", help="print length, sequence root, GC and longest run")
    s.add_argument("fasta")
    s.add_argument("--salt", help="hex salt; prints the Bases root sha256(salt || bases)")
    c = sub.add_parser("cgr", help="draw the frequency chaos-game image of a sequence")
    c.add_argument("fasta")
    c.add_argument("out")
    c.add_argument("--k", type=int, default=6)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "encipher":
            blob = Path(a.web).read_bytes()
            seq = encipher(blob)
            if Path(a.out).exists():
                raise web.Refusal("exists", a.out)
            write_fasta(a.out, seq, f"web1:{web.root(blob)} dna1 nt={len(seq)}")
            print(seq_root(seq), len(seq))
        elif a.cmd == "decipher":
            blob = decipher(read_fasta(a.fasta))
            if a.root and web.root(blob) != a.root:
                raise web.Refusal("root_mismatch", web.root(blob))
            if Path(a.out).exists():
                raise web.Refusal("exists", a.out)
            with open(a.out, "xb") as f:
                f.write(blob)
            print(web.root(blob), len(blob))
        elif a.cmd == "stats":
            seq = read_fasta(a.fasta)
            line = f"nt {len(seq)} sequence {seq_root(seq)} gc {gc(seq):.4f} run {longest_run(seq)}"
            if a.salt:
                line += f" bases {seq_root(seq, bytes.fromhex(a.salt))}"
            print(line)
        elif a.cmd == "cgr":
            if not 1 <= a.k <= 10:
                raise web.Refusal("bad_header", "k must be 1-10")
            img = fcgr(read_fasta(a.fasta), a.k)
            if Path(a.out).exists():
                raise web.Refusal("exists", a.out)
            web.write_png(a.out, img)
    except web.Refusal as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
