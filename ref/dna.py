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
COMPLEMENT = str.maketrans("ACGT", "TGCA")
# pool1: oligos after Goldman et al. 2013
SEG_NT = 100  # payload bases per oligo
STEP_NT = 25  # offset between consecutive oligos, so each base is in up to 4 oligos
LEN_TRITS = 20  # sequence length in the pool trailer; 3**20 > any dna1 length
FILE_TRITS, INDEX_TRITS = 2, 12  # IX = file id, oligo index, parity trit
IX_NT = FILE_TRITS + INDEX_TRITS + 1
OLIGO_NT = 1 + SEG_NT + IX_NT + 1  # 117
# cgr1: the text chaos-game glyph
RAMP = " .:-=+*#%@"  # space = absent; "." = least present ... "@" = most frequent


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


def rotate(trits, prev=START):
    """Trits -> bases, each base the (1 + t)-th successor of the one before (no repeats)."""
    out, p = [], BASES.index(prev)
    for t in trits:
        p = (p + 1 + t) % 4
        out.append(BASES[p])
    return "".join(out)


def unrotate(seq, prev=START):
    """Bases -> trits, the inverse of rotate; refuses a letter outside ACGT or a repeat."""
    out, p = [], BASES.index(prev)
    for i, b in enumerate(seq):
        c = BASES.find(b)
        if c < 0:
            raise web.Refusal("bad_base", f"{b!r} at {i}")
        if c == p:
            raise web.Refusal("homopolymer", f"{b * 2} at {i - 1}")
        out.append((c - p - 1) % 4)
        p = c
    return out


def trits(v, n):
    return [v // 3**j % 3 for j in reversed(range(n))]


def untrits(ts):
    v = 0
    for t in ts:
        v = 3 * v + t
    return v


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


def kmer_counts(seq, k):
    """k-mer counts on the 2^k x 2^k chaos-game grid, row 0 at the top.

    Bit j of a cell's column is base j's x corner and bit j of its height is its y corner, so the
    newest base picks the quadrant, as in the CGR, where each point moves halfway to its base's corner.
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
    return counts


def gray(counts):
    """255 at the most frequent k-mer, at least 1 for any k-mer present, 0 only if absent."""
    img = np.rint(counts * 255.0 / max(int(counts.max()), 1)).astype(np.uint8)
    img[(counts > 0) & (img == 0)] = 1
    return img


def fcgr(seq, k):
    """Frequency chaos-game image of a sequence (Jeffrey 1990) as uint8 gray levels."""
    return gray(kmer_counts(seq, k))[:, :, None]


def allowed(k):
    """Gray image of the k-mers a dna1 sequence may contain (no base repeated): 255, else 0."""
    n = 1 << k
    img = np.zeros((n, n, 1), np.uint8)
    for code in range(4**k):
        kmer = "".join(BASES[(code >> (2 * i)) & 3] for i in range(k))
        if all(a != b for a, b in zip(kmer, kmer[1:])):
            img[:, :, 0] |= np.where(fcgr(kmer, k)[:, :, 0] > 0, 255, 0).astype(np.uint8)
    return img


def revcomp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def parity(ix):
    """Goldman's parity trit: the sum of the odd-numbered (1st, 3rd, ...) file-id and index trits, mod 3."""
    return sum(ix[0::2]) % 3


def pool(seq, file_id=0):
    """dna1 sequence -> 117-nt oligos, after Goldman et al. 2013 (contract §10)."""
    if not 0 <= file_id < 3**FILE_TRITS:
        raise web.Refusal("bad_file", f"file id {file_id} is not 0-{3**FILE_TRITS - 1}")
    t = unrotate(seq)
    if not t:
        raise web.Refusal("bad_length", "empty sequence")
    body = len(t) + LEN_TRITS
    n = max(SEG_NT, -(-body // STEP_NT) * STEP_NT)
    count = (n - SEG_NT) // STEP_NT + 1
    if len(t) >= 3**LEN_TRITS or count > 3**INDEX_TRITS:
        raise web.Refusal("too_large", f"{len(t)} nt")
    s5 = rotate(t + [0] * (n - body) + trits(len(t), LEN_TRITS))
    oligos = []
    for i in range(count):
        f = s5[i * STEP_NT : i * STEP_NT + SEG_NT]
        if i % 2:
            f = revcomp(f)
        ix = trits(file_id, FILE_TRITS) + trits(i, INDEX_TRITS)
        core = f + rotate(ix + [parity(ix)], f[-1])
        oligos.append(("T" if core[0] == "A" else "A") + core + ("G" if core[-1] == "C" else "C"))
    return oligos


def read_oligo(read):
    """One read -> (file id, index, payload in pool order), or None if it fails any oligo check."""
    if len(read) != OLIGO_NT or read.strip(BASES):
        return None
    if read[0] in "CG" and read[-1] in "AT":  # sequenced from the other strand
        read = revcomp(read)
    if read[0] not in "AT" or read[-1] not in "CG" or read[0] == read[1] or read[-1] == read[-2]:
        return None
    f, tail = read[1 : 1 + SEG_NT], read[1 + SEG_NT : -1]
    try:
        ix = unrotate(tail, f[-1])
    except web.Refusal:
        return None
    if ix[-1] != parity(ix[:-1]):
        return None
    i = untrits(ix[FILE_TRITS:-1])
    return untrits(ix[:FILE_TRITS]), i, revcomp(f) if i % 2 else f


def unpool(reads, file_id=0):
    """Reads in any order and orientation -> the dna1 sequence, by majority vote per base.

    Returns (seq, used, dropped): reads that fail an oligo check are dropped, never repaired.
    """
    if not 0 <= file_id < 3**FILE_TRITS:
        raise web.Refusal("bad_file", f"file id {file_id} is not 0-{3**FILE_TRITS - 1}")
    got, dropped = {}, 0
    for r in reads:
        o = read_oligo(r)
        if o is None:
            dropped += 1
        elif o[0] == file_id:
            got.setdefault(o[1], []).append(o[2])
    if not got:
        raise web.Refusal("gap", f"no valid oligo for file {file_id}")
    idx = sorted(got)
    ends = [-1] + [a for a, b in zip(idx, idx[1:]) if b - a > SEG_NT // STEP_NT]
    if idx[0] or len(ends) > 1:
        raise web.Refusal("gap", f"no oligo covers base {0 if idx[0] else ends[1] * STEP_NT + SEG_NT}")
    n = max(got) * STEP_NT + SEG_NT
    lut = np.full(256, 0, np.int64)
    for k, b in enumerate(BASES):
        lut[ord(b)] = k
    votes = np.zeros((n, 4), np.int64)
    cols = np.arange(SEG_NT)
    for i, fs in got.items():
        for f in fs:
            np.add.at(votes, (i * STEP_NT + cols, lut[np.frombuffer(f.encode("ascii"), np.uint8)]), 1)
    top = votes.max(1)
    if (top == 0).any():
        raise web.Refusal("gap", f"no oligo covers base {int(np.argmin(top))}")
    tied = (2 * top <= votes.sum(1)).nonzero()[0]
    if len(tied):
        raise web.Refusal("tie", f"base {int(tied[0])}")
    s5 = "".join(BASES[k] for k in votes.argmax(1))
    t = unrotate(s5)
    size = untrits(t[-LEN_TRITS:])
    if not 0 < size <= n - LEN_TRITS or any(t[size:-LEN_TRITS]) or n != max(SEG_NT, -(-(size + LEN_TRITS) // STEP_NT) * STEP_NT):
        raise web.Refusal("bad_pool", f"trailer says {size} nt in a {n}-nt pool")
    return s5[:size], sum(map(len, got.values())), dropped


def axis(v, k, one, zero):
    return "".join(one if v >> j & 1 else zero for j in range(k))


def counts_root(counts):
    """sha256 of the k-mer counts in grid order (row 0 first), as decimal ASCII joined by spaces."""
    return hashlib.sha256(" ".join(map(str, counts.ravel().tolist())).encode("ascii")).hexdigest()


def glyph(seq, k):
    """cgr1: the chaos-game grid as text an agent reads like a QR code (contract §11).

    Column labels (M = A|C, K = G|T) and row labels (S = C|G, W = A|T) spell each cell's k-mer,
    oldest base first; corners name the bases; the footer hash lets a reader check what it read.
    """
    if not 1 <= k <= 6:
        raise web.Refusal("bad_header", "k must be 1-6")
    counts = kmer_counts(seq, k)
    g, n = gray(counts), 1 << k
    pad = " " * (k + 1)
    cols = [axis(x, k, "K", "M") for x in range(n)]
    lines = [
        f"%cgr1 k={k} nt={len(seq)} peak={int(counts.max())} present={int((counts > 0).sum())}/{n * n} sequence={seq_root(seq)}",
        f'%ramp "{RAMP}"',
    ]
    lines += [pad + " " + "".join(c[j] for c in cols) for j in range(k)]
    lines.append(pad + "C" + "-" * n + "G")
    for r in range(n):
        row = "".join(RAMP[0] if v == 0 else RAMP[1 + (int(v) - 1) * (len(RAMP) - 2) // 254] for v in g[r])
        lines.append(axis(n - 1 - r, k, "S", "W") + " |" + row + "|")
    lines.append(pad + "A" + "-" * n + "T")
    lines.append(f"%counts sha256:{counts_root(counts)}")
    return "\n".join(lines) + "\n"


CELL = {("M", "S"): "C", ("M", "W"): "A", ("K", "S"): "G", ("K", "W"): "T"}


def read_glyph(text):
    """cgr1 text -> {k-mer: ramp character}, refusing a glyph whose frame or labels are off."""
    lines = text.splitlines()
    head = dict(f.split("=", 1) for f in lines[0].split()[1:]) if lines and lines[0].startswith("%cgr1 ") else {}
    try:
        k = int(head["k"])
    except (KeyError, ValueError):
        raise web.Refusal("bad_header", "not a cgr1 glyph") from None
    n = 1 << k
    if not 1 <= k <= 6 or len(lines) != 2 + k + n + 3:
        raise web.Refusal("bad_header", "frame size")
    pad = " " * (k + 1)
    cols = ["".join(lines[2 + j][k + 2 + x] for j in range(k)) for x in range(n)]
    if lines[2 + k] != pad + "C" + "-" * n + "G" or lines[3 + k + n] != pad + "A" + "-" * n + "T":
        raise web.Refusal("bad_header", "corners")
    if cols != [axis(x, k, "K", "M") for x in range(n)]:
        raise web.Refusal("bad_header", "column labels")
    cells = {}
    for r in range(n):
        line = lines[3 + k + r]
        label, row = line[:k], line[k + 2 : -1]
        if label != axis(n - 1 - r, k, "S", "W") or len(row) != n or row.strip(RAMP):
            raise web.Refusal("bad_header", f"row {r}")
        for x, ch in enumerate(row):
            cells["".join(CELL[c, w] for c, w in zip(cols[x], label))] = ch
    return cells


def read_fasta(path):
    try:
        lines = Path(path).read_bytes().decode("ascii").splitlines()
    except UnicodeDecodeError as err:
        raise web.Refusal("bad_base", f"non-ASCII byte at {err.start}") from None
    if not lines or not lines[0].startswith(">"):
        raise web.Refusal("bad_base", "not a FASTA record")
    if any(ln.startswith(">") for ln in lines[1:]):
        raise web.Refusal("bad_base", "more than one FASTA record")
    seq = "".join(ln.strip() for ln in lines[1:])
    if not seq:
        raise web.Refusal("bad_length", "empty sequence")
    bad = seq.strip(BASES)
    if bad:
        raise web.Refusal("bad_base", f"{bad[0]!r} at {seq.index(bad[0])}")
    return seq


def write_fasta(path, seq, header):
    body = "\n".join(seq[i : i + LINE] for i in range(0, len(seq), LINE))
    with open(path, "x", encoding="ascii") as f:
        f.write(f">{header}\n{body}\n")


def read_pool(path):
    """Every record of a FASTA file, as reads; a bad record, even non-ASCII, is kept for unpool to drop."""
    lines = Path(path).read_bytes().decode("ascii", "replace").splitlines()
    reads, cur = [], None
    for ln in lines:
        if ln.startswith(">"):
            cur = []
            reads.append(cur)
        elif cur is None:
            raise web.Refusal("bad_base", "not a FASTA file")
        else:
            cur.append(ln.strip())
    if not reads:
        raise web.Refusal("bad_length", "no FASTA record")
    return ["".join(r) for r in reads]


def write_pool(path, oligos, header):
    with open(path, "x", encoding="ascii") as f:
        f.writelines(f">{header} i={i}\n{o}\n" for i, o in enumerate(oligos))


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
    s.add_argument("--salt", help="32-byte hex salt; prints the Bases root sha256(salt || bases)")
    c = sub.add_parser("cgr", help="draw the frequency chaos-game image of a sequence")
    c.add_argument("fasta")
    c.add_argument("out")
    c.add_argument("--k", type=int, default=6)
    p = sub.add_parser("pool", help="split a dna1 FASTA record into 117-nt pool1 oligos")
    p.add_argument("fasta")
    p.add_argument("out")
    p.add_argument("--file", type=int, default=0, help="file id 0-8, so up to 9 files can share a pool")
    u = sub.add_parser("unpool", help="rebuild the dna1 FASTA record from pool1 reads in any order")
    u.add_argument("reads")
    u.add_argument("out")
    u.add_argument("--file", type=int, default=0)
    g = sub.add_parser("glyph", help="print the cgr1 text chaos-game glyph of a sequence")
    g.add_argument("fasta")
    g.add_argument("--k", type=int, default=3)
    g.add_argument("--out", help="write here instead of stdout")
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
            if a.salt is not None:
                try:
                    salt = bytes.fromhex(a.salt)
                except ValueError:
                    raise web.Refusal("bad_salt", "not hex") from None
                if len(salt) != 32:
                    raise web.Refusal("bad_salt", f"{len(salt)} bytes, a Bases salt is 32")
                line += f" bases {seq_root(seq, salt)}"
            print(line)
        elif a.cmd == "cgr":
            if not 1 <= a.k <= 10:
                raise web.Refusal("bad_header", "k must be 1-10")
            img = fcgr(read_fasta(a.fasta), a.k)
            if Path(a.out).exists():
                raise web.Refusal("exists", a.out)
            web.write_png(a.out, img)
        elif a.cmd == "pool":
            seq = read_fasta(a.fasta)
            oligos = pool(seq, a.file)
            if Path(a.out).exists():
                raise web.Refusal("exists", a.out)
            write_pool(a.out, oligos, f"pool1 file={a.file}")
            gcs = [gc(o) for o in oligos]
            print(f"oligos {len(oligos)} nt {OLIGO_NT * len(oligos)} gc {min(gcs):.4f}-{max(gcs):.4f} run {max(map(longest_run, oligos))}")
        elif a.cmd == "unpool":
            seq, used, dropped = unpool(read_pool(a.reads), a.file)
            if Path(a.out).exists():
                raise web.Refusal("exists", a.out)
            write_fasta(a.out, seq, f"dna1 nt={len(seq)} from pool1 file={a.file}")
            print(seq_root(seq), len(seq), f"used {used} dropped {dropped}")
        elif a.cmd == "glyph":
            text = glyph(read_fasta(a.fasta), a.k)
            if a.out:
                if Path(a.out).exists():
                    raise web.Refusal("exists", a.out)
                with open(a.out, "x", encoding="ascii") as f:
                    f.write(text)
            else:
                sys.stdout.write(text)
    except web.Refusal as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
