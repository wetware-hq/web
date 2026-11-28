"""Reference `loop1`: a pseudovisual chain of thought about one dna1 sequence (contract §12).

Every frame is a text picture an agent reads directly: a view of the data (`think`), a prediction
drawn in the same form (`hypothesis`), one replayable experiment (`act`) and what it gave (`observe`).
Verdicts are computed, never written. Frames are hash-chained, so the head names the whole loop.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dna  # noqa: E402
import web  # noqa: E402

TAG = b"loop1"
VIEW_RAMP = " .:-=+*#%@"  # web view: space = darkest, "@" = brightest
DIFF = {"same": "=", "brighter": "+", "darker": "-"}
OPS = {"lose": ("file", "oligos"), "damage": ("file", "seed", "drop", "sub", "flip"), "reads": ("file", "sha256", "path")}
VIEWS = {"web": ("size",)}
PER = 1000  # damage rates are per mille
READBACK_CODES = ("gap", "tie", "homopolymer", "bad_pool", "bad_length", "bad_word", "too_large") + (
    "bad_magic", "bad_header", "truncated", "trailing_bytes", "bad_weight")  # unpool, then decipher


def chain(blocks):
    """head(empty) = sha256("loop1"); head(w . b) = sha256(head(w) raw || b)."""
    h = hashlib.sha256(TAG).digest()
    for b in blocks:
        h = hashlib.sha256(h + b.encode("ascii")).digest()
    return h.hex()


def fields(line, tag):
    parts = line.split()
    if not parts or parts[0] != tag:
        raise web.Refusal("bad_loop", f"expected {tag}: {line[:40]!r}")
    try:
        return parts[1:], dict(p.split("=", 1) for p in parts[1:] if "=" in p)
    except ValueError:
        raise web.Refusal("bad_loop", line[:40]) from None


def num(f, key, lo, hi):
    v = f.get(key, "")
    if not v.isdigit() or (len(v) > 1 and v[0] == "0") or not lo <= int(v) <= hi:
        raise web.Refusal("bad_loop", f"{key}={v!r} is not {lo}-{hi}")
    return int(v)


# --- views -----------------------------------------------------------------------------------


def cells(blob, size):
    """The web's render as size x size block-mean luminance levels, 0-255."""
    img = web.decode(blob).astype(np.int64)
    rows, cols = img.shape[:2]
    if not 1 <= size <= min(rows, cols, 64):
        raise web.Refusal("bad_loop", f"size {size}")
    lum = img.sum(2) // img.shape[2]
    rb = [r * rows // size for r in range(size + 1)]
    cb = [c * cols // size for c in range(size + 1)]
    return [[int(lum[rb[r] : rb[r + 1], cb[c] : cb[c + 1]].sum() // ((rb[r + 1] - rb[r]) * (cb[c + 1] - cb[c]))) for c in range(size)] for r in range(size)]


def frame(rows):
    return ["+" + "-" * len(rows[0]) + "+"] + ["|" + r + "|" for r in rows] + ["+" + "-" * len(rows[0]) + "+"]


def view(blob, size):
    """view1: the web's render as a size x size text grid on a 10-step ramp, dark to bright."""
    levels = cells(blob, size)
    rows, cols = web.unpack(blob)["grid"][:2]
    lines = [f"%view1 size={size} grid={rows}x{cols} web={web.root(blob)}", f'%ramp "{VIEW_RAMP}"']
    lines += frame(["".join(VIEW_RAMP[v * len(VIEW_RAMP) // 256] for v in row) for row in levels])
    lines.append("%levels sha256:" + hashlib.sha256(" ".join(str(v) for row in levels for v in row).encode("ascii")).hexdigest())
    return lines


def diff(expected, observed, size):
    """Cell by cell, whether the read-back web's view is the same, brighter or darker than the data's."""
    a, b = cells(expected, size), cells(observed, size)
    rows = ["".join(DIFF["same"] if y == x else DIFF["brighter"] if y > x else DIFF["darker"] for x, y in zip(ra, rb)) for ra, rb in zip(a, b)]
    changed = sum(x != y for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    return [f"%diff size={size} changed={changed}/{size * size}", f'%key "{"".join(DIFF.values())}" same brighter darker'] + frame(rows)


# --- experiments -----------------------------------------------------------------------------


def rate_hit(seed, j, slot, rate):
    u = int.from_bytes(hashlib.sha256(TAG + b"/damage" + b"".join(v.to_bytes(8, "big") for v in (seed, j, slot))).digest()[:8], "big")
    return u * PER < rate << 64, u


def damage(oligos, seed, drop, sub, flip):
    """Seeded, replayable damage: drop whole oligos, substitute bases, read some from the other strand."""
    reads = []
    for j, o in enumerate(oligos):
        if rate_hit(seed, j, 0, drop)[0]:
            continue
        bases = list(o)
        for p in range(len(bases)):
            hit, u = rate_hit(seed, j, 2 + p, sub)
            if hit:
                bases[p] = dna.BASES[(dna.BASES.index(bases[p]) + 1 + (u >> 32) % 3) % 4]
        r = "".join(bases)
        reads.append(dna.revcomp(r) if rate_hit(seed, j, 1, flip)[0] else r)
    return reads


def run_act(seq, f, base):
    """One act -> ("web", seq, used, dropped, blob) or ("refused", code). Every op ends in unpool, then decipher."""
    op = f.get("op")
    file_id = num(f, "file", 0, 3**dna.FILE_TRITS - 1)
    if op == "lose":
        oligos = dna.pool(seq, file_id)
        try:
            lost = {int(i) for i in f["oligos"].split(",")}
        except (KeyError, ValueError):
            raise web.Refusal("bad_loop", "oligos") from None
        if f["oligos"] != ",".join(map(str, sorted(lost))) or min(lost) < 0 or max(lost) >= len(oligos):
            raise web.Refusal("bad_loop", "oligos must be ascending pool indices")
        reads = [o for i, o in enumerate(oligos) if i not in lost]
    elif op == "damage":
        rates = [num(f, key, 0, PER) for key in ("drop", "sub", "flip")]
        reads = damage(dna.pool(seq, file_id), num(f, "seed", 0, 2**63 - 1), *rates)
    elif op == "reads":
        if not re.fullmatch(r"[A-Za-z0-9._/-]+", f.get("path", "")):
            raise web.Refusal("bad_loop", f"reads path {f.get('path')!r} is not [A-Za-z0-9._/-]+")
        p = (base / f["path"]).resolve()
        if not p.is_relative_to(base.resolve()) or not p.is_file():
            raise web.Refusal("bad_loop", f"no reads file {f.get('path')}")
        if not re.fullmatch(r"[0-9a-f]{64}", f.get("sha256", "")):
            raise web.Refusal("bad_loop", "reads sha256 is not 64 lowercase hex")
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != f.get("sha256"):
            raise web.Refusal("hash_drift", f"{f.get('path')} does not hash to its pin")
        reads = dna.parse_pool(data)
    else:
        raise web.Refusal("bad_loop", f"op {op!r}")
    try:
        got, used, dropped = dna.unpool(reads, file_id)
        return ("web", got, used, dropped, dna.decipher(got))
    except web.Refusal as err:
        return ("refused", err.code)


# --- frames ----------------------------------------------------------------------------------


def think(blob, f):
    if f.get("view") == "web":
        return view(blob, num(f, "size", 1, 64))
    raise web.Refusal("bad_loop", f"view {f.get('view')!r}")


def expected(f):
    e = f.get("expect", "")
    if e == "intact" or e in {"refused:" + code for code in READBACK_CODES}:
        return e
    raise web.Refusal("bad_loop", f"expect {e!r}")


def hypothesis(blob, f):
    size = num(f, "size", 1, 64)
    return view(blob, size) if expected(f) == "intact" else []


def observe(seq, blob, hyp, outcome):
    """The observe frame for an outcome, against the hypothesis it tests."""
    size, expect = num(hyp, "size", 1, 64), expected(hyp)
    if outcome[0] == "refused":
        got, body = "refused:" + outcome[1], []
        head = f"outcome={got}"
    else:
        got = "intact" if outcome[1] == seq else "changed"
        head = f"outcome={got} sequence={dna.seq_root(outcome[1])} web={web.root(outcome[4])} used={outcome[2]} dropped={outcome[3]}"
        body = view(outcome[4], size) + diff(blob, outcome[4], size)
    return f"{head} verdict={'supports' if got == expect else 'refutes'}", body


def block(i, kind, params, body):
    return "\n".join([f"%frame {i} {kind} {params}".rstrip()] + body) + "\n"


def run(seq, plan, base=Path(".")):
    """A plan (list of steps) -> loop1 text. A step is {"think": view params} or
    {"hypothesis": params, "act": params}; observe is always computed."""
    blob = dna.decipher(seq)
    blocks = [f"%loop1 nt={len(seq)} sequence={dna.seq_root(seq)} web={web.root(blob)}\n"]

    def params(d):
        return " ".join(f"{k}={v}" for k, v in d.items())

    if not isinstance(plan, list):
        raise web.Refusal("bad_loop", "a plan is a list of steps")
    for step in plan:
        i = len(blocks) - 1
        if not isinstance(step, dict) or set(step) not in ({"think"}, {"hypothesis", "act"}) or not all(isinstance(v, dict) for v in step.values()):
            raise web.Refusal("bad_loop", "a step is {think} or {hypothesis, act}")
        if "think" in step:
            t = {k: str(v) for k, v in step["think"].items()}
            p = {"view": t.get("view", ""), **{key: t[key] for key in VIEWS.get(t.get("view"), ()) if key in t}}
            blocks.append(block(i, "think", params(p), think(blob, p)))
            continue
        h = {k: str(v) for k, v in step["hypothesis"].items()}
        h = {key: h[key] for key in ("expect", "size") if key in h}
        a = {k: str(v) for k, v in step["act"].items()}
        op = a.get("op", "")
        a = {"op": op, **{key: a[key] for key in OPS.get(op, ()) if key in a}}
        blocks.append(block(i, "hypothesis", params(h), hypothesis(blob, h)))
        blocks.append(block(i + 1, "act", params(a), []))
        line, body = observe(seq, blob, h, run_act(seq, a, base))
        blocks.append(block(i + 2, "observe", line, body))
    return "".join(blocks) + f"%head {chain(blocks)}\n"


def split(text):
    """loop1 text -> (blocks, head); refuses anything that is not header, frames, head."""
    if not text.endswith("\n") or "\r" in text:
        raise web.Refusal("bad_loop", "lines must end with LF")
    lines = text[:-1].split("\n")
    if len(lines) < 2 or not lines[0].startswith("%loop1 ") or not lines[-1].startswith("%head "):
        raise web.Refusal("bad_loop", "not a loop1 file")
    blocks, cur = [lines[0] + "\n"], None
    for ln in lines[1:-1]:
        if ln.startswith("%frame "):
            cur = [ln]
            blocks.append(cur)
        elif cur is None:
            raise web.Refusal("bad_loop", "text before the first frame")
        else:
            cur.append(ln)
    blocks = blocks[:1] + ["\n".join(b) + "\n" for b in blocks[1:]]
    return blocks, lines[-1][6:]


def head(text):
    """The loop's head, after checking it is the hash chain of its blocks (no replay)."""
    blocks, h = split(text)
    if h != chain(blocks):
        raise web.Refusal("bad_head", f"head is {chain(blocks)}")
    return h


def verify(text, seq, base=Path(".")):
    """Regenerate every frame from its header and the data, replay every act, check the head.

    Returns [(frame index, verdict)] for each observe frame.
    """
    head(text)
    blocks = split(text)[0]
    plan, verdicts, frames = [], [], [fields(b.split("\n", 1)[0], "%frame")[0] for b in blocks[1:]]
    i = 0
    while i < len(frames):
        parts = frames[i]
        if len(parts) < 2 or parts[0] != str(i):
            raise web.Refusal("bad_loop", f"frame {i} is misnumbered")
        kind, f = parts[1], dict(p.split("=", 1) for p in parts[2:] if "=" in p)
        if kind == "think":
            plan.append({"think": f})
            i += 1
        elif kind == "hypothesis" and i + 2 < len(frames) and frames[i + 1][:2] == [str(i + 1), "act"] and frames[i + 2][:2] == [str(i + 2), "observe"]:
            a = dict(p.split("=", 1) for p in frames[i + 1][2:] if "=" in p)
            plan.append({"hypothesis": f, "act": a})
            verdicts.append((i + 2, dict(p.split("=", 1) for p in frames[i + 2][2:] if "=" in p).get("verdict")))
            i += 3
        else:
            raise web.Refusal("bad_loop", f"frame {i}: {kind} out of order (think, or hypothesis act observe)")
    replay = run(seq, plan, base)
    if replay != text:
        redo = split(replay)[0]
        bad = next((j for j, (x, y) in enumerate(zip(blocks, redo)) if x != y), len(redo))
        raise web.Refusal("mismatch", "header" if bad == 0 else f"frame {bad - 1} does not replay")
    return verdicts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="loop", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="think, hypothesize, act and observe over a dna1 sequence")
    r.add_argument("fasta")
    r.add_argument("plan", help="JSON list of steps")
    r.add_argument("out")
    v = sub.add_parser("verify", help="replay a loop1 file against its sequence; print verdicts and head")
    v.add_argument("fasta")
    v.add_argument("loop")
    h = sub.add_parser("head", help="check the hash chain only, without the sequence; print the head")
    h.add_argument("loop")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "run":
            plan = json.loads(Path(a.plan).read_text())
            text = run(dna.read_fasta(a.fasta), plan, Path(a.out).resolve().parent)
            if Path(a.out).exists():
                raise web.Refusal("exists", a.out)
            with open(a.out, "x", encoding="ascii", newline="\n") as f:
                f.write(text)
            print(split(text)[1])
        elif a.cmd == "verify":
            text = Path(a.loop).read_bytes().decode("ascii")
            for i, verdict in verify(text, dna.read_fasta(a.fasta), Path(a.loop).resolve().parent):
                print(f"frame {i} {verdict}")
            print(split(text)[1])
        else:
            print(head(Path(a.loop).read_bytes().decode("ascii")))
    except (web.Refusal, UnicodeDecodeError, json.JSONDecodeError) as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
