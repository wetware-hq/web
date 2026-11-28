"""Reference `deep1`: a deep net as a composition of shallow webs, each interface bound in a hash chain (contract §16).

Needs only Python 3.10+, numpy and `ref/web.py`. A one-stage deep net is a web. A deep net of n + 1
stages feeds the levels of the first n to one more web, read on that web's own grid of 256 levels per
axis, so composing uses nothing but the §3 decode. Each stage's web root and output levels are hashed
into a chain whose 32-byte head names the composition.
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import web  # noqa: E402

TAG = b"deep1"
LEVELS = 256  # an interface is uint8 levels, so a later stage's grid is 256 per axis
AXIS = np.linspace(-1.0, 1.0, LEVELS)  # the §3 axis of size 256: level l sits at AXIS[l]
MAX_STAGES = 64
LUMA = (0.299, 0.587, 0.114)  # ITU-R BT.601, the sample's second stage
LUMA_EPS = 1 / 64  # sin(sin(eps z)) / eps is z to within z^3 eps^2 / 3


def stage_shapes(blobs):
    """The parsed webs, refused unless each stage's grid is 256 levels per channel of the stage before."""
    if not 1 <= len(blobs) <= MAX_STAGES:
        raise web.Refusal("bad_stage", f"{len(blobs)} stages, 1-{MAX_STAGES}")
    ws = [web.unpack(b) for b in blobs]
    for i in range(1, len(ws)):
        want = (LEVELS,) * ws[i - 1]["d_out"]
        if ws[i]["grid"] != want:
            raise web.Refusal("bad_stage", f"stage {i} grid {ws[i]['grid']}, the interface needs {want}")
    return ws


def compose(blobs):
    """[levels of stage 0, ..., levels of stage n-1], each uint8 of shape grid of stage 0 + (d,)."""
    ws = stage_shapes(blobs)
    out = [web.decode(blobs[0])]
    for w in ws[1:]:
        prev = out[-1]
        x = AXIS[prev.reshape(-1, prev.shape[-1])]
        y = np.empty((len(x), w["d_out"]), np.uint8)
        for i in range(0, len(x), web.DECODE_ROWS):
            y[i : i + web.DECODE_ROWS] = web.to_levels(web.forward(x[i : i + web.DECODE_ROWS], w["params"], w["omega0"]))
        out.append(y.reshape(*prev.shape[:-1], w["d_out"]))
    return out


def out_root(levels):
    """sha256 of an interface's levels, uint8 row-major over grid + (d,)."""
    return hashlib.sha256(np.ascontiguousarray(levels, np.uint8).tobytes()).hexdigest()


def links(blobs):
    """[(web root, d, out root)] per stage."""
    return [(web.root(b), int(levels.shape[-1]), out_root(levels)) for b, levels in zip(blobs, compose(blobs))]


def chain(pairs):
    """h_0 = sha256("deep1"); h_(i+1) = sha256(h_i || web root_i || out root_i), all raw 32 bytes."""
    hs = [hashlib.sha256(TAG).digest()]
    for w, o in pairs:
        hs.append(hashlib.sha256(hs[-1] + bytes.fromhex(w) + bytes.fromhex(o)).digest())
    return [h.hex() for h in hs]


def record(blobs):
    """The deep1 text of a composition: header, one line per stage, head."""
    ls, grid = links(blobs), web.unpack(blobs[0])["grid"]
    lines = [f"%deep1 stages={len(ls)} grid={'x'.join(map(str, grid))}"]
    lines += [f"%stage {i} web={w} d={d} out={o}" for i, (w, d, o) in enumerate(ls)]
    lines.append("%head " + chain([(w, o) for w, _, o in ls])[-1])
    return "\n".join(lines) + "\n"


def parse(text):
    """deep1 text -> (header, [(web root, d, out root)], head), refusing anything off the grammar."""
    if not text.endswith("\n") or "\r" in text:
        raise web.Refusal("bad_deep", "lines must end with LF")
    lines = text[:-1].split("\n")
    try:
        head, n = lines[0], int(lines[0].split()[1].removeprefix("stages="))
    except (IndexError, ValueError):
        raise web.Refusal("bad_deep", "not a deep1 file") from None
    if not head.startswith("%deep1 stages=") or not 1 <= n <= MAX_STAGES or len(lines) != n + 2 or not lines[-1].startswith("%head "):
        raise web.Refusal("bad_deep", "header, stage count or head line")
    ls = []
    for i, ln in enumerate(lines[1:-1]):
        parts = ln.split()
        f = dict(p.split("=", 1) for p in parts[2:] if "=" in p)
        if parts[:2] != ["%stage", str(i)] or list(f) != ["web", "d", "out"] or len(parts) != 5:
            raise web.Refusal("bad_deep", f"stage line {i}")
        ls.append((f["web"], f["d"], f["out"]))
    return head, ls, lines[-1][6:]


def head(text):
    """The head, after checking it is the chain of the stage lines (no webs needed)."""
    _, ls, h = parse(text)
    hexes = all(re.fullmatch(r"[0-9a-f]{64}", v) for w, _, o in ls for v in (w, o))
    want = chain([(w, o) for w, _, o in ls])[-1] if hexes else None
    if want is None or h != want:
        raise web.Refusal("bad_head", f"head is {want}")
    return h


def verify(text, blobs):
    """Check the head, that each web is the stage it names, and that every interface replays."""
    h = head(text)
    _, ls, _ = parse(text)
    if len(ls) != len(blobs):
        raise web.Refusal("bad_stage", f"{len(ls)} stages named, {len(blobs)} webs given")
    for i, ((w, _, _), b) in enumerate(zip(ls, blobs)):
        if web.root(b) != w:
            raise web.Refusal("root_mismatch", f"stage {i} is {web.root(b)}")
    replay = record(blobs)
    if replay != text:
        bad = next(i for i, (x, y) in enumerate(zip(text.split("\n"), replay.split("\n"))) if x != y)
        raise web.Refusal("mismatch", "header" if bad == 0 else f"stage {bad - 1} does not replay")
    return h


def luma(eps=LUMA_EPS):
    """A hand-set stage, not a fit: RGB levels -> BT.601 luma, sin(sin(eps c.x)) / eps with H = 1, omega0 = 1."""
    c = np.array(LUMA).reshape(3, 1)
    return web.pack((LEVELS,) * 3, 1, 1, 1, [eps * c, np.zeros(1), np.ones((1, 1)), np.zeros(1), np.full((1, 1), 1 / eps), np.zeros(1)])


def write(path, data):
    if Path(path).exists():
        raise web.Refusal("exists", path)
    with open(path, "xb") as f:
        f.write(data)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="deep", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="compose webs, first stage first, and write the deep1 record")
    r.add_argument("out")
    r.add_argument("webs", nargs="+")
    v = sub.add_parser("verify", help="replay a deep1 record against its webs; print the head")
    v.add_argument("deep")
    v.add_argument("webs", nargs="+")
    d = sub.add_parser("decode", help="write the last stage's levels as a PNG")
    d.add_argument("out")
    d.add_argument("webs", nargs="+")
    sub.add_parser("head", help="check the chain only, without the webs; print the head").add_argument("deep")
    sub.add_parser("luma", help="write the hand-set RGB -> luma stage").add_argument("out")
    a = ap.parse_args(argv)
    try:
        blobs = [Path(p).read_bytes() for p in getattr(a, "webs", [])]
        if a.cmd == "run":
            text = record(blobs)
            write(a.out, text.encode("ascii"))
            print(text.rsplit(" ", 1)[1], end="")
        elif a.cmd == "verify":
            print(verify(Path(a.deep).read_bytes().decode("ascii"), blobs))
        elif a.cmd == "decode":
            levels = compose(blobs)[-1]
            if levels.ndim != 3:
                raise web.Refusal("shape_mismatch", f"a PNG is rank 2, this composition is rank {levels.ndim - 1}")
            if Path(a.out).exists():
                raise web.Refusal("exists", a.out)
            web.write_png(a.out, levels)
        elif a.cmd == "head":
            print(head(Path(a.deep).read_bytes().decode("ascii")))
        else:
            blob = luma()
            write(a.out, blob)
            print(web.root(blob), len(blob))
    except (web.Refusal, UnicodeDecodeError) as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
