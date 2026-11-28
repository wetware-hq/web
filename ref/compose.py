"""Reference `kline1`: composition functions over shallow webs, recalled as K-lines and grown by induction (contract §17).

Needs only Python 3.10+, numpy, `ref/web.py` and `ref/deep.py`. A K-line is an ordered list of stage webs: activated
on any deep net whose last interface it fits, it composes those stages after it (§16). `grow` builds a deep net one
stage at a time, from recalled K-lines, a library of stages, or a stage fitted to the current interface, and keeps a
step only when the deep net's measured utility against a target rises. A K-line's head names its stages; its uses
record the deep nets it was activated on, and only a K-line whose every stage raised utility on two different
targets is recalled.
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deep  # noqa: E402
import web  # noqa: E402

TAG = b"kline1"
MAX_STEPS = 8
RECALL = 2  # a K-line is recalled once it raised utility on this many different targets
FIT = {"hidden": 16, "omega0": 30, "steps": 3000, "seed": 0}  # a fitted curve stage, `fit_curve`


def utility(levels, target):
    """PSNR (§6) of a deep net's last interface against a target of the same shape; None if not comparable."""
    levels, target = np.asarray(levels), np.asarray(target)
    if levels.shape != target.shape:
        return None
    return round(web.psnr(levels, target), 2)


def rises(after, before):
    """A step is kept only when utility rises; no utility is below every number."""
    return after is not None and (before is None or after > before)


def luma_target(media):
    """The BT.601 luma of a uint8 RGB array, (299 R + 587 G + 114 B) / 1000 rounded half to even in integers,
    so every machine gets the same target: the sample task's target, shape grid + (1,)."""
    media = np.asarray(media, np.uint8)
    if media.ndim != 3 or media.shape[-1] != 3:
        raise web.Refusal("shape_mismatch", f"luma needs an RGB image, not {media.shape}")
    n = media.astype(np.int64) @ np.array([299, 587, 114], np.int64)
    q, r = np.divmod(n, 1000)
    q += (r > 500) | ((r == 500) & (q % 2 == 1))
    return q.astype(np.uint8)[..., None]


def curve_table(levels, target):
    """The least-squares map from a 1-channel interface to a 1-channel target: the mean target at each level.

    Levels the interface never takes are filled by linear interpolation between the nearest levels it does take.
    """
    levels, target = np.asarray(levels), np.asarray(target)
    if levels.shape != target.shape or levels.shape[-1] != 1:
        raise web.Refusal("shape_mismatch", "a curve maps one channel onto one channel of the same grid")
    lv, tg = levels.ravel().astype(np.int64), target.ravel().astype(np.float64)
    count = np.bincount(lv, minlength=deep.LEVELS)
    total = np.bincount(lv, weights=tg, minlength=deep.LEVELS)
    seen = np.flatnonzero(count)
    mean = total[seen] / count[seen]
    return np.clip(np.rint(np.interp(np.arange(deep.LEVELS), seen, mean)), 0, 255).astype(np.uint8)


def fit_curve(levels, target, hidden=FIT["hidden"], omega0=FIT["omega0"], steps=FIT["steps"], seed=FIT["seed"]):
    """A stage fitted to the interface: a rank-1 web on the 256-level grid that approximates `curve_table`."""
    table = curve_table(levels, target)
    return web.encode(table.reshape(deep.LEVELS, 1), hidden=hidden, omega0=omega0, steps=steps, seed=seed)


def fits(d, blob):
    """Whether a stage reads an interface of d channels (§16: its grid is 256 per channel)."""
    return web.unpack(blob)["grid"] == (deep.LEVELS,) * d


def last(blobs):
    return deep.compose(blobs)[-1]


def grow(blobs, target, klines=(), library=(), fit=True, max_steps=MAX_STEPS):
    """Induction: extend the deep net `blobs` one step at a time while its utility against `target` rises.

    Each step tries, in order, every recalled K-line (a list of stage blobs) whose first stage fits the interface,
    every library stage that fits it, and, when the interface and the target have one channel, a fitted curve.
    The candidate of highest utility is kept if it raises utility (ties go to the earlier candidate); else growth
    stops. Returns (the stages added, [(source, utility)] from the start, one entry per kept step).
    """
    target = np.asarray(target, np.uint8)
    blobs, added = list(blobs), []
    trace = [("start", utility(last(blobs), target))]
    for _ in range(max_steps):
        d = deep.compose(blobs)[-1].shape[-1]
        if len(blobs) >= deep.MAX_STAGES:
            break
        cands = [(f"kline:{head(k)}", list(k)) for k in klines if k and fits(d, k[0])]
        cands += [(f"stage:{web.root(s)}", [s]) for s in library if fits(d, s)]
        if fit and d == 1 and target.shape[-1] == 1:
            cands.append(("fit", None))
        best = None
        for source, stages in cands:
            if stages is None:
                stages = [fit_curve(last(blobs), target)]
            if len(blobs) + len(stages) > deep.MAX_STAGES:
                continue
            u = utility(last(blobs + stages), target)
            if u is not None and (best is None or u > best[2]):
                best = (source, stages, u)
        if best is None or not rises(best[2], trace[-1][1]):
            break
        blobs += best[1]
        added += best[1]
        trace.append((best[0], best[2]))
    return added, trace


# --- the kline1 record ---------------------------------------------------------------------


def chain(roots):
    """k_0 = sha256("kline1"); k_(i+1) = sha256(k_i || web root_i), raw 32 bytes; the head names the stages."""
    hs = [hashlib.sha256(TAG).digest()]
    for r in roots:
        hs.append(hashlib.sha256(hs[-1] + bytes.fromhex(r)).digest())
    return [h.hex() for h in hs]


def head(stages):
    """The K-line head of a list of stage blobs."""
    return chain([web.root(s) for s in stages])[-1]


def fmt(u):
    return "none" if u is None else f"{u:.2f}"


def use(inputs, stages, target):
    """One use of a K-line on a deep net: (deep head of inputs + stages, sha256 of the target, the utilities).

    The utilities are those of the inputs, then of each prefix of the stages, so a use shows every step it took.
    """
    full = list(inputs) + list(stages)
    dh = deep.record(full).rsplit(" ", 1)[1].strip()
    t = np.ascontiguousarray(target, np.uint8)
    outs = deep.compose(full)[len(inputs) - 1 :]
    return dh, hashlib.sha256(t.tobytes()).hexdigest(), [utility(o, t) for o in outs]


def supports(us):
    """A use supports its K-line only when every stage it added raised utility."""
    return all(rises(a, b) for b, a in zip(us, us[1:]))


def record(stages, uses):
    """The kline1 text: header, one line per stage, the head, then one line per use."""
    ws = [web.unpack(s) for s in stages]
    if not 1 <= len(ws) <= deep.MAX_STAGES:
        raise web.Refusal("bad_stage", f"{len(ws)} stages, 1-{deep.MAX_STAGES}")
    d_in = len(ws[0]["grid"])
    if ws[0]["grid"] != (deep.LEVELS,) * d_in or d_in not in (1, 3):
        raise web.Refusal("bad_stage", "a K-line's first stage reads an interface: its grid is 256 per channel")
    for i in range(1, len(ws)):
        if ws[i]["grid"] != (deep.LEVELS,) * ws[i - 1]["d_out"]:
            raise web.Refusal("bad_stage", f"stage {i} does not read the interface of stage {i - 1}")
    lines = [f"%kline1 stages={len(ws)} in={d_in} out={ws[-1]['d_out']} uses={len(uses)}"]
    lines += [f"%stage {i} web={web.root(s)} d={w['d_out']}" for i, (s, w) in enumerate(zip(stages, ws))]
    lines.append("%head " + head(stages))
    for j, (dh, t, us) in enumerate(uses):
        if len(us) != len(ws) + 1:
            raise web.Refusal("bad_kline", f"use {j} has {len(us)} utilities, needs {len(ws) + 1}")
        lines.append(f"%use {j} deep={dh} target={t} utility={','.join(map(fmt, us))}")
    return "\n".join(lines) + "\n"


def parse(text):
    """kline1 text -> (stages [(web root, d)], head, uses [(deep head, target, [utility])]), or bad_kline."""
    if not text.endswith("\n") or "\r" in text:
        raise web.Refusal("bad_kline", "lines must end with LF")
    lines = text[:-1].split("\n")
    m = re.fullmatch(r"%kline1 stages=(\d+) in=([13]) out=([13]) uses=(\d+)", lines[0])
    if not m:
        raise web.Refusal("bad_kline", "header")
    n, u = int(m[1]), int(m[4])
    if not 1 <= n <= deep.MAX_STAGES or len(lines) != 1 + n + 1 + u:
        raise web.Refusal("bad_kline", "stage or use count")
    stages = []
    for i, ln in enumerate(lines[1 : 1 + n]):
        s = re.fullmatch(rf"%stage {i} web=([0-9a-f]{{64}}) d=([13])", ln)
        if not s:
            raise web.Refusal("bad_kline", f"stage line {i}")
        stages.append((s[1], int(s[2])))
    h = re.fullmatch(r"%head ([0-9a-f]{64})", lines[1 + n])
    if not h:
        raise web.Refusal("bad_kline", "head line")
    uses = []
    for j, ln in enumerate(lines[2 + n :]):
        s = re.fullmatch(rf"%use {j} deep=([0-9a-f]{{64}}) target=([0-9a-f]{{64}}) utility=((?:none|\d+\.\d\d)(?:,(?:none|\d+\.\d\d)){{{n}}})", ln)
        if not s:
            raise web.Refusal("bad_kline", f"use line {j}")
        uses.append((s[1], s[2], [None if v == "none" else float(v) for v in s[3].split(",")]))
    if int(m[3]) != stages[-1][1]:
        raise web.Refusal("bad_kline", "out is not the last stage's d")
    return stages, h[1], uses


def check_head(text):
    """The head, after checking it is the chain over the stage lines (no webs needed)."""
    stages, h, _ = parse(text)
    want = chain([r for r, _ in stages])[-1]
    if h != want:
        raise web.Refusal("bad_head", f"head is {want}")
    return h


def recalled(text):
    """Whether a K-line may be activated by `grow`: its every stage raised utility on RECALL different targets."""
    check_head(text)
    _, _, uses = parse(text)
    return len({t for _, t, us in uses if supports(us)}) >= RECALL


def verify(text, stages, uses):
    """Check the head, that each blob is the stage it names, and that every use replays: `uses` is [(inputs, target)]."""
    h = check_head(text)
    named, _, _ = parse(text)
    if len(named) != len(stages):
        raise web.Refusal("bad_stage", f"{len(named)} stages named, {len(stages)} webs given")
    for i, ((r, _), s) in enumerate(zip(named, stages)):
        if web.root(s) != r:
            raise web.Refusal("root_mismatch", f"stage {i} is {web.root(s)}")
    replay = record(stages, [use(i, stages, t) for i, t in uses])
    if replay != text:
        bad = next(i for i, (x, y) in enumerate(zip(text.split("\n"), replay.split("\n"))) if x != y)
        raise web.Refusal("mismatch", f"line {bad} does not replay")
    return h


def write(path, data):
    if Path(path).exists():
        raise web.Refusal("exists", path)
    with open(path, "xb") as f:
        f.write(data)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="compose", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grow", help="grow a deep net toward the BT.601 luma of an RGB PNG; print each kept step")
    g.add_argument("png", help="the source whose luma is the target")
    g.add_argument("webs", nargs="+", help="the deep net to start from, first stage first")
    g.add_argument("--library", nargs="+", default=[], help="stage webs to try at each step")
    g.add_argument("--recall", nargs="+", action="append", default=[], metavar="KLINE STAGE", help="a kline1 file, then its stage webs; used only if recalled")
    g.add_argument("--no-fit", action="store_true", help="never fit a curve stage")
    g.add_argument("--out", help="write each fitted stage as <out>.<i>.web")
    v = sub.add_parser("verify", help="replay a kline1 record against its stages and uses; print the head")
    v.add_argument("kline")
    v.add_argument("--stages", nargs="+", required=True)
    v.add_argument("--use", nargs="+", action="append", default=[], metavar="PNG WEB", help="a use, in order: the PNG whose luma is the target, then the input webs")
    sub.add_parser("head", help="check the chain only, without the webs; print the head").add_argument("kline")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "grow":
            target = luma_target(web.read_png(a.png))
            lib = [Path(p).read_bytes() for p in a.library]
            klines = []
            for k, *ss in a.recall:
                text, stages = Path(k).read_bytes().decode("ascii"), [Path(p).read_bytes() for p in ss]
                if check_head(text) != head(stages):
                    raise web.Refusal("root_mismatch", f"{k} does not name these stages")
                if recalled(text):
                    klines.append(stages)
            added, trace = grow([Path(p).read_bytes() for p in a.webs], target, klines, lib, fit=not a.no_fit)
            known = {web.root(s) for s in lib} | {web.root(s) for k in klines for s in k}
            for i, s in enumerate(added):
                if a.out and web.root(s) not in known:
                    write(f"{a.out}.{i}.web", s)
            for source, u in trace:
                print(source, fmt(u))
            if added:
                print("kline", head(added))
        elif a.cmd == "verify":
            stages = [Path(p).read_bytes() for p in a.stages]
            uses = [([Path(p).read_bytes() for p in u[1:]], luma_target(web.read_png(u[0]))) for u in a.use]
            print(verify(Path(a.kline).read_bytes().decode("ascii"), stages, uses))
        else:
            print(check_head(Path(a.kline).read_bytes().decode("ascii")))
    except (web.Refusal, UnicodeDecodeError, ValueError) as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
