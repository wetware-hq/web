"""Reference `tmap1` tensor maps: every object in the stack as one set of named tensors (contract §14).

Needs only Python 3.10+, numpy and the other `ref/` modules. A tensor record is a TensorFlow
`TensorProto` written flat: its `DataType` number, its shape, and its `tensor_content` bytes
(row-major, little-endian). `to_tf` and `from_tf` cross into TensorFlow when it is installed.
"""

import argparse
import hashlib
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose  # noqa: E402
import deep  # noqa: E402
import dna  # noqa: E402
import loop  # noqa: E402
import web  # noqa: E402

MAGIC = b"tmap1"
MAX_BYTES = 1 << 30
KIND = re.compile(r"[a-z][a-z0-9-]{0,31}")
NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")

# TensorFlow DataType numbers (tensorflow/core/framework/types.proto) and their numpy forms.
DTYPES = {1: "<f4", 2: "<f8", 3: "<i4", 4: "|u1", 5: "<i2", 6: "|i1", 9: "<i8", 10: "|b1", 17: "<u2", 19: "<f2", 22: "<u4", 23: "<u8"}
CODES = {np.dtype(v): k for k, v in DTYPES.items()}

EXACT = ("web", "sequence", "pool")
VIEWS = ("loop", "deep", "kline")
WEIGHTS = ("w1", "b1", "w2", "b2", "w3", "b3")
LOOP_KINDS = ("think", "hypothesis", "act", "observe")
VERDICTS = {"supports": 1, "refutes": 2}
ACGT = np.frombuffer(dna.BASES.encode("ascii"), np.uint8)

Refusal = web.Refusal


def bad(why):
    return Refusal("bad_map", why)


# --- bytes ----------------------------------------------------------------------------------


def pack(kind, tensors):
    """Serialize {name: array} under `kind` to canonical tmap1 bytes, names in ascending order."""
    if not KIND.fullmatch(kind):
        raise bad(f"kind {kind!r}")
    if not 1 <= len(tensors) <= 1024:
        raise bad(f"{len(tensors)} tensors")
    out = [MAGIC, bytes([len(kind)]), kind.encode(), struct.pack("<H", len(tensors))]
    for name in sorted(tensors):
        a = np.asarray(tensors[name])
        a = a.astype(a.dtype.newbyteorder("<"), order="C")
        code = CODES.get(a.dtype)
        if not NAME.fullmatch(name) or code is None or a.ndim > 8 or max(a.shape, default=0) >> 32:
            raise bad(f"{name}: {a.dtype} {list(a.shape)}")
        out += [bytes([len(name)]), name.encode(), bytes([code, a.ndim]), struct.pack(f"<{a.ndim}I", *a.shape), a.tobytes()]
    blob = b"".join(out)
    unpack(blob)
    return blob


def unpack(blob, kind=None):
    """Parse tmap1 bytes into (kind, {name: array}); refuse anything that would not re-serialize to `blob`."""
    if len(blob) > MAX_BYTES:
        raise Refusal("too_large", f"{len(blob)} bytes")
    if blob[:5] != MAGIC:
        raise Refusal("bad_magic")
    pos = 5

    def take(n):
        nonlocal pos
        if pos + n > len(blob):
            raise Refusal("truncated")
        pos += n
        return blob[pos - n : pos]

    k = take(take(1)[0]).decode("latin-1")
    (count,) = struct.unpack("<H", take(2))
    if not KIND.fullmatch(k) or not 1 <= count <= 1024:
        raise bad(f"kind {k!r}, {count} tensors")
    tensors = {}
    for _ in range(count):
        n = take(take(1)[0]).decode("latin-1")
        code, rank = take(2)
        if not NAME.fullmatch(n) or n <= max(tensors, default="") or code not in DTYPES or rank > 8:
            raise bad(f"record {n!r}: DataType {code}, rank {rank}")
        shape = struct.unpack(f"<{rank}I", take(4 * rank))
        dt = np.dtype(DTYPES[code])
        raw = take(int(np.prod(shape, dtype=np.int64)) * dt.itemsize)
        if code == 10 and raw.strip(b"\0\1"):
            raise bad(f"{n}: bool not 0 or 1")
        a = np.frombuffer(raw, dt).reshape(shape)
        if dt.kind == "f" and not (np.all(np.isfinite(a)) and not np.any(np.signbit(a) & (a == 0))):
            raise bad(f"{n}: NaN, infinity or negative zero")
        tensors[n] = a
    if pos != len(blob):
        raise Refusal("trailing_bytes")
    if kind is not None and k != kind:
        raise Refusal("kind_mismatch", f"{k}, wanted {kind}")
    return k, tensors


def root(blob):
    return hashlib.sha256(blob).hexdigest()


def need(blob, kind, shapes):
    """The tensors of a `kind` map, refused unless their names and ranks are the ones the kind defines."""
    t = unpack(blob, kind)[1]
    if set(t) != set(shapes) or any(t[n].ndim != r for n, r in shapes.items()):
        raise bad(f"{kind} tensors")
    return t


# --- maps: the objects of this repository ----------------------------------------------------


def of_web(blob):
    """A web as its header fields and its six weight tensors; the weight bytes are the web's own."""
    u = web.unpack(blob)
    t = {n: p.astype("<f2") for n, p in zip(WEIGHTS, u["params"])}
    t |= {"activation": np.uint8(web.ACT_SINE), "omega0": np.uint16(u["omega0"]), "grid": np.array(u["grid"], np.uint16)}
    return pack("web", t)


def to_web(blob):
    t = need(blob, "web", {"activation": 0, "omega0": 0, "grid": 1, "w1": 2, "b1": 1, "w2": 2, "b2": 1, "w3": 2, "b3": 1})
    try:
        out = web.pack(tuple(int(g) for g in t["grid"]), t["w3"].shape[1], t["w1"].shape[1], int(t["omega0"]), [t[n] for n in WEIGHTS])
    except (Refusal, ValueError, TypeError) as err:
        raise bad(f"web: {err}") from None
    if of_web(out) != blob:
        raise bad("not the canonical web tensors")
    return out


def of_sequence(seq):
    """A sequence as one uint8 tensor of base indices into ACGT; one-hot is a view of it."""
    if not seq or seq.strip(dna.BASES):
        raise Refusal("bad_base", "a sequence is non-empty ACGT")
    return pack("sequence", {"bases": np.searchsorted(ACGT, np.frombuffer(seq.encode(), np.uint8)).astype(np.uint8)})


def letters(a):
    if np.any(a > 3):
        raise Refusal("bad_base", "base index outside 0-3")
    return ACGT[a].tobytes().decode("ascii")


def to_sequence(blob):
    t = need(blob, "sequence", {"bases": 1})
    if not t["bases"].size or t["bases"].dtype != np.uint8:
        raise bad("sequence tensors")
    return letters(t["bases"])


def file_of(oligos):
    """The one file id the oligos themselves carry (§10), so a map never relabels a pool."""
    reads = [dna.read_oligo(o) for o in oligos]
    if not oligos or None in reads:
        raise Refusal("bad_pool", "a pool is valid 117-nt pool1 oligos")
    ids = {r[0] for r in reads}
    if len(ids) != 1:
        raise Refusal("bad_file", f"the pool carries file ids {sorted(ids)}")
    return ids.pop()


def of_pool(oligos):
    """A pool1 pool as [oligos, 117] base indices in pool order, and the file id its oligos carry."""
    if any(len(o) != dna.OLIGO_NT or o.strip(dna.BASES) for o in oligos):
        raise Refusal("bad_pool", "a pool is 117-nt ACGT oligos")
    file_id = file_of(oligos)
    bases = unpack(of_sequence("".join(oligos)))[1]["bases"]
    return pack("pool", {"file": np.uint8(file_id), "oligos": bases.reshape(len(oligos), dna.OLIGO_NT)})


def to_pool(blob):
    """The pool file `dna.py pool` writes, so the pool map is exact against its pinned sha256."""
    t = need(blob, "pool", {"file": 0, "oligos": 2})
    if t["file"].dtype != np.uint8 or t["oligos"].dtype != np.uint8 or t["oligos"].shape[1] != dna.OLIGO_NT:
        raise bad("pool tensors")
    oligos = [letters(row) for row in t["oligos"]]
    if file_of(oligos) != t["file"]:
        raise Refusal("bad_file", f"the oligos carry file {file_of(oligos)}, not {t['file']}")
    return "".join(f">pool1 file={t['file']} i={i}\n{o}\n" for i, o in enumerate(oligos)).encode("ascii")


def of_loop(text):
    """A loop1 file as its hash chain h_0 .. head, its frame kinds and its written verdicts (a view, §12)."""
    blocks, h = loop.split(text)
    if h != loop.chain(blocks):
        raise Refusal("bad_head", f"head is {loop.chain(blocks)}")
    hs = [hashlib.sha256(loop.TAG).digest()]
    for b in blocks:
        hs.append(hashlib.sha256(hs[-1] + b.encode("ascii")).digest())
    frames = [b.split("\n", 1)[0].split() for b in blocks[1:]]
    verdicts = [next((VERDICTS[p[8:]] for p in f[3:] if p.startswith("verdict=")), 0) for f in frames]
    return pack("loop", {
        "chain": np.frombuffer(b"".join(hs), np.uint8).reshape(len(hs), 32),
        "kind": np.array([LOOP_KINDS.index(f[2]) for f in frames], np.uint8),
        "verdict": np.array(verdicts, np.uint8),
    })


def of_deep(text):
    """A deep1 record as its stage web roots, interface roots and hash chain h_0 .. head (a view, §16)."""
    deep.head(text)
    ls = deep.parse(text)[1]

    def roots(hexes):
        return np.frombuffer(bytes.fromhex("".join(hexes)), np.uint8).reshape(len(hexes), 32)

    return pack("deep", {"chain": roots(deep.chain([(w, o) for w, _, o in ls])), "outs": roots([o for _, _, o in ls]), "webs": roots([w for w, _, _ in ls])})


def of_kline(text):
    """A kline1 record as its stage web roots, its hash chain k_0 .. head, and per use its deep head, its target
    root and whether every stage raised utility (a view, §17)."""
    compose.check_head(text)
    stages, _, uses = compose.parse(text)

    def roots(hexes):
        return np.frombuffer(bytes.fromhex("".join(hexes)), np.uint8).reshape(len(hexes), 32)

    return pack("kline", {
        "chain": roots(compose.chain([r for r, _ in stages])),
        "deeps": roots([d for d, _, _ in uses]),
        "stages": roots([r for r, _ in stages]),
        "supports": np.array([compose.supports(us) for _, _, us in uses], np.bool_),
        "targets": roots([t for _, t, _ in uses]),
    })


# --- TensorFlow ------------------------------------------------------------------------------


def to_tf(blob):
    """{name: tf.Tensor}, each parsed by TensorFlow from the TensorProto its record spells."""
    import tensorflow as tf
    from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2

    out = {}
    for name, a in unpack(blob)[1].items():
        proto = tensor_pb2.TensorProto(
            dtype=CODES[a.dtype],
            tensor_shape=tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=d) for d in a.shape]),
            tensor_content=a.tobytes(),
        )
        out[name] = tf.io.parse_tensor(proto.SerializeToString(), CODES[a.dtype])
    return out


def from_tf(kind, tensors):
    """tmap1 bytes from {name: tf.Tensor}; dtype numbers are TensorFlow's own."""
    return pack(kind, {name: t.numpy() for name, t in tensors.items()})


# --- files -----------------------------------------------------------------------------------


def of_file(kind, path):
    p = Path(path)
    return {
        "web": lambda: of_web(p.read_bytes()),
        "sequence": lambda: of_sequence(dna.read_fasta(p)),
        "pool": lambda: of_pool(dna.read_pool(p)),
        "loop": lambda: of_loop(p.read_bytes().decode("ascii")),
        "deep": lambda: of_deep(p.read_bytes().decode("ascii")),
        "kline": lambda: of_kline(p.read_bytes().decode("ascii")),
    }[kind]()


def inverse(blob):
    """The file an exact map came from; a view has no inverse and is refused as `kind_mismatch`."""
    kind = unpack(blob)[0]
    if kind == "web":
        return to_web(blob)
    if kind == "sequence":
        seq = to_sequence(blob)
        lines = "\n".join(seq[i : i + dna.LINE] for i in range(0, len(seq), dna.LINE))
        return f">tmap1:{root(blob)} nt={len(seq)}\n{lines}\n".encode("ascii")
    if kind == "pool":
        return to_pool(blob)
    raise Refusal("kind_mismatch", f"{kind} is not an exact map ({', '.join(EXACT)})")


def show(blob):
    kind, tensors = unpack(blob)
    lines = [f"tmap1:{root(blob)} kind={kind} tensors={len(tensors)} bytes={len(blob)}"]
    lines += [f"  {n} {a.dtype.name} DataType={CODES[a.dtype]} shape={list(a.shape)}" for n, a in tensors.items()]
    return "\n".join(lines) + "\n"


def write(path, data):
    if Path(path).exists():
        raise Refusal("exists", path)
    with open(path, "xb") as f:
        f.write(data)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tensor", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="write the tmap1 tensor map of an object file")
    m.add_argument("kind", choices=EXACT + VIEWS)
    m.add_argument("src")
    m.add_argument("out")
    sub.add_parser("show", help="print a tensor map's root, kind, names, dtypes and shapes").add_argument("tmap")
    u = sub.add_parser("unmap", help="write back the file an exact tensor map came from")
    u.add_argument("tmap")
    u.add_argument("out")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "map":
            out = of_file(a.kind, a.src)
        elif a.cmd == "show":
            sys.stdout.write(show(Path(a.tmap).read_bytes()))
            return 0
        else:
            out = inverse(Path(a.tmap).read_bytes())
        write(a.out, out)
        print(root(out), len(out))
    except (Refusal, UnicodeDecodeError) as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
