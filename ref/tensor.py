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
import dna  # noqa: E402
import loop  # noqa: E402
import web  # noqa: E402

MAGIC = b"tmap1"
MAX_BYTES = 1 << 30
MAX_RANK = 8
MAX_TENSORS = 1024
KIND = re.compile(rb"[a-z][a-z0-9-]{0,31}")
NAME = re.compile(rb"[a-z][a-z0-9_]{0,63}")

# TensorFlow DataType numbers (tensorflow/core/framework/types.proto) and their numpy forms.
DTYPES = {
    1: "<f4",  # DT_FLOAT
    2: "<f8",  # DT_DOUBLE
    3: "<i4",  # DT_INT32
    4: "|u1",  # DT_UINT8
    5: "<i2",  # DT_INT16
    6: "|i1",  # DT_INT8
    9: "<i8",  # DT_INT64
    10: "|b1",  # DT_BOOL
    17: "<u2",  # DT_UINT16
    19: "<f2",  # DT_HALF
    22: "<u4",  # DT_UINT32
    23: "<u8",  # DT_UINT64
}
CODES = {np.dtype(v): k for k, v in DTYPES.items()}

# Kinds this repository maps; a neighbour defines its own kind under the same bytes and rules.
EXACT = ("media", "web", "sequence", "pool")
VIEWS = ("glyph", "loop")
CODE = {b: i for i, b in enumerate(dna.BASES)}
LOOP_KINDS = ("think", "hypothesis", "act", "observe")
VERDICTS = {"supports": 1, "refutes": 2}


Refusal = web.Refusal


# --- bytes ----------------------------------------------------------------------------------


def check(name, a):
    """Refuse values that have no single byte form: NaN, infinity, negative zero, a bool not 0/1."""
    if a.dtype.kind == "f" and (not np.all(np.isfinite(a)) or np.any(np.signbit(a) & (a == 0))):
        raise Refusal("bad_value", f"{name}: NaN, infinity or negative zero")


def pack(kind, tensors):
    """Serialize {name: array} under `kind` to canonical tmap1 bytes, names in ascending order."""
    k = kind.encode("ascii")
    if not KIND.fullmatch(k):
        raise Refusal("bad_kind", kind)
    if not 1 <= len(tensors) <= MAX_TENSORS:
        raise Refusal("bad_count", str(len(tensors)))
    out = [MAGIC, bytes([len(k)]), k, struct.pack("<H", len(tensors))]
    for name in sorted(tensors, key=lambda s: s.encode("ascii")):
        n = name.encode("ascii")
        if not NAME.fullmatch(n):
            raise Refusal("bad_name", name)
        a = np.asarray(tensors[name])
        code = CODES.get(a.dtype.newbyteorder("<") if a.dtype.byteorder == ">" else a.dtype)
        if code is None:
            raise Refusal("bad_dtype", f"{name}: {a.dtype}")
        if a.ndim > MAX_RANK or any(d >= 1 << 32 for d in a.shape):
            raise Refusal("bad_shape", name)
        a = np.asarray(a, DTYPES[code]).copy(order="C")
        check(name, a)
        out += [bytes([len(n)]), n, bytes([code, a.ndim]), struct.pack(f"<{a.ndim}I", *a.shape), a.tobytes()]
    blob = b"".join(out)
    if len(blob) > MAX_BYTES:
        raise Refusal("too_large", f"{len(blob)} bytes")
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

    k = take(take(1)[0])
    if not KIND.fullmatch(k):
        raise Refusal("bad_kind", k.decode("ascii", "replace"))
    (count,) = struct.unpack("<H", take(2))
    if not 1 <= count <= MAX_TENSORS:
        raise Refusal("bad_count", str(count))
    tensors, last = {}, b""
    for _ in range(count):
        n = take(take(1)[0])
        if not NAME.fullmatch(n):
            raise Refusal("bad_name", n.decode("ascii", "replace"))
        if n <= last:
            raise Refusal("bad_order", n.decode("ascii"))
        last = n
        code, rank = take(2)
        if code not in DTYPES:
            raise Refusal("bad_dtype", f"{n.decode()}: DataType {code}")
        if rank > MAX_RANK:
            raise Refusal("bad_shape", n.decode())
        shape = struct.unpack(f"<{rank}I", take(4 * rank))
        dt = np.dtype(DTYPES[code])
        raw = take(int(np.prod(shape, dtype=np.int64)) * dt.itemsize)
        if code == 10 and any(b > 1 for b in raw):
            raise Refusal("bad_value", f"{n.decode()}: bool not 0 or 1")
        a = np.frombuffer(raw, dt).reshape(shape)
        check(n.decode(), a)
        tensors[n.decode("ascii")] = a
    if pos != len(blob):
        raise Refusal("trailing_bytes")
    kind_s = k.decode("ascii")
    if kind is not None and kind_s != kind:
        raise Refusal("kind_mismatch", f"{kind_s}, wanted {kind}")
    return kind_s, tensors


def root(blob):
    return hashlib.sha256(blob).hexdigest()


def records(blob):
    """Each tensor record as (name, DataType, shape, tensor_content): the fields of its TensorProto."""
    _, tensors = unpack(blob)
    return [(n, CODES[a.dtype], a.shape, a.tobytes()) for n, a in tensors.items()]


# --- maps: the objects of this repository ----------------------------------------------------


def of_media(media):
    a = np.asarray(media)
    if a.dtype != np.uint8 or a.ndim < 2:
        raise Refusal("bad_media", "media is a uint8 grid with a channel axis")
    return pack("media", {"media": a})


def to_media(blob):
    return unpack(blob, "media")[1]["media"].copy()


def of_web(blob):
    """A web as its header fields and its six weight tensors; the weight bytes are the web's own."""
    u = web.unpack(blob)
    names = ("w1", "b1", "w2", "b2", "w3", "b3")
    t = {n: p.astype("<f2") for n, p in zip(names, u["params"])}
    t["activation"] = np.array(web.ACT_SINE, np.uint8)
    t["omega0"] = np.array(u["omega0"], np.uint16)
    t["grid"] = np.array(u["grid"], np.uint16)
    return pack("web", t)


def to_web(blob):
    t = unpack(blob, "web")[1]
    need = {"activation", "omega0", "grid", "w1", "b1", "w2", "b2", "w3", "b3"}
    if set(t) != need or t["activation"].shape != () or t["activation"] != web.ACT_SINE:
        raise Refusal("bad_map", "web tensors")
    grid = tuple(int(g) for g in t["grid"])
    if t["grid"].ndim != 1 or t["w1"].ndim != 2 or t["w3"].ndim != 2 or t["omega0"].shape != ():
        raise Refusal("bad_map", "web shapes")
    hidden, d_out = t["w1"].shape[1], t["w3"].shape[1]
    params = [t[n] for n in ("w1", "b1", "w2", "b2", "w3", "b3")]
    if any(p.dtype != np.float16 or p.shape != s for p, s in zip(params, web.shapes(len(grid), d_out, hidden))):
        raise Refusal("bad_map", "weight shapes")
    out = web.pack(grid, d_out, hidden, int(t["omega0"]), params)
    if of_web(out) != blob:
        raise Refusal("bad_map", "not the canonical web tensors")
    return out


def codes(seq):
    return np.frombuffer(seq.encode("ascii"), np.uint8).copy()


def of_sequence(seq):
    """A sequence as one uint8 tensor of base indices into ACGT; one-hot is a view of it."""
    if not seq or seq.strip(dna.BASES):
        raise Refusal("bad_base", "a sequence is non-empty ACGT")
    lut = np.full(256, 255, np.uint8)
    for b, i in CODE.items():
        lut[ord(b)] = i
    return pack("sequence", {"bases": lut[codes(seq)]})


def bases(a):
    if a.dtype != np.uint8 or np.any(a > 3):
        raise Refusal("bad_base", "base index outside 0-3")
    return np.frombuffer(b"ACGT", np.uint8)[a].tobytes().decode("ascii")


def to_sequence(blob):
    t = unpack(blob, "sequence")[1]
    if set(t) != {"bases"} or t["bases"].ndim != 1 or t["bases"].size == 0:
        raise Refusal("bad_map", "sequence tensors")
    return bases(t["bases"])


def of_pool(oligos, file_id):
    """A pool1 pool as [oligos, 117] base indices in pool order, and its file id."""
    if not 0 <= file_id <= 8:
        raise Refusal("bad_file", str(file_id))
    if not oligos or any(len(o) != dna.OLIGO_NT or o.strip(dna.BASES) for o in oligos):
        raise Refusal("bad_pool", "a pool is 117-nt ACGT oligos")
    seq = unpack(of_sequence("".join(oligos)))[1]["bases"]
    return pack("pool", {"file": np.array(file_id, np.uint8), "oligos": seq.reshape(len(oligos), dna.OLIGO_NT)})


def to_pool(blob):
    t = unpack(blob, "pool")[1]
    if set(t) != {"file", "oligos"} or t["file"].shape != () or t["oligos"].ndim != 2 or t["oligos"].shape[1] != dna.OLIGO_NT:
        raise Refusal("bad_map", "pool tensors")
    return [bases(row) for row in t["oligos"]], int(t["file"])


def pool_bytes(oligos, file_id):
    """The pool file `dna.py pool` writes, so the pool tensor map is exact against its pinned sha256."""
    return "".join(f">pool1 file={file_id} i={i}\n{o}\n" for i, o in enumerate(oligos)).encode("ascii")


def of_glyph(seq, k):
    """A cgr1 glyph's k-mer counts on the chaos-game grid (a view of the sequence, §11)."""
    if not 1 <= k <= 6:
        raise Refusal("bad_header", "k must be 1-6")
    return pack("glyph", {"counts": dna.kmer_counts(seq, k).astype("<i8"), "k": np.array(k, np.uint8)})


def of_loop(text):
    """A loop1 file as its hash chain h_0 .. head, its frame kinds and its written verdicts (a view, §12)."""
    blocks, h = loop.split(text)
    if h != loop.chain(blocks):
        raise Refusal("bad_head", f"head is {loop.chain(blocks)}")
    hs = [hashlib.sha256(loop.TAG).digest()]
    for b in blocks:
        hs.append(hashlib.sha256(hs[-1] + b.encode("ascii")).digest())
    kinds, verdicts = [], []
    for b in blocks[1:]:
        f = b.split("\n", 1)[0].split()
        kinds.append(LOOP_KINDS.index(f[2]))
        v = [p.split("=", 1)[1] for p in f[3:] if p.startswith("verdict=")]
        verdicts.append(VERDICTS[v[0]] if v else 0)
    chain = np.frombuffer(b"".join(hs), np.uint8).reshape(len(hs), 32)
    return pack("loop", {"chain": chain, "kind": np.array(kinds, np.uint8), "verdict": np.array(verdicts, np.uint8)})


def digest(hexroot):
    """A 32-byte root as the uint8 [32] tensor that carries it across a boundary."""
    if not re.fullmatch(r"[0-9a-f]{64}", hexroot):
        raise Refusal("root_mismatch", "a root is 64 lowercase hex digits")
    return np.frombuffer(bytes.fromhex(hexroot), np.uint8).copy()


# --- TensorFlow ------------------------------------------------------------------------------


def to_tf(blob):
    """{name: tf.Tensor}, each parsed by TensorFlow from the TensorProto its record spells."""
    import tensorflow as tf
    from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2

    out = {}
    for name, code, shape, content in records(blob):
        proto = tensor_pb2.TensorProto(
            dtype=code,
            tensor_shape=tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=d) for d in shape]),
            tensor_content=content,
        )
        out[name] = tf.io.parse_tensor(proto.SerializeToString(), tf.dtypes.as_dtype(code))
    return out


def from_tf(kind, tensors):
    """tmap1 bytes from {name: tf.Tensor}; dtype numbers are TensorFlow's own."""
    arrays = {}
    for name, t in tensors.items():
        a = t.numpy()
        if CODES.get(a.dtype) != t.dtype.as_datatype_enum:
            raise Refusal("bad_dtype", f"{name}: {t.dtype.name}")
        arrays[name] = a
    return pack(kind, arrays)


# --- files -----------------------------------------------------------------------------------


def of_file(kind, path, k=3, file_id=0):
    p = Path(path)
    if kind == "media":
        return of_media(web.read_png(p))
    if kind == "web":
        return of_web(p.read_bytes())
    if kind == "sequence":
        return of_sequence(dna.read_fasta(p))
    if kind == "pool":
        return of_pool(dna.read_pool(p), file_id)
    if kind == "glyph":
        return of_glyph(dna.read_fasta(p), k)
    if kind == "loop":
        return of_loop(p.read_bytes().decode("ascii"))
    raise Refusal("bad_kind", kind)


def inverse(blob):
    """The object an exact map came from, as the bytes of its file; a view has no inverse."""
    kind = unpack(blob)[0]
    if kind == "web":
        return to_web(blob)
    if kind == "sequence":
        seq = to_sequence(blob)
        return f">tmap1:{root(blob)} nt={len(seq)}\n".encode("ascii") + "\n".join(
            seq[i : i + dna.LINE] for i in range(0, len(seq), dna.LINE)
        ).encode("ascii") + b"\n"
    if kind == "pool":
        return pool_bytes(*to_pool(blob))
    if kind == "media":
        raise Refusal("bad_kind", "write media with ref/web.py; a PNG has many byte forms")
    raise Refusal("view", f"{kind} is a view, never read back into data")


def show(blob):
    kind, tensors = unpack(blob)
    lines = [f"tmap1:{root(blob)} kind={kind} tensors={len(tensors)} bytes={len(blob)}"]
    for n, a in tensors.items():
        lines.append(f"  {n} {np.dtype(DTYPES[CODES[a.dtype]]).name} DataType={CODES[a.dtype]} shape={list(a.shape)}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tensor", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="write the tmap1 tensor map of an object file")
    m.add_argument("kind", choices=EXACT + VIEWS)
    m.add_argument("src")
    m.add_argument("out")
    m.add_argument("--k", type=int, default=3, help="glyph k, 1-6")
    m.add_argument("--file", type=int, default=0, help="pool file id, 0-8")
    m.add_argument("--root", help="refuse unless the map hashes to this tensor root")
    s = sub.add_parser("show", help="print a tensor map's root, kind, names, dtypes and shapes")
    s.add_argument("tmap")
    u = sub.add_parser("unmap", help="write back the object an exact tensor map came from")
    u.add_argument("tmap")
    u.add_argument("out")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "map":
            blob = of_file(a.kind, a.src, a.k, a.file)
            if a.root and root(blob) != a.root:
                raise Refusal("root_mismatch", root(blob))
            if Path(a.out).exists():
                raise Refusal("exists", a.out)
            with open(a.out, "xb") as f:
                f.write(blob)
            print(root(blob), len(blob))
        elif a.cmd == "show":
            sys.stdout.write(show(Path(a.tmap).read_bytes()))
        else:
            out = inverse(Path(a.tmap).read_bytes())
            if Path(a.out).exists():
                raise Refusal("exists", a.out)
            with open(a.out, "xb") as f:
                f.write(out)
            print(hashlib.sha256(out).hexdigest(), len(out))
    except (Refusal, UnicodeDecodeError) as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
