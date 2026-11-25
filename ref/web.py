"""Reference encoder/decoder for `web` files: static media stored as a 2-hidden-layer sine network.

Needs only Python 3.10+ and numpy. The decoder is normative (docs/follow-agent-contract.md §3);
the encoder is one way to fit weights and is not required to be reproducible.
"""

import argparse
import hashlib
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

MAGIC = b"web1"
ACT_SINE = 1
DTYPE_F16 = 1
MAX_BYTES = 64 * 1024
DECODE_ROWS = 4096  # grid points per forward pass, bounding decode memory


class Refusal(Exception):
    """A web or media file that the contract refuses; `code` is the refusal code."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


# --- PNG (8-bit, non-interlaced, gray / RGB / RGBA) -------------------------------------------


def read_png(path):
    try:
        return _read_png(Path(path).read_bytes())
    except (struct.error, zlib.error, ValueError, IndexError) as err:
        raise Refusal("bad_media", f"damaged PNG ({err})")


def _read_png(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise Refusal("bad_media", "not a PNG")
    pos, idat, ihdr, end = 8, b"", None, False
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n : pos + 12 + n])
        if zlib.crc32(kind + body) != crc:
            raise Refusal("bad_media", f"bad CRC in {kind!r}")
        pos += 12 + n
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            end = True
            break
    if ihdr is None or not end:
        raise Refusal("bad_media", "missing IHDR or IEND")
    w, h, depth, ctype, _, _, interlace = ihdr
    channels = {0: 1, 2: 3, 6: 4}.get(ctype)
    if depth != 8 or channels is None or interlace != 0:
        raise Refusal("bad_media", "only 8-bit non-interlaced gray, RGB or RGBA")
    raw = zlib.decompress(idat)
    stride = w * channels
    if len(raw) != h * (stride + 1):
        raise Refusal("bad_media", "pixel data has the wrong length")
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        f = raw[y * (stride + 1)]
        line = np.frombuffer(raw, np.uint8, stride, y * (stride + 1) + 1).astype(np.int32)
        if f == 0:
            cur = line
        elif f == 2:
            cur = (line + prev) & 255
        else:
            cur = np.zeros(stride, dtype=np.int32)
            for i in range(stride):
                a = cur[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                if f == 1:
                    p = a
                elif f == 3:
                    p = (a + b) >> 1
                elif f == 4:
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    p = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                else:
                    raise Refusal("bad_media", f"filter {f}")
                cur[i] = (line[i] + p) & 255
        out[y] = cur
        prev = cur
    img = out.reshape(h, w, channels)
    return img[:, :, :3] if channels == 4 else img


def write_png(path, img):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim == 2:
        img = img[:, :, None]
    h, w, c = img.shape
    ctype = {1: 0, 3: 2}[c]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def shrink(img, factor):
    """Box-mean downsample by an integer factor, rounding half to even."""
    h, w, c = img.shape
    if factor < 1 or h % factor or w % factor:
        raise Refusal("bad_media", "size is not a multiple of the factor")
    blocks = img.reshape(h // factor, factor, w // factor, factor, c).astype(np.float64)
    return np.rint(blocks.mean(axis=(1, 3))).astype(np.uint8)


# --- web format -----------------------------------------------------------------------------


def shapes(d_in, d_out, hidden):
    return [(d_in, hidden), (hidden,), (hidden, hidden), (hidden,), (hidden, d_out), (d_out,)]


def pack(grid, d_out, hidden, omega0, params):
    """Serialize a web to its canonical bytes. `params` is [W1, b1, W2, b2, W3, b3]."""
    try:
        head = MAGIC + struct.pack("<BHB", ACT_SINE, omega0, len(grid))
        head += struct.pack("<" + "H" * len(grid), *grid) + struct.pack("<BHB", d_out, hidden, DTYPE_F16)
    except struct.error:
        raise Refusal("bad_header", "field out of range")
    body = b""
    for p, s in zip(params, shapes(len(grid), d_out, hidden)):
        q = np.asarray(p, dtype=np.float64).reshape(s).astype("<f2")
        q[q == 0] = 0  # drop negative zero
        if not np.all(np.isfinite(q)):
            raise Refusal("bad_weight", "weight overflows float16")
        body += q.tobytes()
    blob = head + body
    unpack(blob)
    return blob


def unpack(blob):
    """Parse canonical bytes, refusing anything that would not re-serialize to the same bytes."""
    if len(blob) > MAX_BYTES:
        raise Refusal("too_large", f"{len(blob)} bytes")
    try:
        if blob[:4] != MAGIC:
            raise Refusal("bad_magic")
        act, omega0, d_in = struct.unpack_from("<BHB", blob, 4)
        if act != ACT_SINE or not 1 <= omega0 <= 255 or d_in not in (1, 2, 3):
            raise Refusal("bad_header", "activation, omega0 or rank")
        grid = struct.unpack_from("<" + "H" * d_in, blob, 8)
        d_out, hidden, dtype = struct.unpack_from("<BHB", blob, 8 + 2 * d_in)
    except struct.error:
        raise Refusal("truncated")
    if min(grid) < 2 or d_out not in (1, 3) or not 1 <= hidden <= 256 or dtype != DTYPE_F16:
        raise Refusal("bad_header", "grid, channels, width or dtype")
    pos, params = 12 + 2 * d_in, []
    for s in shapes(d_in, d_out, hidden):
        n = int(np.prod(s))
        if pos + 2 * n > len(blob):
            raise Refusal("truncated")
        q = np.frombuffer(blob, "<f2", n, pos)
        if not np.all(np.isfinite(q)) or np.any(np.signbit(q) & (q == 0)):
            raise Refusal("bad_weight", "NaN, infinity or negative zero")
        params.append(q.astype(np.float64).reshape(s))
        pos += 2 * n
    if pos != len(blob):
        raise Refusal("trailing_bytes")
    return {"grid": tuple(grid), "d_out": d_out, "hidden": hidden, "omega0": omega0, "params": params}


def root(blob):
    return hashlib.sha256(blob).hexdigest()


# --- network --------------------------------------------------------------------------------


def coords(grid):
    axes = [np.linspace(-1.0, 1.0, n) for n in grid]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def forward(x, params, omega0):
    W1, b1, W2, b2, W3, b3 = params
    h1 = np.sin(omega0 * (x @ W1 + b1))
    h2 = np.sin(h1 @ W2 + b2)
    return h2 @ W3 + b3


def to_levels(y):
    return np.clip(np.rint((y + 1.0) * 127.5), 0, 255).astype(np.uint8)


def decode(blob):
    """Expand a web into media: uint8 array of shape grid + (channels,), computed in float64."""
    w = unpack(blob)
    x = coords(w["grid"])
    out = np.empty((len(x), w["d_out"]), dtype=np.uint8)
    for i in range(0, len(x), DECODE_ROWS):
        out[i : i + DECODE_ROWS] = to_levels(forward(x[i : i + DECODE_ROWS], w["params"], w["omega0"]))
    return out.reshape(*w["grid"], w["d_out"])


def encode(media, hidden=64, omega0=30, steps=10000, lr=1e-3, seed=0):
    """Fit a 2-hidden-layer sine network to `media` (uint8, grid + channels) and pack it."""
    media = np.asarray(media, dtype=np.uint8)
    grid, d_out = media.shape[:-1], media.shape[-1]
    d_in = len(grid)
    x = coords(grid).astype(np.float32)
    t = (media.reshape(-1, d_out).astype(np.float32) / 127.5) - 1.0
    rng = np.random.default_rng(seed)
    bound = np.sqrt(6.0 / hidden)
    params = [
        rng.uniform(-1.0 / d_in, 1.0 / d_in, (d_in, hidden)),
        rng.uniform(-1.0 / d_in, 1.0 / d_in, hidden),
        rng.uniform(-bound, bound, (hidden, hidden)),
        np.zeros(hidden),
        rng.uniform(-bound, bound, (hidden, d_out)),
        np.zeros(d_out),
    ]
    params = [p.astype(np.float32) for p in params]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    n = x.shape[0]
    for step in range(1, steps + 1):
        W1, b1, W2, b2, W3, b3 = params
        z1 = omega0 * (x @ W1 + b1)
        h1 = np.sin(z1)
        z2 = h1 @ W2 + b2
        h2 = np.sin(z2)
        y = h2 @ W3 + b3
        gy = 2.0 * (y - t) / (n * d_out)
        gW3, gb3 = h2.T @ gy, gy.sum(0)
        gz2 = (gy @ W3.T) * np.cos(z2)
        gW2, gb2 = h1.T @ gz2, gz2.sum(0)
        gz1 = (gz2 @ W2.T) * np.cos(z1) * omega0
        gW1, gb1 = x.T @ gz1, gz1.sum(0)
        grads = [gW1, gb1, gW2, gb2, gW3, gb3]
        for i, g in enumerate(grads):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            mh = m[i] / (1 - 0.9**step)
            vh = v[i] / (1 - 0.999**step)
            params[i] = params[i] - lr * mh / (np.sqrt(vh) + 1e-8)
    return pack(grid, d_out, hidden, omega0, params)


def psnr(a, b):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


# --- CLI ------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(prog="web", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="fit a web to a PNG")
    e.add_argument("png")
    e.add_argument("out")
    e.add_argument("--hidden", type=int, default=64)
    e.add_argument("--omega0", type=int, default=30)
    e.add_argument("--steps", type=int, default=10000)
    e.add_argument("--lr", type=float, default=1e-3)
    e.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("decode", help="expand a web to a PNG")
    d.add_argument("web")
    d.add_argument("out")
    r = sub.add_parser("root", help="print the sha256 root of a web")
    r.add_argument("web")
    c = sub.add_parser("check", help="parse, decode and compare a web with its source PNG")
    c.add_argument("web")
    c.add_argument("png")
    c.add_argument("--min-psnr", type=float, default=0.0)
    c.add_argument("--root", help="refuse unless the web hashes to this root")
    s = sub.add_parser("shrink", help="box-mean downsample a PNG")
    s.add_argument("png")
    s.add_argument("out")
    s.add_argument("factor", type=int)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "encode":
            blob = encode(read_png(a.png), a.hidden, a.omega0, a.steps, a.lr, a.seed)
            Path(a.out).write_bytes(blob)
            print(root(blob), len(blob))
        elif a.cmd == "decode":
            write_png(a.out, decode(Path(a.web).read_bytes()))
        elif a.cmd == "root":
            print(root(Path(a.web).read_bytes()))
        elif a.cmd == "check":
            blob = Path(a.web).read_bytes()
            if a.root and root(blob) != a.root:
                raise Refusal("root_mismatch", root(blob))
            ref = read_png(a.png)
            out = decode(blob)
            if out.shape != ref.shape:
                raise Refusal("shape_mismatch", f"{out.shape} != {ref.shape}")
            q = psnr(out, ref)
            print(f"root {root(blob)} bytes {len(blob)} psnr {q:.2f}")
            if q < a.min_psnr:
                raise Refusal("below_fidelity", f"{q:.2f} < {a.min_psnr}")
        elif a.cmd == "shrink":
            write_png(a.out, shrink(read_png(a.png), a.factor))
    except Refusal as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
