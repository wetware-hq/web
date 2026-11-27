"""Weight manipulation for webs after Tinker's training API: steps, checkpoints, LoRA adapters, LR rules.

Needs only Python 3.10+, numpy and the other `ref/` modules. A `Trainer` holds a web's weights
in binary32 and changes them with Tinker's two calls, `forward_backward` (accumulate gradients)
and `optim_step` (one AdamW update). Its checkpoints are tmap1 maps (contract §15): a `state`
resumes a fit, an `adapter` is a low-rank patch over a committed web that merges into a new web.
"""

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensor  # noqa: E402
import web  # noqa: E402

Refusal = web.Refusal
WEIGHTS = tensor.WEIGHTS
MATRICES = ("w1", "w2", "w3")
LORA_LR_MULTIPLIER = 10.0  # Tinker's LoRA-over-full LR factor; §15 records what webs measure


@dataclass(frozen=True)
class AdamParams:
    """AdamW hyperparameters for one `optim_step`; the defaults are the reference encoder's Adam."""

    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


def init_params(d_in, d_out, hidden, seed):
    """The reference encoder's initial weights (SIREN first layer), binary32."""
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
    return {n: p.astype(np.float32) for n, p in zip(WEIGHTS, params)}


def targets(media):
    """Grid coordinates and targets in [-1, 1] for uint8 media of shape grid + (channels,)."""
    media = np.asarray(media, dtype=np.uint8)
    x = web.coords(media.shape[:-1]).astype(np.float32)
    t = (media.reshape(-1, media.shape[-1]).astype(np.float32) / 127.5) - 1.0
    return x, t


class Trainer:
    """A web being fitted. `train` holds what `optim_step` changes; an adapter trainer also has a frozen base."""

    def __init__(self, grid, d_out, hidden, omega0, train, base=None, rank=0):
        self.grid, self.d_out, self.hidden, self.omega0 = tuple(grid), d_out, hidden, omega0
        self.train = {n: np.asarray(a, np.float32) for n, a in train.items()}
        self.base = base  # web bytes of the frozen base, or None for a full fit
        self.frozen = None if base is None else {n: p.astype(np.float32) for n, p in zip(WEIGHTS, web.unpack(base)["params"])}
        self.rank = rank
        self.m = {n: np.zeros_like(a) for n, a in self.train.items()}
        self.v = {n: np.zeros_like(a) for n, a in self.train.items()}
        self.grads = {}
        self.step = 0

    # --- create ---------------------------------------------------------------------------

    @classmethod
    def fresh(cls, grid, d_out, hidden=64, omega0=30, seed=0):
        return cls(grid, d_out, hidden, omega0, init_params(len(grid), d_out, hidden, seed))

    @classmethod
    def from_web(cls, blob):
        """Warm start: a committed web's weights, widened from binary16, with fresh optimizer state."""
        u = web.unpack(blob)
        return cls(u["grid"], u["d_out"], u["hidden"], u["omega0"], dict(zip(WEIGHTS, u["params"])))

    @classmethod
    def lora(cls, blob, rank=8, seed=0):
        """Freeze a committed web and train W' = W + down @ up on each matrix; `up` starts at zero."""
        u = web.unpack(blob)
        if rank < 1:
            raise Refusal("bad_header", "rank must be at least 1")
        rng = np.random.default_rng(seed)
        train = {}
        for n, p in zip(WEIGHTS, u["params"]):
            if n in MATRICES:
                r = min(rank, *p.shape)
                train[n + "_down"] = rng.uniform(-1.0, 1.0, (p.shape[0], r)) / np.sqrt(p.shape[0])
                train[n + "_up"] = np.zeros((r, p.shape[1]))
        return cls(u["grid"], u["d_out"], u["hidden"], u["omega0"], train, base=blob, rank=rank)

    @classmethod
    def from_state(cls, blob, with_optimizer=True):
        """Resume from a `state` map; without the optimizer, Adam's moments and step restart at zero."""
        t = tensor.unpack(blob, "state")[1]
        try:
            grid, omega0, d_out = tuple(int(g) for g in t["grid"]), int(t["omega0"]), int(t["d_out"])
            names = sorted(n[2:] for n in t if n.startswith("m_"))
            if sorted(n[2:] for n in t if n.startswith("v_")) != names or not names:
                raise KeyError("moments")
            base = None
            if "base" in t:
                base = web.pack(grid, d_out, t["w1"].shape[1], omega0, [t[n] for n in WEIGHTS])
                if web.root(base) != t["base"].tobytes().hex():
                    raise Refusal("root_mismatch", "state base weights do not hash to its base root")
            train = {n: t[n] for n in names}
            hidden = (t["w1"] if base is not None else train["w1"]).shape[1]
            self = cls(grid, d_out, hidden, omega0, train, base=base, rank=int(t["rank"]) if base is not None else 0)
            if with_optimizer:
                self.m = {n: t["m_" + n].astype(np.float32) for n in names}
                self.v = {n: t["v_" + n].astype(np.float32) for n in names}
                self.step = int(t["step"])
        except (KeyError, ValueError, TypeError, IndexError) as err:
            raise tensor.bad(f"state: {err}") from None
        return self

    # --- train ----------------------------------------------------------------------------

    def weights(self):
        """The effective binary32 weights; an adapter adds down @ up to each frozen matrix."""
        if self.base is None:
            return [self.train[n] for n in WEIGHTS]
        return [self.frozen[n] + self.train[n + "_down"] @ self.train[n + "_up"] if n in MATRICES else self.frozen[n] for n in WEIGHTS]

    def forward_backward(self, x, t):
        """Mean-squared loss on a batch and its gradients, added to those accumulated since the last step."""
        W1, b1, W2, b2, W3, b3 = self.weights()
        omega0 = self.omega0
        z1 = omega0 * (x @ W1 + b1)
        h1 = np.sin(z1)
        z2 = h1 @ W2 + b2
        h2 = np.sin(z2)
        y = h2 @ W3 + b3
        n, d_out = t.shape
        gy = 2.0 * (y - t) / (n * d_out)
        gW3, gb3 = h2.T @ gy, gy.sum(0)
        gz2 = (gy @ W3.T) * np.cos(z2)
        gW2, gb2 = h1.T @ gz2, gz2.sum(0)
        gz1 = (gz2 @ W2.T) * np.cos(z1) * omega0
        gW1, gb1 = x.T @ gz1, gz1.sum(0)
        g = dict(zip(WEIGHTS, (gW1, gb1, gW2, gb2, gW3, gb3)))
        if self.base is not None:
            g = {k: v for n in MATRICES for k, v in ((n + "_down", g[n] @ self.train[n + "_up"].T), (n + "_up", self.train[n + "_down"].T @ g[n]))}
        for k, v in g.items():
            self.grads[k] = self.grads[k] + v if k in self.grads else v
        return float(np.mean((y - t) ** 2))

    def optim_step(self, p=AdamParams()):
        """One AdamW update from the accumulated gradients, which are then cleared."""
        if not self.grads:
            raise Refusal("no_gradients", "call forward_backward before optim_step")
        self.step += 1
        scale = 1.0
        if p.grad_clip_norm > 0:
            norm = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in self.grads.values()))
            scale = min(1.0, p.grad_clip_norm / (norm + 1e-12))
        lr, b1, b2 = p.learning_rate, p.beta1, p.beta2
        for k in sorted(self.train):
            g = self.grads[k] if scale == 1.0 else self.grads[k] * np.float32(scale)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1**self.step)
            vh = self.v[k] / (1 - b2**self.step)
            if p.weight_decay:
                self.train[k] = self.train[k] * np.float32(1 - lr * p.weight_decay)
            self.train[k] = self.train[k] - lr * mh / (np.sqrt(vh) + p.eps)
        self.grads = {}

    # --- save -----------------------------------------------------------------------------

    def save_weights(self):
        """A canonical web (Tinker's sampler checkpoint); an adapter's is `merge(base, save_adapter())`."""
        if self.base is not None:
            return merge(self.base, self.save_adapter())
        return web.pack(self.grid, self.d_out, self.hidden, self.omega0, self.weights())

    def save_adapter(self):
        """The low-rank patch as a tmap1 `adapter` map: binary16 factors and the base web root."""
        if self.base is None:
            raise Refusal("kind_mismatch", "a full fit has no adapter; use save_weights")
        t = {k: f16(a) for k, a in self.train.items()}
        t["base"] = np.frombuffer(bytes.fromhex(web.root(self.base)), np.uint8)
        return tensor.pack("adapter", t)

    def save_state(self):
        """Everything needed to resume, as a tmap1 `state` map: binary32 weights, Adam moments and the step."""
        t = {"grid": np.array(self.grid, np.uint16), "omega0": np.uint16(self.omega0), "d_out": np.uint8(self.d_out), "step": np.int64(self.step)}
        t |= {k: clean(a) for k, a in self.train.items()}
        t |= {"m_" + k: clean(a) for k, a in self.m.items()} | {"v_" + k: clean(a) for k, a in self.v.items()}
        if self.base is not None:
            t |= {n: f16(p) for n, p in zip(WEIGHTS, web.unpack(self.base)["params"])}
            t |= {"base": np.frombuffer(bytes.fromhex(web.root(self.base)), np.uint8), "rank": np.uint16(self.rank)}
        return tensor.pack("state", t)


def clean(a):
    a = np.asarray(a, np.float32).copy()
    a[a == 0] = 0  # drop negative zero
    return a


def f16(a):
    q = np.asarray(a, np.float64).astype("<f2")
    q[q == 0] = 0
    if not np.all(np.isfinite(q)):
        raise Refusal("bad_weight", "adapter factor overflows float16")
    return q


def merge(base, adapter):
    """W' = W + down @ up in binary64 for each patched matrix, packed as a new canonical web."""
    t = tensor.unpack(adapter, "adapter")[1]
    if "base" not in t or t["base"].dtype != np.uint8 or t["base"].shape != (32,):
        raise tensor.bad("adapter base root")
    if t["base"].tobytes().hex() != web.root(base):
        raise Refusal("root_mismatch", f"adapter is for web1:{t['base'].tobytes().hex()}")
    u = web.unpack(base)
    names = {n for n in t if n != "base"}
    patched = sorted({n.rsplit("_", 1)[0] for n in names})
    if not patched or names != {f"{n}_{s}" for n in patched for s in ("down", "up")} or not set(patched) <= set(MATRICES):
        raise tensor.bad("adapter tensors")
    params = dict(zip(WEIGHTS, u["params"]))
    for n in patched:
        down, up = t[n + "_down"], t[n + "_up"]
        if down.dtype != np.float16 or up.dtype != np.float16 or down.ndim != 2 or up.ndim != 2 or down.shape[0] != params[n].shape[0] or up.shape != (down.shape[1], params[n].shape[1]):
            raise tensor.bad(f"adapter {n} shapes")
        params[n] = params[n] + down.astype(np.float64) @ up.astype(np.float64)
    return web.pack(u["grid"], u["d_out"], u["hidden"], u["omega0"], [params[n] for n in WEIGHTS])


# --- heuristics (Tinker's hyperparam_utils, for webs) ----------------------------------------


def linear_decay(lr, step, steps):
    """The learning rate at `step` (1-based) of a linear decay to zero over `steps`."""
    return lr * (1.0 - (step - 1) / steps)


def batch_lr(lr, batch, ref_batch):
    """Move a learning rate tuned at `ref_batch` to `batch` by the square-root rule."""
    return lr * math.sqrt(batch / ref_batch)


def lora_lr(full_lr, multiplier=LORA_LR_MULTIPLIER):
    """A starting adapter learning rate from a full-fit one."""
    return full_lr * multiplier


def param_count(hidden, d_in, d_out, rank=0):
    """Trainable parameters: the whole web for a full fit, the factors of each matrix for an adapter."""
    if not rank:
        return d_in * hidden + hidden + hidden * hidden + hidden + hidden * d_out + d_out
    return sum(min(rank, a, b) * (a + b) for a, b in ((d_in, hidden), (hidden, hidden), (hidden, d_out)))


def fit(trainer, media, steps, params=AdamParams(), schedule="constant", batch=0, seed=0, every=0, log=None):
    """Run `steps` of forward_backward + optim_step on `media`, full batch unless `batch` > 0."""
    x, t = targets(media)
    rng = np.random.default_rng(seed)
    for s in range(1, steps + 1):
        if batch:
            i = rng.choice(len(x), batch, replace=False)
            loss = trainer.forward_backward(x[i], t[i])
        else:
            loss = trainer.forward_backward(x, t)
        p = params if schedule == "constant" else AdamParams(**(asdict(params) | {"learning_rate": linear_decay(params.learning_rate, s, steps)}))
        trainer.optim_step(p)
        if log and every and s % every == 0:
            log(trainer.step, loss)
    return trainer


# --- CLI -------------------------------------------------------------------------------------


def write(path, data):
    if Path(path).exists():
        raise Refusal("exists", f"{path}; a new fit is a new file")
    with open(path, "xb") as f:
        f.write(data)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tune", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit or continue fitting a web; write a web, a state or an adapter")
    f.add_argument("png")
    f.add_argument("out", help="the web, or the adapter with --rank")
    src = f.add_mutually_exclusive_group()
    src.add_argument("--from-web", help="warm start from this web (with --rank: the frozen base)")
    src.add_argument("--from-state", help="resume from this state map")
    f.add_argument("--no-optimizer", action="store_true", help="with --from-state, restart Adam's moments")
    f.add_argument("--rank", type=int, default=0, help="train a LoRA adapter of this rank over --from-web")
    f.add_argument("--state", help="also write the state map here, to resume later")
    f.add_argument("--steps", type=int, default=10000)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--beta2", type=float, default=0.999)
    f.add_argument("--eps", type=float, default=1e-8)
    f.add_argument("--weight-decay", type=float, default=0.0)
    f.add_argument("--clip", type=float, default=0.0, help="global gradient norm clip; 0 disables")
    f.add_argument("--schedule", choices=("constant", "linear"), default="constant")
    f.add_argument("--batch", type=int, default=0, help="minibatch rows per step; 0 is full batch")
    f.add_argument("--hidden", type=int, default=64)
    f.add_argument("--omega0", type=int, default=30)
    f.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("merge", help="apply an adapter to its base web, writing a new web")
    m.add_argument("web")
    m.add_argument("adapter")
    m.add_argument("out")
    c = sub.add_parser("count", help="trainable parameters of a full fit or an adapter")
    c.add_argument("--hidden", type=int, default=64)
    c.add_argument("--rank", type=int, default=0)
    c.add_argument("--d-in", type=int, default=2)
    c.add_argument("--d-out", type=int, default=3)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "count":
            print(param_count(a.hidden, a.d_in, a.d_out, a.rank))
            return 0
        if a.cmd == "merge":
            out = merge(Path(a.web).read_bytes(), Path(a.adapter).read_bytes())
            write(a.out, out)
            print(web.root(out), len(out))
            return 0
        media = web.read_png(a.png)
        if a.rank and not a.from_web:
            raise Refusal("bad_header", "--rank needs --from-web, the base the adapter patches")
        if a.from_state:
            tr = Trainer.from_state(Path(a.from_state).read_bytes(), with_optimizer=not a.no_optimizer)
        elif a.from_web and a.rank:
            tr = Trainer.lora(Path(a.from_web).read_bytes(), a.rank, a.seed)
        elif a.from_web:
            tr = Trainer.from_web(Path(a.from_web).read_bytes())
        else:
            tr = Trainer.fresh(media.shape[:-1], media.shape[-1], a.hidden, a.omega0, a.seed)
        if tr.grid != media.shape[:-1] or tr.d_out != media.shape[-1]:
            raise Refusal("shape_mismatch", f"{tr.grid + (tr.d_out,)} != {media.shape}")
        p = AdamParams(a.lr, 0.9, a.beta2, a.eps, a.weight_decay, a.clip)
        fit(tr, media, a.steps, p, a.schedule, a.batch, a.seed)
        out = tr.save_adapter() if tr.base is not None else tr.save_weights()
        write(a.out, out)
        if a.state:
            write(a.state, tr.save_state())
        merged = tr.save_weights()
        print(f"{'tmap1' if tr.base is not None else 'web1'}:{web.root(out)} bytes {len(out)} psnr {web.psnr(web.decode(merged), media):.2f}")
    except Refusal as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
