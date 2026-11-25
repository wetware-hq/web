# `encode-media` — fit a web to an image

`type: resource.skill` · `spec: agent.skill` · subordinate to `docs/follow-agent-contract.md`, which governs on any conflict.

Fit the weights of a 2-hidden-layer sine network to one image and write them as a canonical web.

## Specification

```json { type: notebook.cell }
{
  "id": "skills/encode-media.md",
  "type": "resource.skill",
  "role": "turn pinned media into web bytes",
  "reads": ["PNG", "docs/pin-sources.md"],
  "writes": ["*.web"],
  "must": ["pin the source sha256 first", "pack through §2 only", "record fidelity with the root"],
  "check": ["W1", "W4", "W5", "W7"]
}
```

## Graph

- ← [../docs/follow-agent-contract.md](../docs/follow-agent-contract.md)
- ← [../docs/pin-sources.md](../docs/pin-sources.md)
- → [decode-web.md](decode-web.md)

## Indications

- A new image needs a web, or an existing web needs a better fit (which makes a new web, contract §1).

## Contraindications

- Never re-encode a render as if it were the source.
- Never overwrite a web that has been committed or pinned in `web.json`.

## Algorithm

```text { type: notebook.cell }
python ref/web.py encode <source.png> <out.web> [--hidden 64 --omega0 30 --steps 10000 --lr 1e-3 --seed 0]

encode(media, H, omega0, steps, lr, seed):
  X := grid coordinates (contract §3); T := media / 127.5 - 1
  W1, b1 ~ U(-1/r, 1/r)                   # SIREN first layer
  W2, W3 ~ U(-sqrt(6/H), sqrt(6/H)); b2, b3 := 0
  repeat steps times, full batch:
    Y := forward(X); loss := mean((Y - T)^2)
    one Adam step (beta 0.9, 0.999, eps 1e-8) at rate lr on all six tensors
  return pack(...)                         # round to binary16, drop -0, refuse overflow
```

- Training runs in binary32 for speed. The result is rounded to binary16 once, at `pack`.
- The encoder is not normative. Another optimiser, more steps or a quantisation-aware fit are all valid, as long as the output is a canonical web and its fidelity is measured by `check`.
- Runs are not promised to be bit-reproducible across machines or numpy builds. The committed bytes, not the run, are the record.

## Sample

| Setting | Value |
|---|---|
| source | `samples/astronaut-64.png`, 64 × 64 RGB |
| network | `H = 64`, `omega0 = 30`, 4,547 weights |
| fit | 10,000 full-batch Adam steps at `lr = 1e-3`, `seed = 0`, about 70 s on an 8-core CPU |
| result | `samples/astronaut-64.web`, 9,110 bytes, PSNR 31.18 dB |

Shorter fits, same settings otherwise: 3,000 steps gave 25.62 dB at `lr = 1e-3` and 20.56 dB at `lr = 1e-4`.

## Falsification

- The written file fails `unpack`, or `check` against its source prints a PSNR other than the one recorded.

## Licence

MIT, see `LICENSE`.
