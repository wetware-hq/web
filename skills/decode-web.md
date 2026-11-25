# `decode-web` — render and check a web

`type: resource.skill` · `spec: agent.skill` · subordinate to `docs/follow-agent-contract.md`, which governs on any conflict.

Parse a web, render it to media, and check its root and fidelity.

## Specification

```json { type: notebook.cell }
{
  "id": "skills/decode-web.md",
  "type": "resource.skill",
  "role": "turn web bytes into a render and verify the claims made about them",
  "reads": ["*.web", "web.json", "source PNG (for check)"],
  "writes": ["render PNG (optional, never stored back)"],
  "must": ["refuse before rendering", "compare by PSNR, not pixels"],
  "check": ["W1", "W2", "W3", "W4"]
}
```

## Graph

- ← [encode-media.md](encode-media.md)
- ← [../docs/follow-agent-contract.md](../docs/follow-agent-contract.md)
- → [../systems/commit-web.md](../systems/commit-web.md)

## Indications

- An agent holds a web (from disk, a store or a chain) and needs the media, or needs to know the web is what it claims to be.

## Contraindications

- Never treat a render's hash as the web's root.
- Never repair a refused file. A refused web stays refused.

## Algorithm

```text { type: notebook.cell }
python ref/web.py root   <in.web>
python ref/web.py decode <in.web> <render.png>
python ref/web.py check  <in.web> <source.png> [--min-psnr X] [--root R]

check(web, source, min, R):
  if R and sha256(web) != R:      fail(root_mismatch)
  w := unpack(web)                 # first failing code of contract §5
  out := decode(web); ref := read(source)
  if out.shape != ref.shape:       fail(shape_mismatch)
  q := psnr(out, ref)
  if q < min:                      fail(below_fidelity)
  print root, bytes, q
```

## Check

For every entry in `web.json` `webs`, run `check` with `--root` set to its `root` and confirm the printed PSNR is within 0.01 dB of its `psnr` (W4). `python -m unittest discover -s tests` runs this and the format predicates.

## Falsification

- `check` passes a file whose sha256 differs from `--root`.
- `decode` writes a render for a file that `unpack` refuses.

## Licence

MIT, see `LICENSE`.
