# `commit-web` — put a web on chain and read it back

`type: resource.system` · `spec: agent.system` · subordinate to `docs/follow-agent-contract.md`, which governs on any conflict.

A web is small enough to live on chain in full, and its root is a 32-byte commitment. This file fixes what is written and how a reader trusts it. No contract is deployed yet: `web.json` `chain` is `null`, so nothing can be committed.

## Specification

```json { type: notebook.cell }
{
  "id": "systems/commit-web.md",
  "type": "resource.system",
  "role": "commit a web root on chain and read the web back by root",
  "reads": ["*.web", "web.json"],
  "writes": ["chain (once a contract is pinned)"],
  "must": ["commit the root, not a render", "check the root on every read", "never rewrite a committed root", "commit a deep net by its head, its stages by their roots"],
  "check": ["W3", "W6", "C4"]
}
```

## Graph

- ← [../skills/decode-web.md](../skills/decode-web.md)

## Indications

- A generated image should be held as an on-chain asset whose content anyone can re-render.

## Contraindications

- Never commit before `check` passes against the pinned source.
- Never sign a chain transaction while `web.json` `chain` is `null`.

## Algorithm

```text { type: notebook.cell }
commit(web):
  R := sha256(web)
  write R as bytes32, and the web bytes in the same transaction's calldata
  (or in a content-addressed store keyed by R, with only R on chain)
  emit (R, len(web), H, grid, d) so readers can size and route without parsing

read(R):
  b := calldata or store bytes for R
  if sha256(b) != R: fail(root_mismatch)
  return decode(b)
```

- Mint should follow the commit-reveal pattern of `wetware-hq/bases`, so that a root seen in the mempool cannot be front-run: commit `sha256(R || secret)`, wait, then reveal `R` and the bytes.
- The root is fixed once committed (contract §1, W6). Transfers change who holds a web, never its bytes.

## Composition

A deep net (contract §16) is committed as its 32-byte head. Each stage web is committed by its own root, as above, and the head binds those roots to the interface roots in order. A reader fetches every stage by root, checks each one, and replays the record: `ref/deep.py verify` refuses `root_mismatch` for a wrong stage and `mismatch` for an interface that does not replay. A proof of each stage would let a reader check an interface without the weights (ZKML, `docs/pin-sources.md` §Composition). No proof system is pinned, so replay is the only check.

## Cost

Calldata costs 16 gas per non-zero byte and 4 per zero byte (EIP-2028). The sample web has 9,092 non-zero and 18 zero bytes, so its calldata costs 145,544 gas on top of the 21,000-gas transaction base. Storing the bytes in contract storage instead would cost far more; storage should hold only `R`.

## Falsification

- A reader returns a render for bytes whose sha256 is not the requested root.
- The same root is committed twice with different bytes.

## Licence

MIT, see `LICENSE`.
