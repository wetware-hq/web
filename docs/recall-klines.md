# `recall-klines` — what has worked more than once

`type: resource.memory` · `spec: agent.memory` · non-normative; `docs/follow-agent-contract.md` governs on any conflict.

A **K-line** is Minsky's knowledge-line: a wire that, when activated, switches back on the set of agents that were active the last time a problem of this kind was solved ([*K-Lines: A Theory of Memory*](https://dspace.mit.edu/handle/1721.1/5763), 1980; *The Society of Mind* §8). It is not a fact and not a rule. It is a handle on a working configuration.

This file is the stack's K-line register. A line is listed only when the same move was reached independently in **more than one chain of thought** — separate sessions, usually separate repositories, that did not share a plan — because a move that worked once is a result and a move that worked twice is a method. Each line names what to reactivate, so an agent facing the trigger loads that set first instead of re-deriving it.

## Specification

```json { type: notebook.cell }
{
  "id": "docs/recall-klines.md",
  "type": "resource.memory",
  "role": "reactivate the configuration that solved this kind of problem before",
  "reads": ["docs/organize-stack.md", "docs/follow-agent-contract.md", "the stack's merged work"],
  "writes": [],
  "must": ["one line per replicated move", "at least two independent chains per line", "cite the artifact, never the intention", "a line is a handle, never an authority"],
  "check": ["K1-K9"]
}
```

## Graph

- ← [read-repository.md](read-repository.md)
- ← [organize-stack.md](organize-stack.md)
- → [follow-agent-contract.md](follow-agent-contract.md)
- → [pin-sources.md](pin-sources.md)

## Register

| K | Line | Reactivate | Replicated in |
|---|---|---|---|
| K1 | Refuse, never guess | contract §5 codes, `ref/dna.py unpool`, tesseract §8 refusals, bases `Refusal` | web #3, tesseract #1, bases #4 |
| K2 | The verdict is computed, never authored | contract §12 D16, popper `skills/refute-experiment.md`, tesseract `reason.md` 5 | web #3, popper #1, tesseract #3 |
| K3 | Exact both ways, or declared a view | contract §9–§10, §14, tesseract `compose` | web #2, web #6, tesseract #1 |
| K4 | One root rule per object, no crossed names | §13 S2–S3, `dna.md`, `tensor.md` 4 | web #5, helix #5, tesseract #1 |
| K5 | Replay is the rule; frames chain to a head | contract §12, bases `receipt`, popper traces, `reason.md` 2 | web #3, bases #3, popper #1, tesseract #3 |
| K6 | The machine is data | `tesseract.json` `machines`, `research.json` testbed and charter | tesseract #1, popper #1 |
| K7 | Pin a neighbour, never vendor it | `docs/pin-sources.md` in each repository | tesseract #1, popper #1, #7, tulip #1 |
| K8 | Read the chain at a finalized block | helix `docs/pin-references.md`, bases `solana/chain.ts` | helix #4, bases #4 |
| K9 | A specbase has a fixed file shape | this repository's own shape | web #1, helix #1, tesseract #1, popper #1, tulip #1 |

### K1 — Refuse, never guess

**Trigger.** A conversion is handed data it cannot represent, or a check fails halfway through.

**Line.** Return the object the data determines, or a stable code from a closed list. Never repair, never approximate silently, and never resume a refused object: a refused file stays refused.

**Chains.** `unpool` votes or refuses `tie`, `gap`, `homopolymer` (web #3, contract §10); `compile` refuses with a code from a closed list and never changes a base (tesseract #1, §8); coin issuance refuses `unprinted`, `loop_mismatch` rather than issuing on partial provenance (bases #4, `solana/coin.ts`).

**Falsification.** A conversion in the stack returns a best-effort result where the data is ambiguous.

### K2 — The verdict is computed, never authored

**Trigger.** A process that reasons is also asked whether it was right.

**Line.** Separate the claim from the judgement. The agent writes the prediction before the act; the verdict is a function of the replayed outcome and the prediction, and the reasoner's own file never carries a verdict about the world.

**Chains.** `loop1` sets `verdict = supports` exactly when the replayed outcome equals `expect` (web #3, D16); the Critic applies a registered rule and `self_verdict` names the failure of grading one's own run (popper #1); a `.doll` derivation is `"unverified"`, always (tesseract #3, `reason.md` 5).

**Falsification.** A verdict field anywhere in the stack that a generating agent may write.

### K3 — Exact both ways, or declared a view

**Trigger.** One object has to be read as another.

**Line.** Either the map inverts to the same bytes and a test pins the round trip, or it is labelled a view and is never read back as data. There is no third class.

**Chains.** `web ↔ dna1` and `sequence ↔ pool1` invert, while a `loop1` file is a view (web #3); `compose(plan)` must return the target byte for byte or refuse (tesseract #1); a `tmap1` map is exact for a web, a sequence and a pool, a view for a loop (web #6, `tensor.md` 5).

**Falsification.** A view is decoded back into data, or a claimed inverse is not covered by a round-trip test.

### K4 — One root rule per object, no crossed names

**Trigger.** Two layers hold a hash of the same thing.

**Line.** Fix one hash rule per object and keep the names apart. A sequence root is `sha256(ascii(seq))`; a chain root is `sha256(salt || ascii(seq))`. They meet only through a token id, and a value named as one root is never accepted where another is required — the kind travels in the bytes.

**Chains.** Two sibling sessions reached the same correction independently on the same day: web #5 split the chain root from the sequence root across the stack map, and helix #5 and tesseract #1 fixed the same confusion from the other side, where a `plan` claim binds by page subject because a page holds the salted root. Then `tmap1` added `kind_mismatch`, so a tensor root cannot stand in for any of them (web #6).

**Falsification.** Two files hash one sequence two different ways, or one root is accepted where another is required.

### K5 — Replay is the rule; frames chain to a head

**Trigger.** A record of reasoning or of work has to be trusted by someone who was not there.

**Line.** Write append-only frames, hash each into the next, and make replay — not the writer's account — the thing that is normative. The head is then a 32-byte name that another layer can cite, and derived randomness comes from the chain, so a record cannot be edited in place.

**Chains.** `loop1` frames hash-chain to a head any agent can replay (web #3); a print `receipt` is defined as that head, so a chain record points at a replayable read-back (bases #3, helix #4 folds the print chain to `headOf`); popper derives run seeds from trace roots, so editing one event changes every later seed (popper #1); a `.doll` is valid only if its frames regenerate (tesseract #3).

**Falsification.** A record whose claims cannot be recomputed from its own inputs, or a head that does not change when a frame does.

### K6 — The machine is data

**Trigger.** A physical or deployment limit shows up in a design.

**Line.** Put every limit in the repository's JSON as one knob per limit, and none that is not a limit. The code reads them. A machine that does not exist yet can then be priced today, and the same target compiles shallower as the limit moves.

**Chains.** `tesseract.json` `machines` carries floor, ceiling, pool width, GC and homopolymer limits and enzymes, and the tree collapses to one part at a 50 kb ceiling (tesseract #1, §3); `research.json` carries the testbed, data split, decision rule and charter defaults, so a night is set by data, not by code (popper #1).

**Falsification.** A limit hard-coded in a reference implementation, or a knob that stands for nothing physical.

### K7 — Pin a neighbour, never vendor it

**Trigger.** Work depends on an outside repository, a paper, or a sibling in the stack.

**Line.** Cite it by a stable identifier — a commit plus a file `sha256`, or a DOI — and copy nothing. Licence incompatibility then never propagates, and a neighbour's change is visible as a pin that no longer matches.

**Chains.** `RomeroLab/omega` is GPL-3 and pinned by commit and file hashes, with `omega1` written deliberately simpler (tesseract #1); karpathy/autoresearch is pinned by commit (popper #1) and Cello's primitives are restated and pinned, never copied (popper #7); digital-oulipo is pinned by commit and `sha256` with the texts used structure-only (tulip #1).

**Falsification.** A file in the stack that is a copy of an unpinned or incompatibly licensed source.

### K8 — Read the chain at a finalized block

**Trigger.** An off-chain object cites an on-chain fact.

**Line.** Read at the finalized block by default and record the block with the pin. A value a reorganization can remove is `unavailable`, not a fact.

**Chains.** A chain-row pin must be at or below the finalized block, so a sealed value cannot be withdrawn (helix #4, `docs/pin-references.md`); the coin scanner reads `finalized` by default and refuses a chromosome whose contract does not yet exist at that block (bases #4, `solana/chain.ts`).

**Falsification.** A pinned chain value at a block above the finalized head.

### K9 — A specbase has a fixed file shape

**Trigger.** A new object or a new layer has to be specified.

**Line.** Ship a `DISCLAIMER` read first; a router, `docs/read-repository.md`, that sends each task down the smallest sufficient path; one contract under `docs/` that governs on any conflict; a pins page under `docs/` and one root JSON of pinned sources and samples with their digests; `skills/` and `systems/` that hang off the contract; and one `ooc/<repo>.og` index whose `checksum` covers the file with its `checksum:` line removed. Every file carries a Graph section that each file it links to reciprocates. A reference implementation, where there is one, sits under `ref/` and is subordinate to the contract. Prose is never the authority.

**Chains.** web (#1), helix (#1), tesseract (#1), popper (#1) and tulip (#1) were each built to this shape by different chains, and later work in each repository filled the same slots rather than adding new ones. Only web and tesseract have a `ref/`; helix, popper and tulip are specification only.

**Falsification.** A specification whose rules live only in prose, a Graph link that is not reciprocated, or an `.og` whose checksum does not cover its body.

## Use

```text { type: notebook.cell }
recall(task):
  read DISCLAIMER, then docs/read-repository.md
  for each K in the register:
      if K.trigger matches task:  load K's reactivate set before writing anything
  if a line's falsification holds in what you are about to write:  change the work, not the line
```

A line is a handle, not an authority: `docs/follow-agent-contract.md` and each repository's own contract still govern. A line stops being listed the moment its falsification is found, or it is replicated in only one chain after all.

## Licence

MIT, see `LICENSE`. Neighbouring repositories and cited work are pinned, not vendored.
