# `.dna`

A DNA sequence is the primitive datum of the wetware stack: one ASCII string over `ACGT`, and nothing else. A `.dna` file is one such sequence written as a single FASTA record, and `sha256` of its bases is its root. Every repository that speaks about DNA — webs, oligo pools, loops, build plans, chromosome tokens and wiki pages — speaks about this one object. This page names the file form; [`docs/follow-agent-contract.md`](docs/follow-agent-contract.md) §9 and §13 define it and govern on any conflict.

```toml
[dna]
media_type = "text/vnd.dna"            # vendor tree, not registered with IANA
extension  = ".dna"                    # ".fa" is the same bytes under its FASTA name
alphabet   = "ACGT"                    # uppercase, no N, no gaps, no U, no lowercase
version    = "1.0.0"
spec       = "docs/follow-agent-contract.md"
reference  = "ref/dna.py"

[pipeline]
input   = "bases --one FASTA record { one '>' line, then the sequence }"
process = "constrain { dna1 cipher §9; pool1 oligos §10 } | view { loop1 §12 }"
output  = "sequence --fileformat .dna { root = sha256(ascii(seq)); chain root = sha256(salt || ascii(seq)) }"
```

## Axioms

1. **One alphabet.** A sequence is non-empty and holds only `A`, `C`, `G` and `T`. A file holds exactly one record. Anything else is refused with `bad_base` (contract §5); no layer repairs, pads, or lower-cases a sequence to make it fit.
2. **Root is identity.** `root = sha256(ascii(seq))` in lowercase hex. A chromosome commits its **chain root** `sha256(salt || ascii(seq))` with a random 32-byte salt instead, so the chain does not leak a short sequence (contract §9); Bases calls it `root` and helix `sequence-root`. The two roots are never compared: only a salt holder can link them. A sequence root never stands in for a web root, and a web root never stands in for a sequence root (contract §13, S3).
3. **Bases live in sequence files only.** A token, a wiki page, a view, a diff and a loop hold roots, counts and lengths — never bases. Only a `.dna` file and an oligo pool file hold the sequence itself (contract §13, S4).
4. **Every constraint is a subset, not a dialect.** A `dna1` sequence (no repeated base, contract §9) and a `pool1` oligo (117 nt, indexed, contract §10) are sequences first. A reader that accepts this format accepts them; the extra rules only narrow what is valid.
5. **Conversions are two-way or they are views.** `web ↔ dna1` and `sequence ↔ pool1` are exact in both directions. `loop1` is a view: it is regenerated from the sequence and never read back into one (contract §13, S5).
6. **Refusals are codes.** A conversion that is not determined refuses with a stable code (`bad_base`, `bad_length`, `homopolymer`, `gap`, `tie`, …) and returns nothing, rather than guessing a base (contract §5).
7. **Fixed once committed.** A sequence that has been committed on chain is never rewritten; a changed sequence is a new sequence with a new root ([`docs/organize-stack.md`](docs/organize-stack.md), `wetware-hq/bases`).
