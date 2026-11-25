# `follow-agent-contract` — the web contract

`type: resource.contract` · `spec: agent.contract` · normative. Every other file is subordinate to this one.

A **web** is one piece of static media stored as the weights of a shallow neural network: two hidden layers of sine units that map a position (a pixel's row and column) to the value at that position. By the universal approximation theorem such a network can approximate any continuous signal on a bounded grid as closely as wanted, given enough units, so one fixed architecture can hold any image. The weights are the stored object; the media is regenerated from them on demand.

## Specification

```json { type: notebook.cell }
{
  "id": "docs/follow-agent-contract.md",
  "type": "resource.contract",
  "role": "define the web object, its bytes, the decoder, its root, refusals and predicates",
  "reads": ["DISCLAIMER", "web.json", "ooc/web.og"],
  "writes": [],
  "must": ["one canonical byte form", "decode is the only normative direction", "never rewrite a committed web", "match codes, not prose"],
  "check": ["W1-W8", "D1-D13"]
}
```

## Graph

- ← [read-repository.md](read-repository.md)
- → [pin-sources.md](pin-sources.md)
- → [vary-cybernetics.md](vary-cybernetics.md)
- → [../skills/encode-media.md](../skills/encode-media.md)
- → [../skills/decode-web.md](../skills/decode-web.md)
- → [../systems/commit-web.md](../systems/commit-web.md)

## Composition

```toml { type: notebook.cell }
format = "web.md"              # the .web file format: extension, media type, magic
store = "*.web"                # canonical bytes, §2
pins = "web.json"              # sample roots, fidelity and chain
index = "ooc/web.og"           # ObjectGraph traversal index
ref = "ref/web.py"             # reference decoder (normative) and encoder (one valid fit)
dna = "ref/dna.py"             # dna1 cipher §9, pool1 oligos §10, cgr1 glyph §11; both directions normative
commit = "systems/commit-web.md"
```

## Indications

- Media produced by a generative model needs a compact, fixed-shape, hashable form that can be committed on chain.
- A deep network needs to take media as input in weight space, without a pixel decoder in between (`docs/pin-sources.md` §Weight space).
- An agent needs to check that a stored web still reproduces its source to a stated fidelity.

## Contraindications

- Not lossless. A web approximates its source; only its own bytes are exact.
- Not a proof of authorship, provenance or licence. A root proves only which bytes were committed.
- Not video, audio or 3D yet. The format reserves ranks 1 and 3 (§2), but this prototype pins and checks only images (rank 2).
- Not a compressor that beats PNG at small sizes. The 64 × 64 sample web is 9,110 bytes; its PNG is 11,034 bytes and its raw pixels are 12,288 bytes.

## §0 Objects

| Object | What it is | Exact? |
|---|---|---|
| **media** | a uint8 array of shape `grid + (channels,)`; for an image `grid = (rows, cols)` and `channels` is 1 (gray) or 3 (RGB) | yes |
| **web** | the canonical bytes of one network (§2) | yes |
| **root** | `sha256(web)` as 64 lowercase hex digits | yes |
| **render** | `decode(web)` (§3), a media array | yes, up to §3 note |
| **fidelity** | PSNR in dB of a render against its pinned source media | a measurement, not a property of the bytes |

## §1 Flow

```text { type: notebook.cell }
media --encode--> web --decode--> render
                  |
                  +--sha256--> root --commit--> chain
```

- `encode` is many-to-one in reverse: many webs fit one media, and nothing requires two encoders, or two runs of one encoder, to produce the same bytes.
- `decode` is a pure function of the bytes. It is the only normative direction.
- A render is never stored back as a web or as a new source. Re-encoding a render makes a new web with a new root.
- A committed web is never rewritten. A better fit is a new web.

## §2 Bytes

All integers are unsigned little-endian. `r` is the grid rank, `H` the hidden width, `d` the channel count.

| Offset | Size | Field | Allowed |
|---|---|---|---|
| 0 | 4 | magic | ASCII `web1` |
| 4 | 1 | activation | `1` = sine |
| 5 | 2 | `omega0` | 1–255 |
| 7 | 1 | `r` | 1, 2 or 3 (only 2 is pinned) |
| 8 | 2·r | grid sizes, axis order | each ≥ 2 |
| 8+2r | 1 | `d` | 1 or 3 |
| 9+2r | 2 | `H` | 1–256 |
| 11+2r | 1 | dtype | `1` = IEEE 754 binary16 |
| 12+2r | 2·P | weights | finite, no negative zero |

Weights follow in this order, each row-major: `W1 (r×H)`, `b1 (H)`, `W2 (H×H)`, `b2 (H)`, `W3 (H×d)`, `b3 (d)`. So `P = rH + H + H² + H + Hd + d` and the file is exactly `12 + 2r + 2P` bytes, at most 65,536. For the pinned sample (`r = 2`, `H = 64`, `d = 3`) `P = 4,547` and the file is 9,110 bytes.

There is one byte form per network: a parser that accepts `b` must re-serialize it to `b`. Negative zero, NaN, infinity, a short file and any trailing byte are refused (§5).

## §3 Decode

```text { type: notebook.cell }
decode(web):
  parse web (§2); widen every weight to binary64
  for each axis of size n: a_i = -1 + 2i/(n-1),  i = 0..n-1
  X  := every grid point in row-major order (first axis slowest), shape (N, r)
  h1 := sin(omega0 * (X W1 + b1))
  h2 := sin(h1 W2 + b2)
  y  := h2 W3 + b3
  level := clip(round_half_even((y + 1) * 127.5), 0, 255) as uint8
  return level reshaped to grid + (d,)
```

- Arithmetic is binary64. `sin` may differ by an ulp between maths libraries, so two decoders can disagree on a level only where `(y + 1)·127.5` lies within rounding error of a half. Renders are compared by fidelity (§6), never by pixel equality.
- `omega0` scales only the first layer. The second layer's frequency is carried in `W2`.

## §4 Root

`root = sha256(web)`. The root names the bytes, so it is the only identifier: a web is cited as `web1:<root>`. The root of a render, or of the source media, is not a web root and never stands in for one. On-chain use is in `systems/commit-web.md`.

## §5 Refusal codes

Agents match on codes, never on message text.

| Code | When |
|---|---|
| `bad_magic` | bytes 0–3 are not `web1` |
| `bad_header` | activation, `omega0`, rank, a grid size, `d`, `H` or dtype is outside §2 |
| `truncated` | the file ends before the last weight |
| `trailing_bytes` | bytes remain after the last weight |
| `bad_weight` | a weight is NaN, infinite or negative zero, or the encoder produced a weight that overflows binary16 |
| `too_large` | the file is over 65,536 bytes |
| `bad_media` | the source is not an intact 8-bit non-interlaced gray, RGB or RGBA PNG, or a shrink factor is not a positive divisor of its size |
| `exists` | the encoder's output path already holds a file; a new fit is written to a new path |
| `shape_mismatch` | a render's shape differs from the media it is checked against |
| `below_fidelity` | a render's PSNR is under the threshold asked for |
| `root_mismatch` | the bytes read back do not hash to the root they were fetched by (`systems/commit-web.md`) |
| `bad_length` | a dna1 sequence is empty or its length is not a multiple of 11 (§9) |
| `bad_base` | a dna1 sequence holds a letter other than `A`, `C`, `G`, `T`, or its FASTA file is not exactly one record (§9) |
| `homopolymer` | a dna1 sequence repeats a base, or starts with `A` (§9) |
| `bad_word` | an 11-nt dna1 word encodes a value over 65,535 (§9) |
| `bad_salt` | a Bases salt is not hex or is not exactly 32 bytes (§9) |
| `bad_file` | a pool1 file id is not 0–8 (§10) |
| `gap` | no valid pool1 read covers some base (§10) |
| `tie` | the valid pool1 reads covering a base disagree with no majority (§10) |
| `bad_pool` | the voted pool1 sequence's length trailer or padding is inconsistent (§10) |

## §6 Fidelity

```text { type: notebook.cell }
psnr(a, b) = 10 log10(255^2 / mean((a - b)^2))    # binary64, over every level; inf if equal
```

A fidelity claim names the web root, the source media's sha256, and a PSNR to two decimals. A checker that re-decodes must reproduce that PSNR to within 0.01 dB, or the claim fails.

## §7 Deep-net form

A web is already a tensor: for an `r = 2`, `H = 64`, `d = 3` image it is 4,547 numbers in one fixed order, so a deep network can take it as input directly. Two caveats hold for any reader:

- Hidden units can be permuted without changing the render, so two webs of one image may differ only by a permutation. A reader must be permutation-invariant, or every web it reads must be fitted from the same initial weights (the reference encoder's `seed`).
- Webs of different `H`, `r` or `d` have different shapes and are not comparable tensors.

## §8 Predicates

| Id | Predicate |
|---|---|
| W1 | `pack(unpack(b)) == b` for every accepted `b` |
| W2 | `decode` depends only on the bytes |
| W3 | `root == sha256(bytes)` |
| W4 | every web in `web.json` decodes and reproduces its pinned PSNR within 0.01 dB |
| W5 | every sample source in `web.json` hashes to its pinned sha256 |
| W6 | no web is ever replaced under an existing root or name; a new fit gets a new entry |
| W7 | a web stores no pixel of its source; it holds only the §2 header and weights |
| W8 | `ooc/web.og` checksum equals sha256 of the file with its `checksum:` line removed |

## §9 DNA cipher `dna1`

`dna1` is a lossless, two-way map between a web and one synthetic-DNA sequence over `ACGT`. Both directions are normative: `ref/dna.py` is the reference.

```text { type: notebook.cell }
encipher(web):
  unpack(web)                                     # only accepted webs, §2
  w = web XOR keystream(len(web))                 # keystream = sha256("dna1" || u64be(i)) blocks, i = 0, 1, ...
  for each little-endian uint16 v of w:           # header and weights alike, in file order
    write v as 11 base-3 digits, most significant first
    for each digit t: prev = (prev + 1 + t) mod 4; emit "ACGT"[prev]    # prev starts at A
decipher(seq) = the inverse, then unpack; any other input is refused (§5)
```

- **Shape.** A web of `n` bytes is `11 n / 2` nt. The 9,110-byte sample is 50,105 nt. This is a storage form, not compression: it is 1.375 bits per base, against a maximum of 2.
- **Constraints.** No base is ever repeated (longest run 1), after Goldman et al. 2013. The keystream makes the sequence look random, so GC content is near 50% (sample: 0.5009) and long repeats are unlikely. The keystream is public and fixed. It is not encryption and hides nothing.
- **Root.** The sequence root is `sha256(ascii(seq))`. With a 32-byte salt, `sha256(salt || ascii(seq))` and the length in nt are what a `Bases` chromosome token commits (`wetware-hq/bases`). The sequence root never stands in for the web root (§4).
- **CGR.** `fcgr(seq, k)` draws the frequency chaos-game image of a sequence (Jeffrey 1990): the count of each k-mer on a `2^k × 2^k` grid, scaled so the most common is 255 and every present k-mer is at least 1. A dna1 sequence holds no k-mer with a repeated base, so its image is zero everywhere else, and the excluded k-mers form a visible fractal. The image is itself media, so it can be fitted as a web, which closes the loop web → DNA → image → web.
- **Autoencoder reading.** Fitting is the encoder, the web weights are the latent code, dna1 is the channel code, and decipher plus §3 decode is the decoder.
- **One strand.** A dna1 sequence has no index and no error correction. A substitution changes about one 16-bit word, or is refused if it makes a repeat. An insertion or deletion shifts every later word and is almost always refused. §10 splits it into oligos that can be synthesized and survive loss and substitutions.

| Id | Predicate |
|---|---|
| D1 | `encipher` depends only on the web bytes; every sequence in `web.json` `dna` equals `encipher` of its web |
| D2 | `decipher(encipher(b)) == b` for every accepted web `b` |
| D3 | no sequence from `encipher` repeats a base |
| D4 | `decipher` refuses every sequence that `encipher` could not have written |
| D5 | the non-zero cells of `fcgr(seq, k)` for a pinned sequence are exactly the k-mers with no repeated base |
| D6 | `sequence == sha256(ascii(seq))` for every pinned sequence |

## §10 Oligo pool `pool1`

`pool1` splits a dna1 sequence into 117-nt oligos and rebuilds it from reads in any order or orientation, after Goldman et al. 2013. Both directions are normative: `ref/dna.py` `pool`/`unpool` is the reference.

```text { type: notebook.cell }
pool(seq, id):                                      # id in 0..8
  t  = unrotate(seq)                                # §9 rotating code, back to trits (prev starts at A)
  s4 = t || 0* || base3(len(seq), 20 trits)         # zeros so len(s4) is a multiple of 25 and >= 100
  s5 = rotate(s4)                                   # s5 starts with seq itself
  for i in 0 .. (len(s5) - 100) / 25:
    F  = s5[25 i : 25 i + 100]; if i odd: F = revcomp(F)
    IX = base3(id, 2) || base3(i, 12) || P          # P = sum of IX trits 1, 3, 5, ..., 13 (1-based) mod 3
    O  = F || rotate(IX, prev = last base of F)     # 115 nt
    emit (T if O starts with A else A) || O || (G if O ends with C else C)    # 117 nt
unpool(reads, id):
  flip any read that starts with C|G and ends with A|T (the other strand)
  drop any read that is not 117 nt of ACGT, has bad flanks, a repeat in IX or a wrong P
  keep reads of this id; undo the odd-index revcomp; place F at 25 i
  vote per base; refuse gap or tie; rotate back; check the trailer; return seq
```

- **Shape.** Each base of `s5` is in 4 oligos, fewer within 75 bases of either end and only 1 in the first and last 25. The 50,105-nt sample becomes 2,002 oligos (234,234 nt, 4.67× the strand). Oligo GC is 0.3846–0.6068 and no oligo repeats a base. GC per oligo is reported, not enforced.
- **Survives.** Any read order, either strand, foreign reads and reads from other file ids, the loss of interior oligos while every base stays covered, and substitutions outvoted by the other copies.
- **Refuses, never guesses.** A base covered by no valid read is `gap`. A base whose valid reads disagree with no majority is `tie`. A sequence that fails §9 afterwards is still refused by `decipher`. The first and last 25 bases are in one oligo each, so losing oligo 0 is `gap` and losing the last oligo is `bad_pool`.
- **Parity is weak, as in the source.** `P` sees only 7 of the 14 index trits, so some index errors pass. A misplaced payload is usually outvoted; an index past the end makes the pool refuse.

| Id | Predicate |
|---|---|
| D7 | every pinned pool in `web.json` equals `pool` of its sequence and hashes to its pinned sha256 |
| D8 | every oligo is 117 nt, starts with `A` or `T`, ends with `C` or `G`, and repeats no base |
| D9 | `unpool(pool(seq)) == seq` under any read order, either strand per read, and the loss of any oligos that leaves every base covered |
| D10 | a substitution in one read is outvoted wherever another two valid reads cover the base |
| D11 | `unpool` refuses (`gap`, `tie`, `bad_pool`) rather than return a sequence the reads do not determine |

## §11 Text glyph `cgr1`

`cgr1` prints the k-mer counts of a sequence (§9 CGR, `1 ≤ k ≤ 6`) as plain text an agent reads directly, without an image decoder. It borrows four features from QR codes (ISO/IEC 18004): a format line, finder corners, a coordinate system and a check value. It is a view of a sequence, not a store: the store is dna1 and pool1.

```text { type: notebook.cell }
%cgr1 k=<k> nt=<len> peak=<max count> present=<cells > 0>/<4^k> sequence=<sha256(ascii(seq))>     # format line
%ramp " .:-=+*#%@"                                  # space = absent, "." = least present, "@" = peak
<k column-label lines, offset k + 2>                # line j, column x: K if bit j of x else M  (M = A|C, K = G|T)
<k + 1 spaces>C<2^k dashes>G                        # finder corners: the base whose k-mer fills that corner
<row label> |<2^k ramp characters>|                 # row r: S if bit j of (2^k - 1 - r) else W  (S = C|G, W = A|T)
<k + 1 spaces>A<2^k dashes>T
%counts sha256:<sha256 of the counts, row 0 first, decimal ASCII joined by single spaces>
```

- **Cell address.** Letter `j` of a cell's k-mer, oldest base first, is where its column and row labels meet: `M∩S = C`, `M∩W = A`, `K∩S = G`, `K∩W = T` (IUPAC codes). An agent cites a cell by its k-mer and never needs coordinates.
- **Biology on the axes.** Row labels are strong/weak bonding, so a row's `S` count is that k-mer's GC count. Column labels are amino/keto.
- **Level.** A cell's character is `ramp[0]` if its count is 0, else `ramp[1 + (g - 1) * 8 // 254]`, where `g` is the §9 gray level.
- **Check.** The footer hash covers the exact counts, not the ramp, so two readers agree on what was read only if they also agree on `sha256`.

| Id | Predicate |
|---|---|
| D12 | every pinned glyph in `web.json` equals `glyph` of its sequence |
| D13 | for every cell, the k-mer spelled by its labels is present in the sequence exactly when the cell is not a space |

## Citations

- Universal approximation: Cybenko 1989, Hornik 1991 (`docs/pin-sources.md` §Theory).
- Sine networks for signals: SIREN, arXiv:2006.09661.
- Media as network weights: COIN, arXiv:2103.03123.
- Networks reading weights: functa, arXiv:2201.12204; DWSNets, arXiv:2301.12780.
- DNA storage, rotating code and CGR: Goldman et al. 2013; Jeffrey 1990; iCGR, arXiv:1712.04546; JPEG-DNA, arXiv:2103.09616 (`docs/pin-sources.md` §DNA).

## Falsification

- A file accepted by `unpack` re-serializes to different bytes.
- Two decodes of one web on one machine differ.
- `python ref/web.py check samples/astronaut-64.web samples/astronaut-64.png` prints a PSNR more than 0.01 dB from `web.json`.
- `python ref/dna.py decipher samples/astronaut-64.fa out.web` writes bytes other than `samples/astronaut-64.web`.

## Licence

MIT, see `LICENSE`.
