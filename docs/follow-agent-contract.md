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
  "check": ["W1-W8", "D1-D4", "D6-D11", "D14-D16", "S1-S7", "T1-T4", "U1-U5", "C1-C5", "L1-L5"]
}
```

## Graph

- ← [read-repository.md](read-repository.md)
- ← [recall-klines.md](recall-klines.md)
- → [pin-sources.md](pin-sources.md)
- → [organize-stack.md](organize-stack.md)
- → [vary-cybernetics.md](vary-cybernetics.md)
- → [../skills/encode-media.md](../skills/encode-media.md)
- → [../skills/decode-web.md](../skills/decode-web.md)
- → [../systems/commit-web.md](../systems/commit-web.md)

## Composition

```toml { type: notebook.cell }
format = "web.md"              # the .web file format: extension, media type, magic
primitive = "dna.md"           # the .dna sequence form: the stack primitive, §13
store = "*.web"                # canonical bytes, §2
pins = "web.json"              # sample roots, fidelity and chain
index = "ooc/web.og"           # ObjectGraph traversal index
ref = "ref/web.py"             # reference decoder (normative) and encoder (one valid fit)
dna = "ref/dna.py"             # dna1 cipher §9, pool1 oligos §10; both directions normative
loop = "ref/loop.py"           # loop1 pseudovisual chain of thought §12; replay is normative
commit = "systems/commit-web.md"
stack = "docs/organize-stack.md"  # where each repository sits over the primitive, §13
tensor = "tensor.md"           # the .tmap tensor map: every layer's object as named tensors, §14
tensors = "ref/tensor.py"      # tmap1 maps §14; TensorFlow DataType codes; exact maps invert
tune = "ref/tune.py"            # weight manipulation §15: training steps, state and adapter maps, LR rules
deep = "ref/deep.py"            # deep1 composition §16: shallow webs into a deep net, interfaces hash-chained
compose = "ref/compose.py"      # kline1 §17: composition functions recalled as K-lines, deep nets grown by induction
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
| **sequence** | one non-empty ASCII string over `ACGT`, the stack primitive (`dna.md`, §9, §13) | yes |
| **sequence root** | `sha256(ascii(seq))` | yes |
| **chain root** | `sha256(salt \|\| ascii(seq))` with a 32-byte salt: Bases `root`, helix `sequence-root` | yes, with the salt |
| **deep net** | an ordered list of webs, each stage reading the levels of the one before (§16) | yes |
| **K-line** | a composition function: an ordered list of stage webs that extends any deep net whose last interface it reads, with the uses it was activated on (§17) | yes |
| **utility** | PSNR (§6) to two decimals of a deep net's last interface against a target of the same shape (§17) | a measurement |
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
| `bad_loop` | a loop1 file breaks the §12 grammar: header, numbering, frame order, a parameter, or a reads path outside the loop's directory |
| `bad_head` | a loop1 or deep1 `%head` is not the hash chain of its blocks (§12) or its stages (§16) |
| `mismatch` | a loop1 frame or a deep1 line differs from what its parameters and the data regenerate (§12, §16) |
| `hash_drift` | a reads file named by a loop1 `act` no longer hashes to its pinned sha256 (§12) |
| `bad_map` | a tensor map breaks §14: its kind, a name, the name order, a dtype, a rank, a value, or the names and ranks its kind defines (§14) |
| `kind_mismatch` | a tensor map of one kind is given where another kind is required, or a view map is asked to invert (§14) |
| `bad_stage` | a deep1 composition has 0 or more than 64 stages, a stage's grid is not 256 per channel of the stage before, or a record names other stages than the webs given (§16) |
| `bad_deep` | a deep1 record breaks the §16 grammar: header, stage count, numbering or fields (§16) |
| `bad_kline` | a kline1 record breaks the §17 grammar: header, stage or use count, numbering, fields, or a use without one utility per prefix (§17) |
| `no_gradients` | an optimizer step is asked for before any gradients were accumulated (§15) |

## §6 Fidelity

```text { type: notebook.cell }
psnr(a, b) = 10 log10(255^2 / mean((a - b)^2))    # binary64, over every level; inf if equal
```

A fidelity claim names the web root, the source media's sha256, and a PSNR to two decimals. A checker that re-decodes must reproduce that PSNR to within 0.01 dB, or the claim fails.

## §7 Deep-net form

A web is already a tensor: for an `r = 2`, `H = 64`, `d = 3` image it is 4,547 numbers in one fixed order, so a deep network can take it as input directly. Two caveats hold for any reader:

- Hidden units can be permuted without changing the render, so two webs of one image may differ only by a permutation. A reader must be permutation-invariant, or every web it reads must be fitted from the same initial weights (the reference encoder's `seed`).
- Webs of different `H`, `r` or `d` have different shapes and are not comparable tensors.
- Depth comes by composition, not by a deeper format: a deep net is an ordered list of webs (§16), and a reusable run of stages is a K-line (§17).

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
- **Autoencoder reading.** Fitting is the encoder, the web weights are the latent code, dna1 is the channel code, and decipher plus §3 decode is the decoder.
- **One strand.** A dna1 sequence has no index and no error correction. A substitution changes about one 16-bit word, or is refused if it makes a repeat. An insertion or deletion shifts every later word and is almost always refused. §10 splits it into oligos that can be synthesized and survive loss and substitutions.

| Id | Predicate |
|---|---|
| D1 | `encipher` depends only on the web bytes; every sequence in `web.json` `dna` equals `encipher` of its web |
| D2 | `decipher(encipher(b)) == b` for every accepted web `b` |
| D3 | no sequence from `encipher` repeats a base |
| D4 | `decipher` refuses every sequence that `encipher` could not have written |
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
- **Refuses, never guesses.** A base covered by no valid read is `gap`. A base whose valid reads disagree with no strict majority (more than half) is `tie`. A sequence that fails §9 afterwards is still refused by `decipher`. The first and last 25 bases are in one oligo each, so losing oligo 0 is `gap` and losing the last oligo is `bad_pool`.
- **Parity is weak, as in the source.** `P` sees only 7 of the 14 index trits, so some index errors pass. A misplaced payload is usually outvoted; an index past the end makes the pool refuse.

| Id | Predicate |
|---|---|
| D7 | every pinned pool in `web.json` equals `pool` of its sequence and hashes to its pinned sha256 |
| D8 | every oligo is 117 nt, starts with `A` or `T`, ends with `C` or `G`, and repeats no base |
| D9 | `unpool(pool(seq)) == seq` under any read order, either strand per read, and the loss of any oligos that leaves every base covered |
| D10 | a substitution in one read is outvoted wherever another two valid reads cover the base |
| D11 | `unpool` refuses (`gap`, `tie`, `homopolymer`, `bad_pool`) rather than return a sequence the reads do not determine |

## §11 Retired

The `cgr1` text glyph and the chaos-game image were excised. The number and the predicates D5, D12 and D13 are not reused. A sequence is pictured through its web (§12 `view1`).

## §12 Pseudovisual chain of thought `loop1`

`loop1` is a chain of thought about one dna1 sequence in which every step is a text picture, not prose: `think` views the data, `hypothesis` draws a prediction in the same form, `act` runs one replayable experiment and `observe` shows what it gave. It follows `Loop(DATA%web) { think CHAIN<%web>, HYPOTHESES; act EXPERIMENTS; }`. `ref/loop.py` is the reference; replay is normative.

```text { type: notebook.cell }
loop    := header frame* head                       # every line ends with LF
header  := "%loop1 nt=" n " sequence=" sha256(ascii(seq)) " web=" root(decipher(seq))
frame   := "%frame " i " " kind " " params, then its body lines; i counts from 0
kind    := think | hypothesis act observe            # a hypothesis is always followed by its act, then its observe
think       view=web size=<1-64, at most the shorter grid side>  -> view1 body     # the web's render, block-mean luminance, " .:-=+*#%@" dark to bright
hypothesis  expect=intact size=<1-64>        -> view1 of the data's web: the picture it predicts
hypothesis  expect=refused:<code> size=<1-64> -> no body: it predicts the read-back refuses with that §5 code
            # code: gap|tie|homopolymer|bad_pool (unpool), bad_length|bad_word|too_large|bad_magic|bad_header|truncated|trailing_bytes|bad_weight (decipher)
act         op=lose   file=<0-8> oligos=<distinct ascending pool indices >= 0, comma-joined>
act         op=damage file=<0-8> seed=<u63> drop=<per mille> sub=<per mille> flip=<per mille>
act         op=reads  file=<0-8> sha256=<hex of the reads file> path=<[A-Za-z0-9._/-]+, under the loop file's directory>
observe     outcome=refused:<code> verdict=<v>                                   -> no body
observe     outcome=intact|changed sequence=<hex> web=<root> used=<n> dropped=<n> verdict=<v>  -> view1 of the read-back web at the hypothesis size, then a diff
head    := "%head " h_n, h_0 = sha256("loop1"), h_(j+1) = sha256(h_j raw || block j)   # header, then each frame block, as ASCII
```

- **Act.** Every op ends in `unpool` (§10), then `decipher` (§9): the read-back is a web. `lose` drops the listed oligos of `pool(seq, file)`. `damage` draws `u(j, s)` = the first 8 bytes (big-endian) of `sha256("loop1/damage" || u64be(seed) || u64be(j) || u64be(s))` and hits when `u * 1000 < rate * 2^64`: slot 0 drops oligo `j`, slot `2 + p` substitutes base `p` with `ACGT[(index + 1 + (u >> 32) mod 3) mod 4]`, and slot 1 reads the oligo from the other strand. `reads` unpools a file of real sequencing reads, pinned by hash, so a wet-lab read-back is a loop too.
- **Verdict.** The outcome is `refused:<code>`, `intact` (the unpooled sequence is the data) or `changed` (another sequence that deciphers to another web). The verdict is `supports` when the outcome equals `expect`, else `refutes`. It is computed, never written by the agent.
- **Diff.** `%diff size=<n> changed=<cells>/<n²>` and a `%key` line, then the view1 frame with one character per cell: `=` same level, `+` brighter, `-` darker in the read-back web than in the data's.
- **No prose.** Every line is a fixed-grammar header or a picture an agent addresses by grid cell. A loop says only what its data and its experiments regenerate.
- **Head.** The head names the whole loop and is 32 bytes, so it fits a `bytes32`. A `Bases` print `receipt` is the head of the print's read-back loop (`wetware-hq/bases`): its last hypothesis is `expect=intact`, its act is `op=reads` on the print's sequencing reads, and anyone holding the loop, the sequence and the reads can replay it.
- **Sample.** `samples/astronaut-64.loop.txt`, from `samples/astronaut-64.loop.json`, refutes "pool1 survives 2% oligo loss and 0.1% substitutions" (`tie`: bases covered by only two oligos cannot outvote one error), supports the revised "loss alone is survivable", and supports "losing oligo 0 is a gap" (the first 25 bases are in one oligo).

| Id | Predicate |
|---|---|
| D14 | every pinned loop in `web.json` replays byte for byte from its sequence and hashes to its pinned head |
| D15 | a loop's head equals the §12 chain over its blocks, so changing any byte of any frame changes the head |
| D16 | every verdict equals `supports` exactly when the replayed outcome equals the hypothesis `expect` |

## §13 Stack

The stack is one primitive with layers over it. A **sequence** (§0) is the primitive: one ASCII string over `ACGT`, named by `dna.md` and defined here. Everything the stack calls DNA is that object. `docs/organize-stack.md` maps which repository holds which layer; this section is what binds them.

```text { type: notebook.cell }
web --deep1--> deep net                                             # stages composed, §16
            deep net --kline1--> deeper net                         # a recalled composition function, §17
media --encode--> web --dna1--> sequence --pool1--> arr<oligo>      # bases, two-way
                                   |--loop1-------> loop            # a view, one-way
                                   |--tess1------->  plan            # wetware-hq/tesseract
                                   +--salted root-> token            # wetware-hq/bases --> helix page
```

| Object | Named by | Computed |
|---|---|---|
| web | web root | `sha256(web bytes)` (§4) |
| sequence | sequence root | `sha256(ascii(seq))` (§9) |
| sequence on chain | chain root | `sha256(salt \|\| ascii(seq))` (§9); Bases `root`, helix `sequence-root` |
| oligo pool | pool digest | `sha256` of the pool file, over the sequence root it rebuilds (§10) |
| loop | head | the frame hash chain (§12) |
| deep net | deep head | the stage hash chain over web roots and interface roots (§16) |
| K-line | K-line head | the hash chain over its stage web roots only (§17) |
| plan | plan root | `sha256` of the canonical `.tess` bytes (`wetware-hq/tesseract`) |
| token | token id | the chain root and length it was minted with (`wetware-hq/bases`) |
| page | locus root | the fold over a page's claims (`wetware-hq/helix`) |
| tensor map | tensor root | `sha256` of the canonical `.tmap` bytes of any object above (§14) |

- **One primitive.** A `dna1` sequence (§9), a `pool1` oligo (§10) and a target or part of a build plan are sequences under extra constraints, not other formats. A layer may narrow what is valid; it may not widen the alphabet, change the record form, or accept `N`, a gap or lower case.
- **One root rule.** Nothing in the stack hashes a sequence another way. A salt is 32 bytes and is kept with the sequence, never on chain (§9). The sequence root and the chain root of one sequence differ, and only a holder of the salt can show they name the same sequence; so an off-chain object that names a sequence (a plan target, a loop header) carries the sequence root and reaches a chromosome by token id, never by comparing it with the chain root.
- **No crossed names.** The values in the table above name different objects. A reader given one where another is required refuses (`root_mismatch`), and never tries the other interpretation.
- **Bases stay in sequence files.** Only a `.dna` record and a pool file hold bases. A token holds a root and a length, a page holds claims, and a view, a diff, a loop and a deep record hold levels, counts and hashes. Nothing above the sequence layer stores a base, so an agent can hold, print and cite a chromosome it cannot read.
- **Two-way or a view.** `web ↔ dna1` and `sequence ↔ pool1` reproduce their input exactly. `media → web` is a measured approximation (§6), and a loop or a deep record is regenerated from its data, never read back into it (§1).
- **Refuse, never guess.** Every conversion returns what the data determines or a §5 code. No layer repairs another layer's object.
- **One tensor map.** Every object in the table has exactly one §14 tensor map, defined by the repository that owns it. It is the only form in which a layer hands its object to a network, and the map's kind travels with its bytes.
- **Pin a neighbour.** A repository in the stack cites and pins its neighbours by commit; it never vendors their files (`docs/pin-sources.md`). web holds the lowest layer that stores a sequence, so it needs no neighbour at run time.

| Id | Predicate |
|---|---|
| S1 | every sequence anywhere in this repository matches `dna.md` §1: non-empty, one record, `ACGT` only |
| S2 | every root of a sequence in `web.json` equals `sha256(ascii(seq))`; no other sequence hash is pinned |
| S3 | no pinned value is cited under two names, and no check accepts one root in place of another |
| S4 | no file outside `*.fa`, a pool file and a `sequence` or `pool` tensor map holds a base of a pinned sequence |
| S5 | `decipher(encipher(b)) == b` and `unpool(pool(seq)) == seq`; a loop or a deep record is regenerated, not stored as data |
| S6 | every refused conversion returns a §5 code and no output |
| S7 | every object kind in the stack has exactly one §14 tensor map, and every map pinned here is the map of its object |

## §14 Tensor maps `tmap1`

A tensor map is the stack's middle layer: every object a layer stores has exactly one form as a set of named tensors, so one reader (a deep net, a TensorFlow graph, a checker) reads every layer the same way. The primitive stays the sequence (§13); a tensor map is how any layer's object, the sequence included, is handed to a network. `tensor.md` names the file form; `ref/tensor.py` is the reference.

```text { type: notebook.cell }
tmap    := "tmap1" u8(len(kind)) kind u16(n) record*n        # all integers little-endian; n is 1-1024
kind    := [a-z][a-z0-9-]{0,31}                              # the object kind, table below
record  := u8(len(name)) name u8(dtype) u8(rank) u32(dim)*rank content
name    := [a-z][a-z0-9_]{0,63}                              # strictly ascending, bytewise, across records
dtype   := TensorFlow DataType: 1 float32 | 2 float64 | 3 int32 | 4 uint8 | 5 int16 | 6 int8
         | 9 int64 | 10 bool | 17 uint16 | 19 float16 | 22 uint32 | 23 uint64
content := prod(dims) values, row-major, little-endian       # = TensorProto.tensor_content
root    := sha256(tmap), cited tmap1:<root>
```

- **TensorFlow is the type system.** A record is a `TensorProto` written flat: `dtype`, `tensor_shape` and `tensor_content`. `ref/tensor.py` `to_tf` builds each record's proto and parses it with `tf.io.parse_tensor`; `from_tf` writes `tf.Tensor`s back to the same bytes. Strings, complex numbers, resources and variants are not tensor-map dtypes: anything a map needs is a number, a bool or a root.
- **One byte form.** A rank-0 record is a scalar. A float is finite and not negative zero; a bool is 0 or 1; nothing follows the last record. A parser accepts `b` only if `pack(unpack(b)) == b`. A file is at most 2^30 bytes. Every break of these rules is `bad_map`; `bad_magic`, `truncated`, `trailing_bytes` and `too_large` keep their §5 meaning.
- **Roots.** A 32-byte root in a map is a `uint8 [32]` tensor. The tensor root names the map, not the object: `tmap1:<root>` is never accepted where a web root, a sequence root or any other §13 name is required, nor one kind's map where another kind is required (`kind_mismatch`).
- **Exact or a view.** An exact map inverts to its object byte for byte. A view map, like the view it maps (§13, S5), is regenerated from data and never read back (`kind_mismatch`).

| Kind | Tensors (`name dtype [shape]`) | Map | Owner |
|---|---|---|---|
| `web` | `activation uint8 []`, `omega0 uint16 []`, `grid uint16 [r]`, `w1 float16 [r,H]`, `b1 float16 [H]`, `w2 float16 [H,H]`, `b2 float16 [H]`, `w3 float16 [H,d]`, `b3 float16 [d]` | exact; the six weight contents are the web's own §2 weight bytes | web |
| `sequence` | `bases uint8 [nt]`, values `0-3` index `ACGT` | exact; one-hot is the view `tf.one_hot(bases, 4)` | web |
| `pool` | `file uint8 []`, `oligos uint8 [m,117]` in pool order | exact; inverts to the `ref/dna.py pool` file | web |
| `loop` | `chain uint8 [b+1,32]` (h_0 … head over b blocks), `kind uint8 [f]` (0 think, 1 hypothesis, 2 act, 3 observe), `verdict uint8 [f]` (0 none, 1 supports, 2 refutes) | view; `chain[-1]` is the §12 head | web |
| `kline` | `chain uint8 [m+1,32]` (k_0 … head over m stages), `stages uint8 [m,32]` (stage web roots), `deeps uint8 [u,32]`, `targets uint8 [u,32]` and `supports bool [u]`, one row per use | view; `chain[-1]` is the §17 head | web |
| `deep` | `chain uint8 [n+1,32]` (h_0 … head over n stages), `outs uint8 [n,32]` (interface roots), `webs uint8 [n,32]` (stage web roots) | view; `chain[-1]` is the §16 head | web |
| `state` | `grid uint16 [r]`, `omega0 uint16 []`, `d_out uint8 []`, `step int64 []`, the trainable tensors `float32`, `m_*` and `v_*` `float32` (Adam's moments, one per trainable); the fit's plan: `learning_rate`, `beta1`, `beta2`, `eps`, `weight_decay`, `grad_clip_norm` `float64 []`, `schedule uint8 []` (0 constant, 1 linear), `total int64 []`, `batch int64 []`, `seed uint64 []`; with an adapter also `base uint8 [32]`, `rank uint16 []` and the base's six weights `float16` | exact; resumes a fit (§15) | web |
| `adapter` | `base uint8 [32]` (the base web root), `<w>_down float16 [rows,k]` and `<w>_up float16 [k,cols]` for each patched `<w>` of `w1`, `w2`, `w3` | exact; merges into a new web (§15) | web |
| `plan` | the `.tess` tree as tensors | defined by `wetware-hq/tesseract` | tesseract |
| `token` | a chromosome token's chain state: roots and counts | defined by `wetware-hq/bases` | bases |
| `order` | a print sale's escrow state | defined by `wetware-hq/med3` | med3 |
| `board` | a forum's message multigraph | defined by `wetware-hq/agora` | agora |

A repository that owns a kind defines its tensors in its own contract and adds its row here when it ships, under these bytes and rules, and pins this section by commit. It never adds a dtype, a second map for one kind, or a tensor that holds bases unless its object is a sequence or a pool.

| Id | Predicate |
|---|---|
| T1 | every map pinned in `web.json` `tensors` is the map of its object, hashes to its pinned root, and `pack(unpack(b)) == b`; every exact map inverts to its object |
| T2 | every dtype number is TensorFlow's `DataType`, and TensorFlow parses every record to a tensor of the same dtype, shape and values |
| T3 | no tensor root is pinned under another name, a reader of one kind refuses every other kind, and every refused map returns a §5 code and no output |
| T4 | only `sequence` and `pool` maps hold bases |

## §15 Weight manipulation

A web's weights can be changed as well as read. `ref/tune.py` changes them the way the Tinker training API does (`tinker-docs.thinkingmachines.ai`): a `Trainer` holds the weights in binary32, `forward_backward(x, t)` adds the mean-squared-error gradients of one batch to those accumulated since the last step, and `optim_step(p)` makes one AdamW update from them (`learning_rate`, `beta1`, `beta2`, `eps`, `weight_decay`, `grad_clip_norm`) and clears them; a step with nothing accumulated is `no_gradients`. With its defaults (full batch, Adam 0.9/0.999/1e-8, constant rate, no decay or clip) `fit` is the reference encoder `ref/web.py` `encode`: the same seed gives the same bytes.

```text { type: notebook.cell }
fresh(grid, d, H, omega0, seed)  -> a full fit from `encode`'s initialisation
from_web(web)                     -> a full fit warm-started from a committed web
lora(web, k)                      -> an adapter: W' = W + down @ up on w1, w2, w3; up starts at zero, the base is frozen
from_state(state[, optimizer])    -> resume, or restart Adam's moments and step
save_weights()  -> web     a canonical web (§2), committed like any other (systems/commit-web.md)
save_state()    -> state   a tmap1 state map (§14): everything needed to resume
save_adapter()  -> adapter a tmap1 adapter map (§14): the factors in binary16 and the base root
merge(web, adapter) -> web W + down @ up in binary64 per patched matrix, then pack (§2): a new web, a new root
```

- **Webs stay the only stored media.** A state and an adapter are tensor maps, not webs: nothing decodes them (§4). An adapter names its base by web root; `merge` refuses any other base (`root_mismatch`), and its output is a new fit under a new path and root (§7). A committed web is never rewritten.
- **Weights out are canonical.** Every web the trainer writes goes through `pack`, so binary16 rounding, finiteness and negative zero follow §2 whatever the optimizer did.
- **Heuristics are measured, not assumed.** Tinker's rules were tuned on language-model fine-tunes; for a 64 × 64 web (H = 64, seed 0, one run each, full batch unless noted) they measure:

| Rule | Measured on `samples/astronaut-64.png` | Use for webs |
|---|---|---|
| Tinker's AdamW defaults, `beta2` 0.95, `eps` 1e-12 | 3,000 steps at 1e-3: 25.62 → 29.49 dB (`beta2` alone: 29.37); 10,000 steps: 31.18 → 31.98 dB | yes, for new fits (`--beta2 0.95 --eps 1e-12`); the reference encoder keeps 0.999 so the pinned sample stays reproducible |
| linear decay to zero | 3,000 steps: 25.62 → 23.97 dB at 1e-3, 28.54 → 26.63 dB at 3e-3 | no |
| learning rate ∝ √batch (`batch_lr`) | 1,024 of 4,096 rows, 3,000 steps: 22.61 dB at the rule's 5e-4, 23.51 at 1e-3, 24.68 at 2e-3 | no; a smaller batch wanted a larger rate |
| warm start from the web (full fit) | 2,000 more linear steps: 31.47 dB at 1e-4, 31.48 at 3e-4, 31.76 at 3e-4 with `beta2` 0.95; same 9,110 bytes | yes, to refine a committed web into a new one |
| LoRA rate 10× the full rate (`lora_lr`) | rank 8, 2,000 linear steps, 1e-4 to 3e-3: 31.21–31.22 dB | untested; the rank limits, not the rate |
| LoRA rank (`param_count`) | adapter bytes / merged PSNR: rank 8 2,874 / 31.23, 16 4,922 / 31.29, 32 9,018 / 31.44, 64 17,210 / 31.96 dB | an adapter pays only near the size of the web itself; use it to ship a small named patch, not for fidelity |

| Id | Predicate |
|---|---|
| U1 | `fit` of a fresh trainer with default parameters writes the bytes `encode` writes for the same media, size, steps and seed |
| U2 | a fit saved to a state and resumed with its optimizer writes the bytes of one uninterrupted fit of the same total steps; the state keeps the fit's AdamW settings, schedule, horizon (`--total`), batch and seed, and minibatches and the linear rate are keyed by the absolute step |
| U3 | a rank-k adapter before its first step merges to its base, byte for byte |
| U4 | `merge(samples/astronaut-64.web, samples/astronaut-64.r8.adapter.tmap)` hashes to the root `web.json` `tune` pins, and decodes within 0.01 dB of its PSNR |
| U5 | an adapter merged onto any web but its base is `root_mismatch`; a map of another kind is `kind_mismatch`; an adapter missing a factor or with a factor of another dtype or shape is `bad_map` |

## §16 Composition `deep1`

A deep net is a composition of shallow ones. Each stage is an ordinary web (§2), so composing adds depth without adding a second network form. `ref/deep.py` is the reference; replay is normative.

```text { type: notebook.cell }
deep(w_0)       = decode(w_0)                                   # depth 1: a web is a deep net of one stage (§3)
deep(w_0 .. w_n) = stage(w_n, deep(w_0 .. w_(n-1)))             # depth n + 1, by induction
stage(w, L)     = levels(forward(w, AXIS[L]))  per grid point   # AXIS = the §3 axis of size 256: AXIS[l] = -1 + 2l/255
requires          grid(w_i) = (256,) * d(w_(i-1)) for i >= 1    # a stage's input rank is the channel count before it
out_i           = sha256(levels of stage i)                     # uint8, row-major over grid(w_0) + (d_i,)
h_0 = sha256("deep1"),  h_(i+1) = sha256(h_i || root(w_i) || out_i)   # raw 32-byte values; head = h_n
record := "%deep1 stages=" n " grid=" grid(w_0) LF
          ("%stage " i " web=" root(w_i) " d=" d_i " out=" out_i LF) * n
          "%head " h_n LF
```

- **Induction.** A stage reads the levels before it as positions on its own grid of 256 levels per axis, so `stage(w, L)` is `decode(w)` read at index `L`. Composing uses only the §3 decode. As in §3, two maths libraries can disagree on a level only where a stage's output lies within rounding error of a half, so a record replays byte for byte on one library; `ref/deep.py` evaluates every stage in the same 4,096-point batches as `decode`. A deep net of n stages is a sine network of 2n hidden layers with a uint8 bottleneck between each pair of stages. Universal approximation (`docs/pin-sources.md` §Theory) is a claim about one stage; composing a stage onto a deep net is the only way to make a deeper one.
- **Interfaces are bound.** The interface between two stages is a uint8 tensor, so it has one byte form and one root, `out_i`. The chain binds each stage's web root to the interface it wrote, so a 32-byte head fits a `bytes32` and names every stage and every interface (`systems/commit-web.md`). Each link `(root(w_i), out_(i-1), out_i)` has the shape of the statement a ZKML proof certifies, that a committed model produced this output from this input (Chen et al., EuroSys 2024). ZKML computes in fixed point and looks up non-linearities in tables. A uint8 interface fits that, but the binary64 `sin` of §3 does not, so no proof system is pinned and replay is the verifier.
- **Refuse, never guess.** A stage that does not fit the interface before it, or a composition of 0 or more than 64 stages, is `bad_stage`. A record off the grammar is `bad_deep`, a head that is not the chain of its lines is `bad_head`, a web that is not the stage it is named for is `root_mismatch`, and a line that does not replay is `mismatch`.
- **A view.** A record is regenerated from its webs and never read back into one, like a loop (§13, S5).
- **Sample.** `samples/luma.web` is a 34-byte stage set by hand, not fitted (`ref/deep.py luma`): `H = 1`, `omega0 = 1`, `y = sin(sin(ε c·x)) / ε` with the BT.601 weights `c = (0.299, 0.587, 0.114)` and `ε = 1/64`, so its grid is `256 × 256 × 256` and `d = 1`. `samples/astronaut-64.luma.deep` composes the sample web with it. The two-stage deep net renders the astronaut's luma at 32.47 dB against the BT.601 luma of the source, and its head is pinned in `web.json` `composition`.

| Id | Predicate |
|---|---|
| C1 | `deep([w]) == decode(w)`, and every record pinned in `web.json` `composition` replays byte for byte from its webs and hashes to its pinned sha256 and head |
| C2 | `deep(w_0 .. w_n)` is `decode(w_n)` read at the levels of `deep(w_0 .. w_(n-1))`, and its first n interfaces are those of `deep(w_0 .. w_(n-1))` |
| C3 | a stage whose grid is not 256 per channel of the stage before is `bad_stage`, and nothing is written |
| C4 | the head is the §16 chain over each stage's web root and interface root, so changing any stage or any interface changes the head |
| C5 | every refused record returns a §5 code and no head |

## §17 K-lines `kline1`

A **K-line** is a composition function over shallow webs: an ordered list of stage webs that, activated on any deep net whose last interface it reads, composes those stages after it (§16). It is Minsky's knowledge-line (`docs/recall-klines.md`): a handle that switches back on the set of agents, here stages, that solved a problem of this kind before. `ref/compose.py` is the reference; replay is normative.

```text { type: notebook.cell }
K(D)          = D ++ (s_0 .. s_(m-1))                                # a deep net D, extended by the K-line's stages
requires        grid(s_0) = (256,) * d(D),  grid(s_i) = (256,) * d(s_(i-1))
utility(D, t) = round(psnr(deep(D), t), 2) if shapes agree, else none  # §6; none is below every number
k_0 = sha256("kline1"),  k_(i+1) = sha256(k_i || root(s_i))           # raw 32-byte values; head = k_m
use           = (deep head of K(D), sha256(t), utility of D and of each prefix of K(D) past D)
supports(use) = every utility in the use is above the one before it
recalled(K)   = supports holds for uses on at least 2 different targets
record := "%kline1 stages=" m " in=" d(D) " out=" d(s_(m-1)) " uses=" u LF
          ("%stage " i " web=" root(s_i) " d=" d(s_i) LF) * m
          "%head " k_m LF
          ("%use " j " deep=" deep-head " target=" sha256(t) " utility=" u_0 "," … "," u_m LF) * u   # u_i: none, inf, or 2 decimals
```

```text { type: notebook.cell }
grow(D, t, klines, library):                                          # induction on depth
  loop at most 8 times:
    candidates := every recalled K-line, then every library stage, whose first stage reads d(D)
                  + a fitted curve, when d(D) = 1 and D's last interface has t's shape
    best       := the candidate of highest utility(D ++ candidate, t); ties go to the earlier one
    if utility rises: D := D ++ best, else stop
  return the stages added and the utility after each kept step
```

- **Induction.** The base is a deep net (one web is one, §16). The step adds a K-line or a stage only when it raises the objective, the utility of the output deep net, so every deep net `grow` returns is better than its start at every step, and `grow` stops at the first step that would not help. The stages it added are a new K-line, which counts only once it is replicated.
- **A fitted curve.** On a one-channel interface the stage that best maps levels to the target is the mean target at each level (least squares; levels never taken are interpolated). `fit_curve` fits a rank-1 web (`H = 16`, `omega0 = 30`, 3,000 steps, seed 0) to that table on the 256-level grid, so a fitted stage is an ordinary web. The fit is one valid fit, like `encode`: replay decodes the committed stage, never refits it.
- **Replication is the recall rule.** A K-line's head names its stages, not its evidence, so the K-line keeps one name as uses accumulate. A use records the utility after every stage, so a stage that did not help shows. `grow` activates only recalled K-lines, the rule `docs/recall-klines.md` applies to the stack's own moves: a composition that worked on one target is a result, one that worked on two is a method.
- **Citizenship.** A stage is a web, so it has a sequence (§9) and can be a Bases chromosome; a K-line names its stages by web root and reaches a chromosome by token id, never by its chain root (§13, S3). A K-line holds roots and utilities, never a base (S4).
- **Refuse, never guess.** A record off the grammar is `bad_kline`, a head that is not the chain over its stage lines is `bad_head`, a stage that does not read the interface before it is `bad_stage`, a web that is not the stage it is named for is `root_mismatch`, and a use that does not replay is `mismatch`.
- **A view.** A record is regenerated from its stages, inputs and targets and never read back into a web (§13, S5).
- **Sample.** The task is the BT.601 luma of a source (`compose.luma_target`: `(299 R + 587 G + 114 B) / 1000` in integers, half to even, so the target is the same on every machine), on two pinned sources: the astronaut and a 64 × 64 crop of scikit-image's coffee (`web.json` `webs`). `samples/luma.kline` is the K-line of the hand-set stage `samples/luma.web` (§16): it raises both targets, to 32.47 and 34.93 dB, so it is recalled. `grow` on the astronaut adds that stage, then a fitted curve, `samples/astronaut-64.curve.web` (656 bytes), to 32.55 dB, and stops. `samples/astronaut-64.luma-curve.kline` is those two stages: on the coffee the astronaut's curve lowers 34.93 to 34.42 dB, so that K-line is not recalled. `grow` on the coffee with `luma.kline` recalled takes it as its first step, then fits the coffee's own curve to 35.15 dB.

| Id | Predicate |
|---|---|
| L1 | every record pinned in `web.json` `klines` replays byte for byte from its stages and uses, hashes to its pinned sha256 and head, and is recalled exactly when pinned so |
| L2 | `K(D)` is `deep(D ++ stages)` (§16), and the head is the §17 chain over the stage web roots only, so changing a stage changes the head and adding a use does not |
| L3 | `grow` keeps a step only when utility rises, and every kept step's utility is above the step before it |
| L4 | `grow` activates no K-line that is not recalled |
| L5 | every refused record returns a §5 code and no head |

## Citations

- Universal approximation: Cybenko 1989, Hornik 1991 (`docs/pin-sources.md` §Theory).
- Sine networks for signals: SIREN, arXiv:2006.09661.
- Media as network weights: COIN, arXiv:2103.03123.
- Networks reading weights: functa, arXiv:2201.12204; DWSNets, arXiv:2301.12780.
- Tensors and their wire form: TensorFlow `DataType` and `TensorProto` (`tensorflow/core/framework/types.proto`, `tensor.proto`; `docs/pin-sources.md` §Tensors).
- Training API, checkpoints and LoRA heuristics: Tinker (`tinker-docs.thinkingmachines.ai`); LoRA, arXiv:2106.09685; AdamW, arXiv:1711.05101.
- DNA storage and rotating code: Goldman et al. 2013; JPEG-DNA, arXiv:2103.09616 (`docs/pin-sources.md` §DNA).
- Committed models and proofs of inference: ZKML, Chen et al., EuroSys 2024 (`docs/pin-sources.md` §Composition).
- K-lines: Minsky, *K-Lines: A Theory of Memory*, Cognitive Science 4 (1980) 117–133; greedy layer-wise growth: Bengio et al., NeurIPS 2006 (`docs/pin-sources.md` §Composition).

## Falsification

- A file accepted by `unpack` re-serializes to different bytes.
- Two decodes of one web on one machine differ.
- `python ref/web.py check samples/astronaut-64.web samples/astronaut-64.png` prints a PSNR more than 0.01 dB from `web.json`.
- `python ref/dna.py decipher samples/astronaut-64.fa out.web` writes bytes other than `samples/astronaut-64.web`.
- A sequence in the stack is hashed two different ways, or a token, a page or a loop is found holding bases (§13).
- `python ref/loop.py verify samples/astronaut-64.fa samples/astronaut-64.loop.txt` refuses, or prints a head other than `web.json`.
- `python ref/tensor.py map web samples/astronaut-64.web out.tmap` writes bytes other than `samples/astronaut-64.web.tmap`, or `tf.io.parse_tensor` reads one of its records to another dtype, shape or value (§14).
- `python ref/deep.py verify samples/astronaut-64.luma.deep samples/astronaut-64.web samples/luma.web` refuses, or prints a head other than `web.json` `composition` (§16).
- `python ref/compose.py verify samples/luma.kline --stages samples/luma.web --use samples/astronaut-64.png samples/astronaut-64.web --use samples/coffee-64.png samples/coffee-64.web` refuses, or prints a head other than `web.json` `klines` (§17).
- `python ref/tune.py merge samples/astronaut-64.web samples/astronaut-64.r8.adapter.tmap out.web` writes a web whose root is not `web.json` `tune` `merged` (§15).

## Licence

MIT, see `LICENSE`.
