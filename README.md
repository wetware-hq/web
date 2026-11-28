# web

**Images stored as tiny neural networks, so deep nets can read them, be built from them, and blockchains can hold them.**

> web is academic research material. A web is a lossy approximation of its source, not a copy, a proof of authorship or a licence to it. (Full text: [`DISCLAIMER`](DISCLAIMER).)

## What is this?

A **web** (a `.web` file, defined in [`web.md`](web.md)) is one image saved as the weights of a small neural network instead of as pixels. The network has two hidden layers of 64 sine units. Give it a pixel's position and it returns that pixel's colour. Run it over every position and you get the image back.

This works because of the **universal approximation theorem**: a network of this shape can approximate any reasonable image as closely as you like, given enough units. So one fixed design can hold any image.

Why store media this way:

- **Deep nets can read it directly.** A web is a fixed list of 4,547 numbers, so another network can take it as input with no image decoder in between.
- **Chains can hold it.** The sample web is 9,110 bytes. Its SHA-256 hash (its **root**) is a 32-byte commitment that can go on chain, and the bytes themselves fit in a transaction's calldata for about 146,000 gas.
- **It never changes.** A committed web is never rewritten. A better fit is a new web with a new root.

## Try it

```sh
pip install -r requirements.txt
python ref/web.py check  samples/astronaut-64.web samples/astronaut-64.png   # root, size, PSNR
python ref/web.py decode samples/astronaut-64.web out.png                    # rebuild the image
python ref/web.py encode samples/astronaut-64.png my.web                     # fit a new web (~70 s)
python ref/dna.py encipher samples/astronaut-64.web my.fa                     # web -> synthetic DNA
python ref/dna.py decipher samples/astronaut-64.fa  back.web                 # DNA -> the same web bytes
python ref/deep.py verify samples/astronaut-64.luma.deep samples/astronaut-64.web samples/luma.web   # a deep net of two webs
python -m unittest discover -s tests
```

The sample is a 64 × 64 photo of astronaut Eileen Collins (NASA, public domain). Its web rebuilds it at a PSNR of 31.18 dB.

## How a web is built

| Step | What happens |
|---|---|
| **Encode** | Train the network so its output matches the image (10,000 steps of Adam). Then round every weight to 16-bit floats. |
| **Pack** | Write a short header (shape, width, settings) and the weights in one fixed order. There is exactly one valid byte form. |
| **Root** | Hash the bytes with SHA-256. The root names the web. |
| **Decode** | Run the network at every pixel in 64-bit maths and round to 0–255. This is the only step the rules fix exactly. |

The full byte layout, decoder and error codes are in [`docs/follow-agent-contract.md`](docs/follow-agent-contract.md).

## One primitive: the DNA sequence

Across the wetware stack, DNA means one thing: a single sequence over `ACGT`, named by [`dna.md`](dna.md) and identified by `sha256` of its bases. Every layer is built over it, and each layer is named by one value ([`docs/organize-stack.md`](docs/organize-stack.md), contract §13):

| Layer | Repository | Named by |
|---|---|---|
| shallow network (`.web`) | web | web root |
| sequence (`.dna`), the primitive | web | sequence root |
| `arr<oligo>` (`pool1`) | web | pool digest |
| deep net (`deep1`) | web | deep head |
| chain of thought (`loop1`) | web | loop head |
| build plan (`.tess`) | [tesseract](https://github.com/wetware-hq/tesseract) | plan root |
| any object above, as tensors (`.tmap`) | web ([`tensor.md`](tensor.md)) | tensor root |
| web3 coin (chromosome token) | [bases](https://github.com/wetware-hq/bases) | chain root (salted) and length |
| wiki commit (`dna:` page) | [helix](https://github.com/wetware-hq/helix) | locus root |

Only sequence files hold bases. A token, a page and a loop hold roots and counts, so a chromosome can be held, printed and cited by agents that never see its sequence. `tests/test_stack.py` checks this repository against those rules.

The universal middle layer is the tensor: every object in the table maps to exactly one set of named TensorFlow tensors ([`tensor.md`](tensor.md), contract §14, [`ref/tensor.py`](ref/tensor.py)). The web, the sequence and the pool maps invert exactly; the loop and the deep maps are views.

```sh
python ref/tensor.py map  web samples/astronaut-64.web out.tmap   # = samples/astronaut-64.web.tmap
python ref/tensor.py show out.tmap                                # kind, root and each tensor's dtype and shape
python ref/tensor.py unmap out.tmap back.web                      # the same 9,110 bytes
```

Weights can be changed as well as read ([`ref/tune.py`](ref/tune.py), contract §15), after the [Tinker](https://tinker-docs.thinkingmachines.ai/tinker/) training API: `forward_backward` and `optim_step`, a resumable `state` map, and LoRA `adapter` maps that merge onto their base web as a new web. Each Tinker heuristic is measured on the sample in §15; Tinker's AdamW `beta2` 0.95 is worth +0.8 dB at 10,000 steps.

```sh
python ref/tune.py fit samples/astronaut-64.png new.web --beta2 0.95 --eps 1e-12 --state new.state.tmap
python ref/tune.py fit samples/astronaut-64.png better.web --from-web samples/astronaut-64.web --steps 2000 --lr 3e-4 --beta2 0.95
python ref/tune.py merge samples/astronaut-64.web samples/astronaut-64.r8.adapter.tmap merged.web
```

## A web as synthetic DNA

The `dna1` cipher ([`ref/dna.py`](ref/dna.py), contract §9) writes a web as one DNA sequence and reads it back to the exact same bytes. Each 2 bytes become 11 bases, and no base ever repeats, so the sequence has no homopolymers. The sample web becomes a 50,105-base sequence ([`samples/astronaut-64.fa`](samples/astronaut-64.fa)) with 50.1% GC.

This is a storage form, not compression.

The `pool1` layer (contract §10) splits that sequence into 2,002 oligos of 117 bases each, after Goldman et al. 2013 ([`samples/astronaut-64.pool.fa`](samples/astronaut-64.pool.fa)). Each oligo carries its own index and a parity check, and each base is in up to 4 oligos. The sequence comes back from reads in any order, from either strand, after some oligos are lost, and after substitutions that the other copies outvote. If the reads do not settle a base, `unpool` refuses with `gap` or `tie` rather than guess.

## Thinking in pictures: `loop1`

A `loop1` file (contract §12, [`ref/loop.py`](ref/loop.py)) is a chain of thought about a sequence in which every step is a text picture an agent can read, not prose. It **thinks** by viewing the data (the web's image as a 16 × 16 character grid), states a **hypothesis** as the picture it expects to see or the refusal it expects, **acts** with one replayable experiment on the oligo pool, and **observes** the web read back from it, as the same grid and a diff grid. The verdict is computed, and the frames are hash-chained, so the 32-byte head names the whole loop and can be a `Bases` print receipt.

The sample loop ([`samples/astronaut-64.loop.txt`](samples/astronaut-64.loop.txt)) found a real limit of `pool1`: 0.1% substitutions already cause a `tie` where only two oligos cover a base, while 2% oligo loss alone is survivable.

```sh
python ref/loop.py run    samples/astronaut-64.fa samples/astronaut-64.loop.json my.loop.txt
python ref/loop.py verify samples/astronaut-64.fa samples/astronaut-64.loop.txt   # replay every act; verdicts and head
```

```sh
python ref/dna.py pool samples/astronaut-64.fa pool.fa        # 117-nt oligos
python ref/dna.py unpool pool.fa seq.fa                       # back to the sequence, by majority vote
```

## Deep nets from shallow webs: `deep1`

A deep net is built by induction (contract §16, [`ref/deep.py`](ref/deep.py)). One web is a deep net of one stage. A deep net of n + 1 stages feeds the colour levels of the first n stages into one more web, which reads them as positions on its own grid of 256 levels per channel. Every stage is an ordinary web, so composing needs only the normal decoder.

The interfaces are bound in a hash chain. Each stage contributes its web root and the root of the levels it wrote, and the chain ends in a 32-byte head that fits on chain. Each link is the kind of claim a [ZKML](https://ddkang.github.io/papers/2024/zkml-eurosys.pdf) proof certifies (this committed model turned this input into this output). Here, replaying the composition is what checks it.

The sample ([`samples/astronaut-64.luma.deep`](samples/astronaut-64.luma.deep)) composes the astronaut web with a 34-byte, hand-set stage ([`samples/luma.web`](samples/luma.web)) that turns RGB into luma. The two-stage net renders the astronaut in grey at 32.47 dB against the luma of the source.

```sh
python ref/deep.py run    my.deep samples/astronaut-64.web samples/luma.web     # compose; prints the head
python ref/deep.py decode gray.png samples/astronaut-64.web samples/luma.web    # render the last stage
```

## What is in this repository

| Path | What it covers |
|---|---|
| [`DISCLAIMER`](DISCLAIMER) | The one-line notice. Agents read it first. |
| [`web.md`](web.md) | The `.web` file format: extension, media type, magic and axioms. |
| [`dna.md`](dna.md) | The `.dna` sequence form: the stack primitive, its alphabet, root and axioms. |
| [`tensor.md`](tensor.md) | The `.tmap` tensor map: every stack object as named TensorFlow tensors. |
| [`docs/organize-stack.md`](docs/organize-stack.md) | Where each wetware repository sits over the primitive, and the value each boundary carries. |
| [`docs/recall-klines.md`](docs/recall-klines.md) | The K-lines: the moves that worked in more than one chain of thought, and what to reactivate for each. |
| [`docs/read-repository.md`](docs/read-repository.md) | Where to start: which file to read for which task. |
| [`docs/follow-agent-contract.md`](docs/follow-agent-contract.md) | The rulebook: bytes, decoder, root, error codes and checks. It wins if files disagree. |
| [`docs/pin-sources.md`](docs/pin-sources.md) | The theorems, papers and sample image, each pinned. |
| [`docs/vary-cybernetics.md`](docs/vary-cybernetics.md) | Design notes on linking webs to deep nets and to living neurons (hypotheses only). |
| [`skills/encode-media.md`](skills/encode-media.md) | Fitting a web to an image. |
| [`skills/decode-web.md`](skills/decode-web.md) | Rebuilding an image and checking a web. |
| [`systems/commit-web.md`](systems/commit-web.md) | Putting a web on chain and reading it back. |
| [`ref/web.py`](ref/web.py) | The reference encoder and decoder (Python + numpy). |
| [`ref/dna.py`](ref/dna.py) | The `dna1` cipher and `pool1` oligos: web ↔ synthetic DNA. |
| [`ref/deep.py`](ref/deep.py) | The `deep1` composition: shallow webs into a deep net, interfaces hash-chained. |
| [`ref/loop.py`](ref/loop.py) | The `loop1` pseudovisual chain of thought: think, hypothesize, act, observe. |
| [`web.json`](web.json) | Pins for the sample: source hash, web root, size and PSNR, and its DNA sequence. |
| [`ooc/web.og`](ooc/web.og) | A compact map of the files, for agents that traverse links. |

## Current status

- Images only. The format reserves room for audio and video, but nothing checks them yet.
- No chain contract yet (`web.json` `chain` is `null`), so no web can be committed.
- At 64 × 64 a web is smaller than raw pixels (12,288 bytes) and the PNG (11,034 bytes), but not by much. The point is the form, not compression.

## Licence

MIT. See [`LICENSE`](LICENSE). Cited papers and the sample's upstream image keep their own terms.
