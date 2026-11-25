# web

**Images stored as tiny neural networks, so deep nets can read them and blockchains can hold them.**

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
python ref/dna.py cgr      samples/astronaut-64.fa  cgr.png                  # chaos-game image of the DNA
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

## A web as synthetic DNA

The `dna1` cipher ([`ref/dna.py`](ref/dna.py), contract §9) writes a web as one DNA sequence and reads it back to the exact same bytes. Each 2 bytes become 11 bases, and no base ever repeats, so the sequence has no homopolymers. The sample web becomes a 50,105-base sequence ([`samples/astronaut-64.fa`](samples/astronaut-64.fa)) with 50.1% GC.

Going the other way, a chaos-game representation turns the sequence into an image ([`samples/astronaut-64.cgr.png`](samples/astronaut-64.cgr.png)). That image can be fitted as a web too ([`samples/astronaut-64.cgr.web`](samples/astronaut-64.cgr.web), PSNR 36.36 dB). This is a storage form, not compression.

The `pool1` layer (contract §10) splits that sequence into 2,002 oligos of 117 bases each, after Goldman et al. 2013 ([`samples/astronaut-64.pool.fa`](samples/astronaut-64.pool.fa)). Each oligo carries its own index and a parity check, and each base is in up to 4 oligos. The sequence comes back from reads in any order, from either strand, after some oligos are lost, and after substitutions that the other copies outvote. If the reads do not settle a base, `unpool` refuses with `gap` or `tie` rather than guess.

The `cgr1` glyph (contract §11) prints the chaos-game grid as plain text an agent can read, with labelled corners and axes and a hash of the counts, like the finder patterns and check value on a QR code ([`samples/astronaut-64.cgr3.txt`](samples/astronaut-64.cgr3.txt)).

```sh
python ref/dna.py pool samples/astronaut-64.fa pool.fa        # 117-nt oligos
python ref/dna.py unpool pool.fa seq.fa                       # back to the sequence, by majority vote
python ref/dna.py glyph samples/astronaut-64.fa --k 3         # text CGR
```

## What is in this repository

| Path | What it covers |
|---|---|
| [`DISCLAIMER`](DISCLAIMER) | The one-line notice. Agents read it first. |
| [`web.md`](web.md) | The `.web` file format: extension, media type, magic and axioms. |
| [`docs/read-repository.md`](docs/read-repository.md) | Where to start: which file to read for which task. |
| [`docs/follow-agent-contract.md`](docs/follow-agent-contract.md) | The rulebook: bytes, decoder, root, error codes and checks. It wins if files disagree. |
| [`docs/pin-sources.md`](docs/pin-sources.md) | The theorems, papers and sample image, each pinned. |
| [`docs/vary-cybernetics.md`](docs/vary-cybernetics.md) | Design notes on linking webs to deep nets and to living neurons (hypotheses only). |
| [`skills/encode-media.md`](skills/encode-media.md) | Fitting a web to an image. |
| [`skills/decode-web.md`](skills/decode-web.md) | Rebuilding an image and checking a web. |
| [`systems/commit-web.md`](systems/commit-web.md) | Putting a web on chain and reading it back. |
| [`ref/web.py`](ref/web.py) | The reference encoder and decoder (Python + numpy). |
| [`ref/dna.py`](ref/dna.py) | The `dna1` cipher: web ↔ synthetic DNA, plus chaos-game images. |
| [`web.json`](web.json) | Pins for the sample: source hash, web root, size and PSNR, and its DNA sequence. |
| [`ooc/web.og`](ooc/web.og) | A compact map of the files, for agents that traverse links. |

## Current status

- Images only. The format reserves room for audio and video, but nothing checks them yet.
- No chain contract yet (`web.json` `chain` is `null`), so no web can be committed.
- At 64 × 64 a web is smaller than raw pixels (12,288 bytes) and the PNG (11,034 bytes), but not by much. The point is the form, not compression.

## Licence

MIT. See [`LICENSE`](LICENSE). Cited papers and the sample's upstream image keep their own terms.
