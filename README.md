# web

**Images stored as tiny neural networks, so deep nets can read them and blockchains can hold them.**

> web is academic research material. A web is a lossy approximation of its source, not a copy, a proof of authorship or a licence to it. (Full text: [`DISCLAIMER`](DISCLAIMER).)

## What is this?

A **web** is one image saved as the weights of a small neural network instead of as pixels. The network has two hidden layers of 64 sine units. Give it a pixel's position and it returns that pixel's colour. Run it over every position and you get the image back.

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

## What is in this repository

| Path | What it covers |
|---|---|
| [`DISCLAIMER`](DISCLAIMER) | The one-line notice. Agents read it first. |
| [`docs/read-repository.md`](docs/read-repository.md) | Where to start: which file to read for which task. |
| [`docs/follow-agent-contract.md`](docs/follow-agent-contract.md) | The rulebook: bytes, decoder, root, error codes and checks. It wins if files disagree. |
| [`docs/pin-sources.md`](docs/pin-sources.md) | The theorems, papers and sample image, each pinned. |
| [`docs/vary-cybernetics.md`](docs/vary-cybernetics.md) | Design notes on linking webs to deep nets and to living neurons (hypotheses only). |
| [`skills/encode-media.md`](skills/encode-media.md) | Fitting a web to an image. |
| [`skills/decode-web.md`](skills/decode-web.md) | Rebuilding an image and checking a web. |
| [`systems/commit-web.md`](systems/commit-web.md) | Putting a web on chain and reading it back. |
| [`ref/web.py`](ref/web.py) | The reference encoder and decoder (Python + numpy). |
| [`web.json`](web.json) | Pins for the sample: source hash, web root, size and PSNR. |
| [`ooc/web.og`](ooc/web.og) | A compact map of the files, for agents that traverse links. |

## Current status

- Images only. The format reserves room for audio and video, but nothing checks them yet.
- No chain contract yet (`web.json` `chain` is `null`), so no web can be committed.
- At 64 × 64 a web is smaller than raw pixels (12,288 bytes) and the PNG (11,034 bytes), but not by much. The point is the form, not compression.

## Licence

MIT. See [`LICENSE`](LICENSE). Cited papers and the sample's upstream image keep their own terms.
