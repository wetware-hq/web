# `pin-sources` — what may be cited, fetched or encoded

`type: resource.policy` · `spec: agent.policy` · subordinate to `docs/follow-agent-contract.md`, which governs on any conflict.

Every external input is pinned by a stable URL and a sha256. Sources without a licence are cited and pinned, never vendored. Media that may be committed is public domain or carries a licence that allows it.

## Specification

```json { type: notebook.cell }
{
  "id": "docs/pin-sources.md",
  "type": "resource.policy",
  "role": "list the theory, methods and media this repository may rely on",
  "reads": ["web.json"],
  "writes": [],
  "must": ["pin media by sha256", "vendor only public-domain or licensed media", "cite papers, never copy them"],
  "check": ["W5"]
}
```

## Graph

- ← [follow-agent-contract.md](follow-agent-contract.md)
- ← [recall-klines.md](recall-klines.md)
- → [../skills/encode-media.md](../skills/encode-media.md)

## Theory

| Claim | Source |
|---|---|
| One hidden layer of sigmoidal units approximates any continuous function on a compact set | Cybenko, G. (1989) Approximation by superpositions of a sigmoidal function. *Math. Control Signals Systems* 2, 303–314. doi:10.1007/BF02551274 |
| The result holds for any bounded non-constant activation, and is a property of the architecture | Hornik, K. (1991) Approximation capabilities of multilayer feedforward networks. *Neural Networks* 4(2), 251–257. doi:10.1016/0893-6080(91)90009-T |
| Why two hidden layers (the seed for `--layers 2`) | https://sparsity.ucsd.edu/posts/2020/surprising-properties-two-hidden-layer-neural-networks/ |

The theorems guarantee that a good enough web exists for enough units. They do not say how many units a given image needs, or that gradient descent finds them. The sample's fidelity is measured, not derived.

## Methods

| Use | Source |
|---|---|
| Sine activations, `omega0 = 30`, first-layer and hidden-layer initialisation | Sitzmann et al. (2020) SIREN, arXiv:2006.09661 |
| Storing an image as the quantised weights of a coordinate network | Dupont et al. (2021) COIN, arXiv:2103.03123 |

## Weight space

| Use | Source |
|---|---|
| Treating each datum as a network and training deep nets on the weights; shared initialisation | Dupont et al. (2022) From data to functa, arXiv:2201.12204 |
| Deep nets that respect the permutation symmetry of hidden units | Navon et al. (2023) DWSNets, arXiv:2301.12780 |

## Tensors

| Use | Source |
|---|---|
| `DataType` numbers for the §14 dtype byte | TensorFlow, `tensorflow/core/framework/types.proto` (https://github.com/tensorflow/tensorflow), Apache-2.0; checked against `tensorflow-cpu` 2.21.0 by `tests/test_tensor.py` |
| `TensorProto` `dtype`, `tensor_shape`, `tensor_content`: the record a `.tmap` writes flat | TensorFlow, `tensorflow/core/framework/tensor.proto`, Apache-2.0 |
| Tensors as the common input of deep nets | Abadi, M. et al. (2016) TensorFlow: a system for large-scale machine learning. *OSDI '16*, 265–283. arXiv:1605.08695 |

## DNA

| Use | Source |
|---|---|
| Rotating base-3 code with no repeated base; 117-nt oligos with 4× overlap, index, parity and alternating reverse complement (pool1) | Goldman, N. et al. (2013) Towards practical, high-capacity, low-maintenance information storage in synthesized DNA. *Nature* 494, 77–80. doi:10.1038/nature11875 |
| Chaos-game representation of a sequence | Jeffrey, H. J. (1990) Chaos game representation of gene structure. *Nucleic Acids Res.* 18(8), 2163–2170. doi:10.1093/nar/18.8.2163 |
| Integer CGR: a sequence is recovered exactly from its CGR point | iCGR, arXiv:1712.04546 |
| Image coding for DNA: GC balance and homopolymer limits | JPEG-DNA, arXiv:2103.09616 |
| Autoencoder latents coded into DNA | arXiv:2306.12882 |
| Finder patterns, format information and a check value on a printed code (cgr1 frame) | ISO/IEC 18004:2015 *QR Code bar code symbology specification* |
| Implicit-neural-representation weights coded into DNA | arXiv:2309.06956 |

## Media

| Path | Source | sha256 | Licence |
|---|---|---|---|
| upstream | `https://gitlab.com/scikit-image/data/-/raw/d448e33739b0da758f44c69bd185747ab6696ada/astronaut.png` (512 × 512 RGB) | `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`, the same digest as scikit-image v0.25.2 `skimage/data/_registry.py` | NASA photograph of Eileen Collins; scikit-image's `astronaut()` docstring: "No known copyright restrictions, released into the public domain." |
| `samples/astronaut-64.png` | `python ref/web.py shrink <upstream> samples/astronaut-64.png 8` (8 × 8 box mean, round half to even) | `a4dcbffddaf26269e3b6007f3d7b6e24d529888f71abcd2ac42dea58dcf48aa7` | public domain, as upstream |

The upstream file is fetched and checked, not vendored. Only the 64 × 64 derivative is committed.

## Indications

- Choosing a source image for a new sample.
- Checking that a cited theorem or method says what the contract uses it for.

## Contraindications

- Never encode and commit generated media whose licence does not allow redistribution of a derivative; a web is a derivative.
- Never cite a render as a source.

## Falsification

- An upstream URL returns bytes with a different sha256.
- `samples/astronaut-64.png` differs from `shrink(upstream, 8)`.

## Licence

MIT, see `LICENSE`. Papers and upstream media keep their own terms.
