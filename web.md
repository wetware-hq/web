# `.web`

A `.web` file is one piece of static media stored as the weights of a two-hidden-layer sine network. The network maps a position to the value there. The file holds the weights, not the pixels, and `sha256` of its bytes is its root. This page names the file format. [`docs/follow-agent-contract.md`](docs/follow-agent-contract.md) defines it and governs on any conflict.

```toml
[web]
media_type = "application/vnd.web"     # vendor tree, not registered with IANA
extension  = ".web"
magic      = "web1"                    # bytes 0-3
version    = "1.0.0"
spec       = "docs/follow-agent-contract.md"
reference  = "ref/web.py"

[pipeline]
input   = "static media --uint8 grid + channels { image; audio and video reserved }"
process = "fit { sine(omega0 * (X W1 + b1)) -> sine(h1 W2 + b2) -> h2 W3 + b3, universal approximation }"
output  = "weights --fileformat .web { binary16 LE, root = sha256(bytes) }"
```

## Axioms

1. **One byte form.** A `.web` file starts with the magic `web1` and a 12-byte-plus-grid header, followed by `W1, b1, W2, b2, W3, b3` as row-major little-endian binary16 (contract §2). A parser accepts `b` only if `pack(unpack(b)) == b`. The file is at most 65,536 bytes.
2. **Root is identity.** `root = sha256(bytes)` in lowercase hex. A web is cited as `web1:<root>` (contract §4). A committed web is never rewritten; a better fit is a new file with a new root.
3. **Decode is normative.** The decoder widens weights to binary64, evaluates the network on the grid in `[-1, 1]` per axis, and rounds half to even into `0..255` (contract §3). The encoder is one valid fit, not part of the format.
4. **Refusals are codes.** Bad input fails with a stable code (`bad_magic`, `bad_header`, `truncated`, `trailing_bytes`, `bad_weight`, `too_large`, …), never partial output (contract §5).
5. **Weights as tensor.** A file is a fixed-length weight vector, so a deep net can read it directly. Any reader must allow for hidden-unit permutation (contract §7).
6. **Chain-ready, not chained.** The 32-byte root fits a `bytes32` and the bytes fit calldata ([`systems/commit-web.md`](systems/commit-web.md)). No contract is deployed (`web.json` `chain` is `null`).
7. **DNA form.** The `dna1` cipher writes a file as one `ACGT` sequence with no repeated base and reads it back byte for byte (contract §9). `pool1` splits it into indexed 117-nt oligos that survive loss, either strand and outvoted substitutions (§10).
