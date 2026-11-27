# `.tmap`

A tensor map is the middle layer of the wetware stack: every object any layer stores (a web, a sequence, an oligo pool, a glyph, a loop, a build plan, a token, an order, a board) has exactly one form as a set of named tensors, so one deep net, one TensorFlow graph or one checker reads every layer with the same reader. A `.tmap` file is that set written in one canonical byte form, and `sha256` of its bytes is its **tensor root**. This page names the file form; [`docs/follow-agent-contract.md`](docs/follow-agent-contract.md) §14 defines it and governs on any conflict.

```toml
[tmap]
media_type = "application/vnd.tmap"    # vendor tree, not registered with IANA
extension  = ".tmap"
magic      = "tmap1"                   # bytes 0-4
version    = "1.0.0"
dtypes     = "tensorflow/core/framework/types.proto DataType"   # the dtype byte is TensorFlow's own enum
spec       = "docs/follow-agent-contract.md"
reference  = "ref/tensor.py"

[pipeline]
input   = "object --one layer's stored object { .web, .dna, pool1, cgr1, loop1; .tess, token, order, board in their repositories }"
process = "map { one kind, named tensors in ascending name order, each a TensorProto: DataType, shape, tensor_content }"
output  = "tensors --fileformat .tmap { root = sha256(bytes), cited tmap1:<root> }"
```

## Axioms

1. **One map per object.** Each kind of object has exactly one tensor map, defined by the repository that owns the object (contract §14). Two readers of one object build the same tensors and the same bytes.
2. **TensorFlow's types.** A tensor record is a `TensorProto` written flat: its `DataType` number, its shape, and its `tensor_content` (row-major, little-endian). `tf.io.parse_tensor` reads every record as written; nothing is converted on the way in.
3. **One byte form.** Names are lowercase and strictly ascending; floats are finite with no negative zero; a bool is 0 or 1; no byte follows the last record. A parser accepts `b` only if re-serializing gives `b`.
4. **Kind is part of the bytes.** A map carries its object's kind, and a reader that needs one kind refuses another (`kind_mismatch`), so a tensor root never stands in for a web root, a sequence root or another kind's tensor root (contract §13, S3).
5. **Exact or a view.** A map of a web, a sequence or a pool is exact: it inverts to the same object. A map of a glyph or a loop is a view, like the glyph and the loop themselves, and is never read back into data.
6. **Bases only where bases were.** Only a `sequence` or `pool` map holds bases, as indices `0-3` into `ACGT`. One-hot is a view of that tensor (`tf.one_hot(bases, 4)`), never a second stored form. A map of a token, a page, a glyph or a loop holds roots, counts and lengths only.
7. **Roots are uint8 [32].** A 32-byte root that crosses a boundary in tensor form is a `uint8` tensor of shape `[32]`, never a string or a pair of integers.
