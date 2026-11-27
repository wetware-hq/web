import hashlib
import json
import os
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import dna  # noqa: E402
import tensor  # noqa: E402
import web  # noqa: E402

PINS = json.loads(Path(ROOT, "web.json").read_text())
MAPS = PINS["tensors"]["maps"]


def path(p):
    return os.path.join(ROOT, p)


def mapped(m):
    return tensor.of_file(m["kind"], path(m["of"]), m.get("k", 3))


try:
    import tensorflow as tf
except ImportError:
    tf = None


class Samples(unittest.TestCase):
    def test_pinned_maps(self):  # T1
        self.assertEqual({m["kind"] for m in MAPS}, set(tensor.EXACT + tensor.VIEWS))
        for m in MAPS:
            blob = mapped(m)
            self.assertEqual(tensor.root(blob), m["root"], m["kind"])
            self.assertEqual(len(blob), m["bytes"])
            self.assertEqual(tensor.unpack(blob)[0], m["kind"])
            if "path" in m:
                self.assertEqual(Path(path(m["path"])).read_bytes(), blob)

    def test_exact_maps_invert(self):  # T1
        for m in MAPS:
            blob = mapped(m)
            if m["kind"] == "sequence":
                self.assertEqual(tensor.to_sequence(blob), dna.read_fasta(path(m["of"])))
            elif m["exact"]:
                self.assertEqual(tensor.inverse(blob), Path(path(m["of"])).read_bytes())
            else:
                with self.assertRaises(web.Refusal) as ctx:
                    tensor.inverse(blob)
                self.assertEqual(ctx.exception.code, "kind_mismatch")

    def test_weights_are_the_web_bytes(self):
        w = PINS["webs"][0]
        blob = Path(path(w["path"])).read_bytes()
        t = tensor.unpack(tensor.of_web(blob))[1]
        weights = b"".join(t[n].tobytes() for n in ("w1", "b1", "w2", "b2", "w3", "b3"))
        self.assertEqual(weights, blob[12 + 2 * len(w["header"]["grid"]) :])

    def test_glyph_counts_root(self):
        s = PINS["dna"][0]
        m = [m for m in MAPS if m["kind"] == "glyph"][0]
        counts = tensor.unpack(mapped(m))[1]["counts"]
        self.assertEqual(dna.counts_root(counts), s["glyph"]["counts"])

    def test_loop_chain_ends_at_head(self):
        s = PINS["dna"][0]
        m = [m for m in MAPS if m["kind"] == "loop"][0]
        t = tensor.unpack(mapped(m))[1]
        self.assertEqual(t["chain"][0].tobytes(), hashlib.sha256(b"loop1").digest())
        self.assertEqual(t["chain"][-1].tobytes().hex(), s["loop"]["head"])
        written = {i: tensor.VERDICTS[v] for i, v in s["loop"]["verdicts"].items()}
        self.assertEqual({str(i): int(v) for i, v in enumerate(t["verdict"]) if v}, written)

    def test_no_tensor_root_under_another_name(self):  # T3
        others = {w["root"] for w in PINS["webs"]} | {s["sequence"] for s in PINS["dna"]}
        self.assertFalse(others & {m["root"] for m in MAPS})


class Format(unittest.TestCase):
    def setUp(self):
        self.t = {"b": np.arange(6, dtype=np.int32).reshape(2, 3), "a": np.array(1.5, np.float16), "c": np.array([True, False])}
        self.blob = tensor.pack("probe", self.t)

    def refused(self, blob, code, kind=None):
        with self.assertRaises(web.Refusal) as ctx:
            tensor.unpack(blob, kind)
        self.assertEqual(ctx.exception.code, code)

    def test_canonical(self):  # T1
        kind, t = tensor.unpack(self.blob)
        self.assertEqual(list(t), ["a", "b", "c"])
        self.assertEqual(tensor.pack(kind, t), self.blob)
        self.assertEqual(tensor.pack(kind, {k: self.t[k] for k in reversed(sorted(self.t))}), self.blob)
        self.assertEqual(tensor.pack(kind, {"b": self.t["b"].astype(">i4"), "a": self.t["a"], "c": self.t["c"]}), self.blob)
        self.assertEqual(self.blob[:13], b"tmap1\x05probe\x03\x00")

    def test_refusals(self):  # T3
        self.refused(b"tmap2" + self.blob[5:], "bad_magic")
        self.refused(self.blob[:-1], "truncated")
        self.refused(self.blob + b"\0", "trailing_bytes")
        self.refused(self.blob, "kind_mismatch", "web")
        self.refused(self.blob.replace(b"\x01a", b"\x01d", 1), "bad_map")  # names out of order
        a = self.blob.index(b"\x01a") + 2
        self.refused(self.blob[:a] + b"\x07" + self.blob[a + 1 :], "bad_map")  # DT_STRING
        self.refused(self.blob[: a + 2] + struct.pack("<e", float("nan")) + self.blob[a + 4 :], "bad_map")
        self.refused(self.blob[:-1] + b"\x02", "bad_map")  # bool 2
        for kind, t in (("probe", {"x": np.array(-0.0)}), ("probe", {"x": np.array(["s"])}), ("probe", {"X": np.zeros(1)}),
                        ("probe", {}), ("w\u00e9b", {"x": np.zeros(1)}), ("probe", {"\u00e9": np.zeros(1)})):
            with self.assertRaises(web.Refusal) as ctx:
                tensor.pack(kind, t)
            self.assertEqual(ctx.exception.code, "bad_map")

    def test_pool_refusals(self):
        oligos = dna.pool(dna.read_fasta(path([m for m in MAPS if m["kind"] == "sequence"][0]["of"])), 2)
        t = tensor.unpack(tensor.of_pool(oligos))[1]
        self.assertEqual(t["file"], 2)  # read from the oligos, never relabelled
        for file, rows, code in ((0, t["oligos"], "bad_file"), (9, t["oligos"], "bad_file"), (2, t["oligos"][:0], "bad_pool")):
            with self.assertRaises(web.Refusal) as ctx:
                tensor.to_pool(tensor.pack("pool", {"file": np.uint8(file), "oligos": rows}))
            self.assertEqual(ctx.exception.code, code)
        with self.assertRaises(web.Refusal) as ctx:  # a float file id would write file=2.0
            tensor.to_pool(tensor.pack("pool", {"file": np.float32(2), "oligos": t["oligos"]}))
        self.assertEqual(ctx.exception.code, "bad_map")

    def test_sequence_refusals(self):
        for bad in ("", "ACGN", "acgt"):
            with self.assertRaises(web.Refusal):
                tensor.of_sequence(bad)
        blob = tensor.pack("sequence", {"bases": np.array([0, 4], np.uint8)})
        with self.assertRaises(web.Refusal) as ctx:
            tensor.to_sequence(blob)
        self.assertEqual(ctx.exception.code, "bad_base")


@unittest.skipIf(tf is None, "tensorflow is not installed; CI installs tensorflow-cpu")
class TensorFlow(unittest.TestCase):  # T2
    def test_dtype_numbers_are_tensorflow(self):
        for code, np_dtype in tensor.DTYPES.items():
            d = tf.dtypes.as_dtype(code)
            self.assertEqual(d.as_datatype_enum, code)
            self.assertEqual(np.dtype(d.as_numpy_dtype), np.dtype(np_dtype))

    def test_records_are_tensor_protos(self):
        for m in MAPS:
            blob = mapped(m)
            _, arrays = tensor.unpack(blob)
            tensors = tensor.to_tf(blob)
            for name, a in arrays.items():
                t = tensors[name]
                self.assertEqual(t.dtype.as_datatype_enum, tensor.CODES[a.dtype])
                self.assertEqual(tuple(t.shape), a.shape)
                self.assertTrue(np.array_equal(t.numpy(), a))
                self.assertEqual(tf.make_ndarray(tf.make_tensor_proto(t)).tobytes(), a.tobytes())
            self.assertEqual(tensor.from_tf(m["kind"], tensors), blob)

    def test_one_hot_is_a_view(self):
        m = [m for m in MAPS if m["kind"] == "sequence"][0]
        bases = tensor.to_tf(mapped(m))["bases"]
        hot = tf.one_hot(bases, 4, dtype=tf.uint8)
        self.assertEqual(hot.shape, (len(bases), 4))
        self.assertTrue(np.array_equal(tf.argmax(hot, axis=1, output_type=tf.int32).numpy(), bases.numpy()))


if __name__ == "__main__":
    unittest.main()
