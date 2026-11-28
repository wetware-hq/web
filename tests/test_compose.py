import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import compose  # noqa: E402
import deep  # noqa: E402
import tensor  # noqa: E402
import web  # noqa: E402

PINS = json.loads(Path(ROOT, "web.json").read_text())
KL = PINS["klines"]


def path(p):
    return Path(ROOT, p)


def blob(p):
    return path(p).read_bytes()


def refuses(test, code, fn, *args, **kw):
    with test.assertRaises(web.Refusal) as e:
        fn(*args, **kw)
    test.assertEqual(e.exception.code, code)


USES = [([blob(u["input"])], compose.luma_target(web.read_png(path(u["media"])))) for u in KL["uses"]]
LUMA = blob("samples/luma.web")
CURVE = blob("samples/astronaut-64.curve.web")


class Samples(unittest.TestCase):
    def test_pinned_klines(self):  # L1
        for k in KL["kline"]:
            text = path(k["path"]).read_text("ascii")
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), k["sha256"])
            stages = [blob(p) for p in k["stages"]]
            self.assertEqual(compose.verify(text, stages, USES), k["head"])
            self.assertEqual(compose.recalled(text), k["recalled"])
            self.assertEqual([us for _, _, us in compose.parse(text)[2]], k["utility"])

    def test_targets_pinned(self):
        for u, (_, t) in zip(KL["uses"], USES):
            self.assertEqual(hashlib.sha256(t.tobytes()).hexdigest(), u["target"])

    def test_stages_pinned(self):
        for s in KL["stages"]:
            self.assertEqual(web.root(blob(s["path"])), s["root"])

    def test_tensor_maps(self):
        for m in PINS["tensors"]["maps"]:
            if m["kind"] == "kline":
                b = tensor.of_file("kline", path(m["of"]))
                self.assertEqual((tensor.root(b), len(b)), (m["root"], m["bytes"]))
                refuses(self, "kind_mismatch", tensor.inverse, b)


class Activation(unittest.TestCase):
    def test_kline_is_composition(self):  # L2
        inputs, _ = USES[0]
        got = deep.compose(inputs + [LUMA, CURVE])[-1]
        self.assertTrue(np.array_equal(got, compose.last(inputs + [LUMA, CURVE])))

    def test_head_names_stages_only(self):  # L2
        a = compose.record([LUMA], [compose.use(*USES[0][:1], [LUMA], USES[0][1])])
        b = compose.record([LUMA], [compose.use(i, [LUMA], t) for i, t in USES])
        self.assertEqual(compose.check_head(a), compose.check_head(b))
        self.assertNotEqual(compose.head([LUMA]), compose.head([LUMA, CURVE]))

    def test_one_use_is_not_recalled(self):
        a = compose.record([LUMA], [compose.use(USES[0][0], [LUMA], USES[0][1])])
        self.assertFalse(compose.recalled(a))

    def test_same_target_twice_is_not_recalled(self):
        u = compose.use(USES[0][0], [LUMA], USES[0][1])
        self.assertFalse(compose.recalled(compose.record([LUMA], [u, u])))

    def test_utility(self):
        t = USES[0][1]
        self.assertIsNone(compose.utility(np.zeros((64, 64, 3), np.uint8), t))
        self.assertEqual(compose.utility(t, t), float("inf"))
        self.assertTrue(compose.rises(1.0, None) and not compose.rises(None, None) and not compose.rises(1.0, 1.0))


class Grow(unittest.TestCase):
    def test_grow_keeps_only_rises(self):  # L3
        inputs, t = USES[0]
        added, trace = compose.grow(inputs, t, library=[LUMA], fit=False)
        self.assertEqual([web.root(s) for s in added], [web.root(LUMA)])
        self.assertEqual(trace, [("start", None), (f"stage:{web.root(LUMA)}", 32.47)])
        us = [u for _, u in trace]
        self.assertTrue(all(compose.rises(a, b) for b, a in zip(us, us[1:])))

    def test_grow_fits_a_curve(self):  # L3
        inputs, t = USES[0]
        added, trace = compose.grow(inputs, t, library=[LUMA], max_steps=2)
        self.assertEqual([s for s, _ in trace], ["start", f"stage:{web.root(LUMA)}", "fit"])
        self.assertGreater(trace[-1][1], trace[-2][1])
        self.assertEqual(web.unpack(added[-1])["grid"], (deep.LEVELS,))

    def test_grow_without_a_fitting_stage_stops(self):
        inputs, t = USES[0]
        self.assertEqual(compose.grow(inputs, t, library=[CURVE], fit=False), ([], [("start", None)]))

    def test_cli_recalls_only_recalled(self):  # L4
        import contextlib
        import io

        u = KL["uses"][1]
        for k, want in [(KL["kline"][0], True), (KL["kline"][1], False)]:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = compose.main(["grow", str(path(u["media"])), str(path(u["input"])), "--no-fit", "--recall", str(path(k["path"])), *map(str, map(path, k["stages"]))])
            self.assertEqual(rc, 0)
            self.assertEqual(out.getvalue().startswith(f"start none\nkline:{k['head']}"), want)

    def test_curve_table_is_least_squares(self):
        lv = np.array([[0], [0], [5], [5], [5]], np.uint8)
        tg = np.array([[10], [20], [30], [30], [60]], np.uint8)
        t = compose.curve_table(lv, tg)
        self.assertEqual((t[0], t[5], t[2]), (15, 40, 25))
        refuses(self, "shape_mismatch", compose.curve_table, np.zeros((2, 3), np.uint8), np.zeros((2, 3), np.uint8))


class Edges(unittest.TestCase):
    def test_exact_match_round_trips(self):
        inputs, t = USES[0]
        exact = compose.luma_target(web.read_png(path(KL["uses"][0]["media"])))
        text = compose.record([LUMA], [("0" * 64, "1" * 64, [None, float("inf")])])
        self.assertEqual(compose.parse(text)[2][0][2], [None, float("inf")])
        self.assertTrue(np.array_equal(exact, t))

    def test_fit_needs_a_comparable_interface(self):
        inputs, t = USES[0]
        self.assertEqual(compose.grow(inputs, t[:32], library=[LUMA]), ([], [("start", None)]))

    def test_luma_is_integer_half_even(self):
        px = np.array([[[1, 0, 0]]], np.uint8)  # 0.299 -> 0
        self.assertEqual(compose.luma_target(px)[0, 0, 0], 0)
        # 299*2 + 587*0 + 114*... : find exact .5 cases by construction: 500/1000 with q even/odd
        self.assertEqual(compose.luma_target(np.array([[[0, 0, 0]]], np.uint8))[0, 0, 0], 0)
        self.assertEqual(compose.luma_target(np.array([[[255, 255, 255]]], np.uint8))[0, 0, 0], 255)


class Refusals(unittest.TestCase):  # L5
    def setUp(self):
        self.text = path(KL["kline"][0]["path"]).read_text("ascii")

    def test_grammar(self):
        for bad in [self.text.replace("\n", "\r\n"), self.text[:-1], self.text.replace("stages=1", "stages=2"),
                    self.text.replace("uses=2", "uses=3"), self.text.replace("%stage 0", "%stage 1"),
                    self.text.replace("utility=none,32.47", "utility=none"), self.text.replace("out=1", "out=3"), "%kline1\n"]:
            refuses(self, "bad_kline", compose.parse, bad)

    def test_head(self):
        h = KL["kline"][0]["head"]
        refuses(self, "bad_head", compose.check_head, self.text.replace(h, "0" * 64))

    def test_replay(self):
        refuses(self, "mismatch", compose.verify, self.text.replace("32.47", "32.48"), [LUMA], USES)
        refuses(self, "root_mismatch", compose.verify, self.text, [CURVE], USES)
        refuses(self, "bad_stage", compose.verify, self.text, [LUMA, CURVE], USES)

    def test_stage_interfaces(self):
        refuses(self, "bad_stage", compose.record, [CURVE, LUMA], [])
        refuses(self, "bad_stage", compose.record, [blob("samples/astronaut-64.web")], [])
        refuses(self, "bad_stage", compose.record, [], [])


if __name__ == "__main__":
    unittest.main()
