import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ref"))
import tensor  # noqa: E402
import tune  # noqa: E402
import web  # noqa: E402

PINS = json.loads(Path(ROOT, "web.json").read_text())
TUNE = PINS["tune"]
BASE = Path(ROOT, PINS["webs"][0]["path"]).read_bytes()
MEDIA = web.read_png(os.path.join(ROOT, PINS["webs"][0]["media"]["path"]))
SMALL = MEDIA[::4, ::4]


def refused(case, code, fn, *args, **kw):
    with case.assertRaises(web.Refusal) as ctx:
        fn(*args, **kw)
    case.assertEqual(ctx.exception.code, code)


class Steps(unittest.TestCase):
    def test_u1_default_fit_is_the_reference_encoder(self):
        tr = tune.fit(tune.Trainer.fresh(SMALL.shape[:-1], 3, 16, 30, 3), SMALL, 150)
        self.assertEqual(tr.save_weights(), web.encode(SMALL, hidden=16, steps=150, seed=3))

    def test_u2_resume_with_optimizer_is_one_run(self):
        whole = tune.fit(tune.Trainer.fresh(SMALL.shape[:-1], 3, 16, 30, 1), SMALL, 80)
        half = tune.fit(tune.Trainer.fresh(SMALL.shape[:-1], 3, 16, 30, 1), SMALL, 40)
        state = half.save_state()
        self.assertEqual(tensor.pack(*tensor.unpack(state)), state)
        resumed = tune.fit(tune.Trainer.from_state(state), SMALL, 40)
        self.assertEqual(resumed.step, 80)
        self.assertEqual(resumed.save_weights(), whole.save_weights())
        cold = tune.Trainer.from_state(state, with_optimizer=False)
        self.assertEqual(cold.step, 0)
        self.assertEqual(cold.save_weights(), half.save_weights())

    def test_u2_resume_keeps_the_plan(self):
        p = tune.AdamParams(3e-3, beta2=0.95, eps=1e-12, weight_decay=1e-4, grad_clip_norm=1.0)
        whole = tune.fit(tune.Trainer.fresh(SMALL.shape[:-1], 3, 16, 30, 1), SMALL, 60, p, "linear", 64, 5)
        half = tune.fit(tune.Trainer.fresh(SMALL.shape[:-1], 3, 16, 30, 1), SMALL, 30, p, "linear", 64, 5, total=60)
        resumed = tune.fit(tune.Trainer.from_state(half.save_state()), SMALL)
        self.assertEqual((resumed.step, resumed.plan), (60, whole.plan))
        self.assertEqual(resumed.save_weights(), whole.save_weights())

    def test_accumulated_gradients_are_the_full_batch(self):
        x, t = tune.targets(SMALL)
        a = tune.Trainer.fresh(SMALL.shape[:-1], 3, 16, 30, 2)
        b = tune.Trainer.fresh(SMALL.shape[:-1], 3, 16, 30, 2)
        a.forward_backward(x, t)
        b.forward_backward(x[:128], t[:128])
        b.forward_backward(x[128:], t[128:])
        for k in a.grads:  # two half-batch means sum to twice the full-batch mean
            np.testing.assert_allclose(b.grads[k], 2 * a.grads[k], rtol=1e-4, atol=1e-6)

    def test_clip_and_decay(self):
        x, t = tune.targets(SMALL)
        a = tune.Trainer.fresh(SMALL.shape[:-1], 3, 16, 30, 2)
        a.forward_backward(x, t)
        a.optim_step(tune.AdamParams(grad_clip_norm=1e-9, weight_decay=0.1))
        self.assertEqual(a.step, 1)
        refused(self, "no_gradients", a.optim_step)

    def test_warm_start_keeps_the_web(self):
        self.assertEqual(tune.Trainer.from_web(BASE).save_weights(), BASE)


class Adapters(unittest.TestCase):
    def test_u3_zero_adapter_is_the_base(self):
        tr = tune.Trainer.lora(BASE, 4)
        self.assertEqual(tr.save_weights(), BASE)
        self.assertEqual(tune.merge(BASE, tr.save_adapter()), BASE)

    def test_u4_pinned_adapter_merges_to_its_web(self):
        blob = Path(ROOT, TUNE["adapter"]["path"]).read_bytes()
        self.assertEqual(tensor.root(blob), TUNE["adapter"]["root"])
        self.assertEqual(len(blob), TUNE["adapter"]["bytes"])
        self.assertEqual(tensor.unpack(blob)[1]["base"].tobytes().hex(), PINS["webs"][0]["root"])
        merged = tune.merge(BASE, blob)
        self.assertEqual(web.root(merged), TUNE["merged"]["root"])
        self.assertEqual(len(merged), TUNE["merged"]["bytes"])
        self.assertAlmostEqual(web.psnr(web.decode(merged), MEDIA), TUNE["merged"]["psnr"], delta=0.01)
        self.assertGreater(TUNE["merged"]["psnr"], PINS["webs"][0]["psnr"])

    def test_u5_adapter_refuses_another_base(self):
        blob = Path(ROOT, TUNE["adapter"]["path"]).read_bytes()
        other = tune.Trainer.from_web(BASE)
        other.train["b3"] = other.train["b3"] + np.float32(0.01)
        refused(self, "root_mismatch", tune.merge, other.save_weights(), blob)
        refused(self, "kind_mismatch", tune.merge, BASE, tensor.of_web(BASE))
        t = tensor.unpack(blob)[1]
        refused(self, "bad_map", tune.merge, BASE, tensor.pack("adapter", {k: v for k, v in t.items() if k != "w2_up"}))
        refused(self, "bad_map", tune.merge, BASE, tensor.pack("adapter", t | {"w2_up": t["w2_up"].astype(np.float32)}))

    def test_adapter_state_resumes(self):
        tr = tune.fit(tune.Trainer.lora(BASE, 2), MEDIA, 3)
        again = tune.Trainer.from_state(tr.save_state())
        self.assertEqual((again.rank, again.step, again.base), (2, 3, BASE))
        self.assertEqual(again.save_adapter(), tr.save_adapter())
        refused(self, "kind_mismatch", tune.Trainer.from_state, tr.save_adapter())

    def test_state_refuses_bad_moments(self):
        t = tensor.unpack(tune.fit(tune.Trainer.fresh(SMALL.shape[:-1], 3, 16, 30, 1), SMALL, 2).save_state())[1]
        for k, v in (("m_w2", t["m_w2"][:, :3]), ("v_b1", t["v_b1"].astype(np.float64)), ("w3", t["w3"].T.copy())):
            refused(self, "bad_map", tune.Trainer.from_state, tensor.pack("state", t | {k: v}))


class Heuristics(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(tune.param_count(64, 2, 3), 4547)
        self.assertEqual(tune.param_count(64, 2, 3, rank=8), 2 * 66 + 8 * 128 + 3 * 67)

    def test_rules(self):
        self.assertEqual(tune.linear_decay(1.0, 1, 4), 1.0)
        self.assertEqual(tune.linear_decay(1.0, 4, 4), 0.25)
        self.assertEqual(tune.batch_lr(1e-3, 1024, 4096), 5e-4)
        self.assertEqual(tune.lora_lr(1e-4), 1e-3)


class Cli(unittest.TestCase):
    def test_resume_takes_the_saved_hyperparameters(self):
        png = os.path.join(ROOT, PINS["webs"][0]["media"]["path"])
        with tempfile.TemporaryDirectory() as d:
            o = lambda n: os.path.join(d, n)  # noqa: E731
            flags = ["--steps", "3", "--lr", "3e-3", "--beta2", "0.95", "--eps", "1e-12", "--schedule", "linear", "--hidden", "8"]
            self.assertEqual(tune.main(["fit", png, o("whole.web")] + flags), 0)
            self.assertEqual(tune.main(["fit", png, o("half.web"), "--state", o("half.tmap"), "--total", "3"] + flags[2:] + ["--steps", "2"]), 0)
            self.assertEqual(tune.main(["fit", png, o("rest.web"), "--from-state", o("half.tmap")]), 0)
            self.assertEqual(Path(o("rest.web")).read_bytes(), Path(o("whole.web")).read_bytes())

    def test_fit_refuses_a_taken_state_path_before_writing(self):
        png = os.path.join(ROOT, PINS["webs"][0]["media"]["path"])
        with tempfile.TemporaryDirectory() as d:
            taken = os.path.join(d, "taken.tmap")
            Path(taken).write_bytes(b"")
            self.assertEqual(tune.main(["fit", png, os.path.join(d, "new.web"), "--state", taken, "--steps", "1"]), 1)
            self.assertFalse(os.path.exists(os.path.join(d, "new.web")))

    def test_merge_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "m.web")
            args = ["merge", os.path.join(ROOT, PINS["webs"][0]["path"]), os.path.join(ROOT, TUNE["adapter"]["path"]), out]
            self.assertEqual(tune.main(args), 0)
            self.assertEqual(web.root(Path(out).read_bytes()), TUNE["merged"]["root"])
            self.assertEqual(tune.main(args), 1)


if __name__ == "__main__":
    unittest.main()
