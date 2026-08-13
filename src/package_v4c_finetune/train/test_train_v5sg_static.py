import pathlib
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
TRAIN = ROOT / "train_v5sg.py"
LAUNCHER = ROOT / "run_sft_v5sg_40k.sh"


class V5SgLRScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = TRAIN.read_text(encoding="utf-8")
        start = source.index("def v5sg_lr_factor")
        end = source.index("\n\ndef sha256_file", start)
        namespace = {"np": np}
        exec(source[start:end], namespace)
        cls.factor = staticmethod(namespace["v5sg_lr_factor"])

    def test_frozen_points(self):
        expected = {
            0: 0.0,
            2000: 7e-6,
            4000: 1.4e-5,
            16000: 1.2e-5,
            28000: 1e-5,
            34000: 5e-6,
            40000: 0.0,
        }
        for step, lr in expected.items():
            self.assertAlmostEqual(self.factor(step) * 1.4e-5, lr, places=12)

    def test_post_warmup_is_nonincreasing(self):
        values = [self.factor(step) for step in range(4000, 40001, 50)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))


class V5SgStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TRAIN.read_text(encoding="utf-8")
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")

    def test_checkpoint_and_data_contract(self):
        self.assertIn('V5SG_CHECKPOINT_SCHEMA = "v5sg_training_checkpoint_v1"', self.source)
        self.assertIn('"AudioSHA256"', self.source)
        self.assertIn('"AudioFrames"', self.source)
        self.assertIn("V5-Sg forbids loader cropping", self.source)
        self.assertNotIn("wav = wav[:, :max_samples]", self.source)

    def test_launcher_is_official_fresh_and_uses_v5_pool(self):
        self.assertIn("V5P_20260808/data/frozen", self.launcher)
        self.assertIn("--max_duration 60.1", self.launcher)
        self.assertIn("--warmup_steps 4000", self.launcher)
        self.assertIn("--first_decay_end 28000", self.launcher)
        self.assertIn("--max_steps 40000", self.launcher)
        self.assertNotIn("--warmstart_checkpoint", self.launcher)

    def test_some_and_cpu_ema_are_frozen(self):
        self.assertIn("continuous SOME", self.source)
        self.assertIn("smoothMelody_MIDIFuzzDisturb", self.source)
        self.assertIn("--ema_device cpu", self.launcher)
        self.assertIn("include_online_model=False", self.source)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", self.launcher)

    def test_step_zero_gate_is_supported(self):
        self.assertIn("0 <= args.stop_after_step <= args.max_steps", self.source)
        self.assertIn(
            "args.max_steps if args.stop_after_step is None else args.stop_after_step",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
