import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


class V5PLRScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = pathlib.Path(__file__).with_name("train_v5p.py")
        source = path.read_text(encoding="utf-8")
        start = source.index("def v5p_lr_factor")
        end = source.index("\n\ndef load_state_dict_checked", start)
        namespace = {"np": np}
        exec(source[start:end], namespace)
        cls.factor = staticmethod(namespace["v5p_lr_factor"])

    def test_frozen_points(self):
        expected = {
            0: 0.0,
            2000: 1.4e-5,
            12000: 1.27092098e-5,
            24000: 1.02290879e-5,
            28000: 1e-5,
            34000: 5e-6,
            40000: 0.0,
        }
        for step, lr in expected.items():
            self.assertAlmostEqual(self.factor(step) * 1.4e-5, lr, places=12)

    def test_nonincreasing_after_warmup(self):
        values = [self.factor(step) for step in range(2000, 40001, 100)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))


class V5PgLRScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = pathlib.Path(__file__).with_name("train_v5p.py")
        source = path.read_text(encoding="utf-8")
        start = source.index("def v5pg_lr_factor")
        end = source.index("\n\ndef load_state_dict_checked", start)
        namespace = {"np": np}
        exec(source[start:end], namespace)
        cls.factor = staticmethod(namespace["v5pg_lr_factor"])

    def test_frozen_points(self):
        expected = {
            0: 0.0,
            125: 2.5e-6,
            250: 5e-6,
            6250: 5e-6,
            8125: 2.5e-6,
            10000: 0.0,
        }
        for step, lr in expected.items():
            self.assertAlmostEqual(self.factor(step) * 5e-6, lr, places=12)

    def test_hold_and_decay_are_monotonic(self):
        hold = [self.factor(step) for step in range(250, 6251, 250)]
        self.assertTrue(all(value == 1.0 for value in hold))
        decay = [self.factor(step) for step in range(6250, 10001, 50)]
        self.assertTrue(all(a >= b for a, b in zip(decay, decay[1:])))


class V5Pg20HLR07ScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = pathlib.Path(__file__).with_name("train_v5p.py")
        source = path.read_text(encoding="utf-8")
        start = source.index("def v5p_lr_factor")
        end = source.index("\n\ndef v5pg_lr_factor", start)
        namespace = {"np": np}
        exec(source[start:end], namespace)
        cls.factor = staticmethod(namespace["v5p_lr_factor"])

    def test_frozen_points(self):
        expected = {
            0: 0.0,
            500: 4.9e-6,
            1000: 9.8e-6,
            7500: 8.4e-6,
            14000: 7e-6,
            17000: 3.5e-6,
            20000: 0.0,
        }
        for step, lr in expected.items():
            factor = self.factor(
                step,
                peak_lr=9.8e-6,
                mid_lr=7e-6,
                warmup_steps=1000,
                first_decay_end=14000,
                max_steps=20000,
            )
            self.assertAlmostEqual(factor * 9.8e-6, lr, places=12)

    def test_post_warmup_is_nonincreasing(self):
        values = [
            self.factor(
                step,
                peak_lr=9.8e-6,
                mid_lr=7e-6,
                warmup_steps=1000,
                first_decay_end=14000,
                max_steps=20000,
            )
            for step in range(1000, 20001, 50)
        ]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))


if __name__ == "__main__":
    unittest.main()
