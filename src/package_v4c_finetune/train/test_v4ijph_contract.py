import unittest

import numpy as np
import torch

from v4ijph_contract import (
    InsStyleAdapter,
    V4IJPH_ADAPTER_MID_LR,
    V4IJPH_ADAPTER_MID_STEP,
    V4IJPH_ADAPTER_PEAK_LR,
    V4IJPH_ADAPTER_PEAK_STEP,
    V4IJPH_MAX_STEPS,
    V4IJPH_TRANSITION_STEP,
    adapter_lr_for_global_update,
    adapter_updates_for_global_step,
    assert_tree_equal,
    assert_zero_output_initialization,
    build_style_cond,
    state_dict_sha256,
)


class V4IjPhContractTest(unittest.TestCase):
    def test_zero_initialized_adapter_is_exactly_null(self):
        torch.manual_seed(7)
        adapter = InsStyleAdapter()
        assert_zero_output_initialization(adapter)
        cond, style = build_style_cond(
            adapter, torch.randn(2, 768), 13, drop_ins=False
        )
        self.assertEqual(tuple(style.shape), (2, 64))
        self.assertEqual(tuple(cond.shape), (2, 13, 64))
        self.assertEqual(torch.count_nonzero(cond).item(), 0)

    def test_dropout_bypasses_a_learned_adapter_with_exact_zero(self):
        adapter = InsStyleAdapter()
        with torch.no_grad():
            adapter.output_projection.bias.fill_(1.0)
        cond, style = build_style_cond(
            adapter, torch.randn(1, 768), 4, drop_ins=True
        )
        self.assertEqual(torch.count_nonzero(style).item(), 0)
        self.assertEqual(torch.count_nonzero(cond).item(), 0)

    def test_style_is_broadcast_over_the_complete_timeline(self):
        adapter = InsStyleAdapter()
        with torch.no_grad():
            adapter.output_projection.bias.copy_(torch.arange(64).float())
        cond, style = build_style_cond(
            adapter, torch.ones(1, 768), 3, drop_ins=False
        )
        self.assertTrue(torch.equal(cond[:, 0], style))
        self.assertTrue(torch.equal(cond[:, 1], style))
        self.assertTrue(torch.equal(cond[:, 2], style))

    def test_global_schedule_boundaries(self):
        self.assertEqual(
            adapter_lr_for_global_update(V4IJPH_TRANSITION_STEP), 0.0
        )
        self.assertGreater(
            adapter_lr_for_global_update(V4IJPH_TRANSITION_STEP + 1), 0.0
        )
        self.assertAlmostEqual(
            adapter_lr_for_global_update(V4IJPH_ADAPTER_PEAK_STEP),
            V4IJPH_ADAPTER_PEAK_LR,
        )
        self.assertAlmostEqual(
            adapter_lr_for_global_update(V4IJPH_ADAPTER_MID_STEP),
            V4IJPH_ADAPTER_MID_LR,
        )
        self.assertEqual(adapter_lr_for_global_update(V4IJPH_MAX_STEPS), 0.0)

    def test_adapter_update_count_is_relative_to_transition(self):
        self.assertEqual(adapter_updates_for_global_step(8000), 0)
        self.assertEqual(adapter_updates_for_global_step(8001), 1)
        self.assertEqual(adapter_updates_for_global_step(30000), 22000)

    def test_tree_comparison_is_device_agnostic_and_detects_changes(self):
        assert_tree_equal({"x": torch.ones(2)}, {"x": torch.ones(2)}, "tree")
        with self.assertRaisesRegex(AssertionError, "tree.x tensor differs"):
            assert_tree_equal({"x": torch.ones(2)}, {"x": torch.zeros(2)}, "tree")

    def test_tree_comparison_handles_numpy_rng_state_exactly(self):
        left = {"rng": ("MT19937", np.arange(624, dtype=np.uint32), 17, 0, 0.0)}
        right = {"rng": ("MT19937", np.arange(624, dtype=np.uint32), 17, 0, 0.0)}
        assert_tree_equal(left, right, "rank_states")
        right["rng"][1][42] += 1
        with self.assertRaisesRegex(AssertionError, "NumPy array differs"):
            assert_tree_equal(left, right, "rank_states")

    def test_state_hash_is_stable_and_sensitive(self):
        torch.manual_seed(1)
        adapter = InsStyleAdapter()
        first = state_dict_sha256(adapter)
        self.assertEqual(first, state_dict_sha256(adapter))
        with torch.no_grad():
            adapter.output_projection.bias[0] = 1
        self.assertNotEqual(first, state_dict_sha256(adapter))

if __name__ == "__main__":
    unittest.main()
