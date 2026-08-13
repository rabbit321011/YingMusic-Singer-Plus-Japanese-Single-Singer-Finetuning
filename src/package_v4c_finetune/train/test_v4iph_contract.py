import unittest

import torch

from v4iph_contract import (
    REFERENCE_MODE_ALL_B,
    REFERENCE_MODE_SPLIT,
    compute_flow_losses,
    resolve_ref_len,
)


class V4IPHContractTest(unittest.TestCase):
    def test_all_b_forces_zero_reference(self):
        self.assertEqual(resolve_ref_len(REFERENCE_MODE_ALL_B, 17, 50), 0)

    def test_split_reference_is_unchanged(self):
        self.assertEqual(resolve_ref_len(REFERENCE_MODE_SPLIT, 17, 50), 17)

    def test_all_b_loss_skips_empty_flow_a(self):
        prediction = torch.zeros(1, 4, 2)
        target = torch.ones_like(prediction)
        flow_a, flow_b, flow = compute_flow_losses(
            prediction,
            target,
            ref_len=0,
            flow_b_weight=2.0,
            phase="joint",
            reference_mode=REFERENCE_MODE_ALL_B,
        )
        self.assertEqual(flow_a.item(), 0.0)
        self.assertEqual(flow_b.item(), 1.0)
        self.assertEqual(flow.item(), 2.0)
        self.assertTrue(torch.isfinite(flow))

    def test_split_loss_matches_v4ph_formula(self):
        prediction = torch.zeros(1, 4, 1)
        target = torch.tensor([[[1.0], [1.0], [2.0], [2.0]]])
        flow_a, flow_b, flow = compute_flow_losses(
            prediction,
            target,
            ref_len=2,
            flow_b_weight=2.0,
            phase="joint",
            reference_mode=REFERENCE_MODE_SPLIT,
        )
        self.assertEqual(flow_a.item(), 1.0)
        self.assertEqual(flow_b.item(), 4.0)
        self.assertEqual(flow.item(), 9.0)

    def test_all_b_rejects_nonzero_reference(self):
        value = torch.zeros(1, 4, 1)
        with self.assertRaisesRegex(ValueError, "ref_len=0"):
            compute_flow_losses(
                value,
                value,
                ref_len=1,
                flow_b_weight=2.0,
                phase="joint",
                reference_mode=REFERENCE_MODE_ALL_B,
            )


if __name__ == "__main__":
    unittest.main()
