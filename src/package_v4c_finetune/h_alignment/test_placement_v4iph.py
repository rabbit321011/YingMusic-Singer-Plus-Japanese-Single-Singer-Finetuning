import unittest

from .placement import render_h_pul_placements as render_v4ph
from .placement_v4iph import (
    DEFAULT_FRAME_RATE,
    render_h_pul_placements as render_v4iph,
)


class V4IPHPlacementTest(unittest.TestCase):
    def setUp(self):
        self.phrases = [
            {
                "start": 10.25 / DEFAULT_FRAME_RATE,
                "end": 20 / DEFAULT_FRAME_RATE,
                "tokens": [11, 12],
            },
            {
                "start": 30.25 / DEFAULT_FRAME_RATE,
                "end": 40 / DEFAULT_FRAME_RATE,
                "tokens": [21, 22],
            },
        ]
        self.candidates = [
            {
                "status": "eligible",
                "tokens": [11, 12, 365],
                "relative_frames": [0, 4, 9],
            },
            {
                "status": "eligible",
                "tokens": [21, 22, 365],
                "relative_frames": [0, 5, 9],
            },
        ]

    def test_zero_reference_renders_full_timeline_as_b(self):
        result = render_v4iph(
            self.phrases,
            self.candidates,
            ref_len=0,
            total_frames=50,
        )
        audits = result["phone_pul"]["phrases"]
        self.assertEqual(audits[0]["first_token_frame"], 10)
        self.assertEqual(audits[1]["first_token_frame"], 30)
        self.assertEqual(result["phone_phrase_count"], 2)
        self.assertEqual(
            [token for token in result["phone_pul"]["text"] if token],
            [11, 12, 365, 21, 22, 365],
        )

    def test_positive_split_is_bit_identical_to_v4ph_renderer(self):
        arguments = (self.phrases, self.candidates)
        keywords = {"ref_len": 12, "total_frames": 50}
        self.assertEqual(
            render_v4iph(*arguments, **keywords),
            render_v4ph(*arguments, **keywords),
        )

    def test_empty_generation_region_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty generation"):
            render_v4iph(
                self.phrases,
                self.candidates,
                ref_len=50,
                total_frames=50,
            )


if __name__ == "__main__":
    unittest.main()
