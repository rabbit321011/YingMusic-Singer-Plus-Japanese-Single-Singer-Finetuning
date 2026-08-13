import unittest

from .placement import (
    DEFAULT_FRAME_RATE,
    DEFAULT_PUL_TOKEN_ID,
    render_h_pul_placements,
    render_paired_placements,
    solve_monotonic_frames,
)


class PlacementSolverTest(unittest.TestCase):
    def test_first_frame_is_fixed_and_order_is_strict(self):
        result = solve_monotonic_frames(
            [10, 10, 10, 12],
            first_frame=10,
            lower_frame=10,
            upper_frame=20,
        )
        self.assertEqual(result["frames"][0], 10)
        self.assertTrue(
            all(a < b for a, b in zip(result["frames"], result["frames"][1:]))
        )
        self.assertEqual(result["max_abs_shift"], 2)

    def test_solver_is_deterministic(self):
        arguments = ([5, 5, 6, 6, 8], 5, 5, 15)
        self.assertEqual(
            solve_monotonic_frames(*arguments),
            solve_monotonic_frames(*arguments),
        )

    def test_not_enough_frames_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not enough frames"):
            solve_monotonic_frames([1, 1, 1], 1, 1, 2)

    def test_nonpriority_event_does_not_move_phone(self):
        result = solve_monotonic_frames(
            [10, 20, 20],
            first_frame=10,
            lower_frame=10,
            upper_frame=25,
            priority_mask=[True, False, True],
        )
        self.assertEqual(result["frames"][2], 20)
        self.assertEqual(result["max_abs_priority_shift"], 0)


class PairedPlacementTest(unittest.TestCase):
    def setUp(self):
        start = 10 / DEFAULT_FRAME_RATE
        end = 20 / DEFAULT_FRAME_RATE
        self.phrases = [{"start": start, "end": end, "tokens": [11, 12]}]
        self.candidates = [
            {
                "status": "eligible",
                "tokens": [11, 12, 365],
                "frames": [10, 14, 19],
            }
        ]

    def test_only_later_frames_change(self):
        result = render_paired_placements(
            self.phrases,
            self.candidates,
            ref_len=30,
            total_frames=50,
        )
        self.assertEqual(result["control"]["phrases"][0]["first_token_frame"], 10)
        self.assertEqual(result["phone"]["phrases"][0]["first_token_frame"], 10)
        self.assertEqual(result["phone_phrase_count"], 1)
        self.assertEqual(
            [token for token in result["control"]["text"] if token],
            [token for token in result["phone"]["text"] if token],
        )

    def test_ab_boundary_causes_sentence_fallback(self):
        result = render_paired_placements(
            self.phrases,
            self.candidates,
            ref_len=12,
            total_frames=50,
        )
        audit = result["phone"]["phrases"][0]
        self.assertEqual(audit["placement_mode"], "sentence")
        self.assertEqual(audit["fallback_reason"], "runtime_first_anchor_mismatch")
        self.assertEqual(audit["first_token_frame"], 12)

    def test_relative_candidate_moves_with_runtime_anchor(self):
        self.candidates[0]["relative_frames"] = [0, 4, 9]
        result = render_paired_placements(
            self.phrases,
            self.candidates,
            ref_len=12,
            total_frames=50,
        )
        audit = result["phone"]["phrases"][0]
        self.assertEqual(audit["placement_mode"], "phone")
        self.assertEqual(audit["first_token_frame"], 12)

    def test_offline_failure_causes_sentence_fallback(self):
        self.candidates[0] = {
            "status": "fallback",
            "fallback_reason": "missing_cl",
            "tokens": [11, 12, 365],
            "frames": [10, 14, 19],
        }
        result = render_paired_placements(
            self.phrases,
            self.candidates,
            ref_len=30,
            total_frames=50,
        )
        self.assertEqual(result["phone_phrase_count"], 0)
        self.assertEqual(
            result["phone"]["phrases"][0]["fallback_reason"],
            "missing_cl",
        )

    def test_legacy_collision_forces_whole_sample_control(self):
        phrases = [
            {"start": 10 / DEFAULT_FRAME_RATE, "end": 20 / DEFAULT_FRAME_RATE, "tokens": [11, 12]},
            {"start": 11 / DEFAULT_FRAME_RATE, "end": 25 / DEFAULT_FRAME_RATE, "tokens": [21, 22]},
        ]
        candidates = [
            {"status": "eligible", "tokens": [11, 12, 365], "frames": [10, 14, 18]},
            {"status": "eligible", "tokens": [21, 22, 365], "frames": [11, 20, 24]},
        ]
        result = render_paired_placements(
            phrases,
            candidates,
            ref_len=30,
            total_frames=50,
        )
        self.assertTrue(result["sample_control_anomaly"])
        self.assertEqual(result["phone_phrase_count"], 0)
        self.assertEqual(result["control"]["text"], result["phone"]["text"])


class HPulPlacementTest(unittest.TestCase):
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

    def render(self, ref_len=45, total_frames=50):
        return render_h_pul_placements(
            self.phrases,
            self.candidates,
            ref_len=ref_len,
            total_frames=total_frames,
        )

    def test_phone_sep_is_immediately_before_next_anchor(self):
        result = self.render()
        rendered = result["phone_pul"]
        self.assertEqual(result["phone_phrase_count"], 2)
        self.assertEqual(result["pul_phrase_count"], 0)
        self.assertEqual(rendered["phrases"][0]["sep_frame"], 29)
        self.assertEqual(rendered["phrases"][1]["sep_frame"], 49)
        self.assertEqual(rendered["text"][29], 365)
        self.assertEqual(rendered["text"][30], 21)
        self.assertEqual(rendered["text"][49], 365)

    def test_offline_fallback_fills_every_frame_with_pul(self):
        self.candidates[0] = {
            "status": "fallback",
            "fallback_reason": "missing_cl",
            "tokens": [11, 12, 365],
            "relative_frames": [],
        }
        result = self.render()
        rendered = result["phone_pul"]
        self.assertEqual(result["phone_phrase_count"], 1)
        self.assertEqual(result["pul_phrase_count"], 1)
        self.assertEqual(rendered["text"][10:12], [11, 12])
        self.assertEqual(
            rendered["text"][12:29],
            [DEFAULT_PUL_TOKEN_ID] * 17,
        )
        self.assertEqual(rendered["text"][29], 365)
        self.assertEqual(
            rendered["phrases"][0]["fallback_reason"],
            "missing_cl",
        )

    def test_last_fallback_fills_to_final_frame(self):
        self.candidates[1] = {
            "status": "fallback",
            "fallback_reason": "phone_shift_exceeds_limit",
            "tokens": [21, 22, 365],
            "relative_frames": [],
        }
        result = self.render()
        rendered = result["phone_pul"]
        self.assertEqual(rendered["text"][30:32], [21, 22])
        self.assertEqual(rendered["text"][32:49], [366] * 17)
        self.assertEqual(rendered["text"][49], 365)

    def test_runtime_anchor_is_preserved_across_ab_split(self):
        result = self.render(ref_len=12)
        audits = result["phone_pul"]["phrases"]
        self.assertEqual(audits[0]["first_token_frame"], 12)
        self.assertEqual(audits[0]["control_first_token_frame"], 12)
        self.assertEqual(audits[0]["sep_frame"], 29)

    def test_insufficient_pul_capacity_forces_exact_control(self):
        self.phrases[1]["start"] = 13.25 / DEFAULT_FRAME_RATE
        self.candidates[0] = {
            "status": "fallback",
            "fallback_reason": "missing_cl",
            "tokens": [11, 12, 365],
            "relative_frames": [],
        }
        result = self.render()
        self.assertTrue(result["sample_structural_fallback"])
        self.assertEqual(result["pul_phrase_count"], 0)
        self.assertEqual(
            result["phone_pul"]["text"],
            result["control"]["text"],
        )

    def test_pul_renderer_is_deterministic(self):
        self.candidates[0] = {
            "status": "fallback",
            "fallback_reason": "missing_cl",
            "tokens": [11, 12, 365],
            "relative_frames": [],
        }
        self.assertEqual(self.render(), self.render())


if __name__ == "__main__":
    unittest.main()
