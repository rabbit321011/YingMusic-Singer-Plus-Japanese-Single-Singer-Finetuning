import unittest

from package_v4c_finetune.h_alignment.frontend import (
    build_frontend,
    match_phone_intervals,
)


VOCAB = {
    "g": 0,
    "a": 1,
    "q": 2,
    "k": 3,
    "o": 4,
    ",": 5,
    "!": 6,
}

PHONE_TO_IPA = {
    "g": "g",
    "a": "a",
    "cl": "q",
    "k": "k",
    "o": "o",
}


def fake_ipa(text):
    return "g|a|q|k|o|o|,"


class FrontendTest(unittest.TestCase):
    def test_locked_punctuation_is_preserved_but_lexical_tokens_map(self):
        phrase = {
            "text": "学校!",
            "kana": "がっこう、",
            "start": 0.0,
            "end": 1.0,
            "tokens": [1, 2, 3, 4, 5, 5, 7],
        }
        frontend = build_frontend([phrase], fake_ipa, VOCAB, PHONE_TO_IPA)
        plan = frontend["phrases"][0]
        self.assertEqual(plan["status"], "eligible")
        self.assertEqual(plan["token_match"], "punctuation_only")
        self.assertEqual(
            [event["sofa_phone"] for event in frontend["phone_events"]],
            ["g", "a", "cl", "k", "o", "o"],
        )
        self.assertEqual(plan["token_events"][-1]["kind"], "punctuation")

    def test_only_missing_cl_is_accepted_as_diagnostic_mapping(self):
        phrase = {
            "text": "学校!",
            "kana": "がっこう、",
            "start": 0.0,
            "end": 1.0,
            "tokens": [1, 2, 3, 4, 5, 5, 7],
        }
        frontend = build_frontend([phrase], fake_ipa, VOCAB, PHONE_TO_IPA)
        rows = [
            {"phone": phone, "start": index * 0.1, "end": (index + 1) * 0.1}
            for index, phone in enumerate(["g", "a", "k", "o", "o"])
        ]
        match = match_phone_intervals(frontend, rows)
        self.assertEqual(match["status"], "missing_cl")
        self.assertEqual(match["missing_cl_count"], 1)
        self.assertIsNone(match["mapping"][2])


if __name__ == "__main__":
    unittest.main()
