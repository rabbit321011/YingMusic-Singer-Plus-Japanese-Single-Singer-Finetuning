import difflib
import re

import pyopenjtalk

from .placement import canonical_sha256


SMALL_KATAKANA = set("ァィゥェォャュョヮヵヶ")
PAUSE_MARKS = set("、。，！？・,.!?;:：；「」『』（）()[]【】〈〉《》…'\" ")
IPA_PUNCTUATION = {",", ".", "?", "!", ":", ";", "'", "…"}
SILENCE_PHONES = {"sp", "ap", "pau", "sil", "<sp>", "<ap>"}
VOWELS = set("aiueo")


def katakana_to_hiragana(text):
    output = []
    for char in text:
        code = ord(char)
        output.append(chr(code - 0x60) if 0x30A1 <= code <= 0x30F6 else char)
    return "".join(output)


def pronunciation_katakana(text):
    return pyopenjtalk.g2p(text, kana=True).strip()


def split_pronunciation(pronunciation):
    units = []
    for char in pronunciation:
        if char.isspace() or char in PAUSE_MARKS:
            if units and units[-1]["kind"] != "pause":
                units.append({"kind": "pause", "kana": char})
            continue
        if char in SMALL_KATAKANA:
            if not units or units[-1]["kind"] != "mora":
                raise ValueError(f"orphan small kana: {char!r} in {pronunciation!r}")
            units[-1]["kana"] += char
            continue
        if char == "ー":
            units.append({"kind": "long", "kana": char})
            continue
        if char == "ッ":
            units.append({"kind": "sokuon", "kana": char})
            continue
        if char == "ン":
            units.append({"kind": "mora", "kana": char})
            continue
        if "ァ" <= char <= "ヿ":
            units.append({"kind": "mora", "kana": char})
            continue
        raise ValueError(f"unsupported pronunciation character: {char!r}")
    while units and units[-1]["kind"] == "pause":
        units.pop()
    return units


def _phones_for_unit(unit, previous_vowel):
    if unit["kind"] == "pause":
        return ["SP"], previous_vowel
    if unit["kind"] == "sokuon":
        return ["cl"], previous_vowel
    if unit["kind"] == "long":
        if previous_vowel is None:
            raise ValueError("long mark has no preceding vowel")
        return [previous_vowel], previous_vowel

    phones = pyopenjtalk.g2p(unit["kana"], kana=False).split()
    if not phones or any(phone.lower() in SILENCE_PHONES for phone in phones):
        raise ValueError(f"invalid mora phones: {unit['kana']!r} -> {phones!r}")
    vowel = next(
        (phone.lower() for phone in reversed(phones) if phone.lower() in VOWELS),
        previous_vowel,
    )
    return phones, vowel


def _tokenize_ipa(text, ipa_converter, vocab):
    ipa_text = ipa_converter(text)
    ipa_phones = ipa_text.split("|") if ipa_text else []
    retained = [(phone, int(vocab[phone]) + 1) for phone in ipa_phones if phone in vocab]
    return {
        "ipa_text": ipa_text,
        "phones": [phone for phone, _ in retained],
        "tokens": [token for _, token in retained],
    }


def build_frontend(phrases, ipa_converter, vocab, sofa_phone_to_ipa=None):
    punctuation_token_ids = {
        int(vocab[phone]) + 1 for phone in IPA_PUNCTUATION if phone in vocab
    }
    phonemes = ["SP"]
    words = []
    phoneme_to_word = [-1]
    phone_events = []
    phrase_plans = []

    for phrase_index, phrase in enumerate(phrases):
        source_kana = phrase.get("kana") or phrase["text"]
        pronunciation = pronunciation_katakana(source_kana)
        units = split_pronunciation(pronunciation)
        previous_vowel = None
        phrase_expected_phones = []
        phrase_phone_events = []

        for mora_index, unit in enumerate(units):
            unit_phones, previous_vowel = _phones_for_unit(unit, previous_vowel)
            if unit["kind"] == "pause":
                if phonemes[-1] != "SP":
                    phonemes.append("SP")
                    phoneme_to_word.append(-1)
                continue

            word_index = len(words)
            words.append(unit["kana"])
            for phone in unit_phones:
                normalized_phone = phone.lower() if phone in {"I", "U"} else phone
                event = {
                    "global_phone_index": len(phone_events),
                    "phrase_index": phrase_index,
                    "phrase_phone_index": len(phrase_phone_events),
                    "mora_index": mora_index,
                    "word_index": word_index,
                    "kana": katakana_to_hiragana(unit["kana"]),
                    "katakana": unit["kana"],
                    "mora_kind": unit["kind"],
                    "sofa_phone": normalized_phone,
                }
                phone_events.append(event)
                phrase_phone_events.append(event)
                phrase_expected_phones.append(normalized_phone)
                phonemes.append(normalized_phone)
                phoneme_to_word.append(word_index)

        if phrase_index < len(phrases) - 1 and phonemes[-1] != "SP":
            phonemes.append("SP")
            phoneme_to_word.append(-1)

        normalized = _tokenize_ipa(pronunciation, ipa_converter, vocab)
        normalized_lexical = [
            (phone, token)
            for phone, token in zip(normalized["phones"], normalized["tokens"])
            if phone not in IPA_PUNCTUATION
        ]
        locked_tokens = [int(token) for token in phrase["tokens"]]
        locked_lexical_tokens = [
            token for token in locked_tokens if token not in punctuation_token_ids
        ]
        normalized_lexical_tokens = [token for _, token in normalized_lexical]
        direct_ipa_phones = []
        direct_ipa_tokens = []
        if sofa_phone_to_ipa is not None:
            for phone in phrase_expected_phones:
                ipa_phone = sofa_phone_to_ipa.get(phone)
                if ipa_phone is None or ipa_phone not in vocab:
                    direct_ipa_phones = []
                    direct_ipa_tokens = []
                    break
                direct_ipa_phones.append(ipa_phone)
                direct_ipa_tokens.append(int(vocab[ipa_phone]) + 1)

        fallback_reason = None
        if len(normalized_lexical) != len(phrase_phone_events):
            fallback_reason = "phone_ipa_length_mismatch"
        elif sofa_phone_to_ipa is not None and (
            direct_ipa_phones != [phone for phone, _ in normalized_lexical]
            or direct_ipa_tokens != normalized_lexical_tokens
        ):
            fallback_reason = "direct_phone_token_mismatch"
        elif locked_lexical_tokens != normalized_lexical_tokens:
            fallback_reason = "locked_token_mismatch"

        if fallback_reason is None:
            for event, (ipa_phone, token_id) in zip(
                phrase_phone_events, normalized_lexical
            ):
                event["ipa_phone"] = ipa_phone
                event["token_id"] = token_id

        token_events = []
        lexical_index = 0
        for token_index, token_id in enumerate(locked_tokens):
            if token_id in punctuation_token_ids:
                token_events.append(
                    {
                        "token_index": token_index,
                        "token_id": token_id,
                        "kind": "punctuation",
                    }
                )
                continue
            phone_event = (
                phrase_phone_events[lexical_index]
                if lexical_index < len(phrase_phone_events)
                else None
            )
            token_events.append(
                {
                    "token_index": token_index,
                    "token_id": token_id,
                    "kind": "phone",
                    "global_phone_index": (
                        phone_event["global_phone_index"] if phone_event else None
                    ),
                }
            )
            lexical_index += 1

        phrase_plans.append(
            {
                "phrase_index": phrase_index,
                "source_text": phrase.get("text", ""),
                "source_kana": source_kana,
                "normalized_katakana": pronunciation,
                "normalized_hiragana": katakana_to_hiragana(pronunciation),
                "expected_sofa_phones": phrase_expected_phones,
                "normalized_ipa_phones": normalized["phones"],
                "normalized_tokens": normalized["tokens"],
                "direct_ipa_phones": direct_ipa_phones,
                "direct_ipa_tokens": direct_ipa_tokens,
                "locked_tokens": locked_tokens,
                "locked_token_sha256": canonical_sha256(locked_tokens),
                "token_match": (
                    "exact"
                    if locked_tokens == normalized["tokens"]
                    else "punctuation_only"
                    if fallback_reason is None
                    else "mismatch"
                ),
                "status": "eligible" if fallback_reason is None else "fallback",
                "fallback_reason": fallback_reason,
                "token_events": token_events,
            }
        )

    if phonemes[-1] != "SP":
        phonemes.append("SP")
        phoneme_to_word.append(-1)
    if len(phonemes) != len(phoneme_to_word):
        raise AssertionError("phoneme_to_word length mismatch")

    return {
        "phonemes": phonemes,
        "words": words,
        "phoneme_to_word": phoneme_to_word,
        "phone_events": phone_events,
        "phrases": phrase_plans,
        "punctuation_token_ids": sorted(punctuation_token_ids),
    }


def match_phone_intervals(frontend, predicted_phone_rows):
    expected = [event["sofa_phone"] for event in frontend["phone_events"]]
    predicted_rows = [
        row
        for row in predicted_phone_rows
        if str(row["phone"]).lower() not in SILENCE_PHONES
    ]
    predicted = [str(row["phone"]) for row in predicted_rows]
    matcher = difflib.SequenceMatcher(a=expected, b=predicted, autojunk=False)
    mapping = [None] * len(expected)
    operations = []
    only_missing_cl = True

    for tag, expected_start, expected_end, predicted_start, predicted_end in matcher.get_opcodes():
        expected_slice = expected[expected_start:expected_end]
        predicted_slice = predicted[predicted_start:predicted_end]
        operations.append(
            {
                "op": tag,
                "expected_start": expected_start,
                "expected": expected_slice,
                "predicted_start": predicted_start,
                "predicted": predicted_slice,
            }
        )
        if tag == "equal":
            for offset in range(expected_end - expected_start):
                mapping[expected_start + offset] = predicted_rows[predicted_start + offset]
        elif tag == "delete" and expected_slice and all(
            phone == "cl" for phone in expected_slice
        ):
            continue
        else:
            only_missing_cl = False

    if not only_missing_cl:
        mapping = [None] * len(expected)
        status = "non_cl_phone_edit"
    elif any(row is None for row in mapping):
        status = "missing_cl"
    else:
        status = "exact"

    return {
        "status": status,
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "missing_cl_count": sum(
            expected[index] == "cl" and row is None
            for index, row in enumerate(mapping)
        ),
        "mapping": mapping,
        "operations": operations,
    }
