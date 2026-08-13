# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io, re, os, sys, time, argparse, pdb, json
from io import StringIO
from typing import Optional
import numpy as np
import traceback
import pyopenjtalk
from pykakasi import kakasi

punctuation = [",", ".", "!", "?", ":", ";", "'", "…"]

jp_xphone2ipa = [
    " a a",
    " i i",
    " u ɯ",
    " e e",
    " o o",
    " a: aː",
    " i: iː",
    " u: ɯː",
    " e: eː",
    " o: oː",
    " k k",
    " s s",
    " t t",
    " n n",
    " h ç",
    " f ɸ",
    " m m",
    " y j",
    " r ɾ",
    " w ɰᵝ",
    " N ɴ",
    " g g",
    " j d ʑ",
    " z z",
    " d d",
    " b b",
    " p p",
    " q q",
    " v v",
    " : :",
    " by b j",
    " ch t ɕ",
    " dy d e j",
    " ty t e j",
    " gy g j",
    " gw g ɯ",
    " hy ç j",
    " ky k j",
    " kw k ɯ",
    " my m j",
    " ny n j",
    " py p j",
    " ry ɾ j",
    " sh ɕ",
    " ts t s ɯ",
]

_mora_list_minimum: list[tuple[str, Optional[str], str]] = [
    ("ヴォ", "v", "o"),
    ("ヴェ", "v", "e"),
    ("ヴィ", "v", "i"),
    ("ヴァ", "v", "a"),
    ("ヴ", "v", "u"),
    ("ン", None, "N"),
    ("ワ", "w", "a"),
    ("ロ", "r", "o"),
    ("レ", "r", "e"),
    ("ル", "r", "u"),
    ("リョ", "ry", "o"),
    ("リュ", "ry", "u"),
    ("リャ", "ry", "a"),
    ("リェ", "ry", "e"),
    ("リ", "r", "i"),
    ("ラ", "r", "a"),
    ("ヨ", "y", "o"),
    ("ユ", "y", "u"),
    ("ヤ", "y", "a"),
    ("モ", "m", "o"),
    ("メ", "m", "e"),
    ("ム", "m", "u"),
    ("ミョ", "my", "o"),
    ("ミュ", "my", "u"),
    ("ミャ", "my", "a"),
    ("ミェ", "my", "e"),
    ("ミ", "m", "i"),
    ("マ", "m", "a"),
    ("ポ", "p", "o"),
    ("ボ", "b", "o"),
    ("ホ", "h", "o"),
    ("ペ", "p", "e"),
    ("ベ", "b", "e"),
    ("ヘ", "h", "e"),
    ("プ", "p", "u"),
    ("ブ", "b", "u"),
    ("フォ", "f", "o"),
    ("フェ", "f", "e"),
    ("フィ", "f", "i"),
    ("ファ", "f", "a"),
    ("フ", "f", "u"),
    ("ピョ", "py", "o"),
    ("ピュ", "py", "u"),
    ("ピャ", "py", "a"),
    ("ピェ", "py", "e"),
    ("ピ", "p", "i"),
    ("ビョ", "by", "o"),
    ("ビュ", "by", "u"),
    ("ビャ", "by", "a"),
    ("ビェ", "by", "e"),
    ("ビ", "b", "i"),
    ("ヒョ", "hy", "o"),
    ("ヒュ", "hy", "u"),
    ("ヒャ", "hy", "a"),
    ("ヒェ", "hy", "e"),
    ("ヒ", "h", "i"),
    ("パ", "p", "a"),
    ("バ", "b", "a"),
    ("ハ", "h", "a"),
    ("ノ", "n", "o"),
    ("ネ", "n", "e"),
    ("ヌ", "n", "u"),
    ("ニョ", "ny", "o"),
    ("ニュ", "ny", "u"),
    ("ニャ", "ny", "a"),
    ("ニェ", "ny", "e"),
    ("ニ", "n", "i"),
    ("ナ", "n", "a"),
    ("ドゥ", "d", "u"),
    ("ド", "d", "o"),
    ("トゥ", "t", "u"),
    ("ト", "t", "o"),
    ("デョ", "dy", "o"),
    ("デュ", "dy", "u"),
    ("デャ", "dy", "a"),
    ("ディ", "d", "i"),
    ("デ", "d", "e"),
    ("テョ", "ty", "o"),
    ("テュ", "ty", "u"),
    ("テャ", "ty", "a"),
    ("ティ", "t", "i"),
    ("テ", "t", "e"),
    ("ツォ", "ts", "o"),
    ("ツェ", "ts", "e"),
    ("ツィ", "ts", "i"),
    ("ツァ", "ts", "a"),
    ("ツ", "ts", "u"),
    ("ッ", None, "q"),
    ("チョ", "ch", "o"),
    ("チュ", "ch", "u"),
    ("チャ", "ch", "a"),
    ("チェ", "ch", "e"),
    ("チ", "ch", "i"),
    ("ダ", "d", "a"),
    ("タ", "t", "a"),
    ("ゾ", "z", "o"),
    ("ソ", "s", "o"),
    ("ゼ", "z", "e"),
    ("セ", "s", "e"),
    ("ズィ", "z", "i"),
    ("ズ", "z", "u"),
    ("スィ", "s", "i"),
    ("ス", "s", "u"),
    ("ジョ", "j", "o"),
    ("ジュ", "j", "u"),
    ("ジャ", "j", "a"),
    ("ジェ", "j", "e"),
    ("ジ", "j", "i"),
    ("ショ", "sh", "o"),
    ("シュ", "sh", "u"),
    ("シャ", "sh", "a"),
    ("シェ", "sh", "e"),
    ("シ", "sh", "i"),
    ("ザ", "z", "a"),
    ("サ", "s", "a"),
    ("ゴ", "g", "o"),
    ("コ", "k", "o"),
    ("ゲ", "g", "e"),
    ("ケ", "k", "e"),
    ("グヮ", "gw", "a"),
    ("グ", "g", "u"),
    ("クヮ", "kw", "a"),
    ("ク", "k", "u"),
    ("ギョ", "gy", "o"),
    ("ギュ", "gy", "u"),
    ("ギャ", "gy", "a"),
    ("ギェ", "gy", "e"),
    ("ギ", "g", "i"),
    ("キョ", "ky", "o"),
    ("キュ", "ky", "u"),
    ("キャ", "ky", "a"),
    ("キェ", "ky", "e"),
    ("キ", "k", "i"),
    ("ガ", "g", "a"),
    ("カ", "k", "a"),
    ("オ", None, "o"),
    ("エ", None, "e"),
    ("ウォ", "w", "o"),
    ("ウェ", "w", "e"),
    ("ウィ", "w", "i"),
    ("ウ", None, "u"),
    ("イェ", "y", "e"),
    ("イ", None, "i"),
    ("ア", None, "a"),
]

_mora_list_additional: list[tuple[str, Optional[str], str]] = [
    ("ヴョ", "by", "o"),
    ("ヴュ", "by", "u"),
    ("ヴャ", "by", "a"),
    ("ヲ", None, "o"),
    ("ヱ", None, "e"),
    ("ヰ", None, "i"),
    ("ヮ", "w", "a"),
    ("ョ", "y", "o"),
    ("ュ", "y", "u"),
    ("ヅ", "z", "u"),
    ("ヂ", "j", "i"),
    ("ヶ", "k", "e"),
    ("ャ", "y", "a"),
    ("ォ", None, "o"),
    ("ェ", None, "e"),
    ("ゥ", None, "u"),
    ("ィ", None, "i"),
    ("ァ", None, "a"),
]

mora_phonemes_to_mora_kata: dict[str, str] = {
    (consonant or "") + vowel: kana for [kana, consonant, vowel] in _mora_list_minimum
}

mora_kata_to_mora_phonemes: dict[str, tuple[Optional[str], str]] = {
    kana: (consonant, vowel)
    for [kana, consonant, vowel] in _mora_list_minimum + _mora_list_additional
}


rep_map = {
    "：": ":",
    "；": ";",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "⋯": "…",
    "···": "…",
    "・・・": "…",
    "·": ",",
    "・": ",",
    "•": ",",
    "、": ",",
    "$": ".",
    "‘": "'",
    "’": "'",
}


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def replace_punctuation(text: str) -> str:
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
        + r"\u0041-\u005A\u0061-\u007A"
        + r"\uFF21-\uFF3A\uFF41-\uFF5A"
        + r"\u0370-\u03FF\u1F00-\u1FFF"
        + "".join(punctuation) + r"]+",
        "",
        replaced_text,
    )
    return replaced_text


def fix_phone_tone(phone_tone_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    tone_values = set(tone for _, tone in phone_tone_list)
    if len(tone_values) == 1:
        assert tone_values == {0}, tone_values
        return phone_tone_list
    elif len(tone_values) == 2:
        if tone_values == {0, 1}:
            return phone_tone_list
        elif tone_values == {-1, 0}:
            return [
                (letter, 0 if tone == -1 else 1) for letter, tone in phone_tone_list
            ]
        else:
            raise ValueError(f"Unexpected tone values: {tone_values}")
    else:
        raise ValueError(f"Unexpected tone values: {tone_values}")


def fix_phone_tone_wplen(phone_tone_list, word_phone_length_list):
    phones = []
    tones = []
    w_p_len = []
    p_len = len(phone_tone_list)
    idx = 0
    w_idx = 0
    while idx < p_len:
        offset = 0
        if phone_tone_list[idx] == "▁":
            w_p_len.append(w_idx + 1)

        curr_w_p_len = word_phone_length_list[w_idx]
        for i in range(curr_w_p_len):
            p, t = phone_tone_list[idx]
            if p == ":" and len(phones) > 0:
                if phones[-1][-1] != ":":
                    phones[-1] += ":"
                    offset -= 1
            else:
                phones.append(p)
                tones.append(str(t))
            idx += 1
            if idx >= p_len:
                break
        w_p_len.append(curr_w_p_len + offset)
        w_idx += 1
    return phones, tones, w_p_len


def g2phone_tone_wo_punct(prosodies) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    current_phrase: list[tuple[str, int]] = []
    current_tone = 0
    last_accent = ""
    for i, letter in enumerate(prosodies):
        if letter == "^":
            assert i == 0, "Unexpected ^"
        elif letter in ("$", "?", "_", "#"):
            result.extend(fix_phone_tone(current_phrase))
            if letter in ("$", "?"):
                assert i == len(prosodies) - 1, f"Unexpected {letter}"
            current_phrase = []
            current_tone = 0
            last_accent = ""
        elif letter == "[":
            if last_accent != letter:
                current_tone = current_tone + 1
            last_accent = letter
        elif letter == "]":
            if last_accent != letter:
                current_tone = current_tone - 1
            last_accent = letter
        else:
            if letter == "cl":
                letter = "q"
            current_phrase.append((letter, current_tone))
    return result


def handle_long(sep_phonemes: list[list[str]]) -> list[list[str]]:
    for i in range(len(sep_phonemes)):
        if sep_phonemes[i][0] == "ー":
            sep_phonemes[i][0] = ":"
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = ":"
    return sep_phonemes


def handle_long_word(sep_phonemes: list[list[str]]) -> list[list[str]]:
    res = []
    for i in range(len(sep_phonemes)):
        if sep_phonemes[i][0] == "ー":
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
        res.append(sep_phonemes[i])
        res.append("▁")
    return res


def align_tones(
    phones_with_punct: list[str], phone_tone_list: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    tone_index = 0
    for phone in phones_with_punct:
        if tone_index >= len(phone_tone_list):
            result.append((phone, 0))
        elif phone == phone_tone_list[tone_index][0]:
            result.append((phone, phone_tone_list[tone_index][1]))
            tone_index += 1
        elif phone in punctuation or phone == "▁":
            result.append((phone, 0))
        else:
            print(f"phones: {phones_with_punct}")
            print(f"phone_tone_list: {phone_tone_list}")
            print(f"result: {result}")
            print(f"tone_index: {tone_index}")
            print(f"phone: {phone}")
            raise ValueError(f"Unexpected phone: {phone}")
    return result


def kata2phoneme_list(text: str) -> list[str]:
    if text in punctuation:
        return [text]
    if re.fullmatch(r"[\u30A0-\u30FF]+", text) is None:
        raise ValueError(f"Input must be katakana only: {text}")
    sorted_keys = sorted(mora_kata_to_mora_phonemes.keys(), key=len, reverse=True)
    pattern = "|".join(map(re.escape, sorted_keys))

    def mora2phonemes(mora: str) -> str:
        cosonant, vowel = mora_kata_to_mora_phonemes[mora]
        if cosonant is None:
            return f" {vowel}"
        return f" {cosonant} {vowel}"

    spaced_phonemes = re.sub(pattern, lambda m: mora2phonemes(m.group()), text)

    long_pattern = r"(\w)(ー*)"
    long_replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))
    spaced_phonemes = re.sub(long_pattern, long_replacement, spaced_phonemes)
    return spaced_phonemes.strip().split(" ")


def frontend2phoneme(labels, drop_unvoiced_vowels=False):
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        if p3 == "sil":
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


class JapanesePhoneConverter(object):
    def __init__(self, lexicon_path=None, ipa_dict_path=None):
        self.ipa_dict = {}
        for curr_line in jp_xphone2ipa:
            k, v = curr_line.strip().split(" ", 1)
            self.ipa_dict[k] = re.sub("\s", "", v)
        self.japan_JH2K = kakasi()
        self.table = {ord(f): ord(t) for f, t in zip("67", "_¯")}

    def text2sep_kata(self, parsed) -> tuple[list[str], list[str]]:
        sep_text: list[str] = []
        sep_kata: list[str] = []
        fix_parsed = []
        i = 0
        while i <= len(parsed) - 1:
            yomi = parsed[i]["pron"]
            tmp_parsed = parsed[i]
            if i != len(parsed) - 1 and parsed[i + 1]["string"] in [
                "々",
                "ゝ",
                "ヽ",
                "ゞ",
                "ヾ",
                "゛",
            ]:
                word = parsed[i]["string"] + parsed[i + 1]["string"]
                i += 1
            else:
                word = parsed[i]["string"]
            word, yomi = replace_punctuation(word), yomi.replace("’", "")
            assert yomi != "", f"Empty yomi: {word}"
            if yomi == "、":
                if word not in (
                    ".",
                    ",",
                    "!",
                    "'",
                    "-",
                    "?",
                    ":",
                    ";",
                    "…",
                    "",
                ):
                    print(
                        "Cannot read:{}, yomi:{}, new_word:{};".format(
                            word, yomi, self.japan_JH2K.convert(word)[0]["kana"]
                        )
                    )
                    word = self.japan_JH2K.convert(word)[0]["kana"]
                    tmp_parsed["pron"] = word
                yomi = word
            elif yomi == "？":
                assert word == "?", f"yomi `？` comes from: {word}"
                yomi = "?"
            if word == "":
                i += 1
                continue
            sep_text.append(word)
            sep_kata.append(yomi)
            fix_parsed.append(tmp_parsed)
            i += 1
        return sep_text, sep_kata, fix_parsed

    def getSentencePhone(self, sentence, blank_mode=True, phoneme_mode=False):
        words = []
        words_phone_len = []
        short_char_flag = False
        output_duration_flag = []
        output_before_sil_flag = []
        normed_text = []
        sentence = sentence.strip().strip("'")
        sentence = re.sub(r"\s+", "", sentence)
        output_res = []
        failed_words = []
        last_long_pause = 4
        last_word = None
        frontend_text = pyopenjtalk.run_frontend(sentence)
        try:
            frontend_text = pyopenjtalk.estimate_accent(frontend_text)
        except:
            pass
        sep_text, sep_kata, frontend_text = self.text2sep_kata(frontend_text)
        sep_phonemes = handle_long_word([kata2phoneme_list(i) for i in sep_kata])

        pron_text = [x["pron"].strip().replace("'", "") for x in frontend_text]
        prosodys = pyopenjtalk.make_label(frontend_text)
        prosodys = frontend2phoneme(prosodys, drop_unvoiced_vowels=True)
        normed_text = [x["string"].strip() for x in frontend_text]
        phone_tone_list_wo_punct = g2phone_tone_wo_punct(prosodys)

        phone_w_punct: list[str] = []
        w_p_len = []
        for i in sep_phonemes:
            phone_w_punct += i
            w_p_len.append(len(i))
        phone_w_punct = phone_w_punct[:-1]
        phone_tone_list = align_tones(phone_w_punct, phone_tone_list_wo_punct)

        jp_item = {}
        jp_p = ""
        jp_t = ""
        for p, t in phone_tone_list:
            if p in self.ipa_dict:
                curr_p = self.ipa_dict[p]
                jp_p += curr_p
                jp_t += str(t + 6) * len(curr_p)
            elif p in punctuation:
                jp_p += p
                jp_t += "0"
            elif p == "▁":
                jp_p += p
                jp_t += " "
            else:
                print(p, t)
            jp_p += "|"
            jp_t += "0"
        jp_p = jp_p.replace("▁", " ")
        jp_t = jp_t.translate(self.table)
        jp_l = ""
        for t in jp_t:
            if t == " ":
                jp_l += " "
            else:
                jp_l += "2"
        assert len(jp_p) == len(jp_t) and len(jp_p) == len(jp_l)

        jp_item["jp_p"] = jp_p.replace("| |", "|").rstrip("|")
        jp_item["jp_t"] = jp_t
        jp_item["jp_l"] = jp_l
        jp_item["jp_normed_text"] = " ".join(normed_text)
        jp_item["jp_pron_text"] = " ".join(pron_text)
        return jp_item


jpc = JapanesePhoneConverter()


def japanese_to_ipa(text, text_tokenizer):
    if type(text) == str:
        return jpc.getSentencePhone(text)["jp_p"]
    else:
        result_ph = []
        for t in text:
            result_ph.append(jpc.getSentencePhone(t)["jp_p"])
        return result_ph
