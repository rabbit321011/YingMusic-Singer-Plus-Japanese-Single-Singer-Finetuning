import sys
sys.path.insert(0, ".")

from src.YingMusicSinger.utils.f5_tts.g2p.g2p import PhonemeBpeTokenizer
from src.YingMusicSinger.utils.f5_tts.g2p.g2p.japanese import japanese_to_ipa
from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer

tokenizer = CNENTokenizer()
pbt = PhonemeBpeTokenizer()

test_texts = [
    "たぶん私じゃなくていいね",
    "君を見るたびに思い出す",
    "カッカッカッカッカッ",
    "世界で一番お姫様",
    "私の声を聞いてください",
]

print("=" * 60)
print("G2P Consistency Test")
print("=" * 60)

all_ok = True
for text in test_texts:
    phoneme = japanese_to_ipa(text, None)
    direct_tokens = pbt.phoneme2token(phoneme)
    if isinstance(direct_tokens, list) and len(direct_tokens) > 0 and isinstance(direct_tokens[0], list):
        direct_tokens = direct_tokens[0]

    infer_tokens = tokenizer.encode(text)

    if len(direct_tokens) == 0:
        print(f"  {'❌' if all_ok else '  '} '{text[:30]}': direct_tokens EMPTY")
        all_ok = False
        continue

    match = direct_tokens == [t - 1 for t in infer_tokens]
    status = "OK" if match else "MISMATCH"
    if not match:
        all_ok = False
    print(f"  {'✅' if match else '❌'} '{text[:30]}': {status}")
    if not match:
        print(f"      direct (0-based): {direct_tokens[:10]}{'...' if len(direct_tokens) > 10 else ''}")
        print(f"      encode-1 (0-based): {[t-1 for t in infer_tokens[:10]]}{'...' if len(infer_tokens) > 10 else ''}")

print("=" * 60)
if all_ok:
    print("All tests PASSED")
else:
    print("Some tests FAILED")

















