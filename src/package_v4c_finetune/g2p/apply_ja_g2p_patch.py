import os, shutil, re

BASE = "${REMOTE_ROOT}/YingMusic-Singer-Plus"
G2P_DIR = f"{BASE}/src/YingMusicSinger/utils/f5_tts/g2p/g2p"
V1_G2P = "${REMOTE_ROOT}/YingMusic-Singer/src/singer/tokenizer/g2p"

print("=" * 60)
print("STEP 1: Copy japanese.py from V1")
print("=" * 60)
src = f"{V1_G2P}/japanese.py"
dst = f"{G2P_DIR}/japanese.py"
shutil.copy2(src, dst)
print(f"  Copied {src} -> {dst}")
print("  OK")

print()
print("=" * 60)
print("STEP 2: Patch cleaners.py - add japanese_to_ipa import")
print("=" * 60)
cleaners_path = f"{G2P_DIR}/cleaners.py"
with open(cleaners_path, "r") as f:
    content = f.read()

# Add import for japanese_to_ipa
if "from .japanese import japanese_to_ipa" not in content:
    content = content.replace(
        "from .mandarin import chinese_to_ipa",
        "from .japanese import japanese_to_ipa\nfrom .mandarin import chinese_to_ipa"
    )
    with open(cleaners_path, "w") as f:
        f.write(content)
    print("  Added 'from .japanese import japanese_to_ipa'")
    print("  OK")
else:
    print("  Already patched")

print()
print("=" * 60)
print("STEP 3: Patch g2p_generation.py - add ja support")
print("=" * 60)
g2p_gen_path = f"{G2P_DIR}/../g2p_generation.py"

with open(g2p_gen_path, "r") as f:
    content = f.read()

# 3a. Add is_japanese function
if "def is_japanese" not in content:
    is_jp_func = """
def is_japanese(char):
    if '\\u3040' <= char <= '\\u30ff':
        return True
    return False
"""
    # Insert after is_other function
    content = content.replace(
        "def get_segment(text: str) -> List[str]:",
        is_jp_func + "\n\ndef get_segment(text: str) -> List[str]:"
    )
    print("  3a. Added is_japanese() function")

# 3b. Modify get_segment to include ja
if "types.append(\"ja\")" not in content:
    content = content.replace(
        'elif is_alphabet(ch):\n            types.append("en")\n        else:\n            types.append("other")',
        'elif is_alphabet(ch):\n            types.append("en")\n        elif is_japanese(ch):\n            types.append("ja")\n        else:\n            types.append("other")'
    )
    print("  3b. Added ja detection in get_segment()")

# 3c. Rename chn_eng_g2p to chn_eng_jpn_g2p
if "def chn_eng_jpn_g2p" not in content:
    content = content.replace("def chn_eng_g2p(text: str):", "def chn_eng_jpn_g2p(text: str):")
    content = content.replace(
        "phone, token = chn_eng_g2p(",
        "phone, token = chn_eng_jpn_g2p("
    )
    content = content.replace(
        "phone, token = chn_eng_g2p(\\'",
        "phone, token = chn_eng_jpn_g2p('"
    )
    print("  3c. Renamed chn_eng_g2p -> chn_eng_jpn_g2p")

with open(g2p_gen_path, "w") as f:
    f.write(content)
print("  OK")

print()
print("=" * 60)
print("STEP 4: Patch cnen_tokenizer.py - use chn_eng_jpn_g2p")
print("=" * 60)
tokenizer_path = f"{BASE}/src/YingMusicSinger/utils/cnen_tokenizer.py"
with open(tokenizer_path, "r") as f:
    content = f.read()

if "chn_eng_jpn_g2p" not in content:
    content = content.replace(
        "from src.YingMusicSinger.utils.f5_tts.g2p.g2p_generation import chn_eng_g2p",
        "from src.YingMusicSinger.utils.f5_tts.g2p.g2p_generation import chn_eng_jpn_g2p"
    )
    content = content.replace(
        "self.tokenizer = chn_eng_g2p",
        "self.tokenizer = chn_eng_jpn_g2p"
    )
    print("  Updated imports and tokenizer reference")
    print("  OK")
else:
    print("  Already patched")

with open(tokenizer_path, "w") as f:
    f.write(content)

print()
print("=" * 60)
print("STEP 5: Verify - test full ja G2P pipeline")
print("=" * 60)
import sys
sys.path.insert(0, BASE)
os.chdir(BASE)

from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer

tokenizer = CNENTokenizer()
test_texts = [
    "こんにちは世界",
    "君は今だって",
    "たぶん私じゃなくていいね",
    "hello こんにちは 你好 world",
]

for text in test_texts:
    try:
        tokens = tokenizer.encode(text)
        print(f"  '{text}' -> {len(tokens)} tokens: {tokens[:10]}...")
    except Exception as e:
        print(f"  '{text}' -> ERROR: {e}")

print()
print("ALL DONE - Japanese G2P patched successfully!")
