import json

with open("${REMOTE_ROOT}/final_sum_large/train_singnet.json") as f:
    d = json.load(f)

r0 = d[0]
print("Keys:", list(r0.keys()))
for k,v in r0.items():
    if isinstance(v, str):
        print(f"  {k}: '{v[:120]}'")
    else:
        print(f"  {k}: {v}")

import os
audio_dir = "${REMOTE_ROOT}/final_sum_large/audio"
wavs = os.listdir(audio_dir)
print(f"\nAudio dir: {audio_dir}")
print(f"  Files: {len(wavs)}")
print(f"  Examples: {wavs[:5]}")

# Check if Path is a full path or just stem
path0 = r0["Path"]
if os.path.exists(path0):
    print(f"\nPath IS valid: {path0}")
elif os.path.exists(os.path.join(audio_dir, os.path.basename(path0))):
    print(f"\nPath needs basename: {os.path.basename(path0)}")
else:
    # Try to find matching file
    stem = os.path.splitext(os.path.basename(path0))[0]
    for w in wavs:
        if stem in w:
            print(f"  Found match: {w}")
            break
    else:
        print(f"  No match found for stem: {stem}")

