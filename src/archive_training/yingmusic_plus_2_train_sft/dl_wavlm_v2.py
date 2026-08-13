import os, sys

# Try ModelScope first (Chinese CDN, fast)
print("=== Trying ModelScope ===")
try:
    from modelscope import snapshot_download
    d = snapshot_download("iic/speech_wavlm_large_fixed", cache_dir="${REMOTE_ROOT}/pretrained_models/")
    print(f"ModelScope OK: {d}")
    sys.exit(0)
except Exception as e:
    print(f"ModelScope failed: {e}")

# Try HF mirror with retry
print()
print("=== Trying HF mirror (retry) ===")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

for attempt in range(3):
    try:
        snapshot_download(
            "microsoft/wavlm-large",
            local_dir="${REMOTE_ROOT}/pretrained_models/wavlm-large",
            resume_download=True,
            max_workers=1,
        )
        print("HF mirror OK!")
        sys.exit(0)
    except Exception as e:
        print(f"HF mirror attempt {attempt+1} failed: {e}")

print("ALL FAILED - need manual download")

