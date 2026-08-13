import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
snapshot_download("microsoft/wavlm-large", local_dir="${REMOTE_ROOT}/pretrained_models/wavlm-large", max_workers=1)
print("Done! Files:")
for f in os.listdir("${REMOTE_ROOT}/pretrained_models/wavlm-large"):
    size = os.path.getsize(os.path.join("${REMOTE_ROOT}/pretrained_models/wavlm-large", f))
    print(f"  {f}: {size/1e6:.1f}MB")

