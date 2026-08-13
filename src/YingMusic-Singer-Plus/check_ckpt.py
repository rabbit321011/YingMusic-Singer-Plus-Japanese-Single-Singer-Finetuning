import torch
for step in [10000, 20000]:
    path = f"ckpts/plus_ja_sft_v3/step_{step:06d}.pt"
    ckpt = torch.load(path, map_location="cpu")
    print(f"{path}: keys={list(ckpt.keys())[:3]}, ema_keys_count={len(ckpt.get('ema_model_state_dict',{}))}, model_keys_count={len(ckpt.get('model_state_dict',{}))}")
print("Done")

















