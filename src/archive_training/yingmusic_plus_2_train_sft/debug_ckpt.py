import torch, sys
sys.path.insert(0, "${REMOTE_ROOT}/YingMusic-Singer-Plus")
os = __import__("os"); os.chdir("${REMOTE_ROOT}/YingMusic-Singer-Plus")

ckpt = torch.load("ckpts/YingMusicSinger_model.pt", map_location="cpu")
print("Checkpoint keys:")
for k in list(ckpt.keys())[:5]:
    print(f"  {k}: type={type(ckpt[k]).__name__}")

sd = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict", ckpt))
keys = list(sd.keys())
print(f"\nState dict keys: {len(keys)}")
for k in keys[:15]:
    print(f"  {k}")
print("  ...")
for k in keys[-5:]:
    print(f"  {k}")

