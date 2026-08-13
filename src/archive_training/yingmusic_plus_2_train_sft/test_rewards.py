import torch, os, sys, json
device = "cuda:2"

# Load sample audio
with open("${REMOTE_ROOT}/final_sum_large/train_singnet.json") as f:
    d = json.load(f)
import torchaudio
wav, sr = torchaudio.load(d[0]["Path"])
wav = wav.to(device)
ref_text = d[0]["Text"]
print(f"Audio: {wav.shape}, sr={sr}, text='{ref_text[:60]}...'")

# === TEST 1: Whisper ASR ===
print("\n=== TEST 1: Whisper ===")
from transformers import pipeline
whisper = pipeline("automatic-speech-recognition", model="Systran/faster-whisper-medium",
                   device=int(device.split(":")[-1]))
audio_np = wav.cpu().squeeze().numpy()
if audio_np.ndim > 1:
    audio_np = audio_np.T
result = whisper({"raw": audio_np, "sampling_rate": sr}, generate_kwargs={"language": "ja", "task": "transcribe"})
print(f"  Whisper: '{result['text']}'")
import jiwer
wer_val = jiwer.wer(ref_text, result["text"])
print(f"  WER: {wer_val:.4f}")
print("  OK")

# === TEST 2: WavLM ===
print("\n=== TEST 2: WavLM ===")
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
wavlm_ext = Wav2Vec2FeatureExtractor.from_pretrained("${REMOTE_ROOT}/pretrained_models/wavlm-large")
wavlm = WavLMForXVector.from_pretrained("${REMOTE_ROOT}/pretrained_models/wavlm-large").to(device).eval()
wav_16k = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, 16000)
wav_np = wav_16k.squeeze().cpu().numpy()
feat = wavlm_ext(wav_np, sampling_rate=16000, return_tensors="pt").input_values.to(device)
emb = wavlm(feat).embeddings
print(f"  Embedding shape: {emb.shape}")
print(f"  Norm: {emb.norm().item():.4f}")
print("  OK")

# === TEST 3: RMVPE ===
print("\n=== TEST 3: RMVPE ===")
sys.path.insert(0, "${REMOTE_ROOT}/YingMusic-Singer/src/singer/rmvpe")
from model import RMVPE
rmvpe = RMVPE()
rmvpe.load_state_dict(torch.load("${REMOTE_ROOT}/YingMusic-Singer/rmvpe.pt", map_location="cpu"), strict=False)
rmvpe = rmvpe.to(device).eval()
wav_16k2 = torchaudio.functional.resample(wav[:, :sr*5].mean(0, keepdim=True), sr, 16000)
with torch.no_grad():
    f0 = rmvpe.infer_from_audio(wav_16k2, 16000) if hasattr(rmvpe, 'infer_from_audio') else rmvpe(wav_16k2)
print(f"  F0 shape: {f0.shape if isinstance(f0, torch.Tensor) else 'list'}")
print(f"  F0 range: {f0.min().item():.1f} - {f0.max().item():.1f}")
print("  OK")

print("\nALL TESTS PASSED!")

