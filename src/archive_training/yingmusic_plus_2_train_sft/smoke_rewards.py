import os, sys, json, torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, "${REMOTE_ROOT}/YingMusic-Singer-Plus")
os.chdir("${REMOTE_ROOT}/YingMusic-Singer-Plus")
device = "cuda:2"

# Load sample
with open("${REMOTE_ROOT}/final_sum_large/train_singnet.json") as f:
    d = json.load(f)
import torchaudio
wav, sr = torchaudio.load(d[0]["Path"])
ref_text = d[0]["Text"]
wav_5s = wav[:, :sr*5]
print(f"Audio: {wav_5s.shape}")
print(f"Text: '{ref_text}'")
print()

# === Whisper ===
print("1. Loading Whisper...")
from faster_whisper import WhisperModel
whisper = WhisperModel("Systran/faster-whisper-medium", device="cuda", compute_type="float16",
                       download_root="${REMOTE_ROOT}/.cache/huggingface/hub")
audio_np = wav_5s.mean(0).cpu().numpy()
segments, info = whisper.transcribe(audio_np, language="ja", beam_size=5)
hyp = " ".join(s.text for s in segments)
print(f"   Whisper: '{hyp}'")

import jiwer
wer_val = jiwer.wer(ref_text, hyp)
print(f"   WER: {wer_val:.4f}  OK")

# === WavLM ===
print()
print("2. Loading WavLM...")
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
wavlm_ext = Wav2Vec2FeatureExtractor.from_pretrained("${REMOTE_ROOT}/pretrained_models/wavlm-large")
wavlm = WavLMForXVector.from_pretrained("${REMOTE_ROOT}/pretrained_models/wavlm-large").to(device).eval()
wav_16k = torchaudio.functional.resample(wav_5s.mean(0, keepdim=True), sr, 16000)
wav_np = wav_16k.squeeze().cpu().numpy()
feat = wavlm_ext(wav_np, sampling_rate=16000, return_tensors="pt").input_values.to(device)
emb = wavlm(feat).embeddings
print(f"   Embedding: {emb.shape} norm={emb.norm().item():.3f}  OK")

# === torchcrepe ===
print()
print("3. Loading torchcrepe...")
import torchcrepe
wav_16k2 = torchaudio.functional.resample(wav_5s.mean(0, keepdim=True), sr, 16000).to(device)
f0 = torchcrepe.predict(wav_16k2, 16000, hop_length=160, fmin=50, fmax=1100,
                          model="full", device=device, batch_size=512)
print(f"   F0: shape={f0.shape}, range=[{f0.min().item():.1f}, {f0.max().item():.1f}]  OK")

print()
print("ALL REWARD MODELS WORKING!")

