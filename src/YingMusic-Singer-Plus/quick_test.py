import time, random, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch, torchaudio, numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0")
torch.manual_seed(42)

from omegaconf import OmegaConf
cfg = OmegaConf.load("src/YingMusicSinger/config/YingMusic_Singer.yaml")

from src.YingMusicSinger.models.dit import DiT
from src.YingMusicSinger.models.model import Singer
from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
from ema_pytorch import EMA

print("[QuickTest] Loading models...")
dit = DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
          mel_dim=cfg.model.mel_spec.n_mel_channels, long_skip_connection=True)
singer = Singer(transformer=dit, is_tts_pretrain=cfg.model.is_tts_pretrain,
                melody_input_source=cfg.model.melody_input_source,
                cka_disabled=cfg.model.cka_disabled, num_channels=None,
                extra_parameters=cfg.extra_parameters, mel_spec_kwargs=cfg.model.mel_spec,
                distill_stage=None, use_guidance_scale_embed=False)

ckpt = torch.load("ckpts/YingMusicSinger_model.pt", map_location="cpu")
sd = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict", ckpt))
sd = {k.replace("ema_model.", "").replace("module.", ""): v for k, v in sd.items()}
singer.load_state_dict(sd, strict=False)
singer = singer.to(device).train()

vae = StableAudioInfer(model_config_path="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",
                        model_ckpt_path="ckpts/stable_audio_2_0_vae_20hz_official.ckpt")
vae = vae.to(device).eval()
for p in vae.parameters(): p.requires_grad = False

midi_teacher = MIDIExtractor(in_dim=80)
midi_teacher._load_form_ckpt("ckpts/model_ckpt_steps_100000_simplified.ckpt")
midi_teacher = midi_teacher.to(device).eval()
for p in midi_teacher.parameters(): p.requires_grad = False

mel_spec_ext = MelodySpectrogram()
tokenizer = CNENTokenizer()
ema = EMA(singer, beta=0.995, update_after_step=100, update_every=1).to(device)

optimizer = torch.optim.AdamW(singer.parameters(), lr=7e-6, betas=(0.9, 0.95), weight_decay=1e-2)

import json
with open("${SERVER_ROOT}/final_sum_large/train_singnet.json") as f:
    data = json.load(f)

print(f"[QuickTest] Loaded {len(data)} samples. Pre-loading 4 audio files...")

preloaded = []
for i in range(4):
    wav, sr = torchaudio.load(data[i]["Path"])
    if sr != 44100:
        wav = torchaudio.functional.resample(wav, sr, 44100)
    max_s = int(15 * 44100)
    if wav.shape[-1] > max_s:
        wav = wav[:, :max_s]
    preloaded.append({"wav": wav, "text": data[i]["Text"]})

print("[QuickTest] Running 200 steps, batch=2, grad_accum=4 (effective=8)...")
print(f"{'Step':>5s} {'total':>7s} {'speed':>7s}")

t_total = 0
for step in range(200):
    optimizer.zero_grad()
    t0 = time.time()

    for ga in range(4):
        batch = preloaded[ga % 4]
        wav = batch["wav"].to(device)
        text = batch["text"]
        sr = 44100

        with torch.no_grad():
            w2d = wav
            if w2d.dim() == 1:
                w2d = w2d.unsqueeze(0)
            lat = vae.encode_audio(w2d, in_sr=sr)
            if lat.dim() == 3: lat = lat.squeeze(0)
            full_latent = lat.transpose(0, 1).unsqueeze(0)
            B, T, D = full_latent.shape

            ref_len = T // 2
            cond = torch.zeros_like(full_latent)
            cond[:, :ref_len, :] = full_latent[:, :ref_len, :]

            mel = mel_spec_ext(audio=w2d, sr=sr)
            if mel.dim() == 3: mel = mel.squeeze(0)
            mel = mel.to(device)
            midi_p, _ = midi_teacher(mel.unsqueeze(0).transpose(1, 2))
            if midi_p.shape[1] != T:
                midi_p = F.interpolate(midi_p.transpose(1, 2), size=T, mode="linear",
                                       align_corners=False).transpose(1, 2)
            midi = singer.smoothMelody_MIDIFuzzDisturb(midi_p)
            midi[:, :ref_len, :] = 0

            tokens = tokenizer.encode(text)
            aligned_text = torch.zeros(1, T, dtype=torch.long, device=device)
            n = min(len(tokens), T)
            aligned_text[0, :n] = torch.tensor(tokens[:n], device=device)

        t_v = torch.rand(1, device=device)
        noise = torch.randn_like(full_latent)
        x_t = (1 - t_v[:, None, None]) * noise + t_v[:, None, None] * full_latent
        v_target = full_latent - noise

        da = random.random() < 0.3
        dt = random.random() < 0.3
        dm = random.random() < 0.3

        dit_m = singer.transformer
        time_emb = dit_m.time_embed(t_v)
        x, _ = dit_m.get_input_embed(x_t, cond, aligned_text, midi,
                                      drop_audio_cond=da, drop_text=dt, drop_midi=dm, cache=False)
        rope = dit_m.rotary_embed.forward_from_seq_len(T)
        residual = x
        for block in dit_m.transformer_blocks:
            x = block(x, time_emb, mask=None, rope=rope)
        if dit_m.long_skip_connection is not None:
            x = dit_m.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = dit_m.norm_out(x, time_emb)
        v_pred = dit_m.proj_out(x)

        loss = F.mse_loss(v_pred, v_target) / 4
        loss.backward()

    torch.nn.utils.clip_grad_norm_(singer.parameters(), 1.0)
    optimizer.step()
    ema.update()

    t_step = time.time() - t0
    t_total += t_step

    if step % 20 == 0:
        print(f"  {step:3d}  {t_step:5.2f}s  avg={t_total/(step+1):.2f}s")

print(f"\n[QuickTest] Final avg: {t_total/200:.2f}s/step | Total: {t_total:.1f}s | effective batch=8")

















