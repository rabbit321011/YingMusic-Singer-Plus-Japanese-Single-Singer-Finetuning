import time, random, os, json, sys
import torch, torch.nn.functional as F, torchaudio, numpy as np

def make_trainer(device_str):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    device = torch.device("cuda:0")
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    
    from omegaconf import OmegaConf; from ema_pytorch import EMA
    cfg = OmegaConf.load("src/YingMusicSinger/config/YingMusic_Singer.yaml")
    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
    from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
    
    dit = DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
              mel_dim=cfg.model.mel_spec.n_mel_channels, long_skip_connection=True)
    singer = Singer(transformer=dit, is_tts_pretrain=cfg.model.is_tts_pretrain,
                    melody_input_source=cfg.model.melody_input_source,
                    cka_disabled=cfg.model.cka_disabled, num_channels=None,
                    extra_parameters=cfg.extra_parameters,
                    mel_spec_kwargs=cfg.model.mel_spec,
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
    ema = EMA(singer, beta=cfg.ema_kwargs.beta, update_after_step=cfg.ema_kwargs.update_after_step,
              update_every=cfg.ema_kwargs.update_every).to(device)
    optimizer = torch.optim.AdamW(singer.parameters(), lr=7e-6, betas=(0.9, 0.95), weight_decay=1e-2)
    with open("${SERVER_ROOT}/final_sum_large/train_singnet.json") as f:
        data = json.load(f)
    random.shuffle(data)
    return device, singer, vae, midi_teacher, mel_spec_ext, tokenizer, ema, optimizer, data

def run_bench(device, singer, vae, midi_teacher, mel_spec_ext, tokenizer, ema, optimizer, data, label, n_steps=200, max_dur=30):
    step_times = []
    for step in range(n_steps):
        t0 = time.time()
        optimizer.zero_grad()
        rec = data[step % len(data)]
        wav, sr = torchaudio.load(rec["Path"])
        if sr != 44100: wav = torchaudio.functional.resample(wav, sr, 44100)
        max_s = int(max_dur * 44100)
        if wav.shape[-1] > max_s: wav = wav[:, :max_s]
        wav = wav.to(device)
        text = rec["Text"]
        
        with torch.no_grad():
            w2d = wav; 
            if w2d.dim() == 1: w2d = w2d.unsqueeze(0)
            lat = vae.encode_audio(w2d, in_sr=44100)
            if lat.dim() == 3: lat = lat.squeeze(0)
            full_latent = lat.transpose(0, 1).unsqueeze(0)
            B, T, D = full_latent.shape
            ref_len = T // 2
            cond = torch.zeros_like(full_latent); cond[:, :ref_len, :] = full_latent[:, :ref_len, :]
            mel = mel_spec_ext(audio=w2d, sr=44100)
            if mel.dim() == 3: mel = mel.squeeze(0)
            mel = mel.to(device)
            midi_p, _ = midi_teacher(mel.unsqueeze(0).transpose(1, 2))
            if midi_p.shape[1] != T:
                midi_p = F.interpolate(midi_p.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
            midi = singer.smoothMelody_MIDIFuzzDisturb(midi_p)
            midi[:, :ref_len, :] = 0
            tokens = tokenizer.encode(text)
            aligned_text = torch.zeros(1, T, dtype=torch.long, device=device)
            n = min(len(tokens), T)
            aligned_text[0, :n] = torch.tensor(tokens[:n], device=device)
        
        t_v = torch.rand(1, device=device)
        noise = torch.randn_like(full_latent)
        x_t = (1 - t_v[:, None, None]) * noise + t_v[:, None, None] * full_latent
        da = random.random() < 0.3; dt_ = random.random() < 0.3; dm = random.random() < 0.3
        dit_m = singer.transformer
        time_emb = dit_m.time_embed(t_v)
        x, _ = dit_m.get_input_embed(x_t, cond, aligned_text, midi,
                                      drop_audio_cond=da, drop_text=dt_, drop_midi=dm, cache=False)
        rope = dit_m.rotary_embed.forward_from_seq_len(T)
        residual = x
        for block in dit_m.transformer_blocks:
            x = block(x, time_emb, mask=None, rope=rope)
        if dit_m.long_skip_connection is not None:
            x = dit_m.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = dit_m.norm_out(x, time_emb)
        v_pred = dit_m.proj_out(x)
        loss = F.mse_loss(v_pred, full_latent - noise)  # flow matching target
        loss.backward()
        torch.nn.utils.clip_grad_norm_(singer.parameters(), 1.0)
        optimizer.step()
        ema.update()
        torch.cuda.synchronize(device)
        t_step = time.time() - t0
        step_times.append((t_step, T))
        if step % 20 == 0:
            a = np.mean([x[0] for x in step_times[5:step+1]]) if step >= 5 else np.mean([x[0] for x in step_times[:step+1]])
            print(f"  [{label}] step={step:3d}  {t_step:.2f}s  avg={a:.2f}s  T={T}")
    
    first50 = np.mean([x[0] for x in step_times[10:60]])
    last50 = np.mean([x[0] for x in step_times[-50:]])
    degradation = (last50 - first50) / first50 * 100
    
    # Calculate speed per 1000 frames for fair comparison
    avg_T = np.mean([x[1] for x in step_times])
    avg_speed = np.mean([x[0] for x in step_times])
    speed_per_1k = avg_speed / avg_T * 1000
    
    print(f"[{label}] first50={first50:.2f}s last50={last50:.2f}s degradation={degradation:+.1f}% "
          f"avgT={avg_T:.0f}frames speed={speed_per_1k:.3f}s/1kframes")
    return first50, last50, degradation, np.array([x[0] for x in step_times]), avg_T

if __name__ == "__main__":
    print("=" * 60)
    print("Speed Degradation Diagnosis")
    print("=" * 60)
    
    # Test 1: max_dur=15s (V1 equivalent)
    print("\n=== Test 1: Single GPU set max_dur=15s ===")
    d, s, v, m, ms, tok, e, o, data = make_trainer("0")
    f1, l1, deg1, times1, T1 = run_bench(d, s, v, m, ms, tok, e, o, data, "1G-15s", 200, 15)
    del s, v, m, ms, e, o; torch.cuda.empty_cache()
    
    # Test 2: max_dur=30s (V3 config)
    print("\n=== Test 2: Single GPU set max_dur=30s ===")
    d, s, v, m, ms, tok, e, o, data = make_trainer("0")
    f2, l2, deg2, times2, T2 = run_bench(d, s, v, m, ms, tok, e, o, data, "1G-30s", 200, 30)
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print(f"  max_dur=15s: first50={f1:.2f}s  last50={l1:.2f}s  degradation={deg1:+.1f}%  avgT={T1:.0f}frames")
    print(f"  max_dur=30s: first50={f2:.2f}s  last50={l2:.2f}s  degradation={deg2:+.1f}%  avgT={T2:.0f}frames")
    
    if deg2 > 20:
        print(f"\n  => max_dur=30s causes {deg2:.0f}% degradation on SINGLE GPU")
        print(f"  => NOT a DDP issue. Root cause: long sequence (avg {T2:.0f} frames) memory fragmentation")
    elif deg1 > 20:
        print(f"\n  => Even max_dur=15s degrades on single GPU. Something else is wrong.")
    else:
        print(f"\n  => Single GPU is stable. Degradation is DDP-specific.")

















