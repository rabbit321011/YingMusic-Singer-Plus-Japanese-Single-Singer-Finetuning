import argparse, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml")
    parser.add_argument("--sft_ckpt", required=True, help="SFT checkpoint .pt")
    parser.add_argument("--base_ckpt", default="ckpts/YingMusicSinger_model.pt")
    parser.add_argument("--vae_config", default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json")
    parser.add_argument("--vae_ckpt", default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt")
    parser.add_argument("--midi_ckpt", default="ckpts/model_ckpt_steps_100000_simplified.ckpt")
    parser.add_argument("--wavlm_path", default="${REMOTE_ROOT}/pretrained_models/wavlm-large")
    parser.add_argument("--rmvpe_path", default="${REMOTE_ROOT}/YingMusic-Singer/rmvpe.pt")
    parser.add_argument("--data_json", default="${REMOTE_ROOT}/final_sum_large/train_singnet.json")
    parser.add_argument("--output_dir", default="ckpts/plus_grpo")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--nfe_step", type=int, default=32)
    parser.add_argument("--noise_level", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--clip_epsilon", type=float, default=0.02)
    parser.add_argument("--max_steps", type=int, default=32)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=8)
    parser.add_argument("--max_duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit_n", type=int, default=10)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--reward_device", default="cuda:7")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    reward_device = torch.device(args.reward_device)

    sys.path.insert(0, os.getcwd())
    from omegaconf import OmegaConf
    from ema_pytorch import EMA

    cfg = OmegaConf.load(args.config)

    # ========== Build Singer Model ==========
    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
    from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
    from src.YingMusicSinger.utils.lrc_align import align_lrc_sentence_level

    print("[GRPO] Building Singer...")
    dit = DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
              mel_dim=cfg.model.mel_spec.n_mel_channels, long_skip_connection=True)
    singer = Singer(transformer=dit, is_tts_pretrain=cfg.model.is_tts_pretrain,
                    melody_input_source=cfg.model.melody_input_source,
                    cka_disabled=cfg.model.cka_disabled, num_channels=None,
                    extra_parameters=cfg.extra_parameters,
                    mel_spec_kwargs=cfg.model.mel_spec,
                    distill_stage=None, use_guidance_scale_embed=False)

    # Load SFT weights
    print(f"[GRPO] Loading SFT: {args.sft_ckpt}")
    sft_ckpt = torch.load(args.sft_ckpt, map_location="cpu")
    if "ema_model_state_dict" in sft_ckpt:
        sd = sft_ckpt["ema_model_state_dict"]
        sd = {k.replace("ema_model.", ""): v for k, v in sd.items()}
    elif "model_state_dict" in sft_ckpt:
        sd = sft_ckpt["model_state_dict"]
    else:
        sd = sft_ckpt
    singer.load_state_dict(sd, strict=False)
    singer = singer.to(device).train()

    # Reference (frozen) for KL penalty
    ref_singer = Singer(transformer=DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
                                         mel_dim=cfg.model.mel_spec.n_mel_channels, long_skip_connection=True),
                        is_tts_pretrain=cfg.model.is_tts_pretrain,
                        melody_input_source=cfg.model.melody_input_source,
                        cka_disabled=cfg.model.cka_disabled, num_channels=None,
                        extra_parameters=cfg.extra_parameters,
                        mel_spec_kwargs=cfg.model.mel_spec,
                        distill_stage=None, use_guidance_scale_embed=False)
    ref_singer.load_state_dict(sd, strict=False)
    ref_singer = ref_singer.to(device).eval()
    for p in ref_singer.parameters():
        p.requires_grad = False

    # VAE + MIDI (frozen)
    vae = StableAudioInfer(model_config_path=args.vae_config, model_ckpt_path=args.vae_ckpt)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    mel_spec_ext = MelodySpectrogram()
    midi_teacher = MIDIExtractor(in_dim=80)
    midi_teacher._load_form_ckpt(args.midi_ckpt)
    midi_teacher = midi_teacher.to(device).eval()
    for p in midi_teacher.parameters():
        p.requires_grad = False

    tokenizer = CNENTokenizer()
    vae_frame_rate = 44100 / 2048

    # ========== Reward Models ==========
    # Whisper ASR (faster-whisper)
    from faster_whisper import WhisperModel
    print("[GRPO] Loading Whisper...")
    whisper = WhisperModel("Systran/faster-whisper-medium", device="cuda",
                           compute_type="float16", num_workers=1,
                           download_root="${REMOTE_ROOT}/.cache/huggingface/hub")

    # WavLM for speaker similarity
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    print("[GRPO] Loading WavLM...")
    wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wavlm_path)
    wavlm_model = WavLMModel.from_pretrained(args.wavlm_path).to(reward_device).eval()
    for p in wavlm_model.parameters():
        p.requires_grad = False

    # RMVPE (use torchcrepe for simplicity)
    print("[GRPO] Loading torchcrepe for F0...")
    import torchcrepe

    # ========== Reward Functions ==========
    def compute_wer(ref_text, hyp_text):
        import jiwer
        return jiwer.wer(ref_text, hyp_text) if hyp_text.strip() else 1.0

    @torch.no_grad()
    def compute_f0_corr(audio, ref_audio, sr=44100):
        audio_mono = audio.mean(0) if audio.dim() > 1 and audio.shape[0] > 1 else audio.squeeze(0)
        ref_mono = ref_audio.mean(0) if ref_audio.dim() > 1 and ref_audio.shape[0] > 1 else ref_audio.squeeze(0)
        audio_16k = torchaudio.functional.resample(audio_mono.unsqueeze(0), sr, 16000).to(reward_device)
        ref_16k = torchaudio.functional.resample(ref_mono.unsqueeze(0), sr, 16000).to(reward_device)

        f0_gen = torchcrepe.predict(audio_16k, 16000, hop_length=160, fmin=50, fmax=1100,
                                      model="full", device=reward_device, batch_size=512)
        f0_ref = torchcrepe.predict(ref_16k, 16000, hop_length=160, fmin=50, fmax=1100,
                                      model="full", device=reward_device, batch_size=512)

        min_len = min(f0_gen.shape[-1], f0_ref.shape[-1])
        f0_gen = f0_gen[0, :min_len].cpu().numpy()
        f0_ref = f0_ref[0, :min_len].cpu().numpy()

        mask = (f0_gen > 0) & (f0_ref > 0)
        if mask.sum() < 10:
            return 0.0
        return max(0.0, float(np.corrcoef(f0_gen[mask], f0_ref[mask])[0, 1]))

    @torch.no_grad()
    def compute_speaker_sim(audio, ref_audio, sr=44100):
        target_sr = 16000
        a1 = torchaudio.functional.resample(audio.mean(0, keepdim=True) if audio.dim() > 1 else audio, sr, target_sr)
        a2 = torchaudio.functional.resample(ref_audio.mean(0, keepdim=True) if ref_audio.dim() > 1 else ref_audio, sr, target_sr)
        a1_np = a1.squeeze().cpu().numpy()
        a2_np = a2.squeeze().cpu().numpy()
        feat1 = wavlm_extractor(a1_np, sampling_rate=target_sr, return_tensors="pt").input_values.to(reward_device)
        feat2 = wavlm_extractor(a2_np, sampling_rate=target_sr, return_tensors="pt").input_values.to(reward_device)
        out1 = wavlm_model(feat1).last_hidden_state.mean(dim=1)  # [1, D]
        out2 = wavlm_model(feat2).last_hidden_state.mean(dim=1)
        out1 = F.normalize(out1, dim=-1)
        out2 = F.normalize(out2, dim=-1)
        return max(0.0, (out1 * out2).sum(-1).item())

    @torch.no_grad()
    def compute_rewards(generated_audio, ref_audio, ref_text, sr=44100):
        rewards = {}
        audio_cpu = generated_audio.cpu()

        # 1. Whisper WER
        audio_np = audio_cpu.squeeze().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.T
        try:
            segments, _ = whisper.transcribe(audio_np, language="ja", task="transcribe",
                                              beam_size=5, word_timestamps=False)
            hyp_text = " ".join(s.text for s in segments)
            wer_val = jiwer.wer(ref_text, hyp_text) if hyp_text.strip() else 1.0
            rewards["wer"] = 1.0 - min(wer_val, 1.0)
        except Exception as e:
            print(f"    Whisper error: {e}")
            rewards["wer"] = 0.0

        # 2. F0 correlation
        rewards["f0"] = compute_f0_corr(generated_audio.squeeze(0) if generated_audio.dim() == 3 else generated_audio,
                                         ref_audio.squeeze(0) if ref_audio.dim() == 3 else ref_audio, sr)

        # 3. WavLM speaker similarity
        rewards["spk"] = compute_speaker_sim(generated_audio.squeeze(0) if generated_audio.dim() == 3 else generated_audio,
                                              ref_audio.squeeze(0) if ref_audio.dim() == 3 else ref_audio, sr)

        total = 0.25 * rewards["wer"] + 0.25 * rewards["f0"] + 0.50 * rewards["spk"]
        return total, rewards

    # ========== Dataset ==========
    class SVSDataset(Dataset):
        def __init__(self, records, max_dur):
            self.records = records; self.max_dur = max_dur
        def __len__(self):
            return len(self.records)
        def __getitem__(self, idx):
            rec = self.records[idx]
            wav, sr = torchaudio.load(rec["Path"])
            if sr != 44100:
                wav = torchaudio.functional.resample(wav, sr, 44100)
            max_s = int(self.max_dur * 44100)
            if wav.shape[-1] > max_s:
                wav = wav[:, :max_s]
            return {"wav": wav, "sr": 44100, "text": rec["Text"]}

    with open(args.data_json) as f:
        all_data = json.load(f)
    if args.overfit:
        all_data = all_data[:args.overfit_n]
    dataset = SVSDataset(all_data, args.max_duration)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"[GRPO] Dataset: {len(dataset)} samples | Max steps: {args.max_steps}")

    # ========== Optimizer ==========
    optimizer = torch.optim.AdamW(singer.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== GRPO Training Loop ==========
    global_step = 0
    data_iter = iter(loader)

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        wav = batch["wav"].to(device)
        ref_text = batch["text"][0]
        sr = batch["sr"] if isinstance(batch["sr"], int) else batch["sr"][0]

        # ===== Preprocess: compute latent, MIDI, text tokens =====
        with torch.no_grad():
            wav_dev = wav.to(device)
            # VAE needs [C, T] shape
            wav_2d = wav_dev.squeeze(0)
            latent_raw = vae.encode_audio(wav_2d, in_sr=sr)
            if latent_raw.dim() == 3:
                latent_raw = latent_raw.squeeze(0)
            latent = latent_raw.transpose(0, 1).unsqueeze(0)  # [1, T, 64]
            B, T, D = latent.shape

            mel = mel_spec_ext(audio=wav_2d, sr=sr)
            if mel.dim() == 3:
                mel = mel.squeeze(0)
            mel = mel.to(device)
            midi_p, bound_p = midi_teacher(mel.unsqueeze(0).transpose(1, 2))
            if midi_p.shape[1] != T:
                midi_p = F.interpolate(midi_p.transpose(1, 2), size=T, mode="linear",
                                       align_corners=False).transpose(1, 2)
            midi_fuzz = singer.smoothMelody_MIDIFuzzDisturb(midi_p)

            tokens = tokenizer.encode(ref_text)
            lrc_token, _ = align_lrc_sentence_level(tokenizer=tokenizer, lrc_start_times=[0.0],
                                                     lrc_lines=[ref_text], total_lens=T,
                                                     vae_frame_rate=vae_frame_rate)
            text_ids = torch.zeros(1, T, dtype=torch.long)
            n = min(len(lrc_token), T)
            text_ids[0, :n] = torch.tensor(lrc_token[:n])

        # ===== Generate 8 variants with different seeds =====
        singer.eval()
        all_latents = []
        for k in range(args.num_samples):
            seed = args.seed * 1000 + global_step * args.num_samples + k
            torch.manual_seed(seed)
            noise = torch.randn(1, T, D, device=device)

            t_start = args.noise_level  # start from noise_level
            x_t = (1 - t_start) * noise + t_start * latent[:, :1, :].expand(-1, T, -1)

            t = torch.linspace(t_start, 1, args.nfe_step + 1, device=device)
            t = 0.5 * t / (1 + (0.5 - 1) * t)  # t_shift=0.5

            # Simple Euler ODE
            x = x_t
            cond_ = latent.to(device)
            midi_ = midi_fuzz.to(device)
            text_ = text_ids.to(device)
            for i in range(len(t) - 1):
                dt_val = t[i + 1] - t[i]
                t_val = t[i]
                # CFG forward
                with torch.no_grad():
                    v_cond, _ = singer.transformer(x=x, cond=cond_, text=text_, time=t_val,
                                                    midi=midi_, drop_audio_cond=False,
                                                    drop_text=False, drop_midi=False)
                    v_uncond, _ = singer.transformer(x=x, cond=cond_, text=text_, time=t_val,
                                                      midi=midi_, drop_audio_cond=True,
                                                      drop_text=True, drop_midi=True)
                v = v_cond + (v_cond - v_uncond) * 3.0
                x = x + v * dt_val.item()
            all_latents.append(x)

        all_latents = torch.stack(all_latents, dim=0)  # [K, 1, T, D]

        # ===== Decode all variants to audio =====
        singer.train()
        with torch.no_grad():
            all_audio = []
            for k in range(args.num_samples):
                lat_k = all_latents[k]
                lat_k = lat_k.permute(0, 2, 1).float()  # [1, D, T], on singer device
                audio_k = vae.decode_audio(lat_k)
                all_audio.append(audio_k.cpu())

        # Reference audio for rewards (keep on CPU for processing)
        ref_audio_cpu = wav.squeeze(0).cpu()

        # ===== Compute rewards =====
        all_rewards = []
        for k in range(args.num_samples):
            r, d = compute_rewards(all_audio[k], ref_audio_cpu, ref_text, 44100)
            all_rewards.append(r)

        rewards = torch.tensor(all_rewards, device=device)
        # Group relative
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # ===== PPO update: compute log prob of each variant =====
        total_loss = 0.0
        for k in range(args.num_samples):
            latent_k = all_latents[k].to(device)
            t_train = torch.rand(1, device=device)
            noise_train = torch.randn_like(latent_k)
            x_t_train = (1 - t_train) * noise_train + t_train * latent_k
            v_target = latent_k - noise_train

            v_pred, _ = singer.transformer(x=x_t_train, cond=latent.to(device),
                                            text=text_.to(device), time=t_train,
                                            midi=midi_.to(device),
                                            drop_audio_cond=False, drop_text=False, drop_midi=False)

            log_prob = -F.mse_loss(v_pred, v_target, reduction="sum") / D

            # KL with reference
            with torch.no_grad():
                v_ref, _ = ref_singer.transformer(x=x_t_train.to(device),
                                                   cond=latent.to(device),
                                                   text=text_ids.to(device),
                                                   time=t_train.to(device),
                                                   midi=midi_fuzz.to(device),
                                                   drop_audio_cond=False, drop_text=False, drop_midi=False)
            log_prob_ref = -F.mse_loss(v_ref, v_target, reduction="sum") / D

            ratio = torch.exp(log_prob - log_prob_ref.detach())
            clipped = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
            loss_k = -torch.min(ratio * advantages[k], clipped * advantages[k]) + args.beta * (log_prob_ref.detach() - log_prob)
            total_loss += loss_k

        total_loss = total_loss / args.num_samples
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(singer.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        if global_step % args.log_every == 0:
            print(f"  [GRPO Step {global_step:4d}/{args.max_steps}] "
                  f"Rewards: {rewards.tolist()} | Mean: {mean_r:.4f}±{std_r:.4f} | Loss: {total_loss.item():.6f}")

        if global_step % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"grpo_step_{global_step:04d}.pt")
            torch.save({"model_state_dict": singer.state_dict(), "global_step": global_step}, save_path)
            print(f"  [Save] {save_path}")

    save_path = os.path.join(args.output_dir, f"grpo_final.pt")
    torch.save({"model_state_dict": singer.state_dict(), "global_step": global_step}, save_path)
    print(f"[GRPO] Final: {save_path}")

if __name__ == "__main__":
    main()

