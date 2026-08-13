import argparse, json, os, sys, time
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default="ckpts/YingMusicSinger_model.pt")
    parser.add_argument("--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml")
    parser.add_argument("--vae_config", default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json")
    parser.add_argument("--vae_ckpt", default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt")
    parser.add_argument("--midi_ckpt", default="ckpts/model_ckpt_steps_100000_simplified.ckpt")
    parser.add_argument("--test_json", default="${SERVER_ROOT}/final_sum_large/test_singnet.json")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--nfe_steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
    from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer

    print("[EvalWER] Building DiT...")
    dit = DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
              mel_dim=cfg.model.mel_spec.n_mel_channels, long_skip_connection=True)
    singer = Singer(transformer=dit, is_tts_pretrain=cfg.model.is_tts_pretrain,
                    melody_input_source=cfg.model.melody_input_source,
                    cka_disabled=cfg.model.cka_disabled, num_channels=None,
                    extra_parameters=cfg.extra_parameters,
                    mel_spec_kwargs=cfg.model.mel_spec,
                    distill_stage=None, use_guidance_scale_embed=False)

    print(f"[EvalWER] Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if "ema_model_state_dict" in ckpt:
        sd = ckpt["ema_model_state_dict"]
        sd = {k.replace("ema_model.", ""): v for k, v in sd.items()}
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    singer.load_state_dict(sd, strict=False)
    singer = singer.to(device).eval()

    print("[EvalWER] Loading VAE...")
    vae = StableAudioInfer(model_config_path=args.vae_config, model_ckpt_path=args.vae_ckpt)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    print("[EvalWER] Loading SOME MIDI...")
    midi_teacher = MIDIExtractor(in_dim=80)
    midi_teacher._load_form_ckpt(args.midi_ckpt)
    midi_teacher = midi_teacher.to(device).eval()
    for p in midi_teacher.parameters():
        p.requires_grad = False

    mel_spec_ext = MelodySpectrogram()
    tokenizer = CNENTokenizer()
    vae_frame_rate = 44100 / 2048

    print("[EvalWER] Loading Whisper...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel("Systran/faster-whisper-medium", device="cuda",
                           compute_type="float16", num_workers=1,
                           download_root="${SERVER_ROOT}/.cache/huggingface/hub")

    with open(args.test_json) as f:
        test_data = json.load(f)
    test_data = test_data[:args.n_samples]
    print(f"[EvalWER] Test samples: {len(test_data)}")

    results = []
    t_start = time.time()

    for idx, rec in enumerate(test_data):
        wav, sr = torchaudio.load(rec["Path"])
        if sr != 44100:
            wav = torchaudio.functional.resample(wav, sr, 44100)
            sr = 44100
        max_samples = int(args.max_duration * sr)
        if wav.shape[-1] > max_samples:
            wav = wav[:, :max_samples]
        wav = wav.to(device)
        ref_text = rec["Text"]

        with torch.no_grad():
            wav_2d = wav.squeeze(0) if wav.dim() == 2 else wav
            if wav_2d.dim() == 1:
                wav_2d = wav_2d.unsqueeze(0)

            latent_raw = vae.encode_audio(wav_2d, in_sr=sr)
            if latent_raw.dim() == 3:
                latent_raw = latent_raw.squeeze(0)
            full_latent = latent_raw.transpose(0, 1).unsqueeze(0)
            B, T, D = full_latent.shape

            ref_len = T // 2
            cond = torch.zeros_like(full_latent)
            cond[:, :ref_len, :] = full_latent[:, :ref_len, :]

            mel = mel_spec_ext(audio=wav_2d, sr=sr)
            if mel.dim() == 3:
                mel = mel.squeeze(0)
            mel = mel.to(device)
            midi_p, bound_p = midi_teacher(mel.unsqueeze(0).transpose(1, 2))
            if midi_p.shape[1] != T:
                midi_p = F.interpolate(midi_p.transpose(1, 2), size=T, mode="linear",
                                       align_corners=False).transpose(1, 2)
            midi = midi_p.clone()
            midi[:, :ref_len, :] = 0

            tokens = tokenizer.encode(ref_text)
            text_ids = torch.zeros(1, T, dtype=torch.long, device=device)
            n = min(len(tokens), T)
            text_ids[0, :n] = torch.tensor(tokens[:n], device=device)

            noise = torch.randn(1, T, D, device=device)
            x_t = 0.5 * noise + 0.5 * full_latent[:, :1, :].expand(-1, T, -1)

            t_vals = torch.linspace(0.5, 1, args.nfe_steps + 1, device=device)
            t_vals = 0.5 * t_vals / (1 + (0.5 - 1) * t_vals)

            x = x_t
            for i in range(len(t_vals) - 1):
                dt_val = t_vals[i + 1] - t_vals[i]
                t_val = t_vals[i]
                v_cond, _ = singer.transformer(x=x, cond=cond, text=text_ids,
                                                time=t_val, midi=midi,
                                                drop_audio_cond=False,
                                                drop_text=False, drop_midi=False)
                v_uncond, _ = singer.transformer(x=x, cond=cond, text=text_ids,
                                                  time=t_val, midi=midi,
                                                  drop_audio_cond=True,
                                                  drop_text=True, drop_midi=True)
                v = v_cond + (v_cond - v_uncond) * args.cfg_strength
                x = x + v * dt_val.item()

            generated_latent = x
            lat_dec = generated_latent.permute(0, 2, 1).float()
            audio_gen = vae.decode_audio(lat_dec)

        audio_np = audio_gen.squeeze().cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.T

        try:
            segments, _ = whisper.transcribe(audio_np, language="ja", task="transcribe",
                                              beam_size=5, word_timestamps=False)
            hyp_text = "".join(s.text for s in segments).replace(" ", "")
        except Exception as e:
            hyp_text = ""
            print(f"  [{idx}] Whisper error: {e}")

        import jiwer
        wer_val = jiwer.wer(ref_text, hyp_text) if hyp_text.strip() else 1.0

        results.append({"idx": idx, "ref": ref_text[:60], "hyp": hyp_text[:60], "wer": wer_val,
                         "T": T, "ref_len": ref_len})
        print(f"  [{idx:2d}] WER={wer_val:.3f} | T={T} | ref='{ref_text[:30]}...'")

    t_total = time.time() - t_start
    wer_values = [r["wer"] for r in results]
    mean_wer = np.mean(wer_values)
    print(f"\n{'='*60}")
    print(f"Mean WER: {mean_wer:.4f} (n={len(results)}, time={t_total:.0f}s)")
    for r in results:
        print(f"  [{r['idx']:2d}] WER={r['wer']:.3f} | ref='{r['ref']}' | hyp='{r['hyp']}'")

    if args.output_csv:
        with open(args.output_csv, "w") as f:
            f.write("idx,wer,ref,hyp,T,ref_len\n")
            for r in results:
                f.write(f"{r['idx']},{r['wer']:.4f},{r['ref']},{r['hyp']},{r['T']},{r['ref_len']}\n")
        print(f"Saved: {args.output_csv}")

    return mean_wer

if __name__ == "__main__":
    main()

















