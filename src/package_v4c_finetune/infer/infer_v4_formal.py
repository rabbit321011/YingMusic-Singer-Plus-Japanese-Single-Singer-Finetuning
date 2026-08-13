"""
YingMusic-Singer-Plus 推理脚本（本地 checkpoint 版）
========================================================
翻唱 / 自克隆 / 改词 均可。

用法:
    # 翻唱（不同音色 + 不同旋律/歌词）
    python infer_v4_formal.py --ref_audio ref.wav --melody_audio melody.wav --target_text "歌詞" --output out.wav

    # 自克隆（同一条音频做 ref 和 melody，可选改词）
    python infer_v4_formal.py --ref_audio song.wav --target_text "新歌詞" --output out.wav

    # 关键参数
    --steps 32 --cfg 3.0 --seed 42
"""

import argparse, os, sys

# Fix torch 2.6 weights_only issue
import torch
torch.load_orig = torch.load


def _load(*a, **kw):
    kw.setdefault("weights_only", False)
    return torch.load_orig(*a, **kw)


torch.load = _load

import torchaudio, torch.nn.functional as F
from omegaconf import OmegaConf
from src.YingMusicSinger.models.dit import DiT
from src.YingMusicSinger.models.model import Singer
from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
from src.YingMusicSinger.utils import lrc_align

VFR = 44100 / 2048  # VAE frame rate


def build_model(checkpoint, config="src/YingMusicSinger/config/YingMusic_Singer.yaml",
                vae_config="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",
                vae_ckpt="ckpts/stable_audio_2_0_vae_20hz_official.ckpt",
                midi_ckpt="ckpts/model_ckpt_steps_100000_simplified.ckpt",
                device="cuda:0"):
    """加载模型（兼容本地 checkpoint）"""
    cfg = OmegaConf.load(config)

    policy = Singer(
        transformer=DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
                        mel_dim=cfg.model.mel_spec.n_mel_channels, long_skip_connection=True),
        is_tts_pretrain=cfg.model.is_tts_pretrain,
        melody_input_source=cfg.model.melody_input_source,
        cka_disabled=cfg.model.cka_disabled, num_channels=None,
        extra_parameters=cfg.extra_parameters, mel_spec_kwargs=cfg.model.mel_spec,
        distill_stage=None, use_guidance_scale_embed=False,
    )

    ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "ema_model_state_dict" in ckpt_data:
        sd = ckpt_data["ema_model_state_dict"]
        sd = {k.replace("ema_model.", ""): v for k, v in sd.items()}
    elif "model_state_dict" in ckpt_data:
        sd = ckpt_data["model_state_dict"]
    else:
        sd = ckpt_data
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    policy.load_state_dict(sd, strict=False)
    policy = policy.to(device).eval()

    vae = StableAudioInfer(model_config_path=vae_config, model_ckpt_path=vae_ckpt).to(device).eval()
    midi_teacher = MIDIExtractor(in_dim=80)
    midi_teacher._load_form_ckpt(midi_ckpt)
    midi_teacher = midi_teacher.to(device).eval()
    mel_extract = MelodySpectrogram()
    tokenizer = CNENTokenizer()

    return policy, vae, midi_teacher, mel_extract, tokenizer


def synthesize(policy, vae, midi_teacher, mel_extract, tokenizer,
               ref_audio, melody_audio, target_text,
               steps=32, cfg_strength=3.0, seed=42, device="cuda:0"):
    """
    核心合成函数。
    - ref_audio: 提供音色的参考音频路径
    - melody_audio: 提供旋律和时长的音频路径（默认为 ref_audio）
    - target_text: 目标歌词（仅 B 区生效）
    """
    if melody_audio is None:
        melody_audio = ref_audio

    # 1. 编码 ref_audio（带 0.5s 尾静音）
    ra, rs = torchaudio.load(ref_audio)
    silence = torch.zeros(ra.shape[0], int(rs * 0.5))
    rw = torch.cat([ra, silence], dim=1)
    rl = vae.encode_audio(rw, in_sr=rs).transpose(1, 2)  # [1, T_ref, 64]

    # 2. 编码 melody_audio（带 1.0s 尾静音）
    ma, ms = torchaudio.load(melody_audio)
    silence = torch.zeros(ma.shape[0], int(ms * 1.0))
    mw = torch.cat([ma, silence], dim=1)
    ml = vae.encode_audio(mw, in_sr=ms).transpose(1, 2)  # [1, T_mel, 64]

    midi_in = torch.cat([rl, ml], dim=1)
    rll = rl.shape[1]
    tl = rll + ml.shape[1]
    print(f"[info] ref_frames={rll}, melody_frames={ml.shape[1]}, total_frames={tl}")

    # 3. MIDI 特征（从 melody + ref 的 mel spectrogram）
    ref_mel = mel_extract(audio=rw, sr=rs)
    mel_mel = mel_extract(audio=mw, sr=ms)
    combined_mel = torch.cat([ref_mel, mel_mel], dim=2).to(device)
    with torch.no_grad():
        midi_p, bound_p = midi_teacher(combined_mel.transpose(1, 2))

    # 4. 歌词对齐
    lrc_token, _ = lrc_align.align_lrc_sentence_level(
        tokenizer=tokenizer, lrc_start_times=[0.0, rll / VFR],
        lrc_lines=["", target_text], total_lens=tl, vae_frame_rate=VFR,
    )
    text_tokens = torch.tensor(lrc_token, dtype=torch.int64).unsqueeze(0).to(device)

    # 5. Singer.sample()
    torch.manual_seed(seed)
    with torch.inference_mode():
        gen_latent, _ = policy.sample(
            cond=rl.to(device), text=text_tokens, duration=tl,
            midi_in=midi_in.to(device), midi_p=midi_p, bound_p=bound_p,
            steps=steps, cfg_strength=cfg_strength, guidance_scale=cfg_strength,
            t_shift=0.5, seed=seed, use_epss=False, enable_melody_control=True,
        )

    # 6. 提取 B 区（去掉 A 区和尾部静音）
    rear_frames = int(VFR * 1.0)
    b_lat = gen_latent[:, rll:-rear_frames, :]
    b_lat = b_lat.permute(0, 2, 1).float()
    wav = vae.decode_audio(b_lat).squeeze(0)[:1].reshape(-1).cpu()
    return wav, 44100


def main():
    parser = argparse.ArgumentParser(description="YingMusic-Singer-Plus 本地推理")
    parser.add_argument("--checkpoint", default="ckpts/plus_ja_sft_v4c/step_024000.pt")
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--melody_audio", default=None)
    parser.add_argument("--target_text", default="")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    policy, vae, midi_teacher, mel_extract, tokenizer = build_model(
        checkpoint=args.checkpoint, device=args.device,
    )

    wav, sr = synthesize(
        policy, vae, midi_teacher, mel_extract, tokenizer,
        ref_audio=args.ref_audio, melody_audio=args.melody_audio,
        target_text=args.target_text,
        steps=args.steps, cfg_strength=args.cfg, seed=args.seed,
        device=args.device,
    )

    torchaudio.save(args.output, wav.unsqueeze(0), sr)
    print(f"[done] {args.output} ({wav.shape[0] / sr:.1f}s)")


if __name__ == "__main__":
    main()
