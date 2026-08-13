"""
GRPO 训练主脚本: YingMusic-Singer-Plus V4c DiT + Flow-GRPO 精调。

四阶段:
  1. SDE 采样 G=8 候选 / prompt
  2. 四路 Reward 打分 (Whisper PER / WavLM SIM / torchcrepe F0 / DNSMOS)
  3. 组内标准化 → Advantage
  4. PPO-clip + KL 正则 → 策略梯度更新

用法:
  torchrun --nproc_per_node=4 train_grpo.py \
      --config src/YingMusicSinger/config/YingMusic_Singer.yaml \
      --ckpt_path ckpts/plus_ja_sft_v4c/step_024000.pt \
      --vae_config ... --vae_ckpt ... --midi_ckpt ... \
      --token_dir ${REMOTE_ROOT}/final_sum_large/pretreatment_text/timeset \
      --output_dir ckpts/plus_grpo_v1
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

# 项目路径
PROJECT_ROOT = "${REMOTE_ROOT}/YingMusic-Singer-Plus"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 本地模块
from grpo_utils import (
    compute_advantages,
    compute_ppo_ratio,
    compute_kl_divergence,
    build_sde_timesteps,
    sde_sample,
    compute_ref_len_grpo,
    FRAME_RATE,
)
from reward_models import RewardModels

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SDE_ALPHA = 0.8           # SDE 噪声强度
SDE_ALPHA_SQ = SDE_ALPHA ** 2
SDE_WINDOW_START = 1      # w_min
SDE_WINDOW_END = 9        # w_min + w_s = 9
TRAIN_NFE = 10            # Denoising Reduction
T_SHIFT = 0.5
G = 8                      # 组大小
BETA_KL = 1.0              # KL 正则强度
EPS_L = 0.002              # PPO clip 下界
EPS_U = 0.01               # PPO clip 上界
REWARD_WEIGHTS = torch.tensor([0.25, 0.15, 0.15, 0.45])  # PER, SIM, F0, DNSMOS

# ---------------------------------------------------------------------------
# DDP / 随机种子
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return local_rank, torch.distributed.get_world_size()


def is_main():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def collate_svs(batch):
    return {
        "wav": [item["wav"] for item in batch],
        "sr": [item["sr"] for item in batch],
        "phrases": [item["phrases"] for item in batch],
        "full_tokens": [item["full_tokens"] for item in batch],
        "tier": [item["tier"] for item in batch],
        "duration": [item["duration"] for item in batch],
        "kana": [item["kana"] for item in batch],
        "path": [item["path"] for item in batch],
    }


class GRPODataset(Dataset):
    """GRPO 专用数据集：加载 L1+L2 JSON + 在线 tokenize。"""

    def __init__(self, json_paths, tokenizer, max_duration_sec=30.0):
        self.records = []
        for path, tier in json_paths:
            if not os.path.exists(path):
                print(f"[GRPODataset] WARNING: {path} not found, skipping")
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                # 过滤：必须有至少 2 个短语和 kana
                phrases = item.get("Phrases", [])
                if len(phrases) < 2:
                    continue
                if not all("kana" in p for p in phrases):
                    continue
                item["_tier"] = tier
                self.records.append(item)

        self.tokenizer = tokenizer
        self.max_duration_sec = max_duration_sec

        # 缓存 tokenizer 的 SEP token
        self.sep_token = tokenizer.sep_token_id
        # 缓存全量 kana 文本（用于 PER reward）
        for rec in self.records:
            rec["_all_kana"] = " ".join([p["kana"] for p in rec["Phrases"]])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        wav, sr = torchaudio.load(rec["Path"])
        if sr != 44100:
            wav = torchaudio.functional.resample(wav, sr, 44100)
            sr = 44100
        max_samples = int(self.max_duration_sec * sr)
        if wav.shape[-1] > max_samples:
            wav = wav[:, :max_samples]

        # Tokenize phrases
        phrases = rec["Phrases"]
        tokens_flat = []
        for p in phrases:
            p_tokens = self.tokenizer.encode(p["text"])
            tokens_flat.extend(p_tokens)
            tokens_flat.append(self.sep_token)
        if tokens_flat:
            tokens_flat.pop()  # 去掉末尾 SEP

        return {
            "wav": wav,
            "sr": sr,
            "phrases": phrases,
            "full_tokens": tokens_flat,
            "tier": rec["_tier"],
            "duration": rec["Duration"],
            "kana": rec["_all_kana"],
            "path": rec["Path"],
        }


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO Training for YingMusic-Singer-Plus")
    parser.add_argument("--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml")
    parser.add_argument("--ckpt_path", required=True,
                        help="SFT checkpoint (V4c step_024000.pt)")
    parser.add_argument("--vae_config", default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json")
    parser.add_argument("--vae_ckpt", default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt")
    parser.add_argument("--midi_ckpt", default="ckpts/model_ckpt_steps_100000_simplified.ckpt")
    parser.add_argument("--token_dir", required=True,
                        help="Directory with train_L1_high.json / train_L2_medium.json")
    parser.add_argument("--output_dir", default="ckpts/plus_grpo_v1")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=7e-6)
    parser.add_argument("--max_steps", type=int, default=4800)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tmp_dir", default="/tmp/grpo_wavs",
                        help="临时 wav 输出目录")
    parser.add_argument("--save_baseline_1200", action="store_true", default=True,
                        help="额外保存 step_1200（论文基线）")
    args = parser.parse_args()

    # ---- DDP init ----
    if "LOCAL_RANK" in os.environ:
        local_rank, world_size = setup_ddp()
    else:
        local_rank, world_size = 0, 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    seed_everything(args.seed + local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if is_main():
        print(f"[GRPO] World={world_size} Device={device}")
        print(f"[GRPO] Batch={args.batch_size} G={G} Steps={args.max_steps} LR={args.lr}")
        print(f"[GRPO] Effective prompts/step: {args.batch_size * world_size}")
        print(f"[GRPO] Effective candidates/step: {args.batch_size * world_size * G}")

    cfg = OmegaConf.load(args.config)

    # ---- 导入项目模块 ----
    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
    from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer

    # ---- 初始化 DiT ----
    if is_main():
        print("[GRPO] Building DiT ...", flush=True)
    dit = DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
              mel_dim=cfg.model.mel_spec.n_mel_channels,
              long_skip_connection=True)

    policy = Singer(
        transformer=dit,
        is_tts_pretrain=cfg.model.is_tts_pretrain,
        melody_input_source=cfg.model.melody_input_source,
        cka_disabled=cfg.model.cka_disabled,
        num_channels=None,
        extra_parameters=cfg.extra_parameters,
        mel_spec_kwargs=cfg.model.mel_spec,
        distill_stage=None,
        use_guidance_scale_embed=False,
    )

    # ---- 加载 SFT checkpoint (policy + ref_model) ----
    if is_main():
        print(f"[GRPO] Loading SFT checkpoint: {args.ckpt_path}", flush=True)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "ema_model_state_dict" in ckpt:
        sd = ckpt["ema_model_state_dict"]
        sd = {k.replace("ema_model.", ""): v for k, v in sd.items()}
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    policy.load_state_dict(sd, strict=False)

    # 参考模型：深拷贝一份 SFT 权重并冻结
    ref_model = Singer(
        transformer=DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
                        mel_dim=cfg.model.mel_spec.n_mel_channels,
                        long_skip_connection=True),
        is_tts_pretrain=cfg.model.is_tts_pretrain,
        melody_input_source=cfg.model.melody_input_source,
        cka_disabled=cfg.model.cka_disabled,
        num_channels=None,
        extra_parameters=cfg.extra_parameters,
        mel_spec_kwargs=cfg.model.mel_spec,
        distill_stage=None,
        use_guidance_scale_embed=False,
    )
    ref_model.load_state_dict(sd, strict=False)
    ref_model = ref_model.to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    policy = policy.to(device)
    policy.train()

    if is_main():
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"[GRPO] DiT params: {n_params / 1e6:.1f}M", flush=True)

    # ---- VAE ----
    if is_main():
        print("[GRPO] Loading VAE ...", flush=True)
    vae = StableAudioInfer(model_config_path=args.vae_config, model_ckpt_path=args.vae_ckpt)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ---- SOME Teacher (MIDI) ----
    if is_main():
        print("[GRPO] Loading SOME MIDI teacher ...", flush=True)
    midi_teacher = MIDIExtractor(in_dim=80)
    midi_teacher._load_form_ckpt(args.midi_ckpt)
    midi_teacher = midi_teacher.to(device).eval()
    for p in midi_teacher.parameters():
        p.requires_grad = False

    mel_spec_extract = MelodySpectrogram()

    # ---- Tokenizer ----
    tokenizer = CNENTokenizer()
    SEP_TOKEN_ID = tokenizer.sep_token_id

    # ---- DDP wrap ----
    if world_size > 1:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = policy.module if world_size > 1 else policy
    raw_dit = raw_model.transformer  # DiT 本体

    # 暴露为模块级变量，供 save_ckpt 使用
    global _g_raw_model, _g_optimizer, _g_args, _g_output_dir
    _g_raw_model = raw_model
    _g_output_dir = args.output_dir

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=1e-2)
    _g_optimizer = optimizer
    _g_args = args

    # ---- Dataset ----
    train_json_paths = [
        (os.path.join(args.token_dir, "train_L1_high.json"), "L1"),
        (os.path.join(args.token_dir, "train_L2_medium.json"), "L2"),
    ]
    train_dataset = GRPODataset(train_json_paths, tokenizer, args.max_duration)
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, collate_fn=collate_svs,
        pin_memory=True, drop_last=True,
    )

    if is_main():
        print(f"[GRPO] Train samples: {len(train_dataset)} | Steps/epoch: {len(train_loader)}")
        print(f"[GRPO] Tokenizer SEP={SEP_TOKEN_ID}")
        print("=" * 60, flush=True)

    # ---- Reward 模型（本地 GPU，int8 Whisper） ----
    reward = RewardModels(device=device)
    reward.load_all()  # 只加载 WavLM，Whisper 延迟到打分时加载

    # ---- 临时音频目录 ----
    tmp_wav_dir = os.path.join(args.tmp_dir, f"rank{local_rank}")
    os.makedirs(tmp_wav_dir, exist_ok=True)

    # ---- 输出目录 ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 时间步调度（固定，所有采样复用） ----
    t_schedule = build_sde_timesteps(TRAIN_NFE, t_shift=T_SHIFT, device=device)

    # ---- AB 区准备函数 ----
    def prepare_ab_region(wav, sr, phrases, tier, full_latent, T):
        """构建 AB 区的 cond / midi / text，返回 (cond, midi, aligned_text, ref_len, ref_wav_16k)。"""
        # ref_len: 随机短语边界
        if tier in ("L1", "L2"):
            phrase_boundaries = [int(p["start"] * FRAME_RATE) for p in phrases
                                 if "start" in p]
        else:
            phrase_boundaries = []

        # 去重排序
        phrase_boundaries = sorted(set(b for b in phrase_boundaries if 0 < b < T))
        ref_len = compute_ref_len_grpo(T, phrase_boundaries)

        # cond: A 区 = GT latent, B 区 = 0
        cond = torch.zeros_like(full_latent)
        cond[:, :ref_len, :] = full_latent[:, :ref_len, :]

        # midi: A 区 = 0, B 区 = GT 旋律
        mel = mel_spec_extract(audio=wav.unsqueeze(0) if wav.dim() == 1 else wav, sr=44100).to(device)
        midi_p, _ = midi_teacher(mel.transpose(1, 2))
        if midi_p.shape[1] != T:
            midi_p = F.interpolate(midi_p.transpose(1, 2), size=T,
                                   mode="linear", align_corners=False).transpose(1, 2)
        midi = raw_model.smoothMelody_MIDIFuzzDisturb(midi_p)
        midi[:, :ref_len, :] = 0.0

        # text alignment
        if tier in ("L1", "L2"):
            aligned_text = _align_text_with_timestamps(phrases, ref_len, T)
        else:
            aligned_text = _uniform_align_text(phrases, ref_len, T)
        aligned_text = aligned_text.to(device)

        # ref audio 16kHz（用于 WavLM SIM + torchcrepe F0）
        wav_16k = _to_16k(wav, sr)

        return cond, midi, aligned_text, ref_len, wav_16k

    def _to_16k(wav, sr):
        """转换音频到 16kHz mono。"""
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
        return wav

    def _align_text_with_timestamps(phrases, ref_len, T):
        """基于时间戳的文本对齐（L1/L2）。"""
        aligned_text = torch.zeros(1, T, dtype=torch.long)
        for p in phrases:
            phrase_tokens = list(p.get("tokens", tokenizer.encode(p["text"])))
            phrase_tokens.append(SEP_TOKEN_ID)
            start_frame = int(p["start"] * FRAME_RATE)
            end_frame = int(p["end"] * FRAME_RATE)
            center_frame = int((p["start"] + p["end"]) / 2 * FRAME_RATE)
            if center_frame < ref_len:
                pos = min(max(start_frame, 0), ref_len - 1)
                for j, tid in enumerate(phrase_tokens):
                    if pos + j < ref_len:
                        aligned_text[0, pos + j] = tid
            else:
                pos = max(min(start_frame, T - 1), ref_len)
                for j, tid in enumerate(phrase_tokens):
                    if pos + j < T:
                        aligned_text[0, pos + j] = tid
        return aligned_text

    def _uniform_align_text(phrases, ref_len, T):
        """均匀分布文本对齐（L3 fallback）。"""
        flat = []
        for p in phrases:
            p_tokens = list(p.get("tokens", tokenizer.encode(p["text"])))
            flat.extend(p_tokens)
            flat.append(SEP_TOKEN_ID)
        if flat:
            flat.pop()
        aligned_text = torch.zeros(1, T, dtype=torch.long)
        if len(flat) == 0:
            return aligned_text
        split_idx = int(len(flat) * ref_len / T)
        a_tokens = flat[:split_idx]
        b_tokens = flat[split_idx:]
        a_step = ref_len / max(len(a_tokens), 1)
        for j, tid in enumerate(a_tokens):
            aligned_text[0, min(int(j * a_step), ref_len - 1)] = tid
        b_len = T - ref_len
        b_step = b_len / max(len(b_tokens), 1)
        for j, tid in enumerate(b_tokens):
            aligned_text[0, min(ref_len + int(j * b_step), T - 1)] = tid
        return aligned_text

    @torch.no_grad()
    def vae_decode_to_wav(latent):
        """VAE decode latent -> 波形，返回 44.1kHz 1D tensor [samples]"""
        # latent: [T, D] -> [1, D, T]
        lat = latent.unsqueeze(0).permute(0, 2, 1).float()
        audio = vae.decode_audio(lat)  # [1, 2, T*2048] stereo
        audio = audio.squeeze(0)[:1]   # 取第一声道 [1, T*2048]
        audio = audio.reshape(-1).cpu()
        return audio

    # ---- 训练循环 ----
    global_step = 0
    t_start = time.time()
    data_iter = iter(train_loader)

    # 累积 loss 统计
    accum = defaultdict(float)

    while global_step < args.max_steps:
        optimizer.zero_grad()

        # ---- 加载 batch ----
        try:
            batch = next(data_iter)
        except StopIteration:
            if train_sampler:
                train_sampler.set_epoch(train_sampler.epoch + 1)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        wavs = [b.to(device) for b in batch["wav"]]
        srs = batch["sr"]
        phrases_list = batch["phrases"]
        tiers = batch["tier"]
        kanas = batch["kana"]
        paths = batch["path"]
        B = len(wavs)  # batch_size

        # ---- 预处理：VAE encode + AB 区准备 ----
        prompt_data = []  # list of dict per prompt
        with torch.no_grad():
            for b_idx in range(B):
                w = wavs[b_idx]
                sr = srs[b_idx]
                w_2d = w.unsqueeze(0) if w.dim() == 1 else w
                latent = vae.encode_audio(w_2d, in_sr=sr)
                full_latent = latent.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, T, D]
                _, T, D = full_latent.shape

                cond, midi, aligned_text, ref_len, ref_wav_16k = prepare_ab_region(
                    w, sr, phrases_list[b_idx], tiers[b_idx], full_latent, T
                )

                prompt_data.append({
                    "full_latent": full_latent,   # [1, T, D]
                    "cond": cond,                  # [1, T, D]
                    "midi": midi,                  # [1, T, 128]
                    "aligned_text": aligned_text,  # [1, T]
                    "ref_len": ref_len,
                    "T": T,
                    "ref_wav_path": paths[b_idx],      # 原音频文件路径
                    "ref_wav_16k": ref_wav_16k,   # [samples_16k]
                    "kana": kanas[b_idx],
                    "wav_44k": w_2d,              # 原始 44.1kHz
                })

        # ---- 阶段1: 采样 G=8 候选 / prompt ----
        t_stage1 = time.time()
        all_candidates = []  # 扁平存储所有候选

        for p_idx, p_data in enumerate(prompt_data):
            cond = p_data["cond"]
            midi = p_data["midi"]
            text = p_data["aligned_text"]
            T_val = p_data["T"]
            D_val = cond.shape[-1]

            for g_idx in range(G):
                # 初始噪声（每个候选独立）
                x0 = torch.randn(1, T_val, D_val, device=device, dtype=cond.dtype)

                # SDE 采样
                x_final, sde_traj = sde_sample(
                    raw_dit, x0, cond, text, midi, t_schedule,
                    sde_window_start=SDE_WINDOW_START,
                    sde_window_end=SDE_WINDOW_END,
                    alpha=SDE_ALPHA,
                )

                # 裁剪 B 区 latent + VAE decode
                ref_len = p_data["ref_len"]
                b_latent = x_final[0, ref_len:, :]  # [T_b, D]

                # VAE decode → wav
                wav_gen = vae_decode_to_wav(b_latent)

                # 保存临时 wav（16kHz，用于 reward 模型）
                wav_16k = torchaudio.functional.resample(
                    wav_gen.unsqueeze(0), 44100, 16000
                ).squeeze(0)

                tmp_path = os.path.join(
                    tmp_wav_dir, f"step{global_step:06d}_p{b_idx:02d}_g{g_idx:02d}.wav"
                )
                torchaudio.save(tmp_path, wav_gen.unsqueeze(0), 44100)

                all_candidates.append({
                    "tmp_path": tmp_path,
                    "wav_16k": wav_16k,
                    "sde_traj": sde_traj,       # list of SDE step dicts
                    "cond": cond,
                    "text": text,
                    "midi": midi,
                    "prompt_idx": p_idx,  # 组索引
                    "ref_wav_path": p_data["ref_wav_path"],  # 原音频路径（供 reward server）
                    "ref_wav_16k": p_data["ref_wav_16k"],
                    "kana": p_data["kana"],
                    "ref_len": ref_len,
                })

        assert len(all_candidates) == B * G
        t_stage1 = time.time() - t_stage1

        # ---- 阶段2: Reward 打分（本地并行，optimizer→CPU 腾显存） ----
        t_stage2 = time.time()

        # 1) 保存 optimizer 到 CPU + 删除
        opt_cpu = optimizer.state_dict()
        if isinstance(opt_cpu, dict):
            for s in opt_cpu.get("state", {}).values():
                for k in list(s.keys()):
                    if isinstance(s[k], torch.Tensor):
                        s[k] = s[k].cpu()
        del optimizer
        # 2) ref_model 移到 CPU
        ref_model.to("cpu")
        torch.cuda.empty_cache()

        # 3) 加载 Whisper + 打分
        reward.ensure_whisper()
        scores = reward.score_all(
            [c["tmp_path"] for c in all_candidates],
            [c["ref_wav_16k"] for c in all_candidates],
            [c["kana"] for c in all_candidates],
        )  # [B*G, 4]

        # 4) 释放 Whisper，重建 optimizer，ref_model 移回
        reward.free_whisper()
        torch.cuda.empty_cache()
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.95), weight_decay=1e-2)
        optimizer.load_state_dict(opt_cpu)
        del opt_cpu
        ref_model.to(device)
        torch.cuda.empty_cache()

        # ---- 阶段3: 组内标准化 → Advantage ----
        t_stage2 = time.time() - t_stage2
        t_stage3 = time.time()
        all_scores_local = scores.to(device)  # [B*G, 4]
        all_advantages = []

        for p_idx in range(B):
            g_start = p_idx * G
            g_end = g_start + G
            group_scores = all_scores_local[g_start:g_end]  # [G, 4]
            adv = compute_advantages(group_scores, REWARD_WEIGHTS)  # [G]
            all_advantages.append(adv)
        advantages = torch.cat(all_advantages)  # [B*G]

        # ---- 阶段4: 策略梯度更新（分步 backward，避免显存爆炸） ----
        t_stage3 = time.time() - t_stage3
        t_stage4 = time.time()
        # 总归一化系数：B * G * |S| = 6 * 8 * 8 = 384
        loss_scale = 1.0 / (B * G * len(all_candidates[0]["sde_traj"]))

        L_clip_sum = 0.0
        KL_sum = 0.0
        n_loss_terms = 0

        for cand_idx, cand in enumerate(all_candidates):
            A = advantages[cand_idx]
            cond = cand["cond"]
            text = cand["text"]
            midi = cand["midi"]

            for traj in cand["sde_traj"]:
                x_t = traj["x_t"]          # [1, T, D]
                x_next = traj["x_next"]    # [1, T, D]
                v_old = traj["v_pred"]     # [1, T, D] — 采样时的策略输出
                dt = traj["dt"]
                t_now = traj["t_now"]

                # v_eff = (x_next - x_t) / dt
                v_eff = (x_next - x_t) / dt  # [1, T, D]

                # 当前策略预测 v_new（需要梯度）
                v_new, _ = raw_dit(
                    x=x_t, cond=cond, text=text, time=t_now, midi=midi,
                    drop_audio_cond=False, drop_text=False, drop_midi=False,
                    cfg_infer=False, cache=False,
                )

                # 参考模型预测 v_ref（无需梯度）
                with torch.no_grad():
                    v_ref, _ = ref_model.transformer(
                        x=x_t, cond=cond, text=text, time=t_now, midi=midi,
                        drop_audio_cond=False, drop_text=False, drop_midi=False,
                        cfg_infer=False, cache=False,
                    )

                # PPO-clip
                ratio, ratio_clipped = compute_ppo_ratio(
                    v_eff, v_new, v_old, dt, alpha_sq=SDE_ALPHA_SQ,
                    eps_l=EPS_L, eps_u=EPS_U,
                )
                L_clip = torch.min(ratio * A, ratio_clipped * A)  # [1, T]

                # KL 正则
                kl_val = compute_kl_divergence(
                    v_new, v_ref, dt, alpha_sq=SDE_ALPHA_SQ
                )  # [1, T]

                # 单步 loss
                step_loss = (-L_clip.mean() + BETA_KL * kl_val.mean()) * loss_scale
                step_loss.backward()

                L_clip_sum += L_clip.mean().item()
                KL_sum += kl_val.mean().item()
                n_loss_terms += 1

        L_clip_mean = L_clip_sum / max(n_loss_terms, 1)
        KL_mean = KL_sum / max(n_loss_terms, 1)

        # Grad clip + optimizer step
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        global_step += 1
        t_stage4 = time.time() - t_stage4

        # 每步打印耗时（rank 0）
        if is_main():
            total_step = t_stage1 + t_stage2 + t_stage3 + t_stage4
            print(f"  [Step {global_step:4d}] "
                  f"S1_sample={t_stage1:.0f}s S2_reward={t_stage2:.0f}s "
                  f"S3_adv={t_stage3:.1f}s S4_grad={t_stage4:.1f}s "
                  f"Total={total_step:.0f}s", flush=True)

        # ---- 统计 ----
        with torch.no_grad():
            mean_raw_reward = all_scores_local.mean(dim=0)
            mean_adv = advantages.mean()
            mean_per = mean_raw_reward[0].item()
            mean_sim = mean_raw_reward[1].item()
            mean_f0 = mean_raw_reward[2].item()
            mean_dns = mean_raw_reward[3].item()

        accum["L_clip"] += L_clip_mean
        accum["KL"] += KL_mean
        accum["Loss"] += (-L_clip_mean + BETA_KL * KL_mean)
        accum["Adv"] += mean_adv.item()
        accum["PER"] += mean_per
        accum["SIM"] += mean_sim
        accum["F0"] += mean_f0
        accum["DNS"] += mean_dns

        # ---- 清理临时文件 ----
        for cand in all_candidates:
            try:
                os.remove(cand["tmp_path"])
            except OSError:
                pass

        # ---- 日志 ----
        if global_step % args.log_every == 0 and is_main():
            n = args.log_every
            avg_clip = accum["L_clip"] / n
            avg_kl = accum["KL"] / n
            avg_loss = accum["Loss"] / n
            avg_adv = accum["Adv"] / n
            avg_per = accum["PER"] / n
            avg_sim = accum["SIM"] / n
            avg_f0 = accum["F0"] / n
            avg_dns = accum["DNS"] / n

            elapsed = time.time() - t_start
            sec_per_step = elapsed / max(global_step, 1)
            remaining = args.max_steps - global_step
            eta = remaining * sec_per_step
            if eta > 3600:
                eta_str = f"{eta/3600:.1f}h"
            elif eta > 60:
                eta_str = f"{eta/60:.1f}m"
            else:
                eta_str = f"{eta:.0f}s"

            print(
                f"  [{global_step:5d}/{args.max_steps} | "
                f"{100*global_step/args.max_steps:5.1f}% | "
                f"ETA {eta_str} | {sec_per_step:.1f}s/step] "
                f"Loss={avg_loss:.4f} Clip={avg_clip:.4f} KL={avg_kl:.4f} "
                f"Adv={avg_adv:+.4f} "
                f"PER={avg_per:.3f} SIM={avg_sim:.3f} F0={avg_f0:.3f} DNS={avg_dns:.3f}\n"
                f"  [Timing] S1(sample)={t_stage1:.0f}s S2(reward)={t_stage2:.0f}s "
                f"S3(adv)={t_stage3:.1f}s S4(grad)={t_stage4:.1f}s",
                flush=True,
            )

            for k in accum:
                accum[k] = 0.0

        # ---- Checkpoint ----
        if global_step % args.save_every == 0 and is_main():
            save_ckpt(global_step)
        if args.save_baseline_1200 and global_step == 1200 and is_main():
            save_ckpt(global_step)

        # 每 50 步清理显存
        if global_step % 50 == 0:
            torch.cuda.empty_cache()

    # ---- 最终保存 ----
    if is_main():
        save_ckpt(global_step, final=True)
        print(f"[GRPO] Training complete! ({global_step} steps)", flush=True)
        print(f"[GRPO] Checkpoints saved to: {args.output_dir}", flush=True)


def save_ckpt(step, final=False):
    """保存 checkpoint。"""
    path = os.path.join(_g_output_dir, f"step_{step:06d}{'_final' if final else ''}.pt")
    torch.save({
        "model_state_dict": _g_raw_model.state_dict(),
        "optimizer_state_dict": _g_optimizer.state_dict(),
        "global_step": step,
    }, path)
    print(f"  [Save {step:5d}] {path}", flush=True)


# 模块级全局变量（在 main() 中赋值）
_g_raw_model = None
_g_optimizer = None
_g_args = None
_g_output_dir = None


if __name__ == "__main__":
    main()

