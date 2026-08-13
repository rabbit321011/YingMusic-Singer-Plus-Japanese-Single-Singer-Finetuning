"""
GRPO v3 (remote Whisper on GPU set) 训练主脚本 — 基于 Singer.sample() 管线。

四阶段:
  1. Singer.sample(CFG=3, 32步) 生成 G=8 候选
  2. 四路 Reward 打分 (PER/SIM/F0/DNS)
  3. 组内标准化 → Advantage
  4. 轨迹 KL + PPO-clip → 策略梯度更新

与 v1 的区别:
  - Stage 1 用 Singer.sample()（和推理完全一致的预处理）
  - CFG=3, 32 步 Euler（推理同款）
  - 文本对齐: lrc_align.sentence_level
  - 样本: 只取短样本 (<15s)

用法:
  torchrun --nproc_per_node=4 train_grpo_v3.py \
      --token_dir ${REMOTE_ROOT}/final_sum_large/pretreatment_text/timeset \
      --output_dir ckpts/plus_grpo_v3
"""

import argparse, json, os, random, sys, time, copy
from collections import defaultdict

import numpy as np
import torch, torchaudio, torch.nn.functional as F
# Fix torch 2.6 weights_only issue
torch.load_orig = torch.load
def _load(*a, **kw):
    kw.setdefault("weights_only", False)
    return torch.load_orig(*a, **kw)
torch.load = _load

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

PROJECT_ROOT = "${REMOTE_ROOT}/YingMusic-Singer-Plus"
GRPO_DIR = "${REMOTE_ROOT}/YingMusic-Singer-Plus/scripts_archive/yingmusic_plus/3_train_grpo"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if GRPO_DIR not in sys.path:
    sys.path.insert(0, GRPO_DIR)

from reward_models import RewardModels
from grpo_utils import build_sde_timesteps

# ---------------------------------------------------------------------------
# 超参
# ---------------------------------------------------------------------------
G = 8                      # 组大小
BETA_KL = 1.0
EPS_L = 0.002
EPS_U = 0.01
REWARD_WEIGHTS = torch.tensor([6.0, 1.0, 1.0, 2.0])  # PER 60%, SIM 10%, F0 10%, DNS 20%
VFR = 44100 / 2048          # VAE frame rate
MAX_DURATION_SEC = 15.0     # 只取短样本
TRAIN_STEPS = 32            # Singer.sample 步数
CFG_STRENGTH = 3.0

# ---------------------------------------------------------------------------
# DDP / 种子
# ---------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_world_size()

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GRPODatasetV2(Dataset):
    """短样本数据集: <15s, L1+L2, 至少2个短语"""
    def __init__(self, json_paths):
        self.records = []
        for path, tier in json_paths:
            if not os.path.exists(path): continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if item.get("Duration", 999) > MAX_DURATION_SEC: continue
                phrases = item.get("Phrases", [])
                if len(phrases) < 2: continue
                if not all("kana" in p for p in phrases): continue
                item["_tier"] = tier
                item["_all_kana"] = " ".join([p["kana"] for p in phrases])
                self.records.append(item)
        print(f"[GRPODatasetV3] {len(self.records)} records (≤{MAX_DURATION_SEC}s)")

    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        wav, sr = torchaudio.load(rec["Path"])
        if sr != 44100:
            wav = torchaudio.functional.resample(wav, sr, 44100); sr = 44100
        return {"wav": wav, "sr": sr, "phrases": rec["Phrases"],
                "tier": rec["_tier"], "duration": rec["Duration"],
                "kana": rec["_all_kana"], "path": rec["Path"]}

def collate_v2(batch):
    return {"wav": [b["wav"] for b in batch], "sr": [b["sr"] for b in batch],
            "phrases": [b["phrases"] for b in batch], "tier": [b["tier"] for b in batch],
            "duration": [b["duration"] for b in batch], "kana": [b["kana"] for b in batch],
            "path": [b["path"] for b in batch]}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def compute_advantages(scores: torch.Tensor, weights: torch.Tensor):
    """组内 z-score → advantage [G]"""
    raw = (scores * weights).sum(dim=1)
    mean, std = raw.mean(), raw.std() + 1e-8
    return (raw - mean) / std

def compute_ppo_ratio(v_eff, v_new, v_old, dt, alpha_sq, eps_l, eps_u):
    """PDF ratio + clip"""
    dt_safe = max(dt, 1e-8)
    diff_new = ((v_eff - v_new) ** 2).sum(dim=-1)
    diff_old = ((v_eff - v_old) ** 2).sum(dim=-1)
    log_ratio = (-diff_new + diff_old) / (2.0 * alpha_sq * dt_safe)
    log_ratio = torch.clamp(log_ratio, -5.0, 1.0)  # Prevent NaN/overflow
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - eps_l, 1 + eps_u)
    return ratio, clipped

def compute_kl(v_new, v_ref, dt, alpha_sq):
    dt_safe = max(dt, 1e-8)
    diff = ((v_new - v_ref) ** 2).sum(dim=-1)
    return dt_safe * diff / (2.0 * alpha_sq)

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml")
    parser.add_argument("--ckpt_path", default="ckpts/plus_ja_sft_v4c/step_024000.pt")
    parser.add_argument("--vae_config", default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json")
    parser.add_argument("--vae_ckpt", default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt")
    parser.add_argument("--midi_ckpt", default="ckpts/model_ckpt_steps_100000_simplified.ckpt")
    parser.add_argument("--token_dir", required=True)
    parser.add_argument("--output_dir", default="ckpts/plus_grpo_v3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--max_steps", type=int, default=12000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tmp_dir", default="/tmp/grpo_v2_wavs")
    args = parser.parse_args()

    # ---- DDP ----
    if "LOCAL_RANK" in os.environ:
        local_rank, world_size = setup_ddp()
    else:
        local_rank, world_size = 0, 1; torch.cuda.set_device(0)
    seed_everything(args.seed + local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if is_main():
        print(f"[GRPOv3] World={world_size} Batch={args.batch_size} G={G} Steps={args.max_steps}")
        print(f"[GRPOv3] Candidates/step: {args.batch_size * world_size * G}")
        print(f"[GRPOv3] Max duration: {MAX_DURATION_SEC}s")

    cfg = OmegaConf.load(args.config)

    # ---- 模型加载 ----
    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer, interpolation_midi_continuous
    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
    from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
    from src.YingMusicSinger.utils import lrc_align
    from src.YingMusicSinger.utils.common import lens_to_mask

    def build_policy():
        p = Singer(
            transformer=DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
                            mel_dim=cfg.model.mel_spec.n_mel_channels, long_skip_connection=True),
            is_tts_pretrain=cfg.model.is_tts_pretrain,
            melody_input_source=cfg.model.melody_input_source,
            cka_disabled=cfg.model.cka_disabled, num_channels=None,
            extra_parameters=cfg.extra_parameters, mel_spec_kwargs=cfg.model.mel_spec,
            distill_stage=None, use_guidance_scale_embed=False,
        )
        return p

    if is_main(): print("[GRPOv3] Loading models...", flush=True)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "ema_model_state_dict" in ckpt:
        sd = ckpt["ema_model_state_dict"]; sd = {k.replace("ema_model.", ""): v for k, v in sd.items()}
    elif "model_state_dict" in ckpt: sd = ckpt["model_state_dict"]
    else: sd = ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    policy = build_policy(); policy.load_state_dict(sd, strict=False); policy = policy.to(device)
    ref_model = build_policy(); ref_model.load_state_dict(sd, strict=False); ref_model = ref_model.to(device).eval()
    for p in ref_model.parameters(): p.requires_grad = False
    policy.train()

    vae = StableAudioInfer(model_config_path=args.vae_config, model_ckpt_path=args.vae_ckpt).to(device).eval()
    midi_teacher = MIDIExtractor(in_dim=80); midi_teacher._load_form_ckpt(args.midi_ckpt); midi_teacher = midi_teacher.to(device).eval()
    mel_extract = MelodySpectrogram()
    tokenizer = CNENTokenizer()

    if world_size > 1:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = policy.module if world_size > 1 else policy
    raw_dit = raw_model.transformer

    if is_main():
        n_p = sum(p.numel() for p in raw_model.parameters())
        print(f"[GRPOv3] DiT params: {n_p/1e6:.1f}M", flush=True)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)

    # ---- Dataset ----
    train_json_paths = [
        (os.path.join(args.token_dir, "train_L1_high.json"), "L1"),
        (os.path.join(args.token_dir, "train_L2_medium.json"), "L2"),
    ]
    train_dataset = GRPODatasetV2(train_json_paths)
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=args.num_workers,
                              collate_fn=collate_v2, pin_memory=True, drop_last=True)
    if is_main():
        print(f"[GRPOv3] Train samples: {len(train_dataset)} | Steps/epoch: {len(train_loader)}")

    # ---- Reward 模型 ----
    reward = RewardModels(device=device, use_remote_whisper=True); reward.load_all()

    # ---- 临时目录 ----
    tmp_wav_dir = os.path.join(args.tmp_dir, f"rank{local_rank}")
    os.makedirs(tmp_wav_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 训练循环 ----
    t_schedule = build_sde_timesteps(TRAIN_STEPS, t_shift=0.5, device=device)
    global_step = 0; t_start = time.time()
    accum = defaultdict(float); data_iter = iter(train_loader)
    best_per = 0.0

    while global_step < args.max_steps:
        optimizer.zero_grad()

        # ---- Load batch ----
        try:
            batch = next(data_iter)
        except StopIteration:
            if train_sampler: train_sampler.set_epoch(train_sampler.epoch + 1)
            data_iter = iter(train_loader); batch = next(data_iter)
        B = len(batch["wav"])

        # ================================================================
        # 预处理: VAE encode + Singer.sample 输入准备
        # ================================================================
        prompt_data = []
        skipped = 0
        with torch.no_grad():
            for b_idx in range(B):
                try:
                    w_b = batch["wav"][b_idx].to(device)
                    if w_b.dim() == 1: w_b = w_b.unsqueeze(0)

                    # prepare_input style encode
                    rw = torch.cat([w_b, torch.zeros(w_b.shape[0], int(44100 * 0.5), device=device)], dim=1)
                    rl = vae.encode_audio(rw, in_sr=44100).transpose(1, 2)
                    mw = torch.cat([w_b, torch.zeros(w_b.shape[0], int(44100 * 1.0), device=device)], dim=1)
                    ml = vae.encode_audio(mw, in_sr=44100).transpose(1, 2)
                    midi_in = torch.cat([rl, ml], dim=1)
                    rll = rl.shape[1]; tl = rll + ml.shape[1]

                    # MIDI from combined mel
                    ref_mel = mel_extract(audio=rw, sr=44100)
                    mel_mel = mel_extract(audio=mw, sr=44100)
                    combined_mel = torch.cat([ref_mel, mel_mel], dim=2).to(device)
                    with torch.no_grad(): midi_p, bp = midi_teacher(combined_mel.transpose(1, 2))

                    # Text: A/B 区都用整首歌的 kana（自克隆模式，A/B 同曲）
                    kana_full = batch["kana"][b_idx]
                    lrc_token, _ = lrc_align.align_lrc_sentence_level(
                        tokenizer=tokenizer, lrc_start_times=[0.0, rll / VFR],
                        lrc_lines=[kana_full, kana_full], total_lens=tl, vae_frame_rate=VFR,
                    )
                    text_tokens = torch.tensor(lrc_token, dtype=torch.int64).unsqueeze(0)

                    prompt_data.append({
                        "cond": rl, "midi_in": midi_in, "text": text_tokens,
                        "midi_p": midi_p, "bp": bp, "rll": rll, "tl": tl,
                        "kana": batch["kana"][b_idx], "wav_44k": w_b,
                        "ref_wav": batch["wav"][b_idx],
                        "path": batch["path"][b_idx],
                    })
                except Exception as e:
                    skipped += 1
                    if is_main():
                        print(f"  [WARN] Skipping sample: {e}", flush=True)
                    continue  # skip to next batch element

        if skipped > 0 and is_main():
            print(f"  [WARN] Skipped {skipped}/{B} samples this step", flush=True)
        if len(prompt_data) == 0:
            continue  # all samples bad, skip this step entirely

        # ================================================================
        # Stage 1: Singer.sample() 生成 G=8 候选
        # ================================================================
        t_s1 = time.time()
        all_candidates = []

        for p_idx, pd in enumerate(prompt_data):
            rll = pd["rll"]; tl = pd["tl"]
            for g_idx in range(G):
                seed = (args.seed + global_step * 100 + p_idx * G + g_idx) % 100000
                with torch.inference_mode():
                    gen_latent, traj = raw_model.sample(
                        cond=pd["cond"].to(device), text=pd["text"].to(device), duration=tl,
                        midi_in=pd["midi_in"].to(device), midi_p=pd["midi_p"], bound_p=pd["bp"],
                        steps=TRAIN_STEPS, cfg_strength=CFG_STRENGTH, guidance_scale=CFG_STRENGTH,
                        t_shift=0.5, seed=seed, use_epss=False, enable_melody_control=True,
                    )
                rear = int(VFR * 1.0)
                b_lat = gen_latent[:, rll:-rear, :]
                gen_wav = vae.decode_audio(b_lat.permute(0, 2, 1).float()).squeeze(0)[:1].reshape(-1).cpu()
                tmp_path = os.path.join(tmp_wav_dir, f"step{global_step:06d}_p{p_idx:02d}_g{g_idx:02d}.wav")
                torchaudio.save(tmp_path, gen_wav.unsqueeze(0), 44100)

                all_candidates.append({
                    "tmp_path": tmp_path, "kana": pd["kana"],
                    "traj": traj,  # [nfe+1, 1, tl, 64] tensor
                    "cond": pd["cond"], "text": pd["text"], "midi": pd["midi_p"],
                    "bp": pd["bp"], "rll": rll, "tl": tl,
                    "ref_wav": pd["ref_wav"],
                })
        t_s1 = time.time() - t_s1
        assert len(all_candidates) == B * G

        # ================================================================
        # Stage 2: Reward 打分（optimizer+ref_model→CPU 腾显存给 Whisper）
        # ================================================================
        t_s2 = time.time()
        # optimizer → CPU
        opt_cpu = optimizer.state_dict()
        if isinstance(opt_cpu, dict):
            for s in opt_cpu.get("state", {}).values():
                for k in list(s.keys()):
                    if isinstance(s[k], torch.Tensor): s[k] = s[k].cpu()
        del optimizer
        # ref_model → CPU（仅在 Stage 4 需要 KL，Stage 2 期间释放）
        ref_model.cpu()
        torch.cuda.empty_cache()
        # 二次清理：确保 Whisper 有连续 3GB 块
        torch.cuda.empty_cache()

        ref_wavs_16k = []
        for c in all_candidates:
            rw_16k = torchaudio.functional.resample(c["ref_wav"], 44100, 16000).squeeze(0).cpu()
            ref_wavs_16k.append(rw_16k)
        scores = reward.score_all([c["tmp_path"] for c in all_candidates],
                                  ref_wavs_16k, [c["kana"] for c in all_candidates])

        # 恢复 ref_model → GPU
        ref_model.to(device)
        # 恢复 optimizer
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.95), weight_decay=1e-2)
        optimizer.load_state_dict(opt_cpu); del opt_cpu
        torch.cuda.empty_cache()
        t_s2 = time.time() - t_s2

        # ================================================================
        # Stage 3: Advantage
        # ================================================================
        t_s3 = time.time()
        scores_local = scores.to(device)
        all_advantages = []
        for p_idx in range(B):
            gs = scores_local[p_idx * G:(p_idx + 1) * G]
            all_advantages.append(compute_advantages(gs, REWARD_WEIGHTS.to(device)))
        advantages = torch.cat(all_advantages)
        t_s3 = time.time() - t_s3

        # ================================================================
        # Stage 4: Policy Gradient
        # ================================================================
        t_s4 = time.time()
        alpha_sq = 0.8 ** 2
        L_clip_sum = 0.0; KL_sum = 0.0; n_terms = 0

        # Pre-build Singer.sample inner fn inputs for KL computation
        # We need to replicate step_cond and midi_data (same as Singer.sample internal)
        for p_idx, pd in enumerate(prompt_data):
            # Build Singer.sample internal state for this prompt
            rll = pd["rll"]; tl = pd["tl"]
            cond_g = pd["cond"].to(device)
            text_g = pd["text"].to(device)

            # cond_mask + step_cond
            cond_mask = lens_to_mask(torch.tensor([rll], device=device, dtype=torch.long), length=tl)
            cond_g = F.pad(cond_g, (0, 0, 0, tl - rll), value=0.0)
            cond_mask = F.pad(cond_mask, (0, tl - cond_mask.shape[-1]), value=False).unsqueeze(-1)
            step_cond = torch.where(cond_mask, cond_g, torch.zeros_like(cond_g))

            # midi (replicate Singer.sample internal)
            midi_d, _ = interpolation_midi_continuous(midi_p=pd["midi_p"], bound_p=pd["bp"], total_len=text_g.shape[1])
            midi_d = raw_model.smoothMelody_MIDIFuzzDisturb(midi_d).to(device)
            midi_d = torch.where(cond_mask, torch.zeros_like(midi_d), midi_d)

            # For each candidate in this group
            for g_idx in range(G):
                cand = all_candidates[p_idx * G + g_idx]
                A_i = advantages[p_idx * G + g_idx]
                traj = cand["traj"]  # [nfe+1, 1, tl, 64]

                for i in range(len(t_schedule) - 1):
                    dt = float(t_schedule[i + 1] - t_schedule[i])
                    x_t = traj[i]
                    x_next = traj[i + 1]
                    t_now = t_schedule[i]

                    # v_policy from trajectory
                    v_policy = (x_next - x_t) / dt

                    # v_new from current policy
                    v_new, _ = raw_dit(x=x_t, cond=step_cond, text=text_g, time=t_now,
                                       midi=midi_d, cfg_infer=True, cache=False,
                                       cfg_infer_ids=(True, False, False, True))
                    vp_n, vu_n = torch.chunk(v_new, 2, dim=0)
                    v_new_cfg = vp_n + (vp_n - vu_n) * CFG_STRENGTH

                    # v_old from frozen ref model (this is the "old" policy)
                    with torch.no_grad():
                        v_o, _ = ref_model.transformer(
                            x=x_t, cond=step_cond, text=text_g, time=t_now,
                            midi=midi_d, cfg_infer=True, cache=False,
                            cfg_infer_ids=(True, False, False, True),
                        )
                        vp_o, vu_o = torch.chunk(v_o, 2, dim=0)
                        v_old_cfg = vp_o + (vp_o - vu_o) * CFG_STRENGTH

                    ratio, ratio_clipped = compute_ppo_ratio(v_policy, v_new_cfg, v_old_cfg, dt, alpha_sq, EPS_L, EPS_U)
                    L_clip = torch.min(ratio * A_i, ratio_clipped * A_i)
                    kl_val = compute_kl(v_new_cfg, v_old_cfg, dt, alpha_sq)

                    # B region only
                    L_clip_b = L_clip[:, rll:].mean()
                    kl_b = kl_val[:, rll:].mean()

                    loss_scale = 1.0 / (B * G * (len(t_schedule) - 1))
                    step_loss = (-L_clip_b + BETA_KL * kl_b) * loss_scale
                    step_loss.backward()

                    L_clip_sum += L_clip_b.item(); KL_sum += kl_b.item(); n_terms += 1

        L_clip_mean = L_clip_sum / max(n_terms, 1)
        KL_mean = KL_sum / max(n_terms, 1)

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        global_step += 1
        t_s4 = time.time() - t_s4

        # ---- 统计 ----
        with torch.no_grad():
            mean_scores = scores_local.mean(dim=0)
            mean_per, mean_sim, mean_f0, mean_dns = [mean_scores[i].item() for i in range(4)]
            # 组内极差：raw weighted score 的 (max-min) 平均
            group_ranges = []
            for p_idx in range(B):
                gs = scores_local[p_idx * G:(p_idx + 1) * G]
                raw = gs @ REWARD_WEIGHTS.to(device)
                group_ranges.append((raw.max() - raw.min()).item())
            adv_range = sum(group_ranges) / len(group_ranges)  # 平均组内极差
        accum["L_clip"] += L_clip_mean; accum["KL"] += KL_mean
        accum["Loss"] += (-L_clip_mean + BETA_KL * KL_mean)
        accum["Adv"] += adv_range; accum["PER"] += mean_per
        accum["SIM"] += mean_sim; accum["F0"] += mean_f0; accum["DNS"] += mean_dns

        # ---- 清理临时文件 + 释放 S4 traj 显存 ----
        for c in all_candidates:
            try: os.remove(c["tmp_path"])
            except OSError: pass
        del all_candidates
        torch.cuda.empty_cache()
        # Whisper on remote GPU set — no local reload needed

        # ---- Step timing + 溯源 ----
        if is_main():
            total = t_s1 + t_s2 + t_s3 + t_s4
            print(f"  [Step {global_step:4d}] S1:{t_s1:.0f}s S2:{t_s2:.0f}s S3:{t_s3:.1f}s S4:{t_s4:.1f}s Total:{total:.0f}s",
                  flush=True)
            for p_idx, pd in enumerate(prompt_data):
                dur_a = pd["rll"] / VFR
                dur_b = (pd["tl"] - pd["rll"]) / VFR
                kana_snip = pd["kana"][:30] + ("..." if len(pd["kana"]) > 30 else "")
                fname = os.path.basename(pd["path"])
                # 该 prompt 的 G=8 个候选 PER
                gs = scores_local[p_idx * G:(p_idx + 1) * G]
                per_g = gs[:, 0]  # PER 列
                per_str = ", ".join([f"{x:.2f}" for x in per_g.tolist()])
                print(f"    Prompt{p_idx}: A={dur_a:.1f}s B={dur_b:.1f}s | {kana_snip} | {fname}",
                      flush=True)
                print(f"      PER=[{per_str}]", flush=True)

        # ---- 日志 ----
        if global_step % args.log_every == 0 and is_main():
            n = args.log_every
            avg_clip = accum["L_clip"] / n; avg_kl = accum["KL"] / n
            avg_adv = accum["Adv"] / n; avg_per = accum["PER"] / n
            avg_sim = accum["SIM"] / n; avg_f0 = accum["F0"] / n; avg_dns = accum["DNS"] / n
            elapsed = time.time() - t_start
            sps = elapsed / max(global_step, 1)
            eta = (args.max_steps - global_step) * sps
            eta_s = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"

            print(f"  [{global_step:5d}/{args.max_steps} | {100*global_step/args.max_steps:5.1f}% | "
                  f"ETA {eta_s} | {sps:.1f}s/step]", flush=True)
            print(f"  Loss={avg_clip+avg_kl*BETA_KL:.4f} Clip={avg_clip:.4f} KL={avg_kl:.4f} "
                  f"Adv_rng={avg_adv:.3f} PER={avg_per:.3f} SIM={avg_sim:.3f} F0={avg_f0:.3f} DNS={avg_dns:.3f}",
                  flush=True)
            for k in accum: accum[k] = 0.0

        # ---- Eval (每 save_every 步跑一条) ----
        if global_step % args.save_every == 0 and is_main():
            print(f"  [Eval] Step {global_step} — running quick eval...", flush=True)
            # 简单 eval: 跑一条样本看 PER
            eval_sample = train_dataset[0]
            with torch.inference_mode():
                rw_ev = torch.cat([eval_sample["wav"], torch.zeros(eval_sample["wav"].shape[0], int(44100*0.5))], dim=1)
                rl_ev = vae.encode_audio(rw_ev, in_sr=44100).transpose(1,2)
                mw_ev = torch.cat([eval_sample["wav"], torch.zeros(eval_sample["wav"].shape[0], int(44100*1.0))], dim=1)
                ml_ev = vae.encode_audio(mw_ev, in_sr=44100).transpose(1,2)
                mi_in_ev = torch.cat([rl_ev, ml_ev], dim=1)
                rll_ev = rl_ev.shape[1]; tl_ev = rll_ev + ml_ev.shape[1]
                rm_ev = mel_extract(audio=rw_ev, sr=44100)
                mm_ev = mel_extract(audio=mw_ev, sr=44100)
                cm_ev = torch.cat([rm_ev, mm_ev], dim=2).to(device)
                midi_p_ev, bp_ev = midi_teacher(cm_ev.transpose(1,2))
                lrc_ev, _ = lrc_align.align_lrc_sentence_level(
                    tokenizer=tokenizer, lrc_start_times=[0.0, rll_ev/VFR],
                    lrc_lines=[eval_sample["kana"], eval_sample["kana"]], total_lens=tl_ev, vae_frame_rate=VFR)
                tt_ev = torch.tensor(lrc_ev, dtype=torch.int64).unsqueeze(0).to(device)
                gen_ev, _ = raw_model.sample(
                    cond=rl_ev.to(device), text=tt_ev, duration=tl_ev,
                    midi_in=mi_in_ev.to(device), midi_p=midi_p_ev, bound_p=bp_ev,
                    steps=TRAIN_STEPS, cfg_strength=CFG_STRENGTH, guidance_scale=CFG_STRENGTH,
                    t_shift=0.5, seed=42, use_epss=False, enable_melody_control=True)
                rear_ev = int(VFR*1.0)
                b_ev = gen_ev[:, rll_ev:-rear_ev, :]
                ev_wav = vae.decode_audio(b_ev.permute(0,2,1).float()).squeeze(0)[:1].reshape(-1).cpu()
                ev_path = os.path.join(tmp_wav_dir, f"eval_{global_step:06d}.wav")
                torchaudio.save(ev_path, ev_wav.unsqueeze(0), 44100)
                ev_rw = torchaudio.functional.resample(eval_sample["wav"], 44100, 16000).squeeze(0).cpu()
                # Need whisper loaded for eval
                reward.ensure_whisper()
                ev_score = reward.score_all([ev_path], [ev_rw], [eval_sample["kana"]])[0]
                reward.free_whisper()
                print(f"    Eval PER={ev_score[0]:.3f} SIM={ev_score[1]:.3f} F0={ev_score[2]:.3f} DNS={ev_score[3]:.3f}",
                      flush=True)
                if ev_score[0] > best_per:
                    best_per = ev_score[0]
                    torch.save({"model_state_dict": raw_model.state_dict(), "global_step": global_step},
                               os.path.join(args.output_dir, "best_per.pt"))
                    print(f"    New best PER {best_per:.3f} saved!")

        if global_step % 50 == 0:
            torch.cuda.empty_cache()

        # ---- 保存 ----
        if global_step % args.save_every == 0 and is_main():
            save_path = os.path.join(args.output_dir, f"step_{global_step:06d}.pt")
            torch.save({"model_state_dict": raw_model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step}, save_path)
            print(f"  [Save] {save_path}", flush=True)

    if is_main():
        save_path = os.path.join(args.output_dir, f"step_{global_step:06d}_final.pt")
        torch.save({"model_state_dict": raw_model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step}, save_path)
        print(f"[GRPOv3] Done. Final checkpoint: {save_path}")


if __name__ == "__main__":
    main()

