import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from ema_pytorch import EMA
from omegaconf import OmegaConf

FRAME_RATE = 44100 / 2048
SEP_TOKEN = 365


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


def sampled_model_fingerprint(model):
    """Build a cheap reproducibility fingerprint without copying full checkpoints."""
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            flat = tensor.detach().cpu().reshape(-1)
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            if flat.numel():
                indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
                sample = flat[indices].to(torch.float64).contiguous().numpy()
                digest.update(sample.tobytes())
    return digest.hexdigest()


def sync_gradients(model, world_size, bucket_bytes=25 * 1024 * 1024):
    """Synchronize gradients in fixed buckets because Singer.forward is bypassed."""
    if world_size <= 1:
        return

    def flush(bucket):
        if not bucket:
            return
        flat = torch._utils._flatten_dense_tensors([p.grad for p in bucket])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat.div_(world_size)
        synced = torch._utils._unflatten_dense_tensors(flat, [p.grad for p in bucket])
        for parameter, gradient in zip(bucket, synced):
            parameter.grad.copy_(gradient)

    bucket = []
    bucket_size = 0
    bucket_dtype = None
    for parameter in (p for p in model.parameters() if p.requires_grad):
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradient_bytes = parameter.grad.numel() * parameter.grad.element_size()
        if bucket and (
            parameter.grad.dtype != bucket_dtype
            or bucket_size + gradient_bytes > bucket_bytes
        ):
            flush(bucket)
            bucket = []
            bucket_size = 0
        bucket.append(parameter)
        bucket_size += gradient_bytes
        bucket_dtype = parameter.grad.dtype
    flush(bucket)


def collate_svs(batch):
    return {
        "wav": [item["wav"] for item in batch],
        "sr": [item["sr"] for item in batch],
        "phrases": [item["phrases"] for item in batch],
        "full_tokens": [item["full_tokens"] for item in batch],
        "tier": [item["tier"] for item in batch],
        "duration": [item["duration"] for item in batch],
    }


class V4Dataset(Dataset):
    def __init__(self, token_json_paths, max_duration_sec=30.0):
        self.records = []
        for path, tier in token_json_paths:
            if not os.path.exists(path):
                print(f"[V4cDataset] WARNING: {path} not found, skipping")
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                item["_tier"] = tier
                self.records.append(item)
        self.max_duration_sec = max_duration_sec

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
        return {
            "wav": wav,
            "sr": sr,
            "phrases": rec["Phrases"],
            "full_tokens": rec.get("full_tokens", []),
            "tier": rec["_tier"],
            "duration": rec["Duration"],
        }


def compute_cka_loss_from_hidden(hidden_states, midi_latent, cka_layers=None):
    from src.YingMusicSinger.utils.common import cka_loss, calculate_similarity_matrix_with_mask
    if cka_layers is None:
        cka_layers = [-1, -2, -3]
    total_cka = 0.0
    count = 0
    for layer_idx in cka_layers:
        h = hidden_states[layer_idx]
        B, T_, D_ = h.shape
        b1, t1, d1 = midi_latent.shape
        common_t = min(T_, t1)
        h = h[:, :common_t, :]
        m = midi_latent[:, :common_t, :]
        sim_h = calculate_similarity_matrix_with_mask(h)
        sim_m = calculate_similarity_matrix_with_mask(m)
        total_cka += cka_loss(sim_h, sim_m)
        count += 1
    return total_cka / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml")
    parser.add_argument("--init_mode", choices=("random",), default="random")
    parser.add_argument("--vae_config", default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json")
    parser.add_argument("--vae_ckpt", default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt")
    parser.add_argument("--midi_ckpt", default="ckpts/model_ckpt_steps_100000_simplified.ckpt")
    parser.add_argument("--token_dir", required=True)
    parser.add_argument("--eval_token_dir", default=None)
    parser.add_argument("--output_dir", default="ckpts/plus_ja_sft_v4vf")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--hold_steps", type=int, default=12000,
                        help="Steps to hold peak LR before cosine decay")
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit_n", type=int, default=20)
    parser.add_argument("--find_unused_parameters", action="store_true", default=False)
    parser.add_argument("--cka_weight", type=float, default=0.7)
    parser.add_argument("--t_shift", type=float, default=0.5)
    parser.add_argument("--drop_text", type=float, default=0.15,
                        help="Text CFG dropout probability")
    parser.add_argument("--flow_b_weight", type=float, default=2.0,
                        help="Weight multiplier for B區 flow loss")
    args = parser.parse_args()

    if args.resume and not os.path.isfile(args.resume):
        parser.error(f"--resume checkpoint does not exist: {args.resume}")

    if "LOCAL_RANK" in os.environ:
        local_rank, world_size = setup_ddp()
    else:
        local_rank, world_size = 0, 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # Every rank must construct identical random model and EMA weights.
    seed_everything(args.seed)
    device = torch.device(f"cuda:{local_rank}")

    if is_main():
        print(f"[Train V4vf] World={world_size} Device={device}")
        print(f"[Train V4vf] Args: {args}")

    cfg = OmegaConf.load(args.config)

    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer

    if is_main():
        print("[Train V4vf] Building randomly initialized DiT...")
    dit = DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
              mel_dim=cfg.model.mel_spec.n_mel_channels,
              long_skip_connection=True)

    singer_model = Singer(
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

    for parameter in singer_model.parameters():
        parameter.requires_grad = False
    for parameter in singer_model.transformer.parameters():
        parameter.requires_grad = True

    trainable_parameters = sum(p.numel() for p in singer_model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in singer_model.parameters())
    initial_fingerprint = sampled_model_fingerprint(singer_model)
    if world_size > 1:
        rank_fingerprints = [None] * world_size
        torch.distributed.all_gather_object(rank_fingerprints, initial_fingerprint)
        if len(set(rank_fingerprints)) != 1:
            raise RuntimeError(f"Random initialization differs across ranks: {rank_fingerprints}")

    experiment_init_fingerprint = initial_fingerprint
    resume_global_step = 0
    if args.resume:
        if is_main():
            print(f"[Train V4vf] Resuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model_state_dict" in resume_ckpt:
            sd = resume_ckpt["model_state_dict"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            singer_model.load_state_dict(sd, strict=False)
            if is_main():
                print("[Train V4vf] Loaded model weights from V4vf checkpoint")
        resume_global_step = resume_ckpt.get("global_step", 0)
        experiment_init_fingerprint = resume_ckpt.get(
            "initial_fingerprint", "missing-from-resume-checkpoint"
        )
    elif is_main():
        print("[Train V4vf] Init mode=random; no DiT/Singer checkpoint will be loaded")
        print(f"[Train V4vf] Init seed={args.seed} sampled_sha256={initial_fingerprint}")
        print(
            f"[Train V4vf] DiT trainable parameters={trainable_parameters:,} "
            f"Singer total parameters={total_parameters:,}"
        )

    singer_model = singer_model.to(device)
    singer_model.train()
    raw_model = singer_model

    if is_main():
        print("[Train V4vf] Loading frozen official VAE...")
    vae = StableAudioInfer(model_config_path=args.vae_config, model_ckpt_path=args.vae_ckpt)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    if is_main():
        print("[Train V4vf] Loading frozen SOME MIDI teacher...")
    midi_teacher = MIDIExtractor(in_dim=80)
    midi_teacher._load_form_ckpt(args.midi_ckpt)
    midi_teacher = midi_teacher.to(device).eval()
    for p in midi_teacher.parameters():
        p.requires_grad = False

    mel_spec_extract = MelodySpectrogram()

    ema = EMA(singer_model, beta=cfg.ema_kwargs.beta,
              update_after_step=cfg.ema_kwargs.update_after_step,
              update_every=cfg.ema_kwargs.update_every)
    ema.to(device)
    ema_checkpoint = args.resume
    if ema_checkpoint and os.path.exists(ema_checkpoint):
        ema_ckpt = torch.load(ema_checkpoint, map_location="cpu", weights_only=False)
        if "ema_model_state_dict" in ema_ckpt:
            ema_sd = ema_ckpt["ema_model_state_dict"]
            ema_sd = {k.replace("ema_model.", ""): v for k, v in ema_sd.items()}
            ema.ema_model.load_state_dict(ema_sd, strict=False)
            if is_main():
                print(f"[Train V4vf] Loaded EMA weights from: {ema_checkpoint}")

    # The custom loss path calls transformer internals directly, so wrapping Singer in
    # DDP would bypass DDP.forward. Gradients are synchronized explicitly below.
    raw_model = singer_model

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=1e-2)
    if args.resume and os.path.exists(args.resume):
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            if is_main():
                print("[Train V4vf] Restored optimizer state for interrupted-run resume")

    # === V4c: warmup-hold-decay LR schedule ===
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        elif step < args.warmup_steps + args.hold_steps:
            return 1.0
        else:
            decay_steps = max(1, args.max_steps - args.warmup_steps - args.hold_steps)
            progress = (step - args.warmup_steps - args.hold_steps) / decay_steps
            return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # V4d strict-control data already contains only the retained V4c L1+L2 samples.
    train_token_paths = [
        (os.path.join(args.token_dir, "train_tokens.json"), "L1"),
    ]
    train_dataset = V4Dataset(train_token_paths, args.max_duration)
    if args.overfit:
        train_dataset.records = train_dataset.records[:args.overfit_n]
        if is_main():
            print(f"[Train V4vf] OVERFIT MODE: {len(train_dataset)} samples")
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, collate_fn=collate_svs,
                              pin_memory=True, drop_last=True)

    eval_loader = None
    eval_dataset = None
    eval_token_dir = args.eval_token_dir or args.token_dir
    eval_token_paths = [
        (os.path.join(eval_token_dir, "test_tokens.json"), "L1"),
    ]
    if os.path.exists(eval_token_paths[0][0]):
        eval_dataset = V4Dataset(eval_token_paths, args.max_duration)
        if args.overfit:
            eval_dataset.records = eval_dataset.records[:max(4, args.overfit_n // 5)]
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, collate_fn=collate_svs,
                                 pin_memory=True, drop_last=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main():
        manifest = {
            "branch": "V4vf",
            "init_mode": args.init_mode,
            "model_init_seed": args.seed,
            "initial_fingerprint": experiment_init_fingerprint,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "world_size": world_size,
            "effective_batch": args.batch_size * world_size * args.grad_accum,
            "config": args.config,
            "vae_config": args.vae_config,
            "vae_ckpt": args.vae_ckpt,
            "midi_ckpt": args.midi_ckpt,
            "token_dir": args.token_dir,
            "resume": args.resume,
        }
        manifest_path = os.path.join(args.output_dir, "init_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)
        print(f"[Train V4vf] Init manifest: {manifest_path}")
        print(f"[Train V4vf] Train: {len(train_dataset)} | Eval: {len(eval_dataset) if eval_dataset else 0}")
        print(f"[Train V4vf] Steps/epoch: {len(train_loader)} | Max steps: {args.max_steps}")
        print(f"[Train V4vf] Effective batch: {args.batch_size * world_size * args.grad_accum}")
        print(f"[Train V4vf] lr={args.lr} hold={args.hold_steps} drop_text={args.drop_text} "
              f"flow_b_w={args.flow_b_weight} cka={args.cka_weight}")
        if args.resume:
            print(f"[Train V4vf] Resumed from step {resume_global_step}")
        print("=" * 60)

    def compute_ref_len(T, phrase_boundaries=None):
        five_sec_frames = int(5.0 * FRAME_RATE)
        ref_len = T * random.uniform(0.125, 0.33)
        ref_len = max(ref_len, five_sec_frames)
        ref_len = min(ref_len, int(T * 0.65))
        if phrase_boundaries is not None and len(phrase_boundaries) > 0:
            margin = int(2.5 * FRAME_RATE)
            candidates = [b for b in phrase_boundaries if b > 0 and abs(b - ref_len) <= margin]
            if candidates:
                ref_len = min(candidates, key=lambda b: abs(b - ref_len))
        return int(ref_len)

    def align_text_with_timestamps(phrases, timestamps, ref_len, T):
        aligned_text = torch.zeros(1, T, dtype=torch.long)
        for i, ts in enumerate(timestamps):
            phrase_tokens = phrases[i]["tokens"] + [SEP_TOKEN]
            center_frame = ts["center"] * FRAME_RATE
            start_frame = int(ts["start"] * FRAME_RATE)
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

    def uniform_align_text(phrases, ref_len, T):
        flat = []
        for i, p in enumerate(phrases):
            flat.extend(p["tokens"])
            flat.append(SEP_TOKEN)
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

    def process_batch(wavs, srs, phrases_list, tiers):
        with torch.no_grad():
            w = wavs[0]
            sr = srs[0]
            phrases = phrases_list[0]
            tier = tiers[0]

            w_2d = w.unsqueeze(0) if w.dim() == 1 else w
            latent = vae.encode_audio(w_2d, in_sr=sr)
            full_latent = latent.squeeze(0).transpose(0, 1).unsqueeze(0)
            B, T, D = full_latent.shape

            if tier in ("L1", "L2"):
                phrase_boundaries = [int(p["start"] * FRAME_RATE) for p in phrases]
            else:
                phrase_boundaries = None
            ref_len = compute_ref_len(T, phrase_boundaries)

            cond = torch.zeros_like(full_latent)
            cond[:, :ref_len, :] = full_latent[:, :ref_len, :]

            mel = mel_spec_extract(audio=w_2d, sr=44100).to(device)
            midi_p, _ = midi_teacher(mel.transpose(1, 2))
            if midi_p.shape[1] != T:
                midi_p = F.interpolate(midi_p.transpose(1, 2), size=T,
                                       mode="linear", align_corners=False).transpose(1, 2)
            midi = raw_model.smoothMelody_MIDIFuzzDisturb(midi_p)
            midi[:, :ref_len, :] = 0

            if tier in ("L1", "L2"):
                timestamps = [{"start": p["start"], "end": p["end"],
                               "center": (p["start"] + p["end"]) / 2} for p in phrases]
                aligned_text = align_text_with_timestamps(phrases, timestamps, ref_len, T)
            else:
                aligned_text = uniform_align_text(phrases, ref_len, T)
            aligned_text = aligned_text.to(device)

        return full_latent, cond, midi, midi_p, aligned_text, ref_len, T

    def run_dit(x_t, cond, text_tokens, t, midi, drop_audio, drop_text, drop_midi):
        dit = raw_model.transformer
        B, seq_len = x_t.shape[0], x_t.shape[1]
        if t.ndim == 0:
            t = t.repeat(B)
        time_emb = dit.time_embed(t)

        x, _ = dit.get_input_embed(x_t, cond, text_tokens, midi,
                                    drop_audio_cond=drop_audio, drop_text=drop_text,
                                    drop_midi=drop_midi, cache=False)
        rope = dit.rotary_embed.forward_from_seq_len(seq_len)

        hidden_states = []
        if dit.long_skip_connection is not None:
            residual = x
        for i, block in enumerate(dit.transformer_blocks):
            x = block(x, time_emb, mask=None, rope=rope)
            if i >= len(dit.transformer_blocks) - 3:
                hidden_states.append(x)
        if dit.long_skip_connection is not None:
            x = dit.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = dit.norm_out(x, time_emb)
        output = dit.proj_out(x)
        return output, hidden_states

    def compute_loss(wavs, srs, phrases_list, tiers):
        full_latent, cond, midi, midi_p, aligned_text, ref_len, T = \
            process_batch(wavs, srs, phrases_list, tiers)

        u = torch.rand(1, device=device)
        t = args.t_shift * u / (1 - args.t_shift * u)

        noise = torch.randn_like(full_latent)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * full_latent
        v_target = full_latent - noise

        drop_audio = random.random() < 0.3
        drop_text = random.random() < args.drop_text
        drop_midi = random.random() < 0.3

        v_pred, hidden_states = run_dit(x_t, cond, aligned_text, t, midi,
                                         drop_audio, drop_text, drop_midi)

        L_flow_A = F.mse_loss(v_pred[:, :ref_len, :], v_target[:, :ref_len, :])
        L_flow_B = F.mse_loss(v_pred[:, ref_len:, :], v_target[:, ref_len:, :])
        L_flow = L_flow_A + args.flow_b_weight * L_flow_B

        L_cka = torch.tensor(0.0, device=device)
        if args.cka_weight > 0:
            h_B = [h[:, ref_len:, :] for h in hidden_states]
            m_B = midi_p[:, ref_len:, :]
            L_cka = compute_cka_loss_from_hidden(h_B, m_B)

        loss = L_flow + args.cka_weight * L_cka
        return loss, {"flow_A": L_flow_A.item(), "flow_B": L_flow_B.item(),
                       "flow": L_flow.item(), "cka": L_cka.item()}

    @torch.no_grad()
    def run_eval():
        raw_model.eval()
        total_loss = total_flow_a = total_flow_b = total_cka = 0.0
        count = 0
        for batch in eval_loader:
            wavs = [b.to(device) for b in batch["wav"]]
            srs = batch["sr"]
            phrases_list = batch["phrases"]
            tiers = batch["tier"]
            loss, d = compute_loss(wavs, srs, phrases_list, tiers)
            total_loss += loss.item()
            total_flow_a += d["flow_A"]
            total_flow_b += d["flow_B"]
            total_cka += d["cka"]
            count += 1
        raw_model.train()
        return (total_loss / max(count, 1), total_flow_a / max(count, 1),
                total_flow_b / max(count, 1), total_cka / max(count, 1))

    # Construction used the same seed on all ranks. Training randomness differs by
    # rank so CFG dropout and flow noise are not duplicated.
    seed_everything(args.seed + local_rank)

    global_step = 0
    accum_flow_a = accum_flow_b = accum_cka = 0.0
    accum_grad_norm = 0.0
    clipped_steps = 0
    t_start = time.time()
    data_iter = iter(train_loader)

    # === V4c: proper resume — continue LR curve from checkpoint step ===
    if args.resume:
        global_step = resume_global_step
        for _ in range(global_step):
            scheduler.step()
        if is_main():
            print(f"[Train V4vf] Resumed at step {global_step}, LR={optimizer.param_groups[0]['lr']:.2e}")

    # === Track steps actually run since (re)start for accurate ETA ===
    step_offset = global_step

    while global_step < args.max_steps:
        optimizer.zero_grad()

        for ga_step in range(args.grad_accum):
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
            loss, loss_dict = compute_loss(wavs, srs, phrases_list, tiers)
            loss = loss / args.grad_accum
            loss.backward()
            accum_flow_a += loss_dict["flow_A"]
            accum_flow_b += loss_dict["flow_B"]
            accum_cka += loss_dict["cka"]

        sync_gradients(raw_model, world_size)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        grad_norm_value = float(grad_norm)
        accum_grad_norm += grad_norm_value
        clipped_steps += int(grad_norm_value > 1.0)
        optimizer.step()
        scheduler.step()
        ema.update()
        global_step += 1

        if global_step % 100 == 0:
            torch.cuda.empty_cache()

        if global_step % args.log_every == 0 and is_main():
            n = args.log_every * args.grad_accum
            avg_a = accum_flow_a / n
            avg_b = accum_flow_b / n
            avg_cka = accum_cka / n
            avg_grad_norm = accum_grad_norm / args.log_every
            clip_rate = clipped_steps / args.log_every
            elapsed = time.time() - t_start
            steps_since_start = global_step - step_offset
            sec_per_step = elapsed / max(steps_since_start, 1)
            remaining = args.max_steps - global_step
            eta = remaining * sec_per_step
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m"
            lr_now = optimizer.param_groups[0]["lr"]
            eff_step = global_step if args.resume else global_step
            print(f"  [{global_step:6d}/{args.max_steps} | "
                  f"{100*global_step/args.max_steps:.1f}% | base_eff={eff_step} | "
                  f"ETA {eta_str} | {sec_per_step:.2f}s/step] "
                  f"FlowA={avg_a:.4f} FlowB={avg_b:.4f} CKA={avg_cka:.4f} | "
                  f"GradNorm={avg_grad_norm:.3f} Clip={clip_rate:.0%} | LR={lr_now:.2e}")
            accum_flow_a = accum_flow_b = accum_cka = 0.0
            accum_grad_norm = 0.0
            clipped_steps = 0

        if global_step % args.eval_every == 0 and eval_loader is not None and is_main():
            ev_loss, ev_flow_a, ev_flow_b, ev_cka = run_eval()
            print(f"  >>> EVAL step={global_step} (eff={resume_global_step+global_step if args.resume else global_step}) | "
                  f"Loss={ev_loss:.4f} (FlowA={ev_flow_a:.4f} FlowB={ev_flow_b:.4f} "
                  f"CKA={ev_cka:.4f}) <<<")

        if global_step % args.save_every == 0 and is_main():
            save_path = os.path.join(args.output_dir, f"step_{global_step:06d}.pt")
            torch.save({
                "model_state_dict": raw_model.state_dict(),
                "ema_model_state_dict": ema.ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "resume_global_step": resume_global_step,
                "init_mode": args.init_mode,
                "initial_fingerprint": experiment_init_fingerprint,
                "args": vars(args),
            }, save_path)
            print(f"  [Save  {global_step:6d}] {save_path}")

    if is_main():
        save_path = os.path.join(args.output_dir, f"step_{global_step:06d}_final.pt")
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "ema_model_state_dict": ema.ema_model.state_dict(),
            "global_step": global_step,
            "resume_global_step": resume_global_step,
            "init_mode": args.init_mode,
            "initial_fingerprint": experiment_init_fingerprint,
            "args": vars(args),
        }, save_path)
        print(f"[Train V4vf] Final: {save_path}")
        print("[Train V4vf] Complete!")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
