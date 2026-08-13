#!/usr/bin/env python3
"""Fine-tune YingMusic-Singer DiT on Japanese singing data (DDP)."""
import argparse, json, math, os, random, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from singer.model import (
    DEFAULT_VAE_FRAME_HZ, SAMPLE_RATE_16K, SAMPLE_RATE_44K, SAMPLE_RATE_48K,
    YingSinger, _load_audio_sf, seed_everything,
)
from singer.decoder.utils import list_str_to_idx

_DEFAULT_CFG_DROP = 0.15
_DEFAULT_MAX_SEG_S = 12.0
_DEFAULT_GRAD_CLIP = 10.0
_DEFAULT_WD = 0.01


# ── helpers ──────────────────────────────────────────────

def set_seed(seed: int):
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_audio_resample(path: str, target_sr: int) -> torch.Tensor:
    audio, sr = _load_audio_sf(path)
    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if audio.shape[0] == 1 and target_sr == SAMPLE_RATE_48K:
        audio = audio.repeat(2, 1)
    if sr != target_sr:
        from singer.model import _resampler_cache
        import torchaudio
        cache_key = f"{sr}_{target_sr}"
        if cache_key not in _resampler_cache:
            _resampler_cache[cache_key] = torchaudio.transforms.Resample(sr, target_sr)
        audio = _resampler_cache[cache_key](audio)
    return audio


def vae_encode(audio_48k: torch.Tensor, vae, device: torch.device) -> torch.Tensor:
    chunk_size = SAMPLE_RATE_48K * 60
    total_samples = audio_48k.shape[-1]
    features = []
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk = audio_48k[:, start:end].unsqueeze(0).to(device)
        with torch.no_grad():
            feat = vae.encode_audio(chunk)
        features.append(feat.cpu())
    return torch.cat(features, dim=-1)


def extract_melody_tensor(audio_44k, audio_16k, mel_spec, f0_model, melody_model, device):
    audio_mel = mel_spec(audio_44k).permute(0, 2, 1).to(device)
    audio_f0 = f0_model.infer_from_audio(audio_16k)
    if audio_f0.shape[1] != audio_mel.shape[1]:
        audio_f0 = F.interpolate(audio_f0.unsqueeze(1), size=audio_mel.shape[1], mode="nearest").squeeze(1)
    masks = (audio_f0 > 1).bool()
    return melody_model.get_notemidi_seq(audio_mel, masks=masks)


def melody_to_latent_frames(melody: torch.Tensor, target_frames: int) -> torch.Tensor:
    melody = melody.float()
    if melody.ndim == 1:
        melody = melody.unsqueeze(0)
    if melody.shape[1] != target_frames:
        melody = F.interpolate(melody.unsqueeze(1), size=target_frames, mode="nearest").squeeze(1)
    return melody.clamp(0, 127).long()


def tokenize_text(text: str, tokenizer, language: str = "ja"):
    phoneme_str = tokenizer.tokenize(text, "", language=language)[0]
    return phoneme_str, phoneme_str.split("|")


def sample_log_normal(batch_size: int, device: torch.device, loc=0.0, scale=1.0):
    t = torch.randn(batch_size, device=device) * scale + loc
    t = t.exp()
    t = t / (t + 1)
    return t.clamp(1e-5, 1.0 - 1e-5)


# ── precompute one segment ───────────────────────────────

@torch.no_grad()
def precompute_segment(wav_path, singer, device, max_frames):
    a48 = load_audio_resample(wav_path, SAMPLE_RATE_48K)
    a44 = load_audio_resample(wav_path, SAMPLE_RATE_44K)
    a16 = load_audio_resample(wav_path, SAMPLE_RATE_16K)

    full_latent = vae_encode(a48, singer.vae, device)
    full_frames = full_latent.shape[-1]
    if full_frames < 4:
        return None
    if full_frames > max_frames:
        full_frames = max_frames
        full_latent = full_latent[:, :, :full_frames]
        trim_samp = full_frames * 1920
        a48, a44, a16 = a48[:, :trim_samp], a44[:, :trim_samp], a16[:, :trim_samp]

    half_frames = max(1, full_frames // 2)
    cond_latent = vae_encode(a48[:, : half_frames * 1920], singer.vae, device)

    melody_raw = extract_melody_tensor(a44, a16, singer.mel_spectrogram,
                                       singer.f0_extractor, singer.melody_extractor, device)
    melody = melody_to_latent_frames(melody_raw, full_frames)
    return {"x1": full_latent, "cond": cond_latent, "melody": melody, "full_frames": full_frames}


# ── data collation ───────────────────────────────────────

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, entries: list[dict], singer, device, max_frames, cache_path=None):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.cache = []
        # Reuse cache if exists
        if cache_path and os.path.exists(cache_path):
            if rank == 0:
                print(f"Loading cached precompute: {cache_path}")
            self.cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            return
        # Shard entries across GPUs
        my_entries = entries[rank::world_size]
        bar = tqdm(my_entries, desc=f"Precompute GPU{rank}", position=rank)
        for e in bar:
            seg = precompute_segment(e["Path"], singer, device, max_frames)
            if seg is not None:
                seg["lyrics"] = e["Text"]
                seg["path"] = e["Path"]
                self.cache.append(seg)
        dist.barrier()
        # Gather all caches to rank 0, save to disk, then broadcast via file
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(self.cache, gathered if rank == 0 else None, dst=0)
        if rank == 0:
            all_cache = []
            for g in gathered:
                all_cache.extend(g)
            print(f"Precomputed {len(all_cache)} / {len(entries)} segments")
            if cache_path:
                torch.save(all_cache, cache_path)
        dist.barrier()
        if rank != 0:
            self.cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            self.cache = all_cache

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        seg = self.cache[idx]
        return {
            "x1": seg["x1"],
            "cond": seg["cond"],
            "melody": seg["melody"],
            "full_frames": seg["full_frames"],
            "lyrics": seg["lyrics"],
        }


# ── train step helpers ───────────────────────────────────

def compute_loss(decoder, x1, cond_pad, melody, text_tensor, t_continuous,
                 args, device, amp_dtype):
    x0 = torch.randn_like(x1)
    xt = (1 - t_continuous[:, None, None]) * x0 + t_continuous[:, None, None] * x1
    v_target = x1 - x0
    drop_text = random.random() < args.cfg_drop
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.use_amp):
        v_pred = decoder(x=xt, cond=cond_pad, text=text_tensor, time=t_continuous,
                         melody=melody, tags=["singing"], drop_text=drop_text)
        return F.mse_loss(v_pred, v_target)

# no-diffusion eval MSE (fixed t=0.5, no noise)
def compute_eval_loss(decoder, x1, cond_pad, melody, text_tensor, amp_dtype):
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        t = torch.full((1,), 0.5, device=x1.device)
        xt = 0.5 * torch.randn_like(x1) + 0.5 * x1
        v_target = 2.0 * (x1 - xt)  # x_0 = 2*xt - x1, v = x1 - x0 = 2*(x1 - xt)
        v_pred = decoder(x=xt, cond=cond_pad, text=text_tensor, time=t,
                         melody=melody, tags=["singing"], drop_text=False)
        return F.mse_loss(v_pred, v_target)


# ── main ─────────────────────────────────────────────────

def train(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    set_seed(args.seed + rank)

    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        (output_dir / "cache").mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    else:
        output_dir = Path(args.output_dir)
        writer = None

    # ── load data ──
    train_entries = load_json(args.train_json)
    test_entries = load_json(args.test_json)
    if rank == 0:
        print(f"Train: {len(train_entries)} entries, Test: {len(test_entries)} entries")

    singer = YingSinger(singer_path=args.singer_path, device=device)
    singer.eval()
    for m in [singer.vae, singer.f0_extractor, singer.melody_extractor]:
        if hasattr(m, "parameters"):
            for p in m.parameters():
                p.requires_grad = False

    max_frames = int(args.max_seg_seconds * DEFAULT_VAE_FRAME_HZ)

    cache_dir = Path(args.output_dir) / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = str(cache_dir / "train_cache.pt")
    test_cache = str(cache_dir / "test_cache.pt")

    train_ds = SegDataset(train_entries, singer, device, max_frames, cache_path=train_cache)
    dist.barrier()
    if rank == 0:
        test_ds = SegDataset(test_entries, singer, device, max_frames, cache_path=test_cache)
    else:
        test_ds = SegDataset([], singer, device, max_frames, cache_path=test_cache)
    dist.barrier()

    del singer.vae  # free VAE buffer (no longer needed)
    torch.cuda.empty_cache()

    decoder = singer.singer.decoder
    decoder.train()
    decoder = DDP(decoder, device_ids=[local_rank], find_unused_parameters=True)
    trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in decoder.parameters())
    if rank == 0:
        print(f"DiT params: {trainable:,} trainable / {total_params:,} total")

    amp_dtype = torch.bfloat16 if args.use_amp else torch.float32

    optimizer = AdamW(decoder.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.wd)
    warmup_s = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_steps)
    cosine_s = CosineAnnealingLR(optimizer, T_max=args.max_steps - args.warmup_steps, eta_min=args.lr_min)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_s, cosine_s], milestones=[args.warmup_steps])

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler, num_workers=2, pin_memory=True, prefetch_factor=2)

    if rank == 0 and test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
        eval_idxs = list(range(min(32, len(test_ds))))

    global_step = 0
    total_loss = 0.0
    start_time = time.time()
    tokenizer = singer.tokenizer

    if rank == 0:
        pbar = tqdm(total=args.max_steps, desc="Training", unit="step", dynamic_ncols=True)

    optimizer.zero_grad()
    train_iter = iter(train_loader)

    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_sampler.set_epoch(global_step)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x1 = batch["x1"].float().to(device)
        if x1.dim() == 4:
            x1 = x1.squeeze(1)  # remove DataLoader batch dim
        cond = batch["cond"].float().to(device)
        if cond.dim() == 4:
            cond = cond.squeeze(1)
        melody = batch["melody"].to(device)
        if melody.dim() == 3:
            melody = melody.squeeze(1)  # [1,1,T] -> [1,T]
        full_frames = batch["full_frames"][0].item()
        lyrics = batch["lyrics"][0] if batch["lyrics"][0] else "a"

        x1 = x1.permute(0, 2, 1)
        cond_pad = F.pad(cond.permute(0, 2, 1), (0, 0, 0, full_frames - cond.shape[-1]))

        _, tokens = tokenize_text(lyrics, tokenizer, language=args.language)
        text_tensor = list_str_to_idx([tokens], tokenizer.vocab).to(device)

        t_continuous = sample_log_normal(1, device)[0:1]

        loss = compute_loss(decoder, x1, cond_pad, melody, text_tensor, t_continuous, args, device, amp_dtype)
        loss.backward()

        step_loss = loss.item()
        total_loss += step_loss
        global_step += world_size

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=_DEFAULT_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if rank == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / global_step) * (args.max_steps - global_step) if global_step > 0 else 0
            lr_now = scheduler.get_last_lr()[0]
            pbar.update(world_size)
            pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{lr_now:.2e}",
                             eta=f"{eta/60:.0f}m{eta%60:.0f}s", step=global_step)
            if global_step % args.log_every < world_size:
                avg = total_loss / max(1, args.log_every)
                writer.add_scalar("train/loss", avg, global_step)
                writer.add_scalar("train/lr", lr_now, global_step)
                total_loss = 0.0

            if global_step % args.eval_every < world_size and global_step > 0 and test_ds is not None:
                decoder.eval()
                eval_losses = []
                with torch.no_grad():
                    for i, tb in enumerate(test_loader):
                        if i >= 32:
                            break
                        ex1 = tb["x1"].float().to(device)
                        if ex1.dim() == 4:
                            ex1 = ex1.squeeze(1)
                        ex1 = ex1.permute(0, 2, 1)
                        econ = tb["cond"].float().to(device)
                        if econ.dim() == 4:
                            econ = econ.squeeze(1)
                        emel = tb["melody"].to(device)
                        if emel.dim() == 3:
                            emel = emel.squeeze(1)
                        ef = tb["full_frames"][0].item()
                        ely = tb["lyrics"][0] if tb["lyrics"][0] else "a"
                        econd = F.pad(econ.permute(0, 2, 1), (0, 0, 0, ef - econ.shape[-1]))
                        _, etok = tokenize_text(ely, tokenizer, language=args.language)
                        etex = list_str_to_idx([etok], tokenizer.vocab).to(device)
                        eloss = compute_eval_loss(decoder.module, ex1, econd, emel, etex, amp_dtype)
                        eval_losses.append(eloss.item())
                eval_avg = np.mean(eval_losses)
                writer.add_scalar("eval/loss", eval_avg, global_step)
                pbar.set_postfix(loss=f"{step_loss:.4f}", eval=f"{eval_avg:.4f}",
                                 lr=f"{lr_now:.2e}", eta=f"{eta/60:.0f}m{eta%60:.0f}s")
                decoder.train()

            if global_step % args.save_every < world_size and global_step > 0:
                ckpt_path = output_dir / "checkpoints" / f"step_{global_step:06d}.pt"
                torch.save({"step": global_step, "model_state_dict": decoder.module.state_dict(),
                            "optimizer": optimizer.state_dict()}, ckpt_path)
                print(f"\nSaved: {ckpt_path}")

    if rank == 0:
        final_path = output_dir / "checkpoints" / f"step_{global_step:06d}_final.pt"
        torch.save({"step": global_step, "model_state_dict": decoder.module.state_dict(),
                    "optimizer": optimizer.state_dict()}, final_path)
        print(f"\nFinal: {final_path}")
        writer.close()
        pbar.close()
        elapsed = (time.time() - start_time) / 60
        print(f"Total time: {elapsed:.1f} min | avg {global_step/elapsed:.0f} step/min")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YingMusic-Singer DiT (DDP)")
    parser.add_argument("--train_json", type=str, default="${REMOTE_ROOT}/final_sum_large/train_singnet.json")
    parser.add_argument("--test_json", type=str, default="${REMOTE_ROOT}/final_sum_large/test_singnet.json")
    parser.add_argument("--output_dir", type=str, default="${REMOTE_ROOT}/YingMusic-Singer/finetune_v2")
    parser.add_argument("--singer_path", type=str, default="${REMOTE_ROOT}/YingMusic-Singer")
    parser.add_argument("--language", type=str, default="ja")
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--clip_grad", type=float, default=_DEFAULT_GRAD_CLIP)
    parser.add_argument("--wd", type=float, default=_DEFAULT_WD)
    parser.add_argument("--cfg_drop", type=float, default=_DEFAULT_CFG_DROP)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--max_seg_seconds", type=float, default=_DEFAULT_MAX_SEG_S)
    args = parser.parse_args()
    train(args)

