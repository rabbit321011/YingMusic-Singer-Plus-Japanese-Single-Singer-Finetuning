import argparse, json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from ema_pytorch import EMA
from omegaconf import OmegaConf

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

def collate_svs(batch):
    return {
        "wav": [item["wav"] for item in batch],
        "sr": [item["sr"] for item in batch],
        "text": [item["text"] for item in batch],
        "language": [item["language"] for item in batch],
    }

class SVSDataset(Dataset):
    def __init__(self, records, max_duration_sec=30.0):
        self.records = records
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
            "wav": wav, "sr": sr,
            "text": rec["Text"],
            "language": rec.get("Language", "ja"),
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
    parser.add_argument("--ckpt_path", default="ckpts/YingMusicSinger_model.pt")
    parser.add_argument("--vae_config", default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json")
    parser.add_argument("--vae_ckpt", default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt")
    parser.add_argument("--midi_ckpt", default="ckpts/model_ckpt_steps_100000_simplified.ckpt")
    parser.add_argument("--train_json", default="${SERVER_ROOT}/final_sum_large/train_singnet.json")
    parser.add_argument("--eval_json", default="${SERVER_ROOT}/final_sum_large/test_singnet.json")
    parser.add_argument("--output_dir", default="ckpts/plus_ja_sft_v3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=7e-6)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit_n", type=int, default=20)
    parser.add_argument("--find_unused_parameters", action="store_true", default=False,
                        help="DDP find_unused_parameters flag")
    parser.add_argument("--autocast", action="store_true", default=False,
                        help="Enable bf16 mixed precision")
    parser.add_argument("--cka_weight", type=float, default=1.0)
    parser.add_argument("--reuse_buffers", action="store_true", default=False,
                        help="Pre-allocate cond/midi buffers to match V1 pattern")
    parser.add_argument("--cond_gt", action="store_true", default=False,
                        help="[Speed diag only] Use full latent as cond (V1-style, cond=GT)")
    parser.add_argument("--profile_timing", action="store_true", default=False,
                        help="Log per-phase timing (data/vae/some/token/forward/backward/opt)")
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        local_rank, world_size = setup_ddp()
    else:
        local_rank, world_size = 0, 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    seed_everything(args.seed + local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if is_main():
        print(f"[Train V3] World={world_size} Device={device}")
        print(f"[Train V3] Args: {args}")

    cfg = OmegaConf.load(args.config)

    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
    from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer

    if is_main():
        print("[Train V3] Building DiT...")
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

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        if is_main():
            print(f"[Train V3] Loading checkpoint: {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        if "ema_model_state_dict" in ckpt:
            sd = ckpt["ema_model_state_dict"]
            sd = {k.replace("ema_model.", ""): v for k, v in sd.items()}
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        singer_model.load_state_dict(sd, strict=False)

    singer_model = singer_model.to(device)
    singer_model.train()
    raw_model = singer_model

    if is_main():
        print("[Train V3] Loading VAE...")
    vae = StableAudioInfer(model_config_path=args.vae_config, model_ckpt_path=args.vae_ckpt)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    if is_main():
        print("[Train V3] Loading SOME MIDI teacher...")
    midi_teacher = MIDIExtractor(in_dim=80)
    midi_teacher._load_form_ckpt(args.midi_ckpt)
    midi_teacher = midi_teacher.to(device).eval()
    for p in midi_teacher.parameters():
        p.requires_grad = False

    mel_spec_extract = MelodySpectrogram()
    vae_frame_rate = 44100 / 2048
    tokenizer = CNENTokenizer()

    ema = EMA(singer_model, beta=cfg.ema_kwargs.beta,
              update_after_step=cfg.ema_kwargs.update_after_step,
              update_every=cfg.ema_kwargs.update_every)
    ema.to(device)

    if world_size > 1:
        singer_model = DDP(singer_model, device_ids=[local_rank], find_unused_parameters=args.find_unused_parameters)
    raw_model = singer_model.module if world_size > 1 else singer_model

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    with open(args.train_json) as f:
        train_data = json.load(f)
    if args.overfit:
        train_data = train_data[:args.overfit_n]
        if is_main():
            print(f"[Train V3] OVERFIT MODE: {len(train_data)} samples")

    train_dataset = SVSDataset(train_data, args.max_duration)
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, collate_fn=collate_svs,
                              pin_memory=True, drop_last=True)

    eval_loader = None
    eval_dataset = None
    if args.eval_json and os.path.exists(args.eval_json):
        with open(args.eval_json) as f:
            eval_data = json.load(f)
        if args.overfit:
            eval_data = eval_data[:max(4, args.overfit_n // 5)]
        eval_dataset = SVSDataset(eval_data, args.max_duration)
        eval_loader = DataLoader(eval_dataset, batch_size=min(args.batch_size, 2),
                                 shuffle=False, num_workers=0, collate_fn=collate_svs,
                                 pin_memory=True, drop_last=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main():
        print(f"[Train V3] Train: {len(train_dataset)} | Eval: {len(eval_dataset) if eval_dataset else 0}")
        print(f"[Train V3] Steps/epoch: {len(train_loader)} | Max steps: {args.max_steps}")
        print(f"[Train V3] Effective batch: {args.batch_size * world_size * args.grad_accum}")
        print(f"[Train V3] WER baseline: 1.000 (run eval_wer.py on saved ckpts)")
        if args.reuse_buffers:
            max_frames = int(args.max_duration * 44100 / 2048) + 2
            print(f"[Train V3] Pre-allocating buffers for max {max_frames} frames")
        print("=" * 60)

    cond_buffer = None
    midi_buffer = None
    if args.reuse_buffers:
        max_frames = int(args.max_duration * 44100 / 2048) + 2
        cond_buffer = torch.zeros(args.batch_size, max_frames, 64, device=device)
        midi_buffer = torch.zeros(args.batch_size, max_frames, 128, device=device)

    def process_batch(wavs, srs, texts, langs):
        nonlocal cond_buffer, midi_buffer
        with torch.no_grad():
            latents = []
            for w in wavs:
                lat = vae.encode_audio(w.unsqueeze(0) if w.dim() == 1 else w, in_sr=srs[0])
                latents.append(lat)
            max_lat_len = max(l.shape[-1] for l in latents)
            full_latent = torch.zeros(len(latents), latents[0].shape[1], max_lat_len, device=device)
            for i, l in enumerate(latents):
                full_latent[i, :, :l.shape[-1]] = l.squeeze(0)
            full_latent = full_latent.transpose(1, 2)
            B, T, D = full_latent.shape

            if args.cond_gt:
                cond = full_latent
                ref_len = 0
            else:
                ref_len = T // 2
                if cond_buffer is not None:
                    cond = cond_buffer[:B, :T, :].zero_()
                else:
                    cond = torch.zeros_like(full_latent)
                cond[:, :ref_len, :] = full_latent[:, :ref_len, :]

            mels = []
            for w in wavs:
                m = mel_spec_extract(audio=w, sr=srs[0])
                mels.append(m.squeeze(0) if m.dim() == 3 else m)
            max_mel_len = max(m.shape[-1] for m in mels)
            mel_tensor = torch.zeros(len(mels), mels[0].shape[0], max_mel_len, device=device)
            for i, m in enumerate(mels):
                mel_tensor[i, :, :m.shape[-1]] = m
            midi_p, bound_p = midi_teacher(mel_tensor.transpose(1, 2))
            if midi_p.shape[1] != T:
                midi_p = F.interpolate(midi_p.transpose(1, 2), size=T, mode="linear",
                                       align_corners=False).transpose(1, 2)
            midi = raw_model.smoothMelody_MIDIFuzzDisturb(midi_p)
            midi[:, :ref_len, :] = 0

            all_token_ids = []
            for i, text in enumerate(texts):
                tokens = tokenizer.encode(text)
                lrc_token = tokens[:T]
                all_token_ids.append(lrc_token)
            aligned_text = torch.zeros(B, T, dtype=torch.long, device=device)
            for i, tids in enumerate(all_token_ids):
                n = min(len(tids), T)
                aligned_text[i, :n] = torch.tensor(tids[:n], device=device)

        return full_latent, cond, midi, midi_p, aligned_text, B, T, D

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

    def compute_loss(wavs, srs, texts, langs):
        full_latent, cond, midi, midi_p, aligned_text, B, T, D = process_batch(wavs, srs, texts, langs)

        t = torch.rand(B, device=device)
        noise = torch.randn_like(full_latent)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * full_latent
        v_target = full_latent - noise

        drop_audio = random.random() < 0.3
        drop_text = random.random() < 0.3
        drop_midi = random.random() < 0.3

        if args.autocast:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                v_pred, hidden_states = run_dit(x_t, cond, aligned_text, t, midi,
                                                 drop_audio, drop_text, drop_midi)
                L_flow = F.mse_loss(v_pred, v_target)
        else:
            v_pred, hidden_states = run_dit(x_t, cond, aligned_text, t, midi,
                                             drop_audio, drop_text, drop_midi)
            L_flow = F.mse_loss(v_pred, v_target)

        L_cka = torch.tensor(0.0, device=device)
        if args.cka_weight > 0:
            midi_for_cka = midi_p[:, :T, :]
            L_cka = compute_cka_loss_from_hidden(hidden_states, midi_for_cka)

        loss = L_flow + args.cka_weight * L_cka
        return loss, {"flow": L_flow.item(), "cka": L_cka.item()}

    @torch.no_grad()
    def run_eval():
        raw_model.eval()
        total_loss = total_flow = total_cka = 0.0
        count = 0
        for batch in eval_loader:
            wavs = [b.to(device) for b in batch["wav"]]
            srs = batch["sr"]
            texts = batch["text"]
            langs = batch["language"]
            loss, d = compute_loss(wavs, srs, texts, langs)
            total_loss += loss.item()
            total_flow += d["flow"]
            total_cka  += d["cka"]
            count += 1
        raw_model.train()
        return total_loss / max(count, 1), total_flow / max(count, 1), total_cka / max(count, 1)

    global_step = 0
    accum_flow = accum_cka = 0.0
    t_start = time.time()
    data_iter = iter(train_loader)

    if args.profile_timing:
        timer_data = timer_vae = timer_some = timer_token = timer_fwd = timer_bwd = timer_opt = 0.0
        timer_steps = 0

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

            if args.profile_timing:
                t_d0 = time.time()
                wavs = [b.to(device) for b in batch["wav"]]
                t_d1 = time.time()

                with torch.no_grad():
                    latents = []
                    for w in wavs:
                        lat = vae.encode_audio(w.unsqueeze(0) if w.dim() == 1 else w, in_sr=batch["sr"][0])
                        latents.append(lat)
                    max_lat_len = max(l.shape[-1] for l in latents)
                    full_latent = torch.zeros(len(latents), latents[0].shape[1], max_lat_len, device=device)
                    for i, l in enumerate(latents):
                        full_latent[i, :, :l.shape[-1]] = l.squeeze(0)
                    full_latent = full_latent.transpose(1, 2)
                    B, T, D = full_latent.shape
                    ref_len = T // 2
                    cond = torch.zeros_like(full_latent)
                    cond[:, :ref_len, :] = full_latent[:, :ref_len, :]
                t_d2 = time.time()

                with torch.no_grad():
                    mels = []
                    for w in wavs:
                        m = mel_spec_extract(audio=w, sr=44100)
                        mels.append(m.squeeze(0) if m.dim() == 3 else m)
                    max_mel_len = max(m.shape[-1] for m in mels)
                    mel_tensor = torch.zeros(len(mels), mels[0].shape[0], max_mel_len, device=device)
                    for i, m in enumerate(mels):
                        mel_tensor[i, :, :m.shape[-1]] = m
                    midi_p, bound_p = midi_teacher(mel_tensor.transpose(1, 2))
                    if midi_p.shape[1] != T:
                        midi_p = F.interpolate(midi_p.transpose(1, 2), size=T, mode="linear",
                                               align_corners=False).transpose(1, 2)
                    midi = raw_model.smoothMelody_MIDIFuzzDisturb(midi_p)
                    midi[:, :ref_len, :] = 0
                t_d3 = time.time()

                with torch.no_grad():
                    all_token_ids = []
                    for i, text in enumerate(batch["text"]):
                        tokens = tokenizer.encode(text)
                        lrc_token = tokens[:T]
                        all_token_ids.append(lrc_token)
                    aligned_text = torch.zeros(B, T, dtype=torch.long, device=device)
                    for i, tids in enumerate(all_token_ids):
                        n = min(len(tids), T)
                        aligned_text[i, :n] = torch.tensor(tids[:n], device=device)
                t_d4 = time.time()

                t_val = torch.rand(B, device=device)
                noise = torch.randn_like(full_latent)
                x_t = (1 - t_val[:, None, None]) * noise + t_val[:, None, None] * full_latent
                v_target = full_latent - noise
                drop_audio = random.random() < 0.3
                drop_text = random.random() < 0.3
                drop_midi = random.random() < 0.3
                v_pred, hidden_states = run_dit(x_t, cond, aligned_text, t_val, midi,
                                                 drop_audio, drop_text, drop_midi)
                L_flow = F.mse_loss(v_pred, v_target)
                t_d5 = time.time()

                loss = L_flow / args.grad_accum
                if args.cka_weight > 0:
                    midi_for_cka = midi_p[:, :T, :]
                    L_cka = compute_cka_loss_from_hidden(hidden_states, midi_for_cka)
                    loss = loss + args.cka_weight * L_cka / args.grad_accum
                else:
                    L_cka = torch.tensor(0.0, device=device)
                loss.backward()
                torch.cuda.synchronize(device)
                t_d6 = time.time()

                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ema.update()
                torch.cuda.synchronize(device)
                t_d7 = time.time()

                timer_data  += t_d1 - t_d0
                timer_vae   += t_d2 - t_d1
                timer_some  += t_d3 - t_d2
                timer_token += t_d4 - t_d3
                timer_fwd   += t_d5 - t_d4
                timer_bwd   += t_d6 - t_d5
                timer_opt   += t_d7 - t_d6
                timer_steps += 1
                accum_flow  += L_flow.item()
                accum_cka   += L_cka.item()
                global_step += 1
            else:
                wavs = [b.to(device) for b in batch["wav"]]
                loss, loss_dict = compute_loss(wavs, batch["sr"], batch["text"], batch["language"])
                loss = loss / args.grad_accum
                loss.backward()
                accum_flow += loss_dict["flow"]
                accum_cka += loss_dict["cka"]

        if not args.profile_timing:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update()
            global_step += 1

        if global_step % 100 == 0:
            torch.cuda.empty_cache()

        if args.profile_timing and global_step % 50 == 0 and is_main():
            s = timer_steps
            print(f"  [PROFILE step={global_step}] "
                  f"data={timer_data/s:.3f}s vae={timer_vae/s:.3f}s some={timer_some/s:.3f}s "
                  f"token={timer_token/s:.3f}s fwd={timer_fwd/s:.3f}s bwd={timer_bwd/s:.3f}s "
                  f"opt={timer_opt/s:.3f}s")
            timer_data = timer_vae = timer_some = timer_token = timer_fwd = timer_bwd = timer_opt = 0.0
            timer_steps = 0

        if global_step % args.log_every == 0 and is_main():
            avg_flow = accum_flow / (args.log_every * args.grad_accum)
            avg_cka = accum_cka / (args.log_every * args.grad_accum)
            elapsed = time.time() - t_start
            sec_per_step = elapsed / global_step
            eta = (args.max_steps - global_step) * sec_per_step
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m"
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{global_step:6d}/{args.max_steps} | {100*global_step/args.max_steps:.1f}% | ETA {eta_str} | {sec_per_step:.2f}s/step] "
                  f"Flow={avg_flow:.4f} CKA={avg_cka:.4f} | LR={lr_now:.2e}")
            accum_flow = accum_cka = 0.0

        if global_step % args.eval_every == 0 and eval_loader is not None and is_main():
            ev_loss, ev_flow, ev_cka = run_eval()
            print(f"  >>> EVAL step={global_step} | Loss={ev_loss:.4f} (Flow={ev_flow:.4f} CKA={ev_cka:.4f}) <<<")

        if global_step % args.save_every == 0 and is_main():
            save_path = os.path.join(args.output_dir, f"step_{global_step:06d}.pt")
            torch.save({
                "model_state_dict": raw_model.state_dict(),
                "ema_model_state_dict": ema.ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step, "args": vars(args),
            }, save_path)
            print(f"  [Save  {global_step:6d}] {save_path}")

    if is_main():
        save_path = os.path.join(args.output_dir, f"step_{global_step:06d}_final.pt")
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "ema_model_state_dict": ema.ema_model.state_dict(),
            "global_step": global_step, "args": vars(args),
        }, save_path)
        print(f"[Train V3] Final: {save_path}")
        print("[Train V3] Complete!")

if __name__ == "__main__":
    main()

















