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
    def __init__(self, records, max_duration_sec=15.0):
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
        B, T, D = h.shape
        b1, t1, d1 = midi_latent.shape
        common_t = min(T, t1)
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
    parser.add_argument("--train_json", default="${REMOTE_ROOT}/final_sum_large/train_singnet.json")
    parser.add_argument("--eval_json", default="${REMOTE_ROOT}/final_sum_large/test_singnet.json")
    parser.add_argument("--output_dir", default="ckpts/plus_ja_ft")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=7e-6)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_duration", type=float, default=15.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit_n", type=int, default=20)
    parser.add_argument("--cka_weight", type=float, default=1.0)
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
        print(f"[Train] World={world_size} Device={device}")
        print(f"[Train] Args: {args}")

    cfg = OmegaConf.load(args.config)

    # ========== Build Model ==========
    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
    from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
    from src.YingMusicSinger.utils.lrc_align import align_lrc_sentence_level

    if is_main():
        print("[Train] Building DiT...")
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
            print(f"[Train] Loading checkpoint: {args.ckpt_path}")
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
        print("[Train] Loading VAE...")
    vae = StableAudioInfer(model_config_path=args.vae_config, model_ckpt_path=args.vae_ckpt)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    if is_main():
        print("[Train] Loading SOME MIDI teacher...")
    midi_teacher = MIDIExtractor(in_dim=80)
    midi_teacher._load_form_ckpt(args.midi_ckpt)
    midi_teacher = midi_teacher.to(device).eval()
    for p in midi_teacher.parameters():
        p.requires_grad = False

    mel_spec_extract = MelodySpectrogram()
    vae_frame_rate = 44100 / 2048
    tokenizer = CNENTokenizer()
    # Direct JA tokenizer (bypass language detection for 100% JA data)
    from src.YingMusicSinger.utils.f5_tts.g2p.g2p import PhonemeBpeTokenizer
    from src.YingMusicSinger.utils.f5_tts.g2p.g2p.japanese import japanese_to_ipa
    pbt = PhonemeBpeTokenizer()
    def encode_ja(text):
        phoneme = japanese_to_ipa(text, None)
        tokens = pbt.phoneme2token(phoneme)
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
            tokens = tokens[0]
        return tokens

    ema = EMA(singer_model, beta=cfg.ema_kwargs.beta,
              update_after_step=cfg.ema_kwargs.update_after_step,
              update_every=cfg.ema_kwargs.update_every)
    ema.to(device)

    if world_size > 1:
        singer_model = DDP(singer_model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = singer_model.module if world_size > 1 else singer_model

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ========== Dataset ==========
    with open(args.train_json) as f:
        train_data = json.load(f)
    if args.overfit:
        train_data = train_data[:args.overfit_n]
        if is_main():
            print(f"[Train] OVERFIT MODE: {len(train_data)} samples")

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
        print(f"[Train] Train: {len(train_dataset)} | Eval: {len(eval_dataset) if eval_dataset else 0}")
        print(f"[Train] Steps/epoch: {len(train_loader)} | Max steps: {args.max_steps}")
        print("=" * 60)

    # ========== Training Helpers ==========
    def process_batch(wavs, srs, texts, langs):
        with torch.no_grad():
            # VAE encode (one per sample)
            latents = []
            for w in wavs:
                lat = vae.encode_audio(w.unsqueeze(0) if w.dim() == 1 else w, in_sr=srs[0])
                latents.append(lat)
            max_lat_len = max(l.shape[-1] for l in latents)
            latent = torch.zeros(len(latents), latents[0].shape[1], max_lat_len, device=device)
            for i, l in enumerate(latents):
                latent[i, :, :l.shape[-1]] = l.squeeze(0)
            latent = latent.transpose(1, 2)
            B, T, D = latent.shape

            # Mel spectrograms → SOME MIDI
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

            # Tokenize + align (direct JA G2P, no language detection)
            all_token_ids = []
            for i, text in enumerate(texts):
                tokens = encode_ja(text)
                lrc_token = tokens[:T]
                all_token_ids.append(lrc_token)
            aligned_text = torch.zeros(B, T, dtype=torch.long, device=device)
            for i, tids in enumerate(all_token_ids):
                n = min(len(tids), T)
                aligned_text[i, :n] = torch.tensor(tids[:n], device=device)

        return latent, midi, midi_p, aligned_text, B, T, D

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
        latent, midi, midi_p, aligned_text, B, T, D = process_batch(wavs, srs, texts, langs)

        t = torch.rand(B, device=device)
        noise = torch.randn_like(latent)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * latent
        v_target = latent - noise

        drop_audio = random.random() < 0.2
        drop_text = random.random() < 0.3
        drop_midi = random.random() < 0.3

        v_pred, hidden_states = run_dit(x_t, latent, aligned_text, t, midi,
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

    # ========== Training Loop ==========
    global_step = 0
    accum_flow = accum_cka = 0.0
    t_start = time.time()
    data_iter = iter(train_loader)

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
            loss, loss_dict = compute_loss(wavs, batch["sr"], batch["text"], batch["language"])
            loss = loss / args.grad_accum
            loss.backward()
            accum_flow += loss_dict["flow"]
            accum_cka += loss_dict["cka"]

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update()
        global_step += 1

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
        print(f"[Train] Final: {save_path}")
        print("[Train] Complete!")

if __name__ == "__main__":
    main()

