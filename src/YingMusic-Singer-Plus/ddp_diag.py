import time, random, os, sys, json
import torch, torch.nn.functional as F, torchaudio, numpy as np

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return local_rank

def run(local_rank, world_size, find_unused, label, n_steps=200, max_dur=15.0):
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    
    from omegaconf import OmegaConf; from ema_pytorch import EMA
    cfg = OmegaConf.load("src/YingMusicSinger/config/YingMusic_Singer.yaml")
    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
    from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader
    
    dit = DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
              mel_dim=cfg.model.mel_spec.n_mel_channels, long_skip_connection=True)
    singer = Singer(transformer=dit, is_tts_pretrain=cfg.model.is_tts_pretrain,
                    melody_input_source=cfg.model.melody_input_source,
                    cka_disabled=cfg.model.cka_disabled, num_channels=None,
                    extra_parameters=cfg.extra_parameters, mel_spec_kwargs=cfg.model.mel_spec,
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
    
    singer_ddp = DDP(singer, device_ids=[local_rank], find_unused_parameters=find_unused)
    raw = singer_ddp.module
    
    optimizer = torch.optim.AdamW(singer_ddp.parameters(), lr=7e-6, betas=(0.9, 0.95), weight_decay=1e-2)
    
    with open("${SERVER_ROOT}/final_sum_large/train_singnet.json") as f:
        data = json.load(f)
    random.seed(42 + local_rank)
    random.shuffle(data)
    
    class DS:
        def __init__(self, recs, md):
            self.r = recs; self.md = md
        def __len__(self): return len(self.r)
        def __getitem__(self, i):
            rec = self.r[i]
            w, sr = torchaudio.load(rec["Path"])
            if sr != 44100: w = torchaudio.functional.resample(w, sr, 44100)
            ms = int(self.md * 44100)
            if w.shape[-1] > ms: w = w[:, :ms]
            return w, rec["Text"]
    
    ds = DS(data[:n_steps * 4], max_dur)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=0)
    
    if local_rank == 0:
        print(f"\n[DDP diag] find_unused={find_unused} label='{label}' steps={n_steps} max_dur={max_dur}")
    
    step_times = []
    data_iter = iter(loader)
    sampler.set_epoch(0)
    
    for step in range(n_steps):
        try:
            wav, text = next(data_iter)
        except StopIteration:
            sampler.set_epoch(sampler.epoch + 1)
            data_iter = iter(loader)
            wav, text = next(data_iter)
        
        wav = wav.to(device)
        text = text[0] if isinstance(text, list) else text
        
        t0 = time.time()
        optimizer.zero_grad()
        
        with torch.no_grad():
            w2d = wav.squeeze(0) if wav.dim() == 3 else wav.squeeze(0)
            if w2d.dim() == 1: w2d = w2d.unsqueeze(0)
            lat = vae.encode_audio(wav, in_sr=44100)
            if lat.dim() == 3:
                lat = lat.squeeze(0)
            full_latent = lat.transpose(0, 1).unsqueeze(0)
            B, T, D = full_latent.shape
            ref_len = T // 2
            cond = torch.zeros_like(full_latent); cond[:, :ref_len, :] = full_latent[:, :ref_len, :]
            mel = mel_spec_ext(audio=w2d, sr=44100)
            if mel.dim() == 3: mel = mel.squeeze(0)
            mel = mel.to(device).unsqueeze(0)
            midi_p, _ = midi_teacher(mel.transpose(1, 2))
            if midi_p.shape[1] != T:
                midi_p = F.interpolate(midi_p.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
            midi = raw.smoothMelody_MIDIFuzzDisturb(midi_p)
            midi[:, :ref_len, :] = 0
            tokens = tokenizer.encode(text)
            aligned_text = torch.zeros(1, T, dtype=torch.long, device=device)
            n = min(len(tokens), T)
            aligned_text[0, :n] = torch.tensor(tokens[:n], device=device)
        
        t_v = torch.rand(1, device=device)
        noise = torch.randn_like(full_latent)
        x_t = (1 - t_v[:, None, None]) * noise + t_v[:, None, None] * full_latent
        da = random.random() < 0.3; dt_ = random.random() < 0.3; dm = random.random() < 0.3
        dit_m = raw.transformer
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
        loss = F.mse_loss(v_pred, full_latent - noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(singer_ddp.parameters(), 1.0)
        optimizer.step()
        ema.update()
        torch.cuda.synchronize(device)
        
        t_step = time.time() - t0
        step_times.append(t_step)
        
        if local_rank == 0 and step % 20 == 0:
            a = np.mean(step_times[5:step+1]) if step >= 5 else np.mean(step_times[:step+1])
            print(f"  [{label}] step={step:3d}  {t_step:.2f}s  avg={a:.2f}s  T={T}")
    
    if local_rank == 0:
        first50 = np.mean(step_times[10:60])
        last50 = np.mean(step_times[-50:])
        deg = (last50 - first50) / first50 * 100
        print(f"[{label}] first50={first50:.2f}s last50={last50:.2f}s degradation={deg:+.1f}%")
    
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    local_rank = setup_ddp()
    world_size = torch.distributed.get_world_size()
    label = os.environ.get("FIND_UNUSED", "default")
    find_unused = os.environ.get("FIND_UNUSED", "True").lower() == "true"
    
    if local_rank == 0:
        print(f"[{label}] find_unused_parameters={find_unused}")
    
    run(local_rank, world_size, find_unused, label, n_steps=200, max_dur=15.0)
    
    if local_rank == 0:
        print(f"[{label}] Done.")

















