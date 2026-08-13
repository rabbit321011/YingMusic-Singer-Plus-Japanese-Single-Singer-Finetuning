import torch, torchaudio, json, torch.nn.functional as F

device = torch.device("cuda:0")
torch.manual_seed(42)

from omegaconf import OmegaConf
cfg = OmegaConf.load("src/YingMusicSinger/config/YingMusic_Singer.yaml")
from src.YingMusicSinger.models.dit import DiT
from src.YingMusicSinger.models.model import Singer
from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
from src.YingMusicSinger.utils.cnen_tokenizer import CNENTokenizer

dit = DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
          mel_dim=cfg.model.mel_spec.n_mel_channels, long_skip_connection=True)
singer = Singer(transformer=dit, is_tts_pretrain=cfg.model.is_tts_pretrain,
                melody_input_source=cfg.model.melody_input_source,
                cka_disabled=cfg.model.cka_disabled, num_channels=None,
                extra_parameters=cfg.extra_parameters, mel_spec_kwargs=cfg.model.mel_spec,
                distill_stage=None, use_guidance_scale_embed=False)
ckpt = torch.load("ckpts/plus_ja_sft_v3/step_030000_final.pt", map_location="cpu")
sd = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict", ckpt))
sd = {k.replace("ema_model.", "").replace("module.", ""): v for k, v in sd.items()}
singer.load_state_dict(sd, strict=False)
singer = singer.to(device).eval()

vae = StableAudioInfer(model_config_path="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",
                        model_ckpt_path="ckpts/stable_audio_2_0_vae_20hz_official.ckpt")
vae = vae.to(device).eval()
midi_teacher = MIDIExtractor(in_dim=80)
midi_teacher._load_form_ckpt("ckpts/model_ckpt_steps_100000_simplified.ckpt")
midi_teacher = midi_teacher.to(device).eval()
mel_spec_ext = MelodySpectrogram()
tokenizer = CNENTokenizer()

with open("${SERVER_ROOT}/final_sum_large/test_singnet.json") as f:
    data = json.load(f)

rec = data[0]
wav, sr = torchaudio.load(rec["Path"])
if sr != 44100:
    wav = torchaudio.functional.resample(wav, sr, 44100)
wav = wav.to(device)
ref_text = rec["Text"]
print(f"Ref text: {ref_text[:60]}")
print(f"Tokens: {tokenizer.encode(ref_text)[:10]}")

with torch.no_grad():
    w2d = wav.squeeze(0) if wav.dim() == 2 else wav
    if w2d.dim() == 1:
        w2d = w2d.unsqueeze(0)
    lat = vae.encode_audio(w2d, in_sr=44100)
    if lat.dim() == 3:
        lat = lat.squeeze(0)
    full_latent = lat.transpose(0, 1).unsqueeze(0)
    B, T, D = full_latent.shape
    print(f"Latent shape: {full_latent.shape}, T={T}, D={D}")

    ref_len = T // 2
    cond = torch.zeros_like(full_latent)
    cond[:, :ref_len, :] = full_latent[:, :ref_len, :]
    print(f"cond non-zero: {(cond != 0).sum()}/{cond.numel()}")

    mel = mel_spec_ext(audio=w2d, sr=44100)
    if mel.dim() == 3:
        mel = mel.squeeze(0)
    mel = mel.to(device)
    midi_p, _ = midi_teacher(mel.unsqueeze(0).transpose(1, 2))
    if midi_p.shape[1] != T:
        midi_p = F.interpolate(midi_p.transpose(1, 2), size=T, mode="linear",
                               align_corners=False).transpose(1, 2)
    midi = midi_p.clone()
    midi[:, :ref_len, :] = 0

    tokens = tokenizer.encode(ref_text)
    text_ids = torch.zeros(1, T, dtype=torch.long, device=device)
    n = min(len(tokens), T)
    text_ids[0, :n] = torch.tensor(tokens[:n], device=device)
    print(f"Text tokens: {tokens[:10]}")

    noise = torch.randn(1, T, D, device=device)
    x_t = 0.5 * noise + 0.5 * full_latent[:, :1, :].expand(-1, T, -1)
    print(f"x_t stats: mean={x_t.mean():.4f} std={x_t.std():.4f}")

    nfe = 32
    t_vals = torch.linspace(0.5, 1, nfe + 1, device=device)
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
        v = v_cond + (v_cond - v_uncond) * 3.0
        x = x + v * dt_val.item()

    print(f"Generated latent stats: mean={x.mean():.4f} std={x.std():.4f}")
    lat_dec = x.permute(0, 2, 1).float()
    audio_gen = vae.decode_audio(lat_dec)
    print(f"Generated audio stats: mean={audio_gen.mean():.4f} std={audio_gen.std():.4f}")
    print(f"Generated audio range: [{audio_gen.min():.4f}, {audio_gen.max():.4f}]")

torchaudio.save("debug_gen.wav", audio_gen.cpu(), 44100)
torchaudio.save("debug_ref.wav", wav.cpu(), 44100)
print("Saved debug_gen.wav and debug_ref.wav")

















