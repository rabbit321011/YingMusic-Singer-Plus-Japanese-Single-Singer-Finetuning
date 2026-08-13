"""Verify SFT model: forward pass on training sample should give low loss."""
import torch, json, torchaudio, os, sys, warnings; warnings.filterwarnings("ignore")
os.chdir("${REMOTE_ROOT}/YingMusic-Singer-Plus")
sys.path.insert(0, "${REMOTE_ROOT}/YingMusic-Singer-Plus")
import torch.nn.functional as F

device = torch.device("cuda:0")

from omegaconf import OmegaConf
cfg = OmegaConf.load("src/YingMusicSinger/config/YingMusic_Singer.yaml")
from src.YingMusicSinger.models.dit import DiT
from src.YingMusicSinger.models.model import Singer
from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer
from src.YingMusicSinger.utils.f5_tts.g2p.g2p import PhonemeBpeTokenizer
from src.YingMusicSinger.utils.f5_tts.g2p.g2p.japanese import japanese_to_ipa
from src.YingMusicSinger.utils.common import cka_loss, calculate_similarity_matrix_with_mask

pbt = PhonemeBpeTokenizer()
def encode_ja(text):
    phoneme = japanese_to_ipa(text, None)
    tokens = pbt.phoneme2token(phoneme)
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return tokens

vae = StableAudioInfer(
    model_config_path="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",
    model_ckpt_path="ckpts/stable_audio_2_0_vae_20hz_official.ckpt")
vae = vae.to(device).eval()

midi_teacher = MIDIExtractor(in_dim=80)
midi_teacher._load_form_ckpt("ckpts/model_ckpt_steps_100000_simplified.ckpt")
midi_teacher = midi_teacher.to(device).eval()
mel_ext = MelodySpectrogram()

with open("${REMOTE_ROOT}/final_sum_large/train_singnet.json") as f:
    train_data = json.load(f)

# Pick sample 0
rec = train_data[0]
wav, _ = torchaudio.load(rec["Path"])
wav = wav.to(device)
text = rec["Text"]

print(f"Sample: {text[:60]}")

with torch.no_grad():
    lat = vae.encode_audio(wav, in_sr=44100)
    if lat.dim() == 3: lat = lat.squeeze(0)
    latent = lat.transpose(0, 1).unsqueeze(0)
    B, T, D = latent.shape
    print(f"Latent: {list(latent.shape)}")

    m = mel_ext(audio=wav, sr=44100)
    m = m.squeeze(0) if m.dim() == 3 else m
    m = m.to(device)
    midi_p, _ = midi_teacher(m.unsqueeze(0).transpose(1, 2))
    if midi_p.shape[1] != T:
        midi_p = F.interpolate(midi_p.transpose(1, 2), size=T, mode="linear",
                               align_corners=False).transpose(1, 2)

    tokens = encode_ja(text)
    text_ids = torch.zeros(1, T, dtype=torch.long, device=device)
    n = min(len(tokens), T)
    text_ids[0, :n] = torch.tensor(tokens[:n], device=device)

for ckpt_name in ["base", "sft_20k", "sft_30k"]:
    dit = DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
              mel_dim=cfg.model.mel_spec.n_mel_channels, long_skip_connection=True)
    singer = Singer(transformer=dit, is_tts_pretrain=cfg.model.is_tts_pretrain,
                    melody_input_source=cfg.model.melody_input_source,
                    cka_disabled=cfg.model.cka_disabled, num_channels=None,
                    extra_parameters=cfg.extra_parameters,
                    mel_spec_kwargs=cfg.model.mel_spec,
                    distill_stage=None, use_guidance_scale_embed=False)

    ckpt_path = {"base": "ckpts/YingMusicSinger_model.pt",
                 "sft_20k": "ckpts/plus_ja_sft/step_020000.pt",
                 "sft_30k": "ckpts/plus_ja_sft/step_030000_final.pt"}[ckpt_name]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt_name == "base":
        sd = ck["model_state_dict"]
    elif "ema_model_state_dict" in ck:
        sd = ck["ema_model_state_dict"]
        sd = {k.replace("ema_model.", ""): v for k, v in sd.items()}
    else:
        sd = ck["model_state_dict"]
    singer.load_state_dict(sd, strict=False)
    singer = singer.to(device).eval()

    midi_fuzz = singer.smoothMelody_MIDIFuzzDisturb(midi_p)

    with torch.no_grad():
        t = torch.rand(1, device=device)
        noise = torch.randn_like(latent)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * latent
        v_target = latent - noise

        dit_obj = singer.transformer
        time_emb = dit_obj.time_embed(t)
        seq_len = x_t.shape[1]
        x, _ = dit_obj.get_input_embed(x_t, latent, text_ids, midi_fuzz,
                                        drop_audio_cond=False, drop_text=False,
                                        drop_midi=False)
        rope = dit_obj.rotary_embed.forward_from_seq_len(seq_len)
        hidden = []
        residual = x
        for i, block in enumerate(dit_obj.transformer_blocks):
            x = block(x, time_emb, mask=None, rope=rope)
            if i >= len(dit_obj.transformer_blocks) - 3:
                hidden.append(x)
        if dit_obj.long_skip_connection is not None:
            x = dit_obj.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = dit_obj.norm_out(x, time_emb)
        v_pred = dit_obj.proj_out(x)

        L_flow = F.mse_loss(v_pred, v_target).item()

        L_cka = 0.0
        for h in hidden:
            ct = min(h.shape[1], midi_p.shape[1])
            sh = calculate_similarity_matrix_with_mask(h[:, :ct, :])
            sm = calculate_similarity_matrix_with_mask(midi_p[:, :ct, :])
            L_cka += cka_loss(sh, sm).item()
        L_cka /= 3

    print(f"  {ckpt_name:>10s}: Flow={L_flow:.4f} CKA={L_cka:.4f} Total={L_flow+L_cka:.4f}")

print("Done!")

