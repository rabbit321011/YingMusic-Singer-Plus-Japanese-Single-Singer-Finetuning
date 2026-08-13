import argparse
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YING_REPO = os.path.join(PROJECT, "YingMusic-Singer-Plus-src")
if not os.path.isdir(os.path.join(YING_REPO, "src")):
    YING_REPO = PROJECT
sys.path.insert(0, YING_REPO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(
            YING_REPO, "src", "YingMusicSinger", "config", "YingMusic_Singer.yaml"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt_path", default=None)
    args = parser.parse_args()

    from src.YingMusicSinger.melody.midi_p_v4ph import (
        V4PHMIDIEmbedding,
        structured_pitch_kernel,
    )
    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.utils.common import (
        calculate_similarity_matrix_with_mask,
        cka_loss,
    )

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    torch.manual_seed(42)
    dit = DiT(
        **cfg.model.arch,
        text_num_embeds=cfg.datasets_cfg.text_num_embeds,
        mel_dim=cfg.model.mel_spec.n_mel_channels,
        long_skip_connection=True,
    )
    model = Singer(
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
    model.midi_p_v4ph = V4PHMIDIEmbedding(seed=42)
    if args.ckpt_path:
        checkpoint = torch.load(
            args.ckpt_path, map_location="cpu", weights_only=False
        )
        if "ema_model_state_dict" in checkpoint:
            state_dict = dict(checkpoint["ema_model_state_dict"])
            for key in ("initted", "step"):
                if key not in state_dict:
                    raise RuntimeError(f"Official EMA is missing metadata key: {key}")
                del state_dict[key]
            state_dict = {
                key.replace("ema_model.", ""): value
                for key, value in state_dict.items()
            }
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }
        incompatible = model.load_state_dict(state_dict, strict=False)
        expected_missing = {
            "transformer.long_skip_connection.weight",
            "midi_p_v4ph.embedding.weight",
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Official base incompatible state: "
                f"missing={sorted(incompatible.missing_keys)} "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        projection = model.transformer.input_embed_with_midi.midi_proj.weight
        if not torch.isfinite(projection).all() or projection.abs().sum() == 0:
            raise AssertionError("Official midi_proj is missing, non-finite, or zero")
        probe_source = "official_base"
    else:
        # A fresh DiT zero-initializes midi_proj, which would mask P gradients.
        # Identity isolates graph connectivity when the Official base is unavailable.
        with torch.no_grad():
            projection = model.transformer.input_embed_with_midi.midi_proj
            projection.weight.copy_(torch.eye(projection.weight.shape[0]))
            projection.bias.zero_()
        probe_source = "synthetic_identity_midi_proj"
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.midi_p_v4ph.embedding.weight.requires_grad = True
    model = model.to(device).train()

    batch, frames, latent_dim, ref_len = 1, 32, 64, 8
    full_latent = torch.randn(batch, frames, latent_dim, device=device)
    noise = torch.randn_like(full_latent)
    t = torch.tensor([0.4], device=device)
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * full_latent
    target = full_latent - noise
    cond = torch.zeros_like(full_latent)
    cond[:, :ref_len] = full_latent[:, :ref_len]
    text = torch.zeros(batch, frames, dtype=torch.long, device=device)
    text[:, 10] = 366
    classes = torch.arange(110, 110 + frames, device=device).clamp(max=140)[None]
    midi_full = model.midi_p_v4ph(classes)
    midi = torch.cat(
        [torch.zeros_like(midi_full[:, :ref_len]), midi_full[:, ref_len:]], dim=1
    )

    transformer = model.transformer
    time_emb = transformer.time_embed(t)
    hidden, _ = transformer.get_input_embed(
        x_t,
        cond,
        text,
        midi,
        drop_audio_cond=False,
        drop_text=False,
        drop_midi=False,
        cache=False,
    )
    rope = transformer.rotary_embed.forward_from_seq_len(frames)
    residual = hidden
    last_hidden = []
    for index, block in enumerate(transformer.transformer_blocks):
        hidden = block(hidden, time_emb, mask=None, rope=rope)
        if index >= len(transformer.transformer_blocks) - 3:
            last_hidden.append(hidden)
    hidden = transformer.long_skip_connection(torch.cat((hidden, residual), dim=-1))
    hidden = transformer.norm_out(hidden, time_emb)
    prediction = transformer.proj_out(hidden)

    kernel = structured_pitch_kernel(device=device)[classes]
    cka = 0
    for layer in last_hidden:
        hidden_sim = calculate_similarity_matrix_with_mask(layer[:, ref_len:])
        kernel_sim = calculate_similarity_matrix_with_mask(kernel[:, ref_len:])
        cka = cka + cka_loss(hidden_sim, kernel_sim)
    cka = cka / len(last_hidden)
    flow_b = F.mse_loss(prediction[:, ref_len:], target[:, ref_len:])
    loss = flow_b + 0.7 * cka

    before = model.midi_p_v4ph.embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [model.midi_p_v4ph.embedding.weight], lr=1e-4
    )
    loss.backward()
    gradient = model.midi_p_v4ph.embedding.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
        raise AssertionError("P embedding gradient is missing, non-finite, or zero")
    frozen_grads = [
        name
        for name, parameter in model.named_parameters()
        if name != "midi_p_v4ph.embedding.weight" and parameter.grad is not None
    ]
    if frozen_grads:
        raise AssertionError(f"Frozen parameters received gradients: {frozen_grads}")
    optimizer.step()
    changed = int(
        (before != model.midi_p_v4ph.embedding.weight.detach()).sum().item()
    )
    if changed == 0:
        raise AssertionError("P embedding did not update")
    print(
        f"V4PH phase-A graph passed: source={probe_source} loss={loss.item():.6f} "
        f"flowB={flow_b.item():.6f} cka={cka.item():.6f} changed={changed}"
    )


if __name__ == "__main__":
    main()
