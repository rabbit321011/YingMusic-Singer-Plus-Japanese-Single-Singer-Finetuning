#!/usr/bin/env python3
"""Real single-GPU two-update probe for the active V4IjPH adapter path."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
for candidate in (PROJECT_DIR, PACKAGE_DIR, YING_REPO):
    value = os.fspath(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from train_v4ijph import (  # noqa: E402
    V4IjPHDataset,
    V4IjPHObjective,
    collate_v4ijph,
    load_and_validate_source,
)
from train_v4iph import seed_everything  # noqa: E402
from v4ijph_contract import (  # noqa: E402
    InsStyleAdapter,
    V4IJPH_ADAPTER_BETAS,
    V4IJPH_ADAPTER_WEIGHT_DECAY,
    V4IJPH_TRANSITION_STEP,
    adapter_lr_for_global_update,
    assert_zero_output_initialization,
    state_dict_sha256,
)
from v4ijph_ins_cache import InsCache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml"
    )
    parser.add_argument(
        "--vae-config",
        default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",
    )
    parser.add_argument(
        "--vae-ckpt",
        default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt",
    )
    parser.add_argument("--source-v4iph-checkpoint", required=True)
    parser.add_argument("--source-v4iph-sha256", required=True)
    parser.add_argument("--game-cache-manifest", required=True)
    parser.add_argument("--ins-train-cache", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--h-config-fingerprint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=pathlib.Path, required=True)
    return parser.parse_args()


def grad_norm(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    if not torch.isfinite(parameter.grad).all():
        raise FloatingPointError("Probe produced a non-finite adapter gradient")
    return float(parameter.grad.norm().detach().cpu())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V4IjPH single-card probe requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_everything(42)

    source_args = SimpleNamespace(
        source_v4iph_checkpoint=args.source_v4iph_checkpoint,
        source_v4iph_sha256=args.source_v4iph_sha256,
        config=args.config,
        vae_config=args.vae_config,
        vae_ckpt=args.vae_ckpt,
        game_cache_manifest=args.game_cache_manifest,
    )
    source = load_and_validate_source(source_args, world_size=4)
    cfg = OmegaConf.load(args.config)
    from src.YingMusicSinger.melody.game_cache_v4ph import (  # noqa: E402
        game_cache_to_model_tracks,
    )
    from src.YingMusicSinger.melody.midi_p_v4ph import (  # noqa: E402
        V4PHMIDIEmbedding,
    )
    from src.YingMusicSinger.models.dit import DiT  # noqa: E402
    from src.YingMusicSinger.models.model import Singer  # noqa: E402
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import (  # noqa: E402
        StableAudioInfer,
    )

    singer = Singer(
        transformer=DiT(
            **cfg.model.arch,
            text_num_embeds=cfg.datasets_cfg.text_num_embeds,
            mel_dim=cfg.model.mel_spec.n_mel_channels,
            long_skip_connection=True,
        ),
        is_tts_pretrain=cfg.model.is_tts_pretrain,
        melody_input_source=cfg.model.melody_input_source,
        cka_disabled=cfg.model.cka_disabled,
        num_channels=None,
        extra_parameters=cfg.extra_parameters,
        mel_spec_kwargs=cfg.model.mel_spec,
        distill_stage=None,
        use_guidance_scale_embed=False,
    )
    singer.midi_p_v4ph = V4PHMIDIEmbedding(seed=42)
    normalized = {
        key.replace("module.", ""): value
        for key, value in source["model_state_dict"].items()
    }
    singer.load_state_dict(normalized, strict=True)
    singer = singer.to(device).train()

    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(42 + 9_173)
        torch.cuda.manual_seed(42 + 9_173)
        adapter = InsStyleAdapter().to(device)
    assert_zero_output_initialization(adapter)
    initial_adapter_sha = state_dict_sha256(adapter)

    vae = StableAudioInfer(
        model_config_path=args.vae_config,
        model_ckpt_path=args.vae_ckpt,
    ).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False

    cache = InsCache(
        args.ins_train_cache,
        expected_manifest_sha256=args.train_manifest_sha256,
        expected_max_duration_sec=30.0,
    )
    dataset = V4IjPHDataset(
        args.train_manifest,
        args.train_manifest_sha256,
        args.h_config_fingerprint,
        30.0,
        args.game_cache_manifest,
        None,
        ins_cache=cache,
    )
    batch = collate_v4ijph([dataset[0]])
    objective_args = SimpleNamespace(
        t_shift=0.5,
        drop_text=0.15,
        flow_b_weight=2.0,
        cka_weight=0.7,
        smoke_assertions=True,
    )
    objective = V4IjPHObjective(
        raw_model=singer,
        adapter=adapter,
        vae=vae,
        args=objective_args,
        device=device,
        game_cache_to_model_tracks=game_cache_to_model_tracks,
    )
    model_optimizer = torch.optim.AdamW(
        singer.parameters(), lr=1.4e-5, betas=(0.9, 0.95), weight_decay=1e-2
    )
    model_optimizer.load_state_dict(source["optimizer_state_dict"])
    adapter_optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=adapter_lr_for_global_update(V4IJPH_TRANSITION_STEP + 1),
        betas=V4IJPH_ADAPTER_BETAS,
        weight_decay=V4IJPH_ADAPTER_WEIGHT_DECAY,
    )
    del source

    updates = []
    for global_update in (8001, 8002):
        model_optimizer.zero_grad(set_to_none=True)
        adapter_optimizer.zero_grad(set_to_none=True)
        adapter_optimizer.param_groups[0]["lr"] = adapter_lr_for_global_update(
            global_update
        )
        loss, metrics = objective.compute_loss(batch, force_ins_drop=False)
        loss.backward()
        gradients = {
            "input": grad_norm(adapter.input_projection.weight),
            "hidden": grad_norm(adapter.hidden_projection.weight),
            "output": grad_norm(adapter.output_projection.weight),
        }
        if gradients["output"] <= 0:
            raise AssertionError("Adapter output projection received no gradient")
        if global_update == 8002 and (
            gradients["input"] <= 0 or gradients["hidden"] <= 0
        ):
            raise AssertionError("Adapter hidden projections did not activate on update 2")
        torch.nn.utils.clip_grad_norm_(
            list(singer.parameters()) + list(adapter.parameters()), 1.0
        )
        model_optimizer.step()
        adapter_optimizer.step()
        updates.append(
            {
                "global_update": global_update,
                "loss": float(loss.detach().cpu()),
                "flow_b": metrics["flow_b"],
                "cka": metrics["cka"],
                "style_l2": metrics["style_l2"],
                "adapter_lr": adapter_optimizer.param_groups[0]["lr"],
                "gradient_norms": gradients,
                "adapter_sha256": state_dict_sha256(adapter),
            }
        )
    if state_dict_sha256(adapter) == initial_adapter_sha:
        raise AssertionError("Single-card probe did not update the adapter")

    report = {
        "schema": "v4ijph_v2_single_card_probe_v1",
        "source_global_step": V4IJPH_TRANSITION_STEP,
        "sample_id": batch["sample_id"][0],
        "initial_adapter_sha256": initial_adapter_sha,
        "final_adapter_sha256": state_dict_sha256(adapter),
        "updates": updates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    cache.close()


if __name__ == "__main__":
    main()
