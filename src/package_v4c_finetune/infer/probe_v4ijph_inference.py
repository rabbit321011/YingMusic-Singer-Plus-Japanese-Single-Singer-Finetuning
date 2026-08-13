#!/usr/bin/env python3
"""Real checkpoint inference gate for V4IjPH full-timeline style conditioning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf


INFER_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = INFER_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
TRAIN_DIR = PACKAGE_DIR / "train"
for candidate in (PROJECT_DIR, PACKAGE_DIR, YING_REPO, TRAIN_DIR):
    value = os.fspath(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from train_v4ijph import (  # noqa: E402
    V4IjPHDataset,
    V4IjPHObjective,
    collate_v4ijph,
)
from v4ijph_contract import V4IJPH_CHECKPOINT_SCHEMA  # noqa: E402
from v4ijph_ins_cache import InsCache, sha256_file  # noqa: E402
from v4ijph_sampling import (  # noqa: E402
    full_timeline_cond,
    load_inference_adapter,
    sample_full_timeline_style,
)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--game-cache-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-manifest-sha256", required=True)
    parser.add_argument("--h-config-fingerprint", required=True)
    parser.add_argument("--ins-eval-cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--cfg-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.cfg_strength < 0:
        raise ValueError("steps must be positive and cfg-strength non-negative")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if payload.get("checkpoint_schema") != V4IJPH_CHECKPOINT_SCHEMA:
        raise ValueError("Inference probe requires a V4IjPH v2 checkpoint")

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

    policy = Singer(
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
    policy.midi_p_v4ph = V4PHMIDIEmbedding(seed=42)
    state = {
        key.replace("module.", "").replace("ema_model.", ""): value
        for key, value in payload["ema_model_state_dict"].items()
    }
    policy.load_state_dict(state, strict=True)
    policy = policy.to(device).eval()
    policy.smoothMelody_MIDIFuzzDisturb = torch.nn.Identity()
    del state

    adapter, adapter_provenance = load_inference_adapter(
        args.checkpoint, use_ema=True, device=device
    )
    vae = StableAudioInfer(
        model_config_path=args.vae_config,
        model_ckpt_path=args.vae_ckpt,
    ).to(device).eval()
    cache = InsCache(
        args.ins_eval_cache,
        expected_manifest_sha256=args.eval_manifest_sha256,
        expected_max_duration_sec=30.0,
    )
    dataset = V4IjPHDataset(
        args.eval_manifest,
        args.eval_manifest_sha256,
        args.h_config_fingerprint,
        30.0,
        args.game_cache_manifest,
        None,
        ins_cache=cache,
    )
    if len(dataset) < 3:
        raise ValueError("Inference probe requires at least three eval rows")
    target = collate_v4ijph([dataset[0]])
    objective = V4IjPHObjective(
        raw_model=policy,
        adapter=adapter,
        vae=vae,
        args=SimpleNamespace(smoke_assertions=True),
        device=device,
        game_cache_to_model_tracks=game_cache_to_model_tracks,
    )
    full_latent, midi, _, text, _, _, _ = objective.process_batch(target)
    frames = int(full_latent.shape[1])
    ref_a_id = dataset.records[1]["SampleId"]
    ref_b_id = dataset.records[2]["SampleId"]
    ins_a = cache.vector(ref_a_id).unsqueeze(0)
    ins_b = cache.vector(ref_b_id).unsqueeze(0)
    cond_a, style_a = full_timeline_cond(adapter, ins_a, frames, null_ins=False)
    cond_b, style_b = full_timeline_cond(adapter, ins_b, frames, null_ins=False)
    cond_null, style_null = full_timeline_cond(
        adapter, ins_a, frames, null_ins=True
    )
    if torch.equal(cond_a, cond_b):
        raise AssertionError("Two distinct R samples produced identical style cond")
    if bool(torch.count_nonzero(cond_null)) or bool(torch.count_nonzero(style_null)):
        raise AssertionError("Null INS condition is not exactly zero")

    def sample(cond):
        output, _ = sample_full_timeline_style(
            policy,
            style_cond=cond,
            text=text,
            midi=midi,
            duration=frames,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            seed=args.seed,
        )
        if not torch.isfinite(output).all():
            raise FloatingPointError("Inference probe produced non-finite latent")
        return output

    output_a1 = sample(cond_a)
    output_a2 = sample(cond_a)
    output_b = sample(cond_b)
    output_null = sample(cond_null)
    if not torch.equal(output_a1, output_a2):
        raise AssertionError("Same R/target/seed inference is not bit deterministic")
    if torch.equal(output_a1, output_b):
        raise AssertionError("Different R samples produced identical output latent")

    report = {
        "schema": "v4ijph_v2_real_inference_probe_v1",
        "checkpoint": adapter_provenance,
        "target_sample_id": target["sample_id"][0],
        "reference_a": cache.records[1],
        "reference_b": cache.records[2],
        "frames": frames,
        "steps": args.steps,
        "cfg_strength": args.cfg_strength,
        "seed": args.seed,
        "style_l2": {
            "reference_a": float(style_a.norm().detach().cpu()),
            "reference_b": float(style_b.norm().detach().cpu()),
            "null": float(style_null.norm().detach().cpu()),
        },
        "tensor_sha256": {
            "cond_a": tensor_sha256(cond_a),
            "cond_b": tensor_sha256(cond_b),
            "cond_null": tensor_sha256(cond_null),
            "output_a": tensor_sha256(output_a1),
            "output_b": tensor_sha256(output_b),
            "output_null": tensor_sha256(output_null),
        },
        "runtime_sha256": {
            "probe": sha256_file(__file__),
            "sampling": sha256_file(INFER_DIR / "v4ijph_sampling.py"),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    cache.close()


if __name__ == "__main__":
    main()
