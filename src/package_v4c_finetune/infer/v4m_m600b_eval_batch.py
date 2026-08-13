#!/usr/bin/env python3
"""Run the frozen V4PH evaluation protocol with an M600-B publish checkpoint."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import v4ph_eval_batch as base


PUBLISH_SCHEMA = "v4m_m600b_publish_checkpoint_v1"
TRAINING_SCHEMA = "v4m_m600b_fsdp_checkpoint_v1"
TARGET_DEPTH = 31
TARGET_HEADS = 24
TARGET_FF_MULT = 3
TARGET_DIT_STATE_ELEMENTS = 600_462_048


def strict_checkpoint_metadata(
    checkpoint,
    runtime,
    vae_ckpt,
    game_cache_manifest,
    game_model,
    midi_p_schema,
    game_cache_schema,
):
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("checkpoint_schema") != PUBLISH_SCHEMA:
        raise ValueError(
            f"unexpected M600-B checkpoint schema: {payload.get('checkpoint_schema')}"
        )
    step = int(payload.get("global_step", -1))
    if step != 30000 or payload.get("run_state") != "complete":
        raise ValueError(
            f"M600-B publish state mismatch at step {step}: {payload.get('run_state')}"
        )
    if int(payload.get("ema_step", -1)) != step or not payload.get("ema_initted"):
        raise ValueError("M600-B EMA metadata does not match the publish step")
    if payload.get("source_checkpoint_schema") != TRAINING_SCHEMA:
        raise ValueError("M600-B source checkpoint schema mismatch")
    if int(payload.get("target_dit_state_elements", -1)) != TARGET_DIT_STATE_ELEMENTS:
        raise ValueError("M600-B publish parameter count mismatch")

    contract = payload.get("v4m_training")
    if not isinstance(contract, dict) or contract.get("schema") != TRAINING_SCHEMA:
        raise ValueError("M600-B publish lacks its frozen training contract")
    expected_arch = {
        "dim": 1024,
        "depth": TARGET_DEPTH,
        "heads": TARGET_HEADS,
        "dim_head": 64,
        "ff_mult": TARGET_FF_MULT,
    }
    if contract.get("architecture") != expected_arch:
        raise ValueError("M600-B architecture contract mismatch")
    if int(contract.get("target_dit_parameters", -1)) != TARGET_DIT_STATE_ELEMENTS:
        raise ValueError("M600-B training parameter count mismatch")
    expected_contract = {
        "precision": "fp32",
        "world_size": 4,
        "effective_batch": 16,
        "grad_accum": 4,
        "lr": 1.4e-5,
        "warmup_steps": 500,
        "hold_steps": 23500,
        "flow_b_weight": 2.0,
        "cka_weight": 0.7,
        "drop_text": 0.15,
        "seed": 42,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"M600-B contract {key} mismatch")

    files = contract.get("files") or {}
    actual_files = {
        "model_config": base.sha256_file(runtime / "YingMusic_Singer.yaml"),
        "vae_config": base.sha256_file(
            runtime / "stable_audio_2_0_vae_20hz_official.json"
        ),
        "vae_checkpoint": base.sha256_file(vae_ckpt),
        "game_cache_manifest": base.sha256_file(game_cache_manifest),
    }
    for key, actual in actual_files.items():
        if files.get(key) != actual:
            raise ValueError(f"M600-B {key} SHA256 mismatch: {actual}")

    game_manifest = json.loads(Path(game_cache_manifest).read_text(encoding="utf-8"))
    if game_manifest.get("cache_schema") != game_cache_schema:
        raise ValueError("training GAME manifest schema mismatch")
    if game_manifest.get("game_model_sha256") != base.EXPECTED_GAME_MODEL_SHA256:
        raise ValueError("training GAME model hash mismatch")
    if base.sha256_file(game_model) != base.EXPECTED_GAME_MODEL_SHA256:
        raise ValueError("inference GAME model hash mismatch")
    if not isinstance(midi_p_schema, dict) or {
        "version": midi_p_schema.get("version"),
        "source": midi_p_schema.get("source"),
        "num_embeddings": midi_p_schema.get("num_embeddings"),
        "embedding_dim": midi_p_schema.get("embedding_dim"),
        "fuzz_disturb": midi_p_schema.get("fuzz_disturb"),
    } != {
        "version": 1,
        "source": "OpenVPI/GAME medium K4",
        "num_embeddings": 257,
        "embedding_dim": 128,
        "fuzz_disturb": False,
    }:
        raise ValueError("unexpected MIDI_P runtime schema")
    if not isinstance(game_cache_schema, dict) or game_cache_schema.get(
        "game_commit"
    ) != base.EXPECTED_GAME_COMMIT:
        raise ValueError("unexpected GAME cache runtime schema")
    return payload


def build_model(
    checkpoint,
    runtime,
    singer_root,
    vae_ckpt,
    game_cache_manifest,
    game_model,
    device,
):
    import torch
    from omegaconf import OmegaConf
    from src.YingMusicSinger.melody.game_cache_v4ph import GAME_CACHE_SCHEMA
    from src.YingMusicSinger.melody.midi_p_v4ph import MIDI_P_V4PH_SCHEMA
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer

    checkpoint_payload = strict_checkpoint_metadata(
        checkpoint=checkpoint,
        runtime=runtime,
        vae_ckpt=vae_ckpt,
        game_cache_manifest=game_cache_manifest,
        game_model=game_model,
        midi_p_schema=MIDI_P_V4PH_SCHEMA,
        game_cache_schema=GAME_CACHE_SCHEMA,
    )
    train_dir = Path(singer_root) / "package_v4c_finetune" / "train"
    if str(train_dir) not in base.sys.path:
        base.sys.path.insert(0, str(train_dir))
    from prepare_v4m_m600b_transition import (
        build_singer,
        strict_load,
        transformer_parameter_count,
    )

    cfg = OmegaConf.load(runtime / "YingMusic_Singer.yaml")
    policy = build_singer(
        cfg, TARGET_DEPTH, TARGET_HEADS, TARGET_FF_MULT, 42
    )
    state = checkpoint_payload.get("ema_model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("M600-B publish lacks ema_model_state_dict")
    if transformer_parameter_count(state) != TARGET_DIT_STATE_ELEMENTS:
        raise ValueError("M600-B EMA state parameter count mismatch")
    strict_load(policy, state, "M600-B published EMA")
    del checkpoint_payload, state
    gc.collect()
    policy = policy.to(device).eval()
    policy.smoothMelody_MIDIFuzzDisturb = torch.nn.Identity()
    p_weight = policy.midi_p_v4ph.embedding.weight.detach()
    if tuple(p_weight.shape) != (257, 128) or not torch.equal(
        p_weight[256], torch.zeros_like(p_weight[256])
    ):
        raise ValueError("M600-B P embedding shape/PAD row is invalid")

    vae = StableAudioInfer(
        model_config_path=str(runtime / "stable_audio_2_0_vae_20hz_official.json"),
        model_ckpt_path=str(vae_ckpt),
    ).to(device).eval()
    return policy, vae


def placement_audit(rendered, regions):
    audit = _original_placement_audit(rendered, regions)
    audit["schema"] = "aisvc.v4m-m600b-placement-audit.v1"
    audit["modelFamily"] = "V4M-M600-B"
    return audit


_original_placement_audit = base.placement_audit
base.strict_checkpoint_metadata = strict_checkpoint_metadata
base.build_model = build_model
base.placement_audit = placement_audit


if __name__ == "__main__":
    try:
        base.main()
    except Exception as error:
        base.emit("error", message=str(error))
        raise
