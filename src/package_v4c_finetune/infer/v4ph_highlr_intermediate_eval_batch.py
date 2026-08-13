#!/usr/bin/env python3
"""Run the frozen V4PH protocol on the HighLR 6k/12k controls."""

from __future__ import annotations

import json
from pathlib import Path

import v4ph_eval_batch as base


CHECKPOINT_SCHEMA = "v4ph_training_checkpoint_v1"
ALLOWED_STEPS = {6000, 12000}


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
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected HighLR checkpoint schema")
    step = int(payload.get("global_step", -1))
    if payload.get("run_state") != "running" or step not in ALLOWED_STEPS:
        raise ValueError(f"HighLR control must be running step 6k/12k, got {step}")
    if int(payload.get("ema_step", -1)) != step or not payload.get("ema_initted"):
        raise ValueError("HighLR EMA metadata does not match the checkpoint step")

    metadata = payload.get("v4ph_training")
    if not isinstance(metadata, dict) or metadata.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("HighLR checkpoint lacks authoritative training metadata")
    expected_values = {
        "placement_mode": "phone_pul",
        "phase": "joint",
        "midi_teacher": "GAME medium K4 offline cache",
        "midi_fuzz_disturb": False,
        "schedule_profile": "highlr_24k",
        "warmup_steps": 500,
        "hold_steps": 23500,
        "max_steps": 30000,
        "lr": 1.4e-5,
        "grad_accum": 4,
        "world_size": 4,
        "flow_b_weight": 2.0,
        "cka_weight": 0.7,
        "drop_text": 0.15,
        "seed": 42,
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            raise ValueError(f"HighLR metadata {key} mismatch")
    contract = metadata.get("h_pul") or {}
    expected_contract = {
        "pul_token_id": base.PUL_TOKEN,
        "sep_token_id": base.SEP_TOKEN,
        "sep_policy": "next_runtime_control_anchor_minus_one",
        "final_sep_policy": "last_dense_text_frame",
        "pul_policy": "repeat_after_packed_lyrics_until_sep",
        "hard_fallback_policy": "whole_sample_exact_control",
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"HighLR H/PUL contract mismatch for {key}")
    if payload.get("midi_p_schema") != midi_p_schema:
        raise ValueError("HighLR MIDI_P schema differs from inference runtime")
    if payload.get("game_cache_schema") != game_cache_schema:
        raise ValueError("HighLR GAME cache schema differs from inference runtime")

    expected_hashes = metadata.get("file_sha256") or {}
    actual_paths = {
        "training_code": runtime / "train_v4ph.py",
        "placement_code": runtime / "h_alignment/placement.py",
        "model_config": runtime / "YingMusic_Singer.yaml",
        "vae_config": runtime / "stable_audio_2_0_vae_20hz_official.json",
        "vae_checkpoint": vae_ckpt,
        "game_cache_manifest": game_cache_manifest,
    }
    for key, path in actual_paths.items():
        actual = base.sha256_file(path)
        if actual != expected_hashes.get(key):
            raise ValueError(f"HighLR {key} SHA256 mismatch: {actual}: {path}")

    game_manifest = json.loads(Path(game_cache_manifest).read_text(encoding="utf-8"))
    if game_manifest.get("cache_schema") != game_cache_schema:
        raise ValueError("training GAME manifest schema mismatch")
    if game_manifest.get("game_model_sha256") != base.EXPECTED_GAME_MODEL_SHA256:
        raise ValueError("training GAME model hash mismatch")
    if base.sha256_file(game_model) != base.EXPECTED_GAME_MODEL_SHA256:
        raise ValueError("inference GAME model hash mismatch")
    return payload


def placement_audit(rendered, regions):
    audit = _original_placement_audit(rendered, regions)
    audit["schema"] = "aisvc.v4ph-highlr-intermediate-placement-audit.v1"
    audit["modelFamily"] = "V4PH-30K-HIGHLR"
    return audit


_original_placement_audit = base.placement_audit
base.strict_checkpoint_metadata = strict_checkpoint_metadata
base.placement_audit = placement_audit


if __name__ == "__main__":
    try:
        base.main()
    except Exception as error:
        base.emit("error", message=str(error))
        raise
