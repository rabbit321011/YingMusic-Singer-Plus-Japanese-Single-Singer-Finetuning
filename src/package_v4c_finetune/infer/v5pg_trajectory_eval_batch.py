#!/usr/bin/env python3
"""Run the frozen V5-Pg protocol on every saved 1k Phase-C checkpoint."""

from __future__ import annotations

import v5p_eval_batch as base


ALLOWED_STEPS = tuple(range(1000, 10001, 1000))
_strict_checkpoint_metadata = base.strict_checkpoint_metadata


def strict_checkpoint_metadata(
    checkpoint,
    runtime,
    vae_ckpt,
    game_cache_manifest,
    game_model,
    midi_p_schema,
    game_cache_schema,
):
    return _strict_checkpoint_metadata(
        checkpoint=checkpoint,
        runtime=runtime,
        vae_ckpt=vae_ckpt,
        game_cache_manifest=game_cache_manifest,
        game_model=game_model,
        midi_p_schema=midi_p_schema,
        game_cache_schema=game_cache_schema,
        allowed_steps=ALLOWED_STEPS,
        checkpoint_schema="v5pg_training_checkpoint_v1",
        complete_step=10000,
        ema_step_offset=40000,
        expected_values={
            "placement_mode": "phone_pul",
            "phase": "g_adapt",
            "midi_teacher": "GAME medium K4 offline cache",
            "midi_fuzz_disturb": False,
            "schedule_profile": "v5pg_warmup_hold_cosine",
            "warmup_steps": 250,
            "hold_steps": 6000,
            "decay_start": 6250,
            "first_decay_end": 6250,
            "mid_lr": 0.0,
            "max_steps": 10000,
            "save_every": 1000,
            "eval_every": 500,
            "pool_policy": "KEEP_LONG_DEDUP_SHORT",
            "sampling_policy": "NATURAL_RECORD",
            "engineering_joint_probe": False,
            "engineering_g_probe": False,
            "ema_device": "cpu",
        },
        model_label="V5-Pg",
    )


_placement_audit = base.placement_audit


def placement_audit(rendered, regions):
    audit = _placement_audit(rendered, regions)
    audit["schema"] = "aisvc.v5pg-trajectory-placement-audit.v1"
    audit["modelFamily"] = "V5-Pg"
    return audit


base.strict_checkpoint_metadata = strict_checkpoint_metadata
base.placement_audit = placement_audit


if __name__ == "__main__":
    try:
        base.main()
    except Exception as error:
        base.emit("error", message=str(error))
        raise
