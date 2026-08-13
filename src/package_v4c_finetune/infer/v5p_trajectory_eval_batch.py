#!/usr/bin/env python3
"""Run the frozen V5-P protocol on every saved 2k trajectory checkpoint."""

from __future__ import annotations

import v5p_eval_batch as base


ALLOWED_STEPS = tuple(range(2000, 40001, 2000))
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
    )


_placement_audit = base.placement_audit


def placement_audit(rendered, regions):
    audit = _placement_audit(rendered, regions)
    audit["schema"] = "aisvc.v5p-trajectory-placement-audit.v1"
    audit["modelFamily"] = "V5-P"
    return audit


base.strict_checkpoint_metadata = strict_checkpoint_metadata
base.placement_audit = placement_audit


if __name__ == "__main__":
    try:
        base.main()
    except Exception as error:
        base.emit("error", message=str(error))
        raise
