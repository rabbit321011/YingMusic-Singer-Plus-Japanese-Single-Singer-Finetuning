#!/usr/bin/env python3
"""Bit-compare continuous and resumed V4IjPH v2 checkpoints."""

from __future__ import annotations

import argparse
import json

import torch

from v4ijph_contract import V4IJPH_CHECKPOINT_SCHEMA, assert_tree_equal
from v4ijph_ins_cache import sha256_file


SECTIONS = (
    "model_state_dict",
    "ema_model_state_dict",
    "ema_step",
    "ema_initted",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "adapter_state_dict",
    "adapter_optimizer_state_dict",
    "adapter_ema_model_state_dict",
    "adapter_ema_step",
    "adapter_ema_initted",
    "adapter_sha256",
    "adapter_ema_sha256",
    "initial_adapter_sha256",
    "rank_states",
    "p_support_counts",
    "p_distance_history",
    "initial_p_weight",
    "initial_frozen_fingerprint",
    "phase_a_init_audit",
    "pul_embedding_init",
    "trainable_parameter_names",
    "transition_p_sha256",
    "v4ijph_training",
    "midi_p_schema",
    "game_cache_schema",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()

    left = torch.load(args.left, map_location="cpu", weights_only=False, mmap=True)
    right = torch.load(args.right, map_location="cpu", weights_only=False, mmap=True)
    for label, checkpoint in (("left", left), ("right", right)):
        if checkpoint.get("checkpoint_schema") != V4IJPH_CHECKPOINT_SCHEMA:
            raise ValueError(f"{label} is not a V4IjPH v2 checkpoint")
        if int(checkpoint.get("global_step", -1)) != args.expected_step:
            raise ValueError(f"{label} step mismatch")
    for section in SECTIONS:
        assert_tree_equal(left[section], right[section], section)
    report = {
        "bit_exact": True,
        "step": args.expected_step,
        "sections": list(SECTIONS),
        "left_sha256": sha256_file(args.left),
        "right_sha256": sha256_file(args.right),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
