"""Independently audit a V4M M600-D transition checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

import torch


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src").is_dir():
    YING_REPO = PROJECT_DIR
sys.path.insert(0, str(YING_REPO))

from src.YingMusicSinger.melody.midi_p_v4ph import (  # noqa: E402
    fill_unsupported_pitch_rows,
)


TRANSITION_SCHEMA = "v4m_m600d_transition_v1"
SOURCE_SCHEMA = "v4ph_training_checkpoint_v1"
SOURCE_DEPTH = 22
TARGET_DEPTH = 40
TARGET_DIT_PARAMETERS = 602_599_648
P_KEY = "midi_p_v4ph.embedding.weight"
BLOCK_PATTERN = re.compile(r"^transformer\.transformer_blocks\.(\d+)\.(.+)$")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_sections(state):
    blocks = {}
    non_blocks = {}
    for key, value in state.items():
        match = BLOCK_PATTERN.match(key)
        if match is None:
            non_blocks[key] = value
        else:
            blocks.setdefault(int(match.group(1)), {})[match.group(2)] = value
    return blocks, non_blocks


def all_finite(state):
    for value in state.values():
        if torch.is_tensor(value) and value.is_floating_point():
            if not bool(torch.isfinite(value).all()):
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source_sha256", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    actual_checkpoint_sha = sha256_file(args.checkpoint)
    if actual_checkpoint_sha != args.checkpoint_sha256:
        raise ValueError("M600-D checkpoint SHA256 mismatch")
    actual_source_sha = sha256_file(args.source)
    if actual_source_sha != args.source_sha256:
        raise ValueError("P500 source SHA256 mismatch")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    source = torch.load(args.source, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("checkpoint_schema") != TRANSITION_SCHEMA:
        raise ValueError("M600-D transition schema mismatch")
    if checkpoint.get("run_state") != "transition" or checkpoint.get("global_step") != 0:
        raise ValueError("M600-D transition state/step mismatch")
    if source.get("checkpoint_schema") != SOURCE_SCHEMA:
        raise ValueError("P500 source schema mismatch")
    if source.get("run_state") != "complete" or source.get("global_step") != 500:
        raise ValueError("P500 source state/step mismatch")
    if (source.get("v4ph_training") or {}).get("phase") != "p_only":
        raise ValueError("P500 source is not phase p_only")
    for forbidden in (
        "ema_model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    ):
        if forbidden in checkpoint:
            raise ValueError(f"transition unexpectedly contains {forbidden}")

    metadata = checkpoint.get("v4m_transition") or {}
    if metadata.get("schema") != TRANSITION_SCHEMA:
        raise ValueError("M600-D metadata schema mismatch")
    if metadata.get("source_checkpoint_sha256") != actual_source_sha:
        raise ValueError("M600-D metadata source SHA256 mismatch")
    if metadata.get("source_weight") != "model_state_dict":
        raise ValueError("M600-D source weight policy mismatch")
    if metadata.get("optimizer_policy") != "fresh":
        raise ValueError("M600-D optimizer policy is not fresh")
    if metadata.get("ema_policy") != "fresh_from_transition_weights":
        raise ValueError("M600-D EMA policy mismatch")

    target_state = checkpoint["model_state_dict"]
    source_state = {
        key.replace("module.", ""): value
        for key, value in source["model_state_dict"].items()
    }
    support = source["p_support_counts"].long()
    expected_p = source_state[P_KEY].clone()
    expected_filled = fill_unsupported_pitch_rows(expected_p, support)
    source_state[P_KEY] = expected_p

    target_blocks, target_non_blocks = block_sections(target_state)
    source_blocks, source_non_blocks = block_sections(source_state)
    if sorted(source_blocks) != list(range(SOURCE_DEPTH)):
        raise ValueError("P500 source block inventory mismatch")
    if sorted(target_blocks) != list(range(TARGET_DEPTH)):
        raise ValueError("M600-D target block inventory mismatch")
    block_metadata = metadata.get("block_transition") or {}
    mapping = {
        int(source_index): int(target_index)
        for source_index, target_index in (
            block_metadata.get("source_to_target") or {}
        ).items()
    }
    if sorted(mapping) != list(range(SOURCE_DEPTH)):
        raise ValueError("M600-D block mapping source inventory mismatch")
    if len(set(mapping.values())) != SOURCE_DEPTH:
        raise ValueError("M600-D block mapping targets are not unique")
    new_targets = sorted(set(range(TARGET_DEPTH)) - set(mapping.values()))
    if new_targets != block_metadata.get("new_target_blocks"):
        raise ValueError("M600-D new block inventory mismatch")
    if len(new_targets) != 18:
        raise ValueError("M600-D must contain exactly 18 new blocks")

    copied_tensor_count = 0
    for source_index, target_index in mapping.items():
        source_block = source_blocks[source_index]
        target_block = target_blocks[target_index]
        if set(source_block) != set(target_block):
            raise ValueError(f"mapped block schema mismatch at source {source_index}")
        for suffix, source_tensor in source_block.items():
            if not torch.equal(source_tensor, target_block[suffix]):
                raise ValueError(
                    f"mapped block tensor differs: source={source_index} "
                    f"target={target_index} suffix={suffix}"
                )
            copied_tensor_count += 1
    for target_index in new_targets:
        block = target_blocks[target_index]
        for suffix in ("attn_norm.linear.weight", "attn_norm.linear.bias"):
            if suffix not in block or bool(torch.count_nonzero(block[suffix])):
                raise ValueError(
                    f"new block {target_index} does not have a zero AdaLN gate"
                )

    if set(source_non_blocks) != set(target_non_blocks):
        raise ValueError("M600-D non-block state inventory mismatch")
    for key, source_tensor in source_non_blocks.items():
        if not torch.equal(source_tensor, target_non_blocks[key]):
            raise ValueError(f"M600-D non-block tensor differs: {key}")

    parameter_count = sum(
        value.numel()
        for key, value in target_state.items()
        if key.startswith("transformer.") and torch.is_tensor(value)
    )
    if parameter_count != TARGET_DIT_PARAMETERS:
        raise ValueError("M600-D parameter count mismatch")
    p_metadata = metadata.get("p_transition") or {}
    if p_metadata.get("unsupported_pitch_rows_filled") != expected_filled:
        raise ValueError("M600-D P fill count mismatch")
    if not torch.equal(target_state[P_KEY], expected_p):
        raise ValueError("M600-D transition P differs from expected Phase-B P")
    equivalence = metadata.get("equivalence") or []
    if len(equivalence) != 9 or any(not item.get("exact") for item in equivalence):
        raise ValueError("M600-D equivalence evidence is incomplete")
    if not all_finite(target_state):
        raise ValueError("M600-D transition contains non-finite tensors")

    report = {
        "schema": "v4m_m600d_transition_audit_v1",
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "checkpoint_sha256": actual_checkpoint_sha,
        "source": str(pathlib.Path(args.source).resolve()),
        "source_sha256": actual_source_sha,
        "target_dit_parameters": parameter_count,
        "mapped_source_blocks": len(mapping),
        "new_identity_blocks": len(new_targets),
        "copied_block_tensors": copied_tensor_count,
        "copied_non_block_tensors": len(source_non_blocks),
        "unsupported_pitch_rows_filled": expected_filled,
        "equivalence_cases": len(equivalence),
        "all_equivalence_exact": True,
        "all_finite": True,
        "optimizer_policy": metadata["optimizer_policy"],
        "ema_policy": metadata["ema_policy"],
    }
    output = pathlib.Path(args.report).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
