import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn as nn


PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YING_REPO = os.path.join(PROJECT, "YingMusic-Singer-Plus-src")
if not os.path.isdir(os.path.join(YING_REPO, "src")):
    YING_REPO = PROJECT
sys.path.insert(0, YING_REPO)

from src.YingMusicSinger.melody.midi_p_v4ph import (  # noqa: E402
    pitch_kernel_distance,
)


P_KEY = "midi_p_v4ph.embedding.weight"
PROJ_WEIGHT = "transformer.input_embed_with_midi.midi_proj.weight"
PROJ_BIAS = "transformer.input_embed_with_midi.midi_proj.bias"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path, expected_step):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema") != "v5p_training_checkpoint_v1":
        raise ValueError(f"Invalid V5P schema: {path}")
    if int(checkpoint.get("global_step", -1)) != expected_step:
        raise ValueError(f"Invalid V5P step: {path}")
    return checkpoint


def distance(checkpoint, support):
    state = checkpoint["model_state_dict"]
    projection = nn.Linear(128, 128)
    with torch.no_grad():
        projection.weight.copy_(state[PROJ_WEIGHT])
        projection.bias.copy_(state[PROJ_BIAS])
    return pitch_kernel_distance(state[P_KEY], projection, support)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step300", required=True)
    parser.add_argument("--step500", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    step300 = load_checkpoint(args.step300, 300)
    step500 = load_checkpoint(args.step500, 500)
    if step300["initial_frozen_fingerprint"] != step500["initial_frozen_fingerprint"]:
        raise ValueError("V5P frozen fingerprints differ between checkpoints")
    support = step500["p_support_counts"].long()
    metrics300 = distance(step300, support)
    metrics500 = distance(step500, support)
    checks = {
        "projected_rmse_decreased": (
            metrics500["projected_rmse"] < metrics300["projected_rmse"]
        ),
        "raw_rmse_decreased": metrics500["raw_rmse"] < metrics300["raw_rmse"],
        "raw_cosine_distance_decreased": (
            metrics500["raw_cosine_distance"]
            < metrics300["raw_cosine_distance"]
        ),
        "pitch_geometry_cka_increased": (
            metrics500["pitch_geometry_cka"] > metrics300["pitch_geometry_cka"]
        ),
    }
    passed = checks["projected_rmse_decreased"]
    step300_sha256 = sha256_file(args.step300)
    step500_sha256 = sha256_file(args.step500)
    report = {
        "schema": "v5p_phase_a_distance_adjudication_v2",
        "decision": "random_p_pass" if passed else "random_p_fail_use_kernel",
        "decision_policy": "projected_rmse_primary_support_metrics_diagnostic",
        "step300_checkpoint": os.path.abspath(args.step300),
        "step300_checkpoint_sha256": step300_sha256,
        "step500_checkpoint": os.path.abspath(args.step500),
        "step500_checkpoint_sha256": step500_sha256,
        "step300": metrics300,
        "step500": metrics500,
        "checks": checks,
        "required_checks": ["projected_rmse_decreased"],
        "diagnostic_checks": [
            "raw_rmse_decreased",
            "raw_cosine_distance_decreased",
            "pitch_geometry_cka_increased",
        ],
        "unsupported_pitch_rows": torch.nonzero(
            support[:255] == 0
        ).flatten().tolist(),
        "next": (
            "fill unsupported pitch rows from the structured kernel, then unlock joint"
            if passed
            else "restart short P calibration from the structured pitch kernel"
        ),
    }
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
