#!/usr/bin/env python3
"""Independent schema, schedule, provenance, and state auditor for V4IjPH v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import torch


TRAIN_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = TRAIN_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
for candidate in (PROJECT_DIR, PACKAGE_DIR, YING_REPO):
    value = os.fspath(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from v4ijph_contract import (  # noqa: E402
    InsStyleAdapter,
    V4IJPH_CHECKPOINT_SCHEMA,
    V4IJPH_MAX_STEPS,
    V4IJPH_SOURCE_CHECKPOINT_SCHEMA,
    V4IJPH_TRANSITION_STEP,
    adapter_lr_for_global_update,
    adapter_updates_for_global_step,
    assert_zero_output_initialization,
    state_dict_sha256,
    strict_load_module,
)
from v4ijph_ins_cache import sha256_file  # noqa: E402


P_KEY = "midi_p_v4ph.embedding.weight"


def all_finite(tree) -> bool:
    if torch.is_tensor(tree):
        return not tree.is_floating_point() or bool(torch.isfinite(tree).all())
    if isinstance(tree, dict):
        return all(all_finite(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return all(all_finite(value) for value in tree)
    return True


def load_adapter(state_dict, label):
    adapter = InsStyleAdapter()
    strict_load_module(adapter, state_dict, label)
    return adapter


def optimizer_steps(optimizer_state: dict) -> set[int]:
    result = set()
    for state in optimizer_state.get("state", {}).values():
        step = state.get("step")
        if torch.is_tensor(step):
            step = int(step.item())
        result.add(int(step))
    return result


def current_runtime_paths(checkpoint: dict) -> dict[str, pathlib.Path]:
    args = checkpoint["args"]

    def project_path(value):
        path = pathlib.Path(value)
        return path if path.is_absolute() else PROJECT_DIR / path

    paths = {
        "training_code": TRAIN_DIR / "train_v4ijph.py",
        "contract_code": TRAIN_DIR / "v4ijph_contract.py",
        "ins_cache_code": TRAIN_DIR / "v4ijph_ins_cache.py",
        "placement_code": PACKAGE_DIR / "h_alignment" / "placement_v4iph.py",
        "inference_sampling_code": PACKAGE_DIR / "infer" / "v4ijph_sampling.py",
        "inference_probe_code": PACKAGE_DIR / "infer" / "probe_v4ijph_inference.py",
        "model_config": project_path(args["config"]),
        "vae_config": project_path(args["vae_config"]),
        "vae_checkpoint": project_path(args["vae_ckpt"]),
        "game_cache_manifest": project_path(args["game_cache_manifest"]),
        "source_v4iph_checkpoint": project_path(args["source_v4iph_checkpoint"]),
    }
    for prefix, root_key in (
        ("ins_train_cache", "ins_train_cache"),
        ("ins_eval_cache", "ins_eval_cache"),
    ):
        root = pathlib.Path(args[root_key])
        paths[f"{prefix}_metadata"] = root / "metadata.json"
        for name in (
            "records.jsonl",
            "embeddings.f16.npy",
            "status.u8.npy",
            "waveform_sha256.s64.npy",
            "waveform_samples.i64.npy",
        ):
            paths[f"{prefix}_{name.replace('.', '_')}"] = root / name
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument(
        "--expected-run-state",
        choices=("running", "stopped", "complete"),
        required=True,
    )
    parser.add_argument("--source-v4iph-checkpoint", required=True)
    parser.add_argument("--source-v4iph-sha256", required=True)
    args = parser.parse_args()

    if sha256_file(args.source_v4iph_checkpoint) != args.source_v4iph_sha256:
        raise ValueError("Source V4IPH checkpoint SHA256 mismatch")
    source = torch.load(
        args.source_v4iph_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if source.get("checkpoint_schema") != V4IJPH_SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError("Source checkpoint is not V4IPH")
    if int(source.get("global_step", -1)) != V4IJPH_TRANSITION_STEP:
        raise ValueError("Source checkpoint is not V4IPH step 8000")
    if checkpoint.get("checkpoint_schema") != V4IJPH_CHECKPOINT_SCHEMA:
        raise ValueError("Checkpoint is not V4IjPH v2")
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise ValueError("V4IjPH checkpoint step mismatch")
    if checkpoint.get("run_state") != args.expected_run_state:
        raise ValueError("V4IjPH checkpoint run_state mismatch")
    if args.expected_step == V4IJPH_MAX_STEPS and args.expected_run_state != "complete":
        raise ValueError("V4IjPH step 30000 must be complete")

    metadata = checkpoint.get("v4ijph_training")
    if not isinstance(metadata, dict) or metadata.get("schema") != V4IJPH_CHECKPOINT_SCHEMA:
        raise ValueError("V4IjPH training metadata is missing")
    transition = metadata.get("transition") or {}
    expected_transition = {
        "source_schema": V4IJPH_SOURCE_CHECKPOINT_SCHEMA,
        "source_global_step": V4IJPH_TRANSITION_STEP,
        "source_sha256": args.source_v4iph_sha256,
        "policy": "exact_model_ema_optimizer_scheduler_rank_rng_and_data_cursor",
        "replayed_prefix_steps": 0,
    }
    if transition != expected_transition:
        raise ValueError("V4IjPH transition metadata mismatch")
    if metadata.get("reference_mode") != "all_b":
        raise ValueError("V4IjPH reference mode is not all_b")
    if (metadata.get("ins") or {}).get("style_guidance") != 1.0:
        raise ValueError("V4IjPH style guidance is not fixed at 1")

    recorded_hashes = metadata.get("file_sha256") or {}
    runtime_paths = current_runtime_paths(checkpoint)
    current_hashes = {}
    for label, path in runtime_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        current_hashes[label] = sha256_file(path)
        if recorded_hashes.get(label) != current_hashes[label]:
            raise ValueError(f"V4IjPH runtime SHA256 mismatch: {label}")

    step = args.expected_step
    updates = adapter_updates_for_global_step(step)
    if int(checkpoint["scheduler_state_dict"].get("last_epoch", -1)) != step:
        raise ValueError("DiT scheduler step mismatch")
    if int(checkpoint["ema_step"].item()) != step:
        raise ValueError("DiT EMA step mismatch")
    if len(checkpoint["optimizer_state_dict"].get("state", {})) != 947:
        raise ValueError("DiT optimizer state is incomplete")

    model = checkpoint["model_state_dict"]
    if tuple(model[P_KEY].shape) != (257, 128):
        raise ValueError("V4IjPH P embedding shape mismatch")
    if not torch.equal(model[P_KEY][256], torch.zeros_like(model[P_KEY][256])):
        raise ValueError("V4IjPH P PAD row is nonzero")
    if checkpoint.get("transition_p_sha256") != tensor_sha256(
        source["model_state_dict"][P_KEY]
    ):
        raise ValueError("V4IjPH transition P fingerprint mismatch")

    adapter = load_adapter(checkpoint["adapter_state_dict"], "raw adapter")
    adapter_ema = load_adapter(
        checkpoint["adapter_ema_model_state_dict"], "adapter EMA"
    )
    if state_dict_sha256(adapter) != checkpoint.get("adapter_sha256"):
        raise ValueError("V4IjPH adapter SHA256 mismatch")
    if state_dict_sha256(adapter_ema) != checkpoint.get("adapter_ema_sha256"):
        raise ValueError("V4IjPH adapter EMA SHA256 mismatch")

    torch.manual_seed(42 + 9_173)
    initial_adapter = InsStyleAdapter()
    assert_zero_output_initialization(initial_adapter)
    initial_sha = state_dict_sha256(initial_adapter)
    if checkpoint.get("initial_adapter_sha256") != initial_sha:
        raise ValueError("V4IjPH initial adapter fingerprint mismatch")
    if updates > 0 and state_dict_sha256(adapter) == initial_sha:
        raise ValueError("V4IjPH adapter did not update")

    adapter_optimizer = checkpoint["adapter_optimizer_state_dict"]
    if len(adapter_optimizer.get("param_groups", [])) != 1:
        raise ValueError("V4IjPH adapter optimizer must have one parameter group")
    if len(adapter_optimizer.get("state", {})) != 6:
        raise ValueError("V4IjPH adapter optimizer state is incomplete")
    state_steps = optimizer_steps(adapter_optimizer)
    if state_steps != {updates}:
        raise ValueError(
            f"V4IjPH adapter Adam steps {sorted(state_steps)} != expected {updates}"
        )
    expected_lr = adapter_lr_for_global_update(step)
    actual_lr = float(adapter_optimizer["param_groups"][0]["lr"])
    if actual_lr != expected_lr:
        raise ValueError(f"V4IjPH adapter LR {actual_lr} != expected {expected_lr}")
    if int(checkpoint["adapter_ema_step"].item()) != updates:
        raise ValueError("V4IjPH adapter EMA step mismatch")
    expected_initted = updates > 0
    if bool(checkpoint["adapter_ema_initted"].item()) != expected_initted:
        raise ValueError("V4IjPH adapter EMA initted flag mismatch")

    rank_states = checkpoint.get("rank_states") or []
    world_size = int(metadata["world_size"])
    if sorted(int(item["rank"]) for item in rank_states) != list(range(world_size)):
        raise ValueError("V4IjPH rank states are incomplete")
    for rank_state in rank_states:
        metrics = rank_state.get("metrics") or {}
        if float(metrics.get("flow_a", float("nan"))) != 0.0:
            raise ValueError("V4IjPH rank metrics contain nonzero FlowA")
        if "style_l2" not in metrics or "ins_dropped" not in metrics:
            raise ValueError("V4IjPH rank metrics omit INS fields")
    if not all_finite(checkpoint):
        raise ValueError("V4IjPH checkpoint contains non-finite tensors")

    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "step": step,
        "run_state": checkpoint["run_state"],
        "adapter_updates": updates,
        "adapter_lr": actual_lr,
        "adapter_sha256": checkpoint["adapter_sha256"],
        "adapter_ema_sha256": checkpoint["adapter_ema_sha256"],
        "runtime_sha256": current_hashes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
