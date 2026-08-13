import argparse
import hashlib
import json

import numpy as np
import torch


SOURCE_SCHEMA = "v5p_training_checkpoint_v1"
EXPECTED_SCHEMA = "v5pg_training_checkpoint_v1"
P_KEY = "midi_p_v4ph.embedding.weight"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_state_dict(state):
    return {
        key.removeprefix("module.").removeprefix("ema_model."): value
        for key, value in state.items()
    }


def assert_state_dict_equal(left, right, label):
    left = normalized_state_dict(left)
    right = normalized_state_dict(right)
    if set(left) != set(right):
        raise ValueError(f"{label} keys differ")
    mismatches = [
        key
        for key in left
        if left[key].dtype != right[key].dtype
        or left[key].shape != right[key].shape
        or not torch.equal(left[key], right[key])
    ]
    if mismatches:
        raise ValueError(f"{label} values differ: {mismatches[:8]}")


def all_finite(tree):
    if torch.is_tensor(tree):
        return not tree.is_floating_point() or bool(torch.isfinite(tree).all())
    if isinstance(tree, dict):
        return all(all_finite(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return all(all_finite(value) for value in tree)
    return True


def v5pg_lr(step, peak_lr=5e-6, warmup=250, hold=6000, max_steps=10000):
    if step < warmup:
        factor = step / warmup
    elif step < warmup + hold:
        factor = 1.0
    else:
        progress = (step - warmup - hold) / (max_steps - warmup - hold)
        factor = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    return float(peak_lr * factor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--source_sha256", required=True)
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--vae_sha256", required=True)
    parser.add_argument("--expected_step", type=int, required=True)
    parser.add_argument(
        "--expected_run_state", choices=("running", "stopped", "complete")
    )
    parser.add_argument("--expected_sampler_num_samples", type=int)
    parser.add_argument("--engineering-g-probe", action="store_true")
    args = parser.parse_args()

    if sha256_file(args.source_checkpoint) != args.source_sha256:
        raise ValueError("V5-P source checkpoint SHA256 mismatch")
    if sha256_file(args.vae_checkpoint) != args.vae_sha256:
        raise ValueError("285k online VAE SHA256 mismatch")

    source = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if source.get("checkpoint_schema") != SOURCE_SCHEMA:
        raise ValueError("V5-P source schema mismatch")
    if source.get("run_state") != "complete" or int(source.get("global_step", -1)) != 40000:
        raise ValueError("V5-P source is not the complete 40K checkpoint")
    if checkpoint.get("checkpoint_schema") != EXPECTED_SCHEMA:
        raise ValueError("V5-Pg checkpoint schema mismatch")
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise ValueError("V5-Pg checkpoint step mismatch")
    expected_run_state = args.expected_run_state or (
        "complete" if args.expected_step == 10000 else "running"
    )
    if checkpoint.get("run_state") != expected_run_state:
        raise ValueError("V5-Pg run state mismatch")

    source_metadata = source.get("v5p_training")
    metadata = checkpoint.get("v5p_training")
    if not isinstance(source_metadata, dict) or source_metadata.get("phase") != "joint":
        raise ValueError("V5-P source metadata mismatch")
    if not isinstance(metadata, dict) or metadata.get("phase") != "g_adapt":
        raise ValueError("V5-Pg metadata mismatch")
    expected_contract = {
        "schema": EXPECTED_SCHEMA,
        "placement_mode": "phone_pul",
        "pool_policy": "KEEP_LONG_DEDUP_SHORT",
        "sampling_policy": "NATURAL_RECORD",
        "lr": 5e-6,
        "schedule_profile": "v5pg_warmup_hold_cosine",
        "warmup_steps": 250,
        "hold_steps": 6000,
        "decay_start": 6250,
        "first_decay_end": 6250,
        "mid_lr": 0.0,
        "max_steps": 10000,
        "save_every": 1000,
        "eval_every": 500,
        "cka_weight": 0.7,
        "drop_text": 0.15,
        "flow_b_weight": 2.0,
        "ema_device": "cpu",
        "engineering_joint_probe": False,
        "engineering_g_probe": args.engineering_g_probe,
        "midi_p_init": "inherit_v5p_40k_ema",
        "midi_teacher": "GAME medium K4 offline cache",
        "midi_fuzz_disturb": False,
    }
    for key, expected in expected_contract.items():
        if metadata.get(key) != expected:
            raise ValueError(f"V5-Pg contract mismatch: {key}")
    continuity_keys = [
        "h_config_fingerprint",
        "short_manifest_sha256",
        "long_manifest_sha256",
        "pool_audit_sha256",
        "batch_size",
        "max_duration",
        "t_shift",
    ]
    if not args.engineering_g_probe:
        continuity_keys.extend(
            ("train_manifest_sha256", "eval_manifest_sha256", "world_size", "grad_accum")
        )
    for key in continuity_keys:
        if metadata.get(key) != source_metadata.get(key):
            raise ValueError(f"V5-P/V5-Pg continuity mismatch: {key}")

    warmstart = metadata.get("warmstart", {})
    expected_warmstart = {
        "checkpoint_sha256": args.source_sha256,
        "expected_source_step": 40000,
        "weight_source": "ema_model_state_dict",
        "raw_model_policy": "initialize_from_source_ema",
        "ema_policy": "continue_source_ema_weights_and_counters",
        "optimizer_policy": "fresh",
        "scheduler_policy": "fresh_step_zero",
        "data_cursor_policy": "fresh_step_zero",
        "source_checkpoint_schema": SOURCE_SCHEMA,
        "source_phase": "joint",
        "source_run_state": "complete",
    }
    for key, expected in expected_warmstart.items():
        if warmstart.get(key) != expected:
            raise ValueError(f"V5-Pg warm-start mismatch: {key}")
    file_sha256 = metadata.get("file_sha256", {})
    if file_sha256.get("warmstart_checkpoint") != args.source_sha256:
        raise ValueError("metadata source checkpoint SHA256 mismatch")
    if file_sha256.get("vae_checkpoint") != args.vae_sha256:
        raise ValueError("metadata VAE SHA256 mismatch")

    init_audit = checkpoint.get("warmstart_init_audit")
    if not isinstance(init_audit, dict):
        raise ValueError("V5-Pg warm-start initialization audit is missing")
    if init_audit.get("source_checkpoint_sha256") != args.source_sha256:
        raise ValueError("warm-start initialization source mismatch")
    if init_audit.get("source_ema_step") != 40000:
        raise ValueError("warm-start initialization EMA step mismatch")
    if init_audit.get("source_ema_initted") is not True:
        raise ValueError("warm-start initialization EMA flag mismatch")
    if init_audit.get("raw_ema_bit_exact") is not True:
        raise ValueError("warm-start raw/EMA equality was not established")
    raw_init_sha = init_audit.get("raw_model_parameter_sha256", "")
    ema_init_sha = init_audit.get("ema_model_parameter_sha256", "")
    if raw_init_sha != ema_init_sha or len(raw_init_sha) != 64:
        raise ValueError("warm-start raw/EMA parameter fingerprints mismatch")
    if checkpoint.get("phase_a_init_audit") != source.get("phase_a_init_audit"):
        raise ValueError("Phase-A provenance was not preserved")

    model = normalized_state_dict(checkpoint["model_state_dict"])
    ema_model = normalized_state_dict(checkpoint["ema_model_state_dict"])
    source_ema = normalized_state_dict(source["ema_model_state_dict"])
    if args.expected_step == 0:
        assert_state_dict_equal(model, source_ema, "step-0 raw/source EMA")
        assert_state_dict_equal(ema_model, source_ema, "step-0 EMA/source EMA")
    if not torch.equal(checkpoint["initial_p_weight"], source_ema[P_KEY]):
        raise ValueError("V5-Pg initial P embedding is not source EMA P")
    if not torch.equal(model[P_KEY][256], torch.zeros(128)):
        raise ValueError("V5-Pg raw P PAD row is not zero")
    if not torch.equal(ema_model[P_KEY][256], torch.zeros(128)):
        raise ValueError("V5-Pg EMA P PAD row is not zero")

    optimizer = checkpoint["optimizer_state_dict"]
    scheduler = checkpoint["scheduler_state_dict"]
    parameter_ids = optimizer["param_groups"][0]["params"]
    parameter_names = checkpoint.get("trainable_parameter_names", [])
    if len(parameter_ids) != len(parameter_names):
        raise ValueError("V5-Pg optimizer does not cover every trainable tensor")
    if args.expected_step == 0:
        if optimizer["state"]:
            raise ValueError("V5-Pg step-0 optimizer is not fresh")
    else:
        optimizer_steps = [
            int(state["step"].item())
            for state in optimizer["state"].values()
            if isinstance(state, dict) and "step" in state
        ]
        if not args.engineering_g_probe and len(optimizer_steps) != len(parameter_ids):
            raise ValueError("V5-Pg optimizer state is incomplete")
        if args.engineering_g_probe and not optimizer_steps:
            raise ValueError("V5-Pg engineering probe created no optimizer state")
        if set(optimizer_steps) != {args.expected_step}:
            raise ValueError("V5-Pg optimizer parameter steps mismatch")
        if args.engineering_g_probe:
            state_names = {
                parameter_names[index]
                for index, parameter_id in enumerate(parameter_ids)
                if parameter_id in optimizer["state"]
            }
            if not any(name.startswith("transformer.") for name in state_names):
                raise ValueError("V5-Pg probe lacks DiT optimizer state")
    if int(scheduler.get("last_epoch", -1)) != args.expected_step:
        raise ValueError("V5-Pg scheduler step mismatch")
    optimizer_lr = float(optimizer["param_groups"][0]["lr"])
    expected_lr = v5pg_lr(args.expected_step)
    if optimizer_lr != expected_lr:
        raise ValueError(f"V5-Pg LR mismatch: {optimizer_lr} != {expected_lr}")
    if [optimizer_lr] != [float(value) for value in scheduler.get("_last_lr", [])]:
        raise ValueError("V5-Pg optimizer/scheduler LR mismatch")

    ema_step = int(checkpoint["ema_step"].item())
    if ema_step != 40000 + args.expected_step:
        raise ValueError("V5-Pg EMA counter is not continuous from source 40K")
    if not bool(checkpoint["ema_initted"].item()):
        raise ValueError("V5-Pg EMA is not initialized")

    rank_states = checkpoint.get("rank_states")
    world_size = int(metadata["world_size"])
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise ValueError("V5-Pg rank RNG/cursor states are incomplete")
    if sorted(int(state["rank"]) for state in rank_states) != list(range(world_size)):
        raise ValueError("V5-Pg rank IDs are incomplete or duplicated")
    cursors = {
        (int(state["sampler_epoch"]), int(state["batch_offset"]))
        for state in rank_states
    }
    if len(cursors) != 1:
        raise ValueError("V5-Pg rank data cursors diverged")
    if args.expected_sampler_num_samples is not None:
        consumed = args.expected_step * int(metadata["grad_accum"])
        if consumed == 0:
            expected_cursor = (0, 0)
        else:
            expected_epoch = (consumed - 1) // args.expected_sampler_num_samples
            expected_cursor = (
                expected_epoch,
                consumed - expected_epoch * args.expected_sampler_num_samples,
            )
        if cursors != {expected_cursor}:
            raise ValueError(f"V5-Pg sampler cursor mismatch: {cursors} != {expected_cursor}")
    for state in rank_states:
        rng = state.get("rng")
        if not isinstance(rng, dict) or set(rng) != {
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
        }:
            raise ValueError("V5-Pg rank RNG state is incomplete")

    if not all_finite(checkpoint):
        raise ValueError("V5-Pg checkpoint contains non-finite tensors")

    report = {
        "schema": "v5pg_phase_c_checkpoint_audit_v1",
        "checkpoint": args.checkpoint,
        "step": args.expected_step,
        "run_state": checkpoint["run_state"],
        "source_checkpoint_sha256": args.source_sha256,
        "vae_checkpoint_sha256": args.vae_sha256,
        "optimizer_parameter_tensors": len(parameter_ids),
        "optimizer_lr": optimizer_lr,
        "ema_step": ema_step,
        "rank_state_count": len(rank_states),
        "sampler_cursor": list(next(iter(cursors))),
        "step0_source_equal": args.expected_step == 0,
        "status": "ok",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
