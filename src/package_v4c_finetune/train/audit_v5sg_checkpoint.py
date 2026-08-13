import argparse
import hashlib
import numpy as np
import torch


SCHEMA = "v5sg_training_checkpoint_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(state):
    return {key.removeprefix("module.").removeprefix("ema_model."): value for key, value in state.items()}


def assert_equal(left, right, label):
    left, right = normalized(left), normalized(right)
    if set(left) != set(right):
        raise ValueError(f"{label} keys differ")
    for key in left:
        if left[key].dtype != right[key].dtype or left[key].shape != right[key].shape or not torch.equal(left[key], right[key]):
            raise ValueError(f"{label} differs at {key}")


def lr(step, peak=1.4e-5, mid=1e-5, warmup=4000, shoulder_end=28000, max_steps=40000):
    if step < warmup:
        return peak * step / warmup
    ratio = mid / peak
    if step < shoulder_end:
        p = (step - warmup) / (shoulder_end - warmup)
        return peak * (ratio + (1 - ratio) * 0.5 * (1 + np.cos(np.pi * p)))
    p = min(1.0, (step - shoulder_end) / (max_steps - shoulder_end))
    return peak * ratio * 0.5 * (1 + np.cos(np.pi * p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--source_sha256", required=True)
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--vae_sha256", required=True)
    parser.add_argument("--some_checkpoint", required=True)
    parser.add_argument("--some_sha256", required=True)
    parser.add_argument("--expected_step", type=int, required=True)
    parser.add_argument(
        "--expected_run_state",
        choices=("running", "stopped", "complete"),
        default=None,
    )
    parser.add_argument("--expected_save_every", type=int, default=2000)
    parser.add_argument("--expected_eval_every", type=int, default=1000)
    args = parser.parse_args()

    if sha256_file(args.source_checkpoint) != args.source_sha256:
        raise ValueError("official source SHA256 mismatch")
    if sha256_file(args.vae_checkpoint) != args.vae_sha256:
        raise ValueError("285k VAE SHA256 mismatch")
    if sha256_file(args.some_checkpoint) != args.some_sha256:
        raise ValueError("SOME checkpoint SHA256 mismatch")
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("checkpoint_schema") != SCHEMA:
        raise ValueError("V5-Sg checkpoint schema mismatch")
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise ValueError("checkpoint step mismatch")
    expected_run_state = args.expected_run_state
    if expected_run_state is None:
        expected_run_state = "complete" if args.expected_step == 40000 else "running"
    if checkpoint.get("run_state") != expected_run_state:
        raise ValueError("checkpoint run_state mismatch")

    metadata = checkpoint.get("h_training")
    if not isinstance(metadata, dict) or metadata.get("route") != "V5-Sg":
        raise ValueError("V5-Sg metadata missing")
    expected = {
        "schema": SCHEMA,
        "phase": "s_g",
        "placement_mode": "phone_pul",
        "pool_policy": "KEEP_LONG_DEDUP_SHORT",
        "sampling_policy": "NATURAL_RECORD",
        "lr": 1.4e-5,
        "warmup_steps": 4000,
        "first_decay_end": 28000,
        "mid_lr": 1e-5,
        "max_steps": 40000,
        "save_every": args.expected_save_every,
        "eval_every": args.expected_eval_every,
        "max_duration": 60.1,
        "cka_weight": 0.7,
        "drop_text": 0.15,
        "flow_b_weight": 2.0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"contract mismatch: {key}")
    if metadata.get("midi_teacher") != "continuous SOME frozen V4Hg teacher":
        raise ValueError("SOME teacher metadata mismatch")
    if metadata.get("fresh_start", {}).get("optimizer_policy") != "fresh":
        raise ValueError("optimizer is not declared fresh")
    files = metadata.get("file_sha256", {})
    if files.get("base_checkpoint") != args.source_sha256:
        raise ValueError("official source provenance mismatch")
    if files.get("vae_checkpoint") != args.vae_sha256:
        raise ValueError("VAE provenance mismatch")
    if files.get("midi_checkpoint") != args.some_sha256:
        raise ValueError("SOME provenance mismatch")

    model = checkpoint["model_state_dict"]
    ema = checkpoint["ema_model_state_dict"]
    if args.expected_step == 0:
        assert_equal(model, ema, "step-0 raw/EMA")
        if checkpoint["optimizer_state_dict"]["state"]:
            raise ValueError("step-0 optimizer is not fresh")
    optimizer = checkpoint["optimizer_state_dict"]
    scheduler = checkpoint["scheduler_state_dict"]
    if int(scheduler.get("last_epoch", -1)) != args.expected_step:
        raise ValueError("scheduler step mismatch")
    actual_lr = float(optimizer["param_groups"][0]["lr"])
    expected_lr = lr(args.expected_step)
    if not np.isclose(actual_lr, expected_lr, rtol=1e-12, atol=0.0):
        raise ValueError(f"LR mismatch: {actual_lr} != {expected_lr}")
    if [actual_lr] != [float(v) for v in scheduler.get("_last_lr", [])]:
        raise ValueError("optimizer/scheduler LR mismatch")
    if int(checkpoint["ema_step"].item()) != args.expected_step:
        raise ValueError("EMA counter mismatch")
    if not bool(checkpoint["ema_initted"].item()):
        raise ValueError("EMA is not initialized")
    rank_states = checkpoint.get("rank_states", [])
    if len(rank_states) != int(metadata["world_size"]):
        raise ValueError("rank states incomplete")
    if len({(int(s["sampler_epoch"]), int(s["batch_offset"])) for s in rank_states}) != 1:
        raise ValueError("rank cursors diverged")
    print(f"V5-Sg audit OK: step={args.expected_step} LR={actual_lr:.12g} schema={SCHEMA}")


if __name__ == "__main__":
    main()
