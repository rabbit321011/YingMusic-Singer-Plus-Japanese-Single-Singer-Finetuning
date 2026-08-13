import argparse
import json

import numpy as np
import torch


COMPARE_KEYS = (
    "model_state_dict",
    "ema_model_state_dict",
    "ema_step",
    "ema_initted",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "global_step",
    "v5p_training",
    "rank_states",
    "pul_embedding_init",
    "midi_p_schema",
    "game_cache_schema",
    "p_support_counts",
    "p_distance_history",
    "initial_p_weight",
    "initial_frozen_fingerprint",
    "phase_a_init_audit",
    "warmstart_init_audit",
    "trainable_parameter_names",
)


def equal(left, right, path, failures):
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            failures.append(f"{path}: tensor type mismatch")
        elif left.dtype != right.dtype or left.shape != right.shape:
            failures.append(f"{path}: tensor metadata mismatch")
        elif not torch.equal(left, right):
            failures.append(f"{path}: tensor values differ")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            failures.append(f"{path}: ndarray type mismatch")
        elif not np.array_equal(left, right):
            failures.append(f"{path}: ndarray differs")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            failures.append(f"{path}: dict type mismatch")
            return
        if set(left) != set(right):
            failures.append(f"{path}: dict keys differ")
            return
        for key in left:
            equal(left[key], right[key], f"{path}.{key}", failures)
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            failures.append(f"{path}: sequence metadata differs")
            return
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, f"{path}[{index}]", failures)
        return
    if left != right:
        failures.append(f"{path}: {left!r} != {right!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("continuous")
    parser.add_argument("resumed")
    args = parser.parse_args()
    continuous = torch.load(
        args.continuous, map_location="cpu", weights_only=False, mmap=True
    )
    resumed = torch.load(
        args.resumed, map_location="cpu", weights_only=False, mmap=True
    )
    failures = []
    for key in COMPARE_KEYS:
        if key not in continuous or key not in resumed:
            failures.append(f"missing key: {key}")
            continue
        equal(continuous[key], resumed[key], key, failures)
    report = {
        "schema": "v5pg_exact_resume_comparison_v1",
        "continuous": args.continuous,
        "resumed": args.resumed,
        "compared_keys": list(COMPARE_KEYS),
        "failure_count": len(failures),
        "failures": failures[:100],
        "bit_exact": not failures,
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
