import argparse
import json

import torch


EXPECTED_SCHEMA = "v5p_training_checkpoint_v1"
P_KEY = "midi_p_v4ph.embedding.weight"


def all_finite(tree):
    if torch.is_tensor(tree):
        return not tree.is_floating_point() or bool(torch.isfinite(tree).all())
    if isinstance(tree, dict):
        return all(all_finite(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return all(all_finite(value) for value in tree)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--expected_step", type=int, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema") != EXPECTED_SCHEMA:
        raise ValueError("V5P checkpoint schema mismatch")
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise ValueError("V5P checkpoint step mismatch")
    metadata = checkpoint.get("v5p_training")
    if not isinstance(metadata, dict) or metadata.get("phase") != "p_only":
        raise ValueError("V4PH phase-A metadata is missing or incorrect")
    if metadata.get("midi_teacher") != "GAME medium K4 offline cache":
        raise ValueError("V4PH checkpoint does not declare the GAME cache teacher")
    if metadata.get("midi_fuzz_disturb") is not False:
        raise ValueError("V4PH checkpoint unexpectedly enables MIDI fuzz")

    model = checkpoint["model_state_dict"]
    ema = checkpoint["ema_model_state_dict"]
    if model[P_KEY].shape != (257, 128) or ema[P_KEY].shape != (257, 128):
        raise ValueError("V4PH P embedding shape mismatch")
    initial = checkpoint["initial_p_weight"]
    changed = int((model[P_KEY] != initial).sum())
    if args.expected_step > 0 and changed == 0:
        raise ValueError("V4PH P embedding did not change")
    if not torch.equal(model[P_KEY][256], torch.zeros(128)):
        raise ValueError("V4PH PAD row is not fixed at zero")

    support = checkpoint["p_support_counts"].long()
    if support.shape != (257,) or int(support.sum()) <= 0:
        raise ValueError("V4PH P support counts are invalid")
    history = checkpoint["p_distance_history"]
    expected_history_steps = [step for step in (100, 300, 500) if step <= args.expected_step]
    if [int(item["step"]) for item in history] != expected_history_steps:
        raise ValueError("V4PH P distance history steps are incomplete")

    optimizer = checkpoint["optimizer_state_dict"]
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("V4PH phase A must have one optimizer group")
    parameter_ids = optimizer["param_groups"][0]["params"]
    if len(parameter_ids) != 1:
        raise ValueError("V4PH phase A optimizer must contain only P embedding")
    state = optimizer["state"].get(parameter_ids[0])
    if args.expected_step > 0 and (
        not isinstance(state, dict)
        or "exp_avg" not in state
        or "exp_avg_sq" not in state
        or float(state["exp_avg"].abs().sum()) == 0
    ):
        raise ValueError("V4PH P optimizer state is missing or zero")
    if not all_finite(checkpoint):
        raise ValueError("V4PH checkpoint contains non-finite tensors")

    report = {
        "checkpoint": args.checkpoint,
        "step": args.expected_step,
        "run_state": checkpoint.get("run_state"),
        "p_changed_values": changed,
        "supported_pitch_rows": int((support[:255] > 0).sum()),
        "support_frames": int(support[:255].sum()),
        "distance_history": history,
        "frozen_fingerprint": checkpoint["initial_frozen_fingerprint"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
