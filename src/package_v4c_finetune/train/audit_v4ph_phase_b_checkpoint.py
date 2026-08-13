import argparse
import hashlib
import json
import os
import sys

import torch


PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YING_REPO = os.path.join(PROJECT, "YingMusic-Singer-Plus-src")
if not os.path.isdir(os.path.join(YING_REPO, "src")):
    YING_REPO = PROJECT
sys.path.insert(0, YING_REPO)

from src.YingMusicSinger.melody.midi_p_v4ph import (  # noqa: E402
    fill_unsupported_pitch_rows,
)


EXPECTED_SCHEMA = "v4ph_training_checkpoint_v1"
P_KEY = "midi_p_v4ph.embedding.weight"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value):
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
    parser.add_argument("--phase_a_checkpoint", required=True)
    parser.add_argument("--phase_a_sha256", required=True)
    parser.add_argument("--adjudication_sha256", required=True)
    parser.add_argument("--expected_schedule_profile")
    parser.add_argument("--expected_hold_steps", type=int)
    args = parser.parse_args()

    if (args.expected_schedule_profile is None) != (
        args.expected_hold_steps is None
    ):
        raise ValueError(
            "expected_schedule_profile and expected_hold_steps must be used together"
        )

    if sha256_file(args.phase_a_checkpoint) != args.phase_a_sha256:
        raise ValueError("phase-A checkpoint SHA256 mismatch")
    source = torch.load(
        args.phase_a_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema") != EXPECTED_SCHEMA:
        raise ValueError("V4PH checkpoint schema mismatch")
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise ValueError("V4PH checkpoint step mismatch")
    expected_run_state = "complete" if args.expected_step == 30000 else "stopped"
    if checkpoint.get("run_state") != expected_run_state:
        raise ValueError("V4PH phase-B run state mismatch")

    metadata = checkpoint.get("v4ph_training")
    if not isinstance(metadata, dict) or metadata.get("phase") != "joint":
        raise ValueError("V4PH phase-B metadata is missing or incorrect")
    if args.expected_schedule_profile is not None:
        if metadata.get("schedule_profile") != args.expected_schedule_profile:
            raise ValueError("V4PH phase-B schedule profile mismatch")
        if int(metadata.get("hold_steps", -1)) != args.expected_hold_steps:
            raise ValueError("V4PH phase-B hold step mismatch")
    if metadata.get("midi_p_init") != "phase_a_p500_with_kernel_fill":
        raise ValueError("V4PH phase-B P initialization metadata mismatch")
    transition_metadata = metadata.get("phase_a_init", {})
    if transition_metadata.get("checkpoint_sha256") != args.phase_a_sha256:
        raise ValueError("V4PH phase-B source checkpoint metadata mismatch")
    if transition_metadata.get("adjudication_sha256") != args.adjudication_sha256:
        raise ValueError("V4PH phase-B adjudication metadata mismatch")
    if transition_metadata.get("optimizer_policy") != "fresh":
        raise ValueError("V4PH phase-B optimizer was not declared fresh")
    if transition_metadata.get("ema_policy") != "fresh_from_transition_weights":
        raise ValueError("V4PH phase-B EMA policy mismatch")

    support = source["p_support_counts"].long()
    expected_initial_p = source["model_state_dict"][P_KEY].clone()
    filled = fill_unsupported_pitch_rows(expected_initial_p, support)
    initial_p = checkpoint["initial_p_weight"]
    if not torch.equal(initial_p, expected_initial_p):
        raise ValueError("V4PH phase-B initial P does not match the audited transition")
    transition_audit = checkpoint.get("phase_a_init_audit", {})
    if transition_audit.get("supported_pitch_rows") != int(
        (support[:255] > 0).sum()
    ):
        raise ValueError("V4PH phase-B supported pitch count mismatch")
    if transition_audit.get("unsupported_pitch_rows_filled") != filled:
        raise ValueError("V4PH phase-B filled pitch count mismatch")
    if transition_audit.get("transition_p_sha256") != tensor_sha256(initial_p):
        raise ValueError("V4PH phase-B transition P hash mismatch")

    model = checkpoint["model_state_dict"]
    if model[P_KEY].shape != (257, 128):
        raise ValueError("V4PH phase-B P embedding shape mismatch")
    changed = int((model[P_KEY] != initial_p).sum())
    if args.expected_step > 0 and changed == 0:
        raise ValueError("V4PH phase-B P embedding did not update")
    if not torch.equal(model[P_KEY][256], torch.zeros(128)):
        raise ValueError("V4PH phase-B PAD row is not fixed at zero")

    optimizer = checkpoint["optimizer_state_dict"]
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("V4PH phase B must have one optimizer group")
    parameter_ids = optimizer["param_groups"][0]["params"]
    parameter_names = checkpoint.get("trainable_parameter_names", [])
    p_index = parameter_names.index(P_KEY)
    if len(parameter_ids) != len(parameter_names):
        raise ValueError("V4PH phase-B optimizer does not contain the full model")
    p_state = optimizer["state"].get(parameter_ids[p_index])
    if not isinstance(p_state, dict) or float(p_state["exp_avg"].abs().sum()) == 0:
        raise ValueError("V4PH phase-B P optimizer state is missing or zero")
    if int(checkpoint["scheduler_state_dict"]["last_epoch"]) != args.expected_step:
        raise ValueError("V4PH phase-B scheduler step mismatch")
    if int(checkpoint["ema_step"].item()) != args.expected_step:
        raise ValueError("V4PH phase-B EMA was not restarted at step zero")
    if not all_finite(checkpoint):
        raise ValueError("V4PH phase-B checkpoint contains non-finite tensors")

    report = {
        "checkpoint": args.checkpoint,
        "step": args.expected_step,
        "run_state": checkpoint["run_state"],
        "phase_a_checkpoint_sha256": args.phase_a_sha256,
        "adjudication_sha256": args.adjudication_sha256,
        "schedule_profile": metadata.get("schedule_profile", "legacy_baseline"),
        "hold_steps": int(metadata.get("hold_steps", -1)),
        "supported_pitch_rows": transition_audit["supported_pitch_rows"],
        "unsupported_pitch_rows_filled": filled,
        "p_changed_values": changed,
        "optimizer_parameter_tensors": len(parameter_ids),
        "ema_step": int(checkpoint["ema_step"].item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
