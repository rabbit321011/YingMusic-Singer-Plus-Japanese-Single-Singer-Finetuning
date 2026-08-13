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


EXPECTED_SCHEMA = "v4iph_training_checkpoint_v1"
P_KEY = "midi_p_v4ph.embedding.weight"
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(TRAIN_DIR)
CURRENT_CODE_PATHS = {
    "training_code": os.path.join(TRAIN_DIR, "train_v4iph.py"),
    "contract_code": os.path.join(TRAIN_DIR, "v4iph_contract.py"),
    "placement_code": os.path.join(
        PACKAGE_DIR, "h_alignment", "placement_v4iph.py"
    ),
}


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
    args = parser.parse_args()

    if sha256_file(args.phase_a_checkpoint) != args.phase_a_sha256:
        raise ValueError("phase-A checkpoint SHA256 mismatch")
    source = torch.load(
        args.phase_a_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_schema") != EXPECTED_SCHEMA:
        raise ValueError("V4IPH checkpoint schema mismatch")
    if int(checkpoint.get("global_step", -1)) != args.expected_step:
        raise ValueError("V4IPH checkpoint step mismatch")
    expected_run_state = "complete" if args.expected_step == 30000 else "stopped"
    if checkpoint.get("run_state") != expected_run_state:
        raise ValueError("V4IPH run state mismatch")

    metadata = checkpoint.get("v4iph_training")
    if not isinstance(metadata, dict) or metadata.get("phase") != "joint":
        raise ValueError("V4IPH metadata is missing or incorrect")
    if metadata.get("reference_mode") != "all_b":
        raise ValueError("V4IPH reference mode is not all_b")
    if metadata.get("reference_policy") != (
        "ref_len_exactly_zero_cond_all_zero_full_timeline_is_b"
    ):
        raise ValueError("V4IPH zero-reference policy mismatch")
    if metadata.get("flow_policy") != (
        "flow_a_zero_plus_flow_b_weight_times_full_timeline_flow_b"
    ):
        raise ValueError("V4IPH full-timeline flow policy mismatch")
    recorded_code = metadata.get("file_sha256", {})
    current_code = {
        label: sha256_file(path) for label, path in CURRENT_CODE_PATHS.items()
    }
    for label, current_sha256 in current_code.items():
        if recorded_code.get(label) != current_sha256:
            raise ValueError(f"V4IPH {label} SHA256 mismatch")
    if metadata.get("midi_p_init") != "phase_a_p500_with_kernel_fill":
        raise ValueError("V4IPH P initialization metadata mismatch")
    transition_metadata = metadata.get("phase_a_init", {})
    if transition_metadata.get("checkpoint_sha256") != args.phase_a_sha256:
        raise ValueError("V4IPH source checkpoint metadata mismatch")
    if transition_metadata.get("adjudication_sha256") != args.adjudication_sha256:
        raise ValueError("V4IPH adjudication metadata mismatch")
    if transition_metadata.get("optimizer_policy") != "fresh":
        raise ValueError("V4IPH optimizer was not declared fresh")
    if transition_metadata.get("ema_policy") != "fresh_from_transition_weights":
        raise ValueError("V4IPH EMA policy mismatch")

    checkpoint_args = checkpoint.get("args", {})
    if checkpoint_args.get("reference_mode") != "all_b":
        raise ValueError("V4IPH checkpoint args do not lock all_b")
    for rank_state in checkpoint.get("rank_states", []):
        flow_a = rank_state.get("metrics", {}).get("flow_a")
        if flow_a is None or float(flow_a) != 0.0:
            raise ValueError("V4IPH checkpoint contains nonzero FlowA metrics")

    support = source["p_support_counts"].long()
    expected_initial_p = source["model_state_dict"][P_KEY].clone()
    filled = fill_unsupported_pitch_rows(expected_initial_p, support)
    initial_p = checkpoint["initial_p_weight"]
    if not torch.equal(initial_p, expected_initial_p):
        raise ValueError("V4IPH initial P does not match the audited transition")
    transition_audit = checkpoint.get("phase_a_init_audit", {})
    if transition_audit.get("supported_pitch_rows") != int(
        (support[:255] > 0).sum()
    ):
        raise ValueError("V4IPH supported pitch count mismatch")
    if transition_audit.get("unsupported_pitch_rows_filled") != filled:
        raise ValueError("V4IPH filled pitch count mismatch")
    if transition_audit.get("transition_p_sha256") != tensor_sha256(initial_p):
        raise ValueError("V4IPH transition P hash mismatch")

    model = checkpoint["model_state_dict"]
    if model[P_KEY].shape != (257, 128):
        raise ValueError("V4IPH P embedding shape mismatch")
    changed = int((model[P_KEY] != initial_p).sum())
    if args.expected_step > 0 and changed == 0:
        raise ValueError("V4IPH P embedding did not update")
    if not torch.equal(model[P_KEY][256], torch.zeros(128)):
        raise ValueError("V4IPH PAD row is not fixed at zero")

    optimizer = checkpoint["optimizer_state_dict"]
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("V4IPH must have one optimizer group")
    parameter_ids = optimizer["param_groups"][0]["params"]
    parameter_names = checkpoint.get("trainable_parameter_names", [])
    p_index = parameter_names.index(P_KEY)
    if len(parameter_ids) != len(parameter_names):
        raise ValueError("V4IPH optimizer does not contain the full model")
    p_state = optimizer["state"].get(parameter_ids[p_index])
    if not isinstance(p_state, dict) or float(p_state["exp_avg"].abs().sum()) == 0:
        raise ValueError("V4IPH P optimizer state is missing or zero")
    if int(checkpoint["scheduler_state_dict"]["last_epoch"]) != args.expected_step:
        raise ValueError("V4IPH scheduler step mismatch")
    if int(checkpoint["ema_step"].item()) != args.expected_step:
        raise ValueError("V4IPH EMA was not restarted at step zero")
    if not all_finite(checkpoint):
        raise ValueError("V4IPH checkpoint contains non-finite tensors")

    report = {
        "checkpoint": args.checkpoint,
        "step": args.expected_step,
        "run_state": checkpoint["run_state"],
        "reference_mode": metadata["reference_mode"],
        "code_sha256": current_code,
        "phase_a_checkpoint_sha256": args.phase_a_sha256,
        "adjudication_sha256": args.adjudication_sha256,
        "supported_pitch_rows": transition_audit["supported_pitch_rows"],
        "unsupported_pitch_rows_filled": filled,
        "p_changed_values": changed,
        "optimizer_parameter_tensors": len(parameter_ids),
        "ema_step": int(checkpoint["ema_step"].item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
