import argparse
import json
import os

import torch


def audit_state_dict(state_dict):
    tensor_count = 0
    floating_count = 0
    nonfinite = []
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor_count += 1
        if value.is_floating_point() or value.is_complex():
            floating_count += 1
            if not torch.isfinite(value).all().item():
                nonfinite.append(name)
    return tensor_count, floating_count, nonfinite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model_count, model_float_count, model_nonfinite = audit_state_dict(
        checkpoint["model_state_dict"]
    )
    ema_count, ema_float_count, ema_nonfinite = audit_state_dict(
        checkpoint["ema_model_state_dict"]
    )
    result = {
        "path": os.path.abspath(args.checkpoint),
        "bytes": os.path.getsize(args.checkpoint),
        "global_step": checkpoint.get("global_step"),
        "model_tensors": model_count,
        "model_floating_tensors": model_float_count,
        "ema_tensors": ema_count,
        "ema_floating_tensors": ema_float_count,
        "model_nonfinite": model_nonfinite,
        "ema_nonfinite": ema_nonfinite,
    }
    print(json.dumps(result, indent=2))
    if checkpoint.get("global_step") != args.expected_step:
        raise SystemExit("unexpected global_step")
    if model_nonfinite or ema_nonfinite:
        raise SystemExit("checkpoint contains non-finite tensors")


if __name__ == "__main__":
    main()
