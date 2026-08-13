#!/usr/bin/env python3
"""Create a weights-only V4vf bootstrap checkpoint for VAE adaptation."""

import argparse
import os
import tempfile

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        parser.error(f"source checkpoint does not exist: {args.source}")
    if os.path.exists(args.output):
        parser.error(f"refusing to overwrite bootstrap checkpoint: {args.output}")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("source checkpoint has no model_state_dict")
    if "ema_model_state_dict" not in checkpoint:
        raise KeyError("source checkpoint has no ema_model_state_dict")

    checkpoint["v4vfg_source_global_step"] = checkpoint.get("global_step")
    checkpoint["v4vfg_source_checkpoint"] = os.path.abspath(args.source)
    checkpoint["global_step"] = 0
    checkpoint["resume_global_step"] = 0
    checkpoint.pop("optimizer_state_dict", None)
    checkpoint.pop("scheduler_state_dict", None)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix="v4vfg_bootstrap_", suffix=".pt", dir=output_dir
    )
    os.close(fd)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, args.output)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    print(
        f"Created {args.output} from source step "
        f"{checkpoint['v4vfg_source_global_step']} with optimizer state removed"
    )


if __name__ == "__main__":
    main()
