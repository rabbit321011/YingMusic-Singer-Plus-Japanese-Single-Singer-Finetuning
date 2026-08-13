import argparse
import hashlib
import math
from collections.abc import Mapping

import numpy as np
import torch


SECTIONS = (
    "checkpoint_schema",
    "run_state",
    "global_step",
    "h_training",
    "model_state_dict",
    "ema_model_state_dict",
    "ema_step",
    "ema_initted",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "rank_states",
)
OPTIONAL_SECTIONS = ("pul_embedding_init",)


class StateMismatch(AssertionError):
    pass


def compare_values(left, right, path):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise StateMismatch(f"{path}: tensor/type mismatch")
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise StateMismatch(
                f"{path}: tensor metadata mismatch "
                f"{left.dtype}/{tuple(left.shape)} != "
                f"{right.dtype}/{tuple(right.shape)}"
            )
        if not torch.equal(left, right):
            detail = ""
            if left.is_floating_point() and left.numel():
                max_abs = (left.float() - right.float()).abs().max().item()
                detail = f", max_abs={max_abs:.9g}"
            raise StateMismatch(f"{path}: tensor values differ{detail}")
        return

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise StateMismatch(f"{path}: ndarray/type mismatch")
        if left.dtype != right.dtype or left.shape != right.shape:
            raise StateMismatch(f"{path}: ndarray metadata differs")
        if not np.array_equal(left, right):
            raise StateMismatch(f"{path}: ndarray values differ")
        return

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise StateMismatch(f"{path}: mapping/type mismatch")
        if set(left) != set(right):
            missing = sorted(set(left) - set(right), key=repr)
            extra = sorted(set(right) - set(left), key=repr)
            raise StateMismatch(f"{path}: keys differ, left_only={missing}, right_only={extra}")
        for key in sorted(left, key=repr):
            compare_values(left[key], right[key], f"{path}.{key}")
        return

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise StateMismatch(f"{path}: sequence metadata differs")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_values(left_item, right_item, f"{path}[{index}]")
        return

    if isinstance(left, float) and isinstance(right, float):
        if left == right or (math.isnan(left) and math.isnan(right)):
            return
    elif left == right:
        return
    raise StateMismatch(f"{path}: {left!r} != {right!r}")


def update_digest(digest, value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(b"ndarray\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=repr):
            update_digest(digest, key)
            update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        for item in value:
            update_digest(digest, item)
    else:
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")


def state_sha256(value):
    digest = hashlib.sha256()
    update_digest(digest, value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Require two H training checkpoints to be bit-exact."
    )
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=SECTIONS + OPTIONAL_SECTIONS,
        default=None,
        help="Checkpoint sections to compare (default: all).",
    )
    args = parser.parse_args()

    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    sections = list(SECTIONS if args.sections is None else args.sections)
    if args.sections is None and any(
        section in left or section in right for section in OPTIONAL_SECTIONS
    ):
        sections.extend(OPTIONAL_SECTIONS)
    for section in sections:
        if section not in left or section not in right:
            raise StateMismatch(f"missing required section: {section}")
        compare_values(left[section], right[section], section)
        print(f"[exact] {section}: {state_sha256(left[section])}")
    print("H checkpoint exact-resume gate passed")


if __name__ == "__main__":
    main()
