"""Frozen model, optimizer, and schedule contract for V4IjPH v2."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn


V4IJPH_CHECKPOINT_SCHEMA = "v4ijph_training_checkpoint_v2"
V4IJPH_INS_CACHE_SCHEMA = "v4ijph_ins_cache_v2"
V4IJPH_SOURCE_CHECKPOINT_SCHEMA = "v4iph_training_checkpoint_v1"

V4IJPH_TRANSITION_STEP = 8_000
V4IJPH_ADAPTER_PEAK_STEP = 14_000
V4IJPH_ADAPTER_MID_STEP = 18_000
V4IJPH_MAX_STEPS = 30_000

V4IJPH_INS_DIM = 768
V4IJPH_STYLE_DIM = 64
V4IJPH_INS_DROPOUT = 0.3
V4IJPH_ADAPTER_PEAK_LR = 1e-4
V4IJPH_ADAPTER_MID_LR = 1e-5
V4IJPH_ADAPTER_BETAS = (0.9, 0.95)
V4IJPH_ADAPTER_WEIGHT_DECAY = 1e-2
V4IJPH_ADAPTER_EMA_BETA = 0.995
V4IJPH_ADAPTER_EMA_UPDATE_AFTER_STEP = 100
V4IJPH_ADAPTER_EMA_UPDATE_EVERY = 1


class InsStyleAdapter(nn.Module):
    """Project one normalized 768-D INS row to the frozen 64-D cond slot."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(V4IJPH_INS_DIM, 512)
        self.hidden_projection = nn.Linear(512, 256)
        self.output_projection = nn.Linear(256, V4IJPH_STYLE_DIM)
        self.activation = nn.SiLU()
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, ins: torch.Tensor) -> torch.Tensor:
        validate_ins(ins)
        hidden = self.activation(self.input_projection(ins.float()))
        hidden = self.activation(self.hidden_projection(hidden))
        return self.output_projection(hidden)


def validate_ins(ins: torch.Tensor) -> None:
    if ins.ndim != 2 or ins.shape[1] != V4IJPH_INS_DIM:
        raise ValueError(
            f"INS must have shape [B, {V4IJPH_INS_DIM}], got {tuple(ins.shape)}"
        )
    if not torch.isfinite(ins).all():
        raise ValueError("INS contains non-finite values")


def adapter_lr_for_global_update(global_update: int) -> float:
    """Return the LR used by one completed, one-based global optimizer update."""
    step = int(global_update)
    if not 1 <= step <= V4IJPH_MAX_STEPS:
        raise ValueError(f"global update must be in [1, {V4IJPH_MAX_STEPS}]")
    if step <= V4IJPH_TRANSITION_STEP:
        return 0.0
    if step <= V4IJPH_ADAPTER_PEAK_STEP:
        progress = (step - V4IJPH_TRANSITION_STEP) / (
            V4IJPH_ADAPTER_PEAK_STEP - V4IJPH_TRANSITION_STEP
        )
        return V4IJPH_ADAPTER_PEAK_LR * progress
    if step <= V4IJPH_ADAPTER_MID_STEP:
        progress = (step - V4IJPH_ADAPTER_PEAK_STEP) / (
            V4IJPH_ADAPTER_MID_STEP - V4IJPH_ADAPTER_PEAK_STEP
        )
        return V4IJPH_ADAPTER_MID_LR + 0.5 * (
            V4IJPH_ADAPTER_PEAK_LR - V4IJPH_ADAPTER_MID_LR
        ) * (1.0 + math.cos(math.pi * progress))
    progress = (step - V4IJPH_ADAPTER_MID_STEP) / (
        V4IJPH_MAX_STEPS - V4IJPH_ADAPTER_MID_STEP
    )
    return 0.5 * V4IJPH_ADAPTER_MID_LR * (
        1.0 + math.cos(math.pi * progress)
    )


def adapter_updates_for_global_step(global_step: int) -> int:
    step = int(global_step)
    if not V4IJPH_TRANSITION_STEP <= step <= V4IJPH_MAX_STEPS:
        raise ValueError(
            f"V4IjPH global step must be in "
            f"[{V4IJPH_TRANSITION_STEP}, {V4IJPH_MAX_STEPS}]"
        )
    return step - V4IJPH_TRANSITION_STEP


def build_style_cond(
    adapter: nn.Module,
    ins: torch.Tensor,
    frames: int,
    *,
    drop_ins: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one style row and its full-timeline `[B,T,64]` broadcast."""
    validate_ins(ins)
    frames = int(frames)
    if frames <= 0:
        raise ValueError("frames must be positive")
    if drop_ins:
        style = torch.zeros(
            ins.shape[0], V4IJPH_STYLE_DIM, device=ins.device, dtype=torch.float32
        )
    else:
        style = adapter(ins)
    cond = style[:, None, :].expand(-1, frames, -1)
    return cond, style


def assert_zero_output_initialization(adapter: InsStyleAdapter) -> None:
    if bool(torch.count_nonzero(adapter.output_projection.weight)):
        raise AssertionError("INS adapter output weight is not exactly zero")
    if bool(torch.count_nonzero(adapter.output_projection.bias)):
        raise AssertionError("INS adapter output bias is not exactly zero")


@torch.no_grad()
def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def strict_load_module(module: nn.Module, state_dict: Mapping, label: str) -> None:
    incompatible = module.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"{label} state mismatch: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )


def assert_tree_equal(actual, expected, label: str = "value") -> None:
    """Bit-compare nested checkpoint state without assuming a common device."""
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        if not (torch.is_tensor(actual) and torch.is_tensor(expected)):
            raise AssertionError(f"{label} tensor type differs")
        left = actual.detach().cpu()
        right = expected.detach().cpu()
        if left.dtype != right.dtype or left.shape != right.shape:
            raise AssertionError(f"{label} tensor metadata differs")
        if not torch.equal(left, right):
            raise AssertionError(f"{label} tensor differs")
        return
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if not (isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)):
            raise AssertionError(f"{label} NumPy array type differs")
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise AssertionError(f"{label} NumPy array metadata differs")
        if not np.array_equal(actual, expected):
            raise AssertionError(f"{label} NumPy array differs")
        return
    if isinstance(actual, np.generic) or isinstance(expected, np.generic):
        if not (isinstance(actual, np.generic) and isinstance(expected, np.generic)):
            raise AssertionError(f"{label} NumPy scalar type differs")
        if actual.dtype != expected.dtype or actual != expected:
            raise AssertionError(f"{label} NumPy scalar differs")
        return
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not (isinstance(actual, Mapping) and isinstance(expected, Mapping)):
            raise AssertionError(f"{label} mapping type differs")
        if actual.keys() != expected.keys():
            raise AssertionError(f"{label} mapping keys differ")
        for key in actual:
            assert_tree_equal(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(actual, Sequence) or isinstance(expected, Sequence):
        if isinstance(actual, (str, bytes)) or isinstance(expected, (str, bytes)):
            if actual != expected:
                raise AssertionError(f"{label} differs")
            return
        if not (
            isinstance(actual, Sequence)
            and isinstance(expected, Sequence)
            and len(actual) == len(expected)
        ):
            raise AssertionError(f"{label} sequence shape differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_tree_equal(left, right, f"{label}[{index}]")
        return
    if actual != expected:
        raise AssertionError(f"{label} differs: {actual!r} != {expected!r}")
