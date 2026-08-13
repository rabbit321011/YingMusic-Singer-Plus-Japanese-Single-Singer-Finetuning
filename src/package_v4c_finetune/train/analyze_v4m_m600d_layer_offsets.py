"""Compare 12k layer displacement for M600-D and its 338M HighLR control."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import re
from collections import defaultdict

import torch


BLOCK_PATTERN = re.compile(
    r"^(?:module\.)?transformer\.transformer_blocks\.(\d+)\.(.+)$"
)


def load_checkpoint(path: pathlib.Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def select_state(checkpoint, label: str):
    candidates = {
        "online": ("model_state_dict", "model"),
        "ema": ("ema_model_state_dict", "ema_state_dict", "ema"),
    }[label]
    for key in candidates:
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state, key
    raise KeyError(f"checkpoint has no {label} state; keys={sorted(checkpoint)}")


def block_states(state):
    blocks = defaultdict(dict)
    for name, value in state.items():
        match = BLOCK_PATTERN.match(name)
        if match and torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            blocks[int(match.group(1))][match.group(2)] = value.detach()
    return dict(blocks)


def empty_accumulator():
    return {
        "elements": 0,
        "tensor_count": 0,
        "changed_elements": 0,
        "delta_sq": 0.0,
        "initial_sq": 0.0,
        "delta_abs_sum": 0.0,
        "delta_abs_max": 0.0,
    }


def add_tensor(accumulator, initial, trained):
    if initial.shape != trained.shape:
        raise ValueError(f"tensor shape mismatch: {initial.shape} != {trained.shape}")
    if initial.dtype != trained.dtype:
        raise ValueError(f"tensor dtype mismatch: {initial.dtype} != {trained.dtype}")
    delta = trained - initial
    accumulator["elements"] += delta.numel()
    accumulator["tensor_count"] += 1
    accumulator["changed_elements"] += int(torch.count_nonzero(delta).item())
    accumulator["delta_sq"] += float(torch.sum(delta * delta, dtype=torch.float64))
    accumulator["initial_sq"] += float(
        torch.sum(initial * initial, dtype=torch.float64)
    )
    absolute = delta.abs()
    accumulator["delta_abs_sum"] += float(
        torch.sum(absolute, dtype=torch.float64)
    )
    accumulator["delta_abs_max"] = max(
        accumulator["delta_abs_max"], float(torch.max(absolute))
    )


def merge_accumulator(target, source):
    for key in (
        "elements",
        "tensor_count",
        "changed_elements",
        "delta_sq",
        "initial_sq",
        "delta_abs_sum",
    ):
        target[key] += source[key]
    target["delta_abs_max"] = max(
        target["delta_abs_max"], source["delta_abs_max"]
    )


def finalize(accumulator):
    elements = accumulator["elements"]
    delta_l2 = math.sqrt(accumulator["delta_sq"])
    initial_l2 = math.sqrt(accumulator["initial_sq"])
    return {
        **accumulator,
        "delta_l2": delta_l2,
        "initial_l2": initial_l2,
        "delta_rms": math.sqrt(accumulator["delta_sq"] / elements),
        "delta_mean_abs": accumulator["delta_abs_sum"] / elements,
        "relative_l2": delta_l2 / initial_l2 if initial_l2 else None,
        "changed_fraction": accumulator["changed_elements"] / elements,
    }


def tensor_category(suffix: str):
    if suffix.startswith("attn_norm.linear."):
        return "adaln_zero_gate"
    return suffix.split(".", 1)[0]


def compare_blocks(initial_blocks, trained_blocks, block_roles):
    per_block = []
    grouped = defaultdict(empty_accumulator)
    grouped_by_category = defaultdict(lambda: defaultdict(empty_accumulator))
    for trained_index, role in sorted(block_roles.items()):
        initial_index = role["initial_index"]
        initial = initial_blocks[initial_index]
        trained = trained_blocks[trained_index]
        if set(initial) != set(trained):
            missing = sorted(set(initial) - set(trained))
            extra = sorted(set(trained) - set(initial))
            raise ValueError(
                f"block schema mismatch at {trained_index}: missing={missing} extra={extra}"
            )
        block_accumulator = empty_accumulator()
        categories = defaultdict(empty_accumulator)
        for suffix in sorted(initial):
            tensor_accumulator = empty_accumulator()
            add_tensor(tensor_accumulator, initial[suffix], trained[suffix])
            merge_accumulator(block_accumulator, tensor_accumulator)
            merge_accumulator(categories[tensor_category(suffix)], tensor_accumulator)
        group = role["group"]
        merge_accumulator(grouped[group], block_accumulator)
        for category, accumulator in categories.items():
            merge_accumulator(grouped_by_category[group][category], accumulator)
        per_block.append(
            {
                "trained_index": trained_index,
                **role,
                **finalize(block_accumulator),
                "categories": {
                    category: finalize(accumulator)
                    for category, accumulator in sorted(categories.items())
                },
            }
        )
    return {
        "groups": {
            group: finalize(accumulator)
            for group, accumulator in sorted(grouped.items())
        },
        "group_categories": {
            group: {
                category: finalize(accumulator)
                for category, accumulator in sorted(categories.items())
            }
            for group, categories in sorted(grouped_by_category.items())
        },
        "per_block": per_block,
    }


def summarize_block_distribution(blocks, group):
    selected = [block for block in blocks if block["group"] == group]
    values = sorted(block["delta_rms"] for block in selected)
    relative = sorted(
        block["relative_l2"]
        for block in selected
        if block["relative_l2"] is not None
    )

    def percentile(items, fraction):
        if not items:
            return None
        index = round((len(items) - 1) * fraction)
        return items[index]

    return {
        "count": len(selected),
        "delta_rms_min": min(values),
        "delta_rms_median": percentile(values, 0.5),
        "delta_rms_max": max(values),
        "relative_l2_min": min(relative) if relative else None,
        "relative_l2_median": percentile(relative, 0.5),
        "relative_l2_max": max(relative) if relative else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", type=pathlib.Path, required=True)
    parser.add_argument("--m600", type=pathlib.Path, required=True)
    parser.add_argument("--m600_previous", type=pathlib.Path)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--highlr", type=pathlib.Path, required=True)
    parser.add_argument("--highlr_previous", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    transition_checkpoint = load_checkpoint(args.transition)
    transition_state, transition_state_key = select_state(
        transition_checkpoint, "online"
    )
    transition_blocks = block_states(transition_state)
    transition_metadata = transition_checkpoint.get("v4m_transition") or {}
    block_transition = transition_metadata.get("block_transition") or {}
    mapping = {
        int(source): int(target)
        for source, target in (block_transition.get("source_to_target") or {}).items()
    }
    new_targets = [int(index) for index in block_transition.get("new_target_blocks", [])]
    if sorted(mapping) != list(range(22)) or len(new_targets) != 18:
        raise ValueError("transition block mapping is incomplete")
    if sorted(transition_blocks) != list(range(40)):
        raise ValueError("transition does not contain 40 blocks")
    del transition_state, transition_checkpoint
    gc.collect()

    m600_roles = {
        target: {
            "group": "old",
            "initial_index": target,
            "source_index": source,
        }
        for source, target in mapping.items()
    }
    m600_roles.update(
        {
            target: {
                "group": "new",
                "initial_index": target,
                "source_index": None,
            }
            for target in new_targets
        }
    )
    m600_checkpoint = load_checkpoint(args.m600)
    m600_results = {}
    m600_state_keys = {}
    for label in ("online", "ema"):
        state, state_key = select_state(m600_checkpoint, label)
        trained_blocks = block_states(state)
        if sorted(trained_blocks) != list(range(40)):
            raise ValueError(f"M600-D {label} does not contain 40 blocks")
        comparison = compare_blocks(transition_blocks, trained_blocks, m600_roles)
        comparison["block_distribution"] = {
            group: summarize_block_distribution(comparison["per_block"], group)
            for group in ("old", "new")
        }
        m600_results[label] = comparison
        m600_state_keys[label] = state_key
        del state, trained_blocks
        gc.collect()
    m600_interval = None
    if args.m600_previous:
        previous_checkpoint = load_checkpoint(args.m600_previous)
        m600_interval = {}
        for label in ("online", "ema"):
            previous_state, _ = select_state(previous_checkpoint, label)
            current_state, _ = select_state(m600_checkpoint, label)
            previous_blocks = block_states(previous_state)
            current_blocks = block_states(current_state)
            comparison = compare_blocks(previous_blocks, current_blocks, m600_roles)
            comparison["block_distribution"] = {
                group: summarize_block_distribution(comparison["per_block"], group)
                for group in ("old", "new")
            }
            m600_interval[label] = comparison
            del previous_state, current_state, previous_blocks, current_blocks
            gc.collect()
        del previous_checkpoint
        gc.collect()
    m600_schema = m600_checkpoint.get("checkpoint_schema")
    m600_step = m600_checkpoint.get("global_step")
    del m600_checkpoint, transition_blocks
    gc.collect()

    source_checkpoint = load_checkpoint(args.source)
    source_state, source_state_key = select_state(source_checkpoint, "online")
    source_blocks = block_states(source_state)
    if sorted(source_blocks) != list(range(22)):
        raise ValueError("source does not contain 22 blocks")
    source_schema = source_checkpoint.get("checkpoint_schema")
    source_step = source_checkpoint.get("global_step")
    del source_state, source_checkpoint
    gc.collect()

    highlr_roles = {
        index: {
            "group": "old",
            "initial_index": index,
            "source_index": index,
        }
        for index in range(22)
    }
    highlr_checkpoint = load_checkpoint(args.highlr)
    highlr_results = {}
    highlr_state_keys = {}
    for label in ("online", "ema"):
        state, state_key = select_state(highlr_checkpoint, label)
        trained_blocks = block_states(state)
        if sorted(trained_blocks) != list(range(22)):
            raise ValueError(f"HighLR {label} does not contain 22 blocks")
        comparison = compare_blocks(source_blocks, trained_blocks, highlr_roles)
        comparison["block_distribution"] = {
            "old": summarize_block_distribution(comparison["per_block"], "old")
        }
        highlr_results[label] = comparison
        highlr_state_keys[label] = state_key
        del state, trained_blocks
        gc.collect()
    highlr_interval = None
    if args.highlr_previous:
        previous_checkpoint = load_checkpoint(args.highlr_previous)
        highlr_interval = {}
        for label in ("online", "ema"):
            previous_state, _ = select_state(previous_checkpoint, label)
            current_state, _ = select_state(highlr_checkpoint, label)
            previous_blocks = block_states(previous_state)
            current_blocks = block_states(current_state)
            comparison = compare_blocks(previous_blocks, current_blocks, highlr_roles)
            comparison["block_distribution"] = {
                "old": summarize_block_distribution(comparison["per_block"], "old")
            }
            highlr_interval[label] = comparison
            del previous_state, current_state, previous_blocks, current_blocks
            gc.collect()
        del previous_checkpoint
        gc.collect()
    highlr_schema = highlr_checkpoint.get("checkpoint_schema")
    highlr_step = highlr_checkpoint.get("global_step")

    report = {
        "schema": "v4m_m600d_layer_offset_comparison_v2",
        "definition": {
            "delta": "trained tensor minus its exact transition/source tensor",
            "delta_rms": "sqrt(sum(delta^2) / element_count)",
            "relative_l2": "L2(delta) / L2(initial)",
            "adaln_zero_gate": "attn_norm.linear.{weight,bias}; zeroed for new identity blocks",
        },
        "inputs": {
            "transition": str(args.transition.resolve()),
            "transition_state_key": transition_state_key,
            "m600": str(args.m600.resolve()),
            "m600_previous": (
                str(args.m600_previous.resolve()) if args.m600_previous else None
            ),
            "m600_schema": m600_schema,
            "m600_step": m600_step,
            "m600_state_keys": m600_state_keys,
            "source": str(args.source.resolve()),
            "source_schema": source_schema,
            "source_step": source_step,
            "source_state_key": source_state_key,
            "highlr": str(args.highlr.resolve()),
            "highlr_previous": (
                str(args.highlr_previous.resolve()) if args.highlr_previous else None
            ),
            "highlr_schema": highlr_schema,
            "highlr_step": highlr_step,
            "highlr_state_keys": highlr_state_keys,
        },
        "mapping": {
            "source_to_m600_target": mapping,
            "new_m600_targets": new_targets,
        },
        "m600": m600_results,
        "highlr": highlr_results,
        "interval": {
            "definition": "current checkpoint tensor minus previous checkpoint tensor",
            "m600": m600_interval,
            "highlr": highlr_interval,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
