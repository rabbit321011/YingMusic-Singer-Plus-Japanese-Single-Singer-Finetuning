"""Independently audit the M600-B function-preserving transition."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import torch
from omegaconf import OmegaConf


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src").is_dir():
    YING_REPO = PROJECT_DIR
sys.path.insert(0, str(YING_REPO))

from prepare_v4m_m600b_transition import (  # noqa: E402
    ATTN_INPUTS,
    ATTN_OUTPUT,
    DIM,
    DIM_HEAD,
    FF_INPUTS,
    FF_OUTPUT,
    SOURCE_DEPTH,
    SOURCE_FF_MULT,
    SOURCE_HEADS,
    TARGET_DEPTH,
    TARGET_DIT_PARAMETERS,
    TARGET_FF_MULT,
    TARGET_HEADS,
    TRANSITION_SCHEMA,
)
from prepare_v4m_m600d_transition import (  # noqa: E402
    P_KEY,
    fill_unsupported_pitch_rows,
    sha256_file,
)
from src.YingMusicSinger.melody.midi_p_v4ph import V4PHMIDIEmbedding  # noqa: E402
from src.YingMusicSinger.models.dit import DiT  # noqa: E402
from src.YingMusicSinger.models.model import Singer  # noqa: E402
from src.YingMusicSinger.models.modules import DiTBlock  # noqa: E402


def sections(state):
    blocks = defaultdict(dict)
    non_blocks = {}
    prefix = "transformer.transformer_blocks."
    for name, value in state.items():
        if not name.startswith(prefix):
            non_blocks[name] = value
            continue
        rest = name[len(prefix) :]
        index_text, suffix = rest.split(".", 1)
        blocks[int(index_text)][suffix] = value
    return dict(blocks), non_blocks


def build_singer(cfg, depth, heads, ff_mult, seed):
    arch = dict(cfg.model.arch)
    arch.update({"depth": depth, "heads": heads, "ff_mult": ff_mult})
    dit = DiT(
        **arch,
        text_num_embeds=cfg.datasets_cfg.text_num_embeds,
        mel_dim=cfg.model.mel_spec.n_mel_channels,
        long_skip_connection=True,
    )
    singer = Singer(
        transformer=dit,
        is_tts_pretrain=cfg.model.is_tts_pretrain,
        melody_input_source=cfg.model.melody_input_source,
        cka_disabled=cfg.model.cka_disabled,
        num_channels=None,
        extra_parameters=cfg.extra_parameters,
        mel_spec_kwargs=cfg.model.mel_spec,
        distill_stage=None,
        use_guidance_scale_embed=False,
    )
    singer.midi_p_v4ph = V4PHMIDIEmbedding(seed=seed)
    return singer


def strict_load(module, state, label):
    result = module.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"{label} mismatch: missing={result.missing_keys} "
            f"unexpected={result.unexpected_keys}"
        )


@torch.no_grad()
def whole_model_audit(
    source, target, seed, device, max_abs_limit, relative_limit, enforce
):
    source = source.to(device).eval()
    target = target.to(device).eval()
    reports = []
    with torch.random.fork_rng(devices=[]):
        for sample_seed in (seed, seed + 1):
            torch.manual_seed(sample_seed)
            for frames in (31, 111, 257):
                x = torch.randn(1, frames, 64, device=device)
                cond = torch.randn_like(x)
                text = torch.randint(0, 373, (1, max(1, frames // 3)), device=device)
                midi = torch.randn(1, frames, 128, device=device)
                time = torch.rand(1, device=device)
                for drops in (
                    (False, False, False),
                    (True, False, False),
                    (False, True, True),
                ):
                    source_output = source.transformer(
                        x,
                        cond,
                        text,
                        time,
                        midi,
                        drop_audio_cond=drops[0],
                        drop_text=drops[1],
                        drop_midi=drops[2],
                        cache=False,
                    )[0]
                    target_output = target.transformer(
                        x,
                        cond,
                        text,
                        time,
                        midi,
                        drop_audio_cond=drops[0],
                        drop_text=drops[1],
                        drop_midi=drops[2],
                        cache=False,
                    )[0]
                    delta = (target_output - source_output).float()
                    rmse = delta.square().mean().sqrt()
                    relative = (
                        rmse
                        / source_output.float()
                        .square()
                        .mean()
                        .sqrt()
                        .clamp_min(1e-12)
                    )
                    reports.append(
                        {
                            "seed": sample_seed,
                            "frames": frames,
                            "drops": list(drops),
                            "max_abs": float(delta.abs().max()),
                            "rmse": float(rmse),
                            "relative_rmse": float(relative),
                        }
                    )
    if enforce:
        worst_abs = max(report["max_abs"] for report in reports)
        worst_relative = max(report["relative_rmse"] for report in reports)
        if worst_abs > max_abs_limit:
            raise ValueError(f"independent max_abs limit failed: {worst_abs}")
        if worst_relative > relative_limit:
            raise ValueError(
                f"independent relative RMSE limit failed: {worst_relative}"
            )
    return reports


@torch.no_grad()
def identity_block_audit(block_states, indices, cfg, seed):
    arch = dict(cfg.model.arch)
    reports = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for index in indices:
            block = DiTBlock(
                dim=DIM,
                heads=TARGET_HEADS,
                dim_head=DIM_HEAD,
                ff_mult=TARGET_FF_MULT,
                dropout=float(arch.get("dropout", 0.1)),
                qk_norm=arch.get("qk_norm"),
                pe_attn_head=arch.get("pe_attn_head"),
                attn_backend=arch.get("attn_backend", "torch"),
                attn_mask_enabled=bool(arch.get("attn_mask_enabled", False)),
            ).eval()
            block.load_state_dict(block_states[index], strict=True)
            x = torch.randn(2, 37, DIM)
            t = torch.randn(2, DIM)
            output = block(x, t)
            reports.append({"index": index, "exact": bool(torch.equal(x, output))})
    if any(not report["exact"] for report in reports):
        raise ValueError("new M600-B block is not an exact identity")
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--transition", required=True)
    parser.add_argument("--expected_sha256", required=True)
    parser.add_argument(
        "--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=70_005)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    transition_path = pathlib.Path(args.transition).resolve()
    actual_sha = sha256_file(transition_path)
    if actual_sha != args.expected_sha256:
        raise ValueError(f"transition SHA256 mismatch: {actual_sha}")
    source_path = pathlib.Path(args.source).resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    transition = torch.load(
        transition_path, map_location="cpu", weights_only=False, mmap=True
    )
    if transition.get("checkpoint_schema") != TRANSITION_SCHEMA:
        raise ValueError("transition schema mismatch")
    if transition.get("run_state") != "transition" or transition.get("global_step") != 0:
        raise ValueError("transition state/step mismatch")
    metadata = transition.get("v4m_transition") or {}
    if metadata.get("source_checkpoint_sha256") != sha256_file(source_path):
        raise ValueError("source fingerprint mismatch")
    if metadata.get("target_dit_parameters") != TARGET_DIT_PARAMETERS:
        raise ValueError("target parameter metadata mismatch")
    if metadata.get("target_arch") != {
        "dim": DIM,
        "depth": TARGET_DEPTH,
        "heads": TARGET_HEADS,
        "dim_head": DIM_HEAD,
        "ff_mult": TARGET_FF_MULT,
    }:
        raise ValueError("target architecture metadata mismatch")

    source_state = {
        name.replace("module.", ""): value
        for name, value in source["model_state_dict"].items()
    }
    expected_p = source_state[P_KEY].clone()
    fill_unsupported_pitch_rows(expected_p, source["p_support_counts"].long())
    source_state[P_KEY] = expected_p
    target_state = transition["model_state_dict"]
    source_blocks, source_non_blocks = sections(source_state)
    target_blocks, target_non_blocks = sections(target_state)
    mapping = {
        int(source_index): int(target_index)
        for source_index, target_index in (
            metadata["block_transition"]["source_to_target"]
        ).items()
    }
    new_targets = sorted(set(range(TARGET_DEPTH)) - set(mapping.values()))
    if sorted(source_blocks) != list(range(SOURCE_DEPTH)):
        raise ValueError("source block inventory mismatch")
    if sorted(target_blocks) != list(range(TARGET_DEPTH)):
        raise ValueError("target block inventory mismatch")
    if sorted(mapping) != list(range(SOURCE_DEPTH)) or len(new_targets) != 9:
        raise ValueError("block mapping mismatch")

    exact_tensor_count = 0
    widened_tensor_count = 0
    for source_index, target_index in mapping.items():
        source_block = source_blocks[source_index]
        target_block = target_blocks[target_index]
        if set(source_block) != set(target_block):
            raise ValueError("mapped block state schema mismatch")
        for name, source_value in source_block.items():
            target_value = target_block[name]
            if source_value.shape == target_value.shape:
                if not torch.equal(source_value, target_value):
                    raise ValueError(f"exact mapped tensor differs: {name}")
                exact_tensor_count += 1
            elif name in ATTN_INPUTS or name in FF_INPUTS:
                if not torch.equal(
                    source_value, target_value[: source_value.shape[0]]
                ):
                    raise ValueError(f"preserved widened rows differ: {name}")
                if not torch.isfinite(target_value[source_value.shape[0] :]).all():
                    raise ValueError(f"new widened rows are non-finite: {name}")
                widened_tensor_count += 1
            elif name == ATTN_OUTPUT or name == FF_OUTPUT:
                if not torch.equal(
                    source_value, target_value[:, : source_value.shape[1]]
                ):
                    raise ValueError(f"preserved widened columns differ: {name}")
                if torch.count_nonzero(target_value[:, source_value.shape[1] :]):
                    raise ValueError(f"new output columns are not zero: {name}")
                widened_tensor_count += 1
            else:
                raise ValueError(f"unhandled target widening in audit: {name}")

    for index in new_targets:
        for name in ("attn_norm.linear.weight", "attn_norm.linear.bias"):
            if torch.count_nonzero(target_blocks[index][name]):
                raise ValueError(f"new block {index} has nonzero AdaLN gate")
    if set(source_non_blocks) != set(target_non_blocks):
        raise ValueError("non-block inventory mismatch")
    for name, source_value in source_non_blocks.items():
        if not torch.equal(source_value, target_non_blocks[name]):
            raise ValueError(f"non-block tensor differs: {name}")
    if any(
        not torch.isfinite(value).all()
        for value in target_state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    ):
        raise ValueError("transition contains non-finite tensors")
    parameter_count = sum(
        value.numel()
        for name, value in target_state.items()
        if name.startswith("transformer.") and torch.is_tensor(value)
    )
    if parameter_count != TARGET_DIT_PARAMETERS:
        raise ValueError("target parameter count mismatch")

    cfg = OmegaConf.load(args.config)
    identity_reports = identity_block_audit(target_blocks, new_targets, cfg, args.seed)
    source_model = build_singer(
        cfg, SOURCE_DEPTH, SOURCE_HEADS, SOURCE_FF_MULT, args.seed
    )
    target_model = build_singer(
        cfg, TARGET_DEPTH, TARGET_HEADS, TARGET_FF_MULT, args.seed
    )
    strict_load(source_model, source_state, "source")
    strict_load(target_model, target_state, "target")
    tolerances = metadata.get("equivalence_tolerances") or {}
    whole_reports = whole_model_audit(
        source_model,
        target_model,
        args.seed,
        torch.device(args.device),
        float(tolerances["max_abs"]),
        float(tolerances["relative_rmse"]),
        not args.calibrate,
    )
    report = {
        "schema": "v4m_m600b_transition_independent_audit_v1",
        "status": "calibration" if args.calibrate else "ok",
        "transition": str(transition_path),
        "transition_sha256": actual_sha,
        "target_dit_parameters": parameter_count,
        "source_to_target": mapping,
        "new_target_blocks": new_targets,
        "exact_tensor_count": exact_tensor_count,
        "widened_tensor_count": widened_tensor_count,
        "identity_blocks": identity_reports,
        "whole_model_equivalence": whole_reports,
    }
    output_path = pathlib.Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
