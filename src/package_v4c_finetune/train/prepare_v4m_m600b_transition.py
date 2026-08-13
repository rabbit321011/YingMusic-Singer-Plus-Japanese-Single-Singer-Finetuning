"""Build the function-preserving V4M M600-B transition checkpoint."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import OrderedDict

import torch
from omegaconf import OmegaConf


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src").is_dir():
    YING_REPO = PROJECT_DIR
sys.path.insert(0, str(YING_REPO))

from prepare_v4m_m600d_transition import (  # noqa: E402
    P_KEY,
    SOURCE_DEPTH,
    SOURCE_SCHEMA,
    atomic_torch_save,
    fill_unsupported_pitch_rows,
    sha256_file,
    source_target_mapping,
    split_state_dict,
    strict_load,
    tensor_sha256,
    transformer_parameter_count,
)
from src.YingMusicSinger.melody.midi_p_v4ph import (  # noqa: E402
    MIDI_P_V4PH_SCHEMA,
    V4PHMIDIEmbedding,
)
from src.YingMusicSinger.models.dit import DiT  # noqa: E402
from src.YingMusicSinger.models.model import Singer  # noqa: E402
from src.YingMusicSinger.models.modules import DiTBlock  # noqa: E402


TRANSITION_SCHEMA = "v4m_m600b_transition_v1"
TARGET_DEPTH = 31
TARGET_HEADS = 24
TARGET_FF_MULT = 3
TARGET_DIT_PARAMETERS = 600_462_048
SOURCE_HEADS = 16
SOURCE_FF_MULT = 2
DIM = 1024
DIM_HEAD = 64

ATTN_INPUTS = {
    "attn.to_q.weight",
    "attn.to_q.bias",
    "attn.to_k.weight",
    "attn.to_k.bias",
    "attn.to_v.weight",
    "attn.to_v.bias",
}
ATTN_OUTPUT = "attn.to_out.0.weight"
FF_INPUTS = {"ff.ff.0.0.weight", "ff.ff.0.0.bias"}
FF_OUTPUT = "ff.ff.2.weight"


def target_architecture(source_arch):
    arch = dict(source_arch)
    arch.update(
        {
            "dim": DIM,
            "depth": TARGET_DEPTH,
            "heads": TARGET_HEADS,
            "dim_head": DIM_HEAD,
            "ff_mult": TARGET_FF_MULT,
        }
    )
    return arch


def new_target_block(arch):
    return DiTBlock(
        dim=int(arch["dim"]),
        heads=int(arch["heads"]),
        dim_head=int(arch.get("dim_head", DIM_HEAD)),
        ff_mult=arch["ff_mult"],
        dropout=float(arch.get("dropout", 0.1)),
        qk_norm=arch.get("qk_norm"),
        pe_attn_head=arch.get("pe_attn_head"),
        attn_backend=arch.get("attn_backend", "torch"),
        attn_mask_enabled=bool(arch.get("attn_mask_enabled", False)),
    )


def widen_block(source, target_initial):
    if set(source) != set(target_initial):
        raise ValueError("source/target block state schema differs")
    target = OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in target_initial.items()
    )
    records = []
    for name, source_value in source.items():
        target_value = target[name]
        if source_value.shape == target_value.shape:
            target_value.copy_(source_value)
            policy = "exact_copy"
        elif name in ATTN_INPUTS:
            if source_value.shape[0] != SOURCE_HEADS * DIM_HEAD:
                raise ValueError(f"unexpected source attention input shape: {name}")
            if target_value.shape[0] != TARGET_HEADS * DIM_HEAD:
                raise ValueError(f"unexpected target attention input shape: {name}")
            if source_value.shape[1:] != target_value.shape[1:]:
                raise ValueError(f"attention input trailing shape mismatch: {name}")
            target_value[: source_value.shape[0]].copy_(source_value)
            policy = "copy_old_heads_keep_deterministic_new_heads"
        elif name == ATTN_OUTPUT:
            if source_value.shape != (DIM, SOURCE_HEADS * DIM_HEAD):
                raise ValueError("unexpected source attention output shape")
            if target_value.shape != (DIM, TARGET_HEADS * DIM_HEAD):
                raise ValueError("unexpected target attention output shape")
            target_value[:, : source_value.shape[1]].copy_(source_value)
            target_value[:, source_value.shape[1] :].zero_()
            policy = "copy_old_head_columns_zero_new_head_columns"
        elif name in FF_INPUTS:
            if source_value.shape[0] != DIM * SOURCE_FF_MULT:
                raise ValueError(f"unexpected source FF input shape: {name}")
            if target_value.shape[0] != DIM * TARGET_FF_MULT:
                raise ValueError(f"unexpected target FF input shape: {name}")
            if source_value.shape[1:] != target_value.shape[1:]:
                raise ValueError(f"FF input trailing shape mismatch: {name}")
            target_value[: source_value.shape[0]].copy_(source_value)
            policy = "copy_old_units_keep_deterministic_new_units"
        elif name == FF_OUTPUT:
            if source_value.shape != (DIM, DIM * SOURCE_FF_MULT):
                raise ValueError("unexpected source FF output shape")
            if target_value.shape != (DIM, DIM * TARGET_FF_MULT):
                raise ValueError("unexpected target FF output shape")
            target_value[:, : source_value.shape[1]].copy_(source_value)
            target_value[:, source_value.shape[1] :].zero_()
            policy = "copy_old_unit_columns_zero_new_unit_columns"
        else:
            raise ValueError(
                f"unhandled widened tensor {name}: "
                f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
            )
        records.append(
            {
                "tensor": name,
                "source_shape": list(source_value.shape),
                "target_shape": list(target_value.shape),
                "policy": policy,
            }
        )
    return target, records


def expand_state_dict(source_state, seed, source_arch):
    non_blocks, source_blocks = split_state_dict(source_state)
    mapping = source_target_mapping(SOURCE_DEPTH, TARGET_DEPTH)
    target_to_source = {target: source for source, target in mapping.items()}
    new_targets = sorted(set(range(TARGET_DEPTH)) - set(target_to_source))
    if len(new_targets) != 9:
        raise AssertionError("M600-B must insert exactly nine blocks")
    arch = target_architecture(source_arch)

    target_states = {}
    per_block = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for target_index in range(TARGET_DEPTH):
            initial = OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in new_target_block(arch).state_dict().items()
            )
            if target_index in target_to_source:
                source_index = target_to_source[target_index]
                widened, records = widen_block(source_blocks[source_index], initial)
                target_states[target_index] = widened
                per_block.append(
                    {
                        "target_index": target_index,
                        "source_index": source_index,
                        "policy": "widened_source_block",
                        "tensors": records,
                    }
                )
            else:
                initial["attn_norm.linear.weight"].zero_()
                initial["attn_norm.linear.bias"].zero_()
                target_states[target_index] = initial
                per_block.append(
                    {
                        "target_index": target_index,
                        "source_index": None,
                        "policy": "adaln_zero_identity_block",
                    }
                )

    expanded = OrderedDict()
    inserted_blocks = False
    for key, value in source_state.items():
        if key.startswith("transformer.transformer_blocks."):
            if inserted_blocks:
                continue
            for target_index in range(TARGET_DEPTH):
                for suffix, tensor in target_states[target_index].items():
                    expanded[
                        f"transformer.transformer_blocks.{target_index}.{suffix}"
                    ] = tensor
            inserted_blocks = True
        else:
            expanded[key] = value
    if not inserted_blocks:
        raise ValueError("source state has no transformer blocks")
    return expanded, {
        "source_depth": SOURCE_DEPTH,
        "target_depth": TARGET_DEPTH,
        "source_to_target": {str(source): target for source, target in mapping.items()},
        "new_target_blocks": new_targets,
        "new_block_count": len(new_targets),
        "initialization_seed": seed,
        "per_block": per_block,
    }


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


@torch.no_grad()
def equivalence_audit(source, target, seed, device):
    source = source.to(device).eval()
    target = target.to(device).eval()
    reports = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for frames in (17, 64, 257):
            x = torch.randn(1, frames, 64, device=device)
            cond = torch.randn_like(x)
            text = torch.randint(0, 373, (1, max(1, frames // 3)), device=device)
            midi = torch.randn(1, frames, 128, device=device)
            time = torch.rand(1, device=device)
            for drop_audio, drop_text, drop_midi in (
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
                    drop_audio_cond=drop_audio,
                    drop_text=drop_text,
                    drop_midi=drop_midi,
                    cache=False,
                )[0]
                target_output = target.transformer(
                    x,
                    cond,
                    text,
                    time,
                    midi,
                    drop_audio_cond=drop_audio,
                    drop_text=drop_text,
                    drop_midi=drop_midi,
                    cache=False,
                )[0]
                delta = (source_output - target_output).float()
                source_rms = source_output.float().square().mean().sqrt()
                reports.append(
                    {
                        "frames": frames,
                        "drop_audio": drop_audio,
                        "drop_text": drop_text,
                        "drop_midi": drop_midi,
                        "max_abs": float(delta.abs().max()),
                        "rmse": float(delta.square().mean().sqrt()),
                        "relative_rmse": float(
                            delta.square().mean().sqrt() / source_rms.clamp_min(1e-12)
                        ),
                        "exact": bool(torch.equal(source_output, target_output)),
                    }
                )
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source_sha256", required=True)
    parser.add_argument(
        "--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--equivalence_seed", type=int, default=60_005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_abs_tolerance", type=float)
    parser.add_argument("--relative_rmse_tolerance", type=float)
    parser.add_argument("--skip_model_audit", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    source_path = pathlib.Path(args.source).resolve()
    output_path = pathlib.Path(args.output).resolve()
    actual_sha = sha256_file(source_path)
    if actual_sha != args.source_sha256:
        raise ValueError(f"source SHA256 mismatch: {actual_sha}")
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    if source.get("checkpoint_schema") != SOURCE_SCHEMA:
        raise ValueError("source is not a V4PH checkpoint")
    if source.get("run_state") != "complete" or int(source.get("global_step", -1)) != 500:
        raise ValueError("source must be the completed V4PH Phase-A step 500")
    metadata = source.get("v4ph_training") or {}
    if metadata.get("phase") != "p_only":
        raise ValueError("source metadata is not Phase-A p_only")
    if source.get("midi_p_schema") != MIDI_P_V4PH_SCHEMA:
        raise ValueError("source MIDI_P schema mismatch")

    cfg = OmegaConf.load(args.config)
    arch = dict(cfg.model.arch)
    if (
        int(arch["dim"]) != DIM
        or int(arch["depth"]) != SOURCE_DEPTH
        or int(arch["heads"]) != SOURCE_HEADS
        or arch["ff_mult"] != SOURCE_FF_MULT
    ):
        raise ValueError("source config differs from the M600-B source contract")

    source_state = OrderedDict(
        (key.replace("module.", ""), value)
        for key, value in source["model_state_dict"].items()
    )
    support = source["p_support_counts"].long()
    transition_p = source_state[P_KEY].clone()
    filled_rows = fill_unsupported_pitch_rows(transition_p, support)
    source_state[P_KEY] = transition_p

    expanded_state, block_audit = expand_state_dict(source_state, args.seed, arch)
    parameter_count = transformer_parameter_count(expanded_state)
    if parameter_count != TARGET_DIT_PARAMETERS:
        raise ValueError(
            f"target DiT parameter count {parameter_count} != {TARGET_DIT_PARAMETERS}"
        )

    equivalence = []
    if not args.skip_model_audit:
        source_model = build_singer(
            cfg, SOURCE_DEPTH, SOURCE_HEADS, SOURCE_FF_MULT, args.seed
        )
        target_model = build_singer(
            cfg, TARGET_DEPTH, TARGET_HEADS, TARGET_FF_MULT, args.seed
        )
        strict_load(source_model, source_state, "source transition")
        strict_load(target_model, expanded_state, "M600-B transition")
        equivalence = equivalence_audit(
            source_model,
            target_model,
            args.equivalence_seed,
            torch.device(args.device),
        )
        if not args.dry_run:
            if args.max_abs_tolerance is None or args.relative_rmse_tolerance is None:
                raise ValueError("formal conversion requires both equivalence tolerances")
            worst_abs = max(report["max_abs"] for report in equivalence)
            worst_relative = max(report["relative_rmse"] for report in equivalence)
            if worst_abs > args.max_abs_tolerance:
                raise ValueError(
                    f"M600-B max_abs {worst_abs} exceeds {args.max_abs_tolerance}"
                )
            if worst_relative > args.relative_rmse_tolerance:
                raise ValueError(
                    f"M600-B relative RMSE {worst_relative} exceeds "
                    f"{args.relative_rmse_tolerance}"
                )
            del source_model, target_model

    script_sha = sha256_file(pathlib.Path(__file__).resolve())
    transition = {
        "checkpoint_schema": TRANSITION_SCHEMA,
        "run_state": "transition",
        "global_step": 0,
        "model_state_dict": expanded_state,
        "v4m_transition": {
            "schema": TRANSITION_SCHEMA,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": actual_sha,
            "source_checkpoint_schema": source["checkpoint_schema"],
            "source_global_step": int(source["global_step"]),
            "source_weight": "model_state_dict",
            "source_phase": metadata["phase"],
            "source_arch": {
                "dim": DIM,
                "depth": SOURCE_DEPTH,
                "heads": SOURCE_HEADS,
                "dim_head": DIM_HEAD,
                "ff_mult": SOURCE_FF_MULT,
            },
            "target_arch": {
                "dim": DIM,
                "depth": TARGET_DEPTH,
                "heads": TARGET_HEADS,
                "dim_head": DIM_HEAD,
                "ff_mult": TARGET_FF_MULT,
            },
            "target_dit_parameters": parameter_count,
            "block_transition": block_audit,
            "widening_policy": {
                "attention": "preserve_old_heads_zero_new_output_columns",
                "ffn": "preserve_old_units_zero_new_output_columns",
                "residual_stream": "exact_1024_copy",
            },
            "p_transition": {
                "supported_pitch_rows": int((support[:255] > 0).sum()),
                "unsupported_pitch_rows_filled": int(filled_rows),
                "transition_p_sha256": tensor_sha256(transition_p),
                "pad_fixed_zero": bool(torch.count_nonzero(transition_p[256]) == 0),
            },
            "optimizer_policy": "fresh",
            "scheduler_policy": "fresh_highlr_24k",
            "ema_policy": "fresh_from_transition_weights",
            "conversion_seed": args.seed,
            "equivalence_seed": args.equivalence_seed,
            "equivalence_device": args.device,
            "equivalence_tolerances": {
                "max_abs": args.max_abs_tolerance,
                "relative_rmse": args.relative_rmse_tolerance,
            },
            "equivalence": equivalence,
            "conversion_script_sha256": script_sha,
        },
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": TRANSITION_SCHEMA,
                    "dry_run": True,
                    "target_dit_parameters": parameter_count,
                    "block_transition": block_audit,
                    "p_transition": transition["v4m_transition"]["p_transition"],
                    "equivalence": equivalence,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    atomic_torch_save(transition, output_path)
    output_sha = sha256_file(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{output_sha}  {output_path.name}\n", encoding="ascii"
    )
    report_path = output_path.with_suffix(output_path.suffix + ".audit.json")
    report_path.write_text(
        json.dumps(
            {
                "schema": TRANSITION_SCHEMA,
                "checkpoint": str(output_path),
                "checkpoint_sha256": output_sha,
                **transition["v4m_transition"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint": str(output_path),
                "sha256": output_sha,
                "audit": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
