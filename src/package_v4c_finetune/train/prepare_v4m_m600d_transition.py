"""Build and audit the function-preserving V4M M600-D transition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
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

from src.YingMusicSinger.melody.midi_p_v4ph import (  # noqa: E402
    MIDI_P_V4PH_SCHEMA,
    V4PHMIDIEmbedding,
    fill_unsupported_pitch_rows,
)
from src.YingMusicSinger.models.dit import DiT  # noqa: E402
from src.YingMusicSinger.models.model import Singer  # noqa: E402
from src.YingMusicSinger.models.modules import DiTBlock  # noqa: E402


TRANSITION_SCHEMA = "v4m_m600d_transition_v1"
SOURCE_SCHEMA = "v4ph_training_checkpoint_v1"
SOURCE_DEPTH = 22
TARGET_DEPTH = 40
TARGET_DIT_PARAMETERS = 602_599_648
P_KEY = "midi_p_v4ph.embedding.weight"
BLOCK_PATTERN = re.compile(r"^transformer\.transformer_blocks\.(\d+)\.(.+)$")


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_target_mapping(
    source_depth: int = SOURCE_DEPTH, target_depth: int = TARGET_DEPTH
) -> dict[int, int]:
    """Spread source blocks over the target depth while preserving both endpoints."""
    denominator = source_depth - 1
    mapping = {
        source: (source * (target_depth - 1) + denominator // 2) // denominator
        for source in range(source_depth)
    }
    targets = list(mapping.values())
    if len(set(targets)) != source_depth or targets != sorted(targets):
        raise AssertionError("source-to-target block mapping is not strictly ordered")
    if targets[0] != 0 or targets[-1] != target_depth - 1:
        raise AssertionError("source-to-target block mapping must preserve endpoints")
    return mapping


def split_state_dict(state_dict: dict[str, torch.Tensor]):
    non_blocks = OrderedDict()
    blocks: dict[int, OrderedDict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        match = BLOCK_PATTERN.match(key)
        if match is None:
            non_blocks[key] = value
            continue
        index = int(match.group(1))
        blocks.setdefault(index, OrderedDict())[match.group(2)] = value
    if sorted(blocks) != list(range(SOURCE_DEPTH)):
        raise ValueError(f"source block indices are incomplete: {sorted(blocks)}")
    suffixes = [tuple(block) for block in blocks.values()]
    if any(suffix != suffixes[0] for suffix in suffixes[1:]):
        raise ValueError("source blocks do not share one state schema")
    return non_blocks, blocks


def initialize_identity_blocks(
    target_indices: list[int], seed: int, arch: dict
) -> dict[int, OrderedDict[str, torch.Tensor]]:
    states = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for target_index in target_indices:
            block = DiTBlock(
                dim=int(arch["dim"]),
                heads=int(arch["heads"]),
                dim_head=int(arch.get("dim_head", 64)),
                ff_mult=arch["ff_mult"],
                dropout=float(arch.get("dropout", 0.1)),
                qk_norm=arch.get("qk_norm"),
                pe_attn_head=arch.get("pe_attn_head"),
                attn_backend=arch.get("attn_backend", "torch"),
                attn_mask_enabled=bool(arch.get("attn_mask_enabled", False)),
            )
            torch.nn.init.zeros_(block.attn_norm.linear.weight)
            torch.nn.init.zeros_(block.attn_norm.linear.bias)
            states[target_index] = OrderedDict(
                (key, value.detach().cpu().clone())
                for key, value in block.state_dict().items()
            )
    return states


def expand_state_dict(
    source_state: dict[str, torch.Tensor], seed: int, arch: dict
):
    non_blocks, source_blocks = split_state_dict(source_state)
    mapping = source_target_mapping()
    target_to_source = {target: source for source, target in mapping.items()}
    new_targets = sorted(set(range(TARGET_DEPTH)) - set(target_to_source))
    new_states = initialize_identity_blocks(new_targets, seed, arch)

    expanded = OrderedDict()
    inserted_blocks = False
    for key, value in source_state.items():
        if BLOCK_PATTERN.match(key):
            if inserted_blocks:
                continue
            for target in range(TARGET_DEPTH):
                if target in target_to_source:
                    state = source_blocks[target_to_source[target]]
                else:
                    state = new_states[target]
                for suffix, tensor in state.items():
                    expanded[
                        f"transformer.transformer_blocks.{target}.{suffix}"
                    ] = tensor
            inserted_blocks = True
        else:
            expanded[key] = value
    if not inserted_blocks:
        raise ValueError("source state has no transformer blocks")

    audit = {
        "source_depth": SOURCE_DEPTH,
        "target_depth": TARGET_DEPTH,
        "source_to_target": {str(key): value for key, value in mapping.items()},
        "new_target_blocks": new_targets,
        "new_block_count": len(new_targets),
        "new_block_seed": seed,
    }
    return expanded, audit


def build_singer(cfg, depth: int, seed: int) -> Singer:
    arch = dict(cfg.model.arch)
    arch["depth"] = depth
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


def strict_load(module: torch.nn.Module, state: dict[str, torch.Tensor], label: str):
    incompatible = module.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"{label} state mismatch: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )


@torch.no_grad()
def equivalence_audit(source: Singer, target: Singer, seed: int):
    source.eval()
    target.eval()
    reports = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for frames in (17, 64, 257):
            x = torch.randn(1, frames, 64)
            cond = torch.randn_like(x)
            text = torch.randint(0, 373, (1, max(1, frames // 3)))
            midi = torch.randn(1, frames, 128)
            time = torch.rand(1)
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
                reports.append(
                    {
                        "frames": frames,
                        "drop_audio": drop_audio,
                        "drop_text": drop_text,
                        "drop_midi": drop_midi,
                        "max_abs": float(delta.abs().max()),
                        "rmse": float(delta.square().mean().sqrt()),
                        "exact": bool(torch.equal(source_output, target_output)),
                    }
                )
    return reports


def transformer_parameter_count(state: dict[str, torch.Tensor]) -> int:
    return sum(
        value.numel()
        for key, value in state.items()
        if key.startswith("transformer.") and torch.is_tensor(value)
    )


def atomic_torch_save(payload, output: pathlib.Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source_sha256", required=True)
    parser.add_argument("--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--equivalence_seed", type=int, default=60_004)
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
    if int(arch["dim"]) != 1024 or int(arch["depth"]) != SOURCE_DEPTH:
        raise ValueError("source config is not the audited 1024 x 22 architecture")
    if int(arch["heads"]) != 16 or arch["ff_mult"] != 2:
        raise ValueError("source attention/FFN config differs from M600-D contract")

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
        source_model = build_singer(cfg, SOURCE_DEPTH, args.seed)
        target_model = build_singer(cfg, TARGET_DEPTH, args.seed)
        strict_load(source_model, source_state, "source transition")
        strict_load(target_model, expanded_state, "M600-D transition")
        equivalence = equivalence_audit(
            source_model, target_model, args.equivalence_seed
        )
        if any(not report["exact"] for report in equivalence):
            worst = max(report["max_abs"] for report in equivalence)
            raise ValueError(f"M600-D is not bit-exact to source; worst max_abs={worst}")
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
                "dim": 1024,
                "depth": SOURCE_DEPTH,
                "heads": 16,
                "dim_head": int(arch.get("dim_head", 64)),
                "ff_mult": 2,
            },
            "target_arch": {
                "dim": 1024,
                "depth": TARGET_DEPTH,
                "heads": 16,
                "dim_head": int(arch.get("dim_head", 64)),
                "ff_mult": 2,
            },
            "target_dit_parameters": parameter_count,
            "block_transition": block_audit,
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
    print(json.dumps({"checkpoint": str(output_path), "sha256": output_sha, "audit": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
