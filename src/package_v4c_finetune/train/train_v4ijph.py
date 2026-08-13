#!/usr/bin/env python3
"""V4IjPH v2 training: exact V4IPH@8k transition plus global INS style."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
from ema_pytorch import EMA
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
for candidate in (PROJECT_DIR, PACKAGE_DIR, YING_REPO):
    value = os.fspath(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from h_alignment.placement_v4iph import render_h_pul_placements  # noqa: E402
from train_v4iph import (  # noqa: E402
    HDataset as V4IPHDataset,
    ResumableDistributedSampler,
    assert_rank_parameter_fingerprints,
    atomic_torch_save,
    capture_rng_state,
    compute_cka_loss_from_hidden,
    distributed_barrier,
    restore_rng_state,
    seed_everything,
    setup_ddp,
    sync_gradients,
)
from v4iph_contract import (  # noqa: E402
    REFERENCE_MODE_ALL_B,
    compute_flow_losses,
)
from v4ijph_contract import (  # noqa: E402
    InsStyleAdapter,
    V4IJPH_ADAPTER_BETAS,
    V4IJPH_ADAPTER_EMA_BETA,
    V4IJPH_ADAPTER_EMA_UPDATE_AFTER_STEP,
    V4IJPH_ADAPTER_EMA_UPDATE_EVERY,
    V4IJPH_ADAPTER_WEIGHT_DECAY,
    V4IJPH_CHECKPOINT_SCHEMA,
    V4IJPH_INS_DROPOUT,
    V4IJPH_MAX_STEPS,
    V4IJPH_SOURCE_CHECKPOINT_SCHEMA,
    V4IJPH_TRANSITION_STEP,
    adapter_lr_for_global_update,
    adapter_updates_for_global_step,
    assert_zero_output_initialization,
    build_style_cond,
    state_dict_sha256,
    strict_load_module,
)
from v4ijph_ins_cache import (  # noqa: E402
    InsCache,
    load_h_manifest,
    sha256_file,
)


FRAME_RATE = 44_100 / 2_048
SEP_TOKEN = 365
PUL_TOKEN = 366
MIDI_P_KEY = "midi_p_v4ph.embedding.weight"
EXPECTED_V4IPH_POLICY = {
    "schema": V4IJPH_SOURCE_CHECKPOINT_SCHEMA,
    "placement_mode": "phone_pul",
    "reference_mode": "all_b",
    "phase": "joint",
    "lr": 1.4e-5,
    "warmup_steps": 500,
    "hold_steps": 12_000,
    "max_steps": V4IJPH_MAX_STEPS,
    "grad_accum": 4,
    "batch_size": 1,
    "seed": 42,
    "eval_seed": 1_042,
    "drop_text": 0.15,
    "flow_b_weight": 2.0,
    "cka_weight": 0.7,
}


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def is_main() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def strict_load_model(module, state_dict, label):
    normalized = {key.replace("module.", ""): value for key, value in state_dict.items()}
    strict_load_module(module, normalized, label)


class V4IjPHDataset(V4IPHDataset):
    def __init__(self, *args, ins_cache: InsCache, **kwargs):
        super().__init__(*args, **kwargs)
        self.ins_cache = ins_cache
        self.ins_cache.assert_manifest(self.records)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["ins"] = self.ins_cache.vector(item["sample_id"])
        return item


def collate_v4ijph(batch):
    return {
        "wav": [item["wav"] for item in batch],
        "sr": [item["sr"] for item in batch],
        "phrases": [item["phrases"] for item in batch],
        "tier": [item["tier"] for item in batch],
        "duration": [item["duration"] for item in batch],
        "h_candidates": [item["h_candidates"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "ins": [item["ins"] for item in batch],
        "game_cache": [item["game_cache"] for item in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/YingMusicSinger/config/YingMusic_Singer.yaml",
    )
    parser.add_argument(
        "--vae_config",
        default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",
    )
    parser.add_argument(
        "--vae_ckpt",
        default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt",
    )
    parser.add_argument("--source_v4iph_checkpoint", required=True)
    parser.add_argument("--source_v4iph_sha256", required=True)
    parser.add_argument("--game_cache_manifest", required=True)
    parser.add_argument("--ins_train_cache", required=True)
    parser.add_argument("--ins_eval_cache", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--eval_manifest", required=True)
    parser.add_argument("--train_manifest_sha256", required=True)
    parser.add_argument("--eval_manifest_sha256", required=True)
    parser.add_argument("--h_config_fingerprint", required=True)
    parser.add_argument("--audio_root_override")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--hold_steps", type=int, default=12_000)
    parser.add_argument("--max_steps", type=int, default=V4IJPH_MAX_STEPS)
    parser.add_argument("--save_every", type=int, default=2_000)
    parser.add_argument("--eval_every", type=int, default=1_000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=1_042)
    parser.add_argument("--drop_text", type=float, default=0.15)
    parser.add_argument("--flow_b_weight", type=float, default=2.0)
    parser.add_argument("--cka_weight", type=float, default=0.7)
    parser.add_argument("--t_shift", type=float, default=0.5)
    parser.add_argument("--expected_world_size", type=int, default=4)
    parser.add_argument("--stop_after_step", type=int)
    parser.add_argument("--smoke_assertions", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "batch_size": 1,
        "grad_accum": 4,
        "lr": 1.4e-5,
        "warmup_steps": 500,
        "hold_steps": 12_000,
        "max_steps": V4IJPH_MAX_STEPS,
        "max_duration": 30.0,
        "num_workers": 0,
        "seed": 42,
        "eval_seed": 1_042,
        "drop_text": 0.15,
        "flow_b_weight": 2.0,
        "cka_weight": 0.7,
        "t_shift": 0.5,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V4IjPH v2 freezes {name}={expected}, got {getattr(args, name)}"
            )
    for name in ("save_every", "eval_every", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    run_target = args.stop_after_step or args.max_steps
    if not V4IJPH_TRANSITION_STEP < run_target <= V4IJPH_MAX_STEPS:
        raise ValueError(
            f"run target must be in "
            f"[{V4IJPH_TRANSITION_STEP + 1}, {V4IJPH_MAX_STEPS}]"
        )
    expected_sha = args.source_v4iph_sha256
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("source_v4iph_sha256 must be lowercase SHA256")
    required_files = (
        args.config,
        args.vae_config,
        args.vae_ckpt,
        args.source_v4iph_checkpoint,
        args.game_cache_manifest,
        args.train_manifest,
        args.eval_manifest,
    )
    for path in required_files:
        if not pathlib.Path(path).is_file():
            raise FileNotFoundError(path)
    if args.resume and not pathlib.Path(args.resume).is_file():
        raise FileNotFoundError(args.resume)


def load_and_validate_source(args: argparse.Namespace, world_size: int) -> dict:
    actual_sha = sha256_file(args.source_v4iph_checkpoint)
    if actual_sha != args.source_v4iph_sha256:
        raise ValueError("V4IPH step-8000 source SHA256 mismatch")
    source = torch.load(
        args.source_v4iph_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if source.get("checkpoint_schema") != V4IJPH_SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError("V4IjPH source is not a V4IPH checkpoint")
    if int(source.get("global_step", -1)) != V4IJPH_TRANSITION_STEP:
        raise ValueError("V4IjPH source is not V4IPH step 8000")
    if source.get("run_state") != "running":
        raise ValueError("V4IPH step-8000 source must be a periodic running checkpoint")
    metadata = source.get("v4iph_training")
    if not isinstance(metadata, dict):
        raise ValueError("V4IPH source metadata is missing")
    for key, expected in EXPECTED_V4IPH_POLICY.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"V4IPH source policy {key} mismatch: {metadata.get(key)!r} != {expected!r}"
            )
    if int(metadata.get("world_size", -1)) != world_size:
        raise ValueError("V4IPH source world size differs from this run")
    if int(source["scheduler_state_dict"].get("last_epoch", -1)) != V4IJPH_TRANSITION_STEP:
        raise ValueError("V4IPH source scheduler is not at step 8000")
    if int(source["ema_step"].item()) != V4IJPH_TRANSITION_STEP:
        raise ValueError("V4IPH source EMA is not at step 8000")
    if len(source["optimizer_state_dict"].get("state", {})) != 947:
        raise ValueError("V4IPH source optimizer state is incomplete")
    rank_states = source.get("rank_states") or []
    rank_ids = sorted(int(item["rank"]) for item in rank_states)
    if rank_ids != list(range(world_size)):
        raise ValueError("V4IPH source rank states are incomplete")
    file_sha = metadata.get("file_sha256") or {}
    source_paths = {
        "training_code": SCRIPT_DIR / "train_v4iph.py",
        "contract_code": SCRIPT_DIR / "v4iph_contract.py",
        "placement_code": PACKAGE_DIR / "h_alignment" / "placement_v4iph.py",
        "model_config": pathlib.Path(args.config),
        "vae_config": pathlib.Path(args.vae_config),
        "vae_checkpoint": pathlib.Path(args.vae_ckpt),
        "game_cache_manifest": pathlib.Path(args.game_cache_manifest),
    }
    for label, path in source_paths.items():
        actual = sha256_file(path)
        if file_sha.get(label) != actual:
            raise ValueError(f"V4IPH source {label} runtime SHA256 mismatch")
    return source


def cache_fingerprint_paths(cache: InsCache, prefix: str) -> dict[str, pathlib.Path]:
    paths = {f"{prefix}_metadata": cache.root / "metadata.json"}
    paths.update(
        {
            f"{prefix}_{name.replace('.', '_')}": cache.root / name
            for name in ("records.jsonl", *cache.ARRAY_FILES)
        }
    )
    return paths


def build_training_metadata(
    args: argparse.Namespace,
    world_size: int,
    train_cache: InsCache,
    eval_cache: InsCache,
) -> dict:
    inference_code = PACKAGE_DIR / "infer" / "v4ijph_sampling.py"
    inference_probe = PACKAGE_DIR / "infer" / "probe_v4ijph_inference.py"
    paths = {
        "training_code": pathlib.Path(__file__).resolve(),
        "contract_code": SCRIPT_DIR / "v4ijph_contract.py",
        "ins_cache_code": SCRIPT_DIR / "v4ijph_ins_cache.py",
        "placement_code": PACKAGE_DIR / "h_alignment" / "placement_v4iph.py",
        "inference_sampling_code": inference_code,
        "inference_probe_code": inference_probe,
        "model_config": pathlib.Path(args.config),
        "vae_config": pathlib.Path(args.vae_config),
        "vae_checkpoint": pathlib.Path(args.vae_ckpt),
        "game_cache_manifest": pathlib.Path(args.game_cache_manifest),
        "source_v4iph_checkpoint": pathlib.Path(args.source_v4iph_checkpoint),
        **cache_fingerprint_paths(train_cache, "ins_train_cache"),
        **cache_fingerprint_paths(eval_cache, "ins_eval_cache"),
    }
    file_sha256 = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        file_sha256[label] = sha256_file(path)
    if file_sha256["source_v4iph_checkpoint"] != args.source_v4iph_sha256:
        raise ValueError("Metadata source V4IPH SHA256 mismatch")
    return {
        "schema": V4IJPH_CHECKPOINT_SCHEMA,
        "transition": {
            "source_schema": V4IJPH_SOURCE_CHECKPOINT_SCHEMA,
            "source_global_step": V4IJPH_TRANSITION_STEP,
            "source_sha256": args.source_v4iph_sha256,
            "policy": "exact_model_ema_optimizer_scheduler_rank_rng_and_data_cursor",
            "replayed_prefix_steps": 0,
        },
        "reference_mode": "all_b",
        "reference_policy": "ref_len_zero_global_ins_style_or_exact_zero_null",
        "flow_policy": "flow_a_zero_plus_2_flow_b_full_timeline_plus_0_7_cka",
        "placement_mode": "phone_pul",
        "midi_teacher": "GAME medium K4 offline cache",
        "midi_fuzz_disturb": False,
        "ins": {
            "train_cache": train_cache.provenance(),
            "eval_cache": eval_cache.provenance(),
            "dropout": V4IJPH_INS_DROPOUT,
            "style_guidance": 1.0,
            "training_source": "target_B_exact_VAE_waveform",
            "inference_source": "independent_user_selected_Hanamaru_R",
        },
        "adapter": {
            "architecture": "768_512_silu_256_silu_64_output_zero",
            "first_global_update": V4IJPH_TRANSITION_STEP + 1,
            "peak_step": 14_000,
            "mid_step": 18_000,
            "final_step": V4IJPH_MAX_STEPS,
            "peak_lr": 1e-4,
            "mid_lr": 1e-5,
            "betas": list(V4IJPH_ADAPTER_BETAS),
            "weight_decay": V4IJPH_ADAPTER_WEIGHT_DECAY,
            "ema_beta": V4IJPH_ADAPTER_EMA_BETA,
            "ema_update_after_adapter_step": V4IJPH_ADAPTER_EMA_UPDATE_AFTER_STEP,
            "ema_update_every": V4IJPH_ADAPTER_EMA_UPDATE_EVERY,
        },
        "h_pul": {
            "pul_token_id": PUL_TOKEN,
            "sep_token_id": SEP_TOKEN,
            "sep_policy": "next_runtime_control_anchor_minus_one",
            "final_sep_policy": "last_dense_text_frame",
            "pul_policy": "repeat_after_packed_lyrics_until_sep",
            "hard_fallback_policy": "whole_sample_exact_control",
        },
        "world_size": world_size,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "hold_steps": args.hold_steps,
        "max_steps": args.max_steps,
        "max_duration": args.max_duration,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "drop_text": args.drop_text,
        "flow_b_weight": args.flow_b_weight,
        "cka_weight": args.cka_weight,
        "t_shift": args.t_shift,
        "train_manifest_sha256": args.train_manifest_sha256,
        "eval_manifest_sha256": args.eval_manifest_sha256,
        "h_config_fingerprint": args.h_config_fingerprint,
        "file_sha256": file_sha256,
    }


def validate_resume(checkpoint: dict, metadata: dict, world_size: int) -> None:
    if checkpoint.get("checkpoint_schema") != V4IJPH_CHECKPOINT_SCHEMA:
        raise ValueError("Resume checkpoint schema is not V4IjPH v2")
    if checkpoint.get("v4ijph_training") != metadata:
        raise ValueError("Resume checkpoint metadata differs from current runtime")
    step = int(checkpoint.get("global_step", -1))
    adapter_updates_for_global_step(step)
    rank_states = checkpoint.get("rank_states") or []
    if sorted(int(item["rank"]) for item in rank_states) != list(range(world_size)):
        raise ValueError("Resume checkpoint rank states are incomplete")
    required = {
        "model_state_dict",
        "ema_model_state_dict",
        "ema_step",
        "ema_initted",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "adapter_state_dict",
        "adapter_optimizer_state_dict",
        "adapter_ema_model_state_dict",
        "adapter_ema_step",
        "adapter_ema_initted",
        "rank_states",
        "p_support_counts",
        "p_distance_history",
        "initial_p_weight",
        "initial_frozen_fingerprint",
        "phase_a_init_audit",
        "pul_embedding_init",
        "trainable_parameter_names",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Resume checkpoint lacks keys: {missing}")


class V4IjPHObjective:
    """One shared real-data objective for training and the single-card probe."""

    def __init__(
        self,
        *,
        raw_model,
        adapter,
        vae,
        args,
        device,
        game_cache_to_model_tracks,
    ) -> None:
        self.raw_model = raw_model
        self.adapter = adapter
        self.vae = vae
        self.args = args
        self.device = device
        self.game_cache_to_model_tracks = game_cache_to_model_tracks

    def process_batch(self, batch):
        with torch.no_grad():
            waveform = batch["wav"][0].to(self.device)
            sample_rate = batch["sr"][0]
            waveform_2d = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
            latent = self.vae.encode_audio(waveform_2d, in_sr=sample_rate)
            full_latent = latent.squeeze(0).transpose(0, 1).unsqueeze(0)
            _, frames, _ = full_latent.shape
            game_tracks = self.game_cache_to_model_tracks(
                batch["game_cache"][0],
                num_samples=waveform_2d.shape[-1],
                target_len=frames,
                sample_rate=sample_rate,
            )
            p_classes = game_tracks["p_classes"].unsqueeze(0).to(self.device)
            midi_probs = game_tracks["cka_probs"].unsqueeze(0).to(self.device)
            with torch.enable_grad():
                midi = self.raw_model.midi_p_v4ph(p_classes)
            rendered = render_h_pul_placements(
                batch["phrases"][0],
                batch["h_candidates"][0],
                ref_len=0,
                total_frames=frames,
                sep_token_id=SEP_TOKEN,
                pul_token_id=PUL_TOKEN,
            )
            selected = rendered["phone_pul"]["text"]
            text = torch.tensor(
                selected, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            stats = {
                "phone_phrases": rendered["phone_phrase_count"],
                "fallback_phrases": (
                    rendered["pul_phrase_count"]
                    + rendered["exact_control_phrase_count"]
                ),
                "pul_phrases": rendered["pul_phrase_count"],
                "exact_control_phrases": rendered["exact_control_phrase_count"],
                "pul_frames": rendered["pul_frame_count"],
                "sample_control_anomaly": int(rendered["sample_control_anomaly"]),
                "sample_structural_fallback": int(
                    rendered["sample_structural_fallback"]
                ),
                "nonpad_tokens": sum(token != 0 for token in selected),
            }
            if self.args.smoke_assertions:
                if len(selected) != frames or max(selected, default=0) > PUL_TOKEN:
                    raise AssertionError("V4IjPH H/PUL placement contract failed")
                if tuple(midi.shape) != (1, frames, 128):
                    raise AssertionError("V4IjPH GAME P shape mismatch")
        ins = batch["ins"][0].to(
            device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        return full_latent, midi, midi_probs, text, p_classes, ins, stats

    def run_dit(self, x_t, cond, text, t, midi, drop_ins, drop_text, drop_midi):
        transformer = self.raw_model.transformer
        time_embedding = transformer.time_embed(t)
        x, _ = transformer.get_input_embed(
            x_t,
            cond,
            text,
            midi,
            drop_audio_cond=drop_ins,
            drop_text=drop_text,
            drop_midi=drop_midi,
            cache=False,
        )
        rope = transformer.rotary_embed.forward_from_seq_len(x_t.shape[1])
        hidden_states = []
        residual = x if transformer.long_skip_connection is not None else None
        for index, block in enumerate(transformer.transformer_blocks):
            x = block(x, time_embedding, mask=None, rope=rope)
            if index >= len(transformer.transformer_blocks) - 3:
                hidden_states.append(x)
        if residual is not None:
            x = transformer.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = transformer.norm_out(x, time_embedding)
        return transformer.proj_out(x), hidden_states

    def compute_loss(self, batch, *, force_ins_drop=None):
        (
            full_latent,
            midi,
            midi_probs,
            text,
            p_classes,
            ins,
            placement_stats,
        ) = self.process_batch(batch)
        frames = full_latent.shape[1]
        u = torch.rand(1, device=self.device)
        t = self.args.t_shift * u / (1 - self.args.t_shift * u)
        noise = torch.randn_like(full_latent)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * full_latent
        v_target = full_latent - noise

        sampled_drop = random.random() < V4IJPH_INS_DROPOUT
        drop_text = random.random() < self.args.drop_text
        drop_midi = random.random() < 0.3
        drop_ins = sampled_drop if force_ins_drop is None else bool(force_ins_drop)
        cond, style = build_style_cond(
            self.adapter, ins, frames, drop_ins=drop_ins
        )
        if self.args.smoke_assertions:
            if tuple(cond.shape) != tuple(full_latent.shape):
                raise AssertionError("V4IjPH cond shape differs from VAE latent")
            if drop_ins and bool(torch.count_nonzero(cond)):
                raise AssertionError("V4IjPH dropped INS cond is not exactly zero")
            if not drop_ins and (
                not torch.equal(cond[:, 0], style)
                or not torch.equal(cond[:, -1], style)
            ):
                raise AssertionError("V4IjPH style is not global over the timeline")

        prediction, hidden = self.run_dit(
            x_t, cond, text, t, midi, drop_ins, drop_text, drop_midi
        )
        flow_a, flow_b, flow = compute_flow_losses(
            prediction,
            v_target,
            ref_len=0,
            flow_b_weight=self.args.flow_b_weight,
            phase="joint",
            reference_mode=REFERENCE_MODE_ALL_B,
        )
        cka = compute_cka_loss_from_hidden(hidden, midi_probs)
        loss = flow + self.args.cka_weight * cka
        return loss, {
            "flow_a": flow_a.item(),
            "flow_b": flow_b.item(),
            "cka": cka.item(),
            "style_l2": float(style.norm(dim=1).mean().detach().cpu()),
            "ins_dropped": int(drop_ins),
            "p_class_counts": torch.bincount(
                p_classes.reshape(-1), minlength=257
            ).detach(),
            **placement_stats,
        }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("V4IjPH training requires CUDA")
    if "LOCAL_RANK" in os.environ:
        local_rank, world_size = setup_ddp()
    else:
        local_rank, world_size = 0, 1
        torch.cuda.set_device(0)
    if world_size != args.expected_world_size:
        raise ValueError(f"world size {world_size} != expected {args.expected_world_size}")
    rank = torch.distributed.get_rank() if world_size > 1 else 0
    device = torch.device(f"cuda:{local_rank}")
    seed_everything(args.seed)

    if is_main():
        print(
            f"[V4IjPH v2] world={world_size} device={device} "
            f"transition=V4IPH@{V4IJPH_TRANSITION_STEP}",
            flush=True,
        )
    source = load_and_validate_source(args, world_size)

    train_manifest = load_h_manifest(pathlib.Path(args.train_manifest).resolve())
    eval_manifest = load_h_manifest(pathlib.Path(args.eval_manifest).resolve())
    if sha256_file(args.train_manifest) != args.train_manifest_sha256:
        raise ValueError("Train manifest SHA256 mismatch")
    if sha256_file(args.eval_manifest) != args.eval_manifest_sha256:
        raise ValueError("Eval manifest SHA256 mismatch")
    train_cache = InsCache(
        args.ins_train_cache,
        expected_manifest_sha256=args.train_manifest_sha256,
        expected_max_duration_sec=args.max_duration,
    )
    eval_cache = InsCache(
        args.ins_eval_cache,
        expected_manifest_sha256=args.eval_manifest_sha256,
        expected_max_duration_sec=args.max_duration,
    )
    train_cache.assert_manifest(train_manifest)
    eval_cache.assert_manifest(eval_manifest)

    metadata_box = [
        build_training_metadata(args, world_size, train_cache, eval_cache)
        if is_main()
        else None
    ]
    if world_size > 1:
        torch.distributed.broadcast_object_list(metadata_box, src=0)
    training_metadata = metadata_box[0]

    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        validate_resume(resume, training_metadata, world_size)
    state = resume if resume is not None else source
    global_step = int(state["global_step"])
    run_target = args.stop_after_step or args.max_steps
    if global_step >= run_target:
        raise ValueError(f"checkpoint step {global_step} is not before target {run_target}")

    cfg = OmegaConf.load(args.config)
    from src.YingMusicSinger.melody.game_cache_v4ph import (  # noqa: E402
        GAME_CACHE_SCHEMA,
        game_cache_to_model_tracks,
    )
    from src.YingMusicSinger.melody.midi_p_v4ph import (  # noqa: E402
        MIDI_P_V4PH_SCHEMA,
        V4PHMIDIEmbedding,
    )
    from src.YingMusicSinger.models.dit import DiT  # noqa: E402
    from src.YingMusicSinger.models.model import Singer  # noqa: E402
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import (  # noqa: E402
        StableAudioInfer,
    )

    dit = DiT(
        **cfg.model.arch,
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
    singer.midi_p_v4ph = V4PHMIDIEmbedding(seed=args.seed)
    strict_load_model(singer, state["model_state_dict"], "DiT model")
    singer = singer.to(device).train()
    raw_model = singer

    with torch.random.fork_rng(devices=[local_rank]):
        torch.manual_seed(args.seed + 9_173)
        torch.cuda.manual_seed(args.seed + 9_173)
        adapter = InsStyleAdapter().to(device)
    initial_adapter_sha256 = state_dict_sha256(adapter)
    if resume is None:
        assert_zero_output_initialization(adapter)
    else:
        strict_load_module(adapter, state["adapter_state_dict"], "resume adapter")
        initial_adapter_sha256 = state["initial_adapter_sha256"]
    adapter.train()

    vae = StableAudioInfer(
        model_config_path=args.vae_config,
        model_ckpt_path=args.vae_ckpt,
    ).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False

    ema = EMA(
        raw_model,
        beta=float(cfg.ema_kwargs.beta),
        update_after_step=int(cfg.ema_kwargs.update_after_step),
        update_every=int(cfg.ema_kwargs.update_every),
    ).to(device)
    strict_load_model(ema.ema_model, state["ema_model_state_dict"], "DiT EMA")
    ema.step.copy_(state["ema_step"])
    ema.initted.copy_(state["ema_initted"])

    adapter_ema = EMA(
        adapter,
        beta=V4IJPH_ADAPTER_EMA_BETA,
        update_after_step=V4IJPH_ADAPTER_EMA_UPDATE_AFTER_STEP,
        update_every=V4IJPH_ADAPTER_EMA_UPDATE_EVERY,
    ).to(device)
    if resume is not None:
        strict_load_module(
            adapter_ema.ema_model,
            state["adapter_ema_model_state_dict"],
            "adapter EMA",
        )
        adapter_ema.step.copy_(state["adapter_ema_step"])
        adapter_ema.initted.copy_(state["adapter_ema_initted"])

    optimizer = torch.optim.AdamW(
        [parameter for parameter in raw_model.parameters() if parameter.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=1e-2,
    )

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if step < args.warmup_steps + args.hold_steps:
            return 1.0
        decay_steps = max(
            1, args.max_steps - args.warmup_steps - args.hold_steps
        )
        progress = (step - args.warmup_steps - args.hold_steps) / decay_steps
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    if int(scheduler.last_epoch) != global_step:
        raise ValueError("DiT scheduler step differs from checkpoint")
    if [group["lr"] for group in optimizer.param_groups] != list(
        scheduler.get_last_lr()
    ):
        raise ValueError("DiT optimizer/scheduler LR mismatch")

    adapter_optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=adapter_lr_for_global_update(global_step + 1),
        betas=V4IJPH_ADAPTER_BETAS,
        weight_decay=V4IJPH_ADAPTER_WEIGHT_DECAY,
    )
    if resume is not None:
        adapter_optimizer.load_state_dict(state["adapter_optimizer_state_dict"])

    train_dataset = V4IjPHDataset(
        args.train_manifest,
        args.train_manifest_sha256,
        args.h_config_fingerprint,
        args.max_duration,
        args.game_cache_manifest,
        args.audio_root_override,
        ins_cache=train_cache,
    )
    eval_dataset = V4IjPHDataset(
        args.eval_manifest,
        args.eval_manifest_sha256,
        args.h_config_fingerprint,
        args.max_duration,
        args.game_cache_manifest,
        args.audio_root_override,
        ins_cache=eval_cache,
    )
    train_sampler = ResumableDistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_v4ijph,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_v4ijph,
        pin_memory=True,
        drop_last=True,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    rank_states = {int(item["rank"]): item for item in state["rank_states"]}
    rank_state = rank_states[rank]
    sampler_epoch = int(rank_state["sampler_epoch"])
    batch_offset = int(rank_state["batch_offset"])
    consumed_batches = global_step * args.grad_accum
    expected_epoch = (consumed_batches - 1) // train_sampler.num_samples
    expected_offset = consumed_batches - expected_epoch * train_sampler.num_samples
    if (sampler_epoch, batch_offset) != (expected_epoch, expected_offset):
        raise ValueError("Checkpoint data cursor does not match its global step")

    metric_defaults = {
        "flow_a": 0.0,
        "flow_b": 0.0,
        "cka": 0.0,
        "style_l2": 0.0,
        "ins_dropped": 0,
        "phone_phrases": 0,
        "fallback_phrases": 0,
        "pul_phrases": 0,
        "exact_control_phrases": 0,
        "pul_frames": 0,
        "control_anomalies": 0,
        "structural_fallbacks": 0,
        "nonpad_tokens": 0,
    }
    restored_metrics = dict(rank_state.get("metrics") or {})
    if resume is None:
        if set(restored_metrics) != set(metric_defaults) - {"style_l2", "ins_dropped"}:
            raise ValueError("V4IPH transition metrics have an unexpected schema")
        restored_metrics.update({"style_l2": 0.0, "ins_dropped": 0})
    elif set(restored_metrics) != set(metric_defaults):
        raise ValueError("V4IjPH resume metrics have an unexpected schema")

    accum = dict(restored_metrics)
    support_counts = state["p_support_counts"].to(device).long()
    p_distance_history = list(state["p_distance_history"])
    initial_p_weight = state["initial_p_weight"].to(device)
    initial_frozen_fingerprint = state["initial_frozen_fingerprint"]
    phase_a_init_audit = state["phase_a_init_audit"]
    pul_embedding_init = state["pul_embedding_init"]
    trainable_parameter_names = state["trainable_parameter_names"]
    transition_p_sha256 = tensor_sha256(source["model_state_dict"][MIDI_P_KEY])
    del state, resume, source

    seed_everything(args.seed + rank)
    train_sampler.set_epoch_and_offset(sampler_epoch, batch_offset)
    data_iter = iter(train_loader)
    restore_rng_state(rank_state["rng"], device)

    if args.smoke_assertions:
        assert_rank_parameter_fingerprints(raw_model, world_size, "initial", "V4IjPH-v2")
        assert_rank_parameter_fingerprints(adapter, world_size, "adapter-initial", "V4IjPH-v2")
    if is_main():
        print(
            f"[V4IjPH v2] step={global_step} target={run_target} "
            f"cursor=({sampler_epoch},{batch_offset}) train={len(train_dataset)} "
            f"eval={len(eval_dataset)} effective_batch={world_size * args.grad_accum}",
            flush=True,
        )

    objective = V4IjPHObjective(
        raw_model=raw_model,
        adapter=adapter,
        vae=vae,
        args=args,
        device=device,
        game_cache_to_model_tracks=game_cache_to_model_tracks,
    )

    @torch.no_grad()
    def run_eval(force_ins_drop: bool):
        rng_state = capture_rng_state(device)
        seed_everything(args.eval_seed)
        raw_model.eval()
        adapter.eval()
        totals = {"loss": 0.0, "flow_a": 0.0, "flow_b": 0.0, "cka": 0.0}
        count = 0
        for batch in eval_loader:
            loss, values = objective.compute_loss(
                batch, force_ins_drop=force_ins_drop
            )
            totals["loss"] += loss.item()
            for key in ("flow_a", "flow_b", "cka"):
                totals[key] += values[key]
            count += 1
        raw_model.train()
        adapter.train()
        restore_rng_state(rng_state, device)
        return {key: value / max(count, 1) for key, value in totals.items()}

    def current_metrics():
        return dict(accum)

    def save_checkpoint(path, run_state):
        local_state = {
            "rank": rank,
            "sampler_epoch": sampler_epoch,
            "batch_offset": batch_offset,
            "rng": capture_rng_state(device),
            "metrics": current_metrics(),
        }
        if world_size > 1:
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, local_state)
        else:
            gathered = [local_state]
        gathered.sort(key=lambda item: int(item["rank"]))
        if len(
            {
                (int(item["sampler_epoch"]), int(item["batch_offset"]))
                for item in gathered
            }
        ) != 1:
            raise RuntimeError("V4IjPH rank data cursors diverged")
        if is_main():
            payload = {
                "checkpoint_schema": V4IJPH_CHECKPOINT_SCHEMA,
                "run_state": run_state,
                "global_step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "ema_model_state_dict": ema.ema_model.state_dict(),
                "ema_step": ema.step.detach().cpu().clone(),
                "ema_initted": ema.initted.detach().cpu().clone(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "adapter_state_dict": adapter.state_dict(),
                "adapter_optimizer_state_dict": adapter_optimizer.state_dict(),
                "adapter_ema_model_state_dict": adapter_ema.ema_model.state_dict(),
                "adapter_ema_step": adapter_ema.step.detach().cpu().clone(),
                "adapter_ema_initted": adapter_ema.initted.detach().cpu().clone(),
                "adapter_sha256": state_dict_sha256(adapter),
                "adapter_ema_sha256": state_dict_sha256(adapter_ema.ema_model),
                "initial_adapter_sha256": initial_adapter_sha256,
                "v4ijph_training": training_metadata,
                "rank_states": gathered,
                "midi_p_schema": MIDI_P_V4PH_SCHEMA,
                "game_cache_schema": GAME_CACHE_SCHEMA,
                "p_support_counts": support_counts.detach().cpu().clone(),
                "p_distance_history": list(p_distance_history),
                "initial_p_weight": initial_p_weight.detach().cpu().clone(),
                "initial_frozen_fingerprint": initial_frozen_fingerprint,
                "phase_a_init_audit": phase_a_init_audit,
                "pul_embedding_init": pul_embedding_init,
                "trainable_parameter_names": trainable_parameter_names,
                "transition_p_sha256": transition_p_sha256,
                "args": vars(args),
            }
            atomic_torch_save(payload, path)
            print(f"[V4IjPH v2] saved step={global_step} state={run_state}: {path}")
        if world_size > 1:
            distributed_barrier()

    step_offset = global_step
    started = time.time()
    while global_step < run_target:
        update_step = global_step + 1
        adapter_lr = adapter_lr_for_global_update(update_step)
        adapter_optimizer.param_groups[0]["lr"] = adapter_lr
        optimizer.zero_grad(set_to_none=True)
        adapter_optimizer.zero_grad(set_to_none=True)
        step_support = torch.zeros(257, dtype=torch.long, device=device)

        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                sampler_epoch += 1
                batch_offset = 0
                train_sampler.set_epoch_and_offset(sampler_epoch, batch_offset)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch_offset += 1
            loss, values = objective.compute_loss(batch)
            (loss / args.grad_accum).backward()
            for key in (
                "flow_a",
                "flow_b",
                "cka",
                "style_l2",
                "ins_dropped",
                "phone_phrases",
                "fallback_phrases",
                "pul_phrases",
                "exact_control_phrases",
                "pul_frames",
                "nonpad_tokens",
            ):
                accum[key] += values[key]
            accum["control_anomalies"] += values["sample_control_anomaly"]
            accum["structural_fallbacks"] += values[
                "sample_structural_fallback"
            ]
            step_support += values["p_class_counts"].to(device)

        sync_gradients(raw_model, world_size)
        sync_gradients(adapter, world_size)
        if world_size > 1:
            torch.distributed.all_reduce(step_support)
        support_counts += step_support
        parameters = list(raw_model.parameters()) + list(adapter.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        adapter_optimizer.step()
        scheduler.step()
        ema.update()
        adapter_ema.update()
        optimizer.zero_grad(set_to_none=True)
        adapter_optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % 100 == 0:
            torch.cuda.empty_cache()
        if global_step % args.log_every == 0:
            keys = list(metric_defaults)
            values = torch.tensor([float(accum[key]) for key in keys], device=device)
            if world_size > 1:
                torch.distributed.all_reduce(values)
            if is_main():
                denominator = args.log_every * args.grad_accum * world_size
                elapsed = time.time() - started
                steps_done = global_step - step_offset
                seconds_per_step = elapsed / max(steps_done, 1)
                remaining = run_target - global_step
                eta_seconds = remaining * seconds_per_step
                print(
                    f"[{global_step:6d}/{args.max_steps} "
                    f"{100 * global_step / args.max_steps:5.1f}% "
                    f"elapsed={elapsed / 3600:.2f}h eta={eta_seconds / 3600:.2f}h "
                    f"{seconds_per_step:.2f}s/step] "
                    f"FlowB={values[keys.index('flow_b')].item() / denominator:.4f} "
                    f"CKA={values[keys.index('cka')].item() / denominator:.4f} "
                    f"StyleL2={values[keys.index('style_l2')].item() / denominator:.4f} "
                    f"INSDrop={values[keys.index('ins_dropped')].item() / denominator:.3f} "
                    f"DiTLR={optimizer.param_groups[0]['lr']:.3e} "
                    f"AdapterLR={adapter_lr:.3e}",
                    flush=True,
                )
            accum = dict(metric_defaults)

        if global_step % args.eval_every == 0:
            if is_main():
                for name, drop in (("ins", False), ("null", True)):
                    result = run_eval(drop)
                    print(
                        f"[V4IjPH v2 eval/{name}] step={global_step} "
                        f"Loss={result['loss']:.4f} FlowB={result['flow_b']:.4f} "
                        f"CKA={result['cka']:.4f}",
                        flush=True,
                    )
            if world_size > 1:
                distributed_barrier()

        if global_step % args.save_every == 0 and global_step < run_target:
            save_checkpoint(
                os.path.join(args.output_dir, f"step_{global_step:06d}.pt"),
                "running",
            )

    if args.smoke_assertions:
        assert_rank_parameter_fingerprints(raw_model, world_size, "final", "V4IjPH-v2")
        assert_rank_parameter_fingerprints(adapter, world_size, "adapter-final", "V4IjPH-v2")
    final_state = "complete" if global_step == args.max_steps else "stopped"
    final_path = os.path.join(args.output_dir, f"step_{global_step:06d}_final.pt")
    save_checkpoint(final_path, final_state)
    if is_main():
        print(f"[V4IjPH v2] finished at step {global_step}: {final_path}")

    train_cache.close()
    eval_cache.close()
    if torch.distributed.is_initialized():
        distributed_barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
