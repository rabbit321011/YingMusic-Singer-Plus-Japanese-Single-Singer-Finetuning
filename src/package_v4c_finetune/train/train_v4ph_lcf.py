import argparse
import hashlib
import itertools
import json
import os
import pathlib
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from ema_pytorch import EMA
from omegaconf import OmegaConf

FRAME_RATE = 44100 / 2048
SEP_TOKEN = 365
PUL_TOKEN = 366
H_CHECKPOINT_SCHEMA = "h_training_checkpoint_v1"
H_PUL_CHECKPOINT_SCHEMA = "h_pul_training_checkpoint_v1"
H_PUL_G_CHECKPOINT_SCHEMA = "h_pul_g_training_checkpoint_v1"
V4PH_CHECKPOINT_SCHEMA = "v4ph_training_checkpoint_v1"
V4PH_LCF_CHECKPOINT_SCHEMA = "v4ph_lcf_training_checkpoint_v1"
PHASE_A_ADJUDICATION_SCHEMA = "v4ph_phase_a_distance_adjudication_v2"

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src" / "YingMusicSinger").is_dir():
    YING_REPO = PROJECT_DIR
if str(YING_REPO) not in sys.path:
    sys.path.insert(0, str(YING_REPO))
for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if (candidate / "h_alignment").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from h_alignment.placement import render_h_pul_placements, render_paired_placements


def checkpoint_schema_for_mode(placement_mode, warmstart_checkpoint=None):
    if placement_mode == "phone_pul" and warmstart_checkpoint is None:
        return V4PH_CHECKPOINT_SCHEMA
    if warmstart_checkpoint:
        if placement_mode != "phone_pul":
            raise ValueError("H warm-start adaptation requires phone_pul placement")
        return H_PUL_G_CHECKPOINT_SCHEMA
    return (
        H_PUL_CHECKPOINT_SCHEMA
        if placement_mode == "phone_pul"
        else H_CHECKPOINT_SCHEMA
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value):
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_state_dict_checked(
    module,
    state_dict,
    label,
    expected_missing=(),
    expected_unexpected=(),
):
    incompatible = module.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != set(expected_missing) or unexpected != set(expected_unexpected):
        raise RuntimeError(
            f"{label} incompatible state: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}; "
            f"expected_missing={sorted(expected_missing)} "
            f"expected_unexpected={sorted(expected_unexpected)}"
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class ResumableDistributedSampler(DistributedSampler):
    """Deterministic distributed sampler that can start inside an epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def set_epoch_and_offset(self, epoch, batch_offset):
        epoch = int(epoch)
        batch_offset = int(batch_offset)
        if not 0 <= batch_offset <= self.num_samples:
            raise ValueError(
                f"sampler batch offset {batch_offset} is outside "
                f"[0, {self.num_samples}]"
            )
        super().set_epoch(epoch)
        self.start_index = batch_offset

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.start_index, None)

    def __len__(self):
        return self.num_samples - self.start_index


def capture_rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng_state(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device)


def atomic_torch_save(payload, path):
    temp_path = f"{path}.tmp"
    with open(temp_path, "wb") as file:
        torch.save(payload, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


@torch.no_grad()
def parameter_sha256(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value = parameter.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def frozen_parameter_sha256(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            continue
        value = parameter.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def assert_rank_parameter_fingerprints(module, world_size, label):
    local_fingerprint = parameter_sha256(module)
    fingerprints = [None] * world_size
    if world_size > 1:
        torch.distributed.all_gather_object(fingerprints, local_fingerprint)
    else:
        fingerprints[0] = local_fingerprint
    if len(set(fingerprints)) != 1:
        raise RuntimeError(f"rank parameter mismatch at {label}: {fingerprints}")
    if is_main():
        print(f"[Train V4PH] Rank parameter SHA256 ({label}): {local_fingerprint}")
    return local_fingerprint


def build_training_metadata(args, world_size):
    placement_module = sys.modules[render_paired_placements.__module__]
    fingerprint_paths = {
        "training_code": pathlib.Path(__file__).resolve(),
        "placement_code": pathlib.Path(placement_module.__file__).resolve(),
        "model_config": pathlib.Path(args.config),
        "base_checkpoint": pathlib.Path(args.ckpt_path),
        "vae_config": pathlib.Path(args.vae_config),
        "vae_checkpoint": pathlib.Path(args.vae_ckpt),
        "game_cache_manifest": pathlib.Path(args.game_cache_manifest),
    }
    if args.warmstart_checkpoint:
        fingerprint_paths["warmstart_checkpoint"] = pathlib.Path(
            args.warmstart_checkpoint
        )
    if args.phase_a_checkpoint:
        fingerprint_paths["phase_a_checkpoint"] = pathlib.Path(
            args.phase_a_checkpoint
        )
        fingerprint_paths["phase_a_adjudication"] = pathlib.Path(
            args.phase_a_adjudication
        )
    file_sha256 = {}
    for label, path in fingerprint_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        file_sha256[label] = sha256_file(path)

    if args.warmstart_checkpoint:
        actual_warmstart_sha = file_sha256["warmstart_checkpoint"]
        if actual_warmstart_sha != args.warmstart_expected_sha256:
            raise ValueError(
                "warm-start checkpoint SHA256 mismatch: "
                f"{actual_warmstart_sha} != {args.warmstart_expected_sha256}"
            )
    if args.phase_a_checkpoint:
        actual_phase_a_sha = file_sha256["phase_a_checkpoint"]
        if actual_phase_a_sha != args.phase_a_checkpoint_sha256:
            raise ValueError(
                "phase-A checkpoint SHA256 mismatch: "
                f"{actual_phase_a_sha} != {args.phase_a_checkpoint_sha256}"
            )
        actual_adjudication_sha = file_sha256["phase_a_adjudication"]
        if actual_adjudication_sha != args.phase_a_adjudication_sha256:
            raise ValueError(
                "phase-A adjudication SHA256 mismatch: "
                f"{actual_adjudication_sha} != {args.phase_a_adjudication_sha256}"
            )

    metadata = {
        "schema": (
            V4PH_LCF_CHECKPOINT_SCHEMA
            if args.lcf_enabled
            else checkpoint_schema_for_mode(
                args.placement_mode, args.warmstart_checkpoint
            )
        ),
        "placement_mode": args.placement_mode,
        "h_config_fingerprint": args.h_config_fingerprint,
        "train_manifest_sha256": args.train_manifest_sha256,
        "eval_manifest_sha256": args.eval_manifest_sha256,
        "world_size": int(world_size),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "schedule_profile": args.schedule_profile,
        "warmup_steps": args.warmup_steps,
        "hold_steps": args.hold_steps,
        "max_steps": args.max_steps,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "log_every": args.log_every,
        "max_duration": args.max_duration,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "overfit": args.overfit,
        "overfit_n": args.overfit_n,
        "cka_weight": args.cka_weight,
        "t_shift": args.t_shift,
        "drop_text": args.drop_text,
        "flow_b_weight": args.flow_b_weight,
        "lcf": (
            {
                "paper": "arXiv:2509.20952v1",
                "ymsp_low_noise_policy": "t >= threshold",
                "threshold": args.lcf_threshold,
                "feature_layer_index_zero_based": args.lcf_layer,
                "feature_region": "B_only",
                "feature_pool": "temporal_mean_then_l2_normalize",
                "positive": (
                    "same_x1_noise_and_conditions_at_threshold_detached"
                ),
                "negative": (
                    "current_global_microbatch_queries_detached_"
                    "including_self_copy"
                ),
                "contrastive_batch_size": int(
                    world_size * args.batch_size
                ),
                "temperature": args.lcf_tau,
                "weight": args.lcf_weight,
                "low_noise_flow_policy": "zero_flow_A_and_flow_B",
                "flow_normalization": "global_mean_over_non_lcf_samples",
                "low_noise_cka_policy": "retain",
                "eval_policy": "frozen_standard_flow_plus_cka_no_lcf",
            }
            if args.lcf_enabled
            else None
        ),
        "phase": args.phase,
        "midi_teacher": "GAME medium K4 offline cache",
        "midi_p_init": (
            "phase_a_p500_with_kernel_fill"
            if args.phase == "joint"
            else "matched_random"
        ),
        "midi_fuzz_disturb": False,
        "h_pul": (
            {
                "pul_token_id": PUL_TOKEN,
                "sep_token_id": SEP_TOKEN,
                "sep_policy": "next_runtime_control_anchor_minus_one",
                "final_sep_policy": "last_dense_text_frame",
                "pul_policy": "repeat_after_packed_lyrics_until_sep",
                "hard_fallback_policy": "whole_sample_exact_control",
                "embedding_init": "copy_input_zero_row_once_on_fresh_base",
                "synthetic_pul_probability": 0.0,
            }
            if args.placement_mode == "phone_pul"
            else None
        ),
        "file_sha256": file_sha256,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "allow_tf32": False,
        },
    }
    if args.warmstart_checkpoint:
        metadata["warmstart"] = {
            "checkpoint": str(pathlib.Path(args.warmstart_checkpoint).resolve()),
            "checkpoint_sha256": file_sha256["warmstart_checkpoint"],
            "expected_source_step": args.warmstart_expected_step,
            "weight_source": "ema_model_state_dict",
            "raw_model_policy": "initialize_from_source_ema",
            "ema_policy": "continue_source_ema_weights_and_counters",
            "optimizer_policy": "fresh",
            "scheduler_policy": "fresh_step_zero",
            "data_cursor_policy": "fresh_step_zero",
        }
    if args.phase_a_checkpoint:
        metadata["phase_a_init"] = {
            "checkpoint": str(pathlib.Path(args.phase_a_checkpoint).resolve()),
            "checkpoint_sha256": file_sha256["phase_a_checkpoint"],
            "adjudication": str(pathlib.Path(args.phase_a_adjudication).resolve()),
            "adjudication_sha256": file_sha256["phase_a_adjudication"],
            "expected_source_step": 500,
            "weight_source": "model_state_dict",
            "unsupported_pitch_policy": "fill_from_structured_kernel",
            "rest_policy": "preserve_phase_a_learned_row",
            "pad_policy": "fixed_zero",
            "optimizer_policy": "fresh",
            "scheduler_policy": "fresh_step_zero",
            "ema_policy": "fresh_from_transition_weights",
            "data_cursor_policy": "fresh_step_zero",
        }
    return metadata


def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return local_rank, torch.distributed.get_world_size()


def is_main():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def distributed_barrier():
    if not torch.distributed.is_initialized():
        return
    if torch.distributed.get_backend() == "nccl":
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    else:
        torch.distributed.barrier()


def pool_lcf_feature(hidden, ref_len):
    hidden_b = hidden[:, ref_len:, :]
    if hidden_b.shape[1] <= 0:
        raise ValueError("LCF requires a non-empty B region")
    pooled = hidden_b.mean(dim=1)
    return F.normalize(pooled, p=2, dim=-1, eps=1e-8)


def lcf_per_anchor_objective(query, positive, negative_bank, tau):
    if tau <= 0:
        raise ValueError("LCF temperature must be positive")
    positive_distance = (query - positive.detach()).square().sum(dim=-1)
    negative_distances = (
        query[:, None, :] - negative_bank.detach()[None, :, :]
    ).square().sum(dim=-1)
    per_anchor = (
        positive_distance / tau
        + torch.logsumexp(-negative_distances / tau, dim=1)
    )
    return per_anchor, positive_distance, negative_distances


def sync_gradients(model, world_size, bucket_bytes=25 * 1024 * 1024):
    """Synchronize gradients because the custom loss bypasses Singer.forward."""
    if world_size <= 1:
        return

    def flush(bucket):
        if not bucket:
            return
        flat = torch._utils._flatten_dense_tensors([p.grad for p in bucket])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat.div_(world_size)
        synced = torch._utils._unflatten_dense_tensors(flat, [p.grad for p in bucket])
        for parameter, gradient in zip(bucket, synced):
            parameter.grad.copy_(gradient)

    bucket = []
    bucket_size = 0
    bucket_dtype = None
    for parameter in (p for p in model.parameters() if p.requires_grad):
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradient_bytes = parameter.grad.numel() * parameter.grad.element_size()
        if bucket and (
            parameter.grad.dtype != bucket_dtype
            or bucket_size + gradient_bytes > bucket_bytes
        ):
            flush(bucket)
            bucket = []
            bucket_size = 0
        bucket.append(parameter)
        bucket_size += gradient_bytes
        bucket_dtype = parameter.grad.dtype
    flush(bucket)


def collate_svs(batch):
    return {
        "wav": [item["wav"] for item in batch],
        "sr": [item["sr"] for item in batch],
        "phrases": [item["phrases"] for item in batch],
        "tier": [item["tier"] for item in batch],
        "duration": [item["duration"] for item in batch],
        "h_candidates": [item["h_candidates"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "game_cache": [item["game_cache"] for item in batch],
    }


class HDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        expected_sha256,
        expected_config_fingerprint,
        max_duration_sec=30.0,
        game_cache_manifest=None,
        audio_root_override=None,
    ):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(manifest_path)
        actual_sha256 = sha256_file(manifest_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"manifest SHA256 mismatch: {actual_sha256} != {expected_sha256}"
            )
        with open(manifest_path, encoding="utf-8") as file:
            self.records = json.load(file)
        for index, item in enumerate(self.records):
            if item.get("HConfigFingerprint") != expected_config_fingerprint:
                raise ValueError(f"H config fingerprint mismatch at record {index}")
            if len(item.get("Phrases", [])) != len(
                item["HAlignment"]["phrase_candidates"]
            ):
                raise ValueError(f"H phrase/candidate mismatch at record {index}")
        self.max_duration_sec = max_duration_sec
        self.manifest_sha256 = actual_sha256
        if game_cache_manifest is None:
            raise ValueError("V4PH requires a GAME cache manifest")
        from src.YingMusicSinger.melody.game_cache_v4ph import (
            load_game_cache_manifest,
        )
        cache_manifest_path = pathlib.Path(game_cache_manifest).resolve()
        cache_manifest = load_game_cache_manifest(cache_manifest_path)
        self.cache_root = cache_manifest_path.parent
        self.cache_by_source = {
            item["source_path"]: item["cache"] for item in cache_manifest["entries"]
        }
        self.cache_by_name = {}
        for item in cache_manifest["entries"]:
            name = pathlib.Path(item["source_path"]).name
            if name in self.cache_by_name:
                raise ValueError(f"Duplicate GAME cache basename: {name}")
            self.cache_by_name[name] = item["cache"]
        self.audio_root_override = (
            pathlib.Path(audio_root_override).resolve()
            if audio_root_override
            else None
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from src.YingMusicSinger.melody.game_cache_v4ph import load_game_cache

        rec = self.records[idx]
        audio_path = pathlib.Path(rec["Path"])
        if not audio_path.exists() and self.audio_root_override is not None:
            audio_path = self.audio_root_override / audio_path.name
        wav, sr = torchaudio.load(audio_path)
        if sr != 44100:
            wav = torchaudio.functional.resample(wav, sr, 44100)
            sr = 44100
        max_samples = int(self.max_duration_sec * sr)
        if wav.shape[-1] > max_samples:
            wav = wav[:, :max_samples]
        cache_relative = self.cache_by_source.get(rec["Path"])
        if cache_relative is None:
            cache_relative = self.cache_by_name.get(pathlib.Path(rec["Path"]).name)
        if cache_relative is None:
            raise KeyError(f"No GAME cache for {rec['Path']}")
        return {
            "wav": wav,
            "sr": sr,
            "phrases": rec["Phrases"],
            "tier": "L1",
            "duration": rec["Duration"],
            "h_candidates": rec["HAlignment"]["phrase_candidates"],
            "sample_id": rec["SampleId"],
            "game_cache": load_game_cache(self.cache_root / cache_relative),
        }


def compute_cka_loss_from_hidden(hidden_states, midi_latent, cka_layers=None):
    from src.YingMusicSinger.utils.common import cka_loss, calculate_similarity_matrix_with_mask
    if cka_layers is None:
        cka_layers = [-1, -2, -3]
    total_cka = 0.0
    count = 0
    for layer_idx in cka_layers:
        h = hidden_states[layer_idx]
        B, T_, D_ = h.shape
        b1, t1, d1 = midi_latent.shape
        common_t = min(T_, t1)
        h = h[:, :common_t, :]
        m = midi_latent[:, :common_t, :]
        sim_h = calculate_similarity_matrix_with_mask(h)
        sim_m = calculate_similarity_matrix_with_mask(m)
        total_cka += cka_loss(sim_h, sim_m)
        count += 1
    return total_cka / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml")
    parser.add_argument("--ckpt_path", default="ckpts/YingMusicSinger_model.pt")
    parser.add_argument("--vae_config", default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json")
    parser.add_argument("--vae_ckpt", default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt")
    parser.add_argument("--midi_ckpt", default="ckpts/model_ckpt_steps_100000_simplified.ckpt")
    parser.add_argument("--game_cache_manifest", required=True)
    parser.add_argument("--phase", choices=("p_only", "joint"), default="p_only")
    parser.add_argument("--audio_root_override", default=None)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--eval_manifest", required=True)
    parser.add_argument("--train_manifest_sha256", required=True)
    parser.add_argument("--eval_manifest_sha256", required=True)
    parser.add_argument("--h_config_fingerprint", required=True)
    parser.add_argument(
        "--placement_mode",
        choices=("sentence", "phone", "phone_pul"),
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--warmstart_checkpoint", default=None)
    parser.add_argument("--warmstart_expected_sha256", default=None)
    parser.add_argument("--warmstart_expected_step", type=int, default=None)
    parser.add_argument("--phase_a_checkpoint", default=None)
    parser.add_argument("--phase_a_checkpoint_sha256", default=None)
    parser.add_argument("--phase_a_adjudication", default=None)
    parser.add_argument("--phase_a_adjudication_sha256", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument(
        "--schedule_profile",
        choices=("baseline", "highlr_24k"),
        default="baseline",
        help="Audited LR schedule profile recorded in checkpoint metadata",
    )
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--hold_steps", type=int, default=12000,
                        help="Steps to hold peak LR before cosine decay")
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit_n", type=int, default=20)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_seed", type=int, default=1042)
    parser.add_argument("--cka_weight", type=float, default=0.7)
    parser.add_argument("--t_shift", type=float, default=0.5)
    parser.add_argument("--drop_text", type=float, default=0.15,
                        help="Text CFG dropout probability")
    parser.add_argument("--flow_b_weight", type=float, default=2.0,
                        help="Weight multiplier for B區 flow loss")
    parser.add_argument("--lcf_enabled", action="store_true")
    parser.add_argument("--lcf_threshold", type=float, default=0.95)
    parser.add_argument("--lcf_layer", type=int, default=15)
    parser.add_argument("--lcf_tau", type=float, default=0.5)
    parser.add_argument("--lcf_weight", type=float, default=1.0)
    parser.add_argument("--expected_world_size", type=int, default=4)
    parser.add_argument("--smoke_assertions", action="store_true")
    parser.add_argument(
        "--stop_after_step",
        type=int,
        default=None,
        help="Exit cleanly at this absolute step without changing the LR schedule.",
    )
    args = parser.parse_args()
    expected_checkpoint_schema = (
        V4PH_LCF_CHECKPOINT_SCHEMA
        if args.lcf_enabled
        else checkpoint_schema_for_mode(
            args.placement_mode, args.warmstart_checkpoint
        )
    )

    if args.batch_size != 1:
        raise ValueError("H training currently requires batch_size=1")
    if args.placement_mode != "phone_pul":
        raise ValueError("V4PH requires H phone_pul placement")
    if args.warmstart_checkpoint:
        raise ValueError("V4PH does not inherit a V4H/V4Hg warm-start checkpoint")
    if args.num_workers != 0:
        raise ValueError("exact H resume requires num_workers=0")
    if args.grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if args.lcf_enabled:
        if args.phase != "joint":
            raise ValueError("LCF is frozen for the V4PH joint phase only")
        frozen_lcf = {
            "lcf_threshold": 0.95,
            "lcf_layer": 15,
            "lcf_tau": 0.5,
            "lcf_weight": 1.0,
        }
        for key, expected in frozen_lcf.items():
            actual = getattr(args, key)
            if actual != expected:
                raise ValueError(
                    f"frozen PH-HighLR-LCF contract requires "
                    f"{key}={expected}, got {actual}"
                )
        if args.expected_world_size != 4:
            raise ValueError("LCF contrastive negatives require frozen world_size=4")
    if args.phase == "p_only":
        if args.schedule_profile != "baseline":
            raise ValueError("V4PH p_only phase requires schedule_profile=baseline")
        if args.max_steps != 500:
            raise ValueError("V4PH p_only phase requires max_steps=500")
        if args.lr != 1e-4:
            raise ValueError("V4PH p_only phase requires lr=1e-4")
        if args.drop_text != 0:
            raise ValueError("V4PH p_only phase requires drop_text=0")
        if args.warmup_steps != 0 or args.hold_steps != 500:
            raise ValueError("V4PH p_only phase requires warmup=0 and hold=500")
        if any(
            value is not None
            for value in (
                args.phase_a_checkpoint,
                args.phase_a_checkpoint_sha256,
                args.phase_a_adjudication,
                args.phase_a_adjudication_sha256,
            )
        ):
            raise ValueError("V4PH p_only phase cannot consume a phase-A transition")
    else:
        if args.max_steps != 30000:
            raise ValueError("V4PH joint phase requires max_steps=30000")
        if args.lr != 1.4e-5:
            raise ValueError("V4PH joint phase requires lr=1.4e-5")
        expected_hold_steps = {
            "baseline": 12000,
            "highlr_24k": 23500,
        }[args.schedule_profile]
        if args.warmup_steps != 500 or args.hold_steps != expected_hold_steps:
            raise ValueError(
                "V4PH joint phase schedule mismatch: "
                f"profile={args.schedule_profile} requires warmup=500 "
                f"and hold={expected_hold_steps}"
            )
        if args.drop_text != 0.15:
            raise ValueError("V4PH joint phase requires drop_text=0.15")
        if args.flow_b_weight != 2.0 or args.cka_weight != 0.7:
            raise ValueError("V4PH joint phase requires FlowB=2.0 and CKA=0.7")
        transition_values = (
            args.phase_a_checkpoint,
            args.phase_a_checkpoint_sha256,
            args.phase_a_adjudication,
            args.phase_a_adjudication_sha256,
        )
        if any(value is None for value in transition_values):
            raise ValueError("V4PH joint phase requires the complete phase-A contract")
        for label, expected_sha in (
            ("phase_a_checkpoint", args.phase_a_checkpoint_sha256),
            ("phase_a_adjudication", args.phase_a_adjudication_sha256),
        ):
            if len(expected_sha) != 64 or any(
                character not in "0123456789abcdef" for character in expected_sha
            ):
                raise ValueError(f"{label}_sha256 must be lowercase SHA256")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    for interval_name in ("save_every", "eval_every", "log_every"):
        if getattr(args, interval_name) <= 0:
            raise ValueError(f"{interval_name} must be positive")
    if args.stop_after_step is not None and not (
        0 < args.stop_after_step <= args.max_steps
    ):
        raise ValueError("stop_after_step must be in [1, max_steps]")
    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(args.resume)
    if args.warmstart_checkpoint:
        if not os.path.isfile(args.warmstart_checkpoint):
            raise FileNotFoundError(args.warmstart_checkpoint)
        if args.placement_mode != "phone_pul":
            raise ValueError("warm-start adaptation requires phone_pul placement")
        if args.warmstart_expected_step is None or args.warmstart_expected_step <= 0:
            raise ValueError("warmstart_expected_step must be positive")
        expected_sha = args.warmstart_expected_sha256 or ""
        if len(expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha
        ):
            raise ValueError("warmstart_expected_sha256 must be lowercase SHA256")
    elif args.warmstart_expected_sha256 or args.warmstart_expected_step is not None:
        raise ValueError("warm-start expectations require warmstart_checkpoint")
    if args.phase_a_checkpoint:
        if not os.path.isfile(args.phase_a_checkpoint):
            raise FileNotFoundError(args.phase_a_checkpoint)
        if not os.path.isfile(args.phase_a_adjudication):
            raise FileNotFoundError(args.phase_a_adjudication)
    if not torch.cuda.is_available():
        raise RuntimeError("H training requires CUDA")

    if "LOCAL_RANK" in os.environ:
        local_rank, world_size = setup_ddp()
    else:
        local_rank, world_size = 0, 1
        torch.cuda.set_device(0)

    if world_size != args.expected_world_size:
        raise ValueError(
            f"world_size mismatch: {world_size} != {args.expected_world_size}"
        )

    rank = torch.distributed.get_rank() if world_size > 1 else 0
    # Model construction must be identical across ranks and between the two
    # placement modes. Per-rank training RNG is reset after all setup.
    seed_everything(args.seed)
    device = torch.device(f"cuda:{local_rank}")

    if is_main():
        print(f"[Train V4PH] World={world_size} Device={device}")
        print(f"[Train V4PH] Args: {args}")

    metadata_box = [build_training_metadata(args, world_size) if is_main() else None]
    if world_size > 1:
        torch.distributed.broadcast_object_list(metadata_box, src=0)
    training_metadata = metadata_box[0]

    cfg = OmegaConf.load(args.config)

    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.melody.midi_p_v4ph import (
        MIDI_P_V4PH_SCHEMA,
        V4PHMIDIEmbedding,
        fill_unsupported_pitch_rows,
    )
    from src.YingMusicSinger.melody.game_cache_v4ph import GAME_CACHE_SCHEMA
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer

    if is_main():
        print("[Train V4PH] Building DiT...")
    dit = DiT(**cfg.model.arch, text_num_embeds=cfg.datasets_cfg.text_num_embeds,
              mel_dim=cfg.model.mel_spec.n_mel_channels,
              long_skip_connection=True)

    singer_model = Singer(
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
    singer_model.midi_p_v4ph = V4PHMIDIEmbedding(seed=args.seed)

    resume_global_step = 0
    resume_ckpt = None
    resume_rank_state = None
    warmstart_ckpt = None
    phase_a_ckpt = None
    phase_a_init_audit = None
    if args.resume:
        if is_main():
            print(f"[Train V4PH] Resuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_ckpt.get("checkpoint_schema") != expected_checkpoint_schema:
            raise ValueError(
                f"resume checkpoint schema mismatch: "
                f"{resume_ckpt.get('checkpoint_schema')} != "
                f"{expected_checkpoint_schema}"
            )
        metadata = resume_ckpt.get("v4ph_training")
        if metadata != training_metadata:
            raise ValueError(
                f"resume V4PH metadata mismatch: {metadata} != {training_metadata}"
            )
        required_keys = {
            "model_state_dict",
            "ema_model_state_dict",
            "ema_step",
            "ema_initted",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "global_step",
            "rank_states",
            "p_support_counts",
            "p_distance_history",
            "initial_p_weight",
            "initial_frozen_fingerprint",
        }
        if args.phase == "joint":
            required_keys.update(("phase_a_init_audit", "trainable_parameter_names"))
        missing_keys = sorted(required_keys - resume_ckpt.keys())
        if missing_keys:
            raise ValueError(f"resume checkpoint missing keys: {missing_keys}")
        if resume_ckpt.get("midi_p_schema") != MIDI_P_V4PH_SCHEMA:
            raise ValueError("resume MIDI_P schema mismatch")
        if resume_ckpt.get("game_cache_schema") != GAME_CACHE_SCHEMA:
            raise ValueError("resume GAME cache schema mismatch")
        rank_states = resume_ckpt["rank_states"]
        if len(rank_states) != world_size:
            raise ValueError(
                f"resume rank state count {len(rank_states)} != {world_size}"
            )
        rank_state_by_rank = {int(item["rank"]): item for item in rank_states}
        if sorted(rank_state_by_rank) != list(range(world_size)):
            raise ValueError("resume checkpoint rank IDs are incomplete or duplicated")
        resume_rank_state = rank_state_by_rank[rank]
        sd = resume_ckpt["model_state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        load_state_dict_checked(singer_model, sd, "resume model")
        resume_global_step = int(resume_ckpt["global_step"])
        if not 0 <= resume_global_step <= args.max_steps:
            raise ValueError(
                f"resume global step {resume_global_step} is outside the run"
            )
        phase_a_init_audit = resume_ckpt.get("phase_a_init_audit")
    elif args.warmstart_checkpoint:
        if is_main():
            print(f"[Train V4PH] Warm-starting from EMA: {args.warmstart_checkpoint}")
        warmstart_ckpt = torch.load(
            args.warmstart_checkpoint, map_location="cpu", weights_only=False
        )
        if warmstart_ckpt.get("checkpoint_schema") != H_PUL_CHECKPOINT_SCHEMA:
            raise ValueError(
                "warm-start source schema mismatch: "
                f"{warmstart_ckpt.get('checkpoint_schema')} != "
                f"{H_PUL_CHECKPOINT_SCHEMA}"
            )
        if warmstart_ckpt.get("run_state") != "complete":
            raise ValueError("warm-start source is not a complete checkpoint")
        source_step = int(warmstart_ckpt.get("global_step", -1))
        if source_step != args.warmstart_expected_step:
            raise ValueError(
                f"warm-start source step {source_step} != "
                f"{args.warmstart_expected_step}"
            )
        source_metadata = warmstart_ckpt.get("h_training")
        if not isinstance(source_metadata, dict) or source_metadata.get(
            "placement_mode"
        ) != "phone_pul":
            raise ValueError("warm-start source is not a phone_pul H checkpoint")
        required_warmstart_keys = {
            "ema_model_state_dict",
            "ema_step",
            "ema_initted",
            "pul_embedding_init",
        }
        missing_keys = sorted(required_warmstart_keys - warmstart_ckpt.keys())
        if missing_keys:
            raise ValueError(
                f"warm-start checkpoint missing keys: {missing_keys}"
            )
        sd = {
            k.replace("ema_model.", ""): v
            for k, v in warmstart_ckpt["ema_model_state_dict"].items()
        }
        load_state_dict_checked(singer_model, sd, "warm-start source EMA")
    elif args.phase_a_checkpoint:
        if is_main():
            print(
                "[Train V4PH] Loading adjudicated phase-A P500: "
                f"{args.phase_a_checkpoint}"
            )
        phase_a_ckpt = torch.load(
            args.phase_a_checkpoint, map_location="cpu", weights_only=False
        )
        if phase_a_ckpt.get("checkpoint_schema") != V4PH_CHECKPOINT_SCHEMA:
            raise ValueError("phase-A checkpoint schema mismatch")
        if phase_a_ckpt.get("run_state") != "complete":
            raise ValueError("phase-A checkpoint is not complete")
        if int(phase_a_ckpt.get("global_step", -1)) != 500:
            raise ValueError("phase-A checkpoint is not step 500")
        source_metadata = phase_a_ckpt.get("v4ph_training")
        if not isinstance(source_metadata, dict) or source_metadata.get(
            "phase"
        ) != "p_only":
            raise ValueError("phase-A checkpoint metadata is not p_only")
        source_files = source_metadata.get("file_sha256", {})
        current_files = training_metadata["file_sha256"]
        for key in (
            "base_checkpoint",
            "model_config",
            "vae_config",
            "vae_checkpoint",
            "game_cache_manifest",
        ):
            if source_files.get(key) != current_files.get(key):
                raise ValueError(f"phase-A source file mismatch: {key}")
        if phase_a_ckpt.get("midi_p_schema") != MIDI_P_V4PH_SCHEMA:
            raise ValueError("phase-A MIDI_P schema mismatch")
        if phase_a_ckpt.get("game_cache_schema") != GAME_CACHE_SCHEMA:
            raise ValueError("phase-A GAME cache schema mismatch")

        with open(args.phase_a_adjudication, encoding="utf-8") as handle:
            adjudication = json.load(handle)
        if adjudication.get("schema") != PHASE_A_ADJUDICATION_SCHEMA:
            raise ValueError("phase-A adjudication schema mismatch")
        if adjudication.get("decision") != "random_p_pass":
            raise ValueError("phase-A adjudication did not pass random P")
        if adjudication.get("decision_policy") != (
            "projected_rmse_primary_support_metrics_diagnostic"
        ):
            raise ValueError("phase-A adjudication policy mismatch")
        if adjudication.get("step500_checkpoint_sha256") != (
            args.phase_a_checkpoint_sha256
        ):
            raise ValueError("phase-A adjudication checkpoint hash mismatch")
        if not adjudication.get("checks", {}).get("projected_rmse_decreased"):
            raise ValueError("phase-A primary projected distance did not improve")

        state_dict = {
            key.replace("module.", ""): value
            for key, value in phase_a_ckpt["model_state_dict"].items()
        }
        load_state_dict_checked(singer_model, state_dict, "phase-A source model")
        support = phase_a_ckpt["p_support_counts"].long()
        if support.shape != (257,) or int((support[:255] > 0).sum()) < 2:
            raise ValueError("phase-A support counts are invalid")
        p_weight = singer_model.midi_p_v4ph.embedding.weight
        source_p_sha = tensor_sha256(p_weight)
        rest_sha = tensor_sha256(p_weight[255])
        with torch.no_grad():
            filled_rows = fill_unsupported_pitch_rows(p_weight, support)
        if tensor_sha256(p_weight[255]) != rest_sha:
            raise AssertionError("phase-A transition modified the REST row")
        if not torch.equal(p_weight[256], torch.zeros_like(p_weight[256])):
            raise AssertionError("phase-A transition PAD row is not zero")
        phase_a_init_audit = {
            "source_checkpoint_sha256": args.phase_a_checkpoint_sha256,
            "adjudication_sha256": args.phase_a_adjudication_sha256,
            "source_step": 500,
            "source_weight": "model_state_dict",
            "supported_pitch_rows": int((support[:255] > 0).sum()),
            "unsupported_pitch_rows_filled": int(filled_rows),
            "source_p_sha256": source_p_sha,
            "transition_p_sha256": tensor_sha256(p_weight),
            "rest_sha256": rest_sha,
            "pad_policy": "fixed_zero",
        }
        if is_main():
            print(
                "[Train V4PH] Phase-A transition: "
                f"supported={phase_a_init_audit['supported_pitch_rows']} "
                f"filled={filled_rows} P={phase_a_init_audit['transition_p_sha256']}"
            )
    elif args.ckpt_path and os.path.exists(args.ckpt_path):
        if is_main():
            print(f"[Train V4PH] Loading base checkpoint: {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        if "ema_model_state_dict" in ckpt:
            sd = dict(ckpt["ema_model_state_dict"])
            ema_metadata_keys = {"initted", "step"}
            present_metadata = ema_metadata_keys & set(sd)
            if present_metadata != ema_metadata_keys:
                raise RuntimeError(
                    f"Official EMA metadata mismatch: {sorted(present_metadata)}"
                )
            for key in ema_metadata_keys:
                del sd[key]
            sd = {k.replace("ema_model.", ""): v for k, v in sd.items()}
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        load_state_dict_checked(
            singer_model,
            sd,
            "Official base",
            expected_missing=(
                "transformer.long_skip_connection.weight",
                "midi_p_v4ph.embedding.weight",
            ),
        )
    else:
        raise FileNotFoundError(args.ckpt_path)

    pul_embedding_init = None
    if args.placement_mode == "phone_pul":
        embedding = singer_model.transformer.text_embed_p.text_embed.weight
        source_row = 1
        destination_row = PUL_TOKEN + 1
        if destination_row >= embedding.shape[0]:
            raise ValueError(
                f"PUL input ID {PUL_TOKEN} maps to row {destination_row}, "
                f"outside embedding shape {tuple(embedding.shape)}"
            )
        if resume_ckpt is None and warmstart_ckpt is None and phase_a_ckpt is None:
            source_sha = tensor_sha256(embedding[source_row])
            destination_before_sha = tensor_sha256(embedding[destination_row])
            with torch.no_grad():
                embedding[destination_row].copy_(embedding[source_row])
            destination_after_sha = tensor_sha256(embedding[destination_row])
            if destination_after_sha != source_sha:
                raise AssertionError("PUL embedding copy is not bit-exact")
            pul_embedding_init = {
                "policy": "copy_input_zero_row_once_on_fresh_base",
                "source_input_id": 0,
                "source_row": source_row,
                "destination_input_id": PUL_TOKEN,
                "destination_row": destination_row,
                "source_sha256": source_sha,
                "destination_before_sha256": destination_before_sha,
                "destination_after_sha256": destination_after_sha,
            }
            if is_main():
                print(
                    "[Train V4PH] Initialized PUL embedding row "
                    f"{destination_row} from V row {source_row}: {source_sha}"
                )
        else:
            source_checkpoint = resume_ckpt or warmstart_ckpt or phase_a_ckpt
            pul_embedding_init = source_checkpoint.get("pul_embedding_init")
            if not isinstance(pul_embedding_init, dict):
                raise ValueError(
                    "resume/warm-start H-PUL checkpoint lacks embedding init audit"
                )
            expected_init = {
                "policy": "copy_input_zero_row_once_on_fresh_base",
                "source_input_id": 0,
                "source_row": source_row,
                "destination_input_id": PUL_TOKEN,
                "destination_row": destination_row,
            }
            for key, expected in expected_init.items():
                if pul_embedding_init.get(key) != expected:
                    raise ValueError(
                        f"resume/warm-start H-PUL embedding init {key} mismatch: "
                        f"{pul_embedding_init.get(key)} != {expected}"
                    )
            if (warmstart_ckpt is not None or phase_a_ckpt is not None) and is_main():
                v_sha = tensor_sha256(embedding[source_row])
                pul_sha = tensor_sha256(embedding[destination_row])
                print(
                    "[Train V4PH] Preserved transition V/PUL rows without copy: "
                    f"V={v_sha} PUL={pul_sha}"
                )

    singer_model = singer_model.to(device)
    if args.phase == "p_only":
        for parameter in singer_model.parameters():
            parameter.requires_grad = False
        singer_model.midi_p_v4ph.embedding.weight.requires_grad = True
    singer_model.train()
    raw_model = singer_model
    if resume_ckpt is None:
        initial_p_weight = raw_model.midi_p_v4ph.embedding.weight.detach().clone()
        initial_frozen_fingerprint = frozen_parameter_sha256(raw_model)
    else:
        initial_p_weight = resume_ckpt["initial_p_weight"].to(device)
        initial_frozen_fingerprint = resume_ckpt["initial_frozen_fingerprint"]
    if args.smoke_assertions:
        assert_rank_parameter_fingerprints(raw_model, world_size, "initial")

    if is_main():
        print("[Train V4PH] Loading VAE...")
    vae = StableAudioInfer(model_config_path=args.vae_config, model_ckpt_path=args.vae_ckpt)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    ema = EMA(singer_model, beta=cfg.ema_kwargs.beta,
              update_after_step=cfg.ema_kwargs.update_after_step,
              update_every=cfg.ema_kwargs.update_every)
    ema.to(device)
    if resume_ckpt is not None:
        ema_sd = {
            k.replace("ema_model.", ""): v
            for k, v in resume_ckpt["ema_model_state_dict"].items()
        }
        load_state_dict_checked(ema.ema_model, ema_sd, "resume EMA")
        ema.step.copy_(resume_ckpt["ema_step"])
        ema.initted.copy_(resume_ckpt["ema_initted"])
    elif warmstart_ckpt is not None:
        ema_sd = {
            k.replace("ema_model.", ""): v
            for k, v in warmstart_ckpt["ema_model_state_dict"].items()
        }
        load_state_dict_checked(ema.ema_model, ema_sd, "warm-start EMA")
        ema.step.copy_(warmstart_ckpt["ema_step"])
        ema.initted.copy_(warmstart_ckpt["ema_initted"])
        if is_main():
            print(
                "[Train V4PH] Warm-start EMA counters: "
                f"step={int(ema.step.item())} initted={bool(ema.initted.item())}"
            )

    # The loss calls transformer internals directly, so DDP.forward would be
    # bypassed. Gradients are explicitly synchronized before clipping below.
    raw_model = singer_model

    trainable_parameter_names = [
        name for name, parameter in raw_model.named_parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in raw_model.parameters() if parameter.requires_grad],
        lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=1e-2)

    # Warmup-hold-decay LR schedule shared by H-Control and H.
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        elif step < args.warmup_steps + args.hold_steps:
            return 1.0
        else:
            decay_steps = max(1, args.max_steps - args.warmup_steps - args.hold_steps)
            progress = (step - args.warmup_steps - args.hold_steps) / decay_steps
            return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if resume_ckpt is not None:
        # Construct the scheduler first so it cannot overwrite the restored LR.
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if int(scheduler.last_epoch) != resume_global_step:
            raise ValueError(
                f"scheduler step {scheduler.last_epoch} != global step "
                f"{resume_global_step}"
            )
        restored_lrs = [group["lr"] for group in optimizer.param_groups]
        if restored_lrs != list(scheduler.get_last_lr()):
            raise ValueError(
                f"optimizer/scheduler LR mismatch: {restored_lrs} != "
                f"{scheduler.get_last_lr()}"
            )
        if is_main():
            print("[Train V4PH] Restored optimizer and scheduler state")

    train_dataset = HDataset(
        args.train_manifest,
        args.train_manifest_sha256,
        args.h_config_fingerprint,
        args.max_duration,
        args.game_cache_manifest,
        args.audio_root_override,
    )
    if args.overfit:
        train_dataset.records = train_dataset.records[:args.overfit_n]
        if is_main():
            print(f"[Train V4PH] OVERFIT MODE: {len(train_dataset)} samples")
    train_sampler = ResumableDistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_svs,
                              pin_memory=True, drop_last=True)

    eval_dataset = HDataset(
        args.eval_manifest,
        args.eval_manifest_sha256,
        args.h_config_fingerprint,
        args.max_duration,
        args.game_cache_manifest,
        args.audio_root_override,
    )
    if args.overfit:
        eval_dataset.records = eval_dataset.records[: max(4, args.overfit_n // 5)]
    eval_loaders = {
        "paired": DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_svs,
            pin_memory=True,
            drop_last=True,
        )
    }
    eval_sizes = {"paired": len(eval_dataset)}

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main():
        eval_summary = ", ".join(f"{name}={size}" for name, size in eval_sizes.items()) or "none"
        print(f"[Train V4PH/{args.placement_mode}] Train: {len(train_dataset)} | Eval: {eval_summary}")
        print(f"[Train V4PH] Steps/epoch: {len(train_loader)} | Max steps: {args.max_steps}")
        print(f"[Train V4PH] Effective batch: {args.batch_size * world_size * args.grad_accum}")
        print(f"[Train V4PH] lr={args.lr} hold={args.hold_steps} drop_text={args.drop_text} "
              f"flow_b_w={args.flow_b_weight} cka={args.cka_weight}")
        if args.lcf_enabled:
            print(
                "[Train V4PH/LCF] "
                f"threshold={args.lcf_threshold} layer={args.lcf_layer} "
                f"tau={args.lcf_tau} weight={args.lcf_weight} "
                f"contrastive_batch={world_size * args.batch_size} "
                "pool=B_mean_l2 flow=off_at_low_noise cka=retained"
            )
        if args.resume:
            print(f"[Train V4PH] Resumed from step {resume_global_step}")
        print("=" * 60)

    def compute_ref_len(T, phrase_boundaries=None):
        five_sec_frames = int(5.0 * FRAME_RATE)
        ref_len = T * random.uniform(0.125, 0.33)
        ref_len = max(ref_len, five_sec_frames)
        ref_len = min(ref_len, int(T * 0.65))
        if phrase_boundaries is not None and len(phrase_boundaries) > 0:
            margin = int(2.5 * FRAME_RATE)
            candidates = [b for b in phrase_boundaries if b > 0 and abs(b - ref_len) <= margin]
            if candidates:
                ref_len = min(candidates, key=lambda b: abs(b - ref_len))
        return int(ref_len)

    def process_batch(
        wavs, srs, phrases_list, tiers, candidates_list, game_cache_list
    ):
        from src.YingMusicSinger.melody.game_cache_v4ph import (
            game_cache_to_model_tracks,
        )

        with torch.no_grad():
            w = wavs[0]
            sr = srs[0]
            phrases = phrases_list[0]
            tier = tiers[0]
            h_candidates = candidates_list[0]
            game_cache = game_cache_list[0]

            w_2d = w.unsqueeze(0) if w.dim() == 1 else w
            latent = vae.encode_audio(w_2d, in_sr=sr)
            full_latent = latent.squeeze(0).transpose(0, 1).unsqueeze(0)
            B, T, D = full_latent.shape

            if tier in ("L1", "L2"):
                phrase_boundaries = [int(p["start"] * FRAME_RATE) for p in phrases]
            else:
                phrase_boundaries = None
            ref_len = compute_ref_len(T, phrase_boundaries)

            cond = torch.zeros_like(full_latent)
            cond[:, :ref_len, :] = full_latent[:, :ref_len, :]

            game_tracks = game_cache_to_model_tracks(
                game_cache,
                num_samples=w_2d.shape[-1],
                target_len=T,
                sample_rate=sr,
            )
            p_classes = game_tracks["p_classes"].unsqueeze(0).to(device)
            midi_p = game_tracks["cka_probs"].unsqueeze(0).to(device)
            with torch.enable_grad():
                midi_full = raw_model.midi_p_v4ph(p_classes)
                midi = torch.cat(
                    [torch.zeros_like(midi_full[:, :ref_len]), midi_full[:, ref_len:]],
                    dim=1,
                )

            if args.placement_mode == "phone_pul":
                paired = render_h_pul_placements(
                    phrases,
                    h_candidates,
                    ref_len=ref_len,
                    total_frames=T,
                    sep_token_id=SEP_TOKEN,
                    pul_token_id=PUL_TOKEN,
                )
                selected = paired["phone_pul"]["text"]
            else:
                paired = render_paired_placements(
                    phrases,
                    h_candidates,
                    ref_len=ref_len,
                    total_frames=T,
                    sep_token_id=SEP_TOKEN,
                )
                selected = paired[
                    "control" if args.placement_mode == "sentence" else "phone"
                ]["text"]
            if args.smoke_assertions:
                if len(selected) != T:
                    raise AssertionError(
                        f"placement length {len(selected)} != latent length {T}"
                    )
                if args.placement_mode == "phone_pul":
                    if max(selected, default=0) > PUL_TOKEN:
                        raise AssertionError("H-PUL emitted an out-of-contract token")
                    if not paired["sample_control_anomaly"] and not paired[
                        "sample_structural_fallback"
                    ]:
                        expected_lyrics = [
                            int(token) for phrase in phrases for token in phrase["tokens"]
                        ]
                        rendered_lyrics = [
                            token
                            for token in selected
                            if token not in (0, SEP_TOKEN, PUL_TOKEN)
                        ]
                        if rendered_lyrics != expected_lyrics:
                            raise AssertionError("H-PUL smoke lyric invariant failed")
                else:
                    control_tokens = [
                        token for token in paired["control"]["text"] if token
                    ]
                    phone_tokens = [
                        token for token in paired["phone"]["text"] if token
                    ]
                    if control_tokens != phone_tokens:
                        raise AssertionError("smoke token invariant failed")
            aligned_text = torch.tensor(selected, dtype=torch.long).unsqueeze(0)
            aligned_text = aligned_text.to(device)

            if args.placement_mode == "phone_pul":
                placement_stats = {
                    "phone_phrases": paired["phone_phrase_count"],
                    "fallback_phrases": (
                        paired["pul_phrase_count"]
                        + paired["exact_control_phrase_count"]
                    ),
                    "pul_phrases": paired["pul_phrase_count"],
                    "exact_control_phrases": paired["exact_control_phrase_count"],
                    "pul_frames": paired["pul_frame_count"],
                    "sample_control_anomaly": int(paired["sample_control_anomaly"]),
                    "sample_structural_fallback": int(
                        paired["sample_structural_fallback"]
                    ),
                    "nonpad_tokens": sum(token != 0 for token in selected),
                }
            else:
                placement_stats = {
                    "phone_phrases": paired["phone_phrase_count"],
                    "fallback_phrases": paired["fallback_phrase_count"],
                    "pul_phrases": 0,
                    "exact_control_phrases": paired["fallback_phrase_count"],
                    "pul_frames": 0,
                    "sample_control_anomaly": int(paired["sample_control_anomaly"]),
                    "sample_structural_fallback": 0,
                    "nonpad_tokens": sum(token != 0 for token in selected),
                }

        return (
            full_latent,
            cond,
            midi,
            midi_p,
            aligned_text,
            ref_len,
            T,
            placement_stats,
            p_classes,
        )

    def run_dit(
        x_t,
        cond,
        text_tokens,
        t,
        midi,
        drop_audio,
        drop_text,
        drop_midi,
        capture_lcf=False,
    ):
        dit = raw_model.transformer
        B, seq_len = x_t.shape[0], x_t.shape[1]
        if t.ndim == 0:
            t = t.repeat(B)
        time_emb = dit.time_embed(t)

        x, _ = dit.get_input_embed(x_t, cond, text_tokens, midi,
                                    drop_audio_cond=drop_audio, drop_text=drop_text,
                                    drop_midi=drop_midi, cache=False)
        rope = dit.rotary_embed.forward_from_seq_len(seq_len)

        hidden_states = []
        lcf_hidden = None
        if dit.long_skip_connection is not None:
            residual = x
        for i, block in enumerate(dit.transformer_blocks):
            x = block(x, time_emb, mask=None, rope=rope)
            if capture_lcf and i == args.lcf_layer:
                lcf_hidden = x
            if i >= len(dit.transformer_blocks) - 3:
                hidden_states.append(x)
        if dit.long_skip_connection is not None:
            x = dit.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = dit.norm_out(x, time_emb)
        output = dit.proj_out(x)
        if capture_lcf and lcf_hidden is None:
            raise RuntimeError(f"LCF layer {args.lcf_layer} was not captured")
        return output, hidden_states, lcf_hidden

    def compute_lcf_loss(query, positive, active):
        detached_query = query.detach().contiguous()
        if world_size > 1:
            gathered = [torch.empty_like(detached_query) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, detached_query)
            negative_bank = torch.cat(gathered, dim=0)
        else:
            negative_bank = detached_query

        active_count = torch.tensor(
            [int(active)], dtype=torch.long, device=device
        )
        if world_size > 1:
            torch.distributed.all_reduce(active_count)
        global_anchor_count = int(active_count.item())

        if not active:
            zero = query.sum() * 0.0
            return zero, {
                "lcf": 0.0,
                "lcf_positive_distance": 0.0,
                "lcf_negative_distance": 0.0,
                "lcf_anchor": 0,
                "lcf_global_anchor_count": global_anchor_count,
            }
        if positive is None or global_anchor_count <= 0:
            raise RuntimeError("active LCF query lacks a positive or global anchor")

        per_anchor_values, positive_distance, negative_distances = (
            lcf_per_anchor_objective(
                query, positive, negative_bank, args.lcf_tau
            )
        )
        per_anchor = per_anchor_values.mean()

        # Manual gradient synchronization below averages ranks. This factor
        # makes the global objective the mean over active anchors, not ranks.
        scaled = per_anchor * (world_size / global_anchor_count)
        other_mask = torch.ones(
            negative_bank.shape[0], dtype=torch.bool, device=device
        )
        other_mask[rank * args.batch_size : (rank + 1) * args.batch_size] = False
        other_distances = negative_distances[:, other_mask]
        negative_distance = (
            other_distances.mean()
            if other_distances.numel()
            else negative_distances.mean()
        )
        return scaled, {
            "lcf": float(per_anchor.detach()),
            "lcf_positive_distance": float(positive_distance.detach().mean()),
            "lcf_negative_distance": float(negative_distance.detach()),
            "lcf_anchor": 1,
            "lcf_global_anchor_count": global_anchor_count,
        }

    def compute_loss(
        wavs,
        srs,
        phrases_list,
        tiers,
        candidates_list,
        game_cache_list,
        apply_lcf=True,
    ):
        (
            full_latent,
            cond,
            midi,
            midi_p,
            aligned_text,
            ref_len,
            T,
            placement_stats,
            p_classes,
        ) = process_batch(
            wavs,
            srs,
            phrases_list,
            tiers,
            candidates_list,
            game_cache_list,
        )

        u = torch.rand(1, device=device)
        t = args.t_shift * u / (1 - args.t_shift * u)

        noise = torch.randn_like(full_latent)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * full_latent
        v_target = full_latent - noise

        if args.phase == "p_only":
            drop_audio = drop_text = drop_midi = False
        else:
            drop_audio = random.random() < 0.3
            drop_text = random.random() < args.drop_text
            drop_midi = random.random() < 0.3

        lcf_training = bool(
            args.lcf_enabled and args.phase == "joint" and apply_lcf
        )
        low_noise = bool(
            lcf_training and float(t.item()) >= args.lcf_threshold
        )
        query_rng_before = capture_rng_state(device) if low_noise else None
        v_pred, hidden_states, lcf_hidden = run_dit(
            x_t,
            cond,
            aligned_text,
            t,
            midi,
            drop_audio,
            drop_text,
            drop_midi,
            capture_lcf=lcf_training,
        )
        query_rng_after = capture_rng_state(device) if low_noise else None

        L_flow_A = F.mse_loss(v_pred[:, :ref_len, :], v_target[:, :ref_len, :])
        L_flow_B = F.mse_loss(v_pred[:, ref_len:, :], v_target[:, ref_len:, :])
        L_flow = (
            L_flow_B
            if args.phase == "p_only"
            else L_flow_A + args.flow_b_weight * L_flow_B
        )

        L_cka = torch.tensor(0.0, device=device)
        if args.cka_weight > 0:
            h_B = [h[:, ref_len:, :] for h in hidden_states]
            m_B = midi_p[:, ref_len:, :]
            L_cka = compute_cka_loss_from_hidden(h_B, m_B)

        lcf_metrics = {
            "lcf": 0.0,
            "lcf_positive_distance": 0.0,
            "lcf_negative_distance": 0.0,
            "lcf_anchor": 0,
            "lcf_global_anchor_count": 0,
        }
        L_lcf = torch.tensor(0.0, device=device)
        if lcf_training:
            query = pool_lcf_feature(lcf_hidden, ref_len)
            positive = None
            if low_noise:
                boundary_t = torch.full_like(t, args.lcf_threshold)
                boundary_x = (
                    (1.0 - boundary_t[:, None, None]) * noise
                    + boundary_t[:, None, None] * full_latent
                )
                # Reuse the query's dropout mask for the positive, then put
                # every RNG back where the baseline query left it.
                restore_rng_state(query_rng_before, device)
                try:
                    with torch.no_grad():
                        _, _, boundary_hidden = run_dit(
                            boundary_x,
                            cond,
                            aligned_text,
                            boundary_t,
                            midi,
                            drop_audio,
                            drop_text,
                            drop_midi,
                            capture_lcf=True,
                        )
                        positive = pool_lcf_feature(boundary_hidden, ref_len)
                finally:
                    restore_rng_state(query_rng_after, device)
            L_lcf, lcf_metrics = compute_lcf_loss(query, positive, low_noise)

        if lcf_training:
            # Multiplication by zero retains an explicit zero-gradient path
            # through the velocity head on low-noise samples.
            global_anchor_count = lcf_metrics["lcf_global_anchor_count"]
            global_flow_count = world_size - global_anchor_count
            flow_scale = (
                world_size / global_flow_count
                if not low_noise and global_flow_count > 0
                else 0.0
            )
            training_flow = L_flow * flow_scale
            loss = (
                training_flow
                + args.cka_weight * L_cka
                + args.lcf_weight * L_lcf
            )
        else:
            # Preserve the archived V4PH arithmetic exactly when LCF is off
            # and for the frozen standard-FM evaluator.
            loss = L_flow + args.cka_weight * L_cka
        return loss, {
            "flow_A": L_flow_A.item(),
            "flow_B": L_flow_B.item(),
            "flow": L_flow.item(),
            "cka": L_cka.item(),
            "flow_applied": int(not low_noise),
            **lcf_metrics,
            "p_class_counts": torch.bincount(
                p_classes[:, ref_len:].reshape(-1), minlength=257
            ).detach(),
            **placement_stats,
        }

    @torch.no_grad()
    def run_eval(eval_loader):
        rng_state = capture_rng_state(device)
        seed_everything(args.eval_seed)
        raw_model.eval()
        total_loss = total_flow_a = total_flow_b = total_cka = 0.0
        count = 0
        for batch in eval_loader:
            wavs = [b.to(device) for b in batch["wav"]]
            srs = batch["sr"]
            phrases_list = batch["phrases"]
            tiers = batch["tier"]
            candidates_list = batch["h_candidates"]
            game_cache_list = batch["game_cache"]
            loss, d = compute_loss(
                wavs,
                srs,
                phrases_list,
                tiers,
                candidates_list,
                game_cache_list,
                apply_lcf=False,
            )
            total_loss += loss.item()
            total_flow_a += d["flow_A"]
            total_flow_b += d["flow_B"]
            total_cka += d["cka"]
            count += 1
        raw_model.train()
        result = (total_loss / max(count, 1), total_flow_a / max(count, 1),
                  total_flow_b / max(count, 1), total_cka / max(count, 1))
        restore_rng_state(rng_state, device)
        return result

    def save_training_checkpoint(
        save_path,
        global_step,
        sampler_epoch,
        batch_offset,
        metrics,
        run_state,
    ):
        local_state = {
            "rank": rank,
            "sampler_epoch": int(sampler_epoch),
            "batch_offset": int(batch_offset),
            "rng": capture_rng_state(device),
            "metrics": dict(metrics),
            "lcf_cumulative": lcf_cumulative_state(),
        }
        if world_size > 1:
            rank_states = [None] * world_size
            torch.distributed.all_gather_object(rank_states, local_state)
        else:
            rank_states = [local_state]

        rank_states.sort(key=lambda item: int(item["rank"]))
        cursors = {
            (int(item["sampler_epoch"]), int(item["batch_offset"]))
            for item in rank_states
        }
        if len(cursors) != 1:
            raise RuntimeError(f"rank data cursors diverged: {sorted(cursors)}")

        if is_main():
            payload = {
                "checkpoint_schema": expected_checkpoint_schema,
                "run_state": run_state,
                "model_state_dict": raw_model.state_dict(),
                "ema_model_state_dict": ema.ema_model.state_dict(),
                "ema_step": ema.step.detach().cpu().clone(),
                "ema_initted": ema.initted.detach().cpu().clone(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": int(global_step),
                "v4ph_training": training_metadata,
                "rank_states": rank_states,
                "pul_embedding_init": pul_embedding_init,
                "midi_p_schema": MIDI_P_V4PH_SCHEMA,
                "game_cache_schema": GAME_CACHE_SCHEMA,
                "p_support_counts": support_counts.detach().cpu().clone(),
                "p_distance_history": list(p_distance_history),
                "initial_p_weight": initial_p_weight.detach().cpu().clone(),
                "initial_frozen_fingerprint": initial_frozen_fingerprint,
                "phase_a_init_audit": phase_a_init_audit,
                "trainable_parameter_names": trainable_parameter_names,
                "args": vars(args),
            }
            atomic_torch_save(payload, save_path)
            print(f"  [Save  {global_step:6d}] {save_path} ({run_state})")
        if world_size > 1:
            distributed_barrier()

    if args.eval_only:
        if is_main():
            for eval_name, eval_loader in eval_loaders.items():
                ev_loss, ev_flow_a, ev_flow_b, ev_cka = run_eval(eval_loader)
                print(f"  >>> EVAL_ONLY/{eval_name} seed={args.eval_seed} | "
                      f"Loss={ev_loss:.4f} (FlowA={ev_flow_a:.4f} FlowB={ev_flow_b:.4f} "
                      f"CKA={ev_cka:.4f}) <<<")
        if torch.distributed.is_initialized():
            distributed_barrier()
            torch.distributed.destroy_process_group()
        return

    metric_defaults = {
        "flow_a": 0.0,
        "flow_b": 0.0,
        "cka": 0.0,
        "lcf": 0.0,
        "lcf_positive_distance": 0.0,
        "lcf_negative_distance": 0.0,
        "lcf_anchors": 0,
        "flow_applied": 0,
        "phone_phrases": 0,
        "fallback_phrases": 0,
        "pul_phrases": 0,
        "exact_control_phrases": 0,
        "pul_frames": 0,
        "control_anomalies": 0,
        "structural_fallbacks": 0,
        "nonpad_tokens": 0,
    }
    global_step = resume_global_step
    if resume_ckpt is None:
        support_counts = torch.zeros(257, dtype=torch.long, device=device)
        p_distance_history = []
    else:
        support_counts = resume_ckpt["p_support_counts"].to(device).long()
        p_distance_history = list(resume_ckpt["p_distance_history"])
    if resume_rank_state is None:
        sampler_epoch = 0
        batch_offset = 0
        restored_metrics = metric_defaults
        restored_lcf_cumulative = None
    else:
        sampler_epoch = int(resume_rank_state["sampler_epoch"])
        batch_offset = int(resume_rank_state["batch_offset"])
        restored_metrics = resume_rank_state.get("metrics")
        restored_lcf_cumulative = resume_rank_state.get("lcf_cumulative")
        if not isinstance(restored_metrics, dict) or set(restored_metrics) != set(
            metric_defaults
        ):
            raise ValueError("resume checkpoint has invalid running metrics")

    accum_flow_a = float(restored_metrics["flow_a"])
    accum_flow_b = float(restored_metrics["flow_b"])
    accum_cka = float(restored_metrics["cka"])
    accum_lcf = float(restored_metrics["lcf"])
    accum_lcf_positive_distance = float(
        restored_metrics["lcf_positive_distance"]
    )
    accum_lcf_negative_distance = float(
        restored_metrics["lcf_negative_distance"]
    )
    accum_lcf_anchors = int(restored_metrics["lcf_anchors"])
    accum_flow_applied = int(restored_metrics["flow_applied"])
    accum_phone_phrases = int(restored_metrics["phone_phrases"])
    accum_fallback_phrases = int(restored_metrics["fallback_phrases"])
    accum_pul_phrases = int(restored_metrics["pul_phrases"])
    accum_exact_control_phrases = int(
        restored_metrics["exact_control_phrases"]
    )
    accum_pul_frames = int(restored_metrics["pul_frames"])
    accum_control_anomalies = int(restored_metrics["control_anomalies"])
    accum_structural_fallbacks = int(restored_metrics["structural_fallbacks"])
    accum_nonpad_tokens = int(restored_metrics["nonpad_tokens"])

    lcf_cumulative_defaults = {
        "samples": 0,
        "flow_applied": 0,
        "anchors": 0,
        "loss_sum": 0.0,
        "positive_distance_sum": 0.0,
        "negative_distance_sum": 0.0,
    }
    if restored_lcf_cumulative is None:
        restored_lcf_cumulative = lcf_cumulative_defaults
    if set(restored_lcf_cumulative) != set(lcf_cumulative_defaults):
        raise ValueError("resume checkpoint has invalid LCF cumulative metrics")
    total_samples = int(restored_lcf_cumulative["samples"])
    total_flow_applied = int(restored_lcf_cumulative["flow_applied"])
    total_lcf_anchors = int(restored_lcf_cumulative["anchors"])
    total_lcf_loss = float(restored_lcf_cumulative["loss_sum"])
    total_lcf_positive_distance = float(
        restored_lcf_cumulative["positive_distance_sum"]
    )
    total_lcf_negative_distance = float(
        restored_lcf_cumulative["negative_distance_sum"]
    )

    consumed_batches = global_step * args.grad_accum
    if total_samples != consumed_batches:
        raise ValueError(
            f"LCF cumulative sample count {total_samples} does not match "
            f"consumed batches {consumed_batches}"
        )
    if total_flow_applied + total_lcf_anchors != total_samples:
        raise ValueError("LCF cumulative flow/anchor partition is inconsistent")
    if consumed_batches == 0:
        expected_cursor = (0, 0)
    else:
        expected_epoch = (consumed_batches - 1) // train_sampler.num_samples
        expected_offset = consumed_batches - expected_epoch * train_sampler.num_samples
        expected_cursor = (expected_epoch, expected_offset)
    if (sampler_epoch, batch_offset) != expected_cursor:
        raise ValueError(
            f"data cursor {(sampler_epoch, batch_offset)} does not match "
            f"step {global_step}: expected {expected_cursor}"
        )

    seed_everything(args.seed + rank)
    train_sampler.set_epoch_and_offset(sampler_epoch, batch_offset)
    data_iter = iter(train_loader)
    if resume_rank_state is not None:
        # Iterator construction consumes a torch seed. Restore only after the
        # exact sampler cursor has been rebuilt.
        restore_rng_state(resume_rank_state["rng"], device)
        if is_main():
            print(
                f"[Train V4PH] Exact resume: step={global_step} "
                f"epoch={sampler_epoch} batch_offset={batch_offset} "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

    def running_metrics():
        return {
            "flow_a": accum_flow_a,
            "flow_b": accum_flow_b,
            "cka": accum_cka,
            "lcf": accum_lcf,
            "lcf_positive_distance": accum_lcf_positive_distance,
            "lcf_negative_distance": accum_lcf_negative_distance,
            "lcf_anchors": accum_lcf_anchors,
            "flow_applied": accum_flow_applied,
            "phone_phrases": accum_phone_phrases,
            "fallback_phrases": accum_fallback_phrases,
            "pul_phrases": accum_pul_phrases,
            "exact_control_phrases": accum_exact_control_phrases,
            "pul_frames": accum_pul_frames,
            "control_anomalies": accum_control_anomalies,
            "structural_fallbacks": accum_structural_fallbacks,
            "nonpad_tokens": accum_nonpad_tokens,
        }

    def lcf_cumulative_state():
        return {
            "samples": total_samples,
            "flow_applied": total_flow_applied,
            "anchors": total_lcf_anchors,
            "loss_sum": total_lcf_loss,
            "positive_distance_sum": total_lcf_positive_distance,
            "negative_distance_sum": total_lcf_negative_distance,
        }

    run_until_step = args.stop_after_step or args.max_steps
    if run_until_step < global_step:
        raise ValueError(
            f"run target {run_until_step} precedes resume step {global_step}"
        )
    step_offset = global_step
    t_start = time.time()

    while global_step < run_until_step:
        optimizer.zero_grad(set_to_none=True)
        step_support_counts = torch.zeros(257, dtype=torch.long, device=device)

        for ga_step in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                sampler_epoch += 1
                batch_offset = 0
                train_sampler.set_epoch_and_offset(sampler_epoch, batch_offset)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch_offset += 1

            wavs = [b.to(device) for b in batch["wav"]]
            srs = batch["sr"]
            phrases_list = batch["phrases"]
            tiers = batch["tier"]
            candidates_list = batch["h_candidates"]
            game_cache_list = batch["game_cache"]
            loss, loss_dict = compute_loss(
                wavs,
                srs,
                phrases_list,
                tiers,
                candidates_list,
                game_cache_list,
            )
            loss = loss / args.grad_accum
            loss.backward()
            accum_flow_a += loss_dict["flow_A"]
            accum_flow_b += loss_dict["flow_B"]
            accum_cka += loss_dict["cka"]
            accum_lcf += loss_dict["lcf"]
            accum_lcf_positive_distance += loss_dict[
                "lcf_positive_distance"
            ]
            accum_lcf_negative_distance += loss_dict[
                "lcf_negative_distance"
            ]
            accum_lcf_anchors += loss_dict["lcf_anchor"]
            accum_flow_applied += loss_dict["flow_applied"]
            total_samples += 1
            total_flow_applied += loss_dict["flow_applied"]
            total_lcf_anchors += loss_dict["lcf_anchor"]
            total_lcf_loss += loss_dict["lcf"]
            total_lcf_positive_distance += loss_dict[
                "lcf_positive_distance"
            ]
            total_lcf_negative_distance += loss_dict[
                "lcf_negative_distance"
            ]
            accum_phone_phrases += loss_dict["phone_phrases"]
            accum_fallback_phrases += loss_dict["fallback_phrases"]
            accum_pul_phrases += loss_dict["pul_phrases"]
            accum_exact_control_phrases += loss_dict["exact_control_phrases"]
            accum_pul_frames += loss_dict["pul_frames"]
            accum_control_anomalies += loss_dict["sample_control_anomaly"]
            accum_structural_fallbacks += loss_dict[
                "sample_structural_fallback"
            ]
            accum_nonpad_tokens += loss_dict["nonpad_tokens"]
            step_support_counts += loss_dict["p_class_counts"].to(device)

        sync_gradients(raw_model, world_size)
        if world_size > 1:
            torch.distributed.all_reduce(step_support_counts)
        support_counts += step_support_counts
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if args.phase == "p_only" and global_step in (100, 300, 500):
            from src.YingMusicSinger.melody.midi_p_v4ph import pitch_kernel_distance

            current_distance = pitch_kernel_distance(
                raw_model.midi_p_v4ph.embedding.weight,
                raw_model.transformer.input_embed_with_midi.midi_proj,
                support_counts,
            )
            initial_distance = pitch_kernel_distance(
                initial_p_weight,
                raw_model.transformer.input_embed_with_midi.midi_proj,
                support_counts,
            )
            distance_record = {
                "step": global_step,
                "current": current_distance,
                "initial_on_same_support": initial_distance,
            }
            p_distance_history.append(distance_record)
            if is_main():
                print(
                    "[V4PH P distance] "
                    f"{json.dumps(distance_record, sort_keys=True)}"
                )

        if global_step % 100 == 0:
            torch.cuda.empty_cache()

        if global_step % args.log_every == 0:
            log_values = torch.tensor(
                [
                    accum_flow_a,
                    accum_flow_b,
                    accum_cka,
                    accum_phone_phrases,
                    accum_fallback_phrases,
                    accum_pul_phrases,
                    accum_exact_control_phrases,
                    accum_control_anomalies,
                    accum_structural_fallbacks,
                    accum_pul_frames,
                    accum_nonpad_tokens,
                    accum_lcf,
                    accum_lcf_positive_distance,
                    accum_lcf_negative_distance,
                    accum_lcf_anchors,
                    accum_flow_applied,
                ],
                dtype=torch.float64,
                device=device,
            )
            if world_size > 1:
                torch.distributed.all_reduce(log_values)
            if is_main():
                n = args.log_every * args.grad_accum * world_size
                avg_a = log_values[0].item() / n
                avg_b = log_values[1].item() / n
                avg_cka = log_values[2].item() / n
                anchor_count = int(log_values[14].item())
                avg_lcf = log_values[11].item() / max(anchor_count, 1)
                avg_lcf_positive = log_values[12].item() / max(
                    anchor_count, 1
                )
                avg_lcf_negative = log_values[13].item() / max(
                    anchor_count, 1
                )
                low_noise_fraction = anchor_count / n
                flow_fraction = log_values[15].item() / n
                phone_count = int(log_values[3].item())
                fallback_count = int(log_values[4].item())
                pul_count = int(log_values[5].item())
                exact_count = int(log_values[6].item())
                placement_coverage = phone_count / max(
                    phone_count + fallback_count, 1
                )
                pul_coverage = pul_count / max(phone_count + fallback_count, 1)
                exact_coverage = exact_count / max(
                    phone_count + fallback_count, 1
                )
                elapsed = time.time() - t_start
                steps_since_start = global_step - step_offset
                sec_per_step = elapsed / max(steps_since_start, 1)
                remaining = run_until_step - global_step
                eta = remaining * sec_per_step
                eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m"
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  [{global_step:6d}/{args.max_steps} | "
                    f"{100*global_step/args.max_steps:.1f}% | "
                    f"ETA {eta_str} | {sec_per_step:.2f}s/step] "
                    f"FlowA={avg_a:.4f} FlowB={avg_b:.4f} CKA={avg_cka:.4f} | "
                    f"LCF={avg_lcf:.4f} Anchors={anchor_count}/{n} "
                    f"LowFrac={low_noise_fraction:.4f} "
                    f"PosD={avg_lcf_positive:.4f} "
                    f"NegD={avg_lcf_negative:.4f} "
                    f"FlowFrac={flow_fraction:.4f} | "
                    f"Placement={placement_coverage:.3f} "
                    f"PUL={pul_coverage:.3f} Exact={exact_coverage:.3f} "
                    f"ControlAnomaly={int(log_values[7].item())} "
                    f"Structural={int(log_values[8].item())} "
                    f"PULFrames={int(log_values[9].item())} "
                    f"NonPAD={int(log_values[10].item())} | LR={lr_now:.2e}"
                )
            accum_flow_a = accum_flow_b = accum_cka = 0.0
            accum_lcf = 0.0
            accum_lcf_positive_distance = 0.0
            accum_lcf_negative_distance = 0.0
            accum_lcf_anchors = 0
            accum_flow_applied = 0
            accum_phone_phrases = accum_fallback_phrases = 0
            accum_pul_phrases = accum_exact_control_phrases = 0
            accum_pul_frames = 0
            accum_control_anomalies = accum_structural_fallbacks = 0
            accum_nonpad_tokens = 0

        should_eval = global_step % args.eval_every == 0 and (
            args.phase != "p_only" or global_step in (100, 300, 500)
        )
        if should_eval and eval_loaders:
            if is_main():
                for eval_name, eval_loader in eval_loaders.items():
                    ev_loss, ev_flow_a, ev_flow_b, ev_cka = run_eval(eval_loader)
                    print(f"  >>> EVAL/{eval_name} step={global_step} "
                          f"Loss={ev_loss:.4f} (FlowA={ev_flow_a:.4f} FlowB={ev_flow_b:.4f} "
                          f"CKA={ev_cka:.4f}) <<<")
            if world_size > 1:
                distributed_barrier()

        should_save = global_step % args.save_every == 0 and (
            args.phase != "p_only" or global_step in (100, 300)
        )
        if should_save and global_step < run_until_step:
            save_path = os.path.join(args.output_dir, f"step_{global_step:06d}.pt")
            save_training_checkpoint(
                save_path,
                global_step,
                sampler_epoch,
                batch_offset,
                running_metrics(),
                "running",
            )

    torch.cuda.empty_cache()
    if args.smoke_assertions:
        assert_rank_parameter_fingerprints(raw_model, world_size, "final")
    if args.phase == "p_only":
        final_frozen_fingerprint = frozen_parameter_sha256(raw_model)
        if final_frozen_fingerprint != initial_frozen_fingerprint:
            raise RuntimeError(
                "V4PH p_only modified a frozen parameter: "
                f"{final_frozen_fingerprint} != {initial_frozen_fingerprint}"
            )

    final_state = "complete" if global_step >= args.max_steps else "stopped"
    save_path = os.path.join(args.output_dir, f"step_{global_step:06d}_final.pt")
    save_training_checkpoint(
        save_path,
        global_step,
        sampler_epoch,
        batch_offset,
        running_metrics(),
        final_state,
    )
    if is_main():
        print(f"[Train V4PH] Final: {save_path}")
        print(f"[Train V4PH] {final_state.capitalize()} at step {global_step}")

    if torch.distributed.is_initialized():
        distributed_barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
