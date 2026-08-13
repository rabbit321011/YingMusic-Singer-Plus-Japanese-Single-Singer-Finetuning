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
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from ema_pytorch import EMA
from omegaconf import OmegaConf

FRAME_RATE = 44100 / 2048
SEP_TOKEN = 365
PUL_TOKEN = 366
V5SG_CHECKPOINT_SCHEMA = "v5sg_training_checkpoint_v1"

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if (candidate / "h_alignment").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from h_alignment.placement import render_h_pul_placements, render_paired_placements


def checkpoint_schema_for_mode(placement_mode, warmstart_checkpoint=None):
    if placement_mode != "phone_pul":
        raise ValueError("V5-Sg requires phone_pul placement")
    if warmstart_checkpoint:
        raise ValueError("V5-Sg must start fresh from the official checkpoint")
    return V5SG_CHECKPOINT_SCHEMA


def v5sg_lr_factor(
    step,
    peak_lr=1.4e-5,
    mid_lr=1e-5,
    warmup_steps=4000,
    first_decay_end=28000,
    max_steps=40000,
):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    mid_factor = mid_lr / peak_lr
    if step < first_decay_end:
        progress = (step - warmup_steps) / max(1, first_decay_end - warmup_steps)
        shoulder = 0.5 * (1 + np.cos(np.pi * progress))
        return mid_factor + (1 - mid_factor) * shoulder
    progress = (step - first_decay_end) / max(1, max_steps - first_decay_end)
    return mid_factor * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))


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
        print(f"[Train H] Rank parameter SHA256 ({label}): {local_fingerprint}")
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
        "midi_checkpoint": pathlib.Path(args.midi_ckpt),
        "pool_audit": pathlib.Path(args.pool_audit),
    }
    file_sha256 = {}
    for label, path in fingerprint_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        file_sha256[label] = sha256_file(path)
    if file_sha256["pool_audit"] != args.pool_audit_sha256:
        raise ValueError(
            "pool audit SHA256 mismatch: "
            f"{file_sha256['pool_audit']} != {args.pool_audit_sha256}"
        )

    metadata = {
        "schema": checkpoint_schema_for_mode(args.placement_mode),
        "route": "V5-Sg",
        "phase": "s_g",
        "placement_mode": args.placement_mode,
        "h_config_fingerprint": args.h_config_fingerprint,
        "train_manifest_sha256": args.train_manifest_sha256,
        "eval_manifest_sha256": args.eval_manifest_sha256,
        "short_manifest_sha256": args.short_manifest_sha256,
        "long_manifest_sha256": args.long_manifest_sha256,
        "pool_audit_sha256": args.pool_audit_sha256,
        "pool_policy": "KEEP_LONG_DEDUP_SHORT",
        "sampling_policy": "NATURAL_RECORD",
        "world_size": int(world_size),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "hold_steps": args.hold_steps,
        "first_decay_end": args.first_decay_end,
        "mid_lr": args.mid_lr,
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
        "ema_device": args.ema_device,
        "midi_teacher": "continuous SOME frozen V4Hg teacher",
        "midi_fuzz_disturb": True,
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
    metadata["fresh_start"] = {
        "source": "official checkpoint",
        "weight_source": "official EMA when present, otherwise model_state_dict",
        "raw_model_policy": "initialize_from_official",
        "ema_policy": "fresh_from_initial_weights",
        "optimizer_policy": "fresh",
        "scheduler_policy": "fresh_step_zero",
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
        "pool": [item["pool"] for item in batch],
        "segment_count": [item["segment_count"] for item in batch],
    }


class HDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        expected_sha256,
        expected_config_fingerprint,
        max_duration_sec=60.1,
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
            for required in ("Path", "AudioSHA256", "AudioFrames", "Pool", "SampleId"):
                if required not in item:
                    raise ValueError(f"V5-Sg manifest missing {required} at record {index}")
        self.max_duration_sec = max_duration_sec
        self.manifest_sha256 = actual_sha256

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        wav, sr = torchaudio.load(rec["Path"])
        info = torchaudio.info(rec["Path"])
        if (
            sr != 44100
            or info.num_channels != 1
            or info.encoding != "PCM_S"
            or info.bits_per_sample != 16
        ):
            raise ValueError(f"V5-Sg requires 44.1kHz mono PCM16: {rec['Path']}")
        if sha256_file(rec["Path"]) != rec["AudioSHA256"]:
            raise ValueError(f"audio SHA256 mismatch: {rec['Path']}")
        if wav.shape[-1] != int(rec["AudioFrames"]):
            raise ValueError(f"audio frame mismatch at {rec['Path']}")
        max_samples = int(self.max_duration_sec * sr)
        if wav.shape[-1] > max_samples:
            raise ValueError(f"V5-Sg forbids loader cropping: {wav.shape[-1]} > {max_samples}")
        return {
            "wav": wav,
            "sr": sr,
            "phrases": rec["Phrases"],
            "tier": rec.get("Tier", "L1"),
            "duration": rec["Duration"],
            "h_candidates": rec["HAlignment"]["phrase_candidates"],
            "sample_id": rec["SampleId"],
            "pool": rec["Pool"],
            "segment_count": int(rec.get("SegmentCount", 1)),
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
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--eval_manifest", required=True)
    parser.add_argument("--train_manifest_sha256", required=True)
    parser.add_argument("--eval_manifest_sha256", required=True)
    parser.add_argument("--short_manifest_sha256", required=True)
    parser.add_argument("--long_manifest_sha256", required=True)
    parser.add_argument("--pool_audit", required=True)
    parser.add_argument("--pool_audit_sha256", required=True)
    parser.add_argument("--h_config_fingerprint", required=True)
    parser.add_argument(
        "--placement_mode",
        choices=("sentence", "phone", "phone_pul"),
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--warmstart_checkpoint", default=None,
                        help="Rejected: V5-Sg always starts from official checkpoint")
    parser.add_argument("--warmstart_expected_sha256", default=None)
    parser.add_argument("--warmstart_expected_step", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--hold_steps", type=int, default=0,
                        help="Compatibility field; V5-Sg uses two cosine phases")
    parser.add_argument("--first_decay_end", type=int, default=28000)
    parser.add_argument("--mid_lr", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--max_duration", type=float, default=60.1)
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
    parser.add_argument("--expected_world_size", type=int, default=4)
    parser.add_argument("--ema_device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--smoke_assertions", action="store_true")
    parser.add_argument(
        "--stop_after_step",
        type=int,
        default=None,
        help="Exit cleanly at this absolute step without changing the LR schedule.",
    )
    args = parser.parse_args()
    expected_checkpoint_schema = checkpoint_schema_for_mode(args.placement_mode)

    if args.batch_size != 1:
        raise ValueError("H training currently requires batch_size=1")
    if args.placement_mode != "phone_pul":
        raise ValueError("V5-Sg requires --placement_mode phone_pul")
    if args.num_workers != 0:
        raise ValueError("exact H resume requires num_workers=0")
    if args.grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    for interval_name in ("save_every", "eval_every", "log_every"):
        if getattr(args, interval_name) <= 0:
            raise ValueError(f"{interval_name} must be positive")
    if args.stop_after_step is not None and not (
        0 <= args.stop_after_step <= args.max_steps
    ):
        raise ValueError("stop_after_step must be in [0, max_steps]")
    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(args.resume)
    if args.warmstart_checkpoint or args.warmstart_expected_sha256 or args.warmstart_expected_step is not None:
        raise ValueError("V5-Sg does not accept a warm-start checkpoint; use the official base")
    if args.hold_steps != 0:
        raise ValueError("V5-Sg hold_steps must remain zero")
    if args.warmup_steps != 4000 or args.first_decay_end != 28000 or args.max_steps != 40000:
        raise ValueError("V5-Sg requires 4k/28k/40k scheduler milestones")
    if args.mid_lr != 1e-5:
        raise ValueError("V5-Sg mid_lr must be 1e-5")
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
        print(f"[Train V5-Sg] World={world_size} Device={device}")
        print(f"[Train V5-Sg] Args: {args}")

    metadata_box = [build_training_metadata(args, world_size) if is_main() else None]
    if world_size > 1:
        torch.distributed.broadcast_object_list(metadata_box, src=0)
    training_metadata = metadata_box[0]

    cfg = OmegaConf.load(args.config)

    from src.YingMusicSinger.models.dit import DiT
    from src.YingMusicSinger.models.model import Singer
    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram
    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer

    if is_main():
        print("[Train H] Building DiT...")
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

    resume_global_step = 0
    resume_ckpt = None
    resume_rank_state = None
    warmstart_ckpt = None
    if args.resume:
        if is_main():
            print(f"[Train H] Resuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_ckpt.get("checkpoint_schema") != expected_checkpoint_schema:
            raise ValueError(
                f"resume checkpoint schema mismatch: "
                f"{resume_ckpt.get('checkpoint_schema')} != "
                f"{expected_checkpoint_schema}"
            )
        metadata = resume_ckpt.get("h_training")
        if metadata != training_metadata:
            raise ValueError(
                f"resume H metadata mismatch: {metadata} != {training_metadata}"
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
        }
        missing_keys = sorted(required_keys - resume_ckpt.keys())
        if missing_keys:
            raise ValueError(f"resume checkpoint missing keys: {missing_keys}")
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
    elif args.ckpt_path and os.path.exists(args.ckpt_path):
        if is_main():
            print(f"[Train H] Loading base checkpoint: {args.ckpt_path}")
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
            expected_missing=("transformer.long_skip_connection.weight",),
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
        if resume_ckpt is None and warmstart_ckpt is None:
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
                    "[Train H] Initialized PUL embedding row "
                    f"{destination_row} from V row {source_row}: {source_sha}"
                )
        else:
            source_checkpoint = resume_ckpt or warmstart_ckpt
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
            if warmstart_ckpt is not None and is_main():
                v_sha = tensor_sha256(embedding[source_row])
                pul_sha = tensor_sha256(embedding[destination_row])
                print(
                    "[Train H] Preserved warm-start V/PUL rows without copy: "
                    f"V={v_sha} PUL={pul_sha}"
                )

    singer_model = singer_model.to(device)
    singer_model.train()
    raw_model = singer_model
    if args.smoke_assertions:
        assert_rank_parameter_fingerprints(raw_model, world_size, "initial")

    if is_main():
        print("[Train V5-Sg] Loading 285k VAE...")
    vae = StableAudioInfer(model_config_path=args.vae_config, model_ckpt_path=args.vae_ckpt)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    if is_main():
        print("[Train V5-Sg] Loading continuous SOME teacher...")
    midi_teacher = MIDIExtractor(in_dim=80)
    midi_teacher._load_form_ckpt(args.midi_ckpt)
    midi_teacher = midi_teacher.to(device).eval()
    for p in midi_teacher.parameters():
        p.requires_grad = False

    mel_spec_extract = MelodySpectrogram()

    ema = EMA(
        singer_model,
        beta=cfg.ema_kwargs.beta,
        update_after_step=cfg.ema_kwargs.update_after_step,
        update_every=cfg.ema_kwargs.update_every,
        include_online_model=False,
        allow_different_devices=args.ema_device == "cpu",
    )
    ema.to("cpu" if args.ema_device == "cpu" else device)
    if resume_ckpt is not None:
        ema_sd = {
            k.replace("ema_model.", ""): v
            for k, v in resume_ckpt["ema_model_state_dict"].items()
        }
        load_state_dict_checked(ema.ema_model, ema_sd, "resume EMA")
        ema.step.copy_(resume_ckpt["ema_step"])
        ema.initted.copy_(resume_ckpt["ema_initted"])
    else:
        # Fresh official start: make step-0 raw and EMA bit-exact, with a
        # fresh counter rather than inheriting any historical EMA cadence.
        load_state_dict_checked(ema.ema_model, raw_model.state_dict(), "fresh EMA")
        ema.step.zero_()
        ema.initted.fill_(True)
    # The loss calls transformer internals directly, so DDP.forward would be
    # bypassed. Gradients are explicitly synchronized before clipping below.
    raw_model = singer_model

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=1e-2)

    # V5-Sg: 4k linear warmup, 4k--28k cosine shoulder, 28k--40k cosine decay.
    def lr_lambda(step):
        return v5sg_lr_factor(
            step,
            peak_lr=args.lr,
            mid_lr=args.mid_lr,
            warmup_steps=args.warmup_steps,
            first_decay_end=args.first_decay_end,
            max_steps=args.max_steps,
        )
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
            print("[Train H] Restored optimizer and scheduler state")

    train_dataset = HDataset(
        args.train_manifest,
        args.train_manifest_sha256,
        args.h_config_fingerprint,
        args.max_duration,
    )
    if args.overfit:
        train_dataset.records = train_dataset.records[:args.overfit_n]
        if is_main():
            print(f"[Train H] OVERFIT MODE: {len(train_dataset)} samples")
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
    )
    if args.overfit:
        eval_dataset.records = eval_dataset.records[: max(4, args.overfit_n // 5)]
    eval_loaders = {}
    eval_sizes = {}
    for pool in ("Short", "Long"):
        indices = [
            index for index, record in enumerate(eval_dataset.records)
            if record.get("Pool") == pool
        ]
        if indices:
            eval_loaders[pool.lower()] = DataLoader(
                Subset(eval_dataset, indices),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_svs,
                pin_memory=True,
                drop_last=True,
            )
            eval_sizes[pool.lower()] = len(indices)

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main():
        eval_summary = ", ".join(f"{name}={size}" for name, size in eval_sizes.items()) or "none"
        print(f"[Train V5-Sg/{args.placement_mode}] Train: {len(train_dataset)} | Eval: {eval_summary}")
        print(f"[Train V5-Sg] Steps/epoch: {len(train_loader)} | Max steps: {args.max_steps}")
        print(f"[Train V5-Sg] Effective batch: {args.batch_size * world_size * args.grad_accum}")
        print(f"[Train V5-Sg] lr={args.lr} warmup={args.warmup_steps} decay={args.first_decay_end} "
              f"mid_lr={args.mid_lr} drop_text={args.drop_text} "
              f"flow_b_w={args.flow_b_weight} cka={args.cka_weight}")
        if args.resume:
            print(f"[Train H] Resumed from step {resume_global_step}")
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

    def process_batch(wavs, srs, phrases_list, tiers, candidates_list):
        with torch.no_grad():
            w = wavs[0]
            sr = srs[0]
            phrases = phrases_list[0]
            tier = tiers[0]
            h_candidates = candidates_list[0]

            w_2d = w.unsqueeze(0) if w.dim() == 1 else w
            if w_2d.shape[-1] > 30 * sr:
                # Release cached training allocations before long-form VAE encoding.
                torch.cuda.empty_cache()
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

            mel = mel_spec_extract(audio=w_2d, sr=44100).to(device)
            midi_p, _ = midi_teacher(mel.transpose(1, 2))
            if midi_p.shape[1] != T:
                midi_p = F.interpolate(midi_p.transpose(1, 2), size=T,
                                       mode="linear", align_corners=False).transpose(1, 2)
            midi = raw_model.smoothMelody_MIDIFuzzDisturb(midi_p)
            midi[:, :ref_len, :] = 0

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
        )

    def run_dit(x_t, cond, text_tokens, t, midi, drop_audio, drop_text, drop_midi):
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
        if dit.long_skip_connection is not None:
            residual = x
        for i, block in enumerate(dit.transformer_blocks):
            x = block(x, time_emb, mask=None, rope=rope)
            if i >= len(dit.transformer_blocks) - 3:
                hidden_states.append(x)
        if dit.long_skip_connection is not None:
            x = dit.long_skip_connection(torch.cat((x, residual), dim=-1))
        x = dit.norm_out(x, time_emb)
        output = dit.proj_out(x)
        return output, hidden_states

    def compute_loss(wavs, srs, phrases_list, tiers, candidates_list):
        (
            full_latent,
            cond,
            midi,
            midi_p,
            aligned_text,
            ref_len,
            T,
            placement_stats,
        ) = process_batch(wavs, srs, phrases_list, tiers, candidates_list)

        u = torch.rand(1, device=device)
        t = args.t_shift * u / (1 - args.t_shift * u)

        noise = torch.randn_like(full_latent)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * full_latent
        v_target = full_latent - noise

        drop_audio = random.random() < 0.3
        drop_text = random.random() < args.drop_text
        drop_midi = random.random() < 0.3

        v_pred, hidden_states = run_dit(x_t, cond, aligned_text, t, midi,
                                         drop_audio, drop_text, drop_midi)

        L_flow_A = F.mse_loss(v_pred[:, :ref_len, :], v_target[:, :ref_len, :])
        L_flow_B = F.mse_loss(v_pred[:, ref_len:, :], v_target[:, ref_len:, :])
        L_flow = L_flow_A + args.flow_b_weight * L_flow_B

        L_cka = torch.tensor(0.0, device=device)
        if args.cka_weight > 0:
            h_B = [h[:, ref_len:, :] for h in hidden_states]
            m_B = midi_p[:, ref_len:, :]
            L_cka = compute_cka_loss_from_hidden(h_B, m_B)

        loss = L_flow + args.cka_weight * L_cka
        return loss, {
            "flow_A": L_flow_A.item(),
            "flow_B": L_flow_B.item(),
            "flow": L_flow.item(),
            "cka": L_cka.item(),
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
            loss, d = compute_loss(
                wavs, srs, phrases_list, tiers, candidates_list
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
                "h_training": training_metadata,
                "rank_states": rank_states,
                "pul_embedding_init": pul_embedding_init,
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
    if resume_rank_state is None:
        sampler_epoch = 0
        batch_offset = 0
        restored_metrics = metric_defaults
    else:
        sampler_epoch = int(resume_rank_state["sampler_epoch"])
        batch_offset = int(resume_rank_state["batch_offset"])
        restored_metrics = resume_rank_state.get("metrics")
        if not isinstance(restored_metrics, dict) or set(restored_metrics) != set(
            metric_defaults
        ):
            raise ValueError("resume checkpoint has invalid running metrics")

    accum_flow_a = float(restored_metrics["flow_a"])
    accum_flow_b = float(restored_metrics["flow_b"])
    accum_cka = float(restored_metrics["cka"])
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

    consumed_batches = global_step * args.grad_accum
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
                f"[Train H] Exact resume: step={global_step} "
                f"epoch={sampler_epoch} batch_offset={batch_offset} "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

    def running_metrics():
        return {
            "flow_a": accum_flow_a,
            "flow_b": accum_flow_b,
            "cka": accum_cka,
            "phone_phrases": accum_phone_phrases,
            "fallback_phrases": accum_fallback_phrases,
            "pul_phrases": accum_pul_phrases,
            "exact_control_phrases": accum_exact_control_phrases,
            "pul_frames": accum_pul_frames,
            "control_anomalies": accum_control_anomalies,
            "structural_fallbacks": accum_structural_fallbacks,
            "nonpad_tokens": accum_nonpad_tokens,
        }

    run_until_step = (
        args.max_steps if args.stop_after_step is None else args.stop_after_step
    )
    if run_until_step < global_step:
        raise ValueError(
            f"run target {run_until_step} precedes resume step {global_step}"
        )
    step_offset = global_step
    t_start = time.time()

    while global_step < run_until_step:
        optimizer.zero_grad(set_to_none=True)

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
            loss, loss_dict = compute_loss(
                wavs, srs, phrases_list, tiers, candidates_list
            )
            loss = loss / args.grad_accum
            loss.backward()
            accum_flow_a += loss_dict["flow_A"]
            accum_flow_b += loss_dict["flow_B"]
            accum_cka += loss_dict["cka"]
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

        sync_gradients(raw_model, world_size)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

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
                    f"Placement={placement_coverage:.3f} "
                    f"PUL={pul_coverage:.3f} Exact={exact_coverage:.3f} "
                    f"ControlAnomaly={int(log_values[7].item())} "
                    f"Structural={int(log_values[8].item())} "
                    f"PULFrames={int(log_values[9].item())} "
                    f"NonPAD={int(log_values[10].item())} | LR={lr_now:.2e}"
                )
            accum_flow_a = accum_flow_b = accum_cka = 0.0
            accum_phone_phrases = accum_fallback_phrases = 0
            accum_pul_phrases = accum_exact_control_phrases = 0
            accum_pul_frames = 0
            accum_control_anomalies = accum_structural_fallbacks = 0
            accum_nonpad_tokens = 0

        if global_step % args.eval_every == 0 and eval_loaders:
            if is_main():
                for eval_name, eval_loader in eval_loaders.items():
                    ev_loss, ev_flow_a, ev_flow_b, ev_cka = run_eval(eval_loader)
                    print(f"  >>> EVAL/{eval_name} step={global_step} "
                          f"Loss={ev_loss:.4f} (FlowA={ev_flow_a:.4f} FlowB={ev_flow_b:.4f} "
                          f"CKA={ev_cka:.4f}) <<<")
            if world_size > 1:
                distributed_barrier()

        if global_step % args.save_every == 0 and global_step < run_until_step:
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
        print(f"[Train H] Final: {save_path}")
        print(f"[Train H] {final_state.capitalize()} at step {global_step}")

    if torch.distributed.is_initialized():
        distributed_barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
