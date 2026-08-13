"""Train M600-B with four-way FP32 FSDP and a sharded FP32 EMA."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import random
import re
import shutil
import statistics
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src").is_dir():
    YING_REPO = PROJECT_DIR
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(YING_REPO))

import train_v4ph as base  # noqa: E402
from prepare_v4m_m600b_transition import (  # noqa: E402
    TARGET_DEPTH,
    TARGET_DIT_PARAMETERS,
    TARGET_FF_MULT,
    TARGET_HEADS,
    TRANSITION_SCHEMA,
    build_singer,
    sha256_file,
    strict_load,
)
from src.YingMusicSinger.melody.game_cache_v4ph import (  # noqa: E402
    GAME_CACHE_SCHEMA,
    game_cache_to_model_tracks,
)
from src.YingMusicSinger.melody.midi_p_v4ph import (  # noqa: E402
    MIDI_P_V4PH_SCHEMA,
)
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import (  # noqa: E402
    StableAudioInfer,
)


CHECKPOINT_SCHEMA = "v4m_m600b_fsdp_checkpoint_v1"
REPORT_SCHEMA = "v4m_m600b_fsdp_run_report_v1"


def canonical_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "")


def distributed_barrier():
    dist.barrier(device_ids=[torch.cuda.current_device()])


def atomic_torch_save(payload, output: pathlib.Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def memory_event(stage: str, **extra):
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "stage": stage,
        "time": time.time(),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
        **extra,
    }


def timing_summary(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def tensor_fingerprint(named_tensors):
    digest = hashlib.sha256()
    for name, value in sorted(named_tensors, key=lambda item: item[0]):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def module_local_fingerprint(module):
    tensors = [
        (f"parameter:{canonical_name(name)}", value)
        for name, value in module.named_parameters()
    ]
    tensors.extend(
        (f"buffer:{canonical_name(name)}", value)
        for name, value in module.named_buffers()
    )
    return tensor_fingerprint(tensors)


def optimizer_local_fingerprint(optimizer):
    tensors = []
    scalars = []
    parameter_index = 0
    for group_index, group in enumerate(optimizer.param_groups):
        for key, value in sorted(group.items()):
            if key == "params":
                continue
            scalars.append((f"group:{group_index}:{key}", value))
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            for key, value in sorted(state.items()):
                name = f"state:{parameter_index}:{key}"
                if torch.is_tensor(value):
                    tensors.append((name, value))
                else:
                    scalars.append((name, value))
            parameter_index += 1
    digest = hashlib.sha256()
    digest.update(tensor_fingerprint(tensors).encode("ascii"))
    digest.update(json.dumps(scalars, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


class M600BTrainingGraph(nn.Module):
    def __init__(self, singer, checkpoint_activations: bool):
        super().__init__()
        self.singer = singer
        self.checkpoint_activations = checkpoint_activations

    def forward(
        self,
        x_t,
        cond,
        text_tokens,
        diffusion_time,
        p_classes,
        ref_len,
        drop_audio,
        drop_text,
        drop_midi,
    ):
        midi_full = self.singer.midi_p_v4ph(p_classes)
        midi = torch.cat(
            [torch.zeros_like(midi_full[:, :ref_len]), midi_full[:, ref_len:]],
            dim=1,
        )
        dit = self.singer.transformer
        if diffusion_time.ndim == 0:
            diffusion_time = diffusion_time.repeat(x_t.shape[0])
        time_embedding = dit.time_embed(diffusion_time)
        hidden, _ = dit.get_input_embed(
            x_t,
            cond,
            text_tokens,
            midi,
            drop_audio_cond=drop_audio,
            drop_text=drop_text,
            drop_midi=drop_midi,
            cache=False,
        )
        rope = dit.rotary_embed.forward_from_seq_len(x_t.shape[1])
        residual = hidden
        hidden_states = []
        for index, block in enumerate(dit.transformer_blocks):
            if self.checkpoint_activations:

                def block_forward(value, embedding, module=block):
                    return module(value, embedding, mask=None, rope=rope)

                hidden = torch.utils.checkpoint.checkpoint(
                    block_forward,
                    hidden,
                    time_embedding,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden = block(hidden, time_embedding, mask=None, rope=rope)
            if index >= len(dit.transformer_blocks) - 3:
                hidden_states.append(hidden)
        hidden = dit.long_skip_connection(torch.cat((hidden, residual), dim=-1))
        hidden = dit.norm_out(hidden, time_embedding)
        return dit.proj_out(hidden), hidden_states


class ShardedEMA:
    def __init__(self, online, ema, beta, update_after_step):
        self.online = online
        self.ema = ema
        self.beta = float(beta)
        self.update_after_step = int(update_after_step)
        self.step = 0
        self.initted = False
        self._assert_layout()

    def _parameter_pairs(self):
        online = list(self.online.named_parameters())
        ema = list(self.ema.named_parameters())
        if len(online) != len(ema):
            raise RuntimeError("online/EMA parameter count mismatch")
        for (online_name, online_value), (ema_name, ema_value) in zip(online, ema):
            if canonical_name(online_name) != canonical_name(ema_name):
                raise RuntimeError(
                    f"online/EMA parameter name mismatch: {online_name} != {ema_name}"
                )
            if online_value.shape != ema_value.shape:
                raise RuntimeError(
                    f"online/EMA local shard mismatch for {online_name}: "
                    f"{online_value.shape} != {ema_value.shape}"
                )
            yield online_value, ema_value

    def _buffer_pairs(self):
        online = list(self.online.named_buffers())
        ema = list(self.ema.named_buffers())
        if len(online) != len(ema):
            raise RuntimeError("online/EMA buffer count mismatch")
        for (online_name, online_value), (ema_name, ema_value) in zip(online, ema):
            if canonical_name(online_name) != canonical_name(ema_name):
                raise RuntimeError(
                    f"online/EMA buffer name mismatch: {online_name} != {ema_name}"
                )
            yield online_value, ema_value

    def _assert_layout(self):
        for _ in self._parameter_pairs():
            pass
        for _ in self._buffer_pairs():
            pass

    @torch.no_grad()
    def update(self):
        previous_step = self.step
        self.step += 1
        if not self.initted or previous_step <= self.update_after_step:
            decay = 0.0
            self.initted = True
        else:
            epoch = max(self.step - self.update_after_step - 1, 0)
            decay = 0.0 if epoch <= 0 else 1.0 - (1.0 + epoch) ** -(2.0 / 3.0)
            decay = min(max(decay, 0.0), self.beta)
        for online, ema in self._parameter_pairs():
            if decay == 0.0:
                ema.copy_(online)
            else:
                ema.lerp_(online, 1.0 - decay)
        for online, ema in self._buffer_pairs():
            if decay == 0.0 or not (ema.is_floating_point() or ema.is_complex()):
                ema.copy_(online)
            else:
                ema.lerp_(online, 1.0 - decay)


def wrap_fsdp(graph, device, device_mesh):
    block_type = type(graph.singer.transformer.transformer_blocks[0])
    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={block_type},
    )
    graph = graph.to(device)
    return FSDP(
        graph,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        device_mesh=device_mesh,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
        forward_prefetch=False,
    )


def build_contract(args, transition_metadata):
    files = {
        "transition": args.transition_sha256,
        "train_manifest": args.train_manifest_sha256,
        "game_cache_manifest": sha256_file(args.game_cache_manifest),
        "model_config": sha256_file(args.config),
        "vae_config": sha256_file(args.vae_config),
        "vae_checkpoint": sha256_file(args.vae_ckpt),
    }
    return {
        "schema": CHECKPOINT_SCHEMA,
        "architecture": transition_metadata["target_arch"],
        "target_dit_parameters": transition_metadata["target_dit_parameters"],
        "files": files,
        "world_size": dist.get_world_size(),
        "grad_accum": args.grad_accum,
        "effective_batch": dist.get_world_size() * args.grad_accum,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "hold_steps": args.hold_steps,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "max_duration": args.max_duration,
        "flow_b_weight": args.flow_b_weight,
        "cka_weight": args.cka_weight,
        "drop_text": args.drop_text,
        "checkpoint_activations": args.checkpoint_activations,
        "sync_each_microbatch": args.sync_each_microbatch,
        "min_free_gib": args.min_free_gib,
        "precision": "fp32",
        "fsdp": {
            "strategy": "FULL_SHARD",
            "wrap": "each_transformer_block_plus_root",
            "use_orig_params": True,
            "limit_all_gathers": True,
        },
        "ema": {
            "precision": "fp32",
            "storage": "second_isomorphic_fsdp_model",
            "beta": args.ema_beta,
            "update_after_step": args.ema_update_after_step,
        },
    }


def sharded_state_context(model):
    return FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True, _use_dtensor=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True, _use_dtensor=True),
    )


def initialize_adamw_state_for_load(optimizer):
    """Materialize AdamW's lazy tensors so DCP has load destinations."""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        step = state.get("step")
        if torch.is_tensor(step):
            step.zero_()


def save_checkpoint(
    output,
    model,
    optimizer,
    ema_model,
    ema_tracker,
    scheduler,
    contract,
    global_step,
    sampler_epoch,
    batch_offset,
    trace,
    run_state,
):
    save_started = time.perf_counter()
    rank = dist.get_rank()
    output = pathlib.Path(output).resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if rank == 0:
        if temporary.exists():
            shutil.rmtree(temporary)
        if output.exists():
            raise FileExistsError(output)
        temporary.mkdir(parents=True)
    distributed_barrier()

    with sharded_state_context(model):
        model_state = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
    with sharded_state_context(ema_model):
        ema_state = ema_model.state_dict()
    state = {"model": model_state, "optimizer": optimizer_state, "ema": ema_state}
    dcp.save(state, checkpoint_id=temporary / "distcp")
    rank_state = {
        "rank": rank,
        "sampler_epoch": int(sampler_epoch),
        "batch_offset": int(batch_offset),
        "rng": base.capture_rng_state(
            torch.device("cuda", torch.cuda.current_device())
        ),
        "trace": list(trace),
    }
    atomic_torch_save(rank_state, temporary / f"rank_{rank:02d}.pt")
    if rank == 0:
        atomic_torch_save(
            {
                "checkpoint_schema": CHECKPOINT_SCHEMA,
                "run_state": run_state,
                "global_step": int(global_step),
                "contract": contract,
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_step": int(ema_tracker.step),
                "ema_initted": bool(ema_tracker.initted),
                "midi_p_schema": MIDI_P_V4PH_SCHEMA,
                "game_cache_schema": GAME_CACHE_SCHEMA,
            },
            temporary / "metadata.pt",
        )
    distributed_barrier()
    if rank == 0:
        os.replace(temporary, output)
    distributed_barrier()
    result = [None]
    if rank == 0:
        result[0] = {
            "path": str(output),
            "global_step": int(global_step),
            "bytes": sum(
                path.stat().st_size for path in output.rglob("*") if path.is_file()
            ),
            "elapsed_seconds": time.perf_counter() - save_started,
        }
    dist.broadcast_object_list(result, src=0)
    return result[0]


def load_checkpoint(
    checkpoint,
    model,
    optimizer,
    ema_model,
    ema_tracker,
    scheduler,
    contract,
    device,
):
    load_started = time.perf_counter()
    checkpoint = pathlib.Path(checkpoint).resolve()
    metadata = torch.load(
        checkpoint / "metadata.pt", map_location="cpu", weights_only=False
    )
    if metadata.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("M600-B FSDP checkpoint schema mismatch")
    if metadata.get("contract") != contract:
        raise ValueError("M600-B FSDP resume contract mismatch")
    if metadata.get("midi_p_schema") != MIDI_P_V4PH_SCHEMA:
        raise ValueError("MIDI_P schema mismatch")
    if metadata.get("game_cache_schema") != GAME_CACHE_SCHEMA:
        raise ValueError("GAME cache schema mismatch")

    initialize_adamw_state_for_load(optimizer)
    with sharded_state_context(model):
        model_state = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
    with sharded_state_context(ema_model):
        ema_state = ema_model.state_dict()
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "ema": ema_state,
    }
    dcp.load(state, checkpoint_id=checkpoint / "distcp")
    with sharded_state_context(model):
        model.load_state_dict(model_state)
        optimizer_state = FSDP.optim_state_dict_to_load(
            model, optimizer, optimizer_state
        )
        optimizer.load_state_dict(optimizer_state)
    with sharded_state_context(ema_model):
        ema_model.load_state_dict(ema_state)
    scheduler.load_state_dict(metadata["scheduler_state_dict"])
    ema_tracker.step = int(metadata["ema_step"])
    ema_tracker.initted = bool(metadata["ema_initted"])
    rank_state = torch.load(
        checkpoint / f"rank_{dist.get_rank():02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return metadata, rank_state, time.perf_counter() - load_started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", required=True)
    parser.add_argument("--transition_sha256", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--train_manifest_sha256", required=True)
    parser.add_argument("--h_config_fingerprint", required=True)
    parser.add_argument("--game_cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml"
    )
    parser.add_argument(
        "--vae_config",
        default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",
    )
    parser.add_argument(
        "--vae_ckpt", default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt"
    )
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--hold_steps", type=int, default=23500)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--stop_after_step", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--record_every", type=int, default=10)
    parser.add_argument("--empty_cache_every", type=int, default=100)
    parser.add_argument("--min_free_gib", type=float, default=3.0)
    parser.add_argument("--trace_limit", type=int, default=100)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flow_b_weight", type=float, default=2.0)
    parser.add_argument("--cka_weight", type=float, default=0.7)
    parser.add_argument("--drop_text", type=float, default=0.15)
    parser.add_argument("--t_shift", type=float, default=0.5)
    parser.add_argument("--ema_beta", type=float, default=0.995)
    parser.add_argument("--ema_update_after_step", type=int, default=100)
    parser.add_argument("--checkpoint_activations", action="store_true")
    parser.add_argument("--sync_each_microbatch", action="store_true")
    parser.add_argument("--skip_final_checkpoint", action="store_true")
    parser.add_argument("--audit_parameter_updates", action="store_true")
    parser.add_argument("--fingerprint_state", action="store_true")
    parser.add_argument("--expected_world_size", type=int, default=4)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("M600-B FSDP must be launched with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    if world_size != args.expected_world_size:
        raise ValueError(f"world_size {world_size} != {args.expected_world_size}")
    if args.grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if args.min_free_gib <= 0:
        raise ValueError("min_free_gib must be positive")
    if args.trace_limit < 0:
        raise ValueError("trace_limit must be non-negative")
    if args.max_steps != 30000 or args.lr != 1.4e-5:
        raise ValueError("M600-B requires the frozen HighLR 30k contract")
    if args.warmup_steps != 500 or args.hold_steps != 23500:
        raise ValueError("M600-B requires warmup=500 and hold=23500")
    run_until = args.stop_after_step or args.max_steps
    if not 0 < run_until <= args.max_steps:
        raise ValueError("stop_after_step must be in [1, max_steps]")
    if args.resume and not pathlib.Path(args.resume).is_dir():
        raise FileNotFoundError(args.resume)

    started = time.perf_counter()
    base.seed_everything(args.seed)
    actual_transition_sha = sha256_file(args.transition)
    if actual_transition_sha != args.transition_sha256:
        raise ValueError("M600-B transition SHA256 mismatch")
    transition = torch.load(
        args.transition, map_location="cpu", weights_only=False, mmap=True
    )
    if transition.get("checkpoint_schema") != TRANSITION_SCHEMA:
        raise ValueError("M600-B transition schema mismatch")
    transition_metadata = transition["v4m_transition"]
    if int(transition_metadata["target_dit_parameters"]) != TARGET_DIT_PARAMETERS:
        raise ValueError("M600-B transition parameter count mismatch")

    cfg = OmegaConf.load(args.config)
    online_singer = build_singer(
        cfg, TARGET_DEPTH, TARGET_HEADS, TARGET_FF_MULT, args.seed
    )
    strict_load(online_singer, transition["model_state_dict"], "M600-B online")
    ema_singer = build_singer(
        cfg, TARGET_DEPTH, TARGET_HEADS, TARGET_FF_MULT, args.seed
    )
    strict_load(ema_singer, transition["model_state_dict"], "M600-B EMA")
    del transition

    device_mesh = init_device_mesh("cuda", (world_size,))
    online = wrap_fsdp(
        M600BTrainingGraph(online_singer, args.checkpoint_activations),
        device,
        device_mesh,
    )
    del online_singer
    ema_model = wrap_fsdp(M600BTrainingGraph(ema_singer, False), device, device_mesh)
    del ema_singer
    for parameter in ema_model.parameters():
        parameter.requires_grad = False
    ema_tracker = ShardedEMA(
        online,
        ema_model,
        beta=args.ema_beta,
        update_after_step=args.ema_update_after_step,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in online.parameters() if parameter.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=1e-2,
    )

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if step < args.warmup_steps + args.hold_steps:
            return 1.0
        decay_steps = max(1, args.max_steps - args.warmup_steps - args.hold_steps)
        progress = (step - args.warmup_steps - args.hold_steps) / decay_steps
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    contract = build_contract(args, transition_metadata)

    dataset = base.HDataset(
        args.train_manifest,
        args.train_manifest_sha256,
        args.h_config_fingerprint,
        args.max_duration,
        args.game_cache_manifest,
    )
    sampler = base.ResumableDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=base.collate_svs,
        pin_memory=True,
        drop_last=True,
    )
    vae = (
        StableAudioInfer(
            model_config_path=args.vae_config,
            model_ckpt_path=args.vae_ckpt,
        )
        .to(device)
        .eval()
    )
    for parameter in vae.parameters():
        parameter.requires_grad = False

    global_step = 0
    sampler_epoch = 0
    batch_offset = 0
    trace = []
    resume_load_seconds = None
    if args.resume:
        metadata, rank_state, resume_load_seconds = load_checkpoint(
            args.resume,
            online,
            optimizer,
            ema_model,
            ema_tracker,
            scheduler,
            contract,
            device,
        )
        global_step = int(metadata["global_step"])
        sampler_epoch = int(rank_state["sampler_epoch"])
        batch_offset = int(rank_state["batch_offset"])
        trace = list(rank_state["trace"])
        if global_step > run_until:
            raise ValueError("resume step exceeds stop_after_step")

    consumed_batches = global_step * args.grad_accum
    if consumed_batches == 0:
        expected_cursor = (0, 0)
    else:
        expected_epoch = (consumed_batches - 1) // sampler.num_samples
        expected_offset = consumed_batches - expected_epoch * sampler.num_samples
        expected_cursor = (expected_epoch, expected_offset)
    if (sampler_epoch, batch_offset) != expected_cursor:
        raise ValueError(
            f"resume cursor {(sampler_epoch, batch_offset)} != {expected_cursor}"
        )

    base.seed_everything(args.seed + rank)
    sampler.set_epoch_and_offset(sampler_epoch, batch_offset)
    data_iterator = iter(loader)
    if args.resume:
        base.restore_rng_state(rank_state["rng"], device)

    initial_parameters = None
    if args.audit_parameter_updates:
        initial_parameters = {
            canonical_name(name): parameter.detach().clone()
            for name, parameter in online.named_parameters()
        }

    def compute_ref_len(total_frames, phrase_boundaries=None):
        five_sec_frames = int(5.0 * base.FRAME_RATE)
        ref_len = total_frames * random.uniform(0.125, 0.33)
        ref_len = max(ref_len, five_sec_frames)
        ref_len = min(ref_len, int(total_frames * 0.65))
        if phrase_boundaries:
            margin = int(2.5 * base.FRAME_RATE)
            candidates = [
                boundary
                for boundary in phrase_boundaries
                if boundary > 0 and abs(boundary - ref_len) <= margin
            ]
            if candidates:
                ref_len = min(candidates, key=lambda item: abs(item - ref_len))
        return int(ref_len)

    def process_batch(batch):
        with torch.no_grad():
            wav = batch["wav"][0].to(device)
            sample_rate = batch["sr"][0]
            phrases = batch["phrases"][0]
            tier = batch["tier"][0]
            candidates = batch["h_candidates"][0]
            game_cache = batch["game_cache"][0]
            wav_2d = wav.unsqueeze(0) if wav.dim() == 1 else wav
            latent = vae.encode_audio(wav_2d, in_sr=sample_rate)
            full_latent = latent.squeeze(0).transpose(0, 1).unsqueeze(0)
            total_frames = full_latent.shape[1]
            boundaries = (
                [int(phrase["start"] * base.FRAME_RATE) for phrase in phrases]
                if tier in ("L1", "L2")
                else None
            )
            ref_len = compute_ref_len(total_frames, boundaries)
            cond = torch.zeros_like(full_latent)
            cond[:, :ref_len] = full_latent[:, :ref_len]
            tracks = game_cache_to_model_tracks(
                game_cache,
                num_samples=wav_2d.shape[-1],
                target_len=total_frames,
                sample_rate=sample_rate,
            )
            p_classes = tracks["p_classes"].unsqueeze(0).to(device)
            midi_probs = tracks["cka_probs"].unsqueeze(0).to(device)
            paired = base.render_h_pul_placements(
                phrases,
                candidates,
                ref_len=ref_len,
                total_frames=total_frames,
                sep_token_id=base.SEP_TOKEN,
                pul_token_id=base.PUL_TOKEN,
            )
            aligned_text = torch.tensor(
                paired["phone_pul"]["text"], dtype=torch.long, device=device
            ).unsqueeze(0)
        return full_latent, cond, p_classes, midi_probs, aligned_text, ref_len

    def loss_for_batch(batch):
        full_latent, cond, p_classes, midi_probs, text, ref_len = process_batch(batch)
        u = torch.rand(1, device=device)
        diffusion_time = args.t_shift * u / (1.0 - args.t_shift * u)
        noise = torch.randn_like(full_latent)
        x_t = (1.0 - diffusion_time[:, None, None]) * noise + diffusion_time[
            :, None, None
        ] * full_latent
        target = full_latent - noise
        drops = (
            random.random() < 0.3,
            random.random() < args.drop_text,
            random.random() < 0.3,
        )
        prediction, hidden_states = online(
            x_t,
            cond,
            text,
            diffusion_time,
            p_classes,
            ref_len,
            *drops,
        )
        flow_a = F.mse_loss(prediction[:, :ref_len], target[:, :ref_len])
        flow_b = F.mse_loss(prediction[:, ref_len:], target[:, ref_len:])
        cka = base.compute_cka_loss_from_hidden(
            [hidden[:, ref_len:] for hidden in hidden_states],
            midi_probs[:, ref_len:],
        )
        return flow_a + args.flow_b_weight * flow_b + args.cka_weight * cka, {
            "flow_a": float(flow_a.detach()),
            "flow_b": float(flow_b.detach()),
            "cka": float(cka.detach()),
        }

    output_dir = pathlib.Path(args.output_dir).resolve()
    if rank == 0:
        if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"fresh output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[M600-B FSDP] world={world_size} params={TARGET_DIT_PARAMETERS} "
            f"checkpoint_activations={args.checkpoint_activations} "
            f"sync_each_microbatch={args.sync_each_microbatch} "
            f"step={global_step}->{run_until}",
            flush=True,
        )
    distributed_barrier()

    torch.cuda.reset_peak_memory_stats()
    events = [memory_event("initialized", update=global_step)]
    step_times = []
    checkpoint_records = []
    duration_sum = 0.0
    duration_count = 0
    log_loss = 0.0
    log_flow_a = 0.0
    log_flow_b = 0.0
    log_cka = 0.0
    last_global_min_free_gib = None
    run_started = time.perf_counter()
    while global_step < run_until:
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        micro_traces = []
        for microbatch in range(1, args.grad_accum + 1):
            try:
                batch = next(data_iterator)
            except StopIteration:
                sampler_epoch += 1
                batch_offset = 0
                sampler.set_epoch_and_offset(sampler_epoch, batch_offset)
                data_iterator = iter(loader)
                batch = next(data_iterator)
            batch_offset += 1
            duration = float(batch["duration"][0])
            duration_sum += duration
            duration_count += 1
            should_sync = args.sync_each_microbatch or microbatch == args.grad_accum
            sync_context = contextlib.nullcontext() if should_sync else online.no_sync()
            with sync_context:
                loss, components = loss_for_batch(batch)
                finite_loss = torch.isfinite(loss).to(dtype=torch.int32)
                dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
                if int(finite_loss.item()) != 1:
                    raise FloatingPointError(
                        f"non-finite loss at update={global_step + 1} "
                        f"microbatch={microbatch}"
                    )
                (loss / args.grad_accum).backward()
            log_loss += float(loss.detach())
            log_flow_a += components["flow_a"]
            log_flow_b += components["flow_b"]
            log_cka += components["cka"]
            micro_traces.append(
                {
                    "sample_id": batch["sample_id"][0],
                    "duration": duration,
                    "loss": float(loss.detach()),
                    **components,
                }
            )
        gradient_norm = online.clip_grad_norm_(1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite gradient norm at update={global_step + 1}"
            )
        optimizer.step()
        scheduler.step()
        ema_tracker.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        torch.cuda.synchronize()
        step_time = time.perf_counter() - step_started
        step_times.append(step_time)
        trace.append({"update": global_step, "microbatches": micro_traces})
        if len(trace) > args.trace_limit:
            trace = trace[-args.trace_limit :] if args.trace_limit else []

        recorded_event = None
        if (
            global_step == run_until
            or global_step == 1
            or global_step % args.record_every == 0
        ):
            recorded_event = memory_event(
                "after_update", update=global_step, elapsed_seconds=step_time
            )
            events.append(recorded_event)
            free_gib = torch.tensor(
                recorded_event["device_free_bytes"] / (1024**3),
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(free_gib, op=dist.ReduceOp.MIN)
            last_global_min_free_gib = float(free_gib.item())
            if last_global_min_free_gib < args.min_free_gib:
                raise RuntimeError(
                    f"physical GPU free memory {last_global_min_free_gib:.3f}GiB "
                    f"is below safety floor {args.min_free_gib:.3f}GiB"
                )
        if args.empty_cache_every > 0 and global_step % args.empty_cache_every == 0:
            events.append(memory_event("before_empty_cache", update=global_step))
            torch.cuda.empty_cache()
            events.append(memory_event("after_empty_cache", update=global_step))
        if global_step % args.log_every == 0:
            log_values = torch.tensor(
                [log_loss, log_flow_a, log_flow_b, log_cka],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(log_values, op=dist.ReduceOp.SUM)
            divisor = args.log_every * args.grad_accum * world_size
            mean_so_far = sum(step_times) / len(step_times)
            if rank == 0:
                free_text = (
                    f"{last_global_min_free_gib:.3f}GiB"
                    if last_global_min_free_gib is not None
                    else "unrecorded"
                )
                print(
                    f"[M600-B FSDP] step={global_step} "
                    f"mean={mean_so_far:.3f}s "
                    f"Loss={log_values[0].item() / divisor:.6f} "
                    f"FlowA={log_values[1].item() / divisor:.6f} "
                    f"FlowB={log_values[2].item() / divisor:.6f} "
                    f"CKA={log_values[3].item() / divisor:.6f} "
                    f"max_reserved={torch.cuda.max_memory_reserved() / (1024**3):.3f}GiB "
                    f"min_free={free_text}",
                    flush=True,
                )
            log_loss = log_flow_a = log_flow_b = log_cka = 0.0
        if (
            args.save_every > 0
            and global_step % args.save_every == 0
            and global_step < run_until
        ):
            checkpoint_records.append(
                save_checkpoint(
                    output_dir / f"step_{global_step:06d}.dcp",
                    online,
                    optimizer,
                    ema_model,
                    ema_tracker,
                    scheduler,
                    contract,
                    global_step,
                    sampler_epoch,
                    batch_offset,
                    trace,
                    "running",
                )
            )

    torch.cuda.synchronize()
    local_changed = []
    if initial_parameters is not None:
        for name, parameter in online.named_parameters():
            name = canonical_name(name)
            if not torch.equal(parameter.detach(), initial_parameters[name]):
                local_changed.append(name)

    fingerprints = None
    if args.fingerprint_state:
        fingerprints = {
            "model": module_local_fingerprint(online),
            "ema": module_local_fingerprint(ema_model),
            "optimizer": optimizer_local_fingerprint(optimizer),
            "scheduler": hashlib.sha256(
                json.dumps(scheduler.state_dict(), sort_keys=True, default=str).encode(
                    "utf-8"
                )
            ).hexdigest(),
        }

    local_report = {
        "rank": rank,
        "device": local_rank,
        "events": events,
        "step_times": step_times,
        "step_time_summary": timing_summary(step_times),
        "steady_step_time_summary": timing_summary(step_times[10:]),
        "duration_count": duration_count,
        "duration_mean": duration_sum / duration_count if duration_count else None,
        "trace": trace,
        "changed_parameter_names": local_changed,
        "fingerprints": fingerprints,
    }
    rank_reports = [None] * world_size
    dist.all_gather_object(rank_reports, local_report)

    if not args.skip_final_checkpoint:
        checkpoint_records.append(
            save_checkpoint(
                output_dir / f"step_{global_step:06d}_final.dcp",
                online,
                optimizer,
                ema_model,
                ema_tracker,
                scheduler,
                contract,
                global_step,
                sampler_epoch,
                batch_offset,
                trace,
                "complete" if global_step >= args.max_steps else "stopped",
            )
        )

    if rank == 0:
        global_step_times = [
            max(report["step_times"][index] for report in rank_reports)
            for index in range(len(step_times))
        ]
        steady = global_step_times[10:]
        eta_basis = steady or global_step_times
        eta_mean = sum(eta_basis) / len(eta_basis)
        changed_union = sorted(
            {
                name
                for report in rank_reports
                for name in report["changed_parameter_names"]
            }
        )
        changed_blocks = sorted(
            {
                int(match.group(1))
                for name in changed_union
                if (match := re.search(r"transformer_blocks\.(\d+)\.", name))
            }
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "ok",
            "contract": contract,
            "start_step": global_step - len(step_times),
            "end_step": global_step,
            "updates_this_run": len(step_times),
            "elapsed_seconds": time.perf_counter() - started,
            "training_elapsed_seconds": time.perf_counter() - run_started,
            "resume_load_seconds": resume_load_seconds,
            "checkpoint_records": checkpoint_records,
            "global_step_time_summary": timing_summary(global_step_times),
            "global_steady_step_time_summary": timing_summary(steady),
            "projected_hours": {
                str(target): target * eta_mean / 3600.0
                for target in (6000, 12000, 30000)
            },
            "changed_blocks": changed_blocks,
            "changed_parameter_count": len(changed_union),
            "ranks": rank_reports,
        }
        report_path = output_dir / "run_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "global_step_time_summary": report["global_step_time_summary"],
                    "global_steady_step_time_summary": report[
                        "global_steady_step_time_summary"
                    ],
                    "projected_hours": report["projected_hours"],
                    "changed_blocks": changed_blocks,
                },
                indent=2,
            ),
            flush=True,
        )

    distributed_barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
