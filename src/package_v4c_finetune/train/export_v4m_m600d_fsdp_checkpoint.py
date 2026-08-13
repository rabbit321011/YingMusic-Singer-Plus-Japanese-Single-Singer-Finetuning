"""Merge an M600-D distributed checkpoint into a normal publish checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src").is_dir():
    YING_REPO = PROJECT_DIR
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(YING_REPO))

import train_v4m_m600d_fsdp as train  # noqa: E402
from prepare_v4m_m600d_transition import (  # noqa: E402
    TARGET_DEPTH,
    TARGET_DIT_PARAMETERS,
    build_singer,
    sha256_file,
    strict_load,
    transformer_parameter_count,
)


PUBLISH_SCHEMA = "v4m_m600d_publish_checkpoint_v1"


def full_state_context(model):
    return FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )


def singer_state_dict(graph_state):
    prefix = "singer."
    unexpected = sorted(key for key in graph_state if not key.startswith(prefix))
    if unexpected:
        raise ValueError(f"published graph has unexpected keys: {unexpected[:8]}")
    return {key[len(prefix) :]: value for key, value in graph_state.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_world_size", type=int, default=4)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("M600-D export must be launched with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise ValueError(f"world_size {world_size} != {args.expected_world_size}")
    device = torch.device("cuda", local_rank)
    mesh = init_device_mesh("cuda", (world_size,))

    checkpoint = pathlib.Path(args.checkpoint).resolve()
    metadata = torch.load(
        checkpoint / "metadata.pt", map_location="cpu", weights_only=False
    )
    if metadata.get("checkpoint_schema") != train.CHECKPOINT_SCHEMA:
        raise ValueError("M600-D FSDP checkpoint schema mismatch")
    contract = metadata.get("contract", {})
    if int(contract.get("target_dit_parameters", -1)) != TARGET_DIT_PARAMETERS:
        raise ValueError("M600-D target parameter count mismatch")

    cfg = OmegaConf.load(args.config)
    online_singer = build_singer(cfg, TARGET_DEPTH, args.seed)
    ema_singer = build_singer(cfg, TARGET_DEPTH, args.seed)
    online = train.wrap_fsdp(
        train.M600DTrainingGraph(online_singer, False), device, mesh
    )
    ema = train.wrap_fsdp(train.M600DTrainingGraph(ema_singer, False), device, mesh)
    del online_singer, ema_singer

    with train.sharded_state_context(online):
        online_sharded = online.state_dict()
    with train.sharded_state_context(ema):
        ema_sharded = ema.state_dict()
    state = {"model": online_sharded, "ema": ema_sharded}
    dcp.load(state, checkpoint_id=checkpoint / "distcp")
    with train.sharded_state_context(online):
        online.load_state_dict(online_sharded)
    with train.sharded_state_context(ema):
        ema.load_state_dict(ema_sharded)

    with full_state_context(online):
        online_full = online.state_dict()
    with full_state_context(ema):
        ema_full = ema.state_dict()

    if rank == 0:
        online_state = singer_state_dict(online_full)
        ema_state = singer_state_dict(ema_full)
        if online_state.keys() != ema_state.keys():
            raise ValueError("online/EMA published keys differ")
        nonfinite = []
        for label, state_dict in (("model", online_state), ("ema", ema_state)):
            for name, value in state_dict.items():
                if (
                    value.is_floating_point() or value.is_complex()
                ) and not torch.isfinite(value).all():
                    nonfinite.append(f"{label}:{name}")
        if nonfinite:
            raise ValueError(f"non-finite published tensors: {nonfinite[:8]}")

        probe = build_singer(cfg, TARGET_DEPTH, args.seed)
        strict_load(probe, online_state, "published online")
        strict_load(probe, ema_state, "published EMA")
        dit_trainable_parameters = sum(
            parameter.numel() for parameter in probe.transformer.parameters()
        )
        dit_state_elements = transformer_parameter_count(online_state)
        if dit_state_elements != TARGET_DIT_PARAMETERS:
            raise ValueError(
                f"published DiT state elements {dit_state_elements} != "
                f"{TARGET_DIT_PARAMETERS}"
            )
        del probe

        output = pathlib.Path(args.output).resolve()
        payload = {
            "checkpoint_schema": PUBLISH_SCHEMA,
            "run_state": metadata["run_state"],
            "global_step": int(metadata["global_step"]),
            "model_state_dict": online_state,
            "ema_model_state_dict": ema_state,
            "ema_step": int(metadata["ema_step"]),
            "ema_initted": bool(metadata["ema_initted"]),
            "v4m_training": contract,
            "source_fsdp_checkpoint": str(checkpoint),
            "source_checkpoint_schema": train.CHECKPOINT_SCHEMA,
            "target_dit_state_elements": dit_state_elements,
            "target_dit_trainable_parameters": dit_trainable_parameters,
        }
        train.atomic_torch_save(payload, output)
        output_sha = sha256_file(output)
        audit = {
            "schema": PUBLISH_SCHEMA,
            "checkpoint": str(output),
            "sha256": output_sha,
            "bytes": output.stat().st_size,
            "global_step": payload["global_step"],
            "tensor_count": len(online_state),
            "online_strict_load": True,
            "ema_strict_load": True,
            "all_finite": True,
            "target_dit_state_elements": dit_state_elements,
            "target_dit_trainable_parameters": dit_trainable_parameters,
        }
        output.with_suffix(output.suffix + ".audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
        )
        output.with_suffix(output.suffix + ".sha256").write_text(
            f"{output_sha}  {output.name}\n", encoding="ascii"
        )
        print(json.dumps(audit, indent=2, sort_keys=True), flush=True)

    train.distributed_barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
