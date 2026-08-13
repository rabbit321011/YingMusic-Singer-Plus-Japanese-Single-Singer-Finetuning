"""Run an isolated single-rank V4IjPH update with CUDA memory checkpoints."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch

import train_v4ijph as train


OUTPUT = pathlib.Path(os.environ["V4IJPH_MEMORY_PROBE_OUTPUT"])
EVENTS: list[dict] = []
MICROBATCH = 0
DISABLE_CKA = os.environ.get("V4IJPH_MEMORY_PROBE_DISABLE_CKA") == "1"

if DISABLE_CKA:
    def zero_cka(hidden, midi_probs):
        return torch.zeros((), device=midi_probs.device, dtype=midi_probs.dtype)

    train.compute_cka_loss_from_hidden = zero_cka


def record(stage: str, **extra) -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        fields = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    else:
        fields = {}
    event = {"stage": stage, "time": time.time(), **fields, **extra}
    EVENTS.append(event)
    print(f"[memory-probe] {json.dumps(event, sort_keys=True)}", flush=True)


original_load_source = train.load_and_validate_source


def load_four_rank_source(args, _world_size):
    source = original_load_source(args, 4)
    manifest_size = len(train.load_h_manifest(pathlib.Path(args.train_manifest).resolve()))
    consumed_batches = int(source["global_step"]) * int(args.grad_accum)
    epoch = (consumed_batches - 1) // manifest_size
    offset = consumed_batches - epoch * manifest_size
    rank_state = dict(source["rank_states"][0])
    rank_state.update(rank=0, sampler_epoch=epoch, batch_offset=offset)
    source = dict(source)
    source["rank_states"] = [rank_state]
    return source


train.load_and_validate_source = load_four_rank_source

original_compute_loss = train.V4IjPHObjective.compute_loss


def measured_compute_loss(self, *args, **kwargs):
    global MICROBATCH
    MICROBATCH += 1
    if MICROBATCH == 1:
        torch.cuda.reset_peak_memory_stats()
        record("initialized")
    batch = args[0]
    sample = {
        "sample_id": batch["sample_id"][0],
        "duration": float(batch["duration"][0]),
    }
    record("before_forward", microbatch=MICROBATCH, **sample)
    result = original_compute_loss(self, *args, **kwargs)
    record("after_forward", microbatch=MICROBATCH)
    return result


train.V4IjPHObjective.compute_loss = measured_compute_loss

original_backward = torch.Tensor.backward


def measured_backward(self, *args, **kwargs):
    result = original_backward(self, *args, **kwargs)
    record("after_backward", microbatch=MICROBATCH)
    return result


torch.Tensor.backward = measured_backward

original_adamw_step = torch.optim.AdamW.step
OPTIMIZER_STEP = 0


def measured_adamw_step(self, *args, **kwargs):
    global OPTIMIZER_STEP
    result = original_adamw_step(self, *args, **kwargs)
    OPTIMIZER_STEP += 1
    record("after_optimizer_step", optimizer_index=OPTIMIZER_STEP)
    return result


torch.optim.AdamW.step = measured_adamw_step


def suppress_save(_payload, path):
    record("checkpoint_suppressed", path=str(path))


train.atomic_torch_save = suppress_save


def write_report(status: str, error: str | None = None) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "v4ijph_single_rank_memory_probe_v1",
        "status": status,
        "error": error,
        "microbatches": MICROBATCH,
        "cka_disabled": DISABLE_CKA,
        "events": EVENTS,
    }
    (OUTPUT / "memory_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


try:
    train.main()
except BaseException as exc:
    record("failed", error=f"{type(exc).__name__}: {exc}")
    write_report("failed", f"{type(exc).__name__}: {exc}")
    raise
else:
    record("complete")
    write_report("ok")
