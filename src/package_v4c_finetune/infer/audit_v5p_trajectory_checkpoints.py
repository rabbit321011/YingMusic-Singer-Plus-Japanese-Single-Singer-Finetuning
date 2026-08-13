#!/usr/bin/env python3
"""Audit every saved V5-P 2k checkpoint before trajectory inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def trajectory_contract(model_family):
    if model_family == "V5PG":
        return {
            "steps": tuple(range(1000, 10001, 1000)),
            "checkpoint_schema": "v5pg_training_checkpoint_v1",
            "phase": "g_adapt",
            "max_steps": 10000,
            "ema_offset": 40000,
            "report_schema": "v5pg_trajectory_checkpoint_audit_v1",
        }
    return {
        "steps": tuple(range(2000, 40001, 2000)),
        "checkpoint_schema": "v5p_training_checkpoint_v1",
        "phase": "joint",
        "max_steps": 40000,
        "ema_offset": 0,
        "report_schema": "v5p_trajectory_checkpoint_audit_v1",
    }


def checkpoint_path(root, step, final_step):
    name = f"step_{step:06d}_final.pt" if step == final_step else f"step_{step:06d}.pt"
    return root / name


def normalized_keys(state, ema):
    keys = set()
    for key in state:
        key = key.replace("module.", "")
        if ema:
            key = key.replace("ema_model.", "")
        if key not in {"initted", "step"}:
            keys.add(key)
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-family", choices=("V5P", "V5PG"), default="V5P")
    args = parser.parse_args()

    import torch

    checkpoint_dir = args.checkpoint_dir.resolve()
    contract = trajectory_contract(args.model_family)
    steps = contract["steps"]
    final_step = contract["max_steps"]
    records = []
    reference_keys = None
    for step in steps:
        path = checkpoint_path(checkpoint_dir, step, final_step)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        expected_run_state = "complete" if step == final_step else "running"
        if payload.get("checkpoint_schema") != contract["checkpoint_schema"]:
            raise ValueError(f"checkpoint schema mismatch: {path}")
        if int(payload.get("global_step", -1)) != step:
            raise ValueError(f"global step mismatch: {path}")
        if payload.get("run_state") != expected_run_state:
            raise ValueError(f"run_state mismatch: {path}")
        expected_ema_step = contract["ema_offset"] + step
        if int(payload.get("ema_step", -1)) != expected_ema_step or not bool(
            payload.get("ema_initted")
        ):
            raise ValueError(f"EMA metadata mismatch: {path}")
        metadata = payload.get("v5p_training") or {}
        if (
            metadata.get("schema") != contract["checkpoint_schema"]
            or metadata.get("phase") != contract["phase"]
            or metadata.get("max_steps") != contract["max_steps"]
        ):
            raise ValueError(f"training metadata mismatch: {path}")
        raw = payload.get("model_state_dict")
        ema = payload.get("ema_model_state_dict")
        if not isinstance(raw, dict) or not isinstance(ema, dict):
            raise ValueError(f"raw/EMA state missing: {path}")
        raw_keys = normalized_keys(raw, ema=False)
        ema_keys = normalized_keys(ema, ema=True)
        if raw_keys != ema_keys:
            raise ValueError(f"raw/EMA normalized key mismatch: {path}")
        if reference_keys is None:
            reference_keys = raw_keys
        elif raw_keys != reference_keys:
            raise ValueError(f"cross-checkpoint key mismatch: {path}")
        records.append(
            {
                "step": step,
                "path": str(path),
                "bytes": path.stat().st_size,
                "run_state": expected_run_state,
                "ema_step": expected_ema_step,
                "normalized_state_keys": len(raw_keys),
            }
        )
        del payload, raw, ema

    report = {
        "schema": contract["report_schema"],
        "status": "ok",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_count": len(records),
        "model_family": args.model_family,
        "steps": list(steps),
        "raw_and_ema_present": True,
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
