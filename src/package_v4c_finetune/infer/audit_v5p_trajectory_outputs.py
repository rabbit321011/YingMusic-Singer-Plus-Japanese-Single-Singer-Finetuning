#!/usr/bin/env python3
"""Audit the named V5-P 2k trajectory outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import soundfile as sf


WEIGHT_SOURCES = ("EMA", "RAW")


def trajectory_contract(model_family):
    if model_family == "V5PG":
        return {
            "steps": tuple(range(1000, 10001, 1000)),
            "prefix": "V5PG",
            "final_step": 10000,
            "report_schema": "v5pg_trajectory_named_audit_v1",
        }
    return {
        "steps": tuple(range(2000, 40001, 2000)),
        "prefix": "V5P",
        "final_step": 40000,
        "report_schema": "v5p_trajectory_named_audit_v1",
    }


def safe_name(value):
    value = re.sub(r'[<>:"/\\|?*]', "_", str(value)).strip().rstrip(".")
    return value or "unnamed"


def condition_label(step, weight_source, prefix="V5P"):
    return f"{prefix}_{step // 1000:02d}K_{weight_source}_cfg1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_wav(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 44100 or audio.shape[1] != 1:
        raise ValueError(f"invalid WAV format: {path}: {sample_rate}, {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite WAV: {path}")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if rms <= 1e-5 or peak <= 1e-4:
        raise ValueError(f"silent WAV: {path}: rms={rms} peak={peak}")
    return audio.shape[0], rms, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-per-dir", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-family", choices=("V5P", "V5PG"), default="V5P")
    args = parser.parse_args()

    root = args.root.resolve()
    dataset = args.dataset.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    contract = trajectory_contract(args.model_family)
    steps = contract["steps"]
    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    groups = (dataset_manifest.get("groups") or [])[: args.expected_per_dir]
    group_by_safe_name = {safe_name(item["name"]): item for item in groups}
    if len(group_by_safe_name) != args.expected_per_dir:
        raise ValueError("evaluation group names are not unique after sanitization")

    expected_dirs = [
        condition_label(step, weight_source, contract["prefix"])
        for step in steps
        for weight_source in WEIGHT_SOURCES
    ]
    actual_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    if sorted(expected_dirs) != actual_dirs:
        raise ValueError("trajectory output directory set mismatch")

    records = []
    common_conditions = {}
    wav_hashes = {}
    for step in steps:
        checkpoint_name = (
            f"step_{step:06d}_final.pt"
            if step == contract["final_step"]
            else f"step_{step:06d}.pt"
        )
        checkpoint = checkpoint_dir / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for weight_source in WEIGHT_SOURCES:
            directory_name = condition_label(step, weight_source, contract["prefix"])
            directory = root / directory_name
            audit_dir = directory / "_placement"
            wavs = sorted(directory.glob("*.wav"))
            audits = sorted(audit_dir.glob("*.json"))
            if len(wavs) != args.expected_per_dir or len(audits) != args.expected_per_dir:
                raise ValueError(f"file count mismatch: {directory_name}")
            if {path.stem for path in wavs} != set(group_by_safe_name):
                raise ValueError(f"WAV group mismatch: {directory_name}")
            if {path.stem for path in audits} != set(group_by_safe_name):
                raise ValueError(f"audit group mismatch: {directory_name}")

            for wav_path in wavs:
                group = group_by_safe_name[wav_path.stem]
                audit_path = audit_dir / f"{wav_path.stem}.json"
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                game = audit.get("game") or {}
                if audit.get("group") != group["name"]:
                    raise ValueError(f"audit group mismatch: {audit_path}")
                if audit.get("cfg") != 1.0 or audit.get("steps") != 32 or audit.get("seed") != 42:
                    raise ValueError(f"sampling contract mismatch: {audit_path}")
                if audit.get("weightSource") != weight_source.lower():
                    raise ValueError(f"weight source mismatch: {audit_path}")
                if Path(audit.get("checkpoint", "")).resolve() != checkpoint.resolve():
                    raise ValueError(f"checkpoint mismatch: {audit_path}")
                if game.get("targetPadFrames") != 0 or game.get("promptMidiNonzero") != 0:
                    raise ValueError(f"MIDI mask/PAD failure: {audit_path}")
                if abs(int(game.get("boundaryFrameDelta", 99))) > 1:
                    raise ValueError(f"GAME/VAE boundary failure: {audit_path}")

                samples, rms, peak = load_wav(wav_path)
                b_path = dataset / group["directory"] / "B.wav"
                b_info = sf.info(b_path)
                duration_delta = abs(samples / 44100.0 - b_info.frames / b_info.samplerate)
                if duration_delta > 0.025:
                    raise ValueError(f"duration mismatch {duration_delta:.6f}s: {wav_path}")

                input_condition = (
                    audit.get("denseTextSHA256"),
                    game.get("effectiveSeed"),
                    game.get("combinedWaveformSHA256"),
                    game.get("pClassSHA256"),
                    game.get("notes"),
                    game.get("restNotes"),
                    game.get("sampleBoundaryFrame"),
                    game.get("vaeBoundaryFrame"),
                )
                prior = common_conditions.setdefault(group["name"], input_condition)
                if prior != input_condition:
                    raise ValueError(f"cross-condition input mismatch: {group['name']}")

                wav_hash = sha256_file(wav_path)
                pair_key = (step, group["name"])
                prior_hash = wav_hashes.setdefault(pair_key, wav_hash)
                if weight_source == "RAW" and prior_hash == wav_hash:
                    raise ValueError(f"EMA/raw outputs are byte-identical: {pair_key}")
                records.append(
                    {
                        "step": step,
                        "weight_source": weight_source.lower(),
                        "directory": directory_name,
                        "group": group["name"],
                        "wav": str(wav_path),
                        "wav_sha256": wav_hash,
                        "samples": samples,
                        "duration_delta_seconds": duration_delta,
                        "rms": rms,
                        "peak": peak,
                        "checkpoint": str(checkpoint),
                    }
                )

    if not all(math.isfinite(item["rms"]) and math.isfinite(item["peak"]) for item in records):
        raise ValueError("non-finite audio statistics")
    report = {
        "schema": contract["report_schema"],
        "status": "ok",
        "smoke": args.smoke,
        "root": str(root),
        "dataset": str(dataset),
        "model_family": args.model_family,
        "steps": list(steps),
        "weight_sources": [value.lower() for value in WEIGHT_SOURCES],
        "cfg": 1.0,
        "sampling_steps": 32,
        "seed": 42,
        "directories": expected_dirs,
        "expected_per_directory": args.expected_per_dir,
        "wav_count": len(records),
        "max_duration_delta_seconds": max(item["duration_delta_seconds"] for item in records),
        "minimum_rms": min(item["rms"] for item in records),
        "maximum_peak": max(item["peak"] for item in records),
        "input_condition_mismatches": 0,
        "ema_raw_identical_wav_pairs": 0,
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
