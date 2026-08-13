#!/usr/bin/env python3
"""Audit HighLR/M600-D same-step listening outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import soundfile as sf


EXPECTED_LABELS = ("HIGHLR_06K", "HIGHLR_12K", "M600D_06K", "M600D_12K")
EXPECTED_CFGS = ("cfg3", "cfg1")


def safe_name(value):
    value = re.sub(r'[<>:"/\\|?*]', "_", str(value)).strip().rstrip(".")
    return value or "unnamed"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
    parser.add_argument("--expected-per-dir", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--labels", nargs="+", default=list(EXPECTED_LABELS))
    args = parser.parse_args()

    root = args.root.resolve()
    dataset = args.dataset.resolve()
    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    all_groups = dataset_manifest.get("groups") or []
    groups = all_groups[: args.expected_per_dir]
    group_by_safe_name = {safe_name(item["name"]): item for item in groups}
    if len(group_by_safe_name) != args.expected_per_dir:
        raise ValueError("evaluation group names are not unique after sanitization")

    expected_dirs = []
    for label in args.labels:
        cfgs = ("cfg3",) if args.smoke else EXPECTED_CFGS
        expected_dirs.extend(f"{label}_{cfg}" for cfg in cfgs)
    actual_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    if sorted(expected_dirs) != actual_dirs:
        raise ValueError(f"output directory set mismatch: {actual_dirs}")

    records = []
    conditions_by_group = {}
    midi_by_checkpoint_group = {}
    wav_hashes = {}
    for directory_name in expected_dirs:
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

        label, cfg_name = directory_name.rsplit("_", 1)
        expected_cfg = 3.0 if cfg_name == "cfg3" else 1.0
        for wav_path in wavs:
            group = group_by_safe_name[wav_path.stem]
            audit_path = audit_dir / f"{wav_path.stem}.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            game = audit.get("game") or {}
            if audit.get("group") != group["name"]:
                raise ValueError(f"audit group mismatch: {audit_path}")
            if audit.get("cfg") != expected_cfg or audit.get("steps") != 32:
                raise ValueError(f"sampling contract mismatch: {audit_path}")
            if audit.get("seed") != 42:
                raise ValueError(f"seed mismatch: {audit_path}")
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
            prior = conditions_by_group.setdefault(group["name"], input_condition)
            if prior != input_condition:
                raise ValueError(f"cross-condition input mismatch: {group['name']}")

            checkpoint = audit.get("checkpoint")
            midi_key = (checkpoint, group["name"])
            midi_hash = game.get("midiEmbeddingSHA256")
            prior_midi = midi_by_checkpoint_group.setdefault(midi_key, midi_hash)
            if prior_midi != midi_hash:
                raise ValueError(f"CFG MIDI mismatch: {midi_key}")

            wav_hash = sha256_file(wav_path)
            wav_key = (label, group["name"])
            if wav_key in wav_hashes and wav_hashes[wav_key] == wav_hash:
                raise ValueError(f"CFG outputs are byte-identical: {wav_key}")
            wav_hashes[wav_key] = wav_hash
            records.append(
                {
                    "directory": directory_name,
                    "group": group["name"],
                    "wav": str(wav_path),
                    "wav_sha256": wav_hash,
                    "samples": samples,
                    "duration_delta_seconds": duration_delta,
                    "rms": rms,
                    "peak": peak,
                    "checkpoint": checkpoint,
                }
            )

    if not all(math.isfinite(item["rms"]) and math.isfinite(item["peak"]) for item in records):
        raise ValueError("non-finite audio statistics")
    report = {
        "schema": "v4m_m600d_same_step_listen_audit_v1",
        "status": "ok",
        "smoke": args.smoke,
        "root": str(root),
        "dataset": str(dataset),
        "directories": expected_dirs,
        "expected_per_directory": args.expected_per_dir,
        "wav_count": len(records),
        "max_duration_delta_seconds": max(item["duration_delta_seconds"] for item in records),
        "minimum_rms": min(item["rms"] for item in records),
        "maximum_peak": max(item["peak"] for item in records),
        "input_condition_mismatches": 0,
        "cfg_midi_mismatches": 0,
        "cfg_identical_wav_pairs": 0,
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
