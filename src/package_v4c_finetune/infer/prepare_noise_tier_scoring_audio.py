#!/usr/bin/env python3
"""Build the canonical, linearly RMS-matched audio pool for scorer tiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


TARGET_RMS_DBFS = -28.0
PEAK_CEILING_DBFS = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target-rms-dbfs", type=float, default=TARGET_RMS_DBFS)
    parser.add_argument("--peak-ceiling-dbfs", type=float, default=PEAK_CEILING_DBFS)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def original_rows(root: Path) -> Iterable[dict[str, Any]]:
    for row in load_json(root / "original_pool.json"):
        sample_id = str(row["sample_id"])
        yield {
            "item_id": f"original_{int(row['pool_index']):04d}_{sample_id[:12]}",
            "section": "training_original",
            "clone_model": "source",
            "candidate_index": int(row["pool_index"]),
            "source_sample_id": sample_id,
            "source_path": str(row["audio_path"]),
        }


def self_clone_rows(root: Path, model: str) -> Iterable[dict[str, Any]]:
    selection = load_json(root / f"{model}_inputs" / "selection.json")
    generated_dir = root / "generated" / f"{model}_cfg1"
    for row in selection:
        name = str(row["name"])
        yield {
            "item_id": f"self_{model}_{name}",
            "section": f"self_{model}",
            "clone_model": model,
            "candidate_index": int(row["pool_index"]),
            "source_sample_id": str(row["sample_id"]),
            "source_path": str(generated_dir / f"{name}.wav"),
        }


def cross_source_rows(root: Path, model: str) -> Iterable[dict[str, Any]]:
    source_dir = root / "cross_source" / ("V4PH" if model == "v4ph" else "V4fg")
    for index, path in enumerate(sorted(source_dir.glob("*.wav")), start=1):
        yield {
            "item_id": f"cross_{model}_{index:02d}_{path.stem}",
            "section": "cross_source",
            "clone_model": model,
            "candidate_index": index,
            "source_sample_id": path.stem,
            "source_path": str(path),
        }


def relative_output_path(row: dict[str, Any]) -> Path:
    return Path(str(row["section"])) / f"{row['item_id']}.wav"


def process_one(
    root: Path,
    row: dict[str, Any],
    target_rms_dbfs: float,
    peak_ceiling_dbfs: float,
) -> dict[str, Any]:
    source = Path(str(row["source_path"]))
    if not source.is_file():
        raise FileNotFoundError(source)

    audio, sample_rate = sf.read(source, dtype="float64", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"invalid audio: {source}")
    mono = audio.mean(axis=1)
    source_rms = float(np.sqrt(np.mean(np.square(mono))))
    source_peak = float(np.max(np.abs(mono)))
    if source_rms <= 1e-12:
        raise ValueError(f"silent audio: {source}")

    gain_db = target_rms_dbfs - db(source_rms)
    gain = 10.0 ** (gain_db / 20.0)
    normalized = mono * gain
    normalized_peak = float(np.max(np.abs(normalized)))
    peak_ceiling = 10.0 ** (peak_ceiling_dbfs / 20.0)
    if normalized_peak > peak_ceiling + 1e-8:
        raise ValueError(
            f"linear normalization exceeds peak ceiling: {source} "
            f"peak={db(normalized_peak):.4f} dBFS ceiling={peak_ceiling_dbfs:.4f} dBFS"
        )

    relative = relative_output_path(row)
    destination = root / "normalized" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.wav")
    sf.write(temporary, normalized, sample_rate, format="WAV", subtype="PCM_24")
    os.replace(temporary, destination)

    written, written_sr = sf.read(destination, dtype="float64", always_2d=True)
    written_mono = written.mean(axis=1)
    written_rms = float(np.sqrt(np.mean(np.square(written_mono))))
    written_peak = float(np.max(np.abs(written_mono)))

    result = dict(row)
    result.update(
        {
            "normalized_path": str(destination),
            "normalized_relative_path": relative.as_posix(),
            "source_sha256": sha256_file(source),
            "normalized_sha256": sha256_file(destination),
            "sample_rate": written_sr,
            "channels": 1,
            "duration_seconds": len(written_mono) / written_sr,
            "source_rms_dbfs": db(source_rms),
            "source_peak_dbfs": db(source_peak),
            "linear_gain_db": gain_db,
            "normalized_rms_dbfs": db(written_rms),
            "normalized_peak_dbfs": db(written_peak),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows = list(original_rows(root))
    rows.extend(self_clone_rows(root, "v4ph"))
    rows.extend(self_clone_rows(root, "v4fg"))
    rows.extend(cross_source_rows(root, "v4ph"))
    rows.extend(cross_source_rows(root, "v4fg"))

    expected = {
        "training_original": 2000,
        "self_v4ph": 500,
        "self_v4fg": 500,
        "cross_source": 54,
    }
    observed = {key: sum(row["section"] == key for row in rows) for key in expected}
    if observed != expected:
        raise RuntimeError(f"pool counts differ: expected={expected} observed={observed}")
    if len({row["item_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate item_id in canonical pool")

    output_rows: list[dict[str, Any]] = []
    started = __import__("time").monotonic()
    for index, row in enumerate(rows, start=1):
        output_rows.append(
            process_one(
                root,
                row,
                target_rms_dbfs=args.target_rms_dbfs,
                peak_ceiling_dbfs=args.peak_ceiling_dbfs,
            )
        )
        if index == 1 or index % 50 == 0 or index == len(rows):
            elapsed = __import__("time").monotonic() - started
            rate = index / max(elapsed, 1e-9)
            eta = (len(rows) - index) / max(rate, 1e-9)
            print(
                f"[progress] completed={index}/{len(rows)} elapsed={elapsed:.1f}s "
                f"eta={eta:.1f}s",
                flush=True,
            )

    manifest = root / "canonical_audio_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    audit = {
        "schema": "noise-scorer-canonical-audio.v1",
        "target_rms_dbfs": args.target_rms_dbfs,
        "peak_ceiling_dbfs": args.peak_ceiling_dbfs,
        "counts": observed,
        "total": len(output_rows),
        "sample_rates": sorted({int(row["sample_rate"]) for row in output_rows}),
        "channels": sorted({int(row["channels"]) for row in output_rows}),
        "normalized_rms_dbfs_min": min(float(row["normalized_rms_dbfs"]) for row in output_rows),
        "normalized_rms_dbfs_max": max(float(row["normalized_rms_dbfs"]) for row in output_rows),
        "normalized_peak_dbfs_max": max(float(row["normalized_peak_dbfs"]) for row in output_rows),
        "manifest_sha256": sha256_file(manifest),
    }
    with (root / "canonical_audio_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
