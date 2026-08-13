#!/usr/bin/env python3
"""Normalize, score, and analyze a fixed-task multi-seed UTMOS22 experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import shutil
import time
import zipfile
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


MODELS = ("v4ph", "v4fg")
SEEDS = tuple(range(42, 52))
EXPECTED_TASKS = 27
TARGET_RMS_DBFS = -28.0
PEAK_CEILING_DBFS = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-15))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def discover(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_tasks: list[str] | None = None
    for model in MODELS:
        for seed in SEEDS:
            source_dir = root / "generated" / model / f"seed_{seed:04d}"
            paths = sorted(source_dir.glob("*.wav"), key=lambda path: path.name)
            if len(paths) != EXPECTED_TASKS:
                raise RuntimeError(
                    f"expected {EXPECTED_TASKS} WAV files in {source_dir}, found {len(paths)}"
                )
            tasks = [path.stem for path in paths]
            if reference_tasks is None:
                reference_tasks = tasks
            elif tasks != reference_tasks:
                raise RuntimeError(f"task set/order mismatch: {source_dir}")
            for task_index, source in enumerate(paths, start=1):
                rows.append(
                    {
                        "item_id": f"{model}_seed_{seed:04d}_task_{task_index:02d}",
                        "model": model,
                        "seed": seed,
                        "task_index": task_index,
                        "task": source.stem,
                        "source_path": str(source.resolve()),
                        "normalized_path": str(
                            (
                                root
                                / "normalized"
                                / model
                                / f"seed_{seed:04d}"
                                / source.name
                            ).resolve()
                        ),
                    }
                )
    if len(rows) != len(MODELS) * len(SEEDS) * EXPECTED_TASKS:
        raise AssertionError("unexpected discovered row count")
    return rows


def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_rms = 10.0 ** (TARGET_RMS_DBFS / 20.0)
    peak_ceiling = 10.0 ** (PEAK_CEILING_DBFS / 20.0)
    output: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        source = Path(row["source_path"])
        audio, sample_rate = sf.read(source, dtype="float64", always_2d=True)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(f"invalid source audio: {source}")
        mono = audio.mean(axis=1)
        source_rms = float(np.sqrt(np.mean(np.square(mono))))
        source_peak = float(np.max(np.abs(mono)))
        if source_rms <= 1e-12:
            raise ValueError(f"silent source audio: {source}")
        gain = target_rms / source_rms
        normalized = mono * gain
        normalized_peak = float(np.max(np.abs(normalized)))
        if normalized_peak > peak_ceiling + 1e-8:
            raise ValueError(
                f"normalization exceeds {PEAK_CEILING_DBFS} dBFS: "
                f"{source} ({db(normalized_peak):.4f} dBFS)"
            )
        destination = Path(row["normalized_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.wav")
        sf.write(temporary, normalized, sample_rate, format="WAV", subtype="PCM_24")
        temporary.replace(destination)
        written, written_sr = sf.read(destination, dtype="float64", always_2d=True)
        written = written.mean(axis=1)
        result = dict(row)
        result.update(
            {
                "source_sha256": sha256_file(source),
                "normalized_sha256": sha256_file(destination),
                "sample_rate": written_sr,
                "channels": 1,
                "duration_seconds": len(written) / written_sr,
                "source_rms_dbfs": db(source_rms),
                "source_peak_dbfs": db(source_peak),
                "linear_gain_db": db(gain),
                "normalized_rms_dbfs": db(
                    float(np.sqrt(np.mean(np.square(written))))
                ),
                "normalized_peak_dbfs": db(float(np.max(np.abs(written)))),
            }
        )
        output.append(result)
        if index == 1 or index % 25 == 0 or index == len(rows):
            elapsed = time.monotonic() - started
            eta = elapsed * (len(rows) - index) / index
            print(
                f"[normalize] {index}/{len(rows)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return output


def load_utmos22(scorer_root: Path, device: str):
    import torch

    model = torch.hub.load(
        str(scorer_root / "third_party" / "SpeechMOS"),
        "utmos22_strong",
        source="local",
        pretrained=False,
    )
    weight = scorer_root / "scorer_weights" / "utmos22_strong_step7459_v1.pt"
    state = torch.load(weight, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model.to(device).eval(), weight


def score_one(model: Any, path: Path, device: str) -> float:
    import torch
    import torchaudio

    wave, sample_rate = torchaudio.load(path)
    wave = wave.mean(dim=0, keepdim=True).to(dtype=torch.float32, device=device)
    with torch.inference_mode():
        value = float(model(wave, sample_rate).item())
    if not math.isfinite(value):
        raise ValueError(f"non-finite UTMOS22 result: {path}")
    return value


def score_all(
    root: Path, scorer_root: Path, rows: list[dict[str, Any]], device: str
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    import torch

    score_path = root / "utmos22_scores.csv"
    existing: dict[str, float] = {}
    if score_path.is_file():
        with score_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing[row["item_id"]] = float(row["utmos22"])
    model, weight = load_utmos22(scorer_root, device)
    started = time.monotonic()
    scores: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        value = existing.get(row["item_id"])
        if value is None:
            value = score_one(model, Path(row["normalized_path"]), device)
        scores.append(
            {
                "item_id": row["item_id"],
                "model": row["model"],
                "seed": row["seed"],
                "task_index": row["task_index"],
                "task": row["task"],
                "utmos22": value,
                "normalized_path": row["normalized_path"],
            }
        )
        if index == 1 or index % 10 == 0 or index == len(rows):
            write_csv(score_path, scores, list(scores[0]))
            elapsed = time.monotonic() - started
            eta = elapsed * (len(rows) - index) / index
            print(
                f"[score] {index}/{len(rows)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    primary = {row["item_id"]: float(row["utmos22"]) for row in scores}
    check_rows = [row for row in rows if row["seed"] == SEEDS[0]]
    same_model: list[list[float]] = []
    for repeat in range(2):
        values = [
            score_one(model, Path(row["normalized_path"]), device) for row in check_rows
        ]
        same_model.append(values)
        print(f"[repeat] same_model pass={repeat + 1}/2", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_utmos22(scorer_root, device)
    reload_values = [
        score_one(reloaded, Path(row["normalized_path"]), device) for row in check_rows
    ]
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()

    repeat_rows = []
    for index, row in enumerate(check_rows):
        values = [
            primary[row["item_id"]],
            same_model[0][index],
            same_model[1][index],
            reload_values[index],
        ]
        repeat_rows.append(
            {
                "item_id": row["item_id"],
                "primary": values[0],
                "same_model_repeat_1": values[1],
                "same_model_repeat_2": values[2],
                "model_reload_repeat": values[3],
                "max_abs_delta": max(values) - min(values),
            }
        )
    write_csv(root / "scorer_repeatability.csv", repeat_rows, list(repeat_rows[0]))
    repeatability = {
        "checked_audio": len(repeat_rows),
        "passes_per_audio": 4,
        "max_abs_delta": max(float(row["max_abs_delta"]) for row in repeat_rows),
        "mean_abs_span": float(
            np.mean([float(row["max_abs_delta"]) for row in repeat_rows])
        ),
    }
    return scores, repeatability, weight


def summarize_model(model: str, rows: list[dict[str, Any]]) -> tuple[dict, list[dict]]:
    selected = [row for row in rows if row["model"] == model]
    tasks = sorted({str(row["task"]) for row in selected})
    seeds = list(SEEDS)
    lookup = {
        (str(row["task"]), int(row["seed"])): float(row["utmos22"])
        for row in selected
    }
    matrix = np.array([[lookup[(task, seed)] for seed in seeds] for task in tasks])
    task_rows: list[dict[str, Any]] = []
    all_pair_differences: list[float] = []
    top_seeds: list[int] = []
    for task_index, task in enumerate(tasks):
        values = matrix[task_index]
        differences = [abs(values[a] - values[b]) for a, b in combinations(range(10), 2)]
        all_pair_differences.extend(differences)
        best = int(np.argmax(values))
        worst = int(np.argmin(values))
        top_seeds.append(seeds[best])
        task_rows.append(
            {
                "model": model,
                "task": task,
                "mean": float(np.mean(values)),
                "std_ddof1": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "range": float(np.ptp(values)),
                "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                "worst_seed": seeds[worst],
                "worst_score": float(values[worst]),
                "best_seed": seeds[best],
                "best_score": float(values[best]),
                "pairs_ge_0p02": sum(value >= 0.02 for value in differences),
                "pairs_ge_0p05": sum(value >= 0.05 for value in differences),
                "pairs_ge_0p10": sum(value >= 0.10 for value in differences),
            }
        )

    grand = float(np.mean(matrix))
    task_means = np.mean(matrix, axis=1)
    seed_means = np.mean(matrix, axis=0)
    residual = matrix - task_means[:, None] - seed_means[None, :] + grand
    ss_task = len(seeds) * float(np.sum(np.square(task_means - grand)))
    ss_seed = len(tasks) * float(np.sum(np.square(seed_means - grand)))
    ss_residual = float(np.sum(np.square(residual)))
    ss_total = float(np.sum(np.square(matrix - grand)))
    task_ranges = np.ptp(matrix, axis=1)
    task_stds = np.std(matrix, axis=1, ddof=1)
    pair_array = np.asarray(all_pair_differences)
    ms_between = len(seeds) * float(np.var(task_means, ddof=1))
    ms_within = float(np.mean(np.var(matrix, axis=1, ddof=1)))
    icc_denominator = ms_between + (len(seeds) - 1) * ms_within
    icc_1_1 = (
        (ms_between - ms_within) / icc_denominator
        if icc_denominator > 0
        else float("nan")
    )
    summary = {
        "model": model,
        "tasks": len(tasks),
        "seeds_per_task": len(seeds),
        "score_mean": grand,
        "score_min": float(np.min(matrix)),
        "score_max": float(np.max(matrix)),
        "task_seed_std_mean": float(np.mean(task_stds)),
        "task_seed_std_median": float(np.median(task_stds)),
        "task_seed_std_p10": float(np.quantile(task_stds, 0.10)),
        "task_seed_std_p90": float(np.quantile(task_stds, 0.90)),
        "task_seed_range_mean": float(np.mean(task_ranges)),
        "task_seed_range_median": float(np.median(task_ranges)),
        "task_seed_range_min": float(np.min(task_ranges)),
        "task_seed_range_max": float(np.max(task_ranges)),
        "tasks_range_ge_0p05": int(np.sum(task_ranges >= 0.05)),
        "tasks_range_ge_0p10": int(np.sum(task_ranges >= 0.10)),
        "tasks_range_ge_0p20": int(np.sum(task_ranges >= 0.20)),
        "pair_abs_diff_median": float(np.median(pair_array)),
        "pair_fraction_ge_0p02": float(np.mean(pair_array >= 0.02)),
        "pair_fraction_ge_0p05": float(np.mean(pair_array >= 0.05)),
        "pair_fraction_ge_0p10": float(np.mean(pair_array >= 0.10)),
        "between_task_sd_of_means": float(np.std(task_means, ddof=1)),
        "pooled_within_task_seed_sd": math.sqrt(ms_within),
        "icc_1_1_task_identity": icc_1_1,
        "variance_fraction_task": ss_task / ss_total if ss_total else 0.0,
        "variance_fraction_fixed_seed": ss_seed / ss_total if ss_total else 0.0,
        "variance_fraction_task_by_seed": ss_residual / ss_total if ss_total else 0.0,
        "fixed_seed_mean_range": float(np.ptp(seed_means)),
        "seed_means": {str(seed): float(seed_means[i]) for i, seed in enumerate(seeds)},
        "best_seed_counts": {
            str(seed): count for seed, count in sorted(Counter(top_seeds).items())
        },
    }
    return summary, task_rows


def build_listening_package(
    root: Path, scores: list[dict[str, Any]], task_rows: list[dict[str, Any]]
) -> Path:
    score_lookup = {
        (row["model"], row["task"], int(row["seed"])): row for row in scores
    }
    package = root / "listen_extremes"
    package.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for row in task_rows:
        model = str(row["model"])
        task = str(row["task"])
        for role, order in (("worst", "01"), ("best", "02")):
            seed = int(row[f"{role}_seed"])
            score = float(row[f"{role}_score"])
            source_row = score_lookup[(model, task, seed)]
            safe_score = f"{score:.6f}".replace("-", "m").replace(".", "p")
            destination = (
                package
                / model
                / task
                / f"{order}_{role}_seed_{seed:04d}_score_{safe_score}.wav"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_row["normalized_path"], destination)
            manifest.append(
                {
                    "model": model,
                    "task": task,
                    "role": role,
                    "seed": seed,
                    "utmos22": score,
                    "relative_path": destination.relative_to(package).as_posix(),
                }
            )
    write_csv(package / "manifest.csv", manifest, list(manifest[0]))
    archive = root / "listen_extremes.zip"
    temporary = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(package.parent))
    temporary.replace(archive)
    return archive


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    scorer_root = args.scorer_root.resolve()
    rows = discover(root)
    canonical_rows = normalize(rows)
    canonical_fields = list(canonical_rows[0])
    canonical_path = root / "canonical_manifest.csv"
    write_csv(canonical_path, canonical_rows, canonical_fields)
    scores, repeatability, weight = score_all(
        root, scorer_root, canonical_rows, args.device
    )

    model_summaries = []
    all_task_rows: list[dict[str, Any]] = []
    for model in MODELS:
        summary, task_rows = summarize_model(model, scores)
        model_summaries.append(summary)
        all_task_rows.extend(task_rows)
    write_csv(root / "per_task_stats.csv", all_task_rows, list(all_task_rows[0]))
    archive = build_listening_package(root, scores, all_task_rows)

    report = {
        "schema": "utmos22-seed-variance.v1",
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "tasks_per_model": EXPECTED_TASKS,
        "total_audio": len(rows),
        "normalization": {
            "target_rms_dbfs": TARGET_RMS_DBFS,
            "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
            "sample_rates": sorted({int(row["sample_rate"]) for row in canonical_rows}),
            "channels": [1],
            "normalized_rms_min": min(
                float(row["normalized_rms_dbfs"]) for row in canonical_rows
            ),
            "normalized_rms_max": max(
                float(row["normalized_rms_dbfs"]) for row in canonical_rows
            ),
            "normalized_peak_max": max(
                float(row["normalized_peak_dbfs"]) for row in canonical_rows
            ),
        },
        "scorer_repeatability": repeatability,
        "model_summaries": model_summaries,
        "artifacts": {
            "canonical_manifest_sha256": sha256_file(canonical_path),
            "scores_sha256": sha256_file(root / "utmos22_scores.csv"),
            "per_task_stats_sha256": sha256_file(root / "per_task_stats.csv"),
            "utmos22_weight_sha256": sha256_file(weight),
            "listening_archive": str(archive),
            "listening_archive_sha256": sha256_file(archive),
        },
    }
    report_path = root / "summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
