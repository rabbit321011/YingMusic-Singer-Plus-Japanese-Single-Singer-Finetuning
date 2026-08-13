#!/usr/bin/env python3
"""Prepare independent train-derived pools for noise scorer tier packs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path

import soundfile as sf


VFR = 44100 / 2048
ALIGNMENT_SCHEMA = "aisvc.v4h-eval-alignment.v1"


def safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]', "_", str(value)).strip().rstrip(".")
    return value or "unnamed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio(record: dict, audio_root: Path) -> Path:
    path = Path(record["Path"])
    if path.is_file():
        return path
    local = audio_root / path.name
    if not local.is_file():
        raise FileNotFoundError(local)
    return local


def choose_boundary(record: dict, rng: random.Random) -> tuple[int, int]:
    total_frames = int(record.get("ExpectedLatentFrames") or record["Duration"] * VFR)
    five_seconds = int(5.0 * VFR)
    raw = total_frames * rng.uniform(0.125, 0.33)
    raw = max(raw, five_seconds)
    raw = min(raw, int(total_frames * 0.65))
    boundaries = [
        (index, int(float(phrase["start"]) * VFR))
        for index, phrase in enumerate(record["Phrases"])
        if index > 0
    ]
    margin = int(2.5 * VFR)
    candidates = [item for item in boundaries if abs(item[1] - raw) <= margin]
    if not candidates:
        raise ValueError("no phrase boundary within the training adsorption margin")
    phrase_index, ref_frames = min(candidates, key=lambda item: abs(item[1] - raw))
    if not (five_seconds <= ref_frames <= int(total_frames * 0.65)):
        raise ValueError("adsorbed boundary violates the training ref-length contract")
    return phrase_index, ref_frames


def normalize_phrases(phrases: list[dict], offset: float) -> list[dict]:
    result = []
    for phrase in phrases:
        item = copy.deepcopy(phrase)
        item["start"] = max(0.0, float(item["start"]) - offset)
        item["end"] = max(item["start"], float(item["end"]) - offset)
        result.append(item)
    return result


def normalize_candidates(candidates: list[dict]) -> list[dict]:
    result = copy.deepcopy(candidates)
    for index, candidate in enumerate(result):
        candidate["phrase_index"] = index
    return result


def write_t1(path: Path, phrases: list[dict]) -> None:
    payload = {
        "phrases": [
            {
                "text": phrase.get("text", ""),
                "kana": phrase.get("kana") or phrase.get("text", ""),
                "start": float(phrase["start"]),
                "end": float(phrase["end"]),
            }
            for phrase in phrases
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_pool(
    records: list[dict],
    audio_root: Path,
    output_root: Path,
    server_dataset_path: str,
    count: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    groups = []
    selected = []
    alignment_groups = []
    groups_dir = output_root / "groups"
    align_dir = output_root / "alignment" / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)

    for source_index in order:
        if len(groups) >= count:
            break
        record = records[source_index]
        phrases = record.get("Phrases") or []
        candidates = (record.get("HAlignment") or {}).get("phrase_candidates") or []
        if len(phrases) < 2 or len(phrases) != len(candidates):
            continue
        if not (8.0 <= float(record.get("Duration", 0.0)) <= 30.0):
            continue
        try:
            audio_path = resolve_audio(record, audio_root)
            phrase_index, ref_frames = choose_boundary(record, rng)
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        if sample_rate != 44100:
            continue
        boundary_seconds = float(phrases[phrase_index]["start"])
        boundary_sample = round(boundary_seconds * sample_rate)
        if boundary_sample <= 0 or boundary_sample >= len(audio):
            continue

        pool_index = len(groups) + 1
        sample_id = str(record.get("SampleId") or f"source_{source_index}")
        name = safe_name(f"train_{pool_index:04d}_{sample_id[:12]}")
        group_rel = f"groups/{name}"
        group_dir = output_root / group_rel
        group_dir.mkdir(parents=True, exist_ok=True)
        sf.write(group_dir / "A.wav", audio[:boundary_sample], sample_rate, subtype="PCM_16")
        sf.write(group_dir / "B.wav", audio[boundary_sample:], sample_rate, subtype="PCM_16")

        a_phrases = normalize_phrases(phrases[:phrase_index], 0.0)
        b_phrases = normalize_phrases(phrases[phrase_index:], boundary_seconds)
        a_candidates = normalize_candidates(candidates[:phrase_index])
        b_candidates = normalize_candidates(candidates[phrase_index:])
        write_t1(group_dir / "A_T1.json", a_phrases)
        write_t1(group_dir / "B_T1.json", b_phrases)

        group_entry = {"index": pool_index, "name": name, "directory": group_rel}
        alignment_rel = f"groups/{pool_index:04d}.json"
        alignment_entry = {**group_entry, "alignmentFile": alignment_rel}
        alignment_payload = {
            "schema": ALIGNMENT_SCHEMA,
            "indexEntry": alignment_entry,
            "A": {
                "Path": f"{server_dataset_path}/{group_rel}/A.wav",
                "Duration": boundary_seconds,
                "Phrases": a_phrases,
                "SourceSplit": "train_A",
                "SourceIndex": source_index,
                "SampleId": sample_id,
                "HAlignment": {"schema": "h_alignment_v1", "phrase_candidates": a_candidates},
            },
            "B": {
                "Path": f"{server_dataset_path}/{group_rel}/B.wav",
                "Duration": len(audio[boundary_sample:]) / sample_rate,
                "Phrases": b_phrases,
                "SourceSplit": "train_B",
                "SourceIndex": source_index,
                "SampleId": sample_id,
                "HAlignment": {"schema": "h_alignment_v1", "phrase_candidates": b_candidates},
            },
        }
        (output_root / "alignment" / alignment_rel).write_text(
            json.dumps(alignment_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        groups.append(group_entry)
        alignment_groups.append(alignment_entry)
        selected.append(
            {
                "pool_index": pool_index,
                "name": name,
                "source_index": source_index,
                "sample_id": sample_id,
                "source_audio": str(audio_path),
                "source_audio_sha256": record.get("AudioSHA256") or sha256_file(audio_path),
                "duration": len(audio) / sample_rate,
                "boundary_seconds": boundary_seconds,
                "ref_frames_training_formula": ref_frames,
                "a_phrases": len(a_phrases),
                "b_phrases": len(b_phrases),
            }
        )

    if len(groups) != count:
        raise RuntimeError(f"prepared only {len(groups)} of {count} requested records")
    (output_root / "manifest.json").write_text(
        json.dumps({"schema": "noise-scorer-train-selfclone.v1", "groups": groups}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    alignment_manifest = {
        "schema": ALIGNMENT_SCHEMA,
        "sourceDataset": server_dataset_path,
        "groups": alignment_groups,
    }
    (output_root / "alignment" / "manifest.json").write_text(
        json.dumps(alignment_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "selection.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selected


def prepare_original_pool(
    records: list[dict], audio_root: Path, output_path: Path, count: int, seed: int
) -> None:
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    selected = []
    for source_index in order:
        if len(selected) >= count:
            break
        record = records[source_index]
        if not (8.0 <= float(record.get("Duration", 0.0)) <= 30.0):
            continue
        try:
            audio_path = resolve_audio(record, audio_root)
        except FileNotFoundError:
            continue
        selected.append(
            {
                "pool_index": len(selected) + 1,
                "source_index": source_index,
                "sample_id": record.get("SampleId"),
                "audio_path": str(audio_path),
                "duration": record.get("Duration"),
                "audio_sha256": record.get("AudioSHA256"),
            }
        )
    if len(selected) != count:
        raise RuntimeError(f"prepared only {len(selected)} of {count} original records")
    output_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--server-root", default="${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805")
    parser.add_argument("--original-count", type=int, default=2000)
    parser.add_argument("--selfclone-count", type=int, default=500)
    args = parser.parse_args()

    records = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    prepare_original_pool(
        records,
        args.audio_root,
        args.output_root / "original_pool.json",
        args.original_count,
        seed=2026080501,
    )
    for model_name, seed in (("v4ph", 2026080502), ("v4fg", 2026080503)):
        output = args.output_root / f"{model_name}_inputs"
        selected = prepare_pool(
            records,
            args.audio_root,
            output,
            f"{args.server_root}/{model_name}_inputs",
            args.selfclone_count,
            seed,
        )
        print(f"[{model_name}] prepared {len(selected)} records at {output}")
    print(f"[original] prepared {args.original_count} records")


if __name__ == "__main__":
    main()
