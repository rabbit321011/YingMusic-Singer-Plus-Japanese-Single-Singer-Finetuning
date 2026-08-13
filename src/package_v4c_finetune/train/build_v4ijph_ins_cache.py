#!/usr/bin/env python3
"""Build the resumable, waveform-bound V4IjPH INS cache v2."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from v4ijph_contract import V4IJPH_INS_CACHE_SCHEMA, V4IJPH_INS_DIM
from v4ijph_ins_cache import (
    INTRINSIC_CHECKPOINT_NAME,
    INTRINSIC_CHECKPOINT_SHA256,
    INTRINSIC_MODEL_REVISION,
    INTRINSIC_REPOSITORY,
    MINIMUM_INPUT_SAMPLES,
    MODEL_SAMPLE_RATE,
    PARASPEECHCLAP_SOURCE_REVISION,
    TRAINING_SAMPLE_RATE,
    InsCache,
    ParaSpeechClapIntrinsic,
    atomic_write_json,
    atomic_write_jsonl,
    cache_config_fingerprint,
    load_h_manifest,
    load_training_waveform,
    resolve_audio_path,
    sha256_file,
    sha256_waveform,
    verify_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--speech-model-dir", type=Path, required=True)
    parser.add_argument("--speech-model-inventory", type=Path, required=True)
    parser.add_argument("--text-model-dir", type=Path, required=True)
    parser.add_argument("--text-model-inventory", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-inventory", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--audio-root-override")
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def open_array(path: Path, dtype, shape):
    if path.exists():
        value = np.lib.format.open_memmap(path, mode="r+")
        if value.dtype != np.dtype(dtype) or value.shape != shape:
            raise ValueError(f"Existing cache array contract mismatch: {path}")
        return value
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def retry_indices(status: np.ndarray) -> list[int]:
    """Retry every row that is not already a fully committed success."""
    return [index for index, value in enumerate(status) if int(value) != 1]


def main() -> None:
    args = parse_args()
    if args.max_duration <= 0 or args.flush_every <= 0 or args.progress_every <= 0:
        raise ValueError("duration and progress intervals must be positive")

    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve()
    speech_dir = args.speech_model_dir.resolve()
    text_dir = args.text_model_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint = checkpoint_dir / INTRINSIC_CHECKPOINT_NAME
    manifest = load_h_manifest(manifest_path)

    inventories = {
        "source": verify_inventory(args.source_inventory, source_dir, verify_files=True),
        "speech_model": verify_inventory(
            args.speech_model_inventory, speech_dir, verify_files=True
        ),
        "text_model": verify_inventory(
            args.text_model_inventory, text_dir, verify_files=True
        ),
        "checkpoint": verify_inventory(
            args.checkpoint_inventory, checkpoint_dir, verify_files=True
        ),
    }
    if not checkpoint.is_file() or sha256_file(checkpoint) != INTRINSIC_CHECKPOINT_SHA256:
        raise ValueError("Frozen Intrinsic checkpoint is missing or has the wrong SHA256")

    config = {
        "schema": V4IJPH_INS_CACHE_SCHEMA,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "max_duration_sec": float(args.max_duration),
        "sample_rate_before_encoder": TRAINING_SAMPLE_RATE,
        "channel_policy": "require_mono",
        "truncation_policy": "prefix_after_resample_to_44100",
        "waveform_hash": "sha256_float32_44100_after_resample_and_prefix_crop",
        "model_repository": INTRINSIC_REPOSITORY,
        "model_revision": INTRINSIC_MODEL_REVISION,
        "source_revision": PARASPEECHCLAP_SOURCE_REVISION,
        "checkpoint_name": INTRINSIC_CHECKPOINT_NAME,
        "checkpoint_sha256": INTRINSIC_CHECKPOINT_SHA256,
        "model_sample_rate": MODEL_SAMPLE_RATE,
        "minimum_input_samples": MINIMUM_INPUT_SAMPLES,
        "embedding_dim": V4IJPH_INS_DIM,
        "encoder_call": "get_audio_embedding(normalize=True)",
        "cache_dtype": "float16",
        "training_load_dtype": "float32",
        "inventories": inventories,
        "builder_code_sha256": sha256_file(Path(__file__).resolve()),
        "cache_runtime_code_sha256": sha256_file(
            Path(__file__).resolve().with_name("v4ijph_ins_cache.py")
        ),
    }
    fingerprint = cache_config_fingerprint(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    records_path = output_dir / "records.jsonl"
    failures_path = output_dir / "failures.jsonl"
    array_paths = {
        "embeddings.f16.npy": output_dir / "embeddings.f16.npy",
        "status.u8.npy": output_dir / "status.u8.npy",
        "waveform_sha256.s64.npy": output_dir / "waveform_sha256.s64.npy",
        "waveform_samples.i64.npy": output_dir / "waveform_samples.i64.npy",
    }

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_fingerprint") != fingerprint:
            raise ValueError("Existing cache config differs; use a new output directory")
        if metadata.get("state") == "complete":
            cache = InsCache(
                output_dir,
                expected_manifest_sha256=config["manifest_sha256"],
                expected_max_duration_sec=args.max_duration,
            )
            cache.assert_manifest(manifest)
            print(json.dumps({"state": "already_complete", "rows": len(manifest)}))
            return
    else:
        metadata = {
            "schema": V4IJPH_INS_CACHE_SCHEMA,
            "state": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "config_fingerprint": fingerprint,
            "counts": {
                "total": len(manifest),
                "success": 0,
                "failed": 0,
                "pending": len(manifest),
            },
            "files": {},
        }
        atomic_write_json(metadata_path, metadata)

    rows = len(manifest)
    embeddings = open_array(array_paths["embeddings.f16.npy"], np.float16, (rows, 768))
    status = open_array(array_paths["status.u8.npy"], np.uint8, (rows,))
    waveform_hashes = open_array(
        array_paths["waveform_sha256.s64.npy"], "S64", (rows,)
    )
    waveform_samples = open_array(
        array_paths["waveform_samples.i64.npy"], np.int64, (rows,)
    )
    invalid_status = ~np.isin(status, np.array([0, 1, 2], dtype=np.uint8))
    status[invalid_status] = 0
    pending = retry_indices(status)
    status[pending] = 0
    for array in (embeddings, status, waveform_hashes, waveform_samples):
        array.flush()

    print(
        f"[V4IjPH cache v2] rows={rows} pending={len(pending)} "
        f"device={args.device} max_duration={args.max_duration}s",
        flush=True,
    )
    if pending:
        encoder = ParaSpeechClapIntrinsic(
            source_dir=source_dir,
            speech_model_dir=speech_dir,
            text_model_dir=text_dir,
            checkpoint=checkpoint,
            device=args.device,
        )
        failures_path.touch(exist_ok=True)
        started = time.perf_counter()
        with failures_path.open("a", encoding="utf-8", buffering=1) as failures:
            for attempted, index in enumerate(pending, start=1):
                row = manifest[index]
                try:
                    waveform, resolved_path = load_training_waveform(
                        row["Path"],
                        max_duration_sec=args.max_duration,
                        audio_root_override=args.audio_root_override,
                    )
                    vector = encoder.encode(waveform)
                    embeddings[index] = vector.astype(np.float16)
                    waveform_hashes[index] = sha256_waveform(waveform).encode("ascii")
                    waveform_samples[index] = waveform.numel()
                    if not np.isfinite(np.asarray(embeddings[index], dtype=np.float32)).all():
                        raise FloatingPointError("FP16 conversion produced non-finite INS")
                    status[index] = 1
                except Exception as error:
                    status[index] = 2
                    failures.write(
                        canonical_failure(
                            index=index,
                            row=row,
                            error=error,
                        )
                        + "\n"
                    )
                if attempted % args.flush_every == 0 or attempted == len(pending):
                    for array in (
                        embeddings,
                        status,
                        waveform_hashes,
                        waveform_samples,
                    ):
                        array.flush()
                if attempted % args.progress_every == 0 or attempted == len(pending):
                    elapsed = time.perf_counter() - started
                    success = int(np.count_nonzero(status == 1))
                    failed = int(np.count_nonzero(status == 2))
                    remaining = int(np.count_nonzero(status == 0))
                    rate = attempted / elapsed if elapsed else 0.0
                    eta = remaining / rate if rate else float("inf")
                    print(
                        f"[V4IjPH cache v2] attempted={attempted}/{len(pending)} "
                        f"success={success} failed={failed} pending={remaining} "
                        f"elapsed={elapsed:.1f}s rows_per_s={rate:.3f} eta={eta:.1f}s",
                        flush=True,
                    )

    success = int(np.count_nonzero(status == 1))
    failed = int(np.count_nonzero(status == 2))
    remaining = int(np.count_nonzero(status == 0))
    complete = success == rows and failed == 0 and remaining == 0
    if complete:
        records = []
        for index, row in enumerate(manifest):
            resolved = resolve_audio_path(row["Path"], args.audio_root_override)
            records.append(
                {
                    "cache_index": index,
                    "sample_id": row["SampleId"],
                    "source_path": row["Path"],
                    "resolved_path": str(resolved),
                    "manifest_duration_sec": row.get("Duration"),
                    "training_sample_rate": TRAINING_SAMPLE_RATE,
                    "waveform_samples": int(waveform_samples[index]),
                    "waveform_sha256": bytes(waveform_hashes[index]).decode("ascii"),
                }
            )
        atomic_write_jsonl(records_path, records)

    state = "complete" if complete else "incomplete"
    file_paths = dict(array_paths)
    if records_path.exists():
        file_paths["records.jsonl"] = records_path
    if failures_path.exists():
        file_paths["failures.jsonl"] = failures_path
    metadata.update(
        {
            "state": state,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": (
                datetime.now(timezone.utc).isoformat() if complete else None
            ),
            "counts": {
                "total": rows,
                "success": success,
                "failed": failed,
                "pending": remaining,
            },
            "files": {
                name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for name, path in file_paths.items()
            },
        }
    )
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"state": state, **metadata["counts"]}), flush=True)
    if not complete:
        raise SystemExit(1)


def canonical_failure(*, index: int, row: dict, error: Exception) -> str:
    return json.dumps(
        {
            "cache_index": index,
            "sample_id": row["SampleId"],
            "source_path": row["Path"],
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(limit=8),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


if __name__ == "__main__":
    main()
