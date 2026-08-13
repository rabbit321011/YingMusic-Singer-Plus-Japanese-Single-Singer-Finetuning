#!/usr/bin/env python3
"""Independent full provenance auditor for a completed V4IjPH INS cache v2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from v4ijph_ins_cache import (
    NORM_MAX,
    NORM_MIN,
    InsCache,
    load_h_manifest,
    load_training_waveform,
    sha256_file,
    sha256_waveform,
    verify_inventory,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root-override")
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--skip-waveform-rehash", action="store_true")
    parser.add_argument("--skip-model-rehash", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_h_manifest(manifest_path)
    cache = InsCache(
        args.cache_dir,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_max_duration_sec=args.max_duration,
    )
    cache.assert_manifest(manifest)

    config = cache.metadata["config"]
    if not args.skip_model_rehash:
        for label, record in config["inventories"].items():
            actual = verify_inventory(
                record["path"], record["model_dir"], verify_files=True
            )
            if actual != record:
                raise ValueError(f"{label} inventory provenance differs")

    waveform_mismatches = 0
    if not args.skip_waveform_rehash:
        started = time.perf_counter()
        for index, row in enumerate(manifest):
            waveform, resolved = load_training_waveform(
                row["Path"],
                max_duration_sec=args.max_duration,
                audio_root_override=args.audio_root_override,
            )
            cached = cache.records[index]
            if str(resolved) != cached["resolved_path"]:
                waveform_mismatches += 1
            if waveform.numel() != int(cached["waveform_samples"]):
                waveform_mismatches += 1
            if sha256_waveform(waveform) != cached["waveform_sha256"]:
                waveform_mismatches += 1
            done = index + 1
            if done % args.progress_every == 0 or done == len(manifest):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                eta = (len(manifest) - done) / rate if rate else float("inf")
                print(
                    f"[V4IjPH cache audit] verified={done}/{len(manifest)} "
                    f"elapsed={elapsed:.1f}s rows_per_s={rate:.2f} eta={eta:.1f}s",
                    flush=True,
                )
        if waveform_mismatches:
            raise ValueError(
                f"V4IjPH cache has {waveform_mismatches} waveform provenance mismatches"
            )

    values = np.asarray(cache.embeddings, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or norms.min() < NORM_MIN or norms.max() > NORM_MAX:
        raise ValueError("V4IjPH cache contains invalid embeddings")
    report = {
        "cache_dir": str(cache.root),
        "rows": len(cache.records),
        "embedding_shape": list(values.shape),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "waveform_rehash_performed": not args.skip_waveform_rehash,
        "model_inventory_rehash_performed": not args.skip_model_rehash,
        "waveform_mismatches": waveform_mismatches,
        "provenance": cache.provenance(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
