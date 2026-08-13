#!/usr/bin/env python3
"""Score the canonical noise-tier pool with the original MOSNet CNN-BLSTM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    root = args.root.resolve()
    third_party = root / "third_party" / "MOSNet"
    sys.path.insert(0, str(third_party))

    import scipy.signal

    if not hasattr(scipy.signal, "hamming"):
        scipy.signal.hamming = scipy.signal.windows.hamming

    from model import CNN_BLSTM
    import utils

    with (root / "canonical_audio_manifest.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3054:
        raise RuntimeError(f"expected 3054 rows, found {len(rows)}")
    if args.limit:
        rows = rows[: args.limit]

    output = root / "scores" / "mosnet.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()
    completed: set[str] = set()
    if output.exists():
        with output.open("r", newline="", encoding="utf-8") as handle:
            completed = {row["item_id"] for row in csv.DictReader(handle)}

    print("[load] MOSNet CNN-BLSTM", flush=True)
    model = CNN_BLSTM().build()
    weight_path = third_party / "pre_trained" / "cnn_blstm.h5"
    model.load_weights(weight_path)

    pending = [row for row in rows if row["item_id"] not in completed]
    started = time.monotonic()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "score"])
        if output.stat().st_size == 0:
            writer.writeheader()
        for index, row in enumerate(pending, start=1):
            spectrogram = utils.get_spectrograms(row["normalized_path"])
            spectrogram = spectrogram.reshape(1, spectrogram.shape[0], utils.SGRAM_DIM)
            average_score, _ = model.predict(
                spectrogram, verbose=0, batch_size=1
            )
            score = float(average_score[0][0])
            if not math.isfinite(score):
                raise ValueError(f"non-finite MOSNet score for {row['item_id']}")
            writer.writerow({"item_id": row["item_id"], "score": score})
            handle.flush()
            completed.add(row["item_id"])
            if index == 1 or index % 10 == 0 or index == len(pending):
                elapsed = time.monotonic() - started
                eta = (len(pending) - index) * elapsed / max(index, 1)
                print(
                    f"[progress] scorer_row={index}/{len(pending)} "
                    f"total_done={len(completed)}/{len(rows)} elapsed={elapsed:.1f}s "
                    f"eta={eta:.1f}s",
                    flush=True,
                )

    provenance = {
        "schema": "noise-scorer-run.v1",
        "scorer": "mosnet",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "tensorflow": __import__("tensorflow").__version__,
        "weight_sha256": sha256_file(weight_path),
        "canonical_manifest_sha256": sha256_file(root / "canonical_audio_manifest.csv"),
        "result_sha256": sha256_file(output),
    }
    with (root / "scores" / "mosnet.provenance.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[complete] scorer=mosnet rows={len(rows)} output={output}", flush=True)


if __name__ == "__main__":
    main()
