#!/usr/bin/env python3
"""Independent structural, binning, CRC, and SHA audit for noise-tier delivery."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any


POOLS = ("training_original", "self_v4ph", "self_v4fg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    package_root = root / "packages"
    archive_root = root / "package_archives"
    package_dirs = sorted(path for path in package_root.iterdir() if path.is_dir())
    archives = sorted(archive_root.glob("[0-9][0-9]_*.zip"))
    assert len(package_dirs) == 10, len(package_dirs)
    assert len(archives) == 10, len(archives)

    package_audit = json.loads((root / "package_audit.json").read_text(encoding="utf-8"))
    summary_by_dir = {
        row["package_directory"]: row for row in package_audit["packages"]
    }
    archive_manifest = {
        row["archive"]: row
        for row in read_csv(archive_root / "archives_manifest.csv")
    }
    results: list[dict[str, Any]] = []

    for package_dir in package_dirs:
        summary = summary_by_dir[package_dir.name]
        selected = read_csv(package_dir / "selected_manifest.csv")
        bins = read_csv(package_dir / "bins.csv")
        all_scores = read_csv(package_dir / "all_scores.csv")
        calibration = json.loads(
            (package_dir / "range_calibration.json").read_text(encoding="utf-8")
        )
        edges = [float(value) for value in calibration["edges"]]

        assert len(selected) == int(summary["selected_total"])
        assert len({row["item_id"] for row in selected}) == len(selected)
        assert len(bins) == 10
        assert len(all_scores) == 3054
        assert sum(row["section"] == "cross_source" for row in selected) == 54
        assert sum(
            row["section"] == "cross_source" and row["clone_model"] == "v4ph"
            for row in selected
        ) == 27
        assert sum(
            row["section"] == "cross_source" and row["clone_model"] == "v4fg"
            for row in selected
        ) == 27

        for pool in POOLS:
            percentage_sum = sum(float(row[f"{pool}_percent"]) for row in bins)
            assert math.isclose(percentage_sum, 100.0, abs_tol=1e-7), (
                package_dir.name,
                pool,
                percentage_sum,
            )
        for bin_index in range(1, 11):
            for pool in POOLS:
                count = sum(
                    row["section"] == pool
                    and int(row["bin_index"]) == bin_index
                    for row in selected
                )
                assert count <= 10, (package_dir.name, pool, bin_index, count)

        for row in all_scores:
            expected_bin = bisect.bisect_right(edges[1:-1], float(row["score"])) + 1
            assert int(row["bin_index"]) == expected_bin
        for row in selected:
            audio_path = package_dir / row["package_relative_path"]
            assert audio_path.is_file() and audio_path.stat().st_size > 44, audio_path

        archive = archive_root / f"{package_dir.name}.zip"
        manifest_row = archive_manifest[archive.name]
        archive_sha = sha256_file(archive)
        assert int(manifest_row["bytes"]) == archive.stat().st_size
        assert manifest_row["sha256"] == archive_sha
        with zipfile.ZipFile(archive) as handle:
            bad_member = handle.testzip()
            assert bad_member is None, (archive.name, bad_member)
            wav_count = sum(name.lower().endswith(".wav") for name in handle.namelist())
        assert wav_count == len(selected), (archive.name, wav_count, len(selected))

        results.append(
            {
                "package": package_dir.name,
                "selected_rows": len(selected),
                "cross_source_rows": 54,
                "all_score_rows": len(all_scores),
                "wav_files_in_zip": wav_count,
                "zip_bytes": archive.stat().st_size,
                "zip_sha256": archive_sha,
                "zip_crc": "PASS",
                "bin_caps": "PASS",
                "pool_percent_sums": "PASS",
            }
        )
        print(
            f"[pass] {package_dir.name} selected={len(selected)} "
            f"zip_bytes={archive.stat().st_size}",
            flush=True,
        )

    output = {
        "schema": "noise-scorer-delivery-independent-audit.v1",
        "status": "PASS",
        "package_count": 10,
        "canonical_score_rows": 3054,
        "packages": results,
    }
    (root / "delivery_independent_audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[all pass] packages=10", flush=True)


if __name__ == "__main__":
    main()
