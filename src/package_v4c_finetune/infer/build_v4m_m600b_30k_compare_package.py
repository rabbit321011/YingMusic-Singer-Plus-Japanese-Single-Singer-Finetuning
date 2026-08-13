#!/usr/bin/env python3
"""Build the named HighLR/M600-D/M600-B 30k comparison package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


PACKAGE_NAME = "V4M_M600B_30K_compare_named_20260806"
CONDITIONS = ("HIGHLR_30K", "M600D_30K", "M600B_30K")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardlink_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def zip_tree(source, output):
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    generated = root / "generated"
    dataset = root / "input" / "dataset"
    full_audit = json.loads((root / "full_audit.json").read_text(encoding="utf-8"))
    if full_audit.get("status") != "ok" or full_audit.get("wav_count") != 162:
        raise ValueError("three-model full output audit has not passed")

    package_root = root / "package"
    named_root = package_root / PACKAGE_NAME
    if named_root.exists():
        raise FileExistsError("named comparison package already exists")
    named_root.mkdir(parents=True)

    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    groups = dataset_manifest.get("groups") or []
    if len(groups) != 27:
        raise ValueError("listening dataset must contain 27 groups")

    references = named_root / "References"
    for group in groups:
        source = dataset / group["directory"]
        destination = references / group["directory"]
        for path in sorted(source.iterdir()):
            if path.is_file():
                hardlink_or_copy(path, destination / path.name)

    score_rows = []
    files = []
    for condition in CONDITIONS:
        for cfg in ("cfg3", "cfg1"):
            source_dir = generated / f"{condition}_{cfg}"
            destination_dir = named_root / "Step_30K" / f"{condition}_{cfg}"
            wavs = sorted(source_dir.glob("*.wav"))
            if len(wavs) != 27:
                raise ValueError(f"expected 27 WAVs in {source_dir}, got {len(wavs)}")
            for wav in wavs:
                destination = destination_dir / wav.name
                hardlink_or_copy(wav, destination)
                files.append(
                    {
                        "step": 30000,
                        "condition": condition,
                        "cfg": int(cfg[-1]),
                        "group": wav.stem,
                        "path": str(destination.relative_to(named_root)).replace("\\", "/"),
                        "sha256": sha256_file(wav),
                    }
                )
                score_rows.append(
                    {
                        "步数": "30K",
                        "组名": wav.stem,
                        "模型": condition,
                        "CFG": int(cfg[-1]),
                        "音色相似度": "",
                        "旋律准确度": "",
                        "咬字清晰度": "",
                        "自然度": "",
                        "风格适配": "",
                        "总体偏好": "",
                        "备注": "",
                    }
                )

    with (named_root / "评分表.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    readme = """# M600-B 30k 三模型实名对比包

本包比较相同 30k 训练步数下的三个模型：

- HighLR_30K：338M V4PH HighLR control；
- M600D_30K：约 600M 纯加深结构；
- M600B_30K：约 600M 深宽平衡结构；
- 每个模型均提供 CFG 3 与 CFG 1；
- References/<组名>/A.wav 是音色参考，B.wav 是目标旋律/时长参考。

同组、同 CFG 横向比较是结构裁决主口径。训练指标不属于本听评包的评分证据。
"""
    (named_root / "README.md").write_text(readme, encoding="utf-8")

    checkpoint_paths = {
        "HIGHLR_30K": Path(
            "${REMOTE_ROOT}/YingMusic-Singer-Plus/ckpts/"
            "plus_ja_sft_v4ph_30k_highlr/step_030000_final.pt"
        ),
        "M600D_30K": Path(
            "${REMOTE_ROOT}/TEMP/v4m_m600d_30k_listen_20260806/"
            "publish/V4M_M600D_step_030000.pt"
        ),
        "M600B_30K": root / "publish" / "V4M_M600B_step_030000.pt",
    }
    checkpoint_sha256 = {
        condition: sha256_file(path) for condition, path in checkpoint_paths.items()
    }
    manifest = {
        "schema": "v4m_m600b_30k_three_model_named_package_v1",
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "conditions": list(CONDITIONS),
        "checkpoint_sha256": checkpoint_sha256,
        "steps": [30000],
        "cfg": [3.0, 1.0],
        "sampling_steps": 32,
        "seed": 42,
        "groups": [group["name"] for group in groups],
        "wav_count": len(files),
        "files": files,
    }
    (named_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    zip_path = package_root / f"{PACKAGE_NAME}.zip"
    zip_tree(named_root, zip_path)
    report = {
        "schema": "v4m_m600b_30k_compare_package_report_v1",
        "status": "ok",
        "zip": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
        "generated_wav_count": len(files),
        "reference_wav_count": 54,
        "score_rows": len(score_rows),
        "conditions": list(CONDITIONS),
    }
    (package_root / "package_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(
        f"{report['sha256']}  {zip_path.name}\n", encoding="ascii"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
