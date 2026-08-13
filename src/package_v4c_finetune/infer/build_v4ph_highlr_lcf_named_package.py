#!/usr/bin/env python3
"""Build the named V4PH HighLR-LCF 30k listening package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


PACKAGE_DATE = "20260807"
LABEL = "LCF_30K"


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    checkpoint = args.checkpoint.resolve()
    generated = root / "generated"
    dataset = root / "input" / "dataset"
    full_audit_path = root / "full_audit.json"
    checkpoint_audit_path = root / "checkpoint_audit.json"
    full_audit = json.loads(full_audit_path.read_text(encoding="utf-8"))
    checkpoint_audit = json.loads(
        checkpoint_audit_path.read_text(encoding="utf-8")
    )
    if full_audit.get("status") != "ok" or full_audit.get("wav_count") != 54:
        raise ValueError("LCF full output audit has not passed")
    if checkpoint_audit.get("step") != 30000:
        raise ValueError("LCF final checkpoint audit has not passed")

    package_root = root / "package"
    package_name = f"V4PH_HighLR_LCF_30K_named_{PACKAGE_DATE}"
    named_root = package_root / package_name
    if named_root.exists():
        raise FileExistsError(f"listening package already exists: {named_root}")
    named_root.mkdir(parents=True)

    dataset_manifest = json.loads(
        (dataset / "manifest.json").read_text(encoding="utf-8")
    )
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

    records = []
    score_rows = []
    for cfg in ("cfg3", "cfg1"):
        source_dir = generated / f"{LABEL}_{cfg}"
        destination_dir = named_root / f"{LABEL}_{cfg}"
        wavs = sorted(source_dir.glob("*.wav"))
        if len(wavs) != 27:
            raise ValueError(f"expected 27 WAVs in {source_dir}, got {len(wavs)}")
        for wav in wavs:
            destination = destination_dir / wav.name
            hardlink_or_copy(wav, destination)
            records.append(
                {
                    "condition": "V4PH-30K-HIGHLR-LCF",
                    "cfg": int(cfg[-1]),
                    "group": wav.stem,
                    "path": str(destination.relative_to(named_root)).replace(
                        "\\", "/"
                    ),
                    "sha256": sha256_file(wav),
                }
            )
            score_rows.append(
                {
                    "组名": wav.stem,
                    "模型": "V4PH-30K-HIGHLR-LCF",
                    "CFG": int(cfg[-1]),
                    "音色相似度": "",
                    "旋律准确度": "",
                    "咬字清晰度": "",
                    "自然度": "",
                    "风格适配": "",
                    "相对已有HighLR": "",
                    "备注": "",
                }
            )

    with (named_root / "评分表.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    readme = """# V4PH HighLR-LCF 30K 实名听评包

本包只包含 LCF 30K 的固定 27 组输出，不包含、复制或匿名化已有 HighLR 输出。

- `LCF_30K_cfg3` 与 `LCF_30K_cfg1` 各 27 条；
- `References/<组名>/A.wav` 是音色参考，`B.wav` 是目标旋律和时长参考；
- sampling steps 32，seed 42，CFG 3.0 / 1.0；
- `manifest.json` 保存 54 条生成音频的逐文件 SHA256；
- `Technical` 保存 checkpoint、placement 和音频完整性审计。

请直接与此前已有的 HighLR 包比较。总 loss 不代替人耳评价。
"""
    (named_root / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "schema": "v4ph_highlr_lcf_30k_named_package_v1",
        "condition": "V4PH-30K-HIGHLR-LCF",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "step": 30000,
        "cfg": [3.0, 1.0],
        "sampling_steps": 32,
        "seed": 42,
        "groups": [group["name"] for group in groups],
        "wav_count": len(records),
        "files": records,
    }
    (named_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    technical = named_root / "Technical"
    technical.mkdir()
    shutil.copy2(full_audit_path, technical / full_audit_path.name)
    shutil.copy2(checkpoint_audit_path, technical / checkpoint_audit_path.name)
    for directory in sorted(generated.iterdir()):
        if directory.is_dir():
            shutil.copytree(
                directory / "_placement",
                technical / "placement_audits" / directory.name,
            )

    named_zip = package_root / f"{package_name}.zip"
    zip_tree(named_root, named_zip)
    report = {
        "schema": "v4ph_highlr_lcf_30k_named_package_report_v1",
        "status": "ok",
        "named_directory": str(named_root),
        "named_zip": str(named_zip),
        "named_zip_bytes": named_zip.stat().st_size,
        "named_zip_sha256": sha256_file(named_zip),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "group_count": len(groups),
        "wav_count": len(records),
        "score_rows": len(score_rows),
    }
    report_path = package_root / "package_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    named_zip.with_suffix(named_zip.suffix + ".sha256").write_text(
        f"{report['named_zip_sha256']}  {named_zip.name}\n", encoding="ascii"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
