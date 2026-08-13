#!/usr/bin/env python3
"""Build the named V5-P 2k trajectory package with EMA and raw outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

from audit_v5p_trajectory_outputs import (
    WEIGHT_SOURCES,
    condition_label,
    trajectory_contract,
)


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
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent))


def checkpoint_path(root, step, final_step):
    name = f"step_{step:06d}_final.pt" if step == final_step else f"step_{step:06d}.pt"
    return root / name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-family", choices=("V5P", "V5PG"), default="V5P")
    parser.add_argument("--package-date", default="20260810")
    args = parser.parse_args()

    root = args.root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    contract = trajectory_contract(args.model_family)
    steps = contract["steps"]
    prefix = contract["prefix"]
    final_step = contract["final_step"]
    interval_k = 1 if args.model_family == "V5PG" else 2
    package_name = (
        f"{prefix}_{steps[0] // 1000}K_{steps[-1] // 1000}K_"
        f"EMA_RAW_CFG1_named_{args.package_date}"
    )
    generated = root / "generated"
    dataset = root / "input" / "dataset"
    full_audit = json.loads((root / "full_audit.json").read_text(encoding="utf-8"))
    checkpoint_audit = json.loads((root / "checkpoint_audit.json").read_text(encoding="utf-8"))
    expected_wavs = len(steps) * len(WEIGHT_SOURCES) * 27
    if full_audit.get("status") != "ok" or full_audit.get("wav_count") != expected_wavs:
        raise ValueError(f"{prefix} trajectory full output audit has not passed")
    if (
        checkpoint_audit.get("status") != "ok"
        or checkpoint_audit.get("checkpoint_count") != len(steps)
    ):
        raise ValueError(f"{prefix} trajectory checkpoint audit has not passed")

    package_root = root / "package"
    named_root = package_root / package_name
    if named_root.exists():
        raise FileExistsError(f"named package already exists: {named_root}")
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

    checkpoint_records = {}
    for step in steps:
        path = checkpoint_path(checkpoint_dir, step, final_step)
        checkpoint_records[step] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    audit_records = {
        (int(item["step"]), item["weight_source"], item["group"]): item
        for item in full_audit["records"]
    }
    files = []
    score_rows = []
    for step in steps:
        for weight_source in WEIGHT_SOURCES:
            directory_name = condition_label(step, weight_source, prefix)
            source_dir = generated / directory_name
            destination_dir = named_root / directory_name
            wavs = sorted(source_dir.glob("*.wav"))
            if len(wavs) != 27:
                raise ValueError(f"expected 27 WAVs in {source_dir}, got {len(wavs)}")
            for wav in wavs:
                destination = destination_dir / wav.name
                hardlink_or_copy(wav, destination)
                record = audit_records[(step, weight_source.lower(), wav.stem)]
                files.append(
                    {
                        "step": step,
                        "weight_source": weight_source.lower(),
                        "cfg": 1.0,
                        "group": wav.stem,
                        "path": str(destination.relative_to(named_root)).replace("\\", "/"),
                        "sha256": record["wav_sha256"],
                    }
                )
                score_rows.append(
                    {
                        "步数": f"{step // 1000}K",
                        "权重": weight_source,
                        "组名": wav.stem,
                        "CFG": 1,
                        "音色相似度": "",
                        "旋律准确度": "",
                        "咬字清晰度": "",
                        "自然度": "",
                        "风格适配": "",
                        "总体偏好": "",
                        "备注": "",
                    }
                )

    with (named_root / "评分表.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    phase = "Phase C" if args.model_family == "V5PG" else "Phase B"
    readme = f"""# {prefix} {steps[0] // 1000}K--{steps[-1] // 1000}K EMA/raw CFG1 实名轨迹听评包

本包使用冻结的 27 组 SVS A/B 输入，对 {prefix} {phase} 每 {interval_k}k 保存点同时生成 EMA 与 raw 权重结果。

- {len(steps)} 个 checkpoint：{steps[0] // 1000}K 到 {steps[-1] // 1000}K；
- 每个 checkpoint 有 `EMA` 与 `RAW` 两个实名条件；
- CFG 固定为 1.0，sampling steps 32，seed 42；
- 每个条件 27 条，共 {expected_wavs} 条 WAV；
- `References/<组名>/A.wav` 是音色参考，`B.wav` 是目标旋律和时长参考；
- `Technical` 保存 checkpoint、输入合同、H placement、GAME 与音频完整性审计。

同一步数下先比较 EMA 与 RAW，再沿相同权重来源纵向比较训练轨迹。训练 loss 不替代听评。
"""
    (named_root / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "schema": f"{prefix.lower()}_trajectory_named_package_v1",
        "condition": f"{prefix} {phase} EMA/raw trajectory",
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "steps": list(steps),
        "weight_sources": [value.lower() for value in WEIGHT_SOURCES],
        "cfg": [1.0],
        "sampling_steps": 32,
        "seed": 42,
        "group_count": len(groups),
        "groups": [group["name"] for group in groups],
        "checkpoint_count": len(checkpoint_records),
        "checkpoints": checkpoint_records,
        "condition_count": len(steps) * len(WEIGHT_SOURCES),
        "wav_count": len(files),
        "files": files,
    }
    (named_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    technical = named_root / "Technical"
    technical.mkdir()
    for name in ("checkpoint_audit.json", "smoke_audit.json", "full_audit.json", "pipeline.log"):
        source = root / name
        if source.is_file():
            shutil.copy2(source, technical / name)
    audits_out = technical / "placement_audits"
    for directory in sorted(generated.iterdir()):
        if directory.is_dir():
            shutil.copytree(directory / "_placement", audits_out / directory.name)

    named_zip = package_root / f"{package_name}.zip"
    zip_tree(named_root, named_zip)
    report = {
        "schema": f"{prefix.lower()}_trajectory_named_package_report_v1",
        "status": "ok",
        "named_directory": str(named_root),
        "named_zip": str(named_zip),
        "named_zip_bytes": named_zip.stat().st_size,
        "named_zip_sha256": sha256_file(named_zip),
        "checkpoint_count": len(checkpoint_records),
        "condition_count": len(steps) * len(WEIGHT_SOURCES),
        "group_count": len(groups),
        "wav_count": len(files),
        "score_rows": len(score_rows),
    }
    report_path = package_root / "package_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    named_zip.with_suffix(named_zip.suffix + ".sha256").write_text(
        f"{report['named_zip_sha256']}  {named_zip.name}\n", encoding="ascii"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
