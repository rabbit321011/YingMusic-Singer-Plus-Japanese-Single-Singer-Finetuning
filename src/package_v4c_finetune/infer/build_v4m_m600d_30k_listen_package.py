#!/usr/bin/env python3
"""Build the blinded HighLR/M600-D 30k listening package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import zipfile


INPUT_ARCHIVE_SHA256 = (
    "f9949ec300c86ee2763f0d75f36eb7ff0c9368ca16b572b0a95b52b062ad2aeb"
)
PACKAGE_DATE = "20260806"


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
    if full_audit.get("status") != "ok" or full_audit.get("wav_count") != 108:
        raise ValueError("30k full output audit has not passed")

    package_root = root / "package"
    blind_name = f"V4M_M600D_30K_blind_{PACKAGE_DATE}"
    blind_root = package_root / blind_name
    technical_root = package_root / "technical"
    if blind_root.exists() or technical_root.exists():
        raise FileExistsError("30k listening package output already exists")
    blind_root.mkdir(parents=True)
    technical_root.mkdir(parents=True)

    conditions = ["HIGHLR_30K", "M600D_30K"]
    rng = random.Random(2026080630000)
    rng.shuffle(conditions)
    code_to_condition = {"A": conditions[0], "B": conditions[1]}

    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    groups = dataset_manifest.get("groups") or []
    if len(groups) != 27:
        raise ValueError("listening dataset must contain 27 groups")

    references = blind_root / "References"
    for group in groups:
        source = dataset / group["directory"]
        destination = references / group["directory"]
        for path in sorted(source.iterdir()):
            if path.is_file():
                hardlink_or_copy(path, destination / path.name)

    score_rows = []
    blind_files = []
    for code, condition in code_to_condition.items():
        for cfg in ("cfg3", "cfg1"):
            source_dir = generated / f"{condition}_{cfg}"
            destination_dir = blind_root / "Step_30K" / f"Model_{code}_{cfg}"
            wavs = sorted(source_dir.glob("*.wav"))
            if len(wavs) != 27:
                raise ValueError(f"expected 27 WAVs in {source_dir}, got {len(wavs)}")
            for wav in wavs:
                destination = destination_dir / wav.name
                hardlink_or_copy(wav, destination)
                blind_files.append(
                    {
                        "step": "30K",
                        "model_code": code,
                        "cfg": int(cfg[-1]),
                        "group": wav.stem,
                        "path": str(destination.relative_to(blind_root)).replace("\\", "/"),
                        "sha256": sha256_file(wav),
                    }
                )
                score_rows.append(
                    {
                        "步数": "30K",
                        "组名": wav.stem,
                        "模型代码": f"Model_{code}",
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

    with (blind_root / "评分表.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    readme = """# M600-D 30k 同步数听评包

本包只比较相同 30k 训练步数下的 338M HighLR control 与 M600-D：

- `Step_30K` 中 Model A 与 Model B 分别对应一个匿名模型；
- 每个模型均有 CFG 3 与 CFG 1；
- `References/<组名>/A.wav` 是音色参考，`B.wav` 是目标旋律/时长参考；
- 每项按 `评分表.csv` 的 1--10 分填写，完成评分前不要查看单独保存的解盲密钥。

同组、同 CFG 横向比较是容量裁决主口径。训练 Loss 不属于本听评的评分证据。
"""
    (blind_root / "README.md").write_text(readme, encoding="utf-8")
    blind_manifest = {
        "schema": "v4m_m600d_30k_blind_package_v1",
        "input_archive_sha256": INPUT_ARCHIVE_SHA256,
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "steps": [30000],
        "cfg": [3.0, 1.0],
        "sampling_steps": 32,
        "seed": 42,
        "groups": [group["name"] for group in groups],
        "wav_count": len(blind_files),
        "files": blind_files,
    }
    (blind_root / "manifest.json").write_text(
        json.dumps(blind_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    checkpoint_paths = {
        "HIGHLR_30K": Path(
            "${REMOTE_ROOT}/YingMusic-Singer-Plus/ckpts/"
            "plus_ja_sft_v4ph_30k_highlr/step_030000_final.pt"
        ),
        "M600D_30K": root / "publish" / "V4M_M600D_step_030000.pt",
    }
    key = {
        "schema": "v4m_m600d_30k_unblinding_key_v1",
        "mapping": {
            code: {
                "condition": condition,
                "checkpoint": str(checkpoint_paths[condition]),
                "checkpoint_sha256": sha256_file(checkpoint_paths[condition]),
            }
            for code, condition in code_to_condition.items()
        },
    }
    (technical_root / "解盲密钥.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in ("smoke_audit.json", "full_audit.json", "pipeline.log"):
        shutil.copy2(root / name, technical_root / name)
    for path in sorted((root / "publish").glob("*.audit.json")):
        shutil.copy2(path, technical_root / path.name)
    for path in sorted((root / "publish").glob("*.sha256")):
        shutil.copy2(path, technical_root / path.name)
    audits_out = technical_root / "placement_audits"
    for directory in sorted(generated.iterdir()):
        if directory.is_dir():
            shutil.copytree(directory / "_placement", audits_out / directory.name)

    blind_zip = package_root / f"{blind_name}.zip"
    technical_zip = package_root / f"V4M_M600D_30K_technical_{PACKAGE_DATE}.zip"
    zip_tree(blind_root, blind_zip)
    zip_tree(technical_root, technical_zip)
    package_report = {
        "schema": "v4m_m600d_30k_package_report_v1",
        "status": "ok",
        "blind_zip": str(blind_zip),
        "blind_zip_bytes": blind_zip.stat().st_size,
        "blind_zip_sha256": sha256_file(blind_zip),
        "technical_zip": str(technical_zip),
        "technical_zip_bytes": technical_zip.stat().st_size,
        "technical_zip_sha256": sha256_file(technical_zip),
        "wav_count": len(blind_files),
        "score_rows": len(score_rows),
    }
    (package_root / "package_report.json").write_text(
        json.dumps(package_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    blind_zip.with_suffix(blind_zip.suffix + ".sha256").write_text(
        f"{package_report['blind_zip_sha256']}  {blind_zip.name}\n", encoding="ascii"
    )
    print(json.dumps(package_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
