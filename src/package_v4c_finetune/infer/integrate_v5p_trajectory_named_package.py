#!/usr/bin/env python3
"""Register the verified V5-P trajectory package in the local SVS evaluation root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime


STEPS = tuple(range(2000, 40001, 2000))
WEIGHTS = ("EMA", "RAW")
EXPECTED_PACKAGE_SHA256 = "9c29736929eaee933ffceae8f3c4606e4efbbedf762371857ba2a44ade5b096a"
DATASET_MANIFEST_SHA256 = "307a19f8d12302aaa99b510078fdc8450c38cff795cc771baa0ea0534ff551a1"
ALIGNMENT_MANIFEST_SHA256 = "eb7a86a71059d9c0d8d437a306929741480d0812ead01b33b140692d973eff51"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_label(step, weight):
    return f"V5P_{step // 1000:02d}K_{weight}_cfg1"


def model_label(step, weight):
    return f"V5-P {step // 1000:02d}K {weight}"


def checkpoint_name(step):
    return "step_040000_final.pt" if step == 40000 else f"step_{step:06d}.pt"


def hardlink_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".v5p.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sidecar", type=Path, required=True)
    parser.add_argument("--package-report", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    archive = args.archive.resolve()
    archive_sidecar = args.archive_sidecar.resolve()
    package_report_path = args.package_report.resolve()
    evaluation_root = args.evaluation_root.resolve()
    manifest_path = evaluation_root / "manifest.json"
    readme_path = evaluation_root / "README.md"
    score_path = evaluation_root / "评分表.csv"

    package_manifest = json.loads((package_root / "manifest.json").read_text(encoding="utf-8"))
    full_audit = json.loads(
        (package_root / "Technical" / "full_audit.json").read_text(encoding="utf-8")
    )
    package_report = json.loads(package_report_path.read_text(encoding="utf-8"))
    if package_manifest.get("schema") != "v5p_trajectory_named_package_v1":
        raise ValueError("unexpected package manifest schema")
    if package_manifest.get("wav_count") != 1080 or package_manifest.get("condition_count") != 40:
        raise ValueError("package count mismatch")
    if full_audit.get("status") != "ok" or full_audit.get("wav_count") != 1080:
        raise ValueError("full audit has not passed")
    if package_report.get("status") != "ok" or package_report.get("wav_count") != 1080:
        raise ValueError("package report has not passed")
    if sha256_file(archive) != EXPECTED_PACKAGE_SHA256:
        raise ValueError("package archive SHA256 mismatch")

    source_dataset = Path(
        "E:/AIscene/AISVC-midi-web/exports/测验集合成选择2_SVS评测集_20260727_141113"
    )
    alignment_dataset = Path(
        "E:/AIscene/AISVC-midi-web/exports/测验集合成选择2_V4H_phone_alignment_20260729"
    )
    if sha256_file(source_dataset / "manifest.json") != DATASET_MANIFEST_SHA256:
        raise ValueError("local evaluation dataset manifest mismatch")
    if sha256_file(alignment_dataset / "manifest.json") != ALIGNMENT_MANIFEST_SHA256:
        raise ValueError("local H alignment manifest mismatch")

    expected_directories = [condition_label(step, weight) for step in STEPS for weight in WEIGHTS]
    for directory_name in expected_directories:
        source = package_root / directory_name
        audits = package_root / "Technical" / "placement_audits" / directory_name
        if len(list(source.glob("*.wav"))) != 27 or len(list(audits.glob("*.json"))) != 27:
            raise ValueError(f"source condition count mismatch: {directory_name}")
        if (evaluation_root / directory_name).exists():
            raise FileExistsError(f"evaluation condition already exists: {directory_name}")

    technical_name = "_V5P_trajectory_2k_40k_ema_raw_cfg1_20260810"
    technical_target = evaluation_root / technical_name
    if technical_target.exists():
        raise FileExistsError(technical_target)

    root_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if root_manifest.get("sourceDataset", "").replace("\\", "/") != str(source_dataset).replace("\\", "/"):
        raise ValueError("evaluation root source dataset mismatch")
    existing_ids = {item.get("modelId") for item in root_manifest.get("models", [])}
    new_ids = {f"V5P_{step // 1000:02d}K_{weight}" for step in STEPS for weight in WEIGHTS}
    if existing_ids & new_ids:
        raise ValueError("V5-P trajectory is already registered")

    with score_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        score_rows = list(reader)
    expected_fields = [
        "组名",
        "模型",
        "CFG",
        "音色相似度",
        "旋律准确度",
        "咬字清晰度",
        "自然度",
        "风格适配",
        "总体偏好",
        "备注",
        "工程方法",
    ]
    if fieldnames != expected_fields:
        raise ValueError(f"unexpected score table columns: {fieldnames}")
    existing_score_keys = {
        (row["组名"], row["模型"], row["CFG"], row["工程方法"]) for row in score_rows
    }

    created_directories = []
    try:
        for directory_name in expected_directories:
            source = package_root / directory_name
            audits = package_root / "Technical" / "placement_audits" / directory_name
            destination = evaluation_root / directory_name
            destination.mkdir()
            created_directories.append(destination)
            for wav in sorted(source.glob("*.wav")):
                hardlink_or_copy(wav, destination / wav.name)
            for audit in sorted(audits.glob("*.json")):
                hardlink_or_copy(audit, destination / "_placement" / audit.name)

        technical_target.mkdir()
        created_directories.append(technical_target)
        for name in ("README.md", "manifest.json", "评分表.csv"):
            shutil.copy2(package_root / name, technical_target / name)
        shutil.copytree(package_root / "Technical", technical_target / "Technical")
        shutil.copy2(package_report_path, technical_target / "package_report.json")
        hardlink_or_copy(archive, technical_target / archive.name)
        shutil.copy2(archive_sidecar, technical_target / archive_sidecar.name)

        groups = root_manifest.get("groups") or []
        if len(groups) != 27:
            raise ValueError("evaluation root must contain 27 groups")
        checkpoint_records = package_manifest["checkpoints"]
        full_validation = {
            "filesPerFolder": 27,
            "inputConditionMismatches": 0,
            "emaRawIdenticalWavPairs": 0,
            "maxDurationDeltaSeconds": full_audit["max_duration_delta_seconds"],
            "minimumRms": full_audit["minimum_rms"],
            "maximumPeak": full_audit["maximum_peak"],
            "sourcePackageManifestSHA256": sha256_file(package_root / "manifest.json"),
        }
        new_models = []
        for step in STEPS:
            checkpoint = checkpoint_records[str(step)]
            for weight in WEIGHTS:
                model_id = f"V5P_{step // 1000:02d}K_{weight}"
                new_models.append(
                    {
                        "label": model_label(step, weight),
                        "modelId": model_id,
                        "checkpoint": checkpoint["path"],
                        "inferenceCheckpoint": checkpoint["path"],
                        "checkpointSchema": "v5p_training_checkpoint_v1",
                        "checkpointSHA256": checkpoint["sha256"],
                        "globalStep": step,
                        "weightSource": weight.lower(),
                        "vae": "E:/AIscene/YingMusic_Singer_Plus/ckpts/stable_audio_2_0_vae_20hz_official.ckpt",
                        "placementMode": "phone_pul",
                        "alignmentDataset": str(alignment_dataset).replace("\\", "/"),
                        "runtime": "${REMOTE_ROOT}/V5P_20260810/eval_trajectory_2k_40k_ema_raw_cfg1/input/runtime",
                        "training": {
                            "phase": "joint",
                            "scheduleProfile": "v5p_two_cosine",
                            "warmupSteps": 2000,
                            "firstDecayEnd": 28000,
                            "midLearningRate": 1e-5,
                            "maxSteps": 40000,
                            "poolPolicy": "KEEP_LONG_DEDUP_SHORT",
                            "samplingPolicy": "NATURAL_RECORD",
                        },
                        "melodyTeacher": {
                            "name": "OpenVPI/GAME medium K4",
                            "commit": "4ad815c90dfe2442730f3fdc866fd23e737cbc97",
                            "checkpointSHA256": "e9904159fb0646e1a352b9d2bc74615547cfa3e32d45c7464d440ac142846d93",
                            "trainingManifestSHA256": "891add8fab5e8507b53c7e35fcb7f9d6a5c828490d5d988beb4c347a9c3a53ce",
                            "nsteps": 4,
                            "baseSeed": 20260730,
                        },
                        "folders": {"cfg1": condition_label(step, weight)},
                        "validation": full_validation,
                    }
                )
                for group in groups:
                    key = (group["name"], model_label(step, weight), "1", "")
                    if key in existing_score_keys:
                        raise ValueError(f"duplicate score row: {key}")
                    score_rows.append(
                        {
                            "组名": group["name"],
                            "模型": model_label(step, weight),
                            "CFG": "1",
                            "音色相似度": "",
                            "旋律准确度": "",
                            "咬字清晰度": "",
                            "自然度": "",
                            "风格适配": "",
                            "总体偏好": "",
                            "备注": "",
                            "工程方法": "",
                        }
                    )

        root_manifest["models"].extend(new_models)
        root_manifest["v5pTrajectory"] = {
            "schema": "v5p_trajectory_named_package_v1",
            "packageDirectory": technical_name,
            "packageArchiveSHA256": EXPECTED_PACKAGE_SHA256,
            "datasetManifestSHA256": DATASET_MANIFEST_SHA256,
            "alignmentManifestSHA256": ALIGNMENT_MANIFEST_SHA256,
            "steps": list(STEPS),
            "weightSources": [value.lower() for value in WEIGHTS],
            "cfg": [1.0],
            "samplingSteps": 32,
            "seed": 42,
            "checkpointCount": 20,
            "conditionCount": 40,
            "wavCount": 1080,
            "scoreRows": 1080,
            "directories": expected_directories,
            "validation": full_validation,
        }
        validation = root_manifest["validation"]
        validation["folders"] = int(validation["folders"]) + 40
        validation["checkpointFolders"] = int(validation["checkpointFolders"]) + 40
        validation["totalWavs"] = int(validation["totalWavs"]) + 1080
        validation["maxDurationDeltaSeconds"] = max(
            float(validation["maxDurationDeltaSeconds"]),
            float(full_audit["max_duration_delta_seconds"]),
        )
        validation["emaRawIdenticalWavPairs"] = 0
        root_manifest["updatedAt"] = "2026/8/10"

        readme = readme_path.read_text(encoding="utf-8")
        readme = readme.replace(
            "本目录使用同一套 27 组 A/B 输入，对十八个 SVS checkpoint 分别进行 CFG 3 和 CFG 1 推理。",
            "本目录使用同一套 27 组 A/B 输入，包含十八个既有 SVS checkpoint 的 CFG 3/1 结果，以及 V5-P 2K--40K 每 2K 的 EMA/raw、CFG 1 实名轨迹。",
        )
        marker = "V4IjPH_30k_cfg1/\n```"
        additions = "\n".join(expected_directories)
        if marker not in readme:
            raise ValueError("README result directory marker not found")
        readme = readme.replace(marker, f"V4IjPH_30k_cfg1/\n{additions}\n```", 1)
        section = """## V5-P 2K--40K EMA/raw 轨迹

V5-P 轨迹按用户批准的实名规则登记：20 个 2K 间隔 checkpoint，每个 checkpoint 同时使用 EMA 与 raw 权重，CFG 固定为 1.0。每个条件使用同一套 27 组输入、32 sampling steps、seed 42、official VAE、H/PUL placement 和 GAME medium K=4，共 40 个目录、1080 条 WAV。

同一步数下比较 `EMA` 与 `RAW`，同一权重来源沿步数纵向比较。`_V5P_trajectory_2k_40k_ema_raw_cfg1_20260810/` 保存原始实名包、逐文件 manifest、checkpoint SHA256、完整审计和 placement/GAME 审计。

"""
        if "## V5-P 2K--40K EMA/raw 轨迹" in readme:
            raise ValueError("README V5-P trajectory section already exists")
        readme = readme.replace("## 工程方法", section + "## 工程方法", 1)
        readme = readme.replace(
            "- 38 个结果目录：36 个 checkpoint 条件 + 2 个工程方法",
            "- 78 个结果目录：76 个 checkpoint/权重条件 + 2 个工程方法",
        )
        readme = readme.replace("- 合计 1026 条 WAV", "- 合计 2106 条 WAV")

        backup = evaluation_root / "_registry_backup_before_v5p_trajectory_20260810"
        backup.mkdir()
        shutil.copy2(manifest_path, backup / "manifest.json")
        shutil.copy2(readme_path, backup / "README.md")
        shutil.copy2(score_path, backup / "评分表.csv")

        write_json_atomic(manifest_path, root_manifest)
        readme_tmp = readme_path.with_suffix(".md.v5p.tmp")
        readme_tmp.write_text(readme, encoding="utf-8")
        os.replace(readme_tmp, readme_path)
        score_tmp = score_path.with_suffix(".csv.v5p.tmp")
        with score_tmp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(score_rows)
        os.replace(score_tmp, score_path)
    except Exception:
        for path in reversed(created_directories):
            if path.exists():
                shutil.rmtree(path)
        raise

    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with score_path.open("r", encoding="utf-8-sig", newline="") as handle:
        final_score_rows = sum(1 for _ in csv.DictReader(handle))
    result = {
        "status": "ok",
        "registered_models": len(new_models),
        "registered_directories": len(expected_directories),
        "registered_wavs": 1080,
        "manifest_models": len(final_manifest["models"]),
        "manifest_total_wavs": final_manifest["validation"]["totalWavs"],
        "score_rows": final_score_rows,
        "technical_directory": str(technical_target),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
