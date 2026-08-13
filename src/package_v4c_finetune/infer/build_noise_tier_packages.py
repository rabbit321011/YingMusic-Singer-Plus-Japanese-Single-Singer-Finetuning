#!/usr/bin/env python3
"""Merge scorer outputs and build ten deterministic four-section listening packs."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


AXES = [
    {
        "order": 1,
        "id": "singmos",
        "name": "SingMOS",
        "source": "singmos_v1.csv",
        "column": "score",
        "purpose": "面向歌声的自然度/MOS 预测；以 SingMOS-v1 歌声主观评分训练。",
        "model": "wav2vec2-base-960 + 帧级回归均值，16 kHz 输入。",
        "domain": "歌声；十个候选里与本任务域最接近。",
    },
    {
        "order": 2,
        "id": "singmos_pro",
        "name": "SingMOS-Pro",
        "source": "singmos_pro.csv",
        "column": "score",
        "purpose": "面向多来源歌声的综合质量/MOS 预测；以 SingMOS-Pro 训练。",
        "model": "wav2vec2-large-ll60k + domain embedding + 1--5 有界回归，16 kHz 输入。",
        "domain": "歌声；容量和训练覆盖大于 SingMOS-v1。",
    },
    {
        "order": 3,
        "id": "nisqa_overall",
        "name": "NISQA_Overall",
        "source": "nisqa.csv",
        "column": "mos_pred",
        "purpose": "无参考通信语音总体质量预测，同时建模噪声、染色、断续与响度。",
        "model": "NISQA v2.0 多维 CNN + Self-Attention 官方权重。",
        "domain": "传输/通信语音，不是歌声；用于检验跨域敏感性。",
    },
    {
        "order": 4,
        "id": "nisqa_noisiness",
        "name": "NISQA_Noisiness",
        "source": "nisqa.csv",
        "column": "noi_pred",
        "purpose": "NISQA v2.0 的 Noisiness 质量维度；高分表示主观噪声损伤更小。",
        "model": "与 NISQA Overall 共用同一多维模型和一次前向。",
        "domain": "通信语音噪声维度，不是专门的歌声毛刺检测器。",
    },
    {
        "order": 5,
        "id": "dnsmos_p808",
        "name": "DNSMOS_P808",
        "source": "dnsmos.csv",
        "column": "p808_mos",
        "purpose": "DNS Challenge 的 ITU-T P.808 风格无参考总体质量预测。",
        "model": "官方 ONNX P808 模型；16 kHz、9.01 秒滑窗后取均值。",
        "domain": "降噪/噪声语音；可能对宽带噪声敏感，对歌声周期毛刺未必敏感。",
    },
    {
        "order": 6,
        "id": "dnsmos_sig",
        "name": "DNSMOS_SIG",
        "source": "dnsmos.csv",
        "column": "sig_mos",
        "purpose": "DNSMOS 的 SIG 维度，估计前景语音信号本身的主观质量。",
        "model": "官方非个性化 DNSMOS SIG/BAK/OVRL ONNX 模型中的 SIG 输出。",
        "domain": "噪声语音；理论上比 BAK 更接近破音边缘和哑嗓质感。",
    },
    {
        "order": 7,
        "id": "utmos22",
        "name": "UTMOS22",
        "source": "utmos22.csv",
        "column": "score",
        "purpose": "VoiceMOS Challenge 2022 的合成语音自然度/MOS 预测。",
        "model": "UTMOS strong learner：wav2vec2 + domain/judge embedding + BLSTM，16 kHz。",
        "domain": "TTS/VC 语音，不是歌声。",
    },
    {
        "order": 8,
        "id": "utmosv2",
        "name": "UTMOSv2",
        "source": "utmosv2.csv",
        "column": "score",
        "purpose": "VoiceMOS Challenge 2024 高质量合成语音自然度预测。",
        "model": "wav2vec2-base 与四路多分辨率谱 EfficientNetV2 融合，官方 fold0 权重。",
        "domain": "高质量 TTS/VC 语音；兼看时域 SSL 与细谱纹理。",
    },
    {
        "order": 9,
        "id": "mosnet",
        "name": "MOSNet",
        "source": "mosnet.csv",
        "column": "score",
        "purpose": "最早一代面向 voice conversion 的无参考 MOS 预测。",
        "model": "VCC2018 主观评分训练的 CNN-BLSTM 幅度谱模型，16 kHz。",
        "domain": "VC 语音；作为经典谱模型基线。",
    },
    {
        "order": 10,
        "id": "wvmos",
        "name": "WV-MOS",
        "source": "wvmos.csv",
        "column": "score",
        "purpose": "以 wav2vec2 表示预测有噪/增强语音的 MOS。",
        "model": "facebook/wav2vec2-base + 两层回归头，16 kHz。",
        "domain": "语音质量/增强语音；用于检验 SSL 表示对本类纹理的响应。",
    },
]

POOL_TOTALS = {
    "training_original": 2000,
    "self_v4ph": 500,
    "self_v4fg": 500,
    "cross_source": 54,
}

SECTION_DIRS = {
    "training_original": "01_训练原样本",
    "self_v4ph": "02_V4PH自克隆",
    "self_v4fg": "03_V4fg自克隆",
    "cross_source": "04_跨源克隆_54条",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-archives", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nice_ceil(value: float) -> float:
    if value <= 0:
        raise ValueError(value)
    exponent = math.floor(math.log10(value))
    scale = 10.0**exponent
    normalized = value / scale
    for candidate in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0):
        if normalized <= candidate + 1e-12:
            return candidate * scale
    raise AssertionError(normalized)


def next_nice(value: float) -> float:
    exponent = math.floor(math.log10(value))
    scale = 10.0**exponent
    normalized = value / scale
    for candidate in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0):
        if candidate > normalized + 1e-10:
            return candidate * scale
    return 2.0 * value


def estimate_range(values: list[float]) -> dict[str, Any]:
    if len(values) != 2000:
        raise ValueError(f"range calibration expects 2000 training originals, got {len(values)}")
    q005, q995 = np.percentile(values, [0.5, 99.5]).tolist()
    width = nice_ceil((q995 - q005) / 10.0)
    while True:
        slack = 10.0 * width - (q995 - q005)
        quantum = width / 2.0
        centered_lower = q005 - slack / 2.0
        lower = math.floor(centered_lower / quantum + 1e-10) * quantum
        if lower <= q005 + 1e-10 and lower + 10.0 * width >= q995 - 1e-10:
            break
        width = next_nice(width)
    upper = lower + 10.0 * width
    edges = [lower + index * width for index in range(11)]
    return {
        "method": "training_original q0.5--q99.5, equal-width nice intervals",
        "q0_5": q005,
        "q99_5": q995,
        "observed_min": min(values),
        "observed_max": max(values),
        "nominal_lower": lower,
        "nominal_upper": upper,
        "bin_width": width,
        "edges": edges,
    }


def assign_bin(score: float, edges: list[float]) -> int:
    return bisect.bisect_right(edges[1:-1], score) + 1


def number_token(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def bin_directory(index: int, edges: list[float]) -> str:
    if index == 1:
        return f"01_below_{number_token(edges[1])}"
    if index == 10:
        return f"10_at_least_{number_token(edges[9])}"
    return (
        f"{index:02d}_{number_token(edges[index - 1])}_to_"
        f"{number_token(edges[index])}"
    )


def evenly_spaced_selection(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["score"]), str(row["item_id"])))
    if len(ordered) <= limit:
        return ordered
    positions = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    if len(set(positions)) != limit:
        raise AssertionError((len(ordered), positions))
    return [ordered[position] for position in positions]


def link_audio(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def safe_item_token(item_id: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", item_id).strip("._-")
    token = token[:80] or "item"
    digest = hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:8]
    return f"{token}__{digest}"


def collect_scores(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    manifest = read_csv(root / "canonical_audio_manifest.csv")
    if len(manifest) != 3054:
        raise RuntimeError(f"canonical manifest rows={len(manifest)}")
    item_ids = [row["item_id"] for row in manifest]
    if len(set(item_ids)) != len(item_ids):
        raise RuntimeError("duplicate item_id in canonical manifest")
    expected = set(item_ids)

    merged: dict[str, dict[str, float]] = {item_id: {} for item_id in item_ids}
    for axis in AXES:
        score_rows = read_csv(root / "scores" / str(axis["source"]))
        if len(score_rows) != 3054 or {row["item_id"] for row in score_rows} != expected:
            raise RuntimeError(f"incomplete scorer output: {axis['id']}")
        for row in score_rows:
            value = float(row[str(axis["column"])])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {axis['id']} score: {row['item_id']}")
            merged[row["item_id"]][str(axis["id"])] = value

    hf_rows = read_csv(root / "scores" / "highfreq_reference.csv")
    if len(hf_rows) != 3054 or {row["item_id"] for row in hf_rows} != expected:
        raise RuntimeError("incomplete highfreq_reference output")
    hf_columns = [column for column in hf_rows[0] if column != "item_id"]
    for row in hf_rows:
        for column in hf_columns:
            value = float(row[column])
            if not math.isfinite(value):
                raise ValueError(f"non-finite highfreq value: {row['item_id']} {column}")
            merged[row["item_id"]][column] = value

    full_rows: list[dict[str, Any]] = []
    for row in manifest:
        full_rows.append({**row, **merged[row["item_id"]]})
    return full_rows, merged


def model_info_markdown() -> str:
    lines = [
        "# 十个评分轴的设计目的",
        "",
        "这些模型都输出主观质量代理分数，但训练域不同。本实验不假定谁天然能识别歌声毛刺，而是用十档包直接验证分数是否随目标噪音质感单调变化。",
        "",
        "| # | 评分轴 | 设计目的 | 结构/输入 | 训练域提醒 |",
        "|---:|---|---|---|---|",
    ]
    for axis in AXES:
        lines.append(
            f"| {axis['order']} | {axis['name']} | {axis['purpose']} | "
            f"{axis['model']} | {axis['domain']} |"
        )
    lines.extend(
        [
            "",
            "## 未编号高频对照",
            "",
            "每条音频另以 48 kHz 高质量重采样计算 8--20 kHz 与 12--20 kHz 的能量占比和谱平坦度中位数，只作为诊断参照，不生成第 11 个分阶包。所有统计截止 20 kHz，不把源文件 22.05 kHz Nyquist 以上的空频带当成信息。",
            "",
        ]
    )
    return "\n".join(lines)


def package_readme(axis: dict[str, Any], calibration: dict[str, Any]) -> str:
    return f"""# {axis['name']} 十档分阶包

## 模型是什么

- 设计目的：{axis['purpose']}
- 模型信息：{axis['model']}
- 训练域提醒：{axis['domain']}

## 怎么听

评分按模型原始方向排列：第 01 档最低，第 10 档最高。建议依次听同一部分的 01→10，判断目标问题“随发声附着的毛刺、破音边缘、不自然哑嗓沙声”是否稳定减少。不要把音高、唱法或文本清晰度差异误当成噪音排序。

## 四个部分

1. `01_训练原样本`：正式训练池 2,000 条评分后，每档最多 10 条。
2. `02_V4PH自克隆`：正式训练池独立抽取的 500 条，按训练配方自克隆后每档最多 10 条。
3. `03_V4fg自克隆`：另一组正式训练池 500 条，按 V4fg 训练配方自克隆后每档最多 10 条。
4. `04_跨源克隆_54条`：固定 27 组 × V4PH/V4fg，两模型共 54 条全部保留并入档，不设每档上限。

## 分档方法

- 只用 2,000 条训练原样本估计有效范围；取分数 q0.5--q99.5，再用易读的等宽区间覆盖。
- 不是十分位数分档。十档宽度固定为 `{calibration['bin_width']:.6g}`。
- 名义范围为 `{calibration['nominal_lower']:.6g}` 到 `{calibration['nominal_upper']:.6g}`；首档吸收更低离群值，末档吸收更高离群值。
- 原样本校准观测范围 `{calibration['observed_min']:.6g}` 到 `{calibration['observed_max']:.6g}`，q0.5=`{calibration['q0_5']:.6g}`，q99.5=`{calibration['q99_5']:.6g}`。
- 某档超过 10 条时，按分数排序后确定性抽取 10 个等间隔秩，不随机挑样本。

## 音频口径

全部评分和试听文件是同一份 44.1 kHz 单声道 PCM24 音频，只施加线性增益统一到 `-28 dBFS RMS`。未降噪、未限幅、未改频谱；全池最高峰值低于 `-1 dBFS`。因此模型和人听到的是同一个波形版本。

## 文件

- `bins.csv`：各档边界、三个候选池的全池占比、入选数和跨源数量。
- `all_scores.csv`：3,054 条在本评分轴的分数、档位和未编号高频对照。
- `selected_manifest.csv`：包内每个 WAV 的来源、分数、档位、SHA256 与相对路径。

这份包只回答“该评分轴是否能按目标噪音质感排序”，不能用模型原始 MOS 数字直接宣称歌声绝对质量。
"""


def build_one_package(
    root: Path,
    package_root: Path,
    axis: dict[str, Any],
    all_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    axis_id = str(axis["id"])
    package_name = f"{int(axis['order']):02d}_{axis['name']}"
    destination = package_root / package_name
    destination.mkdir(parents=True)

    calibration_values = [
        float(row[axis_id]) for row in all_rows if row["section"] == "training_original"
    ]
    calibration = estimate_range(calibration_values)
    edges = list(calibration["edges"])
    rows = [
        {
            **row,
            "score": float(row[axis_id]),
            "bin_index": assign_bin(float(row[axis_id]), edges),
        }
        for row in all_rows
    ]

    bins_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for bin_index in range(1, 11):
        bin_rows = [row for row in rows if int(row["bin_index"]) == bin_index]
        stats: dict[str, Any] = {
            "bin_index": bin_index,
            "directory": bin_directory(bin_index, edges),
            "nominal_lower": edges[bin_index - 1],
            "nominal_upper": edges[bin_index],
            "effective_lower": "-inf" if bin_index == 1 else edges[bin_index - 1],
            "effective_upper": "+inf" if bin_index == 10 else edges[bin_index],
        }
        for section in ("training_original", "self_v4ph", "self_v4fg"):
            pool_rows = [row for row in bin_rows if row["section"] == section]
            chosen = evenly_spaced_selection(pool_rows, 10)
            selected.extend(chosen)
            stats[f"{section}_count"] = len(pool_rows)
            stats[f"{section}_percent"] = len(pool_rows) * 100.0 / POOL_TOTALS[section]
            stats[f"{section}_selected"] = len(chosen)
        cross = [row for row in bin_rows if row["section"] == "cross_source"]
        selected.extend(sorted(cross, key=lambda row: (row["clone_model"], row["item_id"])))
        stats["cross_source_count"] = len(cross)
        stats["cross_v4ph_count"] = sum(row["clone_model"] == "v4ph" for row in cross)
        stats["cross_v4fg_count"] = sum(row["clone_model"] == "v4fg" for row in cross)
        bins_rows.append(stats)

    if sum(row["section"] == "cross_source" for row in selected) != 54:
        raise RuntimeError(f"{axis_id}: cross-source selection did not preserve all 54 files")

    selected_manifest: list[dict[str, Any]] = []
    for row in selected:
        bin_dir = bin_directory(int(row["bin_index"]), edges)
        section_dir = SECTION_DIRS[str(row["section"])]
        model_prefix = (
            f"{row['clone_model']}_" if row["section"] == "cross_source" else ""
        )
        filename = (
            f"score_{float(row['score']):.6f}__{model_prefix}"
            f"{safe_item_token(str(row['item_id']))}.wav"
        )
        relative = Path(section_dir) / bin_dir / filename
        link_audio(Path(str(row["normalized_path"])), destination / relative)
        selected_manifest.append(
            {
                "package": package_name,
                "scorer_id": axis_id,
                "bin_index": row["bin_index"],
                "section": row["section"],
                "clone_model": row["clone_model"],
                "item_id": row["item_id"],
                "score": row["score"],
                "source_sample_id": row["source_sample_id"],
                "source_path": row["source_path"],
                "normalized_sha256": row["normalized_sha256"],
                "package_relative_path": relative.as_posix(),
            }
        )

    all_score_rows = [
        {
            "item_id": row["item_id"],
            "section": row["section"],
            "clone_model": row["clone_model"],
            "candidate_index": row["candidate_index"],
            "score": row["score"],
            "bin_index": row["bin_index"],
            "energy_ratio_8_20k_median": row["energy_ratio_8_20k_median"],
            "spectral_flatness_8_20k_median": row[
                "spectral_flatness_8_20k_median"
            ],
            "energy_ratio_12_20k_median": row["energy_ratio_12_20k_median"],
            "spectral_flatness_12_20k_median": row[
                "spectral_flatness_12_20k_median"
            ],
            "normalized_sha256": row["normalized_sha256"],
        }
        for row in rows
    ]
    write_csv(destination / "bins.csv", bins_rows)
    write_csv(destination / "all_scores.csv", all_score_rows)
    write_csv(destination / "selected_manifest.csv", selected_manifest)
    (destination / "README.md").write_text(
        package_readme(axis, calibration), encoding="utf-8"
    )
    (destination / "range_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "order": axis["order"],
        "scorer_id": axis_id,
        "name": axis["name"],
        "package_directory": package_name,
        "selected_total": len(selected_manifest),
        "training_selected": sum(
            row["section"] == "training_original" for row in selected_manifest
        ),
        "self_v4ph_selected": sum(
            row["section"] == "self_v4ph" for row in selected_manifest
        ),
        "self_v4fg_selected": sum(
            row["section"] == "self_v4fg" for row in selected_manifest
        ),
        "cross_source_selected": sum(
            row["section"] == "cross_source" for row in selected_manifest
        ),
        "calibration": calibration,
    }


def create_archive(package_dir: Path, archive_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                zf.write(path, (Path(package_dir.name) / path.relative_to(package_dir)).as_posix())
    os.replace(temporary, archive_path)
    return {
        "archive": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    package_root = root / "packages"
    archive_root = root / "package_archives"
    if package_root.parent != root or archive_root.parent != root:
        raise RuntimeError("package output escaped experiment root")
    for output in (package_root, archive_root):
        if output.exists():
            if not args.overwrite:
                raise FileExistsError(f"output exists; pass --overwrite: {output}")
            shutil.rmtree(output)
        output.mkdir(parents=True)

    print("[load] validating and merging 10 axes over 3054 files", flush=True)
    all_rows, _ = collect_scores(root)
    axis_columns = [str(axis["id"]) for axis in AXES]
    hf_columns = [
        "energy_ratio_8_20k_median",
        "spectral_flatness_8_20k_median",
        "energy_ratio_12_20k_median",
        "spectral_flatness_12_20k_median",
        "active_frame_fraction",
    ]
    combined_fields = [
        "item_id",
        "section",
        "clone_model",
        "candidate_index",
        "source_sample_id",
        "source_path",
        "normalized_path",
        "normalized_sha256",
        *axis_columns,
        *hf_columns,
    ]
    write_csv(root / "all_scores.csv", all_rows, combined_fields)
    (package_root / "MODEL_INFO.md").write_text(model_info_markdown(), encoding="utf-8")

    summaries: list[dict[str, Any]] = []
    for axis in AXES:
        started = time.monotonic()
        summary = build_one_package(root, package_root, axis, all_rows)
        summaries.append(summary)
        print(
            f"[package] {summary['package_directory']} files={summary['selected_total']} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    summary_rows = [
        {key: value for key, value in summary.items() if key != "calibration"}
        for summary in summaries
    ]
    write_csv(package_root / "PACKAGES.csv", summary_rows)
    (package_root / "README.md").write_text(
        "# 噪音评分器十档包\n\n"
        "共 10 个评分轴，每包包含训练原样本、V4PH 自克隆、V4fg 自克隆、固定 54 条跨源克隆四部分。"
        "先读 `MODEL_INFO.md`，再从每包第 01 档听到第 10 档。\n",
        encoding="utf-8",
    )

    archives: list[dict[str, Any]] = []
    if not args.no_archives:
        for summary in summaries:
            package_dir = package_root / str(summary["package_directory"])
            archive = archive_root / f"{package_dir.name}.zip"
            result = create_archive(package_dir, archive)
            archives.append({"package": package_dir.name, **result})
            print(
                f"[archive] {archive.name} bytes={result['bytes']} "
                f"elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
        write_csv(archive_root / "archives_manifest.csv", archives)

    audit = {
        "schema": "noise-scorer-tier-packages.v1",
        "canonical_rows": len(all_rows),
        "axes": len(AXES),
        "packages": summaries,
        "archives": archives,
        "all_scores_sha256": sha256_file(root / "all_scores.csv"),
    }
    (root / "package_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[all complete] packages=10", flush=True)


if __name__ == "__main__":
    main()
