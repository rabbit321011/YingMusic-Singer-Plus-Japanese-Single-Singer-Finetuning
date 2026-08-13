#!/usr/bin/env python3
"""Run one frozen objective scorer over the canonical noise-tier audio pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


SCORERS = (
    "singmos_v1",
    "singmos_pro",
    "nisqa",
    "dnsmos",
    "utmos22",
    "utmosv2",
    "wvmos",
    "highfreq_reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scorer", choices=SCORERS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path, limit: int | None) -> list[dict[str, str]]:
    with (root / "canonical_audio_manifest.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3054:
        raise RuntimeError(f"expected 3054 canonical rows, found {len(rows)}")
    return rows[:limit] if limit else rows


class ResultWriter:
    def __init__(
        self,
        path: Path,
        fieldnames: list[str],
        overwrite: bool,
    ) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and self.path.exists():
            self.path.unlink()
        self.completed: set[str] = set()
        if self.path.exists():
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                self.completed = {
                    row["item_id"] for row in csv.DictReader(handle) if row.get("item_id")
                }
        self.handle = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        if self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()
        self.completed.add(str(row["item_id"]))

    def close(self) -> None:
        self.handle.close()


def load_wave(path: str):
    import torch
    import torchaudio

    wave, sample_rate = torchaudio.load(path)
    wave = wave.mean(dim=0, keepdim=True)
    return wave.to(dtype=torch.float32), sample_rate


def run_sequential(
    rows: list[dict[str, str]],
    writer: ResultWriter,
    score: Callable[[dict[str, str]], dict[str, Any]],
) -> None:
    pending = [row for row in rows if row["item_id"] not in writer.completed]
    started = time.monotonic()
    total = len(pending)
    for index, row in enumerate(pending, start=1):
        result = {"item_id": row["item_id"], **score(row)}
        if not all(
            isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
            for key, value in result.items()
            if key != "item_id"
        ):
            raise ValueError(f"non-finite scorer output: {result}")
        writer.write(result)
        if index == 1 or index % 10 == 0 or index == total:
            elapsed = time.monotonic() - started
            eta = (total - index) * elapsed / max(index, 1)
            print(
                f"[progress] scorer_row={index}/{total} total_done={len(writer.completed)}/"
                f"{len(rows)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )


def load_singmos(root: Path, model_name: str, device: str):
    import torch

    weight_name = "singmos_v1.pt" if model_name == "singmos_v1" else "singmos_pro.pth"
    model = torch.hub.load(
        str(root / "third_party" / "SingMOS"),
        model_name,
        source="local",
        pretrained=False,
        model_path=str(root / "scorer_weights" / weight_name),
    )
    return model.to(device).eval()


def score_singmos(
    root: Path,
    rows: list[dict[str, str]],
    writer: ResultWriter,
    model_name: str,
    device: str,
    batch_size: int,
) -> None:
    import torch
    import torchaudio

    model = load_singmos(root, model_name, device)

    def score(row: dict[str, str]) -> dict[str, float]:
        wave, sample_rate = load_wave(row["normalized_path"])
        if sample_rate != 16000:
            wave = torchaudio.functional.resample(wave, sample_rate, 16000)
        wave = wave.to(device)
        lengths = torch.tensor([wave.shape[1]], dtype=torch.long, device=device)
        with torch.inference_mode():
            value = model(wave, lengths)
        return {"score": float(value.item())}

    run_sequential(rows, writer, score)


def score_dnsmos(
    rows: list[dict[str, str]], writer: ResultWriter, workers: int
) -> None:
    from speechmos.dnsmos import DNSMOS
    import speechmos.dnsmos as module

    package_dir = Path(module.__file__).parent
    model = DNSMOS(
        str(package_dir / "dnsmos_models" / "sig_bak_ovr.onnx"),
        str(package_dir / "dnsmos_models" / "model_v8.onnx"),
    )
    pending = [row for row in rows if row["item_id"] not in writer.completed]

    def evaluate(row: dict[str, str]) -> tuple[str, dict[str, float]]:
        result = model(row["normalized_path"], 16000, False)
        return row["item_id"], {
            "p808_mos": float(result["p808_mos"]),
            "sig_mos": float(result["sig_mos"]),
            "bak_mos": float(result["bak_mos"]),
            "ovrl_mos": float(result["ovrl_mos"]),
        }

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (item_id, values) in enumerate(executor.map(evaluate, pending), start=1):
            writer.write({"item_id": item_id, **values})
            if index == 1 or index % 25 == 0 or index == len(pending):
                elapsed = time.monotonic() - started
                eta = (len(pending) - index) * elapsed / max(index, 1)
                print(
                    f"[progress] scorer_row={index}/{len(pending)} "
                    f"total_done={len(writer.completed)}/{len(rows)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )


def score_utmos22(
    root: Path,
    rows: list[dict[str, str]],
    writer: ResultWriter,
    device: str,
) -> None:
    import torch

    model = torch.hub.load(
        str(root / "third_party" / "SpeechMOS"),
        "utmos22_strong",
        source="local",
        pretrained=False,
    )
    state = torch.load(
        root / "scorer_weights" / "utmos22_strong_step7459_v1.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state)
    model = model.to(device).eval()

    def score(row: dict[str, str]) -> dict[str, float]:
        wave, sample_rate = load_wave(row["normalized_path"])
        with torch.inference_mode():
            value = model(wave.to(device), sample_rate)
        return {"score": float(value.item())}

    run_sequential(rows, writer, score)


def make_flat_pool(root: Path, rows: Iterable[dict[str, str]], name: str) -> Path:
    flat = root / "scorer_work" / name
    if flat.exists():
        shutil.rmtree(flat)
    flat.mkdir(parents=True)
    for row in rows:
        os.symlink(row["normalized_path"], flat / f"{row['item_id']}.wav")
    return flat


def score_utmosv2(
    root: Path,
    rows: list[dict[str, str]],
    writer: ResultWriter,
    device: str,
    workers: int,
    batch_size: int,
) -> None:
    import utmosv2

    pending = [row for row in rows if row["item_id"] not in writer.completed]
    if not pending:
        return
    flat = make_flat_pool(root, pending, "utmosv2")
    model = utmosv2.create_model(pretrained=True, device=device)
    results = model.predict(
        input_dir=flat,
        device=device,
        num_workers=workers,
        batch_size=batch_size,
        num_repetitions=1,
        remove_silent_section=True,
        verbose=True,
    )
    for result in results:
        item_id = Path(str(result["file_path"])).stem
        writer.write({"item_id": item_id, "score": float(result["predicted_mos"])})


def score_wvmos(
    rows: list[dict[str, str]], writer: ResultWriter, device: str, batch_size: int
) -> None:
    from wvmos import get_wvmos

    model = get_wvmos(cuda=device.startswith("cuda"))

    def score(row: dict[str, str]) -> dict[str, float]:
        return {"score": float(model.calculate_one(row["normalized_path"]))}

    run_sequential(rows, writer, score)


def highfreq_metrics(path: str) -> dict[str, float]:
    import librosa

    wave, sample_rate = librosa.load(path, sr=48000, mono=True, res_type="soxr_hq")
    spectrum = np.abs(
        librosa.stft(wave, n_fft=2048, hop_length=512, win_length=2048)
    ) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    frame_energy = spectrum.sum(axis=0)
    threshold = max(float(frame_energy.max()) * 1e-4, 1e-20)
    active = frame_energy >= threshold
    if not active.any():
        active[:] = True
    spectrum = spectrum[:, active]
    total = spectrum[(frequencies >= 80) & (frequencies <= 20000)].sum(axis=0)

    result: dict[str, float] = {}
    for low, high, label in ((8000, 20000, "8_20k"), (12000, 20000, "12_20k")):
        band = spectrum[(frequencies >= low) & (frequencies <= high)]
        energy_ratio = band.sum(axis=0) / np.maximum(total, 1e-20)
        flatness = np.exp(np.mean(np.log(np.maximum(band, 1e-20)), axis=0)) / np.maximum(
            np.mean(band, axis=0), 1e-20
        )
        result[f"energy_ratio_{label}_median"] = float(np.median(energy_ratio))
        result[f"spectral_flatness_{label}_median"] = float(np.median(flatness))
    result["active_frame_fraction"] = float(np.mean(active))
    return result


def score_highfreq(
    rows: list[dict[str, str]], writer: ResultWriter, workers: int
) -> None:
    pending = [row for row in rows if row["item_id"] not in writer.completed]

    def evaluate(row: dict[str, str]) -> tuple[str, dict[str, float]]:
        return row["item_id"], highfreq_metrics(row["normalized_path"])

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (item_id, values) in enumerate(executor.map(evaluate, pending), start=1):
            writer.write({"item_id": item_id, **values})
            if index == 1 or index % 25 == 0 or index == len(pending):
                elapsed = time.monotonic() - started
                eta = (len(pending) - index) * elapsed / max(index, 1)
                print(
                    f"[progress] scorer_row={index}/{len(pending)} "
                    f"total_done={len(writer.completed)}/{len(rows)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )


def score_nisqa(
    root: Path,
    rows: list[dict[str, str]],
    output_path: Path,
    workers: int,
    batch_size: int,
) -> None:
    import pandas as pd
    import torch

    nisqa_root = root / "third_party" / "NISQA"
    sys.path.insert(0, str(nisqa_root))
    from nisqa.NISQA_model import nisqaModel

    input_path = root / "scores" / "nisqa_input.csv"
    pd.DataFrame(
        {
            "item_id": [row["item_id"] for row in rows],
            "filepath_deg": [row["normalized_path"] for row in rows],
        }
    ).to_csv(input_path, index=False)
    output_dir = root / "scores" / "nisqa_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    args = {
        "mode": "predict_csv",
        "pretrained_model": str(nisqa_root / "weights" / "nisqa.tar"),
        "deg": None,
        "data_dir": "",
        "output_dir": str(output_dir),
        "csv_file": str(input_path),
        "csv_deg": "filepath_deg",
        "num_workers": workers,
        "bs": batch_size,
        "ms_channel": None,
        "tr_bs_val": batch_size,
        "tr_num_workers": workers,
        "tr_device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    model = nisqaModel(args)
    result = model.predict()
    columns = [
        "item_id",
        "mos_pred",
        "noi_pred",
        "dis_pred",
        "col_pred",
        "loud_pred",
    ]
    result[columns].to_csv(output_path, index=False)


def provenance(root: Path, scorer: str, output_path: Path) -> dict[str, Any]:
    import importlib.metadata as metadata

    packages = {}
    for package in (
        "torch",
        "torchaudio",
        "librosa",
        "numpy",
        "onnxruntime",
        "s3prl",
        "speechmos",
        "utmosv2",
        "wvmos",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return {
        "schema": "noise-scorer-run.v1",
        "scorer": scorer,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "packages": packages,
        "canonical_manifest_sha256": sha256_file(root / "canonical_audio_manifest.csv"),
        "result_sha256": sha256_file(output_path),
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    root = args.root.resolve()
    rows = load_manifest(root, args.limit)
    output_path = root / "scores" / f"{args.scorer}.csv"

    if args.scorer == "nisqa":
        if args.overwrite and output_path.exists():
            output_path.unlink()
        score_nisqa(root, rows, output_path, args.workers, args.batch_size)
    else:
        fields = {
            "dnsmos": ["item_id", "p808_mos", "sig_mos", "bak_mos", "ovrl_mos"],
            "highfreq_reference": [
                "item_id",
                "energy_ratio_8_20k_median",
                "spectral_flatness_8_20k_median",
                "energy_ratio_12_20k_median",
                "spectral_flatness_12_20k_median",
                "active_frame_fraction",
            ],
        }.get(args.scorer, ["item_id", "score"])
        writer = ResultWriter(output_path, fields, args.overwrite)
        try:
            if args.scorer in ("singmos_v1", "singmos_pro"):
                score_singmos(
                    root, rows, writer, args.scorer, args.device, args.batch_size
                )
            elif args.scorer == "dnsmos":
                score_dnsmos(rows, writer, args.workers)
            elif args.scorer == "utmos22":
                score_utmos22(root, rows, writer, args.device)
            elif args.scorer == "utmosv2":
                score_utmosv2(
                    root,
                    rows,
                    writer,
                    args.device,
                    args.workers,
                    args.batch_size,
                )
            elif args.scorer == "wvmos":
                score_wvmos(rows, writer, args.device, args.batch_size)
            elif args.scorer == "highfreq_reference":
                score_highfreq(rows, writer, args.workers)
            else:
                raise AssertionError(args.scorer)
        finally:
            writer.close()

    with (root / "scores" / f"{args.scorer}.provenance.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(provenance(root, args.scorer, output_path), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[complete] scorer={args.scorer} rows={len(rows)} output={output_path}")


if __name__ == "__main__":
    main()
