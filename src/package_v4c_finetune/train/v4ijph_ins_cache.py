"""Audited ParaSpeechCLAP Intrinsic cache contract for V4IjPH v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from v4ijph_contract import V4IJPH_INS_CACHE_SCHEMA, V4IJPH_INS_DIM


INTRINSIC_REPOSITORY = "ajd12342/paraspeechclap-intrinsic"
INTRINSIC_MODEL_REVISION = "e80ad8eb199f7573f07f617c15a7497504151838"
PARASPEECHCLAP_SOURCE_REVISION = "4767eea9b05bb22598541d3c6ec42f5c6249d2fc"
INTRINSIC_CHECKPOINT_NAME = "slap-intrinsic.pth.tar"
INTRINSIC_CHECKPOINT_SHA256 = (
    "22b37acc1bb27a26034614b6eab1562131a4197b86fd0221e45350f927663ad2"
)
MODEL_SAMPLE_RATE = 16_000
TRAINING_SAMPLE_RATE = 44_100
MINIMUM_INPUT_SAMPLES = 400
NORM_MIN = 0.99
NORM_MAX = 1.01


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_waveform(waveform: torch.Tensor) -> str:
    value = waveform.detach().cpu().float().contiguous()
    digest = hashlib.sha256()
    digest.update(f"sr={TRAINING_SAMPLE_RATE};shape={tuple(value.shape)};".encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def load_h_manifest(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("H manifest must be a non-empty JSON list")
    sample_ids: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {index} is not an object")
        sample_id = row.get("SampleId")
        source_path = row.get("Path")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Manifest row {index} lacks SampleId")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(f"Manifest row {index} lacks Path")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate SampleId in manifest: {sample_id}")
        sample_ids.add(sample_id)
    return value


def resolve_audio_path(source_path: str, audio_root_override: str | None) -> Path:
    path = Path(source_path)
    if path.is_file():
        return path.resolve()
    if audio_root_override:
        candidate = Path(audio_root_override).resolve() / path.name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def load_training_waveform(
    source_path: str,
    *,
    max_duration_sec: float,
    audio_root_override: str | None,
) -> tuple[torch.Tensor, Path]:
    """Reproduce the exact waveform passed to the frozen training VAE."""
    import torchaudio

    audio_path = resolve_audio_path(source_path, audio_root_override)
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(
            f"V4IjPH requires mono training audio, got {tuple(waveform.shape)} "
            f"at {audio_path}"
        )
    if sample_rate != TRAINING_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TRAINING_SAMPLE_RATE
        )
    max_samples = int(float(max_duration_sec) * TRAINING_SAMPLE_RATE)
    waveform = waveform[:, :max_samples].contiguous().float()
    if waveform.shape[-1] == 0:
        raise ValueError(f"Empty waveform after preprocessing: {audio_path}")
    return waveform.squeeze(0), audio_path


def load_reference_waveform(path: str | Path) -> tuple[torch.Tensor, dict]:
    """Load a complete user-selected R; no duration crop is permitted."""
    import torchaudio

    source = Path(path).resolve()
    waveform, original_rate = torchaudio.load(source)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(
            f"V4IjPH style reference must be mono, got {tuple(waveform.shape)}"
        )
    if original_rate != TRAINING_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, original_rate, TRAINING_SAMPLE_RATE
        )
    waveform = waveform.squeeze(0).contiguous().float()
    if waveform.numel() == 0:
        raise ValueError("V4IjPH style reference is empty")
    provenance = {
        "source_path": str(source),
        "source_file_sha256": sha256_file(source),
        "original_sample_rate": int(original_rate),
        "encoder_input_sample_rate": TRAINING_SAMPLE_RATE,
        "encoder_input_samples": int(waveform.numel()),
        "encoder_input_duration_sec": waveform.numel() / TRAINING_SAMPLE_RATE,
        "encoder_input_waveform_sha256": sha256_waveform(waveform),
        "crop_policy": "none_complete_reference",
        "channel_policy": "require_mono",
    }
    return waveform, provenance


def verify_inventory(
    inventory_path: str | Path,
    model_dir: str | Path,
    *,
    verify_files: bool,
) -> dict:
    inventory_path = Path(inventory_path).resolve()
    model_dir = Path(model_dir).resolve()
    value = json.loads(inventory_path.read_text(encoding="utf-8"))
    if Path(value.get("model_dir", "")).resolve() != model_dir:
        raise ValueError(f"Inventory model_dir mismatch: {inventory_path}")
    files = value.get("files")
    if not isinstance(files, list) or int(value.get("file_count", -1)) != len(files):
        raise ValueError(f"Inventory file list is invalid: {inventory_path}")
    if verify_files:
        total_bytes = 0
        for record in files:
            target = model_dir / record["relative_path"]
            if not target.is_file():
                raise FileNotFoundError(target)
            if target.stat().st_size != int(record["bytes"]):
                raise ValueError(f"Inventory byte mismatch: {target}")
            if sha256_file(target) != record["sha256"]:
                raise ValueError(f"Inventory SHA256 mismatch: {target}")
            total_bytes += target.stat().st_size
        if total_bytes != int(value.get("total_bytes", -1)):
            raise ValueError(f"Inventory total byte mismatch: {inventory_path}")
    return {
        "path": str(inventory_path),
        "file_sha256": sha256_file(inventory_path),
        "inventory_sha256": value.get("inventory_sha256"),
        "file_count": int(value["file_count"]),
        "total_bytes": int(value["total_bytes"]),
        "model_dir": str(model_dir),
    }


def cache_config_fingerprint(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def validate_frozen_config(config: dict) -> None:
    expected = {
        "schema": V4IJPH_INS_CACHE_SCHEMA,
        "sample_rate_before_encoder": TRAINING_SAMPLE_RATE,
        "channel_policy": "require_mono",
        "truncation_policy": "prefix_after_resample_to_44100",
        "model_repository": INTRINSIC_REPOSITORY,
        "model_revision": INTRINSIC_MODEL_REVISION,
        "source_revision": PARASPEECHCLAP_SOURCE_REVISION,
        "checkpoint_name": INTRINSIC_CHECKPOINT_NAME,
        "checkpoint_sha256": INTRINSIC_CHECKPOINT_SHA256,
        "model_sample_rate": MODEL_SAMPLE_RATE,
        "minimum_input_samples": MINIMUM_INPUT_SAMPLES,
        "embedding_dim": V4IJPH_INS_DIM,
        "encoder_call": "get_audio_embedding(normalize=True)",
        "cache_dtype": "float16",
        "training_load_dtype": "float32",
        "waveform_hash": "sha256_float32_44100_after_resample_and_prefix_crop",
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(
                f"INS cache config {key} mismatch: "
                f"{config.get(key)!r} != {expected_value!r}"
            )


class ParaSpeechClapIntrinsic:
    """Frozen, strictly loaded encoder used only by cache and inference entrypoints."""

    def __init__(
        self,
        *,
        source_dir: Path,
        speech_model_dir: Path,
        text_model_dir: Path,
        checkpoint: Path,
        device: str,
    ) -> None:
        import sys

        self.device = torch.device(device)
        if checkpoint.name != INTRINSIC_CHECKPOINT_NAME:
            raise ValueError("Unexpected ParaSpeechCLAP checkpoint filename")
        if sha256_file(checkpoint) != INTRINSIC_CHECKPOINT_SHA256:
            raise ValueError("ParaSpeechCLAP Intrinsic checkpoint SHA256 mismatch")
        source = os.fspath(source_dir.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        from paraspeechclap.model import CLAP

        model = CLAP(
            speech_name=os.fspath(speech_model_dir.resolve()),
            text_name=os.fspath(text_model_dir.resolve()),
            embedding_dim=V4IJPH_INS_DIM,
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        result = model.load_state_dict(state_dict, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Strict INS load mismatch: {result}")
        del state_dict
        self.model = model.eval().to(self.device)

        import torchaudio

        self.resampler = torchaudio.transforms.Resample(
            TRAINING_SAMPLE_RATE, MODEL_SAMPLE_RATE
        )

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> np.ndarray:
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono waveform [S], got {tuple(waveform.shape)}")
        audio = self.resampler(waveform.cpu()).float()
        if audio.numel() < MINIMUM_INPUT_SAMPLES:
            audio = torch.nn.functional.pad(
                audio, (0, MINIMUM_INPUT_SAMPLES - audio.numel())
            )
        vector = self.model.get_audio_embedding(
            audio.unsqueeze(0).to(self.device), normalize=True
        )
        value = vector.detach().float().cpu().numpy()
        if value.shape != (1, V4IJPH_INS_DIM):
            raise ValueError(f"Unexpected INS shape {value.shape}")
        row = value[0]
        norm = float(np.linalg.norm(row))
        if not np.isfinite(row).all() or not NORM_MIN <= norm <= NORM_MAX:
            raise ValueError(f"Invalid INS output: finite={np.isfinite(row).all()} norm={norm}")
        return row


class InsCache:
    """Read-only v2 cache with canonical sample-to-waveform provenance."""

    ARRAY_FILES = {
        "embeddings.f16.npy": "embeddings",
        "status.u8.npy": "status",
        "waveform_sha256.s64.npy": "waveform_hashes",
        "waveform_samples.i64.npy": "waveform_samples",
    }

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        expected_max_duration_sec: float,
    ) -> None:
        self.root = Path(cache_dir).resolve()
        metadata_path = self.root / "metadata.json"
        records_path = self.root / "records.jsonl"
        required = [metadata_path, records_path]
        required.extend(self.root / name for name in self.ARRAY_FILES)
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != V4IJPH_INS_CACHE_SCHEMA:
            raise ValueError("INS cache schema mismatch")
        if self.metadata.get("state") != "complete":
            raise ValueError("INS cache is not complete")
        config = self.metadata.get("config")
        if not isinstance(config, dict):
            raise ValueError("INS cache metadata lacks config")
        validate_frozen_config(config)
        if config.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError("INS cache manifest SHA256 mismatch")
        if float(config.get("max_duration_sec", -1)) != float(
            expected_max_duration_sec
        ):
            raise ValueError("INS cache max-duration mismatch")
        if self.metadata.get("config_fingerprint") != cache_config_fingerprint(config):
            raise ValueError("INS cache config fingerprint mismatch")

        files = self.metadata.get("files") or {}
        for name in ("records.jsonl", *self.ARRAY_FILES):
            path = self.root / name
            record = files.get(name) or {}
            if record.get("sha256") != sha256_file(path):
                raise ValueError(f"INS cache file SHA256 mismatch: {name}")
            if int(record.get("bytes", -1)) != path.stat().st_size:
                raise ValueError(f"INS cache file byte mismatch: {name}")

        self.embeddings = np.load(self.root / "embeddings.f16.npy", mmap_mode="r")
        self.status = np.load(self.root / "status.u8.npy", mmap_mode="r")
        self.waveform_hashes = np.load(
            self.root / "waveform_sha256.s64.npy", mmap_mode="r"
        )
        self.waveform_samples = np.load(
            self.root / "waveform_samples.i64.npy", mmap_mode="r"
        )
        rows = int(self.embeddings.shape[0])
        if self.embeddings.shape != (rows, V4IJPH_INS_DIM):
            raise ValueError("INS embedding array shape mismatch")
        if self.status.shape != (rows,) or not np.all(self.status == 1):
            raise ValueError("INS cache contains incomplete rows")
        if self.waveform_hashes.shape != (rows,) or self.waveform_samples.shape != (
            rows,
        ):
            raise ValueError("INS waveform provenance arrays have invalid shapes")
        norms = np.linalg.norm(np.asarray(self.embeddings, dtype=np.float32), axis=1)
        if not np.isfinite(norms).all() or norms.min() < NORM_MIN or norms.max() > NORM_MAX:
            raise ValueError("INS cache contains invalid normalized embeddings")

        self.records: list[dict] = []
        self.by_sample_id: dict[str, int] = {}
        with records_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                row = json.loads(line)
                sample_id = row.get("sample_id")
                if row.get("cache_index") != line_number:
                    raise ValueError("INS records are not canonically indexed")
                if not isinstance(sample_id, str) or sample_id in self.by_sample_id:
                    raise ValueError("INS records contain an invalid/duplicate SampleId")
                expected_hash = bytes(self.waveform_hashes[line_number]).decode("ascii")
                if row.get("waveform_sha256") != expected_hash:
                    raise ValueError("INS record waveform SHA differs from array")
                if int(row.get("waveform_samples", -1)) != int(
                    self.waveform_samples[line_number]
                ):
                    raise ValueError("INS record waveform sample count differs from array")
                self.by_sample_id[sample_id] = line_number
                self.records.append(row)
        if len(self.records) != rows:
            raise ValueError("INS record count differs from embedding rows")

    def assert_manifest(self, manifest: list[dict]) -> None:
        if len(manifest) != len(self.records):
            raise ValueError("INS cache and manifest row counts differ")
        for index, (source, cached) in enumerate(zip(manifest, self.records)):
            if source["SampleId"] != cached["sample_id"]:
                raise ValueError(f"INS SampleId/order mismatch at row {index}")
            if source["Path"] != cached["source_path"]:
                raise ValueError(f"INS source-path mismatch at row {index}")

    def vector(self, sample_id: str) -> torch.Tensor:
        try:
            index = self.by_sample_id[sample_id]
        except KeyError as error:
            raise KeyError(f"No INS cache entry for SampleId {sample_id}") from error
        value = np.asarray(self.embeddings[index], dtype=np.float32).copy()
        return torch.from_numpy(value)

    def provenance(self) -> dict:
        return {
            "schema": V4IJPH_INS_CACHE_SCHEMA,
            "metadata_sha256": sha256_file(self.root / "metadata.json"),
            "config_fingerprint": self.metadata["config_fingerprint"],
            "rows": len(self.records),
            "files": {
                name: self.metadata["files"][name]["sha256"]
                for name in ("records.jsonl", *self.ARRAY_FILES)
            },
        }

    def close(self) -> None:
        for name in ("embeddings", "status", "waveform_hashes", "waveform_samples"):
            value = getattr(self, name, None)
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()
