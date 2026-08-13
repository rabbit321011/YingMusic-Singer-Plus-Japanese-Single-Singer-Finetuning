#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib

from torch.utils.data import Dataset


FRAME_RATE = 44100 / 2048


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio_path(record, audio_root_overrides=None):
    source = pathlib.Path(record["Path"])
    if source.exists():
        return source
    pool = record.get("V4LPool")
    if audio_root_overrides and pool in audio_root_overrides:
        candidate = pathlib.Path(audio_root_overrides[pool]) / source.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(source)


def validate_record(record, max_duration_sec, tolerance_sec=0.05):
    duration = float(record["Duration"])
    if duration <= 0:
        raise ValueError(f"non-positive duration: {record['Path']}: {duration}")
    if duration > max_duration_sec + tolerance_sec:
        raise ValueError(
            f"manifest duration exceeds cap without cropping permission: "
            f"{record['Path']}: {duration:.6f}s > {max_duration_sec:.6f}s"
        )
    previous_start = -1.0
    for phrase_index, phrase in enumerate(record.get("Phrases", [])):
        start = float(phrase["start"])
        end = float(phrase["end"])
        if not 0 <= start < end <= duration + tolerance_sec:
            raise ValueError(
                f"phrase out of bounds: {record['Path']}:{phrase_index}: "
                f"{start:.6f}-{end:.6f}/{duration:.6f}"
            )
        if start < previous_start:
            raise ValueError(
                f"non-monotonic phrase: {record['Path']}:{phrase_index}: "
                f"{start:.6f} < {previous_start:.6f}"
            )
        if not phrase.get("tokens"):
            raise ValueError(f"empty phrase tokens: {record['Path']}:{phrase_index}")
        previous_start = start
    return duration


class V4LManifestDataset(Dataset):
    """Strict V4L loader: validates complete audio and never truncates waveform."""

    def __init__(
        self,
        manifest_path,
        expected_sha256,
        max_duration_sec=60.0,
        audio_root_overrides=None,
        duration_tolerance_sec=0.05,
    ):
        manifest_path = pathlib.Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        actual_sha256 = sha256_file(manifest_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"V4L manifest SHA256 mismatch: {actual_sha256} != {expected_sha256}"
            )
        self.records = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.max_duration_sec = float(max_duration_sec)
        self.audio_root_overrides = dict(audio_root_overrides or {})
        self.duration_tolerance_sec = float(duration_tolerance_sec)
        self.manifest_sha256 = actual_sha256
        self.pool_counts = {}
        for record in self.records:
            validate_record(record, self.max_duration_sec, self.duration_tolerance_sec)
            pool = record.get("V4LPool", "unknown")
            self.pool_counts[pool] = self.pool_counts.get(pool, 0) + 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torchaudio

        record = self.records[index]
        path = resolve_audio_path(record, self.audio_root_overrides)
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != 44100:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 44100)
            sample_rate = 44100
        actual_duration = waveform.shape[-1] / sample_rate
        manifest_duration = float(record["Duration"])
        if abs(actual_duration - manifest_duration) > self.duration_tolerance_sec:
            raise ValueError(
                f"audio/manifest duration mismatch: {path}: "
                f"{actual_duration:.6f}s != {manifest_duration:.6f}s"
            )
        if actual_duration > self.max_duration_sec + self.duration_tolerance_sec:
            raise ValueError(
                f"audio exceeds cap; cropping is forbidden: {path}: "
                f"{actual_duration:.6f}s > {self.max_duration_sec:.6f}s"
            )
        return {
            "wav": waveform,
            "sr": sample_rate,
            "phrases": record.get("Phrases", []),
            "full_tokens": record.get("full_tokens", []),
            "tier": record.get("V4dSourceTier", record.get("V4LPool", "unknown")),
            "duration": actual_duration,
            "pool": record.get("V4LPool", "unknown"),
            "path": str(path),
            "latent_frames": int(actual_duration * FRAME_RATE) + 1,
        }


def collate_v4l(batch):
    return {
        "wav": [item["wav"] for item in batch],
        "sr": [item["sr"] for item in batch],
        "phrases": [item["phrases"] for item in batch],
        "tier": [item["tier"] for item in batch],
        "duration": [item["duration"] for item in batch],
        "pool": [item["pool"] for item in batch],
        "path": [item["path"] for item in batch],
        "latent_frames": [item["latent_frames"] for item in batch],
    }
