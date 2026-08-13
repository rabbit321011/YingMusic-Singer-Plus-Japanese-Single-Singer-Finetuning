import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from build_v4ijph_ins_cache import retry_indices
from v4ijph_contract import V4IJPH_INS_CACHE_SCHEMA
from v4ijph_ins_cache import (
    INTRINSIC_CHECKPOINT_NAME,
    INTRINSIC_CHECKPOINT_SHA256,
    INTRINSIC_MODEL_REVISION,
    INTRINSIC_REPOSITORY,
    MINIMUM_INPUT_SAMPLES,
    MODEL_SAMPLE_RATE,
    PARASPEECHCLAP_SOURCE_REVISION,
    TRAINING_SAMPLE_RATE,
    InsCache,
    cache_config_fingerprint,
    sha256_file,
    sha256_waveform,
    verify_inventory,
)


class InsCacheTest(unittest.TestCase):
    def write_cache(self, root: Path, *, zero_embedding=False):
        embeddings = np.zeros((2, 768), dtype=np.float16)
        if not zero_embedding:
            embeddings[:, 0] = 1
        np.save(root / "embeddings.f16.npy", embeddings)
        np.save(root / "status.u8.npy", np.ones(2, dtype=np.uint8))
        hashes = np.array([b"a" * 64, b"b" * 64], dtype="S64")
        samples = np.array([100, 200], dtype=np.int64)
        np.save(root / "waveform_sha256.s64.npy", hashes)
        np.save(root / "waveform_samples.i64.npy", samples)
        records = [
            {
                "cache_index": 0,
                "sample_id": "a",
                "source_path": "/a.wav",
                "resolved_path": "/a.wav",
                "waveform_sha256": "a" * 64,
                "waveform_samples": 100,
            },
            {
                "cache_index": 1,
                "sample_id": "b",
                "source_path": "/b.wav",
                "resolved_path": "/b.wav",
                "waveform_sha256": "b" * 64,
                "waveform_samples": 200,
            },
        ]
        with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        config = {
            "schema": V4IJPH_INS_CACHE_SCHEMA,
            "manifest_sha256": "manifest-sha",
            "max_duration_sec": 30.0,
            "sample_rate_before_encoder": TRAINING_SAMPLE_RATE,
            "channel_policy": "require_mono",
            "truncation_policy": "prefix_after_resample_to_44100",
            "waveform_hash": "sha256_float32_44100_after_resample_and_prefix_crop",
            "model_repository": INTRINSIC_REPOSITORY,
            "model_revision": INTRINSIC_MODEL_REVISION,
            "source_revision": PARASPEECHCLAP_SOURCE_REVISION,
            "checkpoint_name": INTRINSIC_CHECKPOINT_NAME,
            "checkpoint_sha256": INTRINSIC_CHECKPOINT_SHA256,
            "model_sample_rate": MODEL_SAMPLE_RATE,
            "minimum_input_samples": MINIMUM_INPUT_SAMPLES,
            "embedding_dim": 768,
            "encoder_call": "get_audio_embedding(normalize=True)",
            "cache_dtype": "float16",
            "training_load_dtype": "float32",
        }
        names = (
            "records.jsonl",
            "embeddings.f16.npy",
            "status.u8.npy",
            "waveform_sha256.s64.npy",
            "waveform_samples.i64.npy",
        )
        metadata = {
            "schema": V4IJPH_INS_CACHE_SCHEMA,
            "state": "complete",
            "config": config,
            "config_fingerprint": cache_config_fingerprint(config),
            "files": {
                name: {
                    "sha256": sha256_file(root / name),
                    "bytes": (root / name).stat().st_size,
                }
                for name in names
            },
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def test_loads_canonical_cache_and_binds_manifest_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_cache(root)
            cache = InsCache(
                root,
                expected_manifest_sha256="manifest-sha",
                expected_max_duration_sec=30.0,
            )
            cache.assert_manifest(
                [
                    {"SampleId": "a", "Path": "/a.wav"},
                    {"SampleId": "b", "Path": "/b.wav"},
                ]
            )
            value = cache.vector("b")
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(tuple(value.shape), (768,))
            self.assertEqual(float(value[0]), 1.0)
            cache.close()

    def test_rejects_manifest_mapping_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_cache(root)
            cache = InsCache(
                root,
                expected_manifest_sha256="manifest-sha",
                expected_max_duration_sec=30.0,
            )
            try:
                with self.assertRaisesRegex(ValueError, "source-path mismatch"):
                    cache.assert_manifest(
                        [
                            {"SampleId": "a", "Path": "/wrong.wav"},
                            {"SampleId": "b", "Path": "/b.wav"},
                        ]
                    )
            finally:
                cache.close()

    def test_rejects_zero_or_unnormalized_embeddings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_cache(root, zero_embedding=True)
            with self.assertRaisesRegex(ValueError, "invalid normalized"):
                InsCache(
                    root,
                    expected_manifest_sha256="manifest-sha",
                    expected_max_duration_sec=30.0,
                )

    def test_failed_rows_are_retried(self):
        status = np.array([1, 2, 0, 255], dtype=np.uint8)
        self.assertEqual(retry_indices(status), [1, 2, 3])

    def test_waveform_hash_binds_shape_rate_and_samples(self):
        waveform = torch.arange(8).float()
        self.assertEqual(sha256_waveform(waveform), sha256_waveform(waveform.clone()))
        self.assertNotEqual(
            sha256_waveform(waveform), sha256_waveform(waveform + 1)
        )

    def test_inventory_verifier_checks_each_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            target = model / "value.bin"
            target.write_bytes(b"abc")
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "model_dir": str(model.resolve()),
                        "file_count": 1,
                        "total_bytes": 3,
                        "inventory_sha256": "inventory-id",
                        "files": [
                            {
                                "relative_path": "value.bin",
                                "bytes": 3,
                                "sha256": sha256_file(target),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = verify_inventory(inventory, model, verify_files=True)
            self.assertEqual(result["file_count"], 1)
            target.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_inventory(inventory, model, verify_files=True)


if __name__ == "__main__":
    unittest.main()
