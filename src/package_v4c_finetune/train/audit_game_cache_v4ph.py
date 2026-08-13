import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
YING_REPO = PROJECT / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src").is_dir():
    YING_REPO = PROJECT
sys.path.insert(0, str(YING_REPO))

from src.YingMusicSinger.melody.game_cache_v4ph import (  # noqa: E402
    GAME_CACHE_SCHEMA,
    validate_game_cache_arrays,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected_entries", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_schema") != GAME_CACHE_SCHEMA:
        raise ValueError("GAME cache schema mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("GAME cache entries must be a list")
    if args.expected_entries is not None and len(entries) != args.expected_entries:
        raise ValueError(f"Entry count {len(entries)} != {args.expected_entries}")

    support = np.zeros(257, dtype=np.int64)
    total_notes = total_voiced = total_rest = total_frames = total_bytes = 0
    sources = set()
    cache_paths = set()
    closure_errors = []
    for index, entry in enumerate(entries, start=1):
        source = entry["source_path"]
        if source in sources:
            raise ValueError(f"Duplicate source path: {source}")
        sources.add(source)
        cache_path = (root / entry["cache"]).resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        cache_paths.add(str(cache_path))
        total_bytes += cache_path.stat().st_size
        with np.load(cache_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        validate_game_cache_arrays(arrays)
        valid = arrays["valid"].astype(bool)
        classes = arrays["classes"][valid].astype(np.int64)
        counts = np.bincount(classes, minlength=257)
        support += counts
        note_count = int(valid.sum())
        voiced_count = int((classes < 255).sum())
        rest_count = int((classes == 255).sum())
        frames = int(arrays["duration_frames_100hz"][valid].sum())
        if (
            note_count != entry["notes"]
            or voiced_count != entry["voiced_notes"]
            or rest_count != entry["rest_notes"]
            or frames != entry["native_frames_100hz"]
        ):
            raise ValueError(f"Manifest/cache statistics differ at entry {index}")
        closure_error = abs(frames / 100 - float(entry["duration"]))
        if closure_error > 0.011:
            raise ValueError(f"Duration closure exceeds 11ms at entry {index}")
        closure_errors.append(closure_error)
        total_notes += note_count
        total_voiced += voiced_count
        total_rest += rest_count
        total_frames += frames
        if index % 1000 == 0:
            print(f"[audit {index}/{len(entries)}]", flush=True)

    report = {
        "schema": "v4ph_game_cache_audit_v1",
        "manifest": str(manifest_path),
        "entries": len(entries),
        "unique_cache_files": len(cache_paths),
        "total_bytes": total_bytes,
        "total_notes": total_notes,
        "voiced_notes": total_voiced,
        "rest_notes": total_rest,
        "rest_ratio": total_rest / max(total_notes, 1),
        "total_frames_100hz": total_frames,
        "max_duration_closure_error_seconds": max(closure_errors, default=0),
        "supported_pitch_rows": int((support[:255] > 0).sum()),
        "unsupported_pitch_rows": np.nonzero(support[:255] == 0)[0].tolist(),
        "pitch_support_counts": support.tolist(),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "pitch_support_counts"}, indent=2))


if __name__ == "__main__":
    main()
