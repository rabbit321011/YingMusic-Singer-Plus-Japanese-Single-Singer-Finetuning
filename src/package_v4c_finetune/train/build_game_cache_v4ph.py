import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch


PROJECT = Path(__file__).resolve().parents[2]
YING_REPO = PROJECT / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src").is_dir():
    YING_REPO = PROJECT
sys.path.insert(0, str(YING_REPO))

from src.YingMusicSinger.melody.game_cache_v4ph import (  # noqa: E402
    GAME_CACHE_SCHEMA,
    canonicalize_game_cache,
    load_game_cache,
    save_game_cache,
)
from src.YingMusicSinger.melody.game_runtime_v4ph import (  # noqa: E402
    extract_game_notes_with_posterior,
    stable_game_seed,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio_path(source_path, audio_root):
    source = Path(source_path)
    if source.exists():
        return source.resolve()
    candidate = Path(audio_root) / source.name
    if not candidate.exists():
        raise FileNotFoundError(f"Cannot resolve audio path: {source_path}")
    return candidate.resolve()


def read_records(token_paths):
    records = []
    seen = set()
    for token_path in token_paths:
        payload = json.loads(Path(token_path).read_text(encoding="utf-8"))
        for record in payload:
            source_path = record["Path"]
            if source_path in seen:
                continue
            seen.add(source_path)
            records.append(
                {
                    "source_path": source_path,
                    "language": record.get("Language", "ja"),
                    "declared_duration": float(record["Duration"]),
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_repo", required=True)
    parser.add_argument("--game_model", required=True)
    parser.add_argument("--token_json", action="append", required=True)
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base_seed", type=int, default=20260730)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume_partial", action="store_true")
    args = parser.parse_args()

    game_repo = str(Path(args.game_repo).resolve())
    if game_repo not in sys.path:
        sys.path.insert(0, game_repo)
    from inference.api import load_inference_model

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.resume_partial:
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if output_dir.exists() and (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed cache already has a manifest: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "samples"
    cache_dir.mkdir(exist_ok=True)

    game_model_path = Path(args.game_model).resolve()
    game, language_map = load_inference_model(game_model_path)
    device = torch.device(args.device)
    game = game.to(device).eval()
    records = read_records(args.token_json)
    if args.limit is not None:
        records = records[: args.limit]

    entries = []
    start_time = time.perf_counter()
    for index, record in enumerate(records, start=1):
        audio_path = resolve_audio_path(record["source_path"], args.audio_root)
        audio_sha = sha256_file(audio_path)
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if sample_rate != GAME_CACHE_SCHEMA["sample_rate"] or audio.ndim != 1:
            raise ValueError(
                f"Unexpected audio format for {audio_path}: sr={sample_rate}, ndim={audio.ndim}"
            )
        duration = audio.size / sample_rate
        if abs(duration - record["declared_duration"]) > 0.02:
            raise ValueError(
                f"Duration mismatch for {audio_path}: {duration} vs "
                f"{record['declared_duration']}"
            )
        language = record["language"]
        language_id = language_map.get(language, 0) if language_map else 0
        seed = stable_game_seed(audio_sha, args.base_seed)
        cache_name = f"{audio_sha}.npz"
        cache_path = cache_dir / cache_name
        reused = cache_path.exists()
        if reused:
            restored = load_game_cache(cache_path)
            arrays = {name: value.numpy() for name, value in restored.items()}
        else:
            notes = extract_game_notes_with_posterior(
                model=game,
                waveform=torch.from_numpy(audio),
                duration=duration,
                language_id=language_id,
                nsteps=GAME_CACHE_SCHEMA["nsteps"],
                seed=seed,
                device=device,
            )
            arrays = canonicalize_game_cache(notes)
            save_game_cache(cache_path, arrays)
        valid = arrays["valid"]
        entry = {
            **record,
            "audio_path_local": str(audio_path),
            "audio_sha256": audio_sha,
            "num_samples": int(audio.size),
            "duration": duration,
            "language_id": language_id,
            "seed": seed,
            "cache": f"samples/{cache_name}",
            "notes": int(valid.sum()),
            "voiced_notes": int((valid & arrays["presence"]).sum()),
            "rest_notes": int((valid & ~arrays["presence"]).sum()),
            "native_frames_100hz": int(arrays["duration_frames_100hz"][valid].sum()),
            "content_cache_reused": reused,
        }
        entries.append(entry)
        elapsed = time.perf_counter() - start_time
        rate = elapsed / index
        eta = rate * (len(records) - index)
        print(
            f"[{index}/{len(records)}] audio={audio_sha[:12]} notes={entry['notes']} "
            f"reused={int(reused)} "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )

    manifest = {
        "cache_schema": GAME_CACHE_SCHEMA,
        "game_model": str(game_model_path),
        "game_model_sha256": sha256_file(game_model_path),
        "base_seed": args.base_seed,
        "token_json": [str(Path(path).resolve()) for path in args.token_json],
        "audio_root": str(Path(args.audio_root).resolve()),
        "entries": entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"GAME cache complete: {len(entries)} samples in "
        f"{(time.perf_counter() - start_time)/60:.1f}m"
    )


if __name__ == "__main__":
    main()
