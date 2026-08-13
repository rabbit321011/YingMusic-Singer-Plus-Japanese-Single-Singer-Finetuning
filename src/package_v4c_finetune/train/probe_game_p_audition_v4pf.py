import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PROJECT = Path(__file__).resolve().parents[2]
YING_REPO = PROJECT / "YingMusic-Singer-Plus-src"
sys.path.insert(0, str(YING_REPO))

from src.YingMusicSinger.melody.game_p_v4pf import (  # noqa: E402
    GAME_P_ADAPTER_SCHEMA,
    game_notes_to_tracks,
)
from render_p_timbre_audition_v4pf import render_piano, write_ogg  # noqa: E402


SAMPLE_RATE = 44100
VAE_FRAME_RATE = SAMPLE_RATE / 2048


def stable_seed(sample_id, base_seed):
    payload = f"{sample_id}|{base_seed}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def run_game(model, waveform, duration, language_id, nsteps, seed, device):
    schedule = torch.arange(nsteps, device=device, dtype=torch.float32) / nsteps
    known_durations = torch.tensor([[duration]], device=device, dtype=torch.float32)
    boundary_threshold = torch.tensor(0.2, device=device)
    boundary_radius = torch.tensor(2, device=device, dtype=torch.long)
    score_threshold = torch.tensor(0.2, device=device)
    language = torch.tensor([language_id], device=device, dtype=torch.long)
    waveform = waveform.unsqueeze(0).to(device)

    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            durations, presence, scores = model(
                waveform=waveform,
                known_durations=known_durations,
                boundary_threshold=boundary_threshold,
                boundary_radius=boundary_radius,
                score_threshold=score_threshold,
                language=language,
                t=schedule,
            )
        elapsed = time.perf_counter() - start
    peak_mib = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0
    )
    return (
        durations[0].float().cpu(),
        presence[0].bool().cpu(),
        scores[0].float().cpu(),
        elapsed,
        peak_mib,
    )


def notes_from_tracks(classes, note_ids, duration):
    classes = classes.tolist()
    note_ids = note_ids.tolist()
    notes = []
    start = 0
    while start < len(classes):
        end = start + 1
        while (
            end < len(classes)
            and note_ids[end] == note_ids[start]
            and classes[end] == classes[start]
        ):
            end += 1
        class_id = int(classes[start])
        notes.append(
            {
                "start": start / len(classes) * duration,
                "end": end / len(classes) * duration,
                "class_id": class_id,
                "midi": class_id / 2 if class_id < 255 else None,
                "rest": class_id == 255,
                "note_id": int(note_ids[start]),
            }
        )
        start = end
    return notes


def write_notes(path, notes):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "class_id", "midi", "rest", "note_id"],
        )
        writer.writeheader()
        writer.writerows(notes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--some_pack_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nsteps", type=int, default=4)
    parser.add_argument("--base_seed", type=int, default=20260730)
    args = parser.parse_args()

    game_repo = str(Path(args.game_repo).resolve())
    if game_repo not in sys.path:
        sys.path.insert(0, game_repo)
    from inference.api import load_inference_model

    model_path = Path(args.model).resolve()
    some_pack_dir = Path(args.some_pack_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    report = json.loads(
        (some_pack_dir / "report_selected.json").read_text(encoding="utf-8")
    )
    entries = {entry["id"]: entry for entry in report["entries"]}
    sample_ids = [sample_id.strip() for sample_id in args.ids.split(",")]
    unknown = [sample_id for sample_id in sample_ids if sample_id not in entries]
    if unknown:
        raise ValueError(f"Unknown sample IDs: {unknown}")

    device = torch.device(args.device)
    game, language_map = load_inference_model(model_path)
    game = game.to(device).eval()
    results = []
    for index, sample_id in enumerate(sample_ids, start=1):
        entry = entries[sample_id]
        audio_path = some_pack_dir / sample_id / "original_mono.wav"
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"Unexpected audio format for {sample_id}: {sample_rate}")
        duration = audio.size / sample_rate
        language_id = language_map["ja"] if entry["group"] == "hanamaru_candidate" else 0
        seed = stable_seed(sample_id, args.base_seed)
        durations, presence, scores, elapsed, peak_mib = run_game(
            game,
            torch.from_numpy(audio),
            duration,
            language_id,
            args.nsteps,
            seed,
            device,
        )
        target_len = max(1, round(duration * VAE_FRAME_RATE))
        tracks = game_notes_to_tracks(
            durations=durations,
            presence=presence,
            scores=scores,
            expected_native_frames=round(duration / GAME_P_ADAPTER_SCHEMA["source_timestep"]),
            target_len=target_len,
        )
        native_notes = notes_from_tracks(
            tracks["native_classes"], tracks["native_note_ids"], duration
        )
        model_notes = notes_from_tracks(
            tracks["model_classes"], tracks["model_note_ids"], duration
        )

        sample_dir = output_dir / sample_id
        sample_dir.mkdir()
        write_notes(sample_dir / "game_native_notes.csv", native_notes)
        write_notes(sample_dir / "game_model_notes.csv", model_notes)
        write_ogg(
            sample_dir / "game_native_piano.ogg", render_piano(native_notes, duration)
        )
        write_ogg(
            sample_dir / "game_model_piano.ogg", render_piano(model_notes, duration)
        )
        raw_notes = [
            {
                "duration": float(note_duration),
                "presence": bool(note_presence),
                "score": float(note_score),
            }
            for note_duration, note_presence, note_score in zip(
                durations.tolist(), presence.tolist(), scores.tolist()
            )
            if note_duration > 0
        ]
        result = {
            **entry,
            "duration": duration,
            "language_id": language_id,
            "nsteps": args.nsteps,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "peak_memory_mib": peak_mib,
            "game_note_count": len(raw_notes),
            "game_voiced_count": sum(note["presence"] for note in raw_notes),
            "game_rest_count": sum(not note["presence"] for note in raw_notes),
            "native_frames": int(tracks["native_classes"].numel()),
            "native_frame_delta": int(tracks["native_frame_delta"]),
            "model_frames": int(tracks["model_classes"].numel()),
            "raw_notes": raw_notes,
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)
        print(
            f"[{index}/{len(sample_ids)}] {sample_id} notes={len(raw_notes)} "
            f"voiced={result['game_voiced_count']} time={elapsed:.2f}s "
            f"peak={peak_mib:.0f}MiB"
        )

    probe_report = {
        "schema": "v4pf_game_p_probe_v1",
        "adapter_schema": GAME_P_ADAPTER_SCHEMA,
        "game_commit": "4ad815c90dfe2442730f3fdc866fd23e737cbc97",
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "nsteps": args.nsteps,
        "base_seed": args.base_seed,
        "entries": results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(probe_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
