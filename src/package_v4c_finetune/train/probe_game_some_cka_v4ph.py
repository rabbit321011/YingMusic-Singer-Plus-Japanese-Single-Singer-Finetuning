import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[2]
YING_REPO = PROJECT / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src").is_dir():
    YING_REPO = PROJECT
sys.path.insert(0, str(YING_REPO))

from src.YingMusicSinger.melody.game_cka_v4ph import (  # noqa: E402
    GAME_SOME_CKA_SCHEMA,
    expand_game_note_features,
    game_note_probs_to_some_candidates,
    some_frame_count,
)
from src.YingMusicSinger.melody.game_runtime_v4ph import (  # noqa: E402
    extract_game_notes_with_posterior,
    stable_game_seed,
)
from src.YingMusicSinger.melody.midi_extractor import (  # noqa: E402
    MIDIExtractor,
    decode_bounds_to_alignment,
    decode_gaussian_blurred_probs as decode_some_probs,
)
from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram  # noqa: E402


SAMPLE_RATE = 44100


def linear_cka(x, y):
    if x.shape[0] < 2:
        return float("nan")
    x = F.normalize(x.float(), dim=-1, eps=1e-8)
    y = F.normalize(y.float(), dim=-1, eps=1e-8)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = x.transpose(0, 1) @ y
    xx = x.transpose(0, 1) @ x
    yy = y.transpose(0, 1) @ y
    numerator = cross.square().sum()
    denominator = torch.sqrt(xx.square().sum() * yy.square().sum()).clamp(min=1e-8)
    return float((numerator / denominator).item())


def distribution_stats(probs, voiced):
    selected = probs[voiced]
    if selected.numel() == 0:
        return {
            "peak_mean": None,
            "l1_mean": None,
            "entropy_mean": None,
        }
    peaks = selected.amax(dim=-1)
    l1 = selected.sum(dim=-1)
    normalized = selected / l1[:, None].clamp(min=1e-8)
    entropy = -(normalized * normalized.clamp(min=1e-8).log()).sum(dim=-1)
    return {
        "peak_mean": float(peaks.mean().item()),
        "l1_mean": float(l1.mean().item()),
        "entropy_mean": float(entropy.mean().item()),
    }


def percentile(values, q):
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.float(), q).item())


def boundary_match(reference_starts, candidate_starts, tolerance=2):
    reference = [int(x) for x in reference_starts.tolist() if int(x) > 0]
    candidate = [int(x) for x in candidate_starts.tolist() if int(x) > 0]
    used = set()
    matches = 0
    for point in reference:
        choices = [
            (abs(point - other), index)
            for index, other in enumerate(candidate)
            if index not in used and abs(point - other) <= tolerance
        ]
        if choices:
            _, index = min(choices)
            used.add(index)
            matches += 1
    precision = matches / max(len(candidate), 1)
    recall = matches / max(len(reference), 1)
    return {
        "reference": len(reference),
        "candidate": len(candidate),
        "matched": matches,
        "precision": precision,
        "recall": recall,
    }


def compare_candidate(
    some_probs,
    some_presence,
    some_pitch,
    some_note_ids,
    candidate_track,
):
    candidate_probs = candidate_track["pitch_probs"].float()
    candidate_presence = candidate_track["presence"].bool()
    candidate_note_ids = candidate_track["note_ids"].long()
    if candidate_probs.shape != some_probs.shape:
        raise AssertionError(
            f"SOME/GAME shape mismatch: {some_probs.shape} vs {candidate_probs.shape}"
        )
    candidate_pitch, _ = decode_some_probs(
        candidate_probs.unsqueeze(0), 0, 127, 1.0, 0.1
    )
    candidate_pitch = candidate_pitch[0]
    mutual = some_presence & candidate_presence
    pitch_error = (some_pitch[mutual] - candidate_pitch[mutual]).abs()
    cosine = F.cosine_similarity(
        some_probs[mutual], candidate_probs[mutual], dim=-1, eps=1e-8
    ) if mutual.any() else torch.empty(0)
    close_pitch = mutual & ((some_pitch - candidate_pitch).abs() <= 0.5)
    close_cosine = F.cosine_similarity(
        some_probs[close_pitch], candidate_probs[close_pitch], dim=-1, eps=1e-8
    ) if close_pitch.any() else torch.empty(0)
    some_starts = torch.nonzero(
        torch.diff(some_note_ids, prepend=some_note_ids.new_zeros(1)) > 0
    ).flatten()
    candidate_starts = torch.nonzero(
        torch.diff(candidate_note_ids, prepend=candidate_note_ids.new_zeros(1)) > 0
    ).flatten()
    return {
        "voiced_agreement": float((some_presence == candidate_presence).float().mean()),
        "mutual_voiced_frames": int(mutual.sum()),
        "pitch_error_median": percentile(pitch_error, 0.5),
        "pitch_error_p90": percentile(pitch_error, 0.9),
        "prob_cosine_mean": float(cosine.mean()) if cosine.numel() else None,
        "same_pitch_prob_cosine_mean": (
            float(close_cosine.mean()) if close_cosine.numel() else None
        ),
        "linear_cka": linear_cka(some_probs[mutual], candidate_probs[mutual]),
        "some_distribution": distribution_stats(some_probs, some_presence),
        "candidate_distribution": distribution_stats(
            candidate_probs, candidate_presence
        ),
        "boundary": boundary_match(some_starts, candidate_starts),
    }


def mean_or_none(values):
    valid = [value for value in values if value is not None and math.isfinite(value)]
    return sum(valid) / len(valid) if valid else None


def aggregate(entries, candidate_name):
    metrics = [entry["candidates"][candidate_name] for entry in entries]
    keys = [
        "voiced_agreement",
        "pitch_error_median",
        "pitch_error_p90",
        "prob_cosine_mean",
        "same_pitch_prob_cosine_mean",
        "linear_cka",
    ]
    result = {key: mean_or_none([metric[key] for metric in metrics]) for key in keys}
    result["samples"] = len(entries)
    result["boundary_precision"] = mean_or_none(
        [metric["boundary"]["precision"] for metric in metrics]
    )
    result["boundary_recall"] = mean_or_none(
        [metric["boundary"]["recall"] for metric in metrics]
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_repo", required=True)
    parser.add_argument("--game_model", required=True)
    parser.add_argument("--some_model", required=True)
    parser.add_argument("--some_pack_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nsteps", type=int, default=4)
    parser.add_argument("--base_seed", type=int, default=20260730)
    parser.add_argument("--ids", default=None)
    args = parser.parse_args()

    game_repo = str(Path(args.game_repo).resolve())
    if game_repo not in sys.path:
        sys.path.insert(0, game_repo)
    from inference.api import load_inference_model

    device = torch.device(args.device)
    game_model_path = Path(args.game_model).resolve()
    game, language_map = load_inference_model(game_model_path)
    game = game.to(device).eval()
    some = MIDIExtractor(in_dim=80)
    some._load_form_ckpt(str(Path(args.some_model).resolve()))
    some = some.to(device).eval()
    some_mel = MelodySpectrogram(mel_fmin=40, mel_fmax=8000).to(device)

    pack_dir = Path(args.some_pack_dir).resolve()
    pack_report = json.loads(
        (pack_dir / "report_selected.json").read_text(encoding="utf-8")
    )
    entries = pack_report["entries"]
    if args.ids:
        wanted = {value.strip() for value in args.ids.split(",")}
        entries = [entry for entry in entries if entry["id"] in wanted]
        missing = wanted - {entry["id"] for entry in entries}
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")

    results = []
    for index, entry in enumerate(entries, start=1):
        sample_id = entry["id"]
        audio_path = pack_dir / sample_id / "original_mono.wav"
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"Unexpected audio format for {sample_id}: {sample_rate}")
        waveform = torch.from_numpy(audio)
        duration = audio.size / sample_rate

        with torch.inference_mode():
            mel = some_mel(waveform[None, :].to(device), sample_rate)
            some_logits, some_bound_logits = some(mel.transpose(1, 2))
            some_probs = some_logits[0].sigmoid().float().cpu()
            some_bound_probs = some_bound_logits[0, :, 0].sigmoid().float().cpu()
        expected_some_frames = some_frame_count(audio.size)
        if some_probs.shape != (expected_some_frames, 128):
            raise AssertionError(
                f"{sample_id}: SOME frame mismatch {some_probs.shape} "
                f"!= {(expected_some_frames, 128)}"
            )
        some_pitch, some_rest = decode_some_probs(
            some_probs.unsqueeze(0), 0, 127, 1.0, 0.1
        )
        some_pitch = some_pitch[0]
        some_presence = ~some_rest[0]
        some_note_ids = decode_bounds_to_alignment(
            some_bound_probs.unsqueeze(0)
        )[0]

        language_id = language_map["ja"] if entry["group"] == "hanamaru_candidate" else 0
        seed = stable_game_seed(sample_id, args.base_seed)
        game_output = extract_game_notes_with_posterior(
            game, waveform, duration, language_id, args.nsteps, seed, device
        )
        durations = game_output["durations"]
        presence = game_output["presence"]
        scores = game_output["scores"]
        note_probs = game_output["pitch_probs_257"]
        note_candidates = game_note_probs_to_some_candidates(
            note_probs=note_probs,
            durations=durations,
            presence=presence,
            scores=scores,
        )
        candidate_metrics = {}
        for name, note_features in note_candidates.items():
            track = expand_game_note_features(
                note_features=note_features,
                durations=durations,
                presence=presence,
                audio_duration=duration,
                target_frames=some_probs.shape[0],
            )
            candidate_metrics[name] = compare_candidate(
                some_probs=some_probs,
                some_presence=some_presence,
                some_pitch=some_pitch,
                some_note_ids=some_note_ids,
                candidate_track=track,
            )

        results.append(
            {
                "id": sample_id,
                "group": entry["group"],
                "duration": duration,
                "some_frames": int(some_probs.shape[0]),
                "game_notes": int((durations > 0).sum()),
                "game_voiced_notes": int(((durations > 0) & presence).sum()),
                "language_id": language_id,
                "seed": seed,
                "candidates": candidate_metrics,
            }
        )
        best = candidate_metrics["posterior_broadened"]
        print(
            f"[{index}/{len(entries)}] {sample_id} "
            f"pitch50={best['pitch_error_median']!s} "
            f"cos={best['prob_cosine_mean']!s} cka={best['linear_cka']!s}"
        )

    candidate_names = list(results[0]["candidates"]) if results else []
    groups = sorted({entry["group"] for entry in results})
    aggregates = {
        "all": {name: aggregate(results, name) for name in candidate_names},
        "by_group": {
            group: {
                name: aggregate(
                    [entry for entry in results if entry["group"] == group], name
                )
                for name in candidate_names
            }
            for group in groups
        },
    }
    report = {
        "schema": "v4ph_game_some_cka_equivalence_v1",
        "adapter_schema": GAME_SOME_CKA_SCHEMA,
        "game_commit": "4ad815c90dfe2442730f3fdc866fd23e737cbc97",
        "game_model": str(game_model_path),
        "game_model_sha256": hashlib.sha256(game_model_path.read_bytes()).hexdigest(),
        "some_model": str(Path(args.some_model).resolve()),
        "some_mel": {"fmin": 40, "fmax": 8000, "hop_length": 512},
        "nsteps": args.nsteps,
        "base_seed": args.base_seed,
        "aggregates": aggregates,
        "entries": results,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
