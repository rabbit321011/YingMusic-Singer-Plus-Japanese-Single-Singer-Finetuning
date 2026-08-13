import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.YingMusicSinger.melody.game_cka_v4ph import (
    GAME_SOME_CKA_SCHEMA,
    game_note_probs_to_some_probs,
)
from src.YingMusicSinger.melody.game_p_v4pf import (
    GAME_P_ADAPTER_SCHEMA,
    quantize_game_notes,
)


GAME_CACHE_SCHEMA = {
    "version": 1,
    "format": "npz+json_manifest",
    "sample_rate": 44100,
    "game_commit": "4ad815c90dfe2442730f3fdc866fd23e737cbc97",
    "model_scale": "medium",
    "nsteps": 4,
    "boundary_threshold": 0.2,
    "boundary_radius": 2,
    "presence_threshold": 0.2,
    "p_adapter": GAME_P_ADAPTER_SCHEMA,
    "cka_adapter": GAME_SOME_CKA_SCHEMA,
}


def canonicalize_game_cache(notes):
    durations = notes["durations"].detach().float().cpu()
    presence = notes["presence"].detach().bool().cpu()
    scores = notes["scores"].detach().float().cpu()
    pitch_probs_257 = notes["pitch_probs_257"].detach().float().cpu()
    classes, duration_frames, valid = quantize_game_notes(
        durations=durations,
        presence=presence,
        scores=scores,
    )
    pitch_probs_128 = game_note_probs_to_some_probs(
        note_probs=pitch_probs_257,
        durations=durations,
        presence=presence,
    )
    arrays = {
        "durations": durations.numpy().astype(np.float32, copy=False),
        "presence": presence.numpy().astype(np.bool_, copy=False),
        "scores": scores.numpy().astype(np.float32, copy=False),
        "pitch_probs_257": pitch_probs_257.numpy().astype(np.float32, copy=False),
        "pitch_probs_128": pitch_probs_128.numpy().astype(np.float32, copy=False),
        "classes": classes.numpy().astype(np.int16, copy=False),
        "duration_frames_100hz": duration_frames.numpy().astype(np.int32, copy=False),
        "valid": valid.numpy().astype(np.bool_, copy=False),
    }
    validate_game_cache_arrays(arrays)
    return arrays


def validate_game_cache_arrays(arrays):
    required = {
        "durations",
        "presence",
        "scores",
        "pitch_probs_257",
        "pitch_probs_128",
        "classes",
        "duration_frames_100hz",
        "valid",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"GAME cache missing arrays: {sorted(missing)}")
    count = int(np.asarray(arrays["durations"]).shape[0])
    expected_shapes = {
        "durations": (count,),
        "presence": (count,),
        "scores": (count,),
        "pitch_probs_257": (count, 257),
        "pitch_probs_128": (count, 128),
        "classes": (count,),
        "duration_frames_100hz": (count,),
        "valid": (count,),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"GAME cache {name} shape {value.shape} != {shape}")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"GAME cache {name} contains non-finite values")
    probs_257 = np.asarray(arrays["pitch_probs_257"])
    probs_128 = np.asarray(arrays["pitch_probs_128"])
    if (probs_257 < 0).any() or (probs_257 > 1).any():
        raise ValueError("GAME cache pitch_probs_257 is outside [0,1]")
    if (probs_128 < 0).any() or (probs_128 > 1).any():
        raise ValueError("GAME cache pitch_probs_128 is outside [0,1]")
    valid = np.asarray(arrays["valid"], dtype=bool)
    classes = np.asarray(arrays["classes"])
    if valid.any() and ((classes[valid] < 0) | (classes[valid] > 255)).any():
        raise ValueError("GAME cache valid class is outside pitch/REST range")
    if (~valid).any() and (classes[~valid] != 256).any():
        raise ValueError("GAME cache invalid note does not use PAD=256")
    return count


def save_game_cache(path, arrays):
    validate_game_cache_arrays(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite GAME cache: {path}")
    np.savez(path, **arrays)


def load_game_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_game_cache_arrays(arrays)
    return {name: torch.from_numpy(value.copy()) for name, value in arrays.items()}


def load_game_cache_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("cache_schema") != GAME_CACHE_SCHEMA:
        raise ValueError("GAME cache manifest schema mismatch")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("GAME cache manifest entries must be a list")
    return payload


def crop_game_cache(cache, audio_duration):
    durations = cache["durations"].float()
    if durations.ndim != 1 or audio_duration <= 0:
        raise ValueError("Invalid GAME cache durations or audio duration")
    ends = torch.cumsum(durations, dim=0)
    starts = ends - durations
    cropped_durations = (
        torch.minimum(ends, ends.new_tensor(audio_duration)) - starts
    ).clamp(min=0)
    valid = cropped_durations > 0
    if not valid.any():
        raise ValueError("GAME cache has no notes inside the requested duration")
    result = {
        name: value[valid]
        for name, value in cache.items()
        if value.ndim >= 1 and value.shape[0] == durations.shape[0]
    }
    result["durations"] = cropped_durations[valid]
    closure = float(result["durations"].sum())
    if abs(closure - audio_duration) > 0.011:
        raise ValueError(
            f"Cropped GAME cache does not close: {closure} vs {audio_duration}"
        )
    result["durations"][-1] += audio_duration - closure
    return result


def game_cache_to_model_tracks(cache, num_samples, target_len, sample_rate=44100):
    if num_samples <= 0 or target_len <= 0:
        raise ValueError("num_samples and target_len must be positive")
    audio_duration = num_samples / sample_rate
    cropped = crop_game_cache(cache, audio_duration)
    durations = cropped["durations"].float()
    edges = torch.cumsum(durations, dim=0)
    edges[-1] = edges.new_tensor(audio_duration)

    target_times = (
        (torch.arange(target_len, dtype=edges.dtype, device=edges.device) + 0.5)
        * audio_duration
        / target_len
    )
    target_indices = torch.searchsorted(edges, target_times, right=True).clamp(
        max=durations.numel() - 1
    )
    p_classes = cropped["classes"].long()[target_indices]
    note_ids = target_indices + 1

    some_frames = num_samples // GAME_CACHE_SCHEMA["cka_adapter"]["some_hop_length"] + 1
    some_times = (
        torch.arange(some_frames, dtype=edges.dtype, device=edges.device)
        * GAME_CACHE_SCHEMA["cka_adapter"]["some_hop_length"]
        / sample_rate
    ).clamp(max=torch.nextafter(edges[-1], edges.new_zeros(())))
    some_indices = torch.searchsorted(edges, some_times, right=True).clamp(
        max=durations.numel() - 1
    )
    some_probs = cropped["pitch_probs_128"].float()[some_indices]
    cka_probs = F.interpolate(
        some_probs.transpose(0, 1).unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)[0]
    return {
        "p_classes": p_classes,
        "note_ids": note_ids,
        "cka_probs": cka_probs,
        "some_frames": some_frames,
    }
















