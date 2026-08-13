import math

import torch
import torch.nn.functional as F


GAME_SOME_CKA_SCHEMA = {
    "version": 1,
    "game_pitch_bins": 257,
    "game_midi_min": 0.0,
    "game_midi_max": 128.0,
    "game_pitch_step": 0.5,
    "game_training_sigma": 0.5,
    "some_pitch_bins": 128,
    "some_midi_min": 0.0,
    "some_midi_max": 127.0,
    "some_pitch_step": 1.0,
    "some_training_sigma": 1.0,
    "some_sample_rate": 44100,
    "some_hop_length": 512,
    "output": "sigmoid_pitch_probabilities",
    "selected_pitch_axis_mapping": "game_even_bins_0_254_to_some_bins_0_127",
}


def _validate_note_inputs(note_probs, durations, presence, scores):
    if note_probs.ndim != 2 or note_probs.shape[1] != 257:
        raise ValueError(
            f"Expected GAME note probabilities [N,257], got {tuple(note_probs.shape)}"
        )
    if durations.ndim != 1 or presence.ndim != 1 or scores.ndim != 1:
        raise ValueError("durations, presence and scores must be one-dimensional")
    if not (
        note_probs.shape[0]
        == durations.numel()
        == presence.numel()
        == scores.numel()
    ):
        raise ValueError("GAME note probability and decoded output lengths differ")
    if not torch.isfinite(note_probs).all() or not torch.isfinite(durations).all():
        raise ValueError("GAME probabilities or durations contain non-finite values")
    if not torch.isfinite(scores).all() or (durations < 0).any():
        raise ValueError("GAME scores are non-finite or durations are negative")
    if (note_probs < 0).any() or (note_probs > 1).any():
        raise ValueError("GAME note probabilities must be in [0,1]")


def some_gaussian_pitch_kernel(scores, presence, sigma=1.0):
    if scores.ndim != 1 or presence.ndim != 1 or scores.shape != presence.shape:
        raise ValueError("scores and presence must be equal one-dimensional tensors")
    centers = torch.arange(128, device=scores.device, dtype=scores.dtype)
    kernels = torch.exp(-0.5 * ((centers[None, :] - scores[:, None]) / sigma) ** 2)
    return kernels * presence[:, None].to(kernels.dtype)


def _broaden_game_probabilities(note_probs):
    game_sigma = GAME_SOME_CKA_SCHEMA["game_training_sigma"]
    some_sigma = GAME_SOME_CKA_SCHEMA["some_training_sigma"]
    pitch_step = GAME_SOME_CKA_SCHEMA["game_pitch_step"]
    extra_sigma_bins = math.sqrt(some_sigma**2 - game_sigma**2) / pitch_step
    radius = max(1, math.ceil(4 * extra_sigma_bins))
    offsets = torch.arange(
        -radius, radius + 1, device=note_probs.device, dtype=note_probs.dtype
    )
    kernel = torch.exp(-0.5 * (offsets / extra_sigma_bins) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    broadened = F.conv1d(
        note_probs[:, None, :], kernel, padding=radius
    )[:, 0, :]

    # Preserve GAME's confidence while changing only the pitch-axis width.
    original_peak = note_probs.amax(dim=1, keepdim=True)
    broadened_peak = broadened.amax(dim=1, keepdim=True).clamp(min=1e-8)
    return (broadened * (original_peak / broadened_peak)).clamp(0, 1)


def game_note_probs_to_some_candidates(note_probs, durations, presence, scores):
    """Return candidate GAME-to-SOME note probability mappings.

    All candidates are [N,128] probability arrays. `posterior_integer_bins`
    preserves the raw GAME posterior at integer MIDI centers. `posterior_broadened`
    first reconciles GAME's 0.5-semitone training sigma with SOME's 1-semitone
    sigma. `decoded_score_kernel` is a diagnostic upper bound that discards GAME
    uncertainty and reconstructs the canonical SOME target from decoded scores.
    """
    _validate_note_inputs(note_probs, durations, presence, scores)
    valid = durations > 0
    voiced = valid & presence.bool()
    voiced_float = voiced[:, None].to(note_probs.dtype)
    integer_bins = note_probs[:, 0:255:2] * voiced_float
    broadened = _broaden_game_probabilities(note_probs)[:, 0:255:2] * voiced_float
    canonical = some_gaussian_pitch_kernel(scores, voiced)
    return {
        "posterior_integer_bins": integer_bins,
        "posterior_broadened": broadened,
        "decoded_score_kernel": canonical,
    }


def game_note_probs_to_some_probs(note_probs, durations, presence):
    """Map the retained GAME posterior to the frozen SOME-compatible contract."""
    placeholder_scores = torch.zeros_like(durations)
    _validate_note_inputs(note_probs, durations, presence, placeholder_scores)
    voiced = (durations > 0) & presence.bool()
    return note_probs[:, 0:255:2] * voiced[:, None].to(note_probs.dtype)


def some_frame_count(num_samples, hop_length=512):
    if num_samples <= 0 or hop_length <= 0:
        raise ValueError("num_samples and hop_length must be positive")
    # torch.stft(center=True) pads n_fft//2 on both sides.
    return num_samples // hop_length + 1


def expand_game_note_features(
    note_features,
    durations,
    presence,
    audio_duration,
    target_frames,
    sample_rate=44100,
    hop_length=512,
):
    if note_features.ndim != 2 or note_features.shape[0] != durations.numel():
        raise ValueError("note_features must be [N,D] and match durations")
    if presence.ndim != 1 or presence.numel() != durations.numel():
        raise ValueError("presence must match durations")
    if target_frames <= 0 or audio_duration <= 0:
        raise ValueError("target_frames and audio_duration must be positive")

    valid = durations > 0
    if not valid.any():
        raise ValueError("GAME output has no positive-duration notes")
    features = note_features[valid]
    valid_presence = presence[valid].bool()
    valid_durations = durations[valid].float()
    edges = torch.cumsum(valid_durations, dim=0)
    edges[-1] = torch.as_tensor(audio_duration, device=edges.device, dtype=edges.dtype)
    if (torch.diff(edges) <= 0).any():
        raise ValueError("GAME note edges are not strictly increasing")

    frame_times = (
        torch.arange(target_frames, device=edges.device, dtype=edges.dtype)
        * hop_length
        / sample_rate
    ).clamp(max=torch.nextafter(edges[-1], edges.new_zeros(())))
    note_indices = torch.searchsorted(edges, frame_times, right=True).clamp(
        max=features.shape[0] - 1
    )
    frame_features = features[note_indices]
    frame_presence = valid_presence[note_indices]
    frame_note_ids = note_indices + 1
    boundary = torch.zeros(
        target_frames, 1, device=features.device, dtype=features.dtype
    )
    boundary[0, 0] = 1
    if target_frames > 1:
        boundary[1:, 0] = (note_indices[1:] != note_indices[:-1]).to(features.dtype)
    return {
        "pitch_probs": frame_features,
        "boundary_probs": boundary,
        "presence": frame_presence,
        "note_ids": frame_note_ids,
    }
















