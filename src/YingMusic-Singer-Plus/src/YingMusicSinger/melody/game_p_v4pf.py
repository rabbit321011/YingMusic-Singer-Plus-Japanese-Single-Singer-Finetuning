import torch


GAME_P_ADAPTER_SCHEMA = {
    "version": 1,
    "source": "OpenVPI/GAME durations,presence,scores",
    "source_timestep": 0.01,
    "pitch_scale": 2,
    "pitch_class_count": 255,
    "rest_id": 255,
    "pad_id": 256,
    "time_alignment": "center_aligned_discrete_lookup",
    "out_of_range": "error",
}


def quantize_game_notes(durations, presence, scores, timestep=0.01):
    if durations.ndim != 1 or presence.ndim != 1 or scores.ndim != 1:
        raise ValueError("durations, presence and scores must be one-dimensional")
    if durations.shape != presence.shape or durations.shape != scores.shape:
        raise ValueError(
            f"GAME output shape mismatch: {durations.shape}, {presence.shape}, "
            f"{scores.shape}"
        )
    if not torch.isfinite(durations).all() or not torch.isfinite(scores).all():
        raise ValueError("GAME output contains non-finite durations or scores")
    if (durations < 0).any():
        raise ValueError("GAME output contains negative durations")

    duration_frames = torch.round(durations.float() / timestep).long()
    valid = duration_frames > 0
    voiced = valid & presence.bool()
    rest = valid & ~presence.bool()
    classes = torch.full_like(duration_frames, GAME_P_ADAPTER_SCHEMA["pad_id"])
    classes[rest] = GAME_P_ADAPTER_SCHEMA["rest_id"]

    quantized_pitch = torch.round(
        scores.float() * GAME_P_ADAPTER_SCHEMA["pitch_scale"]
    ).long()
    invalid_pitch = voiced & (
        (quantized_pitch < 0)
        | (quantized_pitch >= GAME_P_ADAPTER_SCHEMA["pitch_class_count"])
    )
    if invalid_pitch.any():
        bad = scores[invalid_pitch].detach().cpu().tolist()
        raise ValueError(f"GAME voiced pitch is outside the V4Pf schema: {bad}")
    classes[voiced] = quantized_pitch[voiced]
    return classes, duration_frames, valid


def expand_game_notes(
    durations,
    presence,
    scores,
    expected_frames,
    timestep=0.01,
    max_frame_delta=1,
):
    classes, duration_frames, valid = quantize_game_notes(
        durations=durations,
        presence=presence,
        scores=scores,
        timestep=timestep,
    )
    valid_classes = classes[valid]
    valid_durations = duration_frames[valid]
    native_classes = torch.repeat_interleave(valid_classes, valid_durations)
    note_numbers = torch.arange(
        1,
        valid_classes.numel() + 1,
        dtype=torch.long,
        device=classes.device,
    )
    native_note_ids = torch.repeat_interleave(note_numbers, valid_durations)
    frame_delta = native_classes.numel() - expected_frames
    if abs(frame_delta) > max_frame_delta:
        raise ValueError(
            f"GAME duration closure mismatch: expanded={native_classes.numel()}, "
            f"expected={expected_frames}"
        )
    return {
        "note_classes": valid_classes,
        "note_duration_frames": valid_durations,
        "native_classes": native_classes,
        "native_note_ids": native_note_ids,
        "native_frame_delta": frame_delta,
    }


def resize_discrete_game_track(native_classes, native_note_ids, target_len):
    if native_classes.ndim != 1 or native_note_ids.ndim != 1:
        raise ValueError("native_classes and native_note_ids must be one-dimensional")
    if native_classes.shape != native_note_ids.shape or native_classes.numel() == 0:
        raise ValueError("GAME native tracks must be non-empty and have equal lengths")
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")

    source_len = native_classes.numel()
    source_indices = torch.floor(
        (torch.arange(target_len, device=native_classes.device).float() + 0.5)
        * source_len
        / target_len
    ).long().clamp(max=source_len - 1)
    return native_classes[source_indices], native_note_ids[source_indices]


def game_notes_to_tracks(
    durations,
    presence,
    scores,
    expected_native_frames,
    target_len,
    timestep=0.01,
    max_frame_delta=1,
):
    tracks = expand_game_notes(
        durations=durations,
        presence=presence,
        scores=scores,
        expected_frames=expected_native_frames,
        timestep=timestep,
        max_frame_delta=max_frame_delta,
    )
    model_classes, model_note_ids = resize_discrete_game_track(
        tracks["native_classes"], tracks["native_note_ids"], target_len
    )
    tracks["model_classes"] = model_classes
    tracks["model_note_ids"] = model_note_ids
    return tracks
















