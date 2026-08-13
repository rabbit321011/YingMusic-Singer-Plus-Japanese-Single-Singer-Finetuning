import torch
import torch.nn as nn
import torch.nn.functional as F


MIDI_P_SCHEMA = {
    "version": 1,
    "pitch_scale": 2,
    "pitch_class_count": 255,
    "rest_id": 255,
    "pad_id": 256,
    "num_embeddings": 257,
    "embedding_dim": 128,
    "fuzz_disturb": False,
    "cka_target": "raw_some",
}


def _decode_frame_pitch(probs, deviation=1.0, rest_threshold=0.1):
    num_bins = probs.shape[-1]
    interval = 127.0 / (num_bins - 1)
    width = int(3 * deviation / interval)
    idx = torch.arange(num_bins, device=probs.device)[None, None, :]
    idx_values = idx * interval
    center = torch.argmax(probs, dim=-1, keepdim=True)
    start = torch.clamp(center - width, min=0)
    end = torch.clamp(center + width + 1, max=num_bins)
    local = (idx >= start) & (idx < end)
    weights = probs * local
    weight_sum = weights.sum(dim=2)
    pitch = (weights * idx_values).sum(dim=2) / (
        weight_sum + (weight_sum == 0)
    )
    rest = probs.max(dim=-1).values < rest_threshold
    return pitch, rest


def _decode_boundaries(bound_probs):
    cumulative = bound_probs.cumsum(dim=1).round().long()
    starts = torch.diff(
        cumulative,
        dim=1,
        prepend=torch.full(
            (bound_probs.shape[0], 1),
            -1,
            dtype=cumulative.dtype,
            device=cumulative.device,
        ),
    ) > 0
    return starts.long().cumsum(dim=1)


def _aggregate_notes(frame_to_note, frame_pitch, voiced_frames, threshold=0.5):
    batch = frame_to_note.shape[0]
    note_slots = int(frame_to_note.max().item()) + 1
    note_duration = frame_to_note.new_zeros(batch, note_slots).scatter_add(
        1, frame_to_note, torch.ones_like(frame_to_note)
    )[:, 1:]
    voiced_duration = frame_to_note.new_zeros(batch, note_slots).scatter_add(
        1, frame_to_note, voiced_frames.long()
    )[:, 1:]
    note_voiced = voiced_duration.float() / note_duration.clamp(min=1).float() >= threshold

    rounded_pitch = frame_pitch.round().long().clamp(0, 127)
    histogram = frame_to_note.new_zeros(batch, note_slots * 128).scatter_add(
        1,
        frame_to_note * 128 + rounded_pitch,
        voiced_frames.long(),
    ).unflatten(1, [note_slots, 128])[:, 1:, :]
    center = histogram.argmax(dim=2).to(frame_pitch.dtype)
    center_per_frame = torch.gather(F.pad(center, [1, 0]), 1, frame_to_note)
    near_center = (
        voiced_frames
        & (frame_pitch >= center_per_frame - 0.5)
        & (frame_pitch <= center_per_frame + 0.5)
    )
    valid_duration = frame_to_note.new_zeros(batch, note_slots).scatter_add(
        1, frame_to_note, near_center.long()
    )[:, 1:]
    note_pitch = frame_pitch.new_zeros(batch, note_slots).scatter_add(
        1, frame_to_note, frame_pitch * near_center
    )[:, 1:] / valid_duration.clamp(min=1)
    return note_pitch, note_duration, note_voiced


def _expand_notes(note_pitch, note_duration, note_voiced):
    pitches = []
    voiced = []
    for batch_index in range(note_pitch.shape[0]):
        durations = note_duration[batch_index]
        pitches.append(torch.repeat_interleave(note_pitch[batch_index], durations))
        voiced.append(torch.repeat_interleave(note_voiced[batch_index], durations))
    return pitches, voiced


def decode_quantized_midi_classes(midi_logits, boundary_logits, target_len):
    if midi_logits.ndim != 3 or midi_logits.shape[-1] != 128:
        raise ValueError(f"Expected MIDI logits [B,T,128], got {tuple(midi_logits.shape)}")
    if boundary_logits.ndim == 3:
        if boundary_logits.shape[-1] != 1:
            raise ValueError(
                f"Expected boundary logits [B,T,1], got {tuple(boundary_logits.shape)}"
            )
        boundary_logits = boundary_logits.squeeze(-1)
    if boundary_logits.shape != midi_logits.shape[:2]:
        raise ValueError(
            f"MIDI/boundary length mismatch: {tuple(midi_logits.shape)} vs "
            f"{tuple(boundary_logits.shape)}"
        )
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")

    with torch.no_grad():
        probs = torch.sigmoid(midi_logits.detach())
        bound_probs = torch.sigmoid(boundary_logits.detach())
        frame_pitch, frame_rest = _decode_frame_pitch(probs)
        frame_to_note = _decode_boundaries(bound_probs)
        note_pitch, note_duration, note_voiced = _aggregate_notes(
            frame_to_note, frame_pitch, ~frame_rest
        )
        expanded_pitch, expanded_voiced = _expand_notes(
            note_pitch, note_duration, note_voiced
        )

        class_rows = []
        for pitch, voiced in zip(expanded_pitch, expanded_voiced):
            if pitch.numel() == 0:
                raise RuntimeError("SOME note decoder produced an empty sequence")
            voiced_float = voiced.float()
            resized_pitch_sum = F.interpolate(
                (pitch * voiced_float)[None, None, :],
                size=target_len,
                mode="linear",
                align_corners=False,
            )[0, 0]
            resized_voiced_weight = F.interpolate(
                voiced_float[None, None, :],
                size=target_len,
                mode="linear",
                align_corners=False,
            )[0, 0]
            nearest_pitch = F.interpolate(
                pitch[None, None, :],
                size=target_len,
                mode="nearest",
            )[0, 0]
            resized_pitch = torch.where(
                resized_voiced_weight > 1e-8,
                resized_pitch_sum / resized_voiced_weight.clamp(min=1e-8),
                nearest_pitch,
            )
            resized_voiced = F.interpolate(
                voiced_float[None, None, :],
                size=target_len,
                mode="nearest",
            )[0, 0].bool()
            pitch_class = torch.round(resized_pitch * 2).long().clamp(0, 254)
            class_rows.append(
                torch.where(
                    resized_voiced,
                    pitch_class,
                    torch.full_like(pitch_class, MIDI_P_SCHEMA["rest_id"]),
                )
            )
        classes = torch.stack(class_rows, dim=0)
    return classes


class QuantizedMIDIPV4Pvf(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            MIDI_P_SCHEMA["num_embeddings"],
            MIDI_P_SCHEMA["embedding_dim"],
        )

    def forward(self, midi_logits, boundary_logits, target_len):
        classes = decode_quantized_midi_classes(
            midi_logits=midi_logits,
            boundary_logits=boundary_logits,
            target_len=target_len,
        )
        return self.embedding(classes), classes


def summarize_midi_classes(classes):
    if classes.ndim != 2:
        raise ValueError(f"Expected class IDs [B,T], got {tuple(classes.shape)}")
    total = classes.numel()
    rest = int((classes == MIDI_P_SCHEMA["rest_id"]).sum().item())
    pad = int((classes == MIDI_P_SCHEMA["pad_id"]).sum().item())
    voiced = classes[(classes >= 0) & (classes < MIDI_P_SCHEMA["pitch_class_count"])]
    return {
        "frames": total,
        "rest_frames": rest,
        "pad_frames": pad,
        "rest_ratio": rest / max(total, 1),
        "unique_pitch_classes": int(torch.unique(voiced).numel()),
        "min_pitch_class": int(voiced.min().item()) if voiced.numel() else None,
        "max_pitch_class": int(voiced.max().item()) if voiced.numel() else None,
    }
















