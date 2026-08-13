import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import torch
import torchaudio


for candidate in (
    os.getcwd(),
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "YingMusic-Singer-Plus-src")
    ),
):
    if os.path.isdir(os.path.join(candidate, "src")) and candidate not in sys.path:
        sys.path.insert(0, candidate)


SAMPLE_RATE = 44100
VAE_FRAME_RATE = SAMPLE_RATE / 2048


def contiguous_notes(classes, note_ids, duration):
    classes = classes.tolist()
    note_ids = note_ids.tolist()
    frame_count = len(classes)
    notes = []
    start = 0
    while start < frame_count:
        end = start + 1
        while (
            end < frame_count
            and note_ids[end] == note_ids[start]
            and classes[end] == classes[start]
        ):
            end += 1
        class_id = int(classes[start])
        notes.append(
            {
                "start": start / frame_count * duration,
                "end": end / frame_count * duration,
                "class_id": class_id,
                "midi": class_id / 2 if class_id < 255 else None,
                "rest": class_id == 255,
                "note_id": int(note_ids[start]),
            }
        )
        start = end
    return notes


def render_sine(notes, duration, sample_rate=SAMPLE_RATE):
    sample_count = max(1, round(duration * sample_rate))
    output = torch.zeros(sample_count, dtype=torch.float32)
    attack_samples = round(0.008 * sample_rate)
    release_samples = round(0.012 * sample_rate)
    for note in notes:
        if note["rest"]:
            continue
        start = max(0, min(sample_count, round(note["start"] * sample_rate)))
        end = max(start, min(sample_count, round(note["end"] * sample_rate)))
        length = end - start
        if length <= 0:
            continue
        frequency = 440.0 * (2.0 ** ((note["midi"] - 69.0) / 12.0))
        time = torch.arange(length, dtype=torch.float32) / sample_rate
        tone = torch.sin(2 * math.pi * frequency * time)
        envelope = torch.ones(length, dtype=torch.float32)
        attack = min(attack_samples, max(1, length // 2))
        release = min(release_samples, max(1, length // 2))
        envelope[:attack] *= torch.linspace(0, 1, attack)
        envelope[-release:] *= torch.linspace(1, 0, release)
        output[start:end] = 0.22 * tone * envelope
    return output.unsqueeze(0)


def write_notes(path, notes):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "class_id", "midi", "rest", "note_id"],
        )
        writer.writeheader()
        writer.writerows(notes)


def track_stats(classes, note_ids):
    from src.YingMusicSinger.melody.midi_p_v4pf import summarize_midi_classes

    stats = summarize_midi_classes(classes.unsqueeze(0))
    stats["decoded_note_count"] = int(torch.unique_consecutive(note_ids).numel())
    if stats["min_pitch_class"] is not None:
        stats["min_midi"] = stats["min_pitch_class"] / 2
        stats["max_midi"] = stats["max_pitch_class"] / 2
    else:
        stats["min_midi"] = None
        stats["max_midi"] = None
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--midi_ckpt", default="ckpts/model_ckpt_steps_100000_simplified.ckpt"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mel_fmin", type=float, default=40.0)
    parser.add_argument("--mel_fmax", type=float, default=8000.0)
    args = parser.parse_args()

    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.melody.midi_p_v4pf import (
        MIDI_P_SCHEMA,
        decode_quantized_midi_tracks,
    )
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram

    manifest_path = Path(args.manifest).resolve()
    input_root = manifest_path.parent
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    device = torch.device(args.device)
    mel_extractor = MelodySpectrogram(
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
    ).to(device)
    teacher = MIDIExtractor(in_dim=80)
    teacher._load_form_ckpt(args.midi_ckpt)
    teacher = teacher.to(device).eval()

    results = []
    for index, entry in enumerate(manifest["entries"], start=1):
        input_path = input_root / entry["input"]
        sample_dir = output_dir / entry["id"]
        sample_dir.mkdir()
        wav, sample_rate = torchaudio.load(input_path)
        wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE)
            sample_rate = SAMPLE_RATE
        duration = wav.shape[-1] / sample_rate

        with torch.inference_mode():
            mel = mel_extractor(audio=wav.to(device), sr=sample_rate)
            midi_logits, boundary_logits = teacher(mel.transpose(1, 2))
            target_len = max(1, round(duration * VAE_FRAME_RATE))
            tracks = decode_quantized_midi_tracks(
                midi_logits=midi_logits,
                boundary_logits=boundary_logits,
                target_len=target_len,
            )

        native_classes = tracks["native_classes"][0].cpu()
        native_note_ids = tracks["native_note_ids"][0].cpu()
        model_classes = tracks["model_classes"][0].cpu()
        model_note_ids = tracks["model_note_ids"][0].cpu()
        native_notes = contiguous_notes(native_classes, native_note_ids, duration)
        model_notes = contiguous_notes(model_classes, model_note_ids, duration)

        torchaudio.save(sample_dir / "original_mono.wav", wav.cpu(), sample_rate)
        torchaudio.save(
            sample_dir / "p_native_sine.wav",
            render_sine(native_notes, duration),
            SAMPLE_RATE,
        )
        torchaudio.save(
            sample_dir / "p_model_sine.wav",
            render_sine(model_notes, duration),
            SAMPLE_RATE,
        )
        write_notes(sample_dir / "p_native_notes.csv", native_notes)
        write_notes(sample_dir / "p_model_notes.csv", model_notes)

        result = {
            **entry,
            "input": str(input_path),
            "decoded_duration": duration,
            "source_sample_rate": sample_rate,
            "source_channels_after_mix": 1,
            "some_frames": int(midi_logits.shape[1]),
            "model_frames": target_len,
            "native": track_stats(native_classes, native_note_ids),
            "model": track_stats(model_classes, model_note_ids),
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        results.append(result)
        print(
            f"[{index}/{len(manifest['entries'])}] {entry['id']} "
            f"duration={duration:.3f}s model={result['model']}"
        )

    report = {
        "schema": "v4pf_some_p_audition_report_v1",
        "midi_p_schema": MIDI_P_SCHEMA,
        "some_mel": {
            "sample_rate": SAMPLE_RATE,
            "n_mels": 80,
            "win_length": 2048,
            "hop_length": 512,
            "fmin": args.mel_fmin,
            "fmax": args.mel_fmax,
            "source": "OpenVPI/SOME configs/base.yaml",
        },
        "manifest": str(manifest_path),
        "entries": results,
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
