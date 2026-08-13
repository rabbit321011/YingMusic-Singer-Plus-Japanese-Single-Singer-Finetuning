import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 44100
HARMONIC_AMPLITUDES = np.array(
    [1.0, 0.48, 0.28, 0.17, 0.10, 0.06, 0.04, 0.025], dtype=np.float64
)


def read_notes(path):
    notes = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            notes.append(
                {
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "midi": None if not row["midi"] else float(row["midi"]),
                    "rest": row["rest"].lower() == "true",
                    "note_id": int(row["note_id"]),
                }
            )
    return notes


def piano_note(midi, sample_count, note_id, sample_rate=SAMPLE_RATE):
    if sample_count <= 0:
        return np.empty(0, dtype=np.float32)
    frequency = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    tone = np.zeros(sample_count, dtype=np.float64)
    stiffness = 1.5e-4
    string_cents = (-0.7, 0.0, 0.7)
    string_weights = (0.22, 0.56, 0.22)

    for harmonic, amplitude in enumerate(HARMONIC_AMPLITUDES, start=1):
        partial_frequency = frequency * harmonic * math.sqrt(
            1.0 + stiffness * harmonic * harmonic
        )
        decay_tau = max(0.22, 3.0 * (440.0 / frequency) ** 0.18 / harmonic**0.42)
        decay = 0.10 + 0.90 * np.exp(-time / decay_tau)
        partial = np.zeros(sample_count, dtype=np.float64)
        for cents, weight in zip(string_cents, string_weights):
            detuned = partial_frequency * (2.0 ** (cents / 1200.0))
            partial += weight * np.sin(2.0 * np.pi * detuned * time)
        tone += amplitude * decay * partial

    rng = np.random.default_rng((round(midi * 2) * 65537 + note_id) & 0xFFFFFFFF)
    hammer = rng.standard_normal(sample_count) * np.exp(-time / 0.009)
    tone += 0.025 * hammer

    attack = min(sample_count, max(1, round(0.004 * sample_rate)))
    release = min(sample_count, max(1, round(0.035 * sample_rate)))
    envelope = np.ones(sample_count, dtype=np.float64)
    envelope[:attack] *= np.linspace(0.0, 1.0, attack, endpoint=True)
    envelope[-release:] *= np.linspace(1.0, 0.0, release, endpoint=True)
    tone *= envelope
    peak = np.max(np.abs(tone))
    if peak > 0:
        tone *= 0.24 / peak
    return tone.astype(np.float32)


def render_piano(notes, duration, sample_rate=SAMPLE_RATE):
    output = np.zeros(max(1, round(duration * sample_rate)), dtype=np.float32)
    for note in notes:
        if note["rest"] or note["midi"] is None:
            continue
        start = max(0, min(output.size, round(note["start"] * sample_rate)))
        end = max(start, min(output.size, round(note["end"] * sample_rate)))
        output[start:end] = piano_note(
            note["midi"], end - start, note["note_id"], sample_rate
        )
    return output


def write_ogg(path, audio, sample_rate=SAMPLE_RATE):
    with sf.SoundFile(
        path,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        format="OGG",
        subtype="VORBIS",
    ) as handle:
        for start in range(0, audio.size, 65536):
            handle.write(audio[start : start + 65536])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack_dir", required=True)
    parser.add_argument(
        "--ids",
        help="Optional comma-separated sample IDs. Defaults to every report entry.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir)
    report = json.loads((pack_dir / "report_selected.json").read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in report["entries"]}
    sample_ids = list(entries)
    if args.ids:
        sample_ids = [sample_id.strip() for sample_id in args.ids.split(",")]
        unknown = [sample_id for sample_id in sample_ids if sample_id not in entries]
        if unknown:
            raise ValueError(f"Unknown sample IDs: {unknown}")

    rendered = []
    for index, sample_id in enumerate(sample_ids, start=1):
        sample_dir = pack_dir / sample_id
        duration = float(entries[sample_id]["decoded_duration"])
        for timebase in ("native", "model"):
            output_path = sample_dir / f"p_{timebase}_piano.ogg"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite: {output_path}")
            notes = read_notes(sample_dir / f"p_{timebase}_notes.csv")
            audio = render_piano(notes, duration)
            write_ogg(output_path, audio)
        rendered.append(sample_id)
        print(f"[{index}/{len(sample_ids)}] {sample_id}")

    manifest = {
        "schema": "v4pf_some_p_timbre_render_v1",
        "instrument": "procedural_piano",
        "sample_rate": SAMPLE_RATE,
        "source": "existing quantized P note CSV; SOME was not rerun",
        "sample_ids": rendered,
        "files_per_sample": ["p_native_piano.ogg", "p_model_piano.ogg"],
    }
    manifest_path = pack_dir / "piano_render.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
