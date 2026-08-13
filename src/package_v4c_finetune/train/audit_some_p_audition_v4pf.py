import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


NON_HANAMARU_IDS = [f"N{index:03d}" for index in range(1, 43)]
HANAMARU_IDS = [
    "H001",
    "H004",
    "H005",
    "H007",
    "H008",
    "H009",
    "H011",
    "H013",
    "H014",
    "H016",
    "H017",
    "H018",
    "H019",
    "H021",
    "H022",
    "H025",
    "H027",
    "H028",
    "H029",
    "H030",
]
WAV_NAMES = ["original_mono.wav", "p_native_sine.wav", "p_model_sine.wav"]
REQUIRED_NAMES = WAV_NAMES + [
    "p_native_notes.csv",
    "p_model_notes.csv",
    "metadata.json",
]


def audit_wav(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 44100:
        raise AssertionError(f"{path}: sample rate {sample_rate}")
    if audio.shape[0] == 0 or audio.shape[1] != 1:
        raise AssertionError(f"{path}: shape {audio.shape}")
    if not np.isfinite(audio).all():
        raise AssertionError(f"{path}: non-finite samples")
    return audio.shape[0], float(np.max(np.abs(audio)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pack_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pack_dir = Path(args.pack_dir)
    if pack_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pack directory: {pack_dir}")

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    entries = report["entries"]
    by_id = {entry["id"]: entry for entry in entries}
    if len(entries) != 72 or len(by_id) != 72:
        raise AssertionError(f"Expected 72 unique entries, got {len(entries)}/{len(by_id)}")
    if report["some_mel"]["fmin"] != 40.0 or report["some_mel"]["fmax"] != 8000.0:
        raise AssertionError(f"Wrong SOME Mel frontend: {report['some_mel']}")

    for entry in entries:
        sample_dir = output_dir / entry["id"]
        missing = [name for name in REQUIRED_NAMES if not (sample_dir / name).is_file()]
        if missing:
            raise AssertionError(f"{entry['id']}: missing {missing}")
        wav_stats = [audit_wav(sample_dir / name) for name in WAV_NAMES]
        frame_counts = [stats[0] for stats in wav_stats]
        if len(set(frame_counts)) != 1:
            raise AssertionError(f"{entry['id']}: unequal WAV lengths {frame_counts}")
        if wav_stats[1][1] <= 1e-6 or wav_stats[2][1] <= 1e-6:
            raise AssertionError(f"{entry['id']}: silent synthesized track")
        if entry["native"]["pad_frames"] or entry["model"]["pad_frames"]:
            raise AssertionError(f"{entry['id']}: unexpected PAD frames")

    selected_ids = NON_HANAMARU_IDS + HANAMARU_IDS
    selected_entries = [by_id[sample_id] for sample_id in selected_ids]
    bucket_counts = {
        bucket: sum(entry["bucket"] == bucket for entry in selected_entries)
        for bucket in ("0-10", "10-20", "20-30")
    }
    if bucket_counts != {"0-10": 5, "10-20": 8, "20-30": 7}:
        raise AssertionError(f"Wrong Hanamaru duration counts: {bucket_counts}")

    pack_dir.mkdir(parents=True)
    for sample_id in selected_ids:
        shutil.copytree(output_dir / sample_id, pack_dir / sample_id)

    selection = {
        "schema": "v4pf_some_p_audition_selection_v3",
        "some_mel": {"fmin": 40.0, "fmax": 8000.0},
        "render_fix": "split when either note_id or quantized class changes",
        "non_hanamaru_ids": NON_HANAMARU_IDS,
        "hanamaru_ids": HANAMARU_IDS,
        "hanamaru_counts": bucket_counts,
        "listen_order": WAV_NAMES,
    }
    selected_report = {
        "schema": "v4pf_some_p_audition_selected_report_v3",
        "midi_p_schema": report["midi_p_schema"],
        "some_mel": report["some_mel"],
        "manifest": report["manifest"],
        "entries": selected_entries,
    }
    (pack_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (pack_dir / "report_selected.json").write_text(
        json.dumps(selected_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audited_candidates": len(entries),
                "selected_entries": len(selected_entries),
                "wav_files_audited": len(entries) * len(WAV_NAMES),
                "selected_wav_files": len(selected_entries) * len(WAV_NAMES),
                "missing_files": 0,
                "pad_frames": 0,
                "hanamaru_counts": bucket_counts,
                "some_mel": report["some_mel"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
