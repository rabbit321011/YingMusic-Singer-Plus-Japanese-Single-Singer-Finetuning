import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def probe_duration(ffprobe, path):
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def source_key(name):
    return re.sub(r"_2_seg\d+(?:_hard)?\.wav$", "", name)


def select_evenly(records, count):
    records = sorted(records, key=lambda row: (row["duration"], row["name"]))
    low = records[0]["duration"]
    high = records[-1]["duration"]
    used_sources = set()
    selected = []
    for index in range(count):
        target = low + (index + 0.5) / count * (high - low)
        available = [row for row in records if row["source"] not in used_sources]
        picked = min(
            available,
            key=lambda row: (
                abs(row["duration"] - target),
                not row["hard"],
                row["name"],
            ),
        )
        selected.append(picked)
        used_sources.add(picked["source"])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external_dir", required=True)
    parser.add_argument("--hanamaru_token_json", required=True)
    parser.add_argument("--hanamaru_audio_dir", required=True)
    parser.add_argument("--staging_dir", required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    external_dir = Path(args.external_dir)
    hanamaru_audio_dir = Path(args.hanamaru_audio_dir)
    staging_dir = Path(args.staging_dir)
    input_dir = staging_dir / "inputs"
    if staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite staging directory: {staging_dir}")
    input_dir.mkdir(parents=True)

    entries = []
    external_paths = sorted(
        path for path in external_dir.iterdir() if path.suffix.lower() in {".wav", ".mp3", ".flac"}
    )
    for index, path in enumerate(external_paths, start=1):
        sample_id = f"N{index:03d}"
        destination = input_dir / f"{sample_id}{path.suffix.lower()}"
        shutil.copy2(path, destination)
        entries.append(
            {
                "id": sample_id,
                "group": "non_hanamaru",
                "bucket": "external",
                "source_name": path.name,
                "input": destination.relative_to(staging_dir).as_posix(),
                "duration": probe_duration(args.ffprobe, path),
            }
        )

    with open(args.hanamaru_token_json, encoding="utf-8") as handle:
        token_records = json.load(handle)
    rows = []
    for item in token_records:
        name = os.path.basename(item["Path"])
        path = hanamaru_audio_dir / name
        if not path.is_file():
            continue
        duration = float(item["Duration"])
        if duration < 10:
            bucket = "0-10"
        elif duration < 20:
            bucket = "10-20"
        elif duration <= 30:
            bucket = "20-30"
        else:
            continue
        rows.append(
            {
                "name": name,
                "path": path,
                "duration": duration,
                "bucket": bucket,
                "hard": name.endswith("_hard.wav"),
                "source": source_key(name),
            }
        )

    candidate_counts = {"0-10": 8, "10-20": 12, "20-30": 10}
    hanamaru_candidates = []
    for bucket, count in candidate_counts.items():
        pool = [row for row in rows if row["bucket"] == bucket]
        hanamaru_candidates.extend(select_evenly(pool, count))

    for index, row in enumerate(hanamaru_candidates, start=1):
        sample_id = f"H{index:03d}"
        destination = input_dir / f"{sample_id}.wav"
        shutil.copy2(row["path"], destination)
        entries.append(
            {
                "id": sample_id,
                "group": "hanamaru_candidate",
                "bucket": row["bucket"],
                "source_name": row["name"],
                "input": destination.relative_to(staging_dir).as_posix(),
                "duration": row["duration"],
                "hard": row["hard"],
            }
        )

    manifest = {
        "schema": "v4pf_some_p_audition_inputs_v1",
        "external_count": len(external_paths),
        "hanamaru_candidate_counts": candidate_counts,
        "entries": entries,
    }
    with open(staging_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
