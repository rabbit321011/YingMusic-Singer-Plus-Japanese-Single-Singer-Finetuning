import argparse
import json
import os

import soundfile as sf


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def compare_records(control, remapped, expected_audio_dir):
    errors = []
    duration_diffs = []
    if len(control) != len(remapped):
        errors.append(f"record count differs: {len(control)} != {len(remapped)}")
    for index, (source, target) in enumerate(zip(control, remapped)):
        expected_path = os.path.join(expected_audio_dir, os.path.basename(source["Path"]))
        if target.get("Path") != expected_path:
            errors.append(f"row {index}: unexpected Path")
            continue
        source_rest = {key: value for key, value in source.items() if key != "Path"}
        target_rest = {key: value for key, value in target.items() if key != "Path"}
        if source_rest != target_rest:
            errors.append(f"row {index}: non-Path fields changed")
        if not os.path.isfile(expected_path):
            errors.append(f"row {index}: missing audio")
            continue
        info = sf.info(expected_path)
        duration_diff = abs(info.frames / info.samplerate - float(source["Duration"]))
        duration_diffs.append(duration_diff)
        if info.samplerate != 44100 or info.channels != 1 or info.subtype != "PCM_16":
            errors.append(f"row {index}: invalid audio format")
        if duration_diff > 0.05:
            errors.append(f"row {index}: duration differs by {duration_diff:.6f}s")
    return errors, duration_diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-token-dir", required=True)
    parser.add_argument("--train-token-dir", required=True)
    parser.add_argument("--train-audio-dir", required=True)
    parser.add_argument("--eval-svc-token-dir", required=True)
    parser.add_argument("--eval-svc-audio-dir", required=True)
    args = parser.parse_args()

    control_train = load(os.path.join(args.control_token_dir, "train_tokens.json"))
    control_test = load(os.path.join(args.control_token_dir, "test_tokens.json"))
    train_s = load(os.path.join(args.train_token_dir, "train_tokens.json"))
    test_s = load(os.path.join(args.eval_svc_token_dir, "test_tokens.json"))

    train_errors, train_diffs = compare_records(control_train, train_s, args.train_audio_dir)
    eval_errors, eval_diffs = compare_records(control_test, test_s, args.eval_svc_audio_dir)
    expected_eval_names = {os.path.basename(row["Path"]) for row in control_test}
    actual_eval_names = {
        name for name in os.listdir(args.eval_svc_audio_dir) if name.lower().endswith(".wav")
    }
    extra_eval = sorted(actual_eval_names - expected_eval_names)
    missing_eval = sorted(expected_eval_names - actual_eval_names)
    result = {
        "train_records": len(train_s),
        "eval_svc_records": len(test_s),
        "train_errors": len(train_errors),
        "eval_svc_errors": len(eval_errors),
        "eval_svc_missing": len(missing_eval),
        "eval_svc_extra": len(extra_eval),
        "train_max_duration_diff_ms": max(train_diffs, default=0.0) * 1000,
        "eval_svc_max_duration_diff_ms": max(eval_diffs, default=0.0) * 1000,
        "errors_preview": (train_errors + eval_errors)[:10],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if train_errors or eval_errors or extra_eval or missing_eval:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
