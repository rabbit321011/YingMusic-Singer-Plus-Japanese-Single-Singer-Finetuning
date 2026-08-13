import argparse
import json
import os
from pathlib import Path


def remap(records, audio_dir):
    mapped = []
    missing = []
    for record in records:
        target = os.path.join(audio_dir, os.path.basename(record["Path"]))
        if not os.path.isfile(target):
            missing.append(target)
            continue
        item = dict(record)
        item["Path"] = target
        mapped.append(item)
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"missing {len(missing)} remapped audio files:\n{preview}")
    return mapped


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"expected a non-empty JSON list: {path}")
    return data


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-token-dir", required=True)
    parser.add_argument("--train-audio-dir", required=True)
    parser.add_argument("--eval-svc-audio-dir", required=True)
    parser.add_argument("--train-output-dir", required=True)
    parser.add_argument("--eval-svc-output-dir", required=True)
    args = parser.parse_args()

    train = load_json(os.path.join(args.control_token_dir, "train_tokens.json"))
    test = load_json(os.path.join(args.control_token_dir, "test_tokens.json"))
    train_s = remap(train, args.train_audio_dir)
    test_s = remap(test, args.eval_svc_audio_dir)

    write_json(os.path.join(args.train_output_dir, "train_tokens.json"), train_s)
    write_json(os.path.join(args.eval_svc_output_dir, "test_tokens.json"), test_s)
    print(json.dumps({"train": len(train_s), "eval_svc": len(test_s)}, indent=2))


if __name__ == "__main__":
    main()
