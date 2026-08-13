#!/usr/bin/env python3
"""Run V4fg training-style sentence inference on a train-derived A/B pool."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys
import time


def safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]', "_", str(value)).strip().rstrip(".")
    return value or "unnamed"


def target_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        str(item.get("kana") or item.get("text") or "").strip()
        for item in payload.get("phrases", [])
    ]
    result = " ".join(line for line in lines if line)
    if not result:
        raise ValueError(f"no target text in {path}")
    return result


def valid_wav(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 44


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--singer-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae-ckpt", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    groups = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))["groups"]
    if args.limit is not None:
        groups = groups[: args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    singer_root = args.singer_root.resolve()
    os.chdir(singer_root)
    sys.path.insert(0, str(singer_root))

    import torch
    import torchaudio
    import infer_v4_formal as infer

    started = time.time()
    print(json.dumps({"event": "loading", "groups": len(groups)}), flush=True)
    policy, vae, midi_teacher, mel_extract, tokenizer = infer.build_model(
        checkpoint=str(args.checkpoint.resolve()),
        vae_ckpt=str(args.vae_ckpt.resolve()),
        device=args.device,
    )
    print(json.dumps({"event": "loaded", "seconds": time.time() - started}), flush=True)

    completed = 0
    skipped = 0
    failures = []
    for index, group in enumerate(groups, start=1):
        group_dir = dataset / group["directory"]
        output_path = output_dir / f"{safe_name(group['name'])}.wav"
        if args.resume and valid_wav(output_path):
            skipped += 1
            continue
        item_started = time.time()
        try:
            wav, sample_rate = infer.synthesize(
                policy,
                vae,
                midi_teacher,
                mel_extract,
                tokenizer,
                ref_audio=str(group_dir / "A.wav"),
                melody_audio=str(group_dir / "B.wav"),
                target_text=target_text(group_dir / "B_T1.json"),
                steps=args.steps,
                cfg_strength=args.cfg,
                seed=args.seed,
                device=args.device,
            )
            torchaudio.save(str(output_path), wav.unsqueeze(0), sample_rate)
            duration = wav.shape[0] / sample_rate
            del wav
            torch.cuda.empty_cache()
            completed += 1
            elapsed = time.time() - started
            rate = elapsed / max(completed, 1)
            print(
                json.dumps(
                    {
                        "event": "done",
                        "index": index,
                        "total": len(groups),
                        "name": group["name"],
                        "duration": duration,
                        "itemSeconds": time.time() - item_started,
                        "elapsedSeconds": elapsed,
                        "etaSeconds": rate * (len(groups) - index),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as error:
            failures.append({"name": group["name"], "error": str(error)})
            print(json.dumps({"event": "failed", **failures[-1]}, ensure_ascii=False), flush=True)

    report = {
        "schema": "noise-scorer-v4fg-server-selfclone.v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "vae": str(args.vae_ckpt.resolve()),
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "seconds": time.time() - started,
    }
    (output_dir / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del policy, vae, midi_teacher, mel_extract, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    if failures:
        raise RuntimeError(f"{len(failures)} V4fg samples failed")
    print(json.dumps({"event": "complete", **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
