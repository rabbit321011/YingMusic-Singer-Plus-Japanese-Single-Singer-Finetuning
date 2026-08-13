"""Full-timeline V4IjPH style conditioning and ODE sampling runtime."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch


INFER_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = INFER_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
TRAIN_DIR = PACKAGE_DIR / "train"
for candidate in (PROJECT_DIR, PACKAGE_DIR, YING_REPO, TRAIN_DIR):
    value = os.fspath(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from v4ijph_contract import (  # noqa: E402
    InsStyleAdapter,
    V4IJPH_CHECKPOINT_SCHEMA,
    build_style_cond,
    strict_load_module,
)
from v4ijph_ins_cache import (  # noqa: E402
    INTRINSIC_CHECKPOINT_NAME,
    ParaSpeechClapIntrinsic,
    load_reference_waveform,
    sha256_file,
    verify_inventory,
)


def load_inference_adapter(
    checkpoint_path: str | pathlib.Path,
    *,
    use_ema: bool,
    device: str | torch.device,
) -> tuple[InsStyleAdapter, dict]:
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if payload.get("checkpoint_schema") != V4IJPH_CHECKPOINT_SCHEMA:
        raise ValueError("Inference checkpoint is not V4IjPH v2")
    metadata = payload.get("v4ijph_training")
    if not isinstance(metadata, dict) or (metadata.get("ins") or {}).get(
        "style_guidance"
    ) != 1.0:
        raise ValueError("V4IjPH inference metadata is missing or invalid")
    adapter = InsStyleAdapter()
    key = "adapter_ema_model_state_dict" if use_ema else "adapter_state_dict"
    strict_load_module(adapter, payload[key], "inference adapter")
    adapter = adapter.to(device).eval()
    provenance = {
        "checkpoint": str(pathlib.Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_schema": V4IJPH_CHECKPOINT_SCHEMA,
        "checkpoint_step": int(payload["global_step"]),
        "adapter_source": "ema" if use_ema else "raw",
        "style_guidance": 1.0,
    }
    return adapter, provenance


def encode_reference(
    reference_audio: str | pathlib.Path,
    *,
    source_dir: pathlib.Path,
    source_inventory: pathlib.Path,
    speech_model_dir: pathlib.Path,
    speech_model_inventory: pathlib.Path,
    text_model_dir: pathlib.Path,
    text_model_inventory: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    checkpoint_inventory: pathlib.Path,
    device: str,
) -> tuple[torch.Tensor, dict]:
    inventories = {
        "source": verify_inventory(
            source_inventory, source_dir, verify_files=True
        ),
        "speech_model": verify_inventory(
            speech_model_inventory, speech_model_dir, verify_files=True
        ),
        "text_model": verify_inventory(
            text_model_inventory, text_model_dir, verify_files=True
        ),
        "checkpoint": verify_inventory(
            checkpoint_inventory, checkpoint_dir, verify_files=True
        ),
    }
    waveform, provenance = load_reference_waveform(reference_audio)
    encoder = ParaSpeechClapIntrinsic(
        source_dir=source_dir,
        speech_model_dir=speech_model_dir,
        text_model_dir=text_model_dir,
        checkpoint=checkpoint_dir / INTRINSIC_CHECKPOINT_NAME,
        device=device,
    )
    vector = torch.from_numpy(encoder.encode(waveform)).float().unsqueeze(0)
    provenance.update(
        {
            "inventories": inventories,
            "embedding_shape": list(vector.shape),
            "embedding_l2": float(vector.norm(dim=1).item()),
        }
    )
    return vector, provenance


def full_timeline_cond(
    adapter: InsStyleAdapter,
    ins: torch.Tensor,
    frames: int,
    *,
    null_ins: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    ins = ins.to(next(adapter.parameters()).device, dtype=torch.float32)
    return build_style_cond(adapter, ins, frames, drop_ins=null_ins)


@torch.inference_mode()
def sample_full_timeline_style(
    policy,
    *,
    style_cond: torch.Tensor,
    text: torch.Tensor,
    midi: torch.Tensor,
    duration: int,
    steps: int,
    cfg_strength: float,
    seed: int,
    t_shift: float = 0.5,
    _odeint=None,
):
    """Sample with ref_len=0 while keeping style present in both CFG branches."""
    duration = int(duration)
    if duration <= 0 or steps <= 0:
        raise ValueError("duration and steps must be positive")
    if tuple(style_cond.shape) != (1, duration, 64):
        raise ValueError("style_cond must have shape [1,duration,64]")
    if tuple(text.shape) != (1, duration):
        raise ValueError("text must have shape [1,duration]")
    if tuple(midi.shape) != (1, duration, 128):
        raise ValueError("midi must have shape [1,duration,128]")
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype
    style_cond = style_cond.to(device=device, dtype=dtype)
    text = text.to(device=device, dtype=torch.long)
    midi = midi.to(device=device, dtype=dtype)
    transformer = policy.transformer

    def velocity(t, x):
        if cfg_strength < 1e-8:
            prediction, _ = transformer(
                x=x,
                cond=style_cond,
                text=text,
                midi=midi,
                time=t,
                mask=None,
                drop_audio_cond=False,
                drop_text=False,
                drop_midi=False,
                cache=False,
            )
            return prediction
        packed, _ = transformer(
            x=x,
            cond=style_cond,
            text=text,
            midi=midi,
            time=t,
            mask=None,
            cfg_infer=True,
            cache=False,
            cfg_infer_ids=(True, False, True, False),
        )
        conditional, content_unconditional = torch.chunk(packed, 2, dim=0)
        return conditional + (
            conditional - content_unconditional
        ) * float(cfg_strength)

    torch.manual_seed(int(seed))
    noise = torch.randn(1, duration, policy.num_channels, device=device, dtype=dtype)
    timeline = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    timeline = t_shift * timeline / (1 + (t_shift - 1) * timeline)
    if _odeint is None:
        from torchdiffeq import odeint as _odeint

    trajectory = _odeint(velocity, noise, timeline, **policy.odeint_kwargs)
    transformer.clear_cache()
    return trajectory[-1], trajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an auditable V4IjPH full-timeline style condition"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--source-inventory", type=pathlib.Path, required=True)
    parser.add_argument("--speech-model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--speech-model-inventory", type=pathlib.Path, required=True)
    parser.add_argument("--text-model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--text-model-inventory", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-inventory", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw-adapter", action="store_true")
    parser.add_argument("--null-ins", action="store_true")
    args = parser.parse_args()

    adapter, checkpoint_provenance = load_inference_adapter(
        args.checkpoint, use_ema=not args.raw_adapter, device=args.device
    )
    ins, reference_provenance = encode_reference(
        args.reference_audio,
        source_dir=args.source_dir,
        source_inventory=args.source_inventory,
        speech_model_dir=args.speech_model_dir,
        speech_model_inventory=args.speech_model_inventory,
        text_model_dir=args.text_model_dir,
        text_model_inventory=args.text_model_inventory,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_inventory=args.checkpoint_inventory,
        device=args.device,
    )
    cond, style = full_timeline_cond(
        adapter, ins, args.frames, null_ins=args.null_ins
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "v4ijph_inference_condition_v1",
            "cond": cond.detach().cpu(),
            "style": style.detach().cpu(),
            "null_ins": bool(args.null_ins),
            "checkpoint": checkpoint_provenance,
            "reference": reference_provenance,
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "frames": args.frames,
                "style_l2": float(style.norm(dim=1).item()),
                "null_ins": bool(args.null_ins),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
