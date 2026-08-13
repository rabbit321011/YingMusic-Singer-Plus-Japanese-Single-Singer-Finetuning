"""Run a real H/GAME-P audio smoke from an M600-B publish checkpoint."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import soundfile as sf
import torch
from omegaconf import OmegaConf


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
TRAIN_DIR = PACKAGE_DIR / "train"
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src").is_dir():
    YING_REPO = PROJECT_DIR
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(YING_REPO))

import train_v4ph as base  # noqa: E402
from prepare_v4m_m600b_transition import (  # noqa: E402
    TARGET_DEPTH,
    TARGET_DIT_PARAMETERS,
    TARGET_FF_MULT,
    TARGET_HEADS,
    build_singer,
    sha256_file,
    strict_load,
    transformer_parameter_count,
)
from src.YingMusicSinger.melody.game_cache_v4ph import (  # noqa: E402
    game_cache_to_model_tracks,
)
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import (  # noqa: E402
    StableAudioInfer,
)


PUBLISH_SCHEMA = "v4m_m600b_publish_checkpoint_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--train_manifest_sha256", required=True)
    parser.add_argument("--h_config_fingerprint", required=True)
    parser.add_argument("--game_cache_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml"
    )
    parser.add_argument(
        "--vae_config",
        default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",
    )
    parser.add_argument(
        "--vae_ckpt", default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt"
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_duration", type=float, default=12.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    base.seed_everything(args.seed)
    started = time.perf_counter()

    actual_sha = sha256_file(args.checkpoint)
    if actual_sha != args.checkpoint_sha256:
        raise ValueError("M600-B publish SHA256 mismatch")
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if payload.get("checkpoint_schema") != PUBLISH_SCHEMA:
        raise ValueError("M600-B publish schema mismatch")
    state = payload.get("ema_model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("M600-B publish checkpoint lacks EMA state")
    if transformer_parameter_count(state) != TARGET_DIT_PARAMETERS:
        raise ValueError("M600-B publish parameter count mismatch")
    checkpoint_step = int(payload["global_step"])

    cfg = OmegaConf.load(args.config)
    policy = build_singer(
        cfg, TARGET_DEPTH, TARGET_HEADS, TARGET_FF_MULT, args.seed
    )
    strict_load(policy, state, "M600-B published EMA")
    del payload, state
    policy = policy.to(device).eval()
    policy.smoothMelody_MIDIFuzzDisturb = torch.nn.Identity()
    vae = (
        StableAudioInfer(
            model_config_path=args.vae_config,
            model_ckpt_path=args.vae_ckpt,
        )
        .to(device)
        .eval()
    )

    dataset = base.HDataset(
        args.train_manifest,
        args.train_manifest_sha256,
        args.h_config_fingerprint,
        max_duration_sec=30.0,
        game_cache_manifest=args.game_cache_manifest,
    )
    index = min(
        range(len(dataset.records)),
        key=lambda item: abs(
            float(dataset.records[item]["Duration"]) - args.target_duration
        ),
    )
    batch = base.collate_svs([dataset[index]])
    wav = batch["wav"][0].to(device)
    sample_rate = int(batch["sr"][0])
    wav_2d = wav.unsqueeze(0) if wav.dim() == 1 else wav
    with torch.inference_mode():
        latent = vae.encode_audio(wav_2d, in_sr=sample_rate)
    full_latent = latent.squeeze(0).transpose(0, 1).unsqueeze(0)
    total_frames = int(full_latent.shape[1])
    ref_len = min(
        int(total_frames * 0.65),
        max(int(5.0 * base.FRAME_RATE), int(total_frames * 0.4)),
    )
    if total_frames - ref_len < int(base.FRAME_RATE):
        raise ValueError("audio smoke target region is shorter than one second")

    tracks = game_cache_to_model_tracks(
        batch["game_cache"][0],
        num_samples=wav_2d.shape[-1],
        target_len=total_frames,
        sample_rate=sample_rate,
    )
    p_classes = tracks["p_classes"].long().unsqueeze(0).to(device)
    with torch.inference_mode():
        midi_full = policy.midi_p_v4ph(p_classes)
    midi = torch.cat(
        [torch.zeros_like(midi_full[:, :ref_len]), midi_full[:, ref_len:]], dim=1
    )
    if torch.count_nonzero(midi[:, :ref_len]) != 0:
        raise AssertionError("M600-B audio smoke prompt MIDI is nonzero")
    if not torch.equal(policy.smoothMelody_MIDIFuzzDisturb(midi), midi):
        raise AssertionError("M600-B audio smoke MIDI transport changed the tensor")

    rendered = base.render_h_pul_placements(
        batch["phrases"][0],
        batch["h_candidates"][0],
        ref_len=ref_len,
        total_frames=total_frames,
        sep_token_id=base.SEP_TOKEN,
        pul_token_id=base.PUL_TOKEN,
    )
    dense_text = rendered["phone_pul"]["text"]
    if len(dense_text) != total_frames:
        raise AssertionError("M600-B audio smoke placement length mismatch")
    text = torch.tensor(dense_text, dtype=torch.long, device=device).unsqueeze(0)
    bound_transport = torch.zeros(1, total_frames, 1, dtype=midi.dtype, device=device)

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        generated, _ = policy.sample(
            cond=full_latent[:, :ref_len],
            text=text,
            duration=total_frames,
            midi_p=midi,
            bound_p=bound_transport,
            steps=args.steps,
            cfg_strength=args.cfg,
            guidance_scale=args.cfg,
            t_shift=0.5,
            seed=args.seed,
            use_epss=False,
            enable_melody_control=True,
        )
        target_latent = generated[:, ref_len:, :].permute(0, 2, 1).float()
        output_audio = vae.decode_audio(target_latent).squeeze(0)[:1].reshape(-1)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    if not torch.isfinite(output_audio).all():
        raise FloatingPointError("M600-B audio smoke produced non-finite audio")
    rms = float(output_audio.square().mean().sqrt())
    peak = float(output_audio.abs().max())
    if rms <= 1e-5 or peak <= 1e-4:
        raise ValueError(f"M600-B audio smoke is silent: rms={rms} peak={peak}")

    output = pathlib.Path(args.output).resolve()
    report_path = pathlib.Path(args.report).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output),
        output_audio.detach().cpu().numpy(),
        44100,
        subtype="PCM_16",
    )
    report = {
        "schema": "v4m_m600b_audio_smoke_v1",
        "status": "ok",
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "checkpoint_sha256": actual_sha,
        "checkpoint_step": checkpoint_step,
        "sample_id": batch["sample_id"][0],
        "source_duration": float(batch["duration"][0]),
        "total_frames": total_frames,
        "ref_frames": ref_len,
        "target_frames": total_frames - ref_len,
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "output": str(output),
        "output_samples": int(output_audio.numel()),
        "output_seconds": output_audio.numel() / 44100.0,
        "rms": rms,
        "peak": peak,
        "inference_seconds": inference_seconds,
        "process_seconds": time.perf_counter() - started,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "max_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "phone_phrases": rendered["phone_phrase_count"],
        "pul_phrases": rendered["pul_phrase_count"],
        "exact_control_phrases": rendered["exact_control_phrase_count"],
        "prompt_midi_nonzero": int(torch.count_nonzero(midi[:, :ref_len])),
        "pad_classes": int((p_classes == 256).sum()),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
