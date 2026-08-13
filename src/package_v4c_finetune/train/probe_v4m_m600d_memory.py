"""Profile one isolated M600-D update with optional activation checkpointing."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from ema_pytorch import EMA
from omegaconf import OmegaConf


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parent
YING_REPO = PROJECT_DIR / "YingMusic-Singer-Plus-src"
if not (YING_REPO / "src").is_dir():
    YING_REPO = PROJECT_DIR
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(YING_REPO))

import train_v4ph as base  # noqa: E402
from prepare_v4m_m600d_transition import (  # noqa: E402
    TARGET_DEPTH,
    TRANSITION_SCHEMA,
    build_singer,
    sha256_file,
    strict_load,
)
from src.YingMusicSinger.melody.game_cache_v4ph import (  # noqa: E402
    game_cache_to_model_tracks,
)
from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import (  # noqa: E402
    StableAudioInfer,
)


def memory_event(stage: str, **extra):
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "stage": stage,
        "time": time.time(),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
        **extra,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", required=True)
    parser.add_argument("--transition_sha256", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--train_manifest_sha256", required=True)
    parser.add_argument("--h_config_fingerprint", required=True)
    parser.add_argument("--game_cache_manifest", required=True)
    parser.add_argument("--config", default="src/YingMusicSinger/config/YingMusic_Singer.yaml")
    parser.add_argument(
        "--vae_config",
        default="src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json",
    )
    parser.add_argument(
        "--vae_ckpt", default="ckpts/stable_audio_2_0_vae_20hz_official.ckpt"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--empty_cache_every", type=int, default=100)
    parser.add_argument("--record_every", type=int, default=10)
    parser.add_argument(
        "--sample_mode", choices=("longest", "shuffled"), default="longest"
    )
    parser.add_argument("--checkpoint_activations", action="store_true")
    parser.add_argument("--omit_ema", action="store_true")
    parser.add_argument("--omit_optimizer", action="store_true")
    args = parser.parse_args()

    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    events = []
    selected = []
    started = time.perf_counter()
    status = "failed"
    error = None

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        base.seed_everything(args.seed)
        actual_sha = sha256_file(args.transition)
        if actual_sha != args.transition_sha256:
            raise ValueError("M600-D transition SHA256 mismatch")
        transition = torch.load(
            args.transition, map_location="cpu", weights_only=False, mmap=True
        )
        if transition.get("checkpoint_schema") != TRANSITION_SCHEMA:
            raise ValueError("M600-D transition schema mismatch")

        cfg = OmegaConf.load(args.config)
        raw_model = build_singer(cfg, TARGET_DEPTH, args.seed)
        strict_load(raw_model, transition["model_state_dict"], "M600-D transition")
        raw_model = raw_model.to(device).train()
        vae = StableAudioInfer(
            model_config_path=args.vae_config, model_ckpt_path=args.vae_ckpt
        ).to(device).eval()
        for parameter in vae.parameters():
            parameter.requires_grad = False

        ema = None
        if not args.omit_ema:
            ema = EMA(
                raw_model,
                beta=float(cfg.ema_kwargs.beta),
                update_after_step=int(cfg.ema_kwargs.update_after_step),
                update_every=int(cfg.ema_kwargs.update_every),
            ).to(device)
        optimizer = None
        if not args.omit_optimizer:
            optimizer = torch.optim.AdamW(
                [p for p in raw_model.parameters() if p.requires_grad],
                lr=1.4e-5,
                betas=(0.9, 0.95),
                weight_decay=1e-2,
            )

        dataset = base.HDataset(
            args.train_manifest,
            args.train_manifest_sha256,
            args.h_config_fingerprint,
            max_duration_sec=30.0,
            game_cache_manifest=args.game_cache_manifest,
        )
        required_samples = args.updates * args.grad_accum
        if args.sample_mode == "longest":
            ranked = sorted(
                range(len(dataset.records)),
                key=lambda index: float(dataset.records[index]["Duration"]),
                reverse=True,
            )
            indices = [ranked[index % len(ranked)] for index in range(required_samples)]
        else:
            generator = torch.Generator().manual_seed(args.seed)
            indices = []
            while len(indices) < required_samples:
                indices.extend(torch.randperm(len(dataset), generator=generator).tolist())
            indices = indices[:required_samples]

        def compute_ref_len(total_frames, phrase_boundaries=None):
            five_sec_frames = int(5.0 * base.FRAME_RATE)
            ref_len = total_frames * random.uniform(0.125, 0.33)
            ref_len = max(ref_len, five_sec_frames)
            ref_len = min(ref_len, int(total_frames * 0.65))
            if phrase_boundaries:
                margin = int(2.5 * base.FRAME_RATE)
                candidates = [
                    boundary
                    for boundary in phrase_boundaries
                    if boundary > 0 and abs(boundary - ref_len) <= margin
                ]
                if candidates:
                    ref_len = min(candidates, key=lambda item: abs(item - ref_len))
            return int(ref_len)

        def process_batch(batch):
            with torch.no_grad():
                wav = batch["wav"][0].to(device)
                sample_rate = batch["sr"][0]
                phrases = batch["phrases"][0]
                candidates = batch["h_candidates"][0]
                game_cache = batch["game_cache"][0]
                wav_2d = wav.unsqueeze(0) if wav.dim() == 1 else wav
                latent = vae.encode_audio(wav_2d, in_sr=sample_rate)
                full_latent = latent.squeeze(0).transpose(0, 1).unsqueeze(0)
                total_frames = full_latent.shape[1]
                boundaries = [
                    int(phrase["start"] * base.FRAME_RATE) for phrase in phrases
                ]
                ref_len = compute_ref_len(total_frames, boundaries)
                cond = torch.zeros_like(full_latent)
                cond[:, :ref_len] = full_latent[:, :ref_len]
                tracks = game_cache_to_model_tracks(
                    game_cache,
                    num_samples=wav_2d.shape[-1],
                    target_len=total_frames,
                    sample_rate=sample_rate,
                )
                p_classes = tracks["p_classes"].unsqueeze(0).to(device)
                midi_probs = tracks["cka_probs"].unsqueeze(0).to(device)
                with torch.enable_grad():
                    midi_full = raw_model.midi_p_v4ph(p_classes)
                    midi = torch.cat(
                        [
                            torch.zeros_like(midi_full[:, :ref_len]),
                            midi_full[:, ref_len:],
                        ],
                        dim=1,
                    )
                paired = base.render_h_pul_placements(
                    phrases,
                    candidates,
                    ref_len=ref_len,
                    total_frames=total_frames,
                    sep_token_id=base.SEP_TOKEN,
                    pul_token_id=base.PUL_TOKEN,
                )
                aligned_text = torch.tensor(
                    paired["phone_pul"]["text"], dtype=torch.long, device=device
                ).unsqueeze(0)
            return full_latent, cond, midi, midi_probs, aligned_text, ref_len

        def run_dit(x_t, cond, text, diffusion_time, midi, drops):
            dit = raw_model.transformer
            time_embedding = dit.time_embed(diffusion_time)
            hidden, _ = dit.get_input_embed(
                x_t,
                cond,
                text,
                midi,
                drop_audio_cond=drops[0],
                drop_text=drops[1],
                drop_midi=drops[2],
                cache=False,
            )
            rope = dit.rotary_embed.forward_from_seq_len(x_t.shape[1])
            residual = hidden
            hidden_states = []
            for index, block in enumerate(dit.transformer_blocks):
                if args.checkpoint_activations:
                    def block_forward(value, embedding, module=block):
                        return module(value, embedding, mask=None, rope=rope)

                    hidden = torch.utils.checkpoint.checkpoint(
                        block_forward,
                        hidden,
                        time_embedding,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                else:
                    hidden = block(hidden, time_embedding, mask=None, rope=rope)
                if index >= len(dit.transformer_blocks) - 3:
                    hidden_states.append(hidden)
            hidden = dit.long_skip_connection(torch.cat((hidden, residual), dim=-1))
            hidden = dit.norm_out(hidden, time_embedding)
            return dit.proj_out(hidden), hidden_states

        def loss_for_batch(batch):
            full_latent, cond, midi, midi_probs, text, ref_len = process_batch(batch)
            u = torch.rand(1, device=device)
            diffusion_time = 0.5 * u / (1 - 0.5 * u)
            noise = torch.randn_like(full_latent)
            x_t = (
                (1 - diffusion_time[:, None, None]) * noise
                + diffusion_time[:, None, None] * full_latent
            )
            target = full_latent - noise
            drops = (
                random.random() < 0.3,
                random.random() < 0.15,
                random.random() < 0.3,
            )
            prediction, hidden_states = run_dit(
                x_t, cond, text, diffusion_time, midi, drops
            )
            flow_a = F.mse_loss(prediction[:, :ref_len], target[:, :ref_len])
            flow_b = F.mse_loss(prediction[:, ref_len:], target[:, ref_len:])
            cka = base.compute_cka_loss_from_hidden(
                [hidden[:, ref_len:] for hidden in hidden_states],
                midi_probs[:, ref_len:],
            )
            return flow_a + 2.0 * flow_b + 0.7 * cka

        torch.cuda.reset_peak_memory_stats()
        events.append(memory_event("initialized"))
        sample_cursor = 0
        step_times = []
        duration_min = float("inf")
        duration_max = 0.0
        duration_sum = 0.0
        for update in range(1, args.updates + 1):
            update_started = time.perf_counter()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            for microbatch in range(1, args.grad_accum + 1):
                index = indices[sample_cursor]
                sample_cursor += 1
                batch = base.collate_svs([dataset[index]])
                sample_info = {
                    "sample_id": batch["sample_id"][0],
                    "duration": float(batch["duration"][0]),
                }
                duration = sample_info["duration"]
                duration_min = min(duration_min, duration)
                duration_max = max(duration_max, duration)
                duration_sum += duration
                if len(selected) < 16 or sample_cursor > required_samples - 16:
                    selected.append(
                        {"update": update, "microbatch": microbatch, **sample_info}
                    )
                micro_started = time.perf_counter()
                loss = loss_for_batch(batch) / args.grad_accum
                if args.updates == 1:
                    events.append(
                        memory_event(
                            "after_forward",
                            update=update,
                            microbatch=microbatch,
                            loss=float(loss.detach()),
                            elapsed_seconds=time.perf_counter() - micro_started,
                        )
                    )
                loss.backward()
                if args.updates == 1:
                    events.append(
                        memory_event(
                            "after_backward",
                            update=update,
                            microbatch=microbatch,
                            elapsed_seconds=time.perf_counter() - micro_started,
                        )
                    )
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            if optimizer is not None:
                optimizer.step()
            if ema is not None:
                ema.update()
            step_time = time.perf_counter() - update_started
            step_times.append(step_time)
            should_record = (
                update == 1
                or update == args.updates
                or update % args.record_every == 0
            )
            if should_record:
                events.append(
                    memory_event(
                        "after_update",
                        update=update,
                        elapsed_seconds=step_time,
                    )
                )
            if args.empty_cache_every > 0 and update % args.empty_cache_every == 0:
                events.append(memory_event("before_empty_cache", update=update))
                torch.cuda.empty_cache()
                events.append(memory_event("after_empty_cache", update=update))
        if args.updates == 1:
            events.append(memory_event("after_optimizer"))
            events.append(memory_event("after_ema"))
        status = "ok"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            events.append(memory_event("failed", error=error))
        except BaseException:
            pass
    finally:
        completed_step_times = step_times if "step_times" in locals() else []
        mean_step_time = (
            sum(completed_step_times) / len(completed_step_times)
            if completed_step_times
            else None
        )
        report = {
            "schema": "v4m_m600d_memory_probe_v1",
            "status": status,
            "error": error,
            "checkpoint_activations": args.checkpoint_activations,
            "ema_included": not args.omit_ema,
            "optimizer_included": not args.omit_optimizer,
            "grad_accum": args.grad_accum,
            "updates": args.updates,
            "empty_cache_every": args.empty_cache_every,
            "record_every": args.record_every,
            "sample_mode": args.sample_mode,
            "selected": selected,
            "duration_summary": {
                "count": sample_cursor if "sample_cursor" in locals() else 0,
                "min": duration_min if "duration_min" in locals() and sample_cursor else None,
                "max": duration_max if "duration_max" in locals() and sample_cursor else None,
                "mean": (
                    duration_sum / sample_cursor
                    if "duration_sum" in locals() and sample_cursor
                    else None
                ),
            },
            "step_time_summary": {
                "count": len(completed_step_times),
                "mean": mean_step_time,
                "median": (
                    statistics.median(completed_step_times)
                    if completed_step_times
                    else None
                ),
                "p95": (
                    sorted(completed_step_times)[
                        max(0, (95 * len(completed_step_times) + 99) // 100 - 1)
                    ]
                    if completed_step_times
                    else None
                ),
                "min": min(completed_step_times) if completed_step_times else None,
                "max": max(completed_step_times) if completed_step_times else None,
                "updates_per_second": 1.0 / mean_step_time if mean_step_time else None,
                "projected_hours": (
                    {
                        str(target): target * mean_step_time / 3600.0
                        for target in (6000, 12000, 30000)
                    }
                    if mean_step_time
                    else None
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "events": events,
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if status != "ok":
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
