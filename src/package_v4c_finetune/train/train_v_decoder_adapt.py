#!/usr/bin/env python3
"""Train the V decoder on cached DiT latents paired with original B audio."""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.autoencoders import AutoencoderTrainingWrapper
from stable_audio_tools.training.losses import MultiLoss
from stable_audio_tools.training.utils import (
    create_optimizer_from_config,
    create_scheduler_from_config,
)

HOP = 2048
FRAMES = 32
SAMPLES = HOP * FRAMES
LATENT_CHANNELS = 64
SAMPLE_RATE = 44100


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_argument(value):
    value = value.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("expected a 64-character SHA256 hex digest")
    return value


def window_start(sample_id, round_index, repeat, max_start, seed):
    payload = f"{seed}:{round_index}:{sample_id}:{repeat}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (max_start + 1)


class VCacheRound(Dataset):
    def __init__(self, root, round_index, vae_sha256, singer_sha256,
                 train_manifest_sha256, seed=42, repeat=8, limit=None, sample_id=None):
        self.root = Path(root)
        self.round_index = round_index
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["vae_checkpoint_sha256"] != vae_sha256:
            raise ValueError(f"VAE mismatch in {manifest_path}")
        if manifest["singer_checkpoint_sha256"] != singer_sha256:
            raise ValueError(f"PgO mismatch in {manifest_path}")
        if manifest["train_manifest_sha256"] != [train_manifest_sha256]:
            raise ValueError(f"train manifest mismatch in {manifest_path}")
        self.manifest_sha256 = sha256_file(manifest_path)
        self.entries = manifest["entries"][:limit]
        if sample_id is not None:
            self.entries = [entry for entry in self.entries if entry["sample_id"] == sample_id]
        if not self.entries:
            raise ValueError(f"empty cache: {self.root}")
        self.repeat = repeat
        rng = np.random.default_rng(seed + round_index)
        self.order = rng.permutation(len(self.entries) * repeat).tolist()
        for entry in self.entries:
            frames = int(entry["x_b_sample_count"]) // HOP
            if frames < FRAMES or int(entry["z_gen_frames"]) < frames:
                raise ValueError(f"invalid B window: {entry['sample_id']}")
            cache = (self.root / entry["cache"]).resolve()
            if not cache.is_relative_to(self.root.resolve()) or not cache.is_file():
                raise ValueError(f"missing/outside cache: {cache}")
        self.seed = seed

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        slot = self.order[index]
        entry = self.entries[slot // self.repeat]
        repeat = slot % self.repeat
        frames = int(entry["x_b_sample_count"]) // HOP
        start = window_start(
            entry["sample_id"], self.round_index, repeat, frames - FRAMES, self.seed
        )
        with np.load(self.root / entry["cache"], allow_pickle=False) as saved:
            full_latent = saved["z_gen"]
            if full_latent.shape != (int(entry["z_gen_frames"]), LATENT_CHANNELS):
                raise ValueError(f"invalid latent shape: {entry['sample_id']}")
            latents = np.ascontiguousarray(full_latent[start:start + FRAMES].T)
        with sf.SoundFile(entry["source_path"]) as audio:
            if audio.samplerate != SAMPLE_RATE or audio.channels not in (1, 2):
                raise ValueError(f"invalid source format: {entry['source_path']}")
            audio.seek(int(entry["x_b_sample_start"]) + start * HOP)
            samples = audio.read(SAMPLES, dtype="float32", always_2d=True)
        if samples.shape[0] != SAMPLES:
            raise ValueError(f"short audio: {entry['sample_id']}")
        if samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)
        reals = np.ascontiguousarray(samples.T)
        if not np.isfinite(latents).all() or not np.isfinite(reals).all():
            raise ValueError(f"non-finite sample: {entry['sample_id']}")
        return torch.from_numpy(reals), {
            "latents": torch.from_numpy(latents),
            "sample_id": entry["sample_id"],
            "round": self.round_index,
            "start_frame": start,
        }


class VDecoderWrapper(AutoencoderTrainingWrapper):
    """Official losses and optimizer cadence, with cached latent input and no KL."""

    def __init__(self, model, config, ema_copy):
        training = config["training"]
        super().__init__(
            model,
            lr=training["learning_rate"],
            warmup_steps=training["warmup_steps"],
            encoder_freeze_on_warmup=training["encoder_freeze_on_warmup"],
            sample_rate=config["sample_rate"],
            loss_config=training["loss_configs"],
            optimizer_configs=training["optimizer_configs"],
            use_ema=training["use_ema"],
            ema_copy=ema_copy,
        )
        retained = [
            module for module in self.gen_loss_modules
            if module.name != "kl_loss"
        ]
        if len(retained) != len(self.gen_loss_modules) - 1:
            raise RuntimeError("expected exactly one implicit KL loss")
        self.gen_loss_modules = retained
        self.losses_gen = MultiLoss(retained)
        if any(module.name == "kl_loss" for module in self.losses_gen.losses):
            raise RuntimeError("KL must not be active in V")

    def configure_optimizers(self):
        if any(parameter.requires_grad for parameter in self.autoencoder.encoder.parameters()):
            raise RuntimeError("encoder must be frozen")
        decoder_parameters = [
            parameter for parameter in self.autoencoder.decoder.parameters()
            if parameter.requires_grad
        ]
        if not decoder_parameters:
            raise RuntimeError("decoder has no trainable parameters")
        gen_config = self.optimizer_configs["autoencoder"]
        disc_config = self.optimizer_configs["discriminator"]
        gen = create_optimizer_from_config(gen_config["optimizer"], decoder_parameters)
        disc = create_optimizer_from_config(
            disc_config["optimizer"], self.discriminator.parameters()
        )
        schedulers = [
            create_scheduler_from_config(gen_config["scheduler"], gen),
            create_scheduler_from_config(disc_config["scheduler"], disc),
        ]
        return [gen, disc], schedulers

    def training_step(self, batch, batch_idx):
        reals, metadata = batch
        latents = metadata["latents"]
        if reals.ndim != 3 or reals.shape[1:] != (2, SAMPLES):
            raise ValueError(f"waveform shape: {tuple(reals.shape)}")
        if latents.shape != (reals.shape[0], LATENT_CHANNELS, FRAMES):
            raise ValueError(f"latent shape: {tuple(latents.shape)}")
        if not torch.isfinite(latents).all() or not torch.isfinite(reals).all():
            raise ValueError("non-finite batch")
        self.warmed_up = True
        decoded = self.autoencoder.decode(latents)
        if decoded.shape != reals.shape:
            raise ValueError(f"decoded shape: {tuple(decoded.shape)}")
        info = {
            "reals": reals,
            "decoded": decoded,
            "reals_left": reals[:, 0:1],
            "reals_right": reals[:, 1:2],
            "decoded_left": decoded[:, 0:1],
            "decoded_right": decoded[:, 1:2],
        }
        loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals, decoded)
        info.update(
            loss_dis=loss_dis,
            loss_adv=loss_adv,
            feature_matching_distance=feature_matching_distance,
        )
        opt_gen, opt_disc = self.optimizers()
        sched_gen, sched_disc = self.lr_schedulers()
        if self.global_step % 2:
            loss, losses = self.losses_disc(info)
            opt_disc.zero_grad()
            self.manual_backward(loss)
            opt_disc.step()
            sched_disc.step()
            logs = {"train/disc_lr": opt_disc.param_groups[0]["lr"]}
        else:
            loss, losses = self.losses_gen(info)
            if self.use_ema:
                self.autoencoder_ema.update()
            opt_gen.zero_grad()
            self.manual_backward(loss)
            opt_gen.step()
            sched_gen.step()
            logs = {
                "train/loss": loss.detach(),
                "train/latent_std": latents.std().detach(),
                "train/data_std": reals.std().detach(),
                "train/gen_lr": opt_gen.param_groups[0]["lr"],
            }
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite loss at step {self.global_step}")
        logs.update({f"train/{name}": value.detach() for name, value in losses.items()})
        self.log_dict(logs, prog_bar=True, on_step=True)
        return loss


def build_wrapper(config, model):
    if not config["training"]["use_ema"]:
        raise ValueError("V requires EMA")
    # The official factory creates the EMA model independently after strict loading.
    ema_copy = create_model_from_config(config)
    ema_copy.load_state_dict(model.state_dict(), strict=True)
    return VDecoderWrapper(model, config, ema_copy)


def state_changed(module, reference):
    current = module.state_dict()
    return any(
        not torch.equal(tensor.detach().cpu(), reference[name])
        for name, tensor in current.items()
    )


class VAudit(pl.Callback):
    def __init__(self, output, model):
        self.output = output
        self.encoder_ref = {
            k: v.cpu().clone() for k, v in model.encoder.state_dict().items()
        }
        self.decoder_ref = {
            k: v.cpu().clone() for k, v in model.decoder.state_dict().items()
        }
        self.start = None
        self.early_checked = False

    def setup(self, trainer, module, stage):
        self.start = time.perf_counter()
        enc = sum(p.numel() for p in module.autoencoder.encoder.parameters() if p.requires_grad)
        dec = sum(p.numel() for p in module.autoencoder.decoder.parameters() if p.requires_grad)
        if enc != 0 or dec == 0:
            raise RuntimeError(f"invalid trainable scope: encoder={enc}, decoder={dec}")
        if any(loss.name == "kl_loss" for loss in module.losses_gen.losses):
            raise RuntimeError("KL is active")
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "initial_audit.json").write_text(
            json.dumps({"encoder_trainable": enc, "decoder_trainable": dec}, indent=2),
            encoding="utf-8",
        )

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step >= 100 and not self.early_checked:
            audit = self._audit(module)
            if not audit["encoder_unchanged"] or not audit["decoder_changed"]:
                raise RuntimeError(f"V update audit failed: {audit}")
            (self.output / "early_audit.json").write_text(
                json.dumps(audit, indent=2), encoding="utf-8"
            )
            self.early_checked = True
        if step in (1, 10) or step % 100 == 0:
            torch.cuda.synchronize()
            scaler = getattr(trainer.precision_plugin, "scaler", None)
            elapsed = time.perf_counter() - self.start
            remaining = elapsed / step * (trainer.max_steps - step) if step else None
            row = {
                "global_step": step,
                "wall_seconds": elapsed,
                "steps_per_second": step / elapsed if elapsed else None,
                "eta_hours": remaining / 3600 if remaining is not None else None,
                "eta_utc": datetime.fromtimestamp(
                    time.time() + remaining, timezone.utc
                ).isoformat(timespec="seconds") if remaining is not None else None,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "amp_scale": float(scaler.get_scale()) if scaler is not None else None,
                "generator_optimizer_state_count": len(trainer.optimizers[0].state),
            }
            for name, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    row[str(name)] = float(value.detach().cpu())
            with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("V_AUDIT " + json.dumps(row, ensure_ascii=False), flush=True)

    def _audit(self, module):
        return {
            "encoder_unchanged": not state_changed(module.autoencoder.encoder, self.encoder_ref),
            "decoder_changed": state_changed(module.autoencoder.decoder, self.decoder_ref),
            "trainable_parameter_scope": "decoder_only",
            "kl_disabled": all(loss.name != "kl_loss" for loss in module.losses_gen.losses),
        }

    def on_save_checkpoint(self, trainer, module, checkpoint):
        audit = self._audit(module)
        if trainer.global_step > 0 and (
            not audit["encoder_unchanged"] or not audit["kl_disabled"]
            or (trainer.global_step >= 100 and not audit["decoder_changed"])
        ):
            raise RuntimeError(f"V checkpoint audit failed: {audit}")
        checkpoint["v_decoder_audit"] = audit


class VDemo(pl.Callback):
    def __init__(self, output, sample, original_decoder, every):
        self.output = output
        self.reals, metadata = sample
        self.latents = metadata["latents"]
        self.original_decoder = original_decoder
        self.every = every

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == 0 or step % self.every:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        latents = self.latents.unsqueeze(0).to(module.device)
        original_decoder = self.original_decoder.to(module.device)
        original_decoder.eval()
        module.autoencoder.eval()
        module.autoencoder_ema.ema_model.eval()
        try:
            waves = {
                "original_b": self.reals,
                "old_decoder": original_decoder(latents)[0],
                "new_raw": module.autoencoder.decode(latents)[0],
                "new_ema": module.autoencoder_ema.ema_model.decode(latents)[0],
            }
            for name, wave in waves.items():
                data = wave.detach().float().cpu().numpy().T
                if data.shape != (SAMPLES, 2) or not np.isfinite(data).all():
                    raise RuntimeError(f"invalid V demo: {name}")
                sf.write(self.output / f"step_{step:06d}_{name}.wav",
                         data, SAMPLE_RATE, subtype="FLOAT")
        finally:
            self.original_decoder.cpu()
            module.autoencoder.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-sha256", type=sha256_argument, required=True)
    parser.add_argument("--singer-sha256", type=sha256_argument, required=True)
    parser.add_argument("--train-manifest-sha256", type=sha256_argument, required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=300000)
    parser.add_argument("--checkpoint-every", type=int, default=15000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--smoke-sample-id")
    parser.add_argument("--precision", default="16-mixed", choices=("16-mixed", "bf16-mixed"))
    parser.add_argument("--demo-every", type=int, default=15000)
    args = parser.parse_args()
    if len(args.cache_dir) != 4:
        raise ValueError("V requires four cache rounds in order")
    if args.smoke_sample_id is not None and args.max_steps > 32:
        raise ValueError("smoke sample selection is limited to 32 steps")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if sha256_file(args.pretrained_checkpoint) != args.vae_sha256:
        raise ValueError("pretrained VAE SHA256 mismatch")
    pl.seed_everything(42, workers=True)
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    if config["sample_size"] != SAMPLES or config["sample_rate"] != SAMPLE_RATE:
        raise ValueError("V audio config mismatch")
    if "bottleneck" in config["training"]["loss_configs"]:
        raise ValueError("KL config must be absent")
    rounds = [
        VCacheRound(path, index, args.vae_sha256, args.singer_sha256,
                    args.train_manifest_sha256, limit=args.limit_records,
                    sample_id=args.smoke_sample_id)
        for index, path in enumerate(args.cache_dir, start=1)
    ]
    dataset = ConcatDataset(rounds)
    if args.limit_records is None and len(dataset) < args.max_steps:
        raise ValueError("four cache rounds do not cover requested max_steps")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    model = create_model_from_config(config)
    print("PRETRAINED_STRICT_LOAD", model.load_state_dict(
        load_ckpt_state_dict(str(args.pretrained_checkpoint)), strict=True
    ), flush=True)
    old_model = create_model_from_config(config)
    old_model.load_state_dict(model.state_dict(), strict=True)
    original_decoder = old_model.decoder.cpu().eval()
    del old_model
    wrapper = build_wrapper(config, model)
    args.output_dir.mkdir(parents=True)
    contract = {
        "cache_manifest_sha256": [round_.manifest_sha256 for round_ in rounds],
        "round_lengths": [len(round_) for round_ in rounds],
        "dataset_length": len(dataset),
        "vae_sha256": args.vae_sha256,
        "singer_sha256": args.singer_sha256,
        "train_manifest_sha256": args.train_manifest_sha256,
        "model_config_sha256": sha256_file(args.model_config),
        "max_steps": args.max_steps,
        "kl_disabled": True,
    }
    (args.output_dir / "contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints", filename="step-{step}",
        every_n_train_steps=args.checkpoint_every, save_top_k=-1,
        save_last=True, save_on_train_epoch_end=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=args.precision,
        max_steps=args.max_steps, max_epochs=1000000,
        callbacks=[
            checkpoint_callback,
            VAudit(args.output_dir, model),
            VDemo(args.output_dir / "demo", rounds[0][0], original_decoder,
                  args.demo_every),
        ],
        logger=False, enable_progress_bar=True, log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    trainer.fit(wrapper, loader)
    final_path = args.output_dir / "checkpoints" / f"final_step_{trainer.global_step}.ckpt"
    trainer.save_checkpoint(final_path)
    result = {
        "global_step": int(trainer.global_step),
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": sha256_file(final_path),
    }
    (args.output_dir / "run_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
