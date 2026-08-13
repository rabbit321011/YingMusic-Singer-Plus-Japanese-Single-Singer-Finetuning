import argparse
import hashlib
import json
import pathlib

import torch
import torchaudio


def sha256_file(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from src.YingMusicSinger.utils.stable_audio_tools.vae_copysyn import StableAudioInfer

    audio, sample_rate = torchaudio.load(args.audio)
    if sample_rate != 44100 or audio.shape[0] != 1:
        raise ValueError("probe input must be 44.1kHz mono")
    audio = audio[:, : 10 * sample_rate].cuda()
    vae = StableAudioInfer(args.config, args.checkpoint).cuda().eval()
    with torch.no_grad():
        latent = vae.encode_audio(audio, in_sr=sample_rate)
        decoded = vae.decode_audio(latent)
    if not torch.isfinite(latent).all() or not torch.isfinite(decoded).all():
        raise ValueError("official VAE emitted non-finite values")
    decoded = decoded.squeeze(0).detach().cpu()
    if decoded.numel() == 0 or float(decoded.abs().max()) <= 1e-6:
        raise ValueError("official VAE decode is blank")
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output, decoded, sample_rate)
    report = {
        "schema": "v5p_official_vae_inference_smoke_v1",
        "input": str(pathlib.Path(args.audio).resolve()),
        "input_sha256": sha256_file(args.audio),
        "vae_checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "vae_checkpoint_sha256": sha256_file(args.checkpoint),
        "latent_shape": list(latent.shape),
        "decoded_shape": list(decoded.shape),
        "decoded_peak": float(decoded.abs().max()),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
