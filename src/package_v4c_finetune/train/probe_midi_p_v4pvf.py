import argparse
import json
import os
import sys

import torch


for candidate in (
    os.getcwd(),
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "YingMusic-Singer-Plus-src")
    ),
):
    if os.path.isdir(os.path.join(candidate, "src")) and candidate not in sys.path:
        sys.path.insert(0, candidate)


def synthetic_checks():
    from src.YingMusicSinger.melody.midi_p_v4pvf import (
        MIDI_P_SCHEMA,
        QuantizedMIDIPV4Pvf,
        decode_quantized_midi_classes,
        summarize_midi_classes,
    )

    midi_logits = torch.full((1, 8, 128), -12.0)
    midi_logits[:, :4, 60] = 12.0
    midi_logits[:, 4:6, :] = -12.0
    midi_logits[:, 6:, 62] = 12.0
    boundary_logits = torch.full((1, 8, 1), -12.0)
    boundary_logits[:, 0, 0] = 12.0
    boundary_logits[:, 4, 0] = 12.0
    boundary_logits[:, 6, 0] = 12.0

    classes = decode_quantized_midi_classes(
        midi_logits, boundary_logits, target_len=8
    )
    expected = torch.tensor(
        [[120, 120, 120, 120, 255, 255, 124, 124]], dtype=torch.long
    )
    if not torch.equal(classes.cpu(), expected):
        raise AssertionError(f"Synthetic class mismatch: {classes.tolist()}")

    module = QuantizedMIDIPV4Pvf()
    embedded, embedded_classes = module(midi_logits, boundary_logits, target_len=8)
    embedded.square().mean().backward()
    grad = module.embedding.weight.grad
    if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
        raise AssertionError("P embedding did not receive finite, non-zero gradients")
    if not torch.equal(embedded_classes.cpu(), expected):
        raise AssertionError("Module and functional class decoding disagree")

    return {
        "schema": MIDI_P_SCHEMA,
        "classes": classes.tolist(),
        "stats": summarize_midi_classes(classes),
        "embedding_grad_abs_sum": float(grad.abs().sum().item()),
    }


def real_audio_probe(audio_path, midi_ckpt, device):
    import torchaudio

    from src.YingMusicSinger.melody.midi_extractor import MIDIExtractor
    from src.YingMusicSinger.melody.midi_p_v4pvf import (
        decode_quantized_midi_classes,
        summarize_midi_classes,
    )
    from src.YingMusicSinger.utils.mel_spectrogram import MelodySpectrogram

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)
    if not os.path.isfile(midi_ckpt):
        raise FileNotFoundError(midi_ckpt)

    wav, sr = torchaudio.load(audio_path)
    mel_extractor = MelodySpectrogram().to(device)
    teacher = MIDIExtractor(in_dim=80)
    teacher._load_form_ckpt(midi_ckpt)
    teacher = teacher.to(device).eval()
    with torch.inference_mode():
        mel = mel_extractor(audio=wav.to(device), sr=sr)
        midi_logits, boundary_logits = teacher(mel.transpose(1, 2))
        target_len = max(1, round(wav.shape[-1] / sr * (44100 / 2048)))
        classes = decode_quantized_midi_classes(
            midi_logits, boundary_logits, target_len=target_len
        )

    stats = summarize_midi_classes(classes)
    if stats["frames"] != target_len:
        raise AssertionError(f"Expected {target_len} frames, got {stats['frames']}")
    if stats["pad_frames"] != 0:
        raise AssertionError(f"Unexpected PAD frames in a single real sample: {stats}")
    if stats["unique_pitch_classes"] == 0:
        raise AssertionError(f"No voiced pitch classes decoded from real audio: {stats}")
    return {
        "audio": os.path.abspath(audio_path),
        "sample_rate": sr,
        "audio_samples": wav.shape[-1],
        "some_frames": midi_logits.shape[1],
        "boundary_frames": boundary_logits.shape[1],
        "target_vae_frames": target_len,
        "stats": stats,
        "first_64_classes": classes[0, :64].cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=None)
    parser.add_argument(
        "--midi_ckpt", default="ckpts/model_ckpt_steps_100000_simplified.ckpt"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    result = {"synthetic_unit_checks": synthetic_checks()}
    if args.audio is not None:
        result["real_audio_probe"] = real_audio_probe(
            audio_path=args.audio,
            midi_ckpt=args.midi_ckpt,
            device=args.device,
        )
    else:
        result["real_audio_probe"] = None

    rendered = json.dumps(result, ensure_ascii=True, indent=2)
    print(rendered)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
