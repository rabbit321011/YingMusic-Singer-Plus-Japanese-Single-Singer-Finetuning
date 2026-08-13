import hashlib

import torch


def stable_game_seed(sample_key, base_seed):
    payload = f"{sample_key}|{base_seed}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def extract_game_notes_with_posterior(
    model,
    waveform,
    duration,
    language_id,
    nsteps,
    seed,
    device,
):
    """Run one frozen GAME pass and retain estimator probabilities."""
    from modules.decoding import decode_gaussian_blurred_probs

    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform [samples], got {tuple(waveform.shape)}")
    schedule = torch.arange(nsteps, device=device, dtype=torch.float32) / nsteps
    known_durations = torch.tensor([[duration]], device=device, dtype=torch.float32)
    boundary_threshold = torch.tensor(0.2, device=device)
    boundary_radius = torch.tensor(2, device=device, dtype=torch.long)
    score_threshold = torch.tensor(0.2, device=device)
    language = torch.tensor([language_id], device=device, dtype=torch.long)
    waveform = waveform.unsqueeze(0).to(device)
    cuda_devices = [device.index or 0] if device.type == "cuda" else []

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            x_seg, x_est, time_mask = model.forward_encoder(
                waveform=waveform,
                duration=known_durations.sum(dim=1),
            )
            durations, regions, max_n = model.forward_segmenter(
                x_seg,
                known_durations=known_durations,
                mask=time_mask,
                language=language,
                t=schedule,
                threshold=boundary_threshold,
                radius=boundary_radius,
            )
            note_index = torch.arange(
                max_n, dtype=torch.long, device=device
            ).unsqueeze(0)
            note_mask = note_index < regions.amax(dim=-1, keepdim=True)
            logits = model.model.forward_estimation(
                x_est,
                regions=regions,
                t_mask=time_mask,
                n_mask=note_mask,
            )
            probs = logits.sigmoid()
            scores, presence = decode_gaussian_blurred_probs(
                probs=probs,
                min_val=model.inference_config.midi_min,
                max_val=model.inference_config.midi_max,
                deviation=model.inference_config.midi_std * 3,
                threshold=score_threshold,
            )
            presence = presence & note_mask
            scores = scores * note_mask.float()

    return {
        "durations": durations[0].float().cpu(),
        "presence": presence[0].bool().cpu(),
        "scores": scores[0].float().cpu(),
        "pitch_probs_257": probs[0].float().cpu(),
    }
