## 16. Formal Decoder-Only VAE Training Started — 2026-07-16

The user requested a formal official-recipe VAE run in tmux session data.

### Trainable boundary

This is not full Encoder+Decoder training. The production latent interface must remain stable for the existing Singer/DiT.

```text
Encoder: frozen, 77,989,888 parameters, requires_grad false
Decoder: trainable, 78,122,626 parameters
Encodec discriminator: trainable, 1,883,530 parameters
EMA: enabled through the official factory ema_copy path
```

The official model factory sets every Encoder parameter to `requires_grad=False`; the training wrapper additionally encodes under `torch.no_grad()` after warmup, with warmup_steps set to zero.

### Train-only dataset construction

A symlink-only server directory was generated from the existing server `split_manifest.json` and audio directory:

```text
${SERVER_ROOT}/experiments/vae_b0_official_239a0d8_20260715/inputs/train_audio
```

Audit:

```text
train BV manifest count: 1495
linked train BV count: 1495
linked train audio files: 16161
excluded test audio files: 331
unknown files: 0
train/test BV overlap: 0
```

No audio was copied. The directory contains links only to audio whose filename BV belongs to `train_bvs`.

### Formal configuration

```text
official commit: 239a0d8477db5477df5f046965bf2f25985510d6
one RTX 4090, CUDA_VISIBLE_DEVICES=0
batch size: 1
precision: 16-mixed
sample_size: 24576 = 0.557279 s
max_steps: 30000
checkpoint interval: 2000 steps
save_last: true
original MR-STFT, adversarial, feature-matching, KL losses
original AdamW and InverseLR schedules
EMA: enabled
```

One GPU was selected because measured throughput was already about 9.3 steps/s and peak reserved memory was only 4.20 GiB. Six-GPU DDP would add avoidable alternating-optimizer and synchronization complexity for an estimated roughly 54-minute single-GPU run.

Server run directory:

```text
${SERVER_ROOT}/experiments/vae_b0_official_239a0d8_20260715/runs/formal_decoder_30k_20260716
```

Tmux session:

```text
vae_decoder_30k
```

Monitor:

```bash
tmux session -f ${SERVER_ROOT}/experiments/vae_b0_official_239a0d8_20260715/runs/formal_decoder_30k_20260716/train.log

tail -f ${SERVER_ROOT}/experiments/vae_b0_official_239a0d8_20260715/runs/formal_decoder_30k_20260716/metrics.jsonl
```

### First checkpoint verification

At observation time the run had passed step 3400 and remained active. Peak measurements:

```text
allocated: 3.848 GiB
reserved: 4.197 GiB
throughput: approximately 9.3 steps/s
```

The step-2000 checkpoint exists, along with `last.ckpt`. Its embedded audit is:

```json
{
  "encoder_unchanged": true,
  "decoder_changed": true,
  "official_commit": "239a0d8477db5477df5f046965bf2f25985510d6"
}
```

This resolves the prior 10-step ambiguity: the exact original scheduler does produce real Decoder updates once its exponential warmup advances. No scheduler or LR modification was needed.

## 17. Formal Decoder-only 30k completion and listening package (2026-07-16)

The official run completed normally with exit code 0 at global step 30,000. Runtime was approximately 55 minutes at about 9 steps/s. Losses remained finite and no OOM occurred.

Final server checkpoint:

```text
${SERVER_ROOT}/experiments/vae_b0_official_239a0d8_20260715/runs/formal_decoder_30k_20260716/checkpoints/final_step_30000.ckpt
SHA-256: 1e787e374fadaff173f020852a6b824495cbde7155524cf0d021b0e3736fea25
```

Checkpoint audit confirmed that the Encoder remained unchanged and the Decoder changed. Strict Lightning wrapper reload returned `<All keys matched successfully>`. The checkpoint contains the online autoencoder, discriminator, EMA copy, two optimizer states, and two scheduler states; `ema_step` is 15,000.

A held-out listening package was generated from three familiar test sample_media.wav For each sample, the original frozen Encoder was run once with seed 0 and the exact same cached latent was decoded through the original, trained-online, and trained-EMA Decoders. Therefore differences among the three reconstructions isolate Decoder behavior rather than Encoder sampling.

Local package and extracted comparison directory:

```text
${LOCAL_PROJECT_ROOT}\experiments\vae_decoder_adaptation_audit_20260715\server_results\formal_decoder_30k_20260716\vae_decoder_30k_test_comparison_20260716.tar.gz
${LOCAL_PROJECT_ROOT}\experiments\vae_decoder_adaptation_audit_20260715\server_results\formal_decoder_30k_20260716\test_comparison
```

Archive SHA-256:

```text
173f19fe707a13d89a6d107829f9e0a3218dc3825f670dc4878de317adec1083
```

Each sample contains `00_normalized_original.wav`, `01_original_decoder.wav`, `02_trained_online_decoder.wav`, and `03_trained_ema_decoder.wav`.

The package structure and all 12 WAV files were verified locally. No claim that the trained Decoder is better is made before listening evaluation. Primary listening criteria are preservation of target singer's delicate timbre, blur/detail loss, naturalness, articulation, and any new GAN/electronic/noise artifacts.

















