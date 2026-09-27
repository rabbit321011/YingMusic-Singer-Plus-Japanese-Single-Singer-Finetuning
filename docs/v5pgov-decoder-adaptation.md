# V5PgOV: generated-latent decoder adaptation

V5PgOV keeps the V5PgO 8K DiT and the 285K online VAE encoder fixed. It
fine-tunes only the VAE decoder on pairs of generated B-region latents
(`z_gen`) and the corresponding authorized original waveform (`x_B`). The
published training entry is
[`train_v_decoder_adapt.py`](../src/package_v4c_finetune/train/train_v_decoder_adapt.py);
the matching model and loss configuration is
[`model_v_decoder_adapt.json`](../config/model_v_decoder_adapt.json).

The experiment used four independently sampled cache rounds, 32 latent
frames per training window (65,536 audio samples at 44.1 kHz), 300,000
training steps, alternating generator/discriminator updates, EMA, and no KL
loss. Its cache-generation sampling used 32 Euler steps with `t_shift=0.5`
and per-record three-way guidance drawn from audio 0.4-0.7, text 0.4-0.6,
MIDI 0.4-0.5. The decoder training itself does not have a CFG scale.

## Inputs and boundaries

The training script expects four cache directories, each with `manifest.json`
and `samples/<sample_id>.npz`. Each NPZ contains float32 `z_gen` with shape
`[decoded_frames, 64]`. Manifest entries provide the cache-relative path,
`z_gen_frames`, `sample_id`, `source_path`, `x_b_sample_start`, and
`x_b_sample_count`. `source_path` points to an authorized local WAV at 44.1
kHz (mono or stereo); the script reads only the B window and never copies
that audio into this repository. The manifest header binds all rounds to
the VAE, singer checkpoint, and one training manifest by SHA256. The
training entry validates those bindings and refuses to overwrite its
output directory.

The original cache builder also depends on a project-specific PgO model
loader, H placement, GAME-P cache, and local dataset contracts. Those
inputs and the generated cache are **not** included here. This is the
decoder-training stage for compatible caches, not a standalone pipeline
that reconstructs private training data or weights.

Install the matching Stable Audio Tools **training** implementation,
PyTorch, PyTorch Lightning, NumPy, and SoundFile in your own environment.
The upstream inference-only VAE helper in this repository is not a
replacement for the training dependency.

```bash
PYTHONPATH=/path/to/stable-audio-tools \
python src/package_v4c_finetune/train/train_v_decoder_adapt.py \
  --model-config config/model_v_decoder_adapt.json \
  --pretrained-checkpoint /path/to/authorized/online_vae.ckpt \
  --vae-sha256 <sha256-of-online-vae> \
  --singer-sha256 <sha256-of-cache-generating-PgO-checkpoint> \
  --train-manifest-sha256 <sha256-of-authorized-training-manifest> \
  --cache-dir /path/to/v_latent_01 \
  --cache-dir /path/to/v_latent_02 \
  --cache-dir /path/to/v_latent_03 \
  --cache-dir /path/to/v_latent_04 \
  --output-dir /path/to/new-output \
  --max-steps 300000 --checkpoint-every 15000 \
  --demo-every 15000 --num-workers 8 --precision 16-mixed
```

All SHA256 values above are supplied by the user; no dataset or
checkpoint fingerprint is embedded in the public script. Keep the cache,
manifests, weights, logs, and output directory outside the Git checkout.
The script uses `source_path` from a local manifest and writes sample WAVs
to its private output directory; do not publish these artifacts without
the relevant rights.

## Observations and limits

In the internal difficult-sample comparison, the adapted raw and EMA
decoders were preferred over the old decoder on five sets, but soft and
low-pressure singing sometimes lost clarity or acquired hollow/windy
artifacts. The final 300K EMA was chosen for stability in a subsequent
longer comparison. These are subjective observations from a small set
that overlaps the training-song pool, not a blinded evaluation or evidence
of generalization to unseen songs. Adapting the decoder does not isolate
the cause of every quality difference, and the decoder weights are not a
standalone VAE: inference still needs the matching base VAE and PgO DiT.

No training audio, generated audio, caches, manifests, checkpoints,
decoder weights, or cloud-storage links are distributed with this update.
