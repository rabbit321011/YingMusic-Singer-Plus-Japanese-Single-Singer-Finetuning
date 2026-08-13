## 15. Server 10-Step Smoke Test Stopped at Decoder-Update Boundary — 2026-07-15

Execution followed the approved download order:

- official source commit `239a0d8` was cloned directly from GitHub on the server;
- the production VAE checkpoint already existed on the server and matched local SHA-256 `dc2c4a8e...ce0b39`, so no model transfer was needed;
- the server had equivalent YMSP config semantics and the two selected train-BV audio sources;
- Hugging Face was unreachable, but GitHub, ModelScope, and ghfast were reachable;
- PyTorch mirror range tests measured about 2.1 MB/s official, 32.1 MB/s Aliyun, 19.8 MB/s SJTU, and 30.9 MB/s NJU;
- the independent environment was therefore installed through the Aliyun PyTorch/PyPI mirrors, not by cloning or modifying an existing server environment.

Server root:

```text
${SERVER_ROOT}/experiments/vae_b0_official_239a0d8_20260715
```

Environment result:

```text
Python 3.10.20
Torch 2.6.0+cu124
Torchaudio 2.6.0+cu124
PyTorch Lightning 2.1.0
pip check: No broken requirements found
GPU: RTX 4090
```

The vendor worktree remained clean. Strict checkpoint loading and an official encode-decode run succeeded. For the original 24576-sample crop, official runtime produced latent shape `[1,64,12]`, decoded shape `[1,2,24576]`, took 0.370 s on first measured inference, and reserved 1.256 GiB.

### Training configuration actually tested

```text
one RTX 4090
batch 1
16-mixed
sample_size 24576 = 0.557279 s
encoder requires_grad false
encoder_freeze_on_warmup true
Decoder + Encodec discriminator trainable
original MR-STFT + adversarial + feature-matching + KL losses
original AdamW + InverseLR schedulers
EMA disabled
max_steps 10 through an external experiment launcher
```

The server source audio files differed slightly from the local B0 sources, so deterministic server-specific -6 dBFS copies were created and their hashes recorded. This is acceptable only for an engineering smoke test; no perceptual conclusion is drawn.

### What succeeded

- Three bounded attempts each completed all ten trainer/global steps.
- No OOM occurred.
- All logged metrics were finite.
- Encoder gradients were absent on every audited step.
- Decoder gradients were nonzero for 182 parameter tensors on generator steps.
- Discriminator gradients were nonzero for at least 80 parameter tensors.
- The official trainer reported 78.0M non-trainable and 80.0M trainable parameters.
- After the first step, callback-observed step time was roughly 0.05-0.10 s for the tiny 0.557-second crop.

### Why execution stopped

The first two attempts completed training but failed only in the external audit callback after Lightning moved the model to CPU:

1. attempt 1 decoded a CUDA fixed latent with CPU Decoder weights;
2. attempt 2 queried CUDA peak memory using `module.device`, which had become CPU.

These did not alter official source or training behavior. The external callback was corrected between attempts.

The final attempt reached the approved substantive stop condition: after five generator updates, Decoder state SHA-256 was byte-identical to its initial state despite nonzero gradients. The exact original scheduler uses `warmup=0.999`, so generator LR was only approximately `1.999e-7` through `5.985e-7` during the ten-step run. It is plausible that the resulting updates were below FP32 parameter resolution, but this is not yet proven.

Because Decoder update was an explicit required condition, no longer run, LR change, scheduler change, loss removal, checkpoint fabrication, or vendor-source modification was attempted. Peak training memory and checkpoint reload remain unverified because the final stop occurred before those artifacts were completed.

Server evidence:

```text
${SERVER_ROOT}/experiments/vae_b0_official_239a0d8_20260715/runs/smoke_single_gpu
```

Local copied summaries:

```text
${LOCAL_PROJECT_ROOT}\experiments\vae_decoder_adaptation_audit_20260715\server_results\smoke_single_gpu_20260715\RESULT.md
${LOCAL_PROJECT_ROOT}\experiments\vae_decoder_adaptation_audit_20260715\server_results\smoke_single_gpu_20260715\result_summary.json
${LOCAL_PROJECT_ROOT}\experiments\vae_decoder_adaptation_audit_20260715\server_results\smoke_single_gpu_20260715\environment.txt
```

### Decision required before another run

Choose explicitly between:

1. preserve the exact original scheduler and run enough generator steps for the exponential warmup to produce representable updates; or
2. create a smoke-only derived scheduler configuration with warmup disabled so parameter-update correctness can be verified quickly.

Neither choice has been executed.

















