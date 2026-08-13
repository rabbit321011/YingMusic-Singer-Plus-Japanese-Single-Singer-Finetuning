# V4g 训练计划

V4g 是 V4f 的 VAE 升级版。唯一变量：将官方 VAE 替换为 full-VAE 285k online。

## 动机

- VAE 线已收尾：full-VAE 285k online 在 8-20kHz 高频表现显著优于 decoder30k_ema
- 需要验证新 VAE 的 latent 空间是否与 DiT 训练兼容
- V4g = V4f 配方 + 285k VAE，控制单变量

## 配置（与 V4f 完全一致，仅 VAE 不同）

| 参数 | V4f | V4g |
|---|---|---|
| DiT base | YingMusicSinger_model.pt | 同 |
| 训练数据 | tokens_SOFA_v4d_control | 同 |
| max_steps | 30000 | 同 |
| LR | 1.4e-5 (warmup 500, hold 12k) | 同 |
| batch | 1 × grad_accum 4 = 有效 16 | 同 |
| 精度 | fp32 | 同 |
| CKA | 0.7 | 同 |
| drop_text | 0.15 | 同 |
| flow_b_weight | 2.0 | 同 |
| **VAE config** | stable_audio_2_0_vae_20hz_official.json | **同** |
| **VAE ckpt** | stable_audio_2_0_vae_20hz_official.ckpt | **autoencoder_285k.ckpt** |

## 脚本

- 训练脚本：`train_plus_v4g.py`（复制自 `train_plus_v4d.py`，仅改第 112 行 VAE ckpt 路径）
- 启动脚本：`run_sft_v4g_ddp.sh`（复制自 `run_sft_v4f_sofa_base30k_ddp.sh`，改名换路径）
- 输出目录：`ckpts/plus_ja_sft_v4g`

## 时间线

- 2026-07-16 15:57：脚本就绪（cp + 一行 sed）
- 2026-07-16 16:09：10 步 smoke 通过，VAE 正常加载
- 2026-07-16 16:12：30k 全量训练启动，tmux session 4090
- 预计完成：约 2-3 小时（2026-07-17 追记：训练仍在运行中，实际耗时远超初始估计。原估计有误，以 tmux session V4f 24k vs V4g 30k：

- 自动指标：固定 120 首 PER（Whisper large-v3）、F0-CORR（torchcrepe）
- 人耳盲听：T1 逐句推理下，VAE 差异是否可感

















