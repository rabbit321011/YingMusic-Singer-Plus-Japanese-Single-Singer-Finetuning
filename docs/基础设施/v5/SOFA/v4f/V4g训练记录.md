# V4g 训练记录

V4g = V4f 配方 + 285k full-VAE online。从官方 YingMusic-Singer-Plus base 出发，使用 SOFA 对齐数据，
验证 285k VAE 的 latent 空间是否与 DiT 训练兼容。

## 训练配置

| 参数 | V4f | V4g |
|---|---|---|
| DiT 起点 | 官方 base | 官方 base |
| VAE ckpt | stable_audio_2_0_vae_20hz_official.ckpt | autoencoder_285k.ckpt |
| 数据 | tokens_SOFA_v4d_control | 同 |
| max_steps | 30000 | 同 |
| LR | 1.4e-5, warmup 500 → hold 12k → decay | 同 |
| 有效 batch | 16 | **4（DDP 异常，实际单卡）** |
| CKA / drop_text | 0.7 / 0.15 | 同 |
| 脚本 | `run_sft_v4g_ddp.sh` | 同 |
| 输出 | `ckpts/plus_ja_sft_v4f/` | `ckpts/plus_ja_sft_v4g/` |

> **DDP 异常**：训练过程中 3 个 DDP worker 崩溃，实际仅有单卡在跑。
> 有效 batch 从预期的 16 降为 4。

## FlowB 轨迹

| Step | Eval FlowA | Eval FlowB | Eval CKA | Eval Loss |
|---:|---:|---:|---:|---:|
| 1000 | 0.840 | 1.248 | 0.250 | 3.512 |
| 5000 | 0.374 | 1.050 | 0.193 | 2.608 |
| 10000 | 0.328 | 1.018 | 0.178 | 2.488 |
| 15000 | 0.397 | 1.005 | 0.173 | 2.528 |
| 18000 | 0.310 | 0.988 | 0.156 | 2.395 |
| 20000 | 0.360 | 0.998 | 0.161 | 2.469 |
| 22000 | 0.390 | 0.985 | 0.157 | 2.470 |
| 24000 | 0.300 | 0.981 | 0.158 | 2.373 |
| 26000 | 0.333 | 0.978 | 0.175 | 2.413 |
| 28000 | 0.360 | 0.986 | 0.170 | 2.450 |
| 29000 | 0.328 | 0.965 | 0.145 | 2.359 |

- FlowB 在 5k 后进入慢速下降阶段，10k 后进入 [0.96, 1.00] 振荡
- 未出现 V4f 的持续下降趋势（V4f 在 10k 时 FlowB 已到 0.934）
- 最高 eval step 29000: FlowB=0.965

## 285k VAE latent 统计

同一批 10 段target singer音频，对比两个 VAE 的 latent：

| 指标 | 官方 VAE | 285k VAE | Ratio |
|---|---|---|---|
| per-dim std | 1.072 | 0.997 | 0.930x |
| per-frame norm | 8.544 | 7.923 | 0.927x |

两个 VAE 的 latent 在尺度上接近（285k 略小 7%）。

## 权重验证

`autoencoder_285k.ckpt` 与原始训练 checkpoint（`step-step=285000.ckpt`）对比：

- online 权重：365/365 keys 完全一致 ✅
- EMA 权重：0/365 keys 匹配

确认 V4g 使用的是 285k online（非 EMA）权重。

## 120 首 PER 评测（step 28000）

| Bucket | V4c 24k | V4f 24k | **V4g 28k** |
|---|---:|---:|---:|
| 1-10 | 0.447 | 0.509 | **0.469** |
| 11-20 | 0.456 | 0.562 | **0.537** |
| 21-30 | 0.592 | 0.649 | **0.649** |
| 30+ | 0.139 | 0.171 | **0.197** |
| **ALL** | 0.409 | 0.473 | **0.463** |

- V4g PER=0.463，介于 V4c(0.409) 和 V4f(0.473) 之间
- 30+ 长句桶 0.197，略高于 V4f(0.171)
- 注意：评测在单卡训练结果上执行，不代表正常 4 卡预期的表现

## 产物

| 产物 | 路径 |
|---|---|
| ckpt 18k/24k/28k | `ckpts/plus_ja_sft_v4g/step_0{18,24,28}000.pt` |
| 训练日志 | `train_v4g.log` |
| 评测原始数据 | `${SERVER_ROOT}${CLOUD_ARTIFACT}` |
| cloud artifact

## 后续

V4fg 已完成：V4f 24k DiT + 285k VAE，15k 步。
详见下面 V4fg 章节。

---

# V4fg 训练记录

V4fg = V4f 24k DiT + 285k VAE，从 V4f 24k checkpoint 出发续训。
验证 V4f 的日语先验能否帮助 DiT 适配 285k VAE 的 latent 空间。

## 训练配置

| 参数 | V4g | V4fg |
|---|---|---|
| DiT 起点 | 官方 base | V4f 24k |
| VAE | 285k | 285k |
| max_steps | 30000 | 15000 |
| LR | 1.4e-5 | 5e-6 (÷3) |
| warmup / hold | 500 / 12000 | 250 / 6000 (等比例) |
| 其他 | — | 同 V4g |

完整 4 卡 DDP，全程无崩溃，1.28s/step。

## FlowB 轨迹

| Step | FlowA | FlowB | CKA | Loss |
|---:|---:|---:|---:|---:|
| 1000 | 0.407 | 1.032 | 0.172 | 2.592 |
| 3000 | 0.350 | 0.996 | 0.161 | 2.455 |
| 5000 | 0.322 | 0.991 | 0.161 | 2.418 |
| 7000 | 0.372 | 0.981 | 0.168 | 2.452 |
| 10000 | 0.304 | 0.986 | 0.157 | 2.385 |
| 12000 | 0.370 | 0.977 | 0.147 | 2.427 |
| 14000 | 0.332 | 0.988 | 0.159 | 2.418 |
| 15000 | 0.383 | 0.989 | 0.161 | 2.473 |

FlowB 从 1k 后进入 [0.97, 1.00] 区间，与 V4g 同平台。V4f 先验未改变收敛天花板。

## Audio Loss 评测（100 首自克隆 Mel L1）

> **注意**：V4g/V4fg 的推理需要使用 285k VAE。首轮评测脚本未区分 VAE 版本，
> V4g/V4fg 的 mel 数据使用了官方 VAE，结果不可靠，仅 V4c/V4f/official_base 的数据有效。

| Model | mel_all | mel_low | mel_mid | mel_high |
|---|---|---|---|---|
| V4c_12k | 7.06 | 12.37 | 6.02 | 3.09 |
| V4c_24k | 7.69 | 13.28 | 6.39 | 3.71 |
| V4f_24k | 7.98 | 13.82 | 7.08 | 3.40 |
| V4g_28k | 19.67 | 28.73 | 14.27 | 16.28 |
| V4fg_15k | 16.17 | 25.58 | 11.67 | 11.62 |
| official_base | 71.21 | 76.32 | 62.53 | 74.54 |

V4c/V4f 数据可信：训练步数越多 audio loss 越大，与"SVC N 形曲线"一致。

## 盲听包

正在生成中：14 个 checkpoint × 20 条样本，匿名 A-N 字母映射。
V4g/V4fg 系列使用正确的 285k VAE。
- 产物：`${CLOUD_ARTIFACT}`
- key：`${SERVER_ROOT}${CLOUD_ARTIFACT}`

## 产物

| 产物 | 路径 |
|---|---|
| V4fg ckpt | `ckpts/plus_ja_sft_v4fg/step_015000_final.pt` |
| V4fg 训练日志 | `train_v4fg.log` |
| Audio eval 脚本 | `${SERVER_ROOT}/tools/audio_eval/` |
| Audio eval 结果 | `${SERVER_ROOT}${CLOUD_ARTIFACT}` |
| 盲听包 | `${CLOUD_ARTIFACT}` |

















