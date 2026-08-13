# Full-VAE 300k 实验结论与归档

> 2026-07-16，VAE 分支收尾。

## 选择

**full-VAE 285k online（B）** 为当前最佳 VAE 模型。

备选：285k EMA（C），极高音场景比 online 略稳，细节略少。

## 结论

### Decoder-only 30k（EMA）

- 低频忠实度高，但 **高频系统性丢失**（"一高就糊"、"高音破音严重"）
- 人耳不可接受，不适合最终使用

### Full-VAE 整个训练轨迹

| 阶段 | 步数范围 | 现象 |
|---|---|---|
| Encoder 重构期 | 15k–75k | Encoder 激进改变 latent，高频牺牲最大 |
| Decoder 追赶期 | 75k–195k | Decoder 慢慢适应新 latent |
| **相变点** | **~225k** | 12-20kHz 改善从 −0.13 跳变到 +0.04，全频段转正 |
| 稳定提升期 | 225k–285k | 所有频段单调改善 |
| 轻度过拟合 | 285k–300k | 4-8k 和 12-20k 出现退化 |

### 最佳 checkpoint：285k（非 300k）

300k 的 4-8k 和 12-20k 相对 285k 有轻度退化。训练 loss 全程下降但评测指标在 285k 后转弱——loss 和感知质量脱节的典型案例。

### 285k online vs decoder30k_ema（客观 + 人耳）

| 频段 | 客观胜出 | 人耳 |
|---|---|---|
| 0-4k 低频 | 76–78/100 | B 细节更好 |
| 4-8k 中高频 | 29/100（最弱项）| B 少数样本糊 |
| 8-20k 高频 | 34–68/100 | **A 完全不能听高音，B 正常** |

decoder30k 的高频丢失是人耳不可接受的致命伤。285k online 在绝大多数样本上更好，人耳盲听确认（10 样本中 5 明确胜出、0 明确输）。

## 产物位置

| 产物 | 路径 |
|---|---|
| **最佳 checkpoint** | `${SERVER_ROOT}/experiments/vae_full_official_300k_20260716/runs/full_vae_300k_20260716/checkpoints/step-step=285000.ckpt` |
| **备选 checkpoint** | 同上目录，285k EMA 在同一文件内 |
| 300k checkpoint | 同上目录 `final_step_300000.ckpt` |
| 轨迹数据（11 点 × 100 样本） | `runs/trajectory_20ckpts/step_*.json` |
| 盲听评测包 v2 | `runs/trajectory_20ckpts/vae_blind_v2.zip`（cloud artifact
| decoder30k_ema | `${SERVER_ROOT}/experiments/vae_b0_official_239a0d8_20260715/runs/formal_decoder_30k_20260716/checkpoints/final_step_30000.ckpt` |

## 后续

1. 用 285k full-VAE 替换 decoder30k，跑 DiT 推理验证 latent 分布兼容性
2. 如果不兼容，考虑 STAR-VAE（同帧率同维度，ICML 2026）作为替代
3. 加帧率/加深层数 = 完全重训路线，暂不投入

















