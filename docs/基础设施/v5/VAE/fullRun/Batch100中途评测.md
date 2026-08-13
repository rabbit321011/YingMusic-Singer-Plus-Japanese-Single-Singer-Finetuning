## 22. 3.3 objective comparison scaffold prepared for full-VAE (2026-07-16)

User chose to postpone 3.2 listening and prioritize 3.3 objective comparison after full-VAE 300k finishes. A dedicated local experiment scaffold was prepared:

```text
${LOCAL_PROJECT_ROOT}\experiments\vae_full_compare_3p3_20260716
```

Prepared artifacts:

```text
${LOCAL_PROJECT_ROOT}\experiments\vae_full_compare_3p3_20260716\PLAN.json
${LOCAL_PROJECT_ROOT}\experiments\vae_full_compare_3p3_20260716\SPEC.json
${LOCAL_PROJECT_ROOT}\experiments\vae_full_compare_3p3_20260716\prepare_full_vae_3p3.py
${LOCAL_PROJECT_ROOT}\experiments\vae_full_compare_3p3_20260716\run_full_vae_3p3.py
```

The fixed held-out samples remain `sample02_test`, `sample03_test`, and `sample04_test`. The planned comparison set is: original VAE, Decoder-only 30k online, Decoder-only 30k EMA, full-VAE 300k online, and full-VAE 300k EMA. Metrics are per-FFT log-magnitude MAE, per-band energy ratio, per-band log-magnitude MAE, and a per-sample summary ranking.

Execution is intentionally deferred until `final_step_300000.ckpt` exists. Then 3.3 should run before any new listening package.

## 23. Batch-100 objective comparison with current full-VAE checkpoint (2026-07-16)

A larger 3.3 objective comparison was run on 100 held-out test samples using the current full-VAE `last.ckpt` rather than waiting for `final_step_300000.ckpt`. The compared set was: `original_vae`, `decoder30k_ema`, `fullvae_current_online`, `fullvae_current_ema`. The full-VAE reference step at evaluation time was 193100.

This comparison did not use simple win-rate. It introduced per-sample reconstruction difficulty based on the `original_vae` baseline and used a composite sample score that combines overall FFT error improvement, `4–8k`, `8–12k`, `12–20k` band error improvement, and light penalties for excessive high-frequency energy deviation.

Aggregate outcome:

```text
decoder30k_ema:
mean_score        =  0.000468
median_score      =  0.008022
weighted_score    =  0.005749
worst_10pct_mean  = -0.083652
hard_subset_mean  =  0.015638

fullvae_current_online:
mean_score        = -0.006018
median_score      =  0.004382
weighted_score    =  0.002838
worst_10pct_mean  = -0.107006
hard_subset_mean  =  0.018422

fullvae_current_ema:
mean_score        = -0.049786
median_score      = -0.040097
weighted_score    = -0.039127
worst_10pct_mean  = -0.177464
hard_subset_mean  = -0.022616
```

Interpretation:

- `decoder30k_ema` remains the most stable overall solution at the current stage.
- `fullvae_current_online` is already stronger on difficult samples, which is the main positive signal for the full-VAE route.
- `fullvae_current_ema` is currently not ready; it is dragged down by unstable `8–12kHz` energy behavior despite some competitive raw reconstruction improvements.
- The evidence does not support saying current full-VAE has already fully surpassed decoder-only. It does support continuing the full-VAE route to 300k, because the online model is already more promising on hard samples.

Recommended focus before 300k completes: if any limited listening is done, compare `decoder30k_ema` against `fullvae_current_online`, not against the current full-VAE EMA.

Detailed result document:

```text
${LOCAL_PROJECT_ROOT}\experiments\vae_full_compare_3p3_20260716\RESULT_BATCH100.md
```

---

## 2026-07-17 追加：中途结论已被最终轨迹分析推翻

本文写于 full-VAE 300k 训练中途（step 193100/300000），结论为「decoder30k_ema 仍是最稳定方案」。

300k 训练完成后，完整 checkpoint 轨迹分析（15k→300k，11 个检查点）和最终人耳盲听得出了不同结论：

- **full-VAE 285k online 优于 decoder30k_ema**，尤其在 8-20kHz 高频表现
- decoder30k 的高频损失被判定为人耳不可接受的致命缺陷
- 195k→225k 被识别为关键相变点，最佳 checkpoint 为 285k（非 300k）

**如只读本文而不读 `VAE实验结论.md`，将得出完全相反的选型结论。** 本文作为训练中途的快照保留，不等同最终结果。
















