# A→B 边界消融

## 目标

定位「第一句前半状态较差、唱到中段改善、第二句开始稳定」是否由推理侧 reference tail 静音造成。

## 固定条件

- 模型：V4fg 10k
- VAE：285k online
- 推理：整段 SOFA phrases T1
- seed：123
- steps：32
- CFG：3.0
- 样本：正式 v2 中的 `03 / 08 / 11 / 13 / 14 / 18`
- 雾感样本 `15` 留给后续 A/B 条件交换，不混入本实验主判断。

## 唯一变量

```text
reference tail = 0 / 0.25 / 0.5s
```

## 产物与审计

- 6 条 × 3 变体 = 18 次生成，全部完成。
- 每条目录包含 GT、B 区 lyrics 和匿名 `variant_A/B/C`。
- `0.5s` 组逐样本、逐采样点精确复现正式 v2 中的 V4fg 10k 输出，确认实验只有 reference tail 一个变量。
- 全部 WAV 与 GT 为 44.1kHz 双声道同帧。
- ZIP 30 个条目，完整性通过，未包含 key。
- SHA256: redacted`4f8165cea4215d0cc45c405dd13c2b3dd2dc082de9890ad6c2433c9093ead8cd`。

cloud artifact

```text
${CLOUD_ARTIFACT}
${CLOUD_ARTIFACT} redacted
```

本地：

```text
${LOCAL_PROJECT_ROOT}\TEMP\ref_tail_ablation_v4fg10k_download\listening_package\
```

## 当前门禁

匿名映射：

```text
A = 0.5s
B = 0.25s
C = 0s
```

用户盲听：

| 样本 | 结果 |
|---|---|
| 03 | 三者接近 |
| 08 | 0s 最好 |
| 11 | 0.5s / 0.25s 都不错，0s 较弱 |
| 13 | 0s 明显最好 |
| 14 | 0.5s 稍好 |
| 18 | 总体接近；0.5s 有咬字问题，但咬字不作为本消融主判据 |

结论：

- 固定 0.5s 静音确实是部分样本第一句状态差的原因，0s 在 08/13 上有明确收益。
- 该效应不是普适单调关系；11/14 仍偏好保留静音。
- 0.25s 没有任何样本独立胜出，不构成新的统一默认值。
- A/B 切点前后 0.25/0.5 秒 RMS 与偏好不一致，不能建立可靠的能量阈值自适应规则。
- 暂不全局改正式推理默认值；0s 保留为候选，待真实跨源 A/B 工作流再做小型复核。
- 下一步转入 sample_15 的轻柔罕见音色雾感诊断，并保持 0.5s 基线以精确复现原观察。

















