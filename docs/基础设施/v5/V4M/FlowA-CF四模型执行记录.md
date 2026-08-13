# FlowA-CF 四模型执行记录

## 目标

在已经冻结的 `FlowA-CF` 协议下比较：

- M600-D 30k；
- HighLR 30k；
- V4PH 30k；
- V4fg 10k。

评测回答正确 A 下的 B 预测质量、自然错误 A 的选择性、B 多模态响应、旋律结构泄漏、预测方差
收缩以及 timestep/难度分层问题。它不把训练日志里的 `FlowA-self` 当成 A 条件利用，也不使用
`distance(B)/distance(A)` 比值。

## 比较边界

四个 checkpoint 不能作为一个严格单变量 cohort 汇总排名。

### 严格 cohort

M600-D 30k、HighLR 30k 和 V4PH 30k 共享：

- 冻结的 201 条 H eval；
- H/PUL 文本 placement；
- GAME-P MIDI 和 CKA target；
- official VAE latent；
- 相同 `ref_len`、五种 A、五个 timestep 和 noise；
- 同一个 stimulus cache。

这三个模型之间允许做逐样本、逐 timestep、逐 A 的 hash-identical 配对比较。

### V4fg 参考 cohort

V4fg 10k 的训练语义绑定为：

- 285k online VAE；
- 旧 SOFA 时间戳文本 placement；
- SOME MIDI teacher；
- `MIDIFuzzDisturb` 条件链。

因此 V4fg 使用独立刺激缓存。SOME teacher 输出及其随机 blur/drop/noise 扰动按样本 seed 确定性
冻结，A 区 MIDI 清零；同一条样本的五种 A 仍共享完全相同的 B latent、B target/noise、B 文本、
B MIDI、CKA target、`ref_len` 和 timestep。V4fg 内部的 A 反事实结论有效，但它与严格 cohort 的
latent、target、文本和 MIDI hash 不相同，只能作为跨语义参考，不能把 Loss 或距离绝对值直接解释
为 checkpoint 单变量差异。

## 冻结输入

严格 cohort 直接复用 12k 基准缓存：

```text
${SERVER_ROOT}${CLOUD_ARTIFACT}
SHA256: redacted
```

冻结 eval：

```text
${SERVER_ROOT}${CLOUD_ARTIFACT}
SHA256: redacted
```

V4fg VAE：

```text
${SERVER_ROOT}/experiments/vae_full_official_300k_20260716/checkpoints/autoencoder_285k.ckpt
SHA256: redacted
```

SOME teacher：

```text
${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/model_ckpt_steps_100000_simplified.ckpt
SHA256: redacted
```

## Checkpoint 审计

Primary 全部使用 EMA。

| 模型 | checkpoint | SHA256: redacted
|---|---|---|---|
| HighLR 30k | `ckpts/plus_ja_sft_v4ph_30k_highlr/step_030000_final.pt` | `0c9aeaf20789af52537c571b855a1041f9cf51c887eda6deea24bd99352a5466` | strict cohort |
| M600-D 30k | `${SERVER_ROOT}${CLOUD_ARTIFACT}` | `d20339789a4f0d8fb3487f0f6158a54011c94b7ae09566b122a588283c3cda70` | strict cohort |
| V4PH 30k | `ckpts/plus_ja_sft_v4ph/step_030000_final.pt` | `80cf7251c19a0d803a7bb0ce9e363fd1700c102d10c52de93a486180aa73372c` | strict cohort |
| V4fg 10k | `ckpts/plus_ja_sft_v4fg/step_010000.pt` | `a9d8a05679d413b11c75c983b24118531e193742bb691a1dc25e9f701daeb891` | reference cohort |

M600-D 30k 原始产物是四 rank FSDP DCP。发布时只使用 GPU set 和独立 torchrun 端口，未占用或
操作 GPU set 的 M600-B 训练。发布审计确认：

- `global_step=30000`；
- online/EMA 均 strict-load；
- 所有 tensor 有限；
- DiT state elements 为 `602,599,648`；
- 发布 checkpoint schema 为 `v4m_m600d_publish_checkpoint_v1`。

## 实现

代码位于：

```text
experiments/v4m_m600d_20260804/a_condition_r2/source/
```

新增或扩展：

- `eval_a_condition_model.py`：支持 HighLR、V4PH、M600-D publish 和 V4fg legacy EMA；
- `build_a_condition_stimuli_v4fg.py`：构造 285k VAE/SOFA/SOME 独立缓存；
- `analyze_flowa_cf_multimodel.py`：支持任意模型数的 cohort 聚合；
- `run_flowa_cf_30k_four.sh`：GPU 门禁、smoke/full、压缩和 SHA256: redacted

多模型聚合器只读取绝对 A/B 距离、配对 ΔLoss、CKA/F0、prediction-target 统计和分层结果。即使
历史 pair CSV 仍包含旧 `gain_*` 列，聚合时也会显式删除，不进入报告。

## Smoke

smoke 使用前两条冻结样本，完整运行五种 A、五个 timestep 和全部 latent/mel/BYOL/F0 特征：

| cohort | 模型数 | forward rows/模型 | pair rows/模型 | 结果 |
|---|---:|---:|---:|---|
| strict | 3 | 50 | 100 | 通过；跨模型 invariant mismatch 为 0 |
| V4fg reference | 1 | 50 | 100 | 通过；独立缓存与 EMA 前向/聚合完成 |

smoke 首次执行依次暴露并修复了 stimulus SHA 抄写、MelodySpectrogram device、inference tensor
clone 和新旧 MIDI transport 守卫问题。所有问题都在正式运行前触发门禁；失败运行未进入结果聚合。

## 正式运行

服务器目录：

```text
${SERVER_ROOT}${CLOUD_ARTIFACT}
```

tmux session

```text
flowa_cf_30k_four_full_v2_20260806
```

只使用物理 GPU set。GPU set 的 M600-B 正式训练保持隔离，GPU set 未使用。正式运行先依次评测
strict cohort，再聚合严格比较，然后构建 V4fg 独立全量缓存并评测参考 cohort。

正式运行退出码为 `0`。四模型各完成 `5,025` 条 forward 和 `10,050` 条 pair；strict cohort
跨模型 invariant mismatch 为 `0`。V4fg 独立缓存为 `180,491,206` bytes，SHA256: redacted
`007c840e7d8b244f2d5cb98e2d3466388a45c15f441d83bca3050738b337ff7c`。新正式 pair CSV 不再
包含任何 `gain_*` 字段。

| 模型 | 前向耗时 | max reserved |
|---|---:|---:|
| HighLR 30k | 5,138.8s | 8.90GiB |
| M600-D 30k | 5,237.9s | 9.07GiB |
| V4PH 30k | 5,058.7s | 8.90GiB |
| V4fg 10k | 5,119.4s | 8.06GiB |

服务器压缩证据已同步到：

```text
${LOCAL_PROJECT_ROOT}\experiments\v4m_m600d_20260804\a_condition_r2\30k_four_20260806
```

本地对 51 个服务器 SHA256: redacted`0`。逐样本 sample-level bootstrap 使用
seed `20260806`、20,000 次重采样，结果保存在 `posthoc/`。

首轮通用聚合器的 `difficulty_summary.csv` 错用了四分位；模型前向、原始行和其他聚合不受影响。
最终难度结论由修正后的五分位 posthoc 产生，未来通用聚合器源码也已改为 Q1--Q5。四分位文件仅
保留作运行审计，不进入最终裁决。

## 严格 cohort 结果

正确 A 总表：

| 模型 | FlowB | CKA | 完整 Loss | pooled R2 | corr | slope | std ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| HighLR 30k | 0.86212 | 0.03408 | 1.75309 | 0.59814 | 0.77341 | 0.60146 | 0.77768 |
| M600-D 30k | 0.86022 | 0.03272 | 1.74721 | 0.59906 | 0.77400 | 0.60203 | 0.77781 |
| V4PH 30k | 0.86586 | 0.03528 | 1.76247 | 0.59640 | 0.77228 | 0.59945 | 0.77622 |

M600-D minus HighLR 的逐样本均值与 95% bootstrap CI：

- FlowB `-0.001903 [-0.002443, -0.001368]`；
- CKA `-0.001356 [-0.001716, -0.001041]`；
- 完整 Loss `-0.005884 [-0.007031, -0.004788]`；
- identity R2 `+0.000890 [+0.000644, +0.001139]`；
- correlation `+0.000576 [+0.000422, +0.000733]`；
- slope `+0.000524 [+0.000344, +0.000709]`；
- std ratio `+0.000114 [-0.000054, +0.000281]`。

Loss、R2、相关和斜率同步改善，方差比没有进一步收缩。两模型绝对方差比都只有约 `0.778`，仍有
共同的 under-dispersion，但 M600-D 的低 Loss 不是通过新增方差收缩或忽略 A 获得。

M600-D 对 same/near/far/zero 的 ΔFlowB 相对 HighLR 分别增加 `0.000592/0.000973/0.000626/
0.000481`，95% CI 均略高于 0；四种 A 的 B latent 和 log-mel 响应也都小幅增加。自然错误 A 的
额外 ΔCKA 与 F0 响应均跨过 0，只有 zero A 的 ΔCKA 有小幅确定增加。因此它支持略强 A 条件利用，
不支持新出现的 A-to-melody 纠缠。

该增强不是更好的单维音色距离标定：far-minus-near ΔFlowB 虽仍为正，但样本排序率从 HighLR 的
`63.7%` 变为 M600-D 的 `60.2%`；BYOL A 音色距离与 ΔFlowB 的逐样本线性相关接近 0。

按 timestep，M600-D 的 low 桶 FlowB 改善 CI 包含 0，高/very-high 桶最稳定；收益不集中在低 t
高不确定区。按协议使用 HighLR 难度五分位，Q5 hardest 的 FlowB/完整 Loss 改善为
`-0.003234/-0.008907`，Q1 easiest 为 `-0.001071/-0.003860`。Q5-minus-Q1 的额外完整 Loss
改善为 `-0.005046 [-0.009228, -0.001369]`，支持困难尾部存在小幅但有价值的 scaling 收益。

## V4fg 参考结果

V4fg 内部对 same/near/far/zero 的 ΔFlowB 为 `0.02302/0.01772/0.02424/0.00950`，明确没有忽略
A。far-minus-near 为 `+0.006526 [0.004172, 0.008934]`，`70.1%` 样本方向为正。换 A 后 ΔCKA
系统性为正，低 t 更明显，说明旧 SOFA/SOME 条件链内存在一些 A-to-structure 交互；F0 变化仍小。
V4fg 的 VAE、文本、MIDI 和 CKA target 与 strict cohort 不同，不能从其绝对 Loss、R2 或 ΔCKA
数值直接宣称优劣或纠缠更强。

## 裁决

M600-D 30k 相对 HighLR 30k 是真实但很小的预测改善，不是平均化收益。它对 A 的错误/缺失略更
敏感，B 声学响应略强而自然 A 的旋律泄漏没有增加；可以表述为略强且基本有效的 A 利用，但不能
表述为音色选择机制发生质变或 near/far 标定显著改善。

容量收益在困难样本尾部更大，但不集中于低 timestep。V4fg 保留为独立参考 profile，不进入严格
Loss 排名。

## 解释门禁

最终解释按以下顺序：

1. strict cohort 的正确 A FlowB、CKA、完整 Loss、R2、斜率和方差比；
2. same/near/far/zero 各自的 ΔFlowB、绝对 B latent/mel/timbre/F0 响应；
3. A 距离与 ΔFlowB/B 响应的有序性，以及低 timestep；
4. HighLR 冻结难度分桶下的普通样本和困难尾部；
5. V4fg 只在自身语义内解释 A 选择性和泄漏，再与 strict profile 做定性参考。

不得从 V4fg 与 strict cohort 的绝对 Loss 差异直接宣称模型优劣，也不得从单一 zero A、单一 R2 或
单一输出距离作因果裁决。

















