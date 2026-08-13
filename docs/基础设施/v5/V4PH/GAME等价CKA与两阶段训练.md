# GAME 等价 CKA 与两阶段训练

## 目标

以 GAME 完整替换 SOME，同时保持 CKA 消费接口不变。训练代码只接收统一的
`[B,T,128]` 音高概率数组，不包含 GAME/SOME 分支判断；GAME 与 SOME 对同一干净样本的输出
应具有相近的音高中心、概率形状、REST 和边界结构。

V4PH 从官方 base model 初始化，同时启用 H phone/PUL 文本放置和 GAME MIDI_P，不继承任何
既有微调 checkpoint。

## 双输出语义

同一次冻结 GAME 推理产生两条用途严格分离的输出：

```text
durations / presence / scores
  -> 0.5-semitone P class
  -> pitch 0..254 / REST 255 / PAD 256
  -> learned P embedding
  -> DiT MIDI condition

estimator pitch logits [N,257]
  -> sigmoid probabilities
  -> SOME-compatible pitch-axis and time-axis adapter
  -> [T_some,128] probability map
  -> CKA target only
```

GAME hidden feature不得进入 DiT 或 CKA。CKA 分支保留的是最终 estimator 音高概率，不是
encoder、segmenter 或 estimator hidden state。

## SOME-compatible 合同

SOME 官方连续模型使用 MIDI `0..127` 的 128 个整数半音 bin，训练 target 是标准差
`1.0` 半音的 Gaussian probability map，REST 为全零。GAME 使用 MIDI `0..128` 的 257 个
0.5 半音 bin，训练 target 标准差为 `0.5` 半音。

GAME adapter 必须输出：

```text
pitch_probs:    float [B,T_some,128], finite, range [0,1]
boundary_probs: float [B,T_some,1],   finite, range [0,1]
```

`T_some` 与同一音频经过官方 SOME `44.1kHz / hop=512 / center=True` 前端得到的帧数一致。
每个 SOME 帧按中心时间查询所属 GAME note；REST 写全零。pitch-axis 转换必须保留 GAME 的真实
概率分布，并校准 0.5/1.0 半音核宽差异，不允许只用最终 `score` 重新伪造概率后丢弃 estimator
不确定性。

正式 CKA 只读取 `pitch_probs`。SOME 只用于离线等价门禁，不进入 V4PH cache、训练或推理。

## 接入正确性门禁

同一批音频分别运行正确官方前端的 SOME 与固定 seed 的 GAME。比较前先映射到完全相同的
`[T_some,128]` 合同，再分别记录：

- shape、dtype、时间闭合、有限值与 `[0,1]` 范围；
- mutually voiced 帧的音高中心绝对误差；
- 128 维 probability vector cosine 与 linear CKA；
- 每帧 peak、L1 norm、背景概率和 entropy 分布；
- voiced/REST 一致率；
- boundary 在固定容差内的匹配率。

target singer干净样本用于裁决 adapter 是否正确。非target singer MSST 困难样本以及已知 SOME 局部低音崩坏
片段单独报告：GAME 在这些位置纠正 SOME 时不计为 adapter 失败。不得用全局平均掩盖分域差异。

硬门禁先冻结为结构与数值不变量；音高、概率和边界相似度阈值必须依据 62 条真实基线分布再
冻结，不能在看到数据前拍脑袋选择。adapter 候选至少比较直接整数 bin 采样与保留 posterior
的核宽校准方案，选择必须同时满足理论语义和真实 SOME 相似度。

## 首轮等价审计结果

固定 GAME medium K=4、per-audio seed 与正确 SOME `40-8000Hz` 前端，在既有 62 条同源音频
上比较三种 pitch-axis adapter：

1. 保留 GAME posterior，直接取偶数 bin `0,2,...,254` 对应 SOME MIDI `0..127`；
2. 保留 posterior，先从 GAME `sigma=0.5` 展宽至 SOME `sigma=1.0` 再取整数 bin；
3. 丢弃 posterior，仅从 decoded score 重建 SOME 标准核。

直接偶数 bin 在 probability cosine 和 CKA 上均优于额外展宽，并略优于 decoded-score 重建；
这说明 GAME 实际 estimator posterior 已与 SOME 输出宽度接近，不应依据训练 target sigma 再做
机械展宽。正式映射冻结为：

```text
sigmoid(GAME estimator logits)[..., 0:255:2]
REST -> zero
```

20 条target singer干净样本的直接映射结果：

| 指标 | 结果 |
|---|---:|
| 平均 sample-level pitch median error | 0.0952 semitone |
| 最差 sample-level pitch median error | 0.2216 semitone |
| 平均 probability cosine | 0.9616 |
| 最低 probability cosine | 0.8826 |
| 平均 same-pitch probability cosine | 0.9937 |
| 最低 same-pitch probability cosine | 0.9845 |
| 平均 linear CKA | 0.9269 |
| 最低 linear CKA | 0.7002 |
| 平均 voiced/REST agreement | 0.9544 |
| 平均 boundary precision / recall | 0.8213 / 0.8455 |

概率尺度同样闭合：target singer SOME/GAME 的平均 peak 为 `0.9095/0.9433`，L1 norm 为
`2.4070/2.4896`，entropy 为 `1.4427/1.4370`。62/62 shape 与 SOME 帧数完全一致，无非有限值
或范围违规。

非target singer 42 条中 N012、N021、N032 存在显著 teacher pitch 分歧；其余大量帧在音高一致时的
probability cosine 仍很高，非target singer总体 same-pitch cosine 为 `0.9778`。这些样本属于此前已知的
复杂源 teacher 质量差异，不推翻 adapter 轴与尺度正确性。

据此冻结首轮回归门禁：target singer 20 条必须保持 62/62 结构审计通过，平均 pitch median error
不高于 `0.15` 半音、平均 probability cosine 不低于 `0.94`、平均 linear CKA 不低于 `0.90`、
平均 boundary precision/recall 均不低于 `0.80`。困难域继续单列，不用 SOME 错误约束 GAME。

完整报告：`TEMP/v4ph_game_some_cka_equivalence_20260730_v1.json`。

## 结构化音高核

Official-base 兼容音高核采用 SOME 官方训练定义：

```text
centers = MIDI 0..127
sigma = 1.0 semitone
kernel[i] = exp(-0.5 * ((i - midi_pitch) / sigma) ** 2)
REST/PAD = zero
```

它只作为 P 初始化诊断和候选初始化，不是 V4PH teacher。随机 P 使用固定 seed，并匹配音高核
整体均值和方差，但不保留音高位置或相邻关系。

阶段 A 在 step 0/100/300/500 报告 pitch rows 的 raw distance、cosine distance、冻结
`midi_proj` 后距离和整表 pitch-geometry CKA。REST 单独报告，PAD 固定为零。

由于 CKA target 本身与 SOME 官方音高坐标同构，阶段 A 的目标就是判断随机 P 能否在冻结
Official base 下学入该坐标。低训练步数不做听感裁决，正式规则为：

```text
primary:   D(midi_proj(P500), midi_proj(kernel))
           < D(midi_proj(P300), midi_proj(kernel))
support:   raw/cosine distance 同向改善
           pitch-geometry CKA 同向提高
```

主指标与支持指标持续改善时，随机初始化通过，阶段 B 使用 P500；主指标不改善或支持指标明显
冲突时，随机初始化失败，改用结构化音高核初始化并重新完成短校准。该结论只表示接口校准
成功，不对 500-step 听感或最终质量作声明。

距离只在阶段 A 实际获得训练支持的 pitch rows 上计算，并同时报告每行出现次数。零支持 rows
没有随机初始化有效性的证据；进入阶段 B 前统一以结构化音高核填充，REST 单独训练，PAD 固定
为零。这样不会让训练音域外的随机向量进入正式推理。

## 阶段 A

```text
init: Official base
text: H phone/PUL placement enabled; PUL row copied then frozen
trainable: P embedding only
loss: FlowB + 0.7 * GAME-compatible CKA
dropout/CFG: disabled
LR: 1e-4
steps: 500
checkpoints: 0 / 100 / 300 / 500
```

CKA 使用 B 区有效时间，A 区与 PAD 不参与。P embedding、loss 和 distance 均有明确学习趋势
后才允许进入阶段 B；阶段 A 不设置听感门禁。

## 阶段 B

```text
init: selected phase-A model weights
trainable: Official DiT + H/PUL + P embedding
loss: FlowA + 2 * FlowB + 0.7 * GAME-compatible CKA
LR: 1.4e-5
warmup: 500
steps: 30000 from a new step 0
VAE: official frozen VAE
```

阶段切换丢弃 P-only optimizer/scheduler，按当前模型权重重建 EMA。训练与推理使用同一 GAME
cache schema、P adapter 和 checkpoint schema。

## 停止条件

- GAME/SOME 统一接口的 shape、时间、范围或有限值不一致；
- 干净target singer样本在排除真实 teacher 分歧后仍出现系统性音高偏移或概率轴错位；
- CKA 消费代码包含 teacher-specific 分支；
- P embedding 未进入 optimizer/checkpoint，或阶段 A 修改了冻结参数；
- 训练与推理使用不同 P class、REST/PAD 或时间对齐；
- loss、CKA、梯度或 checkpoint 出现非有限值。

















