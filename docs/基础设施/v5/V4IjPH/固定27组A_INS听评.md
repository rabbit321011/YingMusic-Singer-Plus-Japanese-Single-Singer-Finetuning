# 固定 27 组 A-INS 听评

## 1. 评价契约

沿用既有 `测验集合成选择2` 的 27 组 A/B 输入，不新增参考集合，也不定义 R1/R2：

```text
每组 A.wav -> frozen ParaSpeechCLAP Intrinsic -> adapter EMA -> full-timeline cond
每组 B.wav -> GAME MIDI_P
每组 B 文本 -> H/PUL
模型 -> 完整 B 时间轴生成结果
```

每组现成的 `A.wav` 是该组唯一 INS 参考。A 使用完整单声道波形，不裁剪、不聚合、不替换；
A 不进入 VAE、不占生成帧、不产生 prompt 区或 reference tail。`A_T1.json` 不参与 V4IjPH。

## 2. 固定参数

- checkpoint：V4IjPH v2 step 30,000 final；
- checkpoint SHA256: redacted`0dfa31c293d3558dbe28868edb66ecf2959a5b92bbdbc96f0819222df351c2e9`；
- adapter：EMA；
- VAE：official 20 Hz；
- Steps：32；
- Seed：42；
- CFG：3.0 与 1.0；
- style guidance：1.0；
- 输出：44.1 kHz、单声道 PCM16 WAV。

CFG 的 conditional 与 content-unconditional 两支保留相同 style，只对 H/PUL 文本与 GAME
MIDI_P 内容做 guidance。

## 3. 本机产物

评测包：

```text
${LOCAL_EXPORT_PATH}
```

新增：

```text
_v4ijph_a_ins/
V4IjPH_30k_cfg3/
V4IjPH_30k_cfg1/
```

`_v4ijph_a_ins/manifest.json` 保存逐组 A 文件 SHA、完整波形时长与 SHA、INS SHA/范数；两个结果
目录的 `_placement/` 保存逐组 A/INS/style/full-timeline cond、H/PUL 与 GAME 审计。

本机批处理入口：

```text
${LOCAL_EXPORT_PATH}
${LOCAL_EXPORT_PATH}
```

## 4. 完整性结果

- A-INS：27/27 成功，矩阵 `[27,768]` FP32，逐行 L2 为 `0.99999994..1.0`；
- 生成：CFG 3 与 CFG 1 各 27 条，共 54 条；
- 审计：54/54；
- 解码失败、格式错误、非有限或全静音：0；
- 全部 44.1 kHz、单声道 PCM16；
- 相对 B 的最大时长偏差：`0.0214058957` 秒；
- 同组 CFG 1/3 WAV 哈希相同：0；
- 两个 CFG 间 A、INS、style、cond、H/PUL、GAME P/MIDI 审计漂移：0；
- style L2 范围：`0.0981473..0.3950762`；
- GAME target PAD frame：全组 0。

27 组中有 25 个唯一 A 文件 SHA；重复项来自评测集本身复用同一完整 A，因此对应 INS/style 相同，
不是缓存串组。

## 5. 听感裁决

优先对同名文件横向比较：

- V4IjPH 30k vs V4IPH 30k：判断每组 A-INS 是否带来可感知且合理的唱法变化；
- V4IjPH 30k vs V4PH 30k：判断 INS 全局参考能否恢复 A 区参考带来的有效信息，同时避免其噪音代价；
- CFG 3 vs CFG 1：判断内容 guidance 对咬字、旋律与发声区纹理的影响。

最终晋级仍由固定同名 target 的人耳感知决定，训练 loss 与 FlowB 只作为健康指标。

## 6. 感知裁决

固定 27 组同名横向试听结论：

```text
V4IjPH 相对 V4IPH 有轻微改善，但仍然不及 V4PH。
```

据此裁决：

- INS adapter 不是完全无效；A 的全局 INS 条件产生了可感知的正收益；
- 该收益不足以补回 V4PH 移除 A 区后损失的参考信息；
- 单一全局 `768 -> 64` style broadcast 不能替代 V4PH 的细粒度 A 声学参考；
- V4IjPH 不晋级；在 PH 家族内部 V4PH 仍优于 V4IjPH，但后续跨分支听评确认 V4PH 音色远不如
  V4fg，不能把 V4PH 称为总体音色主候选；
- 该结果为 V4KPH 的“保留细粒度 A 信息但不让 A 占生成时间轴”假设提供了继续验证的理由，
  但不等同于授权启动 V4KPH。

















