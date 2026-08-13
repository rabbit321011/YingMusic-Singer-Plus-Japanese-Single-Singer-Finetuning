# 与 V4fg 音色对照

## 1. 感知结论

固定 27 组横向试听补充结论：

```text
V4PH 的音色远不如 V4fg。
```

因此 V4PH 不能继续笼统称为总体听感主候选。当前应按维度定位：

- V4fg 10k + 285k online VAE：当前音色基准；
- V4PH 30k：H/PUL 与 GAME MIDI_P 结构控制成立的候选，但音色显著落后；
- V4IjPH 30k：仅较 V4IPH 轻微改善，且仍不及 V4PH，不晋级。

## 2. 因果边界

V4PH 与 V4fg 的直接比较不只包含 H/P 差异，还同时包含：

- official VAE vs 285k online VAE；
- Official base fresh joint training vs 既有路线上的低 LR `g` 适配；
- H phone/PUL 与 GAME MIDI_P 的新增条件；
- 训练谱系与优化阶段差异。

因此当前只能确认“V4PH 音色显著落后 V4fg”，不能仅凭该听感把差距归因给 H placement、
GAME P、official VAE 或某一个独立模块。

## 3. 路线影响

- V4PH 继续作为 H/P 结构控制基座，不再作为总体音色主线；
- 基于 V4PH 的 V4KPH 即使补回 A 信息，也不能被假定会自动达到 V4fg 音色；
- 在投入新的 KPH 长训前，应优先利用现成 V4Hg 与 V4fg/V4PH 的同组结果判断 285k VAE + `g`
  适配能追回多少音色；
- 若仍需保留 PH 的 H/P 结构并追求 fg 音色，V4PH 索引已预留的正式命名是 `V4PHg`，但是否
  实现和训练必须另行授权。

用户当前决定不继续展开这些 PH 家族补丁路线，先等待 M 分支。V4PHg 与 KPH 均保持未授权。

训练 loss、FlowB 与 CKA 不能替代该音色裁决。

















