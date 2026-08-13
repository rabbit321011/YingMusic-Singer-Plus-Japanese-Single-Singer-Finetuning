# HighLR 听评后的来源更新

## 新证据

`V4PH-30K-HIGHLR` 保持历史 V4PH 的 65H 数据、Phase A 与 30k 总步数，仅把 Phase B 峰值
`1.4e-5` 学习率保持到 global step 24k。其 final Eval Loss 为 `2.0841`，低于原 V4PH 的
`2.0937`。

用户完成直接听评后的音色裁决为：

```text
V4fg > V4PH-30K-HIGHLR > V4PH
```

其中 HighLR 只略微强于 V4PH，仍弱于 V4fg。该结论只覆盖音色维度，不宣称 HighLR 的歌词控制、
旋律跟随、稳定性或总体听感已经全面晋级。

## 对 M 来源的影响

- 若 M 首轮以音色为最高优先级，V4fg 仍是领先来源；HighLR 没有消除 V4fg 的音色优势；
- 若 M 首轮以保留 H/PUL 与 GAME MIDI_P 控制接口为最高优先级，HighLR 可替代原 V4PH 成为
  更强的控制路线来源候选；
- 函数保持扩模仍只能继承一个 checkpoint，不能把 V4fg 音色与 HighLR 控制接口自动合并；
- HighLR 的小幅 loss 与音色收益不足以单独冻结 M 来源，正式来源仍取决于 M 的首要目标；
- 本证据不构成实现转换器、占用 GPU 或启动 M 训练的授权。

## 当前裁决

M 的候选来源从“V4fg 或原 V4PH”更新为“音色优先的 V4fg，或控制优先的
V4PH-30K-HIGHLR”。首要目标与单一来源仍未冻结。

















