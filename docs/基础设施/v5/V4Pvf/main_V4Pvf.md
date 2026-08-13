# V4Pvf 分支

本分支验证随机初始化 DiT 与量化 MIDI_P 能否从 step 0 联合训练出有效的日语歌声、
旋律控制和target singer演唱表现。V4vf 只证明训练指标可学习，后续感知门禁已经失败；
V4Pvf 是在该最终结论产生前经用户独立授权启动的实验，必须由自己的最终感知结果裁决。

## 边界

- `v + P` 同时从 step 0 生效；不继承 V4vf、V4f、V4fg 或 V4Pf 的 DiT 权重。
- 使用 V4f 的 SOFA token、A/B 训练结构、官方 VAE、原始 SOME CKA 和其余数据条件。
- MIDI 输入改为 `0.5` 半音量化、独立 REST/PAD、可学习 128 维 embedding。
- 不使用 `MIDIFuzzDisturb`，保留整条 `drop_midi` CFG。
- 训练与推理必须使用同一量化器、类别 schema 和 embedding checkpoint。
- 只新增或复制带 `v4pvf` 标识的文件，不修改既有 V4f/V4vf 或公共源码。

## 路线

```text
V4vf：Random DiT + continuous fuzz SOME，证明训练指标可学习但感知失败
  └─ V4Pvf：Random DiT + quantized MIDI_P，从 step 0 联合训练
       └─ 10-step smoke → 一次真实推理 → 30k 正式训练
```

V4Pvf 不是从 V4vf checkpoint 后切换 P。V4vf 结果只用于确认随机初始化路线和首轮
`LR=1e-4` 的合理性。

## 当前状态

- 文档分支已建立。
- 独占量化器、Singer 副本、训练入口、推理入口、配置、launcher 和探针已实现。
- 合成单元检查、Linux 静态检查、真实target singer SOME 探针和模型结构审计已通过。
- 4 卡 10-step smoke 与 smoke checkpoint 的一次真实 32-step 推理均通过。
- 30k V4Pvf 曾从随机 step 0 启动，数值稳定运行至约 21.15k；V4vf/V4vfg 的大量噪音使随机初始化感知门禁失败后，用户要求主动终止。
- `v4pvf_30k` tmux session 的 10 个 checkpoint 已按用户要求删除，训练日志和工程验证证据保留。
- 本分支没有 30k 最终 checkpoint 或最终质量结论，路线停止；P 后续转入 Official base 起点的 V4Pf。

## 工人

| 文件 | 内容 |
|---|---|
| `量化MIDI联合训练.md` | 设计冻结、阶段记录、验证证据、训练与推理结果 |

## 阅读顺序

先读 `量化MIDI联合训练.md`。每完成一个阶段，先在该文件记录证据并复核分支定义，再进入下一阶段。

















