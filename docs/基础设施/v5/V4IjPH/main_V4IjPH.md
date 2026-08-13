# V4Ij(ins)PH 分支

本分支在 V4IPH 的无 A 区任务上，将原本全零的 64 维 `cond` 改为 INS 全局唱法特征，
验证target singer参考样本能否在不恢复 A 时间轴的前提下控制演唱风格。

## 命名

```text
V4 + I + j(ins) + P + H
```

- `I`：A 区不存在，`ref_len=0`，完整时间轴均为生成区；
- `j(ins)`：把 I 路线空出的 `cond` 通道改为 INS 特征注入；
- `P`：沿用 V4PH 的 GAME MIDI_P；
- `H`：沿用 V4PH 的 H/PUL 歌词放置。

## 边界

- 非 I 分支不实例化、不训练也不推理 INS 注入；
- V4IPH 保持 `cond=0`，作为本分支的正式 null 基线；
- 训练时 INS 从target singer target B 提取；
- 推理时 INS 从用户选择的独立target singer唱法样本 R 提取；
- 推理旋律源 B 不提供 INS，只提供 GAME MIDI_P 等既有旋律条件；
- R 不是 A 区：不进入 VAE、不占生成时间轴、不产生 reference tail；
- 不依赖 MERT、INS 或其他 embedding 的自动聚类与语义命名；
- 缺少 INS 或执行 INS dropout 时，`cond` 严格为零并退化到 V4IPH 条件格式。

## 当前状态

- v1 实现已于 2026-08-01 审计否决：10-step smoke 未激活 adapter，正式 run 约 step 1250
  主动停止；该实现会在 step 8000 的 CPU/CUDA `torch.equal` 门禁必然失败；
- v1 源码与日志作为失败证据保留，不允许训练或 resume；
- v2 改为直接从已审计的 V4IPH step 8000 完整状态精确续接，不重复执行前 8k；
- v2 cache schema、训练/checkpoint schema、active-adapter 单卡/四卡门禁、exact resume 比较和
  全时间轴 style ODE/CFG 采样入口已完成重写；
- cache v2 的 private corpus count train + 201 eval 全量构建、inventory 与 waveform rehash audit 通过；
- 单卡 8001/8002、四卡 active 8010、checkpoint audit、连续 10 对 5+resume5 bit-exact 和真实
  双 R/null INS 推理门禁均已在服务器通过；
- exact-resume 比较器的 NumPy RNG array 缺口已补测试修复，并在新 provenance 下完整重跑通过；
- 正式 v2 四卡训练已从 V4IPH step 8000 完成至 global step 30000，final INS/null Eval
  Loss 为 `1.8123/1.8152`、FlowB 为 `0.8711/0.8725`；final checkpoint 独立 audit 全通过，
  checkpoint 与 SHA256: redacted`${CLOUD_ARTIFACT}`；既有 27 组评测已按每组
  `A.wav` 为唯一 INS 参考生成 CFG 3/1 共 54 条并通过完整性审计。感知裁决为相对 V4IPH
  轻微改善、仍不及 V4PH；V4IjPH 不晋级。进一步跨分支听评确认 V4PH 音色又远不如 V4fg，
  因此该排序只表示 PH 家族内部结果，不把 V4PH 称为总体音色主候选。

## 工人

| 文件 | 内容 |
|---|---|
| `INS全局唱法条件.md` | v1 设计与实现历史，现已封存为审计背景 |
| `V2重写与执行门禁.md` | v1 否决证据、v2 精确 transition、cache/provenance、训练/推理和执行门禁 |
| `固定27组A_INS听评.md` | 既有 27 组每组 A 为唯一 INS 参考的 CFG 3/1 听评包与完整性结果 |

## 阅读顺序

先读 `INS全局唱法条件.md` 了解原始研究问题，再读 `V2重写与执行门禁.md`，最后读
`固定27组A_INS听评.md`。不再修改 v1 历史结论。

















