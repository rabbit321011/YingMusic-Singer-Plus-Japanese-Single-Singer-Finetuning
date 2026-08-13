# V4Pf 分支

本分支验证在保留 YingMusic-Singer-Plus 官方 base model 的条件下，
从训练 step 0 起将连续 fuzz MIDI 替换为音符化 MIDI_P，并使用 V4f 的 SOFA 配方完成训练。

## 边界

- `P` 表示 MIDI_P 从本分支 step 0 起参与训练，不是从既有 V4f checkpoint 后切换。
- DiT 从官方 base model 初始化，不引入随机初始化 `v`。
- 保留 V4f 的 SOFA token、A/B 训练结构、官方 VAE、数据清单和其余控制变量。
- SOME teacher 保持冻结；其输出经过 note pitch/duration 解码、帧展开和离散 embedding 后送入 DiT。
- 训练与推理必须使用同一 MIDI_P 通路。
- 本分支不实现 `V4Pvf`；`V4Pvf` 是 `v + P` 从 step 0 联合训练的独立分支。

## 与其他分支的关系

```text
V4vf：随机初始化可行性门禁
  ├─ 可行：允许启动独立 V4Pvf
  └─ 不可行：V4Pf 成为 P 主路线

Official base + MIDI_P + V4f/SOFA → V4Pf

V4Pvf / V4Pf → V5 → V5-MoE
```

V4Pvf 已因随机初始化路线的感知门禁失败而停止。V4Pf 不再承担与 V4Pvf 严格单变量
对照的职责，而以 Official base 兼容性和最终可用性为优先；已经验证的量化、REST 和
checkpoint 通路仍可作为工程证据复用。

## 冻结训练方案

```text
阶段 A：Official DiT 全冻结
        只训练 P embedding，LR=1e-4，目标 500 步
        关闭 audio/text/MIDI CFG dropout

阶段 B：加载阶段 A 的模型权重
        重建 optimizer，并以当前模型重建 EMA
        Official DiT + P embedding，统一 LR=1.4e-5，联合训练 30k
        恢复 V4f 的 CFG/dropout 配方
```

阶段 A 从第一次参数更新起即使用 P；不存在先使用连续 MIDI 训练 `f` 再切换 P 的阶段，
因此仍属于 `V4Pf`，不是 `V4fP`。500 步是首轮门禁而非机械固定值：在 100/250/500
步保存和真实推理，若 500 步仍持续改善可最多延长至 1k；若无学习趋势则先审计接口，
不直接延长。

P embedding 的初始化尚未最终冻结。结构化标准音高模板是候选启动方式，不对target singer同音高
的全部演唱帧做未经处理的粗均值；它必须与随机初始化完成 P-only 短对照后才能晋级。每个
class 仍只有一个共享音高原型；颤音、滑音和微观音准不进入 P 输入，由 DiT 从上下文、噪声
和target singer目标数据中学习。raw SOME CKA 在正式联合阶段保留，但不作为推理输入；P-only 校准
阶段是否启用 CKA 尚待 SOME 门禁后单独裁决。

## 当前状态

- 文档分支已建立。
- 首轮官方 MIDI_P 静态接口审计已完成，确认原路径存在 rest mask 丢失、离散 pitch 线性插值、特殊类别不明确和 checkpoint 静默漏载风险。
- MIDI_P 的 257 类、REST/PAD、0.5 半音量化、voiced-mask 时间对齐、A/CFG null、raw SOME CKA 和 checkpoint 纪律已冻结。
- V4vf/V4vfg 感知门禁已失败，V4Pf 不再等待随机初始化路线，成为 P 的 Official-base 回退主线。
- P-only 500 步校准再联合 30k 的两阶段训练方案已冻结；阶段切换时重建 optimizer 和 EMA。
- SOME→P v1/v2 因 Mel 范围错误作废；修正 v3 的整体音高正确，但用户试听发现偶发局部低音
  崩坏。OpenVPI/GAME medium K=4 已在同一 62 条完成本机推理、P 映射、钢琴渲染和工程审计，
  用户试听结论为基本可用，允许晋级正式 P teacher。
- P 训练主线已与 H 合并为 `V4PH`：从 Official base 同时启用 H phone/PUL placement 与
  GAME MIDI_P。V4Pf 保留为 teacher 调研和工程证据，不再单独启动训练。

## 工人

| 文件 | 内容 |
|---|---|
| `MIDI音符化实验.md` | V4Pf 实验定义、MIDI_P 接口审计、控制变量、执行阶段与验收标准 |
| `SOME音符化试听验证.md` | SOME→P 测试集、双时间轴正弦渲染、产物审计和人耳记录入口 |
| `GAME接入设计.md` | GAME 官方接口、P schema 映射、离线缓存、运行边界与同源 A/B 门禁 |
| `GAME试听验证.md` | GAME medium K=4 的 62 条同源探针、工程审计、SOME/GAME 人耳入口与裁决 |

## 阅读顺序

先读 `MIDI音符化实验.md`。实验推进和结论持续记录在该工人文件中。

















