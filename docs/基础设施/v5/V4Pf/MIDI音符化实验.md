# MIDI 音符化实验

## 目标

验证将当前连续 SOME melody representation 替换为音符级 MIDI_P 后，是否能减少
原唱微观音高、颤音和滑音信息泄漏，使 DiT 更依赖离散音符控制并自行学习target singer唱法。

本实验对应 V4Pf：DiT 从官方 base model 初始化，MIDI_P 从训练 step 0 起启用，
使用 V4f 的 SOFA 数据与训练结构，不继承 V4f checkpoint。

## 唯一主变量

```text
V4f：  Official base + continuous SOME fuzz MIDI + SOFA + official VAE
V4Pf： Official base + note MIDI_P            + SOFA + official VAE
```

第一轮保持数据清单、A/B 区、SOFA token、VAE、模型规模、seed、batch、训练步数、
loss 权重和评测协议不变。若 MIDI_P 需要独立优化策略，须在完成同配方对照后另开实验记录。

## MIDI_P 预期通路

```text
audio
  → MelodySpectrogram
  → frozen SOME MIDI teacher
  → midi logits + boundary logits
  → note boundary / pitch / duration decoding
  → frame expansion with explicit rest and padding semantics
  → learned discrete MIDI embedding [B, T, 128]
  → DiT MIDI input
```

官方源码已有 `MIDIExtractor.postprocess(..., with_expand=True)` 和
`MIDIDigitalEmbedding`，但当前 V4f 训练脚本直接对连续 128 维 SOME 输出执行
`MIDIFuzzDisturb`。V4Pf 不能只改配置字段，训练、评测和正式推理路径必须同时切换。

## 接口审计门禁

正式训练前必须确认：

1. note boundary、pitch、duration 解码后的长度与原 SOME 帧数关系正确；
2. `note_mask_pred` 不被无声丢弃，rest 不与合法 MIDI 0 或 padding 混为同一类；
3. A 区 MIDI 清零与 CFG `drop_midi` 使用一致的空条件语义；
4. 时间重采样中 REST/PAD 不参与音高插值；有声音高只允许在 voiced mask 保护下平滑并重新量化；
5. `MIDIDigitalEmbedding` 的类别范围、rest/pad class 和输出维度明确；
6. raw SOME CKA 只作为训练辅助监督，不进入 MIDI 条件或推理输入，并且不得反向更新冻结 SOME teacher；
7. checkpoint 明确记录 MIDI_P 配置和 embedding 权重，推理加载不得静默缺 key。

任一项未通过时，不启动长训练。

## 与 V4Pvf 的关系

V4Pf 与 V4Pvf 都从 step 0 使用相同的 MIDI_P 实现：

```text
V4Pf：  Official base + P + f
V4Pvf： Random DiT   + P + f
```

`V4vf` 只负责判断随机初始化是否可行，不向 V4Pvf 传递权重。V4Pf 不等待 V4vf
才能做接口审计和 smoke test；是否将 V4Pvf 纳入主线由 V4vf 门禁结果决定。

## 执行阶段

### 阶段 A：离线 MIDI_P 探针

- 选择包含正常音符、滑音、颤音、短休止和长静音的固定样本；
- 保存原始 SOME logits、解码 note 序列、展开帧序列和类别统计；
- 检查长度、音高范围、休止比例、边界位置与确定性；
- 对相同输入重复运行，结果必须一致。

### 阶段 B：P-only 接口校准

- 从 Official base 加载 DiT 并完全冻结 DiT、`midi_proj`、SOME teacher 和 VAE；
- 只训练 P embedding，`LR=1e-4`，首轮目标 500 步；
- 关闭 audio/text/MIDI CFG dropout，避免冻结阶段产生无效或补偿性梯度；
- 在 100/250/500 步保存 model checkpoint，并用固定真实样本推理；
- 500 步仍稳定改善时最多延长至 1k；无趋势时停止并审计，不以盲目延长替代诊断。

### 阶段 C：训练/推理 smoke

- 先做 CPU/单卡 shape 与 checkpoint round-trip；
- 再做 DDP 10 步 smoke，确认各 rank 数据和 loss 正常；
- 生成固定短样本，确认训练与推理均真正走 MIDI_P；
- 日志实时打印 step、loss、FlowA、FlowB、CKA、运行时长和 ETA。

### 阶段 D：正式联合训练

- 从阶段 B 选定的 P 权重开始，Official DiT 与 P embedding 共同训练；
- 丢弃 P-only optimizer 状态，以 `LR=1.4e-5` 重建 optimizer 和 500-step warmup；
- 以切换时的当前模型权重重新创建 EMA，避免短校准阶段的 EMA 残留初始 P；
- 恢复 V4f 的 audio/text/MIDI CFG/dropout 配方，联合训练另计完整 30k；
- 使用 tmux session 与 V4f 对齐，至少覆盖早期、中期和末期轨迹；
- 不因单次最终 checkpoint 结果覆盖完整训练轨迹；
- 训练异常或 MIDI_P 类别退化时立即停止并保留日志。

## 评价

至少分别记录：

- 文本：PER 与 T1 长、多句稳定性；
- 旋律：F0-CORR、音符命中、起止边界与长音稳定性；
- 身份：target singer与 A 区参考依赖；
- 表现：颤音、滑音、抢拍、拖音和轻柔唱法是否被抹平；
- 声学：浴室感、雾感、电音、雪花声和局部崩坏；
- 条件依赖：关闭 MIDI、改变 MIDI 音高或边界时输出是否按预期变化。

## 停止条件

- rest/padding 语义未解决；
- MIDI_P 展开后出现系统性长度错位或边界插值污染；
- 训练与推理通路不一致；
- MIDI_P embedding 未进入 optimizer、未保存或无法可靠恢复；
- smoke 阶段出现 loss 非有限、DDP rank 分歧或 MIDI 条件完全失效。

## 当前记录

- 文档分支已建立。
- 已确认官方源码存在音符化接口，当前 V4f 使用连续 fuzz MIDI。
- 已完成首轮静态接口审计，发现官方路径不能未经修正直接用于 V4Pf：
  - `MIDIExtractor.postprocess(..., with_expand=True)` 计算了 `note_mask_pred`，但展开时丢弃该 mask，rest 会退化为数值 0；
  - 官方二维插值使用 linear mode，离散音符在重采样边界会产生原序列不存在的中间音高；
  - `MIDIDigitalEmbedding` 预留了特殊类别容量，但当前调用没有显式传入 rest/pad 类；
  - V4f 的 CKA target 是原始 128 维 SOME logits，首轮审计时需裁决它是 P 输入的同构目标，还是只作不进入推理条件的辅助监督；后续已明确选择后者；
  - 现有 checkpoint 保存整个 `raw_model.state_dict()`，但加载使用 `strict=False` 且不检查 missing keys，新增 P embedding 可能静默漏载。

## 冻结的 MIDI_P v1 语义

首轮 V4Pf smoke 与训练统一采用以下定义：

| 项 | 定义 |
|---|---|
| pitch resolution | `mark_distinguish_scale=2`，即 0.5 半音一个 class；MIDI 0..127 映射为 class 0..254 |
| REST | 独立 class 255 |
| PAD | 独立 class 256，仅用于 batch padding，不与 REST 共用 |
| embedding | 257 类可学习 embedding，输出 128 维 |
| embedding 初始化 | 待对照：随机初始化 vs 官方 sigmoid SOME 坐标系中的结构化标准音高模板；模板不得使用未经处理的同音高演唱帧粗均值 |
| 时间对齐 | note pitch 与 voiced/rest mask 分开处理；有声区使用 mask 归一化线性对齐后重新量化，REST/PAD 使用离散对齐 |
| A 区 null | 进入 DiT 拼接前的 128 维全零向量，不编码成 MIDI 0、REST 或 PAD |
| CFG drop MIDI | 与 A 区统一为进入 DiT 拼接前的 128 维全零向量，不允许因 `midi_proj` bias 形成不同 null |
| CKA target | 原始、连续的 128 维 SOME 表示，仅作训练辅助监督；不进入 MIDI 条件或推理输入 |

这里的 MIDI 0 虽在正常歌声中极少出现，仍保留为合法 pitch class，不能兼任 REST。
PAD 不参与 CKA 或有效帧 loss。V4f 当前单样本训练通常没有 batch padding，但实现仍必须显式区分。

每个 P class 只有一个共享 embedding，因此它必然是音高类别原型，不能保存某次具体演唱的
颤音或滑音。这是 P 删除微观表现泄漏的设计目的，不等于生成结果必须平均化。DiT 仍根据
歌词、时长、上下文、A 区参考、扩散状态和初始噪声学习同一音高下的演唱分布。正式评价必须
比较生成 F0 相对量化中心的残差、颤音/滑音统计和多 seed 差异，不能只检查 embedding 本身。

## Checkpoint 与加载纪律

- P embedding 必须挂在训练模型下，进入 optimizer、model state 和 EMA state；
- 从 official base 初始化时，允许且只允许 P embedding 对应 key 缺失；其他 missing/unexpected keys 必须打印并按白名单断言；
- resume V4Pf 时 P embedding key 必须完整存在，不得按初始化白名单放过；
- checkpoint `args` 之外另存 MIDI_P schema/version、类别编号、scale 和时间对齐模式；
- 推理加载时验证 schema 与训练一致，不一致立即失败。

## 两阶段优化纪律

- P-only 阶段只允许 P embedding 进入 optimizer；DiT 和 `midi_proj` 必须冻结且保持 Official base 权重；
- P-only 阶段只允许损失梯度穿过冻结 DiT 回传至 P，不得更新 DiT；首轮使用 FlowB-only 还是 FlowB+raw SOME CKA 尚待 SOME 门禁后裁决；
- 阶段切换只继承模型权重，不继承 Adam moment、scheduler 进度或旧 EMA；
- 联合阶段从新的 step 0 记录完整 30k，日志同时记录此前 P-only 步数；
- 联合阶段 P 与 DiT 使用相同 `LR=1.4e-5`，首轮不引入额外 P LR multiplier；
- P-only 门禁的真实推理必须使用 model 权重；联合阶段使用切换后从当前模型重建的 EMA，禁止用仍残留初始 P 的旧 EMA 下听感结论。

## SOME 音符化试听门禁

有效试听包 v3 已完成，详见 `SOME音符化试听验证.md`。v1/v2 因 SOME Mel 前端频率范围错误
导致绝对音高系统性偏低，均不得用于结论。该阶段只验证冻结 SOME teacher、
note/boundary 解码、0.5 半音量化、REST 和 VAE 时间轴重采样，不验证 P embedding、Official
DiT 或最终生成质量。首轮用户试听发现 SOME 偶发局部低音崩坏，未通过正式 teacher 门禁。
当前改为同一 62 条 GAME/SOME native 钢琴 A/B；GAME 人耳裁决完成前不进入 P-only 训练。

## 与 V4Pvf 的工程关系

已停止的 V4Pvf 完成了量化器、REST 修复、DDP、checkpoint 和真实推理链路验证，这些结果
可以作为 V4Pf 的工程证据，但不能作为权重或质量证据。V4Pf 为兼容 Official base 新增结构化
P 初始化和 P-only 校准，不再声称与 V4Pvf 仅差 DiT 初始化来源。训练与推理仍须使用同一份
V4Pf schema；任何语义变化都必须升级并记录 schema，不能静默沿用旧版本号。

















