# V5-P 训练前完整审计

> 状态说明（2026-08-11）：本文保留为训练前审计快照，以下 `design_frozen=false`、
> `training_authorized=false` 和“启动阻塞项”描述的是 2026-08-08 当时状态，不再代表当前执行状态。
> 这些阻塞项随后均已闭合；V5-P Phase A/B、低 LR V5-Pg 10k 已完成，V5-Pg20-HLR07 20k
> 正式运行中。当前权威状态与新增路线边界见 `V5-P正式训练计划.md` 第 13 节和 `main_V5P.md`。

## 审计结论

V5-P 的代际方向成立：Official fresh 起点、P-only 校准、H/PUL + GAME-P + 去重 Long 的 non-g
阶段、再切换 285k online VAE 的 g 阶段，彼此顺序清楚，历史证据也足以支持进入集成工程。

当前状态仍应保持：

```text
design_frozen = false
training_authorized = false
```

原因不是路线需要推倒重来，而是数据确定性合同、Long-H 统一 schema、V5 专用训练/推理入口和
EvalPool 尚未闭合。正式训练计划在这些阻塞项完成前不能改成已冻结或已授权。

## 审计后裁决

用户于 2026-08-08 闭合设计问题：

- H 容量不足和历史 Control collision 允许既有 token 覆盖/截断，不因 token 丢失淘汰样本；
- Long 与超过 30 秒的 Short 需要重跑的 Whisper/SOFA/H 全部重跑；
- Whisper 沿用历史 `kana similarity >= 0.75`、长度比 `0.75--1.30`，全量完成后复审实际通过率；
- 独立 V5-P 训练入口、Periodic Short/Long EvalPool 和 V5 推理入口均纳入必做工程；
- Phase C 保留 Phase B 的 H/GAME-P/L、loss、dropout 和 trainable 参数，只切换 285k VAE并重建
  optimizer/scheduler/RNG/cursor；raw/EMA transition 沿用 V4Hg 审计语义；
- Phase A 自动裁决沿用 V4PH v2 projected-RMSE 主规则；
- `NATURAL_RECORD` 的 Long 权重后果接受，不修改 sampler；
- Phase B LR 改为 0--2k warmup 至 `1.4e-5`、2k--28k cosine 至 `1e-5`、28k--40k
  cosine 至 0。

因此下列第 1/3/5/6/7 项的设计歧义已闭合，第 2/4 项及全部资产、实现、smoke 仍是启动阻塞项。

## 已通过的设计项

1. 从 Official checkpoint fresh 开始，不继承 V4PH/HighLR/V4H/V4Hg/V4fg/M600 学习状态，谱系干净。
2. Phase A 只训练 P embedding，Phase B 重建 optimizer/scheduler/EMA，Phase C 再独立切换 VAE，
   阶段边界清楚。
3. Long 去重在全部门禁后按实际 `L_accept` 计算；失败 Long 的成员 Short 返回 ShortPool，不丢唯一内容。
4. `KEEP_LONG_DEDUP_SHORT + NATURAL_RECORD` 可完整恢复每个 epoch 的唯一来源内容，不在 loader 内临时去重。
5. Long 以完整 waveform 重跑 Whisper、SOFA、H 和 GAME，不拼旧 Short dense tensor，不增加 JOIN/GAP token。
6. Phase B 的 `2k warmup + 26k cosine shoulder + 12k final cosine` 已闭合；
   12k/24k/28k/34k/40k 保存点有明确含义。
7. 评价明确不允许 Loss/FlowB/CKA 覆盖人耳否决，并同时保留短集、长集和 g 前后对照。
8. 训练前要求单卡、四卡、exact-resume、checkpoint provenance 和 fresh 正式目录，工程门禁方向正确。

## 启动阻塞项

### 1. Long-H 合同与当前 renderer 尚未统一（设计已闭合）

正式计划仍把 Phrase anchor、A/B 穿句、filler、SEP 和容量策略列为待复审，并要求 token 顺序不变、
30 秒后 token 不丢失。当前已确认容量不足时允许截断，因此这些断言不能继续作为硬门禁。

为了让 V5 只集成既有 H，而不是顺手再发明 H-v3，建议冻结现有 V4H runtime 语义：

```text
anchor              = Whisper Phrase start
phone candidate     = 首 phone 相对帧归零，后续保留相对间隔，再平移到 anchor
A/B ownership       = Phrase center 决定整句侧别
ref_len             = 既有动态规则，并在附近存在 Phrase start 时吸附
phone failure       = 既有 PUL fallback
SEP                 = 下一 Phrase anchor 前一帧；末句在最后一帧
unused frame        = 既有 0/V 语义
control anomaly     = H 精确复用既有 dense Control 结果
capacity shortage   = 允许既有前缀截断，不因截断淘汰样本
```

历史三点审计中，token 截断发生于 `186 / 33,696 = 0.552%` runtime 场景，截断事件占
`1,356 / 3,826,632 = 0.0354%`。旧 Control collision 也会触发 exact-Control；V5 必须同时报告
collision、truncation、受影响样本和事件数，不能只登记截断。

### 2. Short 与 Long 不能直接共用现有 H87 fingerprint

现有 H runner 把 `max_training_duration=30` 写入 config fingerprint，并对
`Duration > 30s` 使用历史 fallback。TIME87 的 private corpus count 条 Short 中实际有 26 条超过 30 秒，最长
31.139 秒；旧 30 秒口径累计少读 6.341 秒。V5 声明不裁剪后，不能原样把旧 H87 fingerprint 与
新的 Long-H fingerprint 混进一个要求单一 fingerprint 的 loader。

正式处理应二选一，推荐第一种：

1. 用同一个 V5-H `max_duration >= 60s` schema 重新导出 Short、Long 和 Eval 的紧凑 H manifest；
   既有 `<=30s` Short 可复用已有 SOFA interval，只重建 frontend/candidate/schema并验证 candidate
   字节等价；历史 runner 没有完整处理的 26 条 `>30s` Short 必须补跑完整 SOFA。
2. 或显式支持双 fingerprint，并在 mixed manifest 中逐记录绑定来源 schema；这会扩大训练入口和
   checkpoint 审计复杂度，不推荐。

### 3. Long 数据门禁尚不能确定性计算 `L_accept`（规则已闭合，待实跑）

整段 Whisper 已确定要重转录，但以下参数尚未写入合同：

- kana similarity 阈值和长度比阈值；
- Phrase 空值、越界、重叠及整段漏词的失败规则；
- SOFA 部分失败、非有限 interval 和 phone/text mismatch 的处理；
- token overflow 是拒绝、fallback 还是仅审计。

正式合同已沿用历史 V4L 的 Whisper 门禁 `similarity >= 0.75` 且长度比 `0.75--1.30`；SOFA 必须
整条成功、有限、单调且在 PCM 时长内。token capacity/overflow 不再淘汰 Long，只记录并交给
runtime exact-Control/截断策略；非法 token ID、空歌词或不可构造 Phrase 仍应拒绝。全量后按总量、
时长桶、两段/三段窗口和失败原因报告实际通过率，用户复审后再冻结 `L_accept`。

### 4. 目前没有可执行的 V5-P 训练入口

现有实现不能通过改参数直接启动：

- `tools/alignment/run_h_v1_full.py` 把 total frames 和 config 固定到 30 秒；
- `package_v4c_finetune/train/train_v4ph.py` 会把超限 waveform 静默裁掉；
- V4PH joint 入口硬门禁 `30k / warmup500 / LR 1.4e-5`；
- 现有 V4Hg g 入口使用 SOME，不包含 GAME-P embedding；
- 现有 GAME loader 允许 basename fallback，不满足 Path + audio SHA 的 V5 provenance。

需要独立 V5-P 入口，而不是放宽历史 V4PH 的保护条件。新入口必须覆盖 Phase A/B/C、完整 PCM、
mixed pool、逐 pool 计数、V5 scheduler、GAME-P、H/PUL 和 285k VAE transition。

### 5. Phase C 合同不完整（已闭合）

当前只冻结了 source、VAE、步数和 LR。还必须显式写出：

```text
trainable            = Phase B 的完整 DiT + H/PUL + P embedding
loss                 = FlowA + 2 * FlowB + 0.7 * GAME-compatible CKA
dropout              = audio 0.3 / text 0.15 / MIDI 0.3
data/ref_len         = 与 Phase B 完全相同
raw model source     = V5-P 40k EMA
EMA source/counter   = 复刻 V4Hg：同一 40k EMA 权重初始化 raw 与 EMA，并继承 EMA counter
optimizer/scheduler  = fresh
RNG/sampler cursor   = fresh step 0
```

否则“沿用历史 g 配方”可能被误实现成 continuous SOME 的 V4Hg，而不是保留 GAME-P 的 V5-Pg。

### 6. Periodic EvalPool 与 V5 推理入口缺失（已授权实施）

训练代码要求固定 eval manifest，但计划没有定义 V5 的 periodic EvalPool。至少需要：

- 将既有 201 条短 Eval 重新导出为统一 V5-H schema；
- 建立不与任何 Long 成员重叠的固定 Long Eval；
- 分别报告 Short/Long 的 FlowA、FlowB、CKA、H fallback 和 token anomaly；
- 在正式 Phase A 前完成 V5-P 推理 smoke，证明 H + GAME-P + A/B + 29/31/40/55 秒输入可生成；
- Phase C 前另做同一路径的 285k VAE 绑定 smoke。

长度集必须同时登记 `B_duration` 和 `A + reference_tail + B` 的总 latent 时长。只写 55 秒会混淆
它是 B 长度还是总序列长度。

### 7. Phase A 的自动通过规则需要写死（已闭合）

V4PH 历史 v2 的实际裁决是：P500 相对 P300 的 frozen `midi_proj` projected RMSE 必须改善；raw
RMSE、raw cosine 和 pitch-geometry CKA 全量报告但不是一票否决。V5-P 必须明确复用这一 v2 规则，
并定义主指标未改善时停止 Phase B，而不能只写“结构化音高核距离裁决”。

## 已接受但必须显式记录的风险

### Natural-record 的真实权重

若 3,373 条 Long 全通过：

| 项 | Short | Long |
|---|---:|---:|
| 记录 | 7,563 | 3,373 |
| 时长 | 43.066154h | 44.082823h |
| update 概率 | 69.16% | 30.84% |
| 平均记录时长 | 20.50s | 47.05s |

FlowA/FlowB 在各自区域按 frame 取 mean，因此按直接 MSE frame 权重近似计算，Long 每小时只有 Short
的 `0.436x`。这表示 Long 是重要少数条件，而不是按音频秒数与 Short 平权。当前无需改 sampler，
但正式合同应把它写成有意选择；若 V5 仍在 30 秒后失稳，再讨论 duration-aware sampling。

### LR 已改为略高于 HighLR 的总暴露

以 LR 曲线面积作粗略代理：

```text
V4PH baseline： 0.2940
PH HighLR：     0.3745
V5-P：          0.3860
```

V5-P 现改为 2k warmup 到 `1.4e-5`，随后 26k cosine 缓降到 `1e-5`，最后 12k cosine 到 0。
曲线面积比 baseline 高约 31.29%，比 HighLR 高约 3.07%。这满足“更长、总力度略高”，同时取消
26k 固定峰值 hold，避免长期停留在 `1.4e-5`。

### Long 上的 CKA 与计算量

CKA 在 B 区构造 `T_B x T_B` similarity matrix，60 秒样本相对 30 秒约有 4 倍矩阵面积。恢复静音、
REST 和长 pause 也可能改变 CKA 结构。最长样本门禁必须运行完整 V5 图，包括 GAME-P、三层 CKA、
H/PUL、backward、gradient sync 和 EMA；历史 V4L 的普通最长样本 probe 不能替代它。

### 集成失败时的归因能力有限

V5-P 同时改变 H、P、L、数据量、LR 和后续 VAE。它是正式集成，不是严格单变量实验；若失败，
不能只凭一条 run 判断具体模块。保存 non-g 中间点、分 pool 指标和未来同配方 V5-S，是最低成本的
归因保障。

## 需要补充的运行指标

每个日志窗口除总指标外，必须按 Short/Long 分开记录：

```text
record count / audio seconds / mean duration
FlowA / FlowB / CKA
phone / PUL / exact-Control phrase count
collision / truncated sample count / truncated event count
P voiced / REST / PAD frame count
post-30s FlowB diagnostic（存在该区域时，仅作日志）
step time / max allocated / max reserved
```

混合总 Loss 可能在 Long 单独失效时仍显得正常，因此分 pool 日志是 L 路线的必要观测，不是可选美化。

## 工程细节补充

1. Official source 必须写明加载的是 checkpoint 的哪个 section；历史入口实际优先加载 Official EMA。
2. Short、Long、Eval、GAME 和 H 以 `SampleId/WindowId + canonical Path + audio SHA256: redacted` 联合绑定；
   禁止 basename fallback。
3. 正式 44.1kHz/mono/PCM16 资产不应在 loader 内自动 resample；格式不符应在门禁失败。
4. final TrainPool 数量若不能被 world size 整除，`DistributedSampler(drop_last=false)` 会每 epoch
   填充最多 3 条重复记录；应在 metadata 明示或使用无填充等价 sampler。
5. reference tail 0.5 秒是推理 A/B 边界策略，训练 A/B 来自连续原音频，不应把 0.5 秒静音插入训练波形。
6. 已知恢复 gap 可用于离线 GAME REST 审计，但不得作为 H 的额外输入或生成专用 token。
7. 中间 full-state checkpoint 数量较多，启动前需核算本地盘、服务器盘和cloud storage上传策略。

## 推荐执行顺序

```text
1. 按已裁决合同实现 H/ref_len/truncation/Phase-C/Phase-A 与新 LR scheduler
2. 运行 Long Whisper，报告历史阈值下实际通过率并由用户复审
3. 建 V5-H 统一 schema，重导出 Short + Eval，并生成 Long SOFA/H
4. 生成 Long GAME，合并 Short/Long/Eval GAME provenance
5. 计算 L_accept/U_accept 和最终 DEDUP manifests
6. 实现独立 V5-P Phase A/B/C 与推理入口
7. 静态/CPU/单卡最长/推理 smoke/四卡 mixed/exact-resume/checkpoint audit
8. 用户显式授权 Phase A
9. P500 v2 裁决通过后，用户显式授权 Phase B
10. V5-P 40k 完整审计与听评通过后，用户显式授权 Phase C
```

完成第 1--7 项前，路线可以继续准备，但不能称为 design frozen；完成后仍需用户单独给出训练授权。

















