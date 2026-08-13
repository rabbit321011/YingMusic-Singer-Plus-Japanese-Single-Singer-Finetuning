# V5-P 正式训练计划

## 1. 目标

V5-P 是 V4 后实验收敛后的第一个正式代际集成模型。它不再验证单个模块是否有价值，而是把
已经获选的能力放进一条从官方 checkpoint 重新开始、可以完整审计的训练谱系：

```text
H：mora-anchored phone placement + PUL/SEP
P：GAME medium K=4 -> quantized MIDI_P
L：30--60 秒最大长上下文，恢复 trim gap 静音，成员 Short 去重
g：non-g 完成后切换 285k online VAE 做音色适配
tail：reference tail 0.5 秒
```

首轮只执行 P 路线。未来 V5-S 使用相同的官方起点、H、L、训练日程和 g 阶段，只把
GAME MIDI_P 换成 continuous SOME MIDI。`V5-S` 的 `S` 固定表示 SOME，不表示 `S/` 分支的
SVC 自克隆训练集。

## 2. 命名与谱系

| 名称 | 含义 | VAE | 状态 |
|---|---|---|---|
| `V5-P-A` | 官方起点上的 P-only 500-step 校准 | official | 已完成并通过裁决 |
| `V5-P` | H + GAME-P + L-DEDUP 的 non-g 结构模型 | official | 40k 完成，默认权重为 40K EMA |
| `V5-Pg` | 从 V5-P 40K EMA 出发的低 LR 音色适配 | 285k online | 10k 完成，待轨迹听评 |
| `V5-Pg20-HLR07` | 从同一 V5-P 40K EMA 独立起步的强化适配 | 285k online | 20k 正式训练中 |
| `V5-Pg40-HLR07` | 可能的完整 40k ×0.7 LR 路线 | 285k online | 仅登记决策条件，未授权 |
| `V5-S` | H + continuous SOME + L-DEDUP | official | 后置，不授权 |
| `V5-Sg` | 从 official fresh 直接进入 285k + continuous SOME 的 40k 路线 | 285k online | 合同已设计，未授权 |

正式谱系：

```text
ckpts/YingMusicSinger_model.pt
  -> fresh V5-P-A P500
  -> fresh optimizer/scheduler/EMA 的 V5-P 40k
  -> V5-P 40k EMA warm-start
     |-> fresh optimizer/scheduler 的 V5-Pg 10k（低 LR 基线，已完成）
     `-> fresh optimizer/scheduler 的 V5-Pg20-HLR07 20k（强化路线，运行中）
```

V5-P 不继承历史 V4PH P500、V4PH/HighLR final、V4H/V4Hg、V4fg 或 M600 的模型、EMA、
optimizer、scheduler、RNG、sampler cursor。历史分支只提供设计证据和实现先例。

## 3. 明确不进入首轮的路线

- LCF：严格对照听评无明显改善，不进入 V5；
- M600/1.5B：约 600M 的可听收益很小，首轮保持 338M DiT；
- V4IPH/V4IjPH：无 A 与 INS 替代 A 的感知门禁失败；
- 随机初始化：V4vf/V4vfg 感知门禁失败；
- SVC 自克隆训练集：V4Sf 24k 略弱于原始 waveform 对照；
- GRPO：只允许在 V5 SFT 完成后作为独立增强，不阻塞 V5-P；
- reference tail 扫描：不再搜索，固定为 0.5 秒。

## 4. L-FULL-LENGTH-TIME 正式训练池

### 4.1 冻结输入

音频候选来自 L-FULL-LENGTH-TIME：

```text
TIME87 Short 候选：private corpus count 条，87.001480h
Long 音频候选：3,373 条，44.082823h
Long 使用的唯一 Short：7,003 条
Long 长度：30--60 秒；每条至少两个短段
恢复的 trim gap 静音：0.147497h
```

权威窗口 manifest SHA256: redacted

```text
71e3d32a727b2f1fe5d398257fa5c16aa36363425e3e1bd8c1ed4fe2f9abaecb
```

本阶段沿用 `KEEP_LONG_DEDUP_SHORT`，不使用 `KEEP_LONG_ALLOW_REPEAT`。

### 4.2 去重必须在门禁后计算

`3,373` 是音频候选数，不预先假定全部通过 Whisper、SOFA、H 和 token 门禁。设：

```text
L_accept = 通过全部门禁的 Long
U_accept = L_accept 使用的底层 Short Path 集
ShortPool = 通过 Short 门禁的 TIME87 - U_accept
LongPool = L_accept
TrainPool = ShortPool union LongPool
```

若一条 Long 被淘汰，其成员 Short 必须继续保留在 ShortPool；不得因为 Long 构建失败而丢失唯一
训练内容。只有正式进入 LongPool 的窗口，其成员 Short 才从独立短记录中排除。

若 3,373 条全部通过，则预期为：

| 项 | 值 |
|---|---:|
| ShortPool | 7,563 条 |
| LongPool | 3,373 条 |
| TrainPool | 10,936 条 |
| ShortPool 源音频 | 43.066154h |
| LongPool 表观音频 | 44.082823h |
| Long 记录比例 | 30.84% |
| Long 加载时长比例 | 约 50.58% |

最终数字必须由门禁后的 manifest 重新计算并写入 checkpoint metadata，不能复制预期值。

### 4.3 采样

首轮冻结：

```text
pool_policy = KEEP_LONG_DEDUP_SHORT
sampling_policy = NATURAL_RECORD
long_probability = null
```

ShortPool 与 LongPool 合并后按记录等概率采样。日志每 1,000 step 至少报告：

```text
累计 Short/Long 记录数
累计 Short/Long 音频秒数
Long 成员段数分布
当前样本时长
step time / elapsed / ETA
```

checkpoint 必须保存 pool policy、Short/Long/mixed manifest SHA、sampler RNG、数据游标以及累计
记录数和秒数。不得在 loader 内按 basename 临时去重。

## 5. Long-H 构建合同

### 5.1 原则

Long-H 是同一条 30--60 秒音频全局时间轴上的正常 H，不是两三份旧短样本 dense H tensor 的拼接。

正式 H 来源必须是拼接后的 Long WAV、已知成员歌词和重新计算的 Long 对齐。旧 Short-H 只作为
审计对照，不作为正式 Long-H 输入。

### 5.2 全局时间轴

1. 以正式 Long WAV 的实际 PCM frame 数为唯一时长权威；
2. 成员 offset 按实际拼接帧数累计，不按浮点 duration 重算；
3. 所有 phrase、mora、phone、PUL、SEP 和 GAME frame 最终转换到同一 Long 全局坐标；
4. 任何 30 秒常量、静默裁剪或隐式最大帧缓存均视为错误。

### 5.3 整段 Whisper 与 SOFA

1. 对正式 Long WAV 重新运行整段 Whisper，不拼接旧 Short 的转录结果；
2. Whisper 新转录必须与成员 Short 的旧文本/kana 做一致性门禁；旧文本只作为审计基准，不直接
   覆盖通过门禁的新转录；
3. 通过门禁后，以整段 Whisper 的 Phrase 文本、起止时间和统一 kana 构造新的 Long 训练文本；
4. 对完整 Long WAV 和新文本重新运行 SOFA，得到同一全局时间轴上的 mora/phone interval；
5. 使用与 H 分支相同的 kana 规范化、Japanese IPA tokenizer、vocab 和 phone-to-token 映射；
6. 不接受拼接旧 Short SOFA、旧 phone onset 或旧 dense H tensor 作为正式 Long 结果。

Whisper 一致性门禁沿用历史 V4L 已验证口径：

```text
kana similarity >= 0.75
new/old kana length ratio in [0.75, 1.30]
```

SOFA 结果必须整条成功、有限、单调且全部 interval 位于实际 PCM 时长内。非法 token ID、空歌词或
无法构造 Phrase 的记录失败；单纯 token capacity/overflow 不淘汰 Long，交给 runtime 的既有
exact-Control/截断语义处理。全量完成后必须报告总体通过数/百分比、按 30--35/35--40/40--45/
45--50/50--55/55--60 秒分桶的通过率、两段/三段窗口通过率、kana similarity/长度比分布和逐原因
失败数。通过率报告由用户复审后才能冻结 `L_accept`，不预设一个额外的总通过率硬阈值。

### 5.4 Long-H token 放置规则

Long-H 冻结为既有 V4H renderer 在完整时间轴上的直接延伸，不新增 H 变体：

- Long WAV 是一个连续 waveform 和唯一全局时间轴；H 不读取 trim gap 或成员接缝 metadata；
- 不新增 `LONG`、`JOIN`、`GAP` 或其他专用 token；
- 恢复静音是普通 waveform 内容，不自动映射成 `<PUL>`；
- 整段音频内部仍保留 Whisper/SOFA 给出的多个 Phrase，不把全部歌词压成单一 Phrase；
- phone 对齐可靠句继续使用 phone candidate；`<PUL>=366` 只沿用既有 H 的 phone-ineligible
  fallback 语义，不充当通用静音标签；
- Whisper Phrase start 继续作为 Control/H 共同首 token anchor；
- reliable phone 的首 phone 相对帧归零，后续 token 保留相对 phone 间隔，再整体平移到 anchor；
- Phrase center 决定整句属于 A 或 B，不做句内 token 切分；
- reliable phone 句的未占用帧继续使用既有 `0/V` filler；
- `<SEP>=365` 位于下一 Phrase anchor 前一帧，最后一句 SEP 位于整段最后一帧；
- phone placement 失败且容量允许时继续使用既有 PUL fallback；
- Control 发生跨 Phrase collision、A/B/尾部容量不足或其他历史 anomaly 时，H 精确复用既有 dense
  Control 结果；允许既有覆盖和前缀截断，不因 token 丢失淘汰样本，也不为保留 SEP 改写 renderer；
- 每次运行必须记录 collision、truncated sample/phrase/event 和 exact-Control 数量。

### 5.5 动态 A/B

Long-H 候选描述整条 Long，不预先绑定单一 `ref_len`。训练时仍由既有 renderer 根据动态 A/B
边界产生 dense placement。`ref_len` 沿用历史训练规则：先在 `12.5%--33%` 总帧范围均匀采样，
下限 5 秒、上限 65% 总帧；若 ±2.5 秒内存在 Phrase start，则吸附到最近的 Phrase start。全量
门禁至少覆盖：

```text
ref_len 在第一个成员内部
ref_len 紧邻成员接缝前/后
ref_len 紧邻 30 秒前/后
ref_len 位于 phone 内部
ref_len 位于句间 PUL 区
ref_len 位于最后一个成员内部
```

必须验证：正常非 anomaly 场景 token 顺序不变、跨句 phone 不重叠、SEP/PUL 合法、30 秒后的
Phrase/H/GAME 仍参与 placement、全部索引在有效帧内、同一样本同 ref_len 输出确定性一致。
anomaly 场景允许 token 覆盖/截断，但 H 必须与同场景 Control 结果一致，且丢失计数可复算。

## 6. Long GAME-P 与其他资产

- GAME 必须对正式 Long WAV 重新运行，不能直接拼接旧短 cache；
- 固定 GAME commit、medium K=4、模型 SHA、adapter schema 和按音频 key 的确定性 seed；
- 保存原始 durations/presence/scores、P classes、有效帧和时间闭合误差；
- 0.5 半音 class、REST=255、PAD=256 和 128 维 P embedding 语义不变；
- A 区 MIDI 在训练/推理合同要求下继续清零；
- Long WAV、Whisper、SOFA、H、GAME、token 和训练 manifest 必须通过 Path 与音频 SHA 绑定。

## 7. Phase A：V5-P-A P-only

Phase A 从官方 checkpoint fresh load，只训练新的 P embedding：

```text
source: official YingMusic-Singer-Plus checkpoint
data: V5-P 最终 L-DEDUP TrainPool
placement: H phone/PUL
MIDI: GAME medium K=4 quantized P
trainable: P embedding only
loss: FlowB + 0.7 * GAME-compatible CKA
LR: constant 1e-4
steps: 500
batch: 1/GPU x 4 GPUs x grad accumulation 4 = effective 16
seed / eval seed: 42 / 1042
```

沿用 V4PH 的 fixed-seed matched-random 初始化、结构化音高核距离裁决、REST/PAD 和未支持 row 补齐
规则，但重新生成 V5-P 自己的 P500、裁决报告和 SHA。不得引用历史 V4PH P500 作为训练来源。

Phase A 自动裁决严格沿用 V4PH v2：在 P500 的实际支持 row 集合上，P500 相对 P300 的 frozen
`midi_proj` projected RMSE 必须下降；raw RMSE、raw cosine distance 和 pitch-geometry CKA 全量
保存为诊断项，但不单独一票否决。主指标未改善时停止，不得进入 Phase B。

Phase A 只裁决 P 坐标是否学入，不做人耳低步数听评。通过后只把 P500 的 model state 送入
Phase B transition；不继承 Phase A optimizer、scheduler、EMA、RNG 或 sampler cursor。

## 8. Phase B：V5-P non-g 40k

### 8.1 配方

```text
source: V5-P-A 通过裁决的 P500 transition
trainable: 338M Official DiT + H/PUL + P embedding
VAE: official frozen VAE
data: V5-P L-DEDUP TrainPool
loss: FlowA + 2 * FlowB + 0.7 * GAME-compatible CKA
CFG/dropout: audio 0.3 / text 0.15 / MIDI 0.3
physical batch: 1/GPU
world size: 4
gradient accumulation: 4
effective batch: 16
seed / eval seed: 42 / 1042
EMA: fresh at Phase B step 0
```

`max_duration` 必须大于等于门禁后最长 Long 的实际时长加帧舍入余量。训练入口不得沿用历史
`max_duration=30`，也不得静默裁剪超限样本；最终数值在正式 manifest 冻结后写入合同。

### 8.2 LR 与步数

首轮冻结：

```text
max_steps = 40000
warmup_steps = 2000
0--2k: linear warmup 0 -> 1.4e-5
2k--28k: cosine shoulder 1.4e-5 -> 1e-5
28k--40k: cosine decay 1e-5 -> 0
```

两段 cosine 必须分别以自身区间归一化并在 step 28k 连续衔接：

```text
2k <= s <= 28k:
lr = 1e-5 + 0.5 * (1.4e-5 - 1e-5) * (1 + cos(pi * (s - 2k) / 26k))

28k <= s <= 40k:
lr = 0.5 * 1e-5 * (1 + cos(pi * (s - 28k) / 12k))
```

相对历史 PH HighLR，本轮延长 warmup、增加总 update，并把原 hold 改成从 `1.4e-5` 缓降至
`1e-5` 的 26k cosine shoulder，再用 12k 降至 0。曲线面积粗略为 `0.3860`，比 PH HighLR 的
`0.3745` 高约 3.07%；该面积只作调度审计，不等价于 AdamW 的实际参数位移。

至少保留并独立审计：

```text
12k / 24k / 28k / 34k / 40k final
```

28k 是 cosine shoulder 终点，34k 是最终 decay 中点，40k 是 decay 终点。标准 eval 只证明训练
健康，不能代替音色、咬字、旋律和长程生成听评。

## 9. Phase C：V5-Pg 10k

Phase C 只在 V5-P Phase B 完整结束并通过 checkpoint 与音频门禁后启动：

```text
source: V5-P 40k EMA
VAE: frozen 285k online
H / GAME-P / L pool: 与 V5-P 完全相同
trainable: 完整 DiT + H/PUL embedding + P embedding
loss: FlowA + 2 * FlowB + 0.7 * GAME-compatible CKA
CFG/dropout: audio 0.3 / text 0.15 / MIDI 0.3
ref_len: 与 Phase B 完全相同
steps: 10k
peak LR: 5e-6
schedule: 0--250 warmup；250--6250 hold 5e-6；6250--10000 cosine decay 到 0
optimizer/scheduler: fresh
RNG/sampler cursor: fresh step 0
inference VAE: 必须绑定同一 285k online checkpoint
```

raw model 和 EMA 都从 V5-P 40k EMA 权重初始化，EMA counter 继承 40k source，语义必须复刻已经
审计的 V4Hg g transition；不得调用会把 GAME-P 换回 continuous SOME 的历史 V4Hg 数据/模型入口。
metadata 必须绑定 V5-P 40k source SHA、official/285k VAE SHA、训练代码和全部数据 manifest SHA。

首轮不因 Phase B 延长而自动延长 g。正式训练在第一个 update 前额外保存
`step_000000_transition.pt`，并记录 285k VAE 下的 step-0 Short/Long Eval；随后每 1k 保存一次，
每 500 step 计算一次 Short/Long Eval，因此冻结轨迹为：

```text
0k transition / 1k / 2k / 3k / 4k / 5k / 6k / 7k / 8k / 9k / 10k final
```

`0k transition` 的 raw 与新 EMA 必须逐张量等于 V5-P 40k EMA，optimizer state 为空、scheduler
位于 fresh step 0、EMA counter 为 40k、四 rank 数据游标为 `(0, 0)`。1k--10k 每个 checkpoint
同时保留 raw/EMA，供后续固定 CFG 1.0 实名轨迹听评使用。若未来讨论 10k 之后训练，必须另立
授权和听评门禁。

## 10. 工程门禁

正式 Phase A 前必须全部通过：

1. L_accept、U_accept、ShortPool、LongPool 和 mixed manifest 全量集合审计；
2. train/eval source Path 与成员级 overlap 为 0；
3. Long Whisper 通过率按总量/时长/成员数/失败原因完整报告并经用户复审；
4. Long Whisper、SOFA、token、H、GAME 产物缺失 0、非有限 0、越界 0；
5. 统一 V5-H schema 覆盖 Short/Long/Eval；26 条 `>30s` Short 已完整补跑 SOFA/H；
6. 30 秒后的 H placement、GAME-P 和有效帧覆盖完整；允许的 token 覆盖/截断计数可复算；
7. strict loader 对超限样本明确报错，不裁剪；
8. 最长 Long 在完整 H + GAME-P + 三层 CKA 图上的单卡 forward/backward/update 和显存门禁；
9. V5-P official VAE 推理 shape smoke 与 V5-Pg 285k VAE 绑定 smoke；
10. 四卡真实 mixed pool 10-step smoke；
11. 四卡 continuous-10 与 5+resume5 的 model、EMA、optimizer、scheduler、rank RNG、sampler
   cursor、pool 累计记录/秒数全 section bit-exact；
12. checkpoint 审计绑定 official source、P500 transition、renderer、GAME、VAE、数据和代码 SHA；
13. 所有门禁使用独立 smoke 输出，正式目录只能由通过后的 fresh run 创建。

## 11. 评价与通过条件

### 11.1 固定短集

沿用现有固定 27 组、相同 A/B、seed、sampling steps 和 CFG 3/1，至少比较：

```text
V5-P / V5-Pg
V4PH-30K-HIGHLR
V4H / V4Hg
V4fg
```

分别记录咬字时间、音符跟随、音色细腻度、发声区宽带纹理、极高音、混音输入和跑调。

训练期间的 periodic EvalPool 与听评 27 组分开：固定 Short Eval 沿用现有 201 条并重新导出统一
V5-H schema；另从非训练来源建立固定 Long Eval，保证与 TrainPool 的 Path/BV/Long 成员级 overlap
均为 0。每次 eval 分别报告 Short/Long 的 FlowA、FlowB、CKA、H fallback、collision/truncation，
Long 额外报告存在 30 秒后区域时的 post-30s FlowB 健康指标。

### 11.2 长度边界集

新增以 `B=29s / 31s / 40s / 55s` 为核心的长程验收集，同时记录
`A + 0.5s reference tail + B` 的总 latent 时长。固定 checkpoint、输入内容、seed、CFG、H、MIDI
与 VAE，检查：

- 29 秒到 31 秒是否仍出现断崖；
- 30 秒后的 H token 与音符是否继续生效；
- 后段是否出现噪音、电音、跑调、胡言乱语或声学坍坏；
- 30--60 秒能力是否以牺牲短样本质量换取。

推理接口允许 `B <= 60s`，但当前 Long 训练记录的 30--60 秒是 A+B 总长，不代表训练分布完整
覆盖 B=60s。V5 不对接近 60 秒的边缘质量作承诺；超过 60 秒的 B 属于接口外输入。

### 11.3 V5-Pg 通过条件

- H 的咬字时序收益保留；
- GAME-P 音符控制不退化；
- V5-Pg 音色不再明显落后 V4fg/V4Hg；
- `>30s` 不再出现系统性断崖；
- `<30s` 固定集没有明确回归；
- g 改善音色时不破坏 P/H/L 已获得的结构能力；
- checkpoint、推理 VAE 和全部条件 provenance 审计通过。

Loss、FlowB、CKA 或频谱单项改善均不能覆盖人耳否决。

## 12. V5-Sg 后续边界

V5-Sg 是从官方 checkpoint 直接进入 g 域的 continuous SOME 路线。它不再先训练 official VAE
的 V5-S non-g，再切换 285k VAE；g 从 step 0 加入。当前只完成合同设计，不实现、不训练。

```text
source: official YingMusic-Singer-Plus checkpoint
VAE: frozen 285k online，从 step 0 使用
MIDI: frozen continuous SOME，从 step 0 使用
H/L: 与 V5-P 完全相同的 H/PUL、L_accept、ShortPool、LongPool、L-DEDUP
P: 不加载 GAME-P embedding，不执行 P-only Phase A
steps: 40k
effective batch / seed / dropout / loss: 与 V5-P Phase B 相同
```

### 12.1 V5-Sg LR 合同

以 V5-P Phase B 原 40k 曲线为基准，只将 warmup 从 2k 翻倍为 4k；其余里程碑和 LR 值不变：

```text
0--4k: linear warmup 0 -> 1.4e-5
4k--28k: cosine shoulder 1.4e-5 -> 1e-5
28k--40k: cosine decay 1e-5 -> 0
```

里程碑为 `0/2k/4k/16k/28k/34k/40k = 0/7e-6/1.4e-5/约1.2e-5/1e-5/5e-6/0`。
总 LR 面积约 `0.376 step*LR`，与原 V5-P 40k 的 `0.386 step*LR` 接近；warmup 延长只用于
缓冲 official checkpoint 直接进入 285k latent 域，不改变总训练目标。

### 12.2 V5-Sg 训练入口改造方案

母本固定为 V4Hg 的 `package_v4c_finetune/train/train_plus_h.py`，复制为独立
`train_v5sg.py`；shell 母本为 `run_sft_v4hg_10k.sh`。V4Hg 已原生包含 285k VAE、continuous
SOME、g transition、warm-start、CPU EMA 和 H/PUL 语义，适合作为 V5-Sg 的起点。V5-P 的代码
只吸收 L-FULL-LENGTH-TIME、显存/最长样本、40k scheduler、metadata 和 exact-resume 门禁，
不再反向拆除 GAME-P。

V5-Sg 复制后必须改动：

1. 保留 V4Hg 的 continuous SOME teacher 和 `smoothMelody_MIDIFuzzDisturb` 语义；新增 V5
   SOME provenance/cache schema 与固定 teacher SHA。若最长样本显存 probe 不通过，再切换为
   native-time-axis 离线 SOME cache，不改变训练语义。
2. 将 V4Hg 的旧 H manifest、30 秒截断和 basename 绑定替换为 V5 frozen mixed manifest、
   canonical Path + audio SHA；Short/Long/Eval
   的完整 waveform 重新生成或逐条验证 SOME，Long 不拼旧 Short tensor。
3. 从 V4Hg 的 g transition 改为 official checkpoint fresh：raw 与 EMA 均从 official raw 权重
   初始化，optimizer、scheduler、RNG、sampler、EMA
   counter 均 fresh。V5-Sg 不继承 V5-P 的 P500、40k EMA 或任何 g checkpoint。
4. 将 V4Hg 旧 LR/10k/30 秒合同替换为本节 40k LR；VAE encode、batch/grad accumulation、
   CPU EMA 和显存释放逻辑吸收 V5-P 已验证实现。
5. 新增 `v5sg_training_checkpoint_v1`、step-0/source/SOME cache 逐张量审计、最长 60 秒更新、
   四卡 smoke、exact-resume、formal/resume 和 final audit；输出、tmux session 与 V5-Pg 的比较是完整路线比较：两者均使用 285k VAE，但起点分别为 official fresh 与
V5-P 40k EMA；不能把结果标成纯 SOME/GAME 单变量。V5-Sg 完成后先做 40k raw/EMA 轨迹听评，
不自动启动任何 Sg 后续 g 加训。

首轮资源和实现仍优先当前 V5-P；V5-Sg 不得并发占用其正式输出、GPU 或公共文件，训练授权另行
记录。

## 13. 当前执行状态

```text
design_frozen = true
active_route_authorized = true
first_route = V5-P
completed = Phase A 500 + Phase B non-g 40k + Phase C low-LR g10k
selected_non_g = V5-P 40K EMA
running = V5-Pg20-HLR07 20k（从同一 V5-P 40K EMA fresh 起步）
next_required_work = 完成 20k -> final audit -> 1k--20k raw/EMA 1080-WAV 轨迹 -> 与低 LR 10k 同条件听评
deferred = 任何额外 g 加训（包括完整 g40k）
not_authorized = V5-S；任何额外 g 加训
```

2026-08-08 的首轮授权是完成门禁后执行 Phase A，并在 P500-v2 裁决通过后执行 Phase B 40k；
该历史授权已经履行完毕，不再是“下一步”。任何新增正式路线仍必须在本文追加实际资产路径、
SHA256: redacted

### 13.1 2026-08-08 实际冻结资产

```text
Long Whisper: 3139/3373 = 93.0626%
Long SOFA/H: 3129/3139 = 99.6814%
L_accept: 3129
ShortPool: 8074
TrainPool: 11203
Train loaded hours: 87.138097h
Long loaded-seconds fraction: 46.7906%

frozen root: ${SERVER_ROOT}/V5P_20260808/data/frozen
train_short SHA256: redacted
train_long SHA256: redacted
train_mixed SHA256: redacted
eval_mixed SHA256: redacted
pool_audit SHA256: redacted
H config fingerprint: 67eae32578eeb9a8d0c338dfb6e724c697ad8c3578ec061c0ec3a0f52974dd07

GAME manifest: ${SERVER_ROOT}/V5P_20260808/data/game_final/manifest.json
GAME manifest SHA256: redacted
GAME exact Paths: 11428
GAME unique caches: 11423
```

### 13.2 已通过工程门禁

- 最长 `59.995034s` Long 的 official VAE + H + GAME-P + 三层 CKA 单卡更新通过；
- joint 全参数最长 Long 单卡图通过，smoke checkpoint 明确禁止作为正式来源；
- 4 卡 mixed 10-step 通过，启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；
- continuous10 与 5+resume5 的 model、EMA、optimizer、scheduler、四 rank RNG、sampler cursor
  和累计 pool metrics 全部 bit-exact；
- official VAE encode/decode smoke 有限且非空；
- 独立 `train_v5p.py`、双余弦 LR 单测和 `v5p_eval_batch.py` 已落地并通过静态检查。

### 13.3 Phase A 裁决与 Phase B 显存闭合

```text
Phase A P500 SHA256: redacted
Phase A adjudication SHA256: redacted
P300 projected RMSE: 0.3357250392
P500 projected RMSE: 0.3238271177
decision: random_p_pass
Aliyunpan: /V5/V5P/PhaseA_20260808
```

Phase B 的完整 Adam 状态与未切块 60 秒 official VAE 同时驻留时，GPU EMA 会在 24GB 卡上 OOM。
正式 Phase B 因此冻结为 CPU-resident EMA：raw model、loss、Adam、VAE latent、EMA beta 和每 step
更新频率均不变，仅 EMA 镜像驻留 CPU。禁止用误差较大的 chunked VAE 绕过显存问题。CPU EMA 的
4 卡 mixed10、transition audit 和 continuous10 vs 5+resume5 bit-exact 已通过。

### 13.4 Phase B 40k 完成与归档

2026-08-10，Phase B 从 `step_018000.pt` 精确恢复后在物理 GPU set、2、3、4 完成 40k，退出码为 0。
恢复保持 model、EMA、optimizer、scheduler、四 rank RNG、sampler cursor 和 LR 日程连续；GPU set 未参与
恢复后的正式训练。

```text
final checkpoint: ${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v5p/step_040000_final.pt
checkpoint bytes: 7294114269
checkpoint SHA256: redacted
final parameter SHA256: redacted
final Short Loss: 2.0827
final Long Loss: 2.1066
Aliyunpan: /V5/V5P/PhaseB_20260808
```

### 13.5 每 2k EMA/raw CFG1 实名轨迹听评包

按用户裁决，对 Phase B 保存的 2k、4k、...、40k 共 20 个 checkpoint，同时用 EMA 与 raw 权重
生成固定 CFG 1.0 的实名轨迹。输入、H alignment、official VAE、GAME-P、sampling steps 和 seed
均保持冻结，只允许 checkpoint step 与权重来源变化。

```text
checkpoint count: 20
weight sources: EMA, raw
condition count: 40
groups per condition: 27
WAV count: 1080
sampling steps: 32
CFG: 1.0
seed: 42
dataset manifest SHA256: redacted
alignment manifest SHA256: redacted
package SHA256: redacted
server root: ${SERVER_ROOT}/V5P_20260810/eval_trajectory_2k_40k_ema_raw_cfg1
Aliyunpan: /V5/V5P/Trajectory_2K_40K_EMA_RAW_CFG1_20260810
local registry: ${LOCAL_EXPORT_PATH}
```

40 条单组 smoke 全通过；全量审计确认 1080 条均为 44.1kHz 单声道、有限、非静音，最大 B 时长
偏差 `0.0214058957s`，跨条件输入不一致 0，EMA/raw 逐字节相同对 0。物理 GPU set、2、3、4
四卡分片生成，GPU set 未参与。服务器上传cloud storage后，本机通过 P 盘 `robocopy` 拉取并复核 ZIP
SHA256: redacted

### 13.6 轨迹人耳裁决

2026-08-10，用户完成实名轨迹试听后的裁决为：

- `40K` 基本是当前轨迹中听感最好的 checkpoint；
- 同一步数的 raw 与 EMA 主体质量接近，没有足以改变步数选择的结构性差距；
- EMA 的稳定性略优，因此 V5-P non-g 默认发布/后续 warm-start 来源冻结为 `40K EMA`；
- `40K raw` 保留为技术对照，不作为默认推理权重；
- 该结论是本次固定 27 组、CFG 1.0 实名听评的人耳裁决，不伪装成独立客观指标，也不外推为
  任意 CFG、任意输入或 V5-Pg 的先验结论。

该裁决与原 Phase C 合同一致：随后用户已授权启动 V5-Pg，raw model 与新 EMA 均从
V5-P `40K EMA` 初始化，并按既定 285k online VAE、历史 g 低 LR 10k 配方完成训练。

### 13.7 Phase C 可执行实现：启动前历史快照（2026-08-10）

> 状态说明：本节保留 2026-08-10 授权前的实现与门禁快照。其中“尚未授权”“待执行”只描述
> 当时状态；实际授权、门禁、完成结果以 13.8 为准，当前路线以 13.9 为准。

当时 Phase C 已在本机实现为独立 `g_adapt` 模式，但尚未据此宣称服务器门禁通过或正式训练已授权：

- checkpoint schema 独立为 `v5pg_training_checkpoint_v1`，不复用 V4Hg schema；
- source 只接受完整的 V5-P Phase B 40k checkpoint，实际 SHA256: redacted
  `3a532f5bd5965dff7d011996b7ca72d7884c5494a2d44d6c28b0bab21bace96c`；
- 训练 VAE 只接受 SHA256: redacted
  `f18aeecacc04173cd2ea73bbdf8edae9e976d18e4ca050c38e2723281c5cba85` 的 285k online；
- raw model 与新 EMA 同时从 source EMA 初始化，EMA counter 从 40k 连续；optimizer、scheduler、
  RNG、sampler 与 Phase-C global step 从 0 fresh；
- H/PUL、GAME-P、L/Short 去重池、loss、dropout、reference 与 Phase B 逐项绑定，P/PUL 不重新初始化；
- LR 硬约束为 `0--250 warmup -> 250--6250 hold 5e-6 -> 6250--10000 cosine to 0`；
- 保存硬约束为每 1k，Eval 硬约束为每 500 step，并额外生成 step-0 transition checkpoint/Eval；
- 新增 step-0/source EMA 逐张量审计、任意 step optimizer/LR/EMA/RNG/cursor 审计，以及
  `continuous10 == 5+resume5` 全 section bit-exact 比较器；
- 新增 285k VAE 最长 Long 单卡更新、四卡 smoke、formal/resume、tmux session artifact
  上传脚本；默认物理 GPU 为 1、2、3、4，正式启动前仍必须实时检查共享 GPU 占用。
- 新增 Phase-C 1k--10k EMA/raw、CFG 1.0 实名轨迹流水线：10 个 checkpoint、20 个条件、固定
  27 组，共 540 WAV；checkpoint、H/GAME/input、音频完整性、实名打包和cloud storage上传均有独立
  V5-Pg schema。step-0 transition 作为因果控制保留，不与必然相同的 raw/EMA 重复塞入该轨迹包。

本机与服务器静态状态：训练/推理入口、checkpoint 审计器和 exact-resume 比较器通过
`py_compile`，全部 Bash 通过 `bash -n`，Phase B/Phase C LR 共 4 项单测通过；服务器实测
`LambdaLR` 在 step `0/250/6250/10000` 分别为 `0/5e-6/5e-6/0`，source checkpoint 契约通过。
泛化后的 Phase-B 默认轨迹审计已反跑原 20 checkpoint/1080 WAV，结果仍为 20/1080、错误 0。
在该历史时间点，最长样本更新、四卡 smoke、exact-resume 与正式启动仍是待执行门禁，因此本节
本身不构成 Phase C 已启动的证据；这些项目随后均已通过，见 13.8。

### 13.8 Phase C 门禁通过与正式启动（2026-08-10）

用户授权后，Phase C 按最长样本 → 四卡 exact-resume → 独立审计 → 正式训练的顺序执行：

```text
tmux session
output: ${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v5pg
log: ${SERVER_ROOT}/V5PG_20260810/logs/v5pg_phase_c_10k_fresh.log
source: V5-P step_040000_final.pt / EMA
source SHA256: redacted
VAE SHA256: redacted
```

- 59.995 秒最长 Long 的 285k VAE、H/PUL、GAME-P、三层 CKA、backward、optimizer/EMA 与
  checkpoint 保存通过；step-0/step-1 audit 均为 `status=ok`，无 OOM/NaN/结构异常；
- 四卡 `continuous10` 与 `5+resume5` 对 model、EMA、optimizer、scheduler、四 rank RNG、sampler
  cursor、累计指标和全部 provenance 的比较为 `failure_count=0 / bit_exact=true`；
- 四卡 step-0 audit 确认 raw/new EMA 逐张量等于 source EMA、optimizer LR=0、EMA step=40000、
  rank cursor=`(0,0)`；step-10 audit 为 LR=`2e-7`、EMA step=40010、cursor=`(0,40)`；
- 正式任务于服务器 UTC `2026-08-10 09:17:29` 启动；step-0 Short/Long Eval Loss 分别为
  `3.2465/3.1869`，它们只作为 285k latent 域内基线，不与 official VAE loss 等价比较；
- step 25/50 的 FlowB 为 `1.2770/1.2222`，LR 为 `5e-7/1e-6`，速度约 `3.60s/step`、ETA 约
  `9.9h`；四 worker 与 torchrun 存活，日志无 OOM/NaN/NCCL/Traceback；
- step 25 的 placement、PUL、Exact、ControlAnomaly、Structural、PULFrames 与 NonPAD 和 Phase B
  同一数据顺序逐项一致，证明正式 Phase C 没有改变 renderer 或 sampler 起点。

正式 10k 于服务器 UTC `2026-08-10 19:54:19` 完成全部训练、审计和cloud storage上传，pipeline
`rc=0`。final Short Eval 为 Loss/FlowB `2.2743/0.9494`，Long 为 `2.3028/0.9911`；final
checkpoint audit 为 `status=ok`，EMA step=`50000`，四 rank cursor=`(14,786)`，日志无
OOM/NaN/NCCL/Traceback。final 与 SHA、step-10000 audit、全 checkpoint SHA 清单已上传：

```text
Aliyunpan: /V5/V5P/PhaseC_20260810
final: step_010000_final.pt
```

该结果只确认低 LR 10k 工程完成，不提前宣称感知晋级；1k--10k EMA/raw CFG1 实名轨迹仍需听评。

### 13.9 V5-Pg20-HLR07 强化适配路线（2026-08-11）

用户在低 LR 10k 完成后授权新增独立强化路线，用于检验历史 g 配方是否因 LR/step 预算不足而停在
285k latent 域约 `0.95` 的 FlowB 平台。它不是 10k resume，而是与 10k 共享同一个 V5-P 40k
EMA 起点的平行因果路线：

```text
source: V5-P step_040000_final.pt / EMA
source SHA256: redacted
VAE SHA256: redacted
steps: 20000
0--1000: linear warmup 0 -> 9.8e-6
1000--14000: cosine shoulder 9.8e-6 -> 7e-6
14000--20000: cosine decay 7e-6 -> 0
save_every: 1000
eval_every: 500
```

该 LR 是 Phase-B 40k 双 cosine 在时间轴压缩至 0.5 后，全部 LR 再乘 `0.7`；H、GAME-P、L
去重池、loss、dropout、reference、seed、四卡 effective batch 与 CPU EMA 均与低 LR 10k 相同。
raw/new EMA 仍从 source EMA bit-exact 初始化，EMA counter 从 40k 延续；optimizer、scheduler、
RNG 和 sampler 从 0 fresh。10k 与 20k checkpoint/输出目录不得相互覆盖。

正式启动前完成：

- 本机与服务器 `py_compile`、全部 Bash `bash -n`、六项 LR 单测通过；
- 新曲线冻结点 `0/500/1000/7500/14000/17000/20000` 分别为
  `0/4.9e-6/9.8e-6/8.4e-6/7e-6/3.5e-6/0`；
- 四卡 `continuous10 == 5+resume5` 全 section 比较为 `failure_count=0 / bit_exact=true`；
- step-0 audit 为 `status=ok`：raw/new EMA 等于 source EMA、optimizer LR=0、EMA step=40000、
  cursor=`(0,0)`；continuous/resume step-10 audit 均为 LR=`9.8e-8`、EMA step=40010、
  cursor=`(0,40)`；
- 门禁日志无 OOM/NaN/NCCL/Traceback，启动前物理 GPU set 空闲，磁盘剩余约 `750GB`。

正式任务：

```text
tmux session
output: ${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v5pg20_hlr07
log: ${SERVER_ROOT}/V5PG20_HLR07_20260811/logs/v5pg20_hlr07_20k_fresh.log
launch claim: ${SERVER_ROOT}/V5PG20_HLR07_20260811/control/launch_claim
```

任务于服务器 UTC `2026-08-11 00:31:23` 启动。正式 step-0 Short/Long Eval 与低 LR 10k 起点
逐项相同，独立 step-0 audit 为 `status=ok`；step 25/50/75 的 LR 严格为
`2.45e-7/4.9e-7/7.35e-7`，step 75 FlowB=`1.2040`。torchrun 与四 worker 存活，仅物理
GPU set 有训练负载，GPU set 未参与，初始日志无异常。20k final 不自动成为发布权重，仍须与低 LR
10k 做同输入、同 seed、同 CFG 的轨迹听评。

首个同 step 比较：

| 路线 | step | Short Loss/FlowB | Long Loss/FlowB |
|---|---:|---:|---:|
| V5-Pg 低 LR | 1000 | `2.3547 / 0.9818` | `2.3877 / 1.0223` |
| V5-Pg20-HLR07 | 1000 | `2.3569 / 0.9820` | `2.3948 / 1.0239` |

两条路线在 1k 尚未分化：低 LR 路线从 step 250 起已处于 `5e-6` hold；HLR07 到 step 1000
才结束长 warmup、首次达到 `9.8e-6`。是否突破平台主要看 2k--14k shoulder，而不能用 1k
提前裁决。

20k 完成后冻结实名轨迹为 step 1k--20k、raw/EMA、固定 27 组、CFG 1.0、32 sampling steps、
seed 42，共 `20 * 2 * 27 = 1080 WAV`。step-0 transition 与低 LR 路线相同，只保留作因果审计，
不重复生成 raw/EMA 听评条件。低 LR 10k 与 HLR07 必须使用相同输入、H/GAME、VAE、seed 和
sampling 配置；最终 checkpoint 由听感与结构门禁共同选择，不按最末 step 自动晋级。

### 13.10 可能的 40k g 路线边界（已搁置，未授权）

2026-08-12，用户决定先记录并搁置额外 g 加训。当前 V5-Pg20-HLR07 正常完成既定 20k 后即停止；
不直接续训，不自动启动本节路线，不实现额外 launcher/门禁，不预留 GPU。下一步只完成 20k final
审计和既定轨迹听评。只有听评出现新的、足以支持更大预算的证据，并由用户再次明确授权，才恢复
本节讨论。

若未来验证完整的 Phase-B 40k 曲线乘 `0.7`，冻结候选只能是从 V5-P 40K EMA 再次 fresh 起步：

```text
0--2000: linear warmup 0 -> 9.8e-6
2000--28000: cosine shoulder 9.8e-6 -> 7e-6
28000--40000: cosine decay 7e-6 -> 0
```

它与当前时间压缩的 20k 曲线不等价，因此不得把 V5-Pg20-HLR07 直接续到 40k 后冒充该路线。
当前状态为搁置：不实现、不预留 GPU、不授权训练。未来若重新立项，至少要求 20k 结果同时满足：

1. 10k--20k 的固定 Short/Long Eval 仍有稳定、非噪声级改善，而不是已进入平台；
2. 16k/18k/20k 的实名听感持续优于更早 checkpoint，最佳点仍压在训练尾部；
3. H 咬字、GAME-P 音准、Long 后段、高音和自然度均无随高 LR/长训练产生的回归；
4. 20k 末端像被 LR 归零截断，而不是已经自然收敛或开始反弹。

任一关键条件不成立即不跑 40k。`FlowB < 0.85` 目前只是待验证假设，不是训练通过门槛；跨
official/285k VAE 的绝对 FlowB 仍禁止直接等价比较。

















