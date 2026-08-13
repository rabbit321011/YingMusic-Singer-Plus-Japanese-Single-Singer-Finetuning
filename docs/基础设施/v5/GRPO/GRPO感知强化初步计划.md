# GRPO 感知强化初步计划

## 1. 目标

在最终人工裁决的 V5-Pg checkpoint 上执行小步、强约束的 Flow-GRPO，直接优化 SFT loss
难以表达的终端感知质量。首轮目标按优先级分为三轴：

1. 保持target singer目标样本的全局唱法与音色特征；
2. 保持 V5-P 的 GAME-P 音高结构与不确定性分布；
3. 降低随发声出现的毛刺、破音边缘和砂噪，同时提高整体自然度。

本路线不是歌词强化实验。H/PUL 已负责歌词与时间放置，PER 不进入每步 reward，只在周期评估中
监控歌词能力是否回归。

算法定义以 [`GRPO原论文核验与算法合同.md`](GRPO原论文核验与算法合同.md) 为前置。原始 GRPO、
Flow-GRPO 论文公式、官方代码近似和本项目推导必须分层记录；本文只负责执行顺序。

## 2. 谱系与继承边界

```text
V5-P 40K EMA
  -> V5-Pg 低 LR 10k
  -> V5-Pg20-HLR07 独立 20k
  -> 固定轨迹、人耳与结构门禁裁决唯一基座
  -> GRPO 离线 reward 标定
  -> GRPO 工程门禁
  -> 小步 pilot
  -> 人工裁决后才讨论正式训练
```

GRPO 基座不得按最后 step 自动选定，也不得继承旧 V4c GRPO 的模型、optimizer 或 reward 状态。
首轮保持以下 V5-Pg 合同不变：

- 338M DiT 架构；
- H/PUL placement 与统一 kana；
- GAME medium K=4、离散 P class 和 `[T,128]` CKA probability target；
- V5-P Short/Long 去重池语义；
- frozen 285k online VAE；
- reference tail、A/B 和推理输入契约；
- 最长 B 接口与既有质量不承诺边界。

GRPO 首轮不加入 INS adapter、LCF、SVC 自克隆、无 A、扩模或新的 VAE。INS 只作为冻结评分器，
不进入模型条件通道。

## 3. 候选组合同

- 每个 prompt 生成 `G=8` 个候选；
- 组内的 waveform、H/PUL、GAME-P、target、时长和 reference 完全相同；
- 候选采样使用完整条件，audio/text/midi/INS scorer 均不做训练 dropout；
- 主候选为同组共享同一个初始 latent，仅由真实 SDE window 的独立噪声产生差异；每候选独立初始
  latent 只作为离线对照，是否采用由探索幅度与 credit attribution 实测裁决；
- 候选只允许由冻结且可 replay 的随机探索机制产生差异；
- reward 只在同 prompt 的八个候选内比较，不跨任务使用绝对分数；
- 任一候选非有限、静音、长度异常、真正数字削波、transition/replay 不一致或 scorer 调用失败时，
  组状态为 `GROUP_INVALID`；任一 INS/PITCH/UTMOS22/DNSMOS SIG 轴输出低于 reliability 门槛、重复
  误差/范围/对齐审计失败时同样为 `GROUP_INVALID`；任何失败都跳过整组 `G=8`，不计算 advantage，
  不 `optimizer.step()`；
- 每个 `GROUP_INVALID` 必须记录 `rollout_id/group_id/prompt SampleId/Pool/Short-Long`、manifest/
  policy/reference/scorer provenance SHA256: redacted
  mask、音频 hash/shape/sr/duration/RMS/peak/finite/nonzero/clipping 统计，以及逐 scorer 的 status、
  error class、异常消息摘要、输入 hash、输出 shape/dtype/finite/range、raw score、reliability threshold、
  重复误差和失败轴；
- 不得填中性分、静默重采样、删除失败候选、只保留剩余轴或重新归一化权重；调试重跑必须新建
  `rollout_id`，与原失败组分开登记；
- scorer 使用的派生波形和训练实际消费的生成波形必须分别登记 SHA256: redacted

rollout CFG 延后裁决。`a=0.8, w_min=1, w_s=8` 先作为 YingMusic-Singer-Plus 论文的默认
baseline 候选；音频先例还包括 `a=0.4`、single-step 或 `w=2` 的选择性 SDE。正式冻结前仍必须完成
`NFE/a/window/同组初始 latent` 标定，并证明采样轨迹、行为策略概率和更新概率来自同一 transition
kernel。若 SDE 候选感知质量或 V5-P 控制能力不通过，不得用 ODE 轨迹配虚构 SDE 方差继续计算 PPO
ratio。

SDE 已裁决为 B-only：只有有效 B frame 可注入 SDE noise，并施加相同 mask 的 drift correction、
log-prob、ratio 和 KL；A、PAD、reference tail 与 window 外坐标保持 ODE。先按论文参数运行 baseline，
再按算法合同的 single-step/低噪声对照、累计 diffusion budget、硬门禁、探索效率、等预算短 pilot
和 one-standard-error rule 冻结最终 `a/window/NFE`。

本项目从噪声 `t=0` 积分到数据 `t=1`。候选 transition 必须按权威算法合同使用换元后的 drift：

```text
sigma(t) = a * sqrt((1-t) / t)
b_theta  = v_theta + a^2/(2t) * (-x_t + t*v_theta)
x_next   = x_t + b_theta*delta_t + sigma(t)*sqrt(delta_t)*epsilon
```

`t=0` 不得进入 SDE window；window 外使用真实 ODE 且不参与 ratio/KL。time-shift 后的实际 `t_j`、
`delta_t`、B mask、noise 和 transition mean 必须保存。

## 4. Reward 定义

### 4.1 INS 唱法相似度

```text
generated B waveform -> frozen ParaSpeechCLAP Intrinsic -> e_gen [768]
target B waveform    -> frozen ParaSpeechCLAP Intrinsic -> e_target [768]
R_INS = cosine(e_gen, e_target)
```

冻结语义：

- encoder 为 `ajd12342/paraspeechclap-intrinsic` 的既有固定 revision 与 checkpoint SHA256: redacted
- 输入按 V4IjPH 权威路径转为 16 kHz；
- 调用 `get_audio_embedding(normalize=True)`；
- 直接比较 768 维 L2-normalized embedding，不经过 768 -> 64 INS adapter；
- target 必须是当前训练记录实际送入 VAE 的目标 B waveform，不得使用旋律源之外的替代样本；
- 组内候选共享 prompt、target、音高条件和 reference，固定的音素/内容因素大多在组内比较中抵消；
  首轮将 `R_INS` 作为项目主要目标的唱法相似度轴使用，不额外设置 INS 专项反事实审计前置门槛。
  通用的人耳、held-out 回归与 reward-hacking 停止条件仍然适用。

### 4.2 GAME 概率音高对齐

模型条件仍使用 V5-P 的单一离散 P class，但 reward 不退化为单轨 F0 Pearson：

```text
generated full track -> frozen GAME -> estimator posterior -> [T,128] generated_probs -> exact B slice
target full track    -> frozen GAME/cache -> cka_probs      -> [T,128] target_probs    -> exact B slice
```

两端采用方案 A：先对完整的 target 与 generated track 运行 GAME，再依据同一个 latent-frame 时间
映射精确切出 B。generated track 必须包含与 V5-Pg 推理契约相同的 A+B(+reference tail) 上下文；不能
只对生成 B 单独运行 GAME 后直接与整段 target cache 的 B 片段比较。A、B 边界、尾部和 GAME 时间插值
索引必须进入 prompt/rollout receipt，确保 resume 时可以重建完全相同的 B slice。

两端必须使用 V5-P 已冻结的 GAME medium K=4、时间轴和 pitch-axis adapter：

```text
sigmoid(GAME estimator logits)[..., 0:255:2]
REST -> zero
```

正式 scalar 冻结为按 target voiced/REST 分层均衡的 probability cosine。目标 voiced 且生成 voiced
时比较两个 128 维 GAME posterior；目标 voiced/生成 REST 与目标 REST/生成 voiced 均为 0；双 REST
为 1；最后对 voiced 与 REST 两类分别求均值再平衡平均。这样保留绝对音高约束并能发现错八度，
同时不让长静音帧淹没音高信号。

PITCH_PROB 的职责是防止 SIM/QUA 优化时旋律、绝对音高、REST/voiced 和边界跑偏，不是首轮微调的
主要目标；外层名义权重仍为 `SIM_INS:PITCH_PROB:QUA = 3:2:4`，PITCH 的 `2` 作为护栏预算。Linear
CKA 与论文口径的 F0-CORR 只作离线诊断、结构回归和 reward-hacking 门禁，不进入首轮主 reward。
正式门槛、空 voiced/REST 类处理、scorer 重复误差和对齐产物由离线多 seed 实验冻结；不得只用
argmax 音符或 decoded score 重建 Gaussian 后丢弃 estimator posterior。

### 4.3 QUA 感知质量

`QUA` 是复合质量轴，不再表示旧 V4c GRPO 的 DNSMOS P808：

```text
U10   = Q100_UTMOS(UTMOS22)
SIG10 = Q100_SIG(DNSMOS_SIG)

C_U   = reliability_U   * (U10   - mean_group(U10)) / 10
C_SIG = reliability_SIG * (SIG10 - mean_group(SIG10)) / 10

A_QUA = 0.7 * C_U + 0.3 * C_SIG
```

- UTMOS22 与 DNSMOS SIG 先通过各自冻结的训练原音 Q100 单调分段线性映射并 clip 到 `0--10`，
  再在同一 `G=8` 组内居中，最后按区间宽度 `7:3` 合成；不再逐组除以当组 std；
- 报告时将 `U10/SIG10` 压到 `U7/SIG3` 并相加为 `QUA10`；
- 不能先混合原始 MOS 再统一映射，也不能训练中在线重估 Q100 锚点；
- QUA 使用 44.1 kHz 单声道 PCM24 等响副本，线性增益统一至 `-28 dBFS RMS`；
- QUA 评分单元固定为 B-only：runtime 评分 generated B，5k Q100 评分抽样记录 materialize 后的
  target B；两者的裁切、最短长度处理和 scorer 前处理必须同构，A/PAD/reference tail 不进入 QUA；
- 等响前必须通过非静音、有限值、发声覆盖和峰值门禁；
- 不对评分副本限幅、降噪或改频谱，训练消费的生成波形不做等响替换；
- UTMOS22 当前暂用组内极差 `<0.02` 时关闭该分量；DNSMOS SIG 门槛待标定。

### 4.4 PER 边界

PER 的训练权重固定为 `0`。原因包括：

- 旧 V4c GRPO 的完整 reward 阶段平均 `14.76s/step`，占总步时 `31.5%`，Whisper 还独占一张 GPU；
- 降低 PER 权重不会减少 ASR 前向成本；
- 歌唱日语 ASR 的测量噪声较高，旧 GRPO 最佳 PER 改善未达到显著性；
- 强化 ASR 易识别度可能牺牲自然歌唱辅音、音色或 QUA。

PER 保留为固定 Short/Long checkpoint 的周期离线门禁，与 H/SOFA 错槽率和人工听评共同判断
歌词能力是否回归。只有首轮 pilot 明确出现 H/PUL 无法捕获的感知歌词退化，才重新讨论低频或
低权重 PER；不得在未证明必要性前恢复每步 Whisper。

正式报告使用未截断、越低越好的 `KANA_CER`。为复现 V4c 历史表格，可同时派生
`V4C_PER_SCORE = max(0, 1-KANA_CER)`，但该兼容字段不作为 V5 的正式错误率名称或回归方向。
固定 Eval 逐样本与 step-0/reference 配对：超过 `20` 条样本各自下降超过 `20%` 时立即停止；超过
`10` 条样本各自下降超过 `10%` 时警告。停止条件优先，并输出全部退化样本而非只报告均值。

H/SOFA 错槽回归门禁放宽为上述歌词门禁的 `1.5x`：固定 Eval 中超过 `15` 条样本的 mora 错槽率
相对 step-0 各自增加超过 `15` 个百分点时警告；超过 `30` 条样本各自增加超过 `30` 个百分点时
立即停止。step-0 可对齐而 checkpoint 对齐失败或 mora 数量异常的样本按完全回归计。

GAME 音高回归使用 `GAME_ERR = 1-PITCH_PROB`。固定 Eval 中超过 `10` 条样本的误差相对 step-0
各自增长超过 `10%` 时警告；超过 `20` 条样本各自增长超过 `20%` 时停止。增长率分母采用
`max(GAME_ERR_step0, scorer_noise_floor)`，其中噪声下限在 pilot 前通过重复评分标定。voiced/REST、
绝对音高与 F0-CORR 只用于归因，不再各设一套停止门槛。

不另设自动 `proxy hacking` 停止条件。已有有效性、自动回归、训练健康和人耳 early-stop/veto 各自
执行门禁；reward 上升却被人耳否决或触发已有回归门禁时，仅将 `proxy_hacking` 作为事后归因标签，
记录具体 scorer 偏好及证据。

## 5. Advantage 与外层权重

初始外层权重冻结为：

```text
SIM_INS : PITCH_PROB : QUA = 3 : 2 : 4
```

按总权重 9 归一化：

```text
A = (3 * C_INS
   + 2 * C_PITCH
   + 4 * A_QUA) / 9
```

展开后的名义基础占比为：

| 基础评分 | 名义占比 |
|---|---:|
| INS | 0.3333 |
| GAME probability pitch | 0.2222 |
| UTMOS22 | 0.3111 |
| DNSMOS SIG | 0.1333 |

每个基础评分必须先按自己的冻结训练参考分位映射到共同坐标，再在同一 `G=8` 组内独立居中；不能
先混合 raw reward 再统一缩放。用户已撤销逐组 z-score；QUA 当前采用的合同为：

```text
M_ik = Q100_k(R_ik)  # [0,10]
C_ik = reliability_gk * (M_ik - mean_group(M_k)) / 10
```

均值包含候选自身，不使用 leave-one-out。QUA 的 Q100 来自最终权威训练 manifest 均匀无放回抽取的
5,000 条原始训练音频，固定抽样 seed 为 `20260812`；使用 100 个等百分位锚点、`linear` quantile
和 99 段单调直线，训练中不能在线变化。完整定义见
[QUA Q100 评分映射合同](QUA_Q100评分映射合同.md)。这样：

- 若某组只有很小但可靠的差异，centered signal 仍保持很小，不被强制放成单位方差；
- 若不同 scorer 的自然量纲、范围和分布形状不同，由各自 Q100 映射到共同百分位坐标；
- 若范围小是 scorer 数值噪声或无区分力，由 `reliability` 关闭该分量。

reliability 首轮使用离线冻结的二值门，不做在线可学习权重。任一轴关闭即整组 `GROUP_INVALID` 并
跳过，不把剩余轴自动重归一化；外层 `3:2:4` 和 QUA `7:3` 表达压缩后允许区间宽度；具体组保留真实
幅度。门槛和 Q100 表必须由极值试听、结构有效性、重复误差和梯度统计共同冻结。

## 6. 离线标定

在最终基座选定后，先运行固定 27 组跨源任务、每任务十个可控 SDE 候选的平衡矩阵。主矩阵同组
固定初始 latent，只改变 SDE noise；另做独立初始 latent 对照。至少输出：

1. 四个基础 scorer 的重复前向和模型重载重复性；
2. 每任务标准差、极差、两两分差和 `G=4/G=8` 组合覆盖；
3. task、固定 seed 与 task x seed 方差分解；
4. INS/PITCH/UTMOS/SIG 的组内相关矩阵，识别重复奖励和系统冲突；
5. 每轴最高/最低候选的等响或权威预处理试听/审计包；
6. Q100 合同下 `3:2:4` 复合 advantage 的候选排序、极值、有效组比例，以及按失败轴/原因分层的
   `GROUP_INVALID` 比例；
7. 每轴 100 个锚点的 bootstrap 稳定性、局部斜率、clip 率，以及压缩后实际 std 与允许区间宽度；
8. reliability 门槛的 bootstrap 置信区间和跨任务稳定性；
9. 静音、发声覆盖、高音、长段后半段、混音源和轻柔音色等困难域分层统计。

`reliability_k` 必须同时参考：

- scorer 数值重复误差；
- 同任务 seed 的真实离散；
- 极值候选的人工或结构差异；
- 该轴与其他轴冲突时是否出现 reward hacking。

UTMOS22 在 V4PH/V4fg 上的既有 `<0.02` 门槛只是先验，不能替代最终 V5-Pg 基座复算。

## 7. V4c GRPO 复用与禁止项

可以复用：

- `G=8` 的同 prompt 相对奖励框架；
- 行为策略、当前策略和冻结 reference 的三角色设计；
- 有限 SDE 窗口、PPO clip 与 KL 保护思想；
- reward worker 与训练 worker 分离的服务器编排；
- 分 checkpoint 多 seed 评价方法。

禁止直接复用最终 `train_grpo_v3.py` 的以下行为：

- `Singer.sample()` ODE+CFG3 采样，却按固定 `alpha=0.8` SDE 计算概率；
- 用冻结 reference 同时冒充 PPO old policy 和 KL reference；
- 先混合原始 reward，再只对总分做一次 z-score；
- 使用 DNSMOS P808 代替 DNSMOS SIG；
- scorer 失败后填中性值继续正式更新；
- checkpoint 缺少 rank RNG、sampler cursor、reward provenance 和 exact-resume 状态。

旧文档声称 V4c GRPO 完成 4800 步，但当前服务器可见 checkpoint 只到 1600，日志停在 1657，
没有 4800/final 产物。该路线的历史效果只能按现存 checkpoint 与评估证据解释，不把 4800 当作
已验证事实。

## 8. 工程实现与门禁

实现必须是 V5-Pg 专用入口，不在旧 GRPO 脚本上就地覆盖。最低门禁：

1. reward 单测：四 scorer、Q100 分段线性映射/clip、组内居中、`3:2:4`、QUA `7:3`、可靠性关闭和失败硬停；
2. provenance：基座、VAE、GAME、INS、UTMOS、DNSMOS、数据 manifest 和全部代码 SHA256: redacted
3. 单候选同构：训练采样前端与 V5-Pg H/GAME/A/B/285k VAE 合同一致；
4. SDE 数学门禁：保存 behavior mean、噪声、`x_t/x_next`、实际 `t/delta_t`、B mask、log probability
   和窗口索引，并能逐步 replay `x_next`；
5. action 门禁：一个 SDE timestep 的完整有效 B transition 是 action；逐帧 64-D 只作精确 log-prob
   contribution。单帧内对 64 channel 求和，再对有效 B frame/SDE step 求 mean，使每条记录等权；
   ODE/A/PAD/reference-tail 坐标全部排除；
6. policy 角色门禁：behavior snapshot、current policy 与 frozen reference 不混用；
7. ratio/KL 门禁：初始 ratio 全 1、初始 KL 全 0、synthetic Gaussian 对照、参数微扰方向正确；
8. optimizer 语义门禁：完整 rollout batch 的所有 microbatch 先累计梯度，之后只做一次
   `optimizer.step()`；同一收据不复用，更新前必须始终 `current==behavior`；
9. 单卡 `G=8` 完整 sample -> decode -> reward -> backward -> optimizer -> checkpoint；
10. 最长样本显存门禁，并据此冻结首轮 prompt 时长上限；
11. 四卡 mixed smoke，所有 rank 都有有效组且 collective 顺序一致；
12. `continuous10 == 5+resume5`：model、optimizer、scheduler、reference、四 rank RNG、sampler
   cursor、reward counters 和 provenance 全 section bit-exact；
13. checkpoint 独立审计与实际 resume；
14. 正式目录 fresh、tmux session step
    分层的 ratio/KL/clipfrac 和异常日志。

不得为了省显存静默改变 VAE、候选组大小、prompt、score preprocessing 或 reward 权重。

## 9. 分阶段执行

### 阶段 0：基座裁决

- 完成 V5-Pg 低 LR 与 HLR07 的同输入轨迹生成、结构审计和人工听评；
- 冻结唯一 EMA/raw checkpoint、SHA256: redacted
- 未裁决前不实现绑定某个临时 checkpoint 的正式 GRPO transition。

### 阶段 1：Q100 与 reward 离线标定

- 从最终权威训练 manifest 均匀无放回抽取 5,000 条训练音频，保存 seed、抽样前后 manifest 和
  SHA256: redacted
- 对 5,000 条参考音频完成 UTMOS22/DNSMOS SIG 评分，按 `linear` quantile 构建每轴 100 个锚点；
- 完成 99 段局部斜率、bootstrap 稳定性、clip 率和重复锚点审计；重复锚点或评分失败必须硬停；
- 完成第 6 节的十 seed 平衡矩阵；
- 冻结 INS、PITCH、UTMOS 和 SIG 的 reliability 规则；
- 冻结 Q100 只读产物、每轴 reliability threshold 与 contribution audit；
- 由用户试听极值包并确认 QUA 与 INS 的方向；
- 输出只读 reward contract 和审计报告。

### 阶段 2：SDE sampler 标定

- 先比较训练候选 NFE 与正式 ODE 推理 control；
- 先运行论文 baseline `a=0.8,w_min=1,w_s=8`；再在小型哨兵集扫描
  `a={0.2,0.4,0.6,0.8}`、single-step/`w=2/4/8` 和同组初始 latent；
- 对入围配置扩展到完整 27 组，冻结 transition、window、action mask 和候选质量门槛；
- SDE 本身的感知或结构门禁不通过时停止，不进入策略更新。

### 阶段 3：工程门禁

- 建立独立入口、checkpoint schema、审计器和 exact-resume 比较器；
- 依次完成静态、单卡、最长样本、四卡和 resume 门禁；
- 门禁 checkpoint 禁止进入后续正式谱系。

### 阶段 4：短 pilot

- 从唯一基座 fresh 启动；
- LR、pilot 步数、保存频率和 KL 强度在门禁报告后单独申请授权；
- 保存 step 0 和密集早期 checkpoint；
- 固定 Short/Long/困难域运行 reward、PER、H/SOFA、GAME 和人工听评；
- pilot 只验证方向，不自动续接正式长训。

### 阶段 5：正式训练

仅在 pilot 同时满足感知、结构和工程门禁后讨论。正式训练必须重新登记用户授权、总步数、LR、
保存/Eval 频率、GPU、tmux session

每个 checkpoint 至少保留：

- INS、PITCH、UTMOS22、DNSMOS SIG、QUA 和总 advantage 的 raw/组内统计；
- 固定 Short/Long Eval；
- H/PUL placement、Whisper+SOFA 错槽与 PER 周期评估；
- GAME probability、pitch-center、voiced/REST 与边界指标；
- 发声覆盖、RMS、峰值、削波、静音比例和长度；
- 固定 prompt 集与固定 seed 集的可复现评测输出；不另设独立 seed 稳健性复核集；
- raw 与 EMA（若启用）的独立 provenance。
- ratio mean/std/quantile、`clipfrac_gt_one/lt_one`、behavior staleness 和每个 SDE step 的 KL；

训练与评测分层冻结如下：

- 每个 rollout 记录 reward/advantage、old/new log-prob、ratio/log-ratio/ESS、正负 clipfrac、KL、loss/
  gradient/LR、`GROUP_VALID/GROUP_INVALID`、失败候选/轴/原因、音频有效性、Short/Long、SDE window、
  walltime 与显存；`rollout_attempt` 与 `optimizer_step` 分开计数；
- 不在每个 rollout 额外运行 Whisper PER 或第二套评测生成；
- 固定 checkpoint 使用固定 prompt/seed 的正式 32-step ODE 输出，完成 reward 轴、GAME 扩展音高、
  F0-CORR/CKA、PER、H/SOFA、Short/Long/Long 后段与音频统计；
- 人耳只在预设节点承担 continue/early-stop/veto，具体频率留到 pilot 参数裁决。

自动指标用于训练监控、checkpoint 比较和回归归因；人耳不进入 reward、advantage 或自动指标总分。
人耳只在预先固定的 checkpoint/pilot 节点作为外部 early-stop/veto，决定路线是否继续，因此不与
PITCH、QUA 或其他自动指标做同一维度的数值排序。

任一条件触发停止或回到设计阶段：

1. QUA 上升但人工噪音、自然度或音色不改善；
2. 出现非有限、全静音、长度异常或真正数字削波等音频有效性失败；适度频谱、辅音、发声覆盖或响度/
   动态变化在 KL/reference 约束下不预先视为 hacking；
3. INS 上升但target singer身份、歌词或绝对音高被 target 泄漏带偏；
4. PITCH 上升但出现错误音高平移、REST/voiced 或边界退化；
5. PER、H/SOFA 错槽、长段后半段、高音或困难域出现稳定回归；
6. 有效 reward group 比例过低，或 advantage 主要来自数值误差；
7. KL、ratio、梯度、loss 出现非有限值或系统性触顶；
8. 正 advantage 长期几乎不触发 ratio 上界，而负 advantage 系统性触发下界；
9. exact-resume、checkpoint provenance 或 scorer hash 失配；
10. 固定节点的人耳 early-stop/veto；这不是覆盖客观 reward，而是独立的路线继续/停止决策。

## 11. 当前未决项

- V5-Pg 最终基座 checkpoint；
- PITCH_PROB 的 reliability 门槛、空 voiced/REST 类处理、GAME scorer 重复误差与时间对齐产物；
- INS 与 DNSMOS SIG 的组内有效极差门槛；
- Q100 实际锚点产物、每轴 reliability threshold 和 advantage clip 上限；
- SDE NFE、`a`、窗口、同组初始 latent、A/B noise mask 和候选质量门禁；
- rollout batch、minibatch、inner epoch、action 聚合和 optimizer step 位置；
- GRPO LR、KL 的硬门禁与调整条件、总步数、pilot 步数和保存频率；PPO clip 已按论文表格固定为
  `epsilon_lower=0.002 / epsilon_upper=0.01`，不再做 clip 范围候选实验；advantage 首轮不额外 clip，
  RatioNorm 只作为失败后的独立变体；
- Short/Long prompt 采样比例与首轮最大时长；
- UTMOS/INS/GAME reward worker 的 GPU 布局和吞吐；
- 全参数还是 LoRA；是否维护 EMA。若有 EMA，只作评价 shadow，不充当 behavior 或 current；
- 普通 Gaussian ratio 是否通过双侧 clip 门禁；GRPO-Guard 只作为失败后的独立候选；
- 最终 checkpoint 的发布与云端归档路径。

上述未决项全部闭合前：

```text
design_frozen = false
training_authorized = false
```

















