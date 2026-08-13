# GRPO 待讨论问题

## 用途

本文是 V5-Pg GRPO 的讨论工作台，不提前冻结完整方案。每个问题按“提出问题 -> 补证据 -> 讨论 -> 用户裁决 -> 同步合同/计划”的顺序推进。

当前只记录问题和讨论入口；已经裁决的内容以[算法合同](GRPO原论文核验与算法合同.md)和[执行计划](GRPO感知强化初步计划.md)为准。

## 讨论状态

建议使用以下状态：

```text
[未开始] [讨论中] [待实验] [待用户裁决] [已裁决] [阻塞]
```

每个问题结束时补充：

```text
结论：
依据：
影响的文件：
是否允许进入实现：
```

## 本轮白话解释与澄清（2026-08-12）

### DNSMOS 的“十档”是什么

DNSMOS SIG 本身不是十分类器。DNSMOS P.835 的 SIG 是连续的前景人声信号质量估计，通常可理解为
MOS 风格的连续分数；本项目已有的“十档”是把候选音频按分数划成十个等宽听评包，用来检查它是否对
目标毛刺/破音形成单调排序。它不是训练时的离散 reward。GRPO 训练仍使用每个候选的连续 SIG raw
score。此前拟在同一 `G=8` 组内独立做合同 Z；该选择已于 2026-08-12 被用户撤销。现已裁决为
“5,000 条训练原音 Q100 分位映射 -> 0--10 clip -> 按目标区间压缩 -> 组内只居中”。

#### 正式训练池原音的实测范围

十档实验已对正式训练池确定性抽取的 2,000 条原音做过统一评分。评分输入为 44.1 kHz mono PCM24、
线性等响至 `-28 dBFS RMS` 的派生副本；未降噪、未限幅、未改频谱。DNSMOS SIG 分布为：

| 统计量 | 分数 |
|---|---:|
| observed min / max | 1.8789 / 3.7603 |
| q0.5 / q99.5 | 2.3230 / 3.6960 |
| q5 / q95 | 3.0699 / 3.6405 |
| q25 / median / q75 | 3.3402 / 3.4688 / 3.5513 |
| mean / population std | 3.4219 / 0.2047 |

因此可把 `2.32--3.70` 视为这批训练原音的稳健全域，把 `3.07--3.64` 视为中央 90% 区间，
把 `3.34--3.55` 视为中央 50% 区间。十档名义范围 `2.25--3.75`、档宽 `0.15` 与该分布一致。

训练原音的档位占比高度集中在高端：`80.75%` 不低于 `3.30`，`54.20%` 不低于 `3.45`，仅
`3.40%` 低于 `3.00`。作为生成侧对照，V4PH 自克隆 500 条的 median/mean 为 `3.2384/3.1929`，
V4fg 自克隆 500 条为 `3.3286/3.2899`；两者都低于训练原音的 `3.4688/3.4219`。

这些数字可以用于绝对监控、识别异常低分和观察生成结果是否向训练原音主体区域靠近，但不能直接定义
GRPO reliability gate。训练池的跨样本 `std=0.2047` 主要混合了歌曲、唱法、音高、录音条件和片段
难度；GRPO 需要的是同一 prompt 内仅改变 SDE noise 时的 SIG 极差/标准差。正式门槛仍需在同 prompt
多 seed 矩阵上标定，不能用全训练池标准差代替。

#### V4PH/V4fg 分布宽度与 0--10 映射候选

跨不同 prompt 的 500 条自克隆分布并未塌缩，但 V4PH 明显比 V4fg 更宽、低分尾更长：

| 集合 | min--max 宽度 | q5--q95 宽度 | IQR | population std | median |
|---|---:|---:|---:|---:|---:|
| V4PH 自克隆 | 1.3517 | 0.8270 | 0.3270 | 0.2494 | 3.2384 |
| V4fg 自克隆 | 1.1050 | 0.6581 | 0.2895 | 0.2098 | 3.3286 |

按 q5--q95，V4PH 比 V4fg 宽约 26%；按 population std 宽约 19%。这些是跨 prompt 分布，不能
替代同 prompt、不同 SDE noise 的组内宽度。

基于训练原音十档边界，固定绝对刻度候选为：

```text
SIG10_abs = clip((DNSMOS_SIG - 2.25) / 0.15, 0, 10)
```

即 `2.25 -> 0`、`2.40 -> 1`、...、`3.60 -> 9`、`3.75 -> 10`。该映射只截断 2,000 条
训练原音中的 10 条（低端 8 条、高端 2 条），不截断 500 条 V4PH 或 500 条 V4fg 自克隆。映射后
V4PH median/std 为 `6.589/1.663`，V4fg 为 `7.191/1.399`，训练原音 median/std 为
`8.125/1.364`。

不得按 V4PH、V4fg 或每个 `G=8` 组自己的 min/max 动态铺满 0--10，否则无论绝对质量和真实极差
如何，每组都会被强制制造一个 0 分和一个 10 分。固定 `2.25--3.75` 映射可用于日志、绝对异常门禁
和人类可读报告。

若合同 Z 保持不变，则对未截断线性映射有：

```text
z_group(SIG10_abs) == z_group(DNSMOS_SIG)
```

因此 0--10 只改善可解释性，不改变 advantage。若要直接用 `SIG10_abs` 或只组内居中、不除组内
标准差，则属于修改合同 Z，必须连同 UTMOS22 的固定尺度和 QUA 合成方式重新讨论，不能静默替换。

#### UTMOS22 分布与双 0--10 复合候选

UTMOS22 的训练原音十档范围为 `1.05--2.05`、档宽 `0.10`，但分布明显右偏：

| 集合 | q5 | median | mean | q95 | population std |
|---|---:|---:|---:|---:|---:|
| 训练原音 2,000 条 | 1.2561 | 1.3351 | 1.3717 | 1.6164 | 0.1218 |
| V4PH 自克隆 500 条 | 1.2490 | 1.3132 | 1.3509 | 1.5757 | 0.1122 |
| V4fg 自克隆 500 条 | 1.2460 | 1.3180 | 1.3531 | 1.5708 | 0.1087 |

因此 PH/fg 的 UTMOS22 跨 prompt 分布宽度接近，中位数也只差 `0.0048`；它不像 DNSMOS SIG
那样在这两个生成池上给出明显的绝对位置差。UTMOS22 的同 prompt 多 seed 实验还曾出现跨模型方向
与人工噪音排序不一致，不能把它的绝对值解释为纯噪音度。

若两轴都使用十档固定刻度，候选公式为：

```text
U10_abs   = clip((UTMOS22   - 1.05) / 0.10, 0, 10)
SIG10_abs = clip((DNSMOS_SIG - 2.25) / 0.15, 0, 10)
QUA10_abs = 0.7 * U10_abs + 0.3 * SIG10_abs
```

按已有 3,054 条同构评分数据，训练原音的 `U10/SIG10/QUA10` 中位数为
`2.85/8.13/4.42`；V4PH 为 `2.63/6.59/3.85`；V4fg 为 `2.68/7.19/4.05`。这说明两个固定
刻度虽然都覆盖 0--10，但典型绝对位置并不相同。两轴 Pearson 相关仅约 `0.27--0.44`，具有一定
互补性，不是简单重复评分。

若直接采用 `QUA10_abs`，它属于固定绝对尺度复合；若对 `U10/SIG10` 再分别做组内 z-score，线性
映射会被抵消，仍等价于已撤销的合同 Z。这里曾考虑“组内居中、不除当组 std，再除以离线冻结的
effective scale”；该候选随后又被用户的训练原音固定范围方案取代。

#### 用户目标：有效波动贡献为 7:3

用户进一步明确，`7:3` 应表示 UTMOS22 与 DNSMOS SIG 在加权后的有效波动幅度贡献，而不只是公式
表面的两个系数。若“波动幅度”定义为同 prompt 内的 population std，则在两个轴都通过 reliability
gate 时，合同 Z 已逐组满足：

```text
std_group(Z_U)   = 1
std_group(Z_SIG) = 1

std_group(0.7 * Z_U) : std_group(0.3 * Z_SIG) = 7 : 3
```

固定绝对 `U10/SIG10` 映射本身不能保证该比例；它只能用于日志和绝对位置审计。若不希望每组除自己的
std，则必须在最终 V5-Pg 同 prompt 多 seed 标定集上冻结 `scale_U/scale_SIG`，先用 pooled within-prompt
有效波动把两轴归一到同尺度，再按 `7:3` 合成。该替代只能保证校准总体上的 7:3，不能保证每一组
严格 7:3。

这里暂按“标准差/幅度比 7:3”理解。若用户指的是方差占比 70%/30%，单位尺度后的系数应与
`sqrt(0.7):sqrt(0.3)` 成比例，而不是 `0.7:0.3`；两者必须显式区分。

#### 2026-08-12 裁决：撤销合同 Z

用户裁决不再使用逐组 z-score。原因是逐组除以当组 std 会把真实差异很小的组强制拉成单位波动，
即使 reliability gate 能挡住一部分噪声，也会丢掉“该组本来只差一点”的幅度信息。

当时讨论的替代候选为：

```text
centered_ik = R_ik - mean_group(R_k)
C_ik        = reliability_gk * centered_ik / scale_k_frozen

QUA_i = 0.7 * C_U,i + 0.3 * C_SIG,i
A_i   = (3 * C_INS,i + 2 * C_PITCH,i + 4 * QUA_i) / 9
```

`scale_k_frozen` 必须来自最终 V5-Pg 基座的同 prompt 多 seed 有效波动，而不是训练池跨歌曲分布，也
不能在训练中在线变化。该方法只在冻结校准总体上让有效波动贡献符合 `3:2:4` 和 `7:3`；每个具体组
保留自己的真实幅度，不再被强制铺成相同比例。正式 scale estimator、噪声扣除、稳健统计和门槛仍待
讨论，所以 normalizer 当前是“Z 已否决、替代合同待细化”，不是 design frozen。

#### 2026-08-12 历史口径：按训练原音范围压到 7 格与 3 格

用户进一步明确，不再要求两个加权分量的标准差之比严格为 `7:3`；目标是根据两种评分器在原始训练
数据上的固定有效范围做线性映射，使 UTMOS22 的可用贡献区间宽 7，DNSMOS SIG 的可用贡献区间宽 3。
这会取代上面的 pooled effective-std 配平候选。

这一阶段曾建议不用 observed min/max，避免单个离群值决定全局斜率；沿用十档实验已有证据的训练原音
q0.5--q99.5：

```text
UTMOS22:    L_U=1.2384856421, H_U=1.8839728767
DNSMOS SIG: L_S=2.3229875800, H_S=3.6960248274

N_U   = clip((UTMOS22   - L_U) / (H_U - L_U), 0, 1)
N_SIG = clip((DNSMOS_SIG - L_S) / (H_S - L_S), 0, 1)

QUA10_abs = 7 * N_U + 3 * N_SIG
QUA01_abs = 0.7 * N_U + 0.3 * N_SIG

A_QUA_i = 0.7 * reliability_U   * (N_U,i   - mean_group(N_U))
        + 0.3 * reliability_SIG * (N_SIG,i - mean_group(N_SIG))
```

`QUA10_abs` 只用于 0--10 报告；训练使用等价的 `QUA01` 组内居中形式，不除以当组 std。该定义
严格保证两个分量的**允许区间宽度**为 `7:3`，但不保证任一数据集上的实际 std、方差或逐组波动
也是 `7:3`。这正是范围法与 Z/frozen-effective-std 法的区别。

按已有 2,000 条训练原音，q0.5--q99.5 会让每轴各有 20 条（1%）落在端点外并被 clip；V4PH
自克隆中 UTMOS/SIG 分别有 `2/1` 条 clip，V4fg 为 `6/0`。训练原音 `QUA10` 的
q5/median/q95 为 `2.13/3.53/6.81`，V4PH 为 `1.13/2.81/6.17`，V4fg 为
`1.51/3.04/6.01`。端点与 clip 策略仍待用户最终裁决。

#### 2026-08-12 最终裁决：5k Q100 单调分段线性

用户进一步明确，第一步不是用单一 `L/H` 做 min-max，而是按训练原音样本占比做经验 CDF 近似。
训练参考固定为从最终权威训练 manifest 均匀无放回抽取的 5,000 条训练音频；每个 scorer 独立建立
100 个等百分位锚点，形成 99 段单调直线：

```text
j = 0 ... 99
p_j = j / 99
x_kj = quantile(training_reference_scores_k, p_j, method="linear")
y_j = 10 * p_j
```

对新 raw score `r`，若 `r <= x_k0` 返回 0，若 `r >= x_k99` 返回 10；否则二分找到
`x_kj <= r <= x_k,j+1` 并线性插值：

```text
M_k(r) = y_j + (y_j+1 - y_j) * (r - x_kj) / (x_k,j+1 - x_kj)
```

因此 `M_k(r)=9` 近似表示该 raw score 高于训练参考分布的 90% 样本。随后只做一次线性压缩：

```text
U10    = M_UTMOS(UTMOS22)
SIG10  = M_SIG(DNSMOS_SIG)
U7     = 0.7 * U10
SIG3   = 0.3 * SIG10
QUA10  = U7 + SIG3

A_QUA_i = reliability_U   * (U7_i   - mean_group(U7)) / 10
        + reliability_SIG * (SIG3_i - mean_group(SIG3)) / 10
```

第一步是固定的非线性单调映射，第二步才是纯一次函数。连续训练参考分布经百分位映射后近似均匀，
所以 U10/SIG10 在校准集上不仅范围相同，标准差也近似相同；再乘 `0.7/0.3` 后，表示范围与校准集
波动都近似 7:3。具体生成组仍保留自己的真实差异，不逐组重标定。

Q100 表必须记录 5,000 条训练参考 manifest、固定抽样 seed `20260812`、评分波形预处理、scorer
权重/代码 SHA256: redacted`linear` quantile、全部 100 个 `(p,x,y)` 锚点和 clip 规则。若相邻 raw 锚点
相等，构建必须硬失败并报告重复区间，不得静默加 epsilon 或合并 plateau。训练中禁止重新估计锚点。

第一题至此收口：

```text
状态：[已裁决，产物待构建]
结论：5,000 条训练原音 -> 每轴独立 Q100 -> 0--10 -> UTMOS 0--7 / SIG 0--3 -> 组内只居中
依据：用户裁决；训练原音 2,000 条先导分布；UTMOS22/DNSMOS SIG 互补性与量纲差异审计
影响的文件：QUA_Q100评分映射合同.md、GRPO原论文核验与算法合同.md、GRPO感知强化初步计划.md
是否允许进入实现：只允许在最终训练 manifest 确定后构建并审计 Q100；不授权 GRPO 训练
```

#### 合同 Z 的白话原理

对同一 prompt 的 `G=8` 个候选，某一评分轴先计算组内均值与 population std：

```text
mu_k    = mean(R_1k ... R_8k)
sigma_k = std_pop(R_1k ... R_8k)
Z_ik    = (R_ik - mu_k) / sigma_k
```

`Z=0` 表示候选正好位于同组平均，`Z>0` 表示高于同组平均，`Z<0` 表示低于同组平均；绝对值表示
相差多少个该组标准差。每个通过门禁的评分轴经 Z 后组内均值为 0、population std 为 1，因此
UTMOS22 和 DNSMOS SIG 的原始单位、绝对范围及典型波动不同，不再污染 `7:3` 的线性幅度比例。

例如仅为演示使用四个候选：

```text
UTMOS = [1.30, 1.31, 1.32, 1.33]
SIG   = [3.00, 3.10, 3.20, 3.30]

两轴各自 Z 后都约为：
[-1.34, -0.45, +0.45, +1.34]
```

两轴 raw range 相差十倍，但都被转换为“在自己轴内相对同组的位置”。再计算
`0.7*Z_U + 0.3*Z_SIG`，加权后的波动幅度就是 7:3。

风险也来自同一个机制：若 UTMOS 实际只是 `[1.3000, 1.3001, 1.3002, 1.3003]` 的数值噪声，Z
仍会把它放成同样的 `[-1.34,...,+1.34]`。因此顺序必须是：先用离线冻结的重复误差/有效极差门槛
判断该轴是否有真实信号，通过后才做 Z；不能用 `std+epsilon` 把无效差异继续放大。Z 解决量纲和
prompt 难度，不负责判断差异是否可信。

#### 相关评分器回忆：NISQA Noisiness

十档实验中另有一个更直接的噪声维度：`NISQA Noisiness`。它是 NISQA v2.0 的 Noisiness 输出，
高分表示主观噪声损伤更小；训练原音的观测范围为 `1.6408--4.7580`，q0.5--q99.5 为
`2.1198--4.3202`，十档候选范围为 `1.875--4.375`、档宽 `0.25`。它与 DNSMOS SIG 一样面向
通信/噪声语音，不能因名字直接假定对歌声发声区砂噪有效。

当前十档听评的正式结论只明确选中 UTMOS22（主候选）和 DNSMOS SIG（辅助交叉验证）；没有把
NISQA Noisiness 登记为已验证的正式 reward。因此它可以作为后续第三评分器对照，但不能在未补做
同一批十档试听、重复误差和同 prompt 多 seed 标定前，直接加入 QUA 或改变 `7:3` 合同。

#### 关于“2:8 复合”的历史记忆核对

当前项目文件中没有找到“两个噪音评分器按 `2:8` 合成”的已冻结方案。最接近的历史记录是旧 V4c
v3 的四路权重：

```text
PER : SIM : F0 : DNSMOS = 6 : 1 : 1 : 2
```

其中 DNSMOS 占总权重 `2/10=20%`，其余三项合计 `8/10=80%`，可能被记成了“2:8”。这不是
两个噪音评分器之间的比例。V4c v2 还曾短暂使用 `2:1:1:0.5`，同样不是双噪音复合。

当前 V5-Pg 口径仍是：

```text
QUA 内部 UTMOS22 : DNSMOS SIG = 7 : 3
```

`NISQA Noisiness` 虽然做过十档测试，但没有找到它与 DNSMOS 或 UTMOS 按 `2:8` 合成的历史证据。

### SDE 注噪的作用

Flow 模型原本按 ODE 确定性地从噪声积分到音频 latent；同一初始状态和条件基本只给出一条轨迹，
没有可计算的正方差 transition likelihood。GRPO 需要从行为策略采样、保存 old log-prob，再用
current/old 的概率比更新。因此只在选定的 denoising window 对 latent transition 注入已知 Gaussian
噪声，把它变成可 replay 的随机策略。这个噪声是探索手段，不是要把最终音频变“更吵”；推理阶段仍
可使用 ODE。注噪过强会伤害音质，所以 `a`、窗口和步数必须先标定。

### GRPO 与 PPO 的关系

这里仍然是 GRPO，不是改成 vanilla PPO。GRPO 负责“同一 prompt 的 `G=8` 候选如何用组内均值/标准差
形成 advantage”；更新时借用 PPO 的 clipped probability-ratio surrogate 和 reference KL 保护。
因此第 7 项的“PPO 更新与 KL”指 GRPO 的 PPO-style 更新细节，不引入 PPO 的 value/critic 网络。

### 术语速查

| 术语 | 白话含义 |
|---|---|
| prompt / group | 一组完全相同的输入条件；同组生成 `G=8` 个候选，只在组内比较 |
| behavior (`old`) | 真正生成本轮候选的冻结快照，并保存该 transition 的 old log-prob |
| current | 正在接收梯度、重算新 log-prob 的模型 |
| reference | 整个 run 不更新的 V5-Pg 基座，只用于 KL 约束漂移 |
| action | 一个 SDE timestep 的完整有效 B transition；逐帧 64-D 只作为精确 log-prob 分解与等权归约单位 |
| ratio | `exp(logp_current - logp_old)`；只对真实 SDE、有效 B 坐标计算 |
| A/B/PAD/reference tail | V5-Pg 的时间区域 mask；不是所有 latent 帧都允许成为 RL action |
| NFE | denoising/ODE 求解器的函数评估次数，即采样步数预算 |
| CFG | 延后裁决；项目 API `g` 与常见 scale 的关系为 `s=1+g`，因此 API `g=0` 才是普通 conditional |

### 本轮初步裁决边界

- rollout CFG 等待 V5-Pg 同输入、同 seed、多 CFG 轨迹后裁决；入选档位再做同 CFG 的 ODE/SDE
  标定。不得再使用旧 V4c 的 ODE 轨迹配伪造 SDE 概率。
- “同意”与“值得思考”只表示方向接受或暂不裁决，不自动冻结 Short/Long 比例、PITCH scalar、
  reliability threshold、SDE window 或 PPO-style 超参数。

## 问题目录

### 1. 基座与范围

状态：`[未开始]`

- V5-Pg 两条 g 路线最终选哪个 checkpoint？
- 使用 raw 还是 EMA？
- 首轮只做短音频，还是同时覆盖 Long？

### 2. Reward normalizer 与 reliability

状态：`Q100 映射 [已裁决，产物待构建]；reliability [待实验]`

- Q100 固定使用 5,000 条训练原音、`linear` quantile、100 锚点/99 段，重复锚点硬失败；实际产物
  需等最终训练 manifest 确定后构建；
- INS、PITCH、UTMOS22、DNSMOS SIG 各自的有效极差和重复误差阈值是多少？
- 某个轴关闭时是否保持外层权重总和不变？

### 3. PITCH_PROB

状态：`[已裁决，方案 A；门槛与逐帧同构待实验]`

- 正式 scalar 使用按 target voiced/REST 分层均衡的 GAME probability cosine；linear CKA 与论文口径
  F0-CORR 只作离线诊断、结构回归和 reward-hacking 门禁，不进入首轮主 reward；
- PITCH_PROB 的角色是防止 SIM/QUA 微调时旋律、绝对音高、REST/voiced 或边界跑偏，不是首轮微调的
  主要优化目的；外层名义权重保留 `SIM_INS:PITCH_PROB:QUA = 3:2:4`，其中 PITCH 的 `2` 是护栏预算；
- 由于 GRPO advantage 只做组内中心化，PITCH 只能抑制同组候选之间的相对跑偏；若八个候选共同漂移，
  仍需 held-out 的绝对音高回归门禁或停止条件，不能指望提高 PITCH 权重解决；
- 生成 REST、目标 voiced 的帧得分为 0，生成 voiced、目标 REST 的帧得分为 0，双 REST 得分为 1；
  voiced 帧使用两个 128 维 GAME posterior 的 cosine，并对 voiced/REST 两类做平衡平均；空类处理、
  reliability 门槛和 scorer 重复误差仍待标定；
- target 与 generated 采用方案 A：两端都先对完整 track 运行 GAME，再依据同一 latent-frame 时间映射
  精确切 B；generated 使用与 V5-Pg 推理相同的 A+B(+reference tail) 上下文。两端必须复用 V5-P 的
  medium K=4、pitch-axis adapter、REST 语义和同一目标时间网格；实现前做逐帧同构验证。

### 4. SDE transition

状态：`[transition 已裁决；论文参数作为 baseline，实验保留]`

- 使用与项目 `t=0 -> 1` 时间方向一致的等边际 ODE-to-SDE；不得复用 V4c 的朴素加噪或给 ODE
  轨迹事后虚构 Gaussian 方差；
- SDE 只覆盖有效 B 区。noise、drift correction、transition mean、log-prob、ratio 和 KL 必须使用
  完全相同的 B mask；A、PAD、reference tail 和 window 外坐标继续走 ODE，不作为 RL action；
- `t=0` 奇异，首个 transition 不得进入 SDE window；计算必须使用 time-shift 后的实际 `t_j` 和
  `delta_t`；
- rollout 使用 SDE 服务探索和 likelihood，正式推理仍可使用 ODE；rollout CFG 等 V5-Pg 多 CFG
  轨迹结果后裁决；
- 先按 YingMusic-Singer-Plus 论文 `a=0.8,w_min=1,w_s=8` 跑 baseline；同时保留 single-step、
  `a={0.2,0.4,0.6,0.8}`、`w=2/4/8` 和 NFE 实验；
  以累计 diffusion budget、等边际/结构硬门禁、扣除 scorer 重复误差后的组内探索效率和等预算短
  pilot 裁决。最终在 held-out 32-step ODE 收益距最佳值不超过一个标准误的配置中，选择更低 `a`、
  更小窗口和更低训练成本的配置；
- 论文值是默认 baseline，不是不可修改的最终训练参数；最终值仍由上述实验与门禁裁决。

### 5. 组内探索

状态：`[组合同已裁决，稳定性与显存待验证]`

- `G=8` 暂作为首轮配置；论文与现有合同沿用它，但实际完整 rollout 是否 OOM 必须在 smoke 时验证，
  不能用静态显存估算替代；必要时只改变候选的串行/微批执行，不改变组内 `G=8` 语义；
- `G=8` 是否足够稳定？有效组比例最低接受多少？
- 正式 rollout 同组候选共享同一个初始 latent；候选之间只使用 SDE window 的独立噪声；
  每候选独立初始 latent 只作离线探索幅度与 credit attribution 对照；
- `a`、single-step、`w=2/4/8` 的标定程序已在 SDE transition 中裁决；本节只需用标定结果判断
  `G=8` 和初始 latent 合同。

### 6. Policy、action 与 ratio

状态：`[角色、action、逐帧分解、记录等权与 replay 已裁决；reference 待 V5-Pg 评测]`

- 每轮 rollout 开始前从 current 复制 behavior；本轮 8 个候选及其 old log-prob 期间 behavior 固定，
  本轮更新结束后丢弃，下一轮再从新的 current 复制；
- reference 使用整个 run 不更新的 V5-Pg 基座；具体 checkpoint、raw/EMA 来源和 SHA256: redacted
  V5-Pg 评测后裁决；
- 一个 SDE timestep 的整个有效 B latent transition 是环境 action；因 covariance 为坐标独立 Gaussian，
  log-prob 精确分解为逐帧 64-D contribution。逐帧存储/计算不等于把每帧改成独立环境 action；
- 用户裁决每条训练记录等权：单帧内对 64 channel 的 log-prob 求和，再对有效 B frame 和 SDE step
  求 mean。首轮不使用 frame-block，也不计算高维 joint ratio 参与 PPO clip；
- rollout 必须保存实际 `x_t/x_next/mean_old/variance/t_j/delta_t/B_mask/window/epsilon` 与逐帧
  `logp_old`；用同一收据逐步 replay `x_next`，并重算 `logp_old` 做一致性门禁；不能更新后重新采样
  或用 current 伪造 old log-prob；
- Gaussian ratio 只允许使用真实 SDE、有效 B frame；ODE、A、PAD、reference tail 和 window 外坐标
  一律排除，noise、mean、log-prob、ratio、KL 使用同一 mask；
- 首轮同一 rollout 收据只使用一次；未来若需要重复更新，必须重新讨论高维 joint ratio 与逐帧 PPO
  surrogate，不自动沿用当前合同。

### 7. PPO 更新与 KL

状态：`[已裁决，rollout 组数与累积频率待 smoke]`

- 首轮 `inner_epochs=1` 且严格 on-policy；完整 rollout batch 的全部记录/microbatch 必须先完成
  forward/backward 梯度累积，之后只调用一次 `optimizer.step()`。不得在第一个 microbatch 后更新
  current 再处理后续 old rollout；
- behavior 在整个 rollout batch 的采样与更新期间固定，批次结束后丢弃；rollout batch 组数、完整组的
  梯度累积数量和 optimizer step 频率等 smoke/OOM 测试后冻结；
- PPO clip 直接采用 YingMusic-Singer-Plus 论文表格的非对称范围：`epsilon_lower=0.002`、
  `epsilon_upper=0.01`；不再为此做候选实验，官方 YAML 的 `epsilon_upper=0.02` 仅作为历史不一致记录；
- KL 名义 `beta=1`；使用真实 transition covariance `sigma(t)^2*delta_t`，不假设与官方代码 reduction
  尺度等价；必须记录逐 SDE step/有效 B frame 的实际 KL，并以 held-out 32-step ODE 回归和 KL/ratio
  硬门禁监控，保留 beta sensitivity 实验；
- advantage 首轮不做额外 clip；只做 finite 和理论范围硬审计，超出范围硬失败，不静默 clamp；
- 正负 clipfrac 不要求相等；必须分开记录 positive upper / negative lower clipfrac、log-ratio、
  ratio quantiles/ESS 和分层 KL；只有 replay/mask/variance 等实现门禁通过且出现持续、无法由非对称
  clip 范围解释的单侧 ratio 重尾、非有限或 held-out 回归时，才停止普通方案并另立 RatioNorm 变体；

### 8. Reward hacking 与轴间冲突

状态：`[已裁决，训练/评测/early-stop 分层]`

- 组内候选共享 prompt、target、音高条件和 reference，固定的音素/内容因素大多在组内比较中抵消；
  `SIM_INS` 首轮按项目主要目标定义为唱法相似度轴，不再额外要求启动前完成 INS 专项反事实审计；
- 仍适用通用的人耳、held-out 回归和 reward-hacking 停止条件，但不把 INS 的“纯唱法”证明作为 pilot
  前置门槛；
- QUA 继续作为主要质量轴（外层权重 4）；在 KL 与 reference 约束下，适度削波/频谱变软、辅音能量、
  发声覆盖或响度/动态变化不预先视为 hacking，也不额外加入对应禁止项；只保留非有限、全静音、
  长度异常和真正数字削波等音频有效性硬门禁；
- `PITCH` 上升但听感变差不做同一标量空间内的“谁更大”比较：PITCH 继续作为训练护栏和自动诊断轴，
  不被人耳分数替代；人耳不进入 reward 或中间件，只在固定 checkpoint/pilot 作为外部 early-stop/veto，
  判断这条训练路线是否继续；
- 训练中间件、自动评测和外部决策严格分层：`SIM_INS/PITCH_PROB/QUA` 负责优化，GAME/QUA/PER/H/SOFA
  等负责自动评测与归因，人耳只负责路线停止/继续，不把三者合成一个总分。

### 9. 数据与 prompt 采样

状态：`[已裁决]`

- Short/Long 沿用 V5-P 的 `NATURAL_RECORD`：ShortPool/LongPool 合并后按 manifest 记录等概率采样；
  不按音频秒数重平衡，不手工设置 `long_probability`；Long 的 OOM/耗时由 smoke 和执行层处理；
- 训练 prompt/seed 继续随机推进，不固定；checkpoint 评测使用固定、与 TrainPool 无 overlap 的
  Short/Long prompt 集和固定 seed 集，保存 manifest/条件/初始 latent/SDE noise/scorer provenance；
  不另设独立 seed 稳健性复核集；
- 失败统一记录为 `GROUP_INVALID`，不区分“硬失败”和“单轴可靠性失败”的后续处理：任一候选音频/实现
  失败，或任一 INS/PITCH/UTMOS22/DNSMOS SIG 轴失败，整组 `G=8` 跳过且不 `optimizer.step()`；
- 每个失败组必须记录 `rollout_id/group_id/prompt SampleId/Pool/Short-Long`、manifest/policy/reference/
  scorer provenance SHA256: redacted
  sr/duration/RMS/peak/finite/nonzero/clipping 统计，以及每个 scorer 的 status、error class、异常消息
  摘要、输入波形 hash、输出 shape/dtype/finite/range、raw score、reliability threshold、重复误差和
  失败轴；
- 失败组不得填中性分、静默重采样、删除失败候选、只保留剩余轴或重新归一化权重；调试重跑必须生成
  新 `rollout_id` 并与原失败组分开登记，不得伪装成原组成功。

### 10. 评价与停止

状态：`[已裁决]`

- 每个 rollout 记录 reward/advantage、old/new log-prob、ratio/log-ratio/ESS、正负 clipfrac、KL、loss/
  gradient/LR、`GROUP_VALID/GROUP_INVALID`、失败候选/轴/原因、音频有效性、Short/Long、SDE window、
  walltime 与显存；`rollout_attempt` 与 `optimizer_step` 分开计数；
- 固定 checkpoint 使用固定 prompt/seed 的正式 32-step ODE 输出，离线计算 reward 轴、GAME 扩展音高
  诊断、F0-CORR/CKA、PER、H/SOFA、Short/Long/Long 后段和音频统计；不在每个 rollout 跑额外 PER；
- 人耳不属于离线指标，只在预设 checkpoint/pilot 节点给出 continue/early-stop/veto；具体保存、评测和
  试听频率留到 pilot 参数裁决；
- 歌词错误率正式记为 `KANA_CER = edit_distance(ref_kana, asr_kana) / len(ref_kana)`，越低越好且不在
  `1.0` 截断；V4c 历史所谓 `PER` 实际为 `max(0, 1-KANA_CER)`，只保留为历史兼容字段
  `V4C_PER_SCORE`，不得再与错误率方向混用；
- 歌词回归按固定 Eval 中 checkpoint 相对 step-0/reference 的逐样本变化计数：若超过 `20` 条样本
  各自出现超过 `20%` 的识别分下降，立即停止；若超过 `10` 条样本各自出现超过 `10%` 的识别分
  下降，发出警告。两项同时满足时停止优先；不得用全体平均值掩盖这些退化样本；
- H/SOFA 按摩拉错槽率相对 step-0 的逐样本增量（百分点）判断，容许门槛为歌词门禁的 `1.5x`：
  超过 `15` 条样本各自增加超过 `15` 个百分点时警告；超过 `30` 条样本各自增加超过 `30` 个百分点
  时立即停止。step-0 可对齐而 checkpoint 出现 SOFA 失败或 mora 数量异常，按该样本完全回归计；
- GAME 音高门禁按 `GAME_ERR = 1-PITCH_PROB` 定义逐样本误差，并比较 checkpoint 相对 step-0 的
  `GAME_ERR` 增长率：超过 `10` 条样本各自增长超过 `10%` 时警告；超过 `20` 条样本各自增长超过
  `20%` 时立即停止。增长率分母使用 `max(GAME_ERR_step0, scorer_noise_floor)`；后者由 pilot 前的
  scorer 重复误差标定，不主观拍值。voiced/REST、绝对音高和 F0-CORR 作为归因诊断；
- 人工听评已是外部 early-stop/veto，不再定义与自动指标同维度的数值回归门槛；
- 不另设自动 `proxy hacking` 判定或新的停止门槛。音频有效性、歌词/错槽/音高回归、KL/ratio/梯度
  健康和人耳 early-stop/veto 已分别负责停止；`proxy_hacking` 仅在 reward 上升同时触发人耳否决或
  既有回归门禁后，作为事后归因标签记录具体 scorer 偏好，不独立触发停止。

## 十题后完整对照审计（2026-08-13）

对照 `docs/基础设施${CLOUD_ARTIFACT}`、深度思考报告、风险矩阵和当前三份 V5
GRPO 合同后，原十个设计问题仍视为已经逐项讨论完毕。以下不是“第 11--14 题”，而是十题合同落到
唯一可执行公式时发现的四个缺口；建议仍一次只讨论一项。

### A. CFG 的精确语义与最终状态

状态：`[已裁决等待评测]`

- 用户早期说“RL 使用 CFG1”，但项目 API 实际公式为 `cond + g*(cond-uncond)`：API `g=1` 等价于
  常见写法的 `s=2`，需要 cond/uncond 双 forward；普通 conditional/常见 `s=1` 对应 API `g=0`；
- TEMP 最终共识和当前算法合同记录为“等待 V5-Pg 多 CFG 轨迹后裁决”，不能同时把无口径的
  `CFG1` 写成已冻结值；
- 用户裁决维持延后：等待 V5-Pg 多 CFG 同输入、同 seed 评测，再选择 ODE 候选并完成同 CFG 的
  ODE/SDE 标定。评测与收据必须同时记录 `cfg_api_g` 与 `cfg_standard_s=1+g`；在最终值冻结前禁止
  只写有歧义的 `CFG1`，behavior/current/reference 必须使用完全相同的 guided drift。

### B. `C_INS` 与 `C_PITCH` 的尺度函数

状态：`[外层权重已裁决，但两个输入坐标未定义]`

- `3:2:4` 只有在 `C_INS/C_PITCH/QUA` 处于已冻结且可比较的坐标中才有唯一实现；当前 QUA 已有
  训练原音 Q100，pairwise 的 INS/PITCH 却不能用“原音与自身”构建非退化 Q100；
- 需冻结 INS/PITCH 使用 candidate calibration pool 的经验分位映射，还是使用同一 pool 的稳健线性
  scale。该选择不重新讨论 `3:2:4`，只补齐它在代码中的数值含义；
- calibration pool、映射/scale、clip、reliability 与最终 Eval 必须分别登记且冻结，不能在训练中更新。

### C. QUA 的实际评分单元

状态：`[已裁决为 B-only]`

- 当前 Q100 写的是 5,000 条完整训练音频，runtime 合同只写 `equal-loudness scoring copy`，没有唯一
  指定 UTMOS22/SIG 是评分 generated B，还是完整 A+B(+reference tail)；
- 若 runtime 评 B-only，而 Q100 用完整记录，时长、静音占比和 scorer padding/repeat 行为可能造成
  坐标域偏移；若评完整 track，组内固定 A/tail 会稀释 B action 的贡献，非线性 scorer 也不能保证
  固定部分严格抵消；
- 用户裁决唯一评分单元为 B-only：runtime 只评分 generated B；5k Q100 先按记录抽样，再 materialize
  该记录的 target B 作为实际评分波形。两者使用同构裁切、最短长度处理、等响和 scorer 前处理；A、
  PAD 与 reference tail 不进入 UTMOS22/DNSMOS SIG。INS 同样继续使用 B-only；GAME 的 full-track
  上下文合同只服务 PITCH，不改变 QUA 的评分单元。

### D. 固定 Eval 的样本单位与规模

状态：`[非核心，降为执行前 manifest 参数]`

- 已裁决的歌词、H/SOFA、GAME 门禁使用“超过 10/15/20/30 条样本”的绝对计数，因此 Eval 总数和
  “一条样本”是 unique prompt 还是 prompt-seed 输出，会直接改变门槛强度；
- 用户判断该项不重要，不继续作主观设计讨论。实现阶段按可用评测预算冻结 Short/Long 数量、每
  prompt 固定 seed 数及合并/分桶报告口径即可；固定 Eval 仍与 TrainPool 无 overlap，且不新增独立
  seed 稳健性复核集；
- manifest 一经冻结不得随 checkpoint 改变，否则绝对样本数门槛不可比较。

### 不需要继续主观讨论的内容

以下应由外部结果、标定或工程 smoke 决定，不新增设计题：

- 最终 V5-Pg checkpoint、raw/EMA、SHA256: redacted
- Q100 5k 实际锚点、四轴 repeatability/reliability 数值、GAME exact B slice/D3PM/空类单测；
- `a/window/NFE`、G=8 显存与吞吐、rollout batch 组数；
- GRPO LR、pilot 步数、保存/评测频率、reward worker GPU 布局；
- full parameter/LoRA/last-block 由小型可行性和显存/held-out ODE 结果选择，属于执行选型；
- calibration/pilot-train/fixed-Eval 的 SampleId 隔离是防选择污染的固定工程规则，不等于增加独立
  seed 稳健性复核集。

### 待统一的文档残留

这些是文字同步，不需要用户重新裁决：

- 任一 scorer/单轴 reliability 失败应整组 `GROUP_INVALID`；TEMP 文档中“轴关闭后其余轴不放大”的
  旧表述不得解释成允许剩余轴继续训练；
- QUA 公式只加权一次：`0.7*C_U + 0.3*C_SIG`，其中 `C_U/C_SIG` 来自各自 `U10/SIG10` 居中并除
  `10`；`U7/SIG3` 只是等价报告写法，不能再次乘 `0.7/0.3`；
- beta 首轮名义值为 `1`，不安排常规候选实验；只有既定硬门禁失败才另立变体。TEMP 中泛称
  `sensitivity` 的文字需按此收窄；
- 首轮不启用 advantage clip；inner epoch、action、记录等权、Short/Long 采样等已裁决项应从旧的
  “当前未决项”列表删除；
- 四项补齐后，应把 TEMP 最终共识与当前较新的问题 9/10 裁决合并成一个权威入口，避免两个文件都
  自称最终合同。

### 执行前工程清单（不计入十个设计问题）

状态：`[未开始]`

- full-parameter 还是 LoRA？reward worker 如何分配 GPU？
- checkpoint 需要保存哪些 RNG、sampler cursor、transition 和 scorer provenance？
- 单卡、最长样本、四卡和 exact-resume 的先后顺序是什么？

### Pilot 与正式授权清单（不计入十个设计问题）

状态：`[未开始]`

- pilot 的步数、学习率、保存频率和最大时长是多少？
- pilot 的成功条件和自动停止条件是什么？
- 什么证据出现后，才重新申请正式训练授权？

## 裁决记录

| 问题 | 状态 | 结论 | 依据 | 已同步文件 |
|---|---|---|---|---|
| QUA Q100 映射（第一题） | 已裁决，产物待构建 | 5,000 条训练原音；100 锚点/99 段；0--10 clip；0--7/0--3 压缩；组内只居中 | 用户裁决；训练原音先导分布 | Q100 合同、算法合同、执行计划 |
| Reward reliability | 待实验 | 不逐组除 std；阈值由最终基座同 prompt 多 seed 标定 | 重复误差与组内有效波动尚未测完 | 算法合同、执行计划 |
| 评价与停止（问题 10） | 已裁决 | rollout/checkpoint/人耳分层；KANA_CER、H/SOFA、GAME 样本级警告/停止门槛；proxy hacking 仅事后归因 | 用户逐项裁决；固定 Eval 相对 step-0 | 待讨论问题、算法合同、执行计划 |
| 十个设计问题 | 已完成逐项讨论 | 仍有实验标定值与最终基座 provenance 待补，不构成新增设计题 | 用户逐项裁决 | 待讨论问题、算法合同、执行计划 |

## 当前边界

```text
design_frozen = false
training_authorized = false
```

















