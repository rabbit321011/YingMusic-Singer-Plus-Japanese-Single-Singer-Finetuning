# GRPO 原论文核验与算法合同

## 1. 本文解决什么

本文把三个容易混在一起的问题拆开：

1. 原始 GRPO 到底如何计算组内 advantage；
2. Flow Matching 模型如何得到可计算条件概率的随机策略；
3. V5-Pg 歌声模型应采用什么时间方向、动作粒度、策略角色和验证门禁。

本文只冻结已经有充分证据的合同。奖励权重仍按既定方向：

```text
G = 8
PER per-step reward = 0
SIM_INS : PITCH_PROB : QUA = 3 : 2 : 4
QUA 内部 UTMOS22 : DNSMOS_SIG = 7 : 3
```

用户已于 2026-08-12 撤销逐组 z-score 合同 Z，并裁决采用训练原音 Q100 分位映射：从最终权威训练
manifest 均匀无放回抽取 5,000 条训练音频，每轴用 100 个等百分位锚点和 99 段直线单调映射并 clip
到 `0--10`，再按目标区间线性压缩；GRPO 只做组内居中。映射设计已经裁决，但 Q100 实际产物、
reliability threshold 和其他训练超参数仍未冻结。完整定义见
[QUA Q100 评分映射合同](QUA_Q100评分映射合同.md)。

## 2. 权威来源与证据边界

| 来源 | 核验内容 | 结论边界 |
|---|---|---|
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | 原始 GRPO、组内 advantage、old/current/reference | LLM 离散 token，不直接给连续音频 SDE |
| [Flow-GRPO](https://arxiv.org/abs/2505.05470) | ODE 到等边际 SDE、Gaussian transition、ratio/KL | Rectified Flow 公式；工程代码与论文在聚合单位上有差异 |
| [Flow-GRPO 官方代码](https://github.com/yifan123/flow_grpo) | rollout log-prob、stored old log-prob、LoRA base reference | 代码持续演化；初始公开实现 commit 为 `fc412d22e69ef6870eeea0bbff29ba19797df677` |
| [MixGRPO](https://arxiv.org/abs/2507.21802) | Mixed ODE-SDE、只优化随机窗口、同组固定初始噪声 | 图像任务结论，窗口位置不能直接外推到本项目 |
| [FlowSE-GRPO](https://arxiv.org/abs/2601.16483) | 与本项目同方向的音频 Flow SDE 换元、早期小窗口 | 语音增强，不是歌声合成 |
| [YingMusic-SVC](https://arxiv.org/abs/2512.04793) | 歌声 DiT、`G=8`、同组初始噪声、单随机步、`a=0.4` | SVC 与本项目任务不同，只作为强先验 |
| [GRPO-Guard](https://arxiv.org/abs/2510.22319) | 连续 Gaussian ratio 左偏、clip 失衡和诊断项 | 修正方法尚不是本项目默认，只登记为失败后的独立变体 |
| [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) | 每 prompt 除以组标准差导致 difficulty bias | 分析来自 LLM，但低方差 prompt 被放大的机制直接适用 |
| [YingMusic-Singer-Plus](https://arxiv.org/abs/2603.24589) | 本模型原论文的多奖励 advantage 和超参数 | 论文没有公开本项目所需的完整 transition 实现 |

文中“论文事实”“官方代码事实”“本项目推导”和“执行建议”不得互相替代。

## 3. 原始 GRPO 的精确定义

### 3.1 组内 advantage

DeepSeekMath 对同一问题从行为策略 `pi_old` 采样 `G` 个输出，得到终端奖励
`r_1 ... r_G`，然后计算：

```text
A_i = (r_i - mean(r_1 ... r_G)) / std(r_1 ... r_G)
```

Outcome supervision 下，同一输出的所有 token 共用同一个 `A_i`。原论文有以下明确边界：

- 均值和标准差包含样本 `i` 自身，不是 leave-one-out；
- 公式没有定义 denominator epsilon；
- 没有规定 population std 还是 sample std；
- `G` 是超参数。DeepSeekMath 实验使用 `G=64`，不是 `G=8`；
- `G=8` 是 YingMusic-Singer-Plus 和其他具体工程的配置，不是 GRPO 定义。

因此实现必须自行冻结 `correction`、数值 epsilon 和退化组规则，不能把框架默认值当论文事实。

### 3.2 PPO surrogate 与三个策略角色

原始目标对每个 token 计算：

```text
ratio = pi_current(action | state) / pi_old(action | state)
surrogate = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
loss = -surrogate + beta * KL(pi_current || pi_ref)
```

三个角色不可合并：

| 角色 | 含义 | 本项目实现 |
|---|---|---|
| behavior / `pi_old` | 真正生成本轮轨迹的快照 | rollout 时保存 transition 和 `logp_old` |
| current / `pi_theta` | 接收梯度的当前模型 | 对保存的 `x_t -> x_next` 重算 `logp_new` |
| reference / `pi_ref` | 防止离开基座过远的冻结模型 | 唯一人工裁决的 V5-Pg 基座，整个 run 不更新 |

若每次 rollout 后只累计一次梯度并在所有样本结束后才 optimizer step，则首次更新前
`current == behavior`，所有 ratio 理应严格为 1，PPO clipping 当次实际上不起作用。只有在同一批
rollout 被分 minibatch 更新或重复多个 inner epoch 时，current 才会逐步偏离 behavior，ratio 与
clip 才有实际约束意义。因此计划必须同时冻结 rollout batch、minibatch、optimizer step 位置和
inner epoch，不能只登记一个 `clip_range`。

V5 首轮更新节奏合同：`inner_epochs=1`；同一完整 `G=8` 组内只做梯度累积，不得中途
`optimizer.step()`。显存不足时允许用 microbatch 拆分 forward/backward，但不得改变组内 advantage、
behavior 快照或 action mask；rollout batch 的组数、完整组梯度累积数量和 optimizer step 频率待 smoke
与 OOM 测试后冻结。

PPO clip 合同直接采用 YingMusic-Singer-Plus 论文实验表格的非对称范围：

```text
epsilon_lower = 0.002
epsilon_upper = 0.01
```

官方 YAML 中 `epsilon_upper=0.02` 与论文表格不一致；V5 采用论文表格值，不为 clip 范围安排候选
实验。只有出现已定义的硬性 ratio/clip 门禁失败时，才允许在独立变体中调整，并不得静默改写正式合同。

KL 名义合同采用论文值 `beta=1`。V5 使用完整 B action、逐帧 64-D 分解、真实 transition covariance
和 `delta_t`，不得假定该数值与论文实现的尺度等价；训练必须记录有效 B frame/随机步的实际 KL、分布
和 held-out 32-step ODE 回归。只有 KL/ratio/质量硬门禁触发时，才可在独立变体中调整 beta，不得
训练中静默自适应。

首轮不启用额外 advantage clip。由于各基础轴已固定映射、组内居中且外层权重已冻结，`A` 只做
finite 与理论范围审计；超出审计范围必须硬失败，不得静默 clamp。正负 clipfrac 不要求对称，必须
分别记录 positive upper / negative lower clipfrac、log-ratio 统计、ratio quantiles/ESS 以及按
SDE step/B frame 分层的 KL。只有 replay、mask、variance 和 old-logp 门禁均通过，且出现持续、
无法由非对称 clip 范围解释的单侧 ratio 重尾、非有限或 held-out 回归时，才停止普通方案并另立
RatioNorm 变体。

## 4. 组内标准化不是无条件正确

### 4.1 它解决了什么

同一 prompt 内居中可以消除任务难度和 scorer 绝对偏置。对多个量纲不同的奖励分别标准化后再
相加，也能避免某个原始数值范围较大的 scorer 仅凭单位占据主导。

### 4.2 它引入了什么

对每个 prompt 都除以自己的组标准差，会把几乎没有真实差异的组也缩放到近似单位方差。后续
GRPO 分析称其为 question-level difficulty bias：低方差 prompt 获得更大的等效权重。

对本项目尤其危险：

- scorer 重载误差或浮点误差可能被放大为完整 advantage；
- `G=8` 时，一个离群 winner 的 population z-score 上限为 `sqrt(7)`，其余七个候选会同时成为
  小幅 loser；
- 两个候选的 UTMOS22 只差很小数值时，z-score 本身不能判断这是量纲小但有效，还是根本无感知差；
- `std + epsilon` 只能防除零，不能证明这组有可学习信号。

三种合同及当前状态如下。合同 Z 已被用户否决；合同 C 被后续分位方案取代；合同 Q100 的映射设计
已裁决，实际锚点产物仍待构建：

#### 合同 Z：已否决的逐组标准化基线

```text
Z_ik = (R_ik - mean_group(R_k)) / std_pop_group(R_k)
```

- 每个基础 scorer 独立计算；
- `std_pop` 明确使用 `correction=0`；
- 先通过 frozen reliability gate，未通过的轴置零；
- epsilon 只作数值保护，不能替代 reliability gate。

#### 合同 C：已被范围方案取代的 frozen-std 候选

```text
C_ik = (R_ik - mean_group(R_k)) / scale_k_frozen
```

其中 `scale_k_frozen` 由最终基座的离线平衡矩阵冻结，例如使用 pooled within-prompt std 或其稳健
估计。它保留组内居中，但不让每个低方差组自动获得单位幅度。这样可以同时处理：

- 原始量纲小但稳定有效：由跨任务 frozen scale 恢复合理幅度；
- 候选实际无差异：centered signal 本身保持小；
- scorer 数值噪声：由 reliability gate 关闭；
- 外层 `3:2:4`：继续只表达目标优先级，不承担量纲修复。

#### 合同 Q100：已裁决的分位映射

```text
p_j   = j / 99, j=0...99
x_kj  = quantile(training_reference_scores_k, p_j, method="linear")
y_j   = 10 * p_j

M_k(r) = piecewise_linear_clip(r; x_k0...x_k99, y_0...y_99)
C_ik   = reliability_gk * (M_k(R_ik) - mean_group(M_k(R_k))) / 10
```

`training_reference_scores_k` 来自最终权威训练 manifest 中均匀无放回抽取的 5,000 条训练音频，
固定抽样 seed 为 `20260812`。`M_k` 是训练原音经验 CDF 的 100 点、99 段单调线性近似；范围外返回
0/10，相邻 raw 锚点重复则构建硬失败。合同 Z 和 C 均不再进入正式实现。Q100 实际锚点、scorer
provenance、reliability 和外层 INS/PITCH 映射全部冻结后才可进入 pilot；当前仍不是 design frozen。

## 5. Flow-GRPO 的随机策略

### 5.1 原论文方向

Flow-GRPO 对其 rectified flow 约定推导：

```text
dx_t = v_theta(x_t,t) dt
sigma(t) = a * sqrt(t / (1-t))

b_theta(x,t) = v_theta(x,t)
  + sigma(t)^2 / (2t) * (x + (1-t) * v_theta(x,t))

x_next = x + b_theta(x,t) * dt + sigma(t) * sqrt(dt) * epsilon
epsilon ~ N(0, I)
```

该 drift correction 来自 score 与 velocity 的关系。只写
`x_next = x + v*dt + a*sqrt(dt)*epsilon` 不是论文的等边际 SDE。

“等边际”也不是离散实现的无条件保证。它依赖正确 velocity/score 关系、连续时间理论和精确积分；
Euler-Maruyama、有限 NFE、CFG、条件 mask 和近似模型都会引入偏差，所以必须用基座实测终端质量。

### 5.2 本项目时间方向换元

YingMusic-Singer-Plus 当前 sampler 从噪声积分到数据：

```text
x_t = (1-t) * epsilon + t * x_data
t: 0 -> 1
dx_t = v_theta(x_t,t) dt
```

其时间方向与 Flow-GRPO 主文的采样记号相反。按 FlowSE-GRPO 的同向换元，本项目候选 SDE 应为：

```text
sigma(t) = a * sqrt((1-t) / t)

b_theta(x,t) = v_theta(x,t)
  + sigma(t)^2 / (2(1-t)) * (-x + t * v_theta(x,t))

等价化简：
b_theta(x,t) = v_theta(x,t)
  + a^2 / (2t) * (-x + t * v_theta(x,t))

x_next = x + b_theta(x,t) * delta_t
  + sigma(t) * sqrt(delta_t) * epsilon_sde
```

约束：

- `t=0` 奇异，step 0 不得进入 SDE window；
- 使用 time-shift 后的实际 `t_j` 和 `delta_t=t_{j+1}-t_j`，不能按均匀步长假设；
- 实现用 FP32 计算 drift、variance 和 log-prob；
- 首轮 SDE window 外保持与基座相同的 Euler ODE，不同时引入高阶 solver；
- rollout CFG 延后裁决。项目源码的 guidance 参数 `g` 满足
  `pred=pred_cond+g*(pred_cond-pred_uncond)`；若用常见写法
  `uncond+s*(cond-uncond)`，则 `s=1+g`。因此 API `g=0` 才是普通 conditional/常见 scale `s=1`，
  API `g=1` 等价于常见 scale `s=2`，且需要 cond/uncond 双 forward。待 V5-Pg 多 CFG 同输入轨迹
  评测后先选 ODE 候选，再在相同 CFG 下做 ODE/SDE 标定；入选 CFG 的 rollout、behavior/current/
  reference mean 和 log-prob 必须完全一致，并承认 guidance 使等边际理论变为工程近似。

### 5.3 Mixed ODE-SDE

设离散随机窗口为 `S`：

```text
j in S:     使用上述 SDE transition，保存并优化
j not in S: 使用确定性 ODE transition，不计算 ratio/KL
```

这与 MixGRPO、FlowSE-GRPO 和 YingMusic-SVC 的共同原则一致：只有真正注入了已知噪声的 transition
才有 Gaussian policy density。

YingMusic-Singer-Plus 论文的 `a=0.8, w_min=1, w_s=8` 作为默认 baseline 候选；FlowSE-GRPO 在音频上
使用更小窗口，YingMusic-SVC 则使用 `G=8`、同组共享初始噪声、单个随机 timestep 和 `a=0.4`。
本项目保留 `a/window/NFE` 消融实验：论文值先跑通作为基线，但若候选质量、ratio/replay 或控制
门禁失败，按预注册实验结果替换，不能把论文值当作不可修改的最终合同。

### 5.4 B-only 与参数裁决合同

SDE 只作用于有效 B 区。固定的 B mask 同时约束 noise、drift correction、transition mean、
log-prob、ratio 和 KL；A、PAD、reference tail 及 window 外坐标保持 ODE，不能进入 Gaussian policy
density。正式推理仍可使用 ODE，SDE 是 rollout 探索和 likelihood 构造手段，不是音频加噪目标。

正式 rollout 的同组候选共享同一个初始 latent；候选差异只由 SDE window 的独立噪声产生。每候选
独立初始 latent 仅用于离线探索幅度和 credit attribution 对照，不进入首轮正式策略合同。

`a`、window 和 NFE 必须按实际 time-shift 网格联合比较。单步每个 latent 坐标的注噪方差及窗口累计
diffusion budget 为：

```text
variance_j = a^2 * (1-t_j) / t_j * delta_t_j
D_SDE(S,a) = sum_{j in S} variance_j
```

因此不得脱离 `t_j/delta_t_j/window/NFE` 单独比较 `a`。参数按以下预注册顺序裁决：

1. 冻结 checkpoint、Q100、prompt/seed、CFG、B mask、time-shift 和 scorer provenance；
2. 在固定哨兵集做所有合法 single-step sweep，再验证连续 `w=2/4/8`，不得先验指定前八步最优；
3. 共享初始 latent 的矩阵测组内探索；独立初始 latent 的 ODE/SDE 对照测边际分布和结构漂移；
4. transition replay、`current==behavior` ratio、有限性、静音/长度/削波、ODE/SDE 分布漂移、scorer
   重复误差及轴间冲突任一硬门禁失败即淘汰；
5. 通过门禁后，以扣除 scorer 重复方差的 `A_total` 组内真方差除 rollout walltime 排探索效率，
   其中 `A_total=(3*C_INS+2*C_PITCH+4*QUA)/9`；
6. 入围配置执行等预算短 pilot，只在 held-out 正式 32-step ODE 上比较；使用 one-standard-error rule，
   在与最佳收益相差不超过一个标准误的配置中选择更低 `a`、更小窗口和更低训练成本者；
7. 自动指标只能筛选，最终盲听只作 reward hacking 安全否决，不用于看结果后反向调参。

当前参数裁决程序已冻结，`a`、window 位置/宽度和训练 NFE 仍为待实验产物。

## 6. 动作、likelihood 分解与记录等权

### 6.1 三个概念必须分开

用户于 2026-08-13 裁决每条训练记录等权。正式合同为：

- **环境 action**：一个真实 SDE timestep 的整个有效 B latent transition `x_next_B [F,64]`；
- **likelihood 分解**：因条件 covariance 是坐标独立的 Gaussian，完整 action 的 log-prob 可以精确
  分解为逐帧 64-D contribution；
- **loss 归约**：先在每个有效 B frame 内对 64 个 channel 的 log-prob 求和，再对有效 B frame 与
  SDE step 求 mean，使每条训练记录等权，不让长音频仅因 frame 更多获得更大梯度。

逐帧保存 likelihood contribution 不等于把环境 action 重新定义成独立帧，也不改变真实 transition。

### 6.2 正式一次 on-policy loss

```text
logp_theta[i,j,f]
  = sum_d log Normal(x_next[i,j,f,d]; mean_theta[i,j,f,d], variance_j)

log_ratio[i,j,f] = logp_current - logp_old
ratio[i,j,f] = exp(log_ratio)

loss_record_i
  = mean_{j in S, f in valid_B}
      -A_i * logp_current[i,j,f]
    + beta * KL_frame[i,j,f]
```

首轮采用严格一次 on-policy 更新：完整 rollout batch 的所有记录/microbatch 必须先完成 forward 与
gradient accumulation，之后只调用一次 `optimizer.step()`；同一收据不重复使用。更新前
`current == behavior`，所以 `ratio=1`、PPO clip 不生效，policy-gradient 项可以直接用上式的
`-A*logp_current` 表达。仍计算并审计 ratio，以验证 old/current/replay 完全一致。

不得在处理完第一个 microbatch 后先 `optimizer.step()` 再处理后续 microbatch；否则后续数据的
`current != behavior`，已不是本合同的一次 on-policy 更新。若未来需要复用 rollout 或多次更新，必须
重新讨论高维 joint ratio 与 per-frame PPO surrogate，不自动沿用本合同。frame-block 不进入首轮。

rollout 必须保存实际 `x_t/x_next/mean_old/variance/t_j/delta_t/B_mask/window/epsilon` 及逐帧
`logp_old`。实现必须用同一收据逐步重建 `x_next`，并由保存的 `x_next/mean_old/variance` 重算
`logp_old`；更新后不得重新采样 action、重新抽 epsilon 或用 current 伪造 old log-prob。Gaussian
ratio 只允许使用真实 SDE、有效 B frame，ODE、A、PAD、reference tail 和 window 外坐标全部排除，
noise、mean、log-prob、ratio、KL 使用同一个 mask。

SDE 已裁决为只施加在有效 B 区；drift correction、noise mask、log-prob 和 KL 必须覆盖同一 B
mask。不能把 A、PAD、reference tail、window 外坐标或其他 ODE transition 假装成存在 Gaussian
variance 的动作。

### 6.3 KL

对相同 covariance 的当前策略与冻结 reference：

```text
KL_frame = sum_d (mean_current_d - mean_ref_d)^2 / (2 * sigma(t)^2 * delta_t)
```

最终在有效 B frame 与随机步上平均，使 KL 同样按记录等权。若从 drift difference 化简，必须使用本项目换元后的完整
`b_theta`，不能直接把 `v_current-v_ref` 塞进旧公式。

## 7. 为什么 V4c 最终 GRPO 不能复用

### 7.1 最终 v3

历史 [`train_grpo_v3.py`](../../../../scripts_archive/yingmusic_plus/3_train_grpo/train_grpo_v3.py)
存在四个根本问题：

1. rollout 调用 `Singer.sample()`，实际是 `torchdiffeq.odeint` 的确定性 ODE，并使用 CFG；
2. 随后把 `(x_next-x_t)/dt` 当成 Gaussian SDE 动作，凭空指定 `alpha=0.8`；
3. 冻结 reference 同时充当 `pi_old` 和 `pi_ref`，不是生成轨迹的 behavior；
4. 先按 `[6,1,1,2]` 混合四个 raw reward，再只对 total 做一次 z-score。

ODE transition 条件分布是 Dirac delta，不是正方差 Gaussian。不同初始噪声虽然能产生多个终端
样本，但模型不控制初始噪声分布，无法用伪造的逐步 Gaussian likelihood 得到正确策略梯度。

### 7.2 较早 package

[`package_v4c_finetune/grpo/grpo_utils.py`](../../../../package_v4c_finetune/grpo/grpo_utils.py)
比最终 v3 更接近正确结构：它保存采样时 velocity 作为 old，并对各 reward 独立标准化。但其
SDE 只是 `v*dt + alpha*sqrt(dt)*noise`，缺少等边际 drift correction；scorer 失败还会填均值继续。
它可以作为测试接口参考，不能作为 V5 算法实现起点。

## 8. V5 奖励与 advantage 候选

### 8.1 基础奖励

```text
R_INS   = cosine(ParaSpeechCLAP-Intrinsic(gen_B), target_B)
R_PITCH = frozen GAME probability-space scalar
R_U     = UTMOS22(equal-loudness scoring copy)
R_SIG   = DNSMOS SIG(equal-loudness scoring copy)
```

QUA 的唯一评分单元为 B-only：runtime 对 generated B 建立等响评分副本；5k Q100 对抽中的训练记录
materialize target B 后建立同构评分副本。A、PAD 和 reference tail 均不进入 UTMOS22/DNSMOS SIG。
GAME 的 full-track 上下文只服务 PITCH，不改变该 QUA 合同。

组内候选共享 prompt、target、音高条件和 reference，固定的音素/内容因素大多在组内比较中抵消；
因此 V5 首轮将 `R_INS` 按项目主要目标定义为唱法相似度轴，不额外设置 INS 专项反事实审计作为
进入 pilot 的前置门槛。通用的人耳、held-out 回归和 reward-hacking 停止条件仍然有效。

QUA 继续作为主要质量轴，外层权重为 4。在 KL 与冻结 reference 约束下，适度频谱变软、辅音能量、
发声覆盖或响度/动态变化不预先判为 reward hacking，也不为这些变化额外加入禁止项；仅保留非有限、
全静音、长度异常和真正数字削波等音频有效性硬门禁。若后续人工或 held-out 证据显示 QUA 上升与
实际质量相反，再单独修订 QUA 合同。

`SIM_INS/PITCH_PROB/QUA` 是训练中间件；GAME、QUA、PER、H/SOFA 等是自动评测与归因指标；人耳不
进入 reward、advantage 或自动指标总分。人耳只在预先固定的 checkpoint/pilot 节点承担外部
early-stop/veto，决定训练路线是否继续。因此 PITCH 上升而听感变差不是同一标量空间的比较题：
PITCH 仍报告其护栏状态，人耳独立决定路线继续或停止。

训练 prompt 与 rollout seed 按 TrainPool/RNG 随机推进；checkpoint 评测使用固定且与 TrainPool 无
overlap 的 Short/Long prompt 集及固定 seed 集，并保存 manifest、条件、初始 latent、SDE noise 和
scorer provenance。V5 不另设独立 seed 稳健性复核集。

评测分层合同：每个 rollout 记录 reward/advantage、old/new log-prob、ratio/log-ratio/ESS、正负
clipfrac、KL、loss/gradient/LR、`GROUP_VALID/GROUP_INVALID`、失败原因、音频有效性、Short/Long、
SDE window、walltime 和显存，并将 `rollout_attempt` 与 `optimizer_step` 分开计数；不在每个 rollout
额外运行 PER。固定 checkpoint 以固定 prompt/seed 的正式 32-step ODE 输出计算 reward 轴、GAME
扩展音高诊断、F0-CORR/CKA、PER、H/SOFA、Short/Long/Long 后段和音频统计。人耳只在预设节点承担
continue/early-stop/veto；具体保存、评测和试听频率留到 pilot 参数裁决。

歌词回归的正式错误率字段为
`KANA_CER = edit_distance(ref_kana, asr_kana) / len(ref_kana)`，方向为越低越好，且不将大于 `1.0`
的结果截断。V4c 日志中的 `PER` 实为 `max(0, 1-KANA_CER)`；V5 仅以
`V4C_PER_SCORE` 名称输出该历史兼容值，不用它定义正式回归方向。

歌词回归门禁比较固定 Eval 上同一样本的 checkpoint 与 step-0/reference：超过 `20` 条样本分别下降
超过 `20%` 时立即停止；超过 `10` 条样本分别下降超过 `10%` 时警告。停止优先于警告，且同时报告
退化样本清单和对应 `KANA_CER`，不能只看全体均值。

H/SOFA 使用生成音频与固定输入的 mora 错槽率，并计算相对 step-0 的逐样本百分点增量。其容许门槛
为歌词门禁的 `1.5x`：超过 `15` 条样本各自增加超过 `15` 个百分点时警告；超过 `30` 条样本各自
增加超过 `30` 个百分点时停止。step-0 可对齐而 checkpoint SOFA 失败或 mora 数量异常时，该样本按
完全回归计；句首误差、相对 onset P95 和相邻间隔误差只作归因诊断。

GAME 音高回归门禁使用 `GAME_ERR = 1-PITCH_PROB`，而非直接对接近 `1` 的 cosine 计算下降比例。
固定 Eval 中超过 `10` 条样本的 `GAME_ERR` 相对 step-0 各自增长超过 `10%` 时警告；超过 `20` 条
样本各自增长超过 `20%` 时停止。增长率分母为
`max(GAME_ERR_step0, scorer_noise_floor)`；`scorer_noise_floor` 由 pilot 前重复评分误差标定。另行
报告 voiced/REST、绝对音高和 F0-CORR 以归因，但不为其重复设置停止门槛。人工听评保持外部
early-stop/veto，不定义成自动指标空间内的数值阈值。

不增加独立的自动 `proxy_hacking` 判定。音频有效性门禁、歌词/H-SOFA/GAME 回归门禁、KL/ratio/
梯度健康监控以及外部人耳 early-stop/veto 已覆盖路线停止；当 reward 上升却触发人耳否决或已有
回归门禁时，`proxy_hacking` 只作为事后归因标签，并记录被利用的 scorer 偏好和对应证据。

PITCH 正式 scalar 冻结为按 target voiced/REST 分层均衡的 GAME probability cosine。目标 voiced
且生成 voiced 时比较两个 128 维 GAME posterior；目标 voiced/生成 REST 与目标 REST/生成 voiced
均为 0；双 REST 为 1；对 voiced 与 REST 两类分别求均值后再平衡平均。Linear CKA 与论文口径的
F0-CORR 只作离线诊断、结构回归和 reward-hacking 门禁，不进入首轮主 reward。

PITCH_PROB 的职责是防止 SIM/QUA 微调时旋律、绝对音高、REST/voiced 和边界跑偏，而不是首轮微调
的主要优化目标；外层名义权重仍为 `SIM_INS:PITCH_PROB:QUA = 3:2:4`，其中 PITCH 的 `2` 是护栏
预算。GAME 两端必须复用 V5-P medium K=4、pitch-axis adapter、REST 语义和同一目标时间网格；
reliability 门槛、空类处理、scorer 重复误差和最终对齐产物待离线标定。

GRPO advantage 只做组内中心化，因此该 reward 只能抑制同组候选之间的相对音高跑偏；若整个组共同
漂移，必须由 held-out 绝对音高回归门禁或停止条件拦截，不能通过提高名义 PITCH 权重补救。

### 8.2 名义权重

当前合同 Q100 下：

```text
U10_i   = M_U(R_U,i)
SIG10_i = M_SIG(R_SIG,i)

C_U,i   = reliability_U   * (U10_i   - mean_group(U10)) / 10
C_SIG,i = reliability_SIG * (SIG10_i - mean_group(SIG10)) / 10

QUA_i = 0.7 * C_U,i + 0.3 * C_SIG,i

A_i = (3 * C_INS,i
     + 2 * C_PITCH,i
     + 4 * QUA_i) / 9
```

报告尺度等价写法为 `U7=0.7*U10`、`SIG3=0.3*SIG10`、`QUA10=U7+SIG3`。因此两个分量的
允许范围严格为 `0--7` 与 `0--3`。训练使用上面的除 10 等价形式并只做组内居中，不除以组内 std；
训练中不得在线重估 Q100 锚点。

reliability 首轮采用离线冻结的二值门，不做在线可学习权重：

- 任一候选非有限、静音、长度异常、真正数字削波、transition/replay 不一致或 scorer 调用失败：
  整组状态为 `GROUP_INVALID`；
- 任一 INS/PITCH/UTMOS22/DNSMOS SIG 轴输出低于 reliability 门槛，或该轴重复误差、输出范围、
  时间对齐审计失败：同样整组 `GROUP_INVALID`，不能只关闭该轴继续；
- 任一 `GROUP_INVALID` 整组 `G=8` 跳过，不计算 advantage，不 `optimizer.step()`；不得填中性分、
  静默重采样、删除失败候选、只保留剩余轴或重新归一化权重；
- 首轮不启用 advantage clip；仅执行既定 finite 与理论范围硬审计。

每个失败组必须记录失败原因、`rollout_id/group_id/prompt SampleId/Pool/Short-Long`、manifest/policy/
behavior/reference/scorer provenance SHA256: redacted
音频 hash/shape/sample rate/duration/RMS/peak/finite/nonzero/clipping 统计，以及逐 scorer 的 status、
error class、异常消息摘要、输入 hash、输出 shape/dtype/finite/range、raw score、reliability threshold、
重复误差和失败轴。调试重跑必须创建新的 `rollout_id` 并与原失败组分开登记，不得伪装成原组成功。

## 9. 必做标定

### 9.1 Normalizer 标定

先按 [QUA Q100 评分映射合同](QUA_Q100评分映射合同.md)从最终权威训练 manifest 抽取 5,000 条训练
音频，完成 UTMOS22 与 DNSMOS SIG 的评分、100 锚点构建、局部斜率/bootstrap/clip 审计并冻结只读
产物。随后在最终 V5-Pg 基座上生成固定 27 组任务、每组十个可控候选，枚举所有 `G=4/G=8` 子组，
完成：

1. 每轴 scorer reload repeatability；
2. group range/std 与重复误差的比值；
3. 低方差组在 Z 下获得的等效权重分布；
4. 每轴的 frozen reliability threshold 与 bootstrap 置信区间；
5. 候选排序、advantage 极值、轴间相关和冲突；
6. 每轴与复合分的最高/最低试听或结构审计包；
7. 有效组与 `GROUP_INVALID` 比例，并按失败候选、失败轴和失败原因分层。

主矩阵应固定同组初始 latent，只改变 SDE noise；另做独立初始 latent 对照，判断探索幅度与
credit attribution 的差异。

### 9.2 SDE sampler 标定

顺序执行，避免一次做巨大全因子矩阵：

1. ODE control：在相同基座上比较训练候选 NFE 与正式 32-step 推理；
2. 小型哨兵集扫描 `a={0.2,0.4,0.6,0.8}`；
3. 比较 single-step、`w=2`、`w=4` 和论文 `w=8`；
4. 比较同组共享初始 latent 与每候选独立初始 latent；
5. 对入围配置扩展到 27 组十候选；
6. 比较 ODE/SDE 的终端 latent、INS、PITCH、QUA、发声覆盖、峰值、结构和人工听感；
7. 验证 `x_next == mean_old + sigma*sqrt(dt)*epsilon` 的逐步 replay。

任何 SDE 配置若基座本身已明显劣化，就不能进入 RL，即使 reward range 更大。

## 10. Ratio 与策略门禁

实现前后必须通过：

1. `current == behavior` 时，存储与重算 `logp_old` 一致，ratio 全为 1；
2. `current == reference` 时，KL 全为 0；
3. 人工微扰参数后，ratio、clipfrac 和 KL 按预期变化；
4. synthetic Gaussian Monte Carlo 下，density ratio 的统计量与解析值一致；
5. 正/负 advantage 的 `clipfrac_gt_one/lt_one` 分开记录；
6. ratio 按 SDE step、音频长度、B frame 和困难域分层；
7. 不允许用任意 log-ratio hard clamp 掩盖 stale rollout；非有限或异常尾部必须停批次；
8. 若正 advantage 长期几乎从不触发上界、负 advantage 却频繁触发下界，停止普通 Flow-GRPO，
   再单独评估 GRPO-Guard/RatioNorm，不得训练中静默切换算法。

## 11. 执行顺序

```text
最终 V5-Pg 人工裁决
  -> 冻结 checkpoint/raw-or-EMA/SHA256: redacted
  -> 论文公式的本项目时间方向推导与单元测试
  -> 合同 Q100 的训练参考/锚点/reliability/clip 离线冻结
  -> SDE NFE/a/window/initial-latent 标定
  -> 冻结 reward contract + transition contract
  -> 单组 replay/ratio/KL 门禁
  -> 单卡 G=8
  -> 最长样本
  -> 四卡 collective + exact resume
  -> 另行申请短 pilot 授权
  -> 人工听评后才讨论正式训练
```

当前状态：

```text
design_frozen = false
training_authorized = false
```

















