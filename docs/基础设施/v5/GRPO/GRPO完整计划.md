> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# V5-Pg GRPO 完整计划

> 版本：v0.1（设计与执行总计划）
> 日期：2026-08-13
> 状态：`design_frozen = false`，`training_authorized = false`

## 0. 文档定位

本文是 V5-Pg GRPO 的执行总计划，负责把已经讨论的算法合同、奖励定义、标定顺序、工程门禁、
评价和授权边界串成一条可执行路线。它不授权构建 Q100、不授权实现正式训练，也不授权启动 pilot
或正式训练。

权威分工如下：

1. `GRPO原论文核验与算法合同.md`：论文事实、时间方向、SDE、action、策略角色、ratio/KL 数学合同；
2. `QUA_Q100评分映射合同.md`：UTMOS22/DNSMOS SIG 的 Q100 产物与评分预处理；
3. `GRPO待讨论问题.md`：用户逐项讨论、裁决记录和剩余缺口；
4. 本文：执行顺序、门禁、产物、评价和授权流程；
5. `TEMP/GRPO方向深度探索最终共识.md`：研究共识参考，若与较新的用户裁决冲突，以当前三份 V5
   GRPO 合同和本文为准。

旧 V4c 只提供工程耗时、进程编排和失败经验。不得直接复用其模型、optimizer、reward 混合、
确定性 ODE 配虚构 Gaussian 概率或 checkpoint 状态。

## 1. 目标与边界

### 1.1 首轮目标

首轮 GRPO 只优化终端感知质量，目标按以下三条奖励轴表达：

1. `SIM_INS`：保持目标 B 的唱法/音色相似度；
2. `PITCH_PROB`：保护 GAME 概率音高、voiced/REST 和边界结构；
3. `QUA`：改善发声区纹理、噪音和自然度。

H/PUL、歌词时间放置、长音频能力和绝对旋律回归通过结构与周期评价保护，不把歌词 ASR 放入每步
reward。

### 1.2 明确不做

- `PER` 不进入 per-step reward，训练权重固定为 `0`；
- 不加入 INS adapter、LCF、SVC 自克隆、无 A、扩模或新的 VAE；
- 不把人耳分数混入 reward、advantage 或自动指标总分；
- 不在旧 V4c 脚本上就地覆盖；
- 不因为显存不足静默改变 `G`、reward 权重、评分预处理、VAE 或 prompt 语义。

## 2. 已冻结的核心合同

以下是进入标定和实现时不得悄悄改写的部分。

```text
G = 8（OOM 只由 smoke 决定是否需要另立变体）
组内候选共享初始 latent
PER per-step reward = 0
SIM_INS : PITCH_PROB : QUA = 3 : 2 : 4
QUA 内部 UTMOS22 : DNSMOS SIG = 7 : 3
组内只居中，不除组内 std，不做 z-score
SDE 只对有效 B frame 作用
一个 SDE timestep 的完整有效 B transition 是 action
单帧对 64 channel 求和，再对有效 B frame/SDE step 求 mean
首轮严格一次 on-policy；同一 rollout receipt 不复用
epsilon_lower = 0.002
epsilon_upper = 0.01
KL beta = 1（名义 baseline）
advantage clip = 不启用
任一候选、scorer 或 reliability 轴失败 -> 整组 G=8 跳过
```

### 2.1 基座

基座必须从 V5-Pg 的候选 checkpoint 中，依据同输入轨迹、结构审计和人工听评裁决唯一版本，
同时冻结：

- checkpoint 路径、raw/EMA 来源和 SHA256；
- H/PUL、GAME medium K=4、离散 P class、285k VAE、reference tail 和 A/B 输入契约；
- 推理 NFE、采样前处理、采样率和 CFG 记录格式。

在基座裁决前，不绑定临时 checkpoint 构建正式 Q100、SDE transition 或训练入口。

### 2.2 CFG

CFG 暂不主观冻结，等待 V5-Pg 多 CFG 的同输入、同 seed 轨迹评测。项目 API 为：

```text
pred = cond + g * (cond - uncond)
standard_scale s = 1 + g
```

因此 API `g=0` 对应常见 `s=1`，API `g=1` 对应常见 `s=2`。在最终值冻结前，禁止只写有歧义的
“CFG1”。最终入选档位还必须通过相同 CFG 下的 ODE/SDE 标定。rollout、behavior、current、
reference 的 guided drift 必须完全同构，并同时记录 `cfg_api_g` 与 `cfg_standard_s`。

### 2.3 候选组

每个 prompt 生成 `G=8` 个候选。组内共享：prompt、target、H/PUL、音高条件、时长和 reference。
同组候选共享初始 latent；候选之间的差异只来自 SDE window 的独立 Gaussian noise。独立初始
latent 只作离线对照，不进入首轮正式合同。

reward 只在同一 prompt 的组内比较，不跨任务用绝对分数直接排序。

## 3. Reward 合同

### 3.1 评分波形与评分单元

`SIM_INS` 与 `QUA` 的 runtime 评分单元均为 B-only。A、PAD 和 reference tail 不进入 INS、UTMOS22
或 DNSMOS SIG 的评分波形；GAME 的 full-track 上下文只属于 PITCH 对齐合同。

QUA scorer 的派生副本固定为：44.1 kHz、mono、PCM24，只做线性增益到 `-28 dBFS RMS`；不降噪、
不限幅、不改频谱。训练实际消费的生成波形保持原样，派生副本和原波形分别记录 SHA256。

### 3.2 QUA：UTMOS22 + DNSMOS SIG

最终权威训练 manifest 中均匀无放回抽取 5,000 条记录，固定 seed `20260812`。抽样单位是 manifest
记录，实际 scorer 输入是该记录按正式 A/B 合同 materialize 后的 target B；不得用完整 A+B 代替。

对 UTMOS22 和 DNSMOS SIG 各自独立构建 Q100：

```text
100 个等百分位锚点
99 段单调分段线性映射
范围外 clip 到 [0,10]
U7   = 0.7 * U10
SIG3 = 0.3 * SIG10
QUA10 = U7 + SIG3
```

训练时不直接使用绝对 QUA，而是对两个已映射分量分别组内居中：

```text
C_U   = (U10   - mean_group(U10))   / 10
C_SIG = (SIG10 - mean_group(SIG10)) / 10
C_QUA = 0.7 * C_U + 0.3 * C_SIG
```

`7:3` 表示训练参考坐标中两个压缩分量的允许范围，不承诺任意生成组的实际标准差严格为 7:3。
不在线重估 Q100，不先混合 raw MOS 再统一映射。

若任一 QUA scorer 非有限、输出范围异常、重复误差或评分副本审计失败，整组 `GROUP_INVALID`，
不把剩余轴重放大，不填中性分，不静默重采样。

### 3.3 SIM_INS

使用冻结的 ParaSpeechCLAP Intrinsic revision/checkpoint 和 16 kHz 权威预处理：

```text
generated B -> frozen INS -> normalized embedding e_gen [768]
target B    -> frozen INS -> normalized embedding e_target [768]
R_INS = cosine(e_gen, e_target)
```

训练集 pairwise 标定：随机抽 30 个音频 `a`，每个 `a` 再抽 30 个不同音频 `b`，得到约 900 个 raw
pair cosine。设：

```text
mu_pair = mean(cos(a,b))
f_INS(mu_pair) = 0.5
f_INS(1) = 1
```

`f_INS` 必须单调增。具体单调分段拟合属于 calibration 实现细节，但函数、校准样本和版本必须
冻结；输出低于 `0` 或高于 `1` 时分别 clip 到 `0/1`，并记录 raw、unclipped、clipped 三个值。
校准稳定性按 `a` 做 cluster bootstrap，不能把 900 对当作 900 个独立音频级实验。

正式外层合成不额外乘离线波动增益。某组真实 INS 差异很小，就保留小信号，不强制恢复固定波动。

### 3.4 PITCH_PROB

模型仍接收单个离散 MIDI 音符；reward 比较多个音频经 GAME 得到的概率分布。target 和 generated
都先在完整 A+B(+reference tail) 上下文运行冻结 GAME，再按相同 latent-frame 时间映射精确切 B：

```text
target full track    -> GAME -> [T,128] target_probs    -> exact B slice
generated full track -> GAME -> [T,128] generated_probs -> exact B slice
```

正式 scalar 是 target voiced/REST 分层均衡的 probability cosine：

- target voiced、generated voiced：比较 128-D posterior cosine；
- target voiced、generated REST：0；
- target REST、generated voiced：0；
- 双 REST：1；
- voiced 与 REST 两类分别求 mean，再平衡平均。

PITCH 是旋律和 voiced/REST 的护栏，不是首轮主要优化目标。CKA、F0-CORR、绝对音高和边界统计只
作诊断与回归归因。空类语义、GAME 确定性、exact B slice 和重复误差在 calibration 阶段单测冻结。

正式 PITCH scalar 理论范围已经是 `[0,1]`，首轮使用恒等映射：`mapped_pitch=R_PITCH`。不按
calibration 波动额外放大，不强制其实际贡献恢复到名义权重；calibration 只冻结 reliability 与数值
噪声门槛。

### 3.5 外层 advantage

在每个有效 `G=8` 组内，各基础轴先独立映射、再独立居中：

```text
C_INS   = f_INS(R_INS) - mean_group(f_INS(R_INS))
C_PITCH = R_PITCH - mean_group(R_PITCH)    # identity mapping in [0,1]

A_total = (3*C_INS + 2*C_PITCH + 4*C_QUA) / 9
```

不除组内标准差，不做 leave-one-out，不对总 reward 先混合再 z-score。

## 4. SDE rollout 合同

项目时间方向是噪声 `t=0` 到数据 `t=1`。对真实随机窗口使用：

```text
sigma(t) = a * sqrt((1-t) / t)
b_theta  = v_theta + a^2/(2t) * (-x_t + t*v_theta)
x_next   = x_t + b_theta*delta_t + sigma(t)*sqrt(delta_t)*epsilon
```

实现必须使用 time-shift 后真实 `t_j` 和 `delta_t_j`；`t=0` 不得进入 SDE window。window 外、A、
PAD 和 reference tail 保持真实 ODE，不进入 Gaussian likelihood。

只有有效 B frame 使用 noise、drift correction、mean、variance、log-prob、ratio 和 KL；所有这些
量使用同一个 B mask。SDE 是 rollout 探索和 likelihood 构造手段，不是给训练音频做普通加噪。

论文 baseline 为 `a=0.8, w_min=1, w_s=8`。在最终 CFG 和基座冻结后，按预注册顺序比较：

1. ODE control 与训练 NFE；
2. `a={0.2,0.4,0.6,0.8}`；
3. single-step、`w=2`、`w=4`、`w=8`；
4. 共享初始 latent 与独立初始 latent 对照；
5. 候选质量、发声覆盖、峰值、结构、reward 有效波动、replay 和单位 walltime。

先淘汰本身明显劣化基座的配置，再在收益相差不超过一个标准误的配置中选更低噪声、更小窗口、
更低成本者。最终 `a/window/NFE` 必须写入只读 transition contract。

## 5. 策略、action 与更新

### 5.1 三个策略角色

```text
current   = 接收梯度的 raw policy
behavior  = 每轮 rollout 前从 current 复制的冻结快照
reference = 整个 run 固定的 V5-Pg 基座
EMA       = 可选的评价 shadow，不充当上述任一角色
```

rollout 期间 behavior 固定；reference 整个 run 固定。三者必须使用同一 CFG、同一 transition kernel
和同一 mask 语义。

### 5.2 Action 与 log-prob

一个真实 SDE timestep 的完整有效 B latent transition 是 action：

```text
x_t,B -> x_next,B    # [F,64]
```

由于 covariance 为坐标独立 Gaussian，log-prob 可精确分解：单帧内对 64 channel 求和，再对有效
B frame 和 SDE step 求 mean，使每条训练记录等权。长记录不能仅凭 frame 数获得更大梯度。

rollout 必须保存并可 replay：

```text
x_t, x_next, mean_old, variance
t_j, delta_t_j, B_mask, window, epsilon
逐帧 logp_old
prompt/A/B slice、全部 RNG、policy/reference/scorer provenance
```

不得重新采样 action、重新抽 epsilon 或用 current 伪造 old log-prob。

### 5.3 首轮更新与 KL

首轮 `inner_epochs=1`，严格一次 on-policy：完整 rollout batch 的所有记录和 microbatch 先完成
forward/backward 与 gradient accumulation，之后只调用一次 `optimizer.step()`；同一 receipt 不复用。
更新前 `current == behavior`，所以初始 ratio 应为 1，PPO clip 当次主要承担一致性审计而非实际裁剪。

ratio 只对真实 SDE、有效 B frame 计算：

```text
ratio = exp(logp_current - logp_old)
```

clip 采用论文表格非对称范围：`epsilon_lower=0.002`、`epsilon_upper=0.01`。首轮不启用 advantage
clip；不做任意 log-ratio hard clamp。

相同 covariance 下的 reference KL 为：

```text
KL_frame = sum_d (mean_current_d - mean_reference_d)^2
           / (2 * sigma(t)^2 * delta_t)
```

有效 B frame 与 SDE step 求 mean。`beta=1` 是名义 baseline，不把论文 beta 当作已经校准的绝对强度；
只有既定 KL/ratio/质量硬门禁失败，才另立 beta sensitivity 或 RatioNorm 变体。

## 6. 失败处理与记录

以下任一情况都将整组标记为 `GROUP_INVALID`：

- 候选非有限、全静音、长度异常或真正数字削波；
- transition replay、mask、mean、variance、old log-prob 不一致；
- 任一 INS、PITCH、UTMOS22 或 DNSMOS SIG scorer 失败、低于 reliability、输出范围异常或重复误差
  审计失败；
- GAME full-track 到 exact B slice 不同构；
- 任一评分副本或 provenance/hash 不一致。

失败组统一处理：不计算 advantage，不做 `optimizer.step()`；不填中性分、不删失败候选、不只用
剩余轴、不重新归一化、不静默重采样。所有调试重跑使用新的 `rollout_id`。

每个失败组至少记录 rollout/group/prompt、Pool、Short/Long、所有 policy/reference/scorer SHA256、
candidate seed、共享 latent seed、SDE window、`t/delta_t/mask`、音频 hash/shape/sr/duration/RMS/
peak/finite/nonzero/clipping、每个 scorer 的 status/error/raw score/reliability/重复误差和失败原因。

DDP 下使用全局 valid flag，所有 rank 必须以相同 collective 顺序跳过无效组。

## 7. 工程实现计划

### 7.1 独立入口与资源

建立 V5-Pg 专用 GRPO 入口、transition receipt schema、reward worker、审计器和 exact-resume 比较器。
reward worker 与训练 worker 分离；G=8 可顺序 rollout，WAV、latent 和非必要中间结果及时转 CPU，但
不得改变组内语义。

full-parameter、LoRA 或 last-block 不是当前方向性裁决，先做小型可行性比较。选择满足感知收益和
held-out ODE 无回归的最小可训练范围；若使用 EMA，只作评价 shadow。

### 7.2 必须保存的 checkpoint 状态

- current/raw policy、optimizer、scheduler 和 global step；
- reference provenance 与冻结基座 SHA256；
- EMA（若启用）的独立权重和 provenance；
- sampler cursor、TrainPool manifest、prompt/seed RNG、各 DDP rank RNG；
- rollout counters、optimizer counters、失败组 counters；
- Q100/INS/PITCH 映射、reliability、scorer 和代码 provenance；
- 仍在使用的 rollout receipt 及其 hash。

### 7.3 工程门禁顺序

1. 静态 reward、Q100、INS 映射和公式单测；
2. synthetic Gaussian、SDE replay、ratio=1、KL=0、有限差分方向单测；
3. 单候选完整 sample/decode/reward/backward/checkpoint；
4. 单卡完整 G=8；
5. 最长样本峰值显存与 prompt 时长上限；
6. 四卡 mixed smoke、global valid flag 和 collective 顺序；
7. `continuous10 == 5+resume5` 的 exact-resume 对照；
8. 独立 checkpoint 审计，门禁 checkpoint 不进入正式谱系。

## 8. 评价、警告与停止

### 8.1 评价分层

每个 rollout 记录 reward/advantage、old/new log-prob、ratio/log-ratio/ESS、正负 clipfrac、KL、
loss/gradient/LR、有效性、失败原因、Short/Long、SDE window、walltime 和显存。

固定 checkpoint 使用固定 prompt/seed 的正式 32-step ODE 输出，离线计算 INS、PITCH、UTMOS22、
DNSMOS SIG、QUA、GAME、KANA_CER、H/SOFA、Short/Long、长段后半段、发声覆盖和音频统计；不在每个
rollout 额外运行 Whisper PER。

人耳不进入自动总分，只在预设 checkpoint/pilot 节点作 continue、early-stop 或 veto。

### 8.2 自动回归门禁

正式歌词错误率：

```text
KANA_CER = edit_distance(ref_kana, asr_kana) / len(ref_kana)
```

越低越好且不截断。V4c 的 `max(0, 1-KANA_CER)` 只作为 `V4C_PER_SCORE` 历史兼容字段。

- 超过 10 条样本各自识别分下降超过 10%：警告；
- 超过 20 条样本各自识别分下降超过 20%：立即停止。

H/SOFA 错槽门槛为歌词门槛的 1.5 倍：

- 超过 15 条样本的错槽率各自增加超过 15 个百分点：警告；
- 超过 30 条样本各自增加超过 30 个百分点：立即停止；
- step-0 可对齐而 checkpoint 对齐失败或 mora 数量异常，按该样本完全回归计。

GAME 音高门槛使用 `GAME_ERR=1-PITCH_PROB`，分母为
`max(GAME_ERR_step0, scorer_noise_floor)`：

- 超过 10 条样本误差各自增长超过 10%：警告；
- 超过 20 条样本误差各自增长超过 20%：立即停止。

固定 Eval manifest 与 TrainPool 无 overlap；其具体数量和 Short/Long 报告分桶在执行前冻结即可，
不另设独立 seed 稳健性复核集。

### 8.3 外部停止

以下任一项都停止 pilot 或退回设计阶段：QUA 上升但人工质量不改善；音频有效性失败；PITCH/GAME、
歌词或 H/SOFA 稳定回归；有效组比例过低；ratio/KL/梯度/loss 非有限或系统性触顶；exact-resume
或 provenance 失配；固定节点人耳否决。

不另设自动 `proxy_hacking` 分数。reward 上升同时触发人耳否决或既有回归门禁时，只记录
`proxy_hacking` 事后归因标签和具体 scorer 证据。

## 9. 分阶段执行路线

### 阶段 0：V5-Pg 基座与 CFG 评测

- 完成 V5-Pg 候选 checkpoint 的同输入、同 seed、不同 CFG 轨迹；
- 依据结构、GAME、QUA、Short/Long 和人工听评选择候选；
- 冻结唯一 checkpoint、raw/EMA、SHA256、推理合同和 CFG `g/s`；
- 未完成前不构建正式 Q100 或绑定正式 GRPO 入口。

### 阶段 1：评分与映射标定

- 按 seed `20260812` 构建 QUA 的 5,000 条 Q100 B-only reference；
- 完成 UTMOS22/SIG 锚点、99 段斜率、clip、bootstrap 和 scorer repeatability 审计；
- 构建 INS 30x30 pair calibration，冻结 `f_INS` 锚点与 clip；
- 完成 GAME full-track/exact-B 同构、空类和确定性单测；
- 在最终基座上生成固定 calibration pool，冻结所有 reliability 门。

### 阶段 2：SDE sampler 标定

- 先跑 ODE control 与论文 SDE baseline；
- 扫描 `a/window/NFE` 和 shared-latent 对照；
- 以候选质量、结构门禁、有效波动、replay、walltime 和 one-SE rule 选择最小合格探索配置；
- 不合格的 SDE 配置不进入策略更新。

### 阶段 3：工程门禁

- 完成独立入口、receipt、reward worker、单测、单卡、最长样本、四卡和 exact-resume；
- 确认 `G=8` 的峰值显存和 rollout batch 组数；
- 生成只读 provenance 和门禁报告。

### 阶段 4：短 pilot（需单独授权）

- 从唯一基座 fresh 启动，不继承 V4c optimizer/reward 状态；
- 首轮严格一次 on-policy，不复用 receipt；
- 保存 step 0 和密集早期 checkpoint；
- 同时评价自动 reward、held-out ODE、Short/Long、KANA_CER、H/SOFA、GAME 和人耳；
- pilot 只验证方向，不自动续接正式长训。

### 阶段 5：正式训练（需再次授权）

只有 pilot 通过感知、结构、有效组率、ratio/KL、工程和人耳门禁后，才重新申请：总步数、LR、
参数更新范围、保存/Eval 频率、GPU 编排、输出目录和云端归档。正式训练不得自动从 pilot 续接，
除非授权中明确允许并保存 exact-resume 证据。

## 10. 授权前最终清单

```text
[ ] V5-Pg 唯一基座、raw/EMA、SHA256、CFG g/s
[ ] QUA 5k B-only Q100 只读产物
[ ] INS f_INS 只读映射与 cluster-bootstrap 报告
[ ] GAME exact B slice、空类、确定性和重复误差报告
[ ] UTMOS/SIG/INS/PITCH reliability 门禁
[ ] SDE a/window/NFE/初始 latent 标定报告
[ ] SDE replay、ratio=1、KL=0、synthetic Gaussian 单测
[ ] 单卡 G=8、最长样本、四卡 collective、exact-resume 门禁
[ ] fixed Eval manifest 与 TrainPool 无 overlap
[ ] pilot 参数与授权记录
[ ] 用户明确授权 pilot
```

在以上清单和用户授权完成前，状态保持：

```text
design_frozen = false
training_authorized = false
```
