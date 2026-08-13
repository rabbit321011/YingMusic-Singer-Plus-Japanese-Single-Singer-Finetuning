# QUA Q100 评分映射合同

## 1. 裁决

本合同冻结 V5-Pg GRPO 的 QUA 映射方法。用户于 2026-08-12 裁决：

```text
不用逐组 z-score
训练原音参考数 = 5,000
raw score -> Q100 百分位分段映射到 0--10 -> 线性压缩
UTMOS22 : DNSMOS SIG 的压缩表示范围 = 7 : 3
GRPO 只做组内居中，不除以组内 std
```

本文只冻结映射设计，不表示 Q100 表已经生成，也不授权 GRPO 训练。

## 2. 两个基础评分

```text
R_U   = UTMOS22(equal-loudness scoring copy)
R_SIG = DNSMOS SIG(equal-loudness scoring copy)
```

两者都是高分方向更好。评分副本固定为 44.1 kHz、mono、PCM24，只施加线性增益统一至
`-28 dBFS RMS`；不降噪、不限幅、不改频谱。非有限、静音、长度异常或峰值门禁失败必须硬失败，
不得用替代分数或静默换样本。

评分单元固定为 B-only。训练 runtime 对 generated B 建立等响评分副本；Q100 对抽中的 manifest 记录
先按正式 A/B 合同 materialize target B，再建立同构评分副本。A、PAD 和 reference tail 均不进入
UTMOS22/DNSMOS SIG；两边的裁切、最短长度处理和 scorer 前处理必须完全同构。GAME 为保证边界上下文
而使用 full-track 的合同只属于 PITCH，不改变 QUA 的 B-only 单元。

## 3. 5,000 条训练参考采样

- 输入必须是最终训练数据的权威 manifest，而不是临时试听池或模型生成音频；
- 抽样单位是一条实际训练 manifest 记录，即一个训练音频片段；从通过音频存在性、有限值、非静音和
  长度审计的记录中均匀无放回抽取 5,000 条；
- 抽样概率仍按 manifest 记录；实际进入 scorer 的是该记录 materialize 后的 target B，不得用完整
  A+B、A、PAD 或 reference tail 代替，也不得因 B 时长而修改抽样概率；
- 固定 `q100_reference_seed = 20260812`，并保存抽样前后 manifest SHA256: redacted
- 不对时长、歌曲、音高或困难域做等额重平衡，以保留训练记录的真实占比；
- 另行报告 Short/Long、时长、BV/source、音高和轻柔/困难域覆盖，但审计结果不改变已抽样概率；
- 报告重复音频路径、重复内容 hash 和同歌曲入选数；发现 manifest 异常重复时整次失败，修复权威
  manifest 后按同一 seed 重抽，不在抽样器内临时去重；
- 任一入选样本评分失败则整次构建失败，不允许静默补抽；
- 当前十档实验的 2,000 条原音只作为前期证据，不能充当最终 Q100 表。

5,000 条对应每个 1% 百分位区间约 50 条样本，足以支持 100 个锚点的初版冻结与 bootstrap 审计。

## 4. 第一次映射：Q100

UTMOS22 与 DNSMOS SIG 各自独立构建一张 Q100 表。对某个 scorer `k`：

```text
j = 0 ... 99
p_j = j / 99
x_kj = quantile(training_reference_scores_k, p_j, method="linear")
y_j = 10 * p_j
```

100 个 `(x_kj, y_j)` 锚点形成 99 段直线。对新 raw score `r`：

```text
r <= x_k0  -> M_k(r) = 0
r >= x_k99 -> M_k(r) = 10

x_kj < r < x_k,j+1:
M_k(r) = y_j
       + (y_j+1 - y_j) * (r - x_kj) / (x_k,j+1 - x_kj)
```

实现必须二分查找区间并用 FP64 构建/保存锚点。`M_k(r)=9` 的语义近似为 raw score 高于训练
参考分布 90% 的样本。映射严格单调不减；范围外统一归到 0 或 10。

相邻 raw 锚点若相等会导致零宽区间。Q100 构建器必须硬失败并报告重复区间，不得静默加 epsilon；
是否引入 mid-rank plateau 规则需重新申请裁决。

## 5. 第二次映射：线性压缩

```text
U10   = M_U(R_U)       # [0, 10]
SIG10 = M_SIG(R_SIG)   # [0, 10]

U7    = 0.7 * U10      # [0, 7]
SIG3  = 0.3 * SIG10    # [0, 3]

QUA10 = U7 + SIG3      # [0, 10]
```

第一步是固定的单调分段线性函数；第二步是纯一次函数。训练参考分布经 Q100 后近似均匀，所以
`U10/SIG10` 在参考集上具有近似相同的范围和标准差；线性压缩后两分量的表示范围与参考集波动均
近似 `7:3`。该结论不承诺任意生成模型或每个 `G=8` 组的实际 std 都严格为 `7:3`。

## 6. GRPO 组内 advantage

训练不直接使用绝对 `QUA10`，而是只做同 prompt 组内居中：

```text
A_QUA_i = reliability_U
          * (U7_i - mean_group(U7)) / 10
        + reliability_SIG
          * (SIG3_i - mean_group(SIG3)) / 10
```

- 均值包含候选自身，不使用 leave-one-out；
- 不除以当组 std，不做 z-score；
- 任一 UTMOS22 或 DNSMOS SIG 轴输出低于 reliability 门槛、重复误差/范围/评分副本审计失败：整组
  `GROUP_INVALID`，不把该轴置零后继续；
- 任一 QUA 轴失败，整组 `G=8` 跳过且不 `optimizer.step()`；不把剩余轴自动重放大，不填中性分，
  不静默重采样；
- `QUA10/U10/SIG10/U7/SIG3` 与居中后的两个分量必须全部记录。

## 7. Reliability 与局部斜率

百分位映射会拉伸训练分布的高密度区域、压缩稀疏尾部。它是全程冻结的全局映射，不会像逐组 Z
那样把每个低方差组拉成单位方差，但仍必须审计：

1. scorer 同实例重复、重载重复和跨设备重复误差；
2. 99 段 raw width、映射斜率及最大/P95 斜率；
3. raw 与 mapped 空间中的组内极差和重复误差比；
4. 0/10 clip 率及其 prompt/困难域分层；
5. 映射前后候选排序必须完全一致；
6. U7/SIG3 极值候选的等响试听与 reward hacking 审计。

reliability threshold 仍需在最终 V5-Pg 多 seed 候选上冻结，不能从训练原音跨样本分布直接推出。

## 8. 冻结产物

正式 Q100 构建至少输出：

```text
q100_reference_manifest.jsonl
q100_reference_audit.json
q100_utmos22_knots.csv
q100_dnsmos_sig_knots.csv
q100_contract.json
q100_mapping_audit.json
SHA256: redacted
```

`q100_contract.json` 必须记录 sample seed、quantile method、scorer revision/weight SHA256: redacted
训练 manifest SHA256: redacted
根据在线 rollout 重新估计分位数。

## 9. 当前状态

```text
mapping_design = decided
reference_sample_count = 5000
q100_artifact_built = false
reliability_frozen = false
design_frozen = false
training_authorized = false
```

















