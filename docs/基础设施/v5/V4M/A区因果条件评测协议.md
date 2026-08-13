# A 区因果条件评测协议

## 目的

训练日志中的 `FlowA` 只衡量模型在 A 区自身的 flow-matching 误差：

```text
FlowA-self = MSE(prediction_A, target_A)
```

它不能回答 A 作为参考条件是否被模型读取，也不能回答 A 的音色信息是否正确影响 B。A 在实际任务
中的职责是为 B 提供参考，因此本协议把需要长期复用的诊断命名为 `FlowA-CF`：A 区反事实条件
评测。`CF` 表示 counterfactual。

`FlowA-CF` 是一组固定实验和指标，不是一个新的训练 loss，也不压缩成单一标量。现有日志字段
`FlowA` 不改语义；在文档中需要区分时写作 `FlowA-self`。

## 要回答的问题

`FlowA-CF` 依次回答：

1. 正确 A 下，B 的预测是否准确；
2. 把 A 换成合理反事实后，B 的误差和输出是否发生有序变化；
3. 变化是否主要落在音色/声学，而不是污染文本、旋律或节奏结构；
4. 更低 Loss 是否来自真实预测改善，而不是方差收缩或平均化；
5. 上述结论是否集中在特定 timestep 或样本难度区间。

## 冻结反事实合同

一条 `FlowA-CF` 样本先冻结 B，再构造多个 A。模型之间必须读取同一份模型无关 stimulus，禁止
各自随机生成条件。

### B 区必须固定

- B ground-truth latent；
- B noise 和 flow target；
- B 文本与 placement；
- B MIDI/GAME-P 与 CKA target；
- `ref_len`、timestep；
- 其他推理或训练图条件。

### A 区条件

最小集合为：

1. 原始配对 A；
2. 同歌手不同片段 A；
3. 不同歌手、音色 embedding 较近的 A；
4. 不同歌手、音色 embedding 较远的 A；
5. 全零 A。

near/far 必须由冻结候选池和明确的 embedding 排序产生，不能凭文件名或主观标签猜测。候选池、
embedding 模型、权重 SHA256: redacted

### 同步替换

替换 A 时必须同步改变：

```text
cond_A
x_t 的 A 部分
target 的 A 部分
```

A noise 保持固定。这样每个 A 都满足同一 flow path：

```text
x_t = (1-t) * noise + t * x1
target = x1 - noise
```

禁止只改 `cond_A` 而保留另一个 A 的 noisy latent；这种输入自相矛盾，所得差异不能解释为条件
利用。所有 A 必须确定性裁剪或补齐到相同 `ref_len`。

## 标准指标

### 正确 A 的 B 预测质量

原始配对 A 下报告：

- FlowB；
- CKA；
- 当前冻结 loss 配方的完整 Loss；
- prediction-target identity R2；
- prediction-target correlation；
- 回归斜率；
- `std(prediction)/std(target)`。

Loss、R2、斜率和方差比必须一起解释。Loss 降低但 R2 不升、斜率远离 1 或方差比继续下降，不能
裁决为真实改善。

### A 条件选择性

对每种反事实 A 分别计算：

```text
ΔFlowB(A_i) = FlowB(A_i) - FlowB(A_correct)
ΔCKA(A_i)   = CKA(A_i)   - CKA(A_correct)
```

这些是有直接训练语义的配对差值：错误或缺失参考是否使同一个 B 更难预测。same、near、far、zero
必须分开报告，不能先混成一个均值。

判断“更强的有效 A 利用”至少需要：

- 正确 A 的 B 预测更好；
- 合理错误 A 的 ΔFlowB 更明确；
- far 相对 near 呈现稳定的有序关系；
- 改善不是只有 zero 这种极端条件成立。

只对 zero A 更敏感不足以证明自然参考选择性增强。

### B 输出响应

在固定 B、noise 和 timestep 下，比较不同 A 对应的 B 输出估计。报告绝对距离：

- VAE latent distance；
- mel/频谱 distance；
- 可靠音色 embedding distance；
- F0 contour distance。

这些距离回答“B 是否随 A 改变”和“变化落在哪个声学维度”。它们各自保留原单位和量纲，不再
相除构造所谓统一增益。

### 距离有序性

用冻结样本集合分析：

```text
corr(distance_A, ΔFlowB)
corr(distance_A, distance_B)
```

同时报告 Pearson、Spearman 和单变量回归 R2。A distance 至少包含 VAE latent、mel/频谱和可靠
音色 embedding。相关性只表示反事实距离与响应是否有序，不表示 A distance 是唯一因果变量。

低 t 必须单独报告，因为此时 `x_t` 中 ground truth 少、模型不确定性高，A 条件作用最容易被观察。
把全部 timestep 混在一起会被高 t 的近 ground-truth 状态稀释。

### 旋律与结构泄漏

主要观察：

```text
|ΔCKA|
F0 contour distance
```

若 A 音色变化带来 FlowB、B latent/mel/音色变化，而 CKA 与 F0 相对稳定，支持音色与旋律解耦。
若 CKA 或 F0 随 A 大幅变化，则 A 条件可能错误干扰 B 的旋律结构。

F0 稳定不是要求所有 A 条件输出逐帧完全相同；需要与 B 音色距离、FlowB 和当前 timestep 一起配对
解释。

### 平均化门禁

模型更低 Loss 时必须同时检查：

- R2 是否提高；
- correlation 是否提高；
- regression slope 是否接近 1；
- `std(prediction)/std(target)` 是否接近 1，而不是继续缩小；
- B 输出对 A 的响应是否整体消失。

满足前三项且方差不进一步收缩，才支持真实预测改善。若换 A 后 B 输出和 ΔFlowB 同时明显变小，
再结合方差收缩，才支持忽略 A 或向平均音色回归。

## 不采用 A→B 距离比

旧诊断实现曾输出：

```text
distance(B_i, B_j) / distance(A_i, A_j)
```

该比值不进入 `FlowA-CF` 标准协议，也不参与模型裁决，原因是：

1. 没有经人工感知或任务目标标定，不存在已知的合理目标值；
2. 比值大既可能表示有效条件利用，也可能表示模型对参考过度敏感，方向本身不代表好坏；
3. 分母接近零时不稳定，不同距离空间的局部尺度也不同；
4. 模型响应允许非线性，强行解释为“增益”没有依据；
5. 它不能区分音色的正确传递和内容、响度、F0 等混杂。

历史 CSV 中已计算的列只为运行追溯保留。后续汇总、文档和门禁忽略这些列，不重新命名或赋予物理
意义。

## Timestep 与难度

标准运行至少覆盖训练 timestep distribution 的五个固定分位点，并单列最低两个桶。随机 timestep
可以作为补充，但不能替代固定网格，否则 checkpoint 间配对噪声过大。

样本难度由基线模型在正确 A 下的 FlowB 冻结定义。至少报告五分位：

- 容量收益只集中在容易样本：主要拟合主分布；
- 困难尾部改善更大且趋势稳定：支持有价值的 scaling；
- 全分布小幅改善但不单调：广谱边际收益，不宣称尾部专项突破。

难度分桶不能由待评模型重新计算，否则模型间样本集合会改变。

## 输出形式

`FlowA-CF` 的正式结果是一组 profile：

```text
paired: FlowB / CKA / full Loss / R2 / corr / slope / std ratio
selectivity: ΔFlowB_same / near / far / zero
response: B latent / mel / timbre / F0 distance
ordering: corr(A distance, ΔFlowB) and corr(A distance, B distance)
leakage: ΔCKA and F0 response
strata: fixed timestep buckets and baseline difficulty quantiles
```

逐样本 CSV/JSON、聚合报告、bootstrap CI、checkpoint SHA256: redacted
必需产物。模型比较使用逐样本配对统计；不能只报告总体均值。

## 判据

| 观察 | 解释 |
|---|---|
| 正确 A 改善；错误 A 惩罚有序；B 音色响应存在；CKA/F0 稳定 | A 条件利用有效 |
| 换 A 后 ΔFlowB 和 B 多模态距离普遍缩小；预测方差继续收缩 | 忽略 A 或平均化风险 |
| B 音色响应存在，但正确/错误 A 惩罚不分离 | A 通路存在，选择性未增强 |
| B 音色、CKA、F0 同时大幅变化 | A 与旋律结构错误纠缠 |
| Loss、R2、斜率、方差比同步改善 | 真实预测改善 |
| 只对 zero A 显著，对自然错误 A 不显著 | 极端缺失敏感，不能外推为自然选择性 |

任何单项都不独立构成最终裁决。

## Smoke 与复用

先用至少 2 条样本跑完整五 A、五 timestep 和全部特征链路。smoke 必须逐条验证两个模型的：

- sample identity；
- `ref_len`；
- B latent、noise、target；
- B text、MIDI、CKA target；
- 五种 A latent；
- timestep；
- 所有条件 hash。

smoke 通过后才运行冻结全量 eval。

同一 VAE、数据语义、H/GAME 条件接口下，stimulus cache 可跨 checkpoint 和模型深度直接复用。
以下变化要求重建 cache：

- VAE 或 latent frame rate 改变；
- eval 数据或 B 划分改变；
- H placement、MIDI teacher 或 CKA target 改变；
- A 候选池、裁剪策略或 `ref_len` 策略改变。

只改变 checkpoint、模型深度或训练步数时不得重建 stimulus。EMA 是 primary；online 只能作为补充
并单独标记。

## 当前基准

HighLR 12k vs M600-D 12k 已形成首个完整 `FlowA-CF` 基准：201 条、五 A、五 timestep，跨模型
条件 hash mismatch 为 0。该结果证明两模型都使用 A，M600-D 的低 Loss 不是进一步平均化，但
自然错误 A 的选择性没有普遍增强。

30k 已直接复用同一 stimulus cache 完成 HighLR/M600-D/V4PH 严格 cohort。M600-D 的小幅 Loss
改善伴随 R2、相关和斜率改善，方差比没有收缩；错误 A 的 ΔFlowB 与 B 声学响应略增，困难样本
Q5 收益大于 Q1，但 near/far 标定没有改善。V4fg 10k 因绑定 285k VAE、SOFA/SOME 条件链，使用
独立缓存，只作跨语义参考。12k 结果见 `M600-D平均化与A区影响诊断.md`，30k 结果见
`FlowA-CF四模型执行记录.md`。

















