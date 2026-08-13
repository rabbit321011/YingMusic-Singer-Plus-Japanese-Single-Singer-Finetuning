# A 区并行 COND 实验

## 1. 研究问题

V4PH 将同一条target singer训练音频切成前段 A 与后段 B，并把二者按 `[A | B]` 放在同一模型时间轴。
A 的真实 latent 位于 `cond` 前缀，歌词与 GAME MIDI_P 只作用于 B。该结构提供稳定的声学参考，
但存在两个待解决问题：

1. A 中的沙声、气声、极高音或破音边缘等局部特征可能被模型当成全局参考；B 出现同类特征时，
   可能被错误强化为毛刺、哑嗓或崩坏特征；
2. A 占用 DiT 的序列位置与最大上下文预算，减少同一次推理可生成的 B 时长。

V4IPH 通过完全取消 A 解决了第二项，也切断了第一项传播路径，但头戴耳机听评与客观统计均显示
`V4PH -> V4IPH` 是明显的感知退化。独立工程审计未发现 V4IPH 特有接线错误，因此不能默认依靠
更多训练步数或更大 LR 修复。

K 检验一个中间假设：**保留同源 A 的原始声学 latent，但令 A 只占并行 `cond` 通道，模型时间轴、
歌词、旋律、Flow target 与最终输出全部只包含 B。**

本实验首先回答结构问题，不同时检验 285k VAE、INS、A 自动筛选、新的 CFG 标量或新的 ODE solver。

## 2. 与既有分支的关系

| 项目 | V4PH | V4IPH | V4IjPH | V4KPH |
|---|---|---|---|---|
| 模型时间轴 | `[A | B]` | `B` | `B` | `B` |
| A/B 数据切分 | 同一样本非空切分 | 不切 A | 不切 A | 同 V4PH 非空切分 |
| `cond` 来源 | A raw latent，位于 A 时间段 | 全零 | 独立 R 的 INS 全局特征 | 同源 A raw latent，并行放入 B 长度的 `cond` |
| `model_ref_len` | `len(A)` | `0` | `0` | `0` |
| H/PUL 与 MIDI_P | 仅 B | 完整时间轴 | 完整时间轴 | 完整 B |
| Flow target | `[A | B]` | 完整 B | 完整 B | 完整 B |
| A 占生成上下文 | 是 | 否 | 否 | 否 |
| 是否保留细粒度声学锚点 | 是 | 否 | 否 | 是 |

K 不是 IJPH 的 adapter 变体。IJPH 回答“全局唱法表示是否足够”，K 回答“原始声学 A 是否可以脱离
生成时间轴继续发挥锚定作用”。两者的结果不能用同一个 `cond` 非零事实合并解释。

## 3. 两类长度必须分离

K 禁止继续用一个 `ref_len` 同时表示数据切点和模型 prompt 长度，必须使用三个独立字段：

```text
source_split_frame = S       # 在完整 target latent 上切 A/B，S > 0
model_ref_len = 0            # 模型没有 A prompt 时间段
cond_valid_frames = L        # 实际装入 cond 的 A latent 帧数
```

给定一次完整 VAE 编码：

```text
z_full: [1, T, 64]
z_A   = z_full[:, :S, :]
z_B   = z_full[:, S:, :]
T_A   = S
T_B   = T - S
```

A 与 B 必须来自同一次完整 waveform 编码后再按 latent frame 切分。首版禁止分别编码 A/B，避免
额外引入 VAE receptive field、边界 padding 或重采样差异。

`S` 沿用 V4PH 已冻结的随机切点与 phrase-boundary snapping 规则。K 不把 B 的任何帧复制进 A，
并必须断言 `z_A` 与 `z_B` 的源 frame 区间不重叠。

## 4. K1 张量契约

### 4.1 B 独占模型时间轴

K1 的 DiT 序列长度固定为 `T_B`：

```text
x_0 / flow target: z_B                    [1, T_B, 64]
x_t:                noisy/interpolated B  [1, T_B, 64]
cond_K:             parallel A cond       [1, T_B, 64]
text_K:             B-only H/PUL           [1, T_B]
midi_K:             B-only GAME MIDI_P     [1, T_B, 128]
p_classes_K:        B-only GAME classes    [1, T_B]
model_ref_len:      0
```

K 的 B 控制不得重新估算。对同一完整样本、同一随机切点 `S`，先执行冻结的 V4PH renderer 与 GAME
时间映射，再直接取得 B 后缀：

```text
text_K      == text_V4PH[:, S:]
midi_K      == midi_full[:, S:]
p_classes_K == p_classes_full[:, S:]
z_B         == z_full[:, S:, :]
```

上述四项必须逐值相等。这样 K 改变的只有 A 与 B 的放置关系，不改变 B 的歌词、旋律、latent 域
或对齐任务。

### 4.2 A 放入并行 cond

当前 DiT 的 `InputEmbedding` 在每个位置拼接 `x_t[i]`、`cond[i]`、text embedding 与 MIDI，
因此 `cond` 必须与 B 等长。K1 使用无新参数、无插值的确定性前缀布局：

```text
cond_K = zeros_like(z_B)
L = min(T_A, T_B)
cond_K[:, :L, :] = z_A[:, T_A-L:, :]
```

- 当 `T_A <= T_B` 时使用完整 A，并在其后补零；
- 当 `T_A > T_B` 时只使用 A 的末 `T_B` 帧，使最接近 A/B 原切点的声学状态得到保留；
- 禁止时间插值、加速、减速、循环平铺、均值池化或把 B 补入空余 cond；
- checkpoint/provenance 必须记录 `T_A/T_B/L`、是否裁剪、裁剪方向和源 frame 区间；
- 推理使用完全相同的裁剪与前缀放置规则。

K1 不增加 `cond_valid_mask` 或 reference type embedding。非零 A latent 与尾部精确零共同构成现有
输入结构可消费的最小布局；若该布局失败，不允许在同一正式 run 中临时增加新结构。

### 4.3 不存在 FlowA

虽然数据侧存在 A，A 不属于 flow target，也不参与 noising 或 ODE。K 的目标函数为：

```text
0 + 2 * FlowB(z_B full timeline) + 0.7 * CKA(B full timeline)
```

`FlowA` 数学上不存在并记录为精确 `0`。禁止对 `cond_K` 计算 flow reconstruction loss；否则会
重新把参考通道变成生成目标。

## 5. 初始化与冻结配方

K 是新的训练布局，不应把 IJPH 的 continuation 当成正式 K 结论。正式 K1 从与 V4PH/V4IPH
phase B 相同的起点独立训练：

```text
init: Official base + 已裁决的 V4PH P500
trainable: Official DiT + H/PUL + P embedding
optimizer/scheduler/EMA: fresh
LR: 1.4e-5
warmup: 500
hold: 12000
total steps: 30000
VAE: official frozen VAE
batch_size: 1
world_size: 4
grad_accum: 4
effective batch: 16
seed/eval seed: 42 / 1042
loss: 2 * FlowB + 0.7 * CKA
```

继续冻结以下不变量：

- private corpus count train + 201 eval H manifest 与 SHA256: redacted
- GAME medium K=4 cache、class 定义、P500 补核与 PAD/REST 语义；
- H/PUL token、renderer、fallback 与 phrase-boundary 规则；
- FlowB/CKA 权重、确定性 sampler、显式 DDP 梯度同步与 exact resume；
- 训练数据顺序、checkpoint 周期和固定 paired Eval。

若只为低成本探针从 V4IPH step 8000 continuation，必须另记为 K continuation pilot，不能与正式
K1 混名，也不能用其结果宣称 K 与 PH/IPH 的单变量比较成立。

## 6. dropout 与 CFG

### 6.1 训练

沿用既有独立 dropout 概率：

```text
audio cond dropout: 0.30
text dropout:       0.15
MIDI dropout:       0.30
```

audio cond dropout 命中时，`cond_K` 必须全张量精确为零，但 B 的 `x_t/target`、长度与来源不得改变。
这提供 K-null 路径，并避免模型只能在永远非空的 A 下工作。

不得从 B 提取替代 A，不得在 A dropout 后回退到 INS，也不得联动改变 text/MIDI dropout 抽样。

### 6.2 推理

推理参考 R 进入与训练 A 相同的 official VAE：

1. R 以冻结预处理编码为 `z_R`；
2. 目标 B 长度只由歌词、旋律与既有 renderer 决定；
3. `cond_K` 按 `min(T_R,T_B)` 取 R 尾部并放入 cond 前缀；
4. `ref_len=0`，H/PUL 与 GAME MIDI_P 覆盖完整 B；
5. ODE 只生成 B，输出不裁掉 A，也不按 A 长度平移歌词或 MIDI。

K 禁止把 R latent 直接传给官方 `model.sample(cond=...)`。该入口会按 `cond_seq_len` 把输入解释为
占用时间轴的 acoustic prompt，并据此建立 prompt mask、扩展 duration 和清零 prompt 区 MIDI，
会悄悄退回 PH 语义。K 必须使用独立的 full-timeline ODE 入口：先在入口外构造完整
`cond_K [1,T_B,64]`，再以 `ref_len=0` 直接送入 DiT；该行为需要真实推理门禁覆盖。

CFG 的 conditional 与 content-unconditional 两支必须保留同一个 `cond_K`，只对歌词/MIDI 内容差
执行既有 guidance，即 acoustic-reference guidance 固定为 1。K-null 时 `cond_K` 全零。

首轮继续使用已冻结的 CFG 1、Euler32 与相同 seed；不得把 CFG/solver 调参并入 K 结构判断。

## 7. 主要风险与可证伪信号

### 7.1 伪逐帧对齐

K1 虽不在时间轴拼接 A/B，但当前输入层会在位置 `i` 融合 `x_B[i]` 与 `cond_A[i]`。两者没有
语义时间对齐，模型必须通过后续 self-attention 学会把 A 当作参考而非同帧目标。

必须分别统计 B 的 `0..L` 与 `L..T_B`：

- latent `maxAbs`、时间差分 std 与极值尾部；
- 4--12 kHz、8--20 kHz 能量与谱平坦度；
- 周期相关性、毛刺听感、咬字与发虚。

若退化主要集中在 B 的前 `L` 帧，并随 L 改变边界位置，则判定 K1 的 framewise layout 失败。
下一步应另立 K2 reference encoder/cross-attention 设计，不得通过增加步数或 LR 掩盖。

### 7.2 A 局部特征污染仍可能存在

K 只消除了 A 的序列占用，没有承诺消除 A 内容对 B 的条件影响。固定同一个 B，至少使用三类 R：

- 干净、普通发声；
- 明显气声或沙声；
- 极高音或破音边缘。

若只有特定 R 稳定触发 B 同类位置的毛刺、方波边缘或崩坏，支持“A 局部特征被错误强化”的假设。
该结果应转向 A 筛选、压缩或独立 reference encoder，而不是宣称 K 整体恢复成功。

### 7.3 训练/推理参考差异

训练 A 与 B 来自同一条音频，推理 R 可以来自独立target singer样本。首轮保持这一点与 V4PH 一致，以隔离
layout 变量；但必须额外做同源 R 与跨样本 R 对照。若只在同源 R 有效，则 K 学到的可能是局部连续性，
而不是可泛化的音色锚点。

### 7.4 loss 不代表感知成功

V4IPH 已证明 FlowB/CKA 可健康收敛而感知假设失败。K 的 loss 只作为训练健康指标；不得因 loss
低于 PH/IPH、没有反弹或尚未过拟合而自动增加步数、提高 LR 或宣布成功。

## 8. 实现隔离与 checkpoint 契约

不得修改 V4PH、V4IPH 或 V4IjPH 已冻结入口的行为与文件哈希。K 使用独立训练、contract、renderer、
推理和 auditor 文件。建议责任边界为：

| 文件 | 责任 |
|---|---|
| `train_v4kph.py` | 同源 A/B 切分、B-only flow、DDP、checkpoint 与 exact resume |
| `v4kph_contract.py` | split/model ref 长度分离、parallel cond 与 loss 纯函数契约 |
| `placement_v4kph.py` | 证明 K 控制逐值等于 V4PH B 后缀 |
| `v4kph_sampling.py` | R latent 到 parallel cond、B-only ODE 与 CFG 契约 |
| `audit_v4kph_checkpoint.py` | schema、起点、布局、optimizer/EMA/RNG/runtime SHA 审计 |

正式 schema 冻结前暂定：

```text
checkpoint schema: v4kph_training_checkpoint_v1
metadata key:      v4kph_training
reference mode:    parallel_a_cond_all_b
cond layout:       prefix_tail_crop_zero_pad
model_ref_len:     0
```

K checkpoint 禁止被 V4PH、V4IPH 或 V4IjPH 入口直接 resume；反向亦然。正式实现时 checkpoint 必须
保存并由独立 auditor 核对 source/runtime SHA、P500、official VAE、GAME cache、H renderer、A/B split
RNG、四 rank RNG、sampler cursor、optimizer、scheduler、EMA 和运行状态。

## 9. 强制门禁

### Stage 0：纯函数与回归

1. `S > 0`、`T_A > 0`、`T_B > 0`、`model_ref_len == 0`；
2. A/B 源 frame 区间严格不重叠且并集等于完整 latent；
3. `x_0 == z_full[:, S:]`；
4. K 的 text、MIDI_P、p_classes 逐值等于 V4PH 同切点的 B 后缀；
5. `cond_K` shape 等于 `z_B`，有效前缀等于指定 A 尾部，其余精确为零；
6. A dropout 后 cond 全零，未 dropout 时真实 batch cond 必须非零；
7. FlowA 精确为 0，FlowB/CKA/总 loss 有限；
8. V4PH/H、V4IPH 与 V4IjPH 原测试全部回归通过。

### Stage 1：真实单卡前后向

使用正式 manifest、真实 waveform、official VAE、GAME cache 与 P500 完成至少两个 update，覆盖
非 dropout 和 audio-cond dropout 两条路径。验证 `InputEmbedding` 中 cond 对应投影列梯度非零且
实际更新、B-only 长度与 provenance 正确；不得用 mock latent 代替。

### Stage 2：四卡 active smoke 与 checkpoint audit

使用正式 30k scheduler 跑四卡 10-step，要求：

- 四 rank 初始/最终参数指纹一致；
- 至少一个真实 batch 激活非零 A cond；
- split RNG、dropout RNG、数据游标与指标 schema 完整；
- checkpoint 独立 audit 通过；
- 连续 10 step 与 5 + resume 5 的全部 section bit-exact。

### Stage 3：真实推理探针

至少验证：

- 同 R、同 B、同 seed 输出 bit deterministic；
- 两个不同 R 的 cond 与输出均不同；
- K-null cond 精确为零且输出有效；
- R 不改变 B 的 text/MIDI/目标长度；
- 输出只有 B，不包含、裁切或重复 R；
- 入口未调用会按 `cond_seq_len` 建立 prompt mask 的官方 prompt sampling 路径；
- provenance 可还原 R waveform、VAE、裁剪区间、cond SHA 与采样参数。

### Stage 4：正式训练授权

Stage 0--3 全部通过、IJPH 固定 checkpoint 结果完成复核且用户明确授权后，才允许启动正式四卡
K1。任一门禁失败均停止，不允许以 loss 正常、输出可播放或训练速度正常替代工程证据。

## 10. checkpoint 与听评设计

正式训练至少保存 `10k / 14k / 18k / 22k / 24k / 30k`，并与同步 V4PH、V4IPH 及 IJPH
候选使用同一 B、R、seed、CFG、solver、VAE 和响度协议生成。

首轮每个 B 至少包含：

```text
V4PH + R
V4IPH
V4IjPH + R / null INS
V4KPH + R / null A cond
真实 B 与 official VAE 纯重建
```

音频必须导出无削波 PCM24，并制作等 RMS 盲包。人工听评优先记录：

- 发声区毛刺、破音边缘、哑嗓沙声和近似方波的不自然纹理；
- 周期稳定性、声音虚实、咬字与 17--19 秒等已知敏感段；
- target singer身份、A 特征迁移、极高音、跑调和第一段开头；
- K 是否把异常集中到 B 的 `0..L` 区域；
- 在相同最大序列限制下，K 相对 PH 实际增加的可生成 B 时长。

客观分析继续记录 latent `std/maxAbs/deltaStd/abs>3`、分频能量、谱平坦度、周期相关性、F0、
Whisper 内容与 Whisper+SOFA 错槽；这些指标用于定位，不单独裁决听感。

## 11. 裁决矩阵

| 结果 | 解释 | 下一步 |
|---|---|---|
| IJPH 与 K 都改善 | 全局唱法和声学 A 均可补偿无 A 退化 | 比较污染风险、上下文与稳定性，选更小必要条件 |
| IJPH 无效、K 改善 | 缺失的是细粒度声学锚点，不只是全局风格 | K 晋级；继续检验不同 A 的污染与跨样本泛化 |
| IJPH 改善、K 被特定 A 污染 | 原始 A 信息过细，INS 压缩具有正则化价值 | 优先 IJPH，或另立受控 A 压缩方案 |
| K 仅前 `L` 帧退化 | 逐帧 cond 与 B 形成伪对齐 | 停止 K1，另立 K2 独立 reference memory |
| K 与 IPH 噪音相当 | raw A 并行 cond 未恢复稳定性 | 不增加 LR/步数；复核结构或 285k latent 域 |
| K 接近 PH 且无特定 A 崩坏 | 在不占 A 时间轴时保留声学锚点 | K 通过首轮感知门禁，进入长上下文与多 A 复核 |

K 的成功标准不是 loss 下降或 R 能改变输出，而是：相对 IPH 稳定降低发声区噪音且不牺牲咬字、
身份和旋律；相对 PH 明确释放 B 上下文，并且不同 A 不产生不可接受的特征相关崩坏。

## 12. 当前状态

- 本文只完成实验设计与门禁定义；
- 未创建训练/推理代码、cache、checkpoint 或服务器任务；
- 未授权抢占 IJPH 正式任务资源；
- 实现前仍需冻结正式文件名、schema、source checkpoint SHA 与听评样本清单。

















