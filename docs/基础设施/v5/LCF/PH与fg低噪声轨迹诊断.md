# PH 与 fg 低噪声轨迹诊断

## 问题

需要拆开判断四件事：

1. YMSP 的 flow 路径是否具有论文所述的低噪声病态几何；
2. 这种几何是否在现有模型里形成可测的端点欠响应、学习停滞和表示变化；
3. PH 与 fg 是否呈现不同病型，能否解释两者 loss 与音色判断的脱钩；
4. 现有证据是否足以把 LCF 直接加入后续模型。

诊断只读 checkpoint，不执行 optimizer step，不修改 checkpoint，也不训练 LCF。

## 时间方向

原论文写作约定是 clean data 位于 `t -> 0`。YMSP 使用相反方向：

```text
x_t = (1 - t) * noise + t * x1
v*  = x1 - noise
```

因此 YMSP 的低噪声端是 `t -> 1`。固定 `x1` 并替换 noise 时：

```text
delta(x_t) = (1 - t) * delta(noise)
delta(v*)  = -delta(noise)
target local gain = ||delta(v*)|| / ||delta(x_t)|| = 1 / (1 - t)
```

归档材料中沿用论文记号的 `t < T_min` 不能原样搬入 YMSP；对应实现必须改写为靠近 `t=1` 的区间。

训练采用 `t_shift=0.5`：`t=0.5u/(1-0.5u), u~Uniform(0,1)`。按该真实分布，
`t>=.95/.975/.99/.995` 的概率约为 `2.56%/1.27%/0.50%/0.25%`。端点样本少，但并非训练分布之外。

## 冻结契约

| 项目 | 契约 |
|---|---|
| PH checkpoint | 2k / 10k / 30k |
| fg checkpoint | 2k / 10k / 15k |
| 权重 | 仅 EMA |
| 样本 | 每点 32 条；从既有 201 条审计缓存按 B 长度均匀选取 |
| 时间点 | `.50, .80, .90, .95, .975, .99, .995` |
| hidden 层 | `4, 10, 15, 21` |
| 梯度样本 | 每点固定 8 条，与 checkpoint 配对 |
| 主要变体 | paired A；同一样本 alternate noise |
| condition 消融 | 删除 text + MIDI，保留 audio cond |
| bootstrap | 10,000 次，以 `sample_id` 为重采样单位 |

PH 与 fg 不共享 VAE latent、placement、MIDI 表示、CKA target 和训练阶段。跨家族不比较绝对 MSE、
绝对 hidden 梯度或 hidden 幅值；只比较 NMSE、R2、相关、回归斜率、标准差比、局部 gain 比例和曲线形状。
因果证据只来自各家族内部的配对 checkpoint 演化。

hidden 的 alternate-noise 差值另外除以实际 `delta(x_t)`，形成有限差分 Jacobian 代理；否则
`1-t` 自动缩小会被误读为 hidden 不再响应噪声。梯度负担也除以同模型、同样本、同层的 `t=.8`
值。这里测到的是 hidden 接收的反传信号，不冒充完整参数梯度 Hessian。

## 审计

- 六个 run 全部 `status=ok`，checkpoint step 与预期一致，权重源均为 `ema_model_state_dict`；
- 每个 run 产生 224 条 forward、896 条 hidden、224 条 gradient；合计 `1344/5376/1344`；
- 每个 run 最大 reserved VRAM 约 `3.04 GiB`；无训练和 checkpoint 写入；
- 远端 49 个产物同步到本机后逐文件 SHA256: redacted
  `a25d324aa29f2c8a70915368322264fc8eb5d486d8a949bf0d134fd394e89f21`；
- probe SHA256: redacted`09712e892711270c20117e8f705debea842ed09d2d0fdca4afc778b56e7b4517`；
- 所有分析产物、输入 hash、随机种子和命令由 `analysis_manifest.json` 托管。

原始产物与完整置信区间见：

- [实验说明](../../../../experiments/low_noise_lcf_diagnostic_20260806/README.md)
- [完整统计报告](../../../../experiments/low_noise_lcf_diagnostic_20260806/results/analysis/diagnostic_report.md)
- [分析清单](../../../../experiments/low_noise_lcf_diagnostic_20260806/results/analysis/analysis_manifest.json)

## 结果一：端点欠响应确定存在

理论 target gain 与模型 predicted gain 的比值如下：

| 模型 | t=.50 | t=.80 | t=.95 | t=.99 | t=.995 |
|---|---:|---:|---:|---:|---:|
| PH 30k | 74.3% | 37.0% | 5.0% | 0.83% | 0.43% |
| fg 10k | 67.7% | 24.6% | 3.6% | 0.88% | 0.45% |

`t=.99` 的理论 gain 是 100，PH/fg 实际 predicted gain 约为 `0.826/0.882`。两者都几乎不追踪
低噪声输入里被压缩的 noise 身份。这不是绝对 loss 尺度造成的假象，而是同一样本、同 latent 空间内的无量纲响应比。

欠响应从 `.90-.95` 开始明显弯折，到 `.975` 后 PH 与 fg 几乎重合。狭义的低噪声病态几何和模型欠响应在两家族都成立。

## 结果二：端点学习明显停滞

| 家族轨迹 | B NMSE at t=.8 | B NMSE at t=.99 | gain fraction at t=.8 | gain fraction at t=.99 |
|---|---:|---:|---:|---:|
| PH 2k -> 30k | `.528 -> .447` | `.490 -> .478` | `.233 -> .370` | `.0082 -> .0083` |
| fg 2k -> 15k | `.576 -> .553` | `.517 -> .515` | `.212 -> .249` | `.0087 -> .0088` |

模型继续训练时，中噪声区稳定进步，`.99` 端点只得到很小的 NMSE 改善，noise gain 基本不动。
以 `.8` 为基准的 early-to-late difference-in-differences：

- PH `.99` B NMSE：`+0.0687 [0.0651, 0.0723]`；
- fg `.99` B NMSE：`+0.0216 [0.0200, 0.0232]`；
- PH `.99` gain fraction：`-0.1370 [-0.1452, -0.1288]`；
- fg `.99` gain fraction：`-0.0378 [-0.0404, -0.0352]`。

这些区间均为样本配对 95% bootstrap CI。正的 NMSE DiD 不表示端点绝对变差，而表示端点远远没有跟上 `.8` 的学习速度。

## 结果三：有反传方向转向，没有端点幅度统治

`.99` 相对 `.8` 的 hidden B 梯度如下：

| 模型 | L4 cosine / burden | L10 cosine / burden | L15 cosine / burden | L21 cosine / burden |
|---|---:|---:|---:|---:|
| PH 30k | `-.027 / .652` | `.144 / .316` | `.142 / .401` | `.823 / .603` |
| fg 10k | `.033 / .746` | `.153 / .349` | `.216 / .420` | `.864 / .595` |

`cosine` 接近 0 表示浅层和中层在 `.99` 接收到与 `.8` 近乎正交的反传方向；深层仍大体同向。
但经过真实 timestep density 加权后，四层 `.99` 的逐点梯度负担都小于 `.8`，没有任何一层超过 1。
再考虑 `.99` 以上只有约 0.5% 的训练概率，当前证据不支持“端点梯度在总优化量上压倒其他区间”。

这支持低噪声区发出不同且难以兼容的训练信号，但不支持它以梯度幅度统治训练。

## 结果四：没有一致的噪声 Jacobian 重分配

端点相对 `.8` 的 noise finite-difference Jacobian 代理在 early-to-late 期间的 DiD：

| 家族 | L4 | L10 | L15 | L21 |
|---|---:|---:|---:|---:|
| PH | `-.0415` | `+.1029` | `+.0714` | `-.1095` |
| fg | `-.0117` | `+.0262` | `+.0102` | `-.0325` |

所有列的配对 CI 均不跨 0，但方向在层间相反。PH 的层间重组幅度明显大于 fg，符合 fresh H/P joint
training 比 fg 低 LR 适配改动更大的训练事实；它也可能包含 H placement、GAME MIDI 和 official VAE 的任务学习。

因此可以确认 hidden 发生了 checkpoint-linked reorganization，不能确认论文所述“容量一致地转向 noise 方向”。
condition/noise-Jacobian 代理也没有给出足以排除结构条件学习混杂的单向因果链。

## PH 与 fg 的 loss 是否本质不同

是，但准确说法不是“两个数字都叫 FlowB，所以谁低谁好”。

- PH 30k 的 B NMSE 在 `.8/.99` 为 `.447/.478`；fg 10k 为 `.554/.515`。PH 对自己 target 的无量纲拟合反而更好；
- 用户固定组听评已经裁决 PH 音色远差于 fg；
- 两者在 `.975` 以后的 gain fraction、梯度转向和相对负担却高度相似。

这构成直接反例：更好的无量纲 velocity fit 没有转化为更好的跨家族音色。PH/fg 的绝对 loss 更不能用于音色排名。
差异首先属于 latent、VAE、条件和训练谱系，不是 PH 独有的低噪声病型。

## 裁决

| 命题 | 裁决 |
|---|---|
| YMSP 具有低噪声病态几何 | 确认 |
| PH 与 fg 都存在端点欠响应 | 确认 |
| 端点学习相对中噪声区停滞 | 确认 |
| 低噪声端发出显著转向的 hidden 反传信号 | 确认 |
| 端点梯度幅度统治总体训练 | 不支持 |
| 出现跨层一致的 noise Jacobian 容量重分配 | 不支持；当前为混合方向 |
| 低噪声病解释 PH 相对 fg 的音色差距 | 不支持；两者端点病型近似 |
| LCF 能改善 YMSP 音频 | 未实验，不能裁决 |

最精确的总结是：**Low-Noise Pathology 在 YMSP 中是共享、真实但尚未被证明是当前音色主因的局部优化问题。**

## 对 #45 的影响

#45 不取消，但从“直接进入模型配方”收缩为“单基线受控实验候选”。诊断已经表明 PH 与 fg 的端点病型
高度相似，因此没有理由为两个家族分别铺开阈值、层位和权重的组合矩阵。

若继续，下一步只允许一个代表实际后续配方的 baseline 做严格 A/B：同初始化、同数据次序、同 step、同 seed，
标准 FM 对照 LCF。先冻结一个由 `.90-.975` 弯折区推导的阈值和一个中间层，不做网格搜索。通过条件必须同时包含：

- endpoint gain/方向/表示代理改善；
- 中高噪声区不回归；
- checkpoint-matched 人耳音色、自然度和咬字无退化且有可重复收益。

总 loss 下降不是通过条件。现有诊断本身不授权训练，也不把 LCF 视为 PH 音色修复器。


















