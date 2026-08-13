# PH-HighLR-LCF 严格对照实验

## 研究问题

只回答一个问题：在已完成的 `V4PH-30K-HIGHLR` 完整训练谱系上，将低噪声端 velocity regression
替换为 Local Contrastive Flow，是否能改善低噪声表示与最终歌声，同时不损害中高噪声学习、H/P
结构控制和既有音色。

现有 HighLR 作为冻结 control。LCF 分支不从 HighLR final continuation，而是从同一个历史 P500
重新开始 Phase B 30k；否则 optimizer、EMA、数据游标和已学习表示都会泄漏 control 结果。

## Control 冻结

| 项目 | 冻结值 |
|---|---|
| Phase A | `v4ph_phase_a/step_000500_final.pt` |
| Phase A SHA256: redacted`d82eb26f31429c9ed52f5ad22c95162fa3f3dc4c60f15d9114cc2616bbd772ac` |
| train | private corpus count 条、65.63h；SHA `b2560dc62eace7e294bce800e99a2669973fd3bceed42003c8033e7d8cb24467` |
| eval | 固定 201 条；SHA `befbf1341292f0c26707be6771178b381689c0de83682853e700bd9e1f5f38cc` |
| placement / MIDI | H phone/PUL + GAME medium K4，与 HighLR 相同 |
| seed / eval seed | `42 / 1042` |
| GPU / batch | 4 卡；physical `1/GPU`；grad accumulation 4；effective batch 16 |
| LR | peak `1.4e-5`；warmup 500；保持至 global step 24k；6k cosine 到 0 |
| step | 30k |
| dropout / CKA / FlowB | `.15 / .7 / 2.0` |
| EMA | 与 HighLR 相同：beta `.995`，step 0 fresh |

训练数据、sampler、noise、CFG/dropout RNG、优化器、裁剪、EMA、eval cadence、推理与听评输入不改。
服务器空间预检仅余约 105 GiB，因此 formal checkpoint 保留 cadence 改为 10k；它只改变产物写盘时点，
不参与 forward/backward、RNG 或 optimizer 更新。

## LCF 单变量

YMSP 使用 `x_t=(1-t)noise+t*x1`，因此论文的 `t<T_min` 映射为本项目的高 t 区间。

| 项目 | 冻结值 |
|---|---|
| 激活区间 | `t >= .95` |
| 非激活区间 | 保持原 `FlowA + 2*FlowB + .7*CKA` |
| 激活区间 | FlowA/FlowB 乘 0；保留 `.7*CKA`；加入 LCF |
| hidden | zero-based block index 15，即 22 blocks 的第 16 个输出 |
| feature | 只取 B 区，时间均值后 L2 normalize |
| positive | 同 `x1/noise/condition/CFG dropout mask` 在边界 `t=.95` 的 hidden；detach |
| negatives | 当前四卡物理 microbatch 的 query hidden；detach；分母包含 anchor 自身 detached copy |
| temperature / weight | `.5 / 1.0`，沿用论文默认 |
| Flow 归一化 | 四卡中只对非 LCF 样本取 global mean |
| LCF 归一化 | 四卡中只对 active anchors 取 global mean |
| eval | 冻结为原标准 FM+CKA eval，不用 LCF loss 改写健康指标 |

阈值来自诊断中 `.90-.975` 的明确弯折区；`.95` 以上约占真实训练 timestep 分布的 2.56%。层位不搜索：
论文在 12-layer DiT 使用第 8 层，本项目 index 15 属相近中后段；诊断中该层 `.99` 对 `.8` 的 hidden
gradient cosine 为 `.142`，病理信号明确，而最终层仍为 `.823`。

physical batch 固定为每卡 1，不能把 grad accumulation 假装成同时存在的 contrastive batch。
首轮 negatives 因此诚实地只有四卡当前 microbatch 的 4 条；不加入 memory bank 或 queue，避免新增状态与 resume 混杂。

## RNG 契约

DiT 内有 `dropout=0.1`。positive 多一次 forward 若不处理，会推进 CUDA RNG，并使 LCF 首次激活后的
全部训练序列偏离 control。实现冻结以下策略：

1. query forward 前保存 Python、NumPy、CPU Torch 和当前 CUDA RNG；
2. query forward 后保存 control 本应到达的 RNG；
3. positive 恢复 query 前 RNG，因此复用完全相同的 dropout mask；
4. positive 完成后恢复 query 后 RNG，因此后续 noise/dropout 序列不被额外 forward 推进。

LCF-off 新入口必须与历史 HighLR step-10 的 model、EMA、optimizer、scheduler、sampler cursor、RNG
和共同 running metrics bit-exact，作为该契约的直接门禁。

## 执行门禁

1. CPU/pure Torch：pool、detach、InfoNCE 公式、active-rank 归一化；
2. LCF-off 10-step 对历史 HighLR 10-step 等价；
3. LCF continuous-10 对 5+resume-10 全 checkpoint section bit-exact；
4. fresh 50-step 至少出现一个 anchor，LCF/positive/negative distance 全有限；
5. checkpoint 绑定 LCF schema、训练代码 hash、P500 transition、optimizer、EMA、rank RNG 与累计
   flow/anchor partition；
6. 仅在以上全部通过后创建 formal 30k 输出。

## 评价

训练健康指标：冻结 paired eval 的 Loss/FlowA/FlowB/CKA；它们不能单独裁决 LCF。

机制指标：复用 PH/fg 诊断的 endpoint gain、NMSE、hidden noise Jacobian proxy、gradient cosine 与相对 burden，
对 HighLR control 和 LCF 的相同步数/样本做配对比较。

最终门禁：同一 27 组、同 CFG 3/1、同推理代码生成人耳样本；用户已要求不重新打包现有 HighLR，
因此交付为明确标注的 LCF 单模型包，由用户与既有 HighLR 包直接比较音色、自然度、咬字时间、极高音、
宽带噪音与唱法。LCF 只有在机制指标按预期改善、中高噪声不回归且人耳出现可重复净收益时才通过。

总 loss 更低不是通过条件。

## 执行结果

纯数学、LCF-off 历史 control bit-exact、连续/续训 bit-exact 和 50-step 激活门禁全部通过。正式
30k 从冻结 P500 fresh 启动并正常结束；日志无 traceback、NaN、OOM 或 NCCL 错误：

```text
final checkpoint: plus_ja_sft_v4ph_30k_highlr_lcf/step_030000_final.pt
SHA256: redacted
run_state / step / EMA: complete / 30000 / 30000
final paired eval: Loss 2.0849; FlowA .2764; FlowB .8700; CKA .0977
LCF anchors: 12,357 / 480,000 = 2.574375%
mean LCF / positive distance / negative distance: .069791 / .001421 / 2.283434
```

final 独立审计确认 checkpoint schema、训练代码 SHA、P500 transition、optimizer、scheduler、EMA、
四 rank 累计 flow/anchor partition 和全部 tensor 有限性。总 loss 只表示训练健康，不裁决音频收益。

## 实名听评包

固定 27 组以 32 sampling steps、seed 42、CFG 3/1 各生成一次，共 54 条 LCF WAV；未重新生成、复制或
打包 HighLR。smoke 和全量审计均通过：

```text
WAV / placement audit: 54 / 54
input condition mismatch: 0
CFG MIDI mismatch: 0
CFG byte-identical pair: 0
max duration delta: 21.406 ms
minimum RMS: .075365
```

实名 ZIP 内 manifest 的 54 条 WAV 逐文件 SHA256: redacted
下载并复算 SHA，和服务器上传前一致：

```text
cloud: ${CLOUD_ARTIFACT}
bytes: 192,376,341
SHA256: redacted
```

cloud artifact

```text
root: ${LOCAL_EXPORT_PATH}
CFG 3: ${LOCAL_EXPORT_PATH}
CFG 1: ${LOCAL_EXPORT_PATH}
```

本地 ZIP 与 sidecar SHA 一致；接入后复核 CFG 3/1 各 27 条、54 个 placement audit、根 manifest
模型条目 1 个、评分表新增 54 行，WAV missing 0、SHA mismatch 0。独立实名包目录只作为cloud artifact
接入来源，不是 canonical 听评位置。

## 听评裁决

用户将实名 LCF 30k 输出与既有 HighLR 包直接比较后，结论为：**无明显改善**。

因此本配方没有满足“人耳出现可重复净收益”的预注册通过条件，#45 的首轮
`V4PH-30K-HIGHLR-LCF` 严格对照判定为未通过，不进入 V5 模型配方。训练稳定、标准 eval 下降和
contrastive distance 正常只能证明实现工作，不能覆盖听评裁决。

这一负结果不否定 YMSP 存在共享的低噪声端点欠响应；它说明当前 `t>=.95 / block 15 / tau=.5 /
lambda=1 / four-negative` LCF 方案没有把该局部优化变化转化为可辨认的音频收益。低噪声机制后诊断
若继续，只用于区分“LCF 没修到机制”与“修到机制但听感无收益”，不再作为翻转本轮产品裁决的依据。

















