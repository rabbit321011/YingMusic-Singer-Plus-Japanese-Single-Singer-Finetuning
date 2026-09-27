# V5 版本

V5 是当前版本。V4 达到 FlowB≈0.90 平台后，V5 先从两个方向并发推进：
对齐精度（SOFA）和 VAE 音质（VAE）；两线随后通过 V4g/V4fg 汇合并完成 T1 盲听选型。

**SOFA、VAE、V4g、V4fg 已完成。随机初始化感知门禁已失败；P、S、L、H 与 M 新路线已建立。**

**后续进展：V5-Pg20 已完成；从其 EMA 启动的 V5-PgO 8K 训练及
V5PgOV 300K decoder 适配均已完成。V5PgOV 是当前 V5 最终候选，
但 PgO 仍保留为稳定默认/回滚基线。** 下文的早期路线记录保留
其原有实验时点，最新结论以对应分支文档为准。

## 当前汇合结论

- V5-P 已建立为首个正式代际集成分支。训练谱系为 Official checkpoint → fresh P-only 500-step
  → official VAE 下 H + GAME-P + L-DEDUP 40k → 285k online VAE 下 g 适配 10k。Phase B 冻结
  0--2k warmup 至 `1.4e-5`、2k--28k cosine 缓降至 `1e-5`、12k cosine decay 至 40k 的 0；L 使用
  `KEEP_LONG_DEDUP_SHORT + NATURAL_RECORD`。首轮只执行 P；未来 V5-S 表示 continuous SOME
  对照，不表示既有 SVC 自克隆 `S/` 分支。推理接口允许 `B <= 60s`，边缘质量不承诺。当前
  Long 使用整段 Whisper 后重跑整段 SOFA，并复审历史阈值下的实际通过率；Long-H 冻结沿用
  既有 V4H renderer，允许稀有 token 覆盖/截断。随后才进入 GAME cache、去重 manifest 和训练
  门禁。Phase A 与 Phase B 40k 均已完成；2K--40K EMA/raw、CFG 1.0 的实名轨迹听评裁决
  `40K` 基本最佳，raw/EMA 主体质量接近而 EMA 稳定性略优，因此 non-g 默认权重冻结为
  `40K EMA`。Phase C 已完成本机可执行实现：step-0 transition、每 1k 保存、每 500 step
  Short/Long Eval，LR 为 250-step warmup、6000-step hold、3750-step cosine decay；服务器门禁与
  正式训练最初未在本条结论中自动授权。用户后续明确授权后，最长样本和四卡 exact-resume 门禁
  已通过；低 LR `v5pg_phase_c_10k` 已完成，final Short/Long FlowB 为 `0.9494/0.9911`，pipeline
  audit 与cloud storage上传通过，尚待轨迹听评。为检验历史 g 配方是否欠训练，又建立独立的
  `V5-Pg20-HLR07`：1k warmup 至 `9.8e-6`、1k--14k shoulder 至 `7e-6`、14k--20k decay
  至 0，四卡 exact-resume/step-0 audit 通过并已在物理 GPU set 正式运行。完整 40k ×0.7 路线
  保留实验合同但已于 2026-08-12 搁置，未授权且不得由当前 20k 直接续训冒充；20k 后只进入
  final audit 与轨迹听评。
- SOFA 解决了句级文本条件的可学习性，T1 解决了推理侧长、多句文本锚点缺失；二者没有解决全部美学和声学稳定性问题。
- Full-VAE 的最终选型仍是 285k online。
- 两线汇合后的 14-checkpoint T1 盲听中，**V4fg 10k + 285k online VAE** 是历史同源任务的最佳对照，不再等同当前主线。
- 当前共性问题：第一句从中段开始才趋稳；罕见轻柔音色可能出现全段雾感。
- reference tail `0 / 0.25 / 0.5s` 消融已完成：0s 对部分样本有效但不普适，暂不修改全局默认值。
- 轻柔音色 A/B 交换已完成：A15+B14、A14+B15 均不雾，只有 A15+B15 稳定雾且不轻柔；四组 285k VAE 纯重建全部正常。
- 已排除单区、285k VAE 和源录音自带雾感；当前仅能确认 sample_15 的雾感在 DiT 生成阶段由 A15+B15 条件组合触发，尚不能外推为普遍轻柔唱法缺陷。
- 多样本 DiT 交叉诊断所需的句级声音表示已准备：92,695 句七模型 embedding 完成，MERT 正式冻结 layer1 并双端验收。当前仍缺正式聚类试听、语义命名和可发布样本池；这只阻塞轻柔雾感支线，不是 V5 总门禁。
- V4vf 30k 与 V4vfg 10k 虽有 loss/FlowB 收敛，但用户试听均发现大量噪音；随机初始化路线感知门禁失败，不建议作为后续起点。
- V4Pvf 曾在最终感知结论前获独立授权并启动；量化 MIDI_P 的 smoke 与真实推理链路通过，但随机初始化路线随后被感知门禁否决，30k 在约 21.15k 主动终止，正式 checkpoint 已按用户要求删除。
- V4Pf 的 SOME/GAME teacher 调研已完成，GAME medium K=4 经用户试听判定基本可用。P 主线已
  与 H 合并为 V4PH：从 Official base 同时启用 H phone/PUL placement 与 GAME MIDI_P，先做
  500-step P-only 校准，再联合训练 30k。GAME→SOME CKA 等价接口、全量 cache、四卡 smoke
  和正式 P-only 500-step 均已完成；主 projected RMSE 明确改善，用户裁决随机 P 通过并采用
  P500。Phase B transition、单卡 resume、四卡 10-step 与 checkpoint 审计均通过，正式 30k
  已完整结束并通过 final 审计；30k Eval Loss/FlowB/CKA 为 2.0937/0.8733/0.0989。本地 CFG
  3/1 评分集首次听感确认结构路线成立；后续横向听评明确 V4PH 音色远不如 V4fg，因此 V4PH
  只保留为 H/PUL + GAME MIDI_P 结构控制候选，V4fg 仍是当前音色基准。final 已上传cloud storage
  `${CLOUD_ARTIFACT}`。
- V4IPH 已建立为 V4PH 的严格无 A 区单变量分支：`ref_len` 恒为 0，`cond` 全零，H/PUL、
  GAME MIDI_P、FlowB 与 CKA 覆盖完整时间轴；单卡、四卡 10-step、checkpoint 审计和实际
  resume 门禁均通过，正式 30k 已完成并通过 final 审计；Eval FlowB/CKA 为 `0.8727/0.0998`，
  与 V4PH 的 `0.8733/0.0989` 实质持平，final 与 SHA256: redacted
  `${CLOUD_ARTIFACT}`；后续头戴耳机听评与独立工程审计已裁决无 A 感知假设失败。
- V4Ij(ins)PH v1 经接管审计否决：step-10 smoke 未激活 adapter，正式 run 约 step 1250
  主动停止，且旧 step-8000 CPU/CUDA gate 必然失败。v2 已本地重写为从审计过的 V4IPH
  step 8000 完整状态精确续接，cache v2 绑定逐条 VAE 输入波形，active-adapter 单卡/四卡、
  checkpoint audit、连续 10 对 5+resume5 bit-exact 和真实双 R/null INS 全时间轴采样门禁均已在
  服务器通过；11,232 条 cache waveform rehash mismatch 为 0。正式 v2 四卡训练已从 V4IPH
  step 8000 完成至 global step 30000，final INS/null Eval Loss 为 `1.8123/1.8152`、FlowB 为
  `0.8711/0.8725`，final checkpoint 独立 audit 全通过并上传cloud storage `${CLOUD_ARTIFACT}`；
  既有 27 组已按每组 A 为唯一 INS 参考生成 CFG 3/1 共 54 条并通过完整性审计。用户听评裁决为
  相对 V4IPH 轻微改善、仍不及 V4PH；INS 有正收益但不足以替代细粒度 A 参考，V4IjPH 不晋级，
  且 V4PH 音色又远不如 V4fg。
- V4KPH 已建立为 PH/IPH 之间的待执行结构实验：训练仍从同一条音频切出互不重叠的 A/B，
  但 A raw latent 只进入 B 长度的并行 `cond`，B 独占 noised target、H/PUL、GAME MIDI_P 与
  模型时间轴；目标是在保留细粒度声学锚点的同时不让 A 占用生成上下文。当前只有设计与门禁，
  未实现、未训练；等待 IJPH 固定 checkpoint 听评后再由用户裁决是否启动。
- 头戴耳机发现随发声出现的噪音质感，主观约为 `V4IPH > V4PH > V4fg`。全量频谱与首轮
  单变量消融已建立：V4PH 高频谱平坦度稳定高于 V4fg，V4IPH 高频能量进一步升高；Euler64/
  Midpoint32 未显示主因信号，CFG 0 则使前三组 V4IPH 的 12--20 kHz 宽带化全部下降。等 RMS
  听评与独立工程审计已完成：标量 CFG/solver 无稳定感知收益，V4IPH 实现链路未发现错误；最大
  感知跳变为 V4PH 到 V4IPH，裁决为无 A 感知假设未通过，V4IPH 不再视为与 V4PH 等价候选。
- S 数据路线的 v3 20k、50-step 五卡全量自克隆已完成：private corpus count/private corpus count、失败 0。V4Sf 已用修复后的显式梯度同步复刻 V4f 配方，四卡 50-step smoke、original/SVC 双 Eval 与 checkpoint 审计均通过；形成 S 单变量结论前仍需 corrected-DDP 原始 waveform control。
- V4L 历史 L65 数据阶段已完成：1,296 个 45-60 秒窗口经 Whisper/SOFA/token 自动门禁后保留
  1,191 条、17.272 小时；strict loader、CPU 门禁与 GPU set 最长样本单卡 update probe 已通过。
  历史 L-TIME87H 45--60 秒池保留；V5 已改用新的 `L-FULL-LENGTH-TIME`：同一 private corpus count 条 TIME87
  范围上建立 3,373 个 30--60 秒最大窗口、44.082823h，每条至少两个短段，正 trim gap 以静音
  恢复，7,003 个来源短段不重复；音频与 L30/L65/L87 嵌套清单全量审计 issue=0。新母池尚未运行
  Whisper/SOFA/H/GAME/token。V5-P 继续采用 `KEEP_LONG_DEDUP_SHORT + NATURAL_RECORD`，正式去重
  按通过全部门禁的 Long 计算，被淘汰 Long 的成员 Short 返回 ShortPool；DDP/训练未启动。
- V4TIMETEST 首轮已冻结 `BASE=V4PH`，执行嵌套的 `TIME30H / TIME65H / TIME87H`。TIME30H
  已完成，用户听评为仅非常轻微弱于 TIME65H、但强于 V4IjPH；现有 strict-control 的
  private corpus count 条、65.630 小时冻结为 `TIME65H` 数据集合。当前 14,661 条、87.417 小时全量
  token-safe 产物缺少其中 82 条，不能直接作为母池；正式 TIME87H 已保留全部 65h 记录后
  兼容性重建并冻结为 private corpus count 条、86.999726h。已有 checkpoint 仅在最终基座与完整训练谱系一致时可作为中间组，小数据组不
  允许复用看过更大集合的任何学习状态。三层 control、统一 H 与 GAME cache 已完成全量审计；
  TIME30 已完成 Phase A/Phase B exact-resume、final checkpoint audit、云端归档与用户听评；
  `V4PH-30K-HIGHLR` 也已完成并以 `2.0841` 略低于原 V4PH 的 `2.0937`，final 与 sidecar 已
  云端及本机一致归档；用户听评音色略微强于原 V4PH、但仍弱于 V4fg。TIME87 自动串行已撤销
  并等待单独授权；2026-08-04 数据二次冻结复审全通过，未启动训练。IjPH/LPH 若后续晋级
  必须另建完整三点曲线。
- V4M 容量路线已由 TIME/HighLR 的小幅边际收益进一步支持，但用户否决从 338M 直接跳到 1.5B。
  当前冻结两个约 600M 的函数保持候选：深度先行 `M600-D (1024 x 40, 602,599,648)` 与深宽
  平衡 `M600-B (1024 x 31, heads=24, ff_mult=3, 600,462,048)`；先执行 D，B 只登记。
  M600 共享历史 V4PH Phase-A P500 与 HighLR 的 65H/训练谱系；M600-D 已用 FP32 FSDP
  FULL_SHARD、`no_sync`、activation checkpoint off 和分片 EMA 连续完成 12k，退出码 0，DCP/final
  report 审计通过。HighLR 6k/12k 与 M600-D 6k/12k 已在 GPU set 完成相同 27 组 CFG 3/1 共 216 条
  同 step 听评输出，完整性审计通过，匿名包已交付cloud artifact
  均未授权；1.5B 仍只在 M600 明确胜出后重新讨论。
- H 对齐路线已完成全量数据、共同训练入口、单卡与 corrected-DDP 门禁；严格确定性 4 卡连续
  10 与 5+resume5 的全部 checkpoint section bit-exact。H-Control/H 2k 均已完成并通过 checkpoint
  审计；FlowB 被降级为训练健康指标，不能衡量错节。下一步对生成前源 B 与两组生成结果重跑
  Whisper+SOFA，比较句内 mora 对齐保持度和错槽率并人工复核。H-PUL/H-v2 已命名 V4H，renderer、
  训练入口和全量 CPU 门禁通过；四卡 10-step smoke 与 checkpoint 审计均通过，正式 30k 已从
  Official base 独立完成，final checkpoint 全 section 审计通过，30k Eval FlowB/CKA 为
  0.8662/0.1354；step 24k 与 30k 均已完成cloud storage到本机重要模型目录的 SHA256: redacted
  V4H 24k 纯听感已确认咬字时间优于其他已听分支、基本解决目标错位，定位为切实稳健的小改动；
  音色仍弱于带 `g` 且额外训练 10k 的 V4fg 10k，极高音炸、混音输入炸与跑调未解决。下一步复核
  30k 并补 Whisper+SOFA 错槽评测。首轮 30k/24k 纯听感已确认 30k 音色更好、更细腻；30k 的
  咬字时间保持情况及其相对 V4fg 10k 的最终位置仍待单独记录。用户已授权 V4Hg 10k：从 V4H
  30k EMA 出发，以历史 V4fg 低 LR 配方适配 frozen 285k online VAE；静态、单卡、四卡 smoke、
  checkpoint 与实际 resume 门禁均通过，正式 10k 已完成且 final 全 section 审计通过；10k Eval
  Loss/FlowB 为 2.3034/0.9505，final 已通过cloud storage完成服务器/本机 SHA256: redacted
  正确 VAE 做 V4H/V4Hg/V4fg 同输入听感与错槽比较。

## 入口文档

| 文件 | 内容 |
|---|---|
| `总计划/plus-finetune-fullplan-v5-0704.md` | V5 完整执行计划：目标、根因诊断、V1→V4 失败回顾 |
| `总计划/v5-next-experiments-roadmap-20260715.md` | V5 实验路线图：后续实验、依赖关系、成本、停止条件 |

先读执行计划了解目标，再读路线图了解后续。

## 分支

| 目录 | 内容 | 状态 |
|---|---|---|
| [V5P/](V5P/main_V5P.md) | Official → P500 → non-g 40k → g 适配 → [PgO 8K](V5P/V5PgO/main_V5PgO.md) | PgO 已完成，保留为稳定基线 |
| [V/](V/main_V.md) | 固定 PgO DiT 和 VAE encoder，生成 latent 到原波形的 decoder 适配 | 300K 已完成；[阶段结论](V/V结论.md) |
| `V5S/` | Official fresh → 285k VAE + continuous SOME，从 step 0 的 40k Sg 路线 | 📋 合同已设计；入口、SOME cache、门禁和训练均未实现，未授权 |
| [GRPO/](GRPO/main_GRPO.md) | 最终 V5-Pg 基座上的 INS、GAME 概率音高与 QUA 复合相对奖励训练；[完整研究计划](GRPO/GRPO完整计划.md) | 计划/研究材料，不代表训练已授权 |
| `SOFA/` | 对齐实验：A7 → MFA → SOFA → V4f → V4g → V4fg → T1 盲听选型 | ✅ 已完成，V4fg 10k 为历史同源对照 |
| `VAE/` | 音质实验：VAE 审计 → Decoder-only → Full-VAE 300k | ✅ 285k online 胜出 |
| `STYLE_EMBEDDING/` | 句级 embedding 已完成；聚类试听、命名与样本池发布进行中 | 🔄 支撑分支进行中 |
| `V4vf/` | 随机初始化 DiT 的可行性门禁 | ⛔ 感知门禁失败，路线停止 |
| `V4Pf/` | SOME/GAME MIDI_P 调研、试听与适配工程证据 | ✅ GAME 基本可用，训练主线转入 V4PH |
| [V4Pvf/](V4Pvf/main_V4Pvf.md) | 随机 DiT + 量化 MIDI_P 从 step 0 联合训练；[实验记录](V4Pvf/量化MIDI联合训练.md) | ⛔ 随机初始化门禁失败后停止，checkpoint 已删除 |
| [V5-额外压缩通道实验/](V5-额外压缩通道实验/main_V5-额外压缩通道实验.md) | 表现力条件与压缩通道的研究设计 | 📋 研究方案，不是已训练的 V5 分支 |
| `V4PH/` | Official base + H phone/PUL placement + GAME MIDI_P | ⚠️ 结构控制成立，但音色远不如 V4fg |
| `V4IPH/` | V4PH 严格单变量取消 A 区，固定target singer身份由权重承担 | ⛔ 工程审计通过，但无 A 感知假设失败 |
| `V4IjPH/` | V4IPH + `j(ins)`：target singer INS 全局唱法特征注入空 cond | ⛔ v2 较 V4IPH 轻微改善但仍不及 V4PH，不晋级 |
| `V4KPH/` | 同源 A/B 通道化：A raw latent 进入并行 cond，B 独占生成时间轴 | ⏸️ 设计已建立；用户决定暂停并等待 M |
| `噪音诊断/` | V4IPH/V4PH/V4fg 发声区生成纹理的 CFG、solver 与 VAE 因果消融 | 🔄 等 RMS 听评待裁决 |
| `V4L/` | TIME87 连续短段构成 30--60 秒最大窗口，恢复 `<0.5s` trim gap 静音 | 🔄 L-FULL-LENGTH-TIME 3,373 条完成；V5-P 待 Whisper/SOFA/H/GAME/token |
| `V4TIMETEST/` | V4PH 基座上的 30h/65h/87h 嵌套训练时长消融 | ✅ TIME30/HighLR 完成；TIME87 等待单独授权 |
| `V4M/` | 函数保持模型容量扩展，先做约 600M 门禁 | 🔄 M600-D 先行，M600-B 仅登记，1.5B 暂缓 |
| `S/` | 全量训练音频经target singer SVC 自克隆后重建训练集 | ⚠️ 24k 感知略弱于 V4f；30k 待比较 |
| `H/` | 统一 kana 的摩拉锚定音素级对齐与 text placement | 🔄 V4Hg 10k 已完成并审计；等待 285k VAE 推理听感与错槽评测 |
| `LCF/` | PH/fg 低噪声端轨迹诊断与 Local Contrastive Flow 训练门禁 | ⛔ 30k 严格对照无明显听感改善，不进入 V5 |
| `SOURCE_ROBUSTNESS/` | 全网审计脏 source 训练增强、真实训练实现与本项目接入条件 | ✅ 首轮全网调研与接入审计完成 |
| `协作/` | V4Pf 与 V4vf 双向邮箱、文件所有权和服务器资源协调 | ✅ 协议已执行，V4vf 已结束 |

各分支有自己的索引（`main_V5P.md`、`main_GRPO.md`、`main_SOFA.md`、`main_VAE.md`、`main_STYLE_EMBEDDING.md`、`main_V4vf.md`、`main_V4Pf.md`、`main_V4Pvf.md`、`main_V4PH.md`、`main_V4IPH.md`、`main_V4IjPH.md`、`main_V4KPH.md`、`main_V4L.md`、`main_V4TIMETEST.md`、`main_V4M.md`、`main_S.md`、`main_H.md`、`main_LCF.md`、`main_协作.md`），按任务选择进入。

















