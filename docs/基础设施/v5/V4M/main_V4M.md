# V4M 容量扩展分支

本分支研究如何在架构转换瞬间保持来源 checkpoint 函数和听感的前提下增加 DiT 容量。原
1.5B 目标已降级为远期方向，当前先执行约 600M 的容量门禁。

## 命名

```text
V4 + M
```

- `M`：扩大 DiT 模型容量；
- `M600-D/M600-B`：约 6 亿参数的深度先行与深宽平衡门禁；
- `1.5B`：只有 M600 感知门禁通过后才允许重新讨论的远期目标；
- `M` 不表示 MoE，不表示随机初始化，也不表示 INT8/INT4 量化。

## 边界

- M600 容量实验来源冻结为 V4PH Phase-A P500，使它与现有 338M HighLR control 共享同一来源；
  远期正式 M 来源仍须重新裁决，M600 来源不自动成为 1.5B 来源；
- raw model 与 EMA 都必须从同一个来源 checkpoint 做确定性扩展；不得只复制 shape 恰好相同的层，
  再让其余主体随机参与输出；
- 保持 VAE latent、文本、GAME MIDI_P、H/PUL、INS 和输出接口不变，首轮只改变 DiT 内部容量；
- BF16 是训练期 mixed precision，不是需要校准或不可逆转换的模型量化；来源 checkpoint 和扩模
  等价性审计仍使用 FP32；
- `M600-D: dim=1024, depth=40, heads=16, ff_mult=2` 已完成 30k；
  `M600-B: dim=1024, depth=31, heads=24, ff_mult=3` 为近等参数量深宽平衡对照，用户已于
  2026-08-05 授权按同一 HighLR 合同正式训练到 30k；
- V4M 不自动绑定 V4L。30 秒基线通过后是否加入 45-60 秒数据，必须作为后续独立组合裁决；
- 用户已于 2026-08-04 授权从冻结 transition step 0 直接正式训练到 12k；同 step 盲听解盲后，
  M600-D 12k（Model C）主观小幅胜过 HighLR 12k（Model D），随后授权从 12k 精确续训到 30k。

## 与其他分支的关系

```text
V4vf 随机初始化感知失败
  -> 禁止 1.5B 从随机函数起步

V4PH Phase-A P500
  -> FP32 函数保持扩模
  -> M600-D transition checkpoint
  -> 单卡等价性/显存门禁
  -> 四卡 FSDP smoke 与严格 resume 门禁
  -> 正式连续训练到 12k，保留 6k/12k
  -> 完成后统一生成 338M HighLR / M600-D 6k / M600-D 12k 听评集

M600 门禁通过
  -> 重新裁决 1.5B 来源、宽深比与训练配方
```

## 当前状态

- 当前 DiT 已审计为 `dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2`，参数量
  `338,118,880`；
- 原 1.5B 宽深候选与函数保持方案保留为历史设计，但用户已裁决不得直接跳到 1.5B；
- M600-D/M600-B 已冻结为近等参数量结构对照；D 的 30k 优势稳定但很弱，因此 B 已进入正式
  30k，用于区分纯加深布局问题与整体容量 scaling 边际收益弱；
- M600-B 正式 30k 已完整结束，退出码 0；末段原始分项显示 FlowA 与 D 基本持平，FlowB/CKA
  连续低于 D。30k publish 的 online/EMA strict-load、有限值和参数量审计通过；
- HighLR/M600-D/M600-B 三模型 30k 固定 27 组、CFG 3/1 听评已完成；162 条生成输出的格式、
  时长、非静音、placement、输入条件和 CFG 差异审计通过。已按既有评测集格式合并至
  `${LOCAL_EXPORT_PATH}` 根目录的六个
  实名模型/CFG 目录，并逐文件核 SHA；原始 ZIP 与 manifest 留存于同目录的 `_archive_` 子目录；
- 用户已完成三模型实名听评：M 分支相对 HighLR 有提升，但提升非常小。结合训练 Loss 也只有小幅
  改善，M 分支在约 600M 容量门禁处完成收口；不再追加 M600 训练，也不自动启动 1.5B；
- M600-D transition 已从 V4PH Phase-A P500 独立生成，SHA256: redacted
  `3940dd8d5283445c5eb65e3b47010c542e4d863561bb117894589cc9149ac830`；独立审计确认 22 个来源
  block、18 个新增恒等 block、9/9 等价样例 bit-exact，目标 DiT 为 `602,599,648` 参数；
- GPU set 单卡完整复制的 200-step A/B 已完成：全量 activation checkpoint 仅降低约 0.56GiB
  reserved 高水位，却使平均 step time 增加 34.7%；两组清缓存前物理余量都不足 3GiB，因此
  单卡完整复制被否决为正式训练拓扑；
- 四卡 FP32 FSDP 入口已实现；40/40 block 更新、DTensor checkpoint、严格
  `10 vs 5+resume5`、普通 online+EMA 发布合并和三组 200-step 显存/速度门禁均已通过；
- M600-D 首轮优化裁决为 `FSDP + no_sync + activation checkpoint off`：最差 reserved
  16.402GiB、最小物理 free 6.453GiB、稳态 2.972s/update；若未来加入新组分，优先改为每
  microbatch 同步，以约 6.8% 时间代价换约 2.48GiB 余量；
- 正式 12k 已于北京时间 2026-08-04 13:45 左右完整结束，退出码 0；final DCP/report、四 rank、
  12 个 checkpoint 和全程有限值/显存门禁通过，GPU set 已释放；
- 12k 同 step 盲听后已解盲：C=M600-D 12k、D=HighLR 12k，用户判断 C 好一点；M600-D 已获准
  按原冻结合同从 12k 续训到 30k；
- M600-D 已于 2026-08-05 从 12k 精确续训完成到 30k，退出码 0，final DCP/report 与四 rank
  状态核验通过；最终 last-500/last-1000 训练 Loss 仅比 HighLR 低 0.407%/0.364%；30k
  FlowA-CF paired eval 已完成，确认小幅真实预测改善、没有新增方差收缩或忽略 A，且困难样本尾部
  收益较大；
- M600-D 30k 与 HighLR 30k 的固定 27 组、CFG 3/1 同 step 听评已生成；108 条输出的格式、时长、
  非静音、输入条件、placement、GAME/VAE 边界和 CFG 差异审计通过。匿名包已由cloud storage拉回
  本地并逐文件核 SHA；用户随后撤销匿名，当前本地听评目录和新 ZIP 均使用真实模型名；三模型
  听评已完成，结论记录在 `M600-B30k三模型听评执行记录.md`；
- 6k/12k DCP 均已离线合并为普通 M600-D publish，online/EMA strict-load 与有限值审计通过；
  GPU set 已完成 HighLR 6k/12k、M600-D 6k/12k 的同 step 固定 27 组 CFG 3/1，共 216 条 WAV；
- smoke 与全量输入条件、H/GAME、时长、格式、非静音和 CFG 差异审计全部通过；匿名听评包与单独
  技术/解盲包已上传cloud storage `${CLOUD_ARTIFACT}`，当前等待人工评分；
- HighLR 12k vs M600-D 12k 的全量 201 条 A 区反事实/R2 诊断已完成：EMA 配对 hash 审计通过；
  M600-D 的 FlowB、CKA、完整 Loss、R2、斜率和方差比均小幅改善，不支持靠进一步平均化或忽略 A
  换取低 Loss；两模型都保留明确 A→B 影响，M600-D 对 zero A 更敏感，但普通错误 A 惩罚未普遍
  增强，因此也不足以裁决为更强的有效 A 条件利用；困难尾部完整 Loss 改善略多但不单调；
- 上述诊断已抽象为可复用 `FlowA-CF` 协议，明确区分训练日志的 A 区自预测 `FlowA-self` 与 A 对 B
  的反事实条件作用；标准 profile 使用 ΔFlowB、多模态 B 响应、距离有序性、CKA/F0 泄漏及
  R2/斜率/方差门禁，不再使用未经标定的 A→B 距离比作为“增益”或裁决指标；
- 用户决定暂停 PH 家族后续小结构实验；M600-D 与 M600-B 均已完成，M 分支已在 600M 门禁处收口，
  不自动继续 1.5B；其他窗口的 A 区/R2 诊断与本窗口的 M 分支训练相互独立；
- HighLR 听评音色略微强于原 V4PH、仍弱于 V4fg；M600-D 首轮已冻结从与 HighLR control 相同的
  P500 transition 出发，保持官方 DiT acoustic prior 和严格单变量容量对照。远期 1.5B 来源仍未
  冻结。

## 工人

| 文件 | 内容 |
|---|---|
| `1.5B函数保持扩模设计.md` | 约束来源、架构选型、Net2Wider 迁移、分片训练、验证门禁与停止条件 |
| `音色优先来源重审.md` | IJPH/PH 听评后撤销原来源假设，重审 V4fg 音色与 V4PH 控制的单一来源取舍 |
| `HighLR听评后的来源更新.md` | HighLR 听评后更新音色排序与 M 的音色优先/控制优先来源候选 |
| `M600深宽容量门禁.md` | TIME/HighLR 容量证据、M600-D/B 结构、显存方案、先行顺序与分阶段门禁 |
| `M600-D正式12k执行记录.md` | 正式来源、冻结配置、运行路径、安全门禁、自动监督和实时状态 |
| `M600-D同step听评执行记录.md` | HighLR/M600-D 6k/12k 同 step 生成、审计、匿名打包、cloud artifact
| `M600-D30k听评执行记录.md` | HighLR/M600-D 30k 同 step 生成、审计、匿名打包、cloud artifact
| `M600-D正式30k续训执行记录.md` | 12k 盲听裁决、精确恢复来源、30k 冻结合同、资源预算和监督规则 |
| `M600-B正式30k执行记录.md` | 深宽平衡转换、等价性/恢复/音频门禁、正式 30k 合同、资源预算和监督规则 |
| `M600-B30k三模型听评执行记录.md` | HighLR/M600-D/M600-B 30k 实名同协议生成、交叉审计、cloud artifact
| `M600-D层偏移审计.md` | 6k/12k 新旧层与 HighLR 的累计/区间参数位移、零门唤醒和解释边界 |
| `M600-D平均化与A区影响诊断.md` | 12k EMA 全量反事实 A、R2/方差收缩、A→B 多模态响应、timestep 与难度尾部诊断 |
| `A区因果条件评测协议.md` | 将 A→B 反事实诊断冻结为可复用 FlowA-CF profile；定义合同、指标、判据、smoke 与复用边界，并废止未标定距离比 |
| `FlowA-CF四模型执行记录.md` | HighLR/M600-D/V4PH 30k 严格同语义 cohort 与 V4fg 10k 跨语义参考 cohort 的 checkpoint、缓存、smoke、正式运行和结果解释 |

## 阅读顺序

先读 `1.5B函数保持扩模设计.md` 了解初版方案，再读 `音色优先来源重审.md` 和
`HighLR听评后的来源更新.md` 获取来源边界，最后读 `M600深宽容量门禁.md` 获取当前有效执行
方案。后续转换实现、资源 probe、训练记录和裁决只追加到新工人，不回写既有结论。

















