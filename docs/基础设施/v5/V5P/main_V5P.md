# V5-P 分支

本分支负责执行第一个正式 V5 集成路线。V5-P 从官方 YingMusic-Singer-Plus checkpoint
重新建立训练谱系，同时启用 H/PUL 文本放置、GAME MIDI_P 和 L-FULL-LENGTH-TIME 去重训练池；
结构阶段使用 official VAE，完成后再以独立 `g` 阶段切换 285k online VAE 修复音色。

## 路线

```text
Official checkpoint
  -> V5-P Phase A：P-only 500-step
  -> V5-P Phase B：official VAE，H + GAME-P + L-DEDUP，40k
  -> V5-Pg Phase C：285k online VAE，低 LR 适配 10k
  -> V5-Pg20-HLR07：从同一 V5-P 40k EMA 独立重启，强化适配 20k
```

V5-P 是 non-g 结构模型；V5-Pg 是同一路线完成 `g` 音色适配后的模型。首轮 V5-P 已执行完成，
当前以两个独立 g 适配预算做对照。

## 冻结边界

- DiT 从官方 checkpoint 初始化，不继承 V4PH、V4PH-HighLR、V4H、V4Hg 或 M600 权重；
- H 使用统一 kana、mora-anchored phone interval 和既有 PUL/SEP renderer；
- P 使用 GAME medium K=4、0.5 半音离散类、独立 REST/PAD 和 128 维可学习 embedding；
- L 使用 30--60 秒 L-FULL-LENGTH-TIME 候选，通过门禁后的 Long 保留，其成员 Short 不再独立进入训练池；
- Phase B 使用 official VAE；285k online VAE 只在 Phase C 加入；
- reference tail 固定 0.5 秒；
- 338M DiT 不扩容，不加入 LCF、INS、无 A、SVC 自克隆或 GRPO；
- V5-S 是未来 continuous SOME 对照路线，其中 `S` 表示 SOME，不表示既有 `S/` 的 SVC 自克隆。

## 当前状态

- 用户已冻结官方起点、P-first 顺序、L 去重、non-g 后接 g、H 长样本重建和 P/S 双路线；
- Phase B 冻结为 40k：0--2k warmup 至 `1.4e-5`，2k--28k cosine 缓降至 `1e-5`，再用
  12k cosine decay 到 step 40k 的 0；
- Phase C 冻结为保留 H/GAME-P/L 与 Phase-B loss/dropout 的历史 g 低 LR 10k 配方：0--250
  warmup、250--6250 hold `5e-6`、6250--10000 cosine decay 到 0；step 0 保存 transition 并做
  Short/Long Eval，随后每 1k 保存、每 500 step Eval；
- L-FULL-LENGTH-TIME 已完成全量门禁：Whisper 通过 3,139 条，SOFA/H 最终接受 3,129 条；
  门禁后得到 ShortPool 8,074、LongPool 3,129、TrainPool 11,203，加载时长 87.138097h；
- 推理接口允许 `B <= 60s`，但现有 30--60 秒 A+B 训练池不构成 B=60s 边缘质量承诺；
- Long 使用整段 Whisper 重转录和整段 SOFA；Whisper 门禁沿用历史 `kana>=0.75`、长度比
  `0.75--1.30` 并在全量完成后复审实际通过率；Long-H 沿用既有 V4H renderer，允许稀有
  collision/容量不足场景的既有覆盖与截断；
- 数据、最长 joint 图、四卡 mixed10、exact-resume 和 official VAE 门禁已通过；用户已授权启动
  fresh Phase A，并在 P500-v2 裁决通过后继续 Phase B 40k；
- Phase B 已在物理 GPU set、2、3、4 完成 40k，final checkpoint 与 SHA256: redacted
- 2K--40K 每 2K、EMA/raw、CFG 1.0 的 40 条实名条件已完成 1080 WAV 生成、审计和本机登记；
- 用户试听裁决 `40K` 基本最佳，raw/EMA 主体质量接近，EMA 稳定性略优；V5-P non-g 默认
  发布权重与未来 V5-Pg warm-start 来源因此冻结为 `40K EMA`，raw 只保留作技术对照。
- Phase C 已在本机实现独立 `g_adapt` schema、40K EMA/285k VAE 严格 provenance、step-0
  raw/EMA 等值审计、1K/500 轨迹、最长样本/四卡/exact-resume/formal 脚本。训练与
  1K--10K EMA/raw CFG1 的 540-WAV 实名轨迹代码已部署服务器，
  Bash/Python/LR/source 静态门禁通过；原 V5-P 20-checkpoint/1080-WAV 审计反跑无回归。
- 用户随后授权 Phase C：最长 60 秒单卡门禁通过，四卡 `continuous10 == 5+resume5` 全 section
  bit-exact，正式 `v5pg_phase_c_10k` 已于 2026-08-10 启动在物理 GPU set。正式 step-0
  raw/EMA/source、optimizer、EMA counter 与四 rank cursor 审计通过；任务最终正常完成 10k，
  pipeline `rc=0`，final Short/Long FlowB 为 `0.9494/0.9911`，checkpoint audit 与cloud storage上传通过。
- 为检验历史 g 低 LR/短步数是否欠训练，用户另行授权 `V5-Pg20-HLR07`：仍从 V5-P 40k EMA
  独立起步，将 Phase-B 双 cosine 时间轴压缩一半并把 LR 乘 `0.7`，形成 1k warmup 至
  `9.8e-6`、1k--14k shoulder 至 `7e-6`、14k--20k decay 至 0。四卡 exact-resume 与 step-0
  audit 已通过，正式 `v5pg20_hlr07_20k` 已在物理 GPU set 启动；step 200 健康，1k
  Short/Long FlowB 为 `0.9820/1.0239`，与低 LR 路线同 step 的 `0.9818/1.0223` 尚未分化。
  它是低 LR 10k 的平行对照，不是续训，主要证据区间是 2k--14k shoulder，最终仍由轨迹听评
  选择 checkpoint。
- 2026-08-12，用户决定搁置全部额外 g 加训。完整 40k ×0.7 g 路线只保留实验合同，不是当前
  20k 的自动后继；20k 完成后只做 final audit 和轨迹听评，不实现、不占 GPU。未来只有听评形成
  新证据并再次获得用户明确授权才恢复讨论；若执行，必须从 V5-P 40K EMA 独立 fresh 起步，
  不得直接续训 20k。`FlowB < 0.85` 不是冻结门槛。

## 工人

| 文件 | 内容 |
|---|---|
| `V5-P正式训练计划.md` | 完整谱系、L 去重、Long-H 构建、Phase A/B/C 配方、门禁与验收 |
| `V5-P训练前完整审计.md` | 2026-08-08 训练前历史快照；保留当时通过项、阻塞项、风险和启动顺序 |
| [V5-PgO](V5PgO/main_V5PgO.md) | 从 Pg20 EMA 出发的 8K 目标函数调整与后续质感诊断 |
| `V5-P-Alex数据接入DES_中文.md` / `V5-P-Alex数据接入DES_EN.md` | 外部数据准备接口与帧级 H/P token 合同 |
| `V5-P-Alex数据接入DR.md` / `V5-P-Alex训练计划.md` | 接入决策与训练方案；不包含样本音频 |
| [simply_jp](simply_jp/main_simply_jp.md) | 日语简化交付方案的文档及校验记录 |

## 依赖与阅读顺序

1. `../H/main_H.md`：H/PUL 语义和 renderer；
2. `../V4PH/main_V4PH.md`：GAME-P、P-only 与 joint transition 先例；
3. `../V4L/main_V4L.md`：L-FULL-LENGTH-TIME 音频母池和训练池语义；
4. `../VAE/main_VAE.md`：285k online VAE 选型；
5. `V5-P正式训练计划.md`：当前正式执行合同；
6. `V5-P训练前完整审计.md`：回溯 2026-08-08 的设计闭合依据；不再用其旧阻塞项判断当前状态。

















