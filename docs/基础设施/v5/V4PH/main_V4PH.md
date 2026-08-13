# V4PH 分支

本分支从官方 YingMusic-Singer-Plus base model 出发，同时启用 H 的 phone/PUL 文本放置与
GAME MIDI_P，验证精确歌词放置和音符级旋律条件能否共同改善日语歌声合成。

## 边界

- DiT 从官方 base model 初始化，不继承 V4H、V4Hg、V4f 或 V4Pf checkpoint；
- H phone/PUL 放置与 GAME P 从本分支 step 0 起同时生效；
- 使用 SOFA 数据、A/B 结构和 official VAE；后续接 285k VAE 时另命名为 `V4PHg`；
- GAME 是训练和推理的唯一 MIDI teacher，正式路径不加载 SOME；
- P 条件使用 0.5 半音 class、独立 REST/PAD 和可学习 128 维 embedding；
- CKA 接收 GAME 转换出的 SOME-compatible `[B,T,128]` 音高概率数组，消费方不感知 teacher 类型；
- 不使用 `MIDIFuzzDisturb`。

## 路线

```text
Official base
  + H phone/PUL placement
  + GAME MIDI_P
  + SOFA
  + official VAE
  -> V4PH
```

先以随机初始化 P 完成 500-step P-only 校准，再依据训练轨迹与结构化音高核距离裁决 P
初始化；低步数阶段不做听感裁决。通过后从所选 P 权重开始联合训练 30k。阶段切换重建
optimizer、scheduler 和 EMA，不继承 P-only 优化器状态。

## 当前状态

- GAME medium K=4 已通过 62 条工程审计，用户试听结论为基本可用，允许晋级正式 P teacher；
- V4PH 路线、Official-base 起点、GAME-only teacher 与两阶段训练已经讨论冻结；
- GAME→SOME CKA 等价适配器已在同一 62 条完成基线：target singer 20 条平均音高中位误差 0.0952
  半音、probability cosine 0.9616、linear CKA 0.9269；正式映射冻结为 GAME posterior 偶数
  bin 直接对应 SOME MIDI 0..127；
- 11,232 条 GAME 离线 cache 已完成本机与服务器双端全量审计，H manifest 映射无缺失；
- P-only 训练入口、图探针、四卡 10-step smoke 与 checkpoint 审计均已通过；
- 正式阶段 A 500-step 已完整结束。主 projected RMSE 从 P300 的 0.339379 降到 P500 的
  0.327423；用户已裁决随机 P 通过，v2 pass 报告及 P500 均以 SHA256: redacted
- Phase B 单卡 transition/resume 与四卡 10-step smoke 全部通过，78 个支持 row 保留、177 个
  未支持 row 补核，fresh optimizer/scheduler/EMA 审计通过；正式 30k 已完整结束，final
  checkpoint 与全 section 审计通过，Eval Loss/FlowB/CKA 为 2.0937/0.8733/0.0989。
- final 已同步本机并通过 SHA256: redacted
  H phone/PUL 和 official VAE 完成同一 27 组 CFG 3/1 共 54 条评分集。两档 GAME/P/H 条件
  逐组一致，音频、边界、PAD、prompt mask 与 provenance 门禁通过；首次正式听感确认没有
  大问题、只剩中小问题；后续跨分支听评明确 V4PH 音色远不如 V4fg，因此 V4PH 只保留为
  H/PUL + GAME MIDI_P 结构控制候选，不再称为总体听感主候选。final 与 SHA256: redacted
  cloud storage `${CLOUD_ARTIFACT}`。

## 工人

| 文件 | 内容 |
|---|---|
| `GAME等价CKA与两阶段训练.md` | GAME/SOME 接口等价门禁、P 初始化诊断、阶段 A/B 配方与停止条件 |
| `阶段A执行记录.md` | cache、探针、smoke、500-step 正式运行、checkpoint 审计与纯距离裁决结果 |
| `阶段B执行记录.md` | P500 transition、补核、单卡/四卡门禁与 30k 正式运行状态 |
| `与V4fg音色对照.md` | 固定 27 组下 V4PH 音色显著落后 V4fg 的裁决、因果边界与路线影响 |

## 阅读顺序

先读 `GAME等价CKA与两阶段训练.md`，再依次读 `阶段A执行记录.md`、`阶段B执行记录.md` 和
`与V4fg音色对照.md`。

















