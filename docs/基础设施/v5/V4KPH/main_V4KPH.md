# V4KPH 分支

本分支在保留 V4PH 同源 A/B 切分的前提下，将 A 与 B 从时间拼接改为输入通道分离：
A 的真实 VAE latent 进入 64 维 `cond`，B 独占 noised target、H/PUL、GAME MIDI_P 与模型时间轴。
它验证声学 A 能否在不占用 B 上下文长度的情况下，恢复 V4IPH 因完全取消 A 而损失的发声稳定性。

## 命名

```text
V4 + K + P + H
```

- `K`：同一样本先切出互不重叠的 A/B，再将 A 作为并行 acoustic `cond`，B 作为完整生成区；
- `P`：沿用 V4PH 的 GAME MIDI_P；
- `H`：沿用 V4PH 的 H/PUL 歌词放置。

`K` 不表示恢复 V4PH 的 `[A | B]` 时间拼接，也不表示 IJPH 的 INS 全局 embedding。

## 边界

- 训练仍使用 V4PH 的同源非空 A/B 切分，不从 B 提取或复制 `cond`；
- `source_split_frame > 0`，但 `model_ref_len=0`，模型序列和输出都只包含 B；
- `x/target`、H/PUL 与 GAME MIDI_P 均为切点后的完整 B 后缀；
- `cond` 只包含切点前、与 B 不重叠的 A latent；不做时间插值、循环平铺或 INS 变换；
- 首版保持现有 DiT 架构，不增加 reference encoder 或 cross-attention；
- 首轮绑定 official frozen VAE。切换 285k VAE 必须另命名，不并入 K 单变量实验；
- K 只建立设计，不代表已经实现、通过门禁或获准启动训练。

## 当前状态

- V4PH 到 V4IPH 已被确认是当前最大的感知跳变；V4IPH 独立工程审计未发现特有接线错误；
- IJPH v2 固定 27 组听评已完成：相对 V4IPH 轻微改善、仍不及 V4PH，说明全局 INS 有正收益，
  但不足以补偿移除细粒度声学 A 的感知损失；
- 跨分支听评同时确认 V4PH 音色远不如 V4fg；K 即使补回 A 信息，也不能预设会自动追回该音色
  差距，启动优先级应排在现成 V4Hg 对照与可能的 V4PHg 诊断之后；
- K 的研究问题、张量布局、初始化、训练/推理契约、风险和强制门禁已经写入设计工人；
- IJPH 前置听评条件已经满足；用户当前决定暂停 K 等 PH 家族小结构实验并等待 M，K 未授权实现
  或启动。

## 工人

| 文件 | 内容 |
|---|---|
| `A区并行COND实验.md` | 研究问题、同源 A/B 通道化张量契约、训练/推理配方、门禁与裁决规则 |

## 阅读顺序

先读 `A区并行COND实验.md`。后续实现、门禁和执行记录只追加到该工人，不回写 V4PH、V4IPH
或 V4IjPH 的既有工人。

















