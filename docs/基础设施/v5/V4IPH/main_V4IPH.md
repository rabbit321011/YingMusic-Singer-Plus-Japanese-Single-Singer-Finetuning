# V4IPH 分支

本分支以 V4PH 为严格对照，验证固定target singer歌手模型在完全取消 A 区参考音频后，能否仅依靠
H/PUL 歌词放置与 GAME MIDI_P 完成全段歌声生成。

## 边界

- 起点与 V4PH phase B 完全相同：Official base + 已裁决通过的 V4PH P500；
- 数据、H/PUL、GAME cache、official VAE、优化器、scheduler、EMA、DDP、CFG dropout、
  CKA、训练步数和随机种子均与 V4PH 相同；
- 唯一实验变量是 `ref_len` 从 V4PH 的随机非空 A/B 切分改为恒等于 0；
- 保留 DiT 的 64 维 `cond` 输入结构以兼容原权重，但全时间轴恒为零；
- 全时间轴均为 B 区，完整启用 H/PUL 和 GAME MIDI_P；
- 不添加 speaker token、固定 latent、伪 A 帧或其他身份条件；
- 首轮继续使用 official VAE，切换 285k VAE 必须另命名，不并入本实验。

## 当前状态

- 独立训练脚本、全 B renderer、正式 30k 入口、四卡 10-step smoke 入口和 checkpoint
  审计脚本已建立；
- V4PH 原训练脚本与 renderer 未加入 V4IPH 开关，避免改变既有文件哈希和严格续训契约；
- 本地 Python 静态编译通过；V4PH/H 原 15 项 placement 单测通过；V4IPH 新增 3 项 renderer
  契约测试与 5 项 ref/loss 契约测试通过；
- 服务器单卡真实 1-step、四卡 10-step、checkpoint 审计和 step 10→11 actual resume 均通过；
- 正式 30k 已从 P500 完整结束，final checkpoint 与代码哈希/P500/optimizer/scheduler/EMA
  审计均通过；Eval `Loss/FlowB/CKA=1.8152/0.8727/0.0998`；
- final checkpoint 与 SHA256: redacted`${CLOUD_ARTIFACT}`；
- 与 V4PH 在可比 loss 上实质持平，但头戴耳机听评确认 `V4PH -> V4IPH` 是明显的发声区噪音
  跳变；独立工程审计未发现特有接线错误，最终裁决为“工程实现通过、无 A 感知假设未通过”；
- 本分支作为负结果和 IJPH/K 的 null 基线保留，不再视为与 V4PH 等价的可选候选。

## 工人

| 文件 | 内容 |
|---|---|
| `无A区全段生成实验.md` | 单变量定义、张量契约、实现位置、门禁、评价与执行记录 |

## 阅读顺序

先读 `无A区全段生成实验.md`。服务器门禁、正式训练与 checkpoint 审计结果持续追加到该工人文件。

















