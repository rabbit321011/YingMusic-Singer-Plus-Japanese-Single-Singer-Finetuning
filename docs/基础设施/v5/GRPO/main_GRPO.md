# GRPO 分支

本分支负责在最终人工裁决的 V5-Pg checkpoint 上设计感知相对奖励训练。目标是在保持
H/PUL 歌词放置、GAME-P 旋律控制、target singer唱法特征和长音频能力的前提下，降低发声区生成纹理并
提高整体自然度。

## 边界

- 基座必须来自 V5-Pg 轨迹的独立人工裁决，不能默认使用最后一步；
- 首轮每步 reward 不使用 PER，PER 只作为周期 checkpoint 的歌词回归门禁；
- 正式奖励轴为 INS 唱法相似、GAME 概率音高对齐和 QUA 感知质量；
- QUA 内部固定为 UTMOS22 与 DNSMOS SIG，不能换回旧 V4c GRPO 的 DNSMOS P808；
- 外层 `3:2:4` 与 QUA `7:3` 保持；用户已撤销逐组 population z-score（合同 Z）。当前采用训练
  原音 Q100 分位映射：每轴用 100 个等百分位锚点和 99 段直线单调映射到 `0--10`，范围外 clip，
  UTMOS22/SIG 再线性压到 `0--7/0--3`；GRPO 只做组内居中、不除当组 std；
- 只有真实注噪且 transition kernel 与 log-prob 完全一致的有效 B 区 SDE step 才能进入 ratio/KL；
  A、PAD、reference tail 和 window 外坐标保持 ODE；`a/window/NFE` 采用预注册统计门禁、短 pilot 与
  one-standard-error rule 裁决，当前具体数值仍待实验；
- 旧 V4c GRPO 只作为工程与失败经验来源，不直接 resume 其模型或 optimizer；
- 本分支当前只建立计划，不表示训练已经授权。

## 工人

| 文件 | 内容 |
|---|---|
| `GRPO待讨论问题.md` | 逐项讨论、补证据、用户裁决和同步记录的工作台 |
| `QUA_Q100评分映射合同.md` | 5,000 条训练原音参考、100 分位锚点、99 段插值、0--7/0--3 压缩与组内居中合同 |
| `GRPO原论文核验与算法合同.md` | 原始 GRPO、Flow-GRPO/MixGRPO/音频先例核验，本项目时间方向、动作粒度、策略角色与数学门禁 |
| `GRPO感知强化初步计划.md` | 奖励合同、离线标定、Flow-GRPO 工程边界、执行门禁、评价和停止条件 |

## 依赖与阅读顺序

1. `../V5P/main_V5P.md`：V5-P/Pg 谱系、基座候选、H/GAME/L/285k VAE 合同；
2. `../噪音诊断/main_噪音诊断.md`：发声区噪音定义、UTMOS22 与 DNSMOS SIG 证据；
3. `../V4IjPH/main_V4IjPH.md`：INS encoder、waveform 预处理和已知泄漏风险；
4. `../../v4/GRPO/GRPO_train_plan_0529.md` 与 `../../v4/GRPO/GRPO_v2_debug_log.md`：V4c GRPO
   的实现历史、耗时、结果和失败边界；
5. `GRPO待讨论问题.md`：按目录逐项推进讨论和裁决；
6. `GRPO原论文核验与算法合同.md`：核对论文事实、本项目推导和已裁决算法合同；
7. `GRPO感知强化初步计划.md`：当前设计与后续执行顺序。

















