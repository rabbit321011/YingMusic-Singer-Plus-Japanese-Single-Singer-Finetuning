# V5-PgO：离散旋律路线的目标函数调整

V5-PgO 从 V5-Pg20 EMA 启动，保留 H/PUL 文本条件、GAME-P 离散旋律条件、
训练数据合同与 285K online VAE。它不重新引入 continuous SOME 特征，也不
训练或使用 SOME melody encoder 作为生成模型的旋律输入。

## 训练路线

- 从 V5-Pg20 EMA warm-start，新增 8,000 次 optimizer update；这不是
  原训练的 optimizer、随机数状态和数据游标的 exact resume。
- 去掉 B-only hidden-state CKA 正则，将目标改为 `FlowA + 3 * FlowB`。
- 单卡训练，microbatch 1、梯度累积 16；DiT、H/PUL embedding 和 P
  embedding 可训练，285K online VAE 保持冻结。
- 训练条件 dropout 为 audio 0.3、text 0.15、MIDI 0.3。推理时的 CFG
  另行设置，不能把训练 dropout 当成推理 CFG。
- 学习率：前 1K step warmup 至 `1e-5`，1K--3K 保持，3K--6K 降至
  `3e-6`，6K--8K 降至 0。

正式训练已完成至 8K，并完成 checkpoint 审计。此分支同时改变了 CKA
权重、B 区 Flow 权重和新增训练步数，因此不能把听感变化归因于其中
单独一个因素。完整可运行的私有训练数据、缓存和 checkpoint 不在本仓库。

## 观察与后续

PgO 是后续 [V5PgOV decoder 适配](../../V/main_V.md)的固定生成基座。
针对额外 CFG 系数的两段固定条件听评未发现仅靠调低该系数就能稳定
解决质感缺口；这不等于排除了所有推理设置的影响。

当前仍将 PgO 保留为稳定默认/回滚基线。V5PgOV 在已听困难样本上的
质感更好，但有柔声区域的副作用，不能据此推断对未见曲目的普遍改善。

相关：[V5-P 路线](../main_V5P.md)、
[V 分支阶段结论](../../V/V结论.md)。
