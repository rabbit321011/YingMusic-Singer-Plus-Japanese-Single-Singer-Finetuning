# V 分支：V5PgOV decoder 适配

V5PgOV 固定 V5PgO 8K DiT 和 285K online VAE encoder，仅适配 VAE
decoder。目标是让生成过程实际产生的 B 区 latent `z_gen` 对应授权的
目标歌手原始 B 区波形 `x_B`：

```text
A 参考 + H/PUL 歌词/时间条件 + GAME-P 离散旋律
  -> 冻结的 PgO DiT -> z_gen
  -> 适配的 VAE decoder -> x_B
```

这里训练的是音频 decoder，**不是** SOME melody encoder。缓存必须是
DiT 直接生成的 latent，不能将生成 WAV 再编码后当作 `z_gen`。

阶段一使用四轮独立采样的兼容缓存、32 latent 帧训练窗口、300K step、
EMA 和关闭的 KL 损失。生成缓存时随机采用三路 CFG；decoder 训练本身
没有固定的 CFG。技术参数、缓存格式、运行命令和授权数据边界见
[公开训练配方](../../../v5pgov-decoder-adaptation.md)。

公开仓库提供脱敏的
[训练入口](../../../../src/package_v4c_finetune/train/train_v_decoder_adapt.py)
及[配置](../../../../config/model_v_decoder_adapt.json)，但不提供训练数据、
生成缓存、模型权重、试听音频或云盘地址。训练入口需要用户自行准备
有合法使用权且通过一致性校验的输入。

## 状态

阶段一已完成，300K EMA 在后续长片段听评中因稳定性被选为最终
EMA 候选。该导出只有 decoder 权重，不是可独立推理的完整模型；
仍需匹配的 VAE 和 PgO DiT。它没有自动替换 PgO 的默认预设。

[阶段结论与证据边界](V结论.md) |
[PgO 基座](../V5P/V5PgO/main_V5PgO.md)

## 扩展记录

- [阶段一执行计划与训练合同](V分支执行计划.md)：缓存构建、帧布局、损失和验证。
- [完整阶段结论](V结论文档.md)：听评观察、可能机制及未证实的解释。
- [300K EMA 选型记录](300K_EMA选型与云盘归档.md)：仅保留已脱敏的选型事实；
  原始归档位置和权重指纹不公开。
