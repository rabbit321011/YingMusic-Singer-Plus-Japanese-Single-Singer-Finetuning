# Source Robustness

本分支调查如何把干净训练音频变成贴近真实推理输入的脏 source，使 Singer 在 source 域变化下仍输出干净target singer歌声。

当前阶段只做公开研究、源码验真和项目适配审计，不启动训练。候选方法必须能追溯到论文或真实训练代码；推理后处理、未调用 helper 和无来源猜测不计为可用实现。

## 工人

| 文件 | 内容 | 状态 |
|---|---|---|
| `脏源训练增强全网调研.md` | 搜索日志、候选证据、淘汰原因、数据依赖与 Singer 接入判断 | 已完成首轮全网调研与接入审计 |
| `四轨与训练切段对齐审计.md` | 服务器四轨来源、切段 manifest、实际 token 数据覆盖率与资产缺口 | 已完成服务器只读审计 |

## 当前门禁

1. 先证明方法实际用于训练，并明确输入、目标及被扰动分支。
2. 再证明它不要求项目当前不存在且无法替代的数据资产。
3. 最后审计其是否能进入 `source -> SOME -> MIDI_P -> Singer`，以及扰动后 clean target 是否仍成立。

## 首轮决策

- P0：以 OpenVPI GAME 的 colored noise、natural accompaniment/noise、RIR waveform chain 建立可执行 WET 基线；dirty copy 只进入 SOME，VAE target 与 reference cond 保持 clean。
- P1：按 ROSVC 论文研究“随机 instrumental + RIR -> 实际部署 separator -> dirty source”，用于覆盖真正的分离残留。
- 暂不执行：R2-SVC 未公开 Echo/Reverb/Harmony 算法；YingMusic harmony 需要 aligned 多轨；GAME pitch shift 与 spec masking 不属于第一阶段 source-WET。
- 在固定真实失败集与 frozen SOME 门禁通过前，不启动 Singer 训练。

















