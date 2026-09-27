> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# simply_jp 分支

本分支负责 V5-P 训练脚本的可读化交接与行为校验。

目标是让新接手者能够先理解训练主路径，再在 GPU 7 上完成单卡验证；不改变 V5-P
正式训练的实验语义。旧版 `train_v5p.py` 仅作为只读 oracle，用于回归对照，不在本分支修改。

## 边界

- 不改变 V5-P 的数据、H/PUL、GAME-P、L 去重池、loss、dropout 或 checkpoint schema；
- 不自动启动正式训练；
- 单卡验证使用 `CUDA_VISIBLE_DEVICES=7`，不得占用 V5-P 正式路线的 GPU 1--4；
- 60 秒最长样本门禁沿用 V5-P 训练合同；
- 所有可读化改动必须经过旧版 oracle 对照和 checkpoint 审计。

## 校验路线

按以下顺序执行：

1. 语法、导入与参数解析；
2. V5-P 配置合同、路径和 SHA256；
3. 训练 manifest、H/PUL 和 GAME cache 抽样/全量校验；
4. 单步 forward/backward smoke；
5. GPU 7 单卡 step-1/step-10 checkpoint 审计；
6. 单卡 checkpoint resume；
7. 60 秒最长样本 forward/backward/save；
8. 梯度与参数更新审计；
9. 代码结构审查，确认 loss、mask、detach、EMA 和保存语义未变；
10. 固定输入下与旧版 oracle 回归对照。

## 工人

| 文件 | 内容 |
|---|---|
| `交接导读.md` | 训练入口、数据流、张量和 checkpoint 生命周期的简明说明 |
| `校验清单.md` | 本分支选定的校验项目、命令、通过标准和结果记录 |
| `校验结果_20260901.md` | GPU 7 单卡 smoke、60 秒样本和 resume 的实际结果 |
| `audit_v5p_game_cache.py` | 当前 V5-P GAME cache manifest 的独立全量审计器 |

## 依赖阅读顺序

1. `../main_V5P.md`：V5-P 实验合同和当前状态；
2. `../V5-P正式训练计划.md`：正式谱系、门禁和 60 秒样本要求；
3. `交接导读.md`：简化后的代码阅读入口；
4. `校验清单.md`：修改后执行的验证顺序。
