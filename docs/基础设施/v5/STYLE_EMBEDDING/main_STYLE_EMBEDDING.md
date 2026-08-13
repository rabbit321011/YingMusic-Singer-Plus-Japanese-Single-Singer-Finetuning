# 唱法与音色样本池

本分支负责建立数据集的句级声音表示、可检索唱法/音色样本池和聚类验证，为 SOFA/v4f 的多样本 DiT 条件交叉诊断提供前置能力。

## 当前状态

- 92,695个SOFA句级样本的七模型正式embedding已经完成提取、验收和本地/服务器双端保存。
- MERT已完成13层诊断与96题人耳选层，正式冻结layer1；其更符合当前同歌手句级音色/唱法检索目标，结论强度为weak工程选择。
- embedding/选层任务已结束。正式聚类、选K、cluster试听、语义命名和样本池组织由独立聚类任务继续，本分支状态因此仍为进行中。

## 目标

- 为每个 SOFA 时间戳句子生成稳定主键与统一 manifest。
- 比较 CAM++、emotion2vec 及后续候选 embedding 的跨歌曲近邻质量。
- 至少能够稳定检索普通、略轻柔、明显轻柔等样本。
- 聚类与自动指标只作候选发现，最终语义需由人工试听确认。
- 本地和服务器都保存最终句级向量、模型配置与结果哈希。

## 工人

| 文件 | 内容 |
|---|---|
| `唱法音色样本池.md` | 数据范围、当前产物、模型候选、验证门槛与未决问题 |

## 责任划分

- `experiments/singing_style_embeddings/`：manifest、embedding、MERT选层、向量验收与评估工具的技术权威。
- 聚类任务：正式空间选择、聚类算法/K、人工试听、簇命名及下游样本池发布权威。
- embedding实验生成的聚类预计算只供交接，不自动升级为本分支正式结论。

## 工程位置

`${LOCAL_PROJECT_ROOT}\experiments\singing_style_embeddings\`

完整实验状态、数据契约、模型产物、聚类和试听流程统一见 [`experiments/singing_style_embeddings/README.md`](../../../../experiments/singing_style_embeddings/README.md)。

















