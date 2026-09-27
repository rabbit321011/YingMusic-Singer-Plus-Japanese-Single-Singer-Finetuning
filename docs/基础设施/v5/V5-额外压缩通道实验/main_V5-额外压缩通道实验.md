> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# V5 额外压缩通道实验

本分支研究在现有 `A + TEXT + MIDI` 条件之外增加一条低带宽条件通道，使模型能够从目标音频、
参考音频、SVC 后参考音频或人工控制序列中获得局部歌唱表现信息。

核心实验变量是额外压缩通道本身：输入信息、瓶颈容量、时间分辨率、泄漏边界、条件 dropout、CFG
控制和 B 区注入方式。它不归属于 V5-Pg 的 `g` 适配变量，也不自动继承任何尚未裁决的训练路线。

## 当前状态

- 已完成相关工作与公开方案调研；
- 已冻结 PerformancePlan 和时间轴合同的 draft；
- 已在服务器物理 GPU 7 完成首轮 F0 tracker bake-off；
- 已将低维条件自编码器、第四正式条件、训练 mask 和三前向 CFG 落成可执行实验合同草案；
- 尚未实现额外压缩通道；
- 尚未修改主模型或启动训练；
- 4D、8D、VQ/RVQ、显式控制与混合表示均仍是待验证候选。

## 文档

| 文件 | 内容 | 状态 |
|---|---|---|
| [歌唱表现力控制研究.md](歌唱表现力控制研究.md) | 核心目标、外部证据、压缩表示、第四条件 CFG、泄漏门禁、最小实验和 GPU 实测 | 持续研究，尚未冻结训练合同 |
| [额外压缩通道实验合同草案.md](额外压缩通道实验合同草案.md) | `64D target + MIDI -> 极窄 code -> MIDI 重注入 -> 64D B 条件`、第四条件 mask/CFG、首轮消融与停止条件 | 架构语义已落盘；数值超参和训练授权未冻结 |

## 实验产物

首轮 F0 实验代码和本机结果位于：

```text
${LOCAL_PATH}
```

服务器结果位于：

```text
${PRIVATE_PATH}
```

服务器路径中的 `TEMP` 只表示独立实验产物目录，不表示本分支仍属于文档 TEMP 研究区。

## 执行边界

建立本分支不等于授权完整重训。进入模型实验前至少需要冻结：

1. 起始 checkpoint 与严格 Base 对照；
2. 压缩通道输入和时间轴；
3. bottleneck 容量及 anti-copy probe；
4. 独立 null/dropout/CFG 语义；
5. `P-off` 基线回归、reference swap、shuffle 和错歌词泄漏评测；
6. 独立脚本、checkpoint schema、输出目录和资源声明。
