# encode 子分支

Decoder-only VAE 30k 阶段。冻结 Encoder，仅训练 Decoder 和判别器，
在不改变 latent 接口的前提下改善重建音质。

## 工人

| 文件 | 内容 |
|---|---|
| `服务器SmokeTest.md` | 10 步工程验证：严格加载、梯度审计、Decoder 更新确认 |
| `Decoder30k训练.md` | 正式 30k 训练：配置、运行、checkpoint 审计 |
| `听评与诊断.md` | 用户听评结论、频段误差分析、loss 梯度归因 |

## 阅读顺序

SmokeTest → 30k 训练 → 听评与诊断。时序串行，每篇依赖前一篇的结果。

















