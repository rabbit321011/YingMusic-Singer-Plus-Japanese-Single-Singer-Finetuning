# VAE 分支（已收尾 ✅）

V5 的音质实验分支。从日语 VAE 盲听确认问题（VAE 不是透明组件），
到 Decoder-only 30k 验证改善方向，最终 Full-VAE 300k 完成。

**结论：full-VAE 285k online 为最佳 VAE 模型，高频表现显著优于 decoder30k。**

## 工人

| 文件 | 内容 |
|---|---|
| `动机与架构审计.md` | 盲听确认 VAE 是音质天花板、官方源码版本溯源、干净推理环境搭建 |
| `StageB实验计划.md` | Decoder-only 实验设计、数据准备、损失链选择、停止条件 |

## 子目录

| 目录 | 内容 | 启动条件 |
|---|---|---|
| `encode/` | Decoder-only 30k 实验：SmokeTest → 训练 → 听评诊断 | StageB 计划批准后 |
| `fullRun/` | Full-VAE 300k：训练配置 → 中途评测 → 轨迹分析 → **最终结论** | encode 完成后 |

`fullRun/` 内新增：
- `VAE实验结论.md` — **最终的训练轨迹、选型决定、产物位置、后续建议**

## 阅读顺序

动机与架构审计 → StageB实验计划 → encode/ → fullRun/（先训练配置、再中途评测、最后结论）

















