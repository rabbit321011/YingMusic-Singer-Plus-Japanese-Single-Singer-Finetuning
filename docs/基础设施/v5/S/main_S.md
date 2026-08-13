# S 分支

本分支验证 SVC 自克隆训练集路线：将当前target singer歌声训练集的音频全部经过
target singer YingMusic-SVC 自克隆，再把转换后的音频作为 Singer 训练数据。

## 边界

- `S` 只表示训练 waveform 来自 SVC 自克隆，不绑定 DiT 初始化、MIDI 表示或 VAE 版本。
- SVC 处理覆盖训练集音频；A/B 区都从处理后的同一 waveform 构造。
- 歌词、train/test split 和 SOFA 句级标注原则上沿用，但必须先通过时长与边界保持门禁。
- 所有依赖 waveform 的训练资产必须按冻结规格重新提取，不能把旧音频 latent 与新音频混用。
- 本分支不是 YM-SVC B-cond、推理后处理或少量 replay 数据路线。

## 与其他分支的关系

```text
原始target singer训练集
  -> target singer SVC 自克隆
  -> S 训练集
  -> 与同一 Singer 配方的原始训练集做单变量对照

S 是数据轴，可在证据成立后与 f/P 等配方组合；组合实验必须另行命名。
```

## 当前状态

- 分支已建立，定义与阶段门禁已写入。
- 正式 SVC 主模型已冻结为 v3 20k；每条音频以自身作为 source/target，运行时实时提取 CampPlus。
- 5 卡、50 diffusion steps 全量 SVC 自克隆已完成：private corpus count/private corpus count，失败 0，墙钟 5小时13分32秒。
- 独立全量审计通过：缺失 0、额外 0、格式/时长错误 0；产物 28.7568GB，最大时长差 11.83ms。
- 用户基于既有 YingMusic-SVC 使用经验，裁决不再设置额外配对试听门禁；本轮 waveform 产物正式接受。
- V4Sf 已按 V4f 配方接入 S waveform，并用显式 all-reduce 修复历史 V4f 绕过 `DDP.forward` 的问题。
- 201 条 V4f test 已生成独立 SVC 版本；训练同时保留 original/SVC 两路 Eval，test 不进入训练集。
- 四卡 50-step smoke 已完成，effective batch 16；checkpoint、双 Eval 和数据完整性审计均通过。
- 正式 V4Sf 30k 已正常完成；最终 original/SVC Eval Loss 为 `2.2849 / 2.2358`，四卡已释放。
- 用户完成 24k 同步数对比后判断：V4Sf 24k 整体略弱于 V4f 24k；SVC Eval 更低尚未转化为整体感知优势。

## 工人

| 文件 | 内容 |
|---|---|
| `SVC自克隆训练集.md` | 路线定义、数据重建范围、pilot、全量转换、训练对照与停止条件 |
| `V4Sf训练实验.md` | V4f 配方复刻、DDP 修复、双 Eval 数据与 50-step smoke 结果 |

## 阅读顺序

先读 `SVC自克隆训练集.md`，再读 `V4Sf训练实验.md`。前者冻结 S waveform，后者记录
Singer 训练实现、运行证据与后续对照边界。

















