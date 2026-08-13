# SOFA 接入与 Gold100 评测

V5 P0 对齐线的收束。SOFA 是当前选定的对齐器，使用 Greenleaf2001 JPN_Test2_Plus 模型
搭配 Voicebank2DiffSinger 风格的 PyOpenJTalk G2P 日语前端。

## 路线

最终采用并复刻了社区实际路线：

```
SOFA + Greenleaf2001 JPN_Test2_Plus + Voicebank2DiffSinger 风格的 PyOpenJTalk G2P
```

- 输入：完整音频段 + 该音频段内全部参考文本
- 运行方式：整段直接强制对齐，不经过 A7 分块
- 输出：words + phones 两层 TextGrid
- 验证：只使用人工 Gold 时间戳计算句级命中率

### 关键社区事实

- JPN_Test2_Plus 是当前 Japanese-extension 系较主流的日语 SOFA 模型。
- Voicebank2DiffSinger 使用 pyopenjtalk.g2p() 产生音素，在词间插入 SP。不是简陋日语转罗马音。
- **禁止使用**项目中那个不参考上下文、君不会读成 kimi 的简陋日语转罗马音方案。

### 环境

社区仓库 `Voicebank2DiffSinger-main`，环境：Python 3.11.15，torch 2.6.0，pyopenjtalk-plus 0.3.4.post10。

### I/U 清化音素兼容

pyopenjtalk-plus 产生清化元音 I/U，模型词表只有小写 i/u。社区惯例：对 g2p() 结果调用 .lower() 归一化。

## Gold100 正式结果

剔除 Gold 第 83 条（75 句全相同、75 个 start 全为 0.0 的损坏数据），有效 Gold：99 样本、607 句。

| 方案 | 命中 | 前串 | 前缩 | 后串 | 缺失 | 命中率 |
|---|---:|---:|---:|---:|---:|---:|
| B4：A7 + WhisperX | 332 | 110 | 61 | 5 | 99 | 54.70% |
| SOFA JPN_Test2 | 563 | 33 | 6 | 5 | 0 | 92.75% |
| **SOFA JPN_Test2_Plus** | **573** | **28** | **4** | **2** | **0** | **94.40%** |

Test2_Plus vs B4 逐句对照：B4 非命中→Test2_Plus 命中 248 句，反向 7 句。

## A7 分块对 SOFA 无效

在 10 个样本、55 个可比短语上的窗口扫描：

- 全段 SOFA Test2_Plus：54/55，98.18%
- A7 分块 ±0.5s：56.36%
- ±1.0s：43.64%，±2.0s：23.64%，±3.0s：10.91%，±6.0s：3.64%

A7 边界不准，逐句切窗会把真实发音切出窗口，或在宽窗口中让单句文本吸附到错误位置。
**V5 的"所有候选均先经过 A7 分块"不适用于 SOFA。**

当前 P0 固化为：**完整音频段 + 全部参考文本 → SOFA Test2_Plus 全段强制对齐。**

## 最小样本验证

使用真实target singer歌声音频和对应参考文本，不提供任何人工时间戳，成功生成 TextGrid（words + phones 两层）。社区 `pyopenjtalk → SOFA → 日语模型 → TextGrid` 链路已验证可运行。

















