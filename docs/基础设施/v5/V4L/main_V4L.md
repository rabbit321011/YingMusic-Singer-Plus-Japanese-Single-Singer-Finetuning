# V4L 分支

本分支验证长音频参与训练是否能改善约 20 秒后出现的局部噪音、电音、声学崩坏和长序列不稳定。
`L` 表示训练集新增由既有合格短段自动合并得到的连续长样本。

## 边界

- 首轮目标长度为 45-60 秒，保留现有短样本并新增长样本池，不把全部训练数据替换为长音频。
- 只合并同一 BV、同一 split、编号连续且通过现有 SOFA/token-safe 门禁的短段。
- 直接顺序拼接当前合格 WAV；不跨缺号、不跨歌曲、不重新引入已过滤音频，也不添加 crossfade。
- 合并后重新运行 Whisper 和 SOFA；旧段歌词拼接结果只作为自动一致性基准。
- 全流程不设置人工复审。Whisper、SOFA、音频或 token 门禁失败的长样本自动排除。
- V4L 是长度数据轴，不自动绑定 S、P、随机初始化或特定 VAE；首轮训练前必须冻结唯一对照配方。

## 与其他分支的关系

```text
现有合格短段
  -> 连续段自动合并
  -> Whisper 重新转写 + 旧歌词一致性门禁
  -> SOFA 重新对齐 + token 门禁
  -> V4L 长样本池
  -> 与同一 Singer 配方的短样本基线做单变量对照
```

- T1 解决长、多句推理的文本锚点问题；V4L 主要验证训练长度对长程声学稳定性的影响。
- S 是 waveform 自克隆数据轴，P 是 MIDI 表示轴；与 L 的组合必须在各自单变量证据成立后另行命名。
- V4L 不回改现有短样本、SOFA 资产或历史 checkpoint。

## 当前状态

- 分支已建立，首轮数据定义与自动门禁已冻结。
- 已确认最终数据 1,535 个 BV 来源中，现有未切分源可覆盖约 1,447 个；但首轮采用合格短段合并，不依赖原源重裁。
- 保守的 private corpus count 条 token-safe 子集已存在大量连续候选：45 秒以上 2,023 个 run，60 秒以上 1,269 个 run。
- V4L v1 样本处理已完成：1,296 个 45-60 秒窗口经 Whisper 一致性门禁保留 1,225 条，SOFA 1,225/1,225 成功，token overflow 后正式保留 1,191 条、17.272 小时。
- 最终资产位于服务器 `${SERVER_ROOT}/final_sum_large_V4L_45_60_gap05/`，全量审计 issue=0。
- 独立严格 loader 与自然混合候选已完成 CPU 门禁：不再静默裁剪，短 private corpus count + 长 1,191，长样本自然占比 9.7447%；该比例不是正式训练配方。
- GPU set 隔离单卡 probe 已完成：59.907 秒最长样本一次完整 update 成功，峰值 allocated/reserved 为 11.56/11.65GB，checkpoint 写入被拦截。
- 当前 V4IjPH 使用正式训练资源；V4L 未执行 DDP smoke 或训练。模型基线与正式短长比例仍未冻结。
- L-TIME87H 音频母池已从冻结 TIME87 构建：完整短池为 private corpus count 条、约 87h，另派生 2,168 个
  非重叠 45--60 秒长窗口、31.642634h；长窗口使用 4,565 个唯一短段，WAV/成员关系/连续性审计
  issue=0。当前只完成音频阶段，Whisper、SOFA、token、短长采样政策与训练均未启动。
- L-TIME87H 训练池已冻结两种合法集合语义：`KEEP_LONG_DEDUP_SHORT` 保留 2,168 个 Long 并从
  ShortPool 排除其 4,565 个来源短段；`KEEP_LONG_ALLOW_REPEAT` 则完整保留 Short 与 Long 的双重
  曝光。V5-P 已裁决使用 `KEEP_LONG_DEDUP_SHORT + NATURAL_RECORD`：正式去重集合按通过全部
  Whisper/SOFA/H/token/GAME 门禁的 Long 计算，失败 Long 的成员 Short 返回 ShortPool；Long-H
  和 GAME cache 在 Long 全局时间轴重新构建。该裁决只绑定 V5-P，其他模型路线仍须独立登记。
- V5 已用 `L-FULL-LENGTH-TIME` 替换正式音频候选：同一 TIME87 范围上生成 3,373 个 30--60 秒
  最大合法窗口，每条至少两个短段；相邻 trim gap 严格 `<0.5s`，正 gap 以 44.1kHz PCM 静音
  恢复。窗口使用 7,003 个不重复短段，总长 44.082823h；独立全量审计 issue=0。历史 45--60 秒
  L-TIME87H 保留，但 V5-P 后续资产不得继续读取旧 WindowId。

## 工人

| 文件 | 内容 |
|---|---|
| `长音频训练实验.md` | V4L 数据构建、自动门禁、训练对照、评价与停止条件 |
| `执行交接.md` | 当前资产、SHA256: redacted
| `L-TIME87H音频母池.md` | TIME87 完整短池、派生长窗口、双时长口径、音频审计与cloud artifact
| `L-TIME87H训练池设计.md` | fg/hg/phg 正式命名、Long/Short 去重或重复政策、嵌套集合、sampler 与恢复门禁 |
| `L-FULL-LENGTH-TIME音频母池.md` | V5 正式 30--60 秒最大窗口、trim gap 静音恢复、全量审计与新嵌套子集 |

## 阅读顺序

先读 `长音频训练实验.md` 了解实验定义，再读 `执行交接.md` 获取历史 L65 可执行状态；需要进入
全量数据路线时，依次读 `L-TIME87H音频母池.md` 和 `L-TIME87H训练池设计.md`。后续数据统计、
实现记录和实验结论持续追加到对应工人文件。V5-P 的正式使用合同见
`../V5P/V5-P正式训练计划.md`。V5 正式音频定义最后读取
`L-FULL-LENGTH-TIME音频母池.md`，它覆盖旧池的窗口定义。

















