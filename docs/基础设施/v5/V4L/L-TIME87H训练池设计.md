# L-TIME87H 训练池设计

## 目标

定义冻结 TIME87 数据范围上的 L 长上下文训练池，以及同一底层音频是否允许同时以 Short 和
Long 两种记录参与训练。本文只冻结数据集合关系、命名、采样接口和审计门禁，不授权训练。

## 正式命名

数据轴正式名称为：

```text
L-TIME87H
```

- `L`：训练中加入由连续短段拼成的 45--60 秒长上下文；
- `TIME87H`：唯一底层内容来自冻结的 private corpus count 条 TIME87 Path 集；
- 不命名为 `TIME90H`，因为冻结 30 秒 loader 的有效时长为 86.999726h；
- 长窗口是同一底层内容的派生视图，不把表观增加的 31.642634h 记为新增独立数据。

三个主要模型路线对应：

```text
V4fg-L-TIME87H
V4Hg-L-TIME87H
V4PHg-L-TIME87H
```

名称只描述模型路线与数据轴，不把训练池是否允许重复写进正式模型名。训练池政策必须写入
checkpoint metadata、run report 和评测标签。若同一模型路线同时比较两种政策，实验标签可追加
`DEDUP` 或 `REPEAT`，但该后缀不改变 L-TIME87H 的数据母池定义。

## 集合定义

设：

```text
T87 = 冻结 TIME87 Short 记录集合，private corpus count 条
L87 = 冻结 L87 Long 窗口集合，2,168 条
U87 = L87 使用的底层 Short Path 集，4,565 条
```

每条 Long WAV 由两条或三条连续 Short WAV 直接拼接：

| 每个 Long 的 Short 数 | Long 数 | 占比 |
|---|---:|---:|
| 2 | 1,939 | 89.44% |
| 3 | 229 | 10.56% |

`U87` 内每条 Short 最多进入一个 Long。Long 窗口之间不共享底层 Short，但 `U87` 的记录原本仍
存在于 `T87`，因此训练时需要显式选择是否保留这种 Short/Long 双重曝光。

## 政策 A：KEEP_LONG_DEDUP_SHORT

保留全部 Long，凡已进入 Long 的底层 Short 不再作为单独 Short 记录参与训练：

```text
LongPool  = L87
ShortPool = T87 - U87
TrainPool = ShortPool union LongPool
```

L87 结果：

| 项 | 数量/时长 |
|---|---:|
| ShortPool | 10,001 条 |
| LongPool | 2,168 条 |
| TrainPool | 12,169 条 |
| Short 实际帧时长 | 55.358846h |
| Long 实际帧时长 | 31.642634h |
| 表观合计 | 87.001480h |
| 唯一内容 | 87.001480h |

去重不是删除音频内容。被 ShortPool 排除的 4,565 条仍完整存在于 LongPool，只是不再以独立短记录
重复出现。

若把 12,169 条记录自然等概率 shuffle，Long 占记录数 17.82%，占加载音频时长约 36.37%。这只是
自然比例，不自动成为正式训练比例。

### 构建方法

1. 从 `windows_L87.json` 收集所有 `Segments[].name`，得到 `U87`；
2. 对 TIME87 Short manifest 按 `Path` basename 过滤，只有不在 `U87` 的记录进入 ShortPool；
3. LongPool 完整保留，不重新拼接、不重新选择窗口；
4. 将 Short/Long 分别打上显式 `Pool=short/long` 和稳定 source index；
5. 输出 short、long、mixed、smoke 和 audit 五类文件，不在 loader 内临时猜测去重关系。

必须审计：

```text
ShortPool Path intersect U87 = empty
ShortPool Path union U87 = T87 Path
LongPool source Path union = U87
LongPool 内 source Path 无重复
train/eval Path overlap = empty
```

## 政策 B：KEEP_LONG_ALLOW_REPEAT

不做去重，完整 ShortPool 与完整 LongPool 一起训练：

```text
LongPool  = L87
ShortPool = T87
TrainPool = ShortPool union LongPool records
```

L87 结果：

| 项 | 数量/时长 |
|---|---:|
| ShortPool | private corpus count 条 |
| LongPool | 2,168 条 |
| TrainPool | 16,734 条 |
| Short 实际帧时长 | 87.001480h |
| Long 表观时长 | 31.642634h |
| 表观合计 | 118.644115h |
| 唯一内容 | 87.001480h |

该政策允许 `U87` 的 4,565 条底层内容同时以独立 Short 和 Long 的一部分出现。若直接自然等概率
shuffle，Long 占记录数 12.96%，占表观加载时长约 26.67%。

重复模式不是错误数据，也不是新增 31.64h 独立录音。它是一种有意的重采样：让连续上下文内的
短段同时保留短任务曝光。潜在收益是降低短样本回归风险；潜在代价是 U87 内容相对其他 TIME87
内容获得更高权重，使“L 的长度收益”与“局部数据重复收益”混合。

必须审计：

```text
ShortPool Path = T87 Path
ShortPool Path intersect U87 = U87
overlap count = 4,565
LongPool 内 source Path 无重复
unique source hours 仍按 TIME87 计算
```

## L30/L65/L87

所有 TIME 版本从同一 L87 母池按底层 Short 成员关系过滤，不重新贪心打包：

| 集合 | TIME Short | Long | Long 使用的 Short | DEDUP Short | DEDUP 总记录 | REPEAT 总记录 |
|---|---:|---:|---:|---:|---:|---:|
| L30 | 5,065 | 230 | 471 | 4,594 | 4,824 | 5,295 |
| L65 | private corpus count | 1,257 | 2,628 | 8,403 | 9,660 | 12,288 |
| L87 | private corpus count | 2,168 | 4,565 | 10,001 | 12,169 | 16,734 |

冻结关系：

```text
TIME30 Path subset TIME65 Path subset TIME87 Path
L30 WindowId subset L65 WindowId subset L87 WindowId
```

同一完整 TIME 曲线必须使用相同 pool policy。不能让 TIME30 使用 DEDUP、TIME65 使用 REPEAT，
再把差异解释成数据小时数收益。

## Sampler 设计

集合是否去重与采样比例是两个独立变量。训练入口不得只把两个 JSON 拼接后依赖记录数量隐式决定
比例，除非实验合同明确选择“自然 shuffle”。推荐接口：

```text
pool_policy = KEEP_LONG_DEDUP_SHORT | KEEP_LONG_ALLOW_REPEAT
sampling_policy = NATURAL_RECORD | FIXED_POOL_PROBABILITY
long_probability = null | frozen float
```

### NATURAL_RECORD

Short/Long 合并后每条记录等概率。它简单、可复现，但 Long 的实际音频秒数和 attention 成本都更高，
必须报告每 1,000 step 的 Short/Long 记录数与加载秒数。

### FIXED_POOL_PROBABILITY

每次先按冻结的 `long_probability` 选择 LongPool 或 ShortPool，再在池内采样。它把训练权重从池大小
中解耦，适合控制变量实验。首轮概率尚未冻结，不能把当前自然占比直接写成正式值。

无论使用哪种 policy，日志至少实时打印：

```text
step / pool / sample duration / source segment count
cumulative short records / cumulative long records
cumulative short seconds / cumulative long seconds
elapsed / step time / ETA
```

checkpoint 必须保存 pool policy、sampling policy、long probability、Short/Long manifest SHA、
sampler RNG 与数据游标。连续 10 与 5+resume5 必须在 batch pool 顺序和全部训练状态上可复核。

## 首轮比较建议

若目标是严格回答“长上下文是否有效”，先使用 `KEEP_LONG_DEDUP_SHORT`：它保持唯一内容只曝光一次，
主变量最接近序列组织方式。若担心移除独立 Short 记录导致短任务回归，再用
`KEEP_LONG_ALLOW_REPEAT` 做第二个明确标注的采样消融。

两种政策都属于合法实验。最终不能只看 loss；必须共同比较长程崩坏、后段文本/旋律、发声区纹理、
音色，以及原固定短样本是否回归。

## 当前状态

- L-TIME87H 音频母池、L30/L65/L87 嵌套窗口与双端归档已完成；
- 本文已冻结两种训练池集合语义；
- Whisper、SOFA、token overflow 尚未对 L87 执行；
- 训练 manifest、正式采样比例、模型基线、DDP 和训练均未启动；
- 完成文本与 token 门禁后，必须在启动训练前由用户裁决 pool policy 和 sampling policy。

















