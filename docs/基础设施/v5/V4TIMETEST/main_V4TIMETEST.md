# V4TIMETEST 分支

本分支研究训练数据时长对最终歌声质量、歌词放置、旋律跟随和声学稳定性的影响。首轮已冻结
`BASE=V4PH`：IPH 与 IjPH 均已感知否决，PH 是当前唯一通过感知门禁的可用基座。

## 边界

- `TIME` 是训练数据时长轴，不代表新的模型结构或条件类型；
- 基座冻结后，30h、现有 65h、全量三组必须复刻同一条完整训练谱系，唯一主变量是允许参与训练的
  数据集合；
- 不允许复用看过更大数据集合的中间 checkpoint、P-only 权重、optimizer、EMA 或 continuation
  状态；
- eval 数据不计入训练时长，但三组必须使用同一份冻结 eval 和同一套推理输入；
- 当前 strict-control train 的 65.630h 冻结为 `TIME65H` 中间数据点；已有 checkpoint 只有在
  最终基座和完整训练谱系一致时才能直接充当该组；
- 当前全量 SOFA train 为 90.589h，现有全量 token/frame 门禁产物为 87.417h，但它缺少
  `TIME65H` 中的 82 条，不能直接作为嵌套母池；全量组仍以兼容性重建后的 `TIME87H` 为目标，
  不使用 `TIME90H`；
- 首轮只执行 V4PH 的完整 30/65/87 曲线；IjPH、LPH 或其他基座后续晋级时必须另建完整曲线。

## 命名

基座冻结前使用占位写法：

```text
{BASE}-TIME30H
{BASE}-TIME65H
{BASE}-TIME87H
```

`{BASE}` 必须替换为最终获选的完整路线名。若最终路线为 PH，则对应
`V4PH-TIME30H / V4PH-TIME65H / V4PH-TIME87H`；若最终路线包含 I、j(ins)、L 等条件，名称
必须完整保留这些字母，不能统一伪装成 PH。

## 当前状态

- 已确认现有 V4PH train 为 private corpus count 条、236,268.77 秒、约 65.630h；
- 已冻结该 strict-control manifest 为 `TIME65H` 数据集合，`65H` 是名义标签，精确时长仍为
  65.630h；
- 已确认 SOFA 全量 train 为 private corpus count 条、326,119.63 秒、约 90.589h；
- 已确认当前全量 token-safe 候选为 14,661 条、314,700.92 秒、约 87.417h；
- 交集审计发现 `TIME65H` 有 82 条、2,321.811 秒不在该候选中；两者路径并集为 14,743 条、
  约 88.062h；正式 `TIME87H` 已保留全部 65h 记录并确定性选择增量，冻结为 private corpus count 条、
  loader 有效 86.999726h；
- 已建立候选基座等待、嵌套子集、完整训练谱系和 L 重复内容计时原则；
- 首轮 `{BASE}` 已冻结为 V4PH；TIME30/TIME65/TIME87 control、统一 H 与统一 GAME cache 已完成
  构建和全量审计；
- TIME30 Phase A 500 已通过 projected RMSE 裁决；Phase A/Phase B 四卡连续 10 与
  5+resume5 均为全 section bit-exact，Phase B 30k 已完成；原 TIME87 自动串行已撤销。
  后续获授权的 `V4PH-30K-HIGHLR` 复用历史 V4PH 的冻结
  65H 数据、Phase A P500 与其裁决，Phase B 仍为 30k，唯一主变量是峰值 `1.4e-5` LR 保持到
  global step 24k（`warmup_steps=500, hold_steps=23500`），再用 6k cosine 衰减到零；
  `V4PH-TIME87H` 继续等待用户听评与单独授权，未出现 `$ROOT/APPROVE_TIME87_START` 文件时
  TIME87 管线必须安全退出。
- 2026-08-04 已对 TIME87 冻结资产完成第二次只读全量复审：private corpus count 条、86.999726h，严格
  `TIME30 < TIME65 < TIME87`，eval overlap 0，14,761 个 GAME cache 全部重新打开且数组有限；
  报告 SHA256: redacted`fcd130648dcdda254c0ca6b4abf665fd6f8b2309f4a0de7056ecda5825c7a015`。
  本次仅准备数据，`training_authorized=false`，未启动 TIME87 训练。
- `V4PH-TIME30H` Phase B 30k 已完成并通过 final audit，Eval Loss/FlowA/FlowB/CKA 为
  `2.1042/0.2782/0.8777/0.1007`，final SHA256: redacted
  `05ecf153ab75b013c3db152ea3ab6255dcac3e9307113d84e47542a4cab437cf`；checkpoint 与同名
  SHA256: redacted`${CLOUD_ARTIFACT}`，云端验收为 6.79GB + 87B，
  上传退出码 0、耗时 14 分 36 秒。
- 用户已完成 `V4PH-TIME30H` 对照听评：30H 仅非常轻微弱于现有 `V4PH-TIME65H`，但明确
  强于 V4IjPH。该感知排序与 30H/65H final Eval Loss 仅相差 `0.0105`（约 0.50%）一致，
  表明从 30H 增至 65H 的边际收益真实但很小；同时也表明当前尺度下 PH 路线选择的影响大于
  IjPH 通过更多训练数据可能获得的补偿。TIME87 仍不据此自动启动，下一项先完成并听评
  `V4PH-30K-HIGHLR`。
- `V4PH-30K-HIGHLR` 已完成 30k 与 final audit，Eval Loss/FlowA/FlowB/CKA 为
  `2.0841/0.2763/0.8698/0.0975`，相对原 V4PH final Loss `2.0937` 低 `0.0096`；final
  SHA256: redacted`0c9aeaf20789af52537c571b855a1041f9cf51c887eda6deea24bd99352a5466`。
  Checkpoint 与同名 SHA256: redacted`${CLOUD_ARTIFACT}`，云端验收为
  6.79GB + 87B，上传退出码 0、耗时 14 分 37 秒。用户直接听评裁决为：HighLR 音色略微
  强于原 V4PH，但仍弱于 V4fg；该结果只裁决音色维度，不自动扩展为总体能力晋级。cloud artifact
  `${LOCAL_PROJECT_PATH}`，主文件长度 `7,294,122,525` bytes，
  本机独立重算 SHA256: redacted
- 2026-08-03 已在本机使用冻结 27 组评分集完成 `V4PH-TIME30H` CFG 3/1 同构推理；两目录
  各含 27 WAV 与 27 份 H/GAME 审计，CFG 条件哈希差异 0、相同音频对 0、相对历史 V4PH
  dense text 差异 0、PAD 0、A 区 MIDI 非零 0、最大 GAME 边界差 1 帧。评分集位于
  `${LOCAL_EXPORT_PATH}`，未执行
  生成音频 F0 客观评分。
- 2026-08-03 已完成 `V4PH-30K-HIGHLR` 的同一 27 组 CFG 3/1 本机推理。两目录各 27 WAV
  与 27 份审计，CFG 条件哈希差异 0、相同音频对 0、相对原 V4PH dense text 差异 0、PAD 0、
  A 区 MIDI 非零 0、最大 GAME 边界差 1 帧。EMA 推理 checkpoint SHA256: redacted
  `18f06d63068727ae6263aae0fa168b6ea9cc95d210bc61f7b01cfcd556977d90`；结果已加入统一评分表，
  未执行生成音频 F0 客观评分。

## 工人

| 文件 | 内容 |
|---|---|
| `训练时长消融计划.md` | 基座选择、数据预算、嵌套子集、训练隔离、门禁、评价与停止条件 |
| `TIME87数据冻结.md` | 全量组正式命名、冻结 control/H/GAME、二次复审和训练禁令 |

## 阅读顺序

先读 `训练时长消融计划.md`，再读 `TIME87数据冻结.md` 获取当前全量组交付状态。最终基座裁决、
资产哈希、smoke 与正式训练记录继续追加到该活文档。

















