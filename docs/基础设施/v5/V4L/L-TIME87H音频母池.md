# L-TIME87H 音频母池

## 目标

为 `V4fg / V4Hg / V4PHg` 后续共同使用的长上下文数据轴建立一次冻结的 TIME87 长窗口母池。
本阶段只构建和审计 WAV 与窗口关系，不运行 Whisper、SOFA、tokenizer、GAME cache 或训练。

## 数据含义

`L-TIME87H` 保留完整 TIME87 短样本池，并追加由其中连续短段派生的 45--60 秒长窗口：

```text
TIME87 short: private corpus count 条
L87 long:      2,168 条
```

长窗口不增加独立录音内容，不能把其表观时长再次计入唯一 source 小时。正式报告同时保留三种
时长口径：

| 口径 | 小时 |
|---|---:|
| 冻结 TIME87 旧 30 秒 loader 有效时长 | 86.999726 |
| TIME87 短 WAV 实际帧时长 | 87.001480 |
| L87 派生长窗口 | 31.642634 |
| 短池与长池表观合计 | 118.644115 |
| 唯一 source 内容 | 87.001480 |

旧 loader 与完整 WAV 相差约 6.3 秒。未来 L 严格 loader 允许 60 秒，必须继续分栏报告，不能
静默把两种口径混为同一个 TIME 小时数。

## 构建合同

- 母池只读取冻结 TIME87 的 private corpus count 条 Path；
- 只合并同一 BV、文件编号连续的短段；
- 相邻短段 trim gap 不超过 0.5 秒；
- 窗口为 45--60 秒，非重叠、确定性贪心打包；
- 直接拼接 44.1kHz、单声道、PCM16 的原始帧；不重编码、不 crossfade、不增益处理、不加静音；
- 每个短段最多进入一个长窗口；
- 未来 `L30/L65` 必须从本母池过滤：一个窗口的全部底层短段均属于对应 TIME 集才可保留。

现有历史 L65 继续保留，不要求与新母池的窗口 ID 相同。新的 L30/L65/L87 家族以本母池的一次
切分为权威，从而保证子集关系可解释。

## 冻结输入

```text
TIME87 tokens:
${SERVER_ROOT}${CLOUD_ARTIFACT}
SHA256: redacted

trim manifest:
${SERVER_ROOT}/large_dataset/MSST_3step/trim_manifest.json
SHA256: redacted
```

本机旧 `dataset/trim_manifest.json` 只有约 4,500 条且 schema 不同，已明确排除；正式构建使用
服务器 21,935 条权威清单。

## 音频结果

服务器根目录：

```text
${SERVER_ROOT}/final_sum_large_V4L_TIME87H_45_60_gap05
```

结果：

| 项 | 值 |
|---|---:|
| 长窗口 | 2,168 |
| 使用的唯一短段 | 4,565 |
| TIME87 短段覆盖率 | 31.3401% |
| 长窗口帧数 | 5,023,584,650 |
| 长窗口 WAV 字节 | 10,047,264,692 |
| 45--50 秒 | 708 |
| 50--55 秒 | 739 |
| 55--60 秒 | 721 |
| 审计问题 | 0 |

窗口 manifest canonical SHA256: redacted

```text
5f0926a7be945d8721bc51a6b7797f328fd02f331adeb176c028f3d1438152b2
```

`audio_audit.json` 已逐条检查 TIME87 成员关系、窗口内编号连续、短段不重复、WAV 数量、格式、
帧数、缺失与额外文件，`issue_count=0`。

## 嵌套子集

L30/L65/L87 不重新打包音频，只按底层短段是否全部属于对应 TIME Path 集过滤母池：

| 集合 | 长窗口 | 使用短段 | 长窗口小时 |
|---|---:|---:|---:|
| L30 | 230 | 471 | 3.349609 |
| L65 | 1,257 | 2,628 | 18.303131 |
| L87 | 2,168 | 4,565 | 31.642634 |

独立报告确认 `TIME30 < TIME65 < TIME87`、`L30 subset L65 subset L87`、L87 等于母池且
`windows_repacked=false`，`issue_count=0`。权威派生目录为：

```text
${SERVER_ROOT}/final_sum_large_V4L_TIME87H_45_60_gap05/nested_subsets/
```

## 归档与交付

无重编码 TAR：

```text
V4L_TIME87H_audio_collection_20260806.tar
bytes: 10,070,589,440
SHA256: redacted
entries: 2,177
WAV entries: 2,168
```

cloud artifact

```text
${CLOUD_ARTIFACT}
```

上传使用服务器 Tickstep/aliyunpan v0.4.0 与 30MiB 分片。主 TAR 与 SHA256: redacted
上传退出码 0；cloud artifact

嵌套子集补充包：

```text
V4L_TIME87H_nested_subsets_20260806.tar.gz
bytes: 3,810,845
SHA256: redacted
```

本机通过实际挂载路径 `${CLOUD_MOUNT}` 回读。首次 robocopy
进程虽预分配出完整主 TAR 大小，但中途终止后的 SHA256: redacted
最终本机主 TAR SHA256: redacted
补充包 SHA256: redacted

本机正式交付与元数据位置：

```text
${LOCAL_PROJECT_ROOT}\TEMP\v4l_time87_audio_20260806\
${LOCAL_PROJECT_ROOT}\experiments\v4l_time87_20260806\formal_server\
```

## 工程位置

```text
${LOCAL_PROJECT_ROOT}\experiments\v4l_time87_20260806\
${LOCAL_PROJECT_ROOT}\tools\v4l\audit_v4l_time_family_audio.py
```

本机 dry-run 与服务器正式构建窗口数均为 2,168，但本机音频副本与服务器权威音频总帧相差约
0.19 秒，因此正式窗口 manifest 和时长以服务器结果为准。

## 下一门禁

1. 对 L87 重新运行 Whisper、SOFA 和 token overflow 门禁；
2. 用通过门禁的 L87 token 结果按冻结窗口 ID 派生 L30/L65，不重新对齐或重打包；
3. 冻结正式短长采样政策后构造训练 manifest；
4. 在绑定任何 `fg/hg/phg` 模型前，先完成 loader、单卡、DDP 与 exact-resume 门禁。

音频母池完成不代表训练已授权。

















