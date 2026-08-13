# L-FULL-LENGTH-TIME 音频母池

## 决策

历史 `L-TIME87H` 只包含 45--60 秒窗口，且直接拼接短 WAV 时压缩了 trim 时间轴中的小间隔。
它继续作为已审计历史资产保留，但不再是 V5 的正式 L 音频母池。

V5 使用新的 `L-FULL-LENGTH-TIME`：覆盖 30--60 秒，优先把窗口延伸到接近 60 秒，并把原 trim
时间轴中被裁去的正间隔恢复成等长静音。

## 冻结输入

```text
TIME87 tokens:
${SERVER_ROOT}${CLOUD_ARTIFACT}
SHA256: redacted

trim manifest:
${SERVER_ROOT}/large_dataset/MSST_3step/trim_manifest.json
SHA256: redacted
```

旧池和新池使用相同的 private corpus count 条 TIME87 Path 范围。1,057 条没有权威 keep trim row 的 TIME87
记录仍计入唯一数据小时，但不能进入依赖原 trim 时间轴的 Long 窗口。

## 构建合同

1. 只连接同一 BV、编号连续的短段；输入为冻结 train split，不跨 split。
2. 相邻段原 trim gap 必须严格 `<0.5s`；`gap == 0.5s` 断开。
3. 正 gap 按 `round_half_up(gap * 44100)` 转为 PCM 零帧，插入下一段之前。
4. `-1ms` 以内的时间量化重叠不插负静音，也不裁掉源 WAV 帧。
5. 窗口总长包含恢复静音，范围为 30--60 秒。
6. 每个连续 run 从当前 cursor 开始持续加入下一段，直到再加入一段会超过 60 秒；达到 30 秒后
   输出该最长合法窗口，再从未使用的下一段继续。
7. 每个窗口至少包含两个短段；单条已有 30 秒 WAV 不登记为 Long。
8. 每条短段最多进入一个 Long；不重采样、不 crossfade、不改增益。

## 正式结果

服务器根目录：

```text
${SERVER_ROOT}/final_sum_large_V4L_FULL_LENGTH_TIME_30_60_gaplt05_silence
```

| 项 | 值 |
|---|---:|
| Long 窗口 | 3,373 |
| 使用唯一短段 | 7,003 |
| TIME87 短段覆盖率 | 48.0777% |
| 两段窗口 | 3,116 |
| 三段窗口 | 257 |
| 30--35 秒 | 276 |
| 35--40 秒 | 494 |
| 40--45 秒 | 608 |
| 45--50 秒 | 636 |
| 50--55 秒 | 654 |
| 55--60 秒 | 705 |
| 30--45 秒合计 | 1,378 |
| 45--60 秒合计 | 1,995 |
| 输出表观时长 | 44.082823h |
| 其中恢复静音 | 0.147497h |
| 窗口内源音频 | 43.935326h |

窗口 manifest canonical SHA256: redacted

```text
e21306eb2c6016caf03a1fab6cb393eb3b8682df3922c08cf8223a2cc2ba7133
```

## 审计

独立审计逐条重建来源索引与原始 gap，检查成员、连续编号、严格 gap、静音帧、最长可扩展性、
来源不重复、WAV 格式和精确帧数；另逐块验证 3,373 条输出中的恢复区间全部为零 PCM，所有源
短段 PCM 字节未改变。正式结果 `issue_count=0`；3,373 条 manifest 与 3,373 条 WAV 完全对应。
`windows.json` 文件 SHA256: redacted

```text
71e3d32a727b2f1fe5d398257fa5c16aa36363425e3e1bd8c1ed4fe2f9abaecb
```

## 嵌套子集

沿用同一母池按底层 Short 成员关系过滤，不重新打包：

| 集合 | Long | 使用短段 | Long 时长 |
|---|---:|---:|---:|
| L30 | 402 | 818 | 5.120623h |
| L65 | 1,996 | 4,122 | 25.849794h |
| L87 | 3,373 | 7,003 | 44.082823h |

独立报告确认 `TIME30 < TIME65 < TIME87`、`L30 subset L65 subset L87`、L87 等于母池、
`windows_repacked=false`，`issue_count=0`。

## 工程位置

```text
${LOCAL_PROJECT_ROOT}\tools\v4l\build_v4l_full_length_time.py
${LOCAL_PROJECT_ROOT}\tools\v4l\audit_v4l_full_length_time.py
${LOCAL_PROJECT_ROOT}\experiments\v4l_full_length_time_20260808\
```

## 归档

无重编码 TAR 已上传cloud storage：

```text
cloud: ${CLOUD_ARTIFACT}
archive: V4L_FULL_LENGTH_TIME_audio_collection_20260808.tar
bytes: 14,087,454,720
SHA256: redacted
WAV entries: 3,373
upload exit: 0
```

云端同目录另存最终增强版 `audio_audit.json`、`nested_subsets_audit.json`、`COLLECTION.md` 和
`formal_metadata.sha256: redacted`。最终 audio audit SHA256: redacted
`25cd2ad4ddcd7eaaf8c064010b1162e7193a0b6f2db3bbd4b21a0af866ba7839`。

旧 `L-TIME87H` 音频与文档不删除。V5 后续 Whisper、SOFA、H/PUL、GAME 和 token 资产必须以本
文的新 WindowId 集为输入，不能继续沿用旧 2,168 条窗口。

















