> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# V5-P-Alex 数据接入 DR

> DR = Design Review。本文是内部设计稿，不是直接发给 Alex 的操作文档。

## 1. 目标与边界

V5-P-Alex 的训练入口只接受 Alex 已经生成好的条件数据：

```text
audio
h_tokens
midi_p_tokens
manifest（含 sentence_start_frames）
```

训练端不执行 G2P、SOFA、phone alignment 或 note-event 解析。Alex 在训练端之外完成英语
歌词处理和 MIDI 预处理，训练端只读取最终 token。

CKA target 不属于 Alex 的交付物，由训练端从音频实时提取；这不是本 DR 的数据输入字段。

## 2. 两套独立的 ID 系统

### 2.1 H token（文本条件）

```text
0       = filler / PAD
1..363  = ordinary text token
365     = SEP
366     = PUL
```

普通文本 token 使用绑定的 V5-P `vocab.json` 原始 ID 加 1。H token 是 dense frame-level
序列，不是 phone 列表。

`sentence_start_frames` 是同一 VAE 时间轴上的整数句首帧数组，不是秒或毫秒。它是固定 dense H
与训练时随机 A/B reference 能同时成立所必需的边界元数据。

### 2.2 MIDI-P token（旋律条件）

```text
0..254 = 0.5-semitone pitch class
255    = REST
256    = PAD
```

音高映射固定为：

```text
class = round(MIDI pitch * 2)
C5 (MIDI 72) -> 144
D5 (MIDI 74) -> 148
```

H token 的 `0` 与 MIDI-P token 的 `0` 完全不是同一含义。MIDI-P 的静音必须使用 `255`。

## 3. VAE 与帧数权威

统一使用：

```text
YMSP official VAE configuration
YMSP official VAE checkpoint
```

对每条音频先得到：

```text
audio -> official VAE -> latent [T, D]
```

`T` 是实际 latent frame count，不按照固定 20 Hz 或 100 Hz 猜测。最终必须满足：

```text
len(h_tokens)      == T
len(midi_p_tokens) == T
```

H 与 MIDI-P 共用同一条 latent 时间轴。

## 4. 时间戳到 frame 的规则

设音频时长为 `D` 秒、latent 帧数为 `T`，第 `i` 帧的中心时间为：

```text
center_i = (i + 0.5) * D / T
```

带时间戳的事件落到距离最近的 frame center：

```text
frame_index = argmin_i(abs(center_i - event_time))
```

索引必须位于 `[0, T-1]`。

### H token 栅格化

Alex 的内部流程可以产生带时间的文本事件，但交付前必须栅格化为：

```text
h_tokens: int64[T]
```

规则：

1. 初始全部填 `0`；
2. 文本 token 按事件时间放入对应 frame；
3. SEP 使用 `365`，PUL 使用 `366`；
4. 未占用帧保持 `0`；
5. 普通文本 token 的顺序必须保持；
6. 冲突处理规则必须固定并可复现。

每句第一个普通 H token 的 frame 必须等于 `sentence_start_frames` 中对应元素。每句恰好一个
SEP：下一句存在时放在下一句开始帧减一，最后一句放在 `T-1`。数组必须严格递增，并至少包含
一个大于 0 的帧。

训练端先按原 V5-P 比例提出随机 reference，再无条件吸附到最近的正数句首帧。这里不沿用旧
V5-P 的 2.5 秒 margin：dense H 已经固定，允许 boundary 留在句中会重新引入已实证的语义错位。

### MIDI-P 栅格化

Alex 最终交付：

```text
midi_p_tokens: int16[T]
```

音高持续覆盖哪些 frame，就在这些 frame 重复对应的 pitch class。无音高区域使用 `255`，
序列尾部 padding 使用 `256`。

不交付 note-event 文件作为训练输入。

## 5. 文件与 manifest

推荐目录：

```text
v5p_alex_data/
├── audio/
│   └── song_0001.wav
├── h_tokens/
│   └── song_0001.npy
├── midi_p_tokens/
│   └── song_0001.npy
└── manifest.jsonl
```

音频要求：44.1 kHz、mono、PCM16 WAV。

manifest 最少包含：

```json
{
  "schema": "v5p_alex_input_v2",
  "sample_id": "song_0001",
  "audio_path": "audio/song_0001.wav",
  "audio_sha256": "...",
  "duration_seconds": 12.4,
  "vae": "YMSP official VAE",
  "vae_frame_count": 267,
  "sentence_start_frames": [18, 93, 171],
  "h_tokens_path": "h_tokens/song_0001.npy",
  "midi_p_tokens_path": "midi_p_tokens/song_0001.npy"
}
```

## 6. Alex 的最小交付物

```text
audio/*.wav
h_tokens/*.npy
midi_p_tokens/*.npy
manifest.jsonl
```

Alex 不需要交付：

```text
phone interval
note event
GAME posterior
CKA target
VAE latent
```

## 7. 接口冻结条件

只有在以下问题确认后，才能生成正式 DES 和训练入口：

- Alex 能否使用 YMSP official VAE 得到准确的 `T`；
- H token 是否能直接输出为 V5-P 训练 ID；
- 句开始时间是否能输出为 official VAE 的严格递增 frame index 数组；
- MIDI-P 是否能直接输出为 `[T]` 整数序列；
- Alex 的冲突处理、REST/PAD 和 token 栅格化规则是否确定；
- manifest 是否能绑定音频 SHA、`T` 和两个 token 文件。
