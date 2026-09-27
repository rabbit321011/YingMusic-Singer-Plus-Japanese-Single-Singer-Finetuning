> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# V5-P-Alex 数据接入说明

> 这是给数据准备者 Alex 的操作说明。训练程序只读取最终 token，不会重新执行 G2P 或对齐。

随附的 DataKit 只需要 Python 3.10+ 和 NumPy。它不需要 CUDA、PyTorch、YMSP/GAME 源码、
VAE 权重或训练 checkpoint。Alex 仍使用自己的英语 G2P/对齐和 MIDI 工具生成带时间信息的原始结果。

## 1. 你需要提交的文件

每条样本提交：

```text
audio/song_0001.wav
h_tokens/song_0001.npy
midi_p_tokens/song_0001.npy
manifest.jsonl
```

不要提交 phone interval、note event、GAME posterior 或 CKA target 作为训练输入。

## 2. 音频和 VAE

音频必须是：

```text
sample rate: 44100 Hz
channels: mono
format: PCM16 WAV
```

令 `N` 为 WAV header 中的 sample-frame count。冻结的 V5-P-Alex 合同使用：

```text
T = N // 2048
```

这是当前 YMSP official VAE 在上述音频格式下的精确长度公式，不是按 20 Hz 或 100 Hz 的估算。
可以直接运行 `python tools/timebase.py audio/song_0001.wav` 获取 `N` 和 `T`。Alex 不需要安装
或运行 VAE；训练服务器会再用真实 VAE 严格复核一次。若音频格式或 VAE 改变，不能沿用此公式。

最终必须满足：

```text
len(h_tokens)      == T
len(midi_p_tokens) == T
每个 sentence_start_frames 元素都在 [0, T-1]
```

## 3. H token（文本条件）

H token 是长度为 `T` 的一维整数数组：

```text
h_tokens: int64[T]
```

### ID 规则

本目录附带 `vocab.json`。它把文本音素符号映射为 0-based 原始 ID。

```text
base_id = vocab.json[token]
H_token_id = base_id + 1
```

示例：

```text
vocab.json["ɪ"] = 6  -> H token 7
vocab.json["p"]  = 27 -> H token 28
vocab.json["t"]  = 29 -> H token 30
```

特殊 H token 不使用 `+1`：

```text
0       = filler / empty frame
365     = SEP
366     = PUL
```

不要创建新的 vocab，也不要手动猜测普通文本 token ID。

### H token 生成

你的离线预处理可以使用 YMSP CNEN G2P 和英语对齐工具，但训练端不会再次运行它们：

```text
English lyrics
  -> CNEN G2P
  -> English alignment
  -> text token events with timestamps
  -> H token IDs using vocab.json
  -> dense h_tokens[T]
```

初始数组全部填 `0`。每个文本事件放到对应的 latent frame；SEP 使用 `365`，PUL 使用 `366`。
普通文本 token 的顺序必须保持。多个事件不能静默覆盖；必须使用确定性的冲突处理并保证结果可复现。

### 句首帧数组

每条样本还必须在 manifest 中提供：

```text
sentence_start_frames: [s0, s1, ..., sN-1]
```

它不是毫秒或秒数组。每个 `s_i` 都是 official VAE 时间轴上“第几帧”的整数索引，并且：

1. 数组与句子顺序一致，非空且严格递增；
2. `h_tokens[s_i]` 是该句第一个普通 H token，不是 `0`、SEP 或 PUL；
3. 每句恰好有一个 SEP，因此 SEP 数量等于数组长度；
4. 第 `i` 句的 SEP 位于 `s_(i+1)-1`，最后一句的 SEP 位于 `T-1`；
5. 至少有一个句首帧大于 `0`，供训练端切分 A/B。

H 必须按音频的绝对时间生成，不能先假设一个 A/B 边界再移动整句。训练端会用这个数组选择
不会切进句子内部的 reference boundary。

## 4. MIDI-P token（旋律条件）

MIDI-P 是长度为 `T` 的一维整数数组：

```text
midi_p_tokens: int16[T]
```

ID 规则：

```text
0..254 = 0.5-semitone pitch class
255    = REST
256    = PAD
```

音高转换：

```text
pitch_class = round(MIDI_pitch * 2)
C5 (MIDI 72) -> 144
D5 (MIDI 74) -> 148
```

如果你的内部工具用 `0` 表示静音，导出前必须转换为 `255`。最终文件只保存整数 token，不保存
`C5`、`D5` 或 note event。

## 5. 时间戳到 frame

H 和 MIDI-P 使用相同的 latent 时间轴。设：

```text
D = 音频时长（秒）
T = WAV sample-frame count // 2048
```

第 `i` 帧的中心时间为：

```text
center_i = (i + 0.5) * D / T
```

带时间戳的事件使用距离最近的 frame center：

```text
frame_index = argmin_i(abs(center_i - event_time))
```

例如 `D=2.0` 秒、`T=43` 帧时，约 `0.20` 秒的事件落在 frame `4`。

每句开始时间使用同一规则转换，并把结果写入 `sentence_start_frames`。不要把原始秒数或毫秒数
写入该字段。

MIDI 音符持续一段时间时，将对应 pitch class 填入该时间段内所有 frame center；没有音高的
frame 使用 `255`。

## 6. 目录和 manifest

```text
v5p_alex_data/
├── audio/
│   └── song_0001.wav
├── h_tokens/
│   └── song_0001.npy
├── midi_p_tokens/
│   └── song_0001.npy
├── manifest.jsonl
└── vocab.json
```

manifest 示例：

```json
{
  "schema": "v5p_alex_input_v2",
  "sample_id": "song_0001",
  "language": "en",
  "audio_path": "audio/song_0001.wav",
  "audio_sha256": "...",
  "audio_frames": 546840,
  "duration_seconds": 12.4,
  "vae_contract": "ymsp_official_vae_44100_ratio2048_v1",
  "vae_frame_count": 267,
  "sentence_start_frames": [18, 93, 171],
  "h_tokens_path": "h_tokens/song_0001.npy",
  "h_tokens_sha256": "...",
  "midi_p_tokens_path": "midi_p_tokens/song_0001.npy",
  "midi_p_tokens_sha256": "..."
}
```

必须满足：

```text
vae_frame_count == len(h_tokens) == len(midi_p_tokens)
```

DataKit 提供两组样例：

```text
example/                         一条无版权的 1 秒合成自检样本
examples/hanamaru_10_v2/         十条真实花丸 v2 格式样本
```

花丸样本来自已审计的服务器 fixture，包含 10 条音频、4,776 个 latent frame 和 70 个句首；它们
用于展示真实的音频/H/MIDI-P/句首/SHA256 对应，不是英语训练数据，也不得混入 Alex 的英语集。
目录内保留服务器原始 `source_manifest.jsonl`，DataKit 的 `manifest.jsonl` 只增加日语示例标记和
冻结 VAE 合同，不改变资产、句首或哈希。使用
`tools/create_manifest.py` 生成 manifest，不要手写；最后运行
`tools/validate_dataset.py path/to/dataset/manifest.jsonl --output validation.json`。

## 7. 训练端会做什么

训练端读取 H token 和 MIDI-P token，使用官方 VAE 处理音频；CKA target 由 frozen GAME 在训练
时实时提取。你不需要提供 CKA 数据。

每个 step 中，训练端先按 V5-P 规则随机提出 reference 长度，再强制吸附到最近的正数
`sentence_start_frames`。因此最终 A/B 边界一定是你提供的某个句首帧，不会从一句中间切开。

## 8. 交付前检查

- 音频格式正确且 SHA256 正确；
- H token 长度等于官方 VAE 的 `T`；
- `sentence_start_frames` 是严格递增的 VAE 帧索引，并与每句第一个 H token 对应；
- SEP 数量和位置与句首帧数组一致；
- MIDI-P token 长度等于同一个 `T`；
- H token 使用 `vocab.json` 加 1 的规则；
- H 的 `0`、`SEP=365`、`PUL=366` 使用正确；
- MIDI-P 的 `255=REST`、`256=PAD` 使用正确；
- 所有 manifest 路径都存在；
- 不存在静默覆盖或随机生成的 token。
