> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# GTSinger (JP) + PJS 数据集获取与预处理指南

> 用于另一个 agent 执行。目标：下载日语歌唱数据集并转换为 YingMusic-Plus V4 训练可用格式。

## 数据集概览

| 数据集 | 链接 | 许可 | 内容 |
|------|------|------|------|
| GTSinger JP | https://huggingface.co/datasets/GTSinger/GTSinger | CC BY-NC-SA 4.0 | 多语言 80h，日语子集待提取 |
| PJS | https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus | CC BY-SA 4.0 | 声优统计 corpus 歌唱版，音素平衡 |

## 第一步：下载

### GTSinger (日语子集)
```bash
# 方式1: HuggingFace CLI
pip install huggingface_hub
huggingface-cli download GTSinger/GTSinger --repo-type dataset --local-dir ./GTSinger

# 方式2: Python
from huggingface_hub import snapshot_download
snapshot_download("GTSinger/GTSinger", repo_type="dataset", local_dir="./GTSinger")

# 日语数据筛选：metadata中 language="ja" 的条目
# 文件结构：GTSinger/{singer_name}/{song_name}/*.wav + *.TextGrid + *.musicxml
```

### PJS
```bash
# 从官网下载
wget https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus

# 或 Google Drive 链接（官网提供）
# 包含：wav/ singing/ + parallel/ + labels/
```

## 第二步：数据格式转换

### 目标训练格式 (YingMusic-Plus V4)
每首歌唱片段需要：

```json
{
  "Path": "/path/to/vocal.wav",
  "Duration": 12.5,           // 秒
  "Text": "桜 の 花びら が",   // 空格分隔短语
  "Language": "ja",
  "Phrases": [                 // 句级信息
    {
      "text": "桜",
      "kana": "さくら",
      "start": 0.0,            // 秒
      "end": 2.3,
      "tokens": [32, 56, ...]  // G2P token IDs
    }
  ],
  "full_tokens": [32, 56, ..., 365, 45, ...]  // 含 SEP=365
}
```

完整流水线参考：`docs/DATA_PIPELINE_Timeset.md`

### GTSinger 格式说明
- **音频**：WAV 文件（高品 44.1kHz 或 48kHz）
- **时间戳**：TextGrid 格式（Praat），**手动标注的音素级对齐**
  - 需要从 TextGrid 提取 phoneme tier 的 intervals
  - 可能有多层 tier（phoneme / word / phrase）
- **乐谱**：MusicXML（F0/MIDI 提取用，可选）

### PJS 格式说明
- **音频**：48kHz WAV
- **标签**：.lab 文件（HTS 格式）
  ```
  0.0000000  0.5000000  sil
  0.5000000  0.8000000  k
  0.8000000  1.2000000  a
  ```
  起止时间 + 音素标签，**精确到音素级**
- **MIDI**：同步的 MIDI 文件

## 第三步：预处理关键步骤

```
WAV 音频 → MSST 人声分离（如含伴奏）→ 44.1kHz resample → train_singnet.json
    ↓
TextGrid/lab → 提取 phoneme intervals → 合并为句级短语 → 生成 Phrases[]
    ↓
歌词文本 → CNENTokenizer.encode() → token IDs → full_tokens（含 SEP=365）
    ↓
检查 token 数 < 对应帧数（避免溢出）
```

### 辅助脚本（已有，可直接复用）
- `docs/temp_0529/preprocess_tokens.py` — G2P + 溢出过滤 + JSON 生成
- `scripts_archive/data_pipeline/0_msst/` — MSST 人声分离
- `scripts_archive/data_pipeline/1_whisper/` — Whisper 转录（如果 GTSinger/PJS 已有歌词可跳过）

### G2P Token 注意事项
- Tokenizer：`src/YingMusicSinger/utils/cnen_tokenizer.py` 的 `CNENTokenizer`
- `encode(text)` 返回 1-based token IDs（1~363）
- SEP token ID = 365（`tokenizer.phone2id["<SEP>"]`）
- 完整 token 对齐表：`docs/VOCAB_TOKEN_MAP.md`
- **G2P 一致性测试**：`test_ja_g2p.py` — 处理前必须跑通

### 数据分割
- Train: 90%（按歌手/曲目隔离，避免同曲不同段泄漏）
- Test: 10%（用于 WER 评估）
- 格式：train_singnet.json / test_singnet.json（与现有格式一致）

## 第四步：质量检查清单

- [ ] 所有音频为 44.1kHz 单声道/立体声 WAV
- [ ] 每条有完整的 Phrases[]（含 start/end/tokens）
- [ ] full_tokens 长度不超过 latent 帧数（max 30s × 21.53fps ≈ 646帧）
- [ ] G2P token 一致性通过（`test_ja_g2p.py`）
- [ ] Train/Test 按歌手/曲目隔离
- [ ] 无空短语、无 token 溢出样本

## 产出文件

最终放在 `dataset/gtsinger_pjs_jp/` 目录下：

```
dataset/gtsinger_pjs_jp/
├── metadata/                      # 原始元数据
│   ├── gtsinger_ja_metadata.json
│   └── pjs_metadata.json
├── audio/                         # 处理后的音频
│   ├── gtsinger_xxx.wav
│   └── pjs_xxx.wav
├── train_singnet.json             # 训练集
├── test_singnet.json              # 测试集
└── run_preprocess.log             # 处理日志
```

成功后通知我，我来写 V4d 训练脚本适配新数据路径。
