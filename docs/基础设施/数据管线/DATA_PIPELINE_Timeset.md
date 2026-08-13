# V4 时间戳对齐工作记录

> 状态：第1步 ✅ | 第2步 ✅ | 第3步 ✅ | 第4步 ✅ | 第5步 待定

---

## 一、问题定义

DiT 训练需要**句级时间戳**将歌词 token 逐帧放置到 VAE latent 帧序列上。当前 `train_singnet.json` 有文本无时间戳，需要为每句歌词获取音频中的起始时间。

- **本质**：Forced Alignment（强制对齐），不是 ASR
- **已知**：准确歌词文本（空格分隔短句）
- **未知**：每句话在音频中的起始时间
- **目标**：P95 误差 ≤ ±0.5s

## 二、数据概况

| 项目 | 数值 |
|------|------|
| 数据文件 | `dataset/final_sum_large/train_singnet.json` |
| 样本数 | 15,122 |
| 音频格式 | 短分段（seg），非整首歌 |
| 时长 | min=1.2s, max=31.1s, median=23.2s |
| 短句数 | min=1, max=81, median=5 |
| 语言 | ja（日语） |
| 人声分离 | ✅ 已完成（去伴奏） |
| B站字幕 | ❌ 无 |
| LRC 文件 | ❌ 无 |

## 三、方案决策

> 基于 GPT-5.5 / Gemini 3 Pro / Claude Opus 4.6 三家分析后决策

### 3.1 核心对齐方案（全测）

| # | 方案 | 决定 | 备注 |
|---|------|:--:|------|
| A1 | CTC-Segmentation + wav2vec2-ja | Test → ❌ | Claude 最推荐，但域不匹配（语料→歌唱），输出大量 `<unk>` |
| A2 | MFA + japanese_mfa | Test → ❌ | HMM 音素模型基于语料，唱歌拖长音被压缩，误差逐句累积 |
| A3 | WhisperX align mode | Test → ⚠️ | 有边界时 P95=1.27s（最优），但全量数据无边界信息 |
| A4 | Qwen3-ForcedAligner-0.6B | Test → ❌ | LLM-based NAR，支持日语但结果极度不稳定 |
| A5 | ReazonSpeech / Kotoba-Whisper | Test → ❌ | 日语专精 ASR，唱歌音频识别崩溃（sim 0.00~0.76） |
| A6 | Torchaudio forced_align | Test → ⚠️ | MMS-FA，与 A3 相同需要边界，无边界时塌缩 |
| **A7** | **ASR + fuzzy match** | **Test → ✅** | **Whisper large-v3 ASR + sim 分档，最终方案** |

### 3.2 预处理方案

| # | 方案 | 决定 | 备注 |
|---|------|:--:|------|
| B1 | 人声分离（Demucs/UVR5） | ❌ 放弃 | 数据已去伴奏 |
| B2 | 文本汉字→假名 G2P | ✅ Keep & Test | pykakasi / pyopenjtalk |

### 3.3 兜底方案

| # | 方案 | 决定 | 备注 |
|---|------|:--:|------|
| C1 | VAD 启发式切分 | Test | silero-vad 检测停顿后按句数分配 |
| C2 | 均匀分布兜底 | ❌ 放弃 | 短分段上误差过大 |

### 3.4 质检方案

| # | 方案 | 决定 | 备注 |
|---|------|:--:|------|
| D1 | 自动质检打分 | Test | 句时长/重叠/覆盖率规则引擎 |
| D2 | 人工 Gold Set | ✅ Keep | 100 条手工标注真实时间戳 |
| D3 | 分层抽样校验 | ✅ Keep | 按时长/句数/曲风分层 |

## 四、执行计划

```
第1步：D2 人工 Gold Set ✅
  └─ 100 条已标注，见 tools/gold_annotator/gold_set_100_annotated.json
     标注工具：tkinter GUI（tools/gold_annotator/annotator_gui.py）
     标注内容：句级 start 时间戳（±0.1s 精度）
     日语音拍级罗马音辅助显示（pykakasi）

第2步：B2 文本规范化 ✅
  └─ pyopenjtalk 汉字→平假名（98.4% 纯日文）
      alkana 英文→片假名→平假名（1.6% 纯英文，~80%命中）
      LLM 补丁覆盖 alkana 盲区（0.78%短语，420条）
      输出：train/test/gold_set *_kana.json

第3步：A1~A7 评测 ✅
  └─ 每个方案跑同一批 Gold Set（100条），用 MAE/P95 误差排名
      A1 CTC-Seg + wav2vec2-ja ❌ 
        → wav2vec2-ja 对歌唱音频 greedy decode 大量 <unk>，字符级时间戳全挤在 0~0.26s
        → 结论：域不匹配（语料模型 → 歌唱），不可用
      A2 MFA + japanese_mfa ❌
        → 100条全测：P50=3.62s P95=9.37s，首句MAE=0.54s但误差逐句累积
        → 根因：HMM 音素时长模型基于语料，唱歌拖长音被压缩
        → 结论：不可用
      A3 WhisperX forced alignment ⚠️ 需边界
        → 策略：wav2vec2-large-xlsr-53-japanese CTC强制对齐，绕过ASR
        → 测试一（金标边界引导）：99/100成功，MAE=0.25s P50=0.04s P95=1.27s，87%<0.5s ✅
        → 测试二（完全不估计边界）：94/100成功，MAE=1.95s P50=0.67s P95=10.02s，18.4%≥3s ❌
        → 结论：强制对齐模型本身对歌声有效，但必须提供粗粒度边界，否则长尾崩盘
        → 问题：全量数据集无边界信息，无法直接使用
      A4 Qwen3-ForcedAligner-0.6B ❌
        → 策略：LLM-based NAR时间戳预测器，支持11语言含日语，不需边界
        → 10条抽样：MAE=2.90s P50=0.77s P95=13.97s，极度不稳定
        → 速度极快（CPU 0.7-2.6s/条），但不可预测哪些样本会失败
        → 结论：不可靠，不适合全量15K条
      A5 ReazonSpeech / Kotoba-Whisper ❌
        → 思路：日文专精ASR转录唱歌音频 → 模糊匹配已知歌词定位短语边界
        → Kotoba-Whisper-v2.0 无法下载（网络不通），改用已缓存的reazon wav2vec2（387MB）
        → Reazon wav2vec2 对唱歌ASR质量两极分化：好时 sim=0.76，差时 sim=0.00（完全崩坏）
        → 结论：wav2vec2小模型不适合唱歌ASR，不可用
      A6 Torchaudio forced_align (MMS-FA) ⚠️ 需边界
        → 策略：Meta MMS-FA 多语言强制对齐模型（1.2GB），kana→romaji→tokenize→CTC对齐
        → 5条测试：MAE=12.21s P50=10.33s，75.9%短语误差>3s
        → 与A3完全相同的CTC对齐边界缺失问题，换了模型无济于事
        → 结论：需要边界引导才能工作
      A7 Whisper large-v3 ASR + fuzzy match ✅ 最终方案
        → 策略：faster-whisper large-v3 (GPU) ASR转录 → difflib模糊匹配歌词 → sim分档
        → 100条全测：匹配率85.5% (583/682短语), MAE=1.84s P50=1.14s P95=6.62s
        → 速度：GPU set.7s/条 (vs CPU 10s+), 3条VAD过滤跳过
        → 成功样本sim通常>0.7, 部分完美(sim=1.0, 数十条P95<2s)
        → 长尾问题：16.1%短语>3s，主要由极少数长句串烧拉高
        → A7→A3 混合管线（A7粗边界驱动A3精细对齐）也测试过，P50改善33%但P95几乎不变
        → 混合管线性价比太低（CPU 166s多救4条），果断放弃，只跑纯A7 + 规则分档
        → 结论：无边界方法中最优，通过 sim+句数 规则分档实现可信度分层
      ---
      ### 第3步总结
        A1-A7 全部测试完毕。核心矛盾：「有边界→模型能对齐(P95<1.3s)」vs「全量15K数据无边界」。
        - 有边界最优：A3 WhisperX forced alignment, P95=1.27s（但无法用于全量）
        - 无边界最优：A7 Whisper ASR+fuzzy match, P95=6.62s（可用但要降假消息率）
        - 解法：A7 跑全量 + sim/句数规则分4档，牺牲总量保精度

第4步：全量处理 ✅
  └─ 见下面「五、最终方案与流水线」

第5步：D1 自动质检 + D3 分层校验（可选）
  └─ 目前用 sim 分档已实现自动可信度分层，如需更细粒度质检可额外跑 D1
```

## 五、最终方案与流水线

### 5.1 选型结论

| 组件 | 选型 | 原因 |
|------|------|------|
| ASR 模型 | faster-whisper large-v3 (Systran) | 3GB，日语唱歌识别远超 wav2vec2 系 |
| 对齐方式 | ASR转录 → difflib模糊匹配 → 词级时间戳映射 | 无边界，直接可用 |
| 可信度信号 | `sim`（ASR输出与歌词的文本相似度，0~1） | 与精度强相关 |
| 分档规则 | sim≥0.9 且 句数≤10 → L1；sim≥0.7 → L2；sim≥0.5 → L3；其余 → L4 | 100条Gold Set验证得出 |
| 运行环境 | 服务器 6×RTX 4090, CUDA 12.8, yysinger conda env | 并行处理 |

### 5.2 处理流水线

```
输入：
  train_singnet_kana.json (15,122条) ─── 音频 + 假名歌词
  test_singnet_kana.json  (307条)    ─── 音频 + 假名歌词

处理流程（per sample）:
  1. torchaudio 加载音频 (16kHz)
  2. faster_whisper.large-v3.transcribe()
     ├─ word_timestamps=True  → 词级时间戳
     ├─ vad_filter=True       → 自动过滤静音
     └─ language='ja'         → 日语模式
  3. 拼接 ASR 词 → 全文 ASR 文本
  4. difflib.SequenceMatcher(ASR文本, 歌词文本).ratio() → sim 分数
  5. difflib 匹配块跟踪 → 将词级时间戳映射到歌词短语
     算法: SequenceMatcher.get_opcodes() → 遍历 equal/replace 块
           → 字符级位置 → word_positions 二分查找 → (start, end)
  6. 分档: sim>=0.9 且句数<=10 → L1_high
           sim>=0.7             → L2_medium
           sim>=0.5             → L3_low
           其余                  → L4_discard

输出: 8个JSON文件（train/test × 4档）
```

### 5.3 运行记录

| 项目 | 详情 |
|------|------|
| 环境 | 6×RTX 4090, tmux session 1.2.1, torch 2.9.1+cu128) |
| 并行策略 | 15,122条 train 拆分为 6 份（每GPU ~2520条），6卡并行 |
| 耗时 | train ~20分钟（6GPU），test ~4分钟（1GPU） |
| 数据中转 | cloud storage `TEMP/a7_timeset_package`（输入30MB）、`TEMP/a7_timeset_output.tar.gz`（输出7MB） |
| 脚本位置 | `tools/alignment/a7_full_parallel.py`（并行处理）、`merge_chunks.py`（合并分片）、`launch_6gpu.sh`（tmux session 最终产出

| 文件 | 大小 | 条数 | sim范围 | 用途 |
|------|------|-----:|---------|------|
| `train_L1_high.json` | 12.3MB | 7,725 (51%) | ≥0.9 + 句≤10 | **直接训练**，假消息率 13% |
| `train_L2_medium.json` | 8.0MB | 3,924 (26%) | ≥0.7 | drop训练 / 后训练 |
| `train_L3_low.json` | 1.3MB | 967 (6.4%) | ≥0.5 | 低权重辅助训练 |
| `train_L4_discard.json` | 1.9MB | 2,243 (15%) | <0.5 | 丢弃 |
| `test_L1_high.json` | 0.2MB | 131 (43%) | ≥0.9 + 句≤10 | 评测 |
| `test_L2_medium.json` | 0.2MB | 82 (27%) | ≥0.7 | 评测 |
| `test_L3_low.json` | 0.1MB | 32 (10%) | ≥0.5 | 评测 |
| `test_L4_discard.json` | 0.1MB | 60 (20%) | <0.5 | — |
| **合计 train** | **23.5MB** | **14,859** | — | 覆盖率 98.3% |
| **合计 test** | **0.6MB** | **305** | — | 覆盖率 99.3% |

## 六、输出格式

每行为一个完整的 JSON 对象：

```json
{
  "Path": "sample_media.wav",
  "Duration": 11.239,
  "Text": "...",
  "Language": "ja",
  "Phrases": [
    {
      "text": "たぶん私じゃなくていいね",
      "kana": "たぶんわたしじゃなくていいね",
      "start": 0.020,
      "end": 3.150,
      "match_quality": "equal"
    }
  ],
  "ASRSim": 1.0,
  "ASRText": "たぶん私じゃなくていいね...",
  "Tier": "L1_high",
  "AlignmentMethod": "A7_whisper_largev3",
  "AlignmentStatus": "ok"
}
```

**新增字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `Phrases[].start` | float | 短语起始时间（秒） |
| `Phrases[].end` | float | 短语结束时间（秒） |
| `Phrases[].match_quality` | string | difflib 匹配类型（equal/replace） |
| `ASRSim` | float | ASR 文本与歌词的相似度 (0~1) |
| `ASRText` | string | Whisper 转录的原始文本 |
| `Tier` | string | 可信分档（L1_high/L2_medium/L3_low/L4_discard） |
| `AlignmentMethod` | string | 对齐方法标识 |
| `AlignmentStatus` | string | ok/discard |

## 七、训练使用指南

### L1_high（直接训练）
- sim≥0.9 且句数≤10，假消息率 13%
- 7,725 条，覆盖 51% 的训练数据
- **直接用**：时间戳可信度高，可作为主训练集
- 剩余的 4 条假消息都是 MAE/P95 略超阈值的边缘样本，并非完全错误

### L2_medium（drop训练 / 后训练）
- sim≥0.7，缺乏 L1 级别的精度但方向正确
- 偏差主要来自 ASR 时间戳漂移，不是文本错误
- 作为 dropout 数据：用较低置信度权重参与训练
- 或用于后续 fine-tuning 阶段的辅助数据

### L3_low（低权重）
- sim≥0.5，仅 967 条（6.4%）
- 偏差较大但至少 ASR 方向正确
- 可用极低权重参与训练，或仅用于 MIDI 控制场景（MIDI note onset 覆盖时间戳偏差）

### L4_discard（丢弃）
- sim<0.5，Whisper 完全幻觉（输出 `ご視聴ありがとうございました` 等无关文本）
- 2,243 条（15%）
- 直接丢弃，不参与任何训练

## 八、更新日志

| 日期 | 内容 |
|------|------|
| 2026-05-27 | 初始化，完成方案调研与决策 |
| 2026-05-27 | 第1步完成：开发 tkinter GUI 标注工具，100 条 Gold Set 全部手工标注完成 |
| 2026-05-28 | 第2步完成：B2 文本规范化（pyopenjtalk + alkana + LLM），产出 train/test/gold_set *_kana.json |
| 2026-05-28 | 第3步完成：A1-A7全部评测完毕，A7(ASR+fuzzy)为最优无边界方案 |
| 2026-05-29 | 第4步完成：6×4090 GPU 全量处理 15,429 条，产出 8 档 JSON（L1~L4 × train/test），总计 24MB |

















