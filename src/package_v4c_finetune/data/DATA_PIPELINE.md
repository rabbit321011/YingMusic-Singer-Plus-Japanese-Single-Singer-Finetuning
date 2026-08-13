# 目标歌手 数据集构建流水线

> 最后更新：2026-05-23
> 服务器：`user@[SERVER_IP]` (8×RTX 4090)
> AR 训练：v1/v2/v3 全量 + 6 组消融 + LoRA r=16 全部完成。🔴 诊断：posttrained AR 非英文文本跟随为零。

---

## 概述

从原始目标歌手直播录屏/歌切音频中，经过 **MSST 人声干声提取 → 裁切 → 切分 → RMS 静音筛选 → BYOL 声纹聚类过滤 → Whisper 歌词识别 → 合并去重**，最终构建干净日语歌声数据集。

最终数据集：**16,492 条 / 98.86 小时** (`final_sum_large/`)。

---

## MSST 管道（新·服务器版 v2）

> MSST 已于 2026-05-20 迁移到 Linux 服务器。详见 `docs/MSST_linux_setup.md`。

### 环境

| 项目 | 值 |
|------|-----|
| 环境名 | `MSST_scene` (Python 3.10, torch 2.11.0+cu128) |
| 代码路径 | `${REMOTE_ROOT}/MSST/` |
| CLI 入口 | `${REMOTE_ROOT}/MSST/msst_cli_linux.py` |
| 模型路径 | `${REMOTE_ROOT}/MSST/pretrain/` |
| 输出目录 | `${REMOTE_ROOT}/MSST_output/` |

### 已安装模型（三模型管道）

| 步骤 | 模型 | 用途 | 大小 |
|------|------|------|:---:|
| ① | `melband_roformer_instvox_duality_v2.ckpt` | 人声/伴奏分离 → Vocals.wav | 1.7GB |
| ② | `dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt` | 去混响+去回声 → dry.wav | 435MB |
| ③ | `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` | 降噪 → 干净人声 | 871MB |

推理性能：三模型串行 ~20s/条（GPU cuda，显存峰值 ~1.1GB）。

### 新数据集批次（2026-05-20）

| 项目 | 值 |
|------|------|
| 输入 | `hareru_large_dataset.zip`（阿里云盘 → 服务器），1,417 首 mp3 / 10.6GB |
| 处理 | 6 GPU 并行 batch_msst_pipeline.py v3，289 分钟 |
| step4_final | 1,417 条干净人声 WAV (FLOAT 32-bit) / 168GB |
| step5_preview | 1,417 条 mp3 (150kbps，与输入同码率) / 9.5GB |
| 路径 | `${REMOTE_ROOT}/large_dataset/MSST_3step/` |

**目录结构**：
```
large_dataset/
├── MSST_3step/
│   ├── step1_vocals/     ← duality 分离人声 (2,834 wav, 含 Instrumental)
│   ├── step2_dereverb/   ← 去混响 (2,834 wav, 含 other)
│   ├── step3_denoise/    ← 降噪 (2,834 wav, 含 other)
│   ├── step4_final/      ← 仅保留干净人声 dry*.wav (1,417 wav)
│   ├── step5_preview/    ← 压缩 mp3 用于本地切分 (1,417 mp3, 9.5GB)
│   └── step5_preview.7z  ← 已上传阿里云盘 /dataset/
├── batch_msst.log
├── compress_mp3.log
└── gpu0-5.log            ← 每个 GPU 的详细日志
```

**WAV→MP3 时长验证**：10 对抽样全部偏差 0.0ms（ffmpeg 150kbps CBR，44100Hz 完美对齐），本地切分时间点可直接用于服务器 WAV。

### 分割处理策略

```
服务器: MSST → 上传阿里云盘 (7MB/s) → 本地下载 (10MB/s)
本地:   smart_cut → trim_plus → 输出 JSON manifest
服务器: 按 JSON 执行 → BYOL cluster → Whisper → merge
```

> **为什么要这样分**：音频体积大（168GB），本地↔服务器 SCP 只有 250KB/s，必须走阿里云盘中转。JSON manifest 只有几 KB，scp 直传即可。

---

## 最终合并数据集（2026-05-22 最新）

```
final_sum_large/
├── audio/          ← 16,492 条干净日语花丸独唱
└── singnet.json    ← 训练格式 {Path, Duration, Text, Language}
```

| 项目 | 值 |
|------|-----|
| 路径 (服务器) | `${REMOTE_ROOT}/final_sum_large/` |
| 路径 (本地) | `${LOCAL_PROJECT_PATH}\dataset\final_sum_large\` |
| 原始文件数 | 16,492 |
| **清洗后文件数** | **15,429**（2026-05-23） |
| 总时长 | **98.86 小时** |
| 日语率 | 100% |

### 数据清洗（2026-05-23）

移除 Top3 Whisper 幻觉文本（共 1,063 条 / 6.4%）：

| 幻觉文本 | 删除条数 |
|------|:---:|
| `ご視聴ありがとうございました` | 948 |
| `ご視聴ありがとうございました。` | 91 |
| `字幕作成者 初音ミク` | 24 |

清洗后 train/test 分布：train 15,122 / test 307。

### 数据组成

| 来源 | 数量 | 说明 |
|------|:---:|------|
| 新数据 (large_dataset) | 15,325 | 1,404 BV，经四维过滤 + Whisper |
| 旧数据 (singer_sum) | 1,167 | 原有 3,705 → 去重叠 BV + 去短歌词(<8字) |

### 合并规则
- 重叠 BV（199 个）：取新数据，丢弃旧数据
- 旧数据额外过滤：字数 < 8 字的丢弃（31 条）

### 过滤流水线

```
21,027 段 step7_trimmed (37GB)
  → ① RMS+Crest 联合过滤   → 2,437 DROP (11.6%)
  → ② 保守沉默帧 55%       →   744 DROP (3.5%)
  → ③ Gold Anchor 聚簇     → R集 752 → DBSCAN 20簇
  → ④ 人工审核簇 → 按类型标记 (整BV丢弃/音质受损/普通错误/托簇连带/保留/正常)
  → ⑤ 簇级决策执行         → 2,226 DROP (10.6%)
  → ⑥ Whisper 歌词识别     → 33 empty
  → ⑦ 短歌词 (<8字)        →   346 DROP (2.2%)
  ─────────────────────────────────
  最终: 15,325 (27.16 GB)
```

### 过滤维度

| # | 方法 | 检测什么 | 阈值 |
|------|------|------|------|
| ① | **RMS** | 音量 | <-90 全静音 / >-28 正常 |
| ① | **Crest** (Peak/RMS) | 能量比是否自然 | >22dB = 噪声 |
| ② | **沉默帧比例** | 段内"噪音-空白-唱歌"结构 | >55% = 异常 |
| ③ | **Gold Anchor 聚簇** | 声纹身份异常 | dist>0.065 → DBSCAN |

### 簇类型

| 类型 | 行动 |
|------|------|
| 整BV丢弃 | 该簇每段 → 同 BV 全部 DROP |
| 音质受损 | 同 BV >40% 段在此簇 → 整 BV DROP |
| 普通错误 | 仅删该簇内 R-set 段 |
| 托簇连带 | 删 R-set + 连带 rank/threshold 扩集 |
| 正常/保留 | 全部保留 |

---

## 流水线全景

```
原始花丸直播录屏/翻唱 (mp4/m4a/webm...)
  │
  ├─ Step 0: MSST 音频预处理（经典三模型管道）★ 已迁移 Linux
  │    ├─ ① melband_roformer_instvox_duality_v2: 人声/伴奏分离 → Vocals.wav
  │    ├─ ② dereverb_echo_mbr_fused: 去混响+去回声 → dry.wav
  │    └─ ③ denoise_mel_band_roformer_aufr33: 降噪 → 干净人声
  │    CLI: python ${REMOTE_ROOT}/MSST/msst_cli_linux.py --model ... -d cuda
  │    （详见 docs/MSST_linux_setup.md）
  │
  ├─ Step 1: 裁切首尾静音 (trim_plus.py)
  │    └─ 4065 保留 / 17.24h (本地)
  │
  ├─ Step 1b: 静音切分 (smart_cut.py / recut_dense.py) ★ 新
  │    ├─ smart_cut.py: 通用长音频按静音切分 → 15-30s 段
  │    │   └─ ffmpeg silencedetect, -40dB阈值, 0.15s停顿
  │    └─ recut_dense.py: 密集说话音频切分 → 15s 段
  │        └─ ffmpeg silencedetect, -20dB阈值, 0.2s停顿
  │    （本地参考：smart_cut.py, recut_dense.py）
  │
  ├─ Step 2: RMS 静音筛选 (filter_rms.py)
  │    └─ 阈值 0.005 → 删除 134 条 → 3484 保留
  │
  ├─ Step 3: BYOL 声纹聚类过滤 ★ 改进版 (v2)
  │    ├─ 用 SonyCSLParis/ssl-singer-identity (BYOL) 提取 1000维声纹
  │    ├─ DBSCAN eps=0.07 纯聚类（不参考金标，全数据集盲聚）
  │    ├─ 主簇自动保留，小子簇+噪声簇输出人工审核
  │    └─ 辅助：文本关键词过滤 MC ("ご視聴ありがとう" 等)
  │    └─ 删除 635 条 → 2849 保留 / 17.43h (旧版, Mahalanobis一刀切)
  │    ★ 新版按簇决策优于旧版逐条打分，减少误杀，可解释性更强
  │
  ├─ Step 4: Whisper 歌词识别 (whisper_gpu.py × 6 GPU)
  │    └─ 2849/2849 ja，100%
  │
  ├─ Step 5: 合并 speaker1 金标 (merge_dataset.py)
  │    ├─ speaker1 去 MC → 856 条
  │    └─ + speaker1_plus 2849 条
  │
  ├─ Step 6: 去重 ★ 待实现
  │    └─ 基于文本相似度 (normalized edit distance / 均值hash)
  │
  ├─ Step 7: 生成训练格式
  │    ├─ singnet.json (make_singnet.py)
  │    └─ CS/Prosody 码预提取 (extract_codes.py × 6 GPU) → codes/
  │
  └─ ✅ target_singer_singer_sum (wav + text.jsonl + singnet.json + codes/)
```

---

## 声纹过滤：BYOL 聚类方案（v2）

### 原理

| 项目 | 说明 |
|------|------|
| 模型 | SonyCSLParis/ssl-singer-identity (BYOL variant) |
| 输出 | 1000 维声纹向量 |
| 方法 | DBSCAN(eps=0.07, min_samples=3, metric=cosine) 纯聚类 |
| 金标 | 不参考。全数据集盲聚，按簇决策 |

### eps 分辨率对照（3,705 条全数据集实测）

| eps | 簇数 | 噪声 | 效果 |
|:---:|:---:|:---:|------|
| 0.03 | 15 | 203 | 过细，大量歌声误判为噪声 |
| 0.04 | 10 | 105 | 拆出边缘样本 + "某歌簇" |
| **0.07** | **~7** | **~50** | **推荐：主簇干净 + 异类分离** |
| 0.06 | 5 | 30 | 主簇较大，小子簇为同歌片段 |
| 0.12 | 3 | 4 | 太粗，合并异类 |

> 注：BYOL 是按"说话人特征"聚类，不是按"唱法风格"。同一首歌不同段的录制条件一致，在声纹空间里比不同歌同唱法更近，因此会产生"某歌簇"——这不代表不是同一个人，恰恰说明声纹高度一致。

### 按簇决策流程

```
全数据集 BYOL embeddings
  │
  ├─ DBSCAN eps=0.07
  │
  ├─ 主簇 (>95% 数据)  → ✅ 自动保留
  ├─ 小子簇             → ⚠️ 抽样给人工听 → 决定留/删
  └─ 噪声簇             → ⚠️ 人工审核
```

### 辅助防线：文本过滤

```python
MC_KEYWORDS = ['ご視聴ありがとう', 'ばいばい', 'はなまるはねるでした', 'ありがとうございました']
# 剔除包含 MC 关键词的样本
# 实际上还需要按文本出现次数排序，选出出现次数多的，那些一般也是幻觉
```

### 旧版 vs 新版

| | 旧版 (Mahalanobis) | 新版 (DBSCAN 聚类) |
|------|:---:|:---:|
| 决策粒度 | 逐条 | 按簇 |
| 阈值来源 | p99 一刀切 | 人工审核簇样本 |
| 误杀风险 | 高（单维度波动） | 低（整簇同命运） |
| 可解释性 | "距离 1128" | "这个簇47条，歌词都正常" |
| 人工量 | 每条约看 | 每簇抽3条 |

---

## 服务器文件清单

### 环境

| 项目 | 路径 |
|------|------|
| conda (miniforge3) | `/opt/miniforge3/` |
| MSST_scene 环境 | `${REMOTE_ROOT}/.conda/envs/MSST_scene/` |
| yysinger 环境 | `${REMOTE_ROOT}/.conda/envs/yysinger/` (旧训练环境) |
| vevo2 环境 | `${REMOTE_ROOT}/.conda/envs/vevo2/` (旧训练环境) |

### 数据集

| 目录 | 内容 | 文件数 |
|------|------|:---:|
| `large_dataset/target_singer_large_dataset_filtered_long/` | **最终高质量数据集** (WAV+歌词) | 15,325 |
| `large_dataset/target_singer_large_dataset_filtered/` | RMS+Crest+沉默帧+聚簇过滤后 (Whisper前) | 15,671 |
| `target_singer_singer_sum/` | 旧最终训练数据集 | 3,705 |
| `speaker1/speaker1/` | 金标原始 wav | 1,152 |
| `large_dataset/MSST_3step/step4_final/` | 新批次干净人声 WAV | 1,417 |
| `large_dataset/MSST_3step/step7_trimmed/` | 切分+trimmed 全量 | 21,027 |

### Whisper 结果

| 文件 | 条数 |
|------|:---:|
| `large_dataset/target_singer_whisper/whisper_all.jsonl` | 15,671 (filtered) |
| `large_dataset/target_singer_large_dataset_filtered_long/target_singer_large_dataset_filtered_long_whisper.jsonl` | 15,325 (filtered_long) |

### 中间产物

| 目录 | 内容 |
|------|------|
| `cluster_out/` | BYOL 聚类结果 |
| `cluster_out/voice_filter_v2/` | 新版声纹过滤实验 |
| `cluster_out/voice_filter_v2/hanamaru_v3/` | 全数据集 3,705 条聚类结果 |

### BYOL 模型

| 文件 | 路径 |
|------|------|
| 模型权重 | `${REMOTE_ROOT}/pretrained_singer/byol/model.pt` |
| 配置文件 | `${REMOTE_ROOT}/pretrained_singer/byol/hyperparams.yaml` |
| 源码 | `${REMOTE_ROOT}/ssl-singer-identity/` |

### 训练

| 目录 | 内容 |
|------|------|
| `ckpts/ar_hanamaru/` | AR 模型第一轮 |
| `ckpts/ar_hanamaru_v2/` | AR 模型第二轮 (过拟合) |
| `Amphion/` | 训练框架 |

---

## 统计表

| 阶段 | 文件数 | 时长 | 剔除 |
|------|:---:|------|:---:|
| MSST 干声提取 | - | - | 原唱分离 |
| 裁切后 | 4,065 | 24.15h | 太短/静默/太长 |
| RMS 静音筛 | 3,484 | ~20h | 134 |
| BYOL 声纹筛 | 2,849 | 17.43h | 635 |
| +speaker1 金标 | **3,705** | **22.36h** | MC/空白 96 |

### 数据划分（用于训练验证）

| 集 | BV 数 | 条数 |
|------|:---:|:---:|
| 训练 | 316 | 3,576 |
| 测试 | 14 | 120 |
| 总计 | 330 | 3,696 |

划分文件：`train_test_split.json`（按 BV 号整首分，避免同歌泄漏）

---

## 脚本清单

### 服务器现有脚本（`${REMOTE_ROOT}/`）

| 文件 | 用途 |
|------|------|
| `trim_plus.py` | 裁切首尾静音 |
| `filter_rms.py` | RMS 静音筛选 |
| `filter_by_voice.py` | BYOL 声纹过滤 (旧版 Mahalanobis) |
| `cluster_speaker.py` | 金标聚类分析 (DBSCAN) |
| `whisper_gpu.py` | ── |
| `extract_codes.py` | CS/Prosody 码预提取 |
| `train_ar_qwen.py` | AR 训练脚本 |
| `analyze_clusters_v2.py` | 多分辨率聚类分析 (v2) |
| `cluster_hanamaru_v3.py` | 全数据集纯聚类 (v3) |
| `recluster_eps004.py` | 重聚类 (已有向量换 eps) |
| `batch_msst_pipeline.py` | 6 GPU 并行 MSST 三管道（v3 断点续传） |
| `compress_mp3.py` | 多进程 WAV→MP3 压缩 (150kbps) |

### 本地脚本（`${LOCAL_PROJECT_PATH}\`）

| 文件 | 用途 |
|------|------|
| `smart_cut.py` | 通用长音频静音切分（-40dB, 0.15s, 15-30s段, 6线程） |
| `recut_dense.py` | 密集说话音频切分（-20dB, 0.2s, 15s段） |
| `msst_cli_linux.py` | MSST CLI 服务器版 |
| `copy_clusters_v3.py` | 按聚类标签复制 wav 到本地 |
| `copy_eps004.py` | 按 eps=0.04 标签复制 wav |
| `check_server_data.py` | 快速查看服务器数据集状态 |

### trim_plus.py：裁剪首尾静音

**文件**: `trim_plus.py`
**用途**: 用 RMS 能量检测裁剪 wav 首尾静音，支持本地计算+服务器复刻（通过 JSON manifest 同步）

```python
# 本地计算裁剪参数
python trim_plus.py --mode compute \
  --src "${LOCAL_PROJECT_PATH}\dataset\speaker1-plus" \
  --dst "${LOCAL_PROJECT_PATH}\dataset\speaker1_plus_trimmed" \
  --manifest "${LOCAL_PROJECT_PATH}\dataset\trim_manifest.json"

# 服务器按 JSON 复刻
python trim_plus.py --mode apply \
  --src ${REMOTE_ROOT}/data_speaker1_plus \
  --dst ${REMOTE_ROOT}/data_speaker1_plus_trimmed \
  --manifest ${REMOTE_ROOT}/trim_manifest.json
```

**参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| TOP_DB | 30 | 低于峰值 30dB 算静音 |
| MIN_DURATION | 2.0s | 短于 2s 丢弃 |
| MAX_DURATION | 30.0s | 长于 30s 丢弃 |
| MIN_PEAK | 0.01 | 峰值太低丢弃 |
| MAX_WORKERS | 20 | 多线程并行 |

### smart_cut.py：通用长音频静音切分 ★

**来源**: `E:\AIscene\AISVCs\seed-vc\smart_cut.py`
**用途**: 基于 ffmpeg silencedetect 的长音频语义切分。适合直播录屏、歌回等长音频。

```python
# 本地运行
python smart_cut.py
```

**参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| QUIET_THRESH | -40dB | 静音判定阈值 |
| MIN_SILENCE | 0.15s | 最短静音长度 |
| MAX_SEG | 30s | 最大段长 |
| WORKERS | 6 | 多线程并行数 |

**切分策略（4级回退）**:
1. 找 ≥2s 长停顿，起点 ≥T+10
2. 终点在 [T+22, win_end]，停顿 ≥0.5s
3. 终点在 [T+15, T+22)，停顿 ≥1.0s
4. 任意静音终点在窗口内
5. Fallback: hard cut at 29.5s

### recut_dense.py：密集说话切分 ★

**来源**: `E:\AIscene\AISVCs\seed-vc\recut_dense.py`
**用途**: 花丸说话录音的后处理，密集流式切分。

```python
# 本地运行
python recut_dense.py
```

**参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| QUIET_THRESH | -20dB | 静音判定阈值（比唱歌高，说话停顿更明显） |
| MIN_SILENCE | 0.2s | 最短静音长度 |
| IDEAL_SEG | 15s | 理想段长 |
| MAX_SEG | 30s | 最大段长 |

切分使用多窗口回退策略，超 30s 段输出到 `manual_dir` 人工处理。

### RMS 静音筛选

**文件**: `filter_rms.py`
**用途**: 30 线程并行计算每条音频 RMS，筛掉 RMS < 0.005 的静音片段

```python
python filter_rms.py  # 输出 rms_filter.json
```

### batch_msst_pipeline.py：6 GPU 并行 MSST ★

**文件**: `batch_msst_pipeline.py` (v3)
**用途**: 用 ProcessPoolExecutor + CUDA_VISIBLE_DEVICES 实现 6 GPU 并行串行三模型管道。

```bash
# 服务器 tmux 运行
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate MSST_scene
cd ${REMOTE_ROOT} && python3 batch_msst_pipeline.py 2>&1 | tee large_dataset/batch_msst.log
```

**特性**:
- 每 GPU 独立进程，通过 `CUDA_VISIBLE_DEVICES` 隔离
- 断点续传：检查 step3 已有产出自动跳过
- 每步独立跳过：step1/step2 有产出也能续传
- 子进程 stdout/stderr 重定向到 `gpu{N}.log`，主进程只输出干净进度

**输出**:
```
large_dataset/MSST_3step/
  step1_vocals/    ← duality 人声
  step2_dereverb/  ← 去混响
  step3_denoise/   ← 降噪（干净人声）
```

### compress_mp3.py：多进程 WAV→MP3 压缩

**文件**: `compress_mp3.py`
**用途**: 30 进程 ffmpeg 并行压缩 WAV FLOAT → 150kbps MP3

```bash
python3 compress_mp3.py 2>&1 | tee large_dataset/compress_mp3.log
```

### BYOL 声纹聚类过滤 (v2)

**文件**: `cluster_hanamaru_v3.py`
**用途**: 
- 对全数据集提取 BYOL 1000维声纹
- DBSCAN(eps=0.07, min_samples=3, metric=cosine) 纯聚类
- 按簇输出抽样文本 → 人工审核 → 决定留/删
- 辅助：MC 关键词文本过滤

**原理**:
```
全数据集 wav
  → BYOL → embeddings (N × 1000)
  → cosine 相似度矩阵
  → DBSCAN eps=0.07
  → 主簇 (自动保留) + 小子簇 (人工审核) + 噪声 (人工审核)
  → 文本关键词扫 MC → 输出 clean 文件列表
```

### Whisper 全量歌词识别

**文件**: `whisper_gpu.py` + `launch_whisper.sh` / `launch_whisper_plus.sh`

**用途**: 6 GPU 并行 faster-whisper large-v3 识别日语歌词

```bash
# 启动
tmux new-session -d -s whisper_plus
tmux send-keys -t whisper_plus 'bash /tmp/launch_whisper_plus.sh' Enter

# 合并结果
cat ${REMOTE_ROOT}/whisper_plus_out/whisper_gpu*.jsonl > ${REMOTE_ROOT}/whisper_plus_out/whisper_all.jsonl

# 下载到本地
scp user@[SERVER_IP]:${REMOTE_ROOT}/whisper_plus_out/whisper_all.jsonl \
  "${LOCAL_PROJECT_PATH}\dataset\whisper_plus_all.jsonl"
```

**结果**: 2,849/2,849 ja (100%)，6 GPU 约 6 分钟

### 合并数据集

**文件**: `merge_dataset.py`
**用途**: 将 speaker1 金标 + speaker1_plus 清洗后数据合并到统一目录，自动过滤 MC

```bash
# 服务器
python merge_dataset.py ${REMOTE_ROOT} server

# 本地
python merge_dataset.py "${LOCAL_PROJECT_PATH}\dataset" local
```

**输出**: `target_singer_singer_sum/` 目录

### 生成训练格式 singnet.json

**文件**: `make_singnet.py`
**用途**: 读取 text.jsonl + datas/，生成 Amphion 标准训练格式

```bash
python make_singnet.py
```

**输出**: `singnet.json` — 3,696 条 `{Path, Duration, Text, Language}`

### CS/Prosody 码预提取

**文件**: `extract_codes.py` + `launch_extract.sh`
**用途**: 6 GPU 并行，用冻结的 Coco Tokenizer 从 wav 提取 CS 码(12.5Hz)和 Prosody 码(6.25Hz)

```bash
# 6 GPU 并行启动
bash launch_extract.sh
# 输出: codes/*.pt (每个 ~3-6KB, 总计 27MB)
```

**关键规格**:

| 组件 | 规格 |
|------|------|
| Whisper | medium (1024维), n_mels=80 |
| Chromagram | n_chroma=24, hop=480, sr=24000 |
| CS Tokenizer | contentstyle_fvq16384_12.5hz, downsample=4 |
| Prosody Tokenizer | prosody_fvq512_6.25hz, downsample=8 |
| 输出 CS 码 | 12.5 Hz (音频时长/4) |
| 输出 Prosody 码 | 6.25 Hz (音频时长/8) |

### AR 训练脚本

**文件**: `train_ar_qwen.py` (v1/v2), `train_ar_v3.py` (v3), `train_ar_ablations.py` (消融), `train_ar_lora.py` (LoRA)
**用途**: Qwen2ForCausalLM CE loss 训练，支持 train/test split 和 eval

**全部实验总结**:

| 轮次 | 数据 | 最优 eval_loss | 过拟合 | 输出 |
|------|------|:---:|------|------|
| v1 | 3.7k 全量，无验证 | — | 🔴 极早期 | `ar_hanamaru/` |
| v2 | 3.6k/120, 每 500 步 eval | 6.68 | 🔴 epoch 10 | `ar_hanamaru_v2/` |
| v3 | 16k/331, 零泄漏, 25 epoch | **5.99** | 🔴 epoch 5 | `ar_hanamaru_v3/checkpoint-845/` |
| batch_48 | 16k, eff=48, 25 epoch | 5.95 | 🔴 epoch 4 | `ar_ablations/batch_48/` |
| batch_12 | 16k, eff=12, 25 epoch | 5.96 | 🔴 epoch 4 | `ar_ablations/batch_12/` |
| lr_1e-4 | 16k, 50 epoch | 6.13 | 🟠 epoch 7 | `ar_ablations/lr_1e-4/` |
| lr_5e-5 | 16k, 50 epoch | 6.35 | 🟡 epoch 13 | `ar_ablations/lr_5e-5/` |
| lr_2.5e-5 | 16k, 50 epoch | 7.00 | 🟢 几乎无 | `ar_ablations/lr_2.5e-5/` |
| lr_6.25e-6 | 16k, 50 epoch | 7.76 | 🟢 无（不拟合） | `ar_ablations/lr_6.25e-6/` |
| **LoRA r=16** | 16k, 50 epoch | **7.22** | 🟢 **零过拟合** | `ar_lora/final_lora/` |

**诊断实验**:

| 实验 | 文件 | 结论 |
|------|------|------|
| 三语言 baseline | `run_3lang_baseline.py` | EN/CN/JP 默认 loss ~13.5-13.9 |
| 错配文本 | `test_mismatch.py` | 文本不影响 loss（delta=0.009） |
| 零文本 | `diagnose_text_conditioning.py` | 空文本 loss 低于正确文本 |
| SVS demo | `test_svs_melody.py` | EN 文本跟随完美, JP/CN 完全失败 |

**当前状态**: 🔴 Vevo2 posttrained AR 对非英文的文本跟随能力为零。所有微调实验本质上是在背 CS 序列。需重新评估方案。

### CS/Prosody 码文件

- v1/v2: `${REMOTE_ROOT}/target_singer_singer_sum/codes/` (3,696 条, 27MB)
- v3: `${REMOTE_ROOT}/final_sum_large/codes/` (16,492 条, 119MB)

---

## BYOL 声纹模型安装

**模型**: SonyCSLParis/ssl-singer-identity (BYOL variant)
**用途**: 歌手声纹提取 (1000 维)

```bash
# 服务器安装（GitHub 可访问）
git clone https://github.com/SonyCSLParis/ssl-singer-identity.git
cd ssl-singer-identity
pip install pytorch_lightning nnAudio audiomentations soundfile

# 权重从云盘下载到服务器
${REMOTE_ROOT}/bin/aliyunpan download --saveto ${REMOTE_ROOT}/pretrained_singer --sp 5 /byol/model.pt

# hyperparams.yaml 通过 scp 上传
scp byol_hyperparams.yaml user@[SERVER_IP]:${REMOTE_ROOT}/pretrained_singer/byol/hyperparams.yaml

# 验证加载
python -c "
import sys; sys.path.insert(0,'${REMOTE_ROOT}/ssl-singer-identity')
from singer_identity import load_model
model = load_model('byol', source='${REMOTE_ROOT}/pretrained_singer')
print('OK')
"
```

**目录结构**:
```
${REMOTE_ROOT}/pretrained_singer/byol/
  ├── model.pt          (37 MB)
  └── hyperparams.yaml  (158 B)
```

---

## 宝典：SSH + 远程运行技巧

### 避免 SSH 卡住

```bash
# ✅ 加分号防止卡住
ssh -T -n user@[SERVER_IP] "cmd"

# ✅ Python 复杂代码用脚本，不要内联
scp script.py user@[SERVER_IP]:/tmp/
ssh -T -n user@[SERVER_IP] "python /tmp/script.py"

# ✅ 长脚本用 heredoc
ssh -T -n user@[SERVER_IP] "cat > /tmp/script.py << 'EOF'
...python code...
EOF
python /tmp/script.py"

# ❌ 不要用 bash -c 包装 Python 代码（引号嵌套地狱）
```

### tmux 后台运行

```bash
# 创建 tmux 并发送命令
tmux new-session -d -s SESSION_NAME
tmux send-keys -t SESSION_NAME 'command...' Enter

# 查看列表
tmux ls

# 附加查看
tmux attach -t SESSION_NAME
```

### SCP 传输注意

- `-T -n` 防止 SSH 读 stdin 阻塞
- 大量小文件用 `tar czf - | ssh ... tar xzf -` 管道传输
- 大文件客户端超时设置 `-o ConnectTimeout=30`

---

## 数据同步策略

本地和服务器通过 JSON manifest 同步删减操作：

1. **本地计算** → 输出 `xxx_filter.json`（文件列表 + 判定）
2. **上传 JSON** → `scp xxx_filter.json user@[SERVER_IP]:/path/`
3. **服务器执行** → 按 JSON 删除/保留文件
4. **验证一致性** → 比对文件数 + MD5

```python
# 同步脚本模板
import json, os

with open('filter_xxx.json', 'r') as f:
    data = json.load(f)

srcdir = '/path/to/wavs'
removed = 0
for fname in data['remove_list']:
    fpath = os.path.join(srcdir, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        removed += 1

remaining = len([f for f in os.listdir(srcdir) if f.endswith('.wav')])
print(f'deleted={removed}, remaining={remaining}, expected={data["keep"]}')
```

---

## 其他辅助脚本

| 文件 | 用途 |
|------|------|
| `pick20.py` | 随机抽取 20 条 + 罗马音转换，人工校对 |
| `dist_stats.py` | Mahalanobis 距离分布统计 |
| `count_hours.py` / `count_sum.py` | 统计总时长 |
| `check_whisper.py` | 验证 Whisper 结果格式 |
| `clean_local.py` | 清理本地空歌词条目 |
| `diff_files.py` | 比对本地/服务器文件差异 |
| `split_data.py` | 按 BV 号划分训练/测试集 → `train_test_split.json` |

---

## 网络说明

| 链路 | 速度 | 用途 |
|------|:---:|------|
| 服务器 ↔ 国内 CDN | 40 MB/s | pip/git clone |
| 服务器 ↔ 阿里云盘 | 38 MB/s | 大文件传输 (实测 2.5GB/72s) |
| 本机 ↔ 阿里云盘 | 10 MB/s | 上传文件 |
| 本机 ↔ 服务器 (SCP) | 250 KB/s | 少量文本/代码 |
| 服务器 → 国外 | 不通 | 必须走镜像 |

**传输策略**: 大文件走阿里云盘中转，代码走 kkgithub/hf-mirror 镜像。

### 阿里云盘服务器端

```bash
# 工具路径
${REMOTE_ROOT}/bin/aliyunpan

# 大文件下载 (--sp 5 分片加速)
${REMOTE_ROOT}/bin/aliyunpan download --saveto /目标目录 --sp 5 '/云盘路径'

# 列出文件
${REMOTE_ROOT}/bin/aliyunpan ls /路径/
```

---

## Benchmark 数据（三语言评估，2026-05-23）

> 用于验证默认预训练 AR 在不同语言上的文本跟随能力。

| 项目 | 值 |
|------|-----|
| 路径 | `${REMOTE_ROOT}/TEMP/benchmark_final/` |
| 音频 | `cutted_audio/` — CN 64 / EN 66 / JP 77 段（MSST 已预处理） |
| Whisper 转录 | `pipeline_output/asr_results.json`（按文件夹语言分语言转录） |
| 按歌词重命名 | `cutted_audio_by_lyrics/` — 200 段，文件名即 Whisper 转录文本 |
| 云端备份 | `tmp/benchmark/20260523_084500/cutted_audio_by_lyrics.tar.gz` (131MB) |

**Whisper 转录质量**：中文/日文 Whisper 识别质量极差（见 `cutted_audio_by_lyrics/` 文件名），不建议用做真实歌词。

---

## 当前状态总览（2026-05-23）

### ✅ 已完成

| 项目 | 状态 | 结果 |
|------|:---:|------|
| 数据集构建 | ✅ | 15,429 条 / 98.86h（清洗后） |
| MSST 三模型管道 | ✅ | Linux 服务器运行 |
| CS/Prosody 码提取 | ✅ | 16,492 PT 文件 |
| BV 聚类 train/test 划分 | ✅ | 零泄漏验证 |
| AR 全量微调 × 3 | ✅ | 均过拟合 |
| AR 消融实验 × 6 | ✅ | batch/LR 均无法解决过拟合 |
| AR LoRA r=16 | ✅ | 零过拟合, best=7.22 |
| 三语言 baseline | ✅ | EN/CN/JP loss ~13.5-13.9 |
| 文本条件诊断 | ✅ | 非英文文本跟随为零 |

### 🔴 结论

**Vevo2 posttrained AR 对日文/中文的 text→CS 映射从未学会。** 需重新评估方案。

---

## 参考资料

- MSST 模型文档：`docs/MSST_models.md`
- MSST 服务器部署：`docs/MSST_linux_setup.md`
- Vevo2 训练方案：`VEVO2_TRAINING_PLAN.md`
- Win 本地目录：`${LOCAL_PROJECT_PATH}\`（`dataset/`, `scripts_archive/`, `Vevo2_weights/`）
- 服务器根目录：`${REMOTE_ROOT}/`（`target_singer_singer_sum/`, `ckpts/`, `Amphion/`, `MSST/`）
