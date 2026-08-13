# YingMusic-Singer-Plus 日语微调 V4 — 完整执行计划

> 基于 V3 训练全周期复盘（2026-05-26~27）的全部发现。
> V3 cond 修复正确但 x_t 泄漏导致训练推理不匹配——模型从未被要求从噪声生成。
> V4 通过 A/B 区分割彻底解决。

***

## 一、背景与目标

### 1.1 目标

训练 YingMusic-Singer-Plus 的 DiT 模型，使其具备**目标歌手音色 + 任意旋律 + 任意日文歌词**的日语歌唱合成能力。

### 1.2 当前状态

| 项目 | 状态 |
|------|:--:|
| Plus 官方 CN/EN 推理 | ✅ 正常 |
| Plus JA SFT V1 (`train_plus.py`, 30k步) | ❌ cond=GT bug，rewrite从未成功 |
| Plus JA SFT V3 (`train_plus_v3.py`, 30k步) | ❌ x_t 泄漏 + cond 泄漏 → 推理全噪声 |
| `ckpts/plus_ja_sft_v3/` | 🗑️ 无效权重（~90GB），可删除 |
| V4 计划 | 🔥 当前文档 |

### 1.3 V1 失败 → V3 修复 → V3 新发现

```
V1: cond = full_latent = GT  →  DiT 抄答案不学文本
        ↓ 修复
V3: cond = 前50%帧, x_1 = full_latent
        ↓ 训练成功（Flow 24.9→0.45），推理全噪声
        ↓ 复盘
V3 新问题: x_t = (1-t)×noise + t×full_latent  ← 每一帧都泄露答案
           推理时 x_t 只有首帧重复 → 分布外 → 噪声
```

**V3 全部发现见 [plus-finetune-plan-v0527.md](file:///${LOCAL_PROJECT_PATH}/docs/plus-finetune-plan-v0527.md)。**

### 1.4 V4 修复原理

将 latent 序列分为两个区域，统一训练和推理的信息分布：

| 区域 | 帧范围 | 语义 | x_t | cond | midi | text |
|------|:---:|------|------|------|------|------|
| A（参考区） | `[0, ref_len)` | 已知音色参考 | 标准 flow matching（有cond直接参考） | 真latent | 零 | 真实token |
| B（生成区） | `[ref_len, T)` | 需模型生成 | 标准 flow matching（无cond） | 零 | 真旋律 | 真实token |

***

## 二、需修改的内容（共 6 项）

| # | 模块 | 改动 | 严重度 |
|---|------|------|:---:|
| 1 | x_t 构造 | A区=cond（固定）、B区=标准 flow matching | 🔴 致命 |
| 2 | flow loss | A区+B区分开记录，合并反向传播 | 🔴 致命 |
| 3 | ref_len | 随机 5s~33% + 短语边界吸附（A7 时间戳已就绪） | 🟡 改善 |
| 4 | t_shift | 训练 t 也用 t_shift 变换 | 🟡 对齐 |
| 5 | 推理 x_t 初始化 | A区=cond、B区=纯噪声、t 0→1 | 🔴 致命 |
| 6 | 文本编码（P10） | 空格分句 + SEP 插入 + **A7 句级时间戳精确对齐**（L1/L2）/ 均匀分布兜底（L3） | 🟡 对齐 |

CNENTokenizer G2P 改造、cond 前段填零、midi 区域清零 — 同 V3，无需再改。

### 2.1 短语边界吸附（A7 时间戳已就绪）

句级时间戳已由 A7 流水线产出，优先使用短语边界作为 ref_len 切分锚点（±2.5s 范围内吸附到最近的短语边界）。短语边界比能量波谷更精确。

```python
def compute_ref_len(T, phrase_boundaries=None):
    five_sec_frames = int(5.0 * 44100 / 2048)
    ref_len = T * random.uniform(0.125, 0.33)
    ref_len = max(ref_len, five_sec_frames)
    ref_len = min(ref_len, int(T * 0.65))
    if phrase_boundaries is not None:
        margin = int(2.5 * 44100 / 2048)
        candidates = [b for b in phrase_boundaries if abs(b - ref_len) <= margin]
        if candidates:
            ref_len = min(candidates, key=lambda b: abs(b - ref_len))
    return int(ref_len)
```

- **状态：已就绪** — A7 时间戳产出完成，短语边界可从 L1/L2 JSON 直接读取
- **兜底：** L3 样本无可靠时间戳，回退到随机 ref_len
- **时间戳文档**：见 [DATA_PIPELINE_Timeset.md](file:///${LOCAL_PROJECT_PATH}/docs/DATA_PIPELINE_Timeset.md)

### 2.2 G2P 一致性测试

V3 阶段已验证通过（5/5 日文样本）。如果后续改动 G2P 路径（如恢复 CN/EN 支持），需重新测试。

测试方法：`python test_ja_g2p.py`

***

## 三、服务器资源

| 项目 | 详情 |
|------|------|
| 服务器 | `user@[SERVER_IP]` |
| GPU | 8× RTX 4090 (24GB)，可用 0-3 |
| 磁盘 | 7.0TB 总 |
| 环境 | `yingmusic_plus` (python 3.10, torch 2.6.0+cu124) |
| 项目路径 | `${REMOTE_ROOT}/YingMusic-Singer-Plus/` |

### 3.1 清理任务（已完成 2026-05-27）

```bash
# 释放 GPU — 已完成
# tmux kill-session -t sftv3

# 删除无效 V3 checkpoint（~90GB）— 已完成
# rm -rf ${REMOTE_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v3/
```

***

## 四、数据集

### 4.1 训练/验证集

| 集 | 文件 | 条数 | 时长 | 路径（服务器） |
|------|------|:---:|------|------|
| Train | `train_singnet.json` | 15,122 | 90.7h | `${REMOTE_ROOT}/final_sum_large/` |
| Test | `test_singnet.json` | 307 | ~2h | `${REMOTE_ROOT}/final_sum_large/` |

- 100% 日语数据
- Train/Test 按 BV 号隔离
- 数据格式: `[{Path, Duration, Text, Language}]`，`Language: "ja"`

#### 4.1.1 V4 训练数据策略

**第一阶段（粗训）**：使用 L1+L2+L3（~14,800 条）建立基本能力。L1（7,725 条）精确时间戳对齐，L2（3,924 条）允许偏差，L3（967 条）均匀分布兜底。L4（2,243 条）丢弃。

**第二阶段（精调）**：仅用 L1 高质量数据精调。时机待第一阶段改词门禁通过后决定。

对应 JSON 路径（A7 产出）：
```
${REMOTE_ROOT}/final_sum_large/pretreatment_text/timeset/
├── train_L1_high.json    # sim≥0.9 + 句≤10，精确时间戳
├── train_L2_medium.json  # sim≥0.7
├── train_L3_low.json     # sim≥0.5，均匀分布兜底
└── train_L4_discard.json # sim<0.5，丢弃
```

### 4.2 音频文件

```
${REMOTE_ROOT}/final_sum_large/audio/    (15,429 wav, ~30GB)
```

### 4.3 Text 字段格式

text 字段为空格分隔的连续短句——没有 `|`、没有换行。平均长度 59 字符（8~504），平均 6 个短句/条，仅 2.3% 含标点符号。

#### 4.3.1 V4 的文本编码策略（P10）

官方 `lrc_align.py` 将文本按句子分割，逐句 G2P 后在末尾插入 `<SEP>`（token ID=365）。训练侧需要匹配此格式：

```python
def encode_text_with_sep(text):
    phrases = text.split()          # 按空格分句
    tokens = []
    for i, phrase in enumerate(phrases):
        tokens.extend(tokenizer.encode(phrase))
        if i < len(phrases) - 1:
            tokens.append(365)      # <SEP> token ID
    return tokens
```

效果：
```
输入: "たぶん私じゃなくていいね 余裕のない二人だったし ごめんね"
输出: [phonemes(たぶん...), SEP, phonemes(余裕...), SEP, phonemes(ごめんね)]
```

- `<SEP>` embedding（ID=365）官方预训练阶段已训练（GRPO 阶段有非零优化器动量），V3 中因过滤 SEP 未使用。V4 恢复 SEP 后继续训练获得梯度
- `<PUNCT>`（ID=364）保持不用——全代码库中无使用路径，embedding 继续随机
- `<PAD>`（ID=0）仍用于填充 token 不足 T 的剩余帧

***

## 五、模型权重

| 文件 | 大小 | 训练时 |
|------|------|:---:|
| `YingMusicSinger_model.pt` | 7.6 GB | 🔥 加载 + 可训练 |
| `stable_audio_2_0_vae_20hz_official.ckpt` | 596 MB | ❄️ 冻结 |
| `model_ckpt_steps_100000_simplified.ckpt` | 449 MB | ❄️ 冻结 |

**从 base 权重开始训练（V3权重无效——学到了抄 x_t 的策略，新任务完全不同）。**

***

## 六、改动清单

### 6.1 ref_len 计算

```python
def compute_ref_len(T, phrase_boundaries=None):
    five_sec_frames = int(5.0 * 44100 / 2048)  # 108 frames

    # 1. 在 12.5%~33% 范围内随机
    ref_len = T * random.uniform(0.125, 0.33)

    # 2. 如果 < 5s，强制设为 5s
    ref_len = max(ref_len, five_sec_frames)

    # 3. 如果 > 65% T，强制设为 65%
    ref_len = min(ref_len, int(T * 0.65))

    # 4. 短语边界吸附：吸附到最近的短语边界（±2.5s），见 §2.1
    if phrase_boundaries is not None:
        margin = int(2.5 * 44100 / 2048)  # ~54 frames
        candidates = [b for b in phrase_boundaries if abs(b - ref_len) <= margin]
        if candidates:
            ref_len = min(candidates, key=lambda b: abs(b - ref_len))

    return int(ref_len)
```

| 指标 | 值 |
|------|:--:|
| 中位音频时长 | 23.2s |
| 中位 ref_len | ~6-8s |
| 随机区间 | 12.5%~33% |
| 5s 下限 | 当比例区间结果 < 108帧时生效（约 7.7s~16s 音频） |
| 65% 上限 | 当比例区间结果 > 65%T 时生效（极少触发） |

### 6.2 process_batch 改动（batch=1，无 padding）

> **设计决策**：batch_size=1/GPU，避免 batch 内不同样本 T 不统一导致的 ref_len 切片问题。用 grad_accum=4 补足有效 batch size（4GPU × 1 × 4 = 16）。

```python
def process_batch(wav, sr, text, lang):
    """单样本处理，B=1，T 就是这条音频的帧数"""
    with torch.no_grad():
        # === VAE encode ===
        w_2d = w.unsqueeze(0) if w.dim() == 1 else w
        latent = vae.encode_audio(w_2d, in_sr=sr)
        full_latent = latent.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, T, 64]
        B, T, D = full_latent.shape  # B=1

        # === ref_len 计算：基于本条音频时长 + 短语边界吸附 ===
        audio_dur = w.shape[-1] / 44100
        phrase_boundaries = load_phrase_boundaries(sample_id)  # 从 A7 JSON 读取
        ref_len = compute_ref_len(T, phrase_boundaries)  # L3 传 None → 随机

        # === cond: A区真值 / B区零 ===
        cond = torch.zeros_like(full_latent)
        cond[:, :ref_len, :] = full_latent[:, :ref_len, :]

        # === midi: A区清零 / B区保留 ===
        mel = mel_spec_extract(audio=w_2d, sr=44100)
        midi_p, _ = midi_teacher(mel.transpose(1, 2))
        if midi_p.shape[1] != T:
            midi_p = F.interpolate(midi_p.transpose(1, 2), size=T,
                                   mode="linear", align_corners=False).transpose(1, 2)
        midi = raw_model.smoothMelody_MIDIFuzzDisturb(midi_p)  # sigmoid + 10%等间距dropout
        midi[:, :ref_len, :] = 0    # A区清零

        # === text: 从预计算 token JSON 加载逐句 tokens + A7 时间戳精确对齐（L1/L2）/ 均匀兜底（L3） ===
        phrases = load_phrase_tokens(sample_id)  # 从 tokens/*_tokens.json 加载逐句 token 列表
        timestamps = load_timestamps(sample_id)  # A7 产出的句级时间戳
        if timestamps is not None:
            # L1/L2：按时间戳精确放置 token，句间自动插入 SEP
            aligned_text = align_text_with_timestamps(phrases, timestamps, ref_len, T)
        else:
            # L3 兜底：按比例均匀分布，句间插入 SEP
            aligned_text = uniform_align_text(phrases, ref_len, T)

        return full_latent, cond, midi, midi_p, aligned_text, ref_len, T
```

### 6.3 compute_loss 改动（batch=1）

> **设计变更**：A区 x_t 也走标准 flow matching（噪声→重建），不再特殊设为零。A区有 cond（真值）提供直接参考，重建极其简单。这样：
> - P1（A区 v_pred 漂移）自动消失
> - P5（drop_audio 混淆 AB区）自动消失
> - 与官方推理的 x_t 行为完全一致

```python
def compute_loss(wav, sr, text, lang):
    full_latent, cond, midi, midi_p, aligned_text, ref_len, T = process_batch(wav, sr, text, lang)
    # B=1, full_latent/cond/midi shape: [1, T, 64]

    # === t_shift 对齐 ===
    u = torch.rand(1, device=device)
    t = 0.5 * u / (1 - 0.5 * u)  # t_shift=0.5（同官方推理）

    noise = torch.randn_like(full_latent)

    # === 全帧标准 flow matching：AB区统一 ===
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * full_latent
    v_target = full_latent - noise

    drop_audio = random.random() < 0.3
    drop_text  = random.random() < 0.3
    drop_midi  = random.random() < 0.3

    v_pred, hidden_states = run_dit(x_t, cond, aligned_text, t, midi,
                                     drop_audio, drop_text, drop_midi)

    # === Flow loss: A区+B区分开记录（合并反向传播）===
    L_flow_A = F.mse_loss(v_pred[:, :ref_len, :], v_target[:, :ref_len, :])
    L_flow_B = F.mse_loss(v_pred[:, ref_len:, :], v_target[:, ref_len:, :])
    L_flow   = L_flow_A + L_flow_B

    # === CKA loss: 仅 B 区计算 ===
    # A 区 midi=0，CKA 无意义且会稀释 B 区信号
    # hidden_states 是 list of 3 tensors，需逐个切片
    h_B = [h[:, ref_len:, :] for h in hidden_states]
    m_B = midi_p[:, ref_len:, :]
    L_cka = compute_cka_loss_from_hidden(h_B, m_B)

    loss = L_flow + args.cka_weight * L_cka
    return loss, {"flow_A": L_flow_A.item(), "flow_B": L_flow_B.item(),
                  "cka": L_cka.item()}
```

### 6.4 训练信息流总结

```
                    A 区（参考区）                 B 区（生成区）
                    ──────────────                ──────────────
x_t                 (1-t)×噪声 + t×真latent       (1-t)×噪声 + t×真latent
cond                真latent ⇨ 音色参考            零 ⇨ "没参考了"
midi                零 ⇨ "不用旋律"              真实旋律 ⇨ "音高信息"
text                真实token（A7时间戳精确对齐）    真实token（A7时间戳精确对齐）
模型需要做的        从噪声重建（有cond直接参考）     仅凭 text+midi 生成
v_target            全帧统一: full_latent - noise   全帧统一: full_latent - noise
```

### 6.5 文本对齐策略（P10 扩展）

#### 6.5.1 官方做法：句级时间戳对齐

YingMusic-Singer-Plus 论文（§2.1）：

> *"Each lyric sentence is converted into an IPA subsequence and placed at its **corresponding onset frame** within a padded frame-level sequence"*

即：每句歌词根据其起始时间戳放到对应 latent 帧位置。数据需要**句级时间戳**（精确到秒级即可，±0.5s 足够）。

#### 6.5.2 V4 策略：A7 时间戳精确对齐（L1/L2）+ 均匀分布兜底（L3）

句级时间戳已由 A7 流水线产出（详见 [DATA_PIPELINE_Timeset.md](file:///${LOCAL_PROJECT_PATH}/docs/DATA_PIPELINE_Timeset.md)）。训练时按样本分档使用不同策略：

```python
def align_text_with_timestamps(phrases, timestamps, ref_len, T):
    """L1/L2：按 A7 句级时间戳精确放置 token（秒→帧转换），句间自动插入 SEP"""
    FRAME_RATE = 44100 / 2048
    SEP = 365
    aligned_text = torch.zeros(1, T, dtype=torch.long)
    for i, ts in enumerate(timestamps):
        phrase_tokens = phrases[i] + [SEP]  # 逐句末尾追加 SEP（同官方 lrc_align.py）
        center_frame = ts['center'] * FRAME_RATE
        start_frame  = int(ts['start'] * FRAME_RATE)
        if center_frame < ref_len:
            pos = min(max(start_frame, 0), ref_len - 1)
            for j, tid in enumerate(phrase_tokens):
                if pos + j < ref_len:
                    aligned_text[0, pos + j] = tid
        else:
            pos = max(min(start_frame, T - 1), ref_len)
            for j, tid in enumerate(phrase_tokens):
                if pos + j < T:
                    aligned_text[0, pos + j] = tid
    return aligned_text
```

L3 兜底函数：无可靠时间戳时，按比例均匀分布 token。

```python
def uniform_align_text(phrases, ref_len, T):
    """L3 兜底：按比例均匀分布 token 到 A/B 区，句间插入 SEP"""
    SEP = 365
    flat = []
    for i, tokens in enumerate(phrases):
        flat.extend(tokens)
        flat.append(SEP)  # 逐句末尾追加 SEP（同官方）
    # 注意最后多一个 SEP，去掉
    if flat:
        flat.pop()
    aligned_text = torch.zeros(1, T, dtype=torch.long)
    split_idx = int(len(flat) * ref_len / T)
    a_tokens = flat[:split_idx]
    b_tokens = flat[split_idx:]
    # A区均匀放在 [0, ref_len)
    a_step = ref_len / max(len(a_tokens), 1)
    for j, tid in enumerate(a_tokens):
        aligned_text[0, min(int(j * a_step), ref_len - 1)] = tid
    # B区均匀放在 [ref_len, T)
    b_len = T - ref_len
    b_step = b_len / max(len(b_tokens), 1)
    for j, tid in enumerate(b_tokens):
        aligned_text[0, min(ref_len + int(j * b_step), T - 1)] = tid
    return aligned_text
```

**分档策略：**

| 档位 | 时间戳质量 | 样本数 | 使用方式 |
|------|:---:|:---:|------|
| L1（sim≥0.9, 短语≤10） | 高 | ~7,700 | 精确时间戳对齐 |
| L2（sim≥0.7） | 中 | ~3,900 | 精确对齐，允许个别偏差 |
| L3（sim≥0.5） | 低 | ~1,000 | **均匀分布兜底**（回退到旧方案） |
| L4（sim<0.5） | 不可用 | ~2,200 | 丢弃 |

**对比：**
| | 官方（时间戳） | V4 + A7（精确时间戳） | V4 fallback（均匀分布） | V3（全在帧0） |
|---|---|---|---|---|
| A区 text 位置 | 第 0 帧开始 | 按时间戳精确放置 | 均匀 [0, ref_len) | 帧 0~n |
| B区 text 位置 | B区起始帧开始 | 按时间戳映射到 [ref_len, T) | 均匀 [ref_len, T) | 帧 0~n（无区分） |
| 与推理一致？ | ✅ | ✅（L1/L2） | ⚠️ 近似（L3） | ❌ |
| 适用样本 | — | ~11,600 | ~1,000 | — |

#### 6.5.3 时间戳获取 — A7 流水线（已完成）

时间戳通过以下流水线获取（已对 15,122 条训练样本执行完毕）：

```
faster-whisper large-v3 ASR → difflib fuzzy match 歌词 → sim 四档分类
```

- **方法**：Whisper ASR 输出带词级时间戳 → 与真实歌词逐句 fuzzy match → 按 sim 分 L1~L4
- **硬件**：6×RTX 4090 并行，~20 分钟完成全部 15,122 条
- **产出**：8 个 JSON 文件在 `dataset/final_sum_large/pretreatment_text/timeset/`

### 6.6 推理方案

推理直接使用官方 `Singer.sample()`，无需手写 Euler 循环。只需正确构造输入：

```python
# === 构造推理输入 ===
ref_latent = full_latent[:, :ref_len, :]            # [1, ref_len, 64] — A区真latent
text_tokens = align_lrc_sentence_level(...)           # 同官方 lrc_align.py（flat 1D array + SEP）
midi_p, bound_p = midi_teacher(mel.transpose(1, 2))        # SOME teacher 原始输出（不经过 FuzzDisturb）

# === 推理 ===
result, _ = model.sample(
    cond=ref_latent,         # Singer.sample 自动处理 cond_mask（A区）和 midi 清零
    text=text_tokens,
    duration=T,              # 总 latent 帧数
    midi_p=midi_p,
    bound_p=bound_p,
    steps=32,
    cfg_strength=1.0,        # 先用 1.0，模型稳定后调高
    t_shift=0.5,
)
```

`Singer.sample()` 内部的 `cond_mask` 机制自动实现 A/B 区分割：
- `cond_mask=True` 帧（A 区）：cond 使用真值、midi 清零
- `cond_mask=False` 帧（B 区）：cond 为零、midi 使用旋律值

与训练完全一致，无需手动管理。

**训练/推理对齐验证：**

| | 训练 A区 | 训练 B区 | 推理 A区 | 推理 B区 |
|------|:---:|:---:|:---:|:---:|
| x_t 初值 | (1-t)×噪声+t×真值 | (1-t)×噪声+t×真值 | 纯噪声(t=0) ✅ | 纯噪声(t=0) ✅ |
| cond | 真值 | 零 | 真值 ✅ | 零 ✅ |
| midi | 零 | 真旋律 | 零 ✅ | 真旋律 ✅ |
| text | 真token | 真token | 真token ✅ | 真token ✅ |

### 6.7 G2P 改造

同 V3，无需改动：

| # | 文件 | 改动 |
|---|------|------|
| 1 | `src/YingMusicSinger/utils/f5_tts/g2p/g2p/japanese.py` | 从 Amphion 复制 |
| 2 | `src/YingMusicSinger/utils/f5_tts/g2p/g2p/cleaners.py` | `text_tokenizers["ja"]` → `None` |
| 3 | `src/YingMusicSinger/utils/f5_tts/g2p/g2p_generation.py` | `has_japanese` 检测 → 日语 bypass |
| 4 | `src/YingMusicSinger/utils/cnen_tokenizer.py` | 无需改动 |

已修复的性能 bug：`PhonemeBpeTokenizer` 泄漏 → 复用全局 `text_tokenizer`。

### 6.8 文件结构

```
${REMOTE_ROOT}/YingMusic-Singer-Plus/
├── train_plus_v4.py          # 新训练脚本
├── src/                      # 含 G2P 日文支持
├── ckpts/
│   └── plus_ja_sft_v4/       # V4 SFT checkpoint
└── test_ja_g2p.py            # G2P 一致性测试
```

***

## 七、训练配置

### 7.1 超参数

| 参数 | 值 | 说明 |
|------|:---:|------|
| learning_rate | 7e-6 | 官方 config |
| warmup_steps | 500 | 日文 embedding 从零训 |
| total_steps | 30,000 | 约 24 epoch |
| batch_size | 1 / GPU | 4GPU × 1 × grad_accum(4) = 16 effective |
| grad_accum | 4 | 补足 batch=1 到等效 16 |
| max_grad_norm | 1.0 | |
| EMA beta | 0.995 | |
| EMA update_after_step | 100 | |
| CKA weight | 1.0 | 官方默认；如果文本学习瓶颈可降 |
| optimizer | AdamW β=(0.9, 0.95), wd=1e-2 | |
| LR schedule | Cosine decay (warmup 后) | |
| precision | fp32 | |
| max_duration | 30s | 覆盖 99.8% 数据 |
| t_shift | 0.5 | 训练和推理一致（同官方） |

### 7.2 CFG dropout

| 条件 | 概率 | 说明 |
|------|:---:|------|
| drop_audio (cond) | 30% | A区 cond 丢弃→uncond 训练 |
| drop_text | 30% | |
| drop_midi | 30% | |

三者独立随机，8 种组合均匀覆盖。

> **关于 Embedding[0]（CFG filler token）**：`drop_text=True` 时 text 全零 → 查 `nn.Embedding[0]`。V3 中文本信号被 x_t 泄漏压制 → drop_text 和正常 text 的 v_pred 几乎无差异 → Embedding[0] 收不到有效梯度 → 保持随机 N(0,1) 值。V4 文本条件生效后，drop_text 产生显著差异 → 梯度自然流向 Embedding[0] → **自愈，无需特殊处理。**

### 7.3 显存估算

| 组件 | 显存 |
|------|------|
| DiT (453.6M, fp32) | ~1.8 GB |
| VAE (156.1M, 冻结) | ~0.3 GB |
| SOME (117.6M, 冻结) | ~0.2 GB |
| 激活值 + 优化器 (batch=1, T≈646) | ~5-8 GB |
| **合计** | **~8-11 GB / GPU** |

> batch=1 大幅降低显存需求，24GB 余量充足。不再需要 OOM 应对方案。

### 7.4 训练时间

| 指标 | 值 |
|------|------|
| 每 epoch | 15,122 / 16 ≈ 945 steps |
| ~1.3s/step (4GPU, batch=1+grad_accum=4) | 估算（比 V3 多 ~10% kernel launch 开销） |
| 30k 步 | **~11 小时** |

### 7.5 启动命令

```bash
tmux new-session -d -s sftv4
tmux send-keys -t sftv4 'cd ${REMOTE_ROOT}/YingMusic-Singer-Plus && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_plus_v4.py \
  --train_json ${REMOTE_ROOT}/final_sum_large/train_singnet.json \
  --eval_json ${REMOTE_ROOT}/final_sum_large/test_singnet.json \
  --output_dir ckpts/plus_ja_sft_v4 \
  --batch_size 1 --grad_accum 4 --lr 7e-6 --warmup_steps 500 --max_steps 30000 \
  --max_duration 30 --seed 42' Enter
```

***

## 八、可调参数与调参指南

### 8.1 影响最大的参数

| 参数 | 当前值 | 作用 | 调高后果 | 调低后果 |
|------|:---:|------|------|------|
| `lr` | 7e-6 | 梯度步长 | 收敛快但可能震荡 | 收敛慢但更稳 |
| `batch_size` | 1 | 每GPU 1条，grad_accum=4 补足等效 | OOM风险 | 梯度噪声大 |
| `cka_weight` | 1.0 | CKA loss权重 | 更强旋律约束，可能挤占文本学习 | 弱化旋律，文本可能学更好 |
| `max_duration` | 30 | 音频截断上限 | 覆盖更多数据但帧数高 | 训练更快但截断多 |

### 8.2 可能调整的参数

| 参数 | 当前值 | 作用 |
|------|:---:|------|
| `warmup_steps` | 500 | LR 从 0 线性增长 |
| `drop_audio` | 30% | 丢弃 cond 概率 |
| `drop_text` | 30% | 丢弃 text 概率 |
| `drop_midi` | 30% | 丢弃 midi 概率 |
| `total_steps` | 30,000 | 训练总步数 |

### 8.3 LR 调参决策树

```
Step 0-500 (warmup):
  ├── Flow loss 稳步下降 → ✅ 继续
  └── Flow loss 不降或震荡 → 停止，lr 降到 5e-6

Step 500-3000:
  ├── Flow loss 持续降到 < 2.0 → ✅ lr 合适
  ├── Flow loss 平台不动 → 试 lr 1e-5
  └── Flow loss 剧烈震荡 → 降到 5e-6
```

### 8.4 注意：初始 Flow loss 会变

V4 初始 Flow loss 需要关注 A/B 区各自的行为：

- **L_flow_A**：A区有 cond（真值）直接参考，重建极其简单 → 预期快速降到很低
- **L_flow_B**：B区无 cond，仅凭 text+midi 生成 → 初始值偏高是正常的
- 分开记录便于判断：**如果 A 区 loss 不降 → 模型结构/初始化有问题；如果 B 区不降但有 A 区 → 正常，需更多步数**

V3 的初始 Flow=24.9（全帧混在一起）。V4 分开后 B 区初始 Flow 可能更高（信息密度低），这是**预期的，不是问题。**

***

## 九、分阶段验证计划

**核心原则：不等到 30k 步才验证。每 2000 步验证一次自重建。**

> **关于评估推理**：评估直接使用官方 `Singer.sample()`（详见 §6.6 推理方案）。输入构造：`cond`=A区 latent `[1, ref_len, 64]` + `text`=通过 `align_lrc_sentence_level` 构造的 flat token 数组 + `midi_p/bound_p`=SOME teacher 原始输出（不经过 FuzzDisturb）。`Singer.sample()` 内部自动通过 `cond_mask` 处理 A/B 区分割（cond 区 midi 清零），与训练完全一致。

### 9.1 阶段 0：G2P 一致性（训练前）

| 测试 | 内容 | 通过标准 |
|------|------|:--:|
| token 一致性 | 5 个日文样本 | 全部匹配 |

### 9.2 阶段 1：自重建 loss 下降（Step 0-5000）

| checkpoint | 验证内容 | 通过标准 |
|:---:|------|:---:|
| Step 100 | warmup 中，loss 开始下降 | Flow < 预期初始值 |
| Step 500 | warmup 结束，loss 正常 | Flow 稳定下降，无 NaN |
| **Step 2000** | **自重建音频可辨识** | **Whisper WER < 0.5，输出可辨识日语** |
| Step 5000 | 持续改善 | WER 进一步下降 |

### 9.3 阶段 2：改词测试（Step 5000）← 关键门禁

对 5 条测试样本做改词翻唱：

| 测试 | 方法 | 通过标准 |
|------|------|:--:|
| **自重建** | ref=melody=同段，target_text=原词 | Whisper WER < 0.5 |
| **改词翻唱** | ref=melody=同段，target_text=不同日文词 | **生成音频中出现 target_text 发音** |

### 9.4 阶段 3：全面评估（Step 5000-30000）

| step | 评估内容 |
|:---:|------|
| 5000 | 改词测试 — **门禁** |
| 10000 | 改词 WER + 听感 |
| 20000 | 改词 WER + F0-CORR + 跨角色翻唱 |
| 30000 | 完整评估：7 首 JP benchmark |

### 9.5 评估指标

| 指标 | 方法 | 目标 |
|------|------|:---:|
| 旋律保真度 | F0-CORR (皮尔逊) | > 0.75 |
| 歌词可懂度 | Whisper large-v3 WER | < 30% |
| 音色相似度 | WavLM speaker cosine | > 0.85 |
| 自然度 | 人耳 MOS | > 3.5/5 |

### 9.6 评估推理注意事项

1. **跳过 FuzzDisturb**（数据增强，推理时不应启用）
2. **推理初始化：** A区=cond、B区=噪声、t 0→1
3. **CFG 先用 1.0**（等模型学会 text 后再调高）
4. **Whisper 用 large-v3，`vad_filter=False`**
5. **ref_text 必须精确匹配 ref_audio 内容**

***

## 十、预期风险

| 风险 | 概率 | 影响 | 应对 |
|------|:---:|:---:|------|
| **推理 CFG=3 仍爆炸** | 中 | 中 | V3 CFG=3 爆炸因为文本信号弱。V4 修好后 CFG 差值方向更清晰——预期缓解。先用 CFG=1，训练中逐步调高验证 |
| **27 个日文 embedding 从头训** | 中 | 中 | 嵌入表共 365 行（CN/EN 预训练 + 27 行日语新增）。V3 文本信号被压扁不是 27 行太少——是 x_t 泄漏让模型不需要任何文本。V4 修好后 27 行会被"释放"——模型第一次有动机训练它们。500 步 warmup 保护。 |
| OOM | 低 | 低 | batch=1 后显存 ~8-11GB，24GB 余量充足 |
| NaN 发散 | 低 | 高 | 梯度裁剪 1.0 |
| G2P token 不一致 | 低 | 高 | G2P 验证必须通过才启动 |
| **SEP/PUNCT gap** | ✅ 已解决 | 低 | V4 按空格分句 + 句间插入 SEP，embedding 通过训练获得梯度。详见 4.3.1 |

***

## 十一、执行顺序总览

```
Step 0: ✅ 清理服务器（kill sftv3, rm plus_ja_sft_v3）— 已完成
Step 1: G2P 一致性测试（服务器已有，test_ja_g2p.py）
Step 2: 编写 train_plus_v4.py（基于 train_plus_v3.py）
Step 3: 上传到服务器
Step 4: V4 训练（Step 0-5000）+ 每 2000 步自重建验证
Step 5: Step 5000 改词测试 ← 关键门禁
  ├── 通过 → Step 6
  └── 失败 → 诊断方向：text 信号仍弱？CFG 爆炸？SEP 干扰？→ 可能需要方案 B（异段 cond）
Step 6: 阶段 3 完整训练（Step 5000-30000）+ 全面评估
Step 7: （可选）GRPO 精调
```

***

## 十二、方案 B 备选（异段 cond）

如果 V4 在 Step 5000 改词测试失败：

- cond 改用花丸数据集中**另一段** 3s 干声（同 BV 号不同 segment → 相同音色、不同内容）
- 优势：完全切断 cond 和 GT 的关联。劣势：预处理复杂
- midi A区仍清零

### 十二、GRPO（按需启动）

`grpo_train.py` 已就绪。仅在 SFT 通过改词测试后按需启动。

需提前审计的问题（留到 SFT 完成后讨论）：

| # | 问题 | 严重度 |
|---|------|:---:|
| 1 | cond 来源（异段 or 同曲？） | 🔴 |
| 2 | 推理 midi 来源 | 🟡 |
| 3 | 四卡 DDP 改造 | 🟡 |
| 4 | 日文 reward 模型（Whisper/WavLM） | 🟡 |

***

## 十三、附录：关键文件速查

| 内容 | 路径 |
|------|------|
| Plus 源码 | `YingMusic-Singer-Plus-src/src/YingMusicSinger/` |
| DiT 架构 | `models/dit.py` |
| Singer.sample() | `models/model.py` |
| CNENTokenizer | `utils/cnen_tokenizer.py` |
| g2p_generation | `utils/f5_tts/g2p/g2p_generation.py` |
| vocab.json | `utils/f5_tts/g2p/g2p/vocab.json` |
| config | `config/YingMusic_Singer.yaml` |
| V3 训练脚本 | `train_plus_v3.py` |
| V3 训练记录 | `docs/plus-v3-training-log.md` |
| Q1/Q2 方案 | `docs/plus-finetune-plan-v0527.md` |
| A7 时间戳管道 | `docs/DATA_PIPELINE_Timeset.md` |
| V3 全计划 | `docs/plus-finetune-fullplan-v2.md` |

***

> 最后更新: 2026-05-29 — A7 时间戳已完成，§2.1/§6.1/§6.2/§6.5 已更新
