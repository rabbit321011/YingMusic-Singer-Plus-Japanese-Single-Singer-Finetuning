# YingMusic-Singer-Plus 日语微调 V4 — Q1/Q2 方案

> 基于 V3 训练复盘（2026-05-27）的全部发现。
> 本文档仅覆盖 Q1（训练 x_t 设计）和 Q2（推理 x_t 初始化）。
> Q3-Q8 待后续讨论。

---

## 一、V3 复盘关键发现

### 1.1 根本问题

V3 训练时 `x_t` 的每一帧都包含 `full_latent` 的真实值（加噪）：

```python
# train_plus_v3.py compute_loss — 当前代码（有问题）
x_t = (1-t) × noise + t × full_latent    # full_latent 每帧都有真值
```

模型学到了：从 x_t（有答案）和 cond（有答案）推断 v_pred，text 的边际贡献被压缩到接近零。

**证据链：**
1. 推理 oracle init（含 full_latent）→ audio_std=0.18 ✅
2. 推理 infer init（仅首帧）→ audio_std=0.03 ❌ 噪声
3. base→step_030k 文本 L2 从 0.40 降到 0.05（87% 压缩）
4. cross-seed cos=0.86 证明文本有方向性信号，但量级太小

### 1.2 修复原理

将 latent 序列分为两个区域，训练和推理采用完全一致的信息分布：

| 区域 | 帧范围 | 语义 | 信息来源 |
|------|:---:|------|------|
| A（参考区） | `[0, ref_len)` | 已知音色参考 | cond 提供真值，x_t 始终保持 cond |
| B（生成区） | `[ref_len, T)` | 需要模型生成 | 仅 text+midi 提供指导，x_t 从噪声起步 |

---

## 二、ref_len 计算规则

### 2.1 算法

```python
def compute_ref_len(T, phrase_boundaries=None):
    five_sec_frames = int(5.0 * 44100 / 2048)  # 108 frames

    # 1. 在 12.5%~33% 范围内随机
    ref_len = T * random.uniform(0.125, 0.33)

    # 2. 如果 < 5s，强制设为 5s
    ref_len = max(ref_len, five_sec_frames)

    # 3. 如果 > 65% T，强制设为 65%
    ref_len = min(ref_len, int(T * 0.65))

    # 4. 短语边界吸附：吸附到最近的短语边界（±2.5s），见 §2.3
    if phrase_boundaries is not None:
        margin = int(2.5 * 44100 / 2048)  # ~54 frames
        candidates = [b for b in phrase_boundaries if abs(b - ref_len) <= margin]
        if candidates:
            ref_len = min(candidates, key=lambda b: abs(b - ref_len))

    return int(ref_len)
```

### 2.2 数据分布分析

| 指标 | 值 |
|------|:--:|
| 中位音频时长 | 23.2s |
| 随机区间 | 12.5%~33% |
| 5s 下限 | 当比例区间结果 < 108帧时生效（约 7.7s~16s 音频） |
| 65% 上限 | 当比例区间结果 > 65%T 时生效（极少触发） |

### 2.3 短语边界吸附

现已具备句级时间戳（见 §6.2），优先使用短语边界作为 ref_len 切分锚点：

```python
def compute_ref_len(T, phrase_boundaries=None):
    five_sec_frames = int(5.0 * 44100 / 2048)  # 108 frames

    ref_len = T * random.uniform(0.125, 0.33)
    ref_len = max(ref_len, five_sec_frames)
    ref_len = min(ref_len, int(T * 0.65))

    # 短语边界吸附：吸附到最近的短语边界（±2.5s）
    if phrase_boundaries is not None:
        margin = int(2.5 * 44100 / 2048)  # ~54 frames
        candidates = [b for b in phrase_boundaries if abs(b - ref_len) <= margin]
        if candidates:
            ref_len = min(candidates, key=lambda b: abs(b - ref_len))

    return int(ref_len)
```

**目的：** 避免把参考/生成边界切在半句话中间。短语边界比能量波谷更精确，且时间戳已完成（A7 流水线产出），零额外成本。

**兜底：** L3/L4 档样本无可靠时间戳，回退到随机 ref_len。

---

## 三、Q1：训练 x_t 设计

### 3.1 process_batch 改动（batch=1）

> ⚠️ **本节伪代码已失效**，仅保留作为设计讨论的历史参考。
> 实际实现以 [plus-finetune-fullplan-v0527.md §6.2](plus-finetune-fullplan-v0527.md) 为准（含 `encode_text_with_sep`、A7 时间戳对齐、FuzzDisturb 等完整逻辑）。

> 使用 batch_size=1/GPU + grad_accum=4，避免 batch 内 padding 带来的 ref_len 切片问题。详见 fullplan。

```python
def process_batch(wav, sr, text, lang):
    """单样本处理，B=1"""
    with torch.no_grad():
        w_2d = w.unsqueeze(0) if w.dim() == 1 else w
        latent = vae.encode_audio(w_2d, in_sr=sr)
        full_latent = latent.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, T, 64]
        B, T, D = full_latent.shape  # B=1

        audio_dur = w.shape[-1] / 44100
        ref_len = compute_ref_len(T, audio_dur)

        cond = torch.zeros_like(full_latent)
        cond[:, :ref_len, :] = full_latent[:, :ref_len, :]

        mel = mel_spec_extract(audio=w_2d, sr=44100)
        midi_p, _ = midi_teacher(mel.unsqueeze(0).transpose(1, 2))
        if midi_p.shape[1] != T:
            midi_p = F.interpolate(midi_p.transpose(1, 2), size=T,
                                   mode="linear", align_corners=False).transpose(1, 2)
        midi = raw_model.smoothMelody_MIDIFuzzDisturb(midi_p)  # sigmoid + 10%等间距dropout
        midi[:, :ref_len, :] = 0

        tokens = tokenizer.encode(text)
        aligned_text = torch.zeros(1, T, dtype=torch.long, device=device)
        n = min(len(tokens), T)
        aligned_text[0, :n] = torch.tensor(tokens[:n], device=device)
        # ↑ 无时间戳时回退到简单顺序放置（L3/L4 样本）
        #   有 A7 时间戳时使用 align_text_with_timestamps() 精确放置（见 §6.2.B）

        return full_latent, cond, midi, midi_p, aligned_text, ref_len, T
```

### 3.2 compute_loss 改动（batch=1）

> A区 x_t 走标准 flow matching（噪声→重建），A区有 cond 直接参考。与官方推理一致。

```python
def compute_loss(wav, sr, text, lang):
    full_latent, cond, midi, midi_p, aligned_text, ref_len, T = process_batch(wav, sr, text, lang)

    u = torch.rand(1, device=device)
    t = 0.5 * u / (1 - 0.5 * u)

    noise = torch.randn_like(full_latent)

    # === 全帧标准 flow matching：AB区统一 ===
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * full_latent
    v_target = full_latent - noise

    drop_audio = random.random() < 0.3
    drop_text  = random.random() < 0.3
    drop_midi  = random.random() < 0.3

    v_pred, hidden_states = run_dit(x_t, cond, aligned_text, t, midi,
                                     drop_audio, drop_text, drop_midi)

    L_flow_A = F.mse_loss(v_pred[:, :ref_len, :], v_target[:, :ref_len, :])
    L_flow_B = F.mse_loss(v_pred[:, ref_len:, :], v_target[:, ref_len:, :])
    L_flow   = L_flow_A + L_flow_B

    midi_for_cka = midi_p[:, :T, :]
    L_cka = compute_cka_loss_from_hidden(hidden_states, midi_for_cka)

    loss = L_flow + cka_weight * L_cka
    return loss, {"flow_A": L_flow_A.item(), "flow_B": L_flow_B.item(),
                  "cka": L_cka.item()}
```

### 3.3 关键：模型如何区分 A/B 区

模型通过以下信号自然感知帧的位置和类型：

1. **cond 值差异** — A 区 cond ≠ 0, B 区 cond = 0。最强的区分信号
2. **midi 值差异** — A 区 midi = 0, B 区 midi = 旋律值
3. **RoPE 位置编码** — 帧 0≠帧 322，模型知道位置

### 3.4 训练信息流总结

```
                    A 区（参考区）                 B 区（生成区）
                    ──────────────                ──────────────
x_t                 (1-t)×噪声 + t×真latent       (1-t)×噪声 + t×真latent
cond                真latent ⇨ 音色参考            零 ⇨ "没参考了"
midi                零 ⇨ "不用旋律"              真实旋律 ⇨ "音高信息"
text                真实token                      真实token
模型需要做的        从噪声重建（有cond直接参考）     仅凭 text+midi 生成
v_target            全帧统一: full_latent - noise   全帧统一: full_latent - noise
```

---

## 四、Q2：推理 x_t 初始化

### 4.1 当前（eval_wer.py 中的错误）

```python
noise = torch.randn(1, T, D)
x_t = 0.5 × noise + 0.5 × full_latent[:, :1, :].expand(-1, T, -1)   # 仅首帧重复
t_vals = torch.linspace(0.5, 1, 33)                                   # 只跑后半段
```

问题：
- 首帧重复导致 B 区得不到有效信息
- 从 t=0.5 开始，模型训练的 t∈[0,1] 从未完整使用

### 4.2 修复

```python
# 全帧纯噪声起步（与官方推理一致，与训练 t=0 时一致）
x_t = torch.randn(1, T, D, device=device)

t_vals = torch.linspace(0, 1, 65)                    # 0→1，64 步
t_vals = t_shift * t_vals / (1 + (t_shift - 1) * t_vals)

x = x_t
for i in range(len(t_vals) - 1):
    dt = t_vals[i+1] - t_vals[i]; t_val = t_vals[i]
    
    # CFG=3.0 会在训练初期导致 latent 爆炸（已知 bug: CFG 爆炸）
    # 先用 CFG=1.0，等模型稳定后再调高
    v_cond = model(x, cond, text, t_val, midi, drop_audio=False, drop_text=False, drop_midi=False)
    v_uncond = model(x, cond, text, t_val, midi, drop_audio=True, drop_text=True, drop_midi=True)
    v = v_cond + (v_cond - v_uncond) × cfg_strength
    x = x + v × dt
```

### 4.3 训练/推理对齐验证

| | 训练 A区 | 训练 B区 | 推理 A区 | 推理 B区 |
|------|:---:|:---:|:---:|:---:|
| x_t 初值 | (1-t)×噪声+t×真值 | (1-t)×噪声+t×真值 | 纯噪声(t=0) ✅ | 纯噪声(t=0) ✅ |
| cond | 真值 | 零 | 真值 ✅ | 零 ✅ |
| midi | 零 | 真旋律 | 零 ✅ | 真旋律 ✅ |
| text | 真token | 真token | 真token ✅ | 真token ✅ |

---

## 五、已知风险

| 风险 | 概率 | 影响 | 应对 |
|------|:---:|:---:|------|
| 推理 CFG=3 仍爆炸 | 中 | 中 | 先用 CFG=1，等模型学会 text 再调高 |
| L3 样本无时间戳需均匀分布兜底 | 低 | 低 | L3 占比 ~7%，影响可控；L1/L2 精确对齐 |

> **Embedding[0] 自愈**：V3 中文本信号弱 → drop_text 无差异 → Embedding[0] 收不到梯度。V4 文本生效后梯度自然流向 Embedding[0]，无需特殊处理。详见 fullplan 7.2 节。

---

## 六、Q5：SEP/PUNCT gap & 文本对齐策略

### 6.1 现象

推理时 `lrc_align.py` 在每行歌词末尾插入 `<SEP>` token（ID=365）。官方 `align_lrc_sentence_level` 流程：标点全部删除 → 逐句 G2P → 末尾追加 `<SEP>` → 按时间戳放置。

YingMusic-Singer-Plus 论文（§2.1）：
> *"Each lyric sentence is converted into an IPA subsequence and placed at its **corresponding onset frame**"*

即官方训练也使用句级时间戳对齐，不是全部文本从帧 0 开始。

### 6.2 V4 方案

**两件事一起做：**

#### A. SEP 插入（让训练匹配推理）

训练数据 text 字段为空格分隔的短句。在 `encode_text_with_sep` 中按空格分句、逐句 G2P、句间插入 `<SEP>`：

```python
def encode_text_with_sep(text):
    phrases = text.split()
    tokens = []
    for i, phrase in enumerate(phrases):
        tokens.extend(tokenizer.encode(phrase))
        if i < len(phrases) - 1:
            tokens.append(365)   # <SEP> token ID
    return tokens
```

- `<SEP>` embedding（ID=365）V4 通过训练获得梯度
- `<PUNCT>`（ID=364）保持不用
- `<PAD>`（ID=0）仍用于填充剩余帧

#### B. A7 句级时间戳对齐（已实现）

句级时间戳已由 A7 流水线产出（详见 [DATA_PIPELINE_Timeset.md](repository-relative-source)）。训练时根据时间戳将 token 精确放置到对应帧位置：

```python
def align_text_with_timestamps(phrases, timestamps, ref_len, T):
    """L1/L2：按 A7 句级时间戳精确放置 token（秒→帧转换）"""
    FRAME_RATE = 44100 / 2048
    aligned_text = torch.zeros(1, T, dtype=torch.long)

    for i, ts in enumerate(timestamps):
        phrase_tokens = phrases[i]
        center_frame = ts['center'] * FRAME_RATE
        start_frame  = int(ts['start'] * FRAME_RATE)
        if center_frame < ref_len:
            # A 区：映射到 [0, ref_len)
            pos = min(max(start_frame, 0), ref_len - 1)
            for j, tid in enumerate(phrase_tokens):
                if pos + j < ref_len:
                    aligned_text[0, pos + j] = tid
        else:
            # B 区：映射到 [ref_len, T)
            pos = max(min(start_frame, T - 1), ref_len)
            for j, tid in enumerate(phrase_tokens):
                if pos + j < T:
                    aligned_text[0, pos + j] = tid
    return aligned_text
```

**关键设计：区域映射** — 时间戳是绝对帧位置，而 A/B 区切分点 `ref_len` 是随机可变的。每句按其时间戳中心判断落入哪个区域，再按比例映射到对应区域的相对位置。

**分档策略：**

| 档位 | 时间戳质量 | 使用方式 |
|------|:---:|------|
| L1（sim≥0.9, 短语≤10） | 高（MAE<1.84s, P95<6.62s） | 精确时间戳对齐 |
| L2（sim≥0.7） | 中（部分边界偏差） | 精确对齐，允许个别偏差 |
| L3（sim≥0.5） | 低（偏差较大） | **均匀分布兜底**（回退到旧方案） |
| L4（sim<0.5） | 不可用 | 丢弃 |

#### C. 时间戳获取 — A7 流水线（已完成）

时间戳通过以下流水线获取（已对 15,122 条训练样本执行完毕）：

```
faster-whisper large-v3 ASR → difflib fuzzy match 歌词 → sim 四档分类
```

- **方法**：Whisper ASR 输出带词级时间戳 → 与真实歌词逐句 fuzzy match → 按 sim 分 L1~L4
- **硬件**：6×RTX 4090 并行，~20 分钟完成全部 15,122 条
- **产出**：8 个 JSON 文件在 `dataset/final_sum_large/pretreatment_text/timeset/`

### 6.3 对比

| | 官方（时间戳） | V4 + A7（精确时间戳） | V4 fallback（均匀分布） | V3（全在帧0） |
|---|---|---|---|---|
| A区 text 位置 | 帧 0 | 按时间戳精确放置 | 均匀 [0, ref_len) | 帧 0~n |
| B区 text 位置 | B区起始帧 | 按时间戳映射到 [ref_len, T) | 均匀 [ref_len, T) | 帧 0~n |
| SEP 训练 | 可能有 | ✅ 有 | ✅ 有 | ❌ 无 |
| 与推理一致 | ✅ | ✅（L1/L2） | ⚠️ 近似（L3） | ❌ |
| 适用样本 | — | L1+L2（~11,600） | L3（~1,000） | — |

---

## 七、Q6：t_shift 对齐

### 7.1 现象

训练 t 从 `Uniform(0,1)` 采样，推理 t 经过变换：

```python
# 推理：把步数密度推向 t=1
t = t_shift * t / (1 + (t_shift - 1) * t)    # t_shift=0.5（同官方推理）
```

### 7.2 修复

训练时也使用同一变换：

```python
# compute_loss — 当前
t = torch.rand(B, device=device)

# 改为
u = torch.rand(B, device=device)
t = t_shift * u / (1 + (t_shift - 1) * u)
```

一行替换，零额外成本，训练推理 t 分布完全一致。

---

## 八、未覆盖项目

| # | 问题 | 状态 |
|---|------|:--:|
| Q1 | 训练 x_t A/B 区设计 | ✅ 已定稿 |
| Q2 | 推理 x_t 初始化 | ✅ 已定稿 |
| Q3 | 重训 vs 打捞权重 | ✅ 从 base 重训 |
| Q4 | CFG 爆炸 | ⏳ 等新模型训好再验证 |
| Q5 | SEP/PUNCT gap + 文本对齐 | ✅ A7时间戳精确对齐（L1/L2）+均匀分布兜底（L3） |
| Q6 | t_shift 对齐 | ✅ 训练加 t_shift 变换 |
| Q7 | PhonemeBpeTokenizer 泄漏 | ✅ 已修复 |
| Q8 | 文档更新 | ✅ 本文档 |

---

> 最后更新: 2026-05-29 — A7 时间戳完成，§2.3/§6.2/§6.3 已更新

















