# YingMusic-Singer-Plus 架构深度分析：cond/midi/text 的分离机制

> 只包含从源码直接推演出的确定性事实。不做计划、不预测。

---

## 一、核心问题

用户的诉求：target singer的音色 + 别人的 MIDI 旋律 + 别人的歌词，三者隔离。

架构是否支持这个？回答这个问题需要逐帧搞清楚 cond、midi、text 在推理和训练中各自扮演什么角色。

---

## 二、推理时的帧级行为（官方代码路径）

以具体数值为例，便于理解尺度。假设：

- ref_audio: target singer唱 "たぶん私じゃなくていいね" (约 3 秒)
- melody_audio: 另一个人唱另一首歌 (约 8 秒)
- target_text: "君を見るたびに思い出す"（要唱的歌词）

### 2.1 VAE 编码后的尺度

```
ref_latent    = VAE.encode(ref_audio)     → [1, T_ref=65, 64]
melody_latent = VAE.encode(melody_audio)  → [1, T_mel=172, 64]
total_len = T_ref + T_mel = 237
```

VAE 帧率 = 44100/2048 = 21.533203125 Hz。每帧 ≈ 46.4ms。T_ref=65 → 约 3 秒。

### 2.2 MIDI 提取

[YingMusicSinger.py:L120-L130](repository-relative-source):

```python
midi_in = torch.cat([ref_latent, melody_latent], dim=1)  # [1, 237, 64]
ref_mel = mel_spec(ref_wav)         # Mel 频谱, ref 部分
melody_mel = mel_spec(melody_wav)   # Mel 频谱, melody 部分
melody_mel_spec = torch.cat([ref_mel, melody_mel], dim=2)
midi_p, bound_p = SOME(melody_mel_spec.transpose(1,2))  # → [1, 237, 128]
```

midi_p 覆盖 total_len 全长。ref 部分的 midi 包含了target singer原唱的旋律，melody 部分的 midi 包含了目标旋律。

### 2.3 Singer.sample() 内部的帧级处理

这是理解分离机制的核心。代码在 [model.py:L290-L308](repository-relative-source)：

```python
# cond_mask: [True]*T_ref + [False]*(max_duration-T_ref)
# True = cond 有效帧, False = pad 帧

# 步骤1: cond 尾部零填充
cond = F.pad(cond, (0,0, 0, max_duration - T_ref), value=0.0)
# cond: [1, 237, 64] = [target singer的latent(65帧) | 0(172帧)]

# 步骤2: step_cond — 显式确保区域外全零
step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
# step_cond: [1, 237, 64] = [target singerlatent(65帧) | 0(172帧)]

# 步骤3: midi 在 cond 区域清零
midi = torch.where(cond_mask, torch.zeros_like(midi), midi)
# midi: [1, 237, 128] = [0(65帧) | 目标旋律MIDI(172帧)]
```

**帧 0~64（cond 区域）**：
| 通道 | 值 | 含义 |
|------|-----|------|
| x_t[n] | 噪声 latent | 需要去噪的当前状态 |
| cond[n] | target singer的 latent | target singer在这一帧的真实音频信息 |
| text[n] | ref_text 的 token | "たぶん私じゃ..."对应的音素 |
| midi[n] | **0** | 旋律信息被清零 |

**帧 65~236（target 区域）**：
| 通道 | 值 | 含义 |
|------|-----|------|
| x_t[n] | 噪声 latent | 需要去噪的当前状态 |
| cond[n] | **0** | 无直接音频信息 |
| text[n] | target_text 的 token | "君を見るたびに..."对应的音素 |
| midi[n] | 目标旋律 MIDI | 来自 melody_audio 的旋律 |

### 2.4 自注意力机制

[DiTBlock](repository-relative-source) 中，batch=1 时 mask=None → **全帧双向自注意力**。每个 target 帧可以 attend 到所有 cond 帧。

因此，target 区域的每一帧的输入虽然 cond[n]=0（无直接音频信息），但通过自注意力可以看到帧 0~64 中的target singer latent：

```
帧 100（target 区域）的自注意力：
  Q(帧100) · K(帧0..236) → attention weights
  → 可以关注帧 0-64（target singer的 cond latent）
  → 提取target singer的音色特征
  → 与 midi[100]（旋律）和 text[100]（歌词）组合
  → 预测 v[100]
```

### 2.5 输出切片

[YingMusicSinger.py:L231](repository-relative-source):

```python
generated_latent = generated_latent[:, ref_latent_len: -int(self.vae_frame_rate*self.rear_silent_time), :]
# 切片: 从 T_ref 到 T_total - rear_silent_frames
# 只保留 target 区域
```

cond 区域（帧 0-64，target singer唱 "たぶん私じゃ..."）被**完全丢弃**。ref_text 永远不会出现在输出中。

### 2.6 三元分离的精确定义

| 元素 | 来源 | 传递机制 | 出现区域 |
|------|------|----------|----------|
| **音色** | ref_audio 的 cond latent | 自注意力从 cond 帧传播到 target 帧 | cond 区域 + 通过自注意力到 target |
| **旋律** | melody_audio 的 MIDI | 直接 per-frame 输入, cond 区域清零 | 仅在 target 区域 |
| **歌词** | target_text 的 token | 直接 per-frame 输入 | 全序列, 但只有 target 区域被输出 |

**结论：架构设计上，三元是分离的。**

---

## 三、训练时应该发生什么（正确方案推演）

### 3.1 合理的训练设置

对每条训练样本（target singer的一段 seg，VAE 编码后 [T, 64]）：

```
cond  = 前 T_ref 帧                  → [T_ref, 64]   来自同一段音频的开头
x₁    = 全部 T 帧                     → [T, 64]       完整 GT
midi  = 全 T 帧 MIDI, cond 区域清零   → [T, 128]      旋律条件
text  = 全 T 帧 token (dense)         → [T]           歌词条件
```

此时 cond ≠ x₁（长度不同），cond 帧数 < x₁ 帧数：

| 帧范围 | cond | midi | text | 信息源 |
|--------|:----:|:----:|:----:|--------|
| 0..T_ref-1 | 有 | 0 | 有 | cond 提供直接信息（可抄） |
| T_ref..T-1 | **0** | 有 | 有 | **必须依赖 text + midi + 自注意力到 cond** |

**T_ref..T-1 区域完全没有 cond 直接信息**。模型无法抄答案，必须学会：
1. 通过自注意力从 cond 帧提取音色
2. 使用 midi 获取旋律
3. 使用 text 获取音素

### 3.2 对跨角色翻唱意味着什么

在正确训练后，推理时：
- cond: target singer的 3 秒干声 → 音色来源
- midi: 另一首歌的旋律 → 旋律来源（cond 区域清零）
- text: 另一首歌的歌词 → 咬字来源

由于训练时模型已学会在 **cond≠x₁ 且 target 区域无 cond** 的情况下工作，推理时给不同的 midi 和 text 就能生成target singer + 新旋律 + 新歌词的输出。

---

## 四、train_plus.py 的实际行为与问题

### 4.1 实际代码路径

[train_plus.py:L243](repository-relative-source):

```python
def process_batch(wavs, srs, texts, langs):
    latent = vae.encode_audio(w)     # 完整音频 → [T, 64]
    # ...
    return latent, midi, midi_p, aligned_text, B, T, D

def compute_loss(wavs, srs, texts, langs):
    latent, midi, midi_p, aligned_text, B, T, D = process_batch(...)
    # ...
    v_pred, hidden_states = run_dit(x_t, latent, aligned_text, t, midi,
                                     drop_audio, drop_text, drop_midi)

def run_dit(x_t, cond, text_tokens, t, midi, drop_audio, drop_text, drop_midi):
    # ...
    x, _ = dit.get_input_embed(x_t, cond, text_tokens, midi,
                                drop_audio_cond=drop_audio, ...)
```

第 322 行：`run_dit(x_t, latent, ...)` → `cond = latent`。

### 4.2 帧级分析

```
输入:
  latent = VAE.encode(完整音频)            → [T, 64]   ← 被同时用作 cond 和 x₁

InputEmbedding:
  [ x_t[n](64) | cond[n](64) | text_embed[n](512) | midi_proj[n](128) ]
  
  其中 cond[n] = latent[n] = 这一帧的正确答案（完整音频信息）
```

**每帧**的 cond 都是该帧 x₁ 的完整信息。模型不需要用到 text 和 midi —— cond 直接告诉它答案。

### 4.3 训练的 dropout 无法补救

```python
drop_audio = random() < 0.2    # 只有 20% 的步骤丢弃 cond
```

80% 的训练步中，cond 包含每帧的完整答案。DiT 学到的最短路径：

```
v_pred ≈ denoise(x_t) towards cond
       = latent - noise (当 x_t = (1-t)*noise + t*latent 时)
```

这不需要 text，不需要 midi。drop_text 和 drop_midi 的 30% 概率完全被 cond 的 80% 可用率压倒。

### 4.4 为什么自重建成功而改词翻唱失败

- **自重建**：cond = target singer的原唱 latent → 模型直接还原 → RMS 完美
- **改词翻唱**：cond 仍 = target singer的原唱 latent → 模型抄回原歌词 → 新文本完全无效

这和 Vevo2 的诊断结论完全一致：**Recon 成功 ≠ 文本条件生效**。

---

## 五、与 V1 训练的对比

V1 的 `train.py`（[DEPLOY.md](repository-relative-source)）：

```
cond = VAE.encode(seg 前半段)    → [T/2, 64]
x₁   = VAE.encode(seg 整段)      → [T, 64]

cond ≠ x₁：前半段 ≠ 整段
T/2 < T：后半段没有 cond 直接信息
```

V1 至少有 50% 的帧必须依赖 text 和 midi。所以 V1 的日语 SFT 至少部分学会了文本→音频（"咬字改善"）。

Plus 的 `train_plus.py` 比 V1 更倒退：**cond 覆盖 100% 的帧，没有一帧需要 text**。

---

## 六、关于 midi 中 ref 部分的问题

### 6.1 推理时 ref 部分的 midi 被清零

[model.py:L306](repository-relative-source)：

```python
midi = torch.where(cond_mask, torch.zeros_like(midi), midi)
```

推理时，cond 区域的 midi 清零，target 区域保留。这是正确的：midi 只在需要生成 new content 的区域有效。

### 6.2 训练时应该同样处理

train_plus.py 中，midi 直接从完整音频提取，然后通过 FuzzDisturb 随机丢弃帧。但**没有在 cond 区域清零**。

如果训练时 midi 在全序列上都有值，模型学会的是 "midi 覆盖全帧" 的模式。这与推理时 "midi 只在 target 区域有效" 不一致。

正确做法：训练时也对 cond 区域清零 midi，与推理一致。

---

## 七、TextEmbedding 中 `text + 1` 的精确作用

[TextEmbedding.forward](repository-relative-source)：

```python
def forward(self, text, seq_len, drop_text=False, ...):
    text = text + 1                    # 外部 0-based → 内部 1-based
    text = F.pad(text, ..., value=1)   # PAD token = 1
    if drop_text:
        text = torch.zeros_like(text)  # CFG 时全零
    text = self.text_embed(text)       # [B, n, 512]
```

`self.text_embed = nn.Embedding(text_num_embeds + 1, 512)` = `Embedding(374, 512)`。

| 场景 | 外部传入 | 内部 +1 后 | PAD 后 | drop_text | 查找行 |
|------|----------|-----------|--------|-----------|--------|
| 有效音素 k | k | k+1 | k+1 | k+1 | 第 k+1 行 |
| padding | 0 | 1 | 1 | 1 | 第 1 行 (PAD) |
| CFG drop | — | — | — | 0 | 第 0 行 (filler) |

训练时 `encode_ja()` 返回的 token 是 0→374，+1 后变为 1→375（边界）。实际上 token 范围在 0-345 左右，+1 后在 1-346，全部在 Embedding(374, 512) 的有效范围内。

---

## 八、自注意力与长跳跃连接

### 8.1 自注意力无 mask

batch=1 时 [model.py:L314-L317](repository-relative-source)：

```python
if batch > 1:
    mask = lens_to_mask(duration)
else:
    mask = None
```

mask=None → 全帧双向 Attention。每帧可以关注任意其他帧。

### 8.2 长跳跃连接

[dit.py:L271-L273](repository-relative-source)：

```python
self.long_skip_connection = nn.Linear(dim*2, dim, bias=False)
```

在 forward 中，transformer blocks 的输入 x_in 被保存为 residual，blocks 输出后：

```python
x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
# [1024+1024] → [1024]
```

这意味着 InputEmbedding 的输出（包含 cond、text_embed、midi 的投影）通过 skip 连接直接带到 DiT 最后一层。即使经过 22 层 transformer，最终的输出层仍然能看到原始 cond 信息。

**对训练的影响**：skip 连接进一步加强了 cond 的"抄答案"能力——深层输出可以直接读取原始输入中的 cond。

---

## 九、CFG drop_text 的 filler token 问题

### 9.1 代码路径

[TextEmbedding.forward](repository-relative-source)：

```python
def forward(self, text, seq_len, drop_text=False, ...):
    text = text + 1                    # 外部 0-based → 内部 1-based
    text = F.pad(text, ..., value=1)   # PAD token = 1
    if drop_text:
        text = torch.zeros_like(text)  # CFG 时全零
    text = self.text_embed(text)       # [B, n, 512]
```

`self.text_embed = nn.Embedding(374, 512)`。

### 9.2 问题

| 场景 | 传入值 | Embedding 行 | 该行训练状态 |
|------|:---:|:---:|------|
| 有效音素 | 1~374 | 第 1~374 行 | 被前向传播+梯度更新 |
| padding | 1 | 第 1 行 (PAD) | 被训练（但始终被 mask） |
| **CFG drop_text** | **0** | **第 0 行** | **从未被训练！** |

`nn.Embedding` 的第 0 行在 `initialize_weights()` 中未被显式初始化（只有 AdaLN 和 proj_out 被置零），保留 PyTorch 默认的 `N(0, 1)` 随机值。在整个 SFT 30k 步中，没有任何音素映射到第 0 行——它只被 CFG 的 uncond 分支使用。

### 9.3 后果

CFG 公式 `v = v_cond + (v_cond - v_drop_all) × cfg_strength` 中的 `v_drop_all` 来自 text=全零→Embedding[0]==随机向量 的 uncond 路径。减去的不是"无文本条件"的方向，而是一个随机噪声方向。

这会导致：
- CFG 放大的是 cond 路径与随机噪声之间的差异，而非 cond 与 uncond 之间的差异
- cfg_strength 越高，越可能引入不可控的噪声
- 与 V1 对比：V1 的 drop_text 也用相同方式处理，但 V1 的 Embedding 表只有 366 行，且训练数据中至少 CN+EN 音素覆盖了大部分行

### 9.4 正确做法

两个选项：
1. **训练时显式将第 0 行初始化为所有有效音素 embedding 的均值**，使其代表"无文本"的平均表征
2. **在 `text_num_embeds` 中额外分配一个位置给专门的 `[NULL]` token**，训练时定期用它替代 text（类似 masked language modeling 的 [MASK] 策略）

---

## 十、ref_text 的角色

推理时 `ref_text` 不会出现在输出中（输出从 T_ref 开始切片），但它在训练中有作用：

在正确训练中，ref_text 填充 cond 区域的 text 通道。这告诉模型：
- "cond 帧 0..T_ref-1 中的音色，对应的是 ref_text 这些音素"
- 模型由此学习：同一音色可以发不同的音素（timbre 与 phoneme 无关）
- 这是跨角色翻唱的基础：给定target singer的音色条件 + 任意的 text，发出正确的音素

---

## 十一、总结：三元分离的完整架构图

```
                      ┌──────────────────────┐
  target singer ref_audio ───→│ VAE encoder           │──→ ref_latent [T_ref, 64]
                      └──────────┬───────────┘
                                 │ cond
                                 │
                      ┌──────────▼───────────┐
  目标 melody_audio ─→│ Mel Spec → SOME       │──→ midi_p [T, 128]
                      └──────────┬───────────┘
                                 │ midi (T_ref区清零)
                                 │
                      ┌──────────▼───────────┐
  目标 text ─────────→│ CNENTokenizer        │──→ text_tokens [T]
                      └──────────┬───────────┘
                                 │ text (全长)
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      ▼                       │
          │    InputEmbedding(x_t | cond | text | midi)  │
          │           → Linear(768→1024)                 │
          │                      │                       │
          │              ┌───────▼───────┐               │
          │              │  DiTBlock ×22 │               │
          │              │  SelfAttention │← 全帧双向     │
          │              │  (cond 帧 ──→  target 帧)     │
          │              │  (timbre 传播) │               │
          │              └───────┬───────┘               │
          │                      │                       │
          │              long_skip_connection            │
          │                      │                       │
          │               ┌──────▼──────┐               │
          │               │  proj_out   │               │
          │               │  v_pred     │               │
          │               └──────┬──────┘               │
          │                      │                       │
          │               ODE 32步采样                   │
          │                      │                       │
          │                切片 T_ref..T_end             │
          │                      │                       │
          │                VAE decoder                   │
          │                      │                       │
          │              44.1kHz 立体声                  │
          └──────────────────────┘

三元分离：
  音色 = cond 帧 × 自注意力传播（无 cond 帧无音色信息）
  旋律 = midi（target 区域，cond 区域清零）
  歌词 = text token（target 区域 + ref 区域）
  
  ref_text 不出现在输出中（被切片丢弃）
```

---

## 十二、train_plus.py 每一帧的真相

对一条 10 秒训练样本 (T = 215 帧)：

```
帧 0..214:
┌─────────────────────────────────────────────────────────┐
│ x_t[n]   cond[n]=latent[n]  text[n]  midi[n]           │
│ (含噪声)  (这一帧的答案！！!)  (可忽视)  (次要)          │
└─────────────────────────────────────────────────────────┘

drop_audio=0.8: 80%情况下答案直接可见
→ DiT 学会: v_pred = denoise(x_t) towards cond
→ 与 text 和 midi 无关
```

**对比正确训练应当的样子：**

```
帧 0..T_ref-1 (cond 区域):
┌──────────────────────────────────────────────────┐
│ x_t[n]   cond[n]=ref_latent[n]  text[n]  midi=0 │
│           (target singer参考)          ref 歌词         │
└──────────────────────────────────────────────────┘

帧 T_ref..T-1 (target 区域):
┌──────────────────────────────────────────────────┐
│ x_t[n]   cond[n]=0   text[n]        midi[n]     │
│          (无答案!)   target 歌词      旋律         │
│                                                   │
│  必须用: 自注意力到 cond 帧(音色) + midi(旋律) + text(歌词)  │
└──────────────────────────────────────────────────┘

模型被迫学会分离三要素
```

















