# 日语 SFT 推理调试：遇到的坑与修复

> 2026-05-25 · 目标歌手日语 SFT 模型评估全过程记录

## 背景

对 YingMusic-Singer 做日语 SFT（5 个 checkpoint: 2k/10k/20k/30k steps），需要推理验证效果。自重建（ref=melody=同一段）应输出与原唱一致的音频。

**核心矛盾**：训练脚本 `train_plus.py` 和推理管线 `YingMusicSinger` 用了**两套完全不同的文本 tokenize 方式**，导致 SFT 权重在推理时完全无效。

---

## 坑 1：Token ID +1 偏移

### 现象
base（官方权重）输出 "ha ha ha i yo"，所有 SFT checkpoint 输出纯噪音。Whisper 全识别为 "ご視聴ありがとうございました"。

### 根因

| | 训练 `encode_ja()` | 推理 `CNENTokenizer.encode()` |
|---|---|---|
| 调用 | `pbt.phoneme2token()` | `token = [x + 1 for x in token]` |
| "私" 的 token ID | **210** | **211** |
| 来源 | 直接查 vocab 返回 0-based index | `cnen_tokenizer.py` 第 30 行强制 +1（为中英文预留 0 padding） |

`text_embed` 是 `[374, 512]` 的 Embedding 表。训练学到 token 210="私"，推理查 token 211 → 完全不同的随机向量 → 全噪音。

### 修复
`cnen_tokenizer.py`：
```python
def encode(self, text):
    phone, token = self.tokenizer(text)
    if not has_japanese(text):  # 日文保持 0-based，与训练一致
        token = [x + 1 for x in token]
    return token
```

### 验证
```python
# 修复前: CNEN=211, train=210 → False
# 修复后: CNEN=210, train=210 → MATCH: True
```

---

## 坑 2：特殊 Token 不在训练词汇中

### 现象
修复坑 1 后 DiT 收到 token ID 最高达 365（超 `text_num_embeds=373` 的合理范围）。

### 根因
`align_lrc_sentence_level` 在文本序列中插入特殊分隔 token：
- `<SEP>` = 365（句子分隔）
- `<PUNCT>` = 364（标点）

训练用的 `encode_ja()` 只产生 0~350 左右的 phoneme token，从未见过 364/365。

### 修复
推理时过滤掉：
```python
tokens = [t for t in cnen.encode(text) 
          if t not in (cnen.sep_token_id, cnen.punct_token_id, cnen.pad_token_id)]
```

---

## 坑 3：文本 Token 排列方式（Dense vs Sparse）

### 现象
坑 1+2 全修后 whisper 仍全输出 "ご視聴ありがとうございました"，但训练式 forward 算 loss 正常（Flow=0.02）。

### 根因

| | 训练 | 推理 `align_lrc_sentence_level` |
|---|---|---|
| Token 排列 | **Dense**: 所有 token 紧贴序列头 | **Sparse**: 按时间等间距分布到整个序列 |
| 示例 (T=124, 44 tokens) | `[1,2,3,...,44,0,0,...,0]` | `[1,0,0,2,0,0,3,0,0,...]` |

DiT 的 `TextEmbedding` 含 4 层 ConvNeXtV2Block（kernel=7），sparse token 经过卷积感受野完全不同 → text_embed 向量完全不对 → 即使 embedding 表正确，输出还是错的。

### 修复
用 dense pack 替代 sparse：
```python
text_tokens = torch.zeros(1, T, dtype=torch.long, device=device)
n = min(len(tokens), T)
text_tokens[0, :n] = torch.tensor(tokens[:n], device=device)
```

---

## 坑 4：DiT.get_input_embed 的 `cache=True` 默认值

### 现象
s0 推理成功，s1 报错 `Expected size 213 but got size 124`。

### 根因
`dit.py:342` 的 `get_input_embed` 缓存了 s0 的 `text_embed`/`text_uncond`（长度 124）。s1 的 text token 长度 213 → 遍历时复用缓存的 124 → 尺寸不匹配。

### 修复
全部显式传 `cache=False`。

---

## 坑 5：ODE Solver 无 CFG 导致模型忽略文本

### 现象
自重建输出不错（RMS 0.08~0.14），但改词翻唱（"私は思う"→"君を見る"）输出和原词一样。

### 根因
自重建时，音频条件（latent）+ 旋律条件（MIDI）本身已经包含完整信息，模型可以直接复制条件而忽略文本。需要 **Classifier-Free Guidance（CFG）** 强制模型按文本指引生成——即同时跑 conditional + unconditional 两路，用它们的差作为方向。

### 当前状态
简易 Euler ODE 在无 CFG 时自重建 OK，但 `Singer.sample()` 的完整 CFG 管线有 `max_duration == midi.shape[1]` 断言（因为 `duration = max(text_len, cond_len) + 1` 导致长度差 1）。

**待解决**：适配 `Singer.sample()` 使其接收 training-style text tokens。

---

## 坑 6：Whisper 评估（`vad_filter`）

### 现象
Whisper-medium 对唱歌转录完全错误，原唱 "カッカッカッ..." → "おーおーおー..."

### 根因
1. `faster-whisper` 默认 `vad_filter=True`，把唱歌切成短碎片 → 单段太短 → 瞎猜
2. **medium** 模型对唱歌音色敏感度远低于 **large-v3**

### 修复
```python
model.transcribe(audio, language="ja", beam_size=5, vad_filter=False)
```
必须用 **large-v3**。

### 验证
5 段原唱 `vad_filter=False` + large-v3 的 WER：
- s0 (シカ色デイズ): 0.0000 ✅
- s1 (灶门祢豆子): 0.0000 ✅  
- s2 (輝夜の城): 0.0000 ✅
- s3 (雪月花): 1.0000（注：カッカッカッ… 是节奏型拟声，对模型无意义）
- s4 (シュガーソング): 0.0000 ✅

---

## 坑 7：smoothMelody_MIDIFuzzDisturb 在推理时捣乱

### 现象
自重建音频的旋律和原唱略有偏差。

### 根因
`Singer.smoothMelody_MIDIFuzzDisturb()` 是训练时的数据增强（随机扰动 MIDI 时序），推理时不应该用。推理应该直接用干净的 MIDI。

### 修复
推理时 `midi = midi_p`（跳过 FuzzDisturb）。

---

## 修复汇总

| # | 文件 | 修改 |
|---|------|------|
| 1 | `cnen_tokenizer.py` | `encode()` 中 `has_japanese(text)` 时跳过 +1 |
| 2 | 推理脚本 | 过滤 `<SEP>`/`<PUNCT>` token |
| 3 | 推理脚本 | 文本 token 用 dense pack 排列 |
| 4 | 推理脚本 | `get_input_embed(cache=False)` |
| 5 | （待解决） | 适配 `Singer.sample()` 做 CFG 改词翻唱 |
| 6 | Whisper 脚本 | `vad_filter=False` + 用 large-v3 |
| 7 | 推理脚本 | 跳过 `smoothMelody_MIDIFuzzDisturb` |

---

## 当前 v2 结果

云盘 `yingmusic_ja_sft_eval/v2/` 下 30 条自重建音频：

| Checkpoint | RMS | 听感 |
|---|---|---|
| base | 0.40~0.46 | 噪音（日语 embedding 未训练） |
| step_002000 | 0.08~0.10 | 清晰日语歌声 ✅ |
| step_010000 | 0.08~0.11 | ✅ |
| step_020000 | 0.08~0.11 | ✅ |
| step_030000 | 0.08~0.10 | ✅（最接近原唱 RMS 0.10） |
| ref_audio | 0.10 | 原唱参考 |
