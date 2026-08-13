# YingMusic-Singer-Plus 日语微调 V2 计划

> 基于 V1 方案的失败复盘，逐模块重新规划。

---

## 一、CNENTokenizer — 日文 G2P 支持

### 1.1 当前事实

**CNENTokenizer 是什么**

[CNENTokenizer](repository-relative-source) 是 34 行的胶水层——把 G2P（文字→IPA音素→token ID）包装成统一的 encode/decode 接口：

```
CNENTokenizer.encode(text)
  → chn_eng_g2p(text)                    # 语言路由
    → get_segment(text)                   # 逐字符分类: zh / en / other
      → g2p(seg, text, lang)             # 调 PhonemeBpeTokenizer
        → cjekfd_cleaners(text, lang)    # lang→具体G2P后端
          → phoneme2token(raw_phoneme)    # IPA→token ID (0-based)
  → token = [x+1 for x in token]         # +1 (0号留给对齐层的PAD)
  return token                            # 1-based
```

**日文不可用的两个断点**

- **断点1** [g2p_generation.py:L54-L60](repository-relative-source): `get_segment()` 逐字符判断，假名和日文汉字既不匹配 `is_chinese` 也不匹配 `is_alphabet` → 全部判为 `"other"`
- **断点2** [cleaners.py:L26-L27](repository-relative-source): `cjekfd_cleaners` 对 `"other"` 直接抛 `"Unknown language"` 异常

两个断点连锁：路由错误 → 后端崩溃。

**两次 +1 的设计理由**

| 谁做的 | 零号位留给谁 | 为什么 |
|--------|-------------|--------|
| `CNENTokenizer.encode() +1` | PAD (序列填充) | `align_lrc_sentence_level` 做稀疏时间对齐时，空隙填 0 |
| `TextEmbedding.forward +1` | CFG filler (空文本) | CFG 的 uncond 分支需要 Embedding[0] 代表"无文本" |

### 1.2 G2P 后端选择: pyopenjtalk

`PhonemeBpeTokenizer` 内部注册了 `"ja": "ja"` 后端映射，指向 espeak-ng 的日语模块。但 **espeak-ng ja 产出的 IPA 符号集无法从源码静态验证**——一旦产出 vocab.json 中不存在的 IPA 符号，`phoneme2token` 会静默丢弃它，导致 token 序列残缺。

**pyopenjtalk 是唯一经过验证的后端**。从 Amphion/V1 中的 `japanese.py` (816行) 可提取其完整的 `jp_xphone2ipa` 映射表，共 44 个 IPA 符号。经逐项与 [vocab.json](repository-relative-source) 对比：

| 类别 | 数量 | 详情 |
|------|:---:|------|
| 共享音素 (CN/EN区域已有) | 16 | a, i, iː, o, k, s, t, n, m, j, ɾ, z, d, b, p, v |
| 日文独有音素 (318-345) | 26 | ɯ, e, aː, ɯː, eː, ç, ɸ, ɰᵝ, ɴ, q, ː, bj, tɕ, dej, tej, gj, gɯ, çj, kj, kɯ, mj, nj, pj, ɾj, ɕ, tsɯ |
| 日文独有音素 (other区域 346-358) | 1 | ʑ (ID=351) |
| `g` vs `ɡ` | 1 | pyopenjtalk 产出 ASCII `g` → matches ID=327，而非 `ɡ`(U+0261, ID=32) |
| **总计覆盖** | **44/44** | 全部命中 vocab.json |

**vocab.json 中的 `dʑ`(ID=328) 不会被 pyopenjtalk 使用**——pyopenjtalk 对 `じ/ジ` 产出 `ʑ`(ID=351)。

### 1.3 改造方案

需要改动 **4 个文件**：

| # | 文件 | 改动 | 理由 |
|---|------|------|------|
| 1 | `g2p/g2p/japanese.py` | **新建** — 从 Amphion maskgct 复制 (pyopenjtalk + pykakasi 后端, 816行) | 提供 `japanese_to_ipa(text, None)` |
| 2 | `g2p/g2p/cleaners.py` | `import japanese_to_ipa`；L17 `text_tokenizers["ja"]` → `None` | 绕过 espeak-ng，避免加载崩溃 |
| 3 | `g2p/g2p_generation.py` | 添加 `has_japanese()` 检测函数 → `chn_eng_g2p` 入口处整段 bypass | 绕过 `get_segment` 的断点1 |
| 4 | `cnen_tokenizer.py` | 无需改动 | `encode()` 中 `token = [x+1 for x in token]` 对日文也适用 |

**改造后的 token 路径（统一 CN/EN/JP）：**

```
CN/EN/JP 文本
  → chn_eng_g2p() (含日文检测)
    → 中英: get_segment → PhonemeBpeTokenizer.tokenize → pbt.phoneme2token → 0-based
    → 日文: japanese_to_ipa → pbt.phoneme2token → 0-based
  → CNENTokenizer.encode: token + 1  → 1-based
  → align_lrc_sentence_level          → 稀疏对齐, 空隙填 0(PAD)
  → DiT.TextEmbedding.forward: text + 1 → Embedding[原始值+2]
```

**日文与中英文的偏移量完全一致。**

### 1.4 验证方案

```python
from cnen_tokenizer import CNENTokenizer
from g2p import PhonemeBpeTokenizer
from g2p.japanese import japanese_to_ipa

tokenizer = CNENTokenizer()
pbt = PhonemeBpeTokenizer()

test_texts = [
    "たぶん私じゃなくていいね",
    "君を見るたびに思い出す",
    "カッカッカッカッカッ",
    "世界で一番お姫様",
]

for text in test_texts:
    # 基准: 直接走 pyopenjtalk
    phoneme = japanese_to_ipa(text, None)
    direct_tokens = pbt.phoneme2token(phoneme)
    
    # 改造后: 走 CNENTokenizer.encode()
    infer_tokens = tokenizer.encode(text)
    
    # CNENTokenizer 多一次 +1，所以这里要验证去掉这次 +1 是否匹配
    # (因为 DiT.TextEmbedding 内部也会 +1，两者最终要一致)
    match = direct_tokens == [t-1 for t in infer_tokens]
    print(f"'{text[:20]}': {'✅' if match else '❌'}")
```

> ⚠️ 最终的一致性要到 DiT TextEmbedding 内部才对齐。本测试验证的是：CNENTokenizer 的 +1 偏移量对日文是否可逆/可预测。

### 1.5 日文独有音素 — Embedding 训练状态

Plus 官方权重仅训练了 CN+EN 数据。DiT 的 `nn.Embedding(374, 512)` 中，被梯度更新过的只有 CN+EN 训练数据实际查过的行——其余行保留 PyTorch 默认 `N(0,1)` 随机初始化。

pyopenjtalk 产出的 44 个 IPA 符号的逐项训练状态：

**✅ 被 CN/EN 训练过（17 个, 39%）**

| 拼音 | ID | 训练来源 |
|------|:---:|------|
| `a` | 210 | CN |
| `i` | 55 | CN + EN |
| `iː` | 5 | EN |
| `o` | 215 | CN |
| `k` | 31 | EN |
| `s` | 37 | EN |
| `t` | 29 | EN |
| `n` | 45 | EN + CN |
| `m` | 44 | EN + CN |
| `j` | 47 | EN + CN |
| `ɾ` | 58 | EN |
| `z` | 38 | EN |
| `d` | 30 | EN + CN |
| `b` | 28 | EN + CN |
| `p` | 27 | EN + CN |
| `v` | 34 | EN |
| `ç` | 68 | EN/DE/FR 区域 |

**🔴 从未被训练（27 个, 61%）**

| 拼音 | ID | 原因 |
|------|:---:|------|
| `ɯ` | 318 | CN+EN 都不用此元音 |
| `e` | 319 | EN 用 `ɛ`(17) 和 `eɪ`(19)，不用纯 `e` |
| `aː` | 320 | EN 用 `ɑː`(12)，不同 Unicode |
| `ɯː` | 321 | 日文独有长音 |
| `eː` | 322 | 日文独有长音 |
| `ç` | 323 | 🔴 与 `ç`(U+00E7, ID=68) 是不同 Unicode 字符（`ç`=U+0063+U+0327） |
| `ɸ` | 324 | 日文独有（ふ的辅音） |
| `ɰᵝ` | 325 | 日文独有（わ的辅音） |
| `ɴ` | 326 | 日文独有（拨音 /N/） |
| `g` | 327 | 🔴 EN 用 `ɡ`(U+0261, ID=32)，pyopenjtalk 产出 ASCII `g`(U+0067) |
| `q` | 329 | 日文独有 |
| `ː` | 330 | 日文独有长音标记 |
| `bj` | 331 | 日文拗音 |
| `tɕ` | 332 | 日文独有（ち的辅音）；CN 拼音 `x` 理论上是 `ɕ` 但 CN 区域用了另一套带声调的标注，未触发此 ID |
| `dej` | 333 | 日文拗音 |
| `tej` | 334 | 日文拗音 |
| `gj` | 335 | 日文拗音 |
| `gɯ` | 336 | 日文拗音 |
| `çj` | 337 | 日文拗音 |
| `kj` | 338 | 日文拗音 |
| `kɯ` | 339 | 日文拗音 |
| `mj` | 340 | 日文拗音 |
| `nj` | 341 | 日文拗音 |
| `pj` | 342 | 日文拗音 |
| `ɾj` | 343 | 日文拗音 |
| `ɕ` | 344 | 日文独有（し的辅音）；CN 区域(75-317)未触发此 ID |
| `tsɯ` | 345 | 日文独有（つ） |
| `ʑ` | 351 | 日文独有（じ的辅音），位于 "other" 区域 346-358 |

### 1.6 风险评估

**27 个 embedding 向量初始值为 `N(0,1)` 随机值，从未被梯度更新。** 与 V1 日语 SFT 面对的情况完全一致——V1 的 366 词表中日文独有 embedding 同样未被训练，最终 loss 从 1.0 降到 0.33，咬字变为可辨识。风险可控。

| 风险 | 原因 | 概率 | 应对 |
|------|------|:---:|------|
| 早期训练不稳定 | 27 个 untrained embedding 同时接收大梯度 | 中 | warmup 500步，低初始 LR (7e-6) |
| 日文 embedding 训练不足 | 日文 token 占比约 60%，但 90.7h 数据量充足 | 低 | 30k 步 ≈ 24 epoch，足够 |
| CN/EN 能力遗忘 | Embedding 层整体统计分布被日文梯度改变 | 中 | 训练数据混 5-10% CN/EN |
| `g`(327) vs `ɡ`(32) 分离 | 两个 Embedding 行都表示 /g/ 音，互不通信 | 低 | pyopenjtalk 只用 327，日文不受影响 |
| `ç`(323) vs `ç`(68) 重复 | 两个嵌入向量表示几乎相同的音 | 低 | 日文只用 323，CN/EN 只用 68，各自独立 |

---

## 二、cond — 训练策略修复

### 2.1 问题：cond = GT（每帧都是答案）

**cond 是什么**

cond 不是全局共享的 "timbre vector"，而是 VAE 编码器输出的逐帧 64 维 latent。每帧的 latent 编码了该帧的完整声学信息：音色 + 音素形状 + 音高 + 响度。

**推理时的正确行为** [model.py:L290-L300](repository-relative-source):

```
帧 0..T_ref-1:  cond = ref_audio 的 VAE latent  (target singer真实音频, 64维/帧)
帧 T_ref..T_end: cond = 0                        (零！target 区域无 cond)
```

target 区域没有 cond 的原因是：用户只给了 ref_audio（3 秒的target singer歌声），没有音频可以编码 target 区域的帧。模型必须通过 self-attention 从 cond 帧中提取音色，然后在 target 区域仅凭音色记忆 + text + midi 生成歌声。

**训练时 train_plus.py 的实际行为** [train_plus.py:L283+L322](repository-relative-source):

```python
# process_batch:
latent = vae.encode_audio(w)   # 完整音频 → [T, 64]

# compute_loss:
v_pred = run_dit(x_t, latent, ...)   # cond = latent = 完整音频的 VAE 编码
```

```
帧 0..T-1（全部帧）: cond[n] = x₁[n] = 该帧的 GT latent

DiT 每帧看到: [ x_t | cond(=答案) | text | midi ]
```

80% 的训练步中 `drop_audio=False` → cond 每帧都是标准答案。模型学到的最短路径是 `v_pred ≈ denoise(x_t) towards cond`，不需要读 text 和 midi。

**后果**：
- 自重建 (RMS 0.08) 完美——因为 cond 里有原唱答案
- 改词翻唱完全失败——模型抄 cond 里的原歌词，忽略新文本

### 2.2 修复：前段 → cond，整段 → GT（方案 A，推荐）

回归 V1 的训练策略：

```
同一段target singer歌声(108帧):
  cond = VAE.encode(前 54 帧)    → [54, 64]   target singer参考
  x₁   = VAE.encode(全部 108 帧) → [108, 64]  完整 GT
  
  尾部填零: cond = [cond(54帧) | zeros(54帧)] → [108, 64]

DiT 每帧看到:
  帧 0..53:   cond = target singer真实音频  (有答案，用于音色学习)
  帧 54..107: cond = 0             (无答案！必须用 text + midi + self-attention)
```

**为什么会学到文本条件**：

帧 54-107 完全没有任何 cond 直接信息。DiT 只能：
1. 通过 self-attention attend 到帧 0-53，提取target singer
2. 读 text[54-107]，知道应该发什么音素
3. 读 midi[54-107]，知道应该唱什么旋律
4. 三者组合，预测从 noise 到干净 latent 的速度场

训练和推理在 target 区域完全对齐：cond=0 → 模型被迫依赖 text 和 midi。

### 2.3 备选方案 B：另一段 → cond

```
当前样本 x₁: target singer唱 A 歌的片段

cond 来源: 随机从target singer集中选另一段（不同歌词、不同旋律）:
  cond = VAE.encode(另一段 3 秒)
```

优势是更强的训练信号（cond 的歌词和旋律与 target 完全不同，模型无法"续词"），劣势是需要数据筛选保证音色一致性（target singer的不同直播音色有差异）。

**当前策略：方案 A 起步。** 如果方案 A 已完成且改词翻唱有效但跨歌翻唱不够好，可以升级到 B——训练数据不变，只改 cond 选择逻辑。

### 2.4 其他相关调整

| 调整项 | 原值 | 新值 | 原因 |
|------|:---:|:---:|------|
| cond 来源 | 完整 latent (T帧) | 前 T_ref 帧 (T/2) | 与推理一致 |
| cond 区域外填充 | 无 (全帧都有值) | zeros | 与推理一致 |
| drop_audio 概率 | 20% | **30%** | 更多步中迫使模型依赖 text/midi |
| 样本复用 | cond=GT (同一tensor) | cond 和 x₁ 不同 tensor | 尾部填零 |

具体的代码改动在 `process_batch` 中：

```python
# --- 当前 ---
latent = vae.encode_audio(w)   # [T, 64]
return latent, ...              # latent 同时当 cond 和 x₁

# --- 修复后 ---
full_latent = vae.encode_audio(w)              # [T, 64] → 完整 GT
ref_len = full_latent.shape[-1] // 2            # 前 50% 为 cond 区域
cond_latent = full_latent[..., :ref_len]        # [ref_len, 64]

# cond: 填零到 T 帧，[cond_latent(ref_len) | zeros(T-ref_len)]
cond = torch.zeros_like(full_latent)
cond[..., :ref_len] = cond_latent
cond = cond.transpose(1, 2)                     # [T, 64] 与推理格式一致

return full_latent, cond, ...  # x₁ 和 cond 是两个不同 tensor

---

## 三、midi — cond 区域清零

### 3.1 midi 是什么

midi 不是标准离散 MIDI 编号，而是 **SOME Teacher 从音频中逐帧提取的 128 维连续旋律表示**。

```
音频 → Mel Spectrogram(80维) → 8层 Conformer
  → midi_p  [T, 128]   128维连续值, 每帧≈一个"音高概率分布"
  → bound_p [T, 1]     每帧的"音符边界"概率
```

进入 DiT 前经过 `nn.Linear(128, 128)` 投影（midi_proj），`initialize_weights()` 中该层被置零——训练初始 midi 通道贡献为零。

FuzzDisturb [YingMusic_Singer.yaml:L68-L73](repository-relative-source): `drop_type: equal_space`, `drop_prob: [1, 9]`——每 10 帧丢 1 帧。

### 3.2 问题：训练时 cond 区域的 midi 未清零

推理时 [model.py:L306](repository-relative-source) 显式清零：

```python
midi = torch.where(cond_mask, torch.zeros_like(midi), midi)
# cond 区域(=ref_audio 区域): midi = 0
# target 区域:               midi = 目标旋律
```

训练时 [train_plus.py L266-L270](repository-relative-source) 全帧都有值，仅做了 FuzzDisturb 随机丢帧：

```
帧 0..T-1: 全部有 midi = target singer原唱的旋律
```

训练时所有帧都有 midi → DiT 学到的 self-attention 分布中 cond 区域的 K 包含 midi 信息 → 推理时 cond 区域的 midi=0 → attention pattern 偏移。

### 3.3 严重程度

**不是阻塞问题。** midi 不同于 cond latent——它不直接泄露"答案"，只是 128 维旋律辅助信号。官方 CN/EN 模型在同样 gap 下能正常工作。但修复开销极低（一行代码），建议修掉。

### 3.4 修复

在 `process_batch` 中，与 cond 修复共用 `ref_len`：

```python
midi = raw_model.smoothMelody_MIDIFuzzDisturb(midi_p)
midi[:, :ref_len, :] = 0   # cond 区域 midi 清零
```

### 3.5 为什么 cond 区域不该有 midi

cond 区域有 midi 意味着 self-attention 中 target 帧能看到 ref_audio 的旋律：
- **方案 A（同段）**：ref 和 target 旋律是同源的 → leak 危害有限，但仍是不必要的 gap
- **方案 B（异段）**：ref 和 target 是两首不同歌 → 两个互相矛盾的旋律来源使模型困惑，**必须清零**

无论哪种方案，cond 区域的 midi 清零都是正确的——cond 只负责音色，midi 在 target 区域独立提供旋律。两者各司其职。

---

## 四、CFG dropout 设计 — 无需修改

### 4.1 TextEmbedding 帧级对齐与停顿帧

TextEmbedding 输出 `[B, T, 512]`——每帧独立。序列长度 T 与 cond、midi、x_t 完全一致（在 InputEmbedding 中逐帧拼接后投影到 1024 维），但 **text 通道不对齐**——DiT 通过 22 层全帧双向 self-attention 隐式学习词组到帧的映射。

停顿帧不需要显式"静音 token"。当歌词 token 序列填不满整个 T 帧时，尾部自动以 0（PAD→Embedding[1]）填充。midi 通道同时通过 SOME Teacher 的 rest 检测输出低置信度区域。DiT 在训练中学会：text=PAD + midi=rest → 这是一个停顿帧。

### 4.2 drop_text 的第 0 行机制

**TextEmbedding.forward** [dit.py:L112-L136](repository-relative-source):

```python
def forward(self, text, seq_len, drop_text=False, ...):
    text = text + 1                    # 0-based → 1-based
    text = F.pad(text, ..., value=1)   # PAD = 1
    if drop_text:
        text = torch.zeros_like(text)  # 全零覆盖 → Embedding[0]
    text = self.text_embed(text)
```

`self.text_embed = nn.Embedding(374, 512)`。第 0 行被设计为 "filler token"——CFG 的 "无文本" 场景专用。

当 `drop_text=True` 时，原始 token 序列被全零覆盖，所有帧统一查 Embedding[0]。这不是"扰乱 token 分布"，而是**彻底替换**——不管原来是什么音素，全部抹掉后共享同一个 512 维表征。Embedding[0] 是一个 per-frame 的表征（不是序列级），被训练来代表"这一帧没有文本信息时应输出的默认倾向"。

### 4.3 训练覆盖：独立 null 学习 + 线性可加性

训练中三个条件独立随机 drop：

| 条件 | null 出现频率 | null 表征 |
|------|:---:|------|
| text 空 | 30% | Embedding[0]（全零→查表，梯度反向传播更新）|
| midi 空 | 30% | `zeros → midi_proj → 全零输出` |
| cond 空 | 20% | `zeros → embedding 层 → 全零` |

每个条件的"空"状态被单独且充分地学习。三者在推理时组合为 drop_all 时：

```
drop_all = W_x·x_t + W_c·0 + W_t·Embedding[0] + W_m·0
```

InputEmbedding 的 Linear 投影中三个通道的贡献是可加的——独立学习好的 null 表征线性组合后就是 drop_all 的正确行为。三者的联合出现频率（1.8%）不构成问题。

### 4.4 CFG 推理公式

```python
# model.py L374
v = v_cond + (v_cond - v_drop_all) × cfg_strength
```

| cfg_strength | 含义 |
|:---:|------|
| 0 | 纯 uncond——不理文本和旋律 |
| 1 | 基础条件——正常读所有提示 |
| 3（默认）| 3 倍放大"条件方向" |
| >5 | 过度强调，可能失真 |

### 4.5 cfg_infer_ids 的自由度

[model.py:L370](repository-relative-source):

```python
cfg_infer_ids = (use_cond, use_uncond, use_uncond_cc, use_drop_all)
```

| # | 变量名 | text | cond | midi | 训练覆盖 | 用途 |
|---|--------|:---:|:---:|:---:|:---:|------|
| 0 | cond | ✓ | ✓ | ✓ | 39.2% | 全条件 |
| 1 | uncond | ✓ | ✗ | ✓ | 9.8% | 仅丢音色 |
| 2 | uncond_cc | ✗ | ✓ | ✗ | 7.2% | 丢文本+旋律 |
| 3 | drop_all | ✗ | ✗ | ✗ | 1.8% | 全丢 |

四条路径均被训练覆盖。当前官方只用 (0,3) 两路做 CFG。如果将来需要独立控制"跟文本紧不紧"vs"跟旋律紧不紧"，可以直接启用路径 1 和 2，**不需要重新训练**。

### 4.6 结论：方案 A（不改）

CFG dropout 设计是正确的，不需要修改。三个独立的 null 学习已充分覆盖所有组合，丢弃率的当前值（text 30%、midi 30%、cond 20%）合理。继续下一模块。

















