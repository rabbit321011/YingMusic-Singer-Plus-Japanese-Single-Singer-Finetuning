# YingMusic-Singer-Plus 确定事实汇编

> 基于源码 `YingMusic-Singer-Plus-src/` 与 `scripts_archive/yingmusic_plus/` 的代码级事实。仅包含已确认的内容。

***

## 一、项目溯源

- 仓库: [ASLP-lab/YingMusic-Singer-Plus](https://github.com/ASLP-lab/YingMusic-Singer-Plus)
- 官方定位: CN+EN 双语歌唱合成 (SVS)，支持 SingEdit 和跨角色翻唱
- 与 V1 (YingMusic-Singer) 是不同团队的产物：V1 来自 GiantAILab/ASLP/UCL，Plus 来自 ASLP-lab
- 许可证: CC BY 4.0

***

## 二、四大组件

### 2.1 VAE — Stable Audio 2 (156.1M, 冻结)

- 来源: Stability AI 的 Stable Audio 2
- 输入: 44.1kHz 立体声
- 输出: **21.53 Hz 帧率 × 64 维** latent（压缩比 = 44100/21.53 ≈ 2048 倍）
- 推理代码: [vae\_copysyn.py](repository-relative-source)
- 权重: `stable_audio_2_0_vae_20hz_official.ckpt` (596 MB)
- 纯声学编解码，语言无关

### 2.2 Melody Extractor — SOME Teacher (117.6M, 冻结)

- 架构: 8 层 Conformer
- 输入: Mel 频谱 (80 维) → 输出: **128 维 MIDI 音高 + 音符边界**
- 权重: `model_ckpt_steps_100000_simplified.ckpt` (449 MB)
- 纯声学→MIDI 映射，语言无关
- 训练配置 [YingMusic\_Singer.yaml](repository-relative-source): `melody_input_source: some_pretrain_fuzzdisturb`
- FuzzDisturb [YingMusic\_Singer.yaml:L68-L73](repository-relative-source): `drop_type: equal_space`, `drop_prob: [1, 9]`, `noise_scale: 0.0` — 随机丢弃1-9帧，不加噪声，仅 dropout

### 2.3 CNENTokenizer — 中英 IPA 分词器

源码: [cnen\_tokenizer.py](repository-relative-source)（仅 34 行）

**词表** [vocab.json](repository-relative-source): 363 个 IPA 音素（含标点），0-based 索引。

日语音素区间: ID 318-345（28 个 token）:

```
ɯ(318), e(319), aː(320), ɯː(321), eː(322), ç(323), ɸ(324), ɰᵝ(325),
ɴ(326), g(327), dʑ(328), q(329), ː(330), bj(331), tɕ(332), dej(333),
tej(334), gj(335), gɯ(336), çj(337), kj(338), kɯ(339), mj(340),
nj(341), pj(342), ɾj(343), ɕ(344), tsɯ(345)
```

**CNENTokenizer 初始化** [cnen\_tokenizer.py:L6-L26](repository-relative-source):

```python
self.phone2id = {k: int(v) + 1 for (k, v) in self.phone2id.items()}  # 全部+1
self.pad_token_id = 0           # <PAD>
self.punct_token_id = 364       # <PUNCT>，运行时追加
self.sep_token_id = 365         # <SEP>，运行时追加
```

**encode() 方法** [cnen\_tokenizer.py:L28-L31](repository-relative-source):

```python
def encode(self, text):
    phone, token = self.tokenizer(text)   # → chn_eng_g2p()
    token = [x + 1 for x in token]        # 又 +1
    return token
```

vocab值已+1 → `chn_eng_g2p` 返回的 `pbt.phoneme2token()` 是 0-based → 再 +1。最终 token ID 是 **0-based + 2**。

**语言路由** [g2p\_generation.py:L45-L107](repository-relative-source):

- `get_segment()` 逐字符判断: `is_chinese(ch)` → "zh" / `is_alphabet(ch)` → "en" / 其余 → "other"
- `chn_eng_g2p()` 对 "other" 段调用 `PhonemeBpeTokenizer.tokenize(seg, text, "other")` → 内部尝试英文 phonemizer → 日文汉字/假名无法处理

### 2.4 DiT CFM — F5-TTS 架构 (453.6M, 可训练)

配置文件 [YingMusic\_Singer.yaml:L41-L54](repository-relative-source):

```yaml
dim: 1024, depth: 22, heads: 16, ff_mult: 2
text_dim: 512, conv_layers: 4
text_num_embeds: 373      # 注意: 比 vocab.json(363) 大10
```

**TextEmbedding** [dit.py:L31-L166](repository-relative-source):

```python
self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # [374, 512]
# index 0 reserved as filler token

def forward(self, text, seq_len, drop_text=False, ...):
    text = text + 1           # 外部 token 索引从 0 开始，内部 +1
    text = text[:, :seq_len]
    text = F.pad(text, (0, seq_len - text_len), value=1)  # PAD = 1
    if drop_text:
        text = torch.zeros_like(text)   # CFG 时全零替代
    text = self.text_embed(text)        # [B, n] → [B, n, 512]
    # 4层 ConvNeXtV2Block (kernel=7) + RoPE 位置编码
```

- Embedding 表实际使用范围: idx 1\~374（0 是 filler，从未被有效 look-up）
- 外部传入的 0-based token → 内部变为 idx\[1, 374]
- drop\_text 时用全零 token → idx=0 → 查 Embedding 表第 0 行（filler，未经训练）

**InputEmbedding** [dit.py:L172-L198](repository-relative-source):

```python
self.proj = nn.Linear(mel_dim * 2 + text_dim + midi_dim, out_dim)
# = Linear(64*2 + 512 + 128, 1024) = Linear(768, 1024)

def forward(x, cond, text_embed, midi, drop_audio_cond, drop_midi):
    if drop_audio_cond:
        cond = torch.zeros_like(cond)
    if drop_midi:
        midi = torch.zeros_like(midi, dtype=torch.long)
    x = self.proj(torch.cat((x, cond, text_embed, self.midi_proj(midi)), dim=-1))
    x = self.conv_pos_embed(x) + x
```

**DiTBlock** [modules.py:L700-L748](repository-relative-source): AdaLN 调制 — 时间步 t 经 `SiLU→Linear(dim, 6*dim)` 生成 6 组参数，控制 SelfAttention + FFN 的缩放/偏移/门控。

**CKA 对齐损失** [common.py](repository-relative-source): 提取 DiT 最后 3 层的 hidden states，与 MIDI reference 计算 Gram 矩阵的 CKA 相似度。配置 `cka_disabled: 0`（启用）。

**CFG drop\_text 的 filler token 问题** [dit.py:L135-L136](repository-relative-source):

```python
if drop_text:
    text = torch.zeros_like(text)  # 全零 → idx=0
```

`self.text_embed = nn.Embedding(374, 512)` 的第 0 行（filler）从未被训练过——有效音素占用行 1-364。CFG 的 uncond 分支查 Embedding\[0] 得到一个随机初始化的向量，而非"无文本条件"的有意义表征。

`initialize_weights()` [dit.py:L282-L292](repository-relative-source) 只显式置零 AdaLN 层和 proj\_out，Embedding 表保留 PyTorch 默认的 `N(0,1)` 随机初始化。

***

## 三、Flow Matching 训练原理

```
x_t = (1-t) × noise + t × x₁         线性路径
v_target = x₁ - noise                 目标速度场
L_flow = MSE(v_pred, v_target)
L_cka  = cka_loss(hidden_states[-3:], midi_reference)
L_total = L_flow + λ × L_cka
```

- 采样 t: `torch.rand(B)` U(0,1)
- 优化器: AdamW β=(0.9, 0.95), weight\_decay=1e-2
- 学习率: 7e-6, warmup 500步→Cosine衰减
- 梯度裁剪: max\_norm=1.0
- EMA: β=0.995, update\_after\_step=100
- 精度: fp32（train\_plus.py 未启用 amp）

***

## 四、标准推理流程

[YingMusicSinger.py](repository-relative-source) 的 `forward()`:

```
1. ref_audio → VAE → ref_latent [1, T_ref, 64]      (音色条件)
2. melody_audio → VAE → melody_latent [1, T_mel, 64]  (旋律源)
3. [ref_latent | melody_latent] → Mel Spec → SOME → midi_p [1, T, 128], bound_p
4. ref_text + target_text → CNENTokenizer.encode()
    → chn_eng_g2p() → phone/token → token + 1
    → align_lrc_sentence_level(cnen_tokenizer, start_times=[0, T_ref/frame_rate], ...)
    → 稀疏排列 + <SEP>/<PUNCT> 分隔 → text_tokens [1, T]
5. Singer.sample(cond=ref_latent, text=text_tokens, midi_p=midi_p, bound_p=bound_p, ...)
    → 32步 ODE + CFG(cfg_strength=3.0) → generated_latent
6. 截取 target 段 [T_ref : -rear_silent] → VAE decode → 44.1kHz 立体声
```

***

## 五、Singer.sample() 的 CFG 机制

[model.py:L357-L374](repository-relative-source):

```python
pred_cfg, _ = self.transformer(
    x=x, cond=step_cond, text=text, midi=midi, time=t,
    cfg_infer=True, cfg_infer_ids=(True, False, False, True),
)
pred, pred_drop_all_cond = torch.chunk(pred_cfg, 2, dim=0)
return pred + (pred - pred_drop_all_cond) * float(guidance_scale)
```

`cfg_infer_ids=(True, False, False, True)` 的含义:

| # | 变量名                | text | cond | midi | 用途       |
| - | ------------------ | :--: | :--: | :--: | -------- |
| 0 | x\_cond            |   ✓  |   ✓  |   ✓  | 全条件      |
| 1 | x\_uncond          |   ✓  |   ✗  |   ✓  | 仅丢弃音色条件  |
| 2 | x\_uncond\_cc      |   ✗  |   ✓  |   ✗  | 仅丢弃文本和旋律 |
| 3 | x\_drop\_all\_cond |   ✗  |   ✗  |   ✗  | 丢弃所有条件   |

实际只用 (0, 3): `v = v_cond + (v_cond - v_drop_all) × cfg_strength`

这意味着 CFG 同时放大文本和音色和旋律的方向——三者耦合。

***

## 六、train\_plus.py 训练脚本的实际行为

[scripts\_archive/yingmusic\_plus/2\_train\_sft/train\_plus.py](repository-relative-source)

### 6.1 数据处理

```python
def process_batch(wavs, srs, texts, langs):
    latent = vae.encode_audio(w)     # 完整音频 → latent [T, 64]
    midi_p = SOME(mel_spec)          # 完整音频 → MIDI [T, 128]
    midi = FuzzDisturb(midi_p)       # 随机丢弃帧
    
    # Tokenize: 绕过 CNENTokenizer，直接用训练式路径
    tokens = encode_ja(text)         # japanese_to_ipa → pbt.phoneme2token → 0-based
    aligned_text[0, :n] = tokens[:n] # Dense pack
```

### 6.2 训练循环

```python
def compute_loss(wavs, srs, texts, langs):
    latent, midi, midi_p, aligned_text, B, T, D = process_batch(...)
    
    t = torch.rand(B)
    noise = torch.randn_like(latent)
    x_t = (1-t)*noise + t*latent
    v_target = latent - noise
    
    drop_audio = random() < 0.2    # 20% 丢弃 cond
    drop_text  = random() < 0.3    # 30% 丢弃文本
    drop_midi  = random() < 0.3    # 30% 丢弃 MIDI
    
    v_pred = run_dit(x_t, latent, aligned_text, t, midi,
                     drop_audio, drop_text, drop_midi)
    #                                           ↓
    # run_dit 内部: get_input_embed(x_t, latent, text, midi, ...)
    #                                            ↓
    # cond 参数 = latent = 完整音频的 VAE 编码 ← 关键
```

### 6.3 cond 参数的本质

`run_dit` 将 `latent` 同时用作:

- `cond` (音色条件) → 传入 `get_input_embed(x_t, cond, ...)`
- `v_target = latent - noise`（训练目标）

**cond 和 GT 是同一个 tensor。** 模型输入中，cond 的每一帧都包含了该帧最终输出应该长什么样的完整信息（音色+音素形状+音高+响度）。

80% 的训练步中 cond 未被丢弃 → 模型可以直接从 cond 提取答案，不需要读 text。

### 6.4 与 V1 训练的区别

V1 的 `train.py` (DEPLOY.md):

```
cond = VAE.encode(seg 前半段)   # 只覆盖音频的 50%
x₁   = VAE.encode(seg 整段)     # 完整音频
```

cond 长度 < x₁ 长度 → 后半段必须依赖其他条件（文本/旋律）。

Plus 的 `train_plus.py`:

```
cond = VAE.encode(seg 整段)     # 完整音频
x₁   = VAE.encode(seg 整段)     # 同一个东西
```

cond 长度 = x₁ 长度 → 每帧都是标准答案。

***

## 七、train\_plus\_v2.py 的尝试与失败

[scripts\_archive/yingmusic\_plus/2\_train\_sft/train\_plus\_v2.py](repository-relative-source)

改动: 将 `encode_ja()` (训练式, 0-based) 替换为 `tokenizer.encode()` (CNENTokenizer 路径, vocab+1+1):

```python
# v1 (train_plus.py):
tokens = encode_ja(text)  # 0-based, dense

# v2 (train_plus_v2.py):
tokens = tokenizer.encode(text)  # 1-based + sparse alignment
```

结果: 第一步 Flow=22.4 正常，到 50 步时 CNENTokenizer 对日文返回 "Unknown language: other" → 崩溃。`get_segment()` 将日文判为 "other" → `PhonemeBpeTokenizer` 尝试英文 phonemizer → 失败。

***

## 八、训练/推理 Tokenize 路径的完整对比

| 维度                            | 训练 train\_plus.py               | 推理 YingMusicSinger                                 |
| ----------------------------- | ------------------------------- | -------------------------------------------------- |
| 函数                            | `encode_ja()`                   | `CNENTokenizer.encode()`                           |
| G2P                           | `japanese_to_ipa(text, None)`   | `chn_eng_g2p(text)` → `get_segment` → ja判为 "other" |
| Token查表                       | `pbt.phoneme2token()` → 0-based | vocab值 +1 → `token + 1` → 再加 1                     |
| 最终 ID (以 "私" ≈ 音素 "ɯ"=318 为例) | 318 → DiT 内部 +1 = 319           | 318+1=319 → encode +1=320 → DiT 内部 +1=321          |
| 排列方式                          | Dense (紧凑排列, 剩余填0)              | Sparse (sentence\_level 时间对齐)                      |
| 特殊token                       | 无                               | <SEP>=365, <PUNCT>=364                             |
| 文本长度                          | 实际 token 数 (≤T)                 | total\_len (T\_ref+T\_mel), 超出时截断                  |

两者在两套 Embedding 表的完全不同位置查找，DiT 内部 ConvNeXtV2 的感受野因 dense/sparse 排列而完全不同。

***

## 九、已知的 JA SFT 训练结果

### 9.1 训练日志 (DEPLOY.md)

数据集: `final_sum_large/train_singnet.json` (15,122 条, 90.7h, 100% ja)
GPU: 4×RTX 4090, batch=6/GPU set grad\_accum=1
LR: 7e-6, warmup 500步→Cosine

```
Step   50: Flow=22.94  CKA=0.624  (warmup)
Step  500: Flow= 1.67  CKA=0.196  (峰值 LR)
Step 1000: Flow= 1.03  CKA=0.168  Eval=1.063
Step 1700: Flow= 0.67  CKA=0.162
```

30,000 步完成，checkpoint 保存于 `ckpts/plus_ja_sft/step_*`。

### 9.2 评估结果 (JA\_SFT\_DEBUG\_LOG.md)

使用 `simple_ode.py` 绕过 `Singer.sample()`，直接用欧拉 ODE + 训练式 tokenize:

| Checkpoint   |   自重建 RMS  | 听感     |
| ------------ | :--------: | ------ |
| base (官方)    | 0.40\~0.46 | 噪音     |
| step\_002000 | 0.08\~0.10 | 清晰日语歌声 |
| step\_010000 | 0.08\~0.11 | 清晰日语歌声 |
| step\_020000 | 0.08\~0.11 | 清晰日语歌声 |
| step\_030000 | 0.08\~0.10 | 最接近原唱  |

**自重建 (reconstruction) 成功。改词翻唱 (rewrite) 从未成功。** 在无 CFG 模式下，模型复制 cond 中隐含的原词信息，忽略给定的新文本。

***

## 十、DiT.forward 中 `text + 1` 的精确含义

[dit.py:L120](repository-relative-source):

```python
text = text + 1
```

`text` 是外部传入的 token tensor。+1 后：

- 原值 0 → 1 → Embedding 表第 1 行（第一个有效音素）
- 原值 k → k+1 → Embedding 表第 k+1 行
- 后续 padding 用 1（PAD）
- CFG drop\_text 时用 0（filler，未经训练的行）

Embedding 表大小: `text_num_embeds + 1 = 374`，索引范围 0-373。有效音素占用 1-364（vocab.json 的 0-363 → +1 = 1-364）。

***

## 十一、VAE 帧率精确值

[YingMusic\_Singer.yaml:L16](repository-relative-source):

```yaml
vae_frame_rate: 21.533203125
```

计算公式: 44100 / 2048 = 21.533203125 Hz

每帧对应 2048 个采样点 ≈ 46.4 ms。

***

## 十二、Training Plan 文档中的过时信息

`docs/training-plan.md` 在 "Phase 1: SFT 训练 🔄 进行中" 部分声称：

> "Step 1000: Flow=1.03 CKA=0.168 Eval=1.063 ✅"

该文档最后修改于 5/24 22:17，训练当时仍在进行。后续评估（JA\_SFT\_DEBUG\_LOG.md 5/25 19:03）揭示了自重建之外的所有推理路径均失败。该文档的 "✅" 结论基于不完整的评估。

***

## 十三、`docs/architecture-plus.md` 中的两处事实性补充

原文档中 "DiT CFM 参数量 = 453.6M" 来自官方标注。除此之外：

1. **CKA 代码位置确认为 Plus 独有**：V1 源码中不存在 CKA 相关代码，CKA 是 Plus 新增。原文档已注明此点但未引用具体代码行 → `common.py` 的 `cka_loss()` 函数。
2. **GRPO 权重分配** [YingMusic\_Singer.yaml:L88](repository-relative-source):
   ```yaml
   reward_config: {"qwen_asr_wer": 0.25, "f0_correlation": 0.25,
                   "qwenfeat": 0.25, "sim_wavlm_large": 0.25}
   ```
   四项奖励均等权重（各 25%）。GRPO 脚本 `grpo_train.py` 已就绪（位于 `scripts_archive/yingmusic_plus/3_train_grpo/`），但从未在全量权重上实际运行过。


















