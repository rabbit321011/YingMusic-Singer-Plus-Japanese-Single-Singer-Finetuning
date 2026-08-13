# YingMusic-Singer-Plus 架构详解

> 基于源码 [ASLP-lab/YingMusic-Singer-Plus](https://github.com/ASLP-lab/YingMusic-Singer-Plus) 分析，版本 2026-04-12

## 一、任务本质

YingMusic-Singer-Plus 是一个**旋律可控的歌唱声音合成 (SVS) + 歌词编辑**模型。用户只需提供三样东西：

| 输入 | 举例 | 作用 |
|------|------|------|
| 音色参考音频 | 某歌手的一段干声 | "用这个人的声音唱" |
| 旋律参考音频 | 另一首歌的干声 | "按这个旋律和节奏唱" |
| 目标歌词 | "Missing you in my mind" | "唱这句词" |

输出：一段 **44.1kHz 立体声** 的歌唱波形，音色像参考歌手、旋律跟随参考音频、咬字是目标歌词。

**核心亮点**：不需要人工标注音素时长、不需要 MIDI 钢琴卷、不需要音高序列标注。旋律从音频里自动提取，歌词和旋律之间靠 DiT 隐式对齐。

**当前限制**：仅支持 **中文和英文** 双语。不支持日语、韩语等（G2P 前端虽有接口但词汇表和中英训练数据是限制）。

## 二、整体架构 — 四核引擎

```
输入音频 ──→ VAE Encoder ──→ Latent (64维) ──→ DiT (22层) ──→ Output Latent ──→ VAE Decoder ──→ 44.1kHz 歌声
                                                    ↑
              Melody Ref ──→ Mel Spec ──→ SOME/MIDI ──→ 128维旋律
                                                    ↑
              Lyrics ──→ CNENTokenizer ──→ IPA Token (373词表)
```

| 组件 | 参数量 | 来源 |
|------|--------|------|
| VAE | 156.1M | Stable Audio 2 (冻结) |
| Melody Extractor | 117.6M | SOME 预训练 (冻结) |
| DiT CFM | 453.6M | F5-TTS 架构 (训练) |
| **总计** | **~727.3M** | |

## 三、逐模块拆解

### 3.1 VAE — Stable Audio 2 编码解码器

**来源**：Stability AI 的 **Stable Audio 2 VAE**（权重来自 [stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)，其 VAE 架构即为 SA2）。论文中明确使用 Stable Audio 2 的编码器/解码器。

与 V1 的 `stable_audio_1920_vae` 不同，SA2 VAE 提供了：
- 输入 44.1kHz 立体声 → 输出 **21.53Hz 帧率** 的 **64 维** latent
- 压缩比 = 44100 / 21.53 = **2048 倍**
- 支持立体声（V1 是单声道 48kHz、25Hz 帧率）

**代码证据** [YingMusicSinger.py:57-60](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/infer/YingMusicSinger.py#L57-L60)：`self.vae = StableAudioInfer(model_config_path, model_ckpt_path)`。

**VAE 在推理时的作用**：
1. 编码：参考音频和旋律音频 → latent 序列（拼接后作为 conditioning 和 melody 输入）
2. 解码：DiT 输出的 latent → 44.1kHz 波形

### 3.2 旋律提取器 — SOME 预训练 + 模糊扰动

旋律提取分两层：

**SOME MIDI Teacher（冻结）**：
- 预训练的 MIDI 提取模型，基于 **8 层 Conformer**（CNN + 注意力混合架构）
- 输入：Mel 频谱（80 维）
- 输出：128 维的连续 MIDI 音高表示 + 音符边界
- 通过高斯模糊解码得到连续 MIDI 值，阈值判定休止符
- Teacher 权重 449MB（`model_ckpt_steps_100000_simplified.ckpt`）

**训练时的模糊扰动 (FuzzDisturb)**：
- 训练配置 `melody_input_source: some_pretrain_fuzzdisturb`
- 随机丢弃部分帧的旋律信息（模拟真实场景的旋律不完整）
- 噪声尺度为 0.0（不加噪声，仅 dropout）
- 目的：让 DiT 学会在旋律信息不完整时也能合理推断

**代码证据** [SmoothMelody.py](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/melody/SmoothMelody.py)：`MIDIFuzzDisturb` 类，支持 `drop_prob=[1,9]` 即随机丢弃 1-9 个连续帧。

### 3.3 CNENTokenizer — 中英双语 IPA 分词器

**词表**：`vocab.json` 包含 **373 个 token**（IPA 音素 + 标点 + 特殊标记）

与 V1 的 PhonemeBpeTokenizer 类似，但词表不同（V1: 366 token，Plus: 373 token）。

**G2P 流程** [g2p_generation.py](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/utils/f5_tts/g2p/g2p_generation.py#L92-L107)：

```
输入文本 "你好，hello world"
  ↓ 按字符类型分段
["你好"(zh), "hello world"(en)]
  ↓ PhonemeBpeTokenizer
zh: "n i3 h ao3" → token [n, i3, h, ao3]
en: "h ə l oʊ w ɜː l d" → token [h, ə, l, oʊ, w, ɜː, l, d]
  ↓ 拼接
最终 token 序列: [n, i3, h, ao3, <PUNCT>, h, ə, l, oʊ, w, ɜː, l, d]
```

**关键设计**：
- `PhonemeBpeTokenizer` 支持 **6 种语言的 G2P 后端**（zh/ja/en/fr/ko/de）
- 但 `g2p_generation.chn_eng_g2p()` 只支持 **zh + en** 双语的自动分段
- 日语/韩语等虽然在 LangSegment 中有检测器，但训练数据未包含

**歌词对齐** [lrc_align.py](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/utils/lrc_align.py)：
- `sentence_level` 模式：每句歌词有一个起始时间戳，按时间错位排列 phoneme token
- 用 `<SEP>` 分隔句与句
- 这是**句子级对齐**，没有音素级的精确时间——DiT 靠学习来隐式对齐

### 3.4 DiT — 核心生成器 (F5-TTS 架构)

#### 3.4.1 架构规格

| 参数 | V1 (Wan2.1) | Plus (F5-TTS) |
|------|-------------|----------------|
| 层数 depth | 24 | **22** |
| 隐藏维 dim | 1024 | 1024 |
| 注意力头 heads | 16 | 16 |
| FFN 膨胀 ff_mult | 4 | **2** |
| 词表 text_num_embeds | 366 | **373** |
| 文本维 text_dim | 与 mel 相同 | **512** |
| conv_layers (文本编码) | 0 | **4** (ConvNeXtV2) |
| attn_backend | torch | torch (可选 flash_attn) |
| long_skip_connection | False | True |

**关键变化**：
- **文本有独立的维度 (512)**，不再和 mel 维度 (64) 共用
- **文本编码器前置 4 层 ConvNeXtV2**，增强文本表示能力
- **长跳跃连接** (`long_skip_connection`)：DiT 输入和输出之间加残差
- **RoPE** 旋转位置编码替代 V1 的正弦位置编码
- **核心 Transformer 更轻量**：22 层 × ff_mult=2 = 每层参数量约为 V1 的 **一半**（V1 为 24 层 × ff_mult=4）。但 Plus 增加了文本编码网络 (4 ConvNeXtV2 + 512维独立 embedding)，因此总 CFM 参数为官方公布的 453.6M（含 text_embed + input_embed + transformer_blocks + proj_out 全部）。

#### 3.4.2 输入嵌入：四路拼接

`InputEmbedding` [dit.py:172-198](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/models/dit.py#L172-L198)：

```
x_proj = Linear( [noised_x(64) | cond(64) | text_embed(512) | midi_proj(128)], out_dim=1024)
      = Linear( 768 → 1024 )
```

四个信息源在每一帧拼接后投影到 1024 维统一空间。DiT 的自注意力自己学会对齐。

#### 3.4.3 DiTBlock + AdaLN

每个 DiTBlock [modules.py:700-748](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/models/modules.py#L700-L748)：

```
x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = AdaLayerNorm(x, time_emb)  // 6组调制参数
attn_out = SelfAttention(x_norm, rope)   // 自注意力 + RoPE
x = x + gate_msa * attn_out
x_norm = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
x = x + gate_mlp * FFN(x_norm)
```

**AdaLN 调制**：时间步 t 经 `SiLU → Linear(dim, 6*dim)` 生成 6 组参数，控制注意力和 FFN 的缩放/偏移/门控。

#### 3.4.4 Flow Matching 训练

与 V1 一致的线性流：
- x_t = (1-t) × noise + t × x₁
- v_target = x₁ - noise
- Loss = MSE(v_pred, v_target)

#### 3.4.5 CFG 推理 (4-way 分类器自由引导)

Plus 的 CFG 比 V1 更精细 [model.py:320-374](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/models/model.py#L320-L374)：

```
cfg_infer_ids = (True, False, False, True)
  ├─ x_cond:          text✓  cond✓  midi✓   // 全条件
  ├─ x_uncond:        text✓  cond✗  midi✓   // 仅丢弃音频条件
  ├─ x_uncond_cc:     text✗  cond✓  midi✗   // 仅丢弃文本和旋律 (V1 的方式)
  └─ x_drop_all_cond: text✗  cond✗  midi✗   // 丢弃所有条件

v = pred_cond + (pred_cond - pred_drop_all) × cfg_strength
```

默认 `cfg_infer_ids=(True, False, False, True)` — 即用全条件和全丢弃做 CFG，这对应 V1 中"只对 text 做 CFG"的行为，但 Plus 同时 drop 了 cond、text、midi。

**重要发现**：当前硬编码的 CFG 配置下，**旋律和文本/条件一起被 CFG**，没有独立的旋律 CFG 强度控制。增强 CFG 会同时增强文本清晰度和旋律跟随——这是一个耦合约束。未来可通过调整 `cfg_infer_ids` 实现更细粒度的 CFG（如仅对文本 CFG、仅对旋律 CFG 等）。

### 3.5 CKA 对齐损失 — 旋律保真度机制

**CKA (Centered Kernel Alignment)** 损失在 SFT 训练阶段计算 DiT 中间表示与旋律 GT 之间的相似度，是 Plus 的旋律跟随优化核心：

> **验证事实**：经查 V1 源码 (`YingMusic-Singer/src/`) 中**不存在** CKA 相关代码（`grep -r CKA` 返回空）。CKA 是 Plus 新增的特性，非 V1 继承。

**原理**：
- 提取 DiT 的中间层特征 (hidden states)
- 提取 Melody Teacher 的特征表示
- 计算两者的 CKA 相似度矩阵（实现见 [`common.py:299`](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/utils/common.py#L299-L325) 的 `cka_loss` 函数）
- Loss：`L_total = L_flow + λ × (1 - CKA_sim)`

配置中 `cka_disabled: 0` 表示 CKA 启用（0 = 不禁用，即启用）。

### 3.6 GRPO — 强化学习精调

**GRPO (Group Relative Policy Optimization)** 是 Plus 的二阶段训练核心 [YingMusic_Singer.yaml:75-97](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/config/YingMusic_Singer.yaml#L75-L97)：

**为什么需要 GRPO？** SFT 阶段存在一个本质矛盾：
- 想忠实跟随旋律 → 模型可能"死记"原歌词的发音
- 想正确发出新歌词 → 旋律可能漂移

GRPO 用一个**奖励模型**来平衡：

```
奖励权重分配：
├─ Qwen ASR WER (0.25)    ← 歌词可懂度（越低越好）
├─ F0 Correlation (0.25)   ← 基频相关性（越高越好）
├─ Qwen Features (0.25)    ← 感知语义一致性
└─ WavLM Similarity (0.25) ← 说话人/音色相似度
```

**流程**：
1. 对每个样本生成 **8 个变体**（不同噪声种子）
2. 对每个变体打分（用 ASR + F0-CORR + Qwen + WavLM）
3. 用 GRPO 更新策略（带 clip 的 PPO 变种）
4. 噪声水平 noise_level=0.8，即从 80% 噪声位置开始采样

## 四、训练管线

### 4.1 SFT 阶段

> **数据来源**：以下参数来自项目源码 [YingMusic_Singer.yaml](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/config/YingMusic_Singer.yaml)，非论文公开发表数据，仅供参考。

| 参数 | 值 |
|------|-----|
| 学习率 | 7e-6 |
| 预热步数 | 60 |
| 总步数 | 31,518 |
| 批量大小 | 6 / GPU |
| 梯度裁剪 | 1.0 |
| EMA | beta=0.995, 100步后开始 |
| 保存频率 | 每 100 步 |

数据格式：每条数据包含音色参考音频、旋律参考音频、歌词文本（自给自足）。

### 4.2 GRPO 阶段

在 SFT 模型基础上，用强化学习调优，目标是最小化 WER 同时最大化 F0 相关性。

## 五、推理流程详解

以命令行调用为例 [infer_api.py](file:///${LOCAL_PROJECT_PATH}/YingMusic-Singer-Plus-src/src/YingMusicSinger/infer/YingMusicSinger.py)：

> **关键设计**：`--ref_audio` (音色参考) 和 `--melody_audio` (旋律参考) 是两个**独立参数**，可以指定为不同音频。这就是跨角色翻唱的基础：用 A 的声音唱 B 的旋律。

```
1. 加载音频
   ├─ ref_audio → VAE encode → ref_latent [B, T_ref, 64]     // 音色条件
   └─ melody_audio → VAE encode → melody_latent [B, T_mel, 64] // 旋律源

2. 拼接 latent (用于 teacher melody extraction)
   midi_in = [ref_latent | melody_latent]
   注意: ref_latent 在前作为旋律上下文，melody_latent 在后提供目标旋律

3. 提取 Mel 频谱 + MIDI
   ├─ [ref_mel | melody_mel] → Mel Spectrogram (24kHz, 80维)
   └─ Mel Spectrogram → SOME Teacher → midi_p (128-dim MIDI), bound_p (边界)
   注意: ref 部分的 MIDI 会被置零（仅 melody 部分提供旋律引导）

4. 文本处理
   ├─ ref_text · target_text → CNENTokenizer → token 序列 [1, T_total]
   └─ sentence_level 对齐: ref 放在时间=0，target 放在 ref_latent_len 之后
      T_total = T_ref + T_mel (两个 latent 的总帧数)

5. Flow Matching 采样 (ODE)
   ├─ 初始: noise ~ N(0, I), t_start=0
   ├─ t ∈ [0, 1] 线性分 32 步 (t_shift=0.5 非线性扭曲)
   ├─ 每步: v = DiT(x_t, cond=ref_latent, text, midi, t)
   │         apply CFG: v = v_cond + (v_cond - v_drop_all) * cfg_strength
   └─ 最终: x_1 = 干净 latent

6. 后处理
   ├─ 截取目标段 [T_ref+sil_len : -(rear_silent)]
   │   生成的是 ref+melody 全长，只保留 target 对应部分
   ├─ VAE decode → 44.1kHz 立体声波形
   └─ Smooth ending (淡出尾部)
```

**关键参数**：
- `nfe_step=32`：ODE 步数，越大质量越好但越慢
- `cfg_strength=3.0`：CFG 强度，控制歌词+旋律的跟随程度
- `t_shift=0.5`：时间偏移，控制采样密度在 t→1 时更密
- `sil_len_to_end=0.5`：参考音频尾部追加的静音长度

## 六、Plus vs V1 核心对比

| 维度 | V1 (YingMusic-Singer) | Plus (YingMusic-Singer-Plus) |
|------|----------------------|------------------------------|
| **DiT 骨干** | Wan2.1 (24层, ff_mult=4) | F5-TTS (22层, ff_mult=2) |
| **DiT 核心参数量** | 较大 (24层×ff_mult=4) | 更轻 (22层×ff_mult=2)；总 CFM 含文本编码等 = 453.6M |
| **文本编码** | 无前置网络，直接 Embedding | 4层 ConvNeXtV2 + 独立 512维 |
| **位置编码** | Sinusoidal | RoPE 旋转位置编码 |
| **VAE** | stable_audio_1920 (48kHz单声道) | Stable Audio 2 (44.1kHz立体声) |
| **Latent 帧率** | 25 Hz | 21.53 Hz |
| **语言支持** | CN + EN + JA (训练后可扩展) | CN + EN (双语，设计上可扩展但未训练) |
| **旋律提取** | RMVPE + SOME (student 模式) | SOME Teacher (预训练冻结 + fuzz) |
| **旋律保真** | 只有 flow loss (弱) | CKA 对齐 loss + GRPO (强) |
| **CFG 逻辑** | text 单独 CFG | 4-way CFG (text+cond+midi 一起 drop) |
| **跳跃连接** | 无 | 长跳跃连接 (long_skip_connection) |
| **训练策略** | 单阶段 SFT | 两阶段：SFT → GRPO |
| **输出格式** | 48kHz 单声道 | 44.1kHz 立体声 |
| **推理接口** | 纯代码，无 HF 集成 | PyTorchModelHubMixin，支持 from_pretrained |
| **Gradio WebUI** | 有 | 有 (app.py / app_local.py) |
| **批量推理** | 无 | inference_mp.py (多GPU多进程) |

## 七、V1 的问题在 Plus 中的解决情况

### ✅ 已解决：旋律跟随弱

V1 的旋律只是和 cond/text/melody input 拼在一起，没有针对性优化。Plus 通过：
1. **CKA 对齐损失** 强制中间特征与旋律参考一致
2. **GRPO 奖励模型** 用 F0 相关性直接作为奖励信号
3. 对抗权衡：ASR WER 和 F0-CORR 在奖励中分别占 25%

### ⚠️ 部分解决：跨角色翻唱

Plus 的设计允许 timbre_ref 和 melody_ref 是**不同的人**——这是 explicit design feature。但：
- 音色分离能力受限于训练数据的多样性
- V1 的 timbre → melody 串联问题在 Plus 中不存在（两者是独立输入）

### ❌ 未解决：日语支持

Plus 的 `CNENTokenizer` 和 `g2p_generation.chn_eng_g2p()` 硬编码为 **中英双语**。即使 `PhonemeBpeTokenizer` 注册了日语的 G2P 后端 (`ja` → `ja` backend)，但：
- `get_segment()` 只区分 `zh`、`en`、`other`，日语字符被归为 `other` 然后尝试用英文 phonemizer 处理
- 词表 (373 token) 是为中英 IPA 设计的，日语所需的音素可能不全
- 训练数据不包含日语，DiT 未学过日语音素→音频的映射

### ❌ 未解决：MIDI 精确控制

用户输入仍然是**音频作为旋律参考**。没有精确到音符的 MIDI 编辑能力。GitHub Issue #2 中有人请求 MIDI 条件 SVS，但当前尚未实现。

## 八、关于跨角色翻唱 — 关键洞察

Plus 的跨角色翻唱能力在架构上是**设计内的**：

```
timbre_ref (音色) ──→ VAE encode ──→ cond (条件，前半段)
melody_ref (旋律) ──→ VAE encode ──→ melody latent ──→ SOME ──→ midi
                                      └─ 拼接后用于 MIDI 提取

注意：timbre_ref 和 melody_ref 可以被指定为不同的音频！
```

但在推理时，`ref_audio` 同时承载了双重角色：
1. 作为 `cond`（timbre conditioning）—— 提供音色
2. 作为 melody latent 的前半段（和 melody_audio latent 拼接后送入 SOME）

这意味着：
- ref_audio 的前半段会被 SOME "读到"，作为旋律上下文的一部分
- 但 ref_audio 对应的歌词 (`ref_text`) 是额外给定的，不会被唱出来
- 生成的目标段 = total_len 中 ref_latent_len 之后的部分

## 九、部署环境记录

| 项目 | 值 |
|------|-----|
| 环境名 | `yingmusic_plus` |
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | 8× RTX 4090 |
| 服务器 | user@[SERVER_IP] |
| 代码位置 | `${REMOTE_ROOT}/YingMusic-Singer-Plus/` |
| 模型权重 | `${REMOTE_ROOT}/YingMusic-Singer-Plus/ckpts/` (~13GB) |
| 包数量 | 261 |
| 安装方式 | `uv pip install -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/simple` |
