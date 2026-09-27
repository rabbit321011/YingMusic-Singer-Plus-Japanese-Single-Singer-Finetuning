> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# 日语歌唱合成微调：完整问题陈述

> 自包含文档——无需查阅任何外部材料即可理解全部背景、实验、数据和瓶颈。

***

## 一、我要做什么

在 **YingMusic-Singer-Plus**（arXiv:2603.24589，西北工业大学 ASLP + 巨人网络）的官方中英文权重基础上微调，实现：

> **花丸晴琉音色 + 任意旋律 + 任意日文歌词 → 44.1kHz 日语歌声**

核心应用是类似SVC的**翻唱**：输入一段音色参考 + 一段旋律参考+ 歌词 → 输出新歌声。不需要人工标注音素时长或 MIDI 钢琴卷。

我之所以用SVS是因为SVS能更好的还原咬字习惯

***

## 二、模型架构（完整细节）

### 2.1 四核引擎

| 组件               | 来源             |   参数量  |   微调时  | 输入                         | 输出                              |
| ---------------- | -------------- | :----: | :----: | -------------------------- | ------------------------------- |
| **DiT CFM**      | F5-TTS 架构      | 453.6M | 🔥 可训练 | 拼接后的 latent+cond+text+midi | velocity field v\_pred          |
| VAE              | Stable Audio 2 | 156.1M |  ❄️ 冻结 | 44.1kHz 立体声波形              | 21.53Hz / 64维 latent（下采样 2048×） |
| Melody Extractor | SOME 预训练       | 117.6M |  ❄️ 冻结 | 80维 Mel 频谱                 | 128维 MIDI 表示 + 音符边界             |
| Tokenizer        | CNENTokenizer  |    —   |    —   | 中/英/日文本                    | 366个 token 的 ID 序列              |

总参数 \~727.3M。VAE 和 Melody Extractor 完全冻结，仅训练 DiT。

### 2.2 DiT 架构（F5-TTS 骨干，22层）

```
输入嵌入（逐帧拼接）:
  Linear( [noised_x(64) | cond(64) | text_embed(512) | midi_proj(128)] , 1024 )
  四路信息在每一帧拼接后投影到 1024 维统一空间

每个 DiTBlock:
  time_emb = SiLU(Linear(t, 1024))                        // 时间步编码
  [γ_msa, β_msa, α_mlp, β_mlp, γ_mlp, _] = Linear(time_emb, 6144)  // 6组调制参数

  // AdaLN 调制的自注意力:
  x_norm = RMSNorm(x) * (1 + γ_msa) + β_msa
  attn_out = SelfAttention(x_norm) + RoPE                 // 旋转位置编码
  x = x + gate_msa * attn_out

  // AdaLN 调制的 FFN:
  x_norm = RMSNorm(x) * (1 + α_mlp) + β_mlp
  x = x + γ_mlp * FFN(x_norm)

长跳跃连接: DiT 输入直接加到 DiT 输出
```

**关键规格**：22层 × 16头注意力 × hidden=1024 × FFN膨胀=2。文本编码器前置4层 ConvNeXtV2 + 独立 512 维 embedding（不与 mel 共享维度）。RoPE 旋转位置编码。

### 2.3 Flow Matching

- 线性流: `x_t = (1-t) × noise + t × x₁`
- 目标 velocity: `v_target = x₁ - noise`
- Loss: `L_MSE = MSE(v_pred, v_target)`
- 训练时 t 经过 t\_shift 变换: `t = 0.5 × u / (1 - 0.5 × u)`，其中 u \~ U(0,1)
- 推理: 32 步 ODE + t\_shift=0.5

### 2.4 CKA 旋律对齐损失

从 DiT 的 3 层中间 hidden states 与旋律 GT 计算 Centered Kernel Alignment：

```
K = v_pred · v_pred^T          (DiT hidden states 的 Gram 矩阵)
L = melody · melody^T          (旋律特征 Gram 矩阵)

L_cka = 1 - ‖K^T L‖²_F / (‖K^T K‖_F · ‖L^T L‖_F)
```

总 SFT loss: `L_total = L_flow_A + L_flow_B + λ × L_cka`
CKA 只在 B 区计算（A 区 midi=零，CKA 无意义）。λ 初始 0.3，前 2k 步衰减至目标值（V4 用 0.7\~1.0）。

### 2.5 CFG 推理（4-way Classifier-Free Guidance）

```
四个前向路径:
  x_cond:          text✓  cond✓  midi✓      (全条件)
  x_uncond:        text✓  cond✗  midi✓      (仅丢 cond)
  x_uncond_cc:     text✗  cond✓  midi✗      (丢 text+midi)
  x_drop_all_cond: text✗  cond✗  midi✗      (丢全部)

最终速度场: v = v_cond + (v_cond - v_drop_all) × cfg_strength
```

默认 cfg\_strength=3。注意：旋律和文本/条件被**耦合在一起** CFG——增强 CFG 会同时增强文本清晰度和旋律跟随，无法独立调节。

***

## 三、日语 G2P 与 Token 体系

### 3.1 文本处理流水线

```
原始日文 "たぶん私じゃなくていいね 余裕のない二人だったし"
  ↓ 按空格分句
["たぶん私じゃなくていいね", "余裕のない二人だったし"]
  ↓ pyopenjtalk 逐句 G2P（汉字→假名→IPA 音素）
["t a b ɯ N ɰ a t a ɕ i ...", "j o j ɯ ː ..."]
  ↓ CNENTokenizer 查 vocab.json → token ID（+1 offset）
[318, 319, 320, ...],  [321, 322, ...]
  ↓ 句间插入 <SEP> (ID=365)
[318, 319, 320, ..., 365, 321, 322, ...]
  ↓ 按句级时间戳放置到 latent 帧序列
[0,0,...,318,319,...,SEP,...,321,...]
```

### 3.2 Token ID 表

| Token         |      ID      | 状态                                         |
| ------------- | :----------: | ------------------------------------------ |
| `<PAD>`       |       0      | ✅ 官方已训练（CFG filler——drop\_text 时所有文本替换为 0） |
| CN/EN 活跃音素    |    1\~318    | ✅ 官方已训练（如 n, i3, h, ao3, h, ə, l...）       |
| **JA 日语音素**   | **319\~346** | **❌ 27/28 从未训练。V4 首次从随机初始化开始学习**           |
| 罕见音素（欧语/标点符号） |   347\~363   | ❌ 从未训练（CN/EN/JP 都不使用）                      |
| `<PUNCT>`     |      364     | ❌ 从未训练（全代码库无使用路径）                          |
| `<SEP>`       |      365     | ✅ 官方已训练（lrc\_align.py 逐句插入）                |

- vocab.json 共 363 个 IPA 符号（原始 ID 0\~362），tokenizer 统一 +1 → 1\~363
- Embedding: `nn.Embedding(374, 512)`。config 写 `text_num_embeds=373`，代码+1 → `(374, 512)`。row 366-373 多余永不使用
- pyopenjtalk 产出的 44 个 IPA 符号中：16 个与 CN/EN 共享（已训练行），**28 个日文独有（ID 319-346，27/28 从未训练）**
- **推理侧验证**：训练和推理使用完全相同的 tokenizer + SEP 插入（lrc\_align.py 也用 `phone2id["<SEP>"]`=365），三方一致

### 3.3 关键约束

- 日文 embedding 28 行的初始值为某种缩放随机初始化（L2≈17.2），远低于标准 N(0,1) 的期望 22.6
- SFT warmup=500 步用于保护这 28 行从冷启动
- 该 Embedding 层接收梯度的唯一途径是 text 条件被 DiT 使用（通过 drop\_text=15-30% 的 CFG 训练提供无文本对比信号）

***

## 四、VAE（Stable Audio 2）

| 参数        |                  值                 |
| --------- | :--------------------------------: |
| 架构        |       AutoencoderOobleck，全卷积       |
| 总参数       |               156.1M               |
| 输入/输出     |             44.1kHz 立体声            |
| 压缩比       |                2048×               |
| 帧率        |         21.53Hz（每帧 46.5ms）         |
| latent 维度 |                 64                 |
| 下采样比率     |          \[2, 4, 4, 8, 8]          |
| 训练数据      | AudioSparx: 806,284 文件 / 19,500 小时 |
| 训练数据组成    |        66% 音乐、25% 音效、9% 乐器分轨       |
| **日语含量**  |             **几乎不含日语**             |

**论文自述**：*"speech generations are not intelligible"*；*"trained with English descriptions and will not perform as well in other languages"*。

**对日语的关键限制**：21.5Hz → 每帧 46.5ms。日语短辅音 VOT（Voice Onset Time）典型 20-50ms——与单帧时长相当。这意味着清浊对立（如 /t/ vs /d/）依赖的信息可能在单个 latent 帧内被压缩或丢失。

***

## 五、训练数据

### 5.1 花丸晴琉数据集

| 项目  | 值                   |
| --- | ------------------- |
| 训练集 | 15,122 条            |
| 测试集 | 307 条（按 BV 号隔离，零泄漏） |
| 总时长 | 90.7 小时             |
| 语言  | 100% 日语             |
| 歌手  | 花丸晴琉单人              |
| 来源  | YouTube 直播录屏/翻唱     |

### 5.2 数据预处理链路

```
原始录屏/翻唱
  → MSST 三模型管道（人声分离 → 去混响 → 降噪）
  → 静音切分（smart_cut: -40dB, 0.15s, 15-30s 段）
  → RMS + Crest 静音筛选
  → BYOL 声纹聚类过滤（DBSCAN eps=0.07，移除非花丸音色片段）
  → Whisper large-v3 歌词识别（6 GPU 并行）
  → 数据清洗：移除 Whisper 幻觉文本 1,063 条（"ご視聴ありがとうございました"等）
  → 最终 15,429 → train 15,122 / test 307
```

### 5.3 句级时间戳：A7 管道

训练需要将歌词 token 按句级时间戳放置到 VAE latent 帧序列上。A7 管道使用 faster-whisper large-v3 做 ASR → difflib 模糊匹配歌词 → 按 sim（文本相似度 0\~1）分四档：

| 档位          | 条件            |      条数     | 训练用法        |
| ----------- | ------------- | :---------: | ----------- |
| L1\_high    | sim≥0.9 且句≤10 | 7,725 (51%) | 精确时间戳对齐     |
| L2\_medium  | sim≥0.7       | 3,924 (26%) | 精确对齐，允许个别偏差 |
| L3\_low     | sim≥0.5       |  967 (6.4%) | 均匀分布兜底      |
| L4\_discard | sim<0.5       | 2,243 (15%) | 丢弃          |

P95 误差 6.62s（长尾由少数长句串烧拉高；成功样本中数十条 P95<2s）。

***

## 六、A/B 区分割训练方案 —— 核心设计

### 6.1 设计动机

官方训练通过随机掩码 70-100% VAE latent 帧——未掩蔽帧 = 音色上下文，掩蔽帧 = 生成目标。日语微调无法沿用此方案：官方权重中 28 行日文 embedding 从未训练，从随机掩码的零基础开始难度过高。需要一种更结构化的训练方式。

### 6.2 方案

将每条音频的 VAE latent 序列在随机位置 `ref_len` 处切分为两个区域：

```
帧索引:   0 ─────────────── ref_len ───────────────────── T
          ├─────────────────────────┼────────────────────────┤
          │         A区（参考区）      │       B区（生成区）       │
          │   ref_len = 12.5%~33% T  │                        │
          │   ≥5s (108帧), ≤65% T    │                        │
          ├─────────────────────────┼────────────────────────┤
x_t       │ (1-t)×noise + t×真latent │ (1-t)×noise + t×真latent│
cond      │ 真latent（提供音色）       │ 全零                    │
midi      │ 全零                    │ 真旋律（SOME提取+FuzzDisturb）│
text      │ 真实token（时间戳对齐）    │ 真实token（时间戳对齐）     │
          ├─────────────────────────┼────────────────────────┤
模型需学习  │ 重建（有cond直接参考）     │ 仅凭text+midi+自注意力生成  │
v_target  │ full_latent - noise     │ full_latent - noise     │
```

**四个核心约束**：

1. **A 区 cond=真 latent** → 通过 self-attention 向 B 区传递音色信息
2. **B 区 cond=零** → 训练和推理的 B 区 cond 完全一致（都是零）
3. **A 区 midi=零** → 防止模型依赖 A 区旋律特征
4. **B 区 midi=真旋律** → 提供音高引导信号
5. **全帧统一 flow matching**：A 区 x\_t 也走噪声→重建（不再特殊处理），因 A 区有 cond 直接参考，重建极其简单

### 6.3 ref\_len 计算

```python
ref_len = T × random(0.125, 0.33)    # 12.5%~33%
ref_len = max(ref_len, 108)          # ≥5s
ref_len = min(ref_len, 0.65 × T)     # ≤65% T
# 吸附到最近短语边界：±2.5s 范围内优先短语边界
```

典型音频中位时长 23.2s，中位 ref\_len 约 6-8s。

### 6.4 训练/推理对齐

| <br />  |     训练 A区     |     训练 B区     |    推理 A区    |    推理 B区    |
| ------- | :-----------: | :-----------: | :---------: | :---------: |
| x\_t 初值 | flow matching | flow matching | 纯噪声 (t=0) ✅ | 纯噪声 (t=0) ✅ |
| cond    |    真latent    |     **零**     |  真latent ✅  |   **零** ✅   |
| midi    |     **零**     |      真旋律      |   **零** ✅   |    真旋律 ✅    |
| text    |     真token    |     真token    |   真token ✅  |   真token ✅  |

**对齐是 A/B 分割的核心优势**：B 区训练时 cond=零、推理时 cond=零——分布完全一致，消除了训练/推理不匹配。

### 6.5 训练超参数

| 参数              |                              值                             |
| --------------- | :--------------------------------------------------------: |
| 硬件              |                   4× RTX 4090 (24GB) DDP                   |
| batch\_size     |        1/GPU (T 不统一，无法 batch) + grad\_accum=4（等效 16）       |
| optimizer       |                AdamW β=(0.9, 0.95), wd=1e-2                |
| LR              |         7e-6 \~ 1.4e-5（Cosine / warmup-hold-decay）         |
| warmup          |                            500 步                           |
| max\_grad\_norm |                             1.0                            |
| precision       |                            fp32                            |
| max\_duration   |                      30s（覆盖 99.8% 数据）                      |
| t\_shift        |                             0.5                            |
| EMA             |              β=0.995, update\_after\_step=100              |
| CFG dropout     | drop\_audio=30%, drop\_text=15%\~30%, drop\_midi=30%（三者独立） |

### 6.6 compute\_loss 核心逻辑

```
1. VAE encode wav → full_latent [1, T, 64]
2. 计算 ref_len（含短语边界吸附）
3. cond[0:ref_len]=full_latent[0:ref_len], cond[ref_len:]=0
4. midi = SOME(mel) → FuzzDisturb → midi[0:ref_len]=0
5. text = 逐句 G2P + SEP + 时间戳对齐放置
6. 采样 t = t_shift(u), noise ~ N(0,I)
7. x_t = (1-t)×noise + t×full_latent
8. v_target = full_latent - noise
9. v_pred, hidden_states = DiT(x_t, cond, text, t, midi)
10. L_flow_A = MSE(v_pred[:,:ref_len], v_target[:,:ref_len])
11. L_flow_B = MSE(v_pred[:,ref_len:], v_target[:,ref_len:])
12. L_cka = CKA(hidden_states_B, midi_B)
13. loss = L_flow_A + L_flow_B + λ × L_cka
```

***

## 七、SFT 实验完整记录

### 7.1 四轮训练

| 版本      |   步数   | 数据       | LR 策略                         | CKA λ | drop\_text | FlowB 终点 | 备注                   |
| ------- | :----: | -------- | ----------------------------- | :---: | :--------: | :------: | -------------------- |
| **V4**  | 7k(中断) | L1+L2+L3 | 7e-6 cosine                   |  1.0  |     30%    |     —    | NaN bug 中断           |
| **V4b** |   19k  | L1+L2+L3 | 1.4e-5 cosine                 |  0.7  |     30%    |   0.90   | 自重建糊但能听歌词            |
| **V4c** |   30k  | L1+L2    | 1.4e-5 warmup-hold(12k)-decay |  0.7  |     15%    |   0.92   | **主力模型（step 24000）** |
| V4ca    |   6k   | L1 only  | 8e-7 cosine                   |  0.2  |     15%    |     —    | 和 V4c 差不多            |

### 7.2 V4c 完整 Loss 曲线

| Step      |    LR   | FlowA |   FlowB  |  CKA |
| --------- | :-----: | :---: | :------: | :--: |
| 6000      | 1.40e-5 |  0.41 |   0.95   | 0.18 |
| 12000     | 1.40e-5 |  0.38 |   0.95   | 0.17 |
| 20000     | 8.56e-6 |  0.36 |   0.92   | 0.18 |
| **24000** | 3.68e-6 |  0.30 | **0.90** | 0.15 |
| 30000     |   0.00  |  0.34 |   0.92   | 0.18 |

**24000 是当前最佳 checkpoint**——此后 FlowB 回升到 0.92 且不再改善。

### 7.3 FlowB 天花板：0.90

所有参数组合下 FlowB 最低只能到 **0.90**：

| 尝试的改动                                  |         FlowB 效应         |
| -------------------------------------- | :----------------------: |
| LR 7e-6 → 1.4e-5                       |           -0.08          |
| Cosine → warmup-hold-decay LR schedule |        加速收敛，未突破下限        |
| CKA λ 1.0 → 0.7 → 0.2                  |            微弱            |
| drop\_text 30% → 15%                   |            微弱            |
| B区 loss ×2 加权                          |            微弱            |
| 移除 L3 低质量数据（15,122→\~11,600）           |         与用全量无显著差异        |
| **全部改动累计**                             | **0.99 → 0.90（仅 -0.09）** |

作为对照：**FlowA（A区，有cond）从 0.41 降到 0.30**，证明 DiT 本身学习能力正常。B 区被单独卡住。

**FlowB 平台不等于没学习**：step 22000 的改词可懂度明显优于 step 8000，远优于 step 2000。text→latent 映射在持续改善，只是 loss 数值不反映。但 22000→30000 无进一步改善。

### 7.4 推理效果（V4c step 24000）

- **自重建**（ref=melody=同一段）：歌词可辨识（Whisper WER \~0.3-0.4），但有明显电音感和哑音感
- **改词翻唱**（改歌词不改旋律）：CFG=1 时旋律跟随良好，歌词有变化但不完全精确；CFG=3 时歌词更清晰但音质退化
- **跨旋律翻唱**（换旋律参考）：旋律跟随有效，音质进一步退步

***

## 八、GRPO 强化学习实验完整记录

### 8.1 算法原理

**Flow-GRPO**（NeurIPS 2025, arXiv:2505.05470）将 Flow Matching 的确定性 ODE 转为等价 SDE 引入探索随机性，再用 Group Relative Policy Optimization 做策略梯度更新。

**ODE → SDE 转换**：
仅在有限去噪窗口 `[w_min=1, w_s=8)` 内注入 Wiener 噪声 σ\_t = 0.8 × sqrt(t/(1-t))，其余步走确定性 ODE。该 SDE 在所有时间步的边缘分布与原 ODE 完全等价——引入随机性不破坏生成质量。

**MDP 建模**（多步去噪决策过程）：

- 状态 s\_t = (condition c, 时间步 t, 当前 noisy latent x\_t)
- 动作 a\_t = 去噪一步后的 latent x\_{t-1}
- 策略 π\_θ = DiT 的条件分布 p\_θ(x\_{t-1} | x\_t, c)
- 转移：确定性，下一状态 = (c, t-1, 去噪结果)
- 奖励：仅终端步 t=0 非零：R = r(x\_0, c)

**四阶段训练循环**：

```
阶段1：采样（每 prompt G=8 个候选）
  SDE 窗口内注入随机噪声 → 10步去噪（Denoising Reduction）
  → VAE decode → 8 个音频

阶段2：Reward 打分（4 个 reward model）
  Whisper Large V3 → PER      (音素错误率，越低越好)
  WavLM-large       → SIM     (音色相似度，越高越好)
  torchcrepe        → F0-CORR (旋律保真度，越高越好)
  DNSMOS P.808      → 音频干净度

阶段3：组内标准化
  A_i = Σ_k w_k × (R^i_k - mean({R^j_k}_{j=1}^G)) / std({R^j_k}_{j=1}^G)
  不需要 Value Network——组内均值即 baseline

阶段4：策略更新
  L_GRPO = (1/G) Σ_i (1/|S|) Σ_{t∈S} ( -L_clip + β × D_KL )
  L_clip = min( ratio × A_i, clip(ratio, 0.998, 1.01) × A_i )
  极窄 clip（ε_l=0.002, ε_u=0.01）保护 SFT 基座
  β_KL = 1.0
```

### 8.2 GRPO 训练配置

| 参数                 |                            值                           |
| ------------------ | :----------------------------------------------------: |
| G（组大小）             |                            8                           |
| batch（每步 prompt 数） |                            6                           |
| 训练去噪步数             |                 10（Denoising Reduction）                |
| 推理去噪步数             |                           32                           |
| SDE 窗口             |                 \[1, 9)（步 1 到步 8 注入噪声）                 |
| SDE 噪声 α           |                           0.8                          |
| LR                 |                       7e-6 → 3e-6                      |
| KL 正则 β            |                           1.0                          |
| PPO clip           |                  ε\_l=0.002, ε\_u=0.01                 |
| CFG                |                    训练时禁用，推理时 scale=3                   |
| 总步数                |                          4,800                         |
| GPU                | GPU 0-2 训练 + **GPU 3 独立 Whisper TCP server**（终生不动，零碎片） |
| 速度                 |                       \~43s/step                       |

### 8.3 Reward 模型选型

| Reward  | 模型                       |  权重 | 衡量    | 选型依据                                              |
| ------- | ------------------------ | :-: | ----- | ------------------------------------------------- |
| PER     | **Whisper Large V3**     | 60% | 音素可懂度 | CER=0.161，比 Qwen3-ASR-1.7B（0.279）好 42%。日语歌声转录能力最优 |
| DNSMOS  | DNSMOS P.808 (ONNX, CPU) | 20% | 音频干净度 | 纯测音质无风格偏见，方向正确 9/10。分数范围 2.5-3.4，窄但稳定             |
| SIM     | WavLM-large              | 10% | 音色相似度 | cos\_sim 0.88-0.94，已搭载的验证模型                       |
| F0-CORR | torchcrepe               | 10% | 旋律保真度 | Pearson 0.94-0.98，仅 voiced 帧计算                    |

淘汰的 reward 模型：**VocalVerse2**（评估的是演唱艺术表现力而非音频质量，存在严重风格偏见——帅气唱法被惩罚、甜美唱法受奖励）和 **Qwen3-ASR-1.7B**（日语唱歌 CER 太高）。

### 8.4 评估体系

- **测试集**：120 首歌曲（seed=42 随机抽样），按 B 区假名数分 4 桶（1-10 / 11-20 / 21-30 / 30+），每桶 30 首
- **每首**：5 次推理（seed 42-46），PER 取平均
- **输出**：桶均值、方差、中位数、p30、p70、PER≥0.9 比例

### 8.5 SFT baseline 评估（V4c 24000）

|  桶（B区假名数） |  avg PER  |  中位数  |  p30  |  p70  | PER≥0.9比例 |  稳定性  |
| :-------: | :-------: | :---: | :---: | :---: | :-------: | :---: |
|    1-10   |   0.416   | 0.240 | 0.000 | 0.867 |   26.7%   | 76.7% |
|   11-20   |   0.429   | 0.442 | 0.289 | 0.561 |    6.7%   | 56.7% |
|   21-30   | **0.562** | 0.629 | 0.401 | 0.822 |   10.0%   | 80.0% |
|  **30+**  | **0.141** | 0.044 | 0.015 | 0.091 |     0%    | 96.7% |
| **Total** | **0.387** |   —   |   —   |   —   |     —     | 0.079 |

**两个极端现象**：

- **21-30 字符桶 PER=0.562**：不是越短越好。中位数的 PER=0.629 说明多数样本咬字较好，但 30% 的样本 PER>0.822——存在一段硬失败
- **30+ 字符桶 PER=0.141**：模型基本放弃 text 条件。中位数 0.044、p70=0.091——大多数样本稳定输出一个「安全的」歌声，几乎不跟随 text。**长文本时 text conditioning 几乎完全失效**

### 8.6 GRPO vs SFT 完整对比

| 步数           |       1-10      |  11-20 |  21-30 |   30+  | **Total PER** | Δ vs SFT(0.387) |    趋势    |
| ------------ | :-------------: | :----: | :----: | :----: | :-----------: | :-------------: | :------: |
| SFT 24K      |      0.416      |  0.429 |  0.562 |  0.141 |   **0.387**   |        —        | baseline |
| GRPO 400     |      0.427      |  0.424 |  0.570 |  0.145 |     0.391     |      +1.1%      |     ↑    |
| **GRPO 800** |      0.445      |  0.448 |  0.576 |  0.148 |   **0.404**   |    **+4.4%**    |  **峰值**  |
| GRPO 1200    |        —        |    —   |    —   |    —   |     0.392     |      +1.4%      |     ↓    |
| GRPO 4800    | 回落至 baseline 附近 | <br /> | <br /> | <br /> |       —       |       \~0%      |    ↓↓    |

- **最佳 800 步仅 +4.4%**（提升 \~1.1σ），处于统计噪声边缘——非显著性改善
- 1200 步已回落，4800 步完全回到 baseline 附近
- 短歌（1-10）受益最大（+7%），但整体不足以改变模型行为
- **GRPO 路线已暂停**

### 8.7 GRPO 失败根因推断

V4c SFT 阶段 text dropout 仅 **15%**。这意味着模型在 85% 的训练步中 text 条件都是有效的——模型不需要非常强地依赖 text。GRPO 的 reward 信号（尤其 PER 占 60%）试图强化 text conditioning，但 text conditioning 在 SFT 阶段已经形成了弱依赖的「习惯」——reward 信号无法扭转一个已经固化的行为模式。

如果 SFT 阶段 text dropout 更高（如 V4b 的 30%），模型可能对 text 更敏感，GRPO 效果可能不同。但 V4b 的整体质量不如 V4c。

***

## 九、还尝试了什么（均失败）

### 9.1 V5a1 课程训练（复刻论文 Phase1）

| 项目   | 详情                                                                             |
| ---- | ------------------------------------------------------------------------------ |
| 起点   | V4c step 24000                                                                 |
| 改动   | midi 全零、CKA 关闭、text dropout 关闭（纯 text-only SFT）                                |
| 步数   | 额外 24,000 步                                                                    |
| 训练结果 | FlowA=0.26, FlowB=0.97（\~1.13s/step）                                           |
| 评估   | 15 首双测试集 PER 对比：无显著提升（V4c=0.405 vs V5a1=0.334 测试集1; 0.272 vs 0.299 测试集2，不跨集复现） |
| 人耳   | 5 首 A/B 对比：咬字拉不开差距                                                             |
| 结论   | 放弃，回退 V4c 24K                                                                  |

### 9.2 其他参数扫描

| 尝试                                          |           效果           |
| ------------------------------------------- | :--------------------: |
| CKA λ 1.0 → 0.7 → 0.2                       | FlowB 微弱差异，说明 CKA 不是瓶颈 |
| B区 loss ×2 加权                               |           无效           |
| L3 低质量数据移除                                  |        与全量无显著差异        |
| Cosine / warmup-hold-decay / 固定 LR schedule |      影响收敛速度，不影响终点      |
| V4ca: 仅 L1 精调 + 极低 LR 8e-7                  |        和 V4c 差不多       |

***

## 十、核心瓶颈与请求帮助的问题

### 瓶颈总结

**FlowB 卡在 0.90**，所有参数调优（LR、LR schedule、CKA λ、drop\_text、数据量、B区 loss 加权）累计只从 0.99 降到 0.90。GRPO 仅 +4.4% PER，统计不显著。30+ 字符桶 SFT 基本失效（PER=0.141），说明长文本 text conditioning 尤其弱。

### 四个核心问题

**Q1：A/B 分割方案是否与官方随机掩码训练存在根本性不兼容？**

官方训练通过随机掩码 70-100% latent 帧来做生成——掩码的起点和长度每次随机，Cond 区域和生成区域是交织的。我的 A/B 分割是固定结构的前段 cond + 后段生成。

DiT 的自注意力是全局的——B 区的每一帧都能 attend 到 A 区的所有帧。理论上这应该让模型学会从 A 区提取音色、在 B 区生成。但 30+ 字符桶失效的现象暗示：当 B 区过长时，来自 A 区的音色信息通过 self-attention 的传递距离过远，导致 text 信号被稀释。

**这是否意味着 A/B 分割本质上有「最大有效 B 区长度」限制？** 如果 B 区超过某个长度后 self-attention 无法有效传递 A 区的音色信息，模型会不会掉回一个「安全模式」（输出一个平均歌声，忽略 text）？

**Q2：text conditioning 太弱——如何强化？**

证据链：

- SFT drop\_text=15% → text 在 85% 训练步中有效 → 模型未形成强 text 依赖
- GRPO 仅 +4.4%（\~1.1σ）→ reward 信号无法扭转已经形成的弱 text conditioning
- 30+ 字符桶 PER=0.141 → 模型在长文本时「放弃」text

如果把 text dropout 从 15% 提到 **40-50%**（缩减到仅 50% 步有 text），是否可能：

- 强化 text 条件（模型必须更依赖 text 才能降低 loss）？
- 副作用是什么？会不会导致 text 条件被完全忽略？

或者是否需要更结构化的 text 强化方案——如额外的 text→latent 的辅助 loss、对比学习正则化、或者分阶段逐步提高 text dropout 的课程策略？

**Q3：FlowB 0.90 的天花板背后是什么？**

三个候选根因：

1. **日语数据量不足**（仅 90h）：官方 CN/EN 使用 33,562h 歌唱数据做 SFT。90h 日语只有其 0.27%，且 28 行日文 embedding 从随机值冷启动
2. **VAE 对日语编码能力不足**：SA2 VAE 在 19,500h AudioSparx（66% 音乐，几乎不含日语）上训练。21.5Hz 帧率（46.5ms/帧）对日语短辅音 VOT（20-50ms）在临界点。论文自述 "speech generations are not intelligible"
3. **Flow Matching 在歌声多模态分布上的根本局限**：MSE loss 的最优预测永远是条件均值 E\[v|x\_t, c]。歌声中同一音素+同一音高有无数种合法的表达方式（不同唱法、颤音、滑音），条件均值指向「没有真实轨迹使用的方向」——导致生成模糊。这可能是 FlowB=0.90 的数学原因，增加数据/参数/调参都不改变

**这三个因素各自贡献了多少？如何判断？**

**Q4：如果当前路线无法突破，应该换什么？**

在 A/B 分割 + GRPO + 课程训练均已失败的现状下：

- 是否应该放弃增量训练，改用**混合数据从零/大幅重置 DiT 权重**？（加入朗读数据建立发音基础——日语朗读数据如 Emilia 2,800h 可获得）
- A/B 分割是否应该被**随机掩码**替代？（如果原因确实是 A/B 分割的固定结构与官方随机掩码范式不兼容）
- 是否有比 A/B 分割更适合日语增量训练的范式？（如分阶段渐进式——先长 cond 短生成 → 逐渐缩短 cond 延长生成）
- VAE decoder 微调（冻结 encoder，仅用日语数据微调 decoder）是否必要？（latent 不变 → DiT 不需重训，但有社区先例验证可行）
