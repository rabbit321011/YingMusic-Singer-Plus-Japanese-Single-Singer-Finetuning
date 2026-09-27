> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# YingMusic-Singer-Plus 日语 SFT 微调方案 — V4 综合报告

> 致 YingMusic-Singer-Plus 作者：
>
> 我在官方权重基础上做了花丸晴琉（Hanamaru Hareru）日语歌唱微调，目标是实现**改词翻唱**（任意旋律 + 任意日文歌词 → 花丸音色日语歌声）。
>
> 经过 V1-V3 的失败和 V4 多轮迭代，目前自重建音频可辨识、改词翻唱已产生一定效果但仍有电音感和 FlowB loss 平台。
>
> 本文档完整记录了我的设计方案，**希望能得到您的专业意见**——特别是关于我的 A/B 区分割方案是否有被忽视的问题。

***

## 一、背景

### 1.1 目标

在 YingMusic-Singer-Plus 官方 CN/EN 权重基础上微调，支持**花丸晴琉音色 + 任意旋律 + 任意日文歌词**的歌唱合成。

### 1.2 模型规模

- DiT: 453.6M（22 层 F5-TTS DiT，dim=1024，depth=22，heads=16）
- VAE: Stable Audio 2（冻结）
- SOME MIDI Teacher: 117.6M（冻结）
- Text Embedding: `nn.Embedding(374, 512)`，对应 366 个 token
- 日文新增未训练 embedding：28 行（ID 319\~346），官方 SFT 阶段无日文数据

### 1.3 数据

- 训练集：15,122 条 / 90.7h，100% 花丸晴琉日语翻唱干声
- Whisper large-v3 自动标注 + 清洗
- Train/Test 隔离

### 1.4 硬件

- 4× RTX 4090 (24GB) DDP 训练

***

## 二、V1 → V3 历程（简）

| 版本     | 核心问题                                                                                                  | 结果 |
| ------ | ----------------------------------------------------------------------------------------------------- | -- |
| **V1** | `cond = full_latent = GT`：DiT 每帧都能从 cond 直接抄答案，30k 步从未学会文本→音频映射。自重建完美（RMS 0.08），改词翻唱完全失败              | ❌  |
| **V2** | G2P 改造未完成（日文路由错误），第 50 步崩溃                                                                            | ❌  |
| **V3** | 修复了 cond（前 50%→cond，尾部填零）但 `x_t = (1-t)×noise + t×full_latent` 仍每帧泄漏答案。训练 loss 正常下降（24.9→0.45），但推理全噪声 | ❌  |

V3 复盘的关键发现：模型在训练时从未被要求从纯噪声生成任何帧——x\_t 的每一帧都包含 t×full\_latent 的真实值。推理时 x\_t 初始化为纯噪声 → 分布外 → 崩溃。

***

## 三、V4 核心设计：A/B 区分割

### 3.1 原理

将 latent 序列分为两个区域，训练和推理采用**完全一致**的信息分布：

```
帧索引:  0 ──────── ref_len ───────────────── T
         ├──────────────────┼─────────────────────┤
         │       A区（参考区）  │      B区（生成区）     │
         │  ref_len = 12.5%~33% T                │
         │  ≥ 5s, ≤ 65% T                        │
         ├──────────────────┼─────────────────────┤
x_t      │ (1-t)×噪声+t×真值  │ (1-t)×噪声+t×真值      │ ← 统一标准 flow matching
cond     │ 真latent          │ 0                    │
midi     │ 0                 │ 真旋律                │
text     │ 真实token         │ 真实token             │
         ├──────────────────┼─────────────────────┤
训练目标   │ 有cond参考，重建简单  │ 仅凭 text+midi 生成    │
推理行为   │ 真latent直接提供    │ 从纯噪声生成 + text+midi │ ← 与训练B区完全一致
```

**关键**：B 区训练时 cond=0、推理时 cond=0 —— 分布完全对齐。模型必须在 B 区仅凭 text + midi + self-attention（attend 到 A 区提取音色）生成歌声。

### 3.2 ref\_len 随机化

```python
ref_len = T * random.uniform(0.125, 0.33)   # 12.5%~33% 随机
ref_len = max(ref_len, 5s_frames)             # ≥ 5s 下限
ref_len = min(ref_len, int(T * 0.65))         # ≤ 65% 上限
# + 短语边界吸附（A7 时间戳，±2.5s 范围内吸附到最近短语边界）
```

### 3.3 训练/推理对齐

| <br />  |     训练 A区     |     训练 B区     |    推理 A区   |    推理 B区   |
| ------- | :-----------: | :-----------: | :--------: | :--------: |
| x\_t 初值 | flow matching | flow matching | 纯噪声(t=0) ✅ | 纯噪声(t=0) ✅ |
| cond    |       真值      |     **0**     |    真值 ✅    |   **0** ✅  |
| midi    |     **0**     |      真旋律      |   **0** ✅  |    真旋律 ✅   |
| text    |     真token    |     真token    |  真token ✅  |  真token ✅  |

***

## 四、完整改动清单

### 4.1 G2P：添加日文支持（3 文件）

| 文件                  | 改动                                                     |
| ------------------- | ------------------------------------------------------ |
| `g2p/japanese.py`   | **新建** — 从 Amphion 复制 pyopenjtalk + pykakasi 后端（816 行） |
| `g2p/cleaners.py`   | `text_tokenizers["ja"]` → `None`，绕过 espeak-ng          |
| `g2p_generation.py` | 添加 `has_japanese()` 检测 → `chn_eng_g2p` 入口日语 bypass     |

pyopenjtalk 产出的 44 个 IPA 符号全部命中 vocab.json，其中 16 个与 CN/EN 共享（ID 1\~318），28 个日文独有（ID 319\~346）在官方权重中从未被训练。

### 4.2 训练：A/B 区分割 + 独立 loss 记录

```python
# process_batch (batch=1/GPU, grad_accum=4)
full_latent = vae.encode_audio(wav)  # [1, T, 64]
ref_len = compute_ref_len(T, phrase_boundaries)

# cond: A区真值 / B区零
cond = zeros(1, T, 64)
cond[:, :ref_len, :] = full_latent[:, :ref_len, :]

# midi: A区清零 / B区保留
midi = smoothMelody_MIDIFuzzDisturb(midi_p)
midi[:, :ref_len, :] = 0

# text: 带 SEP 的对齐
aligned_text = align_text_with_timestamps(phrases, timestamps, ref_len, T)

# --- compute_loss ---
# 全帧统一 flow matching（A区 x_t 也走噪声→重建，有 cond 直接参考）
x_t = (1-t) * noise + t * full_latent
v_target = full_latent - noise

v_pred, hidden_states = dit(x_t, cond, aligned_text, t, midi, drop_*)

L_flow_A = MSE(v_pred[:, :ref_len], v_target[:, :ref_len])
L_flow_B = MSE(v_pred[:, ref_len:], v_target[:, ref_len:])
L_cka = cka_loss(hidden_states_B, midi_p_B)   # CKA 仅 B 区计算
loss = L_flow_A + L_flow_B + cka_weight * L_cka
```

### 4.3 t\_shift 对齐

训练 t 使用与推理相同的变换（一行替换）：

```python
u = torch.rand(B)
t = 0.5 * u / (1 - 0.5 * u)   # t_shift=0.5
```

### 4.4 文本编码：SEP + A7 时间戳精确对齐

- 按空格分句 → 逐句 pyopenjtalk G2P → 句间插入 `<SEP>` (ID=365)
- L1/L2 样本：按 A7 Whisper 句级时间戳精确放置 token
- L3 样本：无可靠时间戳 → 均匀分布兜底
- 对标官方 `lrc_align.py` 的逐句 G2P + SEP 格式

### 4.5 推理

直接使用官方 `Singer.sample()`，输入构造：

```python
cond = full_latent[:, :ref_len, :]                     # A区真latent
text_tokens = align_lrc_sentence_level(lrc, ...)         # 带 SEP 的 flat token array
midi_p, bound_p = midi_teacher(mel)                     # SOME 原始输出（不经过 FuzzDisturb）
result = model.sample(cond=cond, text=text_tokens, ...)  # cond_mask 自动处理 A/B 区分割
```

<br />

***

## 五、数据分层策略

A7 流水线（faster-whisper large-v3 ASR → difflib fuzzy match 歌词 → sim 四档）产出的句级时间戳质量分层：

| 档位 |    sim 阈值   |  样本数  | 时间戳精度        | 训练策略        |
| -- | :---------: | :---: | ------------ | ----------- |
| L1 | ≥0.9, 短语≤10 | 7,604 | 高（MAE<1.84s） | 精确时间戳对齐     |
| L2 |     ≥0.7    | 3,546 | 中            | 精确对齐，允许个别偏差 |
| L3 |     ≥0.5    |  840  | 低            | 均匀分布兜底      |
| L4 |     <0.5    | 2,243 | 不可用          | 丢弃          |

***

## 六、训练历程与结果

### 6.1 四轮实验

| 版本       | 数据       |               LR              | 关键参数                          |   步数   | FlowB 最终 | 最佳推理效果       |
| -------- | -------- | :---------------------------: | ----------------------------- | :----: | :------: | ------------ |
| **V4**   | L1+L2+L3 |            7e-6 cos           | CKA=1.0, drop=30%             | 7k（中断） |     —    | NaN bug 中断   |
| **V4b**  | L1+L2+L3 |           1.4e-5 cos          | CKA=0.7, drop\_text=30%       |   19k  |   0.90   | 自重建糊但能听歌词    |
| **V4c**  | L1+L2    | 1.4e-5 warmup-hold(12k)-decay | CKA=0.7, drop\_text=15%, B区×2 |   30k  |   0.92   | 22k 歌词最佳，但是糊 |
| **V4ca** | L1 only  |            8e-7 cos           | CKA=0.2, drop\_text=15%       |  6k 目标 |     —    | 和v4c差不多      |

### 6.2 V4c 详细 loss 曲线

| Step  | LR      | FlowA | FlowB | CKA  |
| ----- | ------- | ----- | ----- | ---- |
| 6000  | 1.40e-5 | 0.41  | 0.95  | 0.18 |
| 12000 | 1.40e-5 | 0.38  | 0.95  | 0.17 |
| 20000 | 8.56e-6 | 0.36  | 0.92  | 0.18 |
| 24000 | 3.68e-6 | 0.30  | 0.90  | 0.15 |
| 30000 | 0.00    | 0.34  | 0.92  | 0.18 |

### 6.3 推理效果（V4c step\_024000）

- **自重建**：歌词可辨识（Whisper WER \~0.3-0.4），电音感，哑音感强
- **改词翻唱**：CFG=1 时旋律跟随良好，歌词有所变化但不完全精确；CFG=3 时歌词改变更明显但音质有退化
- **跨旋律翻唱**：旋律跟随有效，但音质进一步退化
- 2k→22k 持续改善，22k→30k 无明显改善

### 6.4 关键观察：FlowB 天花板

四轮实验所有参数组合下，FlowB 硬地板 = **\~0.90**：

| 改动                 |    FlowB 效应   |
| ------------------ | :-----------: |
| LR 7e-6→1.4e-5     |     -0.08     |
| cosine→warmup-hold |     加速，未突破    |
| CKA 1.0→0.7        |       微弱      |
| drop\_text 30%→15% |       微弱      |
| B区 loss×2          |       微弱      |
| L3 data 移除         |       微弱      |
| **所有改动合计**         | **0.99→0.90** |

对比：V1（cond=GT 作弊）Flow=0.45 → 清晰音频。V4c（无作弊）FlowB=0.90 → 有电音、糊。

**FlowB 平台 ≠ 没学习**：22k 歌词明显比 8k 好、比 2k 好很多——text→latent 映射在持续改善，只是 loss 数值上不反映。

***

## 七、希望得到作者意见的问题

<br />

### Q1：A/B 区分割方案是否有被忽视的问题？

我的 A/B 区分割设计（见第三章）是否与官方训练方式有根本性差异？官方训练时的 cond 处理方式是什么——是否也是前段真值 + 尾部填零？

### Q2：CKA loss 权重建议

我尝试了 CKA=1.0、0.7、0.2，FlowB 差异不明显。官方对 CKA loss 与 Flow loss 的平衡有什么建议？

### Q3：CFG 与 B 区生成质量

V4c 推理中 CFG=3 时歌词变化更明显但音质退化。这可能是模型对 text 条件依赖不够强的表现。官方是否有推荐的 CFG 提高策略（如训练中逐步增加 drop\_text）？

<br />

***

## 训练环境：4× RTX 4090 (24GB)，`yingmusic_plus` conda env，torch 2.6.0+cu124
