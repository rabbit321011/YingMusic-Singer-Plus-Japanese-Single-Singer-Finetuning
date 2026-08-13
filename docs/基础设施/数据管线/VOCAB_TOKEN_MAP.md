# Token / Vocab / Embedding 对照表

> 2026-05-29 | 基于官方代码 diff + L2 静态分析 + AdamW optimizer state 交叉验证
>
> 本文档是 token ID 体系的**唯一权威来源**——所有训练脚本、文档中涉及 token ID 的地方必须与此保持一致。

---

## 一、架构总览

```
文本（日文） → pyopenjtalk → 音素序列 → vocab.json 查表 → token ID (+1 offset) → nn.Embedding → 连续向量
        ↑                        ↑                                            ↑
   g2p/japanese.py          vocab.json                                 YingMusic_Singer.yaml
   (从 Amphion 复制)        (363 个 IPA 符号)                         text_num_embeds: 373
                                                                      → Embedding(374, 512)
```

**三层的独立性：**

| 层 | 负责文件 | 我们的修改 |
|------|------|:--:|
| G2P（文本→音素） | `g2p/japanese.py`, `g2p/cleaners.py`, `g2p_generation.py` | ✅ 已改（加日文支持）|
| Tokenizer（音素→ID） | `cnen_tokenizer.py`, `vocab.json` | ❌ 未改（官方原版）|
| Embedding（ID→向量） | `dit.py` TextEmbedding | ❌ 未改（官方原版）|

---

## 二、文件对比验证（2026-05-29）

| 文件 | 官方（`docs/temp_0529/official/`） | 本地项目 | 是否一致 |
|------|------|------|:--:|
| `vocab.json` | `src/YingMusicSinger/utils/f5_tts/g2p/g2p/vocab.json` | `YingMusic-Singer-Plus-src/...` | ✅ 完全一致 |
| `cnen_tokenizer.py` | `src/YingMusicSinger/utils/cnen_tokenizer.py` | `YingMusic-Singer-Plus-src/...` | ✅ 完全一致 |
| `dit.py` | `src/YingMusicSinger/models/dit.py` | `YingMusic-Singer-Plus-src/...` | ✅ 完全一致 |
| `YingMusic_Singer.yaml` | `src/YingMusicSinger/config/YingMusic_Singer.yaml` | `YingMusic-Singer-Plus-src/...` | ✅ 完全一致 |

**官方代码来源**：`ASLP-lab/YingMusic-Singer-Plus`（GitHub），`--depth 1` clone 到 `docs/temp_0529/official/`，只读不编辑。

---

## 三、Tokenizer ID 体系

### 3.1 加载逻辑（`cnen_tokenizer.py`）

```python
# 第 12 行：vocab.json 的所有 ID +1
self.phone2id = {k: int(v) + 1 for (k, v) in self.phone2id.items()}

# 第 14-15 行：手动追加 PAD
self.pad_token_id = 0
self.phone2id["<PAD>"] = 0

# 第 17-18 行：追加 PUNCT
self.punct_token_id = len(self.phone2id)  # = 364
self.phone2id["<PUNCT>"] = len(self.phone2id)

# 第 20-21 行：追加 SEP
self.sep_token_id = len(self.phone2id)    # = 365
self.phone2id["<SEP>"] = len(self.phone2id)
```

### 3.2 encode() 逻辑

```python
def encode(self, text):
    phone, token = self.tokenizer(text)  # G2P → 原始 vocab.json ID（0~362）
    token = [x + 1 for x in token]       # +1 偏移匹配 phone2id
    return token
```

### 3.3 完整 ID 表

| Token | ID | 用途 | 代码使用路径 | 官方训练状态 |
|-------|:--:|------|------|:--:|
| `<PAD>` | **0** | CFG filler — `drop_text=True` 时所有文本替换为 0 | `dit.py` L136 `torch.zeros_like(text)` | ✅ 训练过（CFG filler）|
| **普通音素（363个）** | **1~363** | CN/EN/JP 音素 | G2P → encode → 查表 | 部分训练过（见 §4.3）|
| `<PUNCT>` | **364** | 无实际用途 | **无任何代码路径使用** | ❌ 从未训练 |
| `<SEP>` | **365** | 句间分隔 — 逐句编码后插入 | `lrc_align.py` `tokenizer.phone2id["<SEP>"]` | ✅ 训练过 |
| **总计** | **366** | — | — | — |

### 3.4 日语音素映射

我们改造的是 G2P（`g2p/japanese.py`），不是 vocab.json。vocab.json 是官方预置的——官方已经有了日语音素符号，**在 SFT 阶段没有日文训练数据，从未使用过**。

| 在 vocab.json 中的原始 ID | tokenizer ID (+1 后) | 符号示例 |
|:--:|:--:|------|
| 318 | 319 | `ɯ` |
| 319 | 320 | `e` |
| 320 | 321 | `aː` |
| ... | ... | ... |
| 345 | 346 | `tsɯ` |
| **共 28 个** | **319~346** | — |

### 3.5 推理侧一致性验证

`lrc_align.py`（官方与本地完全一致）使用动态查表：

```python
# lrc_align.py 第 17/45 行
one_line_token = one_line_token + [tokenizer.phone2id["<SEP>"]]
```

`tokenizer.phone2id["<SEP>"]` = **365**。训练的 `encode_text_with_sep` 修正后也使用 365——**三方一致**。

---

## 四、模型 Embedding

### 4.1 参数来源

| 参数 | 值 | 来源 |
|------|:--:|------|
| `text_num_embeds` | 373 | `YingMusic_Singer.yaml` → `datasets_cfg.text_num_embeds` |
| `text_dim` | 512 | `YingMusic_Singer.yaml` → `model.arch.text_dim` |
| 实际 embedding | `nn.Embedding(374, 512)` | `dit.py` 第 42-44 行：`text_num_embeds + 1` |

### 4.2 Embedding 行映射（以 optimizer state 为准）

checkpoint key: `model_state_dict.transformer.text_embed_p.text_embed.weight` → shape `[374, 512]`

optimizer state key: `optimizer_state_dict.state[4]`（AdamW exp_avg = `[374, 512]`）

```
Row   0 → Token   0 (<PAD>)      ✅ 训练过
Row   1 → Token   1 (音素首)     ✅ 训练过
 ...   → ...
Row 318 → Token 318 (CN 末)     ✅ 训练过
Row 319 → Token 319 (JA 首)     ❌ 未训练 —— V4 首次
 ...   → ...
Row 346 → Token 346 (JA 末)     ❌ 未训练 —— V4 首次
Row 347 → Token 347 (杂项音素)  ❌ 未训练（罕见音素）
 ...   → ...
Row 363 → Token 363 (标点音素)  ❌ 未训练（罕见音素）
Row 364 → Token 364 (<PUNCT>)   ❌ 从未训练（无代码路径使用）
Row 365 → Token 365 (<SEP>)     ✅ 训练过（lrc_align.py 插入）
Row 366 → —       (多余行)      ❌ 无对应 token
 ...   → ...
Row 373 → —       (多余行)      ❌ 无对应 token
```

### 4.3 实测数据两轮交叉验证

**环境**：服务器 `${SERVER_USER}@${SERVER_HOST}`，`yingmusic_plus` conda env，checkpoint 7.6GB

#### 4.3.1 第一轮：L2 范数静态分析

```text
Model shape: [374, 512]

各区域 L2 范数均值（纯随机期望 ~22.6）：
  PAD  [0]:      16.93
  EN   [5-74]:   17.20
  CN   [76-318]: 17.21
  JA   [319-346]:17.60
  PUNCT[364]:    17.13
  SEP  [365]:    17.40
  Extra[366-373]:17.40
```

**结论**：所有行 L2 ≈ 17.2，远低于随机期望 22.6。说明初始化不是标准 N(0,1)，而是某种缩放初始化。**仅凭 L2 无法区分训练与未训练行**。

#### 4.3.2 第二轮：AdamW optimizer state 分析

**⚠️ 方法说明**：AdamW 的 `exp_avg`（m）初始化为零。只有反向传播经过的行，其 m 才变为非零。因此 **m=0 的行从未收到梯度**。

**⚠️ 重要限制**：checkpoint 的 `update=600`，仅 600 步。其中含 `ref_cfm`（GRPO 阶段才有的 key），推断此 checkpoint 处于 GRPO 阶段（optimizer 从 GRPO 开始重新初始化）。600 步 vs 官方 SFT 的 31,518 步差距很大。因此 **SEP 在 GRPO 段训过 = 可以确认 SEP 被使用过**；**PUNCT 在 GRPO 段未训 = 全生命周期可能都未被使用，但 SFT 段无法直接证明**。

```text
Optimizer key: state[4]  (exp_avg shape: [374, 512])

Region breakdown by optimizer momentum:
  PAD     [0]:      m>0  → TRAINED
  EN      [1-74]:   60/74 rows trained
  CN      [75-318]: 160/244 rows trained
  JA      [319-346]:27/28 rows UNTRAINED (仅边界行 319 有微弱动量)
  Rare    [347-363]:16/17 rows UNTRAINED (仅行 363 有微弱动量)
  PUNCT   [364]:    UNTRAINED
  SEP     [365]:    TRAINED
  Extra   [366-373]:UNTRAINED
```

**哪些未训练的 CN/EN 行？** 它们是 vocab.json 中的罕见音素：

| tokenizer ID | vocab.json 符号 | 说明 |
|:--:|------|------|
| 67~74 | `ɑ̃`, `ç`, `ɔ̃` 等 | 法语/德语鼻化音，CN/EN 歌词中不出现 |
| 347~363 | `ɐ`, `ɑ`, `ɒ`, `y`, `ø`, `œ`, `ʁ` 等 | 欧洲语言杂项音素和标点符号 |
| 319~346 | `ɯ`, `e`, `aː` 等 | 日语音素，无日文训练数据 |

### 4.4 综合结论

| 区域 | 官方训练 | 对 V4 的影响 |
|------|:--:|------|
| PAD [0] | ✅ 训过 | 无影响——“自愈”分析不变 |
| EN 活跃 | ✅ 60/74 训过 | 无影响 |
| CN 活跃 | ✅ 160/244 训过 | 无影响 |
| JA [319-346] | ❌ 27/28 未训 | **V4 首次训练**（28行大部分随机init） |
| 罕见杂项 [347-363] | ❌ 16/17 未训 | 无所谓（训了也不会用） |
| PUNCT [364] | ❌ 未训 | 无所谓（代码无使用路径） |
| SEP [365] | ✅ 训过 | 无影响——推理侧也使用相同 token |
| Extra [366-373] | ❌ 未训 | 无对应 token，永不使用 |

---

## 五、fullplan 错误修正（已完成）

### 5.1 `plus-finetune-fullplan-v0527.md` §4.3.1

| 位置 | 修正前 | 修正后 |
|------|------|------|
| L149 代码注释 | `tokens.append(366)` | `tokens.append(365)` |
| SEP 实际 ID | — | **365** — 已确认与 `lrc_align.py` 一致 |

### 5.2 `plus-finetune-plan-v0527.md` §6.2.A

| 位置 | 修正前 | 修正后 |
|------|------|------|
| 代码 | `tokens.append(366)` | `tokens.append(365)` |

**增强建议**（写脚本时顺手改）：`tokens.append(365)` → `tokens.append(tokenizer.sep_token_id)`，与 `lrc_align.py` 的动态查表风格一致，防止未来 tokenizer 变化。

---

## 六、V4 训练注意事项

### 6.1 首次训练的 Embedding 行

V4 SFT 将首次训练以下行（共 ~28 行）：

| 行范围 | Token | 数量 | 初始状态 |
|------|------|:--:|------|
| 319~346 | JA 音素 | 28 行（边界行 319 除外） | 缩放随机初始化 |
| 364 | `<PUNCT>` | 1 行 | 缩放随机初始化（不会用到）|

**SEP [365] 已经训练过**——不是"首次"。fullplan 之前推断 SEP 未训练是基于 V3 过滤掉了 SEP，但 V3 是我们自己写的。**官方训练中 SEP 是被使用的。**

### 6.2 对 warmup 的影响

fullplan 设定的 500 步 warmup 设计得当——~28 行全新 embedding + 边界行从冷启动需要稳定学习率。JA 行数（28）与 fullplan 说的"27 行"差 1 行，不影响设计。

### 6.3 PAD [0] 和 CFG

PAD 在官方训练中已训过。V3 中文本信号被 x_t 泄漏压制 → `drop_text` 无差异 → Embedding[0] 收不到有效梯度。V4 文本条件生效后自动恢复——**自愈分析不变**。

---

## 七、关键文件速查

| 内容 | 路径（相对项目根目录） |
|------|------|
| vocab.json | `YingMusic-Singer-Plus-src/src/YingMusicSinger/utils/f5_tts/g2p/g2p/vocab.json` |
| cnen_tokenizer.py | `YingMusic-Singer-Plus-src/src/YingMusicSinger/utils/cnen_tokenizer.py` |
| dit.py（Embedding 定义） | `YingMusic-Singer-Plus-src/src/YingMusicSinger/models/dit.py` |
| YingMusic_Singer.yaml | `YingMusic-Singer-Plus-src/src/YingMusicSinger/config/YingMusic_Singer.yaml` |
| lrc_align.py（推理 SEP 插入） | `YingMusic-Singer-Plus-src/src/YingMusicSinger/utils/lrc_align.py` |
| 模型 checkpoint | `ckpts/YingMusicSinger_model.pt`（服务器，7.6GB）|
| Checkpoint L2 分析脚本 | `docs/temp_0529/check_embedding.py` |
| Optimizer state 分析脚本 | `docs/temp_0529/check_opt_state.py` |
| 官方只读参照 | `docs/temp_0529/official/` |
| g2p/japanese.py（我们改的） | `YingMusic-Singer-Plus-src/src/YingMusicSinger/utils/f5_tts/g2p/g2p/japanese.py` |
| g2p/cleaners.py（我们改的） | `YingMusic-Singer-Plus-src/src/YingMusicSinger/utils/f5_tts/g2p/g2p/cleaners.py` |
| g2p_generation.py（我们改的） | `YingMusic-Singer-Plus-src/src/YingMusicSinger/utils/f5_tts/g2p/g2p_generation.py` |

---

## 八、变更记录

| 日期 | 变更 |
|------|------|
| 2026-05-29 | 初稿：token ID 体系、L2 静态分析、SEP ID = 366 → 365 修正 |
| 2026-05-29 | v2：第二轮 AdamW optimizer state 交叉验证，修正训练状态判断（SEP/PAD 确认训过，PUNCT 确认未训），添加 lrc_align.py 一致性验证，标注 optimizer 分析的 step=600 限制 |

















