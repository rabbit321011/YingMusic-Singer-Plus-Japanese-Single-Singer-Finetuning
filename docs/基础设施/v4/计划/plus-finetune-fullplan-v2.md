# YingMusic-Singer-Plus 日语微调 V2 — 完整执行计划

> 基于 V1 方案的失败复盘与逐模块讨论结论。只包含已确定的改动与步骤。

***

## 一、背景与目标

### 1.1 目标

训练 YingMusic-Singer-Plus 的 DiT 模型，使其具备target singer（target singer）的日语歌唱合成能力。完成后应支持 **target singer + 任意旋律 + 任意日文歌词** = 高质量日语歌声。

### 1.2 当前状态

| 项目                                                    |               状态               |
| ----------------------------------------------------- | :----------------------------: |
| Plus 官方 CN/EN 推理                                      |              ✅ 正常              |
| Plus JA SFT V1 (`train_plus.py`, 30k步)                | ❌ 训练有根本缺陷（cond=GT），rewrite从未成功 |
| Plus JA SFT V2 (`train_plus_v2.py`, CNENTokenizer 训练) |      ❌ 第50步崩溃（日文 G2P 路由错误）     |
| `ckpts/plus_ja_sft/` (93GB)                           |          🗑️ 无效权重，可删除          |
| `ckpts/plus_ja_sft_v2/`                               |           🗑️ 空目录，可删除          |

### 1.3 V1 方案的根本缺陷

`train_plus.py` 将完整音频的 VAE latent 同时用作 cond（音色条件）和 x₁（训练目标 GT），导致 DiT 每帧都直接从 cond 抄答案，从未学会文本→音频的映射。自重建完美（RMS 0.08）但改词翻唱完全失败。


***

## 二、需修改的内容（共 3 项）

| # | 模块            | 改动                             |  严重程度 |
| - | ------------- | ------------------------------ | :---: |
| 1 | CNENTokenizer | 添加 pyopenjtalk 日文 G2P 支持（4 文件） | 🔴 必须 |
| 2 | cond          | 前 50%→cond，整段→GT，target 区域填零   | 🔴 致命 |
| 3 | midi          | cond 区域清零                      | 🟡 建议 |

**CFG dropout 设计无需修改**（独立 null 学习 + Linear 可加性已验证）。**纯日文训练不需混 CN/EN**（用户目标为日语专用）。

***

## 三、服务器资源

| 项目     | 详情                                                   |
| ------ | ---------------------------------------------------- |
| 服务器    | `${SERVER_USER}@${SERVER_HOST}`                              |
| GPU    | 8× RTX 4090 (24GB)                                   |
| 可用 GPU | GPU set（sftv2 tmux session 磁盘     | 7.0TB 总，已用 4.8TB，可用 1.8TB                            |
| 环境     | `yingmusic_plus` (python 3.10.20, torch 2.6.0+cu124) |
| 项目路径   | `${SERVER_ROOT}/YingMusic-Singer-Plus/`                |

### 3.1 清理任务

```bash
# 释放 GPU set
tmux session 删除无效 V1 checkpoint（93GB）
rm -rf ${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft/
rm -rf ${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v2/
rm -rf ${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_ft/
rm -rf ${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_grpo/
```

***

## 四、数据集

### 4.1 训练/验证集

| 集     | 文件                   |   条数   | 时长    | 路径（服务器）                         |
| ----- | -------------------- | :----: | ----- | ------------------------------- |
| Train | `train_singnet.json` | 15,122 | 90.7h | `${SERVER_ROOT}/final_sum_large/` |
| Test  | `test_singnet.json`  |   307  | \~2h  | `${SERVER_ROOT}/final_sum_large/` |

- 100% 日语数据，无需混合 CN/EN
- Train/Test 按 BV 号隔离，同一个 BV 的所有 segment 全在 Train 或全在 Test → 零泄漏
- 数据格式: `[{Path, Duration, Text, Language}]`，`Language: "ja"`

### 4.2 音频文件

```
${SERVER_ROOT}/final_sum_large/audio/
    ├── sample_media.wav (干花)_2_seg000.wav
    ├── ...
    └── (15,429 个 wav，约 30GB)
```

### 4.3 Whisper 标注质量

Whisper large-v3 自动标注，已清洗：

- 删除了 Top3 幻觉文本 1,063 条（"ご視聴ありがとうございました"等）
- 残留少量生僻字标注（pyopenjtalk 遇到不认识的字输出 "Cannot read" 并跳过，不影响训练）
- 歌词准确率未经人工校对，但 90.7h 的数据量可通过统计覆盖噪声

***

## 五、模型权重

| 文件                                        | 大小     | 路径（服务器）  |     训练时     |
| ----------------------------------------- | ------ | -------- | :---------: |
| `YingMusicSinger_model.pt`                | 7.6 GB | `ckpts/` | 🔥 加载 + 可训练 |
| `stable_audio_2_0_vae_20hz_official.ckpt` | 596 MB | `ckpts/` |    ❄️ 冻结    |
| `model_ckpt_steps_100000_simplified.ckpt` | 449 MB | `ckpts/` |    ❄️ 冻结    |
| `MelBandRoformer.ckpt`                    | 871 MB | `ckpts/` |    不用于训练    |

`YingMusicSinger_model.pt` 包含 `ema_model_state_dict`——使用 EMA 权重初始化，与推理一致。

***

## 六、改动清单

### 6.1 CNENTokenizer — 日文 G2P 支持（4 文件）

**需要改动的文件：**

| # | 文件                                                       | 改动                                                                                   |
| - | -------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 1 | `src/YingMusicSinger/utils/f5_tts/g2p/g2p/japanese.py`   | **新建** — 从 `Amphion_official/models/tts/maskgct/g2p/g2p/japanese.py` (816行) 复制       |
| 2 | `src/YingMusicSinger/utils/f5_tts/g2p/g2p/cleaners.py`   | L1 添加 `from ...japanese import japanese_to_ipa`；L17 `text_tokenizers["ja"]` → `None` |
| 3 | `src/YingMusicSinger/utils/f5_tts/g2p/g2p_generation.py` | 添加 `has_japanese()` 检测 → `chn_eng_g2p` 入口处整段 bypass                                  |
| 4 | `src/YingMusicSinger/utils/cnen_tokenizer.py`            | 无需改动                                                                                 |

**`has_japanese()`** **检测逻辑：**

```python
def has_japanese(text):
    for ch in text:
        if '\u3040' <= ch <= '\u30ff':  # 平假名或片假名
            return True
    return False
```

**`chn_eng_g2p`** **入口改动：**

```python
def chn_eng_g2p(text: str):
    if has_japanese(text):
        from ...g2p.japanese import japanese_to_ipa
        pbt = PhonemeBpeTokenizer()
        phoneme = japanese_to_ipa(text, None)
        tokens = pbt.phoneme2token(phoneme)
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
            tokens = tokens[0]
        return phoneme, tokens
    # 原有中英逻辑
    segments = get_segment(text)
    ...
```

**验证：** 在服务器上运行一致性测试脚本，确认 5 个日文样本的 `japanese_to_ipa → pbt.phoneme2token` 和 `CNENTokenizer.encode` 产出相同 token 序列（去掉 +1 偏移后）。

### 6.2 cond — 前 50% → cond，尾部填零

**改动位置：** `train_plus_v3.py` 的 `process_batch` 函数

```python
# --- 当前 (train_plus.py) ---
latent = vae.encode_audio(w)   # [T, 64]
return latent, midi, midi_p, aligned_text, B, T, D
# latent 同时用作 cond 和 x₁

# --- 修复后 (train_plus_v3.py) ---
full_latent = vae.encode_audio(w).transpose(1, 2)  # [T, 64]

# cond: 前 50% 帧作为音色参考，尾部填零
ref_len = full_latent.shape[0] // 2
cond = torch.zeros_like(full_latent)               # [T, 64]
cond[:ref_len] = full_latent[:ref_len]              # 前 ref_len 帧有值

return full_latent, cond, midi, midi_p, aligned_text, ref_len, B, T, D
#      ↑ x₁=GT    ↑ cond     ↑ midi     ↑ text      ↑ 用于 midi 清零
```

`compute_loss` 中调用：

```python
v_pred, hidden_states = run_dit(x_t, cond, aligned_text, t, midi,
                                drop_audio, drop_text, drop_midi)
# 现在 cond ≠ x₁，target 区域 cond=0
```

### 6.3 midi — cond 区域清零

**改动位置：** `train_plus_v3.py` 的 `process_batch` 函数

```python
midi_p, bound_p = midi_teacher(mel_tensor.transpose(1, 2))
# interpolate to T if needed
midi = raw_model.smoothMelody_MIDIFuzzDisturb(midi_p)  # 先用 FuzzDisturb
midi[:ref_len, :] = 0   # cond 区域清零，与推理一致
```

### 6.4 train\_plus\_v3.py 完整要点

基于 `train_plus.py`，改动如下：

| 项                  | 原值                   | 新值                                     |
| ------------------ | -------------------- | -------------------------------------- |
| CNENTokenizer 日文支持 | ❌                    | ✅ （4 文件改动）                             |
| cond 来源            | 完整 latent            | 前 50% 帧 + 尾部填零                         |
| cond 传递            | x₁ 和 cond 是同一 tensor | 两个独立 tensor                            |
| midi cond 区域       | 不清零                  | 显式清零                                   |
| drop\_audio        | 20%                  | **30%（对齐官方默认值）** |
| max\_duration      | 15秒                  | 30秒（覆盖 99.8% 数据，OOM 则降 batch_size） |
| train tokenize     | encode\_ja()         | CNENTokenizer.encode()（统一的 1-based 路径） |
| eval tokenize      | N/A                  | CNENTokenizer.encode()（与训练一致）          |

### 6.5 文件结构

```
服务器:
${SERVER_ROOT}/YingMusic-Singer-Plus/
├── train_plus_v3.py          # 新训练脚本
├── run_sft_v3.sh             # 启动脚本
├── src/                      # 含 G2P 日文支持的改造
├── ckpts/
│   └── plus_ja_sft_v3/       # 新增：V3 SFT checkpoint
└── test_ja_g2p.py            # G2P 一致性测试
```

***

## 七、训练配置

### 7.1 超参数

| 参数                      | 值                            | 来源                      |
| ----------------------- | ---------------------------- | ----------------------- |
| learning\_rate          | 7e-6                         | 官方 config               |
| warmup\_steps           | 500                          | 官方 60→我们改为 500          |
| total\_steps            | 30,000                       | 约 24 epoch              |
| batch\_size             | 6 / GPU                      | 4 / GPU（30s帧数高，先从 4 开始，能跑再试 6） |
| grad\_accum             | 1                            | 4GPU × 4 = 16 effective |
| max\_grad\_norm         | 1.0                          | 官方                      |
| EMA beta                | 0.995                        | 官方                      |
| EMA update\_after\_step | 100                          | 官方                      |
| CKA weight              | 1.0                          | 官方                      |
| optimizer               | AdamW β=(0.9, 0.95), wd=1e-2 | 官方                      |
| LR schedule             | Cosine decay (warmup 后)      | 官方                      |
| precision               | fp32                         | 与 V1 一致（若 OOM 可改 bf16）  |

### 7.2 CFG dropout

| 条件                 |    概率   | 说明                                |
| ------------------ | :-----: | --------------------------------- |
| drop\_audio (cond) |   30%   | 对齐官方默认值（Singer.__init__ `audio_drop_prob=0.3`）。V1 的 20% 是在 cond=GT bug 下为了加速 loss 下降而调的，修复后回归官方值 |
| drop\_text         |   30%   | 官方默认   |
| drop\_midi         |   30%   | 官方默认   |

三者独立随机，8 种组合均匀覆盖。

### 7.3 显存估算

| 组件                         | 显存                |
| -------------------------- | ----------------- |
| DiT (453.6M, fp32)         | ~1.8 GB          |
| VAE (156.1M, 冻结)           | ~0.3 GB          |
| SOME (117.6M, 冻结)          | ~0.2 GB          |
| 激活值 + 优化器 (batch=4, T≈646) | ~18-22 GB        |
| **合计**                     | **~22+ GB / GPU** |

30s / 646 帧 + batch=4，24GB 临界。如果 OOM：

| 优先级 | 操作 | 省多少 | 代价 |
|:---:|------|------|------|
| 1 | batch_size 4→3 | ~25% | 每 epoch 多 33% 步 |
| 2 | bf16 混合精度 | ~40% | 轻微精度损失 |
| 3 | gradient checkpointing | ~50% | 慢 ~20% |

### 7.4 训练时间

| 指标                 | 值                       |
| ------------------ | ----------------------- |
| 每 epoch            | 15,122 / 16 ≈ 945 steps |
| ~1.2s/step (4GPU set 30s/646帧) | 估算 |
| 30k 步              | **~10 小时**            |

### 7.5 启动命令

```bash
tmux session
tmux session ${SERVER_ROOT}/YingMusic-Singer-Plus && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_plus_v3.py \
  --train_json ${SERVER_ROOT}/final_sum_large/train_singnet.json \
  --eval_json ${SERVER_ROOT}/final_sum_large/test_singnet.json \
  --output_dir ckpts/plus_ja_sft_v3 \
  --batch_size 4 --lr 7e-6 --warmup_steps 500 --max_steps 30000 \
  --max_duration 30 --seed 42' Enter
```

***

## 八、可调参数与调参指南

以下参数可能在训练中需要频繁调整。每次调整后记录在训练日志中。

### 8.1 影响最大的参数

| 参数 | 当前值 | 作用 | 调高后果 | 调低后果 |
|------|:---:|------|------|------|
| `lr` (learning_rate) | 7e-6 | 每步梯度更新的步长 | 收敛快但可能震荡/NaN；日文 untrained embedding 可能被冲散 | 收敛慢，可能需要更多步数；但更稳定 |
| `batch_size` | 4 | 每 GPU 同时处理几条音频 | 梯度更稳定，但显存压力大 → OOM | OOM 风险降低，但梯度噪声大，可能需要更多步数 |
| `max_duration` | 30 | 音频加载截断上限（秒） | 覆盖更多数据（当前 30s 覆盖 99.8%），但帧数高 → OOM | 截断更多数据，但训练更快/更稳定 |
| `cka_weight` | 1.0 | CKA 损失在总 loss 中的权重。CKA 衡量 DiT 中间层与 MIDI 的相似度 | 更强调旋律跟随，但可能牺牲文本学习 | 弱化旋律约束，文本可能学得更好但旋律漂移 |

### 8.2 可能调整的参数

| 参数 | 当前值 | 作用 | 调高后果 | 调低后果 |
|------|:---:|------|------|------|
| `warmup_steps` | 500 | LR 从 0 线性增长到目标值的步数 | 更平滑的启动，但延迟有效训练 | 可能早期不稳定（日文 untrained embedding 大幅梯度） |
| `drop_audio` | 30% | 每步随机丢弃 cond 的概率 | CFG 的 uncond 分支更多训练，但有效学习步减少 | 更多步看到 cond，但 uncond 可能欠拟合 |
| `drop_text` | 30% | 每步随机丢弃 text 的概率 | 同上，针对文本 | 同上 |
| `drop_midi` | 30% | 每步随机丢弃 midi 的概率 | 同上，针对旋律 | 同上 |
| `total_steps` | 30,000 | 训练总步数 | 可能过拟合（需监控 eval loss） | 可能欠拟合（日文 embedding 训练不足） |

### 8.3 LR 调参决策树

```
Step 0-500 (warmup):
  ├── Flow loss 稳步下降 → ✅ 继续
  └── Flow loss 不降或震荡 → 停止，lr 降到 5e-6

Step 500-2000:
  ├── Flow loss 持续降到 < 1.5 → ✅ lr 合适
  ├── Flow loss 在 1.5-2.0 平台不动 → 试 lr 1e-5
  └── Flow loss 剧烈震荡 ±0.5 → 降到 5e-6

Step 2000-5000:
  └── 看改词测试结果。如果改词无效但 loss 正常 → 不是 lr 的问题，是 cond/tokenize 的问题
```

### 8.4 OOM 应对优先级

每次 OOM 按以下顺序尝试，不回溯（即先做 1，还 OOM 再做 2）：

1. `batch_size` 4→3→2
2. 启用 bf16 混合精度
3. `gradient_checkpointing`
4. `max_duration` 30→20→15

***

## 九、分阶段验证计划

**核心原则：不等到 30k 步才验证，每 500-2000 步验证改词能力。**

### 9.1 阶段 0：G2P 一致性（训练前）

| 测试        | 内容                                                                              | 通过标准 |
| --------- | ------------------------------------------------------------------------------- | :--: |
| token 一致性 | 5 个日文样本，`japanese_to_ipa → pbt.phoneme2token` vs `CNENTokenizer.encode` (去+1偏移) | 全部匹配 |

### 9.2 阶段 1：自重建 loss 下降（Step 0-2000）

|   checkpoint  | 验证内容                |              通过标准              |
| :-----------: | ------------------- | :----------------------------: |
|    Step 50    | warmup 结束，loss 开始下降 |            Flow < 20           |
|    Step 500   | loss 正常收敛           |        Flow < 2.0, 无 NaN       |
| **Step 2000** | **自重建音频可听懂**        | **Whisper WER < 0.3，输出可辨识的日语** |

> ⚠️ 自重建只验证"DiT 能生成target singer的音频"，不代表文本条件生效。

### 9.3 阶段 2：改词测试（Step 2000-5000）← 关键门禁

对 5 条测试样本做改词翻唱验证。这是评估的核心转折点。

| 测试       | 方法                                  |             通过标准             |
| -------- | ----------------------------------- | :--------------------------: |
| **自重建**  | ref=melody=同一段，target\_text=原词      |           音频清晰、可辨日语          |
| **改词翻唱** | ref=melody=同一段，target\_text=不同的日文歌词 | **生成音频中出现 target\_text 的发音** |

**如果 Step 5000 时改词翻唱完全无效（模型仍输出原歌词）：**
→ cond 修复未生效或效果不足 → 回退检查训练代码 → 可能需升级到方案 B（异段 cond）

### 9.4 阶段 3：全面评估（Step 5000-30000）

|  step | 评估内容                     |
| :---: | ------------------------ |
|  5000 | 改词测试 — **门禁**            |
| 10000 | 改词 WER + 听感              |
| 20000 | 改词 WER + F0-CORR + 跨角色翻唱 |
| 30000 | 完整评估：7 首 JP benchmark 曲目 |

### 9.5 评估指标

| 指标    | 方法                                      |       目标       |
| ----- | --------------------------------------- | :------------: |
| 旋律保真度 | F0-CORR (皮尔逊)                           |     > 0.75     |
| 歌词可懂度 | Whisper large-v3 WER, vad\_filter=False | < 30% (歌唱转录困难) |
| 音色相似度 | WavLM speaker cosine                    |     > 0.85     |
| 自然度   | 人耳 MOS                                  |     > 3.5/5    |

### 9.6 评估数据

使用 `benchmark_vocals_ds/JP/` 的 7 首完整曲目：

| 歌手   | 歌曲                    |
| ---- | --------------------- |
| source singer | いっかい書いてさようなら、source song |
| source singer  | Awake Now             |
| source singer | メランコリック、轻飘飘时间         |
| 森罗万象 | Toge、ダイヤモンド           |

评估文件位于：

- 本地: `TEMP/benchmark_vocals_ds/JP/`
- 服务器: `${SERVER_ROOT}${CLOUD_ARTIFACT}`

### 9.7 评估推理需注意

1. **跳过 FuzzDisturb**（训练时数据增强，推理时不应启用）
2. **Whisper 评估用 large-v3，`vad_filter=False`**（默认 VAD 切碎唱歌音频导致误判）
3. **ref\_text 必须精确匹配 ref\_audio 内容**（ref\_text 错误导致输出噪声）
4. **音频时长 30s 封顶**（长序列内存压力 + 模型对齐能力有限）

***

## 十、预期风险

| 风险                               |  概率 |  影响 | 应对                                         |
| -------------------------------- | :-: | :-: | ------------------------------------------ |
| 方案 A cond 修复后模型仍不学 text          |  中  |  高  | Step 5000 改词测试门禁 → 升级方案 B（异段 cond）         |
| 27 个 untrained embedding 导致早期不稳定 |  中  |  中  | 500 步 warmup，监控前 50 步 loss                 |
| OOM                              |  低  |  中  | 降 batch\_size→4；开 bf16；降 max\_duration→10s |
| 训练中途 NaN 发散                      |  低  |  高  | 梯度裁剪 1.0，检查数据 Pipeline                     |
| G2P 改造后 token 不一致                |  低  |  高  | 阶段 0 G2P 验证必须通过才进入训练                       |
| 长音频生成不稳定                         |  中  |  低  | 训练数据 max_duration=30s 覆盖绝大多数，推理时分段 |

***

## 十一、执行顺序总览

```
Step 0: 清理服务器（kill sftv2, rm 无效checkpoint）
Step 1: 备份当前 Plus 源码（git stash / cp）
Step 2: CNENTokenizer G2P 改造（4 文件） + 一致性测试
Step 3: 编写 train_plus_v3.py + run_sft_v3.sh
Step 4: 上传到服务器
Step 5: 阶段 1 训练（Step 0-2000）+ 自重建验证
Step 6: 阶段 2 改词测试（Step 5000 ← 关键门禁）
  ├── 通过 → Step 7
  └── 失败 → 回退检查 → 可能升级方案 B
Step 7: 阶段 3 完整训练（Step 5000-30000）+ 全面评估
Step 8: （可选）GRPO 精调（依赖 SFT 改词翻唱验证通过）
```

***

## 十二、方案 B 备选（异段 cond）

如果方案 A 在 Step 5000 改词测试失败：

- 训练数据不变，只改 cond 来源
- 每条样本的 cond = 随机选target singer集中另一段 3 秒干声
- 需要数据筛选保证音色一致性——优先选择同一 BV 号的不同 segment（相同录音条件）
- midi 在 cond 区域仍清零（方案 B 下更关键）

***

## 十三、GRPO（按需启动）

`grpo_train.py` 已就绪（位于 `scripts_archive/yingmusic_plus/3_train_grpo/`）。仅在 SFT 通过改词测试后按需启动。四奖励均等权重 0.25（qwen\_asr\_wer + f0\_correlation + qwenfeat + sim\_wavlm\_large）。

***

## 十四、附录：关键文件速查

| 内容              | 路径                                                         |
| --------------- | ---------------------------------------------------------- |
| Plus 源码         | `YingMusic-Singer-Plus-src/src/YingMusicSinger/`           |
| DiT 架构          | `models/dit.py`                                            |
| Singer.sample() | `models/model.py`                                          |
| CNENTokenizer   | `utils/cnen_tokenizer.py`                                  |
| g2p\_generation | `utils/f5_tts/g2p/g2p_generation.py`                       |
| cleaners        | `utils/f5_tts/g2p/g2p/cleaners.py`                         |
| vocab.json      | `utils/f5_tts/g2p/g2p/vocab.json`                          |
| config          | `config/YingMusic_Singer.yaml`                             |
| V1 训练脚本         | `scripts_archive/yingmusic_plus/2_train_sft/train_plus.py` |
| V1 评估记录         | `docs/JA_SFT_DEBUG_LOG.md`                                 |
| 模块计划            | `docs/plus-finetune-plan-v2.md`                            |


















