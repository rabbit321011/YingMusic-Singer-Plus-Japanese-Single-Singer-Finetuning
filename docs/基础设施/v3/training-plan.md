> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# YingMusic-Singer-Plus 训练计划

> 目标：在官方 Plus 权重基础上，微调支持花丸晴琉日语歌唱能力

## 一、背景

### 1.1 当前状态

| 项 | 状态 |
|------|:--:|
| Plus 推理 (CN/EN) | ✅ 已跑通，旋律跟随效果有限 |
| Plus 推理 (JA) | ❌ G2P 无日语路径, DiT 无日语训练 |
| Plus 训练脚本 | ❌ 仓库未开源，需自写 |
| 花丸晴琉数据 | ✅ 大量日语干声(100h，已切片)就绪 |
| GPU 资源 | ✅ 8×RTX 4090，4张空闲 (GPU 2-5) |
| 磁盘 | ✅ 7.0TB，已用 4.8TB，可用 1.8TB |

### 1.2 核心卡点

Plus 官方未开源训练代码——只提供了推理脚本和模型权重。需要从零写训练管线，参考 F5-TTS / DiffRhythm 的训练范式。

## 二、模块改动清单

### 2.1 G2P 前端 — 支持日语

**影响范围**：推理 + 训练都需要

| 文件 | 改动 | 工作量 |
|------|------|--------|
| `g2p_generation.py` → `is_japanese()` | 新增假名/汉字检测函数 | 10行 |
| `g2p_generation.py` → `get_segment()` | `zh/en/other` → `zh/en/ja/other` | 5行 |
| `g2p_generation.py` → `chn_eng_g2p()` | 改名为 `chn_eng_jpn_g2p()`, 加 ja 分支 | 10行 |
| `cleaners.py` → `cjekfd_cleaners()` | 补充 `japanese_to_ipa` import 和实现 | 15行 |
| `g2p/` → 新建 `japanese.py` | 或直接复用 `phonemizer_ja` (`EspeakBackend("ja")`) | 20行 |
| `cnen_tokenizer.py` | `from chn_eng_g2p` → `from chn_eng_jpn_g2p` | 1行 |

> **总计 ~60 行代码**。底层 `PhonemeBpeTokenizer` 已注册 `ja` 后端，`vocab.json` 已含日语音素 (ID 318-345, 28个token)。核心是把上层胶水代码连上。

**日语检测逻辑**：

```
is_japanese(ch):
    # 平假名 U+3040-U+309F
    # 片假名 U+30A0-U+30FF
    # 日文汉字也纳入 ja 段（和中文共用 Unicode，但由语境决定）
    return '\u3040' <= ch <= '\u30ff' or ch in common_kanji
```

### 2.2 VAE — 不动

Stable Audio 2，冻结，任何语言音频的 encode/decode 完美支持。

### 2.3 SOME MIDI Extractor — 不动

冻结，Mel 频谱提取旋律，纯声学、语言无关。

### 2.4 DiT — 核心训练目标

DiT 从未学过日语音素→音频的映射。需要日语 SVS 数据微调。

### 2.5 GRPO 奖励模型 — 不动

Qwen ASR、F0-CORR、WavLM 均天然支持日语。无需修改。

## 三、训练数据

### 3.1 现有数据

| 数据集 | 位置 | 大小 | 内容 |
|--------|------|------|------|
| `hanamaru_hareru_large_dataset_filtered` | 服务器 | 28GB | 15,671段 日语干声 WAV |
| `hanamaru_hareru_large_dataset_filtered_long` | 服务器 | 28GB | 15,326段 日语长干声 WAV |
| `hanamaru_hareru_whisper` | 服务器 | 12MB | Whisper ASR 结果 JSONL |
| `hanamaru_hareru_singer_sum/datas` | 服务器 | 6.7GB | 3,705段 `_Vocals_dry_dry_` WAV |
| `final_sum_large/train_singnet.json` | 服务器 | 30GB | 15,122条, 90.7h 训练集 |
| `benchmark_vocals_ds/JP/` | 本地+服务器 | — | 7首完整日语曲目 |

### 3.2 数据整合策略

```
方案 A（推荐）：用 filtered + filtered_long 做主数据
  ├─ 31,000+ 段日语干声
  ├─ whisper 结果 → 歌词文本
  └─ 每段自给自足训练 (timbre_ref = melody_ref = 同一音频)

方案 B：补充 final_sum_large CN/EN 数据
  ├─ 15,122条 CN 90.7h
  └─ 中英日三语混合训练，防遗忘中文能力

方案 C：从零收集更多日语 SVS 数据
  └─ GTSinger / Ofuton / Kiritan 等开源数据集
```

> **当前推荐 A+B**：日语数据为主（防遗忘中文），三语混合微调。

### 3.3 数据格式要求

训练时每条样本需要：

```python
{
    "audio_path": "segment.wav",      # 完整干声
    "text": "歌詞テキスト",            # 对应歌词
    "language": "ja",                  # 语言标签
    "duration": 5.2,                   # 秒
}
```

数据预处理流程：

```
WAV → VAE encode → latent [T, 64]
WAV → Mel Spec → SOME Teacher → midi [T, 128]
text → CNENTokenizer → token_ids [T]
```

## 四、训练脚本开发

### 4.1 需要实现的核心组件

| 组件 | 说明 | 参考 |
|------|------|------|
| **Dataset** | 加载 WAV + 歌词 → 在线 VAE encode + SOME MIDI + tokenize | F5-TTS Dataset |
| **Trainer** | Hydra 驱动，DDP 多 GPU | F5-TTS / DiffRhythm |
| **Singer Model** | 包装 DiT + VAE + SOME + CKA loss | 已有 model.py |
| **CKA Loss** | DiT 中间层 hidden states × MIDI reference 对齐 | 已有 `common.py:cka_loss()` |
| **CFG Dropout** | 训练时随机 drop audio_cond/text/midi | 参照 `dit.py` CFG 逻辑 |
| **Checkpoint** | EMA + 定期保存 | 参照 `checkpoint.py` |

### 4.2 训练循环伪代码

```python
for batch in dataloader:
    audio = batch.audio           # [B, T_wav]
    text = batch.text_tokens      # [B, T_text]

    # 1. VAE encode
    latent = vae.encode(audio)    # [B, T, 64]

    # 2. SOME MIDI
    mel = mel_spec(audio)
    midi, bound = some_teacher(mel)  # [B, T, 128]
    midi = MIDIFuzzDisturb(midi)     # equal_space drop

    # 3. Flow Matching
    t = rand(B)                   # U(0, 1)
    noise = randn_like(latent)
    x_t = (1-t) * noise + t * latent
    v_target = latent - noise

    # 4. CFG dropout
    drop_audio = rand(B) < 0.2
    drop_text  = rand(B) < 0.3
    drop_midi  = rand(B) < 0.3

    # 5. DiT forward
    v_pred = dit(x_t, latent, text, t, midi,
                 drop_audio, drop_text, drop_midi)

    # 6. Loss
    L_flow = MSE(v_pred, v_target)
    L_cka  = cka_loss(dit.hidden_states, midi)
    L_total = L_flow + L_cka

    # 7. Backward + EMA
    L_total.backward()
    clip_grad_norm(1.0)
    optimizer.step()
    ema.update()
```

### 4.3 需要写的新文件

```
train_plus/
├── train.py           # 主训练入口 (Hydra)
├── dataset.py         # 数据集类
├── trainer.py         # 训练循环 + EMA + checkpoint
├── config/
│   └── train.yaml     # 训练超参数
└── launch.sh          # 多 GPU 启动脚本
```

## 五、SFT 训练阶段

### 5.1 超参数（参考官方配置）

| 参数 | 官方值 | 我们的值 | 说明 |
|------|--------|----------|------|
| learning_rate | 7e-6 | 7e-6 | 保持 |
| warmup_steps | 60 | 100 | 略微增加 |
| total_steps | 31,518 | **待定** | 取决于数据量 |
| batch_size | 6/GPU | 6/GPU | 24 total @ 4 GPU |
| grad_accum | 1 | 2 | 等效 batch=48 |
| max_grad_norm | 1.0 | 1.0 | 保持 |
| EMA beta | 0.995 | 0.995 | 保持 |
| save_every | 100 | 500 | 保存频率 |
| CFG audio_drop | 0.2 | 0.2 | 保持 |
| CFG cond_drop | 0.2 | 0.3 | 保持 |
| CFG text_drop | 0.3 | 0.3 | 保持 |
| CFG midi_drop | 0.3 | 0.3 | 保持 |

### 5.2 步数估算

```
数据量: ~30,000 段 × 平均 6s = ~50h 日语
每个 epoch 步数: 30,000 / 24 = 1,250 steps (@ batch=24)
目标总步数: 取决于效果，建议先跑 5,000 步评估
完整训练: ~30,000 步 ≈ 24 epochs
```

### 5.3 启动命令

```bash
# 4 GPU DDP
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.run \
    --nproc_per_node=4 train_plus/train.py \
    --config train_plus/config/train.yaml
```

## 六、GRPO 训练阶段（可选）

### 6.1 前置条件

在 SFT 模型基础上，用强化学习精调。需要额外部署：

| 模型 | 来源 | GPU |
|------|------|-----|
| Qwen Omni (ASR) | HuggingFace | 1 GPU |
| WavLM Large | HuggingFace | 1 GPU |
| RMVPE (F0) | 已有 | CPU |

### 6.2 GRPO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| noise_level | 0.8 | 从 80% 噪声位置采样 |
| num_samples | 8 | 每组生成 8 个变体 |
| ppo_epochs | 1 | PPO 更新轮次 |
| num_steps | 32 | GRPO 总步数 |
| beta (KL penalty) | 1 | 防止偏离 SFT |
| upper_clip | 0.02 | PPO clip |
| lower_clip | 0.002 | PPO clip |
| t_shift | 0.5 | 采样时间偏移 |

### 6.3 是否做 GRPO？

| 场景 | 建议 |
|------|------|
| 旋律跟随已够好 | 不做 |
| 发音可懂但旋律漂移 | 做（F0-CORR 奖励会帮助） |
| 音色不稳定 | 做（WavLM 奖励会帮助） |

> **建议策略**：SFT 完成后评估，根据旋律/发音/音色三大指标决定是否 GRPO。

## 七、评估方案

### 7.1 评估指标

| 指标 | 方法 | 目标 |
|------|------|------|
| **旋律保真度** | F0-CORR (皮尔逊相关系数) | > 0.75 |
| **歌词可懂度** | Whisper ASR → WER/CER | < 15% |
| **音色相似度** | WavLM speaker embedding 余弦距离 | > 0.85 |
| **自然度** | MOS 主观评测 | > 3.5/5 |

### 7.2 测试用例

使用 `benchmark_vocals_ds/JP/` 的 7 首完整曲目：

| 歌手 | 歌曲 |
|------|------|
| 名取纱那 | いっかい書いてさようなら、フロントメモリー |
| 恋诗夜 | Awake Now |
| 时雨羽衣 | メランコリック、轻飘飘时间 |
| 森罗万象 | Toge、ダイヤモンド |

### 7.3 测试维度

| 测试类型 | 说明 |
|----------|------|
| 自一致性 (selfcons) | timbre_ref = melody_ref = 同一日语段 |
| 跨角色 CN→JA | 中文音色 × 日语旋律 |
| 跨角色 JA→JA | 不同日语歌手之间互翻 |
| melody_control | 同一旋律换歌词 |

## 八、GPU 资源与时间估算

### 8.1 可用 GPU

| GPU ID | 显存 | 状态 |
|--------|------|:--:|
| 0 | 24GB | 轻载 (1.1GB) |
| 1 | 24GB | ⚠️ 占用高 |
| **2** | **24GB** | **🟢 空闲** |
| **3** | **24GB** | **🟢 空闲** |
| **4** | **24GB** | **🟢 空闲** |
| **5** | **24GB** | **🟢 空闲** |
| 6 | 24GB | ⚠️ 占用中 |
| 7 | 24GB | ⚠️ 占用高 |

> 推荐使用 **GPU 2-5 (4张)** 做 DDP 训练。

### 8.2 显存估算

| 组件 | 显存 |
|------|------|
| DiT (453.6M, bf16) | ~1GB |
| VAE (156.1M, 冻结) | ~0.3GB |
| SOME (117.6M, 冻结) | ~0.2GB |
| 激活值 (batch=6, seq~200) | ~8-12GB |
| 优化器状态 (Adam) | ~2GB |
| 余量 | **~24GB 刚好** |

> 如果 OOM：降低 batch_size 到 4 或启用 gradient checkpointing。

### 8.3 训练时间估算

| 阶段 | 步数 | 预估时间 @4GPU |
|------|------|----------------|
| G2P 修改 + 调试 | — | 半天 |
| 训练脚本编写 + 调试 | — | 1-2天 |
| SFT 30,000 步 | ~30k | **~1.5 天** (0.25s/step) |
| 评估 | — | 半天 |
| GRPO (可选) | 32 | 1天 |

## 九、风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| **训练脚本开发量大** | 延迟 | 参考 F5-TTS/DiffRhythm 训练代码，复用已有 Singer/DiT 模块 |
| **日语数据标注不全** | 训练质量差 | 用 Whisper 自动标注 + 人工抽查 top 500 |
| **Whisper ASR 准确率不足** | 歌词有错 | 日语歌词数据较规范，错误率预期 <5% |
| **显存不足 OOM** | 训练中断 | 降 batch_size、开 gradient checkpointing |
| **微调后遗忘中文** | 中文能力退化 | 混合中英日三语数据训练 |
| **旋律跟随仍弱** | 效果不达预期 | 增大 CKA loss 权重、启用 GRPO |
| **长音频不稳定** | 分段 >30s 质量差 | 训练数据控制在 5-15s 段，推理时分段 |

## 十、实际进展 (2026-05-24)

```
Phase 0: 基础设施 ✅ 已完成
  ├─ G2P 日语支持           ✅ pyopenjtalk + japanese.py 复制
  ├─ espeak-ng ja 不可用     → 改换 pyopenjtalk (wasm "世界"→"chinese letter")
  ├─ has_japanese() 直通     ✅ 检测假名→整段走 japanese_to_ipa
  └─ 训练脚本                ✅ train_plus.py (SFT) + grpo_train.py

Phase 1: SFT 训练 🔄 进行中
  ├─ 小规模过拟合验证        ✅ 20条/200步 Loss 22.8→3.1
  ├─ 全量 SFT 30,000 步     🔄 运行中 (4GPU, ETA ~6.3h)
  │   Step  500: Flow=1.67  CKA=0.196  (warmup 结束)
  │   Step 1000: Flow=1.03  CKA=0.168  Eval=1.063 ✅
  │   Step 1700: Flow=0.67  CKA=0.162  (5.7%)
  └─ 每 2,000 步保存 checkpoint + eval

Phase 2: 评估 ⏳ 待开始
Phase 3: GRPO ⏳ 待开始 (脚本+奖励模型已就绪)
Phase 4: 部署 ⏳
```

### 实际遇到的坑

| 问题 | 原因 | 解决 |
|------|------|------|
| espeak-ng ja 输出乱码 | 不支持日文汉字 | 换 pyopenjtalk |
| get_segment() 判日文汉字为 "zh" | 中日汉字 Unicode 重叠 | `has_japanese()` bypass |
| 生僻汉字 "呸長戮" 报错 | Whisper 幻觉标注 | pyopenjtalk 跳过，不影响训练 |
| align_lrc_sentence_level token 超长 | 歌词长/音频短 | 捕获异常，截断 token |
| DDP rank 1 OOM/设备不匹配 | 多次修复 | 逐音频 encode、统一 device |

## 十一、文件结构总览

```
服务器端:
${PRIVATE_PATH}
├── train_plus.py         # SFT 训练脚本 (Flow+CKA+DDP+EMA+eval)
├── grpo_train.py          # GRPO 训练脚本 (Whisper+WavLM+torchcrepe→PPO)
├── infer_ft.py            # 加载微调权重推理
├── run_sft.sh             # SFT 启动脚本
├── train_sft.log          # 训练日志
├── src/                   # 模型代码
│   └── YingMusicSinger/utils/f5_tts/g2p/g2p/
│       ├── japanese.py    # 新增：日语 G2P (从 V1 复制)
│       ├── cleaners.py    # 修改：+japanese_to_ipa import
│       └── ../g2p_generation.py  # 修改：+has_japanese bypass
├── ckpts/
│   ├── plus_ja_sft/       # 新增：SFT checkpoint (每2000步)
│   └── plus_grpo/         # 新增：GRPO checkpoint
├── pretrained_models/
│   └── wavlm-large/       # 新增：WavLM 1.2GB
└── output/

本地端:
${LOCAL_PATH}
├── docs/
│   ├── architecture-plus.md
│   └── training-plan.md
├── train_plus.py
├── grpo_train.py
└── DEPLOY.md
```
