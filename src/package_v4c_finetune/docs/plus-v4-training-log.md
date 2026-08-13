# YingMusic-Singer-Plus 日语 SFT V4 系列训练记录

## 背景

基于官方 YingMusic-Singer-Plus（453M F5-TTS DiT 22层），在目标歌手日语翻唱数据上进行 SFT，目标是实现**改词翻唱**：保持原曲旋律+音色，替换歌词。

V1/V2/V3 均因 cond=GT 作弊 / G2P 崩溃 / x_t 泄漏失败，权重已删除。

---

## 核心设计：A/B 区分割

| 区域 | 帧范围 | cond | midi | text | 含义 |
|------|:---:|:---:|:---:|:---:|------|
| **A区** | [0, ref_len) | 真latent | 0 | 真实token | 音色参考 |
| **B区** | [ref_len, T) | 0 | 真实旋律 | 真实token | 需模型生成 |
| ref_len | 12.5%~33% T，≥5s, ≤65% T | | | | |

## 数据

| 层级 | 条数 | 时长 | 时间戳 | 说明 |
|------|:---:|------|:---:|------|
| L1 | 7604 | 13.6h | A7 Whisper 精确 | 高质量 |
| L2 | 3546 | — | 中等精度 | — |
| L3 | 840 | — | 均匀兜底 | 低质量 |

---

## 训练记录

### V4（2026-05-30）
- **起点**：官方 base 权重
- **数据**：L1+L2+L3（11990条）
- **参数**：LR=7e-6 cos, grad_accum=16, CKA=1.0, drop=30%/30%/30%
- **步数**：~7k（中断）
- **Bug**：mel device / ref_len=0（单短语吸附到 0）→ FlowA=NaN
- **结论**：修复后 FlowB 从 27.7→0.99，仍在糊

### V4b（2026-05-31）
- **起点**：官方 base 权重
- **数据**：L1+L2+L3（11990条）
- **参数**：LR=1.4e-5 cos, CKA=0.7, drop_text=30%
- **步数**：~19k（V4c 启动后被杀）
- **FlowB 轨迹**：1.27(1k) → 1.04(2k) → 0.98(4k) → 0.92(12k) → 0.90(19k)
- **推理（18k）**：自重建糊但能听歌词，CFG=3 歌词变了但不精确
- **结论**：LR×2 有帮助，但 FlowB 仍平台 0.90

### V4c（2026-06-01）
- **起点**：V4c step_004000 → 4卡 DDP resume
- **数据**：L1+L2（11150条，去 L3）
- **参数**：LR=1.4e-5 warmup-hold(12k)-decay, CKA=0.7, drop_text=15%, B区×2
- **步数**：30k（完成）
- **4卡 DDP**：1.14s/step, eff_batch=16

| Step | LR | FlowA | FlowB | CKA |
|------|-----|-------|-------|-----|
| 6000 | 1.40e-5 | 0.41 | 0.95 | 0.18 |
| 8000 | 1.40e-5 | 0.35 | 0.92 | 0.16 |
| 12000 | 1.40e-5 | 0.38 | 0.95 | 0.17 |
| 16000 | 1.27e-5 | 0.32 | 0.92 | 0.17 |
| 20000 | 8.56e-6 | 0.36 | 0.92 | 0.18 |
| 24000 | 3.68e-6 | 0.30 | 0.90 | 0.15 |
| 30000 | 0.00 | 0.34 | 0.92 | 0.18 |

- **Eval 30000**：FlowA=0.30, FlowB=0.90, CKA=0.14
- **推理（2k→30k 11个 checkpoint）**：
  - 2k: 完全乱码
  - 6k: 有结构
  - 10k: 变清晰
  - 22k: 歌词最好
  - 24k→30k: 无明显改善
- **结论**：SFT 天花板 = FlowB=0.90。LR hold 帮助前期加速但不打破天花板。22k-30k 白跑。

### V4ca（2026-06-01，进行中）
- **起点**：V4c step_024000（EMA 权重）
- **数据**：L1 only（7604条）
- **参数**：LR=8e-7 cos, CKA=0.2, drop_text=15%, B区×2
- **步数**：目标 6k（~5.3 epoch）
- **目的**：超低 LR + L1 精确数据精调

---

## 关键发现

### FlowB 天花板
四轮实验（V4/V4b/V4c/V4ca），所有参数组合下 FlowB 硬地板 = **0.90**。

| 改动 | FlowB 效应 |
|------|:--:|
| LR 7e-6→1.4e-5 | -0.08 |
| cosine→warmup-hold | 加速，未突破 |
| CKA 1.0→0.7 | 微弱 |
| drop_text 30%→15% | 微弱 |
| B区×2 | 微弱 |
| L3→L1+L2 | 微弱 |
| **所有改动合计** | **0.99→0.90** |

**FlowB 平台 ≠ 没学习。** 22k 歌词比 20k 好、比 8k 好——text→latent 映射在持续改善，只是 loss 数值上不反映。

### 模型天花板推断
- V1（cond=GT 作弊）：Flow=0.45 → 清晰音频
- V4c（无作弊）：FlowB=0.90 → 有电音、糊
- 推测：22层 DiT + 28个新 JA embedding → 纯 text+midi→latent 精度上限 ≈ FlowB=0.85-0.90

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `train_plus_v4.py` | V4 单卡训练 |
| `train_plus_v4b.py` | V4b 单卡训练 |
| `train_plus_v4c.py` | V4c 单卡/DDP 训练（含 resume + hold-LR） |
| `train_plus_v4ca.py` | V4ca L1 精调 |
| `run_sft_v4c.sh` | V4c 单卡启动 |
| `run_sft_v4c_ddp.sh` | V4c 4卡 DDP resume |
| `run_sft_v4ca.sh` | V4ca 4卡 DDP |
| `infer_v4.py` | 推理脚本（复制官方 YingMusicSinger 类，改 checkpoint 加载） |
| `batch_infer.sh` | 批量推理（2k-22k） |
| `batch_infer_final.sh` | 批量推理（24k-30k） |
| `scripts/build_l1_html.py` | L1 数据预览 HTML 生成器 |

## Checkpoint 目录

| 版本 | 最佳 checkpoint | 说明 |
|------|:---:|------|
| V4 | step_006000 | 早期训练 |
| V4b | step_018000 | FlowB=0.90 |
| V4c | **step_024000** | FlowB=0.90, 歌词最佳 |
| V4c | step_030000 | 与 24k 无区别 |
| V4ca | 运行中 | L1 精调 |

---

## 下一步

- ✅ V4ca 继续跑（L1 精调）
- 📋 V4cb：A区 30-60% / CKA=0.2 / L1 only
- 📋 V4d：A区 30-60% / CKA=0.4 / L1+L2 从头
- 📋 如果都失败 → YingMusic-SVC 后处理降噪
