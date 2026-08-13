# GRPO 训练日志

> 2026-06-04 ~ 06-06 | v2 同卡 OOM → v3 远程 Whisper → 120 首评估 → GRPO 4800 完成 → V5a1 课程训练

---

## 一、关键脚本索引

| 脚本 | 本地路径 | 用途 |
|------|------|------|
| **train_grpo_v2.py** | [scripts_archive/yingmusic_plus/3_train_grpo/train_grpo_v2.py](repository-relative-source) | 同卡版本（Whisper reload 每 20 步） |
| **train_grpo_v3.py** | [scripts_archive/yingmusic_plus/3_train_grpo/train_grpo_v3.py](repository-relative-source) | **v3 远程 Whisper（当前运行版本）** |
| **reward_server.py** | [scripts_archive/yingmusic_plus/3_train_grpo/reward_server.py](repository-relative-source) | GPU set 上的 Whisper TCP 服务 |
| **reward_models.py** | [scripts_archive/yingmusic_plus/3_train_grpo/reward_models.py](repository-relative-source) | 四路 Reward + RemoteWhisperClient |
| **run_grpo_v3.sh** | [scripts_archive/yingmusic_plus/3_train_grpo/run_grpo_v3.sh](repository-relative-source) | 启动脚本（GPU set server + GPU set 训练） |
| **eval_grpo.py** | [tools/eval_grpo.py](repository-relative-source) | 120 首 × 5 seeds 批量评估 |
| **compare_multi.py** | [tools/compare_multi.py](repository-relative-source) | 5 首快速对比脚本 |
| **run_eval_batch.sh** | [tools/run_eval_batch.sh](repository-relative-source) | 多 checkpoint 串行评估 |
| **infer_v4_formal.py** | [scripts_archive/yingmusic_plus/4_inference/infer_v4_formal.py](repository-relative-source) | 推理脚本（翻唱/自克隆） |
| **compare_per.sh** | [tools/compare_per.sh](repository-relative-source) | 快速单歌 PER 对比 |

---

## 二、显存问题与演进

### 2.1 v2 方案：同卡 Whisper + reload

| 组件 | 显存 |
|------|:--:|
| DiT + VAE + MIDI + ref_model + optimizer | ~10 GB |
| Whisper + WavLM | ~4.3 GB |
| S4 grad + activations | ~5 GB |
| **总需** | **~19-23 GB** |

根因：S4 backward 碎片化，`empty_cache` 无法回收连续 3GB 块。尝试过：
- optimizer/ref_model CPU swap → 无效（碎片在常驻 tensor 之间）
- 每 5/20 步重载 Whisper → 短期有效，跨周期累积后仍 OOM

结论：同卡方案在 batch=1 下可以工作但不可靠。

### 2.2 v3 方案：独立 GPU Whisper（最终）

```
GPU set: DDP 训练（DiT + VAE + MIDI + WavLM + DNS/F0）
GPU set:   Whisper TCP server（终生不动）
```

| GPU | 任务 | 碎片问题 |
|:--:|------|:--:|
| 0-2 | 3×4090 DDP | 碎片但不需连续大块 |
| 3 | Whisper 转录 + PER | **零碎片——无 backward/optimizer** |

**效果：零 OOM，~43s/step，比 v2 快 ~15%。**

---

## 三、v3 架构细节

```
┌─────────────┐   TCP (JSON + pickle)   ┌──────────────┐
│  GPU set    │ ◄──────────────────────► │  GPU set       │
│  train_v3   │   gen_wavs + ref_kanas  │  reward_server│
│              │    → PER scores         │  Whisper L-V3│
│  WavLM local │                          │              │
│  DNS/F0 local│                          │              │
└─────────────┘                          └──────────────┘
```

- `reward_server.py`：加载 Whisper Large V3 (int8)，监听 127.0.0.1:15555
- `RemoteWhisperClient`：在 `reward_models.py` 中，`RewardModels(use_remote_whisper=True)` 激活
- SIM/F0/DNS 仍在训练 GPU 本地计算（WavLM ~1.3GB/卡）
- `free_whisper` / `ensure_whisper` 在远程模式下为 no-op
- 启动时 client 轮询等待 server 就绪（最多 90s）

---

## 四、奖励权重

| 版本 | PER | SIM | F0 | DNS | 说明 |
|------|:--:|:--:|:--:|:--:|------|
| v2 初期 | 2.0 | 1.0 | 1.0 | 0.5 | 偏好咬字 |
| **v3 当前** | **6.0** | **1.0** | **1.0** | **2.0** | **PER 60%, DNS 20%** |

---

## 五、评估体系

### 5.1 方法

- **测试集**：120 首歌曲，seed=42 随机抽样，按 B 区假名字符数分层（1-10/11-20/21-30/30+），每层 30 首
- **每首歌**：5 次推理（seed 42-46），PER 取平均
- **输出维度**：桶均值/方差/中位数/p30/p70/>=0.9比例、稳定性、衰减斜率

### 5.2 SFT 24K baseline

| Bucket | avg_PER | var | median | p30 | p70 | >=0.9 | 稳定性 |
|--------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1-10 | 0.416 | 0.428 | 0.240 | 0.000 | 0.867 | 26.7% | 76.7% |
| 11-20 | 0.429 | 0.290 | 0.442 | 0.289 | 0.561 | 6.7% | 56.7% |
| 21-30 | **0.562** | 0.323 | 0.629 | 0.401 | 0.822 | 10.0% | 80.0% |
| **30+** | **0.141** | 0.225 | 0.044 | 0.015 | 0.091 | 0% | 96.7% |
| **Total** | **0.387** | — | — | — | — | — | 0.079 |

- **21-30 字是最优区**，不是越短越好
- **30+ 字崩塌**（PER=0.141），SFT 对长文本几乎失效
- 衰减斜率 -0.005/char，长度影响小但 30 字是个断崖

### 5.3 GRPO vs SFT

| 步数 | 1-10 | 11-20 | 21-30 | 30+ | Total | Δ vs SFT |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| SFT 24K | 0.416 | 0.429 | 0.562 | 0.141 | 0.387 | — |
| GRPO 400 | 0.427 | 0.424 | 0.570 | 0.145 | 0.391 | +1.1% |
| GRPO 800 | 0.445 | 0.448 | 0.576 | 0.148 | 0.404 | **+4.4%** |

- 每一步都比前一步好，趋势真实
- 短歌（1-10）受益最大（+7%）
- 30+ 区域 Glass ceiling —— Whisper 歌唱转录天花板 ~0.84 CER
- 稳定性未退化（total std ~0.079），音质主观上更清晰

---

## 六、最终训练配置

| 参数 | v2 | **v3（当前）** |
|------|:--:|:--:|
| DDP GPUs | 4 | **3** (GPU set) |
| Reward GPU | — | **GPU set** |
| batch_size | 1 | 1 |
| G | 8 | 8 |
| lr | 3e-6 | 3e-6 |
| CFG | 3.0 | 3.0 |
| TRAIN_STEPS | 32 | 32 |
| BETA_KL | 1.0 | 1.0 |
| 目标步数 | — | **4800** |
| save_every | 500 | **100** |
| ~time/step | 50s | **43s** |

---

## 七、当前状态

**GRPO v3 训练已完成**：4800 步，tmux session`${SERVER_ROOT}/log_grpo_v3.txt`
- Checkpoint：`ckpts/plus_grpo_v3/step_XXXXXX.pt`（每 100 步）
- Whisper server：已随训练结束关闭

### 7.1 GRPO 4800 完整评估

| 步数 | 1-10 | 11-20 | 21-30 | 30+ | Total | Δ vs SFT |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| SFT 24K | 0.416 | 0.429 | 0.562 | 0.141 | 0.387 | — |
| GRPO 400 | 0.427 | 0.424 | 0.570 | 0.145 | 0.391 | +1.1% |
| GRPO 800 | 0.445 | 0.448 | 0.576 | 0.148 | 0.404 | **+4.4%** |
| GRPO 1200 | — | — | — | — | 0.392 | +1.4% |

- 最佳 GRPO 800 比 SFT 仅 +4.4%，1200 已回落
- 提升在统计噪声范围内（~1.1σ），非显著性改善
- 根因：SFT 阶段 text dropout 仅 15%，text conditioning 先天不足，GRPO 难以修复

### 7.2 V5a1 课程训练实验（论文复刻）

**目标**：复刻论文 Phase1（禁 midi text-only SFT），强化 text conditioning。

| 项目 | 值 |
|------|-----|
| 训练脚本 | [train_plus_v5a1.py](repository-relative-source) |
| 启动脚本 | [run_sft_v5a1_ddp.sh](repository-relative-source) |
| 起点 | V4c step_024000 |
| 核心改动 | midi 全零 / CKA 关 / text dropout 关 |
| 步数 | 24,000（4GPU DDP, ~1.13s/step, ~7.5h） |
| 最终 loss | FlowA=0.26, FlowB=0.97 |
| PER 评估 | 15 首双测试集：无明显提升（V4c=0.405 vs V5a1_18K=0.334 测试集1; V4c=0.272 vs V5a1=0.299 测试集2） |
| 人耳 A/B | 5 首对比，咬字拉不开差距 |
| **结论** | ⚠️ 回退 V4c 24K，不进入 Phase2 |

### 7.3 服务器 tmux session | 状态 |
|------|------|:--:|
| `v3` | GRPO v3 训练 | ✅ 已完成 |
| `v5a1` | V5a1 Phase1 训练 | ✅ 已完成 |
| `eval` | 评估脚本（按需启动） | ⏸ 空闲 |

---

## 变更记录

| 日期 | 变更 |
|------|------|
| 2026-06-04 | v2 同卡方案：Whisper reload 每 20 步，OOM 率 5.6% |
| 2026-06-04 | 换 v3 独立 GPU 架构：GPU set Whisper server, 零 OOM, 43s/step |
| 2026-06-04 | 奖励权重调整为 PER 60% + DNS 20%（`[6,1,1,2]`） |
| 2026-06-04 | 修复 eval `.unsqueeze(0)` bug（Step 100 崩溃） |
| 2026-06-04 | 搭建 120 首分层评估体系 |
| 2026-06-05 | SFT 24K baseline 评估完成（Total PER=0.387） |
| 2026-06-05 | GRPO 400/800 评估完成（+4.4% vs SFT @ step 800） |
| 2026-06-05 | GRPO 训练继续，目标 4800 步 |
| 2026-06-05 | GRPO 4800 完成，PER 峰值 800 步 (+4.4%)，整体无显著改善 |
| 2026-06-05 | V5a1 Phase1 启动：禁 midi text-only 24K 步，论文课程训练复刻 |
| 2026-06-06 | V5a1 完成，双测试集 + 人耳 A/B 均无显著咬字提升 |
| 2026-06-06 | **结论**：回退 V4c 24K 为主力模型。GRPO 和课程训练均未解决 text conditioning 问题 |

















