# GRPO 训练计划

> 2026-05-29 | 基于 YingMusic-Singer-Plus 论文 §2.3 + 全套讨论结论
>
> **前置阅读**：[plus-finetune-fullplan-v0527.md](plus-finetune-fullplan-v0527.md)（V4 SFT 设计）→ [temp_quesion_0529.md](temp_quesion_0529.md)（全部决策过程）

---

## 目录

- [一、背景与目标](#一背景与目标)
- [二、论文 GRPO 原理速查](#二论文-grpo-原理速查)
- [三、数据管线](#三数据管线)
- [四、模型与权重](#四模型与权重)
- [五、Reward 模型](#五reward-模型)
- [六、训练设计](#六训练设计)
- [七、超参数汇总](#七超参数汇总)
- [八、GPU 与服务器](#八gpu-与服务器)
- [九、当前进度与前置条件](#九当前进度与前置条件)
- [十、附录：论文关键原文](#十附录论文关键原文)

---

## 一、背景与目标

### 1.1 我们做什么

在 YingMusic-Singer-Plus 的 DiT 模型上，用 **Flow-GRPO**（Group Relative Policy Optimization）对 V4 SFT checkpoint 做强化学习精调，目标：

> **在不牺牲旋律保真度的前提下，提升 B 区生成的歌词可懂度（PER）、音色一致性（SIM）和整体感知质量（VS）。**

### 1.2 为什么需要 GRPO

SFT Phase 2（加 melody conditioning + CKA loss）会导致 PER 反弹。因为：

- `L_MSE` 只能做整体 latent 重建，无法针对"发音不清"定向优化
- CKA loss 的权重 λ 只平移 trade-off 平衡点，不解决根本矛盾
- 大规模歌唱数据中不可避免的噪声构成数据集天花板

GRPO 通过**多目标 reward 直接优化非可微的感知指标**，绕过 SFT 的瓶颈。

### 1.3 论文参考

| 论文 | 链接 | 作用 |
|------|------|------|
| YingMusic-Singer-Plus | [arXiv:2603.24589v2](https://arxiv.org/abs/2603.24589) / [本地PDF](paper_yingmusic_singer_plus.pdf) | GRPO 流程、超参数、reward 模型 |
| Flow-GRPO | [arXiv:2505.05470v4](https://arxiv.org/abs/2505.05470) | ODE→SDE 转换原理、Denoising Reduction |
| DeepSeek GRPO | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | GRPO 原始算法（LLM 版） |

---

## 二、论文 GRPO 原理速查

### 2.1 前置：ODE → SDE 转换

Flow Matching 的采样是确定性 ODE，RL 需要随机探索。论文将 ODE 转为等价 SDE：

$$d\mathbf{x}_t = [\mathbf{v}_\theta(\mathbf{x}_t, t) - \sigma_t^2 t^2 \nabla \log p_t(\mathbf{x}_t)] dt + \sigma_t d\mathbf{w}_t$$

该 SDE 在原 ODE 的**所有时间步边际分布完全等价**——引入随机性不破坏生成质量。

**我们的做法**：SDE 只在有限窗口 [w_min=1, w_s=8] 内激活，其余步走确定性 ODE。目的：将探索归因到特定步，提高梯度稳定性。

### 2.2 GRPO 训练循环（4 阶段）

```
阶段1: 采样         阶段2: Reward打分       阶段3: 组内标准化      阶段4: 策略更新
┌──────────────┐    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 对同一prompt   │    │ Qwen3-ASR    │     │ R_i - mean   │     │ PPO-clip     │
│ 采样 G=8 个   │───▶│ WavLM        │────▶│ ──────────   │────▶│ + KL正则     │
│ 候选音频      │    │ F0-CORR      │     │    std       │     │ → 更新 θ     │
│ (SDE窗口内随机)│    │ VocalVerse2  │     │ 加权求和→A_i │     │              │
└──────────────┘    └──────────────┘     └──────────────┘     └──────────────┘
```

### 2.3 核心公式

**Advantage（组内标准化）**：

$$A^i = \sum_{k=1}^{M} w_k \cdot \frac{R^i_k - \text{mean}(\{R^j_k\}_{j=1}^G)}{\text{std}(\{R^j_k\}_{j=1}^G)}$$

- $M$：reward 模型数（4，等权）
- $G$：组大小（8）
- 不需要 Value Network——组内均值就是 baseline

**GRPO Loss**：

$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|S|} \sum_{t \in S} \left( -\mathcal{L}_{\text{clip}} + \beta \cdot D_{\text{KL}} \right)$$

其中：

$$\mathcal{L}_{\text{clip}} = \min\left(r_t(\theta) \cdot A^i,\ \text{clip}(r_t(\theta), 1-\epsilon_l, 1+\epsilon_u) \cdot A^i\right)$$

$$r_t(\theta) = \frac{\pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t)}{\pi_{\text{old}}(\mathbf{a}_t \mid \mathbf{s}_t)}$$

$$D_{\text{KL}} = D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

- $S$：SDE 窗口内的步（不是所有去噪步）
- $\beta = 1$：KL 正则化强度
- $\epsilon_l = 0.002$、$\epsilon_u = 0.01$：极窄且不对称的 clip range

### 2.4 论文 MDP 建模

| MDP 元素 | 定义 |
|----------|------|
| 状态 $\mathbf{s}_t$ | $(\mathbf{c}, t, \mathbf{x}_t)$ — condition、时间步、当前 noisy latent |
| 动作 $\mathbf{a}_t$ | $\mathbf{x}_{t-1}$ — 去噪一步后的 latent |
| 策略 $\pi_\theta$ | $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{c})$ |
| 转移 | 确定性：下一状态 = (c, t-1, 去噪结果) |
| 奖励 | 仅终端步 t=0 非零：$R = r(\mathbf{x}_0, \mathbf{c})$ |

---

## 三、数据管线

### 3.1 数据来源

| 数据集 | 路径（服务器） | 条数 | 语言 | 状态 |
|--------|------|:---:|:--:|:--:|
| 训练集 | `${SERVER_ROOT}/final_sum_large/train_singnet.json` | 15,122 | ja | 已就绪 |
| 测试集 | `${SERVER_ROOT}/final_sum_large/test_singnet.json` | 307 | ja | BV 号隔离 |
| 音频 | `${SERVER_ROOT}/final_sum_large/audio/` | 15,429 wav | — | 44.1kHz, ~30GB |
| A7 时间戳 | `${SERVER_ROOT}/final_sum_large/pretreatment_text/timeset/` | 8 JSON | — | ✅ 已完成 |

A7 时间戳分档：

| 档位 | 条件 | 样本数 | GRPO 用途 |
|------|------|:---:|------|
| L1 | sim≥0.9, 短语≤10 | ~7,700 | 精确时间戳对齐 |
| L2 | sim≥0.7 | ~3,900 | 精确对齐，允许偏差 |
| L3 | sim≥0.5 | ~1,000 | 均匀分布兜底 |
| L4 | sim<0.5 | ~2,200 | **丢弃** |

### 3.2 数据构造：AB 区分割

GRPO 沿用 V4 SFT 的 AB 区分割。这是论文 Random Mask 的**严格特例**（start=0）：

```
时间轴 →   [0 ───── ref_len ───── T]

A 区（参考区）               B 区（生成区）
████████████████░░░░░░░░░░░░░░░░░░░░░░
cond = GT VAE latent       cond = 零
midi = 零                  midi = GT（target 音频提取）
text = 真 token            text = 真 token
x_t = 标准CFM              x_t = 标准CFM
（有cond直接参考）          （仅凭 text+midi 生成）
```

| 参数 | 值 | 说明 |
|------|:--:|------|
| ref_len | 随机短语边界 | 随机选一个中间短语边界切割，约束 A ≠ ∅, B ≠ ∅ |
| 5s 下限 | 108 帧 | 音频 < 7.7s 时生效 |
| 65% 上限 | 0.65T | 极少触发 |

> 放弃论文的 12.5%-33% 约束——经 L1 随机 20 条人耳验证，时间戳偏差率仅 10%（2/20），且均为 ±1-2 假名级别边界噪声，放宽比率不增加风险。

### 3.3 GRPO 专用数据过滤（已跳过）

| 过滤 | 工具 | 目的 | 状态 |
|------|------|------|:--:|
| ① ASR WER < 5% | Whisper Large V3 | 确保 PER reward 标签可靠 | **已跳过** |
| ② 单人检测 | ~~pyannote diarization~~ | 排除多人合唱干扰 | **已跳过** |
| ③ 音质 ≥ 3.5 | DNSMOS P.808 | 排除低质量音频 | **已跳过** |

> ② target singer单人数据，风险低。
> ① 实测 10 条 L1 标注 vs ASR：CER 偏差来自 ASR 对歌唱音频的转录能力限制（平均 0.161），非 kana 标注错误。按 WER<5% 过滤将删除 >90% 数据，不可行。
> ③ DNSMOS 分数区间窄（2.5-3.4），对target singer区分度有限。

---

## 四、模型与权重

### 4.1 架构

| 组件 | 参数量 | 训练时 |
|------|:--:|:--:|
| DiT CFM backbone | ~453.6M | 🔥 可训练 |
| VAE（Stable Audio 2） | ~156.1M | ❄️ 冻结 |
| Melody Extractor（SOME Teacher） | ~117.6M | ❄️ 冻结 |
| **合计** | ~727.3M | — |

DiT 结构：22 层、16 头注意力、hidden_dim=1024、FF 扩展因子 2、4 个 ConvNeXt 卷积层、Long Skip Connection。

### 4.2 权重文件

| 文件 | 路径（服务器） | 大小 | 用途 |
|------|------|:--:|------|
| SFT 起点 | `ckpts/plus_ja_sft_v4/model_ckpt_steps_XXXXX.pt` | ~7.6GB | 当前策略 $\pi_\theta$ 初始化 |
| 参考模型 | 同上（冻结） | — | KL 正则化用 $\pi_{\text{ref}}$ |
| VAE | `ckpts/stable_audio_2_0_vae_20hz_official.ckpt` | 596MB | encode/decode |
| SOME Teacher | `ckpts/model_ckpt_steps_100000_simplified.ckpt` | 449MB | 旋律提取 |

### 4.3 三个模型角色

GRPO 需要同时维护三份 DiT 权重：

| 角色 | 变量名 | 更新方式 | 用途 |
|------|--------|---------|------|
| **当前策略** | `policy` / $\pi_\theta$ | 每步 optimizer.step() 更新 | 采样候选 + 被优化 |
| **旧策略** | `old_policy` / $\pi_{\text{old}}$ | 每步从 policy 深拷贝 | PPO ratio 分母 |
| **参考模型** | `ref_model` / $\pi_{\text{ref}}$ | 冻结（不更新） | KL 散度正则化目标 |

---

## 五、Reward 模型

### 5.1 最终组合方案（Phase 1 验证完成）

| 奖励 | 指标 | 模型 | 权重 | 实测分数范围 | 状态 |
|------|------|------|:--:|------|:--:|
| $R_{DNSMOS}$ | 音频干净度 | DNSMOS P.808 | **0.45** | 2.5-3.4（原唱>自克隆 9/10） | ✅ 已验证 |
| $R_{PER}$ | 可懂度 | Whisper Large V3 | **0.25** | CER 0.023-0.424（平均 0.161） | ✅ 已验证 |
| $R_{SIM}$ | 音色相似度 | WavLM-large | **0.15** | 0.88-0.94 | ✅ 已验证 |
| $R_{F0}$ | 旋律保真度 | torchcrepe | **0.15** | 0.94-0.98 | ✅ 已验证 |

**总公式**：

$$A^i = 0.45 \cdot A^i_{DNSMOS} + 0.25 \cdot A^i_{PER} + 0.15 \cdot A^i_{SIM} + 0.15 \cdot A^i_{F0}$$

**权重逻辑**：

| 权重 | 模型 | 理由 |
|:--:|------|------|
| 0.45 | DNSMOS | 核心目标：让自克隆输出无需 SVC 即可使用。DNSMOS 纯测音质、无风格偏见，方向 4/4 全对 |
| 0.25 | PER | FlowB 最弱点：长句咬字崩塌。权重要够但不能牺牲音质 |
| 0.15 | SIM | 强约束（midi conditioning + ref latent），基本无退化风险，安全网 |
| 0.15 | F0-CORR | SOME Teacher 输出 128 维帧级 pitch 概率，旋律已高度保真，低权重兜底 |

---

### 5.2 Phase 1 验证记录

**测试数据**：fullseg_short（10 条短段自克隆，1-3 短语，7-20s）

#### ① DNSMOS P.808（音频干净度）

| 项目 | 详情 |
|------|------|
| 安装 | `pip install speechmos onnxruntime` |
| 调用 | `from speechmos import dnsmos; dnsmos.run(path, sr=16000)` |
| 硬件 | CPU（ONNX Runtime），无需 GPU |
| 公式 | $R_{DNSMOS} = \text{P808\_MOS} / 5.0$（归一化到 [0,1]） |

**10 对自克隆实测**：

```
原唱 3.30 3.09 2.96 2.72 3.25 2.91 3.01 3.06 2.89 3.16  (均 3.03)
克隆 2.96 2.88 2.88 2.66 3.16 2.51 2.78 2.77 2.86 3.35  (均 2.88)
差距 +.34 +.21 +.08 +.06 +.09 +.40 +.23 +.29 +.03 -.19
```

- **正确方向 9/10**（仅 slide10 原唱音质受损导致克隆反而更干净）
- 分数窄但稳定（2.5-3.4），对音质受损、长句退化、口胡敏感
- 不受歌唱风格影响（帅气 vs 甜美唱法分数无偏见）

#### ② Whisper Large V3（PER/WER）——已替换 Qwen3-ASR

| 项目 | 详情 |
|------|------|
| 安装 | `pip install faster-whisper`（已安装） |
| 调用 | `WhisperModel("large-v3", device="cuda", compute_type="float16")` |
| 转录 | `model.transcribe(path, language="ja", beam_size=5)` |
| 显存 | 10.9 GB（推理） |
| 公式 | $R_{PER} = 1 - \text{CER}(\text{kana}_{\text{ref}}, \text{kana}_{\text{ASR}})$ |

**10 条 L1 实测（vs Qwen3-ASR）**：

| 模型 | 平均 CER | 显存 |
|------|:--:|:--:|
| Qwen3-ASR-1.7B | 0.279 | 9.1 GB |
| **Whisper Large V3** | **0.161 (-42%)** | 10.9 GB |

Whisper 在 9/10 条上 CER 更低，CER<0.1 从 0 条提升到 3 条。额外 1.8 GB 显存代价可接受（单卡仍 85% 以内）。

#### ③ WavLM-large（音色相似度）

| 项目 | 详情 |
|------|------|
| 路径 | `${SERVER_ROOT}/pretrained_models/wavlm-large/`（1.2GB） |
| 调用 | last_hidden_state → temporal mean → L2 normalize → cosine similarity |
| 公式 | $R_{SIM} = \text{cos}(\text{embed}(\text{gen}), \text{embed}(\text{ref}))$ |

**2 对实测**：slide5 R_SIM=0.940, slide3 R_SIM=0.889

#### ④ F0-CORR / torchcrepe（旋律保真度）

| 项目 | 详情 |
|------|------|
| 工具 | torchcrepe |
| 调用 | `torchcrepe.predict(audio, sr=16000, hop_length=160, fmin=50, fmax=2000, model='full')` |
| 公式 | $\text{Pearson}(\text{F0}_{\text{gen} \text{[voiced]}}, \text{F0}_{\text{ref} \text{[voiced]}})$，仅 voiced 帧 |

**2 对实测**：slide3 R_F0=0.982, slide5 R_F0=0.946

---

### 5.3 淘汰记录

#### VocalVerse2（已淘汰）

| 项目 | 详情 |
|------|------|
| HF | `karl-wang/QwenFeat-Vocal-Score`（audioscore 模块） |
| 架构 | MuQ-large-msd-iter + LoRA + 打分头 |
| 训练数据 | ~1,000 段中文清唱，165 名业余标注者评"听感愉悦度" |
| 安装复杂度 | git-lfs + MuQ 基座 + peft + pyworld + 多个依赖，已全部完成 |

**淘汰原因**：人耳分层验证确认该模型评估的是**歌唱艺术表现力**（音色/气息/情感/技巧），而非音频质量：

| 问题 | 现象 | 根因 |
|------|------|------|
| slide6 高音多 → 3.56 最高分 | 奖励演唱难度 | 艺术性评分 |
| slide1 帅气唱法 → 0.63 垫底 | 惩罚非甜美感 | 风格偏见 |
| slide9 clone 2.02 > original 1.31 | 克隆更"干净"被误判为技巧好 | 音质≠艺术 |

不适合作为 GRPO 质量 reward——会引导模型追求"唱得更花哨"而非"唱得更清晰"。

#### pyannote 单人检测（已跳过）

用户决定跳过——当前训练数据主要为target singer晴陸单人演唱，多人合唱风险低。

---

## 六、训练设计

### 6.1 脚本结构

**完全重写**（不基于 V4 SFT 脚本，防止 SFT 设计污染）。共享底层工具函数：

| 复用函数 | 来源 | 用途 |
|---------|------|------|
| `VAE.encode_audio()` | stable_audio_2_0_vae | latent 编码 |
| `VAE.decode_audio()` | stable_audio_2_0_vae | latent → 波形 |
| `SOME_Teacher.forward()` | SOME checkpoint | 旋律提取 |
| `encode_text_with_sep()` | train_plus_v4.py | 日文 G2P |
| `compute_ref_len()` | train_plus_v4.py | A7 短语边界吸附 |
| `DiT.forward()` | dit.py | CFM 前向 |
| `FuzzDisturb` | model.py | midi 扰动 |

**文件位置**：

```
scripts_archive/yingmusic_plus/3_train_grpo/
├── train_grpo.py          # 主训练脚本
├── grpo_utils.py           # Advantage 计算、组内标准化
├── reward_models.py        # Whisper Large V3 / WavLM / F0-CORR / DNSMOS 封装
└── run_grpo.sh             # 启动脚本
```

### 6.2 训练循环伪代码

```python
# ===== 初始化 =====
policy = DiT(ckpt=V4_SFT_best_checkpoint)
ref_model = DiT(ckpt=V4_SFT_best_checkpoint, frozen=True)
old_policy = copy.deepcopy(policy)

# Reward 模型（单例，部署在 reward GPU 上）
whisper = WhisperModel("large-v3", device="cuda")
wavlm = WavLM(device=reward_gpu)
f0_extractor = torchcrepe
dnsmos = DNSP808()  # ONNX, CPU

optimizer = AdamW(policy.parameters(), lr=7e-6)

for step in range(1, 4801):
    # ===== 阶段1: 采样 G=8 候选 / prompt =====
    batch_candidates = []
    for prompt in batch:                       # batch_size=6
        cond, midi, text, ref_len, T = prepare_ab_region(prompt)
        # AB区: cond A区=真值 B区=零, midi A区=零 B区=真旋律

        for i in range(8):                     # G=8
            x_t = torch.randn(1, T, D)         # 纯噪声起步
            # 10步去噪（Denoising Reduction）
            for t_step in linspace(0, 1, 10):
                # SDE 窗口 [1,1+8) 内加噪声 α=0.8
                if 1 <= t_idx < 9:
                    x_t = x_t + noise * 0.8 * sqrt(dt)
                v = policy(x_t, cond, text, t_step, midi)
                x_t = x_t + v * dt             # Euler 步进
            audio = vae.decode(x_t)            # latent → 波形 44.1kHz
            batch_candidates.append({
                'audio': audio,
                'prompt_idx': prompt_idx,
                'ref_audio': ref_wav,
                'ref_f0': ref_f0
            })

    # ===== 阶段2: Reward 打分 =====
    all_scores = []
    for cand in batch_candidates:
        per = 1 - cer(cand['audio'], cand['ref_kana'])  # Whisper
        sim = wavlm.cos_sim(cand['audio'], cand['ref_audio'])
        f0_corr = pearson_corr(f0_extractor(cand['audio']), cand['ref_f0'])
        dns = dnsmos.run(cand['audio']) / 5.0
        all_scores.append([per, sim, f0_corr, dns])

    # ===== 阶段3: 组内标准化 → Advantage =====
    # 按 prompt 分组，每组 G=8 个候选独立标准化
    advantages = []
    for p_idx in range(batch_size):
        group_scores = all_scores[p_idx*8 : (p_idx+1)*8]  # [8, 4]
        group_scores = torch.tensor(group_scores)           # [8, 4]
        mu = group_scores.mean(dim=0)                       # [4]
        std = group_scores.std(dim=0) + 1e-8               # [4]
        weights = [0.25, 0.15, 0.15, 0.45]  # PER, SIM, F0, DNSMOS
        A = ((group_scores - mu) / std) @ weights           # [8]
        advantages.append(A)
    advantages = torch.cat(advantages)  # [48]

    # ===== 阶段4: 策略梯度更新 =====
    old_policy.load_state_dict(policy.state_dict())

    total_loss = 0
    for cand, A in zip(batch_candidates, advantages):
        # 仅对 SDE 窗口内的步计算 loss
        for sde_step in range(1, 9):  # w_min=1, w_s=8
            logp_new = policy.log_prob(x_t, cond, text, t_sde_step, midi)
            logp_old = old_policy.log_prob(x_t, cond, text, t_sde_step, midi)
            ratio = torch.exp(logp_new - logp_old)

            # PPO-clip（不对称：ε_l=0.002, ε_u=0.01）
            L_clip = torch.min(
                ratio * A,
                torch.clamp(ratio, 0.998, 1.01) * A
            )

            KL = torch.distributions.kl_divergence(
                policy.get_distribution(x_t, cond, text, t_sde_step, midi),
                ref_model.get_distribution(x_t, cond, text, t_sde_step, midi)
            )

            total_loss += (-L_clip + 1.0 * KL) / (8 * batch_size)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # ===== 每 500 steps 保存 =====
    if step % 500 == 0:
        torch.save(policy.state_dict(), f"ckpts/plus_grpo_v1/step_{step}.pt")
```

### 6.3 训练/推理配置

| 配置 | 训练时 | 推理时 | 说明 |
|------|:--:|:--:|------|
| 去噪步数 | **10** | **32** | Denoising Reduction |
| CFG | **禁用** | **scale=3** | 论文做法 |
| SDE 噪声 α | **0.8** | — | 仅训练时 |
| SDE 窗口 | **[1, 9)** | — | 仅训练时 |
| t_shift | 0.5 | 0.5 | 与 V4 SFT 一致（同官方推理） |
| x_t 初始化 | 全帧纯噪声 | 全帧纯噪声 | 与 V4 SFT 一致 |

### 6.4 Checkpoint 策略

| 参数 | 值 |
|------|:--:|
| 保存频率 | 每 500 steps |
| 总步数 | 4,800 |
| checkpoint 数 | 12（step_500 到 step_4800 每 500 步 + step_1200 论文基线） |
| 保存路径 | `ckpts/plus_grpo_v1/` |
| 保存内容 | policy 权重 + optimizer state + step |

> step_1200 作为论文原始配置的参考点额外保存，不在 500 步周期上。

---

## 七、超参数汇总

### 7.1 训练超参数

| 参数 | 值 | 来源 | 说明 |
|------|:--:|------|------|
| G（组大小） | 8 | 论文 §2.3 | 每 prompt 候选数 |
| batch_size | 6 | 论文 §3 | 每步 prompt 数 |
| 训练去噪步数 | 10 | 论文 Denoising Reduction | 加速采样 |
| 推理去噪步数 | 32 | 论文 §3 | 最终推理 |
| SDE 噪声 α | 0.8 | 论文 §3 | 探索噪声强度 |
| SDE 窗口 w_min | 1 | 论文 §3 | 窗口起始步 |
| SDE 窗口 w_s | 8 | 论文 §3 | 窗口宽度 |
| LR | 7e-6 | 论文 §3 | AdamW |
| total_steps | 4,800 | 论文 §3 × 4 | ~1.9 epochs（1.2K 为论文基线，4.8K 为扩展） |
| grad_clip | 1.0 | 标准 |
| checkpoint | 每 500 steps | 用户决定 |
| warmup | 0 | — |

### 7.2 PPO/GRPO 超参数

| 参数 | 值 | 说明 |
|------|:--:|------|
| β（KL 强度） | 1 | 防止 reward hacking |
| ε_l（clip 下界） | 0.002 | 极窄——保护 SFT 基座 |
| ε_u（clip 上界） | 0.01 | 最大更新幅度 1% |
| reward 权重 | DNSMOS 0.45, PER 0.25, SIM 0.15, F0 0.15 | 人耳验证后确定 |

### 7.3 Dropout/CFG

| 参数 | 训练时 | 推理时 |
|------|:--:|:--:|
| drop_audio_cond | — | — |
| drop_text | — | — |
| drop_midi | — | — |
| CFG | **禁用** | scale=3 |

GRPO 阶段所有 dropout 和 CFG 均关闭——策略梯度需要纯净的 reward 信号。

---

## 八、GPU 与服务器

### 8.1 服务器信息

| 项目 | 详情 |
|------|------|
| 地址 | `${SERVER_USER}@${SERVER_HOST}` |
| GPU | 8× RTX 4090 (24GB) |
| 空闲 GPU | 0-5（6 张可用） |
| 磁盘 | 7.0TB |
| 环境 | `yingmusic_plus` (python 3.10, torch 2.6.0+cu124) |
| Conda | `${CONDA_ROOT}` |
| 项目路径 | `${SERVER_ROOT}/YingMusic-Singer-Plus/` |

### 8.2 GPU 分配方案（最终）

**配置**：4× RTX 4090 DDP, batch_size=1/GPU set effective batch=4

| 组件 | 显存 | 生命周期 |
|------|:--:|------|
| DiT + VAE + MIDI Teacher | ~3.8 GB | S1-S4 常驻 |
| ref_model (DiT, 冻结) | ~1.8 GB | S1, S3, S4（**S2→CPU**） |
| AdamW optimizer state | ~3.6 GB | S1, S3, S4（**S2→CPU**） |
| Whisper Large V3 + WavLM | ~4.3 GB | S2 期间 GPU，**S4 后 free+reload** |
| S4 grad + activations | ~5 GB | S4 峰值 |
| torchcrepe + DNSMOS | ~0 GB | 轻量/CPU |

**显存管理策略**：
- S2 前：optimizer + ref_model → CPU swap（腾 ~5.4 GB）
- S4 后：del traj → empty_cache → free_whisper → reload Whisper（防碎片化）
- 单卡峰值 ~22 GB / 24 GB，零 OOM

### 8.3 依赖工具

| 工具 | 状态 | 安装方式 |
|------|:--:|------|
| `torchcrepe` | ✅ 已安装 | pip |
| `faster-whisper` | ✅ 已安装 | pip |
| `speechmos` | ✅ 已安装 | pip |
| `onnxruntime` | ✅ 已安装 | pip |
| `transformers` | ✅ 已安装 | pip |
| `peft` | ✅ 已安装 | pip |
| `muq` | ✅ 已安装 | pip（VocalVerse2 遗留，不再使用） |
| HF 镜像 | ✅ | `HF_ENDPOINT=https://hf-mirror.com` |

---

## 九、当前进度与前置条件

### 9.1 已完成

| 项目 | 状态 | 备注 |
|------|:--:|------|
| 论文调研（GRPO 原理 + 超参数） | ✅ | |
| V4 AB 区分割方案确认 | ✅ | |
| cond/midi/text/x_t 设计对齐 | ✅ | |
| 训练循环设计（伪代码） | ✅ | |
| **Phase 1: Reward 模型验证（四路全部通过）** | ✅ | 2026-06-02 |
| ├─ Qwen3-ASR PER 验证 | ✅ | CER 0.09-0.34, R_PER 0.66-0.91 |
| ├─ WavLM-large SIM 验证 | ✅ | cos_sim 0.88-0.94 |
| ├─ torchcrepe F0-CORR 验证 | ✅ | Pearson 0.94-0.98 |
| ├─ DNSMOS P.808 验证 | ✅ | 方向 9/10, 0.45 权重 |
| └─ VocalVerse2 淘汰 | ✅ | 艺术性评分非质量评分，替换为 DNSMOS |
| pyannote 单人检测（已跳过） | ✅ | target singer单人数据，风险低 |
| **Phase 3: 数据过滤** | ✅ 跳过 | 见 §3.3 — ASR WER 过滤不可行 |
| 最终 reward 权重确定 | ✅ | PER 2.0, SIM 1.0, F0 1.0, DNS 0.5（偏好咬字） |
| **AB 分割时间戳验证** | ✅ | L1 随机 20 条按短语边界 AB 切分，人耳确认 |
| ├─ 偏差率 | ✅ | 2/20 (10%)，仅 ±1-2 假名级别的边界噪声 |
| └─ 对 PER 影响 | ✅ | CER 波动 2-5%，组内标准化后稀释，可接受 |
| 脚本框架设计 | ✅ | |
| 依赖安装（qwen-asr, speechmos, onnxruntime, torchcrepe, peft） | ✅ | |
| **推理管线打通** | ✅ | 2026-06-04 |
| ├─ Singer.sample() 直接调用 | ✅ | 翻唱效果很好（target singer ref × source singer melody） |
| ├─ infer_v4_formal.py | ✅ | 独立干净推理脚本，支持翻唱/自克隆/改词 |
| └─ 确认 Sampler 差异非 root cause | ✅ | ODE+CFG=3 即推理配置，无需 SDE |
| **GRPO v2 Stage 1-4 单卡验证** | ✅ | 49s/step, 10步无NaN |
| **多卡 OOM 解决** | ✅ | 2026-06-04 |
| ├─ 根因定位 | ✅ | S4 反传碎片化导致 Whisper 找不到连续 3GB 块 |
| ├─ ref_model CPU swap | ✅ | Stage 2 期间 ref_model 挪 CPU，腾 1.8GB |
| ├─ S4 后重载 Whisper | ✅ | free_whisper + ensure_whisper，零碎片 |
| └─ batch_size=1 | ✅ | S4 反传图减半配合，~45s/step |
| **GRPO v2 训练启动** | ✅ | 2026-06-04 |
| ├─ 10 步验证 | ✅ | 零 OOM, Adv_std=1.0, KL<0.002 |
| └─ 4800 步正式训练 | ❌ 放弃 | 跨周期 OOM 累积，改为 v3 |
| **GRPO v3：独立 GPU Whisper** | ✅ | 2026-06-04 |
| ├─ reward_server.py | ✅ | GPU set 加载 Whisper，TCP 端口 15555 |
| ├─ RemoteWhisperClient | ✅ | reward_models.py 内，use_remote_whisper=True |
| ├─ 20 步验证 | ✅ | 零 OOM, 43s/step, 比 v2 快 15% |
| └─ 12000→4800 步训练 | ✅ | 已完成，~43s/step，零 OOM |
| **120 首分层评估** | ✅ | 2026-06-05 |
| ├─ SFT 24K baseline | ✅ | Total PER=0.387（4桶×30首×5 seeds） |
| ├─ GRPO 400/800/1200 | ✅ | +1.1%/+4.4%/+1.4% vs SFT，提升在噪声范围内 |
| └─ GRPO 4800 | ✅ | 峰值 800 步，无显著改善 |
| **V5a1 课程训练（论文复刻）** | ⚠️ | 2026-06-05 ~ 06-06 |
| ├─ Phase1 禁midi text-only 24K步 | ✅ | FlowB 0.97→0.93, 1.13s/step |
| ├─ 双测试集 PER 评估 | ⚠️ | 无显著提升（V4c=0.405 vs V5a1=0.334/不跨集复现） |
| ├─ 人耳 A/B 五首对比 | ⚠️ | 咬字拉不开差距 |
| └─ **结论** | ⚠️ | 回退 V4c 24K，不进入 Phase2 |

### 9.2 GPU 显存管理（v3 最终方案）

```
GPU set: DDP 训练（DiT + VAE + MIDI + WavLM + DNS/F0 本地计算）
GPU set:   Whisper TCP server（终生不动，零碎片）
```

| 方案 | 结果 |
|------|------|
| v2 同卡 + Whisper reload | OOM 5.6%，不可靠 |
| **v3 独立 GPU** | **零 OOM, 43s/step, 永久可靠** |

### 9.3 脚本索引

**训练：**
| 脚本 | 路径 | 用途 |
|------|------|------|
| train_grpo_v3.py | [scripts_archive/yingmusic_plus/3_train_grpo/train_grpo_v3.py](repository-relative-source) | v3 主训练脚本 |
| reward_server.py | [scripts_archive/yingmusic_plus/3_train_grpo/reward_server.py](repository-relative-source) | GPU set Whisper 服务 |
| reward_models.py | [scripts_archive/yingmusic_plus/3_train_grpo/reward_models.py](repository-relative-source) | 四路 Reward + RemoteWhisperClient |
| run_grpo_v3.sh | [scripts_archive/yingmusic_plus/3_train_grpo/run_grpo_v3.sh](repository-relative-source) | 启动（server + 训练） |

**评估：**
| 脚本 | 路径 | 用途 |
|------|------|------|
| eval_grpo.py | [tools/eval_grpo.py](repository-relative-source) | 120首×5 seeds 批量评估 |
| compare_multi.py | [tools/compare_multi.py](repository-relative-source) | 5首快速对比 |
| run_eval_batch.sh | [tools/run_eval_batch.sh](repository-relative-source) | 多 ckpt 串行评估 |

**推理：**
| 脚本 | 路径 | 用途 |
|------|------|------|
| infer_v4_formal.py | [scripts_archive/yingmusic_plus/4_inference/infer_v4_formal.py](repository-relative-source) | 翻唱/自克隆/改词推理 |

---

## 十、附录：论文关键原文

### 10.1 GRPO 定义（YingMusic-Singer-Plus §2.3）

> *"Following recent efforts, we convert the deterministic ODE trajectory into an SDE but restrict stochastic steps to a bounded window, ensuring precise advantage attribution to exploratory steps. To prevent collapse toward a single reward dimension, we employ M reward models jointly and compute the advantage for each sample as*
>
> $$A^{i}=\sum_{k=1}^{M}w_{k}\frac{R^{i}_{k}-\mathrm{mean}(\{R^{j}_{k}\})}{\mathrm{std}(\{R^{j}_{k}\})},$$
>
> *and the final GRPO loss is*
>
> $$\mathcal{L}_{\text{GRPO}}(\theta)=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|S|}\sum_{t\in S}\left(-\mathcal{L}_{\text{clip}}+\beta\,D_{\mathrm{KL}}\right),$$
>
> *where $\mathcal{L}_{\text{clip}}$ is the clipped surrogate objective over the current-to-old policy likelihood ratio, $D_{\mathrm{KL}}$ regularizes the current policy toward the reference, $G$ is the group size, $S$ the set of SDE sampling steps, and $\beta$ the KL regularization strength."*

### 10.2 实现细节（YingMusic-Singer-Plus §3）

> *"For GRPO, $G=8$ candidates are scored by $M=4$ equally weighted reward models (SDE noise $a=0.8$, window $w_{min}=1$, $w_s=8$, $\epsilon_u=0.01$, $\epsilon_l=0.002$, $\beta=1$), optimized for $1.2K$ steps (batch size 6, lr $7e-6$) without CFG. Inference uses $32$ ODE steps with CFG scale $3$."*

### 10.3 数据过滤方式（YingMusic-Singer-Plus §3）

> *"The GRPO dataset is constructed by filtering SFT data with three criteria: ASR transcript verification, retaining only clips with a word error rate below 5%, speaker diarization via pyannote, keeping only single-speaker clips, and a DNSMOS P808 quality score threshold of 3.5."*

### 10.4 Denoising Reduction（Flow-GRPO §4.3）

> *"We apply the Denoising Reduction strategy, which reduces denoising steps during training while keeping the full schedule during inference. Our experiments show that using fewer steps maintains performance while significantly reducing data generation costs."*

---

## 变更记录

| 日期 | 变更 |
|------|------|
| 2026-05-29 | 初稿：完整 GRPO 训练计划 |
| 2026-06-02 | **Phase 1 完成**：四路 reward 验证（DNSMOS/PER/SIM/F0）；VocalVerse2 淘汰；pyannote 跳过；权重 0.45/0.25/0.15/0.15；AB 分割验证 2/20 |
| 2026-06-02 | **Phase 2 完成**：GPU 显存实测（Whisper 10.9GB）；确定方案A（4卡DDP同卡，~85%）；AB 比率放宽为随机短语边界 |
| 2026-06-02 | **PER 模型切换**：Qwen3-ASR → Whisper Large V3（CER 0.279→0.161, -42%） |
| 2026-06-02 | **Phase 3 跳过**：10条L1验证后确认 ASR WER 过滤不可行（CER偏差为ASR能力问题非标注问题） |
| 2026-06-03 | **步数扩展**：total_steps 1.2K → 4.8K（~1.9 epochs），step_1200 额外保存作为论文基线参考 |
| 2026-06-04 | **推理管线打通**：Singer.sample() 直接调用，翻唱质量好。GRPO Stage 1-4 单卡全通（49s/step, 10步无NaN） |
| 2026-06-04 | **GRPO v2 脚本完成**：train_grpo_v2.py + run_grpo_v2.sh，使用 Singer.sample() 32步 ODE 替代论文 SDE |
| 2026-06-04 | **多卡 OOM 解决**：S4 反传碎片化为根因。修复：ref_model CPU swap + S4 后重载 Whisper + batch_size=1。零 OOM |
| 2026-06-04 | **GRPO v2 训练启动**：4×4090 DDP, 4800 steps, ~60h ETA, ~45s/step。含溯源日志（A/B时长+歌词+源文件） |
| 2026-06-04 | **GRPO v3：独立 GPU Whisper**：GPU set 专跑 Whisper TCP server，3×4090 训练。零 OOM, 43s/step |
| 2026-06-05 | **120 首分层评估体系**：4 桶 × 30 首 × 5 seeds。SFT 24K total PER=0.387，GRPO 800 = 0.404 (+4.4%) |
| 2026-06-05 | **GRPO 继续训练**：目标 4800 步，当前 ~1500/4800 |
| 2026-06-05 | **GRPO 4800 完成**：PER 峰值 800 步 (+4.4%)，整体无显著改善。V5a1 Phase1 启动 |
| 2026-06-06 | **V5a1 完成**：双测试集 + 人耳 A/B 无显著咬字提升。回退 V4c 24K |

















