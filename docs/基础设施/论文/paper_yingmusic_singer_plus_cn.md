# YingMusic-Singer-Plus：可控歌声合成 —— 灵活歌词编辑与无需标注的旋律引导

**作者：** 郝春波 $^{1,2}$, 郑俊杰 $^{2}$, 马国斌 $^{1}$, 蒋悦鹏$^{1}$, 陈华康$^{1}$, 田文杰$^{1}$, 陈功宇 $^{2}$, 陈子浩 $^{2}$, 谢磊$^{1,*}$

**单位：**
1. 西北工业大学计算机学院 音频语音与语言处理组（ASLP@NPU），中国
2. 巨人网络 AI Lab，中国

**联系方式：** research-contact@example.org, research-contact@example.org, research-contact@example.org

**arXiv：** 2603.24589v2 [eess.AS] 2026年4月9日

**代码与示例：** https://github.com/ASLP-lab/YingMusic-Singer-Plus

**关键词：** singing voice synthesis, lyric editing, reinforcement learning, diffusion model

---

## 摘要

在保持旋律一致性的前提下，用修改后的歌词重新生成歌声仍然充满挑战，因为现有方法要么可控性有限，要么需要繁琐的手工对齐。本文提出 **YingMusic-Singer-Plus**，一个全 diffusion 模型，可实现旋律可控的歌声合成及灵活的歌词编辑。该模型接受三种输入：可选的音色参考片段、提供旋律的歌唱片段以及修改后的歌词，无需手工对齐。通过课程学习（curriculum learning）和 Group Relative Policy Optimization（GRPO）的训练，YingMusic-Singer-Plus 在旋律保持和歌词忠实度方面均优于 Vevo2——目前最具可比性的、支持无需手工对齐的旋律控制 baseline。本文还推出了 **LyricEditBench**，首个用于旋律保持型歌词修改评估的 benchmark。代码、权重、benchmark 及 demo 均已公开发布。

---

## 1. 引言

歌声合成（Singing Voice Synthesis, SVS）旨在从乐谱、歌词和音色参考中生成类人歌声。现代系统 [1–6] 已实现高保真合成，但大多数依赖精确标注的配对数据，即每个音素需关联确切的音高轮廓和时长。虽然这种细粒度控制对专业音乐制作不可或缺，但准备这些标注对日益重要的应用场景——**歌声编辑**——构成了巨大障碍。歌声编辑是指在保持原旋律和节奏结构的同时，用修改后的歌词重新生成现有歌声。这一能力对于歌曲改编、个性化翻唱生成、声乐编排快速原型制作以及多语种歌曲本地化具有极高的价值。

现有的编辑范式可分为两类：

- **上下文学习（In-context learning）**：掩蔽待编辑区域，以周围的上下文和目标歌词为条件重新生成 [7, 8]。虽然便捷，但此方法仅限于局部片段，且旋律控制能力有限。
- **手工对齐（商业 SVS）**：依赖 Synthesizer V、ACE Studio 等工具，用户需手动将修改后的歌词与 MIDI 音符和时值对齐后重新合成音频。该流程虽具有强可控性，但需要大量手工劳动，且对于跨语言翻译等任务的复杂度更高。

近期的一些工作在尝试解决这些挑战。Vevo2 [9] 实现了旋律可控生成，但存在清晰度降低和旋律遵循度差的问题。SoulX-Singer [10] 支持将现有歌声作为旋律输入，但仍需要词级时间戳的手工对齐，核心瓶颈尚未解决。

为应对这些挑战，本文提出 **YingMusic-Singer-Plus**，一个全 diffusion SVS 模型：

1. **简洁的编辑范式**：仅从三种输入合成歌声——可选的音色参考片段、提供目标旋律的歌唱片段和对应的修改歌词——无需手工对齐或精确标注。
2. **课程学习 + GRPO**：为缓解歌唱数据规模小、声乐技巧复杂导致的音素泛化能力有限的问题，并解决歌词忠实再现与旋律遵循之间的固有权衡，我们采用课程学习策略结合 Group Relative Policy Optimization（GRPO）[11]，在两者维度上同时实现强性能。
3. **LyricEditBench**：基于 GTSinger [12] 构建，首个在旋律匹配条件下评估歌词修改的 benchmark，覆盖六种常见编辑场景并进行均衡采样。

实验表明，YingMusic-Singer-Plus 在旋律保持和歌词遵循度方面均优于 Vevo2 [9]。

---

## 2. 方法

### 2.1. 架构总览

YingMusic-Singer-Plus 以 44.1 kHz 采样率生成歌声，接受三种输入：可选的音色参考、提供旋律的歌唱片段以及对应的修改歌词。其组成包括：

1. **Variational Autoencoder（VAE）**：沿用 Stable Audio 2 [13]，其编码器 $E$ 将 44.1 kHz 立体声歌唱波形 $\mathbf{x} \in \mathbb{R}^{T \times 2}$ 下采样 2048 倍得到 $\mathbf{z} = E(\mathbf{x}) \in \mathbb{R}^{T' \times D}$，解码器 $D$ 在推理时重建高保真音频 $\hat{\mathbf{x}} = D(\mathbf{z})$。

2. **Melody Extractor（旋律提取器）**：基于预训练 MIDI 提取模型的编码器构建，其中间表征自然地捕获了解耦的旋律信息。它产生 $\mathbf{h} = M(\mathbf{M}) \in \mathbb{R}^{L \times D_m}$，然后通过时间插值得到 $\tilde{\mathbf{h}} \in \mathbb{R}^{T' \times D_m}$ 以匹配 VAE 隐变量帧率。

3. **IPA Tokenizer（国际音标分词器）**：将中英文歌词统一转换为离散音素序列。为确保模型正确区分提示区域与生成区域，避免边界处的音素遗漏或重复，我们采用 DiffRhythm [14] 的句子级对齐策略。每个歌词句子被转换为 IPA 子序列，并放置在长度为 $T'$ 的 padding 帧级序列中对应的起始帧位置。对齐后的序列通过可学习嵌入层生成 $\mathbf{e} \in \mathbb{R}^{T' \times D_e}$。推理时，提示歌词置于序列开头，目标歌词置于掩蔽区域起始处，无需用户提供任何时间戳标注。

4. **DiT-based CFM Backbone（基于 DiT 的条件流匹配骨干网络）**：沿用 F5-TTS [15]。

训练时，随机掩蔽比例为 $\gamma$ 的 VAE 隐变量帧作为合成目标，未掩蔽部分作为音色上下文。令 $\mathbf{z}_{\text{ctx}}$ 表示未掩蔽的 VAE 隐变量（掩蔽区域零填充）。条件 $\mathbf{c} = [\tilde{\mathbf{h}}; \mathbf{e}; \mathbf{z}_{\text{ctx}}]$ 与带噪隐变量 $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_1$ 沿通道维度拼接后输入 CFM，该网络通过以下方式学习速度场：

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{t,\mathbf{z}_0,\mathbf{z}_1,\mathbf{c}} \left\| \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c}) - (\mathbf{z}_1 - \mathbf{z}_0) \right\|^2 \tag{1}$$

### 2.2. 课程学习（Curriculum Training）

为缓解歌唱数据规模小、声乐技巧复杂导致的音素泛化能力有限的问题，YingMusic-Singer-Plus 首先进行无旋律条件的 **TTS Pretraining（TTS 预训练）**。

随后的 **Singing Voice Supervised Fine-Tuning（SFT，歌声监督微调）** 阶段分为两个子阶段：

- **Phase 1**：在歌唱数据上启用句子级对齐，使模型适应歌唱领域。
- **Phase 2**：激活旋律条件并引入 **Centered Kernel Alignment（CKA）** 损失以强制旋律遵循。

给定预测的 $\mathbf{v}_\theta$ 和旋律 $\tilde{\mathbf{h}}$，CKA 通过 Gram 矩阵 $\mathbf{K} = \mathbf{v}_\theta \mathbf{v}_\theta^\top$ 和 $\mathbf{L} = \tilde{\mathbf{h}} \tilde{\mathbf{h}}^\top$ 衡量其对齐程度：

$$\mathcal{L}_{\text{CKA}} = 1 - \frac{\|\mathbf{K}^\top \mathbf{L}\|_F^2}{\|\mathbf{K}^\top \mathbf{K}\|_F \|\mathbf{L}^\top \mathbf{L}\|_F} \tag{2}$$

Phase 2 的总损失为：

$$\mathcal{L}_{\text{SFT}} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{CKA}}$$

### 2.3. Group Relative Policy Optimization（GRPO）

虽然课程学习取得了高性能，但 SFT Phase 2 同时使 PER 下降，暴露了持续的权衡问题。$\mathcal{L}_{\text{MSE}}$ 以整体隐变量重建为目标，无法针对特定缺陷进行定向优化；调整 $\mathcal{L}_{\text{CKA}}$ 中的 $\lambda$ 仅能改变平衡点，无法从根本上解决该权衡。此外，大规模歌唱数据中不可避免的噪声（如背景和声、质量瑕疵）将模型性能限制在数据集水平。

为克服这些限制，**强化学习（RL）** 变得必要。PPO 需要训练昂贵的价值网络。DPO 等离线方法面临分布偏移问题，因为随着策略改进，预收集的偏好数据会变得过时。**GRPO** [11] 在线运行，利用组内奖励统计估计 baseline，消除了价值网络，同时保持高效和稳定。

沿用近期工作 [16–18] 的做法，我们将确定性 ODE 轨迹转换为 SDE，但将随机步限制在有界窗口内，确保探索步的优势归因精确。为防止向单一奖励维度坍缩，我们联合使用 $M$ 个奖励模型，并计算每个样本的优势为：

$$A_i = \sum_{k=1}^{M} w_k \cdot \frac{R_i^k - \text{mean}(\{R_j^k\})}{\text{std}(\{R_j^k\})} \tag{3}$$

最终的 GRPO 损失为：

$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|S|} \sum_{t \in S} \left( -\mathcal{L}_{\text{clip}} + \beta D_{\text{KL}} \right) \tag{4}$$

其中 $\mathcal{L}_{\text{clip}}$ 是当前策略与旧策略似然比的裁剪代理目标，$D_{\text{KL}}$ 将当前策略向参考策略正则化，$G$ 为组大小，$S$ 为 SDE 采样步集合，$\beta$ 为 KL 正则化强度。

### 2.4. LyricEditBench

我们从 GTSinger [12] 构建 LyricEditBench：移除所有 Paired Speech Group 内容，通过 MD5 哈希去重音频，并排除超过 15 秒的片段。然后使用 DeepSeek V3.2 [19] 为六种修改类型生成修改歌词。

**表1: LyricEditBench 中的歌词修改任务类型**

| 缩写 | 任务类型 | 描述 |
|------|---------|------|
| PSub  | 部分替换（Partial Substitution） | 替换部分词语 |
| FSub  | 全部替换（Full Substitution） | 完全重写歌词 |
| Del   | 删除（Deletion） | 删除部分词语 |
| Ins   | 插入（Insertion） | 插入部分词语 |
| Trans | 翻译（Translation） | 中英文互译 |
| Mix   | 语码混合（Code-Mixing） | 混合中英文歌词 |

给定原始歌词和修改指令，LLM 生成修订版歌词，不合规输出被丢弃，最终得到 **11,535 个有效样本**。样本按歌手性别（男/女）和语言（中文/英文）分四类，再按修改类型组织。每个组合选取每种唱法（覆盖 GTSinger 中的六种唱法）30 个样本，无唱法类别 120 个样本，最终每类每修改类型 **300 个**，总计 **7,200 个测试实例**。每个实例从剩余音频池中随机抽取不超过 15 秒的音色提示片段。

---

## 3. 实验设置

**数据集。** TTS 预训练使用 Emilia [20] 的中英文子集。歌声 SFT 使用经授权的内部音乐曲目，经 SongFormer [21] 处理以分割结构边界并标注功能类别，丢弃非人声段落。人声 stem 使用 Mel-band RoFormer [22] 分离。我们保留 2 至 30 秒之间的片段，在句子边界处切分更长的片段，最终获得 **33,562.6 小时** 的歌唱数据。

GRPO 数据集通过对 SFT 数据进行三个条件筛选构建：
1. ASR 转录验证，仅保留词错误率低于 5% 的片段
2. 通过 pyannote [23, 24] 进行说话人分离，仅保留单人片段
3. DNSMOS P808 质量评分 [25, 26] 阈值 3.5

最终获得约 **20,240 个精选片段**，中英文内容均衡。LyricEditBench 测试集严格排除在训练数据之外。

**评估指标。** 我们使用四个客观指标和两个主观指标：

| 指标 | 描述 |
|------|------|
| **PER**（↓） | 音素错误率（Phoneme Error Rate）——衡量音素级清晰度。中英文片段均由歌唱训练的 Qwen3-ASR-1.7B [27] 转录并转换为移除声调标记的音素序列。 |
| **SIM**（↑） | 说话人相似度（Speaker Similarity）——沿用 F5-TTS [15]，使用基于 WavLM-large 的验证模型提取说话人嵌入并计算余弦相似度。 |
| **F0-CORR**（↑） | F0 皮尔逊相关系数——通过 RMVPE [28] 提取生成片段和参考片段的 F0 轮廓，计算逐帧皮尔逊相关系数以衡量旋律遵循度。 |
| **VS**（↑） | 人声得分（Vocal Score）——采用 VocalVerse2 [29] 作为与人类感知偏好对齐的学习指标。 |
| **N-MOS**（↑） | 自然度平均意见分（Naturalness Mean Opinion Score）——120 个样本在任务类型和语言上均匀采样，由 30 名听者按 5 分制评分，衡量整体感知质量和自然度。 |
| **M-MOS**（↑） | 旋律平均意见分（Melody Mean Opinion Score）——衡量对参考旋律的忠实度，同样按 5 分制评分。 |

**实现细节。** 我们采用 Stable Audio 2 [13] 的 VAE（$D = 64$），SOME 的编码器作为 Melody Extractor（$D_m = 128$，temporal dropout 0.1），DiT 骨干网络沿用 F5-TTS [15]（22 层、16 注意力头、隐藏维度 1024、$D_e = 512$）。

完整系统共有 **~727.3M 参数**（453.6M CFM、156.1M VAE、117.6M Melody Extractor），在 **8× A800 80GB GPU** 上使用 DDP 和 bf16 训练。所有阶段随机掩蔽 70%–100% 的隐变量帧。

| 阶段 | 步数 | Batch 时长 | 学习率 | 备注 |
|------|------|-----------|--------|------|
| TTS Pretrain | 1M | 1.268 h | 1e-4 | 无旋律条件 |
| SFT Phase 1 | 240K | 1.69 h | 2.5e-5 | 旋律条件禁用 |
| SFT Phase 2 | 170K | 1.69 h | 2.5e-5 | 旋律条件启用；$\lambda$ 在前 2K 步从 0.3 衰减至 0.01 |
| GRPO | 1.2K | batch size 6 | 7e-6 | $G = 8$, $M = 4$ 个等权奖励模型；SDE 噪声 $a = 0.8$，窗口 $W$: $w_{\min}=1$, $w_s=8$, $\epsilon_u=0.01$, $\epsilon_l=0.002$, $\beta = 1$；无 CFG |

推理使用 **32 ODE 步**，CFG scale 为 3。

---

## 4. 实验结果

### 4.1. 主要结果

我们与 Vevo2 [9] 对比，后者是一个基于 token 的自回归模型，具有解耦的音色和旋律控制，歌声编辑时音色参考和旋律参考可使用相同或不同片段。Vevo2 是最直接的 baseline，因为其他系统运行在完全不同的范式下：上下文学习方法需要手工对齐编辑边界且仅限于局部片段，而 SoulX-Singer 依赖精确的字符级时间戳——这些时间戳要么难以获取，要么若从精选数据集中获得会带来不公平的优势。

**表2: LyricEditBench 上各任务类型和语言的 Baseline 对比。** 指标：P = PER（↓），S = SIM（↑），F = F0-CORR（↑），V = VS（↑）。最佳结果以**加粗**标示。

| 任务 | 模型 | 指标 | PSub | FSub | Del | Ins | Trans | Mix | PSub | FSub | Del | Ins | Trans | Mix |
|------|------|-----|------|------|-----|-----|-------|-----|------|------|-----|-----|-------|-----|
| | | | *中文* | | | | | | *英文* | | | | | |
| **Melody Control** | Vevo2 | P↓ | 0.1378 | 0.1462 | 0.1545 | 0.1872 | 0.4409 | 0.4757 | 0.3352 | 0.3481 | 0.3812 | 0.3135 | 0.8019 | 0.5132 |
| | | S↑ | 0.6462 | 0.6550 | 0.6457 | 0.6551 | 0.6115 | 0.6490 | 0.6357 | 0.6161 | 0.6277 | 0.6359 | 0.6237 | 0.6325 |
| | | F↑ | 0.8471 | 0.8188 | 0.8345 | 0.8552 | 0.7678 | 0.8526 | 0.8794 | 0.8409 | 0.8924 | 0.8888 | 0.8776 | 0.8927 |
| | | V↑ | 1.3578 | 1.3784 | 1.3491 | 1.1346 | 1.3061 | 1.4208 | 1.0340 | 1.1217 | 1.0476 | 0.9610 | 1.0281 | 1.0925 |
| | **Ours** | P↓ | **0.0192** | **0.0197** | **0.0458** | **0.0208** | **0.0881** | **0.1563** | **0.0685** | **0.0692** | **0.1053** | **0.0716** | **0.0413** | **0.2668** |
| | | S↑ | 0.6543 | 0.6561 | 0.6489 | 0.6552 | 0.5791 | 0.6395 | 0.6078 | 0.5914 | 0.6001 | 0.5889 | 0.5982 | 0.6076 |
| | | F↑ | **0.9364** | **0.9428** | **0.9351** | **0.9381** | **0.9378** | **0.9352** | **0.9355** | **0.9279** | **0.9309** | **0.9315** | **0.9290** | **0.9389** |
| | | V↑ | **2.0779** | **2.1419** | **2.1219** | **1.9887** | **1.9372** | **2.1002** | **1.5054** | **1.5418** | **1.6081** | **1.4036** | **1.5769** | **1.5060** |
| **Sing Edit** | Vevo2 | P↓ | 0.1290 | 0.1303 | 0.1596 | 0.1810 | 0.4111 | 0.4659 | 0.3414 | 0.3538 | 0.3531 | 0.2944 | 0.7680 | 0.4951 |
| | | S↑ | 0.7875 | 0.7729 | 0.8269 | 0.8324 | 0.7252 | 0.8015 | 0.7971 | 0.7729 | 0.8183 | 0.8378 | 0.7563 | 0.8346 |
| | | F↑ | 0.8858 | 0.8805 | 0.8969 | 0.9115 | 0.8456 | 0.9023 | 0.9258 | 0.9278 | 0.9365 | 0.9415 | 0.9137 | 0.9465 |
| | | V↑ | 1.4860 | 1.5100 | 1.5094 | 1.2935 | 1.3535 | 1.4377 | 1.0910 | 1.1110 | 1.1682 | 1.1156 | 1.1178 | 1.1453 |
| | **Ours** | P↓ | **0.0214** | **0.0186** | **0.0946** | **0.0426** | **0.1009** | **0.1903** | **0.0906** | **0.0782** | **0.1700** | **0.1070** | **0.0538** | **0.2946** |
| | | S↑ | 0.7622 | 0.7392 | 0.7874 | 0.8028 | 0.6539 | 0.7564 | 0.7398 | 0.7105 | 0.7764 | 0.7714 | 0.6918 | 0.7708 |
| | | F↑ | **0.9615** | **0.9587** | **0.9628** | **0.9642** | **0.9542** | **0.9607** | **0.9610** | **0.9563** | **0.9675** | **0.9660** | **0.9498** | **0.9668** |
| | | V↑ | **1.9761** | **2.0345** | **1.8837** | **1.7689** | **1.9371** | **1.9283** | **1.4448** | **1.4086** | **1.3820** | **1.2553** | **1.4788** | **1.3464** |

如表2所示，YingMusic-Singer-Plus 在 Melody Control 和 Sing Edit 两种设置下，全部六种修改类型的 PER、F0-CORR 和 VS 均一致优于 Vevo2，展示出对参考旋律和修改歌词的强遵循能力。清晰度差距在 Trans 和 Mix 任务上最为显著，表明在保持旋律的同时重建完全不同的音素序列本质上非常困难。注意 Mix 任务上的 PER 可能因 ASR 对语码转换话语的幻觉而进一步偏高。

Vevo2 不完整的旋律解耦倾向于降低清晰度并引入幻觉，而 YingMusic-Singer-Plus 的统一 IPA tokenization 和基于 GRPO 的歌词遵循优化即使在极端条件下也能保持鲁棒性。F0-CORR 进一步区分了两个系统：得益于 CKA 对齐和 GRPO，YingMusic-Singer-Plus 在所有任务和语言上保持持续高相关性，而 Vevo2 波动较大。

对于 SIM，Vevo2 受益于其多阶段架构，自回归 LLM 处理旋律和内容生成，专用 CFM 聚焦音色重建。YingMusic-Singer-Plus 则采用单阶段 CFM 联合建模所有因素，优先考虑架构简洁性。然而，在实际歌词编辑中，忠实呈现修改歌词同时保持原旋律结构仍然是首要关注点。

**表3: LyricEditBench 主观评测。** N-MOS 和 M-MOS 分别表示自然度和旋律遵循度。

| 任务 | 模型 | 中文 N↑ | 中文 M↑ | 英文 N↑ | 英文 M↑ |
|------|------|----------|----------|----------|----------|
| Melody Control | Vevo2 | 4.25 ± 0.06 | 4.28 ± 0.05 | 4.31 ± 0.05 | 4.31 ± 0.04 |
| | **Ours** | **4.31 ± 0.04** | **4.44 ± 0.04** | **4.36 ± 0.05** | **4.51 ± 0.03** |
| Sing Edit | Vevo2 | 4.48 ± 0.05 | 4.41 ± 0.05 | 4.44 ± 0.04 | 4.50 ± 0.04 |
| | **Ours** | **4.52 ± 0.04** | **4.55 ± 0.04** | **4.55 ± 0.04** | **4.58 ± 0.03** |

YingMusic-Singer-Plus 在两个任务和两种语言上的 N-MOS 和 M-MOS 均一致高于 Vevo2。强劲的主观得分也表明 GRPO 优化并未对奖励模型过拟合，而是泛化到人类感知。Vevo2 整体得分较低且方差更大，听者报告其输出存在可感知的伪影，如歌词呈现不忠实和旋律对齐偏移。

### 4.2. 消融实验

**表4: LyricEditBench 消融实验。** 最佳结果以**加粗**标示，次佳结果以下划线标示。

| 语言 | 变体 | Melody Control | | | | Sing Edit | | | |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | P↓ | S↑ | F↑ | V↑ | P↓ | S↑ | F↑ | V↑ |
| **中文** | TTS Pretrain | 0.41 | 0.57 | 0.01 | 0.50 | 0.37 | 0.59 | 0.06 | 0.49 |
| | SFT Phase1 | <u>0.05</u> | <u>0.68</u> | 0.03 | 1.55 | <u>0.05</u> | 0.73 | 0.31 | 1.53 |
| | SFT Phase2 | 0.08 | 0.63 | 0.92 | 1.57 | 0.11 | 0.75 | 0.95 | 1.62 |
| | -w/o CKA | 0.08 | 0.64 | 0.91 | 1.57 | 0.12 | 0.75 | 0.93 | 1.61 |
| | -w/o Dist | 0.45 | 0.63 | 0.94 | 1.42 | 0.46 | 0.79 | 0.95 | 1.55 |
| | **Full Model** | **0.06** | 0.64 | **0.94** | **2.06** | **0.08** | <u>0.75</u> | **0.96** | **1.92** |
| **英文** | TTS Pretrain | 0.46 | 0.54 | 0.01 | 0.49 | 0.43 | 0.56 | 0.00 | 0.50 |
| | SFT Phase1 | <u>0.10</u> | <u>0.65</u> | 0.03 | 1.07 | <u>0.11</u> | 0.73 | 0.40 | 1.13 |
| | SFT Phase2 | 0.14 | 0.60 | 0.92 | 1.17 | 0.19 | 0.75 | 0.96 | 1.21 |
| | -w/o CKA | 0.14 | 0.60 | 0.90 | 1.19 | 0.20 | 0.75 | 0.94 | 1.18 |
| | -w/o Dist | 0.48 | 0.58 | 0.95 | 1.00 | 0.49 | 0.78 | 0.96 | 0.98 |
| | **Full Model** | **0.10** | 0.60 | **0.93** | **1.52** | **0.13** | <u>0.74</u> | **0.96** | **1.39** |

**消融实验关键发现：**

课程学习流程在每个阶段引入了明显可分的改进：

1. **TTS Pretrain** 建立了发音先验但缺乏歌唱能力（F0-CORR 接近于零），当歌唱片段作为 ICL 提示时 PER 显著下降，表明领域适应不足。

2. **SFT Phase 1** 大幅提升所有指标，在模型适应歌唱领域时实现最低 PER，同时从 ICL 提示中自由生成旋律，绕过了显式旋律对齐的困难任务。Sing Edit 下的 F0-CORR 略有改善，说明仅从上下文中能部分捕获旋律，但忠实再现仍需显式引导。

3. **SFT Phase 2** 激活 Melody Extractor，将 F0-CORR 提升至 0.92 以上，但 PER 增加，反映出在旋律保真度和歌词忠实度之间联合优化的难度。

4. **GRPO** 通过恢复 PER 同时进一步提升 F0-CORR 和 VS（SIM 不变），解决了这一权衡问题，确认基于奖励的优化能够在所有维度上联合增强。

5. **CKA 消融（-w/o CKA）**：在 SFT Phase 2 中移除 CKA 导致 F0-CORR 降低，证实 CKA 能改善旋律遵循度。

6. **Temporal dropout 消融（-w/o Dist）**：移除对旋律隐变量 $\tilde{\mathbf{h}}$ 施加的 temporal dropout 会导致严重的清晰度退化，因为未受扰动的隐变量保留了残留的语义信息，模型可利用这些信息绕过真正的歌词生成。Temporal dropout 消除了这种信息泄漏，迫使模型依赖抽象的旋律轮廓，在保持韵律结构的同时允许自由生成修改后的歌词。

---

## 5. 结论

本文提出了 **YingMusic-Singer-Plus**，一个旋律可控的歌声编辑模型，可从音色参考、提供旋律的歌唱片段和修改后的歌词合成歌声，无需手工对齐。通过课程学习和基于 GRPO 的强化学习，YingMusic-Singer-Plus 在 LyricEditBench 上实现了卓越的旋律保持和歌词遵循度——我们为此任务推出的首个全面 benchmark，展示了端到端歌声编辑在实际应用中的强大潜力。

---

## 6. 生成式 AI 使用声明

生成式 AI 工具仅用于语言润色，在方法论、实验、结果解读或科学结论产出中不发挥任何作用。作者对本手稿中的所有内容承担全部知识产权责任。

---

## 7. 参考文献

[1] J. Liu, C. Li, Y. Ren, F. Chen, and Z. Zhao, "Diffsinger: Singing voice synthesis via shallow diffusion mechanism," in *AAAI*. AAAI Press, 2022, pp. 11,020–11,028.

[2] J. He, J. Liu, Z. Ye, R. Huang, C. Cui, H. Liu, and Z. Zhao, "Rmssinger: Realistic-music-score based singing voice synthesis," in *ACL (Findings)*, vol. ACL 2023. ACL, 2023, pp. 236–248.

[3] Y. Zhang, R. Huang, R. Li, J. He, Y. Xia, F. Chen, X. Duan, B. Huai, and Z. Zhao, "Stylesinger: Style transfer for out-of-domain singing voice synthesis," in *AAAI*. AAAI Press, 2024, pp. 19,597–19,605.

[4] F. Wang, B. Bai, Y. Deng, J. Xue, Y. Gao, and Y. Li, "Expressivesinger: Synthesizing expressive singing voice as an instrument," in *ISCSLP*. IEEE, 2024, pp. 304–308.

[5] Y. Yu, J. Shi, Y. Wu, Y. Tang, and S. Watanabe, "Visinger2+: End-to-end singing voice synthesis augmented by self-supervised learning representation," in *SLT*. IEEE, 2024, pp. 719–726.

[6] Y. Zhang, W. Guo, C. Pan, D. Yao, Z. Zhu, Z. Jiang, Y. Wang, T. Jin, and Z. Zhao, "Tcsinger 2: Customizable multilingual zero-shot singing voice synthesis," in *ACL (Findings)*, vol. ACL 2025. ACL, 2025, pp. 13,280–13,294.

[7] S. Lei, Y. Zhou, B. Tang, M. W. Y. Lam, F. Liu, H. Liu, J. Wu, S. Kang, Z. Wu, and H. Meng, "Songcreator: Lyrics-based universal song generation," in *NeurIPS*, 2024.

[8] C. Yang, S. Wang, H. Chen, J. Yu, W. Tan, R. Gu, Y. Xu, Y. Zhou, H. Zhu, and H. Li, "Songeditor: Adapting zero-shot song generation language model as a multi-task editor," in *AAAI*. AAAI Press, 2025, pp. 25,597–25,605.

[9] X. Zhang, J. Zhang, Y. Wang, C. Wang, Y. Chen, D. Jia, Z. Chen, and Z. Wu, "Vevo2: A unified and controllable framework for speech and singing voice generation," *CoRR*, vol. abs/2508.16332, 2025.

[10] J. Qian, H. Meng, T. Zheng, P. Zhu, H. Lin, Y. Dai, H. Xie, W. Cao, R. Shang, J. Wu, H. Liu, H. Wen, J. Zhao, Z. Jiang, Y. Chen, S. Yin, M. Tao, J. Wei, L. Xie, and X. Wang, "Soulxsinger: Towards high-quality zero-shot singing voice synthesis," *CoRR*, vol. abs/2602.07803, 2026.

[11] D. Guo, D. Yang, H. Zhang, J. Song et al., "Deepseek-r1 incentivizes reasoning in llms through reinforcement learning," *Nat.*, vol. 645, no. 8081, pp. 633–638, 2025.

[12] Y. Zhang, C. Pan, W. Guo, R. Li, Z. Zhu, J. Wang, W. Xu, J. Lu, Z. Hong, C. Wang, L. Zhang, J. He, Z. Jiang, Y. Chen, C. Yang, J. Zhou, X. Cheng, and Z. Zhao, "Gtsinger: A global multi-technique singing corpus with realistic music scores for all singing tasks," in *NeurIPS*, 2024.

[13] Z. Evans, J. D. Parker, C. Carr, Z. Zukowski, J. Taylor, and J. Pons, "Long-form music generation with latent diffusion," in *ISMIR*, 2024, pp. 429–437.

[14] Z. Ning, H. Chen, Y. Jiang, C. Hao, G. Ma, S. Wang, J. Yao, and L. Xie, "Diffrhythm: Blazingly fast and embarrassingly simple end-to-end full-length song generation with latent diffusion," *CoRR*, vol. abs/2503.01183, 2025.

[15] Y. Chen, Z. Niu, Z. Ma, K. Deng, C. Wang, J. Zhao, K. Yu, and X. Chen, "F5-TTS: A fairytaler that fakes fluent and faithful speech with flow matching," in *ACL (1)*. ACL, 2025, pp. 6255–6271.

[16] J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang, "Flow-grpo: Training flow matching models via online RL," *CoRR*, vol. abs/2505.05470, 2025.

[17] J. Li, Y. Cui, T. Huang, Y. Ma, C. Fan, M. Yang, and Z. Zhong, "Mixgrpo: Unlocking flow-based GRPO efficiency with mixed ODE-SDE," *CoRR*, vol. abs/2507.21802, 2025.

[18] H. Wang, B. Tian, Y. Jiang, Z. Pan, S. Zhao, B. Ma, D. Chen, and X. Li, "Flowse-grpo: Training flow matching speech enhancement via online reinforcement learning," *CoRR*, vol. abs/2601.16483, 2026.

[19] DeepSeek-AI, "Deepseek-v3.2: Pushing the frontier of open large language models," *CoRR*, vol. abs/2512.02556, 2025.

[20] H. He, Z. Shang, C. Wang, X. Li, Y. Gu, H. Hua, L. Liu, C. Yang, J. Li, P. Shi, Y. Wang, K. Chen, P. Zhang, and Z. Wu, "Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation," in *SLT*. IEEE, 2024, pp. 885–890.

[21] C. Hao, R. Yuan, J. Yao, Q. Deng, X. Bai, W. Xue, and L. Xie, "Songformer: Scaling music structure analysis with heterogeneous supervision," *CoRR*, vol. abs/2510.02797, 2025.

[22] J. Wang, W. T. Lu, and M. Won, "Mel-band roformer for music source separation," *CoRR*, vol. abs/2310.01809, 2023.

[23] A. Plaquet and H. Bredin, "Powerset multi-class cross entropy loss for neural speaker diarization," in *INTERSPEECH*. ISCA, 2023, pp. 3222–3226.

[24] H. Bredin, "pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe," in *INTERSPEECH*. ISCA, 2023, pp. 1983–1987.

[25] C. K. A. Reddy, V. Gopal, and R. Cutler, "Dnsmos: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors," in *ICASSP*. IEEE, 2021, pp. 6493–6497.

[26] ——, "Dnsmos P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors," in *ICASSP*. IEEE, 2022, pp. 886–890.

[27] X. Shi, X. Wang, Z. Guo, Y. Wang, P. Zhang, X. Zhang, Z. Guo, H. Hao, Y. Xi, B. Yang, J. Xu, J. Zhou, and J. Lin, "Qwen3-asr technical report," *CoRR*, vol. abs/2601.21337, 2026.

[28] H. Wei, X. Cao, T. Dan, and Y. Chen, "RMVPE: A robust model for vocal pitch estimation in polyphonic music," in *INTERSPEECH*. ISCA, 2023, pp. 5421–5425.

[29] Z. Wang, R. Yuan, Z. Geng, H. Li, X. Qu, X. Li, S. Chen, H. Fu, R. B. Dannenberg, and K. Zhang, "Singing timbre popularity assessment based on multimodal large foundation model," in *ACM Multimedia*. ACM, 2025, pp. 12,227–12,236.

















