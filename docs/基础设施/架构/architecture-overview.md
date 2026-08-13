
好的，我已经仔细阅读了两篇论文和全部核心源码。下面从零开始给你梳理。

# YingMusic-Singer 架构详解
## 一、先理解任务本质
你给它三样东西：

输入 举例 作用 音色参考音频 target singer唱的一段干声 "我要这个人的声音" 旋律参考 另一首歌的干声 / MIDI / 哼唱 "我要按这个旋律唱" 目标歌词 "たぶん私じゃなくていいね" "我要唱这句词"

输出：一段 48kHz 的歌声波形 ，音色像target singer、旋律跟着参考走、咬字是目标歌词。

关键挑战： 不需要人工标注音素时长、不需要 MIDI 钢琴卷 。旋律直接从音频里提取出来，歌词和旋律之间靠 DiT 自己隐式对齐。

## 二、整体架构 — 五条"高速公路"在一张网里交汇
## 三、逐模块拆解
### 3.1 VAE — 音频的"压缩解压器"
任何音频处理系统都要面对一个现实：48kHz 的原始音频每秒 48000 个采样点，维度太高了，直接让神经网络处理会炸。

VAE 就是解决这个问题的 。它像一个 zip 压缩软件：

- 编码（压缩） ：输入 48kHz 波形 → 输出 25Hz 的 64 维 latent 向量。压缩比 = 48000/25 = 1920 倍
- 解码（解压） ：输入 latent 向量 → 还原为 48kHz 波形
你可以理解成：VAE 把"声音的 DNA"编码成了 64 个数字，DiT 只需要在这 64 个数字的空间里工作，而不是在 48000 个采样点里折腾。

代码证据 autoencoders.py ：使用的是 Stability AI 的 stable_audio_1920_vae 架构，基于 BigVGAN 的 SnakeBeta 激活函数和残差卷积。

### 3.2 RMVPE + SOME — 从音频里"听"出旋律
这个链条分两步：

Step 1 — RMVPE（F0 提取） ：F0 是"基频"，也就是每时刻声音的最低频分量。对于唱歌，F0 ≈ 音高。RMVPE 是一个专门的深度学习模型，输入 16kHz 音频，输出每帧的 F0 值（Hz）。

Step 2 — SOME（旋律提取） ：把 F0 和 Mel 频谱（80维，类似"声音的色谱图"）一起喂给一个 8 层 CNN + Conformer 注意力网络，输出 128 维的旋律表示——从 0 到 127 的 MIDI 音高序列（只要有人声的部分）。

代码证据 midi_extractor.py ： MIDIExtractor 类，用 Gmidi_conform 做 Conformer 编码，输出通过高斯模糊解码得到连续 MIDI 值。
 训练 SOME 用的是 知识蒸馏 ：有一个更大的 teacher 模型（SOME 预训练权重 471MB，名为 some.pt ），SOME 作为 teacher 先训练好，然后冻结，在 DiT 训练时 teacher 的预测值作为 GT。
### 3.3 G2P + PhonemeBPE — 文字变成数字
这是"文本怎么进入模型"的关键。这条路径是 YingMusic-Singer 和 Vevo2 最核心的区别 。

G2P = Grapheme to Phoneme ，把文字（字母/假名/汉字）转成"怎么发音"的音素序列。日语用 pyopenjtalk （一个基于 OpenJTalk 的日语 TTS 前端），中文用 pypinyin ，英文用 phonemizer 。

为什么 G2P 天然跨语言？ 因为音素是一个 有限的、语言无关的符号集 。日语的 "a"、中文的 "a"、英文的 "a"，在音素层面是同一个东西。BPE tokenizer（Vevo2 用的）则把 "たぶん" 可能编码成完全不同的 token ID，和 "probably" 毫无关联。

代码证据 g2p/ init .py ： PhonemeBpeTokenizer 类，支持 zh/ja/en/ko/fr/de 六种语言，通过 LangSegment 自动检测语言（但日语被错误路由到 zh——需要显式指定 language="ja" ）。

### 3.4 DiT — 核心生成器（最复杂也最关键） 3.4.1 什么是"扩散模型"？
打个比方：你想雕刻一个大理石像。一种做法是直接刻。扩散模型的做法是：先拿一块完美的大理石（GT音频的 latent），往上面糊 100 层泥巴（加噪声），再让 AI 学"怎么把泥巴一层层剥掉"。

实际上，YingMusic-Singer 用的是 Flow Matching ，比传统扩散更直接：

- 传统扩散： x_t = √(1-t) * noise + √t * clean ，学的是"score"（指向干净数据的方向梯度）
- Flow Matching： x_t = (1-t) * noise + t * clean ，学的是 速度场 v = clean - noise
速度场的含义：从噪声到干净数据，每一步该"走多快、往哪走"。好处是：路径是 直线 ，没有曲率，收敛更快。
 3.4.2 DiT 是什么架构？
DiT = Diffusion Transformer ，是 YingMusic-Singer 的心脏。它是基于 Wan2.1 （阿里巴巴的视频生成模型 Wan 的 Transformer 骨干）修改的。

24 层，每层 1024 维，16 个注意力头，FFN 膨胀 4 倍：

这里的核心技巧是 AdaLN（自适应层归一化） ：时间步 t 不是直接拼进 x 里，而是通过一个小网络变成 6 组参数（每层都有），这 6 个参数像"音量旋钮"一样，控制这一层注意力和 FFN 的"强度"。

- t=0（噪声状态）→ 旋钮拧到最大，模型要尽力"想象"声音该长什么样
- t=1（接近干净）→ 旋钮拧小，模型只需要微调细节
代码证据 dit.py ： DiT 类， self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6)) 把时间步变成 6 组调制参数， WanAttentionBlock 的 self.modulation 是一个 [1, 6, dim] 的可学习 baseline。
 3.4.3 五路条件是怎么拼在一起的？
看 InputEmbedding 的 forward：

五个信息源在 每一帧 上都拼接在一起，然后投影到 1024 维统一空间。DiT 的 24 层自注意力会自己学会：哪一帧的文本对应哪一帧的旋律，哪一帧用哪种音色。
 3.4.4 推理时的采样过程（怎么从噪声变歌声）
**CFG（Classifier-Free Guidance）**是一个"放大镜"技巧：

- 模型被训练成两种模式：有文本条件、无文本条件
- 推理时， v = v_cond + (v_cond - v_uncond) × 4.0
- 这相当于：沿着"有文本"方向，再额外推 4 倍远离"无文本"方向
- 效果：咬字清晰度大幅提升（但也有可能过度强调导致失真）
## 四、训练原理
训练时的数据是 自给自足 的：

每步训练：

1. 随机采样 t ∈ (0, 1)（从 LogNormal 分布）
2. 构造： x_t = (1-t) × noise + t × x₁ （线性插值，这就是 Flow Matching）
3. 构造目标： v_target = x₁ - noise
4. DiT 预测： v_pred = DiT(x_t, cond, text, melody, "singing", t)
5. Loss: MSE(v_pred, v_target)
6. 15% 概率随机丢弃 text（训练 CFG 的无条件分支）
训练参数：AdamW β=(0.9,0.98)，lr=1e-4→1e-6 cosine 衰减，warmup 3000步，梯度裁剪 10.0

## 五、Vevo2 的架构（对比理解）
Vevo2 的思路完全不一样——它是 两级级联 ：

### 5.1 两个 Audio Tokenizer（最核心的创新）
Vevo2 设计了两套 VQ-VAE（矢量量化变分自编码器），把音频压缩成 离散的整数序列 ——就像把音频变成"单词"：

Content-Style Tokenizer Prosody Tokenizer 词表大小 16384 512 帧率 12.5 Hz 6.25 Hz 编码了什么 语言学内容 + 说话风格 + 口腔动作 旋律轮廓 + 节奏 输入 Whisper 特征(1024维) + Chromagram(24维) Chromagram(24维)

理解关键：CS Tokenizer 的 16384 个 code 不像"音素 A/B/C"，而是更抽象的"口腔动作模式"。同一个 code 可能对应不同语言的类似发音方式。

### 5.2 AR 阶段：文字 → CS 码
这就是你遇到的致命问题的根源 ：Qwen2.5 用的是 BPE tokenizer，日文被切成 "た"/"ぶん"/"私"/"じゃ" 这样的子词片段，每个片段有完全不同的 token ID。posttraining 的训练数据以英文为主，日文 BPE token → CS token 的映射从未被充分训练过。

所以你的模型退化了 ：不要文本条件也能预测 CS 码（因为 CS 序列本身有很强的统计规律），加了日文文本反而变成噪声——空文本 loss 比正确文本还低。

### 5.3 FM 阶段：CS 码 → Mel 频谱
FM 是 16 层 Transformer，1024 维，16 头。它学的是"给定 CS 码描述的口腔动作 + 文本 + 音色，实际的声音频谱长什么样"。

## 六、两大架构的核心对比
YingMusic-Singer Vevo2 核心生成器 DiT 24层（端到端扩散） AR(FM(370M)) 两级级联 文本路径 G2P→音素→366词表 Qwen BPE→168k词表 跨语言原理 音素是通用符号集 BPE token 语言相关 中间表示 连续 64维 latent 离散 16384 词表 CS码 速度场/流 直线 Flow Matching (32步) FM也是32步 音色控制 VAE latent 的前半段 独立的说话人 embedding 旋律控制 SOME→128维旋律 embedding 离散 512 词表 Prosody码 训练数据 仅歌声，自监督（每条自给自足） 101k小时语音 + 7k小时歌声 参数量 ~329M（DiT 可训练）+ 冻结模块 AR 500M + FM 370M 推理速度 32步 × 2(CFG) = 64次DiT前向 AR自回归 + 32步FM

## 七、为什么 Vevo2 失败了，但 YingMusic-Singer 值得一试
Vevo2 的失败不是"训练不好"，而是 BPE tokenizer 的跨语言断裂 ：

- BPE 把 "たぶん" 切成 ["た", "ぶん"] ，把 "probably" 切成 ["prob", "ably"]
- 这两个序列在 token ID 空间里毫无关系
- 训练数据里日文 BPE→CS 码的样本太少（Emilia 101k小时里日文占比极低）
- AR 模型学会了"背 CS 序列"，没学会"理解文本"
YingMusic-Singer 的 G2P 路径则不同：

- "たぶん"→音素 t a b u N ，"probably"→音素 p r aa b ax b l iy
- 虽然音素不同，但都是 同一个 366 词表里的东西
- DiT 已经学会了"音素序列 → 声音"的映射（在中英文上）
- 日语只是词表里的不同的音素组合，不需要重新学映射规则
打个比方 ：Vevo2 的问题像是"一本书翻译软件，但日语那章完全没装词典"。YingMusic-Singer 则像"一个学会了读字母表的人，现在要学读日语——字母表已经会了，只需要学新的拼读规则"。

还有什么模块想深入展开的吗？
















