> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# YingMusic-Singer-Plus 日语歌声项目交接文档

> 重建日期：2026-07-16
>
> 原 handoff 文件丢失。本文件依据当前会话上下文、现存实验文档、数据产物与服务器状态重建。标为“上下文恢复”的历史结论来自本轮长期会话；标为“文件核验”的内容可由现存文件直接验证。

## 1. 项目目标与当前小图景

项目目标：

```text
花丸晴琉的微妙、温柔、美丽音色
+ 任意旋律
+ 任意日语歌词
+ 实际可用的自然度与稳定性
```

必须长期分开观察三个轴：

1. 文本/咬字正确性；
2. 旋律跟随；
3. 美学、自然度及花丸细腻音色保持。

当前文本对齐主线已经基本固化：

```text
训练：SOFA 日语句级对齐
推理：T1 句首时间戳
```

当前主要研究线已转到 VAE：原 VAE 重建会产生糊感、浴室感，并损失花丸细腻音色。Decoder-only 训练已证明 VAE 是问题来源之一；当前正在训练 Encoder+Decoder 全解冻的新 VAE 代际。

## 2. 用户工作偏好与硬约束

- 说人话，先对齐概念再工程执行；不要抽象指标替代实际问题。
- 优先复用官方/社区成熟路线，不自行造轮子。
- 官方 vendor 不修改；派生 launcher、配置和分析脚本放独立实验目录。
- 大任务、新实验使用新文件夹。
- 长任务可用 tmux；短任务不必都挂 tmux。
- SSH/SCP 很慢；大文件优先阿里云盘。服务器下载优先官方源，其次国内镜像，最后本机经云盘传输。
- 服务器无法稳定访问国外网络。
- Gold timestamps 只用于评测，不能泄露给训练/推理对齐流程。
- `${LOCAL_PATH}` 是特殊目录，不应作为普通实验输出目录。
- 结论必须区分：肯定结论、疑似结论、猜想。

## 3. 数据对齐路线

### 3.1 句级评测标准（上下文恢复）

用户确认过的句级类别：

- **命中**：句首接近本句金标句首，句尾未越过下一句句首；“接近”采用句首误差不超过 1.0 秒。
- **前串**：句首明显落入上一句区域，风险是引入上一句音频咬字。
- **前缩**：句首晚于金标句首 0.5 秒以上，风险是卡掉本句音频咬字。
- **后串**：句尾越过下一句句首，风险是引入下一句咬字。
- **整句错槽**：整句基本放错位置。
- **缺失**：目标句没有生成可评估的匹配区间。

后缩曾计划通过 faster-whisper large-v3 反转录检测：看切出区间的识别文本是否完整包含目标文本。

### 3.2 MFA / A7 结论（上下文恢复）

- Gold 的人工句首不能作为 MFA 输入边界，否则实验泄露且不符合实际推理条件。
- MFA 理想假设是“音频咬字与参考文本一致”；大窗口中存在额外咬字、缺字、跨句时，强制对齐能力受限。
- A7 扩窗思路：句首向前、句尾向后，以减少前缩/后缩；前串和后串若后续对齐器能挽救，风险相对较小。
- large-v3 直接反转录 100 条金标不能作为对齐成功的充分验证；ASR 文本一致不等于时间边界正确。

### 3.3 SOFA 社区路线

最终采用并复刻了社区实际路线：

```text
SOFA
+ Greenleaf2001 JPN_Test2_Plus
+ Voicebank2DiffSinger 风格的 PyOpenJTalk G2P
```

关键社区事实（上下文恢复）：

- `JPN_Test2_Plus` 是当前 Japanese-extension 系较主流的日语 SOFA 模型。
- Voicebank2DiffSinger 使用 `pyopenjtalk.g2p()` 产生音素，并在词间插入 SP；不是依赖不可信的简陋日语转罗马音脚本。
- 禁止使用项目中那个“不参考上下文、君不会读成 kimi”的简陋日语转罗马音方案。
- TextGrid2oto 只是 SOFA 后处理/UTAU oto.ini 工具，不是核心对齐器。

### 3.4 SOFA 全量产物（文件核验）

位置：

```text
${LOCAL_PATH}
```

原始对齐统计：

```text
输入：15429
成功：15414
失败：15
重复路径：0
耗时：837.35 秒
总音频：332856.62 秒
速度：约18.4 files/s
Gold timestamps used：false
```

核心文件：

```text
timeset_SOFA.json
timeset_SOFA_failures.json
train_text.json
train_kana.json
train_tokens.json
test_text.json
test_kana.json
test_tokens.json
```

Kana 直接保留自已有上下文训推管线的输入结果，没有重新硬转。Token 使用项目自定义 `japanese_to_ipa -> vocab token + 1` 管线。

Token overflow 过滤统计：

```text
train：15107 text/kana，14661 token保留，446条音频被过滤
 test：  307 text/kana，  296 token保留， 11条音频被过滤
```

## 4. Singer / DiT 已知现象与待验证路线

用户对早期新旧 Singer 模型的人工观察（上下文恢复）：

### 肯定结论

- 新模型 24k 的旋律跟随弱于旧模型 24k。
- 新模型 24k 的前一句文本跟随强于旧模型。
- 两者变化幅度都不大。
- 新旧模型都存在特定“旋律音频 + text”组合突然崩坏的抽风现象，长音频更常见；旋律小崩、咬字大崩。

### 疑似结论

- 短音频中新模型疑似明显优于旧模型，但美学仍未达到可用。

### 猜想路线

- 推理侧是否应引入句级或字/词级时间戳。
- B 区使用目标音频经 YM-SVC 后作为 condition。
- B 区随机暴露变声后的 condition，以增强鲁棒性。
- A/B 区加入字级时间戳。
- A/B 同源与异源的消融。
- DiT 分层消融：层置零/噪声替代、官方 base 层替换，并比较日语训练后不同层的权重变化。

这些 Singer 路线目前不是正在执行的主线；当前主线是先解决 VAE 表示和重建上限。

## 5. 官方 VAE 身份与架构

### 5.1 官方身份（文件核验）

YMSP VAE 对应的 Stable Audio Tools 官方源码最接近：

```text
commit: 239a0d8477db5477df5f046965bf2f25985510d6
版本时期：0.0.16 之后、0.0.17 之前
```

本机官方 worktree：

```text
${LOCAL_PATH}
```

服务器官方 worktree：

```text
${PRIVATE_PATH}
```

生产 VAE checkpoint：

```text
本机：${LOCAL_PATH}
服务器：${PRIVATE_PATH}
SHA-256：SHA256_REDACTED
```

官方 strict load 已成功，vendor 保持干净。

### 5.2 架构限制

```text
采样率：44.1kHz
声道：stereo
总下采样率：2048
latent帧率：44100 / 2048 ≈ 21.53Hz
每帧跨度：约46.4ms
latent宽度：64维
参数量：约156M（Encoder约78M + Decoder约78M）
```

VAE 的任务不是原样存储音频，而是在约 `21.53帧/秒 × 64维` 的瓶颈里同时压缩：音高、谐波、音色、咬字、气息、空间感、立体声关系、噪声与录音特征。

Decoder-only 的硬边界：Encoder 未写入 latent 的信息无法被准确恢复，只能由 Decoder 猜测。全 VAE 解冻后，帧率和宽度仍不变，但 Encoder 可以重新决定有限容量优先保存什么。

## 6. 官方 VAE Loss 图景

当前官方生成器目标：

```text
多尺度 STFT 重建
+ Feature Matching
+ GAN adversarial
+ KL
```

关键配置：

- FFT 尺度：`2048, 1024, 512, 256, 128, 64, 32`
- `perceptual_weighting=true`，实际为 A-weighting。
- stereo 下包含 sum/difference 与左右声道频谱；官方模块列表把同名 sum/difference `mrstft_loss` 加入两次。
- feature matching 权重：5.0。
- adversarial 权重：0.1。
- KL 权重：0.0001。
- time-domain L1 权重：0。
- phase loss：0。
- 没有独立的 4–20kHz / 8–20kHz 高频保真项。

之前的梯度归因结论：

- 不能简单把高频损失归罪于 GAN；GAN 与高频诊断梯度关系弱且不稳定。
- Feature matching 与高频略同向，但同样很弱。
- 官方频谱目标能改善整体频谱，却没有强力保护高频细节。
- 更符合证据的结构原因：多尺度平均 + A-weighting + 无独立高频保真项，使模型可通过改善总体频谱而少量牺牲最高频。

归因报告：

```text
${LOCAL_PATH}
```

## 7. Decoder-only 30k 实验

### 7.1 训练边界与结果

```text
Encoder：冻结，77,989,888参数
Decoder：训练，78,122,626参数
Encodec判别器：训练，1,883,530参数
EMA：开启
单卡RTX 4090
batch 1
16-mixed
sample_size 24576（0.557秒）
30k global step
约55分钟
```

正式 checkpoint：

```text
${PRIVATE_PATH}
SHA-256：SHA256_REDACTED
```

审计确认：Encoder 未变，Decoder 已变，strict reload 成功。

### 7.2 听感结论

三个 held-out 熟悉样本中：

- 原 Decoder 普遍更糊，部分样本浴室感强。
- online/EMA 普遍更清晰，浴室感减轻。
- 两者仍有残余糊感。
- sample04 暴露疑似高频细节损失。
- online 与 EMA 很接近；只在个别样本弱偏好 EMA。

因此：Decoder-only 是明确正结果，证明 VAE Decoder 是此前糊感/浴室感的重要来源之一；但它没有彻底解决高频和细腻音色问题。

详细审计：

```text
${LOCAL_PATH}
```

## 8. 全 VAE 300k 当前运行

### 8.1 训练边界

从官方生产 base checkpoint 重新开始，不继承 Decoder-only 30k：

```text
Encoder：解冻
Decoder：训练
判别器：官方交替训练
官方loss：不改
train-only split
单卡GPU0
batch 1
16-mixed
300000 global step
每15000 step保存
EMA开启
```

两个必须同时设置的解冻开关：

```json
"model.encoder.requires_grad": true,
"training.encoder_freeze_on_warmup": false
```

官方 wrapper 每个偶数 global step 更新 VAE，每个奇数 global step 更新判别器，因此 300k 约等于 150k 次 Encoder+Decoder 更新与 150k 次判别器更新。

### 8.2 当前状态（2026-07-16 本文重建时）

```text
tmux：vae_full_300k
当前：273000 / 300000
状态：running
峰值allocated：约5.54GiB
峰值reserved：约5.88GiB
已保存到：step-step=270000.ckpt
```

服务器目录：

```text
${PRIVATE_PATH}
```

查看：

```bash
tmux attach -t vae_full_300k

tail -f ${PRIVATE_PATH}
```

第 100 step 审计：

```json
{
  "global_step": 100,
  "encoder_changed": true,
  "decoder_changed": true
}
```

重要风险：这是新的 latent 表示代际，不假定与现有 Singer/DiT 兼容。若最终采用，需考虑用新 Encoder 重编码训练集并重训/大幅微调 Singer。

运行文档：

```text
${LOCAL_PATH}
```

## 9. 中期 3.3：100 样本客观比较

### 9.1 比较边界

在 full-VAE 训练约 193100 step 时，使用当时的 `last.ckpt` 做了 100 条 held-out test 的批量客观比较。

模型：

```text
original_vae
decoder30k_ema
fullvae_current_online
fullvae_current_ema
```

不是只算胜率。先用 original VAE 的误差估计每条样本难度，再计算：

- 单样本综合分；
- 普通平均分；
- 中位数；
- 难度加权分；
- 最差10%均值；
- 高难样本均值。

评分综合整体 FFT 误差改善、4–8k / 8–12k / 12–20k 误差改善，并轻罚高频能量偏差。

### 9.2 聚合结果

```text
decoder30k_ema:
mean_score        =  0.000468
median_score      =  0.008022
weighted_score    =  0.005749
worst_10pct_mean  = -0.083652
hard_subset_mean  =  0.015638

fullvae_current_online:
mean_score        = -0.006018
median_score      =  0.004382
weighted_score    =  0.002838
worst_10pct_mean  = -0.107006
hard_subset_mean  =  0.018422

fullvae_current_ema:
mean_score        = -0.049786
median_score      = -0.040097
weighted_score    = -0.039127
worst_10pct_mean  = -0.177464
hard_subset_mean  = -0.022616
```

### 9.3 正确解释

- `decoder30k_ema` 仍是当前总体最稳方案：均值、难度加权分和尾部风险最好。
- `fullvae_current_online` 在难样本上已强于 `decoder30k_ema`，这是 full-VAE 路线最重要的正信号。
- `fullvae_current_ema` 不是所有误差都差；它主要被 8–12kHz 能量偏差过大拖分，当前不成熟。
- 不能宣布 193100-step full-VAE 已全面超过 Decoder-only。
- 结果支持继续完成 300k，并在最终 checkpoint 上重跑同一套 3.3。

详细报告：

```text
${LOCAL_PATH}
${LOCAL_PATH}
```

服务器原始报告：

```text
${PRIVATE_PATH}
```

## 10. 当前下一步

### P0：等待 full-VAE 300k 完成

完成后检查：

1. `final_step_300000.ckpt` 是否存在；
2. exit code 是否为 0；
3. strict reload 是否成功；
4. checkpoint 内 Encoder/Decoder 是否都改变；
5. online/EMA 状态是否完整；
6. loss 曲线最后阶段是否有限、无异常漂移。

### P1：在 final 300k 上重跑 100 样本 3.3

保持与 193100-step 中期评测相同的：

- 100 条固定 test manifest；
- 四路模型；
- 难度与评分公式；
- 输出指标。

这样才能判断：

- online 是否进一步稳定并整体超过 Decoder-only；
- EMA 的 8–12k 能量问题是否消失；
- 继续训练到 300k 是改善还是过训。

### P2：有空再做 3.2 听感

不需要听大量 checkpoint。优先比较：

```text
decoder30k_ema
vs
fullvae_final_online
```

只有 final EMA 的高频能量稳定性通过 3.3 后，才把它纳入主要听感候选。

### P3：若采用 full-VAE，规划 Singer 迁移

全 VAE 改变 latent 语义。需要：

1. 用新 Encoder 重编码训练集；
2. 重新训练或大幅微调 Singer/DiT；
3. 分开评测文本、旋律、美学；
4. 不可直接把新 VAE 塞进旧 Singer 就宣布成功。

## 11. 关键路径索引

### 本机

```text
项目：
${LOCAL_PATH}

SOFA数据：
${LOCAL_PATH}

VAE总审计：
${LOCAL_PATH}

全VAE运行：
${LOCAL_PATH}

100样本结果：
${LOCAL_PATH}

评分设计：
${LOCAL_PATH}

Loss归因：
${LOCAL_PATH}
```

### 服务器

```text
SSH：USER@IP_REDACTED

官方VAE实验根：
${PRIVATE_PATH}

Decoder-only 30k：
${PRIVATE_PATH}

Full-VAE 300k：
${PRIVATE_PATH}

100样本3.3：
${PRIVATE_PATH}
```

## 12. 文档丢失说明

本次检查发现原 handoff 文件不存在，同时原 `docs` 目录中的 V5、VOCAB token map、论文英文 markdown 等若干历史文档也不在原路径。当前仍存在：

```text
${LOCAL_PATH}
${LOCAL_PATH}
${LOCAL_PATH}
```

因此：

- 本 handoff 可作为当前工作的主恢复入口；
- VAE 结论大部分有现存实验文件直接支撑；
- A7/MFA/SOFA 的部分历史讨论由会话上下文恢复；
- 若后续从备份找回 V5/VOCAB 文档，应与本 handoff 交叉核对，而不是直接覆盖当前已验证结果。
