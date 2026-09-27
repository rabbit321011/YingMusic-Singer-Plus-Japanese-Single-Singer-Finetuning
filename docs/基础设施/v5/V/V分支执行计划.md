> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# V 分支执行计划

状态：阶段一正式 300k 已完成，最终 checkpoint、raw/EMA demo 和困难样本试听包均已生成。V 已证明能改善旧 decoder 的质感，但当前版本仍有柔声清晰度下降、空洞/风声等副作用，尚未达到无条件直接上线边界。V5PgOV 已确定为 V5 世代在当前资源约束下的最佳最终候选；更大规模数据、MOE 和 2--3B 模型留给后续世代。下列设定均为用户已确认，不是提案。

## 一、这件事要干什么

拿 DiT 真正吐出来的 latent（`z_gen`），配同一条花丸原唱 B 段的真实波形（`x_B`），重新训练解码器，让它学会"看到生成器给的东西，就吐出该有的花丸声音"。

目标：直接改善最终生成对花丸原音的拟合——质感、自然度、局部发声细节。不是给输出加颤音，也不是复制别的歌手的唱腔。

定位：**不需要先证明解码器是错误来源**。即使偏差来自 DiT，也先看解码器能不能补回来。

产物的命名是 `V5PgOV`。

一条硬纪律：训练输入必须是 DiT 真正输出、尚未解码的 latent。`E(D(z_gen))` 不是 `z_gen`，不能把生成 WAV 重新编码后冒充。

## 二、数据是怎么来的

三步，只有第一步和第三步每轮重跑：

```
A  →  A latent                          每轮重跑，seed 随机
B  →  midi_p + text                     已经算过，直接复用缓存
A latent + midi_p + text  →  B latent   每轮重跑，seed 随机
```

一轮之内，每条记录只有一份 A latent 和一份 `z_gen`。轮内不同 step 取的是同一份缓存里的不同 32 帧窗口。

## 三、已冻结的设定

### 基座与产物

| 项 | 值 |
|---|---|
| 基座 | V5PgO 8k，`${PRIVATE_PATH}`，SHA256 `SHA256_REDACTED` |
| VAE | 285k **online**，`${PRIVATE_PATH}`，SHA256 `SHA256_REDACTED` |
| 解码器起点 | 上述 285k online VAE 的解码器 |
| encoder latent 兼容性 | 不保留 |
| 产物名 | V5PgOV |
| 阶段二 | 不做，等阶段一听评结束再议 |

上面两个 SHA256 在训练端与部署端是同一个值：训练脚本里的 `V5PG_VAE_SHA256`、部署端 preset `V5PgO_8K.vaeCheckpointSHA256`，以及本文都一致。训练脚本的报错文案直接称这份为 "frozen 285k online VAE"。

选取 online 而不是 EMA，是因为部署端（编辑器）实际加载的就是这一份。训练起点、对照基线、生产推理三者必须同一份权重。

### 数据

| 项 | 值 |
|---|---|
| 数据集 | 服务器冻结的 87H，根目录 `${PRIVATE_PATH}` |
| 训练文件 | `train_short.json` / `train_long.json` / `train_mixed.json` |
| 验证文件 | `eval_mixed.json` |
| GAME manifest | `${PRIVATE_PATH}` |
| 切分与时长 | 沿用 PgO（V5 系列一致） |
| 参与范围 | **只 B 段**，A 段不参与训练 |
| 裁剪长度 | **32 帧 = 65536 采样点 = 1.4859 秒** |
| 裁剪起点 | 随机，且必须落在 B 区内 |

帧率 `44100 / 2048 = 21.533203125 Hz`，一帧 2048 个采样点。

### A/B 布局：按部署端

A 与 B 的**切分规则**沿用 PgO：`T` = 全段帧数；`ref_len = T × uniform(0.125, 0.33)`，下限 5 秒（108 帧）、上限 65% T，再吸附到 ±2.5 秒内最近的乐句起点；A = `[0, ref_len)`，B = `[ref_len, T)`。

A 与 B 的**排布方式**照抄部署端（`AISVC-midi-web/server/scripts/v5p_direct_control.py` 的 frame map），不是训练端的连续排布。

常量：

```text
SAMPLE_RATE                   = 44100
HOP_SAMPLES                   = 2048
NOMINAL_REFERENCE_GAP_SAMPLES = 22050   # 0.5 秒
TARGET_REAR_SAMPLES           = 44100   # 1.0 秒
EVALUATOR_REAR_CROP_FRAMES    = 44100 // 2048 = 21 帧
```

A 段：

```text
b_start_frame            = (reference_samples + 22050 + 1024) // 2048
reference_padded_samples = b_start_frame * 2048
gap_samples              = reference_padded_samples - reference_samples
```

标称间隔是 22050 采样点（0.5 秒），实际值被取整到帧边界，在标称值上下浮动两千个采样点以内。部署报告里的 `paddingAdjustmentSampleCount = gap_samples - 22050` 记录这个偏差，可以为负。

B 段：

```text
target_padded_samples = target_samples + 44100
target_padded_frames  = target_padded_samples // 2048
total_frames          = b_start_frame + target_padded_frames
crop_end              = total_frames - 21
```

填充全部是**零填充**。

裁剪与对齐：

- `z_gen` 的 B 区 = `generated[:, b_start_frame : crop_end, :]`
- `x_B` = 原录音 B 段的前 `target_owned_frames * 2048` 个采样，其中 `target_owned_frames = target_samples // 2048`；最后不足一帧的采样丢掉
- 两者在同一帧网格上，第 n 帧对第 `n*2048 ~ (n+1)*2048` 个采样，不需要额外对齐步骤

**这一套与训练端的 `compute_ref_len` 布局不同。** 训练端 A 和 B 是同一条连续音频切开、中间不插静音；部署端 A 后面有 0.5 秒静音间隔。缓存按部署端这一套生成，两者不可混用。

**缓存生成时不要套用推理管线的 `-rear_silent_frames` 尾部裁剪**（`generated_latent[:, T_ref : -int(vae_frame_rate*rear_silent_time), :]`）。缓存要的是上面的 `crop_end`，不是那个值。

### 生成与缓存

| 项 | 值 |
|---|---|
| CFG | 三路，每次随机取：audio **0.4~0.7**、text **0.4~0.6**、midi **0.4~0.5** |
| seed | 随机，不固定 |
| A latent | 不固定，每轮重新编码，seed 随机 |
| 采样步数 | **32**（部署端默认：`synthesis-direct-control.service.ts` 里 `req.steps ?? 32`，允许 1~256） |
| `t_shift` | **0.5**（部署端 `v5p_direct_runner.py` 里写死） |
| 三路 CFG 公式 | `audio-text-midi-telescoping.v1`，与部署端一致 |
| 缓存刷新 | 已预生成 4 轮；训练时依次使用 `v_latent_01` … `v_latent_04` |

### 损失

| 项 | 权重 |
|---|---|
| 频谱 mrstft | **1.0** |
| 判别器 adversarial | **0.1** |
| 判别器 feature_matching | **5.0** |
| 时域 l1 | 0.0（不启用） |
| 瓶颈 kl | **删除** |

高频保护项不加。判别器开启。除 KL 外，损失与上次 285k 完全一致。

**实现陷阱：只删配置不能关掉 KL。** stable-audio-tools 的 `AutoencoderTrainingWrapper` 只要看见模型的 `VAEBottleneck`，就调用 `create_loss_modules_from_bottleneck` 添加 `kl_loss`；如果 `loss_configs.bottleneck` 不存在，函数会回退到 **`1e-6`**，而非 0。V 必须在专用 wrapper 中显式排除 `kl_loss`，并在启动时断言实际 `losses_gen.losses` 中没有 KL。`model.bottleneck` 仍保留，以严格加载 285k VAE 和保持解码接口一致。

### 训练

| 项 | 值 |
|---|---|
| 官方框架 | stable-audio-tools commit `239a0d8477db5477df5f046965bf2f25985510d6` |
| 优化器（编解码 / 判别器） | AdamW，betas `[0.8, 0.99]`，lr `1e-4` / `3e-4`，weight_decay `8e-4` / `1e-3` |
| 调度器 | InverseLR，`inv_gamma 200000`、`power 0.5`、`warmup 0.999` |
| 总步数 | **300,000** |
| 保存间隔 | 每 **15,000** 步；`save_top_k = -1`、`save_last = True` |
| EMA | `use_ema = true`，计数**从零开始** |
| demo | 每 15,000 步固定同一 `z_gen`，输出原唱 B、旧 decoder、新 raw、新 EMA；不使用普通 VAE 重建 demo |
| 卡 | 单卡 |
| 精度 / batch / workers | 16-mixed / 1 / 8 |
| 随机种子 | `pl.seed_everything(42, workers=True)` |

## 四、与上次 285k 配方的差异

一共四处，其余全部照抄。

| # | 项 | 上次 | 本次 |
|---|---|---|---|
| 1 | 编码器 | `requires_grad = true` ＋ `encoder_freeze_on_warmup = false` | **`requires_grad = false` ＋ `encoder_freeze_on_warmup = true`** |
| 2 | KL 损失 | 1e-4 | **删除** |
| 3 | 解码器输入 | 现场由编码器算出的 latent | **缓存的 `z_gen`** |
| 4 | `sample_size` | 24576（12 帧） | **65536（32 帧）** |

`warmup_steps = 0` 时，两个开关配合下编码器从第一步就在 `torch.no_grad()` 下运行（源码判据是 `if self.warmed_up and self.encoder_freeze_on_warmup:`）。只设第一个开关不够。

## 五、要改的代码，三处

**1. 数据集**

官方 `create_dataloader_from_config` 只有 `audio_dir` 和 `s3` / `wds` 两种类型，没有读 latent 的类型；`SampleDataset.__getitem__` 只做"找文件 → 读波形 → 随机裁 `sample_size`"，也没有地方放预先算好的 `z_gen`。所以要新写一个数据集，产出切好的 `(z_gen 切片, x_B 切片)`。

裁剪要在潜变量域取随机起点，再换算到波形域：取第 `n` 帧起 32 帧，就取 `z_gen[n : n+32]` 和 `x_B[n*2048 : (n+32)*2048]`。

**2. 训练 wrapper**

`training/autoencoders.py` 的 forward 里这一行是写死的：

```python
latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
```

V 专用 wrapper 改成直接取缓存的 `latents`，不产生 `encoder_info`，并显式排除基类自动添加的 `kl_loss`。不可只从 JSON 删除 `loss_configs.bottleneck`。

**3. 损失配置**

删掉 `loss_configs.bottleneck` 整块。

## 六、审计要反过来

上次那份启动脚本的审计会在编码器没变时报错：

```python
if not encoder_changed or not decoder_changed:
    raise RuntimeError(...)
```

V 要求编码器一动不动，所以判据要反成「编码器未变 且 解码器已变」。形状照抄 decoder-only 那次：

```json
{"encoder_unchanged": true, "decoder_changed": true}
```

照抄参数时这一条不要跟着抄，否则第一步就崩。

## 七、缓存规模与时间

- TrainPool 11,127 条可用记录（原始输入 11,203 条，76 条因 B 段不足一个 32 帧窗口而跳过）
- 每条 B 段约 430~1130 帧
- 按 32 帧一段，每条出 13~35 个训练样本
- 合计约 15 万~39 万个样本，取中位约 28 万
- 总步数 30 万步（batch 1）→ **整个训练大约就是过一遍数据**

本轮实际不是每个 epoch 现场重生成，而是预先生成 4 轮缓存。四轮的 A 后验采样与 B 采样 seed 均不同，frame map、输入记录集合和 VAE 版本保持一致；训练时按轮次切换缓存目录。

时间记录：四轮缓存已在 4 张 GPU 并行下完成，每轮约 **464~471 分钟**；训练入口在仅 20 条记录的 100-step smoke 中约 9 step/s，正式吞吐仍待全量运行实测。

缓存磁盘占用约 **4.9 GB**（四轮，每轮约 1.3 GB）。`x_B` 不复制，只记录帧偏移，从原文件读。

服务器数据盘当前 7.0T 用 5.7T，剩 934G。每个 checkpoint 粗估约 1.9 GB，20 个约 38 GB。

## 八、验证

用服务器冻结的 `eval_mixed.json`。

对照方式：对**同一条未参与训练的 `z_gen`**，比较旧解码器与新解码器的输出，以对应原音作参照。

验证样本必须**当场新生成**，不能从训练缓存里挑。seed 是随机的，训练与验证撞上的概率为零，所以不需要为 seed 记账；要守住的纪律只有上面这一条。

听评要求：**质感和内容分开记**。解码器只能重排 `z_gen` 里已有的信息，能修质感，修不了内容。如果某条轨迹本身音符或时序就错了，解码器不会把它变对，只会给它抹上一层好质感。只问"质感好不好"，会把内容退化掩盖掉——音准、咬字、时间必须另列。

边界：GAME-P 用的是原唱自己的音高缓存，比实战从其他歌手取 P/H 更理想。结论只能表述为"训练式条件下的解码器能力诊断"，不能写成"跨歌手格式合成成功"。

## 九、阶段一训练与困难样本听评结果

正式训练：

- 300,000 step 正常完成，最终 checkpoint：
  `runs/decoder_adapt_300k_20260926/checkpoints/final_step_300000.ckpt`
- 审计通过：`encoder_unchanged = true`、`decoder_changed = true`、KL
  关闭；训练期间无 CUDA OOM 或非有限 loss。
- 困难样本试听包：
  `${LOCAL_PATH}`
  每组保留原轨、VAE、旧 decoder、V raw、V EMA。

试听编号固定为：`1 = 原轨`、`2 = VAE`、`3 = 旧 decoder`、
`4 = V raw`、`5 = V EMA`。

用户反馈：

| 组 | 用户听评 |
|---|---|
| V01 相遇天使 | `2 > 1`（VAE 去掉部分录音瑕疵）；`1 > 5 > 4 > 3` |
| V02 夜明けと蛍 | `1 >> 2`，VAE 去掉的细微变化有损失；`2 > 4 >= 5 >> 3` |
| V03 旅人 | `1 = 2`；`2 >> 5 > 4 > 3`，V raw/EMA 都比旧 decoder 好，但浓厚情感仍不足 |
| V04 月兔 | `1 >> 2`，VAE 不能复刻特别温柔的轻声；`2 > 5 > 4 > 3` |
| V05 团子 | `1 > 2`；V raw/EMA 比 VAE 更温柔但牺牲清晰度；V raw 部分位置更温柔，同时出现回声/地下洞穴风声，V EMA 也有极轻微空洞风声；旧 decoder 最差 |

结论：

1. **V 分支有效触及了质感问题。** 五组中 V raw 和 V EMA 都优于
   旧 decoder，说明 `(z_gen, x_B)` decoder 适配不是无效训练。
2. **V 还不能直接上线。** 它改善了平、薄、缺少鲜活感的一部分问题，
   但在柔声和低声压区域出现了新的副作用：声音被自身压低，伴随轻微
   空洞、风声或地下空间感，并可能牺牲清晰度。
3. **VAE 不是简单的质感上限。** 它通常更干净，但会移除录音瑕疵和
   花丸演唱中的细微变化；V 的目标不应退化为单纯复制 VAE 的干净感。
4. **EMA 不总是优于 raw。** EMA 在部分组更平滑或更好，V02 raw
   略优于 EMA，V05 则是两者各有代价，不能默认只部署 EMA。
5. 当前结果支持“decoder 是缺口的一部分”，但不支持“decoder 已经是
   唯一根因”。V 的剩余问题与数据长尾、有效容量和架构分配有关，不适合
   继续用 V5 内部的小规模随机实验解释；不因 loss 仍可继续下降就盲目
   延长训练。

说明：V01--V04 的 old decoder 与历史原始生成轨逐字节一致。V05 历史
试听没有保存 latent，本次 old 是同条件重采样；它与历史 WAV 的差异
很小，但不能宣称使用了完全相同的 latent。

## 十、执行顺序

1. 冻结并核对基座与数据：PgO 8k、285k online VAE、frozen 四件套、GAME manifest 的 SHA256。**已核对完成**（`frozen/` 六个文件、PgO 8k checkpoint、`autoencoder_285k.ckpt` 均与记录相符）。
2. 写缓存生成脚本（A 重编码 ＋ B 重采样），先在 1~2 条上 smoke。**已完成。**
3. 写成对数据集；改 wrapper（跳过编码器）；删 KL。**已完成。**
4. 静态检查 ＋ 单卡几步 smoke，验四件事：编码器未变、解码器已变、loss 有限、显存记录。**100-step smoke 已通过；FP16 前几次 generator 更新因 AMP 缩放而跳步，step 100 已确认 decoder 更新。**
5. 最长样本显存门禁。**已完成：B 区最长 1002 帧，四轮缓存的 32 帧窗口与原 WAV 精确配对，边界窗口完整；单卡 32 step 覆盖四轮，显存峰值约 5.64 GiB，encoder 未变、decoder 已变、KL 关闭。**
6. 正式 300k 单卡跑。**已完成。**
7. 每 15k 保存点生成 demo。**已完成。**
8. 收尾：同一批困难样本重新生成 `z_gen`，比旧/V raw/V EMA，
   配原音和 VAE 参照。**已完成。**
9. 结论按"训练式条件"表述，并保留柔声副作用作为下一阶段诊断目标。

## 十一、本轮范围之外与世代边界

- **V5PgOV 是 V5 世代的最终候选方案**，不是无副作用的完美部署模型。
- 当前 V5PgOV 不替换现有 V5PgO 稳定默认/回滚基线；原 VAE 和旧预设保留。
- **阶段二**（encoder / DiT / decoder 联合 LoRA）不在 V5 内继续启动，
  留作后续世代设计；固定缓存只能训练 decoder，不能直接变成端到端联合训练。
- 更大规模花丸数据、MOE、2--3B 级 VocalRender 和新的容量分配架构，
  统一归入 V6 或后续高预算阶段。

完整阶段判断见 [V 结论](V结论文档.md)。
