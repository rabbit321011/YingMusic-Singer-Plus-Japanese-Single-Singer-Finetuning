# SVS 评测集生成与人工评分规范

## 目的与边界

本规范定义跨训练分支的 SVS 听评集如何生成、审计、登记与人工评分。它用于回答两个问题：

1. 某一批用于横向比较的音频是怎样生成的；
2. 某个模型或工程方法与哪些条件可直接比较。

本规范不替代训练中的 fixed Eval、Flow、CKA 或未来独立的生成后 F0 客观评测。后者必须另立
实验记录，不能混入人工听评表。

当前权威评测根目录：

```text
${LOCAL_EXPORT_PATH}
```

根目录的 `README.md` 面向听评，`manifest.json` 是本批模型、权重、VAE、runtime、teacher、目录和
完整性结果的机器可读记录，`评分表.csv` 是人工评分入口。每次新增模型或工程方法，三者必须同时更新。

## 评测输入集

冻结输入集为 27 组跨源 A/B：

```text
${LOCAL_EXPORT_PATH}
```

每组目录名就是评分表中的组名，包含：

| 文件 | 语义 | 是否可被模型改写 |
|---|---|---|
| `A.wav` | 参考音色/说话人条件 | 否；工程方法若处理 A，必须新建输入集并显式登记 |
| `A_T1.json` | A 区参考文本与时间 | 否 |
| `B.wav` | 目标旋律/时长条件 | 否 |
| `B_T1.json` | B 区目标文本与时间 | 否 |
| `group.json` | 原 Track、片段与导出映射 | 否 |

普通 SVS 以 A 提供音色条件、B 提供目标旋律，输出只对应 B 区。禁止交换 A/B；禁止在一个
checkpoint 的 CFG 1/3 两档间改变任一输入、T1、裁剪、声道策略或预处理。

评测集的 `manifest.json` 中 27 组名称与顺序是本批输入的冻结身份。新模型必须使用完全相同的
27 组，除非新建、命名和登记一套新的评测输入集。

## 通用生成合同

原始 SVS checkpoint 的标准网格为：

| 项 | 固定值 |
|---|---:|
| steps | 32 |
| seed | 42 |
| CFG | 3.0 和 1.0，各一批 |
| device | `cuda:0` |
| 额外音高移调 | 0 / 未启用 |
| 写出格式 | 44.1kHz、单声道 PCM WAV |

一次模型评测的最小产物为两个目录：

```text
<Model>_cfg3/
<Model>_cfg1/
```

每个目录必须有 27 个以组名命名的 WAV。H/V4PH 类还必须有 `_placement/`，其中每组一个 JSON
审计。先以 `--limit 1` 在独立 smoke 目录验证，再运行完整目录；不要拿只跑过一条的目录进入评分表。

只有下面这些变量可以在一个标准 checkpoint 条件间变化：checkpoint、它明确绑定的 VAE、CFG，及
checkpoint 自身强制要求的 teacher/runtime。其余输入、采样步数、seed、输出格式和评测组必须保持不变。

## 四类生成链路

### 普通 T1/SVS checkpoint

适用于普通 T1 文本 placement 和已有标准 MIDI/SOME 推理通路的模型。入口：

```text
${LOCAL_EXPORT_PATH}
```

命令形状：

```powershell
& <python> svs_eval_batch.py `
  --dataset <评测输入集> --output-dir <结果目录> `
  --model-id <模型ID> --checkpoint <checkpoint> --vae-ckpt <VAE> `
  --steps 32 --cfg <3或1> --seed 42 --device cuda:0 --resume
```

模型与 VAE 路径不得手写猜测，以评测根 `manifest.json` 或 checkpoint metadata 为准。

### H 与 V4Hg

H 家族不允许走普通 T1 runner。先用冻结的 SOFA/H candidates，把统一 kana、摩拉、SOFA phone
interval 和 hash-locked `render_h_pul_placements` 变成 phone/PUL 文本条件；再运行：

```text
${LOCAL_EXPORT_PATH}
${LOCAL_EXPORT_PATH}
```

完整评测可复用已经生成的 H alignment：

```text
${LOCAL_EXPORT_PATH}
```

命令形状：

```powershell
& <python> v4h_eval_batch.py `
  --dataset <评测输入集> --alignment-dir <H alignment> --output-dir <结果目录> `
  --runtime <冻结runtime> --singer-root <Singer源码> `
  --checkpoint <EMA推理权重> --vae-ckpt <绑定VAE> --midi-ckpt <SOME MIDI权重> `
  --steps 32 --cfg <3或1> --seed 42 --device cuda:0 --resume
```

`V4Hg` 仍使用 H/PUL placement，但必须使用 checkpoint metadata 绑定的 285k online VAE；不得为了
和 V4H 对齐而误用 official VAE。

### V4PH

V4PH 同样复用 H alignment，但旋律条件不是 SOME。入口：

```text
${LOCAL_EXPORT_PATH}
```

每组执行以下固定过程：

```text
A + 0.5s silence, B + 1.0s silence
  -> A/B 分别经 official VAE 编码
  -> 同一时序的 A+B 经 GAME medium, K=4
  -> 0.5 半音 P class 0..254 / REST 255 / PAD 256
  -> checkpoint 内 learned P embedding [257,128]
  -> A/prompt 区 embedding 清零，B 区作为 DiT MIDI 条件
```

GAME 固定 commit `4ad815c90dfe2442730f3fdc866fd23e737cbc97`、base seed `20260730`。输入为立体声时，
GAME 仅对每个区域做 L/R 算术平均；official VAE 保持其自身双声道前端。P class 必须由权威
cache adapter 的中心时间离散查询产生，禁止对 class ID 做数值插值。

V4PH 配置中原有 `some_pretrain_fuzzdisturb` 不能真的对 learned P embedding 执行 sigmoid 或丢帧。
runner 只借用其等长 128 维 tensor 传输接口，并将无参数 fuzz 模块置为 `Identity`；必须断言传输前后
tensor 完全相同，且不加载 SOME。

命令需要额外指定 GAME repo、依赖、medium 权重和训练时的 GAME cache manifest。任何 schema、
GAME hash、P embedding、PAD row、A 区 MIDI mask 或 A/B 边界检查失败都必须停止，不得降级到普通
MIDI/SOME 路径。

### V4IPH

V4IPH 是 V4PH 的受控单变量实验：取消 A 区参考音频，`ref_len` 恒为 0，完整时间轴均为 B
区，`cond` 全零。target singer身份完全内化在训练权重中。入口：

```text
${LOCAL_EXPORT_PATH}
```

每组执行以下固定过程：

```text
B + 1.0s silence
  -> official VAE 编码 -> ml
  -> B 经 GAME medium, K=4 -> MIDI_P (全时间轴, 无 mask)
  -> 0.5 半音 P class 0..254 / REST 255 / PAD 256
  -> checkpoint 内 learned P embedding [257,128]
  -> cond=zeros(1,1,64), lens=0 (无 prompt 区)
  -> H/PUL placement 使用 B region phrases, ref_len=0
  -> A/prompt 不存在, 全段生成
```

`A.wav` 不进入模型，仅保留作离线 CAM++/人工音色评分参考。GAME 仅编码 B waveform，不再拼接 A。
MIDI 不做 A 区 mask（因为 A 区不存在）。输出取全段去尾 1.0s 静音。

V4IPH checkpoint schema 为 `v4iph_training_checkpoint_v1`，metadata key 为 `v4iph_training`。
独立 `placement_v4iph.py` 允许 `ref_len=0`，签名与 V4PH `render_h_pul_placements` 一致，返回
结构兼容。`train_v4iph.py`、`v4iph_contract.py` 和 `placement_v4iph.py` 的 SHA256: redacted
checkpoint metadata 中记录的值一致。

命令形状：

```powershell
& <python> v4iph_eval_batch.py `
  --dataset <评测输入集> --alignment-dir <H alignment> --output-dir <结果目录> `
  --runtime <V4IH runtime> --singer-root <Singer源码> `
  --checkpoint <EMA推理权重> --vae-ckpt <绑定VAE> `
  --game-repo <GAME源码> --game-deps <GAME依赖> --game-model <GAME权重> --game-cache-manifest <GAME cache manifest> `
  --steps 32 --cfg <3或1> --seed 42 --device cuda:0 --resume
```

V4IPH 与 V4PH 共用相同的 27 组 B.wav / B_T1.json / H alignment / GAME cache / 生成参数，
唯一变量是 A 区存在与否。两组输出可直接横向对比。"音色相似度（vs A）"维度不再适用，改为
"target singer身份保持"：判断输出是否仍像target singer本人，而非跟踪每组 A。

完整性门禁额外要求：`refFrames=0`、`targetPadFrames=0`、`condPolicy=all_zero_full_timeline`。
其余门禁与 V4PH 一致。
### 工程方法

工程方法不产生新的 checkpoint 条件，而是在已生成的标准条件上只引入一个命名清楚的变量。

| 方法 | 正确做法 |
|---|---|
| A 区 MSST 去混响 | 新建输入集，只替换 A；保持 B、T1、采样率与帧数不变；重新运行对应 SVS 条件 |
| SVC cosine3000 | 直接复用已生成的 SVS WAV，不重新运行 SVS；每组 CAM++ prompt 与音色向量取同组原始 `A.wav` |

SVC 批处理入口：

```text
${LOCAL_EXPORT_PATH}
```

当前 cosine3000 条件为 30 diffusion steps、inference CFG 0.7、F0 condition、FP16 和自动音高策略。
每条 `_svc_audit/` 必须记录 SVS source、A、SVC checkpoint、config 与输出 SHA256: redacted

## 完整性门禁

完整生成后，未通过下列全部项目不得登记到 `manifest.json`、README 或评分表：

1. 每个条件恰有 27 WAV；H/V4PH 恰有 27 对同名审计 JSON。
2. 每条 WAV 可解码，44.1kHz、单声道、有限、非静音。
3. 输出相对同组 B 的时长偏差不超过 25ms；当前评测集实测最大为 21.406ms。
4. 同一 checkpoint 的 CFG 1 与 CFG 3 不得有逐字节相同 WAV。
5. H/Hg 的两档 `denseTextSHA256: redacted`、phone/PUL/exact 统计必须逐组一致；full set 当前为
   141 phone / 38 PUL / 0 exact。
6. V4PH 额外要求两档 GAME seed、拼接波形、P class、P embedding、note/REST 逐组一致；PAD frame=0，
   A 区 MIDI 非零=0，GAME/VAE A-B 边界差不超过 1 latent frame。
7. 每条审计中的 checkpoint、VAE、teacher、steps、CFG、seed 与实际条件匹配；所有绑定 hash 通过。

审计只说明输入和生成链路正确，不说明模型质量更好。

## 人工评分

评分表一行对应一个组名、模型、CFG 与工程方法。对每条输出按 1--10 分记录：

| 维度 | 观察重点 |
|---|---|
| 音色相似度 | 是否保留 A 的目标音色与质感 |
| 旋律准确度 | 音高、音符走向、节奏与 B 的一致性 |
| 咬字清晰度 | 字词可辨识度及其是否落在正确时间位置 |
| 自然度 | 连贯性、爆音、断裂、噪声和合成感 |
| 风格适配 | 力度、气声、情绪和唱法是否匹配 A/B 条件 |
| 总体偏好 | 在上述条件下是否愿意选择该结果 |

`1--2` 表示严重失败，`3--4` 表示明显缺陷，`5` 表示可听但普通，`6--7` 表示稳定可用，`8--9`
表示明显优秀，`10` 只用于当前批中几乎无可感知缺陷的结果。同一组优先横向比较；主张某分支优于
另一分支时，必须说明是否同时改变了 VAE、训练步数、teacher 或工程方法。

## 登记与扩展

每新增一个标准 checkpoint：

1. 在相同输入集下生成 cfg3/cfg1 两个目录，共 54 WAV；
2. 完成完整性门禁；
3. 在根 `manifest.json` 增加 checkpoint、EMA、VAE、runtime、teacher 和目录；
4. 更新根 README 的模型表、目录表和总计；
5. 在 `评分表.csv` 追加 27 组乘 2 档共 54 行；
6. 在所属分支实验记录中写入 provenance、生成条件、审计和结果路径。

每新增一个工程方法：保留来源 checkpoint/CFG，新增一个独立目录和 27 行评分表；在 `工程方法` 字段
写明唯一变量，不能把它伪装成新的 checkpoint。

截至 2026-08-01，当前根目录包含 10 个 checkpoint 的 20 个 CFG 条件、2 个工程方法，共 22 个
结果目录和 594 条 WAV。这个数量是当前快照，不是未来扩展时的固定目标。

















