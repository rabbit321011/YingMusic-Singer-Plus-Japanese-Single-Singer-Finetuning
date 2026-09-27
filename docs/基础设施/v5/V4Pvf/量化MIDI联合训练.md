> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# 量化 MIDI 联合训练

## 目标

以随机初始化 DiT 和随机初始化量化 MIDI embedding 从 step 0 联合训练，保持 SOFA、
A/B、官方 VAE 与原始 SOME CKA，验证量化音符能否负责乐谱控制，而模型从花丸目标音频
中学习颤音、滑音和演唱表现。

## 冻结规格

| 项 | V4Pvf 定义 |
|---|---|
| DiT 初始化 | 源码默认随机初始化；不加载任何生成模型 checkpoint |
| MIDI 输入 | SOME pitch/boundary → note pitch/duration → 展开 → 0.5 半音 class |
| Pitch class | MIDI `0..127` 映射为 `0..254` |
| REST/PAD | `REST=255`，`PAD=256` |
| NULL | A 区和 CFG 关闭 MIDI 时使用零向量，不与 REST/PAD 共用 |
| Embedding | `nn.Embedding(257, 128)`，随机初始化并与 DiT 联合训练 |
| 时间对齐 | pitch 允许线性平滑；REST/PAD mask 使用离散对齐 |
| FuzzDisturb | 关闭 |
| CFG drop_midi | 保留 V4vf 的整条 30% drop |
| CKA | 保留原始、未量化的 128 维 SOME 表示作为训练辅助目标 |
| VAE | 官方 VAE；第一轮不同时引入 285k VAE |
| 首轮 LR | `1e-4`，依据 V4vf scratch 30k 结果；后续 LR 另做实验 |

## 独占实现原则

- 以 `train_plus_v4vf.py` 为已验证 DDP 骨架复制出 V4Pvf 训练入口；
- 需要改变的公共模块沿依赖链复制为 `v4pvf` 专属版本，调用方只指向副本；
- 不修改 `train_plus_v4vf.py`、`train_plus_v4d.py`、现有 `model.py`、`midi_extractor.py` 或正式推理脚本；
- 不伪造数据、返回值、checkpoint 或 PASS；探针和推理都使用真实 SOME、真实 VAE 与真实音频；
- 遇到无法确定的语义或异常停止并交由用户裁决，不以绕过检查代替修复。

## 阶段门禁

### 阶段 0：文档与边界

- [x] 建立独立索引与工人；
- [x] 明确 V4Pvf 不是 V4vf 续训；
- [x] 明确只修改复制副本；
- [x] 未占用 GPU 或启动服务器任务。

复核：当前工作定义为 `Random DiT + P from step 0 + SOFA + official VAE + raw SOME CKA`，属于 V4Pvf。

### 阶段 1：实现与离线探针

- [x] 复制依赖链并实现量化器、模型、训练和推理入口；
- [x] 合成边界单元测试通过；
- [x] 真实音频 SOME 量化统计通过；
- [x] P embedding 进入 model、EMA 和 optimizer；checkpoint 文件落盘与恢复在阶段 2 验证；
- [x] Python/Bash 静态检查通过。

### 阶段 2：10-step smoke

- [x] 4 卡 world/effective batch 正确；
- [x] loss、grad、参数和 checkpoint 全部有限；
- [x] P embedding 确实更新；显式 all-reduce 沿用已验证 V4vf 实现并覆盖全部可训练参数；
- [x] tmux、日志、输出独占且完成后释放。

### 阶段 3：真实推理

- [x] 从 smoke checkpoint 恢复模型与 P embedding；
- [x] 使用真实 A/B 音频、真实歌词和真实 SOME；
- [x] 成功生成非空、有限、可播放音频；
- [x] 推理日志证明使用 V4Pvf 量化通路。

### 阶段 4：30k 正式训练

- [x] 再次复核分支定义；
- [x] 启动前检查 GPU/tmux/output/log；
- [x] 使用 tmux，实时输出 step、loss、GradNorm、时长和 ETA；
- [ ] 完成后验证 checkpoint、释放资源并记录结果。

## 当前记录

- 2026-07-27：用户授权开始实现，要求 10-step、一次真实推理通过后再启动 30k。
- 阶段 0 已完成。

## 阶段 1 记录

独占文件：

```text
src/YingMusicSinger/melody/midi_p_v4pvf.py
src/YingMusicSinger/models/model_v4pvf.py
src/YingMusicSinger/config/YingMusic_Singer_v4pvf.yaml
train_plus_v4pvf.py
run_sft_v4pvf_ddp.sh
probe_midi_p_v4pvf.py
infer_v4pvf_smoke.py
```

既有 V4f/V4vf/公共源码未修改。训练入口复制自已通过 4 卡审计的 V4vf，实现继续使用
固定参数顺序、缺失梯度补零和 25MB bucket 显式 all-reduce。

### 合成单元检查

输入构造为 4 帧 MIDI 60、2 帧 REST、2 帧 MIDI 62。输出 class 严格为：

```text
120, 120, 120, 120, 255, 255, 124, 124
```

P embedding 反向传播得到有限且非零梯度。该检查只证明解码和 autograd 基本性质，
不作为真实数据或训练 PASS。

### 真实音频探针

使用 SOFA 训练清单第一条真实花丸 WAV：

```text
${PRIVATE_PATH}
BV1Vz9BYuEEa_【花丸晴琉直播歌切】ひまわりの約束（向日葵的约定）_2_seg002.wav
```

- 音频：22.765 秒，44.1kHz；
- SOME：1961 帧 pitch logits + 1961 帧 boundary logits；
- 目标：490 VAE 帧；
- REST：88 帧，17.96%；
- 有效量化音高：17 类，class 110–129；
- PAD：0；
- 产物：`${PRIVATE_PATH}`。

首轮探针发现休止到发声边界出现 1 帧假 class 0。追溯证明 57 个解码音符中没有低音
有效音符，问题来自线性插值将 REST 占位 0 混入音高。修复为带 voiced mask 的归一化
线性插值，并在线性权重为零的边界使用最近邻音符兜底；复跑后有效范围为 110–129，
假 class 0 消失。有声音符之间仍保留线性平滑。

### 模型结构审计

- `melody_input_source=some_pretrain_quantized_v4pvf`；
- P embedding shape=`(257,128)`，`requires_grad=True`；
- trainable parameters=`338,151,744`；
- `midi_p_v4pvf.embedding.weight` 同时存在于 model 与 EMA state；
- optimizer 参数组包含同一 P embedding parameter；
- 训练时 SOME/量化 class 在 `no_grad` 内计算，embedding lookup 在其外，生成 loss 可反传；
- CKA 单独使用插值后的原始 128 维 SOME logits；未调用 `MIDIFuzzDisturb`。

服务器 `py_compile`、`bash -n`、训练/推理 `--help` 完整导入均通过。服务器提示未安装
`flash_attn` 后自动使用既有非 Flash 路径，不影响当前正式环境行为。

### 阶段复核

当前实现没有加载 V4vf/Official/V4f/V4fg DiT checkpoint；DiT 与 P embedding 均从随机
权重开始，使用 SOFA、官方 VAE、量化 P 输入和 raw SOME CKA，因此仍是 V4Pvf。

## 阶段 2 记录

服务器资源：

```text
GPU: 0,1,2,3
tmux: v4pvf_smoke
log: ${PRIVATE_PATH}
output: ckpts/plus_ja_sft_v4pvf_smoke/
```

启动前 GPU 0–3 均为 18MiB/0%，tmux、日志和输出均不存在。4 卡 10-step 正常完成：

- world size=4，effective batch=4；
- 约 0.61–0.98 秒/step；
- FlowA、FlowB、CKA、GradNorm 全程有限；
- 首批真实训练样本 MIDI_P：625 帧、REST 21 帧、24 个 pitch class、范围 101–137；
- `step_000010.pt` 与 `step_000010_final.pt` 均约 6.0GB；
- tmux 正常结束，无残留训练进程，GPU 0–3 恢复 18MiB/0%。

独立 checkpoint 审计：

- `global_step=10`，schema 与 V4Pvf v1 完全一致；
- model 与 EMA 全部浮点 tensor 有限；
- P embedding shape=`(257,128)`，同时存在于 model 和 EMA；
- optimizer 中 P embedding 的 `exp_avg`、`exp_avg_sq` 存在且非零；
- 与 seed 42 初始模型相比，P embedding 有 813 个数值变化，最大绝对变化
  `4.76837158203125e-07`。

显式梯度同步函数逐个遍历所有 `requires_grad=True` 参数，P embedding 与 DiT 使用同一固定
顺序 bucket all-reduce。该函数未从 V4vf 修改，V4vf 已做 4 卡同步审计；V4Pvf smoke 进一步
确认新增 P 参数产生 optimizer 状态和落盘更新。

### 阶段复核

smoke 从随机权重 step 0 开始，没有 resume 或 base checkpoint；日志明确打印量化 MIDI_P、
无 Fuzz、raw SOME CKA、官方 VAE，属于 V4Pvf。下一步只验证真实推理链路，不把 10-step
随机模型的听感作为质量结论。

## 阶段 3 记录

使用真实 SOFA 测试音频：

```text
${PRIVATE_PATH}
BV13b421J7bR_【花丸晴琉の翻唱】「フライディ・チャイナタウン」（20240728 B限歌回）_2_seg006.wav
```

- 原音频 8.034 秒，同时作为真实 A 音色参考和 B 旋律源；
- 目标歌词：`どこか静かな場所で着替えてみたいのよ`；
- checkpoint：`ckpts/plus_ja_sft_v4pvf_smoke/step_000010.pt` 的 EMA；
- 推理：32 steps、CFG=3、seed=42、官方 VAE、真实 SOME；
- 日志量化统计：377 帧、REST 50 帧、14 个 pitch class、范围 113–132；
- 输出：`${PRIVATE_PATH}`；
- 日志：`${PRIVATE_PATH}`。

输出验收：

```text
bytes: 708686
SHA256: SHA256_REDACTED
shape: (1, 354304)
sample rate: 44100
duration: 8.034104308390022s
peak: 0.79052734375
RMS: 0.07866280525922775
nonzero samples: 354229
```

音频所有样本有限且非静音。10-step 随机模型不具备听感评价意义，本阶段只证明真实 checkpoint、
P embedding、SOME 量化、ODE 采样和 VAE decode 的端到端链路没有异常。

### 阶段复核

推理 checkpoint 来自 Random DiT + Random P embedding 的 step-0 联合训练；推理加载 schema
严格匹配，并明确调用 `some_pretrain_quantized_v4pvf`。没有回退 sampled MIDI、V4vf/V4f
checkpoint 或假音频，因此仍是 V4Pvf。

## 阶段 4 启动记录

启动前：GPU 0–3 均为 18MiB/0%，tmux `v4pvf_30k`、输出目录
`ckpts/plus_ja_sft_v4pvf` 和日志 `train_v4pvf_30k.log` 均不存在；无其他训练进程；磁盘
可用约 1.2TB。

正式配置：

```text
init: random step 0, seed 42, no resume/base checkpoint
world: 4
effective batch: 16
LR: 1e-4
warmup: 500
hold: 12000
max steps: 30000
MIDI input: quantized 0.5-semitone P, no FuzzDisturb
CKA: raw SOME
VAE: official frozen VAE
tmux: v4pvf_30k
```

首批真实 MIDI_P 统计：625 帧、REST 21 帧、24 个 pitch class、范围 101–137。训练日志：

| Step | FlowA | FlowB | CKA | GradNorm | Clip | LR | 速度 | ETA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 2.1248 | 2.1252 | 0.7616 | 5.659 | 100% | 1.00e-5 | 1.46s | 12.2h |
| 100 | 2.0336 | 2.0330 | 0.6956 | 3.170 | 100% | 2.00e-5 | 1.47s | 12.3h |

数值有限、四卡满载，无 traceback/OOM/NaN。

### 启动复核

日志明确显示 Random DiT、Random P embedding、量化输入、无 Fuzz、raw SOME CKA、官方 VAE；
没有加载 smoke/V4vf/Official checkpoint，因此 30k 任务仍是 V4Pvf。

## 运行交接（2026-07-27）

用户确认当前任务可按“实现、10-step、真实推理、30k 启动并稳定运行”完成交付；未要求本线程
继续阻塞等待 30k 结束。交接时服务器状态：

```text
tmux: v4pvf_30k (RUNNING)
progress: 19850 / 30000
speed: 1.53s/step
ETA: 4.3h
latest train: FlowA=0.3618, FlowB=1.0052, CKA=0.1673
latest eval: step 19000, FlowA=0.3049, FlowB=1.0344, CKA=0.2040
latest checkpoint: ckpts/plus_ja_sft_v4pvf/step_018000.pt
log: ${PRIVATE_PATH}
```

截至交接无 traceback、OOM、NaN；2k–18k checkpoint 均按 2k 间隔保存。30k 最终结果尚未
产生，不能宣称最终 checkpoint 或最终指标已验收。查看状态：

```bash
tmux attach -t v4pvf_30k
tail -f ${PRIVATE_PATH}
```

## 主动终止与产物处理（2026-07-27）

V4vf/V4vfg 的最终人耳试听确认随机初始化路线虽有 loss 收敛，但生成包含大量噪音，感知
门禁失败。用户随后要求停止 V4Pvf，不再让随机 DiT 路线继续消耗训练预算。

- `v4pvf_30k` 在约 `21150/30000` 主动终止，tmux 和相关训练进程均已退出；
- 终止前最新完整 checkpoint 为 `step_020000.pt`；
- 用户随后明确要求删除正式训练 checkpoint；`step_002000.pt` 至 `step_020000.pt` 共 10 个
  `.pt` 文件已删除，合计约 63.5 GB；
- 训练日志、实现代码、文档、真实 SOME 探针和 smoke/推理工程证据保留；
- 本分支没有产生 30k 最终 checkpoint，也没有最终质量结论，不得将 21.15k 的数值稳定
  描述为感知门禁通过。

P 的后续主线切换为 Official base 起点的 V4Pf。V4Pvf 的量化、REST 修复、DDP、checkpoint
和真实推理验证可以作为工程证据复用，但随机 DiT 权重和中途 loss 不作为 V4Pf 初始化或
质量依据。
