# INS 全局唱法条件

## 1. 目标

V4Ij(ins)PH 使用 V4IPH 释放出的 `cond` 通道承载target singer唱法参考，不恢复 A 区，也不让旋律源
歌手的表示进入身份条件。目标能力为：

```text
模型权重                         -> target singer身份
target singer参考样本 R 的 INS            -> 怎么唱
GAME MIDI_P                     -> 唱什么音、何时唱
H/PUL                           -> 唱什么字、字落在哪里
```

本分支不要求 INS 空间能够自动聚类出“卖萌、沙哑、气声”等语义。风格由用户直接选择target singer
参考样本，语义名称可以后续人工赋予，不作为训练前置条件。

## 2. 与已有分支的关系

| 分支 | A 区 | `cond` | INS |
|---|---|---|---|
| V4PH | 存在 | A 区为真实 VAE latent，B 区为零 | 不存在 |
| V4IPH | 不存在，`ref_len=0` | 完整时间轴为零 | 不存在 |
| V4Ij(ins)PH | 不存在，`ref_len=0` | 完整时间轴为投影后的 INS | 仅本分支存在 |

V4IPH 是本分支的 null 条件基线。不得向 V4PH、V4H、V4Hg、V4Pf 或其他非 I
分支静默加入 INS adapter 或 INS runtime。

## 3. 训练数据流

训练样本仍是target singer target B：

```text
target singer B waveform
  ├─ VAE encode ----------------------> full latent flow target
  ├─ GAME ----------------------------> MIDI_P
  ├─ H frontend ----------------------> H/PUL
  └─ frozen INS encoder -------------> e_B [768]
                                           |
                                           v
                                  learnable adapter
                            768 -> 512 -> SiLU -> 256 -> SiLU -> 64
                                           |
                                           v
                                  repeat over T frames
                                           |
                                           v
                                    cond [B,T,64]
```

训练时从 target B 提取 INS 是有意的 reference-style self-conditioning：模型学习target singer音频
的全局 INS 与目标唱法之间的对应关系。训练 cache 的输入严格等于实际送入 VAE 的 B waveform：
单声道、44.1 kHz、应用既有最多 30 秒截断。INS 只能作为低维全局条件，不得携带原始波形、
VAE latent 或逐帧声学特征进入 `cond`。

首轮 encoder 固定为 `ajd12342/paraspeechclap-intrinsic` revision
`e80ad8eb199f7573f07f617c15a7497504151838`，checkpoint
`slap-intrinsic.pth.tar` SHA256: redacted
`22b37acc1bb27a26034614b6eab1562131a4197b86fd0221e45350f927663ad2`。输入从 44.1 kHz
重采样到 16 kHz，调用 `get_audio_embedding(normalize=True)`，输出 768 维 L2-normalized 向量。

## 4. 推理数据流

推理使用三类彼此分离的输入：

```text
target singer唱法参考 R -> frozen INS -> adapter -> cond
旋律源 B       -> GAME P
目标歌词       -> H/PUL
```

强制边界：

- INS 只从target singer样本 R 提取；
- 旋律源 B 即使来自其他歌手，也不得进入 INS encoder；
- R 不命名为 A，不进入 VAE，不与生成音频拼接，不贡献时间长度；
- `ref_len` 始终为 0，输出覆盖完整目标时间轴；
- R 的歌词只属于参考样本，不得覆盖或补充目标歌词。
- R 是用户明确选择的完整target singer片段；运行时不得按时长、响度、歌词或相似度静默裁剪、替换或聚合
  R。R 的 SHA256: redacted

## 5. cond 契约

首版逻辑接口冻结为：

```text
e_ins:    [B, 768]
adapter:  Linear(768, 512) -> SiLU -> Linear(512, 256) -> SiLU -> Linear(256, 64)
style:    [B, 64]
cond:     style[:, None, :].expand(B, T, 64)
```

必要约束：

1. INS encoder 冻结；
2. 前两层使用正常初始化；第三层的 weight 和 bias 均零初始化，使 step 0 的 `cond` 精确为零；
3. null INS 与 INS dropout 必须生成逐值全零的 `cond`；
4. 不改变 DiT 输入形状，继续复用既有 64 维 cond 槽；
5. adapter 和 INS 配置进入独立 checkpoint schema 与 SHA256: redacted
6. 同一 INS 向量在首版中覆盖完整时间轴，不先引入逐句或逐帧 INS。
7. ParaSpeechCLAP 输出已 L2-normalized；adapter 以 FP32 消费该向量，首轮不额外加入
   LayerNorm、统计标准化或 feature dropout。

第 6 项用于限制歌词、局部 F0 和逐帧声学泄漏。未来逐句 INS 必须另开消融，不能静默替换
首版全局接口。

## 6. 继承项

除 `j(ins)` 外，首轮默认继承 V4IPH/V4PH 的以下语义：

- `ref_len=0` 与全 B flow target；
- H/PUL renderer；
- GAME medium K=4 MIDI_P 与 GAME-compatible CKA；
- official VAE；
- 完整时间轴 FlowB 与 CKA；
- text/MIDI CFG dropout、EMA、DDP、checkpoint 和 exact resume 框架；
- null INS 时的完整 V4IPH 推理能力。

首轮训练从与 V4IPH 相同的 P500 起点开始，总计 30,000 optimizer steps。DiT 的 optimizer、
scheduler、LR、EMA、seed 和数据游标严格复用 V4IPH；adapter 是独立 optimizer parameter group，
不使用 adapter-only 校准阶段。adapter 的 schedule 按 global step `s` 定义：

```text
0 <= s < 8000:       adapter requires_grad=False, LR=0
8000 <= s < 14000:   linear warmup, 0 -> 1e-4
14000 <= s < 18000:  cosine decay, 1e-4 -> 1e-5
18000 <= s <= 30000: cosine decay, 1e-5 -> 0
```

前 8k 不只是把 LR 数值设为零：adapter 不参与 backward、Adam state 或 EMA 更新，避免末层在
零 LR 时积累动量。adapter 初始化必须使用独立 RNG 或保存并恢复全局 RNG，不能改变 V4IPH 的
随机序列。因而 step 8k 的 DiT 参数 SHA256: redacted
与 V4IPH 同步 step checkpoint 一致；任一不一致都不得进入 INS 学习阶段。

## 7. 已知风险

INS 已被证明不适合直接承担无监督唱法聚类；其向量仍可能混入：

- 参考歌词与内容；
- 绝对 F0、音区和旋律轮廓；
- 录音、响度、混响与频谱；
- target singer样本自身的局部状态。

本分支只验证这些连续特征能否作为有效条件，不把 INS 距离解释为干净唱法距离，也不把
训练成功解释为 INS 已完成语义解耦。

## 8. 强制评价门禁

固定 target 的旋律、歌词、时长、seed、CFG 和 checkpoint，只交换target singer参考 R：

1. 输出必须继续保持target singer身份；
2. F0 必须主要跟随 GAME P，不能被参考 R 的绝对音高带跑；
3. 输出歌词必须跟随 H/PUL，不能混入 R 的歌词；
4. 人耳应能感知唱法随 R 改变，否则 INS 条件无效；
5. null INS 输出应保持 V4IPH 能力，不能出现灾难退化；
6. 相同 R 配不同 target 时，风格影响应可复现而非只在同内容上成立；
7. 不同 R 但相近人工唱法时，输出变化不应主要表现为随机音色或录音污染。

若歌词、F0 或身份任一主轴出现系统性污染，停止正式长训，先做 adapter、输入窗口或 dropout
消融，不以“风格变化明显”掩盖控制泄漏。

训练健康指标固定为与 V4IPH 同步 checkpoint 的 eval FlowB 差值；INS 条件下的 FlowB 应记录为
`FlowB_ins - FlowB_v4iph`，null INS 的 FlowB 单独记录以监视 V4IPH 能力退化。FlowB 不单独决定
成功与否。30k 完成后，用户基于固定 target、seed、CFG、歌词和 GAME P、只交换 R 的生成集进行
最终听感裁决。首轮 INS dropout 比例为 0.3；style guidance 固定为 1，不引入额外 style-CFG
放大语义。

## 9. 未决项

- ParaSpeechCLAP Intrinsic cache 的 `v4ijph_ins_cache_v1` 文件布局、record key 与独立审计器；
- adapter optimizer 的 weight decay 和是否使用与 DiT 相同的 Adam betas；
- 独立全时轴 style 推理入口的文件位置、ODE/CFG 复用方式和 checkpoint 加载接口；
- 固定 R/target 听感集、近邻 INS 对的距离阈值 N，以及生成结果的 provenance manifest；
- target singer参考样本的保存、人工命名与推理 UI 契约。

这些剩余项不改变已冻结的模型语义与训练配方；在静态、单卡和四卡门禁前必须落实为可审计工件。

## 10. 当前状态

- 分支、模型语义、adapter 结构和首轮训练配方已冻结；
- 工程实现已建立并本地验证：adapter 768->512->SiLU->256->SiLU->64、
  v4ijph_contract LR 分段 schedule、8k 冻结门禁、INS cache schema
  v4ijph_ins_cache_v1、checkpoint schema v4ijph_training_checkpoint_v1；
- 本地 py_compile 全通过，V4IjPH 契约 5 项 + INS cache 2 项 + V4IPH 复用
  placement 3 项 + V4IPH contract 5 项共 15 项测试通过；
- 已修复 run_v4ijph_30k_ddp.sh 中 PHASE_A_SHA 与 V4IPH/V4PH phase B 不一致的
  转录错误；
- 未生成服务器 INS cache；
- 未运行服务器静态、单卡、四卡或 checkpoint 门禁；
- 独立全时轴 style 推理入口尚未实现；
- V4IPH 保持独立的零 cond 正式基线，不因本分支定义而改变。

















