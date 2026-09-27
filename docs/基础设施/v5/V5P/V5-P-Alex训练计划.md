> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# V5-P-Alex 训练计划

## 1. 版本定位

`V5-P-Alex` 是面向 Alex 的英语 YMSP P+H 训练路线。它以已完成审计的 `V5-P` 为唯一训练
基线，主要通过删除不适用的日语、L 和 g 路线来得到一个更容易阅读、交接和维护的入口。

版本目标：

```text
Alex preprocessing -> frame-level H tokens
  + Alex MIDI-P token condition
  + runtime GAME-derived CKA target
  + Alex-provided dense H token placement
  + V5-P P+H training recipe
```

这里的 GAME target 由训练端实时提取：

```text
training audio -> frozen GAME posterior
  -> existing GAME-to-SOME-compatible 128-dim adapter
  -> CKA target
```

Alex 不提供 CKA target 或 GAME posterior；它也不表示运行时重新调用 SOME。主 Flow Matching
target 仍是 frozen VAE 产生的 latent velocity。

## 2. 基线与减法原则

### 2.1 唯一基线

- 训练语义以 `package_v4c_finetune/train/train_v5p.py` 为准；
- V5-P 的 GAME-P、GAME CKA、A/B、FlowA/FlowB、EMA、checkpoint、resume 和分布式同步语义保持；
- 旧 `train_v5p.py` 只读保留，不在本版本直接修改；
- 新入口建议命名为 `train_v5p_alex.py`。

### 2.2 只允许的改动

| 区域 | V5-P-Alex 处理 |
|---|---|
| 文本/H 输入 | 训练端不运行 G2P 或 alignment，只接收 Alex 生成的逐帧 H token |
| 数据 manifest | 使用 Alex 提供的英语文本、H token 和 MIDI-P token 资产 |
| 训练入口 | 新建易读入口，保留必要的 V5-P 合同校验 |
| metadata | 明确 `language=en`、CNEN、alignment schema 和 GAME provenance |

### 2.3 明确删除的内容

以下内容不进入 V5-P-Alex 第一版：

- V5-P 的 L-FULL-LENGTH-TIME 长样本去重池及其 Long 构建流程；
- Phase C 的 285k online VAE g-adaptation；
- V5-Pg、V5-Pg20-HLR07 等 warm-start 分支；
- 日语 kana normalization、mora grouping、Japanese SOFA parser；
- continuous SOME 条件或 SOME teacher runtime；
- LCF、INS、SVC 自克隆、GRPO 和模型扩容；
- 为追求多样性而新增的 pitch jitter、随机音高扰动或额外 loss。

删除这些路径不是改变 V5-P 的已验证语义，而是缩小 Alex 需要理解和维护的代码面。若未来
需要其中任一路线，必须另立版本，不在本文件中逐步加回。

## 3. 数据合同与责任边界

本项目当前没有本地英语训练数据。英语文本、逐帧 H token、MIDI-P token 和英语 manifest 由
Alex 准备并交付；runtime GAME teacher 由训练端负责。我们不伪造英语样本，也不把日语数据
冒充英语路线的训练数据。

### 3.0 接口调研结论

仓库的 H 训练入口最终消费的是长度为 `T` 的 dense text token 序列，而不是 phone interval。
因此 V5-P-Alex 不在训练端运行 G2P、SOFA 或其他英语对齐工具；这些步骤由 Alex 在数据准备
阶段完成。训练接口固定为：

```text
Alex preprocessing
  -> frame-level H token sequence [T]
  -> training-side shape/range/ordering audit
```

H token 必须直接使用 V5-P 训练 ID：普通 text token 为 `1..363`，filler/PAD 为 `0`，
`SEP=365`，`PUL=366`。训练端不重新解释音素，也不根据文本重建 H；Alex 交付的 dense H
序列是唯一权威输入。

### 3.1 Alex 提供的英语文本资产

每条音频至少绑定以下字段：

```json
{
  "text": "I am falling",
  "h_token_path": "h/song_0001.npy",
  "midi_p_path": "midi_p/song_0001.npy"
}
```

- `h_token_path` 指向长度为 `T` 的 dense H token 数组；
- Alex 可以在自己的预处理阶段使用 G2P/对齐工具，但训练端不重复执行；
- H token 的时间轴必须与该音频对应的 VAE latent 帧数一致。

### 3.2 MIDI-P 与实时 CKA

MIDI 接口以训练侧使用的 `midi_p` token 序列为主，不要求 Alex 交付 note event。Alex 至少应
提供与音频时间轴对齐的：

```text
midi_p: [T]  # one token per VAE frame, or an explicitly declared source frame rate
```

CKA target 不属于 Alex 的数据合同。训练端从当前 batch 的音频实时运行 frozen GAME：

```text
audio -> frozen GAME posterior -> 128-dim adapter -> cka_probs [T, 128]
```

训练时的两条旋律支路为：

```text
Alex midi_p_tokens [T] -> p_classes -> P embedding -> DiT condition
audio -> frozen GAME -> cka_probs [T, 128] -> B-only CKA target
```

注意当前 V5-P 代码的变量名：`p_classes` 才是送入 P embedding 的离散 token；代码中的
`midi_p` 变量实际承载 `cka_probs` 连续目标。V5-P-Alex 对外接口应使用不歧义的字段名
`midi_p_tokens`（内部映射为 `p_classes`），不要把两者混写。

P class 语义固定为：

```text
0..254 = 0.5-semitone pitch class
255    = REST
256    = PAD
```

不要把 `0` 当作静音；`0` 是合法的最低 pitch class。若 Alex 使用 `C5`、`D5` 等音名，必须
在交付前转换为固定的半音类整数（例如 MIDI 72 -> class 144，MIDI 74 -> class 148），并在
manifest 中声明映射规则。

`cka_probs` 只由训练端的 frozen GAME posterior 和现有 128 维 adapter 实时生成，不能从
Alex 的离散 token 用 one-hot 猜造。

每个 `midi_p` 文件需要绑定 `source_path`、`audio_sha256`、时间基准、sample rate 和 token schema；
禁止使用 basename fallback。训练 checkpoint 另行绑定 runtime GAME commit、模型 SHA、配置和
adapter schema。GAME 必须保持 frozen，实时提取过程不得建立训练梯度。

## 4. 训练链路

```text
English audio + Alex H tokens
  -> H token validation (no G2P/alignment in training)
  -> A/B latent construction with frozen VAE
  -> Alex MIDI-P tokens -> P embedding -> DiT condition
  -> frozen GAME runtime posterior -> 128-dim CKA target
  -> FlowA + 2*FlowB + 0.7*CKA
  -> backward -> optimizer -> CPU EMA -> checkpoint
```

A 区继续作为参考音频，MIDI 条件清零；B 区使用 H token 和 MIDI-P token 条件。VAE、GAME teacher
及其他冻结模块不产生训练梯度。

## 5. 训练阶段

### Phase 0: Alex input gate

先用至少 100 条 English Gold Set 验证：

```text
Alex H tokens + MIDI-P tokens -> shape/range/timebase -> valid tensors
```

这一阶段不更新模型。

### Phase A: P-only calibration

沿用 V5-P 的 500-step P-only 合同，只训练 P embedding。检查 P row 更新、REST/PAD、GAME
support、冻结 DiT 和 checkpoint resume；通过后只传递 model state，不继承 optimizer、scheduler、
EMA、RNG 或 sampler cursor。

### Phase B: English P+H joint

首轮沿用 V5-P joint 配方：official frozen VAE、FlowA + 2*FlowB + 0.7*GAME-compatible CKA、
audio/text/MIDI dropout 为 `0.3/0.15/0.3`、CPU EMA。先做 2k smoke 和 6k pilot，再决定是否
启动正式预算。第一版不自动进入 g-adaptation。

## 6. 可读性和实现边界

新训练文件应按以下顺序组织，并使用有层次的 English comments：

```text
configuration and contracts
data loading and validation
H token validation
GAME adapters
model/optimizer construction
one training step
evaluation and checkpointing
main entry
```

可以删除 V5-P 中与 Alex 路线无关的分支、历史兼容参数和 g 阶段代码；删除后必须保留必要
的 schema、SHA、冻结参数、resume 和 distributed correctness checks。不要把可读性改写成全新
算法，也不要通过大规模重构改变张量命名和 checkpoint 语义。

## 7. 验收门禁

至少执行：

1. Python compile/import/`--help`；
2. H token、MIDI-P token 和时间轴合同检查；
3. MIDI-P token 审计，以及 runtime GAME 输出 shape/finite/frozen 检查；
4. GPU 7 单卡 step-1/step-10 forward/backward/save；
5. 60 秒样本门禁（若英语数据包含该长度）；
6. checkpoint resume 与冻结模块参数审计；
7. 与 V5-P 固定输入的 loss/mask/detach/EMA 结构回归。

英语专项评测另记录 WER、phoneme error rate、phone onset error、vowel duration error 和 F0
correlation，并覆盖长元音、辅音簇、弱读、连读和跨歌手样本。

## 8. 当前待冻结事项

在开始数据驱动训练前，Alex 需要确认并交付：

- H token 的 shape、token ID 表、SEP/PUL/filler 规则和生成工具版本；
- H token 与 VAE 帧的对应规则；
- MIDI-P token 文件格式、帧率、REST/PAD 和音高映射；
- 训练端 runtime GAME 版本、模型 SHA、采样率和 adapter schema；
- 英语训练集、验证集和固定 Gold Set manifest。

在这些接口和资产交付前，我们只能完成合同设计、代码精简、静态检查和无数据的 fixture
测试；不能宣称英语 H、GAME 或训练质量已经通过验证。

## 9. 分工与可执行范围

### 我们现在可以做

- 冻结 `V5-P-Alex` 数据合同、metadata 和 provenance 字段；
- 从 V5-P 做减法，创建易读的 `train_v5p_alex.py` 骨架；
- 删除 L、g、日语和 SOME 分支，同时保留 loss、EMA、checkpoint、resume 和冻结检查；
- 编写 H/MIDI-P/GAME manifest auditor；
- 使用小型 fixture 做 schema、shape、mask、detach 和 checkpoint round-trip 测试。

### 必须等待 Alex

- English H token 与 manifest；
- English MIDI-P token 与 manifest；
- Phase 0 前端门禁；
- P-only、P+H smoke、pilot、正式训练和英语专项评测。

在 Alex 资产到位以前，任何 GPU 训练结果都只能算工程链路测试，不算 V5-P-Alex 模型结果。

## 10. 实施记录

### 2026-09-01：正式实现启动

本轮目标是完成新的 `package_v4c_finetune/train/train_v5p_alex.py`、独立数据审计器和可重复
验证。旧 `train_v5p.py` 作为唯一行为基线，只读保留。

当前冻结的实现边界：

```text
Alex input: audio + h_tokens[T] + midi_p_tokens[T] + manifest
P condition: Alex midi_p_tokens -> p_classes -> existing V5-P P embedding
CKA target: audio -> frozen runtime GAME -> existing 128-dim adapter
Training core: V5-P Phase A p_only + Phase B joint
Removed: HAlignment runtime placement, offline GAME input cache, L pools, g-adaptation,
         Japanese frontend, SOME teacher runtime, historical engineering branches
```

开始编码前的验收判据：

1. 原 `train_v5p.py` 内容哈希保持不变；
2. 10 条花丸 Alex-format fixture 全部通过 schema、audio、dtype、range、SHA 和共同 `T` 审计；
3. 固定 V5-P 样本上，Alex 输入张量与 V5-P reference 张量逐项比较；
4. 固定 seed 下比较 FlowA、FlowB、CKA、total loss、梯度、单步参数和 EMA 更新；
5. Phase A/Phase B checkpoint save、resume 和冻结参数检查通过；
6. GPU 7 完成 step-1/step-10 工程 smoke，并记录实时 GAME 的峰值显存和耗时；
7. 英语数据未交付前，不宣称英语训练质量或 English Gold Set 已验收。

当前已完成的前置资产：

- 中英文数据接入 DES 与绑定 `vocab.json`；
- `examples/v5p_alex_hanamaru_10/` 共 10 条结构样例；
- 10/10 样例已通过一次独立的音频、SHA、shape、dtype 和 token 范围检查。

下一步：审计 V5-P 的模型构造、Phase A 转接、runtime GAME 调用能力和服务器依赖，然后实现
Alex loader/auditor。任何改变训练数学语义的必要修改必须先记录在本节，不能静默引入。

### 2026-09-01：服务器传输规则修正

服务器已有花丸原始音频和 V5-P GAME 资产，本机不得再次传输训练音频、checkpoint 或其他大
文件。后续本机与服务器之间的双向传输限制冻结为：单个文件必须严格小于 `100,000 bytes`。
上传和下载均适用；只允许小脚本、manifest 选择表和小型校验摘要。fixture 必须在服务器侧根据
`source_sample_id`、`audio_sha256` 和现有数据路径原地生成。

若后续确实需要在本机与服务器之间交换大文件，必须使用阿里云盘中转，不得拆分或绕过 SCP
限制。服务器端使用 `${PRIVATE_PATH}`，本机当前已确认挂载可读路径为
`${LOCAL_PATH}`。标准流程是服务器上传云盘、等待挂载刷新、本机用 `robocopy` 从 P 盘读取，最后
对服务器、云盘 sidecar 和本机文件复核 SHA256；P 盘只读，不从本机直接写入或删除。

本轮曾错误启动 10 条 WAV 的传输；传输未完成，已向专用目录
`${PRIVATE_PATH}` 发出清理命令。SSH 随后在 banner
阶段超时，因此恢复连接后的第一项工作是核实该专用目录已经删除。不得在核实前启动训练。

已传输的两个代码文件均小于 100,000 bytes，可保留：

```text
package_v4c_finetune/train/train_v5p_alex.py
package_v4c_finetune/train/audit_v5p_alex_data.py
```

### 2026-09-01：本轮恢复检查点

本轮继续完成 `V5-P-Alex` 训练入口与验收。开始操作前冻结以下当前事实和约束：

- 旧 `train_v5p.py` 仍以 SHA256
  `SHA256_REDACTED` 作为只读基线；
- 服务器侧 10 条 fixture 已在现有 V5-P 数据上原地生成并通过审计，共 236.52 秒、5,098
  latent frames；不得上传 WAV 重新生成；
- GPU 7 的 Phase A step-1 smoke 已在 tmux `v5p_alex_p1` 中启动。恢复后必须先轮询该会话及
  日志，确认终态前不得重复启动；
- 每次上传前检查单个文件大小，只有严格小于 `100 KB` 的脚本、配置或文本报告可以传输；
  下载也执行同一限制；音频、checkpoint、模型权重继续复用服务器资产；
- 当前训练入口仍是第一版，尚未验收。后续按实际日志修复，然后清理旧分支死代码，补 Alex
  checkpoint 审计，完成 step-1、step-10、save/resume、golden-batch 与 60 秒运行验证；
- Alex 英语数据尚未交付，因此工程链路可以验收，但英语数据质量与模型效果不能在本轮宣称
  已验证。

本轮第一项操作：读取 `v5p_alex_p1` 的实际状态、日志和 GPU 7 状态，并将结果继续写入本文档。

#### GPU 7 Phase A step-1 首次结果

已核实 tmux `v5p_alex_p1` 不再存在，GPU 7 空闲；因此该次 smoke 已终止，不是仍在运行。日志证明
模型、base checkpoint、official VAE、frozen runtime GAME 和 10 条 Alex fixture 均能完成初始化，随后
在第一批次进入 `process_batch` 时失败：

```text
ValueError: official VAE frame count 268 != manifest frame count 269
```

同一时刻重新计算旧 `train_v5p.py` SHA256，仍为
`SHA256_REDACTED`，基线未被修改。

下一步先对照 official VAE 的真实长度变换、fixture builder 的 `T` 推导以及 V5-P 原训练裁剪方式。
修复必须统一数据合同与实际模型时基，不得仅删除长度断言或静默截断来绕过错误。

#### official VAE 精确 fixture 修复

根因已定位：旧 fixture builder 读取历史 H manifest 的 `ExpectedLatentFrames`，该字段是时长推导值，
不是 official VAE 的实际输出。训练端 DES 明确禁止用固定帧率估算 `T`，因此训练入口的严格等长断言
保留，修复放在 fixture builder：新增 official VAE config/checkpoint/device 参数，逐条编码现有 hard-link
音频，以 latent 实际末维定义 `T`，再据此渲染 H 与 MIDI-P。

修正后的 `build_v5p_alex_fixture.py` 为 8,206 bytes，上传前已确认小于 100,000 bytes；服务器
`py_compile` 通过。GPU 7 原地生成的新 fixture：

```text
directory: ${PRIVATE_PATH}
entries: 10
audio_seconds: 221.17857142857142
latent_frames: 4758
manifest_sha256: SHA256_REDACTED
auditor: PASS (shared H/MIDI-P T, dtype/range/path/audio/SHA checks)
```

为隔离变量，曾从旧错误 fixture 导出 10 个 source ID 并尝试同样本重建；第 3 个 ID 不在选定的
单一 source manifest 中，证明首次 `--count` 构建没有留下足以复现其统一来源的参数。该尝试已经
终止，不用于训练。后续工程 smoke 使用上面完整且审计通过的 exact fixture；golden-batch 另行建立
显式、单一来源的固定样本合同。

#### GPU 7 Phase A step-1 exact fixture：PASS

使用 `hanamaru10_vae_exact` 和 manifest SHA256 `da58f77e...e64eab37` 重跑未经清理的首版
训练入口，已完成一整个训练 step：

```text
FlowA=9.2701
FlowB=8.9634
CKA=0.9803
step_time=1.20s
final_parameter_sha256=SHA256_REDACTED
checkpoint=${PRIVATE_PATH}
checkpoint_size=3,647,420,909 bytes
run_state=stopped (intentional --stop_after_step 1)
```

该结果证明 Alex H、Alex MIDI-P、official VAE、runtime frozen GAME、Flow A/B、CKA、反向传播、
optimizer、EMA、P-only frozen fingerprint 和 checkpoint 写出均已贯通。3.65 GB checkpoint 只留在服务器，
不得传回本机。下一步以此作为 pre-clean 回归基线，删除 Alex 永远不可达的 g-adaptation warm-start、
placement/pool 旧指标和旧日志标签，再重新跑 step-1。

#### 可读性清理与 step-1 等价回归：PASS

训练入口已完成以下减法：

- 删除不可达的 g-adaptation warm-start 和 `elif False` 整块；
- 删除旧 placement/pool/short-long/segment 指标；
- H 指标改为 Alex dense token 直接可解释的 `HPUL` 与 `HNonzero`；
- 数据指标改为窗口/累计样本数和累计音频秒数；
- 日志前缀统一为 `Train V5P-Alex`；
- checkpoint schema 升为 `v5p_alex_p_checkpoint_v2` / `v5p_alex_joint_checkpoint_v2`；
- metadata 的语言改为逐条来自 manifest，不再误写固定 `en`；
- 增加静态门禁，禁止重新引入 g-adapt、HAlignment、G2P、SOFA、note-event、离线条件字段或
  `ExpectedLatentFrames` 估算。

清理后脚本为 70,962 bytes。上传文件均先检查 `<100,000 bytes`；服务器 12/12 数据与静态测试
通过。相同 GPU、manifest、seed 和参数重跑 step-1：

```text
                         pre-clean             clean
FlowA                    9.2701                9.2701
FlowB                    8.9634                8.9634
CKA                      0.9803                0.9803
final parameter SHA256   368723...dfd0         368723...dfd0
```

loss 和单步参数结果完全一致，证明本轮减法没有改变训练数学。clean checkpoint 留在服务器：
`${PRIVATE_PATH}`。

#### GPU 7 step-10、checkpoint auditor 与 exact resume：PASS

最终观测版再次完成 step-1 等价回归，FlowA/FlowB/CKA 和参数 SHA256 仍与 pre-clean 完全一致。
新增的 smoke 观测不改变训练结果，并记录：official VAE `0.4354s`、runtime GAME `0.2080s`、
peak allocated `9,553 MiB`、peak reserved `9,632 MiB`，VAE/GAME 终态冻结检查通过。

连续 10-step 结果：

```text
checkpoint: ${PRIVATE_PATH}
final_parameter_sha256: SHA256_REDACTED
vae_avg: 0.2157s
game_avg: 0.0984s
peak_allocated: 9,911.5 MiB
peak_reserved: 14,196.0 MiB
frozen_runtime: PASS
```

新的 `audit_v5p_alex_checkpoint.py` 已审计 step 1、step 5 和 step 10：schema、manifest、runtime
GAME、P/PAD、optimizer、scheduler、EMA、rank RNG/cursor、running metrics 和全 checkpoint finite
全部通过。step-10 支持 51 个 pitch row、3,534 个支持帧，cursor 为 `(0, 10)`。

exact resume 使用同一 seed/manifest：连续跑 10 steps，与先跑 5 steps 再从 checkpoint 恢复到
step 10。step 6-10 的逐步 loss 完全一致，最终参数 SHA256 相同。新的
`compare_v5p_alex_resume.py` 对以下状态做 bit-exact 比较：model、EMA、optimizer、scheduler、
metadata、rank RNG/cursor/metrics、P support/history、initial weights/fingerprint 和 transition fields。
共比较 2,013 个 tensor、911,644,218 个 tensor elements 和 1 个 NumPy array，结果 `exact=true`。
只排除必然不同的 `args` 中 output/resume 路径。

下一步：建立 Alex 专用 Phase A adjudicator 与 Phase B 转接 smoke，完成 60 秒样本门禁，再建立
固定 golden-batch 对照证据。英语数据尚未交付，英语效果仍不在本轮可宣称范围内。

#### H 与随机 reference boundary 的语义门禁：FAIL，暂停 Phase A 500-step

在 golden-batch 对照前发现，旧 V5-P 的 `render_h_pul_placements` 不只读取 phone 时间，还直接
读取本 step 随机生成的 `ref_len`。phrase 的 A/B 归属、跨边界 phrase 的首 token 锚点，以及相邻
phrase 间的 SEP/PUL 区间都可能随 `ref_len` 改变。Alex 入口却读取预先生成的固定 `h_tokens[T]`，
同时继续随机生成另一个 `ref_len`；因此此前 step-1、step-10 和 exact-resume 只能证明链路可运行，
不能证明输入 V5-P 数据时与旧 V5-P 等价。

新增只读诊断器 `audit_v5p_alex_h_boundary.py`，在服务器现有 exact fixture 上复现旧 V5-P 的 L1
reference policy。每条样本以确定 seed 抽取 128 次合法 runtime boundary，共 1,280 次：

```text
samples:                         10
runtime draws:                 1280
equal to fixed fixture H:       768
unique runtime ref_len:          40
unique ref_len equal to H:       23
all equal:                    false
```

其中 `hanamaru_02/04/05/06` 的 128/128 次 runtime H 均不等于 fixture 固定 H，单条最多分别有
32/22/87/71 帧不同；其余 6 条对本次边界集合恰好不敏感。根因是 fixture builder 用约 25% 的固定
边界渲染 H，而旧 V5-P 会把随机边界向 phrase start 吸附。审计报告保存在服务器
`${PRIVATE_PATH}`，本机副本为
`experiments/v5p_20260808/server/v5p_alex_h_boundary_audit.json`。

MIDI-P 不存在同类问题：Alex 与旧 V5-P 都先使用完整时间轴的 `p_classes[T]`，再在 embedding 后
依据本 step 的 `ref_len` 清零 A 区。当前阻塞只属于 H/reference 合同。

在以下合同中选择一种以前，不得启动 Phase A 500-step，也不得建立声称 V5-P/Alex 等价的
golden-batch：

1. 在 manifest 增加每条样本的固定 `reference_frame_count`，Alex H 与该边界共同生成，训练端不再
   为该记录随机边界；实现最简单且可做逐张量严格等价，但会减少同一记录跨 epoch 的 A 长度变化；
2. Alex 为同一音频提供多个 `(reference_frame_count, h_tokens)` 变体；可以恢复边界多样性，但数据
   制作和 manifest 更复杂；
3. 扩充 Alex 输入为可在训练时按随机边界重新渲染的 phrase/event metadata；最接近旧 V5-P 的
   随机语义，但违反当前“训练端只接受 dense H、不运行 placement”的减法目标。

当前工程建议是方案 1：只在既有 manifest 增加一个整数，不增加新文件或 runtime 前端；先以同一
固定边界完成 V5-P/Alex golden-batch 等价，再由真实英语数据训练需求决定是否升级为多变体方案 2。

#### 2026-09-02：句首帧数组合同冻结并开始实施

用户选择由 Alex 在 manifest 中提供每句开始位置，但位置不是秒或毫秒，而是 official VAE 时间轴
上的整数 frame index 数组。字段冻结为：

```json
"sentence_start_frames": [18, 93, 171]
```

因此输入 schema 升为 `v5p_alex_input_v2`。每个元素必须与对应句第一个普通 H token 的 frame
完全一致；数组非空、严格递增且至少包含一个正帧。每句恰好一个 SEP，下一句存在时 SEP 固定在
下一句开始帧减一，最后一个 SEP 固定在 `T-1`。

训练端仍保留随机 reference proposal，但最终 boundary 无条件吸附到最近的正数句首帧。旧 V5-P
的 2.5 秒 margin 不适用于固定 dense H：若 margin 内没有句首而保留原 proposal，boundary 仍可能
切进句中。Alex 路线以“boundary 必须是句首”为新增的明确减法合同，从而同时保留跨 step 的句首
边界多样性和固定 H 的时间语义。

本次属于不兼容合同升级：旧 v1 manifest 和此前 v2 checkpoint 不得 resume。训练 checkpoint
schema 升为 `v5p_alex_p_checkpoint_v3` / `v5p_alex_joint_checkpoint_v3`，Phase-A adjudication 升为
`v5p_alex_phase_a_adjudication_v2`。服务器旧 step-1/5/10 checkpoint 只保留历史工程证据，不能作为
新合同的训练起点或最终验收证据。

已开始修改数据 auditor、fixture builder、训练入口、checkpoint auditor 和测试。完成本地门禁后，
必须在服务器原地重建 v2 fixture，再重新执行 golden-batch、step-1/10、checkpoint 与 exact-resume；
不得复用旧 fixture 的 PASS 结论。

#### v2 句首安全 fixture 与边界审计：PASS

本地和服务器均完成 11 项数据测试、6 项训练静态/边界选择测试，共 17/17 PASS；旧
`train_v5p.py` SHA256 仍为 `50e186...e59`。本轮上传的单个脚本最大 73,396 bytes，所有双向
直接传输均严格小于 100,000 bytes，没有通过 SCP 传输音频、checkpoint 或完整 fixture。

首轮同源选择中发现两类应被 v2 合同拒绝的旧 V5-P 记录：一条相邻 phrase 从第 242/243 帧
开始并触发 `sample_control_anomaly`，另一条没有正数句首可供 A/B 分割。它们没有通过放宽规则
进入 fixture，而是被替换为同一冻结 `train_short ∩ GAME manifest` 中满足句首边界合同的记录。

服务器在 GPU 7 使用 official VAE 和现有 hard-link 音频原地生成：

```text
directory: ${PRIVATE_PATH}
schema: v5p_alex_input_v2
entries: 10
audio_seconds: 221.99102040816325
latent_frames: 4776
sentence_starts: 70
manifest_sha256: SHA256_REDACTED
data auditor: PASS
```

新的 boundary auditor 对每条样本确定性抽取 128 次随机 proposal，共 1,280 次。proposal 最终
落入 15 个可达句首边界；15/15 边界和 1,280/1,280 次抽样均与预生成 fixed dense H 逐帧相同，
每条样本的 runtime H 独立输出数均为 1：

```text
all_equal: true
equal_runtime_draws: 1280 / 1280
equal_unique_ref_lens: 15 / 15
```

完整报告只留在服务器：
`${PRIVATE_PATH}`（9,894 bytes）。下一步进入模型级
golden-batch，比较同一 boundary 下的 V5-P/Alex H、P、CKA、mask、loss、梯度、更新与 EMA。

#### v2 step-1 模型级 golden：数值等价 PASS，非 bit-exact

GPU 7 上使用同一条 28.7 秒花丸样本、seed 42、step-1 和同一初始模型分别运行只读旧 V5-P 与
Alex。两边的训练日志一致：

```text
FlowA=9.5496  FlowB=8.9465  CKA=0.7261
H PUL frames=161  H nonzero frames=283
```

第一版严格 comparator 正确 FAIL：最终 P embedding 的 32,896 个元素中有 19 个相差一个 FP32
ULP；对应 Adam 一阶/二阶矩也存在极小差异。没有把该 FAIL 直接改写成 PASS，而是新增只读失败
诊断器和 GAME target 审计器定位根因。结果证明：

- 初始模型、冻结参数、H、句首 boundary 和 617/617 帧 MIDI-P condition 全部 exact；
- `soundfile` 与 `torchaudio` 读取的 1,265,626 个 waveform 样本 exact；
- GAME durations、presence、classes、duration frames 和 valid mask exact；
- GAME 单独驻留 GPU 时，旧 cache 和当前 runtime posterior 可 exact；
- Alex 实际训练同时驻留 DiT、VAE、GAME 时，CUDA 浮点执行路径会让连续 posterior 出现极小差异，
  但离散 MIDI-P 不改变，且既有 exact-resume 证明同一训练环境内可复现。

因此 golden 合同明确分为：离散输入与所有冻结状态必须 bit-exact；实时 GAME 引起的连续 CKA、
P 更新和 Adam moments 采用代码中冻结的窄容差。第二版 comparator 结果：

```text
pass: true
exact: false
numerically_equivalent: true
compared tensors: 2,009
compared tensor elements: 911,639,144
P / EMA max abs diff: 5.960464477539063e-08
Adam exp_avg max abs diff: 7.897615432739258e-07
Adam exp_avg_sq max abs diff: 5.969195626676083e-08
```

完整小型报告只留服务器：

```text
${PRIVATE_PATH}
${PRIVATE_PATH}
${PRIVATE_PATH}
```

两个 3.65 GB checkpoint 未下载。新增/更新的三个诊断脚本分别为 7,520、11,079 和 8,426
bytes，均在传输前确认严格小于 100,000 bytes；旧 `train_v5p.py` 未修改。下一步按 v3 checkpoint
schema 重跑 Alex step-10、checkpoint audit 与 5+resume exact，再进入 60 秒样本门禁。

#### v3 step-10、checkpoint audit 与 5+resume：PASS

新增 4,343-byte 英文注释 runner，仅使用服务器已有 v2 fixture 和模型，在 GPU 7 顺序执行连续
10 step、独立 5 step、从 step 5 恢复到 step 10，并在每个终点运行 checkpoint auditor。runner
终态 `RC=0`，没有上传或下载 checkpoint。

continuous-10：

```text
final parameter SHA256: SHA256_REDACTED
supported pitch rows: 50
support frames: 3501
optimizer/scheduler/EMA step: 10/10/10
sampler cursor: (0, 10)
all checkpoint finite: true
vae_avg: 0.2195s
game_avg: 0.1042s
peak allocated/reserved: 9958.5 / 14218.0 MiB
frozen runtime: PASS
```

split-5 checkpoint 同样通过审计：40 个 pitch row、1,788 个支持帧、optimizer/scheduler/EMA 均为
step 5，cursor 为 `(0, 5)`。恢复后的 step 6--10 loss 与 continuous-10 逐步一致，最终参数 SHA256
同为 `b8423c...c1e2`，恢复终点审计全部通过。

`compare_v5p_alex_resume.py` 比较 checkpoint schema、model、EMA、optimizer、scheduler、训练
metadata、rank RNG/cursor/metrics、P support/history、初始权重/冻结指纹和 transition 字段。结果：

```text
tensor_count: 2013
tensor_elements: 911644218
numpy_array_count: 1
exact: true
```

只排除必然不同的 `args` 中 output/resume 路径。因此两个完整 checkpoint 的文件 SHA256 不同是
预期现象，不能反推内部训练状态不同。服务器报告与产物位于：

```text
${PRIVATE_PATH}
${PRIVATE_PATH}
${PRIVATE_PATH}
```

下一步：约 60 秒样本的 official VAE、runtime GAME、显存、边界和 checkpoint smoke。

#### 59.995 秒最长样本 full-graph：PASS

复用 V5-P 已冻结最长 Long，服务器原地 hard-link 音频并用 official VAE 构建 Alex v2 fixture；
未传输 372,984-byte 源 manifest、音频或 checkpoint。样本事实：

```text
source SampleId: 87941a243c48129fbc21
duration: 59.99503401360544s
segments / sentences: 3 / 9
official VAE frames: 1291
sentence starts: [0,112,250,393,536,655,795,943,1182]
fixture manifest SHA256: SHA256_REDACTED
```

数据审计通过；boundary auditor 确定性抽取 128 次 proposal，落到 3 个可达正数句首，128/128
次和 3/3 个 boundary 的 H 均逐帧等于 fixture，`all_equal=true`。

随后在 GPU 7 运行 official VAE、runtime GAME、H/MIDI-P、三层 CKA、backward、Adam、CPU EMA 和
full-state checkpoint 保存：

```text
FlowA / FlowB / CKA: 9.3107 / 8.9742 / 0.9314
HPUL / HNonzero: 213 / 506
vae / game: 0.6982s / 0.3471s
peak allocated / reserved: 16822 / 16922 MiB
frozen runtime: PASS
final parameter SHA256: SHA256_REDACTED
```

checkpoint auditor 确认 P 32,768 个值发生更新，16 个 pitch row、898 个 B 区支持帧，optimizer、
scheduler、EMA 均为 step 1，cursor `(0,1)`，全 checkpoint finite。checkpoint SHA256 为
`40467271...a0216a`，仅留服务器：

```text
${PRIVATE_PATH}
```

2,483-byte runner 经 `bash -n` 后上传；仍严格遵守单文件小于 100,000 bytes 的直传限制。下一步：
Alex 专用 Phase-A adjudicator、500-step 和 Phase-B transition smoke。

#### Alex Phase-A 500-step 与专用裁决：PASS

新增 `adjudicate_v5p_alex_phase_a.py`，不复用硬编码旧 V5-P schema/cache 的裁决器。它只接受
`v5p_alex_p_checkpoint_v3`、`v5p_alex_input_v2`、runtime GAME K4、共同初始 P/冻结指纹及同一
manifest，并输出 `v5p_alex_phase_a_adjudication_v2`。脚本为 6,669 bytes，本机/服务器
`py_compile` 和服务器 `--help` 通过。

GPU 7 在十条 v2 fixture 上完成单卡工程 calibration；该运行用于验证 P500 生命周期和转接，不
冒充真实英语训练：

```text
steps: 500
effective batch: 1
final parameter SHA256: SHA256_REDACTED
runtime batches including eval: 510
vae_avg / game_avg: 0.2023s / 0.1070s
peak allocated / reserved: 9961.4 / 14218.0 MiB
frozen runtime: PASS
```

step 300 与 step 500 分别通过 Alex checkpoint auditor。step 500 为 `run_state=complete`，P 的
32,768 个值发生更新，50 个 pitch row 获得支持；pitch+REST 总支持帧 174,083，optimizer、
scheduler、EMA 均为 step 500，cursor 为 `(49,10)`，全 checkpoint finite。

裁决器在 step-500 的共同 pitch support 上重算 step-300/500 指标：

```text
projected RMSE:              0.3574734628 -> 0.3425971270  PASS (required)
raw RMSE:                    0.1643518507 -> 0.1642782390  PASS (diagnostic)
raw cosine distance:         0.9754064679 -> 0.9752287865  PASS (diagnostic)
pitch geometry CKA:          0.4080590904 -> 0.4077596068  FAIL (diagnostic only)
decision: random_p_pass
```

冻结政策以 projected RMSE 为唯一 unlock 主判据；CKA 诊断项的小幅下降如实保留，没有改写门槛。
产物 SHA256：

```text
step300: SHA256_REDACTED
step500: SHA256_REDACTED
adjudication: SHA256_REDACTED
```

全部大 checkpoint 只留服务器。下一步使用上述 checkpoint/adjudication SHA 绑定运行 Alex
Phase-B joint transition step-1，并独立重建与审计 supported/unsupported/REST/PAD 转接结果。

#### Phase-B joint transition 与最终工程门禁：PASS

Phase-B 首次 step-1 训练本身成功，但历史通用 auditor 把 warmup 第一步的“FP32 权重必须逐位
改变”误当成必要条件。该步 LR 仅 `7e-9`，Adam state 已建立，但参数更新小于 FP32 舍入单位，
所以 runner 留下 `RC=1`。这是假阴性证据，不是训练失败；原始日志和报告保留，不删除、不改写。

随后使用同一 Phase-A step-500 checkpoint 与 adjudication，在物理 GPU 7 完成 joint step-10：

```text
FlowA / FlowB / CKA: 34.4570 / 37.5583 / 0.8025
LR: 7.00e-08
initial parameter SHA256: SHA256_REDACTED
final parameter SHA256:   SHA256_REDACTED
frozen runtime: PASS
```

Phase-A 转接独立审计确认：50 个已支持 pitch row 原样保留，205 个未支持 row 由冻结 kernel 填充，
REST 保留、PAD 固定为零；转接初始 P SHA256 为
`SHA256_REDACTED`。

step-10 暴露了旧 checkpoint auditor 的第二个错误假设：AdamW 只在参数实际收到梯度后创建 state，
且 `state["step"]` 是该参数的实际更新次数，不是无条件等于 global step。Alex joint loss 直接调用
`transformer.*`，不会经过 `midi_extractor.*`；`drop_midi=0.3` 又会条件跳过 P 与 `midi_proj`。
真实 checkpoint 的分布为：

```text
trainable parameter tensors:        947
optimizer states present / missing: 367 / 580
Transformer states present:         366 / 366
nonzero-moment states:              367 / 367
state step histogram:               step 5 -> 3, step 10 -> 364
```

580 个缺失 state 全部属于本次计算图未经过的 `midi_extractor.*`。step 5 的三项恰为
`midi_p_v4ph.embedding.weight` 与 `midi_proj.weight/bias`，三者计数一致；其余 364 个常驻
Transformer state 均为 step 10。新版 `v5p_alex_checkpoint_audit_v2` 因此冻结为：Phase A 继续
要求 P state 完整且等于 global step；Phase B 必须覆盖 P 和全部 Transformer、全部动量非零、
常驻路径等于 global step、三项 MIDI 条件路径计数一致且位于合法范围，允许未经过的
`midi_extractor.*` 没有 state。

对现有 6.35 GB step-10 checkpoint 原地反跑，无需重新训练，最终 `RC=0`：

```text
checkpoint audit: PASS
transition audit: PASS
checkpoint file SHA256: SHA256_REDACTED
checkpoint audit SHA256: SHA256_REDACTED
transition audit SHA256: SHA256_REDACTED
```

服务器报告位于：

```text
${PRIVATE_PATH}
${PRIVATE_PATH}
${PRIVATE_PATH}
```

最终收尾门禁：本地与服务器均为数据测试 11/11、训练静态/边界测试 6/6，共 17/17 PASS；
所有 Alex Python 入口 `py_compile`、服务器 runner `bash -n` 通过；旧 `train_v5p.py` SHA256 仍为
`SHA256_REDACTED`。本轮直传文件最大仅
15,902 bytes，大 checkpoint 始终留在服务器；若以后需要交换大文件，继续按已挂载
`${LOCAL_PATH}` 的流程中转并复核 SHA256。

至此 V5-P-Alex 的数据合同、脚本、Phase A、Phase-B transition、checkpoint/resume 和最长样本
工程门禁完成。当前仍没有 Alex 的真实英语数据，因此不得宣称英语音质、英语发音或收敛效果已经
验证；这些只能在 Alex 数据接入后另行进行。

#### 最终完成审计启动

本轮不直接沿用“工程门禁完成”作为整个任务已验收的结论，而是回到用户最初选定的校验编号
`1、2、3、4、7、8、9（60 秒样本）、10、14、15`，逐项恢复其准确含义并核对当前文件、服务器
产物、日志和哈希。旧 V5-P 在编号 15 中只作为只读参考，仍以冻结 SHA256 为强门禁。本轮只有在
每个编号都找到范围匹配的权威证据、且 Alex 训练入口和文档合同没有未解决矛盾后，才会把总目标
标记为完成；发现证据缺口则继续补测，不用已有 PASS 代替缺失项。

##### 最终完成审计中间记录

已从 `simply_jp/校验清单.md` 恢复编号原义：1 语法与导入、2 配置合同、3 数据/cache、4 单步
训练、7 GPU 7 checkpoint、8 resume、9 60 秒样本、10 参数更新与冻结、14 结构审查、15 旧版
oracle 回归。当前本地与服务器 `train_v5p_alex.py` SHA256 均为
`SHA256_REDACTED`；旧 V5-P 仍为冻结哈希。
服务器已用当前代码重新通过 compile/import/`--help`，并重新全量审计十条 v2 fixture：10 条、
221.9910 秒、4,776 latent frames、70 个句首、共同帧轴成立，报告 SHA256 为
`SHA256_REDACTED`。

逐项审阅时发现编号 14 只有 golden 与运行日志，缺少独立可重复的机器结构报告，因此新增
`audit_v5p_alex_structure.py`。它不修改训练入口，解析只读旧 V5-P 与 Alex AST，强制验证：

- `run_dit`、`compute_cka_loss_from_hidden` executable AST 完全相同；
- `compute_loss` 去掉各自数据解包并归一化 `cka_target/midi_p`、`token_stats/placement_stats` 后，
  loss、dropout、A/B slicing 与 CKA kernel AST 完全相同；
- official VAE 与 runtime GAME 位于 `torch.no_grad()`，P embedding 仅在显式
  `torch.enable_grad()` 内建立条件；
- `loss.backward -> gradient sync -> clip -> optimizer -> scheduler -> EMA` 顺序不变；
- checkpoint 只删除离线 cache 与无关 warmstart 字段、加入 runtime GAME schema，所有模型、EMA、
  optimizer、scheduler、rank RNG/cursor 和冻结 fingerprint 字段仍存在。

本地结构审计输出 `v5p_alex_structure_audit_v1` 且 `pass=true`。下一步将小脚本做双端 SHA256
同步，在服务器重跑；随后用当前 auditor 反跑 Phase-A step-10、最长 step-1、Phase-A step-500、
Phase-B step-10，并用当前比较器重算 exact-resume 和 golden 报告。

##### 最终完成审计：PASS

新增 8,934-byte `audit_v5p_alex_structure.py` 与 8,455-byte
`run_v5p_alex_final_acceptance.sh`，均严格小于 100,000 bytes；双端 SHA256 分别为：

```text
structure auditor: SHA256_REDACTED
acceptance runner:  SHA256_REDACTED
```

服务器在 tmux `v5p_alex_final_acceptance` 中从 UTC `2026-09-02T00:54:30Z` 运行至
`00:55:56Z`，最终 RC 为 0。runner 使用当前 Alex 训练脚本 SHA256
`SHA256_REDACTED`，只读旧 V5-P SHA256
`SHA256_REDACTED`，逐项结论如下：

| 编号 | 最终证据 | 结论 |
|---:|---|---|
| 1 | 当前代码 `py_compile`、同目录 import、三个 `--help`、11 项数据测试与 6 项静态测试 | PASS |
| 2 | 数据审计、四份当前 checkpoint v2 audit、runtime/manifest/model SHA 与 schema 绑定 | PASS |
| 3 | 10 条 v2 全量审计；runtime GAME 对 cached V5-P GAME 的 waveform、离散 P、257/128 posterior 与 repeat 全部 exact/compatible | PASS |
| 4 | GPU 7 已有 step-1/10 日志无 Traceback/OOM/NaN，loss/backward/optimizer/save 形成可审计 checkpoint | PASS |
| 7 | 当前 auditor 反跑 P-only step-10、最长 step-1、P500 与 joint step-10，SHA、finite、step/cursor/EMA 均通过 | PASS |
| 8 | continuous-10 对 5+resume-10 比较 2,013 tensors、911,644,218 元素和 NumPy RNG，`exact=true` | PASS |
| 9 | 59.995034 秒、1,291 frames；GPU 7 peak allocated/reserved 16,822/16,922 MiB，backward/save 无 OOM | PASS |
| 10 | P-only frozen fingerprint、VAE/GAME runtime frozen、joint 全部 366 Transformer 与 P state/nonzero moment、transition 行保护 | PASS |
| 14 | 专用 AST 审计确认 run_dit/CKA、loss/B-mask、no-grad/detach、更新顺序、EMA 与 checkpoint 语义 | PASS |
| 15 | 当前 comparator 重算固定 step-1 old/Alex：2,009 tensors，离散条件 exact，四个连续 tensor 在冻结窄容差内，`numerically_equivalent=true` | PASS |

统一机器判决：

```text
${PRIVATE_PATH}
schema: v5p_alex_final_acceptance_v1
selected gates: 1,2,3,4,7,8,9,10,14,15 -> PASS
pass: true
SHA256: SHA256_REDACTED
```

同目录 `artifacts.sha256` 已用 `sha256sum -c` 反校验：当前 Alex/旧 V5-P 源码以及全部 checkpoint、
data、GAME、resume、structure、golden、transition 和总判决 JSON 均为 `OK`。最终 GPU 7 为
18/24,564 MiB、利用率 0%，tmux 已终止。本机直接 import GAME auditor 会遇到 Windows
`torchaudio` DLL 不匹配，因此 GAME 权威门禁在实际训练服务器 conda 环境执行并通过；该本机环境
限制没有被记为本机 PASS，也不影响服务器训练入口。

最终验收范围冻结为 **V5-P-Alex 训练脚本与工程合同验收成功**。由于 Alex 尚未交付真实英语数据，
总报告明确记录 `english_quality_validated=false`；这不阻塞脚本交付，但禁止将本次结论表述成英语
音质、发音、收敛或正式模型训练已经验收。

#### Alex 独立 DataKit 打包启动

训练工程验收完成后，进一步审查“直接发给 Alex 的文件夹”发现当前仓库不能原样外发：

- `train_v5p_alex.py` 直接依赖 `src.YingMusicSinger.*`、official VAE、GAME、YMSP checkpoint 和
  本项目的 P/CKA adapter，只属于内部训练端；
- `build_v5p_alex_fixture.py` 还依赖 `h_alignment` 与 YMSP wrapper，只用于内部花丸 fixture；
- `audit_v5p_alex_data.py` 本身只依赖 Python 标准库和 NumPy，可以抽为独立校验器；
- 本地 `examples/v5p_alex_hanamaru_10` 是较早的 `v5p_alex_input_v1`，没有
  `sentence_start_frames`，不得放进 v2 外发包或描述为当前有效示例。

为消除 Alex 对 private/YMSP 代码与 VAE 权重的依赖，已重新核实 official VAE 的长度合同。冻结
配置 sample rate 为 44,100 Hz，Oobleck encoder strides 为 `[2,4,4,8,8]`，总倍率 2,048；对服务器
十条 v2 fixture、59.995 秒样本和 2,048 边界前/等于/后共八个实际 VAE probe，均满足：

```text
T = PCM WAV sample_frames // 2048
```

这不是按 20 Hz 的时长估算，而是冻结 encoder 的精确离散长度变换。外发 DataKit 将只接受
44.1 kHz、mono、PCM16 WAV，并用 WAV header 的 sample-frame count 计算 `T`；内部训练端仍实际运行
official VAE 并严格断言输出长度相等，形成第二层保护。若未来更换 VAE 或允许重采样，必须升级
DataKit schema，不得沿用此公式。

外发包边界冻结为：Python 3.10+、NumPy、`vocab.json`、纯 Python timebase/manifest/validator、
英文 README 和无版权合成 v2 示例。它不包含训练脚本、YMSP/GAME/VAE 源码或权重，也不实现
English G2P/alignment；Alex 继续使用自己的前端生成 timestamped H 与 MIDI 信息，DataKit 只负责
固定 ID 映射、帧化、manifest 和交付前校验。

#### Alex 独立 DataKit 最终交付

已完成可直接外发的干净目录和 ZIP：

```text
deliverables/release/V5P-Alex-DataKit-v1/
deliverables/release/V5P-Alex-DataKit-v1.zip
deliverables/release/V5P-Alex-DataKit-v1.zip.sha256
```

DataKit 版本为 `v1.0.0`，数据 schema 为 `v5p_alex_input_v2`。发布目录共 23 个文件，合计
133,945 bytes；最大单文件是 88,244-byte 的合成 WAV，ZIP 为 26,070 bytes，没有任何单文件或
传输产物超过 100,000 bytes。ZIP SHA256：

```text
SHA256_REDACTED
```

最终包包含英文 README、完整格式合同、依赖边界、`vocab.json` 与许可声明、timebase、manifest
生成器、dataset validator、checksum verifier、7 项单元测试和一条无版权合成 v2 示例；不包含旧
v1 花丸 fixture、`__pycache__`、训练入口、private adapter、YMSP/GAME/VAE 实现或任何权重。

发布前在隔离 venv 中使用 `--no-index` 完成依赖检查，没有联网下载：Python 3.13.7 与已安装的
NumPy 2.3.1 满足 `Python >= 3.10`、`numpy>=1.24,<3`。源目录、干净发布目录和 ZIP 解包副本均
通过 7/7 单元测试、示例完整审计与 22/22 内部 SHA256 反校验；另完成一条独立的
`create_manifest.py -> validate_dataset.py` round-trip。

Alex 可以解压后直接运行工具，但“直接使用”不代表 DataKit 替他完成英语前端。Alex 仍需用自己
的 G2P/forced alignment 与 MIDI 工具得到带时间的信息，并按合同生成最终 H/MIDI-P 数组；不需要
安装或下载本项目/YMSP 的特有训练代码。训练服务器继续负责真实 official VAE 二次长度校验、
runtime GAME CKA target 和全部训练逻辑。

#### Alex DataKit v1.1.0：真实花丸 v2 示例补充

用户复核后指出，v1.0.0 的合成样本不能替代此前明确要求的 10 条真实花丸 Alex-format 示例。
因此保留合成样本作为最小安装自检，同时从服务器权威目录只读导出：

```text
${PRIVATE_PATH}
source manifest SHA256:
SHA256_REDACTED
```

服务器原训练 auditor 与独立 DataKit validator 均确认：10 条、221.99102040816325 秒、4,776 latent
frames、70 个句首、H/MIDI-P 共同帧轴和全部资产 SHA256 通过。旧本地 v1 fixture 没有被升级或
复用；本次内容来自已经完成训练工程验收的 `hanamaru10_v2_sentence_safe`。

由于中转包为 16,700,813 bytes，没有使用 SCP、SSH stdout 或其他直传链路。服务器先生成
`tar.gz` 并上传阿里云盘 `${CLOUD_ARTIFACT}`，本机从只读 P 盘拉取；服务器、
云盘和本机归档 SHA256 一致：

```text
SHA256_REDACTED
```

DataKit 同时保留 byte-exact `source_manifest.jsonl` 与 standalone `manifest.jsonl`。后者只增加
`language="ja"` 和 `vae_contract="ymsp_official_vae_44100_ratio2048_v1"`；回归测试逐条删除这两个
字段后与服务器记录完全相等。validator 只允许带 `example_only=true`、`source_language="ja"`
双标记的内置日语示例例外；Alex 正式数据仍强制 `language="en"`。

新的最终交付为：

```text
deliverables/release/V5P-Alex-DataKit-v1.1.0/
deliverables/release/V5P-Alex-DataKit-v1.1.0.zip
deliverables/release/V5P-Alex-DataKit-v1.1.0.zip.sha256
```

发布目录 57 个文件、19,788,200 bytes；ZIP 为 16,819,566 bytes，SHA256：

```text
SHA256_REDACTED
```

源目录、干净发布目录和 ZIP 解包副本均通过 10/10 单元测试、合成 1/1 与花丸 10/10 manifest
审计、56/56 内部 SHA256 反校验；ZIP 不含 `__pycache__` 或 `.pyc`。v1.0.0 仅保留为历史首包，
对 Alex 的当前交付必须使用 v1.1.0。真实花丸录音只授权本次私下技术交接，不因进入 DataKit
而获得公开再分发许可。
