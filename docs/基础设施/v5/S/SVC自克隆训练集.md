# SVC 自克隆训练集

## 目标

把当前target singer训练集的每条歌声音频输入target singer SVC，生成时长和内容对应的自克隆音频，
再以整套自克隆音频替换原 waveform 训练 Singer。

核心假设：SVC 可以把录音条件、残余噪声、音色漂移和部分唱法差异压到更统一的target singer分布，
使 Singer 更容易学习稳定身份与声学生成。核心风险是 SVC 同时压平有效唱法、引入固定伪影，
或把自身的发音、F0 和频谱上限固化为新训练集上限。

## S 的精确定义

```text
原始训练 waveform
  -> 固定 YingMusic-SVC checkpoint 与固定推理参数
  -> SVC 自克隆 waveform
  -> 时长、内容、F0、音质和身份审计
  -> 重建所有 waveform 派生资产
  -> 使用不变的 Singer 配方训练
```

首轮严格 S 对照不混入原音频，也不增加 SVC B-cond 或输出后处理。若以后测试原音频与
SVC 音频混合比例，必须作为 `S-mix` 新变量记录。

## 与既有 SVC 路线的区别

| 路线 | SVC 输出的位置 | 是否属于本分支 |
|---|---|---|
| S | 替换 Singer 训练集 waveform，A/B 都来自 SVC 音频 | 是 |
| C1/C2 | 作为 B 区完整或稀疏 condition | 否 |
| SM | 对 Singer 最终输出做 SVC 后处理 | 否 |
| replay | 少量 SVC/生成样本混入原训练集 | 否 |

## 数据基准

- 原始数据：`final_sum_large` 清洗集 15,429 条，约 98.86 小时。
- 当前句级对齐成功集：SOFA 15,414 条，约 92.46 小时，其中 train private corpus count 条、test 307 条。
- 首轮 S 只转换 private corpus count 条 train waveform；现有 token-safe 训练子集为 14,661 条，过滤规则保持不变。
- 正式 test 的 307 条原始 waveform 不参与 SVC checkpoint、reference 或参数选择，并作为真实分布主评测集保留。
- 如需转换 test，只能生成独立的 `S-test` 诊断副本，用于区分域内拟合与真实分布泛化，不能替代原始 test 主指标。
- 15 条 SOFA 失败样本不因 SVC 处理自动重新纳入。

## 必须冻结的规格

全量转换前记录并冻结：

- SVC 源码版本、checkpoint、使用 EMA/online 的选择和文件 SHA256: redacted
- source/target 自参考规则；
- ODE steps、CFG、随机 seed、F0 转调、响度和后处理参数；
- 输入输出采样率、声道、位深与响度策略；
- 处理粒度：当前切分段逐条转换，或在上游长录音转换后按原边界切分；
- 对超长、过短、静音和推理失败样本的确定性处理规则；
- 输出命名、manifest schema、失败清单和断点续跑规则。

已有 YingMusic-SVC 经验显示 3k checkpoint 听感最好、后续训练可能发生均值化退化；这只是
候选依据，不直接替代 S pilot 的配对试听和客观验收。

## 资产重建边界

| 资产 | S 路线处理 |
|---|---|
| waveform | 全量替换为 SVC 自克隆版本 |
| train/test split、歌词、语言 | 保持不变 |
| SOFA 句级文本与 token | 时长和边界门禁通过后复用；否则重新对齐 |
| VAE latent / audio codes | 必须从 SVC waveform 重新提取 |
| SOME / MIDI 输入 | 主路线从 SVC waveform 重新提取并记录差异 |
| 声纹、音质、F0、响度统计 | 按下游训练与诊断需要重新计算；不再作为本轮前置配对门禁 |
| A/B 区 | 都从 SVC waveform 构造，不允许 A 原始、B 自克隆的隐式混用 |

仅把 wav 路径替换而沿用旧 latent 不属于 S；它会造成 waveform 与训练 target 不一致。

## 阶段 0：SVC 候选审计

1. 找到可复现的 YingMusic-SVC 推理入口和现有target singer checkpoint。
2. 对候选 checkpoint 做同一批样本的匿名配对试听。
3. 冻结胜出 checkpoint、reference 规则和推理参数。
4. 保存源码、模型、配置和固定样本输出的 SHA256: redacted

### 正式模型选择（2026-07-27）

S 路线正式采用 **v3 20k CampPlus**，不再让 cosine 3k 与其并行跑全量数据：

```text
checkpoint:
${LOCAL_EXPORT_PATH}

SHA256: redacted
E8BF7C3525BB335D53FC04D9303B4E14B1A5791507E79EE08BF66EBF4AE8019B

config SHA256: redacted
9C29F7F04F3AFD250F4ACF703A218438CD18E953A1608E4C3F74A8AE652FB108
```

冻结推理方式：

- `my_inference.py`；
- 每条样本使用同一个文件作为 source 和 target，不使用固定 `target singer-平-voice.mp3`；
- 实时 CampPlus，不使用无 target 的 `inference_spkemb.py` 查表路径；
- F0 condition 开启；
- length adjust=1.0、inference CFG=0.7，沿用已验证的 v3 正式推理路径。

`my_inference.py` 中 target 同时进入实时 CampPlus、参考 F0、Whisper prompt condition、
style residual 和 mel prompt；参考音频最长取前 25 秒。因此 source=target 的精确定义是同样本
自重建，而不是只把自身的固定 speaker embedding 喂给模型。超过 25 秒的样本使用自身前
25 秒作 prompt，必须在长样本 pilot 中单独检查 25 秒附近及后段稳定性。

v3 的训练 checkpoint 包含训练期可学习 `spk_embedding`，但正式 `my_inference.py` 只加载
`net` 模型权重，不读取该查表 embedding；推理时与 cosine 3k 一样，从当前 target 实时计算
CampPlus。两者的实际差异是训练数据、学习率/步数和最终网络权重，不是“固定 embedding vs
实时 embedding”。

选择依据：既有 8 首矩阵评测中，v3 20k 为音色准确度标杆，使用 5,603 条/36.4 小时训练，
CFM cosine 约 0.996；cosine 3k 更清新。S 路线先采用用户指定的 v3 20k。cosine 3k 仅在
v3 pilot 暴露系统性厚重、高频损失或唱法压缩时作为回退诊断，不自动启动第二套全量转换。

### Diffusion steps 门禁

diffusion steps 不随模型选择自动冻结。历史记录混用了不同模型、参考音频和任务：v3 20k 的
既有最佳报告使用 100 steps，但其他 YingMusic/Seed-VC 记录显示 20 steps 后电音已明显消失，
50 到 100 的收益可能较小。S 全量约 92 小时，推理成本近似随 steps 线性增加，因此先在同一
批自参考样本上比较：

```text
20 / 50 / 100 steps
```

固定 v3 checkpoint、source=target、CFG=0.7、F0、fp16 和 seed，只改变 steps；同时记录
人耳、频谱、F0、时长和实测 RTF。选择满足感知门禁的最低 steps；若 100 对 50 有稳定可闻
收益才使用 100，全量运行前将最终值和证据写回本文。

### 服务器耗时实测（2026-07-27）

环境与协议：RTX 4090 单卡、Torch 2.6.0+cu124、v3 20k、source=target、fp16、CFG=0.7、
F0 开启、模型只加载一次、Hugging Face 强制离线命中既有缓存。使用 5/10/15/20/25/29 秒
六条真实 train 音频；每个样本分别运行 20/50/100 steps，计时包含特征提取、SVC 生成、
BigVGAN 解码和 WAV 保存，不包含一次性的模型加载。

| 音频时长 | 20 steps | 50 steps | 100 steps |
|---:|---:|---:|---:|
| 5.004s | 0.461s | 0.833s | 1.358s |
| 10.004s | 0.752s | 1.210s | 2.056s |
| 15.000s | 1.416s | 2.509s | 4.464s |
| 20.004s | 2.207s | 4.226s | 7.867s |
| 25.002s | 4.008s | 8.572s | 16.509s |
| 29.001s | 4.649s | 10.059s | 19.349s |

模型冷加载约 8.7-9.8 秒，每 worker 只付一次；峰值 allocated VRAM 约 2.51GiB。输出时长与
输入逐条相差不超过约 12ms。加载 v3 时出现 `_inv_sigma_running` shape mismatch 跳过警告，
耗时测量不受影响，但正式质量 pilot 前仍需确认该 buffer 在本机历史推理中是否同样被跳过。

自参考存在明显非线性：target prompt 会占用 30 秒上下文，20 秒样本约切 3 块，29 秒样本约
切 7 块。因此总时长不是简单的总音频小时乘单一 RTF。以下估算使用 SOFA train private corpus count 条、
90.589 小时的真实逐文件时长分布，在六个测点间逐文件线性插值后求和。

| Steps | 单卡纯推理 | 单卡计划值（+15%） | 当前5卡计划值 | 8卡计划值 |
|---:|---:|---:|---:|---:|
| 20 | 12.72h | 14.63h | 2.93h | 1.83h |
| 50 | 26.46h | 30.43h | 6.09h | 3.80h |
| 100 | 50.34h | 57.89h | 11.58h | 7.24h |

多卡值按一 worker/卡均匀分片，并加入 15% 调度、校验、失败重试和负载不均余量。测速时 GPU set、7 空闲，故“当前5卡”是当时可立即使用的资源快照；GPU set 有其他进程，不自动占用。
原始 JSON 已双端保留：

```text
服务器：${SERVER_ROOT}${CLOUD_ARTIFACT}
服务器：${SERVER_ROOT}${CLOUD_ARTIFACT}
本机：${LOCAL_PROJECT_ROOT}\TEMP\benchmark_results_10_20_29.json
本机：${LOCAL_PROJECT_ROOT}\TEMP\benchmark_results_05_15_25.json
```

### 全量执行规格与空间（2026-07-27）

用户选择 **5 卡 + 50 diffusion steps**。按实测与 15% 余量预计约 6.09 小时；实际启动前仍
需重新检查 GPU set、7 是否空闲，不把测速时的资源快照当作永久声明。

实测 SVC 输出为 44.1kHz、单声道、PCM16 WAV。private corpus count 条 train waveform 总时长
90.589 小时，对应：

```text
PCM payload：28,763,751,190 bytes
含 WAV header：约 28.76GB（26.79GiB）
建议 waveform 阶段预留：35GB
```

全量实现应写入唯一正式输出树，每条先写临时文件、校验后在同目录原子改名；不要再复制一份
完整 staging tree，否则双份 waveform 会接近 58GB。上述 35GB 不含后续重新提取的 VAE
latent、SOME/MIDI 和诊断特征缓存；这些资产必须使用独立目录和空间统计。服务器审计时根盘
仍有约 1.2TB 可用，当前空间不是阻塞项。

### 全量运行记录（2026-07-27）

正式任务于服务器时间 2026-07-27 04:30:26 UTC（北京时间 12:30:26）启动：

```text
tmux session rank2->2, rank3->3, rank4->7
配置: v3 20k, source=target, 50 steps, fp16, CFG=0.7, F0 on
输入: private corpus count 条 / 90.589 小时
输出根: ${SERVER_ROOT}/final_sum_large_S_v3_20k_d50/
音频: ${SERVER_ROOT}/final_sum_large_S_v3_20k_d50/audio/
主日志: ${SERVER_ROOT}/final_sum_large_S_v3_20k_d50/control/master.log
worker日志: ${SERVER_ROOT}/final_sum_large_S_v3_20k_d50/control/logs/
状态: ${SERVER_ROOT}/final_sum_large_S_v3_20k_d50/control/status/
失败清单: ${SERVER_ROOT}/final_sum_large_S_v3_20k_d50/control/failures/
shard: ${SERVER_ROOT}/final_sum_large_S_v3_20k_d50/control/shards/
```

五个 shard 使用 50-step 实测成本做 LPT 均衡，分别约 3,021-3,022 条、预测纯推理
5.293 小时。启动前 GPU 均为 18MiB/0%，模型、配置和 manifest SHA 校验通过，private corpus count 个
源文件全部存在且 basename 无重复。

首批稳定性验收：2分22秒时主日志为 105/private corpus count、失败 0；随后输出目录已有 129 条、约
230MB，全部由 worker 在原子改名前检查 44.1kHz、mono、PCM16 和输入输出时长差 <=50ms。
五卡进程显存约 3.7-4.4GiB，持续进入推理；滚动 ETA 约 5小时18分。

runner 纪律：

- 每个 worker 只加载一次模型，Hugging Face 强制离线使用已验收缓存；
- 每条使用基于源路径 SHA256: redacted
- 用短名符号链接规避 source=target 导致的双长文件名；
- 临时 WAV 在同文件系统生成，校验后原子改名到正式 basename；
- 已存在且校验通过的文件在恢复时自动跳过；
- 失败不回退原音频，写入独立 JSON；连续 3 条失败时对应 worker 自动停止；
- 主日志每 60 秒汇总完成数、失败数、运行时长和 ETA。

查看：

```bash
tmux session -f ${SERVER_ROOT}/final_sum_large_S_v3_20k_d50/control/master.log
```

若服务器或任务中断，先确认五张 GPU 无残留进程，再用相同 session 名重启；runner 会重新
加载模型、校验并跳过已有正式产物。恢复启动时主日志使用追加模式，不能覆盖历史记录。

### 全量完成与独立验收

任务于服务器时间约 2026-07-27 09:44 UTC（北京时间约 17:44）正常完成：

```text
主日志终态: FULL RUN COMPLETE
完成: private corpus count / private corpus count
失败: 0
跳过: 0
总墙钟: 5小时13分32秒
tmux session 均恢复 18MiB / 0%
```

完成后使用独立脚本重新读取 manifest 与全部 private corpus count 个 WAV header，未复用 worker 状态作为
验收结论：

| 项目 | 结果 |
|---|---:|
| manifest / 输出 | private corpus count / private corpus count |
| 缺失 / 额外 | 0 / 0 |
| 采样率、声道、位深错误 | 0 |
| 时长差 >50ms | 0 |
| 时长差 P50 / P95 / max | 8.10ms / 11.51ms / 11.83ms |
| 输出总时长 | 90.5631h |
| 总大小 | 28,756,766,442 bytes（28.7568GB / 26.7818GiB） |

审计产物：

```text
${SERVER_ROOT}/final_sum_large_S_v3_20k_d50/control/full_output_audit.json
SHA256: redacted
```

该独立验收直接证明全量数据完整、格式一致、可恢复链路正确。

### 用户质量门禁裁决

用户基于既有 YingMusic-SVC 的长期使用经验，确认该 SVC 路线稳定性充足，裁决本轮不再要求
额外的原音频/SVC 分层配对试听、歌词/F0 或固定伪影门禁。该裁决不是“本轮配对听评已完成”，
而是明确豁免该门禁；不得在后续文档中改写成新增客观或人耳证据。

因此本轮 private corpus count 条 waveform 在全量结构审计通过后正式接受，可直接进入 VAE latent、
SOME/MIDI 和其他 waveform 派生资产重建，再启动同配方 Singer 单变量对照。后续若训练或
生成暴露系统性 SVC 伪影，仍可回到本产物定位，但不作为当前前置阻塞。

## 阶段 1：配对 pilot

pilot 不直接随机抽样，应覆盖：

- Gold/正常歌唱样本；
- 短、中、长时长；
- 高低音、强弱声、快速咬字、长音、颤音和滑音；
- 轻柔、雾感、高噪、残余伴奏和边界困难样本；
- 只从 train 侧选择 pilot；正式 test 不用于调参或门禁选择。

至少输出原始与 SVC 一一配对的 manifest，并检查：

- 输出采样率、声道、样本数和时长；
- 是否存在整体时间拉伸、局部漂移、首尾截断或新增静音；
- 歌词可懂度与错字是否恶化；
- F0、voicing、音符起止与长音是否保持；
- target singer身份是否更集中；
- 雪花声、电音、浴室感、毛刺和高频损失；
- MERT layer1 等表示下的唱法多样性是否明显塌缩。

允许为了容器一致性做确定性的尾部 crop/pad，但 crop/pad 不能掩盖内部时间拉伸。

### Pilot 通过条件

- 所有纳入样本能确定性复现，失败和过滤原因完整落盘；
- 输出可规范化为与输入逐样本等长，且无系统性内部时间漂移；
- 文本和旋律没有出现系统性退化；
- 人耳未发现贯穿样本的固定 SVC 伪影；
- 身份统一收益足以抵偿唱法多样性与频谱损失。

数值阈值在首轮 pilot 统计原始分布后冻结，不先用任意阈值替代配对证据。

## 阶段 2：全量转换

- 使用 `tmux` 和独占输出目录，支持断点续跑且不覆盖原始数据；
- 日志实时打印完成数、失败数、吞吐、已运行时间和 ETA；
- 每条记录输入/输出路径、时长、样本数、参数 schema 和 SHA256: redacted
- 完成后校验 manifest 行数、重复/缺失、音频可解码性和有限值；
- 对失败样本不静默回退原音频，统一进入失败清单并决定重跑或剔除；
- 全量资产验收后再重提 VAE latent、SOME/MIDI 和统计特征。

## 阶段 3：Singer 单变量对照

S 是数据变量。正式比较必须选择同一个 Singer 配方，同时固定：

- DiT 初始化来源；
- MIDI 表示与 CKA 定义；
- VAE、SOFA token、A/B 构造和 CFG dropout；
- seed、batch、LR、schedule、步数和 checkpoint 节点；
- 推理输入、T1、CFG、ODE steps 和评价集。

对照只允许：

```text
Control：原始target singer waveform 派生训练集
S：      SVC 自克隆 waveform 派生训练集
```

先做短程 checkpoint 轨迹和固定样本感知门禁；没有人耳改善时不因 loss 更低而扩到长训练。
S 与 P 组合应在 S 单变量证据成立后另开 `SP` 记录，不能把两项同时变化写成 S 结论。

## 评价轴

- 文本：PER、长多句稳定性和具体误读位置；
- 旋律：F0-CORR、voicing、音符起止、颤音与滑音保留；
- 身份：target singer相似度与跨样本一致性；
- 唱法：强弱、轻柔、长音、咬字和风格多样性；
- 声学：高频、噪声、浴室感、电音、雾感和局部崩坏；
- 上限：直接 SVC 输出、Control Singer 输出和 S Singer 输出三者同时盲听。

## 停止条件

- SVC pilot 出现系统性歌词、F0、边界或时长破坏；
- 固定伪影或高频损失在直接 SVC 输出中普遍存在；
- 唱法多样性明显塌缩，且身份收益不足以补偿；
- 全量转换无法确定性复现或失败样本被隐式混回原音频；
- S Singer 只复刻 SVC 上限，未相对同配方 Control 获得净收益；
- loss 改善但固定人耳集持续退化。

## 当前待定项

- 段级转换还是上游长录音转换；
- pilot 样本清单与统计后阈值；
- 首轮 Singer 对照采用的具体 f/P、VAE 和初始化配方；
- 服务器输出、缓存和最终归档位置。

## 服务器环境审计（2026-07-27）

只读检查确认服务器已有 YingMusic-SVC 训练环境与target singer模型资产：

- 代码/产物目录：`${SERVER_ROOT}/YingMusic-SVC`；
- 实际训练环境：`${SERVER_ROOT}/.conda/envs/yingmusic_plus`，Python 3.10.20、Torch 2.6.0+cu124，CUDA 可用；
- 基础模型：`YingMusic-SVC-full.pt`，约 698MB；
- target singer产物：`output_models/target singer_L1L2_30e/`；
- checkpoint 从 step 0 到 43,500，每 500 step 保存，包含候选 step 3,000；
- BigVGAN、Whisper-small、RMVPE 与 CAMPPlus 模型缓存均存在；
- Torch、Torchaudio、Librosa、Transformers、DAC、Hydra 和项目核心模块导入通过，`pip check` 无冲突。

当前服务器目录是训练精简包，不是完整推理仓库，缺少：

```text
my_inference.py
my_infer.sh
mm4.py
configs/
Remix/
utils/
```

本机完整仓库位于 `${LOCAL_EXPORT_PATH}`，包含上述推理入口。正式 pilot 前应只同步
必要的推理文件到服务器独占位置，先做 `--help`、模型加载和单条真实自克隆 smoke；不得直接
安装整份 requirements 覆盖现有可用环境。`resemblyzer` 与 `funasr` 当前未安装，但现有
`my_inference.py` 核心路径不直接依赖这两项，是否需要安装以实际 import/smoke 为准。

















