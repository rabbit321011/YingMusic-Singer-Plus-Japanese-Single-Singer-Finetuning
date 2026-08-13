# GAME 接入设计

## 结论

[OpenVPI/GAME](https://github.com/openvpi/GAME) 是 SOME 的官方后继歌唱 MIDI 提取器，
其输出语义比 SOME 更接近 V4Pf 所需的音符骨架。GAME 不能通过 257 个内部 pitch bin 的
class ID 直接接入 P；正确边界是读取官方推理张量 `durations`、`presence`、`scores`，再由
V4Pf 独立 adapter 映射到既有 P pitch/REST/PAD schema。

当前只冻结离线探针设计，不替换 SOME、不修改 MIDI_P schema，也不进入 P-only 训练。

## 官方接口事实

GAME 1.0 在 44.1kHz 音频上使用 80-bin、`0-8000Hz` log-Mel，hop 为 441 samples，即
100Hz。模型由 encoder、D3PM segmenter 和 note estimator 组成：

```text
waveform
  -> encoder
  -> D3PM boundaries [B,T]
  -> regions [B,T]
  -> estimator logits [B,N,257]
  -> durations [B,N], presence [B,N], scores [B,N]
```

- `durations` 是每个 region 的秒数，由 100Hz region map 统计得到，不是独立回归头；
- `presence` 不是独立类别，而是 `max(sigmoid(pitch_logits)) >= threshold`；官方默认阈值 0.2；
- `scores` 是 Gaussian-blurred centroid 解码后的浮点 MIDI pitch；
- 257 个内部 bin 是 `linspace(0,128,257)`，间隔 0.5 半音；bin 255/256 表示 MIDI
  127.5/128，不是 REST/PAD；
- note estimator 的 pool tokens 可跨 region 共享信息，因此 GAME 本身已经利用 note-level
  上下文；V4Pf 不再额外添加猜测性的连续平滑；
- D3PM 在多步推理中使用随机 boundary removal。同一音频要得到稳定 P，adapter 必须按音频
  key 固定 RNG seed，并单独审计跨 seed 边界波动。

官方 `extract` CLI 的 MIDI/CSV callback 会跳过 `presence=false` 的 region，只保存有声音符。
CSV 中的时间空隙虽可间接重建休止，但会丢失原始 region 语义。因此 V4Pf 不以最终 CSV/MIDI
作为正式接口，而直接消费 Python 推理张量；CSV/MIDI 只用于人工审计。

官方资料：

- [README](https://github.com/openvpi/GAME/blob/main/README.md)
- [技术报告](https://github.com/openvpi/GAME/blob/main/ALGORITHMS.md)
- [推理模型](https://github.com/openvpi/GAME/blob/main/inference/me_infer.py)
- [推理 API](https://github.com/openvpi/GAME/blob/main/inference/api.py)
- [解码实现](https://github.com/openvpi/GAME/blob/main/modules/decoding.py)
- [训练增强](https://github.com/openvpi/GAME/blob/main/training/augmentation.py)
- [配置](https://github.com/openvpi/GAME/blob/main/configs/midi.yaml)

## 与当前问题的关系

GAME 官方明确面向 dirty/separated voice。训练增强包含音乐/环境录音混入、RIR 混响、colored
noise 和时频遮挡；natural noise 数据来源包括 MIR-1K、MusicNet、MUSDB18-HQ 等。官方
dirty evaluation 是在同歌手 held-out 数据上施加同类破坏，不是本项目 MSST stem 的直接
评测，因此只能证明设计方向匹配，不能替代 62 条同源人耳 A/B。

官方三种规模中，medium 在 clean 和 dirty 汇总指标上均最强；large 并不优于 medium。
首轮使用 GAME 1.0 medium，不因 small 的 hidden dim 恰好为 128 而选择 small：GAME hidden
feature 不进入 YingMusic，模型内部维度与 P embedding 维度无关。

## 映射语义

V4Pf 保持现有定义：

```text
class 0..254 = MIDI 0..127，0.5 半音一个 class
class 255    = REST
class 256    = PAD
embedding    = 128 dimensions
```

note-level 映射顺序：

```python
valid = durations > 0
voiced = valid & presence
rest = valid & ~presence

classes[rest] = 255
classes[~valid] = 256
classes[voiced] = torch.round(scores[voiced] * 2).long()
assert ((classes[voiced] >= 0) & (classes[voiced] <= 254)).all()
```

不得对 GAME class ID 直接复制，不得把 presence=false 写成 pitch 0，也不得将 GAME 的
127.5/128 静默 clamp 到 127。超范围 voiced score 在离线预处理阶段隔离样本并记录原始值；
首轮门禁要求越界数为 0。正式生产是否允许显式 fallback 在真实分布出现后另行裁决。

## 时间展开

GAME note duration 先转为整数 100Hz region frames。真实探针确认 Mel/float32 边界可能使
region 总长与 `round(audio_duration * 100)` 相差 1 frame；允许并记录最多 ±1 个 10ms frame，
超过该范围立即失败。原生试听轨直接按 region frames 展开。

进入 DiT/VAE 时间轴时，不对离散 class 或 pitch 数值做 linear interpolation。对每个目标帧
中心时间 `t=(j+0.5)/target_rate`，在累计 note edge 中用 `searchsorted` 查找所属 region，再
复制该 region 的 pitch/REST class。batch 外部补齐才写 PAD=256。最后一个 edge 以权威音频
duration 收口，并记录 GAME 10ms 量化带来的时长误差。

## 运行边界

正式生产仍建议 GAME 使用独立环境。当前本机默认 Python 3.13 是 CPU-only torch，不适合
GPU 推理；试听探针复用现有 AISVCs venv 的 GPU torch/Lightning，并把缺失依赖安装到独立
TEMP target，没有修改原 venv。本机 RTX 5070 Ti Laptop 12GB 已以 medium、batch=1、bf16
完成 62 条，最高峰值显存仅 450.22MiB。

正式路径分为两个进程：

```text
GAME env:
  audio -> pinned GAME medium -> raw note cache

YingMusic env:
  raw note cache -> canonical GAME-to-P adapter -> P classes -> 128d embedding -> DiT
```

训练集离线缓存，避免每个 DDP worker 在线加载 GAME 和执行多步 D3PM。推理端调用同一版本
的 GAME 与同一 adapter；缓存只是性能优化，不允许另写一套量化逻辑。

每份 cache 至少记录：

- audio SHA-256 和权威 duration；
- GAME git commit、model scale、checkpoint SHA-256；
- Mel 配置、language ID、boundary/presence threshold、D3PM steps 和固定 seed；
- 原始 `durations/presence/scores`，以及 adapter/schema version；
- 100Hz region 总帧数、越界统计和时间闭合误差。

首轮固定官方代码 commit `4ad815c90dfe2442730f3fdc866fd23e737cbc97`。GAME 1.0 medium
release zip 的官方 SHA-256 为
`8c5b3e531e2905b935e664e2f533921cd637243770fab5282413bdb5051ca60c`。
代码为 MIT；官方 release 声明模型文件为 CC BY-NC-SA 4.0，后续分发和商业用途必须单独
检查许可证边界。

## 与 Official Base 和 CKA 的关系

GAME 只决定离散 P class，不向 `midi_proj` 输入 hidden feature。P embedding 仍输出 128 维，
因此 Official base 的 `midi_proj: Linear(128,128)`、P-only 校准和 checkpoint 结构不因 GAME
medium 的 256 维内部 hidden size 而改变。

不得把 GAME encoder/estimator hidden feature 投影后直接送入 `midi_proj`，否则重新引入音频
细节泄漏，并把 teacher 架构耦合进推理条件，偏离 P 实验定义。

raw SOME CKA 与 GAME-to-P adapter 正交。首轮 GAME 试听不能用 CKA；若 GAME 晋级，P-only
阶段优先 FlowB-only，先验证 P embedding 能否适配 Official base。联合阶段是否保留 raw SOME
CKA 仍作为独立消融，不用 GAME hidden 替换 CKA target，也不因更换 teacher 静默改变既有
训练配方。

## 首轮 A/B 门禁

1. 使用同一 62 条音频，保留target singer 20 条与非target singer MSST 42 条的域标签；
2. GAME medium 已在少量困难样本比较 K=4 与 K=8；首轮冻结 dirty 官方最优 K=4；
3. 按音频 key 固定 seed；另选困难样本跑三个 seed，报告 boundary/note 数量和 pitch 波动；
4. 生成 `Original / SOME native piano / GAME native piano` 同页试听；先裁决 teacher，不混入
   21.53Hz 重采样变量；
5. GAME native 通过后，再生成 GAME model-rate piano，单独验证时间展开；
6. 分域记录低音崩坏、错音、REST、重复切音和边界问题，不用全局中位音高替代人耳；
7. 审计 62/62 完成、输出有限、时间闭合、越界 0、同 seed 可复现、所有引用存在；
8. GAME 胜出后才升级正式 P teacher、预计算训练缓存并进入 P-only smoke。

不采用“GAME 与 SOME 一致率超过某阈值”作为晋级标准。SOME 正是待替换对象；决定性证据是
相对于原声的音符正确性，特别是用户已经听到的局部低音崩坏是否消失。

## 明确不采用

- 直接复制 GAME 257-bin class ID；
- 只读取官方 MIDI/CSV 并丢失 unvoiced regions；
- 超范围 pitch 静默 clamp；
- 对离散 class 做线性插值；
- 把 GAME hidden feature 当作 YingMusic MIDI condition 或 CKA target；
- 在 DiT DDP 训练 worker 内在线运行 GAME；
- 未固定 commit、checkpoint hash、threshold、D3PM steps 和 RNG seed 的缓存；
- 未通过 62 条人耳 A/B 就开始 V4Pf 训练。

















