# GAME 试听验证

## 目标

在完全相同的 62 条音频上，以人耳比较 SOME 与 GAME 的原生 note-level P，重点判断 SOME
试听中出现的局部低音崩坏、错音、REST 和边界问题是否由 GAME 改善。该门禁只比较 teacher，
首轮不混入 21.53Hz DiT/VAE 时间重采样变量。

## 固定条件

| 项 | 定义 |
|---|---|
| GAME source | OpenVPI/GAME commit `4ad815c90dfe2442730f3fdc866fd23e737cbc97` |
| model | GAME 1.0 medium，49,887,251 parameters |
| release zip SHA-256 | `8c5b3e531e2905b935e664e2f533921cd637243770fab5282413bdb5051ca60c` |
| model.pt SHA-256 | `e9904159fb0646e1a352b9d2bc74615547cfa3e32d45c7464d440ac142846d93` |
| features | 44.1kHz，80-bin Mel，`0-8000Hz`，100Hz |
| model scale | medium |
| D3PM | K=4，boundary threshold 0.2，radius 2 frames |
| presence | threshold 0.2 |
| language | target singer `ja=2`；非target singer未可靠标注语言，使用无语言条件 `0` |
| RNG | 每条音频由 ID 与 base seed `20260730` 生成固定 seed |
| P mapping | float MIDI → `round(midi*2)`；REST=255；PAD=256；越界立即失败 |

medium 官方 clean QER 在 K=4/K=8 分别为 `0.1271/0.1230`，dirty QER 分别为
`0.1334/0.1376`。本测试包含 MSST stem，首轮选择官方 dirty 最优 K=4。四条跨域探针中，
K=4 与 K=8 的全帧一致率为 `88.27%-99.14%`，说明复杂样本的 D3PM steps 是真实变量，
不能混用后再汇总结论。

## 运行环境

探针复用本机 `${LOCAL_EXPORT_PATH}` 的 Python 3.10、torch 2.11.0+cu128、
Lightning 2.6.1 和 RTX 5070 Ti Laptop GPU。缺失的 `colorednoise`、`h5py` 安装到独立
`TEMP/game_v4pf_deps`，没有修改原 venv。GAME 官方源码和模型保留在独立 TEMP 目录，
未进入 YingMusic 训练环境或 DDP worker。

62 条 GAME 推理合计 8.77 秒，最高峰值显存 450.22MiB。该数据只代表当前 1-51 秒试听
片段、batch=1、bf16 mixed inference，不外推到在线长音频吞吐。

## 时间与 schema 审计

- 62/62 完成，所有 voiced pitch 均可映射到 P class 0..254，越界 0；
- 41 条 GAME region 总长与 `round(audio_duration*100)` 完全一致；
- 21 条因 Mel/float32 取整少 1 个 10ms frame，全部显式记录；没有超过 10ms 的闭合误差；
- 最终钢琴轨按权威原音频长度生成，不通过伪造额外 note 修补 GAME 缺帧；
- 相同 seed 的 H009 在两个独立进程中 note CSV 逐字节一致；
- H009 的三个不同 K=4 seeds 与基准全帧一致率为 99.62% 和 99.14%，表明固定 seed 必须
  成为 cache schema 的一部分；
- 124 个 GAME native/model 钢琴 OGG 全部 44.1kHz、与原音频逐样本等长、有限且非静音；
- A/B 索引含 62 行、186 个音频引用，缺失引用 0。

## 试听入口

```text
${LOCAL_PROJECT_ROOT}\TEMP\v4pf_game_audition_k4_20260730_v2\index.html
```

每行只比较：

```text
Original | SOME native piano | GAME native piano
```

target singer 20 条与非target singer MSST 42 条继续分域裁决。优先判断 GAME 是否消除相对于原唱的局部低音
和口胡感；全局中位音高、与 SOME 的一致率、官方 dirty 指标均不能替代人耳。

## 当前裁决

用户试听结论：**GAME 基本可用，允许晋级正式 P teacher。**

训练主线已与 H 合并为 V4PH。下一门禁不是直接启动训练，而是验证 GAME estimator 概率经
adapter 后可等量替换 SOME 的 `[T,128]` CKA 接口；通过后才预计算训练 cache 并开始 P-only
smoke。

## 独占实现

```text
src/YingMusicSinger/melody/game_p_v4pf.py
test_game_p_v4pf.py
probe_game_p_audition_v4pf.py
build_game_some_ab_index_v4pf.py
```

现有 V4f、V4vf、V4Pvf、GAME 官方源码和公共训练入口均未修改。

















