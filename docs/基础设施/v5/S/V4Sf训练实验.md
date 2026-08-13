# V4Sf 训练实验

## 定义

V4Sf 使用 V4f 的 Official base、官方 VAE、SOFA token、continuous SOME fuzz、CKA 与
优化配置，只把训练 waveform 换成 v3 20k CampPlus、50 diffusion steps 的 SVC 自克隆产物。

历史 V4f 的 `train_plus_v4d.py` 虽包装 Singer 为 DDP，却直接调用
`raw_model.transformer`，绕过 `DDP.forward`，各 rank 梯度没有可靠同步。V4Sf 不复制该
缺陷：自定义 loss 保持不变，在梯度裁剪前使用 V4vf 已验收的固定参数顺序、缺失梯度补零和
25MB bucket 显式 `all_reduce`。因此 V4Sf 的实际 effective batch 为 16。

这意味着 V4Sf 配方在工程上复刻 V4f 的预期四卡语义，但不能把历史 V4f checkpoint 当成
DDP 正常的严格单变量 control。若后续要把差异归因于 S waveform，需要补跑同一修复骨架的
原始 waveform control。

## 冻结配置

| 项目 | V4Sf |
|---|---|
| DiT 初始化 | `ckpts/YingMusicSinger_model.pt` |
| VAE | Official config/checkpoint |
| MIDI | continuous SOME fuzz，原 V4f MIDI teacher |
| CKA / FlowB | `0.7` / `2.0` |
| text CFG dropout | `0.15` |
| batch / grad accumulation / world size | `1 / 4 / 4` |
| effective batch | `16` |
| LR | `1.4e-5` |
| warmup / hold | `500 / 12000` |
| seed | `42` |
| max duration | `30s` |

训练入口与 launcher：

```text
${SERVER_ROOT}/YingMusic-Singer-Plus/train_plus_v4sf.py
${SERVER_ROOT}/YingMusic-Singer-Plus/run_sft_v4sf_smoke50_ddp.sh
```

当前入口 SHA256: redacted
`3a34fee880b310b9a812b36ca5ff5431d9fd8cf56241fa30a9c8f8d430307aea`；该版本在 smoke
执行版本上补充了配对 Eval RNG、`eval_only` 和正常销毁 NCCL process group，不改变已保存
step-50 的训练权重。launcher SHA256: redacted
`c0a84f5b7fa5732cc2396eee166f16111740d224bc16553e3c3130d98ce3a47d`。

## 数据与双 Eval

V4f strict-control train 共 private corpus count 条，逐条按 basename 映射到：

```text
${SERVER_ROOT}/final_sum_large_S_v3_20k_d50/audio/
```

生成的 train manifest：

```text
${SERVER_ROOT}/final_sum_large/pretreatment_text/tokens_SOFA_v4sf/train_tokens.json
SHA256: redacted
```

Eval 使用 V4f 同一批 201 条 test，文本、Phrases、顺序和其他字段完全相同：

- `original`：原 V4f `test_tokens.json` 与原始 waveform；
- `svc`：每条 test 独立执行同一个 v3 20k、source=target、50-step SVC，不进入训练集。

SVC test 输出：

```text
${SERVER_ROOT}/final_sum_large_S_v3_20k_d50_test/audio/
```

五卡转换完成 201/201、失败 0，墙钟 4分04秒。独立审计确认缺失、额外、格式错误和
非 Path 字段变化均为 0，最大时长差 11.61ms。SVC Eval manifest：

```text
${SERVER_ROOT}/final_sum_large/pretreatment_text/tokens_SOFA_v4sf_eval/test_tokens.json
SHA256: redacted
```

## 50-step smoke

四卡 tmux session` 正常完成并退出。训练阶段约 1.49s/step，日志确认
`Train=11031`、`Eval original=201`、`Eval svc=201`、effective batch 16；四卡全程没有
all-reduce 死锁、rank 崩溃或非有限 loss。

训练结束时顺序执行的首轮随机 Eval 为：

| Eval | Loss | FlowA | FlowB | CKA |
|---|---:|---:|---:|---:|
| original | 71.3165 | 23.6354 | 23.6081 | 0.6640 |
| svc | 69.4007 | 23.1294 | 22.8939 | 0.6906 |

V4f 的 Eval loss 会随机抽 timestep、reference split 和 CFG dropout；首轮两路顺序执行，随机
抽样不配对，不能直接用两者差值作结论。入口随后固定并恢复 Eval RNG，并从同一 step-50
checkpoint 使用 `eval_seed=1042` 重算：

| Paired Eval | Loss | FlowA | FlowB | CKA |
|---|---:|---:|---:|---:|
| original | 71.8692 | 23.8885 | 23.7563 | 0.6688 |
| svc | 71.2253 | 23.7497 | 23.5008 | 0.6771 |

50 steps 只证明训练、同步、保存和双 Eval 链路成立，不用于判断 S 优于 original。

## Checkpoint 验收

```text
${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v4sf_sofa_base_smoke50/step_000050.pt
bytes=7,293,356,610
SHA256: redacted

${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v4sf_sofa_base_smoke50/step_000050_final.pt
bytes=3,646,769,940
SHA256: redacted
```

final checkpoint 的 `global_step=50`；model 与 EMA 各 1,001 个 tensor，其中各 983 个浮点
tensor 全部有限。训练与 Eval tmux session

## 后续边界

- 扩大 V4Sf 步数前保留 original/SVC 双 Eval，并使用固定 `eval_seed`；
- 要形成 S 单变量结论，补跑 corrected-DDP V4f control，不能继续引用历史无效 DDP 的 V4f
  effective batch 16；
- 50-step checkpoint 是链路产物，不进入候选模型盲听排序。

## 正式 30k

正式任务从 Official base 的 step 0 独立启动，不从 smoke checkpoint 续训：

```text
tmux session ${SERVER_ROOT}/YingMusic-Singer-Plus/train_v4sf_30k.log
output: ${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v4sf_sofa_base30k/
```

启动后通过 step 50 时约 1.51s/step，四卡均有持续负载。正式配置为 30,000 steps、每
2,000 steps 保存、每 1,000 steps 同时执行固定 seed 的 original/SVC 双 Eval。

训练已正常完成 30,000 steps，最终约 1.61s/step；tmux session 30k 双 Eval 为：

| Eval | Loss | FlowA | FlowB | CKA |
|---|---:|---:|---:|---:|
| original | 2.2849 | 0.3020 | 0.9429 | 0.1388 |
| svc | 2.2358 | 0.2991 | 0.9216 | 0.1334 |

正式产物包括 `step_002000.pt` 至 `step_030000.pt` 的 2k 间隔 checkpoint，以及
`step_030000_final.pt`。24k checkpoint 已通过cloud storage分片校验后保存到本机：

```text
${LOCAL_PROJECT_PATH}
bytes=7,293,356,866
SHA256: redacted
```

30k full checkpoint 同样通过 14 个 512MB 分片的云端、本地逐片 SHA1 与重组后整文件
SHA256: redacted

```text
${LOCAL_PROJECT_PATH}
bytes=7,293,356,866
SHA256: redacted
```

## 24k 感知结论

用户对 V4Sf 24k 与 V4f 24k 进行感知比较后判断：**V4Sf 24k 整体略弱于 V4f 24k**。
当前没有进一步拆分到歌词、旋律、音色、自然度或表现力等具体维度，因此文档只记录整体判断，
不把它扩写成尚未提供的分项结论。

该结果说明 V4Sf 在 SVC test 上更低的 Eval Loss 没有在 24k 转化为整体听感优势。S 路线的
核心假设因此受到负面证据；30k 已独立保存，可作为同路线后期点继续比较，但不能用 30k 的
域内 loss 更低预设其听感会反超。

















