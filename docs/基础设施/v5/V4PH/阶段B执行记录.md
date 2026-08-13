# 阶段 B 执行记录

## 初始化契约

阶段 B 从通过距离裁决的 Phase A P500 `model_state_dict` 开始，不使用 Phase A EMA，也不继承
P-only optimizer、scheduler、EMA counter 或数据游标。

P500 支持 78 个 pitch rows。transition 保留这些已训练 rows 与 REST，将其余 177 个 pitch
rows 从结构化音高核补齐，PAD 固定为零。补齐后的 P SHA256: redacted
`4d33e4bdf539befd68e6d8e443ac5ebf8731a2445177b9d1f85831225a86e3fe`。H 的 V/PUL
embedding rows 原样保留，没有再次复制。

启动入口强制校验：

```text
P500 SHA256: redacted
d82eb26f31429c9ed52f5ad22c95162fa3f3dc4c60f15d9114cc2616bbd772ac

adjudication SHA256: redacted
890cf4db13f4ba16462618d82ccaecf1940394a71e7c4d5d246f742bfb60d74a
```

## 配方

```text
trainable: Official DiT + H/PUL + P embedding
loss: FlowA + 2 * FlowB + 0.7 * GAME-compatible CKA
CFG/dropout: audio 0.3, text 0.15, MIDI 0.3
LR: 1.4e-5
warmup: 500
hold: 12000
schedule: cosine decay after hold
steps: 30000 from a new step 0
effective batch: 16
VAE: official frozen VAE
GPUs: 0,1,2,3
```

## 门禁

单卡 fresh step 1 验证了 LR=0 的 warmup 起点；从该 checkpoint 严格 resume 到 step 2 后，
P 有 4,325 个值更新。checkpoint 审计确认 optimizer 覆盖 947 个模型参数张量，EMA step 为 2，
没有继承 Phase A 的 500-step counter。

四卡 fresh 10-step smoke 独立完成并通过 checkpoint 审计：

```text
checkpoint: ckpts/v4ph_phase_b_smoke_20260730/step_000010_final.pt
P changed values: 7526
optimizer parameter tensors: 947
EMA step: 10
supported/fill rows: 78/177
```

## 正式运行

正式 30k 从相同 transition 权重重新开始，不继承单卡或四卡 smoke：

```text
tmux session ${SERVER_ROOT}/YingMusic-Singer-Plus/train_v4ph_30k.log
training code SHA256: redacted
2aceefe0e70970d98cd447fa278d7cd949f1f20f9f1be1011759cf66e52e60f9
```

正式训练完整到达 step 30,000，tmux session DDP
异常。final checkpoint：

```text
${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v4ph/step_030000_final.pt
SHA256: redacted
80cf7251c19a0d803a7bb0ce9e363fd1700c102d10c52de93a486180aa73372c
```

final 独立审计通过：run state `complete`、EMA step 30,000、optimizer 覆盖 947 个参数张量、P
相对 transition 更新 14,870 个值，78/177 row 初始化契约保持不变。

固定 paired Eval 末段：

| step | Loss | FlowA | FlowB | CKA |
|---:|---:|---:|---:|---:|
| 21k | 2.1112 | 0.2809 | 0.8799 | 0.1009 |
| 24k | 2.1015 | 0.2792 | 0.8762 | 0.0999 |
| 27k | 2.0957 | 0.2781 | 0.8741 | 0.0992 |
| 29k | 2.0936 | 0.2778 | 0.8733 | 0.0989 |
| 30k | 2.0937 | 0.2778 | 0.8733 | 0.0989 |

29k 到 30k 已完全平台，没有末段反弹。相同 step 的 V4H 30k 为 Loss 2.1049、FlowA
0.2777、FlowB 0.8662、CKA 0.1354。V4PH 总 Loss 低 0.0112，FlowA 基本相同，FlowB 高
0.0071，CKA 低 0.0365。该比较只说明训练目标分解差异，最终价值仍需推理与听感裁决。

## 本地同构推理与评分集（2026-07-31）

30k final 已同步到本机：

```text
${LOCAL_PROJECT_PATH}
size:   7,294,122,397 bytes
SHA256: redacted
```

本机完整文件与 sidecar SHA256: redacted
的 `step_030000_ema_inference.pt`；1,002 个 EMA 张量与原 checkpoint 逐项 `torch.equal`，
0 mismatch。推理副本大小为 `1,823,693,929` bytes，SHA256: redacted
`44ceead42b62b5ae7bc6b3425d9dd1fa7c0b00fab4917fedc3bbb38e40d969dd`，原 checkpoint 未改动。

新增专用入口：

```text
${LOCAL_EXPORT_PATH}
```

入口严格校验 `v4ph_training_checkpoint_v1`、complete/30k、训练代码、H placement、model/VAE
config、official VAE、训练 GAME cache manifest、MIDI_P/GAME schema 与 GAME medium 模型哈希。
Singer 构图后先挂载 checkpoint 中的 `midi_p_v4ph.embedding.weight [257,128]` 再 strict load
EMA；PAD row 必须为零。

每组沿用 V4H 评分集的 A `+0.5s` 与 B `+1.0s`，VAE 仍按原 official 双声道前端分别编码；
GAME 对各区域做 L/R 算术平均后拼接，使用 commit
`4ad815c90dfe2442730f3fdc866fd23e737cbc97`、medium、K=4、base seed 20260730。GAME
class 经权威 cache adapter 做中心时间离散查询，不插值 class ID，再经 learned P embedding；A 区
embedding 清零。配置原有 `some_pretrain_fuzzdisturb` 只作为等长 128 维 tensor 传输接口，推理时
将无参数 fuzz 模块替换为 `Identity` 并断言逐张量恒等，因此不执行 sigmoid/丢帧，也不加载 SOME。

同一 27 组 CFG 3/1 各 27 条全部完成。两档均为 141 phone / 38 PUL / 0 exact Control；GAME/P
条件的 effective seed、拼接波形、P class、P embedding、notes/REST 与 H placement 逐组一致，
CFG condition mismatch=0。全量 PAD frame=0、A 区 MIDI 非零=0、GAME/VAE A-B 边界最大差 1 帧，
H dense text 相对 V4H 30k mismatch=0。54 条均为 44.1kHz 单声道、可解码、非静音，与 B 最大
时长差 `0.021406` 秒；CFG 3/1 无相同 WAV 哈希，provenance 错误为 0。产物：

```text
${LOCAL_EXPORT_PATH}
${LOCAL_EXPORT_PATH}
```

根目录 manifest、README 与评分表已加入 V4PH 30k，两档共 54 个空白评分项。用户决定暂不做
生成后 F0 客观评分；本阶段结论仅限工程链路同构和评分集有效，模型价值等待人工听感。

首次正式人工听感随后完成。用户结论为整体表现接近理想，没有需要否决路线或推倒重来的大
问题，只剩中小问题。该结果确认 Official base + H phone/PUL + GAME P500 的组合在实际听感
上成立，V4PH 30k 晋级为当前可用主候选；后续应冻结当前 checkpoint，逐项记录并处理剩余
问题，不再以训练 Loss 自动决定方向。

服务器 final checkpoint 与 SHA256: redacted

```text
${CLOUD_ARTIFACT}
${CLOUD_ARTIFACT} redacted
```

上传退出码 0，耗时 14 分 36 秒；云端目录列出 6.79 GB 主文件与 87-byte sidecar。

















