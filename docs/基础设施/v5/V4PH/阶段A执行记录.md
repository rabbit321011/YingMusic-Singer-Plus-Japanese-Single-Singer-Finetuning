# 阶段 A 执行记录

## 裁决边界

阶段 A 只验证随机 P embedding 能否在冻结 Official base 下学入结构化音高坐标。500 step
不足以形成有意义的成品听感，因此本阶段不设置试听或听感门禁，只依据 P300 到 P500 的
结构化音高核距离变化裁决。

## 数据与开跑门禁

- GAME medium K=4 离线 cache 共 11,232 条，对应 train private corpus count、test 201；
- 11,226 个唯一 cache 文件，6 条重复音频复用内容缓存，总大小 1,301,839,760 bytes；
- 共 821,860 个 note，其中 voiced 727,701、REST 94,159；
- 本机和服务器全量 cache 审计均通过，tar 的 SHA256: redacted
  `44ede76d758bc97ec868a5ea3ec7e14fd84640527ef101f0d1ff645ad56ff4c8`；
- H train/test manifest 与 cache 一一解析，缺失 cache 0、缺失音频 0；服务器原生
  `HDataset + torchaudio` 首、中、末样本读取通过；
- 本机 identity `midi_proj` 图探针与服务器 Official-base `midi_proj` 图探针均通过；
- 四卡 10-step smoke 完成，checkpoint 审计通过，P 的 32,768 个值均更新，PAD 保持零，
  optimizer 只含 P embedding。

## 正式运行

```text
init: Official base + fixed-seed matched-random P
placement: H phone/PUL
teacher: GAME medium K4 offline cache
trainable: P embedding only
loss: FlowB + 0.7 * GAME-compatible CKA
dropout/CFG: disabled
LR: constant 1e-4
effective batch: 16
steps: 500
GPUs: 0,1,2,3
```

正式运行完整结束，step 100、300 和 final 500 checkpoint 均已保存。paired eval loss 从
step 100 的 31.7619 降到 step 300 的 30.8744，再降到 step 500 的 30.1299。该 loss 只
作为训练健康指标，不参与初始化裁决。

服务器产物：

```text
${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/v4ph_phase_a/step_000100.pt
${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/v4ph_phase_a/step_000300.pt
${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/v4ph_phase_a/step_000500_final.pt
${SERVER_ROOT}${CLOUD_ARTIFACT}
```

final checkpoint 为 `complete`，冻结参数指纹保持
`538e555ef4e02c1e6d21777f15b999a614d87f4bc9e6dc0e3e1b92c0b3fd1133`，支持 78 个
pitch rows、2,263,270 个 pitch frames。

## 距离结果

P300 与 P500 统一在 P500 的 78-row 支持集合上重算：

| 指标 | P300 | P500 | 方向 |
|---|---:|---:|---|
| projected RMSE，主指标 | 0.339379013 | 0.327423304 | 改善 |
| raw RMSE | 0.164029762 | 0.164010108 | 改善 |
| raw cosine distance | 0.972181201 | 0.972199142 | 反向 0.000017941 |
| pitch-geometry CKA | 0.364585549 | 0.364971489 | 改善 |

主指标下降 0.011955708，约 3.52%。四项中三项改善；唯一反向项是 raw cosine distance，
幅度约 `1.79e-5`。当前裁决脚本执行冻结的“全部检查严格改善”规则，因此输出
`random_p_fail_use_kernel` 并以退出码 2 结束。

这是严格规则的边界结果：主指标有明确改善，raw cosine 的反向量很小。用户随后明确裁决
随机 P 通过，采用 P500；正式规则修订为 frozen Official `midi_proj` 后的 projected RMSE 是
必需主指标，其余距离完整保留为诊断项。

v2 pass 报告没有删除或改写 raw cosine 反向结果，固定契约为：

```text
P500 checkpoint SHA256: redacted
d82eb26f31429c9ed52f5ad22c95162fa3f3dc4c60f15d9114cc2616bbd772ac

adjudication:
${SERVER_ROOT}${CLOUD_ARTIFACT}

adjudication SHA256: redacted
890cf4db13f4ba16462618d82ccaecf1940394a71e7c4d5d246f742bfb60d74a
```

报告 schema 为 `v4ph_phase_a_distance_adjudication_v2`，decision 为 `random_p_pass`；Phase B
必须同时校验上述两个 SHA256: redacted

















