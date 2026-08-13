# TIME87 数据冻结

## 正式命名

用户口语中的“90H 数据”对应本分支的全量组，但正式名称保持：

```text
V4PH-TIME87H
```

全量 SOFA train 原始总量约 90.589h；满足 token/frame 门禁并严格保留全部 TIME65 记录后，
loader 在 `max_duration=30s` 口径下的可用上限为 `86.999726h`。因此不得把该冻结集合登记为
`TIME90H`。

## 冻结集合

| 集合 | 条数 | loader 有效时长 | manifest SHA256: redacted
|---|---:|---:|---|
| TIME30 control | 5,065 | 29.999712h | `d668456ba55d5aed89e3f3300f75c1068fa5c2b99b58641137270be8961cb8a3` |
| TIME65 control | private corpus count | 65.628939h | `2dcc7d0a2874800175bfb0b7b48fe017e1f377cd897547ba45260accd2d093ce` |
| TIME87 control | private corpus count | 86.999726h | `c3e1ce7f011f38ad5e70666211d07ce39c3316bd32d3fcfc4a0fffebfc31c3dd` |

TIME87 由全部 TIME65 记录加 3,535 条确定性选择的增量记录构成。独立审计确认：

```text
TIME30 Path set < TIME65 Path set < TIME87 Path set
```

共享 control 记录逐项未变化，TIME87 与固定 eval 的 Path 交集为 0。

## H 与 GAME 资产

| 资产 | 条数 | SHA256: redacted
|---|---:|---|
| H30 | 5,065 | `727d3ab4ef5c27252eeac7739b4f76935d57ad141e5be81e95001bd307a61de2` |
| H65 | private corpus count | `416bf7ca77ab68fd7d41b11f8d44890440b5067d441d64ea568fb8c13e38b13f` |
| H87 | private corpus count | `85127fe586663a96eba2d882823d20705dd2eccfe5df031da6fbe83f6d7c1d3d` |
| 固定 H eval | 201 | `78f06dc680947c7beb17435e88e48c24ac071126e5cfa853c1a55aa008ae48e0` |
| GAME train87 + eval manifest | 14,767 | `e1ceacd9ed7eb851c08ff81e175d69b754e6cdc0b63caec519ed9ab22d61d9df` |
| H config fingerprint | - | `380a61b485b8cf7c7335ee977bb2eb7378f3d66f3d428355e3751a6a457f8c60` |

H30/H65/H87 保持严格逐记录包含；control 与 H 的顺序、Path、Phrases 全部一致。GAME manifest
严格按 TIME87 + eval 顺序包含 14,767 个 entry，对应 14,761 个唯一 cache 文件。

## 2026-08-04 冻结复审

新增只读审计器 `tools/v4timetest/audit_time87_freeze.py`，在服务器 tmux session` 中完成复审。它没有生成或覆盖任何 manifest，而是：

- 复算 control、H、eval 与 GAME manifest 的全部冻结 SHA256: redacted
- 逐记录验证 TIME30/TIME65/TIME87 control 与 H 的严格包含和共享内容不变；
- 验证 TIME87 与 eval 重叠为 0；
- 按冻结顺序核对 GAME entry 等于 TIME87 + eval；
- 重新打开 14,761 个 GAME NPZ cache，检查 key、shape 与所有浮点数组有限。

所有合同均通过。审计报告位置：

```text
服务器：${SERVER_ROOT}${CLOUD_ARTIFACT}
本机：  ${LOCAL_PROJECT_ROOT}
SHA256: redacted
```

## 服务器交付路径

```text
${SERVER_ROOT}${CLOUD_ARTIFACT}
${SERVER_ROOT}${CLOUD_ARTIFACT}
${SERVER_ROOT}${CLOUD_ARTIFACT}
${SERVER_ROOT}${CLOUD_ARTIFACT}
${SERVER_ROOT}${CLOUD_ARTIFACT}
```

## 训练禁令

本次授权只覆盖数据准备和文档，不覆盖 Phase A、smoke、resume gate 或正式 Phase B：

- `training_authorized=false` 已写入冻结报告；
- `${SERVER_ROOT}${CLOUD_ARTIFACT}` 不存在；
- TIME87 管线在授权文件不存在时必须安全退出；
- 复审结束时不存在 TIME87 `torchrun` 或 `train_v4ph.py` 进程；
- 不得以数据已冻结为理由自动启动训练。

后续只有获得用户新的明确训练授权，才可创建授权文件并从独立 Phase A step 0 开始 TIME87 谱系。

















