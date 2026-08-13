# 无 A 区全段生成实验

## 1. 研究问题

V4PH 使用同一条target singer训练音频的前段作为 A 区真实 latent，后段作为 B 区生成目标。V4IPH
只检验一个问题：将 A 区时间恒定为 0 后，模型能否把target singer身份内化在权重中，并仅由歌词和
音符条件生成完整音频。

本实验不检验新的 VAE、数据增强、speaker token、style embedding 或训练配方。

## 2. 唯一变量

| 项目 | V4PH | V4IPH |
|---|---|---|
| `ref_len` | 由现有规则随机选择非空 A/B 切点 | 恒等于 `0` |
| A 区 | 非空，`cond=A latent`、MIDI 为零 | 不存在 |
| `cond` | A 区为真实 latent，B 区为零 | 全时间轴为零 |
| H/PUL | B 区使用，受 A/B 边界约束 | 全时间轴使用 |
| GAME MIDI_P | B 区使用，A 区为零 | 全时间轴使用 |
| FlowA | 非空区域 MSE | 数学上不存在，记录为精确 `0` |
| FlowB | B 区 MSE | 完整时间轴 MSE |

V4PH joint loss 为：

```text
FlowA + 2 * FlowB + 0.7 * CKA(B)
```

当 `ref_len=0` 时，V4IPH 的直接退化形式为：

```text
0 + 2 * FlowB(full timeline) + 0.7 * CKA(full timeline)
```

FlowA 跳过空张量计算，禁止通过对空切片求 MSE 产生 `NaN`。这不是新增实验变量，而是
`ref_len=0` 的必要算术定义。

## 3. 冻结不变量

- Official DiT base；
- V4PH phase A 已裁决通过的 P500 权重、补核策略与 SHA256: redacted
- private corpus count train + 201 test H manifest 及其 SHA256: redacted
- GAME medium K=4 离线 cache 及 GAME-compatible CKA；
- H/PUL token、renderer 语义和 fallback 规则；
- official VAE；
- `batch_size=1`、四卡、`grad_accum=4`、effective batch 16；
- `lr=1.4e-5`、warmup 500、hold 12,000、总步数 30,000；
- FlowB weight 2.0、CKA weight 0.7、text dropout 0.15；
- seed 42、eval seed 1042、确定性 DDP、fresh optimizer/scheduler/EMA；
- checkpoint、rank RNG、数据游标和 exact resume 契约。

## 4. 实现

为避免修改 V4PH 已冻结文件的哈希，V4IPH 使用独立文件：

| 文件 | 责任 |
|---|---|
| `package_v4c_finetune/train/train_v4iph.py` | V4IPH 训练、元数据、checkpoint 与 exact resume |
| `package_v4c_finetune/train/v4iph_contract.py` | `ref_len=0` 与全 B flow loss 的纯函数契约 |
| `package_v4c_finetune/h_alignment/placement_v4iph.py` | 允许 `ref_len=0` 的 H/PUL renderer |
| `package_v4c_finetune/train/run_v4iph_30k_ddp.sh` | 正式四卡 30k 入口 |
| `package_v4c_finetune/train/run_v4iph_smoke10_ddp.sh` | 保持 30k scheduler 的四卡 10-step smoke |
| `package_v4c_finetune/train/audit_v4iph_checkpoint.py` | schema、全 B 元数据、FlowA=0、P/optimizer/EMA 审计 |

本机与服务器上传后 SHA256: redacted

| 文件 | SHA256: redacted
|---|---|
| `train_v4iph.py` | `e386ace7a5f9d7f52f1b2c18ccf95d54d4c65ee831710249425669fa7bbbac73` |
| `v4iph_contract.py` | `a08bbb49b17dd1186f10963201d26f1bbd3843df8fd20cacc4d63d7baba9b043` |
| `placement_v4iph.py` | `96af5a627e8cbd758a2b89fb29091f8c3257d1f994a5dfecec6eb2ec5b336472` |
| `audit_v4iph_checkpoint.py` | `08ce449d8559512ba4da231d412ca8935a66a67b2d025582a11cfd7e8ebf72c8` |
| `run_v4iph_30k_ddp.sh` | `02f6c0259ea48d1c6162c43dbfedd4b229370b4011c85b58d51f328d39ac43f0` |
| `run_v4iph_smoke10_ddp.sh` | `6aa59a35f8a8ab1ffa5a32e66b95135bd0653101ead2828027e175e28001dc16` |

V4IPH checkpoint schema 独立冻结为：

```text
v4iph_training_checkpoint_v1
```

checkpoint 使用 `v4iph_training` 元数据键，不能与 V4PH checkpoint 互相 resume。

## 5. 强制门禁

每个开启 `--smoke_assertions` 的 batch 必须满足：

1. `ref_len == 0`；
2. `cond` 全张量非零元素数为 0；
3. 输入 DiT 的 MIDI 与完整 `midi_full` 逐值相等；
4. H/PUL 输出长度等于 VAE latent 长度；
5. 歌词 token 顺序、PUL/SEP 约束和现有 H 不变量继续成立；
6. FlowA 精确为 0，FlowB/CKA/总 loss 均为有限数；
7. 四卡初始和结束参数 SHA256: redacted
8. checkpoint schema、P500 transition、optimizer、scheduler、EMA、rank RNG 与数据游标通过审计。

## 6. 执行阶段

### Stage 0：本地静态与纯函数门禁

- Python 编译；
- V4PH/H 原 placement 回归测试；
- V4IPH `ref_len=0` renderer 测试；
- 正数 `ref_len` 下 V4IPH renderer 与 V4PH renderer 结构化结果完全相等；
- 全 B flow loss 有限且满足 `2 * FlowB(full)`。

### Stage 1：服务器单卡前后向

使用真实 manifest、GAME cache、P500 和 official VAE，验证一个完整 batch 的 encode、全 B
placement、DiT forward、backward 与 checkpoint 保存。不得用 mock tensor 代替正式链路。

### Stage 2：四卡 10-step smoke

使用正式 30k scheduler，`stop_after_step=10`，验证四卡显式梯度同步、FlowA 恒零、FlowB/CKA
有限、参数 rank 一致、final checkpoint 和独立审计。

### Stage 3：正式 30k

Stage 1/2 全部通过后才启动。长任务使用独立 tmux session LR。

## 7. 评价

正式比较对象为 V4PH 30k 与 V4IPH 同步 checkpoint。至少记录：

- target singer身份与音色稳定性；
- Whisper 内容与 Whisper+SOFA 错槽率；
- F0 跟随与跑调；
- 第一段开头稳定性；
- 极高音、混音输入和长音频崩坏；
- 自然度、唱法细节及是否出现平均化。

V4IPH 不再接收 A 音频，因此 A 只能作为离线 CAM++/人工音色评分参考，不进入模型输入。

## 8. 当前执行记录

- V4PH 训练脚本和 renderer 已恢复为独立原路径，V4IPH 不通过开关修改它们；
- V4IPH 独立实现已完成；
- 本地 `py_compile` 通过；
- H placement 15 项原测试通过；
- V4IPH renderer 3 项测试通过，其中正数切分与 V4PH 结构化结果完全相等；
- V4IPH ref/loss 5 项测试通过；
- 服务器 bash 语法、Python 编译、3 项 renderer 与 5 项 ref/loss 契约测试通过；
- 单卡真实 1-step 完成：正式 P500、official VAE、GAME cache、H/PUL、forward/backward 和
  checkpoint 保存均通过；`FlowA/FlowB/CKA = 0.0000/22.1270/0.8165`，checkpoint 位于
  `${SERVER_ROOT}${CLOUD_ARTIFACT}`；
- 四卡正式 scheduler 10-step smoke 完成，effective batch 16；step 10 的
  `FlowA/FlowB/CKA = 0.0000/32.0514/0.7978`，初始参数 SHA256: redacted
  `58e57a7b2e6345ccbead9f5b09af8b70b3f1028b9dd77a5ef70622500d168714`，step 10 为
  `2837f26d43e78de1164d92fbc357a30f92f4bf5799a03bdde01000765f668a83`；
- step 10 checkpoint 独立审计通过：schema=`v4iph_training_checkpoint_v1`、reference=`all_b`、
  支持/补核 P row=`78/177`、P changed values=`7776`、optimizer tensors=`947`、EMA step=`10`；
- 四卡 step 10→11 actual resume 通过，准确恢复 `epoch=0, batch_offset=40` 与 LR `2.8e-7`；
  step 11 再审计通过，P changed values=`8042`、EMA step=`11`；
- 首轮正式进程于 2026-07-31 10:46 UTC 启动，在 step 100、尚未产生正式 checkpoint 时主动
  停止：最终复核发现 metadata 尚未包含独立 `v4iph_contract.py` 的哈希。该轮空输出目录和日志
  保留为 `plus_ja_sft_v4iph_precontract_step100_20260731` 与
  `train_v4iph_precontract_step100_20260731.log`，不得作为正式训练续接；
- metadata 已新增 `contract_code` SHA256: redacted
  三份代码哈希；使用最终哈希重新完成四卡 10-step、审计和 step 10→11 actual resume，数值与
  前次门禁一致；
- 最终正式 30k 于 2026-07-31 10:59 UTC 从 P500 全新启动，未继承任何 smoke 或 precontract
  checkpoint；GPU set–3，tmux session 目录
  `${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v4iph/`；最终 run 的首个 step 50 日志为
  `1.72s/step`、ETA `14.3h`、`FlowA/FlowB/CKA=0.0000/26.7483/0.7907`，未发现
  traceback、NaN 或 NCCL 错误；最近一次检查到 step 100，`1.77s/step`、ETA `14.7h`、
  `FlowA/FlowB/CKA=0.0000/10.5718/0.7476`，进程仍正常运行。

## 9. 15k loss 中期分析（2026-08-01）

正式 run 已完成 1k–15k 的固定 paired Eval，FlowA 全程精确为 0，无 NaN、反弹或中断。
V4PH raw Loss 包含 FlowA，不能直接与 V4IPH 比较；下表使用
`V4PH comparable = V4PH Loss - V4PH FlowA`，即只比较 `2*FlowB + 0.7*CKA`。

| Step | V4IPH FlowB | V4PH FlowB | ΔFlowB | V4IPH CKA | V4PH CKA | V4IPH Loss | V4PH comparable |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 1.0646 | 1.0890 | -0.0244 | 0.1670 | 0.1846 | 2.2461 | 2.3073 |
| 5k | 0.9245 | 0.9327 | -0.0082 | 0.1189 | 0.1224 | 1.9322 | 1.9511 |
| 10k | 0.8977 | 0.9036 | -0.0059 | 0.1087 | 0.1099 | 1.8714 | 1.8842 |
| 12k | 0.8938 | 0.8985 | -0.0047 | 0.1084 | 0.1109 | 1.8635 | 1.8747 |
| 15k | 0.8870 | 0.8908 | -0.0038 | 0.1077 | 0.1062 | 1.8495 | 1.8560 |

观察：

- V4IPH Eval FlowB 从 1k 的 1.0646 单调降到 15k 的 0.8870，下降 0.1776（16.68%）；
  CKA loss 从 0.1670 降到 0.1077，下降 0.0593；总 loss 下降 0.3966。
- 12k 开始 LR 从 hold 进入 cosine decay，12k→15k FlowB 仍由 0.8938 降到 0.8870，尚未
  出现停止学习或 Eval 反弹。
- 相对 V4PH，V4IPH 的早期 FlowB/CKA 收敛更快；FlowB 优势由 1k 的 0.0244 缩小到 15k 的
  0.0038。15k CKA 则比 V4PH 高 0.0015，两者已经基本进入同一平台区。
- 15k raw Loss 看似比 V4PH 低 0.2920，其中约 0.2855 只是 V4PH 的 FlowA 项；扣除后真实可比
  优势为 0.0065（约 0.35%），不能把 raw Loss 差解释成取消 A 带来大幅提升。
- V4IPH FlowB 覆盖完整时间轴，V4PH FlowB 只覆盖随机切点后的后缀，评价帧集合并不完全相同；
  当前只能确认无 A 训练稳定且至少不劣，不能仅凭这条曲线证明生成质量优于 V4PH。

决策：继续跑满 30k，不按 loss 提前停止。按 V4PH 15k→30k 的后半程降幅粗略外推，V4IPH
30k Eval FlowB 可能落在约 0.869–0.873，但这不是 checkpoint 选择依据。正式感知比较至少保留
10k、14k、18k、22k、24k、30k，单独评价target singer身份、开头稳定性、错槽、F0 与美学，防止
loss 正常但听感出现 N 形变化。

## 10. 30k final loss 与审计结论（2026-08-01）

正式训练完整结束。final checkpoint：

```text
${SERVER_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v4iph/step_030000_final.pt
```

独立审计通过：`run_state=complete`、`reference_mode=all_b`、EMA step=30,000、
P changed values=14,870、optimizer tensors=947；training、contract 与 renderer 三份当前代码
SHA256: redacted

| Step | V4IPH FlowB | V4PH FlowB | ΔFlowB | V4IPH CKA | V4PH CKA | V4IPH comparable | V4PH comparable |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20k | 0.8808 | 0.8825 | -0.0017 | 0.1026 | 0.1025 | 1.8334 | 1.8367 |
| 25k | 0.8749 | 0.8755 | -0.0006 | 0.1015 | 0.0998 | 1.8208 | 1.8208 |
| 30k | 0.8727 | 0.8733 | -0.0006 | 0.0998 | 0.0989 | 1.8152 | 1.8159 |

最终观察：

- V4IPH 1k→30k Eval FlowB 为 `1.0646→0.8727`，下降 0.1919；CKA loss 为
  `0.1670→0.0998`，下降 0.0672；无 NaN 或 Eval 崩坏。
- 前 10k V4IPH 平均 FlowB 比 V4PH 低 0.0098；21k→30k 平均只低 0.0006。取消 A 的早期
  优势在后期基本收敛到同一平台。
- final FlowB 低 0.0006，CKA loss 高 0.0009，扣除 V4PH FlowA 后的可比总 loss 低 0.0007。
  这些差异均不足以构成模型选择结论。
- 20k→30k，V4IPH FlowB 继续下降 0.0081，V4PH 下降 0.0092；V4IPH 没有后期反弹，但也没有
  显示出更高的优化上限。
- loss 层面的正式结论：**V4IPH 可稳定收敛到与 V4PH 等价的训练健康水平；是否值得取消 A，
  必须由固定同输入的身份、错槽、F0、开头稳定性和人工听感裁决。**

loss 候选上 final 的 total/FlowB 最低，28k/29k 的 CKA 仅低 0.0001；差异没有感知意义。
听感包应至少保留 10k、14k、18k、22k、24k、30k，不因 final loss 极小优势单独选定 30k。

## 11. Final 云端归档（2026-08-01）

服务器 final checkpoint 和同名 SHA256: redacted

```text
${CLOUD_ARTIFACT}
${CLOUD_ARTIFACT} redacted
```

```text
server file size: 7,294,122,781 bytes
SHA256: redacted
```

上传退出码为 0，耗时 26 分 56 秒。云端目录复核显示 6.79GB 主文件和 87-byte sidecar；
sidecar 内容为上述 SHA256: redacted`step_030000_final.pt`。

















