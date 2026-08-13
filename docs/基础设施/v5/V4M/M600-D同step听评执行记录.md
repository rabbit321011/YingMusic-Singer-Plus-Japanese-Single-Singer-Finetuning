# M600-D 同 step 听评执行记录

## 裁决边界

容量裁决只允许比较相同训练步数：

```text
HighLR 6k  vs M600-D 6k
HighLR 12k vs M600-D 12k
```

`V4PH-30K-HIGHLR` final 只能作为成熟上限，不进入本轮容量胜负。四个条件共享 Phase-A P500、
65H manifest、HighLR schedule、seed、有效 batch、loss、H/PUL、GAME-P 和 official VAE；M600-D
相对 control 的主要变量是 DiT 从 338M 扩为 602.6M。训练 Loss 只证明健康，不替代听感。

## Checkpoint

| 条件 | checkpoint | SHA256: redacted
|---|---|---|
| HighLR 6k | `plus_ja_sft_v4ph_30k_highlr/step_006000.pt` | `95f6ea5bebf5e2f6f5a7eeaa4dfc02ab9aea59955f4208890e533fd5f4bd785c` |
| HighLR 12k | `plus_ja_sft_v4ph_30k_highlr/step_012000.pt` | `836017eea0763cfe1758076d050a1c9ea76b08c5b58c21da43575480b25f23ab` |
| M600-D 6k publish | `publish/V4M_M600D_step_006000.pt` | `b6dc69fb4222f94ed6874c645b4efb9ba80a423108c62a2e9238c68bd4c2700f` |
| M600-D 12k publish | `publish/V4M_M600D_step_012000.pt` | `824fd02cd6d00fbaa4c04f03da558f1415a3984269f2b2e117bb9c1d3e066816` |

两个 M600-D publish 各为 5,762,954,850 bytes，均由四卡离线合并；online/EMA strict-load、
1254 tensor 有限值和 `602,599,648` DiT state elements 审计通过。6k 合法状态为 `running`，
12k 因 `stop_after_step=12000` 合法状态为 `stopped`。

## 输入与生成合同

冻结 27 组输入与 H alignment 先打包为 `input.tar.gz`，大小 105,973,605 bytes，SHA256: redacted

```text
f9949ec300c86ee2763f0d75f36eb7ff0c9368ca16b572b0a95b52b062ad2aeb
```

本机与服务器直连受限，按部署运维文档经cloud storage中转；服务器 Go 版 `aliyunpan --sp 5` 在 9 秒
完成下载并通过 SHA256: redacted

```text
groups=27
steps=32
seed=42
CFG=3.0/1.0
A tail silence=0.5s
B tail silence=1.0s
VAE=official
placement=H phone/PUL
teacher=GAME medium K4, commit 4ad815c, base seed 20260730
MIDI=P class + checkpoint learned embedding; A region mask=zero
```

正式目录共 8 个：四个 checkpoint 各 CFG 3/1。HighLR 每目录约 53-54 秒，M600-D 每目录约
70-73 秒。

## 工程修复

首次 smoke 启动器引用服务器不存在的 `/usr/bin/time`，在加载模型前以退出码 127 停止，没有生成
音频；计时改为 Bash 秒表后重启。随后 M600-D 6k 被过严的 `run_state=stopped` 校验拦住；依据
checkpoint 合同改为 6k=`running`、12k=`stopped`。两项都只修正入口工程逻辑，没有改变 checkpoint、
输入、采样、CFG 或模型语义。

## 完整性结果

四条件单样本 smoke 通过：4 条均可解码、44.1kHz、单声道、有限且非静音，最大时长偏差
`7.642ms`，最低 RMS `0.16259`。

全量审计通过：

```text
directories=8
WAV=216
placement audits=216
max duration delta=21.406ms
minimum RMS=0.077957
input condition mismatches=0
CFG MIDI mismatches=0
CFG byte-identical WAV pairs=0
target PAD frames=0
prompt MIDI nonzero=0
GAME/VAE boundary delta<=1 frame
```

## 听评交付

盲包按步数分组：6k 使用 Model A/B，12k 使用 Model C/D；每个匿名模型均有 CFG 3/1。包内含
27 组 A/B references、216 条输出、README、机器 manifest 和 216 行评分表，不含解盲映射。

```text
cloud:
${CLOUD_ARTIFACT}

bytes:
451,361,953

SHA256: redacted
42da14638630cb304a85cab77686e0bc5381d7a17e5b8390fc8b030753c339a6
```

技术包单独保存全部 placement/GAME 审计与解盲密钥，评分前不要打开：

```text
${CLOUD_ARTIFACT}
SHA256: redacted
```

2026-08-04 用户完成首轮主观盲听后认为 12k 的 Model C 比另一 12k 模型好一点，并明确要求
解盲。密钥确认 C=`M600D_12K`、D=`HIGHLR_12K`；因此当前证据为 M600-D 在相同 12k step 下
小幅领先 HighLR，但不是压倒性改善。用户随后授权 M600-D 从已验证 12k final DCP 继续训练到
冻结 schedule 的 30k；M600-B 和 1.5B 仍未启动。

















