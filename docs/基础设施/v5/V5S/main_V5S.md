# V5-Sg 分支

V5-Sg 是 V5 的 continuous SOME 路线：从官方 checkpoint fresh 开始，285k online VAE 与
continuous SOME 从 step 0 同时加入，使用 V5-P 的 H/PUL、L-FULL-LENGTH-TIME 和 40k 训练合同。
这里的 `S` 固定表示 SOME，不表示历史 `S/` SVC 自克隆 waveform 路线。

## 当前状态

- 仅完成路线合同和脚本改造设计，尚未实现、未训练、未占 GPU；
- g 从 step 0 加入，不存在 V5-S non-g -> Sg 的两阶段切换；
- LR 为 `0--4k warmup -> 4k--28k cosine 1.4e-5 到 1e-5 -> 28k--40k cosine 到 0`；
- 训练输入、H、L、loss、dropout、seed、effective batch 与 V5-P Phase B 对齐；
- SOME 必须离线缓存并绑定 waveform SHA、SOME checkpoint SHA 与前端 schema；
- V5-Sg 与 V5-Pg 是完整路线比较，不是纯 MIDI 表示单变量；
- 完成后只做轨迹听评，不自动启动额外 Sg 加训。

## 实现母本

复制 V4Hg 的 `package_v4c_finetune/train/train_plus_h.py` 为独立 `train_v5sg.py`，并以
`run_sft_v4hg_10k.sh` 为 launcher 母本。保留 V4Hg 已验证的 285k VAE、continuous SOME、g
transition、CPU EMA 和 H/PUL 语义；吸收 V5-P 的 L-FULL-LENGTH-TIME、60 秒 Long、显存释放、
40k scheduler、metadata 和 exact-resume 门禁。不得直接复用 V4Hg 的 30 秒 manifest、旧 schema
或 10k LR 合同。

## 阅读顺序

1. `../V5P/V5-P正式训练计划.md` 第 12 节：完整合同和入口改造边界；
2. `../H/摩拉音素级对齐实验.md`：H/PUL renderer 和 continuous SOME 历史语义；
3. `../S/main_S.md`：历史 SVC 自克隆路线，注意它不是 V5-Sg；
4. `../V5P/main_V5P.md`：V5-Pg 对照路线。

















