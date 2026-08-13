# Low-Noise Pathology 与 LCF

本分支负责判断 YMSP 的低噪声端是否形成实际优化病理，以及 Local Contrastive Flow 是否值得进入受控训练实验。

## 当前状态

- PH 与 fg 的六 checkpoint 只读轨迹诊断已完成并通过双端 SHA256: redacted
- 两家族共享真实的端点欠响应与 hidden 梯度转向，但现有证据不支持它是 PH 相对 fg 音色退化的主因；
- 以 `V4PH-30K-HIGHLR` 为冻结 control 的首轮 LCF 已完成 30k 训练、final 审计和固定 27 组实名听评；
- 用户裁决为“无明显改善”，本轮 LCF 未通过预注册音频门禁，不进入 V5 模型配方；
- 低噪声端点欠响应仍是已确认的共享局部优化现象，但当前没有证据表明修正它能带来可辨认音频收益。

## 工人

| 文件 | 内容 |
|---|---|
| `PH与fg低噪声轨迹诊断.md` | 时间方向映射、六 checkpoint 契约、配对统计、因果裁决与 LCF 门禁 |
| `PH-HighLR-LCF严格对照实验.md` | 单一 LCF 配方、HighLR control、RNG/负样本/归一化契约、执行门禁与评价 |

















