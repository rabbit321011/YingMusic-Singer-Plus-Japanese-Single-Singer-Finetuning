# JP 场景对照与 DiT 分层消融计划

V4f 训练完成后规划的 DiT 分析实验。当前不是执行主线（主线是 VAE 音质）。

## 关键修正

- Official 主要是 CN/EN，不能用"Official 听感好、V4f 差"直接证明 JP 微调破坏美学
- 最干净的第一对照是日语原音频 VAE encode/decode
- 主要 DiT 对照应使用 V4f 早期/中期/后期 JP checkpoint 轨迹
- 若未来完成 P1，JP-grounded base 才是更公平的共同起点

## 消融方法论

- "不加载某层"定义为 residual bypass：`x_out=x_in`（不是全 0 或全 1）
- 分层干预包括：bypass、alpha 缩放、Official/V4c/早期 JP 层替换、后置同尺度噪声
- 第一轮按 Early/Middle/Late 层段做，找到敏感区后再做 2-3 层窗口或逐层扫描
- Official replacement 只作语言域混淆下的探针，关键结论优先由早期 JP checkpoint replacement 复核

## 后续猜想路线

- B 区使用目标音频经 YM-SVC 后作为 condition
- B 区随机暴露变声后的 condition 增强鲁棒性
- A/B 区加入字级时间戳
- A/B 同源与异源的消融

这些 Singer 路线当前暂不执行，待 VAE 基线确认后再定优先级。

















