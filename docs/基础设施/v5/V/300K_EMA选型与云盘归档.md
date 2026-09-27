> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# V5PgOV 300K EMA 选型与归档

用户听完 180K、225K、255K、285K、300K 的长片段 EMA 对照后，
裁决 **最终 300K EMA 的稳定性最好**，选作本轮 V5PgOV 的 EMA 权重。
对照包见 `${LOCAL_PATH}`，
包含夜明け、月兔、团子三组，同组固定条件、seed 137 和同一次 DiT 采样 latent。
这属于上述样本上的人工听评，不代表所有未见曲目的普遍最优结论。

## 权重与校验

| 项 | 值 |
|---|---|
| 云盘目录 | `${CLOUD_ARTIFACT}` |
| EMA 导出 | `V5PgOV_300K_EMA_decoder.pt` |
| 清单 | `V5PgOV_300K_EMA_decoder.json` |
| 导出 SHA256 | `SHA256_REDACTED` |
| 训练 checkpoint SHA256 | `SHA256_REDACTED` |
| 285K online VAE SHA256 | `SHA256_REDACTED` |
| 本机副本 | `${LOCAL_PATH}` |

导出从 final 300K checkpoint 的
`autoencoder_ema.ema_model.decoder.*` 提取 182 个权重张量，
保存为 `decoder.*` 键。导出后重新加载并逐张量与原 EMA 核对；
从阿里云盘 P 盘回读到本机后，SHA256 与清单一致。
原完整训练 checkpoint 仍保留在服务器。另已启动后台归档：
tmux `v5pgov_archive_upload` 顺序上传该完整 checkpoint 和 285K online VAE
到 `${CLOUD_ARTIFACT}`，日志为
`${PRIVATE_PATH}`。
启动时源文件 SHA 门禁通过；后台归档是否完成以日志中的
`upload_complete` 和 `upload_exit=0` 为准，不把启动等同于上传验收。

**这是 decoder-only 权重，不是独立可推理的完整 VAE。**
使用时必须先严格加载上述 285K online VAE，再只替换其 decoder；
DiT 仍使用 PgO 8K。上传和选型不等于切换现有 V5PgO 默认预设，
也不消除 V 分支结论文档中记录的柔声副作用。
