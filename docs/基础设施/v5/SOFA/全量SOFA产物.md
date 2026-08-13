# P0 全量 SOFA 对齐产物

## 全量对齐结果

本机 RTX 5070 Ti Laptop 12GB 完成全量运行：

| 指标 | 值 |
|---|---|
| 输入样本 | 15429 |
| 对齐成功 | 15414 |
| 对齐失败 | 15 |
| 成功音频总时长 | 约 92.46h（332856s） |
| 墙钟时间 | 837.35s（约 13.96min） |
| Gold 时间戳参与生成 | 否 |
| 边界完整性检查 | 0 错误 |

15 条失败均为 PyOpenJTalk 无法处理的标点、乱码或非日语短语，非 SOFA 推理失败。

产物位置：
```
${LOCAL_PROJECT_ROOT}\dataset\final_sum_large\pretreatment_text\timeset_SOFA\
├── timeset_SOFA.json
├── timeset_SOFA_failures.json
├── timeset_SOFA_summary.json
├── train_text.json
├── train_kana.json
├── train_tokens.json
├── test_text.json
├── test_kana.json
└── test_tokens.json
```

## train/test × text/kana/tokens 六文件

| 类型 | train | test |
|---|---|---|
| text | 15107 | 307 |
| kana | 15107 | 307 |
| tokens | 14661 | 296 |

Kana 原样保留 `train/test_singnet_kana.json` 中现有上下文处理结果，未重新硬转。
Tokens 沿用既有魔改日语链路：`text → japanese_to_ipa → PhonemeBpeTokenizer → token id + 1`。

与旧 token 文件重叠 65934 短语逐句复算，token 不一致数为 0。SEP token id=365。空音素沿用 CNENTokenizer 兜底。

## token/frame 溢出过滤

标准：任一短语 `token_count > phrase_frames + 5` 则整条音频不进入 tokens 文件。

| Split | text/kana 音频 | tokens 保留 | 溢出过滤 | 触发短语数 |
|---|---:|---:|---:|---:|
| train | 15107 | 14661 | 446 | 605 |
| test | 307 | 296 | 11 | 21 |
| 合计 | 15414 | 14957 | 457 | 626 |

## VOCAB_TOKEN_MAP 审计

- vocab 原始 ID：0~362
- 句内普通 token（+1后）：1~363
- `<PAD>`=0，`<PUNCT>`=364（本产物未使用），`<SEP>`=365（仅插在相邻短语间）
- 模型 embedding 合法范围：0~373
- 全量检查：句内无 0/364/365，`<SEP>=365` 共 73182 次，365 以上 0 次
- 日语专属音素占用 319~346，同时复用通用辅音元音

















