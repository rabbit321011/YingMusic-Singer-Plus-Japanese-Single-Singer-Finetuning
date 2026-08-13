# YingMusic-Singer-Plus V3 日语 SFT — 训练记录

> 启动时间：2026-05-26 15:05 (北京时间)
> 最终版本：第5次启动（token 泄漏修复后）
> tmux session`

---

## 一、启动配置

| 参数 | 值 |
|------|:--:|
| 训练脚本 | `train_plus_v3.py` |
| 启动命令 | `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4` |
| PYTORCH_CUDA_ALLOC_CONF | `expandable_segments:True` |
| GPU | 0,1,2,3 (4× RTX 4090 24GB) |
| Learning rate | 7e-6 |
| Warmup | 500 steps |
| Total steps | 30,000 |
| Batch size | 4 / GPU (effective 16) |
| Max duration | 30 秒 |
| Precision | fp32 |
| CKA weight | 1.0 |
| Drop audio | 30% (对齐官方) |
| Drop text | 30% |
| Drop midi | 30% |
| Seed | 42 |

---

## 二、数据集

| 项目 | 值 |
|------|:--:|
| 训练集 | private corpus statistics |
| 验证集 | 307 条 / ~2h |
| 语言 | 100% 日语 |
| Train/Test 隔离 | 按 BV 号 |
| Steps/epoch | 945 (~24 epochs total) |

---

## 三、V3 相比 V1 的核心修复

| # | 模块 | V1 问题 | V3 修复 |
|---|------|------|------|
| 1 | cond | 完整 latent 同时当 cond 和 GT → DiT 抄答案 | 前 50% 帧→cond（尾部填零），整段→x₁ |
| 2 | midi | cond 区域 midi 未清零 | `midi[:, :ref_len, :] = 0` |
| 3 | CNENTokenizer | 日文 G2P 路由错误（get_segment 判 "other"） | `chn_eng_g2p` 强制日语路径，pyopenjtalk + japanese.py |
| 4 | Token 路径 | V1 用 0-based encode_ja，推理用 1-based CNENTokenizer | V3 统一用 `CNENTokenizer.encode()` (1-based) |

---

## 四、性能退化诊断与修复

### 4.1 诊断过程

| 步骤 | 实验 | 结论 |
|:---:|------|------|
| 1 | 单卡 vs 四卡退化对比 | 单卡也退化（+25%），不是 DDP 专属 |
| 2 | `find_unused_parameters=True/False` | 退化率相同，无关 |
| 3 | bf16 混合精度 | 退化率相同，无关 |
| 4 | `grad_accum=1/2` 对照 | V1(grad_accum=2)=0%退化, V3(grad_accum=2)=+48%退化 |
| 5 | 预分配 buffer（reuse_buffers） | 退化依旧 |
| 6 | cond=GT (V1 式 cond 路径） | 退化依旧 |
| 7 | **细粒度 profile_timing** | ✅ token phase: 0.59→1.02s（+72%），其他全部稳定 |

### 4.2 根因

`chn_eng_g2p()` 每次调用 `PhonemeBpeTokenizer()` 创建新实例，构造 6 个 `EspeakBackend` → 每步 4 条音频 × 1 次 tokenize = 每步 24 个 espeak/C++ 实例从不复用 → 堆内存碎片化 → token 时间从 0.59s 退化为 1.02s。

### 4.3 修复

```python
# g2p_generation.py chn_eng_g2p()
# 修复前:
pbt = PhonemeBpeTokenizer()                     # 每调用一次建一个
tokens = pbt.phoneme2token(phoneme)

# 修复后:
tokens = text_tokenizer.phoneme2token(phoneme)  # 复用模块级全局实例（第118行已存在）
```

| 指标 | 修复前 | 修复后 |
|------|:---:|:---:|
| token phase | 0.59→1.02s (+72%) | **0.047s，完全稳定** |
| 全步速度 (单卡) | 1.01→1.43s | **0.51s，零退化** |
| G2P 等价 | — | ✅ 5/5 通过 |

### 4.4 预计训练时间

单卡 0.51s/step, 四卡 effective batch=16 → 预计 **~1.0-1.5s/step, ETA ~8-12h**。

---

## 五、训练日志

### 5.1 Loss 曲线

```
Step    Flow    CKA    LR       Speed
    待首次 log_every=50 后填入
```

### 5.2 时间线

| 次数 | 问题 | 结果 |
|:---:|------|------|
| 1 | 日文 token 路径 `Unknown language: other` | 修复 `chn_eng_g2p`→强制日语 |
| 2 | Whisper 加载破坏 DDP 同步 | 移除 Whisper，独立 `eval_wer.py` |
| 3 | CUDA 碎片化→速度退化 | 经 7 步诊断，定位 PhonemeBpeTokenizer 泄漏 |
| 4 | bf16/find_unused/reuse_buffer 等无效尝试 | 均排除 |
| 5 | ✅ 稳定运行中 | 修复 token 泄漏，0.51s/步 零退化 |

---

## 六、关键里程碑

| Step | 预计北京时间 | 检查项 | 通过标准 |
|:---:|------|------|------|
| 50 | 15:06 | warmup 中，loss 已开始下降 | Flow < 25 |
| 250 | 15:08 | loss 暴跌 | Flow < 3 |
| 500 | 15:12 | warmup 结束 | Flow < 2.0 |
| 1000 | 15:25 | 首次 eval | Eval loss 正常下降 |
| 2000 | 16:00-17:00 | 第一阶段自重建 | 可辨识日语 |
| **5000** | 当天傍晚 | 🔴 **门禁改词测试** | Whisper WER < 0.5 |
| 10000 | 次日上午 | 改词 WER + 听感 | |
| 30000 | 次日 | 完整评估 | 7 首 JP benchmark |

### 6.1 WER 评估管线

```bash
# baseline: 已跑（官方权重 WER=1.000）
python eval_wer.py --n_samples 10

# checkpoint 评估
python eval_wer.py --ckpt_path ckpts/plus_ja_sft_v3/step_002000.pt --n_samples 10
```

---

## 七、G2P 验证结果

全部 5 个测试样本通过（修复前后均验证）：
```
✅ たぶん私じゃなくていいね
✅ 君を見るたびに思い出す
✅ カッカッカッカッカッ
✅ 世界で一番お姫様
✅ 私の声を聞いてください
```

---

## 八、已知问题与修复记录

| # | 问题 | 现象 | 根因 | 修复 |
|---|------|------|------|------|
| 1 | cnen_tokenizer.py 损坏 | 导入 `chn_eng_jpn_g2p` 失败 | V2 修改残留 | 上传原始 CNENTokenizer |
| 2 | `Unknown language: other` | 训练崩溃 | `has_japanese` 漏检纯汉字 | `chn_eng_g2p` 强制日语 |
| 3 | `Input must be katakana only` | pyopenjtalk 崩溃 | japanese.py 引号编码 | 使用 Amphion 原始文件 |
| 4 | `Cannot read` 生僻字 | 标注噪声 | Whisper 幻觉 | pykakasi fallback |
| 5 | DDP 死锁 | Whisper rank0 阻塞 | 时序不同步 | 独立 eval_wer.py |
| 6 | **速度线性退化** | 1.56→3.80s/step | `chn_eng_g2p` 每步建 PhonemeBpeTokenizer | 复用全局 `text_tokenizer` |
| 7 | **SEP/PUNCT 帧未训练** | 阻塞 | 待 Step 5000 验证 | 方案 A: 训练加 SEP; 方案 B: 推理去 SEP |

---

## 九、checkpoint 记录

| Step | 路径 | WER | 备注 |
|:---:|------|:--:|------|
| 0 | (官方权重) | 1.000 | baseline |
| 2000 | `step_002000.pt` | 待评估 | |
| 4000 | `step_004000.pt` | 待评估 | |
| 5000 | `step_005000.pt` | 待评估 | 🔴 门禁 |

---

## 十、服务器信息

- 服务器: `${SERVER_USER}@${SERVER_HOST}`
- 磁盘: 7.0TB
- tmux session` (训练), `tensorboard`
- 已清理: `plus_ja_sft/` (93GB), `plus_ja_ft/` (9.3GB), `plus_grpo/` (1.7GB)

---

> 最后更新：2026-05-26 15:05 (正式训练启动)

















