> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# YingMusic-Singer 日语微调 — 服务器训练任务

## 背景

在 Windows 笔记本 (RTX 5070 Ti, 12GB) 上已完成：
- 项目克隆、venv 环境搭建、PyTorch 2.9.1+cu128
- 模型权重下载（4.65GB）
- 源码适配（6 处兼容性修改，见 `DEPLOY.md`）
- 推理跑通（日语零样本合成成功）
- 训练脚本 `train.py` 编写 & 50步验证通过
- 数据集清洗（981→625段，~6.9h，60首歌）
- 歌词采集（B站API→歌名→UTA-NET→Whisper对齐）
- 训练数据 `lyrics.csv` 就绪

## 目标

用 8×4090 Linux 服务器对 YingMusic-Singer 做 DDP 多卡微调，使模型学会日语歌声合成。

## 服务器环境

| 项 | 值 |
|---|---|
| OS | Linux |
| GPU | 8× RTX 4090 (24GB) |
| NVLink | **无**（不影响，DDP 走 NCCL/PCIe） |
| 数据 | 需从 Windows 本地上传 |

## 需要从本地上传的文件

全部位于 `${LOCAL_PATH}` 目录下：

```
必须:
  YingMusic-Singer/src/              # 项目源码（含 6 处修改）
  YingMusic-Singer/train.py          # 训练脚本
  YingMusic-Singer/lyrics.csv        # 歌词映射（625行, stem|lyrics）
  YingMusic-Singer/.cache/huggingface/  # 模型权重（4个.pt + 1个.json）
  source_singers/                    # 625个训练wav（~2.2GB）

建议:
  YingMusic-Singer/DEPLOY.md         # 完整部署记录
  YingMusic-Singer/song_names.json   # BV号→歌名
  YingMusic-Singer/lyrics.json       # BV号→完整歌词
```

## 服务器端要做的事

### 1. 环境安装

```bash
# Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# PyTorch (CUDA 12.x → cu126 或 cu128)
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# flash_attn（Linux 上直接装）
pip install flash-attn --no-build-isolation

# 其他依赖
pip install -r requirements.txt
```

### 2. 目录结构

```
${PRIVATE_PATH}
├── train.py
├── lyrics.csv
├── src/                  # 从 YingMusic-Singer/src/ 上传
├── data/                 # source_singers/ 的 625 个 wav
├── checkpoints/          # 权重（.cache/huggingface/ 里的 .pt/.json 文件）
└── output/               # 训练输出目录
```

### 3. DDP 多卡改造

当前 `train.py` 是单卡版本。需要加 DDP 支持，核心改动约 20 行：

```python
# 头部加
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# init
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# model wrap
decoder = DDP(decoder, device_ids=[local_rank])

# DataLoader 加 DistributedSampler
# precompute cache 每卡共享（存磁盘或 broadcast）

# 保存时只在 rank 0 执行
if dist.get_rank() == 0:
    torch.save(...)
```

启动命令：
```bash
torchrun --nproc_per_node=8 train.py \
  --data_dir ./data \
  --lyrics_file ./lyrics.csv \
  --language ja \
  --max_steps 50000 \
  --grad_accum 1 \
  --batch_size 4 \
  --output_dir ./output
```

### 4. 训练参数建议 (8×4090)

| 参数 | 服务器值 | 说明 |
|------|----------|------|
| `--grad_accum` | 1 | DDP 8卡等效 batch=8 |
| `--max_steps` | 50000 | 同上 |
| `--lr` | 1e-4 | 可适当调到 2e-4 |
| `--warmup_steps` | 3000 | |
| `--clip_grad` | 10.0 | |
| `--cfg_drop_prob` | 0.15 | |
| `--use_amp` | True | bf16 |

### 5. 训练原理

Flow Matching，全自监督：
- 每段 seg → 前半段做音色参考 (cond)，整段做 GT (x₁)
- `xₜ = (1-t)x₀ + t x₁` 线性路径
- `loss = MSE(v_pred, x₁ - x₀)`
- 15% 概率 drop_text 训练 CFG
- 冻结: VAE, RMVPE, SOME, Tokenizer
- 可训练: 仅 DiT (~329M 参数)

## 注意事项

1. **flash_attn 依赖**: Linux 上 `pip install flash-attn --no-build-isolation` 即可，不需要改源码
2. **音频 I/O**: `src/singer/model.py` 中 `load_audio` 已改为 `soundfile`（绕过 torchcodec），DDP 环境下无需再改
3. **Tokenizer 日语路径**: `src/singer/tokenizer/g2p/cleaners.py` 中 `language="ja"` 已改为不触发 espeak
4. **LangSegment 兼容**: `.venv/Lib/site-packages/LangSegment/__init__.py` 补了 `setLangfilters` 别名（全局生效）
5. **训练数据路径**: 所有数据文件使用绝对路径或相对路径指向 `./data/`
6. **缺少 espeak**: `phonemizer` 依赖 espeak，日语路径已绕过，不影响训练。若其他语言报错，`apt install espeak-ng` 即可。

## 完整源码修改清单

共修改 4 个文件，详见 `DEPLOY.md`：

| 文件 | 改动 |
|------|------|
| `src/singer/model.py` | `_load_audio_sf` 替代 torchcodec；`--language` CLI 参数；soundfile 保存 |
| `src/singer/decoder/modules.py` | `flash_attention`→`attention`；SDPA dtype 修复 |
| `src/singer/tokenizer/g2p/cleaners.py` | 日语路径传 `None` 避免 espeak |
| LangSegment `__init__.py` | `setLangfilters`/`getLangfilters` 别名 |
