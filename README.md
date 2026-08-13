# 歌声合成研究代码库

这是一个实验性歌声合成项目的公开研究档案，包含版本化的训练代码、推理代码、数据处理逻辑、评估工具、工程记录和实验结果。

本仓库公开的是**算法和工程逻辑**。训练数据、音频、模型权重和运行缓存不随仓库分发；使用者需要自行准备具有合法使用权的数据和模型文件。

## 公开内容

- `docs/`：架构、数据流程、训练计划、实验记录和评估协议
- `src/YingMusic-Singer-Plus/`：基础 YingMusic-Singer-Plus 源码和早期 V3/V4/V5a1 训练入口
- `src/package_v4c_finetune/`：完整的版本化训练包，包括 V4c、V4H/V4Hg、V4PH、V4IPH、V4IjPH、V4M、V4Pvf、V4SF、V4VF/V4VFG、V5P/V5Pg 和 V5Sg
- `src/archive_training/`：早期 SFT、GRPO、YYSinger 和 Vevo2 训练脚本归档
- `third_party/`：允许再分发的第三方源码及其许可证文件
- `config/`：不含私密路径的配置示例
- `results/`：经过筛选和脱敏的少量结果摘要

## 重要边界

仓库不包含：训练音频、生成音频、原始 manifest/JSONL、latent/cache、checkpoint、VAE/MIDI/分离模型权重、私有服务器配置和云存储凭据。

训练语料不授予录音、歌曲、歌词、表演或身份相关权利。请只使用自己有权使用的数据，不要将生成模型用于冒充他人或绕过相关权利人的许可要求。

## 环境准备

以 Linux + NVIDIA CUDA + Python 3.10 为主要运行环境。基础依赖清单位于：

```text
src/YingMusic-Singer-Plus/requirements.txt
```

示例安装方式：

```bash
cd src/YingMusic-Singer-Plus
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

不同 CUDA、PyTorch 和显卡环境可能需要按照官方说明选择对应的 PyTorch wheel。训练前请确认 `torch.cuda.is_available()` 正常，并安装 `ffmpeg`、音频编解码器及项目所需的系统依赖。

## 运行前需要准备的文件

请在本机准备以下内容，并通过命令行参数或环境变量传入：

1. 有授权的数据音频及对应 manifest/token 文件；
2. 基础模型、训练起点 checkpoint、VAE checkpoint、MIDI/SOME teacher checkpoint；
3. 如使用 V4Hg/V5Pg 等分支，还需要该分支要求的 warm-start checkpoint、对齐 manifest、GAME/MIDI/风格缓存或审计文件；
4. 独立的输出目录，避免覆盖输入 checkpoint。

可以从 `config/public.example.yaml` 开始填写本地配置。示例中的 `/path/to/...` 都是占位符，不是仓库运行所需的固定路径。

## 训练复现

### V4c

进入主源码目录，准备 token 数据目录、基础 checkpoint、VAE 和 MIDI teacher：

```bash
cd src/YingMusic-Singer-Plus

CUDA_VISIBLE_DEVICES=0 bash run_sft_v4c.sh \
  --token_dir /path/to/authorized/tokens \
  --output_dir /path/to/output/v4c \
  --ckpt_path /path/to/base/YingMusicSinger_model.pt \
  --vae_ckpt /path/to/weights/vae.ckpt \
  --midi_ckpt /path/to/weights/midi_teacher.ckpt
```

多卡版本使用 `run_sft_v4c_ddp.sh`。更底层、参数更完整的入口是 `train_plus_v4c.py`。

### V4H / V4Hg

对应实现和启动脚本位于：

```text
src/package_v4c_finetune/train/train_plus_h.py
src/package_v4c_finetune/train/run_sft_h_ddp.sh
src/package_v4c_finetune/train/run_sft_v4hg_10k.sh
```

V4Hg 是从 V4H warm-start 并切换到指定 VAE 的适配训练。使用前需要准备授权的训练/评估 manifest、warm-start checkpoint、VAE checkpoint，并设置脚本要求的环境变量，例如：

```bash
cd src/package_v4c_finetune
export H_PROJECT_DIR=/path/to/project
export H_DATA_DIR=/path/to/authorized/h_training
export H_WARMSTART_CHECKPOINT=/path/to/v4h/checkpoint.pt
export H_VAE_CKPT=/path/to/vae_285k.ckpt
export H_OUTPUT_DIR=/path/to/output/v4hg
bash train/run_sft_v4hg_10k.sh
```

### V5P / V5Pg

V5P/V5Pg 训练实现位于：

```text
src/package_v4c_finetune/train/train_v5p.py
```

该入口要求训练和评估 manifest、GAME cache manifest、pool audit、H 配置指纹，以及相应的基础或 warm-start checkpoint。示例：

```bash
cd src/package_v4c_finetune
torchrun --nproc_per_node=4 train/train_v5p.py \
  --train_manifest /path/to/authorized/train_manifest.json \
  --eval_manifest /path/to/authorized/eval_manifest.json \
  --game_cache_manifest /path/to/authorized/game_cache/manifest.json \
  --pool_audit /path/to/authorized/pool_audit.json \
  --h_config_fingerprint YOUR_CONFIG_FINGERPRINT \
  --ckpt_path /path/to/base/YingMusicSinger_model.pt \
  --vae_ckpt /path/to/vae.ckpt \
  --midi_ckpt /path/to/midi_teacher.ckpt \
  --output_dir /path/to/output/v5p
```

V5Pg 还需要通过 `--warmstart_checkpoint` 提供 V5P checkpoint，并按照脚本参数指定对应 phase。完整参数可运行：

```bash
python train/train_v5p.py --help
```

### 其他版本和历史训练线

`src/package_v4c_finetune/train/` 中的 `run_*.sh`、`train_*.py` 和 `train_plus_*.py` 对应其他 V4 分支及其 smoke/resume/FSDP/DDP 入口。早期实验位于 `src/archive_training/`，每个目录保留原训练脚本和启动脚本；运行前请检查脚本参数并替换其中的占位路径。

## 推理和评估

V4 推理入口：

```bash
cd src/YingMusic-Singer-Plus
python infer_v4.py \
  --checkpoint /path/to/checkpoint.pt \
  --ref_audio /path/to/reference.wav \
  --melody_audio /path/to/melody.wav \
  --vae_ckpt /path/to/vae.ckpt \
  --midi_ckpt /path/to/midi_teacher.ckpt \
  --output /path/to/output.wav
```

V5P 轨迹评估、checkpoint 审计、对齐和结果整理脚本位于 `src/package_v4c_finetune/infer/` 与 `src/package_v4c_finetune/train/`。运行这些工具时，同样只使用自己有权处理的输入音频和 manifest。

## 审计

发布前可运行：

```powershell
./scripts/public_release_audit.ps1
```

该审计会检查禁止的二进制/媒体资产、私有路径、服务器标识和凭据模式。

## 许可证

版权归 `rabbit321011` 所有的原创代码、文档和研究材料统一遵循 [PolyForm Noncommercial License 1.0.0](LICENSE)：允许非商业研究、使用、修改和分发，但不授权商业用途。该许可证属于公开源码的非商业许可证，不是 OSI 认定的开源许可证。第三方材料不受根目录许可证覆盖，仍遵循各自的上游许可证和署名要求，详见 `THIRD_PARTY_NOTICES.md` 及各第三方目录中的许可证文件。
