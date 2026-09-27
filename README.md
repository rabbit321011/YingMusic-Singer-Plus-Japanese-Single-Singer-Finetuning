# YingMusic-Singer-Plus 日语单歌手微调

本项目 Fork 自 [ASLP-lab/YingMusic-Singer-Plus](https://github.com/ASLP-lab/YingMusic-Singer-Plus)，用于研究和复现**日语单歌手歌声模型的微调、声音克隆与可控歌声合成**。

## 项目用途

使用者可以用自己获得合法授权的单歌手干声音频，对 YingMusic-Singer-Plus 进行持续训练或微调，得到面向该歌手音色和日语演唱的专属模型。推理时，模型接收参考音色、提供旋律的演唱音频和目标歌词，生成保持目标音色、遵循输入旋律并演唱指定歌词的新歌声音频。

仓库覆盖从数据准备到模型评估的完整工程逻辑，主要用于：

- 构建和清洗单歌手训练数据，生成 token、对齐、MIDI/GAME 和其他训练条件；
- 从官方基础模型或已有 checkpoint 启动日语单歌手 SFT；
- 研究歌词咬字时序、摩拉/音素对齐、旋律保持、音色保持和长音频生成；
- 适配不同 VAE，并比较 V4c、V4H/V4Hg、V4PH、V4Pvf、V5P/V5Pg、V5PgO/V5PgOV 等实验路线；
- 对 checkpoint 执行推理、轨迹生成、客观指标评估和人工听评准备；
- 在使用者自己的授权数据和权重上复现实验，不需要重新编写各版本的训练控制逻辑。

这不是一个附带现成歌手模型的一键演唱程序，而是一套可替换数据、权重和路径后运行的训练、推理与研究工程。仓库包含版本化训练代码、数据处理逻辑、评估工具、工程记录和实验结果。

本仓库公开的是**算法和工程逻辑**。训练数据、音频、模型权重和运行缓存不随仓库分发；使用者需要自行准备具有合法使用权的数据和模型文件。

## 公开内容

- `docs/`：架构、数据流程、训练计划、实验记录和评估协议
- `src/YingMusic-Singer-Plus/`：基础 YingMusic-Singer-Plus 源码和早期 V3/V4/V5a1 训练入口
- `src/package_v4c_finetune/`：完整的版本化训练包，包括 V4c、V4H/V4Hg、V4PH、V4IPH、V4IjPH、V4M、V4Pvf、V4SF、V4VF/V4VFG、V5P/V5Pg 和 V5Sg
- `src/archive_training/`：早期 SFT、GRPO、YYSinger 和 Vevo2 训练脚本归档
- `third_party/`：允许再分发的第三方源码及其许可证文件
- `config/`：不含私密路径的配置示例
- `results/`：经过筛选和脱敏的少量结果摘要
- `config/model_v_decoder_adapt.json`、`src/package_v4c_finetune/train/train_v_decoder_adapt.py`：V 分支 decoder-only 训练配方与入口
- `docs/基础设施/`：脱敏后的实验设计、诊断、训练记录与阶段结论；[公开范围](docs/PUBLICATION_SCOPE.md)说明保留在私有工作区的材料

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

### V5PgOV decoder 适配

V5PgOV 保持 V5PgO 8K DiT 和 285K online VAE encoder 冻结，只用实际生成的 B 区 latent 与对应的授权原始波形训练 VAE decoder。公开了脱敏的 [训练入口](src/package_v4c_finetune/train/train_v_decoder_adapt.py) 与 [配置](config/model_v_decoder_adapt.json)，需要自行提供四轮兼容缓存、基础 VAE 和 SHA256 校验值；不包含数据、缓存或权重。

实验配方为 32 latent 帧窗口、300K step、EMA、关闭 KL。生成训练缓存时使用逐条随机的三路 CFG（audio 0.4–0.7、text 0.4–0.6、MIDI 0.4–0.5）；**decoder 训练本身没有固定 CFG**。小样本听评中 V 相对旧 decoder 有改善，但柔声可能产生清晰度与空间感副作用，不能据此声称对未见曲目普遍更好。缓存格式、依赖、运行命令与边界见 [V5PgOV decoder 适配说明](docs/v5pgov-decoder-adaptation.md)；路线及听评结论见 [V 分支](docs/基础设施/v5/V/main_V.md)，基座见 [V5PgO](docs/基础设施/v5/V5P/V5PgO/main_V5PgO.md)。

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

## 联系方式

项目维护者：[`rabbit321011`](https://github.com/rabbit321011)<br>
邮箱：[`s321011s@foxmail.com`](mailto:s321011s@foxmail.com)

## 许可证

本仓库采用**分层许可**，不是所有文件统一使用同一份许可证：

- `rabbit321011` 原创的新增代码、文档、研究记录、配置及对上游代码具有独创性的修改，遵循 [PolyForm Noncommercial License 1.0.0](LICENSE)，未经另行授权不得用于商业目的；
- 源自 [ASLP-lab/YingMusic-Singer-Plus](https://github.com/ASLP-lab/YingMusic-Singer-Plus) 的上游内容仍遵循 [CC BY 4.0](LICENSE-CC-BY-4.0)，该许可证原有的商业使用权不受 PolyForm 限制；
- VAE 模型权重及 `src/YingMusic-Singer-Plus/src/YingMusicSinger/utils/stable_audio_tools/` 中源自 Stable Audio Open 的推理代码遵循 [Stability AI Community License](LICENSE-STABILITY)；
- Amphion G2P、MusicSourceSeparationTraining、py3langid、PaddlePaddle 以及其他第三方内容保留其各自许可证。

**Powered by Stability AI**

GAME、SOFA、Whisper 等运行时依赖及模型权重不随本仓库分发。特别是本项目使用的 GAME 1.0 权重为 `CC BY-NC-SA 4.0`，SOFA `JPN_Test2_Plus` 发布页明确标注 `Commercial Use: Not Approved`。使用者必须自行取得这些组件并遵守其发布页或模型卡条款。

完整的权属边界和逐组件核验结果见 [LICENSING.md](LICENSING.md)、[NOTICE](NOTICE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
