> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# MSST 服务器部署日志

> 迁移日期：2026-05-20
> 服务器：`USER@IP_REDACTED` (8×RTX 4090, CUDA 13.0)

---

## 部署内容

### 1. Conda 环境

| 项目 | 值 |
|------|-----|
| 环境名 | `MSST_scene` |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| 路径 | `${PRIVATE_PATH}` |

激活方式：
```bash
source ${PRIVATE_PATH}
conda activate MSST_scene
```

### 2. MSST-WebUI 代码

| 项目 | 值 |
|------|-----|
| 上游仓库 | `github.com/SUC-DriverOld/MSST-WebUI` |
| 国内镜像 | `kkgithub.com/SUC-DriverOld/MSST-WebUI` |
| 服务器路径 | `${PRIVATE_PATH}` |

### 3. 已安装模型权重（三模型管道）

| 模型 | 类别 | 大小 | 用途 |
|------|------|:---:|------|
| `melband_roformer_instvox_duality_v2.ckpt` | vocal_models | 1.7GB | 人声/伴奏分离 |
| `dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt` | single_stem_models | 435MB | 去混响+去回声 |
| `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` | single_stem_models | 871MB | 降噪 |

存放结构：
```
${PRIVATE_PATH}
├── vocal_models/
│   └── melband_roformer_instvox_duality_v2.ckpt
└── single_stem_models/
    ├── dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt
    └── denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt
```

配置文件 `configs/*.yaml` 已随代码 clone 完整包含。

### 4. CLI 工具

| 文件 | 路径 |
|------|------|
| 服务器 CLI | `${PRIVATE_PATH}` |
| 本地 CLI | `${LOCAL_PATH}` |

---

## 使用方法

### 单步推理

```bash
# 激活环境
source ${PRIVATE_PATH}
conda activate MSST_scene
cd ${PRIVATE_PATH}

# 人声分离
python msst_cli_linux.py \
  --model melband_roformer_instvox_duality_v2.ckpt \
  --input /path/to/audio.mp3 \
  -o /path/to/output \
  -d cuda

# 去混响
python msst_cli_linux.py \
  --model dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt \
  --input /path/to/Vocals.wav \
  -o /path/to/output \
  -d cuda

# 降噪
python msst_cli_linux.py \
  --model denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt \
  --input /path/to/dry.wav \
  -o /path/to/output \
  -d cuda
```

### 批量处理（经典三模型管道）

用 Python 脚本串联三步：

```python
# batch_pipeline.py —— 放到 ${PRIVATE_PATH} 下运行
import os, sys, shutil, tempfile, glob
from inference.msst_infer import MSSeparator

INPUT_DIR = "/path/to/raw_audio/"
OUTPUT_DIR = "/path/to/clean_vocals/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型配置
MODELS = [
    ("melband_roformer_instvox_duality_v2.ckpt", "vocal_models", "Vocals", "configs/vocal_models/melband_roformer_instvox_duality_v2.ckpt.yaml"),
    ("dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt", "single_stem_models", "dry", "configs/single_stem_models/dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt.yaml"),
    ("denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt", "single_stem_models", "dry", "configs/single_stem_models/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt.yaml"),
]

audio_files = glob.glob(os.path.join(INPUT_DIR, "*"))
for af in sorted(audio_files):
    current_input = af
    basename = os.path.splitext(os.path.basename(af))[0]
    for idx, (model_name, model_class, target_stem, config_path) in enumerate(MODELS):
        sep = MSSeparator(
            model_type="mel_band_roformer",
            config_path=config_path,
            model_path=f"pretrain/{model_class}/{model_name}",
            device="cuda",
            device_ids=[0],
            output_format="wav",
            use_tta=False,
            store_dirs=OUTPUT_DIR,
        )
        tmpdir = tempfile.mkdtemp()
        tmp_input = os.path.join(tmpdir, os.path.basename(current_input))
        shutil.copy2(current_input, tmp_input)
        sep.process_folder(tmpdir)
        sep.del_cache()
        shutil.rmtree(tmpdir, ignore_errors=True)
        # 找这一步输出的目标 stem
        if idx == 0:
            candidates = glob.glob(os.path.join(OUTPUT_DIR, f"*_{target_stem}.wav"))
        else:
            candidates = glob.glob(os.path.join(OUTPUT_DIR, f"*_{target_stem}.wav"))
        if candidates:
            current_input = sorted(candidates, key=os.path.getmtime)[-1]
            print(f"Step {idx+1} complete: {current_input}")
    print(f"Done: {basename}")
```

### CLI 查询命令

```bash
cd ${PRIVATE_PATH}

# 列出所有模型分类
python msst_cli_linux.py --list-categories

# 列出某分类下模型及安装状态
python msst_cli_linux.py --list-models vocal_models

# 搜索模型
python msst_cli_linux.py --search reverb

# 查看模型详情
python msst_cli_linux.py --info melband_roformer_instvox_duality_v2.ckpt
```

### 完整参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model, -m` | 模型文件名 | 必填 |
| `--input, -i` | 输入文件或目录 | `input` |
| `--output, -o` | 输出目录 | `${PRIVATE_PATH}` |
| `--format, -f` | 输出格式 (wav/flac/mp3) | `wav` |
| `--device, -d` | 设备 (auto/cpu/cuda) | `auto` |
| `--device_ids` | GPU ID 列表 | `[0]` |
| `--tta` | 测试时增强 (质量↑, 3倍耗时) | 关闭 |

### 推理性能参考

| 模型 | GPU 耗时 (≈3min 音频) |
|------|:---:|
| duality (1.7GB) | ~5s |
| dereverb (435MB) | ~8s |
| denoise (871MB) | ~8s |

> 显存峰值约 1.1GB/卡，8 张卡可并行跑 8 条管道。

---

## 配置速查

| 项目 | 值 |
|------|-----|
| 服务器 IP | `IP_REDACTED` |
| 用户名 | `USER` |
| SSH 命令 | `ssh USER@IP_REDACTED` |
| conda 路径 | `${PRIVATE_PATH}` |
| MSST 根目录 | `${PRIVATE_PATH}` |
| 模型目录 | `${PRIVATE_PATH}` |
| 输出目录 | `${PRIVATE_PATH}` |

---

## 网络说明

| 链路 | 速度 | 用途 |
|------|:---:|------|
| 服务器 ↔ 国内 CDN | 40 MB/s | pip/git clone |
| 服务器 ↔ 阿里云盘 | 38 MB/s | 大文件传输 |
| 本机 ↔ 阿里云盘 | 10 MB/s | 上传文件 |
| 本机 ↔ 服务器 (SCP) | 250 KB/s | 少量文本/代码 |
| 服务器 → 国外 | 不通 | 必须走镜像 |

**传输策略**：大文件走阿里云盘中转，代码走 kkgithub/hf-mirror 镜像。

### 阿里云盘服务器端

```bash
# 工具路径
${PRIVATE_PATH}

# 大文件下载
${PRIVATE_PATH} download --saveto /目标目录 --sp 5 '/云盘路径'

# 列出文件
${PRIVATE_PATH} ls /路径/
```

---

## 后续扩展

如需安装更多模型，把权重文件上传阿里云盘后下载到对应分类目录即可：

```bash
# 示例：安装更多 vocal_models
# 1. 本机上传 ckpt 到阿里云盘
# 2. 服务器下载
${PRIVATE_PATH} download --saveto ${PRIVATE_PATH} --sp 5 '/新模型.ckpt'
```

所有 78 个模型列表见 `docs/MSST_models.md`。
