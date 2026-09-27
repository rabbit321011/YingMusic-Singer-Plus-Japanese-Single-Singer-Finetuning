> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# YingMusic-Singer 部署记录

## 环境信息

| 项 | 值 |
|---|---|
| 日期 | 2026-05-16 |
| OS | Windows |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop (Blackwell, SM 12.0) |
| CUDA | 12.8 |
| Python | 3.12.7 (venv) |
| PyTorch | 2.9.1+cu128 |
| flash_attn | 未安装（自动降级 PyTorch SDPA） |
| 项目路径 | `${LOCAL_PATH}` |
| 数据路径 | `${LOCAL_PATH}` |

## 环境搭建步骤

```powershell
# 1. 克隆仓库
git clone https://github.com/GiantAILab/YingMusic-Singer.git

# 2. 创建 venv（Python 3.12）
${LOCAL_PATH} -m venv .venv
.venv\Scripts\Activate.ps1

# 3. 安装 PyTorch cu128（hash 不匹配，需绕过）
pip cache purge
pip install --no-deps "https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl"
pip install --no-deps "https://download.pytorch.org/whl/cu128/torchaudio-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl"
pip install sympy typing-extensions jinja2 markupsafe networkx fsspec filelock mpmath setuptools
pip install -r requirements.txt

# 4. 下载模型权重（~4.65GB）
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -c "from huggingface_hub import snapshot_download; snapshot_download('GiantAILab/YingMusic-Singer', cache_dir='./.cache/huggingface', local_dir_use_symlinks=False)"
```

## 源代码修改清单

为了在 **Windows + PyTorch 2.9 cu128 + 无 flash_attn + 无 espeak** 环境下跑通，共修改 4 个文件：

---

### 修改 1: `src/singer/model.py` — 音频 I/O 绕过 torchcodec + 添加 `--language` 参数

**文件路径**: `src/singer/model.py`

**问题**: `torchaudio.load/save` 依赖 `torchcodec`，后者在 cu128 上 DLL 加载失败。

**修改**:

- 在 `_resampler_cache` 行后添加 `_load_audio_sf()` 函数，用 `soundfile` 读取音频
- `load_audio()` 中 `torchaudio.load(path)` → `_load_audio_sf(path)`
- `main()` 中 `torchaudio.save()` → `soundfile.write()`
- `inference()` 方法签名添加 `language: str = "auto"` 参数
- `tokenizer.tokenize()` 调用中 `language="auto"` → `language=language`
- `main()` 添加 `--language` CLI 参数（choices: auto/zh/ja/en/ko/fr/de）

```python
# === 添加的 _load_audio_sf ===
import soundfile as sf

def _load_audio_sf(path: Union[str, PathLike]) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(str(path))
    if data.ndim == 1:
        data = data[None, :]
    else:
        data = data.T
    return torch.from_numpy(data).float(), sr

# === load_audio 中 ===
audio, sr = _load_audio_sf(path)   # 原: torchaudio.load(path)

# === main() 保存部分 ===
import soundfile as sf
audio_np = generated_audio.cpu().squeeze(0).T.numpy()
sf.write(args.out_path, audio_np, SAMPLE_RATE_48K)
# 原: torchaudio.save(args.out_path, generated_audio, SAMPLE_RATE_48K)
```

---

### 修改 2: `src/singer/decoder/modules.py` — 注意力降级 + dtype 修复

**文件路径**: `src/singer/decoder/modules.py`

**问题 1**: `WanSelfAttention.forward()` 硬编码调用 `flash_attention()`，没装 flash_attn 时直接 `assert False`

**问题 2**: SDPA fallback 路径中 `attention()` 函数把 `q/k/v` 转为 `bfloat16` 计算，但返回值未恢复原始 dtype，导致后续 `self.o(x)` Linear 层 `BFloat16 × Float` 报错

**修改**:

- `WanSelfAttention.forward()` 中 `flash_attention(...)` → `attention(...)`
- `attention()` 的 SDPA fallback 分支：保存 `out_dtype = q.dtype`，返回前 `.type(out_dtype)`

```python
# === WanSelfAttention.forward 中 ===
x = attention(          # 原: flash_attention(
    q=rope_apply_1d(q, freqs),
    k=rope_apply_1d(k, freqs),
    v=v,
    k_lens=seq_lens,
    window_size=self.window_size,
)

# === attention() fallback 分支中 ===
out_dtype = q.dtype                        # 新增
# ... sdpa 计算 ...
out = out.transpose(1, 2).contiguous().type(out_dtype)  # 原: .contiguous()
```

---

### 修改 3: `src/singer/tokenizer/g2p/cleaners.py` — 日语路径绕过 espeak

**文件路径**: `src/singer/tokenizer/g2p/cleaners.py`

**问题**: `language="ja"` 时调用 `japanese_to_ipa(text, text_tokenizers["ja"])`，访问 `text_tokenizers["ja"]` 触发了 `_LazyTokenizerDict.__missing__()` → `TextTokenizer(language="ja")` → espeak 初始化失败

`japanese_to_ipa()` 实际上不需要 text_tokenizer 参数（内部只用 pyopenjtalk），但参数传递过程中 dict access 导致 espeak 加载

**修改**:

```python
# cleaners.py 第 19 行
return japanese_to_ipa(text, None)   # 原: text_tokenizers["ja"]
```

---

### 修改 4: `.venv/Lib/site-packages/LangSegment/__init__.py` — API 兼容

**文件路径**: `.venv\Lib\site-packages\LangSegment\__init__.py`

**问题**: LangSegment 0.2.0 移除了 `setLangfilters`/`getLangfilters`（改名 `setfilters`/`getfilters`），但 YingMusic-Singer 代码 import 了旧名字

**修改**:

```python
# LangSegment/__init__.py
from .LangSegment import LangSegment,getTexts,classify,getCounts,printList,setfilters,getfilters

setLangfilters = setfilters   # 新增别名
getLangfilters = getfilters   # 新增别名
```

---

## 首次推理验证

```powershell
$env:PYTHONPATH = "${LOCAL_PATH}"

python src/singer/model.py `
  --timbre_audio_path "${LOCAL_PATH}" `
  --timbre_audio_content "世界で一番お姫様" `
  --melody_audio_path "${LOCAL_PATH}" `
  --lyrics "世界で一番お姫様" `
  --out_path "outputs/test_ja.wav" `
  --cfg_strength 4.0 --nfe_steps 32 --language ja
```

**结果**: 输出 `outputs/test_ja.wav` (2235 KB, 48kHz stereo, 32步/4秒)

**定性评估**:
- 音色: 正确保留了花丸的声音特征 ✅
- 旋律: 正确捕获了 seg001 ("hello, how are you") 的音高轮廓 ✅
- 咬字: 近乎不可辨识（预期行为，日语 token embedding 未经训练） ⚠️

---

## 数据集统计

### 原始数据

| 项 | 值 |
|---|---|
| 路径 | `${LOCAL_PATH}` |
| 原始文件数 | 981 个 .wav |
| 来源歌曲数 | 93 首（不同 BV 号） |
| 原始总时长 | ~10.8 小时 |
| 平均时长/段 | ~39.7 秒 |
| 音频格式 | 44100 Hz / Mono / 16-bit |

### 清洗后

| 项 | 值 |
|---|---|
| 有效文件数 | **625 个 .wav** |
| 有效歌曲数 | **60 首** |
| 总时长 | **~6.9 小时** |
| 歌词覆盖率 | **100%（60 首全部有正确歌词）** |

**清洗过程**: 93 首中 33 首的 UTA-NET 歌词匹配错误（重复/无匹配），删除对应 356 个 wav 文件，保留 60 首正确歌曲。

---

## 歌词采集流水线

### Step 1: BV号 → 歌名

`collect_lyrics.py` — 调 Bilibili API 获取 93 首 BV 号对应的视频标题，产出 `song_names.json`。

```
BV11b421H7G3 → 【花丸晴琉】アイロニ（反语）
BV11cUCBHE9D → 花丸晴琉3D演唱会「花丸日和」歌切—18.小さきもの
...
```

### Step 2: 歌名 → UTA-NET 歌词

`fetch_lyrics.py` — 从歌名中提取核心歌曲名，搜索 uta-net.com 获取完整日文歌词，产出 `lyrics.json`（BV号→完整歌词）。

采用多策略歌名清洗：`「」`提取 → `（）`前截取 → `—`后截取 → `/`前截取。对难以自动提取的歌名（演唱会歌切、中文标题等），使用 `MANUAL` 字典手动指定。

### Step 3: Whisper 对齐

`align_lyrics.py` — 对每段 seg 用 Whisper medium 转写，通过 SequenceMatcher 滑动窗口在 UTA-NET 完整歌词中定位，提取对应段落。产出对齐后的 `lyrics_per_seg.json` 和 `lyrics.csv`。

```
原理:
  Whisper 转写 → 粗糙文本（可能 70-80% 正确）
  SequenceMatcher 滑动窗口 → 在完整歌词中找最佳匹配位置
  从 UTA-NET 原歌词中提取该位置段 → 100% 准确的歌词文本 ✅

验证 (BV1difiB6ETW, なんでもないや, 14段):
  ratio 范围: 0.77 ~ 0.98，全部完美对齐
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `song_names.json` | BV号 → B站视频标题 |
| `lyrics.json` | BV号 → UTA-NET 完整歌词 |
| `lyrics_per_seg.json` | seg stem → 对齐后的歌词段落 |
| `lyrics.csv` | train.py 直接可用的格式（stem\|lyrics）|

---

## 训练脚本

### 脚本位置

`train.py`（项目根目录）

### 训练原理

- **全自监督**: 每条 seg 用自己的前半段做音色条件 (cond)，整段做 GT (x₁)
- **Flow Matching**: 线性路径 `xₜ = (1-t)x₀ + tx₁`，速度目标 `v = x₁ - x₀`
- **CFG 训练**: 15% 概率 drop_text，让 DiT 同时学习有文本/无文本
- **冻结模块**: VAE, RMVPE, SOME, Tokenizer
- **可训练**: 仅 DiT (singer.decoder)，~453M 参数

### 歌词格式

`--lyrics_file` 支持 JSON 或 CSV：

```json
{"seg000": "世界で一番お姫様", "seg001": "そう君は今だって"}
```

或 CSV（`|` 分隔）：

```
seg000|世界で一番お姫様
seg001|そう君は今だって
```

`--dummy_lyrics` 可在无歌词时用占位文本（如 `"a"`）快速验证训练流程。

### 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--data_dir` | source_singers | 训练数据目录 |
| `--max_steps` | 50000 | 总训练步数 |
| `--lr` | 1e-4 | 初始学习率 |
| `--lr_min` | 1e-6 | Cosine 最低学习率 |
| `--warmup_steps` | 3000 | 线性 warmup 步数 |
| `--grad_accum` | 4 | 梯度累积步数 |
| `--clip_grad` | 10.0 | 梯度裁剪阈值 |
| `--cfg_drop_prob` | 0.15 | CFG 文本丢弃概率 |
| `--max_seg_seconds` | 12.0 | 单段最大秒数（截断） |
| `--use_amp` | True | bf16 混合精度 |

### 训练验证（2026-05-16）

```powershell
# 10条数据 + 假歌词 "a"，50步快速验证
python train.py --data_dir "${LOCAL_PATH}" `
  --dummy_lyrics "a" --max_steps 50 --grad_accum 1 `
  --output_dir ./test_run --language ja --max_files 10
```

**结果**:

| 指标 | 值 |
|------|----|
| Precompute | 10 段 / 54 秒 |
| 训练 | 50 步 / 12 秒 |
| 最终 Loss | **0.9792**（正常下降，无 NaN）|
| OOM | 无 |
| DiT 参数 | 329,038,912 |
| 50步后推理 | 音色=花丸 ✅ 旋律=hello how are you ✅ 咬字=糊（预期） |

验证通过，训练脚本无 bug。

### 正式训练命令

```powershell
$env:PYTHONPATH = "${LOCAL_PATH}"

# 全量训练（625段，~6.9h，60首歌）
python train.py --data_dir "${LOCAL_PATH}" `
  --lyrics_file "./lyrics.csv" --language ja `
  --max_steps 50000 --output_dir ./finetune_output
```

### 推理（加载微调权重）

```python
# 训练脚本保存的 checkpoint 只含 DiT 权重，需手动加载
import torch, soundfile as sf
from singer.model import YingSinger, SAMPLE_RATE_48K

singer = YingSinger(device='cuda')
ckpt = torch.load('finetune_output/checkpoints/step_050000_final.pt', map_location='cuda')
singer.singer.decoder.load_state_dict(ckpt['model_state_dict'])

audio = singer.inference(
    timbre_audio_path=r'...seg000.wav',
    timbre_audio_content='...',
    melody_audio_path=r'...seg001.wav',
    lyrics='世界で一番お姫様',
    cfg_strength=4.0, nfe_steps=32, language='ja')

audio_np = audio.cpu().squeeze(0).T.numpy()
sf.write('outputs/finetuned.wav', audio_np, SAMPLE_RATE_48K)
```

> ⚠️ `--ckpt_path` 接受的是模型**目录**（含 `singer.v1.pt` 等），不是单个 checkpoint。因此 CLI 无法直接加载微调权重，需用 Python 脚本。

---

---

# 服务器部署记录 (2026-05-23)

> 服务器：`USER@IP_REDACTED` (8×RTX 4090)

## 服务器环境

| 项 | 值 |
|---|---|
| conda 环境 | `yysinger` |
| Python | 3.12.13 |
| PyTorch | 2.9.1+cu128 |
| flash_attn | ❌ 未安装（无 nvcc，走 SDPA 回退） |
| espeak-ng | ❌ 未安装（无 sudo，英文推理不可用，日语路径已绕过） |
| 项目路径 | `${PRIVATE_PATH}` |
| 代码来源 | `ghfast.top` 镜像 clone（github/kkgithub/gitclone/ghproxy 均不可用） |

## 代码获取

```bash
# 可用镜像：ghfast.top
# 不可用：github.com, kkgithub.com, gitclone.com（ls-remote 通但 clone 超时）, ghproxy.com, mirror.ghproxy.com, github.com.cnpmjs.org
git clone --depth 1 https://ghfast.top/https://github.com/GiantAILab/YingMusic-Singer.git ${PRIVATE_PATH}
```

## 模型权重下载

```bash
# hf-mirror.com 可用，~2.5GB 约 3 分 21 秒
source ${PRIVATE_PATH} && conda activate yysinger
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('GiantAILab/YingMusic-Singer', local_dir='${PRIVATE_PATH}')
"
```

权重文件（5 个，共 ~2.7GB）：

| 文件 | 大小 | 用途 |
|------|------|------|
| `singer.v1.pt` | 1.3 GB | DiT 主模型 |
| `autoencoder_music_dsp1920.ckpt` | 644 MB | VAE |
| `some.pt` | 449 MB | 旋律提取器（SOME teacher） |
| `rmvpe.pt` | 352 MB | F0 提取器 |
| `stable_audio_1920_vae.json` | 4 KB | VAE 配置 |

## 源码修改清单（4 处）

与 Windows 本地相同的修改，通过脚本自动化应用：

### 修改 1: `src/singer/model.py`

- `torchaudio.load/save` → `soundfile`（`_load_audio_sf`）
- `inference()` 添加 `language: str = "auto"` 参数
- `tokenizer.tokenize()` 传入 `language=language`
- `main()` 添加 `--language` CLI 参数

### 修改 2: `src/singer/decoder/modules.py`

- `WanSelfAttention.forward()` 中 `flash_attention()` → `attention()`
- `attention()` 的 SDPA fallback 分支：保存 `out_dtype_sdpa = q.dtype`，返回前 `.type(out_dtype_sdpa)` 修复 dtype 不匹配

### 修改 3: `src/singer/tokenizer/g2p/cleaners.py`

- `japanese_to_ipa(text, text_tokenizers["ja"])` → `japanese_to_ipa(text, None)` 绕过 espeak

### 修改 4: LangSegment 兼容

- Linux 无版本冲突，**无需修改**

## 推理验证

### 测试 1：旋律参考 ≠ 音色参考

```bash
cd ${PRIVATE_PATH} && PYTHONPATH=src python3 -c "
from singer.model import YingSinger
singer = YingSinger(singer_path='.', device='cuda:0')
output = singer.inference(
    timbre_audio_path='hanamaru_hareru_singer_sum/datas/BV1138tzCEuP_..._seg000.wav',
    timbre_audio_content='たぶん私じゃなくていいね 余裕のない二人だったし...',
    melody_audio_path='hanamaru_hareru_singer_sum/datas/BV136421f7V5_..._seg002_hard.wav',
    lyrics='たぶん私じゃなくていいね',
    cfg_strength=4.0, nfe_steps=32, language='ja')
"
```

| 指标 | 值 |
|------|----|
| 输出 | `test_ja.wav` |
| 时长 | 29.5s |
| 推理速度 | 32 步 / ~1.5s (GPU cuda:0) |
| 输出大小 | 5.5 MB, 48kHz 立体声 |

### 测试 2：旋律参考 = 音色参考

音色和旋律使用同一个 `seg000`，目标歌词 `たぶん私じゃなくていいね`（= 原词第一句）。

| 指标 | 值 |
|------|----|
| 输出 | `test_ja_selfcons.wav` |
| 时长 | ~11.7s |
| 推理速度 | 32 步 / 0.87s |
| 输出大小 | 2.2 MB, 48kHz 立体声 |

**听力评估**：
- ✅ 音色正确（花丸的声音质地）
- ✅ 旋律正确（跟随参考音频）
- ⚠️ 咬字模糊 — 参考文本后半段泄漏到输出（如 "no i i ka ma ke se...ho ne e n ne i" 对应原文 `余裕のない...喧嘩...ごめんね`）

### 测试 3：音素对齐测试

目标歌词改为 `好きという気持ちまた香る`（23 音素，与原词第一句 `たぶん私じゃなくていいね` 的 23 音素完全对齐）。

```bash
# 音素: su ki to i u ki mo chi ma ta ka o ru = 23 音素
# 期待听到的罗马音: su-ki-to-i-u-ki-mo-chi-ma-ta-ka-o-ru
```

**Whisper large-v3 转录结果**：

```
[0.0s - 3.3s]  あぶなかっか僕でいえねえ       ← 期待: たぶん私じゃなくていいね
[3.3s - 6.4s]  のいきき負けさていないとし       ← 期待: 余裕のない二人だったし
[6.4s - 9.4s]  おぎわかむてくっかにしかさ       ← 期待: 気づけば喧嘩ばっかりしてさ
[9.4s - 11.7s]  おねんね                         ← 期待: ごめんね
```

**全文 = 参考文本的模糊复读，目标歌词 `好きという気持ちまた香る` 完全未出现。**

### 🔴 结论

**V1 预训练权重的日语 text conditioning 为零。** DiT 的 366 词表中日语音素（`ɯ`、`ɴ`、`ɕ`、`ç`、`ʑ`、`dʑ` 等）对应的 512 维 embedding 从未被训练过，CFG 也无效（"有文本"和"无文本"两条路径都不认识日语）。模型在推理时忽略文本条件，仅跟随音色 + 旋律复现参考音频中的原始发音。

**仅剩一条路：用 98.86h 日语数据微调 DiT。**

## 可用数据集（服务器上）

| 数据集 | 路径 | 条数 | 时长 | 歌词来源 |
|------|------|:---:|------|------|
| final_sum_large | `${PRIVATE_PATH}` | 16,492 | 98.86h | Whisper large-v3 转录（未人工校对） |
| hanamaru_hareru_singer_sum | `${PRIVATE_PATH}` | 3,705 | ~22.4h | Whisper 转录 + 旧版清洗 |

---

# V1 日语故障诊断 (2026-05-23)

## 故障点逐层排查

| 环节 | 测试方法 | 结论 |
|---|---|---|
| G2P（日文→音素） | `PhonemeBpeTokenizer.tokenize("たぶん私じゃなくていいね", language="ja")` | ✅ 23 音素，完全正确 |
| Embedding（音素→DiT 输入） | 对比日文独有音素 vs 通用音素 vs 随机初始化的 L2 范数 | ⚠️ 日文 token embedding 从未被前向传播触发，范数=19.2（接近通用音素 18.6，远离随机 22.6），但来自 AdamW weight decay 尾迹，非真正梯度更新 |
| CFG（日文） | JA CFG=0 vs 4.0 的 MSE diff + Whisper 转录 | ⚠️ diff=0.021（EN diff=0.038），但 Whisper 显示均无意义碎片 → 差异来自随机种子，CFG 实际无效 |
| 中文泛化（EN/CN 原版语种） | female.wav 推理 | ✅ EN "the weather is beautiful today" — Whisper 清晰可辨 |

**根因**：DiT 的 366 词表中日文独有音素（`ɯ,ɴ,ɕ,ç,ʑ,dʑ` 等）的 embedding 在 20,240 条中英训练数据中从未被梯度更新过。`nn.Embedding` 不是 `nn.Linear`，`initialize_weights()` 未覆盖它，保留 PyTorch 默认 `N(0,1)`。训练只包含 CN+EN 音素 → 日文 token 从未被 look up。

## 结论

- G2P ✅ / Embedding ❌（无梯度） / DiT 文本→mel 映射 ❌（从未建立）
- 本质：DiT 没学过"日语音素序列 → mel 频谱"的映射
- 微调可同时解决 Embedding + Transformer 两方面

## espeak-ng 安装（中文/英文推理依赖）

服务器无 sudo，手动从 Ubuntu 24.04 apt 仓库下载 deb 包并解压：

```bash
cd /tmp
apt download espeak-ng libespeak-ng1 espeak-ng-data libpcaudio0 libsonic0
for deb in *.deb; do dpkg -x "$deb" espeak_pkg; done

# 复制到 conda 环境
cp espeak_pkg/usr/bin/espeak-ng ~/.conda/envs/yysinger/bin/
cp espeak_pkg/usr/lib/x86_64-linux-gnu/lib*.so* ~/.conda/envs/yysinger/lib/
cp -r espeak_pkg/usr/lib/x86_64-linux-gnu/espeak-ng-data ~/.conda/envs/yysinger/share/

# 推理脚本需要设置
export LD_LIBRARY_PATH=${PRIVATE_PATH}
export ESPEAK_DATA_PATH=${PRIVATE_PATH}
BaseEspeakBackend.set_library('${PRIVATE_PATH}')
```

---

# 日语微调训练 (2026-05-24)

## 数据集

| 集 | 路径 | 条数 | 来源 |
|---|---|---|---|
| Train | `${PRIVATE_PATH}` | 15,122 | 按 BV 号 95/5 分割 |
| Test | `${PRIVATE_PATH}` | 307 | BV 号隔离 |

格式：`[{Path, Duration, Text, Language}]`，全日语 `Language: "ja"`。

## train_v2.py 改动（vs 原版 train.py）

| 改动 | 原版 | v2 |
|---|---|---|
| 数据 | `--data_dir` 遍历 .wav + `--lyrics_file` | 直接读 singnet.json |
| 分布式 | 单卡 | torchrun DDP（`DistributedSampler + DDP`） |
| 预计算 | 所有卡都跑（重复 6×） | rank 切片分配给各卡 + `torch.save/load` 磁盘缓存 |
| 训练/测试 | 无分割 | 按 BV 号 95/5 分割 |
| Eval | 无 | 每 2000 步 MSE(no diffusion, t=0.5) |
| LR | 1e-4 | 6e-5 → 2e-5 |
| warmup | 3000 步 | 不变 |
| 精度 | bf16 | 不变 |
| 每 epoch | 15,122 ÷ 4 GPU ≈ 3,780 步 | |

## 运行记录

### Run 1: LR=6e-5（过拟合）

| 步数 | Train | Eval | 现象 |
|---|---|---|---|
| 2000 | ~1.0 | 2.29 | |
| 6000 | ~0.62 | 1.73 | eval 谷底 |
| 10000 | ~0.45 | 1.71 | eval 振荡不再降 |
| 12000 | ~0.42 | 1.77 | eval 回升 ← 过拟合 |

1 epoch 就开始过拟合。杀掉。

### Run 2: LR=2e-5（成功）

| Epoch | Step | Eval Loss |
|---|---|---|
| 0.5 | 2000 | 0.68 |
| 1 | 4000 | 0.51 |
| 2 | 8000 | 0.37 |
| 3 | 12000 | 0.36 |
| 5 | 20000 | 0.35 |
| 10 | ~38000 | 0.34 |
| 13 | 50000 | **0.33** |

全程 eval 单调下降，无震荡，无过拟合。LR=2e-5 对这 98.86h 日语数据是甜蜜点。

**命令**：
```bash
export CUDA_VISIBLE_DEVICES=2,3,4,5  # 0-1 别人在用
torchrun --nproc_per_node=4 train_v2.py \
    --train_json ${PRIVATE_PATH} \
    --test_json  ${PRIVATE_PATH} \
    --output_dir ${PRIVATE_PATH} \
    --language ja --lr 2e-5 --lr_min 1e-6 \
    --warmup_steps 3000 --max_steps 50000 --seed 42
```

## 已知问题

- Whisper 转录残留中文汉字（`呀`→pyopenjtalk `Cannot read`），不影响训练但增加 eval loss 天花板
- GPU 0-1 被别人占用，只用 2-5 跑训练

## 推理评估（2026-05-24，日语 38k checkpoint 最佳）

| 维度 | 效果 |
|---|---|
| 音色 | ✅ 花丸音色可辨识 |
| 咬字（同歌） | ✅ `私じゃなくて` `削ぎ落としてく` Whisper 可辨识（vs V1 的 `あぶなかっか僕で`） |
| 咬字（跨歌手） | ⚠️ 碎片化（`文学は` 首词命中） |
| 旋律跟随 | ❌ 非精确（软性参考而非强制乐谱） |

---

# 旋律跟随问题的根因分析

## YingMusic-Singer inference 流程

```python
timbre_melody = extract(timbre_audio)      # 音色参考的旋律
melody        = extract(melody_audio)       # 目标旋律
combined = [timbre_melody | silence | melody]  # 拼接
```

`timbre_audio` 强制拼接在 `melody_audio` 前面，用于"音色建立"。输出前段是音色参考的旋律，后段切到目标旋律。这是零样本音色克隆的代价。

## CFG 仅对文本生效

```python
# model.py L530-540: cfg_infer=True 时
pred_cfg = decoder(x, cond, text, melody, cfg_infer=True)
pred, pred_wo_text = pred_cfg.chunk(2)  # uncond 路径：drop_text=True, drop_melody=False
```

只有文本有无的条件对比。**没有 melody CFG**。

## 三重力量的平衡

```
输出 ≈ 音色条件(弱) + 文本 × CFG_strength(硬) + 旋律(软参考)
```

| 参数 | 可控性 |
|---|---|
| `text_strength` (CFG) | ✅ 1.0–10.0 |
| melody weight | ❌ 不暴露 |

中文跨歌手实测（泠鸢yousa→ChiliChill），CFG=1.0/2.0/4.0 三个值输出几乎无差异——旋律始终飞。降低 text_strength 没有帮助。

## 结论

**YingMusic-Singer V1 的 "annotation-free melody guidance" 是软性参考，不是精确旋律控制。** 这和论文定位一致，但对"跨歌换词+保旋律"的需求不够。

---

# YingMusic-Singer-Plus（ASLP-lab，2026年3月）

## 与 V1 的关键对比

| 特性 | YingMusic V1 | YingMusic-Singer-Plus |
|---|---|---|
| 架构 | Wan2.1 DiT (24层) | F5-TTS DiT (22层) |
| VAE | SA1920 (48kHz) | SA2 (44.1kHz) |
| 旋律 | SOME (软性参考) | SOME + **CKA alignment loss + GRPO** |
| 训练 | SFT | SFT + **GRPO 强化学习** |
| 语言 | CN+EN | CN+EN |
| 总参数 | ~330M DiT | ~727M (453M CFM + 156M VAE + 118M 旋律) |
| 官方定位 | Zero-shot SVS | **"strong melody preservation"** |
| 许可证 | CC BY 4.0 | CC BY 4.0 |

## 权重（HuggingFace, 已开源）

```
ASLP-lab/YingMusic-Singer-Plus/
├── model.safetensors                          # 453.6M CFM DiT
├── ckpts/stable_audio_2_0_vae_20hz_official.ckpt  # VAE
├── ckpts/YingMusicSinger_model.pt             # 模型配置
├── ckpts/MelBandRoformer.ckpt                 # 人声分离
├── ckpts/model_ckpt_steps_100000_simplified.ckpt  # SOME 旋律提取
└── config.json
```

---

---

# YingMusic-Singer-Plus 部署记录 (2026-05-24)

## 环境搭建

新建独立 conda 环境（V1 的 `yysinger` 是 Python 3.12 / torch 2.9，不兼容）：

```bash
conda create -n yingmusic_plus python=3.10 -y
conda activate yingmusic_plus
pip install uv
cd ${PRIVATE_PATH}
uv pip install -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/simple
```

| 项 | 值 |
|---|---|
| 环境名 | `yingmusic_plus` |
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| 包数量 | 261 |
| GPU | 8× RTX 4090 |

### 代码与权重

```bash
# 代码（ghfast.top 镜像）
git clone https://ghfast.top/https://github.com/ASLP-lab/YingMusic-Singer-Plus.git

# 权重（hf-mirror.com, ~12.3GB）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
    ASLP-lab/YingMusic-Singer-Plus --local-dir . --local-dir-use-symlinks False
```

| 文件 | 大小 | 用途 |
|------|------|------|
| `YingMusicSinger_model.pt` | 7.6 GB | CFM DiT + EMA |
| `model.safetensors` | 2.8 GB | HF packaged（含 midi_teacher + VAE） |
| `MelBandRoformer.ckpt` | 871 MB | 人声分离器 |
| `stable_audio_2_0_vae_20hz_official.ckpt` | 596 MB | Stable Audio 2 VAE |
| `model_ckpt_steps_100000_simplified.ckpt` | 449 MB | SOME MIDI Teacher |

### espeak-ng 安装

`yingmusic_plus` 环境缺 espeak，中文 G2P 需要。**需 sudo**：

```bash
sudo apt-get install -y espeak-ng libespeak-ng-dev
```

## 推理模式

| 模式 | ref_audio | melody_audio | 效果 |
|------|-----------|-------------|------|
| **Sing Edit**（同歌改词） | A 歌 | A 歌（同一文件） | ✅ 旋律完美、咬字清晰 |
| **Melody Control**（跨角色） | A 人+B 歌 | B 人+B 歌（官方 example） | ✅ 旋律可跟 |
| **Cross-Singer Cover**（翻唱） | A 人+B 歌 | C 人+C 歌 | ⚠️ 部分跑调，取决于音色兼容性 |

**加载方式**：优先 `from_pretrained`（和 Gradio app 一致），手动构造也可：

```python
model = YingMusicSinger.from_pretrained('${PRIVATE_PATH}')
model.to('cuda')
model.eval()

waveform, sr = model(
    ref_audio_path='timbre.wav',
    melody_audio_path='melody.wav',
    ref_text='timbre audio 的实际歌词',
    target_text='要唱的词|用|分隔',
    seed=42,
)
```

## 关键踩坑总结

### 🔴 杂音/噪声：ref_text 必须精确匹配 ref_audio 内容

这是最容易出问题的点。`ref_text` **不是**随便写一句参考音色说的话——必须是 ref_audio 里**实际唱的歌词**。

| 错误 | 正确 |
|------|------|
| `ref_text='我时常对自己失望'` 对应 30s 音频 | `ref_text='我时常对自己失望没有...害怕被世界遗忘'` 对应 30s 音频的全部歌词 |

模型用 ref 区域建立"这个音色 → 这些音素"的映射。如果 ref_text 只有 1 句但音频有 9 句，映射全乱，输出全是随机噪声。

### 🔴 长音频限制：1~2 句最佳，30s 封顶

80s 的段在 sing-edit 模式下产生全程杂音。30s（约 3-4 句歌词）是安全上限。更长的音频需分段推理。

原因：`max_duration=4096` frames，VAE 帧率 21.53Hz → 最大 190s。但长序列内存压力大（80s/1723 frames 约需 2.6GB），且模型对长文本对齐能力有限。

### 🟡 跨角色翻唱：音色兼容性影响旋律

| 音色 | 旋律 | 效果 |
|------|------|------|
| 阿梓（女声） | ChiliChill（男声） | ⚠️ 稍微跑调，勉强可用 |
| 泠鸢yousa（女声） | ChiliChill（男声） | ❌ 完全跑调 |
| 官方 example（女声） | 官方 example（女声） | ✅ 正常 |

音色越接近训练数据分布、和旋律源的特征越兼容，效果越好。

### 🟢 权重确认：HF 发布的就是 GRPO 精调版

经 Grok 确认 + 官方 SingEdit 示例验证：`ASLP-lab/YingMusic-Singer-Plus` 是 SFT + GRPO 完整版，非中间 SFT 权重。`is_distilled: false` 仅表示非蒸馏版。

### 🟢 nfe_step 对 melody 影响不大

CFG=0→9、nfe=32→64 对跨角色翻唱的旋律跑调无显著改善。问题不在采样参数，在音色域匹配。

## 本地文档

- `docs/architecture-overview.md` — V1 架构详解
- `docs/architecture-plus.md` — Plus 架构详解（含 V1 vs Plus 完整对比）

## 待办

- [x] 歌词标注 (BV号 → 歌名 → UTA-NET → 删除错误 → Whisper 对齐)
- [x] 10条/50步训练验证
- [x] 服务器部署 + 推理跑通 (2026-05-23)
- [x] Whisper 验证：确认 V1 权重日语 text conditioning = 零
- [x] 用 final_sum_large（98.86h）微调 DiT（LR=2e-5, 50k 步）
- [x] 推理评估（38k 最佳，咬字质变，旋律软性）
- [x] 旋律问题根因分析（无 melody CFG，架构限制）
- [x] YingMusic-Singer-Plus 部署 + 推理验证
- [x] 跨角色翻唱测试（阿梓+ChiliChill 勉强可用，1~2句分段）
- [x] Plus G2P 日语支持改造（pyopenjtalk + japanese.py）
- [x] Plus SFT 训练脚本 + GRPO 训练脚本
- [x] Plus 日语全量 SFT 训练启动（30k步/4GPU/90.7h数据）
- [ ] SFT 训练完成 + 评估
- [ ] GRPO 精调（按需）

---

# Plus 日语 SFT 训练 (2026-05-24)

> 训练进行中 — 全量 90.7h 花丸晴琉日语干声，4×RTX 4090 DDP

## 动机

Plus 的 CNENTokenizer G2P 不支持日语（`chn_eng_g2p` 硬编码中英），DiT 的 373 词表中日语音素 embedding 从未被梯度更新。需要修改 G2P 前端 + 日语数据微调。

## G2P 日语支持改造

### 查证结果

| 组件 | 原判断 | 实际 | 结论 |
|------|--------|------|------|
| espeak-ng `ja` backend | ❌ | "世界"→"chinese letter"，不支持汉字 | 不可用 |
| pyopenjtalk 0.4.1 | — | `こんにちは世界` → `k o N n i ch i w a s e k a i` ✅ | **采用** |
| vocab.json 日语音素 | 28 (318-345) | ✅ 覆盖日语 IPA | 无需修改 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `g2p/g2p/japanese.py` | 从 V1 项目复制（814行，pyopenjtalk + 自定义 IPA 映射表） |
| `g2p/g2p/cleaners.py` | +1 行导入 `japanese_to_ipa` |
| `g2p/g2p_generation.py` | `is_japanese()` + `has_japanese()` 检测；日语文本直接走 `japanese_to_ipa` (bypass 字符级分词) |
| `cnen_tokenizer.py` | 导入 `chn_eng_jpn_g2p` |

### 关键设计决策

日文汉字（如 "私"）的 Unicode 范围 `\u4e00-\u9fa5` 与中文重叠。`get_segment()` 无法区分。解决方案：`has_japanese()` 检测文本含假名 → 整个文本直走 `japanese_to_ipa`，完全 bypass 语言检测。

### 数据脏样本

Whisper 自动标注中存在幻觉：`"呸長阿為八阿為一生唯一財命"`、`"殺戮の手"` 等乱码字。pyopenjtalk 对不认识的字输出 "Cannot read" 并跳过，不影响训练。15,122 条中仅 ~10 条有此类问题。

## 训练配置

| 参数 | 值 |
|------|-----|
| 数据集 | `final_sum_large/train_singnet.json` (15,122条, 90.7h, 100% ja) |
| 验证集 | `final_sum_large/test_singnet.json` (307条, BV号隔离) |
| GPU | 4×RTX 4090 (物理 0-3 / `CUDA_VISIBLE_DEVICES=2,3,4,5`) |
| Batch | 6/GPU × 4GPU = 24 (grad_accum=1) |
| LR | 7e-6 (warmup 500步 → cosine decay) |
| Max steps | 30,000 |
| Max duration | 12s/段 |
| CKA weight | 1.0 |
| 精度 | fp32 (未启用 amp) |
| 速度 | ~0.78-0.81s/step |
| ETA | ~6.3h |

### Tokenize 方式

训练时直接在 `train_plus.py` 内 bypass CNENTokenizer：`japanese_to_ipa(text) → pbt.phoneme2token(phoneme)`，避免 `get_segment()` 的中日文歧义。

### 训练进程

```
Step    50: Flow=22.94  CKA=0.624  LR=7.0e-07  (warmup)
Step   500: Flow= 1.67  CKA=0.196  LR=7.0e-06  (峰值 LR)
Step  1000: Flow= 1.03  CKA=0.168  LR=7.0e-06  Eval=1.063 ✅
Step  1700: Flow= 0.67  CKA=0.162  LR=6.97e-06 (5.7%)
```

## 训练产物

| 文件 | 说明 |
|------|------|
| `train_plus.py` | SFT 训练脚本：Flow+CKA+EMA+DDP+eval |
| `grpo_train.py` | GRPO 训练脚本：Whisper+WavLM+torchcrepe→PPO |
| `infer_ft.py` | 加载微调权重跑推理 |
| `run_sft.sh` | 启动脚本 |
| `train_sft.log` | 训练日志 |
| `ckpts/plus_ja_sft/step_*` | EMA checkpoint（每2000步） |

## GRPO 奖励模型

| 模型 | 来源 | 路径 | 状态 |
|------|------|------|:--:|
| Whisper | faster-whisper medium | 已有 HF cache | ✅ |
| WavLM Large | hf-mirror 下载 | `${PRIVATE_PATH}` (1.2GB) | ✅ |
| torchcrepe (F0) | pip install | `yingmusic_plus` 已装 | ✅ |

GRPO 脚本已通过管道验证（8采样→奖励→PPO 更新成功）。SFT 完成后按需启动。
