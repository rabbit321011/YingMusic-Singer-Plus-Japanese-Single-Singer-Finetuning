> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# MSST 音频分离工具 — 操作手册 (给 cyanAI 阅读)

> **工具类型**: 通过 `run_powershell` 间接调用
> **Python venv**: `${LOCAL_PATH}`
> **CLI 脚本**: `${LOCAL_PATH}`
> **工作目录**: `${LOCAL_PATH}`

---

## 调用方式

通过 `run_powershell` 工具执行，命令模板：

```powershell
& '${LOCAL_PATH}' `
  '${LOCAL_PATH}' `
  --model "<模型文件名>" `
  --input "<输入音频绝对路径>" `
  -o "<输出目录绝对路径>"
```

> **重要**: `--input` 和 `-o` 必须是绝对路径。
> **超时**: `run_powershell` 超时 500 秒，单个模型一般 15-60 秒。

### 输出文件位置与命名

- 默认输出目录为 `${LOCAL_PATH}`，可通过 `-o` 指定其他目录
- 输出文件命名规则：`{原始文件名}_{输出轨名}.{格式}`
- 例如输入 `${LOCAL_PATH}` ，使用 duality 模型分离：
  ```
  ${LOCAL_PATH}         ← 人声
  ${LOCAL_PATH}   ← 伴奏
  ```
- 输出轨名对应上方表格的"输出内容"列（如 Vocals、dry、noreverb、karaoke 等）

---

## 任务 → 模型映射表

根据用户需求直接选择模型名。

### 一、人声/伴奏分离

| 用户需求 | 模型名 | 输出内容 |
|---------|--------|---------|
| 分离人声和伴奏（通用首选） | `melband_roformer_instvox_duality_v2.ckpt` | Vocals / Instrumental |
| 纯人声提取（激进去伴奏） | `big_beta5e.ckpt` | vocals / other |
| 提取伴奏（反向，保留伴奏 v2） | `melband_roformer_inst_v2.ckpt` | other / vocals |
| 提取伴奏（反向 v1） | `inst_v1e.ckpt` | other / vocals |
| 卡拉OK（去掉人声留伴奏） | `model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | karaoke / other |
| 男声和女声分开 | `bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt` | male / female |
| BS架构高质量人声 | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | vocals / other |
| BS架构人声（备选） | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` | vocals / instrumental |
| BS架构大型版 | `BS-Roformer_LargeV1.ckpt` | vocals / other |
| Kimberley 人声模型 | `Kim_MelBandRoformer.ckpt` | vocals / other |
| Kimberley Unwa 微调版 | `kimmel_unwa_ft.ckpt` | vocals / other |
| Becruily 人声提取 | `mel_band_roformer_vocals_becruily.ckpt` | vocals / other |
| Becruily 伴奏提取 | `mel_band_roformer_instrumental_becruily.ckpt` | Instrumental / Vocals |
| 通用人声（mel_band） | `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt` | vocals / other |
| 轻量人声（快） | `model_vocals_mel_band_roformer_sdr_8.42.ckpt` | vocals / other |
| HTDemucs 人声 | `model_vocals_htdemucs_sdr_8.78.ckpt` | vocals / other |
| MDX23C 人声 | `model_vocals_mdx23c_sdr_10.17.ckpt` | vocals / other |
| Segm Models 人声 | `model_vocals_segm_models_sdr_9.77.ckpt` | vocals / other |
| Swin UperNet 人声 | `model_swin_upernet_ep_56_sdr_10.6703.ckpt` | vocals / other |
| 提取 Bass 轨 | `HTDemucs4_FT_bass.th` | bass / other |
| 提取 Drum 轨 | `HTDemucs4_FT_drums.th` | drums / other |
| 提取 Other 轨 | `HTDemucs4_FT_other.th` | other / other |
| HTDemucs 官方人声 | `HTDemucs4_FT_vocals_official.th` | vocals / other |
| 单人声提取（BS单轨） | `model_bs_roformer_ep_937_sdr_10.5309.ckpt` | vocals / other |
| 音频相似度/差异分离 | `model_mdx23c_ep_271_l1_freq_72.2383.ckpt` | similarity / difference |

### 二、去混响/去回声

| 用户需求 | 模型名 | 输出内容 |
|---------|--------|---------|
| 去混响+去回声（首选） | `dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt` | dry / other |
| 温和去混响（保自然感） | `dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt` | noreverb / reverb |
| 强力去混响 | `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt` | noreverb / reverb |
| 大混响专用 | `de_big_reverb_mbr_ep_362.ckpt` | dry / other |
| 标准去混响 | `deverb_mel_band_roformer_ep_27_sdr_10.4567.ckpt` | noreverb / reverb |
| 轻量去混响（快） | `deverb_bs_roformer_8_256dim_8depth.ckpt` | noreverb / reverb |
| 重量去混响 | `deverb_bs_roformer_8_384dim_10depth.ckpt` | noreverb / reverb |
| MDX23C 去混响 | `dereverb_mdx23c_sdr_6.9096.ckpt` | dry / other |
| UVR 去回声+去混响 | `UVR-DeEcho-DeReverb.pth` | No Reverb / Reverb |
| UVR 激进去回声 | `UVR-De-Echo-Aggressive.pth` | No Echo / Echo |
| UVR 标准去回声 | `UVR-De-Echo-Normal.pth` | No Echo / Echo |
| UVR 去混响（轻量） | `UVR-DeReverb-aufr33-jarredou_4band_v4_ms_fullband.pth` | Dry / Reverb |

### 三、降噪

| 用户需求 | 模型名 | 输出内容 |
|---------|--------|---------|
| 通用降噪（首选） | `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` | dry / other |
| 激进降噪 | `denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt` | dry / other |
| UVR 降噪 | `UVR-DeNoise.pth` | Noise / No Noise |
| UVR 轻量降噪 | `UVR-DeNoise-Lite.pth` | Noise / No Noise |

### 四、特殊处理

| 用户需求 | 模型名 | 输出内容 |
|---------|--------|---------|
| 去气声/呼吸声 | `aspiration_mel_band_roformer_sdr_18.9845.ckpt` | aspiration / other |
| 温和去气声 | `aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt` | aspiration / other |
| 去人群噪声（Live现场） | `mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt` | crowd / other |
| 低质量MP3修复 | `Apollo_LQ_MP3_restoration.ckpt` | restored / addition |
| Apollo 通用修复 | `apollo_model_uni.ckpt` | restored / addition |
| 谐波/气声分离（VR） | `Harmonic_Noise_Separation_yxlllc.pth` | No Aspiration / Aspiration |

### 五、多轨分离

| 用户需求 | 模型名 | 输出内容 |
|---------|--------|---------|
| 鼓/贝斯/其他/人声四轨（首选） | `bs_roformer_4stems_ft.ckpt` | drums / bass / other / vocals |
| SCNet 四轨分离 | `model_scnet_sdr_9.3244.ckpt` | drums / bass / other / vocals |
| SCNet 四轨轻量版 | `scnet_checkpoint_musdb18.ckpt` | drums / bass / other / vocals |
| MDX23C 四轨分离 | `model_mdx23c_ep_168_sdr_7.0207.ckpt` | vocals / bass / drums / other |
| HTDemucs4 多轨分离 | `HTDemucs4.th` | 4 stems |
| 鼓组细分离（6轨） | `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt` | kick / snare / toms / hh / ride / crash |
| 鼓组四件分离 | `model_drumsep.th` | kick / snare / cymbals / toms |
| 语音/音乐/音效分离 | `model_bandit_plus_dnr_sdr_11.47.chpt` | speech / music / effects |
| 六轨分离（含吉他钢琴） | `HTDemucs4_6stems.th` | drums / bass / other / vocals / guitar / piano |

### 六、UVR 人声/伴奏（小模型，快速）

| 用户需求 | 模型名 | 输出内容 |
|---------|--------|---------|
| UVR 背景人声提取 | `UVR-BVE-4B_SN-44100-1.pth` | Vocals / Instrumental |
| HP 去人声 #1 | `1_HP-UVR.pth` | Instrumental / Vocals |
| HP 去人声 #2 | `2_HP-UVR.pth` | Instrumental / Vocals |
| HP 提取人声 #1 | `3_HP-Vocal-UVR.pth` | Vocals / Instrumental |
| HP 提取人声 #2 | `4_HP-Vocal-UVR.pth` | Vocals / Instrumental |
| HP 卡拉OK #1 | `5_HP-Karaoke-UVR.pth` | Instrumental / Vocals (KARAOKE) |
| HP 卡拉OK #2 | `6_HP-Karaoke-UVR.pth` | Instrumental / Vocals (KARAOKE) |
| HP2 分离 #1 | `7_HP2-UVR.pth` | Instrumental / Vocals |
| HP2 分离 #2 | `8_HP2-UVR.pth` | Instrumental / Vocals |
| HP2 分离 #3 | `9_HP2-UVR.pth` | Instrumental / Vocals |
| SP 2波段 #1 | `10_SP-UVR-2B-32000-1.pth` | Instrumental / Vocals |
| SP 2波段 #2 | `11_SP-UVR-2B-32000-2.pth` | Instrumental / Vocals |
| SP 3波段 | `12_SP-UVR-3B-44100.pth` | Instrumental / Vocals |
| SP 4波段 #1 | `13_SP-UVR-4B-44100-1.pth` | Instrumental / Vocals |
| SP 4波段 #2 | `14_SP-UVR-4B-44100-2.pth` | Instrumental / Vocals |
| SP 中频 #1 | `15_SP-UVR-MID-44100-1.pth` | Instrumental / Vocals |
| SP 中频 #2 | `16_SP-UVR-MID-44100-2.pth` | Instrumental / Vocals |
| MGM 主流 | `MGM_MAIN_v4.pth` | Instrumental / Vocals |
| MGM 高端 | `MGM_HIGHEND_v4.pth` | Instrumental / Vocals |
| MGM 低端 A | `MGM_LOWEND_A_v4.pth` | Instrumental / Vocals |
| MGM 低端 B | `MGM_LOWEND_B_v4.pth` | Instrumental / Vocals |
| 木管乐器分离 | `17_HP-Wind_Inst-UVR.pth` | No Woodwinds / Woodwinds |

---

## 查询工具

如果不知道用什么模型，可以通过 CLI 查询：

```powershell
# 列出所有分类和数量
python msst_cli.py --list-categories

# 搜索关键词（如 "reverb" "noise" "vocal" 等）
python msst_cli.py --search <关键词>

# 列出某个分类的所有模型（vocal_models / single_stem_models / multi_stem_models / VR_Models）
python msst_cli.py --list-models <分类名>
```

---

## 典型处理流程

### 流程1：分离人声并去混响

```
1. run_powershell: 用 melband_roformer_instvox_duality_v2.ckpt 分离人声+伴奏
   → 得到: <output>/Vocals.wav, <output>/Instrumental.wav

2. run_powershell: 用 dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt 处理伴奏
   input: <output>/Instrumental.wav
   → 得到: <output>/dry.wav (去混响后的干净伴奏)
```

### 流程2：降噪 → 分离人声

```
1. run_powershell: 用 denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt 降噪
   → 得到: <output>/dry.wav

2. run_powershell: 用 big_beta5e.ckpt 提取人声
   input: <output>/dry.wav
   → 得到: <output>/vocals.wav
```

### 流程3：制作卡拉OK伴奏带

```
1. run_powershell: 用 model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt
   → 得到: <output>/karaoke.wav (纯伴奏)
```

---

## 注意事项

1. **仅支持 WAV/FLAC/MP3/M4A 格式输入**，输出格式默认 WAV，可用 `--format mp3` 改为 MP3
2. **单次处理约 15-60 秒**，大模型（duality, big_beta5e）约 50-60 秒，小模型约 15-30 秒
3. **多步处理需要多次调用**，每次调用一个模型。输出目录不同可以区分结果
4. **GPU 自动检测** (--device auto)，有 CUDA 会自动用 GPU
5. **输入必须是绝对路径**，输出目录会自动创建
6. **VR 模型 (.pth)** 比 MSST 模型 (.ckpt/.th) 快但质量略低，适合对速度有要求的场景
7. 如果用户要求未知效果的分离，先用 `--search <关键词>` 查找模型，再看模型的中文说明匹配需求

---

## 完整参数说明

```
python msst_cli.py --model <模型名>        # 必需
                   --input <输入路径>       # 必需，绝对路径
                   -o <输出目录>            # 可选，默认 ./results
                   --format wav|flac|mp3    # 可选，默认 wav
                   --device auto|cpu|cuda   # 可选，默认 auto
                   --tta                   # 可选，启用后质量更高但3倍耗时
```

---

## 附录：完整模型清单

以下列出全部 78 个模型，含尺寸信息。上方映射表已覆盖常用场景，以下是完整备查。

### vocal_models (人声/伴奏分离, 19个)

| 模型名 | 尺寸 | 架构 | 输出轨 |
|--------|------|------|--------|
| `BS-Roformer_LargeV1.ckpt` | 706MB | bs_roformer | vocals / other |
| `Kim_MelBandRoformer.ckpt` | 871MB | mel_band_roformer | vocals / other |
| `big_beta5e.ckpt` | 1.4GB | mel_band_roformer | vocals / other |
| `bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt` | 503MB | bs_roformer | male / female |
| `inst_v1e.ckpt` | 871MB | mel_band_roformer | other / vocals |
| `kimmel_unwa_ft.ckpt` | 871MB | mel_band_roformer | vocals / other |
| `mel_band_roformer_instrumental_becruily.ckpt` | 871MB | mel_band_roformer | Instrumental / Vocals |
| `mel_band_roformer_vocals_becruily.ckpt` | 871MB | mel_band_roformer | vocals / other |
| `melband_roformer_inst_v2.ckpt` | 1.5GB | mel_band_roformer | other / vocals |
| `melband_roformer_instvox_duality_v2.ckpt` | 1.6GB | mel_band_roformer | Vocals / Instrumental |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 610MB | bs_roformer | vocals / other |
| `model_bs_roformer_ep_368_sdr_12.9628.ckpt` | 610MB | bs_roformer | vocals / instrumental |
| `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt` | 961MB | mel_band_roformer | vocals / other |
| `model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | 871MB | mel_band_roformer | karaoke / other |
| `model_swin_upernet_ep_56_sdr_10.6703.ckpt` | 896MB | swin_upernet | vocals / other |
| `model_vocals_htdemucs_sdr_8.78.ckpt` | 160MB | htdemucs | vocals / other |
| `model_vocals_mdx23c_sdr_10.17.ckpt` | 427MB | mdx23c | vocals / other |
| `model_vocals_mel_band_roformer_sdr_8.42.ckpt` | 129MB | mel_band_roformer | vocals / other |
| `model_vocals_segm_models_sdr_9.77.ckpt` | 824MB | segm_models | vocals / other |

### single_stem_models (单轨提取(去噪/去混响等), 21个)

| 模型名 | 尺寸 | 架构 | 输出轨 |
|--------|------|------|--------|
| `Apollo_LQ_MP3_restoration.ckpt` | 63MB | apollo | restored / addition |
| `HTDemucs4_FT_bass.th` | 80MB | htdemucs | drums / bass / other / vocals |
| `HTDemucs4_FT_drums.th` | 80MB | htdemucs | drums / bass / other / vocals |
| `HTDemucs4_FT_other.th` | 80MB | htdemucs | drums / bass / other / vocals |
| `HTDemucs4_FT_vocals_official.th` | 80MB | htdemucs | drums / bass / other / vocals |
| `apollo_model_uni.ckpt` | 140MB | apollo | restored / addition |
| `aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt` | 797MB | mel_band_roformer | aspiration / other |
| `aspiration_mel_band_roformer_sdr_18.9845.ckpt` | 797MB | mel_band_roformer | aspiration / other |
| `de_big_reverb_mbr_ep_362.ckpt` | 435MB | mel_band_roformer | dry / other |
| `denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt` | 871MB | mel_band_roformer | dry / other |
| `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` | 871MB | mel_band_roformer | dry / other |
| `dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt` | 435MB | mel_band_roformer | dry / other |
| `dereverb_mdx23c_sdr_6.9096.ckpt` | 427MB | mdx23c | dry / other |
| `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt` | 871MB | mel_band_roformer | noreverb / reverb |
| `dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt` | 871MB | mel_band_roformer | noreverb / reverb |
| `deverb_bs_roformer_8_256dim_8depth.ckpt` | 163MB | bs_roformer | noreverb / reverb |
| `deverb_bs_roformer_8_384dim_10depth.ckpt` | 345MB | bs_roformer | noreverb / reverb |
| `deverb_mel_band_roformer_ep_27_sdr_10.4567.ckpt` | 467MB | mel_band_roformer | noreverb / reverb |
| `mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt` | 871MB | mel_band_roformer | crowd / other |
| `model_bs_roformer_ep_937_sdr_10.5309.ckpt` | 375MB | bs_roformer | vocals / other |
| `model_mdx23c_ep_271_l1_freq_72.2383.ckpt` | 417MB | mdx23c | similarity / difference |

### multi_stem_models (多轨分离, 9个)

| 模型名 | 尺寸 | 架构 | 输出轨 |
|--------|------|------|--------|
| `HTDemucs4.th` | 80MB | htdemucs | drums / bass / other / vocals |
| `HTDemucs4_6stems.th` | 52MB | htdemucs | drums / bass / other / vocals / guitar / piano |
| `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt` | 417MB | mdx23c | kick / snare / toms / hh / ride / crash |
| `bs_roformer_4stems_ft.ckpt` | 503MB | bs_roformer | drums / bass / other / vocals |
| `model_bandit_plus_dnr_sdr_11.47.chpt` | 142MB | bandit | speech / music / effects |
| `model_drumsep.th` | 160MB | htdemucs | kick / snare / cymbals / toms |
| `model_mdx23c_ep_168_sdr_7.0207.ckpt` | 427MB | mdx23c | vocals / bass / drums / other |
| `model_scnet_sdr_9.3244.ckpt` | 161MB | scnet | drums / bass / other / vocals |
| `scnet_checkpoint_musdb18.ckpt` | 40MB | scnet | drums / bass / other / vocals |

### VR_Models (UVR传统模型, 29个)

| 模型名 | 尺寸 | 输出轨 |
|--------|------|--------|
| `10_SP-UVR-2B-32000-1.pth` | 30MB | Instrumental / Vocals |
| `11_SP-UVR-2B-32000-2.pth` | 30MB | Instrumental / Vocals |
| `12_SP-UVR-3B-44100.pth` | 30MB | Instrumental / Vocals |
| `13_SP-UVR-4B-44100-1.pth` | 30MB | Instrumental / Vocals |
| `14_SP-UVR-4B-44100-2.pth` | 30MB | Instrumental / Vocals |
| `15_SP-UVR-MID-44100-1.pth` | 30MB | Instrumental / Vocals |
| `16_SP-UVR-MID-44100-2.pth` | 30MB | Instrumental / Vocals |
| `17_HP-Wind_Inst-UVR.pth` | 213MB | No Woodwinds / Woodwinds |
| `1_HP-UVR.pth` | 121MB | Instrumental / Vocals |
| `2_HP-UVR.pth` | 121MB | Instrumental / Vocals |
| `3_HP-Vocal-UVR.pth` | 121MB | Vocals / Instrumental |
| `4_HP-Vocal-UVR.pth` | 121MB | Vocals / Instrumental |
| `5_HP-Karaoke-UVR.pth` | 121MB | Instrumental / Vocals (KARAOKE) |
| `6_HP-Karaoke-UVR.pth` | 121MB | Instrumental / Vocals (KARAOKE) |
| `7_HP2-UVR.pth` | 525MB | Instrumental / Vocals |
| `8_HP2-UVR.pth` | 525MB | Instrumental / Vocals |
| `9_HP2-UVR.pth` | 525MB | Instrumental / Vocals |
| `Harmonic_Noise_Separation_yxlllc.pth` | 57MB | No Aspiration / Aspiration |
| `MGM_HIGHEND_v4.pth` | 30MB | Instrumental / Vocals |
| `MGM_LOWEND_A_v4.pth` | 30MB | Instrumental / Vocals |
| `MGM_LOWEND_B_v4.pth` | 30MB | Instrumental / Vocals |
| `MGM_MAIN_v4.pth` | 30MB | Instrumental / Vocals |
| `UVR-BVE-4B_SN-44100-1.pth` | 213MB | Vocals / Instrumental (BV) |
| `UVR-De-Echo-Aggressive.pth` | 121MB | No Echo / Echo |
| `UVR-De-Echo-Normal.pth` | 121MB | No Echo / Echo |
| `UVR-DeEcho-DeReverb.pth` | 213MB | No Reverb / Reverb |
| `UVR-DeNoise-Lite.pth` | 17MB | Noise / No Noise |
| `UVR-DeNoise.pth` | 121MB | Noise / No Noise |
| `UVR-DeReverb-aufr33-jarredou_4band_v4_ms_fullband.pth` | 56MB | Dry / Reverb |
