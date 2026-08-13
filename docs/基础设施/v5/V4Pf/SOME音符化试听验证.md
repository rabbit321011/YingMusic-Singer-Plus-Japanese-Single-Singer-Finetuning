# SOME 音符化试听验证

## 目标

用训练时同款 SOME→note→0.5 半音 P 通路生成可直接试听的正弦旋律，验证音高、音符边界、
时值和 REST 是否足以作为 V4Pf 条件。该门禁先于 P embedding 初始化和 P-only 训练。

## 测试集

非target singer域使用用户提供目录中的 42 条 MSST 人声，全部纳入：

```text
${LOCAL_EXPORT_PATH}
```

target singer域从正式 SOFA V4d control train manifest 的 private corpus count 条记录中建立 30 条候选池，候选
按时长均匀覆盖且来源去重。SOME 推理后按时长、音高范围、REST、音符密度、正常样本和
可疑边界样本共同选出最终 20 条：

| 时长桶 | 用户要求 | 最终数量 |
|---|---:|---:|
| 0–10s | 5 | 5 |
| 10–20s | 8 | 8 |
| 20–30s | 7 | 7 |

最终试听集共 62 条。可疑结果没有被清除；例如 MIDI 28、37、89 等极值被保留，供人耳判断
是原音频内容还是 SOME 错判。

## 解码与渲染

每条音频先转为 mono、44.1kHz，然后执行：

```text
audio
  -> MelodySpectrogram
  -> frozen SOME MIDI teacher
  -> pitch logits + boundary logits
  -> sigmoid / voiced-REST / note boundary / note duration decode
  -> 0.5-semitone class, REST=255
```

SOME checkpoint 的 Mel 前端必须与 OpenVPI/SOME 官方配置一致：80 bins、`fmin=40Hz`、
`fmax=8000Hz`。首轮 v1/v2 错误沿用了 YingMusic `MelodySpectrogram` 的默认
`0-22050Hz` 范围；虽然 Mel bin 数量仍为 80，但每个 bin 的频率坐标已改变，导致冻结的
SOME checkpoint 将绝对 MIDI 系统性解码偏低。该错误不能用固定升调补偿，必须从 Mel
输入端修正。

每条样本输出三份同长度 44.1kHz WAV：

| 文件 | 含义 |
|---|---|
| `original_mono.wav` | 原音频转 mono 的试听基准 |
| `p_native_sine.wav` | SOME 原生约 86Hz 时间轴的 P 正弦 |
| `p_model_sine.wav` | 重采样到 DiT/VAE 约 21.53Hz 时间轴的 P 正弦 |
| `p_native_piano.ogg` | 由 native note CSV 本机渲染的程序化钢琴音色 |
| `p_model_piano.ogg` | 由 model note CSV 本机渲染的程序化钢琴音色 |

正弦频率严格由 `midi=class/2` 换算；REST 输出静音。每个 SOME note ID 的起止处使用短
attack/release，同音重复即使 class 不变也会重新起音。因此该试听同时覆盖 pitch、REST、
duration 和 boundary，不把同音重复静默合并成长音。

同时输出 `p_native_notes.csv`、`p_model_notes.csv` 和逐条 `metadata.json`，用于定位具体
时间、class、MIDI、REST 与 note ID。

试听页默认播放钢琴轨，便于判断旋律和音符边界；正弦轨仍保留为不受乐器谐波影响的技术
基准。钢琴仅依据既有 note CSV 本机合成，没有重新运行 SOME，也没有改变 P 类别、REST、
note ID 或两个时间轴。为避免重复制造大体积 WAV，钢琴轨使用 44.1kHz OGG/Vorbis。

## 产物

本机试听入口：

```text
${LOCAL_PROJECT_ROOT}\TEMP\v4pf_some_audition_listen_20260728_v3\index.html
```

该静态索引提供原声、native P、model P 并排播放、分组/时长过滤、结论标记和 JSON 导出。
浏览器标记保存在本机 localStorage；导出的正式人耳结果应另行归档。

压缩包：

```text
${LOCAL_PROJECT_ROOT}\TEMP\v4pf_some_audition_listen_20260728_v3.tar.gz
SHA256: redacted
```

首轮 v1 正弦在模型时间轴中只按 note ID 分段，会忽略同一 note ID 内由线性重采样产生的短暂
class 变化；SOME 解码统计未受影响，但试听不完全忠实。v2 已修复为 note ID 或 class 任一变化
即重新分段，但 v1/v2 均使用了错误的 Mel 频率范围，绝对音高结论无效。v1/v2 包不得用于
人耳结论；v3 同时包含边界渲染修复和官方 `40-8000Hz` Mel 前端，是唯一有效试听包。

## 工程审计

- v3 的 72/72 候选完成真实 SOME 推理，产物记录 Mel `fmin=40`、`fmax=8000`；
- 最终包为 42 条非target singer + 20 条target singer，target singer严格满足 `5/8/7`；
- 72 组输出的 original/native/model WAV 均存在、44.1kHz、长度一致、全部有限；
- native/model 正弦均非静音；
- PAD 总数为 0；
- 最终本机索引含 62 行、186 个 WAV 引用，缺失引用为 0；
- 合成边界检查确认 REST 不产生假 class 0，同音重复保留不同 note ID，降采样后仍可重触发。
- 以 16 组随机长度/logits/boundary 对照既有 V4Pvf 解码器，V4Pf 模型时间轴 class 逐项完全一致。
- 本机 v3 解包后再次审计 62 行、186 个 WAV 引用，缺失引用 0，全部 WAV 格式、长度、有限值
  和非静音检查通过；压缩包 SHA-256 与生成记录一致。
- 独立 `librosa.pyin` 复核 v3：N001 原声/P 中位 MIDI `63.8/64.0`，N009 `79.0/79.0`，
  H009 `65.3/65.5`。三条偏差为 `0.2/0.0/0.2` 半音，旧包的系统性低音已消失。
- 本机新增 124 个 native/model 钢琴 OGG，总计 12.29MiB；全部 44.1kHz、与对应原音频
  逐样本等长、有限且非静音。N001/N009/H009 钢琴轨独立 pYIN 中位 MIDI 为
  `64.0/79.0/65.5`，与 P 数据一致。

独占实现：

```text
src/YingMusicSinger/melody/midi_p_v4pf.py
prepare_some_p_audition_v4pf.py
render_some_p_audition_v4pf.py
build_some_p_audition_index_v4pf.py
audit_some_p_audition_v4pf.py
render_p_timbre_audition_v4pf.py
```

既有 V4f、V4Pvf 和公共源码未修改。

## 人耳裁决

当前状态：**已完成首轮用户试听，未通过为正式 P teacher。**

用户听感结论是整体旋律大体正确，但会出现“正常、正常、突然崩坏、再恢复”的局部错误，
表现为几个音符相对于原唱正确音高突然拉低，听起来像口胡乱唱。该错误不能由全局中位音高
审计发现，也不能仅凭相邻音符连续性判断。target singer来源明显少于复杂非target singer MSST stem，
输入中的伴奏/和声/混响残留或分离伪影是合理假设，但尚未完成逐时间点因果验证。

因此 SOME 保留为 A/B 基线，不进入 P-only 训练。后继 GAME 的同源试听门禁详见
`GAME试听验证.md`。

每条先比较 original 与 native，再比较 native 与 model：

- native 正确、model 错误：优先归因于约 21.53Hz 重采样；
- native 与 model 都错误：优先检查 SOME 或 note/boundary 解码；
- 音高正确但同音重复、起止或 REST 错误：归入 boundary/duration 问题；
- 非target singer与target singer应分别汇总，不能用一个域的结果替代另一个域。

只有主要旋律、REST 和边界达到用户可接受水平，且错误不是系统性的，才允许进入 P 初始化
对照和 P-only 训练。该试听不能验证 P embedding 或 DiT，也不能单独证明最终 V4Pf 质量。

















