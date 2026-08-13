# H 分支

本分支验证细粒度歌词时间条件：先用统一 kana 建立摩拉分组，再由 SOFA 输出
摩拉内 phone interval，最后使用现有日语 IPA tokenizer 生成训练 token 并按 phone 时间放置。

## 边界

- `H` 表示 mora-anchored phoneme-level alignment，不是书写字符级硬切分。
- SOFA 与训练 tokenizer 必须读取同一份规范化 kana；不再混用 kana 对齐与原始 text token 化。
- 摩拉只负责稳定分组，最终时间信息来自 SOFA phone interval。
- 保留现有 token ID、embedding、A/B 结构和 DiT 主体；首轮只改变 text placement。
- 推理侧同样使用 Whisper、kana 规范化和 SOFA；改词时的新旧摩拉映射必须单独冻结并验收。
- 音素帧碰撞、pause/标点和异常 G2P 样本未通过门禁前，不启动正式训练。

## 当前状态

- 分支定义与首个实验工人已建立。
- 全量静态前端审计已完成；统一 kana 后两端音素序列完全一致率为 96.62%。
- target singer 40 条长段已按五档字符密度完成 SOFA 假名/phone 对齐，40/40 结构通过。
- 同步试听网页已生成；每个假名及罗马音随音频高亮，支持单元跳转。
- JPN_Test2_Plus 对促音 `cl` 存在明确限制：88 个促音中 53 个独立解码，35 个需与后续假名合并近似。
- 用户于 2026-07-28 确认同步试听无问题，target singer 40 条 Stage 0 感知门禁通过。
- H-v1 全量 private corpus count train + 201 test 已完成独立 SOFA phone manifest、原子 shard、恢复测试、
  动态 A/B 审计与紧凑训练 manifest；全量句覆盖 82.51%，三点 runtime 覆盖 77.10%。
- Control/H 样本、Path、Phrases、token 序列和动态句首不变量全部通过。
- H-Control/H 共同训练入口已完成；exact resume 保存每 rank RNG 与数据游标。服务器代码哈希、
  Bash/静态检查、11 项单测、sampler offset 与全量紧凑 manifest CPU 门禁已通过。
- 单卡 phone shape/token/forward/backward/完整 checkpoint 门禁已通过。
- 严格确定性 4 卡 continuous 10 与 5+resume5 的 model、EMA、optimizer、scheduler、rank RNG/
  游标已全 section bit-exact，三个 torchrun 均退出 0。
- 全部训练前门禁已通过；H-Control/H 2k 均从同一 Official base 独立完成，两个 final checkpoint
  审计通过。H 在 1k/1.5k/2k 的固定 Eval FlowB 连续小幅低 0.0045-0.0052，但 FlowB 不能判断
  摩拉是否落在原节拍槽，只作为训练健康指标，H 的效果仍未知。
- H-v1 下一阶段冻结为对生成前源 B、H-Control 与 H 输出重跑同一 Whisper+SOFA，比较 ASR 内容、
  句内相对 mora onset、相邻间隔和错槽率，并做三轨网页与人工听评；H-v1 30k 仍未启动。该门禁
  不阻塞用户随后独立授权的 H-PUL/V4H 路线。
- H-PUL/H-v2 概念讨论已收敛并命名 V4H：真实 fallback 使用逐帧 `<PUL>=366` 自动分配到下一句前，
  `<SEP>=365` 固定在下一句运行时首帧的前一帧；不人工降级正常 H 句，不安排额外 H-END 消融。
- V4H 独立 renderer、训练入口、checkpoint schema、15 项单测及 33,696 场景全量确定性 CPU 门禁
  已通过。四卡完整数据 10-step smoke 已正常退出并通过 checkpoint 审计；正式 30k 已从 Official
  base 独立完成，final checkpoint 全 section 审计通过，30k Eval FlowB/CKA 为 0.8662/0.1354。
  step 24k/30k 已经cloud storage同步到本机 `${LOCAL_PROJECT_PATH}` 与 `V4H_30k`，
  并分别通过服务器/本机 SHA256: redacted
  错槽指标与人工试听。
- V4H 24k 本地同构推理已完成：权威配置与 H runtime 哈希通过，27 组跨源 A/B 的 SOFA/H
  candidates、CFG 3/1 共 54 条结果及逐组 placement 审计全部通过完整性门禁。
- V4H 30k 已使用相同冻结输入、SOFA/H candidates、官方 VAE、MIDI teacher、32 steps 与 seed 42
  完成本地 CFG 3/1 共 54 条推理；逐组 placement 与 24k 条件一致，音频完整性门禁通过。
- 用户纯听感确认 V4H 24k 的咬字时间优于其他已听分支，基本解决目标错位问题；音色弱于带 `g`
  且额外训练 10k 的 V4fg 10k，极高音炸、混音输入炸和跑调仍未解决。V4H 暂定位为切实、稳健的
  小改动；下一步复核 30k 并补 Whisper+SOFA 错槽指标。
- 用户进一步确认 V4H 30k 的音色比 V4H 24k 更好、更细腻；30k 是否完整保持 24k 的咬字时间
  收益，以及相对 V4fg 10k 的最终位置，仍按独立听感与错槽评测记录。
- 用户已授权 V4Hg 10k：从 V4H 30k EMA warm-start，保留 H/PUL placement，唯一域变化为切换
  frozen 285k online VAE，并用历史 V4fg 的 `5e-6 / warmup 250 / hold 6000 / 10k` 适配配方。
  本地/服务器静态门禁、单卡 285k VAE forward、四卡 10-step smoke、checkpoint 全 section 审计
  和实际 step 10→11 resume 均通过。正式 10k 已从原 V4H 30k 独立完成，exit 0，final 全 section
  审计通过；10k Eval Loss/FlowB 为 2.3034/0.9505，全程固定 Eval 无反弹。下一步使用 285k VAE
  比较 V4H 30k、V4Hg 10k 与 V4fg 10k 的听感和错槽。V4Hg 10k final 已经cloud storage同步至本机
  `${LOCAL_PROJECT_PATH}`，服务器/本机 SHA256: redacted
- V4Hg 10k 已使用冻结的 V4H 27 组 SOFA/H alignment、285k online VAE、32 steps 与 seed 42
  完成本地 CFG 3/1 共 54 条评分集；placement 与 V4H 30k 逐组一致，音频与 provenance 门禁通过。
- V4Hg 10k CFG 1 的既有 27 条输出已追加 SVC cosine3000 工程方法：不重算 SVS，每组使用原始
  A 段作为 prompt 与 CAM++ 音色参考；27 条输出及逐组 source/A/model/config 哈希审计全部通过。
- V4H 24k/30k 与 V4Hg 10k 已接入桌面网页 SVS 面板：用户 TextObject 句界驱动逐句
  SOFA 裁窗，逸散范围 0s 到 2s；跨句 phone 错位硬失败，其余 mora/phone frontend、token、
  K=4、PUL/SEP renderer 与模型构图继续锁定训练权威文件。V4H 24k/30k 绑定 Official
  VAE，V4Hg 10k 绑定 285k online VAE，三者均只允许对应 EMA inference checkpoint。

## 工人

| 文件 | 内容 |
|---|---|
| `摩拉音素级对齐实验.md` | H 的数据契约、已有证据、推理同构、执行阶段和停止条件 |

## 阅读顺序

先读 `摩拉音素级对齐实验.md`。实验参数、产物位置和阶段结论持续记录在该工人文件中。

















