# YingMusic-Singer-Plus: Controllable Singing Voice Synthesis with Flexible Lyric Manipulation and Annotation-free Melody Guidance

**Authors:** Chunbo Hao $^{1,2}$, Junjie Zheng $^{2}$, Guobin Ma $^{1}$, Yuepeng Jiang$^{1}$, Huakang Chen$^{1}$, Wenjie Tian$^{1}$, Gongyu Chen $^{2}$, Zihao Chen $^{2}$, Lei Xie$^{1,*}$

**Affiliations:**
1. Audio, Speech and Language Processing Group (ASLP@NPU), School of Computer Science, Northwestern Polytechnical University, China
2. AI Lab, GiantNetwork, China

**Contact:** research-contact@example.org, research-contact@example.org, research-contact@example.org

**arXiv:** 2603.24589v2 [eess.AS] 9 Apr 2026

**Code & Demos:** https://github.com/ASLP-lab/YingMusic-Singer-Plus

**Index Terms:** singing voice synthesis, lyric editing, reinforcement learning, diffusion model

---

## Abstract

Regenerating singing voices with altered lyrics while preserving melody consistency remains challenging, as existing methods either offer limited controllability or require laborious manual alignment. We propose **YingMusic-Singer-Plus**, a fully diffusion-based model enabling melody-controllable singing voice synthesis with flexible lyric manipulation. The model takes three inputs: an optional timbre reference, a melody-providing singing clip, and modified lyrics, without manual alignment. Trained with curriculum learning and Group Relative Policy Optimization (GRPO), YingMusic-Singer-Plus achieves stronger melody preservation and lyric adherence than Vevo2, the most comparable baseline supporting melody control without manual alignment. We also introduce **LyricEditBench**, the first benchmark for melody-preserving lyric modification evaluation. The code, weights, benchmark, and demos are publicly available.

---

## 1. Introduction

Singing Voice Synthesis (SVS) aims to generate human-like singing voices from musical scores, lyrics, and timbre references. Modern systems [1–6] achieve high-fidelity synthesis, yet most rely on precisely annotated paired data associating each phoneme with an exact pitch contour and duration. While such fine-grained control is indispensable for professional music production, preparing these annotations creates a prohibitive barrier for an increasingly important use case: **singing voice editing**, which regenerates an existing singing voice with modified lyrics while preserving the original melodic and rhythmic structure. This capability is highly desirable for song adaptation, personalized cover generation, rapid prototyping of vocal arrangements, and multilingual song localization.

Existing editing paradigms fall into two categories:

- **In-context learning**: Masks the region to be edited and regenerates it conditioned on the surrounding context and target lyrics [7, 8]. Although convenient, this approach is restricted to local segments and provides limited melody control.
- **Manual alignment (commercial SVS)**: Relies on tools such as Synthesizer V and ACE Studio, where users manually align modified lyrics with MIDI notes and durations before re-synthesizing. This pipeline offers strong controllability but demands manual effort, and the complexity increases for tasks such as cross-lingual translation.

Several recent efforts address these challenges. Vevo2 [9] achieves melody-controllable generation, yet suffers from reduced intelligibility and poor melody adherence. SoulX-Singer [10] supports existing singing as melody input but still requires manual alignment of word-level timestamps, leaving the core burden unresolved.

To address these challenges, we propose **YingMusic-Singer-Plus**, a fully diffusion-based SVS model:

1. **Streamlined editing paradigm**: Synthesizes singing voices from only three inputs — an optional timbre reference clip, a singing clip providing the target melody, and corresponding modified lyrics — without manual alignment or precise annotation.
2. **Curriculum training + GRPO**: To mitigate limited phoneme generalization from small-scale singing data and address the trade-off between faithful lyric reproduction and melody adherence, we employ a curriculum training strategy combined with Group Relative Policy Optimization (GRPO) [11], enabling strong performance on both dimensions simultaneously.
3. **LyricEditBench**: Constructed from GTSinger [12], the first benchmark for lyric modification evaluation under matched melody conditions, covering six common editing scenarios with balanced sampling.

Experiments show that YingMusic-Singer-Plus outperforms Vevo2 [9] in both melody preservation and lyric adherence.

---

## 2. Methodology

### 2.1. Architecture Overview

YingMusic-Singer-Plus generates singing voices at 44.1 kHz from three inputs: an optional timbre reference, a melody-providing singing clip, and corresponding modified lyrics. It comprises:

1. **Variational Autoencoder (VAE)**: Following Stable Audio 2 [13], whose encoder $E$ downsamples a stereo 44.1 kHz singing waveform $\mathbf{x} \in \mathbb{R}^{T \times 2}$ by a factor of 2048 into $\mathbf{z} = E(\mathbf{x}) \in \mathbb{R}^{T' \times D}$, and whose decoder $D$ reconstructs high-fidelity audio $\hat{\mathbf{x}} = D(\mathbf{z})$ at inference.

2. **Melody Extractor**: Built upon the encoder of a pre-trained MIDI extraction model, whose intermediate representations naturally capture disentangled melody information. It produces $\mathbf{h} = M(\mathbf{M}) \in \mathbb{R}^{L \times D_m}$, which is then temporally interpolated to $\tilde{\mathbf{h}} \in \mathbb{R}^{T' \times D_m}$ to match the VAE latent frame rate.

3. **IPA Tokenizer**: Converts both Chinese and English lyrics into a unified discrete phoneme sequence. To ensure the model correctly distinguishes between prompt and generation regions, avoiding phoneme omission or repetition at boundaries, we adopt sentence-level alignment following DiffRhythm [14]. Each lyric sentence is converted into an IPA subsequence and placed at its corresponding onset frame within a padded frame-level sequence of length $T'$. The aligned sequence is passed through a learnable embedding layer to yield $\mathbf{e} \in \mathbb{R}^{T' \times D_e}$. During inference, prompt lyrics are placed at the beginning of the sequence and target lyrics at the start of the masked region, requiring no timestamp annotation from the user.

4. **DiT-based CFM Backbone**: Following F5-TTS [15].

During training, a proportion $\gamma$ of VAE latent frames is randomly masked as the synthesis target, with the unmasked portion serving as timbre context. Let $\mathbf{z}_{\text{ctx}}$ denote the unmasked VAE latent (zero-filled in masked regions). The condition $\mathbf{c} = [\tilde{\mathbf{h}}; \mathbf{e}; \mathbf{z}_{\text{ctx}}]$ is concatenated with the noisy latent $\mathbf{z}_t = (1-t)\mathbf{z}_0 + t\mathbf{z}_1$ along the channel dimension and fed into the CFM, which learns a velocity field via:

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{t,\mathbf{z}_0,\mathbf{z}_1,\mathbf{c}} \left\| \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c}) - (\mathbf{z}_1 - \mathbf{z}_0) \right\|^2 \tag{1}$$

### 2.2. Curriculum Training

To mitigate limited phoneme generalization caused by the small scale and challenging vocal techniques in singing data, YingMusic-Singer-Plus first undergoes **TTS Pretraining** without melody conditioning.

The subsequent **Singing Voice Supervised Fine-Tuning (SFT)** stage has two phases:

- **Phase 1**: Enables sentence-level alignment on singing data, allowing the model to adapt to the singing domain.
- **Phase 2**: Activates melody conditioning and introduces a **Centered Kernel Alignment (CKA)** loss to enforce melody adherence.

Given the predicted $\mathbf{v}_\theta$ and melody $\tilde{\mathbf{h}}$, CKA measures their alignment via Gram matrices $\mathbf{K} = \mathbf{v}_\theta \mathbf{v}_\theta^\top$ and $\mathbf{L} = \tilde{\mathbf{h}} \tilde{\mathbf{h}}^\top$:

$$\mathcal{L}_{\text{CKA}} = 1 - \frac{\|\mathbf{K}^\top \mathbf{L}\|_F^2}{\|\mathbf{K}^\top \mathbf{K}\|_F \|\mathbf{L}^\top \mathbf{L}\|_F} \tag{2}$$

The total Phase 2 loss is:

$$\mathcal{L}_{\text{SFT}} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{CKA}}$$

### 2.3. Group Relative Policy Optimization

While curriculum training achieves high performance, SFT Phase 2 simultaneously degrades PER, exposing a persistent trade-off. $\mathcal{L}_{\text{MSE}}$ targets holistic latent reconstruction and cannot isolate specific deficiencies for targeted optimization, while adjusting $\lambda$ in $\mathcal{L}_{\text{CKA}}$ only shifts the balance without resolving the trade-off itself. Moreover, inevitable noise in large-scale singing data (backing vocals, quality artifacts) caps model performance at the dataset ceiling.

To overcome these limitations, **reinforcement learning (RL)** becomes essential. PPO requires a value network that is expensive to train. Offline methods such as DPO suffer from distribution shift, as pre-collected preference data becomes stale when the policy improves. **GRPO** [11] operates online and estimates baselines from within-group reward statistics, eliminating the value network while remaining efficient and stable.

Following recent efforts [16–18], we convert the deterministic ODE trajectory into an SDE but restrict stochastic steps to a bounded window, ensuring precise advantage attribution to exploratory steps. To prevent collapse toward a single reward dimension, we employ $M$ reward models jointly and compute the advantage for each sample as:

$$A_i = \sum_{k=1}^{M} w_k \cdot \frac{R_i^k - \text{mean}(\{R_j^k\})}{\text{std}(\{R_j^k\})} \tag{3}$$

The final GRPO loss is:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|S|} \sum_{t \in S} \left( -\mathcal{L}_{\text{clip}} + \beta D_{\text{KL}} \right) \tag{4}$$

where $\mathcal{L}_{\text{clip}}$ is the clipped surrogate objective over the current-to-old policy likelihood ratio, $D_{\text{KL}}$ regularizes the current policy toward the reference, $G$ is the group size, $S$ the set of SDE sampling steps, and $\beta$ the KL regularization strength.

### 2.4. LyricEditBench

We build LyricEditBench from GTSinger [12] by removing all Paired Speech Group content, deduplicating audio via MD5 hashing, and excluding clips exceeding 15 seconds. DeepSeek V3.2 [19] then generates modified lyrics for each of six modification types.

**Table 1: Task types for lyric modification in LyricEditBench.**

| Abbr. | Task Type | Description |
|-------|-----------|-------------|
| PSub  | Partial Substitution | Substitute part of the words |
| FSub  | Full Substitution | Completely rewrite the song |
| Del   | Deletion | Delete some words |
| Ins   | Insertion | Insert some words |
| Trans | Translation | CN ↔ EN translation |
| Mix   | Code-Mixing | Mixed-language lyrics |

Given original lyrics and modification instructions, the LLM produces revised versions, with non-compliant outputs discarded, yielding **11,535 valid samples**. Samples are classified by singer gender (male/female) and language (Chinese/English) into four categories, then organized by modification type. For each combination, we select 30 samples per singing technique (covering the six techniques in GTSinger) and 120 for the technique-free category, resulting in **300 per modification type per category** and **7,200 test instances** in total. For each instance, a timbre prompt of no more than 15 seconds is randomly drawn from the remaining audio pool.

---

## 3. Experimental Setup

**Dataset.** The Chinese and English subsets of Emilia [20] are used for TTS pretraining. For Singing Voice SFT, internally licensed music tracks are processed by SongFormer [21] to segment structural boundaries and label function categories, discarding non-vocal segments. Vocal stems are then isolated using Mel-band RoFormer [22]. We retain clips between 2 and 30 seconds, splitting longer ones at sentence boundaries, ultimately obtaining **33,562.6 hours** of singing data.

The GRPO dataset is constructed by filtering SFT data with three criteria:
1. ASR transcript verification, retaining only clips with word error rate below 5%
2. Speaker diarization via pyannote [23, 24], keeping only single-speaker clips
3. DNSMOS P808 quality score [25, 26] threshold of 3.5

This yields approximately **20,240 curated clips** with balanced Chinese and English content. The test set in LyricEditBench is strictly excluded from training.

**Evaluation Metrics.** We evaluate models with four objective and two subjective metrics:

| Metric | Description |
|--------|-------------|
| **PER** (↓) | Phoneme Error Rate — measures phoneme-level intelligibility. Both Chinese and English clips are transcribed by singing-trained Qwen3-ASR-1.7B [27] and converted to phoneme sequences with tone markers removed. |
| **SIM** (↑) | Speaker Similarity — following F5-TTS [15], extracting speaker embeddings with a WavLM-large-based verification model and computing cosine similarity. |
| **F0-CORR** (↑) | F0 Pearson Correlation — measures melody adherence via frame-wise Pearson correlation between F0 contours of generated and reference clips using RMVPE [28]. |
| **VS** (↑) | Vocal Score — adopts VocalVerse2 [29] as a learned metric aligned with human perceptual preferences. |
| **N-MOS** (↑) | Naturalness Mean Opinion Score — 120 samples uniformly sampled across task types and languages, rated by 30 listeners on a 5-point scale for overall perceptual quality and naturalness. |
| **M-MOS** (↑) | Melody Mean Opinion Score — faithfulness to the reference melody, also on a 5-point scale. |

**Implementation Details.** We adopt the VAE from Stable Audio 2 [13] ($D = 64$), the encoder of SOME as Melody Extractor ($D_m = 128$, temporal dropout 0.1), and a DiT backbone following F5-TTS [15] (22 layers, 16 heads, hidden dim 1024, $D_e = 512$).

The full system has **~727.3M parameters** (453.6M CFM, 156.1M VAE, 117.6M Melody Extractor), trained on **8× A800 80GB GPUs** with DDP and bf16. Across all stages, 70%–100% of latent frames are randomly masked.

| Stage | Steps | Batch Duration | Learning Rate | Notes |
|-------|-------|----------------|---------------|-------|
| TTS Pretrain | 1M | 1.268 h | 1e-4 | No melody conditioning |
| SFT Phase 1 | 240K | 1.69 h | 2.5e-5 | Melody conditioning disabled |
| SFT Phase 2 | 170K | 1.69 h | 2.5e-5 | Melody conditioning enabled; $\lambda$ decayed from 0.3 to 0.01 over first 2K steps |
| GRPO | 1.2K | batch size 6 | 7e-6 | $G = 8$, $M = 4$ equally weighted reward models; SDE noise $a = 0.8$, window $W$ with $w_{\min}=1$, $w_s=8$, $\epsilon_u=0.01$, $\epsilon_l=0.002$, $\beta = 1$; no CFG |

Inference uses **32 ODE steps** with CFG scale 3.

---

## 4. Experimental Results

### 4.1. Main Results

We compare against Vevo2 [9], a token-based autoregressive model with disentangled timbre and melody control, where the timbre and melody references share the same clip for singing voice editing, or use separate clips for melody control. Vevo2 is the most direct baseline, as other systems operate under fundamentally different paradigms: in-context learning approaches require manually aligned edit boundaries and are restricted to local segments, while SoulX-Singer relies on precise character-level timestamps that are either impractical to obtain or would grant an unrealistic advantage if sourced from curated datasets.

**Table 2: Comparison with Baseline Model on LyricEditBench across Task Types and Languages.** Metrics: P = PER (↓), S = SIM (↑), F = F0-CORR (↑), V = VS (↑). Best results are **Bold**.

| Task | Model | M | PSub | FSub | Del | Ins | Trans | Mix | PSub | FSub | Del | Ins | Trans | Mix |
|------|-------|---|------|------|-----|-----|-------|-----|------|------|-----|-----|-------|-----|
| | | | *ZH* | | | | | | *EN* | | | | | |
| **Melody Control** | Vevo2 | P↓ | 0.1378 | 0.1462 | 0.1545 | 0.1872 | 0.4409 | 0.4757 | 0.3352 | 0.3481 | 0.3812 | 0.3135 | 0.8019 | 0.5132 |
| | | S↑ | 0.6462 | 0.6550 | 0.6457 | 0.6551 | 0.6115 | 0.6490 | 0.6357 | 0.6161 | 0.6277 | 0.6359 | 0.6237 | 0.6325 |
| | | F↑ | 0.8471 | 0.8188 | 0.8345 | 0.8552 | 0.7678 | 0.8526 | 0.8794 | 0.8409 | 0.8924 | 0.8888 | 0.8776 | 0.8927 |
| | | V↑ | 1.3578 | 1.3784 | 1.3491 | 1.1346 | 1.3061 | 1.4208 | 1.0340 | 1.1217 | 1.0476 | 0.9610 | 1.0281 | 1.0925 |
| | **Ours** | P↓ | **0.0192** | **0.0197** | **0.0458** | **0.0208** | **0.0881** | **0.1563** | **0.0685** | **0.0692** | **0.1053** | **0.0716** | **0.0413** | **0.2668** |
| | | S↑ | 0.6543 | 0.6561 | 0.6489 | 0.6552 | 0.5791 | 0.6395 | 0.6078 | 0.5914 | 0.6001 | 0.5889 | 0.5982 | 0.6076 |
| | | F↑ | **0.9364** | **0.9428** | **0.9351** | **0.9381** | **0.9378** | **0.9352** | **0.9355** | **0.9279** | **0.9309** | **0.9315** | **0.9290** | **0.9389** |
| | | V↑ | **2.0779** | **2.1419** | **2.1219** | **1.9887** | **1.9372** | **2.1002** | **1.5054** | **1.5418** | **1.6081** | **1.4036** | **1.5769** | **1.5060** |
| **Sing Edit** | Vevo2 | P↓ | 0.1290 | 0.1303 | 0.1596 | 0.1810 | 0.4111 | 0.4659 | 0.3414 | 0.3538 | 0.3531 | 0.2944 | 0.7680 | 0.4951 |
| | | S↑ | 0.7875 | 0.7729 | 0.8269 | 0.8324 | 0.7252 | 0.8015 | 0.7971 | 0.7729 | 0.8183 | 0.8378 | 0.7563 | 0.8346 |
| | | F↑ | 0.8858 | 0.8805 | 0.8969 | 0.9115 | 0.8456 | 0.9023 | 0.9258 | 0.9278 | 0.9365 | 0.9415 | 0.9137 | 0.9465 |
| | | V↑ | 1.4860 | 1.5100 | 1.5094 | 1.2935 | 1.3535 | 1.4377 | 1.0910 | 1.1110 | 1.1682 | 1.1156 | 1.1178 | 1.1453 |
| | **Ours** | P↓ | **0.0214** | **0.0186** | **0.0946** | **0.0426** | **0.1009** | **0.1903** | **0.0906** | **0.0782** | **0.1700** | **0.1070** | **0.0538** | **0.2946** |
| | | S↑ | 0.7622 | 0.7392 | 0.7874 | 0.8028 | 0.6539 | 0.7564 | 0.7398 | 0.7105 | 0.7764 | 0.7714 | 0.6918 | 0.7708 |
| | | F↑ | **0.9615** | **0.9587** | **0.9628** | **0.9642** | **0.9542** | **0.9607** | **0.9610** | **0.9563** | **0.9675** | **0.9660** | **0.9498** | **0.9668** |
| | | V↑ | **1.9761** | **2.0345** | **1.8837** | **1.7689** | **1.9371** | **1.9283** | **1.4448** | **1.4086** | **1.3820** | **1.2553** | **1.4788** | **1.3464** |

As shown in Table 2, YingMusic-Singer-Plus consistently outperforms Vevo2 across all six modification types under both Melody Control and Sing Edit settings in PER, F0-CORR, and VS, demonstrating strong adherence to both the reference melody and modified lyrics. The intelligibility gap is most pronounced on Trans and Mix tasks, indicating that reconstructing a substantially different phoneme sequence while preserving melody is inherently challenging. Note that PER on Mix tasks may be further inflated by ASR hallucinations on code-switched utterances.

Vevo2's incomplete melody disentanglement tends to reduce intelligibility and introduce hallucinations, whereas YingMusic-Singer-Plus's unified IPA tokenization and GRPO-based lyric adherence optimization maintain robustness even under extreme conditions. F0-CORR further distinguishes the two systems: benefiting from CKA alignment and GRPO, YingMusic-Singer-Plus maintains consistently high correlation across all tasks and languages, whereas Vevo2 fluctuates considerably.

For SIM, Vevo2 benefits from its multi-stage architecture, where an autoregressive LLM handles melody and content generation while a dedicated CFM focuses on timbre reconstruction. YingMusic-Singer-Plus instead adopts a single-stage CFM that jointly models all factors, prioritizing architectural simplicity. However, faithfully rendering modified lyrics while preserving the original melodic structure remains the primary concern in lyric editing.

**Table 3: Subjective evaluation on LyricEditBench.** N-MOS and M-MOS denote naturalness and melody adherence, respectively.

| Task | Model | ZH N↑ | ZH M↑ | EN N↑ | EN M↑ |
|------|-------|--------|--------|--------|--------|
| Melody Control | Vevo2 | 4.25 ± 0.06 | 4.28 ± 0.05 | 4.31 ± 0.05 | 4.31 ± 0.04 |
| | **Ours** | **4.31 ± 0.04** | **4.44 ± 0.04** | **4.36 ± 0.05** | **4.51 ± 0.03** |
| Sing Edit | Vevo2 | 4.48 ± 0.05 | 4.41 ± 0.05 | 4.44 ± 0.04 | 4.50 ± 0.04 |
| | **Ours** | **4.52 ± 0.04** | **4.55 ± 0.04** | **4.55 ± 0.04** | **4.58 ± 0.03** |

YingMusic-Singer-Plus consistently achieves higher N-MOS and M-MOS than Vevo2 across both tasks and languages. The strong subjective scores also indicate that GRPO optimization does not overfit to the reward models but generalizes to human perception. Vevo2 receives lower ratings overall with notably higher variance, and listeners report perceptible artifacts such as unfaithful lyric rendering and melodic misalignment.

### 4.2. Ablation Study

**Table 4: Ablation Study on LyricEditBench.** Best results are **bold**, second best underlined.

| Lg | Variant | Melody Control | | | | Sing Edit | | | |
|----|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | P↓ | S↑ | F↑ | V↑ | P↓ | S↑ | F↑ | V↑ |
| **ZH** | TTS Pretrain | 0.41 | 0.57 | 0.01 | 0.50 | 0.37 | 0.59 | 0.06 | 0.49 |
| | SFT Phase1 | <u>0.05</u> | <u>0.68</u> | 0.03 | 1.55 | <u>0.05</u> | 0.73 | 0.31 | 1.53 |
| | SFT Phase2 | 0.08 | 0.63 | 0.92 | 1.57 | 0.11 | 0.75 | 0.95 | 1.62 |
| | -w/o CKA | 0.08 | 0.64 | 0.91 | 1.57 | 0.12 | 0.75 | 0.93 | 1.61 |
| | -w/o Dist | 0.45 | 0.63 | 0.94 | 1.42 | 0.46 | 0.79 | 0.95 | 1.55 |
| | **Full Model** | **0.06** | 0.64 | **0.94** | **2.06** | **0.08** | <u>0.75</u> | **0.96** | **1.92** |
| **EN** | TTS Pretrain | 0.46 | 0.54 | 0.01 | 0.49 | 0.43 | 0.56 | 0.00 | 0.50 |
| | SFT Phase1 | <u>0.10</u> | <u>0.65</u> | 0.03 | 1.07 | <u>0.11</u> | 0.73 | 0.40 | 1.13 |
| | SFT Phase2 | 0.14 | 0.60 | 0.92 | 1.17 | 0.19 | 0.75 | 0.96 | 1.21 |
| | -w/o CKA | 0.14 | 0.60 | 0.90 | 1.19 | 0.20 | 0.75 | 0.94 | 1.18 |
| | -w/o Dist | 0.48 | 0.58 | 0.95 | 1.00 | 0.49 | 0.78 | 0.96 | 0.98 |
| | **Full Model** | **0.10** | 0.60 | **0.93** | **1.52** | **0.13** | <u>0.74</u> | **0.96** | **1.39** |

**Key findings from the ablation:**

The curriculum learning pipeline introduces clearly separable improvements at each stage:

1. **TTS Pretrain** establishes articulatory priors but lacks singing capability (F0-CORR near zero), with PER degrading significantly when a singing clip serves as the ICL prompt, indicating poor domain adaptation.

2. **SFT Phase 1** substantially improves all metrics, achieving the lowest PER as the model adapts to the singing domain while freely generating melody from the ICL prompt, bypassing the demanding task of explicit melody alignment. F0-CORR under Sing Edit improves slightly, showing partial melody capture from context alone, yet explicit guidance remains necessary for faithful reproduction.

3. **SFT Phase 2** activates the Melody Extractor and raises F0-CORR above 0.92, though PER increases, reflecting the difficulty of jointly maintaining melody fidelity and lyric faithfulness.

4. **GRPO** resolves this trade-off by recovering PER while further improving F0-CORR and VS with SIM unchanged, confirming that reward-based optimization jointly enhances all dimensions.

5. **CKA ablation (-w/o CKA)**: In SFT Phase 2, removing CKA reduces F0-CORR, confirming that CKA improves melody adherence.

6. **Temporal dropout ablation (-w/o Dist)**: Removing the temporal dropout applied to the melody latent $\tilde{\mathbf{h}}$ causes severe intelligibility degradation, as the unperturbed latent retains residual semantic information that the model exploits to bypass genuine lyric generation. Temporal dropout eliminates this leakage, forcing reliance on the abstract melodic contour and preserving prosodic structure while allowing free generation of modified lyrics.

---

## 5. Conclusion

We present **YingMusic-Singer-Plus**, a melody-controllable singing voice editing model that synthesizes from a timbre reference, a melody-providing singing clip, and modified lyrics without manual alignment. Through curriculum training and GRPO-based reinforcement learning, YingMusic-Singer-Plus achieves superior melody preservation and lyric adherence on LyricEditBench, the first comprehensive benchmark we introduce for this task, demonstrating strong potential for practical end-to-end singing voice editing.

---

## 6. Generative AI Use Disclosure

Generative AI tools are used solely for linguistic refinement and play no role in methodology, experimentation, interpretation, or the production of scientific results. The authors bear full intellectual responsibility for all content in this manuscript.

---

## 7. References

[1] J. Liu, C. Li, Y. Ren, F. Chen, and Z. Zhao, "Diffsinger: Singing voice synthesis via shallow diffusion mechanism," in *AAAI*. AAAI Press, 2022, pp. 11,020–11,028.

[2] J. He, J. Liu, Z. Ye, R. Huang, C. Cui, H. Liu, and Z. Zhao, "Rmssinger: Realistic-music-score based singing voice synthesis," in *ACL (Findings)*, vol. ACL 2023. ACL, 2023, pp. 236–248.

[3] Y. Zhang, R. Huang, R. Li, J. He, Y. Xia, F. Chen, X. Duan, B. Huai, and Z. Zhao, "Stylesinger: Style transfer for out-of-domain singing voice synthesis," in *AAAI*. AAAI Press, 2024, pp. 19,597–19,605.

[4] F. Wang, B. Bai, Y. Deng, J. Xue, Y. Gao, and Y. Li, "Expressivesinger: Synthesizing expressive singing voice as an instrument," in *ISCSLP*. IEEE, 2024, pp. 304–308.

[5] Y. Yu, J. Shi, Y. Wu, Y. Tang, and S. Watanabe, "Visinger2+: End-to-end singing voice synthesis augmented by self-supervised learning representation," in *SLT*. IEEE, 2024, pp. 719–726.

[6] Y. Zhang, W. Guo, C. Pan, D. Yao, Z. Zhu, Z. Jiang, Y. Wang, T. Jin, and Z. Zhao, "Tcsinger 2: Customizable multilingual zero-shot singing voice synthesis," in *ACL (Findings)*, vol. ACL 2025. ACL, 2025, pp. 13,280–13,294.

[7] S. Lei, Y. Zhou, B. Tang, M. W. Y. Lam, F. Liu, H. Liu, J. Wu, S. Kang, Z. Wu, and H. Meng, "Songcreator: Lyrics-based universal song generation," in *NeurIPS*, 2024.

[8] C. Yang, S. Wang, H. Chen, J. Yu, W. Tan, R. Gu, Y. Xu, Y. Zhou, H. Zhu, and H. Li, "Songeditor: Adapting zero-shot song generation language model as a multi-task editor," in *AAAI*. AAAI Press, 2025, pp. 25,597–25,605.

[9] X. Zhang, J. Zhang, Y. Wang, C. Wang, Y. Chen, D. Jia, Z. Chen, and Z. Wu, "Vevo2: A unified and controllable framework for speech and singing voice generation," *CoRR*, vol. abs/2508.16332, 2025.

[10] J. Qian, H. Meng, T. Zheng, P. Zhu, H. Lin, Y. Dai, H. Xie, W. Cao, R. Shang, J. Wu, H. Liu, H. Wen, J. Zhao, Z. Jiang, Y. Chen, S. Yin, M. Tao, J. Wei, L. Xie, and X. Wang, "Soulxsinger: Towards high-quality zero-shot singing voice synthesis," *CoRR*, vol. abs/2602.07803, 2026.

[11] D. Guo, D. Yang, H. Zhang, J. Song et al., "Deepseek-r1 incentivizes reasoning in llms through reinforcement learning," *Nat.*, vol. 645, no. 8081, pp. 633–638, 2025.

[12] Y. Zhang, C. Pan, W. Guo, R. Li, Z. Zhu, J. Wang, W. Xu, J. Lu, Z. Hong, C. Wang, L. Zhang, J. He, Z. Jiang, Y. Chen, C. Yang, J. Zhou, X. Cheng, and Z. Zhao, "Gtsinger: A global multi-technique singing corpus with realistic music scores for all singing tasks," in *NeurIPS*, 2024.

[13] Z. Evans, J. D. Parker, C. Carr, Z. Zukowski, J. Taylor, and J. Pons, "Long-form music generation with latent diffusion," in *ISMIR*, 2024, pp. 429–437.

[14] Z. Ning, H. Chen, Y. Jiang, C. Hao, G. Ma, S. Wang, J. Yao, and L. Xie, "Diffrhythm: Blazingly fast and embarrassingly simple end-to-end full-length song generation with latent diffusion," *CoRR*, vol. abs/2503.01183, 2025.

[15] Y. Chen, Z. Niu, Z. Ma, K. Deng, C. Wang, J. Zhao, K. Yu, and X. Chen, "F5-TTS: A fairytaler that fakes fluent and faithful speech with flow matching," in *ACL (1)*. ACL, 2025, pp. 6255–6271.

[16] J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang, "Flow-grpo: Training flow matching models via online RL," *CoRR*, vol. abs/2505.05470, 2025.

[17] J. Li, Y. Cui, T. Huang, Y. Ma, C. Fan, M. Yang, and Z. Zhong, "Mixgrpo: Unlocking flow-based GRPO efficiency with mixed ODE-SDE," *CoRR*, vol. abs/2507.21802, 2025.

[18] H. Wang, B. Tian, Y. Jiang, Z. Pan, S. Zhao, B. Ma, D. Chen, and X. Li, "Flowse-grpo: Training flow matching speech enhancement via online reinforcement learning," *CoRR*, vol. abs/2601.16483, 2026.

[19] DeepSeek-AI, "Deepseek-v3.2: Pushing the frontier of open large language models," *CoRR*, vol. abs/2512.02556, 2025.

[20] H. He, Z. Shang, C. Wang, X. Li, Y. Gu, H. Hua, L. Liu, C. Yang, J. Li, P. Shi, Y. Wang, K. Chen, P. Zhang, and Z. Wu, "Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation," in *SLT*. IEEE, 2024, pp. 885–890.

[21] C. Hao, R. Yuan, J. Yao, Q. Deng, X. Bai, W. Xue, and L. Xie, "Songformer: Scaling music structure analysis with heterogeneous supervision," *CoRR*, vol. abs/2510.02797, 2025.

[22] J. Wang, W. T. Lu, and M. Won, "Mel-band roformer for music source separation," *CoRR*, vol. abs/2310.01809, 2023.

[23] A. Plaquet and H. Bredin, "Powerset multi-class cross entropy loss for neural speaker diarization," in *INTERSPEECH*. ISCA, 2023, pp. 3222–3226.

[24] H. Bredin, "pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe," in *INTERSPEECH*. ISCA, 2023, pp. 1983–1987.

[25] C. K. A. Reddy, V. Gopal, and R. Cutler, "Dnsmos: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors," in *ICASSP*. IEEE, 2021, pp. 6493–6497.

[26] ——, "Dnsmos P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors," in *ICASSP*. IEEE, 2022, pp. 886–890.

[27] X. Shi, X. Wang, Z. Guo, Y. Wang, P. Zhang, X. Zhang, Z. Guo, H. Hao, Y. Xi, B. Yang, J. Xu, J. Zhou, and J. Lin, "Qwen3-asr technical report," *CoRR*, vol. abs/2601.21337, 2026.

[28] H. Wei, X. Cao, T. Dan, and Y. Chen, "RMVPE: A robust model for vocal pitch estimation in polyphonic music," in *INTERSPEECH*. ISCA, 2023, pp. 5421–5425.

[29] Z. Wang, R. Yuan, Z. Geng, H. Li, X. Qu, X. Li, S. Chen, H. Fu, R. B. Dannenberg, and K. Zhang, "Singing timbre popularity assessment based on multimodal large foundation model," in *ACM Multimedia*. ACM, 2025, pp. 12,227–12,236.

















