# Third-Party Notices

This repository contains source snapshots, modified upstream files, copied
third-party code, and adapters for external tools and models. This file records
their known provenance. It does not replace the controlling license texts.

License information below was rechecked on 2026-08-17.

## YingMusic-Singer-Plus official snapshot

Upstream: <https://github.com/ASLP-lab/YingMusic-Singer-Plus>

Locations:

- `third_party/YingMusic-Singer-Plus-official/`
- upstream portions of `src/YingMusic-Singer-Plus/`

The upstream code and model-related materials are licensed under CC BY 4.0,
except for the Stable Audio-derived materials described below. The upstream
license is retained in the snapshot and as `LICENSE-CC-BY-4.0` at repository
root. Copyrightable modifications by `rabbit321011` are separately covered by
PolyForm Noncommercial 1.0.0; this does not restrict rights in the upstream
material itself.

The public snapshot omits upstream example audio. Its README command paths were
replaced with generic `/path/to/...` placeholders so the documentation does not
refer to media that is absent from this repository. The retained upstream app
source may still contain its original demo-preset definitions.

## Stable Audio and related components

Stable Audio-derived VAE materials, including the inference code under
`src/YingMusic-Singer-Plus/src/YingMusicSinger/utils/stable_audio_tools/`, are
subject to the Stability AI Community License. Exact copies are retained as
`LICENSE-STABILITY`, `src/YingMusic-Singer-Plus/LICENSE-STABILITY`, and
`third_party/YingMusic-Singer-Plus-official/LICENSE-STABILITY`.

Required Stability AI notice:

> This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved

**Powered by Stability AI**

## Amphion / MaskGCT G2P

Upstream: <https://github.com/open-mmlab/Amphion>

Amphion-derived G2P source is distributed under
`src/YingMusic-Singer-Plus/src/YingMusicSinger/utils/f5_tts/g2p/` and
`src/package_v4c_finetune/g2p/`. The files retain Amphion copyright headers and
are licensed under MIT. The required license text is retained as
`LICENSES/AMPHION-MIT.txt` and beside each redistributed G2P source tree.
PolyForm does not apply to the copied Amphion code;
it applies only to independently copyrightable modifications by
`rabbit321011`, if any.

## F5-TTS

Upstream: <https://github.com/SWivid/F5-TTS>

YingMusic-Singer-Plus credits F5-TTS for its DiT/CFM backbone. F5-TTS is MIT
licensed; its license text is retained as `LICENSES/F5-TTS-MIT.txt`. This notice
does not claim that every F5-inspired architecture is a literal source copy.

## MusicSourceSeparationTraining

Upstream: <https://github.com/ZFTurbo/Music-Source-Separation-Training>

Locations:

- `third_party/YingMusic-Singer-Plus-official/src/third_party/MusicSourceSeparationTraining/`
- `src/YingMusic-Singer-Plus/src/third_party/MusicSourceSeparationTraining/`

This component is MIT licensed. Its upstream license and copyright notice are
retained in each copy's `LICENSE` file. Individual bundled subimplementations
may contain additional copyright headers and notices which remain controlling.

## py3langid / LangSegment-derived code

Upstream: <https://github.com/adbar/py3langid>

`src/YingMusic-Singer-Plus/src/YingMusicSinger/utils/f5_tts/thirdparty/LangSegment/LangSegment.py`
contains py3langid-derived language identification code. It is licensed under
the BSD 3-Clause License. The required notice is retained as
`LICENSES/PY3LANGID-BSD-3-CLAUSE.txt` and beside the redistributed source.

The adjacent `utils/num.py` carries a PaddlePaddle Apache-2.0 copyright and
license header and notes a GPT-SoVITS source. The Apache-2.0 license text is
retained as `LICENSES/PADDLEPADDLE-APACHE-2.0.txt`.

## External runtime tools and model weights

The following components are referenced or loaded at runtime but their source
repositories and model weights are not included in this public repository.
Adapters authored in this repository do not change the external component's
license.

| Component | Code license | Model or asset terms used by this project | Source |
|---|---|---|---|
| OpenVPI GAME | MIT | GAME 1.0 release weights: CC BY-NC-SA 4.0 | <https://github.com/openvpi/GAME>, <https://github.com/openvpi/GAME/releases/tag/v1.0.0> |
| qiuqiao SOFA | MIT | Greenleaf2001 `JPN_Test2_Plus`: release says `Commercial Use: Not Approved`; the model repository has no repository-wide license file | <https://github.com/qiuqiao/SOFA>, <https://github.com/Greenleaf2001/SOFA_Models/releases/tag/JPN_Test2_Plus> |
| OpenAI Whisper | MIT code | `openai/whisper-large-v3` model card: Apache-2.0 | <https://github.com/openai/whisper>, <https://huggingface.co/openai/whisper-large-v3> |
| SYSTRAN faster-whisper | MIT | `Systran/faster-whisper-large-v3` and `faster-whisper-medium` model cards: MIT | <https://github.com/SYSTRAN/faster-whisper>, <https://huggingface.co/Systran/faster-whisper-large-v3> |
| OpenVPI SOME | MIT | Obtain weights separately and review the release terms | <https://github.com/openvpi/SOME> |
| Voicebank2DiffSinger | GPL-3.0 | Referenced as a workflow/G2P pattern; no source snapshot from this project was found in the repository | <https://github.com/Lqm1/Voicebank2DiffSinger> |
| Microsoft WavLM | See its model card and linked UniSpeech license | `microsoft/wavlm-large` is downloaded separately | <https://huggingface.co/microsoft/wavlm-large> |

The SOFA code license and `JPN_Test2_Plus` model terms are separate. The MIT
license on the SOFA program does not grant commercial permission for that
third-party model. Because the model release provides a restriction rather
than a complete standard license, do not infer rights beyond the release page;
contact the model publisher for any commercial use.

## Exclusions

The public repository does not include upstream model weights, GAME or SOFA
weights, Whisper weights, private datasets, training audio, generated audio,
or private manifests. Their licenses and permissions are separate from the
source-code licenses above.
