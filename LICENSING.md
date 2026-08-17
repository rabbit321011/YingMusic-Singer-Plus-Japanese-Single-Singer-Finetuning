# Licensing Guide

This repository contains material under several licenses. It is not licensed as
one indivisible work under a single license.

## Order of interpretation

Apply the following rules in order:

1. A license or copyright header attached to a file or component controls that
   material.
2. A license retained inside a third-party directory controls that component.
3. The exceptions and attributions in `THIRD_PARTY_NOTICES.md` apply next.
4. Only material for which `rabbit321011` owns the copyright falls back to the
   root `LICENSE` (PolyForm Noncommercial 1.0.0).

## Original contributions by rabbit321011

Original code, documentation, research records, configuration examples, and
copyrightable modifications authored by `rabbit321011` are licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

This generally includes original portions of `config/`, `docs/`, `results/`,
`scripts/`, `src/archive_training/`, `src/package_v4c_finetune/`, and original
additions or modifications in `src/YingMusic-Singer-Plus/`. These directory
names describe likely ownership boundaries; they do not override a file header,
copied third-party code, or an upstream license.

Commercial use of those original contributions requires separate permission
from `rabbit321011`.

## YingMusic-Singer-Plus upstream material

Code, documentation, model-related material, and other portions originating
from [ASLP-lab/YingMusic-Singer-Plus](https://github.com/ASLP-lab/YingMusic-Singer-Plus)
remain under [CC BY 4.0](LICENSE-CC-BY-4.0), except for the Stable Audio-derived
materials described below. CC BY 4.0 permits commercial use and those rights are
not restricted by the PolyForm license.

When a file combines upstream material with modifications by `rabbit321011`,
the upstream portion remains under CC BY 4.0 and the copyrightable modifications
are under PolyForm Noncommercial 1.0.0. A recipient who uses the combined file
must respect both sets of rights; a recipient may still isolate and use the
upstream material under CC BY 4.0.

## Stable Audio-derived material

The Stable Audio-derived VAE inference code under
`src/YingMusic-Singer-Plus/src/YingMusicSinger/utils/stable_audio_tools/` is
licensed under the [Stability AI Community License](LICENSE-STABILITY), not
PolyForm or CC BY 4.0.

**Powered by Stability AI**

## Other third-party material and external models

Amphion G2P code, MusicSourceSeparationTraining, py3langid-derived code,
PaddlePaddle-derived code, GAME, SOFA, Whisper, and other third-party components
retain their own terms. See `THIRD_PARTY_NOTICES.md` and the license files
shipped with those components.

External model weights are not relicensed merely because this repository can
load them. Obtain each model separately and review its model card or release
terms before use.

This guide records the repository maintainer's licensing intent and is not
legal advice.
