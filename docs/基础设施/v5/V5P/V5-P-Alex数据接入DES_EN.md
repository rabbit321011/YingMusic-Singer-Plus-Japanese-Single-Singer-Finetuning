> Public sanitized snapshot. Local paths, artifact locations, and fingerprints are redacted; see `docs/PUBLICATION_SCOPE.md`.

# V5-P-Alex Data Integration Guide

This document describes the data package that Alex must prepare. The training script reads the final
token arrays only. It does not run G2P, SOFA, English alignment, or note-event parsing.

The supplied DataKit requires only Python 3.10+ and NumPy. It does not require CUDA, PyTorch,
YMSP/GAME source code, VAE weights, or a training checkpoint. Alex continues to use his own English
G2P/alignment and MIDI tools to produce timestamped source information.

## 1. Required files

For each sample, provide:

```text
audio/song_0001.wav
h_tokens/song_0001.npy
midi_p_tokens/song_0001.npy
manifest.jsonl
```

Do not provide phone intervals, note events, GAME posterior files, or CKA targets as training inputs.

## 2. Audio and VAE

Audio must be:

```text
sample rate: 44100 Hz
channels: mono
format: PCM16 WAV
```

Let `N` be the sample-frame count in the WAV header. The frozen V5-P-Alex contract uses:

```text
T = N // 2048
```

This is the exact length formula for the current YMSP official VAE with the audio format above, not
a 20 Hz or 100 Hz estimate. Run `python tools/timebase.py audio/song_0001.wav` to obtain `N` and `T`.
Alex does not install or run the VAE; the training server runs the real VAE and checks the length
again. Do not reuse this formula if the audio format or VAE changes.

The final arrays must satisfy:

```text
len(h_tokens)      == T
len(midi_p_tokens) == T
every sentence_start_frames value is in [0, T-1]
```

## 3. H tokens (text condition)

H tokens are a dense one-dimensional integer array:

```text
h_tokens: int64[T]
```

### ID mapping

The attached `vocab.json` maps text/phoneme symbols to 0-based base IDs.

```text
base_id = vocab.json[token]
h_token_id = base_id + 1
```

Examples:

```text
vocab.json["ɪ"] = 6  -> H token 7
vocab.json["p"]  = 27 -> H token 28
vocab.json["t"]  = 29 -> H token 30
```

Special H IDs do not use this conversion:

```text
0       = filler / empty frame
365     = SEP
366     = PUL
```

Do not create a new vocabulary or guess ordinary text token IDs manually.

### H-token generation

Your offline preprocessing may use YMSP CNEN G2P and an English alignment tool. The training script
will not run them again:

```text
English lyrics
  -> CNEN G2P
  -> English alignment
  -> timestamped text-token events
  -> H IDs using vocab.json
  -> dense h_tokens[T]
```

Initialize the array with `0`. Place each text event on its latent frame. Use `365` for SEP and `366`
for PUL. Preserve the order of ordinary text tokens. Never silently overwrite two events; collision
handling must be deterministic and reproducible.

### Sentence-start frame array

Each manifest record must also contain:

```text
sentence_start_frames: [s0, s1, ..., sN-1]
```

This is not an array of milliseconds or seconds. Each `s_i` is an integer frame index on the
official-VAE timeline. The following rules are mandatory:

1. The array follows sentence order, is non-empty, and is strictly increasing;
2. `h_tokens[s_i]` is the first ordinary H token of that sentence, not `0`, SEP, or PUL;
3. Each sentence has exactly one SEP, so the SEP count equals the array length;
4. Sentence `i` has SEP at `s_(i+1)-1`; the final sentence has SEP at `T-1`;
5. At least one sentence start is greater than `0`, so training can split A and B.

Generate H from absolute audio timing. Do not assume an A/B boundary and then move an entire
sentence. The training side uses this array to choose a reference boundary that never cuts through
a sentence.

## 4. MIDI-P tokens (melody condition)

MIDI-P is also a dense one-dimensional integer array:

```text
midi_p_tokens: int16[T]
```

ID mapping:

```text
0..254 = 0.5-semitone pitch class
255    = REST
256    = PAD
```

Pitch conversion:

```text
pitch_class = round(MIDI_pitch * 2)
C5 (MIDI 72) -> 144
D5 (MIDI 74) -> 148
```

If your internal tool uses `0` for silence, convert it to `255` before export. The final file must
contain integer tokens, not `C5`, `D5`, or note-event objects.

## 5. Timestamp-to-frame mapping

H and MIDI-P use the same latent timeline. Let:

```text
D = audio duration in seconds
T = WAV sample-frame count // 2048
```

The center time of frame `i` is:

```text
center_i = (i + 0.5) * D / T
```

Map a timestamped event to the nearest frame center:

```text
frame_index = argmin_i(abs(center_i - event_time))
```

Example: with `D=2.0` seconds and `T=43`, an event at about `0.20` seconds maps to frame `4`.

Convert each sentence-start timestamp with the same rule and write the resulting integer frame
indices to `sentence_start_frames`. Do not write the original seconds or milliseconds there.

For a sustained MIDI note, fill the pitch class into every frame whose center lies inside the note
interval. Use `255` for frames without a pitch.

## 6. Directory and manifest

```text
v5p_alex_data/
├── audio/
│   └── song_0001.wav
├── h_tokens/
│   └── song_0001.npy
├── midi_p_tokens/
│   └── song_0001.npy
├── manifest.jsonl
└── vocab.json
```

Manifest example:

```json
{
  "schema": "v5p_alex_input_v2",
  "sample_id": "song_0001",
  "language": "en",
  "audio_path": "audio/song_0001.wav",
  "audio_sha256": "...",
  "audio_frames": 546840,
  "duration_seconds": 12.4,
  "vae_contract": "ymsp_official_vae_44100_ratio2048_v1",
  "vae_frame_count": 267,
  "sentence_start_frames": [18, 93, 171],
  "h_tokens_path": "h_tokens/song_0001.npy",
  "h_tokens_sha256": "...",
  "midi_p_tokens_path": "midi_p_tokens/song_0001.npy",
  "midi_p_tokens_sha256": "..."
}
```

The following equality is mandatory:

```text
vae_frame_count == len(h_tokens) == len(midi_p_tokens)
```

The DataKit includes two example sets:

```text
example/                         one copyright-free, one-second smoke sample
examples/hanamaru_10_v2/         ten real Hanamaru v2 format samples
```

The Hanamaru set comes from the audited server fixture and contains ten audio records, 4,776 latent
frames, and 70 sentence starts. It demonstrates real audio/H/MIDI-P/sentence-start/SHA256
correspondence; it is not English training data and must not be mixed into Alex's English dataset.
The byte-exact server `source_manifest.jsonl` is retained. The DataKit `manifest.jsonl` adds only the
Japanese example marker and frozen VAE contract, without changing assets, sentence starts, or
hashes. Use `tools/create_manifest.py` for new manifests instead of writing them by hand. Finally
run `tools/validate_dataset.py path/to/dataset/manifest.jsonl --output validation.json`.

## 7. What the training side does

The training side loads the H and MIDI-P arrays and uses the official VAE for the audio. The CKA
target is generated at training time by a frozen GAME teacher. You do not provide CKA data.

At every step, training first makes the normal randomized V5-P reference-length proposal, then
snaps it to the nearest positive value in `sentence_start_frames`. The final A/B boundary is always
one of the sentence starts supplied by you.

## 8. Delivery checklist

- Audio format and SHA256 are correct;
- H-token length equals the official VAE frame count `T`;
- `sentence_start_frames` contains strictly increasing VAE-frame indices bound to the first H token
  of each sentence;
- SEP count and positions agree with the sentence-start array;
- MIDI-P-token length equals the same `T`;
- H IDs follow the attached `vocab.json` plus-one rule;
- H uses `0`, `SEP=365`, and `PUL=366` correctly;
- MIDI-P uses `255=REST` and `256=PAD` correctly;
- Every manifest path exists;
- No token collision is silently overwritten or randomly resolved.
