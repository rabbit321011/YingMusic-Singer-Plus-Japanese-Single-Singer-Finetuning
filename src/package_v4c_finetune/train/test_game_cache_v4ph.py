import os
import sys
import tempfile

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YING_REPO = os.path.join(ROOT, "YingMusic-Singer-Plus-src")
if not os.path.isdir(os.path.join(YING_REPO, "src")):
    YING_REPO = ROOT
sys.path.insert(0, YING_REPO)

from src.YingMusicSinger.melody.game_cache_v4ph import (  # noqa: E402
    canonicalize_game_cache,
    game_cache_to_model_tracks,
    load_game_cache,
    save_game_cache,
)


def main():
    centers = torch.linspace(0, 128, 257)
    scores = torch.tensor([60.0, 0.0, 62.5])
    presence = torch.tensor([True, False, True])
    probs = torch.exp(-0.5 * ((centers[None, :] - scores[:, None]) / 0.5) ** 2)
    probs = probs * presence[:, None]
    arrays = canonicalize_game_cache(
        {
            "durations": torch.tensor([0.02, 0.01, 0.02]),
            "presence": presence,
            "scores": scores,
            "pitch_probs_257": probs,
        }
    )
    assert arrays["classes"].tolist() == [120, 255, 125]
    assert arrays["pitch_probs_128"].shape == (3, 128)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "sample.npz")
        save_game_cache(path, arrays)
        restored = load_game_cache(path)
    assert torch.equal(restored["classes"], torch.tensor([120, 255, 125], dtype=torch.int16))
    assert torch.equal(restored["pitch_probs_128"][1], torch.zeros(128))
    tracks = game_cache_to_model_tracks(restored, num_samples=2205, target_len=2)
    assert tracks["p_classes"].shape == (2,)
    assert tracks["cka_probs"].shape == (2, 128)
    assert tracks["some_frames"] == 5
    print("V4PH GAME cache tests passed")


if __name__ == "__main__":
    main()
