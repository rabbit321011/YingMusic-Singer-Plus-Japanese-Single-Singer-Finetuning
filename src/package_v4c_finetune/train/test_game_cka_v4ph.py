import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YING_REPO = os.path.join(ROOT, "YingMusic-Singer-Plus-src")
if not os.path.isdir(os.path.join(YING_REPO, "src")):
    YING_REPO = ROOT
sys.path.insert(0, YING_REPO)

from src.YingMusicSinger.melody.game_cka_v4ph import (  # noqa: E402
    expand_game_note_features,
    game_note_probs_to_some_candidates,
    game_note_probs_to_some_probs,
    some_frame_count,
    some_gaussian_pitch_kernel,
)


def main():
    scores = torch.tensor([60.0, 61.5, 0.0])
    presence = torch.tensor([True, True, False])
    durations = torch.tensor([0.02, 0.01, 0.02])
    centers = torch.linspace(0, 128, 257)
    game_probs = torch.exp(
        -0.5 * ((centers[None, :] - scores[:, None]) / 0.5) ** 2
    ) * presence[:, None]

    candidates = game_note_probs_to_some_candidates(
        note_probs=game_probs,
        durations=durations,
        presence=presence,
        scores=scores,
    )
    assert set(candidates) == {
        "posterior_integer_bins",
        "posterior_broadened",
        "decoded_score_kernel",
    }
    for value in candidates.values():
        assert value.shape == (3, 128)
        assert torch.isfinite(value).all()
        assert ((value >= 0) & (value <= 1)).all()
        assert torch.equal(value[2], torch.zeros(128))

    selected = game_note_probs_to_some_probs(game_probs, durations, presence)
    assert torch.equal(selected, candidates["posterior_integer_bins"])

    canonical = some_gaussian_pitch_kernel(scores, presence)
    assert torch.equal(candidates["decoded_score_kernel"], canonical)
    assert int(canonical[0].argmax()) == 60
    assert torch.isclose(canonical[1, 61], canonical[1, 62])

    target_frames = some_frame_count(2205)
    expanded = expand_game_note_features(
        note_features=candidates["posterior_broadened"],
        durations=durations,
        presence=presence,
        audio_duration=0.05,
        target_frames=target_frames,
    )
    assert expanded["pitch_probs"].shape == (target_frames, 128)
    assert expanded["boundary_probs"].shape == (target_frames, 1)
    assert expanded["note_ids"].shape == (target_frames,)
    assert int(expanded["boundary_probs"].sum()) == 3
    assert not expanded["presence"][-1]
    print("GAME-to-SOME CKA adapter tests passed")


if __name__ == "__main__":
    main()
