import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2] / "YingMusic-Singer-Plus-src"
sys.path.insert(0, str(REPO))

from src.YingMusicSinger.melody.game_p_v4pf import (  # noqa: E402
    game_notes_to_tracks,
    quantize_game_notes,
)


def main():
    durations = torch.tensor([0.02, 0.01, 0.03, 0.0])
    presence = torch.tensor([True, False, True, False])
    scores = torch.tensor([60.24, 0.0, 61.76, 0.0])
    classes, frames, valid = quantize_game_notes(durations, presence, scores)
    assert classes.tolist() == [120, 255, 124, 256]
    assert frames.tolist() == [2, 1, 3, 0]
    assert valid.tolist() == [True, True, True, False]

    tracks = game_notes_to_tracks(
        durations=durations,
        presence=presence,
        scores=scores,
        expected_native_frames=6,
        target_len=3,
    )
    assert tracks["native_classes"].tolist() == [120, 120, 255, 124, 124, 124]
    assert tracks["native_note_ids"].tolist() == [1, 1, 2, 3, 3, 3]
    assert tracks["model_classes"].tolist() == [120, 124, 124]
    assert tracks["model_note_ids"].tolist() == [1, 3, 3]
    assert tracks["native_frame_delta"] == 0

    one_frame_short = game_notes_to_tracks(
        durations=durations,
        presence=presence,
        scores=scores,
        expected_native_frames=7,
        target_len=3,
    )
    assert one_frame_short["native_frame_delta"] == -1

    try:
        quantize_game_notes(
            torch.tensor([0.01]), torch.tensor([True]), torch.tensor([127.6])
        )
    except ValueError as error:
        assert "outside the V4Pf schema" in str(error)
    else:
        raise AssertionError("Out-of-range GAME pitch was silently accepted")

    print("GAME-to-P adapter tests passed")


if __name__ == "__main__":
    main()
