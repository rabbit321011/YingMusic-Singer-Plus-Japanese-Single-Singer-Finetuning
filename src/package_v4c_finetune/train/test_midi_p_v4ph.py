import os
import sys

import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YING_REPO = os.path.join(ROOT, "YingMusic-Singer-Plus-src")
if not os.path.isdir(os.path.join(YING_REPO, "src")):
    YING_REPO = ROOT
sys.path.insert(0, YING_REPO)

from src.YingMusicSinger.melody.midi_p_v4ph import (  # noqa: E402
    V4PHMIDIEmbedding,
    fill_unsupported_pitch_rows,
    matched_random_embedding,
    pitch_kernel_distance,
    structured_pitch_kernel,
)


def main():
    kernel = structured_pitch_kernel()
    assert kernel.shape == (257, 128)
    assert int(kernel[120].argmax()) == 60
    assert torch.isclose(kernel[121, 60], kernel[121, 61])
    assert torch.equal(kernel[255], torch.zeros(128))
    assert torch.equal(kernel[256], torch.zeros(128))

    random_a = matched_random_embedding(42)
    random_b = matched_random_embedding(42)
    assert torch.equal(random_a, random_b)
    assert not torch.equal(random_a[:255], kernel[:255])
    assert torch.isclose(
        random_a[:255].std(unbiased=False),
        kernel[:255].std(unbiased=False),
        rtol=1e-5,
    )

    module = V4PHMIDIEmbedding(seed=42)
    classes = torch.tensor([[120, 255, 125, 256]])
    output = module(classes)
    assert output.shape == (1, 4, 128)
    output[:, :3].sum().backward()
    assert module.embedding.weight.grad is not None
    assert torch.equal(module.embedding.weight.grad[256], torch.zeros(128))

    support = torch.zeros(257, dtype=torch.long)
    support[120] = 10
    support[121] = 5
    projection = nn.Linear(128, 128)
    metrics = pitch_kernel_distance(module.embedding.weight, projection, support)
    assert metrics["supported_pitch_rows"] == 2
    replaced = fill_unsupported_pitch_rows(module.embedding.weight, support)
    assert replaced == 253
    assert torch.equal(module.embedding.weight[0], kernel[0])
    assert torch.equal(module.embedding.weight[256], torch.zeros(128))
    print("V4PH P embedding tests passed")


if __name__ == "__main__":
    main()
