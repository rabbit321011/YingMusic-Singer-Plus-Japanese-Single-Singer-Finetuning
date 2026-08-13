#!/usr/bin/env python3
import importlib.util
import pathlib

import torch


SOURCE = pathlib.Path(__file__).with_name("train_v4ph_lcf.py")


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_v4ph_lcf_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = load_training_module()

    hidden = torch.tensor(
        [[[100.0, 100.0], [3.0, 4.0], [3.0, 4.0]]]
    )
    pooled = module.pool_lcf_feature(hidden, ref_len=1)
    torch.testing.assert_close(pooled, torch.tensor([[0.6, 0.8]]))
    try:
        module.pool_lcf_feature(hidden, ref_len=3)
    except ValueError:
        pass
    else:
        raise AssertionError("empty B region was accepted")

    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    positive = torch.tensor([[0.9, 0.1]], requires_grad=True)
    negative_bank = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
        requires_grad=True,
    )
    loss, positive_distance, negative_distances = (
        module.lcf_per_anchor_objective(
            query, positive, negative_bank, tau=0.5
        )
    )
    expected = (
        positive_distance / 0.5
        + torch.logsumexp(-negative_distances / 0.5, dim=1)
    )
    torch.testing.assert_close(loss, expected)
    loss.mean().backward()
    if query.grad is None or not torch.isfinite(query.grad).all():
        raise AssertionError("LCF query gradient is missing or non-finite")
    if positive.grad is not None or negative_bank.grad is not None:
        raise AssertionError("LCF positive/negative detach contract failed")

    close_loss = float(loss)
    far_positive = torch.tensor([[-1.0, 0.0]])
    far_loss, _, _ = module.lcf_per_anchor_objective(
        query.detach(), far_positive, negative_bank.detach(), tau=0.5
    )
    if not float(far_loss) > close_loss:
        raise AssertionError("LCF positive ordering is inverted")

    # With manual post-backward rank averaging, scaling each active rank by
    # world_size / active_count yields the mean over active anchors.
    rank_losses = torch.tensor([0.0, 2.0, 0.0, 4.0])
    active = torch.tensor([0.0, 1.0, 0.0, 1.0])
    scaled_rank_mean = (
        rank_losses * (4.0 / active.sum()) * active
    ).mean()
    torch.testing.assert_close(scaled_rank_mean, torch.tensor(3.0))

    print("V4PH LCF contract tests passed")


if __name__ == "__main__":
    main()
