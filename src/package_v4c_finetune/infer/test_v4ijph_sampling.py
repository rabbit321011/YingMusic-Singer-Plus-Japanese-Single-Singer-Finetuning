import unittest

import torch
from torch import nn

from v4ijph_sampling import sample_full_timeline_style


class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_cfg_ids = None

    def forward(self, *, x, cond, cfg_infer=False, cfg_infer_ids=None, **kwargs):
        self.last_cfg_ids = cfg_infer_ids
        if cfg_infer:
            return torch.cat((torch.ones_like(x), torch.zeros_like(x)), dim=0), None
        return torch.zeros_like(x), None

    def clear_cache(self):
        return None


class DummyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.transformer = DummyTransformer()
        self.num_channels = 64
        self.odeint_kwargs = {"method": "euler"}


class V4IjPHSamplingTest(unittest.TestCase):
    @staticmethod
    def fake_odeint(function, initial, timeline, **kwargs):
        final = initial + (timeline[-1] - timeline[0]) * function(
            timeline[0], initial
        )
        return torch.stack((initial, final))

    def test_cfg_keeps_style_and_only_drops_content_branch(self):
        policy = DummyPolicy()
        output, trajectory = sample_full_timeline_style(
            policy,
            style_cond=torch.ones(1, 3, 64),
            text=torch.ones(1, 3, dtype=torch.long),
            midi=torch.ones(1, 3, 128),
            duration=3,
            steps=1,
            cfg_strength=2.0,
            seed=7,
            _odeint=self.fake_odeint,
        )
        self.assertEqual(tuple(output.shape), (1, 3, 64))
        self.assertEqual(tuple(trajectory.shape), (2, 1, 3, 64))
        self.assertEqual(
            policy.transformer.last_cfg_ids, (True, False, True, False)
        )

    def test_shape_contract_rejects_prompt_length_cond(self):
        policy = DummyPolicy()
        with self.assertRaisesRegex(ValueError, "style_cond"):
            sample_full_timeline_style(
                policy,
                style_cond=torch.zeros(1, 1, 64),
                text=torch.zeros(1, 3, dtype=torch.long),
                midi=torch.zeros(1, 3, 128),
                duration=3,
                steps=1,
                cfg_strength=1.0,
                seed=7,
                _odeint=self.fake_odeint,
            )


if __name__ == "__main__":
    unittest.main()
