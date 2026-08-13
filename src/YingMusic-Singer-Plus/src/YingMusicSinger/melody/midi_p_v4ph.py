import torch
import torch.nn as nn
import torch.nn.functional as F


MIDI_P_V4PH_SCHEMA = {
    "version": 1,
    "source": "OpenVPI/GAME medium K4",
    "pitch_scale": 2,
    "pitch_class_count": 255,
    "rest_id": 255,
    "pad_id": 256,
    "num_embeddings": 257,
    "embedding_dim": 128,
    "fuzz_disturb": False,
    "cka_target": "game_some_compatible_probabilities",
    "init": "matched_random",
}


def structured_pitch_kernel(device=None, dtype=torch.float32):
    pitches = torch.arange(255, device=device, dtype=dtype) / 2
    centers = torch.arange(128, device=device, dtype=dtype)
    pitch_rows = torch.exp(-0.5 * (centers[None, :] - pitches[:, None]).square())
    special = torch.zeros(2, 128, device=device, dtype=dtype)
    return torch.cat([pitch_rows, special], dim=0)


def matched_random_embedding(seed, device=None, dtype=torch.float32):
    kernel = structured_pitch_kernel(device=device, dtype=dtype)
    generator_device = device if device is not None else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    random_weight = torch.randn(
        kernel.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    pitch_kernel = kernel[:255]
    mean = pitch_kernel.mean()
    std = pitch_kernel.std(unbiased=False)
    pitch_random = random_weight[:255]
    pitch_random = (
        pitch_random - pitch_random.mean()
    ) / pitch_random.std(unbiased=False).clamp(min=1e-8)
    random_weight[:255] = pitch_random * std + mean
    rest_random = random_weight[255]
    rest_random = (
        rest_random - rest_random.mean()
    ) / rest_random.std(unbiased=False).clamp(min=1e-8)
    random_weight[255] = rest_random * std + mean
    random_weight[256].zero_()
    return random_weight


def _linear_cka(x, y):
    x = F.normalize(x.float(), dim=-1, eps=1e-8)
    y = F.normalize(y.float(), dim=-1, eps=1e-8)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = x.transpose(0, 1) @ y
    xx = x.transpose(0, 1) @ x
    yy = y.transpose(0, 1) @ y
    denominator = torch.sqrt(xx.square().sum() * yy.square().sum()).clamp(min=1e-8)
    return cross.square().sum() / denominator


@torch.no_grad()
def pitch_kernel_distance(embedding_weight, midi_proj, support_counts):
    if embedding_weight.shape != (257, 128):
        raise ValueError(f"Unexpected P embedding shape: {embedding_weight.shape}")
    if support_counts.shape != (257,):
        raise ValueError(f"Unexpected support count shape: {support_counts.shape}")
    mask = (support_counts[:255] > 0).to(embedding_weight.device)
    if int(mask.sum()) < 2:
        raise ValueError("At least two supported pitch rows are required")
    learned = embedding_weight[:255][mask].float()
    kernel = structured_pitch_kernel(
        device=embedding_weight.device, dtype=embedding_weight.dtype
    )[:255][mask].float()
    projected_learned = midi_proj(learned)
    projected_kernel = midi_proj(kernel)
    return {
        "supported_pitch_rows": int(mask.sum()),
        "support_frames": int(support_counts[:255].sum()),
        "raw_rmse": float(torch.sqrt((learned - kernel).square().mean())),
        "raw_cosine_distance": float(
            (1 - F.cosine_similarity(learned, kernel, dim=-1)).mean()
        ),
        "projected_rmse": float(
            torch.sqrt((projected_learned - projected_kernel).square().mean())
        ),
        "projected_cosine_distance": float(
            (
                1
                - F.cosine_similarity(
                    projected_learned, projected_kernel, dim=-1
                )
            ).mean()
        ),
        "pitch_geometry_cka": float(_linear_cka(learned, kernel)),
    }


@torch.no_grad()
def fill_unsupported_pitch_rows(embedding_weight, support_counts):
    if embedding_weight.shape != (257, 128) or support_counts.shape != (257,):
        raise ValueError("Unexpected P embedding or support count shape")
    unsupported = support_counts[:255].to(embedding_weight.device) == 0
    kernel = structured_pitch_kernel(
        device=embedding_weight.device, dtype=embedding_weight.dtype
    )
    embedding_weight[:255][unsupported] = kernel[:255][unsupported]
    embedding_weight[256].zero_()
    return int(unsupported.sum())


class V4PHMIDIEmbedding(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        self.embedding = nn.Embedding(257, 128, padding_idx=256)
        with torch.no_grad():
            self.embedding.weight.copy_(matched_random_embedding(seed))

    def forward(self, classes):
        if classes.dtype != torch.long:
            classes = classes.long()
        if (classes < 0).any() or (classes > 256).any():
            raise ValueError("V4PH P class is outside [0,256]")
        return self.embedding(classes)
















