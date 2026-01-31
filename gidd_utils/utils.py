import torch

@torch.no_grad()
def sample_categorical(probs, generator=None):
    # return torch.distributions.Categorical(probs=probs).sample()
    uniform = torch.rand(probs.shape[:-1], dtype=probs.dtype, device=probs.device, generator=generator).unsqueeze(-1)
    cumprobs = probs.cumsum(-1)
    cumprobs[..., -1] = 1 + 1e-4
    samples = torch.searchsorted(cumprobs, uniform, right=True).squeeze(-1)
    return samples

