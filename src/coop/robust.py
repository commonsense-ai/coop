"""Robust aggregation primitives. All operate coordinate-wise on same-shape tensors."""

import torch


def trimmed_mean(tensors: list[torch.Tensor], trim_frac: float) -> torch.Tensor:
    x = torch.stack([t.float() for t in tensors])
    n = x.shape[0]
    k = int(n * trim_frac)
    if k == 0:
        return x.mean(dim=0)
    assert 2 * k < n, f"trim_frac={trim_frac} leaves nothing from {n} tensors"
    x, _ = x.sort(dim=0)
    return x[k : n - k].mean(dim=0)


def geometric_median(
    tensors: list[torch.Tensor], eps: float = 1e-6, iters: int = 100
) -> torch.Tensor:
    """Weiszfeld iteration; breakdown point 0.5 vs the mean's 0."""
    x = torch.stack([t.float() for t in tensors])
    mu = x.mean(dim=0)
    for _ in range(iters):
        d = (x - mu).flatten(1).norm(dim=1).clamp_min(eps)
        w = 1.0 / d
        new = (x * w.view(-1, *([1] * (x.dim() - 1)))).sum(dim=0) / w.sum()
        if (new - mu).norm() < eps:
            return new
        mu = new
    return mu


def cosine_gate(tensors: list[torch.Tensor], ref: torch.Tensor, min_cos: float) -> list[bool]:
    r = ref.flatten().float()
    rn = r.norm()
    keep = []
    for t in tensors:
        v = t.flatten().float()
        denom = v.norm() * rn
        cos = (v @ r) / denom if denom > 0 else torch.tensor(0.0)
        keep.append(bool(cos >= min_cos))
    return keep


def clip_norm(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    n = tensor.float().norm()
    if n <= max_norm:
        return tensor
    return tensor * (max_norm / n)
