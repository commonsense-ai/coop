import torch

from coop.robust import clip_norm, cosine_gate, geometric_median, trimmed_mean

g = torch.Generator().manual_seed(0)
TARGET = torch.randn(64, generator=g)
HONEST = [TARGET + 0.01 * torch.randn(64, generator=g) for _ in range(7)]
ADVERSARIAL = [TARGET + 100.0 * torch.randn(64, generator=g) for _ in range(3)]
HONEST_MEAN = torch.stack(HONEST).mean(0)


def test_trimmed_mean_resists_outliers():
    est = trimmed_mean(HONEST + ADVERSARIAL, trim_frac=0.3)
    assert (est - HONEST_MEAN).norm() < 0.1


def test_trimmed_mean_zero_trim_is_mean():
    est = trimmed_mean(HONEST, trim_frac=0.0)
    assert torch.allclose(est, HONEST_MEAN)


def test_geometric_median_resists_outliers():
    est = geometric_median(HONEST + ADVERSARIAL)
    assert (est - HONEST_MEAN).norm() < 0.1


def test_cosine_gate_rejects_anticorrelated():
    v = torch.randn(32, generator=g)
    tensors = [v, v + 0.01 * torch.randn(32, generator=g), -v]
    assert cosine_gate(tensors, ref=v, min_cos=0.5) == [True, True, False]


def test_cosine_gate_zero_vector():
    v = torch.randn(8, generator=g)
    assert cosine_gate([torch.zeros(8)], ref=v, min_cos=0.0) == [True]
    assert cosine_gate([torch.zeros(8)], ref=v, min_cos=0.1) == [False]


def test_clip_norm():
    v = torch.ones(16) * 10.0  # norm 40
    clipped = clip_norm(v, max_norm=3.0)
    assert torch.isclose(clipped.norm(), torch.tensor(3.0))
    small = torch.ones(4)  # norm 2
    assert torch.equal(clip_norm(small, max_norm=3.0), small)
