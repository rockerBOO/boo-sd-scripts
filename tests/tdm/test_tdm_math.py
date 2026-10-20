import torch

from boo_sd_scripts.tdm import tdm_math as m


def _b(v):
    return torch.tensor(v, dtype=torch.float32).view(-1, 1, 1, 1)


def test_x0_eps_from_v_roundtrip():
    torch.manual_seed(0)
    x0, eps = torch.randn(3, 4, 2, 2), torch.randn(3, 4, 2, 2)
    s = _b([0.2, 0.5, 0.9])
    x_t = (1 - s) * x0 + s * eps
    v = eps - x0
    assert torch.allclose(m.x0_from_v(x_t, v, s), x0, atol=1e-5)
    assert torch.allclose(m.eps_from_v(x_t, v, s), eps, atol=1e-5)


def test_ddim_step_lands_on_noise_level():
    x0, eps = torch.randn(2, 4, 2, 2), torch.randn(2, 4, 2, 2)
    assert torch.allclose(m.ddim_step(x0, eps, _b([0.0, 1.0])), torch.stack([x0[0], eps[1]]))


def test_renoise_same_sigma_is_identity():
    x = torch.randn(2, 4, 2, 2)
    out, beta = m.renoise(x, _b([0.3, 0.6]), _b([0.3, 0.6]), torch.randn_like(x))
    assert torch.allclose(out, x, atol=1e-6)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6)


def test_renoise_variance_matches_target_sigma():
    sa, sb = _b([0.3]), _b([0.7])
    ratio = (1 - sb) / (1 - sa)
    _, beta = m.renoise(torch.zeros(1, 1, 1, 1), sa, sb, torch.zeros(1, 1, 1, 1))
    assert torch.allclose(ratio**2 * sa**2 + beta**2, sb**2, atol=1e-6)


def test_importance_weight_is_one_when_noise_is_unchanged():
    eps, z = torch.randn(2, 4, 2, 2), torch.randn(2, 4, 2, 2)
    # sigma_tau == sigma_mid: mixed noise reduces to eps, so the weight is exp(-0.5*(mean(eps^2)-mean(z^2)))
    w = m.importance_weight(eps, z, _b([0.4, 0.4]), _b([0.4, 0.4]))
    expected = torch.exp(-0.5 * (eps.pow(2).flatten(1).mean(1) - z.pow(2).flatten(1).mean(1)))
    assert w.shape == (2,)
    assert torch.allclose(w, expected, atol=1e-5)


def test_snr_weight_is_capped_at_five():
    w = m.snr_weight(_b([0.02, 0.5, 0.9]))
    assert w.shape == (3,)
    assert w[0].item() == 5.0
    assert abs(w[1].item() - 1.0) < 1e-6
    assert w[2].item() < 1.0


def test_sample_tdm_sigmas_ordering_and_ranges():
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    gen = torch.Generator().manual_seed(0)
    d = m.sample_tdm_sigmas(sigmas, 256, generator=gen)
    idx = d.index
    assert idx.min() >= 0 and idx.max() <= 3
    g, mid, tau = d.sigma_g.flatten(), d.sigma_mid.flatten(), d.sigma_tau.flatten()
    nxt = sigmas[idx + 1]
    assert torch.equal(g, sigmas[idx])
    assert (mid >= nxt - 1e-6).all() and (mid <= g).all()
    assert (tau >= mid - 1e-6).all() and (tau <= g).all()
    assert (tau >= m.SIGMA_MIN - 1e-6).all()
    assert d.sigma_g.shape == (256, 1, 1, 1)


def test_renoise_and_mixed_noise_with_increasing_sigma():
    torch.manual_seed(0)
    sa, sb = _b([0.3, 0.2]), _b([0.7, 0.9])
    x0, eps, noise = torch.randn(2, 4, 2, 2), torch.randn(2, 4, 2, 2), torch.randn(2, 4, 2, 2)
    x_a = (1 - sa) * x0 + sa * eps
    out, beta = m.renoise(x_a, sa, sb, noise)
    ratio = (1 - sb) / (1 - sa)
    assert (beta > 0).all()
    assert torch.allclose(ratio**2 * sa**2 + beta**2, sb**2, atol=1e-6)
    # x_tau built directly from x0 and the effective noise equals the renoised sample
    mixed = m.mixed_noise(eps, noise, sa, sb)
    assert torch.allclose(out, (1 - sb) * x0 + sb * mixed, atol=1e-5)
