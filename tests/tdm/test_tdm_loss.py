import torch

from boo_sd_scripts.tdm import tdm_loss


def test_target_equals_student_when_teacher_equals_fake_and_no_cfg():
    s = torch.randn(2, 4, 3, 3, requires_grad=True)
    c = torch.randn(2, 4, 3, 3)
    target, weighting = tdm_loss.tdm_generator_target(s, c, torch.randn_like(c), c.clone(), cfg_scale=1.0)
    assert torch.allclose(target, s.detach())
    assert weighting.shape == (2, 1, 1, 1)


def test_target_cfg_term():
    s, c, u = torch.randn(1, 2, 2, 2), torch.randn(1, 2, 2, 2), torch.randn(1, 2, 2, 2)
    target, _ = tdm_loss.tdm_generator_target(s, c, u, c.clone(), cfg_scale=3.0)
    assert torch.allclose(target, s + 2.0 * (c - u), atol=1e-6)


def test_target_and_weighting_are_detached():
    s = torch.randn(1, 2, 2, 2, requires_grad=True)
    c = torch.randn(1, 2, 2, 2, requires_grad=True)
    f = torch.randn(1, 2, 2, 2, requires_grad=True)
    target, weighting = tdm_loss.tdm_generator_target(s, c, torch.randn_like(c), f, cfg_scale=2.0)
    assert not target.requires_grad and not weighting.requires_grad


def test_weighting_is_inverse_mean_abs_error_to_cfg_teacher():
    s = torch.zeros(1, 1, 2, 2)
    c = torch.ones(1, 1, 2, 2)
    u = torch.zeros(1, 1, 2, 2)
    # teacher_cfg = c + (cfg-1)(c-u) = 1 + 1 = 2 for cfg=2; mean|s - 2| = 2
    _, weighting = tdm_loss.tdm_generator_target(s, c, u, c, cfg_scale=2.0)
    assert torch.allclose(weighting, torch.full((1, 1, 1, 1), 0.5))


def test_fake_score_loss_shape_and_grad_isolation():
    fake = torch.randn(3, 4, 2, 2, requires_grad=True)
    student = torch.randn(3, 4, 2, 2, requires_grad=True)
    loss = tdm_loss.fake_score_loss(fake, student, torch.ones(3), torch.ones(3))
    assert loss.shape == (3,)
    loss.sum().backward()
    assert fake.grad is not None
    assert student.grad is None
