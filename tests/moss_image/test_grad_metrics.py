import re

import pytest
import torch

from boo_sd_scripts.moss_image import grad_metrics


def _params():
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(2))
    c = torch.nn.Parameter(torch.zeros(2))
    a.grad = torch.tensor([3.0, 0.0])
    b.grad = torch.tensor([0.0, 4.0])
    c.grad = None
    return [("lora_unet_layers_2_q.lora_down.weight", a), ("lora_unet_layers_2_q.lora_up.weight", b), ("x.w", c)]


def test_collect_grad_metrics():
    logs = grad_metrics.collect_grad_metrics([p for _, p in _params()])
    assert logs["grad/norm"] == pytest.approx(5.0)
    assert logs["grad/mean_norm"] == pytest.approx(3.5)
    assert logs["grad/max"] == pytest.approx(4.0)


def test_collect_grad_metrics_empty():
    assert grad_metrics.collect_grad_metrics([torch.nn.Parameter(torch.zeros(1))]) == {}


def test_by_module_and_block():
    named = _params() + [("lora_unet_layers_10_q.lora_down.weight", _with_grad(12.0))]
    logs = grad_metrics.collect_grad_metrics_by_module(named, re.compile(r"_layers_(\d+)_"))
    assert logs["grad_module/lora_unet_layers_2_q"] == pytest.approx(5.0)
    assert logs["grad_block/02"] == pytest.approx(5.0)  # zero-padded to the widest id
    assert logs["grad_block/10"] == pytest.approx(12.0)


def test_block_only_omits_module_entries():
    logs = grad_metrics.collect_grad_metrics_by_module(_params(), re.compile(r"_layers_(\d+)_"), per_module=False)
    assert list(logs) == ["grad_block/2"]


def _with_grad(value):
    p = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.tensor([value])
    return p


def test_block_regex_needs_one_group():
    with pytest.raises(Exception):
        grad_metrics._block_regex_type("_layers_\\d+_")
