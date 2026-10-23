# Gradient diagnostics for LoRA training (ported from musubi-tuner's trainer_base): total/mean/max grad norm, and optional
# per-module and per-block norms. Meant to run right after backward, before clipping.

import argparse
import re

import torch


def add_grad_metrics_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log_grad_metrics",
        action="store_true",
        help="log gradient norm metrics (grad/norm, grad/mean_norm, grad/max, pre-clipping) to the tracker."
        " Adds a small per-step GPU sync overhead",
    )
    parser.add_argument(
        "--log_grad_metrics_per_module",
        action="store_true",
        help="also log per-module gradient norms (grad_module/<name>, pre-clip). Requires --log_grad_metrics."
        " One entry per LoRA module (252 for MOSS-Image's default targets)",
    )
    parser.add_argument(
        "--log_grad_metrics_block_regex",
        type=_block_regex_type,
        default=None,
        help="regex with exactly one capture group; additionally logs grad_block/<id>, aggregating module norms by the"
        " captured id. Requires --log_grad_metrics. For MOSS-Image use '_layers_(\\d+)_'",
    )


def _block_regex_type(value: str) -> re.Pattern:
    try:
        pattern = re.compile(value)
    except re.error as e:
        raise argparse.ArgumentTypeError(f"invalid regex for --log_grad_metrics_block_regex: {e}")
    if pattern.groups != 1:
        raise argparse.ArgumentTypeError(
            f"--log_grad_metrics_block_regex must have exactly one capture group, got {pattern.groups}: {value!r}"
        )
    return pattern


def validate_grad_metrics_args(args) -> None:
    if isinstance(args.log_grad_metrics_block_regex, str):  # from a TOML config, which bypasses argparse's `type=`
        args.log_grad_metrics_block_regex = _block_regex_type(args.log_grad_metrics_block_regex)
    if args.log_grad_metrics_per_module or args.log_grad_metrics_block_regex is not None:
        assert args.log_grad_metrics, "--log_grad_metrics_per_module / --log_grad_metrics_block_regex require --log_grad_metrics"


def collect_grad_metrics(params) -> dict:
    """grad/norm (total L2), grad/mean_norm (mean of per-parameter L2 norms) and grad/max (max abs element)."""
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return {}
    per_norm = torch.stack([g.norm() for g in grads])
    max_grad = torch.stack([g.abs().max() for g in grads]).max()
    return {"grad/norm": per_norm.norm().item(), "grad/mean_norm": per_norm.mean().item(), "grad/max": max_grad.item()}


def collect_grad_metrics_by_module(named_parameters, block_regex: re.Pattern | None = None, per_module: bool = True) -> dict:
    """Per-module (grad_module/<name>) and per-block (grad_block/<id>) L2 grad norms.

    Modules are the part of each parameter name before the first '.', which is how LoRA networks register them.
    """
    items = [(name, p.grad.detach()) for name, p in named_parameters if p.grad is not None]
    if not items:
        return {}

    module_names = [name.split(".", 1)[0] for name, _ in items]
    unique_modules = sorted(set(module_names))
    module_index = {name: i for i, name in enumerate(unique_modules)}

    sq_norms = torch.stack([grad.norm() for _, grad in items]) ** 2  # stays on device until the single tolist()
    module_sq = torch.zeros(len(unique_modules), dtype=sq_norms.dtype, device=sq_norms.device)
    module_sq.scatter_add_(0, torch.tensor([module_index[m] for m in module_names], device=sq_norms.device), sq_norms)
    module_sq_norms = dict(zip(unique_modules, module_sq.tolist()))

    logs: dict[str, float] = {}
    if per_module:
        logs.update({f"grad_module/{name}": sq**0.5 for name, sq in module_sq_norms.items()})

    if block_regex is not None:
        block_sq_norms: dict[str, float] = {}
        for module_name, sq in module_sq_norms.items():
            m = block_regex.search(module_name)
            if m:
                block_sq_norms[m.group(1)] = block_sq_norms.get(m.group(1), 0.0) + sq
        keys = _zero_pad_block_ids(block_sq_norms.keys())
        logs.update({f"grad_block/{keys[block_id]}": sq**0.5 for block_id, sq in block_sq_norms.items()})

    return logs


def _zero_pad_block_ids(block_ids) -> dict:
    """Zero-pad numeric ids so wandb's lexical key sort matches numeric order."""
    block_ids = list(block_ids)
    if not block_ids or not all(bid.isdigit() for bid in block_ids):
        return {bid: bid for bid in block_ids}
    width = max(len(bid) for bid in block_ids)
    return {bid: bid.zfill(width) for bid in block_ids}
