"""Selective TorchAO NVFP4 conversion for the S2A diffusion transformer."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


_CORE_LINEAR_SUFFIXES = (
    ".attention.wqkv",
    ".attention.wo",
    ".feed_forward.w1",
    ".feed_forward.w2",
    ".feed_forward.w3",
)


def _validate_nvfp4_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("--nvfp4 requires an NVIDIA CUDA GPU.")

    major, minor = torch.cuda.get_device_capability(resolved)
    if major < 10:
        name = torch.cuda.get_device_name(resolved)
        raise RuntimeError(
            "--nvfp4 requires NVIDIA Blackwell (compute capability 10.0 or newer); "
            f"detected {name} with capability {major}.{minor}."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("--nvfp4 requires BF16 support for non-quantized DiT operations.")
    return resolved


def enable_dit_nvfp4(
    estimator: nn.Module,
    device: torch.device | str,
) -> Tuple[str, ...]:
    """Pack the attention/FFN linears and enable dynamic NVFP4 activations."""

    resolved = _validate_nvfp4_device(device)
    try:
        from torchao.prototype.mx_formats.inference_workflow import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError(
            "--nvfp4 requires TorchAO with NVFP4 support. Install the optional "
            "dependencies from requirements-nvfp4.txt."
        ) from exc

    selected = tuple(
        name
        for name, module in estimator.named_modules()
        if isinstance(module, nn.Linear)
        and any(name.endswith(suffix) for suffix in _CORE_LINEAR_SUFFIXES)
    )
    if not selected:
        raise RuntimeError("No eligible S2A DiT attention/FFN linear layers were found.")

    estimator.to(device=resolved, dtype=torch.bfloat16)
    selected_set = frozenset(selected)
    config = NVFP4DynamicActivationNVFP4WeightConfig(
        use_dynamic_per_tensor_scale=True,
        use_triton_kernel=True,
    )
    quantize_(
        estimator,
        config=config,
        filter_fn=lambda module, fqn: isinstance(module, nn.Linear) and fqn in selected_set,
    )
    return selected
