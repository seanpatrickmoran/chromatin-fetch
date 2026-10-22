from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def fp32_matrices_to_btchw(x: ArrayLike, *, clamp: bool = False) -> torch.Tensor:
    """
    Placeholder intake for FP32 matrices scaled to [0, 1].

    Accepted input shapes:
      [H, W]          -> [1, 1, 1, H, W]
      [T, H, W]       -> [1, T, 1, H, W]
      [B, T, H, W]    -> [B, T, 1, H, W]
      [B, T, 1, H, W] -> [B, T, 1, H, W]

    Returns a contiguous torch.float32 tensor in public layout [B, T, C, H, W].
    The current model path expects one input channel; spatial patching expands it
    internally to the four channels consumed by the encoder.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not torch.is_tensor(x):
        raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)!r}.")

    x = x.detach().to(dtype=torch.float32).contiguous()

    if x.ndim == 2:
        x = x[None, None, None, :, :]
    elif x.ndim == 3:
        x = x[None, :, None, :, :]
    elif x.ndim == 4:
        x = x[:, :, None, :, :]
    elif x.ndim == 5:
        pass
    else:
        raise ValueError(f"Expected 2D, 3D, 4D, or 5D input; got shape {tuple(x.shape)}.")

    if x.shape[2] != 1:
        raise ValueError(
            f"Expected a single matrix channel at dimension C in [B,T,C,H,W]; got C={x.shape[2]}."
        )

    if not torch.isfinite(x).all():
        raise ValueError("Input contains NaN or Inf.")

    if clamp:
        x = x.clamp(0.0, 1.0)
    else:
        min_value = float(x.min().item())
        max_value = float(x.max().item())
        if min_value < 0.0 or max_value > 1.0:
            raise ValueError(
                f"Expected values in [0,1]; got min={min_value:.6g}, max={max_value:.6g}. "
                "Pass clamp=True only if clipping is intentional."
            )

    _, _, _, height, width = x.shape
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"Expected H and W divisible by 16; got H={height}, W={width}.")

    return x.contiguous()


def btchw_01_to_bcthw_model(x_btchw: torch.Tensor, *, device=None, dtype=None) -> torch.Tensor:
    """Convert public [B,T,C,H,W] in [0,1] to model [B,C,T,H,W] in [-1,1]."""
    if x_btchw.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got shape {tuple(x_btchw.shape)}.")
    x = x_btchw.to(device=device, dtype=(dtype or torch.float32), non_blocking=True)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.mul(2.0).sub(1.0)


def bcthw_model_to_btchw_01(x_bcthw: torch.Tensor) -> torch.Tensor:
    """Convert model [B,C,T,H,W] in [-1,1] back to public [B,T,C,H,W] in [0,1]."""
    if x_bcthw.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got shape {tuple(x_bcthw.shape)}.")
    x = x_bcthw.float().mul(0.5).add(0.5).clamp(0.0, 1.0)
    return x.permute(0, 2, 1, 3, 4).contiguous()
