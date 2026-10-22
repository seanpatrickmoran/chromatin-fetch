from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .modeling import VAE
from .preprocess import (
    bcthw_model_to_btchw_01,
    btchw_01_to_bcthw_model,
    fp32_matrices_to_btchw,
)


REQUIRED_CONFIG_KEYS = {
    "dim",
    "dec_dim",
    "z_dim",
    "dim_mult",
    "num_res_blocks",
}


def _normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Normalize DDP / torch.compile wrapper prefixes without changing tensor names."""
    out: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        parts = [part for part in str(key).split(".") if part not in {"module", "_orig_mod"}]
        out[".".join(parts)] = value
    return out


def load_safetensors(path: str | Path) -> OrderedDict[str, torch.Tensor]:
    """Load a safetensors state dictionary onto CPU."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")
    if path.suffix != ".safetensors":
        raise ValueError(f"CFetch release models must use .safetensors weights, got: {path.name}")

    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    return _normalize_state_dict_keys(state)


def read_model_config(folder: str | Path) -> dict[str, Any]:
    """Read and validate the architecture configuration in a model folder."""
    folder = Path(folder)
    config_path = folder / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {folder}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    missing = sorted(REQUIRED_CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(f"{config_path} is missing required fields: {missing}")
    if "temporal_downsample" not in config and "temperal_downsample" not in config:
        raise ValueError(f"{config_path} must define temporal_downsample")
    return config


def validate_model_folder(
    folder: str | Path,
    *,
    weights_name: str = "model.safetensors",
) -> dict[str, Path]:
    """Validate the files needed for inference without loading weight tensors."""
    folder = Path(folder)
    read_model_config(folder)
    weights_path = folder / weights_name
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing {weights_name} in {folder}")
    if weights_path.suffix != ".safetensors":
        raise ValueError("CFetch release weights must use the .safetensors format")
    return {"folder": folder, "config": folder / "config.json", "weights": weights_path}


def _dtype_from_name(name: str | torch.dtype | None) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    if name is None:
        return torch.float32
    normalized = str(name).lower().replace("torch.", "")
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    if normalized not in table:
        raise ValueError(f"Unsupported dtype {name!r}. Use float32, bfloat16, or float16.")
    return table[normalized]


class VAEInference:
    """Inference wrapper for CFetch reconstruction and latent extraction."""

    def __init__(
        self,
        model: VAE,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=self.dtype).eval()

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = None,
        weights_name: str | None = None,
        strict: bool = True,
        compile_model: bool = False,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> "VAEInference":
        folder = Path(folder)
        selected_weights = weights_name or "model.safetensors"
        validate_model_folder(folder, weights_name=selected_weights)
        config = read_model_config(folder)
        torch_dtype = _dtype_from_name(dtype or config.get("torch_dtype", "float32"))
        temporal_downsample = config.get(
            "temporal_downsample",
            config.get("temperal_downsample"),
        )

        model = VAE(
            dim=int(config["dim"]),
            dec_dim=int(config["dec_dim"]),
            z_dim=int(config["z_dim"]),
            dim_mult=tuple(int(v) for v in config["dim_mult"]),
            num_res_blocks=int(config["num_res_blocks"]),
            attn_scales=tuple(config.get("attn_scales", [])),
            temperal_downsample=tuple(bool(v) for v in temporal_downsample),
            dropout=float(config.get("dropout", 0.0)),
        )
        state = load_safetensors(folder / selected_weights)
        model.load_state_dict(state, strict=strict)

        if compile_model:
            model = torch.compile(model, **(compile_kwargs or {}))
        return cls(model, device=device, dtype=torch_dtype)

    @torch.no_grad()
    def encode_matrices(self, x01, *, return_logvar: bool = False, clamp: bool = False):
        """Encode FP32 [0,1] matrices into latent means and optional log variances."""
        x_btchw = fp32_matrices_to_btchw(x01, clamp=clamp)
        x_model = btchw_01_to_bcthw_model(
            x_btchw,
            device=self.device,
            dtype=self.dtype,
        )
        mu, logvar = self.model.encode(x_model, scale=(0.0, 1.0))
        return (mu, logvar) if return_logvar else mu

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor):
        """Decode latents to FP32 [0,1] matrices in [B,T,C,H,W]."""
        z = z.to(device=self.device, dtype=self.dtype, non_blocking=True)
        return bcthw_model_to_btchw_01(self.model.decode(z, scale=(0.0, 1.0)))

    @torch.no_grad()
    def reconstruct_matrices(self, x01, *, sample_latent: bool = False, clamp: bool = False):
        """Return reconstruction and latent statistics for FP32 [0,1] matrices."""
        x_btchw = fp32_matrices_to_btchw(x01, clamp=clamp)
        x_model = btchw_01_to_bcthw_model(
            x_btchw,
            device=self.device,
            dtype=self.dtype,
        )
        reconstruction, mu, logvar = self.model(
            x_model,
            scale=(0.0, 1.0),
            sample_latent=sample_latent,
        )
        return {
            "reconstruction": bcthw_model_to_btchw_01(reconstruction),
            "mu": mu,
            "logvar": logvar,
        }


__all__ = [
    "VAEInference",
    "load_safetensors",
    "read_model_config",
    "validate_model_folder",
]
