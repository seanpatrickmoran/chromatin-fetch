#!/usr/bin/env python3
"""Reconstruct one matrix or a non-temporal matrix clip stored as NumPy data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfetch.model import VAEInference

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Folder containing config.json and model.safetensors")
    parser.add_argument("--input", required=True, help="Input .npy array in [0,1]")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    matrices = np.load(args.input, allow_pickle=False).astype(np.float32, copy=False)
    vae = VAEInference.from_folder(args.model, device=args.device, dtype=args.dtype)
    result = vae.reconstruct_matrices(matrices)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        reconstruction=result["reconstruction"].cpu().numpy(),
        mu=result["mu"].float().cpu().numpy(),
        logvar=result["logvar"].float().cpu().numpy(),
    )


if __name__ == "__main__":
    main()
