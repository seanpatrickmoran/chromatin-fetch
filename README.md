# Chromatin-Fetch

Chromatin-Fetch (Cfetch) learns a frozen representation of Hi-C and Micro-C contact-map
patches and uses reference examples supplied after training to guide candidate
assignment. This repository contains the model definition, prepared-database
interface, and the three inference methods used in the manuscript.


## Installation

```bash
python -m pip install -e .
```

Install the CUDA build of PyTorch appropriate for the target system before
installing cfetch when GPU inference is required. Tests can be installed with:

```bash
python -m pip install -e '.[test]'
pytest
```

## Model folders

The 135M and 650M checkpoints are external model folders. Each must contain:

```text
vae_650m/
├── config.json
└── model.safetensors
```

The same code loads both model sizes; the architecture is defined by the model
folder's `config.json`. Pickled PyTorch checkpoints will not run. 
untime.

## Inference

The default command runs `conformal`, `graph_diffusion`, and `probe`, then
combines their outputs using a two-of-three categorical vote and a continuous
rank-fused score:

```bash
cfetch-infer \
  --model /path/to/vae_650m \
  --reference-database /path/to/loops.db \
  --reference-database /path/to/negatives.db \
  --reference-labels 'StrongPositive,Negatives' \
  --query-database /path/to/candidates.db \
  --out-dir results/loop_assignment \
  --device cuda
```

The paper-selected parameters are loaded automatically from `configs/paper/`.
Explicit command-line options override values from those files. Use
`--consensus-threshold 1`, `2`, or `3` to request union, majority, or unanimous
categorical support; the default is two-of-three.

Run one method with a numerical or semantic selector:

```bash
cfetch-infer single 1 ...                 # conformal
cfetch-infer single 2 ...                 # graph_diffusion
cfetch-infer single 3 ...                 # probe
cfetch-infer single graph_diffusion ...
```

`--config FILE` selects a different JSON preset for a single method, and
`--no-paper-config` uses the implementation defaults. The older method names
and `--seeded-database`/`--seed-labels` options remain compatibility aliases.

The semantic method names are stable and deliberately contain no encoded
hyperparameters. Selected values are recorded only in the JSON configurations.

See `docs/database_interface.md` for the prepared SQLite contract and
`docs/inference_outputs.md` for generated files.

## Repository scope

- `src/cfetch/model`: VAE architecture, preprocessing, and safe model loading.
- `src/cfetch/data`: prepared SQLite access and clip construction.
- `src/cfetch/inference`: shared inference machinery and three assignment methods.
- `configs/paper`: parameter values used for manuscript analyses.
- `examples`: minimal assignment and reconstruction entry points.

Cluster-specific launchers, parameter sweeps, database builders, checkpoint
conversion scripts, and manuscript figure-generation code are intentionally
excluded from this package.

## Citation

The manuscript describing Chromatin-Fetch is forthcoming on
[bioRxiv](https://www.biorxiv.org/).

> Moran, Sean, et al. *Reference-Guided De Novo Retrieval of 3D Chromatin Organizational Features via Representation Learning
with Chromatin-Fetch*. bioRxiv
> (2026), preprint forthcoming.

The citation metadata will be updated when the preprint is posted.