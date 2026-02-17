# Conditional-Deep-Kriging

Conditional Deep Kriging (CDK) for geomagnetic map reconstruction from sparse flight-line observations.  
This repository provides the code to generate synthetic data, train the CDK model in a self-supervised manner, and visualize reconstruction results.

## Highlights
- Context-conditioned composite covariance kernel (parametric main kernel + low-rank residual kernel).
- Differentiable kriging solver enables end-to-end training using only observed points.
- Designed for sparse and anisotropic sampling patterns such as flight-line surveys.

## Repository Structure

```text
.
├── LICENSE
├── README.md
├── environment.yml        # conda environment specification
├── data_generate.py       # synthetic data generation
├── dataset.py             # dataset loading / sampling utilities
├── model.py               # CDK model (encoder + kernel generator + solver)
├── train.py               # training entry (self-supervised)
└── visualize.py           # visualization / qualitative comparison
```

## Requirements
We recommend using **conda** to create a reproducible environment.

### Create Environment
```bash
conda env create -f environment.yml
conda activate cdk
```

If your environment name in `environment.yml` is not `cdk`, replace it with the actual name.

## Quick Start

### 1) Generate Synthetic Data
Run the script below to generate synthetic anomaly fields and sparse flight-line observations.

```bash
python data_generate.py
```

If your script supports arguments, typical ones might include output directory, grid size, line spacing, and point spacing, e.g.

```bash
python data_generate.py --out_dir ./data --grid_size 128 --alpha 15 --beta 4
```

Please check `data_generate.py` for the available arguments.

### 2) Train (Self-supervised)
Train the model using only the observed points (no dense ground-truth labels required for training).

```bash
python train.py
```

If your training script supports config-like arguments, you can use something like:

```bash
python train.py --data_dir ./data --save_dir ./checkpoints --epochs 200
```

Please check `train.py` for the exact options.


## Notes on Reproducibility
- Set random seeds in your scripts for deterministic behavior if needed.
- Make sure the dataset split strategy used in training matches the paper setting.

## License
This project is released under the license in the LICENSE file.

## Contact
For questions or issues, please open a GitHub Issue.


