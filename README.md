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
├── environment.yml                 # conda environment specification
├── Kluane_Lake_West_mag_res.grd    # public datasets used in the paper
├── data_generate.py                # synthetic data generation
├── simulate_flightline_real.py.py  # preprocessing of public datasets
├── dataset.py                      # synthetic dataset loading / sampling utilities
├── dataset_real.py                 # public dataset loading
├── model.py                        # CDK model (encoder + kernel generator + solver)
├── train.py                        # training and prediction on synthetic datasets (self-supervised)
├── train_real.py                   # Training and prediction on public datasets
├── util.py                         # util package
└── visualize.py                    # visualization
```

## Requirements
We recommend using **conda** to create a reproducible environment.

### Create Environment
```bash
conda env create -f environment.yml
conda activate cdk
```


## Quick Start

### 1. Synthetic Dataset Experiments
#### 1) Generate Synthetic Data
Run the script below to generate synthetic anomaly fields and sparse flight-line observations.

```bash
python data_generate.py
```
Running the script directly generates the dataset used in the paper, which is saved to the magnetic_dataset_zscore directory by default. Additionally, the script supports customizable arguments, such as output directory, grid size, line spacing, and point spacing, e.g.

```bash
python data_generate.py --out_dir ./data --grid_size 128 --alpha 15 --beta 4
```

Please check `data_generate.py` for the available arguments.

#### 2) Train (Self-supervised)
Train the model using only the observed points (no dense ground-truth labels required for training) and output the interpolation results.
Change the --data_dir argument to switch datasets., you can use something like:

```bash
python train.py --data_dir "magnetic_dataset_zscore/exp_A_alphaFixed_beta_dense"
python train.py --data_dir "magnetic_dataset_zscore/exp_A_alphaFixed_beta_medium"
python train.py --data_dir "magnetic_dataset_zscore/exp_A_alphaFixed_beta_sparse"
python train.py --data_dir "magnetic_dataset_zscore/exp_B_betaFixed_alpha_dense"
python train.py --data_dir "magnetic_dataset_zscore/exp_B_betaFixed_alpha_sparse"
```
### 2. Public Dataset Experiments

#### 1) Data Preprocessing
Before training, data preprocessing is required for Kluane_Lake_West_mag_res.grd.
```bash
python simulate_flightline_real.py
```

#### 2) Train
Change the --data_dir argument to switch datasets., you can use something like:

```bash
python train_real.py --data_dir "real_dataset_zscore/real_dense"
python train_real.py --data_dir "real_dataset_zscore/real_medium"
python train_real.py --data_dir "real_dataset_zscore/real_sparse"
```

## Hardware Requirements
Our experiments were conducted on the following hardware configuration:
- CPU: AMD Ryzen 9 7940HX  
- GPU: NVIDIA RTX 4060 Laptop (8GB)  
- RAM: 16GB DDR5  

## Notes on Reproducibility
- Set random seeds in your scripts for deterministic behavior if needed.
- Make sure the dataset split strategy used in training matches the paper setting.
- https://open.canada.ca/data/en/dataset/06d87b8d-2149-4b9d-9737-e5531fea1d45

## License
This project is released under the license in the LICENSE file.

## Contact
For questions or issues, please open a GitHub Issue.


