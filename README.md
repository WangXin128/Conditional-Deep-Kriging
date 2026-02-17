# Conditional-Deep-Kriging

Conditional Deep Kriging (CDK) for geomagnetic map reconstruction from sparse flight-line observations.  
This repository provides the code to generate synthetic data, train the CDK model in a self-supervised manner, and visualize reconstruction results.

## Highlights
- Context-conditioned composite covariance kernel (parametric main kernel + low-rank residual kernel).
- Differentiable kriging solver enables end-to-end training using only observed points.
- Designed for sparse and anisotropic sampling patterns such as flight-line surveys.
## Requirements
We recommend using **conda** to create a reproducible environment.

### Create Environment
```bash
conda env create -f environment.yml
conda activate cdk

### Quick Start
1) Generate Synthetic Data

Run the script below to generate synthetic anomaly fields and sparse flight-line observations.

