# DFL-GMVP: Decision-Focused Learning for Global Minimum Variance Portfolio

Re-implementation of the paper:

**"Estimating Covariance for Global Minimum Variance Portfolio: A Decision-Focused Learning Approach"**
Juchan Kim, Inwoo Tae, Yongjae Lee (UNIST)
ICAIF'25 | [arXiv](https://arxiv.org/abs/2508.10776)

## Overview

This project re-implements the Decision-Focused Learning (DFL) framework for covariance estimation in Global Minimum Variance Portfolio (GMVP) construction. Unlike conventional Prediction-Focused Learning (PFL), which minimizes MSE of covariance predictions, DFL directly optimizes portfolio variance as the loss function, resulting in better out-of-sample portfolio performance.

## Method

- **Model**: DLinear backbone decomposes input returns into trend and seasonal components, then predicts a lower-triangular matrix L to construct Σ̂ = LLᵀ
- **Loss**: Regret loss = w*(Σ̂)ᵀ Σ_true w*(Σ̂) − w*(Σ_true)ᵀ Σ_true w*(Σ_true)
- **Gradient**: Custom autograd function with closed-form gradient derived from the analytic GMVP solution

## Data

- 49 Industry Portfolios (daily returns) from [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

## Requirements

- Python 3.x
- PyTorch
- pandas
- pandas_datareader
- numpy
- matplotlib

## Usage

```bash
pip install torch pandas pandas_datareader numpy matplotlib
jupyter notebook dfl_gmvp.ipynb
```
