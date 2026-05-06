# DFL-GMVP: Decision-Focused Learning for Global Minimum Variance Portfolio

Re-implementation of the paper:

**"Estimating Covariance for Global Minimum Variance Portfolio: A Decision-Focused Learning Approach"**
Juchan Kim, Inwoo Tae, Yongjae Lee (UNIST)
ICAIF'25 | [arXiv](https://arxiv.org/abs/2508.10776)

## Overview

This repository contains a PyTorch implementation of the core components of the DFL framework for GMVP construction:

- **DFL_GMVPLoss**: Custom autograd function implementing the regret loss with closed-form gradient derived from the analytic GMVP solution
- **DFL_DLinear**: DLinear-based covariance prediction model that decomposes input returns into trend and seasonal components and outputs a positive semi-definite covariance matrix via Cholesky-like construction (Σ̂ = LLᵀ)

## Requirements

- Python 3.x
- PyTorch
- numpy
