# Hierarchical Bayesian Models for Medical Imaging and Radiation Oncology

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Stan](https://img.shields.io/badge/Stan-2.34+-red.svg)](https://mc-stan.org)
[![R](https://img.shields.io/badge/R-4.3+-blue.svg)](https://r-project.org)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

**Companion repository for the manuscript:**
*HBM — The Unifying Language for Scale, Resolution, View, Task, and Mode Integration*
Renjie He — Fuller Lab, Department of Radiation Oncology, MD Anderson Cancer Center

---

## Overview

This repository provides working implementations of Hierarchical Bayesian Models (HBMs) for medical imaging and radiation oncology, covering the full spectrum from conjugate closed-form models through MCMC (Stan/NUTS) to scalable approximate methods (R-INLA, TMB, NumPyro). All code cross-references specific sections of the companion manuscript.

### Repository Structure

| Directory | Language/Tool | Contents |
|-----------|--------------|----------|
| `stan/` | Stan | HBM models: scanner harmonization, site-level pooling, dose-response, spatial GMRF |
| `R/` | R (brms, INLA, mstate) | R-INLA spatial models, brms multilevel GLMMs, ComBat comparison, PSIS-LOO diagnostics |
| `python/` | Python (NumPyro, JAX) | NumPyro HBMs, JAX-accelerated NUTS, variational inference, posterior diagnostics |
| `julia/` | Julia (Turing.jl) | Turing.jl hierarchical models with non-centered parameterization |
| `examples/` | Mixed | Worked examples from manuscript: partial pooling, shrinkage visualization, multi-site harmonization |
| `tests/` | Mixed | Verification scripts |
| `docs/` | — | Notation reference, model gallery |

---

## Quick Start

### Stan: Fit a Scanner Harmonization HBM
```bash
cd stan/
pip install cmdstanpy
python fit_scanner_hbm.py
```

### R-INLA: Spatial Dose-Response Model
```r
source("R/spatial_dose_response_inla.R")
```

### NumPyro: GPU-Accelerated Hierarchical Model
```bash
cd python/
pip install -r requirements.txt
python numpyro_site_pooling.py
```

---

## Key References

- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge.
- Gelman, A. et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall.
- Carpenter, B. et al. (2017). Stan: A probabilistic programming language. *J. Stat. Softw.*, 76(1):1–32.
- Rue, H., Martino, S., & Chopin, N. (2009). Approximate Bayesian inference for latent Gaussian models. *JRSS-B*, 71(2):319–392.
- Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields. *JRSS-B*, 73(4):423–498.
- Kristensen, K. et al. (2016). TMB: Automatic differentiation and Laplace approximation. *J. Stat. Softw.*, 70(5):1–21.
- Bürkner, P.-C. (2017). brms: An R package for Bayesian multilevel models. *J. Stat. Softw.*, 80(1):1–28.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Stat. Comput.*, 27(5):1413–1432.
- Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler. *JMLR*, 15(1):1593–1623.
- Phan, D., Pradhan, N., & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. *arXiv:1912.11554*.

---

## License

MIT License. See [LICENSE](LICENSE).
