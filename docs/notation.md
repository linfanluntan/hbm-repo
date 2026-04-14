# HBM Notation Reference

## Core HBM Structure

| Symbol | Meaning |
|--------|---------|
| y_ij | Observation i in group j |
| θ_j | Group-level parameter for group j |
| φ | Population-level hyperparameter |
| p(y \| θ) | Likelihood (data model) |
| p(θ \| φ) | Group-level prior (random effects) |
| p(φ) | Hyperprior |
| p(θ, φ \| y) | Joint posterior |
| p(y^new \| y) | Posterior predictive distribution |

## Partial Pooling

| Symbol | Meaning |
|--------|---------|
| λ_j | Shrinkage factor for group j |
| λ_j = (n_j/σ²) / (n_j/σ² + 1/τ²) | Shrinkage toward population mean |
| σ² | Within-group variance |
| τ² | Between-group variance |
| μ_0 | Population (global) mean |
| ICC = τ²/(τ²+σ²) | Intraclass correlation coefficient |

## Conjugate Families

| Likelihood | Prior | Posterior | Key Parameters |
|-----------|-------|-----------|----------------|
| Normal(μ, σ²) | Normal(μ₀, τ²) | Normal | Mean, variance |
| Binomial(n, θ) | Beta(α, β) | Beta(α+y, β+n-y) | Success probability |
| Poisson(λ) | Gamma(a, b) | Gamma(a+Σy, b+n) | Rate parameter |
| Multinomial(n, π) | Dirichlet(α) | Dirichlet(α+y) | Category probabilities |
| Normal(μ, Σ) | Normal-Wishart | Normal-Wishart | Mean vector, covariance |

## Computational Methods

| Symbol | Meaning |
|--------|---------|
| NUTS | No-U-Turn Sampler (Hoffman & Gelman, 2014) |
| HMC | Hamiltonian Monte Carlo (Neal, 2011) |
| INLA | Integrated Nested Laplace Approximation (Rue et al., 2009) |
| SPDE | Stochastic Partial Differential Equation (Lindgren et al., 2011) |
| GMRF | Gaussian Markov Random Field (Rue & Held, 2005) |
| TMB | Template Model Builder (Kristensen et al., 2016) |
| VI | Variational Inference (Blei et al., 2017) |
| ADVI | Automatic Differentiation VI (Kucukelbir et al., 2017) |
| PSIS-LOO | Pareto-Smoothed IS Leave-One-Out (Vehtari et al., 2017) |
| R̂ | Split-chain convergence diagnostic (Vehtari et al., 2021) |

## Multi-X Framework

| Perspective | HBM Role | Key Structure |
|------------|----------|---------------|
| Multiscale | Nested levels (voxel → region → patient → population) | Hierarchical priors |
| Multiresolution | Fine ↔ coarse detail via conditional priors | SPDE mesh refinement |
| Multiview | Multiple imaging modalities sharing latent space | Shared latent factors |
| Multitask | Multiple clinical endpoints from shared biology | Multi-output regression |
| Multiway | Tensor decomposition of (voxel × patient × modality × time) | Tucker/PARAFAC priors |

## Spatial Models

| Symbol | Meaning |
|--------|---------|
| w(s) | Spatial random field at location s |
| Q | Precision matrix (sparse for GMRF) |
| κ | Spatial range parameter |
| σ_w | Marginal standard deviation of spatial field |
| ν | Matérn smoothness parameter |
| (κ² - Δ)^(α/2) w = W | SPDE representation (Δ = Laplacian, W = white noise) |
