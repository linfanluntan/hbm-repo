# spatial_dose_response_inla.R
# Spatial dose-response model using R-INLA with SPDE/GMRF
# Manuscript: Part 1 Section 11, Part 2 Section 10.2, Section 10.6 archetype A
#
# Model: Voxel-level binary toxicity outcome with spatial random field
#   y_v ~ Bernoulli(logit^{-1}(eta_v))
#   eta_v = alpha + beta * dose_v + w(s_v) + u_p
#   w(s) ~ GF(0, Matern(kappa, sigma_w))  [spatial Gaussian field via SPDE]
#   u_p ~ Normal(0, sigma_u^2)             [patient frailty]
#
# References:
#   Rue, Martino, & Chopin (2009), JRSS-B, 71(2):319-392 (INLA)
#   Lindgren, Rue, & Lindstrom (2011), JRSS-B, 73(4):423-498 (SPDE link)
#   Simpson et al. (2017), Stat. Sci., 32(1):1-28 (PC priors)

cat("=" , rep("=", 69), "\n", sep="")
cat("Spatial Dose-Response Model with R-INLA + SPDE\n")
cat("Manuscript: Part 2, Section 10.2 & 10.6A\n")
cat("=" , rep("=", 69), "\n\n", sep="")

# Check for INLA
if (!requireNamespace("INLA", quietly = TRUE)) {
  cat("Installing INLA...\n")
  install.packages("INLA", repos = c(INLA = "https://inla.r-inla-download.org/R/stable"),
                   dep = TRUE)
}
library(INLA)

# ============================================================
# 1. Simulate Spatial Dose-Toxicity Data
# ============================================================
set.seed(42)

n_patients <- 50
n_voxels_per_patient <- 200  # ROI voxels per patient

# Spatial coordinates (2D slice of organ — e.g., parotid gland)
coords <- matrix(runif(n_voxels_per_patient * 2, 0, 10), ncol = 2)

# True parameters
alpha_true <- -2.0        # baseline log-odds
beta_true <- 0.05         # dose effect (per Gy)
sigma_w_true <- 1.0       # spatial field marginal SD
range_true <- 3.0         # spatial correlation range
sigma_u_true <- 0.5       # patient frailty SD

# Generate data
dat_list <- list()
for (p in 1:n_patients) {
  # Patient frailty
  u_p <- rnorm(1, 0, sigma_u_true)

  # Dose field (simplified: gradient + noise)
  dose <- 30 + 10 * coords[, 1] / 10 + rnorm(n_voxels_per_patient, 0, 2)

  # Spatial field (simplified: smooth Gaussian process)
  # In practice, INLA constructs this via SPDE mesh
  dist_mat <- as.matrix(dist(coords))
  C_w <- sigma_w_true^2 * exp(-dist_mat / range_true)
  w <- MASS::mvrnorm(1, rep(0, n_voxels_per_patient), C_w + diag(1e-6, n_voxels_per_patient))

  # Linear predictor
  eta <- alpha_true + beta_true * dose + w + u_p

  # Toxicity outcome
  prob <- plogis(eta)
  y <- rbinom(n_voxels_per_patient, 1, prob)

  dat_list[[p]] <- data.frame(
    patient = p,
    x = coords[, 1], y_coord = coords[, 2],
    dose = dose, toxicity = y
  )
}

dat <- do.call(rbind, dat_list)
cat("Simulated data:", nrow(dat), "voxel-observations from", n_patients, "patients\n")
cat("Toxicity rate:", round(mean(dat$toxicity), 3), "\n")

# ============================================================
# 2. Build SPDE Mesh (Lindgren et al., 2011)
# ============================================================
cat("\nBuilding SPDE mesh...\n")

# Create mesh from spatial coordinates (using first patient's coords as template)
mesh <- inla.mesh.2d(
  loc = as.matrix(dat[dat$patient == 1, c("x", "y_coord")]),
  max.edge = c(1.5, 4),    # inner and outer max triangle edge length
  cutoff = 0.3,             # minimum distance between mesh nodes
  offset = c(2, 5)          # inner and outer extension
)
cat("Mesh:", mesh$n, "vertices,", nrow(mesh$graph$tv), "triangles\n")

# SPDE model with Penalized Complexity priors (Simpson et al., 2017)
# P(range < 1) = 0.01; P(sigma > 2) = 0.01
spde <- inla.spde2.pcmatern(
  mesh = mesh,
  prior.range = c(1, 0.01),   # P(range < 1) = 0.01
  prior.sigma = c(2, 0.01)    # P(sigma > 2) = 0.01
)

# ============================================================
# 3. Fit Model with INLA (Rue et al., 2009)
# ============================================================
cat("\nFitting INLA model...\n")

# Index for spatial effect (use first patient's coordinates for projection)
# In a full implementation, each patient would have their own spatial field
# or share a common spatial template

# For demonstration, fit a simplified model with:
# - Fixed effect: dose
# - Random effect: patient (iid)
# - Spatial random effect: SPDE on shared coordinate system

# Build A matrix (projector from mesh to observation locations)
# Using patient 1 coords repeated for simplicity
A <- inla.spde.make.A(mesh, loc = as.matrix(dat[, c("x", "y_coord")]))

# Stack data
stk <- inla.stack(
  data = list(y = dat$toxicity),
  A = list(A, 1, 1),
  effects = list(
    spatial = 1:spde$n.spde,
    patient = dat$patient,
    data.frame(intercept = 1, dose = dat$dose)
  ),
  tag = "est"
)

# Formula
formula <- y ~ -1 + intercept + dose +
  f(spatial, model = spde) +
  f(patient, model = "iid",
    hyper = list(prec = list(prior = "pc.prec", param = c(1, 0.01))))

# Fit
fit <- inla(formula,
            family = "binomial",
            data = inla.stack.data(stk),
            control.predictor = list(A = inla.stack.A(stk), compute = TRUE),
            control.compute = list(dic = TRUE, waic = TRUE, config = TRUE),
            verbose = FALSE)

# ============================================================
# 4. Results
# ============================================================
cat("\n", rep("-", 60), "\n", sep="")
cat("Fixed Effects:\n")
print(round(fit$summary.fixed, 4))

cat("\nHyperparameters:\n")
print(round(fit$summary.hyperpar, 4))

cat("\nModel comparison:\n")
cat("  DIC  =", round(fit$dic$dic, 1), "\n")
cat("  WAIC =", round(fit$waic$waic, 1), "\n")

# Extract spatial field estimates
cat("\nSpatial field (SPDE/GMRF):\n")
cat("  Posterior range: ", round(fit$summary.hyperpar["Range for spatial", "mean"], 2),
    " [", round(fit$summary.hyperpar["Range for spatial", "0.025quant"], 2),
    ", ", round(fit$summary.hyperpar["Range for spatial", "0.975quant"], 2), "]\n", sep="")
cat("  Posterior sigma: ", round(fit$summary.hyperpar["Stdev for spatial", "mean"], 2),
    " [", round(fit$summary.hyperpar["Stdev for spatial", "0.025quant"], 2),
    ", ", round(fit$summary.hyperpar["Stdev for spatial", "0.975quant"], 2), "]\n", sep="")

cat("\nTrue values: range =", range_true, ", sigma_w =", sigma_w_true, "\n")
cat("True dose effect: beta =", beta_true, "\n")
cat("Estimated dose effect:", round(fit$summary.fixed["dose", "mean"], 4),
    "[", round(fit$summary.fixed["dose", "0.025quant"], 4),
    ",", round(fit$summary.fixed["dose", "0.975quant"], 4), "]\n")

cat("\nDone.\n")
