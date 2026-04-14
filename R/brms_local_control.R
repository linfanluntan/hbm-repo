# brms_local_control.R
# Bayesian multilevel logistic regression for local control across institutions
# Manuscript: Part 1, Section 7 (Recipe: Outcome modeling across institutions)
#
# Model: Local control (binary) ~ dose + stage + (1 | institution) + (1 | physician)
#
# References:
#   Bürkner (2017), J. Stat. Softw., 80(1):1-28 (brms)
#   Vehtari, Gelman, & Gabry (2017), Stat. Comput. (PSIS-LOO)
#   Gabry et al. (2019), JRSS-A (visualization and workflow)

cat("=" , rep("=", 69), "\n", sep="")
cat("Bayesian Multilevel GLMM for Local Control (brms)\n")
cat("Manuscript: Part 1, Section 7\n")
cat("=" , rep("=", 69), "\n\n", sep="")

library(brms)

# ============================================================
# 1. Simulate Multi-Institution Clinical Data
# ============================================================
set.seed(123)

n_institutions <- 8
n_physicians_per <- 3
n_patients_per_phys <- sample(15:40, n_institutions * n_physicians_per, replace = TRUE)

# True parameters
alpha_0 <- 1.5           # global intercept (log-odds)
beta_dose <- 0.03        # dose effect per Gy
beta_stage <- -0.8       # stage III vs I-II effect
sigma_inst <- 0.6        # institution SD
sigma_phys <- 0.3        # physician SD

dat <- data.frame()
phys_id <- 0
for (inst in 1:n_institutions) {
  u_inst <- rnorm(1, 0, sigma_inst)
  for (phys in 1:n_physicians_per) {
    phys_id <- phys_id + 1
    v_phys <- rnorm(1, 0, sigma_phys)
    n_p <- n_patients_per_phys[phys_id]

    dose <- rnorm(n_p, 70, 5)               # prescribed dose (Gy)
    stage_iii <- rbinom(n_p, 1, 0.4)         # 40% stage III

    eta <- alpha_0 + beta_dose * (dose - 70) + beta_stage * stage_iii + u_inst + v_phys
    local_control <- rbinom(n_p, 1, plogis(eta))

    dat <- rbind(dat, data.frame(
      patient = seq_len(n_p) + nrow(dat),
      institution = paste0("Inst_", inst),
      physician = paste0("Phys_", phys_id),
      dose = dose,
      stage_iii = stage_iii,
      local_control = local_control
    ))
  }
}

cat("Data:", nrow(dat), "patients across", n_institutions, "institutions,",
    phys_id, "physicians\n")
cat("Local control rate:", round(mean(dat$local_control), 3), "\n\n")

# ============================================================
# 2. Fit Bayesian Multilevel Model (brms)
# ============================================================
cat("Fitting brms model...\n")
cat("Model: local_control ~ dose + stage_iii + (1|institution) + (1|physician)\n\n")

# Center dose for interpretability
dat$dose_centered <- dat$dose - 70

fit <- brm(
  local_control ~ dose_centered + stage_iii + (1 | institution) + (1 | physician),
  data = dat,
  family = bernoulli(link = "logit"),
  prior = c(
    prior(normal(0, 5), class = "Intercept"),
    prior(normal(0, 2.5), class = "b"),
    prior(cauchy(0, 1), class = "sd")    # Half-Cauchy on group SDs (Gelman, 2006)
  ),
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  seed = 42,
  control = list(adapt_delta = 0.95)
)

# ============================================================
# 3. Results and Diagnostics
# ============================================================
cat("\n", rep("-", 60), "\n", sep="")
cat("Model Summary:\n")
print(summary(fit))

cat("\nFixed Effects (posterior median [95% CrI]):\n")
fe <- fixef(fit)
for (p in rownames(fe)) {
  cat(sprintf("  %-15s: %6.3f [%6.3f, %6.3f]\n", p, fe[p, "Estimate"],
              fe[p, "Q2.5"], fe[p, "Q97.5"]))
}

cat("\nRandom Effects SDs:\n")
ve <- VarCorr(fit)
for (g in names(ve)) {
  sd_est <- ve[[g]]$sd[1, "Estimate"]
  sd_lo <- ve[[g]]$sd[1, "Q2.5"]
  sd_hi <- ve[[g]]$sd[1, "Q97.5"]
  cat(sprintf("  %-15s: %5.3f [%5.3f, %5.3f]\n", g, sd_est, sd_lo, sd_hi))
}

cat("\nTrue values: sigma_inst =", sigma_inst, ", sigma_phys =", sigma_phys, "\n")
cat("True dose effect:", beta_dose, ", True stage effect:", beta_stage, "\n")

# ============================================================
# 4. PSIS-LOO Cross-Validation (Vehtari et al., 2017)
# ============================================================
cat("\n", rep("-", 60), "\n", sep="")
cat("PSIS-LOO Cross-Validation:\n")
loo_result <- loo(fit)
print(loo_result)

# ============================================================
# 5. Posterior Predictive Check (Gabry et al., 2019)
# ============================================================
cat("\nPosterior Predictive Check:\n")
cat("(Run: pp_check(fit, type = 'bars', ndraws = 100) for visual check)\n")

# Observed vs predicted local control rate
y_rep <- posterior_predict(fit, ndraws = 1000)
pred_rate <- mean(colMeans(y_rep))
cat(sprintf("  Observed LC rate: %.3f\n", mean(dat$local_control)))
cat(sprintf("  Predicted LC rate: %.3f\n", pred_rate))

# ============================================================
# 6. Institution-Level Shrinkage
# ============================================================
cat("\n", rep("-", 60), "\n", sep="")
cat("Institution-Level Shrinkage (Random Intercepts):\n")
re <- ranef(fit)$institution
cat(sprintf("  %-10s %10s %16s\n", "Inst", "Post. Mean", "95% CrI"))
for (inst in rownames(re)) {
  cat(sprintf("  %-10s %10.3f [%6.3f, %6.3f]\n", inst,
              re[inst, "Estimate", "Intercept"],
              re[inst, "Q2.5", "Intercept"],
              re[inst, "Q97.5", "Intercept"]))
}

cat("\nDone.\n")
