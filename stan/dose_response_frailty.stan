// dose_response_frailty.stan
// Hierarchical logistic dose-response model with patient frailty
// Manuscript: Part 1, Section 7 (Recipe: Dose-toxicity GLMM)
//
// Model:
//   y_ip ~ Bernoulli(logit^{-1}(eta_ip))
//   eta_ip = alpha_0 + beta * dose_ip + u_p + v_s[p]
//   u_p ~ Normal(0, sigma_u^2)           [patient-level frailty]
//   v_s ~ Normal(0, sigma_v^2)           [site-level random intercept]
//   alpha_0 ~ Normal(0, 5)               [global intercept]
//   beta ~ Normal(0, 2.5)                [dose coefficient]
//   sigma_u, sigma_v ~ Half-Normal(0, 1) [variance components]
//
// References:
//   Vaupel, Manton, & Stallard (1979), Demography (frailty models)
//   Gelman (2006), Bayesian Analysis (half-Cauchy/half-normal priors)

data {
  int<lower=1> N;                          // total observations
  int<lower=1> P;                          // number of patients
  int<lower=1> S;                          // number of sites
  array[N] int<lower=0, upper=1> y;       // toxicity outcome (0/1)
  vector[N] dose;                          // standardized dose
  array[N] int<lower=1, upper=P> patient; // patient index
  array[N] int<lower=1, upper=S> site;    // site index
}

parameters {
  real alpha_0;                            // global intercept
  real beta_dose;                          // dose effect (log-odds per unit dose)
  real<lower=0> sigma_u;                   // patient frailty SD
  real<lower=0> sigma_v;                   // site effect SD
  vector[P] u_raw;                         // non-centered patient effects
  vector[S] v_raw;                         // non-centered site effects
}

transformed parameters {
  vector[P] u = sigma_u * u_raw;           // patient frailty
  vector[S] v = sigma_v * v_raw;           // site random intercept
  vector[N] eta;
  for (n in 1:N)
    eta[n] = alpha_0 + beta_dose * dose[n] + u[patient[n]] + v[site[n]];
}

model {
  // Priors
  alpha_0 ~ normal(0, 5);
  beta_dose ~ normal(0, 2.5);
  sigma_u ~ normal(0, 1);                 // half-normal (constrained positive)
  sigma_v ~ normal(0, 1);

  // Non-centered group effects
  u_raw ~ std_normal();
  v_raw ~ std_normal();

  // Likelihood
  y ~ bernoulli_logit(eta);
}

generated quantities {
  // Posterior predictive checks
  array[N] int y_rep;
  for (n in 1:N)
    y_rep[n] = bernoulli_logit_rng(eta[n]);

  // Dose at 50% toxicity probability (ED50) for average patient/site
  real ed50 = -alpha_0 / beta_dose;

  // Variance partition coefficient (VPC)
  // fraction of latent-scale variance attributable to patients vs sites
  real vpc_patient = square(sigma_u) / (square(sigma_u) + square(sigma_v) + square(pi()) / 3);
  real vpc_site = square(sigma_v) / (square(sigma_u) + square(sigma_v) + square(pi()) / 3);

  // Log-likelihood for PSIS-LOO
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta[n]);
}
