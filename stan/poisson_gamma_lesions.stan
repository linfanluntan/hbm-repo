// poisson_gamma_lesions.stan
// Poisson-Gamma Hierarchical Model for lesion counts across patients/cohorts
// Manuscript: Part 1, Section 2.3 (Poisson-Gamma Hierarchical Bayes)
//
// Model:
//   y_i ~ Poisson(n_i * lambda_i)         [lesion count for patient i, offset n_i]
//   lambda_i ~ Gamma(alpha, beta)          [patient-specific rate]
//   alpha ~ Exponential(1)                 [shape hyperprior]
//   beta ~ Gamma(1, 1)                     [rate hyperprior]
//
// References:
//   DeGroot (1970), Optimal Statistical Decisions
//   Gelman et al. (2013), Bayesian Data Analysis (3rd ed.)

data {
  int<lower=1> I;                     // number of patients/units
  array[I] int<lower=0> y;           // observed lesion counts
  vector<lower=0>[I] exposure;        // exposure (e.g., scan volume, follow-up time)
}

parameters {
  real<lower=0> alpha_hyper;          // Gamma shape
  real<lower=0> beta_hyper;           // Gamma rate
  vector<lower=0>[I] lambda;          // patient-specific rates
}

model {
  // Hyperpriors
  alpha_hyper ~ exponential(1);
  beta_hyper ~ gamma(1, 1);

  // Group-level
  lambda ~ gamma(alpha_hyper, beta_hyper);

  // Likelihood
  y ~ poisson(exposure .* lambda);
}

generated quantities {
  // Posterior predictive
  array[I] int y_rep;
  for (i in 1:I)
    y_rep[i] = poisson_rng(exposure[i] * lambda[i]);

  // Population mean and variance
  real pop_mean = alpha_hyper / beta_hyper;
  real pop_var = alpha_hyper / square(beta_hyper);

  // Shrinkage: compare posterior rate to MLE rate
  vector[I] mle_rate;
  for (i in 1:I)
    mle_rate[i] = y[i] / exposure[i];

  // Predictive rate for new patient
  real lambda_new = gamma_rng(alpha_hyper, beta_hyper);

  // Log-likelihood for PSIS-LOO
  vector[I] log_lik;
  for (i in 1:I)
    log_lik[i] = poisson_lpmf(y[i] | exposure[i] * lambda[i]);
}
