// scanner_harmonization.stan
// Gaussian Hierarchical Bayesian Model for scanner/site harmonization
// Manuscript: Part 1, Section 1 (§3: Scanner-Specific Bias)
//
// Model:
//   y_ij ~ Normal(mu_j, sigma^2)          [data: measurement i in scanner j]
//   mu_j ~ Normal(mu_0, tau^2)             [scanner-level mean, pooled]
//   mu_0 ~ Normal(0, 10^2)                 [hyperprior on global mean]
//   sigma ~ Half-Cauchy(0, 5)              [within-scanner noise]
//   tau ~ Half-Cauchy(0, 5)                [between-scanner variability]
//
// Partial pooling shrinkage:
//   E[mu_j | y] ≈ lambda_j * ybar_j + (1 - lambda_j) * mu_0
//   lambda_j = (n_j / sigma^2) / (n_j / sigma^2 + 1 / tau^2)
//
// References:
//   Gelman & Hill (2007), Data Analysis Using Regression and Multilevel Models
//   Gelman (2006), Prior distributions for variance parameters. Bayesian Analysis.
//   Carpenter et al. (2017), Stan: A probabilistic programming language. J. Stat. Softw.

data {
  int<lower=1> N;                // total observations
  int<lower=1> J;                // number of scanners/sites
  array[N] int<lower=1, upper=J> scanner;  // scanner index for each obs
  vector[N] y;                   // radiomic feature measurement
}

parameters {
  real mu_0;                     // global mean (hyperparameter)
  real<lower=0> tau;             // between-scanner SD
  real<lower=0> sigma;           // within-scanner SD
  vector[J] mu_raw;             // non-centered scanner means (for efficiency)
}

transformed parameters {
  // Non-centered parameterization (Papaspiliopoulos et al., 2007, Stat. Sci.)
  // mu_j = mu_0 + tau * mu_raw_j, where mu_raw_j ~ Normal(0, 1)
  // This avoids the "funnel" geometry that causes divergent transitions
  vector[J] mu = mu_0 + tau * mu_raw;
}

model {
  // Hyperpriors
  mu_0 ~ normal(0, 10);
  tau ~ cauchy(0, 5);            // Half-Cauchy via constraint <lower=0>
  sigma ~ cauchy(0, 5);

  // Group-level prior (non-centered)
  mu_raw ~ std_normal();

  // Likelihood
  y ~ normal(mu[scanner], sigma);
}

generated quantities {
  // Posterior predictive for model checking
  vector[N] y_rep;
  for (n in 1:N)
    y_rep[n] = normal_rng(mu[scanner[n]], sigma);

  // Shrinkage factors per scanner
  vector[J] lambda;
  {
    // Count observations per scanner
    array[J] int n_per_scanner = rep_array(0, J);
    for (n in 1:N)
      n_per_scanner[scanner[n]] += 1;

    for (j in 1:J)
      lambda[j] = (n_per_scanner[j] / square(sigma)) /
                  (n_per_scanner[j] / square(sigma) + 1.0 / square(tau));
  }

  // Intraclass correlation coefficient (ICC)
  real icc = square(tau) / (square(tau) + square(sigma));

  // Log-likelihood for PSIS-LOO (Vehtari et al., 2017)
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = normal_lpdf(y[n] | mu[scanner[n]], sigma);
}
