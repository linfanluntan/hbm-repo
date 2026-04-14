// beta_binomial_toxicity.stan
// Beta-Binomial Hierarchical Model for toxicity rates across treatment centers
// Manuscript: Part 1, Section 2.2 (Beta-Binomial Hierarchical Bayes)
//
// Model:
//   y_i ~ Binomial(n_i, theta_i)           [observed toxicity events at center i]
//   theta_i ~ Beta(alpha, beta)             [center-specific toxicity rate]
//   alpha = mu * kappa                      [reparameterize: mean mu, concentration kappa]
//   beta = (1 - mu) * kappa
//   mu ~ Beta(1, 1)                         [hyperprior on population toxicity rate]
//   kappa ~ Pareto(1, 1.5)                  [hyperprior on concentration/homogeneity]
//
// References:
//   Bernardo & Smith (2000), Bayesian Theory
//   Gelman et al. (2013), Bayesian Data Analysis (3rd ed.)

data {
  int<lower=1> I;                     // number of centers
  array[I] int<lower=0> y;           // toxicity events per center
  array[I] int<lower=1> n;           // patients per center
}

parameters {
  real<lower=0, upper=1> mu;          // population mean toxicity rate
  real<lower=1> kappa;                // concentration (higher = less between-center variation)
  vector<lower=0, upper=1>[I] theta;  // center-specific toxicity rates
}

transformed parameters {
  real<lower=0> alpha_hyper = mu * kappa;
  real<lower=0> beta_hyper = (1 - mu) * kappa;
}

model {
  // Hyperpriors
  mu ~ beta(1, 1);                    // uniform on [0,1]
  kappa ~ pareto(1, 1.5);            // heavy-tailed prior on concentration

  // Group-level prior
  theta ~ beta(alpha_hyper, beta_hyper);

  // Likelihood
  y ~ binomial(n, theta);
}

generated quantities {
  // Posterior predictive
  array[I] int y_rep;
  for (i in 1:I)
    y_rep[i] = binomial_rng(n[i], theta[i]);

  // Shrinkage: compare posterior mean to MLE
  vector[I] mle = to_vector(y) ./ to_vector(n);
  vector[I] shrinkage = theta - mle;

  // Predictive rate for a NEW center
  real theta_new = beta_rng(alpha_hyper, beta_hyper);

  // Log-likelihood for PSIS-LOO
  vector[I] log_lik;
  for (i in 1:I)
    log_lik[i] = binomial_lpmf(y[i] | n[i], theta[i]);
}
