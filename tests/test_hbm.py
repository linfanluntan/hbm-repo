"""
test_hbm.py — Verify core HBM implementations

Tests:
  1. Conjugate posterior closed-form (Normal-Normal)
  2. Conjugate posterior closed-form (Beta-Binomial)
  3. Shrinkage factor computation
  4. Partial pooling MSE < no-pooling MSE
"""

import numpy as np


def test_normal_normal_conjugate():
    """Verify Normal-Normal conjugate posterior matches closed form.

    Prior: mu ~ Normal(mu_0, tau^2)
    Likelihood: ybar | mu ~ Normal(mu, sigma^2/n)
    Posterior: mu | y ~ Normal(mu_post, sigma_post^2)
      mu_post = lambda * ybar + (1-lambda) * mu_0
      lambda = (n/sigma^2) / (n/sigma^2 + 1/tau^2)
    """
    mu_0, tau, sigma = 50.0, 8.0, 5.0
    n, ybar = 30, 55.0

    # Shrinkage factor
    lam = (n / sigma**2) / (n / sigma**2 + 1 / tau**2)
    mu_post = lam * ybar + (1 - lam) * mu_0
    sigma_post_sq = 1 / (n / sigma**2 + 1 / tau**2)

    # Check: lambda should be between 0 and 1
    assert 0 < lam < 1, f"Lambda out of range: {lam}"
    # Check: posterior mean is between prior mean and data mean
    assert min(mu_0, ybar) <= mu_post <= max(mu_0, ybar), \
        f"Posterior mean {mu_post} not between {mu_0} and {ybar}"
    # Check: posterior variance < both prior and likelihood variance
    assert sigma_post_sq < tau**2, "Posterior variance should be less than prior"
    assert sigma_post_sq < sigma**2 / n, "Posterior variance should be less than likelihood"

    # Numerical check
    assert abs(lam - 0.9846) < 0.01, f"Lambda = {lam}, expected ~0.985"
    assert abs(mu_post - 54.92) < 0.1, f"mu_post = {mu_post}, expected ~54.92"
    print("  [PASS] Normal-Normal conjugate posterior")


def test_beta_binomial_conjugate():
    """Verify Beta-Binomial conjugate posterior.

    Prior: theta ~ Beta(alpha, beta)
    Likelihood: y ~ Binomial(n, theta)
    Posterior: theta | y ~ Beta(alpha + y, beta + n - y)
    """
    alpha_prior, beta_prior = 2.0, 8.0  # prior: mean = 0.2
    n, y = 50, 15  # 15 events in 50 trials

    # Posterior parameters
    alpha_post = alpha_prior + y
    beta_post = beta_prior + n - y

    # Posterior mean
    post_mean = alpha_post / (alpha_post + beta_post)
    mle = y / n

    # Check: posterior mean is between prior mean and MLE
    prior_mean = alpha_prior / (alpha_prior + beta_prior)
    assert min(prior_mean, mle) <= post_mean <= max(prior_mean, mle), \
        f"Post mean {post_mean} not between prior {prior_mean} and MLE {mle}"

    # With large n, posterior should be close to MLE
    assert abs(post_mean - mle) < abs(prior_mean - mle), \
        "Posterior should be closer to MLE than prior with n=50"

    assert abs(alpha_post - 17.0) < 0.01
    assert abs(beta_post - 43.0) < 0.01
    print("  [PASS] Beta-Binomial conjugate posterior")


def test_shrinkage_factor():
    """Verify shrinkage factor properties.

    lambda_j = (n_j/sigma^2) / (n_j/sigma^2 + 1/tau^2)
    - More data (larger n_j) → lambda closer to 1 (less shrinkage)
    - Less data (smaller n_j) → lambda closer to 0 (more shrinkage)
    - Large tau (high between-group variance) → lambda closer to 1
    """
    sigma, tau = 5.0, 1.0  # small tau → more shrinkage visible

    # Test monotonicity in n
    lambdas = []
    for n in [5, 10, 30, 100, 500]:
        lam = (n / sigma**2) / (n / sigma**2 + 1 / tau**2)
        lambdas.append(lam)

    for k in range(len(lambdas) - 1):
        assert lambdas[k] < lambdas[k+1], \
            f"Lambda should increase with n: {lambdas}"
    assert lambdas[0] < 0.3, f"Small n with small tau should have lambda < 0.3, got {lambdas[0]}"
    assert lambdas[-1] > 0.95, f"Large n should have lambda > 0.95, got {lambdas[-1]}"

    # Test: tau = 0 → lambda = 0 (complete pooling)
    lam_zero_tau = (30 / sigma**2) / (30 / sigma**2 + 1 / 0.001**2)
    assert lam_zero_tau < 0.001, "When tau→0, lambda→0 (complete pooling)"

    print("  [PASS] Shrinkage factor properties")


def test_partial_pooling_mse():
    """Verify that partial pooling has lower MSE than no-pooling and complete-pooling."""
    rng = np.random.default_rng(42)
    n_sites, n_reps = 8, 1000
    mu_0, tau, sigma = 50.0, 8.0, 5.0
    n_per = np.array([80, 60, 15, 40, 10, 55, 25, 35])

    mse_none, mse_full, mse_partial = [], [], []

    for _ in range(n_reps):
        mu_true = rng.normal(mu_0, tau, size=n_sites)

        # Generate site means
        ybar = np.array([rng.normal(mu_true[j], sigma / np.sqrt(n_per[j]))
                          for j in range(n_sites)])

        # No pooling: use site-specific means
        mse_none.append(np.mean((ybar - mu_true)**2))

        # Complete pooling: use grand mean
        grand = np.mean(ybar)
        mse_full.append(np.mean((grand - mu_true)**2))

        # Partial pooling
        lam = (n_per / sigma**2) / (n_per / sigma**2 + 1 / tau**2)
        mu_hat = np.mean(ybar)
        pooled = lam * ybar + (1 - lam) * mu_hat
        mse_partial.append(np.mean((pooled - mu_true)**2))

    avg_none = np.mean(mse_none)
    avg_full = np.mean(mse_full)
    avg_partial = np.mean(mse_partial)

    assert avg_partial < avg_none, \
        f"Partial pooling MSE ({avg_partial:.3f}) should be < no-pooling ({avg_none:.3f})"
    assert avg_partial < avg_full, \
        f"Partial pooling MSE ({avg_partial:.3f}) should be < complete-pooling ({avg_full:.3f})"

    reduction = (1 - avg_partial / avg_none) * 100
    print(f"  [PASS] Partial pooling MSE ({avg_partial:.3f}) < no-pooling ({avg_none:.3f}), "
          f"reduction: {reduction:.0f}%")


if __name__ == "__main__":
    print("=" * 50)
    print("HBM Repository Tests")
    print("=" * 50)
    test_normal_normal_conjugate()
    test_beta_binomial_conjugate()
    test_shrinkage_factor()
    test_partial_pooling_mse()
    print("\nAll tests passed!")
