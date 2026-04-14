"""
numpyro_site_pooling.py — JAX-accelerated hierarchical model with NumPyro
Manuscript: Part 1, Section 11.1 (Smarter Warm-Starts and Gradient-Based Samplers)

Demonstrates:
  - Non-centered parameterization in NumPyro
  - GPU/TPU-accelerated NUTS via JAX
  - Posterior predictive checks
  - Comparison: centered vs non-centered divergent transitions

References:
  Phan, Pradhan, & Jankowiak (2019), arXiv:1912.11554 (NumPyro)
  Hoffman & Gelman (2014), JMLR (NUTS)
  Papaspiliopoulos, Roberts, & Sköld (2007), Stat. Sci. (non-centered param.)
  Betancourt (2017), arXiv:1701.02434 (divergent transitions)
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import hpdi, summary
import numpy as np


# ============================================================
# Model Definitions
# ============================================================

def model_noncentered(scanner, y=None, J=None):
    """
    Gaussian HBM with non-centered parameterization.
    mu_j = mu_0 + tau * z_j, z_j ~ Normal(0, 1)
    Avoids funnel geometry (Neal, 2003).
    """
    mu_0 = numpyro.sample("mu_0", dist.Normal(0.0, 10.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(5.0))

    with numpyro.plate("scanners", J):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
        mu = numpyro.deterministic("mu", mu_0 + tau * z)

    with numpyro.plate("obs", len(scanner)):
        numpyro.sample("y", dist.Normal(mu[scanner], sigma), obs=y)


def model_centered(scanner, y=None, J=None):
    """
    Gaussian HBM with centered parameterization.
    mu_j ~ Normal(mu_0, tau)
    Prone to divergent transitions when tau is small (the "funnel").
    """
    mu_0 = numpyro.sample("mu_0", dist.Normal(0.0, 10.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(5.0))

    with numpyro.plate("scanners", J):
        mu = numpyro.sample("mu", dist.Normal(mu_0, tau))

    with numpyro.plate("obs", len(scanner)):
        numpyro.sample("y", dist.Normal(mu[scanner], sigma), obs=y)


# ============================================================
# Simulation & Fitting
# ============================================================

def simulate_data(n_scanners=8, seed=42):
    """Simulate multi-scanner radiomic data."""
    rng = np.random.default_rng(seed)
    mu_0, tau, sigma = 50.0, 8.0, 5.0
    n_per = rng.integers(20, 60, size=n_scanners)
    mu_true = rng.normal(mu_0, tau, size=n_scanners)

    y_list, scanner_list = [], []
    for j in range(n_scanners):
        y_j = rng.normal(mu_true[j], sigma, size=n_per[j])
        y_list.extend(y_j)
        scanner_list.extend([j] * n_per[j])

    return (jnp.array(scanner_list, dtype=jnp.int32),
            jnp.array(y_list, dtype=jnp.float32),
            n_scanners, mu_true, n_per)


def fit_and_compare():
    """Fit both parameterizations and compare divergences."""
    scanner, y, J, mu_true, n_per = simulate_data()

    results = {}
    for name, model_fn in [("non-centered", model_noncentered),
                            ("centered", model_centered)]:
        print(f"\n--- {name.upper()} parameterization ---")
        kernel = NUTS(model_fn, target_accept_prob=0.90)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000,
                     num_chains=4, progress_bar=True)
        mcmc.run(random.PRNGKey(0), scanner=scanner, y=y, J=J)

        # Count divergences
        divergences = mcmc.get_extra_fields()["diverging"].sum().item()
        print(f"  Divergent transitions: {divergences}")

        # Summary
        samples = mcmc.get_samples()
        mu_0_samples = samples["mu_0"]
        tau_samples = samples["tau"]
        sigma_samples = samples["sigma"]

        print(f"  mu_0: {mu_0_samples.mean():.2f} [{jnp.percentile(mu_0_samples, 5):.2f}, {jnp.percentile(mu_0_samples, 95):.2f}]")
        print(f"  tau:  {tau_samples.mean():.2f} [{jnp.percentile(tau_samples, 5):.2f}, {jnp.percentile(tau_samples, 95):.2f}]")
        print(f"  sigma:{sigma_samples.mean():.2f} [{jnp.percentile(sigma_samples, 5):.2f}, {jnp.percentile(sigma_samples, 95):.2f}]")

        results[name] = {"divergences": divergences, "samples": samples}

    # Posterior predictive check (non-centered model)
    print("\n--- Posterior Predictive Check (non-centered) ---")
    predictive = Predictive(model_noncentered, results["non-centered"]["samples"])
    pred_samples = predictive(random.PRNGKey(1), scanner=scanner, J=J)
    y_rep = pred_samples["y"]

    y_np = np.array(y)
    y_rep_np = np.array(y_rep)
    print(f"  Observed mean: {y_np.mean():.2f}, Predicted mean: {y_rep_np.mean(axis=0).mean():.2f}")
    print(f"  Observed SD:   {y_np.std():.2f}, Predicted SD:   {y_rep_np.std(axis=1).mean():.2f}")

    # Comparison
    print("\n" + "=" * 50)
    print("COMPARISON: centered vs non-centered")
    print("=" * 50)
    print(f"  Centered divergences:     {results['centered']['divergences']}")
    print(f"  Non-centered divergences: {results['non-centered']['divergences']}")
    print(f"  → Non-centered parameterization reduces divergent transitions")
    print(f"    by avoiding the 'funnel' geometry (Neal, 2003; Betancourt, 2017)")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("NumPyro: JAX-Accelerated Hierarchical Bayesian Model")
    print("=" * 60)
    numpyro.set_host_device_count(4)  # use 4 CPU cores for chains
    fit_and_compare()
    print("\nDone.")
