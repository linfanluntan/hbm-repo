"""
vi_vs_mcmc.py — Compare Variational Inference (ADVI) with MCMC (NUTS)
Manuscript: Part 1, Section 11.3 (Flexible VI methods)

Demonstrates the speed-accuracy tradeoff:
  - ADVI: fast but potentially biased (underestimates posterior variance)
  - NUTS: exact but slower
  - Comparison on the scanner harmonization HBM

References:
  Kucukelbir et al. (2017), JMLR, 18(14):1-45 (ADVI)
  Blei, Kucukelbir, & McAuliffe (2017), JASA, 112(518):859-877 (VI review)
  Hoffman & Gelman (2014), JMLR (NUTS)
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide
from numpyro.infer.autoguide import AutoNormal, AutoLowRankMultivariateNormal
import numpyro.optim as optim
import numpy as np
import time


def hbm_model(scanner, y=None, J=None):
    """Gaussian HBM, non-centered."""
    mu_0 = numpyro.sample("mu_0", dist.Normal(0.0, 10.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(5.0))
    with numpyro.plate("scanners", J):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
    mu = mu_0 + tau * z
    with numpyro.plate("obs", len(scanner)):
        numpyro.sample("y", dist.Normal(mu[scanner], sigma), obs=y)


def simulate_data(n_scanners=8, seed=42):
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
            n_scanners, mu_true)


def fit_nuts(scanner, y, J, num_warmup=1000, num_samples=2000, num_chains=4):
    """Fit with NUTS (exact MCMC)."""
    kernel = NUTS(hbm_model, target_accept_prob=0.90)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)

    t0 = time.time()
    mcmc.run(random.PRNGKey(0), scanner=scanner, y=y, J=J)
    elapsed = time.time() - t0

    samples = mcmc.get_samples()
    divergences = mcmc.get_extra_fields()["diverging"].sum().item()

    return {
        "mu_0": float(samples["mu_0"].mean()),
        "mu_0_std": float(samples["mu_0"].std()),
        "tau": float(samples["tau"].mean()),
        "tau_std": float(samples["tau"].std()),
        "sigma": float(samples["sigma"].mean()),
        "sigma_std": float(samples["sigma"].std()),
        "time": elapsed,
        "divergences": divergences,
        "method": "NUTS",
    }


def fit_advi(scanner, y, J, guide_type="normal", num_steps=20000, lr=0.01):
    """Fit with ADVI (variational inference)."""
    if guide_type == "normal":
        guide = AutoNormal(hbm_model)
        name = "ADVI (diagonal)"
    else:
        guide = AutoLowRankMultivariateNormal(hbm_model, rank=5)
        name = "ADVI (low-rank)"

    optimizer = optim.Adam(lr)
    svi = SVI(hbm_model, guide, optimizer, loss=Trace_ELBO())

    t0 = time.time()
    svi_result = svi.run(random.PRNGKey(1), num_steps,
                          scanner=scanner, y=y, J=J,
                          progress_bar=False)
    elapsed = time.time() - t0

    # Sample from the variational posterior
    predictive = numpyro.infer.Predictive(guide, params=svi_result.params,
                                           num_samples=4000)
    vi_samples = predictive(random.PRNGKey(2), scanner=scanner, y=y, J=J)

    return {
        "mu_0": float(vi_samples["mu_0"].mean()),
        "mu_0_std": float(vi_samples["mu_0"].std()),
        "tau": float(vi_samples["tau"].mean()),
        "tau_std": float(vi_samples["tau"].std()),
        "sigma": float(vi_samples["sigma"].mean()),
        "sigma_std": float(vi_samples["sigma"].std()),
        "time": elapsed,
        "final_elbo": float(svi_result.losses[-1]),
        "method": name,
    }


def main():
    print("=" * 70)
    print("ADVI vs NUTS for Hierarchical Bayesian Models")
    print("=" * 70)

    numpyro.set_host_device_count(4)
    scanner, y, J, mu_true = simulate_data()
    print(f"\nData: {J} scanners, {len(y)} observations")

    # True values
    print(f"True: mu_0=50.0, tau=8.0, sigma=5.0\n")

    # Fit with all methods
    results = []
    print("Fitting NUTS (4 chains × 2000 samples)...")
    results.append(fit_nuts(scanner, y, J))

    print("Fitting ADVI (diagonal normal guide, 20K steps)...")
    results.append(fit_advi(scanner, y, J, guide_type="normal"))

    print("Fitting ADVI (low-rank MVN guide, 20K steps)...")
    results.append(fit_advi(scanner, y, J, guide_type="lowrank"))

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    header = f"{'Method':<25s} {'mu_0':>10s} {'tau':>10s} {'sigma':>10s} {'Time(s)':>8s}"
    print(header)
    print("-" * len(header))
    print(f"{'TRUE'::<25s} {'50.00':>10s} {'8.00':>10s} {'5.00':>10s}")
    for r in results:
        print(f"{r['method']:<25s} "
              f"{r['mu_0']:.2f}±{r['mu_0_std']:.2f}  "
              f"{r['tau']:.2f}±{r['tau_std']:.2f}  "
              f"{r['sigma']:.2f}±{r['sigma_std']:.2f}  "
              f"{r['time']:7.1f}")

    # Variance underestimation check
    print("\nPosterior SD comparison (ADVI typically underestimates):")
    nuts = results[0]
    for r in results[1:]:
        ratio_mu0 = r["mu_0_std"] / nuts["mu_0_std"]
        ratio_tau = r["tau_std"] / nuts["tau_std"]
        print(f"  {r['method']}: SD(mu_0)/SD_NUTS(mu_0) = {ratio_mu0:.2f}, "
              f"SD(tau)/SD_NUTS(tau) = {ratio_tau:.2f}")
        if ratio_mu0 < 0.8 or ratio_tau < 0.8:
            print(f"    ⚠ Variance underestimation detected (ratio < 0.8)")

    print("\nKey takeaway:")
    print("  ADVI is 5-20× faster but underestimates posterior variance.")
    print("  Use ADVI for initialization/screening; NUTS for final inference.")
    print("  (Blei, Kucukelbir, & McAuliffe, 2017, JASA)")


if __name__ == "__main__":
    main()
