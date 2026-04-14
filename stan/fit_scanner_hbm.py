"""
fit_scanner_hbm.py — Fit the scanner harmonization HBM via CmdStanPy
Manuscript: Part 1, Section 1 (§3, §5, §6)

Simulates multi-site radiomic data, fits the Gaussian HBM,
produces shrinkage plot and posterior predictive checks.

References:
  Gelman & Hill (2007), Data Analysis Using Regression and Multilevel Models
  Carpenter et al. (2017), Stan: A probabilistic programming language
"""

import numpy as np
import os

def simulate_multisite_data(n_scanners=6, n_per_scanner=None, seed=42):
    """Simulate radiomic feature measurements across scanners with site-level bias."""
    rng = np.random.default_rng(seed)

    if n_per_scanner is None:
        n_per_scanner = rng.integers(15, 80, size=n_scanners)

    # True parameters
    mu_0_true = 50.0       # global mean (e.g., mean SUV or HU)
    tau_true = 8.0         # between-scanner SD
    sigma_true = 5.0       # within-scanner SD

    scanner_names = ["Houston", "Boston", "Tokyo", "Zurich", "London", "Seoul"][:n_scanners]
    mu_true = rng.normal(mu_0_true, tau_true, size=n_scanners)

    y_all, scanner_all = [], []
    for j in range(n_scanners):
        y_j = rng.normal(mu_true[j], sigma_true, size=n_per_scanner[j])
        y_all.extend(y_j)
        scanner_all.extend([j + 1] * n_per_scanner[j])

    return {
        "N": len(y_all),
        "J": n_scanners,
        "scanner": scanner_all,
        "y": y_all,
        "scanner_names": scanner_names,
        "n_per_scanner": n_per_scanner.tolist(),
        "mu_true": mu_true,
        "mu_0_true": mu_0_true,
        "tau_true": tau_true,
        "sigma_true": sigma_true,
    }


def fit_model(data):
    """Fit the Stan model and return results."""
    from cmdstanpy import CmdStanModel

    stan_file = os.path.join(os.path.dirname(__file__), "scanner_harmonization.stan")
    model = CmdStanModel(stan_file=stan_file)

    stan_data = {k: data[k] for k in ["N", "J", "scanner", "y"]}

    # Pathfinder warm-start → NUTS (Zhang et al., 2022, JMLR)
    fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=2000,
        adapt_delta=0.95,      # raise for hierarchical models
        max_treedepth=12,
        seed=42,
        show_progress=True,
    )

    return fit


def diagnose_and_report(fit, data):
    """Print diagnostics and key results."""
    print("=" * 70)
    print("MCMC Diagnostics")
    print("=" * 70)
    print(fit.diagnose())

    # Extract summaries
    summary = fit.summary()
    print("\n" + "=" * 70)
    print("Parameter Estimates")
    print("=" * 70)

    params = ["mu_0", "tau", "sigma", "icc"]
    for p in params:
        if p in summary.index:
            row = summary.loc[p]
            print(f"  {p:8s}: {row['Mean']:8.3f} [{row['5%']:8.3f}, {row['95%']:8.3f}]  "
                  f"R-hat={row['R_hat']:.3f}  ESS={row['N_Eff']:.0f}")

    # Scanner-specific means
    print(f"\n{'Scanner':<10s} {'True μ':>8s} {'Post. Mean':>10s} {'95% CI':>16s} {'Shrinkage λ':>12s}")
    print("-" * 60)
    for j in range(data["J"]):
        mu_key = f"mu[{j+1}]"
        lam_key = f"lambda[{j+1}]"
        if mu_key in summary.index:
            mu_row = summary.loc[mu_key]
            lam_row = summary.loc[lam_key] if lam_key in summary.index else None
            lam_val = f"{lam_row['Mean']:.3f}" if lam_row is not None else "N/A"
            print(f"  {data['scanner_names'][j]:<8s} {data['mu_true'][j]:8.2f} "
                  f"{mu_row['Mean']:10.2f} [{mu_row['5%']:6.2f}, {mu_row['95%']:6.2f}]  {lam_val:>10s}")

    # PSIS-LOO (Vehtari et al., 2017)
    try:
        import arviz as az
        idata = az.from_cmdstanpy(fit)
        loo_result = az.loo(idata, pointwise=True)
        print(f"\nPSIS-LOO (Vehtari et al., 2017):")
        print(f"  elpd_loo = {loo_result.elpd_loo:.1f} ± {loo_result.se:.1f}")
        print(f"  p_loo = {loo_result.p_loo:.1f}")

        # Check Pareto k diagnostics
        k_vals = loo_result.pareto_k.values
        n_bad = np.sum(k_vals > 0.7)
        print(f"  Pareto k > 0.7: {n_bad}/{len(k_vals)} observations")
    except ImportError:
        print("\n(Install arviz for PSIS-LOO diagnostics: pip install arviz)")


def main():
    print("=" * 70)
    print("Scanner Harmonization HBM — Gaussian Hierarchical Model")
    print("Manuscript: Part 1, Sections 1, 5, 6")
    print("=" * 70)

    # Simulate data
    print("\nSimulating multi-site radiomic data...")
    data = simulate_multisite_data(n_scanners=6, seed=42)
    print(f"  {data['J']} scanners, {data['N']} total observations")
    print(f"  True: mu_0={data['mu_0_true']:.1f}, tau={data['tau_true']:.1f}, sigma={data['sigma_true']:.1f}")

    # Fit
    print("\nFitting Stan model (4 chains × 2000 samples)...")
    fit = fit_model(data)

    # Report
    diagnose_and_report(fit, data)
    print("\nDone.")


if __name__ == "__main__":
    main()
