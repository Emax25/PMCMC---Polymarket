"""Synthetic data generator for PMCMC validation experiments.

Simulates the full latent-variable model (§5): a regime-switching Gaussian
random walk for log-odds price X, a logistic insider indicator Z driven by
per-wallet propensities θ_w, and a heteroskedastic observation model whose
variance is scaled by trade size.

``SyntheticMarket`` (returned by ``generate_market``) mirrors the
``ProcessedMarket`` interface so that all downstream inference and plotting
code works unchanged on both real and synthetic data.

Two generation modes are supported:

  * **planted-insider** (the default ``generate_market``): wallets
    ``[0, n_insider_wallets)`` are forced to high propensity and trade more
    often, which makes recovery experiments identifiable;
  * **prior-predictive** (``generate_prior_predictive_market``): every wallet
    draws θ_w from the Beta(a, b) prior and trades uniformly. Paired with
    ``params_from_prior`` this yields exact draws from the joint
    ``p(phi) p(latents, data | phi)``, the sampling scheme simulation-based
    calibration requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.default_params import ModelParams, PhiPrior
from src.utils.transforms import logit, sigmoid

# Smallest Inverse-Gamma shape/scale ``params_from_prior`` will sample from. The
# PhiPrior tau2 defaults are IG(1e-9, 1e-9) — a numerically improper prior kept
# deliberately vanishing so it does not perturb the VEM M-step (STATUS.md P11).
# Sampling it is meaningless: the draws span hundreds of orders of magnitude and
# overflow to 0/inf, so simulation callers must supply proper hyperparameters.
_MIN_IG_HYPERPARAM = 1e-3


@dataclass
class SyntheticMarket:
    """One simulated market with ground-truth latents and observations.

    All arrays are length T (number of trades). Ground-truth latents (X, V,
    Z, theta_w) are only available here — they have no analog for real data.
    """

    # Ground-truth latent variables (available only in synthetic experiments)
    X: np.ndarray  # (T,) logit of true public-info probability
    V: np.ndarray  # (T,) int8 volatility regime {0, 1}
    Z: np.ndarray  # (T,) int8 insider indicator {0, 1}
    theta_w: np.ndarray  # (n_wallets,) true wallet propensities

    # Observations
    Y: np.ndarray  # (T,) logit-price observations
    p: np.ndarray  # (T,) trade prices = sigmoid(Y)
    S: np.ndarray  # (T,) trade sizes in USDC
    S_bar: float  # within-market mean size (used to normalise log-size ratios)

    # Trade metadata
    t: np.ndarray  # (T,) trade timestamps in seconds
    delta: np.ndarray  # (T,) inter-trade times; delta[0] = 0 (sentinel)
    wallet_ids: np.ndarray  # (T,) integer wallet index for each trade

    # Which wallet indices were injected as insiders
    insider_wallet_ids: list[int]


# ---------------- Prior draws ----------------


def _check_proper_ig(block: str, alpha: float, beta: float) -> None:
    """Reject Inverse-Gamma hyperparameters too small to sample meaningfully."""
    if not (alpha >= _MIN_IG_HYPERPARAM and beta >= _MIN_IG_HYPERPARAM):
        raise ValueError(
            f"PhiPrior {block} Inverse-Gamma hyperparameters "
            f"(alpha={alpha:g}, beta={beta:g}) fall below {_MIN_IG_HYPERPARAM:g}. "
            "P11: improper prior — supply a proper PhiPrior for simulation."
        )


def _draw_inverse_gamma(alpha: float, beta: float, rng: np.random.Generator) -> float:
    """Draw one Inverse-Gamma(alpha, beta) variate as ``1 / Gamma(alpha, 1/beta)``."""
    return float(1.0 / rng.gamma(alpha, scale=1.0 / beta))


def params_from_prior(prior: PhiPrior, rng: np.random.Generator) -> ModelParams:
    """Draw one ModelParams from the ``PhiPrior`` spec.

    Samples the eight free parameters from exactly the densities
    ``PhiPrior.log_prior`` evaluates, so that ``(phi, data)`` pairs generated
    with ``generate_prior_predictive_market`` are true joint draws — the
    validity precondition for simulation-based calibration. The remaining
    ModelParams fields (``a``, ``b``, ``gamma``, ``s0_2``) are fixed
    hyperparameters, not inferred, and keep their dataclass defaults.

    RNG calls are made in the canonical ``log_prior`` order —
    ``sigma2_0, sigma2_1, q_01, q_10, beta_S, beta_Z, tau2_0, tau2_1`` — eight
    draws per call. Reordering them changes the realization for a given seed,
    matching the fixed-order convention ``generate_market`` documents.

      * ``sigma2_v ~ InvGamma(sigma2_ig_alpha, sigma2_ig_beta)``
      * ``q_0j ~ Beta(q_beta_a, q_beta_b)``
      * ``beta_S, beta_Z ~ Cauchy(0, beta_cauchy_scale)``
      * ``tau2_z ~ InvGamma(tau2_ig_alpha, tau2_ig_beta)``

    The Cauchy draws are deliberately **untruncated**: clipping or rejecting
    heavy tails would make the sampling density differ from the density the
    posterior is scored against, which destroys SBC rank uniformity. Callers
    must tolerate occasional very large |beta| values.

    Args:
        prior: Prior spec supplying every hyperparameter; both Inverse-Gamma
            blocks must be proper (see Raises).
        rng: Random generator; passed explicitly so callers control the seed.

    Returns:
        A ModelParams whose eight sampled fields are a single joint prior draw.

    Raises:
        ValueError: If either Inverse-Gamma block has a shape or scale below
            ``_MIN_IG_HYPERPARAM``. The shipped ``PhiPrior()`` defaults are such
            a case (``tau2`` is IG(1e-9, 1e-9), a placeholder that exists only to
            regularize the M-step), so simulation callers must pass a PhiPrior
            with proper variance hyperparameters rather than the default.
    """
    _check_proper_ig("sigma2", prior.sigma2_ig_alpha, prior.sigma2_ig_beta)
    _check_proper_ig("tau2", prior.tau2_ig_alpha, prior.tau2_ig_beta)

    sigma2_0 = _draw_inverse_gamma(prior.sigma2_ig_alpha, prior.sigma2_ig_beta, rng)
    sigma2_1 = _draw_inverse_gamma(prior.sigma2_ig_alpha, prior.sigma2_ig_beta, rng)
    q_01 = float(rng.beta(prior.q_beta_a, prior.q_beta_b))
    q_10 = float(rng.beta(prior.q_beta_a, prior.q_beta_b))
    beta_S = float(rng.standard_cauchy() * prior.beta_cauchy_scale)
    beta_Z = float(rng.standard_cauchy() * prior.beta_cauchy_scale)
    tau2_0 = _draw_inverse_gamma(prior.tau2_ig_alpha, prior.tau2_ig_beta, rng)
    tau2_1 = _draw_inverse_gamma(prior.tau2_ig_alpha, prior.tau2_ig_beta, rng)

    return ModelParams(
        sigma2_0=sigma2_0,
        sigma2_1=sigma2_1,
        q_01=q_01,
        q_10=q_10,
        beta_S=beta_S,
        beta_Z=beta_Z,
        tau2_0=tau2_0,
        tau2_1=tau2_1,
    )


# ---------------- Market simulation ----------------


def generate_market(
    params: ModelParams,
    *,
    n_trades: int = 500,
    n_wallets: int = 50,
    n_insider_wallets: int = 5,
    mean_inter_trade_time: float = 300.0,  # seconds; Exponential rate
    log_size_mean: float = 4.0,  # log-USDC; mean size ~ $55
    log_size_std: float = 1.5,
    rng: np.random.Generator,
) -> SyntheticMarket:
    """Draw one synthetic market from the generative model (§5).

    Simulates in sequence: wallet propensities θ_w, trade timestamps, trade
    sizes, wallet assignments, the latent state path (V, X, Z), and finally
    noisy logit-price observations Y. RNG calls are made in this fixed order
    — reordering them changes the realization even with the same seed.

    Args:
        params: Model hyperparameters; all variance fields must be non-NaN.
        n_trades: Number of trades T to simulate.
        n_wallets: Total number of wallets in the market.
        n_insider_wallets: Wallets [0, n_insider_wallets) are forced to high
            propensity via Beta(9, 1) (mean 0.9) and are up-weighted 3x in the
            wallet assignment. Pass 0 (or use
            ``generate_prior_predictive_market``) for an unplanted market.
        mean_inter_trade_time: Mean of the Exponential inter-trade gap in
            seconds (delta[1:] ~ Exp(1/mean_inter_trade_time)).
        log_size_mean: Mean of the log-normal trade size distribution (log-USDC).
        log_size_std: Std dev of the log-normal trade size distribution.
        rng: Random generator; passed explicitly so callers control the seed.

    Returns:
        SyntheticMarket with ground-truth latents and noisy observations.
    """
    T = n_trades

    # --- Wallet propensities ---
    # Regular wallets drawn from the prior; insider wallets forced to high propensity
    theta_w = rng.beta(params.a, params.b, size=n_wallets)
    insider_wallet_ids = list(range(n_insider_wallets))
    for w in insider_wallet_ids:
        theta_w[w] = rng.beta(9.0, 1.0)  # Beta(9,1) has mean 0.9

    # --- Trade times ---
    delta = np.zeros(T)
    delta[1:] = rng.exponential(mean_inter_trade_time, size=T - 1)
    t = np.cumsum(delta)

    # --- Trade sizes (lognormal) ---
    S = np.exp(rng.normal(log_size_mean, log_size_std, size=T))
    S_bar = float(S.mean())
    log_size_ratio = np.log(S / S_bar)  # shape (T,)

    # --- Wallet assignments ---
    # Insider wallets trade 3x more often to make them identifiable
    wallet_weights = np.ones(n_wallets)
    for w in insider_wallet_ids:
        wallet_weights[w] = 3.0
    wallet_weights /= wallet_weights.sum()
    wallet_ids = rng.choice(n_wallets, size=T, p=wallet_weights)

    # --- Latent state generation ---
    X = np.empty(T)
    V = np.empty(T, dtype=np.int8)
    Z = np.empty(T, dtype=np.int8)

    # Initialise at stationary distribution of regime Markov chain
    rho_V = params.q_01 / (params.q_01 + params.q_10)
    V[0] = int(rng.random() < rho_V)
    X[0] = rng.normal(0.0, np.sqrt(params.s0_2))
    Z[0] = 0

    sigma2_by_regime = np.array([params.sigma2_0, params.sigma2_1])
    logit_theta = logit(theta_w)  # pre-compute; shape (n_wallets,)

    for i in range(1, T):
        # Volatility regime — flip with row-dependent probability
        flip_prob = params.q_01 if V[i - 1] == 0 else params.q_10
        V[i] = (1 - V[i - 1]) if (rng.random() < flip_prob) else V[i - 1]

        # Latent logit-probability — Gaussian random walk
        X[i] = rng.normal(X[i - 1], np.sqrt(sigma2_by_regime[V[i]] * delta[i]))

        # Insider indicator
        logit_pi_Z = (
            logit_theta[wallet_ids[i]]
            + params.beta_S * log_size_ratio[i]
            + params.beta_Z * float(Z[i - 1])
        )
        pi_Z = float(sigmoid(np.asarray(logit_pi_Z)))
        Z[i] = int(rng.random() < pi_Z)

    # --- Observation model ---
    tau2_Z = np.where(Z == 0, params.tau2_0, params.tau2_1)
    # Floor denominator to avoid near-zero or negative variance for tiny trades.
    # Mirrors kalman._DENOM_FLOOR = 0.1 so synthetic and real paths are consistent.
    denom = np.maximum(1.0 + params.gamma * log_size_ratio, 0.1)
    obs_std = np.sqrt(tau2_Z / denom)
    Y = rng.normal(X, obs_std)
    p = sigmoid(Y)

    return SyntheticMarket(
        X=X,
        V=V,
        Z=Z,
        theta_w=theta_w,
        Y=Y,
        p=p,
        S=S,
        S_bar=S_bar,
        t=t,
        delta=delta,
        wallet_ids=wallet_ids,
        insider_wallet_ids=insider_wallet_ids,
    )


def generate_prior_predictive_market(
    params: ModelParams,
    *,
    rng: np.random.Generator,
    **market_kwargs,
) -> SyntheticMarket:
    """Draw one market with no planted insiders — the SBC generation mode.

    Thin named entry point onto ``generate_market`` with
    ``n_insider_wallets=0``: *every* wallet's propensity comes from the model's
    own Beta(``params.a``, ``params.b``) prior and wallets are assigned to
    trades uniformly, with no 3x frequency up-weighting. That makes the market
    an exact draw from ``p(latents, data | phi)``; the planted-insider default
    is not, because forcing Beta(9, 1) propensities and skewing trade counts
    tilts the generating density away from the one inference assumes.

    Args:
        params: Model hyperparameters, typically from ``params_from_prior``.
        rng: Random generator; passed explicitly so callers control the seed.
        **market_kwargs: Forwarded to ``generate_market`` (n_trades, n_wallets,
            mean_inter_trade_time, ...). ``n_insider_wallets`` is fixed at 0 and
            may not be overridden.

    Returns:
        A SyntheticMarket with ``insider_wallet_ids == []``.

    Raises:
        ValueError: If ``n_insider_wallets`` is passed — planting insiders would
            silently break the prior-predictive property this entry point exists
            to guarantee.
    """
    if "n_insider_wallets" in market_kwargs:
        raise ValueError(
            "generate_prior_predictive_market plants no insiders; "
            "n_insider_wallets cannot be overridden. Call generate_market "
            "directly for planted-insider markets."
        )
    return generate_market(params, n_insider_wallets=0, rng=rng, **market_kwargs)


def generate_dataset(
    params: ModelParams,
    *,
    n_markets: int = 5,
    rng: np.random.Generator,
    **market_kwargs,
) -> list[SyntheticMarket]:
    """Draw K independent synthetic markets sharing one RNG stream.

    Args:
        params: Model hyperparameters forwarded to each ``generate_market`` call.
        n_markets: Number of markets K to simulate.
        rng: Random generator; advanced sequentially across all K markets so
            the seed controls the entire dataset.
        **market_kwargs: Forwarded to ``generate_market`` (n_trades,
            n_wallets, etc.).

    Returns:
        list of K SyntheticMarket objects in simulation order.
    """
    return [generate_market(params, rng=rng, **market_kwargs) for _ in range(n_markets)]
