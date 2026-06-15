"""Synthetic data generator for PMCMC validation experiments.

Simulates the full latent-variable model (§5): a regime-switching Gaussian
random walk for log-odds price X, a logistic insider indicator Z driven by
per-wallet propensities θ_w, and a heteroskedastic observation model whose
variance is scaled by trade size.

``SyntheticMarket`` (returned by ``generate_market``) mirrors the
``ProcessedMarket`` interface so that all downstream inference and plotting
code works unchanged on both real and synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.default_params import ModelParams
from src.utils.transforms import logit, sigmoid


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
            propensity via Beta(9, 1) (mean 0.9).
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
