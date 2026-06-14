from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.default_params import ModelParams
from src.utils.transforms import logit, sigmoid


@dataclass
class SyntheticMarket:
    # Ground-truth latent variables (available only in synthetic experiments)
    X: np.ndarray          # (T,) logit of true public-info probability
    V: np.ndarray          # (T,) int8 volatility regime {0, 1}
    Z: np.ndarray          # (T,) int8 insider indicator {0, 1}
    theta_w: np.ndarray    # (n_wallets,) true wallet propensities

    # Observations
    Y: np.ndarray          # (T,) logit-price observations
    p: np.ndarray          # (T,) trade prices = sigmoid(Y)
    S: np.ndarray          # (T,) trade sizes in USDC
    S_bar: float           # within-market mean size (used to normalise log-size ratios)

    # Trade metadata
    t: np.ndarray          # (T,) trade timestamps in seconds
    delta: np.ndarray      # (T,) inter-trade times; delta[0] = 0 (sentinel)
    wallet_ids: np.ndarray # (T,) integer wallet index for each trade

    # Which wallet indices were injected as insiders
    insider_wallet_ids: list[int]


def generate_market(
    params: ModelParams,
    *,
    n_trades: int = 500,
    n_wallets: int = 50,
    n_insider_wallets: int = 5,
    mean_inter_trade_time: float = 300.0,  # seconds; Exponential rate
    log_size_mean: float = 4.0,            # log-USDC; mean size ~ $55
    log_size_std: float = 1.5,
    rng: np.random.Generator,
) -> SyntheticMarket:
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
    # Floor denominator to avoid near-zero or negative variance for tiny trades
    denom = np.maximum(1.0 + params.gamma * log_size_ratio, 0.1)
    obs_std = np.sqrt(tau2_Z / denom)
    Y = rng.normal(X, obs_std)
    p = sigmoid(Y)

    return SyntheticMarket(
        X=X, V=V, Z=Z, theta_w=theta_w,
        Y=Y, p=p, S=S, S_bar=S_bar,
        t=t, delta=delta, wallet_ids=wallet_ids,
        insider_wallet_ids=insider_wallet_ids,
    )


def generate_dataset(
    params: ModelParams,
    *,
    n_markets: int = 5,
    rng: np.random.Generator,
    **market_kwargs,
) -> list[SyntheticMarket]:
    return [
        generate_market(params, rng=rng, **market_kwargs)
        for _ in range(n_markets)
    ]
