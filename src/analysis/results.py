"""Posterior summaries for PG / iPMCMC outputs (§3.5 quantities of interest).

Four headline quantities are produced here:

  1. P(Z_i = 1 | D)       — per-trade insider probability
  2. E[π_{t_i} | D]       — smoothed price track on the probability scale
  3. P(V_{t_i} = 1 | D)   — regime indicator
  4. E[θ_w | D]           — per-wallet posterior, returned as a ranked table

Plus the synthetic-validation primitives (`roc_auc`, `roc_curve`) used by §9 of
the paper. Everything works uniformly on `PGOutput` and `iPMCMCOutput`: for
the latter we flatten the leading (n_iter, P) axes into one Monte-Carlo axis
of size n_iter*P after dropping burn-in. iPMCMC chains are exchangeable post-
burn-in (decision #13), so pooling is the natural thing.

`summarize_chain` returns a per-parameter table including R-hat / ESS from
`src.inference.diagnostics`; for PG outputs R-hat is reported as NaN (only one
chain per run) — pool multiple PG runs along an explicit chain axis if you
want a multi-chain R-hat.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy.special import expit

from src.data.preprocess import WalletIndex
from src.inference.diagnostics import (
    PHI_PARAM_NAMES,
    compute_ess,
    compute_rhat,
)
from src.inference.ipmcmc import iPMCMCOutput
from src.inference.particle_gibbs import PGOutput

PGorIP = Union[PGOutput, iPMCMCOutput]


# ---------------- Internal helpers ----------------

def _is_ipmcmc(out: PGorIP) -> bool:
    return isinstance(out, iPMCMCOutput)


def _flatten_param(samples: np.ndarray, *, is_ipmcmc: bool) -> np.ndarray:
    """Drop the chain axis for iPMCMC; identity for PG.

    PG: (n_iter, *extra) → (n_iter, *extra)
    iPMCMC: (n_iter, P, *extra) → (n_iter*P, *extra)
    """
    if not is_ipmcmc:
        return samples
    return samples.reshape(-1, *samples.shape[2:])


def _param_samples(out: PGorIP, name: str, n_burnin: int) -> np.ndarray:
    s = np.asarray(getattr(out, name))[n_burnin:]
    return _flatten_param(s, is_ipmcmc=_is_ipmcmc(out))


def _market_samples(
    out: PGorIP, name: str, market_idx: int, n_burnin: int,
) -> np.ndarray:
    """Per-market latent samples, burn-in dropped, chain axis collapsed."""
    s = np.asarray(getattr(out, name)[market_idx])[n_burnin:]
    return _flatten_param(s, is_ipmcmc=_is_ipmcmc(out))


# ---------------- Per-trade quantities (§3.5 #1–#4) ----------------

def posterior_Z_probability(
    out: PGorIP, market_idx: int = 0, n_burnin: int = 0,
) -> np.ndarray:
    """P(Z_i = 1 | D) per trade — the headline anomaly score (§3.5 #1)."""
    Z = _market_samples(out, "Z", market_idx, n_burnin)
    return Z.mean(axis=0).astype(float)


def posterior_regime_probability(
    out: PGorIP, market_idx: int = 0, n_burnin: int = 0,
) -> np.ndarray:
    """P(V_i = 1 | D) per trade (§3.5 #4)."""
    V = _market_samples(out, "V", market_idx, n_burnin)
    return V.mean(axis=0).astype(float)


def posterior_pi_mean(
    out: PGorIP, market_idx: int = 0, n_burnin: int = 0,
) -> np.ndarray:
    """E[π_{t_i} | D] on the probability scale (§3.5 #2).

    Computed as the Monte-Carlo mean of sigmoid(X) over post-burn-in samples;
    not sigmoid(mean(X)) — the two differ whenever the posterior on X is wide.
    """
    X = _market_samples(out, "X", market_idx, n_burnin)
    return expit(X).mean(axis=0)


def posterior_X_mean(
    out: PGorIP, market_idx: int = 0, n_burnin: int = 0,
) -> np.ndarray:
    """E[X_{t_i} | D] in logit space."""
    X = _market_samples(out, "X", market_idx, n_burnin)
    return X.mean(axis=0)


def credible_interval(
    samples: np.ndarray, *, alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column (lower, upper) percentile bounds for a (1−α) CI."""
    lo = np.percentile(samples, 100 * alpha / 2.0, axis=0)
    hi = np.percentile(samples, 100 * (1.0 - alpha / 2.0), axis=0)
    return lo, hi


def flagged_trade_indices(
    z_prob: np.ndarray, *, threshold: float = 0.5,
) -> np.ndarray:
    """Indices i where P(Z_i = 1 | D) ≥ threshold."""
    return np.flatnonzero(np.asarray(z_prob) >= threshold)


# ---------------- Wallet ranking (§3.5 #3) ----------------

def wallet_ranking(
    out: PGorIP,
    wallet_index: WalletIndex,
    *,
    n_burnin: int = 0,
    n_trades_per_wallet: dict[int, int] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Posterior summary for θ_w, ranked by E[θ_w | D] descending (§3.5 #3).

    Args:
        out: PG or iPMCMC output containing the θ_w chain.
        wallet_index: maps id → address (needed to label the table).
        n_burnin: iterations to drop.
        n_trades_per_wallet: optional {wallet_id: trade_count} to annotate
            ranking with how much evidence each wallet contributes. Pulled
            from `MarketData.wallet_ids` across markets by `count_wallet_trades`.
        alpha: credible-interval coverage = 1 − α.

    Returns:
        DataFrame with columns: wallet_id, wallet_address, posterior_mean,
        posterior_median, ci_lo, ci_hi, n_trades.
    """
    theta = _param_samples(out, "theta_w", n_burnin)   # (n_total, n_wallets)
    mean = theta.mean(axis=0)
    median = np.median(theta, axis=0)
    lo, hi = credible_interval(theta, alpha=alpha)

    id_to_addr = {wid: addr for addr, wid in wallet_index.address_to_id.items()}
    n_wallets = theta.shape[1]
    rows = []
    for w in range(n_wallets):
        rows.append({
            "wallet_id": w,
            "wallet_address": id_to_addr.get(w, ""),
            "posterior_mean": float(mean[w]),
            "posterior_median": float(median[w]),
            "ci_lo": float(lo[w]),
            "ci_hi": float(hi[w]),
            "n_trades": int((n_trades_per_wallet or {}).get(w, 0)),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("posterior_mean", ascending=False).reset_index(drop=True)
    return df


def count_wallet_trades(
    wallet_ids_per_market: list[np.ndarray],
    *,
    n_wallets: int | None = None,
) -> dict[int, int]:
    """Trade-count per wallet, pooled across markets (i=0 excluded — see §6
    note that Z_0 := 0 is deterministic, so wallet's first trade contributes
    no posterior info to θ_w)."""
    if n_wallets is None:
        n_wallets = int(max(int(w.max()) for w in wallet_ids_per_market)) + 1
    counts = np.zeros(n_wallets, dtype=int)
    for w in wallet_ids_per_market:
        if len(w) > 1:
            counts += np.bincount(w[1:].astype(np.int64), minlength=n_wallets)
    return {int(i): int(c) for i, c in enumerate(counts)}


# ---------------- Chain-level summary (§5 paper table) ----------------

def summarize_chain(
    out: PGorIP, *, n_burnin: int = 0, alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-φ summary table: mean, median, CI, ESS, R-hat.

    R-hat is computed across the P chains for iPMCMC and reported as NaN for
    PG (single chain).
    """
    rows = []
    is_ip = _is_ipmcmc(out)
    for name in PHI_PARAM_NAMES:
        raw = np.asarray(getattr(out, name))[n_burnin:]   # PG: (n,) | iPMCMC: (n, P)
        flat = _flatten_param(raw, is_ipmcmc=is_ip)
        ess = float(compute_ess(flat))
        rhat = float(compute_rhat(raw)) if is_ip else float("nan")
        lo, hi = credible_interval(flat, alpha=alpha)
        rows.append({
            "parameter": name,
            "posterior_mean": float(flat.mean()),
            "posterior_median": float(np.median(flat)),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "ess": ess,
            "rhat": rhat,
        })
    return pd.DataFrame(rows)


# ---------------- Synthetic-validation metrics (§9) ----------------

def roc_auc(z_true: np.ndarray, z_score: np.ndarray) -> float:
    """Rank-sum AUC of (z_score) against binary (z_true).

    Returns 0.5 if either class is empty. Vectorized; suitable for the §9
    headline metric on synthetic insider-injection runs.
    """
    z_true = np.asarray(z_true).astype(int)
    z_score = np.asarray(z_score).astype(float)
    n_pos = int(z_true.sum())
    n_neg = len(z_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(z_score, kind="mergesort")
    ranks = np.empty(len(z_score), dtype=float)
    ranks[order] = np.arange(1, len(z_score) + 1)
    # Average ranks for ties (only matters if z_score has duplicates)
    _, inv, counts = np.unique(z_score, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=float)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    rank_sum_pos = float(ranks[z_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve(
    z_true: np.ndarray, z_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical ROC curve.

    Returns (fpr, tpr, thresholds), all of length k+1 where k is the number
    of distinct z_score values. Includes the (0, 0) origin.
    """
    z_true = np.asarray(z_true).astype(int)
    z_score = np.asarray(z_score).astype(float)
    n_pos = int(z_true.sum())
    n_neg = len(z_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    order = np.argsort(-z_score, kind="mergesort")
    z_sorted = z_score[order]
    y_sorted = z_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    # Collapse duplicate scores into single threshold points
    distinct = np.r_[np.diff(z_sorted) != 0, True]
    tps = tps[distinct]
    fps = fps[distinct]
    thresholds = z_sorted[distinct]
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[thresholds[0] + 1.0, thresholds]
    return fpr, tpr, thresholds
