"""Vanilla Particle Gibbs sampler.

Outer loop per iteration:
  1. For each market k:
       - Run CSMC pinned to the current reference (V_k, Z_k).
       - Sample a new reference path from the SMC posterior.
       - Draw X_k via FFBS conditional on the new (V_k, Z_k).
  2. Gibbs sweep on the parameter block using the just-sampled latents.

Iteration-0 reference: one bootstrap-SMC pass per market under the warm-start
parameters and a θ_w prior draw (decision #10), then `sample_path`. The
caller can override with `V_ref_init` / `Z_ref_init` for hot starts.

Decision #11's adaptive MH step-size tuning runs in the early burn-in window:
every 20 iterations until min(100, n_burnin/2), recent acceptance rates push
the step sizes by ×1.2 or ÷1.2 toward the 23–44% band. The mutation is
applied to a local copy of `InferenceConfig`, so the caller's config is
untouched after the call.

Naive index-pinning (decision #4) is inherited from CSMC; ancestor sampling
for the reference particle is iPMCMC's job.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from config.default_params import InferenceConfig, ModelParams
from src.inference.csmc import conditional_smc
from src.inference.kalman import ffbs_sample
from src.inference.parameter_updates import MarketLatents, adapt_mh_step, gibbs_sweep
from src.inference.smc import bootstrap_smc, sample_path


@dataclass
class MarketData:
    """One market's observations conditioned on by Particle Gibbs.

    Attributes:
        Y: Logit-transformed prices in time order.
        delta: Inter-trade seconds with `delta[0] == 0`.
        log_size_ratio: Per-trade `log(S / S_bar)` feature.
        wallet_ids: Integer wallet index per trade.
    """

    Y: np.ndarray
    delta: np.ndarray
    log_size_ratio: np.ndarray
    wallet_ids: np.ndarray

    @property
    def T(self) -> int:
        """Return the number of trades in this market."""
        return len(self.Y)


@dataclass
class PGOutput:
    """Container for Particle Gibbs chains and diagnostics.

    All arrays include every iteration (`n_iter`) with burn-in still present.
    Callers apply burn-in slicing downstream.
    """

    # Parameter chains (n_iter,)
    sigma2_0: np.ndarray
    sigma2_1: np.ndarray
    q_01: np.ndarray
    q_10: np.ndarray
    beta_S: np.ndarray
    beta_Z: np.ndarray
    tau2_0: np.ndarray
    tau2_1: np.ndarray
    # Hierarchical wallet effects (n_iter, n_wallets)
    theta_w: np.ndarray
    # Per-market latent chains: list of (n_iter, T_k) arrays
    X: list[np.ndarray]
    V: list[np.ndarray]
    Z: list[np.ndarray]
    # CSMC log-marginal estimate per market per iteration (n_iter, K)
    log_marg: np.ndarray
    # MH acceptance flags (n_iter,) bool
    acc_beta_S: np.ndarray
    acc_beta_Z: np.ndarray
    acc_tau2_0: np.ndarray
    acc_tau2_1: np.ndarray
    # Final adapted step sizes (for diagnostics + carry-over)
    final_mh_step_beta_S: float
    final_mh_step_beta_Z: float
    final_mh_step_log_tau2_0: float
    final_mh_step_log_tau2_1: float


def _csmc_then_ffbs(
    md: MarketData,
    theta_w: np.ndarray,
    params: ModelParams,
    config: InferenceConfig,
    V_ref: np.ndarray,
    Z_ref: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One CSMC pass → sample new reference → FFBS for X."""
    out = conditional_smc(
        md.Y,
        md.delta,
        md.log_size_ratio,
        md.wallet_ids,
        theta_w,
        params,
        config,
        V_ref,
        Z_ref,
        rng=rng,
    )
    V_new, Z_new = sample_path(out, rng)
    X_new = ffbs_sample(
        md.Y,
        V_new,
        Z_new,
        md.delta,
        md.log_size_ratio,
        params,
        rng,
    )
    return V_new, Z_new, X_new, out.log_marginal


def _filter_screen_worker(
    md: MarketData,
    theta_w: np.ndarray,
    params: ModelParams,
    config: InferenceConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run bootstrap_smc on one market and return Z_prob_filt."""
    out = bootstrap_smc(
        md.Y,
        md.delta,
        md.log_size_ratio,
        md.wallet_ids,
        theta_w,
        params,
        config,
        rng=rng,
    )
    return out.Z_prob_filt


def filter_screen(
    markets: list[MarketData],
    params: ModelParams,
    theta_w: np.ndarray,
    config: InferenceConfig,
    *,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> np.ndarray:
    """Fast filter-only screening pass: one SMC filter per market, no MCMC.

    Runs bootstrap SMC on each market with the given (fixed) parameters and
    theta_w, returning per-wallet aggregate Z_prob scores suitable for ranking
    wallets by insider propensity without a full MCMC run.

    This is the "fast shortlist" tier: run filter_screen first, then run
    full particle_gibbs only on the flagged wallets/markets.

    Args:
        markets: List of K markets.
        params: Fixed model parameters (e.g. warm-start or PG posterior mean).
        theta_w: Fixed per-wallet insider propensities, shape (n_wallets,).
        config: InferenceConfig; uses N and ess_resample_threshold.
        rng: Random generator.
        n_jobs: joblib workers for market parallelism (1 = sequential).

    Returns:
        wallet_scores: shape (n_wallets,) — mean Z_prob aggregated over all
            trades belonging to each wallet across all markets. Wallets with
            no trades get score 0.0.
    """
    K = len(markets)
    n_wallets = len(theta_w)
    wallet_sum = np.zeros(n_wallets)
    wallet_count = np.zeros(n_wallets, dtype=int)

    if n_jobs == 1:
        for md in markets:
            out = bootstrap_smc(
                md.Y,
                md.delta,
                md.log_size_ratio,
                md.wallet_ids,
                theta_w,
                params,
                config,
                rng=rng,
            )
            wallet_sum += np.bincount(md.wallet_ids, weights=out.Z_prob_filt, minlength=n_wallets)
            wallet_count += np.bincount(md.wallet_ids, minlength=n_wallets)
    else:
        child_rngs = [
            np.random.default_rng(s)
            for s in np.random.SeedSequence(rng.integers(2**63)).spawn(K)
        ]
        z_probs = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_filter_screen_worker)(markets[k], theta_w, params, config, child_rngs[k])
            for k in range(K)
        )
        for k, z_prob_filt in enumerate(z_probs):
            md = markets[k]
            wallet_sum += np.bincount(md.wallet_ids, weights=z_prob_filt, minlength=n_wallets)
            wallet_count += np.bincount(md.wallet_ids, minlength=n_wallets)

    return np.where(wallet_count > 0, wallet_sum / wallet_count, 0.0)


def particle_gibbs(
    markets: list[MarketData],
    config: InferenceConfig,
    *,
    rng: np.random.Generator,
    n_wallets: int | None = None,
    params_init: ModelParams | None = None,
    theta_w_init: np.ndarray | None = None,
    V_ref_init: list[np.ndarray] | None = None,
    Z_ref_init: list[np.ndarray] | None = None,
    adapt_step_sizes: bool = True,
    progress: bool = False,
) -> PGOutput:
    """Run vanilla Particle Gibbs.

    Args:
        markets: list of MarketData (K markets; K = 1 is fine).
        config: InferenceConfig. `N`, `ess_resample_threshold`, `n_iter`,
            `n_burnin`, and `mh_step_*` are all consumed.
        rng: explicit Generator (§7.1). All randomness — SMC, FFBS, MH —
            consumes from this Generator in a deterministic order.
        n_wallets: override; defaults to max(wallet_ids)+1 across markets.
        params_init: optional warm start; defaults to
            `ModelParams.warm_start(concat(Y_k))`.
        theta_w_init: optional; defaults to `Beta(a, b)` i.i.d. (decision #10).
        V_ref_init, Z_ref_init: optional per-market initial reference
            trajectories. If omitted, one `bootstrap_smc` + `sample_path` pass
            per market provides the seed.
        adapt_step_sizes: whether to run the windowed step-size adaptation
            during early burn-in (decision #11).
        progress: show a tqdm bar.

    Returns:
        Full Particle Gibbs output with parameter chains, latent trajectories,
        marginal likelihood diagnostics, MH acceptance flags, and final adapted
        step sizes.
    """
    # Local copy so we don't mutate the caller's config during adaptation
    config = copy.copy(config)

    if n_wallets is None:
        n_wallets = int(max(int(m.wallet_ids.max()) for m in markets)) + 1

    if params_init is None:
        Y_concat = np.concatenate([m.Y for m in markets])
        params = ModelParams.warm_start(Y_concat)
    else:
        params = params_init

    if theta_w_init is None:
        theta_w = rng.beta(params.a, params.b, size=n_wallets)
    else:
        theta_w = np.array(theta_w_init, copy=True)

    # Seed references via one bootstrap pass per market
    if V_ref_init is None or Z_ref_init is None:
        V_refs: list[np.ndarray] = []
        Z_refs: list[np.ndarray] = []
        for md in markets:
            out0 = bootstrap_smc(
                md.Y,
                md.delta,
                md.log_size_ratio,
                md.wallet_ids,
                theta_w,
                params,
                config,
                rng=rng,
            )
            V_p, Z_p = sample_path(out0, rng)
            V_refs.append(V_p)
            Z_refs.append(Z_p)
    else:
        V_refs = [np.array(v, copy=True) for v in V_ref_init]
        Z_refs = [np.array(z, copy=True) for z in Z_ref_init]

    K = len(markets)
    n_iter = config.n_iter

    # Allocate output buffers
    sigma2_0 = np.empty(n_iter)
    sigma2_1 = np.empty(n_iter)
    q_01 = np.empty(n_iter)
    q_10 = np.empty(n_iter)
    beta_S = np.empty(n_iter)
    beta_Z = np.empty(n_iter)
    tau2_0 = np.empty(n_iter)
    tau2_1 = np.empty(n_iter)
    theta_w_chain = np.empty((n_iter, n_wallets))
    log_marg = np.empty((n_iter, K))
    acc_beta_S = np.empty(n_iter, dtype=bool)
    acc_beta_Z = np.empty(n_iter, dtype=bool)
    acc_tau2_0 = np.empty(n_iter, dtype=bool)
    acc_tau2_1 = np.empty(n_iter, dtype=bool)
    X_chains = [np.empty((n_iter, m.T)) for m in markets]
    V_chains = [np.empty((n_iter, m.T), dtype=np.int8) for m in markets]
    Z_chains = [np.empty((n_iter, m.T), dtype=np.int8) for m in markets]

    # Adaptive tuning window: stop after min(100, n_burnin/2), with floor 20
    adapt_window = 20
    adapt_until = min(100, max(adapt_window, config.n_burnin // 2))

    iterator = range(n_iter)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="PG")

    for it in iterator:
        # ----- CSMC + FFBS per market -----
        latents: list[MarketLatents] = []
        if config.n_jobs == 1:
            for k, md in enumerate(markets):
                V_new, Z_new, X_new, lm = _csmc_then_ffbs(
                    md,
                    theta_w,
                    params,
                    config,
                    V_refs[k],
                    Z_refs[k],
                    rng,
                )
                V_refs[k] = V_new
                Z_refs[k] = Z_new
                log_marg[it, k] = lm
                X_chains[k][it] = X_new
                V_chains[k][it] = V_new
                Z_chains[k][it] = Z_new
                latents.append(
                    MarketLatents(
                        Y=md.Y,
                        delta=md.delta,
                        log_size_ratio=md.log_size_ratio,
                        wallet_ids=md.wallet_ids,
                        X=X_new,
                        V=V_new,
                        Z=Z_new,
                    )
                )
        else:
            iter_seed = int(rng.integers(2**63))
            ss = np.random.SeedSequence(iter_seed)
            child_seeds = ss.spawn(K)
            child_rngs = [np.random.default_rng(s) for s in child_seeds]
            results = Parallel(n_jobs=config.n_jobs, prefer="processes")(
                delayed(_csmc_then_ffbs)(
                    markets[k], theta_w, params, config, V_refs[k], Z_refs[k], child_rngs[k]
                )
                for k in range(K)
            )
            for k, (V_new, Z_new, X_new, lm) in enumerate(results):
                V_refs[k] = V_new
                Z_refs[k] = Z_new
                log_marg[it, k] = lm
                X_chains[k][it] = X_new
                V_chains[k][it] = V_new
                Z_chains[k][it] = Z_new
                latents.append(
                    MarketLatents(
                        Y=markets[k].Y,
                        delta=markets[k].delta,
                        log_size_ratio=markets[k].log_size_ratio,
                        wallet_ids=markets[k].wallet_ids,
                        X=X_new,
                        V=V_new,
                        Z=Z_new,
                    )
                )

        # ----- Gibbs sweep on parameters -----
        params, theta_w, diag = gibbs_sweep(params, theta_w, latents, config, rng)

        sigma2_0[it] = params.sigma2_0
        sigma2_1[it] = params.sigma2_1
        q_01[it] = params.q_01
        q_10[it] = params.q_10
        beta_S[it] = params.beta_S
        beta_Z[it] = params.beta_Z
        tau2_0[it] = params.tau2_0
        tau2_1[it] = params.tau2_1
        theta_w_chain[it] = theta_w
        acc_beta_S[it] = diag.acc_beta_S
        acc_beta_Z[it] = diag.acc_beta_Z
        acc_tau2_0[it] = diag.acc_tau2_0
        acc_tau2_1[it] = diag.acc_tau2_1

        # ----- Adaptive step-size tuning -----
        if adapt_step_sizes and 0 < it < adapt_until and (it + 1) % adapt_window == 0:
            lo, hi = it + 1 - adapt_window, it + 1
            adapt_mh_step(config, "mh_step_beta_S", float(acc_beta_S[lo:hi].mean()))
            adapt_mh_step(config, "mh_step_beta_Z", float(acc_beta_Z[lo:hi].mean()))
            adapt_mh_step(
                config,
                "mh_step_log_tau2_0",
                float(acc_tau2_0[lo:hi].mean()),
            )
            adapt_mh_step(
                config,
                "mh_step_log_tau2_1",
                float(acc_tau2_1[lo:hi].mean()),
            )

    return PGOutput(
        sigma2_0=sigma2_0,
        sigma2_1=sigma2_1,
        q_01=q_01,
        q_10=q_10,
        beta_S=beta_S,
        beta_Z=beta_Z,
        tau2_0=tau2_0,
        tau2_1=tau2_1,
        theta_w=theta_w_chain,
        X=X_chains,
        V=V_chains,
        Z=Z_chains,
        log_marg=log_marg,
        acc_beta_S=acc_beta_S,
        acc_beta_Z=acc_beta_Z,
        acc_tau2_0=acc_tau2_0,
        acc_tau2_1=acc_tau2_1,
        final_mh_step_beta_S=config.mh_step_beta_S,
        final_mh_step_beta_Z=config.mh_step_beta_Z,
        final_mh_step_log_tau2_0=config.mh_step_log_tau2_0,
        final_mh_step_log_tau2_1=config.mh_step_log_tau2_1,
    )
