"""Interacting Particle Markov Chain Monte Carlo (Rainforth et al., 2016)
extended to joint (path, θ) inference.

Maintains P parallel parameter chains (decision #5 / #13: M=8, P=4) plus
M − P unconditional auxiliary chains per iteration. The unconditional chains
inject diverse path candidates that the conditional slots can swap into,
breaking the path-degeneracy that vanilla PG suffers from.

Per iteration:
  1. Run M SMC passes (P conditional CSMC + M − P bootstrap SMC). Each
     unconditional chain m ∈ {P..M−1} borrows (θ, θ_w) from slot (m−P) mod P,
     so every chain has a well-defined parameter set.
  2. Swap step: for each slot j ∈ {0..P−1}, candidates = {j} ∪ {P..M−1};
     sample the new reference source ∝ exp(log_marg_total over markets).
  3. Sample a new reference path from the chosen chain's particle ensemble.
  4. FFBS for X | (V, Z) and Gibbs sweep per slot, using slot j's own
     (θ_p, θ_w_p) — the path can have come from a different parameter regime,
     but the next CSMC at slot j will re-weight it correctly.

Decision #4: naive index-pinning inside each CSMC pass.
Decision #11: shared adaptive MH step-size tuning across slots during early
burn-in, applied to a local config copy.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from config.default_params import InferenceConfig, ModelParams
from src.inference.csmc import conditional_smc
from src.inference.kalman import ffbs_sample
from src.inference.parameter_updates import MarketLatents, gibbs_sweep
from src.inference.particle_gibbs import MarketData
from src.inference.smc import bootstrap_smc, sample_path


@dataclass
class iPMCMCOutput:
    """iPMCMC chain output. Parameter arrays have shape (n_iter, P) — one
    column per conditional slot, ready for R-hat across the P chains."""
    # Parameter chains
    sigma2_0: np.ndarray
    sigma2_1: np.ndarray
    q_01: np.ndarray
    q_10: np.ndarray
    beta_S: np.ndarray
    beta_Z: np.ndarray
    tau2_0: np.ndarray
    tau2_1: np.ndarray
    # (n_iter, P, n_wallets)
    theta_w: np.ndarray
    # Per-market latent chains: list of K arrays each (n_iter, P, T_k)
    X: list[np.ndarray]
    V: list[np.ndarray]
    Z: list[np.ndarray]
    # (n_iter, M) marginal-likelihood estimates per SMC pass
    log_marg: np.ndarray
    # (n_iter, P) which chain index supplied each slot's new reference
    chain_indices: np.ndarray
    # (n_iter, P) MH acceptance flags
    acc_beta_S: np.ndarray
    acc_beta_Z: np.ndarray
    acc_tau2_0: np.ndarray
    acc_tau2_1: np.ndarray
    # Final adapted step sizes (shared across slots — see ipmcmc adapt logic)
    final_mh_step_beta_S: float
    final_mh_step_beta_Z: float
    final_mh_step_log_tau2_0: float
    final_mh_step_log_tau2_1: float


def _adapt_step(
    config: InferenceConfig,
    attr: str,
    rate: float,
    *,
    lo: float = 0.23,
    hi: float = 0.44,
    factor: float = 1.2,
) -> None:
    step = getattr(config, attr)
    if rate > hi:
        setattr(config, attr, step * factor)
    elif rate < lo:
        setattr(config, attr, step / factor)


def ipmcmc(
    markets: list[MarketData],
    config: InferenceConfig,
    *,
    rng: np.random.Generator,
    n_wallets: int | None = None,
    params_init: ModelParams | None = None,
    adapt_step_sizes: bool = True,
    progress: bool = False,
) -> iPMCMCOutput:
    """Run iPMCMC with P parallel θ chains and M − P unconditional auxiliaries.

    Args:
        markets: list of MarketData (K markets).
        config: InferenceConfig; uses M, P, N, n_iter, n_burnin, mh_step_*.
        rng: explicit Generator. Single shared rng for full reproducibility.
        n_wallets: override; defaults to max(wallet_ids) + 1 across markets.
        params_init: optional warm start; defaults to
            `ModelParams.warm_start(concat(Y_k))` for all P slots.
        adapt_step_sizes: enable the windowed MH step-size adaptation
            (decision #11) on a local config copy.
        progress: tqdm progress bar.
    """
    if config.M < config.P:
        raise ValueError(f"Need M >= P; got M={config.M}, P={config.P}.")

    config = copy.copy(config)
    M, P = config.M, config.P
    K = len(markets)
    n_iter = config.n_iter

    if n_wallets is None:
        n_wallets = int(max(int(m.wallet_ids.max()) for m in markets)) + 1

    # Initialize P parallel slots
    if params_init is None:
        Y_concat = np.concatenate([m.Y for m in markets])
        params_init = ModelParams.warm_start(Y_concat)
    params_slots = [copy.copy(params_init) for _ in range(P)]
    theta_w_slots = [
        rng.beta(params_init.a, params_init.b, size=n_wallets) for _ in range(P)
    ]

    # Seed initial references for each slot via one bootstrap pass per market
    refs_slots: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for p in range(P):
        slot_refs: list[tuple[np.ndarray, np.ndarray]] = []
        for md in markets:
            out0 = bootstrap_smc(
                md.Y, md.delta, md.log_size_ratio, md.wallet_ids,
                theta_w_slots[p], params_slots[p], config, rng=rng,
            )
            V_p, Z_p = sample_path(out0, rng)
            slot_refs.append((V_p, Z_p))
        refs_slots.append(slot_refs)

    # Output buffers
    sigma2_0 = np.empty((n_iter, P))
    sigma2_1 = np.empty((n_iter, P))
    q_01 = np.empty((n_iter, P))
    q_10 = np.empty((n_iter, P))
    beta_S = np.empty((n_iter, P))
    beta_Z = np.empty((n_iter, P))
    tau2_0 = np.empty((n_iter, P))
    tau2_1 = np.empty((n_iter, P))
    theta_w_chain = np.empty((n_iter, P, n_wallets))
    log_marg = np.empty((n_iter, M))
    chain_indices = np.empty((n_iter, P), dtype=np.int32)
    acc_beta_S = np.empty((n_iter, P), dtype=bool)
    acc_beta_Z = np.empty((n_iter, P), dtype=bool)
    acc_tau2_0 = np.empty((n_iter, P), dtype=bool)
    acc_tau2_1 = np.empty((n_iter, P), dtype=bool)
    X_chains = [np.empty((n_iter, P, m.T)) for m in markets]
    V_chains = [np.empty((n_iter, P, m.T), dtype=np.int8) for m in markets]
    Z_chains = [np.empty((n_iter, P, m.T), dtype=np.int8) for m in markets]

    uncond_indices = np.arange(P, M)

    # Adaptive tuning window
    adapt_window = 20
    adapt_until = min(100, max(adapt_window, config.n_burnin // 2))

    iterator = range(n_iter)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc="iPMCMC")

    for it in iterator:
        # ---------- Step 1: Run M SMC passes ----------
        smc_outputs: list[list] = []        # M-list of K-list of SMCOutput
        for m_idx in range(M):
            outputs_per_market = []
            lm_total = 0.0
            if m_idx < P:
                slot = m_idx
                for k, md in enumerate(markets):
                    V_ref, Z_ref = refs_slots[slot][k]
                    out = conditional_smc(
                        md.Y, md.delta, md.log_size_ratio, md.wallet_ids,
                        theta_w_slots[slot], params_slots[slot], config,
                        V_ref, Z_ref, rng=rng,
                    )
                    outputs_per_market.append(out)
                    lm_total += out.log_marginal
            else:
                slot = (m_idx - P) % P
                for k, md in enumerate(markets):
                    out = bootstrap_smc(
                        md.Y, md.delta, md.log_size_ratio, md.wallet_ids,
                        theta_w_slots[slot], params_slots[slot], config,
                        rng=rng,
                    )
                    outputs_per_market.append(out)
                    lm_total += out.log_marginal
            smc_outputs.append(outputs_per_market)
            log_marg[it, m_idx] = lm_total

        # ---------- Step 2: Swap step ----------
        for j in range(P):
            if len(uncond_indices) > 0:
                candidates = np.concatenate([[j], uncond_indices])
                log_w = log_marg[it, candidates]
                log_w_norm = log_w - logsumexp(log_w)
                probs = np.exp(log_w_norm)
                probs /= probs.sum()                     # guard float
                chain_indices[it, j] = int(rng.choice(candidates, p=probs))
            else:
                chain_indices[it, j] = j                  # M == P → no swap

        # ---------- Step 3: Sample new reference per slot ----------
        for j in range(P):
            cj = int(chain_indices[it, j])
            new_refs = []
            for k in range(K):
                V_p, Z_p = sample_path(smc_outputs[cj][k], rng)
                new_refs.append((V_p, Z_p))
            refs_slots[j] = new_refs

        # ---------- Step 4: FFBS + Gibbs sweep per slot ----------
        for p in range(P):
            latents: list[MarketLatents] = []
            for k, md in enumerate(markets):
                V_p, Z_p = refs_slots[p][k]
                X_p = ffbs_sample(
                    md.Y, V_p, Z_p, md.delta, md.log_size_ratio,
                    params_slots[p], rng,
                )
                latents.append(MarketLatents(
                    Y=md.Y, delta=md.delta, log_size_ratio=md.log_size_ratio,
                    wallet_ids=md.wallet_ids,
                    X=X_p, V=V_p, Z=Z_p,
                ))
                X_chains[k][it, p] = X_p
                V_chains[k][it, p] = V_p
                Z_chains[k][it, p] = Z_p

            params_slots[p], theta_w_slots[p], diag = gibbs_sweep(
                params_slots[p], theta_w_slots[p], latents, config, rng,
            )

            sigma2_0[it, p] = params_slots[p].sigma2_0
            sigma2_1[it, p] = params_slots[p].sigma2_1
            q_01[it, p] = params_slots[p].q_01
            q_10[it, p] = params_slots[p].q_10
            beta_S[it, p] = params_slots[p].beta_S
            beta_Z[it, p] = params_slots[p].beta_Z
            tau2_0[it, p] = params_slots[p].tau2_0
            tau2_1[it, p] = params_slots[p].tau2_1
            theta_w_chain[it, p] = theta_w_slots[p]
            acc_beta_S[it, p] = diag.acc_beta_S
            acc_beta_Z[it, p] = diag.acc_beta_Z
            acc_tau2_0[it, p] = diag.acc_tau2_0
            acc_tau2_1[it, p] = diag.acc_tau2_1

        # ---------- Adaptive step-size tuning (pooled across slots) ----------
        if adapt_step_sizes and 0 < it < adapt_until and (it + 1) % adapt_window == 0:
            lo, hi = it + 1 - adapt_window, it + 1
            _adapt_step(config, "mh_step_beta_S", float(acc_beta_S[lo:hi].mean()))
            _adapt_step(config, "mh_step_beta_Z", float(acc_beta_Z[lo:hi].mean()))
            _adapt_step(config, "mh_step_log_tau2_0", float(acc_tau2_0[lo:hi].mean()))
            _adapt_step(config, "mh_step_log_tau2_1", float(acc_tau2_1[lo:hi].mean()))

    return iPMCMCOutput(
        sigma2_0=sigma2_0, sigma2_1=sigma2_1,
        q_01=q_01, q_10=q_10,
        beta_S=beta_S, beta_Z=beta_Z,
        tau2_0=tau2_0, tau2_1=tau2_1,
        theta_w=theta_w_chain,
        X=X_chains, V=V_chains, Z=Z_chains,
        log_marg=log_marg,
        chain_indices=chain_indices,
        acc_beta_S=acc_beta_S, acc_beta_Z=acc_beta_Z,
        acc_tau2_0=acc_tau2_0, acc_tau2_1=acc_tau2_1,
        final_mh_step_beta_S=config.mh_step_beta_S,
        final_mh_step_beta_Z=config.mh_step_beta_Z,
        final_mh_step_log_tau2_0=config.mh_step_log_tau2_0,
        final_mh_step_log_tau2_1=config.mh_step_log_tau2_1,
    )
