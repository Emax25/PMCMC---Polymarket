"""CLI: benchmark inference wall-clock and optional synthetic accuracy gate.

Measures end-to-end runtime for Particle Gibbs, iPMCMC, Variational EM, or a
filter-only screening pass across repeated seeds, attributes self-time to
coarse buckets via cProfile for PG only, and optionally runs the synthetic
validation gate.

Thread pinning: BLAS env vars are set from ``--threads`` before numpy is
imported so wall-clock comparisons are not skewed by OpenBLAS/MKL defaults.

Examples:
    # Quick dev benchmark (default synthetic data, Particle Gibbs)
    python -m scripts.benchmark --config dev

    # Variational EM benchmark with synthetic gate
    python -m scripts.benchmark --method vem --gate --vem-iters 50

    # Filter-only screening pass
    python -m scripts.benchmark --method filter --synthetic-K 2

    # iPMCMC ablation row (identical instrumentation to PG)
    python -m scripts.benchmark --method ipmcmc --gate --M 8 --P 4

    # Half-prod-ish PG benchmark (dev preset + overrides)
    python -m scripts.benchmark --n-particles 250 --n-iter 1500 --n-burnin 300

    # Cross-method theta comparison (Kendall tau vs saved baseline)
    python -m scripts.benchmark --method vem --save-theta results/theta_vem.npy
    python -m scripts.benchmark --method pg --compare-theta results/theta_vem.npy

    # Synthetic gate on the last timed run
    python -m scripts.benchmark --gate --strict
"""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import os
import pstats
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

log = logging.getLogger("benchmark")

_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _parse_threads_early(argv: list[str] | None = None) -> int:
    """Read ``--threads`` from argv before numpy initializes BLAS."""
    args = argv if argv is not None else sys.argv[1:]
    threads = 8
    for i, arg in enumerate(args):
        if arg == "--threads" and i + 1 < len(args):
            threads = int(args[i + 1])
            break
        if arg.startswith("--threads="):
            threads = int(arg.split("=", 1)[1])
            break
    return threads


def _pin_blas_threads(threads: int) -> None:
    """Set BLAS thread env vars; must run before importing numpy."""
    val = str(threads)
    for var in _BLAS_THREAD_VARS:
        os.environ[var] = val


# Pin BLAS threads at import time when invoked as ``python -m scripts.benchmark``
# so deferred numpy imports inside ``main`` see bounded thread counts.
_pin_blas_threads(_parse_threads_early())


@dataclass
class _RunArtifacts:
    """Outputs from the last timed inference run used for gate and theta export."""

    theta_w: Any
    z_scores_per_market: list[Any]
    vem_n_iter_run: int | None = None
    vem_final_elbo: float | None = None


def _mean_ci(values: list[float], conf: float = 0.95) -> tuple[float, float]:
    """Return ``(mean, half_width)`` for a two-sided t confidence interval."""
    import numpy as np
    from scipy.stats import t as t_dist

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if n < 2:
        return mean, 0.0
    std = float(arr.std(ddof=1))
    t_crit = float(t_dist.ppf(0.5 + conf / 2.0, df=n - 1))
    half = t_crit * std / (n**0.5)
    return mean, half


def _classify_profile_bucket(filename: str, funcname: str) -> str:
    """Map a cProfile entry to a coarse timing bucket (approximate heuristic)."""
    fname = filename.replace("\\", "/")
    name = funcname.lower()
    if "kalman.py" in fname:
        return "kalman"
    if "parameter_updates.py" in fname:
        return "gibbs"
    if "resample" in name or name == "systematic_resample":
        return "resample"
    if "csmc.py" in fname or "smc.py" in fname:
        return "csmc_other"
    return "other"


def _run_pg_timed(
    markets: list[Any],
    cfg: Any,
    *,
    seed: int,
    n_wallets: int,
) -> tuple[Any, float]:
    """Run one PG chain and return ``(chain, wall_seconds)``."""
    import numpy as np

    from src.inference.particle_gibbs import particle_gibbs

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    chain = particle_gibbs(
        markets,
        cfg,
        rng=rng,
        n_wallets=n_wallets,
        progress=False,
    )
    elapsed = time.perf_counter() - t0
    return chain, elapsed


def _run_ipmcmc_timed(
    markets: list[Any],
    cfg: Any,
    *,
    seed: int,
    n_wallets: int,
) -> tuple[Any, float]:
    """Run one iPMCMC pass (M chains, P conditional) and return ``(chain, wall_seconds)``."""
    import numpy as np

    from src.inference.ipmcmc import ipmcmc

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    chain = ipmcmc(
        markets,
        cfg,
        rng=rng,
        n_wallets=n_wallets,
        progress=False,
    )
    elapsed = time.perf_counter() - t0
    return chain, elapsed


def _run_vem_timed(
    markets: list[Any],
    cfg: Any,
    *,
    seed: int,
    n_wallets: int,
    n_iter: int,
    tol: float,
) -> tuple[Any, float]:
    """Run one VEM fit and return ``(VEMOutput, wall_seconds)``.

    VEM is deterministic given inputs; ``seed`` is accepted for timing
    parity with PG runs but does not affect the fit.
    """
    from src.inference.variational_em import variational_em

    _ = seed
    t0 = time.perf_counter()
    out = variational_em(
        markets,
        cfg,
        n_wallets=n_wallets,
        n_iter=n_iter,
        tol=tol,
    )
    elapsed = time.perf_counter() - t0
    return out, elapsed


def _run_filter_timed(
    markets: list[Any],
    cfg: Any,
    *,
    seed: int,
    n_wallets: int,
) -> tuple[_RunArtifacts, float]:
    """Run filter-only screening and return artifacts plus wall seconds."""
    import numpy as np

    from config.default_params import ModelParams
    from src.inference.smc import bootstrap_smc

    rng = np.random.default_rng(seed)
    Y_all = np.concatenate([md.Y for md in markets])
    params = ModelParams.warm_start(Y_all)
    theta_w = np.full(n_wallets, params.a / (params.a + params.b))

    wallet_sum = np.zeros(n_wallets)
    wallet_count = np.zeros(n_wallets, dtype=int)
    z_scores_per_market: list[np.ndarray] = []

    t0 = time.perf_counter()
    for md in markets:
        out = bootstrap_smc(
            md.Y,
            md.delta,
            md.log_size_ratio,
            md.wallet_ids,
            theta_w,
            params,
            cfg,
            rng=rng,
        )
        z_prob_filt = out.Z_prob_filt
        z_scores_per_market.append(z_prob_filt)
        wallet_sum += np.bincount(
            md.wallet_ids,
            weights=z_prob_filt,
            minlength=n_wallets,
        )
        wallet_count += np.bincount(md.wallet_ids, minlength=n_wallets)
    elapsed = time.perf_counter() - t0

    wallet_scores = np.where(wallet_count > 0, wallet_sum / wallet_count, 0.0)
    artifacts = _RunArtifacts(
        theta_w=wallet_scores,
        z_scores_per_market=z_scores_per_market,
    )
    return artifacts, elapsed


def _time_runs(
    method: str,
    markets: list[Any],
    cfg: Any,
    *,
    seeds: list[int],
    n_wallets: int,
    vem_iters: int,
    vem_tol: float,
) -> tuple[list[float], list[float], _RunArtifacts | Any]:
    """Time inference runs; return seconds, sec/iter, and last-run artifacts."""
    sec_per_run: list[float] = []
    sec_per_iter: list[float] = []
    last_artifacts: _RunArtifacts | Any = None

    for seed in seeds:
        if method == "pg":
            chain, elapsed = _run_pg_timed(
                markets,
                cfg,
                seed=seed,
                n_wallets=n_wallets,
            )
            sec_per_run.append(elapsed)
            sec_per_iter.append(elapsed / cfg.n_iter)
            last_artifacts = chain
        elif method == "ipmcmc":
            chain, elapsed = _run_ipmcmc_timed(
                markets,
                cfg,
                seed=seed,
                n_wallets=n_wallets,
            )
            sec_per_run.append(elapsed)
            sec_per_iter.append(elapsed / cfg.n_iter)
            last_artifacts = chain
        elif method == "vem":
            out, elapsed = _run_vem_timed(
                markets,
                cfg,
                seed=seed,
                n_wallets=n_wallets,
                n_iter=vem_iters,
                tol=vem_tol,
            )
            sec_per_run.append(elapsed)
            sec_per_iter.append(elapsed / out.n_iter_run)
            last_artifacts = _RunArtifacts(
                theta_w=out.theta_w,
                z_scores_per_market=out.Z_prob,
                vem_n_iter_run=out.n_iter_run,
                vem_final_elbo=(
                    float(out.elbo_trace[-1]) if len(out.elbo_trace) else float("nan")
                ),
            )
        elif method == "filter":
            artifacts, elapsed = _run_filter_timed(
                markets,
                cfg,
                seed=seed,
                n_wallets=n_wallets,
            )
            sec_per_run.append(elapsed)
            sec_per_iter.append(elapsed)
            last_artifacts = artifacts
        else:
            raise ValueError(f"unknown method: {method}")

    return sec_per_run, sec_per_iter, last_artifacts


def _profile_breakdown(
    markets: list[Any],
    cfg: Any,
    *,
    seed: int,
    n_wallets: int,
) -> dict[str, float]:
    """Run one profiled PG pass; bucket ``tottime`` by module/function heuristic."""
    import numpy as np

    from src.inference.particle_gibbs import particle_gibbs

    prof = cProfile.Profile()
    prof.enable()
    rng = np.random.default_rng(seed)
    particle_gibbs(
        markets,
        cfg,
        rng=rng,
        n_wallets=n_wallets,
        progress=False,
    )
    prof.disable()

    buckets: dict[str, float] = {
        "kalman": 0.0,
        "resample": 0.0,
        "gibbs": 0.0,
        "csmc_other": 0.0,
        "other": 0.0,
    }
    stats = pstats.Stats(prof)
    for (filename, _line, funcname), (
        _nc,
        _ncc,
        tottime,
        _cum,
        _callers,
    ) in stats.stats.items():
        bucket = _classify_profile_bucket(filename, funcname)
        buckets[bucket] += tottime

    total = sum(buckets.values())
    if total <= 0:
        return {**buckets, "total": 0.0}
    return {**buckets, "total": total}


def _artifacts_from_mcmc_chain(chain: Any, *, n_burnin: int) -> _RunArtifacts:
    """Build gate artifacts from a PG or iPMCMC chain.

    ``posterior_Z_probability`` already pools iPMCMC's extra conditional-chain
    axis internally (src/analysis/results.py), but ``theta_w`` does not go
    through that helper, so iPMCMC's ``(n_iter, P, n_wallets)`` array is
    flattened to ``(n_iter*P, n_wallets)`` here before averaging to match PG's
    ``(n_iter, n_wallets)`` shape.
    """
    from src.analysis.results import posterior_Z_probability
    from src.inference.ipmcmc import iPMCMCOutput

    z_scores = [
        posterior_Z_probability(chain, market_idx=idx, n_burnin=n_burnin)
        for idx in range(len(chain.Z))
    ]
    theta_samples = chain.theta_w[n_burnin:]
    if isinstance(chain, iPMCMCOutput):
        theta_samples = theta_samples.reshape(-1, theta_samples.shape[-1])
    theta_w = theta_samples.mean(axis=0)
    return _RunArtifacts(theta_w=theta_w, z_scores_per_market=z_scores)


def _run_gate(
    artifacts: _RunArtifacts,
    inputs: Any,
    *,
    n_burnin: int,
    method: str,
    auc_target: float = 0.85,
) -> dict[str, Any]:
    """Evaluate synthetic gate from per-market z-scores and wallet scores."""
    import numpy as np

    from src.analysis.results import count_wallet_trades, evaluate_synthetic_gate

    if not inputs.is_synthetic:
        raise ValueError("--gate requires synthetic inputs")

    if method in ("pg", "ipmcmc"):
        if not hasattr(artifacts, "theta_w") or not hasattr(artifacts, "Z"):
            raise TypeError(f"{method} gate requires an MCMC chain output")
        gate_artifacts = _artifacts_from_mcmc_chain(artifacts, n_burnin=n_burnin)
    else:
        gate_artifacts = artifacts

    n_trades = count_wallet_trades(
        [md.wallet_ids for md in inputs.markets],
        n_wallets=inputs.wallet_index.n_wallets,
    )
    return evaluate_synthetic_gate(
        gate_artifacts.z_scores_per_market,
        np.asarray(gate_artifacts.theta_w, dtype=float),
        inputs.market_objs,
        inputs.wallet_index,
        n_trades_per_wallet=n_trades,
        auc_target=auc_target,
    )


def _theta_vector(artifacts: _RunArtifacts | Any, *, method: str, n_burnin: int) -> Any:
    """Extract per-wallet theta/score vector for save/compare."""
    import numpy as np

    if method in ("pg", "ipmcmc"):
        theta_w = _artifacts_from_mcmc_chain(artifacts, n_burnin=n_burnin).theta_w
        return np.asarray(theta_w)
    if isinstance(artifacts, _RunArtifacts):
        return np.asarray(artifacts.theta_w, dtype=float)
    raise TypeError(f"cannot extract theta for method={method}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for benchmark."""
    from scripts._runner import add_common_args

    p = argparse.ArgumentParser(
        description="Benchmark inference runtime and optional synthetic gate.",
    )
    add_common_args(p)
    p.add_argument(
        "--method",
        choices=("pg", "vem", "filter", "ipmcmc"),
        default="pg",
        help="Inference method to benchmark (default: pg).",
    )
    p.add_argument(
        "--real",
        action="store_true",
        help="Load processed markets from disk (default: synthetic).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Pin BLAS/OpenMP thread count for reproducible wall-clock (default: 8).",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="joblib workers for market parallelism in PG (default: 1).",
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of timed runs with distinct seeds (default: 3).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit RNG seeds for timed runs (default: base_seed + 0..n_runs-1).",
    )
    p.add_argument(
        "--M",
        type=int,
        default=None,
        help="Total chains for --method ipmcmc; defaults to preset (8).",
    )
    p.add_argument(
        "--P",
        type=int,
        default=None,
        help="Conditional chains for --method ipmcmc; defaults to preset (4).",
    )
    p.add_argument(
        "--vem-iters",
        type=int,
        default=50,
        help="Maximum EM iterations for --method vem (default: 50).",
    )
    p.add_argument(
        "--vem-tol",
        type=float,
        default=1e-4,
        help="ELBO convergence tolerance for --method vem (default: 1e-4).",
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help="Run synthetic accuracy gate on the last timed run.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when --gate fails (default: report only).",
    )
    p.add_argument(
        "--save-theta",
        type=Path,
        default=None,
        help="Write per-wallet theta/scores from the last run to a .npy file.",
    )
    p.add_argument(
        "--compare-theta",
        type=Path,
        default=None,
        help="Load baseline .npy and report Kendall tau vs the last run.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of timings and gate metrics.",
    )
    return p.parse_args(argv)


def _format_report(
    *,
    method: str,
    cfg: Any,
    inputs: Any,
    seeds: list[int],
    sec_per_run: list[float],
    sec_per_iter: list[float],
    profile_buckets: dict[str, float] | None,
    gate: dict[str, Any] | None,
    vem_n_iter_run: int | None,
    vem_final_elbo: float | None,
    kendall_tau_vs_baseline: float | None,
) -> str:
    """Build a human-readable benchmark report."""
    run_mean, run_hw = _mean_ci(sec_per_run)
    iter_mean, iter_hw = _mean_ci(sec_per_iter)

    method_labels = {
        "pg": "Particle Gibbs",
        "vem": "Variational EM",
        "filter": "Filter screen",
        "ipmcmc": "iPMCMC",
    }
    lines = [
        f"=== {method_labels.get(method, method)} benchmark ===",
        f"Method: {method}",
        f"Markets K={len(inputs.markets)}  N={cfg.N}  "
        f"n_iter={cfg.n_iter}  n_burnin={cfg.n_burnin}  "
        f"synthetic={inputs.is_synthetic}",
    ]
    if method == "vem" and vem_n_iter_run is not None:
        lines.append(
            f"VEM: n_iter_run={vem_n_iter_run}  final_elbo={vem_final_elbo:.4f}  "
            f"(deterministic given inputs; seeds only affect timing repeats)",
        )
    elif method == "vem":
        lines.append(
            "VEM: deterministic given inputs; seeds only affect timing repeats",
        )
    lines.extend(
        [
            f"Seeds ({len(seeds)}): {seeds}",
            "",
            f"Wall-clock (full {method} run, seconds):",
        ],
    )
    iter_label = "sec/iter" if method in ("pg", "ipmcmc") else "sec/iter_equiv"
    for seed, elapsed, spi in zip(seeds, sec_per_run, sec_per_iter):
        lines.append(f"  seed={seed}: {elapsed:.3f}s  ({spi:.4f}s/{iter_label})")
    if len(sec_per_run) > 1:
        lines.append(f"  mean +/- CI: {run_mean:.3f} +/- {run_hw:.3f}s")
        lines.append(
            f"  {iter_label} mean +/- CI: {iter_mean:.4f} +/- {iter_hw:.4f}s",
        )
    else:
        lines.append(f"  mean: {run_mean:.3f}s  ({iter_mean:.4f}s/{iter_label})")

    if profile_buckets is not None:
        total = profile_buckets.get("total", 0.0)
        lines.extend(["", "Cost breakdown (cProfile tottime, one run):"])
        for bucket in ("kalman", "resample", "gibbs", "csmc_other", "other"):
            secs = profile_buckets.get(bucket, 0.0)
            pct = 100.0 * secs / total if total > 0 else 0.0
            lines.append(f"  {bucket:12s}: {secs:8.3f}s  ({pct:5.1f}%)")

    if kendall_tau_vs_baseline is not None:
        lines.extend(
            [
                "",
                f"Kendall tau vs baseline: {kendall_tau_vs_baseline:.4f}",
            ],
        )

    if gate is not None:
        lines.extend(
            [
                "",
                "Synthetic gate (Stage-0/1):",
                f"  pooled ROC AUC: {gate['pooled_auc']:.4f}  "
                f"(target >= 0.85, {'PASS' if gate['auc_pass'] else 'FAIL'})",
            ],
        )
        for idx, auc in enumerate(gate["per_market_auc"]):
            lines.append(f"    market {idx} AUC: {auc:.4f}")
        lines.append(
            f"  pooled Spearman(theta_w): {gate['pooled_spearman']:.4f}  "
            f"(report only; cross-version target >= {gate['spearman_target']})",
        )
        for idx, rho in enumerate(gate["per_market_spearman"]):
            lines.append(f"    market {idx} Spearman: {rho:.4f}")
        lines.append(
            f"  insider ranks (top = rank <= {gate['top_cutoff']}): "
            f"{gate['insider_ranks']}",
        )
        lines.append(
            f"  insiders ranked at top: "
            f"{'PASS' if gate['insider_pass'] else 'FAIL'}",
        )
        lines.append(
            f"  GATE OVERALL: {'PASS' if gate['gate_pass'] else 'FAIL'}",
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Benchmark inference runtime and optionally evaluate the synthetic gate.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 on success; 1 when ``--strict`` and gate fails).
    """
    import numpy as np

    from src.analysis.results import kendall_theta_w

    args = _parse_args(argv)
    _pin_blas_threads(args.threads)

    from scripts._runner import build_config, load_inputs, make_synthetic_inputs

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.real:
        args.synthetic = True

    cfg = replace(build_config(args), n_jobs=args.n_jobs)
    if args.method == "ipmcmc":
        # iPMCMC has no market-level n_jobs (no joblib.Parallel over K); the
        # flag is silently inert there, so warn instead of misleading the
        # JSON config block into implying it did something.
        if args.n_jobs != 1:
            log.warning(
                "--n-jobs=%d has no effect for --method ipmcmc "
                "(no market-level parallelism); ignoring",
                args.n_jobs,
            )
        if args.M is not None:
            cfg = replace(cfg, M=args.M)
        if args.P is not None:
            cfg = replace(cfg, P=args.P)
        if cfg.M < cfg.P:
            raise SystemExit(f"Need M >= P; got M={cfg.M}, P={cfg.P}.")
    base_seed = args.seed if args.seed is not None else cfg.seed
    seeds = (
        args.seeds
        if args.seeds is not None
        else [base_seed + i for i in range(args.n_runs)]
    )

    if args.seeds is None and len(seeds) != args.n_runs:
        seeds = [base_seed + i for i in range(args.n_runs)]

    if args.synthetic:
        inputs = make_synthetic_inputs(
            args.synthetic_K,
            args.synthetic_T,
            args.synthetic_n_wallets,
            seed=base_seed,
        )
    else:
        inputs = load_inputs(args, seed_fallback=cfg.seed)

    log.info(
        "Benchmark method=%s K=%d N=%d n_iter=%d threads=%d n_jobs=%d n_runs=%d",
        args.method,
        len(inputs.markets),
        cfg.N,
        cfg.n_iter,
        args.threads,
        cfg.n_jobs,
        len(seeds),
    )

    sec_per_run, sec_per_iter, last_artifacts = _time_runs(
        args.method,
        inputs.markets,
        cfg,
        seeds=seeds,
        n_wallets=inputs.wallet_index.n_wallets,
        vem_iters=args.vem_iters,
        vem_tol=args.vem_tol,
    )

    profile_buckets: dict[str, float] | None = None
    if args.method == "pg":
        profile_buckets = _profile_breakdown(
            inputs.markets,
            cfg,
            seed=seeds[0],
            n_wallets=inputs.wallet_index.n_wallets,
        )

    vem_n_iter_run: int | None = None
    vem_final_elbo: float | None = None
    if isinstance(last_artifacts, _RunArtifacts):
        vem_n_iter_run = last_artifacts.vem_n_iter_run
        vem_final_elbo = last_artifacts.vem_final_elbo

    gate: dict[str, Any] | None = None
    if args.gate:
        if not inputs.is_synthetic:
            log.error("--gate requires synthetic inputs (omit --real)")
            return 1
        gate = _run_gate(
            last_artifacts,
            inputs,
            n_burnin=cfg.n_burnin,
            method=args.method,
        )

    theta_current = _theta_vector(
        last_artifacts,
        method=args.method,
        n_burnin=cfg.n_burnin,
    )
    if args.save_theta is not None:
        args.save_theta.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_theta, theta_current)
        log.info("wrote theta %s", args.save_theta)

    kendall_tau_vs_baseline: float | None = None
    if args.compare_theta is not None:
        baseline = np.load(args.compare_theta)
        kendall_tau_vs_baseline = float(kendall_theta_w(baseline, theta_current))

    report = _format_report(
        method=args.method,
        cfg=cfg,
        inputs=inputs,
        seeds=seeds,
        sec_per_run=sec_per_run,
        sec_per_iter=sec_per_iter,
        profile_buckets=profile_buckets,
        gate=gate,
        vem_n_iter_run=vem_n_iter_run,
        vem_final_elbo=vem_final_elbo,
        kendall_tau_vs_baseline=kendall_tau_vs_baseline,
    )
    print(report)

    if args.json_out is not None:
        run_mean, run_hw = _mean_ci(sec_per_run)
        iter_mean, iter_hw = _mean_ci(sec_per_iter)
        config_block: dict[str, Any] = {
            "N": cfg.N,
            "n_iter": cfg.n_iter,
            "n_burnin": cfg.n_burnin,
            "n_jobs": cfg.n_jobs,
            "seed_base": base_seed,
            "threads": args.threads,
        }
        if args.method == "ipmcmc":
            config_block["M"] = cfg.M
            config_block["P"] = cfg.P
        payload: dict[str, Any] = {
            "method": args.method,
            "config": config_block,
            "inputs": {
                "K": len(inputs.markets),
                "synthetic": inputs.is_synthetic,
                "seeds": seeds,
            },
            "timings": {
                "sec_per_run": sec_per_run,
                "sec_per_iter": sec_per_iter,
                "mean_sec_per_run": run_mean,
                "ci_half_width_sec_per_run": run_hw,
                "mean_sec_per_iter": iter_mean,
                "ci_half_width_sec_per_iter": iter_hw,
            },
            "profile_tottime": profile_buckets,
            "gate": gate,
        }
        if args.method == "vem":
            payload["vem"] = {
                "n_iter_run": vem_n_iter_run,
                "final_elbo": vem_final_elbo,
                "vem_iters": args.vem_iters,
                "vem_tol": args.vem_tol,
            }
        if kendall_tau_vs_baseline is not None:
            payload["kendall_tau_vs_baseline"] = kendall_tau_vs_baseline
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("wrote %s", args.json_out)

    if args.gate and args.strict and gate is not None and not gate["gate_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
