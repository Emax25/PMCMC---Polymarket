"""CLI: benchmark Particle Gibbs wall-clock and optional synthetic accuracy gate.

Measures end-to-end PG runtime across repeated seeds, attributes self-time to
coarse buckets via cProfile (no hot-path instrumentation), and optionally
runs the Stage-0 synthetic validation gate.

Thread pinning: BLAS env vars are set from ``--threads`` before numpy is
imported so wall-clock comparisons are not skewed by OpenBLAS/MKL defaults.

Examples:
    # Quick dev benchmark (default synthetic data)
    python -m scripts.benchmark --config dev

    # Half-prod-ish benchmark (dev preset + overrides; no preset in _runner)
    python -m scripts.benchmark --n-particles 250 --n-iter 1500 --n-burnin 300

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


def _time_runs(
    markets: list[Any],
    cfg: Any,
    *,
    seeds: list[int],
    n_wallets: int,
) -> tuple[list[float], list[float], Any]:
    """Time full PG runs; return seconds, sec/iter, and the last chain."""
    sec_per_run: list[float] = []
    sec_per_iter: list[float] = []
    last_chain = None
    for seed in seeds:
        chain, elapsed = _run_pg_timed(
            markets,
            cfg,
            seed=seed,
            n_wallets=n_wallets,
        )
        sec_per_run.append(elapsed)
        sec_per_iter.append(elapsed / cfg.n_iter)
        last_chain = chain
    return sec_per_run, sec_per_iter, last_chain


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
    for (filename, _line, funcname), (_nc, _ncc, tottime, _cum, _callers) in stats.stats.items():
        bucket = _classify_profile_bucket(filename, funcname)
        buckets[bucket] += tottime

    total = sum(buckets.values())
    if total <= 0:
        return {**buckets, "total": 0.0}
    return {**buckets, "total": total}


def _wallet_rank(rank_df: Any, wallet_id: int) -> int:
    """Return 1-based rank of ``wallet_id`` in a posterior-mean ranking table."""
    matches = rank_df.index[rank_df["wallet_id"] == wallet_id]
    if len(matches) == 0:
        return int(len(rank_df) + 1)
    return int(matches[0]) + 1


def _run_gate(
    chain: Any,
    inputs: Any,
    *,
    n_burnin: int,
    auc_target: float = 0.85,
) -> dict[str, Any]:
    """Evaluate Stage-0 synthetic accuracy metrics on one PG chain."""
    from src.analysis.results import (
        count_wallet_trades,
        posterior_Z_probability,
        roc_auc,
        spearman_theta_w,
        wallet_ranking,
    )
    from src.data.synthetic import SyntheticMarket

    if not inputs.is_synthetic:
        raise ValueError("--gate requires synthetic inputs")

    n_trades = count_wallet_trades(
        [md.wallet_ids for md in inputs.markets],
        n_wallets=inputs.wallet_index.n_wallets,
    )
    rank_df = wallet_ranking(
        chain,
        inputs.wallet_index,
        n_burnin=n_burnin,
        n_trades_per_wallet=n_trades,
    )
    theta_post = chain.theta_w[n_burnin:].mean(axis=0)

    per_market_auc: list[float] = []
    z_true_all: list[Any] = []
    z_score_all: list[Any] = []
    per_market_spearman: list[float] = []

    for idx, mobj in enumerate(inputs.market_objs):
        if not isinstance(mobj, SyntheticMarket):
            continue
        z_prob = posterior_Z_probability(chain, market_idx=idx, n_burnin=n_burnin)
        z_true = mobj.Z.astype(int)
        per_market_auc.append(float(roc_auc(z_true, z_prob)))
        z_true_all.append(z_true)
        z_score_all.append(z_prob)
        per_market_spearman.append(
            float(spearman_theta_w(mobj.theta_w, theta_post)),
        )

    import numpy as np

    pooled_auc = float(
        roc_auc(np.concatenate(z_true_all), np.concatenate(z_score_all)),
    )
    pooled_spearman = float(np.nanmean(per_market_spearman))

    n_wallets = inputs.wallet_index.n_wallets
    insider_ids = sorted(
        {
            wid
            for mobj in inputs.market_objs
            if isinstance(mobj, SyntheticMarket)
            for wid in mobj.insider_wallet_ids
        },
    )
    # "Planted insiders ranked at the top": the cutoff is a top slice of the
    # wallet ranking, but it must be at least as large as the number of
    # planted insiders -- otherwise the criterion is unsatisfiable (e.g. 3
    # insiders can never all fit inside the top 2 of 20 wallets).
    n_insiders = len(insider_ids)
    top_cutoff = max(1, int(np.ceil(n_wallets * 0.1)), n_insiders)
    insider_ranks = {wid: _wallet_rank(rank_df, wid) for wid in insider_ids}
    insiders_in_top = all(rank <= top_cutoff for rank in insider_ranks.values())

    auc_pass = pooled_auc >= auc_target
    insider_pass = insiders_in_top
    gate_pass = auc_pass and insider_pass

    return {
        "pooled_auc": pooled_auc,
        "per_market_auc": per_market_auc,
        "pooled_spearman": pooled_spearman,
        "per_market_spearman": per_market_spearman,
        "insider_wallet_ids": insider_ids,
        "insider_ranks": insider_ranks,
        "top_cutoff": top_cutoff,
        "insiders_in_top": insiders_in_top,
        "auc_pass": auc_pass,
        "insider_pass": insider_pass,
        "gate_pass": gate_pass,
        "spearman_target": 0.9,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for benchmark."""
    from scripts._runner import add_common_args

    p = argparse.ArgumentParser(
        description="Benchmark Particle Gibbs runtime and optional synthetic gate.",
    )
    add_common_args(p)
    p.add_argument(
        "--real",
        action="store_true",
        help="Load processed markets from disk instead of synthetic (default: synthetic).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Pin BLAS/OpenMP thread count for reproducible wall-clock (default: 8).",
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of timed PG runs with distinct seeds (default: 3).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit RNG seeds for timed runs (default: base_seed + 0..n_runs-1).",
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help="Run synthetic accuracy gate on the last timed PG chain.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when --gate fails (default: report only).",
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
    cfg: Any,
    inputs: Any,
    seeds: list[int],
    sec_per_run: list[float],
    sec_per_iter: list[float],
    profile_buckets: dict[str, float],
    gate: dict[str, Any] | None,
) -> str:
    """Build a human-readable benchmark report."""
    run_mean, run_hw = _mean_ci(sec_per_run)
    iter_mean, iter_hw = _mean_ci(sec_per_iter)

    lines = [
        "=== Particle Gibbs benchmark ===",
        f"Markets K={len(inputs.markets)}  N={cfg.N}  "
        f"n_iter={cfg.n_iter}  n_burnin={cfg.n_burnin}  "
        f"synthetic={inputs.is_synthetic}",
        f"Seeds ({len(seeds)}): {seeds}",
        "",
        "Wall-clock (full PG run, seconds):",
    ]
    for seed, elapsed in zip(seeds, sec_per_run):
        lines.append(f"  seed={seed}: {elapsed:.3f}s  ({elapsed / cfg.n_iter:.4f}s/iter)")
    if len(sec_per_run) > 1:
        lines.append(f"  mean +/- CI: {run_mean:.3f} +/- {run_hw:.3f}s")
        lines.append(
            f"  sec/iter mean +/- CI: {iter_mean:.4f} +/- {iter_hw:.4f}s",
        )
    else:
        lines.append(f"  mean: {run_mean:.3f}s  ({iter_mean:.4f}s/iter)")

    total = profile_buckets.get("total", 0.0)
    lines.extend(["", "Cost breakdown (cProfile tottime, one run):"])
    for bucket in ("kalman", "resample", "gibbs", "csmc_other", "other"):
        secs = profile_buckets.get(bucket, 0.0)
        pct = 100.0 * secs / total if total > 0 else 0.0
        lines.append(f"  {bucket:12s}: {secs:8.3f}s  ({pct:5.1f}%)")

    if gate is not None:
        lines.extend(
            [
                "",
                "Synthetic gate (Stage-0):",
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
    """Benchmark PG runtime and optionally evaluate the synthetic gate.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 on success; 1 when ``--strict`` and gate fails).
    """
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

    cfg = build_config(args)
    base_seed = args.seed if args.seed is not None else cfg.seed
    seeds = args.seeds if args.seeds is not None else [base_seed + i for i in range(args.n_runs)]

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
        "Benchmark: K=%d N=%d n_iter=%d threads=%d n_runs=%d",
        len(inputs.markets),
        cfg.N,
        cfg.n_iter,
        args.threads,
        len(seeds),
    )

    sec_per_run, sec_per_iter, last_chain = _time_runs(
        inputs.markets,
        cfg,
        seeds=seeds,
        n_wallets=inputs.wallet_index.n_wallets,
    )

    profile_seed = seeds[0]
    profile_buckets = _profile_breakdown(
        inputs.markets,
        cfg,
        seed=profile_seed,
        n_wallets=inputs.wallet_index.n_wallets,
    )

    gate: dict[str, Any] | None = None
    if args.gate:
        if not inputs.is_synthetic:
            log.error("--gate requires synthetic inputs (omit --real)")
            return 1
        gate = _run_gate(last_chain, inputs, n_burnin=cfg.n_burnin)

    report = _format_report(
        cfg=cfg,
        inputs=inputs,
        seeds=seeds,
        sec_per_run=sec_per_run,
        sec_per_iter=sec_per_iter,
        profile_buckets=profile_buckets,
        gate=gate,
    )
    print(report)

    if args.json_out is not None:
        run_mean, run_hw = _mean_ci(sec_per_run)
        iter_mean, iter_hw = _mean_ci(sec_per_iter)
        payload: dict[str, Any] = {
            "config": {
                "N": cfg.N,
                "n_iter": cfg.n_iter,
                "n_burnin": cfg.n_burnin,
                "seed_base": base_seed,
                "threads": args.threads,
            },
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
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("wrote %s", args.json_out)

    if args.gate and args.strict and gate is not None and not gate["gate_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
