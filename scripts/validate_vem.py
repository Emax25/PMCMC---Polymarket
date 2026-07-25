"""CLI: samplerless validation suite for the variational-EM fast path.

One command produces one JSON artifact plus figures covering the four checks
that defend the VEM approximation without an exact sampler (plan
``2026-07-23-002``, R6/R7):

  1. **ELBO traces** — the per-iteration ADF log-marginal of every restart.
  2. **Multi-seed stability** — R random restarts, reporting the spread of the
     terminal ELBO, pooled synthetic AUC, and back-transformed betas, plus the
     mean pairwise Jaccard overlap of the top-K suspicious-wallet sets.
  3. **Held-out one-step predictive log-likelihood** — fit on each market's
     temporal head, score its held-out tail through the ADF forward pass.
  4. **PSIS-khat** — Pareto-smoothed importance-sampling diagnostic between the
     Laplace posterior over ``phi`` and the ADF-implied parameter marginal.

Restarts differ only in their *initialization*: VEM is deterministic given
inputs, so a seed changes nothing unless the start point moves. Each restart
draws ``theta_w`` from the Beta(a, b) prior and multiplies the four warm-start
variances by lognormal noise, which is exactly the multi-start protocol that
would expose ELBO multimodality if the objective had it.

The suite is post-hoc: it reads a fitted ``VEMOutput`` and never alters the
inference path. Read the khat value with the R5 scope note it is stored
alongside — it checks Laplace-shape adequacy for the ADF surface, not ADF's own
fidelity; simulation-based calibration is the faithfulness test.

Examples:
    # Dev-scale synthetic validation run (the standard artifact)
    python -m scripts.validate_vem --config dev --json-out \\
        results/validation/dev.json

    # Cheap smoke: two restarts, the PSIS tail-fit minimum of draws
    python -m scripts.validate_vem --synthetic-K 2 --synthetic-T 100 \\
        --n-restarts 2 --psis-draws 50 --vem-iters 5

    # Real processed markets (no AUC — no insider ground truth)
    python -m scripts.validate_vem --real --data-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config.default_params import InferenceConfig, ModelParams, PhiPrior
from scripts._runner import (
    RunInputs,
    add_common_args,
    build_config,
    load_inputs,
    make_synthetic_inputs,
)
from src.analysis.plots import (
    plot_elbo_traces,
    plot_heldout_ll,
    plot_psis_diagnostic,
    save_paper_figure,
    set_paper_style,
)
from src.analysis.results import (
    count_wallet_trades,
    evaluate_synthetic_gate,
    recall_k_cutoff,
)
from src.analysis.validation import (
    PSIS_KHAT_KEY,
    # Imported rather than re-declared so the CLI's up-front check can never
    # drift from the tail-fit minimum ``psis_khat`` itself enforces (a
    # duplicated literal would let the CLI accept a rejected draw count).
    _PSIS_MIN_DRAWS,
    heldout_predictive_summary,
    holdout_split,
    psis_khat,
)
from src.inference.laplace import laplace_from_vem
from src.inference.variational_em import VEMOutput, variational_em

log = logging.getLogger("validate_vem")

# Lognormal sd applied to each warm-start variance when perturbing a restart's
# start point. Large enough that restarts explore genuinely different basins,
# small enough to keep the regime ordering the warm start encodes
# (sigma2_0 = 0.1*var_Y vs sigma2_1 = var_Y are a decade apart, so a 0.1 log-sd
# jitter cannot swap them and hand VEM a label-switched initialization).
_INIT_JITTER_LOG_SD = 0.1

# Variance fields perturbed per restart, in the order the jitter draw indexes.
_JITTERED_VARIANCES = ("sigma2_0", "sigma2_1", "tau2_0", "tau2_1")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the validate_vem argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Samplerless validation suite for the VEM fast path: ELBO traces, "
            "multi-seed stability, held-out predictive log-likelihood, and the "
            "PSIS-khat diagnostic, bundled into one JSON artifact plus figures."
        ),
    )
    add_common_args(p)
    p.add_argument(
        "--real",
        action="store_true",
        help="Load processed markets from disk (default: synthetic).",
    )
    p.add_argument(
        "--n-restarts",
        type=int,
        default=5,
        help="Number of randomly initialized VEM restarts (default: 5).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit restart seeds (default: base_seed + 0..n_restarts-1).",
    )
    p.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Fraction of each market's tail held out for scoring (default: 0.2).",
    )
    p.add_argument(
        "--psis-draws",
        type=int,
        default=1000,
        help="Laplace draws scored for the PSIS-khat diagnostic (default: 1000).",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="joblib workers over PSIS draws (default: 1; results are unaffected).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K wallet-set size for the Jaccard stability metric "
        "(default: top decile, at least 1).",
    )
    p.add_argument(
        "--vem-iters",
        type=int,
        default=50,
        help="Variational EM iteration cap per restart (default: 50).",
    )
    p.add_argument(
        "--vem-tol",
        type=float,
        default=1e-4,
        help="VEM ELBO relative convergence tolerance (default: 1e-4).",
    )
    p.add_argument(
        "--estimate-betas",
        action="store_true",
        help="Fit beta_S/beta_Z by IRLS each M-step. Off by default, matching "
        "variational_em: the ADF E-step cannot identify Z on the current "
        "synthetic generator, so enabling this fits a spurious size-correlated "
        "beta_S and drops the gate AUC.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/validation"),
        help="Directory for the figures (default: results/validation).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="JSON artifact path (default: <out-dir>/vem_validation.json).",
    )
    return p.parse_args(argv)


# ---------------- Restarts ----------------


def _jittered_init(
    markets: list[Any],
    n_wallets: int,
    rng: np.random.Generator,
) -> tuple[ModelParams, np.ndarray]:
    """Draw one restart's VEM start point around the deterministic warm start.

    VEM is a deterministic map from (data, initialization) to a fit, so a
    stability check has to move the initialization: the warm-start variances are
    scaled by ``exp(N(0, _INIT_JITTER_LOG_SD))`` (multiplicative, keeping every
    variance positive) and ``theta_w`` is drawn from its own Beta(a, b) prior
    instead of being pinned at the prior mean.

    Args:
        markets: Markets the restart will be fit on; supplies the warm start.
        n_wallets: Length of the ``theta_w`` vector to draw.
        rng: Restart-seeded generator (never the global RNG).

    Returns:
        The ``(params_init, theta_w_init)`` pair for this restart.
    """
    Y_concat = np.concatenate([md.Y for md in markets])
    params = ModelParams.warm_start(Y_concat)
    jitter = np.exp(rng.normal(0.0, _INIT_JITTER_LOG_SD, size=len(_JITTERED_VARIANCES)))
    params = replace(
        params,
        **{
            name: float(getattr(params, name) * factor)
            for name, factor in zip(_JITTERED_VARIANCES, jitter)
        },
    )
    theta_w = rng.beta(params.a, params.b, size=n_wallets)
    return params, theta_w


def _fit_restart(
    markets: list[Any],
    cfg: InferenceConfig,
    *,
    n_wallets: int,
    seed: int,
    vem_iters: int,
    vem_tol: float,
    prior: PhiPrior,
    estimate_betas: bool,
) -> VEMOutput:
    """Fit one randomly initialized VEM restart.

    Args:
        markets: Markets to fit.
        cfg: InferenceConfig (VEM takes its iteration cap and tolerance as
            explicit kwargs; the config supplies the rest).
        n_wallets: Global wallet count.
        seed: Restart seed; drives the initialization draw only.
        vem_iters: EM iteration cap.
        vem_tol: Relative ELBO convergence tolerance.
        prior: The M-step prior spec, threaded on so the Laplace layer and PSIS
            score against the same densities the fit optimized.
        estimate_betas: Whether the M-step fits beta_S/beta_Z by IRLS.

    Returns:
        The fitted ``VEMOutput``.
    """
    rng = np.random.default_rng(seed)
    params_init, theta_w_init = _jittered_init(markets, n_wallets, rng)
    return variational_em(
        markets,
        cfg,
        n_wallets=n_wallets,
        params_init=params_init,
        theta_w_init=theta_w_init,
        n_iter=vem_iters,
        tol=vem_tol,
        prior=prior,
        estimate_betas=estimate_betas,
    )


def _top_k_wallets(theta_w: np.ndarray, k: int) -> list[int]:
    """Return the k highest-scoring wallet ids, ties broken by ascending id."""
    scores = np.asarray(theta_w, dtype=float)
    k_eff = min(max(1, k), scores.size)
    order = np.argsort(-scores, kind="mergesort")
    return sorted(int(w) for w in order[:k_eff])


def _pooled_auc(vem: VEMOutput, inputs: RunInputs) -> float | None:
    """Pooled synthetic ROC AUC of the per-trade insider scores, or None.

    Delegates to the same ``evaluate_synthetic_gate`` path the benchmark gate
    uses, so a restart's AUC is comparable with the recorded gate numbers.
    Returns None on real data, which carries no insider ground truth.
    """
    if not inputs.is_synthetic:
        return None
    n_trades = count_wallet_trades(
        [md.wallet_ids for md in inputs.markets],
        n_wallets=inputs.wallet_index.n_wallets,
    )
    gate = evaluate_synthetic_gate(
        vem.Z_prob,
        np.asarray(vem.theta_w, dtype=float),
        inputs.market_objs,
        inputs.wallet_index,
        n_trades_per_wallet=n_trades,
    )
    return float(gate["pooled_auc"])


def _restart_record(
    vem: VEMOutput,
    inputs: RunInputs,
    *,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    """Summarize one restart into its JSON record."""
    trace = np.asarray(vem.elbo_trace, dtype=float)
    return {
        "seed": seed,
        "terminal_elbo": float(trace[-1]) if trace.size else float("nan"),
        "n_iter_run": int(vem.n_iter_run),
        "pooled_auc": _pooled_auc(vem, inputs),
        "beta_S_orig": float(vem.beta_S_orig),
        "beta_Z_orig": float(vem.beta_Z_orig),
        "top_k_wallets": _top_k_wallets(vem.theta_w, top_k),
        "elbo_trace": [float(v) for v in trace],
    }


# ---------------- Stability aggregation ----------------


def _spread(values: list[float | None]) -> dict[str, float]:
    """Min/max/mean/sd and max-min spread of a restart metric.

    ``None`` entries (a metric undefined for this run, e.g. AUC on real data)
    are dropped; an all-``None`` metric yields NaNs rather than raising.

    Args:
        values: One value per restart, possibly containing None.

    Returns:
        Dict with ``min``, ``max``, ``mean``, ``sd`` and ``spread`` keys.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        nan = float("nan")
        return {"min": nan, "max": nan, "mean": nan, "sd": nan, "spread": nan}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        # Population sd (ddof=0): the restarts are the whole ensemble being
        # described, not a sample drawn from a larger population.
        "sd": float(arr.std()),
        "spread": float(arr.max() - arr.min()),
    }


def _mean_pairwise_jaccard(wallet_sets: list[list[int]]) -> float:
    """Mean Jaccard overlap over all unordered pairs of top-K wallet sets.

    A value of 1.0 means every restart flagged exactly the same wallets — the
    stability claim the ranking rests on. Fewer than two restarts leaves the
    metric undefined (NaN).

    Args:
        wallet_sets: One top-K wallet id list per restart.

    Returns:
        Mean pairwise ``|A & B| / |A | B|``, or NaN with fewer than two sets.
    """
    if len(wallet_sets) < 2:
        return float("nan")
    sets = [set(s) for s in wallet_sets]
    overlaps = [
        len(a & b) / len(a | b) if (a | b) else 1.0
        for i, a in enumerate(sets)
        for b in sets[i + 1 :]
    ]
    return float(np.mean(overlaps))


def _stability_block(records: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    """Aggregate per-restart records into the R6 stability summary."""
    return {
        "n_restarts": len(records),
        "top_k": top_k,
        "terminal_elbo": _spread([r["terminal_elbo"] for r in records]),
        "pooled_auc": _spread([r["pooled_auc"] for r in records]),
        "beta_S_orig": _spread([r["beta_S_orig"] for r in records]),
        "beta_Z_orig": _spread([r["beta_Z_orig"] for r in records]),
        "mean_pairwise_topk_jaccard": _mean_pairwise_jaccard(
            [r["top_k_wallets"] for r in records],
        ),
    }


# ---------------- Figures and report ----------------


def _write_figures(
    records: list[dict[str, Any]],
    heldout: dict[str, Any],
    psis_log_weights: np.ndarray,
    psis_log_weights_smoothed: np.ndarray,
    khat: float,
    *,
    out_dir: Path,
) -> list[str]:
    """Render and save the three validation figures; return their paths."""
    set_paper_style()
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    plot_elbo_traces(
        [np.asarray(r["elbo_trace"], dtype=float) for r in records],
        labels=[f"seed {r['seed']}" for r in records],
        ax=ax,
    )
    fig.tight_layout()
    paths.extend(save_paper_figure(fig, "vem_elbo_traces", directory=out_dir))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    plot_psis_diagnostic(psis_log_weights, psis_log_weights_smoothed, khat, ax=ax)
    fig.tight_layout()
    paths.extend(save_paper_figure(fig, "vem_psis_khat", directory=out_dir))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    plot_heldout_ll(
        np.asarray([m["mean"] for m in heldout["per_market"]], dtype=float),
        pooled_mean=heldout["pooled_mean"],
        ax=ax,
    )
    fig.tight_layout()
    paths.extend(save_paper_figure(fig, "vem_heldout_ll", directory=out_dir))
    plt.close(fig)

    return [str(p) for p in paths]


def _format_report(payload: dict[str, Any]) -> str:
    """Build a human-readable summary of the validation artifact."""
    stability = payload["stability"]
    heldout = payload["heldout"]
    psis = payload["psis"]
    lines = [
        "=== VEM validation suite ===",
        f"Markets K={payload['inputs']['K']}  "
        f"n_wallets={payload['inputs']['n_wallets']}  "
        f"synthetic={payload['inputs']['synthetic']}",
        f"Restarts: {stability['n_restarts']}  "
        f"(seeds {[r['seed'] for r in payload['restarts']]})",
        f"estimate_betas={payload['config']['estimate_betas']}",
        "",
        "Per-restart:",
    ]
    for r in payload["restarts"]:
        auc = "n/a" if r["pooled_auc"] is None else f"{r['pooled_auc']:.4f}"
        lines.append(
            f"  seed={r['seed']}: elbo={r['terminal_elbo']:.4f}  "
            f"iters={r['n_iter_run']}  auc={auc}  "
            f"beta_S_orig={r['beta_S_orig']:.4f}  "
            f"beta_Z_orig={r['beta_Z_orig']:.4f}",
        )
    lines.extend(["", "Stability (max - min across restarts):"])
    for key in ("terminal_elbo", "pooled_auc", "beta_S_orig", "beta_Z_orig"):
        s = stability[key]
        lines.append(
            f"  {key:14s}: spread={s['spread']:.6g}  mean={s['mean']:.6g}  "
            f"sd={s['sd']:.6g}",
        )
    lines.append(
        f"  top-{stability['top_k']} wallet Jaccard (mean pairwise): "
        f"{stability['mean_pairwise_topk_jaccard']:.4f}",
    )
    lines.extend(
        [
            "",
            f"Held-out predictive LL (h={payload['config']['holdout_frac']}, "
            f"best restart seed={payload['best_restart']['seed']}):",
            f"  pooled: total={heldout['pooled_total']:.4f}  "
            f"n={heldout['pooled_n']}  mean={heldout['pooled_mean']:.4f}",
        ],
    )
    for idx, m in enumerate(heldout["per_market"]):
        lines.append(
            f"    market {idx}: total={m['total']:.4f}  n={m['n_tail']}  "
            f"mean={m['mean']:.4f}",
        )
    lines.extend(
        [
            "",
            f"PSIS ({psis['psis_n_draws']} draws): "
            f"{PSIS_KHAT_KEY} = {psis[PSIS_KHAT_KEY]:.4f}",
            f"  {psis['psis_khat_interpretation']}",
            f"  scope: {psis['psis_scope_note']}",
        ],
    )
    if payload["laplace"]["curvature_fallback"]:
        lines.append(
            f"  NOTE: Laplace curvature fallback used for "
            f"{payload['laplace']['fallback_dims']}",
        )
    return "\n".join(lines)


# ---------------- Entrypoint ----------------


def main(argv: list[str] | None = None) -> int:
    """Run the VEM validation suite and write its JSON + figure bundle.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code 0 on success.

    Raises:
        SystemExit: If ``--psis-draws`` is below the PSIS tail-fit minimum or
            ``--holdout-frac`` is outside [0, 1] — checked before any fitting
            so a misconfigured long run fails immediately.
    """
    args = _parse_args(argv)
    if args.psis_draws < _PSIS_MIN_DRAWS:
        raise SystemExit(
            f"--psis-draws must be at least {_PSIS_MIN_DRAWS} for the PSIS tail "
            f"fit; got {args.psis_draws}.",
        )
    if not 0.0 <= args.holdout_frac <= 1.0:
        raise SystemExit(
            f"--holdout-frac must lie in [0, 1]; got {args.holdout_frac}.",
        )
    if not args.real:
        args.synthetic = True

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = replace(build_config(args), n_jobs=args.n_jobs)
    base_seed = args.seed if args.seed is not None else cfg.seed
    seeds = (
        list(args.seeds)
        if args.seeds is not None
        else [base_seed + i for i in range(args.n_restarts)]
    )

    if args.synthetic:
        # Data is generated once from base_seed and shared by every restart:
        # the stability question is about the fit, not about resampled data.
        inputs = make_synthetic_inputs(
            args.synthetic_K,
            args.synthetic_T,
            args.synthetic_n_wallets,
            seed=base_seed,
        )
    else:
        inputs = load_inputs(args, seed_fallback=cfg.seed)
    n_wallets = inputs.wallet_index.n_wallets
    top_k = args.top_k if args.top_k is not None else recall_k_cutoff(n_wallets, 1)
    prior = PhiPrior()

    log.info(
        "validate_vem K=%d n_wallets=%d restarts=%d psis_draws=%d n_jobs=%d",
        len(inputs.markets),
        n_wallets,
        len(seeds),
        args.psis_draws,
        args.n_jobs,
    )

    # ---------------- R6: multi-seed stability ----------------
    fits: list[VEMOutput] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        log.info("restart seed=%d …", seed)
        vem = _fit_restart(
            inputs.markets,
            cfg,
            n_wallets=n_wallets,
            seed=seed,
            vem_iters=args.vem_iters,
            vem_tol=args.vem_tol,
            prior=prior,
            estimate_betas=args.estimate_betas,
        )
        fits.append(vem)
        records.append(_restart_record(vem, inputs, seed=seed, top_k=top_k))

    # The best restart carries the downstream metrics: multi-start EM keeps the
    # highest-ELBO mode, and reporting held-out LL / khat for a mode the analysis
    # would not have used would understate the method.
    best_idx = int(np.argmax([r["terminal_elbo"] for r in records]))
    best_vem = fits[best_idx]
    log.info("best restart: index=%d seed=%d", best_idx, seeds[best_idx])

    # ---------------- R4: held-out predictive log-likelihood ----------------
    heads, tails = holdout_split(inputs.markets, args.holdout_frac)
    head_vem = _fit_restart(
        heads,
        cfg,
        n_wallets=n_wallets,
        seed=seeds[best_idx],
        vem_iters=args.vem_iters,
        vem_tol=args.vem_tol,
        prior=prior,
        estimate_betas=args.estimate_betas,
    )
    summary = heldout_predictive_summary(head_vem, heads, tails)
    heldout = {
        "pooled_total": summary.pooled_total,
        "pooled_n": summary.pooled_n,
        "pooled_mean": summary.pooled_mean,
        "per_market": [asdict(m) for m in summary.per_market],
    }

    # ---------------- R5: PSIS-khat ----------------
    phi_posterior = laplace_from_vem(best_vem, inputs.markets, prior)
    psis_result = psis_khat(
        best_vem,
        phi_posterior,
        inputs.markets,
        rng=np.random.default_rng(base_seed),
        n_draws=args.psis_draws,
        n_jobs=args.n_jobs,
        prior=prior,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = _write_figures(
        records,
        heldout,
        psis_result.log_weights,
        psis_result.log_weights_smoothed,
        psis_result.khat,
        out_dir=args.out_dir,
    )

    payload: dict[str, Any] = {
        "config": {
            "n_restarts": len(seeds),
            "seeds": seeds,
            "seed_base": base_seed,
            "holdout_frac": args.holdout_frac,
            "psis_draws": args.psis_draws,
            "n_jobs": args.n_jobs,
            "top_k": top_k,
            "vem_iters": args.vem_iters,
            "vem_tol": args.vem_tol,
            "estimate_betas": args.estimate_betas,
            "init_jitter_log_sd": _INIT_JITTER_LOG_SD,
            "config_preset": args.config,
        },
        "inputs": {
            "K": len(inputs.markets),
            "n_wallets": n_wallets,
            "n_trades": int(sum(len(md.Y) for md in inputs.markets)),
            "synthetic": inputs.is_synthetic,
        },
        "prior": asdict(prior),
        "restarts": records,
        "stability": _stability_block(records, top_k=top_k),
        "best_restart": {"index": best_idx, "seed": seeds[best_idx]},
        "heldout": heldout,
        "psis": psis_result.to_dict(),
        "laplace": {
            "curvature_fallback": bool(phi_posterior.curvature_fallback),
            "fallback_dims": list(phi_posterior.fallback_dims),
        },
        "figures": figures,
    }

    print(_format_report(payload))

    json_out = (
        args.json_out
        if args.json_out is not None
        else args.out_dir / "vem_validation.json"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
