"""CLI: evaluate C4 hybrid prefilter + VEM recall gate on synthetic data.

Runs a cheap microstructure prefilter, then variational EM on flagged-wallet
trades only, and checks that planted-insider recall@K does not regress vs a
full-data VEM baseline. Synthetic inputs only (known insider ground truth).

Examples:
    python -m scripts.eval_c4 --synthetic-K 4 --synthetic-T 300

    python -m scripts.eval_c4 --quantile 0.5 --vem-iters 50 \\
        --json-out results/eval_c4_smoke.json

    python -m scripts.eval_c4 --strict   # exit 1 when recall gate fails
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from config.default_params import InferenceConfig
from scripts._runner import add_common_args, build_config, make_synthetic_inputs
from src.analysis.prefilter import prefilter_wallets, subset_markets_to_wallets
from src.analysis.results import insider_recall_at_k, recall_k_cutoff
from src.data.synthetic import SyntheticMarket
from src.inference.variational_em import variational_em

log = logging.getLogger("eval_c4")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the eval_c4 argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Evaluate C4 hybrid (prefilter + VEM subset) vs full VEM on "
            "synthetic data with known insiders. Always uses synthetic inputs."
        ),
    )
    add_common_args(p)
    p.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="Prefilter quantile (flag top 1-quantile fraction; default 0.5).",
    )
    p.add_argument(
        "--vem-iters",
        type=int,
        default=50,
        help="Variational EM iteration cap (default 50).",
    )
    p.add_argument(
        "--vem-tol",
        type=float,
        default=1e-4,
        help="VEM ELBO relative convergence tolerance (default 1e-4).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of timings and gate metrics.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when recall gate fails (default: report only).",
    )
    return p.parse_args(argv)


def _collect_insider_ids(market_objs: list[Any]) -> list[int]:
    """Union of planted insider wallet ids across synthetic markets."""
    return sorted(
        {
            wid
            for mobj in market_objs
            if isinstance(mobj, SyntheticMarket)
            for wid in mobj.insider_wallet_ids
        },
    )


def _c4_ranking_scores(theta_w: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Rank flagged wallets by fitted theta; push non-flagged to the bottom."""
    theta = np.asarray(theta_w, dtype=float)
    keep = np.asarray(flagged, dtype=bool)
    scores = np.empty_like(theta)
    if keep.any():
        floor = float(theta[keep].min()) - 1.0
    else:
        floor = float(theta.min()) - 1.0
    scores[keep] = theta[keep]
    # Deterministic ordering among non-flagged wallets (all below flagged).
    scores[~keep] = floor - np.flatnonzero(~keep).astype(float) / len(theta)
    return scores


def _total_trades(markets: list[Any]) -> int:
    """Count trades across all markets."""
    return sum(int(m.T) for m in markets)


def _format_report(payload: dict[str, Any]) -> str:
    """Build a human-readable C4 evaluation report."""
    lines = [
        "=== C4 hybrid evaluation (synthetic) ===",
        f"Markets K={payload['inputs']['K']}  "
        f"n_wallets={payload['inputs']['n_wallets']}  "
        f"quantile={payload['config']['quantile']}",
        f"VEM: n_iter={payload['config']['vem_iters']}  "
        f"tol={payload['config']['vem_tol']}",
        "",
        "Wall-clock (seconds):",
        f"  prefilter:   {payload['timings']['sec_prefilter']:.3f}",
        f"  VEM full:    {payload['timings']['sec_vem_full']:.3f}",
        f"  VEM subset:  {payload['timings']['sec_vem_subset']:.3f}",
        f"  C4 total:    {payload['timings']['sec_c4_total']:.3f}  "
        f"(prefilter + VEM subset)",
        f"  speedup:     {payload['timings']['speedup']:.2f}x  "
        f"(VEM full / C4 total)",
        "",
        f"Trades kept:   {payload['fractions']['trades_kept']:.1%}  "
        f"({payload['counts']['trades_subset']}/"
        f"{payload['counts']['trades_total']})",
        f"Wallets flagged: {payload['fractions']['wallets_flagged']:.1%}  "
        f"({payload['counts']['wallets_flagged']}/"
        f"{payload['counts']['n_wallets']})",
        "",
        f"Recall@{payload['recall_k']} (insiders in top K):",
        f"  full VEM:  {payload['recall']['full']:.4f}",
        f"  C4 hybrid: {payload['recall']['c4']:.4f}",
        "",
        f"Insider prefilter flags: {payload['insider_flags']}",
        f"GATE (recall_c4 >= recall_full): "
        f"{'PASS' if payload['gate_pass'] else 'FAIL'}",
    ]
    return "\n".join(lines)


def run_eval_c4(
    markets: list[Any],
    market_objs: list[Any],
    *,
    n_wallets: int,
    config: InferenceConfig,
    quantile: float,
    vem_iters: int,
    vem_tol: float,
) -> dict[str, Any]:
    """Run full vs C4-tier evaluation and return a JSON-serializable report dict.

    Args:
        markets: Full ``MarketData`` list for all synthetic markets.
        market_objs: Synthetic market objects with insider ground truth.
        n_wallets: Global wallet count passed to VEM.
        config: InferenceConfig (warm-start / n_jobs only; VEM uses its own
            ``n_iter`` / ``tol`` kwargs).
        quantile: Prefilter quantile passed to ``prefilter_wallets``.
        vem_iters: VEM iteration cap.
        vem_tol: VEM convergence tolerance.

    Returns:
        Dict with timings, recall metrics, insider flags, and ``gate_pass``.
    """
    insider_ids = _collect_insider_ids(market_objs)
    n_insiders = len(insider_ids)
    k = recall_k_cutoff(n_wallets, n_insiders)

    t0 = time.perf_counter()
    pre = prefilter_wallets(markets, quantile=quantile)
    sec_prefilter = time.perf_counter() - t0

    subset_markets, _ = subset_markets_to_wallets(markets, pre.flagged)
    trades_total = _total_trades(markets)
    trades_subset = _total_trades(subset_markets)

    t0 = time.perf_counter()
    vem_full = variational_em(
        markets,
        config,
        n_wallets=n_wallets,
        n_iter=vem_iters,
        tol=vem_tol,
    )
    sec_vem_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    vem_c4 = variational_em(
        subset_markets,
        config,
        n_wallets=n_wallets,
        n_iter=vem_iters,
        tol=vem_tol,
    )
    sec_vem_subset = time.perf_counter() - t0

    sec_c4_total = sec_prefilter + sec_vem_subset
    speedup = sec_vem_full / sec_c4_total if sec_c4_total > 0 else float("inf")

    recall_full = insider_recall_at_k(vem_full.theta_w, insider_ids, k=k)
    c4_scores = _c4_ranking_scores(vem_c4.theta_w, pre.flagged)
    recall_c4 = insider_recall_at_k(c4_scores, insider_ids, k=k)

    insider_flags = {int(wid): bool(pre.flagged[int(wid)]) for wid in insider_ids}
    gate_pass = recall_c4 >= recall_full

    return {
        "config": {
            "quantile": quantile,
            "vem_iters": vem_iters,
            "vem_tol": vem_tol,
            "seed": config.seed,
        },
        "inputs": {
            "K": len(markets),
            "n_wallets": n_wallets,
            "n_insiders": n_insiders,
            "synthetic": True,
        },
        "counts": {
            "trades_total": trades_total,
            "trades_subset": trades_subset,
            "n_wallets": n_wallets,
            "wallets_flagged": int(pre.flagged.sum()),
            "markets_after_subset": len(subset_markets),
        },
        "fractions": {
            "trades_kept": trades_subset / trades_total if trades_total else 0.0,
            "wallets_flagged": float(pre.flagged.mean()),
        },
        "timings": {
            "sec_prefilter": sec_prefilter,
            "sec_vem_full": sec_vem_full,
            "sec_vem_subset": sec_vem_subset,
            "sec_c4_total": sec_c4_total,
            "speedup": speedup,
        },
        "recall_k": k,
        "recall": {
            "full": recall_full,
            "c4": recall_c4,
        },
        "insider_wallet_ids": insider_ids,
        "insider_flags": insider_flags,
        "gate_pass": gate_pass,
    }


def main(argv: list[str] | None = None) -> int:
    """Evaluate C4 hybrid recall gate on synthetic data.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code 0 on success; 1 when ``--strict`` and the recall gate fails.
    """
    args = _parse_args(argv)
    # This CLI is synthetic-only: always generate planted-insider ground truth.
    args.synthetic = True

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = build_config(args)
    seed = args.seed if args.seed is not None else cfg.seed
    cfg = replace(cfg, seed=seed)

    inputs = make_synthetic_inputs(
        args.synthetic_K,
        args.synthetic_T,
        args.synthetic_n_wallets,
        seed=seed,
    )
    n_wallets = inputs.wallet_index.n_wallets

    log.info(
        "eval_c4 K=%d T=%d n_wallets=%d quantile=%.2f vem_iters=%d",
        len(inputs.markets),
        args.synthetic_T,
        n_wallets,
        args.quantile,
        args.vem_iters,
    )

    payload = run_eval_c4(
        inputs.markets,
        inputs.market_objs,
        n_wallets=n_wallets,
        config=cfg,
        quantile=args.quantile,
        vem_iters=args.vem_iters,
        vem_tol=args.vem_tol,
    )

    report = _format_report(payload)
    print(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("wrote %s", args.json_out)

    if args.strict and not payload["gate_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
