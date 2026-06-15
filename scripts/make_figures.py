"""CLI: produce the §5 application and §9 validation figures from a saved chain.

Inputs: one pickle from run_pg.py or run_ipmcmc.py.
Outputs (under --output-dir, default results/figures + results/tables):
    figures/<sampler>_<slug>_overview.pdf      — 3-panel for each market
    figures/<sampler>_chain_diagnostics.pdf    — trace + density per φ
    figures/<sampler>_wallet_ranking.pdf       — forest plot of top-K wallets
    figures/<sampler>_roc.pdf                  — §9 synthetic-only ROC curve
    tables/<sampler>_chain_summary.csv         — per-φ mean/CI/ESS/R-hat
    tables/<sampler>_wallet_ranking.csv        — full θ_w ranking

Example:
    python -m scripts.make_figures --chain results/chains/pg_dev.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless safety; user can override interactively
import matplotlib.pyplot as plt
import numpy as np

from scripts._runner import load_run
from src.analysis.plots import (
    figure_chain_diagnostics,
    figure_market_overview,
    figure_synthetic_validation,
    plot_wallet_ranking,
    save_paper_figure,
    set_paper_style,
)
from src.analysis.results import (
    count_wallet_trades,
    posterior_Z_probability,
    summarize_chain,
    wallet_ranking,
)
from src.data.preprocess import ProcessedMarket
from src.data.synthetic import SyntheticMarket
from src.inference.particle_gibbs import MarketData

log = logging.getLogger("make_figures")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for make_figures."""
    p = argparse.ArgumentParser(description="Generate paper figures.")
    p.add_argument(
        "--chain",
        type=Path,
        required=True,
        help="Pickle produced by run_pg.py or run_ipmcmc.py.",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
    )
    p.add_argument(
        "--top-k-wallets",
        type=int,
        default=20,
        help="How many wallets to show on the ranking forest plot.",
    )
    p.add_argument(
        "--flag-threshold",
        type=float,
        default=0.5,
        help="P(Z=1|D) threshold for marking trades in the overview panel.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def chain_market_data(chain, market_objs):
    """Return a MarketData list for each market (used for trade-count aggregation).

    Args:
        chain: Unused; reserved for future chain-derived feature extraction.
        market_objs: List of ProcessedMarket or SyntheticMarket objects.

    Returns:
        List of MarketData instances in the same order as market_objs.
    """
    out = []
    for m in market_objs:
        if isinstance(m, ProcessedMarket):
            out.append(m.to_market_data())
        else:
            log_sr = np.log(m.S / m.S_bar)
            out.append(
                MarketData(
                    Y=m.Y,
                    delta=m.delta,
                    log_size_ratio=log_sr,
                    wallet_ids=m.wallet_ids,
                )
            )
    return out


def _insider_addrs_from_synthetic(market_objs, wallet_index) -> set[str]:
    """Map ground-truth insider wallet ids → addresses (synthetic only).

    Args:
        market_objs: List of SyntheticMarket objects carrying
            ``insider_wallet_ids``.
        wallet_index: Global WalletIndex mapping id → address.

    Returns:
        Set of wallet addresses flagged as true insiders across all markets.
    """
    id_to_addr = {v: k for k, v in wallet_index.address_to_id.items()}
    addrs: set[str] = set()
    for m in market_objs:
        ids = getattr(m, "insider_wallet_ids", None)
        if not ids:
            continue
        for wid in ids:
            a = id_to_addr.get(int(wid))
            if a is not None:
                addrs.add(a)
    return addrs


def main(argv: list[str] | None = None) -> int:
    """Generate all paper figures and summary tables from a saved chain pickle.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 on success).
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    payload = load_run(args.chain)
    sampler = payload["sampler"]
    cfg = payload["config"]
    chain = payload["chain"]
    market_objs = payload["market_objs"]
    wallet_index = payload["wallet_index"]
    is_synthetic = payload["is_synthetic"]
    n_burnin = cfg.n_burnin

    set_paper_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Per-market overview panels ----------------
    for k, market in enumerate(market_objs):
        slug = getattr(market, "slug", "") or f"market_{k:02d}"
        log.info("%s: market_overview …", slug)
        fig = figure_market_overview(
            market,
            chain,
            market_idx=k,
            n_burnin=n_burnin,
            flag_threshold=args.flag_threshold,
        )
        save_paper_figure(
            fig,
            f"{sampler}_{slug}_overview",
            directory=args.figures_dir,
        )
        plt.close(fig)

    # ---------------- Chain diagnostics ----------------
    log.info("chain_diagnostics …")
    fig = figure_chain_diagnostics(chain, n_burnin=n_burnin)
    save_paper_figure(
        fig,
        f"{sampler}_chain_diagnostics",
        directory=args.figures_dir,
    )
    plt.close(fig)

    # ---------------- Wallet ranking (plot + table) ----------------
    log.info("wallet ranking …")
    n_trades = count_wallet_trades(
        [md.wallet_ids for md in chain_market_data(chain, market_objs)],
        n_wallets=wallet_index.n_wallets,
    )
    ranking = wallet_ranking(
        chain,
        wallet_index,
        n_burnin=n_burnin,
        n_trades_per_wallet=n_trades,
    )

    insider_addrs: set[str] | None = None
    if is_synthetic:
        # Synthetic markets share an insider wallet set across all K markets
        true_insiders = _insider_addrs_from_synthetic(market_objs, wallet_index)
        insider_addrs = true_insiders or None

    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.22 * args.top_k_wallets + 1.0)))
    plot_wallet_ranking(
        ranking,
        top_k=args.top_k_wallets,
        insider_addresses=insider_addrs,
        ax=ax,
    )
    save_paper_figure(
        fig,
        f"{sampler}_wallet_ranking",
        directory=args.figures_dir,
    )
    plt.close(fig)

    ranking.to_csv(
        args.tables_dir / f"{sampler}_wallet_ranking.csv",
        index=False,
    )

    # ---------------- Chain summary table ----------------
    log.info("chain summary table …")
    summary = summarize_chain(chain, n_burnin=n_burnin)
    summary.to_csv(
        args.tables_dir / f"{sampler}_chain_summary.csv",
        index=False,
    )

    # ---------------- §9 validation ROC (synthetic only) ----------------
    if is_synthetic:
        log.info("synthetic ROC …")
        runs = []
        for k, market in enumerate(market_objs):
            if not isinstance(market, SyntheticMarket):
                continue
            z_prob = posterior_Z_probability(chain, k, n_burnin)
            runs.append((f"market {k}", market.Z, z_prob))
        if runs:
            fig = figure_synthetic_validation(runs)
            save_paper_figure(
                fig,
                f"{sampler}_roc",
                directory=args.figures_dir,
            )
            plt.close(fig)

    log.info("done. figures → %s, tables → %s", args.figures_dir, args.tables_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
