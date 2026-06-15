"""CLI: run vanilla Particle Gibbs on real or synthetic data.

Outputs a pickle under results/chains/ containing the PGOutput plus
metadata (config, slugs, wallet index, market objects) so make_figures.py can
reload the run self-contained.

Examples:
    # Quick dev run on whatever is in data/processed/
    python -m scripts.run_pg --config dev

    # Full prod run, single market only
    python -m scripts.run_pg --config prod \\
        --slugs will-donald-trump-win-the-2024-us-presidential-election

    # §9 synthetic validation (5 markets, T=200 each)
    python -m scripts.run_pg --synthetic --config dev
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from scripts._runner import (
    add_common_args,
    build_config,
    default_output_path,
    load_inputs,
    pickle_run,
)
from src.inference.particle_gibbs import particle_gibbs

log = logging.getLogger("run_pg")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for run_pg."""
    p = argparse.ArgumentParser(description="Run Particle Gibbs.")
    add_common_args(p)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress tqdm progress bar (CI / non-TTY).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run Particle Gibbs and pickle the chain.

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

    cfg = build_config(args)
    inputs = load_inputs(args, seed_fallback=cfg.seed)
    log.info(
        "PG: K=%d markets, N=%d particles, n_iter=%d (burn-in %d), seed=%d",
        len(inputs.markets),
        cfg.N,
        cfg.n_iter,
        cfg.n_burnin,
        cfg.seed,
    )
    for md in inputs.markets:
        log.info("  T=%-6d wallets=%d", md.T, int(md.wallet_ids.max()) + 1)

    t0 = time.monotonic()
    rng = np.random.default_rng(cfg.seed)
    chain = particle_gibbs(
        inputs.markets,
        cfg,
        rng=rng,
        n_wallets=inputs.wallet_index.n_wallets,
        progress=not args.no_progress,
    )
    log.info("PG complete in %.1fs", time.monotonic() - t0)

    out_path = args.output or default_output_path("pg", args.config)
    pickle_run(
        out_path,
        sampler="pg",
        config=cfg,
        chain=chain,
        inputs=inputs,
    )
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
