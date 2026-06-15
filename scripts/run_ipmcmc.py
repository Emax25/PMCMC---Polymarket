"""CLI: run iPMCMC on real or synthetic data.

Outputs a pickle under results/chains/ containing the iPMCMCOutput plus
metadata (config, slugs, wallet index, market objects) so make_figures.py can
reload the run self-contained.

Examples:
    python -m scripts.run_ipmcmc --config dev
    python -m scripts.run_ipmcmc --config prod --n-iter 1500  # custom budget
    python -m scripts.run_ipmcmc --synthetic --config dev --M 4 --P 2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace

import numpy as np

from scripts._runner import (
    add_common_args,
    build_config,
    default_output_path,
    load_inputs,
    pickle_run,
)
from src.inference.ipmcmc import ipmcmc

log = logging.getLogger("run_ipmcmc")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for run_ipmcmc."""
    p = argparse.ArgumentParser(description="Run iPMCMC.")
    add_common_args(p)
    p.add_argument(
        "--M",
        type=int,
        default=None,
        help="Total chains; defaults to preset (8 prod, 8 dev).",
    )
    p.add_argument(
        "--P",
        type=int,
        default=None,
        help="Conditional chains; defaults to preset (4).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress tqdm progress bar.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run iPMCMC and pickle the chain.

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
    if args.M is not None:
        cfg = replace(cfg, M=args.M)
    if args.P is not None:
        cfg = replace(cfg, P=args.P)
    if cfg.M < cfg.P:
        raise SystemExit(f"Need M >= P; got M={cfg.M}, P={cfg.P}.")

    inputs = load_inputs(args, seed_fallback=cfg.seed)
    log.info(
        "iPMCMC: K=%d markets, M=%d (P=%d cond), N=%d, n_iter=%d "
        "(burn-in %d), seed=%d",
        len(inputs.markets),
        cfg.M,
        cfg.P,
        cfg.N,
        cfg.n_iter,
        cfg.n_burnin,
        cfg.seed,
    )
    for md in inputs.markets:
        log.info("  T=%-6d wallets=%d", md.T, int(md.wallet_ids.max()) + 1)

    t0 = time.monotonic()
    rng = np.random.default_rng(cfg.seed)
    chain = ipmcmc(
        inputs.markets,
        cfg,
        rng=rng,
        n_wallets=inputs.wallet_index.n_wallets,
        progress=not args.no_progress,
    )
    log.info("iPMCMC complete in %.1fs", time.monotonic() - t0)

    out_path = args.output or default_output_path("ipmcmc", args.config)
    pickle_run(
        out_path,
        sampler="ipmcmc",
        config=cfg,
        chain=chain,
        inputs=inputs,
    )
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
