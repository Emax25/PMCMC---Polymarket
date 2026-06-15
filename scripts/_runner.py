"""Shared helpers for run_pg.py and run_ipmcmc.py.

Both runners need the same things:
  * Config presets (dev / prod / custom via individual flags)
  * Either load processed markets from disk or generate synthetic ones
  * Pickle the chain output with metadata
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config.default_params import PRODUCTION, InferenceConfig, ModelParams
from src.data.preprocess import (
    ProcessedMarket,
    WalletIndex,
    load_processed,
    load_wallet_index,
)
from src.data.synthetic import SyntheticMarket, generate_dataset
from src.inference.particle_gibbs import MarketData

# ---------------- Config presets ----------------

DEV_CONFIG = InferenceConfig(N=50, n_iter=200, n_burnin=50, seed=42)
PROD_CONFIG = replace(PRODUCTION, seed=42)

CONFIG_PRESETS: dict[str, InferenceConfig] = {
    "dev": DEV_CONFIG,
    "prod": PROD_CONFIG,
}


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Register CLI flags shared by run_pg.py and run_ipmcmc.py.

    Adds arguments for config preset, seed, iteration counts, particle
    count, data directory, market slugs, synthetic-data options, output
    path, and log level.

    Args:
        p: Argument parser to register flags on; mutated in place.
    """
    p.add_argument(
        "--config",
        choices=tuple(CONFIG_PRESETS),
        default="dev",
        help="Preset for InferenceConfig (dev=fast, prod=overnight).",
    )
    p.add_argument("--seed", type=int, default=None, help="Override preset seed.")
    p.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="Override preset n_iter (handy for trimming a prod run).",
    )
    p.add_argument(
        "--n-burnin",
        type=int,
        default=None,
        help="Override preset n_burnin.",
    )
    p.add_argument(
        "--n-particles",
        type=int,
        default=None,
        help="Override preset N (particles per CSMC pass).",
    )
    # Data source
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing <slug>.parquet + wallet_index.json from "
        "pull_data.py. Ignored when --synthetic.",
    )
    p.add_argument(
        "--slugs",
        nargs="+",
        default=None,
        help="Restrict to a subset of slugs (default: every parquet in " "--data-dir).",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate K synthetic markets instead of loading from disk "
        "(§9 validation workflow).",
    )
    p.add_argument(
        "--synthetic-K",
        type=int,
        default=5,
        help="Number of synthetic markets when --synthetic is set.",
    )
    p.add_argument(
        "--synthetic-T",
        type=int,
        default=200,
        help="Trades per synthetic market.",
    )
    p.add_argument(
        "--synthetic-n-wallets",
        type=int,
        default=20,
        help="Wallets per synthetic market (insiders = 3).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pickle path. Defaults to results/chains/<sampler>_<config>.pkl.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )


def build_config(args: argparse.Namespace) -> InferenceConfig:
    """Build an InferenceConfig from a preset, applying any CLI overrides.

    Args:
        args: Parsed namespace from an argparse parser that called
            ``add_common_args``.

    Returns:
        InferenceConfig with preset values overridden by any non-None flags.
    """
    cfg = replace(CONFIG_PRESETS[args.config])
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.n_iter is not None:
        cfg = replace(cfg, n_iter=args.n_iter)
    if args.n_burnin is not None:
        cfg = replace(cfg, n_burnin=args.n_burnin)
    if args.n_particles is not None:
        cfg = replace(cfg, N=args.n_particles)
    return cfg


# ---------------- Data loading ----------------


@dataclass
class RunInputs:
    """What both run_pg.py and run_ipmcmc.py operate on."""

    markets: list[MarketData]
    market_objs: list[ProcessedMarket | SyntheticMarket]  # for plotting
    wallet_index: WalletIndex
    is_synthetic: bool


def _market_to_md(market: ProcessedMarket | SyntheticMarket) -> MarketData:
    """Convert a real or synthetic market object to a MarketData for inference."""
    if isinstance(market, ProcessedMarket):
        return market.to_market_data()
    # SyntheticMarket lacks a to_market_data; build it directly
    log_sr = np.log(market.S / market.S_bar)
    return MarketData(
        Y=market.Y,
        delta=market.delta,
        log_size_ratio=log_sr,
        wallet_ids=market.wallet_ids,
    )


def load_real_inputs(
    data_dir: Path,
    slugs: Sequence[str] | None,
) -> RunInputs:
    """Load processed markets and wallet index from a pull_data.py output dir.

    Args:
        data_dir: Directory containing ``<slug>.parquet`` files and
            ``wallet_index.json`` produced by ``pull_data.py``.
        slugs: Explicit list of market slugs to load; all ``*.parquet``
            stems in ``data_dir`` are loaded when None.

    Returns:
        RunInputs with ``is_synthetic=False`` and one ProcessedMarket
        per slug.

    Raises:
        FileNotFoundError: If no ``*.parquet`` files are found under
            ``data_dir`` when ``slugs`` is None.
    """
    if slugs is None:
        slugs = sorted(p.stem for p in data_dir.glob("*.parquet"))
    if not slugs:
        raise FileNotFoundError(
            f"No *.parquet files found under {data_dir}. Did you run pull_data.py?"
        )
    market_objs = [load_processed(data_dir / f"{s}.parquet") for s in slugs]
    wallet_index = load_wallet_index(data_dir / "wallet_index.json")
    mds = [m.to_market_data() for m in market_objs]
    return RunInputs(
        markets=mds,
        market_objs=list(market_objs),
        wallet_index=wallet_index,
        is_synthetic=False,
    )


def make_synthetic_inputs(
    K: int,
    T: int,
    n_wallets: int,
    *,
    seed: int,
) -> RunInputs:
    """Generate K synthetic markets for the §9 validation workflow.

    Warm-starts ModelParams from a short dummy Y draw to avoid degenerate
    priors, then calls ``generate_dataset`` with three insider wallets
    fixed. A synthetic WalletIndex mirroring the integer ids is created
    alongside the markets.

    Args:
        K: Number of synthetic markets to generate.
        T: Number of trades per market.
        n_wallets: Total wallet count; three are designated insiders.
        seed: RNG seed for reproducibility.

    Returns:
        RunInputs with ``is_synthetic=True`` and K SyntheticMarket
        objects.
    """
    rng = np.random.default_rng(seed)
    Y_dummy = rng.standard_normal(max(200, T))
    params = ModelParams.warm_start(Y_dummy)
    markets = generate_dataset(
        params,
        n_markets=K,
        n_trades=T,
        n_wallets=n_wallets,
        n_insider_wallets=3,
        mean_inter_trade_time=1.0,
        rng=rng,
    )
    # Fake a wallet index that mirrors the synthetic integer ids.
    idx = WalletIndex()
    for w in range(n_wallets):
        idx.add(f"synthetic-{w:04d}")
    mds = [_market_to_md(m) for m in markets]
    return RunInputs(
        markets=mds,
        market_objs=list(markets),
        wallet_index=idx,
        is_synthetic=True,
    )


def load_inputs(args: argparse.Namespace, *, seed_fallback: int) -> RunInputs:
    """Dispatch to synthetic or real data loading based on CLI flags.

    Args:
        args: Parsed namespace from an argparse parser that called
            ``add_common_args``.
        seed_fallback: RNG seed used for synthetic generation when ``--seed``
            was not passed explicitly.

    Returns:
        Populated RunInputs ready for inference.
    """
    if args.synthetic:
        return make_synthetic_inputs(
            args.synthetic_K,
            args.synthetic_T,
            args.synthetic_n_wallets,
            seed=(args.seed if args.seed is not None else seed_fallback),
        )
    return load_real_inputs(args.data_dir, args.slugs)


# ---------------- Output ----------------


def default_output_path(sampler: str, config_name: str) -> Path:
    """Return the default pickle path for a sampler + config combination.

    Args:
        sampler: Short sampler name, e.g. ``"pg"`` or ``"ipmcmc"``.
        config_name: Preset name, e.g. ``"dev"`` or ``"prod"``.

    Returns:
        Path of the form ``results/chains/<sampler>_<config_name>.pkl``.
    """
    return Path("results/chains") / f"{sampler}_{config_name}.pkl"


def pickle_run(
    output_path: Path,
    *,
    sampler: str,
    config: InferenceConfig,
    chain: Any,
    inputs: RunInputs,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist chain output and run metadata as a self-contained pickle.

    Writes a dict with keys: sampler, config, chain, is_synthetic, slugs,
    wallet_index, market_objs, plus any entries in ``extra``.

    Args:
        output_path: Destination file path; parent directories are
            created if they do not exist.
        sampler: Short sampler name, e.g. ``"pg"`` or ``"ipmcmc"``.
        config: InferenceConfig used for the run.
        chain: PGOutput or iPMCMCOutput returned by the sampler.
        inputs: RunInputs the sampler consumed; market_objs and
            wallet_index are bundled into the pickle for self-contained
            reload.
        extra: Optional additional keys to merge into the payload dict.

    Returns:
        Path of the pickle file written (same as ``output_path``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sampler": sampler,
        "config": config,
        "chain": chain,
        "is_synthetic": inputs.is_synthetic,
        "slugs": [getattr(m, "slug", "") for m in inputs.market_objs],
        "wallet_index": inputs.wallet_index,
        "market_objs": inputs.market_objs,
        **(extra or {}),
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    return output_path


def load_run(path: Path) -> dict[str, Any]:
    """Load a run payload previously saved with ``pickle_run``.

    Args:
        path: Path to the pickle file produced by ``pickle_run``.

    Returns:
        Dict with at minimum the keys: sampler, config, chain,
        is_synthetic, slugs, wallet_index, market_objs, plus any extras
        stored by the runner.
    """
    with Path(path).open("rb") as f:
        return pickle.load(f)
