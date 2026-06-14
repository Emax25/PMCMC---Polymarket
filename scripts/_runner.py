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
from src.data.synthetic import generate_dataset, SyntheticMarket
from src.inference.particle_gibbs import MarketData


# ---------------- Config presets ----------------

DEV_CONFIG = InferenceConfig(N=50, n_iter=200, n_burnin=50, seed=42)
PROD_CONFIG = replace(PRODUCTION, seed=42)

CONFIG_PRESETS: dict[str, InferenceConfig] = {
    "dev": DEV_CONFIG,
    "prod": PROD_CONFIG,
}


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by run_pg.py and run_ipmcmc.py."""
    p.add_argument(
        "--config", choices=tuple(CONFIG_PRESETS), default="dev",
        help="Preset for InferenceConfig (dev=fast, prod=overnight).",
    )
    p.add_argument("--seed", type=int, default=None,
                   help="Override preset seed.")
    p.add_argument(
        "--n-iter", type=int, default=None,
        help="Override preset n_iter (handy for trimming a prod run).",
    )
    p.add_argument(
        "--n-burnin", type=int, default=None,
        help="Override preset n_burnin.",
    )
    p.add_argument(
        "--n-particles", type=int, default=None,
        help="Override preset N (particles per CSMC pass).",
    )
    # Data source
    p.add_argument(
        "--data-dir", type=Path, default=Path("data/processed"),
        help="Directory containing <slug>.parquet + wallet_index.json from "
             "pull_data.py. Ignored when --synthetic.",
    )
    p.add_argument(
        "--slugs", nargs="+", default=None,
        help="Restrict to a subset of slugs (default: every parquet in "
             "--data-dir).",
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="Generate K synthetic markets instead of loading from disk "
             "(§9 validation workflow).",
    )
    p.add_argument(
        "--synthetic-K", type=int, default=5,
        help="Number of synthetic markets when --synthetic is set.",
    )
    p.add_argument(
        "--synthetic-T", type=int, default=200,
        help="Trades per synthetic market.",
    )
    p.add_argument(
        "--synthetic-n-wallets", type=int, default=20,
        help="Wallets per synthetic market (insiders = 3).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output pickle path. Defaults to results/chains/<sampler>_<config>.pkl.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )


def build_config(args: argparse.Namespace) -> InferenceConfig:
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
    market_objs: list[ProcessedMarket | SyntheticMarket]    # for plotting
    wallet_index: WalletIndex
    is_synthetic: bool


def _market_to_md(market: ProcessedMarket | SyntheticMarket) -> MarketData:
    if isinstance(market, ProcessedMarket):
        return market.to_market_data()
    # SyntheticMarket lacks a to_market_data; build it directly
    log_sr = np.log(market.S / market.S_bar)
    return MarketData(
        Y=market.Y, delta=market.delta,
        log_size_ratio=log_sr, wallet_ids=market.wallet_ids,
    )


def load_real_inputs(
    data_dir: Path, slugs: Sequence[str] | None,
) -> RunInputs:
    """Load processed markets + wallet index from a pull_data.py output dir."""
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
        markets=mds, market_objs=list(market_objs),
        wallet_index=wallet_index, is_synthetic=False,
    )


def make_synthetic_inputs(
    K: int, T: int, n_wallets: int, *, seed: int,
) -> RunInputs:
    """Generate K synthetic markets sharing one wallet space (§9 validation)."""
    rng = np.random.default_rng(seed)
    Y_dummy = rng.standard_normal(max(200, T))
    params = ModelParams.warm_start(Y_dummy)
    markets = generate_dataset(
        params, n_markets=K, n_trades=T, n_wallets=n_wallets,
        n_insider_wallets=3, mean_inter_trade_time=1.0, rng=rng,
    )
    # Fake a wallet index that mirrors the synthetic integer ids.
    idx = WalletIndex()
    for w in range(n_wallets):
        idx.add(f"synthetic-{w:04d}")
    mds = [_market_to_md(m) for m in markets]
    return RunInputs(
        markets=mds, market_objs=list(markets),
        wallet_index=idx, is_synthetic=True,
    )


def load_inputs(args: argparse.Namespace, *, seed_fallback: int) -> RunInputs:
    if args.synthetic:
        return make_synthetic_inputs(
            args.synthetic_K, args.synthetic_T, args.synthetic_n_wallets,
            seed=(args.seed if args.seed is not None else seed_fallback),
        )
    return load_real_inputs(args.data_dir, args.slugs)


# ---------------- Output ----------------

def default_output_path(sampler: str, config_name: str) -> Path:
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
    """Persist the run output + enough metadata to reload it self-contained."""
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
    """Inverse of `pickle_run`."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
