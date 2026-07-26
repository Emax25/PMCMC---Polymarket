"""CLI: pull processed-market data for the §5 shortlist.

For each slug in `_shortlist.SLUGS`:
  1. Resolve slug → MarketMeta via Gamma API.
  2. Paginate /trades for its conditionId — the newest ~10000 trades by
     default, or the entire history under `--full-history`.
  3. Clean + compute features (`build_processed_market`).
  4. Optionally tail to the last `--tail-trades` entries (§8.2 budget).
  5. Build one shared `WalletIndex` over the union of cleaned wallets.
  6. Save `<slug>.parquet` + `<slug>.meta.json` under --output-dir, plus
     `wallet_index.json`.

Usage:

    python -m scripts.pull_data --output-dir data/processed
    python -m scripts.pull_data --tail-trades 2000   # cap per market
    python -m scripts.pull_data --max-pages 20       # cap per market
    python -m scripts.pull_data --full-history       # deep backfill per market

The script is idempotent over the output directory — re-running overwrites.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from scripts._shortlist import SLUGS
from src.data.polymarket_api import (
    MarketMeta,
    fetch_market_by_slug,
    fetch_trades,
    fetch_trades_windowed,
)
from src.data.preprocess import (
    ProcessedMarket,
    _resolution_ts_from_end_date,
    build_genre_dataset,
    save_processed,
    save_wallet_index,
)

log = logging.getLogger("pull_data")

# Page budgets for the two retrieval paths. The tail path is offset-capped by the
# API at ~10000 trades, so 200 pages is already unreachable slack; the windowed
# path walks arbitrarily far back and raises when its budget runs out, so it gets
# the wider default (matching `fetch_trades_windowed`'s own).
_DEFAULT_MAX_PAGES = 200
_FULL_HISTORY_MAX_PAGES = 20_000

# Lower timestamp bound for --full-history. Deliberately 1, not 0: the Data API
# treats a zero `start` as "no filter", so 0 cannot express an explicit bound.
# No Polymarket trade predates epoch 1, so this loses nothing.
_EPOCH_START_TS = 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for pull_data."""
    p = argparse.ArgumentParser(description="Pull §5 shortlist trade data.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to write <slug>.parquet + <slug>.meta.json files.",
    )
    p.add_argument(
        "--tail-trades",
        type=int,
        default=None,
        help="Keep only the last N trades per market (§8.2 target: 500-3000). "
        "Default: keep all surviving trades.",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Polymarket /trades pagination cap per market (500 trades/page). "
        f"Default: {_DEFAULT_MAX_PAGES}, or {_FULL_HISTORY_MAX_PAGES} under "
        "--full-history.",
    )
    p.add_argument(
        "--full-history",
        action="store_true",
        help="Walk timestamp windows to pull each market's entire trade history "
        "instead of only the newest ~10000 trades. Slow; off by default.",
    )
    p.add_argument(
        "--sleep-between",
        type=float,
        default=0.1,
        help="Seconds between paginated /trades calls (politeness).",
    )
    p.add_argument(
        "--slugs",
        nargs="+",
        default=list(SLUGS),
        help="Override the shortlist (debug). Defaults to the 10-market §5 set.",
    )
    p.add_argument(
        "--pre-resolution-days",
        type=float,
        default=7.0,
        help="Drop trades within N days of market resolution/close (default: 7). "
        "Pass 0 to disable pre-resolution filtering.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def _resolve_metadata(slugs: list[str]) -> list[MarketMeta]:
    metas: list[MarketMeta] = []
    for slug in slugs:
        meta = fetch_market_by_slug(slug)
        log.info(
            "%-70s vol=$%-12.0f condition=%s…",
            slug,
            meta.volume,
            meta.condition_id[:14],
        )
        metas.append(meta)
    return metas


def _pull_trades(
    metas: list[MarketMeta],
    max_pages: int,
    sleep_between: float,
    full_history: bool = False,
) -> list[tuple[str, list]]:
    """Returns list of (slug, raw_trades) tuples in shortlist order.

    Args:
        metas: Resolved market metadata, in shortlist order.
        max_pages: Pagination budget per market.
        sleep_between: Seconds between paginated /trades calls.
        full_history: Walk timestamp windows back to epoch instead of taking the
            offset-capped tail of the newest trades.
    """
    out: list[tuple[str, list]] = []
    for meta in metas:
        log.info("pulling /trades for %s …", meta.slug)
        t0 = time.monotonic()
        if full_history:
            trades = fetch_trades_windowed(
                meta.condition_id,
                start_ts=_EPOCH_START_TS,
                max_pages=max_pages,
                sleep_between=sleep_between,
            )
        else:
            trades = fetch_trades(
                meta.condition_id,
                max_pages=max_pages,
                sleep_between=sleep_between,
            )
        log.info(
            "  -> %d raw trades in %.1fs",
            len(trades),
            time.monotonic() - t0,
        )
        out.append((meta.slug, trades))
    return out


def _tail(market: ProcessedMarket, n: int) -> ProcessedMarket:
    """Slice a ProcessedMarket down to its last n trades (delta[0] reset to 0)."""
    if n >= market.T:
        return market
    sl = slice(market.T - n, market.T)
    delta = market.delta[sl].copy()
    delta[0] = 0.0
    return ProcessedMarket(
        Y=market.Y[sl].copy(),
        delta=delta,
        log_size_ratio=market.log_size_ratio[sl].copy(),
        wallet_ids=market.wallet_ids[sl].copy(),
        t=market.t[sl].copy(),
        p=market.p[sl].copy(),
        S=market.S[sl].copy(),
        S_bar=market.S_bar,
        condition_id=market.condition_id,
        slug=market.slug,
    )


def main(argv: list[str] | None = None) -> int:
    """Fetch, clean, and persist the §5 shortlist markets.

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

    metas = _resolve_metadata(args.slugs)
    max_pages = args.max_pages
    if max_pages is None:
        max_pages = _FULL_HISTORY_MAX_PAGES if args.full_history else _DEFAULT_MAX_PAGES
    trades_by_market = _pull_trades(
        metas,
        max_pages=max_pages,
        sleep_between=args.sleep_between,
        full_history=args.full_history,
    )

    resolution_ts_by_slug = {
        meta.slug: _resolution_ts_from_end_date(meta.end_date) for meta in metas
    }
    n_with_resolution = sum(ts is not None for ts in resolution_ts_by_slug.values())
    log.info(
        "pre-resolution filter: %d/%d markets have a usable end_date "
        "(--pre-resolution-days=%.1f)",
        n_with_resolution,
        len(metas),
        args.pre_resolution_days,
    )

    raw_trade_counts = {slug: len(trades) for slug, trades in trades_by_market}

    log.info("cleaning + indexing across %d markets …", len(trades_by_market))
    markets, wallet_index = build_genre_dataset(
        trades_by_market,
        resolution_ts_by_slug=resolution_ts_by_slug,
        pre_resolution_days=args.pre_resolution_days,
    )

    if args.pre_resolution_days > 0 and n_with_resolution > 0:
        dropped = sum(
            raw_trade_counts[m.slug] - m.T
            for m in markets
            if resolution_ts_by_slug.get(m.slug) is not None
        )
        log.info(
            "pre-resolution filter dropped %d trades across markets with end_date",
            dropped,
        )

    if args.tail_trades is not None:
        # The cap is applied *after* retrieval, so --full-history still walks the
        # whole history and --tail-trades only decides how much of it is kept.
        tailed = [_tail(m, args.tail_trades) for m in markets]
        for retrieved, kept in zip(markets, tailed):
            log.info(
                "  %-40s retrieved %d cleaned trades -> kept %d",
                retrieved.slug,
                retrieved.T,
                kept.T,
            )
        markets = tailed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for m in markets:
        save_processed(m, args.output_dir, name=m.slug)
        log.info("  saved %s.parquet (T=%d)", m.slug, m.T)

    save_wallet_index(wallet_index, args.output_dir / "wallet_index.json")
    log.info(
        "wrote wallet_index.json with %d unique wallets across the genre.",
        wallet_index.n_wallets,
    )

    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
