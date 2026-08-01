"""CLI: pull Kalshi public trade history into the repo's normalized schema.

For each ticker on the command line:
  1. Resolve `/markets/{ticker}` → `KalshiMarketMeta` (title, status,
     close_time).
  2. Walk `GetTrades` — the newest `DEFAULT_MAX_TRADES` trades by default, or
     the entire history under `--full-history`.
  3. Normalize + clean (drop dust/out-of-range prices, dedupe on trade id,
     sort by (timestamp, trade id)), then apply the pre-resolution filter.
  4. Optionally tail to the last `--tail-trades` rows (§8.2 budget).
  5. Save `<ticker>.parquet` + `<ticker>.meta.json` under `--output-dir`.

This is the Kalshi counterpart of `scripts/pull_data.py` and follows its
conventions (market ids, `--output-dir`, `--pre-resolution-days`,
`--full-history`, per-market failure tolerance). It stops one step short of it:
Kalshi publishes **no account identifier**, so there is no wallet index to
build and no `ProcessedMarket` to write — the wallet-anchored θ_w prior cannot
be estimated from this venue. What lands on disk is the cleaned, normalized
trade table with a null `wallet` column, which the anonymous-mode data loader
consumes.

`--mode` is the CLI half of the anonymous-mode contract: `auto` (the default)
infers the mode from wallet nullability, and an explicit `anonymous`/`wallet`
overrides it. The resolved mode is recorded in the sidecar metadata so
consumers read it rather than re-deriving it. Asking for `wallet` mode on a
Kalshi pull is an error — the identity it needs does not exist upstream.

Usage:

    python -m scripts.pull_kalshi --tickers KXZELENSKYYOUT-26JUL01
    python -m scripts.pull_kalshi --tickers T1 T2 --full-history
    python -m scripts.pull_kalshi --tickers T1 --pre-resolution-days 0

The script is idempotent over the output directory — re-running overwrites.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.data.kalshi_api import (
    DEFAULT_MAX_TRADES,
    KalshiAPIError,
    KalshiMarketMeta,
    fetch_market,
    fetch_trades,
)
from src.data.polymarket_api import RawTrade
from src.data.preprocess import (
    _resolution_ts_from_end_date,
    clean_trades,
    filter_pre_resolution,
    trades_to_dataframe,
)

log = logging.getLogger("pull_kalshi")

# Safety budget on HTTP GETs per market. Kalshi's cursor pagination has no
# offset ceiling, so `--full-history` only lifts the client-side row budget;
# the page budget stays the same generous ceiling in both modes.
_MAX_PAGES = 20_000

_MODE_ANONYMOUS = "anonymous"
_MODE_WALLET = "wallet"
_MODE_AUTO = "auto"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for pull_kalshi."""
    p = argparse.ArgumentParser(description="Pull Kalshi public trade history.")
    p.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Kalshi market tickers, e.g. KXZELENSKYYOUT-26JUL01.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/kalshi"),
        help="Directory to write <ticker>.parquet + <ticker>.meta.json files.",
    )
    p.add_argument(
        "--tail-trades",
        type=int,
        default=None,
        help="Keep only the last N trades per market (§8.2 target: 500-3000). "
        "Default: keep all surviving trades.",
    )
    p.add_argument(
        "--full-history",
        action="store_true",
        help="Walk the cursor to the market's first trade instead of stopping "
        f"at the newest {DEFAULT_MAX_TRADES} (DEFAULT_MAX_TRADES). Slow; off "
        "by default.",
    )
    p.add_argument(
        "--sleep-between",
        type=float,
        default=0.1,
        help="Seconds between paginated GetTrades calls (politeness).",
    )
    p.add_argument(
        "--pre-resolution-days",
        type=float,
        default=7.0,
        help="Drop trades within N days of market close (default: 7). "
        "Pass 0 to disable pre-resolution filtering.",
    )
    p.add_argument(
        "--mode",
        choices=(_MODE_AUTO, _MODE_ANONYMOUS, _MODE_WALLET),
        default=_MODE_AUTO,
        help="Identity mode recorded in the sidecar metadata. 'auto' infers it "
        "from wallet nullability; the explicit values override that inference.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def _pull_market(
    ticker: str,
    *,
    max_trades: int | None,
    sleep_between: float,
) -> tuple[KalshiMarketMeta, list[RawTrade]]:
    """Fetch one market's metadata and trades.

    Args:
        ticker: Kalshi market ticker.
        max_trades: Row budget passed through to `fetch_trades`; None pulls the
            full history.
        sleep_between: Seconds between paginated calls.

    Returns:
        ``(meta, trades)`` for the ticker.

    Raises:
        KalshiAPIError: On any HTTP or schema failure; the caller decides
            whether one bad market aborts the run.
    """
    meta = fetch_market(ticker)
    log.info(
        "%-32s %-10s vol=%-12.0f %s",
        meta.ticker,
        meta.status,
        meta.volume,
        meta.title[:60],
    )
    trades = fetch_trades(
        ticker,
        max_trades=max_trades,
        max_pages=_MAX_PAGES,
        sleep_between=sleep_between,
    )
    return meta, trades


def _resolve_mode(df: pd.DataFrame, requested: str) -> str:
    """Resolve the identity mode for a cleaned Kalshi trade table.

    Under ``auto`` the mode is read off the data — an all-null wallet column
    means the venue published no identity — which is the data-load contract the
    anonymous-mode consumers share. An explicit ``wallet`` request on a table
    with no wallets is a hard error rather than a silent downgrade: it means
    the caller believes in identity this source cannot supply.

    Args:
        df: Cleaned trade table with a ``wallet`` column.
        requested: One of ``auto``, ``anonymous``, ``wallet``.

    Returns:
        The resolved mode, ``anonymous`` or ``wallet``.

    Raises:
        ValueError: If ``wallet`` mode is requested but no row carries a wallet.
    """
    has_identity = bool(df["wallet"].notna().any())
    if requested == _MODE_WALLET and not has_identity:
        raise ValueError(
            "--mode wallet requested but no trade carries a wallet; Kalshi "
            "publishes no account identifier (see src/data/kalshi_api.py)"
        )
    if requested != _MODE_AUTO:
        return requested
    return _MODE_WALLET if has_identity else _MODE_ANONYMOUS


def _prepare_table(
    trades: list[RawTrade],
    meta: KalshiMarketMeta,
    *,
    pre_resolution_days: float,
    tail_trades: int | None,
) -> pd.DataFrame:
    """Clean, filter, and tail one market's normalized trades.

    Cleaning runs with ``require_wallet=False`` because every Kalshi row is
    legitimately wallet-less; the default filter would empty the table.

    Args:
        trades: Normalized trades as returned by `fetch_trades`.
        meta: Market metadata, read for the close time.
        pre_resolution_days: Exclusion window in days before market close; 0
            disables the filter.
        tail_trades: Keep only the last N rows, or None to keep all.

    Returns:
        Cleaned trade table sorted by (timestamp, trade id); may be empty.
    """
    df = clean_trades(trades_to_dataframe(trades), require_wallet=False)
    if pre_resolution_days > 0:
        resolution_ts = _resolution_ts_from_end_date(meta.close_time)
        if resolution_ts is None:
            log.warning(
                "  %s: no usable close_time; pre-resolution filter skipped",
                meta.ticker,
            )
        df = filter_pre_resolution(df, resolution_ts, days=pre_resolution_days)
    if tail_trades is not None and len(df) > tail_trades:
        df = df.iloc[len(df) - tail_trades :].reset_index(drop=True)
    return df


def _save_market(
    df: pd.DataFrame,
    meta: KalshiMarketMeta,
    directory: Path,
    *,
    mode: str,
) -> Path:
    """Write one market's trade table plus its sidecar metadata.

    Args:
        df: Cleaned trade table.
        meta: Market metadata for the sidecar.
        directory: Destination directory; created if missing.
        mode: Resolved identity mode recorded for downstream consumers.

    Returns:
        Path of the Parquet file written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / f"{meta.ticker}.parquet"
    df.to_parquet(parquet_path, index=False)
    (directory / f"{meta.ticker}.meta.json").write_text(
        json.dumps(
            {
                "source": "kalshi",
                "ticker": meta.ticker,
                "title": meta.title,
                "status": meta.status,
                "close_time": meta.close_time,
                "volume": meta.volume,
                "mode": mode,
                "n_trades": int(len(df)),
            }
        )
    )
    return parquet_path


def main(argv: list[str] | None = None) -> int:
    """Fetch, clean, and persist Kalshi trade history for the given tickers.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 when at least one market was saved — a pull that lost some
        markets to API failures still writes the rest and logs an ``INCOMPLETE``
        line naming them — and 1 when every market failed.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    max_trades = None if args.full_history else DEFAULT_MAX_TRADES
    saved: list[str] = []
    failed: list[str] = []

    for ticker in args.tickers:
        t0 = time.monotonic()
        try:
            meta, trades = _pull_market(
                ticker,
                max_trades=max_trades,
                sleep_between=args.sleep_between,
            )
        except KalshiAPIError as err:
            # One bad ticker must not discard the markets already pulled: a
            # --full-history backfill costs minutes to hours per market.
            log.error(
                "  -> FAILED %s after %.1fs: %s", ticker, time.monotonic() - t0, err
            )
            failed.append(ticker)
            continue

        log.info("  -> %d raw trades in %.1fs", len(trades), time.monotonic() - t0)
        df = _prepare_table(
            trades,
            meta,
            pre_resolution_days=args.pre_resolution_days,
            tail_trades=args.tail_trades,
        )
        if df.empty:
            log.error("  -> FAILED %s: no trades survived cleaning/filtering", ticker)
            failed.append(ticker)
            continue

        mode = _resolve_mode(df, args.mode)
        path = _save_market(df, meta, args.output_dir, mode=mode)
        log.info("  saved %s (rows=%d, mode=%s)", path.name, len(df), mode)
        saved.append(ticker)

    if not saved:
        log.error("every ticker failed to pull (%s) — nothing saved", ", ".join(failed))
        return 1

    if failed:
        # Repeated at the end because the per-market error scrolled past long
        # ago on a multi-market backfill, and this is the line a reader uses to
        # decide whether the directory is a complete dataset.
        log.error(
            "done — INCOMPLETE: %d of %d ticker(s) failed and were not saved: %s",
            len(failed),
            len(args.tickers),
            ", ".join(failed),
        )
    else:
        log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
