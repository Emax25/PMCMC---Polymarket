"""CLI: score a Polymarket trade stream trade-by-trade, live or from replay.

Thin CLI over `src.inference.stream_scoring.StreamScorer` (the scoring loop and
its warm-start handling) and `src.data.trade_stream` (the two readers). The
model-level invariants — no lookahead, the expanding ``S_bar``, per-market
scorer state — are documented in `stream_scoring`; this file only wires input
mode, settings and sinks together.

Two input modes feed **one** scoring loop:

  * ``--replay <path>`` — a finished JSONL/Parquet capture in the
    `stream_trades.py` sink shape. Records are sorted strictly by
    ``(timestamp, transaction_hash)`` first — the same deterministic tie-break
    `preprocess.clean_trades` uses — and then consumed in that order.
  * ``--live <path>`` — tail an append-only sink that `stream_trades.py` is
    writing. Arrival order *is* the order; only a record whose ``timestamp``
    precedes its predecessor's is rejected, because reordering live would mean
    re-scoring trades that have already been emitted. Same-second arrivals are
    normal (Polymarket timestamps are second-resolution) and the hash tie-break
    replay applies is deliberately *not* enforced here — a live sink is in
    arrival order, not sorted order.

Usage:

    python -m scripts.score_stream --replay data/live/trades.jsonl \\
        --warm-start results/streaming/warm_start.json \\
        --output results/streaming/scores.jsonl

    python -m scripts.score_stream --live data/live/trades.jsonl \\
        --wallet-index data/processed/wallet_index.json --forgetting 0.99

Output is JSONL, one object per scored trade:
``{ts, tx_hash, market, wallet, p_z, p_v, x_mean}``, plus a deterministic
``<output>.meta.json`` sidecar naming the run's mode, input and settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from config.default_params import OnlineScorerConfig
from src.data.preprocess import WalletIndex, load_wallet_index
from src.data.trade_stream import (
    OutOfOrderTradeError,
    iter_jsonl,
    read_replay,
    tail_live,
)
from src.inference.stream_scoring import StreamScorer, cold_start, load_warm_start

log = logging.getLogger("score_stream")


def _scored_hashes(path: Path) -> set[str]:
    """Collect the ``tx_hash`` values already present in a scores JSONL."""
    return {
        str(record["tx_hash"])
        for record in iter_jsonl(path)
        if record.get("tx_hash") is not None
    }


def _write_run_meta(args: argparse.Namespace, *, replay: bool) -> Path:
    """Write the ``<output>.meta.json`` sidecar describing one finished run.

    A scores JSONL carries no provenance — the same 500 lines could have come
    from a cold start or from a warm-started run with a different forgetting
    factor, which changes what the numbers mean. The sidecar records the run's
    inputs so an artifact can be read months later. It is a *sidecar* precisely
    so the scores file stays byte-identical across identical runs, and it holds
    no clock reading for the same reason.

    Args:
        args: Parsed CLI arguments of the run that just finished.
        replay: True for ``--replay``, False for ``--live``.

    Returns:
        The sidecar path written.
    """
    meta = {
        "mode": "replay" if replay else "live",
        "input": str(args.replay if replay else args.live),
        "forgetting": float(args.forgetting),
        # None is the default and means "never refresh beta_S/beta_Z", which is
        # a genuine setting, not a missing one — recorded as JSON null.
        "n_refresh": int(args.n_refresh) if args.n_refresh is not None else None,
        "warm_start": str(args.warm_start) if args.warm_start is not None else None,
        "wallet_index": (
            str(args.wallet_index) if args.wallet_index is not None else None
        ),
        "output": str(args.output),
    }
    path = args.output.with_name(args.output.name + ".meta.json")
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for score_stream."""
    p = argparse.ArgumentParser(
        description="Score a Polymarket trade stream, live or from replay."
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Score a finished JSONL/Parquet capture, sorted by "
        "(timestamp, transaction_hash). No lookahead: state never sees a "
        "future trade.",
    )
    source.add_argument(
        "--live",
        type=Path,
        default=None,
        help="Tail an append-only sink written by stream_trades.py. A record "
        "stamped before an already-scored trade is rejected, never reordered.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/streaming/scores.jsonl"),
        help="JSONL score sink; truncated in replay mode, appended in live "
        "mode. Parent directories are created.",
    )
    p.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Fitted VEM artifact (.json or pickle) supplying params, theta_w "
        "and the centering constants. Default: uninformative cold start.",
    )
    p.add_argument(
        "--wallet-index",
        type=Path,
        default=None,
        help="wallet_index.json from pull_data.py, so --warm-start theta_w "
        "lines up with wallet addresses. Default: index built from the stream.",
    )
    p.add_argument(
        "--markets",
        nargs="+",
        default=None,
        help="Condition ids to score (case-insensitive). Default: every market.",
    )
    p.add_argument(
        "--forgetting",
        type=float,
        default=OnlineScorerConfig.forgetting,
        help="Online-EM forgetting factor lambda in (0, 1]; 1.0 freezes the "
        "parameters at the warm start.",
    )
    p.add_argument(
        "--n-refresh",
        type=int,
        default=OnlineScorerConfig.n_refresh,
        help="Trades between IRLS refreshes of beta_S/beta_Z; omit or pass a "
        "non-positive value to never refresh them.",
    )
    p.add_argument(
        "--max-trades",
        type=int,
        default=None,
        help="Stop after scoring N trades. Default: whole capture (replay) or "
        "until interrupted (live).",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Live mode: seconds to sleep when the sink is exhausted.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log a progress line every N scored trades (0 disables).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Score a replayed capture or a live sink to a JSONL of per-trade scores.

    A successful run also drops a ``<output>.meta.json`` sidecar (see
    `_write_run_meta`) so the scores can be traced back to their inputs.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success (including a clean Ctrl-C in live mode), 1 when
        a live sink is rejected for going backwards in time.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.warm_start is not None:
        warm = load_warm_start(args.warm_start)
        log.info("warm start from %s (%d wallets)", args.warm_start, warm.theta_w.size)
    else:
        warm = cold_start()
        log.warning("no --warm-start: cold-starting from uninformative defaults")

    wallet_index = (
        load_wallet_index(args.wallet_index) if args.wallet_index else WalletIndex()
    )
    scorer = StreamScorer(
        warm,
        config=OnlineScorerConfig(
            forgetting=args.forgetting, n_refresh=args.n_refresh
        ),
        wallet_index=wallet_index,
        markets=args.markets,
    )

    replay = args.replay is not None
    if replay:
        records: Iterable[dict[str, Any]] = read_replay(args.replay)
        log.info("replaying %d records from %s", len(records), args.replay)
    else:
        records = tail_live(args.live, poll_interval=args.poll_interval)
        log.info("following %s", args.live)
        if args.output.exists():
            # Live mode appends, and `tail_live` restarts at byte 0 of the sink,
            # so the trades already in the output would otherwise be scored and
            # appended a second time.
            already = _scored_hashes(args.output)
            scorer.mark_seen(already)
            log.info(
                "resuming %s: %d trade(s) already scored will be skipped",
                args.output,
                len(already),
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_scored = 0
    # Truncate on replay (a replay is a pure function of its input, so an old
    # run's scores are stale), append on live (the sink and its scores grow
    # together across restarts).
    with args.output.open("w" if replay else "a", encoding="utf-8") as fh:
        try:
            for scored in scorer.score(records):
                fh.write(scored.to_json() + "\n")
                fh.flush()
                n_scored += 1
                if args.log_every and n_scored % args.log_every == 0:
                    log.info("scored %d trades", n_scored)
                if args.max_trades is not None and n_scored >= args.max_trades:
                    break
        except OutOfOrderTradeError as err:
            log.error("%s", err)
            return 1
        except KeyboardInterrupt:
            log.info("interrupted — closing score sink cleanly")

    meta_path = _write_run_meta(args, replay=replay)
    log.info(
        "done — scored %d trades (%d skipped) across %d market(s) to %s (run "
        "described in %s)",
        n_scored,
        scorer.n_skipped,
        scorer.n_markets,
        args.output,
        meta_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
