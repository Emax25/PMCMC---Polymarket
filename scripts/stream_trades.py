"""CLI: record the live Polymarket trade feed to an append-only JSONL file.

Connects to the public RTDS WebSocket (`src/data/rtds.py`), normalizes every
activity/trades frame onto the same `RawTrade` shape the historical pipeline
uses, and appends one JSON object per trade. The sink is append-only and
flushed per record, so a Ctrl-C (or a killed box) can lose at most the trades
that never arrived — never half of one that did.

Usage:

    python -m scripts.stream_trades --output data/live/trades.jsonl
    python -m scripts.stream_trades --markets 0xabc... 0xdef...   # filter
    python -m scripts.stream_trades --parquet-every 500           # compact
    python -m scripts.stream_trades --max-trades 100              # smoke run

Stop with Ctrl-C; the run exits 0 after closing the sink and (if requested)
writing a final Parquet compaction of everything captured this session.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import FrameType

import pandas as pd

from src.data.polymarket_api import RawTrade
from src.data.rtds import RTDS_URL, RTDSClient, RTDSSocket, default_socket_factory

log = logging.getLogger("stream_trades")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for stream_trades."""
    p = argparse.ArgumentParser(description="Record the live Polymarket trade feed.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/live/trades.jsonl"),
        help="Append-only JSONL sink; parent directories are created.",
    )
    p.add_argument(
        "--markets",
        nargs="+",
        default=None,
        help="Condition ids to keep (case-insensitive). Default: every market.",
    )
    p.add_argument(
        "--parquet-every",
        type=int,
        default=0,
        help="Compact the session's trades to Parquet every N records "
        "(0 disables). The Parquet file is rewritten in full each time.",
    )
    p.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Parquet compaction target. Default: --output with a .parquet suffix.",
    )
    p.add_argument(
        "--max-trades",
        type=int,
        default=None,
        help="Stop after recording N trades. Default: run until interrupted.",
    )
    p.add_argument(
        "--url",
        default=RTDS_URL,
        help="RTDS WebSocket endpoint.",
    )
    p.add_argument(
        "--stale-after",
        type=float,
        default=120.0,
        help="Warn when no frame has arrived for this many seconds.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log a progress line every N recorded trades (0 disables).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


class JsonlTradeSink:
    """Append-only JSONL sink writing one whole line per trade.

    Each record is serialized to a single string ending in ``\\n`` and handed to
    one `write` followed by a `flush`, so an interrupt can never leave a
    half-written record behind for the next reader to choke on. The file is
    opened in append mode, so re-running the CLI extends an existing capture
    instead of clobbering it.
    """

    def __init__(self, path: Path) -> None:
        """Open `path` for appending, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")
        self.n_written = 0

    def append(self, trade: RawTrade) -> None:
        """Serialize and durably append one trade."""
        line = json.dumps(asdict(trade), separators=(",", ":")) + "\n"
        self._fh.write(line)
        self._fh.flush()
        self.n_written += 1

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()


def compact_to_parquet(jsonl_path: Path, parquet_path: Path) -> int:
    """Rewrite a JSONL capture as a Parquet file.

    Full rewrite rather than an incremental append: capture volumes here are
    thousands of rows, so the simplicity is worth more than the I/O, and a
    whole-file rewrite can never leave a partially-appended row group.

    Args:
        jsonl_path: Source capture; malformed trailing lines are ignored.
        parquet_path: Destination; parent directories are created.

    Returns:
        Number of records written.
    """
    records = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Only reachable if an external writer truncated the file; our
                # own sink flushes whole lines. Drop it rather than abort.
                log.warning("skipping unparseable line in %s", jsonl_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(parquet_path, index=False)
    return len(records)


def _install_sigterm_handler() -> None:
    """Make SIGTERM raise KeyboardInterrupt so shutdown has one code path."""

    def _handler(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _handler)


def main(
    argv: list[str] | None = None,
    *,
    socket_factory: Callable[[], RTDSSocket] | None = None,
) -> int:
    """Stream live trades to disk until interrupted or `--max-trades` is hit.

    Args:
        argv: Argument list passed to argparse; defaults to ``sys.argv[1:]``.
        socket_factory: Test seam — injects a fake socket in place of a real
            RTDS connection. Production runs leave this as ``None``.

    Returns:
        Exit code (0 on success, including a clean Ctrl-C shutdown).
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _install_sigterm_handler()

    factory = socket_factory or (lambda: default_socket_factory(args.url))
    client = RTDSClient(
        socket_factory=factory,
        condition_ids=args.markets,
        stale_after=args.stale_after,
    )
    parquet_path = args.parquet_path or args.output.with_suffix(".parquet")

    sink = JsonlTradeSink(args.output)
    log.info("recording RTDS trades to %s", sink.path)
    if args.markets:
        log.info("market filter active: %d condition id(s)", len(args.markets))

    try:
        for trade in client.stream(max_trades=args.max_trades):
            sink.append(trade)
            if args.log_every and sink.n_written % args.log_every == 0:
                log.info("recorded %d trades", sink.n_written)
            if args.parquet_every and sink.n_written % args.parquet_every == 0:
                n = compact_to_parquet(sink.path, parquet_path)
                log.info("compacted %d records to %s", n, parquet_path)
    except KeyboardInterrupt:
        log.info("interrupted — closing sink cleanly")
    finally:
        sink.close()

    if args.parquet_every and sink.n_written:
        n = compact_to_parquet(sink.path, parquet_path)
        log.info("final compaction: %d records to %s", n, parquet_path)

    log.info("done — %d trades recorded to %s", sink.n_written, sink.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
