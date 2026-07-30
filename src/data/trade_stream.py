"""Trade-stream readers: ordered replay of a capture, and live tailing of a sink.

Both readers hand raw trade records (the `stream_trades.JsonlTradeSink` shape,
i.e. `dataclasses.asdict(RawTrade)`) to a consumer in the order that consumer is
allowed to score them. The two ordering contracts differ on purpose:

  * `read_replay` imposes the total ``(timestamp, transaction_hash)`` order —
    the same deterministic tie-break `preprocess.clean_trades` uses — so a
    finished capture replays identically however its lines were stored.
  * `tail_live` preserves *arrival* order and only rejects a record stamped
    before its predecessor, because reordering a live sink would mean un-emitting
    trades that were already scored. Same-second arrivals are normal (Polymarket
    timestamps are second-resolution) and the hash tie-break is deliberately not
    enforced there.

Corruption policy is shared: a blank line is skipped silently and an
unparseable line is skipped with a warning, never raised. Captures are written
by an append-only sink that can be truncated by an outside writer or read
mid-write, and dropping one bad line is preferable to aborting a run and
discarding the scorer state built up so far. `tail_live` additionally rewinds a
line that arrives without its trailing newline, which is a writer caught
mid-record rather than a record to parse.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


class OutOfOrderTradeError(RuntimeError):
    """Raised when a live sink hands back a trade older than its predecessor."""


def _sort_key(record: dict[str, Any]) -> tuple[float, str]:
    """Return the ``(timestamp, transaction_hash)`` total order of a record."""
    return (float(record["timestamp"]), str(record["transaction_hash"]))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed records from a JSONL capture, skipping unparseable lines.

    Args:
        path: JSONL file written one whole JSON object per line.

    Yields:
        One decoded record per well-formed line.
    """
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping unparseable line in %s", path)


def read_replay(path: Path) -> list[dict[str, Any]]:
    """Read a whole capture and return its records in canonical stream order.

    Sorting by ``(timestamp, transaction_hash)`` is what makes replay a
    lookahead-free evaluation substrate: the scorer sees exactly the order the
    market produced, with same-second ties broken deterministically by hash
    (`preprocess.clean_trades` uses the same key, so replay and the batch
    pipeline agree on what "trade i" means).

    Args:
        path: ``.parquet`` capture, or JSONL for any other suffix.

    Returns:
        Records sorted in stream order. Order is total, so the result does not
        depend on the order they were stored in.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        records = pd.read_parquet(path).to_dict("records")
    else:
        records = list(iter_jsonl(path))
    return sorted(records, key=_sort_key)


def tail_live(path: Path, *, poll_interval: float = 0.5) -> Iterator[dict[str, Any]]:
    """Follow an append-only sink, yielding records as they land.

    Yields whole lines only: a `readline` that comes back without its trailing
    newline caught the writer mid-record, so the read position is rewound and
    retried rather than handing a truncated object to `json.loads`.

    Monotonicity is checked on ``timestamp`` alone, not on the full
    ``(timestamp, transaction_hash)`` replay key: Polymarket timestamps are
    second-resolution, so a busy market routinely appends several trades within
    one second, and their hashes arrive in whatever order the fills did. Holding
    a live sink to the sorted-order tie-break would reject roughly half of those
    same-second pairs. The hash stays purely `StreamScorer`'s dedupe key.

    Args:
        path: Sink `stream_trades.py` is appending to; must already exist.
        poll_interval: Seconds to sleep when the file is exhausted.

    Yields:
        One decoded record per appended line, in arrival order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        OutOfOrderTradeError: If a record's ``timestamp`` precedes its
            predecessor's. Live mode never reorders: the earlier trades have
            already been scored and emitted, so the only honest response to a
            trade from the past is to stop.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"live sink {path} does not exist")
    prev_ts: float | None = None
    with path.open(encoding="utf-8") as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line or not line.endswith("\n"):
                fh.seek(pos)
                time.sleep(poll_interval)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping unparseable line in %s", path)
                continue
            ts = float(record["timestamp"])
            if prev_ts is not None and ts < prev_ts:
                raise OutOfOrderTradeError(
                    f"live sink {path} went backwards in time: trade "
                    f"{record['transaction_hash']} is stamped t={ts:.0f}, "
                    f"{prev_ts - ts:.0f}s before the trade already scored at "
                    f"t={prev_ts:.0f}. Live mode scores in arrival order and "
                    "cannot un-emit those trades; re-run with --replay to sort "
                    "the capture instead."
                )
            prev_ts = ts
            yield record
