"""CLI: score a Polymarket trade stream trade-by-trade, live or from replay.

Drives `src.inference.online_scorer.OnlineScorer` (the O(1)-per-trade ADF
E-step plus online-EM adaptation) over a stream of raw trades and emits one
JSON score record per trade. Two input modes feed **one** scoring loop:

  * ``--replay <path>`` — a finished JSONL/Parquet capture in the
    `stream_trades.py` sink shape. Records are sorted strictly by
    ``(timestamp, transaction_hash)`` first — the same deterministic tie-break
    `preprocess.clean_trades` uses — and then consumed in that order.
  * ``--live <path>`` — tail an append-only sink that `stream_trades.py` is
    writing. Arrival order *is* the order; a record that goes backwards in
    ``(timestamp, transaction_hash)`` is a corrupt or interleaved sink and is
    rejected rather than silently reordered, because reordering live would mean
    re-scoring trades that have already been emitted.

**No lookahead** is the property this module exists to guarantee: every feature
fed to the filter is a function of trades ``0..t`` only, so deleting the tail of
an input leaves the surviving scores byte-identical. Two consequences worth
naming, because the batch pipeline does the opposite:

  * ``S_bar`` is an *expanding* mean of the sizes seen so far, not the
    whole-market mean `preprocess.compute_features` uses. The batch mean peeks
    at the future; online it cannot exist.
  * Wallet ids are assigned on first appearance. A wallet the warm-start fit
    never saw lands past the end of ``theta_w`` and cold-starts at the
    ``Beta(a, b)`` prior mean, which `OnlineScorer` already handles.

State is per market (``condition_id``): the Kalman/ADF recursion tracks one
market's price path, so a sink carrying several markets gets one independent
`OnlineScorer` each, created on that market's first trade. Their ``theta_w``
adaptation is therefore *not* shared across markets, unlike the batch
hierarchy — see the note in the run README under ``results/streaming/``.

Usage:

    python -m scripts.score_stream --replay data/live/trades.jsonl \\
        --warm-start results/streaming/warm_start.json \\
        --output results/streaming/scores.jsonl

    python -m scripts.score_stream --live data/live/trades.jsonl \\
        --wallet-index data/processed/wallet_index.json --forgetting 0.99

Output is JSONL, one object per scored trade:
``{ts, tx_hash, market, wallet, p_z, p_v, x_mean}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from config.default_params import ModelParams, OnlineScorerConfig, PhiPrior
from src.data.preprocess import WalletIndex, load_wallet_index
from src.inference.online_scorer import OnlineScorer
from src.inference.variational_em import VEMOutput
from src.utils.transforms import logit

log = logging.getLogger("score_stream")

# Fields of one raw trade record, as written by `stream_trades.JsonlTradeSink`
# (i.e. `dataclasses.asdict(RawTrade)`). Replay accepts any superset.
_REQUIRED_FIELDS = ("timestamp", "price", "size", "wallet", "transaction_hash")

# Cold-start observation scale. With no fitted artifact there is no data to
# moment-match against, so the fallback assumes Var[Y] = 1 — the right order of
# magnitude for logit prices on a politics market, which live in roughly
# (-5, 5). `ModelParams.warm_start` applied to that unit variance gives exactly
# the ratios the batch initializer would; only the scale is a guess.
_COLD_START_VAR_Y = 1.0


class OutOfOrderTradeError(RuntimeError):
    """Raised when a live sink hands back a trade older than its predecessor."""


class WarmStart(NamedTuple):
    """Everything a fresh `OnlineScorer` needs, as restored from an artifact.

    Attributes:
        params: Model parameters; ``beta_S``/``beta_Z`` on the internal
            (standardized) covariate scale, matching `VEMOutput.params`.
        theta_w: (n_wallets,) per-wallet propensities on the probability scale,
            indexed by the *same* wallet index the fit used. May be empty.
        m_S: Pooled mean of ``log_size_ratio`` from the fit (standardization).
        s_S: Pooled std of ``log_size_ratio`` from the fit.
        m_Z: Pooled mean of ``E[Z_prev]`` from the fit (centering).
    """

    params: ModelParams
    theta_w: np.ndarray
    m_S: float
    s_S: float
    m_Z: float


class ScoredTrade(NamedTuple):
    """One emitted score record.

    Attributes:
        ts: Trade timestamp in unix seconds, carried through unchanged.
        tx_hash: Transaction hash — the stream's deduplication key.
        market: ``condition_id`` of the market this trade's scorer tracks.
        wallet: Trader's wallet address (not the integer id, so the record is
            readable without the wallet index that produced it).
        p_z: ``q(Z_t = 1)`` — the insider score.
        p_v: ``q(V_t = 1)`` — the high-volatility regime probability.
        x_mean: Collapsed ``E[X_t | Y_{0:t}]``, the filtered efficient price.
    """

    ts: float
    tx_hash: str
    market: str
    wallet: str
    p_z: float
    p_v: float
    x_mean: float

    def to_json(self) -> str:
        """Serialize to one compact JSON line (no trailing newline)."""
        return json.dumps(self._asdict(), separators=(",", ":"))


# ---------------- Warm start ----------------


def warm_start_payload(vem: VEMOutput) -> dict[str, Any]:
    """Serialize a fitted `VEMOutput` into a warm-start artifact dict.

    Keeps only what the streaming scorer consumes — the parameters, the
    per-wallet propensities, and the three centering/standardization constants.
    The posterior marginals and ELBO trace are batch diagnostics with no
    streaming meaning and are dropped, which keeps the artifact small enough to
    live next to a run's scores.

    Args:
        vem: A fitted batch VEM output.

    Returns:
        A JSON-serializable dict accepted by `load_warm_start`.
    """
    return {
        "params": asdict(vem.params),
        "theta_w": [float(v) for v in np.asarray(vem.theta_w).ravel()],
        "m_S": float(vem.m_S),
        "s_S": float(vem.s_S),
        "m_Z": float(vem.m_Z),
    }


def cold_start() -> WarmStart:
    """Build the no-artifact fallback state, with uninformative constants.

    Used when ``--warm-start`` is omitted. ``theta_w`` is empty (every wallet
    cold-starts at the ``Beta(a, b)`` prior mean) and the centering constants
    are the identity transform, so the logistic predictor sees raw covariates.
    Adaptation then has to discover everything from the stream itself.

    Returns:
        A `WarmStart` at `_COLD_START_VAR_Y` observation scale.
    """
    return WarmStart(
        # The `ModelParams.warm_start` ratios written out at var_Y = 1, rather
        # than a call with a fabricated Y array standing in for data we do not
        # have.
        params=ModelParams(
            sigma2_0=0.1 * _COLD_START_VAR_Y,
            sigma2_1=_COLD_START_VAR_Y,
            tau2_0=_COLD_START_VAR_Y,
            tau2_1=0.01 * _COLD_START_VAR_Y,
        ),
        theta_w=np.zeros(0),
        m_S=0.0,
        s_S=1.0,
        m_Z=0.0,
    )


def _warm_start_from_dict(payload: dict[str, Any]) -> WarmStart:
    """Rebuild a `WarmStart` from a decoded artifact dict.

    Args:
        payload: Dict in the `warm_start_payload` shape. A ``best_restart``
            wrapper (the `validate_vem.py` artifact layout) is unwrapped first.

    Returns:
        The restored warm start; missing centering constants fall back to the
        `cold_start` identity values so a partial artifact still runs.

    Raises:
        KeyError: If no ``params`` block is present anywhere in the payload.
    """
    if "params" not in payload and "best_restart" in payload:
        payload = payload["best_restart"]
    if "params" not in payload:
        raise KeyError("warm-start artifact has no 'params' block")
    fallback = cold_start()
    fields = set(ModelParams.__dataclass_fields__)
    params = ModelParams(
        **{k: float(v) for k, v in payload["params"].items() if k in fields}
    )
    return WarmStart(
        params=params,
        theta_w=np.asarray(payload.get("theta_w", []), dtype=float),
        m_S=float(payload.get("m_S", fallback.m_S)),
        s_S=float(payload.get("s_S", fallback.s_S)),
        m_Z=float(payload.get("m_Z", fallback.m_Z)),
    )


def load_warm_start(path: Path) -> WarmStart:
    """Load a warm start from a JSON or pickled VEM artifact.

    Accepts three shapes, so a fit can be handed over however it was saved:
    a JSON dict from `warm_start_payload`, a `validate_vem.py` JSON artifact
    (unwrapped via its ``best_restart`` block), or a pickle holding either a
    `VEMOutput` directly or a `_runner.pickle_run` payload with a ``vem`` key.

    Args:
        path: Artifact path; ``.json`` is parsed as JSON, anything else as a
            pickle (matching `_runner.load_run`).

    Returns:
        The restored `WarmStart`.

    Raises:
        TypeError: If a pickle contains no recognizable VEM output.
    """
    path = Path(path)
    if path.suffix == ".json":
        return _warm_start_from_dict(json.loads(path.read_text(encoding="utf-8")))

    # Deferred: `_runner` pulls in the full data/inference stack, which the
    # JSON path has no need of.
    from scripts._runner import load_run

    payload = load_run(path)
    if isinstance(payload, VEMOutput):
        vem = payload
    elif isinstance(payload, dict) and isinstance(payload.get("vem"), VEMOutput):
        vem = payload["vem"]
    else:
        raise TypeError(f"{path} holds no VEMOutput (got {type(payload).__name__})")
    return _warm_start_from_dict(warm_start_payload(vem))


# ---------------- Input iterators ----------------


def _sort_key(record: dict[str, Any]) -> tuple[float, str]:
    """Return the ``(timestamp, transaction_hash)`` total order of a record."""
    return (float(record["timestamp"]), str(record["transaction_hash"]))


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
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
        records = list(_iter_jsonl(path))
    return sorted(records, key=_sort_key)


def tail_live(path: Path, *, poll_interval: float = 0.5) -> Iterator[dict[str, Any]]:
    """Follow an append-only sink, yielding records as they land.

    Yields whole lines only: a `readline` that comes back without its trailing
    newline caught the writer mid-record, so the read position is rewound and
    retried rather than handing a truncated object to `json.loads`.

    Args:
        path: Sink `stream_trades.py` is appending to; must already exist.
        poll_interval: Seconds to sleep when the file is exhausted.

    Yields:
        One decoded record per appended line, in arrival order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        OutOfOrderTradeError: If a record precedes its predecessor in
            ``(timestamp, transaction_hash)`` order. Live mode never reorders:
            the earlier trades have already been scored and emitted, so the
            only honest response to a backwards record is to stop.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"live sink {path} does not exist")
    prev_key: tuple[float, str] | None = None
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
            key = _sort_key(record)
            if prev_key is not None and key < prev_key:
                raise OutOfOrderTradeError(
                    f"live sink {path} went backwards: trade {key[1]} at "
                    f"t={key[0]:.0f} follows {prev_key[1]} at t={prev_key[0]:.0f}. "
                    "Live mode does not reorder; re-run with --replay to sort."
                )
            prev_key = key
            yield record


# ---------------- Scoring ----------------


class _MarketStream:
    """One market's scorer plus the causal state its features need.

    Holds the running size mean and the previous timestamp, which are exactly
    the two pieces of history the batch feature builder gets for free from
    having the whole market in memory.
    """

    def __init__(
        self,
        warm: WarmStart,
        *,
        config: OnlineScorerConfig,
        prior: PhiPrior | None,
    ) -> None:
        """Start a fresh scorer at trade 0 of one market.

        Args:
            warm: Parameters, propensities and centering constants to start at.
            config: Forgetting / refresh schedule for the online-EM block.
            prior: MAP prior spec; ``None`` uses `PhiPrior` defaults.
        """
        self.scorer = OnlineScorer(
            warm.params,
            warm.theta_w,
            warm.m_S,
            warm.s_S,
            warm.m_Z,
            config=config,
            prior=prior,
        )
        self._prev_ts: float | None = None
        self._n = 0
        self._sum_S = 0.0

    def features(
        self, ts: float, price: float, size: float
    ) -> tuple[float, float, float]:
        """Turn one raw trade into the filter's ``(y, delta, log_size_ratio)``.

        ``S_bar`` is the mean size over trades ``0..t`` *inclusive*, an
        expanding window rather than the batch whole-market mean: including the
        current trade is still causal, and it keeps the very first trade's
        ratio at exactly ``log(1) = 0`` instead of undefined.

        Args:
            ts: Trade timestamp in unix seconds.
            price: Trade price in (0, 1).
            size: Trade size in USDC; must be positive.

        Returns:
            ``(y, delta, log_size_ratio)`` for `OnlineScorer.step`.
        """
        # Clamped at 0: a same-second pair sorted by hash can differ by a
        # negative float epsilon, and the process-variance statistic divides by
        # delta (ARCHITECTURE.md §6.1).
        delta = 0.0 if self._prev_ts is None else max(ts - self._prev_ts, 0.0)
        self._prev_ts = ts
        self._n += 1
        self._sum_S += size
        S_bar = self._sum_S / self._n
        return float(logit(price)), delta, math.log(size / S_bar)


def _is_scorable(record: dict[str, Any]) -> bool:
    """Report whether a raw record survives `preprocess.clean_trades`' filters.

    Same predicate, applied one record at a time so live and replay share it:
    positive size, a price strictly inside (0, 1), and non-empty wallet and
    transaction hash.

    Args:
        record: Raw trade record.

    Returns:
        True when the record is usable, False when it should be dropped.
    """
    if any(record.get(f) is None for f in _REQUIRED_FIELDS):
        return False
    return (
        float(record["size"]) > 0.0
        and 0.0 < float(record["price"]) < 1.0
        and len(str(record["wallet"])) > 0
        and len(str(record["transaction_hash"])) > 0
    )


class StreamScorer:
    """Scores an ordered stream of raw trades, one independent scorer per market.

    The class owns everything that must persist across trades — the wallet
    index, the seen-hash set, and the per-market `_MarketStream` states — and
    nothing that depends on where the trades came from. That is what lets live
    and replay share a single scoring loop.
    """

    def __init__(
        self,
        warm: WarmStart,
        *,
        config: OnlineScorerConfig | None = None,
        prior: PhiPrior | None = None,
        wallet_index: WalletIndex | None = None,
        markets: Iterable[str] | None = None,
    ) -> None:
        """Configure the scorer; no market state exists until the first trade.

        Args:
            warm: Starting parameters/propensities/centering constants, shared
                by every market's scorer.
            config: Forgetting / refresh schedule; ``None`` uses
                `OnlineScorerConfig` defaults.
            prior: MAP prior spec; ``None`` uses `PhiPrior` defaults.
            wallet_index: Address-to-id map to score against — pass the one the
                warm-start fit used so ``theta_w`` lines up. ``None`` builds a
                fresh index in first-appearance order. Mutated in place as new
                addresses arrive.
            markets: Condition ids to keep (case-insensitive); ``None`` scores
                every market in the stream.
        """
        self._warm = warm
        self._config = config if config is not None else OnlineScorerConfig()
        self._prior = prior
        self.wallet_index = wallet_index if wallet_index is not None else WalletIndex()
        self._markets = {str(m).lower() for m in markets} if markets else None
        self._streams: dict[str, _MarketStream] = {}
        self._seen_hashes: set[str] = set()
        self.n_skipped = 0

    @property
    def n_markets(self) -> int:
        """Number of distinct markets that have an open scorer."""
        return len(self._streams)

    def score(self, records: Iterable[dict[str, Any]]) -> Iterator[ScoredTrade]:
        """Score an ordered stream of raw trade records.

        Ordering is the caller's contract (`read_replay` sorts, `tail_live`
        enforces); this loop only ever moves forward, so trade ``i``'s score is
        a function of trades ``0..i`` alone.

        Args:
            records: Raw trade records in stream order.

        Yields:
            One `ScoredTrade` per accepted trade. Invalid records, filtered
            markets, and repeated transaction hashes are skipped and counted in
            ``n_skipped`` — the hash dedupe matters because a live sink that is
            restarted re-reads lines it already appended.
        """
        for record in records:
            if not _is_scorable(record):
                self.n_skipped += 1
                continue
            market = str(record.get("condition_id", ""))
            if self._markets is not None and market.lower() not in self._markets:
                self.n_skipped += 1
                continue
            tx_hash = str(record["transaction_hash"])
            if tx_hash in self._seen_hashes:
                self.n_skipped += 1
                continue
            self._seen_hashes.add(tx_hash)
            yield self._score_one(record, market, tx_hash)

    def _score_one(
        self, record: dict[str, Any], market: str, tx_hash: str
    ) -> ScoredTrade:
        """Advance one market's scorer by a single validated trade.

        Args:
            record: Raw trade record, already validated by `_is_scorable`.
            market: Condition id keying this trade's scorer.
            tx_hash: Transaction hash, carried onto the output record.

        Returns:
            The `ScoredTrade` for this trade.
        """
        stream = self._streams.get(market)
        if stream is None:
            stream = _MarketStream(self._warm, config=self._config, prior=self._prior)
            self._streams[market] = stream
            log.debug("opened scorer for market %s", market or "<unknown>")

        wallet = str(record["wallet"])
        ts = float(record["timestamp"])
        y, delta, log_size_ratio = stream.features(
            ts, float(record["price"]), float(record["size"])
        )
        out = stream.scorer.step(
            y, delta, log_size_ratio, self.wallet_index.add(wallet)
        )
        return ScoredTrade(
            ts=ts,
            tx_hash=tx_hash,
            market=market,
            wallet=wallet,
            p_z=float(out.Z_prob),
            p_v=float(out.V_prob),
            x_mean=float(out.X_mean),
        )


# ---------------- CLI ----------------


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
        help="Tail an append-only sink written by stream_trades.py. "
        "Out-of-order records are rejected, never reordered.",
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

    log.info(
        "done — scored %d trades (%d skipped) across %d market(s) to %s",
        n_scored,
        scorer.n_skipped,
        scorer.n_markets,
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
