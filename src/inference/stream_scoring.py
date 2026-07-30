"""Streaming insider scoring: one `OnlineScorer` per market over a trade stream.

Drives `src.inference.online_scorer.OnlineScorer` (the O(1)-per-trade ADF E-step
plus online-EM adaptation) over an ordered stream of raw trade records and emits
one `ScoredTrade` per trade. `scripts/score_stream.py` is a thin CLI over this
module — import these names from here, never from `scripts/`.

**No lookahead** is the property this module exists to guarantee: every feature
fed to the filter is a function of trades ``0..t`` only, so deleting the tail of
an input leaves the surviving scores byte-identical. Ordering is the caller's
contract; `src.data.trade_stream` supplies both readers that satisfy it. Two
consequences worth naming, because the batch pipeline does the opposite:

  * ``S_bar`` is an *expanding* mean of the sizes seen so far, not the
    whole-market mean `preprocess.compute_features` uses. The batch mean peeks
    at the future; online it cannot exist.
  * Wallet ids are assigned on first appearance. A wallet the warm-start fit
    never saw lands past the end of ``theta_w`` and cold-starts at the
    ``Beta(a, b)`` prior mean, which `OnlineScorer` already handles.

State is per market (``condition_id``): the Kalman/ADF recursion tracks one
market's price path, so a stream carrying several markets gets one independent
`OnlineScorer` each, created on that market's first trade. Their ``theta_w``
adaptation is therefore *not* shared across markets, unlike the batch
hierarchy — see the note in the run README under ``results/streaming/``.

A run starts from a `WarmStart`: either a fitted batch `VEMOutput` restored
through `load_warm_start`, or the uninformative `cold_start` fallback.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from config.default_params import ModelParams, OnlineScorerConfig, PhiPrior
from src.data.preprocess import WalletIndex
from src.inference.online_scorer import OnlineScorer
from src.inference.variational_em import VEMOutput
from src.utils.transforms import logit

log = logging.getLogger(__name__)

# Fields of one raw trade record, as written by `stream_trades.JsonlTradeSink`
# (i.e. `dataclasses.asdict(RawTrade)`). Replay accepts any superset.
_REQUIRED_FIELDS = ("timestamp", "price", "size", "wallet", "transaction_hash")

# Cold-start observation scale. With no fitted artifact there is no data to
# moment-match against, so the fallback assumes Var[Y] = 1 — the right order of
# magnitude for logit prices on a politics market, which live in roughly
# (-5, 5). `ModelParams.warm_start` applied to that unit variance gives exactly
# the ratios the batch initializer would; only the scale is a guess.
_COLD_START_VAR_Y = 1.0


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

    A partial artifact is only accepted when the substitution it forces is
    harmless. ``beta_S``/``beta_Z`` are fitted against *standardized*
    covariates, so filling in the `cold_start` identity centering would feed
    them raw ``log_size_ratio`` and ``E[Z_prev]`` — every score silently
    mis-scaled. That case raises; a fit whose betas are zero has an inert
    logistic predictor, so it only warns.

    Args:
        payload: Dict in the `warm_start_payload` shape. A ``best_restart``
            wrapper (the `validate_vem.py` artifact layout) is unwrapped first.

    Returns:
        The restored warm start; missing centering constants fall back to the
        `cold_start` identity values, which is safe only for zero betas.

    Raises:
        KeyError: If no ``params`` block is present anywhere in the payload.
        ValueError: If a centering constant is missing while ``beta_S`` or
            ``beta_Z`` is non-zero.
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
    missing = [k for k in ("m_S", "s_S", "m_Z") if k not in payload]
    if missing:
        if params.beta_S != 0.0 or params.beta_Z != 0.0:
            raise ValueError(
                "warm-start artifact is missing the centering constant(s) "
                f"{', '.join(missing)} while carrying beta_S="
                f"{params.beta_S:.6g}, beta_Z={params.beta_Z:.6g}. Those betas "
                "are on the fit's standardized covariate scale, so scoring "
                "without (m_S, s_S, m_Z) would apply them to raw covariates "
                "and mis-scale every score. Re-dump the fit with "
                "stream_scoring.warm_start_payload (a current validate_vem.py "
                "artifact already carries them)."
            )
        log.warning(
            "warm-start artifact is missing %s; substituting the cold-start "
            "identity centering (m_S=%.1f, s_S=%.1f, m_Z=%.1f), which is exact "
            "here only because beta_S and beta_Z are both zero",
            ", ".join(missing),
            fallback.m_S,
            fallback.s_S,
            fallback.m_Z,
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
    # An upstream sink can carry a field no one coerced — a string price
    # ("n/a"), a list where a number belongs. Unscorable is the right verdict,
    # not a crash: in live mode the exception would take down a run that is
    # otherwise healthy, discarding the scorer state built up so far.
    try:
        return (
            float(record["size"]) > 0.0
            and 0.0 < float(record["price"]) < 1.0
            and len(str(record["wallet"])) > 0
            and len(str(record["transaction_hash"])) > 0
        )
    except (TypeError, ValueError):
        return False


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

    def mark_seen(self, hashes: Iterable[str]) -> None:
        """Treat these transaction hashes as already scored.

        The dedupe set lives in this process, but a live run's score sink
        outlives it: `tail_live` re-reads its sink from byte 0 on every start, so
        without this a restart re-scores — and re-appends — every trade already
        in the output. Seeding from that output makes the append idempotent.

        Note:
            Only ``ScoredTrade`` records are suppressed, not scorer state: the
            re-read trades still count toward ``n_skipped``, and the filter
            recursion starts fresh from the warm start, exactly as it would
            without the pre-seeding.

        Args:
            hashes: Transaction hashes whose scores are already persisted.
        """
        self._seen_hashes.update(str(h) for h in hashes)

    def score(self, records: Iterable[dict[str, Any]]) -> Iterator[ScoredTrade]:
        """Score an ordered stream of raw trade records.

        Ordering is the caller's contract (`trade_stream.read_replay` sorts,
        `trade_stream.tail_live` enforces); this loop only ever moves forward, so
        trade ``i``'s score is a function of trades ``0..i`` alone.

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
