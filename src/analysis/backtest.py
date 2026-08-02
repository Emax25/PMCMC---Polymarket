"""Costed tradeability check on ``P(Z)`` — **detection-signal PoC, not alpha**.

**Read this first.** Nothing in this module is a validated trading strategy, and
nothing it emits is a trading recommendation. It answers one narrow,
proof-of-concept question: *does the detector's per-trade insider score survive
a realistic spread and taker fee at all?* A negative answer would say the score
is untradeable; a positive answer says only that a single pre-declared threshold
family cleared costs on a handful of markets, under an honest multiple-testing
correction. Validated alpha is explicitly out of scope for this work — see
`_POC_NOTE`, which is stamped into every artifact this module writes.

**The pre-declared strategy family (fixed before any real-data run).** One
position per market, at most:

  * enter on the *first* scored trade whose ``P(Z) >= tau``, for ``tau`` in the
    declared grid `DECLARED_THRESHOLD_GRID`;
  * take the side the filtered price currently favours — YES when the filtered
    probability is at or above 0.5, NO otherwise;
  * hold to resolution. There is no stop, no sizing rule, and no exit rule to
    search over, which is the point: the smaller the family, the less there is
    to deflate away.

The grid is disclosed in full in the JSON output whether or not a threshold was
selected, so the trial count behind the headline Sharpe is never implicit.

**Costs.** Entry pays half the quoted spread plus Kalshi's taker fee,
``0.07 * p * (1 - p)`` per contract (`CostModel`). That fee is maximal at
``p = 0.5`` and vanishes at the bounds — *exactly* the wrong shape for this
detector, whose insider signal is weakest mid-book and strongest in the tails
where a well-informed trade moves a near-resolved price. A costed check is
therefore not a formality here; the fee is anti-correlated with the signal.
Resolution settles at $1/$0 with no exit fee, so a held-to-resolution position
pays costs once.

**Purged, embargoed walk-forward** (Lopez de Prado). Folds are contiguous blocks
of markets in close-time order; a training market is dropped whenever its label
window ``[first trade, close]`` reaches into the test block's label span, or
within `DEFAULT_EMBARGO_S` of it. Threshold selection happens on the training
block only and is applied to the test block, so the reported returns are
out-of-sample by construction.

**Deflated Sharpe** (Bailey & Lopez de Prado 2014). The headline number is
deflated against ``SR0 = sqrt(Var[trial Sharpes]) * ((1 - gamma) Z^-1[1 - 1/N]
+ gamma Z^-1[1 - 1/(N e)])`` — the **empirical variance across the trial
Sharpes**, not the trial count alone. This distinction is the whole point: the
thresholds on a grid are strongly correlated, so their Sharpes barely differ,
the variance is small, and the honest deflator is small too. Substituting a
count-only deflator would inflate ``SR0`` and understate the strategy, which is
the standard failure mode of this correction.

**No lookahead** is inherited, not re-derived: every score consumed here comes
from `scripts/score_stream.py --replay`, and `read_replay_provenance` (reused
from `src.analysis.event_study`) refuses any score file whose sidecar does not
say ``mode == "replay"``. The entry price is the filtered ``E[X_t | Y_{0:t}]``
mapped through the sigmoid, which is a function of trades ``0..t`` alone.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kurtosis, norm, skew

# `_market_id_from_record` is reached for deliberately: it is the repo's one
# rule for "which field names a market in a metadata record", and a second copy
# here could disagree with the event study about a market's identity — the two
# analyses would then silently be about different market sets.
from src.analysis.event_study import DAY_SECONDS, _market_id_from_record

log = logging.getLogger(__name__)

BACKTEST_SCHEMA_VERSION = 1

# ---- The pre-declared strategy family ----
#
# Nine thresholds spanning "barely elevated" to "the detector is confident".
# Declared here, in source, so the trial count is a property of the code rather
# than of whichever grid a run happened to pass. `run_backtest` reports
# ``len(thresholds)`` as ``n_trials`` even when some threshold took no trade:
# a trial that was searched and found empty was still searched.
DECLARED_THRESHOLD_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)

SIDE_YES = "yes"
SIDE_NO = "no"

# ---- Cost model defaults ----
#
# Kalshi's taker fee is ceil(0.07 * C * p * (1 - p)) rounded up to the cent on an
# order of C contracts (see `src.data.kalshi_api`). 2c is a deliberately
# unflattering standing spread for a politics market: thin books quote wider,
# and a PoC that assumes a tight one is measuring the assumption.
KALSHI_FEE_RATE = 0.07
DEFAULT_SPREAD = 0.02
_CENTS_PER_DOLLAR = 100.0

DEFAULT_N_SPLITS = 4
DEFAULT_EMBARGO_S = 1.0 * DAY_SECONDS

# Euler-Mascheroni, as it appears in the expected-maximum term of the DSR.
EULER_GAMMA = 0.5772156649015329

_POC_NOTE = (
    "DETECTION-SIGNAL EVALUATION, NOT A VALIDATED ALPHA STRATEGY. This is a "
    "proof-of-concept tradeability check on P(Z): it asks only whether the "
    "detector's score survives spreads and taker fees, on one pre-declared "
    "threshold family, over however many markets happened to be scored. It is "
    "not a backtest of a deployable strategy, it is not evidence of alpha, and "
    "nothing in this output is a trading recommendation."
)

_DSR_NOTE = (
    "Deflated Sharpe (Bailey & Lopez de Prado 2014) is the probabilistic Sharpe "
    "ratio evaluated against SR0 = sqrt(Var[trial Sharpes]) * ((1 - gamma) "
    "Z^-1[1 - 1/N] + gamma Z^-1[1 - 1/(N e)]). The deflator uses the EMPIRICAL "
    "VARIANCE ACROSS THE TRIAL SHARPES, not the raw trial count: adjacent "
    "thresholds on a grid select overlapping trade sets, so their Sharpes are "
    "strongly correlated and their variance is small. A count-only deflator "
    "would overstate SR0 and understate the strategy."
)

_COST_NOTE = (
    "Entry pays half the quoted spread plus a taker fee of "
    "fee_rate * p * (1 - p) per contract; resolution settles at $1/$0 with no "
    "exit fee. The fee is maximal at p = 0.5 and vanishes at the bounds, i.e. "
    "it is largest exactly where this detector's signal is weakest."
)

_SPLIT_NOTE = (
    "Purged, embargoed walk-forward: folds are contiguous blocks of markets in "
    "close-time order, and a training market whose label window [first trade, "
    "close] reaches into the test block's label span, or within the embargo of "
    "it, is dropped. Thresholds are selected on training blocks only."
)

# Exclusion reasons, named so the module, the CLI and the tests agree.
REASON_NO_RESOLUTION = "no resolution metadata"
REASON_NO_OUTCOME = "no resolved outcome"
REASON_NO_TRADES_BEFORE_CLOSE = "no scored trades at or before t_close"


# ---------------- Cost model ----------------


@dataclass(frozen=True)
class CostModel:
    """Spread and taker-fee model for a held-to-resolution contract.

    All quantities are per contract, in dollars, on a contract that pays $1 if
    its side resolves true and $0 otherwise. Entering costs

        cost(p, side) = (p if side is YES else 1 - p) + spread / 2
                        + fee_rate * p * (1 - p)

    and the position's net return is ``payout - cost``. Half the spread is
    charged because a taker crosses from mid to the far touch on whichever side
    it lifts; the fee is charged on both sides identically because
    ``p (1 - p)`` is symmetric in ``p``.

    Attributes:
        spread: Quoted bid-ask spread in probability units. Entry pays half.
        fee_rate: Taker-fee coefficient; Kalshi's is `KALSHI_FEE_RATE`.
    """

    spread: float = DEFAULT_SPREAD
    fee_rate: float = KALSHI_FEE_RATE

    def __post_init__(self) -> None:
        """Reject a negative spread or fee rate.

        Raises:
            ValueError: If ``spread`` or ``fee_rate`` is negative. A negative
                cost would be a rebate this venue does not pay, and it would
                silently manufacture edge.
        """
        if self.spread < 0.0:
            raise ValueError(f"spread must be non-negative, got {self.spread!r}")
        if self.fee_rate < 0.0:
            raise ValueError(f"fee_rate must be non-negative, got {self.fee_rate!r}")

    @property
    def half_spread(self) -> float:
        """Cost of crossing from mid to one touch."""
        return 0.5 * self.spread

    def taker_fee(self, p: float) -> float:
        """Continuous per-contract taker fee ``fee_rate * p * (1 - p)``.

        The continuous form is what the per-contract return arithmetic uses; the
        venue's cent rounding (`taker_fee_dollars`) is immaterial on a
        one-contract position and would make the returns a step function of the
        price for no analytical gain.

        Args:
            p: Contract price in ``[0, 1]``, i.e. the implied probability.

        Returns:
            Dollars per contract. Maximal at ``p = 0.5``, zero at 0 and 1.
        """
        return self.fee_rate * p * (1.0 - p)

    def taker_fee_dollars(self, p: float, contracts: int) -> float:
        """Kalshi's billed taker fee: ``0.07 * C * p * (1 - p)``, rounded up.

        `src.data.kalshi_api` documents the venue formula as
        ``ceil(0.07 * C * p * (1 - p))`` with the worked example "~1.75c per
        contract at p = 0.5", so the ceiling is at cent granularity, not dollar
        granularity. This method implements the worked example.

        Args:
            p: Contract price in ``[0, 1]``.
            contracts: Order size in contracts.

        Returns:
            The billed fee in dollars.
        """
        cents = self.fee_rate * contracts * p * (1.0 - p) * _CENTS_PER_DOLLAR
        # Round before the ceiling. 0.07 * 100 * 0.5 * 0.5 is 175.00000000000003
        # cents in binary floating point, and a bare ceil() would bill $1.76 for
        # an exactly-$1.75 fee. Nine digits is far below any real fee's
        # resolution and far above the accumulated representation error.
        return math.ceil(round(cents, 9)) / _CENTS_PER_DOLLAR

    def entry_cost(self, p: float, side: str) -> float:
        """All-in cost of taking one contract of ``side`` at mid price ``p``.

        Args:
            p: Mid price of the YES contract, in ``[0, 1]``.
            side: `SIDE_YES` or `SIDE_NO`.

        Returns:
            Dollars per contract, spread and fee included.

        Raises:
            ValueError: If ``side`` is neither `SIDE_YES` nor `SIDE_NO`.
        """
        if side == SIDE_YES:
            mid = p
        elif side == SIDE_NO:
            mid = 1.0 - p
        else:
            raise ValueError(f"side must be {SIDE_YES!r} or {SIDE_NO!r}, got {side!r}")
        return mid + self.half_spread + self.taker_fee(p)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "spread": self.spread,
            "half_spread": self.half_spread,
            "fee_rate": self.fee_rate,
            "note": _COST_NOTE,
        }


# ---------------- Inputs ----------------


@dataclass(frozen=True)
class MarketPanel:
    """One resolved market's replayed scores, ready to trade against.

    Attributes:
        market: Market id, as the score records carry it.
        ts: (n,) trade timestamps in unix seconds, sorted ascending, all at or
            before ``close_ts``.
        p_z: (n,) ``q(Z_t = 1)`` per trade — the entry signal.
        price: (n,) filtered probability ``sigmoid(E[X_t | Y_{0:t}])``, the
            price a position is opened at.
        close_ts: Resolution time in unix seconds.
        outcome: Realized YES payout, 1.0 or 0.0.
    """

    market: str
    ts: np.ndarray
    p_z: np.ndarray
    price: np.ndarray
    close_ts: float
    outcome: float

    @property
    def label_start(self) -> float:
        """Earliest timestamp any position in this market could be opened at."""
        return float(self.ts[0])


@dataclass(frozen=True)
class ExcludedMarket:
    """One market that carries scores but cannot be traded.

    Attributes:
        market: Market id.
        reason: Why it was dropped, in words a report can print verbatim.
    """

    market: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {"market": self.market, "reason": self.reason}


def build_panels(
    scores_by_market: Mapping[str, Any],
    close_by_market: Mapping[str, float],
    outcome_by_market: Mapping[str, float],
) -> tuple[list[MarketPanel], list[ExcludedMarket]]:
    """Join replayed scores with close times and realized outcomes.

    Trades after ``close_ts`` are dropped rather than traded: a fill stamped
    after resolution is either a data error or a settlement print, and either
    way it is not something a live reader could have taken.

    Args:
        scores_by_market: ``{market: MarketScores}``, as
            `src.analysis.event_study.load_scores` returns.
        close_by_market: ``{market: t_close}``, unix seconds.
        outcome_by_market: ``{market: realized YES payout}``, 1.0 or 0.0.

    Returns:
        ``(panels, excluded)`` with ``panels`` sorted by close time — the order
        the walk-forward splits consume — and ``excluded`` naming every market
        that had scores but no tradeable label.
    """
    # Deferred: `transforms` is cheap, but keeping the import local matches the
    # rest of this module's analysis-only import surface.
    from src.utils.transforms import sigmoid

    panels: list[MarketPanel] = []
    excluded: list[ExcludedMarket] = []
    for market in sorted(scores_by_market):
        scores = scores_by_market[market]
        close_ts = close_by_market.get(market)
        if close_ts is None:
            excluded.append(ExcludedMarket(market, REASON_NO_RESOLUTION))
            continue
        outcome = outcome_by_market.get(market)
        if outcome is None:
            excluded.append(ExcludedMarket(market, REASON_NO_OUTCOME))
            continue
        keep = scores.ts <= float(close_ts)
        if not np.any(keep):
            excluded.append(ExcludedMarket(market, REASON_NO_TRADES_BEFORE_CLOSE))
            continue
        panels.append(
            MarketPanel(
                market=market,
                ts=scores.ts[keep],
                p_z=scores.p_z[keep],
                price=np.asarray(sigmoid(scores.x_mean[keep]), dtype=float),
                close_ts=float(close_ts),
                outcome=float(outcome),
            ),
        )
    panels.sort(key=lambda panel: (panel.close_ts, panel.market))
    return panels, excluded


def _iter_metadata_records(path: Path) -> list[tuple[str, Any]]:
    """Walk a metadata path into ``(key, record)`` pairs.

    Accepts the same three shapes as
    `src.analysis.event_study.load_resolutions` — a directory of
    ``*.meta.json`` sidecars, a JSON object, or a JSON array — so a run can
    point ``--outcomes`` at whatever the pull step left behind, and at the same
    path as ``--resolutions``.

    Args:
        path: Directory of sidecars, or a JSON file.

    Returns:
        ``(key, value)`` pairs; ``key`` is the file stem or the object key, and
        is only a fallback for a record that names no market itself.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a JSON file is malformed, or is neither object nor array.
    """
    if not path.exists():
        raise FileNotFoundError(f"outcome metadata not found: {path}")
    suffix = ".meta.json"
    records: list[tuple[str, Any]] = []
    if path.is_dir():
        for sidecar in sorted(path.glob(f"*{suffix}")):
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{sidecar}: malformed JSON ({exc})") from exc
            records.append((sidecar.name[: -len(suffix)], payload))
        return records
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed JSON ({exc})") from exc
    if isinstance(payload, dict):
        return list(payload.items())
    if isinstance(payload, list):
        return [(f"row{i}", row) for i, row in enumerate(payload)]
    raise ValueError(
        f"{path}: expected a JSON object or array of metadata records, got "
        f"{type(payload).__name__}",
    )


_OUTCOME_KEYS = ("outcome", "result", "settlement", "winning_outcome", "resolved_to")
_TRUE_TOKENS = frozenset({"yes", "y", "true", "1", "up"})
_FALSE_TOKENS = frozenset({"no", "n", "false", "0", "down"})


def _outcome_from_value(value: Any) -> float | None:
    """Coerce one metadata value into a YES payout of 1.0 or 0.0.

    Accepts the venues' two spellings: a string verdict (Kalshi's ``result`` is
    ``"yes"``/``"no"``) or a number (a 0/1 payout). Anything else — an unsettled
    market, a mid-range price, a ``null`` — returns None so the market is
    excluded and counted rather than traded against a guess.

    Args:
        value: The raw field value.

    Returns:
        1.0, 0.0, or None when the value does not resolve the market.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        # Only the two settled payouts are accepted. A 0.97 is a *price*, not a
        # resolution, and trading it as one would book a certain profit that
        # never happened.
        if float(value) in (0.0, 1.0):
            return float(value)
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return 1.0
        if token in _FALSE_TOKENS:
            return 0.0
    return None


def load_outcomes(path: str | Path) -> dict[str, float]:
    """Load ``{market: realized YES payout}`` from resolution metadata.

    Args:
        path: Directory of ``*.meta.json`` sidecars, or a JSON file holding an
            object (``{market: outcome}`` or ``{market: {...}}``) or an array of
            records.

    Returns:
        ``{market: 1.0 or 0.0}``. Markets whose record carries no settled
        outcome are omitted and warned about; `build_panels` then excludes them
        with `REASON_NO_OUTCOME`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a JSON file is malformed or is neither object nor array.
    """
    out: dict[str, float] = {}
    for key, value in _iter_metadata_records(Path(path)):
        if not isinstance(value, Mapping):
            outcome = _outcome_from_value(value)
            if outcome is None:
                log.warning("outcome record for %s is not a settled payout", key)
                continue
            out[str(key)] = outcome
            continue
        outcome = None
        for field in _OUTCOME_KEYS:
            if field in value:
                outcome = _outcome_from_value(value[field])
                if outcome is not None:
                    break
        if outcome is None:
            log.warning("record for %s carries no settled outcome; skipped", key)
            continue
        out[_market_id_from_record(value, str(key))] = outcome
    return out


# ---------------- Positions ----------------


@dataclass(frozen=True)
class Position:
    """One held-to-resolution contract, priced net of costs.

    Attributes:
        market: Market the position is in.
        threshold: The ``tau`` whose crossing opened it.
        entry_ts: Timestamp of the trade that triggered entry.
        exit_ts: Resolution time.
        p_z: The score at entry.
        price: Filtered YES probability at entry, before costs.
        side: `SIDE_YES` or `SIDE_NO`.
        cost: All-in entry cost per contract.
        payout: Realized settlement, 1.0 if the taken side resolved true.
        ret: ``payout - cost`` — net PnL per contract, on $1 of max payout.
    """

    market: str
    threshold: float
    entry_ts: float
    exit_ts: float
    p_z: float
    price: float
    side: str
    cost: float
    payout: float
    ret: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "market": self.market,
            "threshold": self.threshold,
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "p_z": self.p_z,
            "price": self.price,
            "side": self.side,
            "cost": self.cost,
            "payout": self.payout,
            "net_return": self.ret,
        }


def open_position(
    panel: MarketPanel,
    threshold: float,
    cost_model: CostModel,
) -> Position | None:
    """Apply the pre-declared entry rule to one market.

    Enters on the *first* trade with ``p_z >= threshold`` and takes the side the
    filtered price favours, which is the only directional rule available: a
    `ScoredTrade` carries no taker side, so "follow the informed trade's
    direction" is not expressible from the score stream and is not attempted.

    Args:
        panel: The market's scores, prices, close time and outcome.
        threshold: Entry threshold on ``P(Z)``.
        cost_model: Spread and fee model.

    Returns:
        The `Position`, or None when no trade crossed the threshold or when the
        all-in cost is at least $1 — paying a dollar or more for a contract that
        pays at most a dollar is never done, so the trade is skipped rather than
        booked as a certain loss.
    """
    crossings = np.flatnonzero(panel.p_z >= threshold)
    if crossings.size == 0:
        return None
    i = int(crossings[0])
    price = float(panel.price[i])
    side = SIDE_YES if price >= 0.5 else SIDE_NO
    cost = cost_model.entry_cost(price, side)
    if cost >= 1.0:
        return None
    payout = panel.outcome if side == SIDE_YES else 1.0 - panel.outcome
    return Position(
        market=panel.market,
        threshold=float(threshold),
        entry_ts=float(panel.ts[i]),
        exit_ts=panel.close_ts,
        p_z=float(panel.p_z[i]),
        price=price,
        side=side,
        cost=cost,
        payout=float(payout),
        ret=float(payout) - cost,
    )


def _returns(positions: Sequence[Position]) -> np.ndarray:
    """Net per-contract returns of ``positions``, as a float array."""
    return np.asarray([p.ret for p in positions], dtype=float)


# ---------------- Purged, embargoed walk-forward ----------------


@dataclass(frozen=True)
class Split:
    """One walk-forward fold over markets.

    Attributes:
        fold: 0-based fold index.
        train: Indices of training markets, already purged and embargoed.
        test: Indices of test markets, contiguous in close-time order.
        test_start: Earliest label start across the test markets — the point the
            purge measures back from.
        test_end: Latest close time across the test markets.
        n_purged: Training candidates dropped by the purge and embargo.
    """

    fold: int
    train: np.ndarray
    test: np.ndarray
    test_start: float
    test_end: float
    n_purged: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view; index arrays become lists."""
        return {
            "fold": self.fold,
            "n_train": int(self.train.size),
            "n_test": int(self.test.size),
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_purged": self.n_purged,
        }


def purged_walk_forward(
    label_start: np.ndarray,
    label_end: np.ndarray,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_s: float = DEFAULT_EMBARGO_S,
) -> list[Split]:
    """Build purged, embargoed walk-forward folds over labelled observations.

    Observations are ordered by ``label_end`` and cut into ``n_splits + 1``
    contiguous blocks; fold ``i`` tests on block ``i + 1`` and trains on
    everything before it. A training observation is kept only when its label
    window ends at least ``embargo_s`` before the test block's earliest label
    start — so no training label overlaps a test label, which is the property
    purging exists to guarantee (Lopez de Prado). Here a label window is
    ``[first scored trade, resolution]``: a position's outcome is not known
    until the market resolves, so two markets whose lives overlap share
    information no matter how far apart their entries are.

    Args:
        label_start: (n,) time each observation's label window opens.
        label_end: (n,) time each observation's label window closes.
        n_splits: Number of test folds; needs ``n_splits + 1`` blocks of data.
        embargo_s: Extra gap, in seconds, required between the end of a training
            label and the start of the test block's label span.

    Returns:
        One `Split` per fold with a non-empty test block, in time order. Folds
        whose training block is emptied by the purge are still returned — a
        caller that needs a training set decides what to do about it, and
        silently dropping them would hide how much the purge removed.

    Raises:
        ValueError: If ``n_splits < 1``, ``embargo_s < 0``, or the two label
            arrays disagree in length.
    """
    label_start = np.asarray(label_start, dtype=float)
    label_end = np.asarray(label_end, dtype=float)
    if label_start.shape != label_end.shape:
        raise ValueError(
            f"label_start {label_start.shape} and label_end {label_end.shape} "
            "must have the same shape",
        )
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}")
    if embargo_s < 0.0:
        raise ValueError(f"embargo_s must be non-negative, got {embargo_s}")

    n = int(label_end.size)
    if n == 0:
        return []
    order = np.argsort(label_end, kind="stable")
    # `array_split` gives blocks differing by at most one element, so a short
    # panel degrades to fewer, smaller folds instead of raising. Empty blocks
    # (n < n_splits + 1) are skipped below.
    blocks = np.array_split(order, n_splits + 1)

    splits: list[Split] = []
    for fold, test in enumerate(blocks[1:]):
        if test.size == 0:
            continue
        test_start = float(label_start[test].min())
        test_end = float(label_end[test].max())
        candidates = np.concatenate(blocks[: fold + 1])
        keep = label_end[candidates] <= test_start - embargo_s
        splits.append(
            Split(
                fold=fold,
                train=np.sort(candidates[keep]),
                test=np.sort(test),
                test_start=test_start,
                test_end=test_end,
                n_purged=int(candidates.size - int(keep.sum())),
            ),
        )
    return splits


# ---------------- Sharpe, PSR, DSR ----------------


def sharpe_ratio(returns: np.ndarray) -> float:
    """Per-trade Sharpe ratio ``mean / sd`` of ``returns``.

    Deliberately *not* annualized. Positions here are one-per-market and held
    for however long the market runs, so there is no periodicity to annualize
    over; scaling by an invented ``sqrt(252)`` would inflate the headline number
    without adding information. The DSR below is computed in the same
    per-trade units, so the two are consistent.

    Args:
        returns: (n,) net per-contract returns.

    Returns:
        The ratio, or NaN when fewer than two trades were taken or the returns
        have zero dispersion — both cases where a Sharpe is undefined rather
        than infinite.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return math.nan
    sd = float(returns.std(ddof=1))
    if sd <= 0.0:
        return math.nan
    return float(returns.mean()) / sd


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float) -> float:
    """Probabilistic Sharpe ratio: ``P(true SR > sr_benchmark)``.

    Bailey & Lopez de Prado (2014), equation for PSR:

        PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(T - 1)
                        / sqrt(1 - g3 * SR_hat + (g4 - 1) / 4 * SR_hat^2) )

    with ``g3`` the skewness and ``g4`` the (non-excess) kurtosis of the return
    series. The moment terms are what make this more than a t-test: negatively
    skewed, fat-tailed returns — which is exactly what "sell the favourite and
    hold to resolution" produces — inflate the denominator and lower the PSR
    relative to a Gaussian assumption.

    Args:
        returns: (n,) net per-contract returns.
        sr_benchmark: Threshold Sharpe, in the same per-trade units.

    Returns:
        A probability in ``(0, 1)``, or NaN when the Sharpe itself is undefined
        or the variance term is non-positive (possible only at extreme sample
        moments, where the normal approximation has broken down anyway).
    """
    returns = np.asarray(returns, dtype=float)
    sr = sharpe_ratio(returns)
    if not math.isfinite(sr):
        return math.nan
    n = int(returns.size)
    g3 = float(skew(returns, bias=False))
    g4 = float(kurtosis(returns, fisher=False, bias=False))
    variance = 1.0 - g3 * sr + 0.25 * (g4 - 1.0) * sr * sr
    if variance <= 0.0:
        return math.nan
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(variance)))


def expected_max_sharpe(trial_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe under the null, over ``n_trials`` trials.

        SR0 = sqrt(V) * ( (1 - gamma) * Z^-1[1 - 1/N]
                          + gamma * Z^-1[1 - 1/(N e)] )

    with ``V`` the **variance across the trial Sharpes** and ``gamma`` the
    Euler-Mascheroni constant. This is the deflator, and the variance is the
    part that matters: a family of correlated trials (adjacent thresholds
    selecting overlapping trade sets) has a small ``V`` and therefore a small
    ``SR0``, whereas a count-only deflator would charge the full expected
    maximum of ``N`` *independent* trials and understate the strategy.

    Args:
        trial_variance: Sample variance of the trial Sharpes, per-trade units.
        n_trials: Number of trials searched, including trials that took no
            trade — a searched-and-empty trial was still searched.

    Returns:
        The benchmark Sharpe, increasing in ``n_trials``. Exactly 0.0 when
        fewer than two trials were run or the trial Sharpes are identical:
        with no dispersion across trials the selection had nothing to select on,
        so there is nothing to deflate.

    Raises:
        ValueError: If ``trial_variance`` is negative.
    """
    if trial_variance < 0.0:
        raise ValueError(f"trial_variance must be non-negative, got {trial_variance}")
    if n_trials < 2 or trial_variance == 0.0:
        return 0.0
    z_max = float(norm.ppf(1.0 - 1.0 / n_trials))
    z_e = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return math.sqrt(trial_variance) * ((1.0 - EULER_GAMMA) * z_max + EULER_GAMMA * z_e)


@dataclass(frozen=True)
class DeflatedSharpe:
    """The deflated-Sharpe verdict on one selected strategy.

    Attributes:
        sharpe: Per-trade Sharpe of the selected strategy's returns.
        n_returns: Trades behind ``sharpe``.
        skewness: Sample skewness of those returns.
        kurtosis: Sample (non-excess) kurtosis of those returns.
        n_trials: Trials searched — the full declared grid, disclosed.
        n_trial_sharpes: Trials that produced a defined Sharpe and therefore
            contributed to ``trial_variance``.
        trial_variance: Empirical variance across the trial Sharpes.
        sr_benchmark: ``SR0`` from `expected_max_sharpe`.
        psr_zero: PSR against 0 — the undeflated number, shown only so the size
            of the deflation is visible.
        dsr: PSR against ``sr_benchmark``. This is the headline statistic.
    """

    sharpe: float
    n_returns: int
    skewness: float
    kurtosis: float
    n_trials: int
    n_trial_sharpes: int
    trial_variance: float
    sr_benchmark: float
    psr_zero: float
    dsr: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view, with the method note attached."""
        return {
            "sharpe": self.sharpe,
            "n_returns": self.n_returns,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "n_trials": self.n_trials,
            "n_trial_sharpes": self.n_trial_sharpes,
            "trial_variance": self.trial_variance,
            "sr_benchmark": self.sr_benchmark,
            "psr_zero": self.psr_zero,
            "deflated_sharpe": self.dsr,
            "note": _DSR_NOTE,
        }


def deflated_sharpe_ratio(
    returns: np.ndarray,
    trial_sharpes: Sequence[float],
    *,
    n_trials: int | None = None,
) -> DeflatedSharpe:
    """Deflate a selected strategy's Sharpe against the trials it was picked from.

    Args:
        returns: (n,) net returns of the selected strategy.
        trial_sharpes: Sharpe of every trial searched. Non-finite entries (a
            threshold that took fewer than two trades) are dropped from the
            variance but still counted in ``n_trials``.
        n_trials: Trials searched; defaults to ``len(trial_sharpes)``. Pass it
            explicitly to disclose a grid larger than the finite Sharpes
            available.

    Returns:
        The populated `DeflatedSharpe`. ``dsr`` is NaN when the strategy's own
        Sharpe is undefined — no trades, one trade, or zero dispersion.
    """
    returns = np.asarray(returns, dtype=float)
    trials = np.asarray(list(trial_sharpes), dtype=float)
    finite = trials[np.isfinite(trials)]
    declared = int(n_trials) if n_trials is not None else int(trials.size)
    # ddof=1: the trial Sharpes are a sample of what the search could have
    # produced, and with a short grid the difference from ddof=0 is not small.
    variance = float(finite.var(ddof=1)) if finite.size > 1 else 0.0
    sr_benchmark = expected_max_sharpe(variance, declared)
    sr = sharpe_ratio(returns)
    has_moments = returns.size > 2
    return DeflatedSharpe(
        sharpe=sr,
        n_returns=int(returns.size),
        skewness=float(skew(returns, bias=False)) if has_moments else math.nan,
        kurtosis=(
            float(kurtosis(returns, fisher=False, bias=False))
            if has_moments
            else math.nan
        ),
        n_trials=declared,
        n_trial_sharpes=int(finite.size),
        trial_variance=variance,
        sr_benchmark=sr_benchmark,
        psr_zero=probabilistic_sharpe_ratio(returns, 0.0),
        dsr=probabilistic_sharpe_ratio(returns, sr_benchmark),
    )


# ---------------- The run ----------------


@dataclass(frozen=True)
class FoldResult:
    """One walk-forward fold's selection and its out-of-sample outcome.

    Attributes:
        fold: 0-based fold index.
        n_train: Training markets left after the purge and embargo.
        n_test: Test markets.
        n_purged: Training candidates the purge and embargo removed.
        test_start: Earliest test label start, unix seconds.
        test_end: Latest test close, unix seconds.
        threshold: Threshold selected on the training block, or None when the
            block was empty or every trial there was undefined.
        train_sharpe: In-sample Sharpe of the selected threshold; NaN when no
            selection happened.
        positions: Out-of-sample positions the selection produced.
    """

    fold: int
    n_train: int
    n_test: int
    n_purged: int
    test_start: float
    test_end: float
    threshold: float | None
    train_sharpe: float
    positions: list[Position]

    @property
    def returns(self) -> np.ndarray:
        """Net returns of this fold's out-of-sample positions."""
        return _returns(self.positions)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view including the individual positions."""
        returns = self.returns
        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_purged": self.n_purged,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "threshold": self.threshold,
            "train_sharpe": self.train_sharpe,
            "n_positions": int(returns.size),
            "mean_return": float(returns.mean()) if returns.size else math.nan,
            "total_return": float(returns.sum()),
            "positions": [p.to_dict() for p in self.positions],
        }


@dataclass(frozen=True)
class TrialResult:
    """One declared threshold's out-of-sample record, held fixed across folds.

    These are the *trials* the deflated Sharpe deflates against — each one is a
    strategy that could have been reported, so each one has to be disclosed.

    Attributes:
        threshold: The declared ``tau``.
        n_positions: Out-of-sample positions it took.
        mean_return: Mean net return per position.
        total_return: Summed net return.
        hit_rate: Fraction of positions that settled in the money.
        sharpe: Per-trade Sharpe; NaN below two positions.
    """

    threshold: float
    n_positions: int
    mean_return: float
    total_return: float
    hit_rate: float
    sharpe: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "threshold": self.threshold,
            "n_positions": self.n_positions,
            "mean_return": self.mean_return,
            "total_return": self.total_return,
            "hit_rate": self.hit_rate,
            "sharpe": self.sharpe,
        }


def _trial_result(threshold: float, positions: Sequence[Position]) -> TrialResult:
    """Summarize one fixed-threshold trial's out-of-sample positions."""
    returns = _returns(positions)
    return TrialResult(
        threshold=float(threshold),
        n_positions=int(returns.size),
        mean_return=float(returns.mean()) if returns.size else math.nan,
        total_return=float(returns.sum()),
        hit_rate=(
            float(np.mean([p.payout > 0.0 for p in positions]))
            if positions
            else math.nan
        ),
        sharpe=sharpe_ratio(returns),
    )


@dataclass(frozen=True)
class BacktestSummary:
    """Everything one costed-backtest run produces.

    Attributes:
        thresholds: The full declared grid, disclosed whether or not each
            threshold was ever selected.
        cost_model: Spread and fee model the run priced with.
        n_splits: Requested walk-forward folds.
        embargo_s: Embargo in seconds.
        n_markets: Markets seen in the scores file.
        panels: Markets that were tradeable.
        excluded: Markets with scores but no tradeable label.
        exclusion_counts: ``{reason: count}`` over ``excluded``.
        folds: Per-fold selection and out-of-sample results.
        trials: Per-threshold fixed-strategy results, pooled out-of-sample.
        deflated: The deflated-Sharpe verdict on the walk-forward strategy.
        provenance: The replay sidecar of the scores file, carried through.
    """

    thresholds: tuple[float, ...]
    cost_model: CostModel
    n_splits: int
    embargo_s: float
    n_markets: int
    panels: int
    excluded: list[ExcludedMarket]
    exclusion_counts: dict[str, int]
    folds: list[FoldResult]
    trials: list[TrialResult]
    deflated: DeflatedSharpe
    provenance: dict[str, Any]

    @property
    def returns(self) -> np.ndarray:
        """Pooled out-of-sample returns of the walk-forward strategy."""
        if not self.folds:
            return np.zeros(0)
        return np.concatenate([fold.returns for fold in self.folds])

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the whole run, PoC framing first."""
        returns = self.returns
        return {
            "schema_version": BACKTEST_SCHEMA_VERSION,
            "framing": _POC_NOTE,
            "declared_threshold_grid": list(self.thresholds),
            "n_trials_disclosed": len(self.thresholds),
            "cost_model": self.cost_model.to_dict(),
            "walk_forward": {
                "n_splits": self.n_splits,
                "embargo_seconds": self.embargo_s,
                "embargo_days": self.embargo_s / DAY_SECONDS,
                "note": _SPLIT_NOTE,
            },
            "n_markets": self.n_markets,
            "n_tradeable": self.panels,
            "n_excluded": len(self.excluded),
            "exclusion_counts": dict(self.exclusion_counts),
            "excluded": [row.to_dict() for row in self.excluded],
            "out_of_sample": {
                "n_positions": int(returns.size),
                "mean_return": float(returns.mean()) if returns.size else math.nan,
                "total_return": float(returns.sum()),
            },
            "deflated_sharpe": self.deflated.to_dict(),
            "trials": [row.to_dict() for row in self.trials],
            "folds": [row.to_dict() for row in self.folds],
            "provenance": self.provenance,
        }


def _select_threshold(
    train_positions: Mapping[float, list[Position]],
    thresholds: Sequence[float],
) -> tuple[float | None, float]:
    """Pick the training-best threshold, breaking ties toward the lowest.

    The tie-break is fixed rather than arbitrary so a run is reproducible; the
    lowest threshold is preferred because it takes the most trades and is
    therefore the least over-fit member of a tied set.

    A negative in-sample Sharpe is still selected when it is the best on offer.
    Adding a "sit out unless the training Sharpe is positive" rule would be a
    second, undeclared parameter fitted on the same data, and the fold's
    ``train_sharpe`` is reported so a reader can see when that happened.

    Args:
        train_positions: ``{threshold: in-sample positions}``.
        thresholds: The declared grid, in the order ties are broken.

    Returns:
        ``(threshold, train Sharpe)``; ``(None, nan)`` when no threshold has a
        defined in-sample Sharpe.
    """
    best_tau: float | None = None
    best_sr = -math.inf
    for tau in thresholds:
        sr = sharpe_ratio(_returns(train_positions[tau]))
        if math.isfinite(sr) and sr > best_sr:
            best_tau, best_sr = float(tau), sr
    return (best_tau, best_sr) if best_tau is not None else (None, math.nan)


def run_backtest(
    panels: Sequence[MarketPanel],
    *,
    thresholds: Sequence[float] = DECLARED_THRESHOLD_GRID,
    cost_model: CostModel | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_s: float = DEFAULT_EMBARGO_S,
    n_markets: int | None = None,
    excluded: Sequence[ExcludedMarket] = (),
    provenance: Mapping[str, Any] | None = None,
) -> BacktestSummary:
    """Run the purged walk-forward backtest and deflate its Sharpe.

    Each fold selects a threshold on its purged training block and applies it to
    the test block; the pooled test returns are the strategy's out-of-sample
    record. Every declared threshold is *also* run fixed across the same folds,
    and those trial Sharpes — not their count — supply the DSR deflator.

    Complexity is ``O(len(thresholds) * n_markets)``: one position per market
    per threshold, computed once and reused by every fold.

    Args:
        panels: Tradeable markets, as `build_panels` returns them.
        thresholds: The pre-declared grid. Disclosed in full in the output.
        cost_model: Spread and fee model; None uses the Kalshi defaults.
        n_splits: Walk-forward folds.
        embargo_s: Purge embargo in seconds.
        n_markets: Markets seen before exclusions, for the report; defaults to
            ``len(panels)``.
        excluded: Markets dropped by `build_panels`, carried into the summary.
        provenance: Replay sidecar payload to carry into the summary.

    Returns:
        The populated `BacktestSummary`. With no tradeable market, or no fold
        that could both select and trade, the summary is well-formed and the
        deflated Sharpe is NaN — a strategy that took no trade has no Sharpe,
        and that is the honest report rather than an error.

    Raises:
        ValueError: If ``thresholds`` is empty — there is no strategy family to
            evaluate, and an empty grid would make the trial count zero.
    """
    thresholds = tuple(float(t) for t in thresholds)
    if not thresholds:
        raise ValueError("thresholds must declare at least one entry")
    cost_model = cost_model if cost_model is not None else CostModel()

    # One pass over (threshold, market): the entry rule is deterministic, so a
    # market's position at a threshold is the same object in every fold that
    # sees it, whether as training or as test.
    by_threshold: dict[float, list[Position | None]] = {
        tau: [open_position(panel, tau, cost_model) for panel in panels]
        for tau in thresholds
    }

    splits = purged_walk_forward(
        np.asarray([panel.label_start for panel in panels], dtype=float),
        np.asarray([panel.close_ts for panel in panels], dtype=float),
        n_splits=n_splits,
        embargo_s=embargo_s,
    )

    folds: list[FoldResult] = []
    trial_positions: dict[float, list[Position]] = {tau: [] for tau in thresholds}
    for split in splits:
        train = {t: _take(by_threshold[t], split.train) for t in thresholds}
        test = {t: _take(by_threshold[t], split.test) for t in thresholds}
        # Every threshold's *test* positions feed the trial record, including
        # folds where no selection was possible: the trials must see exactly the
        # data the selected strategy could have seen, or the deflator is
        # measured against a different experiment.
        for tau in thresholds:
            trial_positions[tau].extend(test[tau])
        tau_star, train_sharpe = _select_threshold(train, thresholds)
        folds.append(
            FoldResult(
                fold=split.fold,
                n_train=int(split.train.size),
                n_test=int(split.test.size),
                n_purged=split.n_purged,
                test_start=split.test_start,
                test_end=split.test_end,
                threshold=tau_star,
                train_sharpe=train_sharpe,
                positions=[] if tau_star is None else test[tau_star],
            ),
        )

    trials = [_trial_result(tau, trial_positions[tau]) for tau in thresholds]
    oos = np.concatenate([f.returns for f in folds]) if folds else np.zeros(0)
    counts: dict[str, int] = {}
    for row in excluded:
        counts[row.reason] = counts.get(row.reason, 0) + 1
    return BacktestSummary(
        thresholds=thresholds,
        cost_model=cost_model,
        n_splits=n_splits,
        embargo_s=float(embargo_s),
        n_markets=int(n_markets) if n_markets is not None else len(panels),
        panels=len(panels),
        excluded=list(excluded),
        exclusion_counts=counts,
        folds=folds,
        trials=trials,
        deflated=deflated_sharpe_ratio(
            oos,
            [row.sharpe for row in trials],
            n_trials=len(thresholds),
        ),
        provenance=dict(provenance or {}),
    )


def _take(positions: Sequence[Position | None], index: np.ndarray) -> list[Position]:
    """Positions at ``index``, dropping markets where no entry triggered."""
    return [positions[int(i)] for i in index if positions[int(i)] is not None]


def write_summary(
    summary: BacktestSummary,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``summary`` to ``path`` as indented JSON, creating parent dirs.

    Args:
        summary: Summary produced by `run_backtest`.
        path: Destination file.
        extra: Provenance the analysis cannot know (input paths, figure paths);
            merged into the top level of the payload.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict()
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------- Figures ----------------


def figure_backtest(summary: BacktestSummary):
    """Build the two-panel PoC figure, framed as detection evaluation.

    Left: each declared threshold's out-of-sample Sharpe, with the DSR benchmark
    drawn across it — the visual form of "how much of this is selection". Right:
    cumulative net PnL of the walk-forward positions in entry order.

    Args:
        summary: Summary produced by `run_backtest`.

    Returns:
        The matplotlib ``Figure``; the caller closes it.
    """
    # Deferred: `plots` transitively imports the whole inference stack for its
    # PG/iPMCMC panels, which an analysis-only run has no use for.
    import matplotlib.pyplot as plt

    from src.analysis.plots import set_paper_style

    set_paper_style()
    fig, (ax_trials, ax_pnl) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    taus = np.asarray([row.threshold for row in summary.trials], dtype=float)
    sharpes = np.asarray([row.sharpe for row in summary.trials], dtype=float)
    ax_trials.plot(taus, sharpes, marker="o", color="C0", label="trial Sharpe")
    ax_trials.axhline(
        summary.deflated.sr_benchmark,
        color="C3",
        ls="--",
        label=f"DSR benchmark SR0 ({summary.deflated.n_trials} trials)",
    )
    ax_trials.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax_trials.set_xlabel("P(Z) entry threshold")
    ax_trials.set_ylabel("out-of-sample Sharpe (per trade)")
    ax_trials.set_title("Declared grid, costed (PoC)")
    ax_trials.legend(loc="best", fontsize="small")

    positions = [p for fold in summary.folds for p in fold.positions]
    positions.sort(key=lambda p: p.entry_ts)
    if positions:
        ax_pnl.step(
            np.arange(1, len(positions) + 1),
            np.cumsum([p.ret for p in positions]),
            where="post",
            color="C0",
        )
    ax_pnl.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax_pnl.set_xlabel("walk-forward position (entry order)")
    ax_pnl.set_ylabel("cumulative net PnL ($ / contract)")
    ax_pnl.set_title("Out-of-sample, after spread + fees")

    fig.suptitle(
        "Detection-signal tradeability PoC - NOT a validated alpha strategy",
        fontsize="medium",
    )
    fig.tight_layout()
    return fig


def save_figures(summary: BacktestSummary, *, directory: str | Path) -> list[str]:
    """Render and save the backtest figure under ``directory``.

    Args:
        summary: Summary produced by `run_backtest`.
        directory: Destination, typically ``results/figures/backtest``.

    Returns:
        The paths written, as strings for the summary JSON.
    """
    import matplotlib.pyplot as plt

    from src.analysis.plots import save_paper_figure

    fig = figure_backtest(summary)
    paths = save_paper_figure(fig, "backtest", directory=directory)
    plt.close(fig)
    return [str(p) for p in paths]
