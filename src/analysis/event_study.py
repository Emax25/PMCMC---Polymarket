"""No-lookahead event study: does elevated ``P(Z)`` *precede* the terminal move?

Discrimination AUC on synthetic data says the score separates insider trades
from ordinary ones when both are already in hand. It says nothing about whether
the score is *temporally* informative on a real market — whether a reader
watching the stream would have seen ``P(Z)`` rise before the price did. That is
the question this module answers, and it is the one the paper's "streaming,
per-trade" claim rests on.

**Pre-registered primary statistic (plan 2026-07-23-005 R6/KTD4 — committed, do
not redesign).** For one market with close time ``t_close``:

    elevation = mean{ p_z(t) : t in [t_close - W, t_close - w) }
                - mean{ p_z(t) : t <= t_close }

tested against a **within-market time-shifted-window permutation null**: the
same window duration slid back by an offset ``d`` drawn uniformly from the
market's own earlier history, recomputed ``n_permutations`` times. The p-value
is the usual add-one permutation p-value.

The shift null is primary because it conditions on *this* market's ``P(Z)``
baseline and on its own trade intensity. A cross-market shuffle does neither:
markets differ by orders of magnitude in length and in baseline score, so an
extreme statistic pooled across them inflates false positives in exactly the
length-dependent way this design avoids (methods-critic, R6). The baseline term
above is constant across shifts and therefore cancels inside the p-value; it is
kept in the reported ``elevation`` only because a bare window mean is not
readable next to a market whose whole score level is high.

**Robustness variants are labelled, never confirmation.** ``max``-elevation
(same null, max instead of mean) and the cross-market shuffle are computed and
reported side by side so a reader can see whether the primary verdict is fragile
— but a robustness variant agreeing with the primary is not a second piece of
evidence, and this module never counts it as one.

**Why the embargo ``w``.** The window ends ``w`` seconds *before* close, and the
terminal move is measured over the disjoint interval ``[t_close - w, t_close]``
that follows it. Without that gap "elevated ``P(Z)`` precedes the move" would be
a statement about trades happening *during* the move, which any reactive score
would satisfy.

**No lookahead** is inherited, not re-derived. Every ``p_z`` consumed here comes
from `scripts/score_stream.py --replay`, whose scoring loop is a function of
trades ``0..t`` alone (see `src.inference.stream_scoring`). `read_replay_
provenance` refuses any score file whose ``<scores>.meta.json`` sidecar does not
say ``mode == "replay"``, so a live-mode or provenance-less file cannot enter the
study by accident.

``W`` and ``w`` are **locked by synthetic calibration before any real-data run**
(KTD4). See ``LOCKED_WINDOW_S`` / ``LOCKED_EMBARGO_S`` for the numbers and the
evidence that fixed them, and ``calibrate_window`` for the harness that
reproduces it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import chi2

log = logging.getLogger(__name__)

EVENT_STUDY_SCHEMA_VERSION = 1

DAY_SECONDS = 86400.0

# Integer tags that keep the independent RNG streams apart inside one seed.
# `np.random.default_rng` spawns from a list of ints, so every stream is
# reproducible from `seed` alone and adding a market cannot perturb another's
# draws. They are ints, not names, because SeedSequence takes no strings.
_STREAM_PRIMARY = 0
_STREAM_CROSS = 1
_STREAM_SIMULATE = 2
_STREAM_CALIBRATE = 3

# `score_stream.py` writes `<output>.meta.json` with a "mode" field that is
# either "replay" or "live". Both names are read verbatim from
# `scripts.score_stream._write_run_meta`; changing either there breaks the gate
# below, which `tests/test_event_study.py` pins.
PROVENANCE_SUFFIX = ".meta.json"
PROVENANCE_MODE_KEY = "mode"
REPLAY_MODE = "replay"

# ---- Locked window (KTD4) ----
#
# W = 5 days, w = 1 day. Fixed by the synthetic calibration in
# `calibrate_window` *before* any real-data run, and not to be re-tuned
# afterwards: re-picking W once real p-values are visible turns a pre-registered
# test into a search over windows.
#
# Evidence — `calibrate_window`, 60 replicates per arm, 999 permutations,
# alpha = 0.05, seed 2026, ~29-day markets carrying a planted burst ~4.2 days
# long. Reproduce with:
#
#     python -m scripts.event_study --calibrate --n-replicates 60 \
#         --n-permutations 999 --seed 2026
#
#     W (d)   detection   seam artefact   realized size
#       2       0.617         0.050           0.067
#       3       0.750         0.067           0.033
#       4       1.000         0.117           0.050
#       5       1.000         0.117           0.033   <- locked
#       7       1.000         0.133           0.050
#      10       1.000         0.067           0.117
#
# Detection saturates at W >= 4 d, once the window is at least as long as the
# burst; below that the window averages too few insider trades to clear the
# null's tail. Size stays nominal out to 7 d and is highest at 10 d, where the
# window is a third of the market's history and the time-shift null has almost
# no room left to place a comparison window. 5 d is chosen in the *interior* of
# the [4 d, 7 d] plateau rather than at an edge, so a real burst somewhat
# shorter or longer than the synthetic one still lands inside it. At the locked
# setting the null arm rejected 2/60 (binomial p = 0.77 against the nominal
# 0.05) and its p-values are indistinguishable from uniform (KS p = 0.16).
#
# Two limits on how hard this table can be read, both of which cut toward
# keeping the choice conservative rather than overturning it:
#
#   1. 60 replicates per arm gives SE ~ 0.028 on a size estimate, so 0.033,
#      0.050 and 0.117 are not separated by this sweep. The 10 d row is
#      nominally p = 0.030 one-sided, but across the six windows swept that is
#      ~0.18 Bonferroni-adjusted — suggestive, not established. Since detection
#      is 1.000 for every W in {4, 5, 7, 10}, picking 5 d is really a judgement
#      about plausible burst duration (4.2 d synthetic), not a calibrated
#      optimum, and the plateau language should be read that way.
#   2. Power here is an *oracle-regime* number. `_calibration_params` uses
#      beta_S = 0.6, beta_Z = 1.0 and tau2_1/tau2_0 = 0.1 with an oracle warm
#      start — a Z channel far stronger than any fit yet obtained on real data
#      (the 2026-08-02 warm start came back with beta_S = beta_Z = 0). Real-data
#      power at W = 5 d is therefore unmeasured, and "detection 1.000" must not
#      be quoted as a property of the detector on real markets.
LOCKED_WINDOW_S = 5.0 * DAY_SECONDS
LOCKED_EMBARGO_S = 1.0 * DAY_SECONDS

DEFAULT_N_PERMUTATIONS = 999
DEFAULT_ALPHA = 0.05

# A shifted window that catches a single trade has a very noisy mean, which
# fattens the null's tails and makes the p-value conservative rather than
# anti-conservative. Empty shifted windows are dropped instead (they carry no
# statistic at all) and counted, so a market whose null rests on a handful of
# usable placements is visible rather than silently weak.
_MIN_NULL_DRAWS = 30

_ROBUSTNESS_NOTE = (
    "max-elevation and the cross-market shuffle are labelled robustness checks, "
    "not independent confirmation: they reuse the same scores and the same "
    "window as the primary test. Quote `p_value` (mean elevation, within-market "
    "time-shift null) as the result; quote the variants only to say whether the "
    "primary verdict is fragile. The cross-market shuffle in particular ignores "
    "per-market baseline and length, which is why it is not the primary scheme."
)

_FISHER_NOTE = (
    "Fisher's method combines the per-market permutation p-values assuming they "
    "are independent and continuous. They are independent across markets but "
    "discrete, with atoms of size 1 / (n_permutations + 1), so the combined "
    "p-value is approximate and slightly anti-conservative at small "
    "n_permutations. It summarises the study; it does not replace the "
    "per-market column."
)

_NO_LOOKAHEAD_NOTE = (
    "Every score consumed here was produced by `score_stream.py --replay`, "
    "whose per-trade state is a function of trades 0..t only. The no-lookahead "
    "property is inherited from that replay guarantee, not re-established here; "
    "score files without `mode == \"replay\"` provenance are refused."
)


# ---------------- Errors ----------------


class ProvenanceError(RuntimeError):
    """Raised when a score file cannot be shown to have come from replay mode."""


# ---------------- Inputs ----------------


@dataclass(frozen=True)
class WindowSpec:
    """The pre-registered event window, in seconds before market close.

    Attributes:
        W: Window length. Trades in ``[t_close - W, t_close - w)`` carry the
            statistic.
        w: Embargo. Trades in ``[t_close - w, t_close]`` are excluded from the
            statistic and instead define the terminal move the statistic is
            claimed to precede.
    """

    W: float = LOCKED_WINDOW_S
    w: float = LOCKED_EMBARGO_S

    def __post_init__(self) -> None:
        """Reject a window that cannot hold any trade.

        Raises:
            ValueError: If ``W`` is not strictly greater than ``w >= 0``.
        """
        if self.w < 0.0:
            raise ValueError(f"embargo w must be non-negative, got {self.w!r}")
        if self.W <= self.w:
            raise ValueError(
                f"window W={self.W!r} must exceed the embargo w={self.w!r}; "
                "otherwise [t_close - W, t_close - w) is empty",
            )

    @property
    def is_locked(self) -> bool:
        """Whether this is exactly the KTD4-locked ``(W, w)`` pair."""
        return self.W == LOCKED_WINDOW_S and self.w == LOCKED_EMBARGO_S

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view, in seconds and in days."""
        return {
            "W_seconds": self.W,
            "w_seconds": self.w,
            "W_days": self.W / DAY_SECONDS,
            "w_days": self.w / DAY_SECONDS,
            "locked": self.is_locked,
        }


@dataclass(frozen=True)
class MarketScores:
    """One market's replayed per-trade scores, in non-decreasing time order.

    Attributes:
        market: ``condition_id`` (Polymarket) or ``ticker`` (Kalshi), as the
            score records carry it.
        ts: (n,) trade timestamps in unix seconds, sorted ascending.
        p_z: (n,) ``q(Z_t = 1)`` per trade.
        x_mean: (n,) filtered ``E[X_t | Y_{0:t}]`` on the logit-price scale;
            used only for the terminal move.
    """

    market: str
    ts: np.ndarray
    p_z: np.ndarray
    x_mean: np.ndarray

    @property
    def n(self) -> int:
        """Number of scored trades."""
        return int(self.ts.size)


def read_replay_provenance(scores_path: str | Path) -> dict[str, Any]:
    """Load and check the ``<scores>.meta.json`` sidecar written by score_stream.

    This is the gate that keeps the study honest. A scores JSONL is just numbers;
    only the sidecar records whether they came from ``--replay`` (state is a
    function of trades ``0..t``, so the no-lookahead property holds by
    construction) or from ``--live``. Anything else is refused rather than
    analysed with a caveat: a lookahead-contaminated event study looks exactly
    like a successful one.

    Args:
        scores_path: Path of the scores JSONL. The sidecar is that path with
            ``.meta.json`` appended, matching `scripts.score_stream`.

    Returns:
        The decoded sidecar payload.

    Raises:
        ProvenanceError: If the sidecar is missing, unreadable, or does not
            record ``mode == "replay"``.
    """
    scores_path = Path(scores_path)
    sidecar = scores_path.with_name(scores_path.name + PROVENANCE_SUFFIX)
    if not sidecar.is_file():
        raise ProvenanceError(
            f"{scores_path} has no {sidecar.name} provenance sidecar, so it "
            "cannot be shown to be free of lookahead. Re-score the capture with "
            "`python -m scripts.score_stream --replay ...`, which writes one.",
        )
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"{sidecar} is not readable JSON ({exc})") from exc
    mode = payload.get(PROVENANCE_MODE_KEY)
    if mode != REPLAY_MODE:
        raise ProvenanceError(
            f"{sidecar} records {PROVENANCE_MODE_KEY}={mode!r}, not "
            f"{REPLAY_MODE!r}. The event study accepts replay-mode scores only: "
            "only a replay is a pure function of the capture it read, so only a "
            "replay guarantees no lookahead.",
        )
    return payload


def load_scores(path: str | Path) -> dict[str, MarketScores]:
    """Read a `score_stream.py` scores JSONL into per-market sorted arrays.

    Records are grouped by ``market`` and sorted by ``ts``. Replay output is
    already sorted, but a stable sort here costs nothing and makes the window
    slicing below correct for any input.

    Args:
        path: Scores JSONL, one ``ScoredTrade`` object per line.

    Returns:
        ``{market: MarketScores}``. Empty when the file holds no usable record.

    Raises:
        ValueError: If a non-blank line is not valid JSON. A silently skipped
            score would shorten a window without saying so.
    """
    path = Path(path)
    rows: dict[str, list[tuple[float, float, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON ({exc})") from exc
            market = str(record.get("market", ""))
            rows.setdefault(market, []).append(
                (
                    float(record["ts"]),
                    float(record["p_z"]),
                    float(record.get("x_mean", 0.0)),
                ),
            )

    out: dict[str, MarketScores] = {}
    for market, triples in rows.items():
        arr = np.asarray(triples, dtype=float)
        order = np.argsort(arr[:, 0], kind="stable")
        arr = arr[order]
        out[market] = MarketScores(
            market=market,
            ts=arr[:, 0].copy(),
            p_z=arr[:, 1].copy(),
            x_mean=arr[:, 2].copy(),
        )
    return out


def _close_ts_from_record(record: Mapping[str, Any]) -> float | None:
    """Extract a market close time in unix seconds from one metadata record.

    Accepts a numeric ``close_ts`` directly, or the string date fields the two
    pull scripts actually write: Kalshi's ``close_time`` (RFC-3339) and
    Polymarket's ``end_date`` (Gamma ISO or date-only).

    Args:
        record: One market's metadata mapping.

    Returns:
        Unix seconds, or None when no field carries a parseable close time.
    """
    numeric = record.get("close_ts")
    if numeric is not None:
        return float(numeric)
    # `_resolution_ts_from_end_date` is the repo's one parser for "when did this
    # market close", already used by the pre-resolution filter. Reaching for the
    # module-private name is deliberate: a second parser here could disagree
    # with the filter about a market's close time, and then the event window and
    # the trades it slices would be measured from different clocks.
    from src.data.preprocess import _resolution_ts_from_end_date

    for key in ("close_time", "end_date", "endDate"):
        value = record.get(key)
        if value:
            parsed = _resolution_ts_from_end_date(str(value))
            if parsed is not None:
                return parsed
    return None


def _market_id_from_record(record: Mapping[str, Any], fallback: str) -> str:
    """Pick a market's id from a metadata record, defaulting to ``fallback``."""
    for key in ("market", "condition_id", "ticker", "conditionId", "slug"):
        value = record.get(key)
        if value:
            return str(value)
    return fallback


def load_resolutions(path: str | Path) -> dict[str, float]:
    """Load ``{market: t_close}`` from a metadata file or a sidecar directory.

    Three shapes are accepted, so the study can be pointed at whatever the pull
    step left behind:

      * a directory — every ``*.meta.json`` in it, as `scripts.pull_kalshi` and
        `preprocess.save_processed` write them, keyed by the record's own id
        field or by the file stem;
      * a JSON object — ``{market: t_close}`` or ``{market: {...}}``;
      * a JSON array — one metadata record per element.

    Markets whose record carries no parseable close time are dropped here and
    reported as missing by `run_event_study`, which is the only place that knows
    whether they had scores in the first place.

    Args:
        path: Directory of ``*.meta.json`` sidecars, or a JSON file.

    Returns:
        ``{market: close timestamp in unix seconds}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a JSON file is malformed or is neither object nor array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"resolution metadata not found: {path}")

    records: list[tuple[str, Mapping[str, Any] | float]] = []
    if path.is_dir():
        for sidecar in sorted(path.glob(f"*{PROVENANCE_SUFFIX}")):
            stem = sidecar.name[: -len(PROVENANCE_SUFFIX)]
            # Same failure mode as the single-file branch below, and it has to
            # raise the same way: a directory of sidecars is the normal input,
            # so one unreadable file must name itself rather than surface as a
            # bare JSONDecodeError with no path in it.
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{sidecar}: malformed JSON ({exc})") from exc
            records.append((stem, payload))
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: malformed JSON ({exc})") from exc
        if isinstance(payload, dict):
            records.extend(payload.items())
        elif isinstance(payload, list):
            records.extend((f"row{i}", row) for i, row in enumerate(payload))
        else:
            raise ValueError(
                f"{path}: expected a JSON object or array of metadata records, "
                f"got {type(payload).__name__}",
            )

    out: dict[str, float] = {}
    for key, value in records:
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
            continue
        if not isinstance(value, Mapping):
            log.warning("resolution record for %s is not a mapping; skipped", key)
            continue
        close_ts = _close_ts_from_record(value)
        if close_ts is None:
            log.warning("resolution record for %s carries no close time", key)
            continue
        out[_market_id_from_record(value, str(key))] = close_ts
    return out


# ---------------- The statistic ----------------


def _window_slice(ts: np.ndarray, lo: float, hi: float) -> slice:
    """Half-open ``[lo, hi)`` index slice into a sorted timestamp array."""
    return slice(
        int(np.searchsorted(ts, lo, side="left")),
        int(np.searchsorted(ts, hi, side="left")),
    )


def _window_stats(
    ts: np.ndarray,
    p_z: np.ndarray,
    lo: float,
    hi: float,
) -> tuple[int, float, float]:
    """Trade count, mean and max of ``p_z`` over the half-open window ``[lo, hi)``.

    Args:
        ts: Sorted trade timestamps.
        p_z: Per-trade scores aligned with ``ts``.
        lo: Window start, inclusive.
        hi: Window end, exclusive.

    Returns:
        ``(n, mean, max)``; ``(0, nan, nan)`` when the window holds no trade.
    """
    sl = _window_slice(ts, lo, hi)
    chunk = p_z[sl]
    if chunk.size == 0:
        return 0, math.nan, math.nan
    return int(chunk.size), float(chunk.mean()), float(chunk.max())


def _add_one_p_value(observed: float, null: np.ndarray) -> float:
    """Add-one permutation p-value ``(1 + #{null >= observed}) / (1 + n_null)``.

    The add-one form is used rather than the raw exceedance rate because the
    latter can report exactly 0, which is never a defensible claim from a finite
    number of permutations.

    Args:
        observed: Statistic on the real window.
        null: Statistics from the permuted/shifted windows.

    Returns:
        A p-value in ``(0, 1]``.
    """
    n = int(null.size)
    return float(1 + int(np.sum(null >= observed))) / float(1 + n)


def _shifted_null(
    ts: np.ndarray,
    p_z: np.ndarray,
    *,
    lo: float,
    hi: float,
    max_offset: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Draw the within-market time-shifted-window null for mean and max.

    Offsets are drawn uniformly on ``[0, max_offset)`` and slide the *whole*
    window back, so every null placement has the same duration as the observed
    one and sits entirely inside the market's own history. Placements that catch
    no trade carry no statistic and are dropped rather than scored as zero —
    a zero would be a fabricated observation of "no insider activity" at a time
    when nothing was observed at all.

    Args:
        ts: Sorted trade timestamps.
        p_z: Per-trade scores aligned with ``ts``.
        lo: Observed window start.
        hi: Observed window end (exclusive).
        max_offset: Largest admissible backward shift, so ``lo - offset`` stays
            at or after the first trade.
        n_permutations: Offsets to draw.
        rng: Source of randomness; passed explicitly so runs are reproducible.

    Returns:
        ``(null_means, null_maxes, n_empty)`` — the first two of equal length
        ``n_permutations - n_empty``.
    """
    offsets = rng.uniform(0.0, max_offset, size=n_permutations)
    means: list[float] = []
    maxes: list[float] = []
    n_empty = 0
    for offset in offsets:
        n_win, mean, peak = _window_stats(ts, p_z, lo - offset, hi - offset)
        if n_win == 0:
            n_empty += 1
            continue
        means.append(mean)
        maxes.append(peak)
    return np.asarray(means, dtype=float), np.asarray(maxes, dtype=float), n_empty


# ---------------- Per-market results ----------------


@dataclass(frozen=True)
class ExcludedMarket:
    """One market that carries scores but no usable statistic.

    Attributes:
        market: Market id.
        reason: Why it was dropped, in words a report can print verbatim.
    """

    market: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {"market": self.market, "reason": self.reason}


# Exclusion reasons, named so the CLI, the summary JSON and the tests all agree.
REASON_NO_RESOLUTION = "no resolution metadata"
REASON_NO_TRADES_BEFORE_CLOSE = "no scored trades at or before t_close"
REASON_EMPTY_WINDOW = "no scored trades inside the event window"
REASON_HISTORY_TOO_SHORT = "history shorter than the event window"
REASON_TOO_FEW_NULL = "too few usable time-shifted windows for a permutation null"


@dataclass(frozen=True)
class MarketResult:
    """The event-study verdict for one market.

    Attributes:
        market: Market id.
        close_ts: Market close time, unix seconds.
        n_trades: Scored trades at or before ``close_ts``.
        n_window: Scored trades inside ``[t_close - W, t_close - w)``.
        window_mean: Mean ``p_z`` inside the window.
        baseline_mean: Mean ``p_z`` over the whole market.
        elevation: **Primary statistic** — ``window_mean - baseline_mean``.
        p_value: **Primary p-value** — within-market time-shift permutation.
        n_null: Usable shifted placements behind ``p_value``.
        n_null_empty: Drawn placements that caught no trade and were dropped.
        null_mean: Mean of the null elevations, for reporting scale.
        null_sd: Standard deviation of the null elevations.
        z_score: ``(elevation - null_mean) / null_sd``; a readable effect size,
            not a test — the p-value is the test.
        window_max: Max ``p_z`` inside the window.
        max_elevation: *Robustness* — ``window_max - baseline_mean``.
        p_value_max: *Robustness* — same shift null, max statistic.
        p_value_cross: *Robustness* — cross-market shuffle null; None until
            `run_event_study` fills it, and None when no other market has
            scores to shuffle against.
        terminal_move: Change in the filtered logit price over the embargo
            interval ``[t_close - w, t_close]`` — the move the elevation is
            claimed to precede. 0.0 when that interval holds no trade.
        n_terminal: Trades in the embargo interval.
    """

    market: str
    close_ts: float
    n_trades: int
    n_window: int
    window_mean: float
    baseline_mean: float
    elevation: float
    p_value: float
    n_null: int
    n_null_empty: int
    null_mean: float
    null_sd: float
    z_score: float
    window_max: float
    max_elevation: float
    p_value_max: float
    p_value_cross: float | None
    terminal_move: float
    n_terminal: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view; the primary fields lead, variants are suffixed."""
        return {
            "market": self.market,
            "close_ts": self.close_ts,
            "n_trades": self.n_trades,
            "n_window": self.n_window,
            "window_mean": self.window_mean,
            "baseline_mean": self.baseline_mean,
            "elevation": self.elevation,
            "p_value": self.p_value,
            "n_null": self.n_null,
            "n_null_empty": self.n_null_empty,
            "null_mean": self.null_mean,
            "null_sd": self.null_sd,
            "z_score": self.z_score,
            "terminal_move": self.terminal_move,
            "n_terminal": self.n_terminal,
            "robustness": {
                "window_max": self.window_max,
                "max_elevation": self.max_elevation,
                "p_value_max": self.p_value_max,
                "p_value_cross_market": self.p_value_cross,
            },
        }


def _terminal_move(
    ts: np.ndarray,
    x_mean: np.ndarray,
    *,
    lo: float,
    hi: float,
) -> tuple[float, int]:
    """Filtered logit-price change over the embargo interval ``[lo, hi]``.

    Measured on ``x_mean`` (the filter's ``E[X_t | Y_{0:t}]``) rather than on
    raw trade prices: a single wide-spread print at the close would otherwise
    read as a move the market never made.

    Args:
        ts: Sorted trade timestamps.
        x_mean: Filtered logit prices aligned with ``ts``.
        lo: Embargo start, ``t_close - w``.
        hi: Market close, inclusive.

    Returns:
        ``(move, n_trades)``; ``(0.0, 0)`` when the interval holds no trade,
        because a move nobody traded through is not a move.
    """
    sl = _window_slice(ts, lo, np.nextafter(hi, math.inf))
    chunk = x_mean[sl]
    if chunk.size == 0:
        return 0.0, 0
    # Anchored at the last pre-embargo observation when there is one, so the
    # move spans the whole embargo rather than only its interior.
    start = x_mean[sl.start - 1] if sl.start > 0 else chunk[0]
    return float(chunk[-1] - start), int(chunk.size)


def analyze_market(
    scores: MarketScores,
    close_ts: float,
    *,
    window: WindowSpec,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    rng: np.random.Generator,
) -> MarketResult | ExcludedMarket:
    """Run the primary test and the shift-null robustness variant on one market.

    Args:
        scores: The market's replayed scores, sorted by time.
        close_ts: Market close time in unix seconds.
        window: The pre-registered ``(W, w)``.
        n_permutations: Time-shifted placements to draw for the null.
        rng: Source of randomness; passed explicitly so runs are reproducible.

    Returns:
        A `MarketResult`, or an `ExcludedMarket` naming why no statistic exists.
        The cross-market variant is left as None for `run_event_study` to fill.
    """
    keep = scores.ts <= close_ts
    ts, p_z, x_mean = scores.ts[keep], scores.p_z[keep], scores.x_mean[keep]
    if ts.size == 0:
        return ExcludedMarket(scores.market, REASON_NO_TRADES_BEFORE_CLOSE)

    lo, hi = close_ts - window.W, close_ts - window.w
    n_window, window_mean, window_max = _window_stats(ts, p_z, lo, hi)
    if n_window == 0:
        return ExcludedMarket(scores.market, REASON_EMPTY_WINDOW)

    # A shifted window has to fit between the first trade and the observed
    # window's own start; without that room there is no null to compare against.
    max_offset = lo - float(ts[0])
    if max_offset <= 0.0:
        return ExcludedMarket(scores.market, REASON_HISTORY_TOO_SHORT)

    null_means, null_maxes, n_empty = _shifted_null(
        ts,
        p_z,
        lo=lo,
        hi=hi,
        max_offset=max_offset,
        n_permutations=n_permutations,
        rng=rng,
    )
    if null_means.size < min(_MIN_NULL_DRAWS, n_permutations):
        return ExcludedMarket(scores.market, REASON_TOO_FEW_NULL)

    baseline_mean = float(p_z.mean())
    null_sd = float(null_means.std(ddof=1)) if null_means.size > 1 else 0.0
    move, n_terminal = _terminal_move(ts, x_mean, lo=hi, hi=close_ts)
    return MarketResult(
        market=scores.market,
        close_ts=float(close_ts),
        n_trades=int(ts.size),
        n_window=n_window,
        window_mean=window_mean,
        baseline_mean=baseline_mean,
        elevation=window_mean - baseline_mean,
        # The baseline is the same constant in the observed statistic and in
        # every shifted one, so comparing window means is identical to comparing
        # elevations and saves subtracting it n_permutations times.
        p_value=_add_one_p_value(window_mean, null_means),
        n_null=int(null_means.size),
        n_null_empty=n_empty,
        null_mean=float(null_means.mean()) - baseline_mean,
        null_sd=null_sd,
        z_score=(
            (window_mean - float(null_means.mean())) / null_sd
            if null_sd > 0.0
            else math.nan
        ),
        window_max=window_max,
        max_elevation=window_max - baseline_mean,
        p_value_max=_add_one_p_value(window_max, null_maxes),
        p_value_cross=None,
        terminal_move=move,
        n_terminal=n_terminal,
    )


def _market_stream(seed: int, stream: int, market: str) -> np.random.Generator:
    """Build one market's RNG from its *id*, not its position in the run.

    Keying on a digest of the market id is what makes the reproducibility
    promise true: adding or dropping a market leaves every other market's
    permutation draws untouched, so two runs over overlapping market sets agree
    wherever they overlap. An enumerate index would silently reshuffle every
    market after the inserted one. `hashlib` rather than the builtin `hash`,
    whose string salt changes per process and would break replay across runs.

    Args:
        seed: Root seed for the whole study.
        stream: One of the ``_STREAM_*`` tags, keeping schemes independent.
        market: Market id.

    Returns:
        A generator reproducible from ``(seed, stream, market)`` alone.
    """
    digest = hashlib.blake2b(market.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng([seed, stream, int.from_bytes(digest, "big")])


def _cross_market_p_value(
    result: MarketResult,
    pooled: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> float | None:
    """Cross-market shuffle p-value for one market — a *robustness check only*.

    Draws ``n_window`` scores at random from the pooled scores of every *other*
    market and takes their mean, ``n_permutations`` times. This is the scheme the
    primary design rejects: it ignores each market's own baseline and length, so
    a long market with an extreme statistic is compared against a distribution
    that has nothing to do with it. Reported so a reader can see the difference,
    never quoted as confirmation.

    Args:
        result: The market's primary result, read for ``n_window``.
        pooled: Scores from all other markets.
        n_permutations: Draws to take.
        rng: Source of randomness.

    Returns:
        The p-value, or None when no other market contributed scores.
    """
    if pooled.size == 0:
        return None
    draws = rng.choice(pooled, size=(n_permutations, result.n_window), replace=True)
    return _add_one_p_value(result.window_mean, draws.mean(axis=1))


# ---------------- Study summary ----------------


@dataclass(frozen=True)
class EventStudySummary:
    """Everything one event-study run produces.

    Attributes:
        window: The ``(W, w)`` the run used.
        n_permutations: Permutation draws per market and per null scheme.
        seed: RNG seed, so the whole run replays exactly.
        alpha: Level the per-market significance count is taken at.
        results: Per-market results, in input order.
        excluded: Markets with scores but no usable statistic.
        exclusion_counts: ``{reason: count}`` over ``excluded``.
        n_markets: Markets seen in the scores file.
        fisher_stat: ``-2 sum log p`` over the primary p-values.
        fisher_p: Combined p-value from ``fisher_stat``, or None below two
            markets, where combining is meaningless.
        n_significant: Markets with primary ``p_value < alpha``.
        provenance: The replay sidecar of the scores file, carried through.
    """

    window: WindowSpec
    n_permutations: int
    seed: int
    alpha: float
    results: list[MarketResult]
    excluded: list[ExcludedMarket]
    exclusion_counts: dict[str, int]
    n_markets: int
    fisher_stat: float | None
    fisher_p: float | None
    n_significant: int
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the whole run."""
        return {
            "schema_version": EVENT_STUDY_SCHEMA_VERSION,
            "window": self.window.to_dict(),
            "n_permutations": self.n_permutations,
            "seed": self.seed,
            "alpha": self.alpha,
            "n_markets": self.n_markets,
            "n_analysed": len(self.results),
            "n_excluded": len(self.excluded),
            "exclusion_counts": dict(self.exclusion_counts),
            "excluded": [row.to_dict() for row in self.excluded],
            "markets": [row.to_dict() for row in self.results],
            "n_significant": self.n_significant,
            "fisher_stat": self.fisher_stat,
            "fisher_p": self.fisher_p,
            "provenance": self.provenance,
            "no_lookahead_note": _NO_LOOKAHEAD_NOTE,
            "robustness_note": _ROBUSTNESS_NOTE,
            "fisher_note": _FISHER_NOTE,
        }


def _fisher_combine(p_values: Sequence[float]) -> tuple[float | None, float | None]:
    """Fisher's method over independent per-market p-values.

    Args:
        p_values: Primary p-values, one per analysed market.

    Returns:
        ``(statistic, combined p)``, or ``(None, None)`` below two markets —
        combining a single market's p-value with nothing returns that p-value
        and dresses it up as a study.
    """
    if len(p_values) < 2:
        return None, None
    stat = float(-2.0 * np.sum(np.log(np.asarray(p_values, dtype=float))))
    return stat, float(chi2.sf(stat, 2 * len(p_values)))


def run_event_study(
    scores_by_market: Mapping[str, MarketScores],
    close_by_market: Mapping[str, float],
    *,
    window: WindowSpec | None = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
    provenance: dict[str, Any] | None = None,
) -> EventStudySummary:
    """Run the pre-registered event study over every market with scores.

    Markets are processed in sorted id order and each gets its own RNG stream
    derived from ``seed`` and its own id (`_market_stream`), so a run is
    reproducible and adding a market does not perturb the others' primary
    p-values. The cross-market robustness variant is the one exception, and
    unavoidably so: its null *is* the other markets' scores.

    Args:
        scores_by_market: Replayed scores, as `load_scores` returns them.
        close_by_market: ``{market: t_close}``, as `load_resolutions` returns.
            A market absent here is excluded and counted, never guessed at.
        window: The pre-registered ``(W, w)``; None uses the KTD4-locked pair.
        n_permutations: Permutation draws per market and per null scheme.
        seed: Root RNG seed.
        alpha: Level for the per-market significance count.
        provenance: Replay sidecar payload to carry into the summary.

    Returns:
        The populated `EventStudySummary`.

    Raises:
        ValueError: If ``n_permutations`` is not positive — a permutation test
            with no permutations has no null.
    """
    if n_permutations < 1:
        raise ValueError(
            f"n_permutations must be at least 1, got {n_permutations}; the "
            "permutation null is the test",
        )
    window = window if window is not None else WindowSpec()
    if not window.is_locked:
        log.warning(
            "running at W=%.2f d, w=%.2f d, which is NOT the KTD4-locked "
            "(%.2f d, %.2f d). Anything produced at a non-locked window is "
            "exploratory and must be labelled as such.",
            window.W / DAY_SECONDS,
            window.w / DAY_SECONDS,
            LOCKED_WINDOW_S / DAY_SECONDS,
            LOCKED_EMBARGO_S / DAY_SECONDS,
        )

    results: list[MarketResult] = []
    excluded: list[ExcludedMarket] = []
    for market in sorted(scores_by_market):
        close_ts = close_by_market.get(market)
        if close_ts is None:
            log.warning(
                "market %s has scores but no resolution metadata; excluded from "
                "the event study (no t_close means no event window)",
                market,
            )
            excluded.append(ExcludedMarket(market, REASON_NO_RESOLUTION))
            continue
        outcome = analyze_market(
            scores_by_market[market],
            float(close_ts),
            window=window,
            n_permutations=n_permutations,
            rng=_market_stream(seed, _STREAM_PRIMARY, market),
        )
        if isinstance(outcome, ExcludedMarket):
            log.warning("market %s excluded: %s", market, outcome.reason)
            excluded.append(outcome)
        else:
            results.append(outcome)

    # Second pass: the cross-market shuffle needs every other market's scores,
    # so it cannot be computed inside the per-market pass above.
    analysed = {row.market for row in results}
    results = [
        _with_cross_market(
            row,
            np.concatenate(
                [scores_by_market[m].p_z for m in sorted(analysed - {row.market})]
                or [np.zeros(0)],
            ),
            n_permutations=n_permutations,
            rng=_market_stream(seed, _STREAM_CROSS, row.market),
        )
        for row in results
    ]

    counts: dict[str, int] = {}
    for row in excluded:
        counts[row.reason] = counts.get(row.reason, 0) + 1
    fisher_stat, fisher_p = _fisher_combine([row.p_value for row in results])
    return EventStudySummary(
        window=window,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
        results=results,
        excluded=excluded,
        exclusion_counts=counts,
        n_markets=len(scores_by_market),
        fisher_stat=fisher_stat,
        fisher_p=fisher_p,
        n_significant=sum(1 for row in results if row.p_value < alpha),
        provenance=dict(provenance or {}),
    )


def _with_cross_market(
    result: MarketResult,
    pooled: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> MarketResult:
    """Return ``result`` with its cross-market robustness p-value filled in."""
    return replace(
        result,
        p_value_cross=_cross_market_p_value(
            result,
            pooled,
            n_permutations=n_permutations,
            rng=rng,
        ),
    )


def write_summary(
    summary: EventStudySummary,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``summary`` to ``path`` as indented JSON, creating parent dirs.

    Args:
        summary: Summary produced by `run_event_study`.
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


def figure_event_study(summary: EventStudySummary):
    """Build the two-panel event-study figure.

    Left: per-market elevation against the terminal move over the embargo
    interval — the descriptive picture behind the claim, not the test. Right:
    the per-market permutation p-values against the uniform distribution they
    would follow if ``P(Z)`` carried no timing information, which is the visual
    form of the primary result.

    Args:
        summary: Summary produced by `run_event_study`.

    Returns:
        The matplotlib ``Figure``; the caller closes it.
    """
    # Deferred: `plots` transitively imports the whole inference stack for its
    # PG/iPMCMC panels, which an analysis-only run has no use for.
    import matplotlib.pyplot as plt

    from src.analysis.plots import set_paper_style

    set_paper_style()
    fig, (ax_scatter, ax_null) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    elevation = np.asarray([row.elevation for row in summary.results], dtype=float)
    move = np.abs([row.terminal_move for row in summary.results])
    significant = np.asarray(
        [row.p_value < summary.alpha for row in summary.results],
        dtype=bool,
    )
    ax_scatter.scatter(
        elevation[~significant],
        move[~significant],
        facecolors="none",
        edgecolors="0.4",
        label=f"p >= {summary.alpha:g}",
    )
    ax_scatter.scatter(
        elevation[significant],
        move[significant],
        color="C3",
        label=f"p < {summary.alpha:g}",
    )
    ax_scatter.axvline(0.0, color="0.7", lw=0.8, zorder=0)
    ax_scatter.set_xlabel(
        f"mean P(Z) elevation, W={summary.window.W / DAY_SECONDS:.3g} d"
    )
    ax_scatter.set_ylabel("|terminal move| (logit price)")
    ax_scatter.set_title("Elevation vs terminal move")
    if summary.results:
        ax_scatter.legend(loc="best")

    p_values = np.sort([row.p_value for row in summary.results])
    if p_values.size:
        ax_null.step(
            p_values,
            np.arange(1, p_values.size + 1) / p_values.size,
            where="post",
            color="C0",
            label="observed",
        )
    ax_null.plot([0.0, 1.0], [0.0, 1.0], color="0.7", ls="--", label="uniform null")
    ax_null.set_xlim(0.0, 1.0)
    ax_null.set_ylim(0.0, 1.0)
    ax_null.set_xlabel("within-market time-shift permutation p-value")
    ax_null.set_ylabel("ECDF across markets")
    ax_null.set_title("Primary p-values vs the null")
    ax_null.legend(loc="best")

    fig.tight_layout()
    return fig


def save_figures(summary: EventStudySummary, *, directory: str | Path) -> list[str]:
    """Render and save the event-study figure under ``directory``.

    Args:
        summary: Summary produced by `run_event_study`.
        directory: Destination, typically ``results/figures/event_study``.

    Returns:
        The paths written, as strings for the summary JSON.
    """
    import matplotlib.pyplot as plt

    from src.analysis.plots import save_paper_figure

    fig = figure_event_study(summary)
    paths = save_paper_figure(fig, "event_study", directory=directory)
    plt.close(fig)
    return [str(p) for p in paths]


# ---------------- Synthetic calibration (KTD4) ----------------

# The calibration regime. A synthetic market spans ~30 days at ~30-minute mean
# spacing, which puts a day-scale window in the same relationship to the trade
# rate as it is on a real politics market — the point of calibrating the window
# in seconds rather than in trade counts, so the locked W transfers.
CALIBRATION_MEAN_GAP_S = 1800.0
CALIBRATION_N_EARLY = 1200
CALIBRATION_N_LATE = 200
# 400 wallets of which 80 are insiders in the planted segment. The insider
# *share of trades* (weighted 3x by the generator) is what drives detection and
# is held at ~43% by the 20% wallet share; the wallet count itself is high so
# that a segment's mean propensity concentrates around the Beta(a, b) mean. At
# 40 wallets it does not, and the seam control below picks up that population
# shift as a real signal — the artefact this size is chosen to suppress.
CALIBRATION_N_WALLETS = 400
CALIBRATION_N_INSIDERS = 80

# The three calibration arms. Splitting "unplanted" into two is what makes the
# calibration honest: a spliced market is not a draw from the model — its final
# segment has a freshly drawn wallet population — so its rejection rate measures
# the *splice*, not the test's size. The size claim comes from ARM_NULL, a
# single clean draw; ARM_SEAM says how much of ARM_PLANTED's detection could be
# the seam rather than the insiders.
ARM_PLANTED = "planted"  # spliced; the late segment carries insider wallets
ARM_SEAM = "seam"  # spliced; no insiders — the splice-artefact control
ARM_NULL = "null"  # one unspliced draw — the realized-size arm
CALIBRATION_ARMS = (ARM_PLANTED, ARM_SEAM, ARM_NULL)


def _calibration_params():
    """Model parameters for the synthetic calibration markets.

    Chosen to look like a traded politics market rather than to flatter the
    detector: a slow random walk in logit space, a rare high-volatility regime,
    and observation noise well above the state noise.

    Returns:
        The `ModelParams` every calibration market is drawn from.
    """
    from config.default_params import ModelParams

    return ModelParams(
        sigma2_0=2e-6,
        sigma2_1=4e-5,
        q_01=0.02,
        q_10=0.10,
        beta_S=0.6,
        beta_Z=1.0,
        tau2_0=0.05,
        tau2_1=0.005,
    )


def _market_records(market, market_id: str, prefix: str, *, t0: float, tx0: int):
    """Turn one `SyntheticMarket` into `stream_trades.py`-shaped raw records.

    Args:
        market: The simulated market.
        market_id: ``condition_id`` stamped on every record.
        prefix: Wallet-address namespace, so two spliced segments do not share
            wallet identities (their propensities are separate draws).
        t0: Seconds added to every timestamp.
        tx0: First transaction-hash counter value.

    Returns:
        A list of raw trade dicts in time order.
    """
    return [
        {
            "timestamp": float(market.t[i] + t0),
            # The generator can emit a price a hair outside (0, 1) after a large
            # logit excursion; `stream_scoring._is_scorable` would drop those,
            # silently shortening a window.
            "price": float(np.clip(market.p[i], 1e-6, 1.0 - 1e-6)),
            "size": float(market.S[i]),
            "wallet": f"{prefix}{int(market.wallet_ids[i]):04d}",
            "transaction_hash": f"0x{market_id}_{tx0 + i:08d}",
            "condition_id": market_id,
        }
        for i in range(int(market.t.size))
    ]


def _score_calibration_market(records, wallet_addresses, theta_w, params):
    """Score simulated raw trades with the streaming scorer, oracle-warm-started.

    The scorer is warm-started at the *true* parameters and propensities with
    adaptation frozen. Calibration is about the window ``W``, not about whether
    the online-EM block can learn the parameters — that is
    `tests/test_online_scorer.py`'s question, and letting it fail here would
    confound the two.

    Args:
        records: Raw trade dicts in time order.
        wallet_addresses: Every wallet address, in the order ``theta_w`` indexes
            them; pre-seeding the index is what makes the two line up.
        theta_w: True per-wallet propensities.
        params: True model parameters.

    Returns:
        ``(ts, p_z, x_mean)`` arrays, one entry per scored trade.
    """
    from config.default_params import OnlineScorerConfig
    from src.data.preprocess import WalletIndex
    from src.inference.stream_scoring import StreamScorer, WarmStart

    index = WalletIndex()
    for address in wallet_addresses:
        index.add(address)
    warm = WarmStart(
        params=params,
        theta_w=np.asarray(theta_w, dtype=float),
        # Identity centering: the true beta_S/beta_Z act on the raw covariates
        # in the generative model, which is exactly what (0, 1, 0) reproduces.
        m_S=0.0,
        s_S=1.0,
        m_Z=0.0,
    )
    scorer = StreamScorer(
        warm,
        config=OnlineScorerConfig(forgetting=1.0, n_refresh=None),
        wallet_index=index,
    )
    scored = list(scorer.score(records))
    return (
        np.asarray([s.ts for s in scored], dtype=float),
        np.asarray([s.p_z for s in scored], dtype=float),
        np.asarray([s.x_mean for s in scored], dtype=float),
    )


def simulate_market_scores(
    market_id: str,
    *,
    arm: str,
    rng: np.random.Generator,
    n_early: int = CALIBRATION_N_EARLY,
    n_late: int = CALIBRATION_N_LATE,
    mean_gap: float = CALIBRATION_MEAN_GAP_S,
) -> tuple[MarketScores, float]:
    """Simulate one calibration market and score it with the streaming scorer.

    `ARM_NULL` is a single `src.data.synthetic.generate_market` draw of
    ``n_early + n_late`` trades — an exact draw from the model, and therefore
    the only arm whose rejection rate is the test's realized size.

    `ARM_PLANTED` and `ARM_SEAM` splice two draws end to end: an unplanted early
    segment, then a late segment that carries insider wallets (planted) or does
    not (seam). Splicing is how the insider activity is made *late* — the
    generator plants insiders uniformly over a market's life, which is exactly
    the alternative this statistic is not designed to detect. The late segment's
    logit path is shifted to continue the early one's last value, so the seam is
    a continuation of the random walk rather than a jump.

    A splice is still not a model draw: the late segment's wallet population is
    a fresh Beta draw whose mean differs slightly from the early one's, which is
    a genuine late change in the market. `ARM_SEAM` exists to measure exactly
    that, so it is never mistaken for detected insider activity.

    Args:
        market_id: Id stamped on the records and carried onto `MarketScores`.
        arm: One of `CALIBRATION_ARMS`.
        rng: Source of randomness; passed explicitly so replicates are seeded.
        n_early: Trades in the early segment (all of them, for `ARM_NULL`).
        n_late: Trades in the late segment.
        mean_gap: Mean inter-trade gap in seconds, for every segment.

    Returns:
        ``(scores, close_ts)`` with ``close_ts`` the last trade's timestamp.

    Raises:
        ValueError: If ``arm`` is not one of `CALIBRATION_ARMS`.
    """
    if arm not in CALIBRATION_ARMS:
        raise ValueError(f"arm must be one of {CALIBRATION_ARMS}, got {arm!r}")
    # Deferred: the generator and the scorer pull the data/inference stack,
    # which the analysis half of this module never touches.
    from src.data.synthetic import generate_market
    from src.utils.transforms import sigmoid

    params = _calibration_params()
    common = {
        "n_wallets": CALIBRATION_N_WALLETS,
        "mean_inter_trade_time": mean_gap,
        "rng": rng,
    }

    if arm == ARM_NULL:
        market = generate_market(
            params, n_trades=n_early + n_late, n_insider_wallets=0, **common
        )
        prefix = f"{market_id}w"
        records = _market_records(market, market_id, prefix, t0=0.0, tx0=0)
        addresses = [f"{prefix}{w:04d}" for w in range(CALIBRATION_N_WALLETS)]
        theta_w = market.theta_w
    else:
        early = generate_market(params, n_trades=n_early, n_insider_wallets=0, **common)
        late = generate_market(
            params,
            n_trades=n_late,
            n_insider_wallets=CALIBRATION_N_INSIDERS if arm == ARM_PLANTED else 0,
            **common,
        )
        late.Y = late.Y + (early.Y[-1] - late.Y[0])
        late.p = sigmoid(late.Y)
        early_prefix, late_prefix = f"{market_id}e", f"{market_id}l"
        records = _market_records(early, market_id, early_prefix, t0=0.0, tx0=0)
        records += _market_records(
            late,
            market_id,
            late_prefix,
            t0=float(early.t[-1] + mean_gap),
            tx0=n_early,
        )
        # The warm start's theta_w is indexed by wallet id, so the index has to
        # be pre-seeded in the same order the two vectors are concatenated.
        addresses = [
            f"{prefix}{w:04d}"
            for prefix in (early_prefix, late_prefix)
            for w in range(CALIBRATION_N_WALLETS)
        ]
        theta_w = np.concatenate([early.theta_w, late.theta_w])

    ts, p_z, x_mean = _score_calibration_market(records, addresses, theta_w, params)
    scores = MarketScores(market=market_id, ts=ts, p_z=p_z, x_mean=x_mean)
    return scores, float(ts[-1])


def calibrate_window(
    windows: Iterable[WindowSpec],
    *,
    n_replicates: int = 20,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Measure detection rate, seam artefact and realized size per window (KTD4).

    Each replicate simulates one market per arm from a shared RNG stream and
    runs the primary test on all three. Read the output as:

      * ``detection_rate`` (`ARM_PLANTED`) — power against late insider activity;
      * ``seam_rejection_rate`` (`ARM_SEAM`) — how much of that power the splice
        alone could account for. A window is only usable where detection sits
        far above this;
      * ``null_rejection_rate`` (`ARM_NULL`) — the realized size, the only
        rejection rate that should be compared with ``alpha``.

    Run this **before** any real-data run and lock ``W`` from its output; the
    locked pair is `LOCKED_WINDOW_S` / `LOCKED_EMBARGO_S`.

    Args:
        windows: Candidate ``(W, w)`` pairs to score.
        n_replicates: Replicates per arm. The simulation is shared across
            windows, so that cost is paid once however long the grid is.
        n_permutations: Permutation draws per market.
        seed: Root RNG seed for both the simulation and the permutations.
        alpha: Level the three rejection rates are read at.

    Returns:
        ``{"alpha", "n_replicates", "n_permutations", "seed", "windows": [...]}``
        with one entry per candidate window carrying the three rates, the mean
        p-value in each arm, the null arm's raw p-values (for a uniformity
        check), and how many replicates were excluded.
    """
    # Simulate once, reuse for every window: the windows differ only in how the
    # same score paths are sliced, and re-simulating per window would make the
    # comparison across windows noisier than the effect being measured.
    simulated: list[dict[str, tuple[MarketScores, float]]] = []
    for i in range(n_replicates):
        rng = np.random.default_rng([seed, _STREAM_SIMULATE, i])
        simulated.append(
            {
                arm: simulate_market_scores(f"{arm[0]}{i:03d}", arm=arm, rng=rng)
                for arm in CALIBRATION_ARMS
            },
        )

    rows: list[dict[str, Any]] = []
    for window in windows:
        p_by_arm: dict[str, list[float]] = {arm: [] for arm in CALIBRATION_ARMS}
        n_excluded = 0
        for i, replicate in enumerate(simulated):
            for j, arm in enumerate(CALIBRATION_ARMS):
                scores, close_ts = replicate[arm]
                outcome = analyze_market(
                    scores,
                    close_ts,
                    window=window,
                    n_permutations=n_permutations,
                    rng=np.random.default_rng([seed, _STREAM_CALIBRATE, i, j]),
                )
                if isinstance(outcome, ExcludedMarket):
                    n_excluded += 1
                else:
                    p_by_arm[arm].append(outcome.p_value)
        rows.append(
            {
                **window.to_dict(),
                "n_excluded": n_excluded,
                "detection_rate": _rejection_rate(p_by_arm[ARM_PLANTED], alpha),
                "seam_rejection_rate": _rejection_rate(p_by_arm[ARM_SEAM], alpha),
                "null_rejection_rate": _rejection_rate(p_by_arm[ARM_NULL], alpha),
                "mean_p": {
                    arm: _mean_or_nan(p_by_arm[arm]) for arm in CALIBRATION_ARMS
                },
                "p_values_null": [float(p) for p in p_by_arm[ARM_NULL]],
            },
        )
    return {
        "alpha": alpha,
        "n_replicates": n_replicates,
        "n_permutations": n_permutations,
        "seed": seed,
        "arms_note": (
            "detection_rate is power (planted arm); seam_rejection_rate is the "
            "splice-artefact control, not a null; null_rejection_rate is the "
            "realized size and the only rate comparable with alpha."
        ),
        "windows": rows,
    }


def _rejection_rate(p_values: Sequence[float], alpha: float) -> float:
    """Fraction of ``p_values`` below ``alpha``; NaN when there are none."""
    if not p_values:
        return math.nan
    return float(np.mean(np.asarray(p_values, dtype=float) < alpha))


def _mean_or_nan(values: Sequence[float]) -> float:
    """Mean of ``values``; NaN when the sequence is empty."""
    return float(np.mean(values)) if values else math.nan
