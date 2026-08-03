"""Labeled-case study: does the streaming score light up on a *known* insider?

Every other evaluation in this repository is either synthetic (labels we
planted) or unlabelled (real markets, no ground truth). This module is the one
place where an externally labelled real episode is available:
*U.S. v. Gannon Ken Van Dyke* / *CFTC v. Van Dyke*, No. 1:26-cv-03369
(S.D.N.Y.), in which a U.S. Army Master Sergeant is alleged to have bought
"Yes" shares in Polymarket Venezuela/Maduro contracts on classified knowledge
of the operation that captured Nicolas Maduro, before it was announced on
2026-01-03.

**The manifest is the experiment (plan 2026-07-23-005 KTD5).** Which markets
belong to the cluster, which wallet is the anchor, and where the analysis
window starts and stops are *not* decided by code — they are read from
``results/case_studies/van_dyke/markets.json``, which records for each entry
the primary-source citation it came from and whether that source could actually
be read. Nothing in this module infers a market or a wallet from the data. That
is what makes the case study reproducible rather than a story fitted to a
score.

**The headline claim is per-trade timing, not the wallet ranking.** The
complaint describes on the order of ten purchases. ARCHITECTURE.md 9.5 puts the
wallet posterior in prior-dominated territory below ~20 trades and calls it
meaningful only above ~100, so a wallet ranking computed from this episode is
mostly prior. `data_sufficiency_rows` computes that verdict per wallet and
`format_report` prints it in a section that is not optional; `headline_claim`
refuses to lead with a rank when the anchored wallet is prior-dominated.

**Scope.** This module does no fetching, no scoring and no statistics beyond
means and ranks. Trades arrive already scored by ``score_stream.py --replay``
(so the no-lookahead property is inherited, exactly as in
`src.analysis.event_study`), and the provenance gate there is reused verbatim
rather than reimplemented.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

log = logging.getLogger(__name__)

CASE_STUDY_SCHEMA_VERSION = 1

# ARCHITECTURE.md 9.5, "Wallet posteriors": meaningful at >= ~100 trades,
# prior-dominated below ~20. The band between them is neither, and is labelled
# as such rather than being rounded to whichever neighbour is convenient.
THETA_W_MEANINGFUL_N_TRADES = 100
THETA_W_PRIOR_DOMINATED_N_TRADES = 20

SUFFICIENCY_PRIOR_DOMINATED = "prior-dominated"
SUFFICIENCY_WEAK = "weak"
SUFFICIENCY_MEANINGFUL = "meaningful"

DEFAULT_TOP_K = 10
# How many individual trades the timing section lists. Small on purpose: the
# section is evidence about *when* the score moved, and a long tail of ordinary
# trades makes that harder to read, not better supported.
DEFAULT_TOP_TRADES = 15

# A cold-started stream scorer is not a weaker detector, it is *no* detector:
# `stream_scoring.cold_start` leaves `theta_w` empty (every wallet sits at the
# Beta prior mean) and the logistic coefficients at their uninformative
# defaults, and `score_stream --n-refresh` defaults to never refreshing them.
# Every P(Z) is then the prior plus filter noise, and a ranking over it is a
# ranking of noise. This was found the hard way on the first real run of this
# case study, which produced a flat 0.050 for all 22,892 wallets — so the
# report now says so itself rather than leaving a reader to notice.
_COLD_START_WARNING = (
    "**COLD START — THIS RUN IS NOT A RESULT.** The scores provenance records "
    "`warm_start: null`. `stream_scoring.cold_start` leaves `theta_w` empty "
    "and the logistic coefficients uninformative, and `--n-refresh` defaults "
    "to never refreshing them, so every P(Z) is the prior mean plus filter "
    "noise and any ranking over it is a ranking of noise. Re-run "
    "`score_stream.py --replay` with `--warm-start <fitted VEM artifact>` "
    "before reading anything below as evidence about the model."
)

# Below this spread, an in-window score series is one value plus float noise.
# 1e-6 is far above float64 accumulation error over a few thousand filter steps
# and far below any P(Z) difference this project would ever call a signal, so
# the two failure directions cannot both bite.
_FLAT_SCORE_TOL = 1e-6

# A warm start can be present and still leave the anchored wallet structurally
# unscoreable. That is what the first warm-started run of this case study hit:
# the fitted artifact had `estimate_betas: false` (so beta_S = beta_Z = 0, which
# deletes the size and persistence channels) and sigma2_0 == sigma2_1 to machine
# precision (the P10 order-constraint bind, confirmed on real data), while the
# anchored wallet was absent from the training wallet index and so held theta_w
# at the Beta(1, 19) prior mean. logit(pi_Z) was then a constant and every one
# of its trades scored 0.050000. The run cannot distinguish "the model looked
# and saw nothing" from "the model was never able to look", so the report must
# not report the first. This is a *different* failure from a cold start — the
# provenance looks healthy — which is why it is detected from the scores
# themselves rather than from the presence of a warm-start path.
_UNTESTED_ANCHOR_WARNING = (
    "**THE ANCHORED WALLET WAS NOT TESTED BY THIS RUN.** Its in-window P(Z) "
    "series is constant to within 1e-6, so no elevation computed from it is a "
    "measurement in either direction. Check the warm-start artifact before "
    "reading the ranking below: if it was fitted with `estimate_betas: false` "
    "then beta_S = beta_Z = 0 and the only per-trade channel left is "
    "`theta_w`, which sits at the prior mean for any wallet absent from the "
    "training index — a constant. This run therefore reports **no evidence "
    "either way** about whether the model detects this trader."
)

_CAVEATS = (
    "**One case.** n = 1. Nothing here estimates a false-positive rate, a "
    "detection rate, or any quantity that generalizes. A score that lights up "
    "on the one labelled episode available is consistent with a useful "
    "detector and equally consistent with a detector that lights up often.",
    "**Post-hoc identification.** The markets, the wallet pattern and the "
    "window all come from a charging document written after the fact. Nobody "
    "pointed this detector at Polymarket in December 2025 and got an alert. "
    "The no-lookahead guarantee inherited from replay scoring is about the "
    "*scorer's state*, not about how the cluster was chosen.",
    "**No counterfactual.** There is no matched control episode — no market "
    "where the same news broke with no insider present — so the elevation "
    "reported here has no null it was tested against. The pre-registered "
    "permutation test lives in `src.analysis.event_study`; this module "
    "deliberately reports description, not a p-value.",
    "**Resolution-period contamination is not filtered.** The pull that feeds "
    "this study sets `--pre-resolution-days 0` (see the pull section), so the "
    "known over-flagging near resolution (ARCHITECTURE.md 9.5) is present in "
    "the data. Trades after the public announcement are outside the analysis "
    "window for exactly this reason, but the contamination is real and the "
    "in-window scores are not immune to a market already drifting toward its "
    "resolution.",
    "**The anchor is a redacted pattern.** The complaint gives four leading "
    "and four trailing hex characters of the wallet address. A match is "
    "strong evidence but not a certified identification, and a run that "
    "matches zero or several wallets is inconclusive rather than negative.",
    "**Every score inherits the warm start's fit quality.** The scores are "
    "only as good as the VEM artifact named in the provenance section, and "
    "that fit's own diagnostics are not re-checked here. Read them at source: "
    "a fit with the betas not estimated, with the sigma2 order constraint "
    "binding, with a failed PSIS k-hat, or from a single un-jittered restart "
    "carries an initialization sensitivity that this case study reports "
    "nothing about and cannot correct for.",
)


# ---------------- Errors ----------------


class ManifestError(ValueError):
    """Raised when the case manifest is missing, malformed, or incomplete.

    Malformed here always means *fatal*. A case study whose manifest half
    parsed would silently analyse a different cluster than the one documented,
    which is the single failure this design exists to prevent.
    """


# ---------------- Manifest ----------------


def _parse_ts(value: Any, *, field: str) -> float:
    """Parse a manifest timestamp into unix seconds.

    Accepts a number (already unix seconds) or an ISO-8601 string, with or
    without a trailing ``Z``. A naive string is read as UTC: every timestamp in
    the manifest is documented in UTC, and guessing local time would silently
    shift the analysis window by hours.

    Args:
        value: Raw manifest value.
        field: Dotted manifest path, used only in the error message.

    Returns:
        Unix seconds.

    Raises:
        ManifestError: If ``value`` is neither a number nor a parseable
            ISO-8601 string.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ManifestError(f"{field}: {value!r} is not ISO-8601 ({exc})") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    raise ManifestError(f"{field}: expected a timestamp, got {type(value).__name__}")


def _require(payload: Mapping[str, Any], key: str, *, where: str) -> Any:
    """Fetch a required manifest key or raise `ManifestError` naming it."""
    if key not in payload:
        raise ManifestError(f"{where}: required key {key!r} is missing")
    return payload[key]


def _iso(ts: float) -> str:
    """Format unix seconds as a UTC ISO-8601 string, seconds resolution."""
    return (
        datetime.fromtimestamp(float(ts), tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class CaseMarket:
    """One market the manifest places in the cluster.

    Attributes:
        slug: Polymarket slug, as `scripts.pull_data` takes it.
        condition_id: On-chain condition id — the stable key, and the one the
            scores JSONL carries as ``market``.
        question: Market question, for the report table.
        role: ``primary``, ``cluster`` or ``control``; free-form, printed as-is.
        why: Why this market is in the cluster, in the manifest's own words.
        cross_check: The independent fact that confirms the title-to-slug map.
        resolved: Resolved outcome, as recorded in the manifest.
        verified: Whether the manifest claims a readable primary source.
    """

    slug: str
    condition_id: str
    question: str
    role: str
    why: str
    cross_check: str
    resolved: str
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "slug": self.slug,
            "condition_id": self.condition_id,
            "question": self.question,
            "role": self.role,
            "why": self.why,
            "cross_check": self.cross_check,
            "resolved": self.resolved,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class TimelineEvent:
    """One dated event from the charging documents.

    Attributes:
        ts: Unix seconds.
        label: One-line description, printed and used as the figure annotation.
        source: Source id from the manifest's ``case.sources`` list.
        citation: Paragraph or field the fact came from.
        verified: Whether that source could actually be read.
    """

    ts: float
    label: str
    source: str
    citation: str
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view, carrying both epoch and ISO forms."""
        return {
            "ts": self.ts,
            "iso": _iso(self.ts),
            "label": self.label,
            "source": self.source,
            "citation": self.citation,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class AnalysisWindow:
    """The pre-disclosure interval the elevation table is computed over.

    Attributes:
        start_ts: Inclusive lower bound, unix seconds.
        end_ts: Inclusive upper bound, unix seconds.
        rationale: Why the manifest drew it there.
    """

    start_ts: float
    end_ts: float
    rationale: str

    def __post_init__(self) -> None:
        """Reject an empty window.

        Raises:
            ManifestError: If the window does not end after it starts.
        """
        if self.end_ts <= self.start_ts:
            raise ManifestError(
                f"analysis_window: end {_iso(self.end_ts)} does not follow start "
                f"{_iso(self.start_ts)}",
            )

    def mask(self, ts: np.ndarray) -> np.ndarray:
        """Boolean mask of the timestamps inside the closed window."""
        ts = np.asarray(ts, dtype=float)
        return (ts >= self.start_ts) & (ts <= self.end_ts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "start": _iso(self.start_ts),
            "end": _iso(self.end_ts),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class WalletAnchor:
    """How the manifest identifies the charged wallet.

    The complaint redacts the middle of the address, so the anchor is normally
    a regex over the observed wallet set rather than a literal address. Holding
    it as a *pattern* is deliberate: it keeps the identification auditable (the
    pattern is in the manifest, next to its citation) and it makes "how many
    wallets matched" a reportable number instead of an assumption.

    Attributes:
        handle: Platform handle from the charging document, if any.
        address: Full address when a source publishes one; otherwise None.
        address_pattern: Regex matched against wallet addresses; None disables
            pattern anchoring.
        citation: Where in the source the anchor comes from.
        note: The manifest's own caveat, printed verbatim in the report.
    """

    handle: str | None
    address: str | None
    address_pattern: str | None
    citation: str
    note: str

    def matches(self, address: str) -> bool:
        """Whether ``address`` is the anchored wallet under this manifest.

        Args:
            address: Wallet address as the scores JSONL carries it.

        Returns:
            True when it equals the literal address (case-insensitively) or
            matches the pattern. False when the manifest anchors on neither,
            so an anchorless manifest flags nothing rather than everything.
        """
        if self.address and address.lower() == self.address.lower():
            return True
        if self.address_pattern:
            return re.fullmatch(self.address_pattern, address) is not None
        return False

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "handle": self.handle,
            "address": self.address,
            "address_pattern": self.address_pattern,
            "citation": self.citation,
            "note": self.note,
        }


@dataclass(frozen=True)
class PullSpec:
    """The documented data pull, so the bundle is reproducible from here.

    Attributes:
        command: The exact `scripts.pull_data` invocation.
        pre_resolution_days: The filter setting used; 0 for this study.
        full_history: Whether the pull walks the whole history.
        deviation_note: Why the pre-resolution default is overridden.
        capture_note: How the replayable capture is produced.
    """

    command: str
    pre_resolution_days: float
    full_history: bool
    deviation_note: str
    capture_note: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "command": self.command,
            "pre_resolution_days": self.pre_resolution_days,
            "full_history": self.full_history,
            "deviation_note": self.deviation_note,
            "capture_note": self.capture_note,
        }


@dataclass(frozen=True)
class CaseManifest:
    """The whole checked-in case definition (KTD5).

    Attributes:
        path: Where it was loaded from, carried into the summary JSON.
        case: The ``case`` block verbatim — name, citations, source list.
        identification: The ``identification`` block verbatim.
        reconstruction: The ``reconstruction`` block verbatim, if present — the
            record of how a redacted anchor was resolved against public data,
            and of which itemized figures that resolution reproduced. Empty
            when the manifest claims no such check.
        window: The analysis window.
        markets: Cluster markets, in manifest order.
        timeline: Dated events, sorted ascending.
        anchor: Wallet anchor.
        pull: Pull spec.
        unverified: Claims the manifest could *not* stand behind.
    """

    path: str
    case: dict[str, Any]
    identification: dict[str, Any]
    reconstruction: dict[str, Any]
    window: AnalysisWindow
    markets: tuple[CaseMarket, ...]
    timeline: tuple[TimelineEvent, ...]
    anchor: WalletAnchor
    pull: PullSpec
    unverified: tuple[str, ...]

    @property
    def condition_ids(self) -> tuple[str, ...]:
        """Cluster condition ids, in manifest order."""
        return tuple(m.condition_id for m in self.markets)

    def market_by_id(self, condition_id: str) -> CaseMarket | None:
        """Look a cluster market up by condition id, or None if absent."""
        for market in self.markets:
            if market.condition_id == condition_id:
                return market
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the whole manifest."""
        return {
            "path": self.path,
            "case": self.case,
            "identification": self.identification,
            "reconstruction": self.reconstruction,
            "analysis_window": self.window.to_dict(),
            "markets": [m.to_dict() for m in self.markets],
            "doj_timeline": [e.to_dict() for e in self.timeline],
            "wallet_anchor": self.anchor.to_dict(),
            "pull": self.pull.to_dict(),
            "unverified": list(self.unverified),
        }


def _market_from_dict(raw: Any, *, index: int) -> CaseMarket:
    """Build one `CaseMarket`, raising `ManifestError` on a bad entry."""
    where = f"markets[{index}]"
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: expected an object, got {type(raw).__name__}")
    return CaseMarket(
        slug=str(_require(raw, "slug", where=where)),
        condition_id=str(_require(raw, "condition_id", where=where)),
        question=str(raw.get("question", "")),
        role=str(raw.get("role", "cluster")),
        why=str(raw.get("why", "")),
        cross_check=str(raw.get("cross_check", "")),
        resolved=str(raw.get("resolved", "")),
        verified=bool(raw.get("verified", False)),
    )


def _event_from_dict(raw: Any, *, index: int) -> TimelineEvent:
    """Build one `TimelineEvent`, raising `ManifestError` on a bad entry."""
    where = f"doj_timeline[{index}]"
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: expected an object, got {type(raw).__name__}")
    return TimelineEvent(
        ts=_parse_ts(_require(raw, "ts", where=where), field=f"{where}.ts"),
        label=str(_require(raw, "label", where=where)),
        source=str(raw.get("source", "")),
        citation=str(raw.get("citation", "")),
        verified=bool(raw.get("verified", False)),
    )


def load_manifest(path: str | Path) -> CaseManifest:
    """Load and validate the checked-in case manifest.

    Args:
        path: Manifest JSON, normally
            ``results/case_studies/van_dyke/markets.json``.

    Returns:
        The parsed `CaseManifest`.

    Raises:
        ManifestError: If the file is missing, is not JSON, is not an object,
            declares an unknown schema version, or omits a required key. There
            is deliberately no lenient path: a partially understood manifest
            would analyse an undocumented cluster.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"case manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{path} is not readable JSON ({exc})") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError(
            f"{path}: expected a JSON object, got {type(payload).__name__}",
        )

    version = payload.get("schema_version")
    if version != CASE_STUDY_SCHEMA_VERSION:
        raise ManifestError(
            f"{path}: schema_version {version!r}, expected "
            f"{CASE_STUDY_SCHEMA_VERSION}",
        )

    raw_markets = _require(payload, "markets", where=str(path))
    if not isinstance(raw_markets, Sequence) or isinstance(raw_markets, (str, bytes)):
        raise ManifestError(f"{path}: 'markets' must be a JSON array")
    markets = tuple(
        _market_from_dict(raw, index=i) for i, raw in enumerate(raw_markets)
    )
    if not markets:
        raise ManifestError(f"{path}: 'markets' is empty; there is no cluster")

    raw_window = _require(payload, "analysis_window", where=str(path))
    if not isinstance(raw_window, Mapping):
        raise ManifestError(f"{path}: 'analysis_window' must be an object")
    window = AnalysisWindow(
        start_ts=_parse_ts(
            _require(raw_window, "start", where="analysis_window"),
            field="analysis_window.start",
        ),
        end_ts=_parse_ts(
            _require(raw_window, "end", where="analysis_window"),
            field="analysis_window.end",
        ),
        rationale=str(raw_window.get("rationale", "")),
    )

    raw_anchor = payload.get("wallet_anchor") or {}
    if not isinstance(raw_anchor, Mapping):
        raise ManifestError(f"{path}: 'wallet_anchor' must be an object")
    pattern = raw_anchor.get("address_pattern")
    if pattern is not None:
        try:
            re.compile(str(pattern))
        except re.error as exc:
            raise ManifestError(
                f"{path}: wallet_anchor.address_pattern is not a regex ({exc})",
            ) from exc
    anchor = WalletAnchor(
        handle=(
            str(raw_anchor["handle"]) if raw_anchor.get("handle") is not None else None
        ),
        address=(
            str(raw_anchor["address"])
            if raw_anchor.get("address") is not None
            else None
        ),
        address_pattern=str(pattern) if pattern is not None else None,
        citation=str(raw_anchor.get("citation", "")),
        note=str(raw_anchor.get("note", "")),
    )

    raw_pull = payload.get("pull") or {}
    if not isinstance(raw_pull, Mapping):
        raise ManifestError(f"{path}: 'pull' must be an object")
    pull = PullSpec(
        command=str(raw_pull.get("command", "")),
        pre_resolution_days=float(raw_pull.get("pre_resolution_days", 0.0)),
        full_history=bool(raw_pull.get("full_history", True)),
        deviation_note=str(raw_pull.get("deviation_note", "")),
        capture_note=str(raw_pull.get("capture_note", "")),
    )

    raw_timeline = payload.get("doj_timeline") or []
    if not isinstance(raw_timeline, Sequence) or isinstance(raw_timeline, (str, bytes)):
        raise ManifestError(f"{path}: 'doj_timeline' must be a JSON array")
    timeline = tuple(
        sorted(
            (_event_from_dict(raw, index=i) for i, raw in enumerate(raw_timeline)),
            key=lambda event: event.ts,
        ),
    )

    return CaseManifest(
        path=str(path),
        case=dict(payload.get("case") or {}),
        identification=dict(payload.get("identification") or {}),
        reconstruction=dict(payload.get("reconstruction") or {}),
        window=window,
        markets=markets,
        timeline=timeline,
        anchor=anchor,
        pull=pull,
        unverified=tuple(str(x) for x in (payload.get("unverified") or [])),
    )


# ---------------- Scored trades ----------------


@dataclass(frozen=True)
class ScoredTrades:
    """Replayed per-trade scores for the cluster, in time order.

    Unlike `src.analysis.event_study.MarketScores` this keeps the ``wallet``
    column, which is the whole point of a wallet-anchored study, and it does
    *not* split by market: the cluster is analysed as one episode because the
    charged conduct spans several contracts on the same news.

    Attributes:
        ts: (n,) trade timestamps in unix seconds, sorted ascending.
        p_z: (n,) ``q(Z_t = 1)`` per trade.
        market: (n,) condition ids.
        wallet: (n,) wallet addresses.
    """

    ts: np.ndarray
    p_z: np.ndarray
    market: np.ndarray
    wallet: np.ndarray

    @property
    def n(self) -> int:
        """Number of scored trades."""
        return int(self.ts.size)


def load_scored_trades(
    path: str | Path,
    *,
    condition_ids: Iterable[str] | None = None,
) -> ScoredTrades:
    """Read a `score_stream.py` scores JSONL, keeping the wallet column.

    Args:
        path: Scores JSONL, one ``ScoredTrade`` object per line.
        condition_ids: Cluster markets to keep. Records for any other market
            are dropped, so pointing this at a whole-run scores file still
            analyses only the manifest's cluster. None keeps everything.

    Returns:
        `ScoredTrades` sorted by timestamp (stable, so equal-timestamp trades
        keep the replay order that produced them).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a non-blank line is not valid JSON. Skipping a bad line
            would quietly shorten the window it fell in.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"scores file not found: {path}")
    keep = set(condition_ids) if condition_ids is not None else None

    ts: list[float] = []
    p_z: list[float] = []
    market: list[str] = []
    wallet: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON ({exc})") from exc
            condition_id = str(record.get("market", ""))
            if keep is not None and condition_id not in keep:
                continue
            ts.append(float(record["ts"]))
            p_z.append(float(record["p_z"]))
            market.append(condition_id)
            wallet.append(str(record.get("wallet", "")))

    ts_arr = np.asarray(ts, dtype=float)
    order = np.argsort(ts_arr, kind="stable")
    return ScoredTrades(
        ts=ts_arr[order],
        p_z=np.asarray(p_z, dtype=float)[order],
        market=np.asarray(market, dtype=object)[order],
        wallet=np.asarray(wallet, dtype=object)[order],
    )


# ---------------- Wallet rows and data sufficiency ----------------


def sufficiency_label(n_trades: int) -> str:
    """Classify a wallet's evidence against the ARCHITECTURE.md 9.5 thresholds.

    Args:
        n_trades: Trades the wallet contributes to the fit.

    Returns:
        `SUFFICIENCY_PRIOR_DOMINATED` below ~20 trades, `SUFFICIENCY_MEANINGFUL`
        at or above ~100, `SUFFICIENCY_WEAK` in between — the band where the
        posterior has moved off the prior but nothing in this repository has
        ever shown it to be trustworthy.
    """
    if n_trades < THETA_W_PRIOR_DOMINATED_N_TRADES:
        return SUFFICIENCY_PRIOR_DOMINATED
    if n_trades >= THETA_W_MEANINGFUL_N_TRADES:
        return SUFFICIENCY_MEANINGFUL
    return SUFFICIENCY_WEAK


@dataclass(frozen=True)
class WalletRow:
    """One wallet's activity inside the analysis window.

    Attributes:
        wallet: Wallet address.
        n_window: Trades inside the analysis window.
        n_total: Trades anywhere in the pulled cluster history. Reported next
            to ``n_window`` because the `theta_w` reliability thresholds are
            about total evidence, not about the window.
        mean_p_z: Mean ``P(Z)`` over the in-window trades.
        max_p_z: Largest in-window ``P(Z)``.
        min_p_z: Smallest in-window ``P(Z)``. Carried alongside ``max_p_z`` so
            the score *dispersion* is visible in the artifact: a wallet whose
            scores never move is one the run measured nothing about, and that
            has to be readable off the JSON rather than inferred (see
            `WalletRow.is_flat` and `CaseStudySummary.anchor_is_untested`).
        elevation: ``mean_p_z`` minus the cluster-wide baseline mean, so a
            wallet is read against the cluster's own score level.
        anchored: Whether the manifest's wallet anchor matches this address.
        sufficiency: `sufficiency_label` of ``n_total``.
    """

    wallet: str
    n_window: int
    n_total: int
    mean_p_z: float
    max_p_z: float
    min_p_z: float
    elevation: float
    anchored: bool
    sufficiency: str

    @property
    def is_flat(self) -> bool:
        """Whether this wallet's in-window scores carry no dispersion at all.

        The threshold is numerical, not statistical: scores this close together
        are one value plus float noise. A flat series means the score never
        responded to anything the wallet did, so no elevation computed from it
        — in either direction — is a measurement.
        """
        return (self.max_p_z - self.min_p_z) < _FLAT_SCORE_TOL

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "wallet": self.wallet,
            "n_window": self.n_window,
            "n_total": self.n_total,
            "mean_p_z": self.mean_p_z,
            "max_p_z": self.max_p_z,
            "min_p_z": self.min_p_z,
            "elevation": self.elevation,
            "flat": self.is_flat,
            "anchored": self.anchored,
            "sufficiency": self.sufficiency,
        }


def _wallet_rows(
    trades: ScoredTrades,
    manifest: CaseManifest,
    *,
    baseline_mean: float,
) -> tuple[WalletRow, ...]:
    """Build the in-window wallet table, ranked by mean ``P(Z)`` descending.

    Only in-window trades produce a row; a wallet that traded the cluster
    exclusively outside the window has no in-window mean to rank and is
    excluded rather than entered with a fabricated zero. Its trades stay in the
    timeline, which is where "this wallet was active, just not then" belongs.

    Args:
        trades: All cluster trades.
        manifest: The case definition, read for the window and the anchor.
        baseline_mean: Cluster-wide mean ``P(Z)``, subtracted for elevation.

    Returns:
        Rows sorted by ``mean_p_z`` descending, ties broken by address so the
        ordering is reproducible.
    """
    in_window = manifest.window.mask(trades.ts)
    totals: dict[str, int] = {}
    for address in trades.wallet:
        totals[address] = totals.get(address, 0) + 1

    grouped: dict[str, list[float]] = {}
    for address, score in zip(trades.wallet[in_window], trades.p_z[in_window]):
        grouped.setdefault(address, []).append(float(score))

    rows = []
    for address, scores in grouped.items():
        # `totals` is built over every trade, so the in-window `grouped` keys are
        # a subset of it; the `len(scores)` default is only a floor for the
        # impossible case, and it feeds both `n_total` and its sufficiency label.
        n_total = totals.get(address, len(scores))
        mean_p_z = float(np.mean(scores))
        rows.append(
            WalletRow(
                wallet=address,
                n_window=len(scores),
                n_total=n_total,
                mean_p_z=mean_p_z,
                max_p_z=float(np.max(scores)),
                min_p_z=float(np.min(scores)),
                elevation=mean_p_z - baseline_mean,
                anchored=manifest.anchor.matches(address),
                sufficiency=sufficiency_label(n_total),
            )
        )
    return tuple(sorted(rows, key=lambda row: (-row.mean_p_z, row.wallet)))


def data_sufficiency_rows(wallets: Sequence[WalletRow]) -> dict[str, int]:
    """Count wallets by `sufficiency_label`, for the mandatory report section.

    Args:
        wallets: In-window wallet rows.

    Returns:
        ``{label: count}`` covering all three labels, zeros included, so the
        report table has a fixed shape whatever the data looks like.
    """
    counts = {
        SUFFICIENCY_PRIOR_DOMINATED: 0,
        SUFFICIENCY_WEAK: 0,
        SUFFICIENCY_MEANINGFUL: 0,
    }
    for row in wallets:
        counts[row.sufficiency] += 1
    return counts


# ---------------- Summary ----------------


@dataclass(frozen=True)
class CaseStudySummary:
    """Everything one case-study run produces.

    Attributes:
        manifest: The case definition the run was driven by.
        trades: The cluster's scored trades, kept so the figure and the report
            work off one object.
        wallets: In-window wallet rows, ranked.
        top_k: How many rows the report's ranking table prints.
        baseline_mean_p_z: Mean ``P(Z)`` over every cluster trade.
        window_mean_p_z: Mean ``P(Z)`` over in-window trades; NaN when the
            window holds none.
        n_trades_window: Trades inside the window.
        n_wallets_total: Distinct wallets anywhere in the cluster.
        markets_without_trades: Manifest markets the scores file has no trade
            for — a pull gap, reported rather than silently dropped.
        markets_off_manifest: Markets present in the scores file but absent
            from the manifest. Non-empty means the scores were not filtered to
            the cluster.
        anchored_wallets: Addresses matching the manifest anchor, any window.
        top_trades: Highest-``P(Z)`` in-window trades, the timing evidence.
        provenance: The replay sidecar of the scores file, carried through.
    """

    manifest: CaseManifest
    trades: ScoredTrades
    wallets: tuple[WalletRow, ...]
    top_k: int
    baseline_mean_p_z: float
    window_mean_p_z: float
    n_trades_window: int
    n_wallets_total: int
    markets_without_trades: tuple[str, ...]
    markets_off_manifest: tuple[str, ...]
    anchored_wallets: tuple[str, ...]
    top_trades: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]

    @property
    def is_cold_start(self) -> bool:
        """Whether the scores came from a run with no warm-start artifact.

        Read off the replay sidecar, whose ``warm_start`` field
        `scripts.score_stream._write_run_meta` writes as JSON null when
        ``--warm-start`` was omitted. A summary with no provenance at all is
        treated as cold: an unprovenanced score file cannot be shown to have
        been warm-started, and the safe default is the one that refuses to
        make a claim.
        """
        return self.provenance.get("warm_start") is None

    @property
    def anchor_is_untested(self) -> bool:
        """Whether the run carries no information about the anchored wallet.

        True when exactly one wallet matched the anchor, it has in-window
        trades, and those trades' scores are flat (`WalletRow.is_flat`). The
        distinction this guards is the one the whole case study turns on: a
        flat series means the model was never able to look, which is not the
        same finding as the model looking and seeing nothing, and only the
        second would be evidence about the detector.

        Read off the scores rather than the warm-start artifact on purpose —
        the degeneracy has several possible causes (betas not estimated, an
        unseen wallet pinned at the theta_w prior, a bound sigma2 order
        constraint) and the constant series is the one symptom common to all
        of them.
        """
        return len(self.anchored_rows) == 1 and self.anchored_rows[0].is_flat

    @property
    def caveats(self) -> tuple[str, ...]:
        """The caveat list, with any run-invalidating banner in front.

        One source for the report and the JSON so the two cannot drift, and
        ordered worst-first: a banner says the run is not a result, which a
        reader has to see before the caveats that merely qualify one.
        """
        banners = []
        if self.is_cold_start:
            banners.append(_COLD_START_WARNING)
        if self.anchor_is_untested:
            banners.append(_UNTESTED_ANCHOR_WARNING)
        return (*banners, *_CAVEATS)

    @property
    def anchored_rows(self) -> tuple[WalletRow, ...]:
        """In-window rows for anchored wallets, in ranking order."""
        return tuple(row for row in self.wallets if row.anchored)

    @property
    def reported_wallets(self) -> tuple[WalletRow, ...]:
        """The rows the report prints: the top ``top_k``, plus every anchor.

        An anchored wallet ranked below the cut is exactly the case a top-K
        table would hide, so it is appended rather than dropped. The summary
        JSON carries the same set: a real cluster has tens of thousands of
        wallets, and dumping all of them turns a reviewable artifact into a
        multi-megabyte data file that nobody reads and nobody can commit.
        """
        shown = list(self.wallets[: self.top_k])
        shown += [row for row in self.anchored_rows if row not in shown]
        return tuple(shown)

    def rank_of(self, wallet: str) -> int | None:
        """1-based rank of ``wallet`` in the in-window table, or None."""
        for i, row in enumerate(self.wallets, start=1):
            if row.wallet == wallet:
                return i
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the whole run."""
        return {
            "schema_version": CASE_STUDY_SCHEMA_VERSION,
            "manifest": self.manifest.to_dict(),
            "n_trades_total": self.trades.n,
            "n_trades_window": self.n_trades_window,
            "n_wallets_total": self.n_wallets_total,
            "n_wallets_window": len(self.wallets),
            "baseline_mean_p_z": self.baseline_mean_p_z,
            "window_mean_p_z": self.window_mean_p_z,
            "markets_without_trades": list(self.markets_without_trades),
            "markets_off_manifest": list(self.markets_off_manifest),
            "anchored_wallets": list(self.anchored_wallets),
            "anchored_ranks": {
                row.wallet: self.rank_of(row.wallet) for row in self.anchored_rows
            },
            "top_k": self.top_k,
            "wallets": [row.to_dict() for row in self.reported_wallets],
            "wallets_note": (
                "The top-k rows plus every anchored wallet, matching the "
                "report's table. `n_wallets_window` is the full count; the "
                "per-wallet rows behind it are regenerated by re-running the "
                "CLI with a larger --top-k."
            ),
            "data_sufficiency": {
                "thresholds": {
                    "prior_dominated_below": THETA_W_PRIOR_DOMINATED_N_TRADES,
                    "meaningful_at_or_above": THETA_W_MEANINGFUL_N_TRADES,
                    "source": "agent_reference/ARCHITECTURE.md 9.5",
                },
                "counts": data_sufficiency_rows(self.wallets),
            },
            "top_trades": [dict(row) for row in self.top_trades],
            "cold_start": self.is_cold_start,
            "anchor_untested": self.anchor_is_untested,
            "headline_claim": headline_claim(self),
            "caveats": list(self.caveats),
            "provenance": self.provenance,
        }


def _top_trades(
    trades: ScoredTrades,
    manifest: CaseManifest,
    *,
    limit: int,
) -> tuple[dict[str, Any], ...]:
    """Highest-``P(Z)`` in-window trades, newest-first among equals.

    Args:
        trades: All cluster trades.
        manifest: Read for the window and the anchor.
        limit: How many to return.

    Returns:
        Records carrying timestamp, market, wallet, score and whether the
        wallet is anchored.
    """
    in_window = np.flatnonzero(manifest.window.mask(trades.ts))
    if in_window.size == 0:
        return ()
    order = in_window[np.argsort(-trades.p_z[in_window], kind="stable")][:limit]
    return tuple(
        {
            "ts": float(trades.ts[i]),
            "iso": _iso(float(trades.ts[i])),
            "market": str(trades.market[i]),
            "wallet": str(trades.wallet[i]),
            "p_z": float(trades.p_z[i]),
            "anchored": manifest.anchor.matches(str(trades.wallet[i])),
        }
        for i in order
    )


def headline_claim(summary: CaseStudySummary) -> str:
    """State what this run does and does not support, in one paragraph.

    The rule the plan fixes: lean on per-trade ``P(Z)`` timing, and refuse to
    lead with a wallet rank whenever the anchored wallet's evidence is
    prior-dominated by the ARCHITECTURE.md 9.5 thresholds. That is not a
    stylistic preference — a rank computed from a dozen trades is mostly a
    restatement of the prior, and quoting it as a finding would be the single
    easiest way to oversell this case study.

    Args:
        summary: The completed run.

    Returns:
        A sentence-level claim safe to quote.
    """
    if summary.is_cold_start:
        return (
            "No claim. This run was cold-started, so every P(Z) is the prior "
            "mean plus filter noise and neither the wallet ranking nor the "
            "per-trade timing carries information about the model. Re-run "
            "with `--warm-start <fitted VEM artifact>`."
        )
    if not summary.anchored_wallets:
        return (
            "No wallet in the pulled cluster matches the manifest's anchor, so "
            "this run supports no claim about the charged trader. Either the "
            "pull did not reach the alleged trades or the redacted-address "
            "pattern does not identify a wallet in this data; both are pull "
            "problems, not evidence of absence."
        )
    if len(summary.anchored_wallets) > 1:
        return (
            f"{len(summary.anchored_wallets)} wallets match the manifest's "
            "redacted-address pattern, so the anchor does not identify a "
            "single trader and no wallet-level claim is made."
        )
    row = summary.anchored_rows[0] if summary.anchored_rows else None
    if row is None:
        return (
            "The anchored wallet traded the cluster but not inside the "
            "analysis window, so it carries no in-window score and this run "
            "supports no timing claim about it."
        )
    if row.is_flat:
        return (
            f"No evidence either way. The anchored wallet's {row.n_window} "
            f"in-window trade(s) all score P(Z) = {row.mean_p_z:.6f}, a "
            f"constant — the series spread is "
            f"{row.max_p_z - row.min_p_z:.2e}, below the 1e-6 flatness "
            "tolerance. A constant cannot be elevated or unelevated, so this "
            "run does not show that the model fails to detect this trader; it "
            "shows that this configuration never gave the model a channel to "
            "detect them through. Check the warm-start artifact's `beta_S`, "
            "`beta_Z` and `estimate_betas`, and whether the wallet appears in "
            "the training wallet index, then re-run before drawing any "
            "conclusion about the detector."
        )
    rank = summary.rank_of(row.wallet)
    lead = (
        f"The anchored wallet's {row.n_window} in-window trade(s) carry a mean "
        f"P(Z) of {row.mean_p_z:.3f} against a cluster baseline of "
        f"{summary.baseline_mean_p_z:.3f} (elevation {row.elevation:+.3f}, peak "
        f"{row.max_p_z:.3f}), all of it before the public announcement at "
        f"{_iso(summary.manifest.window.end_ts)}."
    )
    if row.sufficiency == SUFFICIENCY_MEANINGFUL:
        return (
            f"{lead} With {row.n_total} trades in total the wallet posterior is "
            f"in the meaningful range (ARCHITECTURE.md 9.5), so its rank "
            f"{rank} of {len(summary.wallets)} is also reportable."
        )
    return (
        f"{lead} The wallet ranking is NOT the claim: with {row.n_total} trades "
        f"in total this wallet is {row.sufficiency} against the "
        f"ARCHITECTURE.md 9.5 thresholds, so its rank {rank} of "
        f"{len(summary.wallets)} is largely a restatement of the prior and is "
        "reported for completeness only."
    )


def run_case_study(
    trades: ScoredTrades,
    manifest: CaseManifest,
    *,
    top_k: int = DEFAULT_TOP_K,
    top_trades: int = DEFAULT_TOP_TRADES,
    provenance: Mapping[str, Any] | None = None,
) -> CaseStudySummary:
    """Score-in, description-out: build the whole case-study summary.

    No statistics beyond means and ranks are computed here, deliberately —
    see the module docstring and the caveats.

    Args:
        trades: Cluster trades, already scored by replay.
        manifest: The case definition.
        top_k: Rows the report's ranking table prints.
        top_trades: Individual trades the timing section lists.
        provenance: Replay sidecar payload to carry into the summary.

    Returns:
        The populated `CaseStudySummary`.

    Raises:
        ValueError: If ``top_k`` is not positive.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    baseline = float(trades.p_z.mean()) if trades.n else math.nan
    in_window = manifest.window.mask(trades.ts)
    n_window = int(in_window.sum())
    seen_markets = set(trades.market.tolist())
    # Distinct wallets, materialized once: the anchor is a regex, so matching it
    # per distinct address rather than per trade is the difference between one
    # `re.fullmatch` per wallet and one per trade on a cluster-sized feed.
    distinct_wallets = set(trades.wallet.tolist())
    wallets = _wallet_rows(trades, manifest, baseline_mean=baseline)

    missing = tuple(
        m.condition_id for m in manifest.markets if m.condition_id not in seen_markets
    )
    for condition_id in missing:
        log.warning(
            "manifest market %s has no scored trade; the cluster is incomplete "
            "and the report says so",
            condition_id,
        )

    anchored = tuple(
        sorted(
            address for address in distinct_wallets if manifest.anchor.matches(address)
        ),
    )
    if not anchored:
        log.warning(
            "no wallet matches the manifest anchor %r; the wallet-anchored half "
            "of the study is inconclusive",
            manifest.anchor.address_pattern or manifest.anchor.address,
        )

    return CaseStudySummary(
        manifest=manifest,
        trades=trades,
        wallets=wallets,
        top_k=top_k,
        baseline_mean_p_z=baseline,
        window_mean_p_z=(
            float(trades.p_z[in_window].mean()) if n_window else math.nan
        ),
        n_trades_window=n_window,
        n_wallets_total=len(distinct_wallets),
        markets_without_trades=missing,
        markets_off_manifest=tuple(sorted(seen_markets - set(manifest.condition_ids))),
        anchored_wallets=anchored,
        top_trades=_top_trades(trades, manifest, limit=top_trades),
        provenance=dict(provenance or {}),
    )


def write_summary(
    summary: CaseStudySummary,
    path: str | Path,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``summary`` to ``path`` as indented JSON, creating parent dirs.

    Args:
        summary: Summary produced by `run_case_study`.
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
        payload.update(dict(extra))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------- Report ----------------

# Section headings are constants because three things agree on them: the
# generated report, the tests that assert the mandatory sections exist, and any
# later reader looking for the data-sufficiency subsection the plan requires.
SECTION_CASE = "## 1. Case and sources"
SECTION_MARKETS = "## 2. Markets in the cluster"
SECTION_WALLETS = "## 3. Wallet ranking in the analysis window"
SECTION_SUFFICIENCY = "## 4. Data sufficiency"
SECTION_TIMING = "## 5. Per-trade P(Z) timing evidence"
SECTION_TIMELINE = "## 6. Charging-document timeline overlay"
SECTION_PULL = "## 7. Pull provenance and the pre-resolution deviation"
SECTION_CAVEATS = "## 8. Caveats"

REPORT_SECTIONS = (
    SECTION_CASE,
    SECTION_MARKETS,
    SECTION_WALLETS,
    SECTION_SUFFICIENCY,
    SECTION_TIMING,
    SECTION_TIMELINE,
    SECTION_PULL,
    SECTION_CAVEATS,
)


def _fmt(value: float, spec: str = ".4f") -> str:
    """Format a float, rendering NaN as ``n/a`` instead of ``nan``."""
    return "n/a" if not math.isfinite(value) else format(value, spec)


def _case_section(summary: CaseStudySummary) -> list[str]:
    """Build the case-and-sources section."""
    case = summary.manifest.case
    lines = [SECTION_CASE, ""]
    if case.get("name"):
        lines.append(f"**{case['name']}**")
        lines.append("")
    for key in ("summary", "criminal", "civil"):
        if case.get(key):
            lines.append(f"- {case[key]}")
    lines.append("")
    lines.append("Sources:")
    lines.append("")
    lines.append("| id | kind | read? | source |")
    lines.append("|---|---|---|---|")
    for source in case.get("sources", []):
        read = "yes" if source.get("retrieved") else "**NO**"
        lines.append(
            f"| `{source.get('id', '')}` | {source.get('kind', '')} | {read} | "
            f"{source.get('title', '')} — <{source.get('url', '')}> |",
        )
    if summary.manifest.unverified:
        lines.extend(["", "Claims this manifest could **not** verify:", ""])
        lines.extend(f"- {claim}" for claim in summary.manifest.unverified)
    if summary.manifest.identification.get("procedure"):
        lines.extend(
            [
                "",
                "Market identification is manual and documented (KTD5): "
                + str(summary.manifest.identification["procedure"]),
            ],
        )
    reconstruction = summary.manifest.reconstruction
    if reconstruction:
        lines.extend(
            [
                "",
                f"Anchor reconstruction ({reconstruction.get('status', '?')}, "
                f"{reconstruction.get('date', 'undated')}): "
                + str(reconstruction.get("method", "")),
                "",
            ],
        )
        lines.extend(f"- {check}" for check in reconstruction.get("checks", []))
        if reconstruction.get("note"):
            lines.extend(["", str(reconstruction["note"])])
    return lines


def _markets_section(summary: CaseStudySummary) -> list[str]:
    """Build the cluster-markets table."""
    counts: dict[str, int] = {}
    for condition_id in summary.trades.market.tolist():
        counts[condition_id] = counts.get(condition_id, 0) + 1

    lines = [
        SECTION_MARKETS,
        "",
        "| slug | role | resolved | scored trades | why |",
        "|---|---|---|---|---|",
    ]
    for market in summary.manifest.markets:
        n = counts.get(market.condition_id, 0)
        lines.append(
            f"| `{market.slug}` | {market.role} | {market.resolved} | "
            f"{n} | {market.why} |",
        )
    if summary.markets_without_trades:
        lines.extend(
            [
                "",
                "**Incomplete cluster.** These manifest markets contributed no "
                "scored trade, so any claim below is made on a subset of the "
                "documented cluster: "
                + ", ".join(f"`{m}`" for m in summary.markets_without_trades),
            ],
        )
    if summary.markets_off_manifest:
        lines.extend(
            [
                "",
                "**Off-manifest markets in the scores file** (not analysed): "
                + ", ".join(f"`{m}`" for m in summary.markets_off_manifest),
            ],
        )
    return lines


def _wallets_section(summary: CaseStudySummary) -> list[str]:
    """Build the top-K wallet ranking table."""
    window = summary.manifest.window
    lines = [
        SECTION_WALLETS,
        "",
        f"Window: {_iso(window.start_ts)} to {_iso(window.end_ts)}. "
        f"{summary.n_trades_window} of {summary.trades.n} cluster trades fall "
        f"inside it, across {len(summary.wallets)} of "
        f"{summary.n_wallets_total} wallets.",
        "",
        f"Rationale: {window.rationale}",
        "",
        "Trades outside the window are excluded from this table by design — a "
        "post-announcement trade is not insider trading — but they remain in "
        "the timeline and in the baseline below.",
        "",
        f"Cluster baseline mean P(Z) (all trades): "
        f"{_fmt(summary.baseline_mean_p_z)}; in-window mean: "
        f"{_fmt(summary.window_mean_p_z)}.",
        "",
        "| rank | wallet | n_window | n_total | mean P(Z) | max P(Z) | "
        "elevation | theta_w evidence | anchored |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary.reported_wallets:
        rank = summary.rank_of(row.wallet)
        lines.append(
            f"| {rank} | `{row.wallet}` | {row.n_window} | {row.n_total} | "
            f"{_fmt(row.mean_p_z)} | {_fmt(row.max_p_z)} | "
            f"{row.elevation:+.4f} | {row.sufficiency} | "
            f"{'**YES**' if row.anchored else ''} |",
        )
    if not summary.wallets:
        lines.append("| — | *no wallet traded inside the window* | | | | | | | |")

    anchor = summary.manifest.anchor
    lines.extend(
        [
            "",
            f"Anchor: handle `{anchor.handle}`, pattern "
            f"`{anchor.address_pattern}` ({anchor.citation}). "
            f"{len(summary.anchored_wallets)} wallet(s) matched.",
            "",
            anchor.note,
        ],
    )
    return lines


def _sufficiency_section(summary: CaseStudySummary) -> list[str]:
    """Build the mandatory data-sufficiency subsection.

    Required by the plan and by ARCHITECTURE.md 9.5: a ranking built from a
    handful of trades is mostly prior, and a report that prints the rank
    without printing that fact is misleading even when every number in it is
    correct.
    """
    counts = data_sufficiency_rows(summary.wallets)
    lines = [
        SECTION_SUFFICIENCY,
        "",
        "ARCHITECTURE.md 9.5 fixes the reliability thresholds for the "
        f"per-wallet posterior `theta_w`: **prior-dominated below ~"
        f"{THETA_W_PRIOR_DOMINATED_N_TRADES} trades**, **meaningful at or "
        f"above ~{THETA_W_MEANINGFUL_N_TRADES}**. Counted on each wallet's "
        "total trades across the pulled cluster, not on its in-window trades:",
        "",
        "| theta_w evidence | wallets |",
        "|---|---:|",
        f"| {SUFFICIENCY_PRIOR_DOMINATED} (< "
        f"{THETA_W_PRIOR_DOMINATED_N_TRADES}) | "
        f"{counts[SUFFICIENCY_PRIOR_DOMINATED]} |",
        f"| {SUFFICIENCY_WEAK} ({THETA_W_PRIOR_DOMINATED_N_TRADES}-"
        f"{THETA_W_MEANINGFUL_N_TRADES - 1}) | {counts[SUFFICIENCY_WEAK]} |",
        f"| {SUFFICIENCY_MEANINGFUL} (>= {THETA_W_MEANINGFUL_N_TRADES}) | "
        f"{counts[SUFFICIENCY_MEANINGFUL]} |",
        "",
    ]
    for row in summary.anchored_rows:
        lines.append(
            f"- Anchored wallet `{row.wallet}`: {row.n_total} trade(s) total, "
            f"{row.n_window} in window -> **{row.sufficiency}**.",
        )
    if not summary.anchored_rows:
        lines.append(
            "- No anchored wallet has in-window trades, so there is no "
            "wallet-level evidence to grade.",
        )
    lines.extend(
        [
            "",
            "The charging documents describe on the order of ten purchases in "
            "this cluster. That is an order of magnitude below the threshold "
            "at which this project's own wallet posterior means anything, so "
            "**the wallet ranking above is prior-dominated and is not the "
            "result of this case study.**",
            "",
            (
                # Only promise the timing section when it can deliver. With a
                # flat anchored series the next section shows a constant, and
                # pointing at it as "the evidence" would dress up a structural
                # zero as a finding.
                "Nor does the per-trade timing section rescue it: the "
                "anchored wallet's scores are constant (see the banner above), "
                "so that section describes the cluster, not the charged "
                "trader."
                if summary.anchor_is_untested
                else "The headline claim rests on the per-trade P(Z) timing "
                "evidence in the next section."
            ),
            "",
            f"**Headline claim.** {headline_claim(summary)}",
        ],
    )
    return lines


def _timing_section(summary: CaseStudySummary) -> list[str]:
    """Build the per-trade timing-evidence section."""
    lines = [
        SECTION_TIMING,
        "",
        "Highest-scoring individual trades inside the analysis window. Every "
        "score is a function of trades 0..t only (inherited from "
        "`score_stream.py --replay`), so each row is what a reader watching "
        "the stream would have seen at that moment.",
        "",
        "| timestamp (UTC) | market | wallet | P(Z) | anchored |",
        "|---|---|---|---:|---|",
    ]
    for row in summary.top_trades:
        market = summary.manifest.market_by_id(row["market"])
        label = market.slug if market is not None else row["market"]
        lines.append(
            f"| {row['iso']} | `{label}` | `{row['wallet']}` | "
            f"{row['p_z']:.4f} | {'**YES**' if row['anchored'] else ''} |",
        )
    if not summary.top_trades:
        lines.append("| — | *no trade inside the window* | | | |")
    return lines


def _timeline_section(summary: CaseStudySummary) -> list[str]:
    """Build the charging-document timeline overlay."""
    lines = [
        SECTION_TIMELINE,
        "",
        "| timestamp (UTC) | event | source | citation | verified |",
        "|---|---|---|---|---|",
    ]
    for event in summary.manifest.timeline:
        lines.append(
            f"| {_iso(event.ts)} | {event.label} | `{event.source}` | "
            f"{event.citation} | {'yes' if event.verified else '**NO**'} |",
        )
    if not summary.manifest.timeline:
        lines.append("| — | *no timeline in the manifest* | | | |")
    return lines


def _pull_section(summary: CaseStudySummary) -> list[str]:
    """Build the pull-provenance section, including the documented deviation."""
    pull = summary.manifest.pull
    lines = [
        SECTION_PULL,
        "",
        "```",
        pull.command,
        "```",
        "",
        f"`--pre-resolution-days {pull.pre_resolution_days:g}` — **a "
        "deliberate deviation from this repository's 7-day default.**",
        "",
        pull.deviation_note,
    ]
    if pull.capture_note:
        lines.extend(["", pull.capture_note])
    if summary.provenance:
        lines.extend(
            [
                "",
                f"Scores provenance: mode="
                f"{summary.provenance.get('mode')!r}, input="
                f"{summary.provenance.get('input')!r}, warm_start="
                f"{summary.provenance.get('warm_start')!r}.",
            ],
        )
    return lines


def _caveats_section(summary: CaseStudySummary) -> list[str]:
    """Build the honest-caveats section, run-invalidating banners first."""
    return [
        SECTION_CAVEATS,
        "",
        *(f"- {caveat}" for caveat in summary.caveats),
    ]


def format_report(summary: CaseStudySummary) -> str:
    """Render the whole case-study report as Markdown.

    Args:
        summary: Summary produced by `run_case_study`.

    Returns:
        The report text. Every heading in `REPORT_SECTIONS` appears exactly
        once, in order.
    """
    name = str(summary.manifest.case.get("name") or "labeled case")
    lines = [
        f"# Case study: {name}",
        "",
        "An externally labelled insider episode — the only kind of ground "
        "truth this project has that it did not plant itself. Read the "
        "data-sufficiency section before quoting anything from the wallet "
        "ranking.",
        "",
        f"Manifest: `{summary.manifest.path}` "
        f"(schema v{CASE_STUDY_SCHEMA_VERSION})",
        "",
    ]
    if summary.is_cold_start:
        lines.extend([_COLD_START_WARNING, ""])
    if summary.anchor_is_untested:
        lines.extend([_UNTESTED_ANCHOR_WARNING, ""])
    for section in (
        _case_section(summary),
        _markets_section(summary),
        _wallets_section(summary),
        _sufficiency_section(summary),
        _timing_section(summary),
        _timeline_section(summary),
        _pull_section(summary),
        _caveats_section(summary),
    ):
        lines.extend(section)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------- Figure ----------------


# Lead-in shown either side of the analysis window in the zoom panel. Long
# enough that the score's level *before* the window is visible — the panel is
# there to answer "did it rise?", which needs a before as well as an after.
_ZOOM_PAD_S = 7.0 * 86400.0


def _draw_timeline(ax, summary: CaseStudySummary, times: np.ndarray, anchored):
    """Draw one score-vs-time panel: trades, window, baseline, event rules.

    Args:
        ax: Target axes.
        summary: The completed run.
        times: Trade timestamps as timezone-aware datetimes, aligned with
            ``summary.trades``.
        anchored: Boolean mask picking out the anchored wallet's trades.
    """
    trades = summary.trades
    window = summary.manifest.window
    ax.axvspan(
        datetime.fromtimestamp(window.start_ts, tz=timezone.utc),
        datetime.fromtimestamp(window.end_ts, tz=timezone.utc),
        color="0.9",
        zorder=0,
        label="analysis window",
    )
    if trades.n:
        ax.scatter(
            times[~anchored],
            trades.p_z[~anchored],
            s=5,
            facecolors="none",
            edgecolors="0.45",
            linewidths=0.4,
            label="other wallets",
        )
    if anchored.any():
        ax.scatter(
            times[anchored],
            trades.p_z[anchored],
            s=26,
            color="C3",
            marker="D",
            zorder=3,
            label="anchored wallet",
        )
    if math.isfinite(summary.baseline_mean_p_z):
        ax.axhline(
            summary.baseline_mean_p_z,
            color="C0",
            ls="--",
            lw=0.8,
            label="cluster baseline",
        )
    for event in summary.manifest.timeline:
        ax.axvline(
            datetime.fromtimestamp(event.ts, tz=timezone.utc),
            color="0.35",
            ls=":",
            lw=0.7,
            zorder=1,
        )


def figure_case_study(summary: CaseStudySummary):
    """Build the per-trade score timeline with the charging-document overlay.

    Two panels of the *same* data, because a cluster can span a year while the
    charged conduct spans a week: the left shows every pulled trade, the right
    zooms to the analysis window plus a week either side. The zoom is a zoom,
    not a filter — nothing is dropped, and the left panel is there so a reader
    can see what the right one is a slice of.

    Args:
        summary: Summary produced by `run_case_study`.

    Returns:
        The matplotlib ``Figure``; the caller closes it.
    """
    # Deferred: `plots` transitively imports the inference stack for its
    # PG/iPMCMC panels, which an analysis-only run has no use for.
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from src.analysis.plots import set_paper_style

    set_paper_style()
    fig, (ax_all, ax_zoom) = plt.subplots(
        1, 2, figsize=(7.6, 3.2), sharey=True, gridspec_kw={"width_ratios": [1, 1]}
    )

    trades = summary.trades
    times = np.asarray(
        [datetime.fromtimestamp(t, tz=timezone.utc) for t in trades.ts],
        dtype=object,
    )
    anchored = np.asarray(
        [summary.manifest.anchor.matches(w) for w in trades.wallet.tolist()],
        dtype=bool,
    )

    _draw_timeline(ax_all, summary, times, anchored)
    _draw_timeline(ax_zoom, summary, times, anchored)

    window = summary.manifest.window
    ax_zoom.set_xlim(
        datetime.fromtimestamp(window.start_ts - _ZOOM_PAD_S, tz=timezone.utc),
        datetime.fromtimestamp(window.end_ts + _ZOOM_PAD_S, tz=timezone.utc),
    )

    ax_all.set_ylim(0.0, 1.0)
    ax_all.set_ylabel("per-trade P(Z)")
    ax_all.set_title("Whole pulled cluster")
    ax_zoom.set_title("Analysis window +/- 7 d")
    for ax in (ax_all, ax_zoom):
        ax.set_xlabel("trade time (UTC)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_all.legend(loc="upper left", fontsize="x-small")
    fig.suptitle(
        "Van Dyke cluster: streaming P(Z); dotted rules are documented events",
        fontsize="medium",
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def save_figures(summary: CaseStudySummary, *, directory: str | Path) -> list[str]:
    """Render and save the case-study figure under ``directory``.

    Args:
        summary: Summary produced by `run_case_study`.
        directory: Destination, typically
            ``results/case_studies/van_dyke/figures``.

    Returns:
        The paths written, as strings for the summary JSON.
    """
    import matplotlib.pyplot as plt

    from src.analysis.plots import save_paper_figure

    fig = figure_case_study(summary)
    paths = save_paper_figure(fig, "van_dyke_case_study", directory=directory)
    plt.close(fig)
    return [str(p) for p in paths]


# ---------------- Replayable capture ----------------


def raw_record(trade: Any) -> dict[str, Any]:
    """Convert one `src.data.polymarket_api.RawTrade` to a stream sink record.

    `scripts.pull_data` writes the *batch* processed format, whose wallet
    column is an integer index into a `WalletIndex` — which the streaming
    scorer cannot replay, and which a wallet-anchored study cannot match a
    redacted address against. This is the same market history in the shape
    `scripts.stream_trades` writes and `score_stream.py --replay` reads.

    Args:
        trade: One `RawTrade`.

    Returns:
        A JSON-serializable record with the six fields the replay path uses.
    """
    return {
        "timestamp": float(trade.timestamp),
        "price": float(trade.price),
        "size": float(trade.size),
        "wallet": str(trade.wallet or ""),
        "transaction_hash": str(trade.transaction_hash),
        "condition_id": str(trade.condition_id),
    }


def write_capture(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write raw trade records to a replayable JSONL capture.

    Records are written in ``(timestamp, transaction_hash)`` order — the same
    total order `src.data.trade_stream.read_replay` imposes — so the capture on
    disk is already the order the scorer will consume, and two captures of the
    same history are byte-identical.

    Args:
        records: Raw records, as `raw_record` produces them.
        path: Destination JSONL; parent directories are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        records,
        key=lambda r: (float(r["timestamp"]), str(r.get("transaction_hash", ""))),
    )
    with path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return path
