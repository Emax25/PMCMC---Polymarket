"""Kalshi public market/trade client: `GetTrades` normalized onto `RawTrade`.

Kalshi is the project's second venue and the first *anonymous* one. Everything
downstream consumes the same `RawTrade`/`MarketMeta`-shaped objects the
Polymarket client emits, so no inference code learns that a second exchange
exists.

Endpoints used (both public, no API key, no signing):

* ``GET {KALSHI_BASE}/markets/trades`` — trade history, newest-first.
  Query params: ``ticker`` (one market), ``limit`` (1..1000; 1001 → HTTP 400),
  ``min_ts``/``max_ts`` (UNIX **seconds**), and ``cursor``.
* ``GET {KALSHI_BASE}/markets/{ticker}`` — market metadata (title, status,
  ``close_time``, ``volume_fp``). HTTP 404 for an unknown ticker.

Properties probed against the live API on 2026-08-01 (ticker
``KXZELENSKYYOUT-26JUL01``, a settled politics market):

* **Pagination is cursor-based, not offset-based.** The response is
  ``{"cursor": "...", "trades": [...]}``; the next page is requested by
  echoing that cursor back as ``cursor``. An **empty-string** cursor means the
  walk is exhausted, so unlike Polymarket's `/trades` there is no offset cap
  and no timestamp-window re-anchoring — one linear walk reaches the first
  trade of a market.
* **Prices and sizes ship as decimal strings, not integers.** Live rows carry
  ``yes_price_dollars``/``no_price_dollars`` (dollars, e.g. ``"0.0040"``) and
  ``count_fp`` (fractional contracts, e.g. ``"43.33"``). The older integer
  ``yes_price``/``no_price`` (**cents**) and ``count`` fields are accepted as
  fallbacks so a rollback of Kalshi's field naming does not break the parser.
  The two legs sum to par (``yes + no == 1``), so the YES leg alone fixes the
  probability regardless of which side the taker lifted.
* **Timestamps are RFC-3339 strings with sub-second precision**
  (``created_time``: ``"2026-06-30T17:10:57.852752Z"``), not UNIX integers.
  `RawTrade.timestamp` is integer seconds, so the fraction is truncated —
  Kalshi therefore looks second-resolution downstream, exactly like
  Polymarket, and same-second ties are broken by trade id when sorting.
* **``min_ts``/``max_ts`` compare against the un-truncated trade time.**
  ``min_ts`` is an inclusive lower bound on seconds, but ``max_ts=T`` *drops*
  any trade inside second ``T`` that carries a fraction — pass a ceiling, not
  a floor, for an upper bound.
* **No identity.** A trade row is
  ``{trade_id, ticker, created_time, count_fp, yes_price_dollars,
  no_price_dollars, taker_side, taker_book_side, taker_outcome_side,
  is_block_trade}`` — there is no account, wallet, or user field, and no
  authenticated endpoint exposes one for other people's fills. This is a
  property of the venue, not of this client: the wallet-anchored θ_w prior
  (§3.2) cannot be estimated from Kalshi data at all, which is why every
  normalized row here carries ``wallet=None`` and why that nullability is the
  signal downstream code keys anonymous mode off.
* **Fee model** (for consumers sizing a live strategy): Kalshi charges
  ``ceil(0.07 · C · p · (1 - p))`` dollars on a taker order of ``C``
  contracts at price ``p``, i.e. ~1.75¢/contract at p = 0.5 and ~0.07¢ at
  p = 0.01. Fees are maximal exactly where this model's insider signal is
  weakest (mid-book) and cheapest in the tails.

Retries mirror `src/data/polymarket_api.py`: bounded exponential backoff on
429/5xx, everything else surfacing as `KalshiAPIError`. The backoff helper is
deliberately duplicated rather than shared, so each venue keeps its own error
type and retry policy and neither module reaches into the other's privates.
Per §7 of the README, no module-level random state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from src.data.polymarket_api import RawTrade

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

DEFAULT_TIMEOUT = 30.0  # seconds per HTTP call
DEFAULT_MAX_RETRIES = 4  # for 429 / 5xx
DEFAULT_BACKOFF_BASE = 1.0  # seconds; doubles each retry

# Server-enforced ceiling on `limit`; 1001 returns HTTP 400.
MAX_PAGE_SIZE = 1000

# Client-side row budget for the default (tail) pull, matching the spirit of
# Polymarket's `DATA_API_MAX_OFFSET`: the newest ~3000 trades are the §8.2
# window. Kalshi imposes no server-side cap, so `--full-history` simply lifts
# this budget rather than switching to a different retrieval strategy.
DEFAULT_MAX_TRADES = 3000

# Kalshi taker sides mapped onto the repo's Polymarket-derived sign convention.
# `taker_side` names the *outcome* the aggressor bought; buying NO is selling
# YES, and every price in this module is quoted on the YES leg.
_TAKER_SIDE_TO_REPO_SIDE = {"yes": "BUY", "no": "SELL"}

# Statuses that mean "no longer trading". Kalshi walks
# active → closed → determined → settled/finalized.
_RESOLVED_STATUSES = frozenset({"closed", "determined", "settled", "finalized"})


class KalshiAPIError(RuntimeError):
    """Surface non-retryable HTTP failures and schema mismatches."""


# ---------------- Dataclasses ----------------


@dataclass
class KalshiMarketMeta:
    """Subset of Kalshi market metadata we use downstream.

    Deliberately *not* `polymarket_api.MarketMeta`: Kalshi has no condition id
    and no slug, and reusing a dataclass whose fields are documented in
    Polymarket's vocabulary would make the ticker masquerade as three
    different identifiers. `ticker` is the one and only market id here.
    """

    ticker: str
    title: str
    status: str
    close_time: str | None  # RFC-3339, e.g. "2026-07-01T03:59:00Z"
    volume: float  # contracts traded over the market's life
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def resolved(self) -> bool:
        """True once the market has stopped trading (closed through settled)."""
        return self.status.strip().lower() in _RESOLVED_STATUSES

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KalshiMarketMeta:
        """Build metadata from one ``/markets/{ticker}`` ``market`` object.

        Args:
            d: The ``market`` sub-object of the endpoint's JSON response.

        Returns:
            KalshiMarketMeta with missing numeric fields defaulted to 0.
        """
        volume = d.get("volume_fp")
        if volume is None:
            volume = d.get("volume")
        return cls(
            ticker=str(d.get("ticker") or ""),
            title=str(d.get("title") or ""),
            status=str(d.get("status") or ""),
            close_time=d.get("close_time") or d.get("expiration_time"),
            volume=float(volume or 0.0),
            raw=d,
        )


# ---------------- HTTP plumbing ----------------


def _get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    session: requests.Session | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
) -> Any:
    """GET with bounded exponential backoff on 429 / 5xx; raises otherwise."""
    sess = session or requests
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = sess.get(url, params=params, timeout=timeout)
        except requests.RequestException as e:
            last_err = e
            if attempt == max_retries:
                raise KalshiAPIError(f"{url}: {e}") from e
            time.sleep(backoff_base * (2**attempt))
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            time.sleep(backoff_base * (2**attempt))
            continue

        raise KalshiAPIError(
            f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    raise KalshiAPIError(f"{url}: retries exhausted ({last_err})")


# ---------------- Normalization ----------------


def _coerce_float(value: Any, field_name: str, trade_id: str) -> float:
    """Coerce one Kalshi numeric field (they ship as strings) to float."""
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise KalshiAPIError(
            f"trade {trade_id!r}: uncoercible {field_name}={value!r}"
        ) from e


def _yes_price(row: dict[str, Any], trade_id: str) -> float:
    """Return the trade's YES-leg price as a probability in (0, 1).

    Tries, in order: ``yes_price_dollars`` (dollars, the current live field),
    ``yes_price`` (cents, the legacy integer field), then the NO leg via
    ``yes = 1 - no`` — the two legs sum to par, so either one determines the
    probability whichever side the taker lifted.

    Args:
        row: One raw trade object from the API.
        trade_id: Trade id, used only to make errors identifiable.

    Returns:
        YES-leg price on the (0, 1) probability scale.

    Raises:
        KalshiAPIError: If the row carries no usable price field, or the one
            it carries does not parse as a number.
    """
    if row.get("yes_price_dollars") is not None:
        return _coerce_float(row["yes_price_dollars"], "yes_price_dollars", trade_id)
    if row.get("yes_price") is not None:
        return _coerce_float(row["yes_price"], "yes_price", trade_id) / 100.0
    if row.get("no_price_dollars") is not None:
        no = _coerce_float(row["no_price_dollars"], "no_price_dollars", trade_id)
        return 1.0 - no
    if row.get("no_price") is not None:
        return 1.0 - _coerce_float(row["no_price"], "no_price", trade_id) / 100.0
    raise KalshiAPIError(f"trade {trade_id!r}: no yes/no price field present")


def _contract_count(row: dict[str, Any], trade_id: str) -> float:
    """Return the trade size in contracts (``count_fp``, else legacy ``count``)."""
    if row.get("count_fp") is not None:
        return _coerce_float(row["count_fp"], "count_fp", trade_id)
    if row.get("count") is not None:
        return _coerce_float(row["count"], "count", trade_id)
    raise KalshiAPIError(f"trade {trade_id!r}: no count field present")


def _created_time_to_unix(value: Any, trade_id: str) -> int:
    """Parse Kalshi's RFC-3339 ``created_time`` to truncated UNIX seconds.

    The sub-second fraction Kalshi publishes is dropped because
    `RawTrade.timestamp` is integer seconds across the whole pipeline. Ordering
    inside a second is therefore *not* recoverable from the timestamp alone,
    which is why the cleaning step breaks ties on trade id.

    Args:
        value: The row's ``created_time``, e.g. ``"2026-06-30T17:10:57.85Z"``.
        trade_id: Trade id, used only to make errors identifiable.

    Returns:
        UNIX seconds (UTC), truncated toward the start of the second.

    Raises:
        KalshiAPIError: If the field is missing, empty, or not RFC-3339.
    """
    text = str(value or "").strip()
    if not text:
        raise KalshiAPIError(f"trade {trade_id!r}: missing created_time")
    try:
        # `fromisoformat` handles arbitrary fractional-second digits on 3.11+
        # but still refuses a bare "Z" suffix, so normalize the zone first.
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as e:
        raise KalshiAPIError(
            f"trade {trade_id!r}: unparseable created_time={text!r}"
        ) from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # Kalshi always sends UTC
    return int(dt.timestamp())


def raw_trade_from_kalshi(row: dict[str, Any]) -> RawTrade:
    """Normalize one Kalshi trade row onto the repo's `RawTrade` schema.

    Field mapping, with the reasoning for the non-obvious ones:

    * ``trade_id`` → ``transaction_hash``. Not a hash, but it plays the same
      role: the venue's unique fill id, used as the dedupe key across pages
      and the deterministic same-second sort tiebreaker.
    * ``ticker`` → ``condition_id``. The ticker is Kalshi's only market id.
    * YES-leg price → ``price``; ``count``/``count_fp`` → ``size`` (contracts,
      not dollars — one contract pays $1, so notional is ``size * price``).
    * ``taker_side`` → ``side``: ``yes`` → ``BUY``, ``no`` → ``SELL``, since
      buying NO is selling YES and prices here are quoted on the YES leg.
    * ``wallet`` → ``None``. Kalshi publishes no account identifier; see the
      module docstring. Never substitute a placeholder string — downstream
      anonymous mode keys off exactly this nullability.
    * ``asset_id`` → ``""``. Kalshi has one binary contract per ticker rather
      than separate YES/NO token ids, so there is no analog to fill in.

    Args:
        row: One object from the ``trades`` array of a `GetTrades` response.

    Returns:
        The normalized `RawTrade`, with ``wallet is None``.

    Raises:
        KalshiAPIError: If ``row`` is not an object, is missing ``trade_id``,
            ``ticker``, a price, a count, or ``created_time``, or carries a
            ``taker_side`` other than ``yes``/``no``.
    """
    if not isinstance(row, dict):
        raise KalshiAPIError(f"trade row is {type(row).__name__}, expected object")

    trade_id = str(row.get("trade_id") or "")
    if not trade_id:
        raise KalshiAPIError(f"trade row is missing trade_id: {str(row)[:200]}")
    ticker = str(row.get("ticker") or "")
    if not ticker:
        raise KalshiAPIError(f"trade {trade_id!r}: missing ticker")

    taker_side = str(row.get("taker_side") or "").strip().lower()
    side = _TAKER_SIDE_TO_REPO_SIDE.get(taker_side)
    if side is None:
        raise KalshiAPIError(
            f"trade {trade_id!r}: unknown taker_side {row.get('taker_side')!r}"
        )

    return RawTrade(
        timestamp=_created_time_to_unix(row.get("created_time"), trade_id),
        price=_yes_price(row, trade_id),
        size=_contract_count(row, trade_id),
        wallet=None,
        side=side,
        transaction_hash=trade_id,
        condition_id=ticker,
        asset_id="",
    )


def _trades_array(payload: Any) -> list[Any]:
    """Extract the ``trades`` array from a `GetTrades` response.

    Args:
        payload: Decoded JSON body of one `GetTrades` call.

    Returns:
        The ``trades`` list, empty when the key is absent (which is how the
        API reports an unknown ticker or an empty time window — it answers
        HTTP 200 with ``{"cursor": "", "trades": []}`` rather than a 404).

    Raises:
        KalshiAPIError: If the payload is not an object, or ``trades`` is
            present but is not a list.
    """
    if not isinstance(payload, dict):
        raise KalshiAPIError(
            f"Expected an object from /markets/trades, got {type(payload).__name__}"
        )
    rows = payload.get("trades")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise KalshiAPIError(
            f"Expected a list under 'trades', got {type(rows).__name__}"
        )
    return rows


# ---------------- GetTrades ----------------


def fetch_market(
    ticker: str,
    *,
    session: requests.Session | None = None,
) -> KalshiMarketMeta:
    """Fetch one market's metadata from ``/markets/{ticker}``.

    Args:
        ticker: Kalshi market ticker, e.g. ``KXZELENSKYYOUT-26JUL01``.
        session: Optional requests.Session for tests / connection pooling.

    Returns:
        KalshiMarketMeta for the ticker.

    Raises:
        ValueError: If ``ticker`` is empty.
        KalshiAPIError: On HTTP failure (404 for an unknown ticker) or if the
            response does not carry a ``market`` object.
    """
    if not ticker:
        raise ValueError("ticker must be non-empty")

    payload = _get_json(f"{KALSHI_BASE}/markets/{ticker}", session=session)
    market = payload.get("market") if isinstance(payload, dict) else None
    if not isinstance(market, dict):
        raise KalshiAPIError(
            f"/markets/{ticker}: expected a 'market' object, got "
            f"{type(market).__name__}"
        )
    return KalshiMarketMeta.from_dict(market)


def fetch_trades(
    ticker: str,
    *,
    min_ts: int | None = None,
    max_ts: int | None = None,
    max_trades: int | None = DEFAULT_MAX_TRADES,
    page_size: int = MAX_PAGE_SIZE,
    max_pages: int = 20_000,
    sleep_between: float = 0.1,
    session: requests.Session | None = None,
) -> list[RawTrade]:
    """Walk `GetTrades` for one market, newest-first, until the cursor empties.

    Cursor pagination has no offset ceiling, so a single linear walk reaches a
    market's first trade — there is no Polymarket-style timestamp-window
    re-anchoring here. Rows are deduplicated on ``trade_id`` anyway: the walk
    is not atomic, and a page boundary that moves under a concurrent fill can
    hand back a row twice.

    Rate discipline: one sleep of ``sleep_between`` between pages, matching
    `polymarket_api.fetch_trades`.

    Args:
        ticker: Kalshi market ticker.
        min_ts: Inclusive lower bound on trade time (UNIX seconds); None omits.
        max_ts: Upper bound on trade time (UNIX seconds); None omits. Compared
            against the un-truncated trade time, so pass a *ceiling* — a floor
            silently drops the trades inside that second.
        max_trades: Client-side row budget; the walk returns as soon as this
            many distinct trades are collected. None pulls the full history.
        page_size: Rows per call; Kalshi caps at `MAX_PAGE_SIZE`.
        max_pages: Safety budget on total HTTP GETs.
        sleep_between: Seconds slept between paged calls.
        session: Optional requests.Session.

    Returns:
        list[RawTrade] in API order (newest-first), deduplicated on trade id,
        every row carrying ``wallet is None``.

    Raises:
        ValueError: If ``ticker`` is empty or ``page_size`` is out of range.
        KalshiAPIError: On HTTP failure, a malformed payload, a cursor the
            server repeats (which would loop forever), or an exhausted
            ``max_pages`` budget.
    """
    if not ticker:
        raise ValueError("ticker must be non-empty")
    if not 0 < page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{MAX_PAGE_SIZE}, got {page_size}")

    seen: set[str] = set()
    out: list[RawTrade] = []
    cursor: str = ""

    for _ in range(max_pages):
        params: dict[str, Any] = {"ticker": ticker, "limit": int(page_size)}
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        if max_ts is not None:
            params["max_ts"] = int(max_ts)
        if cursor:
            params["cursor"] = cursor

        payload = _get_json(f"{KALSHI_BASE}/markets/trades", params, session=session)
        rows = _trades_array(payload)

        for row in rows:
            trade = raw_trade_from_kalshi(row)
            if trade.transaction_hash in seen:
                continue
            seen.add(trade.transaction_hash)
            out.append(trade)
            if max_trades is not None and len(out) >= max_trades:
                return out

        next_cursor = str(payload.get("cursor") or "")
        if not rows or not next_cursor:
            return out  # exhausted: empty page or empty cursor
        if next_cursor == cursor:
            raise KalshiAPIError(
                f"/markets/trades: {ticker} repeated cursor {cursor!r}; "
                "the walk cannot advance"
            )
        cursor = next_cursor
        if sleep_between > 0:
            time.sleep(sleep_between)

    raise KalshiAPIError(
        f"/markets/trades: max_pages={max_pages} exhausted for {ticker}; "
        "history is incomplete"
    )
