"""Polymarket API clients: Gamma (market metadata) and Data API (trades).

Two thin HTTP wrappers; everything downstream consumes the dataclasses defined
here so the rest of the codebase doesn't depend on requests or the specific
JSON schema. Field-name lookups are tolerant of the common variants Polymarket
ships under (`conditionId` vs `condition_id`, `proxyWallet` vs `wallet`, etc.).

The two endpoints used:

* Gamma API:  https://gamma-api.polymarket.com/markets
  Returns market metadata. Filter by tag (e.g. "Politics"), minimum volume,
  and resolution status per the §8.2 selection criteria.

* Data API:   https://data-api.polymarket.com/trades
  Returns trade history for one market keyed by `conditionId`. Paginated;
  we walk forward until we get an empty page.

Per §7 of the README, no module-level random state. HTTP retries are bounded
and use exponential backoff on 429/5xx; everything else surfaces as a
`PolymarketAPIError`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

DEFAULT_TIMEOUT = 30.0  # seconds per HTTP call
DEFAULT_MAX_RETRIES = 4  # for 429 / 5xx
DEFAULT_BACKOFF_BASE = 1.0  # seconds; doubles each retry

# Politics-genre keyword bag for the post-hoc `question_keywords` filter.
# Gamma's `tag_slug` query is silently ignored on the /markets endpoint and the
# `tags` field is empty in the response, so we filter on question text instead.
# Match is case-insensitive substring; one hit retains the market.
POLITICS_KEYWORDS = (
    "election",
    "president",
    "presidential",
    "senate",
    "congress",
    "house of representatives",
    "governor",
    "primary",
    "trump",
    "biden",
    "harris",
    "vance",
    "walz",
    "republican",
    "democrat",
    "gop",
    "dnc",
    "supreme court",
    "impeach",
    "vote",
)


class PolymarketAPIError(RuntimeError):
    """Surface non-retryable HTTP failures and schema mismatches."""


# ---------------- Dataclasses ----------------


@dataclass
class MarketMeta:
    """Subset of Gamma API market metadata we use downstream.

    `condition_id` is the 0x-prefixed identifier used to query trades on the
    Data API; `id` is Gamma's internal market id.
    """

    id: str
    condition_id: str
    slug: str
    question: str
    volume: float
    closed: bool
    end_date: str | None
    tags: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MarketMeta:
        return cls(
            id=str(d.get("id", "")),
            condition_id=str(d.get("conditionId") or d.get("condition_id") or ""),
            slug=str(d.get("slug", "")),
            question=str(d.get("question", "")),
            volume=float(d.get("volume") or 0.0),
            closed=bool(d.get("closed", False)),
            end_date=d.get("endDate") or d.get("end_date"),
            tags=_extract_tags(d.get("tags")),
            raw=d,
        )


@dataclass
class RawTrade:
    """One trade from data-api.polymarket.com/trades.

    `wallet` is the taker's proxy wallet (0x-prefixed Polygon address); this
    is what the model's hierarchical θ_w prior keys on (§3.2).
    """

    timestamp: int  # unix seconds
    price: float  # in (0, 1)
    size: float  # USDC notional
    wallet: str
    side: str  # 'BUY' or 'SELL'
    transaction_hash: str
    condition_id: str
    asset_id: str  # YES/NO token id

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RawTrade:
        return cls(
            timestamp=int(float(d.get("timestamp") or 0)),
            price=float(d.get("price") or 0.0),
            size=float(d.get("size") or 0.0),
            wallet=str(d.get("proxyWallet") or d.get("wallet") or ""),
            side=str(d.get("side") or "").upper(),
            transaction_hash=str(
                d.get("transactionHash") or d.get("transaction_hash") or ""
            ),
            condition_id=str(d.get("conditionId") or d.get("condition_id") or ""),
            asset_id=str(d.get("asset") or d.get("asset_id") or ""),
        )


def _extract_tags(tag_field: Any) -> list[str]:
    """Parse Gamma's tags field: plain strings or label dicts are both accepted.

    Handles both the ``['Politics', ...]`` and ``[{'label': 'Politics'}, ...]``
    variants that the API has shipped at different points in time.
    """
    if not tag_field:
        return []
    out: list[str] = []
    for t in tag_field:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, dict):
            label = t.get("label") or t.get("name") or t.get("slug")
            if label:
                out.append(str(label))
    return out


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
                raise PolymarketAPIError(f"{url}: {e}") from e
            time.sleep(backoff_base * (2**attempt))
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            time.sleep(backoff_base * (2**attempt))
            continue

        raise PolymarketAPIError(
            f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    raise PolymarketAPIError(f"{url}: retries exhausted ({last_err})")


# ---------------- Gamma: markets ----------------


def fetch_markets(
    *,
    tag: str | None = "politics",
    min_volume: float = 50_000.0,
    closed: bool | None = True,
    limit: int = 100,
    offset: int = 0,
    order: str | None = "volumeNum",
    ascending: bool = False,
    question_keywords: list[str] | tuple[str, ...] | None = None,
    session: requests.Session | None = None,
) -> list[MarketMeta]:
    """List Polymarket markets matching the §8.2 filters.

    Three empirically-observed Gamma API quirks shape this signature:
      * `tag_slug` is accepted but silently ignored — Gamma returns markets
        from every genre. Use `question_keywords` for a reliable topic filter.
      * `order=volume` is silently ignored too; Gamma's numeric volume column
        is `volumeNum`, which is what JS-style sort accepts. Without it,
        /markets returns oldest-id first.
      * `volume_num_min=X` server-side filter restricts to volume >= X
        directly, which is the most reliable way to skip dust markets.

    Args:
        tag: Gamma tag slug (kept for forward-compat; currently ignored
            server-side — pass `question_keywords` to actually filter genre).
        min_volume: minimum cumulative USDC volume. Sent server-side via
            `volume_num_min` AND enforced post-fetch (belt-and-suspenders).
        closed: True → resolved/closed only, False → live only, None → either.
        limit: Gamma `limit` parameter (single page).
        offset: Gamma `offset` parameter.
        order: Gamma sort field (e.g. "volumeNum", "endDate"). None disables.
        ascending: sort direction; False = descending (highest volume first).
        question_keywords: if set, retain markets whose question contains
            (case-insensitive substring) at least one of these terms. Use
            `POLITICS_KEYWORDS` for the §8.2 genre filter.
        session: optional requests.Session for tests / connection pooling.

    Returns:
        list[MarketMeta] sorted by volume descending (final post-fetch sort).
    """
    params: dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
    if tag is not None:
        params["tag_slug"] = tag
    if closed is not None:
        params["closed"] = "true" if closed else "false"
    if order is not None:
        params["order"] = order
        params["ascending"] = "true" if ascending else "false"
    if min_volume > 0:
        params["volume_num_min"] = float(min_volume)

    payload = _get_json(f"{GAMMA_BASE}/markets", params=params, session=session)
    if not isinstance(payload, list):
        raise PolymarketAPIError(
            f"Expected list from /markets, got {type(payload).__name__}"
        )

    markets = [MarketMeta.from_dict(d) for d in payload]
    markets = [m for m in markets if m.volume >= min_volume and m.condition_id]

    if question_keywords:
        kws = tuple(k.lower() for k in question_keywords)
        markets = [m for m in markets if _question_matches(m.question, kws)]

    markets.sort(key=lambda m: m.volume, reverse=True)
    return markets


def _question_matches(question: str, keywords_lower: tuple[str, ...]) -> bool:
    """Return True if question contains at least one keyword (pre-lowercased)."""
    q = question.lower()
    return any(k in q for k in keywords_lower)


def fetch_market_by_slug(
    slug: str,
    *,
    closed_first: bool = True,
    scan_limit: int = 2000,
    page_size: int = 500,
    session: requests.Session | None = None,
) -> MarketMeta:
    """Resolve a single market by its slug.

    Gamma's `/markets?slug=X` filter is silently ignored (returns []), so we
    use a two-step lookup:

      1. `/events?slug=X` fast path: newer markets share a slug with their
         event, so this is one call. For multi-market events (e.g.,
         "presidential-election-winner-2024" → Trump, Harris, ...) we look
         inside `event.markets` for an exact slug match.
      2. Paginated `/markets?order=volumeNum&closed={true|false}` scan with
         post-hoc slug match. The §5 shortlist all lives in the top ~100
         closed markets by volume, so this is at most ~1 page each in
         practice.

    Args:
        slug: market slug to resolve.
        closed_first: try closed=true before closed=false in the scan. Our
            shortlist is all resolved, so this is the right default.
        scan_limit: hard cap on markets scanned per closed-state (4 pages of
            500 = top 2000 by volume).
        page_size: Gamma `limit` per call.
        session: optional requests.Session for tests / connection pooling.

    Returns:
        MarketMeta for the first market whose slug matches exactly.

    Raises:
        PolymarketAPIError: If no match is found after exhausting both paths.
    """
    # 1) /events fast path
    payload = _get_json(
        f"{GAMMA_BASE}/events",
        params={"slug": slug, "limit": 1},
        session=session,
    )
    if isinstance(payload, list) and payload:
        markets = payload[0].get("markets") or []
        for m in markets:
            if m.get("slug") == slug:
                return MarketMeta.from_dict(m)
        if len(markets) == 1:
            return MarketMeta.from_dict(markets[0])

    # 2) Paginated /markets scan with post-hoc slug match
    closed_order = ("true", "false") if closed_first else ("false", "true")
    for closed_param in closed_order:
        for offset in range(0, scan_limit, page_size):
            page = _get_json(
                f"{GAMMA_BASE}/markets",
                params={
                    "limit": page_size,
                    "offset": offset,
                    "order": "volumeNum",
                    "ascending": "false",
                    "closed": closed_param,
                },
                session=session,
            )
            if not isinstance(page, list) or not page:
                break
            for raw in page:
                if raw.get("slug") == slug:
                    return MarketMeta.from_dict(raw)
            if len(page) < page_size:
                break

    raise PolymarketAPIError(f"No market found for slug={slug!r}")


# ---------------- Data API: trades ----------------

DATA_API_MAX_OFFSET = 3000  # Polymarket /trades cap on historical pagination


def fetch_trades(
    condition_id: str,
    *,
    page_size: int = 500,
    max_pages: int = 200,
    max_offset: int = DATA_API_MAX_OFFSET,
    sleep_between: float = 0.1,
    session: requests.Session | None = None,
) -> list[RawTrade]:
    """Pull trades for one market via the paginated /trades endpoint.

    Polymarket returns trades newest-first and rejects any offset that would
    exceed `max_historical_activity_offset` (currently 3000) with HTTP 400.
    Because of the newest-first ordering, the resulting window IS the last
    ~3000 trades — i.e., the final price-action period leading into
    resolution, which is exactly the §8.2 budget for the insider-detection
    application. We stop paginating at `max_offset` to avoid the error
    entirely.

    Args:
        condition_id: 0x-prefixed market condition id (from `MarketMeta`).
        page_size: trades per page; Polymarket caps at 500.
        max_pages: client-side safety cap on number of pages.
        max_offset: maximum starting offset to request (default 3000,
            Polymarket's documented limit). The function stops cleanly when
            the next page would exceed this.
        sleep_between: seconds between paged calls; cheap politeness.
        session: optional requests.Session.

    Returns:
        list[RawTrade] in the order the API returned them (newest-first
        within each page; not yet globally sorted).
    """
    if not condition_id:
        raise ValueError("condition_id must be non-empty")

    trades: list[RawTrade] = []
    for page in range(max_pages):
        offset = page * page_size
        if offset >= max_offset:
            break
        # Trim the final page so its end stays within max_offset.
        this_limit = min(int(page_size), int(max_offset - offset))
        params = {
            "market": condition_id,
            "limit": this_limit,
            "offset": int(offset),
        }
        payload = _get_json(f"{DATA_BASE}/trades", params=params, session=session)
        if not isinstance(payload, list):
            raise PolymarketAPIError(
                f"Expected list from /trades, got {type(payload).__name__}"
            )
        if not payload:
            break
        trades.extend(RawTrade.from_dict(d) for d in payload)
        if len(payload) < this_limit:
            break
        if sleep_between > 0:
            time.sleep(sleep_between)
    return trades
