"""Offline tests for src/data/kalshi_api.py.

All HTTP is mocked via `unittest.mock` so the suite never hits Kalshi. The
fake server below reproduces the response shape probed against the live API on
2026-08-01 (decimal-string prices, RFC-3339 `created_time`, cursor pagination
ending in an empty-string cursor).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.kalshi_api import (
    KALSHI_BASE,
    MAX_PAGE_SIZE,
    KalshiAPIError,
    KalshiMarketMeta,
    fetch_market,
    fetch_trades,
    raw_trade_from_kalshi,
)


def _mock_response(json_payload, status_code: int = 200):
    """Build a minimal requests.Response mock with given payload and status."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_payload
    resp.text = json.dumps(json_payload)
    return resp


def _row(
    i: int,
    *,
    ticker: str = "KXTEST-26JUL01",
    yes_price: str = "0.4600",
    taker_side: str = "yes",
    created: str | None = None,
) -> dict:
    """One live-shaped Kalshi trade row.

    ``no_price_dollars`` is derived so the two legs sum to par, as they do
    live; a deliberately unparseable ``yes_price`` leaves the NO leg blank so
    the parser has no second field to fall back on.
    """
    try:
        no_price = f"{1.0 - float(yes_price):.4f}"
    except ValueError:
        no_price = yes_price
    return {
        "trade_id": f"trade-{i:04d}",
        "ticker": ticker,
        "created_time": created or f"2026-06-30T17:10:{i % 60:02d}.852752Z",
        "count_fp": "43.33",
        "yes_price_dollars": yes_price,
        "no_price_dollars": no_price,
        "taker_side": taker_side,
        "taker_book_side": "bid",
        "taker_outcome_side": taker_side,
        "is_block_trade": False,
    }


class _FakeTradesAPI:
    """In-memory GetTrades stand-in reproducing the probed server semantics.

    Cursor pagination: each response echoes the index of the next unread row as
    an opaque cursor, and the final page carries an empty-string cursor — which
    is exactly how the live API signals exhaustion.
    """

    def __init__(
        self,
        rows: list[dict],
        *,
        fail_calls: tuple[int, ...] = (),
        duplicate_first_row: bool = False,
    ):
        """Serve ``rows`` newest-first over cursor-paginated responses.

        Args:
            rows: Trade rows in the order the server would return them.
            fail_calls: 1-based call indices answered with HTTP 429 instead.
            duplicate_first_row: Repeat page N's first row at the head of page
                N+1, mimicking a page boundary shifting under a live fill.
        """
        self.rows = rows
        self.fail_calls = set(fail_calls)
        self.duplicate_first_row = duplicate_first_row
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        """Serve one paged GetTrades request."""
        params = dict(params or {})
        self.calls.append(params)
        if len(self.calls) in self.fail_calls:
            return _mock_response({"error": "rate limited"}, status_code=429)

        start = int(params.get("cursor") or 0)
        limit = int(params["limit"])
        page = self.rows[start : start + limit]
        if self.duplicate_first_row and start > 0:
            page = [self.rows[start - limit]] + page
        nxt = start + limit
        cursor = str(nxt) if nxt < len(self.rows) else ""
        return _mock_response({"cursor": cursor, "trades": page})


def _install(monkeypatch, api: _FakeTradesAPI) -> None:
    """Route requests.get at the fake API and disable real sleeping."""
    monkeypatch.setattr("src.data.kalshi_api.requests.get", api.get)
    monkeypatch.setattr("src.data.kalshi_api.time.sleep", lambda _: None)


# ---------------- Normalization ----------------


def test_raw_trade_from_kalshi_maps_live_row():
    """A live-shaped row maps onto RawTrade field by field."""
    t = raw_trade_from_kalshi(_row(1, yes_price="0.4600"))
    assert t.price == pytest.approx(0.46)
    assert t.size == pytest.approx(43.33)
    assert t.side == "BUY"
    assert t.transaction_hash == "trade-0001"  # trade_id is the dedupe key
    assert t.condition_id == "KXTEST-26JUL01"  # ticker is the market id
    assert t.asset_id == ""  # Kalshi has no separate YES/NO token id
    assert t.timestamp == 1782839401  # 2026-06-30T17:10:01Z, fraction dropped


@pytest.mark.parametrize(
    ("cents", "expected"),
    [(1, 0.01), (50, 0.50), (99, 0.99)],
)
def test_legacy_cents_price_converts_at_boundaries(cents, expected):
    """The legacy integer `yes_price` field is read as cents, not dollars."""
    row = _row(1)
    del row["yes_price_dollars"]
    del row["no_price_dollars"]
    row["yes_price"] = cents
    assert raw_trade_from_kalshi(row).price == pytest.approx(expected)


@pytest.mark.parametrize(
    ("yes_dollars", "expected"),
    [("0.0100", 0.01), ("0.9900", 0.99)],
)
def test_dollar_price_at_boundaries(yes_dollars, expected):
    """Decimal-string dollar prices land on the same (0, 1) scale."""
    assert raw_trade_from_kalshi(_row(1, yes_price=yes_dollars)).price == pytest.approx(
        expected
    )


def test_price_falls_back_to_the_no_leg():
    """With only the NO leg present, price is recovered as 1 - no."""
    row = _row(1)
    del row["yes_price_dollars"]
    row["no_price_dollars"] = "0.9960"
    assert raw_trade_from_kalshi(row).price == pytest.approx(0.004)


def test_taker_side_maps_onto_the_repo_sign_convention():
    """taker_side yes/no becomes BUY/SELL — buying NO is selling YES."""
    assert raw_trade_from_kalshi(_row(1, taker_side="yes")).side == "BUY"
    assert raw_trade_from_kalshi(_row(2, taker_side="no")).side == "SELL"
    assert raw_trade_from_kalshi(_row(3, taker_side="YES")).side == "BUY"


def test_unknown_taker_side_raises():
    """An unrecognised taker_side is a schema change, not a row to guess at."""
    with pytest.raises(KalshiAPIError, match="taker_side"):
        raw_trade_from_kalshi(_row(1, taker_side="buy"))


def test_created_time_truncates_to_whole_seconds():
    """Sub-second precision is dropped — RawTrade.timestamp is integer seconds."""
    a = raw_trade_from_kalshi(_row(1, created="2026-06-30T17:10:57.852752Z"))
    b = raw_trade_from_kalshi(_row(2, created="2026-06-30T17:10:57.06477Z"))
    assert a.timestamp == b.timestamp == 1782839457


# ---------------- No-identity invariant ----------------


def test_normalized_row_has_no_wallet():
    """Regression guard: Kalshi rows must never acquire an invented identity."""
    t = raw_trade_from_kalshi(_row(1))
    assert t.wallet is None
    assert t.wallet != ""  # an empty string would read as "known but blank"


def test_every_fetched_trade_has_wallet_none(monkeypatch):
    """The invariant survives pagination, not just single-row normalization."""
    api = _FakeTradesAPI([_row(i) for i in range(25)])
    _install(monkeypatch, api)
    trades = fetch_trades("KXTEST-26JUL01", page_size=10, sleep_between=0.0)
    assert len(trades) == 25
    assert all(t.wallet is None for t in trades)


# ---------------- Pagination ----------------


def test_pagination_walks_cursor_until_exhausted(monkeypatch):
    """Pages are chained by cursor and stop on the empty-string cursor."""
    api = _FakeTradesAPI([_row(i) for i in range(250)])
    _install(monkeypatch, api)
    trades = fetch_trades("KXTEST-26JUL01", page_size=100, sleep_between=0.0)
    assert [t.transaction_hash for t in trades] == [
        f"trade-{i:04d}" for i in range(250)
    ]
    assert len(api.calls) == 3
    assert "cursor" not in api.calls[0]  # first page sends no cursor
    assert api.calls[1]["cursor"] == "100"


def test_pagination_dedupes_on_trade_id(monkeypatch):
    """A row repeated across a shifting page boundary is returned once."""
    api = _FakeTradesAPI([_row(i) for i in range(30)], duplicate_first_row=True)
    _install(monkeypatch, api)
    trades = fetch_trades("KXTEST-26JUL01", page_size=10, sleep_between=0.0)
    ids = [t.transaction_hash for t in trades]
    assert len(ids) == len(set(ids)) == 30


def test_max_trades_budget_stops_the_walk_early(monkeypatch):
    """The client-side row budget returns as soon as it is met."""
    api = _FakeTradesAPI([_row(i) for i in range(250)])
    _install(monkeypatch, api)
    trades = fetch_trades(
        "KXTEST-26JUL01", page_size=100, max_trades=120, sleep_between=0.0
    )
    assert len(trades) == 120
    assert len(api.calls) == 2  # budget met mid-page-2; page 3 never requested


def test_full_history_lifts_the_budget(monkeypatch):
    """max_trades=None walks to the market's first trade."""
    api = _FakeTradesAPI([_row(i) for i in range(250)])
    _install(monkeypatch, api)
    trades = fetch_trades(
        "KXTEST-26JUL01", page_size=100, max_trades=None, sleep_between=0.0
    )
    assert len(trades) == 250


def test_timestamp_bounds_are_forwarded(monkeypatch):
    """min_ts/max_ts reach the wire as integer seconds; None omits them."""
    api = _FakeTradesAPI([_row(0)])
    _install(monkeypatch, api)
    fetch_trades("KXTEST-26JUL01", min_ts=1_700_000_000, sleep_between=0.0)
    assert api.calls[0]["min_ts"] == 1_700_000_000
    assert "max_ts" not in api.calls[0]
    assert api.calls[0]["ticker"] == "KXTEST-26JUL01"


def test_repeated_cursor_raises_instead_of_looping(monkeypatch):
    """A server that echoes the same cursor would loop forever; we raise."""

    def stuck_get(url, params=None, timeout=None):
        return _mock_response({"cursor": "same", "trades": [_row(0)]})

    monkeypatch.setattr("src.data.kalshi_api.requests.get", stuck_get)
    monkeypatch.setattr("src.data.kalshi_api.time.sleep", lambda _: None)
    with pytest.raises(KalshiAPIError, match="repeated cursor"):
        fetch_trades("KXTEST-26JUL01", max_trades=None, sleep_between=0.0)


def test_max_pages_budget_raises(monkeypatch):
    """Running out of page budget is an incomplete history, not a silent stop."""
    api = _FakeTradesAPI([_row(i) for i in range(500)])
    _install(monkeypatch, api)
    with pytest.raises(KalshiAPIError, match="max_pages"):
        fetch_trades(
            "KXTEST-26JUL01",
            page_size=10,
            max_trades=None,
            max_pages=2,
            sleep_between=0.0,
        )


def test_fetch_trades_rejects_bad_arguments():
    """Empty ticker and out-of-range page sizes fail before any HTTP call."""
    with pytest.raises(ValueError):
        fetch_trades("")
    with pytest.raises(ValueError):
        fetch_trades("KXTEST-26JUL01", page_size=0)
    with pytest.raises(ValueError):
        fetch_trades("KXTEST-26JUL01", page_size=MAX_PAGE_SIZE + 1)


# ---------------- Backoff ----------------


def test_backoff_retries_on_429_then_succeeds(monkeypatch):
    """429/5xx are retried with exponential backoff; the walk then completes."""
    api = _FakeTradesAPI([_row(i) for i in range(15)], fail_calls=(1, 3))
    sleeps: list[float] = []
    monkeypatch.setattr("src.data.kalshi_api.requests.get", api.get)
    monkeypatch.setattr("src.data.kalshi_api.time.sleep", lambda s: sleeps.append(s))

    trades = fetch_trades("KXTEST-26JUL01", page_size=10, sleep_between=0.0)
    ids = [t.transaction_hash for t in trades]
    assert len(ids) == len(set(ids)) == 15
    assert len(sleeps) == 2  # one backoff per 429
    assert all(s > 0 for s in sleeps)


def test_non_retryable_status_raises():
    """HTTP 400 (e.g. limit > 1000) surfaces as KalshiAPIError."""
    with patch("src.data.kalshi_api.requests.get") as g:
        g.return_value = _mock_response({"error": "bad limit"}, status_code=400)
        with pytest.raises(KalshiAPIError, match="HTTP 400"):
            fetch_trades("KXTEST-26JUL01")


def test_request_exception_is_retried_then_wrapped(monkeypatch):
    """A transport error is retried to the cap, then wrapped, never leaked."""
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr("src.data.kalshi_api.requests.get", boom)
    monkeypatch.setattr("src.data.kalshi_api.time.sleep", lambda _: None)
    with pytest.raises(KalshiAPIError):
        fetch_trades("KXTEST-26JUL01")
    assert calls["n"] > 1


# ---------------- Error paths ----------------


def test_empty_result_set_returns_empty_list(monkeypatch):
    """An unknown ticker answers HTTP 200 with no trades, not a 404."""
    api = _FakeTradesAPI([])
    _install(monkeypatch, api)
    assert fetch_trades("KXNOPE-NOPE", sleep_between=0.0) == []


def test_missing_trades_key_is_treated_as_empty(monkeypatch):
    """A response without a `trades` key yields no rows rather than a crash."""

    def get(url, params=None, timeout=None):
        return _mock_response({"cursor": ""})

    monkeypatch.setattr("src.data.kalshi_api.requests.get", get)
    assert fetch_trades("KXTEST-26JUL01") == []


def test_non_object_payload_raises(monkeypatch):
    """A list payload means the endpoint contract changed; fail loudly."""

    def get(url, params=None, timeout=None):
        return _mock_response([{"trade_id": "x"}])

    monkeypatch.setattr("src.data.kalshi_api.requests.get", get)
    with pytest.raises(KalshiAPIError, match="expected|Expected"):
        fetch_trades("KXTEST-26JUL01")


def test_non_list_trades_raises(monkeypatch):
    """`trades` present but not a list is a schema mismatch."""

    def get(url, params=None, timeout=None):
        return _mock_response({"cursor": "", "trades": {"trade_id": "x"}})

    monkeypatch.setattr("src.data.kalshi_api.requests.get", get)
    with pytest.raises(KalshiAPIError, match="trades"):
        fetch_trades("KXTEST-26JUL01")


@pytest.mark.parametrize(
    ("missing", "match"),
    [
        ("trade_id", "trade_id"),
        ("ticker", "ticker"),
        ("created_time", "created_time"),
    ],
)
def test_missing_required_field_raises(missing, match):
    """Dropping any identity/ordering field is fatal, never defaulted away."""
    row = _row(1)
    del row[missing]
    with pytest.raises(KalshiAPIError, match=match):
        raw_trade_from_kalshi(row)


def test_missing_price_and_count_raise():
    """A row with no price or no size cannot be normalized."""
    row = _row(1)
    for key in ("yes_price_dollars", "no_price_dollars"):
        del row[key]
    with pytest.raises(KalshiAPIError, match="price"):
        raw_trade_from_kalshi(row)

    row = _row(2)
    del row["count_fp"]
    with pytest.raises(KalshiAPIError, match="count"):
        raw_trade_from_kalshi(row)


def test_uncoercible_and_unparseable_fields_raise():
    """Non-numeric prices and non-RFC-3339 timestamps are reported by trade id."""
    with pytest.raises(KalshiAPIError, match="trade-0001"):
        raw_trade_from_kalshi(_row(1, yes_price="not-a-number"))
    with pytest.raises(KalshiAPIError, match="created_time"):
        raw_trade_from_kalshi(_row(1, created="30/06/2026"))


def test_non_object_row_raises():
    """A scalar where a trade object belongs is a schema mismatch."""
    with pytest.raises(KalshiAPIError, match="expected object"):
        raw_trade_from_kalshi("trade-0001")


# ---------------- Market metadata ----------------


def test_fetch_market_parses_live_shape(monkeypatch):
    """/markets/{ticker} maps onto KalshiMarketMeta, volume_fp included."""
    payload = {
        "market": {
            "ticker": "KXZELENSKYYOUT-26JUL01",
            "title": "Will Volodymyr Zelenskyy leave office before Jul 1, 2026?",
            "status": "finalized",
            "close_time": "2026-07-01T03:59:00Z",
            "volume_fp": "17651.57",
        }
    }
    seen: dict = {}

    def get(url, params=None, timeout=None):
        seen["url"] = url
        return _mock_response(payload)

    monkeypatch.setattr("src.data.kalshi_api.requests.get", get)
    meta = fetch_market("KXZELENSKYYOUT-26JUL01")
    assert seen["url"] == f"{KALSHI_BASE}/markets/KXZELENSKYYOUT-26JUL01"
    assert meta.ticker == "KXZELENSKYYOUT-26JUL01"
    assert meta.close_time == "2026-07-01T03:59:00Z"
    assert meta.volume == pytest.approx(17651.57)
    assert meta.resolved is True


def test_fetch_market_meta_defaults_and_active_status():
    """Missing volume defaults to 0 and an active market is not resolved."""
    meta = KalshiMarketMeta.from_dict({"ticker": "KXA-1", "status": "active"})
    assert meta.volume == 0.0
    assert meta.resolved is False
    assert meta.close_time is None


def test_fetch_market_raises_on_404(monkeypatch):
    """An unknown ticker 404s and surfaces as KalshiAPIError."""

    def get(url, params=None, timeout=None):
        return _mock_response({"error": {"code": "not_found"}}, status_code=404)

    monkeypatch.setattr("src.data.kalshi_api.requests.get", get)
    with pytest.raises(KalshiAPIError, match="HTTP 404"):
        fetch_market("KXNOPE-NOPE")


def test_fetch_market_raises_without_market_object(monkeypatch):
    """A response missing the `market` object is a schema mismatch."""

    def get(url, params=None, timeout=None):
        return _mock_response({"markets": []})

    monkeypatch.setattr("src.data.kalshi_api.requests.get", get)
    with pytest.raises(KalshiAPIError, match="market"):
        fetch_market("KXTEST-26JUL01")


def test_fetch_market_rejects_empty_ticker():
    """An empty ticker fails before any HTTP call."""
    with pytest.raises(ValueError):
        fetch_market("")
