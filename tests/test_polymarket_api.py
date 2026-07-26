"""Offline tests for src/data/polymarket_api.py.

All HTTP is mocked via `unittest.mock` so the suite never hits the network.
Fixtures live under `tests/fixtures/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.polymarket_api import (
    DATA_API_MAX_OFFSET,
    GAMMA_BASE,
    POLITICS_KEYWORDS,
    MarketMeta,
    PolymarketAPIError,
    RawTrade,
    _extract_tags,
    fetch_market_by_slug,
    fetch_markets,
    fetch_trades,
    fetch_trades_windowed,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict | list:
    """Read and parse a JSON fixture file."""
    return json.loads((FIXTURES / name).read_text())


def _mock_response(json_payload, status_code: int = 200):
    """Build a minimal requests.Response mock with given payload and status."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_payload
    resp.text = json.dumps(json_payload)
    return resp


# ---------------- Dataclass parsing ----------------


def test_market_meta_from_dict_minimal_fields():
    """Defaults to closed=False and empty tags when absent."""
    m = MarketMeta.from_dict({"id": "1", "conditionId": "0xabc", "volume": "1000"})
    assert m.id == "1"
    assert m.condition_id == "0xabc"
    assert m.volume == 1000.0
    assert m.closed is False  # default when missing
    assert m.tags == []


def test_market_meta_from_dict_handles_snake_case_alias():
    """snake_case key accepted as alias for camelCase."""
    m = MarketMeta.from_dict({"condition_id": "0xdef", "volume": 0})
    assert m.condition_id == "0xdef"


def test_market_meta_from_dict_volume_none_safe():
    """None volume coerces to 0.0."""
    m = MarketMeta.from_dict({"conditionId": "0x", "volume": None})
    assert m.volume == 0.0


def test_extract_tags_string_form():
    """List of strings returned as-is."""
    assert _extract_tags(["Politics", "Elections"]) == ["Politics", "Elections"]


def test_extract_tags_object_form():
    """Dict tags extracted from label, name, or slug key."""
    tags = [{"label": "Politics"}, {"name": "Crypto"}, {"slug": "world-cup"}]
    assert _extract_tags(tags) == ["Politics", "Crypto", "world-cup"]


def test_extract_tags_mixed_and_missing():
    """None, empty list, and mixed string/dict forms all handled."""
    assert _extract_tags(None) == []
    assert _extract_tags([]) == []
    assert _extract_tags(["Politics", {"label": "Elections"}]) == [
        "Politics",
        "Elections",
    ]


def test_raw_trade_from_dict_typed():
    """camelCase fields parsed; side upper-cased, numeric types cast."""
    t = RawTrade.from_dict(
        {
            "proxyWallet": "0xWA",
            "side": "buy",
            "asset": "111",
            "conditionId": "0xab",
            "size": "250.5",
            "price": "0.42",
            "timestamp": "1709000000",
            "transactionHash": "0xTX",
        }
    )
    assert t.wallet == "0xWA"
    assert t.side == "BUY"  # upper-cased
    assert t.price == 0.42
    assert t.size == 250.5
    assert t.timestamp == 1709000000  # cast to int
    assert t.transaction_hash == "0xTX"


def test_raw_trade_from_dict_snake_case_alias():
    """snake_case aliases accepted for wallet and asset_id."""
    t = RawTrade.from_dict(
        {
            "wallet": "0xW",
            "side": "SELL",
            "asset_id": "222",
            "condition_id": "0xc",
            "size": 10,
            "price": 0.5,
            "timestamp": 17,
            "transaction_hash": "0xT",
        }
    )
    assert t.wallet == "0xW"
    assert t.asset_id == "222"


# ---------------- fetch_markets ----------------


def test_fetch_markets_filters_and_sorts():
    """Low-volume and no-conditionId markets dropped; sorted by volume."""
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        markets = fetch_markets(min_volume=10_000.0)

    # Low-volume and empty-condition-id markets dropped
    slugs = [m.slug for m in markets]
    assert "low-volume-side-market" not in slugs
    assert "no-condition-id-market" not in slugs
    # Sorted by volume descending
    assert slugs == [
        "presidential-election-winner-2024",
        "senate-control-2024",
    ]
    # Tags propagated for both shapes
    assert "Politics" in markets[0].tags
    assert "Politics" in markets[1].tags


def test_fetch_markets_passes_correct_params():
    """All kwargs serialised into correct query-param names."""
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        fetch_markets(
            tag="politics", closed=True, limit=50, offset=10, min_volume=10_000.0
        )
        url, kwargs = g.call_args[0][0], g.call_args[1]
    assert url == f"{GAMMA_BASE}/markets"
    assert kwargs["params"] == {
        "limit": 50,
        "offset": 10,
        "tag_slug": "politics",
        "closed": "true",
        "order": "volumeNum",
        "ascending": "false",
        "volume_num_min": 10_000.0,
    }


def test_fetch_markets_omits_volume_num_min_when_zero():
    """min_volume=0 disables the server-side volume filter."""
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        fetch_markets(min_volume=0.0)
        sent = g.call_args[1]["params"]
    assert "volume_num_min" not in sent


def test_fetch_markets_order_params_can_be_disabled():
    """order=None omits both order and ascending from params."""
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        fetch_markets(min_volume=0.0, order=None)
        sent = g.call_args[1]["params"]
    assert "order" not in sent and "ascending" not in sent


def test_fetch_markets_ascending_flag_serializes():
    """ascending=True serialises to string 'true'."""
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        fetch_markets(min_volume=0.0, order="endDate", ascending=True)
        sent = g.call_args[1]["params"]
    assert sent["order"] == "endDate"
    assert sent["ascending"] == "true"


def test_fetch_markets_question_keywords_filter():
    """`question_keywords` keeps only markets whose question contains a hit."""
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        # min_volume=0 so the keyword filter is the only thing trimming
        markets = fetch_markets(
            min_volume=0.0,
            question_keywords=["election", "senate"],
        )
    slugs = [m.slug for m in markets]
    assert "presidential-election-winner-2024" in slugs
    assert "senate-control-2024" in slugs
    assert "low-volume-side-market" not in slugs


def test_fetch_markets_question_keywords_case_insensitive():
    """Keyword filter is case-insensitive."""
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        markets = fetch_markets(
            min_volume=0.0,
            question_keywords=["ELECTION"],
        )
    assert any("election" in m.question.lower() for m in markets)
    assert all("election" in m.question.lower() for m in markets)


def test_politics_keywords_constant_is_useful():
    """The exported POLITICS_KEYWORDS bag matches the politics questions in
    our fixture and rejects the non-politics ones."""
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        markets = fetch_markets(
            min_volume=0.0,
            question_keywords=list(POLITICS_KEYWORDS),
        )
    slugs = {m.slug for m in markets}
    assert slugs == {
        "presidential-election-winner-2024",
        "senate-control-2024",
    }


def test_fetch_markets_raises_on_non_list_payload():
    """Non-list API response raises PolymarketAPIError."""
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response({"error": "bad request"})
        with pytest.raises(PolymarketAPIError):
            fetch_markets(min_volume=0.0)


# ---------------- fetch_trades ----------------


def test_fetch_trades_paginates_until_short_page(monkeypatch):
    """Pagination stops when a page shorter than page_size is returned."""
    page1 = _load("data_trades_page1.json")  # length 8
    page2 = _load("data_trades_page2.json")  # length 2, signals end

    seen_offsets: list[int] = []

    def fake_get(url, params=None, timeout=None):
        seen_offsets.append(params["offset"])
        if params["offset"] == 0:
            return _mock_response(page1)
        if params["offset"] == 8:
            return _mock_response(page2)
        return _mock_response([])

    monkeypatch.setattr("src.data.polymarket_api.requests.get", fake_get)
    monkeypatch.setattr("src.data.polymarket_api.time.sleep", lambda _: None)

    trades = fetch_trades(
        "0xaaa000000000000000000000000000000000000000000000000000000000aa01",
        page_size=8,
    )
    assert len(trades) == 10
    assert seen_offsets == [0, 8]  # stopped at short page 2
    assert isinstance(trades[0], RawTrade)
    assert all(t.condition_id.startswith("0xaaa") for t in trades)


def test_fetch_trades_empty_first_page(monkeypatch):
    """Empty first page returns an empty list."""
    monkeypatch.setattr(
        "src.data.polymarket_api.requests.get",
        lambda *a, **k: _mock_response([]),
    )
    monkeypatch.setattr("src.data.polymarket_api.time.sleep", lambda _: None)
    assert fetch_trades("0xabc", page_size=500) == []


def test_fetch_trades_rejects_empty_condition_id():
    """Empty condition_id raises ValueError immediately."""
    with pytest.raises(ValueError):
        fetch_trades("")


def test_fetch_trades_respects_max_offset(monkeypatch):
    """Stop cleanly when the next page would exceed max_offset (Polymarket
    caps historical offset at 3000)."""
    full_page = [
        {
            "proxyWallet": "0xX",
            "side": "BUY",
            "asset": "1",
            "conditionId": "0xabc",
            "size": "1",
            "price": "0.5",
            "timestamp": "1",
            "transactionHash": "0xT" + str(i),
        }
        for i in range(500)
    ]
    seen_params: list[dict] = []

    def fake_get(url, params=None, timeout=None):
        seen_params.append(dict(params or {}))
        return _mock_response(full_page[: int(params["limit"])])

    monkeypatch.setattr("src.data.polymarket_api.requests.get", fake_get)
    monkeypatch.setattr("src.data.polymarket_api.time.sleep", lambda _: None)

    trades = fetch_trades("0xabc", page_size=500, max_offset=1500)
    # Three pages: offset=0, 500, 1000 — offset=1500 hits the cap and stops
    offsets = [p["offset"] for p in seen_params]
    assert offsets == [0, 500, 1000]
    assert len(trades) == 1500


def test_fetch_trades_trims_final_page_at_offset_cap(monkeypatch):
    """Final page is shortened so total offset stays <= max_offset."""
    full_page = [
        {
            "proxyWallet": "0xX",
            "side": "BUY",
            "asset": "1",
            "conditionId": "0xabc",
            "size": "1",
            "price": "0.5",
            "timestamp": "1",
            "transactionHash": "0xT" + str(i),
        }
        for i in range(500)
    ]
    seen_params: list[dict] = []

    def fake_get(url, params=None, timeout=None):
        seen_params.append(dict(params or {}))
        return _mock_response(full_page[: int(params["limit"])])

    monkeypatch.setattr("src.data.polymarket_api.requests.get", fake_get)
    monkeypatch.setattr("src.data.polymarket_api.time.sleep", lambda _: None)

    trades = fetch_trades("0xabc", page_size=500, max_offset=1200)
    # offset=0 → limit=500; offset=500 → limit=500; offset=1000 → limit=200
    assert [p["limit"] for p in seen_params] == [500, 500, 200]
    assert len(trades) == 1200


def test_fetch_trades_respects_max_pages(monkeypatch):
    """If the API never returns a short page, we stop at max_pages."""
    full_page = [
        {
            "proxyWallet": "0xX",
            "side": "BUY",
            "asset": "1",
            "conditionId": "0xabc",
            "size": "1",
            "price": "0.5",
            "timestamp": "1",
            "transactionHash": "0xT" + str(i),
        }
        for i in range(2)
    ]
    monkeypatch.setattr(
        "src.data.polymarket_api.requests.get",
        lambda *a, **k: _mock_response(full_page),
    )
    monkeypatch.setattr("src.data.polymarket_api.time.sleep", lambda _: None)
    trades = fetch_trades("0xabc", page_size=2, max_pages=3)
    assert len(trades) == 6


# ---------------- fetch_trades_windowed ----------------


class _FakeTradesAPI:
    """In-memory /trades stand-in reproducing the probed server semantics.

    Mirrors what the live Data API was measured to do (see the constants block
    in polymarket_api.py): newest-first ordering, inclusive `start`/`end`
    second-resolution filters, and HTTP 400 once `offset` passes the cap.
    """

    def __init__(
        self,
        timestamps: list[int],
        *,
        max_offset: int,
        fail_calls: tuple[int, ...] = (),
    ):
        """Build a market whose i-th trade has ``timestamps[i]`` (descending).

        Args:
            timestamps: trade timestamps, newest first; ties are same-second
                bursts and keep their listed order, as the real API does.
            max_offset: highest `offset` served before HTTP 400.
            fail_calls: 1-based call indices answered with HTTP 429 instead.
        """
        self.rows = [
            {
                "proxyWallet": f"0xW{i % 7}",
                "side": "BUY" if i % 2 else "SELL",
                "asset": "1",
                "conditionId": "0xabc",
                "size": "1",
                "price": "0.5",
                "timestamp": str(ts),
                "transactionHash": f"0xT{i:05d}",
            }
            for i, ts in enumerate(timestamps)
        ]
        self.max_offset = max_offset
        self.fail_calls = set(fail_calls)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        """Serve one paged, filtered /trades request."""
        params = dict(params or {})
        self.calls.append(params)
        if len(self.calls) in self.fail_calls:
            return _mock_response({"error": "rate limited"}, status_code=429)

        offset = int(params["offset"])
        if offset > self.max_offset:
            return _mock_response(
                {"error": f"max historical trades offset of {self.max_offset}"},
                status_code=400,
            )
        start = params.get("start")
        end = params.get("end")
        rows = self.rows
        if start is not None:
            rows = [r for r in rows if int(r["timestamp"]) >= int(start)]
        if end is not None:
            rows = [r for r in rows if int(r["timestamp"]) <= int(end)]
        return _mock_response(rows[offset : offset + int(params["limit"])])


def _install(monkeypatch, api: _FakeTradesAPI) -> None:
    """Route requests.get at the fake API and disable real sleeping."""
    monkeypatch.setattr("src.data.polymarket_api.requests.get", api.get)
    monkeypatch.setattr("src.data.polymarket_api.time.sleep", lambda _: None)


def _assert_complete(trades: list[RawTrade], api: _FakeTradesAPI) -> None:
    """Every trade returned exactly once, in descending timestamp order."""
    got = [t.transaction_hash for t in trades]
    assert len(got) == len(set(got)), "duplicate rows returned"
    assert set(got) == {r["transactionHash"] for r in api.rows}
    ts = [t.timestamp for t in trades]
    assert all(a >= b for a, b in zip(ts, ts[1:])), "not newest-first"


def test_windowed_walks_past_the_offset_cap(monkeypatch):
    """7000 trades behind a 3000-offset cap all come back exactly once."""
    api = _FakeTradesAPI([1_700_010_000 - i for i in range(7000)], max_offset=3000)
    _install(monkeypatch, api)

    trades = fetch_trades_windowed("0xabc", page_size=500, offset_limit=3000)

    assert len(trades) == 7000
    _assert_complete(trades, api)
    # Every window restarts its own offset counter at 0 and re-anchors on `end`.
    assert api.calls[0]["offset"] == 0 and "end" not in api.calls[0]
    reanchors = [c for c in api.calls if c["offset"] == 0 and "end" in c]
    assert len(reanchors) >= 2


def test_windowed_same_second_burst_at_window_boundary(monkeypatch):
    """A burst straddling the truncation point loses and duplicates nothing."""
    # Window 1 = offsets 0..3000 of 500 rows each = rows 0..3499. Rows
    # 3495..3509 share one second, so the cap slices that burst in half.
    timestamps = []
    for i in range(4000):
        if 3495 <= i <= 3509:
            timestamps.append(1_700_000_000 - 3495)
        else:
            timestamps.append(1_700_000_000 - i)
    api = _FakeTradesAPI(timestamps, max_offset=3000)
    _install(monkeypatch, api)

    trades = fetch_trades_windowed("0xabc", page_size=500, offset_limit=3000)

    assert len(trades) == 4000
    _assert_complete(trades, api)
    burst = [t for t in trades if t.timestamp == 1_700_000_000 - 3495]
    assert len(burst) == 15  # the split second is whole again


def test_windowed_raises_when_one_second_cannot_fit(monkeypatch):
    """A second larger than a full window would stall the walk — surface it."""
    api = _FakeTradesAPI([1_700_000_000] * 3000, max_offset=1000)
    _install(monkeypatch, api)
    with pytest.raises(PolymarketAPIError, match="cannot advance"):
        fetch_trades_windowed("0xabc", page_size=500, offset_limit=1000)


def test_windowed_resumes_after_429_without_duplicates(monkeypatch):
    """A 429 mid-window is retried and the walk resumes cleanly."""
    api = _FakeTradesAPI(
        [1_700_010_000 - i for i in range(5000)],
        max_offset=3000,
        fail_calls=(3, 9),  # one inside window 1, one inside window 2
    )
    sleeps: list[float] = []
    monkeypatch.setattr("src.data.polymarket_api.requests.get", api.get)
    monkeypatch.setattr(
        "src.data.polymarket_api.time.sleep", lambda s: sleeps.append(s)
    )

    trades = fetch_trades_windowed("0xabc", page_size=500, offset_limit=3000)

    assert len(trades) == 5000
    _assert_complete(trades, api)
    assert any(s > 0 for s in sleeps)  # backoff actually slept


def test_windowed_bounds_are_inclusive_and_zero_is_omitted(monkeypatch):
    """start/end are forwarded as unix seconds; falsy bounds are never sent."""
    api = _FakeTradesAPI([1_000 + i for i in range(20)][::-1], max_offset=3000)
    _install(monkeypatch, api)

    trades = fetch_trades_windowed(
        "0xabc", start_ts=1_005, end_ts=1_010, page_size=500
    )
    assert [t.timestamp for t in trades] == list(range(1_010, 1_004, -1))
    assert api.calls[0] == {
        "market": "0xabc",
        "limit": 500,
        "offset": 0,
        "start": 1_005,
        "end": 1_010,
    }

    api.calls.clear()
    fetch_trades_windowed("0xabc", start_ts=0, end_ts=0, page_size=500)
    assert "start" not in api.calls[0] and "end" not in api.calls[0]


def test_windowed_respects_max_pages(monkeypatch):
    """The HTTP budget is enforced rather than silently truncating history."""
    api = _FakeTradesAPI([1_700_000_000 - i for i in range(9000)], max_offset=1000)
    _install(monkeypatch, api)
    with pytest.raises(PolymarketAPIError, match="max_pages"):
        fetch_trades_windowed(
            "0xabc", page_size=500, offset_limit=1000, max_pages=6
        )


def test_windowed_rejects_bad_arguments():
    """Guard rails on condition_id and page_size fire before any HTTP."""
    with pytest.raises(ValueError):
        fetch_trades_windowed("")
    with pytest.raises(ValueError):
        fetch_trades_windowed("0xabc", page_size=0)


def test_fetch_trades_default_path_is_unchanged(monkeypatch):
    """Pin the pre-change default: 6 plain pages, no start/end, 3000 rows."""
    api = _FakeTradesAPI(
        [1_700_010_000 - i for i in range(5000)], max_offset=DATA_API_MAX_OFFSET
    )
    _install(monkeypatch, api)

    trades = fetch_trades("0xabc")

    assert [dict(c) for c in api.calls] == [
        {"market": "0xabc", "limit": 500, "offset": off}
        for off in (0, 500, 1000, 1500, 2000, 2500)
    ]
    assert len(trades) == 3000
    assert [t.transaction_hash for t in trades] == [
        f"0xT{i:05d}" for i in range(3000)
    ]


# ---------------- HTTP error handling ----------------


def test_get_json_raises_on_400():
    """HTTP 400 raises PolymarketAPIError."""
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response({"error": "bad"}, status_code=400)
        with pytest.raises(PolymarketAPIError):
            fetch_markets(min_volume=0.0)


def test_get_json_retries_on_429_then_succeeds(monkeypatch):
    """429/503 retried with backoff; third attempt succeeds."""
    payload = _load("gamma_markets_sample.json")
    responses = [
        _mock_response({}, status_code=429),
        _mock_response({}, status_code=503),
        _mock_response(payload),
    ]

    def fake_get(*a, **k):
        return responses.pop(0)

    sleeps: list[float] = []
    monkeypatch.setattr("src.data.polymarket_api.requests.get", fake_get)
    monkeypatch.setattr(
        "src.data.polymarket_api.time.sleep", lambda s: sleeps.append(s)
    )
    markets = fetch_markets(min_volume=10_000.0)
    assert len(markets) >= 1
    assert len(sleeps) == 2  # one per retry, before the success
    assert all(s > 0 for s in sleeps)


def test_fetch_market_by_slug_events_fast_path(monkeypatch):
    """/events?slug=X returns the event; we pull its single market."""
    event_payload = [
        {
            "id": "11143",
            "slug": "will-trump-launch-a-coin-before-the-election",
            "markets": [
                {
                    "id": "540817",
                    "slug": "will-trump-launch-a-coin-before-the-election",
                    "conditionId": "0x70de1b06",
                    "question": "Will Trump launch a coin?",
                    "volume": 76_899_060,
                    "closed": True,
                }
            ],
        }
    ]
    seen_urls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        seen_urls.append(url)
        if url.endswith("/events"):
            return _mock_response(event_payload)
        return _mock_response([])

    monkeypatch.setattr("src.data.polymarket_api.requests.get", fake_get)
    m = fetch_market_by_slug("will-trump-launch-a-coin-before-the-election")
    assert m.slug == "will-trump-launch-a-coin-before-the-election"
    assert m.condition_id == "0x70de1b06"
    # Only the events endpoint was touched — the scan was unnecessary
    assert all(u.endswith("/events") for u in seen_urls)


def test_fetch_market_by_slug_falls_back_to_markets_scan(monkeypatch):
    """When /events returns empty (old markets), we paginate /markets."""
    target_slug = "will-donald-trump-win-the-2024-us-presidential-election"
    page = [
        {"id": "X", "slug": "decoy", "conditionId": "0x1", "volume": 1_500_000_000},
        {
            "id": "Y",
            "slug": target_slug,
            "conditionId": "0xTRUMP24",
            "volume": 1_500_000_000,
            "closed": True,
        },
    ]
    calls: list[tuple[str, dict]] = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params or {}))
        if url.endswith("/events"):
            return _mock_response([])
        return _mock_response(page)

    monkeypatch.setattr("src.data.polymarket_api.requests.get", fake_get)
    m = fetch_market_by_slug(target_slug)
    assert m.slug == target_slug
    assert m.condition_id == "0xTRUMP24"
    # First /markets call should request closed=true (we pass closed_first=True default)
    markets_calls = [(u, p) for (u, p) in calls if u.endswith("/markets")]
    assert markets_calls and markets_calls[0][1].get("closed") == "true"


def test_fetch_market_by_slug_raises_when_truly_missing(monkeypatch):
    """PolymarketAPIError raised when slug absent from all pages."""
    monkeypatch.setattr(
        "src.data.polymarket_api.requests.get",
        lambda *a, **k: _mock_response([]),
    )
    with pytest.raises(PolymarketAPIError):
        fetch_market_by_slug("not-a-real-slug", scan_limit=500)


def test_get_json_request_exception_retried(monkeypatch):
    """ConnectionError retried; succeeds once resolved."""
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("boom")
        return _mock_response([])

    monkeypatch.setattr("src.data.polymarket_api.requests.get", fake_get)
    monkeypatch.setattr("src.data.polymarket_api.time.sleep", lambda _: None)
    # fetch_markets parses [] as an empty list of markets; should not raise
    assert fetch_markets(min_volume=0.0) == []
    assert calls["n"] == 3
