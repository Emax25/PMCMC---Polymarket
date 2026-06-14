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
    DATA_BASE,
    GAMMA_BASE,
    POLITICS_KEYWORDS,
    MarketMeta,
    PolymarketAPIError,
    RawTrade,
    _extract_tags,
    fetch_market_by_slug,
    fetch_markets,
    fetch_trades,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


def _mock_response(json_payload, status_code: int = 200):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_payload
    resp.text = json.dumps(json_payload)
    return resp


# ---------------- Dataclass parsing ----------------

def test_market_meta_from_dict_minimal_fields():
    m = MarketMeta.from_dict(
        {"id": "1", "conditionId": "0xabc", "volume": "1000"}
    )
    assert m.id == "1"
    assert m.condition_id == "0xabc"
    assert m.volume == 1000.0
    assert m.closed is False        # default when missing
    assert m.tags == []


def test_market_meta_from_dict_handles_snake_case_alias():
    m = MarketMeta.from_dict({"condition_id": "0xdef", "volume": 0})
    assert m.condition_id == "0xdef"


def test_market_meta_from_dict_volume_none_safe():
    m = MarketMeta.from_dict({"conditionId": "0x", "volume": None})
    assert m.volume == 0.0


def test_extract_tags_string_form():
    assert _extract_tags(["Politics", "Elections"]) == ["Politics", "Elections"]


def test_extract_tags_object_form():
    tags = [{"label": "Politics"}, {"name": "Crypto"}, {"slug": "world-cup"}]
    assert _extract_tags(tags) == ["Politics", "Crypto", "world-cup"]


def test_extract_tags_mixed_and_missing():
    assert _extract_tags(None) == []
    assert _extract_tags([]) == []
    assert _extract_tags(["Politics", {"label": "Elections"}]) == [
        "Politics",
        "Elections",
    ]


def test_raw_trade_from_dict_typed():
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
    assert t.side == "BUY"             # upper-cased
    assert t.price == 0.42
    assert t.size == 250.5
    assert t.timestamp == 1709000000   # cast to int
    assert t.transaction_hash == "0xTX"


def test_raw_trade_from_dict_snake_case_alias():
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
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        fetch_markets(tag="politics", closed=True, limit=50, offset=10,
                      min_volume=10_000.0)
        url, kwargs = g.call_args[0][0], g.call_args[1]
    assert url == f"{GAMMA_BASE}/markets"
    assert kwargs["params"] == {
        "limit": 50, "offset": 10, "tag_slug": "politics", "closed": "true",
        "order": "volumeNum", "ascending": "false",
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
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        fetch_markets(min_volume=0.0, order=None)
        sent = g.call_args[1]["params"]
    assert "order" not in sent and "ascending" not in sent


def test_fetch_markets_ascending_flag_serializes():
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
            min_volume=0.0, question_keywords=["election", "senate"],
        )
    slugs = [m.slug for m in markets]
    assert "presidential-election-winner-2024" in slugs
    assert "senate-control-2024" in slugs
    assert "low-volume-side-market" not in slugs


def test_fetch_markets_question_keywords_case_insensitive():
    payload = _load("gamma_markets_sample.json")
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response(payload)
        markets = fetch_markets(
            min_volume=0.0, question_keywords=["ELECTION"],
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
            min_volume=0.0, question_keywords=list(POLITICS_KEYWORDS),
        )
    slugs = {m.slug for m in markets}
    assert slugs == {
        "presidential-election-winner-2024", "senate-control-2024",
    }


def test_fetch_markets_raises_on_non_list_payload():
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response({"error": "bad request"})
        with pytest.raises(PolymarketAPIError):
            fetch_markets(min_volume=0.0)


# ---------------- fetch_trades ----------------

def test_fetch_trades_paginates_until_short_page(monkeypatch):
    page1 = _load("data_trades_page1.json")   # length 8
    page2 = _load("data_trades_page2.json")   # length 2, signals end

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
    assert seen_offsets == [0, 8]                  # stopped at short page 2
    assert isinstance(trades[0], RawTrade)
    assert all(t.condition_id.startswith("0xaaa") for t in trades)


def test_fetch_trades_empty_first_page(monkeypatch):
    monkeypatch.setattr(
        "src.data.polymarket_api.requests.get",
        lambda *a, **k: _mock_response([]),
    )
    monkeypatch.setattr("src.data.polymarket_api.time.sleep", lambda _: None)
    assert fetch_trades("0xabc", page_size=500) == []


def test_fetch_trades_rejects_empty_condition_id():
    with pytest.raises(ValueError):
        fetch_trades("")


def test_fetch_trades_respects_max_offset(monkeypatch):
    """Stop cleanly when the next page would exceed max_offset (Polymarket
    caps historical offset at 3000)."""
    full_page = [
        {
            "proxyWallet": "0xX", "side": "BUY", "asset": "1",
            "conditionId": "0xabc", "size": "1", "price": "0.5",
            "timestamp": "1", "transactionHash": "0xT" + str(i),
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
            "proxyWallet": "0xX", "side": "BUY", "asset": "1",
            "conditionId": "0xabc", "size": "1", "price": "0.5",
            "timestamp": "1", "transactionHash": "0xT" + str(i),
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
            "proxyWallet": "0xX", "side": "BUY", "asset": "1",
            "conditionId": "0xabc", "size": "1", "price": "0.5",
            "timestamp": "1", "transactionHash": "0xT" + str(i),
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


# ---------------- HTTP error handling ----------------

def test_get_json_raises_on_400():
    with patch("src.data.polymarket_api.requests.get") as g:
        g.return_value = _mock_response({"error": "bad"}, status_code=400)
        with pytest.raises(PolymarketAPIError):
            fetch_markets(min_volume=0.0)


def test_get_json_retries_on_429_then_succeeds(monkeypatch):
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
    assert len(sleeps) == 2          # one per retry, before the success
    assert all(s > 0 for s in sleeps)


def test_fetch_market_by_slug_events_fast_path(monkeypatch):
    """/events?slug=X returns the event; we pull its single market."""
    event_payload = [{
        "id": "11143",
        "slug": "will-trump-launch-a-coin-before-the-election",
        "markets": [{
            "id": "540817",
            "slug": "will-trump-launch-a-coin-before-the-election",
            "conditionId": "0x70de1b06",
            "question": "Will Trump launch a coin?",
            "volume": 76_899_060,
            "closed": True,
        }],
    }]
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
        {"id": "Y", "slug": target_slug, "conditionId": "0xTRUMP24",
         "volume": 1_500_000_000, "closed": True},
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
    monkeypatch.setattr(
        "src.data.polymarket_api.requests.get",
        lambda *a, **k: _mock_response([]),
    )
    with pytest.raises(PolymarketAPIError):
        fetch_market_by_slug("not-a-real-slug", scan_limit=500)


def test_get_json_request_exception_retried(monkeypatch):
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
