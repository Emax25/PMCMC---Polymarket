"""Tests for src/data/preprocess.py: cleaning, features, wallet index, I/O."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.polymarket_api import RawTrade
from src.data.preprocess import (
    ProcessedMarket,
    WalletIndex,
    build_genre_dataset,
    build_processed_market,
    clean_trades,
    compute_features,
    load_processed,
    save_processed,
    trades_to_dataframe,
)
from src.inference.particle_gibbs import MarketData
from src.utils.transforms import logit

FIXTURES = Path(__file__).parent / "fixtures"


def _load_trades(name: str) -> list[RawTrade]:
    payload = json.loads((FIXTURES / name).read_text())
    return [RawTrade.from_dict(d) for d in payload]


# ---------------- WalletIndex ----------------

def test_wallet_index_inserts_on_demand():
    idx = WalletIndex()
    assert idx.add("0xA") == 0
    assert idx.add("0xB") == 1
    assert idx.add("0xA") == 0       # idempotent
    assert idx.n_wallets == 2


def test_wallet_index_encode_returns_ndarray():
    idx = WalletIndex()
    out = idx.encode(["0xA", "0xB", "0xA", "0xC"])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int64
    assert out.tolist() == [0, 1, 0, 2]


def test_wallet_index_from_trade_tables_global_order():
    df1 = pd.DataFrame({"wallet": ["0xA", "0xB", "0xA"]})
    df2 = pd.DataFrame({"wallet": ["0xC", "0xA"]})
    idx = WalletIndex.from_trade_tables([df1, df2])
    # First-appearance order across the two tables
    assert idx.address_to_id == {"0xA": 0, "0xB": 1, "0xC": 2}


# ---------------- clean_trades ----------------

def test_clean_trades_drops_zero_size_and_oob_price():
    trades = _load_trades("data_trades_page1.json")
    cleaned = clean_trades(trades_to_dataframe(trades))
    # Original 8 rows; 2 should drop (size=0 row, price=1.0 row)
    assert len(cleaned) == 6
    assert (cleaned["size"] > 0).all()
    assert ((cleaned["price"] > 0) & (cleaned["price"] < 1)).all()


def test_clean_trades_drops_missing_wallet_or_hash():
    df = pd.DataFrame({
        "timestamp": [1, 2, 3],
        "price":     [0.5, 0.5, 0.5],
        "size":      [10.0, 10.0, 10.0],
        "wallet":    ["0xA", "", "0xC"],
        "transaction_hash": ["0xT1", "0xT2", ""],
        "side": ["BUY"] * 3,
        "condition_id": ["0xab"] * 3,
        "asset_id": ["1"] * 3,
    })
    cleaned = clean_trades(df)
    assert cleaned["wallet"].tolist() == ["0xA"]


def test_clean_trades_dedupes_on_hash():
    df = pd.DataFrame({
        "timestamp": [1, 1],
        "price":     [0.5, 0.5],
        "size":      [10.0, 10.0],
        "wallet":    ["0xA", "0xA"],
        "transaction_hash": ["0xDUP", "0xDUP"],
        "side": ["BUY", "BUY"],
        "condition_id": ["0xab"] * 2,
        "asset_id": ["1"] * 2,
    })
    cleaned = clean_trades(df)
    assert len(cleaned) == 1


def test_clean_trades_sort_breaks_timestamp_ties_by_hash():
    df = pd.DataFrame({
        "timestamp": [10, 10, 10],
        "price":     [0.5, 0.5, 0.5],
        "size":      [1.0, 1.0, 1.0],
        "wallet":    ["0xA", "0xB", "0xC"],
        "transaction_hash": ["0xC", "0xA", "0xB"],
        "side": ["BUY"] * 3,
        "condition_id": ["0xc"] * 3,
        "asset_id": ["1"] * 3,
    })
    cleaned = clean_trades(df)
    assert cleaned["transaction_hash"].tolist() == ["0xA", "0xB", "0xC"]


def test_clean_trades_empty_input_returns_empty():
    out = clean_trades(pd.DataFrame(columns=[
        "timestamp", "price", "size", "wallet",
        "transaction_hash", "side", "condition_id", "asset_id",
    ]))
    assert out.empty


# ---------------- compute_features ----------------

def test_compute_features_shapes_and_invariants():
    trades = _load_trades("data_trades_page1.json")
    df = clean_trades(trades_to_dataframe(trades))
    feats = compute_features(df)
    T = len(df)
    for key in ("Y", "delta", "log_size_ratio", "t", "p", "S"):
        assert feats[key].shape == (T,)
    # delta[0] = 0 by convention
    assert feats["delta"][0] == 0.0
    assert (feats["delta"][1:] >= 0).all()
    # log(S / S_bar): on raw scale these average to S_bar
    assert np.isclose(feats["S"].mean(), feats["S_bar"])
    # logit equals manual transform
    np.testing.assert_allclose(feats["Y"], logit(feats["p"]))


def test_compute_features_empty_raises():
    with pytest.raises(ValueError):
        compute_features(pd.DataFrame(columns=["timestamp", "price", "size"]))


# ---------------- build_processed_market ----------------

def test_build_processed_market_end_to_end():
    trades = _load_trades("data_trades_page1.json")
    idx = WalletIndex()
    market = build_processed_market(trades, wallet_index=idx, slug="test")

    assert isinstance(market, ProcessedMarket)
    assert market.T == 6              # 8 raw → 6 after cleaning
    assert market.slug == "test"
    assert market.condition_id.startswith("0xaaa")
    # All inference arrays line up
    for arr in (market.Y, market.delta, market.log_size_ratio,
                market.wallet_ids, market.t, market.p, market.S):
        assert arr.shape == (market.T,)
    assert idx.n_wallets >= 1


def test_build_processed_market_rejects_mixed_condition_ids():
    trades = [
        RawTrade.from_dict({
            "proxyWallet": "0xA", "side": "BUY", "asset": "1",
            "conditionId": "0xab1", "size": "1", "price": "0.5",
            "timestamp": "1", "transactionHash": "0xT1",
        }),
        RawTrade.from_dict({
            "proxyWallet": "0xB", "side": "BUY", "asset": "1",
            "conditionId": "0xab2", "size": "1", "price": "0.5",
            "timestamp": "2", "transactionHash": "0xT2",
        }),
    ]
    with pytest.raises(ValueError):
        build_processed_market(trades, wallet_index=WalletIndex())


def test_build_processed_market_to_market_data_round_trip():
    trades = _load_trades("data_trades_page1.json")
    market = build_processed_market(trades, wallet_index=WalletIndex())
    md = market.to_market_data()
    assert isinstance(md, MarketData)
    np.testing.assert_array_equal(md.Y, market.Y)
    np.testing.assert_array_equal(md.wallet_ids, market.wallet_ids)
    np.testing.assert_array_equal(md.delta, market.delta)
    np.testing.assert_array_equal(md.log_size_ratio, market.log_size_ratio)


# ---------------- build_genre_dataset ----------------

def test_build_genre_dataset_shares_wallet_index():
    trades1 = _load_trades("data_trades_page1.json")
    trades2 = _load_trades("data_trades_page2.json")
    markets, idx = build_genre_dataset([
        ("market_alpha", trades1),
        ("market_beta", trades2),
    ])
    assert len(markets) == 2
    assert markets[0].slug == "market_alpha"
    # WALLETA appears in both pages → must map to one id
    a_ids_m1 = set(markets[0].wallet_ids.tolist())
    a_ids_m2 = set(markets[1].wallet_ids.tolist())
    # WALLETA is the first wallet that appears anywhere → id 0
    assert 0 in a_ids_m1 and 0 in a_ids_m2
    # And the global index covers every distinct wallet
    expected_wallets = {
        "0xWALLETA000000000000000000000000000000001",
        "0xWALLETB000000000000000000000000000000002",
        "0xWALLETC000000000000000000000000000000003",
        "0xWALLETD000000000000000000000000000000004",
        "0xWALLETE000000000000000000000000000000005",
    }
    assert set(idx.address_to_id.keys()) == expected_wallets


# ---------------- save / load ----------------

def test_save_load_processed_round_trip(tmp_path):
    trades = _load_trades("data_trades_page1.json")
    market = build_processed_market(trades, wallet_index=WalletIndex(), slug="rt")
    save_processed(market, tmp_path)
    loaded = load_processed(tmp_path / "rt.parquet")

    assert loaded.condition_id == market.condition_id
    assert loaded.slug == market.slug
    assert loaded.S_bar == pytest.approx(market.S_bar)
    np.testing.assert_allclose(loaded.Y, market.Y)
    np.testing.assert_allclose(loaded.delta, market.delta)
    np.testing.assert_allclose(loaded.log_size_ratio, market.log_size_ratio)
    np.testing.assert_array_equal(loaded.wallet_ids, market.wallet_ids)


def test_pipeline_compatible_with_inference():
    """Sanity check: a real-data ProcessedMarket runs through one bootstrap_smc
    pass without crashing — full inference correctness is covered by Phase 4
    tests; this just verifies the integration surface."""
    from config.default_params import InferenceConfig, ModelParams
    from src.inference.smc import bootstrap_smc

    trades = _load_trades("data_trades_page1.json")
    market = build_processed_market(trades, wallet_index=WalletIndex())
    md = market.to_market_data()

    params = ModelParams.warm_start(market.Y)
    cfg = InferenceConfig(N=20, seed=0)
    theta_w = np.full(int(md.wallet_ids.max()) + 1, 0.05)
    out = bootstrap_smc(
        md.Y, md.delta, md.log_size_ratio, md.wallet_ids,
        theta_w, params, cfg, rng=np.random.default_rng(0),
    )
    assert np.isfinite(out.log_marginal)
    assert out.X_filt.shape == (market.T,)
