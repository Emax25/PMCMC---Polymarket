"""Cleaning + feature computation for real Polymarket trade data.

Pipeline (§8.3):
  1. Drop zero-size and zero-price trades; require known wallet + non-empty
     transaction hash.
  2. Sort strictly by (timestamp, transaction_hash) — hash breaks the
     same-second ties deterministically.
  3. Compute Δ_i (inter-trade time), log(S_i / S̄), and Y_i = logit(p_i).
  4. Assign integer wallet ids via a `WalletIndex` shared across markets.

The output `ProcessedMarket` mirrors `SyntheticMarket` (without ground-truth
latents) so downstream inference/plotting code can treat synthetic and real
markets uniformly. `to_market_data()` returns the slim `MarketData` that the
PG/iPMCMC samplers expect.

I/O helpers `save_processed`/`load_processed` use Parquet for the trade-level
columns and a sidecar JSON for scalars/metadata — fast reload per §7.4.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.polymarket_api import RawTrade
from src.inference.particle_gibbs import MarketData
from src.utils.transforms import logit


# ---------------- Wallet index ----------------

@dataclass
class WalletIndex:
    """Global mapping wallet_address → integer id, shared across markets.

    The hierarchical θ_w prior (§3.2) is defined over the union of wallets
    that appear in any market in the genre, so the index must be built once
    over the full list of cleaned trade tables and applied uniformly.
    """
    address_to_id: dict[str, int] = field(default_factory=dict)

    @property
    def n_wallets(self) -> int:
        return len(self.address_to_id)

    def add(self, address: str) -> int:
        """Insert if new; return the id either way."""
        if address not in self.address_to_id:
            self.address_to_id[address] = len(self.address_to_id)
        return self.address_to_id[address]

    def encode(self, addresses: list[str] | np.ndarray | pd.Series) -> np.ndarray:
        """Map a sequence of addresses to ints, inserting unknowns."""
        ids = np.empty(len(addresses), dtype=np.int64)
        for i, a in enumerate(addresses):
            ids[i] = self.add(str(a))
        return ids

    @classmethod
    def from_trade_tables(
        cls,
        cleaned_tables: list[pd.DataFrame],
        *,
        wallet_col: str = "wallet",
    ) -> WalletIndex:
        """Build an index covering every wallet observed in `cleaned_tables`,
        in order of first appearance across the concatenated input."""
        idx = cls()
        for df in cleaned_tables:
            for w in df[wallet_col].tolist():
                idx.add(str(w))
        return idx


# ---------------- ProcessedMarket ----------------

@dataclass
class ProcessedMarket:
    """Real-data analog of `SyntheticMarket` minus the ground-truth latents.

    Carries both the inference-ready arrays (Y, delta, log_size_ratio,
    wallet_ids) and the raw columns kept for analysis/plotting.
    """
    # Inference inputs (consumed by MarketData)
    Y: np.ndarray              # (T,) logit(p_i)
    delta: np.ndarray          # (T,) inter-trade times in seconds; delta[0] = 0
    log_size_ratio: np.ndarray # (T,) log(S_i / S_bar)
    wallet_ids: np.ndarray     # (T,) integer global wallet ids

    # Raw retained for plots / sanity checks
    t: np.ndarray              # (T,) unix timestamps (seconds)
    p: np.ndarray              # (T,) raw trade prices
    S: np.ndarray              # (T,) raw trade sizes (USDC)
    S_bar: float               # within-market mean size

    # Metadata
    condition_id: str
    slug: str = ""

    @property
    def T(self) -> int:
        return len(self.Y)

    def to_market_data(self) -> MarketData:
        return MarketData(
            Y=self.Y,
            delta=self.delta,
            log_size_ratio=self.log_size_ratio,
            wallet_ids=self.wallet_ids,
        )


# ---------------- Cleaning ----------------

def trades_to_dataframe(trades: list[RawTrade]) -> pd.DataFrame:
    """Flat DataFrame for one market's trades; no filtering applied."""
    return pd.DataFrame(
        {
            "timestamp": [t.timestamp for t in trades],
            "price": [t.price for t in trades],
            "size": [t.size for t in trades],
            "wallet": [t.wallet for t in trades],
            "side": [t.side for t in trades],
            "transaction_hash": [t.transaction_hash for t in trades],
            "condition_id": [t.condition_id for t in trades],
            "asset_id": [t.asset_id for t in trades],
        }
    )


def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Apply §8.3 cleaning: drop invalid rows, dedupe, sort.

    Drops:
      * zero or negative size (fee-only and dust)
      * price outside (0, 1) — Polymarket guarantees this but the API has
        rounding pathologies
      * missing wallet or transaction hash
    Sorts by (timestamp asc, transaction_hash asc) — hash breaks same-second
    ties deterministically. Drops exact duplicates on transaction_hash since
    the Data API occasionally double-counts a fill across pages.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out = out[
        (out["size"] > 0)
        & (out["price"] > 0.0)
        & (out["price"] < 1.0)
        & (out["wallet"].astype(str).str.len() > 0)
        & (out["transaction_hash"].astype(str).str.len() > 0)
    ]
    out = out.drop_duplicates(subset=["transaction_hash"], keep="first")
    out = out.sort_values(["timestamp", "transaction_hash"], kind="mergesort")
    out = out.reset_index(drop=True)
    return out


# ---------------- Feature computation ----------------

def compute_features(df: pd.DataFrame) -> dict[str, np.ndarray | float]:
    """Compute the per-trade features the SSM consumes.

    Returns a dict with keys: Y, delta, log_size_ratio, t, p, S, S_bar.
    Caller is responsible for stamping on wallet_ids and metadata.
    """
    if df.empty:
        raise ValueError("compute_features: cannot operate on an empty trade table.")

    t = df["timestamp"].to_numpy(dtype=float)
    p = df["price"].to_numpy(dtype=float)
    S = df["size"].to_numpy(dtype=float)

    delta = np.zeros_like(t)
    delta[1:] = np.diff(t)
    delta = np.maximum(delta, 0.0)   # mergesort is stable; ties → 0 by design

    S_bar = float(S.mean())
    log_size_ratio = np.log(S / S_bar)

    Y = logit(p)

    return {
        "Y": Y,
        "delta": delta,
        "log_size_ratio": log_size_ratio,
        "t": t,
        "p": p,
        "S": S,
        "S_bar": S_bar,
    }


def build_processed_market(
    trades: list[RawTrade],
    *,
    wallet_index: WalletIndex,
    slug: str = "",
) -> ProcessedMarket:
    """End-to-end: list[RawTrade] → cleaned DataFrame → ProcessedMarket.

    Mutates `wallet_index` in place if new wallets appear. Inherits
    `condition_id` from the first trade in the cleaned table (sanity-checked
    to be uniform across the input).
    """
    df = clean_trades(trades_to_dataframe(trades))
    if df.empty:
        raise ValueError("build_processed_market: no trades survived cleaning.")

    cids = df["condition_id"].unique()
    if len(cids) != 1:
        raise ValueError(
            f"All trades must share one condition_id; got {cids.tolist()}"
        )

    feats = compute_features(df)
    wallet_ids = wallet_index.encode(df["wallet"].tolist())

    return ProcessedMarket(
        Y=feats["Y"],
        delta=feats["delta"],
        log_size_ratio=feats["log_size_ratio"],
        wallet_ids=wallet_ids,
        t=feats["t"],
        p=feats["p"],
        S=feats["S"],
        S_bar=feats["S_bar"],
        condition_id=str(cids[0]),
        slug=slug,
    )


def build_genre_dataset(
    trades_by_market: list[tuple[str, list[RawTrade]]],
) -> tuple[list[ProcessedMarket], WalletIndex]:
    """Process K markets sharing one global wallet index.

    Args:
        trades_by_market: list of (slug, raw_trades) pairs in the order to
            process. Slug is informational only (used for the
            ProcessedMarket.slug field and any sidecar metadata).

    Returns:
        (list[ProcessedMarket], WalletIndex). The index is built up across
        all markets in the input order so wallet ids are stable.
    """
    cleaned: list[tuple[str, pd.DataFrame]] = []
    for slug, trades in trades_by_market:
        df = clean_trades(trades_to_dataframe(trades))
        if df.empty:
            raise ValueError(f"market {slug!r}: no trades survived cleaning.")
        cleaned.append((slug, df))

    wallet_index = WalletIndex.from_trade_tables([df for _, df in cleaned])

    out: list[ProcessedMarket] = []
    for slug, df in cleaned:
        feats = compute_features(df)
        wallet_ids = wallet_index.encode(df["wallet"].tolist())
        cids = df["condition_id"].unique()
        if len(cids) != 1:
            raise ValueError(
                f"market {slug!r}: mixed condition_ids {cids.tolist()}"
            )
        out.append(
            ProcessedMarket(
                Y=feats["Y"],
                delta=feats["delta"],
                log_size_ratio=feats["log_size_ratio"],
                wallet_ids=wallet_ids,
                t=feats["t"],
                p=feats["p"],
                S=feats["S"],
                S_bar=feats["S_bar"],
                condition_id=str(cids[0]),
                slug=slug,
            )
        )
    return out, wallet_index


# ---------------- Persistence ----------------

def save_processed(
    market: ProcessedMarket,
    directory: str | Path,
    *,
    name: str | None = None,
) -> Path:
    """Persist one ProcessedMarket as Parquet (columns) + JSON (scalars).

    Returns the Parquet path written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = name or (market.slug or market.condition_id)
    parquet_path = directory / f"{stem}.parquet"
    meta_path = directory / f"{stem}.meta.json"

    df = pd.DataFrame(
        {
            "Y": market.Y,
            "delta": market.delta,
            "log_size_ratio": market.log_size_ratio,
            "wallet_ids": market.wallet_ids,
            "t": market.t,
            "p": market.p,
            "S": market.S,
        }
    )
    df.to_parquet(parquet_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "S_bar": market.S_bar,
                "condition_id": market.condition_id,
                "slug": market.slug,
            }
        )
    )
    return parquet_path


def save_wallet_index(index: WalletIndex, path: str | Path) -> Path:
    """Persist a wallet index as a JSON {address: id, ...} mapping."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.address_to_id))
    return path


def load_wallet_index(path: str | Path) -> WalletIndex:
    """Inverse of `save_wallet_index`."""
    data = json.loads(Path(path).read_text())
    return WalletIndex(address_to_id={str(a): int(i) for a, i in data.items()})


def load_processed(parquet_path: str | Path) -> ProcessedMarket:
    """Inverse of `save_processed`. Reads `<stem>.parquet` + `<stem>.meta.json`."""
    parquet_path = Path(parquet_path)
    meta_path = parquet_path.with_suffix(".meta.json")
    df = pd.read_parquet(parquet_path)
    meta = json.loads(meta_path.read_text())
    return ProcessedMarket(
        Y=df["Y"].to_numpy(dtype=float),
        delta=df["delta"].to_numpy(dtype=float),
        log_size_ratio=df["log_size_ratio"].to_numpy(dtype=float),
        wallet_ids=df["wallet_ids"].to_numpy(dtype=np.int64),
        t=df["t"].to_numpy(dtype=float),
        p=df["p"].to_numpy(dtype=float),
        S=df["S"].to_numpy(dtype=float),
        S_bar=float(meta["S_bar"]),
        condition_id=str(meta["condition_id"]),
        slug=str(meta.get("slug", "")),
    )
