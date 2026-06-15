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
        """Total number of distinct wallet addresses registered so far."""
        return len(self.address_to_id)

    def add(self, address: str) -> int:
        """Insert an address into the index if new; return its integer id.

        Args:
            address: Wallet address string to register.

        Returns:
            Stable integer id in ``[0, n_wallets)`` assigned to ``address``.
        """
        if address not in self.address_to_id:
            self.address_to_id[address] = len(self.address_to_id)
        return self.address_to_id[address]

    def encode(self, addresses: list[str] | np.ndarray | pd.Series) -> np.ndarray:
        """Map a sequence of wallet addresses to integer ids.

        Addresses not yet in the index are inserted (mutates ``self``).

        Args:
            addresses: Sequence of wallet address strings to encode.

        Returns:
            Integer id array of shape ``(len(addresses),)``, dtype int64.
        """
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
        """Build an index covering every wallet observed in ``cleaned_tables``.

        Wallets are inserted in order of first appearance across the
        concatenated tables, giving stable ids for a fixed input order.

        Args:
            cleaned_tables: Cleaned trade DataFrames to scan; each must
                contain a column named ``wallet_col``.
            wallet_col: Name of the wallet-address column (keyword-only).

        Returns:
            WalletIndex whose ids span all wallets seen in
            ``cleaned_tables``.
        """
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
    Y: np.ndarray  # (T,) logit(p_i)
    delta: np.ndarray  # (T,) inter-trade times in seconds; delta[0] = 0
    log_size_ratio: np.ndarray  # (T,) log(S_i / S_bar)
    wallet_ids: np.ndarray  # (T,) integer global wallet ids

    # Raw retained for plots / sanity checks
    t: np.ndarray  # (T,) unix timestamps (seconds)
    p: np.ndarray  # (T,) raw trade prices
    S: np.ndarray  # (T,) raw trade sizes (USDC)
    S_bar: float  # within-market mean size

    # Metadata
    condition_id: str
    slug: str = ""

    @property
    def T(self) -> int:
        """Number of trades in this market."""
        return len(self.Y)

    def to_market_data(self) -> MarketData:
        """Return the slim MarketData view consumed by the PG/iPMCMC samplers."""
        return MarketData(
            Y=self.Y,
            delta=self.delta,
            log_size_ratio=self.log_size_ratio,
            wallet_ids=self.wallet_ids,
        )


# ---------------- Cleaning ----------------


def trades_to_dataframe(trades: list[RawTrade]) -> pd.DataFrame:
    """Convert a list of RawTrade objects to a flat DataFrame.

    No cleaning or filtering is applied; every field of each trade
    becomes a column. Pass the result to ``clean_trades`` before use.

    Args:
        trades: Raw trades for one market as returned by the API.

    Returns:
        DataFrame with columns: timestamp, price, size, wallet, side,
        transaction_hash, condition_id, asset_id.
    """
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

    Args:
        df: Raw trade DataFrame as produced by ``trades_to_dataframe``.

    Returns:
        Cleaned copy with invalid rows removed, transaction_hash duplicates
        de-duplicated, and rows sorted by (timestamp, transaction_hash).
        Returns an empty DataFrame (columns preserved) when ``df`` is empty.
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

    Derives Δ_i (inter-trade time), log(S_i / S̄), and Y_i = logit(p_i)
    from a cleaned, sorted trade table. ``delta[0]`` is set to 0 by
    convention (no predecessor for the first trade). Caller is responsible
    for stamping on wallet_ids and metadata.

    Args:
        df: Cleaned and sorted trade DataFrame; must be non-empty and
            contain columns timestamp, price, size.

    Returns:
        Dict with keys Y, delta, log_size_ratio, t, p, S (each an array
        of shape ``(T,)``), and S_bar (scalar float mean trade size).

    Raises:
        ValueError: If ``df`` is empty.
    """
    if df.empty:
        raise ValueError("compute_features: cannot operate on an empty trade table.")

    t = df["timestamp"].to_numpy(dtype=float)
    p = df["price"].to_numpy(dtype=float)
    S = df["size"].to_numpy(dtype=float)

    delta = np.zeros_like(t)
    delta[1:] = np.diff(t)
    delta = np.maximum(delta, 0.0)  # mergesort is stable; ties → 0 by design

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
    """Build a ProcessedMarket end-to-end from a list of raw trades.

    Mutates ``wallet_index`` in place when new wallets appear. Inherits
    ``condition_id`` from the cleaned table and asserts it is uniform
    across all trades.

    Args:
        trades: Raw trades for a single market.
        wallet_index: Shared global index; updated in place with any
            new wallet addresses found in this market.
        slug: Human-readable market identifier stored in the result.

    Returns:
        ProcessedMarket ready for inference and plotting.

    Raises:
        ValueError: If no trades survive cleaning, or if the trades
            span more than one condition_id.
    """
    df = clean_trades(trades_to_dataframe(trades))
    if df.empty:
        raise ValueError("build_processed_market: no trades survived cleaning.")

    cids = df["condition_id"].unique()
    if len(cids) != 1:
        raise ValueError(f"All trades must share one condition_id; got {cids.tolist()}")

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
            raise ValueError(f"market {slug!r}: mixed condition_ids {cids.tolist()}")
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

    Writes two files under ``directory``:
      * ``<stem>.parquet`` — per-trade columns (Y, delta, log_size_ratio,
        wallet_ids, t, p, S).
      * ``<stem>.meta.json`` — scalar metadata (S_bar, condition_id, slug).

    Args:
        market: Processed market to persist.
        directory: Destination directory; created (including parents) if
            it does not exist.
        name: File stem override; defaults to ``market.slug`` or
            ``market.condition_id`` when not provided.

    Returns:
        Path of the Parquet file written.
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
    """Persist a wallet index as a JSON ``{address: id, …}`` mapping.

    Args:
        index: WalletIndex to serialise.
        path: Destination file path; parent directories are created if
            they do not exist.

    Returns:
        Resolved Path of the JSON file written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.address_to_id))
    return path


def load_wallet_index(path: str | Path) -> WalletIndex:
    """Load a WalletIndex previously saved with ``save_wallet_index``.

    Args:
        path: Path to the JSON file produced by ``save_wallet_index``.

    Returns:
        WalletIndex with the original address-to-id mapping restored.
    """
    data = json.loads(Path(path).read_text())
    return WalletIndex(address_to_id={str(a): int(i) for a, i in data.items()})


def load_processed(parquet_path: str | Path) -> ProcessedMarket:
    """Load a ProcessedMarket previously saved with ``save_processed``.

    Reads ``<stem>.parquet`` for per-trade arrays and the sidecar
    ``<stem>.meta.json`` for scalar metadata. The meta path is derived
    from ``parquet_path`` by replacing the ``.parquet`` suffix.

    Args:
        parquet_path: Path to the Parquet file written by
            ``save_processed``.

    Returns:
        ProcessedMarket with all fields restored from disk.
    """
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
