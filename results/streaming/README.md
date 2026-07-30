# Streaming scorer — documented replay run

Artifacts for one end-to-end `scripts/score_stream.py` replay over a real
Polymarket market. Replay mode is the no-lookahead evaluation substrate: every
score here is a function of that trade and the ones before it, nothing later.

## Files

| File | What it is |
|------|------------|
| `will-trump-nominate-judy-shelton-as-the-next-fed-chair.trades.jsonl` | 500 real trades, `stream_trades.py` sink shape (one `RawTrade` per line). |
| `will-trump-nominate-judy-shelton-as-the-next-fed-chair.scores.jsonl` | 500 score records, one per trade: `{ts, tx_hash, market, wallet, p_z, p_v, x_mean}`. |

A successful run also writes a `<scores>.jsonl.meta.json` sidecar recording the
run's mode, input path, `--forgetting`, `--n-refresh`, and the warm-start /
wallet-index paths, so a scores file can be traced back to what produced it (the
scores committed here predate the sidecar; re-running the command adds one). The
sidecar is deterministic (no timestamps) and lives beside the scores rather than
inside them, which keeps the JSONL itself byte-identical across identical runs.

## Reproducing

**1. Capture.** The trades are the newest 500 fills on the Judy-Shelton Fed
nomination market, pulled from the Polymarket Data API and written through the
same append-only sink `scripts/stream_trades.py` uses for the live feed (so the
replay input is byte-shaped exactly like a live capture):

```bash
SLUG=will-trump-nominate-judy-shelton-as-the-next-fed-chair
.venv/bin/python -c "
from pathlib import Path
from src.data.polymarket_api import fetch_market_by_slug, fetch_trades
from scripts.stream_trades import JsonlTradeSink
meta = fetch_market_by_slug('$SLUG')
sink = JsonlTradeSink(Path('results/streaming/$SLUG.trades.jsonl'))
for t in fetch_trades(meta.condition_id, page_size=500, max_offset=500):
    sink.append(t)
sink.close()
"
```

A live capture is the same thing without the fetch:
`python -m scripts.stream_trades --output data/live/trades.jsonl --markets <condition_id>`.

**2. Score.** The run committed here:

```bash
SLUG=will-trump-nominate-judy-shelton-as-the-next-fed-chair
.venv/bin/python -m scripts.score_stream \
    --replay results/streaming/$SLUG.trades.jsonl \
    --output results/streaming/$SLUG.scores.jsonl \
    --forgetting 0.98
```

Re-running it is byte-identical (`cmp` clean): replay sorts by
`(timestamp, transaction_hash)` and the scorer draws no randomness.

## What the run produced

* 500 trades, 173 distinct wallets, one market
  (`0x46d40e851b24d9b0af4bc1942ccd86439cae82a9011767da14950df0ad997adf`),
  spanning 2026-03-04 18:26:37 – 19:27:07 UTC.
* `p_z`: max 0.220, mean 0.062, 16 trades above 0.10, none above 0.5.
* `p_v`: 0.084 – 0.932, mean 0.264 — the volatility regime does move, so the
  online-EM adaptation of `(sigma2, tau2, q)` is doing work over the hour.

The insider scores sit near the `Beta(1, 19)` prior mean (0.05) because this run
is **cold-started**: no committed real-data VEM fit exists yet, so
`--warm-start` was omitted and the scorer began from uninformative defaults
(`Var[Y] = 1`, identity centering, empty `theta_w`). Treat `p_z` here as a
smoke-test of the plumbing, not a detection result.

## Warm starting

With a fitted `VEMOutput` in hand, dump the artifact once and point the CLI at
it — `params`, `theta_w` and the centering constants `(m_S, s_S, m_Z)` are all
restored:

```python
import json
from scripts.score_stream import warm_start_payload
json.dump(warm_start_payload(vem), open("results/streaming/warm_start.json", "w"))
```

```bash
.venv/bin/python -m scripts.score_stream --replay <capture>.jsonl \
    --warm-start results/streaming/warm_start.json \
    --wallet-index data/processed/wallet_index.json \
    --output <scores>.jsonl
```

Pass `--wallet-index` alongside it: `theta_w` is indexed by the wallet ids the
fit used, and without that mapping every address is a new id that cold-starts at
the prior mean. `load_warm_start` also reads a pickled `VEMOutput` and a
`scripts/validate_vem.py` JSON artifact, whose `best_restart` block now carries
the same five fields (`params`, `theta_w`, `m_S`, `s_S`, `m_Z`).

A warm start missing the centering constants is **rejected**, not silently
patched: `beta_S`/`beta_Z` are fitted against standardized covariates, so
substituting identity centering would apply them to raw ones and mis-scale every
score. Artifacts written by an older `validate_vem.py` (`params` only) therefore
have to be re-dumped through `warm_start_payload` — unless their betas are zero,
in which case the identity substitution is exact and the CLI only warns.

## Two things to know before building on this

* **`S_bar` is causal.** The batch pipeline divides each trade's size by the
  whole-market mean; streaming cannot see the future, so `score_stream.py` uses
  an expanding mean over trades `0..t`. Scores from a replay therefore do not
  equal batch VEM's filtered marginals on the same market, by construction.
* **`theta_w` adaptation is per market.** The ADF recursion tracks one market's
  price path, so a stream carrying several `condition_id`s gets one independent
  scorer each, and a wallet's learned propensity does not cross between them.
  The batch hierarchy pools over the union of wallets; the streaming path does
  not yet.
