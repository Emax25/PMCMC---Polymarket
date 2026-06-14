# Changes since project setup

A focused log of design + implementation changes made during Phase 10–13 work
that aren't already covered by `git log` or the README narrative. New
discoveries about the live Polymarket API and the model's numerical edge
cases live here so future contributors don't have to rediscover them.

For the full system overview see [README.md](README.md). For the
empirically-derived rationale of these changes see README §16.

---

## Phase 10 — Polymarket data pipeline

**Modules added**

- [src/data/polymarket_api.py](src/data/polymarket_api.py) — Gamma + Data API
  clients with retry/backoff, `MarketMeta` and `RawTrade` dataclasses,
  `fetch_markets` / `fetch_market_by_slug` / `fetch_trades`.
- [src/data/preprocess.py](src/data/preprocess.py) — `WalletIndex`,
  `ProcessedMarket`, cleaning + feature computation, Parquet + JSON
  persistence (`save_processed` / `load_processed` /
  `save_wallet_index` / `load_wallet_index`).
- [tests/fixtures/](tests/fixtures/) — canned Gamma + Data API JSON for
  offline testing.

**Polymarket API quirks discovered during the live smoke test** (full
description in README §16.1–16.2):

- `tag_slug=politics` is silently ignored — we filter on question keywords
  instead, via the new `question_keywords` parameter and the
  `POLITICS_KEYWORDS` constant.
- `order=volume` is silently ignored — Gamma's numeric column is
  `volumeNum` (camelCase). Default updated.
- `volume_num_min=X` is a working server-side filter; passed when
  `min_volume > 0`.
- `/markets?slug=X` returns `[]`. `fetch_market_by_slug` now uses
  `/events?slug=X` as a fast path with a paginated `/markets` fallback for
  older multi-market events.
- `/trades?offset>=3000` returns HTTP 400. Because trades come back
  newest-first, this is a feature not a bug — `fetch_trades` stops cleanly
  at `max_offset=3000` and we always get the final 3000 trades of a
  market.

## Phase 11 — Analysis and plots

**Modules added**

- [src/analysis/results.py](src/analysis/results.py) — posterior summaries
  uniform over PG and iPMCMC outputs: `posterior_Z_probability`,
  `posterior_pi_mean`, `posterior_regime_probability`, `wallet_ranking`,
  `summarize_chain`, plus `roc_auc` / `roc_curve` for §9 synthetic
  validation.
- [src/analysis/plots.py](src/analysis/plots.py) — matplotlib-only paper
  figures: single-panel (`plot_price_track`, `plot_z_posterior`,
  `plot_regime_posterior`, `plot_wallet_ranking`, `plot_roc`,
  `plot_parameter_trace`, `plot_parameter_density`) and multi-panel
  composites (`figure_market_overview`, `figure_chain_diagnostics`,
  `figure_synthetic_validation`).

## Phase 12 — CLI entrypoints

**Scripts added**

- [scripts/_shortlist.py](scripts/_shortlist.py) — the 10-market §5
  shortlist as a pinned tuple of slugs. Six pre-election outcome markets
  (Trump / Harris / Trump popular vote / PA Dem 1.5–2.0% / Judy Shelton Fed
  chair / RFK Jr) plus four Trump-specific event markets (inauguration,
  Epstein-files release, Kevin Warsh Fed chair, coin launch).
- [scripts/_runner.py](scripts/_runner.py) — shared helpers: `dev` and
  `prod` config presets, `RunInputs`, real-vs-synthetic loading, pickle
  persistence (`pickle_run` / `load_run`).
- [scripts/pull_data.py](scripts/pull_data.py) — Gamma + Data API pull for
  the shortlist, with `--tail-trades` and `--max-pages` budget controls.
- [scripts/run_pg.py](scripts/run_pg.py),
  [scripts/run_ipmcmc.py](scripts/run_ipmcmc.py) — sampler entrypoints,
  both supporting `--synthetic` for §9 validation runs.
- [scripts/make_figures.py](scripts/make_figures.py) — load a pickled
  chain, produce every figure + CSV table under `results/`.

## Phase 13 — Submission notebook

- [notebooks/_build_writeup.py](notebooks/_build_writeup.py) — programmatic
  builder for the notebook (keeps the source under version control as
  readable Python prose). Regenerate with `python -m notebooks._build_writeup`.
- [notebooks/final_writeup.ipynb](notebooks/final_writeup.ipynb) — 26-cell
  presentation companion to the paper. Cold-run is <2 minutes (cached
  thereafter). Real-data section auto-skips with a helpful notice when
  `data/processed/` is empty.

---

## Bug fixes surfaced by the live-API work

Two narrow numerical issues never triggered by synthetic data but reliably
broke real-data runs. Both have dedicated regression tests in the suite.

### `kalman_step` log-likelihood floor

- **File**: [src/inference/kalman.py](src/inference/kalman.py)
- **Change**: cap `log_lik` at `_LOG_LIK_FLOOR = -500` before returning.
- **Why**: Real Polymarket prices can jump from 0.001 to 0.999 within
  seconds. The Gaussian observation model assigns essentially zero density
  to such jumps, so `innov² / S` overflows to `+inf` and `log_lik` to
  `-inf` for every particle. `logsumexp` then collapses to `-inf` and
  normalized weights become NaN.
- **Test**: existing SMC / CSMC suites all continue to pass; real-data
  PG no longer NaN's at iteration 1.

### `update_sigma2` masks $\Delta_i = 0$ steps

- **File**: [src/inference/parameter_updates.py](src/inference/parameter_updates.py)
- **Change**: drop steps where `delta_i = 0` from both `N_v` and `SS_v`.
- **Why**: Real Polymarket data has many same-second trades. The model
  says $X_i = X_{i-1}$ deterministically there, so no info for $\sigma^2$,
  but the previous code divided by `delta_i = 0` and produced NaN
  posterior parameters that propagated through the chain.
- **Test**: `tests/test_parameter_updates.py::test_update_sigma2_handles_delta_zero_steps`.

---

## Status snapshot at end of session

- **190 unit tests pass** (`python -m pytest tests/ -q`).
- **`data/processed/`** contains all 10 §5 markets, 2000 trades each,
  15,528 unique wallets in `wallet_index.json`.
- **`results/chains/pg_dev.pkl`** — 22-minute PG dev run on real data;
  used by `scripts.make_figures` to populate `results/figures/` and
  `results/tables/`.
- **Half-prod run** (`results/chains/pg_halfprod.pkl`, ~8–13 h) is the
  recommended next sampler run for §5 paper figures; full prod (~55 h on
  this dataset) is overkill unless the half-prod posteriors look noisy.
