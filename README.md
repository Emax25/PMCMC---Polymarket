# Polymarket PMCMC: Insider-Trading Detection via Particle Markov Chain Monte Carlo

## 1. Project Context

This is the **independent project** for STAT 31511 (Monte Carlo Simulation), University of Chicago, taught from the lecture notes by Sanz-Alonso and Al-Ghattas.

The deliverables are:
1. A **10-page paper** in the provided LaTeX template (`Monte_Carlo_Simulation.pdf`), titled like a chapter of the lecture notes.
2. A **companion Python notebook** demonstrating the implementation on a real example.

**Chapter focus:** Chapter 9 — Particle Filters. The project goes beyond the lecture notes by developing a full **Particle Markov Chain Monte Carlo (PMCMC)** pipeline, including **Particle Gibbs (PG)** and **Interacting Particle MCMC (iPMCMC)** for a switching state-space model.

**Working title for the paper:** *Particle Markov Chain Monte Carlo* (alternative: *Particle Filters for Switching State-Space Models*).

---

## 2. The Problem

**Goal:** Detect potentially-informed (insider) trades on Polymarket politics markets by treating each trade as an observation in a state-space model and identifying observations that are inconsistent with public-information dynamics.

**Why prediction markets?** Polymarket trades settle on Polygon (a public blockchain), giving us:
- Tick-level trade data with timestamps, prices, and sizes
- Pseudonymous wallet identifiers for every trade (essential for cross-trade linking)
- Open APIs (Gamma API for metadata, CLOB API, Goldsky subgraph for order fills)

**Why politics?** Politics markets have higher volume and stronger asymmetric-information concerns than sports. Sports has parallel betting markets (Vegas) that arbitrage information; politics often has private information (campaign internals, scheduled announcements, donor info) that can be traded on before public release.

**Why a state-space model?** The "true" probability of an event evolves over time as information arrives. We never observe it directly — we only see noisy market prices and trades. This is the canonical setting for hidden Markov / state-space modeling.

---

## 3. The Statistical Model

### 3.1 Notation

For a single Polymarket market, we observe $N$ trades at times $t_1 < t_2 < \cdots < t_N$. For each trade $i$:
- $p_i \in (0, 1)$ — trade price (interpreted as the market's implied probability of YES)
- $S_i > 0$ — trade size in USDC
- $w_i \in \mathcal{W}$ — wallet address of the taker

Define $\Delta_i := t_i - t_{i-1}$ (inter-trade time) and $Y_i := \text{logit}(p_i)$ (logit-space observation).

We model $K$ markets in the same genre (politics) jointly through shared wallet hyperparameters and shared dynamics parameters.

### 3.2 Latent variables

**Per-market** (subscript $k$ omitted):

| Variable | Type | Meaning |
|----------|------|---------|
| $X_{t_i} \in \mathbb{R}$ | continuous | Logit of true public-info probability $\pi_{t_i}$ |
| $V_{t_i} \in \{0, 1\}$ | discrete | Volatility regime: 0 = calm, 1 = news |
| $Z_i \in \{0, 1\}$ | discrete | Insider indicator for trade $i$ |

**Cross-market:**

| Variable | Type | Meaning |
|----------|------|---------|
| $\theta_w \in [0, 1]$ | continuous | Wallet $w$'s baseline insider propensity |

### 3.3 Generative model

**Hierarchical wallet effects** (shared across markets in the genre):

$$\theta_w \overset{\text{i.i.d.}}{\sim} \text{Beta}(a, b), \qquad w \in \mathcal{W}.$$

**Per-market initialization:**

$$X_{t_0} \sim \mathcal{N}(0, s_0^2), \quad V_{t_0} \sim \text{Bernoulli}(\rho_V), \quad Z_0 := 0.$$

**Volatility regime** — homogeneous Markov chain:

$$V_{t_i} \mid V_{t_{i-1}} \sim \text{MarkovChain}\!\left(P^V = \begin{pmatrix} 1 - q_{01} & q_{01} \\ q_{10} & 1 - q_{10} \end{pmatrix}\right).$$

**Latent logit-probability** — Gaussian random walk with regime-switched variance:

$$X_{t_i} \mid X_{t_{i-1}}, V_{t_i} \sim \mathcal{N}\!\left(X_{t_{i-1}},\; \sigma^2_{V_{t_i}} \cdot \Delta_i\right), \qquad \sigma^2_0 \ll \sigma^2_1.$$

**Insider indicator** — wallet- and size-dependent Markov chain:

$$Z_i \mid Z_{i-1}, w_i, S_i, \theta_{w_i} \sim \text{Bernoulli}\!\left(\pi^Z_i\right),$$

$$\text{logit}(\pi^Z_i) = \text{logit}(\theta_{w_i}) + \beta_S \log\!\frac{S_i}{\bar{S}} + \beta_Z \mathbf{1}\{Z_{i-1} = 1\}.$$

The three terms encode: (1) wallet-level baseline trust ($\theta_{w_i}$), (2) trade size effect ($\beta_S > 0$, large trades more suspicious), and (3) insider-cluster persistence ($\beta_Z > 0$).

**Observation model** — size-weighted Gaussian noise in logit space:

$$Y_i \mid X_{t_i}, Z_i, S_i \sim \mathcal{N}\!\left(X_{t_i},\; \frac{\tau^2_{Z_i}}{1 + \gamma \log\!\frac{S_i}{\bar{S}}}\right), \qquad \tau^2_1 \ll \tau^2_0.$$

Informed observations are tighter ($\tau_1 \ll \tau_0$); large trades are more informative regardless of regime ($\gamma > 0$).

### 3.4 Parameter inventory

Let $\phi := (\sigma^2_0, \sigma^2_1, q_{01}, q_{10}, \beta_S, \beta_Z, \tau^2_0, \tau^2_1, a, b)$ be the parameters of interest. The hyperparameter $\gamma$ and initialization variance $s_0^2$ are fixed (with sensitivity analysis).

### 3.5 Quantities of interest

After inference, we extract:
1. **Per-trade insider probability**: $\mathbb{P}(Z_i = 1 \mid \mathcal{D})$ — the headline anomaly score.
2. **Smoothed price track**: $\mathbb{E}[\pi_{t_i} \mid \mathcal{D}]$ vs. observed $p_i$ — visualize divergence.
3. **Wallet rankings**: $\mathbb{E}[\theta_w \mid \mathcal{D}]$ across the genre — suspect wallets.
4. **Regime indicators**: $\mathbb{P}(V_{t_i} = 1 \mid \mathcal{D})$ — to confirm flagged windows aren't just news regimes.

---

## 4. Inference Methodology

### 4.1 Why PMCMC

Pure Gibbs sampling would work for this model (the conditional structure is clean: FFBS for $X$, forward-backward for $V$ and $Z$, conjugate updates for $\theta_w$). However, this would be a Chapter 5 project, not Chapter 9.

**Particle Gibbs** uses a Conditional Sequential Monte Carlo (CSMC) step in place of FFBS for the joint $(X, V, Z)$ trajectory. Conditional on the discrete states, the model for $X$ is linear-Gaussian, so we **Rao-Blackwellize**: each particle stores only $(V_{t_i}, Z_i)$ and an exact Kalman filter for $X_{t_i}$.

**iPMCMC** (Rainforth et al., 2016) addresses **path degeneracy**, the canonical PG failure mode where particles coalesce on the reference trajectory. It runs $M$ SMC chains in parallel ($P$ conditional, $M-P$ unconditional) and lets conditional nodes swap their references with unconditional trajectories based on marginal-likelihood estimates. Better mixing, suitable for parallelization.

### 4.2 Algorithms to implement

1. **Bootstrap SMC** (sanity check) — `src/inference/smc.py`
2. **Conditional SMC** — `src/inference/csmc.py`
3. **Particle Gibbs** — `src/inference/particle_gibbs.py`
4. **iPMCMC** — `src/inference/ipmcmc.py`
5. **Rao-Blackwellization via Kalman filter** — `src/inference/kalman.py`
6. **Parameter updates** (Gibbs/MH for $\phi$ and $\theta_w$) — `src/inference/parameter_updates.py`

### 4.3 Key references

- Andrieu, Doucet, Holenstein (2010) — *Particle Markov Chain Monte Carlo Methods* (JRSS-B). Foundational paper for PG.
- Rainforth, Naesseth, Lindsten, Paige, van de Meent, Doucet, Wood (2016) — *Interacting Particle Markov Chain Monte Carlo* (ICML). The iPMCMC paper.
- Lindsten, Jordan, Schön (2014) — *Particle Gibbs with Ancestor Sampling* (JMLR). Optional extension.
- Doucet, de Freitas, Gordon (2001) — *Sequential Monte Carlo Methods in Practice*. Standard SMC reference.
- Douc, Cappé, Moulines (2005) — *Comparison of Resampling Schemes for Particle Filtering*. ISPA 2005. Formal proof that systematic resampling dominates multinomial in variance; supports adaptive ESS-based triggering.

---

## 5. Directory Structure

```
polymarket_pmcmc/
├── README.md                       # This file
├── requirements.txt
├── config/
│   └── default_params.py           # ModelParams + InferenceConfig dataclasses
├── data/
│   ├── raw/                        # Raw API pulls (Parquet/CSV)
│   ├── processed/                  # Cleaned trade data
│   └── synthetic/                  # Generated test data
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── transforms.py           # logit, sigmoid, log1pexp, log-weight ops
│   ├── data/
│   │   ├── __init__.py
│   │   ├── polymarket_api.py       # Gamma + Data API clients (see §16.1–16.2)
│   │   ├── preprocess.py           # Cleaning, wallet indexing, ProcessedMarket
│   │   └── synthetic.py            # Generate from the model; insider injection
│   ├── model/                      # (placeholder — generative code lives in data/synthetic.py)
│   │   ├── __init__.py
│   │   ├── ssm.py
│   │   └── params.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── kalman.py               # Kalman filter for X | V, Z (RBPF core)
│   │   ├── smc.py                  # Bootstrap SMC
│   │   ├── csmc.py                 # Conditional SMC (PG engine)
│   │   ├── particle_gibbs.py       # Vanilla PG sampler
│   │   ├── ipmcmc.py               # iPMCMC sampler with swap step
│   │   ├── parameter_updates.py    # Gibbs/MH for hyperparameters
│   │   └── diagnostics.py          # ESS, R-hat, particle-degeneracy metrics
│   └── analysis/
│       ├── __init__.py
│       ├── results.py              # Posterior summaries, wallet rankings, ROC
│       └── plots.py                # All paper figures (matplotlib only)
├── tests/                          # pytest suite (190 tests, ~1 min)
│   ├── fixtures/                   # canned API JSON for offline polymarket_api tests
│   ├── test_csmc.py
│   ├── test_diagnostics.py
│   ├── test_ipmcmc.py
│   ├── test_kalman.py
│   ├── test_parameter_updates.py
│   ├── test_particle_gibbs.py
│   ├── test_plots.py
│   ├── test_polymarket_api.py
│   ├── test_preprocess.py
│   ├── test_results.py
│   ├── test_scripts.py             # end-to-end smoke tests of every CLI
│   ├── test_smc.py
│   └── test_synthetic.py
├── scripts/                        # CLI entrypoints
│   ├── _shortlist.py               # 10-market §5 shortlist (slugs only)
│   ├── _runner.py                  # Shared CLI helpers (config presets, IO)
│   ├── pull_data.py                # Polymarket data pull
│   ├── run_pg.py                   # PG full-pipeline run
│   ├── run_ipmcmc.py               # iPMCMC full-pipeline run
│   └── make_figures.py             # Generate all paper figures from saved chains
├── notebooks/
│   ├── _build_writeup.py           # Builder: regenerates final_writeup.ipynb
│   └── final_writeup.ipynb         # Submission notebook (thin layer over src/)
└── results/
    ├── figures/                    # PDF + PNG outputs for LaTeX
    ├── tables/                     # CSV outputs for LaTeX
    └── chains/                     # Pickled MCMC chains
```

---

## 6. Implementation Phases

Build in this order. Each phase depends on the previous.

| Phase | Component | Depends on |
|-------|-----------|------------|
| 0 | Project setup, requirements, RNG conventions | — |
| 1 | `utils/transforms.py`, `config/default_params.py` | 0 |
| 2 | `data/synthetic.py` (generates from the model) | 1 |
| 3 | `inference/kalman.py` (FFBS for $X$ given $V, Z$) | 1 |
| 4 | `inference/smc.py` (bootstrap, sanity check) | 2, 3 |
| 5 | `inference/csmc.py` (conditional version) | 4 |
| 6 | `inference/parameter_updates.py` (Gibbs/MH steps) | 1 |
| 7 | `inference/particle_gibbs.py` (vanilla PG) | 5, 6 |
| 8 | `inference/ipmcmc.py` (with swap step) | 7 |
| 9 | `inference/diagnostics.py` (ESS, R-hat) | 7 |
| 10 | `data/polymarket_api.py`, `data/preprocess.py` | 0 |
| 11 | `analysis/results.py`, `analysis/plots.py` | 7 |
| 12 | `scripts/run_pg.py`, `scripts/run_ipmcmc.py`, `scripts/make_figures.py` | All |
| 13 | `notebooks/final_writeup.ipynb` | All |

**Validate every phase on synthetic data before moving on.** The synthetic generator is the ground truth.

---

## 7. Coding Conventions

### 7.1 Randomness

- Use `numpy.random.default_rng(seed)` exclusively. Never use the legacy `np.random.*` global state.
- Pass `rng` objects explicitly to every stochastic function. Functions must not call `default_rng()` internally except at the top level (entry point).

### 7.2 Numerical stability

- **All weight operations in log-space.** Use `scipy.special.logsumexp` for normalization. Never store raw weights for $T > \sim 50$.
- **Logit transform is clipped:** `logit(p)` should clip $p$ to $[\epsilon, 1 - \epsilon]$ with $\epsilon \approx 10^{-6}$ to avoid $\pm\infty$.
- For sigmoid of large negative numbers, use `expit` from `scipy.special` (handles overflow).

### 7.3 Vectorization

- The inner SMC loop is over time (sequential) and over particles (parallelizable). **Loops over particles must be NumPy array operations, not Python loops.** This is the single most important performance choice; expect 50–100x speedup over naive loops.
- Loops over time steps remain Python `for` loops — they are inherently sequential.

### 7.4 Persistence

- Pickle (or HDF5) every MCMC chain after running. PMCMC runs are slow; never lose them.
- Save processed data to Parquet for fast reload.
- Cache aggressively in the notebook to keep run-time under 10 minutes.

### 7.5 Optional accelerations

- `numba.njit` on the inner Kalman update and weight computation if hot loops are slow.
- `joblib.Parallel(n_jobs=-1)` for iPMCMC's $M$ chains across CPU cores.

---

## 8. Data Pipeline

### 8.1 Sources

- **Polymarket Gamma API** (`gamma-api.polymarket.com`) — market metadata, list markets by genre. Implementation in `src/data/polymarket_api.py`; several undocumented quirks of this endpoint are recorded in §16.1.
- **Polymarket Data API** (`data-api.polymarket.com`) — trade history, paginated newest-first with a hard cap at `offset=3000` (see §16.2). This is the trade source actually used.
- **Goldsky subgraph** — indexed on-chain order-fill events. Not used by the current pipeline (Data API is sufficient for §5).

Reference: `https://docs.polymarket.com`. Rate limits in practice are loose enough that a 0.1 s sleep between calls suffices.

### 8.2 Genre selection

**Politics markets only**, filtered by:
- Minimum cumulative volume (e.g., $50,000,000 USDC for the §5 shortlist).
- Resolved markets (`closed=true`).
- Topic filter via question-keyword substring match — `tag_slug` is silently ignored by Gamma (§16.1).

The §5 shortlist is the 10 slugs pinned in `scripts/_shortlist.py`: six pre-election outcome markets plus four Trump-specific event markets, all chosen for cross-market wallet overlap (so the hierarchical $\theta_w$ prior has signal to share).

Per-market budget: 500–3000 trades. Because the Data API returns trades newest-first and caps `offset` at 3000, the practical setting is `--tail-trades 2000` (or up to 3000), which gives the last $N$ trades — i.e., the resolution-period price action where insider behaviour is most plausible.

### 8.3 Cleaning

- Drop zero-size or fee-only trades.
- Drop rows with missing wallet or transaction hash.
- Deduplicate on `transaction_hash` (the Data API occasionally double-counts a fill across pages).
- Sort strictly by `(timestamp, transaction_hash)` ascending — hash breaks same-second ties deterministically.
- Compute $\Delta_i$, $\log(S_i / \bar{S})$ where $\bar{S}$ is the within-market mean size, and $Y_i = \text{logit}(p_i)$.
- Build a global wallet index (integer IDs) across all markets; persisted as `wallet_index.json`.

The pipeline is implemented in `src/data/preprocess.py`; the end-to-end CLI is `python -m scripts.pull_data`.

---

## 9. Validation Strategy

Three layers, in order:

1. **Component tests** (`tests/`): each module has unit tests on toy inputs. Kalman recovers truth on linear-Gaussian data; SMC posteriors match analytical answers on small problems.
2. **Synthetic injection test**: generate $K = 5$ synthetic markets with known $Z_i$ ground truth and known insider wallets. Confirm that:
   - $\mathbb{P}(Z_i = 1 \mid \mathcal{D})$ is high for true insider trades (ROC AUC > 0.85 target).
   - $\mathbb{E}[\theta_w \mid \mathcal{D}]$ correctly ranks insider wallets at the top.
3. **Real-data sanity checks**: for flagged trades on real Polymarket data, verify timestamps line up with known news events (qualitative).

---

## 10. Scope Discipline (What NOT to do)

The model and methodology are at the upper limit of a 10-page Ch 9 project. To stay on scope:

- **Do not extend the model further** without explicit reason. The simplifications below are acknowledged in the paper's discussion section, not implemented:
  - i.i.d. $Z$ within a single wallet's trade stream (could be Markov per-wallet)
  - Single insider regime (could be multi-regime)
  - No exogenous covariates (news features, polling data)
  - Constant $\gamma$ (could be learned)
- **Do not put inference code in the notebook.** The notebook is a thin presentation layer over `src/` modules.
- **Do not aim for production code quality.** This is research code: clarity > performance > coverage.
- **Do not pursue ancestor sampling** unless PG/iPMCMC results are complete and there's time.

If something has to be cut due to time pressure, drop in this order:
1. iPMCMC (use vanilla PG only)
2. Wallet hierarchy (do post-hoc wallet analysis from PG output)
3. Multi-market joint inference (run each market independently)

---

## 11. Compute Expectations

Targets on a modern laptop (M-series Mac or 8-core Intel/AMD), well-vectorized NumPy:

- **Single PG iteration**, one market with $T = 1000$, $N = 200$: 0.1–1 second
- **Full PG run** (3000 iterations, 5 markets): 1–8 hours, run sequentially
- **Full iPMCMC run** ($M = 8$, parallelized): 2–10x PG, so 4–40 hours

Plan for an overnight run for the final experiment. Develop on tiny problems ($T = 100$, $N = 50$, 200 iterations) and scale up only after correctness is confirmed.

If the laptop is insufficient, fallback options: Google Colab (free tier, ~hours), UChicago RCC Midway3 (free for students, requires batch job submission), or AWS spot instances ($0.50–$2/hour).

---

## 12. Paper Structure (10 pages)

| Section | Pages | Content |
|---------|-------|---------|
| §1 Introduction | 1 | Problem motivation, why state-space, why PMCMC, roadmap |
| §2 Model | 2 | Setup, generative model, inference target, graphical model figure |
| §3 Particle Gibbs | 2 | CSMC, PG sampler, Rao-Blackwellization (algorithm boxes) |
| §4 iPMCMC | 2 | Path degeneracy, swap step, validity sketch (algorithm box, schematic figure) |
| §5 Polymarket Application | 2.5 | Data, hyperparameters, validation, real results (2 figures, 1 table) |
| §6 Discussion + Bibliography | 0.5 | Limitations, extensions, references (refs don't count) |

---

## 13. Working with This Codebase (for AI Coding Assistants)

If you are an AI coding assistant (Claude Code or similar), please observe:

1. **The model spec in §3 is fixed.** Do not modify the generative model without explicit request from the user. The model was designed iteratively with deliberate trade-offs (tractability vs. realism); changes have downstream consequences for the paper.
2. **The directory structure in §5 is fixed.** Place new files in the appropriate module; do not flatten or reorganize without request.
3. **The implementation order in §6 is the recommended path.** If asked to skip phases, flag the missing dependencies.
4. **Conventions in §7 are non-negotiable.** Code that uses `np.random.*` global state, raw weights without log-space, or per-particle Python loops should be flagged and rewritten.
5. **When extending, prefer minimal changes.** Add tests before adding features. Do not introduce new dependencies without listing them in `requirements.txt`.
6. **Mathematical notation in code should mirror the paper.** Use `X`, `V`, `Z`, `theta_w`, `sigma2_v`, `tau2_z`, etc. — not abbreviations.
7. **The validation strategy in §9 is the source of truth for correctness.** A change is correct iff it passes synthetic injection tests, regardless of how reasonable it looks.
8. **Out-of-scope items in §10 are intentional.** Do not implement them as "nice-to-haves."

If unsure about any design decision, refer back to the conversation history that produced this README, or ask the user.

---

## 14. Design Decisions (Resolved)

These decisions were made interactively and are binding for implementation.

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Particle Kalman state | Each particle carries its own independent $(\mu_i, \sigma^2_i)$ | Different $(V, Z)$ trajectories accumulate different conditional variances; sharing would conflate particles from different regimes |
| 2 | SMC/CSMC proposal for $(V_{t_i}, Z_i)$ | Bootstrap prior for Phase 4 (sanity-check SMC); locally optimal proposal for Phase 5+ (CSMC/PG) | $(V, Z)$ has only 4 joint states — enumerate, weight by $p(Y_i \mid \cdot)$, normalize, sample. Reduces weight variance at tractable cost |
| 3 | Resampling scheme + trigger | Systematic resampling, triggered adaptively when $\text{ESS} < N/2$ | Systematic has strictly lower variance than multinomial at same $O(N)$ cost (Douc et al., 2005); adaptive trigger preserves particle diversity and is required in CSMC to protect the reference trajectory |
| 4 | CSMC reference trajectory | Naive index-pinning; ancestor sampling explicitly deferred | iPMCMC addresses path degeneracy via diverse reference trajectories from unconditional chains — Rainforth et al. (2016) show this dominates PG+ancestor sampling in most settings. Degeneracy in this model is confined to the discrete $(V, Z)$ space (X is marginalized), so iPMCMC's global moves are sufficient. Implementing both would be redundant for a 10-page project |
| 5 | iPMCMC configuration | $M = 8$, $P = 4$ (4 conditional, 4 unconditional); both exposed as config parameters | Equal split $P = M/2$ is the default in Rainforth et al. (2016); maps cleanly onto 8-core laptop via `joblib.Parallel(n_jobs=-1)`. Unconditional chains must be $\geq P$ to generate diverse swap candidates |
| 6 | Parameter update schedule | Conjugate Gibbs for $\sigma^2_0, \sigma^2_1, q_{01}, q_{10}, \theta_w$; MH for $\beta_S, \beta_Z, \tau^2_0, \tau^2_1$; all updated every iteration | Conjugate updates are $O(\|\mathcal{W}\|)$ and cheap; MH on 4 scalars negligible next to the SMC pass. $\theta_w$ is Beta-Bernoulli conjugate given $Z_i$ assignments; transition counts give Beta conjugacy for $q_{01}, q_{10}$; $X$ increments give inverse-Gamma conjugacy for $\sigma^2_v$ |
| 7 | Number of particles $N$ | $N = 50$ for development; $N = 500$ for final runs; uniform across all $M$ iPMCMC chains | Mixing chains with different $N$ corrupts the marginal-likelihood comparisons in the iPMCMC swap step (estimates have different variance). $N = 500$ stays within overnight compute budget per §11 |
| 8 | Multi-market inference | Joint inference across all K markets in the genre. Each outer iteration runs K independent SMC passes (parallel via joblib); the parameter-update step pools Z assignments across markets for the conjugate $\theta_w$​ updates and the a,b hyperprior. Particle state remains size 4 per market — markets are conditionally independent given $\phi$ and $\theta_w$​, so no state-space blow-up.|
| 9 | Burn-in, thinning, chain length | 500 burn-in, no thinning, 3000 total iterations (2500 kept); development: 200 total / 50 burn-in | Thinning never increases ESS per unit compute time (Geyer, 1992) — store all samples and compute ESS directly. If autocorrelation is high, run longer rather than thin |
| 10 | Chain initialization | Moment-matched warm start from one data pass: $\sigma^2_0 = 0.1\cdot\widehat{\text{Var}}(Y)$, $\sigma^2_1 = \widehat{\text{Var}}(Y)$, $\tau^2_0 = \widehat{\text{Var}}(Y)$, $\tau^2_1 = 0.01\cdot\widehat{\text{Var}}(Y)$, $q_{01}=0.05$, $q_{10}=0.5$, $\beta_S=\beta_Z=0$, $a=1$, $b=19$; $\theta_w \sim \text{Beta}(1,19)$ i.i.d. | Cuts burn-in vs. diffuse prior start. $a=1,b=19$ encodes 5% prior mean insider propensity. Drawing $\theta_w$ from the prior (not a single fixed value) avoids a degenerate first CSMC sweep |
| 11 | MH proposals for $\beta_S, \beta_Z, \tau^2_0, \tau^2_1$ | $\beta_S, \beta_Z$: random walk $\mathcal{N}(0, 0.1^2)$; $\tau^2_0, \tau^2_1$: log-normal random walk $\mathcal{N}(0, 0.3^2)$ on log scale with Jacobian correction; updated independently; step sizes in config; adaptive tuning for first 100 dev iterations targeting 23–44% acceptance | Log-normal proposal respects positivity and moves multiplicatively — natural for variance parameters. Block proposals require covariance tuning, overkill for 4 scalars |
| 12 | Fixed hyperparameters $\gamma$, $s_0^2$ | $\gamma = 1.0$, $s_0^2 = 1.0$; sensitivity analysis as two 1D sweeps on synthetic data: $\gamma \in \{0.5, 1.0, 2.0\}$ and $s_0^2 \in \{0.25, 1.0, 4.0\}$, reported as a 2-panel ROC/posterior figure in §5 | $\gamma=1$ means 10x-size trade tightens variance by $\sim3.3\times$ — interpretable default. $s_0^2=1$ spans 27%–73% in probability space, appropriate for markets opening near 50%. Sensitivity on synthetic data only (9 runs, cheap) |
| 13 | Diagnostics | $\hat{R}$ across $P=4$ conditional chains only; flag if $\hat{R} > 1.01$; minimum ESS across $\phi$ components as headline; use `arviz` for both. Particle degeneracy: fraction of steps with particle ESS $< N/4$; flag markets exceeding 10% degeneracy rate | Unconditional chains don't produce $\phi$ / $\theta_w$ samples directly so are excluded from $\hat{R}$. `arviz` handles chain-array format and is standard in this ecosystem |

---

## 15. Status

- [x] Phase 0: Setup
- [x] Phase 1: Utils + config
- [x] Phase 2: Synthetic data generator
- [x] Phase 3: Kalman filter
- [x] Phase 4: Bootstrap SMC
- [x] Phase 5: Conditional SMC
- [x] Phase 6: Parameter updates
- [x] Phase 7: Particle Gibbs
- [x] Phase 8: iPMCMC
- [x] Phase 9: Diagnostics
- [x] Phase 10: Polymarket data pipeline
- [x] Phase 11: Analysis + plots
- [x] Phase 12: Scripts
- [x] Phase 13: Notebook

Update checkboxes as phases complete.

---

## 16. Empirical Notes (Phase 10–13)

Facts discovered while wiring up the real-data pipeline that weren't visible from the API docs or the model on paper. Recorded here so future work doesn't relearn them.

### 16.1 Polymarket Gamma API quirks

Three undocumented behaviours of `gamma-api.polymarket.com/markets` shape how `src/data/polymarket_api.py` queries it:

- **`tag_slug` is silently ignored.** Passing `tag_slug=politics` returns markets from every genre. The reliable topic filter is post-hoc substring matching on the `question` field, exposed as the `question_keywords` parameter on `fetch_markets`. The curated `POLITICS_KEYWORDS` constant in the same module is the §8.2 default.
- **`order=volume` is silently ignored** because the numeric volume column is `volumeNum` (camelCase). Without `order=volumeNum&ascending=false` the endpoint returns markets id-ascending, so naïve `limit=N` calls return the oldest dust markets, not the top-volume ones.
- **`slug=X` on `/markets` returns `[]`.** Single-market lookup uses `/events?slug=X` first (works for newer single-market events); when that returns empty (older multi-market events like the 2024 presidential winner group), `fetch_market_by_slug` falls back to a paginated `/markets?closed=true&order=volumeNum` scan with a post-hoc slug match.

### 16.2 Polymarket Data API offset cap

`data-api.polymarket.com/trades` hard-caps `offset` at 3000 and returns HTTP 400 (`"max historical activity offset of 3000 exceeded"`) past that. **This works in our favour**: the API returns trades newest-first, so the cap means we always get the *last* 3000 trades of any market — exactly the price-action window where insider trading is plausible. `fetch_trades` stops cleanly at `max_offset=3000` to never hit the error, and `--tail-trades N` slices the cleaned, chronologically-sorted result down to the final $N$.

For high-volume markets like Trump 2024 (which has hundreds of thousands of trades total) this is a real ceiling: per-market `T` is bounded by 3000.

### 16.3 Numerical robustness fixes

Two narrow numerical issues surfaced only on real data (synthetic data never triggers either):

- **Log-likelihood floor on `kalman_step`.** Real Polymarket prices can swing from 0.001 to 0.999 within seconds, which the Gaussian observation model assigns vanishing density. Without a floor, `kalman_step` returns `log_lik = -inf` for every particle at such a step, `logsumexp` collapses to `-inf`, and the normalized weights become NaN. The floor `_LOG_LIK_FLOOR = -500` in `src/inference/kalman.py` is well below any realistic value (`exp(-500) ≈ 7×10⁻²¹⁸`); typical posteriors are unaffected.
- **`update_sigma2` masks $\Delta_i = 0$ steps.** Same-second trades have `delta_i = 0`; the model says $X_i = X_{i-1}$ deterministically there, so they carry no information for $\sigma^2$, but the conjugate posterior divides by $\Delta_i$. Without masking those steps the posterior parameters become NaN by iteration 1 on real data. The fix in `src/inference/parameter_updates.py` drops $\Delta_i = 0$ rows from both the count $N_v$ and the sum-of-squares $SS_v$.

Both fixes have dedicated regression tests in the `tests/` suite.

### 16.4 Real-data observations from the §5 shortlist

Patterns from the dev and half-prod runs that are worth knowing before reading the §5 results:

- **Resolved markets pin at 0 or 1** for the final tens of minutes. The Gaussian random-walk model assigns vanishing probability to those discontinuities, so $P(Z_i = 1 \mid \mathcal{D})$ over-flags the resolution period. The §5 paper text should either subset to a pre-resolution window (e.g., final 7 days before close) or acknowledge this misspecification.
- **Wallet posteriors are well-conditioned for wallets with $\geq 100$ trades**, prior-dominated for wallets with $\leq 20$. The `n_trades` column produced by `wallet_ranking()` is the natural filter for "is this finding statistically meaningful?".
- **$\beta_S$ posterior was negative in the dev-mode run** (size apparently *anti*-correlated with insider status). Whether this is a real signal or an MCMC-mixing artefact requires a prod chain to resolve.
- **MCMC budget heuristic for this dataset (10 markets × 2000 trades).** Dev preset ($N=50$, 200 iter) runs in ~22 min; full prod preset ($N=500$, 3000 iter) is roughly $150\times$ that ≈ 55 hours. A practical "half-prod" intermediate (`--n-iter 1500 --n-burnin 300 --n-particles 250`) lands around 8–13 hours and is the recommended overnight target.