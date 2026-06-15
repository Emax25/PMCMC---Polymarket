# Polymarket PMCMC

This project asks one fairly narrow question of Polymarket politics markets: do any individual trades look like they were placed with information the rest of the market didn't have yet?

To get at that, it fits a switching state-space model to each market's trade history using Particle Markov Chain Monte Carlo (PMCMC) — Particle Gibbs and Interacting Particle MCMC (iPMCMC) — and scores each trade for how inconsistent it looks with ordinary, public-information price dynamics.

The accompanying paper lives in `Monte_Carlo_Simulation/`. Everything runs from the command line via `scripts/`.

---

## The problem

Polymarket politics markets settle on Polygon, so every trade comes with a timestamp, price, size, and pseudonymous wallet address. Given only that, can we pick out trades that look like someone knew something?

Politics turns out to be a better testbed than sports. Sports has parallel markets (Vegas lines) that arbitrage information away quickly. Politics often has genuinely private information — campaign internals, announcements scheduled but not yet public — that someone can trade on before everyone else catches up.

The catch is that we never see the "true" probability of an event. We only see noisy prices and a stream of trades as information arrives. That's the classic hidden-Markov / state-space situation, which is what makes the particle-filtering machinery a natural fit.

---

## The model (short version)

For each market we observe trades at times $t_1 < \cdots < t_N$, with price $p_i$, size $S_i$, and wallet $w_i$. Prices live in logit space: $Y_i = \text{logit}(p_i)$.

Latent quantities per trade:

| Symbol | Meaning |
|--------|---------|
| $X_{t_i}$ | Logit of the true public-information probability |
| $V_{t_i} \in \{0,1\}$ | Volatility regime (calm vs news) |
| $Z_i \in \{0,1\}$ | Insider indicator for trade $i$ |

Across markets, each wallet $w$ gets a baseline insider propensity $\theta_w \sim \text{Beta}(a,b)$.

The generative story in plain terms:

1. $V$ follows a two-state Markov chain — calm stretches versus news spikes.
2. $X$ is a Gaussian random walk whose variance switches with $V$.
3. $Z$ depends on the wallet ($\theta_w$), the trade size, and whether the previous trade was already flagged.
4. Observations $Y_i$ are Gaussian around $X_{t_i}$, with tighter noise when the trade is informed ($Z_i = 1$) and when the size is large.

What we actually want out of inference is $\mathbb{P}(Z_i = 1 \mid \mathcal{D})$ per trade, wallet rankings via $\mathbb{E}[\theta_w \mid \mathcal{D}]$, and a sanity check that flagged windows line up with genuine news regimes rather than ordinary volatility.

The full notation and equations are in the paper (`Monte_Carlo_Simulation/`) and in [agent_reference/ARCHITECTURE.md](agent_reference/ARCHITECTURE.md).

---

## Inference

Most of the conditionals here are conjugate, so plain Gibbs would sample the parameters perfectly well. The hard part is the joint latent trajectory $(X, V, Z)$ — and that's exactly where the particle methods earn their keep.

We use Particle Gibbs: a Conditional SMC step samples the whole $(X, V, Z)$ trajectory at once. Because $X$ is linear-Gaussian given the discrete states, we Rao-Blackwellize it with a Kalman filter inside each particle, so only $(V, Z)$ have to be carried around per particle.

iPMCMC (Rainforth et al., 2016) runs several SMC chains side by side and swaps reference trajectories between them. This fights path degeneracy — the usual Particle Gibbs failure mode where every particle collapses onto a single trajectory. The chains run in parallel via `joblib`.

Core modules:

| Module | Role |
|--------|------|
| `src/inference/kalman.py` | Kalman filter for $X \mid V, Z$ |
| `src/inference/smc.py` | Bootstrap SMC (sanity check) |
| `src/inference/csmc.py` | Conditional SMC (the Particle Gibbs engine) |
| `src/inference/particle_gibbs.py` | Vanilla PG sampler |
| `src/inference/ipmcmc.py` | iPMCMC with the swap step |
| `src/inference/parameter_updates.py` | Gibbs / MH for the hyperparameters |

Key references: Andrieu, Doucet & Holenstein (2010) for PG; Rainforth et al. (2016) for iPMCMC; Doucet, de Freitas & Gordon (2001) for SMC.

---

## Repository layout

```
polymarket_pmcmc/
├── config/default_params.py   # ModelParams, InferenceConfig, presets
├── src/
│   ├── data/                  # API clients, preprocessing, synthetic data
│   ├── inference/             # SMC, CSMC, PG, iPMCMC, Kalman, diagnostics
│   ├── analysis/              # posterior summaries and plots
│   └── utils/                 # logit, log-weight helpers
├── scripts/                   # CLI: pull_data, run_pg, run_ipmcmc, make_figures
├── tests/                     # pytest suite (~190 tests)
├── Monte_Carlo_Simulation/    # the LaTeX paper
└── agent_reference/           # architecture, status, and coding-style docs
```

Data and results directories (`data/processed/`, `results/chains/`, `results/figures/`) are created on the first run. All inference logic lives in `src/`; `scripts/` is just a thin command-line layer over it.

---

## Quick start

```bash
pip install -r requirements.txt

# Pull real Polymarket politics data (10-market shortlist)
python -m scripts.pull_data --output-dir data/processed --tail-trades 2000

# Fast dev run on synthetic data
python -m scripts.run_pg --synthetic --config dev

# Full pipeline (slow; plan for hours to overnight)
python -m scripts.run_ipmcmc --config prod

# Regenerate figures from a saved chain
python -m scripts.make_figures --chain results/chains/pg_dev.pkl

pytest   # run the test suite
```

The `dev` preset uses $N = 50$ particles and 200 iterations and finishes in minutes — good for checking that something runs. The `prod` preset uses $N = 500$ and 3000 iterations and can take many hours on a laptop. In practice the sweet spot is "half-prod": `--n-iter 1500 --n-burnin 300 --n-particles 250`, which is the default target for the paper's chains.

---

## Data notes

Trade history comes from the Polymarket Data API (`data-api.polymarket.com`), paginated newest-first with a hard cap at `offset=3000`. That cap is actually convenient — we mostly care about the last few thousand trades near resolution, which is where insider behavior is most plausible.

Market metadata comes from the Gamma API, which has a few quirks worth knowing: `tag_slug` is silently ignored (so we keyword-filter on the question text instead), and `order=volume` needs to be `order=volumeNum`. The full list is in [agent_reference/ARCHITECTURE.md](agent_reference/ARCHITECTURE.md).

The shortlist is ten politics slugs pinned in `scripts/_shortlist.py`, picked for cross-market wallet overlap so the hierarchical $\theta_w$ prior actually has something to learn from.

---

## Validation

1. Unit tests on each module (`tests/`).
2. Synthetic injection: generate markets with known insider trades and wallets, then check we recover them — target ROC AUC > 0.85 on $\mathbb{P}(Z_i = 1 \mid \mathcal{D})$.
3. Real data: a qualitative check that flagged trades line up with known news.

The synthetic injection test is the one that matters most: if a change passes the unit tests but fails it, the change is wrong.

---

## Status

The core implementation is complete and the method runs end to end on real data. Current work, tracked in [agent_reference/STATUS.md](agent_reference/STATUS.md), is mostly about speed (`numba`, `joblib`), filtering out the post-resolution window that tends to over-flag, running longer half-prod chains for the paper, and a few numerical fixes that only surfaced on live data.

For design decisions, API quirks, compute budgets, and coding conventions, see [agent_reference/ARCHITECTURE.md](agent_reference/ARCHITECTURE.md) and [agent_reference/CODE_QUALITY.md](agent_reference/CODE_QUALITY.md).

---

## Scope

The model is deliberately about as elaborate as it needs to be and no more — the goal is something you can actually fit and reason about, not the most complicated thing possible. A few extensions were left out on purpose: ancestor sampling in the CSMC, explicit news covariates, and more than two insider regimes.

A real-time trading signal built on the same inference kernels is a plausible future direction (it's on the roadmap in [agent_reference/STATUS.md](agent_reference/STATUS.md)), but it's outside the current scope.
