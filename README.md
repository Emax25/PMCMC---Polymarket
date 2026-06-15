# Polymarket PMCMC

Independent project for STAT 31511 (Monte Carlo Simulation) at UChicago, built around Chapter 9 (particle filters) from the Sanz-Alonso and Al-Ghattas lecture notes. The deliverables are a ~10-page paper in the course LaTeX template and a companion notebook (`notebooks/final_writeup.ipynb`) that runs the method on real Polymarket data.

The statistical hook: use Particle Markov Chain Monte Carlo (PMCMC), including Particle Gibbs and Interacting Particle MCMC (iPMCMC), to fit a switching state-space model and flag trades that look inconsistent with public-information price dynamics.

---

## The problem

Polymarket politics markets settle on Polygon, so every trade comes with a timestamp, price, size, and pseudonymous wallet address. The question is whether any of those trades look like they were placed with private information.

Politics is a better testbed than sports here. Sports has parallel markets (Vegas) that tend to arbitrage information away; politics often has genuinely private information (campaign internals, scheduled announcements) that can be traded on before it is public.

We never observe the "true" probability of an event. We only see noisy prices and trades as information arrives over time. That is the usual hidden Markov / state-space setup.

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

1. $V$ follows a two-state Markov chain (calm periods vs news spikes).
2. $X$ is a Gaussian random walk whose variance switches with $V$.
3. $Z$ depends on the wallet ($\theta_w$), trade size, and whether the previous trade was flagged.
4. Observations $Y_i$ are Gaussian around $X_{t_i}$, with tighter noise when $Z_i = 1$ (informed trade) and when size is large.

After inference we care about $\mathbb{P}(Z_i = 1 \mid \mathcal{D})$ per trade, wallet rankings via $\mathbb{E}[\theta_w \mid \mathcal{D}]$, and whether flagged windows line up with news regimes rather than ordinary volatility.

Full notation and equations are in the paper (`Monte_Carlo_Simulation/`) and in [agent_reference/ARCHITECTURE.md](agent_reference/ARCHITECTURE.md).

---

## Inference

Plain Gibbs sampling would work (the conditionals are mostly conjugate), but that would be a Chapter 5 project, not Chapter 9.

Instead we use Particle Gibbs: a Conditional SMC step samples the joint $(X, V, Z)$ trajectory. Because $X$ is linear-Gaussian given the discrete states, we Rao-Blackwellize it with a Kalman filter inside each particle. Only $(V, Z)$ are stored per particle.

iPMCMC (Rainforth et al., 2016) runs several SMC chains in parallel and swaps reference trajectories to fight path degeneracy, the usual PG failure mode where all particles collapse onto one path. The implementation parallelizes across chains with `joblib`.

Core modules:

| Module | Role |
|--------|------|
| `src/inference/kalman.py` | Kalman filter for $X \mid V, Z$ |
| `src/inference/smc.py` | Bootstrap SMC (sanity check) |
| `src/inference/csmc.py` | Conditional SMC (PG engine) |
| `src/inference/particle_gibbs.py` | Vanilla PG sampler |
| `src/inference/ipmcmc.py` | iPMCMC with swap step |
| `src/inference/parameter_updates.py` | Gibbs / MH for hyperparameters |

Key references: Andrieu, Doucet & Holenstein (2010) for PG; Rainforth et al. (2016) for iPMCMC; Doucet, de Freitas & Gordon (2001) for SMC.

---

## Repository layout

```
polymarket_pmcmc/
├── config/default_params.py      # ModelParams, InferenceConfig
├── src/
│   ├── data/                     # API clients, preprocessing, synthetic data
│   ├── inference/                # SMC, CSMC, PG, iPMCMC, Kalman, diagnostics
│   ├── analysis/                 # Posterior summaries and plots
│   └── utils/                    # logit, log-weight helpers
├── scripts/                      # CLI: pull_data, run_pg, run_ipmcmc, make_figures
├── notebooks/final_writeup.ipynb # Submission notebook (thin layer over src/)
├── tests/                        # pytest suite (~190 tests)
└── results/                      # chains, figures, tables
```

The notebook is presentation only. Inference lives in `src/`.

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

pytest   # run tests
```

Development preset uses $N = 50$ particles and 200 iterations (~minutes). Production preset uses $N = 500$, 3000 iterations, and can take many hours on a laptop. A practical overnight target is half-prod: `--n-iter 1500 --n-burnin 300 --n-particles 250`.

---

## Data notes

Trade history comes from the Polymarket Data API (`data-api.polymarket.com`), paginated newest-first with a hard cap at `offset=3000`. That cap is actually useful: we mostly want the last few thousand trades near resolution, where insider behavior is most plausible.

Market metadata uses the Gamma API. A few quirks matter in practice (`tag_slug` is ignored; use keyword filtering on the question text; `order=volume` should be `order=volumeNum`). Details are in [agent_reference/ARCHITECTURE.md](agent_reference/ARCHITECTURE.md).

The §5 shortlist is ten politics slugs pinned in `scripts/_shortlist.py`, chosen for cross-market wallet overlap so the hierarchical $\theta_w$ prior has something to learn from.

---

## Validation

1. Unit tests on each module (`tests/`).
2. Synthetic injection: generate markets with known insider trades and wallets; target ROC AUC > 0.85 on $\mathbb{P}(Z_i = 1 \mid \mathcal{D})$.
3. Real data: check whether flagged trades line up with known news (qualitative).

If a change passes unit tests but fails the synthetic injection test, it is wrong.

---

## Status

All implementation phases (setup through notebook) are complete. Active work is documented in [agent_reference/STATUS.md](agent_reference/STATUS.md): speed (`numba`, `joblib`), pre-resolution filtering, half-prod inference runs for the paper, and a few numerical fixes discovered on real data.

For design decisions, API quirks, compute budgets, and coding conventions, see [agent_reference/ARCHITECTURE.md](agent_reference/ARCHITECTURE.md).

---

## Scope

This is a 10-page course project, not production software. The model already sits at the upper edge of what fits in that space. Deliberately out of scope: ancestor sampling, news covariates, multi-regime insiders, and putting inference code in the notebook.

If time runs short, cut in this order: iPMCMC (PG only), wallet hierarchy, then multi-market joint inference.
