Title
Two options, both chapter-style:

"Particle Markov Chain Monte Carlo" — clean, broad, fits the chapter-title convention
"Particle Filters for Switching State-Space Models" — narrower, more descriptive

I'd lean toward the first for its breadth (gives you room to include both PG and iPMCMC under one banner). The application — Polymarket — should be in the abstract/introduction, not the title.

Section-by-section outline (10 pages total)
§1 Introduction (≈1 page)

Motivating problem: information asymmetry in prediction markets; why insider trading is plausible on Polymarket politics; what existing approaches do (briefly).
Why a state-space model: prices noisy, true probability latent and time-evolving.
Why naive Particle Gibbs is insufficient: preview the path-degeneracy issue you'll demonstrate.
Contribution / roadmap: model in §2, methodology in §3–4, empirical results in §5.

Style note: Keep crisp. No equations here.
§2 Model (≈2 pages)

§2.1 Notation and setup — trades, timestamps, prices, sizes, wallets.
§2.2 Generative model — the full spec we wrote down (latent Xt,Vt,ZiX_t, V_t, Z_i
Xt​,Vt​,Zi​; hierarchical θw\theta_w
θw​; size- and persistence-dependent ZZ
Z transition; size-weighted observation noise). Display each component in numbered equations.
§2.3 Inference target — full posterior plus the four quantities of interest (per-trade P(Zi=1)\mathbb{P}(Z_i = 1)
P(Zi​=1), smoothed price track, wallet rankings, regime indicators).

Use one Example environment to illustrate the model on a small toy case (e.g., a 50-trade synthetic market with one injected insider burst). This grounds the reader before §3.
A small graphical model figure (≈1/4 page) showing the dependence structure between X,V,Z,Y,θX, V, Z, Y, \theta
X,V,Z,Y,θ is worth its weight.
§3 Particle Gibbs (≈2 pages)
Most of vanilla PMCMC will be in the lecture notes by the time you submit, so reference Ch 9 for the bootstrap SMC algorithm and keep this focused on what's specific to your model.

§3.1 Conditional SMC for the switching SSM — explain CSMC, give it as an Algorithm box. Highlight the discrete-state proposal: each particle samples (Vti,Zi)(V_{t_i}, Z_i)
(Vti​​,Zi​) from the prior and keeps an exact Kalman filter for XX
X.
§3.2 Particle Gibbs sampler — full algorithm box. CSMC step + Gibbs steps for parameters and wallet effects.
§3.3 Rao-Blackwellization — explain the trick, justify why it dramatically reduces particles needed. Give the explicit Kalman update formulas.

Use the Theorem environment to state validity (target invariance) and just cite Andrieu, Doucet, Holenstein (2010) for the proof — don't re-prove.
§4 Interacting Particle MCMC (≈2 pages)

§4.1 Path degeneracy in PG — informal explanation of why CSMC tends to coalesce particles onto the reference trajectory at early times. Make the issue concrete; it motivates everything that follows.
§4.2 The iPMCMC algorithm — pool of MM
M samplers, PP
P conditional and M−PM-P
M−P unconditional, swap step based on marginal-likelihood estimates. Algorithm box.
§4.3 Why the swap step preserves the target — high-level sketch (state Theorem 1 from Rainforth et al. 2016, cite for proof).

A small figure schematically showing the swap mechanism (conditional and unconditional nodes, swap arrows) helps a lot; this is the single most novel piece for the reader.
§5 Application: Polymarket Politics (≈2.5 pages)

§5.1 Data — which markets, time period, number of trades, source (Polymarket Data API + Goldsky subgraph). One sentence on data cleaning.
§5.2 Hyperparameters — table of values, brief justification for each (calibrated from data, set by prior, etc.).
§5.3 Validation: synthetic insider injection — generate trades from your model with known ZZ
Z, confirm filter recovers them. Report ROC-style detection rates.
§5.4 Real-data results:

Figure: filtered πt\pi_t
πt​ vs observed price for one flagged market, with high-P(Z=1)\mathbb{P}(Z=1)
P(Z=1) trades highlighted.
Figure: ESS curves for vanilla PG vs iPMCMC across iterations — the empirical headline.
Table: top-10 wallets by posterior E[θw∣D]\mathbb{E}[\theta_w \mid \mathcal{D}]
E[θw​∣D], with their cross-market trade volume.
Brief qualitative check: do flagged trades line up with known news events?



This section is where the originality points come from. Be concrete.
§6 Discussion and Bibliography (≈0.5 page)

2–3 sentences summarizing what worked.
2–3 sentences on limitations (pull from the simplifications list — i.i.d. ZZ
Z within a wallet trade-stream, single insider regime, etc.).
2–3 sentences on extensions (Markov ZZ
Z at the wallet level, ancestor sampling, exogenous news features).
References list (doesn't count toward page limit — be generous; 8–15 entries is appropriate).


Figure budget (aim for 4 figures total)

Graphical model of the SSM (§2) — small, schematic.
iPMCMC swap mechanism schematic (§4) — small, conceptual.
Filtered πt\pi_t
πt​ vs observed price with flagged trades (§5) — half-page, color, annotated.
ESS comparison: PG vs iPMCMC (§5) — half-page, two curves.

Resist adding more — figures eat space fast and the cap is strict.

Algorithm boxes (aim for 3)

Conditional SMC (§3.1)
Particle Gibbs full sampler (§3.2)
iPMCMC (§4.2)

You can reference Algorithm 1 of the lecture notes (bootstrap SMC) without re-stating it.

References (target 10–14 entries)
Mandatory:

Andrieu, Doucet, Holenstein (2010) — Particle MCMC
Rainforth et al. (2016) — iPMCMC
Lindsten et al. (2014) — Ancestor sampling (mention as future work)
Doucet, de Freitas, Gordon (2001) book or similar SMC primary reference

Strongly recommended:

Cappé, Moulines, Rydén (2005) — Inference in HMMs (for the switching SSM literature)
Smith & Naik (early prediction market efficiency literature)
Wolfers & Zitzewitz (2004) — prediction markets survey
Almgren or other market-microstructure reference

Application-specific:

A paper or two on insider-trading detection in financial markets (the closest analog literature)
Polymarket / Kalshi documentation as cited URLs


Practical notes

Compress §3 if Ch 9 covers PG in depth — every page you save there goes to §5, which is where your originality lives.
Write §5 first. Empirical results are the hardest to fake at the last minute and the easiest to scope-cut from. Get them done early; prose can be written around them.
Do not re-derive standard results. Cite the lecture notes liberally.
Tight prose discipline. The notes' style is dense — short sentences, numbered equations, minimal connective tissue. Write that way and 10 pages goes further than you'd think.