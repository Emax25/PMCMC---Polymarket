# polymarket_pmcmc — agent guide

PMCMC insider-trading detection on Polymarket politics markets. Python; `scripts/` CLIs are the **only** entrypoints.

## Read first (canonical agent docs — do not re-derive from source)
- `agent_reference/ARCHITECTURE.md` — model, inference, module map, conventions
- `agent_reference/STATUS.md` — current focus, roadmap, gates, resolved decisions
- `agent_reference/CODE_QUALITY.md` — style authority for all Python

## Workflow (each stage is a skill; orchestrators delegate reading to subagents)
1. `/plan` — implementation plan (top-tier model session; explorers + critics do the reading)
2. `/implement` — execute plan via `pmcmc-implementer` workers, per-unit commits
3. `/finish` — tests → debug → review → simplify → docs → `/ce-compound`

`/paper` for the LaTeX paper in `Monte_Carlo_Simulation/`. The compound-engineering plugin (`/ce-*`) backs these stages; plans live in `docs/plans/`, captured lessons in `docs/solutions/`.

## Hard rules
- Math-symbol names (`X`, `V`, `Z`, `theta_w`, `beta_S`) are intentional — keep them.
- Sequential (`n_jobs=1`) inference path stays bit-exact; hot paths (numba kalman, particle loops) must not slow down.
- Tests: `python -m pytest -q -m "not slow"` (fast), `python -m pytest -q` (full). Prefer the `test-triager` subagent to keep logs out of context.
- Doc updates follow ARCHITECTURE.md §0 (or dispatch the `docs-updater` subagent).
