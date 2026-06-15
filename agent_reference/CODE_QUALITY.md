# Code Quality Guide

> **Style and quality standard for all Python in this repository.**
> Read alongside [ARCHITECTURE.md](ARCHITECTURE.md) §13 (Coding Conventions) and
> [STATUS.md](STATUS.md). This file is the authority on *how code is written*;
> ARCHITECTURE.md is the authority on *what the code does*.

All new and edited Python must satisfy every rule below. When touching an existing
file, leave it at least as clean as you found it.

---

## 1. PEP 8 Baseline

Follow [PEP 8](https://peps.python.org/pep-0008/). The non-negotiables:

| Rule | Standard |
|------|----------|
| Indentation | 4 spaces, never tabs |
| Line length | ≤ 88 chars (Black default); wrap with parens, not `\` |
| Blank lines | 2 between top-level defs, 1 between methods |
| Naming | `snake_case` functions/vars, `CapWords` classes, `UPPER_CASE` constants |
| Module-private | Prefix with a single underscore (`_LOG_LIK_FLOOR`, `_runner.py`) |
| Whitespace | No trailing whitespace; one space around binary operators |
| Comparisons | `is None` / `is not None`; never `== None` |
| Strings | Prefer double quotes; be consistent within a file |

**Math-symbol exception:** Variable names may mirror the model's mathematical
symbols (`X`, `V`, `Z`, `Y`, `sigma2_0`, `theta_w`, `beta_S`) even though they
break `snake_case`. Matching the paper's notation aids correctness review and is
preferred over "PEP-8-pure" but unrecognizable names. This is the *only* allowed
deviation from the naming rules.

**Tooling:** Format with `black`, lint with `ruff` (or `flake8`). Code should pass
both before commit. Do not hand-fight the formatter.

---

## 2. Docstrings — Google Style, Always

**Every** public module, class, function, and method must have a docstring.
Private helpers (`_name`) must have at least a one-line docstring stating intent.

Use the [Google docstring style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Function / method template

```python
def update_sigma2(
    markets: list[MarketLatents],
    rng: np.random.Generator,
    *,
    alpha_prior: float = 2.0,
    beta_prior: float = 1.0,
) -> tuple[float, float]:
    """Inverse-Gamma conjugate update for (sigma2_0, sigma2_1).

    Pools sufficient statistics across all markets (decision #8) and draws
    each regime variance from its Inv-Gamma posterior.

    Args:
        markets: Per-market latent states; statistics are summed over the list.
        rng: Source of randomness. Pass explicitly — never use global NumPy RNG.
        alpha_prior: Inverse-Gamma shape hyperparameter (keyword-only).
        beta_prior: Inverse-Gamma scale hyperparameter (keyword-only).

    Returns:
        The sampled ``(sigma2_0, sigma2_1)`` pair.

    Raises:
        ValueError: If every market has only ``delta == 0`` steps, leaving the
            update with no sufficient statistics.
    """
```

### Rules

- **Summary line:** One imperative sentence, ends with a period, fits on one line.
- **Sections:** Use `Args:`, `Returns:`, `Raises:`, `Yields:` (for generators),
  and `Note:` where relevant. Omit a section only when it does not apply
  (e.g. no `Args:` for a no-arg function).
- **Math models:** Restate the equation the function implements in the docstring
  body, in plain ASCII (see `kalman.py`). Reviewers verify code against math.
- **Module docstrings:** Open every file with a docstring explaining the module's
  role and any model equations it realizes.
- **Don't restate types** that the signature already gives; describe *meaning,
  shape, and constraints* instead (e.g. "shape `(N, T)`, log-space weights").

---

## 3. Comments — Explain *Why*, Not *What*

Implemented functions and any non-obvious logic must carry detailed comments that
explain **intent, trade-offs, and constraints** — the things the code cannot say
for itself.

**Do** comment:

```python
# Floor on per-particle log predictive density. Real Polymarket prices can
# swing 0.001 -> 0.999 within seconds, which the Gaussian observation model
# rules out, sending log_lik to -inf and collapsing logsumexp to NaN. exp(-500)
# is negligible (~7e-218) so the floor never perturbs the posterior on real data.
_LOG_LIK_FLOOR = -500.0
```

**Don't** narrate the obvious:

```python
# BAD — adds nothing the code doesn't already say
i = i + 1          # increment i
return result      # return the result
```

- Comment surprising numerical edge cases (NaN guards, variance floors,
  `delta == 0` handling). These have bitten this codebase before — see
  ARCHITECTURE.md §6.1.
- Keep comments truthful and current. A wrong comment is worse than none — update
  it when the code changes.
- Use section separators for long modules, matching the existing style:
  `# ---------------- Conjugate updates ----------------`.

---

## 4. Standardized Imports

Imports are grouped into three blocks, **separated by one blank line**, in this
fixed order. This matches every module in `src/`.

```python
"""Module docstring first."""
from __future__ import annotations          # always immediately after docstring

import math                                  # 1. Standard library
from dataclasses import dataclass, replace

import numpy as np                           # 2. Third-party
from scipy.special import logsumexp

from config.default_params import ModelParams   # 3. First-party (this repo)
from src.utils.transforms import logit, log1pexp
```

### Rules

1. `from __future__ import annotations` is the **first statement after the module
   docstring** in every file (enables cheap forward-referenced type hints).
2. **Three groups, in order:** standard library → third-party → first-party.
   One blank line between groups, none within.
3. Within a group, sort alphabetically; `import x` lines before `from x import y`.
4. **Absolute imports only** from the project root (`from src.inference.kalman
   import kalman_step`). No relative imports (`from ..utils import ...`).
5. **No wildcard imports** (`from x import *`).
6. Import at module top level, not inside functions — *except* to break a genuine
   circular import or to defer a heavy optional dependency, which must be
   commented.
7. Canonical aliases only: `import numpy as np`, `import pandas as pd`,
   `import matplotlib.pyplot as plt`. Do not invent new aliases.

`ruff`/`isort` enforces ordering automatically; configure once and let it run.

---

## 5. Small Helper Functions Over Repetition

Prefer many small, single-purpose functions to long blocks or copy-pasted logic.

- **DRY:** If the same logic appears (or nearly appears) a second time, extract a
  helper. Two is a coincidence; three is a refactor you already owed.
- **Single responsibility:** A function does one thing describable in its summary
  line without "and". Split when you reach for "and".
- **Size:** Aim for functions that fit on one screen (~≤ 40 lines). Longer is a
  smell, not a crime — break out named helpers when a block needs its own comment
  to be understood.
- **Naming as documentation:** A well-named helper (`process_variance`,
  `obs_variance`) often replaces an explanatory comment entirely.
- **Privacy:** Module-internal helpers get a leading underscore. Promote to public
  (and document fully) only when something outside the module needs them.
- **Pure where possible:** Prefer functions that take inputs and return outputs
  over ones that mutate shared state — they are easier to test, reuse, and
  parallelize (important for the joblib work in [STATUS.md](STATUS.md)).

When a helper would only ever be called once and hurts readability by hiding the
flow, inline it. Judgment over dogma.

---

## 6. Performance & Complexity

Speed is the project's standing P0 (see ARCHITECTURE.md §7). Write for performance
by default, without sacrificing the clarity rules above.

### Priorities

1. **Pick the right complexity first.** Choose the algorithm/data structure that
   gives the best practical big-O for the expected input size *before* micro-
   optimizing. Note the complexity in the docstring when it is non-trivial.
2. **Vectorize over the particle axis.** Operate on whole NumPy arrays; never loop
   in Python over the `N`-particle dimension. The particle dimension is the inner
   hot loop (`O(iterations × M × K × T × N × 4)`).
3. **Log-space arithmetic.** Combine weights/likelihoods with `logsumexp`, not by
   multiplying probabilities — both for numerical stability and to stay vectorized.
4. **`numba.njit` the hot path.** `kalman_step` and similar inner kernels are
   numba targets. Keep them as plain array math (no Python objects, no dataclass
   access inside the jitted core) so they stay jit-compatible.
5. **Parallelize the embarrassingly parallel.** Independent chains (M) and markets
   (K) go through `joblib.Parallel`. Keep inference kernels callable *outside* the
   MCMC wrapper so they parallelize cleanly and can be reused for live trading (P6).
6. **Avoid needless allocation/copies** in loops. Preallocate output arrays;
   reuse buffers; prefer in-place ops (`out=`) on hot paths where it stays correct.

### Discipline

- **Profile before optimizing.** Use `cProfile` / `py-spy` on one dev iteration;
  optimize the measured bottleneck, not the guessed one.
- **No speed regressions.** A change to a hot path must not slow the dev-scale
  benchmark (`scripts/benchmark.py`, once it exists). See ARCHITECTURE.md §12.
- **Readability still wins on cold paths.** Setup, I/O, plotting, and CLI code are
  not hot — keep them simple and skip micro-optimizations there.
- **Comment any non-obvious optimization** so a future reader doesn't "simplify"
  it back into a slow or NaN-producing version.

---

## 7. Pre-Commit Checklist

Before committing Python, confirm:

- [ ] `black` and `ruff` pass with no findings.
- [ ] Imports follow §4 (three groups, `__future__` first, absolute only).
- [ ] Every function/method/class/module has a Google-style docstring (§2).
- [ ] Comments explain *why*, edge cases are noted, no narration of obvious code (§3).
- [ ] No duplicated logic that should be a shared helper (§5).
- [ ] Hot-path code is vectorized / jit-compatible; no Python loop over particles (§6).
- [ ] Type hints on all public signatures.
- [ ] `pytest tests/ -q` passes; new behavior has a test.
