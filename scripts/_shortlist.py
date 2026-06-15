"""Genre shortlist: 10 high-volume politics markets used in §5 of the paper.

Pinned by slug. Condition IDs are resolved at pull-time via the Gamma API so
this file stays human-readable and survives Polymarket re-deploys.

Six picks are the same ones recommended in the Phase 11 hand-off (presidential
outcomes, popular-vote, swing-state, RFK third-party, Judy Shelton Fed
nomination). The remaining four are Trump-specific markets chosen to span
diverse insider-information structures:

  * Inauguration — health/death event (different info source than "will win")
  * Epstein-files release — discrete leak announcement
  * Kevin Warsh Fed nomination — companion to Judy Shelton; same private
    decision process, so shared insider wallets should jointly drive both
  * Trump launches a coin pre-election — personal business decision
"""

from __future__ import annotations

# Order matters only for reproducibility — wallet ids assigned in this order.
SLUGS: tuple[str, ...] = (
    "will-donald-trump-win-the-2024-us-presidential-election",
    "will-kamala-harris-win-the-2024-us-presidential-election",
    "will-donald-trump-win-the-popular-vote-in-the-2024-presidential-election",
    "will-the-democratic-candidate-win-pennsylvania-by-1pt5-2pt0",
    "will-trump-nominate-judy-shelton-as-the-next-fed-chair",
    "will-robert-f-kennedy-jr-win-the-2024-us-presidential-election",
    "will-donald-trump-be-inaugurated",
    "will-trump-release-the-epstein-files-by-december-19-771",
    "will-trump-nominate-kevin-warsh-as-the-next-fed-chair",
    "will-trump-launch-a-coin-before-the-election",
)
