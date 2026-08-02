# Case study: U.S. v. Gannon Ken Van Dyke

An externally labelled insider episode — the only kind of ground truth this project has that it did not plant itself. Read the data-sufficiency section before quoting anything from the wallet ranking.

Manifest: `results/case_studies/van_dyke/markets.json` (schema v1)

**COLD START — THIS RUN IS NOT A RESULT.** The scores provenance records `warm_start: null`. `stream_scoring.cold_start` leaves `theta_w` empty and the logistic coefficients uninformative, and `--n-refresh` defaults to never refreshing them, so every P(Z) is the prior mean plus filter noise and any ranking over it is a ranking of noise. Re-run `score_stream.py --replay` with `--warm-start <fitted VEM artifact>` before reading anything below as evidence about the model.

## 1. Case and sources

**U.S. v. Gannon Ken Van Dyke**

- First U.S. criminal and civil insider-trading action over prediction-market event contracts. A U.S. Army Master Sergeant with access to classified planning for the operation that captured Nicolas Maduro is alleged to have bought 'Yes' shares in Polymarket Venezuela/Maduro contracts in the days before the operation was announced publicly on 2026-01-03.
- United States v. Van Dyke, S.D.N.Y. — indictment unsealed 2026-04-23.
- CFTC v. Gannon Ken Van Dyke, No. 1:26-cv-03369 (S.D.N.Y., complaint filed 2026-04-23).

Sources:

| id | kind | read? | source |
|---|---|---|---|
| `cftc_pr` | primary | yes | CFTC Press Release 9217-26 — CFTC Charges U.S. Service Member with Insider Trading in Nicolas Maduro-Related Event Contracts — <https://www.cftc.gov/PressRoom/PressReleases/9217-26> |
| `cftc_complaint` | primary | yes | CFTC v. Van Dyke, Complaint, No. 1:26-cv-03369 (S.D.N.Y. filed 2026-04-23) — <https://www.cftc.gov/media/13761/EnfGannonKenVanDykeComplaint042326/download> |
| `doj_pr` | primary | **NO** | DOJ — U.S. Soldier Charged With Using Classified Information To Profit From Prediction Market Bets — <https://www.justice.gov/opa/pr/us-soldier-charged-using-classified-information-profit-prediction-market-bets> |
| `doj_indictment` | primary | **NO** | DOJ/SDNY — indictment PDF — <https://www.justice.gov/usao-sdny/media/1437781/dl> |
| `gamma` | primary | yes | Polymarket Gamma API market registry (gamma-api.polymarket.com/events?slug=...) — <https://gamma-api.polymarket.com> |

Claims this manifest could **not** verify:

- The DOJ press release and the SDNY indictment PDF were never read: justice.gov returned HTTP 403 to every request. Everything cited to 'doj_pr' anywhere in this project therefore rests on secondary reporting of it.
- 'Approximately 13 bets from December 27, 2025 through the evening of January 2' — a DOJ figure this manifest could not read at source. The on-chain reconstruction above independently produces exactly 13 buys over exactly that span, which is strong corroboration but is not the same as having read the document.
- '~$33,034 total wagered' — this figure appears in the plan that commissioned the case study and could not be located in any primary document. The CFTC complaint gives ~$32,538 for the January Contract alone (paragraph 52). The reconstruction totals $33,934.34 across all four charged markets. Treat $33,034 as unsourced, most likely a transposition of $33,934.
- The full wallet address is redacted in the CFTC complaint. The address recorded in wallet_anchor is derived (see 'reconstruction'), not quoted from a primary source.
- That the four contracts named in complaint paragraphs 29 and 55 are the *complete* set the charged trader touched. The complaint names four; nothing rules out others it did not name.

Market identification is manual and documented (KTD5): The CFTC complaint names four Polymarket contracts by title (paragraphs 29, 55) and states the resolution outcome of three of them (paragraph 56). Each title was matched against the Gamma market registry, and the match was then cross-checked against an independent fact the complaint supplies: listing date, resolved outcome, or resolution time. No market is listed below whose Gamma record contradicts the complaint, and no market is listed on title similarity alone.

Anchor reconstruction (confirmed, 2026-08-02): The manifest's redacted-address pattern was applied to trades pulled from data-api.polymarket.com for the four charged condition ids over the analysis window (2025-12-26 to 2026-01-03T09:21Z). One wallet matched. Its trades were then compared, without tuning, against every quantity the CFTC complaint itemizes.

- 13 buys across the four charged markets, first at 2025-12-27T05:05:41Z (2025-12-27 00:05 ET), last at 2026-01-03T02:58:25Z (2026-01-02 21:58 ET) — matches the DOJ description of 'approximately 13 bets from December 27, 2025 through the evening of January 2' that this manifest otherwise records as unverified.
- January Contract: 7 buys, 436,760 'Yes' shares, $32,538.34 cost, $0.0745 average — complaint paragraph 52 says 'more than 436,000 Yes shares at an average price of approximately $0.074 for a total cost of approximately $32,538'.
- Purchase schedule matches complaint paragraph 50 line for line: 13,769 + 850 = 14,619 shares in two transactions on the evening of Dec 30 ET ('approximately 14,600 shares across two transactions'); 73,685 on the evening of Jan 1 ET ('73,700'); 90,347 on the morning of Jan 2 ET ('approximately 90,300'); 82,421 + 87,500 + 88,187 = 258,108 in three transactions between 20:38 and 21:58 ET on Jan 2 ('more than 250,000 shares across three separate transactions between 8:30 and 10:00 PM ET').
- Zero trades in the 'Maduro out in 2025?' control market, consistent with the complaint alleging no trading there.
- All 13 trades are BUYs; total outlay $33,934.34 across the four markets.

This reconstruction uses only public trade data plus the redacted pattern the complaint published; it adds no non-public information. It is recorded so a reader can see that the wallet anchoring is an identification, not an assumption. It is NOT independent evidence that the trades were insider trades — that allegation is the government's, and it is untested.

## 2. Markets in the cluster

| slug | role | resolved | scored trades | why |
|---|---|---|---|---|
| `maduro-out-by-january-31-2026-318` | primary | Yes | 9367 | The 'January Contract' of the CFTC complaint and the only contract the complaint quantifies: >436,000 'Yes' shares at an average ~$0.074, ~$32,538 cost, >$404,000 profit (paragraphs 29, 52, 53). |
| `us-forces-in-venezuela-by-january-31-2026` | cluster | Yes | 3349 | One of the three further Venezuela contracts the complaint says Van Dyke bought 'Yes' shares in (paragraph 55); paragraph 56 says he held this one to resolution. |
| `will-the-us-invade-venezuela-by-january-31-2026` | cluster | No | 14772 | Named in complaint paragraph 55; paragraph 56 says the 'Yes' position was sold at a profit on 2026-01-03 before the contract resolved. |
| `trump-invokes-war-powers-against-venezuela-by-january-31-134-583` | cluster | Yes | 2999 | Named in complaint paragraph 55 as 'Trump Invokes War Powers Against Venezuela by January 31, 2026?'; paragraph 56 says the 'Yes' shares were sold at a profit on 2026-01-03. |
| `maduro-out-in-2025-411` | control | No | 47279 | The 'December Contract' of complaint paragraph 28 — same question, expiring 2025-12-31, live throughout the alleged trading window. The complaint does NOT allege any trading in it. It is pulled as a within-cluster negative control: a market whose price responded to the same Venezuela news flow but which the anchored wallet is not alleged to have touched. |

## 3. Wallet ranking in the analysis window

Window: 2025-12-26T00:00:00Z to 2026-01-03T09:21:00Z. 15698 of 77766 cluster trades fall inside it, across 6110 of 22892 wallets.

Rationale: Opens on the day the complaint says the Polymarket account was created (paragraph 48) and closes at the first public announcement of the operation — President Trump's TruthSocial post at 4:21 AM ET on 2026-01-03 (paragraph 32). Every trade inside the window is pre-disclosure by construction, which is what makes an elevated score inside it interesting. The window is fixed by the documented case facts, not chosen after looking at scores.

Trades outside the window are excluded from this table by design — a post-announcement trade is not insider trading — but they remain in the timeline and in the baseline below.

Cluster baseline mean P(Z) (all trades): 0.0500; in-window mean: 0.0500.

| rank | wallet | n_window | n_total | mean P(Z) | max P(Z) | elevation | theta_w evidence | anchored |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | `0x2b9edb595ea75c118077c7d2c17d4852b0770db0` | 1 | 4 | 0.0632 | 0.0632 | +0.0132 | prior-dominated |  |
| 2 | `0xfcc4e78ca3f229949b8106fb380bf14266853067` | 1 | 1 | 0.0585 | 0.0585 | +0.0084 | prior-dominated |  |
| 3 | `0xa5c5d420674488b0a11c114627c0f13c564bd8ee` | 1 | 1 | 0.0553 | 0.0553 | +0.0053 | prior-dominated |  |
| 4 | `0x50781ba59a7386fea3ad360a7b62d70ef8a3dd86` | 1 | 1 | 0.0551 | 0.0551 | +0.0051 | prior-dominated |  |
| 5 | `0x202d2b8ed386228fc928dac406d1a888739a9890` | 1 | 1 | 0.0550 | 0.0550 | +0.0050 | prior-dominated |  |
| 6 | `0xa347cd256d8d962715c6bffd4e5f541b2de0ba0c` | 1 | 1 | 0.0549 | 0.0549 | +0.0049 | prior-dominated |  |
| 7 | `0x1a799c0af8a939ba4ed95df95188c2f357491ccb` | 2 | 2 | 0.0546 | 0.0568 | +0.0046 | prior-dominated |  |
| 8 | `0xa2cc453e5d2869d480accf8e71d6328c33391857` | 1 | 1 | 0.0546 | 0.0546 | +0.0046 | prior-dominated |  |
| 9 | `0x7b76ff3a1acd99622af06aeaec49562f0fbb5960` | 1 | 1 | 0.0542 | 0.0542 | +0.0041 | prior-dominated |  |
| 10 | `0xd719eb46721f2618185c99c19d93ba609c01d8e5` | 2 | 2 | 0.0537 | 0.0571 | +0.0037 | prior-dominated |  |
| 1497 | `0x31a56e9e690c621ed21de08cb559e9524cdb8ed9` | 13 | 15 | 0.0500 | 0.0500 | -0.0000 | prior-dominated | **YES** |

Anchor: handle `Burdensome-Mix`, pattern `^0x31a5[0-9a-fA-F]{32}8ed9$` (Complaint paragraphs 48, 57 (pattern); resolved to the full address by the reconstruction below.). 1 wallet(s) matched.

The complaint redacts the middle of the address, quoting it as '0x31a5*...*8ed9' — 4 leading and 4 trailing hex characters of a 40-hex Polygon address. The pattern is therefore the primary-source anchor and remains the identification of record. Applied to the public Data API trade history of the four charged markets over the analysis window, it matched exactly one of 2,247 distinct wallets, and that wallet's trades then reproduce the complaint's itemized share counts (see 'reconstruction'). The full address is recorded here as a derived convenience; the analysis still matches on the pattern, reports how many wallets matched, and treats zero or several matches as inconclusive rather than negative.

## 4. Data sufficiency

ARCHITECTURE.md 9.5 fixes the reliability thresholds for the per-wallet posterior `theta_w`: **prior-dominated below ~20 trades**, **meaningful at or above ~100**. Counted on each wallet's total trades across the pulled cluster, not on its in-window trades:

| theta_w evidence | wallets |
|---|---:|
| prior-dominated (< 20) | 5788 |
| weak (20-99) | 291 |
| meaningful (>= 100) | 31 |

- Anchored wallet `0x31a56e9e690c621ed21de08cb559e9524cdb8ed9`: 15 trade(s) total, 13 in window -> **prior-dominated**.

The charging documents describe on the order of ten purchases in this cluster. That is an order of magnitude below the threshold at which this project's own wallet posterior means anything, so **the wallet ranking above is prior-dominated and is not the result of this case study.** The headline claim rests on the per-trade P(Z) timing evidence in the next section.

**Headline claim.** No claim. This run was cold-started, so every P(Z) is the prior mean plus filter noise and neither the wallet ranking nor the per-trade timing carries information about the model. Re-run with `--warm-start <fitted VEM artifact>`.

## 5. Per-trade P(Z) timing evidence

Highest-scoring individual trades inside the analysis window. Every score is a function of trades 0..t only (inherited from `score_stream.py --replay`), so each row is what a reader watching the stream would have seen at that moment.

| timestamp (UTC) | market | wallet | P(Z) | anchored |
|---|---|---|---:|---|
| 2025-12-30T06:11:31Z | `us-forces-in-venezuela-by-january-31-2026` | `0x2b9edb595ea75c118077c7d2c17d4852b0770db0` | 0.0632 |  |
| 2025-12-30T06:11:39Z | `us-forces-in-venezuela-by-january-31-2026` | `0xbf36df9171ba68b07334f024818df1d9bbef95ad` | 0.0610 |  |
| 2025-12-30T02:24:43Z | `us-forces-in-venezuela-by-january-31-2026` | `0x0de61278e7467aba680659da5b40e122c16ddaa2` | 0.0591 |  |
| 2025-12-26T09:01:39Z | `will-the-us-invade-venezuela-by-january-31-2026` | `0xfcc4e78ca3f229949b8106fb380bf14266853067` | 0.0585 |  |
| 2026-01-03T07:39:41Z | `us-forces-in-venezuela-by-january-31-2026` | `0x2b366034b69b64a444406b2db7ec1ff7869c473b` | 0.0581 |  |
| 2026-01-03T07:39:43Z | `us-forces-in-venezuela-by-january-31-2026` | `0x2b366034b69b64a444406b2db7ec1ff7869c473b` | 0.0580 |  |
| 2026-01-03T07:48:03Z | `us-forces-in-venezuela-by-january-31-2026` | `0x2b366034b69b64a444406b2db7ec1ff7869c473b` | 0.0579 |  |
| 2026-01-03T07:49:13Z | `us-forces-in-venezuela-by-january-31-2026` | `0x2b366034b69b64a444406b2db7ec1ff7869c473b` | 0.0578 |  |
| 2026-01-03T07:49:13Z | `us-forces-in-venezuela-by-january-31-2026` | `0x2b366034b69b64a444406b2db7ec1ff7869c473b` | 0.0577 |  |
| 2026-01-03T08:01:13Z | `us-forces-in-venezuela-by-january-31-2026` | `0x2b366034b69b64a444406b2db7ec1ff7869c473b` | 0.0576 |  |
| 2025-12-28T23:16:03Z | `trump-invokes-war-powers-against-venezuela-by-january-31-134-583` | `0xd719eb46721f2618185c99c19d93ba609c01d8e5` | 0.0571 |  |
| 2025-12-28T23:36:17Z | `trump-invokes-war-powers-against-venezuela-by-january-31-134-583` | `0x1a799c0af8a939ba4ed95df95188c2f357491ccb` | 0.0568 |  |
| 2025-12-26T08:32:29Z | `will-the-us-invade-venezuela-by-january-31-2026` | `0xa8af4c7923857533b687df9bf02115156e3a325e` | 0.0562 |  |
| 2026-01-03T06:15:19Z | `us-forces-in-venezuela-by-january-31-2026` | `0x2b366034b69b64a444406b2db7ec1ff7869c473b` | 0.0559 |  |
| 2025-12-26T10:44:59Z | `us-forces-in-venezuela-by-january-31-2026` | `0xa4bd8f49695cf5d55924aca4cc2694952874e8a2` | 0.0559 |  |

## 6. Charging-document timeline overlay

| timestamp (UTC) | event | source | citation | verified |
|---|---|---|---|---|
| 2025-12-08T00:00:00Z | Classified information security briefing; NDA signed | `cftc_complaint` | paragraph 35 | yes |
| 2025-12-11T21:44:09Z | 'Maduro out by January 31, 2026?' listed | `gamma` | createdAt; complaint paragraph 29 says 'On December 11, 2025' | yes |
| 2025-12-26T00:00:00Z | Polymarket account created; exchange account funded with ~$35,000 | `cftc_complaint` | paragraphs 46, 48 | yes |
| 2025-12-31T02:00:00Z | ~14,600 'Yes' shares across two transactions (evening of Dec 30 ET) | `cftc_complaint` | paragraph 50 | yes |
| 2026-01-02T02:00:00Z | 73,700 'Yes' shares (evening of Jan 1 ET) | `cftc_complaint` | paragraph 50 | yes |
| 2026-01-02T15:00:00Z | ~90,300 'Yes' shares (morning of Jan 2 ET) | `cftc_complaint` | paragraph 50 | yes |
| 2026-01-03T01:30:00Z | >250,000 'Yes' shares across three transactions (8:30-10:00 PM ET Jan 2) | `cftc_complaint` | paragraph 50 | yes |
| 2026-01-03T09:21:00Z | Public announcement of Maduro's capture (TruthSocial, 4:21 AM ET) | `cftc_complaint` | paragraph 32 | yes |
| 2026-01-03T12:14:00Z | 'Maduro out by January 31, 2026?' resolves Yes (7:14 AM ET) | `cftc_complaint` | paragraph 33; Gamma closedTime 2026-01-03T12:14:07Z agrees | yes |

## 7. Pull provenance and the pre-resolution deviation

```
python -m scripts.pull_data --slugs maduro-out-by-january-31-2026-318 us-forces-in-venezuela-by-january-31-2026 will-the-us-invade-venezuela-by-january-31-2026 trump-invokes-war-powers-against-venezuela-by-january-31-134-583 maduro-out-in-2025-411 --full-history --pre-resolution-days 0 --output-dir data/case_studies/van_dyke
```

`--pre-resolution-days 0` — **a deliberate deviation from this repository's 7-day default.**

Every other pull in this repository uses --pre-resolution-days 7, because resolved markets pin at 0/1 and the observation model then reads the pin as an insider signal (ARCHITECTURE.md 9.5). This case study sets it to 0 deliberately: the alleged insider window sits immediately before the EVENT (the 2026-01-03 announcement), which is also immediately before resolution, so the default filter would delete exactly the trades the case is about. The cost is that resolution-period over-flagging is NOT filtered out here, which is one reason the analysis window closes at the public announcement and the headline claim is made on trades strictly inside it.

scripts/pull_data.py writes the batch processed format (integer wallet ids), which the streaming scorer cannot replay. `python -m scripts.case_study --capture` writes the same markets in the stream_trades.py raw record shape (timestamp/price/size/wallet/transaction_hash/condition_id) so that `score_stream.py --replay` can consume them. Both walk the full history and neither applies a pre-resolution filter.

Scores provenance: mode='replay', input='data/case_studies/van_dyke/trades.jsonl', warm_start=None.

## 8. Caveats

- **COLD START — THIS RUN IS NOT A RESULT.** The scores provenance records `warm_start: null`. `stream_scoring.cold_start` leaves `theta_w` empty and the logistic coefficients uninformative, and `--n-refresh` defaults to never refreshing them, so every P(Z) is the prior mean plus filter noise and any ranking over it is a ranking of noise. Re-run `score_stream.py --replay` with `--warm-start <fitted VEM artifact>` before reading anything below as evidence about the model.
- **One case.** n = 1. Nothing here estimates a false-positive rate, a detection rate, or any quantity that generalizes. A score that lights up on the one labelled episode available is consistent with a useful detector and equally consistent with a detector that lights up often.
- **Post-hoc identification.** The markets, the wallet pattern and the window all come from a charging document written after the fact. Nobody pointed this detector at Polymarket in December 2025 and got an alert. The no-lookahead guarantee inherited from replay scoring is about the *scorer's state*, not about how the cluster was chosen.
- **No counterfactual.** There is no matched control episode — no market where the same news broke with no insider present — so the elevation reported here has no null it was tested against. The pre-registered permutation test lives in `src.analysis.event_study`; this module deliberately reports description, not a p-value.
- **Resolution-period contamination is not filtered.** The pull that feeds this study sets `--pre-resolution-days 0` (see the pull section), so the known over-flagging near resolution (ARCHITECTURE.md 9.5) is present in the data. Trades after the public announcement are outside the analysis window for exactly this reason, but the contamination is real and the in-window scores are not immune to a market already drifting toward its resolution.
- **The anchor is a redacted pattern.** The complaint gives four leading and four trailing hex characters of the wallet address. A match is strong evidence but not a certified identification, and a run that matches zero or several wallets is inconclusive rather than negative.
