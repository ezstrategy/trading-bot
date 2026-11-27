# Experiment log

Permanent record (Rulebook rule 13). Entries are never edited after the fact except to add a
later note marked **Note (date)**. All runs: XAUUSD, MT5 broker bars, UTC, signals only.
"Older / unseen" = the first 60% / last 40% of the period by candles.

**Retroactive note (2026-11-26).** EXP-001 to EXP-012 ran before the rulebook. Several of them
searched hundreds or thousands of variants (now forbidden by rule 7), none charged commission or
slippage (rule 6), and the 2020-2026 data has been looked at many times, so it can no longer be
treated as untouched out-of-sample data for any strategy chosen after looking at it. Classifications
below are re-stated under rule 14.

| ID | Date | Hypothesis | What was tested | Period | Result | Decision |
|---|---|---|---|---|---|---|
| EXP-001 | 2026-11-25 | Repeatable M5 event patterns per H4 regime | Pattern miner, 1,625 candidates, BH FDR 10% | 1 yr M5 | 0 passed | REJECTED |
| EXP-002 | 2026-11-25 | Same, longer history | 2,304 candidates | 2015-2026 M5 | 0 passed | REJECTED |
| EXP-003 | 2026-11-25 | Round numbers, S/R levels, news spikes add edge | 3,849 candidates, fixed $0.20 spread | 2015-2026 M5 | 0 passed; family medians -0.18 to -0.29R | REJECTED |
| EXP-004 | 2026-11-25 | Losing patterns win when reversed | 1,918 opposite pairs | 2015-2026 M5 | 91% of pairs lose both ways; pair sum about -0.45R (costs) | REJECTED |
| EXP-005 | 2026-11-25 | Edges on swing timeframes | H1 4,080 / H4 2,053 candidates | 2015-2026 | 0 passed; medians -0.04 to -0.10R | REJECTED |
| EXP-006 | 2026-11-25 | Price action and SMC (sweeps, OB, FVG, pins, engulfing) | M5 1,584 / H1 1,324 / H4 485 | 2020-2026 | 0 passed. Lead: H4 bullish BUY after bearish engulfing +0.27R (74 tr) then +0.32R (69 tr) | RESEARCH REQUIRED (small sample, found in a search) |
| EXP-007 | 2026-11-26 | 70% win rate on 30-50 pip moves in 30 min | 720 setup x target combos | 2020-2026 M5 | none reached 70%; setups predict volatility in both directions | REJECTED |
| EXP-008 | 2026-11-26 | 10 / 20 pip targets at 1:2 | 960 combos, $0.08 spread | 2020-2026 M5 | all-setup averages lose $0.07-0.15/trade; best +0.08R p=0.12 | REJECTED |
| EXP-009 | 2026-11-26 | Owner's entry: sweep of 30-min low/high + reversal candle | 180 variants (filter x session x 1-3R x 60/180 min) | 2020-2026 M5 | 55k signals; about -0.05R older, -0.01 to -0.04R unseen; 0 passed | REJECTED |
| EXP-010 | 2026-11-26 | VWAP position explains the owner's entries | 4 VWAP filters, 420 variants, distance buckets | 2020-2026 M5 | 0 passed; win rates flat across VWAP distance | REJECTED |
| EXP-011 | 2026-11-26 | Published strategies work on gold as published | 9 strategies, no tuning, $0.08 spread | 2020-2026 | see below | see below |
| EXP-012 | 2026-11-26 | Changes improve the London breakout | 360-version grid; best chosen on older data | 2020-2026 M5 | best (H4 trend filter, 3R, BE 1R, close 16:00 NY): older +0.103R / 752 tr, unseen +0.144R / 491 tr, p 0.007; max DD about 26R | RESEARCH REQUIRED (chosen from a grid; violates rule 7) |
| EXP-013 | 2026-11-26 | 50%+ win-rate versions; other symbols | 793-version grid on 5 symbols | — | **cancelled before running** (rule 7) | — |
| EXP-015 | 2026-11-27 | The London breakout edge exists before 2020 | 2 versions fixed in advance, one run, untouched 2015-2019 | 2015-2019 M5 | trend version +0.089R, PF 1.19, 914 tr, p 0.020 (passes Bonferroni 0.025); published +0.033R, p 0.21 | trend version: RESEARCH REQUIRED (survives untouched data, misses PF 1.3 / DD 10% targets); published: REJECTED as stand-alone |

## EXP-011 detail: public strategies, no tuning, $0.08 spread, no commission / slippage

Holding gold over the same parts: +32.7% older, +77.0% unseen.

| Strategy | Older | Unseen | Decision |
|---|---|---|---|
| London breakout, Asian range (mql5 blog 777540) | +0.026R, PF 1.05, 801 tr | +0.144R, PF 1.32, 602 tr, p 0.0016 | RESEARCH REQUIRED: weak older part, max DD about 25R (12.5% at 0.5%) is above the 10% target, no commission / slippage yet |
| Gold overnight long (16:00 -> 09:30 NY) | +35.8% (1x) | +71% (1x) | REJECTED as an edge: mostly gold's own rise; swap not charged |
| COMEX 08:20 opening-range breakout | +0.109R | +0.001R | REJECTED: did not hold up |
| NY 09:30 opening-range breakout | -0.013R | +0.028R | REJECTED |
| Turtle System 1 (daily) | -0.04R, 49 tr | +0.54R, 27 tr | REJECTED for now: too few trades |
| Connors RSI(2) (daily) | -3% | +6.7% | REJECTED for now: too few trades |
| 50/200-day MA cross | 11 tr | 1 tr | REJECTED: no sample |
| ICT Silver Bullet (10-11 NY) | -0.094R | -0.065R | REJECTED |
| Gold US day session (09:30-16:00 NY) | -24.7% | +19.9% | REJECTED |

## GTC-SBR (owner's spec, 2026-11-26)

`research/gtc_sbr.py` codes the owner's GTC-SBR rules (experiments A-F, 60/20/20 split). It has
**not** been run on real data. GTC-SBR v1.0.0 is tested in the Quant Lab (PR #9, programme
`gtc_sbr_2020`), which locks the last 20% of 2020+ as its test period. To keep that period
untouched, `python -m quant_engine gtc` must not be run on real data before the Quant Lab's test.

## EXP-014 (2026-11-27): regime behaviour study (rule 2), descriptive, development data only

Hypothesis: gold behaves differently in bullish / bearish / sideways / uncertain H4 regimes (drift,
momentum, breakout follow-through, sweep reversals), which would justify separate engines.
Data: MT5 XAUUSD M5, 2020-01-02 to 2024-02-22 (first 60% of 2020+). No costs, no strategy.

- Time in regime: bullish 29%, bearish 23%, sideways 18%, uncertain 29% (H4).
- Next-hour drift by regime x session: no cell significant (largest |t| = 1.63).
- Last-hour vs next-hour correlation: -0.04 to +0.02 in every regime (no momentum, no reversion).
- 2-hour range breakouts, +1 ATR before -1 ATR: 45.8-49.2% in every regime, with or against the trend.
- Sweep table in the first run used uneven levels (reversal from the close, failure beyond the wick),
  which biases it above 50%; **those numbers are void**. Fixed to symmetric levels and re-run (commit c1e182a): **49.6-53.6% in every regime, a coin flip** (highest: sideways swept high 53.6%, 1,541 cases, 1 of 8 rows, before costs).

Reading: at the 1-hour horizon on M5, the H4 regime does not change gold's short-term behaviour in
this period. A trend-continuation engine has no support from simple drift or breakout follow-through.
Decision: no strategy; informs the engine hypotheses (rule 2).

## EXP-015 (2026-11-27): London breakout, one-shot test on untouched 2015-2019 data

Hypothesis: the London-breakout edge seen on 2020-2026 (EXP-011 / EXP-012) is not specific to that
period. Two versions fixed in advance in `research/london_oos.py` (commit 080d3a8), run **once** on
MT5 XAUUSD M5 2015-01-02 to 2019-12-31, which no London-breakout test had used. Costs: historical bar
spread (at least $0.08), 2 points slippage per fill, commission 0. 0.5% risk compounded. Bonferroni
over 2 versions (p < 0.025). M5 quality issues reported by validate_bars: 82 spikes, 200 gaps.

| Version | Trades | Win % | PF | Expectancy | Net (0.5%) | Max DD | Max losing streak | p |
|---|---|---|---|---|---|---|---|---|
| published (mql5 777540) | 731 | 43.5 | 1.07 | +0.033R | +11.6% | 11.3% (23.4R) | 10 | 0.211 |
| trend version (EXP-012) | 914 | 33.5 | 1.19 | +0.089R | +47.3% | 10.9% (22.3R) | 19 | **0.020** |

Trend version detail: BUY +0.095R / SELL +0.083R; bull +0.142R, bear +0.157R, sideways +0.072R,
uncertain +0.035R. By year (R): 2015 +38.8, 2016 +18.9, 2017 +30.9, 2018 +12.5, **2019 -19.6**.
"asia" in the session table = entries 01:00-03:00 NY (orders open at 01:00 NY, rule-11 London starts
03:00 NY), not a bug. Published version: losses come from bearish regime (-0.117R, 183 tr).

Decision (rule 14): **trend version RESEARCH REQUIRED.** It is the first result in this log that held
on data never used to choose it, with costs, after a multiple-test correction. It does not meet the
screening targets (PF 1.19 < 1.3, DD 10.9% > 10%), 2019 was a losing year, and it still needs walk-
forward, a nearby-parameter check and an independent audit. Published version: REJECTED as a stand-
alone strategy (not significant). Rules are not changed after this result (rule 7).

## EXP-016 (2026-11-27): Phase 2 research on the London breakout trend version, plan (written before the run)

Owner brief "Phase 2 research" (2026-11-27). Code: `research/lb_phase2.py`, `python -m quant_engine lb-phase2`.
The original (EXP-012 / EXP-015 trend version) is frozen as `ORIGINAL`; the run first reproduces EXP-015
(914 trades, +81.4R) and stops if it cannot. One change at a time against the original:
H1 Asian range width (past-250-day percentile thirds of range / daily ATR), H2 breakout candle (body,
close position, break distance, rejection wick; on close-confirmed entries, thirds fixed on 2015-2019),
H3 previous-day levels (room to the opposing level < 1R / 1-3R / 3R+; side of previous close),
H4 volatility (daily ATR % of price, past-250-day thirds), H5 news (only with a verified historical
calendar file), H6 entry (close-confirmed, retest within 12 candles), exits B-F against A (= original).
Split: development 2015-2019, validation 2020-2022, test 2023+ (none untouched). Screen on development
(diff > 0, block-bootstrap p < 0.05, 100+ trades), confirm on validation with Holm, then nearby settings,
costs x1.5 / x2, yearly results and the test part. Highest possible class here: "promising, needs
forward validation". Part 4 explains 2019 with the same features, each checked on other years.

## EXP-016 result (2026-11-27, commit be8812d, run once on MT5 data 2015-01-02 to 2026-11-27)

Reproduction: **914 trades, +81.4R, trade for trade** with the original code. H5 (news) not run yet
(no calendar); it is added as EXP-016b with an official-source calendar.

Original, unchanged (historical spread + 2 pt slippage), block-bootstrap 95% interval of expectancy:

| Part | Trades | Win % | PF | Expectancy | 95% interval | Max DD | Max losing streak |
|---|---|---|---|---|---|---|---|
| development 2015-2019 | 914 | 33.5 | 1.19 | +0.089R | +0.014 to +0.173 | 10.9% | 19 |
| validation 2020-2022 | 545 | 32.1 | 1.25 | +0.115R | +0.010 to +0.218 | 9.0% | 13 |
| test 2023-2026 | 695 | 32.9 | 1.24 | +0.105R | +0.021 to +0.195 | 12.5% | 27 |
| all | 2,154 | | | +0.101R | +0.051 to +0.154 | | |

Newey-West p: 0.018 / 0.017 / 0.011. Only development was out of sample for this version.
Spread charged in 2015-2019 was much higher (median $0.17-0.40) than from 2020 ($0.08-0.14); at x2
costs development expectancy falls to about +0.02R.

Changes, one at a time (30): **1 passed the development screen** (H3 entry against the previous
close, +0.18R vs the rest, p 0.043, 121 trades) and it **failed validation** (diff +0.21, p 0.31, not
Holm-significant). Every exit variant B-F and both entry variants (close-confirmed, retest) were
worse than or equal to the original on development and validation. By the pre-set code rule three
rows are labelled "2 interesting": H3 against previous close, H4 volatility middle third (dev p 0.087,
validation diff -0.016) and H2 break distance low third (dev p 0.17, negative on validation and test);
the last two reverse out of sample and are **treated as rejected**. Audit notes: H2 close position and
rejection wick are near mirror images (one test counted twice; no effect on the outcome).

Part 4 (2019, 190 trades, -19.7R): no feature known before entry is weak in 2019 and also weak in
2015-2018 and 2020+; sells (-21.7R) and with-trend days (-13.3R) carried the loss but are positive in
every other period. Trades that closed back inside the range within 30 minutes lost -33.5R (an
outcome, not a usable filter). 2019 is about 2.4 standard deviations below the expected year (per-
trade SD about 1.1R): unusual but within what 12 years of trading can produce.

Decision: **no modification improves the original; the original is preserved unchanged.** Original
class: **promising, needs forward validation** (positive in every part with costs, but only 2015-2019
was out of sample, PF 1.19-1.25 < 1.3 and drawdown up to 12.5% > 10%). Next: forward paper signals.

## EXP-017 (2026-11-27): unchanged London breakout on other instruments, plan (written before the run)

Owner asked (2026-11-26, again 2026-11-27) whether other metals / FX pairs do better. Code:
`research/lb_symbols.py`, `python -m quant_engine lb-symbols`. Rules: the EXP-016 original, unchanged.
Instruments fixed in advance: XAGUSD, EURUSD, GBPUSD, USDJPY; Bonferroni over 4 (p < 0.0125, block
bootstrap and Newey-West both required). Costs: each instrument's own historical MT5 bar spread (floor 1
point) + 2 points slippage per fill (GUESS), commission 0; stress x1.5 / x2. All MT5 history from 2015.
Run once; nothing is tuned per instrument. Best possible class: "promising, needs forward validation".

## EXP-017 result (2026-11-27, commit 985a8cc, run once; MT5 history 2015-01-02 to 2026-11-27)

| Instrument | Trades | Win % | PF | Expectancy | 95% interval | Max DD | Since 2023 | Last 2 yrs | Years + | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| XAUUSD (ref.) | 2,152 | 33.0 | 1.23 | +0.102R | +0.051 to +0.155 | 12.6% | +0.109R | +0.118R | 11/12 | reference |
| XAGUSD | 2,214 | 20.5 | 0.60 | -0.290R | -0.348 to -0.229 | 96.2% | -0.107R | -0.057R | 0/12 | REJECTED |
| EURUSD | 2,170 | 27.0 | 0.97 | -0.016R | -0.070 to +0.037 | 37.6% | -0.011R | +0.043R | 6/12 | REJECTED |
| GBPUSD | 2,181 | 25.0 | 0.93 | -0.036R | -0.094 to +0.018 | 43.5% | -0.023R | -0.045R | 4/12 | REJECTED |
| USDJPY | 2,179 | 40.1 | 1.14 | +0.054R | +0.011 to +0.097 | 12.4% | +0.033R | +0.026R | 9/12 | p 0.0085 (< 0.0125), see note |

Note: stored MT5 FX spreads are 0 on 17-31% of M5 candles (floored at 1 point = 0.1 pip), so FX costs are
understated. USDJPY: development +0.019R (p 0.26), validation +0.141R (p 0.003), test +0.033R (p 0.22),
2026 negative; at x2 costs development is negative. Its verdict waits for a re-cost with a realistic
0.5-1.5 pip spread (cost assumption only, rules unchanged). Gold reference differs slightly from EXP-016
(2,152 vs 2,154 trades) because this run floors every instrument at 1 point instead of gold's $0.08.

**Note (2026-11-27), EXP-017 FX cost check** (from the trades files; rules unchanged, cost assumption only):
with a spread floor of 0.5 / 1.0 / 1.5 pips, USDJPY is +0.047R (t 1.99) / +0.035R (t 1.47) / +0.019R
(t 0.79) on all data, and +0.027R (t 0.63) / +0.016R / +0.005R since 2023. The broker's own spread
(MetaQuotes-Demo, London session) was 0.5 pip. Below the EXP-017 bar (p < 0.0125): **USDJPY REJECTED**
(not confirmed after realistic costs). EURUSD and GBPUSD stay negative at every floor. Only XAUUSD holds.

## EXP-016b (2026-11-27): H5 news, add-on run with an official-source calendar

Calendar built on the owner's PC from official pages only (no rule-generated dates): FOMC statements
(federalreserve.gov, time from each statement), BLS Employment Situation and CPI (8:30 ET, embargo time
on each release page), BoE MPC (bankofengland.co.uk, 12:00 UK except three Bank-published exceptions).
483 rows, 2015-01-08 to 2026-11-10. Left out: BoE emergency moves of March 2020 (no official time),
BEA GDP and ONS releases (no reliable archive obtained). Spot checks matched the source pages.
Same code (be8812d), run once with `--calendar`; all 30 non-H5 rows identical to EXP-016.

| Change | Dev n | Dev diff | Dev p | Val diff | Val p | Test diff | Class |
|---|---|---|---|---|---|---|---|
| US high-impact news day only | 107 | -0.283 | 0.998 | -0.024 | 0.51 | +0.001 | rejected |
| UK high-impact news day only | 36 | -0.008 | 0.57 | +0.666 | 0.014 | -0.21 | rejected |
| No US / UK high-impact news | 774 | +0.214 | 0.014 (screen pass) | -0.125 | 0.88 | +0.043 | rejected |

US news days were weak in 2015-2019 (-0.161R) but normal from 2020 (+0.10R). A news filter is not
supported. 2019 is not explained by news (days without US news also lost that year).

## EXP-018 (2026-11-27): scale-out exit, plan (written before the run)

Owner idea: split into 3 x 0.01 lots with targets T1 / T2 / T3, so a move to T1 that reverses still
pays. Code `research/lb_scaleout.py`, `python -m quant_engine lb-scaleout`. Entries are exactly the
original's trades; only the exit changes. Three equal parts; after T1 the other parts' stop goes to
entry. Versions: **S1 1/2/3R (main)**, nearby S2 0.75/1.5/3R, S3 1/2/4R, and S4 1/2/3R without break-
even. Same costs as EXP-016. Compared trade by trade with the original (paired block bootstrap).
"Better" needs: 2015-2019 diff > 0 with p < 0.05, 2020-2022 diff > 0 with p < 0.0125, and 2023+ diff > 0.
Reported with 2023+ as the main window, plus last 2 years and full history; win rate both as trade net
> 0 and as T1 reached; minimum account for 3 x 0.01 lots at 0.5% risk.

## EXP-018 result (2026-11-27, commit cc29fef, run once)

| Exit (same 2,154 entries) | Since 2023: win % | PF | R / trade | Max DD | Longest losing run | vs original (p) |
|---|---|---|---|---|---|---|
| Original 3R, BE 1R | 32.9 | 1.24 | +0.105 | 12.5% | 27 | - |
| S1 1/2/3R, BE after T1 | 52.5 | 1.16 | +0.068 | 12.4% | 6 | -0.037 (0.995) |
| S2 0.75/1.5/3R | 58.3 | 1.11 | +0.045 | 11.3% | 5 | -0.060 (0.993) |
| S3 1/2/4R | 52.5 | 1.15 | +0.064 | 12.2% | 6 | -0.041 (0.995) |
| S4 1/2/3R, no BE | 45.0 | 1.18 | +0.084 | 12.7% | 15 | -0.021 (0.77) |

All four are worse than the original in every window (2015-2019, 2020-2022, 2023+, last 2 years);
verdict **not better, REJECTED**. Scaling out raises the win rate and shortens losing runs but gives up
profit, because two thirds of each winner exit before 3R. Minimum account for 3 x 0.01 lots at 0.5%:
$4,113 at the median stop, $15,380 for 95% of trades. Audit: the original's 27-loss run counts break-
even exits (-0.035R after costs) as losses. One trade (of 2,154) exits stop/breakeven/breakeven: the
break-even step can trigger on the fill candle while a target cannot (same convention as the original
code); negligible.

## EXP-019 (2026-11-27): Monte Carlo account survival, $500 at 1:500, risk 1-5% (strategy frozen)
- **Question (owner)**: can the frozen EXP-016 original run on a $500 live account at 1:500, 5% risk?
  No strategy change; the same 695 trades since 2023 (`research_data/exp016_trades_original.csv`,
  costs included) feed every risk level.
- **Method**: `python -m quant_engine montecarlo` (`research/montecarlo.py`). 10,000 paths x 500
  trades per level, circular block bootstrap (10-trade blocks, same random draws for all levels), lot
  floored to 0.01 (100 oz contract), margin at 1:500, stop-out at 50% margin level (GUESS, broker not
  chosen), swap 0 (trades close same day). Two policies: `min_lot` (always trade, at least 0.01) and
  `skip` (skip a signal when 0.01 lot risks more than the target %). Checks: block length 1 and 30,
  edge at the lower 95% bound, zero edge, costs x2; full history as secondary.
- **Key fact**: median stop $13.62 per oz, so 0.01 lot risks 2.7% of $500 (5.7% at the p95 stop).
- **Results (min_lot)**: every level loses half the account in 23-26% of paths; P(DD>30%) 71% at 1%
  up to 100% at 5%. The risk % barely changes ruin because 0.01 lot dominates.
- **Results (skip)**: 1%: 91% of signals skipped (median $536 after 500 trades). 2%: 58% skipped,
  P(lose 30%) 0.7%, P(DD>30%) 4.8%, P(reach $1,000) 16%, median $671. 3%: P(lose 50%) 0.4%,
  P(DD>30%) 58%, median $960. 5%: P(lose 50%) 10.5%, P(DD>50%) 68%, median $1,797; at the
  lower-bound edge P(lose 50%) 47.8%.
- **Conclusion**: do not use 5%. On $500, 2% with the skip rule is the survival-based choice and 3%
  the upper limit; for real 1-2% sizing on every signal the account needs about $1,400-2,850.
  Strategy class unchanged ("promising, needs forward validation"); forward paper trading first.

## EXP-020 (2026-11-27): natural stop first, position sizing second, plan (written before the run)
- **Owner rule (verbatim in substance)**: stop at the strategy's natural invalidation (opposite Asian range
  boundary), never tightened or moved to fit the account; size for at most 2% of equity with all costs;
  skip when 0.01 lot would risk more; never raise risk to avoid a skip; report how many signals are skipped.
- **Finding before the run**: the EXP-016 original caps the stop at 0.6% of price, which TIGHTENS the stop
  on wide-range days: 548 of 2,154 trades (25%), and 313 of the 695 trades since 2023 (45%). That breaks
  the owner's rule, so the main version drops the cap ("natural stop"); the capped original stays
  unchanged and is shown next to it (rule 18, nothing overwritten).
- **Code**: `lb_trades` takes `stop_cap` (default 0.006 = original; None = natural). `live/sizing.lot_size`
  gains costs (`cost_per_unit`, `commission_per_lot`) and `skip_if_over`; the signal engine now skips
  (logs, sends nothing) when 0.01 lot would exceed `risk.per_trade_pct`, set to 2.0 in config/engine.yaml.
- **Run**: `python -m quant_engine lb-natural` (lb_natural.bat), balances $500 / $1,000 / $2,500 / $5,000,
  windows since 2023 and all history, equity compounding, margin checked at 1:500. One run.
- **Pre-check on the capped original's trade list (not the answer, the cap still tightens stops)**: at
  $500 since 2023, 300 of 695 signals (43%) would be skipped; none at $1,000 or more; none over full
  history from $500 (2015 prices made stops small and the account grows before stops widen).

## EXP-020 result (2026-11-27, commit 3c12ffb, run once on MT5 XAUUSD 2015-01-02 to 2026-11-27)
- **Per trade (costs included)**: natural stop since 2023: 695 trades, win 38.3%, PF 1.32, +0.121R
  (95% CI +0.042 to +0.200, p 0.0015), longest losing run 12; all history: 2,154 trades, PF 1.26, +0.110R.
  Capped original for comparison: since 2023 PF 1.24, +0.105R, losing run 27. Dropping the cap did not
  hurt; on capped days the tightened stop was being hit by noise. Median stop since 2023 $14.40, p95 $65.90
  (capped original p95 $28.50), so wide-stop days now need a bigger account.
- **Skipped signals at 2% max risk (0.01 lot over the limit; no margin skips)**, natural stop:
  since 2023: $500 348 of 695 (50.1%), $1,000 88 (12.7%), $2,500 8 (1.2%), $5,000 1 (0.1%).
  All history: $500 7 of 2,154 (0.3%; the account grows before stops widen), $1,000+ none.
  Skipped signals averaged +0.10R at $500 since 2023 (missed profit, not avoided losses).
- **Sized ledger since 2023, natural stop**: $500 -> $1,024 (max DD 21.3%), $1,000 -> $3,288 (24.5%),
  $2,500 -> $10,449 (28.8%), $5,000 -> $21,865 (30.0%). Worst single loss 2.4-2.6% of equity (costs
  beyond the stop on slippage/spread, see report note). Back-test only; not a forecast.
- **Class**: unchanged, "promising, needs forward validation". The natural stop becomes the reference
  rule set going forward (owner rule); the capped original stays logged as EXP-016.

## EXP-021 (2026-11-27): regime behaviour study on H1 instead of H4, plan (written before the run)
- **Owner ask**: "why our 3 mode strategy not worked?" then "So use h1". The 3-mode idea (bull: buy the dip,
  bear: sell the rally, sideways: fade breakouts) needs gold to behave differently per regime. EXP-014 found
  no difference with the H4 regime. Hypothesis: a faster H1 regime does separate behaviour.
- **Test**: the EXP-014 study unchanged except the regime is worked out on H1 (same regime4 settings,
  counted in H1 candles: EMA 50, slope over 10 candles, etc.; labels used only after the H1 candle closes).
  `python -m quant_engine study --regime-tf H1`, MT5 XAUUSD M5, development part only (first 60% of 2020+,
  about 2020-01 to 2024-02, same as EXP-014). One run, no settings changed after it.
- **Pass bar (fixed now, before costs)**: H1 regimes "behave differently" only if at least one of:
  (a) a drift cell with |t| >= 3.0 (about 5% after Bonferroni over the 16 regime x session cells);
  (b) a breakout or sweep row with >= 500 decided cases and >= 55% first-way;
  (c) a last-hour vs next-hour correlation of |r| >= 0.05 in a regime;
  and the effect must point the way the matching mode needs (continuation in bull / bear, reversal in
  sideways). If none passes: stop, the 3 modes are not backtested on H1 (rule 7, no fishing).
- **If it passes**: one pre-registered 3-mode backtest with the H1 regime, costs included, 2023+ as the
  owner's main window and full history secondary; plan written in this log before that run.
