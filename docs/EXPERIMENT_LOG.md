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
