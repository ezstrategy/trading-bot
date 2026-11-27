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
  which biases it above 50%; **those numbers are void**. Fixed to symmetric levels; re-run pending.

Reading: at the 1-hour horizon on M5, the H4 regime does not change gold's short-term behaviour in
this period. A trend-continuation engine has no support from simple drift or breakout follow-through.
Decision: no strategy; informs the engine hypotheses (rule 2).
