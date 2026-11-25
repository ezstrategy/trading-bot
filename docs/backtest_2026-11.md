# First real-data backtest: XAUUSD, 25 Nov 2025 to 25 Nov 2026

**Data:** MT5 demo broker bars (68,800 M5 candles, UTC; quality check: 43 missing, 11 spikes).
**Engine:** the three-mode strategy (PR 6) with the risk locks; spread from the broker bars.
**News filter:** not applied (the cached calendar does not cover the year).
**Reproduce:** `python -m quant_engine download --days 365`, then
`python -m quant_engine backtest --mode compare` and `python scripts/tune_modes.py`.

> Small samples: every cell below has fewer than 30 trades except where noted. These numbers show
> where the rules break, not whether the strategy has an edge. It has not shown one yet.

## Results by mode and market (before tuning)

Each single mode runs in every market; `regime` uses only the mode that matches the market.
The H4 regime detector labelled the year 46% neutral, 30% bearish, 24% bullish.

| mode | market | trades | win % | avg R | total R | profit factor | max DD (R) |
|---|---|---|---|---|---|---|---|
| regime | all | 21 | 14.3 | -0.12 | -2.6 | 0.86 | 13.0 |
| regime | bullish | 2 | 0 | -1.00 | -2.0 | 0 | 2.0 |
| regime | bearish | 0 | | | | | |
| regime | neutral | 19 | 15.8 | -0.03 | -0.6 | 0.96 | 11.0 |
| bullish | all | 3 | 0 | -1.00 | -3.0 | 0 | 3.0 |
| bearish | all | 0 | | | | | |
| sideways | all | 32 | 15.6 | -0.24 | -7.6 | 0.72 | 15.7 |
| sideways | bullish | 7 | 28.6 | -0.14 | -1.0 | 0.80 | 3.0 |
| sideways | bearish | 6 | 0 | -1.00 | -6.0 | 0 | 6.0 |
| sideways | neutral | 19 | 15.8 | -0.03 | -0.6 | 0.96 | 11.0 |
| v1 | all | 0 | | | | | |

## What the numbers say

1. **Too few signals.** 21 trades a year in `regime` mode; the bearish mode never fired and the
   bullish mode fired twice. v1 produced no trades. Most candidates die at the kill-zone gate
   (by design) and then at the H4/H1 alignment gates.
2. **Sideways TP1 was out of reach.** The entry is the swept range edge with a tight stop
   (about 2-9 USD), so the range middle was often 5-20R away. In `regime` mode 18 of 21 filled
   trades were full stop-outs, and 10 of those 18 had first gone 1R or more in profit (up to 2.6R)
   with nothing banked.
3. **Fills that stop on the same candle.** Six losers were filled and stopped inside one M5
   candle. The backtest assumes the stop came first there (conservative). M1 data would settle it.
4. **Sells lost.** In `regime` mode sells (all sideways) won 1 of 11; buys made +6.6R on 10.
   Too few to act on.

## Tuning tried (both halves of the year, split 26 May 2026)

| variant | regime: total R (1st / 2nd half) | regime max DD (R) | sideways alone: total R (1st / 2nd) |
|---|---|---|---|
| current | -2.9 / +0.3 | 13.0 | -1.9 / -5.7 |
| **sideways TP1 = nearer of middle and 1R** | -2.0 / -0.3 | **9.0** | +5.4 / -2.3 |
| sideways TP1 1.5R | -2.5 / +0.4 | 10.2 | +4.2 / -2.8 |
| sideways stop >= 1 ATR | -1.1 / +1.0 (59 trades) | 15.0 | +3.8 / -13.2 |
| TP1 1R + stop >= 1 ATR | -12.2 / +1.4 | 16.6 | -2.6 / -3.5 |

**Kept:** TP1 capped at 1R in the sideways mode (`modes.sideways.tp1_max_r: 1.0`). It matches the
bullish and bearish modes (TP1 at 1R), raises the win rate (14% to 29% in `regime`) and cuts the
drawdown from 13R to 9R. Total R is unchanged within noise, so this is a risk fix, not an edge.

**Not kept:** widening sideways stops to 1 ATR (more trades, no better, worse in the 2nd half).
It stays available as `modes.sideways.min_stop_atr` (off).

## Next

* More history: the broker gives one year of M5. Dukascopy gives several years, which the
  bullish and bearish modes need before they can be judged at all.
* Loosen the bullish and bearish entry gates on purpose (they trade a few times a year) and
  re-test on the longer history, checking each change on both halves.
* Do not start Telegram signals on these rules until a longer test shows a positive result
  in both halves.
