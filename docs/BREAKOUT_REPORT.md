# Long-only breakout module: completion report (2026-11-27)

Owner spec section 10 asks for five deliverables before live signals may be enabled.

| # | Deliverable | Status |
|---|---|---|
| 1 | Architecture proposal | Done: `docs/BREAKOUT_ARCHITECTURE.md` |
| 2 | Strategy rule implementation | Done: `src/quant_engine/breakout/` (A and B, version 1.0.0, BUY only) |
| 3 | Backtest reproduction report | **Code done, real-data run pending.** It needs the owner's MT5 M5 history (2015+), which is on the PC. The PC was offline while this was built. No number from synthetic data is reported anywhere. Plan logged before the run as EXP-021 in `docs/EXPERIMENT_LOG.md`. |
| 4 | Unit-test results | Done: see below |
| 5 | Paper-trading mode | Done: `breakout-run` / `breakout_paper.bat`, default `mode: paper` (stored + dashboard, nothing sent, nothing traded) |

Live signal generation stays **disabled**: `breakout.yaml` refuses `mode: live` unless
`live_review_approved_by` is filled in by the owner after reviewing 1-5.

## How to produce deliverable 3 on the PC

1. Pull this branch, then refresh the broker spec so swap and lot limits are real:
   `python -m quant_engine mt5-instrument --profile my_broker` and set
   `costs.instrument_profile: my_broker` in `config/breakout.yaml`.
2. Make sure the M5 history is stored (it is, from the earlier research runs), then double-click
   `breakout_backtest.bat` once.
3. Open `reports/breakout/<time>/report.md` or the dashboard's Breakout page. The result is added
   to `docs/EXPERIMENT_LOG.md` as "EXP-021 result" and classified under the rulebook.

What the report will contain: both strategies on their own (2023+ main, full history,
2015-2018 / 2019-2022 / 2023+, train before 2023 / test from 2023, yearly walk-forward), trades,
win rate, profit factor, expectancy and net R, max drawdown, max losing streak, average holding
time, TP/SL/expiry counts, cost per trade, longest recovery, yearly and monthly tables and the
equity curve; the overlap between A and B; and a sized run of both strategies through the risk
engine at $500 and $10,000 with every skipped signal and its reason.

Known before the run (not a result): Strategy A can fire at most once a month, so it cannot reach
the rulebook's 200-trade screening size in any window; and with D1 ATR stops of roughly $40-90,
0.01 lot risks about 8-18% of a $500 account, so at 0.5% risk almost every signal would be skipped
on $500 (the sized table will show the exact count).

## Decisions and assumptions to confirm (GUESS)

* D1 candles are built from M5 with the New York 17:00 day (Sunday evening belongs to Monday).
* Reset condition: A = once per calendar month; B = after a D1 close back at or below the channel.
* A breakout that fails a check (missing data, spread at entry above $1.00) is recorded as blocked
  and still uses up that breakout; it is never traded late.
* Stop and target are measured from the actual fill (ask + slippage). Targets also pay slippage
  and never fill better on a gap.
* Correlated A+B breakouts (within 3 sessions) share one 0.5% budget; A, the primary, takes it.
* The breakout module uses 0.5% / 1% combined (this spec). The London engine still uses the
  earlier 2% natural-stop rule from `engine.yaml`; the two are separate settings.
* Swap is only included when known from the broker or `breakout.yaml`; otherwise the report says
  "Swap: NOT INCLUDED" in its header and adds a stress row at -50 points/lot/night.

## Unit-test results

New: `tests/test_breakout.py` (26 tests) and `tests/test_breakout_service.py` (11 tests), all
without MetaTrader 5. Full suite on this branch: **435 passed** (398 before + 37 new), 0 failed.

Covered: NY-close day boundary incl. DST and Sunday bars; forming day excluded; incomplete days;
previous-month level never uses the current month and refuses a partial first month; channel
excludes the breakout candle; levels, ATR and the whole signal list are identical with and without
future data; state machine (not armed at start, no repeat while above the level, reset modes,
breakouts ignored during a position); entry at the next M5 open with spread and slippage; stop
first when both are hit in one candle; gap through the stop; expiry at the close of session 5;
triple-Wednesday swap; spread limit; SELL refused; lots from tick value / tick size; minimum-lot
skip; allowed risk levels; one position per strategy; correlated budget (primary-only and split);
combined cap; review gate in the config; permanent store (delete / update of events refused by
SQLite); duplicate keys; full manual workflow with paper and actual results kept apart; service
stores a signal once, follows it to TP/SL/expiry, never issues late or stale signals, paper mode
never sends; CLI backtest; `breakout-run --once` against the fake MT5; web API including the
manual actions.
