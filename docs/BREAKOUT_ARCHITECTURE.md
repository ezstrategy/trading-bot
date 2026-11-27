# Long-only XAUUSD breakout module: architecture proposal

Owner spec "XAUUSD QUANT ENGINE — LONG-ONLY BREAKOUT STRATEGY" (2026-11-27). Status: built in
paper mode, **live signals disabled** until the owner reviews (1) this proposal, (2) the rule
implementation, (3) the backtest reproduction on real MT5 data, (4) the unit-test results and
(5) paper mode. Signals only: nothing in this module can place, modify or close an order.

## 1. What already exists (inspected before changing anything)

| Area | Existing piece | Used by the breakout module? |
|---|---|---|
| Market data | `mt5/` (SafeMT5, retrying `MT5Client.call`), `data/collector.py` (server time -> UTC, closed candles only), `data/storage.py` (Parquet bars), `data/catalog.py` (immutable dataset versions), `data/quality.py` | Yes: shared market-data service. Live reads closed M5 from the collector; backtests read stored bars or a catalog version, validated by `quality.py` first (issues reported, nothing patched). |
| Broker spec | `instruments.py` + `config/instruments.yaml` / `instruments.<broker>.yaml` (`mt5-instrument`) | Yes: sizing uses contract size, tick size, tick value, lot limits. Now also reads the broker's swap when MT5 quotes it in points. |
| Live engine | `live/engine.py` (London breakout, Telegram follow-ups) | Not touched. The breakout service is separate. |
| Research platform | `backtest/engine.py`, `strategies/` plug-ins, `experiments/` registry (Quant Lab, PR #9) | Not reused for fills: the breakout spec needs D1 decisions, session-count exits, swap and a stop-first rule without M1 resolution, which the M5 plug-in engine does not model. Results go to `reports/breakout/` and `docs/EXPERIMENT_LOG.md`. |
| Web app | FastAPI + static glass front end (PR #10), localhost only, no login | Yes: new `Breakout` page and `/api/breakout` routes. |
| Authentication | none (the app binds to 127.0.0.1 only) | Unchanged. Manual actions are local records. |
| Database | JSON state for London signals; SQLite for the Quant Lab registry | New SQLite file `data/breakout/signals.sqlite` (append-only events). |

Nothing existing was rebuilt. This branch merges PR #9 (Quant Lab) into PR #10 (web app + London
signals) so the module sits on both; the only edits to existing files are: the router and page
registration, the CLI wiring, `config_dir` in the web app context, and swap reading in
`InstrumentSpec.from_symbol_info`.

## 2. Package layout (`src/quant_engine/breakout/`)

```
config.py    breakout.yaml schema (pydantic, unknown keys rejected, review gate for mode: live)
daily.py     D1 candles from M5 (NY 17:00 day, DST-correct), final only at close_time
rules.py     levels (previous month high, 20-day channel), D1 ATR, strategy definitions
state.py     breakout state machine (one per strategy)
trade.py     one BUY trade on M5: next-open entry, stop-first, session-count expiry, costs in R
scan.py      shared signal-generation interface: replay a strategy D1 close by D1 close
risk.py      shared risk engine (allowed levels, per-strategy, correlation, combined cap, lots)
backtest.py  both strategies, periods, train/test, walk-forward, neighbourhood, sized accounts, report
store.py     permanent signal record (SQLite) + manual workflow + separate performance
service.py   paper / live service on MT5 (same scan code as the backtest)
commands.py  CLI: breakout-backtest, breakout-run, breakout-status
webapp/routes/breakout.py + static/js/pages/breakout.js   dashboard page
```

Strategies are independent: each has its own parameters, version, rule hash, state machine and
trade. They share only the data service, the trade simulator, the risk engine, the store and the
signal interface (`scan`).

## 3. Data and time

* All internal timestamps are UTC. The dashboard shows the viewer's zone (IST by default) and
  broker time (MT5 server offset when known, else New York + 7h, marked GUESS).
* **D1 candles are built from M5**, not taken from MT5's D1, so the backtest and live use the same
  definition and the broker's server-time quirks cannot leak in. Day *D* = `(D-1) 17:00 NY` to
  `D 17:00 NY` (the FX/metals day; Sunday-evening candles belong to Monday). GUESS: the owner's
  broker uses the same day boundary for its D1 candles; `utc_midnight` is available to compare.
* A D1 candle is **final at its official close time** (17:00 NY), even on early-close holidays,
  and only final D1 candles exist in any calculation. A day with fewer than
  `min_m5_bars_per_day` (100) M5 candles is marked incomplete: it still counts as a session, but
  no signal may be generated from it (recorded as a blocked breakout).
* Look-ahead tests: levels, ATR and the full signal list are identical with and without future
  data (`test_levels_have_no_lookahead`, `test_scan_signals_do_not_change_when_future_data_is_added`).

## 4. Strategy rules (exactly as specified, version 1.0.0)

| | Strategy A: Previous Month High (primary) | Strategy B: 20-Day Channel (secondary) |
|---|---|---|
| Level | highest high of the previous completed calendar month (trading-day month); never the current month | highest high of the previous 20 completed D1 candles, breakout candle excluded |
| Breakout | completed D1 close **strictly above** the level while armed | same |
| Signal | generated at the D1 close, after the checks | same |
| Entry | open of the next available M5 candle (normally the 18:00 NY reopen): BUY at ask = bid open + spread, plus slippage | same |
| Stop | entry - 1.5 x D1 ATR(14), Wilder, ATR includes the breakout candle | same |
| Target | entry + 2 x initial risk | same |
| Max hold | 5 trading sessions; the entry session is session 1; close at the close of the last M5 candle of session 5 | same |
| Same M5 candle hits stop and target | stop first | same |

Missing-level rules: A's level is unavailable when the previous month has fewer than 15 complete
D1 candles, or is a partial first month of the data. B needs 20 previous D1 candles.

Interpretations to confirm (GUESS):
* "Entry price" for the stop and target is the actual fill (ask + slippage), so the 1.5 x ATR
  risk is measured from what is really paid.
* Exits: a target fills at the target minus slippage (never better on a gap up); a stop fills at
  the stop, or at the candle open when price gaps below it, minus slippage.
* "Trading session" = one D1 trading day that has data.

## 5. Breakout state machine and the reset condition

```
WAITING ──D1 close > level──▶ CONFIRMED ──checks pass──▶ SIGNAL ──entry──▶ ACTIVE ──exit──▶ CLOSED
   ▲                              └──checks fail──┐                                          │
   └──────────── reset condition ──── WAITING_FRESH ◀────────────────────────────────────────┘
```

* Every breakout **disarms** the strategy: a signal, a breakout that failed a check, and a
  breakout ignored because the strategy already has a position. Price staying above the level
  therefore never produces a new signal each day.
* Reset condition (configurable `reset_mode`, documented choice):
  * **A: `new_month`** (default). The level only changes once a month, so A re-arms on the first
    D1 candle of each new month: at most one breakout per calendar month. Option:
    `close_below_level` or `new_month_or_close_below`.
  * **B: `close_below_level`** (only option). The channel moves daily; B re-arms after a completed
    D1 close at or below that day's channel level, i.e. after price has fallen back inside.
* At the start of the data a strategy is not armed: a reset must be seen first, so a market that
  is already above the level is not reported as a fresh breakout.
* Duplicate protection: the signal key `strategy:version:breakout-day` is UNIQUE in the store.

## 6. Signal checks (before a signal is generated)

1. breakout candle closed (only final D1 candles exist); 2. data valid (complete D1 day; live: the
newest M5 candle is at most 30 minutes old while the market is open, the D1 close at most 20 h old,
no signal is ever issued late); 3. ATR available and > 0; 4. stop and target on the correct side
(checked at the entry candle); 5. spread at entry within `max_spread` (1.00, GUESS: wide enough for
the reopen); 6. no duplicate (state machine + UNIQUE key); 7. exposure rules (risk engine).
A failed check is stored as a blocked breakout with its reason; it never becomes a signal.

## 7. Risk management (shared engine, `risk.py`)

* 0.5% of equity per trade by default; allowed 0.25 / 0.5 / 1%; any other value is refused.
* One active position per strategy; all open + pending breakout positions <= 1% of equity.
* **Correlation**: signals of A and B whose breakout days are within 3 sessions are one exposure.
  Default `primary_only`: A (primary) gets the 0.5%, B is recorded as Cancelled ("correlated,
  risk not doubled"). Option `split`: they share the 0.5%. Two uncorrelated breakouts may be open
  together up to the 1% cap.
* Lots = equity x risk / loss per lot, loss per lot = (entry - stop + exit slippage) x tick value /
  tick size + commission, rounded **down** to the broker's lot step. Live, the spec is read from
  MT5 `symbol_info` (never assumes one point = $0.01). If 0.01 lot would exceed the risk, the
  signal is skipped and reported; the stop is never moved.
* No martingale, grid, averaging down or lot escalation: size depends only on equity, stop distance
  and the fixed percentage.
* Note: these limits apply to the breakout module only (owner spec). The London engine keeps the
  owner's earlier 2% natural-stop rule in `engine.yaml`.

## 8. Backtest (`backtest.py`)

* Input: stored MT5 M5 bars or a catalog dataset version; quality report included in the report.
* Costs: data spread (never below 0.08), 2-point slippage per fill (targets too), commission per
  lot, swap per rollover crossed (triple on Wednesday). **Swap** comes from `breakout.yaml`, else
  from the instrument profile read from MT5; if neither is known the report headline says
  "Swap: NOT INCLUDED" and adds a stress row at a clearly-labelled GUESS of -50 points/lot/night.
  Today's swap applied to old years is conservative (rates were near zero 2015-2021).
* Results in R for each strategy alone (independent), plus a sized simulation of both strategies
  through the risk engine for each account size ($500 and $10,000), listing every skipped signal.
* Windows: 2023+ (main), full history, 2015-2018 / 2019-2022 / 2023+, chronological train (<2023)
  / test (>=2023), and a 3-year-train / 1-year-test walk-forward. **Nothing is fitted**: the
  walk-forward can only choose between parameter sets the owner lists in
  `walk_forward.candidates` (empty by default) and only on train windows. A one-at-a-time
  parameter neighbourhood is shown for robustness and is never used to pick rules.
* Metrics: trades, win rate, profit factor, expectancy (R), net R, max drawdown (R and % in the
  sized runs), max losing streak, average holding time, TP/SL/expired counts, cost per trade,
  t-statistic, longest recovery, yearly and monthly tables, equity curve. All trades are BUY.

## 9. Signal record and manual execution (`store.py`)

Every breakout of the last 20 h seen by the service is stored permanently: strategy id, version
and rule hash, times, breakout level, D1 close, ATR, estimated and actual entry / stop / target,
spread at signal and at entry, estimated costs, suggested lots and risk %, risk decision, checks,
explanation and a snapshot of the last 25 D1 candles. The `events` table records every change and
SQLite triggers forbid updating or deleting events and deleting signals.

Statuses: Waiting (strategy, no signal) · BUY signal (entry candle not open yet) · Awaiting manual
approval · Active · TP hit · SL hit · Expired · Cancelled (blocked, risk-skipped, correlated or
rejected). Owner actions: approve, reject, mark as manually executed (actual entry and lots),
record actual exit and result. Theoretical (paper) outcomes and actual executions are separate
columns, and the dashboard shows backtest, paper and manual performance separately.

## 10. Modes and the review gate

`breakout.yaml mode`: `off` · `paper` (default: stored and shown, never sent) · `live` (also sent
to Telegram). The config validator refuses `live` unless `live_review_approved_by` is filled in,
which only the owner should do after the review. Run on the PC: `breakout_paper.bat`
(`python -m quant_engine breakout-run`) and `breakout_backtest.bat`.

## 11. Adding a SELL strategy later

SELL is disabled: every strategy has `side = +1` and `trade.simulate_long` raises for `side = -1`.
A SELL strategy is added as a new `BreakoutStrategy` (e.g. previous-month-low breakdown) with its
own level function, a short trade simulator (sell at bid, exit at ask), its own tests and its own
experiment-log entry. The state machine, store, risk engine and dashboard are direction-neutral.
The correlation rule must then treat a long and a short as offsetting, not additive (to decide).
