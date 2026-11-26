# Quant Lab Phases 3-9 report

Date: 2026-11-26. The owner asked for all remaining phases to be built in one go, with the
real-data runs on their PC on 2026-11-27. Branch `claude/project-thread-hdljm2` (PR #9, stacked on
PR #8). User guide: [QUANT_LAB.md](QUANT_LAB.md).

**No performance claims.** Every number produced so far comes from synthetic random-walk test
data, which says nothing about gold. Real results come from the runs on the owner's PC.

## Completed

| Phase | Module | What was built | Where |
|---|---|---|---|
| 3 | 3, 5 | Indicators (EMA, EMA slope, ATR, opening range, sessions, EMA trend); multi-timeframe alignment on candle **close** times; `MarketData` / `MarketView`; one event-driven engine (next-open fills, bid/ask, spread floor + stress multiplier, slippage, commission, stop-first or M1 intrabar, gap-through-stop at the open, time exits, data-gap cancel, duplicate / busy / max-concurrent / regime rejection, sizing from current equity rounded down to the lot step, full trade ledger with gross and net); look-ahead harness | `indicators/`, `backtest/` |
| 4 | 4 | Plug-in contract (id, version, family, rule texts, validated params, fingerprint, regime filter) and registry; GTC-SBR as `gtc_sbr_bull` 1.0.0 and `gtc_sbr_bear` 1.0.0 | `strategies/` |
| 5 | 8 | Three cost layers (before costs / after spread / net); net and gross profit and loss, PF, win rate, expectancy ($ and R), avg R, max DD ($ and %), losing streak, Sharpe, Sortino, monthly returns, breakdowns by side, session, regime, strategy, exit, intrabar method, year | `analytics/` |
| 6 | 6, 7 | Append-only SQLite registry (triggers block UPDATE / DELETE) + artefact folder per experiment; permanent per-programme splits; test-period lock and reuse count; walk-forward (fixed, or re-selection from at most 12 pre-declared sets); sensitivity; cost stress; comparison | `experiments/`, `validation/`, `config/lab.yaml` |
| 7 | 9 | Dashboard tab **Quant Lab** with the 12 pages, CSV downloads of trades, equity, comparison and every report file | `dashboard/lab_page.py`, `lab_services.py` |
| 8 | 1, 2 | `mt5-import` (MT5 history -> dataset versions with broker symbol and server-clock settings); `mt5-instrument` (broker spec from MT5 `symbol_info` into `config/instruments.<name>.yaml`, never overwritten) | `cli.py`, `instruments.py` |
| 9 | 10 | `lab-audit`: data checksums and quality, identical re-run, rule set unchanged, look-ahead on the real data, timing, cost and equity reconciliation, overlaps, risk cap, then rule-14 classification registered as its own experiment | `audit/` |

Launchers for tomorrow: `lab_import_mt5.bat` (data + broker spec), `lab_gtc_sbr.bat` (development,
validation, walk-forward / sensitivity / cost stress; does **not** touch the test period).

## Evidence that the engine is right (all tests, synthetic or hand-built data)

- Hand-worked trades: next-open fill with spread and slippage, short stopped on the ask, stop-first,
  M1 deciding target-first vs both-in-one-minute, gap through the stop, target from filled risk,
  lot size 0.20 for $50 at a 2.50 stop, commission, gross = net + costs, equity compounding,
  min-lot skip, spread floor / assumed / x2.
- GTC-SBR plug-ins reproduce all 42 trades of the existing `research/gtc_sbr.py` on 200 days of
  synthetic data (same entries, same R to 1e-6).
- The look-ahead harness catches a strategy that peeks one candle ahead, and passes both GTC-SBR
  plug-ins.
- Registry refuses UPDATE / DELETE; a moved split is refused; the test part is refused without
  `--final-test`; a second test use is reported.
- The audit passes a clean run and rejects a run whose data file was changed afterwards.
- Speed (cloud machine, 6 years synthetic): backtest with M1 intrabar 5.5 s; look-ahead check with
  4 cuts 7.6 s.

Test suite: **360 passed** (316 before + 44 new: engine 21, GTC-SBR plug-ins 6, analytics 5,
experiments / validation / MT5 / audit / CLI 10, dashboard 2), 4 min 57 s.

## Assumptions (GUESS) to check

1. Slippage 2 points per fill, also charged on target exits (conservative).
2. H4 for the regime is built from M5 on the New York-close server grid; M15 is built from M5 when
   no M15 version exists. Both are recorded as "derived from" the M5 version.
3. Sharpe / Sortino annualised with 252 days; risk-free rate 0.
4. Walk-forward 24-month train / 6-month test; split 60 / 20 / 20 by time (`config/lab.yaml`).
5. Instrument profile values other than spread and commission stay guesses until `mt5-instrument`
   reads your broker's specification.

## Outstanding issues

1. **The GTC-SBR test period may already be used.** The other thread's `gtc` run on your PC reports
   the last 20% of 2020+ ("out-of-sample"). The Quant Lab split of `gtc_sbr_2020` covers the same
   last 20%. Once you read that report, GTC-SBR v1.0.0's test period is no longer untouched.
   I recommend recording that as one test use (`lab-note`), and treating forward paper trading as
   the clean test for any GTC-SBR change.
2. No historical news calendar: GTC-SBR still blocks fixed New York release slots every weekday.
3. The live Telegram engine still runs the older three-mode model; plug-in strategies are not yet
   wired to live signals (not part of this spec; can be added once a strategy is validated).
4. The older research commands (`gtc`, `london`, `classics`, ...) still use their own loops. They are
   kept, unchanged, so earlier results stay reproducible; new work should use the `lab-*` commands.
5. M1 history from MT5 may be shorter than M5 history at some brokers; trades without M1 fall back to
   stop-first, and the `intrabar` column shows which method each trade used.
6. Sideways strategies: the plug-in slot is ready; none written yet (needs a hypothesis first, rule 2).

_Signals and research only. No live order execution exists anywhere in the code._
