# XAUUSD Quant Lab: Phase 1 architecture report

Date: 2026-11-26. Scope: the owner's "XAUUSD QUANT LAB, Research & Backtesting Architecture
Specification" (modules 1-10, phases 1-9). This report covers **Phase 1 only**: what exists,
what is missing, and the proposed design. Nothing in the code was changed in this phase.

Inspected: `main` (PRs #1-#6 merged) and branch `claude/project-thread-5uub1y` (PR #8, not yet
merged), which is `main` plus the research tools, the rulebook, the experiment log and GTC-SBR.
The plan below builds on PR #8.

---

## 1. What exists today

| Layer | What it is | Where |
|---|---|---|
| Language / packaging | Python 3.10+, one package `quant_engine`, CLI `python -m quant_engine <command>`, Windows `.bat` launchers | `src/quant_engine/`, `pyproject.toml` |
| Frontend | **Streamlit** dashboard, local only (`localhost:8501` on the owner's PC). Pages: data status + MT5 download, market-structure chart, data quality, signals log, backtest reports, news | `dashboard/app.py`, `services.py`, `chart.py` |
| Backend | No server. The CLI and the dashboard call the library directly | `cli.py` |
| Database | **None.** Bars and ticks are Parquet files, one file per symbol and timeframe, written as upserts. Signal state is JSON files. Experiments are a hand-written Markdown table | `data/storage.py`, `live/state.py`, `docs/EXPERIMENT_LOG.md` |
| Data format | Columns `time, open, high, low, close, tick_volume, spread, real_volume`; `time` = candle open, tz-aware UTC; prices are BID; `spread` in points | `data/schema.py` |
| Data sources | MT5 (`collector.py`, broker server time converted to UTC by `ServerClock`), Dukascopy M1 download resampled to M5-H4 | `data/collector.py`, `mt5/server_time.py`, `data/dukascopy.py` |
| Data quality | Checks for unsorted, duplicate, off-grid, NaN, non-positive, impossible OHLC, spikes, missing candles (with weekend / daily-break tolerance) | `data/quality.py` |
| No-look-ahead helpers | `closed_bars()`, `as_of()`; `regime_per_bar()` maps H4 regime to M5 using closed H4 candles only | `data/candles.py`, `research/events.py` |
| Indicators | ATR, swings, structure (BOS/CHoCH), liquidity sweeps, FVG / order blocks, sessions and kill zones (DST-correct via New York clock), VWAP, volatility, news calendar | `features/` |
| Regime | 3-state H4 trend regime (old) and the rulebook 4-state regime (bullish / bearish / sideways / uncertain) | `strategy/regime.py`, `research/regime4.py` |
| Backtester A | Event-driven loop for the SMC three-mode entry model. Order life (limit fill on ask/bid, stop-first, TP1 partial, BE, time stop) is in `lifecycle.simulate`, **shared with the live signal engine** | `strategy/backtest.py`, `strategy/lifecycle.py` |
| Backtester B | Each research tool (`gtc_sbr`, `london`, `classics`, `entries`, `bigmoves`, `miner`) has **its own loop and its own fill / exit code** | `research/*.py` |
| Costs and metrics | Historical bar spread floored at $0.08, commission 0, slippage 2 points per fill (GUESS); metrics: trades, win rate, PF, net R / %, max DD % and R, expectancy, avg win/loss, max losing streak, p-value; breakdowns by side, session, month, regime | `research/metrics.py` |
| GTC-SBR | Coded to the owner's spec as one function with both directions; experiments A/B/C/F, D (session), E (side), 60/20/20 split, 6-month walk-forward (fixed rules), cost stress, nearby-parameter table, automatic rule-14 verdict. **The real-data run is queued on the owner's PC by the other thread** | `research/gtc_sbr.py`, `tests/test_gtc_sbr.py` |
| Live signals | MT5 watcher, Telegram messages, follow-ups, daily summary, lot size in the message | `live/`, `notify/` |
| Safety | `SafeMT5` wrapper blocks `order_send` / `order_check`; `tests/test_safety.py` scans the source tree; `execution_mode: signals_only` enforced by the config validator | `mt5/safety.py`, `config.py` |
| Config | YAML (`engine`, `mt5`, `data`, `symbols`, `strategy`) validated by pydantic | `config/`, `config.py` |
| Tests | 293 pytest tests (all pass, 3 min 54 s in the cloud), no MT5 needed (`tests/fake_mt5.py`, synthetic bars); GitHub Actions on Python 3.11 / 3.12 | `tests/`, `.github/workflows/tests.yml` |

The project-files folder also holds an older prototype (`trading-bot/bot/`, `main.py`); it is
not in the repository and nothing uses it. It can be ignored.

## 2. Gap analysis against the specification

Legend: **Have** = meets the spec, **Partial** = exists but needs work, **Missing**.

| Module | Requirement | Status | Notes |
|---|---|---|---|
| 1 Data | CSV import | Missing | Only MT5 and Dukascopy downloads |
| 1 | 1m / 5m / 15m | Have | M1-H4 timeframes exist |
| 1 | Future MT5 imports | Have | `collector.py` |
| 1 | UTC normalisation | Partial | MT5 path is correct; a CSV importer needs an explicit source-timezone setting |
| 1 | Missing / duplicate / OHLC / abnormal price checks | Have | `quality.py` |
| 1 | Preserve original data | **Missing** | Storage upserts in place; a re-download silently replaces earlier rows |
| 1 | Source and version tracking | **Missing** | No record of where a file came from or which version a backtest used |
| 2 Broker | Instrument spec (contract, lots, tick size/value, digits, currency, commission, execution) | Partial | Read from MT5 `symbol_info` at run time only; research code hardcodes 100 oz and does not round lots to the broker step |
| 2 | Per-broker profiles | Missing | `symbols.yaml` maps names only |
| 3 Indicators | EMA 20/50, ATR 14, EMA slope, opening range | Partial | Exist inside `gtc_sbr.py` / `modes.py`, not as shared, tested indicators |
| 3 | Session identification | Have | `features/sessions.py`, `metrics.session_of` (two session schemes) |
| 3 | Multi-timeframe sync | Partial | `as_of` and `regime_per_bar` are correct, but each strategy does its own `searchsorted` alignment |
| 4 Plug-ins | Strategy ID, version, entry / exit / risk / SL / TP / session / regime | **Missing** | GTC-SBR is a function inside its report runner; the old model is wired into `strategy/backtest.py` |
| 5 Backtester | One chronological event-driven engine with next-bar fills, spread, slippage, commission, stop-first, equity, trade history | Partial | The logic exists, but **five copies** of fill / exit code exist (lifecycle, gtc_sbr, london, classics, entries). No M1 intrabar resolution |
| 5 | Duplicate-signal / future-access prevention | Partial | Tested per strategy, not enforced by the engine |
| 6 Experiments | ID, version, params, data version, range, assumptions, results, status, notes; never overwrite; compare | Partial | Markdown log is honest but manual; report folders can be overwritten by a re-run |
| 7 Validation | Splits, walk-forward, sensitivity, cost stress, monthly, session, regime, long/short | Partial | All present inside `run_gtc` for GTC-SBR only; walk-forward never re-fits; nothing stops a re-run from reading the test set again |
| 8 Analytics | Net / gross profit, PF, win rate, expectancy, avg R, max DD, losing streak, monthly, long/short | Have | In R and %, no money values |
| 8 | Sharpe, Sortino | Missing | |
| 8 | Gross vs net clearly separated | Partial | `r_gross` vs `r` column exists, but "gross" still includes the spread |
| 9 Dashboard | 12 pages + CSV export | Partial | 6 tabs exist; none of the experiment / replay / comparison / walk-forward pages |
| 10 Tests | All listed areas | Partial | Strong on data, candles, lifecycle, safety; thin on equity / drawdown / sizing / commission maths |

## 3. Why some change is needed (specific reasons)

The spec says not to rewrite without a reason. These are the reasons for each structural change;
everything else is kept as it is.

1. **One execution simulator instead of five.** Fill, spread, stop-first and exit rules are written
   separately in five places, and they already differ slightly (for example how a gap through the
   stop is filled). A bug fixed in one is not fixed in the others, and results from two tools are
   not comparable. Module 5 asks for one engine.
2. **Strategies separate from the engine.** GTC-SBR's rules, its report and its experiment list are
   one 400-line function. Module 4 asks for bullish and bearish GTC-SBR as separate, versioned
   modules, and adding the future sideways engine there would mean copying the runner again.
3. **Immutable, versioned data.** The upsert store cannot answer "which data did EXP-014 use?",
   and a re-download can change history under an old result. Module 1 and rulebook rule 13 need it.
4. **An experiment registry the code writes.** Only a registry that the engine writes on every run
   can guarantee "never overwrite" and record the data version, git commit and test-set use.
5. **Broker spec as data.** Lot rounding, tick value and commission must come from a broker profile,
   not constants, or sizing differs between brokers (module 2).

What stays: the package layout, config system, data schema, `ServerClock`, quality checks,
features, both regime modules, metrics, safety wrapper, live engine, Telegram, Streamlit, all tests.
The existing research tools stay runnable and frozen; their past results remain valid as logged.

## 4. Proposed architecture

All new code goes into new sub-packages of `quant_engine`; existing modules are reused.

```
config/instruments.yaml          broker profiles (module 2)
src/quant_engine/
  data/
    importer.py      CSV + MT5 import -> raw copy + normalised dataset version   (module 1)
    catalog.py       dataset versions: id, source, sha256, tz, rows, range, quality report
  instruments.py     InstrumentSpec + broker profiles, lot rounding, money per point (module 2)
  indicators/        ema, ema_slope, atr, opening_range, sessions (thin wrappers over features/)
  mtf.py             align higher-timeframe values to M5 decision times, closed candles only
  strategies/
    base.py          Strategy protocol, StrategyMeta (id, version, params model), registry
    gtc_sbr.py       GTC-SBR bullish + bearish as two plug-ins sharing one opening-range component
  backtest/
    market.py        MarketView: only closed candles up to the decision time (raises on future access)
    broker_sim.py    next-bar fills, bid/ask from spread, slippage, commission, stop-first,
                     M1 intrabar resolution when M1 exists, gap handling
    engine.py        chronological loop (steps 1-12 of module 5), equity, trade ledger
  experiments/
    registry.py      append-only SQLite store + artefact folder per experiment
  validation/
    splits.py        train / validation / test split with a test-set lock
    walkforward.py   rolling or anchored windows, optional re-selection from a pre-declared small grid
    stress.py        cost stress, parameter neighbourhood
  analytics/
    performance.py   extends research/metrics.py: money values, gross vs net, Sharpe, Sortino
  dashboard/         Streamlit, more pages (module 9)
```

### 4.1 Data management (module 1)
- `import` copies the original file unchanged to `data/imports/<import_id>/` with its SHA-256,
  then writes a normalised Parquet dataset version to `data/datasets/XAUUSD/<tf>/<version>/`.
  Versions are never modified; a new import or download makes a new version.
- The importer needs the source timezone (`UTC`, a fixed offset, or a broker server zone such as
  `EET` with daylight saving). GUESS: MT5 CSV exports are in broker server time, so the default
  for an MT5 CSV is the same `ServerClock` offset rule the collector uses.
- Every version stores its quality report. Issues are reported, never patched: no missing candles
  are created, and if the file has no spread column the spread stays empty and the backtest uses the
  configured fixed spread, labelled as an assumption in the experiment.
- The current `ParquetStore` stays for the live engine and the old tools.

### 4.2 Broker configuration (module 2)
`config/instruments.yaml` holds named profiles (`default_xauusd`, later one per broker) with symbol,
contract size, min / max lot, lot step, tick size, tick value, digits, account currency, commission
per lot, swap (recorded, not charged intraday), execution assumptions (fill at next open, slippage
points, stop-first). Phase 8 adds "fill this profile from MT5 `symbol_info`". The default profile
is marked GUESS for every field the owner has not confirmed (only spread 0.08 and commission 0 are
confirmed).

### 4.3 Strategy plug-ins (module 4)
A strategy declares metadata and a parameter model, and returns **intents**; it never touches fills:

```python
class Strategy(Protocol):
    meta: StrategyMeta          # id="gtc_sbr_bull", version="1.0.0", direction, regimes, sessions
    params: BaseModel           # validated, hashed into the experiment record
    def prepare(self, view: MarketView) -> None: ...          # vectorised indicators, closed data only
    def on_bar(self, view: MarketView) -> list[OrderIntent]: ...  # entry, stop, targets, time exit, reason
```

Session and regime filters are part of the metadata, so the engine can report "blocked by regime"
counts the same way for every strategy. Changing any rule means a new version number; the registry
refuses to record two different rule sets under the same id and version.

### 4.4 Event-driven backtester (module 5)
For each M5 candle close: update the `MarketView` (M5 plus M15 / H1 / H4 candles closed by then),
update the regime, ask each enabled strategy for intents, reject duplicates (one open position per
strategy, configurable), queue orders for the **next candle's open**, then let `broker_sim` process
the next candle: fill with spread and slippage, charge commission, check stop before target, apply
time exits, update equity and write the trade ledger.

- Intrabar: if M1 candles exist for that 5 minutes, the simulator walks the M1 candles in order to
  decide whether stop or target came first; if not, **stop first**. Each trade records which method
  was used, so results can be split by it.
- Look-ahead guard: `MarketView` only exposes arrays sliced to the decision time; a test runs every
  strategy on data cut at each decision time and checks the signals are identical to the full run.
- Speed: precomputed indicators plus a light Python loop should handle 2020-2026 M5 (about 500k
  candles) in under a minute per run on the owner's PC. GUESS, to be measured in Phase 3.

### 4.5 Experiment manager (module 6)
SQLite file `data/experiments.db` (append-only table; updates only to `status` and `notes`, and
each update is itself logged) plus a folder per experiment with `trades.csv`, `equity.csv`,
`report.md` and the exact config. Each record: id, strategy id + version, params + hash, dataset
version ids, date range, execution assumptions, git commit, results, status (rule 14 classes),
notes, whether the test set was read. `docs/EXPERIMENT_LOG.md` stays as the human log; new rows can
be generated from the registry.

### 4.6 Validation (module 7)
Splits are declared per research programme and stored in the registry. The **test part is locked**:
reading it needs an explicit `--final-test` flag, is recorded on the experiment, and a second read
of the same test period for the same strategy family is flagged in the report. Walk-forward offers
fixed-rule windows (what exists now) and true re-selection from a small grid declared before the
run (rulebook rule 7 forbids large sweeps).

### 4.7 Analytics (module 8)
Extend `research/metrics.py` instead of replacing it. Add money values from the instrument profile,
three clearly labelled layers (**before costs**, **after spread**, **after all costs**), Sharpe and
Sortino from daily equity returns (annualised with 252 days, GUESS), and monthly returns in %.

### 4.8 Dashboard (module 9)
Keep **Streamlit** (already installed and working on the owner's PC; it is a web UI in the browser).
Add pages: overview, data manager (imports, versions, quality), strategy manager (registered
strategies and versions), experiment manager and comparison, backtest configuration and launch,
equity and drawdown charts, trade history with CSV export, trade replay (candles with entry, stop,
target and exit markers), walk-forward view, performance report export. A separate FastAPI + React
app would work too but adds a second stack and a build step on Windows; I recommend against it now.

### 4.9 Testing (module 10)
Deterministic fixtures: small hand-built CSV files in `tests/fixtures/` whose correct answers are
worked out by hand (for example a trade where stop and target are both inside one M5 candle, with
and without M1 data). New test files per module; all tests keep running without MT5.

## 5. Phase plan

| Phase | Deliverable | Depends on |
|---|---|---|
| 2 | `importer`, `catalog`, `instruments.yaml` skeleton, CSV fixtures, tests | approval of this report |
| 3 | `MarketView`, `broker_sim`, `engine`, `mtf`, look-ahead harness, tests | 2 |
| 4 | GTC-SBR bull + bear plug-ins; check they reproduce the existing `research/gtc_sbr.py` trades on fixtures (differences explained) | 3 |
| 5 | Analytics layers, Sharpe / Sortino, money values | 3 |
| 6 | Registry, splits with test lock, walk-forward, stress | 4, 5 |
| 7 | Dashboard pages | 6 |
| 8 | MT5 broker spec import, MT5 history import into versioned datasets (on the owner's PC) | 2 |
| 9 | Independent audit of each strategy, rule-14 classification | all |

Each phase ends with a report, a full test run and a pause for the owner's approval of any major
architectural change, as the spec asks.

## 6. Outstanding issues and questions

1. **PR #8 is not merged.** This plan builds on it. Either merge PR #8 first, or I stack the new
   branch on top of it (the default I will use).
2. **GTC-SBR real-data run belongs to the other thread** ("Find a gold strategy with 50%+ wins"),
   queued on the owner's PC. This thread will not run it again; Phase 4 re-implements GTC-SBR as
   plug-ins and checks that both versions give the same trades.
3. **Data is only on the owner's PC.** Phases 2-7 are built and tested on synthetic fixtures in the
   cloud. Real imports happen on the PC (Phase 8, or earlier with a CSV export).
4. **M1 history?** Intrabar resolution needs M1 candles. Does the PC have M1 history back to 2020?
   If not, every backtest uses stop-first, which is conservative but costs some accuracy on targets.
5. **Historical spread.** MT5 bar `spread` is a single value per candle (GUESS: the broker's lowest
   spread in the candle, which understates costs). The engine will floor it at the configured 0.08
   and report runs at 1.5x and 2x.
6. **Untouched data.** 2020-2026 has been looked at heavily by earlier experiments (see the log). For
   GTC-SBR, a brand-new spec, the last 20% of 2020+ can still serve as its test set, but every
   later strategy chosen after seeing GTC-SBR's results should be treated with that in mind;
   forward paper trading is the only fully clean test.
7. **Two session schemes** (kill zones in `features/sessions.py`, rule-11 sessions in
   `research/metrics.py`). The engine will report rule-11 sessions; kill zones stay for the old model.
8. **Broker profile values** other than spread and commission are GUESSES until the broker is chosen.

_No performance claims are made in this report. Signals and research only; no order execution._
