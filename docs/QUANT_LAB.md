# XAUUSD Quant Lab: user guide

Research and signals only. Nothing in the Quant Lab can place, change or close a trade.
Architecture and reasons: [ARCHITECTURE.md](ARCHITECTURE.md). Data: [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md).

## The workflow on your PC

| Step | Double-click | Same as |
|---|---|---|
| 1. Get data (M1, M5, M15 since 2020) and the broker's specification | `lab_import_mt5.bat` | `mt5-import`, `mt5-instrument --profile my_broker` |
| 2. Test GTC-SBR on development and validation data, then walk-forward, sensitivity and cost stress | `lab_gtc_sbr.bat` | `lab-run`, `lab-validate` |
| 3. Look at the results | `start_dashboard.bat`, tab **Quant Lab** | `lab-list`, `lab-show LAB-0001` |
| 4. Audit a run | | `lab-audit LAB-0001` |
| 5. Only at the very end, once: the untouched test period | | `lab-run --part test --final-test`, then `lab-audit` |

## Main ideas

- **Research programme.** A name such as `gtc_sbr_2020`. The first time it is used, its data is split
  by time into development (60%), validation (20%) and an untouched test period (20%). The split
  is stored permanently; it can never move. A new idea that was shaped by looking at results
  should get a new programme name.
- **Test lock.** `--part test` (or `all`) is refused unless `--final-test` is given. Every test
  use is recorded, and the report says how many times that strategy family has already seen the test.
- **Experiments are never overwritten.** Each run gets an id (`LAB-0001`, ...), a folder with
  `report.md`, `trades.csv`, `equity.csv`, `signals.csv`, `monthly.csv` and `experiment.json`, and a
  row in `data/experiments/registry.db`. The database refuses edits and deletes. Notes and status
  changes (`lab-note LAB-0001 "text" --status "RESEARCH REQUIRED"`) are added, the old ones stay.
- **Gross vs net.** Reports show three columns over the same trades: before costs, after spread,
  and NET (after spread, slippage and commission). Only NET is what an account would get.

## The engine (what a backtest assumes)

- Steps through closed M5 candles in time order. Strategies only see candles that have closed;
  higher timeframes (M15, H4) only after they close. The look-ahead harness re-runs strategies on
  data cut at several moments and fails if any signal changes.
- Orders fill at the **next candle's open**. Prices are bid; buys pay bid + spread, shorts are
  stopped and take profit on the ask. Slippage (instrument profile, default 2 points) is charged
  on entries, stops and time exits, and on targets too (conservative, configurable).
- Spread: the candle's spread from the data, never below `typical_spread` (0.08); when the data
  has none, 0.08 is assumed and the share of such candles is printed in the report.
- Stop and target in one M5 candle: the M1 candles decide when M1 data exists; otherwise the
  **stop is assumed first**. Each trade records how it was resolved (`intrabar` column).
- A candle that opens beyond the stop fills at the open. A missing candle before a fill cancels
  the order. One position per strategy and one in total by default.
- Size: 0.5% of current equity at the filled stop distance, rounded **down** to the broker's lot
  step; if even the minimum lot risks more, the trade is skipped (never rounded up).

## Strategy plug-ins

`lab-strategies` lists them with their rules. A strategy has an id, a version, entry / exit /
stop / target / session / regime rules and validated parameters, and only returns order intents;
the engine does all filling and accounting. Changing a rule means a new version.

- `gtc_sbr_bull` 1.0.0 and `gtc_sbr_bear` 1.0.0: GTC-SBR as specified by the owner, split by
  direction. On synthetic test data they give exactly the same trades as the earlier `gtc` command (test
  `test_same_trades_as_the_existing_gtc_sbr_code`). Differences from the old command: a data gap is
  handled per order (the old one skipped a setup if any candle was missing in the next hour), and
  trades of both directions share the one-position limit in the engine.

Adding a strategy (for example the future sideways engine): a class in
`src/quant_engine/strategies/` with `meta`, a `Params` model, `prepare()` and `on_bar()`,
decorated with `@register`, plus tests including `check_no_lookahead`.

## Validation (`lab-validate`)

Runs on development + validation data only:

- walk-forward: 24-month training / 6-month test windows (`config/lab.yaml`); with `--grid` at most
  12 parameter sets are compared on each training window and the best is traded on the next test
  window; without it the rules stay fixed;
- parameter sensitivity (`--neighbours`): each setting moved to nearby values on development data;
  nothing is chosen from this table;
- cost stress: spread x1.5 and x2, 5 points slippage, $7/lot commission;
- monthly, session, H4 regime, long/short and year tables.

## Audit (`lab-audit LAB-xxxx`)

Checks a backtest without trusting it: data checksums, blocking data issues, identical re-run
from the record, rule set unchanged, look-ahead harness on the real data, next-open fills, stops
on the right side, gross - costs = net on every trade, equity recomputes, no overlapping or
duplicate trades, no trade above 0.5% risk. Then it classifies under rule 14:
REJECTED / RESEARCH REQUIRED / VALIDATED FOR PAPER TRADING. Validation for paper trading needs
every acceptance target (PF > 1.3, DD < 10%, 200+ trades, p-value corrected for the number of
backtests in the programme, more than half of walk-forward windows positive, positive at spread
x1.5) **and** the run must be the final test. APPROVED FOR LIVE EVALUATION is never automatic.

## Dashboard

Tab **Quant Lab**, 12 pages: overview, historical data (versions, quality, CSV import), strategies,
experiments (notes, status), backtest configuration, equity curve, drawdown, trade history (CSV),
trade replay (candles with entry / stop / target / exit), strategy comparison, walk-forward, reports
(every file downloadable).
