# XAUUSD Quant Engine

A **signals-only** engine for XAUUSD intraday scalping (M5 entries, M15 confirmation,
H1/H4 context). It reads market data from MetaTrader 5 and, from Phase 2 on, will post
trade signals to Telegram. **It never places orders.**

> Status: **Phase 2 – market structure features** on top of the Phase 1 data layer.
> The engine can describe the market exactly as the strategy spec defines it, with no
> look-ahead. It does **not** produce trade signals yet (Phase 3).

## What Phase 1 gives you

| Piece | Where | What it does |
|---|---|---|
| MT5 connection | `src/quant_engine/mt5/client.py` | Connects with credentials from `.env`, retries with backoff, reconnects if the terminal drops |
| Trade guard | `src/quant_engine/mt5/safety.py` | Blocks `order_send` / `order_check`; a test also scans the code |
| Symbol discovery | `src/quant_engine/mt5/symbols.py` | Finds your broker's gold symbol (`XAUUSD`, `XAUUSDm`, `GOLD`, `XAUUSD.r` …) |
| Server time → UTC | `src/quant_engine/mt5/server_time.py` | Converts broker server time (usually GMT+2/+3) to true UTC, DST-aware |
| Bars and ticks | `src/quant_engine/data/collector.py` | M1, M5, M15, H1, H4 candles and bid/ask ticks, UTC, closed candles only |
| No look-ahead helpers | `src/quant_engine/data/candles.py` | `closed_bars`, `as_of` for safe multi-timeframe reads |
| Data quality | `src/quant_engine/data/quality.py` | Missing, duplicated, misaligned, invalid OHLC, NaN, spikes; bad ticks |
| Storage | `src/quant_engine/data/storage.py` | Parquet files under `data/raw/XAUUSD/` (safe to re-download) |
| Downloader + CLI | `src/quant_engine/data/downloader.py`, `cli.py` | Chunked, resumable history download with a JSON quality report |

## What Phase 2 adds (`src/quant_engine/features/`)

All settings are in `config/strategy.yaml` and follow *XAUUSD & Forex 5-Min Scalping Strategy Spec v1*.

| Feature | Module | Rule |
|---|---|---|
| ATR(14) | `atr.py` | Wilder ATR on M5, M15, H1, H4 |
| Sessions | `sessions.py` | Asia, London and New York kill zones, London close, rollover in **New York time** (DST handled); trading day 17:00–17:00 NY; no new signals Friday after 10:00 NY or on Sunday |
| Swings | `swings.py` | Confirmed fractals, L=2 on M5 and L=3 on M15/H1/H4; each swing carries the time it became known |
| Structure | `structure.py` | BOS / CHoCH on closes; state bullish / bearish / ranging per timeframe |
| Bias | `context.py` | H4/H1 bias table (20/10 points), M15 confirmation gate, H1 premium/discount |
| Imbalance | `imbalance.py` | Displacement (body ≥ 1.2 ATR and ≥ 60% of range), fair value gaps with size check and invalidation, order blocks |
| Liquidity | `liquidity.py` | Previous day, Asia, London highs/lows, H1 swings, equal highs/lows, M5 swings; sweeps (1 tick to 1.5 ATR beyond, reclaim within 3 candles, no two closes beyond, in a kill zone) |
| Filters | `volatility.py` | ATR vs same-time-of-day median (dead < 0.6, chaos > 2.5), spread vs 20-day median |
| Snapshot | `context.py` | `MarketFeatures(...).snapshot(t)`: everything knowable at time *t* |

**No look-ahead, tested:** `tests/test_context.py` rebuilds every feature on data cut off at
a decision time and checks the snapshot is identical to the one built from the full history.
Any feature that peeked at later candles would fail it.

See the context on your own downloaded data:

```powershell
python -m quant_engine context                       # at the last stored M5 close
python -m quant_engine context --at "2026-11-24 13:05"
```

Not in Phase 2: the entry model, confidence score, news calendar, risk sizing and Telegram
signals (Phase 3 and later).

## Dashboard

A simple local web page to operate the engine (signals only, it cannot trade):

```powershell
python -m quant_engine dashboard        # or double-click start_dashboard.bat
```

It opens at http://localhost:8501 (only reachable from your own PC) and gives you:

* **Check MT5 connection**: broker symbol, server-time offset, last tick in UTC.
* **Download history**: pick days and timeframes; the download resumes where it stopped.
* **Live mode**: auto-refresh from MT5 every 30 s to 5 min.
* **Market context**: H4/H1 bias, structure on every timeframe, session, signal block, volatility.
* **Chart**: XAUUSD candles (M5, M15, H1, H4) with kill zones, liquidity levels, fair value gaps,
  sweeps, BOS/CHoCH and the H1 equilibrium. Times in IST, UTC or New York time.
* **Review a past moment**: see exactly what was known at any earlier time (no look-ahead).
* **Data quality** table for every stored timeframe.

## Phase 3: entry model and backtest (`src/quant_engine/strategy/`)

* `entry.py`: on each closed M5 candle that makes a CHoCH, runs the Strategy Spec v1 gates
  (kill zone, H4/H1 bias + M15, liquidity sweep, displacement + FVG, discount/premium entry,
  volatility/spread, stop 0.5-2.5 ATR, TP2 >= 2R) and scores the setup 0-100 (A+ / A / B).
  Every rejected candle is counted by reason (the "signal funnel").
* `backtest.py`: simulates the limit order on BID candles with the spread paid on every trade,
  stop checked before targets, TP1 50% + breakeven, TP2/TP3, 30-minute expiry, 2-hour time stop,
  session close, and the risk locks (daily -2%, weekly -5%, 2 losses in a row, 2 signals per session).
* `regime.py`: labels each trade's market as **bullish / bearish / neutral** (H4 close vs EMA50
  and the EMA's slope), independent of the strategy's own structure reading.
* `report.py` / `runner.py`: results overall and by regime, H4 structure, grade, side, kill zone and month.

Free real history without MT5 (Dukascopy spot gold, UTC):

```powershell
python -m quant_engine dukascopy --days 180         # resumes; stores XAUUSD_DUKASCOPY bars
python -m quant_engine backtest --source dukascopy  # or double-click start_backtest.bat
python -m quant_engine backtest                     # on your MT5 bars from 'download'
```

Reports go to `reports/backtest_<source>_<time>/` (`report.md`, `summary.json`, `trades.csv`).
Dukascopy candles are BID only, so the spread is assumed (`backtest.assumed_spread_points`).
No news filter is applied yet.

## Three strategy modes (one per market)

`strategy.mode: regime` (default) asks the H4 regime detector which market it is at each M5 close
and uses that market's own strategy. Each one has its own setup, trigger, entry, stop and exits:

| Market | Strategy | Why it fits | Trigger and entry | Exits |
|---|---|---|---|---|
| Bullish | **Buy the dip** | Gold uptrends grind up in steps; pullbacks are shallow, hold the higher low and get bought at dynamic support | Pullback to the M15 EMA20/50 zone or H1 discount, higher low intact; M5 **break of structure up** with displacement; buy the FVG | TP1 1R (50%), TP2 next buy-side liquidity or 2R, **runner TP3 3R**, 3-hour time stop |
| Bearish | **Sell the rally** | Gold sell-offs are fast; counter-trend rallies are weak and usually raid buy stops before rolling over | Rally into the H1 premium that **raids an M5 high**; M5 **change of character down**; sell the FVG | TP1 1R (50%), TP2 must be real sell-side liquidity at 2R+, **no runner**, 90-min time stop |
| Sideways | **Fade the false breakout** | Ranges hold their edges; breakouts without expansion fail and return to the middle | H1 range 2-8x ATR, volatility not expanding; **sweep of a range edge that closes back inside**; buy/sell the **retest of the edge** (no displacement needed) | TP1 **range middle**, TP2 just before the **opposite edge** (1.5R+), no runner, 90-min time stop |

Test them: `python -m quant_engine backtest --mode compare` runs every mode in every market
(`comparison.md`), so you can check each one really does best in its own market. Settings: `modes:` in
`config/strategy.yaml`. `mode: v1` switches back to the original sweep-reversal model.

## Pattern research (`src/quant_engine/research/`)

Looks for repeatable gold moves in each H4 market type (bullish / bearish / neutral)
instead of hand-written rules. Double-click **find_patterns.bat** (MT5 open), or:

```
python -m quant_engine download --days 365
python -m quant_engine patterns --source mt5      # report in reports/patterns_mt5_<time>/report.md
```

- Candidates: time of day, liquidity sweeps of the last 2 h / 6 h / previous day, closed
  breakouts, impulse candles, big 1-hour moves and range-edge touches, each x market type x
  session x BUY/SELL x exit rule (stop, target and time limit in M5 ATR). Settings: `research:`
  in `config/strategy.yaml`.
- Also: round-number levels ($50 / $100 handles) rejected or broken, 5-day and 20-day highs/lows
  (major support / resistance) swept or broken, pin bars at those levels, and big candles at US
  release times (08:30 / 10:00 / 14:00 New York, NFP Fridays). One fixed spread (20 points) is
  charged in every year by default. The report shows, per idea family, how many candidates looked
  significant versus what luck alone would give.
- Found on the older 60% of the history only, with a Benjamini-Hochberg false-discovery
  filter across all candidates, a minimum edge after spread and a stability check.
- Checked on the newer 40% it never saw: `confirmed`, `held up (not proven)`, `failed`.
  All discovery survivors also run through the normal backtester on that unseen period.
- Swing version: `patterns --timeframe H1` or `--timeframe H4` finds the same ideas on H1/H4
  candles with bigger stops and targets (H1/H4 ATR) and holds up to 1-3 days (`research.timeframes`).
  Signals are still sent on the M5 candle that closes with the H1/H4 candle.
- Surviving patterns are saved to `reports/patterns_latest.json`; `strategy.mode: patterns`
  (or `backtest --mode patterns`) signals from the `confirmed` ones. Signals only.

## Phases 4-6: live signals, Telegram, news filter

```powershell
python -m quant_engine telegram-test     # checks TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env
python -m quant_engine run               # or double-click start_signals.bat
python -m quant_engine run --dry-run     # print messages here instead of Telegram
python -m quant_engine news              # refresh the calendar and list upcoming news blocks
```

* **Live engine** (`live/engine.py`): every 10 s it reads closed candles from MT5. When a new M5
  candle closes it runs the *same* entry model and risk locks as the backtest, and sends a signal
  only for that candle (never a stale one, one open signal at a time). Lot size is for 0.5% of the
  MT5 account balance (`live/sizing.py`, rounded down to the broker's lot step).
* **Telegram** (`notify/`): the signal (entry, stop, TP1/TP2/TP3, lots, grade, reasons), then follow-ups
  as replies: filled, TP1 (move stop to breakeven), TP2/TP3, stopped out, expired or cancelled, time
  stop, session close, and a news warning 10 min before high-impact USD news while a signal is open.
  A daily summary is sent at 22:30 IST. Follow-ups come from the same `strategy/lifecycle.py` the
  backtest uses, so live and backtest manage trades identically. They describe what the plan says
  should happen at the signal's prices; they do not read your account.
* **News filter** (`features/news.py`): no new signals from 60 min before to 30 min after NFP/CPI/FOMC
  (FOMC 90 min after), 30/15 min around other high-impact USD news, and all day on US bank holidays.
  Calendar: the free ForexFactory weekly export, refreshed hourly and cached in `data/news/calendar.csv`.
  The backtest applies it only for periods the cached calendar covers.
* **Dashboard**: new **Signals** (sent signals, results, message log), **Backtest** (run and read reports,
  results by bullish/bearish/neutral market) and **News** tabs.
* State survives restarts (`data/signals/state.json`); a restart never re-sends a signal.

## Quant Lab (research platform)

The owner's Quant Lab specification is being built in phases on top of this engine. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the plan and [docs/DATA_MANAGEMENT.md](docs/DATA_MANAGEMENT.md)
for Phase 2: versioned, never-modified research datasets (`data-import`, `data-snapshot`,
`data-list`, `data-show`, `data-verify`, `import_csv.bat`, `snapshot_data.bat`) and broker
instrument profiles (`config/instruments.yaml`, `instrument`).

## Requirements

* **Windows** PC with the **MetaTrader 5 terminal** installed and logged in to a broker
  account (a **demo** account is recommended). The `MetaTrader5` Python package only works on Windows.
* **Python 3.11 or 3.12**, 64-bit (from python.org; tick "Add python.exe to PATH").
* In MT5: *Tools → Options → Charts* set **Max bars in chart** to *Unlimited* so long histories download.

The tests run on any OS without MT5.

## Installation (Windows, PowerShell)

```powershell
git clone https://github.com/ezstrategy/trading-bot.git
cd trading-bot
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env
notepad .env          # fill MT5_LOGIN, MT5_PASSWORD, MT5_SERVER (and MT5_PATH if you have several terminals)
```

If you leave the MT5 credentials empty, the engine attaches to whatever account the open
terminal is already logged in to.

## Usage

Open the MT5 terminal first, then:

```powershell
$env:PYTHONPATH="src"                                   # or: pip install -e .
python -m quant_engine check                            # connection, broker symbol, server-time offset
python -m quant_engine symbols                          # gold symbols your broker offers (--all for everything)
python -m quant_engine download                         # last 90 days of M1, M5, M15, H1, H4
python -m quant_engine download --timeframes M5,M15 --start 2026-06-01 --end 2026-11-24
python -m quant_engine ticks --start 2026-11-20 --end 2026-11-21   # bid/ask ticks
python -m quant_engine validate                         # quality checks on stored data
python -m quant_engine latest --timeframe M5 --count 10 # last closed candles
```

Data is saved to `data/raw/XAUUSD/<TF>.parquet` and `data/raw/XAUUSD/ticks/<date>.parquet`.
Each download writes a quality report to `reports/download_<time>.json`. Logs are JSON
lines in the console and in `logs/engine.log`.

## Configuration

| File | Main settings |
|---|---|
| `config/engine.yaml` | instrument, timeframes, risk %, `execution_mode: signals_only`, logging |
| `config/mt5.yaml` | timeouts/retries, **server time mode** |
| `config/symbols.yaml` | XAUUSD → broker symbol mapping (`broker_symbol` to force one) |
| `config/data.yaml` | storage folder, download chunk sizes, market hours for gap checks |

### Getting time exactly right

MT5 stamps candles with the **broker's server clock**, not UTC. The engine converts at the
boundary, so everything stored is UTC.

* `mode: auto` (default) measures the offset from the latest tick when the market is open.
  If it matches the common "New York close" convention (GMT+2 winter / GMT+3 summer) it uses
  that model, which also handles **past DST changes** correctly.
* Otherwise set `mode: fixed` with your broker's offset, or `mode: timezone` with an IANA zone.
* Run `python -m quant_engine check` while the market is open and compare `last_tick_utc` with
  the real UTC time (e.g. time.is/UTC). They should match within a few seconds.

H4 candles are aligned to the broker's server midnight (22:00 or 21:00 UTC), which is normal.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q --cov=quant_engine
```

The suite uses `tests/fake_mt5.py`, a simulated terminal that reproduces server-time
timestamps, the forming candle, weekend/daily closures, the 1 Nov 2026 US DST change,
dropped connections and failed logins. GitHub Actions runs it on every push.

## Safety

* Signals only: there is no code path that sends orders, and the MT5 wrapper refuses to.
* Credentials live only in `.env` (git-ignored) and are masked in logs.
* No performance claims are made anywhere without a reproducible backtest.

## Roadmap

1. **Phase 1 – data layer** (done)
2. **Phase 2 – market structure features** (done)
3. **Phase 3 – entry model, confidence score and backtest** (done)
4. **Phase 4 – live signal engine and lot sizing** (done)
5. **Phase 5 – Telegram signals and follow-ups** (done)
6. **Phase 6 – news filter and dashboard Signals/Backtest/News tabs** (done)
7. Next – run the backtest on real history, tune the score weights from its results, then other pairs
