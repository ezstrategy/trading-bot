# XAUUSD Quant Engine

A **signals-only** engine for XAUUSD intraday scalping (M5 entries, M15 confirmation,
H1/H4 context). It reads market data from MetaTrader 5 and, from Phase 2 on, will post
trade signals to Telegram. **It never places orders.**

> Status: **Phase 1 – data layer** (MT5 connection, symbol discovery, UTC bars and ticks,
> data-quality checks, historical downloader, tests). No strategy or signals yet.

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

1. **Phase 1 – data layer** (this)
2. Phase 2 – market structure features (confirmed fractals, ATR, sessions in New York time), awaiting review
3. Later – strategy rules from the strategy spec, backtests, Telegram signals, news filter
