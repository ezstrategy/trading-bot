# XAUUSD Quant Engine — permanent project rules

These rules apply to every phase and every contributor (human or AI).
If a request conflicts with them, stop and ask the owner.

## What this project is
- A **signals-only** engine for **XAUUSD** intraday scalping.
- Entry timeframe **M5**, confirmation **M15**, context **H1/H4**.
- Risk model: **0.5% per trade** (used only to size the numbers shown in a signal).
- Signals are delivered to Telegram (from Phase 2+). Broker not selected yet; data comes from MetaTrader 5.
- Other symbols are out of scope until the owner asks.

## Hard safety rules
1. **Never place, modify or close orders.** Never call `mt5.order_send()` or `mt5.order_check()`.
   All MT5 access goes through `quant_engine.mt5.safety.SafeMT5`, and `tests/test_safety.py`
   scans the source tree. Do not weaken either.
2. `execution_mode` must stay `signals_only`; the config validator rejects anything else.
3. **No hardcoded credentials.** MT5 login/password/server and Telegram tokens come from
   environment variables / `.env` only. Never log or commit them. `.env` is git-ignored.

## Data and time rules
4. **All stored and computed timestamps are UTC** (tz-aware). MT5 returns broker *server* time;
   convert with `ServerClock` at the boundary (collector) and nowhere else.
5. Bar `time` is the candle **open** time. A candle is final only at `open + timeframe`.
6. **No incomplete candles in signals.** Use `closed_bars()` / `as_of()` from `data/candles.py`.
7. **No look-ahead bias.** A decision at time *t* may only use data whose close time is <= *t*,
   on every timeframe (e.g. an H1 candle is usable only after it closes). Swing points are only
   usable after their confirmation bars have closed.
8. Validate data (`data/quality.py`) before using it for research; report issues, never silently
   drop or patch bars.

## Engineering rules
9. Python 3.10+, type hints and docstrings on public functions, modular packages under `src/quant_engine/`.
10. Structured logging via `quant_engine.logging_setup.get_logger` (JSON lines, extra fields
    via `extra=`). No `print` in library code (CLI output only).
11. Handle MT5 disconnections gracefully (retry with backoff through `MT5Client.call`).
12. Every feature ships with pytest tests that run **without** MetaTrader 5 (use `tests/fake_mt5.py`).
    Run `pytest` before every commit; all tests must pass.
13. **No strategy performance claims** (win rate, profit factor, returns) unless produced by a
    committed, reproducible backtest test or script whose output is included.
14. Settings live in `config/*.yaml`; don't hardcode thresholds that a trader may want to change.

## Process
15. Work in phases. Finish the current phase, run the tests, fix failures, write a completion
    report, then **wait for the owner's review before starting the next phase**.
16. Mark assumptions clearly ("GUESS:" in code comments or in the report) and ask when unsure.

## Research rules (owner, 2026-11-26)
17. All strategy research follows `docs/RESEARCH_RULEBOOK.md` (no mass parameter sweeps,
    development / validation / untouched out-of-sample split + walk-forward, full costs incl.
    commission and slippage, the 14 required metrics, independent audit, rule-14 classification).
18. Log every experiment in `docs/EXPERIMENT_LOG.md`; never overwrite a tested version.
