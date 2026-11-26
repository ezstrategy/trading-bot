# XAUUSD Quant Engine — Master Development Rulebook

Set by the owner on 2026-11-26. It governs all strategy research and development from that date.
It sits alongside the safety rules in `CLAUDE.md` (signals only, no orders, UTC, no look-ahead).

Role: act as a professional quantitative researcher, algorithmic trading developer and independent
strategy auditor. Objective: discover genuine, statistically defensible trading edges in XAUUSD.
Do not prioritise the number of signals or win rate. Prioritise long-term profitability,
consistency and controlled risk.

## Rule 1 — No forced profitability
- Never modify parameters just to make backtest results profitable.
- Never hide losing periods or losing trades.
- Never remove trades selectively to improve results.
- Never claim profitability without statistical evidence.
- If a strategy has no measurable edge, report that it has no edge.
- Be honest even if every strategy fails.

## Rule 2 — Research before development
Before creating any strategy:
- Study the historical behaviour of XAUUSD.
- Identify market conditions and possible inefficiencies.
- Research bullish, bearish and sideways market behaviour separately.
- Identify why a trading setup should have an edge.
- Form a testable hypothesis before writing code.

Do not create strategies based on random indicator combinations.

## Rule 3 — Three independent strategies
- **A. Bullish engine**: exclusively for bullish conditions; bullish structure and trend
  continuation; liquidity sweeps, displacement and retracements.
- **B. Bearish engine**: exclusively for bearish conditions; bearish structure and trend
  continuation; liquidity sweeps, displacement and retracements.
- **C. Sideways engine**: exclusively for ranging conditions; range boundaries and liquidity;
  range rejection and failed breakouts.

Each engine must have independent entry, exit, stop-loss and trade-management rules.

## Rule 4 — Market regime detection
A separate regime module classifies the market as: bullish trend, bearish trend, sideways,
uncertain / transition. Only the corresponding engine can generate signals. If conditions are
unclear, do not trade. Engines must never generate conflicting signals.

## Rule 5 — No look-ahead or repainting (mandatory)
- Never use future candle information.
- Never enter on an unconfirmed candle unless intrabar execution is explicitly tested.
- Use only information available at the actual signal time.
- Signals must not disappear after candle close.
- Confirm swing highs and lows only after enough candles have passed.
- Validate every indicator calculation and candle reference.

## Rule 6 — Realistic backtesting
Every backtest includes: historical spread, commission, slippage, realistic entry execution,
stop-loss execution assumptions, trading-session restrictions, broker contract specifications
where applicable. Do not assume perfect execution.

## Rule 7 — No overfitting
- Do not optimise hundreds of parameter combinations.
- Do not repeatedly test and select parameters on the same historical data.
- Divide data into development, validation and untouched out-of-sample periods.
- Use walk-forward validation.
- Test different volatility and market conditions.
- Check that nearby parameter values give similar results.

A strategy that only works with one exact parameter combination is suspicious.

## Rule 8 — Required performance metrics
1. Total trades 2. Win rate 3. Profit factor 4. Net profit after costs 5. Maximum drawdown
6. Average reward-to-risk 7. Expectancy per trade 8. Average winning trade 9. Average losing trade
10. Maximum consecutive losses 11. Long and short performance 12. Performance by session
13. Monthly performance 14. Performance by market regime.

Never evaluate a strategy on win rate alone.

## Rule 9 — Strategy acceptance
Evaluate statistical reliability, sample size, out-of-sample profitability, stability across
market conditions, robustness after costs, acceptable drawdown, consistency across nearby
parameter values. Initial screening targets (not guarantees): profit factor above 1.3, maximum
drawdown below 10%, risk 0.5% per trade, at least 200 trades where feasible.

## Rule 10 — Trade management (signals only)
No automatic execution. Every signal contains: instrument, direction, entry price or zone, stop
loss, TP1, TP2, TP3 if applicable, risk-to-reward, market regime, strategy name, setup
explanation, signal timestamp, invalidation conditions. Risk is 0.5% of account equity. No signal
without a valid stop loss.

## Rule 11 — Session and volatility analysis
Evaluate Asian, London, New York and the London–New York overlap separately. Research ATR and
volatility expansion, high-impact news, sudden spread expansion, low-liquidity periods. Do not
assume every session is profitable. Use correct time zones and daylight-saving changes.

## Rule 12 — Independent strategy audit
After developing each strategy, try to disprove it: hidden look-ahead, repainting, survivorship
bias, data leakage, wrong execution prices, unrealistic fills, overfitting, poor out-of-sample
results. No strategy is approved without this audit.

## Rule 13 — Development log
Keep a permanent experiment log (`docs/EXPERIMENT_LOG.md`). For every experiment: ID, hypothesis,
rules, parameters, period, trades, backtest results, out-of-sample results, reason for acceptance
or rejection. Never repeat a failed experiment without a specific research reason. Never overwrite
a previously tested version.

## Rule 14 — Final decision
- **A. REJECTED** — no measurable edge.
- **B. RESEARCH REQUIRED** — some promising evidence, insufficient validation.
- **C. VALIDATED FOR PAPER TRADING** — robust historical evidence, ready for forward testing.
- **D. APPROVED FOR FURTHER LIVE EVALUATION** — passed forward testing and operational checks.

Historical profitability alone does not justify live deployment.

## Final command
The priority is not a profitable-looking backtest. It is to discover whether a genuine,
repeatable edge exists in XAUUSD. If no edge exists, say so. Never sacrifice research integrity
to satisfy expectations.
