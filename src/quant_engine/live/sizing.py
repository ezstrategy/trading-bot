"""Position size shown in a signal (the engine never trades; this is advice text).

Owner rule (2026-11-27), "natural stop first, position sizing second": the stop comes from the
strategy (market structure) and is never moved to fit the account. The lot size is worked out from
that stop, all trading costs included, for at most ``risk_pct`` of equity. If even the broker's
minimum lot would risk more than that, the signal is skipped (``Size.skipped``), never resized by
moving the stop or raising the risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Size:
    lots: float
    risk_money: float          # account currency lost at the full stop, costs included
    note: str = ""
    skipped: bool = False      # True: no position fits the risk limit; do not take the signal


def lot_size(balance: float, risk_pct: float, entry: float, stop: float, contract_size: float,
             volume_min: float, volume_step: float, volume_max: float | None = None, *,
             cost_per_unit: float = 0.0, commission_per_lot: float = 0.0,
             skip_if_over: bool = False) -> Size:
    """Lots that lose at most ``risk_pct`` of ``balance`` at the stop, rounded DOWN to the
    broker's volume step.

    The loss per lot is ``(|entry - stop| + cost_per_unit) * contract_size + commission_per_lot``,
    where ``cost_per_unit`` is the price-unit cost not already in ``entry`` / ``stop`` (spread,
    slippage on both fills). The stop itself is never changed here.

    With ``skip_if_over`` a size below ``volume_min`` returns ``skipped=True`` and 0 lots (owner
    rule 2026-11-27); without it the old behaviour is kept (minimum lot plus a warning note).

    GUESS: the account currency equals the symbol's profit currency (USD for XAUUSD);
    for other account currencies the amount is off by the exchange rate.
    """
    risk_money = balance * risk_pct / 100.0
    dist = abs(entry - stop)
    per_lot = (dist + max(cost_per_unit, 0.0)) * contract_size + max(commission_per_lot, 0.0)
    if dist <= 0 or per_lot <= 0 or volume_step <= 0 or balance <= 0:
        return Size(0.0, 0.0, "invalid stop, balance or volume step", skipped=skip_if_over)
    raw = risk_money / per_lot
    lots = math.floor(raw / volume_step + 1e-9) * volume_step
    lots = round(lots, 8)
    if volume_max:
        lots = min(lots, volume_max)
    if lots < volume_min - 1e-12:
        min_pct = 100 * volume_min * per_lot / balance
        if skip_if_over:
            return Size(0.0, 0.0, f"skipped: minimum lot {volume_min:g} would risk {min_pct:.2f}% "
                                  f"(limit {risk_pct:g}%)", skipped=True)
        return Size(volume_min, volume_min * per_lot,
                    f"minimum lot {volume_min:g} risks {min_pct:.2f}% (more than {risk_pct}%)")
    return Size(lots, lots * per_lot)
