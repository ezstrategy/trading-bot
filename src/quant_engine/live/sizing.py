"""Position size shown in a signal (the engine never trades; this is advice text)."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Size:
    lots: float
    risk_money: float          # account currency at the full stop
    note: str = ""


def lot_size(balance: float, risk_pct: float, entry: float, stop: float, contract_size: float,
             volume_min: float, volume_step: float, volume_max: float | None = None) -> Size:
    """Lots that lose about ``risk_pct`` of ``balance`` at the stop, rounded DOWN to the
    broker's volume step.

    GUESS: the account currency equals the symbol's profit currency (USD for XAUUSD);
    for other account currencies the amount is off by the exchange rate.
    """
    risk_money = balance * risk_pct / 100.0
    per_lot = abs(entry - stop) * contract_size
    if per_lot <= 0 or volume_step <= 0:
        return Size(0.0, 0.0, "invalid stop or volume step")
    raw = risk_money / per_lot
    lots = math.floor(raw / volume_step + 1e-9) * volume_step
    lots = round(lots, 8)
    if volume_max:
        lots = min(lots, volume_max)
    if lots < volume_min:
        return Size(volume_min, volume_min * per_lot,
                    f"minimum lot {volume_min:g} risks {100 * volume_min * per_lot / balance:.2f}% (more than {risk_pct}%)")
    return Size(lots, lots * per_lot)
