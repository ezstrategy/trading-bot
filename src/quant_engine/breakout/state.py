"""Breakout state machine (one per strategy).

States::

    WAITING ──D1 close > level──▶ CONFIRMED ──checks pass──▶ SIGNAL ──entry filled──▶ ACTIVE
       ▲                              │ checks fail                                    │ exit
       │ reset condition               ▼                                                ▼
    WAITING_FRESH ◀──────────────────────────────────────────────────────────────── CLOSED

* A breakout is a completed D1 close strictly above the level while the machine is **armed**.
  Every breakout disarms the machine, whether it becomes a signal, fails a check, or is ignored
  because the strategy already has a position. So price staying above the level never produces
  a new signal every day.
* The machine re-arms only on the strategy's reset condition (``reset_mode``):
  ``new_month`` (first D1 candle of a new calendar month), ``close_below_level`` (a D1 close at or
  below that day's level) or ``new_month_or_close_below``. At the start of the data it is
  **not** armed: a reset must be seen first, so a market already above the level at the start is
  not reported as a fresh breakout.
* While a signal is pending or a position is active, breakouts are ignored (one position per
  strategy) but the reset condition is still tracked.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WAITING, CONFIRMED, SIGNAL, ACTIVE, CLOSED, WAITING_FRESH = (
    "waiting_for_breakout", "breakout_confirmed", "signal_generated", "position_active", "position_closed",
    "waiting_for_fresh_breakout")

LABELS = {WAITING: "Waiting for breakout", CONFIRMED: "Breakout confirmed", SIGNAL: "Signal generated",
          ACTIVE: "Position active", CLOSED: "Position closed", WAITING_FRESH: "Waiting for a fresh breakout"}


@dataclass
class BreakoutStateMachine:
    """Explicit state of one strategy; ``history`` keeps every transition (UTC time, from, to, why)."""

    reset_mode: str
    state: str = WAITING_FRESH
    armed: bool = False
    _month: pd.Period | None = None
    history: list[tuple[pd.Timestamp, str, str, str]] = field(default_factory=list)

    def _go(self, when: pd.Timestamp, to: str, why: str) -> None:
        if to != self.state:
            self.history.append((when, self.state, to, why))
            self.state = to

    def on_d1_close(self, when: pd.Timestamp, day: pd.Timestamp, close: float, level: float) -> bool:
        """Feed one completed D1 candle. Returns True when it is a breakout (state CONFIRMED)."""
        month = pd.Timestamp(day).to_period("M")
        in_position = self.state in (SIGNAL, ACTIVE)
        if self.reset_mode in ("new_month", "new_month_or_close_below") and month != self._month \
                and self._month is not None:
            self._arm(when, "new month", in_position)
        self._month = month
        if not np.isfinite(level):
            return False
        if close > level:
            was_armed, self.armed = self.armed, False
            if was_armed and not in_position:
                self._go(when, CONFIRMED, f"D1 close {close:.2f} > level {level:.2f}")
                return True
            if was_armed:
                self.history.append((when, self.state, self.state, "breakout ignored: position active"))
            if self.state == WAITING:
                self._go(when, WAITING_FRESH, "price above the level without a fresh breakout")
            return False
        if self.reset_mode in ("close_below_level", "new_month_or_close_below"):
            self._arm(when, f"D1 close {close:.2f} <= level {level:.2f}", in_position)
        return False

    def _arm(self, when: pd.Timestamp, why: str, in_position: bool) -> None:
        self.armed = True
        if not in_position and self.state in (WAITING_FRESH, CLOSED):
            self._go(when, WAITING, "reset: " + why)

    def signal_generated(self, when: pd.Timestamp) -> None:
        self._go(when, SIGNAL, "all checks passed")

    def checks_failed(self, when: pd.Timestamp, why: str) -> None:
        self._go(when, WAITING_FRESH, "signal not generated: " + why)

    def entry_filled(self, when: pd.Timestamp) -> None:
        self._go(when, ACTIVE, "entry at the next M5 open")

    def position_closed(self, when: pd.Timestamp, why: str) -> None:
        self._go(when, CLOSED, why)
        self._go(when, WAITING if self.armed else WAITING_FRESH,
                 "already re-armed" if self.armed else "waiting for the reset condition")

    @property
    def label(self) -> str:
        return LABELS[self.state]
