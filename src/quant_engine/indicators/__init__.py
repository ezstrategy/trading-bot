"""Decision-time-safe indicators for the Quant Lab (module 3).

Every function returns one value per input candle, computed from that candle and earlier ones
only, so the value on candle *i* is known at the candle's close (``open + timeframe``).
"""
from .core import atr, ema, ema_slope, opening_range, session_of, trend_state

__all__ = ["atr", "ema", "ema_slope", "opening_range", "session_of", "trend_state"]
