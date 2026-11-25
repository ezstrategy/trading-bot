"""Market-structure features for the XAUUSD strategy (Phase 2).

Every function here is *causal*: a value attached to a candle uses only
that candle and earlier ones, and anything that needs later candles to be
known (a fractal swing, a sweep reclaim) carries the time at which it
became known. ``tests/test_no_lookahead.py`` checks this by recomputing on
truncated data.
"""
