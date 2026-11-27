"""API routers, one module per dashboard page. Add a page: add a module and list it in ROUTERS."""
from . import backtest, breakout, data, market, news, research, signals

ROUTERS = [market.router, data.router, signals.router, backtest.router, news.router, research.router, breakout.router]
