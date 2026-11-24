"""MetaTrader 5 connectivity (read-only: market data and symbol info)."""
from .client import MT5Client, MT5Error, MT5NotAvailableError
from .safety import ForbiddenOperationError, SafeMT5
from .server_time import ServerClock
from .symbols import BrokerSymbol, SymbolResolutionError, discover_symbols, resolve_symbol

__all__ = ["MT5Client", "MT5Error", "MT5NotAvailableError", "ForbiddenOperationError", "SafeMT5",
           "ServerClock", "BrokerSymbol", "SymbolResolutionError", "discover_symbols", "resolve_symbol"]
