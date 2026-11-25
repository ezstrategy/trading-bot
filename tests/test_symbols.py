import pytest

from quant_engine.config import SymbolMapping
from quant_engine.mt5.symbols import SymbolResolutionError, discover_symbols, prepare_symbol, resolve_symbol

M = SymbolMapping(candidates=["XAUUSD", "XAUUSDm", "GOLD"], pattern=r"^(XAUUSD|GOLD)([._+\-]?[A-Za-z0-9]{0,4})?$")


@pytest.mark.parametrize("available,expected", [
    (["EURUSD", "XAUUSD"], "XAUUSD"),
    (["EURUSD", "xauusdm"], "xauusdm"),                # case-insensitive candidate
    (["EURUSD", "GOLD", "XAUUSDm"], "XAUUSDm"),        # candidate order wins
    (["EURUSD", "XAUUSD.pro"], "XAUUSD.pro"),          # regex fallback
    (["XAUUSD_i", "XAGUSD_i"], "XAUUSD_i"),
])
def test_resolution(available, expected):
    assert resolve_symbol("XAUUSD", M, available) == expected


def test_explicit_override():
    m = M.model_copy(update={"broker_symbol": "GOLD.ecn"})
    assert resolve_symbol("XAUUSD", m, ["XAUUSD", "GOLD.ecn"]) == "GOLD.ecn"
    with pytest.raises(SymbolResolutionError, match="not offered"):
        resolve_symbol("XAUUSD", m, ["XAUUSD"])


def test_ambiguous_regex():
    with pytest.raises(SymbolResolutionError, match="several"):
        resolve_symbol("XAUUSD", M, ["XAUUSD.a", "XAUUSD.b"])


def test_not_found():
    with pytest.raises(SymbolResolutionError, match="no matching"):
        resolve_symbol("XAUUSD", M, ["EURUSD", "XAGUSD", "XAUEUR"])


def test_discover_and_prepare(client, fake, cfg):
    syms = discover_symbols(client)
    assert {s.name for s in syms} == set(fake.symbol_names)
    assert [s.name for s in discover_symbols(client, "*XAU*")] == ["XAUUSD.r"]
    info = prepare_symbol(client, "XAUUSD", cfg.mapping())
    assert info.name == "XAUUSD.r" and info.digits == 2
    assert fake.selected == ["XAUUSD.r"]  # added to Market Watch
