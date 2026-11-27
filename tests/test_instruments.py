"""Quant Lab module 2: broker-independent instrument profiles."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_engine import cli
from quant_engine.config import ConfigError
from quant_engine.instruments import InstrumentSpec, load_instruments

from .conftest import REPO


def spec(**kw) -> InstrumentSpec:
    base = dict(symbol="XAUUSD", contract_size=100, min_lot=0.01, max_lot=50, lot_step=0.01, tick_size=0.01,
                tick_value=1.0, digits=2, commission_per_lot=7.0)
    return InstrumentSpec(**{**base, **kw})


def test_repo_profile_loads_and_lists_guesses():
    book = load_instruments(REPO / "config")
    s = book.get()
    assert s.name == "xauusd_default" and s.symbol == "XAUUSD"
    assert s.commission_per_lot == 0 and s.typical_spread == 0.08
    assert "commission_per_lot" not in s.guesses() and "contract_size" in s.guesses()
    assert s.consistency_warnings() == []
    with pytest.raises(ConfigError):
        book.get("no_such_broker")


def test_money_lots_and_commission():
    s = spec()
    assert s.money_per_price_unit(1.0) == pytest.approx(100.0)          # $1 move on 1 lot = $100
    assert s.pnl(+1, 2000.00, 2002.50, 0.20) == pytest.approx(50.0)
    assert s.pnl(-1, 2000.00, 2002.50, 0.20) == pytest.approx(-50.0)
    assert s.commission(0.20) == pytest.approx(1.4)
    # 0.5% of 10,000 = $50 at a $2.50 stop -> 0.20 lots
    assert s.lots_for_risk(10_000, 0.5, 2.50) == pytest.approx(0.20)
    # 0.5% of 10,000 at a $3.00 stop = 0.1666 -> rounded DOWN to 0.16
    assert s.lots_for_risk(10_000, 0.5, 3.00) == pytest.approx(0.16)
    # minimum lot would risk too much -> 0 (no trade size), never rounded up
    assert s.lots_for_risk(100, 0.5, 5.0) == 0.0
    assert s.round_lot(999) == 50 and s.round_lot(-1) == 0.0 and s.round_lot(float("nan")) == 0.0
    assert spec(lot_step=0.1, min_lot=0.1).round_lot(0.19) == pytest.approx(0.1)


def test_brokers_can_differ():
    a, b = spec(), spec(contract_size=10, tick_value=0.1, broker="mini")
    assert a.money_per_price_unit(1) == 10 * b.money_per_price_unit(1)
    assert a.lots_for_risk(10_000, 0.5, 2.5) * 10 == pytest.approx(b.lots_for_risk(10_000, 0.5, 2.5))


def test_consistency_warnings():
    assert "differs from contract_size" in spec(tick_value=2.0).consistency_warnings()[0]
    assert "exchange rate" in spec(account_currency="EUR").consistency_warnings()[0]
    with pytest.raises(ValueError):
        spec(min_lot=10, max_lot=1)


def test_from_mt5_symbol_info_keeps_commission_from_base():
    info = SimpleNamespace(name="XAUUSD.r", trade_contract_size=100.0, volume_min=0.01, volume_max=200.0,
                           volume_step=0.01, trade_tick_size=0.01, trade_tick_value=1.0, digits=2,
                           currency_profit="USD")
    s = InstrumentSpec.from_symbol_info(info, "USD", name="brokerx", broker="Broker X", base=spec())
    assert s.broker_symbol == "XAUUSD.r" and s.max_lot == 200 and s.commission_per_lot == 7.0
    assert "contract_size" in s.confirmed and "commission_per_lot" in s.guesses()


def test_cli_instrument(config_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.run(["--config", str(config_dir), "instrument"]) == 0
    out = capsys.readouterr().out
    assert '"profile": "xauusd_default"' in out and '"money_per_1.0_move_per_lot": 100.0' in out
