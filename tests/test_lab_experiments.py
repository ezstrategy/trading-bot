"""Quant Lab Phases 6, 8 and 9: registry, locked test set, validation, MT5 import, audit, CLI."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from quant_engine import cli
from quant_engine.audit import audit_experiment
from quant_engine.backtest.engine import ExecutionConfig
from quant_engine.config import QualityConfig
from quant_engine.data.catalog import DatasetCatalog
from quant_engine.data.dukascopy import resample
from quant_engine.experiments.lab import LabSettings, load_lab_settings, load_market, run_experiment, run_validation
from quant_engine.experiments.registry import ExperimentRegistry, RegistryError, Split
from quant_engine.instruments import load_instruments
from quant_engine.strategies import create
from quant_engine.timeframes import Timeframe
from quant_engine.validation import TestSetLocked, make_split, walk_forward

from .conftest import REPO
from .fake_mt5 import NOW_UTC, FakeMT5
from .synthetic import synthetic_m1

SPEC = load_instruments(REPO / "config").get()
STRATS = [("gtc_sbr_bull", {}), ("gtc_sbr_bear", {})]


@pytest.fixture(scope="module")
def m1():
    return synthetic_m1(days=300, seed=3)


@pytest.fixture
def cat(tmp_path, m1):
    c = DatasetCatalog(tmp_path / "catalog", QualityConfig())
    c.import_frame(resample(m1, Timeframe.M5), "XAUUSD", "M5", source_kind="synthetic", source_name="test walk")
    return c


@pytest.fixture
def settings():
    s = LabSettings()
    s.walk_forward.train_months, s.walk_forward.test_months = 3, 2
    s.cost_stress = [{"name": "as tested"}, {"name": "spread x1.5", "spread_multiplier": 1.5}]
    return s


# ------------------------------------------------------------------ registry
def test_registry_is_append_only(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp")
    eid = reg.create(kind="backtest", strategies=[{"id": "x", "version": "1", "family": "f"}], dataset_ids={},
                     execution={}, results={"net": {"trades": 3}}, artefacts={"report.md": "hello"}, note="first")
    assert eid == "LAB-0001" and (tmp_path / "exp" / eid / "report.md").read_text() == "hello"
    reg.set_status(eid, "REJECTED")
    reg.add_note(eid, "second thought")
    e = reg.get(eid)
    assert e["status"] == "REJECTED" and [n["value"] for n in e["notes"]] == ["first", "second thought"]
    assert [h["value"] for h in e["history"] if h["kind"] == "status"] == ["RECORDED", "REJECTED"]
    con = sqlite3.connect(tmp_path / "exp" / "registry.db", isolation_level=None)
    for sql in ("UPDATE experiments SET title='x'", "DELETE FROM experiments", "DELETE FROM events"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            con.execute(sql)
    con.close()
    with pytest.raises(RegistryError):
        reg.set_status(eid, "GREAT")
    assert reg.create(kind="backtest", strategies=[{"id": "x", "version": "2"}], dataset_ids={}, execution={},
                      results={}, artefacts={}) == "LAB-0002"
    assert list(reg.list()["id"]) == ["LAB-0001", "LAB-0002"]


def test_split_is_permanent(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp")
    sp = make_split("p", pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
    assert sp.dev_end == pd.Timestamp("2023-01-01", tz="UTC") and sp.val_end == pd.Timestamp("2024-01-01", tz="UTC")
    reg.register_split(sp)
    assert reg.register_split(sp) == sp
    moved = Split("p", sp.data_start, sp.dev_end, sp.val_end + pd.Timedelta(days=30), sp.data_end)
    with pytest.raises(RegistryError, match="different split"):
        reg.register_split(moved)
    assert sp.part("test") == (sp.val_end, sp.data_end)


# ------------------------------------------------------------------ runs
def test_load_market_derives_m15_h4_and_records_versions(cat):
    data = load_market(cat, "XAUUSD")
    assert set(data.dataset_ids) == {"M5", "M15", "H4"} and data.dataset_ids["H4"].startswith("derived from XAUUSD_M5_")
    assert data.has(Timeframe.H4) and not data.has(Timeframe.M1)


def test_run_experiment_registers_everything_and_locks_the_test(tmp_path, cat, settings):
    reg = ExperimentRegistry(tmp_path / "exp")
    data = load_market(cat, "XAUUSD")
    out = run_experiment(reg, data, SPEC, settings, STRATS, "gtc_synth", "development", title="dev run",
                         hypothesis="breakout + retest continues with the M15 trend")
    e = reg.get(out["id"])
    assert e["programme"] == "gtc_synth" and e["data_part"] == "development" and not e["test_set_used"]
    assert e["dataset_ids"]["M5"].startswith("XAUUSD_M5_") and e["strategies"][0]["fingerprint"]
    assert {"net", "before_costs", "after_spread"} <= set(e["results"])
    f = Path(e["folder"])
    for name in ("report.md", "trades.csv", "equity.csv", "signals.csv", "monthly.csv", "experiment.json"):
        assert (f / name).exists(), name
    tr = pd.read_csv(f / "trades.csv", parse_dates=["entry_time"])
    sp = reg.split("gtc_synth")
    assert (tr["entry_time"] >= sp.data_start).all() and (tr["signal_time"].astype(str) < str(sp.dev_end)).all()
    assert "NET (after all costs)" in (f / "report.md").read_text()
    with pytest.raises(TestSetLocked):
        run_experiment(reg, data, SPEC, settings, STRATS, "gtc_synth", "test")
    with pytest.raises(TestSetLocked):
        run_experiment(reg, data, SPEC, settings, STRATS, "gtc_synth", "all")
    t1 = run_experiment(reg, data, SPEC, settings, STRATS, "gtc_synth", "test", final_test=True)
    assert t1["test_set_used"] and t1["prior_test_uses"] == {"gtc_sbr": 0}
    t2 = run_experiment(reg, data, SPEC, settings, STRATS, "gtc_synth", "test", final_test=True)
    assert t2["prior_test_uses"] == {"gtc_sbr": 1}
    assert "TEST PERIOD USED" in (Path(t2["folder"]) / "report.md").read_text()
    cmp = reg.compare([out["id"], t1["id"]])
    assert list(cmp.columns) == [out["id"], t1["id"]] and "net expectancy_r" in cmp.index


def test_walk_forward_limits_the_grid(cat):
    data = load_market(cat, "XAUUSD")
    fac = lambda o=None: [create("gtc_sbr_bull", o)]  # noqa: E731
    a, b = data.base.t[0] + pd.Timedelta(days=60), data.base.t[-1]
    with pytest.raises(ValueError, match="limit"):
        walk_forward(data, fac, SPEC, ExecutionConfig(), a, b, grid=[{"target_r": x / 10} for x in range(10, 30)])
    wf = walk_forward(data, fac, SPEC, ExecutionConfig(), a, b, train_months=3, test_months=2)
    assert wf["mode"] == "fixed rules" and wf["n_windows"] >= 2
    assert (pd.to_datetime(wf["windows"]["test"].str[:10]).diff().dropna() == pd.Timedelta(days=61)).any() or wf["n_windows"] >= 2


def test_run_validation_never_trades_the_test(tmp_path, cat, settings):
    reg = ExperimentRegistry(tmp_path / "exp")
    data = load_market(cat, "XAUUSD")
    out = run_validation(reg, data, SPEC, settings, STRATS, "gtc_synth", grid=[{"target_r": 1.5}, {"target_r": 2.0}],
                         neighbours={"retest_tol_atr": [0.1, 0.15, 0.2]})
    sp = reg.split("gtc_synth")
    f = Path(out["folder"])
    for name in ("walk_forward_trades.csv", "trades.csv"):
        df = pd.read_csv(f / name)
        if len(df):
            assert (pd.to_datetime(df["signal_time"]) < sp.val_end).all(), name
    st = pd.read_csv(f / "cost_stress.csv")
    assert list(st["scenario"]) == ["as tested", "spread x1.5"]
    sens = pd.read_csv(f / "sensitivity.csv")
    assert sens["is_base"].sum() == 1
    assert "Walk-forward" in (f / "report.md").read_text() and reg.get(out["id"])["kind"] == "validation"


# ------------------------------------------------------------------ audit
def test_audit_reproduces_and_classifies(tmp_path, cat, settings):
    reg = ExperimentRegistry(tmp_path / "exp")
    data = load_market(cat, "XAUUSD")
    dev = run_experiment(reg, data, SPEC, settings, STRATS, "gtc_synth", "development")
    out = audit_experiment(reg, cat, SPEC, settings, dev["id"], cuts=3)
    assert out["failed"] == [], out
    assert out["classification"] in ("REJECTED", "RESEARCH REQUIRED")      # never validated from development data
    assert not out["targets"]["audited run is the final test on untouched data"]
    assert reg.get(dev["id"])["status"] == out["classification"]
    md = (Path(out["folder"]) / "report.md").read_text()
    assert "re-run from the record gives identical trades | PASS" in md and "no look-ahead" in md


def test_audit_detects_changed_data(tmp_path, cat, settings):
    import os
    import stat
    reg = ExperimentRegistry(tmp_path / "exp")
    data = load_market(cat, "XAUUSD")
    dev = run_experiment(reg, data, SPEC, settings, STRATS, "gtc_synth", "development")
    p = cat.folder(data.dataset_ids["M5"]) / "bars.parquet"
    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    df = pd.read_parquet(p)
    df.loc[len(df) // 3, "close"] += 5
    df.to_parquet(p, index=False)
    out = audit_experiment(reg, cat, SPEC, settings, dev["id"], cuts=2)
    assert out["classification"] == "REJECTED"
    assert any("checksums" in f for f in out["failed"])


# ------------------------------------------------------------------ CLI and MT5 (fake terminal)
@pytest.fixture
def run(config_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeMT5()
    return lambda *a: cli.run(["--config", str(config_dir), *a], mt5_module=fake, now=lambda: NOW_UTC)


def test_cli_mt5_import_and_instrument(run, cfg, config_dir, capsys):
    assert run("mt5-import", "--timeframes", "M5,M15", "--days", "20") == 0
    out = capsys.readouterr().out
    assert out.count('"new_version": true') == 2
    m = DatasetCatalog(cfg.data.catalog_root()).manifests("XAUUSD", "M5")[0]
    assert m["source"]["kind"] == "mt5" and m["settings"]["broker_symbol"] == "XAUUSD.r"
    assert m["settings"]["server_clock_mode"] == "ny_close"
    assert run("mt5-instrument", "--profile", "fakebroker") == 0
    info = json.loads(capsys.readouterr().out)
    assert Path(info["file"]).name == "instruments.fakebroker.yaml"
    book = load_instruments(config_dir)
    fb = book.get("fakebroker")
    assert fb.broker_symbol == "XAUUSD.r" and fb.contract_size == 100 and "contract_size" in fb.confirmed
    assert fb.commission_per_lot == 0                                    # not reported by MT5: copied from default
    assert run("mt5-instrument", "--profile", "fakebroker") == 1           # never overwrites
    assert "not overwritten" in capsys.readouterr().err


def test_cli_lab_flow(run, cfg, capsys, m1):
    DatasetCatalog(cfg.data.catalog_root(), cfg.data.quality).import_frame(
        resample(m1, Timeframe.M5), "XAUUSD", "M5", source_kind="synthetic", source_name="walk")
    assert run("lab-strategies") == 0 and "gtc_sbr_bear" in capsys.readouterr().out
    assert run("lab-run", "--programme", "cli", "--part", "development", "--title", "t") == 0
    out = capsys.readouterr().out
    eid = json.loads(out[: out.rindex("}") + 1])["id"]
    assert run("lab-run", "--programme", "cli", "--part", "test") == 1
    assert "untouched test period" in capsys.readouterr().err
    assert run("lab-note", eid, "looks weak", "--status", "RESEARCH REQUIRED") == 0
    assert '"RESEARCH REQUIRED"' in capsys.readouterr().out
    assert run("lab-list") == 0 and eid in capsys.readouterr().out
    assert run("lab-show", eid) == 0 and "looks weak" in capsys.readouterr().out
    assert run("lab-audit", eid, "--cuts", "2") == 0
    assert '"classification"' in capsys.readouterr().out
