"""Quant Lab Phase 7: dashboard services and a headless render of all 12 Quant Lab pages."""
from __future__ import annotations

import pandas as pd
import pytest

from quant_engine.dashboard import lab_services as ls
from quant_engine.data.catalog import DatasetCatalog
from quant_engine.data.dukascopy import resample
from quant_engine.experiments.lab import LabSettings, load_market, run_experiment, run_validation
from quant_engine.instruments import load_instruments
from quant_engine.timeframes import Timeframe

from .conftest import REPO
from .synthetic import synthetic_m1
from .test_dashboard import _app

SPEC = load_instruments(REPO / "config").get()


@pytest.fixture
def lab(cfg):
    cat = DatasetCatalog(cfg.data.catalog_root(), cfg.data.quality)
    cat.import_frame(resample(synthetic_m1(days=240, seed=3), Timeframe.M5), "XAUUSD", "M5",
                     source_kind="synthetic", source_name="walk")
    reg = ls.registry(cfg)
    s = LabSettings()
    s.walk_forward.train_months, s.walk_forward.test_months = 3, 2
    data = load_market(cat, "XAUUSD")
    a = run_experiment(reg, data, SPEC, s, [("gtc_sbr_bull", {}), ("gtc_sbr_bear", {})], "dash", "development")
    v = run_validation(reg, data, SPEC, s, [("gtc_sbr_bull", {}), ("gtc_sbr_bear", {})], "dash")
    return cat, reg, a["id"], v["id"]


def test_services(cfg, lab):
    cat, reg, eid, vid = lab
    o = ls.overview(cfg)
    assert o["dataset_versions"] == 1 and o["experiments"] == 2 and len(o["programmes"]) == 1
    st = ls.strategies_table()
    assert {"gtc_sbr_bull", "gtc_sbr_bear"} <= set(st["id"]) and st["entry"].str.len().min() > 10
    tr = ls.load_trades(reg, eid)
    eq = ls.load_equity(reg, eid)
    assert len(eq) == len(tr) + 1 and {"drawdown", "drawdown_pct"} <= set(eq.columns)
    assert ls.equity_figure({eid: eq}).data and ls.equity_figure({eid: eq}, "drawdown_pct").data
    if len(tr):
        fig = ls.replay_figure(cat, reg.get(eid)["dataset_ids"]["M5"], tr.iloc[0])
        names = [d.name for d in fig.data]
        assert names[0] == "M5 (bid)" and any(n.startswith("entry") for n in names) and any(n.startswith("stop") for n in names)
    cc = ls.compare_curves(reg, [eid, vid])
    assert all(c.empty or c["equity"].iat[0] == pytest.approx(100) for c in cc.values())
    assert list(ls.walk_forward_experiments(reg)["id"]) == [vid]
    assert ls.csv_bytes(tr).startswith(b"trade_id")
    assert ls.experiment_summary(reg, eid)["status"] == "RECORDED"


def test_all_lab_pages_render(config_dir, cfg, lab, monkeypatch):
    at = _app(config_dir, monkeypatch).run()
    assert not at.exception, [e.value for e in at.exception]
    radio = next(r for r in at.radio if r.label == "Quant Lab page")
    for page in ls.PAGES:
        radio.set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
        errs = [e.value for e in at.error if "Quant Lab problem" in e.value]
        assert not errs, (page, errs)
        radio = next(r for r in at.radio if r.label == "Quant Lab page")
