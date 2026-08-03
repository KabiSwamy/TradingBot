"""End-to-end: the CLI, exercised the way a user runs it.

Everything writes to tmp_path via --out-dir, so these tests can never touch the
real research ledger (the conftest guard would fail them if they did).
"""

from __future__ import annotations

import csv
import json

import pytest

from backtest.experiments import FIELDS
from backtest.run import main

# A short window keeps the test fast; 2010-2013 still spans the SMA(200) warm-up
# plus two years of live trading.
WINDOW = ["--start", "2010-01-01", "--end", "2013-12-31"]


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory):
    """One synthetic bar cache for the whole module — generating bars for ten
    symbols per test dominated the runtime otherwise."""
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory, shared_cache):
    out = tmp_path_factory.mktemp("results")
    code = main([*WINDOW, "--source", "synthetic", "--out-dir", str(out),
                 "--cache-dir", str(shared_cache)])
    return code, out


def test_the_cli_exits_cleanly(completed_run):
    code, _ = completed_run
    assert code == 0


def test_it_prints_every_rule_12_field(completed_run, shared_cache, capsys):
    """A report missing the benchmark or the trade stats is not a report."""
    _, out = completed_run
    # Re-run to capture stdout in this test's context.
    main([*WINDOW, "--source", "synthetic", "--out-dir", str(out),
          "--cache-dir", str(shared_cache), "--no-log", "--no-plot"])
    text = capsys.readouterr().out
    for required in ("Sharpe", "max drawdown", "win rate", "avg win", "avg loss",
                     "BUY & HOLD", "count"):
        assert required in text


def test_it_writes_the_equity_curve(completed_run):
    _, out = completed_run
    plots = list(out.glob("equity_*.png"))
    assert len(plots) == 1
    assert plots[0].stat().st_size > 5000


def test_it_logs_the_run_to_experiments_csv(completed_run):
    """Rule 4: every run is logged, so nothing can be quietly cherry-picked."""
    _, out = completed_run
    ledger = out / "experiments.csv"
    assert ledger.exists()

    with ledger.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]

    assert list(row) == list(FIELDS)
    assert row["data_source"] == "synthetic"
    assert row["window_start"] == "2010-01-01"
    assert int(row["trade_count"]) > 0
    assert row["config_hash"]
    # The full config travels with the row so a result is always reproducible
    # without reconstructing which settings produced it.
    assert json.loads(row["config_json"])["cost_per_side"] == 0.0005


def test_a_second_run_appends_rather_than_overwrites(completed_run, shared_cache):
    _, out = completed_run
    main([*WINDOW, "--source", "synthetic", "--out-dir", str(out),
          "--cache-dir", str(shared_cache), "--no-plot"])
    with (out / "experiments.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_it_writes_a_decision_log_of_json_lines(completed_run):
    """Rule 10: date, indicator values, signals, orders, fills, reasons."""
    _, out = completed_run
    logs = list(out.glob("decisions_*.jsonl"))
    assert logs

    records = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert len(records) > 500

    for record in records[:5]:
        assert {"date", "equity", "cash", "drawdown", "positions",
                "fills", "orders", "halted"} <= set(record)

    # Every order carries the reason it was issued.
    reasons = {o["reason"] for r in records for o in r["orders"]}
    assert reasons and reasons <= {"entry_signal", "rsi_exit", "time_stop", "kill_switch"}


def test_no_fill_ever_precedes_its_own_decision(completed_run):
    """The rule-1 invariant, checked across a full multi-year run."""
    _, out = completed_run
    logs = list(out.glob("decisions_*.jsonl"))
    records = [json.loads(line) for line in logs[0].read_text().splitlines()]

    checked = 0
    for record in records:
        for fill in record["fills"]:
            if "decided_date" in fill:
                assert fill["decided_date"] < record["date"]
                checked += 1
    assert checked > 20


def test_no_log_suppresses_the_ledger_row(tmp_path, shared_cache):
    main([*WINDOW, "--source", "synthetic", "--out-dir", str(tmp_path),
          "--cache-dir", str(shared_cache), "--no-log", "--no-plot"])
    assert not (tmp_path / "experiments.csv").exists()


def test_synthetic_runs_are_labelled_on_the_console(tmp_path, shared_cache, capsys):
    main([*WINDOW, "--source", "synthetic", "--out-dir", str(tmp_path),
          "--cache-dir", str(shared_cache), "--no-log", "--no-plot"])
    text = capsys.readouterr().out
    assert "SYNTHETIC DATA" in text
    assert "NOT A RESEARCH RESULT" in text


def test_cli_overrides_reach_the_run(tmp_path, shared_cache):
    main([*WINDOW, "--source", "synthetic", "--out-dir", str(tmp_path),
          "--cache-dir", str(shared_cache), "--no-plot",
          "--entry-rsi", "5", "--time-stop", "3"])
    with (tmp_path / "experiments.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["entry_rsi"]) == 5.0
    assert int(row["time_stop_days"]) == 3


def test_a_window_touching_out_of_sample_data_says_so(tmp_path, shared_cache, capsys):
    """Rule 4 discipline is announced, not assumed."""
    main(["--start", "2019-01-01", "--end", "2020-06-01", "--source", "synthetic",
          "--out-dir", str(tmp_path), "--cache-dir", str(shared_cache),
          "--no-log", "--no-plot"])
    assert "out-of-sample" in capsys.readouterr().out
