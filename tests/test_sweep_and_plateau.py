"""Phase 2 tests: the sweep, plateau analysis (rule 3), and OOS discipline (rule 4).

Rules 3 and 4 are the two the whole phase exists to serve, and both are the kind
of rule that erodes silently — the best cell is always at the top of a sorted
table, and the test window is always sitting right there on disk. So both are
enforced in code and both are tested here.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from backtest.experiments import prior_oos_runs
from backtest.plateau import analyse, render_plateau_report
from backtest.sweep import AXES, DEFAULT_GRID, OutOfSampleError, sweep
from config.loader import load_settings
from data.sources import SyntheticSource

SETTINGS = load_settings(sma_period=20)


@pytest.fixture(scope="module")
def bars():
    source = SyntheticSource(seed=3)
    return {s: source.fetch(s, "2012-01-01", "2014-12-31")
            for s in ["AAA", "BBB", "CCC"]}


SMALL_GRID = {
    "entry_rsi": (5.0, 10.0, 15.0),
    "exit_rsi": (60.0, 70.0, 80.0),
    "time_stop_days": (5, 7),
}


@pytest.fixture(scope="module")
def results(bars):
    return sweep(bars, SETTINGS, start="2012-01-01", end="2014-12-31",
                 grid=SMALL_GRID)


# --- the grid ---------------------------------------------------------------


def test_default_grid_is_the_one_named_in_the_phase_2_gate():
    assert DEFAULT_GRID["entry_rsi"] == (5.0, 8.0, 10.0, 12.0, 15.0)
    assert DEFAULT_GRID["exit_rsi"] == (60.0, 65.0, 70.0, 75.0, 80.0)
    assert DEFAULT_GRID["time_stop_days"] == (5, 7, 10)


def test_sweep_runs_every_combination(results):
    assert len(results) == 3 * 3 * 2
    assert not results.duplicated(subset=list(AXES)).any()


def test_each_cell_reports_the_metrics_rule_12_cares_about(results):
    for column in ("sharpe", "max_drawdown", "trade_count", "win_rate", "cagr"):
        assert column in results.columns
    assert results["sharpe"].notna().all()


def test_parameters_actually_change_the_outcome(results):
    """If every cell scored the same, a plateau analysis would be theatre."""
    assert results["sharpe"].std(ddof=1) > 0.01
    assert results["trade_count"].nunique() > 1


def test_a_nonsense_grid_cell_is_skipped_not_fatal(bars):
    """entry above exit is invalid; it must not abort the other cells."""
    bad = {"entry_rsi": (5.0, 90.0), "exit_rsi": (70.0,), "time_stop_days": (7,)}
    out = sweep(bars, SETTINGS, start="2012-01-01", end="2014-12-31", grid=bad)
    assert len(out) == 2
    assert out["error"].astype(bool).sum() == 1
    assert out["sharpe"].notna().sum() == 1


# --- rule 4: the sweep must not touch out-of-sample data ---------------------


def test_sweep_refuses_a_window_past_the_train_end(bars):
    with pytest.raises(OutOfSampleError, match="rule 4"):
        sweep(bars, SETTINGS, start="2012-01-01", end="2020-06-01", grid=SMALL_GRID)


def test_sweep_allows_the_train_window(bars):
    out = sweep(bars, SETTINGS, start="2010-01-01",
                end=SETTINGS.train_end, grid=SMALL_GRID)
    assert len(out) == 18


def test_every_swept_run_is_logged_to_the_ledger(bars, tmp_path):
    """Rule 4: log EVERY backtest run so we cannot quietly cherry-pick."""
    ledger = tmp_path / "experiments.csv"
    sweep(bars, SETTINGS, start="2012-01-01", end="2014-12-31",
          grid=SMALL_GRID, ledger_path=ledger)

    with ledger.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    assert {r["window_label"] for r in rows} == {"train"}
    assert len({r["config_hash"] for r in rows}) == 18


# --- rule 3: plateau, not peak ----------------------------------------------


def grid_from(values: dict[tuple, float]) -> pd.DataFrame:
    rows = []
    for (e, x, t), sharpe in values.items():
        rows.append({"entry_rsi": e, "exit_rsi": x, "time_stop_days": t,
                     "sharpe": sharpe, "trade_count": 100})
    return pd.DataFrame(rows)


def synthetic_surface(spike: bool) -> pd.DataFrame:
    """A flat field with either an isolated spike or a broad plateau."""
    values = {}
    for e in (5.0, 10.0, 15.0):
        for x in (60.0, 70.0, 80.0):
            for t in (5, 7):
                values[(e, x, t)] = 0.1
    if spike:
        values[(10.0, 70.0, 7)] = 3.0             # one cell, cliffs all round
    else:
        for e in (10.0, 15.0):                    # a contiguous good region
            for x in (70.0, 80.0):
                for t in (5, 7):
                    values[(e, x, t)] = 1.0
    return grid_from(values)


def test_an_isolated_spike_is_not_recommended():
    """Rule 3: a spike surrounded by cliffs is curve-fit and is rejected."""
    report = analyse(synthetic_surface(spike=True))

    assert report.best["sharpe"] == 3.0
    assert report.best_is_spike
    assert not report.agree
    # The recommendation must not be the spike itself.
    assert not (report.recommended["entry_rsi"] == 10.0
                and report.recommended["exit_rsi"] == 70.0
                and report.recommended["time_stop_days"] == 7)


def test_a_broad_plateau_is_recommended():
    report = analyse(synthetic_surface(spike=False))
    assert report.recommended["sharpe"] == 1.0
    assert report.recommended["neighbour_min"] == 1.0
    assert not report.best_is_spike


def test_robust_score_is_own_metric_or_worst_neighbour_whichever_is_lower():
    report = analyse(synthetic_surface(spike=True))
    row = report.table.set_index(list(AXES)).loc[(10.0, 70.0, 7)]
    assert row["robust_score"] == pytest.approx(min(row["sharpe"], row["neighbour_min"]))
    assert row["robust_score"] == pytest.approx(0.1)   # dragged down by its cliffs


def test_neighbours_are_adjacent_along_one_axis_only():
    """Not the diagonal neighbourhood: a diagonal cell changes every parameter."""
    report = analyse(synthetic_surface(spike=False))
    table = report.table.set_index(list(AXES))
    # Interior on entry and exit, but time stop has only 2 levels -> 5 neighbours.
    assert table.loc[(10.0, 70.0, 5)]["neighbours"] == 5
    # A full corner: one neighbour on each of the three axes.
    assert table.loc[(5.0, 60.0, 5)]["neighbours"] == 3


def test_edge_recommendations_are_flagged():
    """A 'plateau' touching the grid boundary may just be an unexplored region."""
    report = analyse(synthetic_surface(spike=False))
    assert report.recommended_on_edge
    assert any("EDGE" in note for note in report.notes)


def test_report_states_both_cells_and_which_to_take():
    report = analyse(synthetic_surface(spike=True))
    text = render_plateau_report(report, data_source="synthetic")
    assert "BEST CELL" in text
    assert "RECOMMENDED" in text
    assert "DIFFER" in text
    assert "SYNTHETIC DATA" in text
    assert "robust score" in text.lower()


def test_analyse_rejects_a_table_missing_an_axis():
    frame = grid_from({(5.0, 70.0, 7): 1.0}).drop(columns=["time_stop_days"])
    with pytest.raises(ValueError, match="time_stop_days"):
        analyse(frame)


def test_analyse_works_on_real_sweep_output(results):
    report = analyse(results)
    assert set(AXES) <= set(report.best)
    assert report.table["robust_score"].notna().any()
    render_plateau_report(report, data_source="synthetic")


def test_plateau_handles_a_metric_that_is_negative_throughout():
    """Sharpe is routinely negative; spike detection must not divide by it."""
    values = {(e, x, t): -0.5
              for e in (5.0, 10.0, 15.0) for x in (60.0, 70.0, 80.0) for t in (5, 7)}
    values[(10.0, 70.0, 7)] = -0.1
    report = analyse(grid_from(values))
    assert report.best["sharpe"] == -0.1
    assert np.isfinite(report.best["spike_sigma"])


# --- rule 4: one out-of-sample evaluation -----------------------------------


def write_ledger(path, rows):
    from backtest.experiments import FIELDS, append_row
    for row in rows:
        append_row({f: row.get(f, "") for f in FIELDS}, path)


def test_prior_oos_runs_finds_test_and_full_windows(tmp_path):
    ledger = tmp_path / "experiments.csv"
    write_ledger(ledger, [
        {"run_id": "a", "window_label": "train", "config_hash": "h1"},
        {"run_id": "b", "window_label": "test", "config_hash": "h2"},
        {"run_id": "c", "window_label": "full", "config_hash": "h3"},
    ])
    found = prior_oos_runs(ledger)
    assert {r["run_id"] for r in found} == {"b", "c"}


def test_prior_oos_runs_on_a_missing_ledger_is_empty(tmp_path):
    assert prior_oos_runs(tmp_path / "nope.csv") == []


def test_second_oos_look_with_different_parameters_is_refused(tmp_path):
    from backtest.run import _oos_conflict

    ledger = tmp_path / "experiments.csv"
    settings = load_settings()
    write_ledger(ledger, [
        {"run_id": "a", "window_label": "test", "config_hash": "0000deadbeef",
         "sharpe": "1.9", "timestamp_utc": "2026-01-01T00:00:00Z"},
    ])
    message = _oos_conflict(ledger, settings)
    assert message is not None
    assert "rule 4" in message
    assert "0000deadbeef" in message
    assert "--i-have-decided-parameters" in message


def test_reproducing_the_same_oos_configuration_is_allowed(tmp_path):
    """Re-running an identical config is reproduction, not a second look."""
    from backtest.run import _oos_conflict

    ledger = tmp_path / "experiments.csv"
    settings = load_settings()
    write_ledger(ledger, [
        {"run_id": "a", "window_label": "test",
         "config_hash": settings.fingerprint()[0]},
    ])
    assert _oos_conflict(ledger, settings) is None


def test_a_clean_ledger_never_blocks_the_first_oos_run(tmp_path):
    from backtest.run import _oos_conflict
    assert _oos_conflict(tmp_path / "experiments.csv", load_settings()) is None
