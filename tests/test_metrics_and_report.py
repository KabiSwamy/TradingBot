"""Metrics, benchmark, report and ledger tests.

Rule 12 says what a report must contain; a report that can quietly drop the
benchmark or round a Sharpe into respectability is worse than no report.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from backtest.benchmark import buy_and_hold
from backtest.costs import CostModel
from backtest.experiments import FIELDS, append_row, window_label
from backtest.metrics import (
    compute_metrics,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    trade_stats,
)
from backtest.portfolio import Trade
from backtest.report import render_report, warnings_for
from config.loader import load_settings
from tests.conftest import make_bars


def equity_series(values: list[float]) -> pd.Series:
    index = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=len(values)))
    index.freq = None
    return pd.Series(values, index=index, dtype="float64")


def trade(net: float, *, shares: int = 100, entry: float = 100.0, days: int = 3) -> Trade:
    """Build a trade with a chosen net P&L, fees included."""
    exit_price = entry + (net + 2.0) / shares      # 2.0 total fees
    return Trade(
        symbol="TEST",
        shares=shares,
        entry_date=pd.Timestamp("2020-01-01"),
        entry_price=entry,
        entry_fee=1.0,
        exit_date=pd.Timestamp("2020-01-06"),
        exit_price=exit_price,
        exit_fee=1.0,
        exit_reason="rsi_exit",
        entry_bar=0,
        exit_bar=days,
    )


# --- Sharpe and drawdown ----------------------------------------------------


def test_sharpe_of_a_flat_curve_is_zero_not_undefined():
    assert sharpe_ratio(equity_series([100.0] * 10).pct_change().dropna()) == 0.0


def test_sharpe_matches_a_hand_computation():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_uses_sample_standard_deviation():
    """ddof=1, not 0 — the population form would inflate every Sharpe slightly."""
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    population = returns.mean() / returns.std(ddof=0) * np.sqrt(252)
    assert sharpe_ratio(returns) != pytest.approx(population)


def test_max_drawdown_is_hand_verifiable():
    """100 -> 120 -> 90: worst decline is 90/120 - 1 = -25%."""
    equity = equity_series([100.0, 110.0, 120.0, 100.0, 90.0, 95.0])
    dd, peak, trough = max_drawdown(equity)
    assert dd == pytest.approx(-0.25)
    assert peak == equity.index[2]
    assert trough == equity.index[4]


def test_drawdown_is_zero_at_every_new_high():
    equity = equity_series([100.0, 110.0, 120.0])
    assert (drawdown_series(equity) == 0.0).all()


def test_a_monotonic_curve_has_no_drawdown():
    assert max_drawdown(equity_series([100.0, 101.0, 102.0]))[0] == 0.0


# --- trade statistics -------------------------------------------------------


def test_trade_stats_are_per_round_trip_not_per_day():
    stats = trade_stats([trade(100.0), trade(-50.0), trade(200.0), trade(-25.0)])
    assert stats["trade_count"] == 4
    assert stats["win_rate"] == 0.5


def test_a_zero_pnl_trade_counts_as_a_loss():
    stats = trade_stats([trade(0.0), trade(100.0)])
    assert stats["win_rate"] == 0.5


def test_profit_factor_and_win_loss_ratio():
    stats = trade_stats([trade(300.0), trade(-100.0)])
    assert stats["profit_factor"] == pytest.approx(3.0)
    assert stats["win_loss_ratio"] == pytest.approx(3.0, rel=1e-3)


def test_empty_trade_list_does_not_divide_by_zero():
    stats = trade_stats([])
    assert stats["trade_count"] == 0
    assert stats["win_rate"] == 0.0


# --- benchmark --------------------------------------------------------------


def test_buy_and_hold_on_flat_prices_loses_exactly_two_fees():
    """The same exactness property as the strategy's cost test."""
    bars = make_bars(closes=[100.0] * 10, opens=[100.0] * 10)
    costs = CostModel(rate_per_side=0.0005)
    curve = buy_and_hold(bars, bars.index, 100_000.0, costs)

    # shares = 100000 / (100 * 1.0005); final = shares*100 - fee
    shares = 100_000.0 / (100.0 * 1.0005)
    expected = shares * 100.0 * (1.0 - 0.0005)
    assert curve.iloc[-1] == pytest.approx(expected, abs=1e-6)
    assert curve.iloc[-1] < 100_000.0


def test_buy_and_hold_tracks_the_underlying():
    bars = make_bars(closes=[100.0, 110.0, 120.0], opens=[100.0, 100.0, 110.0])
    curve = buy_and_hold(bars, bars.index, 100_000.0, CostModel(0.0005))
    assert curve.iloc[1] > curve.iloc[0]


def test_buy_and_hold_pays_costs_on_both_sides():
    bars = make_bars(closes=[100.0] * 5, opens=[100.0] * 5)
    with_costs = buy_and_hold(bars, bars.index, 100_000.0, CostModel(0.0005))
    without = buy_and_hold(bars, bars.index, 100_000.0, CostModel(0.0))
    assert without.iloc[-1] > with_costs.iloc[-1]
    assert without.iloc[-1] == pytest.approx(100_000.0)


# --- report -----------------------------------------------------------------


def metrics_with(**overrides):
    equity = equity_series([100_000.0, 101_000.0, 102_000.0, 101_500.0])
    m = compute_metrics(equity, label="strategy", trades=[trade(100.0)] * 40)
    return m if not overrides else type(m)(**{**m.as_dict(), **overrides})


def test_report_contains_every_rule_12_field():
    settings = load_settings()
    strategy = metrics_with()
    benchmark = compute_metrics(
        equity_series([100_000.0, 100_500.0, 101_000.0, 100_800.0]), label="benchmark"
    )
    text = render_report(
        strategy, benchmark, data_source="yfinance",
        window_label="full", settings=settings,
    )
    for required in ["Sharpe", "max drawdown", "win rate", "avg win", "avg loss",
                     "count", "BUY & HOLD"]:
        assert required in text, f"report is missing {required!r} (rule 12)"


def test_synthetic_runs_are_labelled_in_the_report():
    text = render_report(
        metrics_with(), None, data_source="synthetic",
        window_label="full", settings=load_settings(),
    )
    assert "SYNTHETIC DATA" in text
    assert "NOT A RESEARCH RESULT" in text


def test_real_runs_carry_no_synthetic_banner():
    text = render_report(
        metrics_with(), None, data_source="yfinance",
        window_label="full", settings=load_settings(),
    )
    assert "SYNTHETIC" not in text


def test_an_implausible_sharpe_is_flagged_as_a_probable_bug():
    """Working style: a Sharpe above ~2.5 is treated as a bug, not a triumph."""
    flagged = warnings_for(metrics_with(sharpe=3.4), None, None)
    assert any("SUSPECT" in w for w in flagged)
    assert not any("SUSPECT" in w for w in warnings_for(metrics_with(sharpe=1.1), None, None))


def test_losing_to_buy_and_hold_is_stated_plainly():
    strategy = metrics_with(sharpe=0.4)
    benchmark = metrics_with(sharpe=0.9)
    flagged = warnings_for(strategy, benchmark, None)
    assert any("Does not beat buy-and-hold" in w for w in flagged)


def test_too_few_trades_is_flagged():
    assert any("too few" in w for w in warnings_for(metrics_with(trade_count=8), None, None))


def test_report_states_its_assumptions():
    text = render_report(
        metrics_with(), None, data_source="yfinance",
        window_label="full", settings=load_settings(),
    )
    assert "risk-free rate 0" in text
    assert "hindsight-selected" in text
    assert "next session's open" in text


def test_equity_plot_is_written(tmp_path):
    from backtest.report import save_equity_plot

    equity = equity_series([100_000.0, 101_000.0, 99_000.0, 102_000.0])
    path = tmp_path / "curve.png"
    save_equity_plot(equity, None, path, title="t", data_source="synthetic")
    assert path.exists() and path.stat().st_size > 1000


# --- experiments ledger -----------------------------------------------------


def sample_row(**overrides) -> dict:
    row = {field: "" for field in FIELDS}
    row.update({"run_id": "r1", "sharpe": "1.23", "data_source": "synthetic"})
    row.update(overrides)
    return row


def test_header_is_written_once_and_rows_append(tmp_path):
    path = tmp_path / "experiments.csv"
    append_row(sample_row(run_id="r1"), path)
    append_row(sample_row(run_id="r2"), path)

    with path.open() as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(FIELDS)
    assert len(rows) == 3
    assert rows[1][FIELDS.index("run_id")] == "r1"
    assert rows[2][FIELDS.index("run_id")] == "r2"


def test_a_mismatched_header_raises_rather_than_corrupting_the_record(tmp_path):
    """Misaligned columns would make every later row subtly wrong and unnoticed."""
    path = tmp_path / "experiments.csv"
    path.write_text("run_id,sharpe\nold,1.0\n")
    with pytest.raises(ValueError, match="different schema"):
        append_row(sample_row(), path)


def test_unknown_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown experiment fields"):
        append_row({"not_a_field": "x"}, tmp_path / "experiments.csv")


def test_window_label_classifies_against_the_train_test_split():
    settings = load_settings()
    assert window_label("2010-01-01", "2019-12-31", settings) == "train"
    assert window_label("2020-01-01", "2026-01-01", settings) == "test"
    assert window_label("2010-01-01", "2026-01-01", settings) == "full"
