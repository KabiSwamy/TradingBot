"""Kill-switch tests — CLAUDE.md Testing family 5, rule 5.

Arithmetic is pinned down with max_positions=1, so all capital is in one
position and the equity path is checkable by hand:

    entry at open 100.00 -> shares = floor(100000 / (100 * 1.0005)) = 999
    notional 99900.00, fee 49.95, residual cash 50.05
    equity at a close of P = 999 * P + 50.05
    peak = 100000.00 (bar 0, before the fill) -> trigger at equity <= 85000.00

        close 86.00 -> equity 85964.05 -> -14.036%  no trigger
        close 85.00 -> equity 84965.05 -> -15.035%  triggers
"""

from __future__ import annotations

import pytest

from backtest.engine import run_backtest
from config.loader import load_settings
from tests.conftest import make_bars, make_signals


def run(closes, *, entries=(0,), exits=(), time_stop=999, **overrides):
    bars = make_bars(closes=closes, opens=closes)
    signals = make_signals(bars.index, entries=entries, exits=exits)
    settings = load_settings(
        max_positions=1, time_stop_days=time_stop, **overrides
    )
    return run_backtest(
        {"TEST": bars}, {"TEST": signals}, settings, force_close_at_end=False
    )


def test_does_not_trigger_above_the_threshold():
    """Negative control: -14.04% must NOT fire a -15% switch."""
    result = run([100.0, 100.0, 95.0, 90.0, 86.0, 86.0, 86.0])
    assert not result.kill_switch.triggered
    assert result.equity.iloc[4] == pytest.approx(85_964.05, abs=1e-6)


def test_triggers_when_drawdown_crosses_fifteen_percent():
    result = run([100.0, 100.0, 95.0, 90.0, 86.0, 85.0, 85.0, 85.0, 85.0])
    assert result.kill_switch.triggered
    assert result.equity.iloc[5] == pytest.approx(84_965.05, abs=1e-6)
    assert result.kill_switch.trigger_drawdown == pytest.approx(-0.1503495, abs=1e-6)
    assert result.kill_switch.trigger_date == result.calendar[5]


def test_flatten_happens_at_the_next_open():
    result = run([100.0, 100.0, 95.0, 90.0, 86.0, 85.0, 85.0, 85.0, 85.0])
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "kill_switch"
    assert trade.exit_bar == 6                    # trigger at bar 5, fill at bar 6


def test_everything_is_flat_after_the_flatten():
    result = run([100.0, 100.0, 95.0, 90.0, 86.0, 85.0, 85.0, 85.0, 85.0])
    after = result.positions_count.iloc[6:]
    assert (after == 0).all()

    equity_after = result.equity.iloc[6:]
    assert equity_after.nunique() == 1, "equity must be flat cash once halted"


def test_no_further_entries_are_taken_after_halting():
    """Rule 5: halting means halted, not 'paused until the next signal'."""
    result = run(
        [100.0, 100.0, 95.0, 90.0, 86.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0],
        entries=(0, 6, 7, 8, 9),
    )
    assert result.kill_switch.triggered
    assert len(result.trades) == 1, "a new position was opened after the halt"

    for record in result.decisions[6:]:
        assert record["halted"] is True
        assert not record["orders"], "orders were still being issued after the halt"


def test_the_equity_curve_runs_to_the_end_of_the_window():
    """Halting must not truncate the run — rule 12 compares against the same span."""
    closes = [100.0, 100.0, 95.0, 90.0, 86.0, 85.0] + [85.0] * 10
    result = run(closes)
    assert len(result.equity) == len(closes)
    assert result.equity.notna().all()


def test_the_threshold_is_inclusive():
    """An exact -15.000% must fire, not sit one cent below the line."""
    # 999 shares * P + 50.05 == 85000  ->  P = 85.0350350...
    exact = (85_000.0 - 50.05) / 999
    result = run([100.0, 100.0, exact, exact, exact])
    assert result.equity.iloc[2] == pytest.approx(85_000.0, abs=1e-6)
    assert result.kill_switch.triggered


def test_the_threshold_is_configurable():
    closes = [100.0, 100.0, 95.0, 94.0, 94.0, 94.0]
    assert not run(closes).kill_switch.triggered
    assert run(closes, kill_switch_drawdown=0.05).kill_switch.triggered


def test_the_trigger_is_recorded_in_the_decision_log():
    """Rule 10: the reason for every action has to be in the log."""
    result = run([100.0, 100.0, 95.0, 90.0, 86.0, 85.0, 85.0, 85.0])
    triggers = [d for d in result.decisions if "kill_switch" in d]
    assert len(triggers) == 1
    assert triggers[0]["kill_switch"]["triggered"] is True
    assert triggers[0]["kill_switch"]["drawdown"] == pytest.approx(-0.1503495, abs=1e-6)
