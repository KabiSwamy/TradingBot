"""Fill-timing and cost tests — CLAUDE.md Testing families 2 and 4.

Signals are injected rather than derived from prices, so every number below is
hand-checkable. See `make_signals` in conftest.py for why.
"""

from __future__ import annotations

import pytest

from backtest.costs import CostModel
from backtest.engine import run_backtest
from config.loader import load_settings
from tests.conftest import make_bars, make_signals


def settings(**overrides):
    return load_settings(**overrides)


def run(bars, signals, **overrides):
    return run_backtest(
        {"TEST": bars}, {"TEST": signals}, settings(**overrides),
        force_close_at_end=False,
    )


# --- fill timing (family 2) -------------------------------------------------


def test_entry_fills_at_next_day_open_not_signal_day_close():
    """The signal-day close and the next open differ sharply on purpose."""
    bars = make_bars(
        closes=[100.0, 100.0, 100.0, 100.0, 100.0],
        opens=[100.0, 100.0, 90.0, 100.0, 100.0],
    )
    signals = make_signals(bars.index, entries=(1,))
    result = run(bars, signals)

    fills = [f for day in result.decisions for f in day["fills"]]
    assert len(fills) == 1
    assert fills[0]["price"] == 90.0, "filled at the signal-day close, not the next open"
    assert result.decisions[2]["fills"], "fill landed on the wrong day"


def test_engine_takes_an_unfavourable_open_too():
    """Mirror of the above: no cherry-picking the better of close and open."""
    bars = make_bars(
        closes=[100.0, 100.0, 100.0, 100.0, 100.0],
        opens=[100.0, 100.0, 130.0, 100.0, 100.0],
    )
    result = run(bars, make_signals(bars.index, entries=(1,)))
    fills = [f for day in result.decisions for f in day["fills"]]
    assert fills[0]["price"] == 130.0


def test_moving_the_signal_moves_the_fill():
    bars = make_bars(closes=[100.0] * 6, opens=[100.0, 100.0, 90.0, 80.0, 100.0, 100.0])
    result = run(bars, make_signals(bars.index, entries=(2,)))
    assert result.decisions[3]["fills"][0]["price"] == 80.0
    assert not result.decisions[2]["fills"]


def test_exit_also_fills_at_the_next_open():
    bars = make_bars(
        closes=[100.0] * 8,
        opens=[100.0, 100.0, 100.0, 100.0, 80.0, 100.0, 100.0, 100.0],
    )
    signals = make_signals(bars.index, entries=(1,), exits=(3,))
    result = run(bars, signals)

    sells = [f for day in result.decisions for f in day["fills"] if f["side"] == "sell"]
    assert len(sells) == 1
    assert sells[0]["price"] == 80.0
    assert result.trades[0].exit_date == bars.index[4]


def test_every_fill_was_decided_strictly_earlier_than_it_filled():
    """A global rule-1 invariant, cheap to assert everywhere."""
    bars = make_bars(closes=[100.0 + i for i in range(20)])
    signals = make_signals(bars.index, entries=(2, 10), exits=(6, 14))
    result = run(bars, signals)

    checked = 0
    for day in result.decisions:
        for fill in day["fills"]:
            assert fill["decided_date"] < day["date"]
            checked += 1
    assert checked >= 2


def test_signal_on_the_final_bar_cannot_fill():
    """There is no next open, so the order must be recorded, not imagined."""
    bars = make_bars(closes=[100.0] * 5)
    result = run(bars, make_signals(bars.index, entries=(4,)))

    assert not result.trades
    assert result.decisions[-1]["unfilled_end_of_data"][0]["symbol"] == "TEST"


# --- costs (family 4) -------------------------------------------------------


def test_round_trip_on_flat_prices_loses_exactly_the_modelled_cost():
    """Rule 2, made exact.

    Capital 100,000, 3 slots, 5bps/side, every price exactly 100.00:
        target notional = 100000 / 3            = 33333.3333
        shares          = floor(33333.3333 / (100 * 1.0005)) = 333
        notional        = 33300.00 per side, fee = 16.65 per side
        total cost      = 33.30 == 2 x fee
    """
    bars = make_bars(closes=[100.0] * 12, opens=[100.0] * 12)
    signals = make_signals(bars.index, entries=(0,), exits=(5,))
    result = run(bars, signals)

    assert len(result.trades) == 1
    trade = result.trades[0]

    assert trade.shares == 333
    assert trade.gross_pnl == 0.0
    assert trade.entry_fee == pytest.approx(16.65, abs=1e-9)
    assert trade.exit_fee == pytest.approx(16.65, abs=1e-9)
    assert trade.net_pnl == pytest.approx(-33.30, abs=1e-9)

    final_equity = result.equity.iloc[-1]
    assert 100_000.0 - final_equity == pytest.approx(33.30, abs=1e-9)


def test_a_flat_trade_counts_as_a_loss():
    """It paid costs and returned nothing; calling it a win would flatter the report."""
    bars = make_bars(closes=[100.0] * 12, opens=[100.0] * 12)
    result = run(bars, make_signals(bars.index, entries=(0,), exits=(5,)))
    assert not result.trades[0].is_win


def test_costs_are_charged_on_both_sides():
    bars = make_bars(closes=[100.0] * 12, opens=[100.0] * 12)
    result = run(bars, make_signals(bars.index, entries=(0,), exits=(5,)))
    trade = result.trades[0]
    assert trade.fees == pytest.approx(trade.entry_fee + trade.exit_fee)
    assert trade.entry_fee > 0 and trade.exit_fee > 0


def test_cost_model_share_sizing_never_overdraws():
    """floor(budget / (price * (1+rate))) must leave cash non-negative."""
    costs = CostModel(rate_per_side=0.0005)
    for budget, price in [(1000.0, 33.33), (100.0, 99.99), (5.0, 100.0), (1e6, 7.0)]:
        shares = costs.shares_for_budget(budget, price)
        assert costs.buy_cost(shares, price) <= budget + 1e-9


def test_cost_model_zero_rate_is_exact():
    """A unit test of the model itself — the only place zero cost is permitted."""
    costs = CostModel(rate_per_side=0.0)
    assert costs.fee(1000.0) == 0.0
    assert costs.buy_cost(10, 100.0) == 1000.0
    assert costs.sell_proceeds(10, 100.0) == 1000.0


def test_insufficient_cash_is_logged_not_silently_dropped():
    bars = make_bars(closes=[100.0] * 8, opens=[100.0] * 8)
    signals = make_signals(bars.index, entries=(0,))
    result = run_backtest(
        {"TEST": bars}, {"TEST": signals},
        settings(initial_capital=10.0),      # cannot afford a single share
        force_close_at_end=False,
    )
    reasons = [c["reason"] for day in result.decisions for c in day["cancelled"]]
    assert "insufficient_cash" in reasons
    assert not result.trades
