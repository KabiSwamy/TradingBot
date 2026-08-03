"""Time stop (rule 6) and slot allocation (rule 5) in the execution layer."""

from __future__ import annotations

from backtest.engine import run_backtest
from config.loader import load_settings
from tests.conftest import make_bars, make_signals


def run(bars: dict, signals: dict, **overrides):
    return run_backtest(
        bars, signals, load_settings(**overrides), force_close_at_end=False
    )


def one(bars, signals, **overrides):
    return run({"TEST": bars}, {"TEST": signals}, **overrides)


# --- time stop (rule 6) -----------------------------------------------------


def test_time_stop_exits_after_seven_sessions():
    """The fill day counts as day 1, so the exit order is placed at the close of
    the 7th session and fills at the open of the 8th.

        signal   close of bar 0
        fill     open  of bar 1   days_held = 1
        ...      bar 7            days_held = 7  -> time-stop order
        exit     open  of bar 8
    """
    bars = make_bars(closes=[100.0] * 15, opens=[100.0] * 15)
    result = one(bars, make_signals(bars.index, entries=(0,)))

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "time_stop"
    assert trade.entry_bar == 1
    assert trade.exit_bar == 8
    assert trade.holding_days == 7   # bought open of bar 1, sold open of bar 8


def test_time_stop_honours_a_configured_value():
    """Rule 6 is non-negotiable but the length is config, not a magic number."""
    bars = make_bars(closes=[100.0] * 15, opens=[100.0] * 15)
    result = one(bars, make_signals(bars.index, entries=(0,)), time_stop_days=3)
    assert result.trades[0].exit_bar == 4


def test_rsi_exit_takes_precedence_over_the_time_stop():
    bars = make_bars(closes=[100.0] * 15, opens=[100.0] * 15)
    result = one(bars, make_signals(bars.index, entries=(0,), exits=(3,)))
    assert result.trades[0].exit_reason == "rsi_exit"


def test_a_position_can_exit_the_day_after_it_fills():
    """Entry fills at open of bar 1; an exit signal at close of bar 1 is legal."""
    bars = make_bars(closes=[100.0] * 10, opens=[100.0] * 10)
    result = one(bars, make_signals(bars.index, entries=(0,), exits=(1,)))
    assert result.trades[0].exit_bar == 2
    assert result.trades[0].exit_reason == "rsi_exit"


def test_time_stop_cannot_be_dodged_by_re_entering_the_same_morning():
    """A symbol exiting today is excluded from today's entry candidates.

    Without that exclusion a time-stopped position could be sold and rebought at
    the same open whenever an entry signal happened to coincide, resetting the
    clock forever and making rule 6 decorative.
    """
    bars = make_bars(closes=[100.0] * 20, opens=[100.0] * 20)
    # Entry signal on every bar: the strongest possible pressure to re-enter.
    signals = make_signals(bars.index, entries=tuple(range(20)))
    result = one(bars, signals)

    for trade in result.trades:
        assert trade.holding_days <= 7

    # And the re-entry must happen strictly after the exit, never at the same open.
    exits = sorted(t.exit_bar for t in result.trades)
    entries = sorted(t.entry_bar for t in result.trades)
    for exit_bar, next_entry in zip(exits, entries[1:]):
        assert next_entry > exit_bar


# --- slot allocation (rule 5) -----------------------------------------------


def five_symbols(rsi_by_symbol: dict[str, float]):
    """Five symbols all signalling on the same close, with distinct RSI values."""
    bars, signals = {}, {}
    for symbol, rsi in rsi_by_symbol.items():
        frame = make_bars(closes=[100.0] * 15, opens=[100.0] * 15)
        bars[symbol] = frame
        signals[symbol] = make_signals(frame.index, entries=(0,), rank_key=rsi)
    return bars, signals


def test_never_exceeds_max_concurrent_positions():
    bars, signals = five_symbols(
        {"AAA": 9.0, "BBB": 3.0, "CCC": 7.0, "DDD": 1.0, "EEE": 5.0}
    )
    result = run(bars, signals)
    assert result.positions_count.max() == 3


def test_lowest_rsi_symbols_win_the_free_slots():
    """Strategy spec: if more signals than free slots, take the lowest RSI first."""
    bars, signals = five_symbols(
        {"AAA": 9.0, "BBB": 3.0, "CCC": 7.0, "DDD": 1.0, "EEE": 5.0}
    )
    result = run(bars, signals)
    filled = {t.symbol for t in result.trades}
    assert filled == {"DDD", "BBB", "EEE"}      # rsi 1, 3, 5


def test_symbols_that_miss_out_are_logged_with_a_reason():
    bars, signals = five_symbols(
        {"AAA": 9.0, "BBB": 3.0, "CCC": 7.0, "DDD": 1.0, "EEE": 5.0}
    )
    result = run(bars, signals)
    skipped = {
        s["symbol"]
        for day in result.decisions
        for s in day["skipped"]
        if s["reason"] == "no_free_slot"
    }
    assert {"AAA", "CCC"} <= skipped


def test_ties_break_alphabetically_so_runs_are_reproducible():
    bars, signals = five_symbols(
        {"AAA": 5.0, "BBB": 5.0, "CCC": 5.0, "DDD": 5.0, "EEE": 5.0}
    )
    result = run(bars, signals)
    assert {t.symbol for t in result.trades} == {"AAA", "BBB", "CCC"}


def test_max_positions_is_configurable():
    bars, signals = five_symbols(
        {"AAA": 9.0, "BBB": 3.0, "CCC": 7.0, "DDD": 1.0, "EEE": 5.0}
    )
    assert run(bars, signals, max_positions=1).positions_count.max() == 1
    assert run(bars, signals, max_positions=5).positions_count.max() == 5


def test_a_freed_slot_is_reusable_by_another_symbol_at_the_same_open():
    """Sells execute before buys, so the proceeds and the slot are both available."""
    bars, signals = {}, {}
    held = make_bars(closes=[100.0] * 15, opens=[100.0] * 15)
    bars["AAA"] = held
    signals["AAA"] = make_signals(held.index, entries=(0,), exits=(3,), rank_key=1.0)

    other = make_bars(closes=[100.0] * 15, opens=[100.0] * 15)
    bars["ZZZ"] = other
    signals["ZZZ"] = make_signals(other.index, entries=(3,), rank_key=2.0)

    result = run(bars, signals, max_positions=1)
    symbols = [t.symbol for t in result.trades]
    assert "AAA" in symbols and "ZZZ" in symbols


def test_positions_open_at_the_end_are_counted_as_trades():
    """Dropping them would quietly remove losers that had not yet recovered."""
    bars = make_bars(closes=[100.0] * 6, opens=[100.0] * 6)
    result = run_backtest(
        {"TEST": bars}, {"TEST": make_signals(bars.index, entries=(0,))},
        load_settings(), force_close_at_end=True,
    )
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end_of_backtest"
    assert result.positions_count.iloc[-1] == 0
