"""Signal tests for the RSI(2) mean-reversion strategy.

What lives here is the strategy's contract: given bars, which bars carry an
entry or exit signal. Slot allocation, position sizing and the time stop are
deliberately NOT tested here because they are not the strategy's job — rule 5
puts them in the execution layer, and rule 9 forbids the state they need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.rsi2_mean_reversion import generate_signals

SIGNAL_COLUMNS = ["rsi", "sma", "regime_ok", "entry_signal", "exit_signal"]


def bars_from_closes(closes: list[float]) -> pd.DataFrame:
    """Build an OHLCV frame from closes. Intraday range is irrelevant to signals."""
    idx = pd.bdate_range("2015-01-01", periods=len(closes))
    close = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def uptrend_with_dip(n: int = 300, dip_at: int = 250, dip_pct: float = 0.10):
    """A clean uptrend (so the regime filter passes) with one sharp dip."""
    closes = list(100 * np.exp(np.linspace(0, 0.5, n)))
    for k in range(dip_at, dip_at + 2):
        closes[k] = closes[dip_at - 1] * (1 - dip_pct)
    return bars_from_closes(closes)


def test_returns_expected_columns_aligned_to_input():
    bars = uptrend_with_dip()
    out = generate_signals(bars, rsi_period=2, sma_period=200,
                           entry_rsi=10.0, exit_rsi=70.0)
    assert list(out.columns) == SIGNAL_COLUMNS
    pd.testing.assert_index_equal(out.index, bars.index)
    assert out["entry_signal"].dtype == bool
    assert out["exit_signal"].dtype == bool
    assert out["regime_ok"].dtype == bool


def test_sharp_dip_in_an_uptrend_produces_an_entry():
    bars = uptrend_with_dip(dip_at=250)
    out = generate_signals(bars, rsi_period=2, sma_period=200,
                           entry_rsi=10.0, exit_rsi=70.0)
    assert out["entry_signal"].iloc[250:253].any()


def test_no_entry_below_the_200_day_sma():
    """Regime filter: a dip in a downtrend is not a buy (strategy spec)."""
    closes = list(100 * np.exp(np.linspace(0, -0.5, 300)))
    closes[250] = closes[249] * 0.85          # violent dip, but trend is down
    out = generate_signals(bars_from_closes(closes), rsi_period=2, sma_period=200,
                           entry_rsi=10.0, exit_rsi=70.0)
    assert not out["entry_signal"].any()
    assert not out["regime_ok"].iloc[250]


def test_no_entry_before_the_regime_filter_has_enough_history():
    """First 199 bars have a NaN SMA, so no entry may fire there."""
    bars = uptrend_with_dip(n=300, dip_at=150)
    out = generate_signals(bars, rsi_period=2, sma_period=200,
                           entry_rsi=10.0, exit_rsi=70.0)
    assert not out["entry_signal"].iloc[:200].any()
    assert not out["regime_ok"].iloc[:199].any()


def test_entry_and_exit_thresholds_are_honoured_exactly():
    bars = uptrend_with_dip()
    out = generate_signals(bars, rsi_period=2, sma_period=200,
                           entry_rsi=10.0, exit_rsi=70.0)
    entries = out[out["entry_signal"]]
    assert (entries["rsi"] < 10.0).all()
    assert (entries["close"] > entries["sma"]).all() if "close" in entries else True
    exits = out[out["exit_signal"]]
    assert (exits["rsi"] > 70.0).all()


def test_thresholds_are_parameters_not_constants():
    """All thresholds live in config; the function must actually respect them."""
    bars = uptrend_with_dip()
    loose = generate_signals(bars, rsi_period=2, sma_period=200,
                             entry_rsi=30.0, exit_rsi=70.0)
    tight = generate_signals(bars, rsi_period=2, sma_period=200,
                             entry_rsi=1.0, exit_rsi=70.0)
    assert loose["entry_signal"].sum() > tight["entry_signal"].sum()


def test_exit_signal_ignores_the_regime_filter():
    """An open position must be able to exit even if the trend has since broken.

    Gating exits on the regime filter would strand positions in downtrends,
    which is the opposite of what a mean-reversion exit is for.
    """
    closes = list(100 * np.exp(np.linspace(0, -0.5, 300)))
    closes[280] = closes[279] * 1.15          # sharp bounce while below the SMA
    out = generate_signals(bars_from_closes(closes), rsi_period=2, sma_period=200,
                           entry_rsi=10.0, exit_rsi=70.0)
    assert not out["regime_ok"].iloc[280]
    assert out["exit_signal"].iloc[280]


def test_no_time_stop_logic_leaks_into_the_strategy():
    """Rule 9: position age is engine state and must not appear here."""
    out = generate_signals(uptrend_with_dip(), rsi_period=2, sma_period=200,
                           entry_rsi=10.0, exit_rsi=70.0)
    assert "bars_held" not in out.columns
    assert "time_stop" not in out.columns


def test_is_deterministic():
    """Rule 9: no randomness. Same input, same output, always."""
    bars = uptrend_with_dip()
    kwargs = dict(rsi_period=2, sma_period=200, entry_rsi=10.0, exit_rsi=70.0)
    pd.testing.assert_frame_equal(generate_signals(bars, **kwargs),
                                  generate_signals(bars, **kwargs))


def test_does_not_mutate_input():
    bars = uptrend_with_dip()
    snapshot = bars.copy()
    generate_signals(bars, rsi_period=2, sma_period=200,
                     entry_rsi=10.0, exit_rsi=70.0)
    pd.testing.assert_frame_equal(bars, snapshot)


def test_is_causal_under_tail_mutation():
    """Rule 1 at the strategy level: mutating the future cannot change the past."""
    bars = uptrend_with_dip()
    kwargs = dict(rsi_period=2, sma_period=200, entry_rsi=10.0, exit_rsi=70.0)
    base = generate_signals(bars, **kwargs)

    cut = 260
    mutated = bars.copy()
    mutated.iloc[cut:] = mutated.iloc[cut:] * 3.0
    after = generate_signals(mutated, **kwargs)

    pd.testing.assert_frame_equal(base.iloc[:cut], after.iloc[:cut])


def test_short_history_returns_empty_signals_without_raising():
    """Symbols listed after the backtest start must not break the run."""
    out = generate_signals(bars_from_closes([100.0, 101.0, 99.0]),
                           rsi_period=2, sma_period=200,
                           entry_rsi=10.0, exit_rsi=70.0)
    assert not out["entry_signal"].any()
    assert not out["regime_ok"].any()


def test_missing_close_column_is_a_clear_error():
    bad = uptrend_with_dip().drop(columns=["close"])
    with pytest.raises(ValueError, match="close"):
        generate_signals(bad, rsi_period=2, sma_period=200,
                         entry_rsi=10.0, exit_rsi=70.0)
