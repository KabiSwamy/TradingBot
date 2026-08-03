"""Indicator tests — CLAUDE.md Testing section, family 3.

RSI(2) and SMA(200) against small hand-computed cases. The RSI smoothing method
is Wilder's and must stay identical everywhere in the project.

The hand computation for the primary case, seed = simple mean of the first
`period` gains/losses, then avg = (prev*(n-1) + current)/n:

    closes  [100, 101, 100, 98, 99, 103]
    deltas  [  +1,  -1,  -2, +1, +4]
    gains   [   1,   0,   0,  1,  4]
    losses  [   0,   1,   2,  0,  0]

    idx  close   avg_gain  avg_loss   RSI(2)
      2  100.00    0.5000    0.5000   50.0000   <- seed: mean of first 2
      3   98.00    0.2500    1.2500   16.6667
      4   99.00    0.6250    0.6250   50.0000
      5  103.00    2.3125    0.3125   88.0952
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.indicators import sma, wilder_rsi

HAND_CLOSES = [100.0, 101.0, 100.0, 98.0, 99.0, 103.0]
HAND_RSI = [np.nan, np.nan, 50.0, 100 / 6, 50.0, 100 - 100 / 8.4]


def series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype="float64")


def test_wilder_rsi_matches_hand_computed_values():
    got = wilder_rsi(series(HAND_CLOSES), period=2)
    expected = pd.Series(HAND_RSI, index=got.index)
    pd.testing.assert_series_equal(got, expected, check_names=False, atol=1e-10)


def test_seed_window_is_nan_not_a_guess():
    """The first `period` rows cannot have an RSI; they must be NaN, not 0 or 50."""
    got = wilder_rsi(series(HAND_CLOSES), period=2)
    assert got.iloc[:2].isna().all()
    assert got.iloc[2:].notna().all()


def test_flat_prices_give_neutral_50_not_100():
    """Both avg_gain and avg_loss are zero on flat prices.

    Order of the degenerate branches matters: a naive `avg_loss == 0 -> 100`
    check fires first on flat data and yields RSI=100, which is above any sane
    exit threshold and would invent exit signals out of nothing. Flat must be
    checked first and is neutral.
    """
    got = wilder_rsi(series([100.0] * 10), period=2)
    assert (got.iloc[2:] == 50.0).all()


def test_monotonic_gains_give_100():
    got = wilder_rsi(series([100.0, 101.0, 102.0, 103.0, 104.0]), period=2)
    assert (got.iloc[2:] == 100.0).all()


def test_monotonic_losses_give_0():
    got = wilder_rsi(series([104.0, 103.0, 102.0, 101.0, 100.0]), period=2)
    assert (got.iloc[2:] == 0.0).all()


def test_rsi_stays_within_bounds_on_noisy_data():
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    got = wilder_rsi(series(list(prices)), period=2).dropna()
    assert got.between(0.0, 100.0).all()


def test_rsi_is_causal():
    """An indicator-level look-ahead check (rule 1).

    RSI at bar i must depend only on bars <= i. Mutating the tail must leave
    every earlier value untouched.
    """
    prices = list(np.linspace(100, 130, 60) + np.sin(np.arange(60)) * 3)
    base = wilder_rsi(series(prices), period=2)

    cut = 40
    mutated_prices = prices[:cut] + [p * 3.0 for p in prices[cut:]]
    mutated = wilder_rsi(series(mutated_prices), period=2)

    pd.testing.assert_series_equal(base.iloc[:cut], mutated.iloc[:cut])


def test_rsi_rejects_nonsense_periods():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            wilder_rsi(series(HAND_CLOSES), period=bad)


def test_rsi_on_series_shorter_than_period_is_all_nan():
    """Newly listed symbols hit this; it must not raise."""
    got = wilder_rsi(series([100.0, 101.0]), period=5)
    assert got.isna().all()


def test_indicators_do_not_mutate_their_input():
    """Rule 9: no side effects anywhere in strategy code."""
    original = series(HAND_CLOSES)
    snapshot = original.copy()
    wilder_rsi(original, period=2)
    sma(original, period=3)
    pd.testing.assert_series_equal(original, snapshot)


# --- SMA -------------------------------------------------------------------


def test_sma_matches_hand_computed_values():
    got = sma(series([1.0, 2.0, 3.0, 4.0, 5.0]), period=3)
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0], index=got.index)
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_sma_needs_a_full_window_before_producing_a_value():
    """SMA(200) must be NaN for the first 199 bars.

    This is what stops a symbol from being traded before it has 200 bars of
    history: the regime filter reads NaN and refuses the entry.
    """
    got = sma(series([100.0] * 250), period=200)
    assert got.iloc[:199].isna().all()
    assert got.iloc[199:].notna().all()
    assert got.iloc[199] == 100.0


def test_sma_rejects_nonsense_periods():
    for bad in (0, -5):
        with pytest.raises(ValueError):
            sma(series(HAND_CLOSES), period=bad)
