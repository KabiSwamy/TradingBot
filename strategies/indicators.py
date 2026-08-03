"""Indicator maths. Pure functions over pandas Series — no I/O, no state (rule 9).

The RSI smoothing method is **Wilder's**, and it is the only one used anywhere in
this project (CLAUDE.md Testing section). Two implementations that disagree by a
smoothing constant would make backtest and live results silently incomparable,
which is exactly what rule 9 exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Seeded with the simple mean of the first `period` gains and losses, then
    smoothed recursively:

        avg = (prev_avg * (period - 1) + current) / period

    The first `period` values are NaN — there is no honest RSI before the seed
    window is full, and emitting a placeholder there would let the strategy
    trade on a number that does not exist yet.

    Returns a Series aligned to `close`, with values in [0, 100].
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    delta = close.diff()
    gains = delta.clip(lower=0.0).to_numpy(dtype="float64")
    losses = (-delta).clip(lower=0.0).to_numpy(dtype="float64")

    n = len(close)
    rsi = np.full(n, np.nan, dtype="float64")
    if n <= period:
        # Not enough history to seed. Newly listed symbols land here.
        return pd.Series(rsi, index=close.index, name="rsi")

    # gains[0] is NaN (no prior bar); the first `period` real deltas are at
    # positions 1..period, and the seed lands on bar `period`.
    avg_gain = float(np.mean(gains[1 : period + 1]))
    avg_loss = float(np.mean(losses[1 : period + 1]))
    rsi[period] = _rsi_from_averages(avg_gain, avg_loss)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i] = _rsi_from_averages(avg_gain, avg_loss)

    return pd.Series(rsi, index=close.index, name="rsi")


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    """Convert smoothed averages to RSI, handling the degenerate cases.

    Branch order is deliberate and load-bearing. On perfectly flat prices both
    averages are zero; checking `avg_loss == 0` first would return 100, which
    sits above any sane exit threshold and would manufacture exit signals from a
    series that never moved. Flat is neutral, so it is tested first.
    """
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0        # flat — no strength in either direction
    if avg_loss == 0.0:
        return 100.0       # only gains
    if avg_gain == 0.0:
        return 0.0         # only losses
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average.

    `min_periods=period` so the first `period - 1` values are NaN. That NaN is
    what keeps a symbol untraded until it has enough history for the regime
    filter to mean anything — a comparison against NaN is False, so no entry.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return close.rolling(window=period, min_periods=period).mean().rename("sma")
