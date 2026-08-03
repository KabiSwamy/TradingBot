"""RSI(2) mean reversion — the v1 strategy.

Buy violent short-term dips inside long-term uptrends, sell the bounce.

    Regime filter: close > 200-day SMA, else no entries in that symbol
    Entry:         RSI(2) < 10 at the close
    Exit:          RSI(2) > 70 at the close

This module is a pure function of its arguments (rule 9): a DataFrame in,
signals out. No I/O, no clock, no randomness, no state. The identical function
is called by the backtester and — from Phase 3 — by the live nightly job, which
is the only reason paper results can be compared to backtest results at all.

Two things are deliberately *absent*:

* The 7-day time stop. Position age is engine state, and tracking it here would
  require the function to remember previous calls (rule 9).
* "If more signals than free slots, take the lowest RSI first." That needs all
  symbols at once plus knowledge of free slots — it is allocation, which rule 5
  places in the execution layer. This function exports the `rsi` column so the
  engine can rank on it.
"""

from __future__ import annotations

import pandas as pd

from strategies.indicators import sma, wilder_rsi

REQUIRED_COLUMNS = ("close",)


def generate_signals(
    bars: pd.DataFrame,
    *,
    rsi_period: int = 2,
    sma_period: int = 200,
    entry_rsi: float = 10.0,
    exit_rsi: float = 70.0,
) -> pd.DataFrame:
    """Compute entry/exit signals for ONE symbol's daily bars.

    Args:
        bars: DatetimeIndex-ed OHLCV frame for a single symbol. Only `close` is
            required; the engine uses `open` separately for fills.
        rsi_period: RSI lookback (Wilder smoothing).
        sma_period: Regime-filter moving-average lookback.
        entry_rsi: Enter when RSI closes strictly below this.
        exit_rsi: Exit when RSI closes strictly above this.

    Returns:
        A frame aligned to `bars.index` with columns:
            rsi           float, NaN until the seed window is full
            sma           float, NaN until `sma_period` bars exist
            regime_ok     bool, close > sma (False while sma is NaN)
            entry_signal  bool, regime_ok and rsi < entry_rsi
            exit_signal   bool, rsi > exit_rsi

        A signal on row t is a decision made at t's close. Turning it into a
        fill at t+1's open is the engine's job (rule 1).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(
            f"bars is missing required column(s): {', '.join(missing)}"
        )

    close = bars["close"].astype("float64")

    rsi = wilder_rsi(close, period=rsi_period)
    trend = sma(close, period=sma_period)

    # Comparison against NaN is False, so a symbol with fewer than `sma_period`
    # bars is simply never in-regime and never entered. That is the intended
    # behaviour for recently listed symbols, not an edge case to special-case.
    regime_ok = close > trend

    entry_signal = regime_ok & (rsi < entry_rsi)

    # Exits are deliberately NOT gated on the regime filter. A position opened
    # while the trend was intact must still be able to take its bounce after the
    # trend breaks; gating here would strand it until the time stop.
    exit_signal = rsi > exit_rsi

    return pd.DataFrame(
        {
            "rsi": rsi,
            "sma": trend,
            "regime_ok": regime_ok.fillna(False).astype(bool),
            "entry_signal": entry_signal.fillna(False).astype(bool),
            "exit_signal": exit_signal.fillna(False).astype(bool),
        },
        index=bars.index,
    )
