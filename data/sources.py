"""Bar sources: where daily OHLCV comes from.

Two implementations behind one protocol.

`YFinanceSource` is the real one. `SyntheticSource` generates deterministic
seeded bars so the engine and the whole test suite are runnable with no market
data access at all — which also keeps the tests offline and reproducible, as the
Testing section requires.

Synthetic bars are NOT a research result and every path that can emit them says
so: the report header labels the run, and results/experiments.csv carries a
`source` column so a synthetic run can never be mistaken for a real one later.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class BarSource(Protocol):
    """Anything that can produce daily OHLCV bars for a symbol."""

    name: str

    def fetch(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        """Return a DatetimeIndex-ed frame with OHLCV_COLUMNS, ascending by date."""
        ...


class YFinanceSource:
    """Daily split/dividend-adjusted bars from Yahoo Finance."""

    name = "yfinance"

    def fetch(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        import yfinance as yf  # imported lazily so offline runs never need it

        raw = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            actions=False,
        )
        if raw is None or raw.empty:
            raise DataFetchError(f"{symbol}: yfinance returned no rows")

        return _normalise_yfinance_frame(raw, symbol)


class DataFetchError(RuntimeError):
    """Raised when a source cannot produce bars."""


def _normalise_yfinance_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flatten yfinance's frame into the project's OHLCV shape.

    `auto_adjust=True` is not a convenience — it is required for correctness.
    It adjusts open, high, low and close together. Because this project fills at
    the OPEN, pairing a raw open with an adjusted close would mix two price
    scales and silently corrupt every fill across every split and dividend.
    """
    frame = raw.copy()

    # yfinance returns MultiIndex columns (field, ticker) in some versions.
    if isinstance(frame.columns, pd.MultiIndex):
        tickers = frame.columns.get_level_values(-1).unique()
        if len(tickers) != 1:
            raise DataFetchError(
                f"{symbol}: expected one ticker in response, got {list(tickers)}"
            )
        frame.columns = frame.columns.get_level_values(0)

    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]

    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise DataFetchError(
            f"{symbol}: yfinance response missing {', '.join(missing)} "
            f"(got {list(frame.columns)})"
        )

    frame = frame[OHLCV_COLUMNS].astype("float64")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"
    return frame.sort_index()


class SyntheticSource:
    """Deterministic pseudo-market bars for offline development and testing.

    Geometric Brownian motion with a mild upward drift, plus injected sharp
    dip-and-recover episodes so the RSI(2) entry condition actually fires inside
    an uptrend — flat noise would produce a backtest with no trades, which tests
    nothing.

    Determinism is per symbol AND stable under a changing end date: the bars for
    a given date never change because a longer range was requested later. That
    property needs a separate generator per quantity (see `_rng`) — with one
    shared stream, every draw's position depends on how many values the previous
    draw consumed, so extending the range silently rewrites history. The bars are
    also independent of which other symbols were requested.
    """

    name = "synthetic"

    def __init__(
        self,
        seed: int = 20100101,
        annual_drift: float = 0.07,
        annual_vol: float = 0.16,
        dip_probability: float = 0.010,
    ) -> None:
        self.seed = seed
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol
        self.dip_probability = dip_probability

    def _rng(self, symbol: str, stream: str) -> np.random.Generator:
        """An independent generator per (symbol, stream).

        Separate streams are not tidiness — they are what makes the series
        stable. Drawing everything from one generator means each draw's position
        in the stream depends on how many values the previous draw consumed, so
        asking for a longer date range shifts every later draw and silently
        rewrites history for dates that were already generated.
        """
        digest = hashlib.sha256(f"{self.seed}:{symbol}:{stream}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big"))

    def fetch(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        end = end or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        # freq is stripped so synthetic bars carry the same index shape as real
        # ones — a real trading calendar has holidays and no regular frequency.
        dates = pd.DatetimeIndex(pd.bdate_range(start=start, end=end), name="date")
        dates.freq = None
        if len(dates) == 0:
            raise DataFetchError(
                f"{symbol}: synthetic range {start}..{end} contains no business days"
            )

        # One generator per concern. See _rng: sharing a single stream makes the
        # whole series depend on the requested end date, so the same dates would
        # come back with different prices tomorrow.
        path_rng = self._rng(symbol, "path")
        dip_rng = self._rng(symbol, "dips")
        gap_rng = self._rng(symbol, "gap")
        spread_rng = self._rng(symbol, "spread")
        volume_rng = self._rng(symbol, "volume")

        n = len(dates)
        dt = 1.0 / 252.0

        mu = self.annual_drift * dt
        sigma = self.annual_vol * np.sqrt(dt)
        shocks = path_rng.normal(mu, sigma, n)

        # Inject dip-and-recover episodes: a few days sharply down, then the
        # same amount handed back over the following days. This is what gives
        # RSI(2) something to find inside an uptrend.
        #
        # Each episode is constructed to sum to zero, so the injections change
        # the *path* without changing the drift. An earlier version subtracted
        # on the way down without adding back, which quietly overwhelmed the 7%
        # drift with roughly -17%/yr of dip and turned the whole series into a
        # permanent bear market — the kill switch fired within three years and
        # the rest of the run was flat cash, which exercises almost nothing.
        i = 0
        while i < n:
            if dip_rng.random() < self.dip_probability:
                fall = int(dip_rng.integers(2, 5))
                recover = int(dip_rng.integers(2, 6))
                depth = float(dip_rng.uniform(0.015, 0.030))
                end_fall = min(i + fall, n)
                shocks[i:end_fall] -= depth
                dropped = depth * (end_fall - i)
                end_recover = min(end_fall + recover, n)
                if end_recover > end_fall:
                    shocks[end_fall:end_recover] += dropped / (end_recover - end_fall)
                i = end_recover
            else:
                i += 1

        close = 100.0 * np.exp(np.cumsum(shocks))

        # Build an internally consistent bar around each close. The open gaps
        # from the previous close, and high/low bracket both by construction so
        # the validator's invariants hold.
        prev_close = np.concatenate([[close[0]], close[:-1]])
        gap = gap_rng.normal(0.0, sigma * 0.5, n)
        open_ = prev_close * np.exp(gap)

        spread = np.abs(spread_rng.normal(0.0, sigma * 0.6, n))
        top = np.maximum(open_, close)
        bottom = np.minimum(open_, close)
        high = top * (1.0 + spread)
        low = bottom * (1.0 - spread)

        volume = volume_rng.integers(1_000_000, 100_000_000, n).astype("float64")

        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        ).astype("float64")


def get_source(name: str, **kwargs) -> BarSource:
    """Resolve a source by CLI name."""
    if name == "yfinance":
        return YFinanceSource()
    if name == "synthetic":
        return SyntheticSource(**kwargs)
    raise ValueError(f"unknown data source {name!r} (expected yfinance or synthetic)")
