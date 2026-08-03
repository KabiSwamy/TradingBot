"""Bar validation.

Bad data is an *assumption breach* in the ops-rulebook sense, not an outcome:
it means the system's picture of the world is wrong, so it must stop and be
looked at rather than quietly trading on it. These checks run on every fetch and
on every cache read.
"""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


class DataValidationError(ValueError):
    """Raised when bars are structurally unusable."""


def validate_bars(
    bars: pd.DataFrame,
    symbol: str,
    *,
    max_gap_business_days: int = 5,
) -> list[str]:
    """Raise on structural corruption; return a list of non-fatal warnings.

    Fatal (raises): missing columns, non-DatetimeIndex, unsorted or duplicated
    dates, NaNs, non-positive prices, or high/low that cannot bracket open/close.
    Each of these would silently produce fake fills or fake indicator values.

    Non-fatal (returned): unusually long calendar gaps. Real markets close for
    holidays, so a gap is suspicious rather than proof of corruption — it is
    surfaced for a human instead of halting the run.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in bars.columns]
    if missing:
        raise DataValidationError(
            f"{symbol}: missing column(s) {', '.join(missing)}"
        )

    if bars.empty:
        raise DataValidationError(f"{symbol}: no bars")

    if not isinstance(bars.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"{symbol}: index must be a DatetimeIndex, got {type(bars.index).__name__}"
        )

    if bars.index.has_duplicates:
        dupes = bars.index[bars.index.duplicated()].unique()[:5]
        raise DataValidationError(
            f"{symbol}: duplicate dates in index, e.g. {list(dupes)}"
        )

    if not bars.index.is_monotonic_increasing:
        raise DataValidationError(f"{symbol}: index is not sorted ascending")

    nan_counts = bars[list(OHLCV_COLUMNS)].isna().sum()
    if nan_counts.any():
        offending = nan_counts[nan_counts > 0].to_dict()
        raise DataValidationError(f"{symbol}: NaNs present: {offending}")

    for col in PRICE_COLUMNS:
        if (bars[col] <= 0).any():
            first = bars.index[bars[col] <= 0][0].date()
            raise DataValidationError(
                f"{symbol}: non-positive {col} price, first at {first}"
            )

    if (bars["volume"] < 0).any():
        raise DataValidationError(f"{symbol}: negative volume")

    # A high below open/close (or a low above them) means the bar is internally
    # inconsistent — most often a botched split adjustment.
    bad_high = bars["high"] < bars[["open", "close"]].max(axis=1)
    if bad_high.any():
        first = bars.index[bad_high][0].date()
        raise DataValidationError(
            f"{symbol}: high below open/close, first at {first} "
            "(usually a broken split adjustment)"
        )

    bad_low = bars["low"] > bars[["open", "close"]].min(axis=1)
    if bad_low.any():
        first = bars.index[bad_low][0].date()
        raise DataValidationError(
            f"{symbol}: low above open/close, first at {first} "
            "(usually a broken split adjustment)"
        )

    return _gap_warnings(bars, symbol, max_gap_business_days)


def _gap_warnings(
    bars: pd.DataFrame, symbol: str, max_gap_business_days: int
) -> list[str]:
    if len(bars) < 2:
        return []

    warnings: list[str] = []
    index = bars.index
    for prev, curr in zip(index[:-1], index[1:]):
        # Business days between consecutive bars; 1 means "no gap".
        gap = len(pd.bdate_range(prev, curr)) - 1
        if gap > max_gap_business_days:
            warnings.append(
                f"{symbol}: {gap} business-day gap between "
                f"{prev.date()} and {curr.date()}"
            )
    if len(warnings) > 10:
        extra = len(warnings) - 10
        warnings = warnings[:10] + [f"{symbol}: ...and {extra} more gaps"]
    return warnings
