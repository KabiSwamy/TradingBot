"""Code paths that only REAL market data reaches.

The synthetic source produces perfectly regular business-day bars for every
symbol over an identical span. Real data does none of that: yfinance returns
MultiIndex columns in some versions and flat ones in others, a tz-aware index
sometimes and a naive one otherwise, calendars have holidays, and symbols list
on different dates.

This environment cannot reach Yahoo Finance (egress policy), so these paths
would otherwise ship completely unexercised — the gap that most deserves tests
precisely because it is the one the day-to-day suite cannot see. Everything here
uses fabricated payloads shaped like the real thing, and touches no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.cache import build_calendar, load_bars
from data.sources import DataFetchError, _normalise_yfinance_frame
from data.validate import validate_bars
from tests.conftest import make_bars


def yfinance_payload(*, multiindex=False, tz=None, ticker="SPY", n=5) -> pd.DataFrame:
    """A frame shaped the way yfinance returns one, with auto_adjust=True."""
    index = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=n))
    index.freq = None
    if tz:
        index = index.tz_localize(tz)

    data = {
        "Open": np.linspace(100, 104, n),
        "High": np.linspace(101, 105, n),
        "Low": np.linspace(99, 103, n),
        "Close": np.linspace(100.5, 104.5, n),
        "Volume": np.full(n, 1_000_000.0),
    }
    frame = pd.DataFrame(data, index=index)
    if multiindex:
        frame.columns = pd.MultiIndex.from_product([frame.columns, [ticker]])
    return frame


# --- yfinance response shapes ----------------------------------------------


def test_flat_columns_are_normalised():
    out = _normalise_yfinance_frame(yfinance_payload(), "SPY")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.name == "date"
    assert out.index.tz is None


def test_multiindex_columns_are_flattened():
    """yfinance returns (field, ticker) columns in several versions."""
    out = _normalise_yfinance_frame(yfinance_payload(multiindex=True), "SPY")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["open"].iloc[0] == pytest.approx(100.0)


def test_a_multi_ticker_response_is_rejected_rather_than_guessed_at():
    """Silently picking one of two tickers would return the wrong symbol's prices."""
    one = yfinance_payload(multiindex=True, ticker="SPY")
    two = yfinance_payload(multiindex=True, ticker="QQQ")
    frame = pd.concat([one, two], axis=1)
    with pytest.raises(DataFetchError, match="expected one ticker"):
        _normalise_yfinance_frame(frame, "SPY")


@pytest.mark.parametrize("tz", ["America/New_York", "UTC", None])
def test_index_is_made_tz_naive_and_midnight_aligned(tz):
    """yfinance is inconsistent about tz-awareness across versions and symbols.

    Mixing tz-aware and tz-naive indices across the universe makes them
    un-unionable into one calendar, so everything is normalised on the way in.
    """
    out = _normalise_yfinance_frame(yfinance_payload(tz=tz), "SPY")
    assert out.index.tz is None
    assert (out.index.normalize() == out.index).all()


def test_a_response_missing_a_column_names_what_is_missing():
    frame = yfinance_payload().drop(columns=["Volume"])
    with pytest.raises(DataFetchError, match="volume"):
        _normalise_yfinance_frame(frame, "SPY")


def test_normalised_output_passes_validation():
    """The two halves of the data layer must agree on the shape."""
    out = _normalise_yfinance_frame(yfinance_payload(multiindex=True, n=30), "SPY")
    assert validate_bars(out, "SPY") == []


def test_rows_are_sorted_even_if_the_response_is_not():
    frame = yfinance_payload().iloc[::-1]
    out = _normalise_yfinance_frame(frame, "SPY")
    assert out.index.is_monotonic_increasing


# --- real calendars: holidays and late listings -----------------------------


def holiday_calendar_bars(n=260, drop=(20, 21, 55)) -> pd.DataFrame:
    """Business days with a few removed, the way a holiday calendar looks."""
    bars = make_bars([100.0 + i * 0.1 for i in range(n)], start="2020-01-01")
    return bars.drop(bars.index[list(drop)])


def test_holiday_gaps_do_not_break_validation():
    bars = holiday_calendar_bars()
    warnings = validate_bars(bars, "SPY", max_gap_business_days=5)
    assert warnings == [], "single-day holidays should not even warn"


def test_a_long_closure_warns_but_does_not_fail():
    """Markets do close for a week (9/11, Sandy). That is a warning, not corruption."""
    bars = make_bars([100.0] * 60, start="2020-01-01")
    gapped = bars.drop(bars.index[20:32])
    warnings = validate_bars(gapped, "SPY", max_gap_business_days=5)
    assert any("business-day gap" in w for w in warnings)


class LateListingSource:
    """A source where one symbol simply did not exist yet."""

    name = "fake"

    def __init__(self):
        self.calls = []

    def fetch(self, symbol, start, end=None):
        self.calls.append(symbol)
        if symbol == "NEW":
            return make_bars([50.0 + i * 0.2 for i in range(120)], start="2020-07-01")
        return make_bars([100.0 + i * 0.1 for i in range(260)], start="2020-01-01")


def test_a_symbol_listed_after_the_start_date_does_not_break_the_run(tmp_path):
    """XLE-style case: the universe does not share one listing date."""
    source = LateListingSource()
    bars = load_bars(["OLD", "NEW"], source, tmp_path,
                     start="2020-01-01", end="2020-12-01")

    assert len(bars["NEW"]) < len(bars["OLD"])
    assert bars["NEW"].index[0] > bars["OLD"].index[0]


def test_the_calendar_is_the_union_so_a_late_listing_drops_no_days(tmp_path):
    source = LateListingSource()
    bars = load_bars(["OLD", "NEW"], source, tmp_path,
                     start="2020-01-01", end="2020-12-01")
    calendar = build_calendar(bars)

    assert len(calendar) == len(bars["OLD"])
    assert calendar.is_monotonic_increasing
    assert not calendar.has_duplicates


def test_the_engine_runs_across_a_late_listing_without_trading_the_void(tmp_path):
    """A symbol with no bar yet must never be entered, and must not crash marking."""
    from backtest.pipeline import run_strategy
    from config.loader import load_settings

    source = LateListingSource()
    bars = load_bars(["OLD", "NEW"], source, tmp_path,
                     start="2020-01-01", end="2020-12-01")
    settings = load_settings(sma_period=20)
    result = run_strategy(bars, settings)

    assert len(result.equity) == len(build_calendar(bars))
    assert result.equity.notna().all()

    first_new_bar = bars["NEW"].index[0].strftime("%Y-%m-%d")
    for record in result.decisions:
        if record["date"] < first_new_bar:
            assert record["marks"]["NEW"]["close"] is None
            assert record["marks"]["NEW"]["entry_signal"] is False
            assert all(p["symbol"] != "NEW" for p in record["positions"])


def test_bars_with_a_ragged_calendar_still_produce_a_valid_decision_log(tmp_path):
    """Marks for an absent symbol must be null, not NaN — the log must stay JSON."""
    import json

    from backtest.pipeline import run_strategy
    from config.loader import load_settings

    source = LateListingSource()
    bars = load_bars(["OLD", "NEW"], source, tmp_path,
                     start="2020-01-01", end="2020-12-01")
    result = run_strategy(bars, load_settings(sma_period=20))

    text = json.dumps(result.decisions)
    assert "NaN" not in text
    json.loads(text)
