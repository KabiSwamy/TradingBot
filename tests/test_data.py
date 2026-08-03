"""Data-layer tests: validation, the parquet cache, and the synthetic source.

Everything here is offline. The yfinance path is exercised through a fake
downloader — the autouse `no_network` fixture in conftest.py makes a real fetch
impossible, which is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.cache import (
    build_calendar,
    load_bars,
    read_cached,
    read_meta,
    refresh_symbol,
    write_cached,
)
from data.sources import SyntheticSource, get_source
from data.validate import DataValidationError, validate_bars
from tests.conftest import make_bars


# --- validation ------------------------------------------------------------


def test_clean_bars_validate_without_warnings():
    assert validate_bars(make_bars([100.0, 101.0, 102.0]), "TEST") == []


def test_nan_prices_are_fatal():
    bars = make_bars([100.0, 101.0, 102.0])
    bars.iloc[1, bars.columns.get_loc("close")] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        validate_bars(bars, "TEST")


@pytest.mark.parametrize("bad_price", [0.0, -5.0])
def test_non_positive_prices_are_fatal(bad_price):
    bars = make_bars([100.0, 101.0, 102.0])
    bars.iloc[1, bars.columns.get_loc("low")] = bad_price
    with pytest.raises(DataValidationError, match="non-positive"):
        validate_bars(bars, "TEST")


def test_high_below_close_is_fatal():
    """The classic symptom of a broken split adjustment."""
    bars = make_bars([100.0, 101.0, 102.0])
    bars.iloc[1, bars.columns.get_loc("high")] = 50.0
    with pytest.raises(DataValidationError, match="high below"):
        validate_bars(bars, "TEST")


def test_low_above_open_is_fatal():
    bars = make_bars([100.0, 101.0, 102.0])
    bars.iloc[1, bars.columns.get_loc("low")] = 500.0
    with pytest.raises(DataValidationError, match="low above"):
        validate_bars(bars, "TEST")


def test_duplicate_dates_are_fatal():
    bars = make_bars([100.0, 101.0, 102.0])
    bars = pd.concat([bars, bars.iloc[[1]]]).sort_index()
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_bars(bars, "TEST")


def test_unsorted_index_is_fatal():
    bars = make_bars([100.0, 101.0, 102.0]).iloc[::-1]
    with pytest.raises(DataValidationError, match="not sorted"):
        validate_bars(bars, "TEST")


def test_missing_column_is_fatal():
    with pytest.raises(DataValidationError, match="missing column"):
        validate_bars(make_bars([100.0, 101.0]).drop(columns=["volume"]), "TEST")


def test_empty_frame_is_fatal():
    with pytest.raises(DataValidationError, match="no bars"):
        validate_bars(make_bars([100.0]).iloc[:0], "TEST")


def test_long_gap_is_a_warning_not_an_error():
    """Markets close for holidays, so a gap is suspicious, not proof of corruption."""
    bars = make_bars([100.0] * 10)
    gapped = pd.concat([bars.iloc[:5], bars.iloc[5:].set_axis(
        pd.DatetimeIndex([d + pd.Timedelta(days=40) for d in bars.index[5:]])
    )])
    warnings = validate_bars(gapped, "TEST", max_gap_business_days=5)
    assert any("business-day gap" in w for w in warnings)


# --- synthetic source ------------------------------------------------------


def test_synthetic_source_is_deterministic_per_symbol():
    a = SyntheticSource(seed=42).fetch("SPY", "2015-01-01", "2016-01-01")
    b = SyntheticSource(seed=42).fetch("SPY", "2015-01-01", "2016-01-01")
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_symbols_are_independent_of_each_other():
    """SPY's bars must not change depending on which symbols were requested."""
    spy = SyntheticSource(seed=42).fetch("SPY", "2015-01-01", "2016-01-01")
    SyntheticSource(seed=42).fetch("QQQ", "2015-01-01", "2016-01-01")
    spy_again = SyntheticSource(seed=42).fetch("SPY", "2015-01-01", "2016-01-01")
    pd.testing.assert_frame_equal(spy, spy_again)


def test_different_symbols_get_different_paths():
    spy = SyntheticSource(seed=42).fetch("SPY", "2015-01-01", "2016-01-01")
    qqq = SyntheticSource(seed=42).fetch("QQQ", "2015-01-01", "2016-01-01")
    assert not np.allclose(spy["close"].to_numpy(), qqq["close"].to_numpy())


def test_synthetic_bars_pass_validation():
    bars = SyntheticSource(seed=7).fetch("SPY", "2010-01-01", "2020-01-01")
    validate_bars(bars, "SPY")  # must not raise


def test_synthetic_bars_actually_trigger_rsi2_entries():
    """A source that never produces a signal would make every engine test vacuous."""
    from strategies.rsi2_mean_reversion import generate_signals

    bars = SyntheticSource(seed=7).fetch("SPY", "2010-01-01", "2020-01-01")
    signals = generate_signals(bars, rsi_period=2, sma_period=200,
                               entry_rsi=10.0, exit_rsi=70.0)
    assert signals["entry_signal"].sum() > 20
    assert signals["exit_signal"].sum() > 20


def test_get_source_resolves_names():
    assert get_source("synthetic").name == "synthetic"
    assert get_source("yfinance").name == "yfinance"
    with pytest.raises(ValueError, match="unknown data source"):
        get_source("bloomberg")


# --- cache -----------------------------------------------------------------


def test_parquet_round_trip_preserves_values_and_index(tmp_path):
    bars = make_bars([100.0, 101.5, 99.25])
    write_cached("TEST", bars, tmp_path)
    back = read_cached("TEST", tmp_path)
    pd.testing.assert_frame_equal(bars, back)


def test_read_cached_returns_none_for_unknown_symbol(tmp_path):
    assert read_cached("NOPE", tmp_path) is None


class FakeSource:
    """A downloader we control, standing in for yfinance."""

    name = "fake"

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list[tuple] = []

    def fetch(self, symbol, start, end=None):
        self.calls.append((symbol, start, end))
        return self.frame.copy()


def test_refresh_writes_cache_and_meta(tmp_path):
    source = FakeSource(make_bars([100.0, 101.0, 102.0]))
    result = refresh_symbol("TEST", source, tmp_path)

    assert result.rows == 3
    assert read_cached("TEST", tmp_path) is not None
    meta = read_meta(tmp_path)
    assert meta["TEST"]["rows"] == 3
    assert meta["TEST"]["auto_adjust"] is True


def test_refresh_replaces_rather_than_appends(tmp_path):
    """Appending would splice two price scales together — see refresh_symbol."""
    source = FakeSource(make_bars([100.0, 101.0, 102.0]))
    refresh_symbol("TEST", source, tmp_path)

    source.frame = make_bars([100.0, 101.0, 102.0, 103.0])
    refresh_symbol("TEST", source, tmp_path)

    cached = read_cached("TEST", tmp_path)
    assert len(cached) == 4  # replaced, not 3 + 4 concatenated


def test_refresh_detects_retroactive_adjustment(tmp_path):
    """A dividend rescales all history; the refresh must notice and report it."""
    source = FakeSource(make_bars([100.0, 101.0, 102.0]))
    first = refresh_symbol("TEST", source, tmp_path)
    assert first.adjusted_rows == 0
    assert not first.readjusted

    # Same dates, every historical price rescaled by ~0.5% — what a dividend does.
    source.frame = make_bars([99.5, 100.495, 101.49])
    second = refresh_symbol("TEST", source, tmp_path)

    assert second.readjusted
    assert second.adjusted_rows == 3
    assert second.max_adjustment == pytest.approx(0.005, rel=0.05)


def test_load_bars_fetches_when_cache_is_cold(tmp_path):
    source = FakeSource(make_bars([100.0] * 10, start="2020-01-01"))
    bars = load_bars(["TEST"], source, tmp_path, start="2020-01-01")
    assert len(bars["TEST"]) == 10
    assert source.calls, "expected a fetch on a cold cache"


def test_load_bars_uses_the_cache_on_the_second_call(tmp_path):
    source = FakeSource(make_bars([100.0] * 10, start="2020-01-01"))
    load_bars(["TEST"], source, tmp_path, start="2020-01-01")
    calls_after_first = len(source.calls)
    load_bars(["TEST"], source, tmp_path, start="2020-01-01")
    assert len(source.calls) == calls_after_first, "second load should hit the cache"


def test_load_bars_slices_to_the_window_but_caches_full_history(tmp_path):
    """Rule: fetch everything, slice afterwards.

    Wilder's RSI is recursive, so a series that starts later produces different
    early values. Caching only the requested window would make results depend on
    the start date in a way nothing in the report would reveal.
    """
    source = FakeSource(make_bars([100.0 + i for i in range(60)], start="2020-01-01"))
    bars = load_bars(["TEST"], source, tmp_path, start="2020-02-14")

    assert len(bars["TEST"]) < 60                      # window is sliced
    assert len(read_cached("TEST", tmp_path)) == 60    # cache keeps everything


def test_a_cache_that_does_not_reach_the_window_end_is_refreshed(tmp_path):
    """Regression: a stale cache silently truncated the backtest.

    Loading 2020-01-01..2020-01-10 then asking for a later window used to reuse
    the cache and return too few bars — or none — without any error. A run that
    quietly covers less history than requested looks exactly like a correct one.
    """
    source = FakeSource(make_bars([100.0] * 10, start="2020-01-01"))
    load_bars(["TEST"], source, tmp_path, start="2020-01-01", end="2020-01-10")
    calls_after_first = len(source.calls)

    source.frame = make_bars([100.0] * 200, start="2020-01-01")
    bars = load_bars(["TEST"], source, tmp_path, start="2020-06-01", end="2020-09-01")

    assert len(source.calls) > calls_after_first, "stale cache was reused"
    assert not bars["TEST"].empty


def test_a_cache_a_few_days_behind_is_still_considered_fresh(tmp_path):
    """Weekends and holidays mean the newest bar legitimately lags the end date."""
    source = FakeSource(make_bars([100.0] * 40, start="2020-01-01"))
    load_bars(["TEST"], source, tmp_path, start="2020-01-01", end="2020-02-25")
    calls_after_first = len(source.calls)

    last_cached = read_cached("TEST", tmp_path).index[-1]
    load_bars(
        ["TEST"], source, tmp_path,
        start="2020-01-01",
        end=str((last_cached + pd.Timedelta(days=2)).date()),
    )
    assert len(source.calls) == calls_after_first, "refetched over a weekend-sized gap"


def test_build_calendar_is_the_union_of_all_symbols(tmp_path):
    """A symbol with a shorter history must not drop days from the run."""
    long_bars = make_bars([100.0] * 10, start="2020-01-01")
    short_bars = make_bars([100.0] * 3, start="2020-01-06")
    calendar = build_calendar({"LONG": long_bars, "SHORT": short_bars})

    assert len(calendar) == 10
    assert calendar.is_monotonic_increasing
    assert not calendar.has_duplicates
