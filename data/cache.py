"""Local parquet cache of daily bars.

One file per symbol: data/cache/{SYMBOL}.parquet. Per-symbol files mean one bad
download cannot poison the other nine, and a refresh rewrites only what changed.

The important rule in this module is that **refresh replaces, it never appends**.
See `refresh_symbol` for why — it is not an efficiency choice, it is a
correctness one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.sources import BarSource, OHLCV_COLUMNS
from data.validate import validate_bars

META_FILENAME = "_meta.json"


@dataclass(frozen=True)
class RefreshResult:
    symbol: str
    rows: int
    first_date: str
    last_date: str
    adjusted_rows: int          # historical rows whose values changed since last fetch
    max_adjustment: float       # largest relative change across the overlap
    warnings: list[str]

    @property
    def readjusted(self) -> bool:
        return self.adjusted_rows > 0


def cache_path(symbol: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{symbol.upper()}.parquet"


def read_cached(symbol: str, cache_dir: Path) -> pd.DataFrame | None:
    """Return cached bars, or None if this symbol has never been fetched."""
    path = cache_path(symbol, cache_dir)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    # `date` is stored as a column, not an index, so the file is readable by any
    # parquet reader and does not depend on pandas index metadata surviving.
    if "date" in frame.columns:
        frame = frame.set_index("date")
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = "date"
    return frame[list(OHLCV_COLUMNS)].astype("float64").sort_index()


def write_cached(symbol: str, bars: pd.DataFrame, cache_dir: Path) -> Path:
    path = cache_path(symbol, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bars.copy()
    out.index.name = "date"
    out.reset_index().to_parquet(path, index=False, compression="snappy")
    return path


def read_meta(cache_dir: Path) -> dict:
    path = Path(cache_dir) / META_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_meta(cache_dir: Path, meta: dict) -> None:
    path = Path(cache_dir) / META_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def refresh_symbol(
    symbol: str,
    source: BarSource,
    cache_dir: Path,
    *,
    start: str = "1990-01-01",
    end: str | None = None,
    max_gap_business_days: int = 5,
) -> RefreshResult:
    """Re-fetch a symbol's full history and replace its cache file.

    This deliberately does NOT append to the existing cache, even though
    appending would be faster.

    Adjusted prices are rescaled retroactively by every split and dividend. SPY
    pays quarterly, so roughly four times a year the *entire historical series*
    changes by the adjustment factor. Appending only the new rows to an old file
    therefore splices two different price scales together, and the seam shows up
    as one fabricated overnight return of about the dividend yield — on a day the
    strategy may well have been holding. That is a silent, directional
    corruption of every backtest run afterwards, and nothing in the metrics would
    look wrong.

    So: fetch everything, compare the overlap with what was cached, report how
    much the history moved, and write the new version.
    """
    fresh = source.fetch(symbol, start=start, end=end)
    warnings = validate_bars(
        fresh, symbol, max_gap_business_days=max_gap_business_days
    )

    previous = read_cached(symbol, cache_dir)
    adjusted_rows, max_adjustment = _compare_overlap(previous, fresh)

    write_cached(symbol, fresh, cache_dir)

    meta = read_meta(cache_dir)
    meta[symbol.upper()] = {
        "rows": int(len(fresh)),
        "first_date": str(fresh.index[0].date()),
        "last_date": str(fresh.index[-1].date()),
        "source": source.name,
        # Recorded so a later change of adjustment convention is detectable
        # rather than being absorbed silently into the numbers.
        "auto_adjust": True,
    }
    write_meta(cache_dir, meta)

    return RefreshResult(
        symbol=symbol.upper(),
        rows=len(fresh),
        first_date=str(fresh.index[0].date()),
        last_date=str(fresh.index[-1].date()),
        adjusted_rows=adjusted_rows,
        max_adjustment=max_adjustment,
        warnings=warnings,
    )


def _compare_overlap(
    previous: pd.DataFrame | None, fresh: pd.DataFrame
) -> tuple[int, float]:
    """How much did already-cached history move? (rows_changed, max_rel_change)."""
    if previous is None or previous.empty:
        return 0, 0.0

    shared = previous.index.intersection(fresh.index)
    if len(shared) == 0:
        return 0, 0.0

    old = previous.loc[shared, "close"]
    new = fresh.loc[shared, "close"]
    rel = (new / old - 1.0).abs()
    changed = rel > 1e-6
    return int(changed.sum()), float(rel.max())


def load_bars(
    symbols: list[str],
    source: BarSource,
    cache_dir: Path,
    *,
    start: str,
    end: str | None = None,
    use_cache: bool = True,
    max_gap_business_days: int = 5,
) -> dict[str, pd.DataFrame]:
    """Load bars for each symbol, fetching and caching as needed.

    Returns a dict of symbol -> validated bars covering `start`..`end`.

    Bars are always FETCHED from the earliest available history and only sliced
    to `start` afterwards. Wilder's RSI is recursive, so its value depends on
    where the series began; fetching only the requested window would give a
    different RSI for the first several weeks of every backtest and make results
    depend on the start date in a way that is invisible in the output.
    """
    bars: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = read_cached(symbol, cache_dir) if use_cache else None
        if frame is None or not _covers(frame, end):
            refresh_symbol(
                symbol,
                source,
                cache_dir,
                end=end,
                max_gap_business_days=max_gap_business_days,
            )
            frame = read_cached(symbol, cache_dir)
        assert frame is not None  # just written
        validate_bars(frame, symbol, max_gap_business_days=max_gap_business_days)
        bars[symbol] = _slice_window(frame, start, end)
    return bars


# Weekends, holidays and a not-yet-closed session mean the newest cached bar is
# legitimately a few days behind any given end date.
_STALENESS_TOLERANCE_DAYS = 5


def _covers(bars: pd.DataFrame, end: str | None) -> bool:
    """Does the cache reach `end`, allowing for non-trading days?

    Without this check a cache built for one window is reused for a later one
    and silently yields too few bars — or none at all. The run does not fail, it
    just quietly reports on a shorter history than was asked for, which is the
    kind of wrong answer that looks exactly like a right one.

    Only the end is checked. Bars are always fetched from the earliest available
    history (see `load_bars`), so a cached first date later than the requested
    start means the symbol simply did not exist yet.
    """
    if bars.empty:
        return False
    if end is None:
        return True
    wanted = pd.Timestamp(end) - pd.tseries.offsets.BDay(_STALENESS_TOLERANCE_DAYS)
    return bars.index[-1] >= wanted


def _slice_window(bars: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    window = bars.loc[bars.index >= pd.Timestamp(start)]
    if end is not None:
        window = window.loc[window.index <= pd.Timestamp(end)]
    return window


def build_calendar(bars: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Master trading calendar: the sorted union of every symbol's dates.

    Union rather than the benchmark's calendar so a symbol with a shorter or
    patchier history cannot silently drop days from the run.
    """
    index = pd.DatetimeIndex([])
    for frame in bars.values():
        index = index.union(frame.index)
    return pd.DatetimeIndex(sorted(index), name="date")
