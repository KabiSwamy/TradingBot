"""Decision-log tests — CLAUDE.md rule 10.

"Each nightly run appends JSON lines: date, prices used, indicator values,
signals, orders placed, fills, and the reason for every action or deliberate
inaction."

The last clause is the demanding one. Recording why something *happened* is
easy; recording why something did *not* happen means the log has to carry the
state of every symbol that was considered and passed over. That is what the
`marks` block is for, and it is also the substrate Phase 3 will reconcile
against.
"""

from __future__ import annotations

import json

import pytest

from backtest.pipeline import run_strategy
from config.loader import load_settings
from data.sources import SyntheticSource

SETTINGS = load_settings(sma_period=20)
MARK_FIELDS = {"close", "rsi", "sma", "regime_ok", "entry_signal", "exit_signal"}


@pytest.fixture(scope="module")
def result():
    source = SyntheticSource(seed=5)
    bars = {s: source.fetch(s, "2015-01-01", "2016-12-31")
            for s in ["AAA", "BBB", "CCC"]}
    return bars, run_strategy(bars, SETTINGS)


def test_every_day_records_every_symbol(result):
    bars, res = result
    for record in res.decisions:
        assert set(record["marks"]) == set(bars)


def test_each_mark_carries_prices_signals_and_indicators(result):
    _, res = result
    for record in res.decisions[:20]:
        for mark in record["marks"].values():
            assert set(mark) == MARK_FIELDS


def test_the_log_is_strictly_valid_json(result):
    """A bare NaN token is not legal JSON and breaks non-Python readers.

    Warm-up rows genuinely have no RSI; null is the honest encoding.
    """
    _, res = result
    text = "\n".join(json.dumps(r, sort_keys=True) for r in res.decisions)
    assert "NaN" not in text
    assert "Infinity" not in text
    for line in text.splitlines():
        json.loads(line)


def test_warmup_days_record_null_indicators_not_zero(result):
    """Reporting 0.0 for an RSI that does not exist yet would be a lie."""
    _, res = result
    first = res.decisions[0]["marks"]
    assert all(m["rsi"] is None for m in first.values())
    assert all(m["sma"] is None for m in first.values())
    assert all(m["regime_ok"] is False for m in first.values())


def test_every_entry_order_is_justified_by_that_days_marks(result):
    """The log must show the entry condition was actually met."""
    _, res = result
    checked = 0
    for record in res.decisions:
        for order in record["orders"]:
            if order["reason"] != "entry_signal":
                continue
            mark = record["marks"][order["symbol"]]
            assert mark["entry_signal"] is True
            assert mark["regime_ok"] is True
            assert mark["rsi"] < SETTINGS.strategy.entry_rsi
            assert mark["close"] > mark["sma"]
            checked += 1
    assert checked > 5


def test_every_rsi_exit_is_justified_by_that_days_marks(result):
    _, res = result
    checked = 0
    for record in res.decisions:
        for order in record["orders"]:
            if order["reason"] != "rsi_exit":
                continue
            assert record["marks"][order["symbol"]]["rsi"] > SETTINGS.strategy.exit_rsi
            checked += 1
    assert checked > 5


def test_deliberate_inaction_is_explainable_from_the_log(result):
    """Rule 10's hardest clause.

    On a day with no orders, the log must contain enough to show why: every
    symbol either failed the regime filter, failed the RSI threshold, or was
    already held. Without per-symbol marks this is unanswerable.
    """
    _, res = result
    quiet = [
        r for r in res.decisions
        if not r["orders"] and not r["fills"] and r["marks"]["AAA"]["rsi"] is not None
    ]
    assert quiet, "fixture produced no quiet days to check"

    held = {p["symbol"] for p in quiet[0]["positions"]}
    for symbol, mark in quiet[0]["marks"].items():
        explained = (
            symbol in held
            or not mark["regime_ok"]
            or mark["rsi"] >= SETTINGS.strategy.entry_rsi
        )
        assert explained, f"{symbol} was passed over with no reason visible in the log"


def test_skipped_candidates_record_why_they_lost_the_slot(result):
    _, res = result
    skips = [s for r in res.decisions for s in r["skipped"]]
    if skips:
        assert all(s["reason"] == "no_free_slot" for s in skips)
        assert all("rsi" in s for s in skips)


def test_marks_match_what_the_strategy_computed(result):
    """The log must reflect the real indicator values, not a rounded re-derivation."""
    from backtest.pipeline import compute_signals

    bars, res = result
    signals = compute_signals(bars, SETTINGS)

    record = res.decisions[400]
    for symbol, mark in record["marks"].items():
        row = signals[symbol].loc[record["date"]]
        assert mark["rsi"] == pytest.approx(round(row["rsi"], 6))
        assert mark["entry_signal"] == bool(row["entry_signal"])
        assert mark["regime_ok"] == bool(row["regime_ok"])


def test_marks_never_read_the_future(result):
    """Rule 1 applied to the log itself.

    The look-ahead suite compares serialised records byte for byte, so it
    already covers this — but marks are new, and a field that leaked tomorrow's
    close would be the easiest possible way to reintroduce look-ahead.
    """
    bars, res = result
    cut = 300
    mutated = {}
    for symbol, frame in bars.items():
        copy = frame.copy()
        tail = copy.index[cut + 1:]
        for column in ("open", "high", "low", "close"):
            copy.loc[tail, column] = copy.loc[tail, column] * 4.0
        mutated[symbol] = copy

    after = run_strategy(mutated, SETTINGS)
    before_marks = [json.dumps(r["marks"], sort_keys=True) for r in res.decisions[:cut]]
    after_marks = [json.dumps(r["marks"], sort_keys=True) for r in after.decisions[:cut]]
    assert before_marks == after_marks
