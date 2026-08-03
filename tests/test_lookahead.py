"""Look-ahead tests — CLAUDE.md Testing family 1, rule 1.

The most important tests in the project. Rule 1 says a decision dated day t may
use only data through day t's close; these prove it by changing the future and
checking the past did not move.

Unlike the other engine tests these run the REAL strategy end to end, because
the mutation has to propagate through indicator computation to be meaningful.

Two variants, catching different bug classes:

* Mutation — multiply everything after day t. Catches anything that reads
  forward: `shift(-1)`, a fill priced at the wrong bar, a centred rolling window.
* Truncation — run on a prefix instead. Catches anything that depends on the
  *extent* of the series rather than a specific future bar: a whole-series
  `.mean()` or `.max()`, min-max normalisation, `bfill`, `rolling(center=True)`.

A mutation test alone would pass on a series-wide `.mean()` if the mutation
happened to leave the mean unchanged, and a truncation test alone would miss a
one-bar forward read. Both are cheap; both are here.
"""

from __future__ import annotations

import json

import pytest

from backtest.pipeline import run_strategy
from config.loader import load_settings
from data.sources import SyntheticSource

# A short SMA so trades actually occur inside a few hundred bars — with
# sma_period=200 a 400-bar window would spend half its life warming up.
SETTINGS = load_settings(sma_period=20, entry_rsi=10.0, exit_rsi=70.0)

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]


def build_bars(seed: int = 11):
    source = SyntheticSource(seed=seed)
    return {s: source.fetch(s, "2015-01-01", "2016-12-31") for s in SYMBOLS}


def serialise(decisions: list[dict]) -> list[str]:
    return [json.dumps(record, sort_keys=True) for record in decisions]


def mutate_after(bars: dict, cut: int) -> dict:
    """Rewrite every bar strictly after index `cut`, leaving dates untouched."""
    factors = {"AAA": 3.0, "BBB": 0.2, "CCC": 5.0, "DDD": 0.5}
    mutated = {}
    for symbol, frame in bars.items():
        copy = frame.copy()
        tail = copy.index[cut + 1:]
        for column in ("open", "high", "low", "close"):
            copy.loc[tail, column] = copy.loc[tail, column] * factors[symbol]
        copy.loc[tail, "volume"] = 0.0
        mutated[symbol] = copy
    return mutated


@pytest.fixture(scope="module")
def baseline():
    bars = build_bars()
    result = run_strategy(bars, SETTINGS, force_close_at_end=False)
    return bars, result


def test_the_fixture_actually_trades(baseline):
    """Guard against every assertion below passing vacuously on an empty run."""
    _, result = baseline
    assert len(result.trades) > 10, "fixture produces too few trades to prove anything"


@pytest.mark.parametrize("cut", [50, 120, 250, 400])
def test_mutating_the_future_leaves_every_earlier_decision_identical(baseline, cut):
    bars, base = baseline
    after = run_strategy(mutate_after(bars, cut), SETTINGS, force_close_at_end=False)

    base_prefix = [r for r in serialise(base.decisions) if json.loads(r)["date"] <= _date(base, cut)]
    after_prefix = [r for r in serialise(after.decisions) if json.loads(r)["date"] <= _date(base, cut)]

    assert base_prefix == after_prefix


@pytest.mark.parametrize("cut", [50, 120, 250, 400])
def test_mutating_the_future_does_change_later_decisions(baseline, cut):
    """Without this the test above passes if the engine ignores its input entirely."""
    bars, base = baseline
    after = run_strategy(mutate_after(bars, cut), SETTINGS, force_close_at_end=False)
    assert serialise(base.decisions) != serialise(after.decisions)


def test_the_last_possible_cut_point(baseline):
    """A mid-series cut can miss an off-by-one right at the t / t+1 boundary."""
    bars, base = baseline
    cut = len(base.calendar) - 2
    after = run_strategy(mutate_after(bars, cut), SETTINGS, force_close_at_end=False)

    boundary = _date(base, cut)
    base_prefix = [r for r in serialise(base.decisions) if json.loads(r)["date"] <= boundary]
    after_prefix = [r for r in serialise(after.decisions) if json.loads(r)["date"] <= boundary]
    assert base_prefix == after_prefix


@pytest.mark.parametrize("cut", [120, 250, 400])
def test_truncating_the_series_reproduces_the_prefix_exactly(baseline, cut):
    """Catches dependence on the length or extent of the data, not just its values."""
    bars, base = baseline
    truncated = {s: f.iloc[: cut + 1] for s, f in bars.items()}
    after = run_strategy(truncated, SETTINGS, force_close_at_end=False)

    assert serialise(after.decisions) == serialise(base.decisions)[: cut + 1]


def test_running_twice_is_bit_for_bit_identical(baseline):
    """Any wall-clock, RNG or set-iteration leak would show up here."""
    bars, base = baseline
    again = run_strategy(bars, SETTINGS, force_close_at_end=False)
    assert serialise(again.decisions) == serialise(base.decisions)


def test_no_fill_precedes_its_own_decision(baseline):
    _, base = baseline
    fills = 0
    for record in base.decisions:
        for fill in record["fills"]:
            assert fill["decided_date"] < record["date"]
            fills += 1
    assert fills > 10


def _date(result, index: int) -> str:
    return result.calendar[index].strftime("%Y-%m-%d")
