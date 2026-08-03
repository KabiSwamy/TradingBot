"""Shared fixtures and test-suite guardrails.

Two guardrails are autouse because forgetting either one is silent:

`no_network` makes any outbound socket a hard failure. Without it a test that
accidentally reaches yfinance would pass on a connected laptop and fail in CI —
or worse, pass in both places while quietly depending on live market data, which
would make the suite non-reproducible.

`no_results_pollution` fails any test that writes into the real results/
directory. results/experiments.csv is the research ledger that rules 3 and 4
depend on; a test appending junk rows to it corrupts the record silently.
"""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_RESULTS_DIR = REPO_ROOT / "results"

_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any non-loopback socket connection is a test failure."""

    def guard(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host in _ALLOWED_HOSTS:
            return _real_create_connection(address, *args, **kwargs)
        raise RuntimeError(
            f"network access is not allowed in tests (attempted {host!r}). "
            "Use the synthetic source or a fake downloader."
        )

    _real_create_connection = socket.create_connection
    monkeypatch.setattr(socket, "create_connection", guard)

    real_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in _ALLOWED_HOSTS:
            raise RuntimeError(
                f"network access is not allowed in tests (attempted {host!r})"
            )
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(autouse=True)
def no_results_pollution():
    """Fail if a test created or modified anything under the real results/."""
    before = _results_snapshot()
    yield
    after = _results_snapshot()
    if before != after:
        added = sorted(set(after) - set(before))
        raise AssertionError(
            "a test wrote into the real results/ directory "
            f"(changed: {added or 'existing files modified'}). "
            "Tests must write to tmp_path — results/experiments.csv is the "
            "research ledger."
        )


def _results_snapshot() -> dict[str, float]:
    if not REAL_RESULTS_DIR.exists():
        return {}
    return {
        str(p.relative_to(REAL_RESULTS_DIR)): p.stat().st_mtime
        for p in REAL_RESULTS_DIR.rglob("*")
        if p.is_file()
    }


# --- fixtures for building exact, hand-verifiable test data -----------------


def make_bars(
    closes: list[float],
    *,
    opens: list[float] | None = None,
    start: str = "2020-01-01",
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Build an OHLCV frame from explicit closes (and optionally opens).

    When `opens` is omitted, each bar opens at the previous close, so the series
    is perfectly continuous and any gap in a test is one that test put there
    deliberately. Highs and lows are derived to bracket open and close so the
    frame always passes validation.
    """
    # freq is stripped so fixtures have the same index shape as real market data
    # (a real calendar has holidays and carries no regular frequency). Leaving it
    # set would let a test pass on an index property that production never has.
    index = pd.DatetimeIndex(pd.bdate_range(start, periods=len(closes)), name="date")
    index.freq = None
    close = np.asarray(closes, dtype="float64")
    if opens is None:
        open_ = np.concatenate([[close[0]], close[:-1]])
    else:
        open_ = np.asarray(opens, dtype="float64")

    top = np.maximum(open_, close)
    bottom = np.minimum(open_, close)
    return pd.DataFrame(
        {
            "open": open_,
            "high": top * 1.001,
            "low": bottom * 0.999,
            "close": close,
            "volume": float(volume),
        },
        index=index,
    )


def make_signals(
    index: pd.DatetimeIndex,
    *,
    entries: tuple[int, ...] = (),
    exits: tuple[int, ...] = (),
    rank_key: float | list[float] = 5.0,
) -> pd.DataFrame:
    """Build a signal frame from explicit integer bar positions.

    Injecting signals is what makes the engine tests exactly hand-verifiable.
    Deriving them from real prices instead would mean every engine test had to
    construct a path that simultaneously satisfies `close > SMA(200)` and
    `RSI(2) < 10` — conditions that pull against each other, since a dip deep
    enough to crush RSI(2) tends to drag price under a short SMA too.
    """
    n = len(index)
    entry_signal = np.zeros(n, dtype=bool)
    exit_signal = np.zeros(n, dtype=bool)
    entry_signal[list(entries)] = True
    exit_signal[list(exits)] = True

    rsi = (
        np.full(n, float(rank_key))
        if np.isscalar(rank_key)
        else np.asarray(rank_key, dtype="float64")
    )
    return pd.DataFrame(
        {
            "rsi": rsi,
            "sma": np.zeros(n),
            "regime_ok": np.ones(n, dtype=bool),
            "entry_signal": entry_signal,
            "exit_signal": exit_signal,
        },
        index=index,
    )


@pytest.fixture
def bars_factory():
    return make_bars


@pytest.fixture
def signals_factory():
    return make_signals
