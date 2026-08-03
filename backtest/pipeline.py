"""Wiring: strategy + engine.

This is the only module that imports both `strategies` and `backtest.engine`.
Keeping the join in one place is what preserves the architecture's dependency
rule — the engine never imports the strategy, and in Phase 3 `jobs/nightly.py`
will perform the same join against `execution/`, so the identical strategy
function drives both. That symmetry is the whole reason paper results will be
comparable to backtest results (rule 9).
"""

from __future__ import annotations

import pandas as pd

from backtest.engine import EngineResult, run_backtest
from config.loader import Settings
from strategies.rsi2_mean_reversion import generate_signals


def compute_signals(
    bars: dict[str, pd.DataFrame], settings: Settings
) -> dict[str, pd.DataFrame]:
    """Run the pure strategy function over every symbol."""
    return {
        symbol: generate_signals(
            frame,
            rsi_period=settings.strategy.rsi_period,
            sma_period=settings.strategy.sma_period,
            entry_rsi=settings.strategy.entry_rsi,
            exit_rsi=settings.strategy.exit_rsi,
        )
        for symbol, frame in bars.items()
    }


def run_strategy(
    bars: dict[str, pd.DataFrame],
    settings: Settings,
    *,
    force_close_at_end: bool = True,
) -> EngineResult:
    """Compute signals from `bars` and run the backtest over them."""
    return run_backtest(
        bars,
        compute_signals(bars, settings),
        settings,
        force_close_at_end=force_close_at_end,
    )
