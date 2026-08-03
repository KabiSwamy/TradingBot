"""Pure signal functions.

CLAUDE.md rule 9: DataFrame in, signals out. No I/O, no API calls, no
randomness, no state. This package imports nothing from backtest/ or execution/
so that the identical function runs in both, which is what makes paper trading
results comparable to backtest results.
"""
