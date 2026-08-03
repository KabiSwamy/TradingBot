"""Backtest engine, cost model, metrics and reporting.

This package never imports from strategies/ — run.py wires the two together,
exactly as jobs/nightly.py will in Phase 3, so both call the identical strategy
function (rule 9).
"""
