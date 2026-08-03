"""results/experiments.csv — the research ledger.

Rule 4: log EVERY backtest run so we cannot quietly cherry-pick. The value of
this file is entirely in its completeness; a run that is not logged is a run that
can be forgotten when it is inconvenient, which is the exact failure the rule
exists to prevent.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

FIELDS = (
    "schema_version",
    "run_id",
    "timestamp_utc",
    "git_commit",
    "data_source",
    "window_label",
    "window_start",
    "window_end",
    "trading_days",
    "symbols",
    "initial_capital",
    "rsi_period",
    "entry_rsi",
    "exit_rsi",
    "sma_period",
    "cost_per_side",
    "max_positions",
    "time_stop_days",
    "kill_switch_drawdown",
    "final_equity",
    "total_return",
    "cagr",
    "sharpe",
    "volatility",
    "max_drawdown",
    "calmar",
    "trade_count",
    "win_rate",
    "avg_win",
    "avg_loss",
    "win_loss_ratio",
    "profit_factor",
    "avg_holding_days",
    "exposure",
    "time_in_market",
    "kill_switch_triggered",
    "kill_switch_date",
    "sharpe_suspect",
    "benchmark_total_return",
    "benchmark_cagr",
    "benchmark_sharpe",
    "benchmark_max_drawdown",
    "config_hash",
    "config_json",
)

SCHEMA_VERSION = 1


def git_commit() -> str:
    """Short SHA of the code that produced a run, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def window_label(start: str, end: str, settings) -> str:
    """Classify the window against the train/test split of rule 4."""
    if start >= settings.test_start:
        return "test"
    if end <= settings.train_end:
        return "train"
    return "full"


def append_row(row: dict, path: Path) -> Path:
    """Append one run to the ledger, writing the header only when new.

    A header that does not match FIELDS raises rather than being written around.
    Silently appending misaligned columns would corrupt the research record in a
    way that is nearly impossible to notice later — every row after the change
    would be subtly wrong, and rules 3 and 4 both read this file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    unknown = set(row) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown experiment fields: {sorted(unknown)}")

    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="") as handle:
            existing = next(csv.reader(handle), [])
        if tuple(existing) != FIELDS:
            raise ValueError(
                f"{path} has a different schema than this version of the code "
                f"writes. Migrate the file or bump SCHEMA_VERSION — do not let "
                f"misaligned columns into the research record."
            )

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})
    return path
