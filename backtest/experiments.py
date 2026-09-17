"""results/experiments.csv — the research ledger.

Rule 4: log EVERY backtest run so we cannot quietly cherry-pick. The value of
this file is entirely in its completeness; a run that is not logged is a run that
can be forgotten when it is inconvenient, which is the exact failure the rule
exists to prevent.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
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


def prior_oos_runs(path: Path) -> list[dict]:
    """Ledger rows that already touched the out-of-sample window.

    Rule 4 says the test window is "touched only for final evaluation". Nothing
    can stop a second look outright — the data is on disk — but "we only looked
    once" should be a fact anyone can check against the ledger rather than a
    claim taken on trust. This is what makes it checkable.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if r.get("window_label") in {"test", "full"}]


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


def build_row(
    run_id, settings, source_name, label, start, end, strategy, benchmark, result
) -> dict:
    """Assemble one ledger row from a completed run.

    Lives here rather than in run.py so the CLI and the Phase 2 sweep write
    identical rows. Two writers with their own idea of the schema is how a
    research ledger quietly becomes uncomparable across runs.
    """
    config_hash, config_json = settings.fingerprint()
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        "git_commit": git_commit(),
        "data_source": source_name,
        "window_label": label,
        "window_start": start,
        "window_end": end,
        "trading_days": strategy.trading_days,
        "symbols": len(settings.universe),
        "initial_capital": settings.initial_capital,
        "rsi_period": settings.strategy.rsi_period,
        "entry_rsi": settings.strategy.entry_rsi,
        "exit_rsi": settings.strategy.exit_rsi,
        "sma_period": settings.strategy.sma_period,
        "cost_per_side": settings.cost_per_side,
        "max_positions": settings.risk.max_positions,
        "time_stop_days": settings.risk.time_stop_days,
        "kill_switch_drawdown": settings.risk.kill_switch_drawdown,
        "final_equity": round(strategy.final_equity, 2),
        "total_return": round(strategy.total_return, 6),
        "cagr": round(strategy.cagr, 6),
        "sharpe": round(strategy.sharpe, 4),
        "volatility": round(strategy.volatility, 6),
        "max_drawdown": round(strategy.max_drawdown, 6),
        "calmar": round(strategy.calmar, 4),
        "trade_count": strategy.trade_count,
        "win_rate": round(strategy.win_rate, 6),
        "avg_win": round(strategy.avg_win, 6),
        "avg_loss": round(strategy.avg_loss, 6),
        "win_loss_ratio": round(strategy.win_loss_ratio, 4),
        "profit_factor": round(strategy.profit_factor, 4),
        "avg_holding_days": round(strategy.avg_holding_days, 2),
        "exposure": round(strategy.exposure, 6),
        "time_in_market": round(strategy.time_in_market, 6),
        "kill_switch_triggered": result.kill_switch.triggered,
        "kill_switch_date": (
            result.kill_switch.trigger_date.strftime("%Y-%m-%d")
            if result.kill_switch.trigger_date is not None
            else ""
        ),
        "sharpe_suspect": strategy.sharpe > 2.5,
        "config_hash": config_hash,
        "config_json": config_json,
    }
    if benchmark is not None:
        row.update(
            {
                "benchmark_total_return": round(benchmark.total_return, 6),
                "benchmark_cagr": round(benchmark.cagr, 6),
                "benchmark_sharpe": round(benchmark.sharpe, 4),
                "benchmark_max_drawdown": round(benchmark.max_drawdown, 6),
            }
        )
    return row
