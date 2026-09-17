"""Phase 2 parameter sweep — `python -m backtest.sweep`.

Runs the grid from CLAUDE.md's Phase 2 gate over the TRAIN window and records
every combination, so rule 3 (plateau, not peak) can be argued from data rather
than asserted.

The one structural decision in this module: it **refuses** to run on a window
that reaches into the out-of-sample period. Sweeping the test window is exactly
the cherry-picking rule 4 exists to stop, and a warning is something you scroll
past at 1am. Out-of-sample is evaluated once, by `backtest.run`, after the
parameters are already chosen.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.experiments import append_row, build_row, window_label
from backtest.metrics import compute_metrics
from backtest.pipeline import run_strategy
from config.loader import ConfigError, Settings
from config.loader import load_settings
from data.cache import load_bars
from data.sources import get_source

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Exactly the grid named in the Phase 2 gate.
DEFAULT_GRID: dict[str, tuple] = {
    "entry_rsi": (5.0, 8.0, 10.0, 12.0, 15.0),
    "exit_rsi": (60.0, 65.0, 70.0, 75.0, 80.0),
    "time_stop_days": (5, 7, 10),
}

AXES = ("entry_rsi", "exit_rsi", "time_stop_days")


class OutOfSampleError(RuntimeError):
    """Raised when a sweep would touch the out-of-sample window (rule 4)."""


def sweep(
    bars: dict[str, pd.DataFrame],
    settings: Settings,
    *,
    start: str,
    end: str,
    grid: dict[str, tuple] | None = None,
    ledger_path: Path | None = None,
    source_name: str = "synthetic",
    progress: bool = False,
) -> pd.DataFrame:
    """Run every grid combination on one window and return a table of results.

    Every combination is appended to the experiments ledger (rule 4) using the
    same `build_row` the CLI uses, so sweep rows and single-run rows are directly
    comparable.

    Raises OutOfSampleError if `end` reaches past the train window.
    """
    grid = grid or DEFAULT_GRID
    _guard_window(end, settings)

    combos = list(itertools.product(*(grid[axis] for axis in AXES)))
    rows: list[dict] = []

    for n, values in enumerate(combos, start=1):
        overrides = dict(zip(AXES, values))
        try:
            combo_settings = settings.with_overrides(**overrides)
        except ConfigError as exc:
            # A caller-supplied grid can contain nonsense (entry above exit).
            # Skip the cell and say so rather than aborting 74 valid runs.
            rows.append({**overrides, "sharpe": float("nan"), "error": str(exc)})
            continue

        result = run_strategy(bars, combo_settings)
        metrics = compute_metrics(
            result.equity,
            label="strategy",
            trades=result.trades,
            market_value=result.market_value,
            positions_count=result.positions_count,
        )

        rows.append(
            {
                **overrides,
                "sharpe": metrics.sharpe,
                "cagr": metrics.cagr,
                "max_drawdown": metrics.max_drawdown,
                "calmar": metrics.calmar,
                "trade_count": metrics.trade_count,
                "win_rate": metrics.win_rate,
                "exposure": metrics.exposure,
                "final_equity": metrics.final_equity,
                "kill_switch": result.kill_switch.triggered,
                "config_hash": combo_settings.fingerprint()[0],
                "error": "",
            }
        )

        if ledger_path is not None:
            run_id = (
                f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
                f"-sweep{n:03d}-{combo_settings.fingerprint()[0][:8]}"
            )
            append_row(
                build_row(
                    run_id, combo_settings, source_name,
                    window_label(start, end, settings),
                    start, end, metrics, None, result,
                ),
                ledger_path,
            )

        if progress:
            print(
                f"  [{n:>3}/{len(combos)}] entry {values[0]:>5} exit {values[1]:>5} "
                f"stop {values[2]:>3}  sharpe {metrics.sharpe:>7.3f}  "
                f"trades {metrics.trade_count:>4}",
                file=sys.stderr,
            )

    return pd.DataFrame(rows)


def _guard_window(end: str, settings: Settings) -> None:
    if end > settings.train_end:
        raise OutOfSampleError(
            f"sweep window ends {end}, past the train window "
            f"({settings.train_end}). CLAUDE.md rule 4: the out-of-sample window "
            "is touched only for final evaluation, never for parameter search. "
            "Sweep on the train window, choose parameters, then evaluate "
            "out-of-sample exactly once with backtest.run."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.sweep",
        description="Phase 2 parameter sweep over the train window.",
    )
    parser.add_argument("--start", default=None, help="defaults to settings train_start")
    parser.add_argument("--end", default=None, help="defaults to settings train_end")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic"])
    parser.add_argument("--seed", type=int, default=20100101)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-log", action="store_true", help="skip experiments.csv rows")
    parser.add_argument("--no-plot", action="store_true", help="skip the heatmap")
    parser.add_argument("--metric", default="sharpe", help="metric to analyse for plateaus")
    return parser


def main(argv: list[str] | None = None) -> int:
    from backtest.plateau import analyse, render_plateau_report
    from backtest.report import save_sweep_heatmap

    args = build_parser().parse_args(argv)
    settings = load_settings()
    start = args.start or settings.train_start
    end = args.end or settings.train_end

    try:
        _guard_window(end, settings)
    except OutOfSampleError as exc:
        print(f"\nREFUSED: {exc}\n", file=sys.stderr)
        return 2

    source = get_source(
        args.source, **({"seed": args.seed} if args.source == "synthetic" else {})
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else REPO_ROOT / settings.cache_dir
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {len(settings.universe)} symbols from {source.name}...", file=sys.stderr)
    bars = load_bars(
        settings.universe, source, cache_dir,
        start=start, end=end,
        max_gap_business_days=settings.max_gap_business_days,
    )

    combos = len(list(itertools.product(*(DEFAULT_GRID[a] for a in AXES))))
    print(f"sweeping {combos} combinations on {start}..{end}", file=sys.stderr)

    results = sweep(
        bars, settings,
        start=start, end=end,
        ledger_path=None if args.no_log else out_dir / "experiments.csv",
        source_name=source.name,
        progress=True,
    )

    table_path = out_dir / f"sweep_{source.name}_{start}_{end}.csv"
    results.to_csv(table_path, index=False)

    report = analyse(results, metric=args.metric)
    print(render_plateau_report(report, metric=args.metric, data_source=source.name))

    if not args.no_plot:
        plot_path = out_dir / f"sweep_heatmap_{source.name}_{start}_{end}.png"
        save_sweep_heatmap(
            results, plot_path, metric=args.metric,
            report=report, data_source=source.name,
            title=f"RSI(2) parameter sweep  {start} to {end}",
        )
        print(f"  heatmap -> {plot_path}")

    print(f"  sweep table -> {table_path}")
    if not args.no_log:
        print(f"  {len(results)} runs logged -> {out_dir / 'experiments.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
