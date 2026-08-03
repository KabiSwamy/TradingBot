"""`python -m backtest.run` — the Phase 1 entrypoint.

Loads bars, computes signals with the pure strategy function, runs the engine,
prints the report, saves the equity curve, and appends the run to
results/experiments.csv.

Every run is logged (rule 4). `--no-log` exists for smoke tests only, and says so
on the console when used, because an unlogged run is a run that can be forgotten
when its result is inconvenient.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.benchmark import buy_and_hold
from backtest.costs import CostModel
from backtest.experiments import SCHEMA_VERSION, append_row, git_commit, window_label
from backtest.metrics import compute_metrics
from backtest.pipeline import run_strategy
from backtest.report import render_report, save_equity_plot
from config.loader import load_settings
from data.cache import build_calendar, load_bars
from data.sources import get_source

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.run",
        description="Run the RSI(2) mean-reversion backtest.",
    )
    parser.add_argument("--start", default="2010-01-01", help="window start (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="window end; defaults to today")
    parser.add_argument(
        "--source",
        default="yfinance",
        choices=["yfinance", "synthetic"],
        help="bar source. 'synthetic' generates deterministic fake bars for "
             "offline verification and is NOT a research result.",
    )
    parser.add_argument("--seed", type=int, default=20100101, help="synthetic source seed")
    parser.add_argument("--refresh", action="store_true", help="re-fetch bars, ignoring the cache")
    parser.add_argument("--no-log", action="store_true", help="skip the experiments.csv row")
    parser.add_argument("--no-plot", action="store_true", help="skip the equity curve image")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write artifacts here instead of results/ (used by tests so they "
             "cannot touch the real research ledger)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="bar cache location; defaults to the path in settings.yaml",
    )
    parser.add_argument("--entry-rsi", type=float, default=None)
    parser.add_argument("--exit-rsi", type=float, default=None)
    parser.add_argument("--time-stop", type=int, default=None)
    parser.add_argument("--capital", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = {}
    if args.entry_rsi is not None:
        overrides["entry_rsi"] = args.entry_rsi
    if args.exit_rsi is not None:
        overrides["exit_rsi"] = args.exit_rsi
    if args.time_stop is not None:
        overrides["time_stop_days"] = args.time_stop
    if args.capital is not None:
        overrides["initial_capital"] = args.capital

    settings = load_settings(**overrides)
    end = args.end or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    source = get_source(args.source, **({"seed": args.seed} if args.source == "synthetic" else {}))
    cache_dir = Path(args.cache_dir) if args.cache_dir else REPO_ROOT / settings.cache_dir

    print(f"loading {len(settings.universe)} symbols from {source.name}...", file=sys.stderr)
    bars = load_bars(
        settings.universe,
        source,
        cache_dir,
        start=args.start,
        end=end,
        use_cache=not args.refresh,
        max_gap_business_days=settings.max_gap_business_days,
    )
    calendar = build_calendar(bars)
    if len(calendar) < 2:
        print("not enough bars to run a backtest", file=sys.stderr)
        return 1

    result = run_strategy(bars, settings)

    strategy_metrics = compute_metrics(
        result.equity,
        label="strategy",
        trades=result.trades,
        market_value=result.market_value,
        positions_count=result.positions_count,
    )

    benchmark_metrics = None
    benchmark_curve = None
    if settings.benchmark in bars:
        benchmark_curve = buy_and_hold(
            bars[settings.benchmark],
            calendar,
            settings.initial_capital,
            CostModel(settings.cost_per_side),
        )
        benchmark_metrics = compute_metrics(benchmark_curve, label="benchmark")

    label = window_label(args.start, end, settings)
    print(
        render_report(
            strategy_metrics,
            benchmark_metrics,
            data_source=source.name,
            window_label=label,
            settings=settings,
            kill_switch=result.kill_switch,
        )
    )

    if label in {"test", "full"}:
        print(
            "\n  NOTE: this window includes out-of-sample data (from "
            f"{settings.test_start}). Rule 4: do not use it to select parameters.\n"
        )

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{settings.fingerprint()[0][:8]}"

    if not args.no_plot:
        plot_path = out_dir / f"equity_{source.name}_{args.start}_{end}.png"
        save_equity_plot(
            result.equity,
            benchmark_curve,
            plot_path,
            title=f"RSI(2) mean reversion  {args.start} to {end}",
            data_source=source.name,
            kill_switch=result.kill_switch,
        )
        print(f"  equity curve -> {_display(plot_path)}")

    if args.no_log:
        print("  NOT logged to experiments.csv (--no-log)")
    else:
        path = append_row(
            _experiment_row(
                run_id, settings, source.name, label, args.start, end,
                strategy_metrics, benchmark_metrics, result,
            ),
            out_dir / "experiments.csv",
        )
        print(f"  logged run {run_id} -> {_display(path)}")

    decisions_path = out_dir / f"decisions_{run_id}.jsonl"
    decisions_path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in result.decisions)
    )
    print(f"  decision log -> {_display(decisions_path)}")

    return 0


def _display(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (e.g. a test tmp_path)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _experiment_row(
    run_id, settings, source_name, label, start, end, strategy, benchmark, result
) -> dict:
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


if __name__ == "__main__":
    raise SystemExit(main())
