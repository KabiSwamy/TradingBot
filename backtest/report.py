"""Text report and equity-curve plot.

Rule 12 sets the required contents. Beyond those, this module is responsible for
the honesty furniture: the synthetic-data banner, the Sharpe sanity warning, the
kill-switch notice, and a footer stating the assumptions that shape the numbers.
A report that omits those is not shorter, it is just less true.
"""

from __future__ import annotations

import matplotlib

# Must precede the pyplot import: this project runs headless (and in CI), where
# an interactive backend fails to initialise.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.metrics import Metrics  # noqa: E402

SHARPE_SUSPECT_THRESHOLD = 2.5


def render_report(
    strategy: Metrics,
    benchmark: Metrics | None,
    *,
    data_source: str,
    window_label: str,
    settings,
    kill_switch=None,
) -> str:
    """Build the full text report."""
    lines: list[str] = []
    w = lines.append

    if data_source == "synthetic":
        w("=" * 72)
        w("***  SYNTHETIC DATA — THIS IS NOT A RESEARCH RESULT  ***")
        w("Generated bars, not market history. Use only to verify the machinery.")
        w("=" * 72)
        w("")

    w("=" * 72)
    w(f"  RSI(2) MEAN REVERSION — {strategy.start} to {strategy.end}")
    w(f"  data source: {data_source}    window: {window_label}    "
      f"trading days: {strategy.trading_days}")
    w("=" * 72)
    w("")

    w(f"  {'':22}{'STRATEGY':>14}{'BUY & HOLD ' + settings.benchmark:>18}")
    w("  " + "-" * 54)
    for name, value, fmt in _rows(strategy):
        bench = _benchmark_cell(benchmark, name, fmt)
        w(f"  {name:22}{value:>14}{bench:>18}")

    w("")
    w("  TRADES")
    w("  " + "-" * 54)
    w(f"  {'count':22}{strategy.trade_count:>14}")
    w(f"  {'win rate':22}{strategy.win_rate:>13.1%}")
    w(f"  {'avg win':22}{strategy.avg_win:>13.2%}")
    w(f"  {'avg loss':22}{strategy.avg_loss:>13.2%}")
    w(f"  {'win/loss ratio':22}{strategy.win_loss_ratio:>14.2f}")
    w(f"  {'profit factor':22}{strategy.profit_factor:>14.2f}")
    w(f"  {'avg holding days':22}{strategy.avg_holding_days:>14.1f}")
    w(f"  {'exposure (capital)':22}{strategy.exposure:>13.1%}")
    w(f"  {'time in market':22}{strategy.time_in_market:>13.1%}")
    w(f"  {'max concurrent':22}{strategy.max_concurrent:>14}")
    w("")

    for warning in warnings_for(strategy, benchmark, kill_switch):
        w(warning)
    if any(warnings_for(strategy, benchmark, kill_switch)):
        w("")

    w("  ASSUMPTIONS")
    w("  " + "-" * 54)
    w(f"  costs {settings.cost_per_side:.4%} per side, charged on both sides of "
      "every round trip")
    w("  Sharpe uses daily returns, 252 trading days, risk-free rate 0")
    w("  fills at the next session's open; signals on the final bar cannot fill")
    w("  strategy buys whole shares; the benchmark holds fractional shares")
    w("  positions open at the window end are closed at the final close and counted")
    w("  universe is hindsight-selected (10 ETFs liquid today), so results are")
    w("    optimistic by an amount this backtest cannot measure")
    w("=" * 72)
    return "\n".join(lines)


def warnings_for(strategy: Metrics, benchmark: Metrics | None, kill_switch) -> list[str]:
    """The lines a reader must not be allowed to miss."""
    out: list[str] = []

    if strategy.sharpe > SHARPE_SUSPECT_THRESHOLD:
        out.append(
            f"  *** SUSPECT: Sharpe {strategy.sharpe:.2f} exceeds "
            f"{SHARPE_SUSPECT_THRESHOLD}. Treat as a probable bug until explained. ***"
        )
    if kill_switch is not None and getattr(kill_switch, "triggered", False):
        date = kill_switch.trigger_date
        out.append(
            f"  *** KILL SWITCH FIRED {date:%Y-%m-%d} at "
            f"{kill_switch.trigger_drawdown:.2%} from peak — trading halted for the "
            "rest of the window. Sharpe after a halt is not comparable. ***"
        )
    if strategy.trade_count < 30:
        out.append(
            f"  *** Only {strategy.trade_count} trades — too few to distinguish "
            "skill from luck. ***"
        )
    if benchmark is not None and strategy.sharpe <= benchmark.sharpe:
        out.append(
            f"  *** Does not beat buy-and-hold risk-adjusted "
            f"({strategy.sharpe:.2f} vs {benchmark.sharpe:.2f} Sharpe). ***"
        )
    return out


def _rows(m: Metrics):
    return [
        ("final equity", f"{m.final_equity:,.0f}", "money"),
        ("total return", f"{m.total_return:.1%}", "pct"),
        ("CAGR", f"{m.cagr:.2%}", "pct"),
        ("Sharpe", f"{m.sharpe:.2f}", "ratio"),
        ("volatility", f"{m.volatility:.1%}", "pct"),
        ("max drawdown", f"{m.max_drawdown:.1%}", "pct"),
        ("Calmar", f"{m.calmar:.2f}", "ratio"),
    ]


def _benchmark_cell(benchmark: Metrics | None, name: str, fmt: str) -> str:
    if benchmark is None:
        return "n/a"
    mapping = {
        "final equity": f"{benchmark.final_equity:,.0f}",
        "total return": f"{benchmark.total_return:.1%}",
        "CAGR": f"{benchmark.cagr:.2%}",
        "Sharpe": f"{benchmark.sharpe:.2f}",
        "volatility": f"{benchmark.volatility:.1%}",
        "max drawdown": f"{benchmark.max_drawdown:.1%}",
        "Calmar": f"{benchmark.calmar:.2f}",
    }
    return mapping.get(name, "")


def save_equity_plot(
    equity: pd.Series,
    benchmark: pd.Series | None,
    path,
    *,
    title: str,
    data_source: str,
    kill_switch=None,
):
    """Write the equity curve against the benchmark, plus the drawdown beneath."""
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    top.plot(equity.index, equity.to_numpy(), label="RSI(2) strategy", linewidth=1.4)
    if benchmark is not None and not benchmark.empty:
        top.plot(
            benchmark.index,
            benchmark.to_numpy(),
            label="buy & hold",
            linewidth=1.2,
            alpha=0.75,
        )

    if kill_switch is not None and getattr(kill_switch, "triggered", False):
        top.axvline(kill_switch.trigger_date, color="red", linestyle="--", linewidth=1)
        top.annotate(
            "kill switch",
            xy=(kill_switch.trigger_date, equity.max()),
            color="red",
            fontsize=9,
        )

    banner = "  [SYNTHETIC DATA — NOT A RESEARCH RESULT]" if data_source == "synthetic" else ""
    top.set_title(title + banner)
    top.set_ylabel("equity ($)")
    top.legend(loc="upper left")
    top.grid(alpha=0.3)

    drawdown = equity / equity.cummax() - 1.0
    bottom.fill_between(drawdown.index, drawdown.to_numpy(), 0, alpha=0.4, color="firebrick")
    bottom.set_ylabel("drawdown")
    bottom.set_xlabel("date")
    bottom.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def save_sweep_heatmap(
    results: pd.DataFrame,
    path,
    *,
    metric: str = "sharpe",
    report=None,
    data_source: str = "",
    title: str = "parameter sweep",
):
    """Heatmap of `metric` across the sweep grid, one panel per time stop.

    All panels share one colour scale, because per-panel scaling would make a
    mediocre region look identical to a strong one and quietly defeat the point
    of comparing time-stop values at all.

    The best cell and the recommended (plateau) cell are marked differently, so
    the gap between "won on history" and "holds up under its neighbours" is
    visible rather than buried in a table.
    """
    stops = sorted(results["time_stop_days"].unique())
    entries = sorted(results["entry_rsi"].unique())
    exits = sorted(results["exit_rsi"].unique())

    vmin = float(results[metric].min())
    vmax = float(results[metric].max())

    figure, axes = plt.subplots(
        1, len(stops), figsize=(5.2 * len(stops), 4.6), squeeze=False
    )

    for panel, stop in zip(axes[0], stops):
        subset = results[results["time_stop_days"] == stop]
        grid = (
            subset.pivot(index="entry_rsi", columns="exit_rsi", values=metric)
            .reindex(index=entries, columns=exits)
        )
        image = panel.imshow(
            grid.to_numpy(), cmap="RdYlGn", vmin=vmin, vmax=vmax,
            origin="lower", aspect="auto",
        )
        panel.set_xticks(range(len(exits)), [f"{x:g}" for x in exits])
        panel.set_yticks(range(len(entries)), [f"{e:g}" for e in entries])
        panel.set_xlabel("exit RSI")
        panel.set_ylabel("entry RSI")
        panel.set_title(f"time stop {stop:g}d")

        for yi, entry in enumerate(entries):
            for xi, exit_ in enumerate(exits):
                value = grid.to_numpy()[yi][xi]
                if value == value:
                    panel.text(xi, yi, f"{value:.2f}", ha="center", va="center",
                               fontsize=7, color="black")

        if report is not None:
            _mark(panel, report.best, stop, entries, exits, "o", "black", "best")
            _mark(panel, report.recommended, stop, entries, exits, "s", "blue",
                  "recommended")

    # Markers are drawn only on the panel whose time stop they belong to, so the
    # handles must be gathered across every panel — collecting them from the
    # first one silently produces no legend whenever the marked cells live
    # elsewhere, which is most of the time.
    handles, labels = [], []
    for panel in axes[0]:
        for handle, label in zip(*panel.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=len(labels),
                      fontsize=9, frameon=True)

    banner = "  [SYNTHETIC DATA — NOT A RESEARCH RESULT]" if data_source == "synthetic" else ""
    figure.suptitle(f"{title}  —  {metric}{banner}")
    figure.colorbar(image, ax=axes[0].tolist(), shrink=0.85, label=metric)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path


def _mark(panel, cell, stop, entries, exits, marker, colour, label):
    """Mark a grid cell, but only on the panel its time stop belongs to."""
    if cell["time_stop_days"] != stop:
        return
    panel.scatter(
        exits.index(cell["exit_rsi"]), entries.index(cell["entry_rsi"]),
        marker=marker, s=190, facecolors="none", edgecolors=colour,
        linewidths=2.2, label=label,
    )
