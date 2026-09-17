"""Plateau analysis — CLAUDE.md rule 3.

    "A parameter value qualifies only if its neighbors also perform well.
     Always pick the middle of a stable region, never the single best
     historical value. A spike surrounded by cliffs = curve-fit = rejected."

The rule is easy to nod along to and easy to quietly ignore, because the best
cell is always sitting right there at the top of a sorted table. So this module
computes the plateau answer explicitly, reports it **alongside** the naive best
cell, and states plainly when the two disagree — which is the normal case and
the whole lesson.

Neighbourhood: cells adjacent by one step along a single axis (the 6-cell von
Neumann neighbourhood), not the 26-cell diagonal one. A diagonal neighbour
differs in every parameter at once, which is not what "a neighbouring parameter
value" means.

Ranking metric: `robust_score` = min(own, worst neighbour). A cell scores well
only if it and everything immediately around it perform — which is the rule
restated as arithmetic. Ranking by this instead of by the raw metric is what
makes the recommendation the middle of a stable region rather than a peak.

Spikes are measured additively (`own - neighbour_mean`) rather than as a ratio,
because Sharpe is routinely negative and ratios of signed numbers are nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

AXES = ("entry_rsi", "exit_rsi", "time_stop_days")

# A cell whose own metric exceeds its neighbourhood mean by more than this many
# standard deviations of the whole surface is called an isolated spike.
SPIKE_SIGMA = 1.0


@dataclass
class PlateauReport:
    metric: str
    table: pd.DataFrame
    best: dict
    recommended: dict
    best_is_spike: bool
    recommended_on_edge: bool
    agree: bool
    surface_std: float
    notes: list[str] = field(default_factory=list)


def analyse(results: pd.DataFrame, metric: str = "sharpe") -> PlateauReport:
    """Score every grid cell by how well its neighbourhood performs."""
    for axis in AXES:
        if axis not in results.columns:
            raise ValueError(f"sweep results missing axis column {axis!r}")
    if metric not in results.columns:
        raise ValueError(f"sweep results missing metric column {metric!r}")

    table = results.copy()
    valid = table[metric].notna()
    if not valid.any():
        raise ValueError(f"no cell produced a usable {metric}")

    levels = {axis: sorted(table[axis].unique()) for axis in AXES}
    index_of = {axis: {v: i for i, v in enumerate(levels[axis])} for axis in AXES}

    lookup: dict[tuple[int, int, int], float] = {}
    for row in table.itertuples():
        key = tuple(index_of[axis][getattr(row, axis)] for axis in AXES)
        lookup[key] = getattr(row, metric)

    neighbour_mean, neighbour_min, robust, counts, on_edge = [], [], [], [], []
    for row in table.itertuples():
        key = tuple(index_of[axis][getattr(row, axis)] for axis in AXES)
        own = getattr(row, metric)
        values = [
            lookup[n] for n in _neighbours(key, levels) if not _isnan(lookup.get(n))
        ]

        if values:
            neighbour_mean.append(float(np.mean(values)))
            neighbour_min.append(float(np.min(values)))
        else:
            neighbour_mean.append(float("nan"))
            neighbour_min.append(float("nan"))

        counts.append(len(values))
        robust.append(
            float(min([own, *values])) if values and not _isnan(own) else float("nan")
        )
        on_edge.append(
            any(key[a] in (0, len(levels[axis]) - 1) for a, axis in enumerate(AXES))
        )

    table["neighbour_mean"] = neighbour_mean
    table["neighbour_min"] = neighbour_min
    table["robust_score"] = robust
    table["neighbours"] = counts
    table["on_edge"] = on_edge
    table["spike_gap"] = table[metric] - table["neighbour_mean"]

    surface_std = float(table[metric].std(ddof=1))
    table["spike_sigma"] = (
        table["spike_gap"] / surface_std if surface_std else float("nan")
    )

    best = table.loc[table[metric].idxmax()].to_dict()
    recommended = table.loc[table["robust_score"].idxmax()].to_dict()

    best_is_spike = bool(
        surface_std and not _isnan(best.get("spike_sigma"))
        and best["spike_sigma"] > SPIKE_SIGMA
    )
    agree = all(best[axis] == recommended[axis] for axis in AXES)

    notes = _notes(best, recommended, best_is_spike, agree, metric)

    return PlateauReport(
        metric=metric,
        table=table,
        best=best,
        recommended=recommended,
        best_is_spike=best_is_spike,
        recommended_on_edge=bool(recommended["on_edge"]),
        agree=agree,
        surface_std=surface_std,
        notes=notes,
    )


def _neighbours(key: tuple[int, ...], levels: dict[str, list]) -> list[tuple[int, ...]]:
    """Cells one step away along exactly one axis."""
    out = []
    for axis_pos, axis in enumerate(AXES):
        for step in (-1, 1):
            moved = list(key)
            moved[axis_pos] += step
            if 0 <= moved[axis_pos] < len(levels[axis]):
                out.append(tuple(moved))
    return out


def _isnan(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _notes(best, recommended, best_is_spike, agree, metric) -> list[str]:
    notes: list[str] = []
    if agree:
        notes.append(
            "The best cell is also the most robust one. That is the good case, "
            "but confirm the surface is genuinely broad rather than uniformly flat."
        )
    else:
        notes.append(
            "Best cell and recommended cell DIFFER. Rule 3 says take the "
            "recommended one: the best cell wins on history, the recommended one "
            "is the cell whose neighbours also hold up."
        )
    if best_is_spike:
        notes.append(
            f"The best cell sits {best['spike_sigma']:.1f} standard deviations "
            "above its own neighbourhood — the signature of a curve-fit spike. "
            "Rule 3: a spike surrounded by cliffs is rejected."
        )
    if recommended["on_edge"]:
        notes.append(
            "The recommended cell is on the EDGE of the grid, so it may not be a "
            "middle at all — the stable region could extend past where the sweep "
            "looked. Widen the grid before trusting it."
        )
    if recommended["neighbour_min"] < 0 and metric == "sharpe":
        notes.append(
            "Even the recommended neighbourhood contains a negative Sharpe, so "
            "there is no region here that is good throughout."
        )
    return notes


def render_plateau_report(
    report: PlateauReport, *, metric: str = "sharpe", data_source: str = ""
) -> str:
    """Human-readable plateau analysis."""
    lines: list[str] = []
    w = lines.append

    if data_source == "synthetic":
        w("=" * 72)
        w("***  SYNTHETIC DATA — THIS IS NOT A RESEARCH RESULT  ***")
        w("Generated bars. This validates the sweep machinery, not the strategy.")
        w("=" * 72)

    w("")
    w("=" * 72)
    w(f"  PLATEAU ANALYSIS — rule 3, ranking on {metric}")
    w("=" * 72)
    w("")
    w(f"  cells swept        {len(report.table)}")
    w(f"  surface spread     {report.table[metric].min():.3f} .. "
      f"{report.table[metric].max():.3f}  (sd {report.surface_std:.3f})")
    w("")

    w(f"  {'':22}{'BEST CELL':>14}{'RECOMMENDED':>16}")
    w("  " + "-" * 52)
    for axis, label in zip(AXES, ("entry RSI", "exit RSI", "time stop")):
        w(f"  {label:22}{report.best[axis]:>14g}{report.recommended[axis]:>16g}")
    w("  " + "-" * 52)
    w(f"  {metric:22}{report.best[metric]:>14.3f}{report.recommended[metric]:>16.3f}")
    w(f"  {'worst neighbour':22}{report.best['neighbour_min']:>14.3f}"
      f"{report.recommended['neighbour_min']:>16.3f}")
    w(f"  {'robust score':22}{report.best['robust_score']:>14.3f}"
      f"{report.recommended['robust_score']:>16.3f}")
    w(f"  {'above neighbours':22}{report.best['spike_sigma']:>13.2f}s"
      f"{report.recommended['spike_sigma']:>15.2f}s")
    w(f"  {'trades':22}{report.best['trade_count']:>14g}"
      f"{report.recommended['trade_count']:>16g}")
    w("")

    w("  VERDICT")
    w("  " + "-" * 52)
    for note in report.notes:
        for line in _wrap(note, 68):
            w(f"  {line}")
        w("")

    w("  Robust score is min(own, worst neighbour): a cell scores well only if")
    w("  everything one step away from it does too. Recommending on that instead")
    w("  of the raw metric is rule 3 expressed as arithmetic.")
    w("=" * 72)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
