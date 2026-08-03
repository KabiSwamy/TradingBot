"""Buy-and-hold benchmark — rule 12.

Every report includes buy-and-hold SPY over the same window with the same costs.
Without it "up 40%" is unreadable: it could be skill, or it could be less than
simply owning the index while taking more risk to get there.
"""

from __future__ import annotations

import pandas as pd

from backtest.costs import CostModel


def buy_and_hold(
    bars: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    initial_capital: float,
    costs: CostModel,
) -> pd.Series:
    """Equity curve for buying at the first open and holding to the last close.

    Costs are charged on both sides, at the same rate the strategy pays.

    Shares are fractional here, deliberately unlike the strategy's whole-share
    rule. The benchmark is a pure index proxy; rounding it to whole shares would
    strand idle cash that acts as a permanent drag and would make the comparison
    partly a story about lot sizes. The strategy rounds because it has to match a
    real broker in Phase 3; the benchmark has no such obligation. The asymmetry
    is noted in the report footer.
    """
    frame = bars.reindex(calendar).ffill().dropna(subset=["close"])
    if frame.empty:
        return pd.Series(dtype="float64", index=calendar, name="benchmark")

    entry_price = float(frame["open"].iloc[0])
    if entry_price <= 0:
        return pd.Series(dtype="float64", index=calendar, name="benchmark")

    # Solve shares * price * (1 + rate) = capital, so the entry fee is funded
    # from the same capital rather than added on top of it.
    shares = initial_capital / (entry_price * (1.0 + costs.rate_per_side))

    curve = (shares * frame["close"]).reindex(calendar).ffill()

    # The exit fee is only paid at the end, so it applies to the final point.
    final_notional = float(curve.iloc[-1])
    curve.iloc[-1] = final_notional - costs.fee(final_notional)

    return curve.rename("benchmark")
