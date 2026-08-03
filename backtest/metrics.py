"""Performance metrics.

Rule 12: a report is Sharpe, max drawdown, win rate, avg win/loss and trade
count — not total return. Every formula here is stated explicitly, including its
assumptions, because a metric whose definition is ambiguous is a metric that can
be quietly reinterpreted later to look better.

Assumption, stated once and repeated in the report footer: the risk-free rate is
0. At a 2010s cash rate this flatters Sharpe slightly; it is applied identically
to the strategy and the benchmark, so the comparison between them is unaffected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from backtest.portfolio import Trade

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Metrics:
    label: str
    start: str
    end: str
    trading_days: int
    initial_equity: float
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    volatility: float
    max_drawdown: float
    max_drawdown_start: str
    max_drawdown_end: str
    calmar: float
    trade_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    profit_factor: float
    avg_holding_days: float
    exposure: float
    time_in_market: float
    max_concurrent: int

    def as_dict(self) -> dict:
        return asdict(self)


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe on daily returns, risk-free rate 0.

    Sample standard deviation (ddof=1). A constant equity curve has zero
    variance and no risk-adjusted information, so it scores 0 rather than
    dividing by zero.
    """
    if len(returns) < 2:
        return 0.0
    sigma = float(returns.std(ddof=1))
    if sigma == 0.0 or np.isnan(sigma):
        return 0.0
    return float(returns.mean() / sigma * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Worst peak-to-trough decline, with the dates of that peak and trough."""
    drawdown = drawdown_series(equity)
    if drawdown.empty:
        return 0.0, pd.NaT, pd.NaT
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    return float(drawdown.min()), peak, trough


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def trade_stats(trades: list[Trade]) -> dict:
    """Per closed round trip — never per day.

    A flat trade counts as a loss: it paid costs and returned nothing.
    """
    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
        }

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]

    avg_win = float(np.mean([t.return_pct for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.return_pct for t in losses])) if losses else 0.0

    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))

    return {
        "trade_count": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": abs(avg_win / avg_loss) if avg_loss else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss else float("inf"),
        "avg_holding_days": float(np.mean([t.holding_days for t in trades])),
    }


def compute_metrics(
    equity: pd.Series,
    *,
    label: str,
    trades: list[Trade] | None = None,
    market_value: pd.Series | None = None,
    positions_count: pd.Series | None = None,
) -> Metrics:
    """Assemble every reported figure from an equity curve and its trades."""
    trades = trades or []
    equity = equity.dropna()
    returns = daily_returns(equity)
    dd, dd_start, dd_end = max_drawdown(equity)
    stats = trade_stats(trades)

    annual_return = cagr(equity)
    volatility = (
        float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(returns) > 1
        else 0.0
    )

    # Exposure: capital-weighted is the headline because it is what explains the
    # Sharpe. With 3 slots at 1/3 each, time-in-market can read 65% while only
    # 25% of capital is actually deployed, so both are reported — their gap is
    # itself diagnostic.
    if market_value is not None:
        exposure = float((market_value / equity).reindex(equity.index).fillna(0).mean())
    else:
        exposure = 0.0
    if positions_count is not None:
        counts = positions_count.reindex(equity.index).fillna(0)
        time_in_market = float((counts > 0).mean())
        max_concurrent = int(counts.max())
    else:
        time_in_market, max_concurrent = 0.0, 0

    return Metrics(
        label=label,
        start=str(equity.index[0].date()) if len(equity) else "",
        end=str(equity.index[-1].date()) if len(equity) else "",
        trading_days=len(equity),
        initial_equity=float(equity.iloc[0]) if len(equity) else 0.0,
        final_equity=float(equity.iloc[-1]) if len(equity) else 0.0,
        total_return=(
            float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
        ),
        cagr=annual_return,
        sharpe=sharpe_ratio(returns),
        volatility=volatility,
        max_drawdown=dd,
        max_drawdown_start=str(dd_start.date()) if pd.notna(dd_start) else "",
        max_drawdown_end=str(dd_end.date()) if pd.notna(dd_end) else "",
        calmar=(annual_return / abs(dd)) if dd else 0.0,
        exposure=exposure,
        time_in_market=time_in_market,
        max_concurrent=max_concurrent,
        **stats,
    )
