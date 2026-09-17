"""The backtest engine: an event loop over daily bars.

The whole file exists to get one boundary right (rule 1):

    a decision dated day t uses only data through day t's close,
    and is filled at day t+1's open.

Each iteration therefore does three things in a fixed order — fill yesterday's
orders at today's open, mark the book at today's close, then decide using only
what today's close revealed. Nothing reads forward.

The engine takes *precomputed* signals rather than calling the strategy itself.
That keeps the dependency rule intact (backtest/ never imports strategies/;
run.py wires them together, exactly as jobs/nightly.py will in Phase 3) and it
lets the engine tests inject hand-made signals. The alternative — deriving
signals from prices inside every engine test — would mean each test had to build
a price path satisfying `close > SMA` and `RSI(2) < 10` at once, and those two
conditions pull against each other, so the tests would stop being hand-checkable
exactly where precision matters most.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.costs import CostModel
from backtest.portfolio import Order, Portfolio, Trade
from config.loader import Settings

SIGNAL_COLUMNS = ("rsi", "sma", "regime_ok", "entry_signal", "exit_signal")


@dataclass(frozen=True)
class _Signals:
    """Strategy output as numpy arrays keyed by symbol, indexed by calendar position."""

    entry: dict[str, np.ndarray]
    exit: dict[str, np.ndarray]
    rsi: dict[str, np.ndarray]
    sma: dict[str, np.ndarray]
    regime: dict[str, np.ndarray]


def _jsonable(value: float) -> float | None:
    """NaN -> None, so the decision log is valid JSON.

    `json.dumps` emits a bare `NaN` token for a float nan, which is not legal
    JSON and is rejected by strict parsers — including anything that will read
    these logs from another language. Warm-up rows genuinely have no RSI, and
    null is the honest encoding of that.
    """
    if value is None or value != value:
        return None
    return round(float(value), 6)


@dataclass(frozen=True)
class KillSwitch:
    triggered: bool = False
    trigger_date: pd.Timestamp | None = None
    trigger_drawdown: float = 0.0
    peak_equity: float = 0.0
    flatten_date: pd.Timestamp | None = None


@dataclass
class EngineResult:
    calendar: pd.DatetimeIndex
    equity: pd.Series
    cash: pd.Series
    market_value: pd.Series
    positions_count: pd.Series
    trades: list[Trade]
    decisions: list[dict]
    kill_switch: KillSwitch = field(default_factory=KillSwitch)

    @property
    def halted(self) -> bool:
        return self.kill_switch.triggered


def run_backtest(
    bars: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    settings: Settings,
    *,
    calendar: pd.DatetimeIndex | None = None,
    force_close_at_end: bool = True,
) -> EngineResult:
    """Run the strategy over `calendar` and return the full result.

    Args:
        bars: symbol -> OHLCV frame.
        signals: symbol -> frame with SIGNAL_COLUMNS, as produced by the strategy.
        settings: validated project settings.
        calendar: master trading calendar; defaults to the union of all bars.
        force_close_at_end: liquidate open positions at the final close.
            Positions still open when the window ends are counted as trades, not
            discarded — dropping them would quietly remove losers that had not
            yet recovered, which biases every trade statistic upward.

    Returns:
        EngineResult with daily series, closed trades, and the decision log.
    """
    symbols = sorted(bars)
    if calendar is None:
        calendar = _union_calendar(bars)

    opens, closes, has_bar = _align(bars, calendar, symbols)
    aligned_signals = _align_signals(signals, calendar, symbols)

    costs = CostModel(rate_per_side=settings.cost_per_side)
    portfolio = Portfolio(cash=settings.initial_capital)
    max_positions = settings.risk.max_positions
    time_stop = settings.risk.time_stop_days
    kill_threshold = -abs(settings.risk.kill_switch_drawdown)

    pending: list[Order] = []
    trades: list[Trade] = []
    decisions: list[dict] = []
    kill_switch = KillSwitch()
    halted = False

    peak_equity = settings.initial_capital
    last_close: dict[str, float] = {}

    equity_series = np.full(len(calendar), np.nan)
    cash_series = np.full(len(calendar), np.nan)
    value_series = np.full(len(calendar), np.nan)
    count_series = np.zeros(len(calendar), dtype=int)

    for i, day in enumerate(calendar):
        record: dict = {
            "date": day.strftime("%Y-%m-%d"),
            "fills": [],
            "orders": [],
            "skipped": [],
            "cancelled": [],
        }

        # --- 1. OPEN of day i: execute orders decided at the previous close ---
        pending = _execute_orders(
            pending, i, day, opens, portfolio, costs, trades, record
        )

        # --- 2. CLOSE of day i: mark the book to market ---
        for symbol in symbols:
            price = closes[symbol][i]
            if not np.isnan(price):
                last_close[symbol] = float(price)

        marks = {s: last_close.get(s, float("nan")) for s in symbols}
        stale = [
            s for s in portfolio.positions if not has_bar[s][i] and s in last_close
        ]
        market_value = portfolio.market_value(marks)
        equity = portfolio.cash + market_value
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0 if peak_equity > 0 else 0.0

        equity_series[i] = equity
        cash_series[i] = portfolio.cash
        value_series[i] = market_value
        count_series[i] = len(portfolio.positions)

        record.update(
            {
                "equity": round(equity, 6),
                "cash": round(portfolio.cash, 6),
                "market_value": round(market_value, 6),
                "peak_equity": round(peak_equity, 6),
                "drawdown": round(drawdown, 8),
                "positions": [
                    {
                        "symbol": p.symbol,
                        "shares": p.shares,
                        "entry_date": p.entry_date.strftime("%Y-%m-%d"),
                        "entry_price": round(p.entry_price, 6),
                        "days_held": p.days_held(i),
                    }
                    for p in sorted(portfolio.positions.values(), key=lambda p: p.symbol)
                ],
                "halted": halted,
                # Rule 10: the prices and indicator values every decision below
                # was taken from. Recorded for EVERY symbol, not just the ones
                # that acted, because the rule asks for the reason behind
                # "deliberate inaction" too — and the only way to show why a
                # symbol was passed over is to show what it looked like. With
                # this, any day's decisions are re-derivable from the log alone.
                "marks": {
                    symbol: {
                        "close": _jsonable(closes[symbol][i]),
                        "rsi": _jsonable(aligned_signals.rsi[symbol][i]),
                        "sma": _jsonable(aligned_signals.sma[symbol][i]),
                        "regime_ok": bool(aligned_signals.regime[symbol][i]),
                        "entry_signal": bool(aligned_signals.entry[symbol][i]),
                        "exit_signal": bool(aligned_signals.exit[symbol][i]),
                    }
                    for symbol in symbols
                },
            }
        )
        if stale:
            record["stale_marks"] = sorted(stale)

        # --- 3. Kill switch, before any decision is taken today (rule 5) ---
        if not halted and drawdown <= kill_threshold:
            halted = True
            flatten_date = calendar[i + 1] if i + 1 < len(calendar) else None
            kill_switch = KillSwitch(
                triggered=True,
                trigger_date=day,
                trigger_drawdown=drawdown,
                peak_equity=peak_equity,
                flatten_date=flatten_date,
            )
            pending = [
                Order("sell", symbol, "kill_switch", day)
                for symbol in sorted(portfolio.positions)
            ]
            record["kill_switch"] = {
                "triggered": True,
                "drawdown": round(drawdown, 8),
                "peak_equity": round(peak_equity, 6),
            }
            record["halted"] = True
            decisions.append(record)
            continue

        if halted:
            # Halt is terminal for the run. The curve continues as flat cash to
            # the end of the window rather than truncating, so the report still
            # covers the same span as the benchmark (rule 12).
            decisions.append(record)
            continue

        # --- 4. DECIDE using only data through this close ---
        exiting = _decide_exits(
            portfolio, aligned_signals, i, day, time_stop, pending, record
        )
        _decide_entries(
            portfolio,
            aligned_signals,
            symbols,
            i,
            day,
            equity,
            max_positions,
            exiting,
            pending,
            record,
        )

        decisions.append(record)

    # --- 5. End of window ---
    if pending and decisions:
        decisions[-1]["unfilled_end_of_data"] = [
            {"side": o.side, "symbol": o.symbol, "reason": o.reason} for o in pending
        ]

    if force_close_at_end and portfolio.positions:
        last = len(calendar) - 1
        for symbol in sorted(portfolio.positions):
            price = last_close.get(symbol)
            if price is None:
                continue
            trade, fill = portfolio.close_position(
                symbol, calendar[last], price, last, "end_of_backtest", costs
            )
            trades.append(trade)
            if decisions:
                decisions[-1]["fills"].append(fill)
        equity_series[last] = portfolio.cash
        cash_series[last] = portfolio.cash
        value_series[last] = 0.0
        count_series[last] = 0
        if decisions:
            decisions[-1]["equity"] = round(portfolio.cash, 6)
            decisions[-1]["cash"] = round(portfolio.cash, 6)
            decisions[-1]["market_value"] = 0.0
            decisions[-1]["positions"] = []

    return EngineResult(
        calendar=calendar,
        equity=pd.Series(equity_series, index=calendar, name="equity"),
        cash=pd.Series(cash_series, index=calendar, name="cash"),
        market_value=pd.Series(value_series, index=calendar, name="market_value"),
        positions_count=pd.Series(count_series, index=calendar, name="positions"),
        trades=trades,
        decisions=decisions,
        kill_switch=kill_switch,
    )


# --- the three phases of a day ---------------------------------------------


def _execute_orders(
    pending: list[Order],
    i: int,
    day: pd.Timestamp,
    opens: dict[str, np.ndarray],
    portfolio: Portfolio,
    costs: CostModel,
    trades: list[Trade],
    record: dict,
) -> list[Order]:
    """Fill yesterday's orders at today's open. Returns orders carried forward.

    Sells are executed before buys so proceeds from a liquidated position are
    available to a purchase filling at the same open — otherwise a slot could be
    freed and refilled on the same morning while the cash to do it was still
    notionally tied up, and buys would be skipped for no real reason.
    """
    carried: list[Order] = []
    ordering = sorted(
        pending,
        key=lambda o: (
            0 if o.side == "sell" else 1,
            o.rank_key if o.rank_key is not None else 0.0,
            o.symbol,
        ),
    )

    for order in ordering:
        price = opens[order.symbol][i]
        if np.isnan(price):
            if order.side == "sell":
                # Never get stuck long because of a data hole: carry the exit
                # forward to the next session that has an open.
                carried.append(order)
                record["cancelled"].append(
                    {"side": "sell", "symbol": order.symbol, "reason": "deferred_no_open"}
                )
            else:
                record["cancelled"].append(
                    {"side": "buy", "symbol": order.symbol, "reason": "no_open_price"}
                )
            continue

        price = float(price)
        if order.side == "sell":
            if order.symbol not in portfolio.positions:
                continue
            trade, fill = portfolio.close_position(
                order.symbol, day, price, i, order.reason, costs
            )
            trades.append(trade)
            fill["decided_date"] = order.decided_date.strftime("%Y-%m-%d")
            record["fills"].append(fill)
            continue

        budget = min(order.target_notional or 0.0, portfolio.cash)
        shares = costs.shares_for_budget(budget, price)
        if shares <= 0:
            record["cancelled"].append(
                {"side": "buy", "symbol": order.symbol, "reason": "insufficient_cash"}
            )
            continue
        _, fill = portfolio.open_position(order.symbol, shares, day, price, i, costs)
        fill["decided_date"] = order.decided_date.strftime("%Y-%m-%d")
        fill["reason"] = order.reason
        record["fills"].append(fill)

    return carried


def _decide_exits(
    portfolio: Portfolio,
    signals: _Signals,
    i: int,
    day: pd.Timestamp,
    time_stop: int,
    pending: list[Order],
    record: dict,
) -> set[str]:
    """Queue exits for positions whose bounce arrived or whose time ran out."""
    exiting: set[str] = set()
    for symbol in sorted(portfolio.positions):
        position = portfolio.positions[symbol]
        if bool(signals.exit[symbol][i]):
            reason = "rsi_exit"          # bounce beats the clock when both fire
        elif position.days_held(i) >= time_stop:
            reason = "time_stop"
        else:
            continue
        pending.append(Order("sell", symbol, reason, day))
        exiting.add(symbol)
        record["orders"].append({"side": "sell", "symbol": symbol, "reason": reason})
    return exiting


def _decide_entries(
    portfolio: Portfolio,
    signals: _Signals,
    symbols: list[str],
    i: int,
    day: pd.Timestamp,
    equity: float,
    max_positions: int,
    exiting: set[str],
    pending: list[Order],
    record: dict,
) -> None:
    """Queue entries for the best candidates that fit in the free slots.

    Position sizing is 1/max_positions of *current* equity, not of initial
    capital. A strategy meant to run indefinitely has to compound: fixing the
    size to starting capital would shrink positions in relative terms as the
    account grew, understate both returns and drawdowns, and leave the 15% kill
    switch measuring a different thing than the positions it is protecting.

    Note the order is sized in dollars, not shares. The share count is computed
    at the fill from the open price. That is not look-ahead — the decision
    (which symbol, how much capital) is fully determined by data through this
    close, and only the mechanical share count depends on the price it fills at,
    exactly as a real market-on-open notional order behaves. Fixing the share
    count tonight would overdraw the account on any gap-up open.
    """
    # A symbol exiting today is still in `positions`, so it is excluded here.
    # That is deliberate: allowing a same-morning sell-and-rebuy would let a
    # position dodge the time stop indefinitely, nullifying rule 6.
    candidates = [
        s
        for s in symbols
        if bool(signals.entry[s][i]) and s not in portfolio.positions
    ]
    if not candidates:
        return

    # Lowest RSI first (strategy spec); symbol name breaks ties so the run is
    # reproducible rather than dependent on dict ordering.
    candidates.sort(key=lambda s: (float(signals.rsi[s][i]), s))

    free_slots = max_positions - (len(portfolio.positions) - len(exiting))
    target_notional = equity / max_positions

    for rank, symbol in enumerate(candidates):
        rank_key = float(signals.rsi[symbol][i])
        if rank >= free_slots:
            record["skipped"].append(
                {"symbol": symbol, "reason": "no_free_slot", "rsi": round(rank_key, 6)}
            )
            continue
        pending.append(
            Order(
                side="buy",
                symbol=symbol,
                reason="entry_signal",
                decided_date=day,
                target_notional=target_notional,
                rank_key=rank_key,
            )
        )
        record["orders"].append(
            {
                "side": "buy",
                "symbol": symbol,
                "reason": "entry_signal",
                "rsi": round(rank_key, 6),
                "target_notional": round(target_notional, 6),
            }
        )


# --- alignment helpers ------------------------------------------------------


def _union_calendar(bars: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex([])
    for frame in bars.values():
        index = index.union(frame.index)
    return pd.DatetimeIndex(sorted(index), name="date")


def _align(
    bars: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, symbols: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Reindex every symbol onto the master calendar as plain numpy arrays."""
    opens, closes, has_bar = {}, {}, {}
    for symbol in symbols:
        frame = bars[symbol].reindex(calendar)
        opens[symbol] = frame["open"].to_numpy(dtype="float64")
        closes[symbol] = frame["close"].to_numpy(dtype="float64")
        has_bar[symbol] = frame["close"].notna().to_numpy()
    return opens, closes, has_bar


def _align_signals(
    signals: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, symbols: list[str]
) -> _Signals:
    """Reindex signals onto the calendar as numpy arrays.

    Arrays rather than DataFrames because the loop below reads a handful of
    scalars per symbol per day, and `.iloc[i]` on a DataFrame is roughly two
    orders of magnitude slower than indexing an ndarray. The values are
    identical; only the access cost changes.

    Days a symbol has no bar for are filled False — an absent symbol can never
    signal.
    """
    entry, exit_, rsi, sma, regime = {}, {}, {}, {}, {}
    flag_columns = ["entry_signal", "exit_signal", "regime_ok"]
    for symbol in symbols:
        source = signals[symbol]
        # Reindex the boolean columns WITH fill_value rather than filling NaN
        # afterwards. Reindexing bools onto a longer calendar first introduces
        # NaN and promotes the column to object dtype; the subsequent fillna
        # then silently downcasts it back, which pandas has deprecated. Filling
        # during the reindex means the NaN never exists and the dtype never
        # changes.
        #
        # This only arises on a ragged calendar — a universe whose symbols do
        # not share one listing date — so it is invisible against synthetic bars
        # and appears only on real data.
        flags = source[flag_columns].reindex(calendar, fill_value=False)
        values = source[["rsi", "sma"]].reindex(calendar)

        entry[symbol] = flags["entry_signal"].to_numpy(dtype=bool)
        exit_[symbol] = flags["exit_signal"].to_numpy(dtype=bool)
        regime[symbol] = flags["regime_ok"].to_numpy(dtype=bool)
        rsi[symbol] = values["rsi"].to_numpy(dtype="float64")
        sma[symbol] = values["sma"].to_numpy(dtype="float64")
    return _Signals(entry=entry, exit=exit_, rsi=rsi, sma=sma, regime=regime)
