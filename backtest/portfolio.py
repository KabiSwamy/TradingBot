"""Positions, trades and orders — the bookkeeping the engine moves around."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from backtest.costs import CostModel


@dataclass
class Position:
    """An open long position.

    `entry_bar` is an integer index into the master calendar rather than an
    incrementing counter. A counter can drift if a day is ever skipped; an index
    cannot, and it makes the time stop trivially correct even when a symbol has
    missing bars.
    """

    symbol: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    entry_bar: int

    def days_held(self, bar: int) -> int:
        """Trading days held as of calendar index `bar`, counting the fill day as 1.

        The fill day counts as day 1, so `days_held >= 7` first becomes true on
        the 7th session the position has been open. Combined with next-open
        fills, a position is exposed across exactly 7 closes and is flat at the
        open of the 8th — which is the reading of "force-exited after 7 trading
        days" this project uses.
        """
        return bar - self.entry_bar + 1


@dataclass(frozen=True)
class Trade:
    """A completed round trip."""

    symbol: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_fee: float
    exit_reason: str
    entry_bar: int
    exit_bar: int

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def fees(self) -> float:
        return self.entry_fee + self.exit_fee

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    @property
    def return_pct(self) -> float:
        cost_basis = self.entry_price * self.shares
        return self.net_pnl / cost_basis if cost_basis else 0.0

    @property
    def holding_days(self) -> int:
        """Trading days from the entry fill to the exit fill.

        Both fills happen at an open, so this is the elapsed span rather than a
        count of endpoints: bought at the open of bar 1, sold at the open of
        bar 8, held 7 days. A 7-day time stop therefore reports 7 here, which is
        what "position age = 7 trading days" should mean when read back.
        """
        return self.exit_bar - self.entry_bar

    @property
    def is_win(self) -> bool:
        """A flat trade is a loss: it paid costs and returned nothing."""
        return self.net_pnl > 0


@dataclass(frozen=True)
class Order:
    """An instruction decided at one close, to be filled at the next open."""

    side: Literal["buy", "sell"]
    symbol: str
    reason: str
    decided_date: pd.Timestamp
    target_notional: float | None = None
    rank_key: float | None = None


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def market_value(self, marks: dict[str, float]) -> float:
        return sum(p.shares * marks[p.symbol] for p in self.positions.values())

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + self.market_value(marks)

    def open_position(
        self,
        symbol: str,
        shares: int,
        date: pd.Timestamp,
        price: float,
        bar: int,
        costs: CostModel,
    ) -> tuple[Position, dict]:
        if symbol in self.positions:
            raise ValueError(f"{symbol}: already holding a position")
        notional = shares * price
        fee = costs.fee(notional)
        self.cash -= notional + fee
        position = Position(
            symbol=symbol,
            shares=shares,
            entry_date=date,
            entry_price=price,
            entry_fee=fee,
            entry_bar=bar,
        )
        self.positions[symbol] = position
        fill = {
            "side": "buy",
            "symbol": symbol,
            "shares": shares,
            "price": round(price, 6),
            "fee": round(fee, 6),
        }
        return position, fill

    def close_position(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        bar: int,
        reason: str,
        costs: CostModel,
    ) -> tuple[Trade, dict]:
        position = self.positions.pop(symbol)
        notional = position.shares * price
        fee = costs.fee(notional)
        self.cash += notional - fee
        trade = Trade(
            symbol=symbol,
            shares=position.shares,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            entry_fee=position.entry_fee,
            exit_date=date,
            exit_price=price,
            exit_fee=fee,
            exit_reason=reason,
            entry_bar=position.entry_bar,
            exit_bar=bar,
        )
        fill = {
            "side": "sell",
            "symbol": symbol,
            "shares": position.shares,
            "price": round(price, 6),
            "fee": round(fee, 6),
            "reason": reason,
        }
        return trade, fill
