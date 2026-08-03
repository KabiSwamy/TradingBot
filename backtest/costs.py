"""The cost model. Rule 2: costs are always on.

Commission and slippage are modelled together as a single rate charged on the
notional of both sides of every round trip. The fee is kept separate from the
fill price rather than folded into it: cash impact is identical either way, but
keeping them apart means `entry_price` and `exit_price` in the trade record stay
the prices that actually printed, which makes the flat-price round-trip test
exact to the cent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Proportional cost charged per side.

    `rate_per_side` of 0.0005 is 0.05% — the project default.
    """

    rate_per_side: float

    def __post_init__(self) -> None:
        if self.rate_per_side < 0:
            raise ValueError("rate_per_side must not be negative")

    def fee(self, notional: float) -> float:
        """Cost of transacting `notional` dollars on one side."""
        return abs(notional) * self.rate_per_side

    def buy_cost(self, shares: float, price: float) -> float:
        """Total cash leaving the account to buy `shares` at `price`."""
        notional = shares * price
        return notional + self.fee(notional)

    def sell_proceeds(self, shares: float, price: float) -> float:
        """Cash arriving after selling `shares` at `price`."""
        notional = shares * price
        return notional - self.fee(notional)

    def shares_for_budget(self, budget: float, price: float) -> int:
        """Largest whole share count affordable within `budget`, fees included.

        Whole shares, floored. Three reasons, in order of weight:

        1. Phase 3 reconciles against a real cash brokerage account, where
           fractional fills are not guaranteed to behave identically. Rule 9's
           whole point is that backtest and live must be comparable, so the
           backtest adopts the more restrictive convention now rather than
           discovering the difference against a broker later.
        2. Flooring can never overdraw cash.
        3. It keeps every engine test exactly hand-verifiable.

        Dividing by `price * (1 + rate)` rather than by `price` is what makes
        `shares * price + fee <= budget` a guarantee instead of a near-miss.
        """
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if budget <= 0:
            return 0
        return max(0, math.floor(budget / (price * (1.0 + self.rate_per_side))))
