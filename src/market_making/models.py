from dataclasses import dataclass
from datetime import datetime
from typing import Literal


Side = Literal["buy", "sell"]
OrderStatus = Literal["pending", "active", "filled", "cancelled", "rejected"]


@dataclass(slots=True)
class Order:
    order_id: str
    side: Side
    price: float
    quantity: float
    submitted_at: datetime
    activates_at: datetime
    filled_quantity: float = 0.0
    status: OrderStatus = "pending"
    cancel_requested_at: datetime | None = None
    cancel_effective_at: datetime | None = None

    # TODO: validate timezone awareness and delay ordering before replay.

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: str
    order_id: str
    timestamp: datetime
    side: Side
    price: float
    quantity: float
    fee: float = 0.0  # Positive charge; negative rebate.

    @property
    def inventory_delta(self) -> float:
        return self.quantity if self.side == "buy" else -self.quantity

    @property
    def cash_delta(self) -> float:
        return -self.inventory_delta * self.price - self.fee


@dataclass(slots=True)
class PositionState:
    cash: float = 0.0
    inventory: float = 0.0
    fees_paid: float = 0.0
    midpoint: float | None = None
    marked_at: datetime | None = None

    @property
    def mtm_value(self) -> float:
        if self.midpoint is None:
            if self.inventory == 0.0:
                return self.cash
            raise ValueError("Cannot value inventory without a midpoint.")
        return self.cash + self.inventory * self.midpoint
