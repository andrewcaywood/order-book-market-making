from dataclasses import dataclass
from datetime import datetime, timedelta
from heapq import heappop, heappush
from itertools import count
from math import isfinite
from typing import Iterable, Literal

from .models import Order, PositionState, Side, Trade


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    activation_delay: timedelta = timedelta(0)
    cancellation_delay: timedelta = timedelta(0)
    order_size: float = 1.0
    max_inventory: float = 10.0
    fee_rate: float = 0.0

    def __post_init__(self) -> None:
        if min(self.activation_delay, self.cancellation_delay) < timedelta(0):
            raise ValueError("Delays must be nonnegative.")
        for value in (self.order_size, self.max_inventory):
            if not isfinite(value) or value <= 0:
                raise ValueError("Size and inventory limit must be positive.")
        if not isfinite(self.fee_rate):
            raise ValueError("Fee rate must be finite.")


@dataclass(frozen=True, slots=True)
class MarketEvent:
    timestamp: datetime
    best_bid: float
    best_ask: float
    trade_price: float | None = None
    trade_quantity: float = 0.0
    aggressor_side: Side | None = None

    def __post_init__(self) -> None:
        if self.timestamp.utcoffset() is None:
            raise ValueError("Market timestamps must be timezone-aware.")
        if not all(isfinite(p) and p > 0 for p in (self.best_bid, self.best_ask)):
            raise ValueError("Book prices must be finite and positive.")
        if self.best_bid > self.best_ask:
            raise ValueError("Crossed market snapshot.")
        if not isfinite(self.trade_quantity) or self.trade_quantity < 0:
            raise ValueError("Trade quantity must be finite and nonnegative.")
        if self.trade_price is None:
            if self.trade_quantity != 0 or self.aggressor_side is not None:
                raise ValueError("Trade fields require a trade price.")
        elif (
            not isfinite(self.trade_price)
            or self.trade_price <= 0
            or self.aggressor_side not in ("buy", "sell")
        ):
            raise ValueError("Trade requires a positive price and aggressor side.")


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    timestamp: datetime
    best_bid: float
    best_ask: float
    cash: float
    inventory: float
    fees_paid: float
    mtm_value: float


Action = Literal["activate", "cancel"]


class Simulator:
    fill_assumption = "volume_limited_trade_through"

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        initial_cash: float = 0.0,
    ) -> None:
        if not isfinite(initial_cash):
            raise ValueError("Initial cash must be finite.")
        self.config = config if config is not None else SimulatorConfig()
        self.position = PositionState(cash=initial_cash)
        self.orders: dict[str, Order] = {}
        self.fills: list[Trade] = []
        self.history: list[PortfolioSnapshot] = []
        self.current_time: datetime | None = None
        self.best_bid: float | None = None
        self.best_ask: float | None = None
        self._sequence = count()
        self._order_ids = count(1)
        self._actions: list[tuple[datetime, int, Action, str]] = []

    def submit_order(
        self,
        side: Side,
        price: float,
        quantity: float | None = None,
    ) -> Order:
        now = self._require_time()
        size = self.config.order_size if quantity is None else quantity
        if side not in ("buy", "sell"):
            raise ValueError("Unknown order side.")
        if not all(isfinite(value) and value > 0 for value in (price, size)):
            raise ValueError("Price and quantity must be finite and positive.")
        reserved = sum(
            order.remaining_quantity
            for order in self.orders.values()
            if order.side == side and order.status in ("pending", "active")
        )
        size = min(size, max(0.0, self._inventory_room(side) - reserved))
        if size <= 0:
            raise ValueError("No unreserved inventory capacity for this order.")
        order = Order(
            order_id=f"order-{next(self._order_ids)}",
            side=side,
            price=price,
            quantity=size,
            submitted_at=now,
            activates_at=now + self.config.activation_delay,
        )
        self.orders[order.order_id] = order
        self._schedule(order.activates_at, "activate", order.order_id)
        return order

    def cancel_order(self, order_id: str) -> None:
        now = self._require_time()
        order = self.orders[order_id]
        if order.status not in ("pending", "active"):
            return
        if order.cancel_requested_at is not None:
            return
        order.cancel_requested_at = now
        order.cancel_effective_at = now + self.config.cancellation_delay
        self._schedule(order.cancel_effective_at, "cancel", order_id)

    def on_event(self, event: MarketEvent) -> PortfolioSnapshot:
        if self.current_time is not None and event.timestamp < self.current_time:
            raise ValueError("Events must arrive in chronological order.")
        self._process_due(event.timestamp)
        self.current_time = event.timestamp
        self.best_bid = event.best_bid
        self.best_ask = event.best_ask
        self.position.midpoint = event.best_bid / 2 + event.best_ask / 2
        self.position.marked_at = event.timestamp
        self._match(event)
        snapshot = PortfolioSnapshot(
            timestamp=event.timestamp,
            best_bid=event.best_bid,
            best_ask=event.best_ask,
            cash=self.position.cash,
            inventory=self.position.inventory,
            fees_paid=self.position.fees_paid,
            mtm_value=self.position.mtm_value,
        )
        self.history.append(snapshot)
        return snapshot

    def run(self, events: Iterable[MarketEvent]) -> None:
        for event in events:
            self.on_event(event)

    def _require_time(self) -> datetime:
        if self.current_time is None:
            raise RuntimeError("Process a market event before submitting orders.")
        return self.current_time

    def _schedule(self, timestamp: datetime, action: Action, order_id: str) -> None:
        heappush(self._actions, (timestamp, next(self._sequence), action, order_id))

    def _process_due(self, timestamp: datetime) -> None:
        # TODO: exact-time actions currently slip to the next event; use <=.
        while self._actions and self._actions[0][0] < timestamp:
            _, _, action, order_id = heappop(self._actions)
            order = self.orders[order_id]
            if action == "activate" and order.status == "pending":
                order.status = "active"
            elif action == "cancel" and order.status in ("pending", "active"):
                order.status = "cancelled"

    def _inventory_room(self, side: Side) -> float:
        signed_inventory = (
            self.position.inventory if side == "buy" else -self.position.inventory
        )
        return max(0.0, self.config.max_inventory - signed_inventory)

    def _match(self, event: MarketEvent) -> None:
        if event.trade_price is None or event.trade_quantity == 0:
            return
        side: Side = "buy" if event.aggressor_side == "sell" else "sell"
        eligible = [
            order
            for order in self.orders.values()
            if order.status == "active"
            and order.side == side
            and (
                event.trade_price < order.price
                if side == "buy"
                else event.trade_price > order.price
            )
        ]
        # Stable sort preserves submission order for equal price and time.
        eligible.sort(
            key=lambda order: (
                -order.price if side == "buy" else order.price,
                order.activates_at,
            )
        )
        available = event.trade_quantity
        for order in eligible:
            quantity = min(
                order.remaining_quantity, available, self._inventory_room(side)
            )
            if quantity <= 0:
                break
            fill = Trade(
                trade_id=f"fill-{len(self.fills) + 1}",
                order_id=order.order_id,
                timestamp=event.timestamp,
                side=side,
                price=order.price,
                quantity=quantity,
                fee=order.price * quantity * self.config.fee_rate,
            )
            self.position.cash += fill.cash_delta
            self.position.inventory += fill.inventory_delta
            self.position.fees_paid += fill.fee
            self.fills.append(fill)
            remaining = order.remaining_quantity
            order.filled_quantity += quantity
            if quantity == remaining:
                order.filled_quantity = order.quantity
                order.status = "filled"
            available -= quantity
