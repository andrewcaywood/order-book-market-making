from dataclasses import dataclass
from math import isfinite

from .models import Order, Side
from .simulator import Simulator


@dataclass(frozen=True, slots=True)
class Quotes:
    bid: float
    ask: float


def _quotes(
    midpoint: float,
    inventory: float,
    half_spread: float,
    sensitivity: float,
) -> Quotes:
    if not isfinite(midpoint) or midpoint <= 0:
        raise ValueError("Midpoint must be finite and positive.")
    if not isfinite(inventory):
        raise ValueError("Inventory must be finite.")
    center = midpoint - sensitivity * inventory
    bid = center - half_spread
    ask = center + half_spread
    if not all(isfinite(price) and price > 0 for price in (bid, ask)):
        raise ValueError("Parameters produce invalid quote prices.")
    return Quotes(bid=bid, ask=ask)


@dataclass(frozen=True, slots=True)
class SymmetricStrategy:
    half_spread: float

    def __post_init__(self) -> None:
        if not isfinite(self.half_spread) or self.half_spread <= 0:
            raise ValueError("Half-spread must be finite and positive.")

    def quotes(self, midpoint: float, inventory: float = 0.0) -> Quotes:
        return _quotes(midpoint, inventory, self.half_spread, 0.0)

    def post(self, simulator: Simulator) -> tuple[Order, ...]:
        """Submit one pair; the caller owns cancellation and replacement."""
        if simulator.current_time is None or simulator.position.midpoint is None:
            raise RuntimeError("Process a market event before posting quotes.")
        inventory = simulator.position.inventory
        quotes = self.quotes(simulator.position.midpoint, inventory)
        reserved: dict[Side, float] = {"buy": 0.0, "sell": 0.0}
        for order in simulator.orders.values():
            if order.status in ("pending", "active"):
                reserved[order.side] += order.remaining_quantity

        # TODO: expose inventory-clamped sizes in quote previews, including
        # capacity still reserved by orders awaiting cancellation.
        orders = []
        sides: tuple[tuple[Side, float, float], ...] = (
            ("buy", quotes.bid, inventory),
            ("sell", quotes.ask, -inventory),
        )
        for side, price, signed_inventory in sides:
            room = max(
                0.0,
                simulator.config.max_inventory - signed_inventory - reserved[side],
            )
            size = min(simulator.config.order_size, room)
            if size > 0:
                orders.append(simulator.submit_order(side, price, size))
        return tuple(orders)


@dataclass(frozen=True, slots=True)
class InventoryAwareStrategy(SymmetricStrategy):
    sensitivity: float

    def __post_init__(self) -> None:
        SymmetricStrategy.__post_init__(self)
        if not isfinite(self.sensitivity) or self.sensitivity < 0:
            raise ValueError("Sensitivity must be finite and nonnegative.")

    def quotes(self, midpoint: float, inventory: float = 0.0) -> Quotes:
        return _quotes(midpoint, inventory, self.half_spread, self.sensitivity)
