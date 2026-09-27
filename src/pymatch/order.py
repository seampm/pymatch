"""Order primitives for pymatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count

_id_gen = count(1)


class Side(Enum):
    BID = auto()  # buy
    ASK = auto()  # sell


class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()


class TimeInForce(Enum):
    GTC = auto()  # good-til-cancelled: rests on the book
    IOC = auto()  # immediate-or-cancel: fill what you can, cancel the rest
    FOK = auto()  # fill-or-kill: fill everything or nothing


@dataclass
class Order:
    side: Side
    quantity: int
    price: int | None = None          # ticks; None for MARKET orders
    order_type: OrderType = OrderType.LIMIT
    tif: TimeInForce = TimeInForce.GTC
    timestamp_ns: int = 0
    order_id: int = field(default_factory=lambda: next(_id_gen))
    leaves: int = field(init=False)   # unfilled quantity

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type is OrderType.LIMIT and self.price is None:
            raise ValueError("LIMIT orders need a price")
        if self.order_type is OrderType.MARKET and self.tif is TimeInForce.GTC:
            raise ValueError("MARKET orders cannot rest; use IOC or FOK")
        self.leaves = self.quantity

    @property
    def is_buy(self) -> bool:
        return self.side is Side.BID


# ---------------------------------------------------------------------------
# Engine events: everything the matching engine did, as data.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderAccepted:
    order_id: int
    timestamp_ns: int


@dataclass(frozen=True)
class OrderFill:
    aggressor_id: int
    resting_id: int
    price: int          # trade happened at the resting order's price
    quantity: int
    timestamp_ns: int


@dataclass(frozen=True)
class OrderCancelled:
    order_id: int
    cancelled_leaves: int
    timestamp_ns: int


@dataclass(frozen=True)
class OrderRejected:
    order_id: int
    reason: str
    timestamp_ns: int


Event = OrderAccepted | OrderFill | OrderCancelled | OrderRejected
