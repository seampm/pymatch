"""Price-time priority limit order book.

Design notes
------------
* Prices are integer ticks. No floats anywhere near money.
* One side of the book: dict[tick -> deque[Order]] plus a sorted price list
  maintained with bisect. Bids are kept ascending and read from the back so
  the best bid is always the last element; asks are ascending, best is first.
* Matching is iterative over price levels, FIFO within a level
  (price-time priority). Trades print at the resting order's price.
* The book never crosses: every add() fully exhausts all matchable
  liquidity before resting anything.
"""

from __future__ import annotations

from bisect import bisect_left, insort_left
from collections import deque
from dataclasses import dataclass

from .order import (
    Event,
    Order,
    OrderAccepted,
    OrderCancelled,
    OrderFill,
    OrderRejected,
    OrderType,
    Side,
    TimeInForce,
)


@dataclass
class Level:
    price: int
    total_qty: int


class _BookSide:
    """One side of the book: price -> FIFO queue, with a sorted price list.

    Prices are stored ascending. For bids the best price is the highest
    (last element); for asks the best price is the lowest (first element).
    """

    __slots__ = ("prices", "queues", "is_bid")

    def __init__(self, is_bid: bool) -> None:
        self.is_bid = is_bid
        self.prices: list[int] = []          # ascending
        self.queues: dict[int, deque[Order]] = {}

    def add(self, order: Order) -> None:
        q = self.queues.get(order.price)
        if q is None:
            q = deque()
            self.queues[order.price] = q
            insort_left(self.prices, order.price)
        q.append(order)

    def remove(self, order: Order) -> bool:
        """Remove a specific resting order. Returns False if not found."""
        q = self.queues.get(order.price)
        if q is None:
            return False
        try:
            q.remove(order)
        except ValueError:
            return False
        if not q:
            del self.queues[order.price]
            i = bisect_left(self.prices, order.price)
            del self.prices[i]
        return True

    def best_price(self) -> int | None:
        if not self.prices:
            return None
        return self.prices[-1] if self.is_bid else self.prices[0]

    def depth(self, n: int) -> list[Level]:
        out: list[Level] = []
        if self.is_bid:
            prices = reversed(self.prices[-n:])
        else:
            prices = iter(self.prices[:n])
        for price in prices:
            qty = sum(o.leaves for o in self.queues[price])
            out.append(Level(price, qty))
        return out

    def total_quantity_at_or_better(self, limit: int) -> int:
        """Total resting qty this side could offer an aggressor at `limit`
        or better. For asks: prices <= limit. For bids: prices >= limit."""
        total = 0
        if self.is_bid:
            for price in reversed(self.prices):   # descending
                if price < limit:
                    break
                total += sum(o.leaves for o in self.queues[price])
        else:
            for price in self.prices:              # ascending
                if price > limit:
                    break
                total += sum(o.leaves for o in self.queues[price])
        return total

    def __len__(self) -> int:
        return sum(len(q) for q in self.queues.values())


class OrderBook:
    """Central limit order book with price-time priority matching."""

    def __init__(self, tick_size: int = 1) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        self.tick_size = tick_size
        self._bids = _BookSide(is_bid=True)
        self._asks = _BookSide(is_bid=False)
        self._orders: dict[int, Order] = {}   # resting orders by id
        self._time_ns = 0

    # -- queries ---------------------------------------------------------
    @property
    def best_bid(self) -> int | None:
        return self._bids.best_price()

    @property
    def best_ask(self) -> int | None:
        return self._asks.best_price()

    @property
    def midprice(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> int | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def depth(self, n: int = 5) -> tuple[list[Level], list[Level]]:
        """(bids, asks): top n levels each, best first."""
        return self._bids.depth(n), self._asks.depth(n)

    def order_count(self) -> int:
        return len(self._orders)

    # -- mutation ----------------------------------------------------------
    def add(self, order: Order) -> list[Event]:
        """Submit an order. Returns the events it caused, in order."""
        events: list[Event] = []
        self._time_ns = max(self._time_ns, order.timestamp_ns)

        if order.price is not None and order.price % self.tick_size:
            events.append(OrderRejected(order.order_id, "price not on tick", self._time_ns))
            return events

        if order.order_type is OrderType.MARKET:
            if (self._asks if order.is_buy else self._bids).best_price() is None:
                events.append(OrderRejected(order.order_id, "no liquidity", self._time_ns))
                return events
        elif order.tif is TimeInForce.FOK:
            if not self._fok_fillable(order):
                events.append(OrderRejected(order.order_id, "FOK not fully fillable", self._time_ns))
                return events

        events.extend(self._match(order))

        if order.leaves > 0:
            if order.order_type is OrderType.MARKET or order.tif is not TimeInForce.GTC:
                # IOC remainder (or market remainder): cancelled, not rejected.
                events.append(OrderCancelled(order.order_id, order.leaves, self._time_ns))
                order.leaves = 0
            else:
                self._rest(order)
                events.append(OrderAccepted(order.order_id, self._time_ns))
        return events

    def cancel(self, order_id: int, timestamp_ns: int = 0) -> list[Event]:
        self._time_ns = max(self._time_ns, timestamp_ns)
        order = self._orders.get(order_id)
        if order is None:
            return [OrderRejected(order_id, "unknown order", self._time_ns)]
        side = self._bids if order.is_buy else self._asks
        side.remove(order)
        del self._orders[order_id]
        leaves = order.leaves
        order.leaves = 0
        return [OrderCancelled(order_id, leaves, self._time_ns)]

    def replace(self, order_id: int, new_price: int, new_quantity: int,
                timestamp_ns: int = 0) -> list[Event]:
        """Cancel/replace: loses time priority (standard behavior)."""
        self._time_ns = max(self._time_ns, timestamp_ns)
        order = self._orders.get(order_id)
        if order is None:
            return [OrderRejected(order_id, "unknown order", self._time_ns)]
        events = self.cancel(order_id, timestamp_ns)
        if any(isinstance(e, OrderRejected) for e in events):
            return events
        new = Order(
            side=order.side,
            quantity=new_quantity,
            price=new_price,
            order_type=order.order_type,
            tif=order.tif,
            timestamp_ns=timestamp_ns,
        )
        events.extend(self.add(new))
        return events

    # -- internals ---------------------------------------------------------
    def _rest(self, order: Order) -> None:
        side = self._bids if order.is_buy else self._asks
        side.add(order)
        self._orders[order.order_id] = order

    def _fok_fillable(self, order: Order) -> bool:
        assert order.price is not None
        if order.is_buy:
            return self._asks.total_quantity_at_or_better(order.price) >= order.leaves
        return self._bids.total_quantity_at_or_better(order.price) >= order.leaves

    def _match(self, aggressor: Order) -> list[Event]:
        """Match against the opposite side until no more fills are possible."""
        events: list[Event] = []
        resting_side = self._asks if aggressor.is_buy else self._bids

        while aggressor.leaves > 0:
            best = resting_side.best_price()
            if best is None:
                break
            if aggressor.order_type is OrderType.LIMIT:
                assert aggressor.price is not None
                if aggressor.is_buy and best > aggressor.price:
                    break
                if not aggressor.is_buy and best < aggressor.price:
                    break

            queue = resting_side.queues[best]
            while queue and aggressor.leaves > 0:
                resting = queue[0]
                fill_qty = min(aggressor.leaves, resting.leaves)
                aggressor.leaves -= fill_qty
                resting.leaves -= fill_qty
                events.append(OrderFill(
                    aggressor_id=aggressor.order_id,
                    resting_id=resting.order_id,
                    price=best,          # trade prints at resting price
                    quantity=fill_qty,
                    timestamp_ns=self._time_ns,
                ))
                if resting.leaves == 0:
                    queue.popleft()
                    del self._orders[resting.order_id]
            if not queue:
                del resting_side.queues[best]
                i = bisect_left(resting_side.prices, best)
                del resting_side.prices[i]
        return events

    # -- validation --------------------------------------------------------
    def check_invariants(self) -> None:
        """Raise AssertionError if any book invariant is violated."""
        bid, ask = self.best_bid, self.best_ask
        assert not (bid is not None and ask is not None and bid >= ask), \
            f"crossed book: bid {bid} >= ask {ask}"
        for side in (self._bids, self._asks):
            assert side.prices == sorted(side.prices), "price list not sorted"
            for price, q in side.queues.items():
                assert price in side.prices
                assert len(q) > 0, "empty price level"
                for o in q:
                    assert o.leaves > 0, "resting order with no leaves"
                    assert o.order_id in self._orders, "queue/order-map mismatch"
        assert len(self._orders) == len(self._bids) + len(self._asks), \
            "order map size mismatch"
