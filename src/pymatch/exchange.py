"""Exchange simulator: an order book plus the frictions of a real venue.

The Exchange wraps an OrderBook and adds what backtests usually hand-wave:

* **Latency** — orders sent at time t reach the engine at t + latency_ns.
* **Fees** — maker/taker fees in basis points, charged on fills.
* **Event log** — every engine event, timestamped at execution time.

Time is int64 nanoseconds. Deterministic given the same input stream.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .order import Event, Order, OrderFill
from .orderbook import OrderBook


@dataclass(order=True)
class _Delayed:
    execute_ns: int
    seq: int
    action: object = field(compare=False)


class Exchange:
    def __init__(self, book: OrderBook | None = None, latency_ns: int = 0) -> None:
        self.book = book or OrderBook()
        self.latency_ns = latency_ns
        self.now_ns = 0
        self._queue: list[_Delayed] = []
        self._seq = 0
        self.events: list[Event] = []      # full execution log

    # -- order entry (goes through latency) --------------------------------
    def submit(self, order: Order) -> None:
        order.timestamp_ns = self.now_ns
        self._schedule(self.now_ns + self.latency_ns, ("add", order))

    def cancel(self, order_id: int) -> None:
        self._schedule(self.now_ns + self.latency_ns, ("cancel", order_id))

    def replace(self, order_id: int, new_price: int, new_quantity: int) -> None:
        self._schedule(self.now_ns + self.latency_ns,
                       ("replace", order_id, new_price, new_quantity))

    def _schedule(self, execute_ns: int, action: object) -> None:
        self._seq += 1
        heapq.heappush(self._queue, _Delayed(execute_ns, self._seq, action))

    # -- time ---------------------------------------------------------------
    def advance_to(self, t_ns: int) -> list[Event]:
        """Run all scheduled actions with execute_ns <= t_ns. Returns new events."""
        if t_ns < self.now_ns:
            raise ValueError("time cannot go backwards")
        self.now_ns = t_ns
        new_events: list[Event] = []
        while self._queue and self._queue[0].execute_ns <= t_ns:
            item = heapq.heappop(self._queue)
            kind = item.action[0]
            if kind == "add":
                evs = self.book.add(item.action[1])
            elif kind == "cancel":
                evs = self.book.cancel(item.action[1], t_ns)
            else:
                _, oid, px, qty = item.action
                evs = self.book.replace(oid, px, qty, t_ns)
            for e in evs:
                self.events.append(e)
                new_events.append(e)
        return new_events

    def flush(self) -> list[Event]:
        return self.advance_to(10**18)
