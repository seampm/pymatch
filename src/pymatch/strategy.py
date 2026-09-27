"""Strategy interface and two example strategies.

A Strategy sees callbacks as the simulation advances and submits orders
through the Exchange. Strategies never touch the OrderBook directly —
like a real colocated client, they only see the public feed plus their
own fills.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exchange import Exchange
from .order import Order, OrderFill, Side, TimeInForce


@dataclass(frozen=True)
class BookView:
    best_bid: int | None
    best_ask: int | None
    midprice: float | None
    timestamp_ns: int


class Strategy:
    def __init__(self, exchange: Exchange) -> None:
        self.exchange = exchange
        self.my_orders: dict[int, Order] = {}

    # -- callbacks (override) -------------------------------------------
    def on_book(self, book: BookView) -> None:
        """Called when the top of book changes."""

    def on_fill(self, fill: OrderFill, maker: bool) -> None:
        """Called on each of our fills. maker=True if we were resting."""

    # -- helpers -----------------------------------------------------------
    def buy(self, price: int, qty: int, tif=TimeInForce.GTC) -> int:
        o = Order(side=Side.BID, quantity=qty, price=price, tif=tif,
                  timestamp_ns=self.exchange.now_ns)
        self.my_orders[o.order_id] = o
        self.exchange.submit(o)
        return o.order_id

    def sell(self, price: int, qty: int, tif=TimeInForce.GTC) -> int:
        o = Order(side=Side.ASK, quantity=qty, price=price, tif=tif,
                  timestamp_ns=self.exchange.now_ns)
        self.my_orders[o.order_id] = o
        self.exchange.submit(o)
        return o.order_id

    def cancel_all(self) -> None:
        for oid in list(self.my_orders):
            self.exchange.cancel(oid)


class MarketMaker(Strategy):
    """Classic inventory-skewed market maker.

    Quotes `size` on both sides of the touch. Skews quotes against
    inventory: long inventory -> shade the bid down (don't want more),
    shade the ask down (want to sell). Re-quotes on every book update.
    """

    def __init__(self, exchange: Exchange, size: int = 100,
                 edge_ticks: int = 1, skew_per_lot: float = 0.5,
                 max_inventory: int = 2000) -> None:
        super().__init__(exchange)
        self.size = size
        self.edge_ticks = edge_ticks
        self.skew_per_lot = skew_per_lot
        self.max_inventory = max_inventory
        self.inventory = 0

    def on_book(self, book: BookView) -> None:
        if book.best_bid is None or book.best_ask is None:
            return
        self.cancel_all()
        # Skew both quotes against inventory: when long, shade quotes down
        # (less eager to buy, more eager to sell); when short, shade up.
        skew = int(self.inventory / self.size * self.skew_per_lot)
        bid_px = book.best_bid - self.edge_ticks - skew
        ask_px = book.best_ask + self.edge_ticks - skew
        if bid_px >= ask_px:      # never cross ourselves
            return
        if self.inventory < self.max_inventory:
            self.buy(bid_px, self.size)
        if self.inventory > -self.max_inventory:
            self.sell(ask_px, self.size)

    def on_fill(self, fill: OrderFill, maker: bool) -> None:
        o = self.my_orders.get(fill.aggressor_id) or \
            self.my_orders.get(fill.resting_id)
        if o is None:
            return
        self.inventory += fill.quantity if o.is_buy else -fill.quantity


class Momentum(Strategy):
    """Naive momentum taker: keeps a midprice history; buys when the
    short-term return over `lookback` steps exceeds `threshold`, sells
    when it drops below -threshold. Flat otherwise. A deliberately simple
    baseline — the point is the framework, not the alpha."""

    def __init__(self, exchange: Exchange, size: int = 50,
                 lookback: int = 20, threshold: float = 0.0004) -> None:
        super().__init__(exchange)
        self.size = size
        self.lookback = lookback
        self.threshold = threshold
        self.mids: list[float] = []
        self.inventory = 0

    def on_book(self, book: BookView) -> None:
        if book.midprice is None:
            return
        self.mids.append(book.midprice)
        if len(self.mids) < self.lookback + 1:
            return
        ret = self.mids[-1] / self.mids[-self.lookback - 1] - 1.0
        from .order import OrderType
        want = 0
        if ret > self.threshold:
            want = self.size
        elif ret < -self.threshold:
            want = -self.size
        delta = want - self.inventory
        if delta == 0:
            return
        side = Side.BID if delta > 0 else Side.ASK
        o = Order(side=side, quantity=abs(delta), order_type=OrderType.MARKET,
                  tif=TimeInForce.IOC, timestamp_ns=self.exchange.now_ns)
        self.my_orders[o.order_id] = o
        self.exchange.submit(o)

    def on_fill(self, fill: OrderFill, maker: bool) -> None:
        o = self.my_orders.get(fill.aggressor_id)
        if o is None:
            return
        self.inventory += fill.quantity if o.is_buy else -fill.quantity
