"""Throughput benchmark for the matching engine.

Measures raw order-book operations/sec (no exchange latency, no I/O):
  1. add-only: non-crossing limit orders (book building)
  2. mixed: realistic mix of adds, cancels, and crossing orders

Run: python bench/bench_match.py
"""

import random
import sys
import time

sys.path.insert(0, "src")

from pymatch import Order, OrderBook, OrderType, Side, TimeInForce


def bench_add_only(n=200_000):
    book = OrderBook()
    rng = random.Random(0)
    orders = [
        Order(side=Side.BID if rng.random() < 0.5 else Side.ASK,
              quantity=rng.randint(1, 100),
              price=rng.randint(9500, 10500),
              timestamp_ns=i)
        for i in range(n)
    ]
    # De-cross so these purely rest (measures book-building throughput).
    t0 = time.perf_counter()
    for o in orders:
        # shift non-aggressive: bids below 9990, asks above 10010
        if o.is_buy and o.price is not None and o.price >= 9995:
            o.price = 9990
        elif not o.is_buy and o.price is not None and o.price <= 10005:
            o.price = 10010
        book.add(o)
    dt = time.perf_counter() - t0
    return n / dt


def bench_mixed(n=200_000):
    book = OrderBook()
    rng = random.Random(1)
    live = []
    t0 = time.perf_counter()
    for i in range(n):
        r = rng.random()
        if r < 0.6:
            side = Side.BID if rng.random() < 0.5 else Side.ASK
            o = Order(side=side, quantity=rng.randint(1, 100),
                      price=rng.randint(9900, 10100), timestamp_ns=i)
            evs = book.add(o)
            if any(type(e).__name__ == "OrderAccepted" for e in evs):
                live.append(o.order_id)
        elif r < 0.8 and live:
            book.cancel(live.pop(rng.randrange(len(live))), timestamp_ns=i)
        else:
            side = Side.BID if rng.random() < 0.5 else Side.ASK
            book.add(Order(side=side, quantity=rng.randint(1, 50),
                           order_type=OrderType.MARKET,
                           tif=TimeInForce.IOC, timestamp_ns=i))
    dt = time.perf_counter() - t0
    return n / dt


if __name__ == "__main__":
    for fn in (bench_add_only, bench_mixed):
        # warmup
        fn(20_000)
        rate = fn()
        print(f"{fn.__name__}: {rate:,.0f} orders/sec")
