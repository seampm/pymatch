"""Correctness tests for the matching engine.

Covers: price priority, time priority, partial fills, IOC/FOK semantics,
cancel/replace, and book invariants under randomized order flow.
"""

import random

import pytest

from pymatch import Order, OrderBook, OrderType, Side, TimeInForce
from pymatch.order import OrderAccepted, OrderCancelled, OrderFill, OrderRejected


def limit(side, price, qty, tif=TimeInForce.GTC, ts=0):
    return Order(side=side, quantity=qty, price=price, tif=tif, timestamp_ns=ts)


def market(side, qty, ts=0):
    return Order(side=side, quantity=qty, order_type=OrderType.MARKET,
                 tif=TimeInForce.IOC, timestamp_ns=ts)


def fills(events):
    return [e for e in events if isinstance(e, OrderFill)]


def test_simple_cross():
    book = OrderBook()
    book.add(limit(Side.ASK, 100, 10))
    evs = book.add(limit(Side.BID, 100, 10))
    f = fills(evs)
    assert len(f) == 1
    assert f[0].price == 100 and f[0].quantity == 10
    assert book.best_bid is None and book.best_ask is None
    book.check_invariants()


def test_trade_prints_at_resting_price():
    book = OrderBook()
    book.add(limit(Side.ASK, 100, 10))
    f = fills(book.add(limit(Side.BID, 105, 10)))[0]
    assert f.price == 100  # resting ask's price, not the aggressor's limit
    book.check_invariants()


def test_price_priority():
    book = OrderBook()
    book.add(limit(Side.ASK, 102, 5, ts=1))
    book.add(limit(Side.ASK, 100, 5, ts=2))   # better price, later time
    book.add(limit(Side.ASK, 101, 5, ts=3))
    f = fills(book.add(limit(Side.BID, 102, 12)))
    assert [x.price for x in f] == [100, 101, 102]
    assert sum(x.quantity for x in f) == 12
    book.check_invariants()


def test_time_priority_within_level():
    book = OrderBook()
    first = limit(Side.ASK, 100, 5, ts=1)
    second = limit(Side.ASK, 100, 5, ts=2)
    book.add(first)
    book.add(second)
    f = fills(book.add(limit(Side.BID, 100, 7)))
    assert f[0].resting_id == first.order_id and f[0].quantity == 5
    assert f[1].resting_id == second.order_id and f[1].quantity == 2
    book.check_invariants()


def test_partial_fill_rests_remainder():
    book = OrderBook()
    book.add(limit(Side.ASK, 100, 5))
    evs = book.add(limit(Side.BID, 100, 12))
    assert sum(e.quantity for e in fills(evs)) == 5
    assert any(isinstance(e, OrderAccepted) for e in evs)
    assert book.best_bid == 100
    assert book.depth(1)[0][0].total_qty == 7
    book.check_invariants()


def test_ioc_cancels_remainder():
    book = OrderBook()
    book.add(limit(Side.ASK, 100, 5))
    evs = book.add(limit(Side.BID, 100, 12, tif=TimeInForce.IOC))
    assert sum(e.quantity for e in fills(evs)) == 5
    cancelled = [e for e in evs if isinstance(e, OrderCancelled)]
    assert len(cancelled) == 1 and cancelled[0].cancelled_leaves == 7
    assert book.order_count() == 0
    book.check_invariants()


def test_fok_all_or_nothing():
    book = OrderBook()
    book.add(limit(Side.ASK, 100, 5))
    evs = book.add(limit(Side.BID, 100, 12, tif=TimeInForce.FOK))
    assert fills(evs) == []
    assert any(isinstance(e, OrderRejected) for e in evs)
    assert book.order_count() == 1  # resting ask untouched
    # exact fill works
    evs = book.add(limit(Side.BID, 100, 5, tif=TimeInForce.FOK))
    assert sum(e.quantity for e in fills(evs)) == 5
    book.check_invariants()


def test_fok_respects_price_limit():
    book = OrderBook()
    book.add(limit(Side.ASK, 100, 5))
    book.add(limit(Side.ASK, 200, 100))  # too expensive for a 100-limit buy
    evs = book.add(limit(Side.BID, 100, 10, tif=TimeInForce.FOK))
    assert any(isinstance(e, OrderRejected) for e in evs)
    book.check_invariants()


def test_market_order_sweeps():
    book = OrderBook()
    book.add(limit(Side.ASK, 100, 5, ts=1))
    book.add(limit(Side.ASK, 101, 5, ts=2))
    f = fills(book.add(market(Side.BID, 8)))
    assert [(x.price, x.quantity) for x in f] == [(100, 5), (101, 3)]
    book.check_invariants()


def test_market_order_no_liquidity_rejected():
    book = OrderBook()
    evs = book.add(market(Side.BID, 8))
    assert any(isinstance(e, OrderRejected) for e in evs)
    book.check_invariants()


def test_cancel():
    book = OrderBook()
    o = limit(Side.BID, 99, 10)
    book.add(o)
    evs = book.cancel(o.order_id)
    assert isinstance(evs[0], OrderCancelled) and evs[0].cancelled_leaves == 10
    assert book.order_count() == 0
    assert book.best_bid is None
    # double cancel -> rejected
    assert isinstance(book.cancel(o.order_id)[0], OrderRejected)
    book.check_invariants()


def test_cancel_unknown():
    book = OrderBook()
    assert isinstance(book.cancel(999999)[0], OrderRejected)


def test_replace_loses_time_priority():
    book = OrderBook()
    a = limit(Side.ASK, 100, 10, ts=1)
    b = limit(Side.ASK, 100, 10, ts=2)
    book.add(a)
    book.add(b)
    book.replace(b.order_id, 100, 10, timestamp_ns=3)
    # a (ts=1) is still first; the replaced b went to the back of the queue
    f = fills(book.add(limit(Side.BID, 100, 12)))
    assert f[0].resting_id == a.order_id
    book.check_invariants()


def test_tick_size_enforced():
    book = OrderBook(tick_size=5)
    evs = book.add(limit(Side.BID, 103, 10))
    assert any(isinstance(e, OrderRejected) for e in evs)
    book.add(limit(Side.BID, 100, 10))
    book.check_invariants()


def test_book_never_crosses():
    book = OrderBook()
    book.add(limit(Side.BID, 99, 10))
    book.add(limit(Side.ASK, 101, 10))
    book.add(limit(Side.BID, 100, 10))   # would cross the ask at 101? no: 100 < 101
    assert book.best_bid == 100 and book.best_ask == 101
    evs = book.add(limit(Side.BID, 101, 3))  # crosses: matches, doesn't rest
    assert not any(isinstance(e, OrderAccepted) for e in evs)
    assert book.best_bid == 100
    book.check_invariants()


def test_randomized_flow_conserves_quantity():
    """Fuzz: random adds/cancels; total resting qty == sum of leaves."""
    rng = random.Random(42)
    book = OrderBook()
    resting: dict[int, Order] = {}
    for i in range(3000):
        r = rng.random()
        if r < 0.55:
            side = Side.BID if rng.random() < 0.5 else Side.ASK
            price = rng.randint(90, 110)
            qty = rng.randint(1, 50)
            tif = rng.choice([TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK])
            o = limit(side, price, qty, tif=tif, ts=i)
            evs = book.add(o)
            if any(isinstance(e, OrderAccepted) for e in evs):
                resting[o.order_id] = o
            for e in evs:
                if isinstance(e, OrderFill):
                    rid = e.resting_id
                    if rid in resting and resting[rid].leaves == 0:
                        del resting[rid]
        elif r < 0.8 and resting:
            oid = rng.choice(list(resting))
            book.cancel(oid, timestamp_ns=i)
            resting.pop(oid, None)
        else:
            book.add(market(Side.BID if rng.random() < 0.5 else Side.ASK,
                            rng.randint(1, 30), ts=i))
            # prune filled resting orders
            for oid in [k for k, v in resting.items() if v.leaves == 0]:
                del resting[oid]
        if i % 500 == 0:
            book.check_invariants()
    book.check_invariants()
    total_leaves = sum(o.leaves for o in resting.values())
    _, asks = book.depth(1000)
    bids, _ = book.depth(1000)
    assert total_leaves == sum(l.total_qty for l in bids + asks)
