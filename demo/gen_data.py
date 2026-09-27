"""Generate demo data: a simulated trading session for the GitHub Pages demo.

Records top-of-book depth snapshots and the trade tape as JSON.
Run: python demo/gen_data.py  (writes demo/session.json)
"""

import json
import sys

sys.path.insert(0, "src")

from pymatch import Exchange, MarketConfig, MarketMaker, OrderBook
from pymatch.market import new_state, step as market_step
from pymatch.order import OrderFill, OrderCancelled, OrderRejected
from pymatch.strategy import BookView

STEPS = 1200
SNAPSHOT_EVERY = 4


def main():
    book = OrderBook()
    exchange = Exchange(book, latency_ns=500_000)
    mm = MarketMaker(exchange, size=50, edge_ticks=1)
    cfg = MarketConfig(steps=STEPS, seed=7)
    rng, state = new_state(cfg)

    snapshots = []
    trades = []
    pnl = []
    cash = inv = 0
    last = (None, None)

    for i in range(STEPS):
        t = i * 1_000_000
        for e in market_step(exchange, cfg, rng, state, t, i):
            if isinstance(e, OrderFill):
                aa = e.aggressor_id in mm.my_orders
                ar = e.resting_id in mm.my_orders
                trades.append({"t": t, "price": e.price, "qty": e.quantity,
                               "mine": bool(aa or ar)})
                if aa or ar:
                    o = mm.my_orders[e.aggressor_id if aa else e.resting_id]
                    s = e.quantity if o.is_buy else -e.quantity
                    cash -= s * e.price
                    inv += s
                    mm.on_fill(e, ar and not aa)
                    key = e.aggressor_id if aa else e.resting_id
                    if mm.my_orders[key].leaves == 0:
                        del mm.my_orders[key]
            elif isinstance(e, (OrderCancelled, OrderRejected)):
                mm.my_orders.pop(e.order_id, None)
        b, a = book.best_bid, book.best_ask
        if (b, a) != last:
            last = (b, a)
            mm.on_book(BookView(b, a, book.midprice, t))
        if i % SNAPSHOT_EVERY == 0 and b is not None and a is not None:
            bids, asks = book.depth(10)
            mid = book.midprice
            snapshots.append({
                "t": t,
                "bids": [[lv.price, lv.total_qty] for lv in bids],
                "asks": [[lv.price, lv.total_qty] for lv in asks],
                "mid": mid,
                "equity": cash + int(inv * mid),
            })

    # Liquidate for final P&L.
    if inv and book.best_bid and book.best_ask:
        cash += inv * (book.best_bid if inv > 0 else book.best_ask)
        inv = 0

    out = {
        "snapshots": snapshots,
        "trades": trades[-400:],
        "final_pnl_ticks": cash,
        "n_snapshots": len(snapshots),
    }
    with open("demo/session.json", "w") as f:
        json.dump(out, f)
    print(f"snapshots={len(snapshots)} trades={len(trades)} final_pnl={cash} ticks")


if __name__ == "__main__":
    main()
