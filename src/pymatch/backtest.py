"""Event-driven backtester.

Replays a market-data stream through the Exchange, lets the Strategy
trade against it with latency and fees, and tracks the portfolio.

Accounting is in integer ticks for cash; inventory is marked to the
midprice at each step for the equity curve. Fees are charged in ticks:
taker fills pay taker_fee_bps, maker fills pay maker_fee_bps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .exchange import Exchange
from .market import MarketConfig, new_state, step as market_step
from .order import OrderAccepted, OrderCancelled, OrderFill, OrderRejected
from .orderbook import OrderBook
from .strategy import BookView, Strategy


@dataclass
class BacktestResult:
    strategy_name: str
    final_cash_ticks: int
    final_inventory: int
    total_fees_ticks: int
    n_fills: int
    n_maker_fills: int
    equity_ticks: list[int] = field(default_factory=list)   # marked to mid
    timestamps_ns: list[int] = field(default_factory=list)

    # -- derived metrics -------------------------------------------------
    @property
    def total_pnl_ticks(self) -> int:
        return self.final_cash_ticks  # inventory liquidated at end below

    def sharpe(self, periods_per_year: float = 252 * 24 * 3600) -> float:
        """Sharpe on equity-curve returns. periods_per_year assumes the
        equity curve is sampled once per simulated second."""
        eq = self.equity_ticks
        if len(eq) < 3:
            return 0.0
        rets = [(eq[i] / eq[i - 1] - 1.0) for i in range(1, len(eq))
                if eq[i - 1] != 0]
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        if var <= 0:
            return 0.0
        return mean / math.sqrt(var) * math.sqrt(periods_per_year)

    def max_drawdown_ticks(self) -> int:
        peak = self.equity_ticks[0] if self.equity_ticks else 0
        worst = 0
        for e in self.equity_ticks:
            peak = max(peak, e)
            worst = max(worst, peak - e)
        return worst


def run_backtest(strategy_cls: type[Strategy], market_cfg: MarketConfig,
                 latency_ns: int = 500_000, maker_fee_bps: float = 0.2,
                 taker_fee_bps: float = 0.5,
                 sample_every_steps: int = 20, **strategy_kwargs) -> BacktestResult:
    """Run one backtest. Deterministic for a fixed market seed."""
    book = OrderBook(tick_size=market_cfg.tick_size)
    exchange = Exchange(book, latency_ns=latency_ns)
    strategy = strategy_cls(exchange, **strategy_kwargs)

    cash = 0            # ticks, signed
    inventory = 0
    fees = 0
    n_fills = 0
    n_maker = 0
    equity: list[int] = []
    times: list[int] = []

    # We drive the market step by step instead of generate()'s fire-and-
    # forget so the strategy can react between steps.
    rng, state = new_state(market_cfg)
    last_bid = last_ask = None

    def pump(events: list) -> None:
        nonlocal cash, inventory, fees, n_fills, n_maker, last_bid, last_ask
        for e in events:
            if isinstance(e, OrderFill):
                mine_as_aggressor = e.aggressor_id in strategy.my_orders
                mine_as_resting = e.resting_id in strategy.my_orders
                if not (mine_as_aggressor or mine_as_resting):
                    continue
                maker = mine_as_resting and not mine_as_aggressor
                signed = e.quantity if (
                    strategy.my_orders[e.aggressor_id if mine_as_aggressor
                                        else e.resting_id].is_buy) else -e.quantity
                # Counterparty side: we bought -> cash down, inventory up.
                cash -= signed * e.price
                inventory += signed
                fee = round(e.price * e.quantity *
                            (maker_fee_bps if maker else taker_fee_bps) / 10_000)
                fees += fee
                cash -= fee
                n_fills += 1
                n_maker += 1 if maker else 0
                strategy.on_fill(e, maker)
                key = e.aggressor_id if mine_as_aggressor else e.resting_id
                filled_order = strategy.my_orders.get(key)
                if filled_order is not None and filled_order.leaves == 0:
                    del strategy.my_orders[key]
            elif isinstance(e, (OrderAccepted, OrderCancelled, OrderRejected)):
                oid = e.order_id
                if isinstance(e, (OrderCancelled, OrderRejected)):
                    # Dead order: drop from attribution tracking so the
                    # strategy's live-order set stays small.
                    strategy.my_orders.pop(oid, None)
        b, a = book.best_bid, book.best_ask
        if (b, a) != (last_bid, last_ask):
            last_bid, last_ask = b, a
            strategy.on_book(BookView(b, a, book.midprice, exchange.now_ns))

    for step_idx in range(market_cfg.steps):
        t = step_idx * 1_000_000
        pump(market_step(exchange, market_cfg, rng, state, t, step_idx))
        if step_idx % sample_every_steps == 0 and book.midprice is not None:
            equity.append(cash + int(inventory * book.midprice))
            times.append(t)

    # Liquidate remaining inventory at the final touch (cross the spread).
    if inventory != 0 and book.best_bid is not None and book.best_ask is not None:
        px = book.best_bid if inventory > 0 else book.best_ask
        cash += inventory * px
        inventory = 0
    equity.append(cash)
    times.append(market_cfg.steps * 1_000_000)

    return BacktestResult(
        strategy_name=strategy_cls.__name__,
        final_cash_ticks=cash,
        final_inventory=inventory,
        total_fees_ticks=fees,
        n_fills=n_fills,
        n_maker_fills=n_maker,
        equity_ticks=equity,
        timestamps_ns=times,
    )
