"""Tests for the exchange simulator and backtester."""

from pymatch import (
    Exchange, MarketConfig, MarketMaker, Momentum, Order, OrderBook,
    Side, TimeInForce, generate, run_backtest,
)
from pymatch.order import OrderType


def test_exchange_latency():
    ex = Exchange(latency_ns=100)
    ex.submit(Order(side=Side.BID, quantity=10, price=99, timestamp_ns=0))
    ex.advance_to(50)
    assert ex.book.order_count() == 0     # not arrived yet
    ex.advance_to(100)
    assert ex.book.order_count() == 1     # arrived
    ex.book.check_invariants()


def test_exchange_time_monotonic():
    ex = Exchange()
    ex.advance_to(10)
    try:
        ex.advance_to(5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_generate_keeps_book_sane():
    ex = Exchange()
    generate(ex, MarketConfig(steps=2000, seed=1))
    ex.book.check_invariants()
    assert ex.book.best_bid is not None and ex.book.best_ask is not None
    assert ex.book.spread > 0


def test_backtest_deterministic():
    cfg = MarketConfig(steps=1500, seed=123)
    r1 = run_backtest(MarketMaker, cfg)
    r2 = run_backtest(MarketMaker, cfg)
    assert r1.final_cash_ticks == r2.final_cash_ticks
    assert r1.n_fills == r2.n_fills


def test_backtest_market_maker_trades():
    cfg = MarketConfig(steps=3000, seed=7)
    r = run_backtest(MarketMaker, cfg, size=50)
    assert r.n_fills > 0
    assert r.n_maker_fills > 0          # got hit passively at least sometimes
    assert r.final_inventory == 0       # liquidated at end
    assert len(r.equity_ticks) > 10


def test_backtest_momentum_trades():
    cfg = MarketConfig(steps=3000, seed=7)
    r = run_backtest(Momentum, cfg, size=25)
    assert r.n_fills > 0
    assert r.final_inventory == 0


def test_backtest_fees_increase_with_rate():
    cfg = MarketConfig(steps=2000, seed=7)
    cheap = run_backtest(MarketMaker, cfg, taker_fee_bps=0.1, maker_fee_bps=0.1)
    pricey = run_backtest(MarketMaker, cfg, taker_fee_bps=5.0, maker_fee_bps=5.0)
    assert pricey.total_fees_ticks > cheap.total_fees_ticks
    assert pricey.final_cash_ticks < cheap.final_cash_ticks
