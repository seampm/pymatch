"""pymatch: a limit order book matching engine, exchange simulator, and backtester."""

from .backtest import BacktestResult, run_backtest
from .exchange import Exchange
from .market import MarketConfig, generate
from .order import Order, OrderType, Side, TimeInForce
from .orderbook import OrderBook
from .strategy import MarketMaker, Momentum, Strategy

__all__ = [
    "BacktestResult", "Exchange", "MarketConfig", "MarketMaker", "Momentum",
    "Order", "OrderBook", "OrderType", "Side", "Strategy", "TimeInForce",
    "generate", "run_backtest",
]

__version__ = "1.0.0"
