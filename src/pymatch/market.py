"""Synthetic market data generator.

Produces a stream of limit orders / cancels that keeps a plausible book
alive around a geometric-Brownian-motion midprice. This is *not* real market
data and the README says so plainly; it exists so strategies can be tested
without a data license.

Microstructure, roughly:
* Background limit orders arrive as a Poisson process, clustered near the
  touch, sizes lognormal.
* Background cancels arrive as a Poisson process on resting orders.
* Every N events the midprice takes a GBM step and the book is re-centered
  by cancelling stale far-touch orders (keeps the simulation stationary).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .exchange import Exchange
from .order import Order, Side, TimeInForce


@dataclass
class MarketConfig:
    start_price: int = 100_00      # ticks; e.g. cents -> $100.00
    tick_size: int = 1
    tick_vol: float = 0.00005      # per-step midprice volatility (~1 tick/step)
    steps: int = 20_000
    seed: int = 7
    # Poisson rates per step
    add_rate: float = 6.0
    cancel_rate: float = 2.0
    trade_rate: float = 1.2        # aggressive market orders (background flow)
    levels: int = 8                # how deep background quotes go
    avg_size: int = 400
    recenter_every: int = 50


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's algorithm; fine for small lambda.
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


@dataclass
class _SimState:
    mid: float
    live: list[int]          # background resting order ids (for cancels)


def new_state(cfg: MarketConfig) -> tuple[random.Random, _SimState]:
    return random.Random(cfg.seed), _SimState(float(cfg.start_price), [])


def step(exchange: Exchange, cfg: MarketConfig, rng: random.Random,
         state: _SimState, t_ns: int, step_idx: int) -> list[Event]:
    """Advance the background market one step. Shared by generate() and
    the backtester so both see identical market dynamics.

    Returns the engine events produced while advancing (fills, accepts,
    cancels) so the caller can attribute them."""
    from .order import Event, OrderType
    events: list[Event] = []
    state.mid *= math.exp(rng.gauss(0.0, cfg.tick_vol))
    if step_idx % cfg.recenter_every == 0:
        for oid in state.live:
            exchange.cancel(oid)
        state.live = []
        events.extend(exchange.advance_to(t_ns))

    touch = int(round(state.mid))
    for _ in range(_poisson(rng, cfg.add_rate)):
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        level = int(rng.expovariate(1.0 / 2.0)) % cfg.levels
        price = touch - level - 1 if side is Side.BID else touch + level + 1
        size = max(1, int(rng.lognormvariate(math.log(cfg.avg_size), 0.8)))
        o = Order(side=side, quantity=size, price=price,
                  tif=TimeInForce.GTC, timestamp_ns=t_ns)
        state.live.append(o.order_id)
        exchange.submit(o)
    for _ in range(_poisson(rng, cfg.cancel_rate)):
        if state.live:
            exchange.cancel(state.live.pop(rng.randrange(len(state.live))))
    for _ in range(_poisson(rng, cfg.trade_rate)):
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        size = max(1, int(rng.lognormvariate(math.log(cfg.avg_size // 4), 0.8)))
        if rng.random() < 0.08:
            size *= 8        # institutional sweep: walks the book
        o = Order(side=side, quantity=size, order_type=OrderType.MARKET,
                  tif=TimeInForce.IOC, timestamp_ns=t_ns)
        exchange.submit(o)
    events.extend(exchange.advance_to(t_ns))
    return events


def generate(exchange: Exchange, cfg: MarketConfig) -> None:
    """Drive `exchange` with synthetic background flow. Mutates in place."""
    rng, state = new_state(cfg)
    for step_idx in range(cfg.steps):
        step(exchange, cfg, rng, state, step_idx * 1_000_000, step_idx)
