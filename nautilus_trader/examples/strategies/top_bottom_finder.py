#!/usr/bin/env python3
"""
Top and Bottom Finder strategy adapted from a TradingView script.
This version is simplified for example purposes and uses the Nautilus Trader API.
"""

from collections import deque
from decimal import Decimal

import pandas as pd

from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


class _ApproxFilter:
    """Internal recursive filter used for smoothing."""

    def __init__(self, coeff: float) -> None:
        self.b = coeff
        self.l0 = 0.0
        self.l1 = 0.0
        self.l2 = 0.0
        self.l3 = 0.0

    def update(self, value: float) -> float:
        l0_prev, l1_prev, l2_prev, l3_prev = self.l0, self.l1, self.l2, self.l3
        self.l0 = (1 - self.b) * value + self.b * l0_prev
        self.l1 = -self.b * self.l0 + l0_prev + self.b * l1_prev
        self.l2 = -self.b * self.l1 + l1_prev + self.b * l2_prev
        self.l3 = -self.b * self.l2 + l2_prev + self.b * l3_prev
        return (self.l0 + 2 * self.l1 + 2 * self.l2 + self.l3) / 6


class TopBottomFinderConfig(StrategyConfig, frozen=True):
    """Configuration for :class:`TopBottomFinder`."""

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    signal_strength: PositiveInt = 10


class TopBottomFinder(Strategy):
    """Simplified implementation of the Top & Bottom Finder strategy."""

    def __init__(self, config: TopBottomFinderConfig) -> None:
        PyCondition.is_true(config.signal_strength > 0, "signal_strength must be > 0")
        super().__init__(config)

        coeffs = [
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ]
        self.open_filters: list[_ApproxFilter] = [_ApproxFilter(c) for c in coeffs]
        self.tr_filters: list[_ApproxFilter] = [_ApproxFilter(c) for c in coeffs]

        self.instrument: Instrument | None = None
        self.rising = 0
        self.falling = 0
        self.close_history: deque[float] = deque(maxlen=5)
        self.high_history: deque[float] = deque(maxlen=config.signal_strength)
        self.low_history: deque[float] = deque(maxlen=config.signal_strength)

        self.prev_upper = None
        self.prev_lower = None
        self.prev_high = None
        self.prev_low = None

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return
        self.subscribe_bars(self.config.bar_type)
        self.request_bars(self.config.bar_type, start=self._clock.utc_now() - pd.Timedelta(days=1))

    def on_bar(self, bar: Bar) -> None:  # noqa: C901
        amlag = sum(f.update(bar.open) for f in self.open_filters) / len(self.open_filters)
        tr = bar.high - bar.low
        inapp = sum(f.update(tr) for f in self.tr_filters) / len(self.tr_filters)

        mult = 1.618
        upper = amlag + 2 * inapp * mult
        lower = amlag - 2 * inapp * mult

        crossdn = False
        crossup = False
        if self.prev_upper is not None and self.prev_high is not None:
            crossdn = bar.high < self.prev_upper and self.prev_high >= self.prev_upper
        if self.prev_lower is not None and self.prev_low is not None:
            crossup = bar.low > self.prev_lower and self.prev_low <= self.prev_lower

        self.close_history.append(bar.close)
        self.high_history.append(bar.high)
        self.low_history.append(bar.low)

        qual_ret = 0
        if len(self.close_history) == 5:
            close_4 = self.close_history[0]
            if bar.close > close_4:
                self.rising += 1
            elif bar.close < close_4:
                self.falling += 1

        if self.rising > 2 and bar.close < bar.open and bar.high >= max(self.high_history):
            self.rising = 0
            qual_ret = -1
        if self.falling > 2 and bar.close > bar.open and bar.low <= min(self.low_history):
            self.falling = 0
            qual_ret = 1

        long_signal = crossup or qual_ret == 1
        short_signal = crossdn or qual_ret == -1

        if long_signal:
            if self.portfolio.is_net_short(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
            if self.portfolio.is_flat(self.config.instrument_id):
                self.buy()

        if short_signal:
            if self.portfolio.is_net_long(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
            if self.portfolio.is_flat(self.config.instrument_id):
                self.sell()

        self.prev_upper = upper
        self.prev_lower = lower
        self.prev_high = bar.high
        self.prev_low = bar.low

    def buy(self) -> None:
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.create_order_qty(),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def sell(self) -> None:
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.create_order_qty(),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def create_order_qty(self) -> Quantity:
        return self.instrument.make_qty(self.config.trade_size)

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)
