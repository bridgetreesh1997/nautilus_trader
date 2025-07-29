#!/usr/bin/env python3
"""Example backtest for the TopBottomFinder strategy on ETHUSDT trade ticks."""

import time
from decimal import Decimal

import pandas as pd

from nautilus_trader.adapters.binance import BINANCE_VENUE
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.examples.strategies.top_bottom_finder import TopBottomFinder
from nautilus_trader.examples.strategies.top_bottom_finder import TopBottomFinderConfig
from nautilus_trader.model.currencies import ETH
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import TradeTickDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider
from nautilus_trader.test_kit.providers import TestInstrumentProvider


if __name__ == "__main__":
    config = BacktestEngineConfig(trader_id=TraderId("BACKTESTER-001"))
    engine = BacktestEngine(config=config)

    engine.add_venue(
        venue=BINANCE_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=None,
        starting_balances=[Money(1_000_000, USDT), Money(10, ETH)],
    )

    ETHUSDT_BINANCE = TestInstrumentProvider.ethusdt_binance()
    engine.add_instrument(ETHUSDT_BINANCE)

    provider = TestDataProvider()
    wrangler = TradeTickDataWrangler(instrument=ETHUSDT_BINANCE)
    ticks = wrangler.process(provider.read_csv_ticks("binance/ethusdt-trades.csv"))
    engine.add_data(ticks)

    strat_config = TopBottomFinderConfig(
        instrument_id=ETHUSDT_BINANCE.id,
        bar_type=BarType.from_str("ETHUSDT.BINANCE-100-TICK-LAST-INTERNAL"),
        trade_size=Decimal("0.10"),
        signal_strength=10,
    )
    strategy = TopBottomFinder(config=strat_config)
    engine.add_strategy(strategy)

    time.sleep(0.1)
    input("Press Enter to continue...")

    engine.run()

    with pd.option_context(
        "display.max_rows", 100, "display.max_columns", None, "display.width", 300
    ):
        print(engine.trader.generate_account_report(BINANCE_VENUE))
        print(engine.trader.generate_order_fills_report())
        print(engine.trader.generate_positions_report())

    engine.reset()
    engine.dispose()
